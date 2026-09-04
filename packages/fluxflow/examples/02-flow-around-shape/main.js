// "现有 grid 能做什么视觉效果" 的一个测试，不是一个正式 example/教程。
//
// 这不是真正的流体解算（没有压力投影/对流求解器，grid_solver2.js 只是个空壳）——
// 这里做的是两件更小的事，都是 grid/ 现有模块真能做到的：
//   1. 用 createSDFRigidBodyCollider2（真实的 fluxflow collider 对象）驱动一个
//      旋转的星形碰撞体，SDF 渲染成热力图——直接体现 polygon_sdf.js 的多边形
//      SDF 光栅化能力。
//   2. 一个轻量的、纯 CPU 的绕流粒子效果：用 polygon_sdf.js 导出的
//      polygonSignedDistance 数值估计梯度方向，让粒子靠近形状时被推开——
//      经典的"势流"风格视觉技巧，不是真正解 Navier-Stokes，但看起来像流体绕过
//      障碍物。故意选纯 CPU：几百个粒子这个规模用不上 GPU 并行，硬套 kernel
//      反而复杂化。
//
// 形状的旋转姿态在这里单独算了一份（调用跟 collider 内部一样的 rotatePolygon/
// translatePolygon/polygonCentroid，来自 polygon_sdf.js），没有直接读 collider
// 内部状态——sdf_collider2.js 没有把摆好姿态的多边形单独暴露出来，为了画布渲染
// 用的是每像素/每格子的 CPU 数值，不需要也不适合走"读 GPU field 再 toArray()"
// 这条路（分辨率对不上，而且见下面的 GPU 往返检查，这条路在当前 sandbox 环境
// 本来就不可靠）。同时也真的调用 collider.update(dt)，证明这个真实对象的旋转
// 运动学本身能跑（跟 sdf_collider2.test.js 里的结构性测试不同，这里是活的、
// 连续多帧调用）。

import * as tsl_array_n from 'tsl_array_n';
import { grid } from 'fluxflow';

const canvas = document.getElementById( 'canvas' );
const ctx = canvas.getContext( '2d' );
const statusEl = document.getElementById( 'status' );

function setStatus( html ) {

	statusEl.innerHTML = html;

}

// ------------------------------------------------------------
// 世界坐标 <-> 画布像素坐标：世界是 [-5,5]x[-5,5]，画布 480x480，y 轴翻转
// （世界 y 朝上，画布 y 朝下）

const WORLD_HALF = 5;
const SCALE = canvas.width / ( WORLD_HALF * 2 );

function worldToCanvas( x, y ) {

	return [ ( x + WORLD_HALF ) * SCALE, ( WORLD_HALF - y ) * SCALE ];

}

// ------------------------------------------------------------
// 五角星多边形（局部坐标，中心在原点）

function makeStarPolygon( outerRadius, innerRadius, points ) {

	const verts = [];

	for ( let i = 0; i < points * 2; i ++ ) {

		const r = i % 2 === 0 ? outerRadius : innerRadius;
		const angle = ( i / ( points * 2 ) ) * Math.PI * 2 - Math.PI / 2;
		verts.push( [ r * Math.cos( angle ), r * Math.sin( angle ) ] );

	}

	return verts;

}

const starLocal = makeStarPolygon( 1.6, 0.65, 5 );
const shapeCenter = [ 0, 0 ];
const angularVelocity = 0.6; // rad/s

try {

	const renderer = await tsl_array_n.init( { allowFallback: true, canvas: document.createElement( 'canvas' ) } );

	// 真实的 fluxflow collider——8x8 只是给这个 GPU 侧 field 一个形状，这个 demo
	// 里视觉渲染不读它（见文件头注释），但下面会用它做一次真实的 GPU 往返检查
	const collider = grid.createSDFRigidBodyCollider2(
		starLocal.map( ( [ x, y ] ) => [ x + shapeCenter[ 0 ], y + shapeCenter[ 1 ] ] ),
		8, 8, 1.25, 1.25, -5, -5,
		[ 0, 0 ], angularVelocity
	);

	let gpuStatusHtml = '';

	try {

		// 摸一下这个 field：先用一个 kernel 把它自己拷贝到自己（确保 GPU 侧 buffer
		// 真的被一次 compute pass 用过——storage buffer 是惰性创建的，见
		// tsl_array_n 的已知边界），再 toArray() 读回来看看
		const touch = tsl_array_n.kernel( collider.grid.dataSize, ( i, j ) => {

			collider.grid.data( i, j ).assign( collider.grid.data( i, j ) );

		} );
		touch();

		const readback = Array.from( await collider.grid.data.toArray() );
		const looksReal = readback.some( ( v ) => v !== 0 );

		gpuStatusHtml = looksReal
			? `<span class="ok">✓ GPU SDF field 往返读回了非零值（backend: ${ renderer.backend?.constructor?.name }）</span>`
			: `<span class="err">✗ GPU SDF field 往返读回全 0 —— 这个 sandbox 环境已知的 fallback 限制（见 README），下面的可视化不依赖这个结果，用的是 CPU 直接算</span>`;

	} catch ( error ) {

		gpuStatusHtml = `<span class="err">✗ GPU 往返检查抛错：${ error.message }</span>`;

	}

	setStatus( `<span class="ok">✓ init() + createSDFRigidBodyCollider2() 构造成功</span>\n${ gpuStatusHtml }\n运行中…` );

	// ------------------------------------------------------------
	// 粒子系统：纯 CPU，用 polygon_sdf.js 的 polygonSignedDistance 数值估计梯度，
	// 靠近形状时把粒子推离——不是真正解流体方程，是经典的"势流"风格视觉近似

	const PARTICLE_COUNT = 400;
	const FLOW_SPEED = 2.2;
	const INFLUENCE_RADIUS = 1.6;

const TRAIL_LENGTH = 10;

	function spawnParticle() {

		const x = - WORLD_HALF - Math.random() * 1.5;
		const y = ( Math.random() - 0.5 ) * WORLD_HALF * 2;

		return { x, y, trail: Array.from( { length: TRAIL_LENGTH }, () => [ x, y ] ) };

	}

	const particles = Array.from( { length: PARTICLE_COUNT }, () => {

		const p = spawnParticle();
		p.x = - WORLD_HALF + Math.random() * WORLD_HALF * 2; // 第一帧就撒满整个画面，不用等粒子从左边流进来
		p.trail = Array.from( { length: TRAIL_LENGTH }, () => [ p.x, p.y ] );
		return p;

	} );

	function sdfGradient( x, y, polygon, eps ) {

		const dx = ( grid.polygonSignedDistance( x + eps, y, polygon ) - grid.polygonSignedDistance( x - eps, y, polygon ) ) / ( 2 * eps );
		const dy = ( grid.polygonSignedDistance( x, y + eps, polygon ) - grid.polygonSignedDistance( x, y - eps, polygon ) ) / ( 2 * eps );
		const len = Math.hypot( dx, dy ) || 1;
		return [ dx / len, dy / len ];

	}

	// ------------------------------------------------------------
	// SDF 热力图：内部暖色、外部按距离渐变到冷色，边界(d≈0)附近亮线高亮

	const HEATMAP_CELLS = 80;
	const cellSize = canvas.width / HEATMAP_CELLS;

	function sdfColor( d ) {

		if ( Math.abs( d ) < 0.06 ) return '#fef08a'; // 边界轮廓线

		if ( d < 0 ) {

			const t = Math.min( 1, - d / 1.5 );
			const r = Math.round( 120 + 100 * t );
			const g = Math.round( 40 + 20 * t );
			return `rgb(${ r},${ g },40)`;

		}

		const t = Math.min( 1, d / 4 );
		const r = Math.round( 10 + 10 * t );
		const g = Math.round( 20 + 60 * ( 1 - t ) );
		const b = Math.round( 40 + 150 * ( 1 - t ) );
		return `rgb(${ r },${ g },${ b })`;

	}

	let currentAngle = 0;
	let lastTime = performance.now();

	function frame( now ) {

		const dt = Math.min( 0.05, ( now - lastTime ) / 1000 );
		lastTime = now;

		currentAngle += angularVelocity * dt;

		// 跟 collider 内部同一套数学（polygon_sdf.js 导出的 rotatePolygon/
		// translatePolygon），单独算一份用于逐像素渲染
		const posedStar = grid.rotatePolygon( starLocal, currentAngle, [ 0, 0 ] )
			.map( ( [ x, y ] ) => [ x + shapeCenter[ 0 ], y + shapeCenter[ 1 ] ] );

		// 也真的驱动一下真实 collider 对象，证明它自己的运动学在跑（不用于渲染）
		collider.update( dt );

		// ---- 画热力图 ----
		for ( let cy = 0; cy < HEATMAP_CELLS; cy ++ ) {

			const wy = WORLD_HALF - ( cy + 0.5 ) * ( WORLD_HALF * 2 / HEATMAP_CELLS );

			for ( let cx = 0; cx < HEATMAP_CELLS; cx ++ ) {

				const wx = - WORLD_HALF + ( cx + 0.5 ) * ( WORLD_HALF * 2 / HEATMAP_CELLS );
				const d = grid.polygonSignedDistance( wx, wy, posedStar );

				ctx.fillStyle = sdfColor( d );
				ctx.fillRect( cx * cellSize, cy * cellSize, cellSize + 0.5, cellSize + 0.5 );

			}

		}

		// ---- 画绕流粒子：每个粒子拖一条渐隐的短尾迹（TRAIL_LENGTH 个历史位置），
		// 比单帧一小段线段更能在静止截图里也看出"在流动" ----
		ctx.lineWidth = 1.5;

		for ( const p of particles ) {

			const d = grid.polygonSignedDistance( p.x, p.y, posedStar );

			if ( d < -0.05 || p.x > WORLD_HALF + 1.5 ) {

				const fresh = spawnParticle();
				p.x = fresh.x; p.y = fresh.y; p.trail = fresh.trail;
				continue;

			}

			let vx = FLOW_SPEED, vy = 0;

			if ( d < INFLUENCE_RADIUS ) {

				const [ gx, gy ] = sdfGradient( p.x, p.y, posedStar, 0.05 );
				const t = Math.max( 0, 1 - d / INFLUENCE_RADIUS );
				const strength = t * t;

				vx = FLOW_SPEED * ( 1 - strength ) + gx * strength * FLOW_SPEED;
				vy = gy * strength * FLOW_SPEED;

			}

			p.x += vx * dt;
			p.y += vy * dt;

			p.trail.push( [ p.x, p.y ] );
			p.trail.shift();

			for ( let i = 1; i < p.trail.length; i ++ ) {

				const [ x0, y0 ] = worldToCanvas( ...p.trail[ i - 1 ] );
				const [ x1, y1 ] = worldToCanvas( ...p.trail[ i ] );
				const alpha = ( i / p.trail.length ) * 0.6;

				ctx.strokeStyle = `rgba(210,235,255,${ alpha })`;
				ctx.beginPath();
				ctx.moveTo( x0, y0 );
				ctx.lineTo( x1, y1 );
				ctx.stroke();

			}

		}

		requestAnimationFrame( frame );

	}

	requestAnimationFrame( frame );

} catch ( error ) {

	setStatus( `<span class="err">✗ failed — ${ error.message }</span>` );
	console.error( error );

}
