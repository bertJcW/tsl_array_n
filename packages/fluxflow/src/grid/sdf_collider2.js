// 移植自 sdf_collider2.py。工厂函数风格，跟这次移植的其它模块一致。
//
// addShapelyGeometry/addSvg 的替代实现（不用 shapely/svg.path，零新依赖）见文件头部
// polygon_sdf.js / svg_utils.js 的注释——多边形几何用手写的 point-in-polygon +
// 点到边界距离，多个形状合并用 SDF pointwise min，SVG 解析用浏览器原生
// SVGPathElement API。这里 addPolygon()/addPolygons()/addSvg() 是新起的名字
// （不叫 addShapelyGeometry，因为压根不涉及 shapely 或任何"geometry 对象"，
// 就是普通的顶点数组）。

import { vec2 } from 'three/tsl';
import { createCellCenteredScalarGrid2 } from './grid_data2.js';
import { collocatedValueAtPosition2, bilinearGradientAtPosition2 } from './grid_math.js';
import { polygonsSignedDistance, polygonCentroid, translatePolygon, rotatePolygon } from './polygon_sdf.js';
import { parseSvgToPolygons } from './svg_utils.js';

const DEFAULT_FRICTION = 0.5;

export function createSDFStaticCollider2( resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY ) {

	const grid = createCellCenteredScalarGrid2( resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY );

	// cell-centered 网格的 CPU 侧原点——跟 grid.dataOrigin（TSL 节点）数值上必须一致，
	// 但纯 JS 场景（addPolygons 的逐格 CPU 循环）不方便读节点的值，独立算一份
	const originXCpu = originX + 0.5 * gridSpacingX;
	const originYCpu = originY + 0.5 * gridSpacingY;

	// 在 kernel 里用 grid_math 的通用双线性采样，直接复用，不重复写插值逻辑
	function sample( pos ) {

		return collocatedValueAtPosition2( grid.data, grid.gridSpacing, grid.dataOrigin, pos, grid.resolution );

	}

	// SDF 在任意连续位置的梯度，给 no-flux 投影求法向量用；用 bilinearGradientAtPosition2
	// （对双线性插值公式解析求导，精确对应 sample() 的返回值），不是会额外做一次跨格
	// 混合的 scalarGradientAtPosition2
	function gradient( pos ) {

		return bilinearGradientAtPosition2( grid.data, grid.gridSpacing, grid.dataOrigin, pos, grid.resolution );

	}

	// 格子索引版本的 inside 判断，给 marker 用（sdf<0 视为在 collider 内部）
	function isInside( i, j ) {

		return grid.data( i, j ).lessThan( 0 );

	}

	// static collider 速度恒为 0
	function velocityAt( /* point */ ) {

		return vec2( 0 );

	}

	// 把若干多边形（[[x,y],...] 顶点数组）栅格化成 SDF、写进 grid.data——对应源码
	// addShapelyGeometry 里 CPU 端逐格算距离、算完整批一次性 from_numpy 上传的做法
	function addPolygons( polygons ) {

		const [ nx, ny ] = grid.resolution;
		const hostSdf = new Float32Array( nx * ny );

		for ( let j = 0; j < ny; j ++ ) {

			const y = originYCpu + j * gridSpacingY;

			for ( let i = 0; i < nx; i ++ ) {

				const x = originXCpu + i * gridSpacingX;
				hostSdf[ i + j * nx ] = polygonsSignedDistance( x, y, polygons );

			}

		}

		grid.data.fromArray( hostSdf );

	}

	function addPolygon( points ) {

		addPolygons( [ points ] );

	}

	function addSvg( svgString, options ) {

		addPolygons( parseSvgToPolygons( svgString, options ) );

	}

	return {
		grid,
		frictionCoefficient: DEFAULT_FRICTION, // 普通可变属性，collider.frictionCoefficient = x 直接改
		clear: grid.clear,
		sample, gradient, isInside, velocityAt,
		addPolygon, addPolygons, addSvg
	};

}

// 运动的刚体 collider：跟 SDFStaticCollider2 共用网格/sample/gradient/isInside，
// 只是 velocityAt 按刚体运动学算，并提供 update(dt)——只有真的在动（线速度或角速度
// 不为 0）才重新摆放几何、重新栅格化 SDF。
//
// geometryPolygon：初始几何形状，[[x,y],...] 顶点数组（对应源码的 shapely geometry
// 参数）。linearVelocityXY：[vx,vy] 普通数组，不是 TSL 节点——运动学积分
// （currentPosition/currentAngle 的更新）是纯 CPU 侧逐帧累加，用节点没有意义。
//
// 已知的架构性限制（不是这次移植引入的新问题，是 Taichi/TSL 两边"图构建一次、
// 之后重复 dispatch"这个执行模型共有的边界）：如果未来某个 kernel 在构建时
// （比如 grid_blocked_boundary_condition_solver2.js 的 _markAndProjectU 之类）
// 调用了这里的 velocityAt(point)，构建出来的 TSL 节点图会把当时 currentPosition/
// currentAngle/linearVelocity 的值烤成图里的常量——之后 update(dt) 改了这些值，
// 已经构建好的 kernel 不会自动感知到。如果确实需要"每帧真正变化、又不想重新构建
// kernel"的碰撞体运动，应该把 position/velocity 换成 tsl_array_n 的 array0/uniform
// （用 .fromArray()/.value= 更新），而不是像现在这样直接捕获 JS 闭包变量——现在
// 这个实现是对源码 Python/Taichi 两边同样存在的这个限制的忠实移植，不是新增的缺陷，
// 但如果后面要做真正逐帧移动的 collider，这是需要重新设计的点。
export function createSDFRigidBodyCollider2(
	geometryPolygon,
	resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY,
	linearVelocityXY = [ 0, 0 ], angularVelocity = 0
) {

	const collider = createSDFStaticCollider2( resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY );

	const baseGeometry = geometryPolygon;
	const linearVelocityNode = vec2( linearVelocityXY[ 0 ], linearVelocityXY[ 1 ] );

	let currentPosition = polygonCentroid( geometryPolygon );
	let currentAngle = 0;

	collider.addPolygon( geometryPolygon );

	function update( dt ) {

		// tm.vec2 的 == 是逐分量比较，这一点 Taichi 里恒为 true 没法直接当整体
		// 相等判断——这里就是普通 JS 数字比较，没有这个坑，但保留同样的"静止就跳过"
		// 优化，避免每帧重新栅格化一个没有移动的碰撞体
		const isStationary = linearVelocityXY[ 0 ] === 0 && linearVelocityXY[ 1 ] === 0 && angularVelocity === 0;
		if ( isStationary ) return;

		currentPosition = [
			currentPosition[ 0 ] + linearVelocityXY[ 0 ] * dt,
			currentPosition[ 1 ] + linearVelocityXY[ 1 ] * dt
		];
		currentAngle += angularVelocity * dt;

		const centroid = polygonCentroid( baseGeometry );
		const translated = translatePolygon( baseGeometry, currentPosition[ 0 ] - centroid[ 0 ], currentPosition[ 1 ] - centroid[ 1 ] );
		const posed = rotatePolygon( translated, currentAngle, currentPosition );

		collider.addPolygon( posed );

	}

	// 刚体运动学：v(point) = linearVelocity + angularVelocity x (point - currentPosition)
	// 2D 里叉乘 angularVelocity x r 就是 angularVelocity * (-r.y, r.x)
	function velocityAt( point ) {

		const r = point.sub( vec2( currentPosition[ 0 ], currentPosition[ 1 ] ) );
		return linearVelocityNode.add( vec2( r.y.negate(), r.x ).mul( angularVelocity ) );

	}

	return {
		...collider,
		baseGeometry,
		linearVelocity: linearVelocityNode,
		angularVelocity,
		update,
		velocityAt
	};

}
