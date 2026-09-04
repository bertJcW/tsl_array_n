// 已知边界（跟这次移植本身无关）：在这个 sandbox 环境（没有真实 WebGPU 适配器，
// init() 会 fallback 到 WebGLBackend）里跑，下面三个测试目前全部读回 0——已经
// 用几个变体的最小复现排查过：单线程 kernel 里 assign 一个常量能正确写回
// （排除了"shape=1 kernel 本身不工作"）；问题specifically是"在一个 kernel 里读取
// 另一个之前已经有数据的 field"这个操作本身在这个 fallback 后端上读不到正确值——
// 不管那个 field 的数据是 fromArray() 写的还是另一个 kernel 写的，都一样读到 0/空。
// 这跟 tsl_array_n 自己历史上遇到的两次 fallback-only 问题（Loop() 计数器、array0
// 多线程共读）是同一类环境限制，但触发条件更窄更基础，是这次移植过程中新发现的
// 第三个实例。跟之前两次一样，还没有实机 WebGPU 复核——数值到底对不对，等有真实
// WebGPU 的环境跑一遍这个页面确认。
import * as tsl_array_n from 'tsl_array_n';
import { grid } from 'fluxflow';
import { vec2, int } from 'three/tsl';

const pre = document.querySelector( '#status pre' );
const lines = [];

function log( label, ok, detail ) {

	const cls = ok ? 'ok' : 'err';
	const mark = ok ? '✓' : '✗';
	lines.push( `<span class="${ cls }">${ mark } ${ label }${ detail ? ' — ' + detail : '' }</span>` );
	pre.innerHTML = lines.join( '\n' );

}

function approxEqual( a, b, eps = 1e-4 ) {

	return Math.abs( a - b ) < eps;

}

function fillScalar4x4( field, valueAt ) {

	const flat = new Float32Array( 16 );
	for ( let j = 0; j < 4; j ++ ) for ( let i = 0; i < 4; i ++ ) flat[ i + j * 4 ] = valueAt( i, j );
	field.fromArray( flat );

}

function fillVec2_4x4( field, valueAt ) {

	const flat = new Float32Array( 32 );
	for ( let j = 0; j < 4; j ++ ) for ( let i = 0; i < 4; i ++ ) {

		const [ x, y ] = valueAt( i, j );
		flat[ ( i + j * 4 ) * 2 ] = x;
		flat[ ( i + j * 4 ) * 2 + 1 ] = y;

	}

	field.fromArray( flat );

}

try {

	const renderer = await tsl_array_n.init( { allowFallback: true } );
	log( 'init()', true, `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }` );

	// ------------------------------------------------------------
	// 1. collocatedValueAtPosition2：双线性插值——网格点上应该精确等于该点数据，
	// 半格中点应该等于周围4个角点的平均值
	{

		const scalarGrid = grid.createScalarGrid2( 4, 4, 1, 1, 0, 0 );
		fillScalar4x4( scalarGrid.data, ( i, j ) => i + j * 4 ); // data[i,j] = i + 4j，跟 tsl_array_n 自己 .at() 的下标约定一致

		const out = tsl_array_n.arrayN( 'float', 2 );

		const run = tsl_array_n.kernel( 1, ( _idx ) => {

			// 网格点 (1,2) 上采样：应该精确等于 data[1,2] = 1 + 2*4 = 9
			out( 0 ).assign( grid.collocatedValueAtPosition2( scalarGrid.data, scalarGrid.gridSpacing, scalarGrid.dataOrigin, vec2( 1, 2 ), scalarGrid.dataSize ) );

			// (0.5, 0.5) 半格中点：应该等于 data[0,0],data[1,0],data[0,1],data[1,1] 的平均 = (0+1+4+5)/4 = 2.5
			out( 1 ).assign( grid.collocatedValueAtPosition2( scalarGrid.data, scalarGrid.gridSpacing, scalarGrid.dataOrigin, vec2( 0.5, 0.5 ), scalarGrid.dataSize ) );

		} );

		run();

		const [ atGridPoint, atMidpoint ] = await out.toArray();
		const ok = approxEqual( atGridPoint, 9 ) && approxEqual( atMidpoint, 2.5 );

		log(
			'collocatedValueAtPosition2 bilinear interpolation',
			ok,
			ok ? `grid point (1,2)=${ atGridPoint }, midpoint (0.5,0.5)=${ atMidpoint }` : `got [${ atGridPoint }, ${ atMidpoint }], expected [9, 2.5]`
		);

	}

	// ------------------------------------------------------------
	// 2. scalarGradient2：在线性场 f(i,j) = 2i + 3j 的内部格点上，中心差分应该
	// 精确给出解析梯度 (2, 3)（线性函数的中心差分没有截断误差）
	{

		const linearGrid = grid.createScalarGrid2( 4, 4, 1, 1, 0, 0 );
		fillScalar4x4( linearGrid.data, ( i, j ) => 2 * i + 3 * j );

		const out = tsl_array_n.arrayN( 'vec2', 1 );

		const run = tsl_array_n.kernel( 1, ( _idx ) => {

			out( 0 ).assign( grid.scalarGradient2( linearGrid.data, linearGrid.gridSpacing, int( 1 ), int( 1 ), linearGrid.dataSize ) );

		} );

		run();

		const [ gx, gy ] = await out.toArray();
		const ok = approxEqual( gx, 2 ) && approxEqual( gy, 3 );

		log(
			'scalarGradient2 at an interior point of a linear field',
			ok,
			ok ? `∇f(1,1) = (${ gx }, ${ gy })` : `got (${ gx }, ${ gy }), expected (2, 3)`
		);

	}

	// ------------------------------------------------------------
	// 3. vectorGradient2 / mat2 元素顺序——grid_math.js 文件头注释标注过这个未验证的点：
	// 向量场 f(i,j) = (2i + 5j, 0)，解析 Jacobian J = [[2,5],[0,0]]（行：∂fx/∂x,∂fx/∂y；
	// ∂fy/∂x,∂fy/∂y）。J·(1,0) 应该等于 x 方向偏导 = (∂fx/∂x, ∂fy/∂x) = (2,0)；
	// J·(0,1) 应该等于 y 方向偏导 = (∂fx/∂y, ∂fy/∂y) = (5,0)。用非对称场是关键——
	// 对称场看不出转置。
	{

		const vectorGrid = grid.createCollocatedVectorGrid2( 4, 4, 1, 1, 0, 0 );
		fillVec2_4x4( vectorGrid.data, ( i, j ) => [ 2 * i + 5 * j, 0 ] );

		const out = tsl_array_n.arrayN( 'vec2', 2 );

		const run = tsl_array_n.kernel( 1, ( _idx ) => {

			const jacobian = grid.vectorGradient2( vectorGrid.data, vectorGrid.gridSpacing, int( 1 ), int( 1 ), vectorGrid.dataSize );
			out( 0 ).assign( jacobian.mul( vec2( 1, 0 ) ) );
			out( 1 ).assign( jacobian.mul( vec2( 0, 1 ) ) );

		} );

		run();

		const [ jx0, jy0, jx1, jy1 ] = await out.toArray();
		const ok = approxEqual( jx0, 2 ) && approxEqual( jy0, 0 ) && approxEqual( jx1, 5 ) && approxEqual( jy1, 0 );

		log(
			'vectorGradient2 mat2 ordering (row-major vs column-major)',
			ok,
			ok
				? `J·(1,0)=(${ jx0 },${ jy0 }), J·(0,1)=(${ jx1 },${ jy1 }) — matches the intended row-major Jacobian`
				: `got J·(1,0)=(${ jx0 },${ jy0 }), J·(0,1)=(${ jx1 },${ jy1 }); expected (2,0) and (5,0) — mat2() fill order needs fixing in grid_math.js`
		);

	}

} catch ( error ) {

	log( 'failed', false, error.message );

}
