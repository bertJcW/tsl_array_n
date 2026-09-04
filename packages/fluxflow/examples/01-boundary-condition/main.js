// 冒烟测试，不是数值正确性验证：这个 sandbox 环境（见 00-grid-math/main.js 头部
// 注释）目前连"kernel 里读另一个 field"这种基础操作都读不到正确值，所以就算这里
// constrainVelocity() 跑完之后读出来的数字看起来"合理"也不能真的当数值验证——
// 这里只确认整条链路（collider 栅格化 → 边界求解器构造 → constrainVelocity 真实
// 派发一整套 kernel）能不能跑通、不抛错，这本身就有价值（能抓到 API 用错、参数
// 个数不对、方法名拼错这类问题）。真正的数值正确性要等真实 WebGPU 环境。

import * as tsl_array_n from 'tsl_array_n';
import { grid } from 'fluxflow';

const pre = document.querySelector( '#status pre' );
const lines = [];

function log( label, ok, detail ) {

	const cls = ok ? 'ok' : 'err';
	const mark = ok ? '✓' : '✗';
	lines.push( `<span class="${ cls }">${ mark } ${ label }${ detail ? ' — ' + detail : '' }</span>` );
	pre.innerHTML = lines.join( '\n' );

}

try {

	const renderer = await tsl_array_n.init( { allowFallback: true } );
	log( 'init()', true, `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }` );

	const nx = 8, ny = 8;
	const velocity = grid.createFaceCenteredGrid2( nx, ny, 1, 1, -4, -4 );

	// 一个小方块 collider，摆在 domain 中间偏左，边长2
	const colliderSquare = [ [ -3, -1 ], [ -1, -1 ], [ -1, 1 ], [ -3, 1 ] ];
	const collider = grid.createSDFStaticCollider2( nx, ny, 1, 1, -4, -4 );

	try {

		collider.addPolygon( colliderSquare );
		log( 'SDFStaticCollider2.addPolygon()', true );

	} catch ( error ) {

		log( 'SDFStaticCollider2.addPolygon()', false, error.message );
		throw error;

	}

	let solver;

	try {

		// 构造函数内部会立刻 dispatch buildBlockMarker()（因为传了真实 collider）——
		// 这是 vitest 测不到的部分，真正的 live 验证从这里开始
		solver = grid.createGridBlockedBoundaryConditionSolver2( velocity, nx, ny, 1, 1, -4, -4, collider );
		log( 'createGridBlockedBoundaryConditionSolver2() with a real collider', true, 'buildBlockMarker() dispatched during construction without throwing' );

	} catch ( error ) {

		log( 'createGridBlockedBoundaryConditionSolver2() with a real collider', false, error.message );
		throw error;

	}

	// 给 velocity 场塞一点初始值（向右的均匀流），让 constrainVelocity() 的
	// no-flux/blocked-boundary 分支真的有非零输入可处理，不是全零场景
	velocity.dataU.fromArray( new Float32Array( velocity.dataSizeU[ 0 ] * velocity.dataSizeU[ 1 ] ).fill( 1 ) );
	velocity.dataV.fromArray( new Float32Array( velocity.dataSizeV[ 0 ] * velocity.dataSizeV[ 1 ] ).fill( 0 ) );

	try {

		solver.constrainVelocity();
		log( 'constrainVelocity() — full dispatch (fill markers, mark+project, extrapolate, no-flux, blocked boundary, domain walls)', true, 'ran without throwing' );

	} catch ( error ) {

		log( 'constrainVelocity()', false, error.message );
		throw error;

	}

	try {

		const uData = await velocity.dataU.toArray();
		const vData = await velocity.dataV.toArray();
		log(
			'readback (NOT a correctness check, see file header)',
			true,
			`dataU sample: [${ Array.from( uData ).slice( 0, 4 ) }...], dataV sample: [${ Array.from( vData ).slice( 0, 4 ) }...]`
		);

	} catch ( error ) {

		log( 'readback', false, error.message );

	}

	// 顺带验证一下 setCollider(null, ...) 能不能把 collider 摘掉、退回"只做 domain
	// boundary"的路径，不抛错
	try {

		solver.setCollider( null, [ nx, ny ], [ 1, 1 ], [ -4, -4 ] );
		solver.constrainVelocity();
		log( 'setCollider(null, ...) then constrainVelocity() again', true, 'falls back to domain-boundary-only path without throwing' );

	} catch ( error ) {

		log( 'setCollider(null, ...) then constrainVelocity() again', false, error.message );

	}

} catch ( error ) {

	log( 'failed', false, error.message );

}
