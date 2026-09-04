// 移植自 grid_blocked_boundary_condition_solver2.py。
//
// 跟源码最大的结构差异：源码构造函数不吃 velocity（velocity 是 constrainVelocity(velocity,...)
// 每次调用时传入的 ti.template() 参数），这里改成构造时就传入固定的 velocity
// （FaceCenteredGrid2）——tsl_array_n 的 kernel(shape,fn) 在构建时就要闭包绑定
// 具体 field，没有"每次调用换绑参数"的能力（移植计划决策4，不是这次引入的新限制）。
// 所以这里的 constrainVelocity() 不再吃 velocity 参数，永远操作构造时绑定的那一个。
//
// collider 则不同——源码本身就支持 setCollider() 中途换 collider（包括从 None 换成
// 有、或者换成另一个），所以所有读 collider 的 kernel 不能在构造时一次性建好，
// 要放进 rebuildColliderKernels()，每次 setCollider() 都重新构建一遍（构建 kernel
// 图本身不需要 GPU/renderer，便宜，重新构建没问题）。不依赖 collider、只依赖
// velocity 的 kernel（四个 _zero* 系列）在构造时建一次就够。

import { vec2, float } from 'three/tsl';
import * as tsl_array_n from 'tsl_array_n';
import * as ls from './level_set_utils.js';
import { createCopyKernel2, createExtrapolateToRegion2 } from './array_utils.js';
import { DIRECTION_LEFT, DIRECTION_RIGHT, DIRECTION_DOWN, DIRECTION_UP, DIRECTION_ALL } from './constant.js';

// uMarker/vMarker 含义一样(1=fluid, 0=collider)，跟源码一样只在这个文件内部用，
// 不放进 constant.js
const K_FLUID = 1;
const K_COLLIDER = 0;

// velt = vel - normal * dot(vel,normal)；如果 velt 长度 > 0，按摩擦系数衰减切向分量。
// 这是"可能原地修改"的模式（不是纯值选择），用 .toVar() + If()，不是 select()。
function projectAndApplyFriction( vel, normal, frictionCoefficient ) {

	const velt = vel.sub( normal.mul( vel.dot( normal ) ) ).toVar();

	tsl_array_n.If( velt.length().greaterThan( 0 ), () => {

		const veln = vel.dot( normal ).negate().max( 0 );
		const scale = float( 1 ).sub( veln.mul( frictionCoefficient ).div( velt.length() ) ).max( 0 );

		velt.mulAssign( scale );

	} );

	return velt;

}

// velocity: 构造时绑定的 FaceCenteredGrid2（见文件头注释）。
// colliderSDF: 可选，SDFStaticCollider2/SDFRigidBodyCollider2，None 时 constrainVelocity
// 只做 domain boundary 那一段。
export function createGridBlockedBoundaryConditionSolver2(
	velocity,
	resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY,
	colliderSDF = null
) {

	const nx = resolutionX;
	const ny = resolutionY;
	const uSize = [ nx + 1, ny ];
	const vSize = [ nx, ny + 1 ];

	const uMarker = tsl_array_n.arrayN( 'int', uSize );
	const vMarker = tsl_array_n.arrayN( 'int', vSize );
	const uTemp = tsl_array_n.arrayN( 'float', uSize );
	const vTemp = tsl_array_n.arrayN( 'float', vSize );
	const blockMarker = tsl_array_n.arrayN( 'int', [ nx, ny ] );

	const solver = {
		uMarker, vMarker, uTemp, vTemp, blockMarker,
		closedDomainBoundaryFlag: DIRECTION_ALL, // 普通可变属性，跟源码 setClosedDomainBoundaryFlag 效果一样
		collider: null
	};

	// ---- 只依赖 velocity 的 kernel：跟 collider 无关，构造时建一次 ----

	const copyUTempToVelocity = createCopyKernel2( uTemp, velocity.dataU, uSize );
	const copyVTempToVelocity = createCopyKernel2( vTemp, velocity.dataV, vSize );

	const zeroULeft  = tsl_array_n.kernel( uSize[ 1 ], ( j ) => { velocity.dataU( 0, j ).assign( 0 ); } );
	const zeroURight = tsl_array_n.kernel( uSize[ 1 ], ( j ) => { velocity.dataU( uSize[ 0 ] - 1, j ).assign( 0 ); } );
	const zeroVDown  = tsl_array_n.kernel( vSize[ 0 ], ( i ) => { velocity.dataV( i, 0 ).assign( 0 ); } );
	const zeroVUp    = tsl_array_n.kernel( vSize[ 0 ], ( i ) => { velocity.dataV( i, vSize[ 1 ] - 1 ).assign( 0 ); } );

	function projectClosedDomainBoundary() {

		const flag = solver.closedDomainBoundaryFlag;

		if ( flag & DIRECTION_LEFT )  zeroULeft();
		if ( flag & DIRECTION_RIGHT ) zeroURight();
		if ( flag & DIRECTION_DOWN )  zeroVDown();
		if ( flag & DIRECTION_UP )    zeroVUp();

	}

	// ---- 依赖 collider 的 kernel：collider 换了要重新建（见文件头注释）----

	let fillUMarker = null, fillVMarker = null;
	let buildBlockMarker = null;
	let markAndProjectU = null, markAndProjectV = null;
	let noFluxProjectionU = null, noFluxProjectionV = null;
	let blockedBoundary = null;
	let extrapolateU = null, extrapolateV = null;

	function rebuildColliderKernels() {

		const collider = solver.collider;

		if ( ! collider ) {

			fillUMarker = fillVMarker = buildBlockMarker = null;
			markAndProjectU = markAndProjectV = null;
			noFluxProjectionU = noFluxProjectionV = null;
			blockedBoundary = null;
			extrapolateU = extrapolateV = null;
			return;

		}

		fillUMarker = tsl_array_n.kernel( uSize, ( i, j ) => { uMarker( i, j ).assign( 1 ); } );
		fillVMarker = tsl_array_n.kernel( vSize, ( i, j ) => { vMarker( i, j ).assign( 1 ); } );

		buildBlockMarker = tsl_array_n.kernel( [ nx, ny ], ( i, j ) => {

			blockMarker( i, j ).assign( collider.isInside( i, j ).select( K_COLLIDER, K_FLUID ) );

		} );

		// assign marker and collider's velocity if marker is 0
		markAndProjectU = tsl_array_n.kernel( uSize, ( i, j ) => {

			const pt = velocity.uPosition( i, j );
			const h = velocity.gridSpacing;

			const phi0 = collider.sample( pt.sub( vec2( h.x.mul( 0.5 ), 0 ) ) );
			const phi1 = collider.sample( pt.add( vec2( h.x.mul( 0.5 ), 0 ) ) );

			const frac = float( 1 ).sub( ls.fractionInsideSdf( phi0, phi1 ).clamp( 0, 1 ) );

			tsl_array_n.If( frac.greaterThan( 0 ), () => {

				uMarker( i, j ).assign( K_FLUID );

			} ).Else( () => {

				velocity.dataU( i, j ).assign( collider.velocityAt( pt ).x );
				uMarker( i, j ).assign( K_COLLIDER );

			} );

		} );

		markAndProjectV = tsl_array_n.kernel( vSize, ( i, j ) => {

			const pt = velocity.vPosition( i, j );
			const h = velocity.gridSpacing;

			const phi0 = collider.sample( pt.sub( vec2( 0, h.y.mul( 0.5 ) ) ) );
			const phi1 = collider.sample( pt.add( vec2( 0, h.y.mul( 0.5 ) ) ) );

			const frac = float( 1 ).sub( ls.fractionInsideSdf( phi0, phi1 ).clamp( 0, 1 ) );

			tsl_array_n.If( frac.greaterThan( 0 ), () => {

				vMarker( i, j ).assign( K_FLUID );

			} ).Else( () => {

				velocity.dataV( i, j ).assign( collider.velocityAt( pt ).y );
				vMarker( i, j ).assign( K_COLLIDER );

			} );

		} );

		// no flux (collider surface)
		noFluxProjectionU = tsl_array_n.kernel( uSize, ( i, j ) => {

			const pt = velocity.uPosition( i, j );

			tsl_array_n.If( ls.isInsideSdf( collider.sample( pt ) ), () => {

				const colliderVel = collider.velocityAt( pt );
				const vel = velocity.sample( pt );
				const g = collider.gradient( pt );

				tsl_array_n.If( g.length().greaterThan( 0 ), () => {

					const n = g.normalize();
					const velr = vel.sub( colliderVel );
					const velt = projectAndApplyFriction( velr, n, collider.frictionCoefficient );
					const velp = velt.add( colliderVel );

					uTemp( i, j ).assign( velp.x );

				} ).Else( () => {

					uTemp( i, j ).assign( colliderVel.x );

				} );

			} ).Else( () => {

				uTemp( i, j ).assign( velocity.dataU( i, j ) );

			} );

		} );

		noFluxProjectionV = tsl_array_n.kernel( vSize, ( i, j ) => {

			const pt = velocity.vPosition( i, j );

			tsl_array_n.If( ls.isInsideSdf( collider.sample( pt ) ), () => {

				const colliderVel = collider.velocityAt( pt );
				const vel = velocity.sample( pt );
				const g = collider.gradient( pt );

				tsl_array_n.If( g.length().greaterThan( 0 ), () => {

					const n = g.normalize();
					const velr = vel.sub( colliderVel );
					const velt = projectAndApplyFriction( velr, n, collider.frictionCoefficient );
					const velp = velt.add( colliderVel );

					vTemp( i, j ).assign( velp.y );

				} ).Else( () => {

					vTemp( i, j ).assign( colliderVel.y );

				} );

			} ).Else( () => {

				vTemp( i, j ).assign( velocity.dataV( i, j ) );

			} );

		} );

		// blocked boundary condition
		blockedBoundary = tsl_array_n.kernel( [ nx, ny ], ( i, j ) => {

			tsl_array_n.If( blockMarker( i, j ).equal( K_COLLIDER ), () => {

				tsl_array_n.If( i.greaterThan( 0 ).and( blockMarker( i.sub( 1 ), j ).equal( K_FLUID ) ), () => {

					velocity.dataU( i, j ).assign( collider.velocityAt( velocity.uPosition( i, j ) ).x );

				} );

				tsl_array_n.If( i.lessThan( nx - 1 ).and( blockMarker( i.add( 1 ), j ).equal( K_FLUID ) ), () => {

					velocity.dataU( i.add( 1 ), j ).assign( collider.velocityAt( velocity.uPosition( i.add( 1 ), j ) ).x );

				} );

				tsl_array_n.If( j.greaterThan( 0 ).and( blockMarker( i, j.sub( 1 ) ).equal( K_FLUID ) ), () => {

					velocity.dataV( i, j ).assign( collider.velocityAt( velocity.vPosition( i, j ) ).y );

				} );

				tsl_array_n.If( j.lessThan( ny - 1 ).and( blockMarker( i, j.add( 1 ) ).equal( K_FLUID ) ), () => {

					velocity.dataV( i, j.add( 1 ) ).assign( collider.velocityAt( velocity.vPosition( i, j.add( 1 ) ) ).y );

				} );

			} );

		} );

		extrapolateU = createExtrapolateToRegion2( velocity.dataU, uMarker, velocity.dataU, uSize );
		extrapolateV = createExtrapolateToRegion2( velocity.dataV, vMarker, velocity.dataV, vSize );

	}

	// 换一个 collider（包括从 null 换成有、或从有换成 null）都走这里，会重新构建
	// 所有依赖 collider 的 kernel，以及 blockMarker。gridSize/gridSpacing/gridOrigin
	// 只是存下来跟源码保持一致——检查过这个文件内部实际没有用到它们（可能是给 grid/
	// 之外的代码用的占位），这里不是漏搬逻辑。
	function setCollider( newCollider, gridSize, gridSpacingXY, gridOrigin ) {

		solver.collider = newCollider;
		solver.gridSize = gridSize;
		solver.gridSpacing = gridSpacingXY;
		solver.gridOrigin = gridOrigin;

		rebuildColliderKernels();

		if ( ! newCollider ) {

			blockMarker.fromArray( new Int32Array( nx * ny ).fill( K_FLUID ) );

		} else {

			buildBlockMarker();

		}

	}

	// velocity 是构造时绑定的那一个（见文件头注释），不再是参数
	function constrainVelocity( extrapolationDepth = 5 ) {

		if ( solver.collider ) {

			fillUMarker();
			fillVMarker();

			markAndProjectU();
			markAndProjectV();

			// free slip - extrapolate
			extrapolateU( extrapolationDepth );
			extrapolateV( extrapolationDepth );

			// no flux (collider surface)
			noFluxProjectionU();
			noFluxProjectionV();

			copyUTempToVelocity();
			copyVTempToVelocity();

			// blocked boundary condition
			blockedBoundary();

		}

		// no flux (domain boundary, if closed) - 跟 collider 无关，没有 collider 时也要做
		projectClosedDomainBoundary();

	}

	solver.velocity = velocity;
	solver.setCollider = setCollider;
	solver.constrainVelocity = constrainVelocity;

	setCollider( colliderSDF, [ resolutionX, resolutionY ], [ gridSpacingX, gridSpacingY ], [ originX, originY ] );

	return solver;

}
