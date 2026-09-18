// 3D counterpart of grid_blocked_boundary_condition_solver2.js. Same
// "velocity bound at construction, collider swappable, kernels rebuilt on
// setCollider()" structure -- see that file's own header comment for the
// full reasoning, none of which is dimension-specific. What changes: a
// third staggered component (W), 6 domain walls instead of 4
// (DIRECTION_BACK/FRONT alongside LEFT/RIGHT/DOWN/UP), and
// blockedBoundary's blocked/fluid neighbour check covers 6 face neighbours
// instead of 4.

import { vec3, float, clamp } from 'three/tsl';
import * as tsl_array_n from 'tsl_array_n';
import * as ls from './level_set_utils.js';
import { createCopyKernel3, createExtrapolateToRegion3 } from './array_utils3.js';
import { DIRECTION_LEFT, DIRECTION_RIGHT, DIRECTION_DOWN, DIRECTION_UP, DIRECTION_BACK, DIRECTION_FRONT, DIRECTION_ALL_3D } from './constant.js';
import { isNonFinite } from '../float_guards.js';

// Same reasoning and same value as grid_blocked_boundary_condition_solver2.js's
// own MAX_VELOCITY_COMPONENT -- see that file's header comment for the
// real-hardware history behind this number; nothing about it is
// dimension-specific.
const MAX_VELOCITY_COMPONENT = 100;

const K_FLUID = 1;
const K_COLLIDER = 0;

// Verbatim copy of grid_blocked_boundary_condition_solver2.js's
// projectAndApplyFriction -- vel/normal are opaque vector nodes
// (.dot()/.length() work identically for vec3), nothing here is
// dimension-specific.
function projectAndApplyFriction( vel, normal, frictionCoefficient ) {

	const velt = vel.sub( normal.mul( vel.dot( normal ) ) ).toVar();

	tsl_array_n.If( velt.length().greaterThan( 0 ), () => {

		const veln = vel.dot( normal ).negate().max( 0 );
		const scale = float( 1 ).sub( veln.mul( frictionCoefficient ).div( velt.length() ) ).max( 0 );

		velt.mulAssign( scale );

	} );

	return velt;

}

// velocity: the FaceCenteredGrid3 bound at construction time.
// colliderSDF: optional, an SDFStaticCollider3/SDFRigidBodyCollider3; when
// null, constrainVelocity only does the domain-boundary part.
// inflows: optional, one createSDFInflow3(...) object or an array of them.
export function createGridBlockedBoundaryConditionSolver3(
	velocity,
	resolutionX, resolutionY, resolutionZ, gridSpacingX, gridSpacingY, gridSpacingZ, originX, originY, originZ,
	colliderSDF = null,
	inflows = null
) {

	const nx = resolutionX;
	const ny = resolutionY;
	const nz = resolutionZ;
	const uSize = [ nx + 1, ny, nz ];
	const vSize = [ nx, ny + 1, nz ];
	const wSize = [ nx, ny, nz + 1 ];

	const uMarker = tsl_array_n.arrayN( 'int', uSize );
	const vMarker = tsl_array_n.arrayN( 'int', vSize );
	const wMarker = tsl_array_n.arrayN( 'int', wSize );
	const uTemp = tsl_array_n.arrayN( 'float', uSize );
	const vTemp = tsl_array_n.arrayN( 'float', vSize );
	const wTemp = tsl_array_n.arrayN( 'float', wSize );
	const blockMarker = tsl_array_n.arrayN( 'int', [ nx, ny, nz ] );

	// Same last-resort circuit breaker as the 2D file's own clampComponent
	// -- see that file's header comment for the real-hardware history.
	function clampComponent( value ) {

		return isNonFinite( value ).select( float( 0 ), clamp( value, - MAX_VELOCITY_COMPONENT, MAX_VELOCITY_COMPONENT ) );

	}

	const clampVelocityU = tsl_array_n.kernel( uSize, ( i, j, k ) => {

		velocity.dataU( i, j, k ).assign( clampComponent( velocity.dataU( i, j, k ) ) );

	} );

	const clampVelocityV = tsl_array_n.kernel( vSize, ( i, j, k ) => {

		velocity.dataV( i, j, k ).assign( clampComponent( velocity.dataV( i, j, k ) ) );

	} );

	const clampVelocityW = tsl_array_n.kernel( wSize, ( i, j, k ) => {

		velocity.dataW( i, j, k ).assign( clampComponent( velocity.dataW( i, j, k ) ) );

	} );

	const solver = {
		uMarker, vMarker, wMarker, uTemp, vTemp, wTemp, blockMarker,
		closedDomainBoundaryFlag: DIRECTION_ALL_3D,
		collider: null
	};

	// ---- kernels that only depend on velocity: built once at construction ----

	const copyUTempToVelocity = createCopyKernel3( uTemp, velocity.dataU, uSize );
	const copyVTempToVelocity = createCopyKernel3( vTemp, velocity.dataV, vSize );
	const copyWTempToVelocity = createCopyKernel3( wTemp, velocity.dataW, wSize );

	// Each is a 2D slice of the relevant face -- e.g. zeroULeft zeroes the
	// whole i=0 plane of dataU, one thread per (j,k).
	const zeroULeft  = tsl_array_n.kernel( [ uSize[ 1 ], uSize[ 2 ] ], ( j, k ) => { velocity.dataU( 0, j, k ).assign( 0 ); } );
	const zeroURight = tsl_array_n.kernel( [ uSize[ 1 ], uSize[ 2 ] ], ( j, k ) => { velocity.dataU( uSize[ 0 ] - 1, j, k ).assign( 0 ); } );
	const zeroVDown  = tsl_array_n.kernel( [ vSize[ 0 ], vSize[ 2 ] ], ( i, k ) => { velocity.dataV( i, 0, k ).assign( 0 ); } );
	const zeroVUp    = tsl_array_n.kernel( [ vSize[ 0 ], vSize[ 2 ] ], ( i, k ) => { velocity.dataV( i, vSize[ 1 ] - 1, k ).assign( 0 ); } );
	const zeroWBack  = tsl_array_n.kernel( [ wSize[ 0 ], wSize[ 1 ] ], ( i, j ) => { velocity.dataW( i, j, 0 ).assign( 0 ); } );
	const zeroWFront = tsl_array_n.kernel( [ wSize[ 0 ], wSize[ 1 ] ], ( i, j ) => { velocity.dataW( i, j, wSize[ 2 ] - 1 ).assign( 0 ); } );

	// ---- inflow: one (applyU, applyV, applyW) kernel triple per inflow object ----

	function buildInflowKernels( inflow ) {

		const applyU = tsl_array_n.kernel( uSize, ( i, j, k ) => {

			const pt = velocity.uPosition( i, j, k );

			tsl_array_n.If( ls.isInsideSdf( inflow.sample( pt ) ), () => {

				const value = inflow.velocity.x;
				if ( inflow.mode === 'add' ) velocity.dataU( i, j, k ).addAssign( value );
				else velocity.dataU( i, j, k ).assign( value );

			} );

		} );

		const applyV = tsl_array_n.kernel( vSize, ( i, j, k ) => {

			const pt = velocity.vPosition( i, j, k );

			tsl_array_n.If( ls.isInsideSdf( inflow.sample( pt ) ), () => {

				const value = inflow.velocity.y;
				if ( inflow.mode === 'add' ) velocity.dataV( i, j, k ).addAssign( value );
				else velocity.dataV( i, j, k ).assign( value );

			} );

		} );

		const applyW = tsl_array_n.kernel( wSize, ( i, j, k ) => {

			const pt = velocity.wPosition( i, j, k );

			tsl_array_n.If( ls.isInsideSdf( inflow.sample( pt ) ), () => {

				const value = inflow.velocity.z;
				if ( inflow.mode === 'add' ) velocity.dataW( i, j, k ).addAssign( value );
				else velocity.dataW( i, j, k ).assign( value );

			} );

		} );

		return { applyU, applyV, applyW };

	}

	let inflowKernels = [];
	let plansGeneration = 0;

	function setInflows( newInflows ) {

		solver.inflows = newInflows;

		const list = ! newInflows ? [] : ( Array.isArray( newInflows ) ? newInflows : [ newInflows ] );
		inflowKernels = list.map( buildInflowKernels );
		plansGeneration ++;

	}

	// ---- kernels that depend on collider: rebuilt on setCollider() ----

	let fillUMarker = null, fillVMarker = null, fillWMarker = null;
	let buildBlockMarker = null;
	let markAndProjectU = null, markAndProjectV = null, markAndProjectW = null;
	let noFluxProjectionU = null, noFluxProjectionV = null, noFluxProjectionW = null;
	let blockedBoundary = null;
	let extrapolateU = null, extrapolateV = null, extrapolateW = null;

	function rebuildColliderKernels() {

		plansGeneration ++;

		const collider = solver.collider;

		if ( ! collider ) {

			fillUMarker = fillVMarker = fillWMarker = buildBlockMarker = null;
			markAndProjectU = markAndProjectV = markAndProjectW = null;
			noFluxProjectionU = noFluxProjectionV = noFluxProjectionW = null;
			blockedBoundary = null;
			extrapolateU = extrapolateV = extrapolateW = null;
			return;

		}

		fillUMarker = tsl_array_n.kernel( uSize, ( i, j, k ) => { uMarker( i, j, k ).assign( 1 ); } );
		fillVMarker = tsl_array_n.kernel( vSize, ( i, j, k ) => { vMarker( i, j, k ).assign( 1 ); } );
		fillWMarker = tsl_array_n.kernel( wSize, ( i, j, k ) => { wMarker( i, j, k ).assign( 1 ); } );

		buildBlockMarker = tsl_array_n.kernel( [ nx, ny, nz ], ( i, j, k ) => {

			blockMarker( i, j, k ).assign( collider.isInside( i, j, k ).select( K_COLLIDER, K_FLUID ) );

		} );

		function makeMarkAndProject( size, positionFn, axisOffset, dataComponent, markerField, component ) {

			return tsl_array_n.kernel( size, ( i, j, k ) => {

				const pt = positionFn( i, j, k );
				const h = velocity.gridSpacing;
				const offset = axisOffset( h );

				const phi0 = collider.sample( pt.sub( offset ) );
				const phi1 = collider.sample( pt.add( offset ) );

				const frac = float( 1 ).sub( ls.fractionInsideSdf( phi0, phi1 ).clamp( 0, 1 ) );

				tsl_array_n.If( frac.greaterThan( 0 ), () => {

					markerField( i, j, k ).assign( K_FLUID );

				} ).Else( () => {

					dataComponent( i, j, k ).assign( collider.velocityAt( pt )[ component ] );
					markerField( i, j, k ).assign( K_COLLIDER );

				} );

			} );

		}

		markAndProjectU = makeMarkAndProject( uSize, velocity.uPosition, ( h ) => vec3( h.x.mul( 0.5 ), 0, 0 ), velocity.dataU, uMarker, 'x' );
		markAndProjectV = makeMarkAndProject( vSize, velocity.vPosition, ( h ) => vec3( 0, h.y.mul( 0.5 ), 0 ), velocity.dataV, vMarker, 'y' );
		markAndProjectW = makeMarkAndProject( wSize, velocity.wPosition, ( h ) => vec3( 0, 0, h.z.mul( 0.5 ) ), velocity.dataW, wMarker, 'z' );

		function makeNoFluxProjection( size, positionFn, dataComponent, tempField, component ) {

			return tsl_array_n.kernel( size, ( i, j, k ) => {

				const pt = positionFn( i, j, k );

				tsl_array_n.If( ls.isInsideSdf( collider.sample( pt ) ), () => {

					const colliderVel = collider.velocityAt( pt );
					const vel = velocity.sample( pt );
					const g = collider.gradient( pt );

					tsl_array_n.If( g.length().greaterThan( 0 ), () => {

						const n = g.normalize();
						const velr = vel.sub( colliderVel );
						const velt = projectAndApplyFriction( velr, n, collider.frictionCoefficient );
						const velp = velt.add( colliderVel );

						tempField( i, j, k ).assign( velp[ component ] );

					} ).Else( () => {

						tempField( i, j, k ).assign( colliderVel[ component ] );

					} );

				} ).Else( () => {

					tempField( i, j, k ).assign( dataComponent( i, j, k ) );

				} );

			} );

		}

		noFluxProjectionU = makeNoFluxProjection( uSize, velocity.uPosition, velocity.dataU, uTemp, 'x' );
		noFluxProjectionV = makeNoFluxProjection( vSize, velocity.vPosition, velocity.dataV, vTemp, 'y' );
		noFluxProjectionW = makeNoFluxProjection( wSize, velocity.wPosition, velocity.dataW, wTemp, 'z' );

		// blocked boundary condition: a collider cell with a fluid neighbour
		// on any of its 6 faces gets that face's own velocity component set
		// from the collider's rigid-body velocity there -- same rule as the
		// 2D file's own 4-neighbour version, 2 more faces.
		blockedBoundary = tsl_array_n.kernel( [ nx, ny, nz ], ( i, j, k ) => {

			tsl_array_n.If( blockMarker( i, j, k ).equal( K_COLLIDER ), () => {

				tsl_array_n.If( i.greaterThan( 0 ).and( blockMarker( i.sub( 1 ), j, k ).equal( K_FLUID ) ), () => {

					velocity.dataU( i, j, k ).assign( collider.velocityAt( velocity.uPosition( i, j, k ) ).x );

				} );

				tsl_array_n.If( i.lessThan( nx - 1 ).and( blockMarker( i.add( 1 ), j, k ).equal( K_FLUID ) ), () => {

					velocity.dataU( i.add( 1 ), j, k ).assign( collider.velocityAt( velocity.uPosition( i.add( 1 ), j, k ) ).x );

				} );

				tsl_array_n.If( j.greaterThan( 0 ).and( blockMarker( i, j.sub( 1 ), k ).equal( K_FLUID ) ), () => {

					velocity.dataV( i, j, k ).assign( collider.velocityAt( velocity.vPosition( i, j, k ) ).y );

				} );

				tsl_array_n.If( j.lessThan( ny - 1 ).and( blockMarker( i, j.add( 1 ), k ).equal( K_FLUID ) ), () => {

					velocity.dataV( i, j.add( 1 ), k ).assign( collider.velocityAt( velocity.vPosition( i, j.add( 1 ), k ) ).y );

				} );

				tsl_array_n.If( k.greaterThan( 0 ).and( blockMarker( i, j, k.sub( 1 ) ).equal( K_FLUID ) ), () => {

					velocity.dataW( i, j, k ).assign( collider.velocityAt( velocity.wPosition( i, j, k ) ).z );

				} );

				tsl_array_n.If( k.lessThan( nz - 1 ).and( blockMarker( i, j, k.add( 1 ) ).equal( K_FLUID ) ), () => {

					velocity.dataW( i, j, k.add( 1 ) ).assign( collider.velocityAt( velocity.wPosition( i, j, k.add( 1 ) ) ).z );

				} );

			} );

		} );

		extrapolateU = createExtrapolateToRegion3( velocity.dataU, uMarker, velocity.dataU, uSize );
		extrapolateV = createExtrapolateToRegion3( velocity.dataV, vMarker, velocity.dataV, vSize );
		extrapolateW = createExtrapolateToRegion3( velocity.dataW, wMarker, velocity.dataW, wSize );

	}

	// See grid_blocked_boundary_condition_solver2.js's own setCollider/
	// colliderMoved header comments for the full "two verbs" reasoning and
	// its measured cost -- unchanged here, one dimension wider.
	function setCollider( newCollider, gridSize, gridSpacingXYZ, gridOrigin ) {

		solver.collider = newCollider;
		solver.gridSize = gridSize;
		solver.gridSpacing = gridSpacingXYZ;
		solver.gridOrigin = gridOrigin;

		rebuildColliderKernels();

		if ( ! newCollider ) {

			blockMarker.fromArray( new Int32Array( nx * ny * nz ).fill( K_FLUID ) );

		} else {

			buildBlockMarker();

		}

	}

	function colliderMoved() {

		if ( ! solver.collider ) {

			throw new Error(
				'grid_blocked_boundary_condition_solver3: colliderMoved() called with no collider bound. ' +
				'Use setCollider( collider, gridSize, gridSpacing, gridOrigin ) to bind one, or to replace the one bound.'
			);

		}

		buildBlockMarker();

	}

	// See grid_blocked_boundary_condition_solver2.js's own constrainVelocityPlan
	// header comment for why this is one batched submission -- unchanged
	// reasoning, one more face-component's worth of kernels.
	const plans = new Map();

	function constrainVelocityPlan( extrapolationDepth ) {

		const sequence = [];

		if ( solver.collider ) {

			sequence.push( fillUMarker, fillVMarker, fillWMarker, markAndProjectU, markAndProjectV, markAndProjectW );

			sequence.push( ...extrapolateU.dispatchers( extrapolationDepth ) );
			sequence.push( ...extrapolateV.dispatchers( extrapolationDepth ) );
			sequence.push( ...extrapolateW.dispatchers( extrapolationDepth ) );

			sequence.push( noFluxProjectionU, noFluxProjectionV, noFluxProjectionW );
			sequence.push( copyUTempToVelocity, copyVTempToVelocity, copyWTempToVelocity );
			sequence.push( blockedBoundary );

		}

		const flag = solver.closedDomainBoundaryFlag;

		if ( flag & DIRECTION_LEFT ) sequence.push( zeroULeft );
		if ( flag & DIRECTION_RIGHT ) sequence.push( zeroURight );
		if ( flag & DIRECTION_DOWN ) sequence.push( zeroVDown );
		if ( flag & DIRECTION_UP ) sequence.push( zeroVUp );
		if ( flag & DIRECTION_BACK ) sequence.push( zeroWBack );
		if ( flag & DIRECTION_FRONT ) sequence.push( zeroWFront );

		for ( const { applyU, applyV, applyW } of inflowKernels ) sequence.push( applyU, applyV, applyW );

		sequence.push( clampVelocityU, clampVelocityV, clampVelocityW );

		return tsl_array_n.createBatch( sequence );

	}

	function constrainVelocity( extrapolationDepth = 5 ) {

		const key = `${ plansGeneration }|${ extrapolationDepth }|${ solver.closedDomainBoundaryFlag }`;

		let plan = plans.get( key );

		if ( plan === undefined ) {

			plan = constrainVelocityPlan( extrapolationDepth );
			plans.set( key, plan );

		}

		plan();

	}

	solver.velocity = velocity;
	solver.setCollider = setCollider;
	solver.setInflows = setInflows;
	solver.constrainVelocity = constrainVelocity;
	solver.colliderMoved = colliderMoved;

	setCollider( colliderSDF, [ resolutionX, resolutionY, resolutionZ ], [ gridSpacingX, gridSpacingY, gridSpacingZ ], [ originX, originY, originZ ] );
	setInflows( inflows );

	return solver;

}
