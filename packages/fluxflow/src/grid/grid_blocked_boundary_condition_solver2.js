// Ported from grid_blocked_boundary_condition_solver2.py.
//
// The biggest structural difference from the source: the source's
// constructor doesn't take velocity (velocity is a ti.template() argument
// passed fresh on every constrainVelocity(velocity,...) call) -- here it's
// passed once, fixed, at construction time instead (a FaceCenteredGrid2) --
// tsl_array_n's kernel(shape,fn) has to closure-bind a concrete field at
// build time, with no ability to "rebind arguments on every call" (port
// plan decision 4, not a new limitation introduced here). So
// constrainVelocity() here no longer takes a velocity argument, and always
// operates on the one bound at construction.
//
// collider is different -- the source itself supports swapping the collider
// mid-lifetime via setCollider() (including going from None to a real one,
// or from one collider to another), so every kernel that reads collider
// can't be built once and for all at construction time; they go into
// rebuildColliderKernels() instead, which reruns on every setCollider() call
// (building a kernel graph doesn't itself need a GPU/renderer, so it's cheap
// and rebuilding is fine). The kernels that only depend on velocity, not
// collider (the four _zero* ones) only need to be built once, at
// construction time.

import { vec2, float, clamp } from 'three/tsl';
import * as tsl_array_n from 'tsl_array_n';
import * as ls from './level_set_utils.js';
import { createCopyKernel2, createExtrapolateToRegion2 } from './array_utils.js';
import { DIRECTION_LEFT, DIRECTION_RIGHT, DIRECTION_DOWN, DIRECTION_UP, DIRECTION_ALL } from './constant.js';

// Last-resort bound on a single velocity component, applied every frame at
// the very end of constrainVelocity() -- see that function's own use of
// this, below, for the full reasoning.
//
// *** Confirmed necessary on real hardware, not a defensive-but-never-
// triggered guard: a real-hardware long-run test of a large (256x128),
// fast-inflow scene showed velocity growing into the *billions* --
// technically still "finite" (Number.isFinite would not catch it) but no
// less broken a result than an outright NaN -- while grid_pressure_
// solver2.js's own per-frame circuit breaker (see that file's own header
// comment) kept *pressure* itself bounded throughout. That combination
// means a persistently-non-converging pressure solve (this port's
// existing MGPCG, undersized/undertuned for a large-enough or fast-enough
// scene) can still let velocity itself run away, frame over frame, even
// with pressure never once going non-finite -- the two fields need
// independent protection, not just one. ***
//
// 1000 is astronomically larger than any physically-intended velocity in
// this port's own examples (all stay under ~30 even at their most
// energetic) -- generous on purpose, this only needs to catch a genuine
// runaway, not bound normal physical variation.
const MAX_VELOCITY_COMPONENT = 1000;

// Same meaning as uMarker/vMarker (1=fluid, 0=collider); kept file-local
// just like in the source, not moved into constant.js
const K_FLUID = 1;
const K_COLLIDER = 0;

// velt = vel - normal * dot(vel,normal); if velt's length > 0, decay the
// tangential component by the friction coefficient. This is a "may modify
// in place" pattern (not a pure value selection), so it uses
// .toVar() + If(), not select().
function projectAndApplyFriction( vel, normal, frictionCoefficient ) {

	const velt = vel.sub( normal.mul( vel.dot( normal ) ) ).toVar();

	tsl_array_n.If( velt.length().greaterThan( 0 ), () => {

		const veln = vel.dot( normal ).negate().max( 0 );
		const scale = float( 1 ).sub( veln.mul( frictionCoefficient ).div( velt.length() ) ).max( 0 );

		velt.mulAssign( scale );

	} );

	return velt;

}

// velocity: the FaceCenteredGrid2 bound at construction time (see the file
// header comment).
// colliderSDF: optional, an SDFStaticCollider2/SDFRigidBodyCollider2; when
// null, constrainVelocity only does the domain-boundary part.
// inflows: optional, one createSDFInflow2(...) object (sdf_inflow_outflow2.js)
// or an array of them -- see setInflows below for how these get applied.
export function createGridBlockedBoundaryConditionSolver2(
	velocity,
	resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY,
	colliderSDF = null,
	inflows = null
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

	// Last-resort circuit breaker -- see MAX_VELOCITY_COMPONENT's own
	// comment above. NaN is set to exactly 0 (not clamped -- a NaN
	// compared against anything is always false, so a plain clamp() would
	// leave it untouched), anything else is bounded to
	// +/-MAX_VELOCITY_COMPONENT.
	function clampComponent( value ) {

		const isNaN = value.notEqual( value );
		return isNaN.select( float( 0 ), clamp( value, - MAX_VELOCITY_COMPONENT, MAX_VELOCITY_COMPONENT ) );

	}

	const clampVelocityU = tsl_array_n.kernel( uSize, ( i, j ) => {

		velocity.dataU( i, j ).assign( clampComponent( velocity.dataU( i, j ) ) );

	} );

	const clampVelocityV = tsl_array_n.kernel( vSize, ( i, j ) => {

		velocity.dataV( i, j ).assign( clampComponent( velocity.dataV( i, j ) ) );

	} );

	const solver = {
		uMarker, vMarker, uTemp, vTemp, blockMarker,
		closedDomainBoundaryFlag: DIRECTION_ALL, // plain mutable property, same effect as the source's setClosedDomainBoundaryFlag
		collider: null
	};

	// ---- kernels that only depend on velocity: independent of collider, built once at construction ----

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

	// ---- inflow: one (applyU, applyV) kernel pair per inflow object, rebuilt whenever setInflows() is called ----

	// inflow.mode is read as a plain JS value *here*, at kernel-build time
	// (inside setInflows()/this function), not per-dispatch -- reassigning
	// inflow.mode after this kernel has already been built only takes
	// effect the next time setInflows() rebuilds it, matching the same
	// "baked in at kernel-build time" limitation sdf_inflow_outflow2.js's
	// own header comment already documents for this exact property.
	function buildInflowKernels( inflow ) {

		const applyU = tsl_array_n.kernel( uSize, ( i, j ) => {

			const pt = velocity.uPosition( i, j );

			tsl_array_n.If( ls.isInsideSdf( inflow.sample( pt ) ), () => {

				const value = inflow.velocity.x;
				if ( inflow.mode === 'add' ) velocity.dataU( i, j ).addAssign( value );
				else velocity.dataU( i, j ).assign( value );

			} );

		} );

		const applyV = tsl_array_n.kernel( vSize, ( i, j ) => {

			const pt = velocity.vPosition( i, j );

			tsl_array_n.If( ls.isInsideSdf( inflow.sample( pt ) ), () => {

				const value = inflow.velocity.y;
				if ( inflow.mode === 'add' ) velocity.dataV( i, j ).addAssign( value );
				else velocity.dataV( i, j ).assign( value );

			} );

		} );

		return { applyU, applyV };

	}

	let inflowKernels = [];

	// newInflows: null, one createSDFInflow2(...) object, or an array of
	// them -- mirrors setCollider's own "swap it at any time" precedent
	// (genuine "customize later" support, not just a fixed constructor
	// option).
	function setInflows( newInflows ) {

		solver.inflows = newInflows;

		const list = ! newInflows ? [] : ( Array.isArray( newInflows ) ? newInflows : [ newInflows ] );
		inflowKernels = list.map( buildInflowKernels );

	}

	function applyInflow() {

		for ( const { applyU, applyV } of inflowKernels ) {

			applyU();
			applyV();

		}

	}

	// ---- kernels that depend on collider: need rebuilding when it changes (see file header comment) ----

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

	// Swapping the collider (whether from null to a real one, or from one
	// real one to null) always goes through here, and rebuilds every
	// collider-dependent kernel plus blockMarker. gridSize/gridSpacing/
	// gridOrigin are stored purely to match the source -- checked that
	// nothing in this file actually reads them (possibly a placeholder for
	// code outside grid/), this isn't a missed piece of logic.
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

	// velocity is the one bound at construction time (see the file header
	// comment), no longer a parameter here
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

		// no flux (domain boundary, if closed) - independent of collider, still needed even without one
		projectClosedDomainBoundary();

		// inflow always wins if a caller's closedDomainBoundaryFlag still
		// happens to include the same wall an inflow object overlaps.
		applyInflow();

		// Last-resort circuit breaker, always run last -- see
		// MAX_VELOCITY_COMPONENT's own comment above for why this exists
		// independently of grid_pressure_solver2.js's own pressure-side
		// one. constrainVelocity() already runs after every stage each
		// frame (forces, pressure, advection -- see this file's own
		// header comment), so this one addition catches a runaway
		// regardless of which stage actually produced it.
		clampVelocityU();
		clampVelocityV();

	}

	solver.velocity = velocity;
	solver.setCollider = setCollider;
	solver.setInflows = setInflows;
	solver.constrainVelocity = constrainVelocity;

	setCollider( colliderSDF, [ resolutionX, resolutionY ], [ gridSpacingX, gridSpacingY ], [ originX, originY ] );
	setInflows( inflows );

	return solver;

}
