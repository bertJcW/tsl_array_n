// Semi-Lagrangian advection, ported from jet/fluid-engine-dev -- no Python
// counterpart (the Python source's grid_solver2.py never got past an
// abstract "computeAdvection" hook, see grid_solver2.js's own header
// comment). See ../../THIRD-PARTY-NOTICES.md for the attribution.
//
// Two pieces, matching jet's own separation of concerns (SemiLagrangian2's
// shared backTrace vs. CubicSemiLagrangian2's overridden sampler -- this
// project's factory-function style folds both into one factory instead of
// a base/override class pair):
//
// 1. backTrace(): traces a point backward through the velocity field for
//    `dt`, using adaptive-substep 2nd-order midpoint (RK2) integration --
//    ported from SemiLagrangian2::backTrace (semi_lagrangian2.cpp). This
//    *is* the boundary handling: after each substep, if the segment would
//    cross into the collider (the boundary SDF changes sign between the
//    substep's start and end), the traced point is clamped to
//    (approximately) the crossing point via linear interpolation in SDF
//    value, and tracing stops there -- not a separate pass bolted on
//    afterward.
// 2. The actual field sample at the traced-back position uses monotonic
//    cubic interpolation (grid_math.js's collocatedCubicValueAtPosition2/
//    faceCenteredCubicValueAtPosition2 -- Fedkiw, Stam & Jensen's
//    clamped-Catmull-Rom scheme, see grid_math.js's own comment) rather
//    than linear, per explicit request. Velocity sampling *during*
//    backTrace's own RK2 steps still uses the velocity grid's existing
//    (bilinear) sample() -- jet's own backTrace does the same regardless
//    of the outer scheme's order; only the final value lookup benefits
//    from the higher order.
//
// tsl_array_n has no native dynamic-count while-loop, only a bounded
// Loop() with early Break() -- backTrace's adaptive substep count (driven
// by the local CFL number) is data-dependent, so this uses a fixed
// maxSubsteps cap (default 32) with an early Break() once the remaining
// time is exhausted or a boundary crossing is found, following the exact
// pattern already proven in tsl_array_n/examples/04-julia/main.js
// (Loop(count, () => { If(cond, () => Break()); ...body...; })). A
// real-world CFL-bounded per-frame dt should need far fewer substeps than
// the cap in practice; silently truncating early (rather than erroring)
// if that assumption ever breaks is a deliberate choice, matching jet's
// own "best effort" character for this scheme.
//
// The boundary-crossing trigger below intentionally differs from jet's
// own `phi0*phi1 < 0` -- confirmed via a plain-JS trace of jet's exact
// condition to have a real edge case (not a translation bug: jet's own
// C++ has the identical condition) where a substep landing precisely on
// the boundary (phi0 == 0) makes the very next crossing undetectable,
// letting the trace leak straight through a collider -- easy to trigger
// with grid-aligned colliders and round velocity/dt values, confirmed for
// real on real WebGPU hardware before the fix below. See `phi1 <=
// 0`'s own comment.
//
// CONFIRMED correct on real WebGPU hardware: examples/08-cubic-interpolation/
// (the cubic sampler) and examples/09-advection/ (both the constant-
// velocity case and, after the fix above, the wall-collider case --
// clamps exactly to the wall's surface, and a cubic interpolant
// reconstructs data exactly at that grid-aligned point regardless of what
// the stencil's other taps touch, so the sampled value came back exactly
// right, not just "close enough").

import * as tsl_array_n from 'tsl_array_n';
import { float, min, max, length, ceil, abs, If, Loop, Break } from 'three/tsl';
import { collocatedCubicValueAtPosition2, faceCenteredCubicValueAtPosition2, bilinearCoordsAndWeights2 } from './grid_math.js';

const EPSILON = 1e-6;

// sampleVelocity/sampleBoundary: (pos) => node, closures bound by the
// factory below. dt: a node (see createSemiLagrangianAdvectionSolver2's
// own comment on why this isn't a plain JS number here). direction:
// default 1 (this function's own original, only behavior for order-1
// advection -- trace *backward* along velocity, unchanged). MacCormack's
// own second (mantaflow calls it its own "-dt") step needs the opposite:
// trace *forward* along velocity for the same duration -- direction=-1
// flips both sub-step formulas' own sign (`pt.sub(v.mul(dt).mul(-1))` ==
// `pt.add(v.mul(dt))`), reusing this exact same adaptive-substep/
// boundary-crossing-clamp logic for both directions rather than
// duplicating it. Provably inert when direction=1 (multiplying by a
// constant 1.0 node changes nothing, in exact IEEE754 arithmetic or after
// any reasonable shader-compiler constant-folding) -- order-1 callers are
// unaffected, confirmed by the real-hardware regression check this
// change's own verification round ran, not just asserted here.
function backTrace( sampleVelocity, sampleBoundary, startPos, dt, h, maxSubsteps, direction = 1 ) {

	const pt = startPos.toVar();
	const remainingT = dt.toVar();

	Loop( maxSubsteps, () => {

		If( remainingT.lessThanEqual( EPSILON ), () => {

			Break();

		} );

		const vel0 = sampleVelocity( pt );
		const numSubSteps = max( ceil( length( vel0 ).mul( remainingT ).div( h ) ), 1 );
		const subDt = remainingT.div( numSubSteps );

		const midPt = pt.sub( vel0.mul( subDt.mul( 0.5 ) ).mul( direction ) );
		const midVel = sampleVelocity( midPt );
		const nextPt = pt.sub( midVel.mul( subDt ).mul( direction ) );

		const phi0 = sampleBoundary( pt );
		const phi1 = sampleBoundary( nextPt );

		// jet's own trigger is phi0*phi1 < 0 (the SDF changes sign across
		// this substep) -- confirmed via a plain-JS trace of this exact
		// algorithm to have a real edge case: whenever a substep happens to
		// land *exactly* on the boundary (phi0 == 0, not uncommon with
		// grid-aligned colliders and round velocity/dt values), the product
		// is 0, never negative, so the very next substep's plunge into the
		// solid (phi1 < 0) goes completely undetected and tracing marches
		// straight through. Using phi1 <= 0 instead (the substep's *end*
		// is at or inside the solid) catches that case too, and still
		// degenerates to the same clamp jet's own w = |phi1|/(|phi0|+|phi1|)
		// formula produces in the normal (phi0 > 0) case.
		If( phi1.lessThanEqual( 0 ), () => {

			const w = abs( phi1 ).div( abs( phi0 ).add( abs( phi1 ) ) );
			pt.assign( pt.mul( w ).add( nextPt.mul( float( 1 ).sub( w ) ) ) );
			remainingT.assign( 0 );
			Break();

		} ).Else( () => {

			remainingT.subAssign( subDt );
			pt.assign( nextPt );

		} );

	} );

	return pt;

}

// options.velocityGrid: the FaceCenteredGrid2 (grid_data2.js) every
// advect call traces backward through, whether or not it's also the
// field being advected (self-advection).
// options.collider: optional SDFStaticCollider2/SDFRigidBodyCollider2
// (sdf_collider2.js); omit for an unbounded domain (sampleBoundary then
// always returns 1, i.e. "outside", matching jet's default
// ConstantScalarField2(kMaxD) -- "no boundary").
// options.dt: the time-step, a plain JS number (baked in as a constant at
// kernel-build time -- fine for a fixed-dt test) or a node obtained by
// *calling* an array0('float') field (e.g. `dtField()`, not the callable
// field reference itself -- see external_force_solver2.js's own dt
// comment for the exact real-hardware error passing the field unInvoked
// produces). The node returned by that call stays live: update the
// array0's *contents* via dtField.fromArray() between dispatches (on the
// field itself, not the node) and every already-built kernel here picks
// up the new value on its next dispatch, the same pattern linalg.js's
// alpha/beta scalars already rely on -- this factory itself only converts
// a plain number to a node once, so a real CFL-adaptive solver can share
// one dt field's own live node across every stage that needs it without
// rebuilding any kernel here.
// options.maxSubsteps: cap on backTrace's adaptive substep loop, default 32.
// options.order: 1 (default) or 2 -- matching mantaflow's own exact option
// name/values (source/plugin/advection.cpp's own `order` parameter). 1 is
// this file's original plain semi-Lagrangian step, completely unchanged.
// 2 is MacCormack -- read directly from mantaflow's own fnAdvectSemiLagrange/
// MacCormackCorrect/MacCormackClamp (see ../../THIRD-PARTY-NOTICES.md for
// the attribution): a second, backward-in-time trace estimates how much
// error the forward step introduced, corrects for half of it, then clamps
// the correction to the locally-observed range (mantaflow's own
// "clampMode 2", the variant its own code comment marks as "recommended
// in Andy's paper" -- Selle, Fedkiw, Kim, Liu, Rossignac, "An
// Unconditionally Stable MacCormack Method" -- over its own more complex
// clampMode 1) so the correction can never introduce a new, unphysical
// extremum -- the well-known fix for plain MacCormack's own unconditional
// instability. Costs 3 dispatches per advected field instead of 1
// (forward, backward, correct+clamp) -- a real, inherent cost, not this
// port's own overhead; defaults to 1 so every existing caller is
// completely unaffected.
export function createSemiLagrangianAdvectionSolver2( { velocityGrid, collider, dt, maxSubsteps = 32, order = 1 } ) {

	const dtNode = typeof dt === 'number' ? float( dt ) : dt;
	const h = min( velocityGrid.gridSpacing.x, velocityGrid.gridSpacing.y );

	function sampleVelocity( pos ) {

		return velocityGrid.sample( pos );

	}

	function sampleBoundary( pos ) {

		return collider ? collider.sample( pos ) : float( 1 );

	}

	// direction: see backTrace's own header comment -- default 1 (trace
	// backward, this file's original and only behavior for order 1).
	// MacCormack's own second step passes -1 (trace forward, the same
	// duration) to estimate the first step's own error -- already exercised
	// on real hardware via that path. grid_flip_solver2.js's own particle
	// advection reuses this same direction=-1 call directly (a particle's
	// own arbitrary position is just another startPos, no different from a
	// grid cell's), which is why this is returned below alongside the two
	// field-advection functions, not left as a construction-only detail.
	function trace( startPos, direction = 1 ) {

		return backTrace( sampleVelocity, sampleBoundary, startPos, dtNode, h, maxSubsteps, direction );

	}

	// Order-2 only: gathers orig-field min/max over the 4 grid cells
	// bilinearCoordsAndWeights2 finds around tracedPos (the exact same
	// position the forward step's own cubic sample used) -- mantaflow's
	// own doClampComponent/doClampComponentMAC, clampMode 2, adapted from
	// its FlagGrid-based "is this neighbor a valid fluid cell" check
	// (checkFlag) to this port's own SDF-collider convention:
	// sampleBoundary(pos) > 0 at each neighbor's own position (matching
	// how every other boundary check in this file already works).
	// data/dataOrigin/gridSpacing/shape/positionFn: the field being read
	// and its own coordinate system (input.data/.dataOrigin/.gridSpacing/
	// .dataSize/.dataPosition for a scalar field, or the dataU/dataV-
	// specific equivalents for a face-centered one -- see the call sites
	// below; taken explicitly rather than reaching into the outer
	// closure's own velocityGrid, so this stays correct even if a future
	// caller's field genuinely has its own distinct gridSpacing node).
	function gatherLocalMinMax( data, dataOrigin, gridSpacing, shape, positionFn, tracedPos ) {

		const { i0c, j0c, i1c, j1c } = bilinearCoordsAndWeights2( tracedPos, dataOrigin, gridSpacing, shape );
		const neighbors = [ [ i0c, j0c ], [ i1c, j0c ], [ i0c, j1c ], [ i1c, j1c ] ];

		let minv = float( 1e30 );
		let maxv = float( -1e30 );
		let haveValid = null;

		for ( const [ ni, nj ] of neighbors ) {

			const valid = sampleBoundary( positionFn( ni, nj ) ).greaterThan( 0 );
			const value = data( ni, nj );

			minv = valid.select( min( minv, value ), minv );
			maxv = valid.select( max( maxv, value ), maxv );
			haveValid = haveValid === null ? valid : haveValid.or( valid );

		}

		return { minv, maxv, haveValid };

	}

	// Order-2 only: mantaflow's own clampMode-2 decision -- if no valid
	// neighbor was found, or the corrected value falls outside the local
	// [min,max] range those neighbors' own orig values span, fall back to
	// the plain forward-step value instead of the (potentially
	// overshooting) corrected one.
	function clampCorrection( correctedValue, fwdValue, data, dataOrigin, gridSpacing, shape, positionFn, tracedPos ) {

		const { minv, maxv, haveValid } = gatherLocalMinMax( data, dataOrigin, gridSpacing, shape, positionFn, tracedPos );
		const inRange = correctedValue.greaterThanEqual( minv ).and( correctedValue.lessThanEqual( maxv ) );

		return haveValid.and( inRange ).select( correctedValue, fwdValue );

	}

	// input, output: FaceCenteredGrid2 (grid_data2.js) -- typically the
	// same velocity grid for self-advection, but any matching-shape pair
	// works (e.g. advecting a separate face-centered field through this
	// solver's own velocityGrid).
	function advectFaceCentered2( input, output ) {

		const dispatchU1 = tsl_array_n.kernel( input.dataSizeU, ( i, j ) => {

			const pos = input.uPosition( i, j );

			If( sampleBoundary( pos ).greaterThan( 0 ), () => {

				const tracedPos = trace( pos );
				output.dataU( i, j ).assign(
					faceCenteredCubicValueAtPosition2(
						input.dataU, input.dataV, input.gridSpacing,
						input.dataOriginU, input.dataOriginV, tracedPos,
						input.dataSizeU, input.dataSizeV
					).x
				);

			} );

		} );

		const dispatchV1 = tsl_array_n.kernel( input.dataSizeV, ( i, j ) => {

			const pos = input.vPosition( i, j );

			If( sampleBoundary( pos ).greaterThan( 0 ), () => {

				const tracedPos = trace( pos );
				output.dataV( i, j ).assign(
					faceCenteredCubicValueAtPosition2(
						input.dataU, input.dataV, input.gridSpacing,
						input.dataOriginU, input.dataOriginV, tracedPos,
						input.dataSizeU, input.dataSizeV
					).y
				);

			} );

		} );

		if ( order === 1 ) return function dispatch() {

			dispatchU1();
			dispatchV1();

		};

		// order 2 (MacCormack) -- see createSemiLagrangianAdvectionSolver2's
		// own header comment. fwdU/fwdV together form one scratch
		// FaceCenteredGrid2-shaped pair (same staggered layout as input),
		// since faceCenteredCubicValueAtPosition2 needs *both* components
		// together to interpolate correctly even when only one component's
		// own result is kept -- exactly how the order-1 kernels above
		// already read both input.dataU/input.dataV regardless of which
		// one (.x/.y) they extract.
		const fwdU = tsl_array_n.arrayN( 'float', input.dataSizeU );
		const fwdV = tsl_array_n.arrayN( 'float', input.dataSizeV );
		const bwdU = tsl_array_n.arrayN( 'float', input.dataSizeU );
		const bwdV = tsl_array_n.arrayN( 'float', input.dataSizeV );

		const dispatchForwardU = tsl_array_n.kernel( input.dataSizeU, ( i, j ) => {

			const pos = input.uPosition( i, j );

			If( sampleBoundary( pos ).greaterThan( 0 ), () => {

				const tracedPos = trace( pos );
				fwdU( i, j ).assign( faceCenteredCubicValueAtPosition2( input.dataU, input.dataV, input.gridSpacing, input.dataOriginU, input.dataOriginV, tracedPos, input.dataSizeU, input.dataSizeV ).x );

			} ).Else( () => {

				fwdU( i, j ).assign( input.dataU( i, j ) );

			} );

		} );

		const dispatchForwardV = tsl_array_n.kernel( input.dataSizeV, ( i, j ) => {

			const pos = input.vPosition( i, j );

			If( sampleBoundary( pos ).greaterThan( 0 ), () => {

				const tracedPos = trace( pos );
				fwdV( i, j ).assign( faceCenteredCubicValueAtPosition2( input.dataU, input.dataV, input.gridSpacing, input.dataOriginU, input.dataOriginV, tracedPos, input.dataSizeU, input.dataSizeV ).y );

			} ).Else( () => {

				fwdV( i, j ).assign( input.dataV( i, j ) );

			} );

		} );

		const dispatchBackwardU = tsl_array_n.kernel( input.dataSizeU, ( i, j ) => {

			const pos = input.uPosition( i, j );

			If( sampleBoundary( pos ).greaterThan( 0 ), () => {

				const tracedPos = trace( pos, -1 );
				bwdU( i, j ).assign( faceCenteredCubicValueAtPosition2( fwdU, fwdV, input.gridSpacing, input.dataOriginU, input.dataOriginV, tracedPos, input.dataSizeU, input.dataSizeV ).x );

			} ).Else( () => {

				bwdU( i, j ).assign( fwdU( i, j ) );

			} );

		} );

		const dispatchBackwardV = tsl_array_n.kernel( input.dataSizeV, ( i, j ) => {

			const pos = input.vPosition( i, j );

			If( sampleBoundary( pos ).greaterThan( 0 ), () => {

				const tracedPos = trace( pos, -1 );
				bwdV( i, j ).assign( faceCenteredCubicValueAtPosition2( fwdU, fwdV, input.gridSpacing, input.dataOriginU, input.dataOriginV, tracedPos, input.dataSizeU, input.dataSizeV ).y );

			} ).Else( () => {

				bwdV( i, j ).assign( fwdV( i, j ) );

			} );

		} );

		const dispatchCorrectAndClampU = tsl_array_n.kernel( input.dataSizeU, ( i, j ) => {

			const pos = input.uPosition( i, j );

			If( sampleBoundary( pos ).greaterThan( 0 ), () => {

				const tracedPos = trace( pos );
				const fwdValue = fwdU( i, j );
				const correctedValue = fwdValue.add( input.dataU( i, j ).sub( bwdU( i, j ) ).mul( 0.5 ) );
				const clamped = clampCorrection( correctedValue, fwdValue, input.dataU, input.dataOriginU, input.gridSpacing, input.dataSizeU, input.uPosition, tracedPos );

				output.dataU( i, j ).assign( clamped );

			} );

		} );

		const dispatchCorrectAndClampV = tsl_array_n.kernel( input.dataSizeV, ( i, j ) => {

			const pos = input.vPosition( i, j );

			If( sampleBoundary( pos ).greaterThan( 0 ), () => {

				const tracedPos = trace( pos );
				const fwdValue = fwdV( i, j );
				const correctedValue = fwdValue.add( input.dataV( i, j ).sub( bwdV( i, j ) ).mul( 0.5 ) );
				const clamped = clampCorrection( correctedValue, fwdValue, input.dataV, input.dataOriginV, input.gridSpacing, input.dataSizeV, input.vPosition, tracedPos );

				output.dataV( i, j ).assign( clamped );

			} );

		} );

		return function dispatch() {

			dispatchForwardU();
			dispatchForwardV();
			dispatchBackwardU();
			dispatchBackwardV();
			dispatchCorrectAndClampU();
			dispatchCorrectAndClampV();

		};

	}

	// input, output: ScalarGrid2 (grid_data2.js) -- e.g. density/
	// temperature, advected through this solver's own velocityGrid.
	function advectScalar2( input, output ) {

		const dispatchOrder1 = tsl_array_n.kernel( input.dataSize, ( i, j ) => {

			const pos = input.dataPosition( i, j );

			If( sampleBoundary( pos ).greaterThan( 0 ), () => {

				const tracedPos = trace( pos );
				output.data( i, j ).assign(
					collocatedCubicValueAtPosition2( input.data, input.gridSpacing, input.dataOrigin, tracedPos, input.dataSize )
				);

			} );

		} );

		if ( order === 1 ) return dispatchOrder1;

		// order 2 (MacCormack) -- see createSemiLagrangianAdvectionSolver2's
		// own header comment for the full algorithm. fwd/bwd: scratch
		// fields scoped to this one advectScalar2(input, output) call,
		// matching this port's established "fields bound at construction/
		// call time" convention (same as e.g. grid_smoke_solver2.js's own
		// ping-pong scratch fields).
		const fwd = tsl_array_n.arrayN( 'float', input.dataSize );
		const bwd = tsl_array_n.arrayN( 'float', input.dataSize );

		const dispatchForward = tsl_array_n.kernel( input.dataSize, ( i, j ) => {

			const pos = input.dataPosition( i, j );

			If( sampleBoundary( pos ).greaterThan( 0 ), () => {

				const tracedPos = trace( pos );
				fwd( i, j ).assign( collocatedCubicValueAtPosition2( input.data, input.gridSpacing, input.dataOrigin, tracedPos, input.dataSize ) );

			} ).Else( () => {

				fwd( i, j ).assign( input.data( i, j ) );

			} );

		} );

		// Traces *forward* (direction=-1) from this cell's own position,
		// sampling fwd (not input) -- mantaflow's own "bwd <- SemiLagrange
		// (fwd, -dt)". If fwd/trace were error-free, bwd would equal input
		// exactly; the difference is this step's own error estimate.
		const dispatchBackward = tsl_array_n.kernel( input.dataSize, ( i, j ) => {

			const pos = input.dataPosition( i, j );

			If( sampleBoundary( pos ).greaterThan( 0 ), () => {

				const tracedPos = trace( pos, -1 );
				bwd( i, j ).assign( collocatedCubicValueAtPosition2( fwd, input.gridSpacing, input.dataOrigin, tracedPos, input.dataSize ) );

			} ).Else( () => {

				bwd( i, j ).assign( fwd( i, j ) );

			} );

		} );

		const dispatchCorrectAndClamp = tsl_array_n.kernel( input.dataSize, ( i, j ) => {

			const pos = input.dataPosition( i, j );

			If( sampleBoundary( pos ).greaterThan( 0 ), () => {

				const tracedPos = trace( pos ); // recomputed -- the exact same position dispatchForward already sampled fwd at
				const fwdValue = fwd( i, j );
				const correctedValue = fwdValue.add( input.data( i, j ).sub( bwd( i, j ) ).mul( 0.5 ) );
				const clamped = clampCorrection( correctedValue, fwdValue, input.data, input.dataOrigin, input.gridSpacing, input.dataSize, input.dataPosition, tracedPos );

				output.data( i, j ).assign( clamped );

			} );

		} );

		return function dispatch() {

			dispatchForward();
			dispatchBackward();
			dispatchCorrectAndClamp();

		};

	}

	return { advectFaceCentered2, advectScalar2, trace };

}
