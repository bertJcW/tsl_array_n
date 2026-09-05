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
import { collocatedCubicValueAtPosition2, faceCenteredCubicValueAtPosition2 } from './grid_math.js';

const EPSILON = 1e-6;

// sampleVelocity/sampleBoundary: (pos) => node, closures bound by the
// factory below. dt: a node (see createSemiLagrangianAdvectionSolver2's
// own comment on why this isn't a plain JS number here).
function backTrace( sampleVelocity, sampleBoundary, startPos, dt, h, maxSubsteps ) {

	const pt = startPos.toVar();
	const remainingT = dt.toVar();

	Loop( maxSubsteps, () => {

		If( remainingT.lessThanEqual( EPSILON ), () => {

			Break();

		} );

		const vel0 = sampleVelocity( pt );
		const numSubSteps = max( ceil( length( vel0 ).mul( remainingT ).div( h ) ), 1 );
		const subDt = remainingT.div( numSubSteps );

		const midPt = pt.sub( vel0.mul( subDt.mul( 0.5 ) ) );
		const midVel = sampleVelocity( midPt );
		const nextPt = pt.sub( midVel.mul( subDt ) );

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
// kernel-build time -- fine for a fixed-dt test) or a node such as an
// array0('float')'s own callable reference (kept live: update the
// array0's contents via fromArray() between dispatches and every
// already-built kernel here picks up the new value on its next dispatch,
// the same pattern linalg.js's alpha/beta scalars already rely on) --
// this factory itself only converts a plain number to a node once, so a
// real CFL-adaptive solver can share one dt field across every stage that
// needs it without rebuilding any kernel here.
// options.maxSubsteps: cap on backTrace's adaptive substep loop, default 32.
export function createSemiLagrangianAdvectionSolver2( { velocityGrid, collider, dt, maxSubsteps = 32 } ) {

	const dtNode = typeof dt === 'number' ? float( dt ) : dt;
	const h = min( velocityGrid.gridSpacing.x, velocityGrid.gridSpacing.y );

	function sampleVelocity( pos ) {

		return velocityGrid.sample( pos );

	}

	function sampleBoundary( pos ) {

		return collider ? collider.sample( pos ) : float( 1 );

	}

	function trace( startPos ) {

		return backTrace( sampleVelocity, sampleBoundary, startPos, dtNode, h, maxSubsteps );

	}

	// input, output: FaceCenteredGrid2 (grid_data2.js) -- typically the
	// same velocity grid for self-advection, but any matching-shape pair
	// works (e.g. advecting a separate face-centered field through this
	// solver's own velocityGrid).
	function advectFaceCentered2( input, output ) {

		const dispatchU = tsl_array_n.kernel( input.dataSizeU, ( i, j ) => {

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

		const dispatchV = tsl_array_n.kernel( input.dataSizeV, ( i, j ) => {

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

		return function dispatch() {

			dispatchU();
			dispatchV();

		};

	}

	// input, output: ScalarGrid2 (grid_data2.js) -- e.g. density/
	// temperature, advected through this solver's own velocityGrid.
	function advectScalar2( input, output ) {

		return tsl_array_n.kernel( input.dataSize, ( i, j ) => {

			const pos = input.dataPosition( i, j );

			If( sampleBoundary( pos ).greaterThan( 0 ), () => {

				const tracedPos = trace( pos );
				output.data( i, j ).assign(
					collocatedCubicValueAtPosition2( input.data, input.gridSpacing, input.dataOrigin, tracedPos, input.dataSize )
				);

			} );

		} );

	}

	return { advectFaceCentered2, advectScalar2 };

}
