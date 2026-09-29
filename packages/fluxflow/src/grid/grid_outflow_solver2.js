// New file, no Python source -- the other two-thirds of what an outflow
// object (sdf_inflow_outflow2.js's createSDFOutflow2) actually does once
// dispatched every frame. sdf_inflow_outflow2.js's own
// createOutflowPressureDirichlet2 covers part 1 (the pressure-side zero-
// pressure ghost boundary, needing zero pressure-solver changes); this
// file covers parts 2 and 3, both read directly from mantaflow
// (https://github.com/thunil/mantaflow, Apache License 2.0, Tobias Pfaff
// & Nils Thuerey -- see THIRD-PARTY-NOTICES.md for the full attribution).
// A proper small solver in its own right (not a one-off helper), parallel
// in spirit to grid_pressure_solver2.js/grid_blocked_boundary_condition_
// solver2.js.
//
// *** Part 2: convective (radiation) outflow velocity boundary condition ***
//
// Confirmed by reading mantaflow's own pressure.cpp directly: outflow
// cells are explicitly EXCLUDED from the normal pressure-based velocity
// correction ("don't change velocities in outflow cells") -- so mantaflow
// does NOT hard-clamp velocity to 0 (or to any fixed value) at an
// outflow. Instead, advection.cpp's applyOutflowBC (getBulkVel +
// extrapolateVelConvectiveBC + copyChangedVels) implements a real, named
// CFD technique (an Orlanski-style convective/radiation boundary
// condition): average a local "bulk velocity" from nearby fluid cells,
// then extrapolate the outflow cell's own velocity using
// `(vel - velPrev) / factor + vel(upstream neighbor)`, where
// `factor = timeStep * max(1, bulkVel[component])` -- this lets velocity
// *structures* (vortices etc.) pass smoothly out through the boundary
// instead of reflecting back in, without ever hard-clamping the value.
//
// Adapted here from mantaflow's own axis-aligned, per-cell-index search
// (which only really works because mantaflow's own outflow regions are
// always literal walls) to this port's SDF-based, arbitrarily-oriented
// outflow objects: the "upstream" direction is read directly from the
// outflow SDF's own gradient (sdf_collider2.js's existing
// `gradient(pos)`, the exact same technique grid_blocked_boundary_
// condition_solver2.js's own noFluxProjectionU/V already use for collider
// normals) rather than a fixed compass direction, and both the "bulk
// velocity" and the "upstream neighbor's velocity" are read via
// FaceCenteredGrid2's own existing bilinear `.sample(pos)` (grid_data2.js)
// rather than mantaflow's own discrete stencil averaging -- a natural
// fit given this port already has continuous sampling infrastructure
// mantaflow's own per-cell-only design didn't need. This also means an
// outflow object placed anywhere (not just hugging an axis-aligned wall)
// gets a *correct* upstream direction, not just a wall-only special case.
//
// *** A real long-run instability this file's own outflow mechanism first
// surfaced, root-caused and fixed elsewhere -- history kept here since two
// hypotheses specific to this file's own code were seriously investigated
// and ruled out along the way ***
//
// While building examples/16-karman-vortex-street/ (a longer, more
// asymmetric domain than example 15's own), real-hardware testing turned
// up a divergence that starts specifically along the outflow strip's own
// inner (fluid-facing) column, eventually reaching float32's own overflow
// range. Two plausible root causes specific to this file were each
// seriously investigated and BOTH RULED OUT by direct experiment, not
// just reasoning:
// 1. Suspected `upstreamPt = pt.sub(n.mul(spacing))` had its sign
//    backwards (a plain 1D hand derivation does suggest `n` -- the
//    outflow SDF's own gradient -- points toward the fluid, so `.add(...)`
//    looked more correct than `.sub(...)`). Tried flipping it to
//    `.add(...)`: example 16 diverged *later* but not never, and example
//    15 -- previously confirmed stable over 1500+ frames -- started
//    diverging within single-digit frames instead. Reverted; the sign
//    shown below (`.sub(...)`) is confirmed, not just inherited, to be the
//    one that doesn't make things worse.
// 2. Suspected the outflow strip's own inner-boundary x-coordinate being
//    exactly integer-aligned with a U-face's own query position (e.g.
//    x=N-2 exactly) could make the SDF sample there straddle 0 and let
//    isInsideSdf's strict `phi<0` check flicker between true/false across
//    frames. Tried offsetting the boundary to a non-integer x (e.g.
//    N-2.5): example 15 diverged *sooner*, not later or never. Ruled out.
//
// The actual root cause turned out to be entirely outside this file: an
// asymmetric red-black relaxation schedule in multigrid.js's own
// createMultigridPreconditioner (pre-smoothing and post-smoothing used
// the same color order instead of reversed ones, breaking the standard
// requirement that a multigrid V-cycle used as a PCG preconditioner must
// itself be a symmetric operator) -- found by comparing against
// mantaflow's own multigrid solver, per the user's own suggestion to
// consult it. This mechanism (an outflow's own Dirichlet-pressure region)
// was simply the first place a Dirichlet mask got exercised on real
// hardware over a long enough run to expose it -- see multigrid.js's own
// header comment for the full investigation, and linalg.js's own beta/
// alpha robustness guards plus grid_pressure_solver2.js's/grid_blocked_
// boundary_condition_solver2.js's own last-resort circuit breakers (for
// pressure and velocity respectively) for the additional defense-in-depth
// added on top of the root-cause fix. Confirmed via multi-thousand-frame
// real-hardware runs on both example 15 and 16: this specific long-run
// divergence no longer occurs.
//
// *** A real, initially-missed porting detail: `factor`'s own `timeStep`
// is NOT the simulation's own dt -- confirmed by reading mantaflow's
// advection.cpp a second time, directly at the exact call site *** --
// `applyOutflowBC` (the function that actually calls
// `extrapolateVelConvectiveBC`) does not pass its own `timeStep` argument
// straight through -- it passes `max(1.0, timeStep*4)` instead. For any
// simulation dt below 0.25 (true of essentially every reasonable CFL-
// bounded timestep, this port's own examples included, dt=1/30 here),
// that max() always wins, meaning the "timeStep" `factor` actually uses
// is a *constant* 1.0, completely decoupled from the simulation's real
// dt. An earlier version of this file used the real `dt` directly here --
// a real, confirmed-on-real-hardware bug, not just the SDF-gradient
// degeneracy bug fixed separately in grid_math.js's own
// bilinearGradientAtPosition2 (see that function's own header comment):
// `factor = dt * max(1, bulkVel)` with dt itself only ~0.033 (a typical
// 1/30 timestep) amplifies `(vel - velPrev)` by roughly 1/0.033 ~= 30x
// before adding the upstream sample -- any ordinary frame-to-frame
// advection noise gets blown up by that same ~30x every single frame,
// which is more than enough on its own to diverge even with a perfectly
// correct upstream direction. Confirmed on real hardware: even after the
// SDF-gradient fix (which independently matters and is also real), this
// scene still reliably diverged within several seconds with the original
// dt-based factor; switching to the mantaflow-literal
// `max(1, dt*4)` here resolved it completely, matching mantaflow's own
// safety margin exactly rather than reinventing a different one.

//
// Writes into a scratch destination array before copying back (uDst/vDst
// + a per-outflow copy step) -- the exact same read/write-race
// precaution grid_solver2.js already uses for self-advection (sampling a
// field at arbitrary positions while also writing it in the same
// dispatch has no cross-thread ordering guarantee on a GPU), not a new
// pattern here, reusing an already-proven one.
//
// *** Part 2 is now the standard convective form, and the upstream
// direction was backwards here too ***
//
// Everything above about mantaflow's `(vel - velPrev)/factor +
// vel(upstream)` and about `factor`'s own timeStep is the history of what
// used to be here, and is kept because two of its findings are still
// load-bearing (the SDF-gradient degeneracy fix, and the multigrid
// asymmetry that the outflow strip was merely the first place to expose).
// The formula itself is gone. grid_outflow_solver3.js's own header carries
// the derivation; the short version is that sandbox/outflow-gradient/
// measures this package's outflow SDFs as having a gradient that points
// TOWARD the fluid -- in 2D exactly as in 3D -- so `pt - n * h` sampled
// downstream of the face, and at a domain-edge outflow two cells deep the
// two exit faces sampled each other while neither tracked the interior.
//
// This file kept the old formula for a while after 3D changed, on the
// grounds that examples 15 and 16 were long-run stable and a boundary
// condition should not be changed under them without its own verification.
// That verification is what changed the answer. Measured with
// solver_health.mjs, which judges conservation rather than convergence,
// example 16 on the old formula reads BROKEN from frame 1720 and settles
// at 13.89x the inflow leaving the outlet with a peak speed of 27.79
// against an inflow of 2 -- while converging on every frame throughout,
// which is why every earlier long run of it looked fine.
//
// *** Part 3: scalar-field cleanup ***
//
// mantaflow's own resetOutflow (extforces.cpp) clears the fluid flag,
// resets the level-set phi, kills particles, and zeros a generic
// caller-supplied scalar grid (typically density) within outflow cells
// every step -- this is the piece that actually corresponds to "the
// fluid disappears here", confirmed directly with the user after an
// initial (wrong) guess that the *pressure* treatment alone meant this.
// This port has no particles/level-set (a pure Eulerian grid method), so
// only the scalar-field-zeroing concept carries over, generalized:
// clearOutflowScalarField(scalarGrid) takes *any* caller-supplied
// ScalarGrid2 (dye, in every existing example -- never a library-owned
// concept in this port) rather than assuming one hardcoded density field.

import * as tsl_array_n from 'tsl_array_n';
import { float, max, clamp } from 'three/tsl';
import { isInsideSdf } from './level_set_utils.js';
import { createOutflowPressureDirichlet2 } from './sdf_inflow_outflow2.js';

// A defensive backstop on the extrapolated velocity below, unrelated to
// any particular scene's own expected velocity scale -- confirmed
// necessary on real WebGPU hardware: an early version of this file (with
// no such guard) combined with a caller-side bug in how an outflow's own
// polygon was padded (see examples/15-flow-past-cylinder/'s own header
// comment for the full mechanism -- two adjacent faces near the outer
// edge could end up sampling *each other* as their own "upstream"
// reference instead of both pointing back toward the true fluid
// interior) produced a genuine, compounding numerical feedback loop that
// reached ~1e34 within the very first frame. The padding bug is the real
// fix (this guard alone wouldn't have produced correct *physics*, just
// stopped it from reaching float32's own overflow range) -- this constant
// is a second, independent layer, the same spirit as this port's own
// isDegenerateDenominator guard in linalg.js: a value this method should never
// legitimately produce is far more likely to be a numerical runaway than
// a real answer, so clamp rather than let it propagate into next frame's
// pressure divergence and corrupt the entire coupled solve.
const EXTRAPOLATED_VELOCITY_CLAMP = 1e3;

// options.velocityGrid: the FaceCenteredGrid2 (grid_data2.js) this solver
// reads/corrects.
// options.velocityPrev: the *same* scratch grid grid_solver2.js already
// maintains for self-advection (copied into every frame via
// array_utils.js's createCopyKernel2, immediately before advecting) --
// reused directly, not duplicated, since it's already exactly "velocity
// before this frame's advection/outflow treatment", matching mantaflow's
// own velPrev meaning precisely.
// options.outflows: one createSDFOutflow2(...) object, or an array of them.
// options.dt: plain number or live node -- same "number or node"
// convention already used for dt elsewhere in this port.
// options.applyVelocityBC: default true -- set false to skip part 2 (the
// convective velocity extrapolation) entirely, keeping only part 1 (the
// pressure-side zero-pressure ghost boundary, always active regardless).
// A real, legitimate configuration knob (a caller may only want the
// pressure-side effect), and also currently in active use as a diagnostic
// tool: isolating whether a real-hardware pressure blowup traces to this
// velocity treatment specifically, versus the pressure/dirichlet-mask
// mechanism itself -- see examples/15-flow-past-cylinder/'s own toggles.
//
// Returns { dirichlet, applyOutflowVelocityBC, clearOutflowScalarField }.
// dirichlet: part 1, re-exposed here for convenience so a caller only
// needs to hold one object for all three parts of what an outflow does.
// applyOutflowVelocityBC(): part 2, called once per frame (see
// grid_solver2.js's own wiring for exactly where).
// clearOutflowScalarField(scalarGrid): part 3, a factory -- call once per
// scalar field to build its own dispatcher, matching this port's
// established "build once, dispatch repeatedly" convention (e.g.
// advectScalar2's own shape) rather than rebuilding a kernel every frame.
export function createGridOutflowSolver2( { velocityGrid, velocityPrev, outflows, dt, applyVelocityBC = true } ) {

	const list = Array.isArray( outflows ) ? outflows : [ outflows ];
	const dtNode = typeof dt === 'number' ? float( dt ) : dt;

	// OUTFLOW_TIMESTEP_FLOOR/SCALE are gone -- they belonged to the
	// `(vel - velPrev)/factor + vel(upstream)` form this file used to carry, and
	// that whole formula was replaced by the standard convective one (see this
	// file's own header). What survives is mantaflow's reason for having had a
	// floor at all: its applyOutflowBC passes max(1.0, dt*4) rather than the raw
	// dt into extrapolateVelConvectiveBC's `factor`. Kept as a
	// plain JS function (not a node computed once up front) so `dtNode.mul`
	// is only ever evaluated lazily, inside a kernel's own callback body --
	// matching every other per-cell computation in this file, and,
	// importantly, matching this port's existing test convention of
	// constructing a solver without a real `dt` (dt is only otherwise
	// needed once a kernel is actually dispatched, never at construction).
	const uDst = tsl_array_n.arrayN( 'float', velocityGrid.dataSizeU );
	const vDst = tsl_array_n.arrayN( 'float', velocityGrid.dataSizeV );

	// One (extrapolateU, copyU, extrapolateV, copyV) quartet per outflow
	// object, all sharing the same uDst/vDst scratch -- safe as long as
	// each outflow's own extrapolate/copy pair runs back-to-back (see
	// applyOutflowVelocityBC below): by the time the *next* outflow
	// object's extrapolateU runs, velocityGrid.dataU already reflects
	// whatever the previous outflow object just wrote via its own copyU,
	// so multiple (non-overlapping) outflow objects compose correctly
	// without clobbering each other.
	function buildOutflowVelocityKernels( outflow ) {

		const extrapolateU = tsl_array_n.kernel( velocityGrid.dataSizeU, ( i, j ) => {

			const pt = velocityGrid.uPosition( i, j );

			tsl_array_n.If( isInsideSdf( outflow.sample( pt ) ), () => {

				const g = outflow.gradient( pt );

				tsl_array_n.If( g.length().greaterThan( 0 ), () => {

					const n = g.normalize();
					// `pt + n * h`, not `pt - n * h` -- see this file's own
					// header comment. The gradient points toward the fluid, so
					// the upstream neighbour is the other way round from what
					// was here.
					const upstreamPt = pt.add( n.mul( velocityGrid.gridSpacing.x ) );

					const bulkVel = velocityGrid.sample( pt );
					const outwardSpeed = max( float( 0 ), bulkVel.dot( n.negate() ) );
					const courant = clamp( outwardSpeed.mul( dtNode ).div( velocityGrid.gridSpacing.x ), float( 0 ), float( 1 ) );

					const upstreamVel = velocityGrid.sample( upstreamPt );
					const current = velocityGrid.dataU( i, j );

					const relaxed = current.add( upstreamVel.x.sub( current ).mul( courant ) );
					uDst( i, j ).assign( clamp( relaxed, - EXTRAPOLATED_VELOCITY_CLAMP, EXTRAPOLATED_VELOCITY_CLAMP ) );

				} ).Else( () => {

					// No usable gradient (a degenerate/flat SDF region) --
					// leave this face's velocity as it already is.
					uDst( i, j ).assign( velocityGrid.dataU( i, j ) );

				} );

			} );

		} );

		const copyU = tsl_array_n.kernel( velocityGrid.dataSizeU, ( i, j ) => {

			const pt = velocityGrid.uPosition( i, j );

			tsl_array_n.If( isInsideSdf( outflow.sample( pt ) ), () => {

				velocityGrid.dataU( i, j ).assign( uDst( i, j ) );

			} );

		} );

		const extrapolateV = tsl_array_n.kernel( velocityGrid.dataSizeV, ( i, j ) => {

			const pt = velocityGrid.vPosition( i, j );

			tsl_array_n.If( isInsideSdf( outflow.sample( pt ) ), () => {

				const g = outflow.gradient( pt );

				tsl_array_n.If( g.length().greaterThan( 0 ), () => {

					const n = g.normalize();
					// `pt + n * h`, not `pt - n * h` -- see this file's own
					// header comment. The gradient points toward the fluid, so
					// the upstream neighbour is the other way round from what
					// was here.
					const upstreamPt = pt.add( n.mul( velocityGrid.gridSpacing.y ) );

					const bulkVel = velocityGrid.sample( pt );
					const outwardSpeed = max( float( 0 ), bulkVel.dot( n.negate() ) );
					const courant = clamp( outwardSpeed.mul( dtNode ).div( velocityGrid.gridSpacing.y ), float( 0 ), float( 1 ) );

					const upstreamVel = velocityGrid.sample( upstreamPt );
					const current = velocityGrid.dataV( i, j );

					const relaxed = current.add( upstreamVel.y.sub( current ).mul( courant ) );
					vDst( i, j ).assign( clamp( relaxed, - EXTRAPOLATED_VELOCITY_CLAMP, EXTRAPOLATED_VELOCITY_CLAMP ) );

				} ).Else( () => {

					vDst( i, j ).assign( velocityGrid.dataV( i, j ) );

				} );

			} );

		} );

		const copyV = tsl_array_n.kernel( velocityGrid.dataSizeV, ( i, j ) => {

			const pt = velocityGrid.vPosition( i, j );

			tsl_array_n.If( isInsideSdf( outflow.sample( pt ) ), () => {

				velocityGrid.dataV( i, j ).assign( vDst( i, j ) );

			} );

		} );

		return { extrapolateU, copyU, extrapolateV, copyV };

	}

	const velocityKernels = list.map( buildOutflowVelocityKernels );

	function applyOutflowVelocityBC() {

		if ( ! applyVelocityBC ) return;

		for ( const { extrapolateU, copyU, extrapolateV, copyV } of velocityKernels ) {

			extrapolateU();
			copyU();
			extrapolateV();
			copyV();

		}

	}

	function clearOutflowScalarField( scalarGrid ) {

		const dispatchers = list.map( ( outflow ) => tsl_array_n.kernel( scalarGrid.dataSize, ( i, j ) => {

			const pt = scalarGrid.dataPosition( i, j );

			tsl_array_n.If( isInsideSdf( outflow.sample( pt ) ), () => {

				scalarGrid.data( i, j ).assign( 0 );

			} );

		} ) );

		return function clear() {

			for ( const dispatch of dispatchers ) dispatch();

		};

	}

	return {
		dirichlet: createOutflowPressureDirichlet2( list ),
		applyOutflowVelocityBC,
		clearOutflowScalarField
	};

}
