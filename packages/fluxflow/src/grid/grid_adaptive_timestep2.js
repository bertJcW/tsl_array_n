// Grid-specific wiring for CFL-based adaptive time-stepping -- the only
// layer in this feature that knows about FaceCenteredGrid2. The
// solver-agnostic pieces this builds on (../linalg/reduction.js's
// createMaxAbsReducer, ../time/cfl.js's computeAdaptiveSubSteps) have no
// grid-specific knowledge at all; a future FLIP solver would write its own
// much shorter analogous wiring, reusing those same two, with its own
// particle-velocity reduction in place of createMaxAbsReducer's grid-field
// one. See ../../docs/perf-investigation-cg-gpu-resident-alpha-beta.md for
// why this was built (the user's own follow-up question after that
// investigation) and jet/fluid-engine-dev's PhysicsAnimation/
// GridFluidSolver2 (../../THIRD-PARTY-NOTICES.md) for the reference this
// mirrors.
//
// *** dt MUST be a live node, not a plain number ***
//
// grid_solver2.js's onAdvanceTimeStep(timeStepInSeconds) passes its own
// argument through to each stage, but every one of its default stage
// functions (defaultComputeExternalForces etc.) takes zero parameters and
// ignores it -- the dt actually baked into the force/advection kernels is
// whatever createGridSolver2({ dt }) captured once, at *construction*
// time (external_force_solver2.js/advection_solver2.js both do
// `typeof dt === 'number' ? float(dt) : dt`, then close over that node
// inside their own kernel-building callbacks). So calling
// onAdvanceTimeStep(newDt) with a fresh number every frame would silently
// do nothing for the built-in stages -- adaptive dt only works if `dt`
// was passed as a live tsl_array_n.array0('float') node (the *same*
// instance also given to createGridSolver2({ dt })), updated via
// dt.fromArray() before each substep, which is exactly what update()
// below does. Checked explicitly and thrown on below, rather than left as
// a silent no-op -- this exact "reads live, but only if it's a node"
// convention was already anticipated by both those files' own header
// comments, just never previously wired up to anything that actually
// changes dt frame to frame.

import { createMaxAbsReducer } from '../linalg/reduction.js';
import { computeAdaptiveSubSteps } from '../time/cfl.js';

// velocityGrid: the same FaceCenteredGrid2 given to createGridSolver2.
// gridSpacing: plain-number array (same convention/duplication as
// createGridSolver2's own options.gridSpacing -- see that file's header
// comment for why the plain numbers can't be recovered from the grid
// itself).
// dt: required live tsl_array_n.array0('float') node, shared with
// createGridSolver2({ dt }) -- see this file's own header comment above.
// targetDt: the nominal per-frame time budget (e.g. 1/30) this divides
// into N equal substeps. Kept as a constructor-time option, matching how
// dt itself is already a fixed-at-construction convention throughout this
// port (external_force_solver2.js/advection_solver2.js), not a per-call
// argument to update().
// courantNumber/maxSubSteps: forwarded to computeAdaptiveSubSteps.
// atomicScale: forwarded to createMaxAbsReducer.
export function createGridAdaptiveTimeStep2( { velocityGrid, gridSpacing, dt, targetDt, courantNumber, maxSubSteps, atomicScale } ) {

	if ( typeof dt === 'number' ) {

		throw new Error( 'createGridAdaptiveTimeStep2: options.dt must be a live array0(\'float\') node (the same instance also passed to createGridSolver2({ dt })), not a plain number -- see this file\'s own header comment for why a plain number silently would not work.' );

	}

	const reducer = createMaxAbsReducer( [ velocityGrid.dataU, velocityGrid.dataV ], { atomicScale } );
	const minGridSpacing = Math.min( ...gridSpacing );

	// Caller-visible diagnostics, matching this port's established
	// diagnostics.converged (grid_pressure_solver2.js) / state.residualSquared
	// (linalg.js) convention -- e.g. for a live on-screen substep-count
	// readout, the same spirit as examples/16-karman-vortex-street/'s own
	// #perf overlay.
	const state = { lastNumSubSteps: 1, lastSubDt: targetDt };

	// Computes this frame's substep count from the velocity field's
	// *current* state (one GPU reduction + one readback), writes the
	// resulting per-substep dt into the shared `dt` node, and returns the
	// substep count for the caller's own loop:
	//   const numSubSteps = await adaptiveTimeStep.update();
	//   for ( let i = 0; i < numSubSteps; i ++ ) await solver.onAdvanceTimeStep();
	//
	// Deliberate simplification versus jet's own PhysicsAnimation::
	// advanceTimeStep (physics_animation.cpp): jet recomputes CFL fresh
	// before *every* substep, using whatever `remainingTime` is left after
	// each one (cheap on its own synchronous, CPU-side simulation). This
	// computes numSubSteps *once* per update() call and takes that many
	// *equal* substeps instead -- re-checking every substep here would mean
	// one GPU readback per substep rather than one per frame, and this
	// project's own CG performance investigation (../../docs/perf-
	// investigation-cg-gpu-resident-alpha-beta.md) found GPU/CPU
	// synchronization points to be a real, non-trivial cost on real
	// hardware -- so this tradeoff is deliberate, not an oversight.
	async function update() {

		const maxVel = await reducer.read();
		const { numSubSteps, subDt } = computeAdaptiveSubSteps( maxVel, minGridSpacing, targetDt, { courantNumber, maxSubSteps } );

		dt.fromArray( new Float32Array( [ subDt ] ) );

		state.lastNumSubSteps = numSubSteps;
		state.lastSubDt = subDt;

		return numSubSteps;

	}

	return { update, state };

}
