// Concrete orchestrator, replacing the earlier pure hooks-forwarder ported
// from grid_solver2.py's abstract base class (all methods `pass`, no
// concrete solver -- that placeholder had "limited value on its own," per
// its own prior header comment, and had no external consumers to keep
// compatible). This now builds and wires real advection/forces/pressure
// solvers from one set of options, in jet/fluid-engine-dev's own
// established stage order (GridFluidSolver2::onAdvanceTimeStep): external
// forces -> viscosity -> pressure -> advection. Viscosity stays a no-op --
// explicitly deferred by the user, not built yet.
//
// onAdvanceTimeStep is now `async` (pressure's CG solve needs an `await`,
// unlike every other stage here).
//
// *** Self-advection needs a scratch "previous velocity" grid, not
// in-place advection of velocityGrid onto itself ***
//
// Confirmed by reading jet's own GridFluidSolver2::computeAdvection
// directly (grid_fluid_solver2.cpp): it always clones velocity first
// (`vel->clone()`), passing the clone as *both* the sampled flow field and
// the thing advected, with the original grid only ever as the output. This
// isn't a style choice -- advectFaceCentered2 samples its input field at
// arbitrary back-traced positions per invocation; if input and output were
// the same buffer, different GPU threads would race on reads/writes with
// no ordering guarantee (a real hazard on any backend, unrelated to this
// project's WebGL2-fallback-only "single writer per field" quirk found
// while building examples/12-interactive-advection/ -- that one was about
// *multiple kernel objects* writing one field; this is a same-kernel,
// same-dispatch read/write race). So this file keeps a scratch
// `velocityPrev` grid, copied from `velocityGrid` (array_utils.js's
// existing createCopyKernel2) immediately before every advect dispatch.
//
// By contrast, grid_pressure_solver2.js's own correction step *is* safe in
// place (self-touch on velocity, cross-*field* read of the separate
// pressure grid, no neighbor-velocity read) -- see that file's own
// project() for why -- so this file uses velocityGrid for both sides of
// that one.
//
// *** A real, confirmed-on-real-hardware bug, found after this file first
// shipped: the domain-boundary velocity constraint was completely
// missing ***
//
// The user reported examples/14-smoke-demo/ "not working well" on real
// WebGPU hardware -- a screenshot showed noisy, scattered dye instead of
// a smooth rising plume, plus a bright artifact along one domain edge.
// Root-caused by reading jet's own GridFluidSolver2::onAdvanceTimeStep
// directly (grid_fluid_solver2.cpp): it calls applyBoundaryCondition()
// (which calls _boundaryConditionSolver->constrainVelocity()) after
// *every* stage -- computeGravity, computeViscosity, computePressure,
// AND computeAdvection -- not just once at the end. This project already
// has that exact mechanism built and confirmed on real hardware
// (grid_blocked_boundary_condition_solver2.js, exercised live by
// examples/01-boundary-condition/), but this orchestrator never called
// it at all. Without it, velocity at the domain edges is completely
// unconstrained: forces/self-advection can push it to arbitrary values
// there with nothing ever resetting it back toward the correct
// zero-flux/closed-wall condition, and that unconstrained edge velocity
// then contaminates the whole domain over many frames via advection and
// an under-constrained pressure solve (which assumes the boundary
// condition upstream of it is already correct, not something it enforces
// itself). Fixed by constructing one createGridBlockedBoundaryConditionSolver2
// (no collider unless the caller supplies one) and calling
// constrainVelocity() from inside each default stage that actually
// changes velocity -- forces, pressure, advection -- matching jet's own
// placement exactly (inside each concrete compute* method, not the outer
// loop, so a caller-supplied stage *override* is responsible for its own
// boundary handling, the same as overriding a virtual method in jet's
// own C++ replaces its entire body).

import { createFaceCenteredGrid2 } from './grid_data2.js';
import { createCopyKernel2 } from './array_utils.js';
import { createExternalForceSolver2 } from './external_force_solver2.js';
import { createGridPressureSolver2 } from './grid_pressure_solver2.js';
import { createSemiLagrangianAdvectionSolver2 } from './advection_solver2.js';
import { createGridBlockedBoundaryConditionSolver2 } from './grid_blocked_boundary_condition_solver2.js';
import { createGridOutflowSolver2 } from './grid_outflow_solver2.js';
import { combineDirichlet } from './sdf_inflow_outflow2.js';

const NOOP = () => {};

// options.velocityGrid: required -- the FaceCenteredGrid2 (grid_data2.js)
// this solver evolves in place, frame to frame.
// options.gridSpacing/origin: plain-number arrays, must match the numbers
// already given to createFaceCenteredGrid2 when velocityGrid was built --
// FaceCenteredGrid2.gridSpacing is a TSL vec2 *node* with no way to read
// the original plain numbers back out of it, so this is a known, accepted
// duplication, not an oversight.
// options.force: optional (pos) => vec2, forwarded to
// createExternalForceSolver2 -- omit for no external forces at all.
// options.dirichlet: optional (pos) => { active, target }, forwarded to
// createGridPressureSolver2 -- omit for a pure zero-flux/Neumann domain.
// options.collider: optional SDFStaticCollider2/SDFRigidBodyCollider2
// (sdf_collider2.js), forwarded to createGridBlockedBoundaryConditionSolver2
// -- omit for an empty domain (the closed-domain-boundary constraint below
// still applies either way; a collider adds *interior* obstacles on top of
// that). Not forwarded to createSemiLagrangianAdvectionSolver2 automatically
// -- pass the same value via options.advection.collider too if advection's
// own boundary-crossing clamp should see it as well. Unlike every other
// option here, a *real* (non-null) collider makes this factory dispatch a
// kernel immediately at construction time (grid_blocked_boundary_condition_
// solver2.js's own setCollider() builds blockMarker via a real kernel call
// when a collider is present, versus a plain CPU-side .fromArray() when
// there's none) -- so createGridSolver2({..., collider}) needs
// tsl_array_n.init() to have already run, unlike a collider-less
// construction. Found while adding vitest coverage for this option.
// options.dt: shared across every sub-solver (a plain number or a live
// node, same flexible convention every solver in this port already uses).
// options.inflows: optional createSDFInflow2(...) (sdf_inflow_outflow2.js)
// or an array of them, forwarded to createGridBlockedBoundaryConditionSolver2.
// options.outflows: optional createSDFOutflow2(...) or an array of them --
// builds a createGridOutflowSolver2 (grid_outflow_solver2.js) internally,
// whose own pressure contribution is combined with options.dirichlet (see
// combineDirichlet, sdf_inflow_outflow2.js) and whose velocity/scalar-field
// steps are exposed on this factory's own return value as `outflowSolver`.
// options.closedDomainBoundaryFlag: optional, forwarded to boundarySolver's
// own mutable property of the same name -- omit to keep its DIRECTION_ALL
// default (every wall closed); e.g. exclude a wall an outflow object
// overlaps so it isn't also forced to zero velocity.
// options.advection: forwarded to createSemiLagrangianAdvectionSolver2
// (e.g. { collider, maxSubsteps }).
// options.pressure: forwarded to createGridPressureSolver2 (e.g.
// { multigrid, tolerance, maxIterations }).
// options.beginAdvanceTimeStep/endAdvanceTimeStep: optional extension
// hooks, jet's own onBeginAdvanceTimeStep/onEndAdvanceTimeStep equivalents
// -- e.g. moving a collider before the frame's physics runs.
// options.computeExternalForces/computeViscosity/computePressure/
// computeAdvection: optional full-stage overrides, replacing this file's
// own concrete default for that stage entirely (not called *alongside*
// it) -- computeViscosity has no built-in default at all (stays NOOP
// unless overridden), since viscosity isn't built yet.
export function createGridSolver2( {
	velocityGrid,
	gridSpacing = [ 1, 1 ],
	origin = [ 0, 0 ],
	force,
	dirichlet,
	collider,
	inflows,
	outflows,
	outflowVelocityBC = true, // forwarded to createGridOutflowSolver2's own applyVelocityBC -- see that file's own comment
	closedDomainBoundaryFlag,
	dt,
	advection = {},
	pressure = {},
	beginAdvanceTimeStep = NOOP,
	endAdvanceTimeStep = NOOP,
	computeExternalForces,
	computeViscosity,
	computePressure,
	computeAdvection
} = {} ) {

	if ( ! velocityGrid ) {

		throw new Error( 'createGridSolver2: options.velocityGrid is required.' );

	}

	const [ resolutionX, resolutionY ] = velocityGrid.resolution;
	const [ gridSpacingX, gridSpacingY ] = gridSpacing;
	const [ originX, originY ] = origin;

	const boundarySolver = createGridBlockedBoundaryConditionSolver2(
		velocityGrid, resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY, collider, inflows
	);

	if ( closedDomainBoundaryFlag !== undefined ) boundarySolver.closedDomainBoundaryFlag = closedDomainBoundaryFlag;

	const forceSolver = force ? createExternalForceSolver2( { velocityGrid, force, dt } ) : null;

	// Built before the pressure solver, for two reasons: (1) the outflow
	// solver's own velocity-extrapolation step (part 2, see
	// grid_outflow_solver2.js) needs velocityPrev to already exist, and
	// (2) its own pressure contribution needs to be combined into
	// `dirichlet` *before* createGridPressureSolver2 is constructed below.
	const velocityPrev = createFaceCenteredGrid2( resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY );
	const copyVelocityU = createCopyKernel2( velocityGrid.dataU, velocityPrev.dataU );
	const copyVelocityV = createCopyKernel2( velocityGrid.dataV, velocityPrev.dataV );

	const outflowSolver = outflows ? createGridOutflowSolver2( { velocityGrid, velocityPrev, outflows, dt, applyVelocityBC: outflowVelocityBC } ) : null;

	// Combines an outflow-derived dirichlet (if any) with a caller-supplied
	// one (e.g. examples/13-interactive-pressure/'s own pointer-driven
	// pressure vent) -- see combineDirichlet's own comment
	// (sdf_inflow_outflow2.js) for what happens if both mark the same cell.
	const combinedDirichlet = outflowSolver ? combineDirichlet( outflowSolver.dirichlet, dirichlet ) : dirichlet;

	const pressureSolver = createGridPressureSolver2( {
		resolution: velocityGrid.resolution, gridSpacing, origin, dirichlet: combinedDirichlet, ...pressure
	} );
	const projectDispatch = pressureSolver.project( velocityGrid, velocityGrid ); // in place -- see header comment

	const advectionSolver = createSemiLagrangianAdvectionSolver2( { velocityGrid: velocityPrev, dt, ...advection } );
	const advectDispatch = advectionSolver.advectFaceCentered2( velocityPrev, velocityGrid ); // clone-then-advect -- see header comment

	async function defaultComputeExternalForces() {

		if ( forceSolver ) {

			forceSolver.applyExternalForces();
			boundarySolver.constrainVelocity();

		}

	}

	function defaultComputeViscosity() {} // no-op -- viscosity is explicitly deferred, not built yet

	async function defaultComputePressure() {

		await projectDispatch();
		boundarySolver.constrainVelocity();

	}

	async function defaultComputeAdvection() {

		copyVelocityU();
		copyVelocityV();
		await advectDispatch();
		// Part 2 of what an outflow does (grid_outflow_solver2.js) -- run
		// right after advection, matching mantaflow's own placement
		// (applyOutflowBC runs right after advection in its own solve
		// loop), and needs velocityPrev (copied above, pre-advection) for
		// its own "before vs after" comparison.
		if ( outflowSolver ) outflowSolver.applyOutflowVelocityBC();
		boundarySolver.constrainVelocity();

	}

	async function onAdvanceTimeStep( timeStepInSeconds ) {

		beginAdvanceTimeStep( timeStepInSeconds );

		await ( computeExternalForces ?? defaultComputeExternalForces )( timeStepInSeconds );
		await ( computeViscosity ?? defaultComputeViscosity )( timeStepInSeconds );
		await ( computePressure ?? defaultComputePressure )( timeStepInSeconds );
		await ( computeAdvection ?? defaultComputeAdvection )( timeStepInSeconds );

		endAdvanceTimeStep( timeStepInSeconds );

	}

	return {
		onAdvanceTimeStep,
		velocityGrid, velocityPrev,
		pressure: pressureSolver.pressure,
		forceSolver, pressureSolver, advectionSolver, boundarySolver, outflowSolver
	};

}
