// 3D counterpart of grid_solver2.js -- same jet-derived orchestration
// (external forces -> viscosity[no-op] -> pressure -> advection, each
// stage's default calling boundarySolver.constrainVelocity() afterward)
// and the same clone-before-self-advection safety reasoning. See that
// file's own header comment for the full history (why velocityPrev exists,
// the real domain-boundary bug this stage ordering fixed) -- none of it
// dimension-specific.

import * as tsl_array_n from 'tsl_array_n';
import { createFaceCenteredGrid3 } from './grid_data3.js';
import { createCopyKernel3 } from './array_utils3.js';
import { createExternalForceSolver3 } from './external_force_solver3.js';
import { createGridPressureSolver3 } from './grid_pressure_solver3.js';
import { createSemiLagrangianAdvectionSolver3 } from './advection_solver3.js';
import { createGridBlockedBoundaryConditionSolver3 } from './grid_blocked_boundary_condition_solver3.js';
import { createGridOutflowSolver3 } from './grid_outflow_solver3.js';
import { combineDirichlet3 } from './sdf_inflow_outflow3.js';

const NOOP = () => {};

// options: see grid_solver2.js's own createGridSolver2 header comment for
// the full parameter list -- unchanged meanings, gridSpacing/origin are
// now triples.
export function createGridSolver3( {
	velocityGrid,
	gridSpacing = [ 1, 1, 1 ],
	origin = [ 0, 0, 0 ],
	force,
	dirichlet,
	collider,
	inflows,
	outflows,
	outflowVelocityBC = true,
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

		throw new Error( 'createGridSolver3: options.velocityGrid is required.' );

	}

	const [ resolutionX, resolutionY, resolutionZ ] = velocityGrid.resolution;
	const [ gridSpacingX, gridSpacingY, gridSpacingZ ] = gridSpacing;
	const [ originX, originY, originZ ] = origin;

	// A static collider's own fractional face occupancy, computed once. It
	// is what the pressure system bakes into its coefficients below, and the
	// boundary solver now reads the same fields to decide which faces are
	// solid -- see grid_blocked_boundary_condition_solver3.js's own
	// makeMarkAndProject comment for why two independent estimates of that
	// are not good enough.
	const colliderWeights = collider && ! pressure.colliderWeights
		? collider.computeFaceWeights( resolutionX, resolutionY, resolutionZ, gridSpacingX, gridSpacingY, gridSpacingZ, originX, originY, originZ )
		: undefined;

	const boundarySolver = createGridBlockedBoundaryConditionSolver3(
		velocityGrid, resolutionX, resolutionY, resolutionZ, gridSpacingX, gridSpacingY, gridSpacingZ, originX, originY, originZ, collider, inflows,
		colliderWeights ?? pressure.colliderWeights ?? null
	);

	if ( closedDomainBoundaryFlag !== undefined ) boundarySolver.closedDomainBoundaryFlag = closedDomainBoundaryFlag;

	const forceSolver = force ? createExternalForceSolver3( { velocityGrid, force, dt } ) : null;

	const velocityPrev = createFaceCenteredGrid3( resolutionX, resolutionY, resolutionZ, gridSpacingX, gridSpacingY, gridSpacingZ, originX, originY, originZ );
	const copyVelocityU = createCopyKernel3( velocityGrid.dataU, velocityPrev.dataU );
	const copyVelocityV = createCopyKernel3( velocityGrid.dataV, velocityPrev.dataV );
	const copyVelocityW = createCopyKernel3( velocityGrid.dataW, velocityPrev.dataW );
	const copyVelocity = tsl_array_n.createBatch( [ copyVelocityU, copyVelocityV, copyVelocityW ] );

	const outflowSolver = outflows ? createGridOutflowSolver3( { velocityGrid, velocityPrev, outflows, dt, applyVelocityBC: outflowVelocityBC } ) : null;

	const combinedDirichlet = outflowSolver ? combineDirichlet3( outflowSolver.dirichlet, dirichlet ) : dirichlet;

	const pressureSolver = createGridPressureSolver3( {
		resolution: velocityGrid.resolution, gridSpacing, origin, dirichlet: combinedDirichlet, colliderWeights, ...pressure
	} );
	const projectDispatch = pressureSolver.project( velocityGrid, velocityGrid );

	const advectionSolver = createSemiLagrangianAdvectionSolver3( { velocityGrid: velocityPrev, dt, ...advection } );
	const advectDispatch = advectionSolver.advectFaceCentered3( velocityPrev, velocityGrid );

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

		copyVelocity();
		await advectDispatch();
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
