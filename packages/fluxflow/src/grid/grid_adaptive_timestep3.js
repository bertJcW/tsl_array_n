// 3D counterpart of grid_adaptive_timestep2.js -- the reduction
// (../linalg/reduction.js) and CFL math (../time/cfl.js) this builds on
// are both already dimension-generic (see reduction.js's own header
// comment: one dispatch per field, whatever shape each has), so the only
// change is reducing over dataW too. See grid_adaptive_timestep2.js's own
// header comment for the full "dt must be a live node" reasoning, which
// carries over unchanged.

import { createMaxAbsReducer } from '../linalg/reduction.js';
import { computeAdaptiveSubSteps } from '../time/cfl.js';

// velocityGrid: the same FaceCenteredGrid3 given to createGridSolver3.
// gridSpacing/dt/targetDt/courantNumber/maxSubSteps/atomicScale: see
// grid_adaptive_timestep2.js's own header comment -- unchanged conventions.
export function createGridAdaptiveTimeStep3( { velocityGrid, gridSpacing, dt, targetDt, courantNumber, maxSubSteps, atomicScale } ) {

	if ( typeof dt === 'number' ) {

		throw new Error( 'createGridAdaptiveTimeStep3: options.dt must be a live array0(\'float\') node (the same instance also passed to createGridSolver3({ dt })), not a plain number -- see this file\'s own header comment for why a plain number silently would not work.' );

	}

	const reducer = createMaxAbsReducer( [ velocityGrid.dataU, velocityGrid.dataV, velocityGrid.dataW ], { atomicScale } );
	const minGridSpacing = Math.min( ...gridSpacing );

	const state = { lastNumSubSteps: 1, lastSubDt: targetDt };

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
