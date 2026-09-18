// 3D counterpart of velocity_damping2.js -- same uniform per-frame decay
// "viscosity" stand-in, one more staggered component. See that file's own
// header comment for the full "why a uniform decay, not real diffusion"
// reasoning, which is entirely dimension-independent.

import { createDecayKernel3 } from './array_utils3.js';

// options.velocityGrid: a FaceCenteredGrid3 (grid_data3.js).
// options.dampingCoefficient: see velocity_damping2.js's own header
// comment -- unchanged convention.
export function createVelocityDamping3( { velocityGrid, dampingCoefficient = 0 } ) {

	const decayU = createDecayKernel3( velocityGrid.dataU, dampingCoefficient );
	const decayV = createDecayKernel3( velocityGrid.dataV, dampingCoefficient );
	const decayW = createDecayKernel3( velocityGrid.dataW, dampingCoefficient );

	function applyDamping() {

		decayU();
		decayV();
		decayW();

	}

	return { applyDamping };

}
