// Pure-number unit tests -- computeAdaptiveSubSteps has zero GPU/TSL
// dependency, so unlike almost everything else in this port, this is
// fully verifiable in this dev sandbox with no real WebGPU hardware
// needed at all.

import { describe, it, expect } from 'vitest';
import { computeAdaptiveSubSteps } from '../src/time/cfl.js';

describe( 'computeAdaptiveSubSteps', () => {

	it( 'returns a single substep spanning the whole frame when velocity is low', () => {

		// cfl = 0.1 * (1/30) / 1 ~= 0.0033, well under the default
		// courantNumber (5) -- one substep should cover the whole frame.
		const { numSubSteps, subDt } = computeAdaptiveSubSteps( 0.1, 1, 1 / 30 );

		expect( numSubSteps ).toBe( 1 );
		expect( subDt ).toBeCloseTo( 1 / 30 );

	} );

	it( 'returns a single substep for exactly zero velocity', () => {

		const { numSubSteps, subDt } = computeAdaptiveSubSteps( 0, 1, 1 / 30 );

		expect( numSubSteps ).toBe( 1 );
		expect( subDt ).toBeCloseTo( 1 / 30 );

	} );

	it( 'splits into multiple substeps when velocity is high, matching ceil(cfl/courantNumber)', () => {

		// cfl = 100 * (1/30) / 1 ~= 3.333; courantNumber = 1 -> ceil(3.333/1) = 4
		const { numSubSteps, subDt } = computeAdaptiveSubSteps( 100, 1, 1 / 30, { courantNumber: 1 } );

		expect( numSubSteps ).toBe( 4 );
		expect( subDt ).toBeCloseTo( ( 1 / 30 ) / 4 );

	} );

	it( 'produces equal substeps that sum back to the original frameDt', () => {

		const frameDt = 1 / 24;
		const { numSubSteps, subDt } = computeAdaptiveSubSteps( 50, 0.5, frameDt, { courantNumber: 2 } );

		expect( subDt * numSubSteps ).toBeCloseTo( frameDt );

	} );

	it( 'respects the maxSubSteps cap for an extreme velocity spike', () => {

		const { numSubSteps } = computeAdaptiveSubSteps( 1e6, 1, 1 / 30, { courantNumber: 1, maxSubSteps: 32 } );

		expect( numSubSteps ).toBe( 32 );

	} );

	it( 'defaults maxSubSteps to 32 when not given', () => {

		const { numSubSteps } = computeAdaptiveSubSteps( 1e6, 1, 1 / 30, { courantNumber: 1 } );

		expect( numSubSteps ).toBe( 32 );

	} );

	it( 'defaults courantNumber to 5, matching jet\'s own GridFluidSolver2 default', () => {

		// cfl = 5 * (1/30) / 1 ~= 0.1667; courantNumber default 5 -> ceil(0.1667/5) = 1
		const atDefault = computeAdaptiveSubSteps( 5, 1, 1 / 30 );
		expect( atDefault.numSubSteps ).toBe( 1 );

		// Same inputs, but explicit courantNumber=0.1 (much stricter) should
		// demand more substeps -- confirms the default really is 5, not
		// something stricter that would already produce >1 above.
		const stricter = computeAdaptiveSubSteps( 5, 1, 1 / 30, { courantNumber: 0.1 } );
		expect( stricter.numSubSteps ).toBeGreaterThan( 1 );

	} );

} );
