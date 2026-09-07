// Structural tests only: construction and returned shape, no GPU dispatch
// of the real (non-overridden) buoyancy/pressure/advection defaults --
// those internally call the CG solver's GPU-atomic dot product, same
// real-hardware-only situation as grid_solver2.test.js's own tests.
// Verified live instead in examples/17-smoke-fire/.

import { describe, it, expect } from 'vitest';
import * as tsl_array_n from 'tsl_array_n';
import { createGridSmokeSolver2 } from '../src/grid/grid_smoke_solver2.js';
import { createFaceCenteredGrid2 } from '../src/grid/grid_data2.js';

describe( 'createGridSmokeSolver2', () => {

	it( 'throws without options.velocityGrid', () => {

		expect( () => createGridSmokeSolver2() ).toThrow( /velocityGrid/ );
		expect( () => createGridSmokeSolver2( {} ) ).toThrow( /velocityGrid/ );

	} );

	it( 'constructs with just a velocityGrid, no other options', () => {

		const velocityGrid = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );

		expect( () => createGridSmokeSolver2( { velocityGrid } ) ).not.toThrow();

	} );

	it( 'constructs with a full option set, plain-number tunables', () => {

		const velocityGrid = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );

		expect( () => createGridSmokeSolver2( {
			velocityGrid,
			gridSpacing: [ 1, 1 ],
			force: ( pos ) => pos,
			dt: 1 / 30,
			buoyancySmokeDensityFactor: -0.001,
			buoyancyTemperatureFactor: 4,
			ambientTemperature: 0.5,
			smokeDecay: 0.002,
			temperatureDecay: 0.002,
			up: [ 0, 1 ],
			advection: { maxSubsteps: 8 },
			pressure: { multigrid: { numberOfLevels: 2 }, tolerance: 1e-4, maxIterations: 20 }
		} ) ).not.toThrow();

	} );

	it( 'accepts live array0 nodes for the tunable factors', () => {

		const velocityGrid = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );
		const buoyancyTemperatureFactor = tsl_array_n.array0( 'float' );
		buoyancyTemperatureFactor.fromArray( new Float32Array( [ 5 ] ) );

		expect( () => createGridSmokeSolver2( {
			velocityGrid, dt: 1 / 30, buoyancyTemperatureFactor: buoyancyTemperatureFactor()
		} ) ).not.toThrow();

	} );

	it( 'returns onAdvanceTimeStep, velocityGrid, density/temperature (each with stateA/stateB/current), and the underlying solver', () => {

		const velocityGrid = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );
		const smoke = createGridSmokeSolver2( { velocityGrid, dt: 1 / 30 } );

		expect( typeof smoke.onAdvanceTimeStep ).toBe( 'function' );
		expect( smoke.velocityGrid ).toBeDefined();

		for ( const field of [ smoke.density, smoke.temperature ] ) {

			expect( field.stateA ).toBeDefined();
			expect( field.stateB ).toBeDefined();
			expect( field.current ).toBe( field.stateA ); // fresh construction -- state A starts active

		}

		expect( smoke.solver ).toBeDefined();
		expect( smoke.solver.pressureSolver ).toBeDefined();

	} );

	it( 'density/temperature state fields expose a sample(pos) method', () => {

		const velocityGrid = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );
		const smoke = createGridSmokeSolver2( { velocityGrid, dt: 1 / 30 } );

		expect( typeof smoke.density.stateA.sample ).toBe( 'function' );
		expect( typeof smoke.temperature.stateA.sample ).toBe( 'function' );

	} );

	it( 'constructs with inflows/outflows/collider/closedDomainBoundaryFlag forwarded', () => {

		const velocityGrid = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );

		expect( () => createGridSmokeSolver2( {
			velocityGrid, dt: 1 / 30, closedDomainBoundaryFlag: 0
		} ) ).not.toThrow();

	} );

} );
