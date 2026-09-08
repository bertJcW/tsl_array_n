// Structural tests only: construction and returned shape, no GPU dispatch
// of the real (non-overridden) advection/burn kernels -- those internally
// need a real renderer, same real-hardware-only situation as
// grid_smoke_solver2.test.js's own tests. Verified live instead in
// examples/19-fuel-fire/.

import { describe, it, expect } from 'vitest';
import * as tsl_array_n from 'tsl_array_n';
import { vec2 } from 'three/tsl';
import { createGridFireSolver2 } from '../src/grid/grid_fire_solver2.js';
import { createFaceCenteredGrid2 } from '../src/grid/grid_data2.js';
import { createSDFFuelSource2 } from '../src/grid/sdf_inflow_outflow2.js';

describe( 'createGridFireSolver2', () => {

	it( 'throws without options.velocityGrid', () => {

		expect( () => createGridFireSolver2() ).toThrow( /velocityGrid/ );
		expect( () => createGridFireSolver2( {} ) ).toThrow( /velocityGrid/ );

	} );

	it( 'constructs with just a velocityGrid, no other options', () => {

		const velocityGrid = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );

		expect( () => createGridFireSolver2( { velocityGrid } ) ).not.toThrow();

	} );

	it( 'constructs with a full option set, plain-number tunables', () => {

		const velocityGrid = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );

		expect( () => createGridFireSolver2( {
			velocityGrid,
			gridSpacing: [ 1, 1 ],
			dt: 1 / 30,
			burningRate: 0.5,
			flameSmoke: 0.8,
			ignitionTemp: 1,
			maxTemp: 2,
			buoyancySmokeDensityFactor: -0.001,
			buoyancyTemperatureFactor: 4,
			ambientTemperature: 0.5,
			smokeDecay: 0.002,
			temperatureDecay: 0.002,
			up: [ 0, 1 ],
			advection: { order: 2, maxSubsteps: 8 }
		} ) ).not.toThrow();

	} );

	it( 'constructs with a single fuel source', () => {

		const velocityGrid = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );
		const source = createSDFFuelSource2( 8, 8, 1, 1, 0, 0 );

		expect( () => createGridFireSolver2( { velocityGrid, dt: 1 / 30, fuelSources: source } ) ).not.toThrow();

	} );

	it( 'constructs with an array of fuel sources', () => {

		const velocityGrid = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );
		const a = createSDFFuelSource2( 8, 8, 1, 1, 0, 0 );
		const b = createSDFFuelSource2( 8, 8, 1, 1, 0, 0, { fuel: 0.5, mode: 'add' } );

		expect( () => createGridFireSolver2( { velocityGrid, dt: 1 / 30, fuelSources: [ a, b ] } ) ).not.toThrow();

	} );

	it( 'constructs with a collider forwarded to its own advection solver', () => {

		const velocityGrid = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );

		// No real collider object needed here -- undefined is the
		// "no collider" default this file already documents; this test
		// only checks that passing the option through doesn't throw at
		// construction time (a real collider would need tsl_array_n.init()
		// first, same caveat createGridSolver2.test.js already documents).
		expect( () => createGridFireSolver2( { velocityGrid, dt: 1 / 30, collider: undefined } ) ).not.toThrow();

	} );

	it( 'accepts live array0 nodes for the tunable factors', () => {

		const velocityGrid = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );
		const buoyancyTemperatureFactor = tsl_array_n.array0( 'float' );
		buoyancyTemperatureFactor.fromArray( new Float32Array( [ 5 ] ) );

		expect( () => createGridFireSolver2( {
			velocityGrid, dt: 1 / 30, buoyancyTemperatureFactor: buoyancyTemperatureFactor()
		} ) ).not.toThrow();

	} );

	it( 'returns onAdvanceTimeStep, force, and fuel/react/density/temperature (each with stateA/stateB/current)', () => {

		const velocityGrid = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );
		const fire = createGridFireSolver2( { velocityGrid, dt: 1 / 30 } );

		expect( typeof fire.onAdvanceTimeStep ).toBe( 'function' );
		expect( typeof fire.force ).toBe( 'function' );

		for ( const field of [ fire.fuel, fire.react, fire.density, fire.temperature ] ) {

			expect( field.stateA ).toBeDefined();
			expect( field.stateB ).toBeDefined();
			expect( field.current ).toBe( field.stateA ); // fresh construction -- state A starts active

		}

	} );

	it( 'force(pos) returns a real node', () => {

		const velocityGrid = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );
		const fire = createGridFireSolver2( { velocityGrid, dt: 1 / 30 } );
		const result = fire.force( vec2( 0, 0 ) );

		expect( result ).toBeTruthy();
		expect( result.isNode ).toBe( true );

	} );

	it( 'fuel/react/density/temperature state fields expose a sample(pos) method', () => {

		const velocityGrid = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );
		const fire = createGridFireSolver2( { velocityGrid, dt: 1 / 30 } );

		expect( typeof fire.fuel.stateA.sample ).toBe( 'function' );
		expect( typeof fire.react.stateA.sample ).toBe( 'function' );
		expect( typeof fire.density.stateA.sample ).toBe( 'function' );
		expect( typeof fire.temperature.stateA.sample ).toBe( 'function' );

	} );

} );
