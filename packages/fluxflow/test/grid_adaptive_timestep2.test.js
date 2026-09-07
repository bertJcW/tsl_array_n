// Structural tests only -- see reduction.test.js's own header comment for
// why real GPU dispatch (update() actually running) isn't exercised here.

import { describe, it, expect } from 'vitest';
import * as tsl_array_n from 'tsl_array_n';
import { createGridAdaptiveTimeStep2 } from '../src/grid/grid_adaptive_timestep2.js';
import { createFaceCenteredGrid2 } from '../src/grid/grid_data2.js';

describe( 'createGridAdaptiveTimeStep2', () => {

	it( 'throws a clear error when dt is a plain number', () => {

		const velocityGrid = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );

		expect( () => createGridAdaptiveTimeStep2( {
			velocityGrid, gridSpacing: [ 1, 1 ], dt: 1 / 30, targetDt: 1 / 30
		} ) ).toThrow( /array0/ );

	} );

	it( 'constructs cleanly when dt is a live array0 node', () => {

		const velocityGrid = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );
		const dt = tsl_array_n.array0( 'float' );
		dt.fromArray( new Float32Array( [ 1 / 30 ] ) );

		expect( () => createGridAdaptiveTimeStep2( {
			velocityGrid, gridSpacing: [ 1, 1 ], dt, targetDt: 1 / 30
		} ) ).not.toThrow();

	} );

	it( 'returns update as a function and exposes initial state', () => {

		const velocityGrid = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );
		const dt = tsl_array_n.array0( 'float' );
		dt.fromArray( new Float32Array( [ 1 / 30 ] ) );

		const adaptiveTimeStep = createGridAdaptiveTimeStep2( {
			velocityGrid, gridSpacing: [ 1, 1 ], dt, targetDt: 1 / 30
		} );

		expect( typeof adaptiveTimeStep.update ).toBe( 'function' );
		expect( adaptiveTimeStep.state.lastNumSubSteps ).toBe( 1 );
		expect( adaptiveTimeStep.state.lastSubDt ).toBeCloseTo( 1 / 30 );

	} );

	it( 'accepts courantNumber/maxSubSteps/atomicScale options without throwing', () => {

		const velocityGrid = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );
		const dt = tsl_array_n.array0( 'float' );
		dt.fromArray( new Float32Array( [ 1 / 30 ] ) );

		expect( () => createGridAdaptiveTimeStep2( {
			velocityGrid, gridSpacing: [ 1, 1 ], dt, targetDt: 1 / 30,
			courantNumber: 2, maxSubSteps: 16, atomicScale: 1024
		} ) ).not.toThrow();

	} );

} );
