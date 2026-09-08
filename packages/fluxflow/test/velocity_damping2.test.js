// Structural tests only -- see reduction.test.js's own header comment for
// why real GPU dispatch (applyDamping() actually running) isn't exercised
// here.

import { describe, it, expect } from 'vitest';
import * as tsl_array_n from 'tsl_array_n';
import { createVelocityDamping2 } from '../src/grid/velocity_damping2.js';
import { createFaceCenteredGrid2 } from '../src/grid/grid_data2.js';

describe( 'createVelocityDamping2', () => {

	it( 'constructs without throwing given a velocityGrid and default dampingCoefficient', () => {

		const velocityGrid = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );

		expect( () => createVelocityDamping2( { velocityGrid } ) ).not.toThrow();

	} );

	it( 'constructs without throwing given a plain-number dampingCoefficient', () => {

		const velocityGrid = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );

		expect( () => createVelocityDamping2( { velocityGrid, dampingCoefficient: 0.02 } ) ).not.toThrow();

	} );

	it( 'accepts a live array0 node for dampingCoefficient', () => {

		const velocityGrid = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );
		const dampingCoefficient = tsl_array_n.array0( 'float' );
		dampingCoefficient.fromArray( new Float32Array( [ 0.02 ] ) );

		expect( () => createVelocityDamping2( { velocityGrid, dampingCoefficient: dampingCoefficient() } ) ).not.toThrow();

	} );

	it( 'returns applyDamping as a function', () => {

		const velocityGrid = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );
		const damping = createVelocityDamping2( { velocityGrid, dampingCoefficient: 0.02 } );

		expect( typeof damping.applyDamping ).toBe( 'function' );

	} );

} );
