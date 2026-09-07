// Structural tests only -- see reduction.test.js's own header comment for
// why real GPU dispatch (update() actually running) isn't exercised here.

import { describe, it, expect } from 'vitest';
import * as tsl_array_n from 'tsl_array_n';
import { createVorticityConfinement2 } from '../src/grid/vorticity_confinement2.js';
import { createFaceCenteredGrid2 } from '../src/grid/grid_data2.js';

describe( 'createVorticityConfinement2', () => {

	it( 'constructs without throwing given a velocityGrid and plain-number strength', () => {

		const velocityGrid = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );

		expect( () => createVorticityConfinement2( { velocityGrid, gridSpacing: [ 1, 1 ], strength: 0.2 } ) ).not.toThrow();

	} );

	it( 'accepts a live array0 node for strength', () => {

		const velocityGrid = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );
		const strength = tsl_array_n.array0( 'float' );
		strength.fromArray( new Float32Array( [ 0.2 ] ) );

		expect( () => createVorticityConfinement2( { velocityGrid, gridSpacing: [ 1, 1 ], strength: strength() } ) ).not.toThrow();

	} );

	it( 'returns update and force as functions', () => {

		const velocityGrid = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );
		const vorticityConfinement = createVorticityConfinement2( { velocityGrid, gridSpacing: [ 1, 1 ], strength: 0.2 } );

		expect( typeof vorticityConfinement.update ).toBe( 'function' );
		expect( typeof vorticityConfinement.force ).toBe( 'function' );

	} );

	it( 'force(pos) builds a valid node graph when used inside a kernel', () => {

		const velocityGrid = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );
		const vorticityConfinement = createVorticityConfinement2( { velocityGrid, gridSpacing: [ 1, 1 ], strength: 0.2 } );

		expect( () => tsl_array_n.kernel( velocityGrid.dataSizeU, ( i, j ) => {

			const pos = velocityGrid.uPosition( i, j );
			velocityGrid.dataU( i, j ).addAssign( vorticityConfinement.force( pos ).x );

		} ) ).not.toThrow();

	} );

} );
