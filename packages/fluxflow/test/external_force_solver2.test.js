// Structural tests only: building the kernel graphs doesn't need a GPU,
// but applying the force does -- the same principle as every other test
// file in this port. Live/numeric verification is in
// examples/10-external-forces/.

import { describe, it, expect } from 'vitest';
import { vec2 } from 'three/tsl';
import { createFaceCenteredGrid2 } from '../src/grid/grid_data2.js';
import { createExternalForceSolver2 } from '../src/grid/external_force_solver2.js';

describe( 'createExternalForceSolver2', () => {

	it( 'constructs with a constant force function and a plain-number dt', () => {

		const velocityGrid = createFaceCenteredGrid2( 4, 4, 1, 1, 0, 0 );
		const force = () => vec2( 0, - 1 );

		expect( () => createExternalForceSolver2( { velocityGrid, force, dt: 0.1 } ) ).not.toThrow();

	} );

	it( 'constructs with a position-dependent force function', () => {

		const velocityGrid = createFaceCenteredGrid2( 4, 4, 1, 1, 0, 0 );
		const force = ( pos ) => vec2( pos.x, 0 );

		expect( () => createExternalForceSolver2( { velocityGrid, force, dt: 0.1 } ) ).not.toThrow();

	} );

	it( 'returns a 0-arg applyExternalForces dispatcher', () => {

		const velocityGrid = createFaceCenteredGrid2( 4, 4, 1, 1, 0, 0 );
		const force = () => vec2( 0, - 1 );

		const solver = createExternalForceSolver2( { velocityGrid, force, dt: 0.1 } );

		expect( typeof solver.applyExternalForces ).toBe( 'function' );
		expect( solver.applyExternalForces.length ).toBe( 0 );

	} );

} );
