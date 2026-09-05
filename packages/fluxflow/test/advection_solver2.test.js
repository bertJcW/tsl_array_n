// Structural tests only: building the kernel graphs (construction,
// advectFaceCentered2/advectScalar2's own kernel() calls) doesn't need a
// GPU, but backTrace's actual numerics do -- the same principle as every
// other test file in this port. Live/numeric verification is in
// examples/08-cubic-interpolation/ and examples/09-advection/.

import { describe, it, expect } from 'vitest';
import { createFaceCenteredGrid2, createScalarGrid2 } from '../src/grid/grid_data2.js';
import { createSDFStaticCollider2 } from '../src/grid/sdf_collider2.js';
import { createSemiLagrangianAdvectionSolver2 } from '../src/grid/advection_solver2.js';

describe( 'createSemiLagrangianAdvectionSolver2', () => {

	it( 'constructs without a collider, dt as a plain number', () => {

		const velocityGrid = createFaceCenteredGrid2( 4, 4, 1, 1, 0, 0 );

		expect( () => createSemiLagrangianAdvectionSolver2( { velocityGrid, dt: 0.1 } ) ).not.toThrow();

	} );

	it( 'constructs with a collider', () => {

		const velocityGrid = createFaceCenteredGrid2( 4, 4, 1, 1, 0, 0 );
		const collider = createSDFStaticCollider2( 4, 4, 1, 1, 0, 0 );

		expect( () => createSemiLagrangianAdvectionSolver2( { velocityGrid, collider, dt: 0.1 } ) ).not.toThrow();

	} );

	it( 'advectFaceCentered2 returns a 0-arg dispatcher without throwing', () => {

		const velocityGrid = createFaceCenteredGrid2( 4, 4, 1, 1, 0, 0 );
		const output = createFaceCenteredGrid2( 4, 4, 1, 1, 0, 0 );
		const solver = createSemiLagrangianAdvectionSolver2( { velocityGrid, dt: 0.1 } );

		let dispatch;
		expect( () => { dispatch = solver.advectFaceCentered2( velocityGrid, output ); } ).not.toThrow();
		expect( typeof dispatch ).toBe( 'function' );
		expect( dispatch.length ).toBe( 0 );

	} );

	it( 'advectScalar2 returns a 0-arg dispatcher without throwing', () => {

		const velocityGrid = createFaceCenteredGrid2( 4, 4, 1, 1, 0, 0 );
		const density = createScalarGrid2( 4, 4, 1, 1, 0, 0 );
		const output = createScalarGrid2( 4, 4, 1, 1, 0, 0 );
		const solver = createSemiLagrangianAdvectionSolver2( { velocityGrid, dt: 0.1 } );

		let dispatch;
		expect( () => { dispatch = solver.advectScalar2( density, output ); } ).not.toThrow();
		expect( typeof dispatch ).toBe( 'function' );
		expect( dispatch.length ).toBe( 0 );

	} );

} );
