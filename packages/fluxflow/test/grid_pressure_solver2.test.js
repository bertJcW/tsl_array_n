// Structural tests only, matching every other solver in this port: pure
// graph construction (no GPU needed), including building project()'s own
// kernels. The actual pressure solve needs the CG solver's GPU-atomic dot
// product, which needs a live renderer -- verified live instead, in
// examples/13-interactive-pressure/ (needs real WebGPU hardware, same as
// every other atomics-based example in this port) and, for the Dirichlet
// mask mechanism specifically (no atomics involved), an added case in
// examples/06-multigrid-preconditioner/.

import { describe, it, expect } from 'vitest';
import { float } from 'three/tsl';
import { createGridPressureSolver2 } from '../src/grid/grid_pressure_solver2.js';
import { createFaceCenteredGrid2 } from '../src/grid/grid_data2.js';

describe( 'createGridPressureSolver2', () => {

	it( 'constructs without a dirichlet function', () => {

		expect( () => createGridPressureSolver2( { resolution: [ 8, 8 ], gridSpacing: [ 1, 1 ] } ) ).not.toThrow();

	} );

	it( 'constructs with a dirichlet function', () => {

		const dirichlet = () => ( { active: float( 0 ), target: float( 0 ) } );

		expect( () => createGridPressureSolver2( { resolution: [ 8, 8 ], gridSpacing: [ 1, 1 ], dirichlet } ) ).not.toThrow();

	} );

	it( 'exposes a pressure grid shaped like a CellCenteredScalarGrid2', () => {

		const solver = createGridPressureSolver2( { resolution: [ 8, 8 ], gridSpacing: [ 1, 1 ] } );

		expect( solver.pressure.dataSize ).toEqual( [ 8, 8 ] );
		expect( typeof solver.pressure.data ).toBe( 'function' );

	} );

	it( 'exposes b (the divergence RHS field), for diagnostics', () => {

		const solver = createGridPressureSolver2( { resolution: [ 8, 8 ], gridSpacing: [ 1, 1 ] } );

		expect( typeof solver.b ).toBe( 'function' );
		expect( solver.b.shape ).toEqual( [ 8, 8 ] );

	} );

	it( 'exposes diagnostics.converged/rejected, starting as null/false before any project() dispatch', () => {

		const solver = createGridPressureSolver2( { resolution: [ 8, 8 ], gridSpacing: [ 1, 1 ] } );

		expect( solver.diagnostics ).toEqual( { converged: null, rejected: false } );

	} );

	it( 'accepts an atomicScale option without throwing', () => {

		expect( () => createGridPressureSolver2( { resolution: [ 8, 8 ], gridSpacing: [ 1, 1 ], atomicScale: 1e9 } ) ).not.toThrow();

	} );

	it( 'project() returns a function without invoking it', () => {

		const solver = createGridPressureSolver2( { resolution: [ 8, 8 ], gridSpacing: [ 1, 1 ] } );
		const velocityGrid = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );

		const dispatch = solver.project( velocityGrid, velocityGrid );

		expect( typeof dispatch ).toBe( 'function' );

	} );

	it( 'project() throws on a resolution mismatch', () => {

		const solver = createGridPressureSolver2( { resolution: [ 8, 8 ], gridSpacing: [ 1, 1 ] } );
		const velocityGrid = createFaceCenteredGrid2( 4, 4, 1, 1, 0, 0 );

		expect( () => solver.project( velocityGrid, velocityGrid ) ).toThrow( /resolution/ );

	} );

} );
