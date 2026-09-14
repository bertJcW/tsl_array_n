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

	it( 'exposes its diagnostics, unset before any project() dispatch', () => {

		const solver = createGridPressureSolver2( { resolution: [ 8, 8 ], gridSpacing: [ 1, 1 ] } );

		// iterations/stoppedBy are forwarded from the CG solver after each
		// dispatch -- they are here because the iteration count is what every
		// performance question about this solver depends on, and it was
		// guessed at more than once before it was measurable.
		expect( solver.diagnostics ).toEqual( {
			converged: null, rejected: false, iterations: null, stoppedBy: null
		} );

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

	// options.faceWeights -- the variable-density (two-phase) projection.
	// Numerical correctness of the stencil and of the beta-weighted
	// correction step lives in variable_density_projection.test.js (plain JS,
	// runnable without a GPU); these check the plumbing.
	it( 'constructs with faceWeights', () => {

		const velocityGrid = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );

		expect( () => createGridPressureSolver2( {
			resolution: [ 8, 8 ], gridSpacing: [ 1, 1 ],
			faceWeights: { u: velocityGrid.dataU, v: velocityGrid.dataV }
		} ) ).not.toThrow();

	} );

	it( 'constructs with faceWeights and a dirichlet function together -- the two-phase configuration', () => {

		const velocityGrid = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );
		const dirichlet = () => ( { active: float( 0 ), target: float( 0 ) } );

		expect( () => createGridPressureSolver2( {
			resolution: [ 8, 8 ], gridSpacing: [ 1, 1 ],
			dirichlet,
			faceWeights: { u: velocityGrid.dataU, v: velocityGrid.dataV }
		} ) ).not.toThrow();

	} );

	it( 'builds project() with faceWeights -- the beta-weighted correction kernels', () => {

		const velocityGrid = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );
		const solver = createGridPressureSolver2( {
			resolution: [ 8, 8 ], gridSpacing: [ 1, 1 ],
			faceWeights: { u: velocityGrid.dataU, v: velocityGrid.dataV }
		} );

		expect( typeof solver.project( velocityGrid, velocityGrid ) ).toBe( 'function' );

	} );

	// The preconditioner choice. Construction is all that can be checked
	// here -- which one converges faster is a GPU question, measured live
	// on examples/15-flow-past-cylinder/ -- but construction is where the
	// mistakes are: a typo silently falling back to a default, or a
	// preconditioner built against the wrong mask/face-weight options.
	describe( 'preconditioner option', () => {

		const base = { resolution: [ 8, 8 ], gridSpacing: [ 1, 1 ] };

		it( 'accepts each supported preconditioner', () => {

			for ( const preconditioner of [ 'multigrid', 'jacobi', 'none' ] ) {

				expect( () => createGridPressureSolver2( { ...base, preconditioner } ) ).not.toThrow();

			}

		} );

		it( 'defaults to multigrid and reports it', () => {

			expect( createGridPressureSolver2( base ).settings.preconditioner ).toBe( 'multigrid' );

		} );

		it( 'rejects an unknown name rather than falling back', () => {

			expect( () => createGridPressureSolver2( { ...base, preconditioner: 'jacobbi' } ) )
				.toThrow( /unknown preconditioner/ );

		} );

		it( 'builds all three regardless of which is selected, so it can be switched at runtime', () => {

			const solver = createGridPressureSolver2( { ...base, preconditioner: 'jacobi' } );

			expect( solver.settings.preconditioner ).toBe( 'jacobi' );
			// The multigrid preconditioner's own runtime switches are only
			// present if it really was constructed.
			expect( solver.settings.multigrid ).toBeDefined();

		} );

		it( 'builds with a Dirichlet mask and with variable density, which is where Jacobi differs from none', () => {

			const dirichlet = () => ( { mask: () => float( 0 ), target: () => float( 0 ) } );
			const u = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 ).dataU;
			const v = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 ).dataV;

			expect( () => createGridPressureSolver2( {
				...base,
				preconditioner: 'jacobi',
				dirichlet,
				faceWeights: { u: ( i, j ) => u( i, j ), v: ( i, j ) => v( i, j ) }
			} ) ).not.toThrow();

		} );

	} );

} );
