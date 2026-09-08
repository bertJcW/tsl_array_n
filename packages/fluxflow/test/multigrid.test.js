// Structural tests only: everything here is pure graph construction (no
// GPU needed) -- including actually calling the returned
// applyPreconditioner(input, output) and building every level's
// relax/restrict/correct/residual kernels, which exercises the
// restriction/correction combinatorics for real without needing a live
// renderer. The V-cycle's actual numerical behavior is verified live
// instead, in examples/06-multigrid-preconditioner/ (standalone, and --
// unlike every other linalg example -- verifiable even in this dev
// sandbox's WebGL2 fallback, since multigrid needs no reduction/atomics
// anywhere) and examples/07-multigrid-preconditioned-cg/ (the full
// pipeline, which does need atomics and so needs real WebGPU hardware).

import { describe, it, expect } from 'vitest';
import * as tsl_array_n from 'tsl_array_n';
import { createMultigridPreconditioner, createLaplacianOperator } from '../src/linalg/multigrid.js';

describe( 'createMultigridPreconditioner', () => {

	it( 'constructs for a 1D shape with the default (single) level', () => {

		expect( () => createMultigridPreconditioner( [ 8 ], [ 1 ] ) ).not.toThrow();

	} );

	it( 'constructs for a 2D shape with multiple levels', () => {

		expect( () => createMultigridPreconditioner( [ 8, 8 ], [ 1, 1 ], { numberOfLevels: 3 } ) ).not.toThrow();

	} );

	it( 'constructs for a 3D shape with multiple levels', () => {

		expect( () => createMultigridPreconditioner( [ 8, 8, 8 ], [ 1, 1, 1 ], { numberOfLevels: 2 } ) ).not.toThrow();

	} );

	it( 'rejects a 4D shape', () => {

		expect( () => createMultigridPreconditioner( [ 8, 8, 8, 8 ], [ 1, 1, 1, 1 ] ) ).toThrow( /1D\/2D\/3D/ );

	} );

	it( 'rejects a gridSpacing length mismatch', () => {

		expect( () => createMultigridPreconditioner( [ 8, 8 ], [ 1 ] ) ).toThrow( /gridSpacing length/ );

	} );

	it( 'rejects a shape not divisible by 2^(numberOfLevels-1)', () => {

		expect( () => createMultigridPreconditioner( [ 10, 10 ], [ 1, 1 ], { numberOfLevels: 4 } ) ).toThrow( /not divisible/ );

	} );

	it( 'returns a 2-arg factory whose own return value is a 0-arg dispatcher (single level)', () => {

		const applyPreconditioner = createMultigridPreconditioner( [ 8 ], [ 1 ] );

		expect( typeof applyPreconditioner ).toBe( 'function' );
		expect( applyPreconditioner.length ).toBe( 2 );

		const r = tsl_array_n.arrayN( 'float', 8 );
		const z = tsl_array_n.arrayN( 'float', 8 );
		const dispatch = applyPreconditioner( r, z );

		expect( typeof dispatch ).toBe( 'function' );
		expect( dispatch.length ).toBe( 0 );

	} );

	it( 'builds every level\'s kernels without throwing for a multi-level 2D shape', () => {

		const applyPreconditioner = createMultigridPreconditioner( [ 16, 16 ], [ 1, 1 ], { numberOfLevels: 4 } );

		const r = tsl_array_n.arrayN( 'float', [ 16, 16 ] );
		const z = tsl_array_n.arrayN( 'float', [ 16, 16 ] );

		expect( () => applyPreconditioner( r, z ) ).not.toThrow();

	} );

	it( 'builds without throwing given a dirichletMask, single level', () => {

		const mask = tsl_array_n.arrayN( 'float', [ 8, 8 ] );
		const applyPreconditioner = createMultigridPreconditioner( [ 8, 8 ], [ 1, 1 ], {
			dirichletMask: ( i, j ) => mask( i, j ).greaterThan( 0.5 )
		} );

		const r = tsl_array_n.arrayN( 'float', [ 8, 8 ] );
		const z = tsl_array_n.arrayN( 'float', [ 8, 8 ] );

		expect( () => applyPreconditioner( r, z ) ).not.toThrow();

	} );

	it( 'builds without throwing given a dirichletMask, multiple levels (mask applied at level 0 only)', () => {

		const mask = tsl_array_n.arrayN( 'float', [ 16, 16 ] );
		const applyPreconditioner = createMultigridPreconditioner( [ 16, 16 ], [ 1, 1 ], {
			numberOfLevels: 4,
			dirichletMask: ( i, j ) => mask( i, j ).greaterThan( 0.5 )
		} );

		const r = tsl_array_n.arrayN( 'float', [ 16, 16 ] );
		const z = tsl_array_n.arrayN( 'float', [ 16, 16 ] );

		expect( () => applyPreconditioner( r, z ) ).not.toThrow();

	} );

	it( 'accepts faceWeights and applies them at the finest level only', () => {

		// Level 0 gets the real per-face coefficients; the coarse levels stay
		// constant-coefficient by design (multigrid.js decision 4), exactly
		// as dirichletMask already does. What this pins down is that a
		// multi-level V-cycle still builds with the option present -- the
		// coarse level shapes do not match the fine face arrays, so a version
		// that leaked faceWeights downward would throw here.
		const betaU = tsl_array_n.arrayN( 'float', [ 9, 8 ] );
		const betaV = tsl_array_n.arrayN( 'float', [ 8, 9 ] );

		const applyPreconditioner = createMultigridPreconditioner( [ 8, 8 ], [ 1, 1 ], {
			numberOfLevels: 3,
			faceWeights: [ betaU, betaV ]
		} );

		const input = tsl_array_n.arrayN( 'float', [ 8, 8 ] );
		const output = tsl_array_n.arrayN( 'float', [ 8, 8 ] );

		expect( () => applyPreconditioner( input, output ) ).not.toThrow();

	} );

} );

describe( 'createLaplacianOperator', () => {

	it( 'returns a 2-arg factory whose own return value is a 0-arg dispatcher', () => {

		const applyLaplacian = createLaplacianOperator( [ 8, 8 ], [ 1, 1 ] );

		expect( typeof applyLaplacian ).toBe( 'function' );
		expect( applyLaplacian.length ).toBe( 2 );

		const input = tsl_array_n.arrayN( 'float', [ 8, 8 ] );
		const output = tsl_array_n.arrayN( 'float', [ 8, 8 ] );
		const dispatch = applyLaplacian( input, output );

		expect( typeof dispatch ).toBe( 'function' );
		expect( dispatch.length ).toBe( 0 );

	} );

	it( 'builds without throwing given a dirichletMask', () => {

		const mask = tsl_array_n.arrayN( 'float', [ 8, 8 ] );
		const applyLaplacian = createLaplacianOperator( [ 8, 8 ], [ 1, 1 ], {
			dirichletMask: ( i, j ) => mask( i, j ).greaterThan( 0.5 )
		} );

		const input = tsl_array_n.arrayN( 'float', [ 8, 8 ] );
		const output = tsl_array_n.arrayN( 'float', [ 8, 8 ] );

		expect( () => applyLaplacian( input, output ) ).not.toThrow();

	} );

	// options.faceWeights (decision 4 in multigrid.js's header comment):
	// per-face coefficients for the variable-density two-phase projection.
	// The stencil's own numerical correctness is covered separately in
	// variable_density_projection.test.js, which reimplements it in plain JS
	// so it can actually be *run* without a GPU; what's checked here is that
	// the MAC-shaped face arrays are accepted and the graph builds.
	it( 'builds without throwing given faceWeights (MAC face arrays)', () => {

		const betaU = tsl_array_n.arrayN( 'float', [ 9, 8 ] ); // [nx+1, ny]
		const betaV = tsl_array_n.arrayN( 'float', [ 8, 9 ] ); // [nx, ny+1]

		const applyLaplacian = createLaplacianOperator( [ 8, 8 ], [ 1, 1 ], {
			faceWeights: [ betaU, betaV ]
		} );

		const input = tsl_array_n.arrayN( 'float', [ 8, 8 ] );
		const output = tsl_array_n.arrayN( 'float', [ 8, 8 ] );

		expect( () => applyLaplacian( input, output ) ).not.toThrow();

	} );

	it( 'builds without throwing given faceWeights AND a dirichletMask together', () => {

		const mask = tsl_array_n.arrayN( 'float', [ 8, 8 ] );
		const betaU = tsl_array_n.arrayN( 'float', [ 9, 8 ] );
		const betaV = tsl_array_n.arrayN( 'float', [ 8, 9 ] );

		const applyLaplacian = createLaplacianOperator( [ 8, 8 ], [ 1, 1 ], {
			dirichletMask: ( i, j ) => mask( i, j ).greaterThan( 0.5 ),
			faceWeights: [ betaU, betaV ]
		} );

		const input = tsl_array_n.arrayN( 'float', [ 8, 8 ] );
		const output = tsl_array_n.arrayN( 'float', [ 8, 8 ] );

		expect( () => applyLaplacian( input, output ) ).not.toThrow();

	} );

	it( 'builds a 1D faceWeights operator -- the option is dimension-generic too', () => {

		const beta = tsl_array_n.arrayN( 'float', [ 9 ] );
		const applyLaplacian = createLaplacianOperator( [ 8 ], [ 1 ], { faceWeights: [ beta ] } );

		const input = tsl_array_n.arrayN( 'float', [ 8 ] );
		const output = tsl_array_n.arrayN( 'float', [ 8 ] );

		expect( () => applyLaplacian( input, output ) ).not.toThrow();

	} );

} );
