// Structural tests only: construction (allocating scratch fields, building
// the update kernels, calling applyOperator twice) doesn't need a GPU, but
// solve() itself does (every iteration reads fields back via toArray()) --
// so the actual convergence behavior is verified live instead, in
// examples/04-conjugate-gradient/, against a diagonal operator with a known
// exact solution.

import { describe, it, expect } from 'vitest';
import * as tsl_array_n from 'tsl_array_n';
import { createConjugateGradientSolver } from '../src/linalg/linalg.js';

// A no-op stand-in for a real matvec factory -- fine for these tests since
// none of them call .solve() (which is the only thing that would actually
// invoke the dispatcher this returns).
const noopOperator = () => () => {};

describe( 'createConjugateGradientSolver', () => {

	it( 'constructs for a 1D field pair without throwing', () => {

		const b = tsl_array_n.arrayN( 'float', 8 );
		const x = tsl_array_n.arrayN( 'float', 8 );

		expect( () => createConjugateGradientSolver( noopOperator, b, x ) ).not.toThrow();

	} );

	it( 'constructs for a 2D field pair without throwing', () => {

		const b = tsl_array_n.array2( 'float', 4, 4 );
		const x = tsl_array_n.array2( 'float', 4, 4 );

		expect( () => createConjugateGradientSolver( noopOperator, b, x ) ).not.toThrow();

	} );

	it( 'constructs for a 3D field pair without throwing', () => {

		const b = tsl_array_n.array3( 'float', 3, 3, 3 );
		const x = tsl_array_n.array3( 'float', 3, 3, 3 );

		expect( () => createConjugateGradientSolver( noopOperator, b, x ) ).not.toThrow();

	} );

	it( 'rejects a 4D field pair (kernel dispatch is inherently <=3D)', () => {

		const b = tsl_array_n.arrayN( 'float', [ 2, 2, 2, 2 ] );
		const x = tsl_array_n.arrayN( 'float', [ 2, 2, 2, 2 ] );

		expect( () => createConjugateGradientSolver( noopOperator, b, x ) ).toThrow( /1D\/2D\/3D/ );

	} );

	it( 'rejects a b/x element type mismatch', () => {

		const b = tsl_array_n.array2( 'float', 4, 4 );
		const x = tsl_array_n.array2( 'vec2', 4, 4 );

		expect( () => createConjugateGradientSolver( noopOperator, b, x ) ).toThrow( /type mismatch/ );

	} );

	it( 'rejects a b/x shape mismatch', () => {

		const b = tsl_array_n.array2( 'float', 4, 4 );
		const x = tsl_array_n.array2( 'float', 4, 5 );

		expect( () => createConjugateGradientSolver( noopOperator, b, x ) ).toThrow( /shape mismatch/ );

	} );

	it( 'returns a solve function and the internal scratch fields', () => {

		const b = tsl_array_n.array2( 'float', 4, 4 );
		const x = tsl_array_n.array2( 'float', 4, 4 );

		const solver = createConjugateGradientSolver( noopOperator, b, x );

		expect( typeof solver.solve ).toBe( 'function' );
		expect( solver.p.shape ).toEqual( [ 4, 4 ] );
		expect( solver.r.shape ).toEqual( [ 4, 4 ] );
		expect( solver.Ap.shape ).toEqual( [ 4, 4 ] );
		expect( solver.Ax.shape ).toEqual( [ 4, 4 ] );

	} );

	it( 'calls applyOperator exactly twice, for (x, Ax) and (p, Ap)', () => {

		const calls = [];
		const spyOperator = ( input, output ) => {

			calls.push( { input, output } );
			return () => {};

		};

		const b = tsl_array_n.array2( 'float', 4, 4 );
		const x = tsl_array_n.array2( 'float', 4, 4 );

		const solver = createConjugateGradientSolver( spyOperator, b, x );

		expect( calls.length ).toBe( 2 );
		expect( calls[ 0 ].input ).toBe( x );
		expect( calls[ 0 ].output ).toBe( solver.Ax );
		expect( calls[ 1 ].input ).toBe( solver.p );
		expect( calls[ 1 ].output ).toBe( solver.Ap );

	} );

} );
