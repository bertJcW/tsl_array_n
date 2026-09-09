// Structural tests only: construction (allocating scratch fields, building
// the update kernels, calling applyOperator/applyPreconditioner) doesn't
// need a GPU, but solve() itself does (every iteration reads fields back
// via toArray()) -- so the actual convergence behavior is verified live
// instead: examples/04-conjugate-gradient/ (plain CG) and
// examples/05-preconditioned-conjugate-gradient/ (preconditioned CG),
// both against a diagonal operator with a known exact solution.
//
// isDegenerateDenominator is the one exception: a pure function of a plain
// number, no GPU involved at all, so its own logic is verified directly
// below instead of only through solve()'s live behavior.

import { describe, it, expect } from 'vitest';
import * as tsl_array_n from 'tsl_array_n';
import { createConjugateGradientSolver, createPreconditionedConjugateGradientSolver, isDegenerateDenominator, createDotReducer } from '../src/linalg/linalg.js';

// A no-op stand-in for a real matvec/preconditioner factory -- fine for
// these tests since none of them call .solve() (the only thing that would
// actually invoke the dispatcher this returns).
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

	it( 'rejects a matching-but-non-float element type (GPU atomic dot only supports float)', () => {

		const b = tsl_array_n.array2( 'vec2', 4, 4 );
		const x = tsl_array_n.array2( 'vec2', 4, 4 );

		expect( () => createConjugateGradientSolver( noopOperator, b, x ) ).toThrow( /only supports type "float"/ );

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

describe( 'createPreconditionedConjugateGradientSolver', () => {

	it( 'constructs for a 1D field pair without throwing', () => {

		const b = tsl_array_n.arrayN( 'float', 8 );
		const x = tsl_array_n.arrayN( 'float', 8 );

		expect( () => createPreconditionedConjugateGradientSolver( noopOperator, noopOperator, b, x ) ).not.toThrow();

	} );

	it( 'constructs for a 2D field pair without throwing', () => {

		const b = tsl_array_n.array2( 'float', 4, 4 );
		const x = tsl_array_n.array2( 'float', 4, 4 );

		expect( () => createPreconditionedConjugateGradientSolver( noopOperator, noopOperator, b, x ) ).not.toThrow();

	} );

	it( 'constructs for a 3D field pair without throwing', () => {

		const b = tsl_array_n.array3( 'float', 3, 3, 3 );
		const x = tsl_array_n.array3( 'float', 3, 3, 3 );

		expect( () => createPreconditionedConjugateGradientSolver( noopOperator, noopOperator, b, x ) ).not.toThrow();

	} );

	it( 'rejects a 4D field pair (kernel dispatch is inherently <=3D)', () => {

		const b = tsl_array_n.arrayN( 'float', [ 2, 2, 2, 2 ] );
		const x = tsl_array_n.arrayN( 'float', [ 2, 2, 2, 2 ] );

		expect( () => createPreconditionedConjugateGradientSolver( noopOperator, noopOperator, b, x ) ).toThrow( /1D\/2D\/3D/ );

	} );

	it( 'rejects a b/x element type mismatch', () => {

		const b = tsl_array_n.array2( 'float', 4, 4 );
		const x = tsl_array_n.array2( 'vec2', 4, 4 );

		expect( () => createPreconditionedConjugateGradientSolver( noopOperator, noopOperator, b, x ) ).toThrow( /type mismatch/ );

	} );

	it( 'rejects a b/x shape mismatch', () => {

		const b = tsl_array_n.array2( 'float', 4, 4 );
		const x = tsl_array_n.array2( 'float', 4, 5 );

		expect( () => createPreconditionedConjugateGradientSolver( noopOperator, noopOperator, b, x ) ).toThrow( /shape mismatch/ );

	} );

	it( 'rejects a matching-but-non-float element type (GPU atomic dot only supports float)', () => {

		const b = tsl_array_n.array2( 'vec2', 4, 4 );
		const x = tsl_array_n.array2( 'vec2', 4, 4 );

		expect( () => createPreconditionedConjugateGradientSolver( noopOperator, noopOperator, b, x ) ).toThrow( /only supports type "float"/ );

	} );

	it( 'returns a solve function and the internal scratch fields, including z', () => {

		const b = tsl_array_n.array2( 'float', 4, 4 );
		const x = tsl_array_n.array2( 'float', 4, 4 );

		const solver = createPreconditionedConjugateGradientSolver( noopOperator, noopOperator, b, x );

		expect( typeof solver.solve ).toBe( 'function' );
		expect( solver.p.shape ).toEqual( [ 4, 4 ] );
		expect( solver.r.shape ).toEqual( [ 4, 4 ] );
		expect( solver.z.shape ).toEqual( [ 4, 4 ] );
		expect( solver.Ap.shape ).toEqual( [ 4, 4 ] );
		expect( solver.Ax.shape ).toEqual( [ 4, 4 ] );

	} );

	it( 'calls applyOperator exactly twice for (x, Ax)/(p, Ap), and applyPreconditioner exactly once for (r, z)', () => {

		const operatorCalls = [];
		const spyOperator = ( input, output ) => {

			operatorCalls.push( { input, output } );
			return () => {};

		};

		const preconditionerCalls = [];
		const spyPreconditioner = ( input, output ) => {

			preconditionerCalls.push( { input, output } );
			return () => {};

		};

		const b = tsl_array_n.array2( 'float', 4, 4 );
		const x = tsl_array_n.array2( 'float', 4, 4 );

		const solver = createPreconditionedConjugateGradientSolver( spyOperator, spyPreconditioner, b, x );

		expect( operatorCalls.length ).toBe( 2 );
		expect( operatorCalls[ 0 ].input ).toBe( x );
		expect( operatorCalls[ 0 ].output ).toBe( solver.Ax );
		expect( operatorCalls[ 1 ].input ).toBe( solver.p );
		expect( operatorCalls[ 1 ].output ).toBe( solver.Ap );

		expect( preconditionerCalls.length ).toBe( 1 );
		expect( preconditionerCalls[ 0 ].input ).toBe( solver.r );
		expect( preconditionerCalls[ 0 ].output ).toBe( solver.z );

	} );

} );

describe( 'isDegenerateDenominator', () => {

	it( 'flags exactly 0', () => {

		expect( isDegenerateDenominator( 0 ) ).toBe( true );
		expect( isDegenerateDenominator( - 0 ) ).toBe( true );

	} );

	it( 'flags non-finite values', () => {

		expect( isDegenerateDenominator( NaN ) ).toBe( true );
		expect( isDegenerateDenominator( Infinity ) ).toBe( true );
		expect( isDegenerateDenominator( - Infinity ) ).toBe( true );

	} );

	it( 'does NOT flag a merely small value', () => {

		// This is the whole point of the change away from a magnitude
		// threshold: a converging CG's dot products get small as a matter of
		// course, and the old scale-based floor could not tell "the operator
		// is singular" from "the solve is nearly finished". A tiny
		// denominator is now allowed through and the *quotient* is checked
		// for plausibility instead (MAX_ALPHA_MAGNITUDE).
		expect( isDegenerateDenominator( 1e-8 ) ).toBe( false );
		expect( isDegenerateDenominator( - 1e-8 ) ).toBe( false );
		expect( isDegenerateDenominator( Number.MIN_VALUE ) ).toBe( false );
		expect( isDegenerateDenominator( 1 ) ).toBe( false );

	} );

} );

describe( 'createDotReducer', () => {

	it( 'builds without throwing for 1D/2D/3D shapes', () => {

		const a1 = tsl_array_n.arrayN( 'float', 8 );
		const a2 = tsl_array_n.array2( 'float', 4, 4 );
		const a3 = tsl_array_n.array3( 'float', 2, 2, 2 );

		expect( () => createDotReducer( [ 8 ], a1, a1 ) ).not.toThrow();
		expect( () => createDotReducer( [ 4, 4 ], a2, a2 ) ).not.toThrow();
		expect( () => createDotReducer( [ 2, 2, 2 ], a3, a3 ) ).not.toThrow();

	} );

	it( 'rejects a 4D shape, matching buildElementwiseKernel own limit', () => {

		const a4 = tsl_array_n.arrayN( 'float', [ 2, 2, 2, 2 ] );

		expect( () => createDotReducer( [ 2, 2, 2, 2 ], a4, a4 ) ).toThrow( /1D\/2D\/3D/ );

	} );

	it( 'uses one lane per first-axis index in 2D, and caps lanes in 1D', () => {

		const wide = tsl_array_n.array2( 'float', 6, 9 );
		const long = tsl_array_n.arrayN( 'float', 4096 );
		const short = tsl_array_n.arrayN( 'float', 10 );

		expect( createDotReducer( [ 6, 9 ], wide, wide ).lanes ).toBe( 6 );
		// 1D caps at 64 lanes and lets each stride, rather than reading back
		// one partial per element.
		expect( createDotReducer( [ 4096 ], long, long ).lanes ).toBe( 64 );
		expect( createDotReducer( [ 10 ], short, short ).lanes ).toBe( 10 );

	} );

} );
