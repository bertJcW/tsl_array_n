// Structural tests only: construction (allocating scratch fields, building
// the update kernels, calling applyOperator/applyPreconditioner) doesn't
// need a GPU, but solve() itself does (every iteration reads fields back
// via toArray()) -- so the actual convergence behavior is verified live
// instead: examples/04-conjugate-gradient/ (plain CG) and
// examples/05-preconditioned-conjugate-gradient/ (preconditioned CG),
// both against a diagonal operator with a known exact solution.
//
// isDegenerateDot is the one exception: a pure function of two plain
// numbers, no GPU involved at all, so its own threshold math is verified
// directly below instead of only through solve()'s live behavior.

import { describe, it, expect } from 'vitest';
import * as tsl_array_n from 'tsl_array_n';
import { createConjugateGradientSolver, createPreconditionedConjugateGradientSolver, isDegenerateDot } from '../src/linalg/linalg.js';

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

describe( 'isDegenerateDot', () => {

	it( 'flags exactly 0 as degenerate for any scale', () => {

		expect( isDegenerateDot( 0, 65536 ) ).toBe( true );

	} );

	it( 'flags a value smaller than half the quantization step as degenerate', () => {

		// half-step is 0.5/65536 ~= 7.63e-6 -- anything strictly under that
		// couldn't have been read back as a nonzero quantized value anyway.
		expect( isDegenerateDot( 1e-8, 65536 ) ).toBe( true );
		expect( isDegenerateDot( - 1e-8, 65536 ) ).toBe( true );

	} );

	it( 'does not flag a value at least half the quantization step', () => {

		expect( isDegenerateDot( 1 / 65536, 65536 ) ).toBe( false );
		expect( isDegenerateDot( - 1 / 65536, 65536 ) ).toBe( false );
		expect( isDegenerateDot( 1, 65536 ) ).toBe( false );

	} );

	it( 'scales its threshold with the atomicScale argument', () => {

		// a coarser (smaller) scale has a wider quantization floor, so the
		// same value can be degenerate there yet perfectly resolvable once
		// scale is large enough to shrink the floor below it.
		expect( isDegenerateDot( 1e-4, 64 ) ).toBe( true );
		expect( isDegenerateDot( 1e-4, 1e9 ) ).toBe( false );

	} );

} );
