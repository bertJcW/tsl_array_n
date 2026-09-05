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
import { createMultigridPreconditioner } from '../src/linalg/multigrid.js';

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

} );
