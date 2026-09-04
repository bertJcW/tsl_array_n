// Only verifies that the kernel graph builds successfully and that a
// callable dispatcher is returned -- doesn't actually call the dispatcher
// (that needs a real renderer set up by tsl_array_n.init(), which the
// vitest/Node environment doesn't have); the actual diffusion/copy numeric
// behavior has to be verified with toArray() in a live example.

import { describe, it, expect } from 'vitest';
import * as tsl_array_n from 'tsl_array_n';
import { createCopyKernel2, createExtrapolateToRegion2 } from '../src/grid/array_utils.js';

describe( 'array_utils', () => {

	it( 'createCopyKernel2 returns a reusable dispatcher', () => {

		const src = tsl_array_n.array2( 'float', 4, 4 );
		const dst = tsl_array_n.array2( 'float', 4, 4 );

		const copy = createCopyKernel2( src, dst );

		expect( typeof copy ).toBe( 'function' );

	} );

	it( 'createCopyKernel2 infers shape from src when not given explicitly', () => {

		const src = tsl_array_n.array2( 'float', 3, 5 );
		const dst = tsl_array_n.array2( 'float', 3, 5 );

		expect( () => createCopyKernel2( src, dst ) ).not.toThrow();

	} );

	it( 'createExtrapolateToRegion2 returns a run() function, in-place (output === input)', () => {

		const field = tsl_array_n.array2( 'float', 4, 4 );
		const valid = tsl_array_n.array2( 'int', 4, 4 );

		const run = createExtrapolateToRegion2( field, valid, field );

		expect( typeof run ).toBe( 'function' );

	} );

	it( 'createExtrapolateToRegion2 returns a run() function, out-of-place (separate output field)', () => {

		const input = tsl_array_n.array2( 'float', 4, 4 );
		const valid = tsl_array_n.array2( 'int', 4, 4 );
		const output = tsl_array_n.array2( 'float', 4, 4 );

		const run = createExtrapolateToRegion2( input, valid, output );

		expect( typeof run ).toBe( 'function' );

	} );

} );
