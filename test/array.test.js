import { describe, it, expect } from 'vitest';
import { normalizeShape, computeStrides, flattenIndex } from '../src/array.js';

describe( 'normalizeShape', () => {

	it( 'wraps a plain number into a 1D shape', () => {

		expect( normalizeShape( 5 ) ).toEqual( [ 5 ] );

	} );

	it( 'passes an array shape through', () => {

		expect( normalizeShape( [ 4, 8, 16 ] ) ).toEqual( [ 4, 8, 16 ] );

	} );

	it( 'rejects non-positive dimensions', () => {

		expect( () => normalizeShape( 0 ) ).toThrow();
		expect( () => normalizeShape( [ 4, - 1 ] ) ).toThrow();

	} );

	it( 'rejects non-integer dimensions', () => {

		expect( () => normalizeShape( 2.5 ) ).toThrow();

	} );

	it( 'rejects an empty shape', () => {

		expect( () => normalizeShape( [] ) ).toThrow();

	} );

} );

describe( 'computeStrides', () => {

	it( 'gives stride 1 for 1D', () => {

		expect( computeStrides( [ 10 ] ) ).toEqual( [ 1 ] );

	} );

	it( 'makes the first dimension fastest-varying in 2D', () => {

		// shape [width, height] -> strides [1, width]
		expect( computeStrides( [ 4, 8 ] ) ).toEqual( [ 1, 4 ] );

	} );

	it( 'extends the same pattern to 3D', () => {

		// shape [w, h, d] -> strides [1, w, w*h]
		expect( computeStrides( [ 4, 8, 2 ] ) ).toEqual( [ 1, 4, 32 ] );

	} );

	it( 'extends to arbitrary N dimensions', () => {

		expect( computeStrides( [ 2, 3, 4, 5 ] ) ).toEqual( [ 1, 2, 6, 24 ] );

	} );

} );

describe( 'flattenIndex', () => {

	it( 'matches plain offset in 1D', () => {

		const strides = computeStrides( [ 10 ] );
		expect( flattenIndex( strides, [ 3 ] ) ).toBe( 3 );

	} );

	it( 'computes i + j*width in 2D', () => {

		const strides = computeStrides( [ 4, 8 ] ); // width 4, height 8
		expect( flattenIndex( strides, [ 0, 0 ] ) ).toBe( 0 );
		expect( flattenIndex( strides, [ 3, 0 ] ) ).toBe( 3 );
		expect( flattenIndex( strides, [ 0, 1 ] ) ).toBe( 4 );
		expect( flattenIndex( strides, [ 3, 7 ] ) ).toBe( 31 ); // last element

	} );

	it( 'computes i + j*w + k*w*h in 3D', () => {

		const strides = computeStrides( [ 4, 8, 2 ] );
		expect( flattenIndex( strides, [ 1, 2, 1 ] ) ).toBe( 1 + 2 * 4 + 1 * 32 );

	} );

	it( 'produces a bijection over the full range in N dimensions', () => {

		const shape = [ 3, 4, 5 ];
		const strides = computeStrides( shape );
		const seen = new Set();

		for ( let k = 0; k < shape[ 2 ]; k ++ ) {

			for ( let j = 0; j < shape[ 1 ]; j ++ ) {

				for ( let i = 0; i < shape[ 0 ]; i ++ ) {

					const index = flattenIndex( strides, [ i, j, k ] );
					expect( index ).toBeGreaterThanOrEqual( 0 );
					expect( index ).toBeLessThan( 3 * 4 * 5 );
					seen.add( index );

				}

			}

		}

		expect( seen.size ).toBe( 3 * 4 * 5 );

	} );

	it( 'rejects an index count that does not match the rank', () => {

		const strides = computeStrides( [ 4, 8 ] );
		expect( () => flattenIndex( strides, [ 1 ] ) ).toThrow();
		expect( () => flattenIndex( strides, [ 1, 2, 3 ] ) ).toThrow();

	} );

} );
