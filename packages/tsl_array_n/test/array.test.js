import { describe, it, expect } from 'vitest';
import { normalizeShape, computeStrides, flattenIndex, arrayN, array0, array2 } from '../src/array.js';

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

	it( 'accepts an empty shape as 0-D (array0)', () => {

		expect( normalizeShape( [] ) ).toEqual( [] );

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

describe( 'arrayN / array2 — callable field', () => {

	it( 'returns a callable function, not a plain object', () => {

		const field = arrayN( 'float', 4 );
		expect( typeof field ).toBe( 'function' );

	} );

	it( 'still exposes shape/count/type/node/at/toArray/fromArray', () => {

		const field = array2( 'float', 4, 8 );

		expect( field.shape ).toEqual( [ 4, 8 ] );
		expect( field.count ).toBe( 32 );
		expect( field.type ).toBe( 'float' );
		expect( field.node ).toBeDefined();
		expect( field.at( 3, 7 ) ).toBe( 3 + 7 * 4 );
		expect( typeof field.toArray ).toBe( 'function' );
		expect( typeof field.fromArray ).toBe( 'function' );

	} );

	it( 'calling the field itself builds a GPU element node (no renderer needed to construct)', () => {

		const field = array2( 'float', 4, 8 );
		const element = field( 1, 2 );

		expect( element ).toBeDefined();
		expect( element.isNode ).toBe( true );

	} );

	it( 'rejects calling the field with the wrong number of indices', () => {

		const field = array2( 'float', 4, 8 );
		expect( () => field( 1 ) ).toThrow();
		expect( () => field( 1, 2, 3 ) ).toThrow();

	} );

} );

describe( 'array0 — 0-D field (ti.field(dtype, shape=()) equivalent)', () => {

	it( 'has an empty shape and a count of 1', () => {

		const scalar = array0( 'float' );
		expect( scalar.shape ).toEqual( [] );
		expect( scalar.count ).toBe( 1 );

	} );

	it( '.at() takes zero indices and returns 0', () => {

		const scalar = array0( 'float' );
		expect( scalar.at() ).toBe( 0 );

	} );

	it( 'is callable with zero arguments and builds a GPU element node', () => {

		const scalar = array0( 'vec2' );
		const element = scalar();
		expect( element ).toBeDefined();
		expect( element.isNode ).toBe( true );

	} );

	it( 'rejects being called with any index', () => {

		const scalar = array0( 'float' );
		expect( () => scalar( 0 ) ).toThrow();
		expect( () => scalar.at( 0 ) ).toThrow();

	} );

	it( 'fromArray()/toArray() still work on a single element', () => {

		const scalar = array0( 'float' );
		expect( typeof scalar.fromArray ).toBe( 'function' );
		expect( typeof scalar.toArray ).toBe( 'function' );

	} );

} );
