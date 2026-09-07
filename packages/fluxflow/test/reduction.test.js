// Structural tests only, matching linalg.test.js/multigrid.test.js's own
// style: real GPU dispatch (the atomicMax reduction actually running) needs
// a live WebGPU renderer, so `.read()` is never invoked here -- only that
// construction (building the reduction kernel(s) for one field or several
// differently-shaped ones) doesn't throw and returns the expected shape.

import { describe, it, expect } from 'vitest';
import * as tsl_array_n from 'tsl_array_n';
import { createMaxAbsReducer } from '../src/linalg/reduction.js';

describe( 'createMaxAbsReducer', () => {

	it( 'constructs for a single field without throwing', () => {

		const field = tsl_array_n.arrayN( 'float', [ 8, 8 ] );

		expect( () => createMaxAbsReducer( field ) ).not.toThrow();

	} );

	it( 'constructs for an array of differently-shaped fields (mirroring dataU/dataV) without throwing', () => {

		const dataU = tsl_array_n.arrayN( 'float', [ 9, 8 ] ); // (resX+1) x resY
		const dataV = tsl_array_n.arrayN( 'float', [ 8, 9 ] ); // resX x (resY+1)

		expect( () => createMaxAbsReducer( [ dataU, dataV ] ) ).not.toThrow();

	} );

	it( 'returns an object exposing read as an async function', () => {

		const field = tsl_array_n.arrayN( 'float', [ 8, 8 ] );
		const reducer = createMaxAbsReducer( field );

		expect( typeof reducer.read ).toBe( 'function' );

	} );

	it( 'accepts a custom atomicScale option without throwing', () => {

		const field = tsl_array_n.arrayN( 'float', [ 8, 8 ] );

		expect( () => createMaxAbsReducer( field, { atomicScale: 1024 } ) ).not.toThrow();

	} );

} );
