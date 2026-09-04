// Structural tests only (same reasoning as grid_math.test.js): building a
// TSL node graph doesn't need a GPU, but reading back real numeric values
// does -- there's no simple hand-computable expected value for noise
// functions the way linear-field gradients had, so real numeric
// verification for this file happens via the live example instead
// (examples/03-noise/), by visually checking the output looks like the
// expected kind of noise.

import { describe, it, expect } from 'vitest';
import { float, vec3, vec4 } from 'three/tsl';
import { permute, taylorInvSqrt, perlinNoise3d, simplexNoise3d, cellular3d } from '../src/noise/noise.js';

function expectNode( value ) {

	expect( value ).toBeTruthy();
	expect( value.isNode ).toBe( true );

}

const P = vec3( 1.5, 2.5, 3.5 );

describe( 'noise helpers', () => {

	it( 'permute builds without throwing for both scalar and vec4 input', () => {

		expectNode( permute( float( 3 ) ) );
		expectNode( permute( vec4( 1, 2, 3, 4 ) ) );

	} );

	it( 'taylorInvSqrt builds without throwing for both scalar and vec4 input', () => {

		expectNode( taylorInvSqrt( float( 0.5 ) ) );
		expectNode( taylorInvSqrt( vec4( 1, 2, 3, 4 ) ) );

	} );

} );

describe( 'perlinNoise3d', () => {

	it( 'builds a node graph without throwing', () => {

		expectNode( perlinNoise3d( P ) );

	} );

} );

describe( 'simplexNoise3d', () => {

	it( 'builds a node graph without throwing', () => {

		expectNode( simplexNoise3d( P ) );

	} );

} );

describe( 'cellular3d', () => {

	it( 'builds a node graph without throwing, returns a vec2 (F1, F2)', () => {

		const result = cellular3d( P );

		expectNode( result );
		expectNode( result.x );
		expectNode( result.y );

	} );

} );
