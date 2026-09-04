import { describe, it, expect } from 'vitest';
import { float } from 'three/tsl';
import { isInsideSdf, fractionInsideSdf } from '../src/grid/level_set_utils.js';

function expectNode( value ) {

	expect( value ).toBeTruthy();
	expect( value.isNode ).toBe( true );

}

describe( 'level_set_utils', () => {

	it( 'isInsideSdf builds a boolean node', () => {

		expectNode( isInsideSdf( float( -1 ) ) );

	} );

	it( 'fractionInsideSdf builds without throwing for both-inside/both-outside/mixed phi pairs', () => {

		expectNode( fractionInsideSdf( float( -1 ), float( -2 ) ) ); // both inside
		expectNode( fractionInsideSdf( float( 1 ), float( 2 ) ) );   // both outside
		expectNode( fractionInsideSdf( float( -1 ), float( 1 ) ) );  // straddling the surface
		expectNode( fractionInsideSdf( float( 1 ), float( -1 ) ) );  // straddling, opposite order

	} );

} );
