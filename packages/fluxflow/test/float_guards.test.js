// Tests for src/float_guards.js -- the non-finite detection this library's
// safety nets depend on.
//
// The behaviour these guards exist to work around only manifests on a GPU,
// and this suite has no GPU. So it tests the two things that CAN be checked
// without one, both of which would have caught the original bug:
//
//   1. The bit-pattern predicate itself, as a plain-JS mirror. If the mask
//      is wrong the GPU version is wrong in exactly the same way, and that
//      is checkable here against Number.isFinite over a table that includes
//      the values a float32 pipeline actually produces.
//   2. That no source file has gone back to `x != x`. This reads like a
//      lint rule rather than a test, and it is deliberate: the idiom is
//      what every WGSL reference recommends, it looks obviously correct,
//      and it silently does nothing here. Someone will write it again. The
//      failure it caused (a liquid collapsing to a point with every
//      diagnostic reporting healthy) cost far more to find than this test
//      costs to keep.

import { describe, it, expect } from 'vitest';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

const EXPONENT_MASK = 0x7f800000;

// Plain-JS mirror of isNonFinite's GPU expression, operating on the same
// float32 bit pattern the shader sees.
const f32 = new Float32Array( 1 );
const u32 = new Uint32Array( f32.buffer );

function isNonFiniteByBits( value ) {

	f32[ 0 ] = value;
	return ( u32[ 0 ] & EXPONENT_MASK ) === EXPONENT_MASK;

}

describe( 'float_guards: the bit-pattern non-finite test', () => {

	const table = [
		[ 'NaN', NaN ],
		[ '+Infinity', Infinity ],
		[ '-Infinity', - Infinity ],
		[ 'zero', 0 ],
		[ 'negative zero', - 0 ],
		[ 'one', 1 ],
		[ 'minus one', - 1 ],
		[ 'a typical pressure', 6.83 ],
		[ 'a large but finite value', 1e30 ],
		[ 'float32 max', 3.4028234663852886e38 ],
		[ 'a subnormal', 1e-42 ],
		[ 'the smallest normal float32', 1.1754943508222875e-38 ]
	];

	for ( const [ name, value ] of table ) {

		it( `agrees with Number.isFinite for ${ name }`, () => {

			// float32 max must survive: it is finite, and a guard that
			// flagged it would reject legitimate solves. 1e39 would not
			// survive -- it overflows to Infinity in float32 -- which is a
			// real property of the pipeline, not a bug in the predicate.
			expect( isNonFiniteByBits( value ) ).toBe( ! Number.isFinite( value ) );

		} );

	}

	it( 'flags a value that only becomes non-finite once narrowed to float32', () => {

		// Finite as a JS double, Infinity as a float32. Anything stored in a
		// grid goes through this narrowing, so the predicate must judge the
		// narrowed value -- which it does, because it reads the float32 bits.
		expect( Number.isFinite( 1e39 ) ).toBe( true );
		expect( isNonFiniteByBits( 1e39 ) ).toBe( true );

	} );

	it( 'flags every NaN encoding, not just the canonical quiet NaN', () => {

		// A GPU can produce a signalling NaN or a NaN with an arbitrary
		// payload. The exponent mask ignores both the sign and the mantissa,
		// so all of them are caught -- unlike a payload-sensitive test.
		for ( const bits of [ 0x7fc00000, 0xffc00000, 0x7f800001, 0xffbfffff ] ) {

			u32[ 0 ] = bits;
			expect( Number.isNaN( f32[ 0 ] ) ).toBe( true );
			expect( ( u32[ 0 ] & EXPONENT_MASK ) === EXPONENT_MASK ).toBe( true );

		}

	} );

} );

describe( 'float_guards: the negated-bound form', () => {

	// isNonFiniteOrAbove writes its magnitude test as !(abs(x) <= limit)
	// rather than abs(x) > limit. In IEEE arithmetic both are false for a
	// NaN operand -- but negating the first turns that false into "bad",
	// which is the answer we want, while the second reports "fine".
	const limit = 100;

	it( 'asserted and negated bounds agree on every finite value', () => {

		for ( const v of [ 0, 1, - 1, 99.9, 100, - 100, 100.1, - 100.1, 1e6 ] ) {

			expect( Math.abs( v ) > limit ).toBe( ! ( Math.abs( v ) <= limit ) );

		}

	} );

	it( 'and disagree on NaN, which is the entire point', () => {

		expect( Math.abs( NaN ) > limit ).toBe( false );
		expect( ! ( Math.abs( NaN ) <= limit ) ).toBe( true );

	} );

} );

describe( 'float_guards: no source file uses the broken NaN idiom', () => {

	function jsFilesUnder( dir ) {

		const out = [];

		for ( const entry of readdirSync( dir ) ) {

			const full = join( dir, entry );
			if ( statSync( full ).isDirectory() ) out.push( ...jsFilesUnder( full ) );
			else if ( entry.endsWith( '.js' ) ) out.push( full );

		}

		return out;

	}

	// `foo.notEqual( foo )` / `foo( i ).notEqual( foo( i ) )` -- a node
	// compared against a textually identical node, which is only ever
	// written as a NaN test.
	const selfNotEqual = /([A-Za-z_$][\w$]*(?:\s*\([^()]*\))?)\s*\.notEqual\(\s*\1\s*\)/;

	for ( const file of jsFilesUnder( new URL( '../src', import.meta.url ).pathname ) ) {

		it( `${ file.split( '/src/' )[ 1 ] } does not test for NaN with self-inequality`, () => {

			const lines = readFileSync( file, 'utf8' ).split( '\n' );
			const offenders = lines
				.map( ( line, i ) => ( { line, n: i + 1 } ) )
				.filter( ( { line } ) => ! line.trim().startsWith( '//' ) )
				.filter( ( { line } ) => selfNotEqual.test( line ) );

			expect(
				offenders.map( ( o ) => `${ o.n }: ${ o.line.trim() }` ),
				'use isNonFinite() from src/float_guards.js -- `x != x` compiles to a constant false on the backends this was measured on'
			).toEqual( [] );

		} );

	}

} );
