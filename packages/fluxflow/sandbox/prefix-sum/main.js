// Does createInclusivePrefixSum compute a prefix sum, and does it compute
// the same one every time?
//
// The atomic cursor this replaces got an isolated real-hardware test before
// anything relied on it, and that test checked the wrong property: every
// slot written exactly once, which is true of a nondeterministic assignment
// too. So this page checks both, and keeps them apart.
//
//   CORRECTNESS -- the scan matches a reference computed in JS, on inputs
//   chosen to break the things that break scans: a length that is not a
//   power of two, a length of exactly a power of two, a length of 1, all
//   ones, all zeros, and a single flag at each end.
//
//   DETERMINISM -- the thing the cursor could not do. The same input is
//   scanned repeatedly and every run must be bit-identical. A cursor passes
//   correctness and fails this.
//
//   RANK ORDER -- what the resampler actually needs: the rank of a flagged
//   element must be the count of flagged elements before it, so ranks come
//   out in index order with no gaps and no repeats.

import * as tsl_array_n from 'tsl_array_n';
import { createInclusivePrefixSum } from '../../src/linalg/prefix_sum.js';

const out = document.querySelector( '#out' );
const lines = [];
const log = ( text ) => { lines.push( text ); out.textContent = lines.join( '\n' ); };

let passed = 0, failed = 0;
const check = ( name, ok, detail ) => {

	log( `${ ok ? '✓' : '✗' } ${ name }${ detail ? ' — ' + detail : '' }` );
	ok ? passed ++ : failed ++;

};

function reference( values ) {

	const result = new Int32Array( values.length );
	let running = 0;
	for ( let i = 0; i < values.length; i ++ ) { running += values[ i ]; result[ i ] = running; }
	return result;

}

async function scanOnce( values ) {

	const source = tsl_array_n.arrayN( 'int', [ values.length ] );
	source.fromArray( new Int32Array( values ) );

	const { kernels, result } = createInclusivePrefixSum( values.length, source );
	const batch = tsl_array_n.createBatch( kernels );
	batch();

	return Array.from( await result.toArray() );

}

try {

	const renderer = await tsl_array_n.init( { allowFallback: true } );
	const backend = renderer.backend?.constructor?.name ?? 'unknown';
	log( `backend: ${ backend }` );
	if ( backend !== 'WebGPUBackend' ) log( '  (WebGL2 fallback -- nothing below is evidence about real hardware)' );
	log( '' );

	// ---- correctness, on the lengths and patterns that break scans ----

	const cases = [
		[ 'length 1', [ 1 ] ],
		[ 'length 2', [ 0, 1 ] ],
		[ 'all zeros, length 64', new Array( 64 ).fill( 0 ) ],
		[ 'all ones, length 64', new Array( 64 ).fill( 1 ) ],
		[ 'power of two, length 1024', Array.from( { length: 1024 }, ( _, i ) => ( i % 3 === 0 ? 1 : 0 ) ) ],
		[ 'not a power of two, length 1000', Array.from( { length: 1000 }, ( _, i ) => ( i % 7 === 0 ? 1 : 0 ) ) ],
		[ 'one flag at index 0, length 500', Array.from( { length: 500 }, ( _, i ) => ( i === 0 ? 1 : 0 ) ) ],
		[ 'one flag at the last index, length 500', Array.from( { length: 500 }, ( _, i ) => ( i === 499 ? 1 : 0 ) ) ],
		[ 'counts, not flags, length 300', Array.from( { length: 300 }, ( _, i ) => i % 5 ) ],
		[ 'a resampler-sized length, 30000', Array.from( { length: 30000 }, ( _, i ) => ( i % 11 === 0 ? 1 : 0 ) ) ]
	];

	for ( const [ name, values ] of cases ) {

		const got = await scanOnce( values );
		const want = reference( values );
		let firstBad = -1;
		for ( let i = 0; i < values.length; i ++ ) if ( got[ i ] !== want[ i ] ) { firstBad = i; break; }

		check(
			`scan matches the reference — ${ name }`,
			firstBad < 0,
			firstBad < 0
				? `total ${ got[ got.length - 1 ] }`
				: `first difference at ${ firstBad }: got ${ got[ firstBad ] }, expected ${ want[ firstBad ] }`
		);

	}

	log( '' );

	// ---- determinism: the property the atomic cursor could not have ----

	{
		const values = Array.from( { length: 30000 }, ( _, i ) => ( ( i * 2654435761 ) % 97 < 40 ? 1 : 0 ) );
		const runs = [];
		for ( let r = 0; r < 8; r ++ ) runs.push( await scanOnce( values ) );

		let identical = true, where = -1;
		for ( let r = 1; r < runs.length && identical; r ++ ) {

			for ( let i = 0; i < values.length; i ++ ) if ( runs[ r ][ i ] !== runs[ 0 ][ i ] ) { identical = false; where = i; break; }

		}

		check(
			'eight scans of the same input are bit-identical',
			identical,
			identical ? `30000 elements, total ${ runs[ 0 ][ 29999 ] }, 8 runs` : `runs differ at index ${ where }`
		);
	}

	log( '' );

	// ---- rank order: what the resampler needs from it ----

	{
		const flags = Array.from( { length: 4096 }, ( _, i ) => ( ( i * 7 ) % 13 < 5 ? 1 : 0 ) );
		const scanned = await scanOnce( flags );

		// rank = inclusive scan - 1, for flagged elements only
		const ranks = [];
		for ( let i = 0; i < flags.length; i ++ ) if ( flags[ i ] === 1 ) ranks.push( { i, rank: scanned[ i ] - 1 } );

		const total = scanned[ flags.length - 1 ];
		let dense = ranks.length === total;
		for ( let r = 0; r < ranks.length && dense; r ++ ) if ( ranks[ r ].rank !== r ) dense = false;

		check(
			'flagged elements rank 0..n-1 in index order, no gaps, no repeats',
			dense,
			dense
				? `${ ranks.length } flagged of ${ flags.length }, ranks 0..${ ranks.length - 1 }`
				: `${ ranks.length } flagged but total reads ${ total }`
		);

		// And the property that makes it reproducible rather than merely
		// unique: a lower index always gets a lower rank. An atomic cursor
		// gives each flagged element a unique rank too, in whatever order
		// the threads arrived.
		let monotonic = true;
		for ( let r = 1; r < ranks.length && monotonic; r ++ ) if ( ranks[ r ].rank <= ranks[ r - 1 ].rank || ranks[ r ].i <= ranks[ r - 1 ].i ) monotonic = false;
		check( 'rank is increasing in the index, which is what makes it reproducible', monotonic );
	}

	log( '' );
	log( failed === 0 ? `all ${ passed } checks passed` : `${ failed } of ${ passed + failed } checks FAILED` );

} catch ( error ) {

	log( `\nthrew: ${ error.message }` );
	console.error( error );

}
