// Demonstrates -- or rather measures -- what src/float_guards.js is for.
//
// Every other example in this directory shows the library doing something.
// This one exists because a piece of the library could not be justified
// without a measurement, and a measurement nobody can re-run is a claim, not
// a result. float_guards.js replaced the standard WGSL NaN idiom with a
// bit-pattern test on the strength of the table this page prints; that table
// is device behaviour, not spec behaviour, so it can differ on your GPU and
// you should be able to check.
//
// *** What is being tested ***
//
// Core WGSL has no isnan()/isinf(). The documented replacement is
// self-inequality: a NaN is the only value not equal to itself, so `x != x`
// should be true exactly for NaN. But WGSL's floating point is explicitly
// permissive about non-finite values -- an implementation may assume they
// never arise -- which leaves a compiler free to fold `x != x` to a constant
// false. Nothing warns you; the guard simply stops guarding.
//
// Four candidate tests, run as one kernel over the same inputs:
//
//   x != x            the documented idiom
//   abs(x) > limit    an asserted out-of-range bound
//   !(abs(x) <= limit) the same bound negated
//   bit test          exponent field all ones, via floatBitsToUint
//
// The last two are what float_guards.js uses. The negated bound works for a
// reason worth internalising: BOTH bound forms return false for a NaN, since
// an unordered comparison is false either way -- but only for the negated one
// does that false lead to the right conclusion. Which way round you write a
// bound decides whether it survives a NaN.
//
// *** What it cost to find out ***
//
// grid_pressure_solver2.js snapshots pressure before every solve and reverts
// if the result looks implausible. With its NaN half inert, a solve that came
// back NaN across the whole fluid region counted zero bad cells and was
// accepted; the correction multiplied it into the velocity field, the
// boundary clamp turned NaN into its own bound (clamp(NaN,-100,100) yields
// -100, not NaN), and the advection clamp folded every particle into the
// domain corner. examples/20-flip-dam-break/ died at frame 124 that way, with
// converged and rejected both reporting healthy at frame 122.

import * as tsl_array_n from 'tsl_array_n';
import { float, int, uint, abs, floatBitsToUint } from 'three/tsl';

const statusEl = document.querySelector( '#status' );
const tableEl = document.querySelector( '#results' );
const verdictEl = document.querySelector( '#verdict' );
const perfEl = document.querySelector( '#perf' );

function status( text, isErr ) {

	statusEl.textContent = text;
	statusEl.className = isErr ? 'err' : '';

}

const LIMIT = 100;

// Built from raw bit patterns rather than from JS expressions, so that what
// reaches the GPU is exactly the float32 encoding intended -- a source-level
// `0 / 0` or `1e39` would be constant-folded by the JS engine first, and a
// quiet NaN written as `NaN` gives no control over the payload.
const CASES = [
	{ name: 'NaN', bits: 0x7fc00000, nan: true, nonFinite: true, aboveLimit: false },
	{ name: 'NaN (payload)', bits: 0x7f800001, nan: true, nonFinite: true, aboveLimit: false },
	{ name: '+Infinity', bits: 0x7f800000, nan: false, nonFinite: true, aboveLimit: true },
	{ name: '-Infinity', bits: 0xff800000, nan: false, nonFinite: true, aboveLimit: true },
	{ name: '1e30', bits: 0x7149f2ca, nan: false, nonFinite: false, aboveLimit: true },
	{ name: '150.0', bits: 0x43160000, nan: false, nonFinite: false, aboveLimit: true },
	{ name: '5.0', bits: 0x40a00000, nan: false, nonFinite: false, aboveLimit: false },
	{ name: '-0.0', bits: 0x80000000, nan: false, nonFinite: false, aboveLimit: false },
	{ name: '1e-42 (subnormal)', bits: 0x00000012, nan: false, nonFinite: false, aboveLimit: false }
];

// Each predicate is scored against what IT claims, not against one shared
// answer. Getting this wrong makes the table lie in a flattering direction:
// score the bit test against "unsafe to use" and it reports two failures for
// 1e30 and 150.0, which are finite and which it never claimed to catch.
//
// `aboveLimit` is therefore false for NaN. A NaN is not "greater than 100";
// it is unordered with 100, and pretending otherwise would build the answer
// being tested into the expectation. What makes a NaN unsafe is that it is
// non-finite, which is a different column.
const UNSAFE = ( c ) => c.nonFinite || c.aboveLimit;

const TESTS = [
	// The two component predicates, each judged on its own terms.
	{
		label: 'x != x',
		note: 'wants: is NaN',
		expected: ( c ) => c.nan,
		build: ( v ) => v.notEqual( v )
	},
	{
		label: 'bit test',
		note: 'wants: is non-finite',
		expected: ( c ) => c.nonFinite,
		build: ( v ) => floatBitsToUint( v ).bitAnd( uint( 0x7f800000 ) ).equal( uint( 0x7f800000 ) )
	},
	// The two compound guards, which is the comparison that decides the code.
	// Same intent, same limit, both judged against "unsafe to use".
	{
		label: `OLD: x != x || abs(x) > ${ LIMIT }`,
		note: 'wants: unsafe to use',
		expected: UNSAFE,
		build: ( v ) => v.notEqual( v ).or( abs( v ).greaterThan( float( LIMIT ) ) )
	},
	{
		label: `NEW: bits || !(abs(x) <= ${ LIMIT })`,
		note: 'wants: unsafe to use',
		expected: UNSAFE,
		build: ( v ) => floatBitsToUint( v ).bitAnd( uint( 0x7f800000 ) ).equal( uint( 0x7f800000 ) )
			.or( abs( v ).lessThanEqual( float( LIMIT ) ).not() )
	}
];

const OLD_GUARD = 2;
const NEW_GUARD = 3;

try {

	const renderer = await tsl_array_n.init( { canvas: document.createElement( 'canvas' ), allowFallback: true } );

	const N = CASES.length;

	const source = tsl_array_n.arrayN( 'float', N );
	source.fromArray( new Float32Array( Uint32Array.from( CASES.map( ( c ) => c.bits ) ).buffer ) );

	// One output array per test. Deliberately not a single [N, TESTS.length]
	// array: a 2D buffer's element order is the library's business, and
	// getting it wrong here would silently transpose the results into a table
	// that looks meaningful and is not. One flat array per column has no such
	// ambiguity.
	const outputs = TESTS.map( () => {

		const a = tsl_array_n.arrayN( 'int', N );
		a.fromArray( new Int32Array( N ) );
		return a;

	} );

	const probe = tsl_array_n.kernel( N, ( i ) => {

		const v = source( i );

		for ( let t = 0; t < TESTS.length; t ++ ) {

			outputs[ t ]( i ).assign( TESTS[ t ].build( v ).select( int( 1 ), int( 0 ) ) );

		}

	} );

	const started = performance.now();
	probe();

	const columns = await Promise.all( outputs.map( ( a ) => a.toArray() ) );
	const elapsed = performance.now() - started;

	// ---------------------------------------------------------------- render

	const header = document.createElement( 'tr' );
	header.innerHTML = '<th>value</th>' +
		TESTS.map( ( t ) => `<th>${ t.label }<br /><span style="font-weight:400;opacity:0.6">${ t.note }</span></th>` ).join( '' ) +
		'<th>unsafe?</th>';
	tableEl.appendChild( header );

	const wrong = TESTS.map( () => 0 );

	CASES.forEach( ( c, i ) => {

		const row = document.createElement( 'tr' );
		let html = `<td>${ c.name }</td>`;

		TESTS.forEach( ( t, k ) => {

			const got = columns[ k ][ i ] === 1;
			const ok = got === t.expected( c );
			if ( ! ok ) wrong[ k ] ++;
			html += `<td class="${ ok ? 'ok' : 'bad' }">${ got ? 1 : 0 }${ ok ? '' : ' ✗' }</td>`;

		} );

		html += `<td>${ UNSAFE( c ) ? 1 : 0 }</td>`;
		row.innerHTML = html;
		tableEl.appendChild( row );

	} );

	console.log(
		'fluxflow float-guard probe | ' +
		TESTS.map( ( t, k ) => `${ t.label }: ${ wrong[ k ] === 0 ? 'sound' : `${ wrong[ k ] } wrong` }` ).join( ' | ' )
	);

	// The verdict is about the two compound guards, because those are the
	// actual before/after in the library. The component columns above are
	// there to show WHERE the old one goes wrong, not to be scored on.
	const oldSound = wrong[ OLD_GUARD ] === 0;
	const newSound = wrong[ NEW_GUARD ] === 0;
	const idiomSound = wrong[ 0 ] === 0;

	if ( ! newSound ) {

		verdictEl.className = 'warn';
		verdictEl.textContent =
			`The guard src/float_guards.js relies on is wrong on ${ wrong[ NEW_GUARD ] } of ` +
			`${ CASES.length } values here. Every non-finite guard in this library is unsound on ` +
			'this device. Please report it with your GPU and browser — this is not a case the ' +
			'library was built to handle, and no amount of tuning a scene works around it.';

	} else if ( oldSound ) {

		verdictEl.className = 'good';
		verdictEl.textContent =
			'Both guards are sound on this device' +
			( idiomSound ? ', including the bare x != x idiom' : '' ) +
			'. That is not the behaviour this library was developed against. It does not make the ' +
			'old form safe to use — it makes this device one that happens not to fold it away, and ' +
			'the same code will still fail silently elsewhere.';

	} else {

		verdictEl.className = 'warn';
		verdictEl.textContent =
			`The old guard is wrong on ${ wrong[ OLD_GUARD ] } of ${ CASES.length } values here; the ` +
			'new one is sound. This is exactly the situation src/float_guards.js exists for, and it ' +
			'is what collapsed examples/20-flip-dam-break/ at frame 124. Any NaN check you write ' +
			'yourself with the old form will compile, run, and protect nothing.';

	}

	status( `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' } — ${ N } values × ${ TESTS.length } tests` );
	perfEl.textContent = `one dispatch, ${ elapsed.toFixed( 1 ) } ms including readback`;

} catch ( error ) {

	status( `error: ${ error.message }`, true );
	console.error( error );

}
