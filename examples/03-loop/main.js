import * as tsl_array_n from 'tsl_array_n';
import { float } from 'three/tsl';

const pre = document.querySelector( '#status pre' );
const lines = [];

function log( label, ok, detail ) {

	const cls = ok ? 'ok' : 'err';
	const mark = ok ? '✓' : '✗';
	lines.push( `<span class="${ cls }">${ mark } ${ label }${ detail ? ' — ' + detail : '' }</span>` );
	pre.innerHTML = lines.join( '\n' );

}

try {

	const renderer = await tsl_array_n.init( { allowFallback: true } );
	log( 'init()', true, `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }` );

	// Loop(): a real per-thread, variable-length inner loop — a prefix sum.
	// Each thread `idx` sums values[0..idx], a DIFFERENT number of iterations per thread.
	// A literal JS for-loop can't do this (its bound would have to be the node `idx`,
	// which JS can't branch on at graph-construction time) — this only works through
	// a genuine GPU-side Loop().
	try {

		const values = tsl_array_n.arrayN( 'float', 5 );
		values.fromArray( [ 1, 2, 3, 4, 5 ] );

		const sums = tsl_array_n.arrayN( 'float', 5 );

		const prefixSum = tsl_array_n.kernel( 5, ( idx ) => {

			const sum = float( 0 ).toVar();

			tsl_array_n.Loop( idx.add( 1 ), ( { i } ) => {

				sum.addAssign( values( i ) );

			} );

			sums( idx ).assign( sum );

		} );

		prefixSum();

		const readback = Array.from( await sums.toArray() );
		const expected = [ 1, 3, 6, 10, 15 ];
		const matches = expected.every( ( v, i ) => readback[ i ] === v );

		log(
			'Loop() — per-thread-variable-length prefix sum',
			matches,
			matches ? `[${ readback }]` : `got [${ readback }], expected [${ expected }]`
		);

	} catch ( error ) {

		log( 'Loop() — per-thread-variable-length prefix sum', false, error.message );

	}

	// Break() / Continue(), each gated by If() — the realistic pattern (an unconditional
	// break/continue on iteration 0 would be a degenerate test).
	try {

		const values = tsl_array_n.arrayN( 'float', 5 );
		values.fromArray( [ 1, 2, 3, 4, 5 ] );

		const oddSum = tsl_array_n.arrayN( 'float', 1 );      // sum of odd-index elements only (Continue)
		const earlyStop = tsl_array_n.arrayN( 'float', 1 );   // running sum, stops once it reaches 6 (Break)

		const scan = tsl_array_n.kernel( 1, ( _unused ) => {

			const a = float( 0 ).toVar();

			tsl_array_n.Loop( 5, ( { i } ) => {

				tsl_array_n.If( i.mod( 2 ).equal( 0 ), () => {

					tsl_array_n.Continue();

				} );

				a.addAssign( values( i ) );

			} );

			oddSum( 0 ).assign( a );

			const b = float( 0 ).toVar();

			tsl_array_n.Loop( 5, ( { i } ) => {

				tsl_array_n.If( b.greaterThanEqual( 6 ), () => {

					tsl_array_n.Break();

				} );

				b.addAssign( values( i ) );

			} );

			earlyStop( 0 ).assign( b );

		} );

		scan();

		const oddResult = ( await oddSum.toArray() )[ 0 ];
		const earlyResult = ( await earlyStop.toArray() )[ 0 ];
		const matches = oddResult === 6 && earlyResult === 6; // values[1]+values[3]=6; values[0]+values[1]+values[2]=6

		log(
			'Break() / Continue() (gated by If())',
			matches,
			`oddSum (skip even index → expect 6) = ${ oddResult }, earlyStop (stop once sum≥6 → expect 6) = ${ earlyResult }`
		);

	} catch ( error ) {

		log( 'Break() / Continue() (gated by If())', false, error.message );

	}

} catch ( error ) {

	log( 'failed', false, error.message );

}
