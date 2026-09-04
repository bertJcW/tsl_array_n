import * as tsl_array_n from 'tsl_array_n';

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

	const grid = tsl_array_n.array2( 'float', 4, 4 );
	const width = grid.shape[ 0 ];

	// func(): small device-side helper, forwarded to Fn() — usable from inside a kernel.
	// NOTE: Fn()-wrapped functions always receive a single destructured array param,
	// not separate positional params — see README's "Known limitations" note on func().
	const flatten = tsl_array_n.func( ( [ x, y ] ) => x.add( y.mul( width ) ) );

	// kernel(): auto-parallel over the field's full 2D shape, i/j come back already unflattened
	const fill = tsl_array_n.kernel( grid.shape, ( x, y ) => {

		grid( x, y ).assign( flatten( x, y ).toFloat() );

	} );

	try {

		fill();

		const readback = Array.from( await grid.toArray() );
		const expected = Array.from( { length: 16 }, ( _, i ) => i );
		const matches = readback.length === 16 && readback.every( ( v, i ) => v === expected[ i ] );

		log(
			'kernel() + func() fill, then toArray()',
			matches,
			matches
				? `[${ readback }] — closes the Step 2 "empty readback" gap now that a kernel has touched the buffer`
				: `got [${ readback }], expected [${ expected }]`
		);

	} catch ( error ) {

		log(
			'kernel() + func() fill, then toArray()',
			false,
			`${ error.message } — likely this backend (${ renderer.backend?.constructor?.name }) doesn't support real compute (WebGL2 fallback has no compute shaders; see init()'s allowFallback warning)`
		);

	}

	// "case A" from the design discussion: dispatch over just one axis of a 2D field
	const row = tsl_array_n.array2( 'float', 4, 4 );

	try {

		const fillFirstRow = tsl_array_n.kernel( row.shape[ 0 ], ( x ) => {

			row( x, 0 ).assign( x.toFloat() );

		} );

		fillFirstRow();

		const rowData = Array.from( await row.toArray() );
		const rowMatches = [ 0, 1, 2, 3, 0, 0, 0, 0 ].every( ( v, i ) => rowData[ i ] === v );

		log(
			'kernel(shape[0], (x) => …) single-axis dispatch',
			rowMatches,
			`row 0 = [${ rowData.slice( 0, 4 ) }], row 1 (never dispatched) = [${ rowData.slice( 4, 8 ) }]`
		);

	} catch ( error ) {

		log( 'kernel(shape[0], (x) => …) single-axis dispatch', false, error.message );

	}

	try {

		tsl_array_n.kernel( grid.shape, ( onlyOneParam ) => {} ); // eslint-disable-line no-unused-vars
		log( 'kernel() rejects fn arity mismatch', false, 'expected a throw' );

	} catch {

		log( 'kernel() rejects fn arity mismatch', true );

	}

} catch ( error ) {

	log( 'failed', false, error.message );

}
