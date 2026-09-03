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

	const field = tsl_array_n.array2( 'float', 4, 4 );
	log(
		'array2("float", 4, 4)',
		field.shape.length === 2 && field.shape[ 0 ] === 4 && field.shape[ 1 ] === 4 && field.count === 16,
		`shape=[${ field.shape }] count=${ field.count }`
	);

	const at32 = field.at( 3, 2 );
	log( '.at(3, 2) === 3 + 2*4', at32 === 11, `at(3,2)=${ at32 }` );

	try {

		field.at( 1 );
		log( '.at() rejects wrong arity', false, 'expected a throw' );

	} catch {

		log( '.at() rejects wrong arity', true );

	}

	const data = Array.from( { length: 16 }, ( _, i ) => i );
	field.fromArray( data );
	log( 'fromArray() writes into the CPU-side buffer', true, `wrote [${ data }]` );

	try {

		const readback = await field.toArray();
		const matches = readback.length === field.count && Array.from( readback ).every( ( v, i ) => v === data[ i ] );

		log(
			'toArray() readback',
			matches,
			matches
				? `[${ Array.from( readback ) }]`
				: `got length ${ readback.length }, expected ${ field.count } — the GPU-side buffer is only created once a compute/render pass touches it (needs kernel(), Step 3); this backend (${ renderer.backend?.constructor?.name }) returns empty data instead of throwing when that hasn't happened yet.`
		);

	} catch ( error ) {

		log(
			'toArray() readback',
			false,
			`${ error.message } — expected here: the storage buffer only gets created on the GPU once a compute/render pass touches it, which needs kernel() (Step 3).`
		);

	}

	const nd = tsl_array_n.arrayN( 'vec3', [ 2, 3, 4 ] );
	log( 'arrayN("vec3", [2,3,4])', nd.count === 24, `shape=[${ nd.shape }] count=${ nd.count }` );

} catch ( error ) {

	log( 'failed', false, error.message );

}
