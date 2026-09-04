// A smoke test, not a numeric-correctness check: this sandbox environment
// (see 00-grid-math/main.js's header comment) currently can't even read
// back the right value for a basic "read a different field inside a kernel"
// operation, so even if the numbers read back after constrainVelocity()
// runs here "look reasonable", that still isn't real numeric verification --
// this only confirms that the full pipeline (collider rasterization ->
// boundary-condition-solver construction -> constrainVelocity() actually
// dispatching a whole set of kernels) runs end to end without throwing,
// which has value on its own (it catches API misuse, wrong argument counts,
// misspelled method names). Real numeric correctness has to wait for a real
// WebGPU environment.

import * as tsl_array_n from 'tsl_array_n';
import { grid } from 'fluxflow';

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

	const nx = 8, ny = 8;
	const velocity = grid.createFaceCenteredGrid2( nx, ny, 1, 1, -4, -4 );

	// a small square collider, placed left-of-center in the domain, side length 2
	const colliderSquare = [ [ -3, -1 ], [ -1, -1 ], [ -1, 1 ], [ -3, 1 ] ];
	const collider = grid.createSDFStaticCollider2( nx, ny, 1, 1, -4, -4 );

	try {

		collider.addPolygon( colliderSquare );
		log( 'SDFStaticCollider2.addPolygon()', true );

	} catch ( error ) {

		log( 'SDFStaticCollider2.addPolygon()', false, error.message );
		throw error;

	}

	let solver;

	try {

		// The constructor immediately dispatches buildBlockMarker() internally
		// (because a real collider was passed in) -- this is the part vitest
		// can't test; real live verification starts here
		solver = grid.createGridBlockedBoundaryConditionSolver2( velocity, nx, ny, 1, 1, -4, -4, collider );
		log( 'createGridBlockedBoundaryConditionSolver2() with a real collider', true, 'buildBlockMarker() dispatched during construction without throwing' );

	} catch ( error ) {

		log( 'createGridBlockedBoundaryConditionSolver2() with a real collider', false, error.message );
		throw error;

	}

	// Seed the velocity field with some initial values (a uniform rightward
	// flow), so constrainVelocity()'s no-flux/blocked-boundary branches
	// actually have nonzero input to process instead of an all-zero scenario
	velocity.dataU.fromArray( new Float32Array( velocity.dataSizeU[ 0 ] * velocity.dataSizeU[ 1 ] ).fill( 1 ) );
	velocity.dataV.fromArray( new Float32Array( velocity.dataSizeV[ 0 ] * velocity.dataSizeV[ 1 ] ).fill( 0 ) );

	try {

		solver.constrainVelocity();
		log( 'constrainVelocity() — full dispatch (fill markers, mark+project, extrapolate, no-flux, blocked boundary, domain walls)', true, 'ran without throwing' );

	} catch ( error ) {

		log( 'constrainVelocity()', false, error.message );
		throw error;

	}

	try {

		const uData = await velocity.dataU.toArray();
		const vData = await velocity.dataV.toArray();
		log(
			'readback (NOT a correctness check, see file header)',
			true,
			`dataU sample: [${ Array.from( uData ).slice( 0, 4 ) }...], dataV sample: [${ Array.from( vData ).slice( 0, 4 ) }...]`
		);

	} catch ( error ) {

		log( 'readback', false, error.message );

	}

	// While at it, also verify that setCollider(null, ...) can remove the
	// collider and fall back to the "domain-boundary-only" path without throwing
	try {

		solver.setCollider( null, [ nx, ny ], [ 1, 1 ], [ -4, -4 ] );
		solver.constrainVelocity();
		log( 'setCollider(null, ...) then constrainVelocity() again', true, 'falls back to domain-boundary-only path without throwing' );

	} catch ( error ) {

		log( 'setCollider(null, ...) then constrainVelocity() again', false, error.message );

	}

} catch ( error ) {

	log( 'failed', false, error.message );

}
