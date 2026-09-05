// Verifies createPreconditionedConjugateGradientSolver against the same
// A = diag(1,2,...,N) diagonal operator as examples/04-conjugate-gradient/,
// but this time paired with its own exact Jacobi preconditioner
// M^-1 = diag(1, 1/2, ..., 1/N) -- i.e. M = A exactly, for this diagonal
// system. That's a deliberately strong test case: with a perfect
// preconditioner, z0 = M^-1 @ r0 = A^-1 @ r0, and since p0 = z0,
// Ap0 = A @ p0 = r0, so alpha0 = (r0.z0)/(p0.Ap0) works out to exactly 1
// (both sides are the same dot product), giving x1 = x0 + 1*A^-1@r0 =
// A^-1@b, the exact answer, in a single iteration. A bug in how r.z (as
// opposed to r.r) is threaded through alpha/beta/p would very likely show
// up as divergence or stagnation well before 20 iterations, so checking
// the final x against the known exact answer is a meaningfully strong
// check, same as examples/04-conjugate-gradient/'s.
//
// Known limitation (sandbox environment, not this code): same mechanism,
// same limitation as examples/04-conjugate-gradient/ -- both dot products
// this solver needs (r.r, r.z) run through the same GPU-atomic reduction
// (see src/linalg/linalg.js's header comment), which is WebGPU-only.
// Running this example in this dev sandbox (no real WebGPU adapter, falls
// back to WebGLBackend) confirms exactly that: the console shows two
// separate vertex shader compile errors, one per atomic-dot kernel this
// solver builds (`nodeVarying1 * nodeVarying1` for r.r, `nodeVarying1 *
// nodeVarying2` for r.z, confirming they really are two distinct compiled
// dispatches, not an accidental reuse) -- both fail on the same `'&' :
// syntax error` already confirmed for examples/04-conjugate-gradient/
// (WGSL pointer syntax with no GLSL equivalent). Nothing new to isolate
// here.
//
// CONFIRMED correct on real WebGPU hardware -- reported console output:
//
//   init() — backend: WebGPUBackend
//   createPreconditionedConjugateGradientSolver — A=diag(1..8),
//     M^-1=diag(1..1/8), b=[1,...,1] —
//     x = [1.0000, 0.5000, 0.3333, 0.2500, 0.2000, 0.1667, 0.1429, 0.1250]
//     (expected the same), succeeded=true
//
// An exact match to 4 decimal places -- even tighter than
// examples/04-conjugate-gradient/'s ~1e-4 deviation, consistent with this
// test case's perfect preconditioner converging in a single iteration (as
// predicted above) and therefore accumulating far less atomic
// fixed-point quantization noise than the 8-iteration unpreconditioned
// case.

import * as tsl_array_n from 'tsl_array_n';
import { linalg } from 'fluxflow';

const pre = document.querySelector( '#status pre' );
const lines = [];

function log( label, ok, detail ) {

	const cls = ok ? 'ok' : 'err';
	const mark = ok ? '✓' : '✗';
	lines.push( `<span class="${ cls }">${ mark } ${ label }${ detail ? ' — ' + detail : '' }</span>` );
	pre.innerHTML = lines.join( '\n' );

}

const N = 8;

try {

	const renderer = await tsl_array_n.init( { allowFallback: true } );
	log( 'init()', true, `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }` );

	const b = tsl_array_n.arrayN( 'float', N );
	const x = tsl_array_n.arrayN( 'float', N );

	b.fromArray( new Float32Array( N ).fill( 1 ) ); // b = [1,1,...,1]
	x.fromArray( new Float32Array( N ) ); // x0 = [0,0,...,0]

	// A = diag(1,2,...,N): output(i) = input(i) * (i+1)
	function diagonalOperator( input, output ) {

		return tsl_array_n.kernel( N, ( i ) => {

			output( i ).assign( input( i ).mul( i.add( 1 ).toFloat() ) );

		} );

	}

	// M^-1 = diag(1, 1/2, ..., 1/N): the exact inverse of A above --
	// output(i) = input(i) / (i+1)
	function jacobiPreconditioner( input, output ) {

		return tsl_array_n.kernel( N, ( i ) => {

			output( i ).assign( input( i ).div( i.add( 1 ).toFloat() ) );

		} );

	}

	const solver = linalg.createPreconditionedConjugateGradientSolver( diagonalOperator, jacobiPreconditioner, b, x );

	const succeeded = await solver.solve( 1e-5, 20 );

	const result = Array.from( await x.toArray() );
	const expected = Array.from( { length: N }, ( _, i ) => 1 / ( i + 1 ) );

	// The length check guards against a vacuous "match": [].every(...) is
	// trivially true in JS regardless of the predicate, so an empty/short
	// GPU readback (a real failure mode seen elsewhere in this project)
	// would otherwise silently report a false pass instead of itself.
	const matches = result.length === expected.length && result.every( ( v, i ) => Math.abs( v - expected[ i ] ) < 1e-3 );

	log(
		'createPreconditionedConjugateGradientSolver — A=diag(1..8), M^-1=diag(1..1/8), b=[1,...,1]',
		succeeded && matches,
		matches
			? `x = [${ result.map( ( v ) => v.toFixed( 4 ) ) }] (expected [${ expected.map( ( v ) => v.toFixed( 4 ) ) }]), succeeded=${ succeeded }`
			: `got [${ result }], expected [${ expected }], succeeded=${ succeeded }`
	);

} catch ( error ) {

	log( 'failed', false, error.message );
	console.error( error );

}
