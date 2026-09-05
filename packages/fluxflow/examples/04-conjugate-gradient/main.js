// Verifies createConjugateGradientSolver against a diagonal operator with a
// known exact solution: A = diag(1,2,...,N), so x = A^-1 @ b has a trivial
// closed form (x[i] = b[i]/(i+1)) to check against. A has N distinct
// eigenvalues, so this also genuinely exercises the multi-iteration
// loop/beta-update logic (in exact arithmetic CG needs up to N iterations
// here), not just a same-first-step trivial case.
//
// Known limitation (sandbox environment, not this code): the solver's dot
// products now run as a genuine GPU-side reduction via WebGPU atomics (see
// src/linalg/linalg.js's header comment, decision 1) instead of a CPU JS
// sum. Atomics are WebGPU-only, and running this example in-sandbox (no
// real WebGPU adapter, init() falls back to WebGLBackend) confirms it
// exactly as predicted from reading three.js's own source beforehand: the
// browser console shows a vertex shader compile error, not a silent wrong
// answer --
//
//   ERROR: 0:68: '&' : syntax error
//   > 68: atomicAdd( &nodeVarying0, int( round( ( ( nodeVarying1 *
//         nodeVarying1 ) * 65536.0 ) ) ) );
//
// -- i.e. the generated code is doing exactly the right thing (a real
// atomic add, correctly encoding r*r at the configured fixed-point scale),
// it's just emitting WGSL pointer syntax (`&x`) that has no GLSL
// equivalent, because the WebGL2 fallback backend's node builder never
// learned the atomic method names at all. This is a *confirmed*
// fallback-only failure, unlike earlier sandbox mysteries in this project
// that were only suspected until a real-hardware run -- there is nothing
// left to isolate here.
//
// CONFIRMED correct on real WebGPU hardware -- reported console output:
//
//   init() — backend: WebGPUBackend
//   createConjugateGradientSolver — A=diag(1..8), b=[1,...,1] —
//     x = [1.0000, 0.4999, 0.3335, 0.2498, 0.2001, 0.1666, 0.1429, 0.1250]
//     (expected [1.0000, 0.5000, 0.3333, 0.2500, 0.2000, 0.1667, 0.1429,
//     0.1250]), succeeded=true
//
// The small (~1e-4) per-element deviation from the exact answer is
// consistent with the atomic dot product's fixed-point quantization noise
// (decision 1 in linalg.js's header comment: each of the 8 per-cell
// products is rounded to the nearest 1/65536 before the atomic add, and
// quantization error accumulates across the sum) -- well within this
// example's own 1e-3 comparison tolerance, not a sign of a real bug.

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

	const solver = linalg.createConjugateGradientSolver( diagonalOperator, b, x );

	const succeeded = await solver.solve( 1e-5, 20 );

	const result = Array.from( await x.toArray() );
	const expected = Array.from( { length: N }, ( _, i ) => 1 / ( i + 1 ) );

	const matches = result.every( ( v, i ) => Math.abs( v - expected[ i ] ) < 1e-3 );

	log(
		'createConjugateGradientSolver — A=diag(1..8), b=[1,...,1]',
		succeeded && matches,
		matches
			? `x = [${ result.map( ( v ) => v.toFixed( 4 ) ) }] (expected [${ expected.map( ( v ) => v.toFixed( 4 ) ) }]), succeeded=${ succeeded }`
			: `got [${ result }], expected [${ expected }], succeeded=${ succeeded }`
	);

} catch ( error ) {

	log( 'failed', false, error.message );
	console.error( error );

}
