// Verifies the full pipeline: createMultigridPreconditioner wired into
// createPreconditionedConjugateGradientSolver against a real 2D Poisson
// problem (the same constant-coefficient Laplacian createLaplacianOperator
// implements), unlike examples/04-/05-'s 1D diagonal toy case -- multigrid
// needs actual multi-dimensional grid structure to mean anything.
//
// The right-hand side is *constructed* from a chosen exact solution
// (b = A @ xExpected, computed via the same operator instance the solver
// itself uses) rather than picked independently, for a reason specific to
// this operator: a pure zero-flux (Neumann) boundary condition -- the
// choice this file's Laplacian uses, see multigrid.js's header comment --
// has a null space (constant fields: A @ constant = 0), so an arbitrary
// `b` might not even be solvable, and even a solvable one only determines
// `x` up to an arbitrary additive constant. Building `b` this way
// guarantees a solution exists; choosing `xExpected` to itself be
// zero-mean (a product with a full-period sin() factor along one axis,
// which sums to exactly zero discretely) additionally guarantees it's
// *the* zero-mean solution CG converges to starting from x0=0 (whose own
// residual b-Ax0=b is already zero-mean) -- so a direct element-wise
// comparison against xExpected is meaningful, not just "up to a constant".
//
// Known limitation (sandbox environment, not this code): this solver's dot
// products (r.r, r.z, p.Ap) run through the same GPU-atomic reduction as
// examples/04-/05-, which is WebGPU-only -- expect the same confirmed
// fallback-only vertex shader compile error in this dev sandbox. This
// example additionally wires in createMultigridPreconditioner, whose own
// standalone verification (examples/06-multigrid-preconditioner/) found
// this sandbox unreliable for a *different* reason too (see that file's
// header comment).
//
// CONFIRMED correct on real WebGPU hardware: `backend: WebGPUBackend`,
// converged to xExpected (max |diff| < 1e-2), succeeded=true -- the full
// pipeline (GPU-atomic CG reduction + multigrid preconditioner together)
// genuinely solves this 2D Poisson problem, not just each piece in
// isolation.
//
// One thing worth flagging about this example's *own* verification code,
// found while first running it here: `x.toArray()` came back as an empty
// array (0 elements, not 256) in this dev sandbox -- the same "no buffer"
// symptom already seen elsewhere in this project. `[].every(...)` is
// vacuously `true` in JavaScript regardless of the predicate, so the
// original version of this check (`result.every(...)`, with no length
// guard) silently reported a false "converged" pass instead of the
// empty-readback failure it actually was. Fixed by checking
// `result.length === expected.length` first. Worth remembering generally:
// an empty/short GPU readback in this sandbox can masquerade as success
// through `.every()`'s vacuous-truth behavior, not just as an obviously-
// wrong all-zero array the way earlier examples happened to show it.

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

const N = 16;
const shape = [ N, N ];
const gridSpacing = [ 1, 1 ];

try {

	const renderer = await tsl_array_n.init( { allowFallback: true } );
	log( 'init()', true, `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }` );

	// xExpected(i,j) = sin(2*pi*i/N) * cos(2*pi*j/N) -- zero-mean (sum_i
	// sin(2*pi*i/N) = 0 exactly), tsl_array_n's flat layout is i + N*j.
	const xExpectedArray = new Float32Array( N * N );
	for ( let j = 0; j < N; j ++ ) {

		for ( let i = 0; i < N; i ++ ) {

			xExpectedArray[ i + N * j ] = Math.sin( 2 * Math.PI * i / N ) * Math.cos( 2 * Math.PI * j / N );

		}

	}

	const applyLaplacian = linalg.createLaplacianOperator( shape, gridSpacing );

	const xExpected = tsl_array_n.arrayN( 'float', shape );
	xExpected.fromArray( xExpectedArray );

	const b = tsl_array_n.arrayN( 'float', shape );
	const dispatchB = applyLaplacian( xExpected, b );
	dispatchB(); // b = A @ xExpected

	const x = tsl_array_n.arrayN( 'float', shape );
	x.fromArray( new Float32Array( N * N ) );

	const applyPreconditioner = linalg.createMultigridPreconditioner( shape, gridSpacing, { numberOfLevels: 4 } );
	const solver = linalg.createPreconditionedConjugateGradientSolver( applyLaplacian, applyPreconditioner, b, x );

	const succeeded = await solver.solve( 1e-5, 50 );

	const result = Array.from( await x.toArray() );
	const expected = Array.from( xExpectedArray );

	// result.length check guards against a vacuous "match": if x.toArray()
	// comes back empty (the established "no buffer" sandbox symptom, see
	// this file's own header comment), [].every(...) is trivially true in
	// JS regardless of the predicate -- without this check, that would
	// silently report a false pass instead of the empty-readback failure
	// it actually is.
	const matches = result.length === expected.length && result.every( ( v, i ) => Math.abs( v - expected[ i ] ) < 1e-2 );

	const detail = matches
		? `converged to xExpected (max |diff| < 1e-2), succeeded=${ succeeded }`
		: result.length !== expected.length
			? `x.toArray() returned ${ result.length } elements, expected ${ expected.length } -- empty/short readback, not a convergence failure (see this file's header comment)`
			: `did not match xExpected -- max |diff| = ${ Math.max( ...result.map( ( v, i ) => Math.abs( v - expected[ i ] ) ) ).toFixed( 4 ) }, succeeded=${ succeeded }`;

	log( 'multigrid-preconditioned CG — 2D Poisson, 16x16, zero-flux boundary', succeeded && matches, detail );

} catch ( error ) {

	log( 'failed', false, error.message );
	console.error( error );

}
