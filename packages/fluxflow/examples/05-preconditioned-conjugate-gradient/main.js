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
import { float } from 'three/tsl';
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

	// Both scalar paths have to produce this exact answer. The GPU-resident
	// one computes alpha and beta in kernels and reads the loop's scalars
	// back once per iteration instead of three times, which means every
	// guard the host loop applies before touching x had to move into those
	// kernels too -- so "does it still land on A^-1@b" is only half of what
	// needs checking here. The guard cases below are the other half.
	for ( const gpuResidentScalars of [ false, true ] ) {

		const label = gpuResidentScalars ? 'GPU-resident alpha/beta' : 'host alpha/beta';

		x.fromArray( new Float32Array( N ) ); // x0 = [0,0,...,0]

		const solver = linalg.createPreconditionedConjugateGradientSolver( diagonalOperator, jacobiPreconditioner, b, x );

		const succeeded = await solver.solve( 1e-5, 20, 1, gpuResidentScalars );

		const result = Array.from( await x.toArray() );
		const expected = Array.from( { length: N }, ( _, i ) => 1 / ( i + 1 ) );

		// The length check guards against a vacuous "match": [].every(...) is
		// trivially true in JS regardless of the predicate, so an empty/short
		// GPU readback (a real failure mode seen elsewhere in this project)
		// would otherwise silently report a false pass instead of itself.
		const matches = result.length === expected.length && result.every( ( v, i ) => Math.abs( v - expected[ i ] ) < 1e-3 );

		log(
			`A=diag(1..8), M^-1=diag(1..1/8), b=[1,...,1] — ${ label }`,
			succeeded && matches,
			matches
				? `x = [${ result.map( ( v ) => v.toFixed( 4 ) ) }] (expected [${ expected.map( ( v ) => v.toFixed( 4 ) ) }]), succeeded=${ succeeded }`
				: `got [${ result }], expected [${ expected }], succeeded=${ succeeded }`
		);

	}

	// *** The guards, which are the part that had to be rewritten ***
	//
	// The host loop checks its denominators on the CPU and breaks *before*
	// the kernel that would consume a bad alpha. The GPU-resident path
	// cannot do that -- it only finds out an iteration later -- so instead
	// its kernels write alpha (or beta) as exactly 0, which makes the update
	// they feed a no-op. Both arrangements have to leave x untouched and
	// report the same reason; that equivalence is what these two cases test,
	// and nothing else in this directory reaches them. The first is not
	// hypothetical: a fully closed, all-Neumann pressure domain is singular,
	// and grid_pressure_solver2.js hits it.
	function identityPreconditioner( input, output ) {

		return tsl_array_n.kernel( N, ( i ) => {

			output( i ).assign( input( i ) );

		} );

	}

	const guardCases = [
		{
			name: 'singular operator (A = 0) — p.Ap is exactly 0',
			expectedStop: 'degenerate-pAp',
			operator: ( input, output ) => tsl_array_n.kernel( N, ( i ) => {

				output( i ).assign( float( 0 ).mul( input( i ) ) );

			} )
		},
		{
			name: 'near-null operator (A = diag(1e-12)) — alpha overflows its bound',
			expectedStop: 'alpha-magnitude',
			operator: ( input, output ) => tsl_array_n.kernel( N, ( i ) => {

				output( i ).assign( input( i ).mul( 1e-12 ) );

			} )
		}
	];

	for ( const guardCase of guardCases ) {

		const outcomes = {};

		// Interval 4 is here because it is the case the sticky stop code
		// exists for: a guard can trip on an iteration whose readback is
		// skipped, and the host must still find out. If the code were
		// overwritten by the next clean iteration, this arm would report
		// 'none' and the test would catch it.
		for ( const [ key, gpuResidentScalars, interval ] of [
			[ 'host', false, 1 ],
			[ 'gpu', true, 1 ],
			[ 'gpu, check every 4', true, 4 ]
		] ) {

			x.fromArray( new Float32Array( N ) );

			const solver = linalg.createPreconditionedConjugateGradientSolver( guardCase.operator, identityPreconditioner, b, x );

			await solver.solve( 1e-5, 20, interval, gpuResidentScalars );

			const finalX = Array.from( await x.toArray() );

			outcomes[ key ] = {
				stoppedBy: solver.state.stoppedBy,
				// x must still be finite. A guard that let a bad alpha
				// through would show up here and nowhere else.
				finite: finalX.length === N && finalX.every( Number.isFinite ),
				x: finalX
			};

		}

		const keys = Object.keys( outcomes );
		const agree = keys.every( ( k ) => outcomes[ k ].stoppedBy === guardCase.expectedStop );
		const finite = keys.every( ( k ) => outcomes[ k ].finite );

		log(
			`guard — ${ guardCase.name }`,
			agree && finite,
			keys.map( ( k ) => `${ k }: '${ outcomes[ k ].stoppedBy }' (x finite: ${ outcomes[ k ].finite })` ).join( ', ' ) +
			` — expected '${ guardCase.expectedStop }'`
		);

	}

} catch ( error ) {

	log( 'failed', false, error.message );
	console.error( error );

}
