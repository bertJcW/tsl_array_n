// Does a damped-Jacobi smoother beat red-black Gauss-Seidel inside the
// V-cycle?
//
// *** Why this page exists rather than a flag in src/ ***
//
// The change is structural, not a setting. A Jacobi sweep is defined on
// the *old* iterate, so it cannot write into the array it is reading;
// every level needs a second x and the cycle has to ping-pong between
// them. Red-black avoids that by writing one colour per dispatch, whose
// neighbours are all the other colour -- it buys in-place updating with a
// second dispatch. So the trade is: half the dispatches, one extra buffer
// per level.
//
// ../../sandbox/jacobi-smoother/multigrid_jacobi.js is a copy of
// src/linalg/multigrid.js with exactly that one change. This page builds
// both from the same operator and hands both to the same CG solver, so
// the smoother is the only difference between the arms.
//
// *** What is being decided ***
//
// A V-cycle costs 0.155 ms, measured (see the performance document), and
// a whole CG iteration costs 0.861 ms. Halving the V-cycle's dispatches
// is therefore worth about 4% of a solver step AT BEST, and only if the
// iteration count does not rise. A float64 reference priced damped Jacobi
// at 29-31 iterations against red-black's 30 -- on a clean
// constant-coefficient Poisson. The two cases below are the ones that
// reference cannot speak for:
//
//   - a DIRICHLET MASK, where a masked row's diagonal is -1 against an
//     interior row's -4/h^2, a 4x variation the damping factor was not
//     tuned for;
//   - VARIABLE FACE WEIGHTS, the density coupling, where the diagonal
//     varies with the density ratio.
//
// Iteration count is the number to read here. Wall time on a 32x32 grid
// is dominated by fixed costs and says very little; the question this
// page answers is whether Jacobi smoothing costs iterations on the real
// operator, because if it does the dispatch saving is already spent.

import * as tsl_array_n from 'tsl_array_n';
import { linalg } from 'fluxflow';
import { createMultigridPreconditioner as createJacobiSmootherPreconditioner } from '../../sandbox/jacobi-smoother/multigrid_jacobi.js';

const pre = document.querySelector( '#status pre' );
const lines = [];

function log( label, ok, detail ) {

	const cls = ok ? 'ok' : 'err';
	const mark = ok ? '✓' : '✗';
	lines.push( `<span class="${ cls }">${ mark } ${ label }${ detail ? ' — ' + detail : '' }</span>` );
	pre.innerHTML = lines.join( '\n' );

}

function note( text ) {

	lines.push( `<span style="opacity:.7">  ${ text }</span>` );
	pre.innerHTML = lines.join( '\n' );

}

const N = 32;
const shape = [ N, N ];
const gridSpacing = [ 1, 1 ];
const TOLERANCE = 1e-5;
const MAX_ITERATIONS = 400;

try {

	const renderer = await tsl_array_n.init( { allowFallback: true } );
	log( 'init()', true, `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }` );

	const b = tsl_array_n.arrayN( 'float', shape );
	const x = tsl_array_n.arrayN( 'float', shape );

	// A right-hand side with structure at several scales, so the coarse
	// levels have something to do -- a single smooth mode would be solved
	// by the coarse grid alone and would not exercise the smoother.
	const rhs = new Float32Array( N * N );
	for ( let j = 0; j < N; j ++ ) {

		for ( let i = 0; i < N; i ++ ) {

			const s1 = Math.sin( 2 * Math.PI * i / N ) * Math.cos( 2 * Math.PI * j / N );
			const s2 = Math.sin( 8 * Math.PI * i / N ) * Math.cos( 6 * Math.PI * j / N );
			rhs[ i + N * j ] = s1 + 0.4 * s2;

		}

	}

	// A circular Dirichlet region in the middle, the same shape of mask
	// examples/15-flow-past-cylinder/ puts in the operator.
	const maskField = tsl_array_n.arrayN( 'float', shape );
	const maskData = new Float32Array( N * N );
	for ( let j = 0; j < N; j ++ ) {

		for ( let i = 0; i < N; i ++ ) {

			const dx = i - N * 0.5, dy = j - N * 0.5;
			maskData[ i + N * j ] = dx * dx + dy * dy < ( N * 0.15 ) ** 2 ? 1 : 0;

		}

	}

	maskField.fromArray( maskData );

	// Face weights standing in for a density ratio, so the diagonal varies
	// the way it does in the two-phase scenes.
	const weightU = tsl_array_n.arrayN( 'float', [ N + 1, N ] );
	const weightV = tsl_array_n.arrayN( 'float', [ N, N + 1 ] );
	const fill = ( field, w, h, heavyBelow ) => {

		const data = new Float32Array( w * h );
		for ( let j = 0; j < h; j ++ ) {

			for ( let i = 0; i < w; i ++ ) data[ i + w * j ] = j < heavyBelow ? 1 : 1 / 8;

		}

		field.fromArray( data );

	};

	fill( weightU, N + 1, N, N * 0.5 );
	fill( weightV, N, N + 1, N * 0.5 );

	const cases = [
		{ name: 'plain Poisson (no mask, uniform)', options: {} },
		{ name: 'Dirichlet mask (a circle, as in example 15)', options: { dirichletMask: ( i, j ) => maskField( i, j ).greaterThan( 0.5 ) } },
		{ name: 'variable density (8:1 face weights)', options: { faceWeights: [ ( i, j ) => weightU( i, j ), ( i, j ) => weightV( i, j ) ] } }
	];

	const omegas = [ 2 / 3, 0.8, 1.0 ];

	for ( const testCase of cases ) {

		const operatorOptions = { ...testCase.options };
		const applyOperator = linalg.createLaplacianOperator( shape, gridSpacing, operatorOptions );

		async function solveWith( preconditioner ) {

			b.fromArray( rhs );
			x.fromArray( new Float32Array( N * N ) );

			const cg = linalg.createPreconditionedConjugateGradientSolver( applyOperator, preconditioner, b, x );
			const converged = await cg.solve( TOLERANCE, MAX_ITERATIONS );

			return { converged, iterations: cg.state.iterations, stoppedBy: cg.state.stoppedBy };

		}

		// Red-black, with the single-workgroup coarse kernel OFF, because
		// the sandbox has no Jacobi counterpart for it. Both arms
		// therefore run the same coarse-level construction and the
		// smoother is the only difference.
		const redBlack = await solveWith( linalg.createMultigridPreconditioner(
			shape, gridSpacing, { ...operatorOptions, numberOfLevels: 4, coarseSingleGroup: false }
		) );

		log(
			`${ testCase.name } — red-black`,
			redBlack.converged,
			`${ redBlack.iterations } iterations, stoppedBy ${ redBlack.stoppedBy }`
		);

		for ( const omega of omegas ) {

			const jacobi = await solveWith( createJacobiSmootherPreconditioner(
				shape, gridSpacing, { ...operatorOptions, numberOfLevels: 4, omega }
			) );

			const ratio = jacobi.iterations / redBlack.iterations;

			note(
				`damped Jacobi, omega = ${ omega.toFixed( 3 ) }: ` +
				`${ jacobi.converged ? `${ jacobi.iterations } iterations` : `DID NOT CONVERGE (${ jacobi.iterations }, ${ jacobi.stoppedBy })` }` +
				( jacobi.converged ? ` — ${ ratio.toFixed( 2 ) }x red-black's count` : '' )
			);

		}

	}

	note( '' );
	note( 'Reading it: Jacobi halves the V-cycle\'s dispatches, and a V-cycle is' );
	note( '0.155 ms against 0.861 ms for a whole CG iteration. So the saving is' );
	note( 'worth ~4% of a step at best, and any ratio above ~1.04 spends it.' );

} catch ( error ) {

	log( 'failed', false, error.message );
	console.error( error );

}
