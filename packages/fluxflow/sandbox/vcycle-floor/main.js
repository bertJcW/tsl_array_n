// How far down can this V-cycle actually drive a residual, and what moves that
// floor?
//
// examples/17-smoke-fire/, 18 and 19 ask for a relative residual of 1e-6 and
// converge on 0 of 12,001 frames. Freezing one of their frames and re-solving
// it (diag_iterations.mjs) showed the multigrid arm reaching 1.53e-5 relative
// at 60 iterations and then getting WORSE -- 2.77e-2 by 3000 -- while the same
// system with jacobi or with no preconditioner converged monotonically to
// 5.85e-6. So the floor belongs to the V-cycle, not to the operator or to
// float32.
//
// That leaves the question this page exists for, which cannot be asked of a
// scene: is 1e-6 reachable AT ALL by this V-cycle on a problem of this shape,
// and if not, which of its knobs is the binding one? A scene's right-hand side
// changes every frame and its mask can move, so nothing measured on one is
// separable. Here the system is built as b = A @ xStar from a known xStar, the
// mask is fixed, and the only thing that varies between arms is the
// preconditioner's own configuration.
//
// The shape and mask are example 17's: 96x128 with the top two rows pinned,
// which is exactly the 192 pinned cells that scene reports.

import * as tsl_array_n from 'tsl_array_n';
import { linalg } from 'fluxflow';

const pre = document.querySelector( '#status pre' );
const lines = [];
let passed = 0, failed = 0;

function log( label, ok, detail ) {

	lines.push( `<span class="${ ok ? 'ok' : 'err' }">${ ok ? '✓' : '✗' } ${ label }${ detail ? ' — ' + detail : '' }</span>` );
	pre.innerHTML = lines.join( '\n' );
	ok ? passed ++ : failed ++;

}

function note( text ) {

	lines.push( `<span class="note">${ text }</span>` );
	pre.innerHTML = lines.join( '\n' );

}

const SHAPE = [ 96, 128 ];
const SPACING = [ 1, 1 ];
const CELLS = SHAPE[ 0 ] * SHAPE[ 1 ];

const dot = ( a, b ) => { let s = 0; for ( let i = 0; i < a.length; i ++ ) s += a[ i ] * b[ i ]; return s; };

try {

	const renderer = await tsl_array_n.init( { allowFallback: true } );
	const backend = renderer.backend?.constructor?.name ?? 'unknown';
	log( 'init()', backend === 'WebGPUBackend', `backend: ${ backend }` );
	if ( backend !== 'WebGPUBackend' ) throw new Error( 'not WebGPU' );

	// Example 17's vent: the top two rows in j.
	const maskArray = new Float32Array( CELLS );
	let pinned = 0;
	for ( let n = 0; n < CELLS; n ++ ) {

		const j = Math.floor( n / SHAPE[ 0 ] );
		if ( j >= SHAPE[ 1 ] - 2 ) { maskArray[ n ] = 1; pinned ++; }

	}
	const maskField = tsl_array_n.arrayN( 'float', SHAPE );
	maskField.fromArray( maskArray );
	const dirichletMask = ( ...I ) => maskField( ...I ).greaterThan( 0.5 );
	note( `${ SHAPE.join( 'x' ) }, ${ pinned } pinned cells in the top two rows — examples/17-smoke-fire/ reports 192` );

	// Smooth, not noise: white noise is all high frequency, which CG removes in
	// a handful of iterations while leaving the low-frequency error a real scene
	// cares about. Noise flatters the solver and hides the floor being looked
	// for. Same reasoning as examples/31-conjugate-gradient-3d/.
	const xStarArray = new Float32Array( CELLS );
	for ( let n = 0; n < CELLS; n ++ ) {

		const i = n % SHAPE[ 0 ];
		const j = Math.floor( n / SHAPE[ 0 ] );
		xStarArray[ n ] = Math.sin( 2 * Math.PI * i / SHAPE[ 0 ] ) * Math.cos( 2 * Math.PI * j / SHAPE[ 1 ] )
			+ 0.3 * Math.sin( 6 * Math.PI * i / SHAPE[ 0 ] );

	}

	const applyLaplacian = linalg.createLaplacianOperator( SHAPE, SPACING, { dirichletMask } );

	const xStar = tsl_array_n.arrayN( 'float', SHAPE );
	const b = tsl_array_n.arrayN( 'float', SHAPE );
	xStar.fromArray( xStarArray );
	b.fromArray( new Float32Array( CELLS ) );
	applyLaplacian( xStar, b )();

	const bArray = await b.toArray();
	const bNorm = Math.sqrt( dot( bArray, bArray ) );
	note( `system built as b = A @ xStar, |b| = ${ bNorm.toExponential( 3 ) } — so no scene's right-hand side is on trial here` );
	note( '' );

	const x = tsl_array_n.arrayN( 'float', SHAPE );

	async function arm( label, preconditioner, tolerance, maxIterations ) {

		x.fromArray( new Float32Array( CELLS ) );
		const solver = linalg.createPreconditionedConjugateGradientSolver( applyLaplacian, preconditioner, b, x );
		const ok = await solver.solve( tolerance, maxIterations, 1, false, true, false, true );

		const st = solver.state ?? {};
		const residual = st.residualSquared !== undefined ? Math.sqrt( Math.abs( st.residualSquared ) ) : NaN;
		const xd = await x.toArray();
		let maxErr = 0;
		for ( let n = 0; n < CELLS; n ++ ) maxErr = Math.max( maxErr, Math.abs( xd[ n ] - xStarArray[ n ] ) );

		return { label, ok, iterations: st.iterations, relative: residual / bNorm, stoppedBy: st.stoppedBy, maxErr };

	}

	// ---- 1. is 1e-6 reachable at all, at the shipped configuration? ----

	{
		const M = linalg.createMultigridPreconditioner( SHAPE, SPACING, { numberOfLevels: 4, dirichletMask } );
		if ( M.refreshCoarseLevels ) M.refreshCoarseLevels();

		const loose = await arm( 'tol 1e-5', M, 1e-5, 3000 );
		const tight = await arm( 'tol 1e-6', M, 1e-6, 3000 );

		log( `multigrid x4, tolerance 1e-5`, loose.ok === true,
			`${ loose.iterations } iterations, res/|b| ${ loose.relative.toExponential( 2 ) }, max|x-xStar| ${ loose.maxErr.toExponential( 2 ) }, stoppedBy ${ loose.stoppedBy }` );

		log( `multigrid x4, tolerance 1e-6 — the one examples 17-19 ask for`, tight.ok === true,
			`${ tight.iterations } iterations, res/|b| ${ tight.relative.toExponential( 2 ) }, max|x-xStar| ${ tight.maxErr.toExponential( 2 ) }, stoppedBy ${ tight.stoppedBy }` );
	}

	note( '' );

	// ---- 2. what moves the floor: levels, and smoothing sweeps ----
	//
	// numberOfFinalIterations is the V-cycle's own smoothing count and is fixed
	// at construction, which is why this cannot be asked of a live scene.

	note( 'asking for 1e-6 in every arm below; "floor" is the best res/|b| it reached' );

	for ( const levels of [ 2, 3, 4 ] ) {

		for ( const sweeps of [ 2, 4, 8 ] ) {

			const M = linalg.createMultigridPreconditioner( SHAPE, SPACING, {
				numberOfLevels: levels, numberOfFinalIterations: sweeps, dirichletMask
			} );
			if ( M.refreshCoarseLevels ) M.refreshCoarseLevels();

			const r = await arm( `x${ levels }, ${ sweeps } sweeps`, M, 1e-6, 3000 );

			log( `multigrid x${ levels }, ${ sweeps } smoothing sweeps`, r.ok === true,
				`${ r.iterations } iterations, floor ${ r.relative.toExponential( 2 ) }, max|x-xStar| ${ r.maxErr.toExponential( 2 ) }, stoppedBy ${ r.stoppedBy }` );

		}

	}

	note( '' );
	note( failed === 0
		? `all ${ passed } arms reached their tolerance`
		: `${ failed } of ${ passed + failed } arms did not reach 1e-6 — read the floors above: whichever knob moves them is the binding one` );

} catch ( error ) {

	log( 'failed', false, error.message );
	console.error( error );

}
