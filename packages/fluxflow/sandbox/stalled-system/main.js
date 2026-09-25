// The pressure system that examples/35-karman-vortex-street-3d/ cannot
// solve, rebuilt from a captured frame so that it can be re-run in seconds.
//
// Reproducing the failure used to cost a few hundred frames of simulation,
// which is minutes per attempt and makes every question expensive. What is
// captured here is the system and nothing about how it was reached: the
// grid, the Dirichlet and vent masks, the collider's face weights, the
// right-hand side the solver built, and the pressure it arrived at. The
// operator is then rebuilt from those exactly as grid_pressure_solver3.js
// builds it -- same accessors, same options -- and handed to the same CG.
//
// Two frames are captured on purpose, because a fixture that only
// reproduces the failure proves nothing about itself:
//
//   frame 110  the solver converged in 15 iterations, residual 1.3e-5
//   frame 112  the solver reported pAp-growth at iteration 38, residual 257
//
// Both are two frames apart in the same run, with the same static operator
// -- the masks and the weights do not change from frame to frame -- so the
// only thing that differs between them is the right-hand side. That is the
// anomaly this fixture exists to chase: a fixed symmetric positive definite
// pair cannot solve one right-hand side in 15 iterations and fail on
// another, and one of those three words must be wrong.
//
// See docs/3d-solver-investigation.md section 6a, cause C1.

import * as tsl_array_n from 'tsl_array_n';
import { linalg } from 'fluxflow';

const pre = document.querySelector( '#out' );
const lines = [];

function log( text, ok ) {

	lines.push( ok === undefined ? text : `<span class="${ ok ? 'ok' : 'err' }">${ ok ? '✓' : '✗' } ${ text }</span>` );
	pre.innerHTML = lines.join( '\n' );

}

async function loadFloats( url ) {

	const response = await fetch( url );
	if ( ! response.ok ) throw new Error( `${ url }: ${ response.status }` );
	return new Float32Array( await response.arrayBuffer() );

}

async function loadCapture( name ) {

	const base = `./data/${ name }`;
	const meta = await ( await fetch( `${ base }/meta.json` ) ).json();

	const [ b, pressure, dirichletMask, ventMask, weightU, weightV, weightW ] = await Promise.all( [
		loadFloats( `${ base }/b.bin` ),
		loadFloats( `${ base }/pressure.bin` ),
		loadFloats( `${ base }/dirichletMask.bin` ),
		loadFloats( `${ base }/ventMask.bin` ),
		loadFloats( `${ base }/weightU.bin` ),
		loadFloats( `${ base }/weightV.bin` ),
		loadFloats( `${ base }/weightW.bin` )
	] );

	return { meta, b, pressure, dirichletMask, ventMask, weightU, weightV, weightW };

}

// Rebuilds what grid_pressure_solver3.js builds, from captured fields.
function buildSystem( capture ) {

	const shape = capture.meta.resolution;
	const gridSpacing = capture.meta.gridSpacing;

	const field = ( values, fieldShape ) => {

		const array = tsl_array_n.arrayN( 'float', fieldShape );
		array.fromArray( values );
		return array;

	};

	const maskField = field( capture.dirichletMask, shape );
	const ventField = field( capture.ventMask, shape );
	const wu = field( capture.weightU, capture.meta.dataSizeU );
	const wv = field( capture.weightV, capture.meta.dataSizeV );
	const ww = field( capture.weightW, capture.meta.dataSizeW );

	return {
		shape, gridSpacing,
		dirichletMask: ( ...I ) => maskField( ...I ).greaterThan( 0.5 ),
		coarseDirichletMask: ( ...I ) => ventField( ...I ).greaterThan( 0.5 ),
		faceWeights: [ ( ...I ) => wu( ...I ), ( ...I ) => wv( ...I ), ( ...I ) => ww( ...I ) ],
		b: field( capture.b, shape ),
		bHost: capture.b,
		cells: shape.reduce( ( a, n ) => a * n, 1 )
	};

}

function norm( values ) {

	let sum = 0;
	for ( const x of values ) sum += x * x;
	return Math.sqrt( sum );

}

// ?frames=frame-112 and ?cap=300 keep a quick check quick; the defaults run
// both captures against every preconditioner, which is the slow, complete
// form and takes about a minute and a half.
const query = new URLSearchParams( location.search );
const FRAMES = ( query.get( 'frames' ) ?? 'frame-110,frame-112' ).split( ',' );
const CAP = Number( query.get( 'cap' ) ?? 3000 );
const ONLY = query.get( 'only' );

try {

	const renderer = await tsl_array_n.init( { allowFallback: true } );
	log( `init() — backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }`, true );

	if ( renderer.backend?.constructor?.name !== 'WebGPUBackend' ) {

		log( 'not WebGPU — nothing below would be evidence', false );
		throw new Error( 'not WebGPU' );

	}

	for ( const name of FRAMES ) {

		const capture = await loadCapture( name );
		const system = buildSystem( capture );
		const bNorm = norm( system.bHost );

		log( '' );
		log( `<b>${ name }</b> — captured with converged=${ capture.meta.diagnostics.converged }, ` +
			`${ capture.meta.diagnostics.iterations } iterations, residual ${ Number( capture.meta.diagnostics.residual ).toExponential( 2 ) }, ` +
			`stoppedBy ${ capture.meta.diagnostics.stoppedBy }. ||b|| = ${ bNorm.toExponential( 3 ) }` );

		const applyLaplacian = linalg.createLaplacianOperator( system.shape, system.gridSpacing, {
			dirichletMask: system.dirichletMask,
			faceWeights: system.faceWeights
		} );

		const arms = [
			[ 'multigrid x4', () => linalg.createMultigridPreconditioner( system.shape, system.gridSpacing, {
				...capture.meta.multigrid, dirichletMask: system.dirichletMask,
				coarseDirichletMask: system.coarseDirichletMask, faceWeights: system.faceWeights
			} ) ],
			[ 'jacobi', () => linalg.createJacobiPreconditioner( system.shape, system.gridSpacing, {
				dirichletMask: system.dirichletMask, faceWeights: system.faceWeights
			} ) ],
			[ 'none', () => linalg.createIdentityPreconditioner( system.shape ) ]
		].filter( ( [ label ] ) => ! ONLY || label.startsWith( ONLY ) );

		for ( const [ label, build ] of arms ) {

			const maxIterations = CAP;

			const x = tsl_array_n.arrayN( 'float', system.shape );
			x.fromArray( new Float32Array( system.cells ) );

			const M = build();
			if ( M.refreshCoarseLevels ) M.refreshCoarseLevels();

			const solver = linalg.createPreconditionedConjugateGradientSolver( applyLaplacian, M, system.b, x );
			const ok = await solver.solve( capture.meta.tolerance, maxIterations, 1, false, true, false, true );

			const state = solver.state ?? {};
			const residual = state.residualSquared !== undefined ? Math.sqrt( Math.abs( state.residualSquared ) ) : NaN;

			log(
				`&nbsp;&nbsp;${ label.padEnd( 14 ) } converged=${ ok } iters=${ String( state.iterations ).padStart( 5 ) } ` +
				`residual/||b|| = ${ ( residual / bNorm ).toExponential( 2 ) } stoppedBy=${ state.stoppedBy }`,
				ok === true
			);

		}

	}

	log( '' );
	log( 'done', true );

} catch ( error ) {

	log( `failed: ${ error.message }`, false );
	console.error( error );

}
