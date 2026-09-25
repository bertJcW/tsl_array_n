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
import { Loop } from 'three/tsl';
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

// The captures made before export_system.mjs learned to record the cycle's
// shape carry only its runtime switches, which silently rebuilds a
// one-level preconditioner. Fall back to the settings
// examples/35-karman-vortex-street-3d/ is written with, which is what those
// captures were taken from.
function cycleOptions( capture ) {

	return capture.meta.multigridOptions ?? {
		numberOfLevels: 4, numberOfSmoothingIterationsDown: 3,
		numberOfSmoothingIterationsUp: 3, numberOfCoarsestIterations: 30
	};

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
const TRACE = Number( query.get( 'trace' ) ?? 0 );
const DOT = query.get( 'dot' ) === '1';

try {

	const renderer = await tsl_array_n.init( { allowFallback: true } );
	log( `init() — backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }`, true );

	if ( renderer.backend?.constructor?.name !== 'WebGPUBackend' ) {

		log( 'not WebGPU — nothing below would be evidence', false );
		throw new Error( 'not WebGPU' );

	}

	for ( const name of FRAMES ) {

		if ( TRACE > 0 || DOT ) break; // ?trace and ?dot run their own sections below

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

		// ?coarsen=mask,weights sweeps what the coarse levels are told, so
		// that what each is worth on a real captured system is a number
		// rather than an argument. Re-taken after the dot product fix,
		// because the first set was measured through it.
		const sweeps = ( query.get( 'coarsen' ) ?? 'both' ).split( ',' );
		const coarsenArms = {
			both: { coarsenDirichletMask: true, coarsenFaceWeights: true },
			mask: { coarsenDirichletMask: true, coarsenFaceWeights: false },
			weights: { coarsenDirichletMask: false, coarsenFaceWeights: true },
			neither: { coarsenDirichletMask: false, coarsenFaceWeights: false }
		};

		const arms = [
			...sweeps.map( ( sweep ) => [ `multigrid (${ sweep })`, () => linalg.createMultigridPreconditioner( system.shape, system.gridSpacing, {
				...cycleOptions( capture ), dirichletMask: system.dirichletMask,
				coarseDirichletMask: system.coarseDirichletMask, faceWeights: system.faceWeights,
				...coarsenArms[ sweep ]
			} ) ] ),
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
			// Evidence that a sweep arm actually changed something: with no
			// coarsening there are no coarsening kernels to dispatch.
			const coarsening = M.refreshCoarseLevels ? 'coarsens' : 'no coarsen kernels';
			if ( M.refreshCoarseLevels ) M.refreshCoarseLevels();

			const solver = linalg.createPreconditionedConjugateGradientSolver( applyLaplacian, M, system.b, x );
			const ok = await solver.solve( capture.meta.tolerance, maxIterations, 1, false, true, false, true );

			const state = solver.state ?? {};
			const residual = state.residualSquared !== undefined ? Math.sqrt( Math.abs( state.residualSquared ) ) : NaN;

			log(
				`&nbsp;&nbsp;${ label.padEnd( 14 ) } converged=${ ok } iters=${ String( state.iterations ).padStart( 5 ) } ` +
				`residual/||b|| = ${ ( residual / bNorm ).toExponential( 2 ) } stoppedBy=${ state.stoppedBy } [${ coarsening }]`,
				ok === true
			);

		}

	}

	// ---- ?dot=1 : is the solver's own dot product accurate here? ----
	//
	// Its lanes accumulate in float32 -- on this grid, 576 products each --
	// and alpha and beta are quotients of two of them, so an error here is
	// an error in every step CG takes. This repo has had one root cause in
	// exactly this place before (project-history.md, Debugging #5).
	//
	// Both vectors are on the device without passing through this page's
	// own uploads, which would otherwise be what the test measured: `b` was
	// uploaded when the capture loaded and has been read by many dispatches
	// since, and `z` is written by the V-cycle itself. The host-side value
	// is computed from a read-back of those same two fields, in double
	// precision.
	if ( DOT ) {

		// The simplest possible question first, with no capture, no
		// preconditioner and no floating point subtlety in it: a field
		// written to exactly 1.0 by a kernel, dotted with itself. The
		// answer is the cell count, exactly, in any arithmetic.
		{

			const shape = [ 48, 24, 24 ];
			const cells = shape[ 0 ] * shape[ 1 ] * shape[ 2 ];
			const ones = tsl_array_n.arrayN( 'float', shape );
			tsl_array_n.kernel( shape, ( i, j, k ) => { ones( i, j, k ).assign( 1 ); } )();

			const reducer = linalg.createDotReducer( shape, ones, ones );
			const value = await reducer.read();
			const partials = await reducer.partial.toArray();
			const perLane = cells / shape[ 0 ];
			let wrongLanes = 0;
			for ( const x of partials ) if ( x !== perLane ) wrongLanes ++;

			log( '' );
			log( `a field of ones, dotted with itself: ${ value } against ${ cells }`, value === cells );
			log( `&nbsp;&nbsp;lanes not equal to ${ perLane }: ${ wrongLanes } of ${ partials.length }`, wrongLanes === 0 );

			// Which cells does a pair of nested Loops actually visit? The
			// reducer's own structure, writing a marker instead of summing.
			{

				const visited = tsl_array_n.arrayN( 'float', shape );
				tsl_array_n.kernel( shape, ( i, j, k ) => { visited( i, j, k ).assign( 0 ); } )();

				tsl_array_n.kernel( [ shape[ 0 ] ], ( i ) => {

					Loop( shape[ 1 ], shape[ 2 ], ( { i: j, j: k } ) => {

						visited( i, j, k ).addAssign( 1 );

					} );

				} )();

				const marks = await visited.toArray();
				let touchedOnce = 0, touchedMore = 0, untouched = 0, diagonalOnly = true;
				for ( let k = 0; k < shape[ 2 ]; k ++ ) for ( let j = 0; j < shape[ 1 ]; j ++ ) for ( let i = 0; i < shape[ 0 ]; i ++ ) {

					const count = marks[ i + shape[ 0 ] * j + shape[ 0 ] * shape[ 1 ] * k ];
					if ( count === 0 ) untouched ++;
					else if ( count === 1 ) touchedOnce ++;
					else touchedMore ++;
					if ( count !== 0 && j !== k ) diagonalOnly = false;

				}

				log( '' );
				log( `Loop( ny, nz ) over a [${ shape }] field: ` +
					`${ untouched } cells never written, ${ touchedOnce } written once, ${ touchedMore } written more than once` +
					`${ diagonalOnly ? ', and every cell written has j === k' : '' }`,
					untouched === 0 && touchedMore === 0 );

			}

			// A uniform field cannot detect a wrong index mapping: any 576
			// ones sum to 576 whichever cells they are. A single cell set to
			// one can. Each probe below should light up exactly one lane --
			// the one matching its own first index -- with exactly 1.0.
			const delta = tsl_array_n.arrayN( 'float', shape );
			const probes = [ [ 5, 0, 0 ], [ 5, 12, 12 ], [ 5, 23, 23 ], [ 7, 12, 12 ], [ 0, 0, 1 ], [ 0, 1, 0 ], [ 1, 0, 0 ] ];
			const deltaReducer = linalg.createDotReducer( shape, delta, ones );

			for ( const [ pi, pj, pk ] of probes ) {

				tsl_array_n.kernel( shape, ( i, j, k ) => {

					const hit = i.equal( pi ).and( j.equal( pj ) ).and( k.equal( pk ) );
					delta( i, j, k ).assign( hit.select( 1, 0 ) );

				} )();

				const total = await deltaReducer.read();
				const lanes2 = await deltaReducer.partial.toArray();
				const lit = [];
				for ( let lane = 0; lane < lanes2.length; lane ++ ) if ( lanes2[ lane ] !== 0 ) lit.push( `${ lane }:${ lanes2[ lane ] }` );

				log(
					`&nbsp;&nbsp;one cell at (${ pi },${ pj },${ pk }) — total ${ total }, lanes lit ${ lit.join( ' ' ) || 'none' }`,
					total === 1 && lit.length === 1 && lit[ 0 ] === `${ pi }:1`
				);

			}

		}

		for ( const name of FRAMES ) {

			const capture = await loadCapture( name );
			const system = buildSystem( capture );

			// The prediction this is really testing: cancellation inside a
			// lane is what costs the accuracy, and only a V-cycle produces a
			// z whose products against r change sign. r.r is a sum of
			// squares and Jacobi's z is r scaled by a diagonal of one sign,
			// so both should cancel nothing and both should be accurate.
			const builders = [
				[ 'multigrid', () => { const M = linalg.createMultigridPreconditioner( system.shape, system.gridSpacing, {
					...cycleOptions( capture ), dirichletMask: system.dirichletMask,
					coarseDirichletMask: system.coarseDirichletMask, faceWeights: system.faceWeights } );
					if ( M.refreshCoarseLevels ) M.refreshCoarseLevels(); return M; } ],
				[ 'jacobi', () => linalg.createJacobiPreconditioner( system.shape, system.gridSpacing, {
					dirichletMask: system.dirichletMask, faceWeights: system.faceWeights } ) ],
				[ 'none (r.r)', () => linalg.createIdentityPreconditioner( system.shape ) ]
			];

			log( '' );
			log( `<b>${ name }</b> — the solver's own dot product against double precision` );

			for ( const [ label, build ] of builders ) {

			const M = build();

			const zField = tsl_array_n.arrayN( 'float', system.shape );
			zField.fromArray( new Float32Array( system.cells ) );
			M( system.b, zField )();

			const reducer = linalg.createDotReducer( system.shape, system.b, zField );
			const first = await reducer.read();
			const second = await reducer.read();

			const bHost = await system.b.toArray();
			const zHost = await zField.toArray();

			// Before trusting either side: does reading `b` back give what
			// was uploaded into it? If not, the disagreement is in the
			// read-back path and says nothing about the reducer.
			let readbackMismatch = 0, worstReadback = 0;
			for ( let i = 0; i < capture.b.length; i ++ ) {

				const delta = Math.abs( bHost[ i ] - capture.b[ i ] );
				if ( delta !== 0 ) readbackMismatch ++;
				if ( delta > worstReadback ) worstReadback = delta;

			}

			let exact = 0;
			for ( let i = 0; i < bHost.length; i ++ ) exact += bHost[ i ] * zHost[ i ];

			// what one lane is asked to do, and what it is asked to cancel
			const lanes = system.shape[ 0 ];
			const perLane = new Array( lanes ).fill( 0 );
			const perLaneMagnitude = new Array( lanes ).fill( 0 );
			for ( let i = 0; i < bHost.length; i ++ ) {

				const lane = i % lanes;
				const product = bHost[ i ] * zHost[ i ];
				perLane[ lane ] += product;
				perLaneMagnitude[ lane ] += Math.abs( product );

			}

			let worstCancellation = 0;
			for ( let lane = 0; lane < lanes; lane ++ ) {

				const ratio = perLaneMagnitude[ lane ] / Math.max( Math.abs( perLane[ lane ] ), 1e-30 );
				if ( ratio > worstCancellation ) worstCancellation = ratio;

			}

			const relative = Math.abs( first - exact ) / Math.max( Math.abs( exact ), 1e-30 );

			// *** Which of the two is lying ***
			//
			// The reducer and a read-back disagree about a sum of positive
			// squares, which float32 accumulation cannot explain and which
			// a field of ones does not reproduce. Either the reducer reads
			// something other than what the buffers hold, or toArray
			// returns something other than what the buffers hold. Splitting
			// the multiplication from the accumulation settles it:
			//
			//   productsThenHost   a kernel writes a*b per cell; the host
			//                      reads that field back and sums it in
			//                      double precision. Device multiplication,
			//                      host accumulation.
			//   productsThenDevice the same product field, reduced by the
			//                      reducer against a field of ones. Device
			//                      multiplication, device accumulation.
			//
			// If productsThenHost matches the host's own `exact`, the two
			// sides agree about the DATA and the reducer's accumulation is
			// what differs. If it matches the reducer instead, toArray is
			// the one not telling the truth.
			const products = tsl_array_n.arrayN( 'float', system.shape );
			tsl_array_n.kernel( system.shape, ( i, j, k ) => {

				products( i, j, k ).assign( system.b( i, j, k ).mul( zField( i, j, k ) ) );

			} )();

			const productsHost = await products.toArray();
			let productsThenHost = 0;
			for ( let i = 0; i < productsHost.length; i ++ ) productsThenHost += productsHost[ i ];

			const onesField = tsl_array_n.arrayN( 'float', system.shape );
			tsl_array_n.kernel( system.shape, ( i, j, k ) => { onesField( i, j, k ).assign( 1 ); } )();
			const productsThenDevice = await linalg.createDotReducer( system.shape, products, onesField ).read();

			// And the same question about a and b themselves: does a kernel
			// see the value that comes back from toArray?
			const disagreements = tsl_array_n.arrayN( 'float', system.shape );
			tsl_array_n.kernel( system.shape, ( i, j, k ) => {

				disagreements( i, j, k ).assign( system.b( i, j, k ) );

			} )();
			const echoed = await disagreements.toArray();
			let echoMismatch = 0, worstEcho = 0;
			for ( let i = 0; i < echoed.length; i ++ ) {

				const delta = Math.abs( echoed[ i ] - bHost[ i ] );
				if ( delta !== 0 ) echoMismatch ++;
				if ( delta > worstEcho ) worstEcho = delta;

			}

			// Per lane, device against host. A reducer that is merely
			// imprecise is wrong everywhere by a little; one that is
			// counting the wrong elements is wrong in particular lanes.
			const devicePartials = await reducer.partial.toArray();
			const laneDiffs = [];
			for ( let lane = 0; lane < lanes; lane ++ ) laneDiffs.push( { lane, diff: devicePartials[ lane ] - perLane[ lane ], device: devicePartials[ lane ], host: perLane[ lane ] } );
			laneDiffs.sort( ( a, c ) => Math.abs( c.diff ) - Math.abs( a.diff ) );
			const totalDiff = laneDiffs.reduce( ( a, x ) => a + x.diff, 0 );

			log(
				`&nbsp;&nbsp;${ label.padEnd( 12 ) } rel err ${ relative.toExponential( 2 ) }, ` +
				`worst per-lane cancellation ${ worstCancellation.toExponential( 2 ) }, ` +
				`reducer ${ first.toExponential( 6 ) } | host products, host sum ${ exact.toExponential( 6 ) } | ` +
				`device products, host sum ${ productsThenHost.toExponential( 6 ) } | ` +
				`device products, device sum ${ productsThenDevice.toExponential( 6 ) } | ` +
				`kernel echo of b differs in ${ echoMismatch } cells (worst ${ worstEcho.toExponential( 1 ) })`,
				relative < 1e-4
			);

			}

		}

	}

	// ---- ?trace=N : PCG by hand, so every scalar it depends on is visible
	//
	// The arms above say that multigrid cannot solve frame 112 while Jacobi
	// and plain CG can. PCG needs its preconditioner to be symmetric AND
	// positive definite; `M` here measured as both, but on RANDOM vectors,
	// and the residual a real scene produces is smooth and concentrated.
	// This runs the iteration in JavaScript, applying the library's own
	// operators on the GPU, and records the scalars PCG actually turns on:
	//
	//   r . z   must keep one sign. This file's A is negative definite, so
	//           M approximates a negative definite inverse and r . z should
	//           stay negative. A sign that wanders is an indefinite
	//           preconditioner, which is enough on its own to stall PCG.
	//   p . Ap  the same, for the operator.
	//   beta    above 1 iteration after iteration is the textbook runaway:
	//           p = z + beta p compounds geometrically.
	//   |z|/|r| how far the preconditioner moves the residual at all.
	if ( TRACE > 0 ) {

		for ( const name of FRAMES ) {

			const capture = await loadCapture( name );
			const system = buildSystem( capture );
			const cells = system.cells;

			const applyLaplacian = linalg.createLaplacianOperator( system.shape, system.gridSpacing, {
				dirichletMask: system.dirichletMask, faceWeights: system.faceWeights
			} );

			const M = linalg.createMultigridPreconditioner( system.shape, system.gridSpacing, {
				...cycleOptions( capture ), dirichletMask: system.dirichletMask,
				coarseDirichletMask: system.coarseDirichletMask, faceWeights: system.faceWeights
			} );
			if ( M.refreshCoarseLevels ) M.refreshCoarseLevels();

			// scratch on the device; the iteration itself is in JS so that
			// every scalar is inspectable rather than inferred
			const inField = tsl_array_n.arrayN( 'float', system.shape );
			const outField = tsl_array_n.arrayN( 'float', system.shape );
			const applyA = applyLaplacian( inField, outField );
			const applyM = M( inField, outField );

			const zero = new Float32Array( cells );
			const apply = async ( dispatch, vector ) => {

				inField.fromArray( vector );
				outField.fromArray( zero ); // a V-cycle relaxes what its output holds
				dispatch();
				return new Float32Array( await outField.toArray() );

			};

			const dot = ( a, c ) => { let t = 0; for ( let i = 0; i < a.length; i ++ ) t += a[ i ] * c[ i ]; return t; };
			const axpy = ( a, x, y ) => { const o = new Float32Array( y.length ); for ( let i = 0; i < y.length; i ++ ) o[ i ] = a * x[ i ] + y[ i ]; return o; };

			let x = new Float32Array( cells );
			let r = Float32Array.from( capture.b );          // x starts at zero, so r = b
			let z = await apply( applyM, r );
			let pdir = Float32Array.from( z );
			let rz = dot( r, z );

			const r0 = Math.sqrt( dot( r, r ) );

			log( '' );
			log( `<b>${ name }</b> — PCG traced by hand, ||r0|| = ${ r0.toExponential( 3 ) }` );
			log( '&nbsp;&nbsp;iter      |r|/|r0|         r.z        p.Ap       alpha        beta    |z|/|r|' );

			for ( let iteration = 0; iteration < TRACE; iteration ++ ) {

				const Ap = await apply( applyA, pdir );
				const pAp = dot( pdir, Ap );
				const alpha = rz / pAp;

				x = axpy( alpha, pdir, x );
				r = axpy( - alpha, Ap, r );

				z = await apply( applyM, r );
				const newRZ = dot( r, z );
				const beta = newRZ / rz;

				const rNorm = Math.sqrt( dot( r, r ) );
				const zNorm = Math.sqrt( dot( z, z ) );


				log(
					`&nbsp;&nbsp;${ String( iteration ).padStart( 4 ) }  ${ ( rNorm / r0 ).toExponential( 3 ).padStart( 12 ) }  ` +
					`${ rz.toExponential( 2 ).padStart( 10 ) }  ${ pAp.toExponential( 2 ).padStart( 10 ) }  ` +
					`${ alpha.toExponential( 2 ).padStart( 10 ) }  ${ beta.toExponential( 2 ).padStart( 10 ) }  ` +
					`${ ( zNorm / Math.max( rNorm, 1e-30 ) ).toExponential( 2 ).padStart( 9 ) }`
				);

				pdir = axpy( beta, pdir, z );
				rz = newRZ;

			}

		}

	}

	log( '' );
	log( 'done', true );

} catch ( error ) {

	log( `failed: ${ error.message }`, false );
	console.error( error );

}
