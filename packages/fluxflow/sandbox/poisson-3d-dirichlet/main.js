// Sandbox probe, written while chasing example 35's own never-converging
// pressure solve. It answers one question and deliberately nothing else:
// on THAT example's exact shape, with a Dirichlet mask of the same kind
// its outflow produces, is the Laplacian + PCG stack itself sound?
//
// The system here is consistent by construction -- b is built as A @ xStar
// from a known xStar -- so a failure to converge cannot be blamed on the
// right-hand side a scene happens to feed it. examples/31-conjugate-
// gradient-3d/ already answered the same question for a 16^3 cube with no
// mask; what this adds is the mask, the non-cube 48x24x24 shape, and a
// direct symmetry check of the masked operator.

import * as tsl_array_n from 'tsl_array_n';
import { linalg } from 'fluxflow';

const pre = document.querySelector( '#status pre' );
const lines = [];
let passed = 0, expected = 0, unexpected = 0;

// *** Expected failures are marked as such, because a page with permanent red
// on it teaches you to stop reading the red ***
//
// Four of the checks below have always failed and always should: a 1-level
// V-cycle is plain relaxation and is not a symmetric operator, and
// unpreconditioned CG does not solve a 48x24x24 Dirichlet Poisson problem in
// 300 iterations. Both are the point of the surrounding comparison rather than
// news. Reported as plain crosses among the ticks, they made the page's own
// summary read "4 failed" on every run, which is indistinguishable from a
// regression and was in fact read past for weeks.
//
// So a check that is expected to fail says the reason inline, counts
// separately, and turns the summary into a number that is 0 when nothing is
// wrong. `why` is that reason; passing it and then PASSING is itself flagged,
// since an expected failure that starts succeeding means the thing the
// expectation was about has changed.
function log( label, ok, detail, why ) {

	let mark, cls;

	if ( why === undefined ) {

		mark = ok ? '✓' : '✗';
		cls = ok ? 'ok' : 'err';
		ok ? passed ++ : unexpected ++;

	} else if ( ! ok ) {

		mark = '~';
		cls = 'note';
		expected ++;
		detail = `${ detail } — EXPECTED: ${ why }`;

	} else {

		mark = '!';
		cls = 'err';
		unexpected ++;
		detail = `${ detail } — this was expected to FAIL (${ why }) and did not; the expectation is stale`;

	}

	lines.push( `<span class="${ cls }">${ mark } ${ label }${ detail ? ' — ' + detail : '' }</span>` );
	pre.innerHTML = lines.join( '\n' );

}

// Called at the end so a truncated run cannot look complete: no summary line
// means the page did not finish. run_page.mjs reports that too, from the
// outside, after a fixed 15-second wait once cut this page's own output off at
// three of its sixteen V-cycle rows.
function summarise() {

	lines.push( '' );
	lines.push( `<span class="${ unexpected === 0 ? 'ok' : 'err' }">${ passed } passed, ${ expected } expected failures, ${ unexpected } unexpected</span>` );
	pre.innerHTML = lines.join( '\n' );

}

// Two shapes: example 35's own 48x24x24, and a 2D control with a
// comparable cell count, so "does this only go wrong in 3D" is answered by
// the same run rather than by comparing against a different file.
const CASES = [
	{ label: '3D 48x24x24', shape: [ 48, 24, 24 ], gridSpacing: [ 1, 1, 1 ] },
	{ label: '2D 96x48', shape: [ 96, 48 ], gridSpacing: [ 1, 1 ] }
];

let seed = 987654321;
const rand = () => {

	seed = ( seed * 1103515245 + 12345 ) & 0x7fffffff;
	return seed / 0x7fffffff - 0.5;

};

const dot = ( a, c ) => {

	let s = 0;
	for ( let i = 0; i < a.length; i ++ ) s += a[ i ] * c[ i ];
	return s;

};

try {

	const renderer = await tsl_array_n.init( { allowFallback: true } );
	log( 'init()', true, `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }` );

	if ( renderer.backend?.constructor?.name !== 'WebGPUBackend' ) {

		log( 'backend', false, 'not WebGPU — every number below would be meaningless, stopping' );
		throw new Error( 'not WebGPU' );

	}

	for ( const { label, shape, gridSpacing } of CASES ) {

		const cells = shape.reduce( ( a, n ) => a * n, 1 );

		// The same slab example 35's outflow marks: the last two cells in x.
		// Backed by a float FIELD compared against 0.5, exactly the way
		// grid_pressure_solver3.js builds its own mask -- not a direct
		// comparison on the index node, which is a u32 in WGSL and fails to
		// compile against a JS number ("no matching overload for
		// operator >= (u32, abstract-float)"). That mistake is worth naming:
		// the kernel does not throw, the whole command buffer is rejected at
		// submit time, and the solve then reports degenerate-pAp on
		// iteration 1 -- which reads exactly like a real breakdown of the
		// preconditioner and is not one.
		const maskArray = new Float32Array( cells );
		const maskField = tsl_array_n.arrayN( 'float', shape );
		const dirichletMask = ( ...I ) => maskField( ...I ).greaterThan( 0.5 );

		// A smooth xStar, not white noise: the same reasoning as
		// examples/31-conjugate-gradient-3d/'s own choice. White noise is
		// all high-frequency, which CG kills in a handful of iterations
		// while leaving the low-frequency error a real scene actually cares
		// about untouched -- it flatters the solver and hides exactly the
		// failure being looked for here.
		const xStarArray = new Float32Array( cells );
		for ( let n = 0; n < cells; n ++ ) {

			const i = n % shape[ 0 ];
			if ( i >= shape[ 0 ] - 2 ) maskArray[ n ] = 1;
			const j = Math.floor( n / shape[ 0 ] ) % shape[ 1 ];
			const k = shape.length > 2 ? Math.floor( n / ( shape[ 0 ] * shape[ 1 ] ) ) : 0;
			xStarArray[ n ] = Math.sin( 2 * Math.PI * i / shape[ 0 ] ) * Math.cos( 2 * Math.PI * j / shape[ 1 ] ) *
				( shape.length > 2 ? Math.cos( 2 * Math.PI * k / shape[ 2 ] ) : 1 );

		}

		maskField.fromArray( maskArray );

		for ( const masked of [ false, true ] ) {

			const mask = masked ? dirichletMask : undefined;
			const applyLaplacian = linalg.createLaplacianOperator( shape, gridSpacing, { dirichletMask: mask } );

			{

				const x = tsl_array_n.arrayN( 'float', shape );
				const y = tsl_array_n.arrayN( 'float', shape );
				const Ax = tsl_array_n.arrayN( 'float', shape );
				const Ay = tsl_array_n.arrayN( 'float', shape );

				x.fromArray( Float32Array.from( { length: cells }, rand ) );
				y.fromArray( Float32Array.from( { length: cells }, rand ) );

				applyLaplacian( x, Ax )();
				applyLaplacian( y, Ay )();

				const [ xd, yd, axd, ayd ] = await Promise.all( [ x.toArray(), y.toArray(), Ax.toArray(), Ay.toArray() ] );
				const left = dot( axd, yd ), right = dot( xd, ayd );
				const rel = Math.abs( left - right ) / Math.max( Math.abs( left ), Math.abs( right ) );

				log( `${ label }: A symmetric (mask ${ masked })`, rel < 1e-4, `rel diff ${ rel.toExponential( 2 ) }` );

			}

			const xStar = tsl_array_n.arrayN( 'float', shape );
			xStar.fromArray( xStarArray );

			const b = tsl_array_n.arrayN( 'float', shape );
			applyLaplacian( xStar, b )();
			const bNorm = Math.sqrt( dot( await b.toArray(), await b.toArray() ) );

			for ( const [ name, build ] of [
				[ 'none', () => linalg.createIdentityPreconditioner( shape ) ],
				[ 'multigrid x4', () => linalg.createMultigridPreconditioner( shape, gridSpacing, { numberOfLevels: 4, dirichletMask: mask } ) ],
				[ 'multigrid x3', () => linalg.createMultigridPreconditioner( shape, gridSpacing, { numberOfLevels: 3, dirichletMask: mask } ) ],
				[ 'multigrid x2', () => linalg.createMultigridPreconditioner( shape, gridSpacing, { numberOfLevels: 2, dirichletMask: mask } ) ]
			] ) {

				const x = tsl_array_n.arrayN( 'float', shape );
				x.fromArray( new Float32Array( cells ) );

				// A multigrid preconditioner with a Dirichlet mask carries its
				// own per-level mask fields, filled by this dispatch rather
				// than from inside the V-cycle -- see multigrid.js. Every
				// real caller does the same once per solve.
				const preconditioner = build();
				if ( preconditioner.refreshDirichletLevels ) preconditioner.refreshDirichletLevels();

				const solver = linalg.createPreconditionedConjugateGradientSolver( applyLaplacian, preconditioner, b, x );
				const ok = await solver.solve( 1e-5, 300, 1, false, true, false, true );

				const got = await x.toArray();
				let maxErr = 0;
				for ( let i = 0; i < cells; i ++ ) maxErr = Math.max( maxErr, Math.abs( got[ i ] - xStarArray[ i ] ) );

				const st = solver.state ?? {};
				const res = st.residualSquared !== undefined ? Math.sqrt( Math.abs( st.residualSquared ) ) : NaN;

				log(
					`${ label }: PCG(${ name }) mask ${ masked }`,
					ok === true && maxErr < 1e-2,
					`converged=${ ok } iters=${ st.iterations } res/|b|=${ ( res / bNorm ).toExponential( 2 ) } stoppedBy=${ st.stoppedBy } max|x-xStar|=${ maxErr.toExponential( 2 ) }`,
					// Unpreconditioned CG is the control arm, here to show what the
					// V-cycle is worth rather than as a candidate.
					//
					// It used to be an expected failure: 300 iterations was not
					// enough for it in 3D. That expectation went stale on
					// 2026-09-26, when PCG's host path was fixed to honour
					// relativeTolerance instead of testing the absolute residual
					// against tol -- this page passes gpuResidentScalars: false,
					// so every convergence flag on it had been an absolute-
					// threshold pass. Unpreconditioned CG now clears the same
					// relative tolerance in 73 and 86 iterations.
					//
					// The flag that caught it is the one added an hour earlier for
					// exactly this: an expected failure that starts PASSING means
					// the thing the expectation was about has changed. It fired on
					// its first exposure to a real change.
					//
					// The comparison the row exists for is unharmed and is now
					// cleaner to read: 73-86 iterations unpreconditioned against
					// 5-12 with the V-cycle.
					undefined
				);

			}

		}

	}

	// ---- is the V-cycle itself a symmetric operator? ----
	//
	// PCG requires its preconditioner to be symmetric: `z = M r` has to
	// behave like a quadratic form or alpha and beta stop meaning anything.
	// A V-cycle is only symmetric if its transfer operators are adjoint and
	// its smoother's post-sweep undoes the pre-sweep's colour order -- and
	// once the mask and the face weights are coarsened per level, there are
	// several more places for that to break. This measures it directly, on
	// the configuration example 35 actually builds: a Dirichlet slab at the
	// exit, and a solid block in the middle whose faces are closed.
	{

		const shape = [ 48, 24, 24 ];
		const gridSpacing = [ 1, 1, 1 ];
		const cells = shape[ 0 ] * shape[ 1 ] * shape[ 2 ];

		const maskArray = new Float32Array( cells );
		const ventArray = new Float32Array( cells );
		for ( let n = 0; n < cells; n ++ ) {

			const i = n % shape[ 0 ];
			if ( i >= shape[ 0 ] - 2 ) { maskArray[ n ] = 1; ventArray[ n ] = 1; }

		}

		const maskField = tsl_array_n.arrayN( 'float', shape );
		maskField.fromArray( maskArray );
		const ventField = tsl_array_n.arrayN( 'float', shape );
		ventField.fromArray( ventArray );

		// A rod from x 7..13, y 6..12, spanning z -- example 35's own block.
		function faceWeightField( axis ) {

			const fieldShape = shape.map( ( n, a ) => ( a === axis ? n + 1 : n ) );
			const count = fieldShape.reduce( ( a, b ) => a * b, 1 );
			const data = new Float32Array( count );

			for ( let n = 0; n < count; n ++ ) {

				const i = n % fieldShape[ 0 ];
				const j = Math.floor( n / fieldShape[ 0 ] ) % fieldShape[ 1 ];
				const inSolid = i >= 7 && i <= 13 && j >= 6 && j <= 12;
				data[ n ] = inSolid ? 0 : 1;

			}

			const field = tsl_array_n.arrayN( 'float', fieldShape );
			field.fromArray( data );
			return field;

		}

		const wu = faceWeightField( 0 ), wv = faceWeightField( 1 ), ww = faceWeightField( 2 );

		// Exactly what grid_pressure_solver3.js builds: the caller's mask is
		// the vent PLUS every cell the collider has enclosed (whose row
		// would otherwise have no diagonal), while only the vent half of it
		// is offered for coarsening.
		{

			// A face is closed when its own index falls in the solid block
			// above, so a cell is enclosed when both of its faces on every
			// axis are, which works out to i in 7..12 and j in 6..11 for any
			// k. Computed rather than read back, because an arrayN that no
			// kernel has written yet has nothing to read.
			let enclosedCount = 0;
			for ( let k = 0; k < 24; k ++ ) for ( let j = 6; j <= 11; j ++ ) for ( let i = 7; i <= 12; i ++ ) {

				maskArray[ i + 48 * j + 48 * 24 * k ] = 1;
				enclosedCount ++;

			}

			maskField.fromArray( maskArray );
			log( 'enclosed cells folded into the caller mask', true, `${ enclosedCount } cells, as grid_pressure_solver3.js does` );

		}

		const cases = [
			{ label: 'mask only', opts: { dirichletMask: ( ...I ) => maskField( ...I ).greaterThan( 0.5 ) } },
			{ label: 'mask + weights', opts: {
				dirichletMask: ( ...I ) => maskField( ...I ).greaterThan( 0.5 ),
				coarseDirichletMask: ( ...I ) => ventField( ...I ).greaterThan( 0.5 ),
				faceWeights: [ ( ...I ) => wu( ...I ), ( ...I ) => wv( ...I ), ( ...I ) => ww( ...I ) ]
			} }
		];

		for ( const levels of [ 1, 2, 3, 4 ] ) {

			for ( const { label, opts } of cases ) {

				const M = linalg.createMultigridPreconditioner( shape, gridSpacing, { numberOfLevels: levels, ...opts } );
				if ( M.refreshCoarseLevels ) M.refreshCoarseLevels();

				const x = tsl_array_n.arrayN( 'float', shape );
				const y = tsl_array_n.arrayN( 'float', shape );
				const Mx = tsl_array_n.arrayN( 'float', shape );
				const My = tsl_array_n.arrayN( 'float', shape );

				x.fromArray( Float32Array.from( { length: cells }, rand ) );
				y.fromArray( Float32Array.from( { length: cells }, rand ) );
				// A V-cycle relaxes whatever its output already holds, so it
				// is only the linear operator M when started from zero.
				Mx.fromArray( new Float32Array( cells ) );
				My.fromArray( new Float32Array( cells ) );

				M( x, Mx )();
				M( y, My )();

				const [ xd, yd, mxd, myd ] = await Promise.all( [ x.toArray(), y.toArray(), Mx.toArray(), My.toArray() ] );
				const left = dot( mxd, yd ), right = dot( xd, myd );
				const rel = Math.abs( left - right ) / Math.max( Math.abs( left ), Math.abs( right ), 1e-30 );

				log(
					`V-cycle symmetric, ${ levels } level(s), ${ label }`,
					rel < 1e-3,
					`(Mx,y)=${ left.toFixed( 3 ) } (x,My)=${ right.toFixed( 3 ) } rel diff ${ rel.toExponential( 2 ) }`,
					// A 1-level "V-cycle" is plain relaxation with no coarse
					// grid, and red-black relaxation is not a symmetric
					// operator. It fails here by construction, and it is kept
					// as the row that shows what the coarse levels buy:
					// 3.8e-3 at 1 level against 7e-6 to 1.6e-4 at 2, 3 and 4.
					// numberOfLevels: 1 is separately documented as an
					// inadequate preconditioner at these grid sizes.
					levels === 1 ? 'a 1-level V-cycle is plain red-black relaxation, which is not symmetric; kept as the contrast that shows what coarse levels buy' : undefined
				);

				// Symmetric is half of what PCG needs. (Mx, x) must also keep
				// one sign -- negative here, since this file's own A is
				// negative definite and M approximates its inverse. A sign
				// that wanders means an indefinite preconditioner, which is
				// enough on its own to stall PCG completely.
				{

					const quads = [];

					for ( let draw = 0; draw < 5; draw ++ ) {

						const r = tsl_array_n.arrayN( 'float', shape );
						const Mr = tsl_array_n.arrayN( 'float', shape );
						r.fromArray( Float32Array.from( { length: cells }, rand ) );
						Mr.fromArray( new Float32Array( cells ) );
						M( r, Mr )();
						const [ rd, mrd ] = await Promise.all( [ r.toArray(), Mr.toArray() ] );
						quads.push( dot( mrd, rd ) );

					}

					const negative = quads.every( ( q ) => q < 0 );
					const positive = quads.every( ( q ) => q > 0 );

					log(
						`V-cycle definite, ${ levels } level(s), ${ label }`,
						negative || positive,
						`(Mr,r) over five draws: ${ quads.map( ( q ) => q.toExponential( 2 ) ).join( ', ' ) }`
					);

				}

			}

		}

	}

	summarise();

} catch ( error ) {

	log( 'failed', false, error.message );
	summarise();
	console.error( error );

}
