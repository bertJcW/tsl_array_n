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

function log( label, ok, detail ) {

	lines.push( `<span class="${ ok ? 'ok' : 'err' }">${ ok ? '✓' : '✗' } ${ label }${ detail ? ' — ' + detail : '' }</span>` );
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
					`converged=${ ok } iters=${ st.iterations } res/|b|=${ ( res / bNorm ).toExponential( 2 ) } stoppedBy=${ st.stoppedBy } max|x-xStar|=${ maxErr.toExponential( 2 ) }`
				);

			}

		}

	}

	log( 'done', true );

} catch ( error ) {

	log( 'failed', false, error.message );
	console.error( error );

}
