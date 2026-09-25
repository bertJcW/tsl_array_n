// Step 1 of the 3D effort: does the part of this library that was already
// built dimension-generic -- linalg.js's CG/PCG solvers and multigrid.js's
// preconditioner, both explicitly written to branch on shape.length rather
// than assume 2D -- actually work in genuine 3D? Nothing in src/grid/ is
// touched or ported here; every file there is a 2D-specific implementation
// (the "2" in grid_math.js, grid_data2.js, etc.) and stays exactly as it is.
// This example exists to answer one question before any of that starts:
// is the foundation those files would sit on already sound in 3D, or does
// it need fixing first.
//
// Structure mirrors examples/07-multigrid-preconditioned-cg/ exactly, one
// dimension up: same real Poisson problem (not the diagonal toy case
// examples/04-/05- use, which never exercises a shape-dependent code path
// at all), same "construct b from a chosen xExpected" trick for the same
// reason (a pure zero-flux/Neumann Laplacian has constants in its null
// space, so an arbitrary b might not be solvable, and picking xExpected to
// be zero-mean is what makes a direct comparison against it meaningful
// rather than "correct up to an additive constant"), same restrict/
// correct adjoint-pair check afterward.
//
// Not yet run on real WebGPU hardware as of writing -- same class of
// GPU-atomic-dot-product code as examples/04-/05-/07-, so the same
// fallback-only failure mode is expected in a WebGL2 sandbox and proves
// nothing either way there.

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
const shape = [ N, N, N ];
const gridSpacing = [ 1, 1, 1 ];

try {

	const renderer = await tsl_array_n.init( { allowFallback: true } );
	log( 'init()', true, `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }` );

	// xExpected(i,j,k) = sin(2*pi*i/N) * cos(2*pi*j/N) * cos(2*pi*k/N) --
	// zero-mean for the same reason the 2D version is: summed over i alone,
	// the sin factor sums to exactly zero regardless of what the other two
	// factors are. tsl_array_n's flat layout for a 3D shape is
	// i + N*j + N*N*k (computeStrides in array.js: stride[0]=1,
	// stride[1]=shape[0], stride[2]=shape[0]*shape[1]).
	const xExpectedArray = new Float32Array( N * N * N );
	for ( let k = 0; k < N; k ++ ) {

		for ( let j = 0; j < N; j ++ ) {

			for ( let i = 0; i < N; i ++ ) {

				xExpectedArray[ i + N * j + N * N * k ] =
					Math.sin( 2 * Math.PI * i / N ) * Math.cos( 2 * Math.PI * j / N ) * Math.cos( 2 * Math.PI * k / N );

			}

		}

	}

	const applyLaplacian = linalg.createLaplacianOperator( shape, gridSpacing );

	const xExpected = tsl_array_n.arrayN( 'float', shape );
	xExpected.fromArray( xExpectedArray );

	const b = tsl_array_n.arrayN( 'float', shape );
	const dispatchB = applyLaplacian( xExpected, b );
	dispatchB(); // b = A @ xExpected

	const x = tsl_array_n.arrayN( 'float', shape );
	x.fromArray( new Float32Array( N * N * N ) );

	const applyPreconditioner = linalg.createMultigridPreconditioner( shape, gridSpacing, { numberOfLevels: 4 } );
	const solver = linalg.createPreconditionedConjugateGradientSolver( applyLaplacian, applyPreconditioner, b, x );

	const succeeded = await solver.solve( 1e-5, 50 );

	const result = Array.from( await x.toArray() );
	const expected = Array.from( xExpectedArray );

	// Length-guarded for the same reason examples/07's own check is -- see
	// that file's header comment: an empty/short GPU readback would
	// otherwise pass [].every(...) vacuously.
	const matches = result.length === expected.length && result.every( ( v, i ) => Math.abs( v - expected[ i ] ) < 1e-2 );

	const detail = matches
		? `converged to xExpected (max |diff| < 1e-2), succeeded=${ succeeded }`
		: result.length !== expected.length
			? `x.toArray() returned ${ result.length } elements, expected ${ expected.length } -- empty/short readback, not a convergence failure`
			: `did not match xExpected -- max |diff| = ${ Math.max( ...result.map( ( v, i ) => Math.abs( v - expected[ i ] ) ) ).toFixed( 4 ) }, succeeded=${ succeeded }`;

	log( 'multigrid-preconditioned CG — 3D Poisson, 16x16x16, zero-flux boundary', succeeded && matches, detail );

	// Same adjoint-pair check as examples/07's own, one dimension up -- see
	// that file's header comment for the full reasoning (a V-cycle is only
	// a symmetric operator, which PCG requires, if restriction and
	// prolongation are adjoint or differ by a constant scale).
	{

		const fineShape = [ N, N, N ];
		const coarseShape = [ N / 2, N / 2, N / 2 ];

		const u = tsl_array_n.arrayN( 'float', fineShape ); // fine
		const v = tsl_array_n.arrayN( 'float', coarseShape ); // coarse
		const Ru = tsl_array_n.arrayN( 'float', coarseShape );
		const Pv = tsl_array_n.arrayN( 'float', fineShape );

		const restrict = linalg.buildRestrictKernel( u, Ru, coarseShape, undefined );
		const prolong = linalg.buildCorrectKernel( v, Pv, fineShape, undefined );

		let seed = 12345;
		const rand = () => {

			seed = ( seed * 1103515245 + 12345 ) & 0x7fffffff;
			return seed / 0x7fffffff - 0.5;

		};

		async function adjointRatio() {

			u.fromArray( Float32Array.from( { length: N * N * N }, rand ) );
			v.fromArray( Float32Array.from( { length: ( N / 2 ) * ( N / 2 ) * ( N / 2 ) }, rand ) );
			Pv.fromArray( new Float32Array( N * N * N ) ); // buildCorrectKernel ADDS, so start at zero

			restrict();
			prolong();

			const [ ruData, vData, uData, pvData ] = await Promise.all( [
				Ru.toArray(), v.toArray(), u.toArray(), Pv.toArray()
			] );

			let left = 0, right = 0;
			for ( let i = 0; i < ruData.length; i ++ ) left += ruData[ i ] * vData[ i ];
			for ( let i = 0; i < uData.length; i ++ ) right += uData[ i ] * pvData[ i ];

			return { left, right, ratio: left !== 0 ? right / left : NaN };

		}

		const draws = [ await adjointRatio(), await adjointRatio(), await adjointRatio() ];
		const ratios = draws.map( ( d ) => d.ratio );
		const spread = Math.max( ...ratios ) - Math.min( ...ratios );
		const constant = spread / Math.abs( ratios[ 0 ] ) < 1e-4;

		log(
			'restriction and prolongation differ by a constant — (u,Pv) / (Ru,v)',
			constant,
			`ratios over three random draws: ${ ratios.map( ( r ) => r.toFixed( 6 ) ).join( ', ' ) } ` +
			`(spread ${ spread.toExponential( 2 ) }). ` +
			( constant
				? `A constant factor of ${ ratios[ 0 ].toFixed( 4 ) } does not break the V-cycle's symmetry.`
				: 'A wandering ratio WOULD break it: R is not a scalar multiple of P^T.' )
		);

	}

	// *** The check this example should have had, and did not ***
	//
	// Everything above passed for as long as createDotReducer's 3D branch
	// was summing a diagonal twenty-four times over instead of the whole
	// field -- see linalg.js's own comment there for the mechanism and what
	// it cost. A multigrid-preconditioned solve reaches a smooth answer to
	// 1e-2 even when every alpha and beta is scaled wrongly, so the test
	// above cannot see it, and neither can the obvious direct test: a field
	// of ones reduces to exactly the right total however wrongly its cells
	// are chosen, because any 576 ones sum to 576.
	//
	// What does see it is a single cell. Set one cell to one, reduce
	// against a field of ones, and the answer must be 1, in the lane
	// matching that cell's first index and nowhere else. Under the bug a
	// delta at (5,0,0) reduced to 24 and one at (0,0,1) to 0.
	{

		const probeShape = [ 6, 4, 5 ]; // deliberately not a cube, so an axis swap shows up too
		const ones = tsl_array_n.arrayN( 'float', probeShape );
		const delta = tsl_array_n.arrayN( 'float', probeShape );

		tsl_array_n.kernel( probeShape, ( i, j, k ) => { ones( i, j, k ).assign( 1 ); } )();

		const reducer = linalg.createDotReducer( probeShape, delta, ones );

		const probes = [ [ 0, 0, 0 ], [ 0, 0, 1 ], [ 0, 1, 0 ], [ 5, 3, 4 ], [ 2, 3, 1 ] ];
		const failures = [];

		for ( const [ pi, pj, pk ] of probes ) {

			tsl_array_n.kernel( probeShape, ( i, j, k ) => {

				delta( i, j, k ).assign( i.equal( pi ).and( j.equal( pj ) ).and( k.equal( pk ) ).select( 1, 0 ) );

			} )();

			const total = await reducer.read();
			const lanes = Array.from( await reducer.partial.toArray() );
			const lit = lanes.map( ( value, lane ) => ( value !== 0 ? `${ lane }:${ value }` : null ) ).filter( Boolean );

			if ( total !== 1 || lit.length !== 1 || lit[ 0 ] !== `${ pi }:1` ) {

				failures.push( `(${ pi },${ pj },${ pk }) gave ${ total } with lanes [${ lit.join( ' ' ) }]` );

			}

		}

		// And the whole field, which a wrong index mapping gets right and a
		// wrong sum does not: every cell holding its own flat index, so the
		// total is a number only the correct traversal produces.
		const ramp = tsl_array_n.arrayN( 'float', probeShape );
		const [ px, py ] = probeShape;
		tsl_array_n.kernel( probeShape, ( i, j, k ) => {

			ramp( i, j, k ).assign( i.add( j.mul( px ) ).add( k.mul( px * py ) ).toFloat() );

		} )();

		const cells = probeShape.reduce( ( a, n ) => a * n, 1 );
		const expected = ( cells - 1 ) * cells / 2; // the sum of 0..cells-1
		const rampTotal = await linalg.createDotReducer( probeShape, ramp, ones ).read();

		log(
			'the dot reducer visits every cell exactly once — single-cell probes, and a ramp',
			failures.length === 0 && rampTotal === expected,
			failures.length === 0 && rampTotal === expected
				? `five single-cell probes each reduced to 1 in their own lane, and a ${ probeShape.join( 'x' ) } ramp summed to ${ rampTotal }`
				: `${ failures.join( '; ' ) }${ failures.length ? '; ' : '' }ramp summed to ${ rampTotal }, expected ${ expected }`
		);

	}

} catch ( error ) {

	log( 'failed', false, error.message );
	console.error( error );

}
