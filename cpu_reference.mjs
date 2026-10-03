// Solve an exported pressure system on the CPU, in double precision, with no
// GPU anywhere in it.
//
// The 2D smoke scenes floor at a relative residual near 1.4e-5 with the
// multigrid preconditioner. Four candidates are eliminated (the stop test, the
// operator's symmetry, the shape and mask, and any geometric blind spot). Two
// possibilities are left and they call for opposite work: either that
// right-hand side is genuinely hard, or something in the GPU implementation
// cannot go below that level. A double-precision CPU solve of the identical
// system separates them.
//
// *** The CPU operator has to BE the GPU operator, and that is checked before
// anything else is believed ***
//
// A reimplementation that merely looks right proves nothing: if the two
// stencils differ, the comparison measures the difference between them rather
// than between the machines. Two cross-checks run first. The operator must be
// symmetric to double-precision rounding, and -- the one that actually binds --
// ||b - A @ pressureAfter|| computed here must reproduce the residual the GPU
// solver reported for that same pressure. If it does, the two operators agree
// on the vector that matters.
//
// The stencil mirrors multigrid.js's laplacianAt exactly, including the part
// that is easy to miss: a pinned cell's own row is not a Laplacian row at all,
// it is -x (see that function's closing `select( center.negate(), sum )`), and
// a pinned NEIGHBOUR is eliminated with opposite substitutions on the two
// sides -- `center` for the lower difference and `-center` for the upper.
//
// usage: node cpu_reference.mjs [dir] [maxIterations]

import { readFileSync } from 'node:fs';

const DIR = process.argv[ 2 ] ?? 'system-2d';
const MAX_ITERATIONS = Number( process.argv[ 3 ] ?? 40000 );

const meta = JSON.parse( readFileSync( `${ DIR }/meta.json`, 'utf8' ) );
const readField = ( name ) => new Float32Array( readFileSync( `${ DIR }/${ name }.bin` ).buffer.slice( 0 ) );

const [ NX, NY ] = meta.shape;
const CELLS = NX * NY;
const [ HX, HY ] = meta.gridSpacing;

const b = Float64Array.from( readField( 'b' ) );
const maskRaw = readField( 'dirichletMask' );
const pinned = new Uint8Array( CELLS );
for ( let n = 0; n < CELLS; n ++ ) pinned[ n ] = maskRaw[ n ] > 0.5 ? 1 : 0;
const pressureAfter = Float64Array.from( readField( 'pressureAfter' ) );
const pressureBefore = Float64Array.from( readField( 'pressureBefore' ) );

const at = ( i, j ) => i + NX * j;

console.log( `${ meta.url } frame ${ meta.frame }` );
console.log( `${ NX }x${ NY }, ${ CELLS } cells, ${ meta.pinned } pinned, spacing ${ HX }x${ HY }, |b| = ${ meta.bNorm.toExponential( 4 ) }` );
console.log( `the GPU reported: ${ meta.gpu.iterations } iterations, residual ${ meta.gpu.residual.toExponential( 4 ) } (${ ( meta.gpu.residual / meta.bNorm ).toExponential( 2 ) } relative), converged ${ meta.gpu.converged }\n` );

// ---- the operator -------------------------------------------------------

function applyA( x, out, round ) {

	const f = round ? Math.fround : ( v ) => v;
	const ihx2 = f( 1 / ( HX * HX ) );
	const ihy2 = f( 1 / ( HY * HY ) );

	for ( let j = 0; j < NY; j ++ ) {

		for ( let i = 0; i < NX; i ++ ) {

			const c = at( i, j );
			const center = x[ c ];

			if ( pinned[ c ] ) { out[ c ] = f( - center ); continue; }

			let dLower = 0, dUpper = 0;

			if ( i > 0 ) dLower = pinned[ at( i - 1, j ) ] ? center : f( center - x[ at( i - 1, j ) ] );
			if ( i < NX - 1 ) dUpper = pinned[ at( i + 1, j ) ] ? - center : f( x[ at( i + 1, j ) ] - center );
			let sum = f( f( dUpper - dLower ) * ihx2 );

			dLower = 0; dUpper = 0;
			if ( j > 0 ) dLower = pinned[ at( i, j - 1 ) ] ? center : f( center - x[ at( i, j - 1 ) ] );
			if ( j < NY - 1 ) dUpper = pinned[ at( i, j + 1 ) ] ? - center : f( x[ at( i, j + 1 ) ] - center );
			sum = f( sum + f( f( dUpper - dLower ) * ihy2 ) );

			out[ c ] = sum;

		}

	}

}

// *** Three dot products, because the accumulator is the whole question ***
//
// The first version of this file ran a "float32" arm that rounded every
// arithmetic step with Math.fround and then summed the dot products in a plain
// JS number -- which is float64. So the arm that was supposed to model single
// precision was accumulating its reductions in double, and the one structural
// difference from the GPU went untested. The GPU sums in float32, partitioned
// across lanes.
//
// Three accumulators, then. Double is the reference. Naive float32 is the worst
// case, rounding after every term, and is pessimistic: nothing sums 12,288
// terms serially on a GPU. Lane-partitioned float32 is the shape the GPU
// actually uses -- independent float32 partial sums combined at the end -- and
// is the one to compare against.
const dot = ( a, c ) => { let s = 0; for ( let i = 0; i < a.length; i ++ ) s += a[ i ] * c[ i ]; return s; };

const dotF32Naive = ( a, c ) => {

	let s = 0;
	for ( let i = 0; i < a.length; i ++ ) s = Math.fround( s + Math.fround( a[ i ] * c[ i ] ) );
	return s;

};

const LANES = 256;

const dotF32Lanes = ( a, c ) => {

	const partial = new Float32Array( LANES );
	for ( let i = 0; i < a.length; i ++ ) partial[ i % LANES ] = Math.fround( partial[ i % LANES ] + Math.fround( a[ i ] * c[ i ] ) );
	let s = 0;
	for ( let l = 0; l < LANES; l ++ ) s = Math.fround( s + partial[ l ] );
	return s;

};

const norm = ( a ) => Math.sqrt( dot( a, a ) );

// ---- cross-check 1: is this operator symmetric? -------------------------

{
	let seed = 12345;
	const rand = () => { seed = ( seed * 1103515245 + 12345 ) & 0x7fffffff; return seed / 0x7fffffff - 0.5; };
	const x = Float64Array.from( { length: CELLS }, rand );
	const y = Float64Array.from( { length: CELLS }, rand );
	const Ax = new Float64Array( CELLS ), Ay = new Float64Array( CELLS );
	applyA( x, Ax, false );
	applyA( y, Ay, false );
	const l = dot( Ax, y ), r = dot( x, Ay );
	const rel = Math.abs( l - r ) / Math.max( norm( Ax ) * norm( y ), 1e-30 );
	console.log( `${ rel < 1e-12 ? '✓' : '✗' } CPU operator symmetric -- |(Ax,y)-(x,Ay)| / ||Ax||||y|| = ${ rel.toExponential( 2 ) }` );
}

// ---- cross-check 2: does it agree with the GPU on the GPU's own answer? --

{
	const Ap = new Float64Array( CELLS );
	applyA( pressureAfter, Ap, false );
	const r = new Float64Array( CELLS );
	for ( let n = 0; n < CELLS; n ++ ) r[ n ] = b[ n ] - Ap[ n ];
	const rn = norm( r );
	const ratio = rn / meta.gpu.residual;
	console.log( `${ ratio > 0.5 && ratio < 2 ? '✓' : '✗' } CPU operator agrees with the GPU's -- ||b - A@pressure|| = ${ rn.toExponential( 4 ) } against the GPU's reported ${ meta.gpu.residual.toExponential( 4 ) }, ratio ${ ratio.toFixed( 3 ) }` );
	if ( ! ( ratio > 0.5 && ratio < 2 ) ) {
		console.log( '   the two operators do not agree, so nothing below would be a comparison between machines' );
		process.exit( 2 );
	}
}

console.log( '' );

// ---- plain CG, from zero ------------------------------------------------

// *** The starting guess is not a detail, and comparing against x = 0 was an
// unfair comparison ***
//
// The live solver warm-starts from the previous frame's pressure; this file
// started from zero. If the warm start is what slows CG down, then the GPU
// taking 3000 iterations where this takes 490 is not a GPU problem at all --
// the two were not solving the same problem. So the start is an argument.
// recomputeEvery: rebuild r as b - A@x from scratch every N iterations instead
// of updating it incrementally. The live solver does this EVERY iteration --
// linalg.js's RESIDUAL_RECOMPUTE_INTERVAL is 1, deliberately, so that what the
// stop test sees is the true residual and not a drifted one. That is the right
// call for honesty and it is the one thing the arms above did not model: they
// all ran the incremental r throughout.
//
// It matters exactly where the floor is. b - A@x in float32, once ||r|| is far
// below ||A@x||, is a cancellation: the two operands agree to within the
// residual, so the difference keeps only the digits they disagree in. Below
// some level the recomputed residual is noise, and CG cannot descend on noise.
function cg( round, maxIterations, label, dotFn = dot, x0 = null, recomputeEvery = 0 ) {

	const f = round ? Math.fround : ( v ) => v;
	const x = x0 ? Float64Array.from( x0 ) : new Float64Array( CELLS );
	const r = new Float64Array( CELLS );
	const p = new Float64Array( CELLS );
	const Ap = new Float64Array( CELLS );

	if ( x0 ) {

		const Ax = new Float64Array( CELLS );
		applyA( x, Ax, round );
		for ( let n = 0; n < CELLS; n ++ ) { r[ n ] = f( b[ n ] - Ax[ n ] ); p[ n ] = r[ n ]; }

	} else {

		for ( let n = 0; n < CELLS; n ++ ) { r[ n ] = f( b[ n ] ); p[ n ] = r[ n ]; }

	}

	let rr = dotFn( r, r );
	const bNorm = norm( b );
	const target = 1e-6 * bNorm;

	const history = [];
	let best = Math.sqrt( rr );
	let iterations = 0;

	for ( let k = 0; k < maxIterations; k ++ ) {

		applyA( p, Ap, round );
		const pAp = dotFn( p, Ap );
		if ( ! Number.isFinite( pAp ) || pAp === 0 ) break;

		const alpha = f( rr / pAp );
		for ( let n = 0; n < CELLS; n ++ ) { x[ n ] = f( x[ n ] + alpha * p[ n ] ); r[ n ] = f( r[ n ] - alpha * Ap[ n ] ); }

		if ( recomputeEvery > 0 && ( k + 1 ) % recomputeEvery === 0 ) {

			const Ax = new Float64Array( CELLS );
			applyA( x, Ax, round );
			for ( let n = 0; n < CELLS; n ++ ) r[ n ] = f( b[ n ] - Ax[ n ] );

		}

		const rrNew = dotFn( r, r );
		const res = Math.sqrt( Math.abs( rrNew ) );
		if ( res < best ) best = res;
		iterations = k + 1;

		if ( [ 60, 300, 1000, 3000, 10000, 30000 ].includes( iterations ) ) history.push( [ iterations, res / bNorm ] );
		if ( res < target ) break;

		const beta = f( rrNew / rr );
		for ( let n = 0; n < CELLS; n ++ ) p[ n ] = f( r[ n ] + beta * p[ n ] );
		rr = rrNew;

	}

	const final = Math.sqrt( Math.abs( dot( r, r ) ) ) / bNorm;
	console.log( `${ final < 1e-6 ? '✓' : '✗' } ${ label } -- ${ iterations } iterations, res/|b| ${ final.toExponential( 2 ) }, best ${ ( best / bNorm ).toExponential( 2 ) }` );
	for ( const [ k, rel ] of history ) console.log( `       at ${ String( k ).padStart( 5 ) }: ${ rel.toExponential( 2 ) }` );

	return { iterations, final };

}

// The spread of b matters to a float32 reduction and nothing else here, so it
// is reported: a wide dynamic range is what makes a single-precision sum lose
// digits.
{
	let mn = Infinity, mx = 0, nz = 0;
	for ( const v of b ) { const a = Math.abs( v ); if ( a > 0 ) { nz ++; if ( a < mn ) mn = a; if ( a > mx ) mx = a; } }
	console.log( `b: ${ nz } nonzero of ${ CELLS }, magnitudes ${ mn.toExponential( 2 ) } to ${ mx.toExponential( 2 ) } -- a range of ${ ( mx / mn ).toExponential( 1 ) }
` );
}

console.log( 'plain CG, no preconditioner, from x = 0, target 1e-6 relative:' );
const dbl = cg( false, MAX_ITERATIONS, 'double arithmetic, double reductions', dot );
const flt = cg( true, MAX_ITERATIONS, 'float32 arithmetic, double reductions', dot );
const lanes = cg( true, MAX_ITERATIONS, `float32 arithmetic, float32 reductions in ${ LANES } lanes -- the GPU's shape`, dotF32Lanes );
const naive = cg( true, MAX_ITERATIONS, 'float32 arithmetic, float32 serial reduction -- the worst case', dotF32Naive );

console.log( '\nfrom the warm start the live solver actually uses, the previous frame pressure:' );
const warm = cg( false, MAX_ITERATIONS, 'double, warm start', dot, pressureBefore );
const warmLanes = cg( true, MAX_ITERATIONS, `float32, float32 lane reductions, warm start, incremental r`, dotF32Lanes, pressureBefore );

console.log( `
and with the residual recomputed from scratch every iteration, which is what
the live solver does and the one thing none of the arms above modelled:` );
// *** Sweeping the replacement interval ***
//
// Computing r = b - Ax in f32 is accurate to 3.57e-8 relative on this system
// (where_error.mjs measures it directly), which is 160x below the 5.84e-6 floor.
// So the floor is not the accuracy of the recomputation. What is left is the
// REPLACEMENT itself: CG's recurrences assume r is the one its own iteration
// produced, and substituting the true residual for it each step breaks the
// conjugacy those recurrences rely on. If that is the mechanism, the floor
// should depend on how often the substitution happens.
for ( const every of [ 1, 2, 5, 10, 50, 0 ] ) {

	const label = every === 0 ? 'never (incremental only)' : `every ${ every }`;
	const arm = cg( true, MAX_ITERATIONS, `float32, warm start, recompute ${ label }`, dotF32Lanes, pressureBefore, every );
	void arm;

}

const recompDouble = cg( false, MAX_ITERATIONS, 'double, warm start, r = b - Ax every iteration', dot, pressureBefore, 1 );
const recompF32 = cg( true, MAX_ITERATIONS, 'float32 + float32 lane reductions, warm start, r = b - Ax every iteration -- the live solver exactly', dotF32Lanes, pressureBefore, 1 );

console.log( '' );
if ( dbl.final >= 1e-6 ) {

	console.log( `Even double-precision CG floors at ${ dbl.final.toExponential( 2 ) } after ${ dbl.iterations } iterations, so the floor is in the system and not in the GPU.` );

} else {

	console.log( `The system reaches 1e-6 in ${ dbl.iterations } unpreconditioned iterations in double, so nothing about b prevents it.` );
	console.log( `float32 arithmetic with double reductions: ${ flt.final < 1e-6 ? 'reaches it in ' + flt.iterations : 'floors at ' + flt.final.toExponential( 2 ) }.` );
	console.log( `float32 arithmetic AND float32 reductions, the GPU's shape: ${ lanes.final < 1e-6 ? 'reaches it in ' + lanes.iterations : 'floors at ' + lanes.final.toExponential( 2 ) }.` );
	console.log( `the same with a serial float32 sum, worst case: ${ naive.final < 1e-6 ? 'reaches it in ' + naive.iterations : 'floors at ' + naive.final.toExponential( 2 ) }.` );
	console.log( `` );
	console.log( `warm-started from the previous frame's pressure, as the live solver does:` );
	console.log( `  double: ${ warm.final < 1e-6 ? 'reaches 1e-6 in ' + warm.iterations : 'floors at ' + warm.final.toExponential( 2 ) }` );
	console.log( `  float32 with float32 lane reductions: ${ warmLanes.final < 1e-6 ? 'reaches 1e-6 in ' + warmLanes.iterations : 'floors at ' + warmLanes.final.toExponential( 2 ) }` );
	console.log( `` );
	console.log( `` );
	console.log( `recomputing r = b - Ax every iteration, as the live solver does:` );
	console.log( `  double:  ${ recompDouble.final < 1e-6 ? 'reaches 1e-6 in ' + recompDouble.iterations : 'floors at ' + recompDouble.final.toExponential( 2 ) + ' after ' + recompDouble.iterations }` );
	console.log( `  float32: ${ recompF32.final < 1e-6 ? 'reaches 1e-6 in ' + recompF32.iterations : 'floors at ' + recompF32.final.toExponential( 2 ) + ' after ' + recompF32.iterations }` );
	console.log( `` );
	console.log( `the GPU's own unpreconditioned arm reached 6.96e-6 at 3000 iterations without converging.` );
	console.log( recompF32.final >= 1e-6 && recompDouble.final < 1e-6
		? `ROOT CAUSE: the float32 recomputation of r = b - Ax is the floor. Double precision with the same recomputation reaches the tolerance; single precision does not.`
		: `The recomputation is not the difference either -- ${ recompF32.final < 1e-6 ? 'float32 reaches the tolerance with it' : 'and double floors too, so the floor is not precision' }.` );

}
