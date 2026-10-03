// Where exactly does the f32 error in `r = b - A@x` come from?
//
// Three candidates have been claimed at different times in this project, twice
// by me:
//
//   (a) cancellation in the subtraction `b - Ax`
//   (b) rounding in the accumulation that forms `Ax`
//   (c) neither, and the floor is something else
//
// (a) was retracted on 2026-09-15 on Sterbenz's lemma: the difference of two
// nearby f32 values is exact. (b) was asserted the same day with a per-cell
// magnitude of 550 -- which was the field's NORM used as if it were a cell value,
// when cells are 7 to 45 -- and the whole mechanism was retracted along with the
// bad arithmetic. Then on 2026-10-03 I re-adopted (b) while citing those same
// retracted numbers.
//
// So: measure it. The exported system gives x, b and the operator, and the error
// in any one step is the difference between computing it in f32 and in double.
// Nothing here is inferred.

import { readFileSync } from 'node:fs';

const DIR = process.argv[ 2 ] ?? 'system-2d';
const meta = JSON.parse( readFileSync( `${ DIR }/meta.json`, 'utf8' ) );
const field = ( name ) => new Float32Array( readFileSync( `${ DIR }/${ name }.bin` ).buffer.slice( 0 ) );

const [ NX, NY ] = meta.shape;
const CELLS = NX * NY;
const [ HX, HY ] = meta.gridSpacing;

const b = Float64Array.from( field( 'b' ) );
const maskRaw = field( 'dirichletMask' );
const pinned = new Uint8Array( CELLS );
for ( let n = 0; n < CELLS; n ++ ) pinned[ n ] = maskRaw[ n ] > 0.5 ? 1 : 0;
const x = Float64Array.from( field( 'pressureAfter' ) );

const at = ( i, j ) => i + NX * j;
const dot = ( a, c ) => { let s = 0; for ( let i = 0; i < a.length; i ++ ) s += a[ i ] * c[ i ]; return s; };
const norm = ( a ) => Math.sqrt( dot( a, a ) );
const maxAbs = ( a ) => a.reduce( ( m, v ) => Math.max( m, Math.abs( v ) ), 0 );

// `mode` picks the arithmetic used INSIDE the stencil accumulation.
//   'double'  -- everything in float64
//   'f32'     -- every operation rounded to f32, as the GPU does
//   'f32dsAcc'-- terms rounded to f32 but the running sum kept in double, which
//                is what a double-single accumulator buys, idealised
function applyA( out, mode ) {

	const r = mode === 'double' ? ( v ) => v : Math.fround;
	const accExact = mode === 'f32dsAcc' || mode === 'double';
	const ihx2 = r( 1 / ( HX * HX ) );
	const ihy2 = r( 1 / ( HY * HY ) );

	for ( let j = 0; j < NY; j ++ ) {

		for ( let i = 0; i < NX; i ++ ) {

			const c = at( i, j );
			const center = x[ c ];

			if ( pinned[ c ] ) { out[ c ] = r( - center ); continue; }

			const terms = [];

			let dLo = 0, dHi = 0;
			if ( i > 0 ) dLo = pinned[ at( i - 1, j ) ] ? center : r( center - x[ at( i - 1, j ) ] );
			if ( i < NX - 1 ) dHi = pinned[ at( i + 1, j ) ] ? - center : r( x[ at( i + 1, j ) ] - center );
			terms.push( r( r( dHi - dLo ) * ihx2 ) );

			dLo = 0; dHi = 0;
			if ( j > 0 ) dLo = pinned[ at( i, j - 1 ) ] ? center : r( center - x[ at( i, j - 1 ) ] );
			if ( j < NY - 1 ) dHi = pinned[ at( i, j + 1 ) ] ? - center : r( x[ at( i, j + 1 ) ] - center );
			terms.push( r( r( dHi - dLo ) * ihy2 ) );

			let sum = 0;
			for ( const t of terms ) sum = accExact ? sum + t : r( sum + t );
			out[ c ] = sum;

		}

	}

}

const AxDouble = new Float64Array( CELLS );
const AxF32 = new Float64Array( CELLS );
const AxDsAcc = new Float64Array( CELLS );
applyA( AxDouble, 'double' );
applyA( AxF32, 'f32' );
applyA( AxDsAcc, 'f32dsAcc' );

const bNorm = norm( b );
const diff = ( p, q ) => { const d = new Float64Array( CELLS ); for ( let n = 0; n < CELLS; n ++ ) d[ n ] = p[ n ] - q[ n ]; return d; };

console.log( `${ meta.url } frame ${ meta.frame }` );
console.log( `${ NX }x${ NY }, |b|_2 = ${ bNorm.toExponential( 3 ) }, max|b| = ${ maxAbs( b ).toExponential( 3 ) }` );
console.log( `|x|_2 = ${ norm( x ).toExponential( 3 ) }, max|x| = ${ maxAbs( x ).toExponential( 3 ) }` );
console.log( `|A@x|_2 = ${ norm( AxDouble ).toExponential( 3 ) }, max|A@x| = ${ maxAbs( AxDouble ).toExponential( 3 ) }\n` );

console.log( 'the measured floor this is trying to explain: 5.84e-6 relative, on the CPU with f32 and per-iteration recomputation\n' );

// ---- candidate (b): the error in forming Ax at all ----
{
	const e = diff( AxF32, AxDouble );
	console.log( `error from computing A@x in f32 at all:` );
	console.log( `  |error|_2 = ${ norm( e ).toExponential( 3 ) }  ->  ${ ( norm( e ) / bNorm ).toExponential( 2 ) } relative to |b|` );
	console.log( `  max|error| = ${ maxAbs( e ).toExponential( 3 ) }` );
}

// ---- how much of that is the ACCUMULATION, as opposed to the terms ----
{
	const e = diff( AxF32, AxDsAcc );
	console.log( `\nof which the running sum alone (f32 terms, exact accumulator):` );
	console.log( `  |error|_2 = ${ norm( e ).toExponential( 3 ) }  ->  ${ ( norm( e ) / bNorm ).toExponential( 2 ) } relative` );
	const e2 = diff( AxDsAcc, AxDouble );
	console.log( `and the terms themselves (exact accumulator, f32 differences):` );
	console.log( `  |error|_2 = ${ norm( e2 ).toExponential( 3 ) }  ->  ${ ( norm( e2 ) / bNorm ).toExponential( 2 ) } relative` );
}

// ---- candidate (a): the subtraction b - Ax ----
{
	const rDouble = diff( b, AxDouble );
	const rSubF32 = new Float64Array( CELLS );
	for ( let n = 0; n < CELLS; n ++ ) rSubF32[ n ] = Math.fround( Math.fround( b[ n ] ) - Math.fround( AxDouble[ n ] ) );
	const e = diff( rSubF32, rDouble );
	console.log( `\nerror from the subtraction b - Ax alone, with an EXACT A@x:` );
	console.log( `  |error|_2 = ${ norm( e ).toExponential( 3 ) }  ->  ${ ( norm( e ) / bNorm ).toExponential( 2 ) } relative` );
	console.log( `  (Sterbenz says this should be ~0 where b and Ax are close; |r|/|b| here is ${ ( norm( rDouble ) / bNorm ).toExponential( 2 ) })` );
}

// ---- the prediction the floor formula makes ----
{
	const predicted = ( 2 ** -24 ) * 4 * norm( x ) / bNorm;
	console.log( `\nfor comparison, eps * |A| * |x| / |b| with |A| = 4: ${ predicted.toExponential( 2 ) }` );
}
