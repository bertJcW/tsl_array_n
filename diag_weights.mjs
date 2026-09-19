// Temporary: the distribution of a collider's own face weights, computed on
// the CPU with the same helpers sdf_collider3.js uses. The question this
// answers is whether a rod's sharp edges leave many faces at the 0.01
// floor, and whether any cell ends up with every face nearly closed -- a
// row a hundred times weaker than an interior one, which is what a
// nearly-singular row means here.
import { primitivesDistanceCpu, box, sphere } from './packages/fluxflow/src/grid/primitive_sdf3.js';
import { fractionInsideCpu } from './packages/fluxflow/src/grid/level_set_utils.js';

const NX = 48, NY = 24, NZ = 24;
const MIN_FACE_WEIGHT = 0.01;
const cx = 10, cy = NY / 2 - 3, cz = NZ / 2, halfWidth = 3;

function weightsFor( shapes, label ) {

	const phi = ( x, y, z ) => primitivesDistanceCpu( shapes, x, y, z );

	function faceWeights( dataSize, origin, corners ) {

		const [ nx, ny, nz ] = dataSize;
		const [ ox, oy, oz ] = origin;
		const out = new Float32Array( nx * ny * nz );

		for ( let k = 0; k < nz; k ++ ) for ( let j = 0; j < ny; j ++ ) for ( let i = 0; i < nx; i ++ ) {

			const x = ox + i, y = oy + j, z = oz + k;
			const [ bl, br, tl, tr ] = corners.map( ( [ dx, dy, dz ] ) => phi( x + dx, y + dy, z + dz ) );
			let w = Math.min( 1, Math.max( 0, 1 - fractionInsideCpu( bl, br, tl, tr ) ) );
			if ( w > 0 && w < MIN_FACE_WEIGHT ) w = MIN_FACE_WEIGHT;
			out[ i + nx * j + nx * ny * k ] = w;

		}

		return out;

	}

	const h = 0.5;
	const u = faceWeights( [ NX + 1, NY, NZ ], [ 0, 0.5, 0.5 ], [ [ 0, -h, -h ], [ 0, h, -h ], [ 0, -h, h ], [ 0, h, h ] ] );
	const v = faceWeights( [ NX, NY + 1, NZ ], [ 0.5, 0, 0.5 ], [ [ -h, 0, -h ], [ h, 0, -h ], [ -h, 0, h ], [ h, 0, h ] ] );
	const w = faceWeights( [ NX, NY, NZ + 1 ], [ 0.5, 0.5, 0 ], [ [ -h, -h, 0 ], [ h, -h, 0 ], [ -h, h, 0 ], [ h, h, 0 ] ] );

	const all = [ ...u, ...v, ...w ];
	const zero = all.filter( ( x ) => x === 0 ).length;
	const floored = all.filter( ( x ) => x > 0 && x <= MIN_FACE_WEIGHT + 1e-9 ).length;
	const partial = all.filter( ( x ) => x > MIN_FACE_WEIGHT + 1e-9 && x < 1 - 1e-9 ).length;
	const open = all.filter( ( x ) => x >= 1 - 1e-9 ).length;

	// per-cell total coupling, which is what a row's diagonal is made of
	const U = ( i, j, k ) => u[ i + ( NX + 1 ) * j + ( NX + 1 ) * NY * k ];
	const V = ( i, j, k ) => v[ i + NX * j + NX * ( NY + 1 ) * k ];
	const W = ( i, j, k ) => w[ i + NX * j + NX * NY * k ];

	const totals = [];
	for ( let k = 0; k < NZ; k ++ ) for ( let j = 0; j < NY; j ++ ) for ( let i = 0; i < NX; i ++ )
		totals.push( U( i, j, k ) + U( i + 1, j, k ) + V( i, j, k ) + V( i, j + 1, k ) + W( i, j, k ) + W( i, j, k + 1 ) );

	const enclosed = totals.filter( ( t ) => t < 1e-6 ).length;
	const weak = totals.filter( ( t ) => t >= 1e-6 && t < 0.5 ).length;
	const mid = totals.filter( ( t ) => t >= 0.5 && t < 6 - 1e-6 ).length;
	const full = totals.filter( ( t ) => t >= 6 - 1e-6 ).length;
	const nonzero = totals.filter( ( t ) => t >= 1e-6 );
	const minNonzero = Math.min( ...nonzero );

	console.log( `${ label }:` );
	console.log( `  faces      ${ all.length }: ${ zero } closed, ${ floored } AT THE 0.01 FLOOR, ${ partial } partial, ${ open } open` );
	console.log( `  cell rows  ${ totals.length }: ${ enclosed } enclosed (pinned), ${ weak } with total coupling < 0.5, ${ mid } partial, ${ full } interior` );
	console.log( `  weakest non-enclosed row: ${ minNonzero.toFixed( 4 ) } against an interior row's 6 -- a factor of ${ ( 6 / minNonzero ).toFixed( 0 ) }` );

}

weightsFor( [ box( [ cx, cy, cz ], [ halfWidth, halfWidth, NZ ] ) ], 'rod (a box across the full Z extent)' );
weightsFor( [ sphere( [ cx, cy, cz ], halfWidth ) ], 'sphere of the same half-width' );
