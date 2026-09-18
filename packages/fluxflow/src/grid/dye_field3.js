// 3D counterpart of dye_field2.js -- same "fine passive-scalar grid,
// advected by a coarser velocity grid" design, one more axis. See that
// file's own header comment for the full "why this exists, why it's
// passive" reasoning, which is entirely dimension-independent.

import * as tsl_array_n from 'tsl_array_n';
import { float, clamp } from 'three/tsl';
import { createCellCenteredScalarGrid3 } from './grid_data3.js';
import { collocatedValueAtPosition3 } from './grid_math3.js';
import { createCopyKernel3 } from './array_utils3.js';
import { createSemiLagrangianAdvectionSolver3 } from './advection_solver3.js';

// Explicit diffusion stability bound in 3D: D dt / h^2 < 1/6 (one more
// neighbour pair than 2D's 1/4 -- the standard explicit-diffusion CFL
// number is 1/(2*dimensions)).
const MAX_DIFFUSION_NUMBER = 1 / 6;

function numberOrNode( value ) {

	return typeof value === 'number' ? float( value ) : value;

}

// See dye_field2.js's own createDyeField2 header comment for the full
// parameter list's meaning -- resolution/gridSpacing/origin/subdivisions
// are now triples, everything else is unchanged.
export function createDyeField3( {
	velocityGrid,
	resolution,
	gridSpacing = [ 1, 1, 1 ],
	origin = [ 0, 0, 0 ],
	subdivisions = 3,
	dt,
	diffusion = 0,
	liquidMask,
	collider,
	order = 2,
	maxSubsteps
} ) {

	if ( ! Number.isInteger( subdivisions ) || subdivisions < 1 ) {

		throw new Error( `createDyeField3: subdivisions must be a positive integer, got ${ subdivisions }.` );

	}

	const [ coarseX, coarseY, coarseZ ] = resolution;
	const fineX = coarseX * subdivisions;
	const fineY = coarseY * subdivisions;
	const fineZ = coarseZ * subdivisions;
	const fineHx = gridSpacing[ 0 ] / subdivisions;
	const fineHy = gridSpacing[ 1 ] / subdivisions;
	const fineHz = gridSpacing[ 2 ] / subdivisions;
	const fineShape = [ fineX, fineY, fineZ ];
	const fineCount = fineX * fineY * fineZ;

	const field = createCellCenteredScalarGrid3( fineX, fineY, fineZ, fineHx, fineHy, fineHz, origin[ 0 ], origin[ 1 ], origin[ 2 ] );
	const scratch = createCellCenteredScalarGrid3( fineX, fineY, fineZ, fineHx, fineHy, fineHz, origin[ 0 ], origin[ 1 ], origin[ 2 ] );

	const zeros = new Float32Array( fineCount );
	field.data.fromArray( zeros );
	scratch.data.fromArray( zeros );

	const advectionSolver = createSemiLagrangianAdvectionSolver3( { velocityGrid, collider, dt, order, maxSubsteps } );
	const dispatchAdvect = advectionSolver.advectScalar3( field, scratch );
	const copyBack = createCopyKernel3( scratch.data, field.data, fineShape );

	let dispatchMask = null;

	if ( liquidMask ) {

		const maskShape = liquidMask.resolution ?? resolution;
		const maskSpacing = [ gridSpacing[ 0 ], gridSpacing[ 1 ], gridSpacing[ 2 ] ];
		const maskOrigin = [ origin[ 0 ] + maskSpacing[ 0 ] * 0.5, origin[ 1 ] + maskSpacing[ 1 ] * 0.5, origin[ 2 ] + maskSpacing[ 2 ] * 0.5 ];

		dispatchMask = tsl_array_n.kernel( fineShape, ( i, j, k ) => {

			const pos = field.dataPosition( i, j, k );
			const inside = collocatedValueAtPosition3( liquidMask.field, maskSpacing, maskOrigin, pos, maskShape );

			field.data( i, j, k ).mulAssign( clamp( inside, float( 0 ), float( 1 ) ) );

		} );

	}

	let dispatchDiffuse = null;
	const diffusionIsLive = typeof diffusion !== 'number';

	if ( diffusionIsLive || diffusion > 0 ) {

		const dtNode = numberOrNode( dt );
		const diffusionNode = numberOrNode( diffusion );

		const invHx2 = 1 / ( fineHx * fineHx );
		const invHy2 = 1 / ( fineHy * fineHy );
		const invHz2 = 1 / ( fineHz * fineHz );

		dispatchDiffuse = tsl_array_n.kernel( fineShape, ( i, j, k ) => {

			const c = field.data( i, j, k );

			const iLo = i.sub( 1 ).max( 0 );
			const iHi = i.add( 1 ).min( fineX - 1 );
			const jLo = j.sub( 1 ).max( 0 );
			const jHi = j.add( 1 ).min( fineY - 1 );
			const kLo = k.sub( 1 ).max( 0 );
			const kHi = k.add( 1 ).min( fineZ - 1 );

			const lapX = field.data( iLo, j, k ).add( field.data( iHi, j, k ) ).sub( c.mul( 2 ) ).mul( invHx2 );
			const lapY = field.data( i, jLo, k ).add( field.data( i, jHi, k ) ).sub( c.mul( 2 ) ).mul( invHy2 );
			const lapZ = field.data( i, j, kLo ).add( field.data( i, j, kHi ) ).sub( c.mul( 2 ) ).mul( invHz2 );

			const step = clamp( diffusionNode.mul( dtNode ), float( 0 ), float( MAX_DIFFUSION_NUMBER / Math.max( invHx2, invHy2, invHz2 ) ) );

			scratch.data( i, j, k ).assign( c.add( lapX.add( lapY ).add( lapZ ).mul( step ) ) );

		} );

	}

	function step() {

		dispatchAdvect();
		copyBack();

		if ( dispatchDiffuse ) {

			dispatchDiffuse();
			copyBack();

		}

		if ( dispatchMask ) dispatchMask();

	}

	function clear() {

		field.data.fromArray( zeros );

	}

	return {
		field,
		resolution: fineShape,
		gridSpacing: [ fineHx, fineHy, fineHz ],
		subdivisions,
		step,
		clear,
		advect: () => {

			dispatchAdvect();
			copyBack();

		}
	};

}

// See dye_field2.js's own computeDyeSeed header comment.
export function computeDyeSeed3( { resolution, gridSpacing, origin = [ 0, 0, 0 ], valueAt } ) {

	const [ nx, ny, nz ] = resolution;
	const out = new Float32Array( nx * ny * nz );

	for ( let k = 0; k < nz; k ++ ) {

		const z = origin[ 2 ] + ( k + 0.5 ) * gridSpacing[ 2 ];

		for ( let j = 0; j < ny; j ++ ) {

			const y = origin[ 1 ] + ( j + 0.5 ) * gridSpacing[ 1 ];

			for ( let i = 0; i < nx; i ++ ) {

				const x = origin[ 0 ] + ( i + 0.5 ) * gridSpacing[ 0 ];
				out[ i + nx * j + nx * ny * k ] = valueAt( x, y, z );

			}

		}

	}

	return out;

}
