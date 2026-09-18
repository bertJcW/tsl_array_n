// 3D counterpart of grid_data2.js -- same factory-function shape (see that
// file's own header comment for why factories rather than classes, and for
// the "inheritance via spread + override" convention), one axis wider.
// grid_data2.js is untouched.

import * as tsl_array_n from 'tsl_array_n';
import { vec3, float } from 'three/tsl';
import { faceCenteredValueAtPosition3 } from './grid_math3.js';

function dataPositionFn( dataOrigin, gridSpacing ) {

	return function dataPosition( i, j, k ) {

		return dataOrigin.add( gridSpacing.mul( vec3( i.toFloat(), j.toFloat(), k.toFloat() ) ) );

	};

}

function zeroScalarField3( sizeX, sizeY, sizeZ ) {

	const data = tsl_array_n.array3( 'float', sizeX, sizeY, sizeZ );
	data.fromArray( new Float32Array( sizeX * sizeY * sizeZ ) );
	return data;

}

function zeroVectorField3( sizeX, sizeY, sizeZ ) {

	const data = tsl_array_n.array3( 'vec3', sizeX, sizeY, sizeZ );
	data.fromArray( new Float32Array( sizeX * sizeY * sizeZ * 3 ) );
	return data;

}

// dataSize for the vertex-centered variants: +1 on each dimension. See
// grid_data2.js's own vertexDataSize header comment for why the source's
// "(0,0) stays (0,0)" defensive branch isn't ported here either -- same
// reasoning, no 0-cell grid is ever actually constructed and array3()
// rejects one outright.
function vertexDataSize( resolutionX, resolutionY, resolutionZ ) {

	return [ resolutionX + 1, resolutionY + 1, resolutionZ + 1 ];

}

// ------------------------------------------------------------
// collocated vector grid

export function createCollocatedVectorGrid3( resolutionX, resolutionY, resolutionZ, gridSpacingX, gridSpacingY, gridSpacingZ, originX, originY, originZ ) {

	const resolution = [ resolutionX, resolutionY, resolutionZ ];
	const gridSpacing = vec3( gridSpacingX, gridSpacingY, gridSpacingZ );
	const dataOrigin = vec3( originX, originY, originZ );
	const dataSize = resolution;

	const data = zeroVectorField3( dataSize[ 0 ], dataSize[ 1 ], dataSize[ 2 ] );
	const clear = tsl_array_n.kernel( dataSize, ( i, j, k ) => {

		data( i, j, k ).assign( vec3( 0 ) );

	} );

	return {
		resolution, gridSpacing, dataOrigin, dataSize, data, clear,
		dataPosition: dataPositionFn( dataOrigin, gridSpacing )
	};

}

// ------------------------------------------------------------
// cell centered vector grid

export function createCellCenteredVectorGrid3( resolutionX, resolutionY, resolutionZ, gridSpacingX, gridSpacingY, gridSpacingZ, originX, originY, originZ ) {

	const grid = createCollocatedVectorGrid3( resolutionX, resolutionY, resolutionZ, gridSpacingX, gridSpacingY, gridSpacingZ, originX, originY, originZ );
	const dataOrigin = grid.dataOrigin.add( grid.gridSpacing.mul( 0.5 ) );

	return {
		...grid,
		dataOrigin,
		dataPosition: dataPositionFn( dataOrigin, grid.gridSpacing )
	};

}

// ------------------------------------------------------------
// vertex centered vector grid

export function createVertexCenteredVectorGrid3( resolutionX, resolutionY, resolutionZ, gridSpacingX, gridSpacingY, gridSpacingZ, originX, originY, originZ ) {

	const resolution = [ resolutionX, resolutionY, resolutionZ ];
	const gridSpacing = vec3( gridSpacingX, gridSpacingY, gridSpacingZ );
	const dataOrigin = vec3( originX, originY, originZ );
	const dataSize = vertexDataSize( resolutionX, resolutionY, resolutionZ );

	const data = zeroVectorField3( dataSize[ 0 ], dataSize[ 1 ], dataSize[ 2 ] );
	const clear = tsl_array_n.kernel( dataSize, ( i, j, k ) => {

		data( i, j, k ).assign( vec3( 0 ) );

	} );

	return {
		resolution, gridSpacing, dataOrigin, dataSize, data, clear,
		dataPosition: dataPositionFn( dataOrigin, gridSpacing )
	};

}

// ------------------------------------------------------------
// face centered grid (MAC grid / staggered grid)
//
// Three staggered components instead of two: dataU lives on the grid's
// x-faces (integer x, cell-centred in y AND z), dataV on y-faces
// (cell-centred in x and z), dataW on z-faces (cell-centred in x and y) --
// each origin offsets by half a cell along the two axes it is NOT
// staggered on, same convention grid_data2.js's dataOriginU/V follow one
// axis narrower.

export function createFaceCenteredGrid3( resolutionX, resolutionY, resolutionZ, gridSpacingX, gridSpacingY, gridSpacingZ, originX, originY, originZ ) {

	const resolution = [ resolutionX, resolutionY, resolutionZ ];
	const gridSpacing = vec3( gridSpacingX, gridSpacingY, gridSpacingZ );
	const origin = vec3( originX, originY, originZ );

	const dataSizeU = [ resolutionX + 1, resolutionY, resolutionZ ];
	const dataSizeV = [ resolutionX, resolutionY + 1, resolutionZ ];
	const dataSizeW = [ resolutionX, resolutionY, resolutionZ + 1 ];

	const dataOriginU = origin.add( vec3( 0, gridSpacing.y.mul( 0.5 ), gridSpacing.z.mul( 0.5 ) ) );
	const dataOriginV = origin.add( vec3( gridSpacing.x.mul( 0.5 ), 0, gridSpacing.z.mul( 0.5 ) ) );
	const dataOriginW = origin.add( vec3( gridSpacing.x.mul( 0.5 ), gridSpacing.y.mul( 0.5 ), 0 ) );

	const dataU = zeroScalarField3( dataSizeU[ 0 ], dataSizeU[ 1 ], dataSizeU[ 2 ] );
	const dataV = zeroScalarField3( dataSizeV[ 0 ], dataSizeV[ 1 ], dataSizeV[ 2 ] );
	const dataW = zeroScalarField3( dataSizeW[ 0 ], dataSizeW[ 1 ], dataSizeW[ 2 ] );

	// Three kernels, one per component -- see grid_data2.js's own comment
	// on why this doesn't collapse into one (dataU/dataV/dataW don't even
	// share a shape).
	const clearU = tsl_array_n.kernel( dataSizeU, ( i, j, k ) => { dataU( i, j, k ).assign( float( 0 ) ); } );
	const clearV = tsl_array_n.kernel( dataSizeV, ( i, j, k ) => { dataV( i, j, k ).assign( float( 0 ) ); } );
	const clearW = tsl_array_n.kernel( dataSizeW, ( i, j, k ) => { dataW( i, j, k ).assign( float( 0 ) ); } );

	function clear() {

		clearU();
		clearV();
		clearW();

	}

	const uPosition = dataPositionFn( dataOriginU, gridSpacing );
	const vPosition = dataPositionFn( dataOriginV, gridSpacing );
	const wPosition = dataPositionFn( dataOriginW, gridSpacing );

	// Trilinearly samples the u/v/w components and combines them into a
	// vec3 velocity -- thin wrapper, the actual math lives in grid_math3.
	function sample( pos ) {

		return faceCenteredValueAtPosition3(
			dataU, dataV, dataW, gridSpacing,
			dataOriginU, dataOriginV, dataOriginW, pos,
			dataSizeU, dataSizeV, dataSizeW
		);

	}

	return {
		resolution, gridSpacing, origin,
		dataSizeU, dataSizeV, dataSizeW, dataOriginU, dataOriginV, dataOriginW,
		dataU, dataV, dataW, clear, uPosition, vPosition, wPosition, sample
	};

}

// ------------------------------------------------------------
// scalar grid

export function createScalarGrid3( resolutionX, resolutionY, resolutionZ, gridSpacingX, gridSpacingY, gridSpacingZ, originX, originY, originZ ) {

	const resolution = [ resolutionX, resolutionY, resolutionZ ];
	const gridSpacing = vec3( gridSpacingX, gridSpacingY, gridSpacingZ );
	const dataOrigin = vec3( originX, originY, originZ );
	const dataSize = resolution;

	const data = zeroScalarField3( dataSize[ 0 ], dataSize[ 1 ], dataSize[ 2 ] );
	const clear = tsl_array_n.kernel( dataSize, ( i, j, k ) => {

		data( i, j, k ).assign( float( 0 ) );

	} );

	return {
		resolution, gridSpacing, dataOrigin, dataSize, data, clear,
		dataPosition: dataPositionFn( dataOrigin, gridSpacing )
	};

}

// ------------------------------------------------------------
// cell centered scalar grid

export function createCellCenteredScalarGrid3( resolutionX, resolutionY, resolutionZ, gridSpacingX, gridSpacingY, gridSpacingZ, originX, originY, originZ ) {

	const grid = createScalarGrid3( resolutionX, resolutionY, resolutionZ, gridSpacingX, gridSpacingY, gridSpacingZ, originX, originY, originZ );
	const dataOrigin = grid.dataOrigin.add( grid.gridSpacing.mul( 0.5 ) );

	return {
		...grid,
		dataOrigin,
		dataPosition: dataPositionFn( dataOrigin, grid.gridSpacing )
	};

}

// ------------------------------------------------------------
// vertex centered scalar grid

export function createVertexCenteredScalarGrid3( resolutionX, resolutionY, resolutionZ, gridSpacingX, gridSpacingY, gridSpacingZ, originX, originY, originZ ) {

	const resolution = [ resolutionX, resolutionY, resolutionZ ];
	const gridSpacing = vec3( gridSpacingX, gridSpacingY, gridSpacingZ );
	const dataOrigin = vec3( originX, originY, originZ );
	const dataSize = vertexDataSize( resolutionX, resolutionY, resolutionZ );

	const data = zeroScalarField3( dataSize[ 0 ], dataSize[ 1 ], dataSize[ 2 ] );
	const clear = tsl_array_n.kernel( dataSize, ( i, j, k ) => {

		data( i, j, k ).assign( float( 0 ) );

	} );

	return {
		resolution, gridSpacing, dataOrigin, dataSize, data, clear,
		dataPosition: dataPositionFn( dataOrigin, gridSpacing )
	};

}
