// Ported from grid_data2.py.
//
// The source expresses "shared construction logic + a few differences" via
// class inheritance (CellCenteredVectorGrid2 extends CollocatedVectorGrid2,
// only doing one extra dataOrigin offset at the end of __init__). Here that
// becomes factory functions instead -- keeping the same style as
// tsl_array_n itself (arrayN/kernel/func are all closures, not classes).
// "Inheritance" is expressed via { ...baseGrid, overriddenFields }: call the
// base factory to get the full object, then override just the fields that
// need to change -- equivalent to a subclass's __init__ reassigning them.
//
// resolution/dataSize are plain JS arrays (pure CPU-side shape metadata),
// while gridSpacing/dataOrigin/origin are real TSL vec2 nodes (built once at
// construction time, then referenced directly by that same node object
// inside any kernel/func later -- a TSL node doesn't need to be constructed
// "inside a kernel", the same as how tsl_array_n's own uniform()/array0()
// are used).
//
// .clear() is always a real, pre-built kernel already bound to a specific
// field (an existing capability boundary of tsl_array_n's kernel(shape,fn) --
// closure-bound at construction time, not rebuilt on every call); at
// construction time the initial value is separately zeroed out via
// fromArray(an all-zero typed array) (a CPU-side write, doesn't need
// tsl_array_n.init() to have already set up a renderer), which corresponds
// to the source's own self.clear() call at the end of __init__ -- this
// avoids depending on a timing assumption about whether init() has already
// run by the time this grid is constructed.

import * as tsl_array_n from 'tsl_array_n';
import { vec2, float } from 'three/tsl';
import { faceCenteredValueAtPosition2 } from './grid_math.js';

function dataPositionFn( dataOrigin, gridSpacing ) {

	return function dataPosition( i, j ) {

		return dataOrigin.add( gridSpacing.mul( vec2( i.toFloat(), j.toFloat() ) ) );

	};

}

function zeroScalarField2( sizeX, sizeY ) {

	const data = tsl_array_n.array2( 'float', sizeX, sizeY );
	data.fromArray( new Float32Array( sizeX * sizeY ) );
	return data;

}

function zeroVectorField2( sizeX, sizeY ) {

	const data = tsl_array_n.array2( 'vec2', sizeX, sizeY );
	data.fromArray( new Float32Array( sizeX * sizeY * 2 ) );
	return data;

}

// dataSize for the vertex-centered variants: +1 on each dimension. The
// source's VertexCenteredScalarGrid2/VertexCenteredVectorGrid dataSize
// property keeps (0,0) when resolution is exactly (0,0) (a defensive
// branch) -- not carried over here: nowhere in the whole grid/ folder ever
// actually constructs either class with (0,0) (grepped for it -- it's a
// defensive branch in a plain property getter that's never triggered), and
// tsl_array_n.array2() itself flatly rejects zero-length dimensions (throws
// immediately), so there's no way to produce a genuine 0x0 field anyway --
// meaning this branch has neither a usage precedent nor underlying platform
// support on this port's target, so it wasn't ported. If an "unconfigured
// placeholder grid" concept is ever actually needed, it should be expressed
// as null/undefined at the JS level, not by constructing a 0x0 field.
function vertexDataSize( resolutionX, resolutionY ) {

	return [ resolutionX + 1, resolutionY + 1 ];

}

// ------------------------------------------------------------
// collocated vector grid

export function createCollocatedVectorGrid2( resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY ) {

	const resolution = [ resolutionX, resolutionY ];
	const gridSpacing = vec2( gridSpacingX, gridSpacingY );
	const dataOrigin = vec2( originX, originY );
	const dataSize = resolution;

	const data = zeroVectorField2( dataSize[ 0 ], dataSize[ 1 ] );
	const clear = tsl_array_n.kernel( dataSize, ( i, j ) => {

		data( i, j ).assign( vec2( 0 ) );

	} );

	return {
		resolution, gridSpacing, dataOrigin, dataSize, data, clear,
		dataPosition: dataPositionFn( dataOrigin, gridSpacing )
	};

}

// ------------------------------------------------------------
// cell centered vector grid

export function createCellCenteredVectorGrid2( resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY ) {

	const grid = createCollocatedVectorGrid2( resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY );
	const dataOrigin = grid.dataOrigin.add( grid.gridSpacing.mul( 0.5 ) );

	return {
		...grid,
		dataOrigin,
		dataPosition: dataPositionFn( dataOrigin, grid.gridSpacing )
	};

}

// ------------------------------------------------------------
// vertex centered vector grid

export function createVertexCenteredVectorGrid2( resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY ) {

	const resolution = [ resolutionX, resolutionY ];
	const gridSpacing = vec2( gridSpacingX, gridSpacingY );
	const dataOrigin = vec2( originX, originY );
	const dataSize = vertexDataSize( resolutionX, resolutionY );

	const data = zeroVectorField2( dataSize[ 0 ], dataSize[ 1 ] );
	const clear = tsl_array_n.kernel( dataSize, ( i, j ) => {

		data( i, j ).assign( vec2( 0 ) );

	} );

	return {
		resolution, gridSpacing, dataOrigin, dataSize, data, clear,
		dataPosition: dataPositionFn( dataOrigin, gridSpacing )
	};

}

// ------------------------------------------------------------
// face centered grid (MAC grid / staggered grid)

export function createFaceCenteredGrid2( resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY ) {

	const resolution = [ resolutionX, resolutionY ];
	const gridSpacing = vec2( gridSpacingX, gridSpacingY );
	const origin = vec2( originX, originY );

	const dataSizeU = [ resolutionX + 1, resolutionY ];
	const dataSizeV = [ resolutionX, resolutionY + 1 ];

	const dataOriginU = origin.add( vec2( 0, gridSpacing.y.mul( 0.5 ) ) );
	const dataOriginV = origin.add( vec2( gridSpacing.x.mul( 0.5 ), 0 ) );

	const dataU = zeroScalarField2( dataSizeU[ 0 ], dataSizeU[ 1 ] );
	const dataV = zeroScalarField2( dataSizeV[ 0 ], dataSizeV[ 1 ] );

	// The source's clear() is one kernel containing two top-level for loops
	// (one for dataU, one for dataV) -- Taichi supports multiple parallel
	// for-loops inside a single kernel, but tsl_array_n's kernel(shape,fn)
	// is one shape per dispatch, and dataU/dataV don't even share a shape,
	// so they can't naturally fit into the same kernel anyway. This becomes
	// two kernels instead, with clear() calling both in sequence -- same
	// effect, just "one kernel, two loop bodies" turning into "two kernels,
	// dispatched one after the other".
	const clearU = tsl_array_n.kernel( dataSizeU, ( i, j ) => { dataU( i, j ).assign( float( 0 ) ); } );
	const clearV = tsl_array_n.kernel( dataSizeV, ( i, j ) => { dataV( i, j ).assign( float( 0 ) ); } );

	function clear() {

		clearU();
		clearV();

	}

	const uPosition = dataPositionFn( dataOriginU, gridSpacing );
	const vPosition = dataPositionFn( dataOriginV, gridSpacing );

	// Bilinearly samples the u and v components and combines them into a
	// vec2 velocity -- a thin wrapper, the actual math lives in grid_math so
	// it isn't duplicated here
	function sample( pos ) {

		return faceCenteredValueAtPosition2(
			dataU, dataV, gridSpacing,
			dataOriginU, dataOriginV, pos,
			dataSizeU, dataSizeV
		);

	}

	return {
		resolution, gridSpacing, origin,
		dataSizeU, dataSizeV, dataOriginU, dataOriginV,
		dataU, dataV, clear, uPosition, vPosition, sample
	};

}

// ------------------------------------------------------------
// scalar grid

export function createScalarGrid2( resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY ) {

	const resolution = [ resolutionX, resolutionY ];
	const gridSpacing = vec2( gridSpacingX, gridSpacingY );
	const dataOrigin = vec2( originX, originY );
	const dataSize = resolution;

	const data = zeroScalarField2( dataSize[ 0 ], dataSize[ 1 ] );
	const clear = tsl_array_n.kernel( dataSize, ( i, j ) => {

		data( i, j ).assign( float( 0 ) );

	} );

	return {
		resolution, gridSpacing, dataOrigin, dataSize, data, clear,
		dataPosition: dataPositionFn( dataOrigin, gridSpacing )
	};

}

// ------------------------------------------------------------
// cell centered scalar grid

export function createCellCenteredScalarGrid2( resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY ) {

	const grid = createScalarGrid2( resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY );
	const dataOrigin = grid.dataOrigin.add( grid.gridSpacing.mul( 0.5 ) );

	return {
		...grid,
		dataOrigin,
		dataPosition: dataPositionFn( dataOrigin, grid.gridSpacing )
	};

}

// ------------------------------------------------------------
// vertex centered scalar grid

export function createVertexCenteredScalarGrid2( resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY ) {

	const resolution = [ resolutionX, resolutionY ];
	const gridSpacing = vec2( gridSpacingX, gridSpacingY );
	const dataOrigin = vec2( originX, originY );
	const dataSize = vertexDataSize( resolutionX, resolutionY );

	const data = zeroScalarField2( dataSize[ 0 ], dataSize[ 1 ] );
	const clear = tsl_array_n.kernel( dataSize, ( i, j ) => {

		data( i, j ).assign( float( 0 ) );

	} );

	return {
		resolution, gridSpacing, dataOrigin, dataSize, data, clear,
		dataPosition: dataPositionFn( dataOrigin, gridSpacing )
	};

}
