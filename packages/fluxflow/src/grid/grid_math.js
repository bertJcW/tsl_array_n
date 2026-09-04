// Ported from grid_math.py. These are all plain JS functions (building/
// composing TSL node graphs), not tsl_array_n.func() -- these functions
// often need to return several values (e.g. bilinearCoordsAndWeights2
// returns 8), and the parameters are all named, while func()'s (the bare Fn
// re-export) calling convention is a "single destructured array parameter" --
// forcing that here would be awkward and easy to get wrong (see tsl_array_n
// README's known-limitations note). These functions are only ever called by
// other functions/kernels during JS graph construction, so they don't need
// to be "device-side callable subroutines" -- plain JS functions that
// directly return/compose nodes are enough.
//
// Naming-wise, the source's mixed snake_case+camelCase has been unified into
// plain camelCase (e.g. collocated_valueAtPosition2 -> collocatedValueAtPosition2),
// and the source's "colloacted" typo has been fixed to "collocated" along the
// way -- pure naming cleanup, the numeric logic corresponds line-for-line to
// the source.

import { int, float, vec2, mat2, min, max, floor } from 'three/tsl';

// ------------------------------------------------------------
// interpolation

export function faceCenteredValueAtCellCenter2( dataU, dataV, i, j ) {

	return vec2(
		dataU( i, j ).add( dataU( i.add( 1 ), j ) ),
		dataV( i, j ).add( dataV( i, j.add( 1 ) ) )
	).mul( 0.5 );

}

// Given a continuous position, finds the 4 surrounding grid indices (clamped
// to the bounds, so positions outside the grid still sample the boundary
// value) and their bilinear weights, in the order (i0,j0) (i1,j0) (i0,j1) (i1,j1)
export function bilinearCoordsAndWeights2( pos, dataOrigin, gridSpacing, shape ) {

	const [ nx, ny ] = shape;

	const gridPos = pos.sub( dataOrigin ).div( gridSpacing );
	const i0 = int( floor( gridPos.x ) );
	const j0 = int( floor( gridPos.y ) );
	const fx = gridPos.x.sub( i0.toFloat() );
	const fy = gridPos.y.sub( j0.toFloat() );

	const i0c = max( 0, min( i0, nx - 1 ) );
	const i1c = max( 0, min( i0.add( 1 ), nx - 1 ) );
	const j0c = max( 0, min( j0, ny - 1 ) );
	const j1c = max( 0, min( j0.add( 1 ), ny - 1 ) );

	const gx = float( 1 ).sub( fx );
	const gy = float( 1 ).sub( fy );

	const w00 = gx.mul( gy );
	const w10 = fx.mul( gy );
	const w01 = gx.mul( fy );
	const w11 = fx.mul( fy );

	return { i0c, j0c, i1c, j1c, w00, w10, w01, w11 };

}

// Bilinear sample of a collocated field (scalar or vector, either works) at
// a continuous position
export function collocatedValueAtPosition2( data, gridSpacing, dataOrigin, pos, shape ) {

	const { i0c, j0c, i1c, j1c, w00, w10, w01, w11 } = bilinearCoordsAndWeights2( pos, dataOrigin, gridSpacing, shape );

	return data( i0c, j0c ).mul( w00 )
		.add( data( i1c, j0c ).mul( w10 ) )
		.add( data( i0c, j1c ).mul( w01 ) )
		.add( data( i1c, j1c ).mul( w11 ) );

}

// Bilinear sample of a face-centered (MAC) velocity field at a continuous
// position. dataOriginU/dataOriginV are the staggered origins of the u-face
// and v-face respectively
export function faceCenteredValueAtPosition2( dataU, dataV, gridSpacing, dataOriginU, dataOriginV, pos, shapeU, shapeV ) {

	const u = collocatedValueAtPosition2( dataU, gridSpacing, dataOriginU, pos, shapeU );
	const v = collocatedValueAtPosition2( dataV, gridSpacing, dataOriginV, pos, shapeV );

	return vec2( u, v );

}

// ------------------------------------------------------------
// gradient

// shape is the field's shape; gradient at a grid point
export function scalarGradient2( data, gridSpacing, i, j, shape ) {

	const [ nx, ny ] = shape;
	const center = data( i, j );

	const left  = i.greaterThan( 0 ).select( data( i.sub( 1 ), j ), center );
	const right = i.add( 1 ).lessThan( nx ).select( data( i.add( 1 ), j ), center );
	const down  = j.greaterThan( 0 ).select( data( i, j.sub( 1 ) ), center );
	const up    = j.add( 1 ).lessThan( ny ).select( data( i, j.add( 1 ) ), center );

	return vec2( right.sub( left ), up.sub( down ) ).mul( 0.5 ).div( gridSpacing );

}

// shape is the field's shape; gradient (Jacobian) at a grid point
export function vectorGradient2( data, gridSpacing, i, j, shape ) {

	const [ nx, ny ] = shape;
	const center = data( i, j );

	const left  = i.greaterThan( 0 ).select( data( i.sub( 1 ), j ), center );
	const right = i.add( 1 ).lessThan( nx ).select( data( i.add( 1 ), j ), center );
	const down  = j.greaterThan( 0 ).select( data( i, j.sub( 1 ) ), center );
	const up    = j.add( 1 ).lessThan( ny ).select( data( i, j.add( 1 ) ), center );

	const gradX = vec2( right.x.sub( left.x ), up.x.sub( down.x ) ).mul( 0.5 ).div( gridSpacing );
	const gradY = vec2( right.y.sub( left.y ), up.y.sub( down.y ) ).mul( 0.5 ).div( gridSpacing );

	// CONFIRMED on real WebGPU (examples/00-grid-math/, asymmetric test field
	// (2x+5y, 0)): TSL's mat2(a,b,c,d) fills column-major (column0=(a,b),
	// column1=(c,d)), unlike Taichi's tm.mat2(a,b,c,d), which is row-major
	// (row0=(a,b), row1=(c,d)) -- a direct argument-order translation of the
	// source therefore produced a transposed Jacobian (verified: got
	// J.(1,0)=(2,5) and J.(0,1)=(0,0) instead of the correct (2,0) and
	// (5,0)). Fixed by swapping the middle two arguments, so each *column*
	// of the constructed mat2 holds one direction's partial derivatives of
	// both components -- column0 = (dfx/dx, dfy/dx), column1 = (dfx/dy, dfy/dy) --
	// which is what makes J.direction produce the correct directional derivative.
	return mat2(
		gradX.x, gradY.x,
		gradX.y, gradY.y
	);

}

// Gradient of a scalar field at a continuous position: bilinearly blends the
// discrete gradients at the 4 surrounding grid points
export function scalarGradientAtPosition2( data, gridSpacing, dataOrigin, pos, shape ) {

	const { i0c, j0c, i1c, j1c, w00, w10, w01, w11 } = bilinearCoordsAndWeights2( pos, dataOrigin, gridSpacing, shape );

	return scalarGradient2( data, gridSpacing, i0c, j0c, shape ).mul( w00 )
		.add( scalarGradient2( data, gridSpacing, i1c, j0c, shape ).mul( w10 ) )
		.add( scalarGradient2( data, gridSpacing, i0c, j1c, shape ).mul( w01 ) )
		.add( scalarGradient2( data, gridSpacing, i1c, j1c, shape ).mul( w11 ) );

}

// Gradient of a scalar field at a continuous position: an analytic
// derivative of the same bilinear surface that collocatedValueAtPosition2
// samples (not a blend of the 4 corners' own discrete gradients like
// scalarGradientAtPosition2) -- cheaper (reads the 4 corner values only
// once) and matches exactly what collocatedValueAtPosition2 returns at that
// point. Use this when the gradient needs to match the sampled value
// precisely (e.g. an SDF's surface normal); use scalarGradientAtPosition2
// when a bit more smoothing across neighboring cells is desired.
export function bilinearGradientAtPosition2( data, gridSpacing, dataOrigin, pos, shape ) {

	const { i0c, j0c, i1c, j1c, w10, w01, w11 } = bilinearCoordsAndWeights2( pos, dataOrigin, gridSpacing, shape );

	const fx = w10.add( w11 );
	const fy = w01.add( w11 );

	const v00 = data( i0c, j0c );
	const v10 = data( i1c, j0c );
	const v01 = data( i0c, j1c );
	const v11 = data( i1c, j1c );

	const gx = float( 1 ).sub( fy ).mul( v10.sub( v00 ) ).add( fy.mul( v11.sub( v01 ) ) ).div( gridSpacing.x );
	const gy = float( 1 ).sub( fx ).mul( v01.sub( v00 ) ).add( fx.mul( v11.sub( v10 ) ) ).div( gridSpacing.y );

	return vec2( gx, gy );

}

// Gradient (Jacobian) of a vector field at a continuous position
export function vectorGradientAtPosition2( data, gridSpacing, dataOrigin, pos, shape ) {

	const { i0c, j0c, i1c, j1c, w00, w10, w01, w11 } = bilinearCoordsAndWeights2( pos, dataOrigin, gridSpacing, shape );

	return vectorGradient2( data, gridSpacing, i0c, j0c, shape ).mul( w00 )
		.add( vectorGradient2( data, gridSpacing, i1c, j0c, shape ).mul( w10 ) )
		.add( vectorGradient2( data, gridSpacing, i0c, j1c, shape ).mul( w01 ) )
		.add( vectorGradient2( data, gridSpacing, i1c, j1c, shape ).mul( w11 ) );

}

// ------------------------------------------------------------
// divergence

// data is a 2D vector field, gridSpacing is a vector; divergence at a grid point
export function collocatedDivergence2( data, gridSpacing, i, j, shape ) {

	const [ nx, ny ] = shape;
	const center = data( i, j );

	const left  = i.greaterThan( 0 ).select( data( i.sub( 1 ), j ).x, center.x );
	const right = i.add( 1 ).lessThan( nx ).select( data( i.add( 1 ), j ).x, center.x );
	const down  = j.greaterThan( 0 ).select( data( i, j.sub( 1 ) ).y, center.y );
	const up    = j.add( 1 ).lessThan( ny ).select( data( i, j.add( 1 ) ).y, center.y );

	return right.sub( left ).mul( 0.5 ).div( gridSpacing.x )
		.add( up.sub( down ).mul( 0.5 ).div( gridSpacing.y ) );

}

// dataU/dataV are two scalar fields; divergence at each cell center
export function faceCenteredDivergenceAtCenter2( dataU, dataV, gridSpacing, i, j ) {

	const leftU   = dataU( i, j );
	const rightU  = dataU( i.add( 1 ), j );
	const bottomV = dataV( i, j );
	const topV    = dataV( i, j.add( 1 ) );

	return rightU.sub( leftU ).div( gridSpacing.x )
		.add( topV.sub( bottomV ).div( gridSpacing.y ) );

}

// Divergence of a collocated vector field at a continuous position
export function collocatedDivergenceAtPosition2( data, gridSpacing, dataOrigin, pos, shape ) {

	const { i0c, j0c, i1c, j1c, w00, w10, w01, w11 } = bilinearCoordsAndWeights2( pos, dataOrigin, gridSpacing, shape );

	return collocatedDivergence2( data, gridSpacing, i0c, j0c, shape ).mul( w00 )
		.add( collocatedDivergence2( data, gridSpacing, i1c, j0c, shape ).mul( w10 ) )
		.add( collocatedDivergence2( data, gridSpacing, i0c, j1c, shape ).mul( w01 ) )
		.add( collocatedDivergence2( data, gridSpacing, i1c, j1c, shape ).mul( w11 ) );

}

// Divergence of a face-centered (MAC) vector field at a continuous position.
// cellCenterOrigin is the origin of the cell-center layout
// (dataOrigin + 0.5*gridSpacing), shape is the cell-center resolution (nx, ny)
export function faceCenteredDivergenceAtPosition2( dataU, dataV, gridSpacing, cellCenterOrigin, pos, shape ) {

	const { i0c, j0c, i1c, j1c, w00, w10, w01, w11 } = bilinearCoordsAndWeights2( pos, cellCenterOrigin, gridSpacing, shape );

	return faceCenteredDivergenceAtCenter2( dataU, dataV, gridSpacing, i0c, j0c ).mul( w00 )
		.add( faceCenteredDivergenceAtCenter2( dataU, dataV, gridSpacing, i1c, j0c ).mul( w10 ) )
		.add( faceCenteredDivergenceAtCenter2( dataU, dataV, gridSpacing, i0c, j1c ).mul( w01 ) )
		.add( faceCenteredDivergenceAtCenter2( dataU, dataV, gridSpacing, i1c, j1c ).mul( w11 ) );

}

// ------------------------------------------------------------
// curl

// data is a 2D vector field, gridSpacing is a vector; curl at a grid point
export function collocatedCurl2( data, gridSpacing, i, j, shape ) {

	const [ nx, ny ] = shape;
	const center = data( i, j );

	const left  = i.greaterThan( 0 ).select( data( i.sub( 1 ), j ), center );
	const right = i.add( 1 ).lessThan( nx ).select( data( i.add( 1 ), j ), center );
	const down  = j.greaterThan( 0 ).select( data( i, j.sub( 1 ) ), center );
	const up    = j.add( 1 ).lessThan( ny ).select( data( i, j.add( 1 ) ), center );

	const fxYm = down.x;
	const fxYp = up.x;
	const fyXm = left.y;
	const fyXp = right.y;

	return fyXp.sub( fyXm ).div( gridSpacing.x )
		.sub( fxYp.sub( fxYm ).div( gridSpacing.y ) )
		.mul( 0.5 );

}

// dataU/dataV are two scalar fields; curl at each cell center
export function faceCenteredCurlAtCenter2( dataU, dataV, gridSpacing, i, j, shape ) {

	const [ nx, ny ] = shape;

	const left  = faceCenteredValueAtCellCenter2( dataU, dataV, max( i.sub( 1 ), 0 ), j );
	const right = faceCenteredValueAtCellCenter2( dataU, dataV, min( i.add( 1 ), nx - 1 ), j );
	const up    = faceCenteredValueAtCellCenter2( dataU, dataV, i, min( j.add( 1 ), ny - 1 ) );
	const down  = faceCenteredValueAtCellCenter2( dataU, dataV, i, max( j.sub( 1 ), 0 ) );

	const fxYm = down.x;
	const fxYp = up.x;
	const fyXm = left.y;
	const fyXp = right.y;

	return fyXp.sub( fyXm ).div( gridSpacing.x )
		.sub( fxYp.sub( fxYm ).div( gridSpacing.y ) )
		.mul( 0.5 );

}

// Curl of a collocated vector field at a continuous position
export function collocatedCurlAtPosition2( data, gridSpacing, dataOrigin, pos, shape ) {

	const { i0c, j0c, i1c, j1c, w00, w10, w01, w11 } = bilinearCoordsAndWeights2( pos, dataOrigin, gridSpacing, shape );

	return collocatedCurl2( data, gridSpacing, i0c, j0c, shape ).mul( w00 )
		.add( collocatedCurl2( data, gridSpacing, i1c, j0c, shape ).mul( w10 ) )
		.add( collocatedCurl2( data, gridSpacing, i0c, j1c, shape ).mul( w01 ) )
		.add( collocatedCurl2( data, gridSpacing, i1c, j1c, shape ).mul( w11 ) );

}

// Curl of a face-centered (MAC) vector field at a continuous position.
// cellCenterOrigin/shape: see faceCenteredDivergenceAtPosition2
export function faceCenteredCurlAtPosition2( dataU, dataV, gridSpacing, cellCenterOrigin, pos, shape ) {

	const { i0c, j0c, i1c, j1c, w00, w10, w01, w11 } = bilinearCoordsAndWeights2( pos, cellCenterOrigin, gridSpacing, shape );

	return faceCenteredCurlAtCenter2( dataU, dataV, gridSpacing, i0c, j0c, shape ).mul( w00 )
		.add( faceCenteredCurlAtCenter2( dataU, dataV, gridSpacing, i1c, j0c, shape ).mul( w10 ) )
		.add( faceCenteredCurlAtCenter2( dataU, dataV, gridSpacing, i0c, j1c, shape ).mul( w01 ) )
		.add( faceCenteredCurlAtCenter2( dataU, dataV, gridSpacing, i1c, j1c, shape ).mul( w11 ) );

}

// ------------------------------------------------------------
// laplacian

// shape is the field's shape; laplacian at a grid point
export function scalarLaplacian2( data, gridSpacing, i, j, shape ) {

	const [ nx, ny ] = shape;
	const center = data( i, j );
	const zero = float( 0 );

	const dleft  = i.greaterThan( 0 ).select( center.sub( data( i.sub( 1 ), j ) ), zero );
	const dright = i.lessThan( nx - 1 ).select( data( i.add( 1 ), j ).sub( center ), zero );
	const dup    = j.lessThan( ny - 1 ).select( data( i, j.add( 1 ) ).sub( center ), zero );
	const ddown  = j.greaterThan( 0 ).select( center.sub( data( i, j.sub( 1 ) ) ), zero );

	return dright.sub( dleft ).div( gridSpacing.x.mul( gridSpacing.x ) )
		.add( dup.sub( ddown ).div( gridSpacing.y.mul( gridSpacing.y ) ) );

}

// Laplacian of a scalar field at a continuous position
export function scalarLaplacianAtPosition2( data, gridSpacing, dataOrigin, pos, shape ) {

	const { i0c, j0c, i1c, j1c, w00, w10, w01, w11 } = bilinearCoordsAndWeights2( pos, dataOrigin, gridSpacing, shape );

	return scalarLaplacian2( data, gridSpacing, i0c, j0c, shape ).mul( w00 )
		.add( scalarLaplacian2( data, gridSpacing, i1c, j0c, shape ).mul( w10 ) )
		.add( scalarLaplacian2( data, gridSpacing, i0c, j1c, shape ).mul( w01 ) )
		.add( scalarLaplacian2( data, gridSpacing, i1c, j1c, shape ).mul( w11 ) );

}

// shape is the field's shape; laplacian at a grid point
export function vectorLaplacian2( data, gridSpacing, i, j, shape ) {

	const [ nx, ny ] = shape;
	const center = data( i, j );
	const zero = vec2( 0 );

	const dleft  = i.greaterThan( 0 ).select( center.sub( data( i.sub( 1 ), j ) ), zero );
	const dright = i.lessThan( nx - 1 ).select( data( i.add( 1 ), j ).sub( center ), zero );
	const dup    = j.lessThan( ny - 1 ).select( data( i, j.add( 1 ) ).sub( center ), zero );
	const ddown  = j.greaterThan( 0 ).select( center.sub( data( i, j.sub( 1 ) ) ), zero );

	return dright.sub( dleft ).div( gridSpacing.x.mul( gridSpacing.x ) )
		.add( dup.sub( ddown ).div( gridSpacing.y.mul( gridSpacing.y ) ) );

}

// Laplacian of a vector field at a continuous position
export function vectorLaplacianAtPosition2( data, gridSpacing, dataOrigin, pos, shape ) {

	const { i0c, j0c, i1c, j1c, w00, w10, w01, w11 } = bilinearCoordsAndWeights2( pos, dataOrigin, gridSpacing, shape );

	return vectorLaplacian2( data, gridSpacing, i0c, j0c, shape ).mul( w00 )
		.add( vectorLaplacian2( data, gridSpacing, i1c, j0c, shape ).mul( w10 ) )
		.add( vectorLaplacian2( data, gridSpacing, i0c, j1c, shape ).mul( w01 ) )
		.add( vectorLaplacian2( data, gridSpacing, i1c, j1c, shape ).mul( w11 ) );

}
