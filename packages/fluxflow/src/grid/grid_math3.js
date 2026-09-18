// 3D counterpart of grid_math.js -- ported by extending that file's own
// already-verified formulas one axis at a time, not by re-deriving from
// grid_math.py, since the 2D file is itself the settled, real-hardware-
// confirmed reference for every formula here that has a direct 2D
// analogue. grid_math.js is untouched; this is a new, parallel file, same
// as jet/fluid-engine-dev's own Grid2/Grid3 split this whole port already
// follows for constant.js's DIRECTION_* flags and elsewhere.
//
// *** What does NOT extend by just adding an axis ***
//
// Curl is not a bigger version of the same thing in 3D -- in 2D, the curl
// of a vector field is a SCALAR (the z-component of the full 3D curl,
// since a 2D flow has no vorticity in-plane); in 3D it is a genuine VECTOR,
// curl(F) = (dFz/dy - dFy/dz, dFx/dz - dFz/dx, dFy/dx - dFx/dy). The
// collocated*Curl*/faceCentered*Curl* functions below are therefore not a
// line-for-line extension of grid_math.js's -- they are the standard
// central-difference discretisation of that vector formula, built fresh
// against the same boundary-clamping convention (data() clamped to the
// nearest interior neighbour past an edge) every other function here uses.
//
// *** What DOES extend directly, and the one place that needed care ***
//
// Interpolation, gradient, divergence and laplacian are literally the same
// stencil with a third axis appended -- bilinear's 4 corners become
// trilinear's 8, bicubic's two nested 1D passes become tricubic's three.
// vectorGradient3's mat3 construction is the one spot that needed real
// thought rather than mechanical extension: grid_math.js's own
// vectorGradient2 carries a documented, real-hardware-confirmed fix for
// TSL's mat2() filling column-major (column0=(dfx/dx,dfy/dx), not
// row-major) -- see that function's own header comment for the story.
// mat3() follows the same column-major convention (standard GLSL/WGSL, and
// three.js's mat2/mat3/mat4 constructors are consistent with each other),
// so vectorGradient3 below applies the identical column-per-axis layout
// one column wider: column0=(dfx/dx,dfy/dx,dfz/dx), column1=(.../dy),
// column2=(.../dz). Derived by symmetry with the already-fixed 2D case,
// not yet independently confirmed on real hardware the way the 2D fix
// was -- examples/00-grid-math/'s own asymmetric-test-field check is what
// caught the 2D bug, and a 3D counterpart of that check is what should
// confirm or refute this one.

import { int, float, vec3, mat3, min, max, floor, abs, or, clamp } from 'three/tsl';

// ------------------------------------------------------------
// interpolation

export function faceCenteredValueAtCellCenter3( dataU, dataV, dataW, i, j, k ) {

	return vec3(
		dataU( i, j, k ).add( dataU( i.add( 1 ), j, k ) ),
		dataV( i, j, k ).add( dataV( i, j.add( 1 ), k ) ),
		dataW( i, j, k ).add( dataW( i, j, k.add( 1 ) ) )
	).mul( 0.5 );

}

// Given a continuous position, finds the 8 surrounding grid indices
// (clamped to the bounds, so positions outside the grid still sample the
// boundary value) and their trilinear weights. Ordering matches the
// binary count of (x,y,z) high/low: 000, 100, 010, 110, 001, 101, 011, 111
// -- i.e. w000..w111 read as bits (z,y,x) from the suffix.
export function trilinearCoordsAndWeights3( pos, dataOrigin, gridSpacing, shape ) {

	const [ nx, ny, nz ] = shape;

	const gridPos = pos.sub( dataOrigin ).div( gridSpacing );
	const i0 = int( floor( gridPos.x ) );
	const j0 = int( floor( gridPos.y ) );
	const k0 = int( floor( gridPos.z ) );
	const fx = gridPos.x.sub( i0.toFloat() );
	const fy = gridPos.y.sub( j0.toFloat() );
	const fz = gridPos.z.sub( k0.toFloat() );

	const i0c = max( 0, min( i0, nx - 1 ) );
	const i1c = max( 0, min( i0.add( 1 ), nx - 1 ) );
	const j0c = max( 0, min( j0, ny - 1 ) );
	const j1c = max( 0, min( j0.add( 1 ), ny - 1 ) );
	const k0c = max( 0, min( k0, nz - 1 ) );
	const k1c = max( 0, min( k0.add( 1 ), nz - 1 ) );

	const gx = float( 1 ).sub( fx );
	const gy = float( 1 ).sub( fy );
	const gz = float( 1 ).sub( fz );

	const w000 = gx.mul( gy ).mul( gz );
	const w100 = fx.mul( gy ).mul( gz );
	const w010 = gx.mul( fy ).mul( gz );
	const w110 = fx.mul( fy ).mul( gz );
	const w001 = gx.mul( gy ).mul( fz );
	const w101 = fx.mul( gy ).mul( fz );
	const w011 = gx.mul( fy ).mul( fz );
	const w111 = fx.mul( fy ).mul( fz );

	return { i0c, j0c, k0c, i1c, j1c, k1c, w000, w100, w010, w110, w001, w101, w011, w111 };

}

// Trilinear sample of a collocated field (scalar or vector, either works)
// at a continuous position
export function collocatedValueAtPosition3( data, gridSpacing, dataOrigin, pos, shape ) {

	const { i0c, j0c, k0c, i1c, j1c, k1c, w000, w100, w010, w110, w001, w101, w011, w111 } =
		trilinearCoordsAndWeights3( pos, dataOrigin, gridSpacing, shape );

	return data( i0c, j0c, k0c ).mul( w000 )
		.add( data( i1c, j0c, k0c ).mul( w100 ) )
		.add( data( i0c, j1c, k0c ).mul( w010 ) )
		.add( data( i1c, j1c, k0c ).mul( w110 ) )
		.add( data( i0c, j0c, k1c ).mul( w001 ) )
		.add( data( i1c, j0c, k1c ).mul( w101 ) )
		.add( data( i0c, j1c, k1c ).mul( w011 ) )
		.add( data( i1c, j1c, k1c ).mul( w111 ) );

}

// Trilinear sample of a face-centered (MAC) velocity field at a continuous
// position. dataOriginU/dataOriginV/dataOriginW are the staggered origins
// of the u/v/w faces respectively
export function faceCenteredValueAtPosition3( dataU, dataV, dataW, gridSpacing, dataOriginU, dataOriginV, dataOriginW, pos, shapeU, shapeV, shapeW ) {

	const u = collocatedValueAtPosition3( dataU, gridSpacing, dataOriginU, pos, shapeU );
	const v = collocatedValueAtPosition3( dataV, gridSpacing, dataOriginV, pos, shapeV );
	const w = collocatedValueAtPosition3( dataW, gridSpacing, dataOriginW, pos, shapeW );

	return vec3( u, v, w );

}

// Monotonic cubic Hermite interpolation between f1 (at f=0) and f2 (at
// f=1) -- identical formula to grid_math.js's monotonicCubic1d, kept as
// its own copy rather than imported so this file has no dependency on the
// 2D one (see this file's own header: no shared state between the two).
// Named with the "3" suffix (unlike its logic-identical 2D sibling) so
// grid/index.js's barrel export can reach both -- an unsuffixed name here
// would collide with grid_math.js's own export in that shared namespace
// and silently disappear from it (ES module ambiguous star-export
// resolution, not an error, just a missing binding).
export function monotonicCubic1d3( f0, f1, f2, f3, f ) {

	const D1 = f2.sub( f1 );
	const rawD1 = f2.sub( f0 ).mul( 0.5 );
	const rawD2 = f3.sub( f1 ).mul( 0.5 );

	const isFlat = abs( D1 ).lessThan( 1e-12 );
	const d1 = or( isFlat, rawD1.mul( D1 ).lessThan( 0 ) ).select( float( 0 ), rawD1 );
	const d2 = or( isFlat, rawD2.mul( D1 ).lessThan( 0 ) ).select( float( 0 ), rawD2 );

	const a3 = d1.add( d2 ).sub( D1.mul( 2 ) );
	const a2 = D1.mul( 3 ).sub( d1.mul( 2 ) ).sub( d2 );
	const a1 = d1;
	const a0 = f1;

	return a3.mul( f ).mul( f ).mul( f )
		.add( a2.mul( f ).mul( f ) )
		.add( a1.mul( f ) )
		.add( a0 );

}

// The 4 clamped indices (im1, i0, i1, i2) and fractional part along one
// axis for a tricubic sample -- same both-ends-clamped convention as
// grid_math.js's cubicIndices1d.
function cubicIndices1d( coord, n ) {

	const i0 = int( floor( coord ) );
	const f = coord.sub( i0.toFloat() );

	const im1 = max( 0, min( i0.sub( 1 ), n - 1 ) );
	const i0c = max( 0, min( i0, n - 1 ) );
	const i1c = max( 0, min( i0.add( 1 ), n - 1 ) );
	const i2c = max( 0, min( i0.add( 2 ), n - 1 ) );

	return { im1, i0c, i1c, i2c, f };

}

// Monotonic tricubic sample of a collocated field at a continuous position
// -- the 3D tensor-product extension of collocatedCubicValueAtPosition2's
// 2D one: 4 cubic-in-x passes for each of the 4x4 (y,z) offset
// combinations (16 values), then 4 cubic-in-y passes reducing those to one
// value per z-offset (4 values), then one final cubic-in-z pass. 64 taps
// total, same monotonicCubic1d/cubicIndices1d building blocks as the 2D
// version, one nesting level deeper.
export function collocatedCubicValueAtPosition3( data, gridSpacing, dataOrigin, pos, shape ) {

	const [ nx, ny, nz ] = shape;
	const gridPos = pos.sub( dataOrigin ).div( gridSpacing );

	const xi = cubicIndices1d( gridPos.x, nx );
	const yi = cubicIndices1d( gridPos.y, ny );
	const zi = cubicIndices1d( gridPos.z, nz );

	const zOffsets = [ zi.im1, zi.i0c, zi.i1c, zi.i2c ];
	const yOffsets = [ yi.im1, yi.i0c, yi.i1c, yi.i2c ];

	const zValues = zOffsets.map( ( k ) => {

		const rowValues = yOffsets.map( ( j ) => monotonicCubic1d3(
			data( xi.im1, j, k ), data( xi.i0c, j, k ), data( xi.i1c, j, k ), data( xi.i2c, j, k ), xi.f
		) );

		return monotonicCubic1d3( rowValues[ 0 ], rowValues[ 1 ], rowValues[ 2 ], rowValues[ 3 ], yi.f );

	} );

	return monotonicCubic1d3( zValues[ 0 ], zValues[ 1 ], zValues[ 2 ], zValues[ 3 ], zi.f );

}

// Monotonic tricubic sample of a face-centered (MAC) velocity field at a
// continuous position -- cubic counterpart of faceCenteredValueAtPosition3.
export function faceCenteredCubicValueAtPosition3( dataU, dataV, dataW, gridSpacing, dataOriginU, dataOriginV, dataOriginW, pos, shapeU, shapeV, shapeW ) {

	const u = collocatedCubicValueAtPosition3( dataU, gridSpacing, dataOriginU, pos, shapeU );
	const v = collocatedCubicValueAtPosition3( dataV, gridSpacing, dataOriginV, pos, shapeV );
	const w = collocatedCubicValueAtPosition3( dataW, gridSpacing, dataOriginW, pos, shapeW );

	return vec3( u, v, w );

}

// ------------------------------------------------------------
// gradient

// shape is the field's shape; gradient at a grid point
export function scalarGradient3( data, gridSpacing, i, j, k, shape ) {

	const [ nx, ny, nz ] = shape;
	const center = data( i, j, k );

	const left  = i.greaterThan( 0 ).select( data( i.sub( 1 ), j, k ), center );
	const right = i.add( 1 ).lessThan( nx ).select( data( i.add( 1 ), j, k ), center );
	const down  = j.greaterThan( 0 ).select( data( i, j.sub( 1 ), k ), center );
	const up    = j.add( 1 ).lessThan( ny ).select( data( i, j.add( 1 ), k ), center );
	const back  = k.greaterThan( 0 ).select( data( i, j, k.sub( 1 ) ), center );
	const front = k.add( 1 ).lessThan( nz ).select( data( i, j, k.add( 1 ) ), center );

	return vec3( right.sub( left ), up.sub( down ), front.sub( back ) ).mul( 0.5 ).div( gridSpacing );

}

// shape is the field's shape; gradient (Jacobian) at a grid point. See this
// file's own header comment for the mat3 column-major reasoning.
export function vectorGradient3( data, gridSpacing, i, j, k, shape ) {

	const [ nx, ny, nz ] = shape;
	const center = data( i, j, k );

	const left  = i.greaterThan( 0 ).select( data( i.sub( 1 ), j, k ), center );
	const right = i.add( 1 ).lessThan( nx ).select( data( i.add( 1 ), j, k ), center );
	const down  = j.greaterThan( 0 ).select( data( i, j.sub( 1 ), k ), center );
	const up    = j.add( 1 ).lessThan( ny ).select( data( i, j.add( 1 ), k ), center );
	const back  = k.greaterThan( 0 ).select( data( i, j, k.sub( 1 ) ), center );
	const front = k.add( 1 ).lessThan( nz ).select( data( i, j, k.add( 1 ) ), center );

	const gradX = vec3( right.x.sub( left.x ), up.x.sub( down.x ), front.x.sub( back.x ) ).mul( 0.5 ).div( gridSpacing );
	const gradY = vec3( right.y.sub( left.y ), up.y.sub( down.y ), front.y.sub( back.y ) ).mul( 0.5 ).div( gridSpacing );
	const gradZ = vec3( right.z.sub( left.z ), up.z.sub( down.z ), front.z.sub( back.z ) ).mul( 0.5 ).div( gridSpacing );

	// column0 = (dfx/dx, dfy/dx, dfz/dx), column1 = (.../dy), column2 = (.../dz)
	// -- one column per differentiation axis, each holding every component's
	// partial derivative along it, matching vectorGradient2's fix one column
	// wider. See this file's header comment: derived by symmetry, not yet
	// independently confirmed on real hardware.
	return mat3(
		gradX.x, gradY.x, gradZ.x,
		gradX.y, gradY.y, gradZ.y,
		gradX.z, gradY.z, gradZ.z
	);

}

// Gradient of a scalar field at a continuous position: trilinearly blends
// the discrete gradients at the 8 surrounding grid points
export function scalarGradientAtPosition3( data, gridSpacing, dataOrigin, pos, shape ) {

	const { i0c, j0c, k0c, i1c, j1c, k1c, w000, w100, w010, w110, w001, w101, w011, w111 } =
		trilinearCoordsAndWeights3( pos, dataOrigin, gridSpacing, shape );

	return scalarGradient3( data, gridSpacing, i0c, j0c, k0c, shape ).mul( w000 )
		.add( scalarGradient3( data, gridSpacing, i1c, j0c, k0c, shape ).mul( w100 ) )
		.add( scalarGradient3( data, gridSpacing, i0c, j1c, k0c, shape ).mul( w010 ) )
		.add( scalarGradient3( data, gridSpacing, i1c, j1c, k0c, shape ).mul( w110 ) )
		.add( scalarGradient3( data, gridSpacing, i0c, j0c, k1c, shape ).mul( w001 ) )
		.add( scalarGradient3( data, gridSpacing, i1c, j0c, k1c, shape ).mul( w101 ) )
		.add( scalarGradient3( data, gridSpacing, i0c, j1c, k1c, shape ).mul( w011 ) )
		.add( scalarGradient3( data, gridSpacing, i1c, j1c, k1c, shape ).mul( w111 ) );

}

// One axis's gradient-safe (lo, hi, t) triple -- see grid_math.js's own
// gradientAxisInfo header comment for the full reasoning (a *gradient*
// needs two genuinely distinct indices near the grid's edge, unlike a
// *sampled value*, which is allowed to collapse both indices to the same
// cell there).
function gradientAxisInfo( coord, n ) {

	const idx = int( floor( coord ) );
	const lo = max( 0, min( idx, n - 2 ) );
	const hi = min( lo.add( 1 ), n - 1 );
	const t = clamp( coord.sub( lo.toFloat() ), 0, 1 );

	return { lo, hi, t };

}

// Gradient of a scalar field at a continuous position: an analytic
// derivative of the same trilinear surface that collocatedValueAtPosition3
// samples -- see grid_math.js's own bilinearGradientAtPosition2 for why
// this exists separately from scalarGradientAtPosition3 (matches the
// sampled value exactly; cheaper; use the other when a bit more smoothing
// across neighbouring cells is wanted instead).
export function trilinearGradientAtPosition3( data, gridSpacing, dataOrigin, pos, shape ) {

	const [ nx, ny, nz ] = shape;
	const gridPos = pos.sub( dataOrigin ).div( gridSpacing );

	const xi = gradientAxisInfo( gridPos.x, nx );
	const yi = gradientAxisInfo( gridPos.y, ny );
	const zi = gradientAxisInfo( gridPos.z, nz );

	const v000 = data( xi.lo, yi.lo, zi.lo );
	const v100 = data( xi.hi, yi.lo, zi.lo );
	const v010 = data( xi.lo, yi.hi, zi.lo );
	const v110 = data( xi.hi, yi.hi, zi.lo );
	const v001 = data( xi.lo, yi.lo, zi.hi );
	const v101 = data( xi.hi, yi.lo, zi.hi );
	const v011 = data( xi.lo, yi.hi, zi.hi );
	const v111 = data( xi.hi, yi.hi, zi.hi );

	// Each axis's derivative is the bilinear blend, over the OTHER two
	// axes' interpolation parameters, of that axis's own finite difference
	// between its lo/hi corners -- the trilinear surface's exact partial
	// derivative, same construction as bilinearGradientAtPosition2's gx/gy
	// one axis wider.
	const gx = float( 1 ).sub( yi.t ).mul( float( 1 ).sub( zi.t ) ).mul( v100.sub( v000 ) )
		.add( yi.t.mul( float( 1 ).sub( zi.t ) ).mul( v110.sub( v010 ) ) )
		.add( float( 1 ).sub( yi.t ).mul( zi.t ).mul( v101.sub( v001 ) ) )
		.add( yi.t.mul( zi.t ).mul( v111.sub( v011 ) ) )
		.div( gridSpacing.x );

	const gy = float( 1 ).sub( xi.t ).mul( float( 1 ).sub( zi.t ) ).mul( v010.sub( v000 ) )
		.add( xi.t.mul( float( 1 ).sub( zi.t ) ).mul( v110.sub( v100 ) ) )
		.add( float( 1 ).sub( xi.t ).mul( zi.t ).mul( v011.sub( v001 ) ) )
		.add( xi.t.mul( zi.t ).mul( v111.sub( v101 ) ) )
		.div( gridSpacing.y );

	const gz = float( 1 ).sub( xi.t ).mul( float( 1 ).sub( yi.t ) ).mul( v001.sub( v000 ) )
		.add( xi.t.mul( float( 1 ).sub( yi.t ) ).mul( v101.sub( v100 ) ) )
		.add( float( 1 ).sub( xi.t ).mul( yi.t ).mul( v011.sub( v010 ) ) )
		.add( xi.t.mul( yi.t ).mul( v111.sub( v110 ) ) )
		.div( gridSpacing.z );

	return vec3( gx, gy, gz );

}

// Gradient (Jacobian) of a vector field at a continuous position
export function vectorGradientAtPosition3( data, gridSpacing, dataOrigin, pos, shape ) {

	const { i0c, j0c, k0c, i1c, j1c, k1c, w000, w100, w010, w110, w001, w101, w011, w111 } =
		trilinearCoordsAndWeights3( pos, dataOrigin, gridSpacing, shape );

	return vectorGradient3( data, gridSpacing, i0c, j0c, k0c, shape ).mul( w000 )
		.add( vectorGradient3( data, gridSpacing, i1c, j0c, k0c, shape ).mul( w100 ) )
		.add( vectorGradient3( data, gridSpacing, i0c, j1c, k0c, shape ).mul( w010 ) )
		.add( vectorGradient3( data, gridSpacing, i1c, j1c, k0c, shape ).mul( w110 ) )
		.add( vectorGradient3( data, gridSpacing, i0c, j0c, k1c, shape ).mul( w001 ) )
		.add( vectorGradient3( data, gridSpacing, i1c, j0c, k1c, shape ).mul( w101 ) )
		.add( vectorGradient3( data, gridSpacing, i0c, j1c, k1c, shape ).mul( w011 ) )
		.add( vectorGradient3( data, gridSpacing, i1c, j1c, k1c, shape ).mul( w111 ) );

}

// ------------------------------------------------------------
// divergence

// data is a 3D vector field, gridSpacing is a vector; divergence at a grid point
export function collocatedDivergence3( data, gridSpacing, i, j, k, shape ) {

	const [ nx, ny, nz ] = shape;
	const center = data( i, j, k );

	const left  = i.greaterThan( 0 ).select( data( i.sub( 1 ), j, k ).x, center.x );
	const right = i.add( 1 ).lessThan( nx ).select( data( i.add( 1 ), j, k ).x, center.x );
	const down  = j.greaterThan( 0 ).select( data( i, j.sub( 1 ), k ).y, center.y );
	const up    = j.add( 1 ).lessThan( ny ).select( data( i, j.add( 1 ), k ).y, center.y );
	const back  = k.greaterThan( 0 ).select( data( i, j, k.sub( 1 ) ).z, center.z );
	const front = k.add( 1 ).lessThan( nz ).select( data( i, j, k.add( 1 ) ).z, center.z );

	return right.sub( left ).mul( 0.5 ).div( gridSpacing.x )
		.add( up.sub( down ).mul( 0.5 ).div( gridSpacing.y ) )
		.add( front.sub( back ).mul( 0.5 ).div( gridSpacing.z ) );

}

// dataU/dataV/dataW are three scalar fields; divergence at each cell centre
export function faceCenteredDivergenceAtCenter3( dataU, dataV, dataW, gridSpacing, i, j, k ) {

	const leftU   = dataU( i, j, k );
	const rightU  = dataU( i.add( 1 ), j, k );
	const bottomV = dataV( i, j, k );
	const topV    = dataV( i, j.add( 1 ), k );
	const backW   = dataW( i, j, k );
	const frontW  = dataW( i, j, k.add( 1 ) );

	return rightU.sub( leftU ).div( gridSpacing.x )
		.add( topV.sub( bottomV ).div( gridSpacing.y ) )
		.add( frontW.sub( backW ).div( gridSpacing.z ) );

}

// Divergence of a collocated vector field at a continuous position
export function collocatedDivergenceAtPosition3( data, gridSpacing, dataOrigin, pos, shape ) {

	const { i0c, j0c, k0c, i1c, j1c, k1c, w000, w100, w010, w110, w001, w101, w011, w111 } =
		trilinearCoordsAndWeights3( pos, dataOrigin, gridSpacing, shape );

	return collocatedDivergence3( data, gridSpacing, i0c, j0c, k0c, shape ).mul( w000 )
		.add( collocatedDivergence3( data, gridSpacing, i1c, j0c, k0c, shape ).mul( w100 ) )
		.add( collocatedDivergence3( data, gridSpacing, i0c, j1c, k0c, shape ).mul( w010 ) )
		.add( collocatedDivergence3( data, gridSpacing, i1c, j1c, k0c, shape ).mul( w110 ) )
		.add( collocatedDivergence3( data, gridSpacing, i0c, j0c, k1c, shape ).mul( w001 ) )
		.add( collocatedDivergence3( data, gridSpacing, i1c, j0c, k1c, shape ).mul( w101 ) )
		.add( collocatedDivergence3( data, gridSpacing, i0c, j1c, k1c, shape ).mul( w011 ) )
		.add( collocatedDivergence3( data, gridSpacing, i1c, j1c, k1c, shape ).mul( w111 ) );

}

// Divergence of a face-centered (MAC) vector field at a continuous
// position. cellCenterOrigin is the origin of the cell-center layout
// (dataOrigin + 0.5*gridSpacing), shape is the cell-center resolution
export function faceCenteredDivergenceAtPosition3( dataU, dataV, dataW, gridSpacing, cellCenterOrigin, pos, shape ) {

	const { i0c, j0c, k0c, i1c, j1c, k1c, w000, w100, w010, w110, w001, w101, w011, w111 } =
		trilinearCoordsAndWeights3( pos, cellCenterOrigin, gridSpacing, shape );

	return faceCenteredDivergenceAtCenter3( dataU, dataV, dataW, gridSpacing, i0c, j0c, k0c ).mul( w000 )
		.add( faceCenteredDivergenceAtCenter3( dataU, dataV, dataW, gridSpacing, i1c, j0c, k0c ).mul( w100 ) )
		.add( faceCenteredDivergenceAtCenter3( dataU, dataV, dataW, gridSpacing, i0c, j1c, k0c ).mul( w010 ) )
		.add( faceCenteredDivergenceAtCenter3( dataU, dataV, dataW, gridSpacing, i1c, j1c, k0c ).mul( w110 ) )
		.add( faceCenteredDivergenceAtCenter3( dataU, dataV, dataW, gridSpacing, i0c, j0c, k1c ).mul( w001 ) )
		.add( faceCenteredDivergenceAtCenter3( dataU, dataV, dataW, gridSpacing, i1c, j0c, k1c ).mul( w101 ) )
		.add( faceCenteredDivergenceAtCenter3( dataU, dataV, dataW, gridSpacing, i0c, j1c, k1c ).mul( w011 ) )
		.add( faceCenteredDivergenceAtCenter3( dataU, dataV, dataW, gridSpacing, i1c, j1c, k1c ).mul( w111 ) );

}

// ------------------------------------------------------------
// curl -- a VECTOR in 3D, not a scalar. See this file's own header comment.

// data is a 3D vector field, gridSpacing is a vector; curl at a grid point.
// curl(F) = (dFz/dy - dFy/dz, dFx/dz - dFz/dx, dFy/dx - dFx/dy), each
// partial a central difference with the same boundary-clamp convention
// (data() clamped to the nearest interior neighbour) every function here
// uses.
export function collocatedCurl3( data, gridSpacing, i, j, k, shape ) {

	const [ nx, ny, nz ] = shape;
	const center = data( i, j, k );

	const left  = i.greaterThan( 0 ).select( data( i.sub( 1 ), j, k ), center );
	const right = i.add( 1 ).lessThan( nx ).select( data( i.add( 1 ), j, k ), center );
	const down  = j.greaterThan( 0 ).select( data( i, j.sub( 1 ), k ), center );
	const up    = j.add( 1 ).lessThan( ny ).select( data( i, j.add( 1 ), k ), center );
	const back  = k.greaterThan( 0 ).select( data( i, j, k.sub( 1 ) ), center );
	const front = k.add( 1 ).lessThan( nz ).select( data( i, j, k.add( 1 ) ), center );

	const dFzDy = up.z.sub( down.z ).mul( 0.5 ).div( gridSpacing.y );
	const dFyDz = front.y.sub( back.y ).mul( 0.5 ).div( gridSpacing.z );

	const dFxDz = front.x.sub( back.x ).mul( 0.5 ).div( gridSpacing.z );
	const dFzDx = right.z.sub( left.z ).mul( 0.5 ).div( gridSpacing.x );

	const dFyDx = right.y.sub( left.y ).mul( 0.5 ).div( gridSpacing.x );
	const dFxDy = up.x.sub( down.x ).mul( 0.5 ).div( gridSpacing.y );

	return vec3(
		dFzDy.sub( dFyDz ),
		dFxDz.sub( dFzDx ),
		dFyDx.sub( dFxDy )
	);

}

// dataU/dataV/dataW are three scalar fields; curl at each cell centre
export function faceCenteredCurlAtCenter3( dataU, dataV, dataW, gridSpacing, i, j, k, shape ) {

	const [ nx, ny, nz ] = shape;

	const left  = faceCenteredValueAtCellCenter3( dataU, dataV, dataW, max( i.sub( 1 ), 0 ), j, k );
	const right = faceCenteredValueAtCellCenter3( dataU, dataV, dataW, min( i.add( 1 ), nx - 1 ), j, k );
	const down  = faceCenteredValueAtCellCenter3( dataU, dataV, dataW, i, max( j.sub( 1 ), 0 ), k );
	const up    = faceCenteredValueAtCellCenter3( dataU, dataV, dataW, i, min( j.add( 1 ), ny - 1 ), k );
	const back  = faceCenteredValueAtCellCenter3( dataU, dataV, dataW, i, j, max( k.sub( 1 ), 0 ) );
	const front = faceCenteredValueAtCellCenter3( dataU, dataV, dataW, i, j, min( k.add( 1 ), nz - 1 ) );

	const dFzDy = up.z.sub( down.z ).mul( 0.5 ).div( gridSpacing.y );
	const dFyDz = front.y.sub( back.y ).mul( 0.5 ).div( gridSpacing.z );

	const dFxDz = front.x.sub( back.x ).mul( 0.5 ).div( gridSpacing.z );
	const dFzDx = right.z.sub( left.z ).mul( 0.5 ).div( gridSpacing.x );

	const dFyDx = right.y.sub( left.y ).mul( 0.5 ).div( gridSpacing.x );
	const dFxDy = up.x.sub( down.x ).mul( 0.5 ).div( gridSpacing.y );

	return vec3(
		dFzDy.sub( dFyDz ),
		dFxDz.sub( dFzDx ),
		dFyDx.sub( dFxDy )
	);

}

// Curl of a collocated vector field at a continuous position
export function collocatedCurlAtPosition3( data, gridSpacing, dataOrigin, pos, shape ) {

	const { i0c, j0c, k0c, i1c, j1c, k1c, w000, w100, w010, w110, w001, w101, w011, w111 } =
		trilinearCoordsAndWeights3( pos, dataOrigin, gridSpacing, shape );

	return collocatedCurl3( data, gridSpacing, i0c, j0c, k0c, shape ).mul( w000 )
		.add( collocatedCurl3( data, gridSpacing, i1c, j0c, k0c, shape ).mul( w100 ) )
		.add( collocatedCurl3( data, gridSpacing, i0c, j1c, k0c, shape ).mul( w010 ) )
		.add( collocatedCurl3( data, gridSpacing, i1c, j1c, k0c, shape ).mul( w110 ) )
		.add( collocatedCurl3( data, gridSpacing, i0c, j0c, k1c, shape ).mul( w001 ) )
		.add( collocatedCurl3( data, gridSpacing, i1c, j0c, k1c, shape ).mul( w101 ) )
		.add( collocatedCurl3( data, gridSpacing, i0c, j1c, k1c, shape ).mul( w011 ) )
		.add( collocatedCurl3( data, gridSpacing, i1c, j1c, k1c, shape ).mul( w111 ) );

}

// Curl of a face-centered (MAC) vector field at a continuous position.
// cellCenterOrigin/shape: see faceCenteredDivergenceAtPosition3
export function faceCenteredCurlAtPosition3( dataU, dataV, dataW, gridSpacing, cellCenterOrigin, pos, shape ) {

	const { i0c, j0c, k0c, i1c, j1c, k1c, w000, w100, w010, w110, w001, w101, w011, w111 } =
		trilinearCoordsAndWeights3( pos, cellCenterOrigin, gridSpacing, shape );

	return faceCenteredCurlAtCenter3( dataU, dataV, dataW, gridSpacing, i0c, j0c, k0c, shape ).mul( w000 )
		.add( faceCenteredCurlAtCenter3( dataU, dataV, dataW, gridSpacing, i1c, j0c, k0c, shape ).mul( w100 ) )
		.add( faceCenteredCurlAtCenter3( dataU, dataV, dataW, gridSpacing, i0c, j1c, k0c, shape ).mul( w010 ) )
		.add( faceCenteredCurlAtCenter3( dataU, dataV, dataW, gridSpacing, i1c, j1c, k0c, shape ).mul( w110 ) )
		.add( faceCenteredCurlAtCenter3( dataU, dataV, dataW, gridSpacing, i0c, j0c, k1c, shape ).mul( w001 ) )
		.add( faceCenteredCurlAtCenter3( dataU, dataV, dataW, gridSpacing, i1c, j0c, k1c, shape ).mul( w101 ) )
		.add( faceCenteredCurlAtCenter3( dataU, dataV, dataW, gridSpacing, i0c, j1c, k1c, shape ).mul( w011 ) )
		.add( faceCenteredCurlAtCenter3( dataU, dataV, dataW, gridSpacing, i1c, j1c, k1c, shape ).mul( w111 ) );

}

// ------------------------------------------------------------
// laplacian

// shape is the field's shape; laplacian at a grid point
export function scalarLaplacian3( data, gridSpacing, i, j, k, shape ) {

	const [ nx, ny, nz ] = shape;
	const center = data( i, j, k );
	const zero = float( 0 );

	const dleft  = i.greaterThan( 0 ).select( center.sub( data( i.sub( 1 ), j, k ) ), zero );
	const dright = i.lessThan( nx - 1 ).select( data( i.add( 1 ), j, k ).sub( center ), zero );
	const dup    = j.lessThan( ny - 1 ).select( data( i, j.add( 1 ), k ).sub( center ), zero );
	const ddown  = j.greaterThan( 0 ).select( center.sub( data( i, j.sub( 1 ), k ) ), zero );
	const dfront = k.lessThan( nz - 1 ).select( data( i, j, k.add( 1 ) ).sub( center ), zero );
	const dback  = k.greaterThan( 0 ).select( center.sub( data( i, j, k.sub( 1 ) ) ), zero );

	return dright.sub( dleft ).div( gridSpacing.x.mul( gridSpacing.x ) )
		.add( dup.sub( ddown ).div( gridSpacing.y.mul( gridSpacing.y ) ) )
		.add( dfront.sub( dback ).div( gridSpacing.z.mul( gridSpacing.z ) ) );

}

// Laplacian of a scalar field at a continuous position
export function scalarLaplacianAtPosition3( data, gridSpacing, dataOrigin, pos, shape ) {

	const { i0c, j0c, k0c, i1c, j1c, k1c, w000, w100, w010, w110, w001, w101, w011, w111 } =
		trilinearCoordsAndWeights3( pos, dataOrigin, gridSpacing, shape );

	return scalarLaplacian3( data, gridSpacing, i0c, j0c, k0c, shape ).mul( w000 )
		.add( scalarLaplacian3( data, gridSpacing, i1c, j0c, k0c, shape ).mul( w100 ) )
		.add( scalarLaplacian3( data, gridSpacing, i0c, j1c, k0c, shape ).mul( w010 ) )
		.add( scalarLaplacian3( data, gridSpacing, i1c, j1c, k0c, shape ).mul( w110 ) )
		.add( scalarLaplacian3( data, gridSpacing, i0c, j0c, k1c, shape ).mul( w001 ) )
		.add( scalarLaplacian3( data, gridSpacing, i1c, j0c, k1c, shape ).mul( w101 ) )
		.add( scalarLaplacian3( data, gridSpacing, i0c, j1c, k1c, shape ).mul( w011 ) )
		.add( scalarLaplacian3( data, gridSpacing, i1c, j1c, k1c, shape ).mul( w111 ) );

}

// shape is the field's shape; laplacian at a grid point
export function vectorLaplacian3( data, gridSpacing, i, j, k, shape ) {

	const [ nx, ny, nz ] = shape;
	const center = data( i, j, k );
	const zero = vec3( 0 );

	const dleft  = i.greaterThan( 0 ).select( center.sub( data( i.sub( 1 ), j, k ) ), zero );
	const dright = i.lessThan( nx - 1 ).select( data( i.add( 1 ), j, k ).sub( center ), zero );
	const dup    = j.lessThan( ny - 1 ).select( data( i, j.add( 1 ), k ).sub( center ), zero );
	const ddown  = j.greaterThan( 0 ).select( center.sub( data( i, j.sub( 1 ), k ) ), zero );
	const dfront = k.lessThan( nz - 1 ).select( data( i, j, k.add( 1 ) ).sub( center ), zero );
	const dback  = k.greaterThan( 0 ).select( center.sub( data( i, j, k.sub( 1 ) ) ), zero );

	return dright.sub( dleft ).div( gridSpacing.x.mul( gridSpacing.x ) )
		.add( dup.sub( ddown ).div( gridSpacing.y.mul( gridSpacing.y ) ) )
		.add( dfront.sub( dback ).div( gridSpacing.z.mul( gridSpacing.z ) ) );

}

// Laplacian of a vector field at a continuous position
export function vectorLaplacianAtPosition3( data, gridSpacing, dataOrigin, pos, shape ) {

	const { i0c, j0c, k0c, i1c, j1c, k1c, w000, w100, w010, w110, w001, w101, w011, w111 } =
		trilinearCoordsAndWeights3( pos, dataOrigin, gridSpacing, shape );

	return vectorLaplacian3( data, gridSpacing, i0c, j0c, k0c, shape ).mul( w000 )
		.add( vectorLaplacian3( data, gridSpacing, i1c, j0c, k0c, shape ).mul( w100 ) )
		.add( vectorLaplacian3( data, gridSpacing, i0c, j1c, k0c, shape ).mul( w010 ) )
		.add( vectorLaplacian3( data, gridSpacing, i1c, j1c, k0c, shape ).mul( w110 ) )
		.add( vectorLaplacian3( data, gridSpacing, i0c, j0c, k1c, shape ).mul( w001 ) )
		.add( vectorLaplacian3( data, gridSpacing, i1c, j0c, k1c, shape ).mul( w101 ) )
		.add( vectorLaplacian3( data, gridSpacing, i0c, j1c, k1c, shape ).mul( w011 ) )
		.add( vectorLaplacian3( data, gridSpacing, i1c, j1c, k1c, shape ).mul( w111 ) );

}
