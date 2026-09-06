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

import { int, float, vec2, mat2, min, max, floor, abs, or, clamp } from 'three/tsl';

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

// Monotonic cubic Hermite interpolation between f1 (at f=0) and f2 (at
// f=1), using f0/f3 as the outer neighbors for tangent estimation --
// Fedkiw, Stam & Jensen's clamped-Catmull-Rom scheme ("Visual Simulation
// of Smoke", SIGGRAPH 2001, section on higher-order advection). The raw
// Catmull-Rom tangent at each of the two inner points (f1, f2) is zeroed
// whenever it disagrees in sign with the actual secant slope between them
// (or whenever that secant is itself ~0) -- this is what prevents the
// cubic from overshooting past f1/f2 the way a raw, unclamped
// Catmull-Rom spline can, at the cost of degrading to a flatter
// (sometimes locally linear-ish) interpolant right where the data isn't
// monotonic. Ported from jet/fluid-engine-dev's `monotonicCatmullRom`
// (math_utils.h) -- see ../../THIRD-PARTY-NOTICES.md. Verified against a
// plain-JS reference before porting: reconstructs linear data exactly,
// stays within [min,max] of the interpolated interval on a step function
// (no overshoot), and doesn't overshoot past a local extremum either.
export function monotonicCubic1d( f0, f1, f2, f3, f ) {

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
// axis for a bicubic sample -- both-ends-clamped exactly like jet's
// CubicArraySampler2 (`max(i-1,0)`, `i`, `min(i+1,n-1)`, `min(i+2,n-1)`),
// so positions outside the grid still sample the boundary value, same
// spirit as bilinearCoordsAndWeights2 above.
function cubicIndices1d( coord, n ) {

	const i0 = int( floor( coord ) );
	const f = coord.sub( i0.toFloat() );

	const im1 = max( 0, min( i0.sub( 1 ), n - 1 ) );
	const i0c = max( 0, min( i0, n - 1 ) );
	const i1c = max( 0, min( i0.add( 1 ), n - 1 ) );
	const i2c = max( 0, min( i0.add( 2 ), n - 1 ) );

	return { im1, i0c, i1c, i2c, f };

}

// Monotonic bicubic sample of a collocated field at a continuous position
// -- the 2D tensor-product structure jet's CubicArraySampler2 uses (X
// across each of 4 rows, then Y on those 4 results), built from
// monotonicCubic1d/cubicIndices1d above. Unlike bilinearCoordsAndWeights2,
// the monotonicity clamp depends on the actual field values (not just
// position), so there's no reusable position-only "weights" struct here --
// this takes the field and position together and returns the value
// directly, same shape as collocatedValueAtPosition2 itself.
export function collocatedCubicValueAtPosition2( data, gridSpacing, dataOrigin, pos, shape ) {

	const [ nx, ny ] = shape;
	const gridPos = pos.sub( dataOrigin ).div( gridSpacing );

	const xi = cubicIndices1d( gridPos.x, nx );
	const yi = cubicIndices1d( gridPos.y, ny );

	const rowValues = [ yi.im1, yi.i0c, yi.i1c, yi.i2c ].map( ( j ) => monotonicCubic1d(
		data( xi.im1, j ), data( xi.i0c, j ), data( xi.i1c, j ), data( xi.i2c, j ), xi.f
	) );

	return monotonicCubic1d( rowValues[ 0 ], rowValues[ 1 ], rowValues[ 2 ], rowValues[ 3 ], yi.f );

}

// Monotonic bicubic sample of a face-centered (MAC) velocity field at a
// continuous position -- cubic counterpart of faceCenteredValueAtPosition2.
export function faceCenteredCubicValueAtPosition2( dataU, dataV, gridSpacing, dataOriginU, dataOriginV, pos, shapeU, shapeV ) {

	const u = collocatedCubicValueAtPosition2( dataU, gridSpacing, dataOriginU, pos, shapeU );
	const v = collocatedCubicValueAtPosition2( dataV, gridSpacing, dataOriginV, pos, shapeV );

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

// One axis's gradient-safe (lo, hi, t) triple, used only by
// bilinearGradientAtPosition2 below -- deliberately NOT the same clamped
// pair bilinearCoordsAndWeights2 returns (i0c/i1c above), because that
// pair is allowed to collapse to the *same* index once `coord` runs a half
// cell past the grid's own last cell center (whichever of i0c/i1c would
// have gone out of range gets clamped to the other one instead). That's
// the right behavior for a *sampled value* (clamp-to-boundary), but wrong
// for a *gradient*: reading the same cell twice makes that axis's finite
// difference exactly (v-v)=0, not a real degenerate case, just an
// implementation artifact of clamping both indices to the same cell.
//
// CONFIRMED as a real, real-hardware bug via examples/15-flow-past-cylinder/:
// grid_outflow_solver2.js's applyOutflowVelocityBC() samples an outflow
// SDF's gradient (this function, via sdf_collider2.js's own `gradient()`)
// at U-face positions that run half a cell past the SDF's own last cell
// center (the SDF is a resolutionX-wide cell-centered grid, cell centers
// at x=0.5..63.5 for resolutionX=64, but a face-centered velocity grid's
// own U-faces run one further, to x=64 exactly). At that exact point,
// i0c/i1c both clamped to the same last column, degenerating the SDF's
// own gradient to (0, gy) instead of a real, non-zero horizontal
// component -- silently breaking this code's own "which way is upstream"
// reference exactly where it mattered most, confirmed as a genuine
// contributor to a real-hardware divergence (not merely float32 overflow:
// the outer PCG loop this fed into is separately confirmed to need real
// multigrid coarsening -- see multigrid.js's own header comment -- but a
// spuriously-zeroed gradient direction here is a second, independent
// problem on top of that one).
//
// Fixed here (in the shared function, not just outflow's own call site --
// sdf_collider2.js's `gradient()` is also used for collider no-flux
// projection, and this fix helps that too, at no cost, since a collider
// SDF query point rarely lands exactly on the grid's own physical edge):
// `lo`/`hi` are kept as two genuinely DISTINCT indices whenever the grid
// has at least 2 cells along this axis (clamping `lo` back by one instead
// of letting `hi` collapse onto it), and `t` -- the blend weight between
// them -- is clamped to [0,1] rather than left to run past 1 the way a
// query point beyond the grid would otherwise produce. Net effect: a
// query point beyond the grid's own cell-center range now reads back the
// boundary's own real (non-degenerate) gradient, a 0th-order constant
// extrapolation past the edge -- not a spurious 0 in that axis. Fully
// backward-compatible for any in-range query (`idx` already within
// `[0, n-2]`): `lo=idx, hi=idx+1, t=frac` exactly matches the old
// i0c/i1c/fx-or-fy triple, so nothing changes for a collider whose SDF
// query points stay within the grid's own interior, only for one that
// runs past its edge.
function gradientAxisInfo( coord, n ) {

	const idx = int( floor( coord ) );
	const lo = max( 0, min( idx, n - 2 ) );
	const hi = min( lo.add( 1 ), n - 1 );
	const t = clamp( coord.sub( lo.toFloat() ), 0, 1 );

	return { lo, hi, t };

}

// Gradient of a scalar field at a continuous position: an analytic
// derivative of the same bilinear surface that collocatedValueAtPosition2
// samples (not a blend of the 4 corners' own discrete gradients like
// scalarGradientAtPosition2) -- cheaper (reads the 4 corner values only
// once) and matches exactly what collocatedValueAtPosition2 returns at that
// point. Use this when the gradient needs to match the sampled value
// precisely (e.g. an SDF's surface normal); use scalarGradientAtPosition2
// when a bit more smoothing across neighboring cells is desired.
//
// Deliberately does NOT reuse bilinearCoordsAndWeights2 the way the sibling
// gradient/sample functions in this file do -- see gradientAxisInfo's own
// header comment for why a gradient needs its own, non-collapsing index
// pair near the grid's edge, unlike a sampled value.
export function bilinearGradientAtPosition2( data, gridSpacing, dataOrigin, pos, shape ) {

	const [ nx, ny ] = shape;
	const gridPos = pos.sub( dataOrigin ).div( gridSpacing );

	const xi = gradientAxisInfo( gridPos.x, nx );
	const yi = gradientAxisInfo( gridPos.y, ny );

	const v00 = data( xi.lo, yi.lo );
	const v10 = data( xi.hi, yi.lo );
	const v01 = data( xi.lo, yi.hi );
	const v11 = data( xi.hi, yi.hi );

	const gx = float( 1 ).sub( yi.t ).mul( v10.sub( v00 ) ).add( yi.t.mul( v11.sub( v01 ) ) ).div( gridSpacing.x );
	const gy = float( 1 ).sub( xi.t ).mul( v01.sub( v00 ) ).add( xi.t.mul( v11.sub( v10 ) ) ).div( gridSpacing.y );

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
