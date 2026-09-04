// Ported from sdf_collider2.py. Factory-function style, consistent with the
// rest of this port.
//
// The replacement for addShapelyGeometry/addSvg (no shapely/svg.path, zero
// new dependencies) is described in polygon_sdf.js's / svg_utils.js's header
// comments -- polygon geometry uses hand-rolled point-in-polygon + distance
// to boundary, multiple shapes are combined via SDF pointwise min, and SVG
// parsing uses the browser's native SVGPathElement API. addPolygon()/
// addPolygons()/addSvg() are new names here (not addShapelyGeometry, since
// there's no shapely or any "geometry object" involved at all -- just plain
// vertex arrays).

import { vec2 } from 'three/tsl';
import { createCellCenteredScalarGrid2 } from './grid_data2.js';
import { collocatedValueAtPosition2, bilinearGradientAtPosition2 } from './grid_math.js';
import { polygonsSignedDistance, polygonCentroid, translatePolygon, rotatePolygon } from './polygon_sdf.js';
import { parseSvgToPolygons } from './svg_utils.js';

const DEFAULT_FRICTION = 0.5;

export function createSDFStaticCollider2( resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY ) {

	const grid = createCellCenteredScalarGrid2( resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY );

	// CPU-side origin of the cell-centered grid -- must numerically match
	// grid.dataOrigin (a TSL node), but a plain-JS context (addPolygons'
	// per-cell CPU loop) can't conveniently read a node's value, so it's
	// computed independently here
	const originXCpu = originX + 0.5 * gridSpacingX;
	const originYCpu = originY + 0.5 * gridSpacingY;

	// Reuses grid_math's general bilinear sampling directly inside kernels,
	// instead of duplicating the interpolation logic
	function sample( pos ) {

		return collocatedValueAtPosition2( grid.data, grid.gridSpacing, grid.dataOrigin, pos, grid.resolution );

	}

	// Gradient of the SDF at an arbitrary continuous position, used to get a
	// normal vector for no-flux projection; uses bilinearGradientAtPosition2
	// (an analytic derivative of the bilinear interpolation formula, exactly
	// matching sample()'s return value), not scalarGradientAtPosition2,
	// which additionally blends across neighboring cells
	function gradient( pos ) {

		return bilinearGradientAtPosition2( grid.data, grid.gridSpacing, grid.dataOrigin, pos, grid.resolution );

	}

	// Grid-index version of the inside check, used for markers (sdf<0 counts
	// as inside the collider)
	function isInside( i, j ) {

		return grid.data( i, j ).lessThan( 0 );

	}

	// A static collider's velocity is always zero
	function velocityAt( /* point */ ) {

		return vec2( 0 );

	}

	// Rasterizes a set of polygons ([[x,y],...] vertex arrays) into an SDF
	// and writes it into grid.data -- corresponds to the source's
	// addShapelyGeometry computing the distance per-cell on the CPU side and
	// uploading it in one batch via from_numpy
	function addPolygons( polygons ) {

		const [ nx, ny ] = grid.resolution;
		const hostSdf = new Float32Array( nx * ny );

		for ( let j = 0; j < ny; j ++ ) {

			const y = originYCpu + j * gridSpacingY;

			for ( let i = 0; i < nx; i ++ ) {

				const x = originXCpu + i * gridSpacingX;
				hostSdf[ i + j * nx ] = polygonsSignedDistance( x, y, polygons );

			}

		}

		grid.data.fromArray( hostSdf );

	}

	function addPolygon( points ) {

		addPolygons( [ points ] );

	}

	function addSvg( svgString, options ) {

		addPolygons( parseSvgToPolygons( svgString, options ) );

	}

	return {
		grid,
		frictionCoefficient: DEFAULT_FRICTION, // plain mutable property, just do collider.frictionCoefficient = x
		clear: grid.clear,
		sample, gradient, isInside, velocityAt,
		addPolygon, addPolygons, addSvg
	};

}

// A moving rigid-body collider: shares the grid/sample/gradient/isInside
// from SDFStaticCollider2, but computes velocityAt from rigid-body
// kinematics and provides update(dt) -- geometry only gets re-posed and the
// SDF only gets re-rasterized when it's actually moving (nonzero linear or
// angular velocity).
//
// geometryPolygon: the initial shape, a [[x,y],...] vertex array
// (corresponds to the source's shapely geometry argument). linearVelocityXY:
// a plain [vx,vy] array, not a TSL node -- the kinematic integration
// (updating currentPosition/currentAngle) is a plain per-frame CPU
// accumulation, where a node would serve no purpose.
//
// A known architectural limitation (not a new problem introduced by this
// port -- a boundary shared by both Taichi's and TSL's "build the graph
// once, dispatch repeatedly" execution model): if some future kernel (e.g.
// grid_blocked_boundary_condition_solver2.js's _markAndProjectU or similar)
// calls this velocityAt(point) while it's being built, the resulting TSL
// node graph bakes in whatever currentPosition/currentAngle/linearVelocity
// were at that moment as constants -- a later update(dt) that changes those
// values will not be picked up by a kernel that's already been built. If
// collider motion that genuinely changes every frame (without rebuilding
// the kernel) is ever actually needed, position/velocity should become a
// tsl_array_n array0/uniform (updated via .fromArray()/.value=) instead of
// being captured as a plain JS closure variable the way it is now -- the
// current implementation is a faithful port of this same limitation that
// exists on both the Python/Taichi side of the source, not a new defect,
// but it's the piece that would need redesigning if a genuinely
// frame-by-frame moving collider is built on top of this later.
export function createSDFRigidBodyCollider2(
	geometryPolygon,
	resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY,
	linearVelocityXY = [ 0, 0 ], angularVelocity = 0
) {

	const collider = createSDFStaticCollider2( resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY );

	const baseGeometry = geometryPolygon;
	const linearVelocityNode = vec2( linearVelocityXY[ 0 ], linearVelocityXY[ 1 ] );

	let currentPosition = polygonCentroid( geometryPolygon );
	let currentAngle = 0;

	collider.addPolygon( geometryPolygon );

	function update( dt ) {

		// tm.vec2's == is a component-wise comparison, which in Taichi is
		// always truthy as a whole and can't be used directly as an overall
		// equality check -- here it's just plain JS number comparison, so
		// that particular pitfall doesn't apply, but the same
		// "skip when stationary" optimization is kept, to avoid
		// re-rasterizing a collider that hasn't moved every frame
		const isStationary = linearVelocityXY[ 0 ] === 0 && linearVelocityXY[ 1 ] === 0 && angularVelocity === 0;
		if ( isStationary ) return;

		currentPosition = [
			currentPosition[ 0 ] + linearVelocityXY[ 0 ] * dt,
			currentPosition[ 1 ] + linearVelocityXY[ 1 ] * dt
		];
		currentAngle += angularVelocity * dt;

		const centroid = polygonCentroid( baseGeometry );
		const translated = translatePolygon( baseGeometry, currentPosition[ 0 ] - centroid[ 0 ], currentPosition[ 1 ] - centroid[ 1 ] );
		const posed = rotatePolygon( translated, currentAngle, currentPosition );

		collider.addPolygon( posed );

	}

	// Rigid-body kinematics: v(point) = linearVelocity + angularVelocity x (point - currentPosition)
	// In 2D, the cross product angularVelocity x r is just angularVelocity * (-r.y, r.x)
	function velocityAt( point ) {

		const r = point.sub( vec2( currentPosition[ 0 ], currentPosition[ 1 ] ) );
		return linearVelocityNode.add( vec2( r.y.negate(), r.x ).mul( angularVelocity ) );

	}

	return {
		...collider,
		baseGeometry,
		linearVelocity: linearVelocityNode,
		angularVelocity,
		update,
		velocityAt
	};

}
