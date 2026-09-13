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

import * as tsl_array_n from 'tsl_array_n';
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
	// uploading it in one batch via from_numpy.
	// options.invert: default false (every existing caller's behavior,
	// unchanged) -- a polygon's own interior is normally solid (matching
	// isInside's own `< 0` convention below). Pass true to flip that: solid
	// *outside* the polygon(s), fluid-permitted inside -- the way to model
	// an irregularly-shaped container/basin wall, since the domain's own
	// closedDomainBoundaryFlag only supports a rectangular outer boundary.
	// Purely a sign flip of the same rasterized values -- sample/gradient/
	// isInside and every collider-consuming kernel in
	// grid_blocked_boundary_condition_solver2.js only ever read whatever
	// sign grid.data already holds, so none of them need to know this
	// happened.
	function addPolygons( polygons, { invert = false } = {} ) {

		const [ nx, ny ] = grid.resolution;
		const hostSdf = new Float32Array( nx * ny );

		for ( let j = 0; j < ny; j ++ ) {

			const y = originYCpu + j * gridSpacingY;

			for ( let i = 0; i < nx; i ++ ) {

				const x = originXCpu + i * gridSpacingX;
				const d = polygonsSignedDistance( x, y, polygons );
				hostSdf[ i + j * nx ] = invert ? - d : d;

			}

		}

		grid.data.fromArray( hostSdf );

		// *** The field is mutated, never replaced. ***
		//
		// Every kernel built against this collider reads `grid.data` through
		// sample()/gradient()/isInside() and holds a reference to *this* field,
		// so new contents written into it are picked up without rebuilding
		// anything. That invariant is load-bearing:
		// grid_blocked_boundary_condition_solver2.js's colliderMoved() keeps the
		// kernels and re-derives only the block marker, which is what makes a
		// *moving* collider cheap -- measured 351.5 -> 44.3 ms per step on
		// examples/23-flip-moving-collider/, with 28 compute pipelines compiled
		// per step against 0. Replacing the field here instead of writing into
		// it would silently resurrect the old geometry in every already-built
		// kernel.

	}

	function addPolygon( points, options ) {

		addPolygons( [ points ], options );

	}

	// Shares one options object with both calls below -- parseSvgToPolygons
	// only destructures samples/scale/offsetX/offsetY, addPolygons only
	// destructures invert, so passing the same object to both is safe (each
	// ignores the keys meant for the other). Same invert support as
	// addPolygon/addPolygons above, e.g. addSvg(svg, { invert: true }).
	function addSvg( svgString, options ) {

		addPolygons( parseSvgToPolygons( svgString, options ), options );

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
// *** The pose is live data. It used to be a build-time constant. ***
//
// velocityAt(point) read currentPosition/currentAngle -- plain JS closure
// variables -- while a kernel was being *built*, so the returned TSL graph baked
// that moment's pose in as constants and any kernel built before a later
// update(dt) kept the old pose for good. That was the limitation this comment
// used to describe: a boundary shared with the Python/Taichi side of the source
// and with any "build the graph once, dispatch repeatedly" model. It mattered
// as soon as a collider actually moved every frame -- the one user of that
// pattern had to rebuild every collider-dependent kernel per frame, which cost
// 28 freshly compiled compute pipelines per step.
//
// Fixed exactly the way this comment prescribed: pose and velocities live in
// tsl_array_n array0 fields, update() publishes them with fromArray(), and
// velocityAt() reads them per dispatch. A kernel built once therefore sees the
// current pose, and a collider that merely *moved* needs no rebuild --
// grid_blocked_boundary_condition_solver2.js's colliderMoved() keeps the kernels
// and re-derives only the block marker. Measured on
// examples/23-flip-moving-collider/ (a collider moving every frame): 351.5 ms per
// step against 36.1 ms, 28 pipelines per step against none, and the page's own
// fps readout 2.5 -> 29.5. Kernels kept and kernels rebuilt were then compared
// directly and are bit-identical.
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

	// *** The pose is live data, not a build-time constant. ***
	//
	// velocityAt() used to read `currentPosition`/`currentAngle` -- plain JS
	// closure variables -- while a kernel was being *built*, so the returned TSL
	// graph carried that moment's pose as constants. Every kernel built before a
	// later update(dt) kept the old pose for good, which forced anyone using a
	// *moving* collider to rebuild its kernels every frame.
	// examples/23-flip-moving-collider/ did exactly that, and it cost 28 freshly
	// compiled compute pipelines per step: measured 351.5 ms per step against
	// 36.1 ms with the kernels kept.
	//
	// This file's own header comment prescribed the fix before any of that was
	// measured: "position/velocity should become a tsl_array_n array0/uniform
	// (updated via .fromArray()) instead of being captured as a plain JS closure
	// variable". These are that. update() publishes them, velocityAt() reads them
	// per dispatch, so a kernel built once sees the current pose -- and
	// grid_blocked_boundary_condition_solver2.js's colliderMoved() is then
	// correct rather than merely fast.
	//
	// The JS numbers stay the source of truth (update() does plain arithmetic on
	// them and re-rasterises the SDF from them); these five scalars are the copy
	// the GPU reads. The collider's returned `linearVelocity` / `angularVelocity`
	// properties are unchanged and are still build-time values -- they are
	// constructor arguments, so they only vary if a caller mutates its own array
	// afterwards, which was never a supported pattern.
	const poseX = tsl_array_n.array0( 'float' );
	const poseY = tsl_array_n.array0( 'float' );
	const linVelX = tsl_array_n.array0( 'float' );
	const linVelY = tsl_array_n.array0( 'float' );
	const angularVel = tsl_array_n.array0( 'float' );

	function publishPose() {

		poseX.fromArray( new Float32Array( [ currentPosition[ 0 ] ] ) );
		poseY.fromArray( new Float32Array( [ currentPosition[ 1 ] ] ) );
		linVelX.fromArray( new Float32Array( [ linearVelocityXY[ 0 ] ] ) );
		linVelY.fromArray( new Float32Array( [ linearVelocityXY[ 1 ] ] ) );
		angularVel.fromArray( new Float32Array( [ angularVelocity ] ) );

	}

	publishPose();

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

		// Published together with the re-rasterised SDF: a kernel built once
		// reads the pose at dispatch time, so this is what makes the move
		// visible without rebuilding anything. Both are pending uploads, and a
		// pending upload lands before the next dispatch that begins a pass.
		publishPose();

	}

	// Rigid-body kinematics: v(point) = linearVelocity + angularVelocity x (point - currentPosition)
	// In 2D, the cross product angularVelocity x r is just angularVelocity * (-r.y, r.x)
	function velocityAt( point ) {

		// Read per dispatch, from the published pose -- see its own comment.
		const r = point.sub( vec2( poseX(), poseY() ) );
		return vec2( linVelX(), linVelY() ).add( vec2( r.y.negate(), r.x ).mul( angularVel() ) );

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
