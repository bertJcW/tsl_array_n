// 3D counterpart of sdf_collider2.js. Same factory shape and the same
// "the field is mutated, never replaced" invariant that makes a moving
// collider cheap (see that file's own header comment for the full
// measurement) -- only the geometry representation differs: primitive_sdf3.js's
// sphere/box shapes instead of polygon_sdf.js's polygons, for the reason
// that file's own header comment gives (no faithful 3D analogue of
// "polygon"/"SVG path" without taking on a full mesh-SDF system).

import * as tsl_array_n from 'tsl_array_n';
import { vec3 } from 'three/tsl';
import { createCellCenteredScalarGrid3 } from './grid_data3.js';
import { collocatedValueAtPosition3, trilinearGradientAtPosition3 } from './grid_math3.js';
import { primitivesDistanceCpu, shapesCentroid, translateShapes, rotateShapes } from './primitive_sdf3.js';
import { fractionInsideCpu } from './level_set_utils.js';

const DEFAULT_FRICTION = 0.5;

// Same floor jet/fluid-engine-dev's own buildWeights applies (kMinWeight in
// grid_fractional_single_phase_pressure_solver3.cpp) -- a face that is open
// by any nonzero amount still gets a small minimum conductance rather than
// the near-zero value the raw fraction can produce, which would otherwise
// make that row of the pressure system nearly singular.
const MIN_FACE_WEIGHT = 0.01;

export function createSDFStaticCollider3( resolutionX, resolutionY, resolutionZ, gridSpacingX, gridSpacingY, gridSpacingZ, originX, originY, originZ ) {

	const grid = createCellCenteredScalarGrid3( resolutionX, resolutionY, resolutionZ, gridSpacingX, gridSpacingY, gridSpacingZ, originX, originY, originZ );

	const originXCpu = originX + 0.5 * gridSpacingX;
	const originYCpu = originY + 0.5 * gridSpacingY;
	const originZCpu = originZ + 0.5 * gridSpacingZ;

	function sample( pos ) {

		return collocatedValueAtPosition3( grid.data, grid.gridSpacing, grid.dataOrigin, pos, grid.resolution );

	}

	// trilinearGradientAtPosition3, not scalarGradientAtPosition3 -- exactly
	// matches sample()'s own return value, see grid_math3.js's own header
	// comment (mirrors grid_math.js's bilinearGradientAtPosition2/
	// scalarGradientAtPosition2 distinction).
	function gradient( pos ) {

		return trilinearGradientAtPosition3( grid.data, grid.gridSpacing, grid.dataOrigin, pos, grid.resolution );

	}

	function isInside( i, j, k ) {

		return grid.data( i, j, k ).lessThan( 0 );

	}

	function velocityAt( /* point */ ) {

		return vec3( 0 );

	}

	let currentShapes = [];
	let currentInvert = false;

	// Rasterizes a set of primitive shapes into an SDF and writes it into
	// grid.data -- see sdf_collider2.js's own addPolygons for the
	// "mutate, don't replace" invariant this preserves, and for
	// options.invert (unchanged: default false, solid inside the shape(s);
	// true flips to solid outside, for an irregular container/basin wall).
	function addShapes( shapes, { invert = false } = {} ) {

		currentShapes = shapes;
		currentInvert = invert;

		const [ nx, ny, nz ] = grid.resolution;
		const hostSdf = new Float32Array( nx * ny * nz );

		for ( let k = 0; k < nz; k ++ ) {

			const z = originZCpu + k * gridSpacingZ;

			for ( let j = 0; j < ny; j ++ ) {

				const y = originYCpu + j * gridSpacingY;

				for ( let i = 0; i < nx; i ++ ) {

					const x = originXCpu + i * gridSpacingX;
					const d = primitivesDistanceCpu( shapes, x, y, z );
					hostSdf[ i + nx * j + nx * ny * k ] = invert ? - d : d;

				}

			}

		}

		grid.data.fromArray( hostSdf );

	}

	function addShape( shape, options ) {

		addShapes( [ shape ], options );

	}

	// Computes this STATIC collider's own per-face open-area fraction for a
	// velocity grid's U/V/W faces -- the fractional occupancy weight that
	// grid_pressure_solver3.js's own colliderWeights option bakes directly
	// into the pressure Poisson system, matching jet/fluid-engine-dev's own
	// GridFractionalSinglePhasePressureSolver3::buildWeights (see
	// grid_pressure_solver3.js's own header comment for why this project
	// needed it: the plain binary face-blocking this port used before could
	// not represent a partially-open face at all, which is what made the
	// pressure solve sensitive to collider shape/orientation in the first
	// place).
	//
	// Evaluates the SAME shapes (and invert flag) this collider was last
	// built from via primitivesDistanceCpu directly -- the analytic
	// distance, not grid.data's own rasterized-and-interpolated copy,
	// matching jet's own buildWeights sampling its analytic boundarySdf --
	// so resolutionX/Y/Z, gridSpacingX/Y/Z, originX/Y/Z here describe the
	// VELOCITY grid (grid_solver3.js's own), independent of this collider's
	// own SDF storage resolution; the two need not match.
	//
	// A static collider only needs this computed once, at setup time (see
	// grid_solver3.js's own call site) -- there is no moving-collider
	// counterpart yet (createSDFRigidBodyCollider3 inherits this method but
	// nothing recomputes it per frame; wiring that up is out of scope for
	// now, see the project notes this fix grew out of).
	function computeFaceWeights( resolutionX, resolutionY, resolutionZ, gridSpacingX, gridSpacingY, gridSpacingZ, originX, originY, originZ ) {

		const sign = currentInvert ? - 1 : 1;
		const phi = ( x, y, z ) => sign * primitivesDistanceCpu( currentShapes, x, y, z );

		// cornerOffsets returns the four corners of a face's own
		// cross-section, in the (bottomLeft, bottomRight, topLeft, topRight)
		// order fractionInsideCpu expects, as [dx,dy,dz] offsets from the
		// face centre -- see that function's own header comment for why the
		// two "axes held fixed" choice below (matching jet's own U/V/W
		// buildWeights: a face's cross-section spans the two axes it is NOT
		// staggered on).
		function faceWeightArray( dataSizeAxis, dataOriginAxis, cornerOffsets ) {

			const [ nx, ny, nz ] = dataSizeAxis;
			const [ ox, oy, oz ] = dataOriginAxis;
			const out = new Float32Array( nx * ny * nz );

			for ( let k = 0; k < nz; k ++ ) {

				const z = oz + k * gridSpacingZ;

				for ( let j = 0; j < ny; j ++ ) {

					const y = oy + j * gridSpacingY;

					for ( let i = 0; i < nx; i ++ ) {

						const x = ox + i * gridSpacingX;

						const corners = cornerOffsets.map( ( [ dx, dy, dz ] ) => phi( x + dx, y + dy, z + dz ) );
						const [ bl, br, tl, tr ] = corners;
						const frac = fractionInsideCpu( bl, br, tl, tr );

						let w = Math.min( 1, Math.max( 0, 1 - frac ) );

						// *** A face lying exactly ON the surface is closed, and
						// nothing else in this file would say so ***
						//
						// fractionInsideCpu inherits the library's usual "inside
						// means phi < 0" convention, so a face whose four corners
						// all read exactly zero comes back as zero fraction
						// inside, i.e. fully open. For any curved collider that
						// is a measure-zero case nobody ever hits. For a
						// grid-aligned box it is every face of the box: an
						// obstacle at integer coordinates puts its whole front
						// and back faces exactly on a row of velocity faces, and
						// this function was handing back weight 1 for all of
						// them -- 576 fully-open faces through the middle of a
						// solid, in examples/35-karman-vortex-street-3d/.
						//
						// The pressure system then believed the obstacle's
						// upstream face was open while
						// grid_blocked_boundary_condition_solver3.js's own
						// no-flux projection skipped it too (same convention, the
						// other half of this fix), so flow went straight into the
						// solid with nothing to stop it. Measured: the
						// right-hand-side divergence piled up in the cells just
						// upstream of that face, at up to 22 against 3.6 for the
						// rest of the domain, and the scene blew up.
						//
						// A sphere has only the handful of faces its surface
						// happens to pass exactly through, which is why the same
						// scene merely drifted with one and detonated with the
						// other -- and why that difference was previously put
						// down to linear face fractions on sharp geometry.
						if ( corners.every( ( value ) => value <= 0 ) ) w = 0;
						else if ( w > 0 && w < MIN_FACE_WEIGHT ) w = MIN_FACE_WEIGHT;

						out[ i + nx * j + nx * ny * k ] = w;

					}

				}

			}

			return out;

		}

		function upload( dataSizeAxis, hostArray ) {

			const arr = tsl_array_n.arrayN( 'float', dataSizeAxis );
			arr.fromArray( hostArray );
			return arr;

		}

		const hy = 0.5 * gridSpacingY, hz = 0.5 * gridSpacingZ, hx = 0.5 * gridSpacingX;

		const dataSizeU = [ resolutionX + 1, resolutionY, resolutionZ ];
		const dataSizeV = [ resolutionX, resolutionY + 1, resolutionZ ];
		const dataSizeW = [ resolutionX, resolutionY, resolutionZ + 1 ];

		const dataOriginU = [ originX, originY + hy, originZ + hz ];
		const dataOriginV = [ originX + hx, originY, originZ + hz ];
		const dataOriginW = [ originX + hx, originY + hy, originZ ];

		const uHost = faceWeightArray( dataSizeU, dataOriginU, [
			[ 0, - hy, - hz ], [ 0, hy, - hz ], [ 0, - hy, hz ], [ 0, hy, hz ]
		] );
		const vHost = faceWeightArray( dataSizeV, dataOriginV, [
			[ - hx, 0, - hz ], [ hx, 0, - hz ], [ - hx, 0, hz ], [ hx, 0, hz ]
		] );
		const wHost = faceWeightArray( dataSizeW, dataOriginW, [
			[ - hx, - hy, 0 ], [ hx, - hy, 0 ], [ - hx, hy, 0 ], [ hx, hy, 0 ]
		] );

		const u = upload( dataSizeU, uHost );
		const v = upload( dataSizeV, vHost );
		const w = upload( dataSizeW, wHost );

		return {
			u: ( i, j, k ) => u( i, j, k ),
			v: ( i, j, k ) => v( i, j, k ),
			w: ( i, j, k ) => w( i, j, k ),
			// The fields behind those accessors, for measurement only -- a
			// health check on the solver has to read the same weights the
			// solver used, and an accessor closure cannot be read back.
			fields: { u, v, w }
		};

	}

	return {
		grid,
		frictionCoefficient: DEFAULT_FRICTION,
		clear: grid.clear,
		sample, gradient, isInside, velocityAt,
		addShape, addShapes, computeFaceWeights
	};

}

// A moving rigid-body collider -- see sdf_collider2.js's own
// createSDFRigidBodyCollider2 header comment for the full "pose is live
// data, not a build-time constant" story and its measured cost (that
// investigation and fix are dimension-independent; nothing about it
// changes here). The one genuine 3D difference: angular velocity is a
// VECTOR (an axis of rotation scaled by rate, not a single signed number),
// and the rigid-body velocity formula is the real 3D cross product rather
// than 2D's "rotate 90 degrees and scale" shortcut.
//
// geometryShapes: the initial shape set, an array of primitive_sdf3.js
// shape descriptors. linearVelocityXYZ: a plain [vx,vy,vz] array.
// angularVelocityXYZ: a plain [wx,wy,wz] array, the rotation axis scaled
// by its rate (so |angularVelocityXYZ| is radians/sec and its direction is
// the axis, right-hand rule) -- both plain JS, not TSL nodes, the kinematic
// integration is a per-frame CPU accumulation the same as the 2D version's.
export function createSDFRigidBodyCollider3(
	geometryShapes,
	resolutionX, resolutionY, resolutionZ, gridSpacingX, gridSpacingY, gridSpacingZ, originX, originY, originZ,
	linearVelocityXYZ = [ 0, 0, 0 ], angularVelocityXYZ = [ 0, 0, 0 ]
) {

	const collider = createSDFStaticCollider3( resolutionX, resolutionY, resolutionZ, gridSpacingX, gridSpacingY, gridSpacingZ, originX, originY, originZ );

	const baseGeometry = geometryShapes;
	const linearVelocityNode = vec3( linearVelocityXYZ[ 0 ], linearVelocityXYZ[ 1 ], linearVelocityXYZ[ 2 ] );

	let currentPosition = shapesCentroid( geometryShapes );
	// Accumulated rotation from the base pose, tracked as (axis, angle)
	// rather than composed incrementally -- see primitive_sdf3.js's own
	// rotateShapes header comment on why composing two DIFFERENT axes isn't
	// supported there; accumulating a single running (axis, angle) here and
	// re-applying it fresh from baseGeometry each update() sidesteps that
	// entirely as long as angularVelocityXYZ's direction stays fixed over
	// the collider's lifetime, which every constructor argument here is.
	let currentAngle = 0;
	const rotationAxis = normalize3( angularVelocityXYZ );
	const angularSpeed = Math.hypot( ...angularVelocityXYZ );

	// Same "pose is live GPU data" fix as sdf_collider2.js's own -- see that
	// file's header comment for the measurement this is built to satisfy.
	const poseX = tsl_array_n.array0( 'float' );
	const poseY = tsl_array_n.array0( 'float' );
	const poseZ = tsl_array_n.array0( 'float' );
	const linVelX = tsl_array_n.array0( 'float' );
	const linVelY = tsl_array_n.array0( 'float' );
	const linVelZ = tsl_array_n.array0( 'float' );
	const angVelX = tsl_array_n.array0( 'float' );
	const angVelY = tsl_array_n.array0( 'float' );
	const angVelZ = tsl_array_n.array0( 'float' );

	function publishPose() {

		poseX.fromArray( new Float32Array( [ currentPosition[ 0 ] ] ) );
		poseY.fromArray( new Float32Array( [ currentPosition[ 1 ] ] ) );
		poseZ.fromArray( new Float32Array( [ currentPosition[ 2 ] ] ) );
		linVelX.fromArray( new Float32Array( [ linearVelocityXYZ[ 0 ] ] ) );
		linVelY.fromArray( new Float32Array( [ linearVelocityXYZ[ 1 ] ] ) );
		linVelZ.fromArray( new Float32Array( [ linearVelocityXYZ[ 2 ] ] ) );
		angVelX.fromArray( new Float32Array( [ angularVelocityXYZ[ 0 ] ] ) );
		angVelY.fromArray( new Float32Array( [ angularVelocityXYZ[ 1 ] ] ) );
		angVelZ.fromArray( new Float32Array( [ angularVelocityXYZ[ 2 ] ] ) );

	}

	publishPose();

	collider.addShapes( geometryShapes );

	function update( dt ) {

		const isStationary = linearVelocityXYZ.every( ( v ) => v === 0 ) && angularSpeed === 0;
		if ( isStationary ) return;

		currentPosition = [
			currentPosition[ 0 ] + linearVelocityXYZ[ 0 ] * dt,
			currentPosition[ 1 ] + linearVelocityXYZ[ 1 ] * dt,
			currentPosition[ 2 ] + linearVelocityXYZ[ 2 ] * dt
		];
		currentAngle += angularSpeed * dt;

		const centroid = shapesCentroid( baseGeometry );
		const translated = translateShapes(
			baseGeometry,
			currentPosition[ 0 ] - centroid[ 0 ], currentPosition[ 1 ] - centroid[ 1 ], currentPosition[ 2 ] - centroid[ 2 ]
		);
		const posed = angularSpeed > 0
			? rotateShapes( translated, rotationAxis, currentAngle, currentPosition )
			: translated;

		collider.addShapes( posed );
		publishPose();

	}

	// Rigid-body kinematics: v(point) = linearVelocity + angularVelocity x (point - currentPosition),
	// the real 3D cross product (2D's "rotate 90 degrees" shortcut was
	// itself just this cross product specialised to an angular velocity
	// confined to the z-axis).
	function velocityAt( point ) {

		const r = point.sub( vec3( poseX(), poseY(), poseZ() ) );
		const w = vec3( angVelX(), angVelY(), angVelZ() );
		const wCrossR = vec3(
			w.y.mul( r.z ).sub( w.z.mul( r.y ) ),
			w.z.mul( r.x ).sub( w.x.mul( r.z ) ),
			w.x.mul( r.y ).sub( w.y.mul( r.x ) )
		);

		return vec3( linVelX(), linVelY(), linVelZ() ).add( wCrossR );

	}

	return {
		...collider,
		baseGeometry,
		linearVelocity: linearVelocityNode,
		angularVelocity: angularVelocityXYZ,
		update,
		velocityAt
	};

}

function normalize3( v ) {

	const len = Math.hypot( v[ 0 ], v[ 1 ], v[ 2 ] );
	return len > 0 ? [ v[ 0 ] / len, v[ 1 ] / len, v[ 2 ] / len ] : [ 0, 0, 1 ];

}
