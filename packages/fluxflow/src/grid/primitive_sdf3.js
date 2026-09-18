// 3D collider geometry -- the counterpart of polygon_sdf.js/svg_utils.js,
// but not a port of either: "polygon" and "SVG path" are inherently 2D
// concepts with no direct 3D analogue (a full mesh-SDF system would be the
// faithful extension, and is real, separate scope this file deliberately
// does not take on). Instead this provides the two primitives that cover
// the large majority of what a 3D collider actually needs -- a sphere and
// an oriented box -- combined the same way polygon_sdf.js already combines
// multiple polygons: pointwise min of signed distance (CSG union). Each
// primitive is a plain JS object describing its own shape/pose (not a TSL
// node), and distance()/normal() below build the TSL expression for a
// given (already-node) query position -- same "plain JS description in,
// TSL graph out" shape sdf_collider3.js's own addPolygon(s) convention
// uses.
//
// Signed-distance convention matches polygon_sdf.js exactly: negative
// inside the shape, positive outside, zero at the surface.

import { vec3, length, max as tslMax, min as tslMin, abs } from 'three/tsl';

// A sphere: { type: 'sphere', center: [x,y,z], radius }.
export function sphere( center, radius ) {

	return { type: 'sphere', center, radius };

}

// An axis-aligned-at-rest box: { type: 'box', center: [x,y,z],
// halfExtents: [hx,hy,hz], rotation: { axis: [x,y,z], angle } }. rotation
// is optional (defaults to identity) -- a caller only pays for the extra
// trig/matrix work when the box is actually posed.
//
// *** A real, confirmed-on-real-hardware limitation: a box collider's
// own sharp edges/corners can destabilize the pressure solve, especially
// for an elongated shape with a lot of edge-adjacent surface area. ***
//
// grid_blocked_boundary_condition_solver3.js's own face-marking step
// (makeMarkAndProject) estimates how much of a velocity face lies inside
// a collider via level_set_utils.js's own fractionInsideSdf -- a linear
// interpolation between two SDF samples straddling that face (a
// standard technique, Bridson's own *Fluid Simulation for Computer
// Graphics*), which assumes the SDF varies smoothly between those two
// points. A sphere's SDF does. This box() primitive's own -- correct,
// textbook, Inigo-Quilez-standard -- SDF formula does NOT: its gradient
// is genuinely discontinuous at every edge and corner, by construction,
// and there is no formula fix for that (it is inherent to a box's own
// geometry, not a bug in this function). examples/35-karman-vortex-
// street-3d/'s own header comment records the full investigation: a
// box-cornered rod obstacle caused persistent pressure-solve instability
// (the flow got stuck pinned at its own velocity clamp, not merely slow
// to converge -- confirmed by testing far more CG iterations, which made
// it fail *faster*, not slower), while an otherwise-identical sphere
// recovered from its own transient spikes into bounded, decaying
// behavior every time. A caller wanting a smooth, round obstacle (a
// cylinder, in particular) should use sphere() or approximate the
// desired shape with a many-sided union of primitives whose own corner
// angles stay shallow (the way examples/16-karman-vortex-street/'s own
// 2D reference approximates a circle with a 48-sided polygon, not a
// literal square) -- a single sharp-edged box is the one shape this
// collider mechanism is not currently robust against.
export function box( center, halfExtents, rotation = null ) {

	return { type: 'box', center, halfExtents, rotation };

}

// Rodrigues' rotation formula, applied to a TSL vec3 node -- rotates `v`
// by `angle` radians about `axis` (assumed already unit-length; every
// caller below normalises once in plain JS at shape-construction time
// rather than per query). Used to bring a query point into a rotated
// primitive's own local (unrotated) frame, so the inverse rotation
// (negate the angle) is what callers below actually apply.
function rotateAroundAxis( v, axis, angle ) {

	const cosA = Math.cos( angle );
	const sinA = Math.sin( angle );
	const k = vec3( axis[ 0 ], axis[ 1 ], axis[ 2 ] );
	const kCrossV = vec3(
		k.y.mul( v.z ).sub( k.z.mul( v.y ) ),
		k.z.mul( v.x ).sub( k.x.mul( v.z ) ),
		k.x.mul( v.y ).sub( k.y.mul( v.x ) )
	);
	const kDotV = k.x.mul( v.x ).add( k.y.mul( v.y ) ).add( k.z.mul( v.z ) );

	return v.mul( cosA )
		.add( kCrossV.mul( sinA ) )
		.add( k.mul( kDotV ).mul( 1 - cosA ) );

}

function normalizeAxis( axis ) {

	const len = Math.hypot( axis[ 0 ], axis[ 1 ], axis[ 2 ] );
	return len > 0 ? [ axis[ 0 ] / len, axis[ 1 ] / len, axis[ 2 ] / len ] : [ 0, 0, 1 ];

}

// Signed distance from a single primitive to a TSL vec3 query position.
export function primitiveDistance( shape, pos ) {

	const center = vec3( shape.center[ 0 ], shape.center[ 1 ], shape.center[ 2 ] );
	const local = pos.sub( center );

	if ( shape.type === 'sphere' ) {

		return length( local ).sub( shape.radius );

	}

	if ( shape.type === 'box' ) {

		const q = shape.rotation
			? rotateAroundAxis( local, normalizeAxis( shape.rotation.axis ), - shape.rotation.angle )
			: local;

		const b = vec3( shape.halfExtents[ 0 ], shape.halfExtents[ 1 ], shape.halfExtents[ 2 ] );
		const d = abs( q ).sub( b );
		const outside = length( tslMax( d, vec3( 0 ) ) );
		const inside = tslMin( tslMax( d.x, tslMax( d.y, d.z ) ), 0 );

		return outside.add( inside );

	}

	throw new Error( `primitive_sdf3: unknown shape type '${ shape.type }'.` );

}

// Union (pointwise min of signed distance) of several primitives -- same
// CSG combinator polygon_sdf.js's own polygonsSignedDistance uses for
// multiple polygons.
export function primitivesDistance( shapes, pos ) {

	if ( shapes.length === 0 ) throw new Error( 'primitive_sdf3: at least one shape is required.' );

	let d = primitiveDistance( shapes[ 0 ], pos );

	for ( let i = 1; i < shapes.length; i ++ ) d = tslMin( d, primitiveDistance( shapes[ i ], pos ) );

	return d;

}

// Plain-JS (not TSL) signed distance, for the CPU-side per-cell
// rasterisation loop sdf_collider3.js's own addShapes() runs -- mirrors
// primitiveDistance/primitivesDistance's node-graph versions exactly, in
// plain number arithmetic, the same "one CPU copy, one GPU copy of the
// same formula" split polygon_sdf.js's own polygonsSignedDistance/
// pointInPolygon pair already uses.
function rotateAroundAxisCpu( v, axis, angle ) {

	const cosA = Math.cos( angle );
	const sinA = Math.sin( angle );
	const [ kx, ky, kz ] = axis;
	const kCrossV = [
		ky * v[ 2 ] - kz * v[ 1 ],
		kz * v[ 0 ] - kx * v[ 2 ],
		kx * v[ 1 ] - ky * v[ 0 ]
	];
	const kDotV = kx * v[ 0 ] + ky * v[ 1 ] + kz * v[ 2 ];

	return [
		v[ 0 ] * cosA + kCrossV[ 0 ] * sinA + kx * kDotV * ( 1 - cosA ),
		v[ 1 ] * cosA + kCrossV[ 1 ] * sinA + ky * kDotV * ( 1 - cosA ),
		v[ 2 ] * cosA + kCrossV[ 2 ] * sinA + kz * kDotV * ( 1 - cosA )
	];

}

function primitiveDistanceCpu( shape, x, y, z ) {

	const lx = x - shape.center[ 0 ];
	const ly = y - shape.center[ 1 ];
	const lz = z - shape.center[ 2 ];

	if ( shape.type === 'sphere' ) {

		return Math.hypot( lx, ly, lz ) - shape.radius;

	}

	if ( shape.type === 'box' ) {

		const [ qx, qy, qz ] = shape.rotation
			? rotateAroundAxisCpu( [ lx, ly, lz ], normalizeAxis( shape.rotation.axis ), - shape.rotation.angle )
			: [ lx, ly, lz ];

		const dx = Math.abs( qx ) - shape.halfExtents[ 0 ];
		const dy = Math.abs( qy ) - shape.halfExtents[ 1 ];
		const dz = Math.abs( qz ) - shape.halfExtents[ 2 ];

		const outside = Math.hypot( Math.max( dx, 0 ), Math.max( dy, 0 ), Math.max( dz, 0 ) );
		const inside = Math.min( Math.max( dx, Math.max( dy, dz ) ), 0 );

		return outside + inside;

	}

	throw new Error( `primitive_sdf3: unknown shape type '${ shape.type }'.` );

}

export function primitivesDistanceCpu( shapes, x, y, z ) {

	let d = Infinity;
	for ( const shape of shapes ) d = Math.min( d, primitiveDistanceCpu( shape, x, y, z ) );
	return d;

}

// Centroid of a set of shapes, weighted by an approximate "size" per shape
// (radius^3 for a sphere, the half-extents' product for a box) -- used the
// same way polygon_sdf.js's polygonCentroid is: as the pivot rigid-body
// rotation happens about. Single-shape colliders (the common case) reduce
// to exactly that shape's own centre.
export function shapesCentroid( shapes ) {

	let sx = 0, sy = 0, sz = 0, total = 0;

	for ( const shape of shapes ) {

		const weight = shape.type === 'sphere'
			? shape.radius ** 3
			: shape.halfExtents[ 0 ] * shape.halfExtents[ 1 ] * shape.halfExtents[ 2 ];

		sx += shape.center[ 0 ] * weight;
		sy += shape.center[ 1 ] * weight;
		sz += shape.center[ 2 ] * weight;
		total += weight;

	}

	return total > 0 ? [ sx / total, sy / total, sz / total ] : [ 0, 0, 0 ];

}

// Translates every shape in a set by a fixed offset -- plain-JS, returns
// new shape descriptors (shapes are immutable data, same convention
// polygon_sdf.js's translatePolygon/rotatePolygon use).
export function translateShapes( shapes, dx, dy, dz ) {

	return shapes.map( ( shape ) => ( {
		...shape,
		center: [ shape.center[ 0 ] + dx, shape.center[ 1 ] + dy, shape.center[ 2 ] + dz ]
	} ) );

}

// Rotates every shape's centre about `pivot` by (axis, angle), and composes
// that same rotation onto each box's own local orientation (a sphere is
// rotationally symmetric, so only its centre moves).
export function rotateShapes( shapes, axis, angle, pivot ) {

	const unitAxis = normalizeAxis( axis );

	return shapes.map( ( shape ) => {

		const rel = [ shape.center[ 0 ] - pivot[ 0 ], shape.center[ 1 ] - pivot[ 1 ], shape.center[ 2 ] - pivot[ 2 ] ];
		const rotatedRel = rotateAroundAxisCpu( rel, unitAxis, angle );
		const center = [ pivot[ 0 ] + rotatedRel[ 0 ], pivot[ 1 ] + rotatedRel[ 1 ], pivot[ 2 ] + rotatedRel[ 2 ] ];

		if ( shape.type === 'sphere' ) return { ...shape, center };

		// Composing two axis-angle rotations into a single equivalent one is
		// a real (if standard, via quaternion multiplication) derivation this
		// file does not take on -- box colliders that are both rotated at
		// construction AND spun about a different axis afterwards are out of
		// scope for now. The common case (posed once at construction, or spun
		// about its own already-applied axis) works: composing two rotations
		// that share the same axis stays that axis with the angles added.
		const existing = shape.rotation;
		const sameAxis = existing && unitAxis.every( ( v, i ) => Math.abs( v - normalizeAxis( existing.axis )[ i ] ) < 1e-9 );

		const rotation = existing === null || existing === undefined
			? { axis: unitAxis, angle }
			: sameAxis
				? { axis: existing.axis, angle: existing.angle + angle }
				: ( () => { throw new Error( 'primitive_sdf3: composing rotations about two different axes is not supported -- rebuild the box with its final orientation instead.' ); } )();

		return { ...shape, center, rotation };

	} );

}
