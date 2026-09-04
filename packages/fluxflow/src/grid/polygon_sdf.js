// Pure CPU-side geometry, doesn't touch any TSL node -- used by
// sdf_collider2.js's addPolygon()/addSvg() to rasterize polygon vertex
// arrays into SDF grid data.
//
// Corresponds to sdf_collider2.py's use of shapely (boundary.distance +
// contains/touches for inside/outside, unary_union to merge multiple
// shapes), but without depending on shapely or any JS geometry library --
// these are well-established small algorithms, cheaper to hand-write:
//   - point inside a polygon: the standard ray-casting point-in-polygon test
//   - shortest distance from a point to a polygon's boundary: distance to
//     each edge segment, take the minimum -- this corresponds directly to
//     shapely's boundary.distance() (distance to the boundary polyline, not
//     to the "filled" area, which is also why the shapely version has to use
//     boundary.distance rather than geom.distance)
//   - "union" SDF of multiple polygons: pointwise min -- doesn't need a real
//     polygon boolean operation at all; an SDF union is mathematically just
//     a pointwise min, the standard graphics technique, and much simpler and
//     more robust than hand-rolling polygon boolean ops
//
// Polygons are plain vertex arrays: [[x0,y0],[x1,y1],...] (implicitly
// closed -- an edge is automatically formed between the last point and the
// first, no need to repeat the first point).

function pointInPolygon( x, y, ring ) {

	let inside = false;

	for ( let i = 0, j = ring.length - 1; i < ring.length; j = i ++ ) {

		const [ xi, yi ] = ring[ i ];
		const [ xj, yj ] = ring[ j ];

		const intersect = ( ( yi > y ) !== ( yj > y ) ) &&
			( x < ( xj - xi ) * ( y - yi ) / ( yj - yi ) + xi );

		if ( intersect ) inside = ! inside;

	}

	return inside;

}

function pointToSegmentDistance( x, y, x1, y1, x2, y2 ) {

	const dx = x2 - x1;
	const dy = y2 - y1;
	const lengthSq = dx * dx + dy * dy;

	let t = lengthSq === 0 ? 0 : ( ( x - x1 ) * dx + ( y - y1 ) * dy ) / lengthSq;
	t = Math.max( 0, Math.min( 1, t ) );

	const px = x1 + t * dx;
	const py = y1 + t * dy;

	return Math.hypot( x - px, y - py );

}

function pointToRingBoundaryDistance( x, y, ring ) {

	let minDist = Infinity;

	for ( let i = 0, j = ring.length - 1; i < ring.length; j = i ++ ) {

		const d = pointToSegmentDistance( x, y, ring[ j ][ 0 ], ring[ j ][ 1 ], ring[ i ][ 0 ], ring[ i ][ 1 ] );
		if ( d < minDist ) minDist = d;

	}

	return minDist;

}

// Signed distance of a single polygon at (x,y): negative inside, positive
// outside (matches the source's host_sdf sign convention)
export function polygonSignedDistance( x, y, ring ) {

	const dist = pointToRingBoundaryDistance( x, y, ring );
	return pointInPolygon( x, y, ring ) ? - dist : dist;

}

// Union SDF of multiple polygons: pointwise min
export function polygonsSignedDistance( x, y, rings ) {

	let minDist = Infinity;

	for ( const ring of rings ) {

		const d = polygonSignedDistance( x, y, ring );
		if ( d < minDist ) minDist = d;

	}

	return minDist;

}

// Area-weighted centroid of a polygon (the standard formula, not a plain
// average of vertex coordinates -- matches shapely's .centroid definition;
// SDFRigidBodyCollider2 uses this as its rotation pivot)
export function polygonCentroid( ring ) {

	let area = 0, cx = 0, cy = 0;

	for ( let i = 0, j = ring.length - 1; i < ring.length; j = i ++ ) {

		const [ xi, yi ] = ring[ i ];
		const [ xj, yj ] = ring[ j ];

		const cross = xj * yi - xi * yj;
		area += cross;
		cx += ( xj + xi ) * cross;
		cy += ( yj + yi ) * cross;

	}

	area *= 0.5;
	const scale = 1 / ( 6 * area );

	return [ cx * scale, cy * scale ];

}

export function translatePolygon( ring, dx, dy ) {

	return ring.map( ( [ x, y ] ) => [ x + dx, y + dy ] );

}

export function rotatePolygon( ring, angle, origin ) {

	const cos = Math.cos( angle );
	const sin = Math.sin( angle );
	const [ ox, oy ] = origin;

	return ring.map( ( [ x, y ] ) => {

		const dx = x - ox;
		const dy = y - oy;

		return [ ox + dx * cos - dy * sin, oy + dx * sin + dy * cos ];

	} );

}
