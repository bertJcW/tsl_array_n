import { describe, it, expect } from 'vitest';
import { polygonSignedDistance, polygonsSignedDistance, polygonCentroid, translatePolygon, rotatePolygon } from '../src/grid/polygon_sdf.js';

const square = [ [ 0, 0 ], [ 4, 0 ], [ 4, 4 ], [ 0, 4 ] ]; // 4x4, centered at (2,2)

describe( 'polygonSignedDistance', () => {

	it( 'is negative inside, magnitude = distance to nearest edge', () => {

		expect( polygonSignedDistance( 2, 2, square ) ).toBeCloseTo( -2, 6 ); // center, 2 away from every edge

	} );

	it( 'is positive outside, magnitude = distance to nearest edge', () => {

		expect( polygonSignedDistance( 2, -1, square ) ).toBeCloseTo( 1, 6 ); // 1 below the bottom edge
		expect( polygonSignedDistance( 5, 2, square ) ).toBeCloseTo( 1, 6 ); // 1 outside the right edge

	} );

} );

describe( 'polygonsSignedDistance', () => {

	it( 'unions multiple polygons via pointwise min', () => {

		const squareB = [ [ 10, 0 ], [ 14, 0 ], [ 14, 4 ], [ 10, 4 ] ]; // a second square, not overlapping square

		// close to square's center, far from squareB -- the union SDF should pick square's (smaller distance)
		expect( polygonsSignedDistance( 2, 2, [ square, squareB ] ) ).toBeCloseTo( -2, 6 );

		// between the two squares, not particularly close to either: should pick whichever is nearer (squareB's left edge is closer)
		const midway = polygonsSignedDistance( 7, 2, [ square, squareB ] );
		expect( midway ).toBeCloseTo( 3, 6 ); // min(distance to square's right edge = 3, distance to squareB's left edge = 3) -- should still be 3 when the two are equal

	} );

} );

describe( 'polygonCentroid', () => {

	it( 'is the exact center for a symmetric square', () => {

		const [ cx, cy ] = polygonCentroid( square );
		expect( cx ).toBeCloseTo( 2, 6 );
		expect( cy ).toBeCloseTo( 2, 6 );

	} );

	it( 'matches the vertex average for a triangle (a known identity for triangles specifically)', () => {

		const triangle = [ [ 0, 0 ], [ 4, 0 ], [ 0, 3 ] ];
		const [ cx, cy ] = polygonCentroid( triangle );

		expect( cx ).toBeCloseTo( 4 / 3, 6 );
		expect( cy ).toBeCloseTo( 1, 6 );

	} );

} );

describe( 'translatePolygon / rotatePolygon', () => {

	it( 'translatePolygon shifts every vertex by the same offset', () => {

		const shifted = translatePolygon( square, 10, -5 );
		expect( shifted ).toEqual( [ [ 10, -5 ], [ 14, -5 ], [ 14, -1 ], [ 10, -1 ] ] );

	} );

	it( 'rotatePolygon rotates (1,0) by 90° around the origin to (0,1)', () => {

		const [ [ x, y ] ] = rotatePolygon( [ [ 1, 0 ] ], Math.PI / 2, [ 0, 0 ] );
		expect( x ).toBeCloseTo( 0, 6 );
		expect( y ).toBeCloseTo( 1, 6 );

	} );

} );
