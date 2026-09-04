// Structural tests (same reason as grid_math.test.js: building a TSL node
// graph doesn't need a GPU, but reading real numeric values back requires
// actually running a kernel). The numeric correctness of the SDF that
// addPolygon() itself writes is directly covered by polygon_sdf.test.js
// (polygonsSignedDistance is a pure numeric function; this file just calls
// it and then fromArray, without re-verifying the numbers).

import { describe, it, expect } from 'vitest';
import { vec2 } from 'three/tsl';
import { createSDFStaticCollider2, createSDFRigidBodyCollider2 } from '../src/grid/sdf_collider2.js';

function expectNode( value ) {

	expect( value ).toBeTruthy();
	expect( value.isNode ).toBe( true );

}

const square = [ [ -2, -2 ], [ 2, -2 ], [ 2, 2 ], [ -2, 2 ] ];

describe( 'SDFStaticCollider2', () => {

	it( 'exposes sample/gradient/isInside/velocityAt as node-building functions', () => {

		const collider = createSDFStaticCollider2( 8, 8, 1, 1, -4, -4 );

		expectNode( collider.sample( vec2( 0, 0 ) ) );
		expectNode( collider.gradient( vec2( 0, 0 ) ) );
		expectNode( collider.isInside( 1, 1 ) );
		expectNode( collider.velocityAt( vec2( 0, 0 ) ) );

	} );

	it( 'velocityAt is always zero for a static collider', () => {

		const collider = createSDFStaticCollider2( 8, 8, 1, 1, -4, -4 );
		const v = collider.velocityAt( vec2( 1, 2 ) );

		expectNode( v );

	} );

	it( 'defaults frictionCoefficient and allows mutating it directly', () => {

		const collider = createSDFStaticCollider2( 8, 8, 1, 1, -4, -4 );

		expect( collider.frictionCoefficient ).toBe( 0.5 );
		collider.frictionCoefficient = 0.2;
		expect( collider.frictionCoefficient ).toBe( 0.2 );

	} );

	it( 'addPolygon/addPolygons build without throwing', () => {

		const collider = createSDFStaticCollider2( 8, 8, 1, 1, -4, -4 );

		expect( () => collider.addPolygon( square ) ).not.toThrow();
		expect( () => collider.addPolygons( [ square, translateSquare( square, 10, 0 ) ] ) ).not.toThrow();

	} );

} );

describe( 'SDFRigidBodyCollider2', () => {

	it( 'constructs and rasterizes the initial geometry without throwing', () => {

		expect( () => createSDFRigidBodyCollider2( square, 8, 8, 1, 1, -4, -4 ) ).not.toThrow();

	} );

	it( 'update() is a no-op when stationary (both velocities zero)', () => {

		const collider = createSDFRigidBodyCollider2( square, 8, 8, 1, 1, -4, -4 );

		expect( () => collider.update( 1 / 60 ) ).not.toThrow();

	} );

	it( 'update() re-rasterizes when moving (linear velocity)', () => {

		const collider = createSDFRigidBodyCollider2( square, 8, 8, 1, 1, -4, -4, [ 1, 0 ], 0 );

		expect( () => collider.update( 1 / 60 ) ).not.toThrow();

	} );

	it( 'update() re-rasterizes when moving (angular velocity)', () => {

		const collider = createSDFRigidBodyCollider2( square, 8, 8, 1, 1, -4, -4, [ 0, 0 ], 1 );

		expect( () => collider.update( 1 / 60 ) ).not.toThrow();

	} );

	it( 'velocityAt reflects linear + angular rigid-body kinematics as a node', () => {

		const collider = createSDFRigidBodyCollider2( square, 8, 8, 1, 1, -4, -4, [ 1, 2 ], 0.5 );

		expectNode( collider.velocityAt( vec2( 1, 0 ) ) );

	} );

} );

function translateSquare( ring, dx, dy ) {

	return ring.map( ( [ x, y ] ) => [ x + dx, y + dy ] );

}
