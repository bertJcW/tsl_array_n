// 结构性测试（跟 grid_math.test.js 一样的原因：TSL 节点图构建不需要 GPU，但读出
// 真实数值要实跑 kernel）。addPolygon() 本身写进的 SDF 数值正确性由
// polygon_sdf.test.js 直接覆盖（polygonsSignedDistance 是纯数字函数，这里只是
// 调用它再 fromArray，不重复验证一遍数值）。

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
