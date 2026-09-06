// Structural tests only (same reason as sdf_collider2.test.js: building a
// TSL node graph doesn't need a GPU). The underlying SDF numerics are
// already covered by polygon_sdf.test.js/sdf_collider2.test.js -- these
// tests just confirm the inflow/outflow-specific additions on top.

import { describe, it, expect } from 'vitest';
import { vec2, float } from 'three/tsl';
import {
	createSDFInflow2,
	createSDFOutflow2,
	createOutflowPressureDirichlet2,
	combineDirichlet
} from '../src/grid/sdf_inflow_outflow2.js';

function expectNode( value ) {

	expect( value ).toBeTruthy();
	expect( value.isNode ).toBe( true );

}

describe( 'createSDFInflow2', () => {

	it( 'exposes the same SDF interface as a collider', () => {

		const inflow = createSDFInflow2( 8, 8, 1, 1, -4, -4 );

		expect( typeof inflow.sample ).toBe( 'function' );
		expect( typeof inflow.gradient ).toBe( 'function' );
		expect( typeof inflow.isInside ).toBe( 'function' );
		expect( typeof inflow.addPolygon ).toBe( 'function' );

	} );

	it( 'defaults velocity to zero and mode to "set"', () => {

		const inflow = createSDFInflow2( 8, 8, 1, 1, -4, -4 );

		expectNode( inflow.velocity );
		expect( inflow.mode ).toBe( 'set' );

	} );

	it( 'accepts a plain [vx,vy] velocity array, baked to a node', () => {

		const inflow = createSDFInflow2( 8, 8, 1, 1, -4, -4, { velocity: [ 3, 0 ] } );

		expectNode( inflow.velocity );

	} );

	it( 'accepts a live vec2 node as velocity', () => {

		const liveVelocity = vec2( float( 1 ), float( 2 ) );
		const inflow = createSDFInflow2( 8, 8, 1, 1, -4, -4, { velocity: liveVelocity } );

		expect( inflow.velocity ).toBe( liveVelocity );

	} );

	it( 'accepts an explicit mode option', () => {

		const inflow = createSDFInflow2( 8, 8, 1, 1, -4, -4, { mode: 'add' } );

		expect( inflow.mode ).toBe( 'add' );

	} );

	it( 'mode is a plain mutable property', () => {

		const inflow = createSDFInflow2( 8, 8, 1, 1, -4, -4 );

		inflow.mode = 'add';
		expect( inflow.mode ).toBe( 'add' );

	} );

} );

describe( 'createSDFOutflow2', () => {

	it( 'exposes the same SDF interface as a collider', () => {

		const outflow = createSDFOutflow2( 8, 8, 1, 1, -4, -4 );

		expect( typeof outflow.sample ).toBe( 'function' );
		expect( typeof outflow.gradient ).toBe( 'function' );
		expect( typeof outflow.isInside ).toBe( 'function' );
		expect( typeof outflow.addPolygon ).toBe( 'function' );

	} );

} );

describe( 'createOutflowPressureDirichlet2', () => {

	it( 'returns a dirichlet(pos) function', () => {

		const outflow = createSDFOutflow2( 8, 8, 1, 1, -4, -4 );
		const dirichlet = createOutflowPressureDirichlet2( outflow );

		expect( typeof dirichlet ).toBe( 'function' );

	} );

	it( 'the returned {active, target} are real nodes', () => {

		const outflow = createSDFOutflow2( 8, 8, 1, 1, -4, -4 );
		const dirichlet = createOutflowPressureDirichlet2( outflow );
		const { active, target } = dirichlet( vec2( 0, 0 ) );

		expectNode( active );
		expectNode( target );

	} );

	it( 'works with a single outflow object or an array of them', () => {

		const a = createSDFOutflow2( 8, 8, 1, 1, -4, -4 );
		const b = createSDFOutflow2( 8, 8, 1, 1, -4, -4 );

		expect( () => createOutflowPressureDirichlet2( a )( vec2( 0, 0 ) ) ).not.toThrow();
		expect( () => createOutflowPressureDirichlet2( [ a, b ] )( vec2( 0, 0 ) ) ).not.toThrow();

	} );

	it( 'accepts a custom target value', () => {

		const outflow = createSDFOutflow2( 8, 8, 1, 1, -4, -4 );
		const dirichlet = createOutflowPressureDirichlet2( outflow, { target: 42 } );

		expectNode( dirichlet( vec2( 0, 0 ) ).target );

	} );

} );

describe( 'combineDirichlet', () => {

	it( 'returns the other function unchanged when one side is omitted', () => {

		const only = ( pos ) => ( { active: pos.x.greaterThan( 0 ), target: float( 0 ) } );

		expect( combineDirichlet( only, undefined ) ).toBe( only );
		expect( combineDirichlet( undefined, only ) ).toBe( only );

	} );

	it( 'combines two dirichlet functions into one that returns real nodes', () => {

		const a = ( pos ) => ( { active: pos.x.greaterThan( 0 ), target: float( 0 ) } );
		const b = ( pos ) => ( { active: pos.y.greaterThan( 0 ), target: float( 1 ) } );

		const combined = combineDirichlet( a, b );
		const { active, target } = combined( vec2( 1, 1 ) );

		expectNode( active );
		expectNode( target );

	} );

} );
