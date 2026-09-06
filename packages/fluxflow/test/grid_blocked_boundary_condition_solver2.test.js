// Only tests "building" (the constructor, setCollider(null,...)), not
// "dispatching" (constrainVelocity() and setCollider(a real collider,...)
// both trigger real kernel calls, which need a real renderer set up by
// tsl_array_n.init() that the vitest/Node environment doesn't have) -- the
// same principle as every other test file in this port: building a graph
// doesn't need a GPU, dispatching does.

import { describe, it, expect } from 'vitest';
import { createFaceCenteredGrid2 } from '../src/grid/grid_data2.js';
import { createGridBlockedBoundaryConditionSolver2 } from '../src/grid/grid_blocked_boundary_condition_solver2.js';
import { createSDFInflow2 } from '../src/grid/sdf_inflow_outflow2.js';
import { DIRECTION_ALL } from '../src/grid/constant.js';

describe( 'GridBlockedBoundaryConditionSolver2', () => {

	it( 'constructs without a collider (safe fromArray path, no kernel dispatch)', () => {

		const velocity = createFaceCenteredGrid2( 4, 4, 1, 1, 0, 0 );

		expect( () => createGridBlockedBoundaryConditionSolver2( velocity, 4, 4, 1, 1, 0, 0 ) ).not.toThrow();

	} );

	it( 'exposes the expected marker/temp fields sized to dataU/dataV shapes', () => {

		const velocity = createFaceCenteredGrid2( 4, 4, 1, 1, 0, 0 );
		const solver = createGridBlockedBoundaryConditionSolver2( velocity, 4, 4, 1, 1, 0, 0 );

		expect( solver.uMarker.shape ).toEqual( velocity.dataSizeU );
		expect( solver.vMarker.shape ).toEqual( velocity.dataSizeV );
		expect( solver.uTemp.shape ).toEqual( velocity.dataSizeU );
		expect( solver.vTemp.shape ).toEqual( velocity.dataSizeV );
		expect( solver.blockMarker.shape ).toEqual( [ 4, 4 ] );

	} );

	it( 'defaults closedDomainBoundaryFlag to DIRECTION_ALL, mutable directly', () => {

		const velocity = createFaceCenteredGrid2( 4, 4, 1, 1, 0, 0 );
		const solver = createGridBlockedBoundaryConditionSolver2( velocity, 4, 4, 1, 1, 0, 0 );

		expect( solver.closedDomainBoundaryFlag ).toBe( DIRECTION_ALL );
		solver.closedDomainBoundaryFlag = 0;
		expect( solver.closedDomainBoundaryFlag ).toBe( 0 );

	} );

	it( 'exposes constrainVelocity and setCollider as functions', () => {

		const velocity = createFaceCenteredGrid2( 4, 4, 1, 1, 0, 0 );
		const solver = createGridBlockedBoundaryConditionSolver2( velocity, 4, 4, 1, 1, 0, 0 );

		expect( typeof solver.constrainVelocity ).toBe( 'function' );
		expect( typeof solver.setCollider ).toBe( 'function' );

	} );

	it( 'setCollider(null, ...) stays on the safe fromArray path, no dispatch', () => {

		const velocity = createFaceCenteredGrid2( 4, 4, 1, 1, 0, 0 );
		const solver = createGridBlockedBoundaryConditionSolver2( velocity, 4, 4, 1, 1, 0, 0 );

		expect( () => solver.setCollider( null, [ 4, 4 ], [ 1, 1 ], [ 0, 0 ] ) ).not.toThrow();
		expect( solver.collider ).toBe( null );

	} );

	it( 'constructs with a single inflow object', () => {

		const velocity = createFaceCenteredGrid2( 4, 4, 1, 1, 0, 0 );
		const inflow = createSDFInflow2( 4, 4, 1, 1, 0, 0, { velocity: [ 1, 0 ] } );

		expect( () => createGridBlockedBoundaryConditionSolver2( velocity, 4, 4, 1, 1, 0, 0, null, inflow ) ).not.toThrow();

	} );

	it( 'constructs with an array of inflow objects', () => {

		const velocity = createFaceCenteredGrid2( 4, 4, 1, 1, 0, 0 );
		const a = createSDFInflow2( 4, 4, 1, 1, 0, 0, { velocity: [ 1, 0 ] } );
		const b = createSDFInflow2( 4, 4, 1, 1, 0, 0, { velocity: [ 0, 1 ], mode: 'add' } );

		expect( () => createGridBlockedBoundaryConditionSolver2( velocity, 4, 4, 1, 1, 0, 0, null, [ a, b ] ) ).not.toThrow();

	} );

	it( 'exposes setInflows as a function, callable with null/single/array', () => {

		const velocity = createFaceCenteredGrid2( 4, 4, 1, 1, 0, 0 );
		const solver = createGridBlockedBoundaryConditionSolver2( velocity, 4, 4, 1, 1, 0, 0 );
		const inflow = createSDFInflow2( 4, 4, 1, 1, 0, 0, { velocity: [ 1, 0 ] } );

		expect( typeof solver.setInflows ).toBe( 'function' );
		expect( () => solver.setInflows( inflow ) ).not.toThrow();
		expect( () => solver.setInflows( [ inflow ] ) ).not.toThrow();
		expect( () => solver.setInflows( null ) ).not.toThrow();

	} );

	it( 'existing 8-argument calls (no inflows) are unaffected', () => {

		const velocity = createFaceCenteredGrid2( 4, 4, 1, 1, 0, 0 );

		expect( () => createGridBlockedBoundaryConditionSolver2( velocity, 4, 4, 1, 1, 0, 0, null ) ).not.toThrow();

	} );

} );
