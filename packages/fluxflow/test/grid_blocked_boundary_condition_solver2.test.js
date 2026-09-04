// 只测"构建"（构造函数、setCollider(null,...)）不测"派发"（constrainVelocity()、
// setCollider(真实collider,...) 都会触发真正的 kernel 调用，需要 tsl_array_n.init()
// 建好的真实 renderer，vitest/Node 环境没有）——跟这次移植其它所有测试文件的
// 一致原则：图构建不需要 GPU，dispatch 需要。

import { describe, it, expect } from 'vitest';
import { createFaceCenteredGrid2 } from '../src/grid/grid_data2.js';
import { createGridBlockedBoundaryConditionSolver2 } from '../src/grid/grid_blocked_boundary_condition_solver2.js';
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

} );
