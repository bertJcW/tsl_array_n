// Structural tests only, matching every other solver in this port: pure
// graph construction (no GPU needed), including building the U/V
// extrapolation and scalar-clear kernels. The actual outflow behavior
// (velocity extrapolation, scalar clearing) needs real dispatch + readback,
// verified live instead, in examples/15-flow-past-cylinder/ (needs real
// WebGPU hardware, same as every other atomics-adjacent example in this
// port -- the pressure side of this solver goes through
// createGridPressureSolver2's own MGPCG).

import { describe, it, expect } from 'vitest';
import { float } from 'three/tsl';
import { createGridOutflowSolver2 } from '../src/grid/grid_outflow_solver2.js';
import { createSDFOutflow2 } from '../src/grid/sdf_inflow_outflow2.js';
import { createFaceCenteredGrid2, createScalarGrid2 } from '../src/grid/grid_data2.js';

function makeVelocityGrids() {

	return {
		velocityGrid: createFaceCenteredGrid2( 8, 8, 1, 1, -4, -4 ),
		velocityPrev: createFaceCenteredGrid2( 8, 8, 1, 1, -4, -4 )
	};

}

describe( 'createGridOutflowSolver2', () => {

	it( 'constructs with a single outflow object without throwing', () => {

		const { velocityGrid, velocityPrev } = makeVelocityGrids();
		const outflow = createSDFOutflow2( 8, 8, 1, 1, -4, -4 );

		expect( () => createGridOutflowSolver2( { velocityGrid, velocityPrev, outflows: outflow, dt: 1 / 60 } ) ).not.toThrow();

	} );

	it( 'constructs with an array of outflow objects without throwing', () => {

		const { velocityGrid, velocityPrev } = makeVelocityGrids();
		const a = createSDFOutflow2( 8, 8, 1, 1, -4, -4 );
		const b = createSDFOutflow2( 8, 8, 1, 1, -4, -4 );

		expect( () => createGridOutflowSolver2( { velocityGrid, velocityPrev, outflows: [ a, b ], dt: 1 / 60 } ) ).not.toThrow();

	} );

	it( 'returns dirichlet/applyOutflowVelocityBC/clearOutflowScalarField', () => {

		const { velocityGrid, velocityPrev } = makeVelocityGrids();
		const outflow = createSDFOutflow2( 8, 8, 1, 1, -4, -4 );

		const solver = createGridOutflowSolver2( { velocityGrid, velocityPrev, outflows: outflow, dt: 1 / 60 } );

		expect( typeof solver.dirichlet ).toBe( 'function' );
		expect( typeof solver.applyOutflowVelocityBC ).toBe( 'function' );
		expect( typeof solver.clearOutflowScalarField ).toBe( 'function' );

	} );

	it( 'clearOutflowScalarField(scalarGrid) returns a dispatcher without invoking it', () => {

		const { velocityGrid, velocityPrev } = makeVelocityGrids();
		const outflow = createSDFOutflow2( 8, 8, 1, 1, -4, -4 );
		const dyeGrid = createScalarGrid2( 8, 8, 1, 1, -4, -4 );

		const solver = createGridOutflowSolver2( { velocityGrid, velocityPrev, outflows: outflow, dt: 1 / 60 } );
		const dispatch = solver.clearOutflowScalarField( dyeGrid );

		expect( typeof dispatch ).toBe( 'function' );

	} );

	it( 'accepts a live dt node in addition to a plain number', () => {

		const { velocityGrid, velocityPrev } = makeVelocityGrids();
		const outflow = createSDFOutflow2( 8, 8, 1, 1, -4, -4 );

		expect( () => createGridOutflowSolver2( { velocityGrid, velocityPrev, outflows: outflow, dt: float( 1 / 60 ) } ) ).not.toThrow();

	} );

	it( 'accepts applyVelocityBC: false without throwing, still returns applyOutflowVelocityBC as a function', () => {

		const { velocityGrid, velocityPrev } = makeVelocityGrids();
		const outflow = createSDFOutflow2( 8, 8, 1, 1, -4, -4 );

		const solver = createGridOutflowSolver2( { velocityGrid, velocityPrev, outflows: outflow, dt: 1 / 60, applyVelocityBC: false } );

		expect( typeof solver.applyOutflowVelocityBC ).toBe( 'function' );

	} );

} );
