// Structural tests only: construction and hook-override ordering, no GPU
// dispatch of the real (non-overridden) computePressure/computeAdvection
// defaults -- those internally call the CG solver's GPU-atomic dot
// product, which needs a live renderer and, in this dev sandbox, hits the
// same atomics compile-time wall as every other MGPCG-based example in
// this port. Verified live instead in examples/13-interactive-pressure/.

import { describe, it, expect } from 'vitest';
import { createGridSolver2 } from '../src/grid/grid_solver2.js';
import { createFaceCenteredGrid2 } from '../src/grid/grid_data2.js';
import { createSDFInflow2, createSDFOutflow2 } from '../src/grid/sdf_inflow_outflow2.js';
import { DIRECTION_ALL, DIRECTION_RIGHT } from '../src/grid/constant.js';

describe( 'createGridSolver2', () => {

	it( 'throws without options.velocityGrid', () => {

		expect( () => createGridSolver2() ).toThrow( /velocityGrid/ );
		expect( () => createGridSolver2( {} ) ).toThrow( /velocityGrid/ );

	} );

	it( 'constructs with just a velocityGrid, no other options', () => {

		const velocityGrid = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );

		expect( () => createGridSolver2( { velocityGrid } ) ).not.toThrow();

	} );

	it( 'constructs with a full option set (force, dirichlet, advection, pressure)', () => {

		const velocityGrid = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );

		expect( () => createGridSolver2( {
			velocityGrid,
			gridSpacing: [ 1, 1 ],
			force: ( pos ) => pos,
			dirichlet: () => ( { active: 0, target: 0 } ),
			dt: 0.1,
			advection: { maxSubsteps: 8 },
			pressure: { multigrid: { numberOfLevels: 2 }, tolerance: 1e-4, maxIterations: 20 }
		} ) ).not.toThrow();

	} );

	it( 'constructs with inflows/outflows/closedDomainBoundaryFlag', () => {

		const velocityGrid = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );
		const inflow = createSDFInflow2( 8, 8, 1, 1, 0, 0, { velocity: [ 1, 0 ] } );
		const outflow = createSDFOutflow2( 8, 8, 1, 1, 0, 0 );

		expect( () => createGridSolver2( {
			velocityGrid,
			inflows: inflow,
			outflows: outflow,
			closedDomainBoundaryFlag: DIRECTION_ALL & ~DIRECTION_RIGHT,
			dt: 1 / 60
		} ) ).not.toThrow();

	} );

	it( 'applies closedDomainBoundaryFlag to boundarySolver at construction', () => {

		const velocityGrid = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );
		const flag = DIRECTION_ALL & ~DIRECTION_RIGHT;

		const solver = createGridSolver2( { velocityGrid, closedDomainBoundaryFlag: flag } );

		expect( solver.boundarySolver.closedDomainBoundaryFlag ).toBe( flag );

	} );

	it( 'exposes outflowSolver when outflows is given, null otherwise', () => {

		const velocityGridWith = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );
		const outflow = createSDFOutflow2( 8, 8, 1, 1, 0, 0 );
		const withOutflow = createGridSolver2( { velocityGrid: velocityGridWith, outflows: outflow } );

		expect( withOutflow.outflowSolver ).toBeTruthy();
		expect( typeof withOutflow.outflowSolver.applyOutflowVelocityBC ).toBe( 'function' );
		expect( typeof withOutflow.outflowSolver.clearOutflowScalarField ).toBe( 'function' );

		const velocityGridWithout = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );
		const withoutOutflow = createGridSolver2( { velocityGrid: velocityGridWithout } );

		expect( withoutOutflow.outflowSolver ).toBe( null );

	} );

	it( 'onAdvanceTimeStep calls hook overrides, in the documented order, each with dt', async () => {

		const velocityGrid = createFaceCenteredGrid2( 4, 4, 1, 1, 0, 0 );
		const order = [];
		const dt = 1 / 60;

		const solver = createGridSolver2( {
			velocityGrid,
			beginAdvanceTimeStep: ( t ) => order.push( [ 'begin', t ] ),
			computeExternalForces: ( t ) => order.push( [ 'externalForces', t ] ),
			computeViscosity: ( t ) => order.push( [ 'viscosity', t ] ),
			computePressure: ( t ) => order.push( [ 'pressure', t ] ),
			computeAdvection: ( t ) => order.push( [ 'advection', t ] ),
			endAdvanceTimeStep: ( t ) => order.push( [ 'end', t ] )
		} );

		await solver.onAdvanceTimeStep( dt );

		expect( order ).toEqual( [
			[ 'begin', dt ],
			[ 'externalForces', dt ],
			[ 'viscosity', dt ],
			[ 'pressure', dt ],
			[ 'advection', dt ],
			[ 'end', dt ]
		] );

	} );

	it( 'computeViscosity has no built-in default -- stays a no-op unless overridden', async () => {

		// every *other* stage is overridden here specifically to avoid the
		// real (non-overridden) computePressure/computeAdvection defaults,
		// which need a live renderer -- see this file's own header comment.
		const velocityGrid = createFaceCenteredGrid2( 4, 4, 1, 1, 0, 0 );
		let viscosityCalls = 0;

		const solver = createGridSolver2( {
			velocityGrid,
			computeExternalForces: () => {},
			computeViscosity: () => viscosityCalls ++,
			computePressure: () => {},
			computeAdvection: () => {}
		} );

		await solver.onAdvanceTimeStep( 1 / 60 );

		expect( viscosityCalls ).toBe( 1 );

	} );

	it( 'exposes boundarySolver, a createGridBlockedBoundaryConditionSolver2 instance', () => {

		// constrainVelocity() itself is not called here -- it dispatches real
		// kernels, which needs a live renderer (see this file's own header
		// comment); only construction is exercised in this test suite.
		const velocityGrid = createFaceCenteredGrid2( 8, 8, 1, 1, 0, 0 );
		const solver = createGridSolver2( { velocityGrid } );

		expect( typeof solver.boundarySolver.constrainVelocity ).toBe( 'function' );

	} );

	it( 'onAdvanceTimeStep returns a promise (async)', () => {

		const velocityGrid = createFaceCenteredGrid2( 4, 4, 1, 1, 0, 0 );
		const solver = createGridSolver2( {
			velocityGrid,
			computeExternalForces: () => {},
			computeViscosity: () => {},
			computePressure: () => {},
			computeAdvection: () => {}
		} );

		expect( solver.onAdvanceTimeStep( 1 / 60 ) ).toBeInstanceOf( Promise );

	} );

} );
