import { describe, it, expect } from 'vitest';
import { createGridSolver2 } from '../src/grid/grid_solver2.js';

describe( 'GridSolver2', () => {

	it( 'onAdvanceTimeStep calls hooks in the documented order, each with dt', () => {

		const order = [];
		const dt = 1 / 60;

		const solver = createGridSolver2( {
			beginAdvanceTimeStep: ( t ) => order.push( [ 'begin', t ] ),
			computeExternalForces: ( t ) => order.push( [ 'externalForces', t ] ),
			computeViscosity: ( t ) => order.push( [ 'viscosity', t ] ),
			computePressure: ( t ) => order.push( [ 'pressure', t ] ),
			computeAdvection: ( t ) => order.push( [ 'advection', t ] ),
			endAdvanceTimeStep: ( t ) => order.push( [ 'end', t ] )
		} );

		solver.onAdvanceTimeStep( dt );

		expect( order ).toEqual( [
			[ 'begin', dt ],
			[ 'externalForces', dt ],
			[ 'viscosity', dt ],
			[ 'pressure', dt ],
			[ 'advection', dt ],
			[ 'end', dt ]
		] );

	} );

	it( 'missing hooks default to no-ops, do not throw', () => {

		const solver = createGridSolver2();
		expect( () => solver.onAdvanceTimeStep( 1 / 60 ) ).not.toThrow();

	} );

	it( 'partial hooks: unspecified stages are no-ops, specified ones still run', () => {

		let pressureCalls = 0;
		const solver = createGridSolver2( { computePressure: () => pressureCalls ++ } );

		solver.onAdvanceTimeStep( 1 / 60 );

		expect( pressureCalls ).toBe( 1 );

	} );

} );
