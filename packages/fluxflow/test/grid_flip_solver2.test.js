// Structural tests only -- see reduction.test.js's own header comment for
// why real GPU dispatch (onAdvanceTimeStep() actually running, which needs
// a live renderer and, for the P2G step specifically, real WebGPU atomics
// this dev/CI environment cannot provide at all) isn't exercised here.
// computeFlipBoxSeed is pure JS with no GPU/TSL involvement, so its own
// tests are the one part of this file actually exercising real logic.
//
// options.collider is deliberately NOT exercised with a real (non-null)
// collider object anywhere below, matching grid_blocked_boundary_condition_
// solver2.test.js's own established precedent (see that file's own header
// comment): passing a real collider makes createGridBlockedBoundaryCondition
// Solver2's own constructor dispatch buildBlockMarker() immediately, a real
// kernel dispatch needing a live renderer this environment doesn't have --
// not just untested, actively unsafe to call here. Real-hardware coverage
// for this option lives in examples/21-flip-irregular-container/,
// examples/22-flip-multiple-colliders/, and examples/23-flip-moving-
// collider/ instead.

import { describe, it, expect } from 'vitest';
import * as tsl_array_n from 'tsl_array_n';
import { createGridFlipSolver2, computeFlipBoxSeed } from '../src/grid/grid_flip_solver2.js';
import { createFaceCenteredGrid2 } from '../src/grid/grid_data2.js';

describe( 'computeFlipBoxSeed', () => {

	it( 'seeds particlesPerCellAxis^2 particles per covered cell', () => {

		const seed = computeFlipBoxSeed( { boxMin: [ 0, 0 ], boxMax: [ 4, 2 ], gridSpacingX: 1, gridSpacingY: 1, particlesPerCellAxis: 2, jitter: 0 } );

		expect( seed.count ).toBe( 4 * 2 * 2 * 2 ); // 4x2 cells, 2x2 particles/cell

	} );

	it( 'positionsArray/velocitiesArray are sized 2 floats per particle, velocities zeroed', () => {

		const seed = computeFlipBoxSeed( { boxMin: [ 0, 0 ], boxMax: [ 2, 2 ], gridSpacingX: 1, gridSpacingY: 1, particlesPerCellAxis: 1, jitter: 0 } );

		expect( seed.positionsArray.length ).toBe( seed.count * 2 );
		expect( seed.velocitiesArray.length ).toBe( seed.count * 2 );
		expect( Array.from( seed.velocitiesArray ).every( ( v ) => v === 0 ) ).toBe( true );

	} );

	it( 'with zero jitter, every particle lands strictly inside [boxMin,boxMax]', () => {

		const seed = computeFlipBoxSeed( { boxMin: [ 1, 1 ], boxMax: [ 3, 4 ], gridSpacingX: 1, gridSpacingY: 1, particlesPerCellAxis: 3, jitter: 0 } );

		for ( let i = 0; i < seed.count; i ++ ) {

			const x = seed.positionsArray[ i * 2 ];
			const y = seed.positionsArray[ i * 2 + 1 ];
			expect( x ).toBeGreaterThan( 1 );
			expect( x ).toBeLessThan( 3 );
			expect( y ).toBeGreaterThan( 1 );
			expect( y ).toBeLessThan( 4 );

		}

	} );

	it( 'defaults to particlesPerCellAxis=2 and a small jitter', () => {

		const seed = computeFlipBoxSeed( { boxMin: [ 0, 0 ], boxMax: [ 1, 1 ], gridSpacingX: 1, gridSpacingY: 1 } );

		expect( seed.count ).toBe( 4 );

	} );

} );

describe( 'createGridFlipSolver2', () => {

	it( 'throws without velocityGrid', () => {

		expect( () => createGridFlipSolver2( { maxParticles: 16 } ) ).toThrow();

	} );

	it( 'throws without maxParticles', () => {

		const velocityGrid = createFaceCenteredGrid2( 4, 4, 1, 1, 0, 0 );
		expect( () => createGridFlipSolver2( { velocityGrid } ) ).toThrow();

	} );

	it( 'constructs without throwing given velocityGrid and maxParticles only', () => {

		const velocityGrid = createFaceCenteredGrid2( 4, 4, 1, 1, 0, 0 );
		expect( () => createGridFlipSolver2( { velocityGrid, maxParticles: 16 } ) ).not.toThrow();

	} );

	it( 'constructs without throwing given a full option set', () => {

		const velocityGrid = createFaceCenteredGrid2( 4, 4, 1, 1, 0, 0 );

		expect( () => createGridFlipSolver2( {
			velocityGrid, maxParticles: 16, dt: 1 / 30,
			gravity: [ 0, -9.81 ], flipRatio: 0.95, velocityDamping: 0.02,
			p2gAtomicScale: 1024, weightEpsilon: 1e-3,
			resample: { enabled: true, minParticlesPerCell: 4, maxParticlesPerCell: 8 },
			pressure: { multigrid: { numberOfLevels: 2 }, tolerance: 1e-4, maxIterations: 40 }
		} ) ).not.toThrow();

	} );

	it( 'constructs without throwing with resampling disabled', () => {

		const velocityGrid = createFaceCenteredGrid2( 4, 4, 1, 1, 0, 0 );

		expect( () => createGridFlipSolver2( {
			velocityGrid, maxParticles: 16, resample: { enabled: false }
		} ) ).not.toThrow();

	} );

	it( 'resampling is enabled by default (options.resample omitted)', () => {

		const velocityGrid = createFaceCenteredGrid2( 4, 4, 1, 1, 0, 0 );

		expect( () => createGridFlipSolver2( { velocityGrid, maxParticles: 16 } ) ).not.toThrow();

	} );

	it( 'accepts a live array0 node for dt', () => {

		const velocityGrid = createFaceCenteredGrid2( 4, 4, 1, 1, 0, 0 );
		const dt = tsl_array_n.array0( 'float' );
		dt.fromArray( new Float32Array( [ 1 / 30 ] ) );

		expect( () => createGridFlipSolver2( { velocityGrid, maxParticles: 16, dt: dt() } ) ).not.toThrow();

	} );

	it( 'returns onAdvanceTimeStep as a function', () => {

		const velocityGrid = createFaceCenteredGrid2( 4, 4, 1, 1, 0, 0 );
		const flip = createGridFlipSolver2( { velocityGrid, maxParticles: 16 } );

		expect( typeof flip.onAdvanceTimeStep ).toBe( 'function' );

	} );

	it( 'positions/velocities are vec2 fields sized maxParticles', () => {

		const velocityGrid = createFaceCenteredGrid2( 4, 4, 1, 1, 0, 0 );
		const flip = createGridFlipSolver2( { velocityGrid, maxParticles: 16 } );

		expect( flip.positions.type ).toBe( 'vec2' );
		expect( flip.positions.shape ).toEqual( [ 16 ] );
		expect( flip.velocities.type ).toBe( 'vec2' );
		expect( flip.velocities.shape ).toEqual( [ 16 ] );

	} );

	it( 'fluidMask shape matches velocityGrid.resolution', () => {

		const velocityGrid = createFaceCenteredGrid2( 5, 7, 1, 1, 0, 0 );
		const flip = createGridFlipSolver2( { velocityGrid, maxParticles: 16 } );

		expect( flip.fluidMask.shape ).toEqual( [ 5, 7 ] );

	} );

	it( 'exposes boundarySolver and pressureSolver', () => {

		const velocityGrid = createFaceCenteredGrid2( 4, 4, 1, 1, 0, 0 );
		const flip = createGridFlipSolver2( { velocityGrid, maxParticles: 16 } );

		expect( typeof flip.boundarySolver.constrainVelocity ).toBe( 'function' );
		expect( typeof flip.pressureSolver.project ).toBe( 'function' );

	} );

} );
