// Structural tests only, matching grid_flip_solver2.test.js's own scope and
// its header comment's reasoning: onAdvanceTimeStep() needs a live renderer
// and, for the P2G scatter and the per-cell phase counting specifically,
// real WebGPU atomics this dev/CI environment cannot provide at all. So
// what's exercised here is construction, option validation, and field
// shapes/types -- plus computeTwoPhaseBoxSeed, which (like
// computeFlipBoxSeed) is pure JS with no GPU/TSL involvement and so is the
// one part below testing real logic.
//
// The physics of the variable-density projection this solver is built on --
// operator symmetry, the diagonal matching the operator, the projection
// actually removing divergence, and buoyancy emerging from the density jump
// with no buoyancy force anywhere -- is covered separately in
// variable_density_projection.test.js, which reimplements the same stencil
// in plain JS specifically so it CAN be run without a GPU.
//
// options.collider is deliberately NOT exercised with a real collider, same
// as grid_flip_solver2.test.js: a real collider makes
// createGridBlockedBoundaryConditionSolver2's constructor dispatch a kernel
// immediately, which needs a renderer this environment doesn't have.

import { describe, it, expect } from 'vitest';
import * as tsl_array_n from 'tsl_array_n';
import {
	createGridTwoPhaseFlipSolver2, computeTwoPhaseBoxSeed,
	PHASE_LIQUID, PHASE_GAS
} from '../src/grid/grid_two_phase_flip_solver2.js';
import { createFaceCenteredGrid2 } from '../src/grid/grid_data2.js';

describe( 'computeTwoPhaseBoxSeed', () => {

	it( 'seeds particlesPerCellAxis^2 particles per covered cell', () => {

		const seed = computeTwoPhaseBoxSeed( { boxMin: [ 0, 0 ], boxMax: [ 4, 2 ], gridSpacingX: 1, gridSpacingY: 1, particlesPerCellAxis: 2, jitter: 0 } );

		expect( seed.count ).toBe( 4 * 2 * 2 * 2 );

	} );

	it( 'positions/velocities/phases are sized per particle, velocities zeroed', () => {

		const seed = computeTwoPhaseBoxSeed( { boxMin: [ 0, 0 ], boxMax: [ 2, 2 ], gridSpacingX: 1, gridSpacingY: 1, particlesPerCellAxis: 1, jitter: 0 } );

		expect( seed.positionsArray.length ).toBe( seed.count * 2 );
		expect( seed.velocitiesArray.length ).toBe( seed.count * 2 );
		expect( seed.phasesArray.length ).toBe( seed.count );
		expect( Array.from( seed.velocitiesArray ).every( ( v ) => v === 0 ) ).toBe( true );

	} );

	it( 'tags every particle liquid by default', () => {

		const seed = computeTwoPhaseBoxSeed( { boxMin: [ 0, 0 ], boxMax: [ 2, 2 ], gridSpacingX: 1, gridSpacingY: 1, jitter: 0 } );

		expect( Array.from( seed.phasesArray ).every( ( p ) => p === PHASE_LIQUID ) ).toBe( true );
		expect( seed.liquidCount ).toBe( seed.count );
		expect( seed.gasCount ).toBe( 0 );

	} );

	it( 'splits phases by the isLiquid predicate, and the counts agree with the tags', () => {

		const seed = computeTwoPhaseBoxSeed( {
			boxMin: [ 0, 0 ], boxMax: [ 4, 4 ], gridSpacingX: 1, gridSpacingY: 1,
			particlesPerCellAxis: 2, jitter: 0,
			isLiquid: ( [ , y ] ) => y < 2
		} );

		expect( seed.liquidCount ).toBe( seed.count / 2 );
		expect( seed.gasCount ).toBe( seed.count / 2 );

		for ( let p = 0; p < seed.count; p ++ ) {

			const y = seed.positionsArray[ p * 2 + 1 ];
			expect( seed.phasesArray[ p ] ).toBe( y < 2 ? PHASE_LIQUID : PHASE_GAS );

		}

		expect( seed.liquidCount + seed.gasCount ).toBe( seed.count );

	} );

	it( 'passes the particle position to isLiquid as an [x, y] pair', () => {

		const seen = [];
		computeTwoPhaseBoxSeed( {
			boxMin: [ 0, 0 ], boxMax: [ 1, 1 ], gridSpacingX: 1, gridSpacingY: 1,
			particlesPerCellAxis: 1, jitter: 0,
			isLiquid: ( pos ) => { seen.push( pos ); return true; }
		} );

		expect( seen ).toHaveLength( 1 );
		expect( seen[ 0 ] ).toEqual( [ 0.5, 0.5 ] );

	} );

	it( 'with zero jitter, every particle lands strictly inside [boxMin,boxMax]', () => {

		const seed = computeTwoPhaseBoxSeed( { boxMin: [ 1, 1 ], boxMax: [ 3, 4 ], gridSpacingX: 1, gridSpacingY: 1, particlesPerCellAxis: 3, jitter: 0 } );

		for ( let i = 0; i < seed.count; i ++ ) {

			expect( seed.positionsArray[ i * 2 ] ).toBeGreaterThan( 1 );
			expect( seed.positionsArray[ i * 2 ] ).toBeLessThan( 3 );
			expect( seed.positionsArray[ i * 2 + 1 ] ).toBeGreaterThan( 1 );
			expect( seed.positionsArray[ i * 2 + 1 ] ).toBeLessThan( 4 );

		}

	} );

} );

describe( 'createGridTwoPhaseFlipSolver2', () => {

	const makeGrid = ( nx = 4, ny = 4 ) => createFaceCenteredGrid2( nx, ny, 1, 1, 0, 0 );

	it( 'throws without velocityGrid', () => {

		expect( () => createGridTwoPhaseFlipSolver2( { maxParticles: 16 } ) ).toThrow();

	} );

	it( 'throws without maxParticles', () => {

		expect( () => createGridTwoPhaseFlipSolver2( { velocityGrid: makeGrid() } ) ).toThrow();

	} );

	it( 'constructs with velocityGrid and maxParticles only', () => {

		expect( () => createGridTwoPhaseFlipSolver2( { velocityGrid: makeGrid(), maxParticles: 16 } ) ).not.toThrow();

	} );

	it( 'constructs given a full option set', () => {

		expect( () => createGridTwoPhaseFlipSolver2( {
			velocityGrid: makeGrid(), maxParticles: 16, dt: 1 / 60,
			gravity: [ 0, - 9.81 ],
			liquidDensity: 1, gasDensity: 0.01,
			flipRatio: 0.97, gasFlipRatio: 0.9, velocityDamping: 0.02,
			p2gAtomicScale: 1024, weightEpsilon: 1e-3,
			pressurePin: 'topCenter',
			resample: { enabled: true, minParticlesPerCell: 3, maxParticlesPerCell: 8 },
			pressure: { multigrid: { numberOfLevels: 2 }, tolerance: 1e-4, maxIterations: 40, atomicScale: 256 }
		} ) ).not.toThrow();

	} );

	it( 'constructs with resampling disabled', () => {

		expect( () => createGridTwoPhaseFlipSolver2( {
			velocityGrid: makeGrid(), maxParticles: 16, resample: { enabled: false }
		} ) ).not.toThrow();

	} );

	it( 'rejects a non-positive density on either phase', () => {

		expect( () => createGridTwoPhaseFlipSolver2( { velocityGrid: makeGrid(), maxParticles: 16, gasDensity: 0 } ) ).toThrow();
		expect( () => createGridTwoPhaseFlipSolver2( { velocityGrid: makeGrid(), maxParticles: 16, liquidDensity: - 1 } ) ).toThrow();

	} );

	it( 'rejects gasDensity heavier than liquidDensity (silently inverts buoyancy otherwise)', () => {

		expect( () => createGridTwoPhaseFlipSolver2( {
			velocityGrid: makeGrid(), maxParticles: 16, liquidDensity: 1, gasDensity: 2
		} ) ).toThrow( /gasDensity/ );

	} );

	it( 'rejects a malformed pressurePin', () => {

		expect( () => createGridTwoPhaseFlipSolver2( {
			velocityGrid: makeGrid(), maxParticles: 16, pressurePin: [ 1, 2, 3 ]
		} ) ).toThrow( /pressurePin/ );

	} );

	it( "pressurePin: 'topCenter' resolves to mantaflow's own (sizeX/2, sizeY-1) cell", () => {

		const solver = createGridTwoPhaseFlipSolver2( { velocityGrid: makeGrid( 8, 6 ), maxParticles: 16 } );

		expect( solver.pinCell ).toEqual( [ 4, 5 ] );

	} );

	it( 'pressurePin accepts an explicit cell index, and null opts out entirely', () => {

		expect( createGridTwoPhaseFlipSolver2( { velocityGrid: makeGrid(), maxParticles: 16, pressurePin: [ 1, 2 ] } ).pinCell ).toEqual( [ 1, 2 ] );
		expect( createGridTwoPhaseFlipSolver2( { velocityGrid: makeGrid(), maxParticles: 16, pressurePin: null } ).pinCell ).toBe( null );

	} );

	it( 'accepts a live array0 node for dt and for velocityDamping', () => {

		const dt = tsl_array_n.array0( 'float' );
		dt.fromArray( new Float32Array( [ 1 / 60 ] ) );
		const damping = tsl_array_n.array0( 'float' );
		damping.fromArray( new Float32Array( [ 0.02 ] ) );

		expect( () => createGridTwoPhaseFlipSolver2( {
			velocityGrid: makeGrid(), maxParticles: 16, dt: dt(), velocityDamping: damping()
		} ) ).not.toThrow();

	} );

	it( 'returns onAdvanceTimeStep as a function', () => {

		const solver = createGridTwoPhaseFlipSolver2( { velocityGrid: makeGrid(), maxParticles: 16 } );

		expect( typeof solver.onAdvanceTimeStep ).toBe( 'function' );

	} );

	it( 'positions/velocities are vec2 and phase is float, all sized maxParticles', () => {

		const solver = createGridTwoPhaseFlipSolver2( { velocityGrid: makeGrid(), maxParticles: 16 } );

		expect( solver.positions.type ).toBe( 'vec2' );
		expect( solver.positions.shape ).toEqual( [ 16 ] );
		expect( solver.velocities.type ).toBe( 'vec2' );
		expect( solver.velocities.shape ).toEqual( [ 16 ] );
		expect( solver.phase.type ).toBe( 'float' );
		expect( solver.phase.shape ).toEqual( [ 16 ] );

	} );

	it( 'liquidFraction and density are cell-centered, matching velocityGrid.resolution', () => {

		const solver = createGridTwoPhaseFlipSolver2( { velocityGrid: makeGrid( 5, 7 ), maxParticles: 16 } );

		expect( solver.liquidFraction.shape ).toEqual( [ 5, 7 ] );
		expect( solver.density.shape ).toEqual( [ 5, 7 ] );

	} );

	it( 'betaU/betaV are MAC face fields -- exactly the layout multigrid faceWeights expects', () => {

		const solver = createGridTwoPhaseFlipSolver2( { velocityGrid: makeGrid( 5, 7 ), maxParticles: 16 } );

		expect( solver.betaU.shape ).toEqual( [ 6, 7 ] ); // [resX+1, resY]
		expect( solver.betaV.shape ).toEqual( [ 5, 8 ] ); // [resX, resY+1]

	} );

	it( 'exposes boundarySolver and pressureSolver', () => {

		const solver = createGridTwoPhaseFlipSolver2( { velocityGrid: makeGrid(), maxParticles: 16 } );

		expect( typeof solver.boundarySolver.constrainVelocity ).toBe( 'function' );
		expect( typeof solver.pressureSolver.project ).toBe( 'function' );

	} );

} );

describe( 'createGridTwoPhaseFlipSolver2 -- live gasDensity', () => {

	const makeGrid = () => createFaceCenteredGrid2( 4, 4, 1, 1, 0, 0 );

	it( 'accepts a live array0 node for gasDensity', () => {

		const gasDensity = tsl_array_n.array0( 'float' );
		gasDensity.fromArray( new Float32Array( [ 0.01 ] ) );

		expect( () => createGridTwoPhaseFlipSolver2( {
			velocityGrid: makeGrid(), maxParticles: 16, gasDensity: gasDensity()
		} ) ).not.toThrow();

	} );

	it( 'does not reject an out-of-range live gasDensity node -- the clamp is in the kernel graph instead', () => {

		// A node cannot be range-checked in JS, and refusing every node would
		// defeat the point of making it live. The invariants (> 0, and lighter
		// than the liquid) are enforced where the value is read instead.
		const gasDensity = tsl_array_n.array0( 'float' );
		gasDensity.fromArray( new Float32Array( [ - 5 ] ) );

		expect( () => createGridTwoPhaseFlipSolver2( {
			velocityGrid: makeGrid(), maxParticles: 16, gasDensity: gasDensity()
		} ) ).not.toThrow();

	} );

	it( 'still rejects a bad NUMERIC gasDensity, where a JS check is possible', () => {

		expect( () => createGridTwoPhaseFlipSolver2( { velocityGrid: makeGrid(), maxParticles: 16, gasDensity: - 5 } ) ).toThrow();

	} );

	it( 'rejects a live node for liquidDensity -- it is the normalization reference, not a physical knob', () => {

		const liquidDensity = tsl_array_n.array0( 'float' );
		liquidDensity.fromArray( new Float32Array( [ 1 ] ) );

		expect( () => createGridTwoPhaseFlipSolver2( {
			velocityGrid: makeGrid(), maxParticles: 16, liquidDensity: liquidDensity()
		} ) ).toThrow( /liquidDensity/ );

	} );

} );
