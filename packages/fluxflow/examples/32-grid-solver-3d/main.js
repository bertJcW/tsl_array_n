// Step 2 of the 3D effort: examples/31-conjugate-gradient-3d/ already
// confirmed linalg.js/multigrid.js's own dimension-generic core works in
// 3D (and found/fixed a real bug, buildCoarseSweepKernel, along the way).
// This example is the first real-hardware exercise of the actual grid/
// layer ported on top of that (grid_solver3.js, grid_flip_solver3.js,
// and everything they in turn depend on) -- none of that has run on real
// WebGPU before this. Two independent checks, in one page, same
// multi-check-per-page shape as examples/31's own:
//
//   A) createGridSolver3 -- the plain (non-particle) solver: advection +
//      pressure + a constant force, in a closed box. After each step the
//      projection should leave the velocity field divergence-free (up to
//      solver tolerance); this is checked directly by evaluating
//      faceCenteredDivergenceAtCenter3 at every cell, not inferred.
//
//   B) createGridFlipSolver3 -- the particle solver, exercising the parts
//      A) cannot: P2G/G2P (bilinear -> trilinear scatter/gather), the
//      GPU-atomic accumulators, reduced-pressure's air-cell Dirichlet
//      target, and the resampling pass's 6-neighbour occupancy check.
//      A small box of particles is dropped under gravity in a closed
//      domain; the check is that nothing goes non-finite and every
//      particle stays inside the domain the advection/resample clamps are
//      supposed to keep it in -- not a quantitative "does it look like
//      water" check, which needs a renderer, not this harness.
//
// *** Run on real WebGPU hardware (real Chrome via claude-in-chrome, not
// this project's own WebGL2 fallback sandbox, which cannot compile the
// atomics this file's FLIP check depends on) -- both checks passed. ***
//
// A) finite=true, maxAbsVelocityComponent=0.0000, maxAbsDivergence~3.7e-8
// across two runs. The exact-zero velocity is not a bug: a uniform body
// force in a fully closed box is hydrostatic equilibrium, and pressure
// exactly cancels it everywhere, so zero velocity is the physically
// correct steady state, not an inert solver. maxAbsDivergence at machine-
// epsilon scale confirms the projection is actually running, not silently
// no-op-ing.
//
// B) finite=true, withinBounds=true across two separate runs (particle
// jitter differs seed to seed by design, so maxParticleSpeed varied
// 4.5-29.2 between them -- both comfortably under MAX_PARTICLE_VELOCITY's
// 500 circuit breaker, and neither run needed it). Confirms the P2G
// atomic scatter/gather, the reduced-pressure air-cell Dirichlet target,
// and the resampling pass's atomics all execute correctly together on
// real hardware, not just compile.
//
// One real finding along the way, in this file's own test harness rather
// than the solver: arrayN('vec3', N).toArray() returns the RAW GPU
// buffer, which pads every vec3 element to 4 floats (WGSL's standard
// storage-array stride for vec3 -- 16-byte alignment), not the 3 you'd
// naively expect -- confirmed directly with a controlled 3-element probe
// on this same page, which came back as
// [1,2,3,0, 10,20,30,0, 100,200,300,0]. fromArray() has no equivalent
// gotcha (it already accepts tightly-packed 3-per-element input
// correctly), so this only matters for reading a vec3 array back into
// JS, not for constructing/seeding one -- worth remembering the next time
// any 3D code reads a vec3 arrayN's raw buffer directly, as
// grid_flip_solver3.js's own positions/velocities would be if inspected
// the same way.

import * as tsl_array_n from 'tsl_array_n';
import { grid } from 'fluxflow';
import { vec3 } from 'three/tsl';

const pre = document.querySelector( '#status pre' );
const lines = [];

function log( label, ok, detail ) {

	const cls = ok ? 'ok' : 'err';
	const mark = ok ? '✓' : '✗';
	lines.push( `<span class="${ cls }">${ mark } ${ label }${ detail ? ' — ' + detail : '' }</span>` );
	pre.innerHTML = lines.join( '\n' );

}

try {

	const renderer = await tsl_array_n.init( { allowFallback: true } );
	log( 'init()', true, `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }` );

	// ---- A) createGridSolver3 -------------------------------------------
	{

		const N = 10;
		const dt = 0.05;

		const velocityGrid = grid.createFaceCenteredGrid3( N, N, N, 1, 1, 1, 0, 0, 0 );
		const force = () => vec3( 0, -1, 0 );

		const solver = grid.createGridSolver3( {
			velocityGrid, gridSpacing: [ 1, 1, 1 ], origin: [ 0, 0, 0 ], force, dt
		} );

		for ( let step = 0; step < 20; step ++ ) {

			await solver.onAdvanceTimeStep( dt );

		}

		const [ uArr, vArr, wArr ] = await Promise.all( [
			velocityGrid.dataU.toArray(), velocityGrid.dataV.toArray(), velocityGrid.dataW.toArray()
		] );
		const allComponents = [ ...uArr, ...vArr, ...wArr ];
		const allFinite = allComponents.length > 0 && allComponents.every( Number.isFinite );
		const maxAbsComponent = allComponents.reduce( ( m, v ) => Math.max( m, Math.abs( v ) ), 0 );

		const divergence = tsl_array_n.arrayN( 'float', [ N, N, N ] );
		const computeDivergence = tsl_array_n.kernel( [ N, N, N ], ( i, j, k ) => {

			divergence( i, j, k ).assign( grid.faceCenteredDivergenceAtCenter3(
				velocityGrid.dataU, velocityGrid.dataV, velocityGrid.dataW, velocityGrid.gridSpacing, i, j, k
			) );

		} );
		computeDivergence();

		const divArr = await divergence.toArray();
		const maxAbsDivergence = divArr.length > 0 ? divArr.reduce( ( m, v ) => Math.max( m, Math.abs( v ) ), 0 ) : NaN;

		const ok = allFinite && divArr.length > 0 && maxAbsDivergence < 1e-3;

		log(
			'createGridSolver3 — 20 steps, closed 10^3 domain, constant downward force',
			ok,
			`finite=${ allFinite }, maxAbsVelocityComponent=${ maxAbsComponent.toFixed( 4 ) }, maxAbsDivergence=${ maxAbsDivergence.toExponential( 2 ) }`
		);

	}

	// ---- B) createGridFlipSolver3 ----------------------------------------
	{

		const N = 8;
		const gridSpacing = [ 1, 1, 1 ];
		const origin = [ 0, 0, 0 ];
		const dt = 0.02;

		const velocityGrid = grid.createFaceCenteredGrid3( N, N, N, ...gridSpacing, ...origin );

		const seed = grid.computeFlipBoxSeed3( {
			boxMin: [ 1, 4, 1 ], boxMax: [ 7, 7, 7 ],
			gridSpacingX: gridSpacing[ 0 ], gridSpacingY: gridSpacing[ 1 ], gridSpacingZ: gridSpacing[ 2 ],
			particlesPerCellAxis: 2
		} );

		const flip = grid.createGridFlipSolver3( {
			velocityGrid, gridSpacing, origin, maxParticles: seed.count, dt, gravity: [ 0, -9.81, 0 ]
		} );
		flip.positions.fromArray( seed.positionsArray );
		flip.velocities.fromArray( seed.velocitiesArray );

		for ( let step = 0; step < 60; step ++ ) {

			await flip.onAdvanceTimeStep();

		}

		// arrayN('vec3', ...)'s toArray() returns the raw GPU buffer, which
		// pads each vec3 element to 4 floats (xyz + one padding float, the
		// standard WGSL storage-array stride for vec3) -- confirmed directly
		// against this same page (a controlled 3-element vec3 probe came
		// back as [1,2,3,0, 10,20,30,0, 100,200,300,0]), not assumed. Stride
		// is 4 here for that reason; fromArray() has no such gotcha (it
		// already accepts tightly-packed 3-per-element data correctly, as
		// this solver's own construction-time zeroing relies on).
		const [ posArr, velArr ] = await Promise.all( [ flip.positions.toArray(), flip.velocities.toArray() ] );
		const stride = 4;
		const allFinite = posArr.length > 0 && posArr.every( Number.isFinite ) && velArr.every( Number.isFinite );

		let withinBounds = true;
		let maxSpeed = 0;

		for ( let p = 0; p < seed.count; p ++ ) {

			const x = posArr[ p * stride ], y = posArr[ p * stride + 1 ], z = posArr[ p * stride + 2 ];
			if ( x < 0 || x > N || y < 0 || y > N || z < 0 || z > N ) withinBounds = false;

			const vx = velArr[ p * stride ], vy = velArr[ p * stride + 1 ], vz = velArr[ p * stride + 2 ];
			maxSpeed = Math.max( maxSpeed, Math.hypot( vx, vy, vz ) );

		}

		const ok = allFinite && posArr.length === seed.count * stride && withinBounds;

		log(
			'createGridFlipSolver3 — 60 steps, dam-break-style drop, closed 8^3 domain',
			ok,
			`particles=${ seed.count }, finite=${ allFinite }, withinBounds=${ withinBounds }, maxParticleSpeed=${ maxSpeed.toFixed( 3 ) }`
		);

	}

} catch ( error ) {

	log( 'failed', false, error.message );
	console.error( error );

}
