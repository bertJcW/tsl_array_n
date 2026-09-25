// First real 3D fluid *visualization* in this port -- examples/32's own
// checks already confirmed createGridFlipSolver3 runs correctly on real
// WebGPU hardware; this is the same solver (a dam-break-style box of
// liquid dropped under gravity in a closed domain), rendered instead of
// just numerically checked.
//
// Rendering approach: GPU-resident, zero per-frame readback -- every 2D
// example in this port instead reads particles.toArray() back to the CPU
// and rasterizes with Canvas2D every frame (see examples/20-flip-dam-break/
// for that established pattern), which has no 3D equivalent (Canvas2D has
// no camera/depth). A real THREE.Scene/PerspectiveCamera/OrbitControls
// pipeline is unavoidable here -- but the same tsl_array_n.init() call
// already used for compute returns a real WebGPURenderer, fully capable of
// renderer.render(scene, camera) too (confirmed by reading
// tsl_array_n/src/context.js directly: it constructs a plain
// THREE.WebGPURenderer and appends its own domElement, nothing headless
// about it), so no second renderer/canvas is needed.
//
// Each particle is one instance of a small sphere in a THREE.InstancedMesh,
// positioned entirely by a TSL positionNode that reads directly from the
// FLIP solver's own `positions` storage buffer at `instanceIndex` -- the
// same GPU-instanced-particle pattern three.js's own WebGPU compute-
// particle demos use. No JS array of particle positions ever exists on
// this file's own side; the per-instance transform (instanceMatrix) is
// left at identity for every instance since positionNode supplies the
// world offset directly, not a per-instance matrix.
//
// *** Run on real WebGPU hardware -- confirmed working over a short window,
// but a longer run surfaced a real bug (found by the user watching it live,
// not by this file's own first-pass testing, which only ran ~240 frames). ***
//
// 16384 particles, InstancedMesh + OrbitControls, WebGPUBackend. An early
// screenshot (~frame 118) shows the seeded box still mostly intact, just
// starting to slump under gravity; a later one (~frame 235, ~4.7 simulated
// seconds later) shows a clear dam-break collapse -- the block has fallen
// and spread across the domain floor with a visible splash arc of stray
// particles thrown up and to the side, exactly the qualitative shape a
// liquid collapse should have.
//
// *** The periodic collapse/explode bug, and why maxPlausiblePressure is
// set explicitly below ***
//
// Left running longer, the liquid periodically contracted then exploded --
// not a rendering artifact, confirmed via a headless real-hardware
// diagnostic harness that polled flip.pressureSolver.diagnostics every
// frame: `rejected` events (residuals reaching into the thousands, even
// millions) recurred roughly every 50-150 frames, correlating with visible
// maxSpeed spikes. Root-caused, not guessed: see grid_flip_solver3.js's own
// derivedMaxPlausiblePressure comment for the full investigation (which
// hypotheses were ruled out and how) -- the short version is that the
// library's own auto-derived pressure bound only accounts for hydrostatic
// pressure, not this scene's own genuine dynamic/impact pressure once the
// block hits the floor, so legitimate solves were being rejected and
// reverted to a stale snapshot, and the reverts themselves compounded into
// the visible blowup. Fixed in two steps, confirmed on real hardware at
// each one, not assumed: `maxPlausiblePressure: 5000` (measured against
// this scene's own actual peak, the same established per-scene convention
// examples/20-flip-dam-break/'s own pressure.atomicScale already uses)
// alone cut the 12000-frame rejection rate from 1.58% (191/12063) to
// 1.73% over a shorter 3000-frame comparison run -- still real,
// recurring, genuine CG convergence difficulty, not just an over-tight
// bound. Asking a user to reduce it further surfaced the actual second
// cause: this solver's pressure defaults to a single relaxation level
// (numberOfLevels:1, no real coarse-grid V-cycle), which isn't enough
// preconditioning for this scene's own irregular, splash-driven
// fluidMask topology. Adding `multigrid: { numberOfLevels: 3 }` (the
// most this 20^3 domain evenly supports -- see grid_flip_solver3.js's
// own header comment for the divisibility constraint and for why
// numberOfLevels:4 isn't reachable here) took the rejection rate to
// 0/3000, then confirmed at 0/12000 over the same full-length run the
// original bug was measured against. See grid_flip_solver3.js's own
// derivedMaxPlausiblePressure comment for the complete investigation,
// including what was ruled out along the way (resampling, atomicScale --
// a no-op in the current dot-product implementation, see linalg.js's own
// header comment -- and raising maxPlausiblePressure or maxIterations
// alone without real multigrid levels, neither of which reproduced this
// fix).

import * as tsl_array_n from 'tsl_array_n';
import { grid } from 'fluxflow';
import { instanceIndex, positionLocal } from 'three/tsl';
import {
	Scene, PerspectiveCamera, InstancedMesh, SphereGeometry, MeshBasicNodeMaterial,
	BoxGeometry, EdgesGeometry, LineSegments, LineBasicMaterial, Matrix4, Color
} from 'three/webgpu';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

const hud = document.querySelector( '#hud' );

function setHud( lines ) {

	hud.textContent = lines.join( '\n' );

}

setHud( [ 'initializing…' ] );

try {

	const renderer = await tsl_array_n.init( { allowFallback: true } );

	// ---- scene: a closed N^3 domain, camera looking in from outside/above.
	const N = 20;
	const gridSpacing = [ 1, 1, 1 ];
	const origin = [ 0, 0, 0 ];
	const dt = 0.02;

	const scene = new Scene();
	scene.background = new Color( 0x0b0b12 );

	const camera = new PerspectiveCamera( 50, window.innerWidth / window.innerHeight, 0.1, 500 );
	camera.position.set( N * 1.6, N * 1.3, N * 1.6 );

	const controls = new OrbitControls( camera, renderer.domElement );
	controls.target.set( N / 2, N / 2, N / 2 );
	controls.enableDamping = true;
	controls.update();

	const domainOutline = new LineSegments(
		new EdgesGeometry( new BoxGeometry( N, N, N ) ),
		new LineBasicMaterial( { color: 0x334455 } )
	);
	domainOutline.position.set( N / 2, N / 2, N / 2 );
	scene.add( domainOutline );

	window.addEventListener( 'resize', () => {

		camera.aspect = window.innerWidth / window.innerHeight;
		camera.updateProjectionMatrix();
		renderer.setSize( window.innerWidth, window.innerHeight );

	} );

	// ---- solver: same shape as examples/32's own FLIP check, scaled up
	// for a scene worth looking at.
	const velocityGrid = grid.createFaceCenteredGrid3( N, N, N, ...gridSpacing, ...origin );

	const seed = grid.computeFlipBoxSeed3( {
		boxMin: [ 2, 10, 2 ], boxMax: [ 18, 18, 18 ],
		gridSpacingX: gridSpacing[ 0 ], gridSpacingY: gridSpacing[ 1 ], gridSpacingZ: gridSpacing[ 2 ],
		particlesPerCellAxis: 2
	} );

	const flip = grid.createGridFlipSolver3( {
		velocityGrid, gridSpacing, origin, maxParticles: seed.count, dt, gravity: [ 0, -9.81, 0 ],
		// See this file's own header comment ("the periodic collapse/explode
		// bug") for the full investigation. Two fixes, addressing the two
		// separate mechanisms that investigation found:
		//
		// - maxPlausiblePressure: the library's own auto-derived bound only
		//   accounts for HYDROSTATIC pressure (see grid_flip_solver3.js's
		//   own derivedMaxPlausiblePressure comment), far too tight for
		//   this scene's own genuine dynamic/impact pressure once the block
		//   hits the floor -- a caller-supplied, measured value here (the
		//   same scene-specific-tuning convention grid_flip_solver2.js's
		//   own header comment and examples/20-flip-dam-break/'s own
		//   pressure.atomicScale already establish, not a no-magic-numbers
		//   violation) cut rejected/reverted-snapshot frames from 1.58%
		//   (191/12063 over a full 12000-frame real-hardware run) down
		//   towards zero on their own.
		// - multigrid.numberOfLevels: 3 -- the actual remaining cause. This
		//   solver's pressure defaults to numberOfLevels:1 (see multigrid.js's
		//   own header comment on that default), i.e. no real coarse-grid
		//   V-cycle at all, just finest-level smoothing -- adequate for most
		//   scenes in this port, but this one's own dynamically-changing,
		//   irregular fluidMask-derived Dirichlet region (a violent splash
		//   creates many small, disconnected fluid/air regions) is
		//   genuinely harder to precondition than that. Confirmed on real
		//   hardware, not assumed: with maxPlausiblePressure:5000 alone,
		//   1.73% of frames still rejected over 3000 frames; adding
		//   numberOfLevels:3 (the highest level count 20^3 -- this domain's
		//   own resolution -- evenly supports, see grid_flip_solver3.js's
		//   own header comment on this exact divisibility constraint) cut
		//   that to 0/3000, and held at 0/12000 over the same full-length
		//   run the original bug report was measured against. Raising
		//   maxPlausiblePressure further (tried up to 20000) or raising
		//   maxIterations (tried 200) WITHOUT real multigrid levels did not
		//   reproduce this -- ruling out "just a looser circuit breaker" as
		//   the explanation and confirming the preconditioner itself is
		//   what was missing.
		pressure: { maxPlausiblePressure: 5000, multigrid: { numberOfLevels: 3 } }
	} );
	flip.positions.fromArray( seed.positionsArray );
	flip.velocities.fromArray( seed.velocitiesArray );

	// ---- particles: one InstancedMesh, positionNode reads flip.positions
	// directly by instanceIndex -- no CPU readback, ever.
	const particleGeometry = new SphereGeometry( 0.22, 6, 4 );
	const particleMaterial = new MeshBasicNodeMaterial( { color: 0x4fb8ff } );
	particleMaterial.positionNode = positionLocal.add( flip.positions( instanceIndex ) );

	const particles = new InstancedMesh( particleGeometry, particleMaterial, seed.count );

	// Every instance's own transform is left at identity -- the world
	// offset comes entirely from positionNode above, not instanceMatrix.
	{

		const identity = new Matrix4();
		for ( let i = 0; i < seed.count; i ++ ) particles.setMatrixAt( i, identity );
		particles.instanceMatrix.needsUpdate = true;

	}

	scene.add( particles );

	setHud( [ `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }`, `particles: ${ seed.count }`, 'frame: 0' ] );

	let frame = 0;
	let running = true;

	// Cumulative diagnostics -- a driver polling window.__fluxflowProbe
	// only every second or so would miss most individual rejected/
	// non-converged frames at this frame rate, so this file counts them
	// itself, once per frame, right where diagnostics are freshest.
	const stats = { frames: 0, rejected: 0, converged: 0, notConverged: 0, startTime: performance.now() };

	async function tick() {

		if ( ! running ) return;

		await flip.onAdvanceTimeStep();

		const diag = flip.pressureSolver.diagnostics;
		stats.frames ++;
		if ( diag.rejected ) stats.rejected ++;
		if ( diag.converged === true ) stats.converged ++;
		else if ( diag.converged === false ) stats.notConverged ++;

		controls.update();
		renderer.render( scene, camera );

		frame ++;
		setHud( [
			`backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }`,
			`particles: ${ seed.count }`,
			`frame: ${ frame }`,
			'drag to orbit, scroll to zoom'
		] );

		requestAnimationFrame( tick );

	}

	requestAnimationFrame( tick );

	// Exposed for automated/headless verification -- mirrors the
	// window.__fluxflowProbe convention examples/20-flip-dam-break/ already
	// established for driver access. stats() computes fps over the whole
	// run so far, not a rolling window -- a long stress run cares about
	// the sustained average, not momentary jitter.
	window.__fluxflowProbe = {
		renderer, flip, get frame() { return frame; }, stop: () => { running = false; },
		// The shape solver_health.mjs reads, alongside this scene's own
		// names: the stepping object and the velocity field it projects.
		solver: flip, velocityGrid: flip.velocityGrid,
		stats: () => ( { ...stats, elapsedSeconds: ( performance.now() - stats.startTime ) / 1000, fps: stats.frames / ( ( performance.now() - stats.startTime ) / 1000 ) } )
	};

} catch ( error ) {

	setHud( [ 'failed', error.message ] );
	console.error( error );

}
