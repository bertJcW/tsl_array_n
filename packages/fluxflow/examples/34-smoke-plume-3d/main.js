// Second representative 3D rendering example, covering the smoke/fire
// family (createGridSmokeSolver3 -- grid_fire_solver3.js shares the same
// density/temperature/buoyancy machinery, see that file's own header
// comment, so this one example exercises the mechanism both depend on).
//
// Rendering approach, same GPU-resident philosophy as examples/33's own
// FLIP point cloud, applied to a GRID field instead of particles: one
// InstancedMesh instance per CELL of the density field (not per particle --
// this solver has no particles), positioned at that cell's own world
// center and coloured/faded by that cell's own density value, both read
// directly from the solver's storage buffers inside the material's node
// graph. No volume raymarcher, no per-frame readback -- the user chose
// this over a fragment-shader raymarch specifically to avoid that new
// piece of infrastructure (see this session's own scoping discussion).
//
// One wrinkle particles didn't have: density/temperature are each a
// 2-field ping-pong (grid_smoke_solver3.js's own `current` getter flips
// which physical buffer is "now" every frame -- see that file's header
// comment for why: a force() closure built once at construction can't
// re-target a JS reference, so it needs a live GPU-readable flag instead).
// A NodeMaterial's positionNode/colorNode is exactly that same shape of
// problem -- built once, evaluated every frame -- so this file keeps its
// own live parity flag (mirroring smoke.density.current === smoke.density.
// stateA in plain JS each frame, same as every existing 2D smoke example's
// own injection-target check) and selects between stateA/stateB inside the
// colour node with it, rather than binding to one fixed buffer.
//
// *** Run on real WebGPU hardware -- confirmed working, one real bug found
// and fixed along the way. ***
//
// First run (no outflow, just an open no-flux top wall) rendered as a
// near-solid lit-up cube by ~9.5 simulated seconds (frame 190), not a
// plume -- traced to a real gap, not a rendering bug: an "open" wall lets
// velocity pass through but gives density nowhere to actually go, and with
// a continuous source (density decays at only 0.001/frame) and no exit,
// the whole closed room slowly floods. examples/17-smoke-fire/'s own
// domain already solves this with a real outflow strip (see that file's
// own header comment) -- this file was missing the 3D equivalent, added
// above (`outflowTop`) once the cause was confirmed by running this exact
// scene without it, not assumed from reading the 2D file alone. With the
// outflow in place, frame ~184 shows a clean rising plume (bright dense
// source at the bottom widening into a cone toward the open top); by frame
// ~485 (24+ simulated seconds of continuous, undiminished injection into a
// mostly-closed room with one small vent) density has spread through much
// of the domain again -- expected given how little of the domain that one
// opening covers, not a recurrence of the same bug, and the early frame is
// the more representative "plume" shot. Density stayed finite and bounded
// (max ~1.0-1.1) across the whole run.

import * as tsl_array_n from 'tsl_array_n';
import { grid } from 'fluxflow';
import { float, int, max, clamp, vec3, instanceIndex, positionLocal } from 'three/tsl';
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

	const N = 16;
	const gridSpacing = [ 1, 1, 1 ];
	const origin = [ 0, 0, 0 ];
	const dt = 0.05;

	const scene = new Scene();
	scene.background = new Color( 0x0b0b12 );

	const camera = new PerspectiveCamera( 50, window.innerWidth / window.innerHeight, 0.1, 500 );
	camera.position.set( N * 1.9, N * 1.4, N * 1.9 );

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

	const velocityGrid = grid.createFaceCenteredGrid3( N, N, N, ...gridSpacing, ...origin );

	// A real outflow at the top, not just an open (no-flux-constraint-free)
	// wall -- matches examples/17-smoke-fire/'s own domain exactly (see
	// that file's own header comment, "a tall, otherwise closed box with an
	// outflow strip"). An open wall alone lets velocity pass through but
	// gives density nowhere to actually go; without this the source's own
	// continuous injection (density decays at only 0.001/frame) has no
	// exit and slowly fills the whole closed room instead of forming a
	// plume -- found by running this exact scene without an outflow first,
	// not assumed. The outflow box sits well above and wider than the
	// domain (matching example 17's own "padding falls away upward" idea)
	// so its own Dirichlet-zero pressure ring falls entirely outside the
	// simulated region, not through the middle of it.
	const outflowTop = grid.createSDFOutflow3( N, N, N, ...gridSpacing, ...origin );
	outflowTop.addShape( grid.box( [ N / 2, N + 20, N / 2 ], [ N * 2, 22, N * 2 ] ) );

	const smoke = grid.createGridSmokeSolver3( {
		velocityGrid, gridSpacing, origin, dt,
		outflows: outflowTop,
		// Open top -- matches examples/17-smoke-fire/'s own domain, so the
		// plume can actually leave the top instead of pooling at a closed
		// ceiling.
		closedDomainBoundaryFlag: grid.DIRECTION_ALL_3D & ~grid.DIRECTION_UP,
		pressure: { multigrid: { numberOfLevels: 3 }, maxIterations: 60 }
	} );

	// Explicit clear -- see grid_smoke_solver2.js's own header comment on
	// why this isn't done internally (needs tsl_array_n.init() to have
	// already run).
	smoke.density.stateA.clear();
	smoke.density.stateB.clear();
	smoke.temperature.stateA.clear();
	smoke.temperature.stateB.clear();

	// ---- source: a small ball of hot, dense gas at the bottom centre,
	// re-injected every frame -- same createSourceKernel(stateGrid,
	// sourceValue) shape as examples/17-smoke-fire/'s own createSourceKernel,
	// one axis wider.
	const sourceX = N / 2, sourceY = 1.5, sourceZ = N / 2, sourceRadius = 1.8;
	const sourceDensity = 1, sourceTemperature = 1;

	function createSourceKernel( stateGrid, sourceValue ) {

		return tsl_array_n.kernel( stateGrid.dataSize, ( i, j, k ) => {

			const pos = stateGrid.dataPosition( i, j, k );
			const dx = pos.x.sub( sourceX );
			const dy = pos.y.sub( sourceY );
			const dz = pos.z.sub( sourceZ );
			const inSource = dx.mul( dx ).add( dy.mul( dy ) ).add( dz.mul( dz ) ).lessThan( sourceRadius * sourceRadius );

			stateGrid.data( i, j, k ).assign( max( stateGrid.data( i, j, k ), inSource.select( float( sourceValue ), float( 0 ) ) ) );

		} );

	}

	const injectDensityA = createSourceKernel( smoke.density.stateA, sourceDensity );
	const injectDensityB = createSourceKernel( smoke.density.stateB, sourceDensity );
	const injectTemperatureA = createSourceKernel( smoke.temperature.stateA, sourceTemperature );
	const injectTemperatureB = createSourceKernel( smoke.temperature.stateB, sourceTemperature );

	// ---- rendering: one instance per cell, position from the cell's own
	// dataPosition (identical for stateA/stateB -- both share the same
	// origin/spacing), colour/opacity from whichever slot is live this
	// frame via parityLive (see this file's own header comment).
	const cellShape = smoke.density.stateA.resolution;
	const cellCount = cellShape[ 0 ] * cellShape[ 1 ] * cellShape[ 2 ];

	const parityLive = tsl_array_n.array0( 'float' );
	parityLive.fromArray( new Float32Array( [ 0 ] ) ); // 0 = stateA current, matches activeIsA's own initial true

	const nx = int( cellShape[ 0 ] ), ny = int( cellShape[ 1 ] );
	const ix = int( instanceIndex ).mod( nx );
	const iy = int( instanceIndex ).div( nx ).mod( ny );
	const iz = int( instanceIndex ).div( nx.mul( ny ) );

	const cellPosition = smoke.density.stateA.dataPosition( ix, iy, iz );
	const densityHere = parityLive().equal( 0 ).select(
		smoke.density.stateA.data( ix, iy, iz ),
		smoke.density.stateB.data( ix, iy, iz )
	);
	const densityClamped = clamp( densityHere, float( 0 ), float( 1 ) );

	// depthWrite:true (not the usual soft-cloud recipe's false) -- found
	// necessary on real hardware, not assumed. A first version used
	// depthWrite:false and flickered badly; a per-cell density time-series
	// check (both near the source and at the plume's own turbulent edge)
	// showed the underlying density signal itself is smooth frame to
	// frame, ruling out the solver -- which leaves depth-sort instability
	// across thousands of overlapping, unsorted, alpha-blended
	// InstancedMesh instances (three.js draws instances in a fixed index
	// order, not back-to-front, and depthWrite:false means nothing ever
	// occludes anything consistently) as the cause. Writing depth means
	// whichever instance's fragment reaches a pixel first (front-to-back,
	// the normal opaque draw order) wins that pixel outright instead of
	// blending with everything behind it -- a harder, less cloud-like
	// edge than true depth-sorted translucency would give, accepted here
	// over a flickering render. (An alphaTest/opaque-discard alternative
	// was also tried and rendered nothing at all -- not chased further,
	// this smaller change was sufficient.)
	const particleGeometry = new SphereGeometry( 0.42, 6, 4 );
	const particleMaterial = new MeshBasicNodeMaterial();
	particleMaterial.positionNode = positionLocal.mul( densityClamped ).add( cellPosition );
	particleMaterial.colorNode = vec3( 0.75, 0.82, 1.0 ).mul( densityClamped.add( 0.15 ) );
	particleMaterial.opacityNode = densityClamped;
	particleMaterial.transparent = true;
	particleMaterial.depthWrite = true;

	const cells = new InstancedMesh( particleGeometry, particleMaterial, cellCount );

	{

		const identity = new Matrix4();
		for ( let i = 0; i < cellCount; i ++ ) cells.setMatrixAt( i, identity );
		cells.instanceMatrix.needsUpdate = true;

	}

	scene.add( cells );

	setHud( [ `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }`, `cells: ${ cellCount }`, 'frame: 0' ] );

	let frame = 0;
	let running = true;

	// See examples/33-flip-dam-break-3d/'s own stats comment -- same
	// per-frame cumulative counter, same reason.
	const stats = { frames: 0, rejected: 0, converged: 0, notConverged: 0, startTime: performance.now() };

	async function tick() {

		if ( ! running ) return;

		await smoke.onAdvanceTimeStep();

		const diag = smoke.solver.pressureSolver.diagnostics;
		stats.frames ++;
		if ( diag.rejected ) stats.rejected ++;
		if ( diag.converged === true ) stats.converged ++;
		else if ( diag.converged === false ) stats.notConverged ++;

		const usingA = smoke.density.current === smoke.density.stateA;
		if ( usingA ) { injectDensityA(); injectTemperatureA(); } else { injectDensityB(); injectTemperatureB(); }
		parityLive.fromArray( new Float32Array( [ usingA ? 0 : 1 ] ) );

		controls.update();
		renderer.render( scene, camera );

		frame ++;
		setHud( [
			`backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }`,
			`cells: ${ cellCount }`,
			`frame: ${ frame }`,
			'drag to orbit, scroll to zoom'
		] );

		requestAnimationFrame( tick );

	}

	requestAnimationFrame( tick );

	window.__fluxflowProbe = {
		renderer, smoke, get frame() { return frame; }, stop: () => { running = false; },
		// The shape solver_health.mjs reads, alongside this scene's own
		// names: the stepping object and the velocity field it projects.
		solver: smoke, velocityGrid: smoke.velocityGrid,
		stats: () => ( { ...stats, elapsedSeconds: ( performance.now() - stats.startTime ) / 1000, fps: stats.frames / ( ( performance.now() - stats.startTime ) / 1000 ) } )
	};

} catch ( error ) {

	setHud( [ 'failed', error.message ] );
	console.error( error );

}
