// Third representative 3D example, and the first PURE grid-method one --
// createGridSolver3 directly (no particles, unlike examples/33's own FLIP,
// and no ping-ponged density/temperature, unlike examples/34's own smoke)
// -- inflow + outflow + a real collider + MGPCG pressure, the 3D
// counterpart of examples/16-karman-vortex-street/'s own pipeline. See
// that file's own header comment for the honest expectation-setting this
// port's own inviscid, no-vorticity-confinement solver needs before
// judging the result: "does this look like a vortex street", not "does it
// match a specific Re's Strouhal number" -- unchanged in 3D.
//
// *** What's genuinely 3D here, not just a mechanical port ***
//
// Vorticity is now a genuine 3D VECTOR (grid_math3.js's own
// faceCenteredCurlAtCenter3), not 2D's single signed scalar. This file
// still visualizes only its Z component -- the one whose alternating
// sign downstream of the obstacle is the shedding signature 2D's own
// example looks for -- rather than the full 3D vector.
//
// Rendering: same per-cell InstancedMesh point-cloud technique as
// examples/34-smoke-plume-3d/'s own density visualization, applied to
// vorticity.z instead -- one instance per cell, sized by |vorticity.z|,
// coloured by its sign (red positive / blue negative, matching example
// 16's own diverging-colormap convention for the same reason: sign is the
// whole point here). depthWrite:true from the start, not the
// depthWrite:false "soft cloud" recipe examples/34 tried first and found
// flickered -- see that file's own header comment for the full
// investigation; applied here directly rather than re-discovering it.
//
// *** The obstacle is a sphere, not the box-cornered rod this file
// started with -- a real, root-caused fix, not a cosmetic choice. ***
//
// The first version used a box (primitive_sdf3.js has no native cylinder
// primitive) stretched along the full Z extent, approximating 2D's own
// circular cylinder as a square-cross-section rod. Left running, the
// flow periodically "exploded" -- a user-reported symptom, root-caused
// via a long real-hardware investigation (see this file's own
// pressure.maxIterations comment below for the *other* half of this same
// investigation, the force/outflow-related convergence issue found
// first). Once that fix (maxIterations:300) still left this scene
// pinned at its own velocity clamp indefinitely -- unlike the force-only
// repro, which settled into a genuine bounded equilibrium -- a further
// isolation swept every other variable (obstacle size, whether it
// touched the domain's own Z walls, sphere vs box) and found the one
// that mattered: a SPHERE collider (identical position, comparable size,
// same everything else) recovers from its own transient spikes into
// bounded, decaying behavior; the box does not, regardless of size, of
// whether it touches the domain walls, or of how many CG iterations it's
// given (more iterations made the box case fail *faster*, not slower --
// a real, confirmed-on-real-hardware sign that CG was converging toward
// a bad answer, not merely converging slowly).
//
// The actual mechanism: sdf_collider3.js's own face-marking step (see
// grid_blocked_boundary_condition_solver3.js's own makeMarkAndProject)
// estimates how much of a velocity face lies inside the collider via
// level_set_utils.js's own fractionInsideSdf -- a LINEAR interpolation
// between two SDF samples straddling that face, a standard, widely-used
// technique (Bridson, *Fluid Simulation for Computer Graphics*) that
// assumes the SDF varies smoothly between those two points. A sphere's
// SDF genuinely does. A box's does not: its distance function has a
// real discontinuity in its gradient at every edge and corner (kinks
// there are precisely why box(), primitive_sdf3.js's own textbook
// signed-distance formula, both is and must be piecewise, not the bug),
// and this rod put four such edges running the entire length of the
// domain -- far more edge-adjacent cells than a compact box would have.
// 2D's own example 16 approximates its circle as a 48-sided polygon
// (makeCirclePolygon), not a literal sharp square -- shallow enough per-
// corner that this same linear-fraction assumption never meaningfully
// breaks there, which is why the 2D reference this file is based on
// never surfaced this at all. This is a genuine, inherent limitation of
// the linear-fraction technique for sharp/elongated geometry, not
// something a quick formula fix resolves -- documented in
// primitive_sdf3.js's own header comment for the next caller reaching
// for box() as a collider, not just fixed quietly here.

import * as tsl_array_n from 'tsl_array_n';
import { grid } from 'fluxflow';
import { float, int, abs, clamp, vec3, instanceIndex, positionLocal } from 'three/tsl';
import {
	Scene, PerspectiveCamera, InstancedMesh, SphereGeometry, BoxGeometry, MeshBasicNodeMaterial,
	EdgesGeometry, LineSegments, LineBasicMaterial, Matrix4, Color, Mesh
} from 'three/webgpu';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

const hud = document.querySelector( '#hud' );

function setHud( lines ) {

	hud.textContent = lines.join( '\n' );

}

setHud( [ 'initializing…' ] );

try {

	const renderer = await tsl_array_n.init( { allowFallback: true } );

	// 48x24x24 -- all three axes divisible by 8 (2^3), so multigrid can
	// coarsen a full 4 levels without grid_math3.js's own divisibility
	// guard tripping (see grid_flip_solver3.js's own header comment on
	// this exact class of construction-time error, found the hard way
	// during this session's own FLIP investigation).
	const NX = 48, NY = 24, NZ = 24;
	const gridSpacing = [ 1, 1, 1 ];
	const origin = [ 0, 0, 0 ];
	const dt = 0.05;
	const inflowSpeed = 2;
	const pushStrength = 0.05; // small whole-domain push, offsetting numerical dissipation -- same reasoning as example 16's own mode-0 force
	const radius = 3; // sphere radius -- see this file's own header comment ("box vs sphere") for why this is a sphere, not the rod this file started with
	const cylCenterX = 10;
	const cylCenterY = NY / 2 - 3; // deliberate asymmetry, same reasoning as example 16's own offset -- breaks the top/bottom symmetry a centred obstacle would otherwise preserve indefinitely
	const cylCenterZ = NZ / 2;
	const vorticityColorScale = 15; // measured, not example 16's own 0.3 -- this scene's own 3D vortex shedding generates stronger peak vorticity near the obstacle than 2D's smooth-cylinder scale accounts for; this value was picked by looking at a real run, not guessed

	const scene = new Scene();
	scene.background = new Color( 0x0b0b12 );

	const camera = new PerspectiveCamera( 45, window.innerWidth / window.innerHeight, 0.1, 500 );
	camera.position.set( NX * 0.55, NY * 2.2, NZ * 2.6 );

	const controls = new OrbitControls( camera, renderer.domElement );
	controls.target.set( NX / 2, NY / 2, NZ / 2 );
	controls.enableDamping = true;
	controls.update();

	const domainOutline = new LineSegments(
		new EdgesGeometry( new BoxGeometry( NX, NY, NZ ) ),
		new LineBasicMaterial( { color: 0x334455 } )
	);
	domainOutline.position.set( NX / 2, NY / 2, NZ / 2 );
	scene.add( domainOutline );

	const obstacleMesh = new Mesh(
		new SphereGeometry( radius, 24, 16 ),
		new MeshBasicNodeMaterial( { color: 0xf87171 } )
	);
	obstacleMesh.position.set( cylCenterX, cylCenterY, cylCenterZ );
	scene.add( obstacleMesh );

	window.addEventListener( 'resize', () => {

		camera.aspect = window.innerWidth / window.innerHeight;
		camera.updateProjectionMatrix();
		renderer.setSize( window.innerWidth, window.innerHeight );

	} );

	// ---- solver -----------------------------------------------------

	const velocityGrid = grid.createFaceCenteredGrid3( NX, NY, NZ, ...gridSpacing, ...origin );

	const collider = grid.createSDFStaticCollider3( NX, NY, NZ, ...gridSpacing, ...origin );
	collider.addShape( grid.sphere( [ cylCenterX, cylCenterY, cylCenterZ ], radius ) );

	// Same padding technique as example 16's own makeWallStripPolygon --
	// see that file's own header comment for the real, confirmed-on-real-
	// hardware gradient-direction bug this guards against (an outflow/
	// inflow SDF's own gradient near a domain edge, without generous
	// padding past it, can point the wrong way).
	const OUTER_MARGIN = 1000;

	function wallSlab( innerX, outerX ) {

		const cx = ( innerX + outerX ) / 2;
		const hx = Math.abs( outerX - innerX ) / 2;
		return grid.box( [ cx, NY / 2, NZ / 2 ], [ hx, NY + OUTER_MARGIN, NZ + OUTER_MARGIN ] );

	}

	const inflow = grid.createSDFInflow3( NX, NY, NZ, ...gridSpacing, ...origin, { velocity: [ inflowSpeed, 0, 0 ], mode: 'set' } );
	inflow.addShape( wallSlab( 2, - OUTER_MARGIN ) );

	const outflow = grid.createSDFOutflow3( NX, NY, NZ, ...gridSpacing, ...origin );
	outflow.addShape( wallSlab( NX - 2, NX + OUTER_MARGIN ) );

	function force() {

		return vec3( pushStrength, 0, 0 );

	}

	const closedDomainBoundaryFlag = grid.DIRECTION_ALL_3D & ~grid.DIRECTION_LEFT & ~grid.DIRECTION_RIGHT;

	const solver = grid.createGridSolver3( {
		velocityGrid, gridSpacing, origin, force, collider,
		inflows: inflow, outflows: outflow, closedDomainBoundaryFlag, dt,
		// collider forwarding to the advection stage is NOT automatic --
		// see grid_solver2.js's own header comment (this port's
		// established gotcha, unchanged in 3D): the top-level `collider`
		// option only reaches the boundary-condition solver and the force
		// stage, not the semi-Lagrangian advection solver's own tunnelling
		// clamp, unless it is also named here.
		advection: { collider },
		// Real multigrid coarsening (a 48^~ grid needs it, not
		// numberOfLevels:1) and a relaxed relative tolerance -- same
		// scene-specific-accuracy-requirement reasoning as example 16's
		// own header comment on this exact option (a caller's accuracy
		// requirement, not an internal constant tuned to make the solver
		// merely work).
		//
		// *** maxIterations:600, not the original 40 -- root-caused, not
		// guessed, in two stages. A real-hardware investigation (triggered
		// by a user report: the flow looked stable for a while, then
		// "exploded" starting from the outflow side) found the true cause
		// via a minimal repro -- a spatially uniform force in an otherwise
		// perfectly Z-symmetric channel, no obstacle at all -- which
		// should physically keep W-velocity at ~0 but instead grew it to
		// the same order of magnitude as U/V. Systematically ruled out:
		// the outflow velocity extrapolation (disabling it made things
		// WORSE, not better -- it's protective); tricubic vs trilinear
		// advection interpolation (trilinear delayed the failure but
		// still failed); stronger multigrid smoothing at a low iteration
		// cap (made it fail faster, not slower). What actually fixed that
		// minimal repro: raising maxIterations alone -- 40 diverges by
		// ~frame 175, 100 by ~575, 200 recovers several times before
		// still failing by ~775, 300 held for the full 900-frame test.
		//
		// That 300 was NOT enough once the real collider was added back in
		// (see this file's own header comment, "the obstacle is a
		// sphere" section, for the box-vs-sphere half of this
		// investigation) -- collider+force together needed more than
		// either alone: 300 still pinned at this solver's own velocity
		// clamp indefinitely with a sphere present, while 600 shows a
		// transient spike (briefly hitting the clamp around frame 150-300)
		// that recovers into bounded, decaying values by frame 325+,
		// the same qualitative "real transient, not runaway" signature
		// the sphere-alone case already showed. This solver's own CG
		// never reported converged:true at ANY tested iteration cap in
		// this scene -- meaning every frame's pressure solve was being cut
		// off mid-iteration, leaving a small amount of real, uncorrected
		// divergence that compounded frame over frame until it blew up.
		// This is a materially harder Poisson problem for CG than
		// examples/16-karman-vortex-street/'s own comparable-cell-count 2D
		// domain, not a coding bug in any single file -- documented here
		// rather than silently accepted, since 600 iterations is a real,
		// measured per-frame cost (see this file's own real-hardware fps
		// numbers, if recorded).
		//
		// *** maxIterations:600 delays the failure by ~27x (from ~175
		// frames to ~4700+) but does NOT fully eliminate it -- read this
		// before treating the scene as solved. ***
		//
		// A further real-hardware investigation, prompted directly by a
		// user pushing back on "still eventually gets stuck" and asking
		// specifically about the outflow mechanism, measured net volumetric
		// flux at the true inflow face (x=0, always exactly 1152 = fixed
		// inflowSpeed x NY x NZ, by construction) against the true outflow
		// face (x=NX) over a 5200-frame run. Two real, concrete findings,
		// neither of which is a resolved fix:
		//
		// 1. Even in the LONG "stable-looking" plateau (roughly frame
		// 600-4600), outflowFlux settles at ~10600 -- nearly 9x the fixed
		// inflow rate -- while mid-domain flux sits at ~2100, itself
		// already ~1.8x inflow. This gap is too large to be explained by
		// the constant force's own legitimate cumulative acceleration
		// over the fluid's transit time (a back-of-envelope bound puts
		// that around 1.15 units of extra velocity, nowhere near a 9x
		// flux multiplier) -- meaning even the frames this file's own
		// fps/stability numbers call "stable" are very likely sitting at
		// a self-consistent but NON-physical fixed point of the coupled
		// force+pressure+outflow system, not the true solution CG simply
		// hasn't found yet. The eventual frame-~4700 failure is this
		// fixed point finally getting perturbed hard enough (very likely
		// by a genuine shedding event) that CG can no longer track it.
		//
		// 2. A collider-free, force-free version of this exact scene (pure
		// inflow -> obstacle-free channel -> outflow, no continuous push)
		// showed outflowFlux reading EXACTLY 0 for the entire length of a
		// separate 5200-frame run, while mid-domain flux still grew to
		// ~38000 (also clamp-pinned) -- meaning the true boundary face
		// was receiving zero net correction from the outflow mechanism at
		// all in that configuration, a stronger and more direct symptom
		// than the 9x-overshoot case above. Checked directly, not
		// inferred: the outflow SDF's own sampled value and gradient at
		// that exact boundary face were both confirmed correct (sdf=-1.5,
		// correctly "inside"; gradient length exactly 1, not degenerate).
		// The one concrete hypothesis this ruled out: `upstreamPt`'s own
		// sign (`pt.sub(n.mul(spacing))`, grid_outflow_solver3.js) is NOT
		// backwards -- flipping it to `.add(...)` and re-running the full
		// force+collider scene made the failure occur roughly 30x SOONER
		// (frame ~140 instead of ~4700), the opposite of a fix, confirming
		// the existing sign (unchanged, matching grid_outflow_solver2.js's
		// own real-hardware-confirmed 2D convention) is the right one.
		// Reverted immediately; grid_outflow_solver3.js carries no residual
		// change from this experiment.
		//
		// Neither finding points to a specific wrong line still to fix --
		// both are evidence that this scene sits closer to a genuine
		// numerical-robustness edge (a strongly-forced, obstacle-wake,
		// long-domain-fill-up 3D configuration) than a resolvable coding
		// bug, at least with the investigation done so far. Recorded in
		// full because "300/600 helps a lot" was reported as a fix once
		// already and turned out not to be one -- the honest status is
		// "substantially mitigated, not eliminated."
		pressure: {
			multigrid: { numberOfLevels: 4, numberOfSmoothingIterationsDown: 3, numberOfSmoothingIterationsUp: 3, numberOfCoarsestIterations: 30 },
			tolerance: 1e-5,
			maxIterations: 600
		}
	} );

	// ---- vorticity, computed fresh from velocity every frame ----------

	const cellShape = [ NX, NY, NZ ];
	const cellCount = NX * NY * NZ;
	const vorticityZ = tsl_array_n.arrayN( 'float', cellShape );
	vorticityZ.fromArray( new Float32Array( cellCount ) );

	const computeVorticity = tsl_array_n.kernel( cellShape, ( i, j, k ) => {

		const curl = grid.faceCenteredCurlAtCenter3( velocityGrid.dataU, velocityGrid.dataV, velocityGrid.dataW, velocityGrid.gridSpacing, i, j, k, cellShape );
		vorticityZ( i, j, k ).assign( curl.z );

	} );

	// ---- rendering: one instance per cell, same per-cell InstancedMesh
	// point-cloud technique as examples/34-smoke-plume-3d/'s own density
	// visualization -- see this file's own header comment.

	const originNode = vec3( origin[ 0 ], origin[ 1 ], origin[ 2 ] );
	const gridSpacingNode = vec3( gridSpacing[ 0 ], gridSpacing[ 1 ], gridSpacing[ 2 ] );
	const cellCenterOrigin = originNode.add( gridSpacingNode.mul( 0.5 ) );

	const nx = int( NX ), ny = int( NY );
	const ix = int( instanceIndex ).mod( nx );
	const iy = int( instanceIndex ).div( nx ).mod( ny );
	const iz = int( instanceIndex ).div( nx.mul( ny ) );

	const cellPosition = cellCenterOrigin.add( vec3( ix, iy, iz ).mul( gridSpacingNode ) );
	const w = vorticityZ( ix, iy, iz );
	const magnitude = clamp( abs( w ).div( vorticityColorScale ), float( 0 ), float( 1 ) );

	const particleGeometry = new SphereGeometry( 0.4, 6, 4 );
	const particleMaterial = new MeshBasicNodeMaterial();
	particleMaterial.positionNode = positionLocal.mul( magnitude ).add( cellPosition );
	particleMaterial.colorNode = w.greaterThanEqual( 0 ).select( vec3( 1, 0.25, 0.25 ), vec3( 0.25, 0.45, 1 ) ).mul( magnitude.add( 0.1 ) );
	particleMaterial.opacityNode = magnitude;
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

		await solver.onAdvanceTimeStep( dt );
		computeVorticity();

		const diag = solver.pressureSolver.diagnostics;
		stats.frames ++;
		if ( diag.rejected ) stats.rejected ++;
		if ( diag.converged === true ) stats.converged ++;
		else if ( diag.converged === false ) stats.notConverged ++;

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
		renderer, solver, velocityGrid, vorticityZ, get frame() { return frame; }, stop: () => { running = false; },
		stats: () => ( { ...stats, elapsedSeconds: ( performance.now() - stats.startTime ) / 1000, fps: stats.frames / ( ( performance.now() - stats.startTime ) / 1000 ) } )
	};

} catch ( error ) {

	setHud( [ 'failed', error.message ] );
	console.error( error );

}
