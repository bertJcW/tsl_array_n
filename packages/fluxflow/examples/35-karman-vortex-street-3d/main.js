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
// *** The obstacle is a spanwise rod again, and the case against it was
// measured on a broken solver ***
//
// A Karman vortex street is what a bluff body spanning the flow sheds. A
// sphere does not shed one -- its wake is a hairpin/ring structure, not an
// alternating street -- so for the scene this file is named after, the
// obstacle has to be a rod across the full Z extent. primitive_sdf3.js has
// no cylinder primitive, so it is a box with a square cross-section, the
// same approximation this file started with.
//
// It was replaced by a sphere by an earlier session, on what looked like
// strong evidence: with a box the flow "exploded", a sphere of comparable
// size recovered from the same transients, and more CG iterations made the
// box case fail *sooner* -- read at the time as CG converging toward a bad
// answer, and pinned on fractionInsideSdf's linear interpolation between
// two SDF samples, which a box's gradient discontinuity at every edge
// genuinely does violate.
//
// All of it was measured while the multigrid preconditioner was carrying
// neither the Dirichlet mask nor the collider's face weights to its coarse
// levels, i.e. while every frame's pressure solve was being cut off
// mid-iteration, so none of it is evidence about the obstacle. But the box
// is NOT stable on the fixed solver either, and an earlier version of this
// comment claimed it was, on the strength of a ten-minute run whose closing
// readings were all bounded. Read from frame zero instead, and repeated
// five times from scratch with identical numbers every time:
//
//   frame 532   CG breaks down (pAp-growth), 19 times over the next frames
//   frame 535   the pressure circuit breaker begins rejecting, 15 frames
//   frame 540   every velocity component sits at this solver's own clamp
//   frame ~560  the solve converges again and the field recovers
//
// The run that was called stable was measuring the flow that came out the
// far side of that. Its bounded maxU of ~17.5 is not a healthy scene's
// number; it is what this one settles into afterwards.
//
// A sphere does not do this. Run the same way it never rejects a frame and
// never reaches the clamp: maxU ramps smoothly from 2 to ~18 between frames
// 800 and 960 and then holds, with the interior staying at 2 to 2.5 and
// only the two outflow columns large. Both obstacles end up with those same
// outflow columns, which is the separate defect described further down;
// only the rod adds a bang on the way.
//
// It also does what the sphere could not, and this is measured rather than
// looked at -- though note that all of it was measured after frame 532, so
// it describes the flow this scene recovers into rather than one that ran
// cleanly from the start. The sphere's wake is essentially steady -- a shear layer with
// maxV at 1.54 and only mild fluctuation. The rod's sheds periodically:
// sampling transverse velocity on the rod's own centre line over 4293
// frames gives a regular oscillation with a period of ~270 frames at
// x=22, x=30 and x=40 alike (2, 3.3 and 5 rod widths downstream), which
// with D=6 and U=2 is a Strouhal number of 0.22 -- the textbook value for
// a cylinder. Right behind the rod, at x=16, the signal is weaker (std
// 0.18 against 1.29 at x=30) and less regular, which is the wake not
// having rolled up yet rather than a disagreement.
//
// That the period is the same at three widely separated probes is not by
// itself enough, since this scene is known to reflect disturbances off its
// outflow (below), and a reflection would also show up as a periodic
// signal. What separates them is direction: cross-correlating the probes
// puts x=30 a consistent +117 frames behind x=22 and x=40 the same +117
// behind x=30, i.e. the pattern travels DOWNSTREAM at 1.37 and 1.71
// units/time, 0.69 and 0.86 of the free stream. A reflection travels the
// other way and would lag negative.
//
// Worth knowing before quoting that Strouhal number at anyone: the domain
// is 8 rod widths long and the rod blocks a quarter of the channel height,
// and blockage that severe normally raises St. The number establishes that
// periodic shedding is there; it is not a calibration.
//
// tools: karman_shedding.mjs at the repo root is what produced all of the
// above, and is the honest way to answer "is it shedding" for this or any
// similar scene -- two screenshots taken a few hundred frames apart cannot,
// which is how an earlier claim in this file's own history got made on
// evidence that did not support it.
//
// What breaks at frame 532 is the pressure system becoming roughly three
// orders of magnitude harder, not the iteration cap being too low. Freezing
// the scene at frame 524 and re-solving that one system every way
// available: MGPCG reaches a relative residual of 1.8e-3 in 100 iterations,
// 6.9e-4 in 600 and 1.9e-4 in 3000, against a 1e-5 target -- while frame
// 510, fourteen frames earlier, converges in 8. Jacobi manages 1.1e-2 in
// 3000 and unpreconditioned CG 5.5e-2, so multigrid is still much the best
// of them and it is the system that went wrong, not the preconditioner.
//
// Why it gets that much harder that fast is not known. The standing
// suspect, a hypothesis with no measurement behind it yet, is this
// obstacle's own sharp edges: fractionInsideSdf estimates a face's open
// fraction by linear interpolation between two SDF samples, a box's SDF has
// a gradient discontinuity along every edge, and a nearly closed face
// carries a nearly singular row -- which is why sdf_collider3.js floors
// those weights at 0.01 at all, following jet's own kMinWeight.
//
// The concern about linear face fractions on sharp geometry is real and is
// not retracted -- fractionInsideSdf does assume the SDF varies smoothly
// between the two samples straddling a face, and a box's does not at its
// edges (the kinks are why primitive_sdf3.js's box() both is and must be
// piecewise; that is the formula being correct, not a bug). What is
// retracted is that this is what made the scene diverge.

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
	const halfWidth = 3; // half the rod's square cross-section, so a 6x6 rod across the full Z extent -- see this file's own header comment for why this is a rod and not the sphere an earlier session replaced it with
	const cylCenterX = 10;
	const cylCenterY = NY / 2 - 3; // deliberate asymmetry, same reasoning as example 16's own offset -- breaks the top/bottom symmetry a centred obstacle would otherwise preserve indefinitely
	const cylCenterZ = NZ / 2;
	const vorticityColorScale = 2.5; // measured, and re-measured twice: 15 came off a field pinned at the velocity clamp and renders an almost empty box against a healthy one, and 1.5 was right for the sphere this scene briefly used. With the rod, |vorticity.z| runs to 8.4 domain-wide with its 99th percentile at 2.0 and its 95th at 1.3, so 2.5 shows the wake without saturating most of it

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
		new BoxGeometry( halfWidth * 2, halfWidth * 2, NZ ),
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
	// Z half-extent NZ, not NZ/2: the rod runs past both Z walls rather than
	// stopping flush against them, so no sliver of fluid is left between its
	// end and the wall for the flow to squeeze through.
	collider.addShape( grid.box( [ cylCenterX, cylCenterY, cylCenterZ ], [ halfWidth, halfWidth, NZ ] ) );

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
		// *** The iteration cap was never what was wrong here. ***
		//
		// It was 100 when this paragraph was written and is derived from the grid
		// now (72 for 48^3), against a measured worst of 54 iterations actually
		// spent -- so the point below stands and the number it named has moved.
		//
		// It was 600, arrived at by an earlier session raising it in stages
		// (40, 100, 200, 300, 600) because each rise pushed the frame at
		// which this scene blew up further out, and 600 pushed it past
		// ~4700. That reading -- "a materially harder Poisson problem for
		// CG than 2D, not a coding bug in any single file" -- was wrong.
		// The pressure solve was not slow. It was not converging at all, on
		// any frame, at any iteration count: unpreconditioned CG on the
		// same system reached a relative residual of 2e-6 while MGPCG sat
		// at 4e-2 after 300 iterations, which is a broken preconditioner
		// and not a hard problem.
		//
		// The cause was in multigrid.js: a Dirichlet mask was evaluated at
		// the finest level only, so every coarse level solved a zero-flux
		// problem that had no way to represent an error field vanishing at
		// this scene's outflow -- and that whole-domain, low-frequency mode
		// is exactly what the coarse levels exist to supply. The file said
		// so itself, filed under convergence speed; in 2D it costs 6
		// iterations against 16, and in 3D it costs everything. Each level
		// now carries a coarsened mask (see that file's own level-mask
		// comment, and packages/fluxflow/sandbox/poisson-3d-dirichlet/,
		// which measures the operator and the preconditioner against a
		// system that is consistent by construction).
		//
		// With that fixed, this scene's solve converges in roughly 11-30
		// iterations on the frames it converges on. 100 is a cap chosen to
		// be several times that, so that a frame hitting it is a signal
		// rather than routine.
		//
		// *** The second half of the same fix, and what it leaves ***
		//
		// Carrying the mask down was not enough on its own: the scene then
		// converged on most frames, missed on the rest, and saturated at
		// the velocity clamp within a few hundred frames anyway. The
		// remaining piece was the other scope cut in the same file --
		// faceWeights also stopped at level 0, and a collider reaches the
		// pressure system only through those weights, so the preconditioner
		// could not see this scene's obstacle at all. Coarsening them too
		// is what makes this scene steady.
		//
		// Measured over a 10-minute headless run, 18,414 frames: velocity
		// bounded from roughly frame 1700 onward (maxV and maxW hold at
		// 1.54 and never move), pressure peaking at 0.43, no rejected
		// frames, nothing non-finite, and net flux through the middle of
		// the domain at 1153.9 against a fixed inflow of 1152.0 -- the mass
		// balance that used to read -50000. It also runs about three times
		// faster, at ~30 fps rather than ~10, because a solve that
		// converges in 11-30 iterations replaces one that ground through
		// its whole iteration budget every frame.
		//
		// Two things are worth knowing before reading this scene's numbers:
		//
		// 1. Roughly a quarter of frames still do not reach the 1e-5
		// relative tolerance within 100 iterations. That is not what the
		// old failure was -- those frames miss narrowly and the field stays
		// bounded across 18,000 of them -- but it is not nothing either.
		//
		// 2. The two outflow-boundary u columns (x = 47 and x = 48) sit at
		// ~17.5 rather than the ~2.2 the rest of the domain runs at, which
		// is where this scene's outflow flux being ~8.7x its inflow comes
		// from. It is a fixed point, not a drift: 17.65 at frame 1301,
		// 17.49 at 1677, 17.52 at 7688. Those faces are inside the outflow
		// region, so they are Dirichlet-pinned in pressure and receive no
		// correction, and the convective boundary condition leaves them
		// reading each other -- the outflow SDF's gradient points *toward*
		// the fluid, so grid_outflow_solver3.js's own `pt.sub(n * h)` puts
		// the "upstream" sample downstream of the face. Flipping that sign
		// is not the answer and this has now been tested on a working
		// solver rather than a broken one: with the flip the whole domain
		// blows up, pressure at 2e4 including at the inflow end. The
		// artefact stays in those two columns and the interior is unharmed,
		// which the mid-domain flux above is the direct evidence for.
		//
		// Numbers in the header comment above that predate the
		// preconditioner fix are not evidence of anything about this scene
		// any more. That includes both flux findings (the ~9x outflow
		// overshoot, and the collider-free case reading exactly 0 at the
		// boundary face) and, less obviously, the box-vs-sphere conclusion:
		// every one of those runs was made while every frame's pressure
		// solve was being cut off mid-iteration. The sphere is kept because
		// it is a better approximation of 2D's own cylinder regardless, but
		// "a box collider cannot be stabilised here" is now an untested
		// claim rather than a measured one.
		pressure: {
			multigrid: { numberOfLevels: 4, numberOfSmoothingIterationsDown: 3, numberOfSmoothingIterationsUp: 3, numberOfCoarsestIterations: 30 },
			tolerance: 1e-5,
			// maxIterations was 100. The derived cap is LOWER -- 72 for 48^3, since
			// mantaflow's factor is 1 in 3D rather than 4 -- and that is still
			// above the 54 iterations this scene was measured spending at its
			// worst. Verified rather than assumed, because a reduction is the
			// direction that can truncate.
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
