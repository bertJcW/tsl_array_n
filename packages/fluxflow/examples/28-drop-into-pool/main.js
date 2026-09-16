// A dyed drop falling into a pool -- the standard drop-into-a-liquid
// experiment, built on grid.createGridFlipSolver2's free-surface solver with
// `carryConcentration` and variable-density coupling.
//
// *** Why this scene exists alongside examples/26-dye-free-surface/ ***
//
// Example 26 collapses two dyed columns into each other. It is a good
// transport test -- the colour boundary stays razor sharp through violent
// folding, which is the claim it was built to make -- but as a *look* it is
// over in about a second: the columns collide, throw up a sheet, and settle
// into two flat layers that then sit there. The density coupling it also
// demonstrates is real but nearly invisible in that layout, because the scene
// is left-right symmetric and both columns start at the same height.
//
// A drop entering a pool is the opposite: the interesting part starts at
// impact and keeps going. Measured on real WebGPU at the shipped defaults,
// it produces three distinct phases, each recognisable from the real
// experiment:
//
//   frames ~40-120   an impact crater, its whole wall lined with dye
//   frames ~200-320  the crater rebounds into a Worthington jet -- a dyed
//                    spike thrown back up above the surface
//   frames ~400 on   the jet falls back, the dyed mass rolls up into a
//                    mushroom and starts descending
//
// The dye's mean height traces that directly: 66.9 -> 64.9 as the crater
// opens, back up to 73.6 at the top of the jet, then down through 66.3,
// 62.9, 61.3 and still falling at frame 500.
//
// Two mechanisms, and it is worth being clear about which does what, because
// the first version of this scene got it wrong. The *shape* -- crater, jet,
// mushroom -- is all impact: the drop deposits vorticity and the vorticity
// rolls the dye up. The slow *descent* afterwards is the density coupling,
// which is the pressure projection knowing the two liquids weigh different
// amounts; there is no buoyancy force anywhere in this file or in the
// solver. Set the dye density to 1.00 and the first two phases are unchanged
// while the third stops.
//
// *** What it is a test of, not just what it looks like ***
//
// Three claims, each with a number printed in the console:
//
//   `dyeDepth`  -- mean height of the dyed particles. Not monotonic, and it
//                  should not be: it dips as the crater opens, rises with the
//                  jet, and then falls steadily once the mushroom forms. It is
//                  that final descent that the density ratio controls -- at
//                  1.00 it stops, at 1.25 it keeps going. Identical scene,
//                  identical drop, one number changed.
//   `spread`    -- fraction of particles whose concentration is neither near 0
//                  nor near 1. With `mixing` at zero it stays near zero however
//                  far the plume stretches, because a particle carries its own
//                  value exactly and nothing diffuses it. Turn `mixing` up and
//                  watch it climb.
//   `filled`    -- occupied cells, the collapse detector this port's stability
//                  work standardised on. A liquid conserving its volume fills a
//                  roughly constant number of cells; one eating itself fills
//                  steadily fewer.
//
// *** Classical background ***
//
// A dyed drop falling into a still liquid producing a descending vortex ring
// is one of the oldest documented fluid experiments (J. J. Thomson and H. F.
// Newall, "On the formation of vortex rings by drops falling into liquids",
// Proc. R. Soc. London 39, 1885). This is a 2D solver, so what forms here is
// the 2D section of that -- a pair of counter-rotating lobes rather than a
// ring -- and the resemblance is qualitative. Recorded so the claim is not
// mistaken for a quantitative reproduction of the experiment.
//
// URL parameters, all optional:
//   ?density=1.25    starting dye density (the slider's initial value)
//   ?speed=22        the drop's downward speed at t = 0
//   ?radius=6        drop radius in cells
//   ?height=0.92     drop centre height as a fraction of the tank
//   ?pool=0.75       pool depth as a fraction of the tank
//   ?resX=64 ?resY=96  grid resolution
//   ?particlesPerCellAxis=2  particles per cell, per axis (4 per cell at
//                    the default of 2) -- higher raises the concentration
//                    field's own resolution independent of the grid
//   ?targetDt=0.016  per-frame simulated time

// Measurement handle only: `window.__fluxflowProbe` exposes the renderer and the
// solver so a driver can count `renderer.compute()` calls per frame and flip
// the pressure solver's runtime switches (`settings.batchIterations`) inside a
// single run. Nothing here changes what the scene does.

import * as tsl_array_n from 'tsl_array_n';
import { grid, float_guards } from 'fluxflow';
import { texture, uv, uvec2, vec3, vec4, clamp, float, int, atomicAdd, textureStore } from 'three/tsl';
import { Scene, OrthographicCamera, PlaneGeometry, Mesh, StorageTexture, MeshBasicNodeMaterial } from 'three/webgpu';

const canvas = document.querySelector( '#out' );
const statusEl = document.querySelector( '#status' );
const perfEl = document.querySelector( '#perf' );
const densityRatioInput = document.querySelector( '#densityRatio' );
const densityRatioValueEl = document.querySelector( '#densityRatioValue' );
const mixingInput = document.querySelector( '#mixing' );
const mixingValueEl = document.querySelector( '#mixingValue' );
const resetButton = document.querySelector( '#reset' );

function status( text, isErr ) {

	statusEl.textContent = text;
	statusEl.className = isErr ? 'err' : '';

}

const params = new URLSearchParams( location.search );
// A tall tank, not a square one. The scene needs depth far more than it
// needs width: a drop punching into a shallow pool makes a wide crater and
// stops, while the same drop entering a deep column rolls up and keeps
// travelling, which is the part worth watching. Taller also costs nothing
// extra per particle -- the particle count follows the *filled* area, and a
// narrow deep tank holds fewer particles than a square one of the same
// depth.
//
// *** The resolution is not the thing to turn down. ***
//
// Asked to make this scene cheaper, the two obvious knobs are less water
// and a coarser grid, and only one of them works. Measured by the dye's
// lateral extent in drop radii -- the ring spreading is what makes this
// scene worth looking at, and it collapses long before the penetration
// depth does:
//
//   64x96, pool 0.75 (default)    5.4 / 8.4 / 8.7 / 7.3 / 7.5
//   64x96, pool 0.55              5.2 / 7.9 / 7.9 / 6.1 / 6.2
//   64x96, pool 0.45              5.2 / 7.3 / 6.5 / 4.4 / 4.9
//   48x64, pool 0.75              4.7 / 4.7 / 2.2 / 3.5 / 5.1
//   48x64, pool 0.55              5.7 / 6.4 / 3.1 / 2.6 / 4.8
//   40x56, pool 0.50              6.2 / 6.4 / 3.0 / 3.3 / 5.7
//
// Halving the water degrades it gently -- 0.55 costs 26% of the particles
// and stays inside the default's own run-to-run spread, 0.45 visibly
// shrinks the ring. None of it is a stability problem: pool 0.45 and even
// 0.30 run 397/400 and 398/400 frames converged, zero rejections, zero
// non-finite values, and the same peak pressure (15.61) as the deep pool,
// because peak pressure is set by the impact and not by how much water is
// under it. It is purely how wide the ring gets.
//
// Dropping to 48x64 is the one that actually breaks the scene: the ring
// collapses to barely more than the drop's own diameter at any pool depth
// -- and it is the grid, not the drop being under-resolved, since holding
// the drop at the default's 5.76-cell radius on a 48x64 grid collapses the
// same way (5.1 / 6.3 / 5.2 / 2.8 / 3.6). So the pool depth is where the
// savings are if they are wanted, and 64x96 stays either way.
const NX = Number( params.get( 'resX' ) ?? 64 );
const NY = Number( params.get( 'resY' ) ?? 96 );
const targetDt = Number( params.get( 'targetDt' ) ?? 1 / 60 );
const initialDensity = Number( params.get( 'density' ) ?? 1.25 );
const dropRadius = Number( params.get( 'radius' ) ?? NX * 0.09 );
// The pool is deeper than the dye ever needs -- the dye's deepest reach is
// about 2 drop radii below the surface, against a pool 12 radii deep -- so
// it is the obvious place to save particles, and it does save them without
// breaking anything. It stays at 0.75 anyway because this scene exists to
// be looked at, and the ring is widest here. The cost of going shallower is
// measured under "the resolution is not the thing to turn down" above; pair
// any change with `height`, since what matters is the fall in cells (~16)
// rather than the fraction.
const dropHeight = Number( params.get( 'height' ) ?? 0.92 ) * NY;
const poolDepth = Number( params.get( 'pool' ) ?? 0.75 ) * NY;
// Impact speed, downward, given to the drop's particles at t = 0.
//
// *** This is the difference between a scene that shows something and one
// that does not, and it took a measurement to find out. ***
//
// The first version released the drop from rest and let gravity do the work,
// with the dye's own weight expected to drive it down into the pool. Measured
// over 400 frames at a 1.15 density ratio, the dye's mean height fell from 68
// to 40.8 and then essentially stopped -- it landed, spread across the
// surface, and sat there as a compact blob. Correct, and dull.
//
// The reason is physical, not a solver problem: in the real experiment almost
// all of the visible structure comes from the *impact*, which deposits a ring
// of vorticity that then carries the dye down as a mushroom. Buoyancy from a
// 10-15% density difference is a slow secondary effect and cannot produce
// that shape on its own on any timescale worth watching. Giving the drop the
// speed it would have had after a longer fall puts the energy where the real
// experiment has it, without needing a domain tall enough to fall through.
const impactSpeed = Number( params.get( 'speed' ) ?? 22 );

const diagnosticInterval = 60;
const DRAW_INTERVAL = 1;

try {

	// `?profile=1` turns on WebGPU timestamp queries, which is the only
	// way to see how much of a step the GPU is actually busy for. Off by
	// default: three.js then writes a timestamp around every compute
	// pass. tsl_array_n.init forwards anything it does not recognise
	// straight to the WebGPURenderer constructor.
	const profileEnabled = new URLSearchParams( location.search ).get( 'profile' ) === '1';

	// This example's own dye visualisation renders through this same
	// renderer (see "GPU-native rendering" below) rather than through a
	// separate Canvas2D context, so the renderer's canvas has to be the one
	// actually on the page.
	const renderer = await tsl_array_n.init( {
		canvas,
		allowFallback: true,
		trackTimestamp: profileEnabled
	} );
	status( `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }` );

	const velocityGrid = grid.createFaceCenteredGrid2( NX, NY, 1, 1, 0, 0 );

	// Seed the whole domain and keep the two regions wanted, the same
	// carve-out approach example 26 uses -- computeFlipBoxSeed fills a
	// rectangle, and anything shaped differently is a filter on top of it.
	const particlesPerCellAxis = Number( params.get( 'particlesPerCellAxis' ) ?? 2 );

	const seed = grid.computeFlipBoxSeed( {
		boxMin: [ 0, 0 ],
		boxMax: [ NX, NY ],
		gridSpacingX: 1, gridSpacingY: 1,
		particlesPerCellAxis
	} );

	const dropCenterX = NX * 0.5;
	const keptPositions = [];
	const keptConcentration = [];

	for ( let p = 0; p < seed.count; p ++ ) {

		const x = seed.positionsArray[ p * 2 ];
		const y = seed.positionsArray[ p * 2 + 1 ];

		if ( y < poolDepth ) {

			keptPositions.push( x, y );
			keptConcentration.push( 0 );
			continue;

		}

		const dx = x - dropCenterX;
		const dy = y - dropHeight;

		if ( dx * dx + dy * dy < dropRadius * dropRadius ) {

			keptPositions.push( x, y );
			keptConcentration.push( 1 );

		}

	}

	const count = keptConcentration.length;
	const positionsArray = Float32Array.from( keptPositions );
	const concentrationArray = Float32Array.from( keptConcentration );
	const velocitiesArray = new Float32Array( count * 2 );
	const dyedCount = keptConcentration.reduce( ( a, c ) => a + c, 0 );

	// Only the drop moves at t = 0; the pool is at rest.
	for ( let p = 0; p < count; p ++ ) {

		if ( concentrationArray[ p ] > 0.5 ) velocitiesArray[ p * 2 + 1 ] = - impactSpeed;

	}

	// Live, because adaptive dt only works if every kernel that bakes dt in
	// reads the same node -- see grid_adaptive_timestep2.js's own header.
	const dtUniform = tsl_array_n.array0( 'float' );
	dtUniform.fromArray( new Float32Array( [ targetDt ] ) );

	const densityRatioUniform = tsl_array_n.array0( 'float' );
	densityRatioUniform.fromArray( new Float32Array( [ initialDensity ] ) );
	densityRatioInput.value = String( initialDensity );
	densityRatioValueEl.textContent = initialDensity.toFixed( 2 );

	const mixingUniform = tsl_array_n.array0( 'float' );
	mixingUniform.fromArray( new Float32Array( [ 0 ] ) );

	const flip = grid.createGridFlipSolver2( {
		velocityGrid,
		gridSpacing: [ 1, 1 ],
		origin: [ 0, 0 ],
		maxParticles: count,
		dt: dtUniform(),
		maxDt: targetDt,
		carryConcentration: true,
		mixing: mixingUniform(),
		// The pool is the reference (concentration 0); the dye is whatever the
		// slider says. Equal densities is the control case, and the scene is
		// deliberately shipped slightly away from it so the plume is the first
		// thing a visitor sees.
		//
		// This is a per-CELL-average force (see grid_flip_solver2.js's own
		// cellDensity), not a per-particle one -- a stray dyed particle that
		// separates from the bulk plume, sharing its cell mostly with plain
		// water, stops registering as "denser" there and all but stops
		// sinking. Confirmed, not fixed -- see docs/project-history.md's own
		// Open items entry for this example for the investigation.
		ambientDensity: 1,
		componentDensity: densityRatioUniform(),
		// Lower than the 0.02 default. Damping is there to keep a FLIP
		// solver's own noise in check, and at the default it also flattens
		// the impact vortex within a few hundred frames -- which is the one
		// thing this scene is for. Measured: the plume stays legible for
		// thousands of frames at this value and is gone by frame ~400 at
		// 0.02.
		velocityDamping: 0.003,
		// Off, for the reason example 26 records: a relocated particle carries
		// its concentration to its new home, so reseeding blurs exactly the
		// colour boundary this scene exists to show.
		resample: { enabled: false }
		// No pressure options at all: the solver derives its own plausibility
		// bound from dt, gravity and the domain size, and the dot product has
		// no scale to set.
	} );

	function seedScene() {

		flip.positions.fromArray( positionsArray );
		flip.velocities.fromArray( velocitiesArray );
		flip.concentration.fromArray( concentrationArray );

	}

	seedScene();

	const adaptiveTimeStep = grid.createGridAdaptiveTimeStep2( {
		velocityGrid, gridSpacing: [ 1, 1 ], dt: dtUniform, targetDt, courantNumber: 1
	} );

	// One rendered frame is `targetDt` of simulated time, in as many equal
	// substeps as the Courant condition asks for. A drop landing in a pool is
	// the case that actually needs this: impact speed here is several times
	// the settled speed, so the frame that needs splitting is not the frame
	// most of the run is made of.
	async function step() {

		const numSubSteps = await adaptiveTimeStep.update();

		for ( let i = 0; i < numSubSteps; i ++ ) await flip.onAdvanceTimeStep();

		return numSubSteps;

	}

	// ---------------------------------------------------------------- drawing

	// Sized from the grid rather than fixed in the HTML, so a non-square tank
	// is not stretched.
	const PIXELS_PER_CELL = 8;
	canvas.width = NX * PIXELS_PER_CELL;
	canvas.height = NY * PIXELS_PER_CELL;
	canvas.style.width = `${ NX * ( 560 / NY ) }px`;
	canvas.style.height = '560px';

	// tsl_array_n.init() sized the renderer's own targets (including its
	// depth buffer) from window.innerWidth/innerHeight, the right default
	// for a detached compute-only canvas but wrong now that this canvas is
	// the one actually on the page -- its internal buffers have to match
	// the canvas's own backing-store resolution, not the window's. `false`
	// so it does not also overwrite the CSS display size just set above.
	renderer.setSize( canvas.width, canvas.height, false );

	// *** GPU-native rendering: a texture, not per-particle Canvas2D draws ***
	//
	// Two earlier directions were tried and shelved first -- see
	// docs/project-history.md's Open items entry for this example for the
	// full record (per-particle Canvas2D fill() cost, then colour-bucketed
	// batching of it, then a scoped-but-not-built GPU sprite-instancing
	// path). This is the third, and the one that shipped: instead of
	// drawing individual particles at all, `flip.cellConcentration` --
	// already computed every step for the density-coupling force, one
	// value per grid cell -- is written straight into a small GPU texture
	// and a plane samples it. No CPU readback, no Canvas2D, no per-particle
	// anything; the picture this produces is a smooth field rather than
	// discrete dots, which is a real, deliberate look change the user
	// confirmed is fine here.
	//
	// The texture is deliberately grid-resolution (NX x NY), not
	// canvas-resolution: StorageTexture defaults to LinearFilter, so the
	// GPU's own bilinear sampling does the upscaling to canvas size for
	// free, which is exactly the soft/abstract look asked for -- no blur
	// pass to write.
	const dyeTexture = new StorageTexture( NX, NY );

	// Same ramp the old per-particle draw used: rgb(38+t*214, 96+t*42,
	// 140-t*78), deep water to hot dye, normalised to TSL's [0,1] colour
	// space.
	function dyeColorNode( c ) {

		const t = clamp( c, 0, 1 );

		return vec3(
			float( 38 / 255 ).add( t.mul( 214 / 255 ) ),
			float( 96 / 255 ).add( t.mul( 42 / 255 ) ),
			float( 140 / 255 ).sub( t.mul( 78 / 255 ) )
		);

	}

	// Background colour for cells with no fluid in them at all -- without
	// this every cell reads as pool water (concentration 0 is still a
	// valid, coloured value), so the air above the water would render as
	// water instead of the near-black background the old Canvas2D fillRect
	// gave it. flip.fluidMask is the same field markFluidCellsKernel
	// already refreshes every step for exactly this fluid/air distinction.
	const BACKGROUND = vec3( 6 / 255, 8 / 255, 12 / 255 );

	const writeDyeTexture = tsl_array_n.kernel( [ NX, NY ], ( i, j ) => {

		const isFluid = flip.fluidMask( i, j ).greaterThan( 0.5 );
		const color = isFluid.select( dyeColorNode( flip.cellConcentration( i, j ) ), BACKGROUND );

		textureStore( dyeTexture, uvec2( i, j ), vec4( color, 1 ) ).toWriteOnly();

	} );

	// A flat scene: one textured plane, one orthographic camera spanning
	// exactly the grid's own coordinates (0..NX, 0..NY) so no separate
	// pixel<->cell scale factor is needed anywhere in this file the way the
	// Canvas2D version needed `scale`.
	const scene = new Scene();
	const camera = new OrthographicCamera( 0, NX, NY, 0, -1, 1 );

	const dyeMaterial = new MeshBasicNodeMaterial();
	dyeMaterial.colorNode = texture( dyeTexture, uv() );

	const dyePlane = new Mesh( new PlaneGeometry( NX, NY ), dyeMaterial );
	dyePlane.position.set( NX / 2, NY / 2, 0 );
	scene.add( dyePlane );

	function draw() {

		writeDyeTexture();
		renderer.render( scene, camera );

	}

	// *** The NaN guard, without a full per-frame readback ***
	//
	// Rendering no longer needs particle data back on the host at all, so
	// checkForNonFinite can no longer piggyback on a readback draw() was
	// doing anyway. Same shape as grid_pressure_solver2.js's own circuit
	// breaker: one atomic count of bad particles, computed on the GPU and
	// read back as a single int, instead of the whole positions/
	// concentration arrays. isNonFinite2/isNonFinite come from
	// float_guards.js rather than a hand-rolled comparison -- see that
	// file's own header comment for why a NaN check written any other way
	// is not trustworthy on this hardware.
	const badParticleAccum = tsl_array_n.array0( 'int' );
	badParticleAccum.node.toAtomic();

	const checkParticlesKernel = tsl_array_n.kernel( [ count ], ( p ) => {

		const isBad = float_guards.isNonFinite2( flip.positions( p ) ).or( float_guards.isNonFinite( flip.concentration( p ) ) );
		atomicAdd( badParticleAccum(), isBad.select( int( 1 ), int( 0 ) ) );

	} );

	const badParticleZero = new Int32Array( [ 0 ] );

	async function countBadParticlesNow() {

		badParticleAccum.fromArray( badParticleZero );
		checkParticlesKernel();
		const [ badCount ] = await badParticleAccum.toArray();
		return badCount;

	}

	// ---------------------------------------------------------------- diagnostics

	function stats( concentrationData, positionsData ) {

		let mixed = 0, dyeY = 0, dyeN = 0, dyeMinY = Infinity;

		for ( let p = 0; p < concentrationData.length; p ++ ) {

			const c = concentrationData[ p ];
			if ( c > 0.05 && c < 0.95 ) mixed ++;

			if ( c > 0.5 ) {

				const y = positionsData[ p * 2 + 1 ];
				dyeY += y;
				dyeN ++;
				if ( y < dyeMinY ) dyeMinY = y;

			}

		}

		const occupied = new Uint8Array( NX * NY );
		let filled = 0;

		for ( let p = 0; p < concentrationData.length; p ++ ) {

			const i = Math.min( NX - 1, Math.max( 0, Math.floor( positionsData[ p * 2 ] ) ) );
			const j = Math.min( NY - 1, Math.max( 0, Math.floor( positionsData[ p * 2 + 1 ] ) ) );
			const k = i + NX * j;
			if ( occupied[ k ] === 0 ) { occupied[ k ] = 1; filled ++; }

		}

		return {
			filled,
			mixed,
			spread: mixed / concentrationData.length,
			dyeDepth: dyeN ? dyeY / dyeN : 0,
			dyeDeepest: Number.isFinite( dyeMinY ) ? dyeMinY : 0
		};

	}

	let frame = 0;
	let nanDetected = false;

	async function checkForNonFinite( frameNumber ) {

		if ( nanDetected ) return;

		const badCount = await countBadParticlesNow();

		if ( badCount > 0 ) {

			nanDetected = true;
			status( `non-finite value at frame ${ frameNumber } (${ badCount } particle${ badCount === 1 ? '' : 's' })`, true );
			console.error( `fluxflow drop-into-pool: ${ badCount } non-finite particle(s) at frame ${ frameNumber }` );

		}

	}

	const FPS_WINDOW = 30;
	const frameTimes = [];
	let lastFrameTime = performance.now();

	function updatePerf() {

		const now = performance.now();
		frameTimes.push( now - lastFrameTime );
		lastFrameTime = now;
		if ( frameTimes.length > FPS_WINDOW ) frameTimes.shift();

		const avgMs = frameTimes.reduce( ( a, b ) => a + b, 0 ) / frameTimes.length;
		perfEl.textContent = `fps: ${ ( 1000 / avgMs ).toFixed( 1 ) } | ${ NX }x${ NY } | particles: ${ count } (${ dyedCount } dyed) | substeps: ${ adaptiveTimeStep.state.lastNumSubSteps }`;

	}

	// See examples/26-dye-free-surface/'s own hook for why an automated
	// stability run has to be able to stop this loop.
	let driverPaused = false;

	async function animate() {

		if ( driverPaused ) return;

		updatePerf();
		await step();

		if ( ! nanDetected && frame % DRAW_INTERVAL === 0 ) {

			// The NaN guard rides on a single small readback (see
			// countBadParticlesNow) instead of the full positions +
			// concentration arrays draw() used to need every frame -- GPU
			// rendering below needs neither back on the host at all.
			await checkForNonFinite( frame );

			if ( ! nanDetected ) {

				if ( frame % diagnosticInterval === 0 ) {

					// Only the diagnostic log still needs the full arrays,
					// so only it still pays for reading them.
					const [ positionsData, concentrationData ] = await Promise.all( [
						flip.positions.toArray(),
						flip.concentration.toArray()
					] );

					const s = stats( concentrationData, positionsData );
					console.log(
						`fluxflow drop-into-pool [frame ${ frame }] ` +
						`converged=${ flip.pressureSolver.diagnostics.converged } ` +
						`rejected=${ flip.pressureSolver.diagnostics.rejected } | ` +
						`dyeDepth=${ s.dyeDepth.toFixed( 2 ) } deepest=${ s.dyeDeepest.toFixed( 2 ) } ` +
						`spread=${ ( s.spread * 100 ).toFixed( 2 ) }% filled=${ s.filled }`
					);

				}

				draw();

			}

		}

		frame ++;
		requestAnimationFrame( animate );

	}

	densityRatioInput.addEventListener( 'input', () => {

		const v = parseFloat( densityRatioInput.value );
		densityRatioUniform.fromArray( new Float32Array( [ v ] ) );
		densityRatioValueEl.textContent = v.toFixed( 2 );

	} );

	mixingInput.addEventListener( 'input', () => {

		const v = parseFloat( mixingInput.value );
		mixingUniform.fromArray( new Float32Array( [ v ] ) );
		mixingValueEl.textContent = v.toFixed( 3 );

	} );

	resetButton.addEventListener( 'click', () => {

		seedScene();
		velocityGrid.clear();
		frame = 0;
		nanDetected = false;
		status( `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' } — reset` );

	} );

	window.__fluxflowProbe = {
		flip, velocityGrid, stats, seedScene, step, adaptiveTimeStep,
		// Measurement handle only -- see 16-karman-vortex-street's own probe
		// comment. flip.pressureSolver.settings carries the runtime switches.
		renderer,
		// Exposed so a driver that has paused the rAF loop can still render
		// -- which is the only way to check the panel is not blank when
		// the page is not the foreground tab and rAF never fires.
		draw,
		pause: async () => {

			driverPaused = true;
			await new Promise( ( resolve ) => setTimeout( resolve, 100 ) );

		},
		resume: () => {

			driverPaused = false;
			requestAnimationFrame( animate );

		}
	};

	requestAnimationFrame( animate );

} catch ( error ) {

	status( error.message, true );
	console.error( error );

}
