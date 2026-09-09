// Demonstrates grid.createGridTwoPhaseFlipSolver2 in its MISCIBLE mode --
// dye injected into water. See that file's own header comment, section
// "Miscible mixing", for the design and the papers.
//
// The contrast with examples/24-two-phase-bubble-rise/ is the point of having
// both, and it is entirely a matter of configuration rather than a different
// solver: that scene has a 100:1 density ratio and a concentration that is only
// ever 0 or 1, this one has a ~1.05:1 ratio and a concentration that varies
// continuously and is allowed to blend. Everything else -- the particles, the
// variable-density projection, the pinned closed-domain pressure system -- is
// the same code running with different numbers in it.
//
// *** What to actually look at ***
//
// **dye density is the physical dial, not a look control.** At 1.000 the dye is
// a purely passive tracer: the density field is uniform, beta is 1 everywhere,
// and the projection reduces *exactly* to the constant-density one, so the dye
// is carried by the flow and drawn but exerts nothing. Push it up and the dye
// sinks and plumes; push it down and it rises. That is the same
// variable-density mechanism that lifts the bubble in example 24, just gentle
// enough to look like liquid rather than gas.
//
// **mixing and fade are the two endings.** `mixing` blends each particle toward
// the concentration around it, so the dye softens and eventually goes uniform.
// `fade` decays it toward zero, so the dye disappears instead. Both default to
// off; this scene starts with a little mixing so the effect is visible without
// touching anything.
//
// *** Pressure tuning -- and a wrong analogy that took a blow-up to catch ***
//
// This scene was first given examples/20-flip-dam-break/'s values
// (`atomicScale: 256`, `maxPlausiblePressure: 100`) on the reasoning that it is
// the same 64x64 gravity-driven size. That analogy was wrong in the one way that
// mattered, and the result was max particle speed going 0.3 -> 1.4 -> 252 over
// 120 frames.
//
// Example 20 has a FREE SURFACE. Its pressure is referenced to the air right
// above the liquid, so it stays small -- around 6 or 7. This tank is SEALED and
// full, with pressure pinned at a single cell at the top, so pressure has to
// build hydrostatically all the way down the column: rho*g*h is about
// 1 * 9.81 * 64, i.e. of order 600. That is two orders of magnitude larger, and
// it breaks both settings at once -- `maxPlausiblePressure: 100` would reject
// the correct answer, and `atomicScale: 256` overflows the CG solve's
// fixed-point dot-product accumulator on magnitudes that size, which is what
// actually produced the divergence.
//
// Note this is NOT the same cause as example 24's identical-looking failure,
// even though the fix is the same number. There the magnitudes were large
// because beta carried a 100:1 density ratio; here beta is within 15% of 1 and
// the magnitudes are large because a sealed column has nowhere to reference its
// pressure to. Two different routes to the same overflow, which is worth
// separating: the lesson is not "two-phase needs a small atomicScale", it is
// "work out what magnitudes YOUR scene actually produces".
//
// `maxPlausiblePressure: 5000` is roughly 8x the expected hydrostatic peak --
// generous enough not to reject a correct solve, tight enough to still catch a
// genuine runaway.

// *** KNOWN LIMITATION: this scene is correct but not yet convincing ***
//
// Read this before assuming the solver is at fault. As configured, the dye
// blends at its edges and drifts slowly downward, and that is all -- there is no
// plume, no roll-up, none of the filament structure that makes ink-in-water
// worth looking at. Everything measurable is healthy (concentration conserved to
// within 0.05% over 540 frames on real WebGPU, the pressure solve converging,
// bounded velocities), so this is a scene-design problem, not a solver bug.
//
// The cause is the sealed tank, and it is a genuine tension rather than an
// oversight:
//
//   * A sealed box completely full of incompressible fluid can only circulate.
//     There is no free surface to rise or fall, so a mild density contrast has
//     very little room to actually move anything.
//   * The obvious fixes all trade against stability. Lowering velocityDamping
//     below the library default, or raising the density contrast past about
//     1.1:1, both make the plume livelier AND make the pressure solve stop
//     converging -- and every divergence observed while building this scene
//     followed a frame where `converged` went false. Measured, not assumed.
//
// The promising direction, not yet tried: host the concentration in the
// FREE-SURFACE solver (grid_flip_solver2.js) instead of this all-fluid one. An
// open tank with air above it can slosh and circulate freely, which is what the
// look actually needs, and the concentration machinery is independent of which
// solver carries it -- it is a per-particle attribute plus a density blend,
// neither of which cares whether there is a free surface. That is a bigger
// change than tuning this scene, which is why it is written down here rather
// than half-done.

import * as tsl_array_n from 'tsl_array_n';
import { grid } from 'fluxflow';

const particlesCanvas = document.querySelector( '#outParticles' );
const concentrationCanvas = document.querySelector( '#outConcentration' );
const statusEl = document.querySelector( '#status' );
const perfEl = document.querySelector( '#perf' );
const dyeDensityInput = document.querySelector( '#dyeDensity' );
const dyeDensityValueEl = document.querySelector( '#dyeDensityValue' );
const mixingInput = document.querySelector( '#mixing' );
const mixingValueEl = document.querySelector( '#mixingValue' );
const fadeInput = document.querySelector( '#fade' );
const fadeValueEl = document.querySelector( '#fadeValue' );
const injectButton = document.querySelector( '#inject' );
const resetButton = document.querySelector( '#reset' );

function status( text, isErr ) {

	statusEl.textContent = text;
	statusEl.className = isErr ? 'err' : '';

}

// 64x64 deliberately, matching examples/24-two-phase-bubble-rise/ rather than
// going finer: that resolution and particle count (16384) are the ones actually
// measured on hardware in this project, so performance here is a known quantity
// instead of a guess. Dye filaments do reward more resolution -- raise both if
// your GPU has the headroom.
const NX = 64;
const NY = 64;
const dt = 1 / 60;

const WATER_DENSITY = 1;
const INITIAL_DYE_DENSITY = 1.05;
const INITIAL_MIXING = 0.005;

// *** Why the dye is placed at rest rather than squirted in -- a real failure ***
//
// An earlier version of this scene gave the dyed particles a downward velocity
// as well as a concentration, on the reasoning that "injection" means a jet and
// that shear is what makes ink look like ink. It destabilised the simulation:
// within a few seconds every particle collapsed into the domain's minimum
// corner. That failure is worth recording twice over.
//
// First, the physics. This tank is SEALED and COMPLETELY FULL of an
// incompressible fluid. There is no free surface and no compressibility, so
// there is nowhere for injected volume to go -- a jet is not a thing that can
// happen here, and asking the projection for one asks it to solve a problem with
// no good answer. A real dropper works because the glass it squirts into has a
// free surface. Circulation is fine in a sealed box; net injection is not.
//
// Second, how it hid. The advection step clamps a particle's traced position
// into the domain, and `clamp` turns a NaN into the clamp's minimum rather than
// propagating it. So a velocity blow-up did not show up as a NaN anywhere a
// finite-value check could see it -- it showed up as every particle quietly
// stacked in one corner, with all the readback still perfectly finite. Worth
// knowing before trusting a non-finite check on positions alone.
//
// So the motion here is driven by density alone, which is what the scene is
// actually about. The dye is heavier than the water, and it falls -- an
// unstable heavy-over-light layer, which is the classic way to get a plume with
// a rolled-up cap and trailing filaments, and needs no injected momentum at all.

// Where a fresh squirt of dye lands: a disc near the top, slightly off-centre so
// the plume is not perfectly symmetric and actually curls.
const INJECT_CENTER = [ NX * 0.5, NY * 0.80 ];
const INJECT_RADIUS = NY * 0.14;

const diagnosticInterval = 60;
const DRAW_INTERVAL = 2;

function dyeAt( [ x, y ] ) {

	const dx = x - INJECT_CENTER[ 0 ];
	const dy = y - INJECT_CENTER[ 1 ];
	const d = Math.sqrt( dx * dx + dy * dy );

	// A soft edge rather than a hard disc: a hard edge quantises visibly against
	// the particle spacing, and a dye blob does not have one anyway.
	return Math.max( 0, Math.min( 1, ( INJECT_RADIUS - d ) / ( INJECT_RADIUS * 0.5 ) ) );

}

try {

	const renderer = await tsl_array_n.init( { canvas: document.createElement( 'canvas' ), allowFallback: true } );
	status( `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }` );

	const velocityGrid = grid.createFaceCenteredGrid2( NX, NY, 1, 1, 0, 0 );

	// The whole tank is water -- there is no free surface and no air in this
	// scene, it is a closed vessel filled to the brim. That means every cell is
	// fluid, which is exactly the case the solver's `pressurePin` exists for.
	//
	// Worth knowing what that costs, because it shapes what this scene can look
	// like: a sealed box of incompressible fluid can *circulate* freely but it
	// cannot have net inflow anywhere, so the injected jet does not simply drive
	// straight down -- the projection immediately turns most of it into a
	// recirculating plume, with the displaced water returning up the sides. That
	// is correct physics rather than a limitation to work around (it is why a
	// real ink drop in a *sealed, completely full* vial behaves quite differently
	// from one in an open glass), but it does mean the injection speed here buys
	// less downward travel than the number suggests.
	const seed = grid.computeTwoPhaseBoxSeed( {
		boxMin: [ 0, 0 ],
		boxMax: [ NX, NY ],
		gridSpacingX: 1, gridSpacingY: 1,
		particlesPerCellAxis: 2,
		concentrationAt: dyeAt
	} );

	const dyeDensityUniform = tsl_array_n.array0( 'float' );
	dyeDensityUniform.fromArray( new Float32Array( [ INITIAL_DYE_DENSITY ] ) );

	const mixingUniform = tsl_array_n.array0( 'float' );
	mixingUniform.fromArray( new Float32Array( [ INITIAL_MIXING ] ) );

	const fadeUniform = tsl_array_n.array0( 'float' );
	fadeUniform.fromArray( new Float32Array( [ 0 ] ) );

	const solver = grid.createGridTwoPhaseFlipSolver2( {
		velocityGrid,
		gridSpacing: [ 1, 1 ],
		origin: [ 0, 0 ],
		maxParticles: seed.count,
		dt,
		// Neutral names, not gasDensity/liquidDensity: the dye is allowed to be
		// LIGHTER than the water, which the gas/liquid vocabulary forbids for
		// good reason (a "gas" heavier than its "liquid" means the tags are
		// backwards) but which is perfectly ordinary here.
		ambientDensity: WATER_DENSITY,
		componentDensity: dyeDensityUniform(),
		mixing: mixingUniform(),
		fade: fadeUniform(),

		// *** Both of these override a solver default, and both had to, for the
		// same reason: the defaults are tuned for a free-surface liquid that
		// should settle, and this scene wants the opposite. ***
		//
		// The first version of this example inherited them and the plume simply
		// died -- the dye ended up sitting almost exactly where it was injected,
		// which looked like a broken solver and was in fact a correctly working
		// one being asked to damp everything.
		//
		// velocityDamping is left at the library default here, NOT lowered, and
		// that is a deliberate retreat from an earlier attempt. Lowering it does
		// make the plume livelier -- 2% per frame compounds to about 3% of the
		// motion surviving three seconds, which is a lot to ask of an ink plume
		// -- but it also removes the margin that keeps this scene stable when the
		// pressure solve has a bad frame. See the stability note in the header.
		velocityDamping: 0.02,
		// gasFlipRatio's default of 0.90 exists because the light phase in a
		// gas/liquid scene is where FLIP noise does the most damage, so it gets a
		// more dissipative (more PIC) blend. Here the two components differ in
		// density by 8%, not 100x -- there is no fragile light phase to protect,
		// and the extra dissipation only blurs the filaments that are the whole
		// point. Setting it equal to flipRatio switches the per-component
		// distinction off entirely.
		gasFlipRatio: 0.97,

		// tolerance is loosened from the library's 1e-5 and the iteration budget
		// doubled, and this is the fix for a real failure rather than a tweak.
		// Running undamped exposed that the solve was not converging in 100
		// iterations at 1e-5 -- which the default velocityDamping had been
		// quietly covering up, by bleeding off the energy that the leftover
		// divergence kept adding. Undamped there is nothing to cover it, so the
		// residual compounds and every particle ends up pinned against a wall
		// within a few seconds.
		//
		// 1e-5 on a divergence residual is a tight ask for a graphics scene;
		// 1e-4 is converged for anything visible here and reaches it comfortably.
		// A small velocityDamping is kept as well, an order of magnitude below
		// the default -- enough to stop FLIP noise accumulating over a long run,
		// far too little to flatten a plume in the first few seconds.
		pressure: { atomicScale: 1, maxPlausiblePressure: 5000, tolerance: 1e-4, maxIterations: 200 }
	} );

	function seedScene() {

		solver.positions.fromArray( seed.positionsArray );
		solver.velocities.fromArray( seed.velocitiesArray ); // at rest -- see above
		solver.concentration.fromArray( seed.phasesArray );

	}

	seedScene();

	// Re-stamping dye onto the particles currently inside the injection disc.
	// Deliberately a CPU round trip rather than a kernel: it happens on a button
	// press, not per frame, and a one-off readback is far less machinery than a
	// dispatch that would exist only to be used occasionally.
	async function injectDye() {

		const [ positionsData, concentrationData ] = await Promise.all( [
			solver.positions.toArray(),
			solver.concentration.toArray()
		] );

		for ( let p = 0; p < concentrationData.length; p ++ ) {

			const added = dyeAt( [ positionsData[ p * 2 ], positionsData[ p * 2 + 1 ] ] );
			if ( added <= 0 ) continue;

			concentrationData[ p ] = Math.min( 1, concentrationData[ p ] + added );

		}

		// Concentration only -- velocity is deliberately left alone, for the
		// reason in the header comment: stamping momentum into a sealed, full,
		// incompressible tank is what broke this scene once already.
		solver.concentration.fromArray( concentrationData );

	}

	// ---------------------------------------------------------------- drawing

	const particlesCtx = particlesCanvas.getContext( '2d' );
	const PARTICLE_CANVAS_SIZE = particlesCanvas.width;
	const particleScale = PARTICLE_CANVAS_SIZE / NX;

	const concentrationCtx = concentrationCanvas.getContext( '2d' );
	const concentrationImage = concentrationCtx.createImageData( NX, NY );

	function clamp01( v ) {

		return Math.min( 1, Math.max( 0, v ) );

	}

	function drawConcentration( data ) {

		for ( let j = 0; j < NY; j ++ ) {

			for ( let i = 0; i < NX; i ++ ) {

				const t = clamp01( data[ i + NX * j ] );
				const pixel = ( ( NY - 1 - j ) * NX + i ) * 4;
				concentrationImage.data[ pixel ] = Math.round( 255 * t );
				concentrationImage.data[ pixel + 1 ] = Math.round( 40 + 120 * t );
				concentrationImage.data[ pixel + 2 ] = Math.round( 90 + 60 * ( 1 - t ) );
				concentrationImage.data[ pixel + 3 ] = 255;

			}

		}

		concentrationCtx.putImageData( concentrationImage, 0, 0 );

	}

	// Radii larger than the 0.5-world-unit particle spacing so neighbours
	// overlap and the water reads as a body rather than a dot lattice -- the
	// same lesson example 24's first version had to learn the hard way.
	const PARTICLE_SPACING_PX = particleScale * 0.5;
	const RADIUS = PARTICLE_SPACING_PX * 0.95;

	// Clear water is a dark blue-green; dye ramps to a hot magenta. Drawing the
	// clear water at all (rather than leaving it as background) matters: it is
	// what makes the dye read as being *inside* something.
	function dyeColor( c ) {

		const t = clamp01( c );
		return `rgb(${ Math.round( 18 + t * 237 ) },${ Math.round( 46 + t * 30 ) },${ Math.round( 62 + t * 140 ) })`;

	}

	function drawParticles( positionsData, concentrationData ) {

		particlesCtx.fillStyle = '#0a1418';
		particlesCtx.fillRect( 0, 0, PARTICLE_CANVAS_SIZE, PARTICLE_CANVAS_SIZE );

		// Two passes, clear water first: with dye drawn last a thin filament
		// stays visible instead of being overpainted by whatever water particle
		// happened to come after it in the buffer.
		for ( const dyePass of [ false, true ] ) {

			for ( let p = 0; p < concentrationData.length; p ++ ) {

				const c = concentrationData[ p ];
				if ( ( c > 0.02 ) !== dyePass ) continue;

				particlesCtx.fillStyle = dyeColor( c );
				particlesCtx.beginPath();
				particlesCtx.arc(
					positionsData[ p * 2 ] * particleScale,
					( NY - positionsData[ p * 2 + 1 ] ) * particleScale,
					RADIUS, 0, Math.PI * 2
				);
				particlesCtx.fill();

			}

		}

	}

	// ---------------------------------------------------------------- diagnostics

	// Total dye is the number worth watching. With mixing and fade both off it
	// should be very nearly conserved -- particles carry their concentration and
	// nothing creates or destroys it, so a drift means the resample pass is
	// moving dye around (see the solver's note on the Houdini reseeding lesson).
	// With fade on it should fall smoothly; with mixing on it should hold while
	// the *spread* narrows.
	function dyeStats( concentrationData, velocitiesData ) {

		let total = 0, peak = 0, dyed = 0, maxSpeed = 0;

		for ( let p = 0; p < concentrationData.length; p ++ ) {

			const c = concentrationData[ p ];
			total += c;
			if ( c > peak ) peak = c;
			if ( c > 0.02 ) dyed ++;

			const vx = velocitiesData[ p * 2 ], vy = velocitiesData[ p * 2 + 1 ];
			const speed = Math.sqrt( vx * vx + vy * vy );
			if ( speed > maxSpeed ) maxSpeed = speed;

		}

		return { total, peak, dyed, maxSpeed };

	}

	let frame = 0;
	let nanDetected = false;

	function checkForNonFinite( data, frameNumber ) {

		if ( nanDetected ) return;

		for ( let i = 0; i < data.length; i ++ ) {

			if ( ! Number.isFinite( data[ i ] ) ) {

				nanDetected = true;
				status( `non-finite value detected at frame ${ frameNumber } (index ${ i }, value ${ data[ i ] })`, true );
				console.error( `fluxflow dye-injection: non-finite value at frame ${ frameNumber }, index ${ i }:`, data[ i ] );
				return;

			}

		}

	}

	const FPS_WINDOW = 30;
	const frameTimes = [];
	let lastFrameTime = performance.now();

	function updatePerf( stats ) {

		const now = performance.now();
		frameTimes.push( now - lastFrameTime );
		lastFrameTime = now;
		if ( frameTimes.length > FPS_WINDOW ) frameTimes.shift();

		const avgMs = frameTimes.reduce( ( a, b ) => a + b, 0 ) / frameTimes.length;
		perfEl.textContent = `fps: ${ ( 1000 / avgMs ).toFixed( 1 ) } | particles: ${ seed.count }`
			+ ( stats ? ` | dyed: ${ stats.dyed }` : '' );

	}

	let lastStats = null;

	async function animate() {

		updatePerf( lastStats );
		await solver.onAdvanceTimeStep();

		if ( ! nanDetected && frame % DRAW_INTERVAL === 0 ) {

			const [ positionsData, concentrationData, cellConcentration, velocitiesData ] = await Promise.all( [
				solver.positions.toArray(),
				solver.concentration.toArray(),
				solver.liquidFraction.toArray(),
				solver.velocities.toArray()
			] );

			checkForNonFinite( positionsData, frame );
			checkForNonFinite( concentrationData, frame );

			if ( ! nanDetected ) {

				lastStats = dyeStats( concentrationData, velocitiesData );

				if ( frame % diagnosticInterval === 0 ) {

					console.log(
						`fluxflow dye-injection [frame ${ frame }] ` +
						`converged=${ solver.pressureSolver.diagnostics.converged } ` +
						`rejected=${ solver.pressureSolver.diagnostics.rejected } | ` +
						`totalDye=${ lastStats.total.toFixed( 2 ) } peak=${ lastStats.peak.toFixed( 3 ) } ` +
						`dyedParticles=${ lastStats.dyed } maxSpeed=${ lastStats.maxSpeed.toFixed( 2 ) }`
					);

				}

				drawParticles( positionsData, concentrationData );
				drawConcentration( cellConcentration );

			}

		}

		frame ++;
		requestAnimationFrame( animate );

	}

	// ---------------------------------------------------------------- controls

	dyeDensityInput.addEventListener( 'input', () => {

		const v = parseFloat( dyeDensityInput.value );
		dyeDensityUniform.fromArray( new Float32Array( [ v ] ) );
		dyeDensityValueEl.textContent = v.toFixed( 3 );

	} );

	mixingInput.addEventListener( 'input', () => {

		const v = parseFloat( mixingInput.value );
		mixingUniform.fromArray( new Float32Array( [ v ] ) );
		mixingValueEl.textContent = v.toFixed( 3 );

	} );

	fadeInput.addEventListener( 'input', () => {

		const v = parseFloat( fadeInput.value );
		fadeUniform.fromArray( new Float32Array( [ v ] ) );
		fadeValueEl.textContent = v.toFixed( 3 );

	} );

	injectButton.addEventListener( 'click', () => { injectDye(); } );

	resetButton.addEventListener( 'click', () => {

		seedScene();
		velocityGrid.clear();
		frame = 0;
		nanDetected = false;
		status( `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' } — reset` );

	} );

	requestAnimationFrame( animate );

} catch ( error ) {

	status( error.message, true );
	console.error( error );

}
