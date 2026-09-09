// Demonstrates grid.createGridFlipSolver2's `carryConcentration` -- a dye
// carried on the particles of an ordinary FREE-SURFACE liquid.
//
// *** Why this scene exists alongside examples/25-dye-injection/ ***
//
// Example 25 puts the same dye in the sealed, all-fluid two-phase solver, and
// it is the honest comparison: that version is stable and correct and visually
// almost inert. A sealed box completely full of an incompressible liquid can
// only circulate -- there is no free surface to rise or fall -- so a dye blob
// in it blurs at the edges and drifts, and every attempt to liven it up (less
// damping, more density contrast) pushed the pressure solve out of convergence.
// That was measured, not assumed; see that file's own header comment.
//
// A free surface removes the constraint entirely. Liquid that can slosh, break
// and fold over is what stretches a dye blob into filaments, and folding is
// free here -- it is what a dam break does anyway. The dye costs almost nothing
// on top: transport is exact because the particle simply carries its value, and
// only the optional `mixing` needs a grid field at all.
//
// *** Variable-density coupling, added after this scene first shipped passive ***
//
// The first version of this example had no density coupling at all: the dye was
// a passive tracer, carried and drawn but exerting nothing, and the honest
// statement at the time was that dye whose weight drives the flow was the
// two-phase solver's job. That gap is now closed -- this solver takes
// `ambientDensity`/`componentDensity` too, so the concentration drives a
// variable-density projection here exactly as it does there.
//
// The density-ratio slider is therefore a real physical dial and a self-check
// at the same time. At 1.000 the two liquids weigh the same, the density field
// is uniform, beta is 1 everywhere, and the projection reduces EXACTLY to the
// constant-density one -- so that setting reproduces the original passive
// behaviour bit for bit, and is the control case. Move it away from 1 and the
// heavier liquid works its way underneath the lighter one, which is
// stratification driven purely by the projection knowing the two weigh
// different amounts. No buoyancy force exists anywhere in this file or the
// solver.
//
// One genuine difference from the sealed two-phase solver, and the reason the
// coupling needed new code rather than a copy: air has no mass. A cell with no
// particles here is air held at p = 0, so a face between liquid and air takes
// the LIQUID's density alone rather than an average with a fluid that is not
// there. See the solver's own comment where faceDensity is defined.
//
// *** Two dyed columns rather than one ***
//
// A single dyed column collapsing produces a pretty splash but the dye stays in
// one connected sheet. Two columns of different colour collapsing INTO each
// other is what actually exercises the mechanism: the collision folds one into
// the other, and because transport is diffusion-free the interface between them
// stays sharp for as long as `mixing` allows -- which is the whole claim this
// example is here to make visible. Turn `mixing` up and watch the boundary
// dissolve; leave it at zero and it stays crisp indefinitely.
//
// Concentration is used as a two-colour mixing fraction here (0 = one dye,
// 1 = the other) rather than as "dye vs clear water", which is the more
// demanding test -- there is nowhere for a mixing error to hide against a
// neutral background.

// *** How the density coupling was actually verified, and why it needed a
// second layout to do it ***
//
// `?layout=layered&density=X` runs a Rayleigh-Taylor test: the two liquids
// stacked, heavier on top, with a small sinusoidal perturbation on the
// interface. `heightGap` (mean height of amber minus mean height of teal) is
// then a direct readout of whether the density difference is reaching the
// projection at all. Measured on real WebGPU, 300 frames each, identical scene
// and identical perturbation, only the density differing:
//
//     density=1.00 (control):  22.42  22.41  22.40  22.40  22.41  22.40   flat
//     density=1.35 (coupled):  22.42  22.42  22.39  22.35  22.28  22.19   falling
//
// The coupled case is monotone AND accelerating (deltas 0.00, -0.03, -0.04,
// -0.07, -0.09), which is the exponential growth signature Rayleigh-Taylor is
// supposed to have. The control is flat. Nothing but the density differs
// between the two runs, so the overturning is the coupling and not the
// perturbation, the free surface, or anything else in the scene.
//
// Two false starts on the way there, both worth keeping:
//
//   * The default `columns` layout cannot test this. It is left-right
//     symmetric with both columns at the same height, so `heightGap` sat at
//     0.00 for 120 frames at a 1.35 ratio and briefly looked like proof the
//     coupling was broken. It is a fine transport test and a useless density
//     test.
//   * The first layered version used a perfectly FLAT interface, which is an
//     unstable *equilibrium* -- it sits there forever regardless of the density
//     ordering, and did, for 300 frames. That is why the perturbation is there.
//
// Both are the same underlying mistake: a test that cannot fail for the reason
// you are testing. Worth naming, because the second one produced a completely
// convincing-looking flat line.

import * as tsl_array_n from 'tsl_array_n';
import { grid } from 'fluxflow';

const canvas = document.querySelector( '#out' );
const statusEl = document.querySelector( '#status' );
const perfEl = document.querySelector( '#perf' );
const densityRatioInput = document.querySelector( '#densityRatio' );
const densityRatioValueEl = document.querySelector( '#densityRatioValue' );
const mixingInput = document.querySelector( '#mixing' );
const mixingValueEl = document.querySelector( '#mixingValue' );
const fadeInput = document.querySelector( '#fade' );
const fadeValueEl = document.querySelector( '#fadeValue' );
const resetButton = document.querySelector( '#reset' );

function status( text, isErr ) {

	statusEl.textContent = text;
	statusEl.className = isErr ? 'err' : '';

}

const NX = 64;
const NY = 64;
const dt = 1 / 60;
const diagnosticInterval = 60;
const DRAW_INTERVAL = 2;

try {

	const renderer = await tsl_array_n.init( { canvas: document.createElement( 'canvas' ), allowFallback: true } );
	status( `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }` );

	const velocityGrid = grid.createFaceCenteredGrid2( NX, NY, 1, 1, 0, 0 );

	// Two columns, one against each wall, leaving the middle and the top empty.
	// The empty region is what makes this a free surface: those cells have no
	// particles, so the solver treats them as air at p = 0 and the liquid is
	// free to move into them.
	const COLUMN_WIDTH = NX * 0.30;
	const COLUMN_HEIGHT = NY * 0.62;

	// Two layouts, because they test different claims and one of them cannot
	// test the other at all.
	//
	//   `columns` (default) -- two dyed columns collapsing into each other. This
	//   is the diffusion-free transport demo: violent folding, and the colour
	//   boundary stays razor sharp because a particle carries its value exactly.
	//   It is a WEAK test of density coupling though, and that is worth stating:
	//   the scene is left-right symmetric and both columns start at the same
	//   height, so even a large density difference produces almost no signal in
	//   mean height over the first few seconds. Measured, not assumed -- at a
	//   1.35 ratio `heightGap` sat at 0.00 for 120 frames here.
	//
	//   `layered` -- the heavier liquid resting ON TOP of the lighter one, across
	//   the full width. That is Rayleigh-Taylor unstable: it MUST overturn, and
	//   it can only overturn if the density difference is actually reaching the
	//   projection. If the coupling were silently doing nothing this layout would
	//   sit there forever, which is exactly what makes it the decisive test.
	//
	// `?layout=layered&density=1.35` is the configuration that check runs under.
	const layout = new URLSearchParams( location.search ).get( 'layout' ) ?? 'columns';
	const layered = layout === 'layered';

	const seed = grid.computeFlipBoxSeed( {
		boxMin: [ 0, 0 ],
		boxMax: [ NX, layered ? NY * 0.70 : COLUMN_HEIGHT ],
		gridSpacingX: 1, gridSpacingY: 1,
		particlesPerCellAxis: 2
	} );

	const keptPositions = [];
	const keptConcentration = [];

	for ( let p = 0; p < seed.count; p ++ ) {

		const x = seed.positionsArray[ p * 2 ];
		const y = seed.positionsArray[ p * 2 + 1 ];

		if ( layered ) {

			keptPositions.push( x, y );
			// Amber (1) on top. With amber heavier this is the unstable
			// configuration; with the density slider at 1 it is just two colours
			// stacked, and nothing should happen.
			//
			// The interface is given a small sinusoidal perturbation, and that is
			// not decoration -- a perfectly flat interface is an unstable
			// EQUILIBRIUM, so an ideal simulation of one sits there forever no
			// matter how wrong the density ordering is. A first run without this
			// held heightGap at 22.43 for 300 frames and briefly looked like the
			// density coupling was doing nothing at all. Every Rayleigh-Taylor
			// setup seeds a perturbation for this reason.
			const interface_ = NY * 0.35 + NY * 0.03 * Math.sin( 2 * Math.PI * x / ( NX * 0.5 ) );
			keptConcentration.push( y > interface_ ? 1 : 0 );

		} else {

			// computeFlipBoxSeed fills a box, so the middle gap is carved out
			// here and the survivors tagged left (0) or right (1).
			if ( x > COLUMN_WIDTH && x < NX - COLUMN_WIDTH ) continue;

			keptPositions.push( x, y );
			keptConcentration.push( x <= COLUMN_WIDTH ? 0 : 1 );

		}

	}

	const count = keptConcentration.length;
	const positionsArray = Float32Array.from( keptPositions );
	const concentrationArray = Float32Array.from( keptConcentration );
	const velocitiesArray = new Float32Array( count * 2 );

	// `?density=1.3` deep-links a starting density, which is also how this
	// scene's two claims get tested non-interactively: density=1 is the control.
	const initialDensity = Number( new URLSearchParams( location.search ).get( 'density' ) ?? 1 );
	const densityRatioUniform = tsl_array_n.array0( 'float' );
	densityRatioUniform.fromArray( new Float32Array( [ initialDensity ] ) );
	densityRatioInput.value = String( initialDensity );
	densityRatioValueEl.textContent = initialDensity.toFixed( 2 );

	const mixingUniform = tsl_array_n.array0( 'float' );
	mixingUniform.fromArray( new Float32Array( [ 0 ] ) );
	const fadeUniform = tsl_array_n.array0( 'float' );
	fadeUniform.fromArray( new Float32Array( [ 0 ] ) );

	const flip = grid.createGridFlipSolver2( {
		velocityGrid,
		gridSpacing: [ 1, 1 ],
		origin: [ 0, 0 ],
		maxParticles: count,
		dt,
		carryConcentration: true,
		mixing: mixingUniform(),
		fade: fadeUniform(),
		// Teal (concentration 0) is the reference; amber (concentration 1) is
		// whatever the slider says. Starting them equal means the scene opens in
		// the passive-tracer case, which is the control.
		ambientDensity: 1,
		componentDensity: densityRatioUniform(),
		// Resampling relocates particles, and a relocated particle carries its
		// concentration with it -- which is exactly the mechanism SideFX's own
		// users disable reseeding to avoid when they need a sharp colour
		// boundary. This scene is about keeping that boundary crisp, so it is
		// off. See the solver's own note on the Houdini lesson.
		resample: { enabled: false },
		// examples/20-flip-dam-break/'s values: same solver, same resolution,
		// same free-surface gravity-driven setup, so the analogy actually holds
		// here (unlike example 25's sealed column, where it did not).
		pressure: { atomicScale: 256, maxPlausiblePressure: 100 }
	} );

	function seedScene() {

		flip.positions.fromArray( positionsArray );
		flip.velocities.fromArray( velocitiesArray );
		flip.concentration.fromArray( concentrationArray );

	}

	seedScene();

	// ---------------------------------------------------------------- drawing

	const ctx = canvas.getContext( '2d' );
	const SIZE = canvas.width;
	const scale = SIZE / NX;
	const RADIUS = scale * 0.5 * 0.95; // overlap neighbours, per example 24's lesson

	function clamp01( v ) {

		return Math.min( 1, Math.max( 0, v ) );

	}

	// Teal at 0, amber at 1. Two saturated hues far apart in the colour wheel so
	// that partial mixing reads as an obvious intermediate rather than as noise.
	function dyeColor( c ) {

		const t = clamp01( c );
		return `rgb(${ Math.round( 30 + t * 225 ) },${ Math.round( 170 + t * 20 ) },${ Math.round( 190 - t * 130 ) })`;

	}

	function draw( positionsData, concentrationData ) {

		ctx.fillStyle = '#0b0d10';
		ctx.fillRect( 0, 0, SIZE, SIZE );

		for ( let p = 0; p < concentrationData.length; p ++ ) {

			ctx.fillStyle = dyeColor( concentrationData[ p ] );
			ctx.beginPath();
			ctx.arc(
				positionsData[ p * 2 ] * scale,
				( NY - positionsData[ p * 2 + 1 ] ) * scale,
				RADIUS, 0, Math.PI * 2
			);
			ctx.fill();

		}

	}

	// ---------------------------------------------------------------- diagnostics

	// Two numbers worth watching, each pinning down one claim.
	//
	// `spread` -- the fraction of particles whose concentration is neither near 0
	// nor near 1, i.e. genuinely mixed. With `mixing` at zero it should stay near
	// zero however violently the liquid moves. That is the diffusion-free
	// transport claim as a measurement rather than an impression.
	//
	// `heightGap` -- mean height of the amber liquid minus mean height of the
	// teal. At a density ratio of 1 it should wander around zero; make amber
	// heavier and it should go decisively negative as amber settles underneath.
	// That is the density-coupling claim, and it is the one that would quietly
	// read zero if beta were never actually reaching the projection.
	function stats( concentrationData, positionsData ) {

		let total = 0, mixed = 0;
		let amberY = 0, amberN = 0, tealY = 0, tealN = 0;

		for ( let p = 0; p < concentrationData.length; p ++ ) {

			const c = concentrationData[ p ];
			total += c;
			if ( c > 0.05 && c < 0.95 ) mixed ++;

			const y = positionsData[ p * 2 + 1 ];
			if ( c > 0.5 ) { amberY += y; amberN ++; } else { tealY += y; tealN ++; }

		}

		return {
			total, mixed,
			spread: mixed / concentrationData.length,
			heightGap: ( amberN ? amberY / amberN : 0 ) - ( tealN ? tealY / tealN : 0 )
		};

	}

	let frame = 0;
	let nanDetected = false;

	function checkForNonFinite( data, frameNumber ) {

		if ( nanDetected ) return;

		for ( let i = 0; i < data.length; i ++ ) {

			if ( ! Number.isFinite( data[ i ] ) ) {

				nanDetected = true;
				status( `non-finite value at frame ${ frameNumber } (index ${ i }, value ${ data[ i ] })`, true );
				console.error( `fluxflow dye-free-surface: non-finite at frame ${ frameNumber }, index ${ i }:`, data[ i ] );
				return;

			}

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
		perfEl.textContent = `fps: ${ ( 1000 / avgMs ).toFixed( 1 ) } | particles: ${ count }`;

	}

	async function animate() {

		updatePerf();
		await flip.onAdvanceTimeStep();

		if ( ! nanDetected && frame % DRAW_INTERVAL === 0 ) {

			const [ positionsData, concentrationData ] = await Promise.all( [
				flip.positions.toArray(),
				flip.concentration.toArray()
			] );

			checkForNonFinite( positionsData, frame );
			checkForNonFinite( concentrationData, frame );

			if ( ! nanDetected ) {

				if ( frame % diagnosticInterval === 0 ) {

					const s = stats( concentrationData, positionsData );
					console.log(
						`fluxflow dye-free-surface [frame ${ frame }] ` +
						`converged=${ flip.pressureSolver.diagnostics.converged } ` +
						`rejected=${ flip.pressureSolver.diagnostics.rejected } | ` +
						`totalDye=${ s.total.toFixed( 2 ) } mixedParticles=${ s.mixed } ` +
						`spread=${ ( s.spread * 100 ).toFixed( 2 ) }% ` +
						`heightGap=${ s.heightGap.toFixed( 2 ) }`
					);

				}

				draw( positionsData, concentrationData );

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

	fadeInput.addEventListener( 'input', () => {

		const v = parseFloat( fadeInput.value );
		fadeUniform.fromArray( new Float32Array( [ v ] ) );
		fadeValueEl.textContent = v.toFixed( 4 );

	} );

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
