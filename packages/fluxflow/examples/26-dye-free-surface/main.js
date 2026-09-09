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
// The trade is that this solver has no variable-density coupling, so the dye is
// a *passive tracer*: it is carried by the flow and drawn, but exerts nothing.
// If you want dye whose weight drives the flow -- a heavy dye sinking and
// pluming -- that is example 25's solver, and the two scenes together are the
// honest statement of the trade-off rather than one of them being the answer.
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

import * as tsl_array_n from 'tsl_array_n';
import { grid } from 'fluxflow';

const canvas = document.querySelector( '#out' );
const statusEl = document.querySelector( '#status' );
const perfEl = document.querySelector( '#perf' );
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

	const seed = grid.computeFlipBoxSeed( {
		boxMin: [ 0, 0 ],
		boxMax: [ NX, COLUMN_HEIGHT ],
		gridSpacingX: 1, gridSpacingY: 1,
		particlesPerCellAxis: 2
	} );

	// computeFlipBoxSeed fills a box, so the middle gap is carved out here and
	// the surviving particles are tagged left (0) or right (1).
	const keptPositions = [];
	const keptConcentration = [];

	for ( let p = 0; p < seed.count; p ++ ) {

		const x = seed.positionsArray[ p * 2 ];
		const y = seed.positionsArray[ p * 2 + 1 ];

		if ( x > COLUMN_WIDTH && x < NX - COLUMN_WIDTH ) continue;

		keptPositions.push( x, y );
		keptConcentration.push( x <= COLUMN_WIDTH ? 0 : 1 );

	}

	const count = keptConcentration.length;
	const positionsArray = Float32Array.from( keptPositions );
	const concentrationArray = Float32Array.from( keptConcentration );
	const velocitiesArray = new Float32Array( count * 2 );

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

	// `spread` is the number worth watching: the fraction of particles whose
	// concentration is neither near 0 nor near 1, i.e. genuinely mixed. With
	// mixing at zero it should stay near zero however violently the liquid
	// moves -- that is the diffusion-free transport claim, stated as a number
	// rather than an impression.
	function stats( concentrationData ) {

		let total = 0, mixed = 0;

		for ( let p = 0; p < concentrationData.length; p ++ ) {

			const c = concentrationData[ p ];
			total += c;
			if ( c > 0.05 && c < 0.95 ) mixed ++;

		}

		return { total, mixed, spread: mixed / concentrationData.length };

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

					const s = stats( concentrationData );
					console.log(
						`fluxflow dye-free-surface [frame ${ frame }] ` +
						`converged=${ flip.pressureSolver.diagnostics.converged } ` +
						`rejected=${ flip.pressureSolver.diagnostics.rejected } | ` +
						`totalDye=${ s.total.toFixed( 2 ) } mixedParticles=${ s.mixed } ` +
						`spread=${ ( s.spread * 100 ).toFixed( 2 ) }%`
					);

				}

				draw( positionsData, concentrationData );

			}

		}

		frame ++;
		requestAnimationFrame( animate );

	}

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
