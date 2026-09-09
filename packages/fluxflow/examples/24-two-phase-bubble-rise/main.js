// Demonstrates grid.createGridTwoPhaseFlipSolver2 (packages/fluxflow/src/
// grid/grid_two_phase_flip_solver2.js) -- see that file's own header comment
// for the full design, the papers it comes from, and what this first version
// deliberately leaves out.
//
// The scene is the canonical two-phase test, and it is chosen because it is
// the one thing a single-phase FLIP solver CANNOT produce at all: a gas
// bubble released at the bottom of a tank of liquid, under a layer of the
// same gas above the water line. In examples/20-flip-dam-break/ the air is a
// Dirichlet `p = 0` void with no dynamics, so a "bubble" there is simply an
// absence of particles and nothing makes it go anywhere. Here the gas is a
// real simulated phase, and it rises.
//
// *** The thing to actually watch for, and the reason the density-ratio
// slider is here rather than being a fixed constant ***
//
// Nothing in this file, and nothing in the solver, applies a buoyancy force.
// Gravity is uniform across every face for both phases. The bubble rises
// only because the variable-density pressure projection knows the gas is
// lighter than the liquid around it. The slider changes exactly one thing --
// how much lighter -- so it is a direct, visible test of that claim: at 1:400
// the bubble tears upward and breaks the surface hard; at 1:2 it barely
// drifts, because there is almost no density contrast left for the
// projection to act on. If the bubble ever rises at 1:1, something is
// applying a force that shouldn't be.
//
// *** Scene-specific pressure tuning -- measured, not guessed ***
//
// This scene was first written with examples/20-flip-dam-break/'s own values
// (`atomicScale: 256`), on the reasoning that it starts from the same 64x64
// gravity-driven setup. Running it on real WebGPU showed that reasoning was
// simply wrong, and the failure was not subtle: the CG solve's fixed-point
// atomic dot-product accumulator overflowed, and pressure went 1023 cells
// out of 1024 non-finite within ten frames. Two-phase changes the magnitudes
// structurally, because `Ap` carries a factor of beta and beta IS the density
// ratio -- so the safe scale shrinks roughly in proportion to it. `1` is the
// measured-stable value here; see the solver's own header comment
// ("pressure.atomicScale") for the full sweep.
//
// `maxPlausiblePressure: 100` is kept as defence in depth, the same
// belt-and-braces conclusion examples/19 and 20 both reached: with the scale
// right, this scene's pressure peaks around 5, so 100 is ~20x generous.
//
// One genuine expectation to set, so it isn't mistaken for a bug: a gas
// face's pressure correction is scaled by beta, so the light phase
// legitimately moves far faster than anything the single-phase solver
// produced, and `converged` does go false on many frames -- the multigrid
// preconditioner is still constant-coefficient (multigrid.js decision 4) and
// preconditions this system less well the higher the ratio goes. That was
// true throughout the real-hardware run that confirmed this scene works, and
// the bubble rose cleanly anyway. Watch `rejected` instead: that is the one
// that means a solve was actually thrown away.
//
// *** The dark speckles left in the liquid, which are real and are not a
// rendering bug ***
//
// After the bubble has passed, dark dots stay behind in the water. Measuring
// per-cell occupancy over a 420-frame run showed they are two different
// things wearing the same colour:
//
//   1. Genuinely torn-off gas -- a couple of hundred cells that are pure gas
//      and below the water line. This is the documented consequence of the
//      one thing this port leaves out of MultiFLIP (Boyd & Bridson 2012):
//      both phases share a single velocity field, so a gas particle dragged
//      into the liquid has no per-phase velocity to separate it back out,
//      and it just sits there. Those dots are physically wrong but they are
//      an omission this solver's header already declares, not a defect in
//      what it does implement.
//
//   2. Cells with no particles in them at all -- around 8% of the 4096 cells
//      by frame 420, and still climbing. An empty cell reads as gas density
//      (see computeDensityKernel in the solver), so it draws exactly like
//      case 1 even though nothing is there. Advection alone never refills a
//      cell it emptied; only resampling does.
//
// Neither is fixable by tuning this scene. Case 1 needs per-phase velocities,
// case 2 needs a narrow-band or level-set surface. Both are on the solver's
// "deliberately left out of this version" list.

import * as tsl_array_n from 'tsl_array_n';
import { grid } from 'fluxflow';

const particlesCanvas = document.querySelector( '#outParticles' );
const densityCanvas = document.querySelector( '#outDensity' );
const statusEl = document.querySelector( '#status' );
const perfEl = document.querySelector( '#perf' );
const densityRatioInput = document.querySelector( '#densityRatio' );
const densityRatioValueEl = document.querySelector( '#densityRatioValue' );
const velocityDampingInput = document.querySelector( '#velocityDamping' );
const velocityDampingValueEl = document.querySelector( '#velocityDampingValue' );
const resetButton = document.querySelector( '#reset' );

function status( text, isErr ) {

	statusEl.textContent = text;
	statusEl.className = isErr ? 'err' : '';

}

const NX = 64;
const NY = 64;
const dt = 1 / 60;

// The scene's own geometry, in grid units.
const WATER_LINE = NY * 0.72;
const BUBBLE_CENTER = [ NX * 0.5, NY * 0.18 ];
const BUBBLE_RADIUS = NY * 0.11;

const LIQUID_DENSITY = 1;
const INITIAL_DENSITY_RATIO = 100;

const diagnosticInterval = 60;
const DRAW_INTERVAL = 2;

function isLiquidAt( [ x, y ] ) {

	if ( y >= WATER_LINE ) return false; // the air above the surface

	const dx = x - BUBBLE_CENTER[ 0 ];
	const dy = y - BUBBLE_CENTER[ 1 ];
	return dx * dx + dy * dy > BUBBLE_RADIUS * BUBBLE_RADIUS; // the bubble

}

try {

	const renderer = await tsl_array_n.init( { canvas: document.createElement( 'canvas' ), allowFallback: true } );
	status( `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }` );

	const velocityGrid = grid.createFaceCenteredGrid2( NX, NY, 1, 1, 0, 0 );

	// The whole domain is seeded, not just the liquid -- that is the
	// structural difference from computeFlipBoxSeed's usual use. Both phases
	// have to exist as particles for either to push on the other, so particle
	// count scales with the entire grid rather than with the liquid's own
	// footprint.
	const seed = grid.computeTwoPhaseBoxSeed( {
		boxMin: [ 0, 0 ],
		boxMax: [ NX, NY ],
		gridSpacingX: 1, gridSpacingY: 1,
		particlesPerCellAxis: 2,
		isLiquid: isLiquidAt
	} );

	const gasDensityUniform = tsl_array_n.array0( 'float' );
	gasDensityUniform.fromArray( new Float32Array( [ LIQUID_DENSITY / INITIAL_DENSITY_RATIO ] ) );

	const velocityDampingUniform = tsl_array_n.array0( 'float' );
	velocityDampingUniform.fromArray( new Float32Array( [ 0.02 ] ) );

	const solver = grid.createGridTwoPhaseFlipSolver2( {
		velocityGrid,
		gridSpacing: [ 1, 1 ],
		origin: [ 0, 0 ],
		maxParticles: seed.count,
		dt,
		liquidDensity: LIQUID_DENSITY,
		gasDensity: gasDensityUniform(),
		velocityDamping: velocityDampingUniform(),
		// atomicScale: 1, NOT the 256 that examples/20-flip-dam-break/ uses
		// and not the library's own 65536 -- both were measured on real
		// hardware to make this scene's pressure field go almost entirely
		// non-finite within ten to twenty frames. See the solver's own header
		// comment ("pressure.atomicScale") for the measurements and for why
		// the safe value scales with the density ratio. If you raise the
		// ratio slider a long way past its default, this may need to come
		// down further still.
		pressure: { atomicScale: 1, maxPlausiblePressure: 100 }
	} );

	function seedScene() {

		solver.positions.fromArray( seed.positionsArray );
		solver.velocities.fromArray( seed.velocitiesArray );
		solver.phase.fromArray( seed.phasesArray );

	}

	seedScene();

	// ---------------------------------------------------------------- drawing

	const particlesCtx = particlesCanvas.getContext( '2d' );
	const PARTICLE_CANVAS_SIZE = particlesCanvas.width;
	const particleScale = PARTICLE_CANVAS_SIZE / NX; // gridSpacing is 1 here

	const densityCtx = densityCanvas.getContext( '2d' );
	const densityImage = densityCtx.createImageData( NX, NY );

	function clamp01( v ) {

		return Math.min( 1, Math.max( 0, v ) );

	}

	function flippedPixelIndex( i, j ) {

		return ( ( NY - 1 - j ) * NX + i ) * 4;

	}

	// Normalized against the live gas/liquid range rather than [0,1], so the
	// picture stays readable as the slider moves the gas density around.
	function drawDensity( data, gasDensity ) {

		const span = Math.max( 1e-6, LIQUID_DENSITY - gasDensity );

		for ( let j = 0; j < NY; j ++ ) {

			for ( let i = 0; i < NX; i ++ ) {

				const t = clamp01( ( data[ i + NX * j ] - gasDensity ) / span );
				const pixel = flippedPixelIndex( i, j );
				const bright = Math.round( 255 * t );
				densityImage.data[ pixel ] = bright;
				densityImage.data[ pixel + 1 ] = bright;
				densityImage.data[ pixel + 2 ] = bright;
				densityImage.data[ pixel + 3 ] = 255;

			}

		}

		densityCtx.putImageData( densityImage, 0, 0 );

	}

	// Particle spacing is gridSpacing/particlesPerCellAxis = 0.5 world units,
	// which at this canvas scale is 8px -- so a radius meaningfully under 8
	// draws the seeding lattice rather than a fluid, complete with a moiré
	// pattern once the canvas is scaled down to its CSS size. The first
	// version of this file used 2.6 and looked exactly like that: a dot grid.
	// Radii here are deliberately larger than the spacing so neighbouring
	// particles overlap and the liquid reads as one connected body.
	const PARTICLE_SPACING_PX = particleScale * 0.5;
	const LIQUID_RADIUS = PARTICLE_SPACING_PX * 0.95;
	const GAS_RADIUS = PARTICLE_SPACING_PX * 0.75;
	const MAX_SPEED = 12;

	// Liquid goes deep blue at rest and pale cyan where it moves fast; gas is
	// a dim warm haze. The two ramps are deliberately different rather than
	// one shared speed ramp: the gas routinely moves several times faster than
	// the liquid (its beta is the whole density ratio), so a shared ramp would
	// peg the gas at one saturated colour and make the liquid look static next
	// to it.
	function liquidColor( t ) {

		const u = clamp01( t );
		return `rgba(${ Math.round( 40 + u * 175 ) },${ Math.round( 110 + u * 130 ) },${ Math.round( 200 + u * 55 ) },0.85)`;

	}

	function drawParticles( positionsData, velocitiesData, phaseData ) {

		particlesCtx.fillStyle = '#080f1a';
		particlesCtx.fillRect( 0, 0, PARTICLE_CANVAS_SIZE, PARTICLE_CANVAS_SIZE );

		// Gas first so the liquid draws over it. `lighter` makes overlapping
		// gas accumulate into a visible haze instead of flat-shading to a
		// single grey -- with normal alpha at this radius the air layer was
		// nearly invisible against the dark ground.
		particlesCtx.globalCompositeOperation = 'lighter';
		particlesCtx.fillStyle = 'rgba(70, 62, 52, 1)';

		for ( let p = 0; p < phaseData.length; p ++ ) {

			if ( phaseData[ p ] > 0.5 ) continue;

			const px = positionsData[ p * 2 ] * particleScale;
			const py = ( NY - positionsData[ p * 2 + 1 ] ) * particleScale; // canvas Y is down-positive

			particlesCtx.beginPath();
			particlesCtx.arc( px, py, GAS_RADIUS, 0, Math.PI * 2 );
			particlesCtx.fill();

		}

		particlesCtx.globalCompositeOperation = 'source-over';

		for ( let p = 0; p < phaseData.length; p ++ ) {

			if ( phaseData[ p ] <= 0.5 ) continue;

			const px = positionsData[ p * 2 ] * particleScale;
			const py = ( NY - positionsData[ p * 2 + 1 ] ) * particleScale;
			const vx = velocitiesData[ p * 2 ];
			const vy = velocitiesData[ p * 2 + 1 ];

			particlesCtx.fillStyle = liquidColor( Math.sqrt( vx * vx + vy * vy ) / MAX_SPEED );
			particlesCtx.beginPath();
			particlesCtx.arc( px, py, LIQUID_RADIUS, 0, Math.PI * 2 );
			particlesCtx.fill();

		}

	}

	// ---------------------------------------------------------------- diagnostics

	function summarize( arr ) {

		let lo = Infinity, hi = - Infinity, nonFiniteCount = 0, sum = 0;

		for ( let i = 0; i < arr.length; i ++ ) {

			const v = arr[ i ];

			if ( ! Number.isFinite( v ) ) {

				nonFiniteCount ++;
				continue;

			}

			if ( v < lo ) lo = v;
			if ( v > hi ) hi = v;
			sum += v;

		}

		return { lo, hi, sum, nonFiniteCount, length: arr.length };

	}

	function fmt( label, s ) {

		if ( s.length === 0 ) return `${ label } EMPTY-READBACK`;
		if ( s.nonFiniteCount > 0 ) return `${ label } ${ s.nonFiniteCount }/${ s.length } NON-FINITE (finite range [${ s.lo.toFixed( 4 ) }, ${ s.hi.toFixed( 4 ) }])`;
		return `${ label } [${ s.lo.toFixed( 4 ) }, ${ s.hi.toFixed( 4 ) }]`;

	}

	// The one number that actually says whether the two phases are still two
	// phases: the mean height of the gas particles. It should climb steadily
	// while the bubble rises, then flatten once it has merged with the layer
	// on top. If it never climbs, buoyancy isn't happening; if it climbs and
	// then keeps climbing past the surface, the phases are mixing.
	function meanGasHeight( positionsData, phaseData ) {

		let sum = 0, count = 0;

		for ( let p = 0; p < phaseData.length; p ++ ) {

			if ( phaseData[ p ] > 0.5 ) continue;
			sum += positionsData[ p * 2 + 1 ];
			count ++;

		}

		return count > 0 ? sum / count : NaN;

	}

	let frame = 0;
	let nanDetected = false;

	function checkForNonFinite( data, frameNumber ) {

		if ( nanDetected ) return;

		for ( let i = 0; i < data.length; i ++ ) {

			if ( ! Number.isFinite( data[ i ] ) ) {

				nanDetected = true;
				status( `non-finite value detected at frame ${ frameNumber } (index ${ i }, value ${ data[ i ] })`, true );
				console.error( `fluxflow two-phase-bubble-rise: non-finite value at frame ${ frameNumber }, index ${ i }:`, data[ i ] );
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
		perfEl.textContent = `fps: ${ ( 1000 / avgMs ).toFixed( 1 ) } | particles: ${ seed.count } (${ seed.liquidCount } liquid, ${ seed.gasCount } gas)`;

	}

	let currentGasDensity = LIQUID_DENSITY / INITIAL_DENSITY_RATIO;

	async function animate() {

		updatePerf();
		await solver.onAdvanceTimeStep();

		if ( ! nanDetected && frame % DRAW_INTERVAL === 0 ) {

			const [ positionsData, velocitiesData, phaseData, densityData ] = await Promise.all( [
				solver.positions.toArray(),
				solver.velocities.toArray(),
				solver.phase.toArray(),
				solver.density.toArray()
			] );

			checkForNonFinite( positionsData, frame );

			if ( ! nanDetected && frame % diagnosticInterval === 0 ) {

				console.log(
					`fluxflow two-phase-bubble-rise [frame ${ frame }] ` +
					`converged=${ solver.pressureSolver.diagnostics.converged } ` +
					`rejected=${ solver.pressureSolver.diagnostics.rejected } | ` +
					`ratio 1:${ ( LIQUID_DENSITY / currentGasDensity ).toFixed( 0 ) } | ` +
					`meanGasHeight=${ meanGasHeight( positionsData, phaseData ).toFixed( 2 ) } | ` +
					`${ fmt( 'velocities', summarize( velocitiesData ) ) } | ` +
					`${ fmt( 'density', summarize( densityData ) ) }`
				);

			}

			if ( ! nanDetected ) {

				drawParticles( positionsData, velocitiesData, phaseData );
				drawDensity( densityData, currentGasDensity );

			}

		}

		frame ++;
		requestAnimationFrame( animate );

	}

	// ---------------------------------------------------------------- controls

	densityRatioInput.addEventListener( 'input', () => {

		const ratio = parseFloat( densityRatioInput.value );
		currentGasDensity = LIQUID_DENSITY / ratio;
		gasDensityUniform.fromArray( new Float32Array( [ currentGasDensity ] ) );
		densityRatioValueEl.textContent = ratio.toFixed( 0 );

	} );

	velocityDampingInput.addEventListener( 'input', () => {

		const v = parseFloat( velocityDampingInput.value );
		velocityDampingUniform.fromArray( new Float32Array( [ v ] ) );
		velocityDampingValueEl.textContent = v.toFixed( 3 );

	} );

	resetButton.addEventListener( 'click', () => {

		seedScene();
		velocityGrid.clear();
		frame = 0;
		nanDetected = false;
		status( `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' } — scene reset` );

	} );

	requestAnimationFrame( animate );

} catch ( error ) {

	status( error.message, true );
	console.error( error );

}
