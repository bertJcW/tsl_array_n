// Demonstrates grid.createGridFlipSolver2 (packages/fluxflow/src/grid/
// grid_flip_solver2.js) -- see that file's own header comment for the full
// design (ported from mantaflow's flip.cpp + scenes/flip01_simple.py).
// This scene mirrors flip01_simple.py's own two commented-out options'
// first one directly: a box of fluid in one bottom corner, released under
// gravity in an otherwise closed, empty domain -- the classic "dam break"
// starting condition.
//
// gravity is this port's own factory default (-9.81, real-world units) --
// NOT mantaflow's own literal -0.002, which is tuned for that project's own
// unrelated grid-spacing/unit convention (this session's own mantaflow-
// buoyancy comparison earlier already found copying a literal constant
// across differently-scaled systems doesn't transfer -- see grid_fire_
// solver2.js's own header comment for that same lesson). Confirmed legible
// on real hardware at this value: a clean dam-break curl-over within the
// first ~100 frames, settling into a resting puddle by ~frame 600-700.
//
// *** A real, one-frame pressure-solve blowup found on the very first
// real-hardware run, root-caused (not just caught by a safety net) ***
//
// pressure stayed perfectly healthy (peak ~6-7, matching this scene's own
// small domain) through frame 11, then exploded to the range
// [-136929, 92338] in a single frame (12) -- `rejected` didn't catch it
// (grid_pressure_solver2.js's own default maxPlausiblePressure, 1e6, is
// far too loose for a value merely in the hundred-thousands), and the
// still-bad snapshot then persisted into the *next* frame's own "reverted"
// state too, since the circuit breaker reverts to whatever was snapshotted
// immediately before -- which was already corrupted. Root cause, confirmed
// by testing rather than assumed: `pressure.atomicScale` was left at its
// own library default (`DEFAULT_ATOMIC_DOT_SCALE`, 65536, from linalg.js)
// -- too large for this scene's own actual r.r/p.Ap magnitude (this
// scene's own gravity-driven collapse produces meaningfully larger
// divergence than examples/19-fuel-fire's own gentler buoyancy, despite a
// *smaller* grid than that example's own already-tuned-down case),
// overflowing the CG solve's own atomic dot-product accumulator. Reducing
// it to 256 made the blowup **not happen at all** (not merely caught
// after the fact) across a fresh 700+-frame real-hardware re-run -- this
// is the confirmation it's the actual mechanism, not just a plausible
// story. `maxPlausiblePressure: 100` (still >10x this scene's own real
// peak) is kept as well, matching this same session's own established
// defense-in-depth conclusion from examples/19-fuel-fire's own near-
// identical finding: a correctly-tuned atomicScale prevents the *usual*
// case, a tight circuit breaker still catches whatever it doesn't.
// Neither value is this port's own new global default anywhere -- both
// are scene-specific options here, exactly like examples/19-fuel-fire's
// own PRESSURE_ATOMIC_SCALE/PRESSURE_MAX_PLAUSIBLE, following
// grid_pressure_solver2.js's own header comment on why a "safe" pressure-
// solve magnitude does not transfer between differently-scaled scenes.

import * as tsl_array_n from 'tsl_array_n';
import { grid } from 'fluxflow';

const particlesCanvas = document.querySelector( '#outParticles' );
const fluidMaskCanvas = document.querySelector( '#outFluidMask' );
const statusEl = document.querySelector( '#status' );
const perfEl = document.querySelector( '#perf' );
const velocityDampingInput = document.querySelector( '#velocityDamping' );
const velocityDampingValueEl = document.querySelector( '#velocityDampingValue' );

function status( text, isErr ) {

	statusEl.textContent = text;
	statusEl.className = isErr ? 'err' : '';

}

const NX = 64;
const NY = 64;
const dt = 1 / 30;
const diagnosticInterval = 30;
const DRAW_INTERVAL = 2;

try {

	const renderer = await tsl_array_n.init( { canvas: document.createElement( 'canvas' ), allowFallback: true } );
	status( `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }` );

	const velocityGrid = grid.createFaceCenteredGrid2( NX, NY, 1, 1, 0, 0 );

	// Dam-break box: bottom-left corner, 40% wide / 60% tall -- mantaflow's
	// own flip01_simple.py first commented-out option (p0=(0,0,0),
	// p1=(0.4,0.6,1) of the domain).
	const seed = grid.computeFlipBoxSeed( {
		boxMin: [ 0, 0 ],
		boxMax: [ NX * 0.4, NY * 0.6 ],
		gridSpacingX: 1, gridSpacingY: 1,
		particlesPerCellAxis: 2
	} );

	// Live-adjustable, not baked in as a constant -- grid_flip_solver2.js's
	// own "number or node" convention (same as dt): passing an already-
	// invoked array0('float') reference means later .fromArray() calls
	// (from the velocityDamping slider below) reach the already-built G2P
	// kernel directly, no rebuild needed. Initial value matches this port's
	// own real-hardware-verified default; the slider's own [0,0.1] range is
	// a *useful* exploration window, not the actual safety bound -- the
	// solver itself unconditionally clamps to [0,1] regardless of what this
	// (or any other caller) passes in, see that file's own header comment
	// ("Velocity damping") for why.
	const velocityDampingUniform = tsl_array_n.array0( 'float' );
	velocityDampingUniform.fromArray( new Float32Array( [ 0.02 ] ) );

	const flip = grid.createGridFlipSolver2( {
		velocityGrid,
		gridSpacing: [ 1, 1 ],
		origin: [ 0, 0 ],
		maxParticles: seed.count,
		dt,
		velocityDamping: velocityDampingUniform(),
		pressure: { atomicScale: 256, maxPlausiblePressure: 100 }
	} );

	// Explicit seed -- grid_flip_solver2.js's own header comment on why
	// this is a separate step after construction, not baked into the
	// factory (needs tsl_array_n.init() already run, same reasoning as
	// every other explicit-clear/seed convention this port already uses).
	flip.positions.fromArray( seed.positionsArray );
	flip.velocities.fromArray( seed.velocitiesArray );

	const particlesCtx = particlesCanvas.getContext( '2d' );
	const PARTICLE_CANVAS_SIZE = particlesCanvas.width; // 512 -- see index.html; NOT the grid's own NX/NY
	const particleScale = PARTICLE_CANVAS_SIZE / NX; // world units -> canvas pixels (gridSpacing is 1 in this example)
	const fluidMaskCtx = fluidMaskCanvas.getContext( '2d' );
	const fluidMaskImage = fluidMaskCtx.createImageData( NX, NY );

	function clamp01( v ) {

		return Math.min( 1, Math.max( 0, v ) );

	}

	// canvas Y is down-positive, this grid's Y is up-positive -- flip rows,
	// same convention every other example's own drawing function already uses.
	function flippedPixelIndex( i, j ) {

		return ( ( NY - 1 - j ) * NX + i ) * 4;

	}

	function drawGray( ctx, image, data ) {

		for ( let j = 0; j < NY; j ++ ) {

			for ( let i = 0; i < NX; i ++ ) {

				const v = data[ i + NX * j ];
				const pixel = flippedPixelIndex( i, j );
				const bright = 255 * clamp01( v );

				image.data[ pixel ] = bright;
				image.data[ pixel + 1 ] = bright;
				image.data[ pixel + 2 ] = bright;
				image.data[ pixel + 3 ] = 255;

			}

		}

		ctx.putImageData( image, 0, 0 );

	}

	// Draws each particle as a "+" at its own continuous position (not
	// snapped to a grid cell, unlike drawGray's own per-cell fluidMask
	// view) -- particlesCanvas is deliberately a higher native resolution
	// than the NXxNY grid (see PARTICLE_CANVAS_SIZE above), originally so
	// individual marks stayed visually distinct rather than merging into a
	// solid block the way one-pixel-per-particle at grid resolution used
	// to. CROSS_ARM/CROSS_THICKNESS were later enlarged at the user's own
	// direct request for a more visible mark, with overlap between nearby
	// particles explicitly accepted as fine at that size -- no longer
	// trying to keep every mark visually separate. Plain fillRect pairs,
	// not putImageData -- cheap enough at this particle count (a few
	// thousand) via ordinary Canvas2D calls, and simpler than
	// hand-rasterizing a cross into a pixel buffer.
	const CROSS_ARM = 6;
	const CROSS_THICKNESS = 2;

	// Color by speed (requested directly by the user) -- a fixed reference
	// scale, not renormalized to each frame's own max, so color stays
	// physically comparable across the whole run: the same speed always
	// reads the same color, so the whole cloud visibly cooling toward blue
	// as it settles is itself a meaningful, at-a-glance correctness signal
	// (matching this whole session's own "does it actually settle" check),
	// rather than a per-frame-renormalized view that would always show
	// *some* red even once the sim has essentially stopped. MAX_SPEED (15)
	// is a round number close to this scene's own real observed peaks
	// (roughly 18-20 during the initial splash, confirmed on real hardware
	// earlier this session) -- speeds above it just clamp to the hottest
	// color rather than needing an exact ceiling.
	const MAX_SPEED = 15;

	// blue (slow) -> yellow (mid) -> red (fast), two linear segments --
	// an ordinary "hot" colormap, picked over a single blue->red lerp so
	// the middle of the range is still visually distinguishable from both
	// ends, not just a muddy blend.
	function speedColor( t ) {

		if ( t < 0.5 ) {

			const u = t / 0.5;
			return `rgb(${ Math.round( 80 + u * 175 ) },${ Math.round( 130 + u * 100 ) },${ Math.round( 255 - u * 175 ) })`;

		}

		const u = ( t - 0.5 ) / 0.5;
		return `rgb(255,${ Math.round( 230 - u * 170 ) },${ Math.round( 80 - u * 20 ) })`;

	}

	function drawParticles( positionsData, velocitiesData ) {

		particlesCtx.clearRect( 0, 0, PARTICLE_CANVAS_SIZE, PARTICLE_CANVAS_SIZE );

		for ( let p = 0; p < positionsData.length; p += 2 ) {

			const px = positionsData[ p ] * particleScale;
			const py = ( NY - positionsData[ p + 1 ] ) * particleScale; // flip Y, canvas Y is down-positive

			const vx = velocitiesData[ p ];
			const vy = velocitiesData[ p + 1 ];
			const speed = Math.sqrt( vx * vx + vy * vy );
			const t = Math.min( 1, speed / MAX_SPEED );

			particlesCtx.fillStyle = speedColor( t );
			particlesCtx.fillRect( px - CROSS_ARM, py - CROSS_THICKNESS / 2, CROSS_ARM * 2, CROSS_THICKNESS );
			particlesCtx.fillRect( px - CROSS_THICKNESS / 2, py - CROSS_ARM, CROSS_THICKNESS, CROSS_ARM * 2 );

		}

	}

	// Same non-finite-aware diagnostic pair as every other example.
	function summarize( arr ) {

		let lo = Infinity, hi = -Infinity, nonFiniteCount = 0, sum = 0;

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
		return `${ label } [${ s.lo.toFixed( 4 ) }, ${ s.hi.toFixed( 4 ) }] sum=${ s.sum.toFixed( 4 ) }`;

	}

	async function logDiagnostics( frameNumber, positionsData, velocitiesData, fluidMaskData ) {

		console.log(
			`fluxflow flip-dam-break [frame ${ frameNumber }] ` +
			`converged=${ flip.pressureSolver.diagnostics.converged } rejected=${ flip.pressureSolver.diagnostics.rejected } | ` +
			`${ fmt( 'positions', summarize( positionsData ) ) } | ` +
			`${ fmt( 'velocities', summarize( velocitiesData ) ) } | ` +
			`${ fmt( 'fluidMask', summarize( fluidMaskData ) ) }`
		);

	}

	let frame = 0;
	let nanDetected = false;

	function checkForNonFinite( data, frameNumber ) {

		if ( nanDetected ) return;

		for ( let i = 0; i < data.length; i ++ ) {

			if ( ! Number.isFinite( data[ i ] ) ) {

				nanDetected = true;
				status( `non-finite value detected at frame ${ frameNumber } (index ${ i }, value ${ data[ i ] })`, true );
				console.error( `fluxflow flip-dam-break: non-finite value at frame ${ frameNumber }, index ${ i }:`, data[ i ] );
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
		perfEl.textContent = `fps: ${ ( 1000 / avgMs ).toFixed( 1 ) } | particles: ${ seed.count }`;

	}

	async function animate() {

		updatePerf();

		await flip.onAdvanceTimeStep();

		if ( ! nanDetected && frame % DRAW_INTERVAL === 0 ) {

			const [ positionsData, velocitiesData, fluidMaskData ] = await Promise.all( [
				flip.positions.toArray(),
				flip.velocities.toArray(),
				flip.fluidMask.toArray()
			] );

			checkForNonFinite( positionsData, frame );

			if ( ! nanDetected && frame % diagnosticInterval === 0 ) await logDiagnostics( frame, positionsData, velocitiesData, fluidMaskData );

			if ( ! nanDetected ) {

				drawParticles( positionsData, velocitiesData );
				drawGray( fluidMaskCtx, fluidMaskImage, fluidMaskData );

			}

		}

		frame ++;
		requestAnimationFrame( animate );

	}

	velocityDampingInput.addEventListener( 'input', () => {

		const v = parseFloat( velocityDampingInput.value );
		velocityDampingUniform.fromArray( new Float32Array( [ v ] ) );
		velocityDampingValueEl.textContent = v.toFixed( 3 );

	} );

	requestAnimationFrame( animate );

} catch ( error ) {

	status( error.message, true );
	console.error( error );

}
