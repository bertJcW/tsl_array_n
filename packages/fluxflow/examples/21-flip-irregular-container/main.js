// Demonstrates grid.createGridFlipSolver2's new options.collider (grid_flip_
// solver2.js) against an *irregular* container -- see that file's own header
// comment (the "Collider/obstacle interaction" section) for the full design.
// This is the first real-hardware exercise of sdf_collider2.js's new
// addPolygon(points, { invert: true }) option: a hand-authored closed
// polygon ring with a wavy (sine-perturbed) bottom edge, rasterized inverted
// so the ring's own *interior* is fluid-permitted and everything *outside*
// it is solid -- modeling a basin's own inner wall, which the domain's own
// rectangular closedDomainBoundaryFlag has no way to express on its own.
//
// The ring is inset with a comfortable margin from the rectangular domain on
// every side, so the two independent boundary mechanisms (the domain's own
// flat walls, and this collider's own irregular walls) don't interact at a
// shared edge -- see grid_flip_solver2.js's own Risks section (the plan this
// was built from) for why that's worth avoiding rather than relying on.
//
// pressure.atomicScale/maxPlausiblePressure start from examples/20-flip-dam-
// break's own already-tuned values (same 64x64 grid scale) as a starting
// point, not copied blindly -- this project has repeatedly found these need
// per-scene tuning (see that example's own header comment for the full
// story); adjusted here if real-hardware testing shows a need.

import * as tsl_array_n from 'tsl_array_n';
import { grid } from 'fluxflow';

const particlesCanvas = document.querySelector( '#outParticles' );
const fluidMaskCanvas = document.querySelector( '#outFluidMask' );
const colliderCanvas = document.querySelector( '#outCollider' );
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

// A closed ring tracing the basin's own inner wall: straight left wall ->
// wavy floor (sampled from a sine curve) -> straight right wall -> the top
// edge closes the ring implicitly (first point connects back to last).
// Both ends of the wavy floor land exactly on the wall bottoms (waveCycles
// chosen as a half-integer so sin() returns to 0 at both t=0 and t=1) --
// a clean, non-self-intersecting simple polygon, no seam to worry about.
function buildBasinRing() {

	const left = 4, right = 60, top = 60;
	const baseFloor = 10;
	const waveAmplitude = 5;
	const waveCycles = 2.5;
	const segments = 40;

	const bottomPoints = [];

	for ( let s = 0; s <= segments; s ++ ) {

		const t = s / segments;
		const x = left + t * ( right - left );
		const y = baseFloor + waveAmplitude * Math.sin( t * waveCycles * Math.PI * 2 );
		bottomPoints.push( [ x, y ] );

	}

	return [ [ left, top ], ...bottomPoints, [ right, top ] ];

}

try {

	const renderer = await tsl_array_n.init( { canvas: document.createElement( 'canvas' ), allowFallback: true } );
	status( `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }` );

	const velocityGrid = grid.createFaceCenteredGrid2( NX, NY, 1, 1, 0, 0 );

	const collider = grid.createSDFStaticCollider2( NX, NY, 1, 1, 0, 0 );
	collider.addPolygon( buildBasinRing(), { invert: true } );

	// Well above the wavy floor's own highest point (baseFloor+waveAmplitude
	// = 15) and well clear of the basin's own left/right walls (x=4/x=60).
	const seed = grid.computeFlipBoxSeed( {
		boxMin: [ 15, 35 ],
		boxMax: [ 45, 55 ],
		gridSpacingX: 1, gridSpacingY: 1,
		particlesPerCellAxis: 2
	} );

	// Live-adjustable velocity damping -- see examples/20-flip-dam-break/'s
	// own identical wiring for the full derivation (grid_flip_solver2.js's
	// own "number or node" convention, same as dt).
	const velocityDampingUniform = tsl_array_n.array0( 'float' );
	velocityDampingUniform.fromArray( new Float32Array( [ 0.02 ] ) );

	const flip = grid.createGridFlipSolver2( {
		velocityGrid,
		gridSpacing: [ 1, 1 ],
		origin: [ 0, 0 ],
		maxParticles: seed.count,
		dt,
		collider,
		velocityDamping: velocityDampingUniform(),
		pressure: { atomicScale: 256, maxPlausiblePressure: 100 }
	} );

	flip.positions.fromArray( seed.positionsArray );
	flip.velocities.fromArray( seed.velocitiesArray );

	const particlesCtx = particlesCanvas.getContext( '2d' );
	const PARTICLE_CANVAS_SIZE = particlesCanvas.width; // 1024 -- see index.html; NOT the grid's own NX/NY
	const particleScale = PARTICLE_CANVAS_SIZE / NX; // world units -> canvas pixels (gridSpacing is 1 in this example)
	const fluidMaskCtx = fluidMaskCanvas.getContext( '2d' );
	const fluidMaskImage = fluidMaskCtx.createImageData( NX, NY );
	const colliderCtx = colliderCanvas.getContext( '2d' );
	const colliderImage = colliderCtx.createImageData( NX, NY );

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

	// Draws each particle as a small "+" at its own continuous position,
	// colored by speed -- see examples/20-flip-dam-break/main.js's own
	// identical function for the full derivation of both (a higher native
	// canvas resolution than the NXxNY grid was originally meant to keep
	// individual marks visually distinct, though CROSS_ARM/CROSS_THICKNESS
	// were later enlarged enough that nearby marks now overlap, accepted
	// as fine; a fixed, not per-frame-renormalized, speed scale keeps
	// color physically comparable across the whole run).
	const CROSS_ARM = 6;
	const CROSS_THICKNESS = 2;
	const MAX_SPEED = 15;

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

	// Static collider -- draw once, not every frame. Sign only (not the
	// continuous SDF value) is what matters here: a visual confirmation
	// that invert:true actually flipped inside/outside the way intended.
	const colliderData = await collider.grid.data.toArray();

	for ( let j = 0; j < NY; j ++ ) {

		for ( let i = 0; i < NX; i ++ ) {

			const v = colliderData[ i + NX * j ];
			const pixel = flippedPixelIndex( i, j );
			const bright = v >= 0 ? 255 : 0;

			colliderImage.data[ pixel ] = bright;
			colliderImage.data[ pixel + 1 ] = bright;
			colliderImage.data[ pixel + 2 ] = bright;
			colliderImage.data[ pixel + 3 ] = 255;

		}

	}

	colliderCtx.putImageData( colliderImage, 0, 0 );

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
			`fluxflow flip-irregular-container [frame ${ frameNumber }] ` +
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
				console.error( `fluxflow flip-irregular-container: non-finite value at frame ${ frameNumber }, index ${ i }:`, data[ i ] );
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
