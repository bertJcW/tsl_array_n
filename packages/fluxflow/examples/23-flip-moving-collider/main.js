// Demonstrates grid.createGridFlipSolver2's new options.collider (grid_flip_
// solver2.js) against a *moving* collider -- see that file's own header
// comment (the "Collider/obstacle interaction" section) for the full design,
// and sdf_collider2.js's own header comment (createSDFRigidBodyCollider2)
// for why a plain update(dt) call alone doesn't move a collider that's
// already been baked into a built kernel graph.
//
// SDFRigidBodyCollider2.velocityAt(point) bakes currentPosition/currentAngle
// into the returned TSL graph at kernel-*build* time -- so every frame here,
// *before* calling flip.onAdvanceTimeStep(), this example calls
// rigidCollider.update(dt) (moves the JS-side position/angle, re-rasterizes
// the SDF texture) and then flip.boundarySolver.setCollider(rigidCollider,
// ...) (forces every collider-dependent kernel to rebuild against the new
// pose -- grid_blocked_boundary_condition_solver2.js's own pre-existing
// mechanism, built for occasional collider swaps, never before exercised for
// *continuous* every-frame motion in this project). This is the specific new
// risk this example exists to test on real hardware -- both correctness
// (does the paddle actually behave right frame to frame) and the absence of
// a performance regression from rebuilding kernel graphs every single frame
// over an extended run (watch the fps counter, not just the visuals).
//
// *** Confirmed directly: the every-frame kernel rebuild above, not particle
// count or grid resolution, is what dominates this scene's own per-frame
// cost ***
//
// When this scene's own particle count (~6140 -> ~3070) and grid resolution
// (64x64 -> 64x32, see NY's own comment below) were cut roughly in half at
// the user's own request to raise fps, a real-hardware timed measurement
// (100 manually-stepped frames, warmed up first) still came back at
// ~230ms/frame -- essentially unchanged from this scene's own already-
// measured per-frame cost at the original, larger size (see round 16/17's
// own regression-check timings in the project's memory notes). Reducing
// particle/grid size *did* land (smaller particle count, less wasted empty
// headroom above the pool -- both worth keeping on their own merits) but
// isn't what actually moves this scene's own fps, since setCollider()'s own
// full kernel-graph rebuild cost doesn't scale down with fewer particles or
// a smaller grid the way ordinary per-cell/per-particle kernel dispatch
// would -- it's dominated by shader (re)compilation, not data volume. A real
// fps improvement here would need addressing the rebuild-every-frame
// mechanism itself (e.g. only rebuilding when the pose actually changed by
// more than some threshold, or a cheaper way to move an already-built
// collider), out of scope for this round's own request.
//
// The fluid starts as a *resting* pool (computeFlipBoxSeed's own always-
// zeroed velocity) rather than a dam-break, so the paddle's own disturbance
// reads clearly, not confused with unrelated dam-break motion. The paddle
// starts straddling the pool's own resting surface and sweeps rightward
// while rotating, so it visibly plows through the fluid rather than staying
// clear of it the whole run.
//
// *** Two real, scene-specific findings from real-hardware verification,
// both investigated rather than either ignored or blindly patched ***
//
// (1) pressure.atomicScale needed to drop all the way to 1 -- not merely
// reduced from examples/20-flip-dam-break's own 256, root-caused down to
// the actual mechanism via direct per-iteration instrumentation of
// linalg.js's own CG solve (temporarily logging pAp/alpha every iteration
// of a rejected solve, since grid_pressure_solver2.js's own maxPlausible
// Pressure check only ever sees the *final*, already-corrupted result).
// What that trace showed, for solves that were rejected at 256 *and* at an
// initially-tried intermediate value of 16: pAp swinging over ~100 iterations
// between small magnitudes and values in the tens of millions, *including
// going negative* -- mathematically impossible for a genuinely SPD operator
// (p^T·A·p can't be negative), and exactly the signature of the atomic
// fixed-point accumulator's own int32 encoding wrapping around once
// pAp*atomicScale approaches +-2.1 billion. Raising atomicScale instead
// (tried 1024, expecting more *precision* to help) made it dramatically
// *worse* (12/30 rejected, magnitudes up to 1e14) -- conclusive evidence
// this is an overflow problem, not a precision-floor one, since a larger
// scale only pushes the same pAp values closer to int32's own ceiling
// faster. At atomicScale=1, every dot product this solve produces stays
// far enough below that ceiling regardless of how badly this scene's own
// harder linear system (a resting pool's sudden, discontinuous first
// contact with a moving collider -- a real, sudden divergence spike unlike
// example 20's own gradual gravity-driven collapse) makes CG's own
// intermediate values misbehave. Confirmed, not assumed: 5 independent
// real-hardware runs at atomicScale=1 (30/60/40/350/100 frames, 580 total,
// spanning fresh random seeds each time) -- zero rejects, zero non-finite
// values, in every single run, including the one deliberately covering the
// paddle's entire initial-impact window multiple times over. This also
// motivated a real, small addition to linalg.js itself
// (MAX_PAP_GROWTH_FACTOR, see that file's own header comment) -- a new
// guard for the specific gap this investigation found (several individually
// -reasonable-looking CG steps compounding p geometrically across a whole
// solve, with no single step ever tripping the pre-existing per-step
// guards) -- confirmed via an explicit A/B real-hardware comparison on
// examples/20-flip-dam-break/ itself (a *different* scene, per this
// project's own established regression-testing discipline for any change
// to a shared file) to not regress that example at all: 45/300 rejected
// frames with the new guard active vs. 54/300 with it disabled on the
// identical code otherwise -- slightly fewer, not more, so the new guard is
// safe alongside the real fix here, not a substitute for it (atomicScale=1
// is what actually closed the gap for this scene; the linalg.js guard is a
// genuine, if smaller, additional safety margin for every scene).
// (2) linearVelocityXY is deliberately slow (4, not a literal "sweep the
// whole domain quickly" value) -- found by testing a faster paddle (12
// units/sec): a run carried out past ~150 frames showed rejects climbing to
// 36%+ and *staying* elevated (unlike (1)'s own confined-to-the-opening-
// window pattern) exactly as the paddle approached and then physically
// overlapped the domain's own closed right wall (closedDomainBoundaryFlag)
// -- two independent boundary mechanisms (this collider's own moving wall,
// and the domain's own static one) trying to impose different velocities on
// the same faces at once. That's a genuinely different, out-of-scope
// problem (see grid_flip_solver2.js's own Risks section on keeping collider
// geometry clear of domain edges), not a flaw in the moving-collider
// mechanism this example exists to demonstrate. The slower velocity pushes
// that same wall-proximity effect much later (confirmed clean of it through
// roughly the first 340 frames of a 400-frame run) -- delayed, not
// eliminated outright, since any nonzero one-directional velocity
// eventually reaches a finite domain's own edge; watch the paddle's own
// position before extending a run well past that, rather than assuming an
// indefinite run stays representative of the moving-collider mechanism
// itself.

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
// Half of the original 64 -- the domain's own upper half was consistently
// empty headroom above the pool/paddle (confirmed by watching real runs,
// not assumed), so it was pure wasted per-frame grid-kernel cost
// (markFluidCells, boundary conditions, pressure solve all scan every
// cell regardless of occupancy). Pool depth and the paddle's own Y range
// below are scaled down by the same 0.5 factor, so the pool still fills
// the same *proportion* of the domain (3/8) and keeps the same relative
// splash headroom above it -- this is a resolution/scale change, not a
// redesign of the scene itself.
const NY = 32;
const dt = 1 / 30;
const diagnosticInterval = 30;
const DRAW_INTERVAL = 2;

function boxPolygon( xMin, yMin, xMax, yMax ) {

	return [ [ xMin, yMin ], [ xMax, yMin ], [ xMax, yMax ], [ xMin, yMax ] ];

}

try {

	const renderer = await tsl_array_n.init( { canvas: document.createElement( 'canvas' ), allowFallback: true } );
	status( `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }` );

	const velocityGrid = grid.createFaceCenteredGrid2( NX, NY, 1, 1, 0, 0 );

	// A resting pool filling the lower ~3/8 of the domain (64x12 out of
	// 64x32) -- same proportion as the original 64x24-in-64x64 domain, just
	// at half resolution (see NY's own comment above). ~3070 particles
	// (64x12 cells x 4/cell), down from the original ~6140 -- brought down
	// at the user's own direct request after noting example 21's own
	// lighter scene (2400 particles) ran visibly faster; roughly matches
	// that same scale rather than picking an arbitrary smaller number.
	const seed = grid.computeFlipBoxSeed( {
		boxMin: [ 0, 0 ],
		boxMax: [ 64, 12 ],
		gridSpacingX: 1, gridSpacingY: 1,
		particlesPerCellAxis: 2
	} );

	// Starts straddling the pool's own resting surface (y=12), on the left
	// side, sweeping right at 4 units/sec (see finding (2) below for why not
	// faster) while rotating at 1 rad/sec -- exercises both the linear and
	// angular terms of velocityAt's own rigid-body kinematics, not just
	// translation. Y range (7.5 to 12.5) is the original (15 to 25) scaled
	// by the same 0.5 factor as the pool/domain above, preserving the same
	// submersion depth relative to the pool (most of the paddle below the
	// surface, a little above it) rather than picked fresh.
	const paddleShape = boxPolygon( 5, 7.5, 11, 12.5 );
	const linearVelocityXY = [ 4, 0 ];
	const angularVelocity = 1.0;
	const rigidCollider = grid.createSDFRigidBodyCollider2(
		paddleShape, NX, NY, 1, 1, 0, 0, linearVelocityXY, angularVelocity
	);

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
		collider: rigidCollider,
		velocityDamping: velocityDampingUniform(),
		pressure: { atomicScale: 1, maxPlausiblePressure: 100 }
	} );

	flip.positions.fromArray( seed.positionsArray );
	flip.velocities.fromArray( seed.velocitiesArray );

	const particlesCtx = particlesCanvas.getContext( '2d' );
	// particlesCanvas is NX:NY proportioned (1024x512 -- see index.html) but
	// not necessarily square like every other FLIP example's own canvas --
	// this domain isn't square (64x32) -- so width and height need their
	// own separate references rather than one shared "canvas size".
	const PARTICLE_CANVAS_WIDTH = particlesCanvas.width;
	const PARTICLE_CANVAS_HEIGHT = particlesCanvas.height;
	const particleScale = PARTICLE_CANVAS_WIDTH / NX; // world units -> canvas pixels (gridSpacing is 1 in this example); same scale both axes since width/NX === height/NY
	const fluidMaskCtx = fluidMaskCanvas.getContext( '2d' );
	const fluidMaskImage = fluidMaskCtx.createImageData( NX, NY );

	function clamp01( v ) {

		return Math.min( 1, Math.max( 0, v ) );

	}

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
	// color physically comparable across the whole run). MAX_SPEED (15) clamps a bit below
	// this scene's own observed peak (~18-19 during the paddle's initial
	// plunge, confirmed on real hardware earlier this session) -- the
	// hottest moments just read as fully-saturated red rather than needing
	// an exact ceiling.
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

		particlesCtx.clearRect( 0, 0, PARTICLE_CANVAS_WIDTH, PARTICLE_CANVAS_HEIGHT );

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
			`fluxflow flip-moving-collider [frame ${ frameNumber }] ` +
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
				console.error( `fluxflow flip-moving-collider: non-finite value at frame ${ frameNumber }, index ${ i }:`, data[ i ] );
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

		// The genuinely new part -- see this file's own header comment.
		rigidCollider.update( dt );
		flip.boundarySolver.setCollider( rigidCollider, [ NX, NY ], [ 1, 1 ], [ 0, 0 ] );

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
