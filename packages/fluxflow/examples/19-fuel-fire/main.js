// Demonstrates grid.createGridFireSolver2 (packages/fluxflow/src/grid/
// grid_fire_solver2.js) -- a real fuel/combustion solver, decoupled from
// any specific velocity solver (see that file's own header comment for
// the full design). This scene deliberately builds a *plain*
// grid.createGridSolver2 directly, not grid.createGridSmokeSolver2, and
// composes the fire solver's own buoyancy force into it -- proving the
// decoupling actually works, not just describing it. Fuel is supplied
// continuously by an SDF fuel source (grid.createSDFFuelSource2,
// src/grid/sdf_inflow_outflow2.js), the same SDF-scene-object family as
// collider/inflow/outflow -- unlike examples/18-explosion's own one-shot
// burst, this is a candle-like *continuous* flame: the source keeps
// topping fuel back up to 1 every frame within its own small disc, so the
// flame should hold steady rather than decay away.
//
// Domain/outflow/boundary setup mirrors examples/17-smoke-fire/ exactly
// (a tall, otherwise-closed box with a top-wall outflow strip) -- see that
// file's own header comment for why (smoke needs somewhere to leave, and
// an outflow's own Dirichlet-zero pressure ring avoids the closed-box
// pure-Neumann null-space issue examples/14 needed isDegenerateDot for).
//
// *** buoyancyTemperatureFactor is tuned well above
// grid_fire_solver2.js's own default (5.0) ***
//
// mantaflow's own fire.cpp temperature scale (ignitionTemp=1.25,
// maxTemp=1.75) is much smaller than examples/17-18's own arbitrary
// sourceTemperature/burstTemperature (4/10) -- the library's own default
// buoyancyTemperatureFactor was tuned assuming that larger scale, so this
// scene (which inherits mantaflow's own literal temperature range, not
// examples/17-18's own invented one) needs a proportionally larger factor
// for a comparably visible rise -- but NOT as large as examples/17-18's
// own factors (5.0/6.5) scaled up by the same ratio: those scenes have a
// heat source that's either continuous-but-modest (17) or a single burst
// that decays away (18, no new heat added after frame 0), while this
// scene's own fuel source keeps burning *forever*, continuously adding
// buoyant energy every single frame with nothing but this port's own
// numerical (not physical) dissipation to remove it. Confirmed on real
// hardware: an initial, larger value (22) looked fine for several hundred
// frames but was a genuine, still-growing instability, not just
// turbulence -- u/v's own per-frame *sum* (not just min/max) grew
// steadily in magnitude past frame ~500 with no sign of saturating,
// eventually into the tens of thousands, while individual cell values and
// converged/rejected still looked superficially fine. 6 (much closer to
// grid_fire_solver2.js's own default of 5.0) is the corrected value.
//
// *** A second, more serious stability bug, found via an even-longer real-
// hardware run (3000+ frames) prompted directly by the user's own report
// of a lopsided, "spilled on the floor" plume shape around frame 2000+ ***
//
// The 22-vs-6 investigation above only ran to ~1900 frames -- not far
// enough. Extending the same buoyancyTemperatureFactor=6 run to 3000
// frames found solver.pressureSolver.diagnostics.rejected (grid_pressure_
// solver2.js's own circuit breaker, which discards an implausible pressure
// update and keeps the last known-good one rather than corrupting it --
// see that file's own comment) going permanently true starting somewhere
// around frame ~2250 and staying true through frame 3000 -- i.e. pressure
// effectively stopped being corrected at all for hundreds of frames
// straight, letting the velocity field drift away from incompressibility:
// u.sum/v.sum stopped oscillating and reversing sign (their own healthy
// pattern through frame 2000) and started climbing/falling *monotonically*
// instead, and the fire canvas visibly turned into flat rectangular blocks
// (the multigrid preconditioner's own coarse-level correction showing
// through undamped, no longer smoothed out by a converging fine-level
// solve). Root cause: this scene's own pressure options never set
// `atomicScale` (linalg.js's own DEFAULT_ATOMIC_DOT_SCALE comment),
// silently defaulting to 65536 -- but this scene's grid (96x128=12288
// cells) is 3x larger than examples/15-flow-past-cylinder's own
// 64x64=4096 cells, which *already* needed atomicScale reduced to 1024 to
// avoid int32-overflowing the CG solver's own atomic dot-product
// accumulator (see that file's own header comment for the full
// derivation) -- a bigger grid *and* this scene's own velocities reaching
// into the 40s-80s by this point (vs that file's own plateau of ~28) both
// push the accumulated sum further toward the same overflow ceiling.
// Fixed the same way (atomicScale: 1024), re-confirmed over a fresh
// 3000+-frame run: converged/rejected stayed healthy throughout and u/v
// went back to bounded oscillation, not monotonic drift.
//
// sourceRadius and buoyancyTemperatureFactor were retuned together for a
// thicker plume "stem" (requested directly: the mushroom-cloud shape's
// own middle column looked too thin). A wider source directly widens the
// rising column's own base; a gentler buoyancy accelerates it less
// violently, so it doesn't neck down into a thin, fast jet as quickly
// before spreading into the cap -- reducing buoyancy further is only ever
// safer, never riskier (in the sense of the 22-vs-6 divergence finding
// above), but it has a *separate*, non-safety cost that a first attempt
// (4) went too far on: weaker buoyancy also means combustion products
// linger longer in this tall, mostly-closed domain (only a small top-wall
// outflow strip) before reaching the outflow, giving a large-scale
// recirculation more time to develop and grow enough to dominate the
// entire canvas -- confirmed directly from the user's own real-hardware
// screenshot around frame 3060 (rejected still false, so not the
// atomicScale bug recurring, but a single rotating vortex visibly filling
// most of the frame, reading as "out of control" even though bounded).
// 5 (splitting the difference with the original 6) is the corrected
// value, trading a little of the stem-thickening effect back for a
// visibly calmer, less recirculation-dominated plume -- but re-confirming
// this over another long run showed u.sum still reaching similarly large
// magnitudes (still bounded, rejected still false, just not meaningfully
// smaller), staying one-signed for thousands of frames at a stretch
// rather than reversing every few hundred the way the original 6/6
// configuration did. So this alone doesn't fix the "looks chaotic"
// complaint -- see velocity_damping2.js's own header comment for the
// actual fix the user chose after being asked directly (accept the
// recirculation as authentic fire-in-an-enclosure behavior, keep the
// existing look, or invest in a real damping mechanism): a small uniform
// velocity decay, plugged into createGridSolver2's own computeViscosity
// stage hook below. flameSmoke is
// raised from grid_fire_solver2.js's own default (1.0) for visibly more
// smoke -- a real, solver-level increase in how much density the burn
// kernel emits per unit fuel consumed, distinct from this file's own
// SMOKE_VISUAL_GAIN further down (a display-only multiplier) -- see
// drawFire's own comment for why both exist separately, matching how
// mantaflow's own scene scripts let smoke and flame/heat be tuned
// independently too.
//
// *** A rare, real corruption cascade, found via the user's own real-
// hardware testing after this scene had already looked "confirmed stable"
// -- root-caused, not just papered over with a bigger safety net ***
//
// One run out of several looked fine for ~1300 frames, then visibly
// corrupted (fuel readback above 1, temperature above its own 1.75 cap,
// v.sum in the millions, u/v pinned at grid_blocked_boundary_condition_
// solver2.js's own then-1000 velocity clamp, `rejected` permanently true)
// -- immediate re-runs of the identical code stayed clean for 2000-4000
// frames each, confirming this is real but rare, not deterministic.
// Mechanism, traced from the symptom backward: an occasional CG solve
// (still not further root-caused beyond round 9's own already-fixed
// asymmetric-relaxation bug and its own beta/alpha restart guards --
// linalg.js's own header comments) produces a pressure cell in the low
// thousands -- nowhere near grid_pressure_solver2.js's own old
// MAX_PLAUSIBLE_PRESSURE (1e6), so `rejected` never fires and that
// value's own gradient correction reaches velocity uncorrected, pushing
// it up near the (also too-loose) old 1000 velocity clamp. A "merely"
// ~1000 velocity, clamped or not, is still a ~33-cell-per-frame
// displacement at this scene's own dt -- more than enough to badly
// corrupt the very next fire-solver advection sample (fuel/temperature
// come back implausible-but-finite, e.g. fuel above 1), which then feeds
// buoyancy and compounds over subsequent frames until it's visibly
// obvious. Both circuit breakers' own comments already called their old
// bounds "astronomically larger than any physically meaningful value
// this port produces" -- true, but far too loose to catch something
// merely 100-1000x too large rather than literally infinite.
//
// Fix: tightened both circuit breakers so they actually catch a value
// that size -- grid_blocked_boundary_condition_solver2.js's own
// MAX_VELOCITY_COMPONENT (1000 -> 100, a shared global default, see that
// file's own comment for why velocity's own healthy range is consistent
// enough across every scene checked to make this safe everywhere) and
// this scene's own PRESSURE_MAX_PLAUSIBLE (50) passed explicitly via
// options.maxPlausiblePressure to createGridPressureSolver2 -- NOT as a
// new shared global default. **A real near-miss regression, worth
// recording so it isn't repeated**: tightening the pressure bound as a
// shared global default first (matching the velocity one) directly broke
// examples/16-karman-vortex-street/ on real hardware -- that scene's own
// pressure legitimately reaches 500+ (a much stronger whole-domain
// continuous force than this scene's own gentler buoyancy), so it got
// rejected on literally every single frame, silently freezing pressure
// (and therefore velocity) forever. Caught by real-hardware regression-
// testing that *other* example before considering this fix done, not
// assumed safe from this scene's own numbers alone -- see grid_pressure_
// solver2.js's own header comment for the full lesson (pressure's own
// healthy scale turns out to be as caller-specific as linalg.js's own
// already-documented MAX_ALPHA_MAGNITUDE, unlike velocity).
//
// Re-verified after the fix: 3 independent real-hardware runs (6000,
// 3000, 2500 frames respectively, 11500 frames total, zero rejections,
// zero non-finite values) plus a regression check confirming examples/
// 16-karman-vortex-street/ genuinely develops real flow again (pressure
// reaching its own legitimate hundreds, not stuck at zero) with the
// now-tightened *velocity* default still in place.

import * as tsl_array_n from 'tsl_array_n';
import { grid } from 'fluxflow';

const fuelCanvas = document.querySelector( '#outFuel' );
const fireCanvas = document.querySelector( '#outFire' );
const statusEl = document.querySelector( '#status' );
const perfEl = document.querySelector( '#perf' );
const fuelAmountInput = document.querySelector( '#fuelAmount' );
const fuelAmountValueEl = document.querySelector( '#fuelAmountValue' );

function status( text, isErr ) {

	statusEl.textContent = text;
	statusEl.className = isErr ? 'err' : '';

}

const NX = 96;
const NY = 128;
const dt = 1 / 30;

const sourceX = NX / 2;
const sourceY = 6; // near the bottom, same placement as examples/17-18's own hot source
const sourceRadius = 10; // widened from 6 -- see this file's own header comment (thicker plume stem)

const maxTemp = 1.75; // grid_fire_solver2.js's own default -- restated here since the draw function's own ramp normalization needs it
const buoyancyTemperatureFactor = 5; // reduced from 6, but not all the way to an initially-tried 4 -- see this file's own header comment (thicker plume stem, without over-empowering large-scale recirculation)
const flameSmoke = 3; // raised from grid_fire_solver2.js's own default of 1.0 -- see this file's own header comment (more smoke)
const PRESSURE_ATOMIC_SCALE = 1024; // see this file's own header comment for the real long-run bug this fixes
const PRESSURE_MAX_PLAUSIBLE = 50; // this scene's own real, verified-tight value -- see this file's own header comment; NOT a safe value for every scene (see grid_pressure_solver2.js's own header comment on why this is a per-instance option, not a shared default)
const velocityDampingCoefficient = 0.02; // grid.createVelocityDamping2 -- see this file's own header comment (suppresses the large-scale recirculation)
const diagnosticInterval = 30;
const DRAW_INTERVAL = 2;

const OUTER_MARGIN = 1000; // same padding technique as examples/15-18's own makeWallStripPolygon precedent -- see examples/15-flow-past-cylinder/main.js's own header comment for the real gradient-direction bug this guards against

function makeTopWallStripPolygon( innerY, outerY ) {

	return [ [ - OUTER_MARGIN, innerY ], [ NX + OUTER_MARGIN, innerY ], [ NX + OUTER_MARGIN, outerY ], [ - OUTER_MARGIN, outerY ] ];

}

function makeCirclePolygon( cx, cy, radius, segments ) {

	const verts = [];

	for ( let i = 0; i < segments; i ++ ) {

		const angle = ( i / segments ) * Math.PI * 2;
		verts.push( [ cx + radius * Math.cos( angle ), cy + radius * Math.sin( angle ) ] );

	}

	return verts;

}

try {

	const renderer = await tsl_array_n.init( { canvas: document.createElement( 'canvas' ), allowFallback: true } );
	status( `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }` );

	const velocityGrid = grid.createFaceCenteredGrid2( NX, NY, 1, 1, 0, 0 );

	const outflow = grid.createSDFOutflow2( NX, NY, 1, 1, 0, 0 );
	outflow.addPolygon( makeTopWallStripPolygon( NY - 2, NY + OUTER_MARGIN ) );

	// Live-adjustable fuel amount -- same "plain array0, updated via
	// .fromArray() from a DOM event handler" pattern examples/16's own
	// force-strength slider already established. Passed as a *node* (not a
	// plain number) to createSDFFuelSource2, so buildApplyFuel's own kernel
	// (grid_fire_solver2.js) reads back whatever the slider currently says
	// on every dispatch -- no kernel rebuild needed, see that file's own
	// "number or node" convention.
	const fuelAmountUniform = tsl_array_n.array0( 'float' );
	fuelAmountUniform.fromArray( new Float32Array( [ 1 ] ) );

	const fuelSource = grid.createSDFFuelSource2( NX, NY, 1, 1, 0, 0, { fuel: fuelAmountUniform(), mode: 'set' } );
	fuelSource.addPolygon( makeCirclePolygon( sourceX, sourceY, sourceRadius, 32 ) );

	fuelAmountInput.addEventListener( 'input', () => {

		const v = parseFloat( fuelAmountInput.value );
		fuelAmountUniform.fromArray( new Float32Array( [ v ] ) );
		fuelAmountValueEl.textContent = v.toFixed( 2 );

	} );

	// The fire solver itself -- read-only against velocityGrid, owns none
	// of it (see grid_fire_solver2.js's own header comment). Built *before*
	// the velocity solver below purely so `fire.force` already exists to
	// hand to it; construction order between the two doesn't matter
	// otherwise, since neither reads the other's state at construction
	// time, only from onAdvanceTimeStep() onward.
	const fire = grid.createGridFireSolver2( {
		velocityGrid,
		gridSpacing: [ 1, 1 ],
		origin: [ 0, 0 ],
		dt,
		fuelSources: fuelSource,
		buoyancyTemperatureFactor,
		flameSmoke
	} );

	// grid.createVelocityDamping2 -- see that file's own header comment for
	// the full design and why a uniform decay, not a real viscosity model,
	// directly targets the large-scale recirculation problem this fixes.
	// Plugged into createGridSolver2's own computeViscosity stage hook
	// below (jet's own force -> viscosity -> pressure -> advection order,
	// already composable with zero changes to that file) -- boundarySolver
	// isn't wired in until right after construction, since it doesn't
	// exist yet at the point computeViscosity itself is *written* here,
	// only once onAdvanceTimeStep() actually calls it on some later frame.
	const damping = grid.createVelocityDamping2( { velocityGrid, dampingCoefficient: velocityDampingCoefficient } );
	let boundarySolver;

	// The velocity solver -- deliberately createGridSolver2 directly, not
	// createGridSmokeSolver2, to prove the fire solver composes with
	// *any* chosen velocity solver rather than needing its own bespoke
	// one. fire.force is composed in exactly like vorticity_confinement2.js's
	// own force is elsewhere in this port -- a plain (pos)=>vec2 the
	// caller wires into whichever solver's own force option.
	const solver = grid.createGridSolver2( {
		velocityGrid,
		gridSpacing: [ 1, 1 ],
		origin: [ 0, 0 ],
		force: fire.force,
		outflows: outflow,
		closedDomainBoundaryFlag: grid.DIRECTION_ALL & ~grid.DIRECTION_UP,
		dt,
		computeViscosity: () => {

			damping.applyDamping();
			boundarySolver.constrainVelocity();

		},
		pressure: { multigrid: { numberOfLevels: 4 }, tolerance: 1e-4, maxIterations: 60, atomicScale: PRESSURE_ATOMIC_SCALE, maxPlausiblePressure: PRESSURE_MAX_PLAUSIBLE }
	} );

	boundarySolver = solver.boundarySolver;

	// Explicit clear -- grid_fire_solver2.js's own header comment on why
	// this isn't done internally (needs tsl_array_n.init(), which has
	// already run by this point, but not necessarily during a structural
	// vitest construction).
	fire.fuel.stateA.clear();
	fire.fuel.stateB.clear();
	fire.react.stateA.clear();
	fire.react.stateB.clear();
	fire.density.stateA.clear();
	fire.density.stateB.clear();
	fire.temperature.stateA.clear();
	fire.temperature.stateB.clear();

	const fuelCtx = fuelCanvas.getContext( '2d' );
	const fuelImage = fuelCtx.createImageData( NX, NY );
	const fireCtx = fireCanvas.getContext( '2d' );
	const fireImage = fireCtx.createImageData( NX, NY );

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

	// Same hand-rolled hot-color ramp as examples/17-18's own -- see those
	// files' own comment for why this is deliberately stylized, not a
	// radiometric blackbody calculation. Extended with a final pale-blue
	// stop (requested directly: make the brightest/hottest point read as
	// hotter than white) -- real blackbody radiation does shift white-hot
	// toward blue as temperature keeps rising, so this isn't purely
	// decorative.
	const HOT_RAMP = [
		[ 0, 0, 0 ], [ 0.6, 0.05, 0 ], [ 1, 0.35, 0 ], [ 1, 0.8, 0.1 ], [ 1, 1, 1 ], [ 0.8, 0.9, 1 ]
	];

	function hotRampColor( t ) {

		const clamped = clamp01( t );
		const scaled = clamped * ( HOT_RAMP.length - 1 );
		const idx = Math.min( HOT_RAMP.length - 2, Math.floor( scaled ) );
		const frac = scaled - idx;
		const a = HOT_RAMP[ idx ];
		const b = HOT_RAMP[ idx + 1 ];

		return [
			a[ 0 ] + ( b[ 0 ] - a[ 0 ] ) * frac,
			a[ 1 ] + ( b[ 1 ] - a[ 1 ] ) * frac,
			a[ 2 ] + ( b[ 2 ] - a[ 2 ] ) * frac
		];

	}

	// Two independent display-only gains -- deliberately separate from
	// each other, and from flameSmoke above, matching how mantaflow's own
	// scene scripts let "how much smoke shows up" and "how bright the
	// flame reads" be tuned independently of each other (and of how much
	// smoke the solver actually produces). Neither affects the simulation.
	const FLAME_BRIGHTNESS = 1.8; // boosts the flame's own hot-ramp color
	const FLAME_SHARPNESS = 3; // heat exponent before it drives flame color -- see drawFire's own comment
	// SMOKE_VISUAL_GAIN was originally 3, tuned back when flameSmoke's own
	// density output was still at its library default (1.0) -- once
	// flameSmoke was raised to 3 for "more smoke" (this file's own header
	// comment), the same gain meant any density above ~0.33 already
	// clipped smokeAlpha to a fully-opaque 1.0, so a large fraction of the
	// visible plume (this scene's own observed density regularly reaches
	// 0.6+) rendered as one flat, fully-opaque patch of SMOKE_COLOR with no
	// graduated wispy-to-thick falloff left at all -- reported directly as
	// "too much white smoke." 1.5 keeps opacity ramping smoothly across
	// close to this scene's own real density range instead of saturating
	// a third of the way through it. SMOKE_COLOR darkened from 0.7 to 0.45
	// for the same reason -- even a fully-opaque patch of 0.7 gray reads
	// as pale/whitish next to a black background; 0.45 reads as smoke.
	const SMOKE_VISUAL_GAIN = 1.5;
	const SMOKE_COLOR = [ 0.45, 0.45, 0.45 ];

	// Smoke and flame render into one canvas (previously two: a
	// smoke-only canvas plus this one) -- requested directly, and a more
	// honest combined view besides, since real fire and smoke occupy the
	// same space rather than being two separate fields.
	//
	// *** Why the smoke used to render red, and the fix ***
	//
	// Temperature advects and decays *alongside* density (both are emitted
	// together by the burn kernel and carried by the same velocity field),
	// so residual heat is present almost everywhere smoke has drifted to --
	// heat is essentially never exactly 0 in the visible plume, only small.
	// The previous formula used raw heat to drive the flame-color blend,
	// so that small residual heat kept tinting the *entire* plume with
	// flame color, not just the actually-reacting core. FLAME_SHARPNESS
	// raises heat to a power before using it as the flame blend weight,
	// suppressing small/residual heat much more than the genuinely hot
	// core (heat=0.3 contributes ~3% at FLAME_SHARPNESS=3 instead of 30%)
	// -- smoke away from the core now reads as neutral gray, matching what
	// was actually asked for.
	//
	// Smoke itself is composited as a real alpha-over layer (SMOKE_COLOR,
	// smokeAlpha) on top of the flame color, rather than added into the
	// same RGB sum -- closer to how smoke actually behaves optically (thin
	// smoke lets the flame's own glow show through, thick smoke occludes
	// it), and a more literal reading of "gray, with transparency" than
	// baking density straight into brightness the way the old formula did.
	function drawFire( densityData, temperatureData ) {

		for ( let j = 0; j < NY; j ++ ) {

			for ( let i = 0; i < NX; i ++ ) {

				const idx = i + NX * j;
				const density = clamp01( densityData[ idx ] );
				const heat = clamp01( temperatureData[ idx ] / maxTemp );
				const [ hr, hg, hb ] = hotRampColor( heat );
				const pixel = flippedPixelIndex( i, j );

				// Clamped *before* multiplying into each color channel, not
				// after -- clamping per-channel afterward (255*clamp01(hr*
				// boost)) let a boost this large clip every channel to full
				// regardless of the ramp's own hue once heat neared 1,
				// bleaching the intended pale-blue tip back to plain white.
				// Clamping the scalar mix first preserves the ramp's own
				// color ratio exactly, while still lifting mid-range heat
				// that FLAME_SHARPNESS otherwise leaves looking dim.
				const flameMix = clamp01( FLAME_BRIGHTNESS * Math.pow( heat, FLAME_SHARPNESS ) );
				const flameR = hr * flameMix;
				const flameG = hg * flameMix;
				const flameB = hb * flameMix;

				const smokeAlpha = clamp01( density * SMOKE_VISUAL_GAIN );
				const keep = 1 - smokeAlpha;

				fireImage.data[ pixel ] = 255 * clamp01( SMOKE_COLOR[ 0 ] * smokeAlpha + flameR * keep );
				fireImage.data[ pixel + 1 ] = 255 * clamp01( SMOKE_COLOR[ 1 ] * smokeAlpha + flameG * keep );
				fireImage.data[ pixel + 2 ] = 255 * clamp01( SMOKE_COLOR[ 2 ] * smokeAlpha + flameB * keep );
				fireImage.data[ pixel + 3 ] = 255;

			}

		}

		fireCtx.putImageData( fireImage, 0, 0 );

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

	async function logDiagnostics( frameNumber, fuelData, densityData, temperatureData ) {

		const [ uData, vData ] = await Promise.all( [
			velocityGrid.dataU.toArray(),
			velocityGrid.dataV.toArray()
		] );

		console.log(
			`fluxflow fuel-fire [frame ${ frameNumber }] ` +
			`converged=${ solver.pressureSolver.diagnostics.converged } rejected=${ solver.pressureSolver.diagnostics.rejected } | ` +
			`${ fmt( 'fuel', summarize( fuelData ) ) } | ` +
			`${ fmt( 'density', summarize( densityData ) ) } | ` +
			`${ fmt( 'temperature', summarize( temperatureData ) ) } | ` +
			`${ fmt( 'u', summarize( uData ) ) } | ` +
			`${ fmt( 'v', summarize( vData ) ) }`
		);

	}

	let frame = 0;
	let nanDetected = false;

	function checkForNonFinite( data, frameNumber ) {

		if ( nanDetected ) return;

		for ( let i = 0; i < data.length; i ++ ) {

			if ( ! Number.isFinite( data[ i ] ) ) {

				nanDetected = true;
				status( `non-finite fuel value detected at frame ${ frameNumber } (index ${ i }, value ${ data[ i ] })`, true );
				console.error( `fluxflow fuel-fire: non-finite fuel value at frame ${ frameNumber }, index ${ i }:`, data[ i ] );
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
		perfEl.textContent = `fps: ${ ( 1000 / avgMs ).toFixed( 1 ) }`;

	}

	// solver.onAdvanceTimeStep() (the caller's own chosen velocity solver)
	// must run *before* fire.onAdvanceTimeStep() every frame -- see
	// grid_fire_solver2.js's own header comment for why (its own advection
	// needs this frame's already-updated velocity, and its own buoyancy
	// force reads the *previous* frame's density/temperature regardless).
	async function animate() {

		updatePerf();

		await solver.onAdvanceTimeStep();
		await fire.onAdvanceTimeStep();

		if ( ! nanDetected && frame % DRAW_INTERVAL === 0 ) {

			const [ fuelData, densityData, temperatureData ] = await Promise.all( [
				fire.fuel.current.data.toArray(),
				fire.density.current.data.toArray(),
				fire.temperature.current.data.toArray()
			] );

			checkForNonFinite( fuelData, frame );

			if ( ! nanDetected && frame % diagnosticInterval === 0 ) await logDiagnostics( frame, fuelData, densityData, temperatureData );

			if ( ! nanDetected ) {

				drawGray( fuelCtx, fuelImage, fuelData );
				drawFire( densityData, temperatureData );

			}

		}

		frame ++;
		requestAnimationFrame( animate );

	}

	requestAnimationFrame( animate );

} catch ( error ) {

	status( error.message, true );
	console.error( error );

}
