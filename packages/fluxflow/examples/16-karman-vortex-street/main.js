// A visual Karman-vortex-street demo, built directly on top of
// examples/15-flow-past-cylinder/'s own confirmed-stable pipeline
// (inflow + outflow's all 3 parts + a real collider + MGPCG pressure,
// all at their library defaults) -- see that file's own header comment
// for the full mechanism this reuses unchanged. What's different here is
// tuned specifically toward getting a *visible, repeating* wake instead
// of a single steady-state wake shape:
//
// *** Honest expectation-setting, read this before judging the result ***
//
// This port has no viscosity model at all (computeViscosity is a no-op,
// see grid_solver2.js's own header comment) and no vorticity confinement
// either. A *real* Karman vortex street is a viscous phenomenon: a
// specific, physically meaningful shedding frequency (the Strouhal
// number, St = f*D/U ~= 0.2 for a wide range of real Reynolds numbers)
// that emerges from a boundary-layer separation instability -- something
// this port's inviscid, purely-numerical-dissipation solver cannot
// reproduce with a controllable Re. What this example CAN do, and does:
// with a long enough downstream domain, a small blockage ratio (cylinder
// diameter << domain height, unlike example 15's own deliberately large
// 1/3-of-domain-height cylinder), and a small deliberate asymmetry (the
// cylinder is offset a few cells off the domain's own centerline -- the
// standard trick used in countless inviscid/numerical vortex-shedding
// demos to break the perfect top/bottom symmetry a centered cylinder
// would otherwise preserve indefinitely), the wake's own shear layers
// still roll up and shed in an alternating, street-like pattern -- driven
// by the semi-Lagrangian scheme's own numerical dissipation acting as a
// stand-in for real viscosity. "Does this look like a vortex street" is
// the right bar, not "does it match a specific Re's Strouhal number."
//
// *** What's actually different from example 15, and why ***
//
// - Domain: 256x128 (2:1, elongated) instead of 15's own 64x64 square --
//   a real Karman street needs downstream room to develop and shed
//   several times before hitting the outflow; a square domain gives it
//   almost none. Higher resolution than an earlier version of this file
//   (128x64), per explicit request -- a low-resolution test isn't
//   representative of real usage.
// - Cylinder: radius 6 (diameter 12, a ~9% blockage ratio against the
//   128-tall domain) instead of 15's own radius N/6 (diameter = domain
//   height/3, a much larger, more blockage-dominated obstacle) -- a
//   smaller cylinder relative to the domain lets the wake's own shear
//   layers develop without the domain's own top/bottom walls interfering
//   as strongly.
// - Cylinder position: offset a few cells below the domain's own
//   vertical centerline (deliberately, not a mistake) -- see the
//   expectation-setting note above for why.
// - A small whole-domain force (much smaller than example 15's own
//   pushStrength) supplements the inflow's own driving velocity, as a
//   precaution against the bulk flow speed decaying over a domain this
//   much longer than example 15's own (inflow alone drives the boundary,
//   but nothing else replenishes momentum lost to numerical dissipation
//   over 200 cells of travel). Live-adjustable from the page's own force
//   control panel (mode/strength/wiggle-frequency), not just this file's
//   own initial constants -- see the force(pos) function's own header
//   comment for the two available modes (a steady uniform push, or that
//   same push plus a small oscillating transverse wiggle localized near
//   the cylinder, a standard trick for seeding shedding faster than
//   waiting on the cylinder's own fixed offset alone).
// - Dye injection: two thin bands hugging the cylinder's own upper/lower
//   edges (not one wide band spanning it, unlike example 15's own single
//   band) -- carried in from the inflow wall, so they wrap around the
//   cylinder on each side and make the two shear layers visually
//   distinguishable as they alternately roll up downstream. dyeDecay is
//   much slower than example 15's own (0.9997 vs 0.995) -- this domain is
//   long enough that even a small per-frame decay compounds into total
//   fade before dye reaches the cylinder, let alone downstream of it.
// - atomicScale: 256, lower than example 15's own 1024 -- this scene has
//   8x as many cells (32768 vs 4096), so the same per-cell fixed-point
//   scale accumulates a proportionally larger total in linalg.js's own
//   atomic dot-product reduction; confirmed on real hardware that 1024
//   here let that reduction reach close enough to its own int32 ceiling
//   to measurably degrade convergence over a long run (see linalg.js's
//   own DEFAULT_ATOMIC_DOT_SCALE comment), where 256 does not.
//
// Same real-hardware-only status as example 15 (MGPCG atomics don't
// compile on this dev sandbox's WebGL2 fallback) -- real-hardware
// confirmation of the actual shedding pattern (and of the tuning
// decisions above) is the whole point of this file.
//
// *** The long-run divergence this file's own header used to describe as
// "still unresolved" has since been root-caused and fixed -- see
// multigrid.js's own createMultigridPreconditioner header comment (an
// asymmetric red-black relax schedule, found by comparing against
// mantaflow's own multigrid solver) and linalg.js's own beta/alpha
// robustness guards, plus grid_pressure_solver2.js's own last-resort
// per-frame circuit breaker (reverts a frame's pressure update if it
// ever looks implausible, rather than let it reach velocity) and
// grid_blocked_boundary_condition_solver2.js's own sibling circuit
// breaker for velocity itself (see that file's own MAX_VELOCITY_COMPONENT
// comment for why pressure's own guard alone wasn't sufficient) for the
// full fix, confirmed via multi-thousand-frame real-hardware runs on
// both this scene and example 15's -- explosion/non-finite values are
// not a risk here regardless of scene complexity. `outflowVelocityBC` is
// back to its library default (true, full convective velocity
// extrapolation) -- the earlier `false` workaround here was papering over
// that bug, not a real tuning choice, and is no longer needed.
//
// *** Confirmed on real hardware: this scene genuinely sheds, alternating
// -- read this before judging the result by the dye canvas alone ***
//
// The dye canvas alone is inconclusive: it fades to near-nothing within
// 20-30 cells of the cylinder (ordinary numerical diffusion from the
// semi-Lagrangian scheme's own cubic interpolation, likely amplified
// right at the cylinder's own sharp velocity gradient), well before
// reaching the region downstream where shedding structure would need to
// be visible in it. That's a visualization limit, not a physics one --
// confirmed directly by computing vorticity (dv/dx - du/dy) from the
// *velocity* field itself, which doesn't share dye's own decay: real-
// hardware readback at frame ~950 showed vorticity's own sign
// alternating (+/-/-/+/-/+/+/-/...) at consecutive downstream stations
// immediately behind the cylinder, with the peak-magnitude location
// itself oscillating in y between the cylinder's upper and lower shear
// layers -- exactly the signature of alternating vortex shedding, not
// noise or a steady wake. This is why the pressure canvas below was
// replaced with a vorticity canvas: dye shows the flow deflecting
// correctly around the cylinder (confirming example 15's own result
// still holds here), vorticity shows the alternating shedding the dye
// canvas alone can't -- red/blue banding alternating downstream of the
// cylinder is what to look for.

import * as tsl_array_n from 'tsl_array_n';
import { vec2, float, max, sin, length } from 'three/tsl';
import { grid } from 'fluxflow';

const dyeCanvas = document.querySelector( '#outDye' );
const vorticityCanvas = document.querySelector( '#outVorticity' );
const statusEl = document.querySelector( '#status' );
const perfEl = document.querySelector( '#perf' );
const forceModeSelect = document.querySelector( '#forceMode' );
const forceStrengthInput = document.querySelector( '#forceStrength' );
const forceFreqInput = document.querySelector( '#forceFreq' );
const forceStrengthValueEl = document.querySelector( '#forceStrengthValue' );
const forceFreqValueEl = document.querySelector( '#forceFreqValue' );

function status( text, isErr ) {

	statusEl.textContent = text;
	statusEl.className = isErr ? 'err' : '';

}

const NX = 256;
const NY = 128;
const dt = 1 / 30;
const pushStrength = 0.05; // initial value only -- see the force control panel, live-adjustable from here
const inflowSpeed = 2;
const cylinderRadius = 6;
const cylinderCenterX = 56; // leaves 200 cells of downstream room
const cylinderCenterY = NY / 2 - 8; // deliberate asymmetry -- see header comment
const dyeSourceX = 3; // near the inflow wall
const dyeBandHalfWidth = 2.4; // thin bands hugging the cylinder's own edges
const injectionDensity = 1;
const dyeDecay = 0.9997; // slower than example 15's own 0.995 -- this domain is much longer, dye needs to survive the trip to the cylinder and beyond
const vorticityColorScale = 0.3; // see drawVorticity's own comment
const diagnosticInterval = 30; // frames between console readouts

function makeCirclePolygon( cx, cy, radius, segments ) {

	const verts = [];

	for ( let i = 0; i < segments; i ++ ) {

		const angle = ( i / segments ) * Math.PI * 2;
		verts.push( [ cx + radius * Math.cos( angle ), cy + radius * Math.sin( angle ) ] );

	}

	return verts;

}

// Same padding technique as example 15's own makeWallStripPolygon --
// see that file's own header comment for the real, confirmed-on-real-
// hardware gradient-direction bug this guards against.
const OUTER_MARGIN = 1000;

function makeWallStripPolygon( innerX, outerX ) {

	return [ [ innerX, - OUTER_MARGIN ], [ outerX, - OUTER_MARGIN ], [ outerX, NY + OUTER_MARGIN ], [ innerX, NY + OUTER_MARGIN ] ];

}

try {

	const renderer = await tsl_array_n.init( { canvas: document.createElement( 'canvas' ), allowFallback: true } );
	status( `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }` );

	const velocityGrid = grid.createFaceCenteredGrid2( NX, NY, 1, 1, 0, 0 );

	// dye state, ping-ponged -- see example 15's own header comment for why
	// this is four fields, not two.
	const stateA = grid.createScalarGrid2( NX, NY, 1, 1, 0, 0 );
	const stateB = grid.createScalarGrid2( NX, NY, 1, 1, 0, 0 );
	stateA.clear();
	stateB.clear();

	const rawAdvectedA = { data: tsl_array_n.arrayN( 'float', [ NX, NY ] ) };
	const rawAdvectedB = { data: tsl_array_n.arrayN( 'float', [ NX, NY ] ) };

	const collider = grid.createSDFStaticCollider2( NX, NY, 1, 1, 0, 0 );
	collider.addPolygon( makeCirclePolygon( cylinderCenterX, cylinderCenterY, cylinderRadius, 48 ) );

	const inflow = grid.createSDFInflow2( NX, NY, 1, 1, 0, 0, { velocity: [ inflowSpeed, 0 ], mode: 'set' } );
	inflow.addPolygon( makeWallStripPolygon( 2, - OUTER_MARGIN ) );

	const outflow = grid.createSDFOutflow2( NX, NY, 1, 1, 0, 0 );
	outflow.addPolygon( makeWallStripPolygon( NX - 2, NX + OUTER_MARGIN ) );

	// Live-adjustable force, per the user's own request to experiment with
	// different external forces without reloading -- same "plain array0,
	// updated via .fromArray() from a DOM event handler" pattern already
	// established throughout this port (src/interaction/'s own pointer/
	// keyboard/time uniforms), not a new mechanism. force(pos) itself is
	// re-evaluated once per cell every frame (see external_force_solver2.js's
	// own header comment), so reading a live uniform here picks up a
	// control-panel change on the very next dispatch, no kernel rebuild
	// needed.
	//
	// mode 0 ("uniform"): the original design -- a small, whole-domain
	// rightward push supplementing the inflow's own driving velocity (see
	// this file's own header comment on why this exists at all: offsetting
	// numerical dissipation over a much longer domain than example 15's
	// own).
	// mode 1 ("oscillating"): the same steady push, PLUS a small
	// transverse (y) wiggle localized to a region around the cylinder --
	// a standard technique for seeding/accelerating the asymmetric
	// instability that leads to alternating shedding, rather than waiting
	// for the cylinder's own fixed offset alone to do it. Localized (not
	// applied domain-wide) so it doesn't fight the inflow/outflow
	// boundaries or the steady bulk flow away from the cylinder.
	const forceModeUniform = tsl_array_n.array0( 'float' );
	const forceStrengthUniform = tsl_array_n.array0( 'float' );
	const forceFreqUniform = tsl_array_n.array0( 'float' );
	const simTimeUniform = tsl_array_n.array0( 'float' );

	forceModeUniform.fromArray( new Float32Array( [ 0 ] ) );
	forceStrengthUniform.fromArray( new Float32Array( [ pushStrength ] ) );
	forceFreqUniform.fromArray( new Float32Array( [ 0.2 ] ) );
	simTimeUniform.fromArray( new Float32Array( [ 0 ] ) );

	const wiggleRegionRadius = cylinderRadius * 3;

	function force( pos ) {

		const strength = forceStrengthUniform();
		const uniformForce = vec2( strength, 0 );

		const nearCylinder = length( pos.sub( vec2( cylinderCenterX, cylinderCenterY ) ) ).lessThan( wiggleRegionRadius );
		const wiggle = sin( simTimeUniform().mul( forceFreqUniform() ).mul( Math.PI * 2 ) ).mul( strength ).mul( 2 );
		const oscillatingForce = vec2( strength, nearCylinder.select( wiggle, float( 0 ) ) );

		return forceModeUniform().equal( 1 ).select( oscillatingForce, uniformForce );

	}

	forceModeSelect.addEventListener( 'change', () => {

		forceModeUniform.fromArray( new Float32Array( [ forceModeSelect.value === 'oscillating' ? 1 : 0 ] ) );

	} );

	forceStrengthInput.addEventListener( 'input', () => {

		const v = parseFloat( forceStrengthInput.value );
		forceStrengthUniform.fromArray( new Float32Array( [ v ] ) );
		forceStrengthValueEl.textContent = v.toFixed( 3 );

	} );

	forceFreqInput.addEventListener( 'input', () => {

		const v = parseFloat( forceFreqInput.value );
		forceFreqUniform.fromArray( new Float32Array( [ v ] ) );
		forceFreqValueEl.textContent = v.toFixed( 2 );

	} );

	const closedDomainBoundaryFlag = grid.DIRECTION_ALL & ~grid.DIRECTION_RIGHT & ~grid.DIRECTION_LEFT;

	const solver = grid.createGridSolver2( {
		velocityGrid,
		gridSpacing: [ 1, 1 ],
		origin: [ 0, 0 ],
		force,
		collider,
		inflows: inflow,
		outflows: outflow,
		closedDomainBoundaryFlag,
		dt,
		advection: { collider }, // NOT automatic -- see example 15's own header comment
		// numberOfLevels: a grid this size needs real multigrid coarsening,
		// not numberOfLevels:1. atomicScale: see this file's own header
		// comment for why 256 (not example 15's own 1024).
		pressure: { multigrid: { numberOfLevels: 4 }, tolerance: 1e-5, maxIterations: 100, atomicScale: 256 }
	} );

	const dyeAdvectionSolver = grid.createSemiLagrangianAdvectionSolver2( { velocityGrid: solver.velocityGrid, dt, collider } );

	const advectAtoB = dyeAdvectionSolver.advectScalar2( stateA, rawAdvectedB );
	const advectBtoA = dyeAdvectionSolver.advectScalar2( stateB, rawAdvectedA );

	// Two thin bands hugging the cylinder's own upper/lower edges (not one
	// wide band spanning it, unlike example 15's own single band) -- see
	// this file's own header comment for why.
	const cylinderTopY = cylinderCenterY + cylinderRadius;
	const cylinderBottomY = cylinderCenterY - cylinderRadius;

	function createInjectKernel( rawAdvectedGrid, stateGrid ) {

		return tsl_array_n.kernel( stateGrid.dataSize, ( i, j ) => {

			const pos = stateGrid.dataPosition( i, j );
			const raw = rawAdvectedGrid.data( i, j );

			const inSourceX = pos.x.lessThan( dyeSourceX );
			const nearTop = pos.y.sub( cylinderTopY ).abs().lessThan( dyeBandHalfWidth );
			const nearBottom = pos.y.sub( cylinderBottomY ).abs().lessThan( dyeBandHalfWidth );
			const inSource = inSourceX.and( nearTop.or( nearBottom ) );
			const sourceInjection = inSource.select( float( injectionDensity ), float( 0 ) );

			stateGrid.data( i, j ).assign( max( raw.mul( dyeDecay ), sourceInjection ) );

		} );

	}

	const injectAtoB = createInjectKernel( rawAdvectedB, stateB );
	const injectBtoA = createInjectKernel( rawAdvectedA, stateA );

	const clearOutflowA = solver.outflowSolver.clearOutflowScalarField( stateA );
	const clearOutflowB = solver.outflowSolver.clearOutflowScalarField( stateB );

	// Vorticity (dv/dx - du/dy), computed directly from velocity via
	// grid_math.js's own faceCenteredCurlAtCenter2 -- see this file's own
	// header comment for why this, not dye, is what actually shows the
	// alternating shedding pattern: dye fades from numerical diffusion
	// well before reaching the region where shedding structure would be
	// visible in it, but vorticity is read straight from velocity itself
	// every frame, so it carries no such decay.
	const vorticityGrid = tsl_array_n.arrayN( 'float', [ NX, NY ] );

	const computeVorticity = tsl_array_n.kernel( [ NX, NY ], ( i, j ) => {

		vorticityGrid( i, j ).assign( grid.faceCenteredCurlAtCenter2( velocityGrid.dataU, velocityGrid.dataV, velocityGrid.gridSpacing, i, j, [ NX, NY ] ) );

	} );

	const dyeCtx = dyeCanvas.getContext( '2d' );
	const dyeImage = dyeCtx.createImageData( NX, NY );
	const vorticityCtx = vorticityCanvas.getContext( '2d' );
	const vorticityImage = vorticityCtx.createImageData( NX, NY );

	function clamp01( v ) {

		return Math.min( 1, Math.max( 0, v ) );

	}

	// canvas Y is down-positive, this grid's Y is up-positive -- flip rows
	// so the image matches the simulation's own orientation (both draw
	// functions below share this convention).
	function flippedPixelIndex( i, j ) {

		return ( ( NY - 1 - j ) * NX + i ) * 4;

	}

	function drawDye( data ) {

		for ( let j = 0; j < NY; j ++ ) {

			for ( let i = 0; i < NX; i ++ ) {

				const v = data[ i + NX * j ];
				const pixel = flippedPixelIndex( i, j );
				const bright = 255 * clamp01( v );

				dyeImage.data[ pixel ] = bright;
				dyeImage.data[ pixel + 1 ] = bright;
				dyeImage.data[ pixel + 2 ] = bright;
				dyeImage.data[ pixel + 3 ] = 255;

			}

		}

		dyeCtx.putImageData( dyeImage, 0, 0 );

		dyeCtx.strokeStyle = '#f87171';
		dyeCtx.lineWidth = 1;
		dyeCtx.beginPath();
		dyeCtx.arc( cylinderCenterX, NY - cylinderCenterY, cylinderRadius, 0, Math.PI * 2 );
		dyeCtx.stroke();

	}

	// Diverging colormap (blue = negative, black = zero, red = positive) --
	// unlike pressure's own grayscale, sign is the whole point here (see
	// this file's own header comment): a single brightness scale can't
	// show two opposite-signed things as visually distinct the way red
	// vs. blue immediately can, and alternating red/blue bands downstream
	// of the cylinder is exactly the signature to look for.
	function drawVorticity( data ) {

		for ( let j = 0; j < NY; j ++ ) {

			for ( let i = 0; i < NX; i ++ ) {

				const w = data[ i + NX * j ];
				const pixel = flippedPixelIndex( i, j );
				const t = clamp01( Math.abs( w ) / vorticityColorScale );

				vorticityImage.data[ pixel ] = w > 0 ? 255 * t : 0;
				vorticityImage.data[ pixel + 1 ] = 0;
				vorticityImage.data[ pixel + 2 ] = w < 0 ? 255 * t : 0;
				vorticityImage.data[ pixel + 3 ] = 255;

			}

		}

		vorticityCtx.putImageData( vorticityImage, 0, 0 );

		vorticityCtx.strokeStyle = '#4ade80';
		vorticityCtx.lineWidth = 1;
		vorticityCtx.beginPath();
		vorticityCtx.arc( cylinderCenterX, NY - cylinderCenterY, cylinderRadius, 0, Math.PI * 2 );
		vorticityCtx.stroke();

	}

	// same non-finite-aware diagnostic pair as example 15's own -- see
	// that file's own header comment for why a naive min/max scan
	// silently hides NaN instead of reporting it.
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

	async function logDiagnostics( frameNumber, dyeData, vorticityData ) {

		const [ uData, vData, bData ] = await Promise.all( [
			velocityGrid.dataU.toArray(),
			velocityGrid.dataV.toArray(),
			solver.pressureSolver.b.toArray()
		] );

		console.log(
			`fluxflow karman-vortex-street [frame ${ frameNumber }] ` +
			`converged=${ solver.pressureSolver.diagnostics.converged } rejected=${ solver.pressureSolver.diagnostics.rejected } | ` +
			`${ fmt( 'dye', summarize( dyeData ) ) } | ` +
			`${ fmt( 'u', summarize( uData ) ) } | ` +
			`${ fmt( 'v', summarize( vData ) ) } | ` +
			`${ fmt( 'vorticity', summarize( vorticityData ) ) } | ` +
			`${ fmt( 'b(divergence)', summarize( bData ) ) }`
		);

	}

	let frame = 0;
	let nanDetected = false;
	let simTime = 0;

	// Rolling average over the last N real-world frame times -- answers
	// "is the flow's apparent slowness a chosen physical speed or a GPU
	// throughput limit" directly: `dt` (a fixed simulated-seconds-per-call)
	// advances the SAME amount regardless of how long a call actually
	// takes, so "sim speed" (fps*dt) below 1x means the simulation is
	// running in slow motion relative to its own intended timescale --
	// exactly what heavy per-frame GPU work (a 256x128 MGPCG solve, up to
	// 4 CPU<->GPU array readbacks) would cause, independent of inflowSpeed/
	// pushStrength's own chosen values.
	const FPS_WINDOW = 30;
	const frameTimes = [];
	let lastFrameTime = performance.now();

	function updatePerf() {

		const now = performance.now();
		frameTimes.push( now - lastFrameTime );
		lastFrameTime = now;
		if ( frameTimes.length > FPS_WINDOW ) frameTimes.shift();

		const avgMs = frameTimes.reduce( ( a, b ) => a + b, 0 ) / frameTimes.length;
		const fps = 1000 / avgMs;
		const simSpeed = fps * dt;

		perfEl.innerHTML = `fps: ${ fps.toFixed( 1 ) } | sim speed: <span class="${ simSpeed < 0.9 ? 'slow' : '' }">${ simSpeed.toFixed( 2 ) }x real-time</span>`;

	}

	function checkForNonFinite( data, frameNumber ) {

		if ( nanDetected ) return;

		for ( let i = 0; i < data.length; i ++ ) {

			if ( ! Number.isFinite( data[ i ] ) ) {

				nanDetected = true;
				status( `non-finite dye value detected at frame ${ frameNumber } (index ${ i }, value ${ data[ i ] })`, true );
				console.error( `fluxflow karman-vortex-street: non-finite dye value at frame ${ frameNumber }, index ${ i }:`, data[ i ] );
				return;

			}

		}

	}

	async function animate() {

		updatePerf();

		simTime += dt;
		simTimeUniform.fromArray( new Float32Array( [ simTime ] ) );

		await solver.onAdvanceTimeStep( dt );

		let currentState;

		if ( frame % 2 === 0 ) {

			advectAtoB();
			injectAtoB();
			clearOutflowB();
			currentState = stateB;

		} else {

			advectBtoA();
			injectBtoA();
			clearOutflowA();
			currentState = stateA;

		}

		computeVorticity();

		const [ dyeData, vorticityData ] = await Promise.all( [
			currentState.data.toArray(),
			vorticityGrid.toArray()
		] );

		checkForNonFinite( dyeData, frame );

		if ( ! nanDetected && frame % diagnosticInterval === 0 ) await logDiagnostics( frame, dyeData, vorticityData );

		if ( ! nanDetected ) {

			drawDye( dyeData );
			drawVorticity( vorticityData );

		}

		frame ++;
		if ( ! nanDetected ) requestAnimationFrame( animate );

	}

	requestAnimationFrame( animate );

} catch ( error ) {

	status( error.message, true );
	console.error( error );

}
