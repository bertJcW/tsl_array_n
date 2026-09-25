// Does this 3D scene run, or has it broken? -- a verdict, not a number to
// squint at.
//
// The detector this replaces watched one scalar, the maximum velocity
// component, against a fixed threshold of 90. It cannot decide the
// question, for two reasons that both bit:
//
//   - The solver's own circuit breaker clamps velocity at 100, so a field
//     that has completely come apart still reads as a finite number, and a
//     legitimately fast flow can approach the same number while being fine.
//   - A field can be thoroughly wrong far below any clamp.
//     examples/35-karman-vortex-street-3d/ settles after its blow-up into a
//     state whose maximum is 17.5 and whose outflow carries 8.7x the mass
//     its inflow admits. That state was reported as stable on the strength
//     of that 17.5.
//
// So this measures conservation laws instead, which do not have a scale to
// be tuned against:
//
//   1. FLUX. In an incompressible channel with impermeable walls and one
//      inlet, the weighted flux through every cross-section equals the
//      inflow. Cross-sections need the collider's own face weights to be
//      meaningful, since velocity inside a collider is extrapolated on
//      purpose; where the solver has no weights to offer, this is reported
//      rather than judged and the mass balance below carries the decision.
//      2D scenes are always in that position -- the 2D solver has no
//      fractional weights at all.
//   2. DIVERGENCE, in fluid cells only. The solve deliberately leaves the
//      vent's Dirichlet cells and the collider's enclosed cells with
//      divergence, so a check that includes them measures nothing --
//      which is how an earlier reading of this scene got explained away.
//      The masks come from the solver itself rather than being rebuilt here.
//   3. BOUNDEDNESS, as a multiple of the scene's own inflow speed rather
//      than an absolute number, and measured over fluid faces only.
//   4. FINITENESS.
//
// Three things are measured and reported but deliberately do NOT decide the
// verdict, because neither is invariant for this solver:
//
//   - Velocity on fully closed faces. constrainVelocity extrapolates
//     velocity into the collider on purpose, so that semi-Lagrangian
//     advection sampling inside it gets something sane; a nonzero value
//     there is the design, not a leak. (The first version of this probe
//     called a healthy frame broken for exactly this.) It is still worth
//     watching, since a jump in it means something changed.
//   - Divergence, which is sampled at the end of a frame rather than
//     immediately after the projection. Every boundary stage that runs
//     after the solve -- the inflow band, the outflow, the collider
//     constraint -- puts divergence back in on purpose, for the next
//     frame's projection to take out. What is measured is therefore the
//     work waiting for the next solve, not how well the last one did.
// Flux through the domain's last face against the inlet's IS a criterion,
// and an earlier version of this file was wrong to demote it. The argument
// for demoting it was that the vent's cells are pinned and their divergence
// deliberately not removed, so they are a sink by construction -- but that
// was reasoning from a symptom. A healthy scene carries 0.99 of its inflow
// out of the outlet, and the outflow boundary condition this package
// shipped before carried 8.67. Since the scene converges on every frame in
// both cases, this is the only criterion here that can tell them apart.
//
// It cannot apply while the domain is still filling, and the probe takes
// that from the data rather than a frame count -- see the establishment
// rule below.
//
// *** What this cannot check ***
//
// A free-surface scene -- anything built on the FLIP solvers -- is outside
// all of this. Its solved region is the liquid, whose shape changes every
// frame, so net flux across that region's boundary is not zero: it is
// exactly what moves the surface. The plane machinery does not fit it
// either, since the liquid is not a prefix of planes. Such a scene reports
// finiteness and nothing else, and says so rather than dressing it up.
//
// Every threshold is a multiple of something the scene itself defines --
// the flux admitted at the inlet plane, and the mean speed that implies --
// so the same probe works on any scene built from grid_solver3 without
// being retuned. Each frame's actual numbers are reported alongside the
// verdict, so a threshold can be re-judged without re-running anything.
//
// The solver's own counters -- circuit-breaker rejections, CG breakdowns --
// are reported but do NOT decide the verdict. A bare channel converges on
// 28 frames out of 900 while being perfectly healthy, and a broken scene
// converged on 73% of its frames, so those counters are evidence to explain
// rather than a criterion. The physics decides.
//
// usage: node solver_health.mjs <url> [frames] [sampleEvery]

import { chromium } from 'playwright-core';

const URL = process.argv[ 2 ] ?? 'http://localhost:5200/examples/35-karman-vortex-street-3d/';
const FRAMES = Number( process.argv[ 3 ] ?? 2000 );
const SAMPLE = Number( process.argv[ 4 ] ?? 5 );

// Thresholds, all relative to the scene's own inflow.
const FLUX_TOLERANCE = 0.05;    // of the inlet flux, across interior cross-sections
const BALANCE_TOLERANCE = 0.05; // of the inlet flux, what enters against what leaves
const SPEED_BOUND = 6;          // fluid speed, as a multiple of the inflow's own

const browser = await chromium.launch( {
	headless: true,
	executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
	args: [
		'--headless=new', '--enable-unsafe-webgpu', '--enable-features=Vulkan',
		'--use-angle=d3d11', '--ignore-gpu-blocklist', '--disable-gpu-sandbox',
		'--disable-background-timer-throttling', '--disable-backgrounding-occluded-windows',
		'--disable-renderer-backgrounding', '--disable-features=CalculateNativeWinOcclusion'
	]
} );

const page = await browser.newPage();
page.on( 'pageerror', ( e ) => console.log( '[pageerror]', e.message ) );
page.on( 'console', ( m ) => { const t = m.text(); if ( /error|Error/.test( t ) ) console.log( '[console]', t.slice( 0, 300 ) ); } );

await page.addInitScript( ( [ frames, sample ] ) => {

	let stored, counter = 0;
	window.__health = { samples: [], counters: { rejected: 0, breakdowns: 0, converged: 0, frames: 0 }, geometry: null };
	window.__done = false;

	Object.defineProperty( window, '__fluxflowProbe', {
		configurable: true,
		get: () => stored,
		set: ( probe ) => {

			stored = probe;
			const solver = probe.solver;
			// A scene may step a wrapper -- the smoke and fire solvers own a
			// grid solver rather than being one -- so the pressure solver
			// and the collider weights are looked for one level down too.
			const inner = solver.pressureSolver ? solver : ( solver.solver ?? solver );
			const original = solver.onAdvanceTimeStep;

			solver.onAdvanceTimeStep = async ( dt ) => {

				if ( window.__done ) return;
				await original( dt );

				try {

				const n = counter ++;
				const d = inner.pressureSolver.diagnostics;
				const c = window.__health.counters;
				c.frames ++;
				if ( d.rejected ) c.rejected ++;
				if ( d.stoppedBy && d.stoppedBy !== 'none' ) c.breakdowns ++;
				if ( d.converged === true ) c.converged ++;

				if ( n >= frames ) { window.__done = true; probe.stop && probe.stop(); return; }
				if ( n % sample !== 0 ) return;

				const g = probe.velocityGrid;
				const dims = g.dataSizeU.length;
				const [ NUx, NUy, NUz = 1 ] = g.dataSizeU;
				const [ NVx, NVy, NVz = 1 ] = g.dataSizeV;
				const [ NWx = NUx - 1, NWy = NUy, NWz = 2 ] = g.dataSizeW ?? [];
				const NX = NUx - 1, NY = NUy, NZ = NUz;

				const [ u, v, w ] = await Promise.all( [
					g.dataU.toArray(), g.dataV.toArray(),
					dims === 3 ? g.dataW.toArray() : Promise.resolve( new Float32Array( 0 ) )
				] );

				// The solver's own view of its geometry, read once.
				if ( window.__health.geometry === null ) {

					const fw = inner.colliderFaceWeights && inner.colliderFaceWeights.fields;
					const masks = inner.pressureSolver;
					window.__health.geometry = {
						dims,
						wu: fw ? Array.from( await fw.u.toArray() ) : null,
						wv: fw ? Array.from( await fw.v.toArray() ) : null,
						ww: fw && fw.w ? Array.from( await fw.w.toArray() ) : null,
						pinned: masks.dirichletMask ? Array.from( await masks.dirichletMask.toArray() ) : null,
						vent: masks.ventMask ? Array.from( await masks.ventMask.toArray() ) : null,
						hasWeights: Boolean( fw ),
						size: { NX, NY, NZ, NUx, NUy, NVx, NVy, NWx, NWy }
					};

				}

				const geo = window.__health.geometry;

				// Does the solved region hold still? A vent is static; a free
				// surface is not, and a region that moves is exactly the
				// thing every criterion here assumes away -- net flux across
				// its boundary is then not zero, it is what moves it. Read
				// again each sample rather than assumed, because the scenes
				// that need this are the ones nobody remembers to declare.
				let regionMoved = false;

				if ( geo.pinned ) {

					const now = await inner.pressureSolver.dirichletMask.toArray();
					for ( let i = 0; i < now.length; i ++ ) {

						if ( ( now[ i ] > 0.5 ) !== ( geo.pinned[ i ] > 0.5 ) ) { regionMoved = true; break; }

					}

				}
				const WU = geo.wu ? ( i, j, k ) => geo.wu[ i + NUx * j + NUx * NUy * k ] : () => 1;
				const WV = geo.wv ? ( i, j, k ) => geo.wv[ i + NVx * j + NVx * NVy * k ] : () => 1;
				const WW = geo.ww ? ( i, j, k ) => geo.ww[ i + NWx * j + NWx * NWy * k ] : () => 1;
				const PINNED = geo.pinned ? ( i, j, k ) => geo.pinned[ i + NX * j + NX * NY * k ] > 0.5 : () => false;
				const VENT = geo.vent ? ( i, j, k ) => geo.vent[ i + NX * j + NX * NY * k ] > 0.5 : () => false;

				const U = ( i, j, k ) => u[ i + NUx * j + NUx * NUy * k ];
				const V = ( i, j, k ) => v[ i + NVx * j + NVx * NVy * k ];
				const W = ( i, j, k ) => ( dims === 3 ? w[ i + NWx * j + NWx * NWy * k ] : 0 );

				let nonFinite = 0;
				for ( const a of [ u, v, w ] ) for ( const x of a ) if ( ! Number.isFinite( x ) ) nonFinite ++;

				// 1. weighted flux through each x plane, and how far the
				//    planes disagree with the inlet's own
				const flux = new Array( NUx ).fill( 0 );
				let inletOpenArea = 0;
				for ( let i = 0; i < NUx; i ++ ) {

					let q = 0;
					for ( let k = 0; k < NUz; k ++ ) for ( let j = 0; j < NUy; j ++ ) {

						const weight = WU( i, j, k );
						q += weight * U( i, j, k );
						if ( i === 0 ) inletOpenArea += weight;

					}
					flux[ i ] = q;

				}

				const inletFlux = flux[ 0 ];

				// *** Across the boundary of the SOLVED region, not of the
				// grid ***
				//
				// A vent is a region of pinned pressure. The solver does not
				// make those cells divergence-free and does not correct the
				// velocities inside them -- mantaflow says the same thing in
				// one line, "don't change velocities in outflow cells" --
				// so the grid's own outermost faces carry whatever the
				// outflow boundary condition last wrote there, constrained
				// by nothing. Measuring conservation across them measures
				// the boundary condition's bookkeeping rather than the
				// solver's.
				//
				// An earlier version of this probe did exactly that and
				// reported three scenes as losing all of their mass. On
				// example 17 the grid's top face reads 840.9 while the row
				// three cells inside the fluid reads 14.0, and on example 19
				// the same pair reads 93.9 against -0.03. Example 15 read
				// 22% out of balance at the grid edge and 0.5% across its
				// own fluid region. None of them had a conservation problem.
				//
				// What is summed here is every face with fluid on exactly
				// one side -- the other being a pinned cell or outside the
				// grid -- which is the fluid region's own surface whatever
				// shape the vent is, and it is where the projection's
				// promise actually applies.
				let net = 0, gross = 0;

				{

					const perFace = { 'x-': 0, 'x+': 0, 'y-': 0, 'y+': 0, 'z-': 0, 'z+': 0 };
					const solved = ( i, j, k ) => i >= 0 && j >= 0 && k >= 0 && i < NX && j < NY && k < NZ && ! PINNED( i, j, k );

					// One pass per axis over every face, keeping the ones
					// with fluid on exactly one side, signed outward.
					const sweep = ( size, weightOf, valueOf, axis, minus, plus ) => {

						const [ sx, sy, sz ] = size;
						for ( let k = 0; k < sz; k ++ ) for ( let j = 0; j < sy; j ++ ) for ( let i = 0; i < sx; i ++ ) {

							const lower = axis === 0 ? [ i - 1, j, k ] : axis === 1 ? [ i, j - 1, k ] : [ i, j, k - 1 ];
							const upper = [ i, j, k ];
							const lowerSolved = solved( ...lower );
							const upperSolved = solved( ...upper );
							if ( lowerSolved === upperSolved ) continue;

							// outward from the fluid: +value when the fluid
							// is on the lower side, -value when on the upper
							const value = weightOf( i, j, k ) * valueOf( i, j, k );
							const outward = lowerSolved ? value : - value;
							net += outward;
							gross += Math.abs( outward );
							perFace[ lowerSolved ? plus : minus ] += outward;

						}

					};

					sweep( [ NUx, NUy, NUz ], WU, U, 0, 'x-', 'x+' );
					sweep( [ NVx, NVy, NVz ], WV, V, 1, 'y-', 'y+' );
					if ( dims === 3 ) sweep( [ NWx, NWy, NWz ], WW, W, 2, 'z-', 'z+' );

					window.__lastFaces = perFace;

				}

				// The scene's own velocity scale: from its inlet when it has
				// one, and otherwise from the traffic across its boundary.
				const speedScale = inletOpenArea > 0 && Math.abs( inletFlux ) > 0
					? Math.abs( inletFlux ) / inletOpenArea
					: ( gross > 0 ? gross / ( 2 * inletOpenArea ) : 1 );

				// A vent cell does not conserve -- it is where mass is
				// allowed to leave -- so the comparison stops at the first
				// plane that touches one.
				let lastFluidPlane = NUx - 1;
				for ( let i = 0; i < NX; i ++ ) {

					let touches = false;
					for ( let k = 0; k < NZ && ! touches; k ++ ) for ( let j = 0; j < NY; j ++ ) if ( VENT( i, j, k ) ) { touches = true; break; }
					if ( touches ) { lastFluidPlane = i; break; }

				}

				let worstFlux = 0, worstFluxPlane = 0;
				for ( let i = 0; i <= lastFluidPlane; i ++ ) {

					const deviation = Math.abs( flux[ i ] - inletFlux ) / Math.max( Math.abs( inletFlux ), 1e-9 );
					if ( deviation > worstFlux ) { worstFlux = deviation; worstFluxPlane = i; }

				}

				// 2. weighted divergence, in cells the solver actually
				//    promises to make divergence-free
				let worstDiv = 0, worstDivCell = null, fluidCells = 0;
				for ( let k = 0; k < NZ; k ++ ) for ( let j = 0; j < NY; j ++ ) for ( let i = 0; i < NX; i ++ ) {

					if ( PINNED( i, j, k ) ) continue;
					const total = WU( i, j, k ) + WU( i + 1, j, k ) + WV( i, j, k ) + WV( i, j + 1, k ) + WW( i, j, k ) + WW( i, j, k + 1 );
					if ( total < 1e-6 ) continue;
					fluidCells ++;

					const div =
						WU( i + 1, j, k ) * U( i + 1, j, k ) - WU( i, j, k ) * U( i, j, k ) +
						WV( i, j + 1, k ) * V( i, j + 1, k ) - WV( i, j, k ) * V( i, j, k ) +
						( dims === 3 ? WW( i, j, k + 1 ) * W( i, j, k + 1 ) - WW( i, j, k ) * W( i, j, k ) : 0 );

					if ( Math.abs( div ) > worstDiv ) { worstDiv = Math.abs( div ); worstDivCell = [ i, j, k ]; }

				}

				// 3. what is moving through faces the collider has closed
				let solidLeak = 0, solidLeakFace = null;
				const scan = ( size, Wf, Vf, label ) => {

					const [ sx, sy, sz ] = size;
					for ( let k = 0; k < sz; k ++ ) for ( let j = 0; j < sy; j ++ ) for ( let i = 0; i < sx; i ++ ) {

						if ( Wf( i, j, k ) > 0 ) continue;
						const value = Math.abs( Vf( i, j, k ) );
						if ( value > solidLeak ) { solidLeak = value; solidLeakFace = `${ label }(${ i },${ j },${ k })`; }

					}

				};
				if ( geo.wu ) { scan( [ NUx, NUy, NUz ], WU, U, 'u' ); scan( [ NVx, NVy, NVz ], WV, V, 'v' ); if ( dims === 3 ) scan( [ NWx, NWy, NWz ], WW, W, 'w' ); }

				// 4. how fast the fluid is, on open faces outside the vent
				let fluidSpeed = 0;
				for ( let k = 0; k < NUz; k ++ ) for ( let j = 0; j < NUy; j ++ ) for ( let i = 0; i <= lastFluidPlane; i ++ ) {

					if ( WU( i, j, k ) <= 0 ) continue;
					const value = Math.abs( U( i, j, k ) );
					if ( value > fluidSpeed ) fluidSpeed = value;

				}

				window.__health.samples.push( {
					// "Driven" has to be relative to the scene's own traffic, not
					// an absolute floor. A closed scene's inlet plane reads
					// -0.0 or some rounding dust, and against an absolute
					// threshold that once made a free-surface scene look
					// driven with a reference speed of 1e-12, so every
					// velocity in it came out as hundreds of times "the
					// inflow".
					n, dims, regionMoved, hasWeights: Boolean( geo.wu ), driven: Math.abs( inletFlux ) > 0.01 * Math.max( gross, 1e-12 ),
					nonFinite, inletFlux, speedScale, worstFlux, worstFluxPlane,
					worstDiv, worstDivCell, solidLeak, solidLeakFace, fluidSpeed, fluidCells, lastFluidPlane,
					outletFlux: flux[ NUx - 1 ], arrivalFlux: flux[ lastFluidPlane ], net, gross, perFace: window.__lastFaces
				} );

				} catch ( error ) {

					window.__health.error = `${ error && error.message } | ${ error && error.stack }`.slice( 0, 500 );
					window.__done = true;

				}

			};

		}
	} );

}, [ FRAMES, SAMPLE ] );

await page.goto( URL, { waitUntil: 'load' } );
await page.waitForFunction( () => window.__fluxflowProbe !== undefined, undefined, { timeout: 120000 } );

const backend = await page.evaluate( () => window.__fluxflowProbe?.renderer?.backend?.constructor?.name ?? 'unknown' );
console.log( `backend: ${ backend }` );
if ( backend !== 'WebGPUBackend' ) { console.log( 'FATAL: not WebGPU, nothing below would be evidence' ); await browser.close(); process.exit( 1 ); }

await page.waitForFunction( () => window.__done === true, undefined, { timeout: 900000 } ).catch( () => console.log( '(ran out of time before the frame count)' ) );

const health = await page.evaluate( () => window.__health );
if ( health.error ) { console.log( 'probe failed inside the page:', health.error ); await browser.close(); process.exit( 1 ); }
await browser.close();

const { samples, counters } = health;
if ( samples.length === 0 ) { console.log( 'no samples' ); process.exit( 1 ); }

// The scene's own scales, taken from its first healthy sample.
const first = samples[ 0 ];
const U = first.speedScale;

function verdictFor( s ) {

	if ( s.nonFinite > 0 ) return `${ s.nonFinite } non-finite values`;

	// A moving solved region puts every other criterion out of scope.
	if ( s.regionMoved ) return null;

	// Cross-sections are only meaningful where solid faces can be excluded.
	if ( s.driven && s.hasWeights && s.worstFlux > FLUX_TOLERANCE ) return `flux through plane x=${ s.worstFluxPlane } is off the inlet's by ${ ( s.worstFlux * 100 ).toFixed( 1 ) }%`;

	// Averaged, not instantaneous. An unsteady wake makes the boundary's own
	// flux oscillate as vortices pass through it -- measured at up to 7% on
	// a scene that conserves mass perfectly well -- so the conservation
	// statement is about the mean, and judging the instantaneous value calls
	// a healthy scene broken.
	// Only where an inlet pins it. With an inflow, the projection has to
	// push out exactly what comes in, and the balance is a real promise. A
	// scene driven from inside with a vent as its only opening has no such
	// promise to check: a pinned cell absorbs whatever is pushed into it,
	// which is what this formulation does and what mantaflow's own empty
	// outflow cells do too. This probe cannot certify those scenes, and
	// saying so is better than inventing a criterion for them.
	if ( s.driven && s.meanBalance !== null && s.meanBalance > BALANCE_TOLERANCE ) {

		return `net flux across the boundary averages ${ s.meanNet.toFixed( 1 ) } against ${ s.meanGross.toFixed( 1 ) } crossing it, ${ ( s.meanBalance * 100 ).toFixed( 1 ) }% out of balance`;

	}

	// Only where the scene defines a speed to be a multiple OF. A
	// buoyancy-driven scene -- smoke, fire, an explosion -- has no inlet,
	// so there is no such number, and an earlier draft invented one and
	// duly called a plume broken for reaching 20 when it had decided the
	// scale was 1. For those scenes the verdict rests on the two criteria
	// that need no scale: nothing non-finite, and a boundary that neither
	// gains nor loses mass.
	if ( s.driven && s.fluidSpeed > SPEED_BOUND * U ) return `fluid speed ${ s.fluidSpeed.toFixed( 2 ) }, ${ ( s.fluidSpeed / U ).toFixed( 1 ) }x the inflow`;
	return null;

}

// A scene starts from rest, so its first frames do not satisfy anything --
// the inlet is admitting mass into a domain where nothing is moving yet.
// Rather than skip a fixed number of frames, the probe waits for the scene
// to establish itself: the first sample that meets every criterion is the
// baseline, and only a violation AFTER that is a scene coming apart. Never
// meeting them at all is its own verdict, and a different one.
// The mean is only defined once the scene has established itself, so the
// two are found together: establishment first, ignoring the balance, then
// the running mean from there on.
const WINDOW = 20;   // samples averaged over, so one vortex's worth of swing cannot decide anything
const ARRIVED = 0.9; // of the inlet flux at the outlet: the flow has crossed the domain
for ( const s of samples ) { s.meanBalance = null; s.meanNet = 0; s.meanGross = 0; }

// The balance says nothing while the domain is still filling -- the outlet
// carries almost nothing then, by construction -- so it is judged only
// after the flow has reached it, which the run itself says rather than a
// frame number.
// A driven channel has to fill before its boundary can balance, and the run
// says when that happened rather than a frame number. A scene with no inlet
// has nothing to fill and starts balanced.
const driven = Math.abs( samples[ 0 ].inletFlux ) > 0;
const arrivedAt = driven
	? samples.findIndex( ( s ) => Math.abs( s.outletFlux ) >= ARRIVED * Math.abs( s.inletFlux ) )
	: 0;

if ( arrivedAt >= 0 ) {

	for ( let i = arrivedAt + WINDOW - 1; i < samples.length; i ++ ) {

		let net = 0, gross = 0;
		for ( let w = i - WINDOW + 1; w <= i; w ++ ) { net += samples[ w ].net; gross += samples[ w ].gross; }
		samples[ i ].meanNet = net / WINDOW;
		samples[ i ].meanGross = gross / WINDOW;
		samples[ i ].meanBalance = samples[ i ].meanGross > 1e-9
			? Math.abs( samples[ i ].meanNet ) / samples[ i ].meanGross
			: 0;

	}

}

const establishedAt = samples.findIndex( ( s ) => verdictFor( s ) === null );

let firstBad = null;
if ( establishedAt >= 0 ) {

	for ( let i = establishedAt + 1; i < samples.length; i ++ ) if ( verdictFor( samples[ i ] ) !== null ) { firstBad = { sample: samples[ i ], index: i }; break; }

}

console.log( `\n${ counters.frames } frames, sampled every ${ SAMPLE }; ${ first.dims }D; inflow ${ U.toFixed( 3 ) } per open face, inlet flux ${ first.inletFlux.toFixed( 1 ) }` );
if ( ! first.hasWeights ) console.log( 'no collider face weights from this solver, so cross-sections are reported and the mass balance decides' );
if ( ! first.driven ) console.log( 'no inlet: nothing in this formulation forces a vent-only domain to balance, so the verdict here rests on finiteness alone and everything else is reported' );
console.log( `solver counters (reported, not part of the verdict): ${ counters.converged } converged, ${ counters.rejected } rejected, ${ counters.breakdowns } CG breakdowns` );

console.log( '\n                 VERDICT CRITERIA                 |      reported only' );
console.log( 'frame    interior flux   out/in    fluid speed  |   max div   solid faces   net imbalance' );

const step = Math.max( 1, Math.floor( samples.length / 40 ) );
for ( let i = 0; i < samples.length; i += step ) {

	const s = samples[ i ];
	const ratio = s.driven ? s.outletFlux / s.inletFlux : NaN;
	console.log(
		`${ String( s.n ).padStart( 6 ) }  ${ ( s.driven ? ( s.worstFlux * 100 ).toFixed( 2 ) + '%' : '-' ).padStart( 13 ) }  ` +
		`${ ( Number.isFinite( ratio ) ? ratio.toFixed( 2 ) : '-' ).padStart( 7 ) }  ${ s.fluidSpeed.toFixed( 2 ).padStart( 11 ) }  |  ` +
		`${ s.worstDiv.toExponential( 2 ).padStart( 8 ) }  ${ s.solidLeak.toExponential( 2 ).padStart( 11 ) }  ` +
		`${ s.meanBalance === null ? '          -' : ( s.meanBalance * 100 ).toFixed( 2 ).padStart( 10 ) + '%' }`
	);

}

if ( establishedAt < 0 ) {

	const last = samples[ samples.length - 1 ];
	console.log( `
VERDICT: NEVER ESTABLISHED -- no sample in ${ counters.frames } frames met every criterion` );
	console.log( `   last sample (frame ${ last.n }): ${ verdictFor( last ) }` );
	process.exit( 2 );

}

console.log( `
the flow reaches the outlet at ${ arrivedAt >= 0 ? 'frame ' + samples[ arrivedAt ].n : 'no point in this run' }; ` +
	`established at frame ${ samples[ establishedAt ].n } (the startup transient, judged from the data rather than skipped by count)` );

{
	const last = samples[ samples.length - 1 ];
	if ( last.perFace ) console.log( 'outward flux across the fluid region, per side, at the end: ' + Object.entries( last.perFace ).map( ( [ k, v ] ) => `${ k } ${ v.toFixed( 2 ) }` ).join( ', ' ) );
}

if ( firstBad ) {

	console.log( `\nVERDICT: BROKEN, first at frame ${ firstBad.sample.n } -- ${ verdictFor( firstBad.sample ) }` );
	const last = samples[ samples.length - 1 ];
	console.log( `   at the end (frame ${ last.n }): interior flux off by ${ ( last.worstFlux * 100 ).toFixed( 1 ) }%, ${ ( last.outletFlux / ( last.inletFlux || 1 ) ).toFixed( 2 ) }x leaving, fluid speed ${ last.fluidSpeed.toFixed( 2 ) }` );
	process.exit( 2 );

} else {

	const settled = samples.slice( establishedAt );
	const worst = settled.reduce( ( a, s ) => ( s.worstFlux > a.worstFlux ? s : a ), settled[ 0 ] );
	const last = samples[ samples.length - 1 ];
	console.log( `\nVERDICT: HEALTHY over ${ counters.frames } frames` );
	console.log( `   worst interior flux deviation ${ ( worst.worstFlux * 100 ).toFixed( 2 ) }% at frame ${ worst.n }; ${ ( last.outletFlux / ( last.inletFlux || 1 ) ).toFixed( 3 ) }x leaving at the end; every sample inside every bound` );
	if ( last.regionMoved ) console.log( '   note: the solved region moves between samples -- a free surface -- so every criterion but finiteness is out of scope here. This is a weak pass, not a certificate' );
	else if ( ! last.driven ) console.log( '   note: with no inlet, the only criterion that applied was finiteness. This is a weak pass, not a certificate' );

}
