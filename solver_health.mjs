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
// How long to let the page run before giving up on the frame count. The
// default suits the couple-thousand-frame runs this is normally used for; a
// 12,000-frame run on a 3D scene needs more, so long_run.mjs raises it.
const TIMEOUT = Number( process.env.SOLVER_HEALTH_TIMEOUT ?? 900000 );

// Thresholds, all relative to the scene's own inflow.
const FLUX_TOLERANCE = 0.05;    // of the inlet flux, across interior cross-sections
const BALANCE_TOLERANCE = 0.05; // of the inlet flux, what enters against what leaves
const SPEED_BOUND = 6;          // fluid speed, as a multiple of the inflow's own
// What the projection may leave behind, as a fraction of what it was asked
// to remove. Calibrated rather than picked: across the three 3D scenes the
// healthy median runs 4.5e-7 to 4.7e-6 and the worst single sample seen is
// 1.6e-3, while the same scenes with the dot product defect put back run a
// median of 5.0e-3 and 5.9 with worst samples of 1.3 and 7.4. A per-sample
// bar at 1e-2 sits six times above anything healthy and well below the
// broken side's own upper half.
const RESIDUAL_TOLERANCE = 1e-2;

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

// SOLVER_HEALTH_TOLERANCE overrides the scene's own pressure tolerance for the
// run. For answering "would this scene be stable at a looser tolerance" without
// editing the scene, which is a question about the solver rather than about the
// scene's source.
const TOLERANCE = process.env.SOLVER_HEALTH_TOLERANCE ? Number( process.env.SOLVER_HEALTH_TOLERANCE ) : null;

await page.addInitScript( ( [ frames, sample, tolerance ] ) => {

	let stored, counter = 0;
	window.__health = { samples: [], counters: { rejected: 0, breakdowns: 0, converged: 0, frames: 0, breakdownFrames: [], stopReasons: {}, zeroRhsFrames: 0, stopResiduals: [], stopsOnSolved: 0, atRestFrames: 0, stoppedGrowing: 0, belowFloor: null, belowFloorFrames: 0, overFloor: [], floorRelative: [], iterations: [], maxIterationsSeen: 0, maxOverL2: [] }, geometry: null };
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
			if ( tolerance !== null && inner.pressureSolver ) inner.pressureSolver.settings.tolerance = tolerance;

			// Ask the solver what floor its own arithmetic can verify a residual
			// against. Off by default in the library because it costs a host read per
			// solve; this probe exists to explain a verdict, so it pays.
			if ( inner.pressureSolver ) inner.pressureSolver.settings.reportNoiseFloor = true;

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
				// *** A guard that stops an already-solved system is the guard
				// working, and counting it as a breakdown was this probe's fifth
				// false positive ***
				//
				// linalg.js's applySnapshot already refuses to name this case,
				// in a comment that gives the reason: there is nothing left to
				// solve, so p is ~0, so p.Ap is ~0, and alpha forced to 0 leaves
				// x exactly where it belongs. That test guards one of the five
				// sites that set stoppedBy; the other four set it unconditionally,
				// so the reason reaches here anyway.
				//
				// Measured on examples/33-flip-dam-break-3d/, whose liquid comes
				// to rest partway through a 12,000-step run: one run reported
				// 7,557 stops from frame 4444 and another 2,589 from frame 9412,
				// every one of them degenerate-pAp and every one leaving a
				// residual of exactly 0. Two other runs of the same build
				// reported none at all.
				//
				// This trusts d.residual to be a true residual, which is exactly
				// what the 3D dot product defect made false -- a broken reducer
				// can report 0 for a system that is nowhere near solved. What
				// keeps it honest is that the residual is not the only thing
				// measured: the sampled divergence below is computed here from the
				// velocity field itself, independently of anything the solver
				// says about its own progress.
				// How many frames handed the solver a system that was already
				// solved -- a scene at rest. Counted independently of any stop
				// reason, because once linalg.js stops reporting the non-event
				// there is otherwise no way to tell a run whose liquid settled
				// from one whose liquid never did, and those two runs are exactly
				// what has to be distinguished to know the library fix fired.
				if ( d.residual === 0 ) c.atRestFrames ++;

				const stoppedOnASolvedSystem = d.stoppedBy && d.stoppedBy !== 'none' && d.residual === 0;

				if ( stoppedOnASolvedSystem ) c.stopsOnSolved ++;

				// *** 'residual-growing' is not a breakdown, and lumping it in with
				// the corruption guards was a mistake this probe made on the day
				// that reason was added ***
				//
				// The four guards this criterion was calibrated against --
				// degenerate-pAp, pAp-growth, alpha-magnitude, degenerate-oldRZ --
				// fire when the iteration is about to produce garbage. A scene
				// reaching them after it has established itself is in trouble, which
				// is why any of them decides the verdict.
				//
				// 'residual-growing' is the opposite kind of event: the solve noticed
				// it was walking away from its own best answer and stopped BEFORE any
				// harm, returning a usable iterate. Whether that mattered is a
				// question about the answer it returned, and the residual criterion
				// already measures exactly that, independently.
				//
				// Measured, which is what settles it rather than the argument above:
				// examples/14-stable-fluids/ over 12,001 steps converges on 11,999 of
				// them and hits this reason twice, at frames 4852 and 4863, with every
				// residual sample inside the bar -- counted as breakdowns that made
				// the scene BROKEN on two frames in twelve thousand. Meanwhile
				// examples/25-dye-injection/ is BROKEN either way, on the residual
				// criterion, at 1.54e+1 against a bar of 1e-2. So this change lets a
				// healthy scene pass without letting a broken one through, which is
				// the only form of evidence that justifies relaxing a criterion.
				const stoppedWhileStillSafe = d.stoppedBy === 'residual-growing';
				if ( stoppedWhileStillSafe ) c.stoppedGrowing ++;

				if ( d.stoppedBy && d.stoppedBy !== 'none' && ! stoppedOnASolvedSystem && ! stoppedWhileStillSafe ) {

					c.breakdowns ++; c.breakdownFrames.push( n );
					// WHY, not just how often. A guard that fires because the
					// system it was handed is degenerate -- a settled free surface
					// whose right-hand side is zero, so the answer is x = 0 and
					// there is nothing for pAp to be -- is the guard working, and
					// reads identically in a count to a real breakdown. The
					// residual norm at the stop separates them: b = 0 means
					// nothing was asked of the solve.
					c.stopReasons[ d.stoppedBy ] = ( c.stopReasons[ d.stoppedBy ] ?? 0 ) + 1;
					// The residual the stop left behind. diagnostics.residual is
					// sqrt(|r.r|) and is set every frame, so this costs nothing and
					// is the number that separates the two cases.
					if ( d.residual === 0 ) c.zeroRhsFrames ++;
					if ( c.stopResiduals.length < 40 ) c.stopResiduals.push( d.residual );

				}
				if ( d.converged === true ) c.converged ++;

				// The answer to "why does this never converge", when it is the answer:
				// the tolerance asked for is under what float32 can verify, so no
				// iteration count reaches it. Recorded once -- it is a property of the
				// configuration, not news on every frame.
				if ( d.toleranceBelowFloor && c.belowFloor === null ) c.belowFloor = { ...d.toleranceBelowFloor, relative: d.noiseFloorRelative, frame: n };
				if ( d.toleranceBelowFloor ) c.belowFloorFrames ++;
				if ( Number.isFinite( d.residualOverFloor ) ) c.overFloor.push( d.residualOverFloor );
				// The floor's own distribution, which is what setting a scene's tolerance
				// needs: it moves frame to frame with |x|, so one sample -- least of all
				// the first frame that happened to exceed the tolerance -- is not the
				// number to choose from.
				if ( Number.isFinite( d.noiseFloorRelative ) ) c.floorRelative.push( d.noiseFloorRelative );
				// Iterations actually spent, which is what says whether raising the cap
				// costs anything: a cap only binds on the frames that reach it.
				// max|r| against |r|_2: mantaflow stops on the former, this port on the
				// latter. Their ratio says whether the choice matters on this scene.
				if ( Number.isFinite( d.maxResidual ) && Number.isFinite( d.residual ) && d.residual > 0 ) c.maxOverL2.push( d.maxResidual / d.residual );
				if ( Number.isFinite( d.iterations ) ) { c.iterations.push( d.iterations ); if ( d.iterations > c.maxIterationsSeen ) c.maxIterationsSeen = d.iterations; }

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

				// How well the projection itself did, which needs no control
				// surface and so is the one thing measurable on every scene
				// -- including the ones where every flux criterion is out of
				// scope. residual is what CG reports; ||b|| is what it was
				// asked to remove.
				let relativeResidual = null;

				{

					const bHost = await inner.pressureSolver.b.toArray();
					let bSquared = 0;
					for ( let i = 0; i < bHost.length; i ++ ) bSquared += bHost[ i ] * bHost[ i ];
					const bNorm = Math.sqrt( bSquared );
					if ( bNorm > 1e-12 && Number.isFinite( d.residual ) ) relativeResidual = Math.abs( d.residual ) / bNorm;

				}

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
					n, dims, regionMoved, relativeResidual, stoppedBy: d.stoppedBy, hasWeights: Boolean( geo.wu ), driven: Math.abs( inletFlux ) > 0.01 * Math.max( gross, 1e-12 ),
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

}, [ FRAMES, SAMPLE, TOLERANCE ] );

await page.goto( URL, { waitUntil: 'load' } );
await page.waitForFunction( () => window.__fluxflowProbe !== undefined, undefined, { timeout: 120000 } );

const backend = await page.evaluate( () => window.__fluxflowProbe?.renderer?.backend?.constructor?.name ?? 'unknown' );
console.log( `backend: ${ backend }` );
if ( TOLERANCE !== null ) console.log( `pressure tolerance overridden to ${ TOLERANCE }` );
if ( backend !== 'WebGPUBackend' ) { console.log( 'FATAL: not WebGPU, nothing below would be evidence' ); await browser.close(); process.exit( 1 ); }

await page.waitForFunction( () => window.__done === true, undefined, { timeout: TIMEOUT } ).catch( () => console.log( '(ran out of time before the frame count)' ) );

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

	// How much divergence the projection actually removed. This needs no
	// control surface, so unlike everything below it applies to every scene
	// -- including the free-surface and no-inlet ones where the flux
	// criteria are out of scope and finiteness was previously all that was
	// left. Example 34 spent its entire life not converging a single frame
	// and read HEALTHY for it; this is the criterion that catches that.
	if ( s.relativeResidual !== null && s.relativeResidual > RESIDUAL_TOLERANCE ) {

		return `the projection left ${ s.relativeResidual.toExponential( 2 ) } of the divergence it was asked to remove`;

	}

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

// A CG breakdown while the scene is starting from rest is one thing -- the
// healthy runs of every scene here have at most one, at frame 0 -- and one
// after it has settled is another: the same scenes with the dot product
// defect put back break down 48 and 225 times, from frame 83 and frame 0 on
// to frame 997. Counted over the whole run rather than per sample, because
// a breakdown can land on a frame this probe does not sample.
const establishedFrame = establishedAt >= 0 ? samples[ establishedAt ].n : 0;
const lateBreakdowns = counters.breakdownFrames.filter( ( f ) => f > establishedFrame );

let firstBad = null;
if ( establishedAt >= 0 ) {

	for ( let i = establishedAt + 1; i < samples.length; i ++ ) if ( verdictFor( samples[ i ] ) !== null ) { firstBad = { sample: samples[ i ], index: i }; break; }

}

console.log( `\n${ counters.frames } frames, sampled every ${ SAMPLE }; ${ first.dims }D; inflow ${ U.toFixed( 3 ) } per open face, inlet flux ${ first.inletFlux.toFixed( 1 ) }` );
if ( ! first.hasWeights ) console.log( 'no collider face weights from this solver, so cross-sections are reported and the mass balance decides' );
if ( ! first.driven ) console.log( 'no inlet: nothing in this formulation forces a vent-only domain to balance, so the verdict here rests on finiteness alone and everything else is reported' );
console.log( `solver counters: ${ counters.converged } converged, ${ counters.rejected } rejected, ${ counters.breakdowns } CG breakdowns` +
	( counters.breakdownFrames.length ? ` (first at frame ${ counters.breakdownFrames[ 0 ] }, last ${ counters.breakdownFrames[ counters.breakdownFrames.length - 1 ] })` : '' ) );
if ( counters.maxOverL2 && counters.maxOverL2.length ) {

	const sorted = counters.maxOverL2.slice().sort( ( a, b ) => a - b );
	const q = ( f ) => sorted[ Math.min( sorted.length - 1, Math.floor( f * ( sorted.length - 1 ) ) ) ];
	console.log( `   max|r| / |r|_2: median ${ q( 0.5 ).toFixed( 3 ) }, 90th ${ q( 0.9 ).toFixed( 3 ) }, worst ${ q( 1 ).toFixed( 3 ) }` +
		` -- mantaflow stops on the max norm where this stops on the L2 one; a ratio that stays put means the choice does not matter here` );

}
if ( counters.iterations && counters.iterations.length ) {

	const sorted = counters.iterations.slice().sort( ( a, b ) => a - b );
	const q = ( f ) => sorted[ Math.min( sorted.length - 1, Math.floor( f * ( sorted.length - 1 ) ) ) ];
	const mean = sorted.reduce( ( a, v ) => a + v, 0 ) / sorted.length;
	console.log( `   iterations per solve: mean ${ mean.toFixed( 1 ) }, median ${ q( 0.5 ) }, 90th ${ q( 0.9 ) }, worst ${ counters.maxIterationsSeen }` );

}
if ( counters.floorRelative && counters.floorRelative.length ) {

	const sorted = counters.floorRelative.slice().sort( ( a, b ) => a - b );
	const q = ( f ) => sorted[ Math.min( sorted.length - 1, Math.floor( f * ( sorted.length - 1 ) ) ) ];
	console.log( `   the floor itself, relative to |b|, over ${ sorted.length } samples: median ${ q( 0.5 ).toExponential( 2 ) }, 90th ${ q( 0.9 ).toExponential( 2 ) }, worst ${ q( 1 ).toExponential( 2 ) }` );
	console.log( `   a tolerance for this scene has to clear the worst of those, not the median -- the floor moves with |x| every frame` );

}
if ( counters.overFloor && counters.overFloor.length ) {

	const sorted = counters.overFloor.slice().sort( ( a, b ) => a - b );
	const median = sorted[ Math.floor( sorted.length / 2 ) ];
	const worst = sorted[ sorted.length - 1 ];
	console.log( `   residual as a multiple of the floor: median ${ median.toFixed( 1 ) }, worst ${ worst.toFixed( 1 ) }` +
		` -- ~1 means the solve is at what float32 allows; larger means the budget or the preconditioner is the constraint, not precision` );

}
if ( counters.belowFloor ) {

	const f = counters.belowFloor;
	console.log( `   *** the tolerance asked for is BELOW this solver's own noise floor, on ${ counters.belowFloorFrames } of ${ counters.frames } frames ***` );
	console.log( `   asked for a residual under ${ f.requested.toExponential( 2 ) }; float32 can only verify b - Ax down to ${ f.floor.toExponential( 2 ) }` +
		( f.relative !== null && f.relative !== undefined ? ` (${ f.relative.toExponential( 2 ) } relative)` : '' ) +
		`, first seen at frame ${ f.frame }` );
	console.log( `   no iteration count reaches it. Raise the tolerance above the floor or accept that this scene cannot report convergence.` );

}
if ( counters.stoppedGrowing ) console.log( `   ${ counters.stoppedGrowing } solves stopped early because the residual was growing away from their best -- reported, not counted as breakdowns, because the residual criterion judges what they returned` );
if ( counters.atRestFrames ) console.log( `   ${ counters.atRestFrames } frames handed the solve a system already at a residual of exactly 0 -- the scene was at rest for those` );
if ( counters.stopsOnSolved ) console.log( `   ${ counters.stopsOnSolved } guard stops on a system already solved to a residual of exactly 0 -- not counted as breakdowns, see the note in this file` );
if ( counters.breakdowns ) console.log( `   stopped by: ${ Object.entries( counters.stopReasons ).map( ( [ k, v ] ) => `${ k } x${ v }` ).join( ', ' ) }` +
	( counters.zeroRhsFrames ? `; ${ counters.zeroRhsFrames } of those left a residual of exactly 0, meaning the guard tripped on a system that was already solved -- see linalg.js's applySnapshot, which declines to name that a breakdown` : '' ) );
if ( counters.stopResiduals && counters.stopResiduals.length ) console.log( `   residual left at the first ${ counters.stopResiduals.length } stops: ${ counters.stopResiduals.map( ( r ) => ( r === null ? 'null' : r.toExponential( 1 ) ) ).join( ', ' ) }` );
{
	const withResidual = samples.filter( ( s ) => s.relativeResidual !== null );
	if ( withResidual.length ) {
		const sorted = withResidual.map( ( s ) => s.relativeResidual ).sort( ( a, b ) => a - b );
		const q = ( f ) => sorted[ Math.floor( f * ( sorted.length - 1 ) ) ];
		console.log( `residual the projection left, relative to what it was asked to remove: median ${ q( 0.5 ).toExponential( 2 ) }, 90th ${ q( 0.9 ).toExponential( 2 ) }, worst ${ q( 1 ).toExponential( 2 ) }` );
	}
}

console.log( '\n                 VERDICT CRITERIA                 |      reported only' );
console.log( 'frame    interior flux   out/in    fluid speed  |   max div   solid faces   net imbalance   residual/|b|' );

const step = Math.max( 1, Math.floor( samples.length / 40 ) );
for ( let i = 0; i < samples.length; i += step ) {

	const s = samples[ i ];
	const ratio = s.driven ? s.outletFlux / s.inletFlux : NaN;
	console.log(
		`${ String( s.n ).padStart( 6 ) }  ${ ( s.driven ? ( s.worstFlux * 100 ).toFixed( 2 ) + '%' : '-' ).padStart( 13 ) }  ` +
		`${ ( Number.isFinite( ratio ) ? ratio.toFixed( 2 ) : '-' ).padStart( 7 ) }  ${ s.fluidSpeed.toFixed( 2 ).padStart( 11 ) }  |  ` +
		`${ s.worstDiv.toExponential( 2 ).padStart( 8 ) }  ${ s.solidLeak.toExponential( 2 ).padStart( 11 ) }  ` +
		`${ s.meanBalance === null ? '          -' : ( s.meanBalance * 100 ).toFixed( 2 ).padStart( 10 ) + '%' }  ` +
		`${ s.relativeResidual === null ? '           -' : s.relativeResidual.toExponential( 2 ).padStart( 12 ) }`
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

if ( ! firstBad && lateBreakdowns.length > 0 ) {

	console.log( `
VERDICT: BROKEN -- ${ lateBreakdowns.length } CG breakdowns after the scene established itself at frame ${ establishedFrame }, the first at frame ${ lateBreakdowns[ 0 ] }` );
	process.exit( 2 );

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

	// Only quote a figure the criteria actually used. The flux and outlet
	// ratios divide by an inlet flux, so on a scene with no inlet they are a
	// division by something near zero -- one of them printed as
	// "21932594728469.85%" next to a HEALTHY verdict, which is how a report
	// loses the reader's trust even when the verdict is right.
	const inScope = [];
	if ( last.driven && last.hasWeights ) inScope.push( `worst interior flux deviation ${ ( worst.worstFlux * 100 ).toFixed( 2 ) }% at frame ${ worst.n }` );
	if ( last.driven ) inScope.push( `${ ( last.outletFlux / last.inletFlux ).toFixed( 3 ) }x leaving at the end` );
	{
		const withResidual = samples.filter( ( x ) => x.relativeResidual !== null ).map( ( x ) => x.relativeResidual );
		if ( withResidual.length ) inScope.push( `the projection's worst leftover residual ${ Math.max( ...withResidual ).toExponential( 2 ) } of what it was asked to remove` );
	}
	console.log( `   ${ inScope.join( '; ' ) }${ inScope.length ? '; ' : '' }every sample inside every bound that applied` );

	// Name what did NOT apply, rather than letting a pass stand for more than
	// it covers. The residual criterion needs no geometry, so it is in scope
	// even here -- worth saying, because it is the one criterion the dot
	// product defect could not hide from.
	if ( last.regionMoved ) console.log( '   note: the solved region moves between samples -- a free surface -- so the flux and mass-balance criteria are out of scope. What still applied: finiteness, the projection residual, and CG breakdowns. A narrower pass than a fixed-region scene gets' );
	else if ( ! last.driven ) console.log( '   note: with no inlet, the flux and mass-balance criteria are out of scope. What still applied: finiteness, the projection residual, and CG breakdowns. A narrower pass than a driven scene gets' );

}
