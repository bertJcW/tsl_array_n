// A fully autonomous, deterministic solver-stability test -- no
// mouse/keyboard input at all, per the user's own explicit request after
// an earlier mouse-driven version of this file (see git history) failed
// on real WebGPU hardware three different ways in a row (a long dye
// streak, a stall, then an instant NaN-looking blackout) without a
// confirmed root cause: rather than keep guessing at fixes for a
// mouse-driven scenario this dev sandbox can't reproduce at all (MGPCG
// doesn't even compile here), this removes every source of
// nondeterminism -- timing, human input, pointer-velocity computation --
// to isolate a single question: with a fixed, simple, always-the-same
// forcing pattern, does createGridSolver2's pipeline (external forces,
// Dirichlet-aware MGPCG pressure projection, real self-advection, closed-
// domain boundary conditions) stay numerically stable over many frames,
// or does it blow up regardless of what's driving it?
//
// The forcing itself: a small fixed region near the left wall gets a
// constant rightward push, every frame, forever -- picture a small fan
// built into that wall. Dye is injected in the same region so the flow
// is visible. Nothing here depends on wall-clock time or user action.
//
// Dye uses the same four-field (state + rawAdvected) x2 ping-pong
// structure as examples/12-interactive-advection/'s own dye handling --
// see that file's header comment for why (a single field written by two
// different tsl_array_n.kernel() objects crashes this dev sandbox's
// WebGL2 fallback backend).
//
// A periodic full diagnostic readout (velocity/pressure/dye/divergence-
// RHS min/max/sum, every 30 frames) is logged to the console --
// deliberately more verbose than a normal example, specifically so that
// if this still fails on real hardware, the failure can be diagnosed
// from actual data (which field diverges first, how fast, whether it's
// gradual or sudden) instead of another guess. The same non-finite guard
// as the mouse-driven version freezes the loop and reports the exact
// frame number the moment any field goes non-finite.
//
// *** The first version of this diagnostic had its own real bug, worth
// remembering generally: a plain running min/max over an array silently
// swallows NaN instead of reporting it ***
//
// `NaN < x` and `NaN > x` are *always* false in JS, so `if (v < lo) lo =
// v` never fires for a NaN entry -- it's skipped with no trace, and an
// array that's entirely NaN leaves `lo`/`hi` at their untouched
// `Infinity`/`-Infinity` starting sentinels, while an array that's a MIX
// of NaN and real zeros reports a misleadingly clean-looking `[0, 0]`.
// The very first real-hardware run of this file's autonomous version
// showed exactly that: `p [Infinity, -Infinity]` (consistent with an
// all-NaN or empty pressure readback) alongside `u [0.0000, 0.0000]`
// (consistent with a mix of real zeros and silently-ignored NaN) --
// *not* evidence that pressure was literally infinite, just this
// diagnostic failing to say "NaN" plainly. Fixed by explicitly counting
// non-finite entries and reporting that count (and the *finite* range
// separately) rather than folding non-finite values into the same
// min/max scan; also now reports the divergence RHS (`b`) itself, whose
// sum should be ~0 for a fully closed domain (a discrete divergence-
// theorem consequence -- boundary-normal velocity is zeroed by
// grid_blocked_boundary_condition_solver2.js before b is built, so every
// interior term telescopes away, leaving only the -- zeroed -- boundary
// terms) -- if it isn't, that itself is a real, worth-knowing finding.
//
// In this dev sandbox: constructs and runs every frame without throwing
// (confirmed via manual frame stepping, including checking the
// diagnostic readout itself), but MGPCG's atomic dot product fails to
// *compile* here -- the exact same `ERROR: '&' : syntax error` examples
// 04/05/07/13 all hit. Needs the user's real WebGPU hardware to confirm
// actual stability.
//
// *** Real-hardware round 2 found and fixed a genuine bug in the pressure
// solver itself, worth remembering generally: CG on a singular operator
// can divide by an exact zero, and this port's own fixed-point atomic
// reduction makes hitting that zero *more* likely than in a textbook
// double-precision CG ***
//
// The above diagnostic readout (this file's own summarize()/fmt()) showed
// something specific on real hardware: at frame 0, b (the divergence RHS)
// was completely finite and well-posed (sum ~0, as the closed-domain
// compatibility note two paragraphs up predicts) -- yet pressure was
// *already* 100% non-finite that same frame, before it had a chance to
// drift from anything upstream. That rules out "accumulated drift over
// many frames" and points squarely at the very first pressure solve()
// call itself producing NaN from a valid input.
//
// The actual mechanism: this scene has no `dirichlet` option at all (a
// deliberate choice -- this is meant to be a plain closed box, matching
// Jos Stam's own classic Stable Fluids setup, not this port's separate
// Dirichlet-pin feature), so the Laplacian CG solves against is a pure
// Neumann (all zero-flux) operator -- singular, with the constant field
// in its null space (see grid_pressure_solver2.js's own header comment
// for the general compatibility condition this implies). CG never
// reduces a residual's null-space component (A@constant=0 exactly, so
// that part is invisible to every dot product CG computes), so over
// enough iterations the search direction p drifts to be dominated by it
// -- at which point Ap collapses toward 0 everywhere, and p.Ap (the
// denominator of CG's own alpha=r.z/p.Ap step) heads toward an exact
// zero. jet's own reference solver (cg-inl.h, plain double-precision CPU
// arithmetic) would only hit a truly-exact zero here in a rare exact
// floating-point coincidence -- but this port's own GPU-atomic dot
// product (linalg.js, needed since WGSL has no float atomics) quantizes
// every dot product to a fixed-point integer before summing, so a
// small-but-nonzero p.Ap can round all the way down to the integer 0
// well *before* p is anywhere near purely null-space-aligned -- a second,
// much easier way to hit the same exact-zero division, unique to this
// port's own reduction strategy.
//
// Fixed in linalg.js: both CG solve() functions now check every division
// site's denominator against isDegenerateDot() (a threshold tied directly
// to the atomic scale's own quantization floor) and break out cleanly
// (keeping whatever pressure already holds) instead of dividing by
// (near-)zero. grid_pressure_solver2.js now also exposes
// diagnostics.converged (logged below alongside b/pressure) so a false
// here can be told apart from a genuine bug -- for this deliberately
// singular closed-box scene, an early, non-converged stop is an expected,
// honest outcome of the underlying math, not a defect in the fix itself.
//
// *** Round 2: this file's own plain closed-box scene (no dirichlet, no
// collider) was later used, with a TEMPORARY added Dirichlet vent, as the
// minimal repro for a completely separate long-run divergence bug --
// resolved, not left in this file ***
//
// A real asymmetric red-black relaxation schedule in multigrid.js's own
// createMultigridPreconditioner (found by comparing against mantaflow's
// own multigrid solver) turned out to be the root cause of a long-run
// divergence that only manifested once a Dirichlet mask was active
// (examples/15/16's own outflow objects both use one) -- see that
// function's own header comment for the full investigation and fix, plus
// linalg.js's own beta/alpha robustness guards and grid_pressure_
// solver2.js's/grid_blocked_boundary_condition_solver2.js's own last-
// resort circuit breakers for pressure and velocity, added as additional
// defense-in-depth on top of the root-cause fix. This file itself has no
// Dirichlet mask at all (see the design note above) and was never
// actually at risk from that specific bug -- it served purely as the
// fastest-iterating diagnostic vehicle while chasing it, and has been
// restored to its own original, deliberate closed-box design now that
// the investigation is done.

import * as tsl_array_n from 'tsl_array_n';
import { vec2, float, length, clamp, max } from 'three/tsl';
import { grid } from 'fluxflow';

const canvas = document.querySelector( '#out' );
const statusEl = document.querySelector( '#status' );

function status( text, isErr ) {

	statusEl.textContent = text;
	statusEl.className = isErr ? 'err' : '';

}

const N = 32;
const dt = 1 / 30;
const sourceX = 3;
const sourceY = N / 2;
const sourceRadius = 4;
const pushStrength = 3; // constant rightward force within the source region
const injectionDensity = 1;
const dyeDecay = 0.995;
const diagnosticInterval = 30; // frames between console readouts

try {

	const renderer = await tsl_array_n.init( { canvas: document.createElement( 'canvas' ), allowFallback: true } );
	status( `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }` );

	const velocityGrid = grid.createFaceCenteredGrid2( N, N, 1, 1, 0, 0 );

	// dye state, ping-ponged -- see header comment for why this is four
	// fields, not two.
	const stateA = grid.createScalarGrid2( N, N, 1, 1, 0, 0 );
	const stateB = grid.createScalarGrid2( N, N, 1, 1, 0, 0 );
	stateA.clear();
	stateB.clear();

	const rawAdvectedA = { data: tsl_array_n.arrayN( 'float', [ N, N ] ) };
	const rawAdvectedB = { data: tsl_array_n.arrayN( 'float', [ N, N ] ) };

	// the *only* force: a constant rightward push in a small fixed region
	// near the left wall, every frame, unconditionally -- no time
	// dependence, no user input, always exactly the same.
	function force( pos ) {

		const sourceDist = length( pos.sub( vec2( sourceX, sourceY ) ) );
		const inSource = clamp( float( 1 ).sub( sourceDist.div( sourceRadius ) ) );

		return vec2( pushStrength, 0 ).mul( inSource );

	}

	const solver = grid.createGridSolver2( {
		velocityGrid,
		gridSpacing: [ 1, 1 ],
		origin: [ 0, 0 ],
		force,
		dt,
		pressure: { multigrid: { numberOfLevels: 4 }, tolerance: 1e-5, maxIterations: 100 }
	} );

	// dye's own advection, bound to the solver's already-projected
	// velocityGrid -- separate from grid_solver2.js's internal advection
	// (which only self-advects velocity), since dye isn't part of that
	// orchestrator's scope at all.
	const dyeAdvectionSolver = grid.createSemiLagrangianAdvectionSolver2( { velocityGrid: solver.velocityGrid, dt } );

	const advectAtoB = dyeAdvectionSolver.advectScalar2( stateA, rawAdvectedB );
	const advectBtoA = dyeAdvectionSolver.advectScalar2( stateB, rawAdvectedA );

	// same fixed source region as the force above -- paints dye there
	// every frame, decays existing dye elsewhere. A cross-field copy-with-
	// modification (reads rawAdvected, writes state), not self-touch, so
	// state's only writer stays this one kernel, matching example 12's own
	// single-writer-per-field structure.
	function createInjectKernel( rawAdvectedGrid, stateGrid ) {

		return tsl_array_n.kernel( stateGrid.dataSize, ( i, j ) => {

			const pos = stateGrid.dataPosition( i, j );
			const raw = rawAdvectedGrid.data( i, j );

			const sourceDist = length( pos.sub( vec2( sourceX, sourceY ) ) );
			const sourceInjection = clamp( float( 1 ).sub( sourceDist.div( sourceRadius ) ) ).mul( injectionDensity );

			stateGrid.data( i, j ).assign( max( raw.mul( dyeDecay ), sourceInjection ) );

		} );

	}

	const injectAtoB = createInjectKernel( rawAdvectedB, stateB );
	const injectBtoA = createInjectKernel( rawAdvectedA, stateA );

	const ctx = canvas.getContext( '2d' );
	const image = ctx.createImageData( N, N );

	function draw( data ) {

		for ( let j = 0; j < N; j ++ ) {

			for ( let i = 0; i < N; i ++ ) {

				const v = data[ i + N * j ];

				// canvas Y is down-positive, this grid's Y is up-positive --
				// flip rows so the image matches the simulation's own orientation
				const pixel = ( ( N - 1 - j ) * N + i ) * 4;
				const bright = 255 * Math.min( 1, Math.max( 0, v ) );

				image.data[ pixel ] = bright;
				image.data[ pixel + 1 ] = bright;
				image.data[ pixel + 2 ] = bright;
				image.data[ pixel + 3 ] = 255;

			}

		}

		ctx.putImageData( image, 0, 0 );

	}

	// unlike a naive min/max scan, this explicitly counts non-finite
	// entries instead of letting them silently fail every `<`/`>`
	// comparison (in JS, `NaN < x` and `NaN > x` are *always* false, so a
	// plain running min/max over an array containing NaN just skips it
	// without any indication -- an earlier version of this diagnostic had
	// exactly that bug, reporting a misleadingly clean-looking `[0, 0]` or
	// the untouched `[Infinity, -Infinity]` sentinel instead of surfacing
	// the NaN it was silently ignoring).
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

	async function logDiagnostics( frameNumber, dyeData ) {

		const [ uData, vData, pData, bData ] = await Promise.all( [
			velocityGrid.dataU.toArray(),
			velocityGrid.dataV.toArray(),
			solver.pressure.data.toArray(),
			solver.pressureSolver.b.toArray()
		] );

		console.log(
			`fluxflow stable-fluids [frame ${ frameNumber }] ` +
			`converged=${ solver.pressureSolver.diagnostics.converged } | ` +
			`${ fmt( 'dye', summarize( dyeData ) ) } | ` +
			`${ fmt( 'u', summarize( uData ) ) } | ` +
			`${ fmt( 'v', summarize( vData ) ) } | ` +
			`${ fmt( 'p', summarize( pData ) ) } | ` +
			`${ fmt( 'b(divergence)', summarize( bData ) ) }`
		);

	}

	let frame = 0;
	let nanDetected = false;

	// cheap: reuses the array already read back for drawing, no extra GPU
	// readback -- catches the same kind of failure logDiagnostics reports
	// on, just every frame instead of every diagnosticInterval frames.
	function checkForNonFinite( data, frameNumber ) {

		if ( nanDetected ) return;

		for ( let i = 0; i < data.length; i ++ ) {

			if ( ! Number.isFinite( data[ i ] ) ) {

				nanDetected = true;
				status( `non-finite dye value detected at frame ${ frameNumber } (index ${ i }, value ${ data[ i ] }) -- likely an unstable/diverged pressure solve, see this file's own header comment`, true );
				console.error( `fluxflow stable-fluids: non-finite dye value at frame ${ frameNumber }, index ${ i }:`, data[ i ] );
				return;

			}

		}

	}

	async function animate() {

		await solver.onAdvanceTimeStep( dt );

		let currentState;

		if ( frame % 2 === 0 ) {

			advectAtoB();
			injectAtoB();
			currentState = stateB;

		} else {

			advectBtoA();
			injectBtoA();
			currentState = stateA;

		}

		const data = await currentState.data.toArray();
		checkForNonFinite( data, frame );

		if ( ! nanDetected && frame % diagnosticInterval === 0 ) await logDiagnostics( frame, data );

		if ( ! nanDetected ) draw( data );

		frame ++;
		if ( ! nanDetected ) requestAnimationFrame( animate );

	}

	requestAnimationFrame( animate );

} catch ( error ) {

	status( error.message, true );
	console.error( error );

}
