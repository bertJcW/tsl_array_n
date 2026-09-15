// *** The null-net probe: shortlist item 1 of
// docs/machine-learning-fluid-research.md. ***
//
// The research document's conclusion is that the interesting machine-learning
// direction for this package is not "approximate the pressure solve more
// cheaply" -- the GPU is only a few per cent busy, so cheaper arithmetic buys
// almost nothing -- but "replace an iterative, host-synchronised loop with a
// fixed-length feed-forward one". A solver step is ~1416 dispatches, a
// data-dependent iteration count and several host round trips at ~1.1 ms
// each. A convolutional network of the candidate shape is 12 dispatches,
// one submission, and no round trips at all.
//
// That is an argument, not a measurement. This page is the measurement.
//
// The network here has **random weights**. It computes nonsense, and that is
// the entire point: what is being priced is the *shape*, and a shape can be
// priced before a single training run exists. If a 12-dispatch feed-forward
// pass does not beat the solve it would replace on this hardware, every
// direction in the research document's family A is dead and nothing was
// spent finding out. If it does, the rest of the ladder has a budget.
//
// Three things are reported, in order of how much they decide:
//
//   1. **Correctness of the kernels.** The GPU forward pass against
//      reference.js's float64 JavaScript one, on the same weights and the
//      same input. Without this the timing below is the timing of an
//      unknown computation. The reference executes the *plan* the builder
//      emitted, not a hand-transcription of the architecture, so it cannot
//      drift out of date.
//   2. **The cost of the shape.** Dispatches, submissions, CPU encode time
//      and (where the adapter supports timestamp queries) GPU time, per
//      forward pass, batched and unbatched.
//   3. **The comparison.** The same measurements for one multigrid-
//      preconditioned CG solve on the same 64x64 grid -- the thing a
//      learned pressure solve would be replacing.
//
// *** How the wall-clock number is taken honestly ***
//
// A `renderer.compute()` returns without waiting for the GPU, so timing a
// loop of forward passes with nothing else in it measures encoding, not
// execution. The loop below therefore ends with one readback, which drains
// the queue, and that single round trip is amortised over every pass in the
// loop. It is stated in the output rather than hidden, because this
// package's own performance history is a list of measurements that turned
// out to be measuring the harness (see the perf document's "The harness was
// wrong, and fixing it reversed the answer").
//
// Needs real WebGPU. The dev sandbox's WebGL2 fallback has an established
// history of unreliable repeated same-buffer dispatch/readback cycles in
// this project (examples/06's header records it), so a bad correctness
// number there is not evidence of anything.

import * as tsl_array_n from 'tsl_array_n';
import { ml, linalg, profiling } from 'fluxflow';

const pre = document.querySelector( '#status pre' );
const lines = [];

function log( text, cls = '' ) {

	lines.push( cls ? `<span class="${ cls }">${ text }</span>` : text );
	pre.innerHTML = lines.join( '\n' );

}

function check( label, ok, detail ) {

	log( `${ ok ? '✓' : '✗' } ${ label }${ detail ? ' — ' + detail : '' }`, ok ? 'ok' : 'err' );

}

const N = 64;
const SHAPE = [ N, N ];
const CHANNELS = 16;
const LEVELS = 3;
const SEED = 20260915;
const PASSES = 60;
const WARMUP = 15;

try {

	const renderer = await tsl_array_n.init( { allowFallback: true, trackTimestamp: true } );
	const backend = renderer.backend?.constructor?.name ?? 'unknown';
	log( `<span class="head">backend</span>: ${ backend }` );

	if ( ! backend.includes( 'WebGPU' ) ) {

		log( 'Running on the WebGL2 fallback. Timings here mean nothing and correctness is unreliable — see this file\'s header.', 'err' );

	}

	// ---------------------------------------------------------------
	// Build
	// ---------------------------------------------------------------

	const net = ml.createUNet2( {
		shape: SHAPE,
		channels: CHANNELS,
		levels: LEVELS,
		inChannels: 1,
		outChannels: 1,
		activation: 'relu',
		padding: 'clamp',
		seed: SEED
	} );

	log( '' );
	log( `<span class="head">the shape being priced</span>` );
	log( `  grid            ${ N }x${ N }, ${ LEVELS } levels, ${ CHANNELS } channels` );
	log( `  dispatches      ${ net.stats.dispatches } per forward pass` );
	log( `  weight blocks   ${ net.blocks.length }` );
	log( `  parameters      ${ net.stats.parameters.toLocaleString() } (${ ( net.stats.bytes / 1024 ).toFixed( 1 ) } KiB as float32)` );
	log( `  arithmetic      ${ ( net.stats.macs / 1e6 ).toFixed( 1 ) } MMAC = ${ ( net.stats.flops / 1e6 ).toFixed( 1 ) } MFLOP per pass` );

	// ---------------------------------------------------------------
	// 1. Correctness: GPU against the float64 reference
	// ---------------------------------------------------------------

	// Generate the weights here rather than using the constructor's own
	// randomize(), so the exact same numbers go to both arms and there is no
	// assumption about generator ordering to get wrong.
	const random = ml.createSeededRandom( SEED );
	const weightsByName = {};

	for ( const block of net.blocks ) {

		weightsByName[ block.name ] = {
			weights: ml.heNormalWeights( block.inChannels, block.kernelSize, block.outChannels, random ),
			// A zero bias would make the ReLU's clipping behaviour trivial and
			// hide a whole class of disagreement, so give it something.
			bias: Float32Array.from( { length: block.outChannels }, () => random() * 0.2 - 0.1 )
		};

	}

	net.loadWeights( weightsByName );

	// A smooth, signed input, in the spirit of the low-frequency right-hand
	// side examples/06 uses -- not noise, so a sign or transpose error shows
	// up as structure rather than as more noise.
	const inputData = new Float32Array( N * N );

	for ( let y = 0; y < N; y ++ ) {

		for ( let x = 0; x < N; x ++ ) {

			inputData[ x + N * y ] = Math.sin( 2 * Math.PI * x / N ) * Math.cos( 2 * Math.PI * y / N );

		}

	}

	net.input.fromArray( inputData );
	net.forward();

	const gpuOutput = Array.from( await net.output.toArray() );

	const referenceBuffers = ml.forwardReference(
		net.plan,
		Float64Array.from( inputData ),
		weightsByName,
		net.config
	);
	const cpuOutput = referenceBuffers.get( 'output' );

	// An empty readback is the established sandbox symptom and would make
	// `every()` vacuously true -- examples/07's header records that trap.
	if ( gpuOutput.length !== cpuOutput.length ) {

		check( 'GPU forward matches the float64 reference', false,
			`readback returned ${ gpuOutput.length } elements, expected ${ cpuOutput.length } — empty/short readback, not a numerical disagreement` );

	} else {

		let maxAbs = 0;
		let maxMagnitude = 0;

		for ( let i = 0; i < cpuOutput.length; i ++ ) {

			maxAbs = Math.max( maxAbs, Math.abs( gpuOutput[ i ] - cpuOutput[ i ] ) );
			maxMagnitude = Math.max( maxMagnitude, Math.abs( cpuOutput[ i ] ) );

		}

		const relative = maxMagnitude > 0 ? maxAbs / maxMagnitude : 0;
		const finite = gpuOutput.every( Number.isFinite );

		// float32 against float64 through 8 convolutions cannot match
		// exactly. A relative difference at the 1e-5 level is accumulated
		// rounding; orders of magnitude above that is a bug, not precision.
		check( 'GPU forward matches the float64 reference', finite && relative < 1e-4,
			`max |diff| = ${ maxAbs.toExponential( 3 ) }, relative = ${ relative.toExponential( 3 ) }, all finite = ${ finite }` );

	}

	// Determinism: no atomics anywhere in the layers, so two passes over the
	// same input must agree bit for bit. This is a property the CG solver
	// next door does *not* have, and it is worth confirming rather than
	// assuming.
	net.forward();
	const secondOutput = Array.from( await net.output.toArray() );
	const identical = secondOutput.length === gpuOutput.length
		&& secondOutput.every( ( v, i ) => Object.is( v, gpuOutput[ i ] ) );

	check( 'two forward passes are bit-identical', identical,
		identical ? 'no atomics, so no run-to-run reduction-order variation' : 'differs between runs — unexpected for this module' );

	// ---------------------------------------------------------------
	// 2. The cost of the shape
	// ---------------------------------------------------------------

	async function measure( label, run, iterations, drain ) {

		for ( let k = 0; k < WARMUP; k ++ ) run();
		await drain();

		profiling.resetProfiling();
		profiling.startProfiling();

		const t0 = performance.now();
		for ( let k = 0; k < iterations; k ++ ) run();
		// One round trip, amortised over `iterations` passes, to make the
		// wall time include GPU execution rather than encoding alone.
		await drain();
		const wallMs = performance.now() - t0;

		profiling.stopProfiling();

		const report = profiling.profilingReport( iterations );
		const gpuMs = await profiling.readComputeTimestampMs( renderer );

		return {
			label,
			iterations,
			wallMsPer: wallMs / iterations,
			dispatchesPer: report.dispatchesPerFrame,
			submissionsPer: report.submissionsPerFrame,
			cpuEncodeMsPer: report.cpuMsPerFrame,
			gpuMsPer: gpuMs === null ? null : gpuMs / iterations,
			labels: report.labels
		};

	}

	const drainNet = () => net.output.toArray();

	const batched = await measure( 'network, batched', () => net.forward(), PASSES, drainNet );
	const unbatched = await measure( 'network, unbatched', () => net.forwardUnbatched(), PASSES, drainNet );

	function report( m ) {

		log( `  ${ m.label.padEnd( 22 ) } ${ m.wallMsPer.toFixed( 3 ) } ms   ${ m.dispatchesPer.toFixed( 1 ) } dispatches   ${ m.submissionsPer.toFixed( 1 ) } submissions   ${ m.cpuEncodeMsPer.toFixed( 3 ) } ms encode` + ( m.gpuMsPer === null ? '' : `   ${ m.gpuMsPer.toFixed( 3 ) } ms GPU` ) );

	}

	log( '' );
	log( `<span class="head">cost per forward pass</span> (medianless: mean of ${ PASSES }, ${ WARMUP } warm-up passes discarded, one readback amortised over the loop)` );
	report( batched );
	report( unbatched );
	log( `  batching is worth ${ ( unbatched.wallMsPer / batched.wallMsPer ).toFixed( 2 ) }x here — same dispatches, ${ ( unbatched.submissionsPer / Math.max( batched.submissionsPer, 1e-9 ) ).toFixed( 0 ) }x the submissions`, 'note' );

	// ---------------------------------------------------------------
	// 3. The comparison arm: one MGPCG solve on the same grid
	// ---------------------------------------------------------------

	const gridSpacing = [ 1, 1 ];
	const applyLaplacian = linalg.createLaplacianOperator( SHAPE, gridSpacing );
	const applyPreconditioner = linalg.createMultigridPreconditioner( SHAPE, gridSpacing, { numberOfLevels: LEVELS + 1 } );

	const xExpected = tsl_array_n.arrayN( 'float', SHAPE );
	xExpected.fromArray( inputData );

	const b = tsl_array_n.arrayN( 'float', SHAPE );
	applyLaplacian( xExpected, b )();

	const x = tsl_array_n.arrayN( 'float', SHAPE );
	const solver = linalg.createPreconditionedConjugateGradientSolver( applyLaplacian, applyPreconditioner, b, x );

	// *** The comparison arm must run the solver the library actually ships,
	// not the one `solve()`'s own parameter defaults give you. ***
	//
	// `createPreconditionedConjugateGradientSolver().solve()` defaults to the
	// host path with a residual check every iteration -- the slow arm, kept
	// as the default for compatibility. The shipped defaults live in
	// grid_pressure_solver2.js: GPU-resident scalars, GPU-resident setup,
	// batched iterations, and a residual check every 4. Measuring against
	// anything else would flatter the network by several of the
	// optimisations this repo has already landed.
	//
	// This exact trap has been sprung here before: the perf document records
	// that example 15 passed `residualCheckInterval: 1` explicitly and so
	// "silently opted the scene every performance measurement runs on out of
	// the library default that had just been changed to 4". Naming the
	// options here is the fix, not the hazard -- the hazard is naming them
	// wrongly, so they are named to match grid_pressure_solver2.js's own
	// defaults and will need updating together with it.
	const SOLVE_ARGS = [
		1e-5, // tolerance
		100,  // maxIterations
		4,    // residualCheckInterval
		true, // gpuResidentScalars
		true, // batchIterations
		true, // gpuResidentSetup
		false // relativeTolerance
	];

	// A solve is already synchronous at the host -- it cannot finish without
	// reading its own residual -- so it needs no artificial drain.
	const SOLVES = 20;

	for ( let k = 0; k < 5; k ++ ) {

		x.fromArray( new Float32Array( N * N ) );
		await solver.solve( ...SOLVE_ARGS );

	}

	profiling.resetProfiling();
	profiling.startProfiling();

	const solveStart = performance.now();
	let iterationsTotal = 0;
	let convergedCount = 0;

	for ( let k = 0; k < SOLVES; k ++ ) {

		x.fromArray( new Float32Array( N * N ) );
		// solve() returns a boolean; the iteration count and the stop reason
		// live on the solver's own `state`.
		if ( await solver.solve( ...SOLVE_ARGS ) ) convergedCount ++;
		iterationsTotal += solver.state.iterations;

	}

	const solveWallMs = performance.now() - solveStart;
	profiling.stopProfiling();

	const solveReport = profiling.profilingReport( SOLVES );
	const solveGpuMs = await profiling.readComputeTimestampMs( renderer );

	log( '' );
	log( `<span class="head">what it would be replacing</span> (one multigrid-preconditioned CG solve, same ${ N }x${ N } grid, mean of ${ SOLVES })` );
	log( `  ${ 'MGPCG solve'.padEnd( 22 ) } ${ ( solveWallMs / SOLVES ).toFixed( 3 ) } ms   ${ solveReport.dispatchesPerFrame.toFixed( 1 ) } dispatches   ${ solveReport.submissionsPerFrame.toFixed( 1 ) } submissions   ${ solveReport.cpuMsPerFrame.toFixed( 3 ) } ms encode` + ( solveGpuMs === null ? '' : `   ${ ( solveGpuMs / SOLVES ).toFixed( 3 ) } ms GPU` ) );

	log( `  ${ ( iterationsTotal / SOLVES ).toFixed( 1 ) } CG iterations per solve on average, ${ convergedCount }/${ SOLVES } converged, stopped by: ${ solver.state.stoppedBy }`, 'note' );

	if ( convergedCount < SOLVES ) {

		log( '  Not every solve converged, so the solve arm is timing a capped iteration count rather than a completed solve. Fix that before quoting the ratio below.', 'err' );

	}

	// ---------------------------------------------------------------
	// The answer
	// ---------------------------------------------------------------

	const wallRatio = ( solveWallMs / SOLVES ) / batched.wallMsPer;
	const dispatchRatio = solveReport.dispatchesPerFrame / batched.dispatchesPer;

	log( '' );
	log( `<span class="head">the number this page exists for</span>` );
	log( `  a forward pass of the candidate shape is ${ wallRatio.toFixed( 2 ) }x ${ wallRatio >= 1 ? 'faster' : 'SLOWER' } than the solve it would replace,` );
	log( `  at ${ dispatchRatio.toFixed( 1 ) }x fewer dispatches and ${ ( solveReport.submissionsPerFrame / Math.max( batched.submissionsPer, 1e-9 ) ).toFixed( 1 ) }x fewer submissions.` );
	log( '' );

	if ( wallRatio < 2 ) {

		log( 'Under 2x. That is not enough headroom: a trained network has to be *accurate* as well as fast, and family A of the research document needs the speed to pay for the accuracy it gives up. Read this as the direction being unattractive on this hardware, and say so in the document rather than training anything.', 'err' );

	} else {

		log( `Over 2x, so the shape is affordable and the question becomes accuracy rather than cost. That is what the rest of the research document's ladder is for — and note that this says nothing whatsoever about whether a *trained* network of this shape can produce a usable pressure field. It cannot, yet. Nothing here is trained.`, 'note' );

	}

	log( '' );
	log( 'Per-layer breakdown of the unbatched arm (the batched one reports itself as a single entry and cannot say what was inside it):', 'note' );

	for ( const entry of unbatched.labels ) {

		log( `  ${ entry.label.padEnd( 22 ) } ${ entry.callsPerFrame.toFixed( 1 ) } calls   ${ entry.cpuMsPerFrame.toFixed( 4 ) } ms encode   ${ ( entry.cpuShare * 100 ).toFixed( 1 ) }%`, 'note' );

	}

} catch ( error ) {

	check( 'probe', false, error?.message ?? String( error ) );
	console.error( error );

}
