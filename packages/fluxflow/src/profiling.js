// Dispatch-level profiling for the compute path.
//
// *** Why this exists ***
//
// Two separate investigations into this port's pressure-solve performance
// (../docs/perf-investigation-cg-gpu-resident-alpha-beta.md, both rounds)
// ended at the same place: "dispatch count is what costs" -- inferred from
// frames-per-second deltas across implementation variants, never measured.
// Both rounds' own conclusions say, in as many words, that the next step is
// real instrumentation rather than another implementation built on the
// inference. This is that instrumentation.
//
// It answers three questions the fps number cannot:
//
//   1. **How many dispatches is a frame actually made of?** Guessed at
//      repeatedly, never counted. Every kernel in linalg.js and multigrid.js
//      -- the whole pressure-solve hot path -- is built through one function,
//      `buildElementwiseKernel`, so counting there catches all of them.
//   2. **How much wall-clock time goes into *issuing* them?** A WebGPU
//      dispatch call returns without waiting for the GPU, so the time spent
//      inside the call is pure CPU-side encoding overhead. If that number is
//      most of the frame, the fix is fewer dispatches and nothing else will
//      help.
//   3. **How much GPU time does the work actually take?** three.js exposes
//      WebGPU timestamp queries; the difference between that and the frame's
//      wall time is what is being lost to encoding and synchronisation.
//   4. **How many *submissions* is it made of?** Added later, and it turned
//      out to be the question that mattered most. Every `renderer.compute()`
//      call is its own command buffer, its own compute pass and its own queue
//      submit, and on this hardware one submission costs **38.8 us of wall
//      time regardless of how much work it carries** -- measured by holding
//      the dispatch count fixed at 64 and varying only the number of
//      submits, with the JavaScript side of it costing 3 us and the GPU
//      executing all 64 dispatches in 0.052 ms. Example 15 measured 272
//      submissions and ~800 dispatches per frame, so roughly a third of that
//      frame was the submit path. `dispatches` alone hides this completely:
//      fourteen one-dispatch submissions and one fourteen-dispatch
//      submission are the same number and not remotely the same cost.
//
// Question 2 is the one that decides the next piece of work, and it is
// cheap: no GPU features required, no browser flags, works on any backend.
// Questions 3 and 4 need different care -- see readComputeTimestampMs and
// the report's own `submissionsPerFrame`.
//
// *** Cost when off ***
//
// One boolean test per dispatch. The wrapper is installed unconditionally,
// because kernels are built once at construction time and profiling is
// usually switched on afterwards -- an install-time switch would mean
// deciding to profile before building anything, which is exactly the
// ordering that makes a profiler annoying enough not to use.

// Deliberately a module-level singleton rather than an object threaded
// through every factory: the whole point is to be able to switch it on from
// a console in a running page, without a scene having had to plan for it.
const state = {
	enabled: false,
	dispatches: 0,
	// Submissions, counted alongside dispatches because they turned out to be
	// two different costs -- see the file header's "submissions" note.
	submissions: 0,
	cpuMs: 0,
	byLabel: new Map()
};

function record( label, cpuMs ) {

	state.dispatches ++;
	state.cpuMs += cpuMs;

	let entry = state.byLabel.get( label );

	if ( entry === undefined ) {

		entry = { calls: 0, cpuMs: 0 };
		state.byLabel.set( label, entry );

	}

	entry.calls ++;
	entry.cpuMs += cpuMs;

}

/**
 * Wraps a dispatcher so that, while profiling is on, each call is counted
 * and its CPU-side cost accumulated under `label`.
 *
 * Returns the dispatcher unchanged in spirit -- same arguments, same return
 * value -- so it is safe to wrap unconditionally at construction time.
 */
export function instrumentDispatch( label, dispatch ) {

	const wrapped = function instrumentedDispatch( ...args ) {

		if ( ! state.enabled ) return dispatch( ...args );

		const t0 = performance.now();
		const result = dispatch( ...args );
		// One dispatcher call is one renderer.compute(): one command buffer,
		// one compute pass, one queue submit.
		state.submissions ++;
		record( label, performance.now() - t0 );

		return result;

	};

	// Forwarded so a wrapped dispatcher can still be handed to
	// tsl_array_n's dispatchBatch, which submits several kernels in one
	// command buffer by reaching for this. A batched dispatcher never runs
	// the wrapper above, so batched work is counted by profileBatch instead
	// -- see its own comment.
	wrapped.computeNode = dispatch.computeNode;

	return wrapped;

}

/**
 * Records a batched submission: `count` dispatches that went out together,
 * under one label, with the batch's own CPU cost.
 *
 * Batching is the point of tsl_array_n's dispatchBatch, and it necessarily
 * bypasses the per-dispatch wrapper above -- the individual dispatchers are
 * never called, only their compute nodes are collected. So the batch has to
 * report itself, and the per-label breakdown inside it is not available.
 * That is the trade: to see where dispatches go by label, turn batching off
 * (every batching caller here keeps a switch for exactly that reason) and
 * re-run.
 */
export function profileBatch( label, count, run ) {

	if ( ! state.enabled ) return run();

	const t0 = performance.now();
	// A batch is one renderer.compute() carrying `count` dispatches.
	state.submissions ++;
	const result = run();
	const cpuMs = performance.now() - t0;

	// Attributed as `count` dispatches so the per-frame dispatch total stays
	// comparable across the batched and unbatched paths -- the whole reason
	// to measure this is to see that number fall.
	for ( let i = 0; i < count; i ++ ) record( label, i === 0 ? cpuMs : 0 );

	return result;

}

/** Starts counting, from zero. */
// *** Phase timing, for the parts that are not dispatches ***
//
// Counting dispatches and submissions says nothing about where a *wait*
// goes, and waits are most of a solve: the fixed cost of one pressure
// solve was measured at about as much as all of its CG iterations put
// together, with roughly four host round trips in it. A round trip's cost
// IS the wall time of its await -- it cannot return until the queue in
// front of it has drained -- so timing the await directly is exact rather
// than inferred, and it correctly attributes the queued work to the thing
// that waits for it.
const phases = new Map();

/** Records `ms` spent in a named phase. No-op unless profiling is on. */
export function markPhase( label, ms ) {

	if ( ! state.enabled ) return;

	const entry = phases.get( label );

	if ( entry === undefined ) phases.set( label, { calls: 1, ms } );
	else {

		entry.calls ++;
		entry.ms += ms;

	}

}

/**
 * Times `run()` under `label` and returns its result. Async-aware: the
 * await is the point, so the caller must await this too.
 */
export async function timePhase( label, run ) {

	if ( ! state.enabled ) return run();

	const t0 = performance.now();
	const result = await run();
	markPhase( label, performance.now() - t0 );

	return result;

}

export function startProfiling() {

	resetProfiling();
	state.enabled = true;

}

/** Stops counting. Accumulated numbers stay readable. */
export function stopProfiling() {

	state.enabled = false;

}

export function resetProfiling() {

	state.dispatches = 0;
	state.submissions = 0;
	phases.clear();
	state.cpuMs = 0;
	state.byLabel.clear();

}

export function isProfiling() {

	return state.enabled;

}

/**
 * A snapshot of what has been counted since the last reset.
 *
 * `frames` is whatever the caller says it is -- this module has no idea what
 * a frame is -- and exists so the per-frame figures, which are the ones
 * worth reading, come out of here rather than being recomputed by every
 * caller.
 */
export function profilingReport( frames = 1 ) {

	const labels = [ ...state.byLabel.entries() ]
		.map( ( [ label, entry ] ) => ( {
			label,
			calls: entry.calls,
			callsPerFrame: entry.calls / frames,
			cpuMs: entry.cpuMs,
			cpuMsPerFrame: entry.cpuMs / frames,
			// The share of encoding time this label is responsible for --
			// the number that says where to spend effort.
			cpuShare: state.cpuMs > 0 ? entry.cpuMs / state.cpuMs : 0
		} ) )
		.sort( ( a, b ) => b.cpuMs - a.cpuMs );

	return {
		frames,
		dispatches: state.dispatches,
		dispatchesPerFrame: state.dispatches / frames,
		phases: [ ...phases.entries() ]
			.map( ( [ label, e ] ) => ( {
				label,
				calls: e.calls,
				callsPerFrame: e.calls / frames,
				ms: e.ms,
				msPerFrame: e.ms / frames
			} ) )
			.sort( ( a, b ) => b.ms - a.ms ),
		submissions: state.submissions,
		submissionsPerFrame: state.submissions / frames,
		// How much work each submission carries. The number to raise: measured
		// on example 15, a submission costs ~38.8 us of wall time whether it
		// carries one dispatch or sixty-four, and the GPU executes all 64 in
		// 0.052 ms.
		dispatchesPerSubmission: state.submissions > 0 ? state.dispatches / state.submissions : 0,
		cpuMs: state.cpuMs,
		cpuMsPerFrame: state.cpuMs / frames,
		labels
	};

}

/**
 * GPU time spent in compute passes since the last read, in milliseconds, or
 * null when it is not available.
 *
 * Requires the renderer to have been created with `trackTimestamp: true`
 * (tsl_array_n's `init()` forwards its options straight to the
 * WebGPURenderer constructor, so `init({ canvas, trackTimestamp: true })` is
 * all it takes) *and* the adapter to support the `timestamp-query` feature.
 *
 * *** Call this every frame, or the number will be null. ***
 *
 * Both conditions above do hold on the development machine this was measured
 * on (verified directly: `backend.hasFeature('timestamp-query')` is true),
 * and the first rounds of this investigation still recorded null, because
 * three's compute query pool is 2048 queries = 1024 timestamped passes while
 * a frame is ~272 passes, and its pool allocates a fresh uid per frame
 * (`updateTimeStampUID` keys off `info.compute.frameCalls`). Left unresolved,
 * the pool saturates after about four frames, prints
 * "Maximum number of queries exceeded" once, and returns null from then on.
 * Resolving per frame keeps it inside its budget; that is what the examples'
 * probes now do.
 * Neither is guaranteed, which is why this returns null rather than throwing
 * -- the CPU-side numbers above are the ones the decision rests on, and they
 * always work.
 */
export async function readComputeTimestampMs( renderer ) {

	if ( ! renderer || renderer.trackTimestamp !== true ) return null;

	try {

		await renderer.resolveTimestampsAsync( 'compute' );

		const value = renderer.info?.compute?.timestamp;
		return typeof value === 'number' ? value : null;

	} catch ( error ) {

		// A backend without timestamp support rejects here rather than
		// reporting the feature as missing up front; a profiler that throws
		// is worse than one that says "not available".
		return null;

	}

}
