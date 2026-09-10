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
//
// Question 2 is the one that decides the next piece of work, and it is
// cheap: no GPU features required, no browser flags, works on any backend.
// Question 3 needs `trackTimestamp` and is best-effort -- see
// readComputeTimestampMs.
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

	return function instrumentedDispatch( ...args ) {

		if ( ! state.enabled ) return dispatch( ...args );

		const t0 = performance.now();
		const result = dispatch( ...args );
		record( label, performance.now() - t0 );

		return result;

	};

}

/** Starts counting, from zero. */
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
