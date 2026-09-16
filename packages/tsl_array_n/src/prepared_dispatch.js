// Encoding a batch of compute dispatches without re-entering three.js's
// per-call machinery.
//
// *** What this is for ***
//
// `renderer.compute( nodes )` resolves node state, bindings and a pipeline
// for every dispatch and wraps the array in its own command encoder, compute
// pass and queue submission. Measured on fluxflow's
// examples/15-flow-past-cylinder/ by wrapping three.js's own methods: ~33 us
// per compute() call and ~3.5 us per dispatch, against 0.56 us to encode a
// dispatch and 4.44 us to submit a command buffer through the WebGPU API
// directly. On a step of 999 dispatches in 67 calls that is ~6 ms of host
// time for ~1.1 ms of API work, while the GPU itself is busy 0.1 ms.
//
// The industry answer to that gap is "record once, replay" -- ONNX Runtime
// Web's graph capture, Babylon.js's snapshot rendering, WebGPU's own render
// bundles. None of those is available here: WebGPU has no compute bundle and
// a command buffer cannot be submitted twice (gpuweb#4138, gpuweb#1971). So
// what is recorded is not the command buffer but the *resolution* above it:
// which pipeline, which bind groups, how many workgroups.
//
// *** What is borrowed and what is not ***
//
// Nothing of three.js's implementation is reproduced here. The values this
// module needs are asked of three.js through its own objects -- it hands
// back the bind groups, the pipeline and the workgroup counts it already
// computed and cached -- and this module then calls the public WebGPU API
// (setPipeline / setBindGroup / dispatchWorkgroups). In particular the
// workgroup-count arithmetic is *not* reimplemented: the first run of every
// batch goes through `renderer.compute()`, which computes and caches it, and
// later runs read that cached value back.
//
// *** What still goes through three.js on every dispatch, and why ***
//
// `nodes.updateForCompute` and `bindings.updateForCompute`. Those are what
// upload changed uniforms, and this package re-exports three's own
// `uniform()` rather than wrapping it, so a caller can change a value
// without this module ever seeing it. Skipping them would make a stale `dt`
// a silent wrong answer, which is not a trade worth 2 us. What is skipped is
// the per-call encoder/pass/submission bookkeeping and the dispatch-time
// indirection, which is where the money is.
//
// *** When it refuses to run ***
//
// - the first execution of a batch, which is what populates the caches;
// - a renderer whose backend is not WebGPU (no device to encode against);
// - `trackTimestamp`, because three.js writes its timestamp queries around
//   its own compute passes, and a pass encoded here would be invisible to
//   them -- profiling has to measure the thing everyone else runs;
// - anything whose resolution does not look the way this module expects
//   (an indirect dispatch size, a missing cache entry). It falls back to
//   `renderer.compute()` rather than guessing.

export const settings = {
	// Runtime switch so both paths exist in one build: this project's
	// measurements are paired within a single run, and a constructor-time
	// choice can only be compared across runs.
	preparedDispatch: true
};

/**
 * Wraps a fixed list of compute nodes in a dispatcher that encodes them
 * directly once three.js has resolved them.
 *
 * @param {Array} nodes - compute nodes, in execution order.
 * @param {Function} fallback - runs the same nodes through the renderer.
 * @return {Function} the dispatcher.
 */
export function createPreparedDispatcher( nodes, fallback ) {

	// Populated after the first run through three.js; null means "not
	// resolved yet, or resolved into something this module will not encode".
	let prepared = null;
	let preparedFor = null;

	return function dispatchPrepared( renderer ) {

		if ( settings.preparedDispatch !== true ) {

			prepared = null;
			return fallback();

		}

		const backend = renderer.backend;
		const device = backend !== undefined ? backend.device : undefined;

		if ( device === undefined || backend.isWebGPUBackend !== true || backend.trackTimestamp === true ) {

			return fallback();

		}

		if ( prepared === null || preparedFor !== device ) {

			fallback();
			prepared = resolve( renderer, nodes );
			preparedFor = device;
			return;

		}

		encode( renderer, device, prepared );

	};

}

// Asks three.js for what it resolved, after it has resolved it. Returns null
// if anything is missing or is a shape this module does not encode, so the
// caller keeps using the renderer instead.
function resolve( renderer, nodes ) {

	const backend = renderer.backend;
	const resolved = [];

	for ( const node of nodes ) {

		// three.js caches the workgroup counts for a node whose dispatch
		// size is a plain count -- which is every kernel() in this package.
		// An indirect or externally supplied dispatch size is not handled
		// here and sends the whole batch back to the renderer.
		const dispatchSize = backend.get( node ).dispatchSize;

		if ( Array.isArray( dispatchSize ) !== true ) return null;

		resolved.push( { node, dispatchSize } );

	}

	return resolved;

}

function encode( renderer, device, prepared ) {

	const backend = renderer.backend;
	const nodes = renderer._nodes;
	const bindings = renderer._bindings;
	const pipelines = renderer._pipelines;

	// three.js keys "have I already updated this node for the current
	// frame?" off a rising counter, so a batch that never touched the
	// renderer would look to it like the same frame forever. Advancing it
	// keeps the update semantics a caller gets from renderer.compute().
	renderer.info.calls ++;
	const nodeFrame = nodes.nodeFrame;
	const previousRenderId = nodeFrame.renderId;
	nodeFrame.renderId = renderer.info.calls;

	const encoder = device.createCommandEncoder();
	const pass = encoder.beginComputePass();

	let currentPipeline = null;

	for ( const entry of prepared ) {

		const node = entry.node;

		// The uniform path, kept -- see this file's header.
		nodes.updateForCompute( node );
		bindings.updateForCompute( node );

		// Re-asked rather than cached: a resized or recreated resource
		// gives three.js a new bind group, and a rebuilt shader a new
		// pipeline. Both lookups are cheap; a stale GPU object is not.
		const nodeBindings = bindings.getForCompute( node );
		const pipeline = pipelines.getForCompute( node, nodeBindings );
		const pipelineGPU = backend.get( pipeline ).pipeline;

		if ( pipelineGPU === undefined ) {

			// Not resolved after all. Abandon the pass and let the renderer
			// run the rest; the next call re-resolves from scratch.
			pass.end();
			nodeFrame.renderId = previousRenderId;
			return false;

		}

		if ( pipelineGPU !== currentPipeline ) {

			pass.setPipeline( pipelineGPU );
			currentPipeline = pipelineGPU;

		}

		for ( let i = 0; i < nodeBindings.length; i ++ ) {

			pass.setBindGroup( i, backend.get( nodeBindings[ i ] ).group );

		}

		const size = entry.dispatchSize;
		pass.dispatchWorkgroups( size[ 0 ], size[ 1 ] || 1, size[ 2 ] || 1 );

	}

	pass.end();
	device.queue.submit( [ encoder.finish() ] );

	nodeFrame.renderId = previousRenderId;

	return true;

}
