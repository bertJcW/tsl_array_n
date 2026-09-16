import { Fn, instanceIndex } from 'three/tsl';
import { getRenderer } from './context.js';
import { normalizeShape } from './array.js';
import { createPreparedDispatcher } from './prepared_dispatch.js';

function unflattenNodeIndex( flatIndexNode, dims ) {

	const indices = [];
	let remaining = flatIndexNode;

	for ( let d = 0; d < dims.length; d ++ ) {

		if ( d === dims.length - 1 ) {

			indices.push( remaining );

		} else {

			indices.push( remaining.mod( dims[ d ] ) );
			remaining = remaining.div( dims[ d ] );

		}

	}

	return indices;

}

// options.workgroupSize: the compute shader's workgroup dimensions, as
// [x] / [x, y] / [x, y, z]. Left alone (three.js picks 64) this is an
// implementation detail nothing should care about -- with one exception,
// which is why it is exposed: a kernel that uses storageBarrier() or
// workgroupBarrier() to synchronise between its own invocations only
// synchronises *within a workgroup*, so such a kernel is only correct if
// every invocation it needs to coordinate is in the same one. Setting the
// workgroup size equal to the dispatch count is how that is guaranteed.
//
// WebGPU caps a workgroup at maxComputeInvocationsPerWorkgroup, commonly
// 256, so this only helps for small dispatches -- which is exactly the case
// where a barrier is worth having.
export function kernel( shape, fn, options = {} ) {

	const dims = normalizeShape( shape );
	const count = dims.reduce( ( total, dim ) => total * dim, 1 );
	const { workgroupSize } = options;

	if ( fn.length !== dims.length ) {

		throw new Error(
			`tsl_array_n: kernel() callback expected ${ dims.length } index parameter(s) for shape [${ dims.join( ', ' ) }], got ${ fn.length }.`
		);

	}

	const computeNode = Fn( () => {

		fn( ...unflattenNodeIndex( instanceIndex, dims ) );

	} )().compute( count, workgroupSize );

	const dispatch = () => getRenderer().compute( computeNode );

	// The underlying compute node, exposed so several dispatches can be
	// submitted together -- see dispatchBatch() for why that matters.
	dispatch.computeNode = computeNode;

	return dispatch;

}

// *** Why batching exists: every compute() call is its own queue submit. ***
//
// three.js's WebGPU backend ends each `renderer.compute()` with
// `passEncoderGPU.end()` followed by a `submit()` of that call's own command
// buffer. So a dispatcher returned by kernel() above costs one command
// encoder, one compute pass and one queue submission, every time it is
// called -- where native WebGPU code would record many dispatches into one
// encoder and submit once.
//
// `renderer.compute()` already accepts an array and wraps the whole array in
// a single begin/finish pair, which is exactly the native shape. Measured on
// real WebGPU hardware, 64 trivial dispatches over a 4096-element buffer,
// warmed up, ten repetitions:
//
//     one compute() per node      62.2 us per dispatch
//     one compute() for the array  6.7 us per dispatch     9.3x
//
// This matters at solver scale rather than at toy scale: fluxflow's pressure
// solve was measured at 1075 dispatches per frame, spending 38% of the frame
// inside the dispatch calls themselves. See that package's
// docs/perf-investigation-cg-gpu-resident-alpha-beta.md.
//
// Ordering is preserved, and that was checked rather than assumed, since
// iterative solvers depend on it: 40 dispatches each read-modify-writing the
// same cell, batched into one pass, produce exactly 40 -- identical to
// running them as 40 separate passes. WebGPU orders dispatches within a pass
// and handles the hazard between them.

// Resolves a list of dispatchers into the segments a batched submission
// needs: runs of kernel() dispatchers, which become one compute() call each,
// separated by anything else, which is called in place so that overall
// ordering is exactly what calling the list one by one would have given.
function planBatch( dispatchers ) {

	const plan = [];
	let nodes = null;

	for ( const dispatcher of dispatchers ) {

		if ( dispatcher === undefined || dispatcher === null ) continue;

		const node = dispatcher.computeNode;

		if ( node === undefined ) {

			if ( nodes !== null ) { plan.push( nodes ); nodes = null; }
			plan.push( dispatcher );
			continue;

		}

		if ( nodes === null ) nodes = [];
		nodes.push( node );

	}

	if ( nodes !== null ) plan.push( nodes );

	return plan;

}

/**
 * Builds a dispatcher that submits a fixed sequence of dispatchers in one
 * command buffer.
 *
 * Prefer this over dispatchBatch() for anything called every frame. The
 * sequence is resolved once, here, so the returned function does no work
 * beyond the compute() call itself -- dispatchBatch() has to walk the list
 * and collect nodes again on every call.
 */
export function createBatch( dispatchers ) {

	const plan = planBatch( dispatchers ).map( ( segment ) => {

		if ( Array.isArray( segment ) !== true ) return segment;

		// A run of kernels is one compute() and one submit -- and, once
		// three.js has resolved it, one directly encoded pass instead. See
		// prepared_dispatch.js for what that skips and what it keeps.
		const run = createPreparedDispatcher( segment, () => getRenderer().compute( segment ) );

		return { run };

	} );

	return function dispatchPlannedBatch() {

		const renderer = getRenderer();

		for ( const segment of plan ) {

			if ( typeof segment === 'function' ) segment();
			else segment.run( renderer );

		}

	};

}

/**
 * One-shot form: submits `dispatchers` in one command buffer, resolving the
 * list on every call.
 *
 * Use it for a sequence that genuinely differs each time. For a fixed
 * sequence -- a solver's inner loop, say -- use createBatch() instead and
 * pay the resolution once.
 */
export function dispatchBatch( dispatchers ) {

	createBatch( dispatchers )();

}
