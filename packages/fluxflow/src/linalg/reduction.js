// GPU max-magnitude reduction, generalizing linalg.js's own atomic-dot-
// product machinery (buildAtomicDotKernel/resetAndDispatch/
// DEFAULT_ATOMIC_DOT_SCALE's fixed-point-int encoding) from atomicAdd to
// atomicMax. Kept as its own file rather than folded into linalg.js
// because it's meant for reuse outside CG entirely (velocity-magnitude
// reduction for CFL/adaptive-timestep today, potentially other future
// per-frame diagnostics) -- same reasoning multigrid.js already
// established for keeping a related-but-distinct numerical primitive out
// of linalg.js itself.
//
// Same WGSL-atomics-are-int-only constraint as linalg.js (see that file's
// own header comment, decision 1): magnitude is encoded as a fixed-point
// int via the same round(value * scale) trick. Unlike the dot-product
// case, magnitude is always >=0, so no sign handling is needed -- a plain
// atomicMax over the encoded values is exactly the reduction wanted, with
// no risk of a negative encoding confusing a signed-int comparison the
// way it could for atomicAdd's own accumulator.

import * as tsl_array_n from 'tsl_array_n';
import { atomicMax, round, abs } from 'three/tsl';
import { buildElementwiseKernel, DEFAULT_ATOMIC_DOT_SCALE } from './linalg.js';

// fields: one arrayN/array2 field, or an array of them. Each gets its own
// dispatch (built once, at construction time, same "kernels bind to a
// concrete field" convention as every other factory in this port) --
// necessary rather than a single shared kernel because sibling fields can
// have genuinely different shapes (e.g. a FaceCenteredGrid2's dataU/dataV,
// a staggered MAC grid's two components -- (resX+1)*resY and
// resX*(resY+1), never equal). All dispatches feed the *same* accumulator
// with no reset in between, since the reduction wanted is the max across
// every field together, not a separate max per field.
export function createMaxAbsReducer( fields, options = {} ) {

	const list = Array.isArray( fields ) ? fields : [ fields ];
	const scale = options.atomicScale ?? DEFAULT_ATOMIC_DOT_SCALE;

	const accum = tsl_array_n.array0( 'int' );
	accum.node.toAtomic();

	const dispatchers = list.map( ( field ) => buildElementwiseKernel( field.shape, ( I ) => {

		const encoded = round( abs( field( ...I ) ).mul( scale ) ).toInt();
		atomicMax( accum(), encoded );

	} ) );

	// Resets the accumulator, dispatches every field's own reduction kernel
	// in sequence (no reset between them -- see this function's own header
	// comment), then reads back the single accumulated int and decodes it.
	// One readback per call, regardless of how many fields were given.
	async function read() {

		accum.fromArray( new Int32Array( [ 0 ] ) );
		for ( const dispatch of dispatchers ) dispatch();

		const [ scaledMax ] = await accum.toArray();
		return scaledMax / scale;

	}

	return { read };

}
