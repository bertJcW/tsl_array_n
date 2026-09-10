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
// own header comment, decision 1): the value has to reach the atomic as an
// integer. Unlike a *sum*, though, a maximum needs no arithmetic on the
// encoded values -- only their ordering -- and IEEE-754 hands that over for
// free: for non-negative floats the bit pattern read as an integer is
// monotonically increasing in the value, and the sign bit is 0 so it never
// looks negative to a signed comparison. So this encodes the bit pattern
// itself and decodes it back on the host, exactly, with no scale anywhere.
//
// This used to be a fixed-point `round(value * scale)` encoding sharing
// linalg.js's own DEFAULT_ATOMIC_DOT_SCALE. That was survivable here in a
// way it was not for the dot product (a max never accumulates, so it cannot
// overflow the way a sum does -- see createDotReducer's own comment for
// what that cost), but it still quantized the answer, still capped the
// representable magnitude at 2^31/scale, and still left a knob for a caller
// to get wrong. A bit-pattern max has none of those properties and is
// simpler. `atomicScale` is accepted and ignored so existing callers keep
// working.

import * as tsl_array_n from 'tsl_array_n';
import { atomicMax, abs, floatBitsToUint } from 'three/tsl';
import { buildElementwiseKernel } from './linalg.js';

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

	const accum = tsl_array_n.array0( 'int' );
	accum.node.toAtomic();

	const dispatchers = list.map( ( field ) => buildElementwiseKernel( field.shape, ( I ) => {

		// Bit pattern, not a scaled integer -- see this file's own header
		// comment. abs() first, so the sign bit is always 0 and the ordering
		// is the float ordering.
		atomicMax( accum(), floatBitsToUint( abs( field( ...I ) ) ).toInt() );

	} ) );

	// Resets the accumulator, dispatches every field's own reduction kernel
	// in sequence (no reset between them -- see this function's own header
	// comment), then reads back the single accumulated int and decodes it.
	// One readback per call, regardless of how many fields were given.
	async function read() {

		// 0 as a bit pattern is +0.0, which is the correct identity for a
		// maximum over magnitudes.
		accum.fromArray( new Int32Array( [ 0 ] ) );
		for ( const dispatch of dispatchers ) dispatch();

		const [ bits ] = await accum.toArray();
		return new Float32Array( new Int32Array( [ bits ] ).buffer )[ 0 ];

	}

	return { read };

}
