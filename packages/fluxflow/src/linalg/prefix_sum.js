// An inclusive prefix sum over a 1-D int array, on the GPU.
//
// *** Why this exists: an atomic counter is unique but not reproducible ***
//
// The FLIP resamplers used to hand out slots with `atomicAdd( cursor, 1 )`
// and take the returned pre-increment value -- the classic GPU
// claim-a-unique-slot pattern, and it was verified on real hardware to be
// exactly that: every slot from 0..N-1 written exactly once, no collisions,
// at N = 256 and N = 4000.
//
// Uniqueness was the right property to check for the data structure and the
// wrong one for the simulation. WHICH thread gets slot 0 depends on which
// thread reaches the atomic first, and that order is not fixed between runs.
// The resampler then teleports the particle in each slot to a different
// cell, so two runs of the same build move different particles and part
// company. Measured on examples/33-flip-dam-break-3d/: four 12,000-step runs
// gave two different verdicts and fields that already differ by frame 300.
//
// A rank defined by INDEX rather than by arrival is reproducible, and a
// prefix sum over a 0/1 flag is how you compute one: the rank of element i
// among the flagged elements is the number of flagged elements before it.
// Nothing about the result depends on execution order, so the same input
// gives the same output on every run, on every device.
//
// *** The algorithm, and why this one ***
//
// Hillis-Steele: log2(n) passes, each adding the element `offset` back.
//
//   for offset = 1, 2, 4, ... < n:  next[i] = cur[i] + cur[i - offset]
//
// It does O(n log n) work where a Blelloch scan does O(n), which for the
// sizes here -- tens of thousands of elements, once per frame -- is the
// wrong thing to optimise: what costs on this backend is submissions and
// host round trips, not arithmetic. This form needs neither. The pass count
// is fixed by the length, so every kernel is built once at construction and
// the whole scan goes into a single batch with the kernels around it; there
// is no host readback anywhere in it, and no scalar has to come back to the
// CPU to decide anything.
//
// Two scratch buffers ping-pong. Which one holds the answer depends only on
// whether the pass count is odd or even, which is known at construction, so
// the caller is handed the right array rather than a runtime choice.

import * as tsl_array_n from 'tsl_array_n';
import { int, max } from 'three/tsl';

/**
 * Builds an inclusive prefix sum of `source` over `[ length ]`.
 *
 * `source` is read and never written. The returned `kernels` must be
 * dispatched in order -- put them in a batch with whatever produces
 * `source` and whatever consumes the result, which is what the FLIP
 * resamplers do.
 *
 * @param {number} length number of elements
 * @param {object} source an int arrayN of shape [ length ]
 * @returns {{ kernels: Function[], result: object }} result( i ) is the sum
 *   of source( 0 .. i ) inclusive; result( length - 1 ) is the total.
 */
export function createInclusivePrefixSum( length, source ) {

	if ( ! Number.isInteger( length ) || length < 1 ) {

		throw new Error( `createInclusivePrefixSum: length must be a positive integer, got ${ length }` );

	}

	const shape = [ length ];
	const bufferA = tsl_array_n.arrayN( 'int', shape );
	const bufferB = tsl_array_n.arrayN( 'int', shape );

	// A length of 1 is already its own prefix sum, but the caller still needs
	// an array it can read, so copy rather than special-case at every call
	// site.
	const kernels = [];
	let read = source;
	let write = bufferA;

	for ( let offset = 1; offset < length; offset *= 2 ) {

		const src = read;
		const dst = write;
		const step = int( offset );

		kernels.push( tsl_array_n.kernel( shape, ( i ) => {

			// Both sides of a select are evaluated in the generated shader,
			// so the index has to be in range even on the branch that is
			// thrown away -- hence max(0, i - offset) rather than i - offset.
			const carried = i.greaterThanEqual( step ).select( src( max( int( 0 ), i.sub( step ) ) ), int( 0 ) );
			dst( i ).assign( src( i ).add( carried ) );

		} ) );

		read = dst;
		write = ( dst === bufferA ) ? bufferB : bufferA;

	}

	if ( kernels.length === 0 ) {

		kernels.push( tsl_array_n.kernel( shape, ( i ) => {

			bufferA( i ).assign( source( i ) );

		} ) );

		read = bufferA;

	}

	return { kernels, result: read };

}
