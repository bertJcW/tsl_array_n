// Neural-network inference layers as TSL compute kernels, on
// tsl_array_n arrays.
//
// *** Why these are hand-written kernels and not a runtime ***
//
// The obvious alternative is ONNX Runtime Web's WebGPU backend, which is
// mature and would save writing any of this. It is the right choice for a
// *post-process* -- a super-resolution pass on the dye field, say -- and
// the wrong one for anything inside the solver loop, for a reason this
// package has already measured to death.
//
// docs/perf-investigation-cg-gpu-resident-alpha-beta.md's whole subject is
// that a solver step here is dominated by dispatch count, submission count
// and host round trips, not by arithmetic: ~1416 dispatches per step on a
// 64x64 grid, one submission costing 38.79 us, one host round trip
// costing 1.1 ms. A second inference runtime means a second device and
// buffer world. Even with ORT's `Tensor.fromGpuBuffer()` and IO binding --
// which do exist, and do avoid the CPU copy -- it is a separate
// scheduler's submissions interleaved with three.js's, with no way to put
// a convolution and a Laplacian into the same command buffer.
//
// These kernels are ordinary tsl_array_n kernels, so:
//
//   * they read and write the same arrays the solver already owns, with no
//     copy and no handoff;
//   * their dispatchers go into `tsl_array_n.createBatch()` exactly like
//     the V-cycle's do, so a whole forward pass is **one submission**
//     (batching the V-cycle was worth 1.53x-1.74x, and that is the same
//     mechanism);
//   * `profiling.js` counts them by label with no extra work, so a network
//     is measurable in the same units as everything else here.
//
// *** Determinism, which is not a small thing in this codebase ***
//
// There are no atomics anywhere in this file. Every kernel writes each
// output element from exactly one thread, reading only inputs no thread
// writes. So a forward pass is bit-identical run to run.
//
// That is worth stating because the solver it might sit next to is not:
// linalg.js's dot products are lane-partitioned float32 sums whose
// reduction order varies, and its own comment records the consequence --
// "no atomics means no run-to-run variation from GPU scheduling order,
// which also removes one source of the 'same input, different outcome'
// noise this port's stability testing kept hitting." A network in the loop
// does not add to that noise. It is one of the few things about this
// direction that is unambiguously good news.
//
// Layouts and padding are documented once, in reference.js. Every kernel
// here mirrors the corresponding function there, and the pair is checked
// against each other in examples/31-null-net-probe/.

import * as tsl_array_n from 'tsl_array_n';
import { float, int, Loop, If } from 'three/tsl';
import { buildElementwiseKernel } from '../linalg/linalg.js';

/**
 * Allocates a `[width, height, channels]` feature map.
 */
export function createFeatureMap( width, height, channels ) {

	const map = tsl_array_n.arrayN( 'float', [ width, height, channels ] );

	map.width = width;
	map.height = height;
	map.channels = channels;

	return map;

}

// Offsets a node index by a build-time constant and clamps it into
// [0, count-1]. Only the side that can actually go out of range is tested,
// which is known at build time -- the same "do the combinatorics in JS,
// emit a flat expression" approach multigrid.js takes with its stencil
// taps.
function clampedAxis( index, delta, count ) {

	if ( delta === 0 ) return index;

	const raw = index.add( delta );

	if ( delta < 0 ) return raw.lessThan( 0 ).select( int( 0 ), raw );

	return raw.greaterThan( count - 1 ).select( int( count - 1 ), raw );

}

// The in-bounds test for zero padding, or null when the tap can never
// leave the domain.
function boundsCondition( x, y, dx, dy, width, height ) {

	let condition = null;

	const add = ( next ) => {

		condition = condition === null ? next : condition.and( next );

	};

	if ( dx < 0 ) add( x.add( dx ).greaterThanEqual( 0 ) );
	if ( dx > 0 ) add( x.add( dx ).lessThan( width ) );
	if ( dy < 0 ) add( y.add( dy ).greaterThanEqual( 0 ) );
	if ( dy > 0 ) add( y.add( dy ).lessThan( height ) );

	return condition;

}

function activate( value, activation, leakySlope ) {

	if ( activation === 'none' ) return value;
	if ( activation === 'relu' ) return value.max( float( 0 ) );
	if ( activation === 'leakyRelu' ) return value.greaterThan( 0 ).select( value, value.mul( leakySlope ) );

	throw new Error( `ml: unknown activation '${ activation }'.` );

}

/**
 * Builds a 2D convolution as a single dispatch over
 * `[width, height, outChannels]`.
 *
 * One dispatch, one thread per output element. The `kernelSize x
 * kernelSize` taps are unrolled in JavaScript at build time (they need
 * per-tap index clamping, which is build-time knowledge); the input-channel
 * sum is a runtime `Loop`, so shader size stays proportional to the tap
 * count rather than to `inChannels * kernelSize^2`. At 3x3 and 16 channels
 * unrolling everything would emit 144 multiply-adds per output, and this
 * package has an established reason to keep generated kernels small: a
 * kernel binding too many buffers fails at pipeline creation as an
 * *uncaptured* error, silently never executing (tsl_array_n's README
 * records two shipped examples that were doing exactly that).
 *
 * `weights` is `[inChannels, kernelSize, kernelSize, outChannels]`.
 * `bias` is `[outChannels]`, or null for no bias.
 */
export function buildConv2dKernel( input, output, weights, bias, options ) {

	const {
		width, height, inChannels, outChannels,
		kernelSize = 3, padding = 'clamp',
		activation = 'none', leakySlope = 0.01,
		label = 'ml-conv2d'
	} = options;

	if ( kernelSize % 2 !== 1 ) throw new Error( `ml: kernelSize must be odd, got ${ kernelSize }.` );
	if ( padding !== 'clamp' && padding !== 'zero' ) throw new Error( `ml: unknown padding '${ padding }'.` );

	const radius = ( kernelSize - 1 ) / 2;

	return buildElementwiseKernel( [ width, height, outChannels ], ( [ x, y, co ] ) => {

		const acc = ( bias ? bias( co ) : float( 0 ) ).toVar();

		for ( let ky = 0; ky < kernelSize; ky ++ ) {

			for ( let kx = 0; kx < kernelSize; kx ++ ) {

				const dx = kx - radius;
				const dy = ky - radius;

				const accumulate = () => {

					const sx = padding === 'clamp' ? clampedAxis( x, dx, width ) : x.add( dx );
					const sy = padding === 'clamp' ? clampedAxis( y, dy, height ) : y.add( dy );

					Loop( inChannels, ( { i: ci } ) => {

						acc.addAssign( input( sx, sy, ci ).mul( weights( ci, int( kx ), int( ky ), co ) ) );

					} );

				};

				if ( padding === 'zero' ) {

					const inBounds = boundsCondition( x, y, dx, dy, width, height );

					// A centre tap is in bounds by construction; emitting an
					// always-true If around it would cost a branch for nothing.
					if ( inBounds === null ) accumulate();
					else If( inBounds, accumulate );

				} else {

					accumulate();

				}

			}

		}

		output( x, y, co ).assign( activate( acc, activation, leakySlope ) );

	}, label );

}

// The 4 (index, weight) taps of the full-weighting restriction filter along
// one axis, boundary-clamped. Deliberately identical to multigrid.js's
// `restrictionTapsForAxis` -- see buildRestrict2Kernel.
function restrictionTapsForAxis( coarseIndex, coarseCount ) {

	const lower = coarseIndex.greaterThan( 0 )
		.select( coarseIndex.mul( 2 ).sub( 1 ), coarseIndex.mul( 2 ) );
	const upper = coarseIndex.add( 1 ).lessThan( coarseCount )
		.select( coarseIndex.mul( 2 ).add( 2 ), coarseIndex.mul( 2 ).add( 1 ) );

	return [
		{ index: lower, weight: 0.125 },
		{ index: coarseIndex.mul( 2 ), weight: 0.375 },
		{ index: coarseIndex.mul( 2 ).add( 1 ), weight: 0.375 },
		{ index: upper, weight: 0.125 }
	];

}

/**
 * Builds a 2x restriction (full weighting) as a single dispatch over the
 * coarse shape. `[w, h, c]` -> `[w/2, h/2, c]`, channels untouched.
 *
 * Not average pooling, and reference.js's `restrict2Reference` carries the
 * full argument for why: paired with buildUpsample2Kernel below this is the
 * exact adjoint (`R = P^T / 4` in 2D, as the perf investigation measured on
 * multigrid's own pair), and average pooling is not. A symmetric transfer
 * pair is what keeps this shape usable as a CG preconditioner rather than
 * only as a filter, and it costs one dispatch either way.
 */
export function buildRestrict2Kernel( fine, coarse, options ) {

	const { width, height, channels, label = 'ml-restrict' } = options;

	if ( width % 2 !== 0 || height % 2 !== 0 ) {

		throw new Error( `ml: restrict2 needs even dimensions, got ${ width }x${ height }.` );

	}

	const coarseWidth = width / 2;
	const coarseHeight = height / 2;

	return buildElementwiseKernel( [ coarseWidth, coarseHeight, channels ], ( [ x, y, c ] ) => {

		let sum = null;

		for ( const tx of restrictionTapsForAxis( x, coarseWidth ) ) {

			for ( const ty of restrictionTapsForAxis( y, coarseHeight ) ) {

				const term = fine( tx.index, ty.index, c ).mul( tx.weight * ty.weight );
				sum = sum === null ? term : sum.add( term );

			}

		}

		coarse( x, y, c ).assign( sum );

	}, label );

}

// The 2 (index, weight) taps of the bilinear 1/4-3/4 filter along one axis,
// by fine-index parity, boundary-clamped.
//
// Deliberately identical to multigrid.js's `correctionTapsForAxis`, and
// transposed to the restriction in buildRestrict2Kernel above. The down/up
// pair is therefore adjoint, which is the property that lets this shape be
// used as a *preconditioner* rather than only as a filter: CG requires a
// symmetric preconditioner, and multigrid.js's relax() comment records what
// this codebase's liquid does when that is violated.
function upsampleTapsForAxis( fineIndex, coarseCount ) {

	const ci = fineIndex.div( 2 );
	const isEven = fineIndex.mod( 2 ).equal( 0 );

	const lowerIndex = ci.greaterThan( 0 ).select( ci.sub( 1 ), ci );
	const upperIndex = ci.lessThan( coarseCount - 1 ).select( ci.add( 1 ), ci );

	return [
		{ index: isEven.select( lowerIndex, ci ), weight: isEven.select( float( 0.25 ), float( 0.75 ) ) },
		{ index: isEven.select( ci, upperIndex ), weight: isEven.select( float( 0.75 ), float( 0.25 ) ) }
	];

}

/**
 * Builds a 2x bilinear upsample as a single dispatch over the fine shape.
 *
 * `additive: true` adds into `fine` instead of overwriting it, which is how
 * the skip connection is made: the decoder upsamples the coarse level
 * *into* the buffer still holding the encoder's output at this level, so no
 * separate concatenate, projection or add kernel is needed. One dispatch
 * does the upsample and the skip together.
 *
 * The cost of that trick is that the forward pass overwrites its own
 * encoder activations. That is fine for inference -- the encoder always
 * runs immediately before the decoder -- and it is stated here because it
 * would not be fine for backpropagation, which needs those activations kept.
 */
export function buildUpsample2Kernel( coarse, fine, options ) {

	const { coarseWidth, coarseHeight, channels, additive = false, label = 'ml-upsample' } = options;

	const fineWidth = coarseWidth * 2;
	const fineHeight = coarseHeight * 2;

	return buildElementwiseKernel( [ fineWidth, fineHeight, channels ], ( [ x, y, c ] ) => {

		let sum = null;

		for ( const tx of upsampleTapsForAxis( x, coarseWidth ) ) {

			for ( const ty of upsampleTapsForAxis( y, coarseHeight ) ) {

				const term = coarse( tx.index, ty.index, c ).mul( tx.weight ).mul( ty.weight );
				sum = sum === null ? term : sum.add( term );

			}

		}

		if ( additive ) fine( x, y, c ).addAssign( sum );
		else fine( x, y, c ).assign( sum );

	}, label );

}

/**
 * Builds a kernel copying one channel of a `[w, h, 1]`-shaped (or wider)
 * feature map out into a plain `[w, h]` field, and the reverse.
 *
 * These are the joins to the rest of the library: the solver's fields are
 * 2D `array2`s, a network's are 3D feature maps, and something has to
 * bridge them. One dispatch each, and they are the only place in this
 * module that knows anything about fluxflow's own data.
 */
export function buildPackFieldKernel( field, featureMap, options ) {

	const { width, height, channel = 0, scale = 1, label = 'ml-pack' } = options;

	return buildElementwiseKernel( [ width, height ], ( [ x, y ] ) => {

		featureMap( x, y, int( channel ) ).assign( field( x, y ).mul( scale ) );

	}, label );

}

export function buildUnpackFieldKernel( featureMap, field, options ) {

	const { width, height, channel = 0, scale = 1, label = 'ml-unpack' } = options;

	return buildElementwiseKernel( [ width, height ], ( [ x, y ] ) => {

		field( x, y ).assign( featureMap( x, y, int( channel ) ).mul( scale ) );

	}, label );

}
