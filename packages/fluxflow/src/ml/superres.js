// Super-resolution for the *display* path: simulate coarse, render fine.
//
// This is family C of docs/machine-learning-fluid-research.md -- detail
// synthesis as a post-process on the output field. It is the one direction
// in that document with **no solver risk at all**: it reads a field the
// solver has already finished with, preserves no invariant, and touches
// neither pressure nor divergence nor volume. The worst a bug here can do
// is look wrong.
//
// It is also the one direction whose budget is independent of the null-net
// probe. Family A has to beat the pressure solve to be worth anything; this
// runs once per *rendered frame* rather than per solver step, and it is
// allowed to cost real time because it buys something the user can see.
//
// *** What it is replacing ***
//
// Nothing, currently -- and that is the point. examples/17-smoke-fire/
// simulates at 96x128, draws to a 96x128 canvas, and lets CSS scale it to
// 384x512 with `image-rendering: pixelated`. So the shipped "renderer" for
// a 4x upscale is nearest-neighbour. Bilinear and monotonic-bicubic are the
// cheap non-ML improvements on that, and both already exist in this package
// (`grid_math.js`); the network is the expensive one, and it has to beat
// bicubic, not nearest, to have earned anything.
//
// *** The architecture, and the one idea that makes it affordable ***
//
// Every convolution runs at **low** resolution. The last one emits
// `factor^2` channels, and a single fused kernel rearranges those into the
// high-resolution image (sub-pixel convolution, Shi et al., CVPR 2016 --
// "ESPCN"). The alternative, upsampling first and convolving at high
// resolution, costs `factor^2` times the arithmetic: at 4x that is 16x, for
// the same number of parameters. On a package whose whole performance story
// is that work should not be done where it is not needed, doing the
// convolutions at 96x128 instead of 384x512 is not an optimisation, it is
// the only sane arrangement.
//
// *** The property worth designing in: it degrades to bicubic ***
//
// The network predicts a **residual** on top of a classical upsample, and
// the two are added in the same kernel that does the rearrange. So:
//
//     output = bicubic(input) + network(input)
//
// With zero weights, the output is *exactly* bicubic. Not approximately,
// not "close enough" -- the residual path contributes a hard zero. That has
// three consequences worth having:
//
//   1. **It is useful before it is trained.** Ship it untrained and it is a
//      bicubic upsampler with some wasted dispatches, not a noise generator.
//   2. **Training can only ever add detail**, which is what the
//      super-resolution literature settled on for the same reason (VDSR and
//      after): the low frequencies are already right, so spending network
//      capacity on reproducing them is waste.
//   3. **A broken or half-trained network is recognisable**, because the
//      failure mode is "bicubic plus garbage" rather than "garbage", and
//      the two look completely different.
//
// *** What this cannot do yet, and it matters ***
//
// **There is no temporal term, so a trained network of this shape will
// flicker.** That is not speculation: it is tempoGAN's (Xie et al.,
// SIGGRAPH 2018) central finding, and the stated motivation for the
// diffusion work that followed -- a network trained on isolated frames
// synthesises detail that is plausible per frame and uncorrelated between
// them, which reads as boiling.
//
// The fix is available and this package is unusually well placed for it: a
// simulation *has* the velocity field, so the previous high-resolution
// output can be advected forward and fed in as an extra input channel,
// which is what temporal upsampling in real-time rendering does. The
// machinery exists (`advection_solver2.js`'s semi-Lagrangian backtrace).
// It is deliberately not built here, because an untrained temporal input
// channel costs dispatches and buys nothing -- it only starts to matter
// once there is a training loop to teach the network to use it. The
// `inChannels` parameter is the seam: extra channels are already carried
// through to the first convolution.

import * as tsl_array_n from 'tsl_array_n';
import { int, vec2 } from 'three/tsl';
import { collocatedValueAtPosition2, collocatedCubicValueAtPosition2 } from '../grid/grid_math.js';
import { buildElementwiseKernel } from '../linalg/linalg.js';
import { profileBatch } from '../profiling.js';
import { createFeatureMap, buildConv2dKernel } from './layers.js';
import { createSeededRandom, heNormalWeights } from './reference.js';

/**
 * Builds the fused sub-pixel rearrange + classical base + residual add, as
 * a single dispatch over the high-resolution shape.
 *
 * `shuffleSource` is `[lowWidth, lowHeight, factor^2]`, `base` is the
 * low-resolution input feature map (channel 0 is upsampled classically),
 * and `output` is `[lowWidth * factor, lowHeight * factor, 1]`.
 *
 * Three operations in one dispatch rather than three. The rearrange alone
 * would be a dispatch, the classical upsample another, and the add a third;
 * fusing them costs nothing because every one of them is a pure per-output
 * -pixel function of inputs no thread writes.
 */
export function buildShuffleResidualKernel( shuffleSource, base, output, options ) {

	const {
		lowWidth, lowHeight, factor,
		baseKind = 'bicubic',
		label = 'ml-shuffle-residual'
	} = options;

	if ( ! Number.isInteger( factor ) || factor < 1 ) {

		throw new Error( `ml: factor must be a positive integer, got ${ factor }.` );

	}

	if ( baseKind !== 'bicubic' && baseKind !== 'bilinear' && baseKind !== 'none' ) {

		throw new Error( `ml: unknown base '${ baseKind }'.` );

	}

	const highWidth = lowWidth * factor;
	const highHeight = lowHeight * factor;

	// grid_math's samplers take a 2D accessor; channel 0 of the input is the
	// field being upsampled. Reusing them rather than reimplementing is the
	// point -- the monotonicity clamp in `monotonicCubic1d` is subtle, it is
	// already tested here, and a second copy would be a second thing to keep
	// right.
	const baseChannel0 = ( i, j ) => base( i, j, int( 0 ) );
	const gridSpacing = vec2( 1, 1 );
	const dataOrigin = vec2( 0, 0 );

	return buildElementwiseKernel( [ highWidth, highHeight ], ( [ X, Y ] ) => {

		const lowX = X.div( factor );
		const lowY = Y.div( factor );

		// Which of the factor^2 sub-pixels this output is. x-fastest, so it
		// matches the feature-map layout everywhere else in this module.
		const sub = X.sub( lowX.mul( factor ) ).add( Y.sub( lowY.mul( factor ) ).mul( factor ) );

		const residual = shuffleSource( lowX, lowY, sub );

		if ( baseKind === 'none' ) {

			output( X, Y, int( 0 ) ).assign( residual );
			return;

		}

		// The half-cell shift matters: low-res cell centres are at i + 0.5 in
		// low-res units, high-res centres at (X + 0.5) / factor. Dropping it
		// shifts the whole image by half a low-res cell -- two high-res
		// pixels at 4x, which is visible and reads as the network having
		// learned an offset rather than as a sampling bug.
		const pos = vec2(
			X.toFloat().add( 0.5 ).div( factor ).sub( 0.5 ),
			Y.toFloat().add( 0.5 ).div( factor ).sub( 0.5 )
		);

		const sampled = baseKind === 'bicubic'
			? collocatedCubicValueAtPosition2( baseChannel0, gridSpacing, dataOrigin, pos, [ lowWidth, lowHeight ] )
			: collocatedValueAtPosition2( baseChannel0, gridSpacing, dataOrigin, pos, [ lowWidth, lowHeight ] );

		output( X, Y, int( 0 ) ).assign( sampled.add( residual ) );

	}, label );

}

/**
 * Builds a classical-only upsampler: the same kernel with no residual path.
 *
 * This is the baseline arm. Having it as a first-class thing rather than as
 * "the network with its weights zeroed" means the comparison in the example
 * is a comparison of two *pipelines*, and the baseline costs one dispatch
 * rather than the network's several.
 */
export function createClassicalUpsampler2( options = {} ) {

	const { shape, factor = 4, baseKind = 'bicubic', input = null } = options;

	if ( ! Array.isArray( shape ) || shape.length !== 2 ) {

		throw new Error( 'ml: createClassicalUpsampler2 needs a 2D shape, e.g. [96, 128].' );

	}

	const [ lowWidth, lowHeight ] = shape;

	const source = input ?? createFeatureMap( lowWidth, lowHeight, 1 );
	const output = createFeatureMap( lowWidth * factor, lowHeight * factor, 1 );

	// A zero residual source, so the one fused kernel serves both pipelines
	// and the baseline cannot drift away from the network's base term.
	//
	// Filled explicitly rather than trusting the buffer to start at zero: a
	// storage buffer does start zeroed, but tsl_array_n only allocates one
	// lazily, on first real use by a pass (its README records this), and a
	// baseline that silently picked up whatever was there would be a very
	// quiet way to make every comparison on this page wrong.
	const zeros = createFeatureMap( lowWidth, lowHeight, factor * factor );
	zeros.fromArray( new Float32Array( lowWidth * lowHeight * factor * factor ) );

	const dispatch = buildShuffleResidualKernel( zeros, source, output, {
		lowWidth, lowHeight, factor, baseKind, label: `ml-upsample-${ baseKind }`
	} );

	return {
		input: source,
		output,
		dispatchers: [ dispatch ],
		upsample: dispatch,
		stats: { dispatches: 1, factor, baseKind, shape: [ lowWidth, lowHeight ] }
	};

}

/**
 * Builds the super-resolution network.
 *
 * @param {object} options
 * @param {number[]} options.shape        `[width, height]` of the simulation grid.
 * @param {number} [options.factor]       Integer upscale factor.
 * @param {number} [options.channels]     Feature width of the low-res trunk.
 * @param {number} [options.layers]       Convolutions in the trunk, before the sub-pixel head.
 * @param {number} [options.inChannels]   Input channels. Channel 0 is the field being upsampled;
 *                                        the rest are free (velocity, a warped previous frame).
 * @param {string} [options.baseKind]     'bicubic' | 'bilinear' | 'none'. The residual base.
 * @param {string} [options.activation]   Trunk activation.
 * @param {number} [options.seed]         Seed for the initial randomize().
 */
export function createSuperResolver2( options = {} ) {

	const {
		shape,
		factor = 4,
		channels = 16,
		layers = 3,
		inChannels = 1,
		kernelSize = 3,
		baseKind = 'bicubic',
		activation = 'relu',
		padding = 'clamp',
		leakySlope = 0.01,
		seed = 1
	} = options;

	if ( ! Array.isArray( shape ) || shape.length !== 2 ) {

		throw new Error( 'ml: createSuperResolver2 needs a 2D shape, e.g. [96, 128].' );

	}

	if ( ! Number.isInteger( factor ) || factor < 1 ) {

		throw new Error( `ml: factor must be a positive integer, got ${ factor }.` );

	}

	if ( layers < 1 ) throw new Error( `ml: layers must be at least 1, got ${ layers }.` );

	const [ lowWidth, lowHeight ] = shape;
	const subPixels = factor * factor;

	const input = createFeatureMap( lowWidth, lowHeight, inChannels );
	const trunkA = createFeatureMap( lowWidth, lowHeight, channels );
	const trunkB = createFeatureMap( lowWidth, lowHeight, channels );
	const shuffleSource = createFeatureMap( lowWidth, lowHeight, subPixels );
	const output = createFeatureMap( lowWidth * factor, lowHeight * factor, 1 );

	const blocks = [];
	const dispatchers = [];
	const plan = [];

	// Ping-pong between two trunk buffers rather than allocating one per
	// layer: a convolution reads every neighbour of its input, so writing
	// into the buffer it is reading would be a race, but two buffers are
	// enough for a straight chain with no skips.
	const bufferName = ( index ) => ( index % 2 === 0 ? 'trunkA' : 'trunkB' );
	const bufferFor = ( index ) => ( index % 2 === 0 ? trunkA : trunkB );

	function addConv( from, to, fromName, toName, fromChannels, toChannels, act, name ) {

		const weights = tsl_array_n.arrayN( 'float', [ fromChannels, kernelSize, kernelSize, toChannels ] );
		const bias = tsl_array_n.arrayN( 'float', [ toChannels ] );

		dispatchers.push( buildConv2dKernel( from, to, weights, bias, {
			width: lowWidth,
			height: lowHeight,
			inChannels: fromChannels,
			outChannels: toChannels,
			kernelSize,
			padding,
			activation: act,
			leakySlope,
			label: `ml-sr-${ name }`
		} ) );

		plan.push( {
			op: 'conv',
			from: fromName,
			to: toName,
			name,
			width: lowWidth,
			height: lowHeight,
			inChannels: fromChannels,
			outChannels: toChannels,
			activation: act
		} );

		blocks.push( {
			name,
			inChannels: fromChannels,
			outChannels: toChannels,
			kernelSize,
			weights,
			bias,
			macs: lowWidth * lowHeight * fromChannels * toChannels * kernelSize * kernelSize
		} );

	}

	// Trunk, all at low resolution.
	addConv( input, bufferFor( 0 ), 'input', bufferName( 0 ), inChannels, channels, activation, 'trunk0' );

	for ( let layer = 1; layer < layers; layer ++ ) {

		addConv(
			bufferFor( layer - 1 ), bufferFor( layer ),
			bufferName( layer - 1 ), bufferName( layer ),
			channels, channels, activation, `trunk${ layer }`
		);

	}

	// Sub-pixel head: linear, because it emits a residual that has to be
	// free to be negative. A ReLU here would make the network able to add
	// detail and unable to remove it, which on a density field means it
	// could only ever brighten.
	addConv(
		bufferFor( layers - 1 ), shuffleSource,
		bufferName( layers - 1 ), 'shuffleSource',
		channels, subPixels, 'none', 'head'
	);

	dispatchers.push( buildShuffleResidualKernel( shuffleSource, input, output, {
		lowWidth, lowHeight, factor, baseKind
	} ) );

	plan.push( {
		op: 'shuffleResidual',
		from: 'shuffleSource',
		base: 'input',
		to: 'output',
		lowWidth,
		lowHeight,
		factor,
		baseKind
	} );

	const batched = tsl_array_n.createBatch( dispatchers );

	const parameters = blocks.reduce( ( total, b ) => total + b.weights.count + b.bias.count, 0 );
	const macs = blocks.reduce( ( total, b ) => total + b.macs, 0 );

	function randomize( withSeed = seed ) {

		const random = createSeededRandom( withSeed );

		for ( const block of blocks ) {

			block.weights.fromArray( heNormalWeights( block.inChannels, block.kernelSize, block.outChannels, random ) );
			block.bias.fromArray( new Float32Array( block.outChannels ) );

		}

	}

	/**
	 * Sets every weight and bias to zero, which makes the output **exactly**
	 * the classical upsample -- see this file's header.
	 *
	 * This is the useful untrained state, and the one the example ships in.
	 * `randomize()` exists to price the shape (an all-zero network's
	 * arithmetic is the same, but zeros are not representative of trained
	 * magnitudes if the hardware does anything special with them); it is not
	 * the state to look at.
	 */
	function zero() {

		for ( const block of blocks ) {

			block.weights.fromArray( new Float32Array( block.weights.count ) );
			block.bias.fromArray( new Float32Array( block.bias.count ) );

		}

	}

	function loadWeights( byName ) {

		for ( const block of blocks ) {

			const entry = byName[ block.name ];

			if ( entry === undefined ) throw new Error( `ml: no weights supplied for block '${ block.name }'.` );

			if ( entry.weights.length !== block.weights.count ) {

				throw new Error( `ml: block '${ block.name }' expects ${ block.weights.count } weights, got ${ entry.weights.length }.` );

			}

			block.weights.fromArray( entry.weights );
			block.bias.fromArray( entry.bias ?? new Float32Array( block.outChannels ) );

		}

	}

	zero();

	return {
		input,
		output,
		blocks,
		dispatchers,
		plan,
		config: { kernelSize, padding, leakySlope, factor, baseKind, inChannels, channels, layers, lowWidth, lowHeight },

		/** One super-resolution pass, as a single submission. */
		resolve: () => profileBatch( 'ml-superres', dispatchers.length, batched ),

		resolveUnbatched: () => {

			for ( const dispatch of dispatchers ) dispatch();

		},

		randomize,
		zero,
		loadWeights,

		stats: {
			shape: [ lowWidth, lowHeight ],
			highShape: [ lowWidth * factor, lowHeight * factor ],
			factor,
			channels,
			layers,
			baseKind,
			dispatches: dispatchers.length,
			parameters,
			bytes: parameters * 4,
			macs,
			flops: macs * 2,
			// What upsampling first and convolving at high resolution would
			// have cost, for the same parameters. The reason for the
			// sub-pixel arrangement, in one number.
			macsIfUpsampledFirst: macs * factor * factor
		}
	};

}
