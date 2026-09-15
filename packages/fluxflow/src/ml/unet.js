// A small U-Net, composed from layers.js, as the concrete candidate shape
// for the directions in docs/machine-learning-fluid-research.md.
//
// *** Why this shape ***
//
// The research document's shortlist opens with a measurement, not a model:
// build the candidate network shape with random weights and price it
// against the solver it would replace, before any training run happens.
// This is that shape, and examples/31-null-net-probe/ is that measurement.
//
// Three choices are worth defending, because each of them is a departure
// from what a machine-learning practitioner would write by default:
//
// **The channel count is constant across levels.** A textbook U-Net
// doubles channels on every downsample. Doing that would mean a 1x1
// projection convolution at every level change (the skip connection adds
// fine and coarse activations together, and they have to match), which is
// four more dispatches for a network whose entire budget is measured in
// dispatches. A constant width also makes the cost model legible: every
// convolution has identical weights and identical arithmetic, so the whole
// network's cost is (dispatches) x (cells at that level), and the coarse
// levels are nearly free. This is also what the lightweight neural
// preconditioners in the literature do; it is not a corner being cut.
//
// **Skips are additive, not concatenated.** Concatenation doubles the
// channel count entering the next convolution, so it needs either a wider
// convolution or a projection. Addition is free: the upsample kernel adds
// straight into the buffer already holding the encoder's activation at that
// level, so the skip costs zero extra dispatches. See
// buildUpsample2Kernel's note about what this trades away (the encoder's
// activations are overwritten, which inference does not care about and
// training would).
//
// **Down and up are the adjoint pair from multigrid.js, not learned.**
// Learned downsampling (strided convolution) and learned upsampling
// (transposed convolution) are the usual choices and both cost parameters,
// dispatches, and -- decisively -- the guarantee that the pair is adjoint.
// If this network is ever used as a preconditioner rather than as a filter,
// CG requires it to be symmetric, and a learned non-adjoint transfer pair
// is the fastest way to lose that. Keeping the multigrid transfer operators
// costs nothing and keeps the option open.
//
// *** What this is not ***
//
// It is not trained, and nothing here trains it. `randomize()` produces a
// network that computes nonsense with realistic magnitudes, which is
// exactly what a cost measurement needs and exactly what an accuracy
// measurement cannot use. `loadWeights()` is the seam for weights trained
// offline; reference.js's `fromPyTorchConv2dWeights` is how they get into
// this library's layout.

import * as tsl_array_n from 'tsl_array_n';
import { profileBatch } from '../profiling.js';
import {
	createFeatureMap,
	buildConv2dKernel,
	buildRestrict2Kernel,
	buildUpsample2Kernel
} from './layers.js';
import { createSeededRandom, heNormalWeights } from './reference.js';

/**
 * Builds the U-Net's dispatchers, buffers and weight blocks.
 *
 * @param {object} options
 * @param {number[]} options.shape        `[width, height]` of the finest level.
 * @param {number} [options.channels]     Feature width, constant across levels.
 * @param {number} [options.levels]       Resolution levels including the finest.
 * @param {number} [options.inChannels]   Channels of the input feature map.
 * @param {number} [options.outChannels]  Channels of the output feature map.
 * @param {number} [options.kernelSize]   Odd convolution kernel size.
 * @param {string} [options.activation]   'relu' | 'leakyRelu' | 'none'.
 * @param {string} [options.padding]      'clamp' | 'zero'. See reference.js.
 * @param {number} [options.seed]         Seed for the initial randomize().
 */
export function createUNet2( options = {} ) {

	const {
		shape,
		channels = 16,
		levels = 3,
		inChannels = 1,
		outChannels = 1,
		kernelSize = 3,
		activation = 'relu',
		padding = 'clamp',
		leakySlope = 0.01,
		seed = 1
	} = options;

	if ( ! Array.isArray( shape ) || shape.length !== 2 ) {

		throw new Error( 'ml: createUNet2 needs a 2D shape, e.g. [64, 64].' );

	}

	const [ width, height ] = shape;
	const divisor = 2 ** ( levels - 1 );

	if ( width % divisor !== 0 || height % divisor !== 0 ) {

		throw new Error( `ml: shape ${ width }x${ height } is not divisible by 2^(levels-1) = ${ divisor }.` );

	}

	if ( levels < 1 ) throw new Error( `ml: levels must be at least 1, got ${ levels }.` );

	// One feature map per level. Level 0 is the finest.
	const maps = [];

	for ( let level = 0; level < levels; level ++ ) {

		const scale = 2 ** level;
		maps.push( createFeatureMap( width / scale, height / scale, channels ) );

	}

	const input = createFeatureMap( width, height, inChannels );
	const output = createFeatureMap( width, height, outChannels );

	// Every weight block the network owns, in build order, each carrying
	// enough metadata to be randomised, saved, or loaded without the caller
	// having to know the architecture.
	const blocks = [];

	function addConv( from, to, fromChannels, toChannels, level, act, name ) {

		const map = maps[ level ];
		const weightArray = tsl_array_n.arrayN( 'float', [ fromChannels, kernelSize, kernelSize, toChannels ] );
		const biasArray = tsl_array_n.arrayN( 'float', [ toChannels ] );

		const dispatch = buildConv2dKernel( from, to, weightArray, biasArray, {
			width: map.width,
			height: map.height,
			inChannels: fromChannels,
			outChannels: toChannels,
			kernelSize,
			padding,
			activation: act,
			leakySlope,
			label: `ml-conv-${ name }`
		} );

		blocks.push( {
			name,
			level,
			inChannels: fromChannels,
			outChannels: toChannels,
			kernelSize,
			weights: weightArray,
			bias: biasArray,
			// Multiply-accumulates this convolution performs per forward pass.
			// The cost model in the research document is built from these.
			macs: map.width * map.height * fromChannels * toChannels * kernelSize * kernelSize
		} );

		return dispatch;

	}

	const dispatchers = [];

	// A machine-readable description of the same forward pass, emitted
	// alongside the dispatchers rather than written out separately.
	//
	// reference.js executes this plan in float64 JavaScript, so the CPU
	// cross-check runs *the architecture this function actually built*
	// rather than a hand-transcription of it. A transcribed reference drifts
	// the first time the architecture changes and then disagrees for a
	// reason that looks like a GPU bug. Buffers are named: 'input',
	// 'output', and `level<n>` for each resolution level.
	const plan = [];

	const levelName = ( level ) => `level${ level }`;

	function recordConv( from, to, fromChannels, toChannels, level, act, name ) {

		plan.push( {
			op: 'conv',
			from, to, name,
			width: maps[ level ].width,
			height: maps[ level ].height,
			inChannels: fromChannels,
			outChannels: toChannels,
			activation: act
		} );

	}

	// Stem: lift the input into the feature width at the finest level.
	dispatchers.push( addConv( input, maps[ 0 ], inChannels, channels, 0, activation, 'stem' ) );
	recordConv( 'input', levelName( 0 ), inChannels, channels, 0, activation, 'stem' );

	// Encoder: one convolution per level, then pool into the next.
	for ( let level = 0; level < levels - 1; level ++ ) {

		dispatchers.push( addConv( maps[ level ], maps[ level ], channels, channels, level, activation, `down${ level }` ) );
		recordConv( levelName( level ), levelName( level ), channels, channels, level, activation, `down${ level }` );

		dispatchers.push( buildRestrict2Kernel( maps[ level ], maps[ level + 1 ], {
			width: maps[ level ].width,
			height: maps[ level ].height,
			channels,
			label: `ml-restrict${ level }`
		} ) );

		plan.push( {
			op: 'restrict',
			from: levelName( level ),
			to: levelName( level + 1 ),
			width: maps[ level ].width,
			height: maps[ level ].height,
			channels
		} );

	}

	// Bottom: two convolutions at the coarsest level. Cheap -- at three
	// levels the coarsest has a sixteenth of the finest level's cells --
	// and it is where the network's receptive field is widest, which for a
	// Poisson-shaped problem is the part that matters.
	const bottom = levels - 1;
	dispatchers.push( addConv( maps[ bottom ], maps[ bottom ], channels, channels, bottom, activation, 'bottomA' ) );
	recordConv( levelName( bottom ), levelName( bottom ), channels, channels, bottom, activation, 'bottomA' );
	dispatchers.push( addConv( maps[ bottom ], maps[ bottom ], channels, channels, bottom, activation, 'bottomB' ) );
	recordConv( levelName( bottom ), levelName( bottom ), channels, channels, bottom, activation, 'bottomB' );

	// Decoder: upsample-and-skip in one dispatch, then one convolution.
	for ( let level = levels - 2; level >= 0; level -- ) {

		dispatchers.push( buildUpsample2Kernel( maps[ level + 1 ], maps[ level ], {
			coarseWidth: maps[ level + 1 ].width,
			coarseHeight: maps[ level + 1 ].height,
			channels,
			additive: true,
			label: `ml-up${ level }`
		} ) );

		plan.push( {
			op: 'upsample',
			from: levelName( level + 1 ),
			to: levelName( level ),
			coarseWidth: maps[ level + 1 ].width,
			coarseHeight: maps[ level + 1 ].height,
			channels,
			additive: true
		} );

		dispatchers.push( addConv( maps[ level ], maps[ level ], channels, channels, level, activation, `up${ level }` ) );
		recordConv( levelName( level ), levelName( level ), channels, channels, level, activation, `up${ level }` );

	}

	// Head: project down to the output channels, linear. An activation here
	// would bound the output's sign or magnitude, and nothing this network
	// is meant to predict -- a pressure field, a correction, a velocity --
	// is one-sided.
	dispatchers.push( addConv( maps[ 0 ], output, channels, outChannels, 0, 'none', 'head' ) );
	recordConv( levelName( 0 ), 'output', channels, outChannels, 0, 'none', 'head' );

	const batched = tsl_array_n.createBatch( dispatchers );

	const parameters = blocks.reduce(
		( total, block ) => total + block.weights.count + block.bias.count,
		0
	);

	const macs = blocks.reduce( ( total, block ) => total + block.macs, 0 );

	/**
	 * Fills every weight block with He-normal values from a seeded
	 * generator. Reproducible: the same seed gives the same network, which
	 * is what makes a paired measurement against it valid.
	 */
	function randomize( withSeed = seed ) {

		const random = createSeededRandom( withSeed );

		for ( const block of blocks ) {

			block.weights.fromArray(
				heNormalWeights( block.inChannels, block.kernelSize, block.outChannels, random )
			);

			block.bias.fromArray( new Float32Array( block.outChannels ) );

		}

	}

	/**
	 * Loads trained weights. `byName` maps a block name to
	 * `{ weights, bias }`, both flat and already in this library's layout
	 * (reference.js's `fromPyTorchConv2dWeights` converts from PyTorch's).
	 *
	 * Every block must be supplied and every length must match. A network
	 * silently running one randomised layer among trained ones produces
	 * plausible-looking rubbish, and this is the only place that can catch
	 * it.
	 */
	function loadWeights( byName ) {

		for ( const block of blocks ) {

			const entry = byName[ block.name ];

			if ( entry === undefined ) {

				throw new Error( `ml: no weights supplied for block '${ block.name }'.` );

			}

			if ( entry.weights.length !== block.weights.count ) {

				throw new Error( `ml: block '${ block.name }' expects ${ block.weights.count } weights, got ${ entry.weights.length }.` );

			}

			if ( entry.bias && entry.bias.length !== block.bias.count ) {

				throw new Error( `ml: block '${ block.name }' expects ${ block.bias.count } biases, got ${ entry.bias.length }.` );

			}

			block.weights.fromArray( entry.weights );
			block.bias.fromArray( entry.bias ?? new Float32Array( block.outChannels ) );

		}

	}

	randomize( seed );

	return {
		input,
		output,
		maps,
		blocks,
		dispatchers,
		// Feed this to reference.js's `forwardReference` together with the
		// same weights to get a float64 CPU answer for the same network.
		plan,
		// Everything reference.js needs that the plan does not carry.
		config: { kernelSize, padding, leakySlope, inChannels, outChannels, channels, levels, width, height },

		/**
		 * One forward pass as a single submission. This is the form to use:
		 * the whole network goes out in one command buffer, which is the
		 * same mechanism that made the V-cycle 1.53x-1.74x faster.
		 */
		forward: () => profileBatch( 'ml-unet', dispatchers.length, batched ),

		/**
		 * One forward pass as one submission per layer. Slower by
		 * construction, and the only way to get a per-label dispatch
		 * breakdown out of profiling.js -- a batch reports itself as one
		 * entry and cannot say what was inside it.
		 */
		forwardUnbatched: () => {

			for ( const dispatch of dispatchers ) dispatch();

		},

		randomize,
		loadWeights,

		stats: {
			shape: [ width, height ],
			levels,
			channels,
			dispatches: dispatchers.length,
			parameters,
			// float32 on the wire and in the buffer.
			bytes: parameters * 4,
			macs,
			flops: macs * 2
		}
	};

}
