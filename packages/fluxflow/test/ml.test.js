// Two kinds of test, for the reason test/multigrid.test.js's header gives:
// GPU kernels can only be checked structurally without a renderer, so the
// arithmetic they implement is pinned down against reference.js instead,
// which is ordinary float64 JavaScript. examples/31-null-net-probe/ closes
// the loop on real hardware by running both on the same weights and
// reporting the difference.
//
// The adjointness test below is the one worth reading. It is not a
// property a machine-learning layer normally has to have, and this module
// pays real arithmetic for it -- see restrict2Reference's header.

import { describe, it, expect } from 'vitest';
import {
	featureIndex,
	weightIndex,
	conv2dReference,
	restrict2Reference,
	upsample2Reference,
	createSeededRandom,
	heNormalWeights,
	fromPyTorchConv2dWeights,
	forwardReference
} from '../src/ml/reference.js';
import { createUNet2 } from '../src/ml/unet.js';

function filledFeatureMap( width, height, channels, fn ) {

	const data = new Float64Array( width * height * channels );

	for ( let c = 0; c < channels; c ++ ) {

		for ( let y = 0; y < height; y ++ ) {

			for ( let x = 0; x < width; x ++ ) {

				data[ featureIndex( width, height, x, y, c ) ] = fn( x, y, c );

			}

		}

	}

	return data;

}

function dot( a, b ) {

	let sum = 0;
	for ( let i = 0; i < a.length; i ++ ) sum += a[ i ] * b[ i ];
	return sum;

}

describe( 'layout helpers', () => {

	it( 'flattens feature maps first-axis-fastest, matching tsl_array_n', () => {

		expect( featureIndex( 4, 4, 1, 0, 0 ) ).toBe( 1 );
		expect( featureIndex( 4, 4, 0, 1, 0 ) ).toBe( 4 );
		expect( featureIndex( 4, 4, 0, 0, 1 ) ).toBe( 16 );

	} );

	it( 'gives every weight a distinct index', () => {

		const inChannels = 3, kernelSize = 3, outChannels = 2;
		const seen = new Set();

		for ( let co = 0; co < outChannels; co ++ ) {
			for ( let ky = 0; ky < kernelSize; ky ++ ) {
				for ( let kx = 0; kx < kernelSize; kx ++ ) {
					for ( let ci = 0; ci < inChannels; ci ++ ) {
						seen.add( weightIndex( inChannels, kernelSize, ci, kx, ky, co ) );
					}
				}
			}
		}

		expect( seen.size ).toBe( inChannels * kernelSize * kernelSize * outChannels );
		expect( Math.max( ...seen ) ).toBe( inChannels * kernelSize * kernelSize * outChannels - 1 );

	} );

} );

describe( 'conv2dReference', () => {

	const options = { width: 4, height: 4, inChannels: 1, outChannels: 1, kernelSize: 3 };

	it( 'reproduces its input through a centre-delta kernel', () => {

		const input = filledFeatureMap( 4, 4, 1, ( x, y ) => x + 10 * y );
		const weights = new Float32Array( 9 );
		weights[ weightIndex( 1, 3, 0, 1, 1, 0 ) ] = 1;

		const output = conv2dReference( input, weights, null, options );

		for ( let i = 0; i < input.length; i ++ ) expect( output[ i ] ).toBeCloseTo( input[ i ], 12 );

	} );

	it( 'shifts by one cell through an off-centre delta kernel', () => {

		const input = filledFeatureMap( 4, 4, 1, ( x, y ) => x + 10 * y );
		const weights = new Float32Array( 9 );
		// kx = 2 is dx = +1: the output at x reads the input at x+1.
		weights[ weightIndex( 1, 3, 0, 2, 1, 0 ) ] = 1;

		const output = conv2dReference( input, weights, null, options );

		expect( output[ featureIndex( 4, 4, 0, 0, 0 ) ] ).toBeCloseTo( 1, 12 );
		expect( output[ featureIndex( 4, 4, 1, 2, 0 ) ] ).toBeCloseTo( 22, 12 );
		// Clamped at the far edge: x = 3 reads x = 3, not off the end.
		expect( output[ featureIndex( 4, 4, 3, 0, 0 ) ] ).toBeCloseTo( 3, 12 );

	} );

	// Tolerance is float32's, not float64's, and deliberately so: the
	// weights are a Float32Array because that is what ships to the GPU, and
	// 1/9 is not representable there, so nine of them do not sum to exactly
	// one. The reference accumulates in float64 but it cannot un-round its
	// inputs. 7.00000005 is the right answer here.
	it( 'averages a constant field to that constant (partition of unity)', () => {

		const input = filledFeatureMap( 4, 4, 1, () => 7 );
		const weights = new Float32Array( 9 ).fill( 1 / 9 );

		const output = conv2dReference( input, weights, null, options );

		for ( let i = 0; i < output.length; i ++ ) expect( output[ i ] ).toBeCloseTo( 7, 6 );

	} );

	it( 'differs from zero padding only at the boundary', () => {

		const input = filledFeatureMap( 4, 4, 1, () => 1 );
		const weights = new Float32Array( 9 ).fill( 1 / 9 );

		const clamped = conv2dReference( input, weights, null, { ...options, padding: 'clamp' } );
		const zeroed = conv2dReference( input, weights, null, { ...options, padding: 'zero' } );

		// Interior cells see no boundary, so both agree there.
		expect( zeroed[ featureIndex( 4, 4, 1, 1, 0 ) ] ).toBeCloseTo( clamped[ featureIndex( 4, 4, 1, 1, 0 ) ], 10 );
		// A corner sees 4 of 9 taps under zero padding and all 9 under clamping.
		expect( zeroed[ featureIndex( 4, 4, 0, 0, 0 ) ] ).toBeCloseTo( 4 / 9, 6 );
		expect( clamped[ featureIndex( 4, 4, 0, 0, 0 ) ] ).toBeCloseTo( 1, 6 );

	} );

	it( 'adds the bias and applies the activation after it', () => {

		const input = filledFeatureMap( 4, 4, 1, () => 0 );
		const weights = new Float32Array( 9 );
		const bias = new Float32Array( [ -2 ] );

		const linear = conv2dReference( input, weights, bias, options );
		expect( linear[ 0 ] ).toBeCloseTo( -2, 12 );

		const relu = conv2dReference( input, weights, bias, { ...options, activation: 'relu' } );
		expect( relu[ 0 ] ).toBe( 0 );

		const leaky = conv2dReference( input, weights, bias, { ...options, activation: 'leakyRelu' } );
		expect( leaky[ 0 ] ).toBeCloseTo( -0.02, 12 );

	} );

	it( 'sums across input channels', () => {

		const input = filledFeatureMap( 4, 4, 2, ( x, y, c ) => ( c === 0 ? 3 : 5 ) );
		const weights = new Float32Array( 2 * 9 );
		weights[ weightIndex( 2, 3, 0, 1, 1, 0 ) ] = 1;
		weights[ weightIndex( 2, 3, 1, 1, 1, 0 ) ] = 1;

		const output = conv2dReference( input, weights, null, { ...options, inChannels: 2 } );

		expect( output[ 0 ] ).toBeCloseTo( 8, 12 );

	} );

	it( 'rejects an even kernel size', () => {

		expect( () => conv2dReference( new Float64Array( 16 ), new Float32Array( 4 ), null, { ...options, kernelSize: 2 } ) )
			.toThrow( /kernelSize must be odd/ );

	} );

} );

describe( 'restrict2Reference / upsample2Reference', () => {

	it( 'restricts a constant field to the same constant', () => {

		const input = filledFeatureMap( 8, 8, 2, () => 3 );
		const output = restrict2Reference( input, { width: 8, height: 8, channels: 2 } );

		expect( output.length ).toBe( 4 * 4 * 2 );
		for ( let i = 0; i < output.length; i ++ ) expect( output[ i ] ).toBeCloseTo( 3, 10 );

	} );

	it( 'upsamples a constant field to the same constant', () => {

		const coarse = filledFeatureMap( 4, 4, 1, () => 2 );
		const fine = upsample2Reference( coarse, { coarseWidth: 4, coarseHeight: 4, channels: 1 } );

		expect( fine.length ).toBe( 8 * 8 );
		for ( let i = 0; i < fine.length; i ++ ) expect( fine[ i ] ).toBeCloseTo( 2, 10 );

	} );

	it( 'adds into an existing fine field when asked (the skip connection)', () => {

		const coarse = filledFeatureMap( 4, 4, 1, () => 2 );
		const skip = filledFeatureMap( 8, 8, 1, () => 10 );
		const fine = upsample2Reference( coarse, { coarseWidth: 4, coarseHeight: 4, channels: 1, addTo: skip } );

		for ( let i = 0; i < fine.length; i ++ ) expect( fine[ i ] ).toBeCloseTo( 12, 10 );
		// addTo is not mutated -- the reference returns a new array.
		expect( skip[ 0 ] ).toBe( 10 );

	} );

	// *** The reason restriction is full-weighting rather than average
	// pooling. *** If this fails, the transfer pair is not symmetric and the
	// shape is no longer usable as a CG preconditioner -- see
	// restrict2Reference's header for what this codebase has already paid
	// for getting that wrong once.
	it( 'restriction is the exact adjoint of upsampling, R = P^T / 4', () => {

		const random = createSeededRandom( 12345 );
		const fineWidth = 8, fineHeight = 8, channels = 3;
		const coarseWidth = 4, coarseHeight = 4;

		const u = filledFeatureMap( fineWidth, fineHeight, channels, () => random() * 2 - 1 );
		const v = filledFeatureMap( coarseWidth, coarseHeight, channels, () => random() * 2 - 1 );

		const Ru = restrict2Reference( u, { width: fineWidth, height: fineHeight, channels } );
		const Pv = upsample2Reference( v, { coarseWidth, coarseHeight, channels } );

		expect( dot( Ru, v ) * 4 ).toBeCloseTo( dot( u, Pv ), 10 );

	} );

	it( 'is adjoint on a non-square grid too, where the boundary clamping differs per axis', () => {

		const random = createSeededRandom( 999 );
		const u = filledFeatureMap( 8, 4, 1, () => random() * 2 - 1 );
		const v = filledFeatureMap( 4, 2, 1, () => random() * 2 - 1 );

		const Ru = restrict2Reference( u, { width: 8, height: 4, channels: 1 } );
		const Pv = upsample2Reference( v, { coarseWidth: 4, coarseHeight: 2, channels: 1 } );

		expect( dot( Ru, v ) * 4 ).toBeCloseTo( dot( u, Pv ), 10 );

	} );

	it( 'rejects odd dimensions', () => {

		expect( () => restrict2Reference( new Float64Array( 15 ), { width: 5, height: 3, channels: 1 } ) )
			.toThrow( /even dimensions/ );

	} );

} );

describe( 'weight plumbing', () => {

	it( 'is reproducible from a seed', () => {

		const a = createSeededRandom( 42 );
		const b = createSeededRandom( 42 );

		for ( let i = 0; i < 10; i ++ ) expect( a() ).toBe( b() );

	} );

	it( 'initialises He-normal weights of the right count, all finite', () => {

		const weights = heNormalWeights( 4, 3, 8, createSeededRandom( 7 ) );

		expect( weights.length ).toBe( 4 * 3 * 3 * 8 );
		for ( const w of weights ) expect( Number.isFinite( w ) ).toBe( true );

		// Variance should sit near 2 / fanIn. Loose bounds -- this is a
		// sanity check that the scaling was applied at all, not a
		// distribution test.
		const fanIn = 4 * 3 * 3;
		const expected = Math.sqrt( 2 / fanIn );
		let sumSq = 0;
		for ( const w of weights ) sumSq += w * w;
		const observed = Math.sqrt( sumSq / weights.length );

		expect( observed ).toBeGreaterThan( expected * 0.7 );
		expect( observed ).toBeLessThan( expected * 1.3 );

	} );

	it( 'converts PyTorch conv2d weights into this library\'s layout', () => {

		const inChannels = 2, outChannels = 3, kernelSize = 3;
		const total = inChannels * outChannels * kernelSize * kernelSize;

		// Each entry encodes its own (co, ci, ky, kx) so a permutation
		// error cannot pass.
		const torch = new Float32Array( total );

		for ( let co = 0; co < outChannels; co ++ ) {
			for ( let ci = 0; ci < inChannels; ci ++ ) {
				for ( let ky = 0; ky < kernelSize; ky ++ ) {
					for ( let kx = 0; kx < kernelSize; kx ++ ) {
						const index = kx + ky * kernelSize + ci * kernelSize * kernelSize
							+ co * kernelSize * kernelSize * inChannels;
						torch[ index ] = 1000 * co + 100 * ci + 10 * ky + kx;
					}
				}
			}
		}

		const converted = fromPyTorchConv2dWeights( torch, { inChannels, outChannels, kernelSize } );

		for ( let co = 0; co < outChannels; co ++ ) {
			for ( let ci = 0; ci < inChannels; ci ++ ) {
				for ( let ky = 0; ky < kernelSize; ky ++ ) {
					for ( let kx = 0; kx < kernelSize; kx ++ ) {
						expect( converted[ weightIndex( inChannels, kernelSize, ci, kx, ky, co ) ] )
							.toBe( 1000 * co + 100 * ci + 10 * ky + kx );
					}
				}
			}
		}

	} );

	it( 'rejects a PyTorch tensor of the wrong length', () => {

		expect( () => fromPyTorchConv2dWeights( new Float32Array( 5 ), { inChannels: 2, outChannels: 3, kernelSize: 3 } ) )
			.toThrow( /expected 54 weights/ );

	} );

} );

// Structural only -- no renderer, so nothing here executes on a GPU. It
// does build every kernel for real, which exercises the tap combinatorics
// and the index clamping the same way test/multigrid.test.js does.
describe( 'createUNet2', () => {

	it( 'constructs at the candidate shape and reports its own cost', () => {

		const net = createUNet2( { shape: [ 64, 64 ], channels: 16, levels: 3 } );

		// stem + (conv + restrict) x 2 + 2 bottom + (upsample + conv) x 2 + head
		expect( net.stats.dispatches ).toBe( 12 );
		expect( net.dispatchers.length ).toBe( 12 );

		// 8 convolutions: 1 stem (1->16), 2 encoder (one per level above the
		// bottom), 2 bottom, 2 decoder, 1 head (16->1), all 3x3. Plus 2
		// restrictions and 2 upsamples, which carry no weights, making up
		// the 12 dispatches above.
		expect( net.blocks.length ).toBe( 8 );

		const expectedParameters = net.blocks.reduce(
			( total, block ) => total + block.inChannels * block.kernelSize * block.kernelSize * block.outChannels + block.outChannels,
			0
		);

		expect( net.stats.parameters ).toBe( expectedParameters );
		expect( net.stats.bytes ).toBe( expectedParameters * 4 );
		expect( net.stats.flops ).toBe( net.stats.macs * 2 );

	} );

	it( 'allocates one feature map per level, each halved', () => {

		const net = createUNet2( { shape: [ 64, 64 ], levels: 3 } );

		expect( net.maps.map( ( m ) => [ m.width, m.height ] ) )
			.toEqual( [ [ 64, 64 ], [ 32, 32 ], [ 16, 16 ] ] );

	} );

	it( 'builds a single-level network with no transfer operators', () => {

		const net = createUNet2( { shape: [ 16, 16 ], levels: 1 } );

		// stem + 2 bottom + head, and nothing to restrict or upsample.
		expect( net.stats.dispatches ).toBe( 4 );

	} );

	it( 'rejects a shape that cannot be halved enough times', () => {

		// 12 is divisible by 4 but not by 8, so three levels are fine and
		// four are not.
		expect( () => createUNet2( { shape: [ 12, 12 ], levels: 3 } ) ).not.toThrow();
		expect( () => createUNet2( { shape: [ 12, 12 ], levels: 4 } ) ).toThrow( /not divisible/ );

	} );

	it( 'rejects a non-2D shape', () => {

		expect( () => createUNet2( { shape: [ 64, 64, 64 ] } ) ).toThrow( /2D shape/ );

	} );

	it( 'names every weight block, uniquely', () => {

		const net = createUNet2( { shape: [ 32, 32 ], levels: 3 } );
		const names = net.blocks.map( ( b ) => b.name );

		expect( new Set( names ).size ).toBe( names.length );
		expect( names ).toContain( 'stem' );
		expect( names ).toContain( 'head' );

	} );

	it( 'refuses to load a partial weight set', () => {

		const net = createUNet2( { shape: [ 16, 16 ], levels: 1 } );

		expect( () => net.loadWeights( {} ) ).toThrow( /no weights supplied for block/ );

	} );

	it( 'refuses weights of the wrong length', () => {

		const net = createUNet2( { shape: [ 16, 16 ], levels: 1 } );
		const byName = {};

		for ( const block of net.blocks ) {

			byName[ block.name ] = { weights: new Float32Array( block.weights.count ) };

		}

		byName.head = { weights: new Float32Array( 3 ) };

		expect( () => net.loadWeights( byName ) ).toThrow( /expects .* weights, got 3/ );

	} );

} );

// The plan is what keeps the CPU reference and the GPU kernels describing
// the same network. These tests check the plan executes and that its wiring
// -- which buffer feeds which, and which block is last -- is what the
// architecture says it is.
describe( 'forwardReference over a createUNet2 plan', () => {

	function zeroWeights( net, overrides = {} ) {

		const byName = {};

		for ( const block of net.blocks ) {

			byName[ block.name ] = {
				weights: new Float32Array( block.weights.count ),
				bias: new Float32Array( block.outChannels )
			};

		}

		for ( const [ name, mutate ] of Object.entries( overrides ) ) mutate( byName[ name ] );

		return byName;

	}

	it( 'produces an output of the right shape', () => {

		const net = createUNet2( { shape: [ 16, 16 ], levels: 3, channels: 4 } );
		const input = new Float64Array( 16 * 16 * 1 );

		const buffers = forwardReference( net.plan, input, zeroWeights( net ), net.config );

		expect( buffers.get( 'output' ).length ).toBe( 16 * 16 * 1 );

	} );

	it( 'returns zeros for an all-zero network, whatever the input', () => {

		const net = createUNet2( { shape: [ 16, 16 ], levels: 3, channels: 4 } );
		const random = createSeededRandom( 3 );
		const input = Float64Array.from( { length: 16 * 16 }, () => random() );

		const output = forwardReference( net.plan, input, zeroWeights( net ), net.config ).get( 'output' );

		for ( const v of output ) expect( v ).toBe( 0 );

	} );

	// The head is linear and last, so its bias survives to the output
	// untouched. If the plan ever ordered the head before something else,
	// or applied an activation to it, this is what would catch it.
	it( 'passes the head bias straight through to the output', () => {

		const net = createUNet2( { shape: [ 16, 16 ], levels: 3, channels: 4 } );
		const input = new Float64Array( 16 * 16 );

		const byName = zeroWeights( net, { head: ( entry ) => { entry.bias[ 0 ] = -2.5; } } );
		const output = forwardReference( net.plan, input, byName, net.config ).get( 'output' );

		for ( const v of output ) expect( v ).toBeCloseTo( -2.5, 10 );

	} );

	// ReLU on the interior blocks means a negative bias there is clipped and
	// cannot reach the output, unlike the head's. This pins down that the
	// activation really is applied where the plan says.
	it( 'clips a negative interior bias at the ReLU', () => {

		const net = createUNet2( { shape: [ 16, 16 ], levels: 3, channels: 4, activation: 'relu' } );
		const input = new Float64Array( 16 * 16 );

		const byName = zeroWeights( net, { stem: ( entry ) => entry.bias.fill( -5 ) } );
		const output = forwardReference( net.plan, input, byName, net.config ).get( 'output' );

		for ( const v of output ) expect( v ).toBe( 0 );

	} );

	it( 'names every plan buffer before reading it', () => {

		const net = createUNet2( { shape: [ 32, 32 ], levels: 4, channels: 4 } );
		const written = new Set( [ 'input' ] );

		for ( const step of net.plan ) {

			expect( written.has( step.from ) ).toBe( true );
			// An additive upsample reads its destination as well as writing it.
			if ( step.op === 'upsample' && step.additive ) expect( written.has( step.to ) ).toBe( true );
			written.add( step.to );

		}

		expect( written.has( 'output' ) ).toBe( true );

	} );

	it( 'refuses a plan step with no weights for it', () => {

		const net = createUNet2( { shape: [ 16, 16 ], levels: 1, channels: 4 } );

		expect( () => forwardReference( net.plan, new Float64Array( 256 ), {}, net.config ) )
			.toThrow( /no weights supplied for block/ );

	} );

} );
