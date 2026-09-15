// Super-resolution: the display-path network. Same split as ml.test.js --
// the arithmetic is pinned against the float64 reference here, the GPU
// kernels are covered structurally, and examples/32-superres-smoke/ closes
// the loop on hardware.
//
// The test that matters most is "zero weights reproduce the classical
// upsample exactly". That property is what makes this thing shippable
// before it is trained, and it is the one a refactor would silently break.

import { describe, it, expect } from 'vitest';
import {
	featureIndex,
	sampleBilinearReference,
	sampleBicubicReference,
	monotonicCubic1dReference,
	shuffleResidualReference,
	lowResCoordinate,
	forwardReference
} from '../src/ml/reference.js';
import { createSuperResolver2, createClassicalUpsampler2 } from '../src/ml/superres.js';

function mapOf( width, height, channels, fn ) {

	const data = new Float64Array( width * height * channels );

	for ( let c = 0; c < channels; c ++ )
		for ( let y = 0; y < height; y ++ )
			for ( let x = 0; x < width; x ++ )
				data[ featureIndex( width, height, x, y, c ) ] = fn( x, y, c );

	return data;

}

describe( 'lowResCoordinate', () => {

	// The half-cell shift. A 4x upscale puts a low-res cell's centre midway
	// between high-res pixels 1 and 2 of its block, so that pixel position
	// must map to grid coordinate 0 exactly.
	it( 'puts a low cell centre at an integer grid coordinate', () => {

		expect( lowResCoordinate( 1.5, 4 ) ).toBeCloseTo( 0, 12 );
		expect( lowResCoordinate( 5.5, 4 ) ).toBeCloseTo( 1, 12 );
		expect( lowResCoordinate( 0.5, 2 ) ).toBeCloseTo( 0, 12 );

	} );

	it( 'is the identity at factor 1', () => {

		for ( const X of [ 0, 1, 7 ] ) expect( lowResCoordinate( X, 1 ) ).toBeCloseTo( X, 12 );

	} );

	// Dropping the shift is the classic off-by-half-a-cell bug; this pins
	// down that we are not doing the naive X/factor.
	it( 'is not X / factor', () => {

		expect( lowResCoordinate( 0, 4 ) ).toBeCloseTo( -0.375, 12 );
		expect( lowResCoordinate( 0, 4 ) ).not.toBeCloseTo( 0, 3 );

	} );

} );

describe( 'classical samplers', () => {

	const data = mapOf( 4, 4, 1, ( x, y ) => x + 10 * y );

	it( 'bilinear reproduces grid values at integer coordinates', () => {

		expect( sampleBilinearReference( data, 4, 4, 2, 1 ) ).toBeCloseTo( 12, 10 );
		expect( sampleBilinearReference( data, 4, 4, 0, 0 ) ).toBeCloseTo( 0, 10 );

	} );

	it( 'bilinear interpolates linearly on a linear field', () => {

		expect( sampleBilinearReference( data, 4, 4, 1.5, 1 ) ).toBeCloseTo( 11.5, 10 );

	} );

	it( 'bilinear clamps outside the domain rather than extrapolating off the end', () => {

		expect( sampleBilinearReference( data, 4, 4, -3, 0 ) ).toBeCloseTo( 0, 10 );
		expect( sampleBilinearReference( data, 4, 4, 9, 3 ) ).toBeCloseTo( 33, 10 );

	} );

	it( 'bicubic reproduces grid values at integer coordinates', () => {

		for ( let y = 0; y < 4; y ++ )
			for ( let x = 0; x < 4; x ++ )
				expect( sampleBicubicReference( data, 4, 4, x, y ) ).toBeCloseTo( x + 10 * y, 8 );

	} );

	it( 'both reproduce a constant field exactly', () => {

		const flat = mapOf( 6, 6, 1, () => 2.5 );

		for ( const gx of [ -1, 0, 0.5, 2.25, 5, 7 ] ) {

			expect( sampleBilinearReference( flat, 6, 6, gx, 1.3 ) ).toBeCloseTo( 2.5, 10 );
			expect( sampleBicubicReference( flat, 6, 6, gx, 1.3 ) ).toBeCloseTo( 2.5, 10 );

		}

	} );

} );

describe( 'monotonicCubic1dReference', () => {

	it( 'passes through its two centre samples', () => {

		expect( monotonicCubic1dReference( 0, 1, 2, 3, 0 ) ).toBeCloseTo( 1, 12 );
		expect( monotonicCubic1dReference( 0, 1, 2, 3, 1 ) ).toBeCloseTo( 2, 12 );

	} );

	// The whole reason this package uses a monotonic cubic and not
	// Catmull-Rom: on a density field an overshoot is an invented bright
	// spot, and an undershoot is a negative density.
	it( 'does not overshoot across a step', () => {

		for ( let t = 0; t <= 1.0001; t += 0.05 ) {

			const v = monotonicCubic1dReference( 0, 0, 1, 1, t );
			expect( v ).toBeGreaterThanOrEqual( -1e-12 );
			expect( v ).toBeLessThanOrEqual( 1 + 1e-12 );

		}

	} );

	it( 'is flat where its centre samples are equal', () => {

		expect( monotonicCubic1dReference( 5, 3, 3, -2, 0.5 ) ).toBeCloseTo( 3, 12 );

	} );

} );

describe( 'shuffleResidualReference', () => {

	const factor = 2;

	// The sub-pixel index is x-fastest within each factor x factor block.
	// Getting this transposed is the easiest bug in the whole module and
	// looks like a diagonal smear rather than like noise.
	it( 'rearranges sub-pixel channels x-fastest', () => {

		const source = mapOf( 2, 2, factor * factor, ( x, y, c ) => c );
		const base = mapOf( 2, 2, 1, () => 0 );

		const out = shuffleResidualReference( source, base, { lowWidth: 2, lowHeight: 2, factor, base: 'none' } );

		const at = ( X, Y ) => out[ featureIndex( 4, 4, X, Y, 0 ) ];

		expect( at( 0, 0 ) ).toBe( 0 );
		expect( at( 1, 0 ) ).toBe( 1 );
		expect( at( 0, 1 ) ).toBe( 2 );
		expect( at( 1, 1 ) ).toBe( 3 );
		// Second low-res cell along x repeats the pattern.
		expect( at( 2, 0 ) ).toBe( 0 );
		expect( at( 3, 1 ) ).toBe( 3 );

	} );

	it( 'adds the classical base to the residual', () => {

		const source = mapOf( 2, 2, factor * factor, () => 7 );
		const base = mapOf( 2, 2, 1, () => 1.5 );

		const out = shuffleResidualReference( source, base, { lowWidth: 2, lowHeight: 2, factor, base: 'bicubic' } );

		for ( const v of out ) expect( v ).toBeCloseTo( 8.5, 10 );

	} );

	it( 'produces the right output size', () => {

		const source = mapOf( 3, 5, 16, () => 0 );
		const base = mapOf( 3, 5, 1, () => 0 );

		const out = shuffleResidualReference( source, base, { lowWidth: 3, lowHeight: 5, factor: 4, base: 'none' } );

		expect( out.length ).toBe( 12 * 20 );

	} );

} );

describe( 'createSuperResolver2', () => {

	it( 'reports the shape it built', () => {

		const sr = createSuperResolver2( { shape: [ 96, 128 ], factor: 4, channels: 16, layers: 3 } );

		// 3 trunk convolutions + the sub-pixel head + the fused shuffle.
		expect( sr.stats.dispatches ).toBe( 5 );
		expect( sr.blocks.length ).toBe( 4 );
		expect( sr.stats.highShape ).toEqual( [ 384, 512 ] );
		expect( sr.plan.map( ( s ) => s.op ) ).toEqual( [ 'conv', 'conv', 'conv', 'conv', 'shuffleResidual' ] );

	} );

	// The reason every convolution runs at low resolution.
	it( 'prices the sub-pixel arrangement against upsampling first', () => {

		const sr = createSuperResolver2( { shape: [ 96, 128 ], factor: 4 } );

		expect( sr.stats.macsIfUpsampledFirst ).toBe( sr.stats.macs * 16 );

	} );

	it( 'emits a head with factor^2 output channels', () => {

		const sr = createSuperResolver2( { shape: [ 32, 32 ], factor: 3, channels: 8, layers: 2 } );
		const head = sr.blocks.at( - 1 );

		expect( head.name ).toBe( 'head' );
		expect( head.outChannels ).toBe( 9 );

	} );

	// The head must stay linear: a ReLU there would let the network brighten
	// a density field and never darken it.
	it( 'keeps the head linear', () => {

		const sr = createSuperResolver2( { shape: [ 16, 16 ], factor: 2, layers: 2 } );
		const headStep = sr.plan.find( ( s ) => s.name === 'head' );

		expect( headStep.activation ).toBe( 'none' );

	} );

	it( 'never writes a trunk buffer it is reading', () => {

		const sr = createSuperResolver2( { shape: [ 16, 16 ], factor: 2, layers: 4 } );

		for ( const step of sr.plan ) {

			if ( step.op === 'conv' ) expect( step.from ).not.toBe( step.to );

		}

	} );

	it( 'rejects a non-integer or non-positive factor', () => {

		expect( () => createSuperResolver2( { shape: [ 16, 16 ], factor: 2.5 } ) ).toThrow( /positive integer/ );
		expect( () => createSuperResolver2( { shape: [ 16, 16 ], factor: 0 } ) ).toThrow( /positive integer/ );

	} );

	it( 'rejects a non-2D shape', () => {

		expect( () => createSuperResolver2( { shape: [ 16 ] } ) ).toThrow( /2D shape/ );

	} );

} );

// *** The property the whole design rests on. ***
//
// With zero weights the residual path contributes a hard zero, so the
// output is the classical upsample exactly -- which is what makes this
// shippable untrained, and what makes a half-trained network read as
// "bicubic plus garbage" rather than as "garbage".
describe( 'zero weights reproduce the classical upsample exactly', () => {

	function zeroWeights( sr ) {

		const byName = {};

		for ( const block of sr.blocks ) {

			byName[ block.name ] = {
				weights: new Float32Array( block.weights.count ),
				bias: new Float32Array( block.outChannels )
			};

		}

		return byName;

	}

	for ( const baseKind of [ 'bicubic', 'bilinear' ] ) {

		it( `matches ${ baseKind } at every pixel`, () => {

			const lowWidth = 8, lowHeight = 6, factor = 4;
			const sr = createSuperResolver2( {
				shape: [ lowWidth, lowHeight ], factor, channels: 4, layers: 3, baseKind
			} );

			// Something with real structure, so a shift or a transpose would
			// show up rather than cancelling.
			const input = mapOf( lowWidth, lowHeight, 1, ( x, y ) =>
				Math.sin( 2 * Math.PI * x / lowWidth ) * Math.cos( 2 * Math.PI * y / lowHeight ) );

			const output = forwardReference( sr.plan, input, zeroWeights( sr ), sr.config ).get( 'output' );

			const sample = baseKind === 'bicubic' ? sampleBicubicReference : sampleBilinearReference;
			const highWidth = lowWidth * factor;
			const highHeight = lowHeight * factor;

			expect( output.length ).toBe( highWidth * highHeight );

			for ( let Y = 0; Y < highHeight; Y ++ ) {

				for ( let X = 0; X < highWidth; X ++ ) {

					const expected = sample(
						input, lowWidth, lowHeight,
						lowResCoordinate( X, factor ), lowResCoordinate( Y, factor ), 0
					);

					expect( output[ featureIndex( highWidth, highHeight, X, Y, 0 ) ] ).toBeCloseTo( expected, 12 );

				}

			}

		} );

	}

	it( 'preserves a constant field through the whole pipeline', () => {

		const sr = createSuperResolver2( { shape: [ 8, 8 ], factor: 4, channels: 4, layers: 2 } );
		const input = mapOf( 8, 8, 1, () => 0.375 );

		const output = forwardReference( sr.plan, input, zeroWeights( sr ), sr.config ).get( 'output' );

		for ( const v of output ) expect( v ).toBeCloseTo( 0.375, 10 );

	} );

} );

describe( 'createClassicalUpsampler2', () => {

	it( 'is one dispatch', () => {

		const up = createClassicalUpsampler2( { shape: [ 96, 128 ], factor: 4 } );

		expect( up.stats.dispatches ).toBe( 1 );
		expect( [ up.output.width, up.output.height ] ).toEqual( [ 384, 512 ] );

	} );

	it( 'rejects a non-2D shape', () => {

		expect( () => createClassicalUpsampler2( { shape: [ 96 ] } ) ).toThrow( /2D shape/ );

	} );

} );
