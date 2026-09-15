// Plain-JS float64 reference implementations of every layer in layers.js,
// plus the weight plumbing around them (seeded initialisation, PyTorch
// layout conversion, flat save/load).
//
// *** Why a reference implementation exists at all ***
//
// This package has done this once before and it paid for itself. When
// multigrid.js's V-cycle looked wrong in the dev sandbox, the thing that
// separated "real bug" from "WebGL2-fallback artifact" was a plain-JS,
// float64, no-GPU-at-all implementation of the same formula
// (examples/06-multigrid-preconditioner/'s header records the whole
// episode, and the bug it found was real: a per-cell diagonal, not a
// constant one). A neural network is strictly worse to debug than a
// Laplacian -- the output of a wrong convolution and the output of a right
// one both look like noise until the weights are trained -- so the
// reference comes first here rather than after the first confusing result.
//
// Two concrete uses:
//
//   1. **Unit tests without a GPU.** Everything in this file is ordinary
//      JavaScript, so `test/ml.test.js` can check the convolution
//      arithmetic against hand-computed values. The GPU kernels can only be
//      tested structurally (the repo's established convention -- see
//      test/multigrid.test.js's header), so the arithmetic has to be
//      pinned down somewhere, and this is that somewhere.
//   2. **Cross-checking the real kernels.** examples/31-null-net-probe/
//      runs the GPU forward pass and this implementation on the same
//      weights and the same input, and reports the max absolute
//      difference. float32-versus-float64 makes an exact match impossible;
//      a disagreement larger than accumulated rounding is a bug.
//
// *** Layouts, stated once ***
//
// Feature maps: `[width, height, channels]`, first index varying fastest,
// matching tsl_array_n's own flattening (`index = i + j*width + ...`). So
// one channel's plane is contiguous, adjacent threads in a kernel dispatch
// over `[width, height, channels]` write adjacent addresses, and the
// convolution's inner loop over input channels sweeps whole planes.
//
// Convolution weights: `[inChannels, kernelWidth, kernelHeight, outChannels]`,
// again first-fastest. Input channel is the innermost loop of the
// convolution, so making it the fastest-varying axis keeps that loop's
// reads adjacent.
//
// That second layout is this library's own, and it is **not** PyTorch's
// (which is `[outChannels, inChannels, kernelHeight, kernelWidth]`).
// Anything trained offline will come out in PyTorch's, so
// `fromPyTorchConv2dWeights` below converts, and it is the only sanctioned
// way to do it -- transposing by hand at a call site is how a network ends
// up silently computing a transposed convolution and looking merely
// "undertrained".

/**
 * Flat index into a `[width, height, channels]` feature map.
 * Mirrors tsl_array_n's own strides (first axis fastest).
 */
export function featureIndex( width, height, x, y, c ) {

	return x + y * width + c * width * height;

}

/**
 * Flat index into a `[inChannels, kw, kh, outChannels]` weight block.
 */
export function weightIndex( inChannels, kernelSize, ci, kx, ky, co ) {

	return ci
		+ kx * inChannels
		+ ky * inChannels * kernelSize
		+ co * inChannels * kernelSize * kernelSize;

}

/**
 * Clamp-to-edge (replicate) sampling, the default padding.
 *
 * Zero padding is the machine-learning default, and it is the wrong default
 * *here*. A zero ring around the domain is a statement that the field is
 * zero just outside it, and in this package the domain edge is a real
 * physical boundary carrying its own condition (a wall, an inflow, an
 * outflow) -- never a zero. Replicating the edge says "no information",
 * which is the honest thing for a layer that has not been told what the
 * boundary is. `padding: 'zero'` is available for parity with offline
 * training code that assumed it; if you train with zero padding you must
 * infer with it too.
 */
function sampleClamped( data, width, height, x, y, c ) {

	const sx = x < 0 ? 0 : ( x > width - 1 ? width - 1 : x );
	const sy = y < 0 ? 0 : ( y > height - 1 ? height - 1 : y );

	return data[ featureIndex( width, height, sx, sy, c ) ];

}

function sampleZero( data, width, height, x, y, c ) {

	if ( x < 0 || y < 0 || x > width - 1 || y > height - 1 ) return 0;

	return data[ featureIndex( width, height, x, y, c ) ];

}

export function applyActivation( value, activation, leakySlope = 0.01 ) {

	if ( activation === 'none' ) return value;
	if ( activation === 'relu' ) return value > 0 ? value : 0;
	if ( activation === 'leakyRelu' ) return value > 0 ? value : value * leakySlope;

	throw new Error( `ml: unknown activation '${ activation }'.` );

}

/**
 * Reference 2D convolution. Returns a new Float64Array in
 * `[width, height, outChannels]` layout.
 *
 * Accumulates in float64, which is the point: the GPU accumulates in
 * float32, and the difference between the two is the number the probe
 * reports. This package has already been bitten once by float32 running out
 * of digits rather than iterations (examples/28-drop-into-pool/), so having
 * a float64 answer to compare against is not a luxury.
 */
export function conv2dReference( input, weights, bias, options ) {

	const {
		width, height, inChannels, outChannels,
		kernelSize = 3, padding = 'clamp',
		activation = 'none', leakySlope = 0.01
	} = options;

	if ( kernelSize % 2 !== 1 ) throw new Error( `ml: kernelSize must be odd, got ${ kernelSize }.` );

	const radius = ( kernelSize - 1 ) / 2;
	const sample = padding === 'zero' ? sampleZero : sampleClamped;
	const output = new Float64Array( width * height * outChannels );

	for ( let co = 0; co < outChannels; co ++ ) {

		for ( let y = 0; y < height; y ++ ) {

			for ( let x = 0; x < width; x ++ ) {

				let acc = bias ? bias[ co ] : 0;

				for ( let ky = 0; ky < kernelSize; ky ++ ) {

					for ( let kx = 0; kx < kernelSize; kx ++ ) {

						const sx = x + kx - radius;
						const sy = y + ky - radius;

						for ( let ci = 0; ci < inChannels; ci ++ ) {

							acc += sample( input, width, height, sx, sy, ci )
								* weights[ weightIndex( inChannels, kernelSize, ci, kx, ky, co ) ];

						}

					}

				}

				output[ featureIndex( width, height, x, y, co ) ] = applyActivation( acc, activation, leakySlope );

			}

		}

	}

	return output;

}

// The 4 (index, weight) taps of the full-weighting restriction filter along
// one axis, boundary-clamped. Identical to multigrid.js's
// `restrictionTapsForAxis`, and that is the whole point -- see
// restrict2Reference below.
function restrictionTapsForAxis( coarseIndex, coarseCount ) {

	return [
		{ index: coarseIndex > 0 ? 2 * coarseIndex - 1 : 2 * coarseIndex, weight: 0.125 },
		{ index: 2 * coarseIndex, weight: 0.375 },
		{ index: 2 * coarseIndex + 1, weight: 0.375 },
		{ index: coarseIndex + 1 < coarseCount ? 2 * coarseIndex + 2 : 2 * coarseIndex + 1, weight: 0.125 }
	];

}

/**
 * Reference 2x restriction (full weighting). `[w, h, c]` -> `[w/2, h/2, c]`.
 *
 * *** Why this is not plain 2x2 average pooling ***
 *
 * Average pooling is what a machine-learning library would use and it is
 * the wrong operator here, for a reason that is specific to what this
 * network might be asked to do. Paired with the bilinear upsample below,
 * plain averaging is **not** adjoint: the upsample's transpose is the
 * 4-tap filter above, scaled, not the 2-tap box filter. The two differ,
 * and the difference is not a constant.
 *
 * Adjointness matters because a symmetric preconditioner is what
 * conjugate gradients requires, and `(Ru, v) == (u, Pv)/4` is exactly the
 * property that makes a down/up pair symmetric. multigrid.js carries the
 * scar tissue from getting this wrong once already -- its `relax()` and
 * `buildCorrectKernel()` comments record a real, confirmed-on-hardware
 * long-run divergence bug that came from a transfer-pair asymmetry -- and
 * the perf investigation later measured the pair and found `R = P^T/4`
 * exactly. Reproducing that pair costs one dispatch either way (16 taps
 * instead of 4, and arithmetic is not what costs here) and keeps the
 * preconditioner option open. Taking the box filter would close it
 * silently.
 *
 * The channel axis is not restricted. Only the two spatial axes are.
 */
export function restrict2Reference( input, options ) {

	const { width, height, channels } = options;

	if ( width % 2 !== 0 || height % 2 !== 0 ) {

		throw new Error( `ml: restrict2 needs even dimensions, got ${ width }x${ height }.` );

	}

	const cw = width / 2;
	const ch = height / 2;
	const output = new Float64Array( cw * ch * channels );

	for ( let c = 0; c < channels; c ++ ) {

		for ( let y = 0; y < ch; y ++ ) {

			for ( let x = 0; x < cw; x ++ ) {

				let sum = 0;

				for ( const tx of restrictionTapsForAxis( x, cw ) ) {

					for ( const ty of restrictionTapsForAxis( y, ch ) ) {

						sum += input[ featureIndex( width, height, tx.index, ty.index, c ) ]
							* tx.weight * ty.weight;

					}

				}

				output[ featureIndex( cw, ch, x, y, c ) ] = sum;

			}

		}

	}

	return output;

}

/**
 * Reference 2x bilinear upsample, optionally adding into an existing fine
 * field (the skip connection). `[w, h, c]` -> `[2w, 2h, c]`.
 *
 * The tap weights are the same 3/4-1/4 pair multigrid.js's own
 * `correctionTapsForAxis` uses, and boundary-clamped the same way, for the
 * same reason: it is the transpose of the restriction above, so the
 * down/up pair is adjoint rather than merely "roughly inverse". That
 * property is not decorative here -- if this network is ever used as a
 * preconditioner, a non-symmetric operator is exactly what breaks CG, and
 * multigrid.js's relax() comment records what that failure looks like in
 * this codebase.
 */
export function upsample2Reference( coarse, options ) {

	const { coarseWidth, coarseHeight, channels, addTo = null } = options;

	const fw = coarseWidth * 2;
	const fh = coarseHeight * 2;
	const output = addTo ? Float64Array.from( addTo ) : new Float64Array( fw * fh * channels );

	const tapsForAxis = ( fineIdx, coarseCount ) => {

		const base = Math.floor( fineIdx / 2 );

		// Odd fine index leans towards base+1, even towards base-1, both
		// clamped at the ends -- the transpose of the averaging above.
		if ( fineIdx % 2 === 0 ) {

			const other = base > 0 ? base - 1 : base;
			return [ { index: base, weight: 0.75 }, { index: other, weight: 0.25 } ];

		}

		const other = base + 1 < coarseCount ? base + 1 : base;
		return [ { index: base, weight: 0.75 }, { index: other, weight: 0.25 } ];

	};

	for ( let c = 0; c < channels; c ++ ) {

		for ( let y = 0; y < fh; y ++ ) {

			for ( let x = 0; x < fw; x ++ ) {

				let sum = 0;

				for ( const tx of tapsForAxis( x, coarseWidth ) ) {

					for ( const ty of tapsForAxis( y, coarseHeight ) ) {

						sum += coarse[ featureIndex( coarseWidth, coarseHeight, tx.index, ty.index, c ) ]
							* tx.weight * ty.weight;

					}

				}

				output[ featureIndex( fw, fh, x, y, c ) ] += sum;

			}

		}

	}

	return output;

}

// *** Seeded initialisation ***
//
// mulberry32: small, fast, and -- the only property that matters here --
// reproducible from an integer seed. The null-net probe measures the cost
// of a network *shape*, and a shape whose weights change from run to run is
// not a shape you can measure twice. Every measurement in this repo is a
// paired comparison with state restored between arms; random weights that
// do not restore would quietly break that discipline.
export function createSeededRandom( seed ) {

	let a = seed >>> 0;

	return function random() {

		a = ( a + 0x6D2B79F5 ) >>> 0;
		let t = Math.imul( a ^ ( a >>> 15 ), 1 | a );
		t = ( t + Math.imul( t ^ ( t >>> 7 ), 61 | t ) ) ^ t;
		return ( ( t ^ ( t >>> 14 ) ) >>> 0 ) / 4294967296;

	};

}

/**
 * He/Kaiming normal initialisation, the standard choice for ReLU networks:
 * variance 2 / fanIn, so activations neither vanish nor explode through
 * depth. Box-Muller for the normal draw.
 *
 * This produces a network that computes nonsense, and that is correct for
 * its purpose. The probe needs realistic *magnitudes* (so nothing
 * denormalises or overflows and distorts the timing) with no training run
 * behind them.
 */
export function heNormalWeights( inChannels, kernelSize, outChannels, random ) {

	const fanIn = inChannels * kernelSize * kernelSize;
	const scale = Math.sqrt( 2 / fanIn );
	const out = new Float32Array( fanIn * outChannels );

	for ( let i = 0; i < out.length; i ++ ) {

		// Box-Muller. u is nudged off zero because log(0) is -Infinity and
		// one Infinity in a weight block poisons every output that reads it.
		const u = Math.max( random(), Number.MIN_VALUE );
		const v = random();
		out[ i ] = Math.sqrt( -2 * Math.log( u ) ) * Math.cos( 2 * Math.PI * v ) * scale;

	}

	return out;

}

/**
 * Converts PyTorch's `[outChannels, inChannels, kernelHeight, kernelWidth]`
 * weight tensor (flat, row-major, as `state_dict()[...].flatten()` gives
 * it) into this library's `[inChannels, kw, kh, outChannels]`.
 *
 * The only sanctioned way to bring in offline-trained weights. Both
 * layouts hold the same numbers in different orders, so getting it wrong
 * throws no error and produces a network that trains fine and infers
 * wrong -- which is indistinguishable from undertraining unless you know
 * to suspect it.
 */
export function fromPyTorchConv2dWeights( flat, { inChannels, outChannels, kernelSize } ) {

	const expected = inChannels * outChannels * kernelSize * kernelSize;

	if ( flat.length !== expected ) {

		throw new Error( `ml: expected ${ expected } weights for ${ inChannels }->${ outChannels } k${ kernelSize }, got ${ flat.length }.` );

	}

	const out = new Float32Array( expected );

	for ( let co = 0; co < outChannels; co ++ ) {

		for ( let ci = 0; ci < inChannels; ci ++ ) {

			for ( let ky = 0; ky < kernelSize; ky ++ ) {

				for ( let kx = 0; kx < kernelSize; kx ++ ) {

					const torch = kx
						+ ky * kernelSize
						+ ci * kernelSize * kernelSize
						+ co * kernelSize * kernelSize * inChannels;

					out[ weightIndex( inChannels, kernelSize, ci, kx, ky, co ) ] = flat[ torch ];

				}

			}

		}

	}

	return out;

}

/**
 * Executes a `createUNet2().plan` in float64 JavaScript, returning the
 * named buffers it produced (including `'output'`).
 *
 * This is the CPU arm of the cross-check in
 * examples/31-null-net-probe/. It runs the plan the builder emitted rather
 * than a transcription of the architecture, so it cannot drift out of sync
 * with what the GPU is doing -- which is the failure mode that makes a
 * reference implementation worse than useless: a disagreement that looks
 * like a GPU bug and is actually the reference being out of date.
 *
 * `weightsByName` is the same object shape `loadWeights` takes.
 */
export function forwardReference( plan, inputData, weightsByName, config ) {

	const { kernelSize, padding, leakySlope } = config;
	const buffers = new Map( [ [ 'input', inputData ] ] );

	const require = ( name ) => {

		const buffer = buffers.get( name );

		if ( buffer === undefined ) throw new Error( `ml: plan reads buffer '${ name }' before anything wrote it.` );

		return buffer;

	};

	for ( const step of plan ) {

		if ( step.op === 'conv' ) {

			const entry = weightsByName[ step.name ];

			if ( entry === undefined ) throw new Error( `ml: no weights supplied for block '${ step.name }'.` );

			buffers.set( step.to, conv2dReference( require( step.from ), entry.weights, entry.bias ?? null, {
				width: step.width,
				height: step.height,
				inChannels: step.inChannels,
				outChannels: step.outChannels,
				kernelSize,
				padding,
				activation: step.activation,
				leakySlope
			} ) );

		} else if ( step.op === 'restrict' ) {

			buffers.set( step.to, restrict2Reference( require( step.from ), {
				width: step.width,
				height: step.height,
				channels: step.channels
			} ) );

		} else if ( step.op === 'upsample' ) {

			buffers.set( step.to, upsample2Reference( require( step.from ), {
				coarseWidth: step.coarseWidth,
				coarseHeight: step.coarseHeight,
				channels: step.channels,
				addTo: step.additive ? require( step.to ) : null
			} ) );

		} else if ( step.op === 'shuffleResidual' ) {

			buffers.set( step.to, shuffleResidualReference( require( step.from ), require( step.base ), {
				lowWidth: step.lowWidth,
				lowHeight: step.lowHeight,
				factor: step.factor,
				base: step.baseKind
			} ) );

		} else {

			throw new Error( `ml: unknown plan op '${ step.op }'.` );

		}

	}

	return buffers;

}

// *** Classical resampling, mirroring src/grid/grid_math.js exactly ***
//
// These are the *baseline* the super-resolution network has to beat, and
// also the base it adds its residual onto, so they have to agree with the
// GPU path element for element. They are transcriptions of
// `bilinearCoordsAndWeights2` and `collocatedCubicValueAtPosition2` /
// `monotonicCubic1d` / `cubicIndices1d`, clamping included -- the GPU side
// calls those functions directly rather than reimplementing them, so this
// is the only copy, and test/ml.test.js pins the two together through the
// shared plan.

function clampIndex( i, n ) {

	return i < 0 ? 0 : ( i > n - 1 ? n - 1 : i );

}

/**
 * Bilinear sample of one channel of a feature map at continuous grid
 * coordinates. Indices clamp; the fractional weights come from the
 * unclamped position, matching `bilinearCoordsAndWeights2`.
 */
export function sampleBilinearReference( data, width, height, gx, gy, channel = 0 ) {

	const i0 = Math.floor( gx );
	const j0 = Math.floor( gy );
	const fx = gx - i0;
	const fy = gy - j0;

	const i0c = clampIndex( i0, width );
	const i1c = clampIndex( i0 + 1, width );
	const j0c = clampIndex( j0, height );
	const j1c = clampIndex( j0 + 1, height );

	const at = ( i, j ) => data[ featureIndex( width, height, i, j, channel ) ];

	return at( i0c, j0c ) * ( 1 - fx ) * ( 1 - fy )
		+ at( i1c, j0c ) * fx * ( 1 - fy )
		+ at( i0c, j1c ) * ( 1 - fx ) * fy
		+ at( i1c, j1c ) * fx * fy;

}

/**
 * The monotonicity-clamped cubic of `monotonicCubic1d`. The clamp is what
 * stops a cubic from overshooting into new extrema, which on a density
 * field means invented bright spots and negative densities -- the reason
 * this package uses a monotonic cubic rather than a plain Catmull-Rom.
 */
export function monotonicCubic1dReference( f0, f1, f2, f3, f ) {

	const D1 = f2 - f1;
	const rawD1 = ( f2 - f0 ) * 0.5;
	const rawD2 = ( f3 - f1 ) * 0.5;

	const isFlat = Math.abs( D1 ) < 1e-12;
	const d1 = ( isFlat || rawD1 * D1 < 0 ) ? 0 : rawD1;
	const d2 = ( isFlat || rawD2 * D1 < 0 ) ? 0 : rawD2;

	const a3 = d1 + d2 - 2 * D1;
	const a2 = 3 * D1 - 2 * d1 - d2;

	return a3 * f * f * f + a2 * f * f + d1 * f + f1;

}

export function sampleBicubicReference( data, width, height, gx, gy, channel = 0 ) {

	const indices = ( coord, n ) => {

		const i0 = Math.floor( coord );

		return {
			taps: [
				clampIndex( i0 - 1, n ),
				clampIndex( i0, n ),
				clampIndex( i0 + 1, n ),
				clampIndex( i0 + 2, n )
			],
			f: coord - i0
		};

	};

	const xi = indices( gx, width );
	const yi = indices( gy, height );

	const at = ( i, j ) => data[ featureIndex( width, height, i, j, channel ) ];

	const rows = yi.taps.map( ( j ) => monotonicCubic1dReference(
		at( xi.taps[ 0 ], j ), at( xi.taps[ 1 ], j ), at( xi.taps[ 2 ], j ), at( xi.taps[ 3 ], j ), xi.f
	) );

	return monotonicCubic1dReference( rows[ 0 ], rows[ 1 ], rows[ 2 ], rows[ 3 ], yi.f );

}

/**
 * The high-resolution pixel's position in *low-resolution grid
 * coordinates*, for a cell-centred collocated field.
 *
 * Low-res cell centres sit at `i + 0.5` in low-res units; high-res centres
 * at `(X + 0.5) / factor`. So the sample position is
 * `(X + 0.5) / factor - 0.5`. Getting this wrong by the half-cell shifts the
 * whole image by half a low-res cell, which at 4x is two high-res pixels --
 * visible, and easy to mistake for the network having learned an offset.
 */
export function lowResCoordinate( highIndex, factor ) {

	return ( highIndex + 0.5 ) / factor - 0.5;

}

/**
 * Sub-pixel rearrange plus the classical base, in float64.
 *
 * `shuffleSource` is `[lowWidth, lowHeight, factor^2]`; output is
 * `[lowWidth * factor, lowHeight * factor, 1]`.
 */
export function shuffleResidualReference( shuffleSource, baseData, options ) {

	const { lowWidth, lowHeight, factor, base = 'bicubic', baseChannels = 1 } = options;

	const highWidth = lowWidth * factor;
	const highHeight = lowHeight * factor;
	const output = new Float64Array( highWidth * highHeight );

	for ( let Y = 0; Y < highHeight; Y ++ ) {

		for ( let X = 0; X < highWidth; X ++ ) {

			const lowX = Math.floor( X / factor );
			const lowY = Math.floor( Y / factor );
			const sub = ( X - lowX * factor ) + factor * ( Y - lowY * factor );

			const residual = shuffleSource[ featureIndex( lowWidth, lowHeight, lowX, lowY, sub ) ];

			let baseValue = 0;

			if ( base !== 'none' ) {

				const gx = lowResCoordinate( X, factor );
				const gy = lowResCoordinate( Y, factor );
				const sample = base === 'bicubic' ? sampleBicubicReference : sampleBilinearReference;
				baseValue = sample( baseData, lowWidth, lowHeight, gx, gy, 0 );

			}

			output[ featureIndex( highWidth, highHeight, X, Y, 0 ) ] = baseValue + residual;

		}

	}

	void baseChannels;

	return output;

}
