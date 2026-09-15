// *** Simulate coarse, render fine. ***
//
// The same 96x128 smoke scene as examples/17-smoke-fire/, displayed four
// ways at 384x512, so the gap this direction is trying to close is visible
// side by side rather than argued about:
//
//   1. **nearest** -- a 96x128 canvas scaled by CSS with
//      `image-rendering: pixelated`. This is not a straw man: it is exactly
//      what every drawing example in this package ships today.
//   2. **bilinear** -- one GPU dispatch, `grid_math.js`'s own sampler.
//   3. **monotonic bicubic** -- one GPU dispatch, same file. The real
//      baseline any learned upsampler has to beat.
//   4. **the network** -- `ml.createSuperResolver2`, residual on top of
//      bicubic, four dispatches of convolution at *low* resolution plus one
//      fused sub-pixel rearrange.
//
// *** Panel 4 is identical to panel 3, and that is the point today ***
//
// The network's weights are all zero, and its residual is added to a
// bicubic base, so it computes bicubic exactly. Panel 4 is therefore a
// **correctness check on the GPU pipeline**, not a demonstration of
// super-resolution: it exercises the trunk convolutions, the factor^2 head,
// the sub-pixel rearrange's index arithmetic, the half-cell sample shift
// and the residual add, and if any of those is wrong the two panels stop
// matching. The page reports the max absolute difference every frame, and
// anything above float32 rounding is a bug.
//
// Nothing here is trained. `docs/machine-learning-fluid-research.md`'s
// family C is where the training question lives, and `superres.js`'s header
// records the one thing that will have to be solved before a trained
// version of this is usable: without a temporal term it will flicker, which
// is tempoGAN's central finding, and the fix (advect the previous
// high-resolution frame by the velocity field and feed it in as an extra
// input channel) is designed for but not built.
//
// *** What the timings do and do not mean ***
//
// Each arm's cost below **includes a GPU-to-CPU readback of the
// high-resolution field**, because that is how every example in this
// package draws: read back, fill an ImageData, putImageData. A real
// renderer would keep the high-resolution field on the GPU and sample it as
// a texture, with no readback at all. So the per-arm numbers are the right
// way to compare the four arms against each other on this page, and the
// wrong way to estimate what this would cost in a real application --
// where the readback, not the network, is most of what is measured here.
// At 384x512 a readback is 786 KB, against 86.7 MMAC for the network.

import * as tsl_array_n from 'tsl_array_n';
import { float, max } from 'three/tsl';
import { grid, ml } from 'fluxflow';

const statusEl = document.querySelector( '#status' );
const perfEl = document.querySelector( '#perf' );

function status( text, isError = false ) {

	statusEl.innerHTML = isError ? `<span class="err">${ text }</span>` : text;

}

const NX = 96;
const NY = 128;
const FACTOR = 4;
const HX = NX * FACTOR;
const HY = NY * FACTOR;
const dt = 1 / 30;
const sourceX = NX / 2;
const sourceY = 6;
const sourceRadius = 6;
const sourceDensity = 1;
const sourceTemperature = 4;
const DRAW_INTERVAL = 2;

const OUTER_MARGIN = 1000;

function makeTopWallStripPolygon( innerY, outerY ) {

	return [ [ - OUTER_MARGIN, innerY ], [ NX + OUTER_MARGIN, innerY ], [ NX + OUTER_MARGIN, outerY ], [ - OUTER_MARGIN, outerY ] ];

}

try {

	const renderer = await tsl_array_n.init( { canvas: document.createElement( 'canvas' ), allowFallback: true } );
	const backend = renderer.backend?.constructor?.name ?? 'unknown';
	status( `backend: ${ backend }` );

	// ---------------------------------------------------------------
	// The simulation: examples/17-smoke-fire/'s scene, unchanged
	// ---------------------------------------------------------------

	const velocityGrid = grid.createFaceCenteredGrid2( NX, NY, 1, 1, 0, 0 );

	const outflow = grid.createSDFOutflow2( NX, NY, 1, 1, 0, 0 );
	outflow.addPolygon( makeTopWallStripPolygon( NY - 2, NY + OUTER_MARGIN ) );

	const smoke = grid.createGridSmokeSolver2( {
		velocityGrid,
		gridSpacing: [ 1, 1 ],
		origin: [ 0, 0 ],
		outflows: outflow,
		closedDomainBoundaryFlag: grid.DIRECTION_ALL & ~grid.DIRECTION_UP,
		dt,
		pressure: { multigrid: { numberOfLevels: 4 }, tolerance: 1e-4, maxIterations: 60 }
	} );

	smoke.density.stateA.clear();
	smoke.density.stateB.clear();
	smoke.temperature.stateA.clear();
	smoke.temperature.stateB.clear();

	function createSourceKernel( stateGrid, sourceValue ) {

		return tsl_array_n.kernel( stateGrid.dataSize, ( i, j ) => {

			const pos = stateGrid.dataPosition( i, j );
			const dx = pos.x.sub( sourceX );
			const dy = pos.y.sub( sourceY );
			const inSource = dx.mul( dx ).add( dy.mul( dy ) ).lessThan( sourceRadius * sourceRadius );

			stateGrid.data( i, j ).assign( max( stateGrid.data( i, j ), inSource.select( float( sourceValue ), float( 0 ) ) ) );

		} );

	}

	const injectDensityA = createSourceKernel( smoke.density.stateA, sourceDensity );
	const injectDensityB = createSourceKernel( smoke.density.stateB, sourceDensity );
	const injectTemperatureA = createSourceKernel( smoke.temperature.stateA, sourceTemperature );
	const injectTemperatureB = createSourceKernel( smoke.temperature.stateB, sourceTemperature );

	// ---------------------------------------------------------------
	// The four display arms
	// ---------------------------------------------------------------

	const superres = ml.createSuperResolver2( {
		shape: [ NX, NY ],
		factor: FACTOR,
		channels: 16,
		layers: 3,
		inChannels: 1,
		baseKind: 'bicubic'
	} );

	// Zero weights, so the residual is a hard zero and this is bicubic
	// exactly. createSuperResolver2 already does this at construction; it is
	// repeated here because it is the whole premise of panel 4 and should
	// not be something a reader has to go and check.
	superres.zero();

	// Both classical arms read the *same* input feature map the network
	// does, so all three see identical data and a difference can only come
	// from the upsampling itself.
	const bilinear = ml.createClassicalUpsampler2( {
		shape: [ NX, NY ], factor: FACTOR, baseKind: 'bilinear', input: superres.input
	} );

	const bicubic = ml.createClassicalUpsampler2( {
		shape: [ NX, NY ], factor: FACTOR, baseKind: 'bicubic', input: superres.input
	} );

	// The density field flips between two state buffers every step, exactly
	// like the injection kernels above, so pack from whichever is current.
	const packA = ml.buildPackFieldKernel( smoke.density.stateA.data, superres.input, { width: NX, height: NY, label: 'ml-pack-A' } );
	const packB = ml.buildPackFieldKernel( smoke.density.stateB.data, superres.input, { width: NX, height: NY, label: 'ml-pack-B' } );

	status( `backend: ${ backend } — network: ${ superres.stats.dispatches } dispatches, `
		+ `${ superres.stats.parameters.toLocaleString() } parameters (${ ( superres.stats.bytes / 1024 ).toFixed( 1 ) } KiB), `
		+ `${ ( superres.stats.macs / 1e6 ).toFixed( 1 ) } MMAC/frame `
		+ `(${ ( superres.stats.macsIfUpsampledFirst / 1e6 ).toFixed( 0 ) } MMAC if it convolved at full resolution instead)` );

	// ---------------------------------------------------------------
	// Drawing
	// ---------------------------------------------------------------

	function makeDrawer( canvasId, width, height ) {

		const ctx = document.querySelector( canvasId ).getContext( '2d' );
		const image = ctx.createImageData( width, height );

		return function draw( data ) {

			for ( let j = 0; j < height; j ++ ) {

				for ( let i = 0; i < width; i ++ ) {

					// Canvas Y is down-positive, this grid's Y is up-positive.
					const pixel = ( ( height - 1 - j ) * width + i ) * 4;
					const v = data[ i + width * j ];
					const bright = 255 * Math.min( 1, Math.max( 0, v ) );

					image.data[ pixel ] = bright;
					image.data[ pixel + 1 ] = bright;
					image.data[ pixel + 2 ] = bright;
					image.data[ pixel + 3 ] = 255;

				}

			}

			ctx.putImageData( image, 0, 0 );

		};

	}

	const drawNearest = makeDrawer( '#outNearest', NX, NY );
	const drawBilinear = makeDrawer( '#outBilinear', HX, HY );
	const drawBicubic = makeDrawer( '#outBicubic', HX, HY );
	const drawNetwork = makeDrawer( '#outNetwork', HX, HY );

	// ---------------------------------------------------------------
	// The loop
	// ---------------------------------------------------------------

	const timings = { sim: [], bilinear: [], bicubic: [], network: [] };
	let worstDifference = 0;
	let frame = 0;
	let stopped = false;

	function record( key, ms ) {

		const list = timings[ key ];
		list.push( ms );
		if ( list.length > 30 ) list.shift();

	}

	function meanOf( key ) {

		const list = timings[ key ];
		return list.length === 0 ? 0 : list.reduce( ( a, b ) => a + b, 0 ) / list.length;

	}

	async function timed( key, run ) {

		const t0 = performance.now();
		const result = await run();
		record( key, performance.now() - t0 );
		return result;

	}

	async function animate() {

		const simStart = performance.now();
		await smoke.onAdvanceTimeStep();

		const usingA = smoke.density.current === smoke.density.stateA;

		if ( usingA ) { injectDensityA(); injectTemperatureA(); }
		else { injectDensityB(); injectTemperatureB(); }

		record( 'sim', performance.now() - simStart );

		if ( frame % DRAW_INTERVAL === 0 ) {

			// Pack the current density into the shared input feature map, once
			// for all three GPU arms.
			if ( usingA ) packA(); else packB();

			const lowData = await smoke.density.current.data.toArray();

			const bilinearData = await timed( 'bilinear', async () => {

				bilinear.upsample();
				return bilinear.output.toArray();

			} );

			const bicubicData = await timed( 'bicubic', async () => {

				bicubic.upsample();
				return bicubic.output.toArray();

			} );

			const networkData = await timed( 'network', async () => {

				superres.resolve();
				return superres.output.toArray();

			} );

			// *** The correctness check. *** Zero weights make the residual a
			// hard zero, so the network arm must reproduce the bicubic arm
			// exactly, up to float32 rounding in the trunk (which multiplies
			// by zero, so in practice exactly).
			if ( networkData.length === bicubicData.length && networkData.length > 0 ) {

				let worst = 0;
				for ( let i = 0; i < networkData.length; i ++ ) worst = Math.max( worst, Math.abs( networkData[ i ] - bicubicData[ i ] ) );
				worstDifference = Math.max( worstDifference, worst );

			}

			drawNearest( lowData );
			drawBilinear( bilinearData );
			drawBicubic( bicubicData );
			drawNetwork( networkData );

			const checkOk = worstDifference < 1e-6;

			perfEl.innerHTML =
				`sim step        ${ meanOf( 'sim' ).toFixed( 2 ) } ms   (96×128, MGPCG)\n`
				+ `bilinear  ↑4    ${ meanOf( 'bilinear' ).toFixed( 2 ) } ms   1 dispatch  + 384×512 readback\n`
				+ `bicubic   ↑4    ${ meanOf( 'bicubic' ).toFixed( 2 ) } ms   1 dispatch  + 384×512 readback\n`
				+ `network   ↑4    ${ meanOf( 'network' ).toFixed( 2 ) } ms   ${ superres.stats.dispatches } dispatches + 384×512 readback\n`
				+ `\n`
				+ `<span class="${ checkOk ? 'ok' : 'err' }">${ checkOk ? '✓' : '✗' } network ≡ bicubic   max |diff| = ${ worstDifference.toExponential( 2 ) }</span>`
				+ `${ checkOk ? '' : '   ← pipeline bug: zero weights must reproduce bicubic exactly' }`;

		}

		frame ++;

		if ( ! stopped ) requestAnimationFrame( animate );

	}

	requestAnimationFrame( animate );

	window.addEventListener( 'error', () => { stopped = true; } );

} catch ( error ) {

	status( error.message, true );
	console.error( error );

}
