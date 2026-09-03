import * as tslify from 'tslify';
import { float } from 'three/tsl';

const statusEl = document.querySelector( '#status' );
const canvas = document.querySelector( '#out' );

function status( text, ok ) {

	statusEl.textContent = text;
	statusEl.className = ok === undefined ? '' : ( ok ? 'ok' : 'err' );

}

try {

	// offscreen canvas for the renderer — this demo draws its own output via 2D canvas.
	const renderer = await tslify.init( { canvas: document.createElement( 'canvas' ), allowFallback: true } );

	const width = canvas.width;
	const height = canvas.height;
	const maxIter = 120;

	const iterCounts = tslify.array2( 'float', width, height );

	// Same idea as examples/04-julia (an animated Julia constant), but this time via
	// tslify.uniform() instead of array0() — two real WebGPU uniforms, GLSL-style. No
	// callable-element step needed: cReal/cImag ARE the node, used directly in expressions.
	// Updated via .value = x (not .fromArray()).
	const cReal = tslify.uniform( -0.4 );
	const cImag = tslify.uniform( 0.6 );

	const juliaKernel = tslify.kernel( iterCounts.shape, ( x, y ) => {

		// map pixel -> complex plane [-1.5, 1.5]
		const zx = x.toFloat().div( width ).mul( 3 ).sub( 1.5 ).toVar();
		const zy = y.toFloat().div( height ).mul( 3 ).sub( 1.5 ).toVar();
		const count = float( 0 ).toVar();

		tslify.Loop( maxIter, () => {

			tslify.If( zx.mul( zx ).add( zy.mul( zy ) ).greaterThan( 4 ), () => {

				tslify.Break();

			} );

			const xTemp = zx.mul( zx ).sub( zy.mul( zy ) ).add( cReal );
			zy.assign( zx.mul( zy ).mul( 2 ).add( cImag ) );
			zx.assign( xTemp );
			count.addAssign( 1 );

		} );

		iterCounts( x, y ).assign( count.div( maxIter ) );

	} );

	// a different palette mood (cooler, more phase-shifted) from 04-julia's.
	function palette( t ) {

		const r = 0.5 + 0.5 * Math.cos( 6.28318 * ( t + 0.55 ) );
		const g = 0.5 + 0.5 * Math.cos( 6.28318 * ( t + 0.15 ) );
		const b = 0.5 + 0.5 * Math.cos( 6.28318 * ( t + 0.85 ) );
		return [ r * 255, g * 255, b * 255 ];

	}

	const ctx = canvas.getContext( '2d' );
	const image = ctx.createImageData( width, height );

	function draw( data ) {

		for ( let y = 0; y < height; y ++ ) {

			for ( let x = 0; x < width; x ++ ) {

				const t = data[ x + y * width ];
				const pixel = ( y * width + x ) * 4;

				if ( t >= 0.999 ) {

					image.data[ pixel ] = 0;
					image.data[ pixel + 1 ] = 0;
					image.data[ pixel + 2 ] = 0;

				} else {

					const [ r, g, b ] = palette( t );
					image.data[ pixel ] = r;
					image.data[ pixel + 1 ] = g;
					image.data[ pixel + 2 ] = b;

				}

				image.data[ pixel + 3 ] = 255;

			}

		}

		ctx.putImageData( image, 0, 0 );

	}

	// Different parameters/path from 04-julia's plain circle: a Lissajous curve
	// (1:2 frequency ratio) offset from the origin, sweeping through a region of
	// parameter space near the classic "dendrite" constant (-0.4, 0.6) instead of
	// orbiting the origin at fixed radius 0.7885.
	const ampReal = 0.35;
	const ampImag = 0.35;
	const centerReal = -0.4;
	const centerImag = 0.3;
	const cycleMs = 16000;
	const start = performance.now();
	let frame = 0;

	async function animate() {

		const t = ( ( performance.now() - start ) % cycleMs ) / cycleMs * Math.PI * 2;

		cReal.value = centerReal + ampReal * Math.cos( t );
		cImag.value = centerImag + ampImag * Math.sin( 2 * t );

		juliaKernel();

		const data = await iterCounts.toArray();
		draw( data );

		frame ++;
		status(
			`backend: ${ renderer.backend?.constructor?.name ?? 'unknown' } — frame ${ frame }, c=(${ cReal.value.toFixed( 2 ) }, ${ cImag.value.toFixed( 2 ) })`,
			true
		);

		requestAnimationFrame( animate );

	}

	requestAnimationFrame( animate );

} catch ( error ) {

	status( error.message, false );
	console.error( error );

}
