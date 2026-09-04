import * as tsl_array_n from 'tsl_array_n';
import { float } from 'three/tsl';

const statusEl = document.querySelector( '#status' );
const canvas = document.querySelector( '#out' );

function status( text, ok ) {

	statusEl.textContent = text;
	statusEl.className = ok === undefined ? '' : ( ok ? 'ok' : 'err' );

}

try {

	// offscreen canvas for the renderer — this demo draws its own output via 2D canvas,
	// it doesn't need the WebGPU canvas visible.
	const renderer = await tsl_array_n.init( { canvas: document.createElement( 'canvas' ), allowFallback: true } );

	const width = canvas.width;
	const height = canvas.height;
	const maxIter = 100;

	const iterCounts = tsl_array_n.array2( 'float', width, height );

	// c: the Julia constant, animated each frame — a 0-D field (array0), tsl_array_n's
	// equivalent of ti.field(dtype, shape=()) / a "uniform" for this purpose.
	//
	// KNOWN SANDBOX-ONLY ISSUE: on this dev sandbox's WebGL2-fallback backend, many
	// threads concurrently reading the same array0 element inside a Loop() produces wrong
	// results (isolated via a minimal repro: identical multi-thread kernel gives correct,
	// varied output when c is baked in as JS constants, but all-zero when read from an
	// array0 — a single-thread read of the same array0 is fine on its own). Same category
	// as the earlier Loop()-on-fallback finding that turned out not to reproduce on real
	// WebGPU — needs the same real-hardware check before trusting this animation.
	const c = tsl_array_n.array0( 'vec2' );

	const juliaKernel = tsl_array_n.kernel( iterCounts.shape, ( x, y ) => {

		// map pixel -> complex plane [-1.5, 1.5]
		const zx = x.toFloat().div( width ).mul( 3 ).sub( 1.5 ).toVar();
		const zy = y.toFloat().div( height ).mul( 3 ).sub( 1.5 ).toVar();
		const count = float( 0 ).toVar();
		const cVal = c(); // read once per invocation — re-read fresh on every dispatch of this compiled kernel

		tsl_array_n.Loop( maxIter, () => {

			tsl_array_n.If( zx.mul( zx ).add( zy.mul( zy ) ).greaterThan( 4 ), () => {

				tsl_array_n.Break();

			} );

			const xTemp = zx.mul( zx ).sub( zy.mul( zy ) ).add( cVal.x );
			zy.assign( zx.mul( zy ).mul( 2 ).add( cVal.y ) );
			zx.assign( xTemp );
			count.addAssign( 1 );

		} );

		iterCounts( x, y ).assign( count.div( maxIter ) );

	} );

	// cosine palette (Inigo Quilez style); points that never escaped -> black.
	function palette( t ) {

		const r = 0.5 + 0.5 * Math.cos( 6.28318 * ( t + 0.0 ) );
		const g = 0.5 + 0.5 * Math.cos( 6.28318 * ( t + 0.33 ) );
		const b = 0.5 + 0.5 * Math.cos( 6.28318 * ( t + 0.67 ) );
		return [ r * 255, g * 255, b * 255 ];

	}

	const ctx = canvas.getContext( '2d' );
	const image = ctx.createImageData( width, height );

	function draw( data ) {

		for ( let y = 0; y < height; y ++ ) {

			for ( let x = 0; x < width; x ++ ) {

				const t = data[ x + y * width ]; // matches array2's x + y*width flat layout
				const pixel = ( y * width + x ) * 4; // canvas ImageData is also row-major

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

	// classic animation trick: walk c around a circle in the complex plane —
	// R ~= 0.7885 sweeps through a rich variety of Julia set shapes.
	const R = 0.7885;
	const cycleMs = 20000;
	const start = performance.now();
	let frame = 0;

	// A manual, self-scheduling loop rather than renderer.setAnimationLoop(): each cycle
	// (set c -> dispatch -> readback -> draw) must fully finish, including the async
	// toArray() readback, before the next one starts — otherwise frames race ahead of their
	// own readback and overwrite c before it's been read back.
	async function animate() {

		const theta = ( ( performance.now() - start ) % cycleMs ) / cycleMs * Math.PI * 2;

		c.fromArray( [ R * Math.cos( theta ), R * Math.sin( theta ) ] );
		juliaKernel();

		const data = await iterCounts.toArray();
		draw( data );

		frame ++;
		status( `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' } — frame ${ frame }, θ=${ theta.toFixed( 2 ) }`, true );

		requestAnimationFrame( animate );

	}

	requestAnimationFrame( animate );

} catch ( error ) {

	status( error.message, false );
	console.error( error );

}
