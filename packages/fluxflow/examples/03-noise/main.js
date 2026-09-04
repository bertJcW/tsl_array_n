// Visual sanity check for src/noise/noise.js, not a numeric correctness
// proof -- noise functions don't have a simple hand-computable expected
// value the way e.g. a linear field's gradient does, so the practical way
// to catch a translation mistake here is to look at the output: Perlin and
// Simplex noise should both look like smooth, blobby, continuous noise
// (Simplex with less axis-aligned bias than Perlin), and cellular noise
// should look like a Voronoi cell pattern (flat-ish regions meeting at
// sharp ridges). Structural/no-throw coverage lives in test/noise.test.js.
//
// Unlike examples/00-grid-math/, none of these kernels read a different,
// already-populated field -- each thread only reads its own (x,y) dispatch
// index and writes its own output pixel, which is exactly the pattern that
// already works fine even on this dev sandbox's WebGL2 fallback (see
// README's "Verified on real WebGPU" section) -- so, unlike 00-grid-math,
// this one is expected to render correctly here too, not just on real
// WebGPU.

import * as tsl_array_n from 'tsl_array_n';
import { noise } from 'fluxflow';
import { vec3, int } from 'three/tsl';

const statusEl = document.getElementById( 'status' );

function setStatus( html ) {

	statusEl.innerHTML = html;

}

const SIZE = 160;
const SCALE = 1 / 20; // world units per pixel -- a few noise "cells" across the image

function drawGrayscale( canvas, data, mapToByte ) {

	const ctx = canvas.getContext( '2d' );
	const image = ctx.createImageData( SIZE, SIZE );

	for ( let j = 0; j < SIZE; j ++ ) {

		for ( let i = 0; i < SIZE; i ++ ) {

			const value = mapToByte( data[ i + j * SIZE ] );
			const idx = ( i + j * SIZE ) * 4;

			image.data[ idx ] = value;
			image.data[ idx + 1 ] = value;
			image.data[ idx + 2 ] = value;
			image.data[ idx + 3 ] = 255;

		}

	}

	ctx.putImageData( image, 0, 0 );

}

try {

	const renderer = await tsl_array_n.init( { allowFallback: true, canvas: document.createElement( 'canvas' ) } );
	setStatus( `<span class="ok">✓ init() — backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }</span>\nrunning…` );

	const perlinOut = tsl_array_n.array2( 'float', SIZE, SIZE );
	const perlinKernel = tsl_array_n.kernel( perlinOut.shape, ( x, y ) => {

		const pos = vec3( x.toFloat().mul( SCALE ), y.toFloat().mul( SCALE ), 0 );
		perlinOut( x, y ).assign( noise.perlinNoise3d( pos ) );

	} );
	perlinKernel();

	const simplexOut = tsl_array_n.array2( 'float', SIZE, SIZE );
	const simplexKernel = tsl_array_n.kernel( simplexOut.shape, ( x, y ) => {

		const pos = vec3( x.toFloat().mul( SCALE ), y.toFloat().mul( SCALE ), int( 5 ).toFloat() ); // different z-slice than perlin, just so the two aren't the exact same slice
		simplexOut( x, y ).assign( noise.simplexNoise3d( pos ) );

	} );
	simplexKernel();

	const cellularOut = tsl_array_n.array2( 'float', SIZE, SIZE );
	const cellularKernel = tsl_array_n.kernel( cellularOut.shape, ( x, y ) => {

		const pos = vec3( x.toFloat().mul( SCALE ), y.toFloat().mul( SCALE ), 0 );
		cellularOut( x, y ).assign( noise.cellular3d( pos ).x ); // F1 only

	} );
	cellularKernel();

	const [ perlinData, simplexData, cellularData ] = await Promise.all( [
		perlinOut.toArray(),
		simplexOut.toArray(),
		cellularOut.toArray()
	] );

	// Perlin/Simplex are roughly in [-1, 1]; map to [0, 255]
	drawGrayscale( document.getElementById( 'perlin' ), perlinData, ( v ) => Math.max( 0, Math.min( 255, Math.round( ( v * 0.5 + 0.5 ) * 255 ) ) ) );
	drawGrayscale( document.getElementById( 'simplex' ), simplexData, ( v ) => Math.max( 0, Math.min( 255, Math.round( ( v * 0.5 + 0.5 ) * 255 ) ) ) );

	// cellular3d's F1 is a small positive distance (jitter=1, so typically
	// well under ~1 in these grid units); scale up so the pattern is visible
	drawGrayscale( document.getElementById( 'cellular' ), cellularData, ( v ) => Math.max( 0, Math.min( 255, Math.round( v * 255 ) ) ) );

	const allFinite = [ perlinData, simplexData, cellularData ].every( ( arr ) => Array.from( arr ).every( ( v ) => Number.isFinite( v ) ) );

	setStatus(
		`<span class="ok">✓ init() — backend: ${ renderer.backend?.constructor?.name }</span>\n` +
		`<span class="${ allFinite ? 'ok' : 'err' }">${ allFinite ? '✓' : '✗' } all three noise fields dispatched and read back ${ allFinite ? 'finite values' : 'some non-finite (NaN/Infinity) values' }</span>\n` +
		'Check the three images above look like noise, not a numeric proof — see this file\'s header comment.'
	);

} catch ( error ) {

	setStatus( `<span class="err">✗ failed — ${ error.message }</span>` );
	console.error( error );

}
