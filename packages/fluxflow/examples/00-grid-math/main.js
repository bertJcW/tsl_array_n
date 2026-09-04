// CONFIRMED on real WebGPU: all three tests below run correctly on a real
// WebGPUBackend (init() reports "backend: WebGPUBackend"). In the dev
// sandbox (no real WebGPU adapter, init() falls back to WebGLBackend), the
// first two tests read back correct values but the third used to read back
// zeros -- isolated at the time to "reading a different field that already
// has data, from inside a kernel" not working on that fallback backend
// (regardless of whether the other field's data came from fromArray() or
// another kernel). Real hardware confirms this was indeed the fourth
// instance of a fallback-only limitation tsl_array_n has hit (after the
// Loop() counter, array0 multi-thread shared-read, and this port's own
// GPU-round-trip self-touch case in examples/02-flow-around-shape/), not a
// bug in this port's code.
//
// The mat2 ordering question this third test was actually designed to
// answer turned out to be a real bug, not a fallback artifact: it read back
// J.(1,0)=(2,5), J.(0,1)=(0,0) instead of the expected (2,0) and (5,0) --
// confirming TSL's mat2() fills column-major where the direct translation
// of Taichi's source assumed row-major. Fixed in grid_math.js's
// vectorGradient2 (see the comment there); this test is now green.
import * as tsl_array_n from 'tsl_array_n';
import { grid } from 'fluxflow';
import { vec2, int } from 'three/tsl';

const pre = document.querySelector( '#status pre' );
const lines = [];

function log( label, ok, detail ) {

	const cls = ok ? 'ok' : 'err';
	const mark = ok ? '✓' : '✗';
	lines.push( `<span class="${ cls }">${ mark } ${ label }${ detail ? ' — ' + detail : '' }</span>` );
	pre.innerHTML = lines.join( '\n' );

}

function approxEqual( a, b, eps = 1e-4 ) {

	return Math.abs( a - b ) < eps;

}

function fillScalar4x4( field, valueAt ) {

	const flat = new Float32Array( 16 );
	for ( let j = 0; j < 4; j ++ ) for ( let i = 0; i < 4; i ++ ) flat[ i + j * 4 ] = valueAt( i, j );
	field.fromArray( flat );

}

function fillVec2_4x4( field, valueAt ) {

	const flat = new Float32Array( 32 );
	for ( let j = 0; j < 4; j ++ ) for ( let i = 0; i < 4; i ++ ) {

		const [ x, y ] = valueAt( i, j );
		flat[ ( i + j * 4 ) * 2 ] = x;
		flat[ ( i + j * 4 ) * 2 + 1 ] = y;

	}

	field.fromArray( flat );

}

try {

	const renderer = await tsl_array_n.init( { allowFallback: true } );
	log( 'init()', true, `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }` );

	// ------------------------------------------------------------
	// 1. collocatedValueAtPosition2: bilinear interpolation -- at a grid
	// point it should exactly equal that point's data; at a half-cell
	// midpoint it should equal the average of the 4 surrounding corners
	{

		const scalarGrid = grid.createScalarGrid2( 4, 4, 1, 1, 0, 0 );
		fillScalar4x4( scalarGrid.data, ( i, j ) => i + j * 4 ); // data[i,j] = i + 4j, matches tsl_array_n's own .at() indexing convention

		const out = tsl_array_n.arrayN( 'float', 2 );

		const run = tsl_array_n.kernel( 1, ( _idx ) => {

			// sampled at grid point (1,2): should exactly equal data[1,2] = 1 + 2*4 = 9
			out( 0 ).assign( grid.collocatedValueAtPosition2( scalarGrid.data, scalarGrid.gridSpacing, scalarGrid.dataOrigin, vec2( 1, 2 ), scalarGrid.dataSize ) );

			// (0.5, 0.5) half-cell midpoint: should equal the average of data[0,0],data[1,0],data[0,1],data[1,1] = (0+1+4+5)/4 = 2.5
			out( 1 ).assign( grid.collocatedValueAtPosition2( scalarGrid.data, scalarGrid.gridSpacing, scalarGrid.dataOrigin, vec2( 0.5, 0.5 ), scalarGrid.dataSize ) );

		} );

		run();

		const [ atGridPoint, atMidpoint ] = await out.toArray();
		const ok = approxEqual( atGridPoint, 9 ) && approxEqual( atMidpoint, 2.5 );

		log(
			'collocatedValueAtPosition2 bilinear interpolation',
			ok,
			ok ? `grid point (1,2)=${ atGridPoint }, midpoint (0.5,0.5)=${ atMidpoint }` : `got [${ atGridPoint }, ${ atMidpoint }], expected [9, 2.5]`
		);

	}

	// ------------------------------------------------------------
	// 2. scalarGradient2: at an interior grid point of the linear field
	// f(i,j) = 2i + 3j, a central difference should give the exact analytic
	// gradient (2, 3) (a linear function's central difference has no
	// truncation error)
	{

		const linearGrid = grid.createScalarGrid2( 4, 4, 1, 1, 0, 0 );
		fillScalar4x4( linearGrid.data, ( i, j ) => 2 * i + 3 * j );

		const out = tsl_array_n.arrayN( 'vec2', 1 );

		const run = tsl_array_n.kernel( 1, ( _idx ) => {

			out( 0 ).assign( grid.scalarGradient2( linearGrid.data, linearGrid.gridSpacing, int( 1 ), int( 1 ), linearGrid.dataSize ) );

		} );

		run();

		const [ gx, gy ] = await out.toArray();
		const ok = approxEqual( gx, 2 ) && approxEqual( gy, 3 );

		log(
			'scalarGradient2 at an interior point of a linear field',
			ok,
			ok ? `∇f(1,1) = (${ gx }, ${ gy })` : `got (${ gx }, ${ gy }), expected (2, 3)`
		);

	}

	// ------------------------------------------------------------
	// 3. vectorGradient2 / mat2 element order -- this is the unverified
	// point flagged in grid_math.js's header comment: vector field
	// f(i,j) = (2i + 5j, 0), analytic Jacobian J = [[2,5],[0,0]] (rows:
	// dfx/dx,dfx/dy; dfy/dx,dfy/dy). J.(1,0) should equal the x-direction
	// partials = (dfx/dx, dfy/dx) = (2,0); J.(0,1) should equal the
	// y-direction partials = (dfx/dy, dfy/dy) = (5,0). Using an asymmetric
	// field is the key part -- a symmetric field can't reveal a transpose.
	{

		const vectorGrid = grid.createCollocatedVectorGrid2( 4, 4, 1, 1, 0, 0 );
		fillVec2_4x4( vectorGrid.data, ( i, j ) => [ 2 * i + 5 * j, 0 ] );

		const out = tsl_array_n.arrayN( 'vec2', 2 );

		const run = tsl_array_n.kernel( 1, ( _idx ) => {

			const jacobian = grid.vectorGradient2( vectorGrid.data, vectorGrid.gridSpacing, int( 1 ), int( 1 ), vectorGrid.dataSize );
			out( 0 ).assign( jacobian.mul( vec2( 1, 0 ) ) );
			out( 1 ).assign( jacobian.mul( vec2( 0, 1 ) ) );

		} );

		run();

		const [ jx0, jy0, jx1, jy1 ] = await out.toArray();
		const ok = approxEqual( jx0, 2 ) && approxEqual( jy0, 0 ) && approxEqual( jx1, 5 ) && approxEqual( jy1, 0 );

		log(
			'vectorGradient2 mat2 ordering (row-major vs column-major)',
			ok,
			ok
				? `J·(1,0)=(${ jx0 },${ jy0 }), J·(0,1)=(${ jx1 },${ jy1 }) — matches the intended row-major Jacobian`
				: `got J·(1,0)=(${ jx0 },${ jy0 }), J·(0,1)=(${ jx1 },${ jy1 }); expected (2,0) and (5,0) — mat2() fill order needs fixing in grid_math.js`
		);

	}

} catch ( error ) {

	log( 'failed', false, error.message );

}
