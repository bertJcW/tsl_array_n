// Verifies grid_math.js's collocatedCubicValueAtPosition2 (monotonic
// bicubic, Fedkiw et al.'s clamped-Catmull-Rom scheme) against an
// independent plain-JS reference implementation of the exact same
// formula, at several non-grid-aligned query positions on a small 8x8
// scalar field -- including one position exactly on a grid point (should
// match the source data exactly) and one out of bounds (should sample
// the clamped boundary value, not extrapolate or error).
//
// Known limitation (sandbox environment, not this code): this reads back
// wrong (effectively zero) results in this dev sandbox. Isolated by
// checking this exact sandbox's *current* behavior for
// collocatedValueAtPosition2 (the plain bilinear sibling of the function
// under test here, in the same grid_math.js) via examples/00-grid-math/ --
// it fails identically right now (`got [0, 0], expected [9, 2.5]`), even
// though that specific function was already confirmed correct on real
// WebGPU hardware early in this project. Both functions share the same
// shape (a kernel reading a field that was populated by an earlier,
// separate fromArray() call), so this is almost certainly the same
// well-established "cross-field read returns 0" WebGL2-fallback
// limitation already confirmed fallback-only for this exact sibling
// function, not a new bug in the cubic code specifically -- but that's
// still an inference from a closely related case, not a direct
// confirmation of *this* function. Needs a real WebGPU run to actually
// confirm.

import * as tsl_array_n from 'tsl_array_n';
import { grid } from 'fluxflow';

const pre = document.querySelector( '#status pre' );
const lines = [];

function log( label, ok, detail ) {

	const cls = ok ? 'ok' : 'err';
	const mark = ok ? '✓' : '✗';
	lines.push( `<span class="${ cls }">${ mark } ${ label }${ detail ? ' — ' + detail : '' }</span>` );
	pre.innerHTML = lines.join( '\n' );

}

const N = 8;
const gridSpacing = [ 1, 1 ];
const dataOrigin = [ 0, 0 ];

function jsCubicIndices1d( coord, n ) {

	const i0 = Math.floor( coord );
	const f = coord - i0;
	const clamp = ( x ) => Math.max( 0, Math.min( x, n - 1 ) );
	return { im1: clamp( i0 - 1 ), i0c: clamp( i0 ), i1c: clamp( i0 + 1 ), i2c: clamp( i0 + 2 ), f };

}

function jsMonotonicCubic1d( f0, f1, f2, f3, f ) {

	let d1 = ( f2 - f0 ) / 2;
	let d2 = ( f3 - f1 ) / 2;
	const D1 = f2 - f1;

	if ( Math.abs( D1 ) < 1e-12 ) { d1 = 0; d2 = 0; }
	if ( Math.sign( D1 ) !== Math.sign( d1 ) ) d1 = 0;
	if ( Math.sign( D1 ) !== Math.sign( d2 ) ) d2 = 0;

	const a3 = d1 + d2 - 2 * D1;
	const a2 = 3 * D1 - 2 * d1 - d2;
	const a1 = d1;
	const a0 = f1;

	return a3 * f * f * f + a2 * f * f + a1 * f + a0;

}

function jsCollocatedCubicValueAtPosition2( dataArr, nx, ny, pos ) {

	const gx = ( pos[ 0 ] - dataOrigin[ 0 ] ) / gridSpacing[ 0 ];
	const gy = ( pos[ 1 ] - dataOrigin[ 1 ] ) / gridSpacing[ 1 ];

	const xi = jsCubicIndices1d( gx, nx );
	const yi = jsCubicIndices1d( gy, ny );

	const jIndices = [ yi.im1, yi.i0c, yi.i1c, yi.i2c ];
	const rowValues = jIndices.map( ( j ) => jsMonotonicCubic1d(
		dataArr[ xi.im1 + nx * j ], dataArr[ xi.i0c + nx * j ], dataArr[ xi.i1c + nx * j ], dataArr[ xi.i2c + nx * j ], xi.f
	) );

	return jsMonotonicCubic1d( rowValues[ 0 ], rowValues[ 1 ], rowValues[ 2 ], rowValues[ 3 ], yi.f );

}

try {

	const renderer = await tsl_array_n.init( { allowFallback: true } );
	log( 'init()', true, `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }` );

	const scalarGrid = grid.createScalarGrid2( N, N, gridSpacing[ 0 ], gridSpacing[ 1 ], dataOrigin[ 0 ], dataOrigin[ 1 ] );

	// f(i,j) = sin(i*0.7) * cos(j*0.5) -- smooth but genuinely non-linear,
	// tsl_array_n's flat layout is i + N*j (i fastest-varying).
	const fieldArray = new Float32Array( N * N );
	for ( let j = 0; j < N; j ++ ) {

		for ( let i = 0; i < N; i ++ ) {

			fieldArray[ i + N * j ] = Math.sin( i * 0.7 ) * Math.cos( j * 0.5 );

		}

	}

	scalarGrid.data.fromArray( fieldArray );

	const queryPositions = [
		[ 2.3, 3.7 ],
		[ 0.5, 0.5 ],
		[ 6.9, 1.2 ],
		[ 4.0, 4.0 ], // exactly on a grid point -- should match fieldArray[4+N*4] exactly
		[ - 1.0, 3.0 ], // out of bounds -- should clamp, not extrapolate/error
	];

	const positionsField = tsl_array_n.arrayN( 'vec2', queryPositions.length );
	positionsField.fromArray( new Float32Array( queryPositions.flat() ) );

	const resultsField = tsl_array_n.arrayN( 'float', queryPositions.length );

	const dispatch = tsl_array_n.kernel( queryPositions.length, ( q ) => {

		resultsField( q ).assign(
			grid.collocatedCubicValueAtPosition2( scalarGrid.data, scalarGrid.gridSpacing, scalarGrid.dataOrigin, positionsField( q ), scalarGrid.resolution )
		);

	} );

	dispatch();

	const results = Array.from( await resultsField.toArray() );
	const expected = queryPositions.map( ( pos ) => jsCollocatedCubicValueAtPosition2( fieldArray, N, N, pos ) );

	const matches = results.length === expected.length && results.every( ( v, i ) => Math.abs( v - expected[ i ] ) < 1e-4 );

	log(
		'collocatedCubicValueAtPosition2 vs. independent JS reference',
		matches,
		matches
			? `all ${ results.length } query points matched (max |diff| < 1e-4)`
			: `results=[${ results.map( ( v ) => v.toFixed( 4 ) ) }], expected=[${ expected.map( ( v ) => v.toFixed( 4 ) ) }]`
	);

	// exact-grid-point check on its own line, since it's the strongest/most
	// legible single assertion (no interpolation ambiguity at all)
	const exactIdx = 3; // [4.0, 4.0]
	const exactExpected = fieldArray[ 4 + N * 4 ];
	log(
		'exact grid-point query [4,4] matches the source data directly',
		Math.abs( results[ exactIdx ] - exactExpected ) < 1e-4,
		`got ${ results[ exactIdx ]?.toFixed?.( 6 ) }, source data = ${ exactExpected.toFixed( 6 ) }`
	);

} catch ( error ) {

	log( 'failed', false, error.message );
	console.error( error );

}
