// Verifies createExternalForceSolver2 (grid/external_force_solver2.js)
// two ways, both hand-verifiable exactly (no interpolation involved at
// all, unlike the advection examples):
//
// 1. A constant force (0,-1), gravity-like: on a zero-initialized 4x4
//    velocity grid with dt=0.1, every dataV sample should increase by
//    exactly -1*0.1=-0.1, every dataU sample should stay exactly 0.
// 2. A position-*dependent* custom force, (pos) => vec2(pos.x, 0), to
//    prove the general (pos)=>vec2 path works, not just the constant
//    case: dataU(i,j) should become exactly uPosition(i,j).x * dt (a
//    ramp in i, since uPosition's own x-component is just i for this
//    grid's spacing/origin), dataV should stay exactly 0 (force.y is
//    always 0 here).

import * as tsl_array_n from 'tsl_array_n';
import { vec2 } from 'three/tsl';
import { grid } from 'fluxflow';

const pre = document.querySelector( '#status pre' );
const lines = [];

function log( label, ok, detail ) {

	const cls = ok ? 'ok' : 'err';
	const mark = ok ? '✓' : '✗';
	lines.push( `<span class="${ cls }">${ mark } ${ label }${ detail ? ' — ' + detail : '' }</span>` );
	pre.innerHTML = lines.join( '\n' );

}

const N = 4;
const dt = 0.1;

function allClose( arr, expected, tol = 1e-4 ) {

	return arr.length === expected.length && arr.every( ( v, i ) => Math.abs( v - expected[ i ] ) < tol );

}

try {

	const renderer = await tsl_array_n.init( { allowFallback: true } );
	log( 'init()', true, `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }` );

	// ---- Test 1: constant force ----

	const velocityGrid1 = grid.createFaceCenteredGrid2( N, N, 1, 1, 0, 0 );
	const solver1 = grid.createExternalForceSolver2( {
		velocityGrid: velocityGrid1,
		force: () => vec2( 0, - 1 ),
		dt
	} );
	solver1.applyExternalForces();

	const u1 = Array.from( await velocityGrid1.dataU.toArray() );
	const v1 = Array.from( await velocityGrid1.dataV.toArray() );

	const expectedU1 = new Array( velocityGrid1.dataSizeU[ 0 ] * velocityGrid1.dataSizeU[ 1 ] ).fill( 0 );
	const expectedV1 = new Array( velocityGrid1.dataSizeV[ 0 ] * velocityGrid1.dataSizeV[ 1 ] ).fill( - 1 * dt );

	const matches1 = allClose( u1, expectedU1 ) && allClose( v1, expectedV1 );

	log(
		'constant force (0,-1), dt=0.1',
		matches1,
		matches1 ? 'dataU all 0, dataV all -0.1 (max |diff| < 1e-4)' : `dataU=[${ u1 }], dataV=[${ v1 }]`
	);

	// ---- Test 2: position-dependent force ----

	const velocityGrid2 = grid.createFaceCenteredGrid2( N, N, 1, 1, 0, 0 );
	const solver2 = grid.createExternalForceSolver2( {
		velocityGrid: velocityGrid2,
		force: ( pos ) => vec2( pos.x, 0 ),
		dt
	} );
	solver2.applyExternalForces();

	const u2 = Array.from( await velocityGrid2.dataU.toArray() );
	const v2 = Array.from( await velocityGrid2.dataV.toArray() );

	const [ nxU, nyU ] = velocityGrid2.dataSizeU;
	const expectedU2 = new Array( nxU * nyU );
	for ( let j = 0; j < nyU; j ++ ) for ( let i = 0; i < nxU; i ++ ) expectedU2[ i + nxU * j ] = i * dt; // uPosition(i,j).x = i for this grid's spacing/origin

	const expectedV2 = new Array( velocityGrid2.dataSizeV[ 0 ] * velocityGrid2.dataSizeV[ 1 ] ).fill( 0 );

	const matches2 = allClose( u2, expectedU2 ) && allClose( v2, expectedV2 );

	log(
		'position-dependent force (pos) => vec2(pos.x, 0), dt=0.1',
		matches2,
		matches2
			? 'dataU is the expected i*0.1 ramp, dataV all 0 (max |diff| < 1e-4)'
			: `dataU=[${ u2 }], expected=[${ expectedU2 }]`
	);

} catch ( error ) {

	log( 'failed', false, error.message );
	console.error( error );

}
