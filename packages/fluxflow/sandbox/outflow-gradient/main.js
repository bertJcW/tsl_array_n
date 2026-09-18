// Sandbox probe: what an outflow object's own SDF and gradient actually
// are along the x axis, for the exact slab examples/35-karman-vortex-
// street-3d/ builds -- and, beside it, for the geometrically identical
// strip examples/16-karman-vortex-street/ builds in 2D, since the 2D
// scene is stable and the 3D one is not.
//
// The convective outflow boundary condition (grid_outflow_solver3.js)
// takes its upstream direction straight from this gradient --
// `upstreamPt = pt - normalize(gradient) * spacing` -- so the sign of the
// x component at the faces inside the outflow decides whether those faces
// read from the fluid behind them or from the boundary in front of them.
// One is a boundary condition and the other is a feedback loop.

import * as tsl_array_n from 'tsl_array_n';
import { grid } from 'fluxflow';
import { vec2, vec3, float } from 'three/tsl';

const out = document.querySelector( '#out' );
const lines = [];
const log = ( text ) => { lines.push( text ); out.textContent = lines.join( '\n' ); };

try {

	const renderer = await tsl_array_n.init( { allowFallback: true } );
	log( `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }` );

	const NX = 48, NY = 24, NZ = 24;
	const OUTER_MARGIN = 1000;

	const outflow = grid.createSDFOutflow3( NX, NY, NZ, 1, 1, 1, 0, 0, 0 );

	function wallSlab( innerX, outerX ) {

		const cx = ( innerX + outerX ) / 2;
		const hx = Math.abs( outerX - innerX ) / 2;
		return grid.box( [ cx, NY / 2, NZ / 2 ], [ hx, NY + OUTER_MARGIN, NZ + OUTER_MARGIN ] );

	}

	outflow.addShape( wallSlab( NX - 2, NX + OUTER_MARGIN ) );

	// One sample per u-face position along x, at the domain's own middle in
	// y and z: x = 0..NX, y = 12.5, z = 12.5 (a u face sits on a cell
	// boundary in x and at a cell centre in y and z).
	const COUNT = NX + 1;
	const phi = tsl_array_n.arrayN( 'float', [ COUNT ] );
	const gx = tsl_array_n.arrayN( 'float', [ COUNT ] );
	const gy = tsl_array_n.arrayN( 'float', [ COUNT ] );
	const gz = tsl_array_n.arrayN( 'float', [ COUNT ] );

	const probe = tsl_array_n.kernel( [ COUNT ], ( i ) => {

		const pt = vec3( i.toFloat(), float( NY / 2 + 0.5 ), float( NZ / 2 + 0.5 ) );
		phi( i ).assign( outflow.sample( pt ) );
		const g = outflow.gradient( pt );
		gx( i ).assign( g.x );
		gy( i ).assign( g.y );
		gz( i ).assign( g.z );

	} );

	probe();

	const [ p, x, y, z ] = await Promise.all( [ phi.toArray(), gx.toArray(), gy.toArray(), gz.toArray() ] );

	log( '' );
	log( 'x      phi     grad.x   grad.y   grad.z   |grad|   inside   upstreamPt.x' );

	for ( let i = 0; i < COUNT; i ++ ) {

		const len = Math.hypot( x[ i ], y[ i ], z[ i ] );
		const inside = p[ i ] < 0;
		// grid_outflow_solver3.js: upstreamPt = pt - normalize(g) * spacing
		const upstream = len > 0 ? i - x[ i ] / len : NaN;
		log(
			`${ String( i ).padStart( 2 ) }  ${ p[ i ].toFixed( 3 ).padStart( 8 ) } ${ x[ i ].toFixed( 3 ).padStart( 8 ) } ` +
			`${ y[ i ].toFixed( 3 ).padStart( 8 ) } ${ z[ i ].toFixed( 3 ).padStart( 8 ) } ${ len.toFixed( 3 ).padStart( 8 ) }   ` +
			`${ inside ? 'yes' : ' no' }      ${ Number.isFinite( upstream ) ? upstream.toFixed( 2 ) : '-' }`
		);

	}

	// ---- the 2D scene's own outflow, built the same way ----

	{

		const outflow2 = grid.createSDFOutflow2( NX, NY, 1, 1, 0, 0 );
		outflow2.addPolygon( [
			[ NX - 2, - OUTER_MARGIN ], [ NX + OUTER_MARGIN, - OUTER_MARGIN ],
			[ NX + OUTER_MARGIN, NY + OUTER_MARGIN ], [ NX - 2, NY + OUTER_MARGIN ]
		] );

		const phi2 = tsl_array_n.arrayN( 'float', [ COUNT ] );
		const gx2 = tsl_array_n.arrayN( 'float', [ COUNT ] );
		const gy2 = tsl_array_n.arrayN( 'float', [ COUNT ] );

		const probe2 = tsl_array_n.kernel( [ COUNT ], ( i ) => {

			const pt = vec2( i.toFloat(), float( NY / 2 + 0.5 ) );
			phi2( i ).assign( outflow2.sample( pt ) );
			const g = outflow2.gradient( pt );
			gx2( i ).assign( g.x );
			gy2( i ).assign( g.y );

		} );

		probe2();

		const [ p2, x2, y2 ] = await Promise.all( [ phi2.toArray(), gx2.toArray(), gy2.toArray() ] );

		log( '' );
		log( '2D (examples/16), same strip:' );
		log( 'x      phi     grad.x   grad.y   |grad|   inside   upstreamPt.x' );

		for ( let i = COUNT - 6; i < COUNT; i ++ ) {

			const len = Math.hypot( x2[ i ], y2[ i ] );
			const upstream = len > 0 ? i - x2[ i ] / len : NaN;
			log(
				`${ String( i ).padStart( 2 ) }  ${ p2[ i ].toFixed( 3 ).padStart( 8 ) } ${ x2[ i ].toFixed( 3 ).padStart( 8 ) } ` +
				`${ y2[ i ].toFixed( 3 ).padStart( 8 ) } ${ len.toFixed( 3 ).padStart( 8 ) }   ${ p2[ i ] < 0 ? 'yes' : ' no' }      ` +
				`${ Number.isFinite( upstream ) ? upstream.toFixed( 2 ) : '-' }`
			);

		}

	}

} catch ( error ) {

	log( 'failed: ' + error.message );
	console.error( error );

}
