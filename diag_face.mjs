// Temporary: what the two halves of the solver each think about the rod's
// own boundary faces -- the pressure system's open-area weight, and the
// velocity constraint's inside/outside test -- for faces that lie exactly
// on a grid-aligned collider's surface.
import { primitivesDistanceCpu, box } from './packages/fluxflow/src/grid/primitive_sdf3.js';
import { fractionInsideCpu } from './packages/fluxflow/src/grid/level_set_utils.js';

const NY = 24, NZ = 24;
const shapes = [ box( [ 10, NY / 2 - 3, NZ / 2 ], [ 3, 3, NZ ] ) ];
const phi = ( x, y, z ) => primitivesDistanceCpu( shapes, x, y, z );

console.log( 'u faces along y = 9.5, z = 12.5 (the rod spans x 7..13, y 6..12)\n' );
console.log( ' x   phi(face)   weight   noFlux applies?   markAndProject' );

for ( let i = 5; i <= 15; i ++ ) {

	const y = 9.5, z = 12.5;
	const p = phi( i, y, z );

	// the pressure system's own face weight (sdf_collider3.js)
	const h = 0.5;
	const corners = [ [ 0, -h, -h ], [ 0, h, -h ], [ 0, -h, h ], [ 0, h, h ] ].map( ( [ dx, dy, dz ] ) => phi( i + dx, y + dy, z + dz ) );
	let w = Math.min( 1, Math.max( 0, 1 - fractionInsideCpu( ...corners ) ) );
	if ( corners.every( ( value ) => value <= 0 ) ) w = 0;
	else if ( w > 0 && w < 0.01 ) w = 0.01;

	// the velocity constraint's own tests (grid_blocked_boundary_condition_solver3.js)
	const noFlux = p <= 0;
	const phi0 = phi( i - h, y, z ), phi1 = phi( i + h, y, z );
	const frac = 1 - Math.min( 1, Math.max( 0, phi0 < 0 && phi1 < 0 ? 1 : phi0 >= 0 && phi1 >= 0 ? 0 : ( phi0 < 0 ? phi0 : phi1 ) / ( phi0 - phi1 ) ) );

	console.log(
		`${ String( i ).padStart( 2 ) }  ${ p.toFixed( 2 ).padStart( 8 ) }  ${ w.toFixed( 3 ).padStart( 7 ) }   ` +
		`${ ( noFlux ? 'yes' : 'NO ' ).padStart( 14 ) }   ${ frac > 0 ? 'fluid' : 'collider (velocity set)' }`
	);

}
