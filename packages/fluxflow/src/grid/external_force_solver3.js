// 3D counterpart of external_force_solver2.js -- same "apply a caller-
// supplied (pos) => vec3 force function to each staggered face" shape, one
// more component.

import * as tsl_array_n from 'tsl_array_n';
import { float } from 'three/tsl';

// options.velocityGrid: the FaceCenteredGrid3 (grid_data3.js).
// options.force: (pos) => vec3.
// options.dt: same "number or live node" convention as every dt parameter
// in this port -- see external_force_solver2.js's own header comment.
export function createExternalForceSolver3( { velocityGrid, force, dt } ) {

	const dtNode = typeof dt === 'number' ? float( dt ) : dt;

	const dispatchU = tsl_array_n.kernel( velocityGrid.dataSizeU, ( i, j, k ) => {

		const pos = velocityGrid.uPosition( i, j, k );
		velocityGrid.dataU( i, j, k ).addAssign( force( pos ).x.mul( dtNode ) );

	} );

	const dispatchV = tsl_array_n.kernel( velocityGrid.dataSizeV, ( i, j, k ) => {

		const pos = velocityGrid.vPosition( i, j, k );
		velocityGrid.dataV( i, j, k ).addAssign( force( pos ).y.mul( dtNode ) );

	} );

	const dispatchW = tsl_array_n.kernel( velocityGrid.dataSizeW, ( i, j, k ) => {

		const pos = velocityGrid.wPosition( i, j, k );
		velocityGrid.dataW( i, j, k ).addAssign( force( pos ).z.mul( dtNode ) );

	} );

	const applyAll = tsl_array_n.createBatch( [ dispatchU, dispatchV, dispatchW ] );

	function applyExternalForces() {

		applyAll();

	}

	return { applyExternalForces };

}
