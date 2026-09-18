// 3D counterpart of vorticity_confinement2.js. The underlying formula is
// mantaflow's own general N-D one -- eta = normalize(grad(|curl|)),
// force = strength * cross(eta, curl) -- and vorticity_confinement2.js's
// own header comment already derives the 2D file as a *specialisation* of
// this exact formula (curl3D=(0,0,w), eta3D=(ex,ey,0), expanding the cross
// product). This file is that general formula directly: curl is a genuine
// vec3 here (grid_math3.js's faceCenteredCurlAtCenter3), |curl| is its
// length rather than abs() of a scalar, and the force is the real 3D cross
// product rather than the 2D file's algebraically-expanded shortcut.

import * as tsl_array_n from 'tsl_array_n';
import { vec3, float, length, max } from 'three/tsl';
import { createCellCenteredScalarGrid3 } from './grid_data3.js';
import { collocatedValueAtPosition3, scalarGradient3, faceCenteredCurlAtCenter3 } from './grid_math3.js';

const GRADIENT_EPSILON = 1e-6;

function withSample( grid ) {

	return {
		...grid,
		sample( pos ) {

			return collocatedValueAtPosition3( grid.data, grid.gridSpacing, grid.dataOrigin, pos, grid.resolution );

		}
	};

}

function numberOrNode( value ) {

	return typeof value === 'number' ? float( value ) : value;

}

// options: see vorticity_confinement2.js's own header comment for
// velocityGrid/gridSpacing/strength and the update()-before-
// onAdvanceTimeStep() sequencing requirement -- unchanged conventions.
export function createVorticityConfinement3( { velocityGrid, gridSpacing, strength } ) {

	const [ resolutionX, resolutionY, resolutionZ ] = velocityGrid.resolution;
	const [ gridSpacingX, gridSpacingY, gridSpacingZ ] = gridSpacing;
	const shape = [ resolutionX, resolutionY, resolutionZ ];
	const strengthNode = numberOrNode( strength );
	const gridSpacingNode = vec3( gridSpacingX, gridSpacingY, gridSpacingZ );

	// Vector curl, cell-centered -- was a scalar field in 2D.
	const curlField = tsl_array_n.arrayN( 'vec3', shape );

	const computeCurl = tsl_array_n.kernel( shape, ( i, j, k ) => {

		curlField( i, j, k ).assign( faceCenteredCurlAtCenter3( velocityGrid.dataU, velocityGrid.dataV, velocityGrid.dataW, velocityGrid.gridSpacing, i, j, k, shape ) );

	} );

	function buildForceComponent() {

		return withSample( createCellCenteredScalarGrid3( resolutionX, resolutionY, resolutionZ, gridSpacingX, gridSpacingY, gridSpacingZ, 0, 0, 0 ) );

	}

	const forceX = buildForceComponent();
	const forceY = buildForceComponent();
	const forceZ = buildForceComponent();

	// |curl|, not abs(curl) -- curl is a vector now.
	function curlMagnitude( i, j, k ) {

		return length( curlField( i, j, k ) );

	}

	const computeConfinementForce = tsl_array_n.kernel( shape, ( i, j, k ) => {

		const grad = scalarGradient3( curlMagnitude, gridSpacingNode, i, j, k, shape );
		const gradLen = length( grad );
		const degenerate = gradLen.lessThan( GRADIENT_EPSILON );
		const eta = grad.div( max( gradLen, GRADIENT_EPSILON ) );
		const curl = curlField( i, j, k );

		// The real 3D cross product eta x curl (the 2D file's own
		// (eta.y*curl, -eta.x*curl) is this same formula's algebraic
		// specialisation to curl=(0,0,w), not a separate derivation).
		const forceVec = vec3(
			eta.y.mul( curl.z ).sub( eta.z.mul( curl.y ) ),
			eta.z.mul( curl.x ).sub( eta.x.mul( curl.z ) ),
			eta.x.mul( curl.y ).sub( eta.y.mul( curl.x ) )
		).mul( strengthNode );

		forceX.data( i, j, k ).assign( degenerate.select( float( 0 ), forceVec.x ) );
		forceY.data( i, j, k ).assign( degenerate.select( float( 0 ), forceVec.y ) );
		forceZ.data( i, j, k ).assign( degenerate.select( float( 0 ), forceVec.z ) );

	} );

	function update() {

		computeCurl();
		computeConfinementForce();

	}

	function force( pos ) {

		return vec3( forceX.sample( pos ), forceY.sample( pos ), forceZ.sample( pos ) );

	}

	return { update, force };

}
