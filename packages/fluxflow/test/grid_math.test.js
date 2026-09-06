// Structural tests: only verify that the node graph builds successfully
// (doesn't throw) and that the return value's shape/fields are correct --
// vitest can't verify whether the numbers coming out of the bilinear
// interpolation/gradient/divergence/curl/laplacian are actually right
// (building a node graph doesn't need a real GPU, but reading real numeric
// values back requires actually running a kernel); that needs a live
// example once real fields are available, reading back via toArray() and
// comparing against hand-computed expected values.

import { describe, it, expect } from 'vitest';
import { int, vec2 } from 'three/tsl';
import * as tsl_array_n from 'tsl_array_n';
import * as gm from '../src/grid/grid_math.js';

const shape = [ 4, 4 ];
const scalar = tsl_array_n.array2( 'float', 4, 4 );
const vector = tsl_array_n.array2( 'vec2', 4, 4 );
const dataU = tsl_array_n.array2( 'float', 5, 4 ); // (nx+1, ny)
const dataV = tsl_array_n.array2( 'float', 4, 5 ); // (nx, ny+1)

const i = int( 1 );
const j = int( 2 );
const pos = vec2( 1.5, 2.5 );
const dataOrigin = vec2( 0, 0 );
const gridSpacing = vec2( 1, 1 );

function expectNode( value ) {

	expect( value ).toBeTruthy();
	expect( value.isNode ).toBe( true );

}

describe( 'grid_math — interpolation', () => {

	it( 'faceCenteredValueAtCellCenter2', () => {

		expectNode( gm.faceCenteredValueAtCellCenter2( dataU, dataV, i, j ) );

	} );

	it( 'bilinearCoordsAndWeights2 returns all 8 named node fields', () => {

		const result = gm.bilinearCoordsAndWeights2( pos, dataOrigin, gridSpacing, shape );

		for ( const key of [ 'i0c', 'j0c', 'i1c', 'j1c', 'w00', 'w10', 'w01', 'w11' ] ) {

			expect( result ).toHaveProperty( key );
			expectNode( result[ key ] );

		}

	} );

	it( 'collocatedValueAtPosition2', () => {

		expectNode( gm.collocatedValueAtPosition2( scalar, gridSpacing, dataOrigin, pos, shape ) );

	} );

	it( 'faceCenteredValueAtPosition2', () => {

		expectNode( gm.faceCenteredValueAtPosition2(
			dataU, dataV, gridSpacing, dataOrigin, dataOrigin, pos, dataU.shape, dataV.shape
		) );

	} );

} );

describe( 'grid_math — gradient', () => {

	it( 'scalarGradient2', () => {

		expectNode( gm.scalarGradient2( scalar, gridSpacing, i, j, shape ) );

	} );

	it( 'vectorGradient2', () => {

		expectNode( gm.vectorGradient2( vector, gridSpacing, i, j, shape ) );

	} );

	it( 'scalarGradientAtPosition2', () => {

		expectNode( gm.scalarGradientAtPosition2( scalar, gridSpacing, dataOrigin, pos, shape ) );

	} );

	it( 'bilinearGradientAtPosition2', () => {

		expectNode( gm.bilinearGradientAtPosition2( scalar, gridSpacing, dataOrigin, pos, shape ) );

	} );

	// A position exactly on (or past) the grid's own last cell edge --
	// confirmed on real hardware to matter: grid_outflow_solver2.js samples
	// this exact function at face-centered positions that run half a cell
	// past a cell-centered SDF grid's own last cell center, which used to
	// collapse this function's internal index pair to the same column/row
	// and silently zero out that axis's gradient component. This is a
	// structural (no-throw) check only -- see this file's own header
	// comment for why a numeric assertion needs a live example instead; the
	// actual before/after numeric behavior was verified with an independent
	// plain-JS reference implementation of the same formula, not here.
	it( 'bilinearGradientAtPosition2 at the grid\'s own boundary edge', () => {

		const edgePos = vec2( shape[ 0 ], shape[ 1 ] );
		expectNode( gm.bilinearGradientAtPosition2( scalar, gridSpacing, dataOrigin, edgePos, shape ) );

	} );

	it( 'vectorGradientAtPosition2', () => {

		expectNode( gm.vectorGradientAtPosition2( vector, gridSpacing, dataOrigin, pos, shape ) );

	} );

} );

describe( 'grid_math — divergence', () => {

	it( 'collocatedDivergence2', () => {

		expectNode( gm.collocatedDivergence2( vector, gridSpacing, i, j, shape ) );

	} );

	it( 'faceCenteredDivergenceAtCenter2', () => {

		expectNode( gm.faceCenteredDivergenceAtCenter2( dataU, dataV, gridSpacing, i, j ) );

	} );

	it( 'collocatedDivergenceAtPosition2', () => {

		expectNode( gm.collocatedDivergenceAtPosition2( vector, gridSpacing, dataOrigin, pos, shape ) );

	} );

	it( 'faceCenteredDivergenceAtPosition2', () => {

		expectNode( gm.faceCenteredDivergenceAtPosition2( dataU, dataV, gridSpacing, dataOrigin, pos, shape ) );

	} );

} );

describe( 'grid_math — curl', () => {

	it( 'collocatedCurl2', () => {

		expectNode( gm.collocatedCurl2( vector, gridSpacing, i, j, shape ) );

	} );

	it( 'faceCenteredCurlAtCenter2', () => {

		expectNode( gm.faceCenteredCurlAtCenter2( dataU, dataV, gridSpacing, i, j, shape ) );

	} );

	it( 'collocatedCurlAtPosition2', () => {

		expectNode( gm.collocatedCurlAtPosition2( vector, gridSpacing, dataOrigin, pos, shape ) );

	} );

	it( 'faceCenteredCurlAtPosition2', () => {

		expectNode( gm.faceCenteredCurlAtPosition2( dataU, dataV, gridSpacing, dataOrigin, pos, shape ) );

	} );

} );

describe( 'grid_math — laplacian', () => {

	it( 'scalarLaplacian2', () => {

		expectNode( gm.scalarLaplacian2( scalar, gridSpacing, i, j, shape ) );

	} );

	it( 'scalarLaplacianAtPosition2', () => {

		expectNode( gm.scalarLaplacianAtPosition2( scalar, gridSpacing, dataOrigin, pos, shape ) );

	} );

	it( 'vectorLaplacian2', () => {

		expectNode( gm.vectorLaplacian2( vector, gridSpacing, i, j, shape ) );

	} );

	it( 'vectorLaplacianAtPosition2', () => {

		expectNode( gm.vectorLaplacianAtPosition2( vector, gridSpacing, dataOrigin, pos, shape ) );

	} );

} );
