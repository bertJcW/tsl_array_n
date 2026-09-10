// Structural tests only: construction builds the kernels, which needs no
// GPU, but the numbers surface tension produces do -- so correctness is
// verified live instead, by examples/29-static-droplet/, which measures the
// pressure jump across a blob's interface against the Young-Laplace value
// sigma/R. See that example's own header comment.

import { describe, it, expect } from 'vitest';
import * as tsl_array_n from 'tsl_array_n';
import { createSurfaceTension2 } from '../src/grid/surface_tension2.js';
import { createFaceCenteredGrid2 } from '../src/grid/grid_data2.js';

function makeScene( nx = 8, ny = 8 ) {

	const velocityGrid = createFaceCenteredGrid2( nx, ny, 1, 1, 0, 0 );
	const phaseField = tsl_array_n.array2( 'float', nx, ny );

	return { velocityGrid, phase: phaseField, resolution: [ nx, ny ] };

}

describe( 'createSurfaceTension2', () => {

	it( 'constructs for a constant-density scene without throwing', () => {

		const { velocityGrid, phase, resolution } = makeScene();

		expect( () => createSurfaceTension2( {
			velocityGrid, phase, resolution, gridSpacing: [ 1, 1 ], sigma: 0.05, dt: 1 / 60
		} ) ).not.toThrow();

	} );

	it( 'constructs with faceWeights, for a variable-density scene', () => {

		const { velocityGrid, phase, resolution } = makeScene();
		const betaU = tsl_array_n.arrayN( 'float', velocityGrid.dataSizeU );
		const betaV = tsl_array_n.arrayN( 'float', velocityGrid.dataSizeV );

		expect( () => createSurfaceTension2( {
			velocityGrid, phase, resolution, gridSpacing: [ 1, 1 ], sigma: 0.05, dt: 1 / 60,
			faceWeights: { u: betaU, v: betaV }, referenceDensity: 1000
		} ) ).not.toThrow();

	} );

	it( 'accepts live nodes for sigma and dt, not just numbers', () => {

		const { velocityGrid, phase, resolution } = makeScene();
		const sigma = tsl_array_n.array0( 'float' );
		const dt = tsl_array_n.array0( 'float' );

		expect( () => createSurfaceTension2( {
			velocityGrid, phase, resolution, gridSpacing: [ 1, 1 ], sigma: sigma(), dt: dt()
		} ) ).not.toThrow();

	} );

	it( 'exposes apply() and the curvature field it computes', () => {

		const { velocityGrid, phase, resolution } = makeScene();
		const st = createSurfaceTension2( {
			velocityGrid, phase, resolution, gridSpacing: [ 1, 1 ], sigma: 0.05, dt: 1 / 60
		} );

		expect( typeof st.apply ).toBe( 'function' );
		// Exposed so a scene can read curvature back and check it against a
		// known shape -- examples/29-static-droplet/ does exactly that.
		expect( st.curvature ).toBeTruthy();
		expect( st.curvature.shape ).toEqual( [ 8, 8 ] );

	} );

	it( 'sizes its normal fields to the velocity grid faces, not the cells', () => {

		const { velocityGrid, phase, resolution } = makeScene( 6, 9 );
		const st = createSurfaceTension2( {
			velocityGrid, phase, resolution, gridSpacing: [ 1, 1 ], sigma: 0.05, dt: 1 / 60
		} );

		// A MAC grid's u faces are (nx+1) x ny and its v faces nx x (ny+1);
		// storing each normal component only where it is defined is what
		// makes the curvature a plain difference. See the module's header.
		expect( st.nHatU.shape ).toEqual( [ 7, 9 ] );
		expect( st.nHatV.shape ).toEqual( [ 6, 10 ] );

	} );

} );
