import { describe, it, expect } from 'vitest';
import { int } from 'three/tsl';
import * as gd from '../src/grid/grid_data2.js';

function expectNode( value ) {

	expect( value ).toBeTruthy();
	expect( value.isNode ).toBe( true );

}

describe( 'ScalarGrid2 family', () => {

	it( 'createScalarGrid2 — dataSize === resolution, data shape matches', () => {

		const grid = gd.createScalarGrid2( 4, 6, 1, 1, 0, 0 );

		expect( grid.resolution ).toEqual( [ 4, 6 ] );
		expect( grid.dataSize ).toEqual( [ 4, 6 ] );
		expect( grid.data.shape ).toEqual( [ 4, 6 ] );
		expectNode( grid.gridSpacing );
		expectNode( grid.dataOrigin );
		expect( typeof grid.clear ).toBe( 'function' );
		expectNode( grid.dataPosition( int( 1 ), int( 2 ) ) );

	} );

	it( 'createCellCenteredScalarGrid2 — same dataSize as base, offset dataOrigin (different node than base)', () => {

		const grid = gd.createCellCenteredScalarGrid2( 4, 6, 1, 1, 0, 0 );

		expect( grid.dataSize ).toEqual( [ 4, 6 ] );
		expect( grid.data.shape ).toEqual( [ 4, 6 ] );
		expectNode( grid.dataOrigin );

	} );

	it( 'createVertexCenteredScalarGrid2 — dataSize is resolution+1 on each axis', () => {

		const grid = gd.createVertexCenteredScalarGrid2( 4, 6, 1, 1, 0, 0 );

		expect( grid.dataSize ).toEqual( [ 5, 7 ] );
		expect( grid.data.shape ).toEqual( [ 5, 7 ] );

	} );


} );

describe( 'VectorGrid2 family', () => {

	it( 'createCollocatedVectorGrid2 — dataSize === resolution, vec2 element type', () => {

		const grid = gd.createCollocatedVectorGrid2( 3, 5, 1, 1, 0, 0 );

		expect( grid.dataSize ).toEqual( [ 3, 5 ] );
		expect( grid.data.type ).toBe( 'vec2' );

	} );

	it( 'createCellCenteredVectorGrid2 — same dataSize as base, offset origin', () => {

		const grid = gd.createCellCenteredVectorGrid2( 3, 5, 1, 1, 0, 0 );

		expect( grid.dataSize ).toEqual( [ 3, 5 ] );
		expectNode( grid.dataOrigin );

	} );

	it( 'createVertexCenteredVectorGrid2 — dataSize is resolution+1', () => {

		const grid = gd.createVertexCenteredVectorGrid2( 3, 5, 1, 1, 0, 0 );

		expect( grid.dataSize ).toEqual( [ 4, 6 ] );

	} );

} );

describe( 'FaceCenteredGrid2 (MAC grid)', () => {

	it( 'dataSizeU/dataSizeV are staggered relative to resolution', () => {

		const grid = gd.createFaceCenteredGrid2( 4, 4, 1, 1, 0, 0 );

		expect( grid.dataSizeU ).toEqual( [ 5, 4 ] );
		expect( grid.dataSizeV ).toEqual( [ 4, 5 ] );
		expect( grid.dataU.shape ).toEqual( [ 5, 4 ] );
		expect( grid.dataV.shape ).toEqual( [ 4, 5 ] );

	} );

	it( 'exposes callable clear/uPosition/vPosition/sample', () => {

		const grid = gd.createFaceCenteredGrid2( 4, 4, 1, 1, 0, 0 );

		expect( typeof grid.clear ).toBe( 'function' );
		expectNode( grid.uPosition( int( 1 ), int( 2 ) ) );
		expectNode( grid.vPosition( int( 1 ), int( 2 ) ) );
		expect( () => grid.sample( grid.uPosition( int( 1 ), int( 2 ) ) ) ).not.toThrow();

	} );

} );
