// 3D counterpart of array_utils.js -- same factory shape, extrapolation
// diffuses into 6 face neighbours instead of 4. array_utils.js untouched.

import * as tsl_array_n from 'tsl_array_n';
import { int, float } from 'three/tsl';
import { createCellCenteredScalarGrid3 } from './grid_data3.js';
import { collocatedValueAtPosition3 } from './grid_math3.js';

export function createCopyKernel3( src, dst, shape = src.shape ) {

	return tsl_array_n.kernel( shape, ( i, j, k ) => {

		dst( i, j, k ).assign( src( i, j, k ) );

	} );

}

export function createDecayKernel3( field, decay, shape = field.shape ) {

	const decayNode = typeof decay === 'number' ? float( decay ) : decay;

	return tsl_array_n.kernel( shape, ( i, j, k ) => {

		field( i, j, k ).assign( field( i, j, k ).mul( float( 1 ).sub( decayNode ) ) );

	} );

}

// See array_utils.js's own createAdvectedScalarField header comment for
// the full shape/purpose -- same "two state fields + two raw scratch
// fields" ping-pong, one dimension wider.
export function createAdvectedScalarField3( resolutionX, resolutionY, resolutionZ, gridSpacingX, gridSpacingY, gridSpacingZ, originX, originY, originZ ) {

	function buildState() {

		const grid = createCellCenteredScalarGrid3( resolutionX, resolutionY, resolutionZ, gridSpacingX, gridSpacingY, gridSpacingZ, originX, originY, originZ );

		return {
			...grid,
			sample( pos ) {

				return collocatedValueAtPosition3( grid.data, grid.gridSpacing, grid.dataOrigin, pos, grid.resolution );

			}
		};

	}

	const stateA = buildState();
	const stateB = buildState();
	const rawA = { data: tsl_array_n.arrayN( 'float', [ resolutionX, resolutionY, resolutionZ ] ) };
	const rawB = { data: tsl_array_n.arrayN( 'float', [ resolutionX, resolutionY, resolutionZ ] ) };

	return { stateA, stateB, rawA, rawB };

}

// One round of diffusion into the 6 face neighbours (was 4 in 2D) -- same
// race-condition argument as array_utils.js's own createExtrapolateStepKernel2:
// the read set (already-valid neighbours) and write set (still-invalid
// cells) never overlap within one dispatch, so this parallelises safely
// with no extra synchronisation.
function createExtrapolateStepKernel3( output, validSrc, validDst, shape ) {

	const [ nx, ny, nz ] = shape;

	return tsl_array_n.kernel( shape, ( i, j, k ) => {

		tsl_array_n.If( validSrc( i, j, k ).notEqual( 0 ), () => {

			validDst( i, j, k ).assign( 1 );

		} ).Else( () => {

			const total = float( 0 ).toVar();
			const count = int( 0 ).toVar();

			tsl_array_n.If( i.add( 1 ).lessThan( nx ).and( validSrc( i.add( 1 ), j, k ).notEqual( 0 ) ), () => {

				total.addAssign( output( i.add( 1 ), j, k ) );
				count.addAssign( 1 );

			} );

			tsl_array_n.If( i.greaterThan( 0 ).and( validSrc( i.sub( 1 ), j, k ).notEqual( 0 ) ), () => {

				total.addAssign( output( i.sub( 1 ), j, k ) );
				count.addAssign( 1 );

			} );

			tsl_array_n.If( j.add( 1 ).lessThan( ny ).and( validSrc( i, j.add( 1 ), k ).notEqual( 0 ) ), () => {

				total.addAssign( output( i, j.add( 1 ), k ) );
				count.addAssign( 1 );

			} );

			tsl_array_n.If( j.greaterThan( 0 ).and( validSrc( i, j.sub( 1 ), k ).notEqual( 0 ) ), () => {

				total.addAssign( output( i, j.sub( 1 ), k ) );
				count.addAssign( 1 );

			} );

			tsl_array_n.If( k.add( 1 ).lessThan( nz ).and( validSrc( i, j, k.add( 1 ) ).notEqual( 0 ) ), () => {

				total.addAssign( output( i, j, k.add( 1 ) ) );
				count.addAssign( 1 );

			} );

			tsl_array_n.If( k.greaterThan( 0 ).and( validSrc( i, j, k.sub( 1 ) ).notEqual( 0 ) ), () => {

				total.addAssign( output( i, j, k.sub( 1 ) ) );
				count.addAssign( 1 );

			} );

			tsl_array_n.If( count.greaterThan( 0 ), () => {

				output( i, j, k ).assign( total.div( count.toFloat() ) );
				validDst( i, j, k ).assign( 1 );

			} ).Else( () => {

				validDst( i, j, k ).assign( 0 );

			} );

		} );

	} );

}

// See array_utils.js's own createExtrapolateToRegion2 header comment for
// the full run()/dispatchers() shape and why the sequence is batched.
export function createExtrapolateToRegion3( inputField, validField, outputField, shape = validField.shape ) {

	const copyInputToOutput = outputField !== inputField
		? createCopyKernel3( inputField, outputField, shape )
		: null;

	const validA = tsl_array_n.arrayN( 'int', shape );
	const validB = tsl_array_n.arrayN( 'int', shape );
	const copyValidToA = createCopyKernel3( validField, validA, shape );

	const stepAtoB = createExtrapolateStepKernel3( outputField, validA, validB, shape );
	const stepBtoA = createExtrapolateStepKernel3( outputField, validB, validA, shape );

	const batches = new Map();

	function dispatchers( numberOfIterations = 5 ) {

		const sequence = [];

		if ( copyInputToOutput ) sequence.push( copyInputToOutput );

		sequence.push( copyValidToA );

		for ( let iter = 0; iter < numberOfIterations; iter ++ ) {

			sequence.push( iter % 2 === 0 ? stepAtoB : stepBtoA );

		}

		return sequence;

	}

	function run( numberOfIterations = 5 ) {

		let batch = batches.get( numberOfIterations );

		if ( batch === undefined ) {

			batch = tsl_array_n.createBatch( dispatchers( numberOfIterations ) );
			batches.set( numberOfIterations, batch );

		}

		batch();

	}

	run.dispatchers = dispatchers;

	return run;

}
