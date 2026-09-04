// Ported from array_utils.py.
//
// Called differently from the source: Taichi's @ti.kernel functions
// "dispatch on call" themselves (Taichi caches the compiled result per
// argument-type signature behind the scenes, so each call is just a fresh
// dispatch). tsl_array_n's kernel(shape,fn) instead "builds once, returns a
// reusable dispatch function" -- it has to closure-bind a concrete field at
// build time (an existing tsl_array_n capability boundary, see decision 4 of
// the port plan, not a limitation introduced by this port). So both
// functions here are "factories": call once to build a kernel (already
// bound to specific fields), and the returned function is the actual
// dispatcher that can be called every frame -- don't assume calling
// createCopyKernel2(...) itself copies the data; you still need to call its
// return value.

import * as tsl_array_n from 'tsl_array_n';
import { int, float } from 'three/tsl';

export function createCopyKernel2( src, dst, shape = src.shape ) {

	return tsl_array_n.kernel( shape, ( i, j ) => {

		dst( i, j ).assign( src( i, j ) );

	} );

}

// One round of diffusion: cells where validSrc=1 (valid) are marked valid
// as-is; cells where validSrc=0 (invalid) look at how many of their
// up/down/left/right neighbors have validSrc=1, average those neighbors'
// output values into their own output, and stay invalid if there are no
// valid neighbors at all.
//
// Race-condition check: this function both reads and writes output within
// the same dispatch -- but only threads where validSrc[i,j]==0 ever write
// output(i,j), and every neighbor value read comes from a cell where
// validSrc[neighbor]!=0, and such a cell necessarily takes the "already
// valid" branch this round and never writes output. In other words, the set
// of cells that get read and the set of cells that get written this round
// never overlap, so this is safe to parallelize without extra atomics or a
// double-buffered output.
function createExtrapolateStepKernel2( output, validSrc, validDst, shape ) {

	const [ nx, ny ] = shape;

	return tsl_array_n.kernel( shape, ( i, j ) => {

		tsl_array_n.If( validSrc( i, j ).notEqual( 0 ), () => {

			validDst( i, j ).assign( 1 );

		} ).Else( () => {

			const total = float( 0 ).toVar();
			const count = int( 0 ).toVar();

			tsl_array_n.If( i.add( 1 ).lessThan( nx ).and( validSrc( i.add( 1 ), j ).notEqual( 0 ) ), () => {

				total.addAssign( output( i.add( 1 ), j ) );
				count.addAssign( 1 );

			} );

			tsl_array_n.If( i.greaterThan( 0 ).and( validSrc( i.sub( 1 ), j ).notEqual( 0 ) ), () => {

				total.addAssign( output( i.sub( 1 ), j ) );
				count.addAssign( 1 );

			} );

			tsl_array_n.If( j.add( 1 ).lessThan( ny ).and( validSrc( i, j.add( 1 ) ).notEqual( 0 ) ), () => {

				total.addAssign( output( i, j.add( 1 ) ) );
				count.addAssign( 1 );

			} );

			tsl_array_n.If( j.greaterThan( 0 ).and( validSrc( i, j.sub( 1 ) ).notEqual( 0 ) ), () => {

				total.addAssign( output( i, j.sub( 1 ) ) );
				count.addAssign( 1 );

			} );

			tsl_array_n.If( count.greaterThan( 0 ), () => {

				output( i, j ).assign( total.div( count.toFloat() ) );
				validDst( i, j ).assign( 1 );

			} ).Else( () => {

				validDst( i, j ).assign( 0 );

			} );

		} );

	} );

}

// Diffuses values from the validField=1 (valid) region into the =0
// (invalid) region, expanding outward by one ring of cells per iteration.
// Returns a run(numberOfIterations=5) function -- the validA/validB internal
// scratch buffers and the two directional step kernels (A->B / B->A) are
// built once here, and every subsequent call to run() reuses them instead of
// rebuilding the kernel graph (this corresponds to the source's
// validA, validB = validB, validA ping-pong, just that a kernel here can't
// swap bound arguments the way a Python function can, so it becomes two
// fixed-direction kernels called alternately by parity instead -- same
// effect).
export function createExtrapolateToRegion2( inputField, validField, outputField, shape = validField.shape ) {

	const copyInputToOutput = outputField !== inputField
		? createCopyKernel2( inputField, outputField, shape )
		: null;

	const validA = tsl_array_n.arrayN( 'int', shape );
	const validB = tsl_array_n.arrayN( 'int', shape );
	const copyValidToA = createCopyKernel2( validField, validA, shape );

	const stepAtoB = createExtrapolateStepKernel2( outputField, validA, validB, shape );
	const stepBtoA = createExtrapolateStepKernel2( outputField, validB, validA, shape );

	return function run( numberOfIterations = 5 ) {

		if ( copyInputToOutput ) copyInputToOutput();

		copyValidToA();

		for ( let iter = 0; iter < numberOfIterations; iter ++ ) {

			if ( iter % 2 === 0 ) stepAtoB();
			else stepBtoA();

		}

	};

}
