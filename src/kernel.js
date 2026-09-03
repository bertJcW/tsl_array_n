import { Fn, instanceIndex } from 'three/tsl';
import { getRenderer } from './context.js';
import { normalizeShape } from './array.js';

function unflattenNodeIndex( flatIndexNode, dims ) {

	const indices = [];
	let remaining = flatIndexNode;

	for ( let d = 0; d < dims.length; d ++ ) {

		if ( d === dims.length - 1 ) {

			indices.push( remaining );

		} else {

			indices.push( remaining.mod( dims[ d ] ) );
			remaining = remaining.div( dims[ d ] );

		}

	}

	return indices;

}

export function kernel( shape, fn ) {

	const dims = normalizeShape( shape );
	const count = dims.reduce( ( total, dim ) => total * dim, 1 );

	if ( fn.length !== dims.length ) {

		throw new Error(
			`tsl_array_n: kernel() callback expected ${ dims.length } index parameter(s) for shape [${ dims.join( ', ' ) }], got ${ fn.length }.`
		);

	}

	const computeNode = Fn( () => {

		fn( ...unflattenNodeIndex( instanceIndex, dims ) );

	} )().compute( count );

	return () => getRenderer().compute( computeNode );

}
