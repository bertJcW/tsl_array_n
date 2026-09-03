import { instancedArray } from 'three/tsl';
import { getRenderer } from './context.js';

export function normalizeShape( shape ) {

	const dims = Array.isArray( shape ) ? shape : [ shape ];

	if ( dims.length === 0 ) {

		throw new Error( 'tslify: array shape must have at least one dimension.' );

	}

	for ( const dim of dims ) {

		if ( ! Number.isInteger( dim ) || dim <= 0 ) {

			throw new Error( `tslify: array shape dimensions must be positive integers, got ${ dim }.` );

		}

	}

	return dims;

}

export function computeStrides( shape ) {

	const strides = new Array( shape.length );
	let stride = 1;

	for ( let i = 0; i < shape.length; i ++ ) {

		strides[ i ] = stride;
		stride *= shape[ i ];

	}

	return strides;

}

export function flattenIndex( strides, indices ) {

	if ( indices.length !== strides.length ) {

		throw new Error(
			`tslify: expected ${ strides.length } index(es), got ${ indices.length }.`
		);

	}

	let index = 0;

	for ( let i = 0; i < indices.length; i ++ ) {

		index += indices[ i ] * strides[ i ];

	}

	return index;

}

export function arrayN( type, shape ) {

	const dims = normalizeShape( shape );
	const strides = computeStrides( dims );
	const count = dims.reduce( ( total, dim ) => total * dim, 1 );

	const node = instancedArray( count, type );

	return {

		shape: dims,
		count,
		type,
		node,

		at( ...indices ) {

			return flattenIndex( strides, indices );

		},

		toArray() {

			return getRenderer().getArrayBufferAsync( node.value ).then( ( arrayBuffer ) => {

				return new node.value.array.constructor( arrayBuffer );

			} );

		},

		fromArray( data ) {

			node.value.array.set( data );
			node.value.needsUpdate = true;

		}

	};

}

export function array2( type, width, height ) {

	return arrayN( type, [ width, height ] );

}

export function array3( type, width, height, depth ) {

	return arrayN( type, [ width, height, depth ] );

}
