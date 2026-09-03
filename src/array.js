import { instancedArray, int } from 'three/tsl';
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

export function flattenNodeIndex( strides, indices ) {

	if ( indices.length !== strides.length ) {

		throw new Error(
			`tslify: expected ${ strides.length } index(es), got ${ indices.length }.`
		);

	}

	let flat = int( indices[ 0 ] );

	for ( let i = 1; i < indices.length; i ++ ) {

		flat = flat.add( int( indices[ i ] ).mul( strides[ i ] ) );

	}

	return flat;

}

export function arrayN( type, shape ) {

	const dims = normalizeShape( shape );
	const strides = computeStrides( dims );
	const count = dims.reduce( ( total, dim ) => total * dim, 1 );

	const node = instancedArray( count, type );

	// callable: field( i, j, ... ) -> the GPU element at that (node-valued) index,
	// for use inside kernel()/func() bodies.
	const field = ( ...indices ) => node.element( flattenNodeIndex( strides, indices ) );

	field.shape = dims;
	field.count = count;
	field.type = type;
	field.node = node;

	// CPU-side flat index math (plain numbers) — distinct from calling field(...) itself.
	field.at = ( ...indices ) => flattenIndex( strides, indices );

	field.toArray = () => {

		return getRenderer().getArrayBufferAsync( node.value ).then( ( arrayBuffer ) => {

			return new node.value.array.constructor( arrayBuffer );

		} );

	};

	field.fromArray = ( data ) => {

		node.value.array.set( data );
		node.value.needsUpdate = true;

	};

	return field;

}

export function array2( type, width, height ) {

	return arrayN( type, [ width, height ] );

}

export function array3( type, width, height, depth ) {

	return arrayN( type, [ width, height, depth ] );

}
