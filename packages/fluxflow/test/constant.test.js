import { describe, it, expect } from 'vitest';
import {
	FLOAT_TYPE, DIRECTION_NONE, DIRECTION_LEFT, DIRECTION_RIGHT, DIRECTION_DOWN, DIRECTION_UP, DIRECTION_ALL,
	DEFAULT_GRAVITY, createGravity, setGravity
} from '../src/grid/constant.js';

describe( 'constant', () => {

	it( 'exposes the float element type', () => {

		expect( FLOAT_TYPE ).toBe( 'float' );

	} );

	it( 'direction flags are distinct bits', () => {

		const flags = [ DIRECTION_LEFT, DIRECTION_RIGHT, DIRECTION_DOWN, DIRECTION_UP ];

		expect( new Set( flags ).size ).toBe( 4 );
		flags.forEach( ( flag ) => expect( flag ).not.toBe( DIRECTION_NONE ) );

	} );

	it( 'DIRECTION_ALL is the OR of all four direction flags', () => {

		expect( DIRECTION_ALL ).toBe( DIRECTION_LEFT | DIRECTION_RIGHT | DIRECTION_DOWN | DIRECTION_UP );

	} );

} );

describe( 'gravity', () => {

	it( 'createGravity returns a 0-D callable array0 field', () => {

		const gravity = createGravity();

		expect( typeof gravity ).toBe( 'function' ); // 可调用 field，tsl_array_n 的既有约定
		expect( gravity.shape ).toEqual( [] );
		expect( gravity.count ).toBe( 1 );

	} );

	it( 'createGravity defaults to DEFAULT_GRAVITY, accepts a custom value', () => {

		expect( () => createGravity() ).not.toThrow();
		expect( () => createGravity( -1.62 ) ).not.toThrow(); // 比如月球重力
		expect( DEFAULT_GRAVITY ).toBe( -9.81 );

	} );

	it( 'setGravity updates an existing gravity field without throwing', () => {

		const gravity = createGravity();

		expect( () => setGravity( gravity, -3.7 ) ).not.toThrow(); // 比如火星重力

	} );

} );
