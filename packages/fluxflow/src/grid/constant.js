// Ported from constant.py.
//
// DTYPE's switchable precision (setPrecision(use_double)) has no equivalent
// in WGSL/WebGPU compute (no native f64) -- not ported, fixed to float(f32)
// only.
//
// initConstant() exists in the source because GRAVITY needs setPrecision to
// have already fixed DTYPE before it can allocate a field at the right
// precision, so a "call initConstant before using GRAVITY" orchestration
// step is needed. There's no precision switch on the JS side, so there's
// nothing to wait for -- GRAVITY can be created whenever it's needed, and
// the initConstant orchestration layer itself just isn't needed -- not a
// missed port, the reason it existed is gone.
//
// GRAVITY itself also isn't a module-level singleton (in the source, GRAVITY
// is a global field shared by the whole constant.py module): on the JS side
// each solver/simulation creates its own, avoiding shared mutable global
// state if a page ever runs multiple simulations at once.

import * as tsl_array_n from 'tsl_array_n';

export const FLOAT_TYPE = 'float';

export const DIRECTION_NONE  = 0;
export const DIRECTION_LEFT  = 1 << 0;
export const DIRECTION_RIGHT = 1 << 1;
export const DIRECTION_DOWN  = 1 << 2;
export const DIRECTION_UP    = 1 << 3;
export const DIRECTION_ALL   = DIRECTION_LEFT | DIRECTION_RIGHT | DIRECTION_DOWN | DIRECTION_UP;

// The two extra walls a 3D domain has and a 2D one doesn't -- added here,
// not in a separate constant3.js, because this file already has no "2"
// suffix and every 2D-specific symbol above is untouched by this addition
// (DIRECTION_ALL still means exactly the 4 bits 2D code has always meant
// by it; 3D solvers use DIRECTION_ALL_3D instead).
export const DIRECTION_BACK  = 1 << 4;
export const DIRECTION_FRONT = 1 << 5;
export const DIRECTION_ALL_3D = DIRECTION_LEFT | DIRECTION_RIGHT | DIRECTION_DOWN | DIRECTION_UP | DIRECTION_BACK | DIRECTION_FRONT;

export const DEFAULT_GRAVITY = -9.81;

// Corresponds to the two lines in the source's initConstant():
// GRAVITY = ti.field(...); GRAVITY[None] = -9.81.
export function createGravity( value = DEFAULT_GRAVITY ) {

	const gravity = tsl_array_n.array0( 'float' );
	gravity.fromArray( new Float32Array( [ value ] ) );
	return gravity;

}

// Corresponds to the source's setGravity(newGravity) -- the source mutates
// an implicit module-level GRAVITY; here the caller must pass which gravity
// to update (since it's no longer a singleton).
export function setGravity( gravity, newValue ) {

	gravity.fromArray( new Float32Array( [ newValue ] ) );

}
