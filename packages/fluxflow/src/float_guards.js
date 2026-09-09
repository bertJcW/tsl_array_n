// Robust non-finite detection for TSL kernels.
//
// *** Why this file exists: the standard WGSL NaN idiom does not work. ***
//
// Core WGSL dropped isnan()/isinf(), and the documented replacement is the
// self-inequality trick: a NaN is the only value not equal to itself, so
// `x != x` is supposed to be true exactly for NaN. Every safety net in this
// library was written that way.
//
// It does not hold. WGSL's floating point is explicitly permissive about
// non-finite values -- an implementation is allowed to assume they never
// occur and to fold expressions accordingly -- so `x != x` is free to
// compile to a constant `false`, and on the backends this library was
// measured on, it does. The test was silently a no-op everywhere it was
// used.
//
// Measured directly on a WebGPU device, one value per row, with a kernel
// that writes each predicate's result to a buffer and reads it back. The
// four columns are the four candidate tests; the marked cells are where a
// test gives the wrong answer. test/float_guards.test.js cannot re-run this
// (it has no GPU) and does not pretend to -- it checks the bit predicate's
// arithmetic and guards against the broken idiom coming back:
//
//     value    x != x   abs(x) > 100   !(abs(x) <= 100)   bit test
//     NaN        0  X       0  X            1  ok           1  ok
//     +Inf       0          1  ok           1  ok           1  ok
//     -Inf       0          1  ok           1  ok           1  ok
//     1e30       0          1  ok           1  ok           0  (finite)
//     5.0        0  ok      0  ok           0  ok           0  ok
//     -0.0       0  ok      0  ok           0  ok           0  ok
//
// Two things survive contact with a NaN, and this file uses both:
//
//   * **The bit test.** A float is non-finite exactly when its exponent
//     field is all ones, which is integer arithmetic on the bit pattern and
//     so cannot be constant-folded away by an assumption about float
//     values. This is the primary test.
//   * **An inverted bound.** `abs(x) <= limit` is false for a NaN (an
//     unordered comparison is false), so *negating* a "definitely fine"
//     test flags NaN correctly, where asserting a "definitely bad" test
//     with `>` does not. Note the asymmetry: both forms return false for a
//     NaN, but only for the inverted one is false the answer that leads to
//     the right conclusion. Write bounds this way round.
//
// *** What this was costing ***
//
// grid_pressure_solver2.js snapshots pressure before every solve and reverts
// if the result looks implausible. With the NaN half of that check inert,
// a solve that returned NaN across the whole fluid region was accepted as
// good, the velocity correction multiplied it through the grid, the
// boundary clamp turned the NaN into its own bound (clamp(NaN, -100, 100)
// yields -100, not NaN), and the particle advection clamp then folded every
// particle into the domain corner. The visible result is a liquid that runs
// correctly for a couple of hundred frames and then collapses to a point in
// a single step, with every diagnostic reporting healthy right up to the
// frame it happens: examples/20-flip-dam-break/ did this at frame 124 (dead
// by 124, healthy at 122) and examples/26-dye-free-surface/ somewhere
// between frames 120 and 180, and neither was caused by anything in the
// scene. Both run indefinitely with the guard repaired -- 2070 and 2100
// frames respectively, coming to rest rather than diverging.

import { float, uint, abs, floatBitsToUint } from 'three/tsl';

// IEEE-754 binary32: sign(1) exponent(8) mantissa(23). Exponent all ones is
// Inf (zero mantissa) or NaN (non-zero mantissa) -- this masks off sign and
// mantissa and asks only about the exponent, so it catches both without
// caring which.
const EXPONENT_MASK = 0x7f800000;

/**
 * True when a float node is NaN or +/-Infinity. Bit-pattern based, so it is
 * unaffected by whatever the backend assumes about float values.
 */
export function isNonFinite( value ) {

	return floatBitsToUint( value ).bitAnd( uint( EXPONENT_MASK ) ).equal( uint( EXPONENT_MASK ) );

}

/**
 * True when a vec2 node has a non-finite value in either component.
 */
export function isNonFinite2( value ) {

	return isNonFinite( value.x ).or( isNonFinite( value.y ) );

}

/**
 * True when a float node is non-finite OR larger in magnitude than `limit`.
 * The bound is written as a negated "within range" test on purpose -- see
 * this file's header comment on why `abs(x) > limit` is not equivalent.
 */
export function isNonFiniteOrAbove( value, limit ) {

	return isNonFinite( value ).or( abs( value ).lessThanEqual( float( limit ) ).not() );

}
