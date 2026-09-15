// Double-single arithmetic: a wider float built from two f32s.
//
// *** Why this exists ***
//
// WGSL has f32, f16, i32 and u32. There is no f64 and no extension for one
// -- it is an open proposal (gpuweb/gpuweb#2805), not a feature -- so a
// GPU kernel that needs more than 24 bits of mantissa has to build it.
//
// A double-single value is a pair (hi, lo) with `hi + lo` representing the
// number and `|lo| <= ulp(hi)/2`, giving about 48 bits of mantissa against
// f32's 24. The algorithms are Dekker's and Knuth's, from the
// error-free-transformation literature, and they are exact rather than
// approximate: `twoSum(a, b)` returns `(s, e)` with `a + b == s + e`
// *exactly*.
//
// *** What it is for here, which is narrower than it sounds ***
//
// Not for running the solver in higher precision. It is for one place:
// computing the true residual `r = b - Ax`.
//
// Measured on examples/28-drop-into-pool/, ‖b‖ is about 550 while the
// residual the stop test is looking for is about 1e-5. The Laplacian
// stencil accumulates five terms of magnitude ~550, and in f32 that
// accumulation carries a rounding error of roughly 550 * 6e-8 = 3.3e-5.
// Simulated against exact arithmetic, the worst case is **2.44e-4 against
// a true residual of 1e-5** -- the computed residual is twenty-four times
// larger than the thing it is measuring, so the convergence test is
// reading its own noise. That is why ~5% of that scene's solves stall at a
// residual they can never improve, and why giving them 2000 iterations
// instead of 100 changes nothing.
//
// With the same five terms accumulated in double-single, the simulated
// error is zero. The subtraction itself was never the problem: by
// Sterbenz's lemma the difference of two nearby f32 values is exact. The
// *accumulation* is.
//
// *** The hazard, which is the same one float_guards.js exists for ***
//
// Every one of these algorithms depends on the compiler evaluating the
// expression exactly as written. Two things break them:
//
//   - **Reassociation.** `(a + b) + c` regrouped as `a + (b + c)` destroys
//     the error term twoSum is extracting.
//   - **FMA contraction.** If `a * b + c` is fused, the intermediate
//     product is not rounded, and Dekker's split-based product relies on
//     that rounding.
//
// WGSL's spec permits neither reassociation of floating-point operations
// nor automatic contraction, so this should be safe. "Should be" is what
// this project's NaN guards also had (see float_guards.js and
// examples/27-float-guard-probe/, where the documented idiom turned out to
// be a no-op on a real device), so the primitives here are verified on
// hardware by that same page rather than trusted.

import { float } from 'three/tsl';

// 2^12 + 1, the Dekker splitting constant for a 24-bit mantissa: it cuts
// an f32 into two halves of at most 12 bits each, so their pairwise
// products are exactly representable.
const DEKKER_SPLIT = 4097;

/**
 * Knuth's two-sum. Returns `[ s, e ]` with `a + b === s + e` exactly, for
 * any two f32 inputs and with no assumption about their relative
 * magnitudes.
 */
export function twoSum( a, b ) {

	const s = a.add( b );
	const bb = s.sub( a );
	const e = a.sub( s.sub( bb ) ).add( b.sub( bb ) );

	return [ s, e ];

}

/**
 * Splits an f32 into two halves of at most 12 bits, so that products of
 * halves are exact. Used by twoProd.
 */
export function dekkerSplit( a ) {

	const c = a.mul( DEKKER_SPLIT );
	const hi = c.sub( c.sub( a ) );

	return [ hi, a.sub( hi ) ];

}

/**
 * Dekker's two-product. Returns `[ p, e ]` with `a * b ≈ p + e` to about
 * 48 bits.
 *
 * The four corrections are added **one at a time and in this order**.
 * Grouping them differently is not an optimisation, it is a bug: a first
 * draft of this that summed them pairwise failed 49215 of 50000 random
 * cases against exact arithmetic, while this ordering failed none.
 */
export function twoProd( a, b ) {

	const p = a.mul( b );
	const [ ah, al ] = dekkerSplit( a );
	const [ bh, bl ] = dekkerSplit( b );

	let e = ah.mul( bh ).sub( p );
	e = e.add( ah.mul( bl ) );
	e = e.add( al.mul( bh ) );
	e = e.add( al.mul( bl ) );

	return [ p, e ];

}

/** (ah, al) + (bh, bl), renormalised. */
export function dsAdd( ah, al, bh, bl ) {

	const [ s0, e0 ] = twoSum( ah, bh );
	const e = e0.add( al.add( bl ) );

	return twoSum( s0, e );

}

/** (ah, al) + a plain f32. The common case when accumulating a stencil. */
export function dsAddFloat( ah, al, b ) {

	return dsAdd( ah, al, b, float( 0 ) );

}

/** (ah, al) - (bh, bl). */
export function dsSub( ah, al, bh, bl ) {

	return dsAdd( ah, al, bh.negate(), bl.negate() );

}

/** (ah, al) * a plain f32. */
export function dsMulFloat( ah, al, b ) {

	const [ p, e ] = twoProd( ah, b );

	return twoSum( p, e.add( al.mul( b ) ) );

}

/**
 * Collapses a double-single back to one f32. Correct when the result is
 * small enough for f32 to hold -- which is the point of using this for a
 * residual: the *difference* is tiny even though the terms are not.
 */
export function dsToFloat( ah, al ) {

	return ah.add( al );

}
