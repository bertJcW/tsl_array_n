// Ported from level_set_utils.py.

import { float } from 'three/tsl';

export function isInsideSdf( phi ) {

	return phi.lessThan( 0 );

}

// Corresponds to the source's if/elif/elif chain -- the three branches are
// mutually exclusive and each is a pure value selection (no side effects or
// early exit), so this translates directly to nested select() rather than
// If() (the source's own comment notes that Taichi's @ti.func doesn't
// support an early return inside a runtime branch, only a single return at
// the end -- TSL is even more restrictive here: select() has no statement
// form at all, only an expression).
// The second level only needs to check inside0 (no need for "and not
// inside1"): reaching this level means the first branch (inside0 and
// inside1) already evaluated false, and inside0 being true at that point
// implies inside1 must be false; the third level follows the same logic --
// reaching it means neither of the first two held, so inside1 being true
// there implies inside0 must be false.
export function fractionInsideSdf( phi0, phi1 ) {

	const inside0 = isInsideSdf( phi0 );
	const inside1 = isInsideSdf( phi1 );

	return inside0.and( inside1 ).select(
		float( 1 ),
		inside0.select(
			phi0.div( phi0.sub( phi1 ) ),
			inside1.select(
				phi1.div( phi1.sub( phi0 ) ),
				float( 0 )
			)
		)
	);

}

// ---------------------------------------------------------------- CPU
// (plain-number) versions below, for computing a static collider's own
// per-face open-area fraction once at setup time (see
// sdf_collider3.js's own computeFaceWeights) -- not a TSL/kernel path,
// matches this project's established "one CPU copy, one GPU copy of the
// same formula" split (polygon_sdf.js's own polygonsSignedDistance/
// pointInPolygon pair, primitive_sdf3.js's own primitiveDistance/
// primitiveDistanceCpu pair).

function fractionInsideSdfCpu( phi0, phi1 ) {

	const inside0 = phi0 < 0;
	const inside1 = phi1 < 0;

	if ( inside0 && inside1 ) return 1;
	if ( inside0 && ! inside1 ) return phi0 / ( phi0 - phi1 );
	if ( ! inside0 && inside1 ) return phi1 / ( phi1 - phi0 );
	return 0;

}

function cycleArrayCpu( arr ) {

	const t = arr[ 0 ];
	for ( let i = 0; i < arr.length - 1; i ++ ) arr[ i ] = arr[ i + 1 ];
	arr[ arr.length - 1 ] = t;

}

// The fraction of a square's own area that lies inside the implicit
// surface, given the four corner SDF values -- a real sub-cell area
// computation (marching-squares-style, not a linear approximation),
// ported directly from Christopher Batty's own variational-fluids code
// (http://www.cs.ubc.ca/labs/imager/tr/2007/Batty_VariationalFluids/,
// https://github.com/christopherbatty/Fluid3D), via jet/fluid-engine-dev's
// own `fractionInside` (include/jet/detail/level_set_utils-inl.h, MIT,
// Doyub Kim -- see ../../THIRD-PARTY-NOTICES.md). This is what jet's own
// GridFractionalSinglePhasePressureSolver3 uses to weight a collider's
// partial occupancy of each velocity face directly into the pressure
// Poisson system's own coefficients -- see sdf_collider3.js's own
// computeFaceWeights header comment for why this port needed it too.
// A capped loop guard (arr.length iterations) replaces the source's own
// unbounded `while` -- four rotations always returns to the starting
// order, so the source's own loop can never legitimately need more, but
// this is plain JS (not compiled, GPU-dispatched code), so an exact-tie
// floating-point edge case that the source's own logic doesn't expect
// gets a bounded fallback here instead of a hung tab.
export function fractionInsideCpu( phiBottomLeft, phiBottomRight, phiTopLeft, phiTopRight ) {

	const insideCount =
		( phiBottomLeft < 0 ? 1 : 0 ) + ( phiTopLeft < 0 ? 1 : 0 ) +
		( phiBottomRight < 0 ? 1 : 0 ) + ( phiTopRight < 0 ? 1 : 0 );

	const list = [ phiBottomLeft, phiBottomRight, phiTopRight, phiTopLeft ];

	if ( insideCount === 4 ) return 1;

	if ( insideCount === 3 ) {

		for ( let guard = 0; guard < list.length && list[ 0 ] < 0; guard ++ ) cycleArrayCpu( list );

		const side0 = 1 - fractionInsideSdfCpu( list[ 0 ], list[ 3 ] );
		const side1 = 1 - fractionInsideSdfCpu( list[ 0 ], list[ 1 ] );
		return 1 - 0.5 * side0 * side1;

	}

	if ( insideCount === 2 ) {

		for ( let guard = 0; guard < list.length && ( list[ 0 ] >= 0 || ! ( list[ 1 ] < 0 || list[ 2 ] < 0 ) ); guard ++ ) cycleArrayCpu( list );

		if ( list[ 1 ] < 0 ) { // the matching signs are adjacent

			const sideLeft = fractionInsideSdfCpu( list[ 0 ], list[ 3 ] );
			const sideRight = fractionInsideSdfCpu( list[ 1 ], list[ 2 ] );
			return 0.5 * ( sideLeft + sideRight );

		}

		// matching signs are diagonally opposite -- the centre point's own
		// sign disambiguates which pair of triangles to use.
		const middlePoint = 0.25 * ( list[ 0 ] + list[ 1 ] + list[ 2 ] + list[ 3 ] );

		if ( middlePoint < 0 ) {

			let area = 0;
			const side1 = 1 - fractionInsideSdfCpu( list[ 0 ], list[ 3 ] );
			const side3 = 1 - fractionInsideSdfCpu( list[ 2 ], list[ 3 ] );
			area += 0.5 * side1 * side3;
			const side2 = 1 - fractionInsideSdfCpu( list[ 2 ], list[ 1 ] );
			const side0 = 1 - fractionInsideSdfCpu( list[ 0 ], list[ 1 ] );
			area += 0.5 * side0 * side2;
			return 1 - area;

		}

		let area = 0;
		const side0 = fractionInsideSdfCpu( list[ 0 ], list[ 1 ] );
		const side1 = fractionInsideSdfCpu( list[ 0 ], list[ 3 ] );
		area += 0.5 * side0 * side1;
		const side2 = fractionInsideSdfCpu( list[ 2 ], list[ 1 ] );
		const side3 = fractionInsideSdfCpu( list[ 2 ], list[ 3 ] );
		area += 0.5 * side2 * side3;
		return area;

	}

	if ( insideCount === 1 ) {

		for ( let guard = 0; guard < list.length && list[ 0 ] >= 0; guard ++ ) cycleArrayCpu( list );

		const side0 = fractionInsideSdfCpu( list[ 0 ], list[ 3 ] );
		const side1 = fractionInsideSdfCpu( list[ 0 ], list[ 1 ] );
		return 0.5 * side0 * side1;

	}

	return 0;

}
