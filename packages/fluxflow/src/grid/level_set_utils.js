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
