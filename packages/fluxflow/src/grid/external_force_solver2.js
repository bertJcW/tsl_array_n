// External forces: applies a user-supplied force field to a velocity grid.
// No Python or jet source to port here beyond the general shape -- jet's
// own GridFluidSolver2::computeExternalForces/computeGravity only ever
// applies a *constant* gravity vector (velocity += gravity*dt); this
// generalizes that same update to an arbitrary caller-supplied force
// *function* instead of a hardcoded constant, which jet itself has no
// built-in equivalent for.
//
// Deliberately minimal: this only knows how to apply "a function of
// position" to the velocity grid. Where that function comes from is
// entirely up to the caller -- a hand-written closure is the only source
// supported for now (a VDB-backed force field was considered and
// deferred; see the design note below), but the interface is shaped so
// that whatever produces `(pos) => vec2` plugs in here unchanged.

import * as tsl_array_n from 'tsl_array_n';
import { float } from 'three/tsl';

// options.velocityGrid: the FaceCenteredGrid2 (grid_data2.js) forces are
// applied to.
// options.force: (pos) => vec2, a TSL-node-returning function the caller
// writes -- same idiom as advection_solver2.js's sampleVelocity/
// sampleBoundary. Called once per face-centered velocity sample (u-faces
// and v-faces separately, at their own staggered positions), added to
// that face's velocity component scaled by `dt`. Required -- no default,
// since the caller is expected to author this themselves (the trivial
// constant-gravity case is already just a one-line closure:
// `(pos) => vec2(0, -9.8)`, no separate convenience wrapper needed).
//
// A VDB-backed force field, whenever one is built, is meant to plug in
// here exactly the same way: parse the file, upload the result into a
// tsl_array_n field, and expose it as a plain (pos) => vec2 closure via
// collocatedValueAtPosition2/collocatedCubicValueAtPosition2 (grid_math.js)
// -- no changes needed to this file at all.
//
// options.dt: same flexible convention as advection_solver2.js -- a plain
// JS number (baked in as a constant at kernel-build time) or a node such
// as an array0('float')'s own callable reference (kept live: update the
// array0's contents via fromArray() between dispatches and this solver's
// already-built kernels pick up the new value on their next dispatch,
// the same pattern linalg.js's alpha/beta scalars and
// advection_solver2.js's own dt already rely on).
export function createExternalForceSolver2( { velocityGrid, force, dt } ) {

	const dtNode = typeof dt === 'number' ? float( dt ) : dt;

	const dispatchU = tsl_array_n.kernel( velocityGrid.dataSizeU, ( i, j ) => {

		const pos = velocityGrid.uPosition( i, j );
		velocityGrid.dataU( i, j ).addAssign( force( pos ).x.mul( dtNode ) );

	} );

	const dispatchV = tsl_array_n.kernel( velocityGrid.dataSizeV, ( i, j ) => {

		const pos = velocityGrid.vPosition( i, j );
		velocityGrid.dataV( i, j ).addAssign( force( pos ).y.mul( dtNode ) );

	} );

	function applyExternalForces() {

		dispatchU();
		dispatchV();

	}

	return { applyExternalForces };

}
