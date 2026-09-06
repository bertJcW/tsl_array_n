// New file, no Python source -- inflow/outflow as reusable, SDF-based scene
// objects, architecturally parallel to sdf_collider2.js's own
// createSDFStaticCollider2/createSDFRigidBodyCollider2 (this file is their
// sibling: a different family of SDF-based scene objects, same "one file
// per family" organization). Added at the user's own explicit request,
// after an initial draft (a per-wall bitmask baked directly into
// grid_blocked_boundary_condition_solver2.js) was rejected in favor of
// this more general shape: the user can freely configure and customize
// inflow/outflow -- arbitrary shape, arbitrary placement, not hardcoded to
// "the whole left/right wall".
//
// Both factories below *compose* createSDFStaticCollider2 directly (reuse,
// not duplication -- same grid/sample/gradient/isInside/addPolygon/
// addPolygons/addSvg/clear interface a caller already knows from
// colliders), adding only what's semantically new for each role.
//
// The underlying mechanism these objects feed into -- what an inflow's or
// outflow's presence actually *does* to the simulation -- is read directly
// from mantaflow (https://github.com/thunil/mantaflow, Apache License 2.0,
// Tobias Pfaff & Nils Thuerey; see THIRD-PARTY-NOTICES.md for the full
// attribution). mantaflow's own outflow treatment turned out to be three
// independent mechanisms, confirmed by reading its source directly
// (source/plugin/extforces.cpp, pressure.cpp, advection.cpp) rather than
// assumed from a single file:
//
// 1. Pressure: outflow/empty cells get no pressure equation of their own,
//    and any *neighboring* fluid cell's own row treats that neighbor's
//    pressure as exactly 0 (mantaflow's knCorrectVelocity: `if
//    (isEmpty(i-1,j,k)) vel.x -= pressure[idx]`) -- a zero-pressure ghost
//    boundary. createOutflowPressureDirichlet2 below reproduces this with
//    *zero* changes to this port's own pressure solver at all: marking a
//    cell Dirichlet-pinned to 0 (grid_pressure_solver2.js's existing
//    `dirichlet` mechanism, built on multigrid.js's `dirichletMask`)
//    already makes every neighboring fluid cell see a correct zero-
//    pressure boundary through the completely unmodified stencil --
//    verified directly against multigrid.js's own laplacianAt/
//    laplacianDiagonalAt (a masked cell's whole row is overridden
//    unconditionally via `.select(center, sum)`, discarding whatever the
//    boundary-clamped `sum` computed) and grid_math.js's
//    faceCenteredDivergenceAtCenter2 (what actually builds the pressure
//    solver's RHS -- confirmed to have no edge-case branching at all, so
//    marking *any* cell, including a domain-edge one, Dirichlet-zero
//    works exactly as it would anywhere else in the domain).
// 2. Velocity and 3. scalar-field cleanup are NOT reproduced in this file
//    -- see grid_outflow_solver2.js, a proper small solver in its own
//    right (not a one-off helper) for the other two-thirds of what an
//    outflow object actually does once dispatched every frame.
//
// A nice emergent generalization worth noting: because inflow/outflow are
// SDF-shaped here (not wall-only, unlike mantaflow's own openBound
// string), inflow can represent things mantaflow's own literal mechanism
// can't -- e.g. a circular "fountain" source in the middle of the domain,
// not just a wall. Outflow, by physical necessity, still only makes sense
// at/overlapping the actual domain boundary (there's no cells beyond the
// array for fluid to go to otherwise) -- but can now be an arbitrary-
// shaped *portion* of a wall, not "the whole wall or nothing".

import { vec2, float } from 'three/tsl';
import { createSDFStaticCollider2 } from './sdf_collider2.js';
import { isInsideSdf } from './level_set_utils.js';

// resolutionX/Y, gridSpacingX/Y, originX/Y: forwarded as-is to
// createSDFStaticCollider2 -- an inflow's own grid must match the
// velocity/pressure grids it will be used against (same requirement a
// collider already has, see grid_blocked_boundary_condition_solver2.js's
// own buildBlockMarker, which indexes a collider by the same raw (i,j) as
// the velocity grid).
//
// options.velocity: [vx,vy] (a plain array, baked to a constant vec2 node
// at construction time) or a live vec2 node (e.g. built from two
// tsl_array_n.array0('float')s) -- same "number or node" flexible
// convention already used for dt elsewhere in this port. A plain-array
// value can't be changed after construction without rebuilding whatever
// kernel already captured it (same documented limitation as collider's
// own frictionCoefficient/velocityAt) -- pass a live node instead for a
// truly runtime-updatable inflow velocity.
//
// options.mode: 'set' (hard override -- matches mantaflow's own literal
// KnSetInflow behavior: the fluid at this inflow simply has its velocity
// forced to this value every frame) or 'add' (superimpose onto whatever
// velocity is already at that face -- an original extension beyond
// mantaflow's own literal mechanism, requested directly by the user: if
// an inflow object carries its own velocity, the fluid it emits should be
// able to inherit that velocity superimposed onto whatever velocity is
// already there, rather than always clamping it. Default 'set'.
//
// Both `velocity` and `mode` are plain mutable properties on the returned
// object (same spirit as collider's own `frictionCoefficient`) -- `mode`
// is read as a plain JS value every time grid_blocked_boundary_condition_
// solver2.js's own kernel-builder closure runs (at *construction* time of
// that kernel, not per-dispatch -- see that file's own comment on this),
// so reassigning it takes effect on the next call that (re)builds the
// affected kernels, not instantly on every future dispatch without one.
export function createSDFInflow2( resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY, options = {} ) {

	const sdf = createSDFStaticCollider2( resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY );
	const { velocity = [ 0, 0 ], mode = 'set' } = options;
	const velocityNode = Array.isArray( velocity ) ? vec2( velocity[ 0 ], velocity[ 1 ] ) : velocity;

	return { ...sdf, velocity: velocityNode, mode };

}

// Same SDF machinery as a collider, semantically distinct (an open
// boundary fluid can leave through, not a solid obstacle) -- no extra
// properties needed today; kept as its own thin, separately-named wrapper
// (rather than callers using createSDFStaticCollider2 directly for this
// role) so call sites read as what they mean, even though today the two
// are structurally identical.
export function createSDFOutflow2( resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY ) {

	return createSDFStaticCollider2( resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY );

}

// Combines one or more outflow objects into a single (pos) => {active,
// target} function, suitable for createGridPressureSolver2's own
// `dirichlet` option (and grid_solver2.js's own `dirichlet`/`outflows`
// combination, see combineDirichlet below) -- see this file's own header
// comment for the full derivation of why this reproduces mantaflow's
// zero-pressure ghost boundary with zero pressure-solver changes.
export function createOutflowPressureDirichlet2( outflows, { target = 0 } = {} ) {

	const list = Array.isArray( outflows ) ? outflows : [ outflows ];
	const targetNode = typeof target === 'number' ? float( target ) : target;

	return function dirichlet( pos ) {

		let active = null;

		for ( const outflow of list ) {

			const inside = isInsideSdf( outflow.sample( pos ) );
			active = active ? active.or( inside ) : inside;

		}

		return { active, target: targetNode };

	};

}

// Combines two optional dirichlet(pos) functions (either may be omitted)
// into one -- used by grid_solver2.js to let an outflow-derived dirichlet
// (this file's own createOutflowPressureDirichlet2) coexist with a
// caller-supplied one (e.g. examples/13-interactive-pressure/'s own
// pointer-driven pressure vent). If both happen to mark the same cell
// active, `b`'s target wins (evaluated second) -- avoiding overlapping
// regions for well-defined behavior is the caller's own responsibility,
// not something arbitrated further here.
export function combineDirichlet( a, b ) {

	if ( ! a ) return b;
	if ( ! b ) return a;

	return function dirichlet( pos ) {

		const resultA = a( pos );
		const resultB = b( pos );

		return {
			active: resultA.active.or( resultB.active ),
			target: resultB.active.select( resultB.target, resultA.target )
		};

	};

}
