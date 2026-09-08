// Pressure projection, read directly from jet/fluid-engine-dev's
// GridSinglePhasePressureSolver2 (grid_single_phase_pressure_solver2.h/.cpp)
// -- no Python source to port here at all (grid_solver2.py's own
// computePressure never got past an abstract hook, same situation as
// advection_solver2.js). See ../../THIRD-PARTY-NOTICES.md for exactly
// what carries over versus what's new here.
//
// What carries over from jet: the fluid/pinned two-way cell classification
// (jet has a third, "boundary"/solid category too -- not ported here, see
// below) and the overall per-cell stencil shape (an identity row for a
// pinned cell, a divergence-based row for a fluid cell, a pinned neighbor
// contributing to a fluid cell's row exactly like a normal fluid neighbor
// would).
//
// What's generalized beyond jet: jet's own "air" cells are hardcoded to
// exactly 0 (an open-boundary/atmosphere assumption); this generalizes
// that to an arbitrary caller-supplied target value per cell (true
// Dirichlet, not a single constant) -- see multigrid.js's own
// `dirichletMask` parameter (decision 3 in that file's header comment),
// which this file builds from a `(pos) => {active, target}` function, the
// same idiom external_force_solver2.js's `force: (pos) => vec2` already
// established. jet's own "boundary" (interior solid/collider) category is
// NOT ported -- matching multigrid.js's existing no-collider scope cut,
// this file only knows about "fluid" and "Dirichlet-pinned" cells, not a
// third "solid" one.
//
// *** A REAL SIGN FLIP, relative to jet's own literal code -- read this
// before touching either the RHS or the correction formula below ***
//
// jet's own `buildSingleSystem` builds the *negated* Laplacian as its `A`
// (row.center += invHSqr -- a positive diagonal, row.right -= invHSqr --
// a negative off-diagonal), pairs it with `b = +divergence(u*)`, and
// corrects with a PLUS: `u0(i+1,j) = u(i+1,j) + invH.x*(p(i+1,j)-p(i,j))`.
// This port's own multigrid.js `laplacianAt`, already shipped and used by
// examples 04-07, computes the *standard* (non-negated) textbook Laplacian
// instead (`(upper - 2*center + lower)/h^2`, confirmed by reading its body
// directly -- no negation anywhere). Reusing jet's own correction sign
// with *this* port's own `A` would silently un-project instead of
// project -- the solver would still run, just be wrong, with no error to
// notice it by.
//
// Re-deriving from scratch for this port's own `A = +Laplacian`: the
// pressure-projection equation `u = u* - grad(p)` requires
// `Laplacian(p) = divergence(u*)` -- the *same* sign jet itself writes for
// `b`, no change needed there. But the correction step needs the opposite
// sign from jet's literal code: `u = u* - grad(p)`, not `u* + grad(p)`.
// Verified concretely on a 3-cell 1D case (h=1): u*=[0,1,1,0] gives
// b=[1,0,-1]; solving Laplacian(p)=b with Neumann edges gives p=[0,1,2].
// Applying `u_new(k) = u*(k) - (p(k)-p(k-1))` gives u_new=[0,0,0,0] --
// exactly divergence-free, as required. Applying jet's own literal PLUS
// sign to this same (correctly-solved-for-this-port's-A) p instead gives
// u_new=[0,2,2,0], divergence [2,0,-2] -- not divergence-free. So: reuse
// jet's own `b` formula verbatim, but flip jet's own correction-step sign
// from + to -.
//
// The Dirichlet identity-row substitution itself is sign-agnostic (an
// identity row is an identity row regardless of the surrounding
// operator's sign) -- only the fluid-cell RHS and the correction step
// depend on which sign convention `A` uses, and both are handled above.
//
// *** options.faceWeights -- variable-density (two-phase) projection ***
//
// Added for grid_two_phase_flip_solver2.js; every pre-existing caller
// passes none and gets exactly the code path above, unchanged. With it,
// this file solves `div(beta grad(p)) = div(u*)` and corrects with
// `u = u* - beta grad(p)`, where `beta = rho_liquid/rho_face` is supplied
// as a pair of MAC face arrays `{ u, v }` (accessor functions, so the
// caller keeps ownership and can rebuild them every frame -- which a
// two-phase solver must, since its interface moves). beta = 1 everywhere
// reduces exactly to the constant-density form, and that is not merely
// true in the limit: the option genuinely emits no extra nodes when
// absent, so nothing about the shipped single-phase scenes shifts.
//
// The one thing a caller MUST get right is that a closed all-fluid domain
// (which two-phase always is -- there is no "air" Dirichlet region left
// once the air is itself simulated) makes this system singular: pure
// Neumann, pressure defined only up to an additive constant. The fix is
// the caller's, not this file's, and it needs no new machinery here --
// pinning a single cell via the existing `dirichlet` option IS mantaflow's
// own `fixPressure`/`zeroPressureFixing` treatment of exactly this case.
// See grid_two_phase_flip_solver2.js's own header comment.
//
// dt plays no role anywhere in this file, matching jet's own solve()
// (which marks its own timeIntervalInSeconds parameter UNUSED_VARIABLE):
// any dt-scaling was already baked into u* by whatever produced it
// upstream (external forces, advection). grid_solver2.js's own
// computePressure(dt) hook still receives dt for signature uniformity
// with the other stages -- it just never forwards it here.

import * as tsl_array_n from 'tsl_array_n';
import { float, int, abs, atomicAdd, If } from 'three/tsl';
import { createCellCenteredScalarGrid2 } from './grid_data2.js';
import { faceCenteredDivergenceAtCenter2 } from './grid_math.js';
import { createCopyKernel2 } from './array_utils.js';
import { createLaplacianOperator, createMultigridPreconditioner } from '../linalg/multigrid.js';
import { createPreconditionedConjugateGradientSolver } from '../linalg/linalg.js';

// Last-resort bound on a single pressure cell's own magnitude -- see
// dispatch()'s own use, below, for the full circuit-breaker this backs.
// Default (1e6) is astronomically larger than any physically meaningful
// pressure this port has produced in every scene checked so far --
// generous on purpose, this only needs to catch a cell that's actually
// run away, not bound normal physical variation.
//
// *** This check was tried first against cg.state.residualSquared (a
// scalar already computed inside solve() via linalg.js's own atomic-int
// dot-product reduction) instead of pressure itself -- confirmed on real
// hardware to be UNRELIABLE, worth recording so it isn't retried: that
// reduction quantizes every per-cell term to a fixed-point *integer*
// before summing (linalg.js's own buildAtomicDotKernel), and converting a
// NaN or Infinity float to an integer is not guaranteed to produce
// anything recognizable as "huge" -- confirmed directly: a frame where
// pressure had already gone 992/1024 non-finite still reported a
// perfectly ordinary-looking, small residualSquared the very next frame,
// because whatever integer NaN-to-int conversion produced on this specific
// GPU/backend happened to look mundane. A scalar derived *through* that
// same reduction can never be fully trusted to reveal what it's
// summarizing -- only a direct, per-cell check (this file's own
// notEqual-self NaN test, the standard WGSL idiom since core WGSL has no
// isnan()/isinf(), plus this magnitude bound for Infinity and any
// still-finite runaway) on the actual field is reliable. ***
//
// *** Made a per-instance option, NOT tightened as a shared global
// default, after a real near-miss found via real-hardware regression
// testing -- worth recording so this isn't retried ***
//
// A real cascade found in examples/19-fuel-fire/ (see that file's own
// header comment for the full story) prompted trying a much tighter
// hardcoded default (50) here, reasoning that "every healthy scene tested
// stayed under ~1" (this comment's own original wording) meant 50 was
// still generous. That reasoning was directly falsified by testing
// examples/16-karman-vortex-street/ with the tightened value: its own
// pressure legitimately reaches 500+ (a much stronger whole-domain
// continuous force than example 19's own gentler buoyancy), so it got
// rejected on literally every single frame, silently reverting pressure
// to its initial zero snapshot forever and leaving velocity essentially
// frozen -- a real, self-inflicted regression, not a false alarm. This
// mirrors linalg.js's own MAX_ALPHA_MAGNITUDE comment exactly (alpha's
// own healthy magnitude "genuinely depends on a caller's specific problem
// scale... there's no single universal healthy range") -- pressure turns
// out to share that same property, unlike velocity (see
// grid_blocked_boundary_condition_solver2.js's own MAX_VELOCITY_COMPONENT,
// confirmed empirically to stay consistent, ~28 peak, across every scene
// checked so far, so tightening *that* one globally was safe). The actual
// fix: this bound is now `options.maxPlausiblePressure` on
// createGridPressureSolver2, defaulting back to the original safe 1e6 for
// every scene that doesn't explicitly opt into something tighter --
// examples/19-fuel-fire/ passes its own real, verified-tight value (50)
// explicitly; every other scene keeps the safe default unchanged.
const DEFAULT_MAX_PLAUSIBLE_PRESSURE = 1e6;

// options.resolution/gridSpacing/origin: plain-number arrays, matching
// multigrid.js's own convention -- NOT grid_data2.js's node-based
// gridSpacing (FaceCenteredGrid2.gridSpacing is a vec2 *node*, with no way
// to read the original plain numbers back out of it, so the caller must
// supply them again here; see grid_solver2.js's own comment on this same
// wart).
// options.dirichlet: optional (pos) => { active: BoolNode, target: FloatNode }.
// A hand-written function, same idiom as external_force_solver2.js's
// `force` -- called once per cell, per project() dispatch (not just once
// at construction), so a live/moving region (e.g. driven by a pointer
// position uniform) is fully supported. Omit entirely for a pure
// zero-flux/Neumann domain (every cell solved as fluid, matching
// multigrid.js's own default when no mask is given at all).
// options.multigrid: forwarded as-is to createMultigridPreconditioner's
// own options (e.g. { numberOfLevels: 4 }).
// options.tolerance/maxIterations: forwarded to the underlying CG solve().
// options.atomicScale: forwarded to createPreconditionedConjugateGradientSolver
// as its own options.atomicScale -- the fixed-point scale for the GPU atomic
// dot product (see linalg.js's DEFAULT_ATOMIC_DOT_SCALE comment). Exposed
// here because a pressure/divergence field's natural magnitude can be much
// smaller than the O(1) values that default was tuned against (a real
// closed-domain test case measured divergence in the 0.01-0.05 range) --
// too coarse a scale for the actual problem risks the dot products the CG
// solver depends on quantizing down toward 0 more often than they should,
// which isDegenerateDot's own guard now catches safely (stopping the solve
// early) instead of letting it corrupt pressure with Infinity/NaN, but
// stopping early on every solve still means the pressure never actually
// converges -- tune this if diagnostics (e.g. examples/14-stable-fluids/'s
// own summarize()/fmt() readout) show that happening.
//
// Returns { project(inputVelocity, outputVelocity), pressure, b, diagnostics }.
// b: the divergence RHS field, exposed for diagnostics (its sum should be
// ~0 for a fully closed/all-Neumann domain -- see grid_solver2.js's own
// note on this). diagnostics.converged: boolean, updated after every
// project()-dispatch call -- see its own declaration below for what a
// `false` here can mean (not necessarily a bug on its own; see
// linalg.js's isDegenerateDot). diagnostics.rejected: boolean, true
// whenever this project() call's own circuit breaker discarded a
// pressure update that looked implausible (see dispatch()'s own use of
// maxPlausiblePressure) -- pressure keeps its last known-good value on
// such a frame instead.
// options.maxPlausiblePressure: see DEFAULT_MAX_PLAUSIBLE_PRESSURE's own
// comment for why this is a per-instance option, not a shared constant --
// a scene with an unusually strong force (or otherwise a genuinely larger
// natural pressure scale) should pass its own real, verified value here
// rather than rely on the generous default being tight enough to catch a
// runaway early.
export function createGridPressureSolver2( {
	resolution, gridSpacing, origin = [ 0, 0 ],
	dirichlet,
	faceWeights,
	multigrid = {},
	tolerance = 1e-5,
	maxIterations = 100,
	atomicScale,
	maxPlausiblePressure = DEFAULT_MAX_PLAUSIBLE_PRESSURE
} = {} ) {

	const [ resolutionX, resolutionY ] = resolution;
	const [ gridSpacingX, gridSpacingY ] = gridSpacing;
	const [ originX, originY ] = origin;
	const shape = resolution;

	const pressureGrid = createCellCenteredScalarGrid2( resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY );
	const b = tsl_array_n.arrayN( 'float', shape );

	let dirichletMaskField, dirichletTargetField, updateDirichletFields, dirichletMask;

	if ( dirichlet ) {

		dirichletMaskField = tsl_array_n.arrayN( 'float', shape );
		dirichletTargetField = tsl_array_n.arrayN( 'float', shape );

		updateDirichletFields = tsl_array_n.kernel( shape, ( i, j ) => {

			const pos = pressureGrid.dataPosition( i, j );
			const { active, target } = dirichlet( pos );

			dirichletMaskField( i, j ).assign( active.select( float( 1 ), float( 0 ) ) );
			dirichletTargetField( i, j ).assign( target );

		} );

		dirichletMask = ( i, j ) => dirichletMaskField( i, j ).greaterThan( 0.5 );

	}

	// options.faceWeights: { u, v } -- per-face inverse-density coefficients
	// for a variable-density (two-phase) projection, see this file's own
	// header comment. Passed straight through to multigrid.js's own
	// dimension-generic [axis] form: the u array's own (i,j) IS the lower-x
	// face of cell (i,j), the v array's own (i,j) IS the lower-y face, which
	// is exactly what that parameter is defined to mean.
	const faceWeightAccessors = faceWeights ? [ faceWeights.u, faceWeights.v ] : undefined;
	const applyLaplacian = createLaplacianOperator( shape, gridSpacing, { dirichletMask, faceWeights: faceWeightAccessors } );
	const applyPreconditioner = createMultigridPreconditioner( shape, gridSpacing, { ...multigrid, dirichletMask, faceWeights: faceWeightAccessors } );
	const cg = createPreconditionedConjugateGradientSolver( applyLaplacian, applyPreconditioner, b, pressureGrid.data, { atomicScale } );

	// Updated after every project()-dispatch below, for diagnostics -- cg.solve()
	// itself returns whether it actually converged (true residual < tolerance),
	// but that return value had nowhere to go before this (project()'s own
	// dispatcher is void); a caller logging b/pressure each frame (e.g.
	// examples/14-stable-fluids/'s own summarize()/fmt() readout) can now also
	// see whether the solve genuinely converged or bailed out early -- either
	// hitting maxIterations, or via linalg.js's own isDegenerateDot guard
	// (see that function's comment: the search direction ran into the
	// operator's null space, or the atomic dot product's fixed-point
	// quantization rounded a denominator down to 0 -- both stop the iteration
	// safely rather than risk Infinity/NaN, but neither is "true" convergence).
	// rejected: see dispatch()'s own use of maxPlausiblePressure below --
	// true whenever this project() call's own solve() looked bad enough
	// that its pressure update was discarded rather than trusted.
	const diagnostics = { converged: null, rejected: false };

	// Last-resort circuit breaker: a snapshot of pressure taken right
	// before every solve(), restored in place of this frame's own result
	// if that result looks implausible (see dispatch()'s own use, below).
	// Deliberately a plain copy, not part of the CG solver itself -- this
	// is a caller-level policy decision (skip *this frame's* pressure
	// update, keep simulating with the last known-good one, rather than
	// let a bad solve reach velocity), not a numerical-method concern
	// linalg.js itself should own.
	const pressureSnapshot = tsl_array_n.arrayN( 'float', shape );
	const snapshotPressure = createCopyKernel2( pressureGrid.data, pressureSnapshot, shape );
	const restorePressure = createCopyKernel2( pressureSnapshot, pressureGrid.data, shape );

	// Reliable (see maxPlausiblePressure's own comment on why a scalar
	// derived from linalg.js's atomic-int reduction isn't) bad-cell
	// detector: atomically counts cells that are either NaN (the standard
	// `x != x` WGSL idiom, since core WGSL dropped isnan()/isinf()) or
	// past maxPlausiblePressure in magnitude (catches +/-Infinity too,
	// since Infinity compares greater than any finite bound). A plain
	// atomicAdd of 0/1 flags, not a value-weighted reduction like
	// linalg.js's own dot product -- there's no float-to-fixed-point
	// quantization step here for a non-finite input to be silently
	// laundered through.
	const badCountAccum = tsl_array_n.array0( 'int' );
	badCountAccum.node.toAtomic();

	const countBadPressureCells = tsl_array_n.kernel( shape, ( i, j ) => {

		const value = pressureGrid.data( i, j );
		const isBad = value.notEqual( value ).or( abs( value ).greaterThan( maxPlausiblePressure ) );

		atomicAdd( badCountAccum(), isBad.select( int( 1 ), int( 0 ) ) );

	} );

	async function countBadPressureCellsNow() {

		badCountAccum.fromArray( new Int32Array( [ 0 ] ) );
		countBadPressureCells();
		const [ count ] = await badCountAccum.toArray();
		return count;

	}

	// input/output: FaceCenteredGrid2 (grid_data2.js) -- typically the same
	// grid passed twice (in place), safe because the correction step below
	// only ever reads/writes its own velocity index (a self-touch, the same
	// pattern already established as reliable in external_force_solver2.js's
	// .addAssign()) plus a cross-*field* read of pressureGrid -- no
	// neighbor-velocity read, unlike advection, so no race even in place.
	function project( input, output ) {

		if ( input.resolution.join() !== resolution.join() ) {

			throw new Error( `createGridPressureSolver2: project() input resolution [${ input.resolution }] does not match constructed resolution [${ resolution }].` );

		}

		const dispatchBuildSystem = tsl_array_n.kernel( shape, ( i, j ) => {

			const divergence = faceCenteredDivergenceAtCenter2( input.dataU, input.dataV, pressureGrid.gridSpacing, i, j );

			if ( dirichletMask ) {

				const isDirichlet = dirichletMask( i, j );
				const target = dirichletTargetField( i, j );

				// -target, not target: multigrid.js's own masked row is
				// `-1 * p(I)` (laplacianAt/laplacianDiagonalAt, negated to
				// match this operator's own overall sign convention -- see
				// laplacianDiagonalAt's header comment for the real bug this
				// fixes), so this row's own b must be negated to match:
				// `-p(I) = -target` solves to the same `p(I) = target`.
				b( i, j ).assign( isDirichlet.select( target.negate(), divergence ) );

				// seed x at newly-Dirichlet cells to their target *before*
				// solving -- essential for a live/moving region: without this,
				// a cell that just became Dirichlet this frame keeps CG's
				// warm-started (stale, solved-as-fluid) value as its initial
				// guess instead of the new target.
				If( isDirichlet, () => {

					pressureGrid.data( i, j ).assign( target );

				} );

			} else {

				b( i, j ).assign( divergence );

			}

		} );

		// correction, every *interior* face unconditionally -- no marker
		// check needed at all here (unlike jet's own applyPressureGradient,
		// which gates on markers(i,j)==kFluid): in this port's simplified
		// two-way (fluid/Dirichlet) scope, every interior face separates two
		// cells that both have a meaningful, correctly-solved pressure, so
		// correcting unconditionally is both simpler and still correct. Only
		// the domain-edge faces (k=0, k=resolution) are left untouched --
		// same "caller's responsibility at the edges" spirit as
		// advectFaceCentered2's own docs.
		const dispatchCorrectU = tsl_array_n.kernel( input.dataSizeU, ( k, j ) => {

			If( k.greaterThan( 0 ).and( k.lessThan( resolutionX ) ), () => {

				// u = u* - beta grad(p), not u = u* - grad(p): the SAME beta
				// this face contributed to the operator above. Correcting
				// with an unweighted gradient after solving a weighted
				// system leaves the result not divergence-free at all --
				// the two halves are one derivation, `div(beta grad(p)) =
				// div(u*)` is only the right equation to solve *because*
				// the correction that follows it is `beta grad(p)`.
				const gradient = pressureGrid.data( k, j ).sub( pressureGrid.data( k.sub( 1 ), j ) ).div( pressureGrid.gridSpacing.x );
				const weighted = faceWeights ? gradient.mul( faceWeights.u( k, j ) ) : gradient;
				output.dataU( k, j ).assign( input.dataU( k, j ).sub( weighted ) );

			} );

		} );

		const dispatchCorrectV = tsl_array_n.kernel( input.dataSizeV, ( i, k ) => {

			If( k.greaterThan( 0 ).and( k.lessThan( resolutionY ) ), () => {

				const gradient = pressureGrid.data( i, k ).sub( pressureGrid.data( i, k.sub( 1 ) ) ).div( pressureGrid.gridSpacing.y );
				const weighted = faceWeights ? gradient.mul( faceWeights.v( i, k ) ) : gradient;
				output.dataV( i, k ).assign( input.dataV( i, k ).sub( weighted ) );

			} );

		} );

		return async function dispatch() {

			// *** Last-resort circuit breaker, confirmed necessary on real
			// hardware: even with linalg.js's own beta/alpha robustness
			// guards (see that file's own header comments), a preconditioned
			// solve() call can still -- rarely, but confirmed on real
			// hardware across many thousand frames of a Dirichlet-masked
			// scene -- come back with an implausible residual without ever
			// tripping any single guard along the way (each individual
			// iteration can look locally reasonable while the solve as a
			// whole still ends up somewhere it shouldn't). Snapshotting
			// pressure before every solve and reverting to it whenever the
			// result looks implausible makes the *outward-facing* guarantee
			// unconditional -- this project() call can never hand a caller a
			// pressure field worse than the last known-good one, regardless
			// of what happens inside any single solve() call. The tradeoff:
			// one frame's pressure update is silently skipped (velocity gets
			// corrected against a one-frame-stale pressure gradient instead)
			// -- a minor, bounded inaccuracy, never a divergent one.
			snapshotPressure();

			if ( updateDirichletFields ) updateDirichletFields();
			dispatchBuildSystem();
			diagnostics.converged = await cg.solve( tolerance, maxIterations );

			diagnostics.rejected = ( await countBadPressureCellsNow() ) > 0;

			if ( diagnostics.rejected ) restorePressure();

			dispatchCorrectU();
			dispatchCorrectV();

		};

	}

	return { project, pressure: pressureGrid, b, diagnostics };

}
