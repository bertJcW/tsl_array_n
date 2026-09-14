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
import { float, int, max, min, atomicAdd, If } from 'three/tsl';
import { createCellCenteredScalarGrid2 } from './grid_data2.js';
import { faceCenteredDivergenceAtCenter2 } from './grid_math.js';
import { createCopyKernel2 } from './array_utils.js';
import { createLaplacianOperator, createMultigridPreconditioner, createJacobiPreconditioner, createIdentityPreconditioner } from '../linalg/multigrid.js';
import { createPreconditionedConjugateGradientSolver } from '../linalg/linalg.js';
import { isNonFiniteOrAbove } from '../float_guards.js';
import { instrumentDispatch, timePhase } from '../profiling.js';

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
// options.preconditioner: 'multigrid' (default), 'jacobi' or 'none'. All
// three are built at construction and selected per solve, so they can be
// compared inside one run. **Use the default.** The other two are
// measurement instruments, not alternatives: measured paired, multigrid
// needs ~20x fewer iterations and ~12x less wall time, and on a liquid
// scene neither cheap arm converges at all within a sane iteration cap.
// multigrid.js carries the numbers. Mutable at runtime through the
// returned `settings`, which is what the paired comparison needs.
// options.tolerance/maxIterations/residualCheckInterval: forwarded to the
// underlying CG solve(). maxIterations is also mutable at runtime through
// `settings`, which is what pricing a CG iteration needs -- see its own
// comment there.
// underlying CG solve().
// options.batchIterations: submit the GPU-resident CG loop's dispatches as one
// batch per iteration instead of one submission per kernel -- see linalg.js's
// iteration-batch comment for the measurement (38.8 us per submission, fifteen
// submissions per iteration before, five after). Mutable at runtime through the
// returned `settings` object, because a constructor-time choice can only be
// compared across runs and these scenes drift. Default true.
// options.atomicScale: accepted and ignored, forwarded only so callers that
// still pass one keep working. The CG dot product no longer uses a
// fixed-point encoding at all -- see linalg.js's createDotReducer for why
// that knob had to go. Historically this was described as the fixed-point
// scale for the GPU atomic
// dot product (see linalg.js's DEFAULT_ATOMIC_DOT_SCALE comment). Exposed
// here because a pressure/divergence field's natural magnitude can be much
// smaller than the O(1) values that default was tuned against (a real
// closed-domain test case measured divergence in the 0.01-0.05 range) --
// too coarse a scale for the actual problem risks the dot products the CG
// solver depends on quantizing down toward 0 more often than they should,
// which isDegenerateDenominator's own guard now catches safely (stopping the solve
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
// linalg.js's isDegenerateDenominator). diagnostics.rejected: boolean, true
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
	preconditioner = 'multigrid',
	tolerance = 1e-5,
	maxIterations = 100,
	// How often the CG loop evaluates its true-residual stop test, in
	// iterations. 1 asks every iteration. Higher values are not a speedup --
	// see linalg.js's own comment for the measurement and why the default
	// stays at 1 -- but they are the instrument that priced the round trip,
	// and mutable at runtime via `settings` below so that pricing can be
	// done paired.
	residualCheckInterval = 1,
	// Compute alpha and beta on the GPU and read the loop's scalars back in
	// one trip per iteration instead of three -- see
	// solveWithGpuResidentScalars in linalg.js. On by default: measured
	// paired on example 15 (twice, phase swapped) at 1.13x and 1.24x with
	// the iteration count unchanged to within 1%, and verified to produce
	// the same answers, the same guard outcomes and the same stop reasons
	// as the host path in examples/05-preconditioned-conjugate-gradient/.
	// `false` restores the host path, which is what that example compares
	// against and what the guards are specified by. Mutable via `settings`
	// below, because comparing the two across separate runs of an unsteady
	// scene compares the scene to itself.
	gpuResidentScalars = true,
	// Submit the CG iteration's dispatches as one batch instead of one
	// submission per kernel. See linalg.js's iteration-batch comment for the
	// measurement: 38.8 us per submission on this hardware, fourteen
	// submissions per iteration before this, three after. Mutable at runtime
	// through `settings` so the two forms can be compared inside one run --
	// paired measurement is the only kind that means anything on these scenes.
	batchIterations = true,
	atomicScale,
	maxPlausiblePressure = DEFAULT_MAX_PLAUSIBLE_PRESSURE
} = {} ) {

	// The options that are safe to change between frames, exposed on the
	// returned object so they can be. Everything else here is structural --
	// it decides what gets allocated and which kernels get built, so it is
	// fixed once construction is done. `residualCheckInterval` only decides
	// when the CG loop asks whether it is finished, which is a policy the
	// next solve can answer differently without anything being rebuilt.
	//
	// The reason it is mutable at all is measurement: comparing two
	// scheduling policies across separate runs of an unsteady scene
	// compares the scene to itself as much as the policies (its pressure
	// problem gets harder as the wake develops, and the machine's clocks
	// drift under sustained load). Alternating the policy *within* one run
	// makes the comparison paired, and paired is the only kind that means
	// anything here.
	// maxIterations is here rather than only a constructor value because
	// capping the iteration count is the only way to vary it over a range
	// wide enough to price one. Its natural spread on a settled scene is
	// 8-12, and regressing step time on that gave an R-squared of 0.009 --
	// the noise is larger than the signal, so the observational form of
	// this measurement does not work and the interventional one needs the
	// cap to move inside a single run. A capped solve does not converge and
	// is not a correctness configuration; see the tolerance note above.
	const settings = { residualCheckInterval, gpuResidentScalars, batchIterations, preconditioner, maxIterations, checkBadCells: true };

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
	// *** All three preconditioners, chosen per solve ***
	//
	// Built together at construction for the same reason every other
	// comparable choice in this package is: comparing two across separate
	// runs of an evolving scene compares the scene to itself. The CG solver
	// is handed one function, which dispatches whichever `settings` names.
	//
	// The wrapper costs the cheap preconditioners one submission. An entry
	// in the CG iteration's batch joins it only if it exposes a
	// `computeNode`, and a selector cannot -- it does not know at resolve
	// time which kernel it will run. The multigrid V-cycle was already a
	// batch break (linalg.js says so at its batch definition), so it loses
	// nothing; Jacobi and identity are single kernels that could otherwise
	// have been batched, so they pay ~38.8 us per iteration they need not
	// have. That bias runs *against* the cheap options, which is the safe
	// direction for a measurement meant to decide whether they are worth
	// having at all.
	const preconditionerBuilders = {
		multigrid: createMultigridPreconditioner( shape, gridSpacing, { ...multigrid, dirichletMask, faceWeights: faceWeightAccessors } ),
		jacobi: createJacobiPreconditioner( shape, gridSpacing, { dirichletMask, faceWeights: faceWeightAccessors } ),
		none: createIdentityPreconditioner( shape )
	};

	if ( ! Object.prototype.hasOwnProperty.call( preconditionerBuilders, preconditioner ) ) {

		throw new Error( `grid_pressure_solver2: unknown preconditioner '${ preconditioner }' (expected ${ Object.keys( preconditionerBuilders ).join( ', ' ) }).` );

	}

	function applyPreconditioner( input, output ) {

		const built = {};

		for ( const name of Object.keys( preconditionerBuilders ) ) {

			built[ name ] = preconditionerBuilders[ name ]( input, output );

		}

		return function dispatchSelectedPreconditioner() {

			built[ settings.preconditioner ]();

		};

	}
	const cg = createPreconditionedConjugateGradientSolver( applyLaplacian, applyPreconditioner, b, pressureGrid.data, { atomicScale } );

	// Updated after every project()-dispatch below, for diagnostics -- cg.solve()
	// itself returns whether it actually converged (true residual < tolerance),
	// but that return value had nowhere to go before this (project()'s own
	// dispatcher is void); a caller logging b/pressure each frame (e.g.
	// examples/14-stable-fluids/'s own summarize()/fmt() readout) can now also
	// see whether the solve genuinely converged or bailed out early -- either
	// hitting maxIterations, or via linalg.js's own isDegenerateDenominator guard
	// (see that function's comment: the search direction ran into the
	// operator's null space, or the atomic dot product's fixed-point
	// quantization rounded a denominator down to 0 -- both stop the iteration
	// safely rather than risk Infinity/NaN, but neither is "true" convergence).
	// rejected: see dispatch()'s own use of maxPlausiblePressure below --
	// true whenever this project() call's own solve() looked bad enough
	// that its pressure update was discarded rather than trusted.
	const diagnostics = { converged: null, rejected: false, iterations: null, stoppedBy: null };

	// Last-resort circuit breaker: a snapshot of the last pressure field
	// that actually *passed* this solver's own bad-cell check, restored in
	// place of this frame's own result if that result looks implausible
	// (see dispatch()'s own use, below). Deliberately a plain copy, not
	// part of the CG solver itself -- this is a caller-level policy
	// decision (skip *this frame's* pressure update, keep simulating with
	// the last known-good one, rather than let a bad solve reach
	// velocity), not a numerical-method concern linalg.js itself should
	// own.
	//
	// *** "Last known-good", not "last frame's": a real, measured defect
	// this used to get wrong. ***
	//
	// The snapshot used to be taken at the *start* of dispatch(), before
	// the solve. That is one frame too early to be safe: a frame whose
	// solve returns NaN leaves NaN in pressureGrid, the next frame
	// snapshots that NaN as its "known-good" baseline, and from then on
	// every rejection restores NaN. The field is permanently poisoned and
	// nothing outside this file can clear it -- confirmed on real WebGPU
	// hardware with examples/26-dye-free-surface/, where a full scene
	// reset (particles, velocities, dye, velocity grid, and zeroing
	// pressureGrid itself with a kernel dispatch) still blew up inside ten
	// frames on 6 of 6 attempts, because this buffer still held the NaN
	// and the first rejection handed it straight back.
	//
	// Snapshotting only *after* the check passes makes the invariant
	// structural rather than incidental: every value ever written into
	// this buffer has been through countBadPressureCellsNow() and come
	// back clean, so restorePressure() cannot reintroduce a non-finite or
	// implausible value no matter what any individual solve does. The
	// initial zero fill below is the baseline for the case where no solve
	// has passed yet -- zero pressure means "no projection this frame",
	// which is a bounded, recoverable error, unlike a NaN.
	const pressureSnapshot = tsl_array_n.arrayN( 'float', shape );
	pressureSnapshot.fromArray( new Float32Array( shape.reduce( ( a, n ) => a * n, 1 ) ) );
	const snapshotPressure = instrumentDispatch( 'pressure-snapshot', createCopyKernel2( pressureGrid.data, pressureSnapshot, shape ) );
	const restorePressure = instrumentDispatch( 'pressure-restore', createCopyKernel2( pressureSnapshot, pressureGrid.data, shape ) );

	// Reliable (see maxPlausiblePressure's own comment on why a scalar
	// derived from linalg.js's atomic-int reduction isn't) bad-cell
	// detector: atomically counts cells that are non-finite or past
	// maxPlausiblePressure in magnitude. A plain atomicAdd of 0/1 flags,
	// not a value-weighted reduction like linalg.js's own dot product --
	// there's no float-to-fixed-point quantization step here for a
	// non-finite input to be silently laundered through.
	//
	// *** This check was inert for NaN, and that was this library's worst
	// bug. ***
	//
	// It was written as `x != x || abs(x) > limit`, the documented WGSL
	// replacement for the isnan()/isinf() that core WGSL dropped. Measured
	// on real hardware, BOTH halves return false for a NaN (float_guards.js
	// has the full table), so a solve that came back NaN counted as zero bad
	// cells, was declared good, and had its NaN pressure multiplied straight
	// into the velocity field by the correction step below. What the user
	// saw was a liquid simulating correctly for a couple of hundred frames
	// and then collapsing to a point between one frame and the next, with
	// converged/rejected both reporting healthy. isNonFiniteOrAbove is the
	// same intent expressed two ways that survive a NaN: a bit-pattern
	// exponent test, and a bound written as a negated "within range" rather
	// than an asserted "out of range".
	const badCountAccum = tsl_array_n.array0( 'int' );
	badCountAccum.node.toAtomic();

	const countBadPressureCells = tsl_array_n.kernel( shape, ( i, j ) => {

		const value = pressureGrid.data( i, j );
		const isBad = isNonFiniteOrAbove( value, maxPlausiblePressure );

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

		// Sum of `beta_face * target_neighbour / h^2` over this cell's own
		// Dirichlet neighbours -- the term the elimination in
		// multigrid.js's laplacianAt leaves behind. See its use below for
		// the full derivation. Returns exactly 0 when no neighbour is
		// masked, and when every target is 0, so no shipped scene's own
		// kernel graph changes value.
		function neighbourDirichletContribution( i, j ) {

			if ( ! dirichletMask ) return float( 0 );

			const invHx2 = 1 / ( gridSpacing[ 0 ] * gridSpacing[ 0 ] );
			const invHy2 = 1 / ( gridSpacing[ 1 ] * gridSpacing[ 1 ] );

			// Index clamping keeps the read in bounds; the `inBounds` flag is
			// what actually decides whether the term counts, so a clamped
			// out-of-domain read can never contribute. A missing neighbour is
			// a wall (Neumann), not a Dirichlet value.
			const term = ( ni, nj, inBounds, weight, invH2 ) => {

				const isDir = dirichletMaskField( ni, nj ).greaterThan( 0.5 );
				const contribution = weight.mul( dirichletTargetField( ni, nj ) ).mul( invH2 );

				return inBounds.and( isDir ).select( contribution, float( 0 ) );

			};

			const one = float( 1 );
			const lo = ( n ) => max( 0, n );
			const hiI = ( n ) => min( n, resolutionX - 1 );
			const hiJ = ( n ) => min( n, resolutionY - 1 );

			const wLower = ( axisU ) => {

				if ( ! faceWeights ) return one;
				return axisU ? faceWeights.u( i, j ) : faceWeights.v( i, j );

			};

			const wUpper = ( axisU ) => {

				if ( ! faceWeights ) return one;
				return axisU ? faceWeights.u( i.add( 1 ), j ) : faceWeights.v( i, j.add( 1 ) );

			};

			return term( lo( i.sub( 1 ) ), j, i.greaterThan( 0 ), wLower( true ), invHx2 )
				.add( term( hiI( i.add( 1 ) ), j, i.lessThan( resolutionX - 1 ), wUpper( true ), invHx2 ) )
				.add( term( i, lo( j.sub( 1 ) ), j.greaterThan( 0 ), wLower( false ), invHy2 ) )
				.add( term( i, hiJ( j.add( 1 ) ), j.lessThan( resolutionY - 1 ), wUpper( false ), invHy2 ) );

		}

		const dispatchBuildSystem = tsl_array_n.kernel( shape, ( i, j ) => {

			const divergence = faceCenteredDivergenceAtCenter2( input.dataU, input.dataV, pressureGrid.gridSpacing, i, j );

			if ( dirichletMask ) {

				const isDirichlet = dirichletMask( i, j );
				const target = dirichletTargetField( i, j );

				// *** A Dirichlet neighbour's known value has to be moved to
				// this row's RHS. It never was, and nothing noticed for as
				// long as every target in this port happened to be 0. ***
				//
				// multigrid.js's laplacianAt eliminates a masked neighbour by
				// substituting exactly 0 for its value (see that function's
				// own comment -- it does this to keep A symmetric, which PCG
				// requires, and that part is right). Eliminating a *known*
				// value from the left-hand side is only half of the standard
				// reduction though: the value has to reappear on the right.
				// For a fluid row, the true stencil contributes
				// `beta_face * p_neighbour / h^2` for each neighbour, so with
				// the neighbour eliminated the row solves the wrong equation
				// unless `beta_face * target / h^2` is subtracted from b.
				//
				// With target == 0 the correction is identically 0, which is
				// why every scene shipped so far was unaffected and why this
				// stayed hidden: the free surface pins air to 0, and the one
				// scene with a nonzero target (examples/13-interactive-
				// pressure/) only ever checked the pinned cell's own value,
				// never a neighbour's. It surfaced the moment
				// grid_flip_solver2.js started solving for the reduced
				// pressure, whose air targets are `-dt (g.x)` and therefore
				// vary with height: interior divergence stayed at exactly 0
				// while the free-surface cells came back with divergence up
				// to 10, i.e. precisely the rows with a Dirichlet neighbour.
				const dirichletNeighbourRhs = neighbourDirichletContribution( i, j );

				// -target, not target: multigrid.js's own masked row is
				// `-1 * p(I)` (laplacianAt/laplacianDiagonalAt, negated to
				// match this operator's own overall sign convention -- see
				// laplacianDiagonalAt's header comment for the real bug this
				// fixes), so this row's own b must be negated to match:
				// `-p(I) = -target` solves to the same `p(I) = target`.
				b( i, j ).assign( isDirichlet.select( target.negate(), divergence.sub( dirichletNeighbourRhs ) ) );

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
			if ( updateDirichletFields ) updateDirichletFields();
			dispatchBuildSystem();
			diagnostics.converged = await timePhase( 'pressure-cg-solve', () => cg.solve(
				tolerance, settings.maxIterations,
				settings.residualCheckInterval, settings.gpuResidentScalars, settings.batchIterations
			) );

			// Forwarded from the CG solver so a caller can see *why* a frame
			// was expensive without reaching into linalg.js -- the iteration
			// count is the number every performance question here turns out
			// to depend on.
			diagnostics.iterations = cg.state ? cg.state.iterations : null;
			diagnostics.stoppedBy = cg.state ? cg.state.stoppedBy : null;

			// *** Measurement instrument, not a feature ***
			//
			// This is a host round trip on every solve, outside the CG
			// loop, and the fixed cost of a solve turns out to be about as
			// expensive as all of its iterations put together (~9 ms
			// against ~10 ms on example 15), so the pieces of that fixed
			// cost are worth pricing individually.
			//
			// Setting this false disables the circuit breaker, which is
			// the guard that stops a NaN solve from reaching velocity --
			// see project-history.md, Debugging #2 and #4, for the two
			// separate occasions this project shipped a liquid that
			// collapsed because that guard was not working. It exists to
			// answer "what does the check cost?", and the answer may
			// justify checking periodically rather than every solve. It
			// does not justify not checking.
			diagnostics.rejected = settings.checkBadCells
				? ( await timePhase( 'pressure-badcells-read', () => countBadPressureCellsNow() ) ) > 0
				: false;

			// Restore the last known-good field, or -- this solve having
			// just been checked and passed -- become the last known-good
			// field. See pressureSnapshot's own comment for why the
			// snapshot happens here rather than before the solve.
			if ( diagnostics.rejected ) restorePressure();
			else snapshotPressure();

			dispatchCorrectU();
			dispatchCorrectV();

		};

	}

	// The preconditioner's own runtime-switchable settings (V-cycle
	// batching, coarsest-level form), surfaced next to this solver's so a
	// measurement has one place to reach for. See multigrid.js.
	settings.multigrid = preconditionerBuilders.multigrid.settings;

	return { project, pressure: pressureGrid, b, diagnostics, settings };

}
