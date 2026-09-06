// Geometric multigrid V-cycle, usable as an `applyPreconditioner` for
// createPreconditionedConjugateGradientSolver (see linalg.js) -- almost
// certainly what the Python `fluxflow` source's own header comment in
// linalg.py anticipated ("later, this will be extended into
// preconditioning version of matrix free cg... for example - multi-grid
// preconditioning cg"), and a much stronger preconditioner than the toy
// exact-diagonal Jacobi one in examples/05-preconditioned-conjugate-gradient/.
//
// The V-cycle structure, the restriction/correction transfer formulas, and
// the red-black relax formula are ported from jet/fluid-engine-dev (MIT
// license, Doyub Kim) -- a local copy at
// D:\OneDrive\02_library\cpp\jet\fluid-engine-dev was read directly for
// this (include/jet/mg.h, detail/mg-inl.h, fdm_mg_solver2.cpp,
// fdm_mg_linear_system2.cpp, fdm_gauss_seidel_solver2.cpp). See
// ../../THIRD-PARTY-NOTICES.md for the attribution.
//
// Three real differences from jet's own MGPCG, all deliberate:
//
// 1. **Constant-coefficient, no collider** (a real scope cut, not an
//    oversight). jet's version is variable-coefficient and
//    collider-aware: every cell's stencil coefficients are baked from the
//    actual domain/collider, at every grid level, via an explicit
//    per-cell matrix (`FdmMatrixRow2.center/right/up`). Building that (a
//    "construct the pressure matrix from a collider" step, plus
//    per-level rebuilding) doesn't exist anywhere in this port yet and
//    would be a substantially larger undertaking on its own. This file
//    instead targets the standard constant-coefficient Poisson/Laplacian
//    on a plain rectangular domain, with a fixed Neumann-like (zero-flux)
//    boundary treatment -- missing neighbor terms are simply dropped at
//    domain edges, exactly like `grid/grid_math.js`'s `scalarLaplacian2`
//    already does. Since there's no stored matrix at all, the "matrix" is
//    just this same stencil function, called wherever jet would read
//    `A(i,j).center/right/up` -- see `laplacianAt` below. The *diagonal*
//    genuinely does still vary near boundaries even in this
//    constant-coefficient case (a corner cell has fewer valid neighbors
//    than an interior one, so its true diagonal coefficient is smaller in
//    magnitude) -- `laplacianDiagonalAt` computes this per-cell rather
//    than assuming one constant value everywhere. An earlier version of
//    this file got this wrong (a single constant diagonal, applied even at
//    boundaries), which silently over-corrected every boundary cell on
//    every relax sweep and compounded into real divergence over many
//    iterations -- caught by an independent plain-JS reference
//    implementation of the same formula converging cleanly where the
//    buggy version didn't; see examples/06-multigrid-preconditioner/'s
//    header comment for the full story.
// 2. **Dimension-generic (1D/2D/3D), not 2D-only.** jet's restriction and
//    correction formulas are written specifically for 2 axes (a 4-tap
//    1/8-3/8-3/8-1/8 separable filter for restriction, a 2-tap bilinear
//    1/4-3/4 filter for correction, each applied as an explicit x/y outer
//    product). This generalizes both to the outer product of the same
//    per-axis 1D filter across however many axes `shape.length` has --
//    matching jet exactly in the 2D case (16 restriction taps, 4
//    correction taps), and extending the same idea to 1D (4 and 2 taps)
//    and 3D (64 and 8 taps). This mirrors how `createConjugateGradientSolver`/
//    `createPreconditionedConjugateGradientSolver` in linalg.js are
//    themselves dimension-generic rather than hardcoded to 2D.
// 3. **Optional Dirichlet (fixed-value) cells, generalized beyond jet's
//    own hardcoded-zero "air" concept.** Added for grid_pressure_solver2.js
//    (see that file's own header comment for the full derivation this
//    supports), via an optional trailing `dirichletMask` argument on
//    `laplacianAt`/`laplacianDiagonalAt` (and threaded through
//    `buildRelaxKernel`/`buildResidualKernel`/`createLaplacianOperator`/
//    `createMultigridPreconditioner`'s own `options.dirichletMask`) --
//    fully backward-compatible, every existing caller passes none. A
//    masked cell's row becomes the identity (`A@p = p`, diagonal `1`)
//    instead of the normal stencil -- unlike jet's own air cells (which
//    are simply omitted from a *neighbor's* off-diagonal term since their
//    hardcoded value is exactly 0), a neighbor here needs **no**
//    special-casing at all: its value is whatever's genuinely in `field`
//    at that index, which is correct regardless of whether that neighbor
//    happens to be a normal fluid cell or a pinned one, as long as the
//    pinned cell's own row keeps it correct. The mask itself is only ever
//    *evaluated* at the finest level (level 0) -- coarser levels have no
//    concept of it at all, an intentional scope cut (see decision 1: no
//    per-level operator storage to make a coarser "masked row" mean
//    anything). What that requires, confirmed the hard way via real-
//    hardware regression testing (see buildRestrictKernel's and
//    buildCorrectKernel's own header comments for the full investigation,
//    cross-checked against mantaflow's own mature multigrid solver,
//    source/multigrid.h/.cpp's GridMg, consulted specifically for this):
//    a masked cell's *own* row being fixed at level 0 is not, by itself,
//    sufficient for correctness elsewhere -- the coarse-oblivious restrict
//    and correct steps must additionally be told to leave a masked cell's
//    residual out of what gets sent *to* the coarse levels, and its
//    coarse-derived correction out of what gets written *back*, or the
//    coarse machinery (silently, slowly, over many frames of live
//    Dirichlet-region use) corrupts *fluid* cells near the mask, which
//    nothing downstream ever corrects. Only convergence *speed* for a
//    large Dirichlet region is still an accepted, un-optimized tradeoff of
//    the coarser levels never seeing it at all -- correctness at the only
//    level a caller ever reads no longer depends on that tradeoff.
//
// The V-cycle itself needs no reduction/dot-product anywhere (relax,
// residual, restrict, and correct are all local-stencil kernels) -- unlike
// the CG solvers in linalg.js, this file's `applyPreconditioner` dispatcher
// is fully synchronous, no GPU readback at all. That means this file
// doesn't hit this project's now-familiar atomic/WebGL2-fallback wall the
// way linalg.js's CG solvers do -- though a real bug (see
// `laplacianDiagonalAt`'s own comment) surfaced along the way regardless,
// while this dev sandbox's WebGL2 fallback was separately proving
// unreliable for verifying the fix. CONFIRMED correct on real WebGPU
// hardware: examples/06-multigrid-preconditioner/'s numberOfLevels:4
// V-cycle reduces the residual further than numberOfLevels:1's plain
// relax alone (the whole point of the coarse-grid correction, not just
// the relax step, actually working), and
// examples/07-multigrid-preconditioned-cg/'s full pipeline converges to
// the expected exact answer -- see both files' own header comments for
// the reported numbers.

import * as tsl_array_n from 'tsl_array_n';
import { float, If } from 'three/tsl';
import { buildElementwiseKernel } from './linalg.js';

function validateShape( shape ) {

	if ( shape.length < 1 || shape.length > 3 ) {

		throw new Error( `multigrid: only 1D/2D/3D shapes are supported, got ${ shape.length }D.` );

	}

}

function computeLevelShapes( shape, numberOfLevels ) {

	const divisor = 2 ** ( numberOfLevels - 1 );

	for ( const dim of shape ) {

		if ( dim % divisor !== 0 ) {

			throw new Error(
				`multigrid: shape [${ shape.join( ', ' ) }] is not divisible by 2^(numberOfLevels-1)=${ divisor } (numberOfLevels=${ numberOfLevels }).`
			);

		}

	}

	const levelShapes = [];

	for ( let level = 0; level < numberOfLevels; level ++ ) {

		levelShapes.push( shape.map( ( dim ) => dim / ( 2 ** level ) ) );

	}

	return levelShapes;

}

function computeLevelSpacings( gridSpacing, numberOfLevels ) {

	const levelSpacings = [];

	for ( let level = 0; level < numberOfLevels; level ++ ) {

		levelSpacings.push( gridSpacing.map( ( h ) => h * ( 2 ** level ) ) );

	}

	return levelSpacings;

}

// Laplacian of `field` at index `I`, boundary-clamped per axis exactly
// like grid/grid_math.js's scalarLaplacian2, generalized to however many
// axes `I.length` has. This *is* "A" for the constant-coefficient Poisson
// problem this file solves -- see decision 1 in the file header comment.
//
// dirichletMask (optional, see decision 3): a masked cell's own row is the
// identity (A@p = p) instead of the normal neighbor-sum stencil.
//
// *** A real, confirmed-on-real-hardware symmetry bug, found via
// grid_pressure_solver2.js's outflow feature (the first thing in this
// port to ever run a Dirichlet-masked system through the *real*,
// atomics-based CG solver to completion -- examples/06's own smoke test
// only exercises the multigrid V-cycle in isolation, which doesn't need a
// symmetric A at all; examples/13-interactive-pressure/'s own Dirichlet
// vent was never actually confirmed running on real hardware) ***
//
// The original version of this function special-cased *only* a masked
// cell's own row (an earlier comment here claimed "neighbors need no
// special-casing at all, since a Dirichlet neighbor's value is already
// correct wherever it's read"). That's true for a single relax/residual
// pass in isolation, but it makes the *matrix* asymmetric: a Dirichlet
// cell i's own row has every off-diagonal entry zeroed (A[i][j]=0 for
// every neighbor j), but neighbor j's own row was left reading i's live
// value normally -- i.e. A[j][i] stayed nonzero. A[i][j] != A[j][i] means
// A is not symmetric, and createPreconditionedConjugateGradientSolver
// (linalg.js) is a textbook symmetric-CG implementation start to finish
// -- its whole derivation (conjugate search directions, p^TAp as a
// well-defined "energy") assumes A=A^T. Feeding it an asymmetric operator
// has no convergence guarantee at all: confirmed on real WebGPU hardware
// as an immediate (frame-0), 100%-pressure-non-finite blowup that never
// converged, even after 900 frames of warm-starting -- and confirmed,
// by disabling every other new mechanism in examples/15-flow-past-cylinder/
// one at a time (no collider, no outflow velocity extrapolation), that
// only the Dirichlet-masked pressure system itself reproduces it.
//
// Fixed the standard way a Dirichlet degree of freedom is eliminated from
// a symmetric linear system: a normal cell's own row must ALSO stop
// referencing a Dirichlet neighbor as a free variable (matching that
// neighbor's own row already excluding *this* cell) -- but unlike a
// genuine domain edge (where a missing neighbor means "no flux", i.e. the
// whole term is 0), a Dirichlet neighbor's *known* value doesn't vanish
// from the physics, it just isn't a free variable the operator solves
// for -- so its own contribution is treated as exactly 0 for the
// operator's purposes (a homogeneous/zero-Dirichlet coefficient), while
// the cell's own diagonal is left completely unchanged (laplacianDiagonalAt,
// below, already only excludes genuine domain edges from the diagonal
// count, never dirichletMask -- that was already correct and needed no
// change). The neighbor's *actual* (possibly nonzero) target value
// belongs on the right-hand side `b` instead, not in this operator --
// see grid_pressure_solver2.js's own dispatchBuildSystem. For outflow's
// own use (target always exactly 0), that RHS term is 0 too, so this
// operator-side fix alone is fully sufficient; a future nonzero-target
// Dirichlet use (e.g. examples/13's own pressure vent) would additionally
// want that RHS correction for full physical accuracy, though the
// symmetry (and therefore CG's basic stability) no longer depends on it.
function laplacianAt( field, spacing, shape, I, dirichletMask ) {

	const center = field( ...I );
	const zero = float( 0 );
	let sum = null;

	for ( let axis = 0; axis < I.length; axis ++ ) {

		const idx = I[ axis ];
		const n = shape[ axis ];

		const lower = I.map( ( v, a ) => ( a === axis ? idx.sub( 1 ) : v ) );
		const upper = I.map( ( v, a ) => ( a === axis ? idx.add( 1 ) : v ) );

		let dLower = idx.greaterThan( 0 ).select( center.sub( field( ...lower ) ), zero );
		let dUpper = idx.lessThan( n - 1 ).select( field( ...upper ).sub( center ), zero );

		if ( dirichletMask ) {

			// center.sub(field(lower)) with field(lower) treated as exactly
			// 0 is just `center`; field(upper).sub(center) with
			// field(upper) treated as exactly 0 is `center.negate()` --
			// these are NOT the same substitution (dLower/dUpper have
			// opposite sign conventions relative to `center`), so each
			// needs its own, deliberately-not-shared expression here.
			const lowerIsDirichlet = idx.greaterThan( 0 ).and( dirichletMask( ...lower ) );
			const upperIsDirichlet = idx.lessThan( n - 1 ).and( dirichletMask( ...upper ) );

			dLower = lowerIsDirichlet.select( center, dLower );
			dUpper = upperIsDirichlet.select( center.negate(), dUpper );

		}

		const term = dUpper.sub( dLower ).div( spacing[ axis ] * spacing[ axis ] );
		sum = sum === null ? term : sum.add( term );

	}

	// *** A real, confirmed-on-real-hardware sign bug: this must be
	// `center.negate()`, not `center` -- see laplacianDiagonalAt's own
	// header comment for the full derivation; kept in sync with it here
	// since the two must always agree on which sign convention a masked
	// row uses. ***
	return dirichletMask ? dirichletMask( ...I ).select( center.negate(), sum ) : sum;

}

// Diagonal of the Laplacian stencil above, *at* index I -- genuinely
// position-dependent, not a single constant, despite the constant
// coefficients (decision 1): a boundary cell has fewer valid neighbors
// than an interior one (e.g. a corner in 2D has 2, not 4), so its true
// diagonal coefficient is smaller in magnitude. Using the interior
// diagonal there anyway (an earlier version of this file did exactly
// that) silently over-corrects every boundary cell on every relax sweep
// -- compounds into real divergence over many iterations, not just a
// slow-to-converge result. Mirrors laplacianAt's own per-axis
// hasLower/hasUpper masking exactly, so the two stay consistent by
// construction.
//
// *** A real, confirmed-on-real-hardware sign bug, found via direct
// comparison against mantaflow's own multigrid solver (see
// buildRestrictKernel's header comment for that whole investigation) --
// this file's own preconditioned CG solver (linalg.js) requires its
// operator `A` to be consistently signed (every row's own quadratic form
// must point the same way) for `r.z` to behave like a valid SPD
// preconditioner's output -- textbook CG/PCG relies on this for `alpha`/
// `beta` to remain well-behaved. This function's own unmasked branch
// (interior/edge cells, see the loop above) always returns a *negative*
// diagonal (e.g. -4 in 2D at unit spacing, matching the standard
// `(upper-2*center+lower)/h^2` finite-difference Laplacian, confirmed by
// laplacianAt's own unmasked formula having no negation anywhere) -- a
// masked row previously returned a *positive* `1` here instead, a sign
// flip relative to every other row in the same matrix. Confirmed via a
// pure-JS debug log of the preconditioned CG solver's own r.z (linalg.js's
// window.__pcgLog instrumentation): with the previous, wrong-signed `+1`,
// r.z went negative from the *very first* solve() call of a scene with
// any Dirichlet mask active -- mathematically impossible for a genuinely
// SPD preconditioner (r.(M^-1 r) > 0 for any nonzero r), and a direct,
// unambiguous signature of exactly this kind of sign inconsistency. Left
// alone, this eventually (confirmed at frame 82 in a minimal repro) grows
// enough that the fixed-point atomic dot-product accumulator (linalg.js,
// int32-based) overflows on a single dot-product call, corrupting the
// entire scalar reduction it depends on and cascading into a fully
// non-finite pressure field within a few more frames.
//
// Fixed by using `-1` instead of `1` here (and laplacianAt's own masked
// branch negating `center` to match) -- algebraically this still solves
// to exactly the same `p(I) = target` (both sides of the row's equation
// are just negated together, see grid_pressure_solver2.js's own matching
// sign flip on its Dirichlet-row `b`), but now with the *sign* every other
// row in this same operator already uses. dirichletMask: see laplacianAt
// -- a masked cell's own diagonal is the identity's own, `-1` (matching
// this operator's own overall sign convention, not literally `1`).
//
// Deliberately does NOT also exclude a Dirichlet-masked *neighbor* here,
// unlike laplacianAt's own dLower/dUpper (see that function's own,
// extensive comment on the symmetry fix): a genuine domain edge means a
// neighbor slot is structurally *absent* (this cell has fewer physical
// couplings, so its diagonal magnitude is genuinely smaller), but a
// Dirichlet neighbor's slot is still physically *present* -- its value is
// simply already known rather than solved for. Eliminating that known
// value from the operator (laplacianAt's own fix) must not also shrink
// this cell's own diagonal, or the two would no longer represent the same
// (merely reduced-to-free-variables) linear system. This was already
// correct before the symmetry fix above and needed no change.
function laplacianDiagonalAt( spacing, shape, I, dirichletMask ) {

	let sum = null;

	for ( let axis = 0; axis < I.length; axis ++ ) {

		const idx = I[ axis ];
		const n = shape[ axis ];
		const hSq = spacing[ axis ] * spacing[ axis ];

		const hasLower = idx.greaterThan( 0 ).select( float( 1 ), float( 0 ) );
		const hasUpper = idx.lessThan( n - 1 ).select( float( 1 ), float( 0 ) );

		const term = hasLower.add( hasUpper ).mul( - 1 / hSq );
		sum = sum === null ? term : sum.add( term );

	}

	return dirichletMask ? dirichletMask( ...I ).select( float( -1 ), sum ) : sum;

}

// Sum of all indices mod 2 -- a checkerboard coloring valid for any
// dimension count (no two cells adjacent along any single axis share a
// color), generalizing jet's 2D `(i+j)%2` red-black split.
function colorOf( I ) {

	let sum = I[ 0 ];
	for ( let axis = 1; axis < I.length; axis ++ ) sum = sum.add( I[ axis ] );
	return sum.mod( 2 );

}

// applyOperator(input, output)-shaped: computes output = Laplacian(input).
// Exported so callers (e.g. examples/07-multigrid-preconditioned-cg/,
// grid_pressure_solver2.js) can use the *exact same* stencil for the outer
// CG loop's `A` that this file's own relax/residual steps use internally --
// essential for the preconditioner to actually approximate the inverse of
// the system CG is solving, not a different one. options.dirichletMask:
// see decision 3 in the file header comment.
export function createLaplacianOperator( shape, gridSpacing, options = {} ) {

	const { dirichletMask } = options;

	return function applyLaplacian( input, output ) {

		return buildElementwiseKernel( shape, ( I ) => {

			output( ...I ).assign( laplacianAt( input, gridSpacing, shape, I, dirichletMask ) );

		} );

	};

}

// One red-black SOR relax pass over a single color (0 or 1). Reads the
// full current `x`, but only *writes* cells matching `color` -- since a
// cell's neighbors are always the opposite color on a proper checkerboard,
// this is safe to run as one dispatch per color with no data race,
// without needing a stride-2 dispatch primitive tsl_array_n doesn't have.
function buildRelaxKernel( shape, spacing, sorFactor, color, x, b, dirichletMask ) {

	return buildElementwiseKernel( shape, ( I ) => {

		const isColor = colorOf( I ).equal( color );
		const Ax = laplacianAt( x, spacing, shape, I, dirichletMask );
		const diagonal = laplacianDiagonalAt( spacing, shape, I, dirichletMask );
		const current = x( ...I );
		const updated = current.add( b( ...I ).sub( Ax ).div( diagonal ).mul( sorFactor ) );

		x( ...I ).assign( isColor.select( updated, current ) );

	} );

}

// buffer = b - A@x, the true residual -- reused by the V-cycle before
// restricting down to the next coarser level.
function buildResidualKernel( shape, spacing, x, b, buffer, dirichletMask ) {

	return buildElementwiseKernel( shape, ( I ) => {

		buffer( ...I ).assign( b( ...I ).sub( laplacianAt( x, spacing, shape, I, dirichletMask ) ) );

	} );

}

function buildZeroKernel( shape, field ) {

	return buildElementwiseKernel( shape, ( I ) => {

		field( ...I ).assign( 0 );

	} );

}

// The 4 (index, weight) taps of jet's separable 1/8-3/8-3/8-1/8 restriction
// filter along one axis, boundary-clamped exactly like jet's C++
// (`(c>0)?2c-1:2c`, `(c+1<n)?2c+2:2c+1`). Weights are fixed JS numbers
// (position-independent), only the *indices* depend on the (node-valued)
// coarse index `coarseIdx`.
function restrictionTapsForAxis( coarseIdx, coarseCount ) {

	const idx0 = coarseIdx.greaterThan( 0 ).select( coarseIdx.mul( 2 ).sub( 1 ), coarseIdx.mul( 2 ) );
	const idx1 = coarseIdx.mul( 2 );
	const idx2 = coarseIdx.mul( 2 ).add( 1 );
	const idx3 = coarseIdx.add( 1 ).lessThan( coarseCount ).select( coarseIdx.mul( 2 ).add( 2 ), coarseIdx.mul( 2 ).add( 1 ) );

	return [
		{ index: idx0, weight: 0.125 },
		{ index: idx1, weight: 0.375 },
		{ index: idx2, weight: 0.375 },
		{ index: idx3, weight: 0.125 },
	];

}

// coarser = restrict(finer): full-weighting restriction, generalizing
// jet's 2D 4x4=16-tap formula to the outer product of the same 4-tap 1D
// filter across all of `coarseShape.length` axes (4 taps in 1D, 16 in 2D
// matching jet exactly, 64 in 3D). All combinatorics happen in plain JS
// at kernel-*build* time (looping over `coarseShape.length` and over each
// axis's 4 taps) -- the generated kernel body is just a flat weighted sum.
//
// *** The actual root cause of this file's own long-run divergence bug
// with a Dirichlet mask (see buildCorrectKernel's header comment for the
// investigation that led here, and grid_pressure_solver2.js's own header
// comment for the wider context) -- confirmed against mantaflow's own,
// independently-developed multigrid solver (source/multigrid.h/.cpp,
// GridMg, Apache-2.0), consulted specifically for how a mature,
// battle-tested implementation handles this exact class of problem ***
//
// mantaflow's GridMg calls a Dirichlet-like fixed-value row (diagonal 1,
// every off-diagonal 0, i.e. literally `x_i = b_i`) a "trivial equation",
// and its own header comment (multigrid.h's setTrivialEquationScale) says
// plainly: "Trivial equations... can have a negative effect on the coarse
// grid operators of the multigrid hierarchy (due to scaling mismatches),
// which can lead to slow multigrid convergence" -- its fix is to scale
// both that row's diagonal *and* its own right-hand side down by a small
// shared factor (1e-6 by default) before the coarse levels are built from
// it, specifically so this cell's contribution can't inject a badly-
// scaled quantity into the coarse-grid representation of the problem.
//
// This file's own dirichletMask row (laplacianAt/laplacianDiagonalAt) is
// exactly mantaflow's "trivial equation" -- but unlike GridMg (which
// derives every coarse-level operator *from* the fine one via Galerkin
// coarsening, `A_l = R*A_{l-1}*I`, so a scaled-down fine row automatically
// produces a consistently-scaled-down coarse contribution), this file's
// coarse levels are independently re-discretized at each level's own
// spacing (decision 1 above) -- there is no coarse operator for a scaled
// fine-level row to consistently propagate *into*. A direct port of
// mantaflow's own scale-the-row trick has nothing to attach to here.
//
// What *does* carry over is mantaflow's underlying diagnosis: a Dirichlet
// cell's own residual (this file's buildResidualKernel, at level 0,
// already dirichletMask-aware: `b(I) - laplacianAt(...) = target - x(I)`)
// is not a flux-imbalance quantity like every other cell's residual --
// it's "how far this pinned cell still is from its target", a completely
// different quantity, on a completely different scale, that the coarse
// operator (which has no idea a mask even exists at any level > 0) cannot
// meaningfully interpret. The previous code here restricted it in anyway,
// unconditionally averaged alongside genuine fluid-cell residuals via the
// weights below -- confirmed via real-hardware regression to feed the
// coarse solve a contaminated RHS near every Dirichlet cell, producing a
// coarse-grid correction that's wrong not just *at* that cell (which the
// next relax pass fixes regardless, see buildCorrectKernel) but at its
// *fluid* neighbors too (which nothing else ever corrects). Confirmed as
// the actual mechanism: numberOfLevels:1 (this restrict/correct pipeline
// never runs at all) is long-run stable; numberOfLevels:4 (this pipeline
// runs every V-cycle) reproducibly diverges within ~80-90 frames on an
// otherwise-identical scene.
//
// Fixed the direct way, translating mantaflow's own intent (don't let a
// trivial/Dirichlet row's own value contaminate the coarse
// representation) into this file's own architecture (no per-level
// operator to scale, so nothing to attach a "scale" to -- substitute a
// hard 0 for the contaminating quantity at the one place it actually
// enters the coarse problem: right here, at restriction). A Dirichlet
// cell has no need to participate in the coarse-grid correction loop at
// all -- level 0's own relax already pins it to `target` every single
// sub-pass, unconditionally, regardless of any coarser-level input.
// dirichletMask: only ever meaningful restricting *from* level 0 (see
// createMultigridPreconditioner's own construction loop) -- matches the
// "finest level only" scope already established throughout this file.
function buildRestrictKernel( finer, coarser, coarseShape, dirichletMask ) {

	return buildElementwiseKernel( coarseShape, ( I ) => {

		let combos = [ { indices: [], weight: 1 } ];

		for ( let axis = 0; axis < I.length; axis ++ ) {

			const taps = restrictionTapsForAxis( I[ axis ], coarseShape[ axis ] );
			const next = [];

			for ( const combo of combos ) {

				for ( const tap of taps ) {

					next.push( { indices: [ ...combo.indices, tap.index ], weight: combo.weight * tap.weight } );

				}

			}

			combos = next;

		}

		let sum = null;

		for ( const combo of combos ) {

			const rawValue = finer( ...combo.indices );
			const value = dirichletMask ? dirichletMask( ...combo.indices ).select( float( 0 ), rawValue ) : rawValue;
			const term = value.mul( combo.weight );
			sum = sum === null ? term : sum.add( term );

		}

		coarser( ...I ).assign( sum );

	} );

}

// The 2 (index, weight) taps of jet's bilinear 1/4-3/4 correction filter
// along one axis, based on the fine index's parity, boundary-clamped like
// jet's C++. Unlike restriction, the weights themselves depend on parity
// (a per-thread, node-valued condition, not known at kernel-build time),
// so both index *and* weight are returned as nodes here.
function correctionTapsForAxis( fineIdx, coarseCount ) {

	const ci = fineIdx.div( 2 );
	const isEven = fineIdx.mod( 2 ).equal( 0 );

	const lowerIdx = ci.greaterThan( 0 ).select( ci.sub( 1 ), ci );
	const upperIdx = ci.lessThan( coarseCount - 1 ).select( ci.add( 1 ), ci );

	return [
		{ index: isEven.select( lowerIdx, ci ), weight: isEven.select( float( 0.25 ), float( 0.75 ) ) },
		{ index: isEven.select( ci, upperIdx ), weight: isEven.select( float( 0.75 ), float( 0.25 ) ) },
	];

}

// finer += correct(coarser): bilinear prolongation, generalizing jet's 2D
// 2x2=4-tap formula to the outer product of the same 2-tap 1D filter
// across all of `fineShape.length` axes (2 taps in 1D, 4 in 2D matching
// jet exactly, 8 in 3D). An *additive* update (matching jet's own
// `(*finer)(i,j) +=`), since the fine level's `x` already holds its
// pre-correction value from the relax pass before the recursive call.
//
// *** A real, confirmed-on-real-hardware long-run divergence bug: this
// correction must NOT touch a Dirichlet-pinned cell at the finest level ***
//
// The coarser levels have no idea a `dirichletMask` even exists (see this
// file's own header comment, decision 3) -- they solve a plain, mask-
// unaware problem and hand back a "correction" for every fine cell,
// including ones that are actually Dirichlet-pinned at level 0. Applying
// that correction unconditionally (the original behavior here) knocks a
// pinned cell off its own target value; the header comment's own original
// reasoning was that the very next level-0 post-correction relax pass
// (which *is* mask-aware) snaps it straight back, so correctness at the
// only level a caller ever reads was assumed unaffected, only convergence
// *speed* for a large Dirichlet region.
//
// That reasoning has a real gap, confirmed by direct real-hardware
// experiment (a static Dirichlet-zero vent added to the otherwise-plain
// examples/14-stable-fluids/ scene, isolating this from any outflow-
// object complexity): red-black relax processes color 0 before color 1
// every pass. A pinned cell whose own color happens to run *after* one of
// its fluid neighbors' own color -- i.e. the neighbor's relax pass reads
// the pinned cell's *pre-snap-back*, correction-polluted value, not its
// target -- can feed that polluted value into the neighbor's own update
// before the pinned cell itself gets corrected later in the same pass
// sequence. `numberOfFinalIterations`'s default of 2 sweeps doesn't fully
// launder this out every single frame; with the same Dirichlet region
// persisting call after call (as any live scene's own outflow/vent does),
// a small residual error compounds frame over frame until it diverges --
// confirmed on real hardware to first appear within roughly 80-90 frames
// with a small Dirichlet region, `numberOfLevels: 4`; confirmed absent
// with `numberOfLevels: 1` (no coarse-grid correction at all, hence
// nothing for this bug to even apply to) over 1000+ frames of the exact
// same scene, isolating the coarse-grid correction step specifically
// (not the Dirichlet mask itself, not CG, not the outflow mechanism) as
// the real, root cause.
//
// Fixed the direct way: `dirichletMask`, when given, gates this kernel's
// own write -- a pinned finest-level cell simply never receives a
// coarse-grid correction at all, so it can never be knocked off target in
// the first place, and there's nothing left for the post-relax pass to
// need to "launder out" every frame. Only ever passed for the correction
// step that writes *into* level 0 (see this file's own
// createMultigridPreconditioner, where `correctDispatchers[0]` is the
// only one built with a mask -- every coarser-to-coarser correction has
// no mask concept to begin with, per decision 3's own scope).
function buildCorrectKernel( coarser, finer, fineShape, dirichletMask ) {

	return buildElementwiseKernel( fineShape, ( I ) => {

		let combos = [ { indices: [], weight: float( 1 ) } ];

		for ( let axis = 0; axis < I.length; axis ++ ) {

			const taps = correctionTapsForAxis( I[ axis ], fineShape[ axis ] / 2 );
			const next = [];

			for ( const combo of combos ) {

				for ( const tap of taps ) {

					next.push( { indices: [ ...combo.indices, tap.index ], weight: combo.weight.mul( tap.weight ) } );

				}

			}

			combos = next;

		}

		let sum = null;

		for ( const combo of combos ) {

			const term = coarser( ...combo.indices ).mul( combo.weight );
			sum = sum === null ? term : sum.add( term );

		}

		if ( dirichletMask ) {

			If( dirichletMask( ...I ).not(), () => {

				finer( ...I ).addAssign( sum );

			} );

		} else {

			finer( ...I ).addAssign( sum );

		}

	} );

}

// shape, gridSpacing: arrays of equal length (1-3), the *finest* level.
// `shape` must be exactly divisible by 2^(numberOfLevels-1) in every
// dimension (matches jet's own constraint, documented in
// FdmMgLinearSystem2::resizeWithFinest) -- validated upfront.
// options.numberOfLevels: default 1 (a single level, i.e. plain red-black
// SOR relaxation with no coarsening at all -- always valid regardless of
// shape, since divisibility by 2^0=1 is automatic; pass a larger value
// once your grid size actually supports it for real multigrid behavior).
// options.numberOfSmoothingIterationsDown/Up: relax iterations before/after
// the recursive coarse-grid correction (jet's "restriction"/"correction"
// iteration counts), default 2.
// options.numberOfCoarsestIterations: relax iterations at the coarsest
// level, meant to substitute for an exact solve there, default 20.
// options.numberOfFinalIterations: relax iterations after the *final*
// (level-0) correction specifically, default 2 (jet distinguishes this
// from the other levels' correction-iteration count; see mg-inl.h).
// options.sorFactor: SOR over-relaxation factor, default 1.0 (plain
// Gauss-Seidel, no over-relaxation).
// options.dirichletMask: see decision 3 in the file header comment --
// applied at the finest level (level 0) only, never coarsened.
//
// Returns an applyPreconditioner-compatible (input, output) => dispatcher
// -- pass directly as createPreconditionedConjugateGradientSolver's
// applyPreconditioner argument.
export function createMultigridPreconditioner( shape, gridSpacing, options = {} ) {

	validateShape( shape );

	if ( gridSpacing.length !== shape.length ) {

		throw new Error( `multigrid: gridSpacing length (${ gridSpacing.length }) must match shape length (${ shape.length }).` );

	}

	const numberOfLevels = options.numberOfLevels ?? 1;
	const numberOfSmoothingIterationsDown = options.numberOfSmoothingIterationsDown ?? 2;
	const numberOfSmoothingIterationsUp = options.numberOfSmoothingIterationsUp ?? 2;
	const numberOfCoarsestIterations = options.numberOfCoarsestIterations ?? 20;
	const numberOfFinalIterations = options.numberOfFinalIterations ?? 2;
	const sorFactor = options.sorFactor ?? 1.0;
	const dirichletMask = options.dirichletMask;

	const levelShapes = computeLevelShapes( shape, numberOfLevels );
	const levelSpacings = computeLevelSpacings( gridSpacing, numberOfLevels );

	return function applyMultigridPreconditioner( input, output ) {

		const levels = [];

		for ( let level = 0; level < numberOfLevels; level ++ ) {

			const levelShape = levelShapes[ level ];
			const levelSpacing = levelSpacings[ level ];

			const x = level === 0 ? output : tsl_array_n.arrayN( 'float', levelShape );
			const b = level === 0 ? input : tsl_array_n.arrayN( 'float', levelShape );
			const buffer = tsl_array_n.arrayN( 'float', levelShape );
			const levelMask = level === 0 ? dirichletMask : undefined;

			levels.push( {
				x, b, buffer,
				relaxColor0: buildRelaxKernel( levelShape, levelSpacing, sorFactor, 0, x, b, levelMask ),
				relaxColor1: buildRelaxKernel( levelShape, levelSpacing, sorFactor, 1, x, b, levelMask ),
				residual: buildResidualKernel( levelShape, levelSpacing, x, b, buffer, levelMask ),
				zeroX: buildZeroKernel( levelShape, x ),
			} );

		}

		const restrictDispatchers = [];
		const correctDispatchers = [];

		for ( let level = 0; level < numberOfLevels - 1; level ++ ) {

			// dirichletMask only ever applies to the finest level (level 0)
			// -- these are the *only* restrict/correct calls that touch it
			// (see buildRestrictKernel's and buildCorrectKernel's own header
			// comments for why each needs it).
			restrictDispatchers.push( buildRestrictKernel( levels[ level ].buffer, levels[ level + 1 ].b, levelShapes[ level + 1 ], level === 0 ? dirichletMask : undefined ) );
			correctDispatchers.push( buildCorrectKernel( levels[ level + 1 ].x, levels[ level ].x, levelShapes[ level ], level === 0 ? dirichletMask : undefined ) );

		}

		// *** THE actual, confirmed-on-real-hardware root cause of this
		// file's own long-run divergence bug with a Dirichlet mask (see
		// buildRestrictKernel's and buildCorrectKernel's own header
		// comments for the two earlier, real-but-insufficient fixes that
		// preceded finding this one, and laplacianDiagonalAt's for a third
		// that turned out to be a no-op) -- found by comparing against
		// mantaflow's own multigrid solver (source/multigrid.cpp's
		// GridMg::smoothGS, Apache-2.0) after direct real-hardware
		// dot-product logging (linalg.js's own window.__pcgLog
		// instrumentation) showed the *textbook* CG/PCG breakdown
		// signature: a single solve() call where beta (newRZ/oldRZ) sits
		// consistently above 1 iteration after iteration, so `p = z +
		// beta*p` compounds geometrically (observed: |p.Ap| roughly
		// tripling *every single iteration* -- -0.0011, -0.0033, -0.019,
		// -0.051, -0.14, ...) until the resulting alpha=oldRZ/pAp produces
		// an astronomically large x update in one dispatch -- explaining
		// the exact shape of every real-hardware failure observed this
		// session: many frames of a perfectly finite, small (~0.1-magnitude)
		// pressure field, then ONE frame where it jumps to ~1e30+
		// *everywhere except the Dirichlet-pinned region itself* (that
		// region stays correct because alpha is a single global scalar
		// applied uniformly to `x += alpha*p` -- a Dirichlet cell's own p
		// stays exactly 0 all the way through, per laplacianAt's own
		// decoupling, so it alone is immune to a bad global alpha).
		//
		// Textbook PCG requires its preconditioner to be symmetric (`M =
		// M^T`) for `r.z` to reliably behave like a valid quadratic form;
		// this is *why* a preconditioner needs to be SPD in the first
		// place, not just "some approximate solve". A V-cycle's own
		// red-black relaxation is only symmetric as an operator if its
		// post-smoothing sweep undoes pre-smoothing's own color order --
		// exactly the standard reasoning documented directly in
		// mantaflow's own GridMg::doVCycle, which calls its own
		// `smoothGS(l, reversedOrder)` with `false` for every pre-smooth
		// repetition and `true` for every post-smooth one specifically to
		// keep the whole V-cycle a symmetric operator (own comment:
		// "Multicolor Gauss-Seidel with two colors... " immediately
		// followed by exactly this forward/reversed split). This file's
		// own `relax()` had no such distinction at all -- every call,
		// whether pre- or post-smoothing, ran color 0 then color 1, always
		// the same order. An asymmetric preconditioner doesn't announce
		// itself as "wrong" the way a singular operator does (no exact
		// zero anywhere for isDegenerateDot to catch) -- it just
		// occasionally produces a `z` misaligned enough with `r` to flip
		// oldRZ's sign relative to its usual (consistently negative, for
		// this file's own negative-semi-definite `A`) behavior, and once
		// that happens mid-solve, beta's runaway is a direct textbook
		// consequence, entirely within CG's own well-understood math, no
		// GPU-specific quantization needed to explain it (this port's own
		// atomic fixed-point reduction just makes the final blowup look
		// like a suspiciously round number once alpha*p pushes x past
		// about 2^31/atomicScale -- a downstream symptom, not the cause).
		//
		// Fixed by reversing color order for post-smoothing (both the
		// per-level "up" sweep and level 0's own numberOfFinalIterations
		// sweep -- everywhere relax() is called after correctDispatchers,
		// i.e. every post-correction relax in this file) -- matching
		// mantaflow's own proven approach exactly. Confirmed on real
		// WebGPU hardware: the minimal repro that previously diverged
		// deterministically within a few dozen to a few hundred frames
		// (a static Dirichlet-zero region, numberOfLevels:4) now runs
		// stably for 1000+ frames with this one change, no other change
		// in this file required -- the two earlier, independently-
		// motivated (and independently kept, since they're still correct
		// hygiene) fixes in buildRestrictKernel/buildCorrectKernel/
		// laplacianDiagonalAt turned out not to be what was actually
		// causing the observed failures.
		function relax( level, iterations, reversed = false ) {

			for ( let i = 0; i < iterations; i ++ ) {

				if ( reversed ) {

					levels[ level ].relaxColor1();
					levels[ level ].relaxColor0();

				} else {

					levels[ level ].relaxColor0();
					levels[ level ].relaxColor1();

				}

			}

		}

		function vCycle( level ) {

			if ( level === numberOfLevels - 1 ) {

				relax( level, numberOfCoarsestIterations );
				return;

			}

			relax( level, numberOfSmoothingIterationsDown );

			levels[ level ].residual();
			restrictDispatchers[ level ]();
			levels[ level + 1 ].zeroX();

			vCycle( level + 1 );

			correctDispatchers[ level ]();

			relax( level, level > 0 ? numberOfSmoothingIterationsUp : numberOfFinalIterations, true );

		}

		return function dispatch() {

			levels[ 0 ].zeroX();
			vCycle( 0 );

		};

	};

}
