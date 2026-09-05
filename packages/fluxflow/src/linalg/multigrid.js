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
// Two real differences from jet's own MGPCG, both deliberate:
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
import { float } from 'three/tsl';
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
function laplacianAt( field, spacing, shape, I ) {

	const center = field( ...I );
	const zero = float( 0 );
	let sum = null;

	for ( let axis = 0; axis < I.length; axis ++ ) {

		const idx = I[ axis ];
		const n = shape[ axis ];

		const lower = I.map( ( v, a ) => ( a === axis ? idx.sub( 1 ) : v ) );
		const upper = I.map( ( v, a ) => ( a === axis ? idx.add( 1 ) : v ) );

		const dLower = idx.greaterThan( 0 ).select( center.sub( field( ...lower ) ), zero );
		const dUpper = idx.lessThan( n - 1 ).select( field( ...upper ).sub( center ), zero );

		const term = dUpper.sub( dLower ).div( spacing[ axis ] * spacing[ axis ] );
		sum = sum === null ? term : sum.add( term );

	}

	return sum;

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
function laplacianDiagonalAt( spacing, shape, I ) {

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

	return sum;

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
// Exported so callers (e.g. examples/07-multigrid-preconditioned-cg/) can
// use the *exact same* stencil for the outer CG loop's `A` that this
// file's own relax/residual steps use internally -- essential for the
// preconditioner to actually approximate the inverse of the system CG is
// solving, not a different one.
export function createLaplacianOperator( shape, gridSpacing ) {

	return function applyLaplacian( input, output ) {

		return buildElementwiseKernel( shape, ( I ) => {

			output( ...I ).assign( laplacianAt( input, gridSpacing, shape, I ) );

		} );

	};

}

// One red-black SOR relax pass over a single color (0 or 1). Reads the
// full current `x`, but only *writes* cells matching `color` -- since a
// cell's neighbors are always the opposite color on a proper checkerboard,
// this is safe to run as one dispatch per color with no data race,
// without needing a stride-2 dispatch primitive tsl_array_n doesn't have.
function buildRelaxKernel( shape, spacing, sorFactor, color, x, b ) {

	return buildElementwiseKernel( shape, ( I ) => {

		const isColor = colorOf( I ).equal( color );
		const Ax = laplacianAt( x, spacing, shape, I );
		const diagonal = laplacianDiagonalAt( spacing, shape, I );
		const current = x( ...I );
		const updated = current.add( b( ...I ).sub( Ax ).div( diagonal ).mul( sorFactor ) );

		x( ...I ).assign( isColor.select( updated, current ) );

	} );

}

// buffer = b - A@x, the true residual -- reused by the V-cycle before
// restricting down to the next coarser level.
function buildResidualKernel( shape, spacing, x, b, buffer ) {

	return buildElementwiseKernel( shape, ( I ) => {

		buffer( ...I ).assign( b( ...I ).sub( laplacianAt( x, spacing, shape, I ) ) );

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
function buildRestrictKernel( finer, coarser, coarseShape ) {

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

			const term = finer( ...combo.indices ).mul( combo.weight );
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
function buildCorrectKernel( coarser, finer, fineShape ) {

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

		finer( ...I ).addAssign( sum );

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

			levels.push( {
				x, b, buffer,
				relaxColor0: buildRelaxKernel( levelShape, levelSpacing, sorFactor, 0, x, b ),
				relaxColor1: buildRelaxKernel( levelShape, levelSpacing, sorFactor, 1, x, b ),
				residual: buildResidualKernel( levelShape, levelSpacing, x, b, buffer ),
				zeroX: buildZeroKernel( levelShape, x ),
			} );

		}

		const restrictDispatchers = [];
		const correctDispatchers = [];

		for ( let level = 0; level < numberOfLevels - 1; level ++ ) {

			restrictDispatchers.push( buildRestrictKernel( levels[ level ].buffer, levels[ level + 1 ].b, levelShapes[ level + 1 ] ) );
			correctDispatchers.push( buildCorrectKernel( levels[ level + 1 ].x, levels[ level ].x, levelShapes[ level ] ) );

		}

		function relax( level, iterations ) {

			for ( let i = 0; i < iterations; i ++ ) {

				levels[ level ].relaxColor0();
				levels[ level ].relaxColor1();

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

			relax( level, level > 0 ? numberOfSmoothingIterationsUp : numberOfFinalIterations );

		}

		return function dispatch() {

			levels[ 0 ].zeroX();
			vCycle( 0 );

		};

	};

}
