// Ported from linalg/linalg.py's `mfcg` (matrix-free conjugate gradient),
// whose own header comment says it's "almost identical to the original
// Taichi source code", ported from Taichi's own
// python/taichi/linalg/matrixfree_cg.py, Apache License 2.0. See
// ../../THIRD-PARTY-NOTICES.md for the attribution.
//
// Several real structural differences from the source, driven by platform
// gaps rather than stylistic preference:
//
// 1. No native on-device reduction. Taichi's `result += p[I]*q[I]` inside a
//    `@ti.kernel` for-loop is a genuine parallel reduction the Taichi
//    compiler handles on-device. tsl_array_n has no reduction primitive of
//    its own, but three.js TSL exposes real WebGPU atomic operations
//    (`atomicAdd`, on a storage buffer marked `.toAtomic()`), so the two
//    dot products this solver needs (`r.r`, `p.Ap`) are computed by having
//    every thread atomically add its own per-cell product into a single
//    shared accumulator -- a genuine GPU-side parallel reduction, not a CPU
//    loop. The catch: WGSL atomics only exist for `atomic<i32>`/
//    `atomic<u32>`, never float, so each per-cell product is scaled by
//    `ATOMIC_DOT_SCALE` and rounded to a fixed-point integer before the
//    atomic add, then divided back after reading the single accumulated
//    int back to the CPU. This trades exact float precision (and a little
//    int32 headroom -- see the constant's own comment) for turning an O(N)
//    CPU-bound reduction + an O(N)-element GPU->CPU transfer into an O(N)
//    GPU-bound reduction + a single-int transfer. It also means this
//    solver only supports scalar `'float'` fields for now: extending the
//    same trick to vector element types would need a per-component
//    accumulator (or an `atomicAdd` per swizzle component), not attempted
//    here since nothing in this project needs it yet.
//    Also a real, load-bearing platform gap: WebGPU atomics are exactly
//    that -- WebGPU-only. TSL's own `AtomicFunctionNode` docs say so
//    explicitly, and reading the WebGL2 fallback backend's own node
//    builder (`GLSLNodeBuilder.js`) confirms it never learned any of the
//    atomic method names at all -- `getMethod('atomicAdd')` falls through
//    to returning the literal string `'atomicAdd'`, which isn't valid
//    GLSL. On this project's dev sandbox (no real WebGPU adapter, falls
//    back to WebGLBackend), the generated shader fails to compile outright
//    (a GLSL syntax error on the WGSL pointer syntax `&x`) rather than
//    silently misbehaving -- see examples/04-conjugate-gradient/main.js's
//    header comment for the exact console output. CONFIRMED correct on
//    real WebGPU hardware: the example converges to the expected exact
//    solution (see that same file's header comment for the reported
//    numbers).
// 2. `solve()` is therefore `async` (every reduction is awaited) where the
//    source's `mfcg` is fully synchronous.
// 3. "Create once, solve many times" instead of a single `mfcg(...)` call.
//    The source's `p`/`r`/`Ap`/`Ax`/`alpha`/`beta` are dynamically
//    allocated and destroyed on every call via `ti.FieldsBuilder()` --
//    tsl_array_n has no equivalent field-disposal mechanism at all yet, so
//    allocating fresh scratch fields on every solve would leak GPU buffers
//    for a solver called every frame. Restructured to match this whole
//    port's established convention instead (createCopyKernel2 and friends
//    in array_utils.js, the boundary-condition solver): a factory builds
//    the scratch fields and kernels *once*, and returns a `solve(tol,
//    maxiter)` method meant to be called repeatedly. Callers should create
//    one solver per (b, x) pair they intend to reuse across frames, not a
//    fresh one per solve.
// 4. `A` (the source's `LinearOperator`/`matvec_kernel`) becomes a factory
//    `applyOperator(input, output) => dispatcher`, called *by this module*
//    exactly twice: once for `(x, Ax)`, once for this solver's own `(p,
//    Ap)` scratch pair -- because tsl_array_n kernels bind to concrete
//    fields at construction time (no per-call rebinding), a single fixed
//    "matvec kernel" the way Taichi's real template kernels work isn't
//    possible; a factory that *builds* a kernel for whichever field pair
//    it's given is the idiomatic equivalent (same shape as
//    createCopyKernel2 in array_utils.js).
// 5. Generic over 1D/2D/3D shapes, matching the source's own
//    ti.i/ti.ij/ti.ijk branching -- capped at 3D for the same reason
//    array2/array3 are: a WebGPU compute dispatch is inherently <=3D.
// 6. Periodic true-residual recomputation, absent from both the Python
//    source and Taichi's own upstream -- ported directly from
//    jet/fluid-engine-dev's own `pcg()` as a numerical-robustness
//    improvement. See RESIDUAL_RECOMPUTE_INTERVAL's own comment below and
//    ../../THIRD-PARTY-NOTICES.md for the attribution.

import * as tsl_array_n from 'tsl_array_n';
import { atomicAdd, round } from 'three/tsl';

// Fixed-point scale for encoding a float product as an atomically-summable
// int32 -- see decision 1 above. Too small loses precision (residuals near
// `tol` can round to the same integer and stall convergence); too large
// risks int32 overflow once summed over many grid cells (int32 range is
// about +/-2.1e9). 65536 (2^16) is a reasonable default for O(1)-magnitude
// fields; tune via the `atomicScale` option for your own problem's actual
// value range.
const DEFAULT_ATOMIC_DOT_SCALE = 65536;

// jet/fluid-engine-dev's own `pcg()` (include/jet/detail/cg-inl.h, MIT
// license, Doyub Kim -- see ../../THIRD-PARTY-NOTICES.md) recomputes the
// true residual `r = b - Ax` from scratch every 50 iterations, instead of
// always using the cheaper incremental `r -= alpha*Ap`: repeatedly
// applying the incremental update accumulates floating-point drift over
// many iterations, and periodically recomputing from scratch corrects it.
// jet also forces an extra recompute whenever the tracked residual grows
// between iterations (a sign the incremental value has already drifted).
// Neither the Python `linalg.py` source nor Taichi's own upstream
// `matrixfree_cg.py` do this -- ported here directly from jet's C++
// source as a genuine robustness improvement, used by both solve()
// functions below. jet also guards its final `sqrt(sigmaNew)` with
// `fabs()` ("workaround for negative zero"); both solve() functions below
// apply the same guard at every sqrt() call site, not just the final one,
// since (unlike jet's own while-loop, which only takes a sqrt once at the
// very end for reporting) this port's for-loop-with-early-break shape
// uses sqrt(...) < tol directly as its own per-iteration break condition,
// where a NaN from a tiny negative residual (plausible here given the
// atomic dot product's own fixed-point quantization noise, on top of the
// float drift jet's own comment already worries about) would silently
// stop the loop from ever detecting convergence.
//
// Added after both examples/04-conjugate-gradient/ and
// examples/05-preconditioned-conjugate-gradient/ were already confirmed
// correct on real WebGPU hardware -- neither example actually exercises
// this addition (both converge in well under 50 iterations, and
// monotonically, so neither the periodic-interval branch nor the
// residual-grew trigger ever fires for N=8), so that earlier confirmation
// still covers every code path those examples actually reach, but this
// specific addition remains structurally untested until either example
// is grown large/ill-conditioned enough to actually take more than 50
// iterations, or the residual genuinely increases at some point.
const RESIDUAL_RECOMPUTE_INTERVAL = 50;

// Beta-restart threshold for createPreconditionedConjugateGradientSolver's
// own solve() -- see that function's own header comment at the beta
// computation site for the full derivation. Every healthy beta actually
// observed across many real-hardware solves (a Dirichlet-masked pressure
// system, this port's own hardest case for a preconditioner to stay SPD
// on) fell in the 0.03-0.3 range; 10 is a generous, order-of-magnitude-plus
// margin above the highest of those, chosen so this never fires on a
// legitimately converging solve while still catching the runaway (observed
// betas of 2-4, sustained, compounding geometrically) well before it can
// reach a magnitude that corrupts x.
const MAX_BETA_MAGNITUDE = 10;

// Sibling bound for alpha (see this same file's own beta-restart comment,
// and the sign-check right next to this constant's own use site, for the
// full derivation). Unlike MAX_BETA_MAGNITUDE (tuned against a specific
// observed healthy range, since beta is a dimensionless ratio of
// like-scaled quantities), alpha's own "reasonable" magnitude genuinely
// depends on a caller's specific problem scale (grid spacing, atomicScale,
// the physical magnitude of b) -- there's no single universal healthy
// range to tune against the way there is for beta. This is deliberately
// generous purely as a last-resort backstop against a runaway magnitude
// the sign check alone wouldn't catch (oldRZ and pAp both negative, so
// alpha is positive, but pAp is only *just* above isDegenerateDot's own
// floor while oldRZ is comparatively large) -- callers with an unusually
// large problem scale should pass a larger atomicScale (see that option's
// own comment) rather than rely on this bound being loose enough.
const MAX_ALPHA_MAGNITUDE = 1e6;

function shapesEqual( a, b ) {

	return a.length === b.length && a.every( ( v, i ) => v === b[ i ] );

}

// Builds a kernel with the right explicit positional arity for `shape`'s
// dimensionality, calling `indexedFn(indices)` with all of them packed into
// one array -- indexedFn itself can then be written once, generically,
// regardless of dimensionality. (Needed because tsl_array_n's kernel()
// validates the callback's own declared arity against shape's
// dimensionality, so a single rest-param callback -- whose .length is
// always 0 -- can't be used directly here.) Exported for reuse by
// multigrid.js, which needs the exact same dimension-generic kernel
// construction for its own 1D/2D/3D-generic stencil operators.
export function buildElementwiseKernel( shape, indexedFn ) {

	if ( shape.length === 1 ) return tsl_array_n.kernel( shape, ( i ) => indexedFn( [ i ] ) );
	if ( shape.length === 2 ) return tsl_array_n.kernel( shape, ( i, j ) => indexedFn( [ i, j ] ) );
	if ( shape.length === 3 ) return tsl_array_n.kernel( shape, ( i, j, k ) => indexedFn( [ i, j, k ] ) );

	throw new Error( `conjugateGradient: only 1D/2D/3D shapes are supported, got ${ shape.length }D.` );

}

// Builds a dispatcher that atomically accumulates sum(fieldA[I] * fieldB[I])
// into `accum` (a 0-D 'int' field already marked `.toAtomic()`) -- see
// decision 1 in the file header comment. `accum` is shared across every
// dot product this solver needs (r.r and p.Ap): they never run
// concurrently, so one accumulator plus a reset before each dispatch is
// enough.
function buildAtomicDotKernel( shape, scale, accum, fieldA, fieldB ) {

	return buildElementwiseKernel( shape, ( I ) => {

		const scaled = round( fieldA( ...I ).mul( fieldB( ...I ) ).mul( scale ) ).toInt();
		atomicAdd( accum(), scaled );

	} );

}

// Resets the shared accumulator, dispatches one atomic-dot kernel, reads
// back the single accumulated int, and decodes it -- the only GPU->CPU
// transfer this reduction needs, versus reading back two full O(N) arrays.
async function readAtomicDot( accum, scale, dispatch ) {

	accum.fromArray( new Int32Array( [ 0 ] ) );
	dispatch();

	const [ scaledSum ] = await accum.toArray();
	return scaledSum / scale;

}

// A dot product read back through the fixed-point atomic accumulator above
// is quantized in units of `1/scale` (decision 1 in the file header
// comment): the smallest magnitude it can ever report as nonzero is
// `1/scale` (a single-count accumulated int), so anything genuinely smaller
// than half that is indistinguishable from an exact 0 -- it either reads
// back as literal `0`, or as noise no more meaningful than 0 would be.
// alpha/beta below both divide by exactly this kind of quantity
// (p.Ap for alpha, the previous iteration's r.r or r.z for beta); a
// genuine 0 denominator -- or one quantized down to it -- produces
// Infinity/NaN with no way to recover, so every division site checks its
// own denominator against this floor first and bails out (breaking the
// iteration, keeping whatever x already holds) rather than risk it.
//
// This is NOT a jet-ported check -- jet's own reference `pcg()`
// (cg-inl.h, see THIRD-PARTY-NOTICES.md) has no equivalent guard, because
// it runs in plain double-precision CPU arithmetic with no artificial
// quantization step at all, so an exactly-zero denominator there could
// only come from a genuinely singular operator (e.g. a fully closed,
// all-Neumann pressure domain with no Dirichlet anchor -- see
// grid_pressure_solver2.js's own header comment) landing the search
// direction exactly in that operator's null space, an already-rare event
// in floating point. This port's own GPU-atomic reduction adds a second,
// independent, and much more easily triggered way to land on an exact
// zero: quantization simply rounding a small-but-nonzero true value down
// to the integer 0 before the atomic add ever happens -- a real risk
// unique to this port's own reduction strategy, confirmed as the likely
// cause of a real-hardware failure where a closed-domain pressure solve's
// RHS was finite and well-posed (sum of divergence ~0, as a closed domain
// requires) yet its pressure came back 100% non-finite from the very
// first solve() call: exactly what a `pAp` (or `oldRZ`/`oldRTr`) that
// quantized to 0 partway through the iteration would produce.
// Exported (like buildElementwiseKernel above) so its threshold math can be
// unit-tested directly without a GPU -- both solve() functions below use it
// as an internal guard, not something a caller normally calls itself.
export function isDegenerateDot( value, scale ) {

	return Math.abs( value ) < 0.5 / scale;

}

// applyOperator(input, output): a factory called exactly twice by this
// function (once for (x, Ax), once for this solver's own (p, Ap) scratch),
// each call expected to return a reusable dispatcher that computes
// output = A @ input when called -- same shape as createCopyKernel2 in
// array_utils.js.
// b, x: tsl_array_n fields, same shape and element type, bound for this
// solver's lifetime. x also serves as the initial guess (matching the
// source: x is both input and output).
// options.atomicScale: fixed-point scale for the GPU atomic dot product,
// see DEFAULT_ATOMIC_DOT_SCALE's own comment -- tune this if your fields'
// typical value range risks int32 overflow or under-precision.
export function createConjugateGradientSolver( applyOperator, b, x, options = {} ) {

	if ( b.type !== x.type ) {

		throw new Error( `conjugateGradient: element type mismatch, b.type(${ b.type }) != x.type(${ x.type }).` );

	}

	if ( ! shapesEqual( b.shape, x.shape ) ) {

		throw new Error( `conjugateGradient: shape mismatch, b.shape(${ b.shape }) != x.shape(${ x.shape }).` );

	}

	if ( b.type !== 'float' ) {

		throw new Error( `conjugateGradient: GPU atomic dot product only supports type "float", got "${ b.type }".` );

	}

	const shape = b.shape;
	const type = b.type;
	const atomicScale = options.atomicScale ?? DEFAULT_ATOMIC_DOT_SCALE;

	const p  = tsl_array_n.arrayN( type, shape );
	const r  = tsl_array_n.arrayN( type, shape );
	const Ap = tsl_array_n.arrayN( type, shape );
	const Ax = tsl_array_n.arrayN( type, shape );

	const alpha = tsl_array_n.array0( 'float' );
	const beta  = tsl_array_n.array0( 'float' );

	const dotAccum = tsl_array_n.array0( 'int' );
	dotAccum.node.toAtomic();

	const dispatchDotRR  = buildAtomicDotKernel( shape, atomicScale, dotAccum, r, r );
	const dispatchDotPAp = buildAtomicDotKernel( shape, atomicScale, dotAccum, p, Ap );

	const applyToX = applyOperator( x, Ax );
	const applyToP = applyOperator( p, Ap );

	const init = buildElementwiseKernel( shape, ( I ) => {

		r( ...I ).assign( b( ...I ).sub( Ax( ...I ) ) );
		p( ...I ).assign( 0 );
		Ap( ...I ).assign( 0 );

	} );

	// r = b - Ax, the *true* residual -- unlike init() above, does not
	// touch p/Ap, so it's safe to call mid-loop for the periodic drift
	// correction (see RESIDUAL_RECOMPUTE_INTERVAL's comment). Callers must
	// call applyToX() first to refresh Ax from the current x.
	const recomputeR = buildElementwiseKernel( shape, ( I ) => {

		r( ...I ).assign( b( ...I ).sub( Ax( ...I ) ) );

	} );

	const updateX = buildElementwiseKernel( shape, ( I ) => {

		x( ...I ).addAssign( p( ...I ).mul( alpha() ) );

	} );

	const updateR = buildElementwiseKernel( shape, ( I ) => {

		r( ...I ).subAssign( Ap( ...I ).mul( alpha() ) );

	} );

	const updateP = buildElementwiseKernel( shape, ( I ) => {

		p( ...I ).assign( r( ...I ).add( p( ...I ).mul( beta() ) ) );

	} );

	function setScalar( field, value ) {

		field.fromArray( new Float32Array( [ value ] ) );

	}

	async function solve( tol, maxiter ) {

		applyToX(); // Ax = A @ x
		init(); // r = b - Ax, p = 0, Ap = 0

		const initRTr = await readAtomicDot( dotAccum, atomicScale, dispatchDotRR );
		let oldRTr = initRTr;
		let newRTr = initRTr;

		updateP(); // p0 = r0 = b - A@x0

		let forceResidualRecompute = false;

		if ( Math.sqrt( Math.abs( initRTr ) ) >= tol ) {

			for ( let iter = 0; iter < maxiter; iter ++ ) {

				applyToP(); // Ap = A @ p
				const pAp = await readAtomicDot( dotAccum, atomicScale, dispatchDotPAp );

				// p has (numerically) run into A's null space, or the atomic
				// reduction quantized a tiny-but-nonzero pAp down to 0 -- see
				// isDegenerateDot's own comment. Either way alpha would be a
				// degenerate division; stop here rather than corrupt x.
				if ( isDegenerateDot( pAp, atomicScale ) ) break;

				setScalar( alpha, oldRTr / pAp ); // alpha = rTr / pTAp
				updateX();

				if ( forceResidualRecompute || ( iter % RESIDUAL_RECOMPUTE_INTERVAL === 0 && iter > 0 ) ) {

					applyToX(); // Ax = A @ x (refresh -- x just changed)
					recomputeR(); // r = b - Ax, correcting any drift from past incremental updates
					forceResidualRecompute = false;

				} else {

					updateR(); // r -= alpha * Ap (cheap incremental update)

				}

				newRTr = await readAtomicDot( dotAccum, atomicScale, dispatchDotRR );

				if ( Math.sqrt( Math.abs( newRTr ) ) < tol ) break;

				// Residual grew since last iteration -- shouldn't happen in
				// exact arithmetic; a sign the incremental r has drifted,
				// so force a true recompute next iteration (see
				// RESIDUAL_RECOMPUTE_INTERVAL's comment).
				if ( newRTr > oldRTr ) forceResidualRecompute = true;

				// Guards the beta division below -- see isDegenerateDot's own
				// comment. oldRTr is *this* iteration's denominator (not yet
				// overwritten from newRTr), so check it here, right before use.
				if ( isDegenerateDot( oldRTr, atomicScale ) ) break;

				setScalar( beta, newRTr / oldRTr ); // beta = rTr_i+1 / rTr_i
				updateP();
				oldRTr = newRTr;

			}

		}

		// Matches the source's own final check exactly: comparing the raw
		// (squared) newRTr against tol directly, NOT sqrt(newRTr) the way
		// every check inside the loop does. This looks like it could be an
		// inconsistency in the original Taichi source, but is harmless for
		// any realistic (small, < 1) tolerance: newRTr < tol^2 (implied by
		// the loop's own break condition) already implies newRTr < tol
		// whenever tol < 1, so the two checks agree in the normal operating
		// range. Preserved as-is rather than "fixed", since this is meant
		// to be a faithful port. (No fabs() needed here -- this is a raw
		// number-vs-number comparison, not a sqrt() call, so there's no
		// NaN-from-negative risk to guard against.)
		return newRTr < tol;

	}

	return { solve, p, r, Ap, Ax };

}

// createPreconditionedConjugateGradientSolver: the same matrix-free CG
// above, extended to accept a preconditioner M (an approximate inverse of
// A) via an `applyPreconditioner(input, output) => dispatcher` factory --
// same idiom as `applyOperator`, called by this module exactly once
// (bound to `(r, z)`, since the preconditioned residual is always
// recomputed fresh from the current r; there's no separate "p-side"
// application the way `applyOperator` needs one for both `x` and `p`).
//
// This is NOT a port of anything -- there's nothing to port from. This
// project's own Python `linalg.py` flags preconditioning as explicit
// future work in its own header comment ("later, this will be extended
// into preconditioning version of matrix free cg"), and Taichi Lang's own
// upstream `matrixfree_cg.py` (checked directly) only has the plain CG
// already ported above, plus an unrelated BiCGSTAB solver -- neither has a
// preconditioned CG to port from. This is an original implementation of
// the standard preconditioned CG algorithm (see e.g. Shewchuk, "An
// Introduction to the Conjugate Gradient Method Without the Agonizing
// Pain", section B2), built on the exact same infrastructure as the
// plain-CG solver above (GPU-atomic reduction, create-once-solve-many,
// 1D/2D/3D-generic, periodic true-residual recomputation) -- see
// decisions 1/3/4/5/6 in the file's top comment, all of which carry over
// unchanged.
//
// What's actually different from plain CG:
// - A new scratch field `z`, the preconditioned residual `M^-1 @ r`.
// - `p`/`alpha`/`beta` are now driven by `r.z` (not `r.r`): `p = z +
//   beta*p`, `alpha = oldRZ/pAp`, `beta = newRZ/oldRZ` -- while
//   convergence is still checked against the *true* residual `r.r`, not
//   `r.z`. A preconditioner can scale z arbitrarily relative to r, so
//   `r.z` isn't a faithful stand-in for "how close is Ax to b" the way it
//   is in the unpreconditioned case (where z===r makes the two
//   identical, and one reduction serves both purposes).
// - A second dot product per iteration (`r.z` alongside `r.r`), plus one
//   `applyPreconditioner` dispatch per iteration -- a real extra
//   GPU-round-trip cost every iteration, in exchange for a good
//   preconditioner typically needing far fewer iterations to converge
//   than plain CG on the same system. Whether that trade is worth it
//   depends entirely on how good/cheap `applyPreconditioner` is for the
//   caller's actual `A` -- this module has no way to judge that itself.
// - The final success check compares `sqrt(newRTr)` against `tol`,
//   *unlike* the sibling function above (which preserves the Python
//   source's own raw-vs-sqrt inconsistency as a faithful-port quirk).
//   There's no "faithful port" obligation here, so this uses the
//   straightforwardly correct comparison instead of replicating a wart
//   from code this function doesn't actually derive from.
//
// CONFIRMED correct on real WebGPU hardware -- see
// examples/05-preconditioned-conjugate-gradient/main.js's header comment
// for the reported numbers (an exact match to 4 decimal places, even
// tighter than the plain-CG sibling's result, consistent with this test
// case's perfect preconditioner converging in a single iteration and
// therefore accumulating far less atomic fixed-point quantization noise).
export function createPreconditionedConjugateGradientSolver( applyOperator, applyPreconditioner, b, x, options = {} ) {

	if ( b.type !== x.type ) {

		throw new Error( `conjugateGradient: element type mismatch, b.type(${ b.type }) != x.type(${ x.type }).` );

	}

	if ( ! shapesEqual( b.shape, x.shape ) ) {

		throw new Error( `conjugateGradient: shape mismatch, b.shape(${ b.shape }) != x.shape(${ x.shape }).` );

	}

	if ( b.type !== 'float' ) {

		throw new Error( `conjugateGradient: GPU atomic dot product only supports type "float", got "${ b.type }".` );

	}

	const shape = b.shape;
	const type = b.type;
	const atomicScale = options.atomicScale ?? DEFAULT_ATOMIC_DOT_SCALE;

	const p  = tsl_array_n.arrayN( type, shape );
	const r  = tsl_array_n.arrayN( type, shape );
	const z  = tsl_array_n.arrayN( type, shape );
	const Ap = tsl_array_n.arrayN( type, shape );
	const Ax = tsl_array_n.arrayN( type, shape );

	const alpha = tsl_array_n.array0( 'float' );
	const beta  = tsl_array_n.array0( 'float' );

	const dotAccum = tsl_array_n.array0( 'int' );
	dotAccum.node.toAtomic();

	const dispatchDotRR  = buildAtomicDotKernel( shape, atomicScale, dotAccum, r, r );
	const dispatchDotRZ  = buildAtomicDotKernel( shape, atomicScale, dotAccum, r, z );
	const dispatchDotPAp = buildAtomicDotKernel( shape, atomicScale, dotAccum, p, Ap );

	// Live view of the most recent solve()'s own final r.r -- exposed (not
	// just returned from solve() as a boolean, which every existing caller
	// already treats as "converged or not") so a caller can add its own
	// last-resort safety net on top: a non-finite or wildly-implausible
	// residual is a direct, *already computed* (no extra GPU work) signal
	// that x itself likely just got corrupted this call, even in cases
	// (confirmed on real hardware) where the loop ran to maxiter without
	// ever tripping isDegenerateDot or either restart guard below --
	// neither guard is a substitute for a caller-side check on the actual
	// outcome. See grid_pressure_solver2.js's own use of this for exactly
	// that: reverting a frame's pressure update entirely if this comes
	// back non-finite, rather than let a bad solve reach velocity.
	const state = { residualSquared: 0 };

	const applyToX = applyOperator( x, Ax );
	const applyToP = applyOperator( p, Ap );
	const applyPreconditionerToR = applyPreconditioner( r, z );

	const init = buildElementwiseKernel( shape, ( I ) => {

		r( ...I ).assign( b( ...I ).sub( Ax( ...I ) ) );
		p( ...I ).assign( 0 );
		Ap( ...I ).assign( 0 );

	} );

	// r = b - Ax, the *true* residual -- unlike init() above, does not
	// touch p/Ap, so it's safe to call mid-loop for the periodic drift
	// correction (see RESIDUAL_RECOMPUTE_INTERVAL's comment). Callers must
	// call applyToX() first to refresh Ax from the current x.
	const recomputeR = buildElementwiseKernel( shape, ( I ) => {

		r( ...I ).assign( b( ...I ).sub( Ax( ...I ) ) );

	} );

	const updateP = buildElementwiseKernel( shape, ( I ) => {

		p( ...I ).assign( z( ...I ).add( p( ...I ).mul( beta() ) ) );

	} );

	const updateX = buildElementwiseKernel( shape, ( I ) => {

		x( ...I ).addAssign( p( ...I ).mul( alpha() ) );

	} );

	const updateR = buildElementwiseKernel( shape, ( I ) => {

		r( ...I ).subAssign( Ap( ...I ).mul( alpha() ) );

	} );

	function setScalar( field, value ) {

		field.fromArray( new Float32Array( [ value ] ) );

	}

	async function solve( tol, maxiter ) {

		applyToX(); // Ax = A @ x
		init(); // r = b - Ax, p = 0, Ap = 0

		const initRTr = await readAtomicDot( dotAccum, atomicScale, dispatchDotRR );
		let newRTr = initRTr;
		let oldRTr = initRTr;

		applyPreconditionerToR(); // z0 = M^-1 @ r0
		let oldRZ = await readAtomicDot( dotAccum, atomicScale, dispatchDotRZ );

		updateP(); // p0 = z0 (p was 0)

		let forceResidualRecompute = false;

		if ( Math.sqrt( Math.abs( initRTr ) ) >= tol ) {

			for ( let iter = 0; iter < maxiter; iter ++ ) {

				applyToP(); // Ap = A @ p
				const pAp = await readAtomicDot( dotAccum, atomicScale, dispatchDotPAp );

				// p has (numerically) run into A's null space, or the atomic
				// reduction quantized a tiny-but-nonzero pAp down to 0 -- see
				// isDegenerateDot's own comment. Either way alpha would be a
				// degenerate division; stop here rather than corrupt x. This
				// is exactly the failure mode confirmed on real WebGPU
				// hardware for a fully closed (all-Neumann, no Dirichlet
				// anchor) pressure domain: p accumulates a null-space
				// (constant) component over the iterations (CG never reduces
				// it, since A@constant=0 exactly), and once p is dominated by
				// it, Ap collapses toward 0 everywhere -- exactly the
				// condition this check catches.
				if ( isDegenerateDot( pAp, atomicScale ) ) break;

				const alphaValue = oldRZ / pAp;

				// Sibling of the beta-restart fix below (see that one's own
				// header comment for the fuller derivation) -- same class of
				// concern, one step earlier: an implausibly large alpha,
				// magnitude-wise, is caught here the same way
				// isDegenerateDot's own pAp check above stops the solve
				// rather than risk a corrupting update.
				//
				// *** A negative-alpha check was tried here too and
				// confirmed, via real-hardware regression, to make things
				// *worse* -- worth recording so it isn't tried again ***
				//
				// alpha=oldRZ/pAp "should" be non-negative if oldRZ and pAp
				// always shared the operator's own consistent sign -- but a
				// merely-approximate preconditioner (a few V-cycle sweeps)
				// apparently produces an occasional, small, genuinely-
				// recoverable negative alpha as ordinary noise, not
				// exclusively as a sign of the runaway this file actually
				// needs to guard against. Breaking the whole solve on every
				// such occurrence (tried directly) stopped CG earlier than
				// it needed to, leaving a worse-converged pressure behind
				// each time -- confirmed to *reduce* real-hardware stability
				// (956 stable frames -> 178) on the exact same repro this
				// fix's other half (beta) was confirmed against. Magnitude
				// alone, without the sign condition, is what's kept here.
				if ( Math.abs( alphaValue ) > MAX_ALPHA_MAGNITUDE ) break;

				setScalar( alpha, alphaValue ); // alpha = rz / pTAp
				updateX();

				if ( forceResidualRecompute || ( iter % RESIDUAL_RECOMPUTE_INTERVAL === 0 && iter > 0 ) ) {

					applyToX(); // Ax = A @ x (refresh -- x just changed)
					recomputeR(); // r = b - Ax, correcting any drift from past incremental updates
					forceResidualRecompute = false;

				} else {

					updateR(); // r -= alpha * Ap (cheap incremental update)

				}

				newRTr = await readAtomicDot( dotAccum, atomicScale, dispatchDotRR );

				if ( Math.sqrt( Math.abs( newRTr ) ) < tol ) break;

				// Residual grew since last iteration -- shouldn't happen in
				// exact arithmetic; a sign the incremental r has drifted,
				// so force a true recompute next iteration (see
				// RESIDUAL_RECOMPUTE_INTERVAL's comment). Compared against
				// the true rTr, not rz, consistent with this function's
				// own convergence check above.
				if ( newRTr > oldRTr ) forceResidualRecompute = true;
				oldRTr = newRTr;

				applyPreconditionerToR(); // z = M^-1 @ r, for the updated r
				const newRZ = await readAtomicDot( dotAccum, atomicScale, dispatchDotRZ );

				// Guards the beta division below -- see isDegenerateDot's own
				// comment. oldRZ is *this* iteration's denominator (not yet
				// overwritten from newRZ), so check it here, right before
				// use. Unlike oldRTr in the sibling function above, r.z has
				// no other convergence check anywhere in this loop (only r.r
				// is compared against tol), so this is the *only* guard
				// protecting this particular division.
				if ( isDegenerateDot( oldRZ, atomicScale ) ) break;

				// *** A real, confirmed-on-real-hardware CG robustness fix,
				// found via direct real-hardware dot-product logging ***
				//
				// Textbook PCG assumes the preconditioner M is SPD, which
				// guarantees r.z keeps one consistent sign *and* a bounded
				// relative magnitude from one iteration to the next (see
				// multigrid.js's own createMultigridPreconditioner header
				// comment for a real asymmetric-relax bug this port had and
				// fixed, since found and fixed there first). Even with that
				// fix, a truncated, approximate preconditioner (a few
				// V-cycle sweeps, not an exact solve) still isn't
				// *guaranteed* SPD for every r that occurs in practice --
				// confirmed on real hardware: newRZ occasionally comes back
				// with the opposite sign from oldRZ, or simply far larger in
				// magnitude, with neither denominator ever near zero
				// (isDegenerateDot's own magnitude check, above, never
				// trips). Either way, beta ends up large enough that
				// `p = z + beta*p` compounds geometrically -- observed
				// directly: |p.Ap| roughly tripling every single iteration,
				// from betas consistently in the 2-4 range -- into an
				// astronomically large x update within the same solve()
				// call, entirely without either denominator ever looking
				// "degenerate" by magnitude, and (confirmed by testing the
				// sign check alone first) without every occurrence even
				// being a clean sign flip -- some are simply an
				// implausibly large *same-signed* ratio. Guarding both
				// ways: a flipped sign, or a magnitude far outside every
				// healthy value actually observed across many real-hardware
				// solves (typically 0.03-0.3; MAX_BETA_MAGNITUDE's own 10x
				// margin above the highest of those is generous enough to
				// never fire on a legitimately converging solve). Either
				// condition means z is no longer a trustworthy continuation
				// of p's own search history, so the safe, standard response
				// (the same idea as restarted/flexible CG variants in the
				// literature) is to restart: fall back to steepest descent
				// for this one step (beta=0, i.e. p=z) -- keeping z itself,
				// still a perfectly good direction on its own, while
				// discarding the now-untrustworthy combination with p's
				// prior history, rather than let a bad beta compound across
				// iterations.
				const rzSignFlipped = ( oldRZ > 0 ) !== ( newRZ > 0 );
				const betaRaw = newRZ / oldRZ;
				const restartP = rzSignFlipped || Math.abs( betaRaw ) > MAX_BETA_MAGNITUDE;

				setScalar( beta, restartP ? 0 : betaRaw ); // beta = rz_i+1 / rz_i
				updateP();
				oldRZ = newRZ;

			}

		}

		state.residualSquared = newRTr;

		// Unlike the sibling function above, this checks the true residual
		// directly -- see this function's own header comment for why.
		return Math.sqrt( Math.abs( newRTr ) ) < tol;

	}

	return { solve, p, r, z, Ap, Ax, state };

}
