# Performance investigation: GPU-resident CG alpha/beta ("option 5")

Status: **abandoned, reverted**. `linalg.js`'s `createPreconditionedConjugateGradientSolver` is back
to the CPU-round-trip design (the same one confirmed stable across this whole project's earlier
verification rounds). This document records three independent implementations that were built and
measured on real WebGPU hardware, why all three were slower than the baseline they were meant to
replace, and the evidence for where the cost actually comes from. Kept so nobody re-attempts the
same direction without seeing this data first.

## Motivation

Among a list of options for improving `examples/16-karman-vortex-street/`'s (256x128) pressure-solve
throughput, option 5 was the "biggest effort, biggest payoff" item: an architectural rewrite to
replace the per-iteration "GPU atomic reduce -> read back to CPU -> compute alpha/beta in JS -> write
back to GPU" cycle (needed because WGSL has no floating-point atomics) with alpha/beta computed and
consumed entirely on the GPU via a small kernel reading the previous iteration's result and storing
into another buffer for the next -- leaving only one CPU readback per iteration (the convergence
check) instead of three.

Options 1-3 (lower tolerance/maxIterations, a stronger multigrid preconditioner, decoupling draw
calls from sim steps) were tried first, confirmed working with a real improvement, and shipped
(`283c5b9`). This document covers option 5 only, tried afterward at the user's explicit request,
tested first for stability (confirmed fine, see below) and then for performance (the actual subject
of this document).

## Baseline design (what was being replaced)

Per CG iteration, the pre-existing code did, for **both** scalars the loop needs:

1. `resetAndDispatch(dotAccum, dispatchDotXxx)` -- reset the atomic accumulator, dispatch the kernel
   that has every thread `atomicAdd` its own per-cell product into it.
2. `await dotAccum.toArray()` -- a real CPU readback (**readback #1**, for `p.Ap` -> alpha).
3. Compute alpha in JS (with the existing degenerate-denominator / magnitude / sign-flip guards).
4. `alphaField.fromArray([alpha])` -- a small CPU->GPU buffer upload.
5. Dispatch `updateX`/`updateR` (each reads `alpha` as a plain, already-resolved `array0` scalar).
6. The same 4-step cycle again for `r.z` -> beta -> `updateP` (**readback #2**).
7. Plus the one readback every version of this loop needs regardless: `r.r`, so the JS `for` loop
   knows when to stop (**readback #3**).

So baseline: **3 readbacks/iteration**, **2 small CPU->GPU buffer uploads/iteration**, and the
`updateX`/`updateR`/`updateP` kernels themselves do trivial per-thread work (read one already-resolved
scalar, multiply-add).

## What was tried

### Round 1 -- separate single-thread "compute alpha/beta" kernels

Added `oldRZGpu`/`doneGpu` (`array0('float')`, GPU-resident replacements for the JS `oldRZ` variable
and the "loop already broke" state) and two new `tsl_array_n.kernel([1], ...)` dispatches,
`computeAlphaKernel`/`computeBetaKernel`, each doing the **entire** alpha/beta computation (guards
included) on a single GPU thread, reading `atomicLoad(dotAccum())` directly (a real WGSL/TSL atomic
op -- `atomicLoad` is exported from three.js's `AtomicFunctionNode.js`) and writing the result to its
own `array0('float')`. `updateX`/`updateR`/`updateP` then read that resolved scalar, same as baseline.

Readbacks/iteration: 1 (`r.r` only). Extra dispatches/iteration versus baseline: **+2**
(`computeAlphaKernel`, `computeBetaKernel`).

**Measured on `examples/15-flow-past-cylinder/` (64x64 grid): 2.2 fps** (100 frames / 44.9s).

### Round 2 -- fold alpha/beta directly into the existing big kernels

Hypothesis: round 1's 2 extra dispatches cost more than the 2 readbacks they removed. Removed
`computeAlphaKernel`/`computeBetaKernel` entirely; instead, `buildAlphaValue()`/`buildBetaValue()`
(plain JS functions returning TSL node graphs, not kernels themselves) are called *from inside*
`updateX`/`updateR`/`updateP`'s own kernel-building closures, so **every thread in those
already-necessary, thousands-of-threads dispatches redundantly computes the identical alpha/beta**
from `atomicLoad(dotAccum())` plus `oldRZGpu()`/`doneGpu()`. Safe as a "benign race": every thread
computes and writes back the same value from the same dispatch-invariant inputs, so the
nondeterministic write order across threads doesn't matter.

Readbacks/iteration: 1. Extra dispatches/iteration versus baseline: **0**.

**Measured: 4.4 fps** (100 frames / 22.76s) -- 2x faster than round 1, but still far below baseline.

### Baseline, re-measured precisely

Round 1/2 exposed a gap in this project's own measurement history: every earlier precise-FPS
measurement in this session was taken on `examples/16` (256x128); `examples/15` (64x64) had only ever
been verified for stability (frame counts, rejection counts), never timed. Fetched the pre-rewrite
`linalg.js` via `git show origin/main:packages/fluxflow/src/linalg/linalg.js` and re-measured on the
same scene/hardware:

**Baseline: 22.2 fps** (100 frames / 4.5s).

So round 2 -- the *better* of the first two rewrites -- is still **~5x slower** than what it replaced.

### Round 3 -- isolate whether `atomicLoad` itself (not dispatch count) is the cost

Round 1 and round 2 differ in two variables at once (dispatch count, *and* how many threads call
`atomicLoad`), so neither cleanly proves which one matters. Round 3 held dispatch structure identical
to round 1 (+2 dispatches/iteration) but changed what runs in them: two tiny `[1]`-shape "distill"
kernels (`distillPAp`, `distillRZ`) do *only* `atomicLoad(dotAccum())` + scale, writing into a plain
(non-atomic) `array0('float')` (`pApGpu`/`newRZGpu`). `buildAlphaValue`/`buildBetaValue` then read
*that* plain buffer from inside `updateX`/`updateR`/`updateP` -- so the guard math (divide/abs/compare/
select) still runs redundantly across thousands of threads, exactly as in round 2, but `atomicLoad`
itself now runs on exactly one thread, same as round 1.

If "many threads calling `atomicLoad`" were the dominant cost, this should recover most of round 2's
speed. If dispatch count dominates regardless of content, this should land near round 1.

**Measured: 2.36 fps** (100 frames / 42.3s) -- matches round 1, not round 2.

## Conclusions

Four data points, two independent confirmations of each pattern:

| Variant | Dispatches/iter vs. baseline | `atomicLoad` call sites | fps |
|---|---|---|---|
| Baseline (CPU round-trip) | N | none | **22.2** |
| Round 2 (folded into big kernels) | N (same) | 3 big kernels, thousands of threads each | **4.4** |
| Round 1 (separate small kernels, full alpha/beta) | N+2 | 2 single-thread kernels | **2.2** |
| Round 3 (distill only, guards stay in big kernels) | N+2 | 2 single-thread kernels | **2.36** |

1. **Merely using `atomicLoad` anywhere in the loop costs roughly 5x, independent of dispatch count.**
   Round 2 has the *exact same* dispatch count as baseline, plus *fewer* readbacks and *fewer* CPU->GPU
   uploads than baseline -- by the original "readbacks are expensive" premise it should have been
   faster, not 5x slower.
2. **Every extra dispatch costs roughly another 2x, independent of what's inside it.** Round 1 (full
   alpha/beta math on 1 thread) and round 3 (`atomicLoad` on 1 thread, guard math on thousands) land
   within 7% of each other despite very different per-thread work -- the only thing they share is +2
   dispatches/iteration versus round 2.
3. Round 3 directly refutes the "thousands of threads contending on one atomic address" hypothesis
   floated after round 1: moving `atomicLoad` down to a single thread while keeping everything else
   the same as round 2 did *not* recover round 2's speed -- it dropped back to round 1's level, because
   the 2 extra dispatches needed to do the distillation cost more than whatever `atomicLoad`-per-thread
   was saving.

**Best-supported remaining explanation**: the cost is tied to *touching the atomic-marked buffer
(`dotAccum`, created via `.toAtomic()`) as a dependency for a later dispatch*, not to per-thread
compute. A CPU readback (`await buffer.toArray()`) of that same buffer is, empirically, *cheaper* on
this hardware/browser (WebGPU/Dawn) than a same-buffer `atomicLoad` consumed by a dependent dispatch --
plausibly because the WebGPU implementation inserts a more conservative synchronization barrier around
atomic-flagged resources than around a plain storage buffer's normal write-then-read hazard tracking,
to guarantee cross-invocation ordering that atomics are specified to provide. This wasn't verified at
the GPU-trace level (no timestamp-query instrumentation or `chrome://tracing` capture was taken) --
if this direction is ever revisited, that's the first thing to add, rather than another blind
implementation variant.

## Decision

Reverted `linalg.js` to the pre-round-1 baseline (`git checkout` back to the last commit, `283c5b9`).
All three rounds were correctness-verified before the performance work started (examples 05 and 07
both reproduce baseline-identical results), and no new instability was observed in ~300+-frame runs of
any round -- this is a pure performance regression, not a correctness or stability finding. Options
1-3 (already shipped in `283c5b9`) remain the effective performance work from this investigation;
option 5 is not recommended for future reattempt without first adding real GPU-side profiling to test
the atomic-barrier hypothesis above, since three structurally different implementations all failed for
what appears to be the same underlying reason.

## Reference: round 2's design (the best-performing variant, still 5x slower than baseline)

```js
// oldRZGpu / doneGpu: array0('float'), GPU-resident replacements for the JS
// `oldRZ` variable and the "loop already broke" state a CPU for-loop used to
// track implicitly. doneGpu latches to 1 once any guard trips; every later
// iteration's inlined alpha/beta forces its own result back to 0 regardless,
// freezing x/r/p forward instead of actually breaking the JS loop (a GPU
// kernel can't do that) -- costs `maxiter - iter` extra dispatches on a
// frozen solve, free otherwise.

function buildAlphaValue() {
	const pAp = float( atomicLoad( dotAccum() ) ).div( atomicScale );
	const pApDegenerate = abs( pAp ).lessThan( 0.5 / atomicScale );
	const alphaRaw = oldRZGpu().div( pAp );
	const alphaTooLarge = abs( alphaRaw ).greaterThan( MAX_ALPHA_MAGNITUDE );
	const shouldFreeze = doneGpu().greaterThan( 0.5 ).or( pApDegenerate ).or( alphaTooLarge );
	return { alphaValue: shouldFreeze.select( float( 0 ), alphaRaw ), shouldFreeze };
}

// Called fresh (independent node-graph copy) from inside updateX AND
// updateR's own kernel-building closures -- every thread in both
// (thousands-of-threads) dispatches redundantly recomputes the identical
// alpha via atomicLoad; safe because every thread reads the same
// dispatch-invariant inputs and writes the same result (a "benign race").
const updateX = buildElementwiseKernel( shape, ( I ) => {
	const { alphaValue, shouldFreeze } = buildAlphaValue();
	x( ...I ).addAssign( p( ...I ).mul( alphaValue ) );
	doneGpu().assign( shouldFreeze.select( float( 1 ), doneGpu() ) );
} );
```

Beta followed the same shape (`buildBetaValue()`, called from `updateP`). Full alpha/beta guard logic
(degenerate-denominator floor, magnitude cap, beta sign-flip check) was carried over verbatim from the
pre-rewrite JS version -- only *where* it runs (GPU vs. CPU) and *how a trip is expressed* (a latched
`doneGpu` flag vs. an immediate JS `break`) changed.

## Reference: round 3's delta from round 2 (the distillation experiment)

```js
// Distilled once per accumulator-read (2 dispatches/iteration, one thread
// each) instead of read via atomicLoad from inside the big kernels.
const pApGpu   = tsl_array_n.array0( 'float' );
const newRZGpu = tsl_array_n.array0( 'float' );

const distillPAp = tsl_array_n.kernel( [ 1 ], ( i ) => {
	pApGpu().assign( float( atomicLoad( dotAccum() ) ).div( atomicScale ) );
} );
const distillRZ = tsl_array_n.kernel( [ 1 ], ( i ) => {
	newRZGpu().assign( float( atomicLoad( dotAccum() ) ).div( atomicScale ) );
} );

// buildAlphaValue/buildBetaValue changed to read the plain distilled copy:
function buildAlphaValue() {
	const pAp = pApGpu(); // was: float( atomicLoad( dotAccum() ) ).div( atomicScale )
	// ...guard math unchanged...
}
```

`solve()`'s loop called `distillPAp()`/`distillRZ()` immediately after each `resetAndDispatch(dotAccum,
...)`, before `updateX()`/`updateP()` respectively.

## Measurement methodology

All numbers above were taken with the same precise-timing harness, driven manually to avoid any
`requestAnimationFrame` throttling or rolling-average smoothing:

```js
// Test-harness setup (examples/15's own main.js, removed again after use):
// window.requestAnimationFrame = function() { return 0; }; // neutralize the native rAF loop
// window.debugTick = animate;                                // manual per-frame driver

const t0 = performance.now();
for ( let i = 0; i < 100; i++ ) await window.debugTick();
const t1 = performance.now();
JSON.stringify( { framesRun: 100, elapsedMs: t1 - t0, fps: 100 / ( ( t1 - t0 ) / 1000 ) } );
```

Neutralizing the native rAF loop matters: driving frames manually while the tab is also foregrounded
lets the browser's own rAF fire concurrently, causing overlapping `animate()` invocations that produce
a false "stuck" symptom unrelated to actual solver performance (seen once earlier in this project's
own history, traced via duplicate same-frame console log lines).
