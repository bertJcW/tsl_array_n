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

---

# Re-measured after the dot product stopped using atomics

The investigation above concluded that its best-supported explanation was
"the cost is tied to touching the atomic-marked buffer (`dotAccum`, created
via `.toAtomic()`) as a dependency for a later dispatch", and recommended
against reattempting the direction without real GPU profiling.

That premise has since expired: `linalg.js`'s dot product is no longer an
atomic reduction at all. It is a lane-partitioned float reduction into a
plain storage buffer, with no `.toAtomic()` anywhere in the CG path (see
`createDotReducer`). So the direction was re-opened, and re-measured from
scratch on the same reference scene, `examples/15-flow-past-cylinder/`
(64x64), with the same harness.

## Baseline is no longer 22.2 fps, and the reason matters

**Current: 8.1 fps** (100 frames, 20 warm-up frames discarded), at a mean of
**12.4 CG iterations per frame, 100% of them converging**, with no guard
tripping on any frame.

That is 2.7x slower than the 22.2 fps recorded above, and it is not a
regression to undo. The old number was measured when the dot product was
still fixed-point, which meant `isDegenerateDot`'s quantization floor
(`0.5/atomicScale`) aborted CG as soon as `p.Ap` fell below it -- long before
the solve was done. Roughly half of all frames were returning unconverged,
leaving real divergence in the velocity field, which is what
`createDotReducer`'s own comment records and what made a whole free-surface
scene collapse. The solver is slower now because it finishes.

## A readback costs 3.3 ms here, and does not care how big it is

Measured directly in the page, 50 repetitions each:

| readback | ms |
|---|---|
| `b` (one 64x64 field) | 3.29 |
| `dataU` (a 65x64 face field) | 3.53 |

Pure round-trip latency, essentially independent of payload. There are
exactly three `toArray()` call sites in the whole solve path -- the two dot
reducers and the pressure solver's own bad-cell count -- so a 12.4-iteration
frame is about 26 round trips. At 3.3 ms each that is 86 ms of a 123 ms
frame, which makes "reduce the number of readbacks" the obvious lever.

## It is not the lever. Fusing two readbacks into one changed nothing.

`r.r` and `r.z` are both read after `updateR`, over the same shape, from
fields that are final by then. A `createMultiDotReducer` was written to
compute both in one dispatch and return both from one readback, taking the
loop from three round trips per iteration to two. The preconditioner had to
be hoisted above the convergence check to make `z` available in time, which
costs one wasted V-cycle on the final iteration and nothing otherwise.

**Measured: 8.11 fps, against 8.10 fps before.** Same iteration count, same
convergence. Removing a third of the loop's synchronisations produced no
measurable change at all.

That falsifies the naive latency model in the section above it: if 26 round
trips at 3.3 ms really composed the frame, removing 12 of them could not be
free. The most likely reconciliation is that a readback inside the loop is
not paying the standalone 3.3 ms -- it waits on GPU work that is already
queued, so its cost is `max(queued GPU work, latency)` rather than additive,
and the queued work is what dominates. The isolated 3.3 ms is a floor
measured against an idle queue, not the marginal cost of one more readback in
a busy one.

The fusion was **reverted**: it bought nothing measurable and the hoist costs
a wasted V-cycle. `state.iterations`/`stoppedBy` and the `?mgLevels=`
parameter were kept, because without them none of the above is measurable.

## Where the time actually is: dispatches inside the V-cycle

Varying the multigrid depth separates iteration count from per-iteration
cost:

| `?mgLevels=` | fps | mean iterations | ms per iteration |
|---|---|---|---|
| 1 | 6.45 | 22.7 | 6.83 |
| 2 | 7.10 | 17.4 | 8.07 |
| 4 | 8.11 | 12.4 | 9.90 |

Deeper is a net win -- the iterations it saves outweigh what each costs --
but the per-iteration cost climbs with depth, and depth is exactly what
multiplies the number of small dispatches in the V-cycle (relaxation sweeps,
restriction, prolongation, per level, twice). Even at one level, where the
"V-cycle" is plain relaxation, an iteration still costs 6.83 ms.

This lines up with the original investigation's second conclusion -- "every
extra dispatch costs roughly another 2x, independent of what's inside it" --
which, unlike its atomic-buffer conclusion, was never invalidated. Dispatch
count, not synchronisation count and not arithmetic, is the thing to attack.

## What to do next, and what not to

Do not reattempt GPU-resident alpha/beta on the strength of "it removes
readbacks". That was this round's hypothesis in a cheaper form, it was tested
directly, and removing readbacks did nothing. GPU-resident alpha/beta would
also *add* dispatches, which is the one thing both investigations agree is
expensive.

The promising direction is the opposite one: **fewer, larger dispatches in
the V-cycle**. Fusing the red and black halves of a relaxation sweep, or a
whole level's down-sweep, into single kernels attacks the quantity both
investigations independently identified. That is a real change to
`multigrid.js` rather than a knob.

And the standing recommendation from the original investigation still holds
and is still unmet: before another implementation round, add actual GPU-side
profiling (timestamp queries, or a `chrome://tracing` capture) so that
"dispatch overhead" stops being an inference from fps deltas and becomes a
measurement.

---

# The instrumentation both rounds asked for, and what it says

Both investigations above end with the same recommendation: stop inferring
dispatch cost from fps deltas and measure it. `src/profiling.js` does that.
Every kernel in `linalg.js` and `multigrid.js` is built through one function,
`buildElementwiseKernel`, so wrapping the dispatcher there counts the entire
pressure-solve hot path without threading a profiler through a dozen
factories. Cost when off: one boolean test per dispatch.

`examples/15-flow-past-cylinder/` exposes it as `__fluxflowProbe.profile(n)`,
and `?profile=1` additionally asks the renderer for WebGPU timestamp queries
(`tsl_array_n.init` forwards `trackTimestamp` straight to the
WebGPURenderer, so that needed no change to that package).

## The number nobody had

**1075 dispatches per frame**, at 64x64, for 12.4 CG iterations.

| label | dispatches/frame | encode ms/frame | share of encode |
|---|---|---|---|
| `mg-relax` | **829.9** | **43.24** | **75%** |
| `mg-clear` | 51.0 | 2.90 | 5% |
| `cg-dot` | 39.3 | 2.41 | 4% |
| `mg-restrict` | 38.3 | 2.24 | 4% |
| `mg-residual` | 38.3 | 2.11 | 4% |
| `mg-prolong` | 38.3 | 2.01 | 3% |
| `pcg-updateX/P/R` | 12.8 each | ~0.9 each | ~5% total |

Frame: 151 ms wall (with profiling on; 123 ms without). **57.7 ms of it --
38% -- is spent inside the dispatch calls themselves**, i.e. CPU-side
encoding, before any GPU work or synchronisation. Three quarters of that is
one label.

GPU timestamps came back null on this machine: the adapter does not expose
`timestamp-query`, so `readComputeTimestampMs` correctly reports "not
available" rather than throwing. The CPU-side numbers are the ones the
decision rests on and they need no special support.

## Where the relaxation dispatches are, and it is not where the work is

A V-cycle at 4 levels with the default sweep counts costs, per cycle:

    down    3 levels x 2 sweeps x 2 colours  = 12
    coarsest        20 sweeps x 2 colours    = 40
    up      2 levels x 2 sweeps x 2 colours  =  8
    level 0  final 2 sweeps x 2 colours      =  4
                                        total  64

**Forty of sixty-four -- 62% -- go to the coarsest level, which at 4 levels
from 64x64 is an 8x8 grid of 64 unknowns.** Each of those dispatches does
almost no arithmetic and pays a full dispatch's overhead. That is the
pathology, stated precisely for the first time.

Confirmed by measurement, `?coarseIter=4` against the default 20:

| | dispatches/frame | `mg-relax` | encode ms/frame | CG iterations | fps |
|---|---|---|---|---|---|
| `numberOfCoarsestIterations: 20` | 1075 | 830 | 57.7 | 12.4 | 6.60 |
| `numberOfCoarsestIterations: 4` | 750 | 475 | 53.5 | 14.0 | **7.31** |

11% faster, for four extra CG iterations. Note also that encode time fell
only 7% while dispatch count fell 30%: the coarse-level dispatches are the
*cheap* ones to encode, so cutting them helps less than counting them
suggests. Encode cost per dispatch is not constant -- it rises with grid
size -- which is worth knowing before anyone budgets a fusion by dispatch
count alone.

## What this makes the next piece of work

Not "fewer sweeps": that is a knob, it trades against CG iterations, and it
is already exposed. The measurement points somewhere better.

1. **Solve the coarsest level directly instead of relaxing it.** Sixty-four
   unknowns is small enough to solve exactly, and it would replace 40
   dispatches per V-cycle with one -- or with zero, by reading the level back
   and solving on the CPU, which at 64 values costs one readback against the
   forty dispatches it removes.
2. **Fuse the two colour passes.** Red-black needs a global barrier between
   the halves, and a dispatch is the only global barrier available, so this
   is not free -- it means either damped Jacobi (one dispatch per sweep,
   weaker smoothing) or a workgroup-level scheme. Worth measuring against
   option 1, not before it.

Both are changes to `multigrid.js`, and both are now measurable end to end
rather than argued: `profile(n)` reports dispatches per frame per label
before and after.

---

# Is TSL the problem? Measured: no, but one call in it is

Prompted by a direct question -- whether three.js's TSL is costing enough
against native WebGPU to justify replacing fluxflow's foundation. The
profiler above put a number on the suspicion: **54 microseconds of CPU time
per dispatch**, where a native `dispatchWorkgroups()` is a few microseconds
of appending to a command encoder. An order of magnitude is worth chasing.

## Reading the code first

`tsl_array_n`'s `kernel()` returns `() => getRenderer().compute( computeNode )`
-- one `compute()` call per dispatch. And in three.js's WebGPU backend:

    finishCompute( computeGroup ) {
        groupData.passEncoderGPU.end();
        submit( this.device, groupData.cmdEncoderGPU.finish() );
    }

Every `compute()` call creates its own command encoder, opens its own
compute pass, and **submits its own command buffer to the queue**. At 1075
dispatches per frame that is 1075 encoders, 1075 passes and 1075 queue
submits, where native code would use one of each and 1075
`dispatchWorkgroups()` calls inside them.

But `renderer.compute()` already accepts an **array**, and wraps the whole
array in a single `beginCompute`/`finishCompute` pair -- one encoder, one
pass, one submit. The batched form exists; nothing is using it.

## Measured, not inferred

64 trivial dispatches over a 4096-element buffer, ten repetitions, warmed up
so pipeline creation is excluded:

| | per dispatch |
|---|---|
| `compute(node)` once per node | **62.2 us** |
| `compute([ ...nodes ])` once for all | **6.7 us** |

**9.26x.** And 62.2 us against the 54 us the profiler measured inside the
real solver is two independent measurements agreeing, which is the main
reason to believe either.

## Batching does not break ordering

Worth checking rather than assuming, because red-black Gauss-Seidel depends
on it: 40 dispatches each doing a read-modify-write of the *same* cell, run
as one batched pass, produce exactly 40 -- identical to running them as 40
separate passes. WebGPU orders dispatches within a pass and handles the
hazard between them, so a batched V-cycle stays correct.

## What this means for the architecture question

**Do not replace the foundation.** The gap between this port and native
WebGPU, on the evidence, is not TSL's authoring model, the generated WGSL,
or three.js's node system -- it is one call doing per-dispatch submission,
and three.js already ships the fix. Roughly nine tenths of a 9x gap is
available without leaving TSL, without touching a kernel, and without
touching the multigrid algorithm.

Projected on the real solver: 57.7 ms of encoding per frame becomes of order
6 ms, taking encoding from 38% of the frame to about 4%, and the frame from
~123 ms to ~72 ms -- about 1.7x -- before any algorithmic change. It also
composes with the coarse-level work above rather than competing with it:
fewer dispatches and cheaper dispatches multiply.

What native WebGPU would still buy, and what it would cost, for the record:
finer control of bind-group reuse and buffer lifetimes, no node-system
bookkeeping, and no dependence on three.js's release cadence -- against
rewriting every kernel in this port by hand in WGSL and giving up the
authoring model that is the entire premise of `tsl_array_n`. That trade is
not worth making to recover an overhead that a batched call already
recovers.

## What the change actually is

`tsl_array_n.kernel()` would need to expose its compute node, or grow a
batch API -- something like a `dispatchBatch( [ ...dispatchers ] )` that
calls `renderer.compute()` once with the underlying nodes. fluxflow would
then group the V-cycle's dispatches, which is where 77% of them are.

That is a change to `tsl_array_n`, not to fluxflow, so it is a decision
about that package's API rather than something to do unilaterally from here.

---

# Batching, implemented and measured across every solver

`tsl_array_n` grew `createBatch( dispatchers )` (and a one-shot
`dispatchBatch`), and `kernel()` now exposes its compute node so a batch can
collect them. `multigrid.js` builds its V-cycle as a list of dispatchers
once at construction and submits the whole thing through one `createBatch`
dispatcher -- a V-cycle contains no readback, so there is nothing forcing it
to be eighty submissions.

`createBatch` rather than the one-shot form is deliberate and was measured:
resolving the same fixed list on every call is real work, and on a light
scene it ate the gain.

## Results, same harness, 20 warm-up frames discarded

| scene | before | after | |
|---|---|---|---|
| 15 flow-past-cylinder | 8.10 | **12.6-15.3** | **~1.7x** |
| 20 flip-dam-break | 7.95 | **12.72** | **1.60x** |
| 28 drop-into-pool | 3.72 | **4.90** | **1.32x** |
| 26 dye-free-surface | 11.15 | **12.34** | **1.11x** |
| 29 static-droplet | 61-75 | 60-63 | ~0.9x, inside the noise |

With profiling on, example 15's breakdown shows where it went: encoding fell
from **57.7 ms per frame to 11.7 ms**, and from 38% of the frame to 15%. The
V-cycle's own share went from 43.2 ms to 3.2 ms -- 13.7x -- for the same
1075 dispatches of actual work.

The gradient across the table is the mechanism, visible: the more of a
frame's time is the pressure solve, the more batching returns. Example 29 is
the limiting case, at a single CG iteration per frame -- one V-cycle to
batch, and the rest of the frame is FLIP stages that are still submitted one
at a time. Its numbers are noisy in both builds (61-75 unbatched over four
repeats) and the difference does not survive the spread.

## Correctness

Unchanged everywhere it can be checked exactly: examples 04 and 05 still
reach the analytic solution to all four printed digits, 07 still converges
to its reference within 1e-2, and 29's Young-Laplace ratio is still 1.019.

One thing found while isolating this, and *not* caused by it:
examples/06-multigrid-preconditioner/'s Dirichlet check reports its pinned
cell as -42 against a target of 42, identically with and without batching.
That is the example not having followed the masked-row sign flip documented
in `laplacianDiagonalAt` -- the operator's masked row is negated, so a caller
driving the preconditioner directly has to negate its own `b` for that row,
which `grid_pressure_solver2.js` does and this example does not. Worth
fixing; unrelated to anything here.

## What is left

Batching the V-cycle covers 77% of the dispatches. The remainder are in the
FLIP stages (P2G, G2P, advection, boundary), which are `tsl_array_n.kernel`
dispatchers too and could be batched the same way wherever a run of them has
no readback between. That is a fluxflow-side change with no new machinery
needed.

And the coarse-level finding above still stands and is now worth more, not
less: forty of a V-cycle's sixty-four relaxation dispatches still go to an
8x8 grid. Batching made them cheap to submit; it did not make them useful.

---

# Re-locating the bottleneck after batching

Batching moved the frame's composition, so the "what next" list written
against the old composition was re-derived rather than executed. Two
measurements, both on the post-batching build.

## Synchronisation is free at the margin. The GPU is the bottleneck.

Added 8 extra full readbacks per frame to examples/28-drop-into-pool/ and
measured the frame:

    baseline            220.4 ms
    + 8 readbacks       207.2 ms      marginal cost per readback: ~0

Eight extra round trips, each of which costs 3.3 ms measured standalone
against an idle queue, added nothing at all -- the difference is noise and
points the wrong way. If the CPU were waiting on latency, eight more
latencies would have added about 26 ms.

So the CPU is not waiting on the round trip, it is waiting on GPU work that
is already queued. **The frame is GPU-execution-bound.** This also settles,
rather than contradicts, the earlier surprise that fusing two dot products
into one readback changed nothing: there was never any latency to save.

(A resolution sweep was tried as the complementary test and thrown away: at
96x144 against 64x96 the particle count, the CG iteration count and the
scene's state all move together, and it produced a *faster* frame at 2.25x
the cells. Too many things vary at once to mean anything.)

## Which makes the coarse level worth far more than it was

Before batching, cutting `numberOfCoarsestIterations` from 20 to 4 was worth
11%. Re-measured now:

| `?coarseIter=` | ms/frame | mean CG iterations |
|---|---|---|
| 20 (default) | 80.7 | 12.5 |
| 4 | **52.7** | 14.2 |

**1.53x**, for 1.7 extra CG iterations.

The reason it grew is the reason the whole picture changed. Those forty
dispatches per V-cycle were previously buried in submission cost, so
removing thirty-two of them barely showed. With submission nearly free,
what is left of them is GPU execution -- and an 8x8 grid is 64 threads,
which is far below what any GPU fills a launch with. They are close to pure
launch overhead, now measurably so.

## Conclusion for the next piece of work

**Solve the coarsest level directly.** It is worth roughly 1.5x on this
scene, and unlike the `coarseIter` knob it costs nothing in convergence: 64
unknowns is small enough to solve exactly, replacing forty dispatches of
Gauss-Seidel with one solve.

**Do not batch the FLIP stages.** That was the other candidate, and this
measurement rules it out: those stages are a few dozen dispatches per frame
against the pressure solve's thousand, and submission -- the only thing
batching addresses -- is no longer what costs. It would buy a percent or
two.

After the coarse solve, the next question is occupancy rather than count:
whether the relaxation kernels on the finer levels are themselves filling
the GPU, which is a different measurement again and needs the timestamp
queries this machine's adapter does not expose.

## The coarse solve was built, measured, and rejected

The recommendation above -- replace the coarsest level's forty relaxation
dispatches with one solve -- was implemented as a single-thread kernel that
walks the whole coarse grid every sweep in sequence. One launch instead of
forty, real Gauss-Seidel rather than red-black (a single thread sees each
neighbour's update immediately), no race to reason about, gated to grids of
at most 256 cells so a larger coarse level keeps the old path.

It is worse on every axis that matters:

| | red-black, 40 dispatches | single thread, 1 dispatch |
|---|---|---|
| examples/15 frame | **80.7 ms** | 120.0 ms |
| mean CG iterations | **12.5** | 18.6 |
| examples/07 max abs diff | **< 0.01** | 0.0132 |

Slower, *and* a worse preconditioner, *and* it moved a reference test past
its tolerance. Reverted.

Two things were wrong in the reasoning that produced it, and both are worth
keeping:

1. **"One GPU thread is slow, but the grid is small" underestimated how
   slow.** Twenty sweeps over sixty-four cells is 1280 serial iterations of
   a dependent read-modify-write chain on a single lane, and that costs more
   than forty launches of sixty-four parallel threads -- even though the
   launches were the thing being removed. The dispatches were nearly pure
   overhead; one lane is nearly pure latency.
2. **"Sequential Gauss-Seidel converges faster per sweep than red-black" is
   not true for this operator.** It is true in general for a fixed ordering,
   but red-black is the standard multigrid smoother precisely because its
   *smoothing* factor -- how fast it kills high-frequency error, which is
   all a smoother is for -- is better than lexicographic ordering's for a
   Poisson stencil. The iteration count says so directly: 12.5 to 18.6.

So the coarse level is still worth attacking, but not by serialising it.
What remains, in order of promise:

- **A genuinely direct solve**, not more iterations: sixty-four unknowns is
  a small dense system, and the operator only changes when the Dirichlet
  mask does, i.e. once per frame rather than once per V-cycle. A per-frame
  factorisation reused across the frame's dozen V-cycles is the version
  worth building.
- **A single-workgroup parallel solve** with `workgroupBarrier()` between
  sweeps -- keeps the parallelism, still one dispatch. Needs a barrier
  primitive tsl_array_n does not currently expose.
- **Fewer levels**, so the coarsest grid is large enough to be worth a
  launch. Already measurable via `?mgLevels=`, and already known to trade
  against iteration count.

## The coarse level, done with a barrier instead: it works, and it prices barriers

Second attempt at the same target, keeping red-black rather than replacing
it. The coarsest grid fits in one workgroup, so its invocations can
synchronise with each other between colours instead of needing a fresh
dispatch as the barrier: one dispatch, `storageBarrier()` between the two
halves, the arithmetic byte-for-byte what the two-dispatch version did.
`tsl_array_n.kernel` grew a `workgroupSize` option to pin every cell into
the same workgroup, which is what makes the barrier mean anything.

Measured on examples/15-flow-past-cylinder/:

| | ms/frame | CG iterations |
|---|---|---|
| 40 dispatches (batched) | 80.7 | 12.5 |
| **1 dispatch + barriers** | **72.3** | **12.5** |

**1.12x, with the iteration count unchanged** -- unlike the single-thread
attempt, this is the same preconditioner, and examples/07 is back inside its
1e-2 tolerance. Neutral on the FLIP scenes (20: 12.7 to 12.1; 28: 4.90 to
4.94), where the pressure solve is a smaller share of the frame.

### What the numbers price, and it caps this whole direction

Cross-referencing the four measurements gives the cost of a barrier
directly:

    40 dispatches, 20 sweeps    80.7 ms
     8 dispatches,  4 sweeps    52.7 ms     (-28.0)
     1 dispatch,   20 sweeps    72.3 ms     (-8.4)
     1 dispatch,    4 sweeps    68.1 ms     (-4.2 from the line above)

The last line isolates the *work*: sixteen sweeps of arithmetic on an 8x8
grid are worth 4.2 ms. So of the 28 ms that cutting sweeps saved in the
40-dispatch world, about 4 ms was work and about **24 ms was the 32 removed
dispatches -- roughly 0.75 ms each**.

Removing 39 dispatches with barriers should then have been worth ~29 ms. It
was worth 8.4. The difference is what the barriers cost: 20 sweeps x 2
barriers, about **0.5 ms per storageBarrier** -- two thirds of a full
dispatch launch.

That is the useful, transferable number here, and it caps the whole
"fuse dispatches with barriers" direction: **fusing N dispatches into one
recovers only about a third of what removing them outright would.** Worth
doing where it is free, as here, and not worth building elaborate machinery
for.

Which leaves the original recommendation standing, and now better
motivated: a *direct* coarse solve does not need barriers at all. Sixty-four
unknowns, an operator that only changes when the Dirichlet mask does -- once
per frame, not once per V-cycle -- so one factorisation could serve a
frame's dozen cycles and cost neither launches nor barriers.

---

# Three direct-solve directions, measured before building any of them

## 1. CPU coarse solve, one per V-cycle: dead

The plan was to read the coarse `b` back, solve sixty-four unknowns exactly
on the CPU, and upload `x` -- once per V-cycle. Its whole viability rested
on an assumption worth testing first: that a readback *inside* the V-cycle
is as cheap as the ones measured at the end of a frame, which cost nothing
at the margin.

It is not. Probe: one extra readback per CG iteration, right after the
preconditioner, on examples/15-flow-past-cylinder/.

    no sync            46.0 ms/frame   (34.9 on a repeat)
    mid-cycle sync     87.1 ms/frame

About **3.9 ms per mid-cycle synchronisation** -- the full standalone
readback latency, none of it absorbed. The reason the end-of-frame ones were
free is that the GPU had queued work to get on with; mid-cycle it does not,
so the CPU stalls and then has to re-encode the rest of the cycle.

Twelve iterations means twelve of those: **~47 ms per frame, to save the
15-20 ms the coarse level still costs.** A clear net loss, and the reason to
measure before building.

**What survives**: the same idea with the synchronisation moved to *once per
frame* instead of once per V-cycle. The coarse operator only changes when
the Dirichlet mask and face weights do, which is once a frame -- so the CPU
could read those back once, invert the 64x64 system once (trivial in JS),
upload the inverse once, and every V-cycle in that frame does one dense
matvec: sixty-four threads, sixty-four multiply-adds each, one dispatch, no
barrier, no sync. One stall per frame rather than twelve. That is the
version worth building.

## 2. FFT/DCT preconditioner: not attempted, and why

The largest available lever and the only one that would remove the V-cycle
rather than shrink it: a DCT solves the constant-coefficient Neumann Poisson
problem *exactly* in O(N log N), and an exact solve of a nearby operator is
an excellent preconditioner for the real one. Two transforms of about six
passes each would replace roughly eighty dispatches, and the iteration count
could fall well below 12.5.

Not attempted here, deliberately, and not because of the build cost alone:
its payoff is scene-dependent in a way the others are not. MGPCG is the
graphics standard precisely because it copes with irregular domains, and a
free surface makes every air cell a Dirichlet cell -- exactly where a
constant-coefficient FFT preconditioner is worst. The closed-domain grid
scenes (14, 15, 16) should benefit strongly; the free-surface FLIP scenes
(20-29) might not benefit at all. Building it means building the transform
*and* the per-scene comparison, and shipping it means a
`preconditioner: 'fft' | 'multigrid'` choice that the library picks
automatically from whether a scene has a Dirichlet region at all.

That is a real piece of work rather than an afternoon, and it should start
from its own measurement: how far below 12.5 iterations a *perfect*
preconditioner would get each scene, which bounds what any of this can buy.

## 3. Per-frame precompute: a large ceiling, an uncertain floor

The structural observation is that the operator changes once per frame but
is applied about twelve times, and nothing exploits that. The clearest
candidate is `laplacianDiagonalAt`, recomputed inside every relax
invocation, on every sweep, at every level, from inputs that are fixed for
the whole frame.

Probed by substituting a constant diagonal -- numerically wrong, so the
iteration count moves and only the *per-iteration* figure is meaningful:

    computed diagonal   72.3 ms / 12.5 iterations = 5.78 ms per iteration
    constant diagonal   95.8 ms / 25.7 iterations = 3.73 ms per iteration

**35% of per-iteration cost is the diagonal computation** -- a big ceiling
for a term that is pure redundancy within a frame.

The floor is the uncertain part, and it is why this is a measurement rather
than a change: precomputing the diagonal replaces that arithmetic with a
*buffer read*. On a GPU, recomputing cheap ALU work is frequently faster
than loading a value, and this scene's diagonal is select-heavy arithmetic
with no memory access at all (it has no face weights). So the realistic gain
is somewhere between 35% and negative, and only an implementation settles
it. It is cheap enough to be worth that: one extra field per level, filled
once per frame.

## Where this leaves things

Ordered by expected value per unit of work, on the evidence above:

1. **Per-frame diagonal precompute** -- cheapest to build, 35% ceiling on
   per-iteration cost, real risk of being a wash. Build it and measure.
2. **Coarse inverse, uploaded once per frame** -- the surviving form of
   direction 1, one stall per frame instead of twelve.
3. **FFT preconditioner** -- biggest ceiling by far, biggest build, and
   needs its own scene-by-scene justification first.

---

# Options 1 and 2, built and priced

## 1. Per-frame diagonal precompute: built, 30% slower, reverted

Implemented as a `diagonalField` per level, filled by its own kernel at the
head of the V-cycle batch (rather than through a `refresh()` the caller has
to remember, which would give a silently wrong answer -- a zero diagonal is
a division by zero -- the first time someone forgot). relax and the coarse
sweep read the field instead of recomputing `laplacianDiagonalAt`.

Correct: examples/07 still converges inside 1e-2.

| | ms/frame | CG iterations | ms/iteration |
|---|---|---|---|
| recomputed (baseline) | **72.3** | 12.5 | **5.78** |
| precomputed field | 93.7 | 11.2 | 8.36 |

**30% slower per frame, 45% worse per iteration.** Reverted.

The 35% ceiling measured earlier was real as a ceiling and useless as a
prediction, exactly as flagged: the substitution being priced there removed
the arithmetic *and put nothing in its place*, while the real change
replaces it with a buffer read. On this GPU that read costs more than the
select-heavy arithmetic it replaced -- and this scene's diagonal has no
memory access at all to begin with, so the exchange was ALU for bandwidth
in the worst direction. Four extra dispatches per V-cycle for the fills add
to it.

Worth stating as a rule, since it will come up again: **a "remove the work"
probe does not price a "cache the work" change.** They differ by whatever
the cache costs to read, and on a GPU that is frequently more than the work.

## 2. Coarse inverse uploaded once per frame: not built, priced out of it

The surviving form of direction 1 in the section above -- read the mask and
face weights back once a frame, invert the 64x64 coarse system on the CPU,
upload the inverse, and let each V-cycle do a dense matvec instead of twenty
sweeps.

It does not need building, because the data already collected bounds what it
could possibly win. From the two barrier-kernel measurements:

    1 dispatch, 20 sweeps    72.3 ms
    1 dispatch,  4 sweeps    68.1 ms

Sixteen sweeps of coarse-level work are **4.2 ms**, so all twenty are about
**5.3 ms of a 72 ms frame -- roughly 7%**. That is the entire prize, and it
is what a *perfect, free* coarse solve would return.

Against it: one mid-frame synchronisation, measured at **3.9 ms**, plus a
dense-matvec dispatch in each of the frame's twelve V-cycles. The cost is
the same order as the prize before any of the machinery is written.

So it is not worth building, and the reason is worth keeping: the coarse
level looked expensive when it was forty dispatches, and the two changes
that have already landed -- batching the V-cycle, then folding the coarse
sweeps into one barrier kernel -- are precisely what took its cost from
"dominant" to "seven percent". **The target was removed by the earlier work,
not by this analysis.**

## What that leaves

Only direction 3, the FFT/DCT preconditioner, and its case is unchanged: it
is the one remaining lever with a large ceiling, because it attacks the
iteration count rather than the cost of an iteration. Everything measured
since batching has been chipping at the second, and the pieces left there
are single-digit percentages of the frame.

The honest next step for it is still its own bound rather than its
implementation: measure how few iterations each scene would need with a very
strong preconditioner (for instance by running many more multigrid V-cycles
per CG iteration and watching where the iteration count bottoms out). That
number decides whether the transform is worth writing, and it is a
measurement rather than a build.

> **Retraction, added by the section after this one.** Every millisecond
> figure in the section below was measured with a harness that warmed up for
> 30 frames and then timed 60. That is not steady state -- this scene needs
> several hundred frames to reach it, and the measured window sat on the
> steepest part of the ramp. The frame times (96.8, 80.4, 66.7, 57.3), the
> 8.0 ms-per-iteration slope, the 3.0 ms-per-round-trip figure and the
> "1.45x" all measured warm-up. They are wrong and are kept only so the
> mistake stays visible. What survives is stated at the end of the next
> section; the correct measurements are there too.

# The bound was measured, the probe was wrong, and the answer was elsewhere

The measurement the section above asked for was run. It produced a number,
the number was wrong, and chasing why it was wrong is what finally located
the bottleneck. Both halves are recorded here, because the wrong number was
convincing.

## The probe: more V-cycles per apply, and why it cannot answer the question

`createMultigridPreconditioner` was given a temporary `vCyclesPerApply`
option -- repeated V-cycles, each warm-started from the last, which is the
standard way to make a stronger preconditioner out of the machinery already
present. Example 15, 64x64, four levels:

| V-cycles per apply | ms/frame | mean CG iterations |
| --- | --- | --- |
| 1 (baseline) | 72.3 | 12.5 |
| 4 | 16.2 | 1 |

A 4.5x speedup, and the iteration count collapsing to 1, is exactly the
shape the FFT direction was hoping for. It is also exactly the shape of a
solver that has stopped solving, which is why it was checked before being
believed. Over 40 frames past warm-up:

| V-cycles per apply | frames converged | max post-projection divergence |
| --- | --- | --- |
| 1 | 40 / 40 | 2.38 |
| 4 | 0 / 40 | 11.43 |

Five times the divergence it was supposed to be removing. Two cycles per
apply is the same failure in its other form: 100 iterations every frame,
`stoppedBy: 'pAp-growth'`, never converged.

`pAp-growth` is this file's own PCG-breakdown detector, and it fires for one
reason: the preconditioner is not symmetric. `B_n = (I - (I - BA)^n) A^-1`
is symmetric whenever `B` is, so composition breaking means the single cycle
was already slightly asymmetric. It is: the smoothing colour order is
reversed on the way up (the mantaflow-derived fix documented in
multigrid.js), but the coarsest level runs `colour 0, colour 1` on every one
of its sweeps and never reverses, so it stands in for an operator whose
adjoint runs them the other way. One cycle is close enough that CG tolerates
it; composing amplifies the gap rather than damping it.

So this probe cannot price a stronger preconditioner without the coarsest
solve being made symmetric first. It was removed rather than fixed, because
by then the bound had been established a cheaper way and it pointed
somewhere else entirely.

## Pricing one CG iteration instead, which needs no new machinery

A perfect preconditioner still costs one iteration. So the ceiling on every
possible preconditioner is just the frame time at one iteration, and that is
measurable by capping the iteration count -- a configuration that does not
solve correctly but times honestly. Example 15, `?maxIter=`, 60 frames after
30 warm-up, each run ending in a readback so queued GPU work is included:

| iteration cap | ms/frame | mean iterations |
| --- | --- | --- |
| 1 | 20.29 | 1 |
| 2 | 26.49 | 2 |
| 4 | 42.55 | 4 |
| 8 | 76.21 | 8 |
| 100 (uncapped) | 96.81 | 11.62 |

Straight line: **12.3 ms fixed, 8.0 ms per CG iteration** (slope from 1 to 8;
the 2 and 4 points sit 1.8 and 1.7 ms under the fit). The ceiling on the
whole preconditioner direction is therefore 96.8 -> 20.3 ms.

That is a large ceiling, and it is also the number that made the real
problem visible. The grid is 64x64. A V-cycle is about 40 dispatches over
4096 cells, which is microseconds of arithmetic, not 8 ms.

## Where the frame actually goes

The profiler, on the uncapped baseline:

- 502 dispatches per frame, 420 of them inside the V-cycle's single batched
  submission
- **CPU-side encoding: 2.28 ms, 2.5% of an 89.8 ms frame**

Encoding is not the bottleneck, which retires dispatch-fusion as a
direction. The other 97.5% is spent waiting. The profiler says on what:
`cg-dot` runs **34.2 times per frame against 11.62 iterations -- 2.94, three
per iteration**. Those are `dotPAp`, `dotRR` and `dotRZ`, and each one is a
`.read()`: a GPU->CPU round trip.

It is not the transfer that costs. A bare 4096-float readback on an idle
queue measures 0.307 ms. It is that a round trip cannot return until
everything queued in front of it has finished, so each one drains the
pipeline -- three times per iteration, cutting the V-cycle's 40 batched
dispatches into pieces that cannot overlap.

Two of the three are load-bearing: alpha needs `pAp` and beta needs `rz`,
both on the CPU, both this iteration. The third, `dotRR`, only answers
"should we stop?".

## Asking the stop question less often: 1.45x, no correctness cost

`residualCheckInterval` (linalg.js, threaded through
grid_pressure_solver2.js, `?checkEvery=` on example 15) evaluates the
true-residual stop test every k iterations instead of every one. The
criterion is unchanged -- the loop still never stops on anything but a true
residual below tolerance, and a final read after the loop makes sure the
residual it reports belongs to the `x` it is actually leaving behind. It is
only asked less often. The default is 1, which is the previous behaviour
exactly.

| checkEvery | ms/frame | mean iterations | converged | max divergence |
| --- | --- | --- | --- | --- |
| 1 | 96.81 | 11.62 | 60 / 60 | 2.38 |
| 2 | 80.44 | 11.10 | 60 / 60 | - |
| 4 | 66.74 | 12.07 | 60 / 60 | 2.17 |

**1.45x**, with every frame still converging and slightly *less* leftover
divergence than the baseline. The cost is the up-to-`k-1` extra iterations a
solve may run past the point it could have stopped, and at k=4 that is worth
about half an iteration against roughly ten round trips saved.

Fitting the round-trip count (`2 * iterations + checks`) against these three
points prices one drain at **about 3.0 ms**, and puts 25-35 of them in every
frame. That is the frame.

## What this reorders

The preconditioner direction (option 3 above) is not dead but it is no
longer first: its ceiling is 96.8 -> 20.3 ms, and it needs an FFT/DCT
transform written from scratch plus a symmetric coarsest solve before it can
even be priced honestly.

Making alpha and beta GPU-resident -- the thing this document has been named
after since it was created -- removes the other two round trips per
iteration. At 3.0 ms each and 11.6 iterations, that is about 70 ms of drain
per frame, and unlike the preconditioner direction it needs no new numerics,
only the scalars kept in device memory and the branch that consumes them
moved onto the GPU. It is now the largest measured item by a wide margin.

A fixed `checkEvery` default is not the right way to ship the 1.45x: k=4
helps a scene that takes twelve iterations and hurts one that would have
converged in two, which is per-scene tuning wearing a global constant's
clothes. The mechanism that avoids it is to predict rather than poll --
CG's residual falls close to geometrically, so two checks give a rate, the
rate gives an estimated crossing iteration, and the next check goes there.
That self-tunes per solve, with no number for a user to pick.


# The harness was wrong, and fixing it reversed the answer

Everything above rests on frame times from one harness: fresh page load,
30 frames of warm-up, time the next 60. Re-running the same configuration
three times in a row through it gave 54.77, 64.66 and 69.78 ms -- a 27%
spread on identical work, which is more than most of the differences it had
been used to decide.

## What the harness was actually measuring

Timing consecutive segments from a single fresh load, `checkEvery=1`:

| segment | frames | ms/frame | mean iterations |
| --- | --- | --- | --- |
| 1 | 30 | 100.65 | 14.03 |
| 2 | 60 | 71.10 | 12.47 |
| 3 | 60 | 52.98 | 11.83 |
| 4 | 60 | 41.86 | 10.05 |
| 5 | 60 | 40.30 | 10.68 |
| 6 | 60 | 58.49 | 11.68 |
| 7 | 60 | 40.57 | 12.27 |

There is a ramp several hundred frames long -- 100 ms down to about 40 --
and the old harness timed segments 1 and 2 of it. The total time gives the
same thing away: 60 frames at "96.8 ms" and 150 frames at "44 ms" are both
about six seconds of wall clock, because both were mostly paying the same
fixed startup.

So the retracted numbers were a warm-up curve sampled at slightly different
points, and the ordering they implied was an artifact. Measured properly,
`checkEvery=1` (44.06 ms) came out *faster* than `checkEvery=4` (46.67 ms)
-- the reverse of what had been reported.

## Why a single frame-time number cannot work on this scene

Warm-up is not the only problem. After 300 frames of warm-up, five
consecutive 60-frame segments gave 41.6, 46.5, 50.9, 67.4 and 71.2 ms --
monotonically rising, with the mean iteration count rising too as the wake
develops. The scene's own pressure problem gets harder over time, and the
machine's clocks drift under sustained load. Comparing configuration A in
one run against configuration B in another compares those two things as
much as it compares A and B.

## Paired measurement, which is what the comparison needed

`residualCheckInterval` moved onto a mutable `settings` object on the
pressure solver, so the policy can change between frames without anything
being rebuilt. The measurement then interleaves both policies inside one
run -- 30 blocks of 10 timed frames, alternating, 150 frames per arm, three
untimed frames between blocks -- so scene evolution and thermal drift act
on both arms equally.

Two runs, the second with the block phase swapped:

| run | arm | ms/frame | mean iterations | **ms per iteration** | converged |
| --- | --- | --- | --- | --- | --- |
| 1 | check every iteration | 87.83 | 14.37 | **6.11** | 150/150 |
| 1 | predictive schedule | 86.08 | 18.11 | **4.75** | 150/150 |
| 2 | check every iteration | 64.27 | 14.31 | **4.49** | 150/150 |
| 2 | predictive schedule | 67.29 | 18.87 | **3.57** | 150/150 |

Frame time says nothing: the predictive schedule is 2% faster in run 1 and
5% slower in run 2. Per-iteration cost says the same thing twice: **21-22%
cheaper, both runs**, for dropping roughly one of the three round trips.

## What that establishes, and what it kills

Round trips are about **three quarters of what a CG iteration costs**. That
is the reproducible result, and it holds up where the frame-time numbers
did not.

It also explains why asking the stop question less often is worth nothing.
The predictive schedule -- fit a geometric rate to the last two true
residuals, predict the crossing iteration, check there -- works as designed:
it cuts per-iteration cost by 21%. It also runs 26-32% more iterations,
because a skipped check means the loop runs past the point it could have
stopped. The two cancel. It was built, tested, measured, and reverted; a
fixed interval is the same trade with worse ergonomics, so the default
stays at 1.

The knob and the mutable `settings` object were kept, because they are the
instrument that produced the 21-22% and the same instrument is what will
have to validate the next change.

## Which makes GPU-resident alpha and beta the whole answer

The other two round trips per iteration -- `pAp` for alpha, `rz` for beta --
are removable without adding a single iteration, which is exactly the
penalty that made the stop-test schedule a wash. Removing all three takes an
iteration from about 4.5 ms to about 1.1 ms on run 2's numbers, with the
iteration count unchanged at 14.3.

Nothing else measured here comes close, and unlike the preconditioner
directions it needs no new numerics -- only the scalars kept in device
memory and the arithmetic that consumes them moved onto the GPU. The
comparison must be paired, and against per-iteration cost rather than frame
time.


# GPU-resident alpha and beta: built, and it is the win

The thing this document has been named after since it was created. alpha and
beta are consumed by kernels (updateX, updateR, updateP); the host was
reading their ingredients back only to divide two numbers and upload the
answer again. Now they are divided where they already live, and the loop
reads one buffer once per iteration instead of making three round trips.

## What had to move with them

Every guard the host loop applies *before* touching x. The host can check
`pAp` and break before updateX ever runs; a loop that only finds out an
iteration later cannot. So the guards moved into the alpha and beta kernels,
in a stronger form: a tripped guard writes the scalar as exactly **0**,
which makes the update it feeds a no-op. x cannot be corrupted even for the
one iteration before the host reads the flag.

Non-finite detection goes through `src/float_guards.js` rather than any
comparison written by hand. That module exists because the documented WGSL
idiom is not dependable -- core WGSL has no `isnan()`, the spec lets an
implementation assume non-finite values never arise, and a compiler may fold
`x != x` to a constant false with no warning
(`examples/27-float-guard-probe/` is the measurement that settled it). A
first draft of these kernels used plain comparisons whose direction happened
to be safe; that is not the same as being right, and it was replaced.

One real accuracy change: `createDotReducer.read()` sums the per-lane
partials in JS doubles, and the new GPU reduction sums them in float32. It
is safe here for a specific reason -- all three dot products are sums of
same-signed terms (`r.r` and `r.z` positive, `p.Ap` consistently negative
for this file's negative semi-definite A) -- so there is no cancellation for
the narrower type to amplify. A mixed-sign dot product would still need the
host sum.

## Measured, paired, iteration counts matched

| run | arm | ms/frame | mean iterations | converged |
| --- | --- | --- | --- | --- |
| 1 | GPU-resident | 52.42 | 14.30 | 150/150 |
| 1 | host | 64.87 | 14.45 | 150/150 |
| 2 | GPU-resident | 49.42 | 14.31 | 150/150 |
| 2 | host | 55.75 | 14.33 | 150/150 |

**1.24x and 1.13x**, second run with the block phase swapped. The iteration
counts agree to within 1%, which is the whole point: this is the change the
stop-test schedule could not be, because it removes round trips without
buying them with extra iterations.

### Correcting the previous section's estimate

That section priced round trips at "about three quarters of what an
iteration costs", from a 21-22% drop in ms-per-iteration. That figure was
inflated by a confound: ms-per-iteration is `(fixed frame cost / iterations)
+ per-iteration cost`, so *any* change that raises the iteration count
lowers it, whether or not iterations got cheaper -- and the predictive
schedule raised it by 26-32%.

The table above has matched iteration counts, so it does not have that
problem. Removing two round trips per iteration saves 12.45 ms and 6.33 ms
per frame across roughly 28.7 removed trips: **one round trip costs about
0.2-0.4 ms**, and three per iteration are 10-20% of the frame, not 75%.

## Equivalence, checked where it actually matters

`examples/05-preconditioned-conjugate-gradient/` now runs both paths, and
adds the two cases that reach the guards -- nothing else in the examples
directory does, and every scene run during this work reported a stop reason
of none, so the rewritten guards would otherwise have shipped unexercised.
On real WebGPU hardware:

| case | host | GPU-resident |
| --- | --- | --- |
| A=diag(1..8), M^-1=diag(1..1/8) | exact answer | exact answer |
| singular operator (A = 0) | degenerate-pAp, x finite | degenerate-pAp, x finite |
| near-null operator (A = diag(1e-12)) | alpha-magnitude, x finite | alpha-magnitude, x finite |

The first guard case is not hypothetical: a fully closed, all-Neumann
pressure domain is singular, and grid_pressure_solver2.js reaches it.

Across the scenes, with the new path as the default:

| example | converged | rejected | non-finite pressures | stop reasons |
| --- | --- | --- | --- | --- |
| 21 irregular container | 250/250 | 0 | 0 | all none |
| 22 multiple colliders | 248/250 | 0 | 0 | all none |
| 26 dye in free surface | 199/200 | 0 | 0 | all none |
| 28 drop into pool | 238/250 | 0 | 0 | all none |
| 29 static droplet | 80/80 | 0 | 0 | all none |

Example 26 was run against the host path in the same session and agreed on
peak pressure to every digit (10.382), with the same converged count.

## Where that leaves the frame

On by default, with `gpuResidentScalars: false` restoring the host path --
which is not dead code: it is what the guards are specified by and what the
equivalence test compares against.

One round trip per iteration is left, and it is the stop test, which
genuinely has to be on the host. Removing it means not asking every
iteration, and the section above measured what that trade costs: the extra
iterations give it all back. So this direction is finished at roughly
1.2x, and the next question is a different one -- the frame still spends
about 500 dispatches on a 64x64 grid, and at 2.28 ms of CPU encoding for
all of them, whatever the rest of that time is, it is not the host.
