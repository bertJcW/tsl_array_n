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
