# Performance investigation: GPU-resident CG alpha/beta ("option 5")

Status: **abandoned, reverted**. `linalg.js`'s `createPreconditionedConjugateGradientSolver` is back
to the CPU-round-trip design (the same one confirmed stable across this whole project's earlier
verification rounds). This document records three independent implementations that were built and
measured on real WebGPU hardware, why all three were slower than the baseline they were meant to
replace, and the evidence for where the cost actually comes from. Kept so nobody re-attempts the
same direction without seeing this data first.

> **This document is now a record of five rounds, and the paragraph above
> describes only the first one.** Round 1 (option 5: alpha/beta computed on the
> GPU with no host readback) was indeed abandoned and reverted. A later round
> revisited the same target with the guards moved *into* the kernels, and that
> one landed -- 1.24x/1.13x paired -- alongside V-cycle dispatch batching
> (1.53x/1.74x) and a single-workgroup coarse solve (1.09x/1.10x): 3.10x/2.63x
> with all three on, all three on by default. The closing section is the most
> recent round -- what a submission costs, and why the CG iteration now goes out
> as one batch. The retractions are part of the record and are marked in place
> rather than edited away.

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
  submission **-- an undercount, and the submission count beside it was never
  taken at all; see the closing section**
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
all of them, whatever the rest of that time is, it is not the host. **The
closing section answers this: ~800 dispatches and ~272 submissions, ~38.8 us
per submission, and about 1% of the frame spent on the GPU.**


# Re-measuring the two earlier optimisations, paired

Batching and the single-workgroup coarse solve were both landed on numbers
from the 30-warm-up/60-frame harness -- the one shown above to be timing a
several-hundred-frame warm-up ramp. Their reported figures (1.1x-1.7x and
1.12x) were therefore not trustworthy, whatever their mechanisms suggested.
Both are now re-measured the same way the GPU-resident change was.

Both were constructor-time decisions, which is the reason they could only
ever be compared across runs. They are now runtime choices instead: every
form each one can take -- two coarse-level queues, each with a batched and
an unbatched submitter, four dispatchers in all -- is built once at
construction, and `settings` on the preconditioner picks per call. That is
the same move `residualCheckInterval` needed, for the same reason.

Example 15, paired, 150 frames per arm, each measured twice with the block
phase swapped and with the other two optimisations left on:

| optimisation | run 1 | run 2 | iterations matched |
| --- | --- | --- | --- |
| V-cycle dispatch batching | 1.53x | 1.74x | 14.41 vs 14.77 |
| single-workgroup coarse solve | 1.10x | 1.09x | 14.41 vs 14.77 |
| GPU-resident alpha/beta | 1.24x | 1.13x | 14.30 vs 14.45 |

The old figures survive re-measurement: batching's range was right, and the
coarse solve's 1.12x lands within noise of 1.09-1.10x. That they were
arrived at unreliably did not make them wrong -- but it was not knowable
until now, which is the point.

## All three together

Everything off against everything on, paired, 100 frames per arm:

| run | all off | all on | speedup |
| --- | --- | --- | --- |
| 1 | 148.96 ms | 48.06 ms | **3.10x** |
| 2 | 135.85 ms | 51.62 ms | **2.63x** |

Iteration counts matched in both (14.36 vs 14.01, 13.62 vs 13.86) and every
frame converged on both arms.

The compound is larger than the product of the three individual figures
(about 2.1x). That is an interaction, not an error: each one was measured
with the other two *on*, and removing a single optimisation from an
already-fast configuration costs less than removing it from a slow one.


---

# What a submission costs, and batching the CG iteration (2026-09-13)

Every earlier round of this document inferred dispatch cost from frame-time
deltas. This one measured it, and the measurement turned one question into two.

## A dispatch and a submission are not the same thing

`tsl_array_n`'s `kernel.js` has said since it was written that every
`renderer.compute()` call is its own command buffer, its own compute pass and
its own queue submit, and that batching several kernels into one call measured
9.3x cheaper *per dispatch*. What nobody had done is price one submission on
its own, with the work held fixed so that only the number of submits varies.

`submit_cost_probe.html`, 64 dispatches of one real kernel, medians of 11
rounds, on the machine this port is developed against:

| submits | 1 | 8 | 16 | 32 | 64 |
| --- | --- | --- | --- | --- | --- |
| CPU ms (JS only) | 0 | 0 | 0.1 | 0.1 | 0.2 |
| wall ms | 2.8 | 3.1 | 3.2 | 4.2 | 5.0 |

Least squares gives **38.79 us per submission** with a fixed term of 2.65 ms,
and a timestamp pair around the batched case says the GPU executes all 64
dispatches in **0.052 ms**. So the cost is neither the JavaScript (3 us per
submission) nor the GPU work: it is the submit path itself.

## The frame, measured

Example 15, 64x64, driver loop paused so this is the simulation without
drawing, `renderer.compute` patched in the live page:

| quantity | value |
| --- | --- |
| wall per frame | **31.8 - 35.9 ms** |
| `renderer.compute()` calls per frame | **259 - 278** |
| dispatches per frame | **740 - 813** |
| dispatches per submission | 2.92 |
| **GPU compute time per frame** | **0.21 - 0.40 ms (~1%)** |

Three things follow. The recorded figure of 502 dispatches per frame was an
undercount, because the profiler only wraps the kernels built through
`buildElementwiseKernel` (~15% of them are built with `tsl_array_n.kernel`
directly and were invisible). At 38.8 us each, 272 submissions is **~10.6 ms,
about a third of the frame** -- and the GPU is idle for all but 1% of it. And
the earlier puzzle -- why batching the V-cycle was worth 1.53x when CPU
encoding is only 2.5% of a frame -- has an answer: batching was never saving
*encoding*. It was saving submissions.

## The change: one submission per CG iteration

The GPU-resident loop made fourteen `renderer.compute()` calls per iteration,
one per kernel, and only the final readback needed to break the submission. The
whole iteration body now goes out as one batch, with the preconditioner as the
single entry that has no `computeNode` -- so `planBatch` calls it in place,
preserving exactly the ordering the loop needs. Both variants (the incremental
residual update and the periodic true-residual recompute) are resolved at
construction and selected per call, so the two can be compared inside one run;
`settings.batchIterations` is the switch, on by default.

Sandbox harness (fake renderer, dispatch count held fixed): **15 submissions
per iteration before, 5 after**, dispatch count unchanged at 64.3. On the real
machine, submissions per dispatch -- the workload-independent number, because
these scenes keep evolving while they are being measured:

| example | submissions/frame (before -> after) | dispatches/submission |
| --- | --- | --- |
| 15 flow-past-cylinder | 49.9 -> 31.4 | 0.345 -> 0.193 |
| 16 karman-vortex-street | 43.9 -> 23.6 | 0.140 -> 0.083 |
| 20 flip-dam-break | 53.5 -> 28.3 | 0.279 -> 0.149 |
| 28 drop-into-pool | 48.1 -> 27.4 | 0.267 -> 0.129 |

## And the part that did not work: the frame time did not move

| example | median ms/frame (before -> after) | mean ms/frame |
| --- | --- | --- |
| 15 | 17.4 -> 17.3 | 13.11 -> 13.61 |
| 16 | 17.5 -> 17.3 | 14.41 -> 13.93 |
| 20 | 16.6 -> 16.6 | 13.14 -> 12.82 |
| 28 | 16.6 -> 16.6 | 14.11 -> 14.38 |

Removing ~20 submissions per frame produced nothing above the noise, and two of
the four mean figures are slightly *worse* for the batched arm. Read naively
that is a regression; read with the workload difference between arms in mind
(the arms run at different simulation times, so they do not execute the same
number of dispatches) it is "no effect either way".

The explanation that fits both this and the 1.53x from V-cycle batching:
**38.8 us is the cost of a submission on an idle queue.** In a real frame the
submissions are issued while the GPU still has work queued, so most of that cost
overlaps with GPU execution instead of adding to the frame. Batching the
V-cycle removed 37 submissions per iteration out of ~50 and showed up; this
change removes 10 of the remaining ~13 and does not.

So the acceptance criterion for this change is **"same dispatches, half the
submissions, frame time within noise"**, and it passes that. Settling the
frame-time question needs a frozen-workload measurement: driver paused, both
arms stepping an identical sequence, GPU timestamps per arm rather than wall
time. That is the next measurement, not the next implementation.

# That measurement, done: 1.15x - 1.56x per solver step

It changes the conclusion, so the previous section is left standing above and
corrected here rather than rewritten.

`measure_frozen.js`: the driver loop is paused (no rAF pacing, no drawing, no
second stepper), and the two arms **alternate every step** rather than every
block. Per-step alternation is what makes this valid -- the two forms are
numerically identical, so alternating every step puts both arms on the *same*
state sequence and therefore on the same workload, which the per-arm dispatch
and CG-iteration counts then confirm. `renderer.compute` is counted per arm.

| scene | method | batched ms/step | unbatched ms/step | speedup | submissions removed/step | dispatches/step |
| --- | --- | --- | --- | --- | --- | --- |
| 15 flow-past-cylinder | FLIP | 16.06 | 24.75 | **1.54x** | 232 | 1416 / 1434 |
| 16 karman-vortex-street | grid | 24.53 | 28.32 | **1.15x** | 169 | 2416 / 2430 |
| 20 flip-dam-break | FLIP | 21.81 | 34.00 | **1.56x** | 284 | 1800 / 1746 |
| 28 drop-into-pool | two-phase FLIP | 55.14 | 82.87 | **1.50x** | 764 | 3934 / 4159 |

Normalised per dispatch, which removes the residual workload drift: 1.52x,
1.15x, 1.61x, 1.42x -- the same picture.

## Why the frame harness could not see it

Because a *frame* is not a *step* in these drivers, and the frames were on the
vsync floor. Example 28: 16.6 ms per rendered frame carrying ~27 submissions,
against 359 submissions per solver step taking 55 ms -- so a frame held roughly
a thirteenth of a step's work and the median was the 60 Hz floor either way.
The change is a large fraction of the solver step and a small fraction of the
rendered frame, and the frame harness could only see the second.

That also repairs the interpretation two sections above: "no measurable frame
time" was true of the frames that were measured and was not evidence about the
solver. It is also why the 38.8 us figure from the submission microbenchmark had
appeared not to transfer -- it does, per step; the frame simply contained too
little of the step for it to show.

## One caveat: the workload matched to within 6%, not exactly

Dispatch counts differ between arms by 0.6-5.7% and CG iterations by up to 3%,
because the dot products are GPU atomic reductions whose summation order is
non-deterministic: `newRTr > oldRTr` occasionally takes a different branch in
one arm than the other, so the arms' iteration counts drift. The drift goes both
ways (example 20's batched arm did 3% *more* work and still won 1.56x; example
28's did 5.7% *less*), and the per-dispatch normalisation agrees with the raw
ratio, so the conclusion holds -- but "identical sequence" is exact only up to
that reordering.


---

# Step 2, measured: the stage batching is worth 6%, not the 10% estimated

`grid_flip_solver2.js` and `grid_two_phase_flip_solver2.js` now submit their
stage groups as batches instead of one submission per stage -- the same change
the CG iteration already got, and with the same shape: everything between two
readbacks goes out together, and the entries that are composite functions
rather than plain dispatchers (`boundarySolver.constrainVelocity()`, the
optional passes, a caller-supplied force) become batch breaks, because
tsl_array_n's planBatch calls those in place. The accumulator resets stay
*outside* the batches deliberately: they are CPU->GPU uploads rather than
dispatches, and a pending upload is only guaranteed to land before a dispatch
that *begins* a pass, so keeping them between batches is what makes the
accumulators provably zero before the scatter reads them.

Measured with the frozen harness (`which=stages`, per-step alternation, CG
batching left on in both arms):

| scene | batched ms/step | unbatched ms/step | speedup | submissions removed/step |
| --- | --- | --- | --- | --- |
| 20 flip-dam-break | 32.72 | 32.67 | 0.998x | 15.2 of 239 |
| 28 drop-into-pool | 43.38 | 46.09 | 1.062x | 17.0 of 364 |

**About 6% of the submissions, and nothing measurable in time** -- against an
estimate of roughly 10% of the frame. The estimate was wrong for a reason worth
recording: it assumed the FLIP step's submissions sit at its top level. They do
not. Batching the top-level entries removes 15-17 of them, so the other ~140
per step are *inside* composites:

- `boundarySolver.constrainVelocity()`, called three times per step, which runs
  the collider blocks and the domain boundary as its own dispatches;
- the P2G scatter and its finalize, built by `buildScatter`/`buildFinalize`;
- the accumulator/count resets.

Those have to expose their dispatchers before they can join a batch. That is a
change to the boundary solver and the scatter builders rather than a call-site
edit, and it is the version of this step that would be worth measuring.

The change is kept because it is numerically identical -- same dispatches, same
order -- it is a prerequisite for that deeper version, and it is neutral rather
than negative. On its own it does not earn its complexity, and saying so is the
point.


---

# The collider rebuild, and why a scene was running at 3 fps

The largest win found in this whole effort, and it was not in the solver.

## How it surfaced

Example 23 (`23-flip-moving-collider`) was measured as the collider case for the
step-2b decision. Its step turned out to cost **367 ms** -- against 24.3 ms for
`onAdvanceTimeStep()` alone and 0.5 ms for `setCollider()` alone, which do not
add up. `rigidCollider.update(dt)` was the obvious suspect and the obvious
suspect was wrong: timed inside the real step it is **0.43 ms**, 0.1%. The
missing ~335 ms is an *interaction* between two cheap calls.

## What it is

`setCollider()` calls `rebuildColliderKernels()`, which builds about fifteen new
TSL kernels, and three.js's pipeline cache is keyed on the compute node. Kernels
built a moment ago therefore miss the cache, and the next solve compiles them:

| measurement | value |
| --- | --- |
| step, rebuild every call (the old behaviour) | **351.5 ms** |
| step, rebuild skipped | **44.3 ms** |
| compute pipelines created per step, old | **28** |
| compute pipelines created per step, new | **0** |
| submissions per step, both arms | 328 / 327 |
| dispatches per step, both arms | 1594 / 1583 |

**7.94x** as the paired mean (351.5 / 44.3) and 8.46x as the paired median
(362.1 / 42.8). The dispatch and submission counts are the control: the *work* is
identical, only the compilation is gone. The isolated run below, which does not
interleave with an arm that allocates 28 pipelines a step, measured 33.5 ms per
step -- about 11x against the 367 ms the same scene measured before the fix, but
that comparison is across runs and only indicative.

## Why skipping the rebuild is correct

A collider that moved does not need new kernels.
`createSDFRigidBodyCollider2`'s `update()` re-rasterises the posed polygon
through `collider.addPolygon()`, which computes the SDF on the CPU and writes it
with `grid.data.fromArray(hostSdf)` -- **the same field object, new contents**.
Every kernel built here reads that field through
`collider.sample()`/`gradient()`/`isInside()`, so the field's *identity* is what
they hold, and no geometry is baked into them. The existing kernels read the new
geometry on the next dispatch by construction.

So `setCollider()` now skips `rebuildColliderKernels()` when it is handed the
same collider object with the same grid parameters, and still runs
`buildBlockMarker()` every call -- which cells are solid *does* depend on where
the collider is now. The decision belongs to the caller, not to this file: only
the caller knows whether it is re-binding the same collider or a different one,
and a library that guesses has to guess wrong on one of the two.
The two calls were then checked against each other directly, because "the kernels
still read the same field" is an argument and not a measurement. The test holds
the input still -- it snapshots the velocity field, runs one pass, restores the
input with `fromArray`, runs the other pass -- and compares the two outputs
element by element, with each call at the *same* position in its own sequence so
that leftover scratch state cannot masquerade as a difference:

| pair | max abs difference in the resulting velocity |
| --- | --- |
| `colliderMoved()` twice in a row | **0** |
| `setCollider()` twice in a row | **0** |
| `colliderMoved()` against `setCollider()` | **0** |

> **Correction, same session, after the pose went live.** The claim above is
> too strong and the measurement behind it was blind in a specific way.
> `SDFRigidBodyCollider2.velocityAt()` baked `currentPosition`/`currentAngle` into
> the graph at *build* time, and the boundary solver calls `velocityAt` in eight
> places -- so keeping the kernels gave a moved collider a **stale velocity**.
> Example 23 rebuilds every frame for exactly that reason, which is why it was
> the one scene that must not have used `colliderMoved()` as it then stood. The
> test above missed it because it ran a `setCollider()` rebuild *before* taking
> the "old kernels" snapshot, so both sides carried the same pose.
>
> Fixed at the source instead: the pose is live data now (`tsl_array_n.array0`
> fields published by `update()`), which is what `sdf_collider2.js`'s own
> `createSDFRigidBodyCollider2` comment had prescribed from the start. Re-measured
> with the ordering corrected, kernels kept and kernels rebuilt are bit-identical
> **for a translating collider too** (a translating *and* rotating move, then
> `colliderMoved()` before any rebuild against `setCollider()`: 0 difference on
> both velocity components), and example 23's page went from **2.5 fps to
> 29.5 fps** on the very page a user opens.

Bit-identical. An earlier version of that test compared the two calls at
different positions in the sequence and reported differences of 0.5-0.76 -- which
was `constrainVelocity`'s own scratch state (its `uTemp`/`validA` fields), not
the collider, and the tell was that the difference persisted with the collider
held stationary.

## Verification

- The paired run above: identical submissions and dispatches, CG iterations
  within the usual drift (11.75 against 11.55).
- 200 steps of the moving collider with the skip on: **zero non-finite values**,
  peak speed 46.3, 33.5 ms per step in isolation (faster than the 44.3 ms
  measured while alternating with an arm that was allocating 28 pipelines per
  step).
- `fluxflow` 347 tests, `tsl_array_n` 23, all pass.


# What the V-cycle is worth: Jacobi and no preconditioner, measured (2026-09-14)

MGPCG issues ~64 dispatches per CG iteration against a grid of 4096 cells,
and a dispatch on that grid is almost entirely fixed overhead. That ratio
raises an obvious question -- would a preconditioner that needs more
iterations but issues *one* dispatch win? -- and the honest position was
that nobody knew, because there was nothing in the package to compare
against. `createJacobiPreconditioner` (z = r / diag(A)) and
`createIdentityPreconditioner` (z = r, making PCG plain CG) exist so the
question has an answer.

## Correctness first

All three must reach the same solution; a preconditioner changes the path,
not the answer. From an identical warmed state on example 15, restoring
the velocity and pressure fields before each arm and running one step:

| comparison | max abs difference, u | v |
| --- | --- | --- |
| multigrid vs jacobi | 1.9e-5 | 1.4e-5 |
| jacobi vs none | **4e-6** | **4e-6** |

against a peak velocity of 3.545. The second row is the self-check the
code predicted before the run: for a constant-coefficient Laplacian
diag(A) is uniform, so Jacobi is a scalar multiple of the identity and
cannot change the Krylov subspace. It measures as the same solver.

## Paired, round-robin, every arm warmed

| example 15 (uniform, cylinder mask, cap 2000) | ms/step | iterations | converged |
| --- | --- | --- | --- |
| **multigrid** | **32.5** | **9.8** | 12/12 |
| jacobi | 697.6 | 342.8 | 11/12 |
| none | 385.1 | 196.9 | 12/12 |

| example 28 (variable density 1.40, cap 100) | ms/step | iterations | converged |
| --- | --- | --- | --- |
| **multigrid** | **92.2** | **39.7** | 9/10 |
| jacobi | 162.6 | 100 (capped) | **0/10** |
| none | 134.5 | 100 (capped) | **0/10** |

## Two results

**The V-cycle is worth about 20x in iterations and 12x in wall time.** That
is the number this document has been implicitly assuming for its whole
length without ever measuring it. Thirty-six dispatches per application
buys a factor of twenty in iterations, and on this grid that trade is
clearly right rather than marginal. The dispatch-count worry that motivated
the experiment was real but badly mispriced: cutting 36 dispatches to 1
costs 20x the iterations, and each of those iterations still carries the CG
vector operations and a host round trip.

**Jacobi is worse than no preconditioner.** 342.8 iterations against 196.9
on example 15, and slower at equal capped iterations on example 28 -- two
scenes, same direction. The prediction written into the code beforehand was
"about the same". Being reliably *worse* means the diagonal varies in a way
that hurts, and the likely reason is that most of the variation is
artificial: a Dirichlet-masked row's diagonal is -1 where an interior row's
is -4/h^2, so Jacobi rescales the masked rows by four relative to the fluid
and those rows are not part of the problem. That is a hypothesis about a
rejected option, not a measured cause.

Note also that the variable-density scene -- the one case where theory says
Jacobi should earn its keep -- is where both cheap arms fail hardest: zero
converged solves in ten steps. Non-convergence is not a quality setting
here. It is the condition this project has already measured to destroy a
free-surface liquid (Debugging #5 in `project-history.md`: occupied cells
1536 -> 340-700).

## Kept, but not offered as an alternative

The option stays because it is the instrument that produced the 20x, and
because a future change to the operator -- a much larger density ratio, a
non-uniform grid -- can be re-checked against it inside one run. The
default is unchanged and the option's own documentation says to use it.

This also closes the direction the TouchDesigner/LiquiGen research opened
(`realtime-fluid-tools-research.md`). "Fewer dispatches per iteration" was
one of three suggestions there; measured, it is the wrong axis. The other
two -- making the V-cycle itself cheaper in dispatches, and a fixed
iteration count with a rare convergence check -- are untouched by this
result, because neither of them gives up the V-cycle's factor of twenty.


# Step 0: what a solver step is actually made of (2026-09-14)

Three optimisation directions were queued off the back of the submission
work -- fold `mg-clear` into `restrict`, swap red-black for damped Jacobi,
make the transfer operators exact transposes -- and all three target the
V-cycle. Before building any of them, this prices the step they live in.

## The observational form does not work

Regressing step time on the iteration count over 70 consecutive steps, no
intervention: **R-squared 0.009**. The natural spread of the iteration
count on a settled scene is 8-12, and the timing noise is larger than the
signal across that range. `maxIterations` therefore moved onto the runtime
`settings` object so the cap can be varied inside one run.

## Interventional, paired, 20 rounds, medians

State (u, v, pressure) restored before every arm so each solves the
identical problem:

| iteration cap | iterations | median ms |
| --- | --- | --- |
| **0** | 0 | **10.5** |
| 1 | 1 | 11.5 |
| 2 | 2 | 12.3 |
| 4 | 4 | 14.1 |
| 8 | 8 | 17.2 |
| 16 | 9 (natural) | 18.6 |

**0.861 ms per CG iteration**, intercept 10.61 ms, **R-squared 0.996**. The
intercept agrees with the directly measured zero-iteration point (10.5 ms)
to 1%, which is the cross-check that makes the fit believable.

## Splitting the intercept

The zero-iteration point is *not* "everything except the pressure solve" --
it still runs the solve's whole setup. Timing the projection on its own
against the full step, alternating, same restored state:

| | ms | iterations |
| --- | --- | --- |
| full solver step | 22.4 | 12 |
| pressure projection alone | 21.0 | 13 |
| **everything else** | **1.4** | -- |

The model checks out: at 13 iterations it predicts 10.5 + 13 x 0.861 =
21.7 ms against 21.0 measured, within 3%.

## The accounting

| component | ms | share |
| --- | --- | --- |
| non-pressure stages (advection, forces, dye, boundary) | 1.4 | **6%** |
| CG iterations (0.861 x 12) | 10.3 | **46%** |
| pressure-solve fixed cost | ~9.1 | **~40%** |

Two things follow, and both reorder the queue.

**The non-pressure half of the solver is 6% of it.** Everything left to win
is inside the pressure solve.

**The fixed cost of one solve is about as expensive as all of its
iterations.** Nothing has ever targeted it -- every optimisation in this
document attacks per-iteration cost -- and all three queued directions
attack the V-cycle, which lives inside the 46%. Halving the V-cycle's
dispatches cannot touch the 40%.

## Inside the fixed cost

Per solve, outside the CG loop: `updateDirichletFields`,
`dispatchBuildSystem`, then inside `solve()` an `applyToX`, an `init`, a
`dotRR` read, **a full V-cycle** (the initial preconditioner apply), a
`dotRZ` read, an `updateP` and a closing residual read; then
`countBadPressureCellsNow`, the snapshot or restore, and the two velocity
corrections. That is roughly four host round trips and one V-cycle.

One of them is now priced. `settings.checkBadCells` (a measurement
instrument, guarded by a comment that says so) turns off the circuit
breaker's readback:

| | median ms | iterations |
| --- | --- | --- |
| check on | 29.9 | 11 |
| check off | 28.8 | 11 |

**1.1 ms**, about 12% of the fixed cost. That also corrects an earlier
figure in this document: a host round trip was estimated at 0.2-0.4 ms
from the GPU-resident alpha/beta work; measured directly on an equivalent
one it is **1.1 ms**.

The remaining ~8 ms is unattributed. The candidates are the three other
round trips and the initial V-cycle, but four round trips at 1.1 ms plus a
V-cycle at 0.9 ms only reaches ~5 ms, so something in
`dispatchBuildSystem` / the corrections / the snapshot is unaccounted for.
That is the next measurement, and it is worth more than any of steps 3-5.

## A correction to this session's own earlier number

The preconditioner comparison above reports 32.5 ms for a multigrid step.
That was measured in a three-way round-robin whose other two arms take 385
and 698 ms per step, which moves the GPU's clocks. A clean natural step is
**18.6-22.4 ms** depending on warm-up. The preconditioner *ratios* stand --
both arms of each comparison sat in the same thermal environment -- but the
absolute figure does not.

## Finding the missing 8 ms: phase timing

The ablations above priced two pieces of the fixed cost and left ~8 ms
unattributed. `profiling.js` gained `markPhase`/`timePhase` for the rest,
because counting dispatches says nothing about where a *wait* goes, and a
round trip's cost IS the wall time of its await -- it cannot return until
the queue in front of it has drained. Example 15, 50 steps, profiling on:

| phase | calls/step | ms/step |
| --- | --- | --- |
| `pressure-cg-solve` (the whole solve) | 1 | **23.02** |
| `solve-iteration-read` (the in-loop readback) | 11.5 | **13.93** |
| `solve-setup-readRR` | 1 | **5.38** |
| `solve-setup-readRZ` | 1 | 1.04 |
| `pressure-badcells-read` | 1 | 0.83 |
| *step wall* | | *26.69* |
| *of which CPU encoding* | | *3.78* |
| *submissions* | *30* | *~1.2 (at 38.79 us)* |

**Host round trips are 21.2 ms of a 26.7 ms step -- 79%.**

The missing 8 ms was never separate work. It is inside
`solve-setup-readRR`: that is the first readback of the solve, so it
drains everything queued ahead of it -- `dispatchBuildSystem`,
`applyToX`, `init`. One await, 5.38 ms, absorbing the whole prologue.

Two things this retires:

- **Submissions are no longer the story.** 30 per step at 38.79 us is
  ~1.2 ms, 4% of the step. The batching work already collected that; the
  272-per-frame figure predates it.
- **The V-cycle is not the story either.** Priced by ablation (preconditioner
  `multigrid` vs `none` at a zero iteration cap, so the only V-cycle is the
  initial preconditioner apply): **0.2 ms**. Against 0.861 ms for a whole CG
  iteration, the V-cycle is under a quarter of it. Every one of the three
  queued directions -- fold `mg-clear` into `restrict`, red-black to damped
  Jacobi, exact-transpose transfer operators -- attacks that 0.2 ms.

### Two numbers that look contradictory and are not

The cap sweep gives **0.861 ms as the marginal cost of one more iteration**.
The phase timer gives **1.21 ms as the observed cost of one in-loop read**
(13.93 / 11.5). A read cannot cost more than the iteration containing it.

Both are correct because they measure different things. An await's wall
time includes draining work that was queued before it, so phase timing
says *where the cost is observed*; the sweep's slope says *what one more
iteration adds*. They must not be added together.

### The actionable gap

The largest single item is the per-iteration readback, 13.93 ms, 52% of
the step -- and **there is currently no way to reduce it on the default
path**. `residualCheckInterval` exists for exactly this, but it is
consulted only in the host `solve()`; `solveWithGpuResidentScalars`, which
is the default, reads `scalars` unconditionally every iteration. Measured:
`checkEvery` 1 / 2 / 4 / 8 all ran 9 iterations and differed only by noise
(18.8 / 22.1 / 20.7 / 20.4 ms), which is what a knob that does nothing
looks like.

That also revisits an earlier conclusion in this document. The predictive
stop-test schedule was reverted as a wash, on the reasoning that skipping
a check costs extra iterations. Under the numbers now measured -- 1.21 ms
observed per read against 0.861 ms marginal per iteration -- that trade is
worth re-testing, but only once the GPU-resident path can actually skip
the read. It cannot today, so the earlier measurement was made on a path
where the knob was inert.

**Next: make the GPU-resident loop's readback conditional, then re-run the
interval sweep.** That is the one change the accounting points at, and
it is worth more than steps 3-5 combined.

## Testing steps 3 and 5 rather than rejecting them by inference

The accounting above rejected all three queued V-cycle directions on the
grounds that a V-cycle is 0.2 ms. That figure came from a difference of
medians (10.6 - 10.4) on a base of 10.5 ms, which is 2% -- **inside the
noise floor**, so it did not establish anything. Re-measured properly, and
then two of the three built and run.

### The gate: what a V-cycle actually costs

`tolerance` moved onto the runtime settings so a solve can be made never
to converge, which pins both arms to exactly the same iteration count and
multiplies the signal by that count. At `tolerance: 0`, cap 40,
preconditioner `multigrid` against `none` (whose apply is a single copy):

| | ms/step | iterations |
| --- | --- | --- |
| multigrid | 45.10 | 40 |
| none | 38.90 | 40 |
| difference | 6.20 over 40 cycles | |

**0.155 ms per V-cycle**, now with 40x amplification and matched iteration
counts. The earlier 0.2 ms was right, but it was not measured; this is.

Against 0.861 ms for a whole CG iteration, a V-cycle is under a fifth of
one. That bounds step 3 at ~0.6% of a step and step 4 at ~3.5%.

### Step 5: the premise is wrong

Restriction and prolongation are not an adjoint pair, and
`examples/07-multigrid-preconditioned-cg/` now measures it on the real
kernels: `(Ru, v) = -0.149227` against `(u, Pv) = -0.596909`, a relative
mismatch of 75%.

But the mismatch is **exactly a factor of 4**, and that is the whole
question. Over three random draws the ratio is **4.000000, 3.999999,
4.000003** -- spread 3.79e-6, i.e. constant to float32 round-off. 4 is
2^d for d = 2.

If `R = c P^T` for a fixed c, the coarse-grid correction
`P A_c^-1 R = c (P A_c^-1 P^T)` is still symmetric. **A constant scale
factor does not break the property PCG needs.** A ratio that wandered
between draws would; this one does not.

So step 5 is not the cause of anything, and the "more likely true cause of
V-cycle asymmetry" it was filed under is not a cause at all. The check
stays in example 07 as a regression test, phrased as "differ by a
constant" rather than "are adjoint", because the constant is the finding.

### Step 3: correct, and worth nothing measurable

`buildRestrictKernel` can now write the coarse level's zeroed `x` from the
same dispatch that restricts into its `b`
(`settings.multigrid.foldClearIntoRestrict`, off by default). Different
arrays, same shape, no aliasing.

**Bit-identical, confirmed rather than argued**: same restored input, one
step each way, max absolute difference **0** on both velocity components,
100% of cells exactly equal, 12 iterations either way.

Paired timing, twice, phase swapped:

| run | fold off | fold on | delta |
| --- | --- | --- | --- |
| 1 | 21.9 ms | 24.3 ms | fold **2.4 ms slower** |
| 2 (phase swapped) | 16.2 ms | 15.9 ms | fold **0.3 ms faster** |

The direction flips, so it is noise -- which is what the 0.155 ms V-cycle
predicts: three dispatches out of ~36 is ~0.013 ms per cycle, ~0.15 ms per
step, an order of magnitude below what this harness can resolve. The
absolute step time also moved from ~22 ms to ~16 ms between the two runs,
which is the usual reminder that only within-run pairs compare.

Kept behind a default-off flag: it is numerically free and it is a
prerequisite for anything that fuses more of the V-cycle. It is not a
speedup.

### Step 4: not tested, and why it is a bigger change than the other two

Damped Jacobi cannot update in place. Red-black is safe as two dispatches
precisely because a cell's neighbours are the other colour; a Jacobi sweep
reads every neighbour from the *old* iterate, so writing into the array it
is reading is a race. Doing it properly needs a second `x` per level and
ping-pong bookkeeping through the cycle -- with an even number of sweeps
each way (2 down, 2 up) it lands back in the original buffer, so no copy
is needed, but it is a structural change to the level state rather than a
flag.

Its ceiling, from the measured V-cycle cost: halving 0.155 ms across ~12
iterations is **~0.9 ms of a ~20 ms step, about 4%** -- and only if the
sandbox's iteration-count pricing (29-31 against 30) survives contact with
the real operator, whose diagonal varies by 4x between masked and interior
rows. The damping factor is also a constant with 7x leverage on the result
(omega = 1 gives 204 iterations against 29 at 2/3), which is the shape of
number this project has twice removed rather than tuned.

### Step 4: built in a sandbox copy, measured, rejected

The change is structural rather than a flag -- a Jacobi sweep is defined
on the old iterate, so it cannot write into the array it reads, and every
level needs a second `x` with ping-pong through the cycle. Rather than
make `src/` pay for that before knowing whether it is worth it,
`sandbox/jacobi-smoother/multigrid_jacobi.js` is a copy of
`src/linalg/multigrid.js` with exactly that one change, and
`examples/30-jacobi-smoother-sandbox/` hands both to the same operator and
the same CG solver so the smoother is the only difference. Both arms run
with the single-workgroup coarse kernel off, since it is a red-black
construction with no Jacobi counterpart.

Ping-pong needs no copies: every sweep count in the cycle is even (2 down,
2 up, 2 final, 20 coarsest), so a run of sweeps always lands back in `x`
and nothing downstream knows the second buffer exists.

Iterations to `1e-5`, 32x32, four levels:

| operator | red-black | omega = 2/3 | omega = 0.8 | omega = 1.0 |
| --- | --- | --- | --- | --- |
| plain Poisson | 5 | 7 (1.40x) | 6 (1.20x) | 33 (6.60x) |
| Dirichlet mask (a circle) | 10 | 10 (1.00x) | **9 (0.90x)** | 72 (7.20x) |
| variable density, 8:1 | 19 | 21 (1.11x) | 20 (1.05x) | **did not converge (400)** |

**Rejected, on three grounds the measurement establishes rather than
suggests.**

*The saving is already spent.* Halving a 0.155 ms V-cycle across ~12
iterations is ~4% of a step, so any iteration ratio above ~1.04 costs more
than it saves. At the best damping factor the ratios are 1.20, 0.90 and
1.05 -- averaging just past break-even, and negative on the plain operator.

*The damping factor does not transfer.* The float64 reference priced
damped Jacobi at 29-31 iterations against red-black's 30, i.e. free. That
held for the masked case (1.00x at 2/3) and did not for the plain one
(1.40x). Which omega is best also moves between operators, and this
project has twice removed a constant of exactly this shape rather than
tune it per scene.

*Getting it wrong is not graceful.* Undamped Jacobi is 6.6x and 7.2x on
the first two operators and **does not converge at all** on the
variable-density one -- which is the operator the liquid scenes actually
use, and non-convergence there is the condition already measured to
destroy a free-surface liquid.

The one real attraction survives and is worth recording: a Jacobi sweep
has no colour order, so the V-cycle is symmetric by construction rather
than by remembering to reverse post-smoothing. That is a robustness
argument, not a performance one, and after the step 5 measurement above
there is no known symmetry defect left for it to fix.

## Where steps 3, 4 and 5 leave the queue

All three tested, none kept:

| | claim | measured |
| --- | --- | --- |
| 3. fold `mg-clear` into `restrict` | ~58 dispatches/frame, zero risk | bit-identical, and within noise (-2.4 ms then +0.3 ms) |
| 4. red-black to damped Jacobi | 25 to 13 dispatches, free in iterations | 1.05x-1.20x iterations at best omega; catastrophic at omega = 1 |
| 5. exact-transpose transfer operators | the likely cause of V-cycle asymmetry | R = P^T/4 exactly; a constant factor, symmetry intact |

They were all priced against the V-cycle, and the V-cycle is 0.155 ms of a
~20 ms step. The accounting section above says where the step actually
goes: **host round trips, 79%** -- and the single largest item, the
per-iteration readback at 52%, still has no way to be reduced on the
default path.

# The readback the accounting pointed at (2026-09-14)

The step accounting found host round trips at 79% of a solver step, with
the per-iteration readback the largest single item -- 13.93 ms of 26.69,
52% -- and no way to reduce it, because `residualCheckInterval` was
consulted only in the host `solve()` while the default GPU-resident path
read `scalars` unconditionally. This makes that read conditional.

## What had to change for skipping to be safe

**The stop code is now sticky.** `alphaKernel` runs every iteration and
the host no longer looks every iteration, so an unconditional write would
let a clean iteration erase the record of a guard that tripped before it.
A tripped guard already leaves its scalar at exactly 0, so the updates it
feeds are no-ops and x cannot be corrupted while the host is not looking
-- but beta is computed from an unchanged r.z and the loop would
otherwise carry on as though nothing had happened. The kernels now keep
the first non-zero code, so the host finds out whenever it next looks.

**A final read after the loop.** The loop can exit on an iteration that
skipped its read, leaving both the residual and the stop code stale, and
a stale "no guard tripped" is the dangerous one.

`examples/05-preconditioned-conjugate-gradient/` runs its two guard cases
at interval 4 as well as 1, which is the case the sticky code exists for:
a guard tripping on an iteration whose readback is skipped. Both cases
report the right reason from all three arms and leave x finite.

## Measured

Example 15, paired, state restored before every arm, medians of 24:

| interval | run 1 ms | run 1 iters | run 2 ms | run 2 iters |
| --- | --- | --- | --- | --- |
| 1 | 40.5 | 13 | 20.6 | 12 |
| 2 | 26.1 | 13 | 20.5 | 13 |
| **4** | **19.1** | 13 | **16.6** | 13 |
| 8 | 20.1 | **17** | 18.1 | **17** |

4 is fastest in both runs and 8 gives the win back by pushing the
iteration count 13 -> 17. The magnitude does not reproduce (2.12x then
1.24x, the first run's every-iteration arm being anomalously slow), so the
honest figure is **1.24x on this scene, possibly more**; the direction and
the shape of the curve reproduce exactly.

Example 28, a liquid with variable density, 250 steps per interval,
scene re-seeded each time:

| interval | converged | rejected | mean iterations | peak pressure | non-finite |
| --- | --- | --- | --- | --- | --- |
| 1 | 240/250 | 0 | 33.8 | 15.799 | 0 |
| 2 | **250/250** | 0 | 29.9 | 15.614 | 0 |
| 4 | **250/250** | 0 | 30.5 | 15.614 | 0 |
| 8 | **250/250** | 0 | 31.3 | 15.614 | 0 |

Checking less often converges **more** often, which is not a paradox: the
loop has run further by the time it asks, so it is more likely to be under
tolerance when it does. The ten non-converged frames at interval 1 ran to
the cap, which is also what inflates that row's mean iteration count.

Example 29, which converges in a single iteration, is unaffected --
1 iteration at every interval. Iteration 0 is always a check point, so an
easy solve still stops immediately and the feared "a 1-iteration scene now
runs 4" does not happen.

## The default is now 4

Verified on the new default: example 26 converges **250/250** (was
199/200) and example 22 **250/250** (was 248/250), both with zero
rejections, zero non-finite pressures, and peak pressures identical to
their historical values to every digit (10.382 and 20.765).

This is a global constant rather than a per-scene one, and what it prices
is hardware -- a round trip against an iteration -- not scene shape.

## What this says about an earlier retraction

The predictive stop-test schedule was built and reverted earlier in this
document as "a wash". Its premise was right; the measurement was made on
a path where the knob was inert, because the default had already moved to
the GPU-resident loop and that loop ignored the interval entirely. The
fixed interval is the simpler thing and it works. The predictor is still
not worth rebuilding -- 4 is one constant with measured behaviour at 1, 2,
4 and 8 on three scenes, against a rate estimator with a safety factor.

## The setup round trips (2026-09-15)

Two of the three reads outside the CG loop were moving numbers from one
place on the GPU to another by way of the host:

- `dotRZ.read()` produced `initRZ`, whose only use was `seed[SLOT_RZ_OLD]`.
  `reduceRZ` had already written that value into `SLOT_RZ`.
- `dotRR.read()` produced `initRTr`, used for `SLOT_PAP_BASELINE` (a GPU
  slot), for the drift detector's seed, and -- the one genuinely host-side
  use -- for the early exit when a solve arrives already converged.

`seedScalarsKernel` does the first two on the GPU. The early exit is gone;
an already-converged solve now runs one iteration and stops at the first
check, which is iteration 0. That case trips `degenerate-pAp` on the way
(p is ~0, so p.Ap is ~0) and alpha is forced to 0, leaving x exactly where
it belonged -- so the loop now reports a guard name only when the residual
is still above tolerance, because a guard tripping on a converged residual
is not a failure.

**Bit-identical**: same restored input, max absolute difference **0** on
both velocity components, same 13 iterations, same `converged`, same
`stoppedBy`. Which is what should happen -- the values are the same, they
just stop going round the houses.

Paired, twice, phase swapped: **1.144x** (30.2 -> 26.4 ms) and **1.297x**
(19.2 -> 14.8 ms). The phase report confirms the mechanism rather than
inferring it -- `solve-setup-readRR` and `solve-setup-readRZ` are simply
absent afterwards, leaving `solve-iteration-read` and the circuit
breaker's 0.73 ms.

On by default. Verified on the new defaults (`gpuResidentSetup` on,
interval 4): example 26 converges 250/250 with peak pressure 10.382, its
historical value to every digit, and example 28 249/250 -- both with zero
rejections and zero non-finite pressures.

### A default that was being overridden

Example 15 passed `residualCheckInterval: 1` explicitly, which silently
opted the scene every performance measurement runs on out of the library
default that had just been changed to 4. It now passes the option only
when `?checkEvery=` is given. Worth noting as a hazard: a default is not
in force anywhere a caller names the option, and the caller here was the
benchmark.

### What is left of the fixed cost

Of the ~9 ms of per-solve fixed cost the accounting found, the two setup
reads were ~6.4 ms and are gone. The circuit breaker's read is 0.73 ms
and stays -- it is the guard this project shipped broken twice, and
making it periodic is a change in safety posture rather than a free win.
The remaining in-loop reads are now the whole story again.

# "The GPU is idle 99% of the time" -- re-measured (2026-09-15)

That figure comes from a 2026-09-13 run: 0.21-0.40 ms of GPU compute in a
31.8-35.9 ms frame. Since then the in-loop readbacks went from every
iteration to every fourth, the two setup round trips were removed, and
frames came down to 16-28 ms -- all of which changes the denominator, so
the share had to be measured again rather than quoted.

**It is current.** Three scenes, paused driver, warm, timestamps resolved
every step:

| scene | wall / step | GPU compute / step | busy |
| --- | --- | --- | --- |
| 15 flow past cylinder | 27.45 ms | 0.218 ms | **0.79%** |
| 20 FLIP dam break | 16.28 ms | 0.131 ms | **0.81%** |
| 28 drop into pool | 27.83 ms | 0.173 ms | **0.62%** |

The optimisations landed since did not move the ratio, and that is the
expected result: every one of them removed *host* work (fewer round
trips, fewer submissions, less encoding), so both numerator and
denominator shrank together. The GPU was never the constraint.

## Where the 27 ms goes

From the same page's `profile( 40 )` (example 15, 892 dispatches and 44.6
submissions per step):

| | ms / step | share |
| --- | --- | --- |
| `pressure-cg-solve` total | 14.23 | 70% |
| ...of which `solve-iteration-read` (5 round trips) | 10.63 | 52% |
| ...of which the circuit breaker's read | 0.80 | 4% |
| CPU-side dispatch encoding | 4.98 | 25% |
| GPU compute | 0.22 | **1%** |

A host round trip costs ~2.1 ms and there are six of them. Encoding 892
dispatches costs 5 ms at ~5.6 us each. The work itself costs 0.22 ms. The
library is not GPU-bound, it is **latency-bound and encode-bound**, and
the ceiling if both went to zero is roughly a 90x step-time reduction --
which is the size of the prize the two obvious directions are chasing:
fewer, larger dispatches (persistent-kernel or indirect-dispatch V-cycles)
and a stop test that never leaves the GPU (the residual-gap estimator
parked in project-history.md).

## A profiler bug this exposed

`readComputeTimestampMs` tested `renderer.trackTimestamp`, and three.js
keeps that flag on the *backend* -- `Backend`'s constructor sets
`this.trackTimestamp = ( parameters.trackTimestamp === true )` and nothing
copies it up to the renderer. So the test read `undefined`, and the
function returned `null` for every measurement taken before today, on a
machine whose adapter does support `timestamp-query`. That is why the
2026-09-13 number could only be quoted, not reproduced. Fixed.

The second half of the fix is that the query pool must be drained every
step: it holds 2048 queries = 1024 timestamped passes, a step here has
~159, and `updateTimeStampUID` keys allocations off `info.compute.frame`
-- which does not advance while the rAF driver is paused, so a single
resolve after 40 steps overruns the pool. Example 15's `profile()` now
times the frames in one pass (no resolves, so mapAsync latency is not
counted as frame time) and measures GPU time in a second pass, resolving
per step. Examples 20 and 28 accept `?profile=1` for the same reason.

# What a dispatch and a round trip actually cost, and the 107 single-dispatch submissions (2026-09-16)

The GPU-idle re-measurement above says the step is host-bound but not what
the host is paying for. These are microbenchmarks against raw WebGPU on the
same device, in the same page, so the library's numbers can be compared
against the API's own floor.

## The floor

| operation | cost |
| --- | --- |
| encode one dispatch (pipeline + bind group + dispatchWorkgroups) | **0.56 us** |
| `queue.submit` of an empty command buffer | **4.44 us** |
| 900 dispatches across 45 submissions, encode + submit | **0.40 ms** |
| `queue.onSubmittedWorkDone()` -- wait for the GPU, no readback | **0.155 ms** |
| `mapAsync` readback, 16 bytes | **2.70 ms** |
| `mapAsync` readback, 64 KB | **3.04 ms** |
| the same through a *fresh* staging buffer each time | 2.84 / 3.00 ms |

Two things follow immediately.

**A readback costs the same whatever it reads.** 16 bytes and 64 KB are
within 12% of each other, and a persistent staging buffer is no cheaper than
allocating one per read. The cost is not the copy. Waiting for the GPU is
0.155 ms; the other ~2.5 ms is `mapAsync` itself -- the map completion coming
back through Chrome's GPU-process round trip. So "read fewer bytes" is not an
optimisation here, and neither is pooling staging buffers. Only *not waiting*
is.

**And waiting parallelises perfectly:**

| | total |
| --- | --- |
| 1 readback | 3.45 ms |
| 4 readbacks, awaited one after another | 11.35 ms |
| 4 readbacks, all in flight, `Promise.all` | **2.91 ms** |
| 8 readbacks, all in flight | **3.06 ms** |

Eight concurrent maps cost what one costs. The solver's ~11 ms of round trips
is therefore not "six reads" but *six serial waits*: each stop test has to be
answered before the host knows whether to encode the next batch. That is a
dependency-chain cost, not a bandwidth cost, and it is why raising
`residualCheckInterval` cannot fix it (re-swept today at 4/8/16/32/64 on
example 15: 4 and 8 both converge 100% at 20.0 / 19.2 ms, while 16 and 32
collapse to 47% / 39% converged as the iterate is driven past the point where
the recursive residual still tracks the true one).

## Against the floor: three.js's per-call cost

Wrapping `Renderer.compute` and its callees on example 15 (no `?profile=1`,
so no timestamp writes), per step:

| | ms/step | calls/step | us/call |
| --- | --- | --- | --- |
| `renderer.compute` **total** | **8.81** | 159 | 55.4 |
| ...`backend.finishCompute` (pass end + queue submit) | 2.70 | 159 | 17.0 |
| ...`bindings.updateForCompute` | 1.16 | 999 | 1.16 |
| ...`backend.compute` (the dispatch itself) | 1.01 | 999 | 1.01 |
| ...`nodes.updateForCompute` | 1.02 | 999 | 1.02 |
| ...`backend.beginCompute` | 0.83 | 159 | 5.2 |
| ...`pipelines.getForCompute` | 0.34 | 999 | 0.34 |
| ...`backend.updateTimeStampUID` | 0.14 | 159 | 0.88 |

The same 999 dispatches in 159 submissions cost **1.27 ms** through raw
WebGPU. So ~7.5 ms per step is three.js's node/binding/pipeline bookkeeping
above the API, split about evenly between a per-dispatch cost of ~3.5 us and
a per-`compute()`-call cost of ~33 us.

## The finding: 107 of 159 calls carried one dispatch

Counting the array length at every `compute()` call, per step:

| nodes per call | calls/step |
| --- | --- |
| **1** | **107** |
| 38 (the V-cycle) | 18 |
| 4 | 17 |
| 8 | 13 |
| 9 | 4 |

11% of the dispatches were taking ~70% of the per-call overhead. Sampling the
stacks of the single-dispatch calls named two sites, both fixed sequences with
no host decision inside them:

- `array_utils.js`'s `createExtrapolateToRegion2().run( n )` -- `n + 2`
  dispatches, one submission each;
- `grid_blocked_boundary_condition_solver2.js`'s `constrainVelocity()` --
  ~13 of its own plus the two extrapolations, and it runs three times a step
  (after forces, after pressure, after advection).

Both now build one `tsl_array_n.createBatch` plan. `run()` also exposes its
dispatcher list so `constrainVelocity` can splice it in rather than nest a
submission inside its own, which collapses the whole of `constrainVelocity`
to a single 27-dispatch pass. `grid_flip_solver2.js`'s resample pass (seven
dispatches) went the same way.

Calls per step, example 15: **159 -> 81**, with the dispatch count unchanged
at 999.

## Measured

Paired and phase-alternated inside one run, 15 rounds of 4 steps per arm. The
"unbatched" arm re-splits the merged pass into one submission per dispatch at
the `renderer.compute` boundary, so both arms run identical kernels in
identical order and differ only in submission granularity:

| scene | unbatched | batched | |
| --- | --- | --- | --- |
| 15 flow past cylinder (has a collider) | 24.51 ms | 19.95 ms | **1.229x** |
| 20 FLIP dam break (no collider; extrapolation only) | 18.71 ms | 17.79 ms | **1.052x** |

The spread between the two is the collider: without one, `constrainVelocity`
is only the closed-boundary and clamp kernels, and the extrapolation is all
there is to batch.

**Identical output, checked rather than assumed.** 150 steps from a fresh
load in each mode, hashing every cell of u, v and pressure: 857113263 /
-742805524 / -914744686 in both. WebGPU's ordering guarantee inside a pass
holds on this device for a 27-dispatch chain that reads what the dispatch
before it wrote. Example 28 over 300 steps: 300/300 converged, zero
rejections, peak pressure 15.614 -- its historical value to every digit.

## What is left

26 single-dispatch calls per step remain on example 15 and 18 on example 20,
scattered across the FLIP and advection paths rather than concentrated in one
sequence. At ~33 us each that is ~0.8 ms, worth having but no longer the
shape of a finding. The two big items are unchanged and both are now
quantified: ~7.5 ms/step of three.js bookkeeping above what the WebGPU calls
themselves cost, and ~11 ms/step of serial `mapAsync` waits that would cost
~2.3 ms if they could be issued together.

## Second pass: the remaining singles (2026-09-16)

The 26 single-dispatch calls left on example 15 were scattered rather than
concentrated, but every one of them sat in a fixed sequence with no host step
inside it:

| site | was | now |
| --- | --- | --- |
| `linalg.js` GPU-resident CG setup | 11 submissions | 2 (V-cycle splits it) |
| `advection_solver2.js` MacCormack face advection | 6 | 1 |
| `advection_solver2.js` scalar advection / order-1 forms | 3 / 2 | 1 |
| `grid_pressure_solver2.js` dirichlet refresh + build system | 2 | 1 |
| `grid_pressure_solver2.js` snapshot-or-restore + both corrections | 3 | 1 |
| `external_force_solver2.js` force pair | 2 | 1 |
| `grid_solver2.js` velocity clone | 2 | 1 |

Calls per step on example 15: **159 -> 81 -> 67**, single-dispatch calls
**107 -> 26 -> 5**, dispatches unchanged at 999 throughout.

Paired, phase-alternated, the unbatched arm re-splitting exactly the groups
this pass added: **1.043x** (19.15 -> 18.36 ms). Cumulative against the
pre-batching baseline on this scene: 24.51 -> 18.36 ms, **1.33x**.

Output identical again, and to the same hashes as before either pass: 150
steps from a fresh load, u/v/pressure hashing 857113263 / -742805524 /
-914744686 in both arms and in both rounds. Example 28 over 300 steps:
300/300 converged, zero rejections, every cell finite.

The five singles that remain are one-off kernels with host work on both
sides of them; there is no sequence left to merge.

# Proposal A, built and measured: encoding a batch without three.js (2026-09-16)

`packages/tsl_array_n/src/prepared_dispatch.js`. `createBatch` still runs its
first execution through `renderer.compute()`; afterwards it asks three.js for
what that call resolved -- the bind groups, the pipeline and the workgroup
counts three.js itself cached -- and encodes the pass directly with
`setPipeline` / `setBindGroup` / `dispatchWorkgroups`, one command buffer,
one submit.

Nothing of three.js's implementation is reproduced. In particular the
workgroup-count arithmetic is *not* reimplemented: it is read back from the
value three.js computed and cached on the first run. What is skipped is the
per-call encoder/pass/submission bookkeeping and the dispatch-time
indirection; what is kept, on every dispatch, is `nodes.updateForCompute` and
`bindings.updateForCompute`, because those upload changed uniforms and this
package re-exports three's `uniform()` rather than wrapping it -- a stale
`dt` would be a silent wrong answer.

It falls back to `renderer.compute()` on the first run of a batch, on a
non-WebGPU backend, on anything it cannot resolve, and -- deliberately --
whenever `trackTimestamp` is on, because three.js writes its timestamp
queries around its own passes and a pass encoded here would be invisible to
them. **Consequence worth knowing: `?profile=1` measures the slow path.**
Wall-clock comparisons have to be run without it.

## Measured

Paired, phase-alternated within one run, the switch being
`tsl_array_n.dispatchSettings.preparedDispatch`:

| scene | renderer calls/step | three.js path | prepared | |
| --- | --- | --- | --- | --- |
| 15 flow past cylinder | 67 -> 5 | 21.97 ms | 18.01 ms | **1.22x** |
| 20 FLIP dam break | 48 -> 5 | 15.45 / 14.38 / 12.14 | 14.27 / 13.51 / 11.14 | **1.08x** (3 reps) |
| 28 drop into pool | 109 -> 12 | 35.09 / 35.03 | 34.16 / 37.00 | **no effect** |

Two honest notes on that table. The first example-20 run, 12 rounds on a
still-settling scene, read 0.945x; three repetitions of 20 rounds each then
read 1.082, 1.064 and 1.090, so the first was noise and the later figure is
the one to believe. And example 28 shows nothing despite the fast path being
active on 1627 of its 1649 dispatches -- at 35-42 ms/step with ~28 iterations
its time is going somewhere else (seven serial stop-test reads is ~16 ms of
it), and the host bookkeeping this removes is not what it is waiting on. The
optimisation is worth what the scene was spending on dispatch overhead, which
is a lot on example 15 and not much on example 28.

## Correctness

- **Bit identity on example 15:** 150 steps from a fresh load with the fast
  path on and off, u/v/pressure hashing 857113263 / -742805524 / -914744686
  in both -- the same values as before any of today's changes.
- Example 20: 150 steps, 100% converged, particle positions all finite.
- Example 28: 200 steps, 200/200 converged, zero rejections, all finite.
- Example 23 (moving collider, kernels rebuilt every frame -- the case that
  invalidates a cached plan): 250 steps, 250/250 converged, zero rejections,
  and the scene renders correctly.
- The timestamp fallback was verified rather than assumed: with `?profile=1`
  the renderer sees all 67 calls again and GPU timestamps still resolve.

# Proposal B: the circuit breaker stops taking a round trip of its own (2026-09-16)

`countBadPressureCellsNow()` was one host wait for one integer -- `phases`
priced it at 0.80 ms/step. The CG solver is already waiting on a readback
every fourth iteration, and concurrent maps are nearly free (eight cost what
one costs), so the count now rides along with that read: `cg.setReadCompanion`
takes a dispatcher to run before the read and a read of its own, issues both
together with `Promise.all`, and leaves the value in `state.companionValue`.

The check itself did not move. Same kernel, same threshold, same
reject-and-restore decision in `grid_pressure_solver2.js` -- only the wait is
shared. `settings.badCellsRideAlong` switches it at runtime so the comparison
can be paired, and the old separate read is still the fallback for a solve
that ends before any read happens.

Why the value is the right one: whatever the companion measures describes the
x that the batch immediately before the read produced, and for the read that
ends the solve that is the x the solver is about to return -- which is the
property the guard needs. That is why the dispatch happens inside the solver
rather than after `solve()` returns.

## Measured

Paired, phase-alternated on `badCellsRideAlong`:

| scene | separate read | riding along | | |
| --- | --- | --- | --- | --- |
| 28 drop into pool | 35.53 / 33.20 / 29.02 ms | 31.11 / 31.74 / 26.08 ms | **1.10x** | +1.4 to +4.4 ms |
| 15 flow past cylinder | 20.72 / 22.52 / 21.60 / 22.13 | 19.31 / 21.18 / 22.24 / 21.55 | **1.045x** (median) | +0.67 ms mean, one of four reps negative |

Example 28 gains more because it waits more: ~28 iterations means seven stop
tests per solve, and its bad-cell kernel covers a bigger field. On example 15
the saving is real but close to that scene's noise -- four repetitions of 20
rounds, three positive and one negative, is what a ~0.7 ms effect looks like
on a ~21 ms step.

## The guard still guards, checked directly

The historical failure this check exists for is a NaN reaching velocity, so
the test injects one: write a NaN into the pressure field, step once, and
require that the solve is **rejected in that same frame**, the field comes
back finite, and the next frame is clean and not rejected. All three hold.
Five clean frames either side reject nothing, and example 28 runs 150 steps
with 150/150 converged, zero rejections and every cell finite.

# Proposal C: asking the stop test one batch before listening to it (2026-09-16)

The loop used to encode a batch, await its readback, decide, and only then
encode the next batch. The readback's ~3 ms is latency, not work -- the GPU
finished long before the map resolves -- so the loop now issues this check's
read *before* settling the previous one. Two maps are in flight at once,
which costs what one costs, and the batch of iterations encoded in between is
time the wait no longer charges for.

The waits also pipeline into each other, which is why the effect is bigger
than "hide one batch's encode time": if each check blocks for X, the time
between issuing a read and awaiting it is X plus the interval's encode, so X
settles at roughly `latency - X - encode` -- about a third of the original
wait rather than all of it.

**What it costs:** the stop decision arrives one interval late, so a
converged solve runs up to `residualCheckInterval` iterations it did not
need -- measured as ~40% more iterations. The read issued at the check that
decides to stop is awaited before breaking, so what the solver reports still
describes the x it leaves behind, and no promise is left dangling to resolve
into a later solve's state. The last two intervals before `maxIterations` are
always asked synchronously: a solve near the cap cannot afford the overshoot,
and would be reported as not converged for the sake of a hidden wait.

`settings.optimisticStopTest`, on by default.

## Measured

Paired, phase-alternated, `mapAsync` latency on this machine measured at
3.11 ms during the same session:

| scene | synchronous | pipelined | | iterations |
| --- | --- | --- | --- | --- |
| 15 flow past cylinder | 16.5-21.2 ms | 11.0-14.3 ms | **1.37-1.81x** (7 reps) | 21 -> 29 |
| 28 drop into pool | 29.7 / 29.1 / 26.6 | 19.3 / 19.2 / 17.5 | **1.52x** | 30 -> 38 |
| 20 FLIP dam break | 15.5 / 15.5 / 15.3 | 11.4 / 11.8 / 12.1 | **1.31x** | |
| 26 dye free surface | 17.0 / 19.2 | 15.1 / 10.1 | faster, unpaired | 14-18 -> 21-27 |

One run of three repetitions on example 15, taken earlier in the session,
read 1.097 / 0.904 / 0.985 -- a wash. It is recorded here because it did not
reproduce: two later harnesses, seven repetitions between them, all read
1.37x or better, and the baseline arm in the anomalous run was itself
unusually fast (14 ms against 19-21 ms later). Readback latency is a browser
and driver state, not a constant, and this optimisation is worth exactly what
that latency costs at the time.

## Convergence, which is the thing this could have broken

| scene | steps | converged | rejected | |
| --- | --- | --- | --- | --- |
| 28 drop into pool | 300 | 300/300 | 0 | all finite, 36.2 mean iterations |
| 20 FLIP dam break | 200 | 200/200 | 0 | positions finite |
| 23 moving collider | 200 | 200/200 | 0 | positions finite |
| 26 dye free surface | 100 + 100 | 100% both arms | 0 | |
| 15 flow past cylinder | 100 + 100 | 100% both arms | 0 | 16.9 -> 24.1 iterations |

## Where the step time stands

Example 15, same scene and same machine as the 24.51 ms that started today's
work: **11.12 ms/step**, about 2.2x, from four changes that between them
removed 100 submissions, ~5 ms of three.js bookkeeping, one round trip and
two thirds of what the rest of the round trips cost.

# Proposal D: the stop test moves onto the GPU (2026-09-16)

Two things had to be true before this was safe, and one of them turned out
to be a speedup in its own right.

## First: the true residual, every iteration

The GPU can only be trusted to decide convergence if the residual it tests
is the true `b - Ax`. Recomputing that costs one operator apply -- a handful
of dispatches against a V-cycle's 38 -- and the GPU is busy under 1% of a
step, so the question was whether it is affordable. It is better than
affordable. Paired, phase-alternated, recompute interval 50 against 1:

| scene | interval 50 | interval 1 | | iterations |
| --- | --- | --- | --- | --- |
| 15 flow past cylinder | 13.08 / 11.68 / 13.01 ms | 9.14 / 10.19 / 10.94 | **1.43 / 1.15 / 1.19x** | 25.0 -> 16.8 |
| 20 FLIP dam break | 11.37 / 12.39 / 13.05 | 9.12 / 9.00 / 9.21 | **1.25 / 1.38 / 1.42x** | 24.3 -> 16.2 |
| 28 drop into pool | 23.74 / 21.84 / 22.06 | 18.38 / 18.13 / 16.38 | **1.29 / 1.20 / 1.35x** | 37.5 -> 30.6 |

Every arm converged 100% of the time. It is faster *because* it is honest:
with the residual always true, `verifyConvergence` never spends a second
stop-test cycle confirming a claim, so solves stop sooner. The default is
now 1, and the residual-gap estimator parked in `project-history.md` is no
longer worth building -- there is no gap left to estimate.

## Then: the test itself, and the freeze

`convergenceKernel` evaluates the same criterion the host did --
`sqrt(|r.r|)` against `max(tol * |b|, tol)`, with `isNonFiniteOrAbove` so a
NaN residual cannot read as convergence -- and writes `STOP_CONVERGED` into
the sticky stop slot, along with the iteration it happened at.

What makes that worth anything is the other half: `alphaKernel` and
`betaKernel` now zero their outputs whenever the stop slot is set, not only
when this iteration tripped a guard. Every iteration after the stop is a
no-op, so the host can be arbitrarily late in noticing. That is exactly what
the interval sweep said was impossible before -- a solve driven 16 iterations
past its stop test dropped to 47% converged, because those iterations were
real and they degraded the iterate.

The host then runs a chunk sized from the previous solve (`lastIterationCount
+ 2`, floor 4) and looks once. The first margin tried was 8 and it was a
wash: ~2.4 ms of frozen iterations to save ~3 ms of reads, on example 15. A
frozen iteration is not free -- ~0.3 ms, mostly three.js's per-dispatch
uniform bookkeeping rather than GPU work -- which is the same overhead
proposal A left on the table.

The chunked path refuses to run unless the recompute interval is 1 and the
setup is GPU-resident. A drifting residual with no host check left to catch
it is this library's own worst bug with the safety net removed.

## Measured, against C rather than against nothing

`settings.gpuStopTest`, paired, with `optimisticStopTest` (proposal C) as the
baseline arm:

| scene | host-in-the-loop | GPU stop test | | iterations |
| --- | --- | --- | --- | --- |
| 20 FLIP dam break | 9.75 / 10.46 / 9.65 ms | 6.82 / 6.79 / 7.77 | **1.43 / 1.54 / 1.24x** | 16.2 -> 10.8 |
| 15 flow past cylinder | 9.89 / 9.95 / 9.52 | 7.31 / 8.16 / 9.19 | **1.35 / 1.22 / 1.04x** | 16.8 -> 11.7 |
| 28 drop into pool | 17.76 / 18.27 / 18.58 | 18.12 / 16.03 / 16.84 | **0.98 / 1.14 / 1.10x** | 30.6 -> 24.9 |

Iteration counts fall because the GPU tests every iteration, so a solve stops
the moment it crosses the threshold instead of at the next multiple of four
plus an interval's deferral.

## Does it stop at the right place?

The host no longer checks anything, so this is the question that matters.
Over 120 consecutive steps on example 15, computing `|b|` independently on
the host each frame and comparing the reported residual against
`max(tol * |b|, tol)`:

- **worst ratio of residual to threshold: 0.998**, and
- **zero solves reported converged with a residual above the threshold.**

It stops right at the criterion rather than past it, which is also why the
reported residual on example 28 is now ~4.4e-4 against ~1.6e-4 before: the
old path kept going for another few iterations after crossing. Both are
inside `tol * |b|`, which is the contract.

The NaN guard still fires in the same frame, checked by injection: the solve
is rejected with `degenerate-pAp`, the field comes back finite, the next
frame is clean.

## Long runs on the new defaults

| scene | steps | converged | rejected | |
| --- | --- | --- | --- | --- |
| 28 drop into pool | 400 | 400/400 | 0 | 17.95 ms/step, peak 16.766, finite |
| 20 FLIP dam break | 300 | 300/300 | 0 | 8.51 ms/step, positions finite |
| 23 moving collider | 250 | 250/250 | 0 | positions finite |
| 26 dye free surface | 250 | 250/250 | 0 | peak **10.382**, its historical value to every digit |

## Where the day ends

Example 15, same scene, same machine: **24.51 ms -> 9.54 ms per step, 2.6x**,
at 5 renderer calls and 12.2 iterations per step against 159 calls and 17
iterations this morning.
