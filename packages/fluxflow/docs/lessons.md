# What this project learned the hard way

A cross-cutting index of the transferable parts: what made it faster, what the
platform will not let you do, the mistakes that cost the most, and the mechanisms
that came out of them. Everything here is measured on this codebase on the
author's own machine, Chrome/WebGPU, an RTX-class desktop GPU.

The per-investigation narratives live elsewhere and are not repeated:
`project-history.md` for the debugging episodes and the research arc,
`perf-investigation-cg-gpu-resident-alpha-beta.md` for the full performance
record including every rejected direction, `long-run-stability.md` for the
long-run verdicts and the residual-floor thread, `optimisation-agent-guide.md`
for the action space and measurement protocol, `verification.md` for how a
solver is decided to work. This file is the shortest path to "what should I
already know".

---

## 1. The cost model, because every optimisation argument reduces to it

| quantity | cost |
| --- | --- |
| one GPU→CPU round trip (`mapAsync`), **any size** | **2.7–3.4 ms** |
| the same, 8 issued concurrently | 3.06 ms **total** |
| `queue.onSubmittedWorkDone()`, no readback | 0.155 ms |
| one dispatch encoded through three.js | ~3.5 µs |
| one `renderer.compute()` call, fixed cost | ~33 µs |
| one dispatch encoded through **raw WebGPU** | 0.56 µs |
| one raw `queue.submit` of an empty buffer | 4.44 µs |
| one CG iteration (marginal) | ~0.3 ms |
| one multigrid V-cycle | 0.155 ms |
| GPU compute, whole solver step | **0.10–0.22 ms** |

Four consequences that decide most arguments before any code is written:

**A readback costs the same whatever it reads, and concurrent readbacks are
nearly free.** "Read fewer bytes" and "pool the staging buffer" are both
worthless here, measured. Only *not waiting* helps — by not needing the answer,
or by overlapping the wait with something.

**The GPU is busy under 1% of a step.** So arithmetic is nearly free and
*structure* is everything: dispatch counts, submission counts, round trips. An
algorithm that does ten times the arithmetic to remove one host wait is a win.

**A round trip's measured cost includes draining everything queued behind it.**
"Where the wait is observed" and "what the work costs" are different questions.
Do not add them.

**Superseded claim, kept because it was believed for months:** "a dispatch is
nearly free; a round trip is 400× more expensive, so ideas that remove
dispatches are worth nothing." The ratio was right and the conclusion was wrong
— dispatch *overhead*, at ~1000 dispatches and ~160 submissions a step, was
several milliseconds.

---

## 2. What actually made it faster

Each of these is in the tree and measured; `perf-investigation-…md` has the
paired runs.

**Batch dispatches into one submission.** One `renderer.compute(array)` instead
of one call per kernel. The fixed cost is per *call* (~33 µs), not per dispatch
(~3.5 µs), so merging a 25-stage pass saves most of 25 × 33 µs. Safe because
WebGPU orders dispatches within a pass.

**Keep the scalars on the GPU.** CG's α and β are quotients of dot products. Done
on the host, each iteration costs round trips; computed by a kernel into a small
scalar buffer, the iteration never leaves the device. This is what made the
chunked loop below possible.

**Run a chunk of iterations and look once.** With the stop test on the GPU and
the iterate frozen the moment it fires, the host does not have to be present for
the decision. It runs a chunk sized from the last solve, looks once, and either
finds the solve finished — and at which iteration — or runs another chunk. Trades
cheap GPU work for expensive host waits.

**Encode dispatches directly when the framework's per-call cost dominates.** The
gap between 0.56 µs and 3.5 µs, and between 4.44 µs and 33 µs, was worth 1.22×
on its own.

**Derive the iteration budget from the grid rather than setting it per scene.**
Borrowed from mantaflow: `1.5 × max(resolution) × (3D ? 1 : 4)`. Six scenes'
hand-set caps had come loose from their grids — one had a cap of 40 against a
measured worst of exactly 40, sitting on its own ceiling.

**Fix correctness to get speed.** `examples/24-two-phase-bubble-rise/` ran at
8 fps against 30–59 for the rest of the suite. The cause was a tolerance beneath
what the solve could reach, so every frame spent its entire iteration budget. The
tolerance fix took it to 33 fps. **In this suite, the slowest scene was the broken
one** — worth trying as a first-pass diagnostic before profiling anything.

---

## 3. What was rejected, by measurement

The reasoning recurs, so the rejections are worth as much as the landings.

| idea | outcome |
| --- | --- |
| Cheaper preconditioners (Jacobi, none) | ~20× the iterations, ~12× the wall time; on a liquid scene neither converges at all. Jacobi came out *worse* than no preconditioner, twice. Kept as the instrument, not as an option |
| Mass-weighted P2G | built, measured, rejected |
| CPU coarse solve, one per V-cycle | 46 → 87 ms per frame; ~3.9 ms per mid-cycle stall |
| Direct coarse-level solve, serialised | slower *and* a worse preconditioner (12.5 → 18.6 iterations) |
| Per-frame diagonal precompute | 72 → 94 ms, 30% slower. A 35% ceiling measured beforehand was real as a ceiling and worthless as a prediction |
| Coarse inverse uploaded once per frame | priced from data in hand; never built |
| Predictive stop-test scheduling | 21% cheaper per iteration and 26–32% more iterations. They cancel exactly: 2% faster in one paired run, 5% slower in the next |
| Reading fewer bytes / pooling staging buffers | a round trip costs the same at 16 B and 64 KB |
| `double_single` for the residual | the accuracy of `b - A@x` was never the constraint — measured at 3.57e-8 where the floor is 5.84e-6 |

---

## 4. WebGPU and WGSL limits that cost real time

**There is no f64, and no extension for one.** An open proposal
(gpuweb/gpuweb#2805), not a feature. A kernel needing more than 24 bits of
mantissa has to build it — `src/linalg/double_single.js` does, as a pair of f32s
giving ~48 bits on Dekker's and Knuth's error-free transformations.

**`x != x` does not detect NaN.** Core WGSL dropped `isnan()`, every reference
recommends self-inequality in its place, and WGSL *also* permits an
implementation to assume non-finite values never occur and fold accordingly. On
the device this library was developed against, it folds to a constant `false`.
**Every NaN guard in the library was silently inert** — which cost a liquid
solver that ran correctly for a couple of hundred frames, collapsed to a point in
one step, and reported healthy throughout. `src/float_guards.js` exists entirely
for this, and `examples/27-float-guard-probe/` re-measures it per device because
the answer is per device: on today's hardware the bare idiom happens to work,
which does not make it safe.

**Default device limits are downlevel, not what the adapter offers.**
`requestDevice()` with no `requiredLimits` silently gets the defaults. On this
machine the adapter reported `maxStorageBuffersPerShaderStage = 16` and
`maxComputeInvocationsPerWorkgroup = 1024` while the device three.js built had
**8** and **256**. A kernel binding nine storage buffers failed at pipeline
creation — and WebGPU reports that as an *uncaptured* error on the first dispatch
using it, so **the pass was silently never executed with nothing visible from
JavaScript**. Two examples were dropping a compute pass and a third sat exactly
on the limit. Fixed by requesting the adapter's own limits in `init()`.

**Eight storage buffers per stage is the guaranteed limit, and good hardware
hides that.** A kernel binding twelve failed outright with pipeline creation
rejected and the kernel doing nothing every frame. Plenty of real hardware raises
the limit, which is exactly what makes it a portability bug a good GPU conceals.
Count bindings deliberately; this codebase keeps its hottest kernel at seven and
says so in a comment.

**WGSL atomics exist only for `atomic<i32>`.** No float atomics. A float
reduction through atomics therefore needs either fixed-point encoding — and
integer addition is exactly order-independent, which turns out to matter more
than the precision it costs — or a lane-partitioned float reduction.

**`select()` over an `atomicLoad()` result generates invalid code.**

```js
pick.select( atomicLoad( a( 0 ) ), atomicLoad( a( 1 ) ) )              // 2 errors
pick.select( atomicLoad( a( 0 ) ).toInt(), atomicLoad( a( 1 ) ).toInt() ) // clean
```

An atomic load's result does not carry a plain `int` type through `select`, and
everything derived from the bad select inherits it — two such selects produced
four errors. Narrow: an atomic result in a *comparison* is fine, and an
`atomicAdd` result as an index into a *non-atomic* array is fine.

**A WGSL compile error rejects the whole command buffer, silently.** Comparing a
loop index (`u32`) against a JS number produced "no matching overload for
operator >= (u32, abstract-float)", the entire submission was rejected, and the
*solver* then reported `degenerate-pAp` at iteration 1 — which reads exactly like
a preconditioner breakdown and is not one. If a guard starts firing immediately
after a kernel edit, suspect the kernel compiled, not the mathematics.

**Dispatches in one pass are ordered, but anything reading a buffer another
thread writes *within* one dispatch is a race.** See §5.

---

## 5. The mistakes, as shapes rather than incidents

These are the ones that generalise. Each cost days.

**A test that cannot see the failure.** The structural suite builds node graphs
without a device, so a binding-count limit, a folded NaN guard and a wrong
reduction are all invisible to it. Every bug in `project-history.md`'s debugging
section passed a green suite. A green CLI run means "nothing is obviously
malformed", never "it works".

**An input that cannot distinguish the answer.** A dot product over a field of
*ones* returns the right total however wrongly its cells are chosen — any 576
ones sum to 576. The 3D reducer summed a diagonal twenty-four times over and the
obvious test could not see it; single-cell deltas could. Pick the input that
distinguishes the failure, not the one whose answer is easy to predict.

**Uniqueness is not reproducibility.** `atomicAdd(cursor, 1)` hands every
claimant a distinct slot — verified on hardware, no collisions at N=256 and
N=4000 — and *which* claimant gets slot 0 depends on arrival order. The
resamplers moved a different set of particles every run. A GPU primitive can be
correct and unrepeatable at once, and only a test that runs it twice can tell.

**Reading and writing one buffer in one dispatch.** The velocity extrapolation
averaged a cell's neighbours out of `output` and wrote back into `output`, in the
same kernel, with the *valid mask* beside it carefully double-buffered. Threads
in different workgroups have no ordering, so whether a cell saw a neighbour's old
or new value depended on scheduling. Every caller passed the same array as input
and output, so nearly every scene was affected.

**Seeding from `Math.random()`.** Every FLIP scene started from a different
initial condition, so no long run was repeatable and no regression bisectable.
Found only after two wrong mechanisms had been chased. Jitter is wanted; a
*reproducible* jitter is what was missing.

**A counter that cannot be a criterion, read as one — or ignored entirely.**
Convergence counts: a healthy channel converges on 28 frames of 900; a broken
scene converged on 73%; three scenes converged on **0 of 12,001** while perfectly
healthy by conservation. And a scene converging on **2 of 1,001** was passed for
weeks because the verdict does not judge convergence. Both failure directions are
real: judging the counter is wrong, and printing it without explaining it is also
wrong.

**A fix that creates the next problem.** Recomputing the true residual every
iteration removed a drift bug that had the solver reporting convergence it had
not achieved — and put a floor under every tolerance, because substituting the
true residual into CG's recurrence breaks the conjugacy it depends on. Both the
fix and its cost are real; the error was recording only the first.

**A comment that prescribes what the code does not do.**
`examples/25-dye-injection/` carried a paragraph explaining that its tolerance is
loosened to 1e-4 and why. The `tolerance: 1e-4` was never in the options. It
inherited the default, the failure the loosening was meant to fix came back, and
the comment did not change. Audit comment-claimed values against actual ones.

**Harnesses that report a partial or absent run as a pass.** Three times. A
14,643-frame stability claim read from a log's tail, where the blow-up in the
first 600 frames had scrolled past. A determinism sweep that diffed two runs
which had both failed to start — identical failures diff clean. And a page driver
that waited a fixed fifteen seconds for a page taking thirty-four, truncating a
symmetry sweep at three of sixteen rows, and a written conclusion reasoned from
the truncation. **A check that cannot tell "finished and fine" from "did not
finish" is not a check.**

**A formula that matches a number without a reason.** `eps * |A| * |x| / |b|` was
adopted as the residual floor because it landed within a factor of two of the
measurement. Measured directly, the quantity it names is ninety times smaller.
It fit, and it was wrong.

**Trusting a header over the history of the file it heads.** `double_single.js`'s
header argues from a per-cell magnitude its own introducing commit had already
retracted. Re-adopting the argument meant repeating a mistake that was documented
two lines of `git log` away.

---

## 6. The mechanisms that came out of it

What is in the tree as a result, and worth copying.

**Judge conservation, not appearance.** `solver_health.mjs` gives one verdict
from mass balance and the residual the projection leaves, exits 0 or 2, and prints
counters beside it as evidence rather than criteria. It replaced a detector
watching maximum velocity against a fixed 90 — which cannot decide from either
side, since the circuit breaker clamps at 100.

**Every criterion states when it does not apply.** A free surface's solved region
moves, so flux balance is not zero there; a vent-only domain is not forced to
balance by this formulation or by mantaflow's. Those criteria go out of scope and
say so, rather than passing quietly. Five false positives were found by pursuing a
verdict instead of accepting it.

**Thresholds calibrated on both sides.** Not "1e-2 feels right" but "healthy
scenes run a median of 4.5e-7 to 4.7e-6 and the same scenes with the defect put
back run 5.0e-3 and 5.9." The same discipline relaxed a criterion later: the
breakdown rule became rate-based only with a table showing the highest rate that
must pass (0.05%) and the lowest that must fail (1.46%).

**Order-independence by construction where it is cheap.** The P2G scatter
accumulates in *fixed-point integers* precisely so that atomic order cannot change
the answer — integer addition is associative and exact. Where that was not done,
ranks come from a prefix sum instead of a cursor: a donor's slot is the number of
donors before it in index order.

**Verify a primitive on hardware, not from the spec.** `float_guards.js` exists
because the documented NaN idiom is a no-op, so `double_single.js`'s error-free
transformations are checked by the same page rather than trusted to WGSL's
promise that it reassociates nothing.

**Offline fixtures.** `export_system.mjs` captures the pressure system from a
chosen frame and a sandbox page rebuilds it, so a hypothesis costs seconds rather
than minutes. Two frames are captured on purpose — one the solver handled and one
it broke on — because a fixture that only reproduces the failure has not shown it
reproduces anything.

**Take the problem off the GPU when the GPU is the suspect.**
`cpu_reference.mjs` solves an exported system in Node, in double and in three
flavours of single precision, and proves its own operator first — symmetry, and
`‖b - A@pressure‖` against the number the GPU reported, which agreed to a ratio
of 1.000. That is what separated "the arithmetic cannot do this" from "the
algorithm is doing it wrong", and the answer was the second.

**Report what the arithmetic can and cannot deliver.** `settings.reportNoiseFloor`
turns `converged: false` from ambiguous into explained — "the budget was too
small" and "no iteration count reaches this" call for opposite responses, and
telling them apart by hand took eliminating four candidates. (Its formula names
the wrong quantity and is pending replacement by the measured discrepancy between
the recurrence's residual and the recomputed one.)

---

## 7. The shortest version

- Measure before building; price the idea against §1 first.
- The GPU is idle. Spend arithmetic to buy structure.
- A green test suite means almost nothing about a solver.
- Pick inputs that can distinguish the failure.
- Run it twice before believing it once.
- A criterion nobody has seen fail is not a criterion.
- When a harness and the code disagree, suspect the harness.
- Retract in place, and read the history of the file before trusting its header.
