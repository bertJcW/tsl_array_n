# fluxflow as an optimisation target: a guide for an agent

Written for an agent whose job is to **make this library faster without
breaking it**. It describes what the system does, what can be changed,
what can be measured, what must not regress, and — at length, because it
is where most of the effort in this project has actually gone — **how to
measure without fooling yourself**.

Every number here was measured on this codebase. Where a figure was later
found to be wrong it is marked, because the wrong ones are instructive.

---

## 1. The system in one page

A GPU fluid simulator. WebGPU, reached through three.js's TSL via a
sibling package (`tsl_array_n`) that wraps it as a Taichi-style
compute-kernel API. FLIP/PIC particles on a staggered MAC grid; the
expensive part is a pressure projection solved by
multigrid-preconditioned conjugate gradient (MGPCG).

One **solver step** is:

```
advect → external forces → boundary conditions → PRESSURE PROJECTION → transfer
```

Measured on `examples/15-flow-past-cylinder/`, a 64×64 grid:

| component | ms | share |
| --- | --- | --- |
| everything except the pressure projection | 1.4 | **6%** |
| pressure projection | 21.0 | **94%** |
| — of which: CG iterations (~12 × 0.861 ms) | 10.3 | 46% |
| — of which: fixed cost of one solve | ~9.1 | 40% |

**The first thing to internalise: the non-pressure half of the solver is
6% of it.** Optimising advection, forces or transfer cannot matter much.

**The second: the GPU is idle ~99% of the time.** Measured GPU compute is
0.21–0.40 ms per step. The grid is 4096 cells; the arithmetic is
microseconds. This is not a compute-bound program and never has been.

**The third: the time goes into waiting for the host.** Host round trips
were 79% of a step before the most recent work. That is the axis along
which every real win in this project has been found.

---

## 2. Cost model (measured constants)

Use these to price an idea before building it. All measured on this
codebase, RTX-class desktop GPU, Chrome/WebGPU.

| quantity | cost | how it was measured |
| --- | --- | --- |
| one GPU→CPU round trip (`toArray()` mid-solve) | **~1.1–1.2 ms** | ablating the circuit-breaker read; phase timing |
| one `renderer.compute()` submission | **38.79 µs** | fixed workload, varying only submission count, least squares |
| one CG iteration (marginal) | **0.861 ms** | iteration-cap sweep, R² 0.996 |
| one multigrid V-cycle | **0.155 ms** | `multigrid` vs `none` preconditioner at a pinned 40 iterations |
| one dispatch, CPU encoding | **~2.8 µs** | encode total ÷ dispatch count |
| GPU compute, whole step | **0.21–0.40 ms** | WebGPU timestamp queries |
| bare readback, idle queue, 4096 floats | 0.307 ms | direct |

Two consequences worth stating explicitly:

- **A dispatch is nearly free; a round trip is ~400× more expensive.**
  Ideas that remove dispatches are usually worth nothing. Ideas that
  remove round trips are usually worth a lot.
- **A round trip's measured cost includes draining everything queued
  behind it.** So "where the wait is observed" and "what the work costs"
  are different questions. Do not add them.

---

## 3. Action space

### Runtime-switchable — `solver.pressureSolver.settings`

These can be changed between steps with no rebuild, which is what makes
paired measurement possible. **Use them; do not compare across separate
runs.**

| setting | values | default | measured effect |
| --- | --- | --- | --- |
| `residualCheckInterval` | integer ≥ 1 | **4** | 1 → 4 is 1.16–1.52× faster. 8 pushes iterations 13 → 17 and gives it back. |
| `gpuResidentScalars` | bool | **true** | true is 1.13–1.24× against computing alpha/beta on the host |
| `gpuResidentSetup` | bool | **true** | true is 1.14–1.30×; removes the two setup round trips |
| `batchIterations` | bool | **true** | 15 → 5 submissions per iteration; 1.15–1.56× per solver step |
| `preconditioner` | `'multigrid'` \| `'jacobi'` \| `'none'` | **multigrid** | multigrid needs ~20× fewer iterations and ~12× less time. The others are instruments, not options. |
| `maxIterations` | integer ≥ 0 | 100 | measurement instrument: capping prices an iteration. 0 runs the setup only. |
| `tolerance` | float ≥ 0 | 1e-5 | 0 makes a solve never converge, pinning both arms of a comparison to the same iteration count |
| `checkBadCells` | bool | **true** | the circuit breaker's readback, 1.1 ms. **Safety, not a setting** — see §5 |
| `settings.multigrid.batchDispatches` | bool | **true** | 1.53× and 1.74× |
| `settings.multigrid.coarseSingleGroup` | bool | **true** | 1.09–1.10× |
| `settings.multigrid.foldClearIntoRestrict` | bool | false | bit-identical, effect within noise |

### Construction-time — `createGridPressureSolver2({ multigrid: { … } })`

Cannot be changed per step, so comparing them needs two solver instances
or two page loads — which the protocol in §6 says not to do. Build both
forms at construction and select per call if you need to compare them.

`numberOfLevels` (4), `numberOfSmoothingIterationsDown` / `…Up` (2),
`numberOfCoarsestIterations` (20), `numberOfFinalIterations` (2),
`sorFactor` (1.0).

### Structural changes

Anything touching level state, buffer layout or the kernel graph. There is
a worked example of how to explore one without destabilising the library:
`sandbox/jacobi-smoother/` is a copy of `src/linalg/multigrid.js` with one
change, compared against the real one by
`examples/30-jacobi-smoother-sandbox/` through the same operator and the
same CG solver. **Copy, measure, and only then decide whether `src/` pays
for it.**

---

## 4. Observation space

### Per-solve diagnostics — `solver.pressureSolver.diagnostics`

| field | meaning |
| --- | --- |
| `converged` | true residual reached `tolerance` |
| `iterations` | CG iterations the solve ran |
| `stoppedBy` | `'none'`, or a guard name: `degenerate-pAp`, `pAp-growth`, `alpha-magnitude`, `degenerate-oldRZ` |
| `rejected` | the circuit breaker reverted this solve's pressure |

### Profiling — `src/profiling.js`

`startProfiling()` / `profilingReport(frames)` give, per frame:
`dispatchesPerFrame`, `submissionsPerFrame`, `cpuMsPerFrame` (encoding),
and two breakdowns — `labels` (per-kernel dispatch counts and encode time)
and **`phases`** (wall time inside named awaits, which is where the round
trips show up). `readComputeTimestampMs(renderer)` gives GPU busy time.

Two traps in the instrumentation itself, both of which produced wrong
conclusions before being found:

- The profiler only wraps kernels built through `buildElementwiseKernel`.
  About 15% are built with `tsl_array_n.kernel` directly. A recorded
  figure of "502 dispatches per frame" was that undercount; the true
  number was ~800.
- three.js's compute timestamp pool is 1024 passes and keys allocation off
  the frame counter, so it **saturates after ~4 frames unless resolved
  every frame**. "Timestamps are unavailable on this adapter" was wrong;
  they were never being read.

### Driving a scene — `window.__fluxflowProbe`

Nine examples expose `pause()`, `resume()`, `step()`, the solver and the
renderer; some also expose `draw()`. `pause()` returns a promise that
resolves after any in-flight step.

**A backgrounded browser tab throttles `requestAnimationFrame` to roughly
one frame every several seconds.** Any run longer than a few seconds must
be driven through `step()`, not by the page's own loop.

---

## 5. Constraints — what "not breaking it" means

An optimisation that violates any of these is a regression regardless of
its speed. This project has twice shipped a liquid that collapsed, so
these are not hypothetical.

| signal | requirement | how |
| --- | --- | --- |
| convergence | converged fraction must not fall | count `diagnostics.converged` |
| rejections | **0** | count `diagnostics.rejected` |
| non-finite | **0** in pressure, velocity and particle positions | scan `toArray()` output |
| post-projection divergence | comparable to baseline | `∂u/∂x + ∂v/∂y` over fluid cells |
| volume | occupied-cell count flat over thousands of frames | rasterise particle positions |
| peak pressure | comparable to baseline | max abs of the pressure field |
| guard behaviour | same `stoppedBy` on a singular and a near-null operator | `examples/05-…/` runs this |

**Why non-convergence is fatal rather than merely inaccurate.** Residual
divergence is a per-frame compressibility error; it accumulates as volume
change. Measured: at ~50% of frames converging, occupied cells collapsed
from 1536 to 340–700 and the liquid died. At 100%, flat at 1536 for 2000
frames. Smoke tolerates this; a free-surface liquid does not.

**The circuit breaker is not a performance setting.** `checkBadCells`
turns off the guard that stops a NaN solve reaching velocity. Two separate
shipped bugs were exactly that guard silently not working. It costs 1.1 ms
and it stays on.

---

## 6. Measurement protocol

**This section is the most valuable part of the document.** Most of the
wrong conclusions in this project's history came from measurement
mistakes, not from bad ideas, and every one of them is easy to repeat.

### The rules

1. **Compare within one run, never across runs.** Build both arms at
   construction, alternate them step by step, restore the same state
   before each arm. The scene evolves and the machine's clocks drift; both
   will masquerade as your change.
2. **Warm up for hundreds of frames.** This scene ramps from 100 ms/frame
   to ~40 over several hundred frames. A harness that warms 30 frames and
   times 60 measures the ramp. *This invalidated an entire round of work.*
3. **Use medians, and run the pair twice with the order swapped.** If the
   sign flips between runs, it is noise. Identical work has shown an 18–26%
   spread.
4. **A frame is not a solver step.** In some drivers a rendered frame
   carries a fraction of a step (example 28: ~27 submissions per frame
   against 359 per step) and frame time sits on the 60 Hz floor either way.
   Measure the step.
5. **Check convergence before believing a speedup.** A solver that stops
   solving looks exactly like a breakthrough. Two examples below.
6. **Beware normalised metrics.** ms-per-iteration is
   `fixed/iterations + per-iteration`, so *any* change that raises the
   iteration count lowers it whether or not iterations got cheaper. Use it
   only when the iteration counts match.
7. **A default is not in force where a caller names the option.** The
   benchmark scene passed `residualCheckInterval: 1` explicitly and
   silently opted itself out of a changed default.

### The two canonical false positives

**A V-cycle repetition probe reported 4.5× and CG iterations collapsing
from 12.5 to 1.** It had broken PCG: 0 of 40 frames converged and
post-projection divergence was 5× the baseline.

**A stop-test schedule reported 1.45×.** It was measured on the warm-up
ramp; measured properly the ordering reversed and the "faster"
configuration was slower.

### A harness that works

```js
const P = window.__fluxflowProbe;
const ps = P.solver.pressureSolver;          // or P.flip.pressureSolver
await P.pause();
for ( let k = 0; k < 300; k ++ ) await P.step();   // warm up

const u0 = await P.velocityGrid.dataU.toArray();   // snapshot
const v0 = await P.velocityGrid.dataV.toArray();
const p0 = await ps.pressure.data.toArray();
const restore = () => { P.velocityGrid.dataU.fromArray( u0 );
                        P.velocityGrid.dataV.fromArray( v0 );
                        ps.pressure.data.fromArray( p0 ); };

const arms = { a: [], b: [] };
for ( let r = 0; r < 20; r ++ ) {
  for ( const name of [ 'a', 'b' ] ) {
    restore();
    ps.settings.someKnob = ( name === 'a' );
    const t0 = performance.now();
    await P.step();
    await ps.pressure.data.toArray();           // force completion
    arms[ name ].push( performance.now() - t0 );
  }
}
// compare medians, then run again with the arm order swapped
```

---

## 7. What has already been tried

Do not re-derive these. Numbers are paired measurements unless noted.

### Landed

| change | effect |
| --- | --- |
| V-cycle dispatch batching | 1.53×, 1.74× |
| single-workgroup coarse solve (barriers) | 1.10×, 1.09× |
| GPU-resident alpha/beta | 1.24×, 1.13× |
| CG iteration submission batching | 1.15–1.56× per step |
| `residualCheckInterval` 1 → 4 | 1.16–1.52× |
| GPU-resident solve setup | 1.14×, 1.30× |
| collider kernels not rebuilt when it only moves | **11.8×** on example 23 (2.5 → 29.5 fps) |
| device limits from the adapter, not WebGPU defaults | correctness: example 16 was not running at all |

The first three, measured together off-against-on: **2.63× and 3.10×**.

### Rejected, with the reason

| idea | why it failed |
| --- | --- |
| Jacobi preconditioner | 342.8 iterations vs 9.8; worse than *no* preconditioner |
| no preconditioner | 196.9 iterations vs 9.8 |
| damped-Jacobi smoother in the V-cycle | best-ω iteration ratios 1.20/0.90/1.05 against a ~4% ceiling; ω=1 does not converge on variable density |
| fold `mg-clear` into `restrict` | bit-identical, effect within noise (V-cycle is only 0.155 ms) |
| exact-transpose transfer operators | premise wrong: R = Pᵀ/4 exactly, a constant, symmetry intact |
| predictive stop-test scheduling | buys 21%/iteration, pays 26–32% more iterations; also measured on an inert path |
| CPU coarse solve per V-cycle | 46 → 87 ms/frame |
| direct coarse-level solve | slower *and* a worse preconditioner (12.5 → 18.6 iterations) |
| per-frame diagonal precompute | 30% slower against a measured 35% ceiling |
| FLIP stage submission batching | ~6% of submissions, time-neutral |
| mass-weighted P2G | measured, rejected |

**The pattern: everything that attacked dispatches or arithmetic failed.
Everything that attacked host round trips worked.**

---

## 8. Open items

- **The remaining in-loop readback.** One round trip per checked
  iteration. `residualCheckInterval` reduces how often; removing it
  entirely means the host never learns whether to stop.
- **The circuit breaker's read**, 1.1 ms per solve. Making it periodic is
  a change in safety posture, not a free win.
- **`sandbox/jacobi-smoother/`** exists as a worked pattern for structural
  experiments, and is not wired into the library.
- **mantaflow provenance** is verified at two points only — see
  `provenance-audit.md`.

## Where the detail is

- `docs/perf-investigation-cg-gpu-resident-alpha-beta.md` — every
  measurement, including the retracted rounds, in chronological order.
- `docs/project-history.md` — research, the ten debugging episodes,
  testing layers, and the methodology lessons.
- `docs/provenance-audit.md` — licence and copying audit.
- `docs/realtime-fluid-tools-research.md` — how TouchDesigner and LiquiGen
  reach real time, and why it is algorithmic rather than API-level.
