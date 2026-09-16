# Long-run stability: 12,000 frames per scene

Every drivable example run for **12,000 solver steps** on real WebGPU
hardware, 2026-09-15, on the current defaults (`preconditioner:
'multigrid'`, `residualCheckInterval: 4`, `gpuResidentScalars` and
`gpuResidentSetup` on). **108,000 steps in total.**

The question is not speed. It is whether these solvers converge and stay
converged, and whether anything drifts, blows up, or quietly loses fluid
over a run far longer than any demo.

---

> ## ⚠ Read this before the table
>
> **The `converged` column below is not valid**, and the run has not been
> repeated since. Two separate reasons, both documented in full further
> down and in `perf-investigation-cg-gpu-resident-alpha-beta.md`:
>
> 1. **It was measured before the convergence bug was fixed** (the same
>    day, a few commits later). The solver was reporting convergence
>    against a residual that had drifted optimistically from the true
>    `b - Ax` -- see "The real cause" below. Every convergence figure in
>    that table is a claim the solver was not entitled to make.
> 2. **The defaults it names no longer exist.** It ran with
>    `residualRecomputeInterval: 50` and a host-side stop test. The
>    current solver recomputes the true residual every iteration, runs its
>    stop test on the GPU, and freezes the iterate when it fires
>    (2026-09-16).
>
> What the table *is* still evidence for, because none of it depends on
> the stop test: non-finite counts, rejection counts, peak pressures and
> occupied-cell drift over 12,000 steps. Those columns held.
>
> Shorter runs on the current defaults (250-400 steps on examples 15, 20,
> 23, 26 and 28) report 100% convergence with zero rejections, and
> example 26's peak pressure is unchanged at 10.382. **A full 12,000-step
> re-run is the outstanding item.**

---

## Results

| example | converged | rejected | mean iters | max iters | non-finite | peak pressure, first → last | occupied cells, first → last |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 15 flow past cylinder | **12000/12000** | 0 | 16.81 | 21 | 0 | 121.438 → 0.672 | — |
| 16 Kármán vortex street | 11999/12000 | 0 | 20.28 | 40 | 0 | 505.581 → 0.622 | — |
| 20 FLIP dam break | **12000/12000** | 0 | 1.68 | 21 | 0 | 20.765 → 20.765 | 988 → 960 |
| 21 irregular container | **12000/12000** | 0 | 6.38 | 9 | 0 | 20.765 → 20.765 | 600 → 452 |
| 22 multiple colliders | **12000/12000** | 0 | 11.06 | 17 | 0 | 20.765 → 20.765 | 630 → 478 |
| 23 moving collider | **12000/12000** | 0 | 1.94 | 17 | 0 | 10.301 → 10.301 | 765 → 640 |
| 26 dye in free surface | **12000/12000** | 0 | 1.61 | 21 | 0 | 10.382 → 10.382 | 1555 → 1536 |
| 28 drop into pool | **11256/12000** | 0 | 32.47 | **100 (cap)** | 0 | 18.336 → 15.614 | 4724 → 4480 |
| 29 static droplet | **12000/12000** | 0 | 1.21 | 17 | 0 | 0.002 → 0.002 | 3712 → 3712 |

Across all 108,000 steps:

- **Zero rejections.** The pressure circuit breaker never reverted a solve.
- **Zero non-finite values**, in pressure fields or particle positions, at
  every sample point.
- **Every stop reason was `none`.** Not one of the four CG guards
  (`degenerate-pAp`, `pAp-growth`, `alpha-magnitude`, `degenerate-oldRZ`)
  fired in 108,000 solves.

## Convergence

**Seven of nine converged on every single step.**

**Example 16 missed one step of 12,000.** Its peak pressure at step 1 is
505.6 against 0.62 for the rest of the run, so the miss is the startup
transient — the first solve of an impulsively started flow — not a
recurring failure.

**Example 28 is the real exception: 93.8%, with 744 steps hitting the
100-iteration cap.** It is the hardest scene in the set by a wide margin
(64×96, a 1.25 density ratio, a deep pool, mean 32.5 iterations against
1.2–20.3 everywhere else). Nothing downstream broke — no rejections, no
non-finite values, peak pressure settled and flat from step 4000 — so the
unconverged steps are leaving a small compressibility error rather than
destabilising anything. It is still the scene to watch, and the obvious
first response is to raise its `maxIterations` above 100 and re-measure
rather than to accept 6.2% unconverged.

## Robustness

**Peak pressure is the clearest signal, and it is flat.** Five scenes
report the *same value to three decimal places* at step 1 and step 12,000
— 20.765, 20.765, 20.765, 10.301, 10.382. A field that drifts, ratchets or
diverges cannot do that. The two grid scenes (15, 16) start with a large
transient (121 and 506) that decays within the first 3000 steps and then
holds to within 8% for the remaining 9000.

**Particle counts are exactly constant** wherever recorded: 3952 (ex 20),
3072 (ex 23), 14848 (ex 29), unchanged at every sample.

**Occupied-cell counts fall and then hold.** 988 → 960, 600 → 452,
630 → 478, 765 → 640, 1555 → 1536, 4724 → 4480. Read this carefully:
occupied cells count *distinct cells containing a particle*, so a liquid
that settles and compacts legitimately occupies fewer of them while
conserving every particle. The signature that matters is the shape of the
curve, and in every case it drops during the settling phase and is then
flat — example 22 reads 481 / 479 / 484 / 478 across steps 3000 to 12,000,
example 21 reads 463 / 461 / 452 / 452.

For contrast, the collapse this project has actually suffered looked
nothing like that: occupied cells fell from 1536 to 340–700 and kept
falling, with roughly half of frames failing to converge. See
`project-history.md`, Debugging #5.

**Example 29 does not move at all** — 14,848 particles, 3712 occupied
cells and a peak pressure of 0.002, identical at every sample across
12,000 steps. For a static droplet held by surface tension against nothing
that is the correct answer, and it is a strong null test: a solver with a
slow leak somewhere would not sit that still for that long.

---

## Coverage, and what was not run

Nine of 31 examples expose the `window.__fluxflowProbe` driver hook and
were run. The other 22 fall into two groups:

- **Self-checking pages with no frames to run**: 04 and 05 (CG and PCG
  against exact solutions), 07 (multigrid PCG against a reference, plus
  the restriction/prolongation adjoint check), 27 (the float-guard device
  probe), 30 (the Jacobi-smoother sandbox comparison). These verify
  correctness on load and have no long-run behaviour.
- **Demo scenes without a driver hook**, which cannot be stepped faster
  than `requestAnimationFrame` allows and were not run. Adding the probe
  to them is the way to extend this coverage; it is about five lines per
  example.

The nine that were run are the solvers: both grid scenes, all four FLIP
collider scenes, both free-surface/two-phase scenes, and the surface
tension scene.

## Method

Driven through `probe.pause()` then `probe.step()` in a promise loop —
never `requestAnimationFrame`, which a backgrounded tab throttles to
roughly one frame every several seconds. Counters accumulate every step;
fields are read back and scanned at step 1 and every 1000–3000 steps
thereafter, since reading 12,000 times would dominate the run.

Three scenes were run concurrently in separate tabs. That contends for the
GPU and makes the wall-clock column meaningless, which is why there is no
timing column here — this measures convergence and robustness, and those
are contention-independent.

---

# Why example 28 does not converge, and what to do about it (2026-09-15)

The report above recommended raising `maxIterations` on example 28 and
re-measuring. **That recommendation was wrong, and the measurement is what
says so.**

## Raising the cap does nothing

1200 steps per arm, scene re-seeded each time:

| iteration cap | converged | mean iterations | steps that hit the cap |
| --- | --- | --- | --- |
| 100 | 1192/1200 | 33.2 | 9 |
| 400 | 1185/1200 | 37.4 | 15 |
| 2000 | 1194/1200 | 42.7 | **6** |

Solves are running to **2000 iterations** and still not converging. The
distribution is bimodal: 1156 of 1200 finish in 20-49 iterations and a
handful never finish at any budget. These are not slow solves. They are
stalled.

## It is not a singular sub-domain

The first hypothesis was an isolated pocket of fluid with no Dirichlet
(air) neighbour, which would make the local system singular. Connected-
component analysis of the fluid mask on the first failing step refutes it:
**4731 fluid cells, one component, zero orphans.** Same at the end of the
run.

## It is float32 running out of digits

The stop test compares `sqrt(r.r)` — an **absolute** L2 norm over the whole
field — against `tolerance`. What that demands therefore depends on how
big the problem is. Measured:

| scene | ‖b‖ | reduction demanded by `tolerance = 1e-5` |
| --- | --- | --- |
| 28 drop into pool | **~550** | **5.5 × 10⁷** |
| 26 dye in free surface | ~370 | 3.7 × 10⁷ |

float32 has a 24-bit mantissa, about **6 × 10⁻⁸** of relative precision, so
the largest reduction the arithmetic can express is around **1.7 × 10⁷**.
Both scenes are asking for more digits than exist. Example 26 clears it on
margin; example 28 is far enough past that ~5% of its solves stall at a
residual they cannot improve.

Three independent measurements agree:

- more iterations do not help (2000 is no better than 100);
- a tolerance of **1e-4** converges **1500/1500**, and 5e-5 converges
  1491/1500 — the failures sit between those two;
- ‖b‖ puts the demand past the float32 limit arithmetically.

**This is a property of the default, not of example 28.** Example 26 is
passing by luck rather than by margin.

## The fix, and the level matters more than the criterion

`relativeTolerance` (in `settings`, off by default) compares against
`tolerance * ‖b‖` instead of `tolerance`, with the absolute test kept as an
OR so a zero right-hand side still converges. ‖b‖ is reduced into the
scalar buffer during the setup, which is already all GPU dispatches, and
read out of the snapshot the host already takes — **no extra round trip.**

Relative is the textbook criterion and it is scale-free, which is what this
project wants. But it changes what accuracy is being asked for, and that
has to be measured rather than assumed. 1200 steps, post-projection
divergence over fluid cells:

| setting | converged | mean iterations | worst max \|div\| | mean \|div\| |
| --- | --- | --- | --- | --- |
| absolute 1e-5 (today's default) | 1193/1200 | 32.8 | 5.0e-5 | 5.75e-6 |
| relative 1e-5 | **1200/1200** | 15.3 | **1.88e-3** | 3.07e-5 |
| **relative 1e-7** | 1198/1200 | **29.6** | **4.0e-5** | **5.08e-6** |

**Relative 1e-5 converges every time by asking for much less.** With
‖b‖ ≈ 550 its threshold is 5.5e-3, some 550× looser than today's, and the
divergence is 37× worse. The halved iteration count and the higher peak
pressure (23.6 against 15.6) are the same fact seen three ways. It is not
the fix.

**Relative 1e-7 is better than today's default on every axis measured** —
more solves converged, fewer iterations, and slightly *better* divergence
— because its threshold (~5.5e-5) is loose enough for float32 to reach and
tight enough to preserve the physics.

## What this does not fix

Relative 1e-7 still leaves 2 of 1200 unconverged. The floor is real: on
this scene the residual cannot go far below ~4e-5 in float32 whatever the
criterion. A criterion change moves the goalposts to where the arithmetic
can reach; it does not add precision. Closing the last fraction of a
percent would need a higher-precision residual (compensated summation in
the reduction, or a mixed-precision correction step), not another
threshold.

## Recommendation, and why the default is unchanged

For a scene like example 28, `relativeTolerance: true` with
`tolerance: 1e-7` is the measured best of the three.

The default is left at absolute for now because `tolerance` is a public
option and switching the criterion silently changes what every existing
caller's number means — the same hazard that had example 15 pinning
`residualCheckInterval: 1` and opting itself out of a changed default. A
default change here should come with the `tolerance` default moving to
1e-7 in the same commit, and a check across every scene, not just this one.

---

# The real cause: the solver was reporting convergence it had not achieved

The section above blamed float32 precision. That was the wrong mechanism
too. The cause is a correctness bug, it affects **every scene**, and it
invalidates the convergence column of the 12,000-step table at the top of
this document.

## What was happening

CG does not recompute `b - Ax` every iteration. It tracks the residual
incrementally with `r -= alpha * Ap`, and that estimate **drifts
optimistically** -- the update is a near-cancellation of two similar
quantities, so the error grows relative to a shrinking residual. The true
residual is recomputed only every 50 iterations.

A typical solve on these scenes finishes in **14 to 33 iterations**. It
therefore never reached the recompute, and stopped on a number that had
drifted below the threshold without the solution following it there.

Measured on `examples/28-drop-into-pool/`, same frame, same restored state
(particles, velocities and pressure), only the recompute interval varying,
with the true residual computed independently on the host in double
precision:

| recompute every | iterations | reported converged | solver's residual | true residual | ratio |
| --- | --- | --- | --- | --- | --- |
| **50 (the default)** | 32 | **yes** | 7.40e-6 | **3.571e-4** | **48x** |
| 10 | 48 | yes | 7.43e-6 | 8.773e-5 | 11.8x |
| 5 | 200 (cap) | no | 2.279e-4 | 2.325e-4 | 1.02 |
| 2 | 200 (cap) | no | 2.130e-4 | 2.131e-4 | 1.00 |
| 1 | 200 (cap) | no | 2.038e-4 | 2.038e-4 | 1.00 |

Read the bottom three rows first: when the residual is recomputed often
enough, the solver's number matches the independent host computation to
three significant figures, which validates the host reconstruction of the
operator and with it everything above. Those rows also never reach 1e-5,
because ~2e-4 is what this operator can actually achieve.

Then read the top row. At the shipped default the solver stopped at
iteration 32 claiming 7.40e-6 while the truth was 3.571e-4.

**So example 28 never converged. Nor did any other scene.** With
verification switched on and the old absolute 1e-5 target,
`examples/26-dye-free-surface/` converges **0 times in 600**, against the
800/800 it used to report.

## What was ruled out on the way

Each by measurement, and each worth not re-testing:

- **Iteration budget.** A 2000-iteration cap fails as often as 100.
- **Isolated fluid pockets.** Connected-component analysis on a failing
  step: 4731 cells, one component, zero orphans.
- **Variable density.** Density ratio 1.00 fails as often as 1.40.
- **The reduction's summation precision.** Host double-precision partials
  are no better than GPU float32 ones (19 failures against 9).
- **Submission batching.** Identical numbers batched and unbatched, same
  frame.
- **Catastrophic cancellation in the stencil**, which the previous section
  claimed. It used norm-of-b as if it were a per-cell value, and by
  Sterbenz's lemma the difference of two nearby f32 values is exact.

`src/linalg/double_single.js` was written for that last hypothesis -- a
~48-bit float from two f32s, since WGSL has no f64. It is kept, and
verified on hardware by `examples/27-float-guard-probe/` (plain f32 error
5.31e-5 against double-single 0.00e+0, so the compiler does not optimise
the error-free transformations away), but it is **not** what fixed this.

## The fix, which is two changes that only work together

**`verifyConvergence`** (default on): when the tracked residual first
claims success, recompute the true residual and test that instead. Costs
one Laplacian apply and one dot product per solve. After a failed
verification the loop recomputes every iteration, without which it
oscillates -- the tracked residual is already under the threshold, so it
asks, fails, takes one step, dips under again, and runs to the cap. That
oscillation showed up as every arm reporting a mean iteration count of
exactly the cap, which is how it was caught.

**`relativeTolerance`** (default on) with **`tolerance` now 1e-6**: the
test compares against `tolerance * norm(b)`. Verification alone would be
honest and useless -- every scene would run to its cap chasing an
unreachable absolute target. The norm is reduced into the scalar buffer
during the already-GPU-resident setup and read from the snapshot the host
takes anyway, so it costs no round trip.

## Measured after the fix, with verification on

| example | converged | rejected | mean iterations | peak pressure | historical |
| --- | --- | --- | --- | --- | --- |
| 16 Karman vortex street | 400/400 | 0 | 19.3 | - | - |
| 20 FLIP dam break | **600/600** | 0 | 15.6 | 20.765 | **20.765** |
| 26 dye in free surface | **800/800** | 0 | 12.3 | 10.382 | **10.382** |
| 28 drop into pool | **800/800** | 0 | 27.8 | 15.614 | **15.614** |

Every peak pressure matches its historical value to every digit, so the
physics is unchanged. Example 28, the scene this investigation started
from, now converges on every step -- honestly -- where it previously
reported 93.8% dishonestly.

The iteration counts are also lower than the old dishonest ones (28: 27.8
against 33.2). An achievable target is reached and left; an unachievable
one is chased.

## Two consequences for callers

**`tolerance` has changed meaning.** It is relative now. Every example
that pinned it has had the pin removed so the default applies -- the same
"a default is not in force where a caller names the option" hazard that
had example 15 opting out of a changed `residualCheckInterval`.

**`examples/16-karman-vortex-street/` keeps an explicit `tolerance: 1e-5`**
and says why: at the library default of 1e-6 it converges 0 times in 400
within its 40-iteration budget, and 18 of 400 even at 100 iterations, so
1e-6 is below what that operator achieves. A caller's accuracy requirement
is not the kind of constant this project's no-magic-numbers rule forbids;
an internal number tuned per scene to make the solver work is.
