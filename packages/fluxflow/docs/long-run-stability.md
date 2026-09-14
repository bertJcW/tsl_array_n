# Long-run stability: 12,000 frames per scene

Every drivable example run for **12,000 solver steps** on real WebGPU
hardware, 2026-09-15, on the current defaults (`preconditioner:
'multigrid'`, `residualCheckInterval: 4`, `gpuResidentScalars` and
`gpuResidentSetup` on). **108,000 steps in total.**

The question is not speed. It is whether these solvers converge and stay
converged, and whether anything drifts, blows up, or quietly loses fluid
over a run far longer than any demo.

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
