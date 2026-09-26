# Long-run stability: 12,000 frames per scene

Two runs are recorded here, and the older one is kept rather than replaced
because most of what it measured still holds and because this repository
retracts in place instead of quietly dropping things.

- **[The 2026-09-26 run](#the-2026-09-26-run-15-scenes-180000-steps-judged-not-counted)
  is the current one.** 15 scenes, 180,000 solver steps, each scene given a
  verdict from conservation laws rather than a column of counters. It is the
  re-run the 2026-09-15 report asked for.
- **[The 2026-09-15 run](#the-2026-09-15-run-superseded-kept-for-the-record)**
  is below, with its own warning box intact. Its `converged` column was
  never valid; its non-finite counts, rejection counts, peak pressures and
  occupied-cell curves were, and the two investigations that follow it --
  why example 28 stalled, and the float32 digit budget -- are still the
  reference for those questions.

---

# The 2026-09-26 run: 15 scenes, 180,000 steps, judged not counted

Every example that exposes a probe, 12,000 solver steps each, sampled every
100, on real WebGPU, run sequentially. `node long_run.mjs`.

## Why the instrument changed

The 2026-09-15 report ends with "a full 12,000-step re-run is the
outstanding item". This is that re-run, and it does not report the same
thing, because the 3D investigation established that the thing the old
report measured cannot decide the question.

Counters cannot. `examples/16-karman-vortex-street/` converged on **every
frame** of a run in which its outflow carried 13.89x its inflow out of the
domain, and `examples/35-karman-vortex-street-3d/` settled after blowing up
into a state whose maximum velocity was a perfectly ordinary 17.5. Both
would have passed any column in the old table. So each scene here goes
through `solver_health.mjs`, which judges conservation of mass and the
residual the projection leaves behind, exits 0 or 2, and prints the counters
beside the verdict as evidence to explain rather than as criteria.
`../docs/verification.md` describes the instrument and the three rules it
is built on.

## Results

180,000 solver steps. Every run confirmed `WebGPUBackend` before any number
from it was believed.

| example | verdict | steps | converged | rejected | CG breakdowns | residual left, median / worst | fps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 15 flow past cylinder | **HEALTHY** | 12,001 | 11,998 | 0 | 0 | 5.03e-7 / 2.84e-4 | 57 |
| 16 Kármán vortex street | **HEALTHY** | 12,001 | 12,000 | 0 | 0 | 7.84e-6 / 1.27e-3 | 58 |
| 17 smoke and fire | **HEALTHY** (narrow) | 12,001 | **0** | 0 | 1 (frame 0) | 2.21e-5 / 3.57e-4 | 55 |
| 18 explosion | **HEALTHY** (narrow) | 12,001 | **0** | 0 | 0 | 1.99e-5 / 3.42e-4 | 54 |
| 19 fuel fire | **HEALTHY** (narrow) | 12,001 | **0** | 0 | 1 (frame 0) | 5.43e-5 / 4.62e-4 | 58 |
| 20 FLIP dam break | **HEALTHY** (narrow) | 12,001 | 12,001 | 0 | 0 | 1.95e-8 / 9.50e-7 | 59 |
| 21 irregular container | **HEALTHY** (narrow) | 12,001 | 12,001 | 0 | 0 | 2.41e-7 / 9.98e-7 | 37 |
| 22 multiple colliders | **HEALTHY** (narrow) | 12,001 | 12,001 | 0 | 0 | 3.86e-7 / 9.60e-7 | 49 |
| 23 moving collider | **HEALTHY** (narrow) | 12,001 | 12,001 | 0 | 0 | 1.02e-8 / 9.48e-7 | 59 |
| 26 dye in free surface | **HEALTHY** (narrow) | 12,001 | 12,001 | 0 | 0 | 2.43e-8 / 9.76e-7 | 59 |
| 28 drop into pool | **HEALTHY** (narrow) | 12,001 | 12,001 | 0 | 0 | 7.97e-7 / 9.97e-7 | 53 |
| 29 static droplet | **HEALTHY** (narrow) | 12,001 | 12,001 | 0 | 0 | 4.47e-5 / 5.70e-5 | 48 |
| 33 FLIP dam break 3D | **HEALTHY** (narrow) | 12,001 | see below | 0 | 0 (see below) | 7.95e-8 / 9.81e-7 | 39 |
| 34 smoke plume 3D | **HEALTHY** (narrow) | 12,001 | 12,000 | 0 | 1 (frame 0) | 4.33e-7 / 1.20e-6 | 59 |
| 35 Kármán vortex street 3D | **HEALTHY** | 12,001 | 12,000 | 0 | 0 | 4.37e-6 / 1.61e-3 | 51 |

**Zero rejections in 180,000 solves.** The pressure circuit breaker never
reverted a solve in any scene.

"Narrow" is not a hedge, it is the scope of the pass. Twelve of these
scenes are free-surface or vent-only, so the flux and mass-balance criteria
do not apply to them and the probe says so rather than counting them as
passes; what still applied is finiteness, the projection residual and the
CG guards. Only three scenes -- 15, 16 and 35, the ones with an inlet --
were judged on every criterion. See `verification.md`, rule 3.

The headline: **`examples/35-karman-vortex-street-3d/` runs 12,001 steps
clean at 51 fps.** That is the scene the entire 3D investigation was about,
the one that used to pin all three velocity components at the solver's
clamp of 100 within 600 frames.

## What this run found

### 1. Example 33 was flagged BROKEN by a criterion that was wrong

The first 12,000-step run of `examples/33-flip-dam-break-3d/` reported
**2,589 CG breakdowns beginning at frame 9412** -- every frame from 9412 to
the end -- and was judged BROKEN. Two things came out of chasing it, and the
second one is a defect in this probe rather than in the solver.

**It does not give the same answer twice.** Four runs of the same build:

| run | verdict | guard stops | from frame |
| --- | --- | --- | --- |
| 1 | BROKEN | 2,589 | 9412 |
| 2 | HEALTHY | 0 | — |
| 3 | BROKEN | 7,557 | 4444 |
| 4 | HEALTHY | 0 | — |

When it settles it settles anywhere: a later run reached rest at frame 1605
and spent the remaining 10,396 frames there.

The sampled fields diverge early -- maximum divergence at frame 300 reads
1.45 in one run and 2.51 in another -- so this is not a threshold being
grazed, it is two different trajectories.

> **Retracted (2026-09-26).** This section said the mechanism was the
> resampler's atomic donor cursor. That was wrong, and it was wrong in the
> way this document keeps warning about: it named the first plausible
> mechanism found by reading code, and did not test it. Replacing the cursor
> with a deterministic rank did not make the scene reproducible. The real
> cause and the measurements that found it are in
> [T2](#t2-the-scenes-are-reproducible-now-and-the-cause-was-not-what-i-said-it-was)
> below. The cursor was a genuine order-dependence and is fixed; it was not
> what made these runs differ.

**But the stops were never a failure, and the probe should not have said
they were.** Instrumenting the stop reason settles it: all 7,557 stops in
run 3 are `degenerate-pAp`, and **all 7,557 left a residual of exactly 0**.
The first forty read `0.0e+0` without exception. The liquid in this scene
comes to rest partway through the run; after that the pressure system is
solved exactly, so the search direction is zero, so `p.Ap` is zero, and the
guard fires every frame on a system that has nothing left to solve.

`linalg.js`'s `applySnapshot` already refused to name this case, in a comment
that gives the reason in full -- "a guard tripping on an already-converged
residual is not a failure: there is nothing left to solve, p is ~0 so p.Ap is
~0, and alpha being forced to 0 leaves x exactly where it belongs." That test
guarded **one of the seven sites that set `stoppedBy`**, and not the one this
scene goes through.

An earlier draft of this section named the four host-loop sites as the
culprits. That was wrong: a 3D scene runs the GPU-resident path, and the two
sites there that return a stop code directly (`solveWithGpuResidentScalars`'s
chunk loop and its final read) are what reported the non-event here. The host
loop's four guards can reach the same state by a different route --
`residualCheckInterval > 1` skips the convergence test on most iterations, so
the residual can fall below `tol` without the loop noticing and the next
iteration's `p` is then ~0. Six sites were missing the test, not four.

`solver_health.mjs` no longer counts a stop that left a residual of exactly
0 as a breakdown. It reports them on their own line instead, because a scene
coming to rest is worth seeing.

**The fix was verified by watching it fire, not by watching the suite go
green.** Two runs immediately after it came back HEALTHY with zero stops --
and proved nothing, because they were runs in which the liquid never settled,
so the corrected path was never reached. That is the same shape of mistake as
the ones-field dot product test: an input that cannot distinguish the
failure. A third run settled at frame 1605 and reported

```
solver counters: 1605 converged, 0 rejected, 0 CG breakdowns
   10396 guard stops on a system already solved to a residual of exactly 0
          -- not counted as breakdowns
VERDICT: HEALTHY over 12001 frames
```

which before the fix would have read BROKEN with 10,396 breakdowns. Six
12,000-step runs of this scene in total; every one of them HEALTHY under the
corrected criterion, and two of six would have been BROKEN under the old
one.

This is the fifth false positive this probe has produced and the fifth to be
found by pursuing a verdict instead of accepting it; `verification.md`'s
rule 3 exists because of the first four. Note what did *not* happen: no
threshold was widened and no scene-specific exception was added. A stop that
leaves zero residual is benign by definition, everywhere.

**The library was fixed too, and that is the part that matters.** All seven
sites now go through one function, `guardIsANonEvent`, so the rule exists in
one place instead of being reimplemented -- or forgotten -- at each. The probe
excluding these stops only protected the probe; every other caller of
`solve()` was still being told a solve broke down when it had finished.

The regression that had to hold is that genuine failures still report, and
`examples/05-preconditioned-conjugate-gradient/` is the page that says so: a
singular operator (A = 0) still reaches `degenerate-pAp` and a near-null one
(A = diag(1e-12)) still reaches `alpha-magnitude`, on all three paths -- host,
GPU-resident, and GPU-resident checking every 4th iteration. Silencing a real
guard while fixing a false one is the obvious way for this change to go wrong,
and that page is what rules it out.

Verified the way the correction before it was not: by finding a run in which
it fires. The probe now counts frames whose residual arrives at exactly 0 --
the scene at rest -- independently of any stop reason, because once the
library stops reporting the non-event a settling run and a non-settling run
look identical from outside, and those are precisely the two that have to be
told apart. A run that settled at frame 6330 and spent its remaining 5,672
frames at rest reports **0 CG breakdowns**, and the probe's own exclusion
never fires, because there is no longer anything to exclude. Before this
change that run would have reported 5,672 breakdowns.

What remained undone after T1 -- making these scenes reproducible -- is T2,
below.

One caveat on the fix itself, recorded because it is the obvious way for
this to go wrong later: excluding a zero-residual stop trusts `d.residual`
to be a true residual, and the 3D dot product defect is precisely what made
that false -- a broken reducer can report 0 for a system nowhere near
solved. What keeps it honest is that the residual is not the only thing
measured: the sampled divergence is computed by the probe from the velocity
field itself, independently of anything the solver claims about its own
progress.

### 2. Three scenes never converge once in 12,001 steps, and are healthy

`17-smoke-fire`, `18-explosion` and `19-fuel-fire` report **0 converged
frames out of 12,001**. They are also fine: the projection leaves a median
of 2.0e-5 to 5.4e-5 of the divergence it was asked to remove, worst 4.6e-4,
against a bar of 1e-2.

All three run `maxIterations: 60`, so they spend the full budget every
frame and never reach the tolerance they asked for. The control is
`34-smoke-plume-3d`, which has the same cap and converges on 12,000 of
12,001 steps.

This is the sharpest instance yet of why the convergence counter is
reported and not judged. Read as a criterion it says these three scenes are
0% healthy, which is wrong. Ignored entirely it hides that they are burning
60 iterations a frame to buy nothing they asked for, which is real. Both
numbers are needed; only one of them can decide.

Pursued as T3, below, where it turns out to be worse than wasted iterations.

### 3. Example 28 no longer stalls

The 2026-09-15 report's own follow-up investigation established that
`28-drop-into-pool` could not converge at any iteration budget -- 1194 of
1200 at a cap of 2000 -- because its absolute stop test asked float32 for a
reduction of 5.5e7 when the arithmetic can express about 1.7e7. That
investigation is below and its reasoning is unchanged.

It converges on **12,001 of 12,001 steps** now, with a worst leftover
residual of 9.97e-7.

The likely cause is the 2026-09-16 change in which the solver recomputes
the true residual every iteration and runs its stop test on the GPU, which
postdates that investigation. This has not been confirmed by bisection, and
is recorded as the plausible explanation rather than the established one.

### 4. Four scenes were discarded, and the probe was right to discard them

`21`, `22`, `26` and `29` first reported `NOT WEBGPU` and 0 frames. Their
probes did not expose `renderer`, so the backend check read `unknown` and
`solver_health.mjs` threw the whole run away rather than report a number
whose provenance it could not confirm. That is the correct behaviour and it
cost four runs, which is the cheaper failure by a wide margin. `renderer`
is now on those four probes and all four pass on re-run.

## Coverage, and what was not run

15 of 36 examples expose `window.__fluxflowProbe` and were run, against 9
in 2026-09-15. The rest, named rather than left implicit:

- **Self-checking pages, which have no long run to do**: 00-13, 27, 30, 31,
  32. They verify against a known answer on load and finish -- layer 2 in
  `verification.md`, not layer 3. `31-conjugate-gradient-3d` and
  `32-grid-solver-3d` are the 3D linear algebra and grid solver checks.
- **Scenes with no probe, which was a real gap**: `14-stable-fluids`,
  `24-two-phase-bubble-rise`, `25-dye-injection`. All three have one now --
  see [closing the probe gap](#closing-the-probe-gap-and-what-was-behind-it)
  below, which is where the gap turned out to be hiding a broken scene.

## Method

Each scene runs in its own real Chrome, launched with the
anti-occlusion-throttling flags, at its own frame rate through its own rAF
loop -- what a long run asks about is the scene as it actually runs.
`solver_health.mjs` intercepts the assignment to `window.__fluxflowProbe`
before the first frame, so recording starts at frame 0; reading a run from
its tail is how `35-karman-vortex-street-3d` was once reported stable over
14,643 frames while its first 600 frames contained the blow-up.

Scenes run **sequentially**, one browser at a time, which is why there is
an fps column here and there was none in 2026-09-15: concurrent tabs
contend for the GPU and make wall-clock meaningless.

Full per-scene logs, including every sampled row, are written to
`long-run-logs/` by the orchestrator and are not committed.

```bash
node long_run.mjs                          # all of it, 12,000 steps each
node long_run.mjs 2000 50 35-karman        # one scene, shorter
```

---

# T2: the scenes are reproducible now, and the cause was not what I said it was

Every FLIP scene in this package gave different results on every run. The
write-up above blamed the resampler's atomic donor cursor, on the strength of
reading the code. Replacing that cursor with a deterministic rank changed
nothing, which is the only reason the actual cause was ever found.

## What it actually was

`computeFlipBoxSeed2`, `computeFlipBoxSeed3` and `computeTwoPhaseBoxSeed`
jittered every particle's starting position with `Math.random()`. **Every run
of every FLIP scene began from a different initial condition.** Not a race,
not an atomic, not the solver -- the seeding.

The jitter itself is wanted: particles left on an exact lattice produce
visible artefacts, and every FLIP implementation breaks the lattice up. What
was missing was a jitter that repeats. `jitter_random.js` supplies one
(mulberry32, four lines, one uint32 of state) and all three seeding functions
take a `randomSeed` that defaults to a constant, so a scene is reproducible
unless its author asks otherwise.

## How it was found, after guessing failed

By dropping to a scene cheap enough to run repeatedly. The 12,000-step runs
cost five minutes each, which is long enough that hypotheses get tested one
per coffee; `examples/32-grid-solver-3d/`'s own FLIP self-check -- 864
particles, a closed 8x8x8 domain, 60 steps -- runs in seconds and reported a
peak particle speed of **3.532, 5.020 and 5.396 on three consecutive runs**.
After the fix, 3.885 three times.

The other half was localising it before hunting. Two runs of
`examples/35-karman-vortex-street-3d/` are identical line for line, which
clears the whole shared core -- CG, multigrid, advection, the boundary
conditions, the fixed-point atomic dot products -- and left the search inside
the particle code. That took one measurement and saved reading several
thousand lines.

## Two real races found on the way, and kept

Neither turned out to explain the divergence. Both are defects anyway, and
one of them is worse than the thing that was being looked for.

**The velocity extrapolation read and wrote the same buffer in one dispatch.**
`createExtrapolateStepKernel2`/`3` averaged a cell's valid neighbours out of
`output` and wrote the result back into `output`, in the same kernel. The
valid mask beside it is carefully double-buffered and ping-ponged -- the sweep
is meant to be Jacobi -- but the values were not, so whether a cell saw a
neighbour's old value or its just-written new one depended on how the device
scheduled that dispatch. **Every caller passes the same array as input and
output**: both blocked-boundary solvers, both FLIP solvers and the two-phase
solver, which is to say nearly every scene in the package. It is a true Jacobi
sweep now, with the value field double-buffered and a copy back when an odd
iteration count leaves the answer in the scratch buffer.

**The donor cursors were order-dependent, in all three resamplers.** This one
has a clean attribution, which arrived by accident: with the seeding fixed and
the cursors still in place, `examples/20-flip-dam-break/` was the one FLIP
scene still giving different answers, while `examples/28-drop-into-pool/` --
the same solver, but calm enough that the resampler rarely has both an
over-dense and an under-filled cell at once -- was already reproducible. That
difference is what pinned it. All three resamplers now take ranks from a
prefix sum: a donor's slot is the number of donors before it in particle
order, and a cell's slice of the pool is fixed by cell order. The donor set,
the receiving cells and what happens to a relocated particle are unchanged;
only the arbitrary choice between them is now made the same way every time.

The cursors were verified on real hardware before being relied on -- N threads
each claim a slot, every slot 0..N-1 written exactly once, no collisions, at
N = 256 and N = 4000. **That test checked uniqueness, which is the right
property for the data structure and the wrong one for the simulation.** A GPU
primitive can be correct and still be unrepeatable, and only a test that runs
it twice can tell. `sandbox/prefix-sum/` is the replacement's test and it
checks both: 13 checks on real WebGPU, including eight scans of one input
coming back bit-identical, and ranks increasing in the index rather than
merely being distinct.

## Verified

Nine scenes, two 1,000-step runs each, logs diffed line for line -- every
sampled row, not just the verdict.

| scene | two runs identical | verdict |
| --- | --- | --- |
| 20 FLIP dam break | yes | HEALTHY |
| 21 irregular container | yes | HEALTHY |
| 22 multiple colliders | yes | HEALTHY |
| 23 moving collider | yes | HEALTHY |
| 24 two-phase bubble rise | yes (three runs) | HEALTHY |
| 26 dye in free surface | yes | HEALTHY |
| 28 drop into pool | yes | HEALTHY |
| 29 static droplet | yes | HEALTHY |
| 33 FLIP dam break 3D | yes | HEALTHY |

## The sweep's own false pass, and what it exposed

The first run of that sweep reported all nine identical, including
`24-two-phase-bubble-rise`. It had not run at all: that scene had no probe, so
`solver_health.mjs` timed out waiting for one, and **two identically-failed
runs diff clean.** A determinism check built on comparing outputs will call a
pair of failures a pass, every time.

Fixing it turned up something worse than the false pass. With a probe added,
example 24 ran -- and on two runs under the old code gave **HEALTHY once and
BROKEN once**, the broken one leaving 1.46e-2 of the divergence the projection
was asked to remove, against a bar of 1e-2. The two-phase solver had never had
a long run or a determinism check, and it was reachable that both would fail.
It is deterministic and HEALTHY over three runs now.

**Recorded rather than claimed:** the trajectory that read 1.46e-2 is no
longer the one this scene takes, which is not the same as showing the solver
cannot produce it. Whether that excursion recurs under other seeds is
untested, and is now testable precisely because the seed is a parameter --
which is the practical argument for determinism, over and above tidiness.

---

# T3: the solve was walking away from its own best answer, silently

`17-smoke-fire`, `18-explosion` and `19-fuel-fire` converge on 0 of 12,001
steps. The obvious readings are "the budget is too small" and "the
preconditioner is too weak", and the way to tell them apart is to freeze one
frame and re-solve the identical system at increasing budgets
(`diag_iterations.mjs`, written for this). Example 17, frame 400, every arm
starting from the velocity and pressure the frame arrived with:

| preconditioner | 60 iterations | 300 | 3000 |
| --- | --- | --- | --- |
| multigrid | 4.12e-4 | 2.17e-3 | **7.45e-1** |
| jacobi | 1.50e+0 | 8.78e-3 | 1.58e-4 |
| none | 1.50e+0 | 9.18e-3 | 1.87e-4 |

Neither reading was right. **With the multigrid preconditioner the residual
grows with the iteration count** -- 1800x worse at 3000 iterations than at 60
-- while the same frozen system with jacobi, or with no preconditioner at all,
converges normally over the same 3000. Unpreconditioned CG converging
monotonically clears the operator and the arithmetic; the V-cycle is what is
being amplified.

`|b|` here is 26.9 and the tolerance is 1e-6 relative, so the stop test
demands a residual below 2.69e-5 and the best the stack reaches is 4.12e-4.
**That tolerance is unreachable at any budget**, which is why the counter read
0 and kept reading 0.

## What it was not

Two hypotheses died on measurement, and the second one inverted.

**Not the level count.** `16-karman-vortex-street` uses the same
`numberOfLevels: 4` and converges on 12,000 of 12,001 steps.

**Not the V-cycle's asymmetry**, which is the thing this repository has been
caught by before and the obvious suspect for a preconditioner that destroys
PCG. Measuring it directly on each scene's own frozen system
(`diag_symmetry.mjs`, which needed `grid_pressure_solver2.js` to expose its
preconditioner builder for measurement):

| scene | (Mx,y) vs (x,My), relative | tolerance / budget | converged |
| --- | --- | --- | --- |
| 16 Kármán vortex street | **1.00e-2** | 1e-5 / 40 | 12,000 / 12,001 |
| 17 smoke and fire | 7.10e-5 | 1e-6 / 60 | 0 / 12,001 |
| 19 fuel fire | 7.10e-5 | 1e-6 / 60 | 0 / 12,001 |

The scene that converges has a V-cycle **140x less symmetric** than the ones
that do not. Asymmetry is real here and it is not the discriminator; the
discriminator is whether the tolerance asked for is one this stack can deliver
at that budget.

(Examples 17 and 19 report identical figures because they are the same shape
with the same mask and the probe seeds its random draws identically -- which
is a determinism check passing, not a copy-paste error.)

## The fix, and why it is not a per-scene number

The tempting fix is to loosen these three scenes' tolerance to something
reachable. That is a magic number per scene, it leaves the next scene to
rediscover the same thing, and it does not address what actually went wrong,
which is that **nothing told anyone.** `converged: false, stoppedBy: none` is
what a solver reports when the budget was simply too small, and it was
reporting that while returning an answer 1800x worse than the one it had
passed through at iteration 60.

So `linalg.js` now distinguishes the two. It tracks the best residual a solve
has seen, and if the residual grows past 100x that best -- 10x in the norm,
chosen to be unmistakable rather than a fluctuation -- the loop stops and sets
`stoppedBy = 'residual-growing'`. Both paths do it: the host loop at its own
residual check, and the GPU-resident path in the chunk loop, which is the only
place a host sees a residual in that path at all.

Measured on the same frozen frame, same command:

```
multigrid      3000   1250       false    2.20e-2      8.18e-4   residual-growing
jacobi         3000   3000       false    1.58e-4      5.85e-6   none
none           3000   3000       false    1.87e-4      6.96e-6   none
```

It stops at 1250 instead of 3000, returns a residual **34x better** than
before, and names the reason. The jacobi and unpreconditioned arms of the same
run are untouched, which is the evidence that it fires on the pathology and
stays quiet on ordinary convergence -- both sides of the threshold, in one
measurement.

## Verified, including the part that could have gone wrong

The risk in a new stop condition is that it fires on a healthy scene and turns
a good verdict bad.

- `examples/05-preconditioned-conjugate-gradient/` still reaches
  `degenerate-pAp` on a singular operator and `alpha-magnitude` on a near-null
  one, on all three paths. The new reason did not displace the old ones.
- 375 tests pass.
- It fires **once** on `35-karman-vortex-street-3d`, at frame 0 -- the first
  solve of an impulsively started flow, which is the known hardest frame in
  these scenes and the one example 16 has always missed. 2,000 of 2,001 frames
  converge and the verdict stays HEALTHY. That is a real detection stopping a
  frame early rather than a misfire.

## What is still open, and is a question rather than a defect

The three smoke scenes still ask for 1e-6 and still get ~1.5e-5. Now they say
so instead of leaving a counter at zero, and the reachable floor is measured,
so the choice is informed: either accept a tolerance this V-cycle can deliver,
or make the V-cycle exact enough to deliver 1e-6. The second is the real
answer and it is a preconditioner project, not a constant.

## A retraction, and the tool that caused it

The paragraph that stood here said `sandbox/poisson-3d-dirichlet/` should have
caught this and did not, because it only checked V-cycle symmetry at 1 and 2
levels while every scene that uses the V-cycle in earnest runs 3 or 4.

**That was false. The sandbox has always looped over 1, 2, 3 and 4 levels.**
What was wrong was `run_page.mjs`, which waited a fixed fifteen seconds and
then printed whatever the page had produced so far. This page takes 34 seconds.
Fifteen got three of its sixteen V-cycle rows, and **the output of a truncated
run looks exactly like the output of a complete one** -- a list of ticks. The
conclusion above was reasoned from that list.

`run_page.mjs` now waits for the page to stop producing output rather than for
a clock, and labels a run PARTIAL, loudly, when it hits its cap instead. The
page also prints its own summary line as its last act, so a run with no summary
is visibly unfinished from the inside as well as the outside.

This is the third time in this document that a measurement harness reported a
partial or absent run as a pass -- after reading a 14,643-frame log from its
tail, and after a determinism sweep diffing two runs that had both failed to
start. It is the same mistake each time and it is worth naming as a class: **a
check that cannot tell "finished and fine" from "did not finish" is not a
check.** `verification.md`'s rule 0 covers the comparison case; this one adds
the truncation case.

## What the complete run actually says

V-cycle symmetry, relative difference between (Mx,y) and (x,My), on example
35's own shape and mask:

| levels | mask only | mask + weights |
| --- | --- | --- |
| 1 | 3.76e-3 (expected failure) | 2.80e-3 (expected failure) |
| 2 | 7.03e-6 | 5.55e-5 |
| 3 | 1.60e-4 | 9.77e-6 |
| 4 | 1.07e-4 | 1.63e-4 |

So symmetry does not degrade with the level count -- it is best at 2 levels and
sits around 1e-4 at 3 and 4, which is consistent with the 7.10e-5 measured on
example 17's own live system. A 1-level V-cycle is plain red-black relaxation
and is not symmetric at all; that row is the contrast that shows what the
coarse levels buy, and it is now marked as an expected failure rather than
appearing as a plain cross.

It also puts a number on how little this explains. `16-karman-vortex-street`
runs a V-cycle asymmetric to 1.00e-2 -- worse than the 1-level case that fails
this page outright -- and converges on 12,000 of 12,001 frames. Whatever sets
the floor on achievable residual, it is not asymmetry alone.

## The page had four permanent red marks and nobody read them

Two unpreconditioned-CG rows and the two 1-level symmetry rows have always
failed and always should: the first pair is the control arm the V-cycle is
being measured against, the second is relaxation being asked to be symmetric.
Reported as plain crosses among the ticks, they made the page's own summary
read "4 failed" on every run, which is indistinguishable from a regression.

Expected failures now say their reason inline, count separately, and the
summary reads `34 passed, 4 expected failures, 0 unexpected` -- a number that
is zero when nothing is wrong. An expected failure that starts *passing* is
also flagged, since that means the thing the expectation was about has changed.

Extending the level coverage is no longer the next step on this thread; it was
never missing. Making the V-cycle exact enough for a 1e-6 tolerance is.

---

# Closing the probe gap, and what was behind it

Three scenes had no `window.__fluxflowProbe`, so nothing could step them
faster than `requestAnimationFrame` and none of them had ever had a long run.
That was listed as an open gap rather than hidden, and closing it cost about
five lines each.

It was worth it on the first run.

| scene | verdict over 12,000 steps | converged | breakdowns | stopped early |
| --- | --- | --- | --- | --- |
| 14 stable fluids | **HEALTHY** (narrow) | 11,999 / 12,001 | 0 | 2 |
| 24 two-phase bubble rise | **HEALTHY** (narrow) | — | 0 | 0 |
| 25 dye injection | **BROKEN at frame 800** | 1,283 / 12,001 | 175 | 221 |

`14-stable-fluids` is the reassuring one, and it is the scene that most needed
asking: a fully closed autonomous domain, nothing driving it and nothing
leaving, which is the configuration a slow leak would show up in. 11,999 of
12,001 solves converged, and the two that did not are the subject of the next
section.

## The new stop reason was making a healthy scene read BROKEN

Example 14's first 12,000-step run came back **BROKEN**, on two frames out of
12,001 -- 4852 and 4863 -- both of them T3's new `residual-growing`. Every
residual sample was inside the bar. A criterion added one commit earlier was
failing a scene that is otherwise flawless.

The criterion was wrong, and the reason is a distinction T3 introduced without
revisiting what consumes it. The four guards that criterion was calibrated
against -- `degenerate-pAp`, `pAp-growth`, `alpha-magnitude`,
`degenerate-oldRZ` -- fire when the iteration is about to produce garbage, so
any of them after establishment decides the verdict. `residual-growing` is the
opposite kind of event: the solve noticed it was walking away from its best
answer and stopped **before** any harm, returning a usable iterate. Whether
that mattered is a question about what it returned, and the residual criterion
measures exactly that, independently.

So `solver_health.mjs` reports these separately and does not count them as
breakdowns.

**Relaxing a criterion is the easiest way to fool yourself, so the only
evidence that justifies it is that the healthy scene passes and the broken one
still fails.** These two supply exactly that pair, in one measurement:

| | before | after |
| --- | --- | --- |
| 14 stable fluids | BROKEN, 2 breakdowns | **HEALTHY**, 2 early stops reported |
| 25 dye injection | BROKEN, 396 breakdowns | **BROKEN**, 175 breakdowns + 221 early stops, on the residual criterion at 1.54e+1 |

## 25-dye-injection is broken, and was broken before any of this

The projection leaves **15.4 times** the divergence it was asked to remove --
against a bar of 1e-2, so three orders of magnitude past it. The shape of the
failure is intermittent rather than terminal:

```
frame    fluid speed     max div     residual/|b|
  500          2.38      3.76e-6         3.96e-6
  600          3.90      4.32e-6         8.18e-6
  700          1.51      8.11e-6         3.33e-5
  800          0.86      2.24e+0         1.54e+1
  900          1.18      2.64e-2         2.27e-2
```

Median residual across the run is 3.33e-5, which is fine; the 90th percentile
is 2.27e-2 and the worst is 1.54e+1. So most frames are healthy and a few come
apart completely -- and at length it is worse than the first thousand frames
suggested: only **1,283 of 12,001** solves converge.

The stop reasons say what is happening: over 12,000 steps, **`pAp-growth`
x175** and `residual-growing` x221. The first is a guard that predates all of this work,
and `linalg.js` documents exactly the failure it catches -- several
individually-reasonable betas compounding `p` geometrically within one solve,
with no denominator ever looking degenerate. That guard firing seventeen times
in a thousand frames is not a threshold being grazed. Nothing was rejected by
the pressure circuit breaker, so the pressure stayed inside its plausible
bounds the whole time; what was left behind was the divergence.

Two things make this tractable rather than just alarming:

- **It is reproducible.** Two 1,000-step runs are identical line for line, so
  it can be bisected. That is a direct return on T2 -- before the seeding was
  made deterministic, a scene that failed on 4% of frames would have failed on
  a different 4% every run.
- **It is intermittent in a specific way.** The frames that fail are few and
  the median is healthy, which points at a condition the scene reaches
  occasionally rather than a systematically wrong operator.

Recorded, not fixed. It is its own investigation and it does not belong folded
into the one that found it.

---

# The 2026-09-15 run (superseded, kept for the record)

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
> **The `converged` column below is not valid.** Two separate reasons, both
> documented in full further down and in
> `perf-investigation-cg-gpu-resident-alpha-beta.md`:
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
> example 26's peak pressure is unchanged at 10.382.
>
> **The re-run this asked for is done: see
> [the 2026-09-26 run](#the-2026-09-26-run-15-scenes-180000-steps-judged-not-counted)
> above.** It covers 15 scenes rather than nine, and it reports a verdict
> per scene instead of this table's counters -- for the reason this box is
> itself an example of, which is that a counter can be green and wrong at
> the same time.

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

  Still true of these nine scenes, and worth reading against the 2026-09-26
  run's finding on `33-flip-dam-break-3d`, where `degenerate-pAp` fires on
  thousands of consecutive frames without anything being wrong -- a scene
  whose liquid comes to rest hands the solver a system that is already
  solved. None of the nine scenes here settles that completely, which is
  why the guards stayed quiet rather than because a firing guard would have
  meant trouble.

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
