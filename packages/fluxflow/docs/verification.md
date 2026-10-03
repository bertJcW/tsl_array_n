# How this package decides whether a solver works

`project-history.md`'s [Testing](project-history.md#testing) section
describes three layers: structural tests, self-checking example pages, and
long runs on real hardware. That was the state of the method before the 3D
work, and it was not enough. It let through a defect in which every 3D dot
product summed a diagonal twenty-four times over -- while the test suite
stayed green, the example page whose whole job is 3D linear algebra kept
printing a tick, and the long runs reported bounded velocities and
convergence on most frames. `3d-solver-investigation.md` is the account of
finding it. This file is the method that replaced the one that missed it.

Five layers, each answering a different question, and three rules the layers
are worth nothing without.

---

## The three rules

**0. A check that cannot tell "finished and fine" from "did not finish" is
not a check.** Learned three times, each time from a harness rather than from
the code under test. A 14,643-frame stability claim came from reading a log
from its tail, where the blow-up in the first 600 frames had scrolled past. A
determinism sweep reported nine scenes reproducible including one that had
never started, because it had no probe, timed out twice, and two
identically-failed logs diff clean. And `run_page.mjs` waited a fixed fifteen
seconds for a page that takes thirty-four, which cut a symmetry sweep off at
three of its sixteen rows -- a truncated list of ticks that reads exactly like
a complete one, and that a written conclusion was then reasoned from.

So: a comparison-based check confirms both sides produced something before
comparing them; a page-driving check waits for the page to stop rather than for
a clock, and says PARTIAL when it gives up; and a self-checking page prints its
own summary last, so that a missing summary is itself the signal.

**1. A criterion nobody has seen fail is not a criterion.** Every check
described here has been run against a build with a known defect deliberately
put back, and kept only if it fired. `examples/31-conjugate-gradient-3d/` is
the cautionary case: it exists to answer "is the linear algebra sound in
3D", and it passed on every run for the entire life of a defect that made
every 3D dot product wrong. Its new dot-reducer check was verified the other
way round -- reintroduce the defect and four of its five probes read 0
instead of 1, and its ramp summed 4560 instead of 7140, while the
convergence test beside it still passed.

**2. Thresholds are calibrated on both sides, not chosen.** The residual
bar in `solver_health.mjs` is 1e-2 because across the three 3D scenes the
healthy median runs 4.5e-7 to 4.7e-6 with the worst single sample at 1.6e-3,
and the same scenes with the dot product defect put back run medians of
5.0e-3 and 5.9 with worst samples of 1.3 and 7.4. Both sets of numbers are
in the file, beside the constant. A threshold with only one side measured is
a guess wearing a number.

**3. A check that does not apply says so, out loud.** Weak passes and
misapplied criteria are the failure mode of a probe, and this one has been
corrected five times -- each time because it had produced a confident wrong
verdict. It divided by an inflow of zero on buoyancy scenes and reported
`Infinity`. It invented a velocity scale of 1 and called a plume broken for
reaching 20. It measured conservation across the grid's outermost faces,
which lie inside a region nothing constrains, and declared three healthy
scenes to be losing all their mass. It applied a fixed-region criterion to a
free surface, whose region moves by definition. And it counted a CG guard
stopping an already-solved system as a breakdown, which flagged
`examples/33-flip-dam-break-3d/` as BROKEN over 7,557 consecutive frames on
which nothing whatever was wrong -- the scene's liquid had come to rest, so
the residual was exactly 0, so the search direction was 0, so the guard fired
every frame. `linalg.js` already declined to call that a breakdown at one of
the seven places it sets a stop reason; all seven now share one test, and
`examples/05-preconditioned-conjugate-gradient/` is what proves a real guard
still fires on all three code paths.

Every one of those looked convincing at the time. Note what none of them was
fixed by: widening a threshold, or adding an exception for the scene that
tripped it. The probe now names the criteria it is *not* applying to a scene
rather than quietly counting them as passes, and `long-run-stability.md`
records each correction with the measurement that forced it.

---

## Layer 1 -- structural tests, no GPU

`npm test`: 398 tests in 30 files (375 in `fluxflow`, 23 in `tsl_array_n`) at the
time of writing -- the count drifts, since one of the checks is a lint-style test
per source file and two of these numbers had already gone stale within days of
being written,
Vitest under Node, seconds to run. They cover API shapes, construction-time
validation, and the arithmetic that can be done on the host, plus
lint-style tests such as the one forbidding `x != x`.

`solve()` needs a GPU, so nothing about numerical behaviour is tested here.
This layer was green throughout the dot product defect and throughout every
boundary-condition defect in the investigation. Read a green run as "nothing
is obviously malformed", never as evidence that a solver works.

## Layer 2 -- in-page self-checks, real GPU, known answers

Pages that compute something whose answer is known independently, and print
a tick or a cross per claim. The pre-existing ones are tabulated in
`project-history.md`; these are the ones the 3D work added or leaned on.

| page | what it establishes |
| --- | --- |
| `examples/31-conjugate-gradient-3d/` | MGPCG against an analytic 3D Poisson solution; restriction and prolongation are an adjoint pair; **the dot reducer visits every cell exactly once** |
| `examples/32-grid-solver-3d/` | `createGridSolver3` leaves the field divergence-free; `createGridFlipSolver3`'s particles stay finite and in bounds |
| `sandbox/poisson-3d-dirichlet/` | operator symmetry with and without a Dirichlet mask; V-cycle symmetry and sign-definiteness at 1, 2, 3 and 4 levels; convergence on a system built as `A @ xStar`, so no scene's right-hand side can be blamed for a failure. Its four permanent failures are marked as expected, with the reason inline, so the summary is a number that reads zero when nothing is wrong |
| `sandbox/outflow-gradient/` | which way an outflow SDF's gradient actually points, measured, in 2D and in 3D |
| `sandbox/prefix-sum/` | the GPU prefix sum the resamplers rank donors with: correct against a JS reference on the lengths that break scans, bit-identical across eight runs, and ranks that increase in the index rather than merely being distinct |
| `sandbox/vcycle-floor/` | how far down the V-cycle drives a residual on a clean system of a scene's own shape, across levels and smoothing sweeps. Built to ask what a scene cannot: it found PCG's host path ignoring `relativeTolerance` |

The dot-reducer check is the one whose shape is worth copying. A field of
ones reduces to the right total however wrongly its cells are chosen -- any
120 ones sum to 120 -- so the obvious test cannot see an index-mapping bug
at all. Single cells can: with the defect present, four of five single-cell
deltas reduced to 0 in their own lane. The probe shape is `6x4x5`,
deliberately not a cube, so that an axis swap shows up too. Pick the input
that distinguishes the failure, not the input whose answer is easiest to
predict.

## Layer 3 -- the verdict, `solver_health.mjs`

One command, one verdict, exit code 0 for healthy and 2 for broken. It runs
a scene in real Chrome, records from frame zero by intercepting the probe's
own assignment before the first frame renders, and judges conservation laws
rather than appearance.

```bash
node solver_health.mjs <url> [frames] [sampleEvery]
node long_run.mjs                    # every drivable scene, 12,000 steps each
```

`long_run.mjs` is the whole-suite form: it runs each scene through the probe in
turn, writes a per-scene log, and prints one table.
`long-run-stability.md` holds the most recent results.

Every scene with a solver to drive now carries a probe. The last three to get
one had been listed as open coverage gaps, and the first run of them found
`examples/25-dye-injection/` leaving 15x the divergence its projection was
asked to remove -- a scene that had looked fine for as long as nothing could
drive it. A gap in coverage is not a neutral state.

It replaced a detector that watched one scalar -- maximum velocity against a
fixed 90 -- which cannot decide the question from either side: the solver's
circuit breaker clamps at 100, so a field that has completely come apart
still reads finite, and a legitimately fast flow can approach the same
number while being fine. Example 35 settled after its blow-up into a state
whose maximum was 17.5 and whose outflow carried 8.7x the mass its inflow
admitted, and that 17.5 was reported as stability.

| criterion | applies when |
| --- | --- |
| nothing non-finite | always |
| the projection left under 1e-2 of the divergence it was asked to remove | always -- it needs no geometry, so free surfaces and vent-only domains are covered by it |
| the tolerance asked for is above the floor the solve can actually reach | reported, never judged -- `settings.reportNoiseFloor`, which `solver_health.mjs` turns on. It is the difference between "the budget was too small" and "no iteration count reaches this". Its formula estimates the wrong quantity and overshoots it by about ninety; it discriminates correctly in practice by tracking the real floor coincidentally, so read it as an order of magnitude. `long-run-stability.md` has the measurement and what the sound version would be |
| no CG breakdowns after the scene established itself -- the four corruption guards only, not counting stops that left a residual of exactly 0 nor stops on a growing residual, both of which are reported instead | always |
| weighted flux equal across interior cross-sections, within 5% | the solver offers collider face weights, so solid faces can be excluded (the 2D solver has none, and reports instead of judging) |
| net flux across the **solved region's** boundary in balance, on a trailing 20-sample average | the scene has an inlet |
| fluid speed within 6x the inflow's own | the scene has an inlet |

Every threshold is a multiple of something the scene itself defines, so the
same probe works on any scene built from `grid_solver3` without retuning --
which is the point, since a per-scene constant is not a criterion. Three
details are easy to get wrong and were got wrong first:

- **It measures across the boundary of the solved region, not of the grid.**
  A vent is a region of pinned pressure; the solver does not make those
  cells divergence-free and does not correct the velocities inside them, and
  mantaflow says the same thing in one line of `pressure.cpp` -- do not
  touch velocities in outflow cells. The grid's own outermost faces
  therefore carry whatever the outflow condition last wrote, constrained by
  nothing. On `examples/17-smoke-fire/` the grid's top face reads 840.9
  where the fluid three rows inside reads 14.0; on example 19 the same pair
  reads 93.9 against -0.03. What is summed instead is every face with fluid
  on exactly one side, which is the fluid region's own surface whatever
  shape the vent takes.
- **It detects a solved region that moves.** A free surface is exactly that,
  so net flux across its boundary is not zero -- it is what moves the
  surface. The probe re-reads the mask each sample, and when the region has
  changed it puts the flux criteria out of scope and says so.
- **It reports the solver's own counters without judging them.**
  Convergence counts are useless as a criterion: a bare channel converges on
  28 frames out of 900 while being perfectly healthy, a broken scene
  converged on 73% of its frames, and `examples/17-smoke-fire/`,
  `18-explosion` and `19-fuel-fire` once converged on **0 of 12,001** frames
  while leaving a median relative residual of 2e-5 to 5e-5 against a bar of
  1e-2. Those three were asking for a tolerance beneath what float32 can verify
  a residual against -- not a weakness of the preconditioner, as an earlier
  version of this line said, but the cancellation in recomputing `b - A@x` in
  single precision. They have measured tolerances now and converge on
  essentially every frame; `long-run-stability.md`'s T3 has the derivation and
  the numbers. Velocity on fully closed collider faces is
  the same -- `constrainVelocity` extrapolates into the solid on purpose, so
  a nonzero reading there is the design. Both are evidence to explain,
  printed beside the verdict.

## Setting a scene's tolerance

A scene's pressure tolerance is a stated accuracy requirement, not a constant
tuned to make the solver behave -- the distinction the no-magic-numbers rule
turns on, in `examples/16-karman-vortex-street/`'s own words. It is measured, and
the procedure is:

1. Run the scene through `solver_health.mjs`, which switches on
   `settings.reportNoiseFloor`. Read the floor's distribution and
   `residualOverFloor`. That says which order of magnitude to try.
2. Try candidates **over 12,000 steps, never a short run.** Example 17 converges
   on 600 of its first 600 frames at 1e-5 and on 4,667 of 12,001; example 19
   reads 65% short and 3% long. Scenes get harder as they develop.
3. Take the tightest candidate that converges on essentially every frame, and
   check `residualOverFloor` is near 1 -- which says the solve is taking what the
   arithmetic allows rather than stopping early with accuracy unspent.
4. For a scene with no inlet the verdict is a narrow pass, so read the sampled
   speed and divergence from first frame to last as well, and compare against the
   old tolerance rather than explaining a trend away.
5. Write the measured numbers in the comment beside the tolerance, so the choice
   is checkable instead of folkloric.

The floor estimate errs in both directions -- optimistic by 2x on one scene,
pessimistic enough on another to warn on 94% of frames that nonetheless converge
-- so treat it as the hint that saves the bisection, not as the answer.

## Layer 4 -- offline fixtures

Reproducing a failure inside a running simulation costs minutes per attempt,
which is the difference between testing a hypothesis and guessing at one.
`export_system.mjs` captures the pressure system from a chosen frame -- the
grid, the Dirichlet and vent masks, the collider's face weights, the
right-hand side and the pressure -- and `sandbox/stalled-system/` rebuilds
it and hands it to the same CG in seconds.

Two frames are captured on purpose, two apart in the same run: one the
solver handled and one it broke on. A fixture that only reproduces the
failure has not shown that it reproduces anything.

What that page can then do:

- `?trace=N` runs PCG by hand in JavaScript against the library's own GPU
  operators, so every scalar the iteration turns on is visible rather than
  inferred. That is what showed a hand-rolled iteration reaching in sixteen
  steps what the library took 3000 to approach -- which moved the suspicion
  off the preconditioner and onto the dot product, where it belonged.
- `?dot=1` weighs the reducer against double precision, and tests
  visitation directly.
- `?coarsen=` sweeps what the coarse levels are told.
- `?norefresh=1` skips a dispatch the API says a caller may skip, and checks
  that skipping it still leaves a usable preconditioner.

One warning from experience: a fixture is only as honest as what it
captures. This one once rebuilt a single-level cycle while labelling itself
four-level, because the exporter had recorded the runtime switches rather
than the cycle shape. It was caught only because a coarsening sweep returned
four identical numbers.

## Layer 5 -- diagnostics, once something has failed

Reached for after a verdict, not before one. Each answers one question.

| tool | question |
| --- | --- |
| `diag_history.mjs` | what did **every** frame's solve do, from frame zero? |
| `diag_repeat.mjs` | does a fresh run do the same thing -- deterministic, or lucky? Every FLIP scene once answered no, because the particle seeding came from `Math.random()`; they are reproducible now and a fresh run is a real check again |
| `diag_event.mjs` | what did the field look like at exactly these frames? |
| `diag_freeze.mjs` | stop at one frame and re-solve that system every way available |
| `diag_where.mjs` | which part of the domain is the large one? |
| `diag_iterations.mjs` | is this solve slow, stalled, or diverging? One frozen frame, re-solved at three budgets against every preconditioner |
| `diag_symmetry.mjs` | is this scene's own V-cycle symmetric, and does (Mr,r) keep one sign? Measured on its frozen system rather than inferred from a residual curve |
| `diag_control_volume.mjs` | which surface should conservation be measured across? |
| `diag_weights.mjs`, `diag_face.mjs` | what does each half of the collider think a face is? (CPU, no browser) |
| `karman_shedding.mjs` | is the wake shedding -- period, Strouhal number, and which way the pattern travels |
| `karman_verify.mjs` | a ten-minute run with a flux balance |
| `export_system2.mjs` + `cpu_reference.mjs` | take a 2D scene's pressure system off the GPU entirely and solve it in Node, in double and in three flavours of single precision. It proves its own operator first - symmetry, and the norm of `b - A@pressure` against the number the GPU reported - because a reimplementation that merely looks right measures itself rather than the machine |
| `run_page.mjs` | load a page and print what it says, waiting for it to finish rather than for a clock -- and saying PARTIAL rather than printing a truncated run as if it were whole |

`diag_history.mjs` exists because polling a scene from outside cannot see
what it does. With nothing reading back from it, a 48x24x24 scene runs at
several hundred frames a second, so a poll that feels prompt lands thousands
of frames in; several early "stable for N frames" readings were sampling
artefacts of exactly that.

---

## Two environment facts that invalidate everything if forgotten

Both are also in `project-history.md`, and both still hold.

**Real Chrome, real WebGPU.** Every tool here launches Chrome with the
anti-occlusion-throttling flags and asserts `WebGPUBackend` before believing
a number. The in-app browser falls back to WebGL2, which has a documented
history in this repo of giving wrong answers on repeated same-buffer
dispatch cycles -- and therefore of producing false bug reports. A
backgrounded tab throttles rAF to roughly a frame every few seconds, so any
multi-hundred-frame run is driven through `window.__fluxflowProbe` rather
than by the page's own loop.

**Read a run from frame zero, not from its tail.** A run whose closing
readings are bounded is not a stable run.
`examples/35-karman-vortex-street-3d/` was reported stable over 14,643
frames on the strength of a ten-minute log whose first 600 frames contained
the blow-up -- recorded in the same log, as a count of rejected frames that
the report waved off as a startup transient. Any nonzero count of
circuit-breaker rejections or CG breakdowns is a thing to explain before
anything else in that run is quoted.
