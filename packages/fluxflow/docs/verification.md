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
the five places it sets a stop reason; the probe now declines at all of them.

Every one of those looked convincing at the time. Note what none of them was
fixed by: widening a threshold, or adding an exception for the scene that
tripped it. The probe now names the criteria it is *not* applying to a scene
rather than quietly counting them as passes, and `long-run-stability.md`
records each correction with the measurement that forced it.

---

## Layer 1 -- structural tests, no GPU

`npm test`: 396 tests in 30 files (373 in `fluxflow`, 23 in `tsl_array_n`),
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
| `sandbox/poisson-3d-dirichlet/` | operator symmetry with and without a Dirichlet mask; V-cycle symmetry and positive-definiteness; convergence on a system built as `A @ xStar`, so no scene's right-hand side can be blamed for a failure |
| `sandbox/outflow-gradient/` | which way an outflow SDF's gradient actually points, measured, in 2D and in 3D |

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

`long_run.mjs` is the whole-suite form: it runs each scene through the probe
in turn, writes a per-scene log, and prints one table.
`long-run-stability.md` holds the most recent results.

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
| no CG breakdowns after the scene established itself, not counting stops that left a residual of exactly 0 | always |
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
  `18-explosion` and `19-fuel-fire` converge on **0 of 12,001** frames while
  leaving a median relative residual of 2e-5 to 5e-5 against a bar of 1e-2. Velocity on fully closed collider faces is
  the same -- `constrainVelocity` extrapolates into the solid on purpose, so
  a nonzero reading there is the design. Both are evidence to explain,
  printed beside the verdict.

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
| `diag_repeat.mjs` | does a fresh run do the same thing -- deterministic, or lucky? (`33-flip-dam-break-3d` does not: four runs of one build gave BROKEN, HEALTHY, BROKEN, HEALTHY) |
| `diag_event.mjs` | what did the field look like at exactly these frames? |
| `diag_freeze.mjs` | stop at one frame and re-solve that system every way available |
| `diag_where.mjs` | which part of the domain is the large one? |
| `diag_control_volume.mjs` | which surface should conservation be measured across? |
| `diag_weights.mjs`, `diag_face.mjs` | what does each half of the collider think a face is? (CPU, no browser) |
| `karman_shedding.mjs` | is the wake shedding -- period, Strouhal number, and which way the pattern travels |
| `karman_verify.mjs` | a ten-minute run with a flux balance |
| `run_page.mjs` | load a page and print what it says |

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
