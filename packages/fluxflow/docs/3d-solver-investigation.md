# The 3D grid solver: what was wrong, what is fixed, what is not

A record of one investigation, written so the numbers can be checked and the
runs repeated. It follows `project-history.md`'s two conventions: a
measurement nobody can re-run is a claim rather than a result, and a claim
that turned out to be wrong is retracted in place rather than quietly
dropped. Section 6 is nothing but retractions, and three of them are of
statements made during this same investigation.

## 0. Summary

| | before | after |
| --- | --- | --- |
| `examples/35-karman-vortex-street-3d/` pressure solve | 600 iterations, never converged, on every frame | 8-30 iterations on the frames it converges |
| net flux through mid-domain (inflow is fixed at 1152.0) | **-50000** | 1153.9 (sphere), 1154.9 (rod) |
| velocity | all three components pinned at the solver's clamp of 100 | bounded; interior 2.0-3.9 |
| frame rate | ~10 fps | ~30 fps (sphere), ~24 fps (rod) |
| MGPCG on a 48x24x24 Poisson problem with a Dirichlet slab | does not converge at any iteration count | 8 iterations |

Two root causes, both in `src/linalg/multigrid.js`, both of them scope cuts
that the file documented as costing convergence *speed* and that in 3D cost
everything. Both are fixed. Two defects remain and are described in section
5 with the measurements that bound them.

## 1. Where this started

The 3D port -- roughly 7,300 lines: a `*3.js` counterpart to most of
`src/grid/`, five examples, and three changes to shared code -- existed only
as untracked files in a working copy. `examples/35-karman-vortex-street-3d/`
was the scene that did not work. Over a ten-minute headless run the pressure
solve reported not-converged on 6,085 of 6,089 frames, all three velocity
components sat at the solver's own clamp of 100, and net flux through the
middle of the domain read about -50000 against a fixed inflow of 1152.

The scene's own comments at that point argued that this was a materially
harder Poisson problem than the 2D equivalent, having arrived at
`maxIterations: 600` by raising the cap in stages and watching the failure
move further out each time.

## 2. How this was measured

Everything below was run in real Chrome on real WebGPU, headless, with the
anti-occlusion-throttling flags this sandbox needs, and every page checks
that it is on `WebGPUBackend` before reporting anything. A WebGL2 fallback
has a history in this repo of producing wrong answers on repeated
same-buffer dispatch cycles (see `examples/06-multigrid-preconditioner/`),
so a number measured there is not evidence.

The tools are kept at the repo root, each answering one question:

| tool | question |
| --- | --- |
| `karman_verify.mjs` | does the scene survive ten minutes, and what is its flux balance? |
| `diag_history.mjs` | what does *every* frame's solve do, from frame zero? |
| `diag_repeat.mjs` | does it do the same thing on every fresh run? |
| `diag_event.mjs` | what does the field look like at these exact frames? |
| `diag_freeze.mjs` | stop at one frame and re-solve that system every way available |
| `diag_where.mjs` | which part of the domain is large? |
| `karman_shedding.mjs` | is the wake shedding, or is something else periodic? |
| `sandbox/poisson-3d-dirichlet/` | is the operator and preconditioner sound, on a system that is consistent by construction? |
| `sandbox/outflow-gradient/` | which way does an outflow SDF's gradient actually point? |

Two methodological mistakes were made and are worth repeating here because
both produced confident wrong answers:

**Polling a scene from outside cannot see what it does.** With nothing
reading back from it this scene runs at several hundred frames a second, so
a poll every second lands thousands of frames in, and the frames it happens
to sample are not representative. Several early readings of "stable for N
frames" were sampling artefacts. `diag_history.mjs` intercepts the probe's
own assignment before the first frame and records every frame instead.

**A run whose closing readings are bounded is not a stable run.** The
ten-minute run that produced the "after" column of section 0 for the rod was
read from its tail. Its opening 600 frames contained the blow-up described
in section 5.1, recorded in the same log as a count of rejected frames that
the report waved off as a harmless transient.

## 3. Root cause 1: the Dirichlet mask stopped at the finest level

`multigrid.js` evaluated `options.dirichletMask` at level 0 only. Coarser
levels had no concept of it, which the file's own header recorded as an
intentional scope cut whose only cost was convergence speed for a large
Dirichlet region.

`sandbox/poisson-3d-dirichlet/` measures that directly. It builds `b` as
`A @ xStar` from a known `xStar`, so a failure cannot be blamed on whatever
right-hand side a scene happens to produce, and it uses example 35's own
grid shape with a two-cell Dirichlet slab of the kind its outflow creates.
Relative residual after 300 iterations, target 1e-5:

| | no mask | with mask (before) | with mask (after) |
| --- | --- | --- | --- |
| 3D 48x24x24, MGPCG, 4 levels | converged, 6 iterations | 4.2e-2, never | converged, 8 iterations |
| 3D, MGPCG, 3 levels | converged, 7 iterations | 2.5e-3, never | converged, 31 iterations |
| 3D, MGPCG, 2 levels | -- | 2.0e-5, never | -- |
| 3D, unpreconditioned CG | -- | 2.2e-6 | -- |
| 2D 96x48, MGPCG, 4 levels | converged, 6 iterations | converged, 16 iterations | converged, 7 iterations |

A preconditioner beaten by no preconditioner at all, and beaten worse the
more levels it is given, is broken rather than slow. The mechanism follows
from what the coarse levels were solving: a mask-oblivious coarse operator
has zero-flux boundaries everywhere, so nothing in it can represent an error
field that has to vanish at a vent -- and for a pressure solve with an
outflow, that is exactly the whole-domain, low-frequency mode the coarse
levels exist to supply. Every V-cycle handed level 0 a correction pulling
the wrong way and level 0 spent the cycle undoing it. In 2D the same mask
costs 6 iterations against 16, which is why this survived: every shipped
scene that uses a mask is 2D.

Each level now carries its own mask, coarsened from the level above and
refreshed once per solve by the new `refreshCoarseLevels()` rather than from
inside a V-cycle that runs once per CG iteration. Restriction additionally
pins a masked coarse cell's right-hand side to zero, which is what makes the
coarse row hold it there: the coarse problem solves for the error, and the
error at a Dirichlet cell is exactly zero.

Two details are measured rather than reasoned, and both bite:

- **Coarsening is ALL, not ANY.** ANY keeps a thin boundary alive on every
  level and is slightly faster on a synthetic problem whose only mask is a
  slab at the domain edge. It is wrong as soon as the mask has an interior:
  a static collider pins the cells it encloses, and under ANY that blob
  grows a cell of halo per level, pinning the error to zero in real fluid
  right where a wake's pressure varies fastest. On example 35 that turned a
  scene that failed to converge into one that reached 1e7 pressure by frame
  3.
- **Collider bookkeeping is kept out of coarsening,** via the new
  `options.coarseDirichletMask`. A cell pinned only because a collider
  closed every one of its faces is not a boundary condition -- the fine
  operator has already removed it from the problem.

## 4. Root cause 2: the face weights stopped there too

Fixing the mask alone was not enough. The scene then converged on most
frames and still saturated its velocity clamp within a few hundred. A
collider reaches the pressure system only through `faceWeights`, and those
also stopped at level 0, so the preconditioner could not see the obstacle --
and with the mask coarsened but the weights not, the coarse levels were
being told where the boundaries were while still solving an open box.

Each level now averages its parent's face weights, one coarse face from the
2^(d-1) fine faces it covers. Arithmetic averaging, which is what a coarse
cell's flux balance wants from an open-area fraction and is cruder than a
density ratio deserves; both kinds of weight arrive through the same option,
so that tradeoff is noted where a caller can see it.

This needs one guard: coarsened weights can leave a coarse cell with nothing
but zero faces, which is an all-zero row whose diagonal relax would divide
by. Such a cell is pinned by the same mask mechanism, which is why the mask
fields now exist whenever weights do, even for a caller that passed no mask.

Ten minutes of real WebGPU with a sphere obstacle, 18,414 frames: velocity
bounded from about frame 1700, pressure peaking at 0.43, no rejected frames,
nothing non-finite, mid-domain flux 1153.9 against an inflow of 1152.0, and
about three times the frame rate, since a solve that converges in 11-30
iterations replaces one that spent its whole budget every frame.

Regression: 396 unit tests pass; `examples/16-karman-vortex-street/`
converges on every one of 1,367 consecutive frames with maxU 2.89; examples
20, 24 and 28 -- the other `faceWeights` callers -- run clean on WebGPU at
normal frame rates with example 24's liquid/gas interface intact.

## 5. What is still wrong

### 5.1 The rod obstacle blows up at frame 532

`examples/35-karman-vortex-street-3d/` uses a spanwise rod, because a sphere
does not shed a Karman street. Five fresh runs of 900 frames produce
identical numbers -- same converged count, same frames -- so this is
deterministic, not flaky:

```
frame 532   CG breaks down with pAp-growth, 19 times over the next frames
frame 535   the pressure circuit breaker begins rejecting, 15 frames
frame 540   every velocity component sits at the solver's clamp of 100
frame ~560  the solve converges again and the field recovers
```

The trigger is the pressure system, not the iteration cap. Freezing the
scene at frame 524 and re-solving that one system, with `||b||` = 4.665:

| | iterations | relative residual |
| --- | --- | --- |
| MGPCG | 100 | 1.8e-3 |
| MGPCG | 600 | 6.9e-4 |
| MGPCG | 3000 | 1.9e-4 |
| Jacobi-PCG | 3000 | 1.1e-2 |
| CG, no preconditioner | 3000 | 5.5e-2 |
| CG, no preconditioner | 20000 | 2.5e-1 |

Frame 510, fourteen frames earlier, converges in 8 iterations. So the system
becomes roughly three orders of magnitude harder inside fifteen frames, and
multigrid remains much the best of the three: it is the system that went
wrong, not the preconditioner.

Why is not known. The standing suspect -- a hypothesis with no measurement
behind it -- is the rod's sharp edges: `fractionInsideSdf` estimates a
face's open fraction by linear interpolation between two SDF samples, a
box's SDF has a gradient discontinuity along every edge, and a nearly closed
face carries a nearly singular row. `sdf_collider3.js` floors those weights
at 0.01 for exactly that reason, following jet's own `kMinWeight`. The first
falsifiable step is to measure how many faces sit at that floor.

A sphere in the same scene never rejects a frame and never reaches the
clamp: maxU ramps smoothly from 2 to ~18 between frames 800 and 960 and then
holds, with the interior at 2.0-2.5. The bang belongs to this obstacle. Both
obstacles end at the same outflow columns described next.

### 5.2 The outflow cannot empty the domain

At frames 510 and 525 the exit faces carry 1.15 against an inflow of 2.00.
Barely half the mass entering is leaving, the pressure field absorbs the
rest, and by frame 532 that pressure is a smooth global ramp from 48 at the
inlet to 0 at the vent -- the shape of a domain that cannot empty. After the
scene recovers, those two columns sit at ~17.5 while the interior runs at
2.0-3.9, which is where this scene's outflow flux being ~8.7x its inflow
comes from.

The cause is in `grid_outflow_solver3.js`'s convective boundary condition.
It takes its upstream direction from the outflow SDF's gradient,
`upstreamPt = pt - normalize(gradient) * spacing`.
`sandbox/outflow-gradient/` measures that gradient directly: it is
(-1, 0, 0) at every point along the axis, in 3D and in the 2D equivalent
alike, i.e. it points *toward* the fluid. So `pt - n*h` samples downstream
of the face, the two exit faces end up sampling each other, and neither
tracks the interior.

Flipping the sign is not the fix, and this has now been tested with
frame-resolved tooling rather than one late snapshot. With `pt + n*h` the
exit does track the interior correctly -- 2.02 at frame 200 against an
interior 2.0 -- and the solve then stops converging by frame 300 and never
recovers after frame 425, with 273 rejected frames in 900 against 15.
Two fresh runs, identical. That the *more* correct outflow fails sooner
points back at 5.1: the harder the solve has to work, the sooner it hits
whatever degrades at frame 520.

The same defect reflects disturbances back upstream, which a convective
outflow exists precisely to prevent: a vorticity peak of 7.411 at x=47 on
frame 1228 reappears as 7.412 at x=28 on frame 1524, having travelled 19
cells against a flow of 2.2.

## 6. Retractions

Claims made and since withdrawn, each with what replaced it:

- **"A materially harder Poisson problem for CG than 2D, not a coding bug in
  any single file."** It was two coding bugs, sections 3 and 4. The solve
  was not slow, it was not converging at any iteration count.
- **"A box collider cannot be stabilised here; the linear face-fraction
  technique breaks on sharp geometry."** Measured while every frame's solve
  was being cut off mid-iteration. The rod runs for 14,643 frames on the
  fixed solver. The concern about linear face fractions is not retracted --
  it is now the standing hypothesis in 5.1 -- but it is not what made the
  scene diverge.
- **"Outflow flux settles at ~9x inflow; the frames in between are a
  self-consistent but non-physical fixed point."** The 9x is real and is
  explained in 5.2. The fixed-point reading was drawn from runs with a
  broken preconditioner.
- **"`upstreamPt`'s sign is not backwards; flipping it made things worse."**
  The conclusion holds and the reasoning did not: both earlier flip
  experiments were run on a solver that was itself broken. Re-tested in 5.2.
- **"The scene is stable over 14,643 frames."** Made in a commit message
  during this investigation, from a run read at its tail. Section 5.1.
- **"The wake sheds; the structures visibly advect downstream."** Made from
  two screenshots a few hundred frames apart and one scalar. It does shed --
  `karman_shedding.mjs` gives a period of ~270 frames at three stations,
  Strouhal 0.22, and cross-correlation puts the pattern travelling
  downstream at 0.69 and 0.86 of the free stream, which is what rules out
  the reflection in 5.2 -- but every frame of that measurement is after the
  blow-up in 5.1, so it describes the flow this scene recovers into. The
  Strouhal number is also not a calibration: the domain is 8 rod widths long
  and the rod blocks a quarter of the channel, and blockage that severe
  normally raises St.
- **"The outflow artefact stays in two columns and the interior is
  unharmed."** True of the recovered state and false as a description of the
  scene: the same defect is what fills the domain in the first place, 5.2.

## 6a. The register: every root cause, and where each one stands

Kept as a checklist rather than prose, because the point of it is to be
worked through. "Confirmed" means a measurement, named in the row, that
distinguishes this cause from the alternatives; "suspected" means a
hypothesis with a test but no result yet.

### Fixed, on main

| | cause | state | evidence |
| --- | --- | --- | --- |
| A1 | The Dirichlet mask stopped at the finest multigrid level, so every coarse level solved a zero-flux problem and could not represent an error field vanishing at the vent | fixed | Same grid, same operator: no mask converges in 6 iterations, mask never converges at 300, mask after the fix converges in 8. In 2D the same mask costs 6 against 16, which is why it survived |
| A2 | `faceWeights` stopped there too, so the preconditioner could not see the collider at all -- and with the mask coarsened and the weights not, coarse levels knew where the boundaries were while still solving an open box | fixed | Fixing A1 alone left the scene saturating within a few hundred frames; with both, 18,414 frames with mid-domain flux 1153.9 against an inflow of 1152.0 |

### Fixed, on `claude/collider-face-consistency`, deliberately not merged

| | cause | state | evidence |
| --- | --- | --- | --- |
| B1 | The pressure system and the velocity constraint each decided which faces were solid, by different means -- analytic distance per face versus a second sampling of an interpolated SDF grid | fixed | At the rod's own upstream face, x=7: weight 1.000 (fully open), no-flux projection skipped, marker "fluid". 576 faces, with divergence piling up just upstream at 22 against 3.6 for the rest of the domain |
| B2 | A face lying exactly on the surface counted as fully open, since "inside" means `phi < 0` | fixed | Same measurement. Curved colliders hit it on a handful of faces; an axis-aligned box hits it on every face it has |
| B3 | The convective outflow sampled its upstream neighbour downstream, so the two exit faces sampled each other and neither tracked the interior; and its formula added an unbounded increment | fixed | The outflow SDF's gradient measured at (-1,0,0) throughout, in 2D and 3D alike. Exit carried 1.15 against an inflow of 2.00. Replacing it with the convective form took a bare channel from 28 converged frames in 900 to 1778 in 2000 |

### Fixed: C1 and C2 were one defect, and it was the dot product

| | cause | state | evidence |
| --- | --- | --- | --- |
| C1/C2 | `createDotReducer`'s 3D branch nested two `Loop` nodes. TSL names a loop's index by position, so both called their variable `i` and the inner shadowed the outer: the reducer summed cells `(i, t, t)`, i.e. 24 times the diagonal | fixed | A marker written instead of summed over a [48,24,24] grid: 26,496 cells never visited, 1,152 visited more than once, every visited cell with `j === k`. A single cell set to one at (5,0,0) reduced to 24; one at (0,0,1) to 0 |

The anomaly recorded here before -- that a fixed symmetric positive definite
pair cannot converge in 15 iterations on one frame and make no progress
fifteen frames later -- was the right thing to hold on to. The pair was
fixed and was symmetric and positive definite. What was not fixed were the
scalars the iteration is steered by: alpha and beta are quotients of two dot
products, and the reducer was off by 1.4% on the total and 31% on one lane
against the same sum in double precision.

That also explains why nothing caught it. CG with a wrong step size is still
a descent method, so easy problems converge anyway and look fine, and the
one direct test anyone writes -- a field of ones -- reduces to exactly the
right answer however wrongly its cells are chosen, because any 576 ones sum
to 576. `examples/31-conjugate-gradient-3d/` passed throughout the whole
period the defect existed. It now carries the check that does see it: single
cells, and a ramp whose total only the correct traversal produces, verified
to fail when the defect is put back.

After the fix, measured on the same three things in turn:

| | before | after |
| --- | --- | --- |
| the reducer against double precision | 1.4e-2 | 7.8e-7 |
| `sandbox/stalled-system/` frame 112 | 3000 iterations, 8.0e-4, never converged | 23 iterations, 7.9e-7 |
| `examples/35-karman-vortex-street-3d/` | BROKEN at frame 115 | HEALTHY over 6001 frames |

The scene now converges on 6000 of 6001 frames with zero circuit-breaker
rejections and zero CG breakdowns, its worst interior flux deviation is
2.2%, and 0.994 of what enters leaves at the outlet. Three fresh runs give
identical numbers. 1D and 2D were never affected -- single `Loop` each --
and that is confirmed rather than assumed: example 16 runs 1368 frames with
every frame converged at maxU 2.89, unchanged, and every 2D row of
`sandbox/poisson-3d-dirichlet/` matches to the iteration.

### What this defect invalidates

Every 3D measurement in this document older than the fix was taken through
it. Some conclusions survive because they rest on geometry or on code
reading rather than on convergence; others do not and are listed here rather
than left standing:

| finding | standing |
| --- | --- |
| A1, the Dirichlet mask not reaching coarse levels | Holds. Measured against a synthetic system built as `A @ xStar`, and the comparison was between preconditioners on the same broken reducer, which the defect scales but does not reorder |
| A2, the face weights not reaching coarse levels | The fix is sound on its own terms, since jet rebuilds its system per level, but its measured justification -- "fixing A1 alone left the scene saturating" -- is a scene measurement through the defect. Worth re-measuring whether it is needed |
| B1 and B2, the two halves of the collider disagreeing | Holds. The evidence is a direct reading of what each half computes for a given face, not a convergence result |
| B3, the outflow's upstream direction | The gradient measurement holds. The convergence figures quoted for it (28 frames in 900 against 1778 in 2000) were taken through the defect and mean less than they appeared to |
| "The vent is a mass sink by construction, so in need not equal out" | **Wrong.** With the reducer fixed, 0.994 of the inflow leaves at the outlet. That reasoning was built to explain a symptom of this defect |
| "Each correctness fix made the scene fail sooner" | Accounted for. Every one of those runs was steered by wrong step sizes; the ordering between them says nothing |
| The Strouhal number of 0.22, and the shedding measurement behind it | Taken after the frame-532 blow-up and through the defect. Needs re-measuring before it is quoted |
| The ruled-out list below | Still ruled out, but for a reason now visible: those experiments all ran through the defect, which is why none of them helped |

### Known, lower priority

| | item |
| --- | --- |
| E1 | `grid_outflow_solver2.js` has B3's upstream-direction error unfixed. Examples 15 and 16 are long-run stable as they stand, which may be luck |
| E2 | Restriction and prolongation are unweighted transfer operators while the operator is now weighted per level. Whether that costs anything is untested |
| E3 | `refreshCoarseLevels` must be dispatched or the coarse weight fields stay zero and the coarse operators are degenerate. The claim that not dispatching it reproduces the old behaviour is true of the mask and false of the weights |
| E4 | This scene has no CFL-based adaptive time step while its 2D counterpart does. A safety net, not a cause |

### Ruled out, with the measurement that ruled it out

| ruled out | by |
| --- | --- |
| The CG fast-path switches (GPU-resident scalars and setup, batched iterations, the GPU stop test, the fused V-cycle) | All off: same failure, same frames |
| The iteration cap | 600 and 3000 do not help, and the misses are by three to four orders of magnitude |
| The multigrid level count | `numberOfLevels: 1`, which makes the preconditioner plain SOR and unquestionably valid, fails on the same frame |
| Face-weight conditioning | Zero faces sit at the 0.01 floor; the weakest non-enclosed row is a factor of 6 off an interior one |
| The scene's whole-domain push force | `pushStrength: 0` fails the same way |
| Flipping the outflow's upstream sign on its own | Measured twice, frame-resolved the second time: worse, and it never recovers |
| Sharp geometry breaking linear face fractions, as the cause of divergence | The weights are fine (above). It was a disagreement between two tests, B1, not a discretisation error |

## 6b. The plan, in the order that settles the most

Steps 1 to 4 are done: the fixture exists, the trace was run, and what it
found was the dot product above rather than any of the three suspects the
plan listed. Steps 5 and 6 are what remains, with re-measurement added.

1. ~~**Make the failure a seconds-long offline test.**~~ Reproducing C1 currently
   costs a few hundred frames of simulation. Export the failing frame's
   system -- the masks, the face weights, the right-hand side and the
   initial pressure -- and rebuild it in a sandbox page. Everything after
   this step is cheap; without it, everything after this step costs minutes
   per attempt.
2. **Probe the V-cycle with the real residual, not random vectors.** Record
   `r . z` and `||z|| / ||r||` per iteration on that fixed system. A sign
   that wanders, or a `z` that collapses, localises the defect; bisecting
   over `numberOfLevels` then says which level introduces it.
3. **Test the three specific suspects on the same fixed system**, each
   switchable on its own: the coarse enclosed-cell guard pinning cells whose
   fine children are fluid; unweighted transfer against a weighted operator
   (E2); and the coarsest level's 30 SOR sweeps on a weighted operator.
4. **Verify against the scene**, with `solver_health.mjs` and several fresh
   runs, reading from frame zero.
5. **Re-measure what the defect touched**, in the order the table above
   puts it: whether A2 is needed at all, what B3 is actually worth, and the
   shedding figures.
6. **Merge, then 2D.** Bring the branch to main, then port B3 to
   `grid_outflow_solver2.js` with its own verification of examples 15 and 16
   rather than riding along with this.
7. **The rest**: E3, E4, and the documents.

## 6d. The re-measurements, and what they changed

Everything the dot product defect touched, re-taken against the fixed code.

| | re-measured | result |
| --- | --- | --- |
| A1, mask coarsening | Both captures in `sandbox/stalled-system/`, four-level cycle | Worth about 40%: 17 iterations without it, 11 with. It earns its place |
| A2, face-weight coarsening | Same sweep, plus the scene with it off | Worth roughly one iteration in eleven on top of A1, and nothing at all on its own. The scene reads HEALTHY without it, so it is not load-bearing. Kept on -- one dispatch per solve for a tenth of the iterations -- but the claim attached to it is now this one |
| B3, the convective outflow | The previous formula put back, judged by `solver_health.mjs` | Load-bearing. The old formula converges on every one of 2000 frames with no rejections and is BROKEN from frame 1540 all the same: 8.67x the inflow leaving the outlet. The 8.67 this investigation started from belonged to the boundary condition |
| The Strouhal number | 11,997 frames on a scene with no blow-up in it | Survives: period 267 frames at three stations, St 0.225, pattern travelling downstream at 0.69 to 0.90 of the free stream, and the aliasing in the old lag measurement is gone |

Two things came out of it that were not on the list.

**The fixture had been rebuilding a one-level preconditioner while calling
it four-level.** `export_system.mjs` captured `settings.multigrid`, which
holds the cycle's runtime switches and not its shape. The options are now
exposed and captured beside them. Nothing about the dot product conclusion
depended on it -- that was measured on the reducer directly -- but the first
coarsening sweep it produced was meaningless and said so by returning four
identical numbers.

**2D was broken too, and had been all along.** Porting the outflow to
`grid_outflow_solver2.js` needed a 2D verdict, which needed the probe to
read 2D scenes, which is when example 16 turned out to read BROKEN from
frame 1720 on the formula it had shipped with: 13.89x the inflow leaving,
peak speed 27.79 against an inflow of 2, interior flux 1289% off -- while
converging on every frame, which is why none of its long runs ever showed
it. On the ported formula it reads HEALTHY over 4001 frames at 1.013x.

### F1 and F2: one mechanism, and it is structural

Measured per boundary face, which is what made it legible:

| scene | in | out | closed walls | out of balance |
| --- | --- | --- | --- | --- |
| example 16 | 256.00 | 259.28 | 0.00 | 1.3% |
| example 15 | 128.00 | 156.56 | 0.00 | 22% |
| example 17 | none | 532.23 | 0.00 exactly, all three | 100% |

The closed walls carry exactly zero, so the boundary conditions there are
doing their job. Everything else leaves through the vent and nothing
replaces it.

The mechanism is the same in all of them and it is not the defect the
convective outflow fixed -- every number above is with that already in
place. A vent is a region of pinned pressure, a pinned row is an identity
row, and nothing in the pressure solve constrains the flux into one. The
convective boundary condition then writes the LOCAL interior velocity onto
the exit faces, and that velocity is under no obligation to conserve mass.
Example 15's cylinder blocks more of its channel than example 16's does, so
its flow is still accelerated where it reaches the exit and carries 22% more
out than comes in; example 16's is nearly recovered by then and carries
1.3% more. Examples 17, 18 and 19 have no inlet at all, so anything leaving
is unbalanced by construction: a domain whose only opening is a vent can
pump indefinitely.

The standard remedy is a global mass-flux correction: after the convective
update, scale the outflow faces so that net flux across the boundary is
zero. It is one reduction and one scaling pass per frame, it is textbook for
convective outlets, and in the no-inlet case it reduces to exactly "make the
net zero", so it addresses all five scenes at once. Not implemented -- it
changes a boundary condition under every outflow scene in the package, which
wants deciding rather than assuming.

| | item |
| --- | --- |
| F1/F2 | The global mass-flux correction above, or a decision not to have one |
| E1 | Closed: the 2D outflow is ported |
| E2, E3, E4 | Unchanged |

## 7. Re-running any of this

```bash
npm install
npm test                                  # 396 tests
npm run dev -w fluxflow                   # vite; note the port it prints
```

Then, from the repo root, against whatever port that is (the tools are
written for a port already serving `packages/fluxflow`):

```bash
node diag_repeat.mjs 5 900       # five fresh runs, per-frame, section 5.1
node diag_freeze.mjs 524         # freeze one frame and re-solve it, 5.1
node diag_event.mjs 510,525,532,536,560
node karman_shedding.mjs 240 3   # period, Strouhal, propagation direction
node karman_verify.mjs           # the ten-minute run; read it from the START
```

The sandbox pages are ordinary examples: open
`/sandbox/poisson-3d-dirichlet/` and `/sandbox/outflow-gradient/` in a
WebGPU browser.

## 8. Open, in the order that settles the most

1. Measure the distribution of collider face weights for the rod -- how many
   faces sit at the 0.01 floor, and where. This is the first falsifiable
   step on 5.1's hypothesis, needs no new machinery, and either implicates
   the sharp edges or clears them.
2. Whatever 5.1 turns out to be, it is upstream of 5.2: the corrected
   outflow makes the solve fail sooner, not later.
3. The convective outflow needs deriving for this port's own SDF sign
   convention rather than transcribing a formula whose normal points the
   other way. Neither of the two signs available is right.
4. Restriction and prolongation are still unweighted transfer operators
   while the operator itself is now weighted per level. Whether that costs
   anything measurable is untested.
