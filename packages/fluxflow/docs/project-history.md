# fluxflow: research, debugging, performance and testing

An index and a narrative for work that is otherwise spread across 34 commit
messages, three research documents and a lot of source comments. The commit
messages remain the authority on any single change; this exists so the
*arc* is legible -- what was tried, what was measured, what was wrong, and
what the wrong turns taught.

Two conventions used throughout, worth stating once:

- **A measurement nobody can re-run is a claim, not a result.** Several
  entries below exist only because a throwaway measurement was later
  re-run and disagreed. Where a number appears here, the thing that
  produced it is in the repository.
- **Retractions stay visible.** Where a conclusion was wrong, it is marked
  wrong in place rather than edited away, because the reasoning that
  produced it usually recurs.

## Contents

- [The package](#the-package)
- [Research](#research)
- [Debugging](#debugging)
- [Performance](#performance)
- [Testing](#testing)
- [Methodology, and what went wrong with it](#methodology-and-what-went-wrong-with-it)
- [Open items](#open-items)

---

## The package

fluxflow is a GPU fluid simulation package built on `tsl_array_n`, which
wraps three.js's TSL as a Taichi-style compute-kernel API. Apache-2.0.
FLIP/PIC particle-grid simulation on a staggered MAC grid, with pressure
projection by multigrid-preconditioned conjugate gradient.

Provenance matters here and is tracked in `THIRD-PARTY-NOTICES.md`:

| Source | Licence | How it is used |
| --- | --- | --- |
| jet / fluid-engine-dev | MIT | Structure and algorithms ported; cited at each site |
| mantaflow | Apache-2.0 | Multigrid smoother ordering, compared against directly |
| OpenFOAM | **GPL-3.0** | **Methods only.** No source read into code, no transliteration |
| Houdini | proprietary | Architecture reference only |

The OpenFOAM line is a hard constraint, not a formality: GPL-3.0 and
Apache-2.0 are incompatible, so `docs/openfoam-two-phase-flow.md` contains
no OpenFOAM source excerpt at all. Every mechanism there is restated as
mathematics and in this project's own prose, and the citation that lands in
the code is the original paper -- Rusche 2002, Rhie & Chow 1983, Zalesak
1979, Brackbill/Kothe/Zemach 1992, Rudman 1998 -- not OpenFOAM.

---

## Research

### Two-phase bubbles (`docs/two-phase-bubbles-research.md`)

The first pass, before the two-phase solver existed. Archived rather than
extended; its main durable outcome was recording Houdini as an architecture
reference.

### OpenFOAM, first reading (`docs/openfoam-two-phase-flow.md`)

How an industrial CFD code arranges two-phase solving, read for what could
be borrowed. Produced a seven-item work list. The items subsequently built:

1. **Reduced pressure** `p = p_rgh + rho (g.x)` (Rusche 2002). Take the
   hydrostatic part out analytically instead of making the solver
   rediscover it every frame.
2. **Well-balanced face-assembled gravity**, so the discrete rest state
   cancels face by face rather than only in the limit.
3. **Momentum-consistent transport** (Rudman 1998) -- which is where the
   mass-weighted P2G experiment came from.

Items examined and *not* built, with reasons recorded in the doc: MULES /
Zalesak flux limiting and interface compression, which address a
volume-fraction advection scheme this port does not use -- particles carry
the interface here.

### OpenFOAM, second reading

A deliberate second pass on the two-phase specifics after the first round
of fixes had landed. Produced the mass-weighted P2G idea, which was built,
measured, and rejected (below).

### mantaflow's multigrid

Read while chasing the long-run divergence bug with a Dirichlet mask.
`GridMg::doVCycle` calls its smoother with a `reversedOrder` flag -- `false`
on the way down, `true` on the way up -- specifically to keep the V-cycle a
symmetric operator. This port's `relax()` had no such distinction. That
comparison is what found the bug; see Debugging #7.

---

## Debugging

Roughly chronological. Every one of these was found by running on real
hardware; none was caught by the structural test suite, and that is the
recurring theme.

### 1. Three bugs in the two-phase solver, all invisible to green tests

**Twelve storage buffers in one compute stage.** The guaranteed WebGPU
limit is 8. Pipeline creation was rejected outright and the kernel silently
did nothing, every frame. Plenty of real hardware raises the limit, which
is what makes it dangerous: *a portability bug a good GPU hides.* Fixed by
packing four atomic cursors into one array, two donor pools into one array
filled from both ends, and splitting eligibility into its own kernel --
12 bindings down to 7.

**`select()` over an `atomicLoad()` result generates invalid code.** This
produced "THREE.TSL: Invalid generated code, expected a int". Isolated with
a standalone probe rather than guessed at, because the construct looks
entirely ordinary. The fault is narrow and worth not over-generalising: an
atomic result in a *comparison* is fine, and an atomicAdd result as an
*index* is fine. Only `select()` over one is broken. Resolved with a real
`If`/`Else`.

**The pressure solve blew up at any real density ratio.** Bisecting on the
ratio was decisive: 1:1 converged, 1:100 went 1023/1024 cells non-finite
within ten frames. Both causes were configuration rather than formulation
-- the fixed-point `atomicScale` (see #5, which later removed the concept
entirely) and a multigrid depth of 1, which is plain relaxation with no
coarse-grid correction. Single-phase scenes get away with that; a
variable-coefficient system does not.

### 2. Every NaN guard in the library was a no-op

All six safety nets detected NaN with `x != x`, the documented WGSL
replacement for the `isnan()` that core WGSL does not have. **On a real
device that expression returns false for a NaN** -- WGSL permits an
implementation to assume non-finite values never occur and fold
accordingly.

The expensive one was the pressure circuit breaker: a solve returning NaN
across the whole fluid region counted as *zero* bad cells and was accepted.
The correction step multiplied the NaN into velocity, the boundary clamp
turned it into its own bound (`clamp(NaN, -100, 100)` yields `-100`, not
NaN), and the advection clamp folded every particle into the domain corner.
The visible symptom: a liquid running correctly for a couple of hundred
frames and then collapsing to a point in a single step, with `converged`
and `rejected` both reporting healthy right up to that frame.

This was not caused by the two-phase work. `20-flip-dam-break`, which
carries no dye and no density coupling, dies at frame 124 on the commit
that introduced it.

The fix is `src/float_guards.js`: `isNonFinite` as a bit-pattern exponent
test (integer arithmetic, out of reach of float assumptions), and
`isNonFiniteOrAbove`, whose bound is written as a *negated* "within range"
because `abs(x) > limit` also reports a NaN as fine while
`!(abs(x) <= limit)` does not. **Which way round the comparison is written
is the whole fix** -- both forms return false for a NaN, and only one of
them draws the right conclusion from that false.

`examples/27-float-guard-probe/` exists because the truth table in that
module is *device* behaviour, not spec behaviour. It runs four predicates
over nine float32 bit patterns in one dispatch, so the table can be re-run
on any GPU.

### 3. Figures measured under a bug

The "8% and climbing" empty-cell count recorded for the two-phase solver
had been measured before the NaN fix, on a simulation that was quietly
degrading. Re-measured: about 30 of 4096 cells, under 1%, and plateauing.
That changed the conclusion and not just the number -- the dark speckles in
the water are not empty cells, they are ~200 cells of genuinely trapped
gas, which is a different (and declared) limitation.

### 4. The circuit breaker made NaN permanent

The pressure snapshot it reverts to was taken *before* every solve, one
frame too early to be safe: a frame whose solve returns NaN leaves NaN in
the field, the next frame snapshots that NaN as its "known-good" baseline,
and every rejection afterwards hands the NaN straight back. The field is
then permanently poisoned, and a full scene reset does not clear it.

Fixed by snapshotting only *after* the bad-cell check passes, with a
zero-filled baseline for "no solve has passed yet". Necessary but not
sufficient, and the commit said so: a 1000-frame run still collapsed
afterwards, with pressure finite throughout. One failure mode removed, not
the root cause.

### 5. The root cause: the fixed-point atomic dot product

WGSL has no float atomics, so both CG solvers scaled each per-cell product
by a caller-supplied `atomicScale`, rounded to int, and atomicAdd-ed into a
shared int32. The encoding was forced; **its dynamic range was not.** Too
large overflows; too small quantizes everything under `0.5/scale` to zero
-- and a solve's dot products fall by orders of magnitude between the first
iteration and the last, so no single window works for a whole solve. Worse,
every division site had to treat a denominator under the quantization floor
as degenerate and stop, so the solver could not distinguish "this operator
is singular" from "this solve is nearly finished".

Eight examples each carried their own hand-tuned value -- 1024, 256, 1 --
with long comments recording how each was found. *That was the symptom.*

Measured on `26-dye-free-surface` at its own `atomicScale: 256`: only about
half of frames converged, and non-converged frames left real divergence
behind, which is a per-frame compressibility error that ate the liquid's
volume until it collapsed.

| | before | after |
| --- | --- | --- |
| frames converged | ~50% | **100%** |
| solves rejected | 5-17 per 150 frames | **0 in 2000** |
| max abs div post-projection | 0.1-0.4 | **< 0.005** |
| occupied cells | collapsed to 340-700 | **1536, flat for 2000** |
| peak velocity | hit the 100 clamp | 15.1, physical |

The replacement encodes nothing: each lane sums a slice in float32, writes
one partial, and the host adds the handful of partials in double precision.
Same cost profile, no scale at any problem size, and deterministic --
dropping atomics also drops run-to-run variation from GPU scheduling order.

### 6. The missing half of the Dirichlet elimination

`multigrid.js`'s `laplacianAt` eliminates a masked neighbour by
substituting 0 for its value, which is the right way to keep A symmetric
for PCG. But eliminating a *known* value is only half of the standard
reduction -- it has to reappear on the right-hand side as
`beta_face * target / h^2`, and it never did.

**With every target 0, the missing term is identically 0, so nothing
noticed.** It became visible only when reduced pressure made the air cells'
targets nonzero: interior divergence stayed exactly 0 while free-surface
cells came back with divergence up to 10.2 -- precisely the rows with a
Dirichlet neighbour -- and the liquid lost a third of its volume in 300
frames. With the term added, all three regions read exactly 0.

### 7. The V-cycle was not a symmetric operator

A long-run divergence bug with a Dirichlet mask, found by comparing against
mantaflow after direct dot-product logging showed the textbook PCG
breakdown signature: beta sitting consistently above 1 iteration after
iteration, so `p = z + beta*p` compounds geometrically until alpha produces
an astronomically large update in a single dispatch.

PCG requires a symmetric preconditioner. A V-cycle's red-black relaxation
is only symmetric as an operator if post-smoothing *reverses*
pre-smoothing's colour order. This file's `relax()` ran colour 0 then
colour 1 every time. An asymmetric preconditioner does not announce itself
-- there is no exact zero for a degeneracy check to catch; it just
occasionally produces a `z` misaligned enough with `r` to flip the sign of
`r.z`, and beta's runaway follows from there.

Two earlier, independently-motivated fixes in `buildRestrictKernel` and
`buildCorrectKernel` turned out **not** to be what was causing the observed
failures. They were kept, being correct hygiene, but the record says
plainly that they were not the fix.

### 8. A derived bound of zero

Reduced pressure allowed `maxPlausiblePressure` to be *derived* rather than
hand-picked: the pressure's scale is set by the Dirichlet targets, so it
cannot exceed `dt |g| L`. This removed the last per-scene magic number.

It then rejected every solve in `29-static-droplet`, which has **zero
gravity** -- the derived bound was 0, so a pressure peak of 0.25 was
"implausible" and the jump was exactly 0. Fixed by deriving the bound only
when `gravityMagnitude > 0`. A floor was deliberately *not* added, because
a floor is the magic number the derivation had just removed.

### 9. Harness bugs that looked like physics

Three distinct ones, each of which produced a confident wrong claim:

- **The page's rAF loop raced the driver loop.** Both called
  `onAdvanceTimeStep()` concurrently and their dispatches interleaved. This
  invalidated a reported "6 of 6 blow-ups within ten frames after a full
  scene reset", which does not reproduce once the loop is stopped. Fixed by
  giving the probe a `pause()` that returns a promise resolving after any
  in-flight step.
- **`new URL(...).pathname` yields `/D:/...` on Windows**, so the
  float-guard lint suite -- the thing that stops `x != x` coming back --
  was not running on Windows at all. 285 tests to 337.
- **A `/@fs` duplicate module import** gave a second profiling singleton, so
  counts read zero.

### 10. A comment that was simply false

A comment claimed sequential Gauss-Seidel converges faster per sweep than
red-black. That is untrue for a Poisson stencil. Corrected in the
write-up. Noted here because it is the failure mode documentation uniquely
enables: a wrong statement, confidently written, that outlives the code it
described.

---

### 11. The device was running on WebGPU's downlevel default limits

Found while trying to measure example 16, which turned out not to run at all in
a headless Chrome session -- the console filled with `Compute pipeline creation
failed: [Invalid BindGroupLayout] ... While validating binding counts`.

WebGPU's *default* device limits are downlevel values, not what the adapter
offers, and `requestDevice()` with no `requiredLimits` silently gets the
defaults. On this machine the adapter reports
`maxStorageBuffersPerShaderStage = 16` and `maxComputeInvocationsPerWorkgroup =
1024`, while the device three.js built had **8** and **256**. A kernel binding
nine storage buffers therefore failed at pipeline creation -- and WebGPU
reports a pipeline-creation failure as an *uncaptured* error on the first
dispatch that uses it, so the pass was silently never executed with nothing
visible from the JavaScript side.

Instrumenting `createBindGroupLayout` over the live page, after the fix:

| example | largest bind group layout | GPU errors before the fix |
| --- | --- | --- |
| 15 flow-past-cylinder | 2 entries | none |
| 16 karman-vortex-street | 8 entries (the one that failed predates the instrumentation) | many |
| 20 flip-dam-break | 8 entries | none |
| 28 drop-into-pool | **9 entries** | not checked, but over the limit |

So it was not one broken example: **example 28 was dropping a compute pass too**,
and example 20 was sitting exactly on the limit. Nothing on the JavaScript side
said so, and nothing in the test suite could -- the structural tests build node
graphs without a device, so a binding-count limit is invisible to them.

Fixed in `tsl_array_n`'s `init()`, which now requests the adapter and builds the
device itself with every limit at the adapter's own value, falling back to
three.js's own device if anything in that path throws. All four scenes run with
zero GPU errors afterwards.

## Performance

The full arc is in `docs/perf-investigation-cg-gpu-resident-alpha-beta.md`,
1300 lines including every rejected direction.

### What was landed

| Optimisation | Speedup | How measured |
| --- | --- | --- |
| V-cycle dispatch batching | **1.53x, 1.74x** | paired, phase swapped |
| Single-workgroup coarse solve with barriers | **1.10x, 1.09x** | paired, phase swapped |
| GPU-resident alpha and beta | **1.24x, 1.13x** | paired, phase swapped |
| **All three, off against on** | **3.10x, 2.63x** | paired, 100 frames per arm |

The compound exceeds the product of the parts (~2.1x) because each was
measured with the other two *on*, and removing one optimisation from an
already-fast configuration costs less than removing it from a slow one.

### What was rejected, by measurement

Recorded because the reasoning recurs:

1. **Mass-weighted P2G** (from the second OpenFOAM reading). Built,
   measured, rejected.
2. **CPU coarse solve, one per V-cycle.** 46.0 to 87.1 ms per frame; about
   3.9 ms per mid-cycle stall, and there are several.
3. **Direct coarse-level solve, serialised.** Slower, *and* a worse
   preconditioner: CG iterations 12.5 to 18.6, and it pushed a reference
   test past its tolerance.
4. **Per-frame diagonal precompute.** 72.3 to 93.7 ms -- 30% slower. The
   35% ceiling measured beforehand was real as a ceiling and worthless as a
   prediction.
5. **Coarse inverse uploaded once per frame.** Priced from data already in
   hand; never built.
6. **FFT/DCT preconditioner.** Not attempted. Its bound was to be measured
   first, and that measurement turned out to be invalid (below).
7. **Predictive stop-test scheduling.** Built, tested, measured, reverted.
   It works as designed -- 21% cheaper per iteration -- and it runs 26-32%
   more iterations, because a skipped check means the loop runs past the
   point it could have stopped. The two cancel exactly: 2% faster in one
   paired run, 5% slower in the next.

### The two measurement failures

**The V-cycle repetition probe.** To price "a much stronger
preconditioner", the V-cycle was applied N times per preconditioner
application. It reported 4.5x and a collapse from 12.5 CG iterations to 1.
Checked before being believed: **0 of 40 frames converged**, and
post-projection divergence was 5x the baseline. The solver had stopped
solving. Composing the V-cycle exposes that it is not quite symmetric --
the coarsest level never reverses its colour order, so `(RB)^k` stands in
for an operator whose adjoint is `(BR)^k`. One cycle is close enough that
CG tolerates it; composing amplifies the gap. The probe was removed.

**The benchmark harness.** Every millisecond figure in one round of work
was measured after 30 warm-up frames. This scene ramps for *several
hundred*:

| frames | ms/frame |
| --- | --- |
| 1-30 | 100.65 |
| 31-90 | 71.10 |
| 91-150 | 52.98 |
| 151-210 | 41.86 |
| 211-270 | 40.30 |

Measured properly, the ordering reversed: a configuration reported as 1.45x
faster was in fact slower. The tell was in the data all along -- "60 frames
at 96.8 ms" and "150 frames at 44 ms" are both about six seconds of wall
clock, because both were mostly paying the same fixed startup.

A second confound followed. This is an unsteady wake, so its own pressure
problem gets harder as it develops, and the machine's clocks drift under
sustained load. After 300 warm-up frames, five consecutive segments still
gave 41.6, 46.5, 50.9, 67.4, 71.2 ms -- monotonically rising. **A single
frame-time number cannot work on this scene.**

The fix was structural: every constructor-time decision that needed
comparing became a *runtime* choice, with all forms built once at
construction (two coarse-level queues, each with a batched and an unbatched
submitter), so both arms can be interleaved *inside one run*. Paired
measurement is the only kind that means anything here.

A third confound, caught later: **ms-per-iteration is not a fair metric
either.** It equals `fixed frame cost / iterations + per-iteration cost`,
so any change that raises the iteration count lowers it, whether or not
iterations got cheaper. A "round trips are three quarters of an iteration's
cost" conclusion drawn from it was wrong. With matched iteration counts, a
GPU-to-CPU round trip is about 0.2-0.4 ms, and three per iteration are
10-20% of a frame.

### Where the time actually goes

Established with `src/profiling.js`, which exists because two earlier
investigations both concluded "dispatch count is what costs" *by inference
from frame times, never measured*, and both said the next step was real
instrumentation.

- 502 dispatches per frame on a 64x64 grid **-- an undercount; the true figure
  is ~800 dispatches, and alongside them ~272 `renderer.compute()` calls, which
  turned out to be the thing that mattered. See the Performance section.**
- **CPU-side encoding: 2.28 ms of an 89.8 ms frame.** Encoding is not the
  bottleneck, which retires dispatch fusion as a direction.
- `cg-dot` runs three times per CG iteration -- `pAp` for alpha, `r.r` for
  the stop test, `r.z` for beta -- and each is a GPU-to-CPU round trip.
- It is not the transfer that costs. A bare 4096-float readback on an idle
  queue is 0.307 ms. It is that a round trip cannot return until the queue
  in front of it has drained.

That is what GPU-resident alpha/beta addressed: two of the three are
consumed only by kernels, so the host was reading them back in order to
divide two numbers and upload the answer again.

Moving them required moving every guard that ran *before* `updateX`, since
a loop that reads back at the end of an iteration cannot break before the
update. They take a stronger form in the kernels: **a tripped guard writes
the scalar as exactly 0**, making the update it feeds a no-op, so x cannot
be corrupted even for the one iteration before the host sees the flag.

---

### The submission path: what one dispatch actually costs

Recorded in full at the end of
`docs/perf-investigation-cg-gpu-resident-alpha-beta.md`. The measurements:

| quantity | value |
| --- | --- |
| wall time per `renderer.compute()` call, work held fixed | **38.8 us** |
| of which JavaScript encoding | 3 us |
| GPU execution of 64 dispatches in one submission | 0.052 ms |
| example 15 per frame | **272 submissions**, ~800 dispatches, **~1% GPU** |
| adapter `maxStorageBuffersPerShaderStage` vs device default | 16 vs **8** |

Two changes came out of it. The GPU-resident CG loop now submits its whole
iteration body as one batch -- **15 submissions per iteration before, 5 after**,
dispatch count unchanged, and submissions per dispatch halved on examples
15/16/20/28. And `tsl_array_n.init()` now asks for the adapter's own limits,
which is Debugging #11 above: that one was a correctness bug, not a performance
one.

**The frame-time payoff was reported as not established, and that report was
wrong.** Removing ~20 submissions per frame did not show up above the noise on
any of the four scenes measured (means differing by +-3%, in both directions),
and the conclusion drawn was "same dispatches, half the submissions, frame time
within noise".

A frozen-workload measurement -- driver paused, arms alternating every *step*
so both walk the identical sequence -- says otherwise:

| scene | batched ms/step | unbatched ms/step | speedup | submissions removed/step |
| --- | --- | --- | --- | --- |
| 15 flow-past-cylinder | 16.06 | 24.75 | **1.54x** | 232 |
| 16 karman-vortex-street | 24.53 | 28.32 | **1.15x** | 169 |
| 20 flip-dam-break | 21.81 | 34.00 | **1.56x** | 284 |
| 28 drop-into-pool | 55.14 | 82.87 | **1.50x** | 764 |

The earlier number was a measurement artifact of two things at once: a rendered
frame is not a solver step in these drivers (example 28's frame carries ~27
submissions where its step carries 359), and the frame time was pinned to the
60 Hz floor. **The criterion belongs on the step**, which is the unit the
library owns.

### Step 2: batching the FLIP stages (measured, and much smaller than estimated)

The same submission batching applied to the FLIP solvers' own stage sequences,
with the accumulator resets deliberately left outside the batches (they are
CPU->GPU uploads, and a pending upload only lands before a dispatch that begins
a pass). Frozen-workload measurement, `which=stages`, CG batching on in both
arms: **15.2 of 239 submissions removed per step on example 20 (0.998x in
time), 17.0 of 364 on example 28 (1.062x)**.

So roughly 6% of the submissions and nothing measurable, against an estimate of
~10% of the frame. Batching the top-level stage entries can only reach 15-17 of
them; the other ~140 per step live inside composites -- `constrainVelocity()`
(three calls per step), the P2G scatter and its finalize, the resets. Those
builders have to expose their dispatchers before they can join a batch, which
is the version of this step that would be worth measuring. The change is kept:
numerically identical, neutral in time, and a prerequisite for that.

## Testing

Three layers, because no one of them is sufficient here.

### 1. Structural tests -- `npm test`, 29 files, 347 tests

Vitest under Node, no GPU. They cover construction: allocating fields,
building kernels, shape and option validation, and pure functions
(`isDegenerateDenominator`, the float-guard bit predicate, level-shape
computation). `solve()` itself needs a GPU, so convergence is *not* tested
here.

They include a lint-style test that no source file reintroduces `x != x`,
verified to fail when the idiom is added back.

**What this layer cannot catch is most of the [Debugging](#debugging)
section.** Every bug there passed a green suite.

### 2. Example pages as live verification

The repository convention is that numerical behaviour is verified by pages
that check themselves against a known answer:

| Example | Checks |
| --- | --- |
| `04-conjugate-gradient` | exact solution, diagonal operator |
| `05-preconditioned-conjugate-gradient` | exact solution; **both scalar paths**; guard equivalence on a singular and a near-null operator |
| `07-multigrid-preconditioned-cg` | reference solution within 1e-2 |
| `27-float-guard-probe` | the float-guard truth table, on your GPU |
| `29-static-droplet` | Young-Laplace, verified to within 2% |

Example 05's guard cases were added late and matter: **every scene run
during the performance work reported a stop reason of `none`**, so the
rewritten guards would otherwise have shipped unexercised. A singular
operator (A = 0) and a near-null one (A = diag(1e-12)) reach
`degenerate-pAp` and `alpha-magnitude` respectively; both paths stop for
the same reason and leave x finite. The first case is not hypothetical: a
fully closed, all-Neumann pressure domain is singular, and
`grid_pressure_solver2.js` reaches it.

### 3. Long runs on real hardware

Driven through `window.__fluxflowProbe` -- `pause()`, `resume()`, `step()`,
and on some scenes `draw()`, so a driver that has stopped the rAF loop can
still render. Typical acceptance evidence per scene: frames converged,
solves rejected, non-finite counts, peak pressure, post-projection
divergence, and a conserved quantity (occupied cells, dye depth).

Representative, on the current default configuration:

| Example | converged | rejected | non-finite |
| --- | --- | --- | --- |
| 21 irregular container | 250/250 | 0 | 0 |
| 22 multiple colliders | 248/250 | 0 | 0 |
| 26 dye in free surface | 199/200 | 0 | 0 |
| 28 drop into pool | 238/250 | 0 | 0 |
| 29 static droplet | 80/80 | 0 | 0 |

### Two environment facts that invalidate results if forgotten

- **Real WebGPU requires the Chrome integration.** The in-app browser falls
  back to WebGL2 and produces false bug reports.
- **A backgrounded tab throttles rAF** to roughly one frame every several
  seconds, so any multi-hundred-frame run must be driven through the probe
  rather than by the page's own loop.

---

## Methodology, and what went wrong with it

The transferable part. Each of these was learned by getting it wrong.

**A fast result that is also a wrong result looks exactly like a
breakthrough.** Both the V-cycle repetition probe (4.5x) and the early
stop-test numbers (1.45x) were celebrations of a solver doing less work.
The habit that catches it: before believing a speedup, check that the thing
still converges and still produces a divergence-free field.

**Price the ceiling before building the machine.** Several directions were
retired for the cost of a measurement rather than an implementation. The
counter-lesson is equally real: a ceiling is not a prediction. The diagonal
precompute had a measured 35% ceiling and came out 30% *slower*.

**Compare within one run, never across runs.** An evolving scene and a
drifting machine will both masquerade as your change.

**Normalise, then check the normaliser.** Per-iteration cost looked like
the fair metric and was not, because fixed per-frame cost divided by a
changing iteration count moves it on its own.

**No per-scene magic numbers.** A standing constraint on this project: the
goal is that every user runs smoothly out of the box. Two per-scene
constants have been eliminated by deriving them -- `atomicScale` removed
entirely, `maxPlausiblePressure` derived from `dt |g| L` -- and at least
twice a proposed fix was rejected for being a magic number wearing a global
constant's clothes, including a floor for the derived pressure bound and a
fixed stop-test interval.

**Use the codebase's own settled answers.** The GPU guard kernels were
first written with hand-rolled comparisons whose direction happened to be
safe. `float_guards.js` already existed, with a measured device truth table
behind it, precisely because that reasoning is not dependable. Happening to
be safe is not the same as being right.

**When tuning by a proxy metric, look at the actual artefact.** The
vorticity panel's colour mapping was tuned three times against "percentage
of background pixels" before the field's own distribution was measured
(median 0.0004, p99 1.548, max 4.502 -- four orders of magnitude), which
immediately explained why no power law could work. One screenshot would
have said the same thing sooner.

---

## Open items

- **`src/grid/dye_field2.js`** is committed but deliberately not exported --
  a higher-resolution passive dye field, parked pending the agreed ordering
  (surface tension, then performance, then dye). The performance phase is
  now complete.
- **`examples/06-multigrid-preconditioner/`** reports its pinned cell as
  -42 against a target of 42. Pre-existing and unrelated to any of the
  above: the example never followed the masked-row sign flip that
  `grid_pressure_solver2.js` applies.
- **The coarsest level is still not a symmetric operator** -- its sweeps
  never reverse colour order. Harmless at one V-cycle per application, and
  the reason a stronger preconditioner cannot currently be built by
  composition.
- **One GPU-to-CPU round trip per CG iteration remains**, the stop test,
  which genuinely has to be on the host. Removing it means asking less
  often, and that trade has been measured to be worth nothing.
- **The frame is submission-bound, not compute-bound, and the numbers are in.**
  ~800 dispatches *and* ~272 `renderer.compute()` calls per frame on a 64x64
  grid, one submission costing 38.8 us, the GPU busy for ~1% of the frame. The
  earlier figure of 502 dispatches was an undercount: the profiler saw only the
  kernels built through `buildElementwiseKernel`.
- **Still open: whether removing submissions buys frame time.** ~20 submissions
  per frame removed, nothing measurable -- the Performance section above has
  why the synthetic slope does not transfer, and the frozen-workload
  measurement that would settle it.
- **The documentation backlog in `fluxflow-rules.md` section 8 is still
  outstanding** and was deliberately not folded into the submission commit:
  the stale `isDegenerateDot` name in README, `atomicScale` documented as live
  although deleted, the grid table's missing rows, examples 22/27/28/29 absent
  from the README index, and the perf document's own top Status (annotated in
  that file, not rewritten).
