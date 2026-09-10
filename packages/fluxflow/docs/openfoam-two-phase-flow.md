# Reference: how OpenFOAM solves two-phase flow, and what fluxflow should take from it

**Status: reference notes + a work list.** Written after reading OpenFOAM's own
two-phase solvers while root-causing a real blow-up in
`examples/26-dye-free-surface/` (see "The measurement that prompted this" below).
Items marked **[done]** have landed; items marked **[open]** have not.

## Licensing: read this first

**OpenFOAM is GPL-3.0. fluxflow is Apache-2.0. The two are incompatible in the
direction that matters here: GPL-licensed code cannot be copied, transliterated,
or adapted into an Apache-2.0 package.**

- **Source:** https://github.com/OpenFOAM/OpenFOAM-10 (OpenFOAM Foundation)
- **Licence:** GNU General Public License version 3, `COPYING` in that repository,
  "OpenFOAM is Copyright (C) 2011-2017 OpenFOAM Foundation".

So the rule followed in this document, and in any fluxflow change that cites it, is
the stricter one already established for SideFX Houdini in `THIRD-PARTY-NOTICES.md`:

- **Methods and algorithms are used.** Copyright protects expression, not ideas; a
  discretisation, a stage ordering, or a variable substitution is free to
  reimplement.
- **No code is copied, and none is transliterated.** Nothing below is a
  line-by-line rendering of an OpenFOAM file into JavaScript. The mechanisms are
  restated here in mathematical notation and in this project's own words, and any
  implementation is written from that description, not from the C++.
- **Nothing is quoted verbatim** -- not source lines, not comments, not
  documentation text. This document deliberately contains no OpenFOAM source
  excerpt, which is why everything below is written as equations rather than as
  code.

### Most of this is not OpenFOAM's invention anyway

Worth stating plainly, because it changes both the attribution and the risk: the
methods below are published, decades-old numerics. OpenFOAM is where a
production-quality *arrangement* of them was read, not where they come from.

| Mechanism | Actually from |
|---|---|
| Pressure split `p = p_rgh + rho*g*h` for free surfaces | Rusche, *Computational Fluid Dynamics of Dispersed Two-Phase Flows at High Phase Fractions*, PhD thesis, Imperial College, 2002 |
| Face-based (well-balanced) gravity at a density jump | Rusche 2002; same idea as the "hydrostatic reconstruction" family in shallow-water schemes |
| Correcting the *face flux* and reconstructing the cell velocity | Rhie & Chow, AIAA J. 21(11), 1983 |
| Surface tension as a volumetric force (CSF) | Brackbill, Kothe & Zemach, J. Comput. Phys. 100, 1992 |
| Bounded advection by a flux limiter (what MULES is) | Zalesak, J. Comput. Phys. 31, 1979 (FCT) |
| SIMPLE / PISO / their merge (PIMPLE) | Patankar & Spalding 1972; Issa, J. Comput. Phys. 62, 1986 |
| Partial elimination of interphase drag | Spalding, 1980 |
| Geometric VOF advection (isoAdvector) | Roenby, Bredmose & Jasak, R. Soc. Open Sci. 3, 2016 |

Where fluxflow implements one of these, the citation to record is the paper, with
OpenFOAM credited as the reference implementation that was consulted.

## The measurement that prompted this

`examples/26-dye-free-surface/` blows up on roughly half of page loads on real
WebGPU hardware (the scene seeds with `Math.random()` jitter, so every load is a
different initial condition). Traced frame by frame, on a failing load:

```
f1  b=[-2.08,0.20]  p=[-0.08,8.23]   u=[-12.0,10.8]   converged=false
f2  b=[-7.74,1.70]  p=[-19.0,4.91]   u=[-17.3,15.7]   converged=false
f3  b=[-18.2,17.8]  p=[-13.4,48.9]   v=[-11.2,31.9]   converged=false
f4  b=[-7.17,14.5]  p=[-153,256]     u,v saturated at the +/-100 clamp
f6                  p=[-8.8e6,1.5e7]                  rejected=true
```

Three facts from that run, each of which one of the items below speaks to:

1. **The very first solve is already wrong.** From rest, one step of gravity is
   `9.81/60 = 0.16`. The projection instead put `+/-12` into the velocity field, so
   the frame-1 pressure is not a small hydrostatic correction, it is garbage.
2. **Once `|p|` passes `maxPlausiblePressure`, the circuit breaker rejects every
   frame**, pressure stops updating entirely, and the liquid is in free fall until
   the velocity clamp catches it. Skipping the projection is not a stability
   mechanism; repeated, it *guarantees* divergence.
3. **NaN, once it lands, is permanent.** Later frames return NaN in exactly the
   fluid cells (893 of 893 fluid cells; 0 of the air cells) from a *finite*
   velocity field and a *finite* RHS. Reverting to the pre-solve snapshot cannot
   help, because the snapshot is by then NaN too.

## Part 1 -- the two families

OpenFOAM has two unrelated things both called "two-phase", and they answer
different questions.

| | **Interface capturing (VOF)**, `interFoam` family | **Euler-Euler two-fluid**, `twoPhaseEulerFoam` / `multiphaseEulerFoam` |
|---|---|---|
| Velocity | one shared field, one mixture momentum equation | one field *per phase*, one momentum equation per phase |
| Phase | volume fraction `alpha` in [0,1], interface at `alpha = 0.5` | a fraction per phase, phases interpenetrate |
| Density | mixture, `rho = alpha*rho1 + (1-alpha)*rho2` | each phase keeps its own |
| Coupling | implicit (same velocity field) | interphase drag `Kd`, partially eliminated |
| Good for | sharp interfaces: dam break, waves, drops, a rising bubble | dispersed phases: fluidised beds, bubble columns |

A free-surface liquid is the VOF case. The Euler-Euler machinery is relevant only
to `grid_two_phase_flip_solver2.js`'s sealed, all-fluid scenes, and even there only
as a comparison -- see Part 4.

## Part 2 -- what `interFoam` actually does, per time step

Notation: `u` velocity, `phi` the *face* volumetric flux, `p_rgh` the reduced
pressure, `rho` mixture density, `g` gravity, `h` height, `alpha` phase fraction.
Subscript `f` means "evaluated on a face".

### Stage order

1. Compute a flow Courant number **and a separate interface Courant number**, and
   shrink `dt` to satisfy both.
2. Outer (PIMPLE) loop, repeated `nOuterCorrectors` times:
   1. Advect `alpha` (optionally sub-cycled at its own smaller `dt`).
   2. Update `rho`, `mu` from the new `alpha`.
   3. Momentum predictor for `u` -- **without** gravity, surface tension, or the
      pressure gradient.
   4. Pressure-correction loop, repeated `nCorrectors` times.
3. Write.

### The alpha equation

Two mechanisms stacked:

- **Boundedness by construction.** The advective update is formed as a guaranteed-
  bounded low-order (upwind) flux plus an anti-diffusive correction that a limiter
  scales down, per face, by exactly as much as is needed to keep every cell inside
  `[alpha_min, alpha_max]`. That is Zalesak's FCT; OpenFOAM's version is called
  MULES. The point is that `alpha` is never *clamped after the fact* -- it is
  never allowed out of range in the first place, so no mass is invented or lost by
  the clamp.
- **Interface compression**, an artificial counter-gradient flux added along the
  interface normal, with magnitude `c_alpha * |phi| / |Sf|` and direction `n_hat`,
  active only where `alpha` is strictly between 0 and 1 (because it is multiplied
  through by `alpha*(1-alpha)`). It exists purely to undo numerical diffusion and
  keep the interface a few cells thick forever. `c_alpha = 1` is the usual value.

### The pressure equation -- the part worth studying

This is where the design decisions are, and all five of them are transferable.

**(a) Solve for the reduced pressure.** Substituting

```
p = p_rgh + rho * (g . x)
```

and solving for `p_rgh` removes the hydrostatic part analytically. The remaining
unknown is small, its magnitude does not scale with the depth of the liquid
column, and the solve is far better conditioned.

**(b) Gravity and surface tension enter as a face flux, not as a cell body force.**
The gravity contribution assembled onto each face is

```
phi_g = ( f_sigma - (g . x)_f * snGrad(rho) ) * rAU_f * |Sf|
```

where `snGrad(rho)` is the surface-normal gradient of density across that face.
This is the *well-balanced* property: at hydrostatic rest, `snGrad(p_rgh)` cancels
`(g . x)_f * snGrad(rho)` face by face, exactly, in the discrete equations -- not
merely in the continuum limit. A density jump therefore generates no spurious
velocity. Adding gravity as a cell-centred body force and then projecting does
*not* have this property, and the error it makes is largest exactly at the
interface.

**(c) The variable coefficient lives on faces.** `rAU = 1/A(u)` is the reciprocal
diagonal of the momentum matrix; it is interpolated to faces as `rAU_f` and the
pressure Laplacian is `laplacian(rAU_f, p_rgh)`. One stored value per face, read by
both adjacent cells, so the operator is symmetric by construction rather than by
two separately-averaged expressions that ought to agree. **fluxflow already does
this** -- `betaU`/`betaV` in `grid_flip_solver2.js` and `faceWeights` in
`multigrid.js` are the same decision, arrived at independently. No change needed;
recorded here as corroboration.

**(d) The constraint is enforced on the face flux; the cell velocity is
reconstructed.** After the solve,

```
phi = phi_HbyA + flux(p_rgh equation)          <- this is what is divergence-free
u   = HbyA + rAU * reconstruct( (phi_g + flux) / rAU_f )
```

The discretely divergence-free object is the *face flux*, and the cell-centred
velocity is a derived quantity reconstructed from it. This is the Rhie-Chow
arrangement, and it is what makes the discrete continuity error machine-zero rather
than "small".

**(e) A singular system is pinned, not survived.** When no boundary fixes the
pressure level (all-Neumann), the matrix is singular by one constant mode, and
OpenFOAM sets a reference cell to a reference value before solving. It does not
solve the singular system and hope. fluxflow currently handles the same situation
at the other end -- `linalg.js`'s `isDegenerateDot` detects that the search
direction has fallen into the operator's null space and bails out of the
iteration -- which keeps the solve from producing Infinity but leaves it
unconverged.

### Linear algebra, and the absence of a circuit breaker

The pressure system is solved with GAMG (agglomerated algebraic multigrid) or PCG
with a DIC preconditioner, to a stated `tolerance` and `relTol`, with the pressure
field under-relaxed between outer iterations. Stability comes from three places:
the adaptive `dt`, the repeated outer iterations, and the linear solve actually
converging.

**There is no equivalent of fluxflow's "the solve looked implausible, revert to
last frame's pressure" circuit breaker.** That is worth stating explicitly because
this session measured what that mechanism does when it fires repeatedly: it
converts a recoverable bad frame into unconditional divergence, and it cannot
recover from NaN at all because the value it reverts to is itself the poisoned one.

## Part 3 -- the work list for fluxflow

Ordered by measured value against the example 26 blow-up.

### 0. The actual root cause, found while working this list **[done]**

Not an OpenFOAM item at all, and worth putting first because it turned out to
be the whole of the blow-up rather than a contributing factor: the CG dot
product was a fixed-point value accumulated in an int32, scaled by a
per-scene `atomicScale`, and that encoding's dynamic range was too narrow for
a real solve. Every division site also had to refuse any denominator below
`0.5/atomicScale`, which a converging CG produces as a matter of course, so
the iteration stopped early on about half of all frames. See `linalg.js`'s
`createDotReducer` for the full account and the numbers.

Replaced with a lane-partitioned float reduction: no scale, no atomics, no
quantization floor, and deterministic. Measured on this scene, 2000 frames on
real WebGPU:

| | before | after |
|---|---|---|
| frames converged | ~50% | **100%** |
| pressure solves rejected | 5-17 per 150 frames | **0 in 2000** |
| max\|div\| after projection | 0.1-0.4 unconverged, 0.01-0.03 converged | **< 0.005 throughout** |
| occupied cells (1550 = healthy) | collapsed to 340-700 | **1536, flat for 2000 frames** |
| peak velocity | hit the 100 clamp | 15.1, physical |

Eight examples each carried their own hand-tuned `atomicScale` (1024, 256, 1,
...) with long comments about how each was found. All of them are now deleted:
the option is accepted and ignored, and the scenes run without it.

The remaining items are still worth doing -- item 3 in particular, since
`maxPlausiblePressure` is the last per-scene magic number in this stack -- but
they are improvements now, not fixes for a broken solve.

### 1. Never snapshot a bad pressure field **[done]**

Not from OpenFOAM -- this is a straight defect found while measuring. The circuit
breaker in `grid_pressure_solver2.js` snapshots pressure *before* every solve, so
once a frame ends with NaN in the field, every subsequent "revert to the last known
good value" reverts to NaN, permanently. Verified: a full scene reset (particles,
velocities, dye, velocity grid, and zeroing the pressure field with a kernel
dispatch) still blew up within ten frames on 6 of 6 attempts, because the internal
snapshot is unreachable from outside.

Fix: snapshot only a field that has *passed* the bad-cell check, so the snapshot is
known-good by construction, and clear to zero if there is no known-good field yet.

**Necessary but demonstrably not sufficient, and that is the important part.**
After the fix, the pressure field has not ended a frame non-finite in any run
measured (before it, runs that went bad held 500-900 NaN cells indefinitely) -- but
a 1000-frame run still collapsed: occupied cells fell from ~1550 to ~350 while
pressure stayed finite the whole time and `|p|` never passed 34. So the fix removes
one failure *mode* (permanent NaN poisoning, and the NaN reaching velocity through
the correction step) without touching the underlying problem, which is item 5 plus
whatever is making the very first solve wrong. Do not read a clean `pNaN` counter as
a healthy simulation; `filledCells` is the metric that actually detects this.

### 2. Adaptive `dt` from a Courant condition **[open]**

Example 26 hard-codes `dt = 1/60`. At the point of failure the grid velocity is
100 with `h = 1`, i.e. a Courant number near 1.7 -- well past where a
semi-Lagrangian/FLIP step is stable. OpenFOAM would have cut `dt`. fluxflow already
has `src/grid/grid_adaptive_timestep2.js` (CFL-based, built on `time/cfl.js` and
the GPU max-magnitude reduction); the FLIP solver already accepts `dt` as a live
node rather than a baked constant. This is mostly wiring, and it is the cheapest
thing that could plausibly stop the runaway before it starts.

### 3. Solve for `p - rho*g*h` instead of `p` **[open]**

Item (a) above. Two concrete payoffs here, beyond conditioning:

- `maxPlausiblePressure` becomes a meaningful bound rather than a per-scene magic
  number. Today it is 100 in examples 20 and 26 and different elsewhere, because
  the quantity being bounded scales with column height and grid spacing. The
  reduced pressure does not.
- The first solve of a settled column becomes trivial (the answer is near zero)
  instead of being the hydrostatic field.

### 4. Assemble gravity onto faces, well-balanced **[open]**

Item (b). fluxflow adds gravity through `external_force_solver2.js` as a per-face
force and then projects. For a *constant* density that is equivalent; at the free
surface and at a dye density jump it is not, and it is precisely the case
`grid_flip_solver2.js`'s one-sided `faceDensity` was written to handle. Pairing
that with the `snGrad(rho)` term is what makes the rest state discretely exact.

### 5. More than one pressure correction per step, with under-relaxation **[open]**

Item in "Stage order". fluxflow projects once per frame and accepts whatever the CG
returned, including `converged=false`. The measured failure has `converged=false`
on frame 1 and every frame after. An outer loop that re-solves would at minimum
turn a single bad solve into a recoverable one.

### 6. Pin a reference cell in a closed domain **[open]**

Item (e). Cheaper and better conditioned than detecting the null-space excursion
after the fact, and it would remove one of the two reasons `isDegenerateDot`
exists.

### 7. Bounded transport for the carried concentration **[open]**

The dye is currently clamped to `[0,1]` after the fact. A limiter-based update
(Zalesak/MULES-style) would keep it in range without a clamp, and an interface
compression term would stop a dye boundary diffusing away over long runs. Lower
priority: the particle-carried concentration in the FLIP solver is already
diffusion-free by construction, so this only matters for the grid-based `mixing`
path.

## Part 4 -- deliberately not taken

- **The Euler-Euler two-fluid formulation.** Right for dispersed phases, wrong for
  a sharp free surface, and its partial-elimination machinery only pays for itself
  when there are two genuinely separate momentum fields.
- **A real gas phase.** OpenFOAM's VOF always simulates the air (`rho ~ 1` against
  water's `1000`) and never uses the "empty cell is air held at `p = 0` with no
  mass" trick that this port's free-surface solver relies on. That trick is the
  graphics-side standard (Bridson), it is what makes the solver affordable, and it
  stays. The consequence to be aware of is that the well-balanced argument in item
  4 has to be re-derived for a one-sided face density rather than taken as read.
- **Geometric interface advection (isoAdvector/PLIC).** A sharper interface than
  compression can give, but it is a per-cell geometric reconstruction, which is a
  poor fit for the flat, uniform-grid, one-kernel-per-pass structure this port is
  built on.
- **Anything from the source text.** No OpenFOAM source, comment, or documentation
  wording is reproduced in this repository. See the licensing section at the top.

---

# Second pass: the two-phase specifics

The first pass above was about the pressure-velocity machinery, and everything
it recommended has now landed. This second reading went after the parts that
are specifically about carrying *two fluids*, which is where the remaining
value is. Same licensing rule throughout: methods only, no source excerpt,
originals cited where they exist.

## 1. Mass and momentum must be transported by the *same* flux **[open, highest value]**

The single most transferable idea in the whole two-phase solver, and the one
fluxflow currently gets wrong.

OpenFOAM does not advect momentum with the velocity flux. It builds a **mass**
flux from the *already-limited* phase flux that transported the phase
fraction -- schematically `rhoPhi = alphaPhi1 (rho1 - rho2)_f + phi rho2_f`,
where `alphaPhi1` is exactly the flux MULES limited -- and the momentum
equation's convection term is then the divergence of *that*. Mass and momentum
cross every face together, computed once.

Why it matters, and why it is not pedantry: if momentum is transported by a
flux that differs at all from the one that moved the mass, then a cell
receives an amount of momentum that does not correspond to the mass it
received, and the implied velocity is wrong by the ratio of the two. At a 1:1
density ratio the error is invisible. At 1000:1 it is a spurious acceleration
concentrated exactly on the interface, and it is the standard explanation for
why a naive VOF implementation blows up at high density ratios while looking
fine in a density-matched test. Rudman (*Int. J. Numer. Meth. Fluids* 28,
1998) is the usual citation for making the two consistent.

**What this maps to here.** fluxflow's FLIP has the same problem in
particle-to-grid form. `grid_flip_solver2.js`'s P2G accumulates
`sum(w * v) / sum(w)` -- a *kernel-weighted* velocity average, with no
particle mass in it anywhere. That is exactly right while every particle
weighs the same, and it is the wrong average the moment they do not, which is
precisely the case `carryConcentration` + `ambientDensity`/`componentDensity`
creates, and the case `grid_two_phase_flip_solver2.js` exists for. The
momentum-consistent form is `sum(m w v) / sum(m w)` with `m` the particle's
own mass from its carried density -- one extra multiply in the scatter, and
the finalize divides as before.

This is testable rather than a matter of taste: a density-stratified scene at
rest should stay at rest, and a volume-weighted P2G will show interface
velocity that a mass-weighted one does not.

## 2. Surface tension, which this port does not have at all **[open]**

Continuum Surface Force (Brackbill, Kothe & Zemach, *J. Comput. Phys.* 100,
1992), and the implementation details are the interesting part:

- The interface normal is `n = grad(alpha)_f / (|grad(alpha)_f| + deltaN)`,
  where `deltaN` is a small stabiliser derived from the average cell volume
  (order `1e-8 / V^(1/3)`). It exists because `grad(alpha)` is zero
  *everywhere except* at the interface, so the normalisation is 0/0 over most
  of the domain; the stabiliser makes it harmlessly zero instead.
- Curvature is the negative divergence of the face-normal field,
  `K = -div(n . Sf)` -- computed from the same face-interpolated normals, not
  from a second derivative of alpha directly.
- The force is `sigma K grad(alpha)`, and it is added **to the face flux**,
  in the same slot as the gravity term. fluxflow already built that slot for
  the reduced-pressure gravity term, so the plumbing exists.
- A wall contact angle enters by rotating the normal at the boundary before
  the curvature is taken, rather than as a separate force.

For a particle solver the alpha field is the P2G-scattered concentration or
fluid mask, which is noisier than a VOF alpha; smoothing before taking the
gradient is the known cost. Worth attempting only when bubbles or droplets
are actually the goal -- see `docs/two-phase-bubbles-research.md`.

## 3. The interface has its own Courant number **[open, cheap]**

Alongside the ordinary flow Courant number, OpenFOAM computes a second one
restricted to cells near the interface (`0.5 * max(sum|phi|/V) * dt`, over
interface cells only) and takes the smaller time step of the two. The bulk
can tolerate a larger step than the interface can.

fluxflow now has a single CFL limit over the whole velocity field. Adding an
interface-restricted one is cheap: the same max-magnitude reduction, run over
faces adjacent to a mixed cell, with a tighter Courant number. It would only
matter in a scene where the interface is fast and the bulk is slow -- which is
most splash scenes.

## 4. MULES, concretely enough to implement **[open]**

The first pass recorded what MULES is for. The algorithm itself, from this
reading:

1. For each cell, form bounds from its neighbours' values (the max and min
   over the neighbour set, optionally relaxed).
2. Split each cell's incoming corrective flux into its positive and negative
   parts separately, so the two directions can be limited independently.
3. Iterate a per-face limiter `lambda`, starting at 1: each sweep computes,
   per cell, how much of the correction the bounds still allow, and each face
   takes the *minimum* of the two allowances of the cells it separates.
   Three sweeps is the usual setting.
4. The final flux is the guaranteed-bounded upwind flux plus `lambda` times
   the correction.

That shape maps cleanly onto this port: a few elementwise kernels plus three
iterations of a face kernel, no atomics and no solve. It is the principled
replacement for clamping the dye to [0,1] after the fact.

## 5. A closed domain needs a *compatibility* fix, not just a pinned cell **[open]**

Worth separating two failure modes this port has so far treated as one.

- **Singular**: with no Dirichlet anywhere, the pressure is defined only up to
  a constant. The fix is to pin a reference cell. fluxflow does this.
- **Inconsistent**: if the prescribed boundary fluxes do not balance -- more
  coming in than going out -- then the Poisson equation has *no solution* at
  all, and pinning does not help. Before solving, OpenFOAM scales the
  adjustable outflow so that global inflow equals global outflow, and refuses
  to continue if the imbalance cannot be removed that way.

fluxflow's `sdf_inflow_outflow2.js` is exactly exposed to the second case: an
inflow and an outflow specified independently need not balance, and nothing
currently checks. The symptom would be a solve that never converges, on a
system that has no answer to converge to -- which is worth being able to tell
apart from a solver that is merely struggling.

## 6. Recorded as deliberately not applicable

- **Rhie-Chow interpolation and the ddt flux correction.** OpenFOAM stores
  velocity at cell centres, so it needs both to stop the pressure field
  checkerboarding. fluxflow is staggered (MAC): the pressure gradient is
  evaluated exactly where the velocity component lives, and the checkerboard
  mode cannot form. Nothing to import, and worth writing down so nobody
  imports a cure for an absent disease.
- **The optional momentum predictor.** A solver-cost knob for an implicit
  momentum equation; this port has no implicit momentum solve to skip.

## 7. Not from OpenFOAM: one magic number is still in the stack

Noticed while checking P2G for item 1, and recorded here because it is the
same class of defect the scale-free dot product removed: the particle-to-grid
scatter still accumulates through a **fixed-point atomic** with its own
`p2gAtomicScale`, defaulting to the old shared constant. A per-cell scatter is
much harder to overflow than a whole-field dot product, so this has not caused
a measured failure -- but it quantizes every transferred velocity, and it is a
knob a caller can still get wrong. The same treatment applies.
