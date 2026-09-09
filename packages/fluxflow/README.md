# fluxflow

Browser-side GPU fluid simulation, built on [tsl_array_n](../tsl_array_n), ported from a [Taichi Lang](https://www.taichi-lang.org/) fluid-simulation library (`D:\OneDrive\04_lib_fluxflow`). The end goal is real-time browser fluid visualization and interaction paired with three.js.

> **Status**: the Python source's `grid/` folder (MAC-grid data structures + numeric helpers + SDF colliders + boundary-condition solver), `noise/` folder (Perlin/Simplex/cellular noise), and `linalg/` folder (matrix-free conjugate gradient, preconditioned CG, and -- ported separately from jet/fluid-engine-dev rather than the Python source -- a geometric multigrid preconditioner) have all been ported. The actual fluid solver has now been built stage by stage on top of this foundation: semi-Lagrangian advection, external forces (a pluggable force-*function* mechanism), and pressure projection (Dirichlet-aware MGPCG, original generalization of jet's own hardcoded-zero "air" cells -- see below) are all done and wired together by `grid_solver2.js`, now a concrete orchestrator (external forces -> pressure -> advection every frame) rather than a hooks skeleton. Only viscosity remains unbuilt, explicitly deferred; `examples/14-stable-fluids/` is a fully-closed autonomous stability test, and `examples/15-flow-past-cylinder/` (new -- inflow/outflow + a real collider, see below) is the roadmap's concrete "final demo" scenario. A small `interaction/` module (original code, no Python/jet counterpart) adds pointer/keyboard DOM-event wiring so a hand-written force function (or Dirichlet-pressure function) can react to more than just position -- see below. CG/PCG, multigrid, advection (including a real boundary-crossing edge case found and fixed in its collider handling), external forces, and the `interaction/` module are all confirmed correct on real WebGPU (external forces additionally confirmed even in this dev sandbox's WebGL2 fallback -- see below for why); `examples/14-stable-fluids/` is now also confirmed stable over 1000+ real-hardware frames, after two real bugs were found and fixed: a missing boundary-condition constraint in `grid_solver2.js`, and a genuine CG solver bug (`linalg.js`'s `isDegenerateDot` guard, see below) where a closed/pure-Neumann domain's singular operator combined with this port's own fixed-point atomic reduction could produce a 0/0 division and 100%-non-finite pressure from a perfectly valid input. `sdf_inflow_outflow2.js`/`grid_outflow_solver2.js` (new -- inflow/outflow as reusable, SDF-based scene objects, derived from mantaflow, see below), `examples/15-flow-past-cylinder/`, and `examples/16-karman-vortex-street/` (new -- a longer, more asymmetric domain built to elicit alternating vortex shedding) are built and confirmed stable on real WebGPU hardware. A real long-run instability *was* found while building these (two specific hypotheses directly disproved by real-hardware experiment, an earlier "confirmed stable" claim retracted after it didn't hold up to further testing) -- but it has since been root-caused (an asymmetric red-black relaxation schedule in `createMultigridPreconditioner`, see below) and fixed; both examples were re-confirmed stable over multiple thousand real-hardware frames after the fix, and independently re-confirmed again in a later session (5300+ frames on example 15, 5750+ on example 16, both with the full outflow mechanism -- including velocity extrapolation -- active, no drift or non-finite values). See the inflow/outflow section below and `grid_outflow_solver2.js`'s own header comment for the full history. **Newest addition: `grid_two_phase_flip_solver2.js` (see below), a two-phase liquid+gas FLIP solver in which the air is a simulated phase rather than a `p = 0` void, so bubbles rise and trapped air pushes back -- coupled through a variable-density pressure projection, which needed new `faceWeights` (variable-coefficient) support in `multigrid.js`/`grid_pressure_solver2.js`. Confirmed on real WebGPU over a 780-frame run -- which found three real bugs the (green) tests could not, including a storage-buffer limit only a weaker GPU exposes and a `select()`-over-`atomicLoad()` codegen fault; see "Two-phase: confirmed on real WebGPU" below.** Three earlier, real, confirmed-and-fixed bugs remain fixed and are not in question: `numberOfLevels: 1` (plain relaxation, no actual multigrid coarsening) was not an adequate MGPCG preconditioner at this grid size (see `createMultigridPreconditioner`'s own section below); `grid_math.js`'s `bilinearGradientAtPosition2` degenerated to a zero gradient exactly at a grid's own physical edge (fixed generically, benefiting collider normals too); and `grid_outflow_solver2.js`'s own convective-velocity-extrapolation `factor` used the simulation's raw `dt` instead of mantaflow's own `max(1.0, dt*4)` floor.

## Current state: `grid`

```js
import { grid } from 'fluxflow';
```

| File | Corresponding Python source | Contents |
|---|---|---|
| `constant.js` | `constant.py` | Direction flags, `FLOAT_TYPE`, `createGravity()`/`setGravity()` (`array0('float')`, each simulation builds its own, not a module-level singleton) |
| `grid_math.js` | `grid_math.py` | Bilinear sampling / gradient / divergence / curl / laplacian, plus (not from the Python source -- see below) monotonic cubic sampling: `monotonicCubic1d`/`collocatedCubicValueAtPosition2`/`faceCenteredCubicValueAtPosition2`. Plain JS functions (building TSL node graphs), not `tsl_array_n.func()` |
| `level_set_utils.js` | `level_set_utils.py` | SDF's `isInsideSdf`/`fractionInsideSdf` |
| `array_utils.js` | `array_utils.py` | `createCopyKernel2`/`createExtrapolateToRegion2`, factory functions: build a field-bound kernel once, the returned function is the repeatedly-callable dispatcher |
| `grid_data2.js` | `grid_data2.py` | `ScalarGrid2`/`CellCenteredScalarGrid2`/`VertexCenteredScalarGrid2`/`CollocatedVectorGrid2`/`CellCenteredVectorGrid2`/`VertexCenteredVectorGrid`/`FaceCenteredGrid2`, factory functions thinly wrapping `tsl_array_n.array2()` |
| `polygon_sdf.js` | (no counterpart, replaces shapely) | Pure CPU geometry: point-in-polygon, point-to-polygon-boundary distance, polygon-union SDF (pointwise min), polygon centroid/translate/rotate |
| `svg_utils.js` | (no counterpart, replaces svg.path) | Samples vertices along a path using the browser's native `SVGPathElement` API, browser-only |
| `sdf_collider2.js` | `sdf_collider2.py` | `createSDFStaticCollider2`/`createSDFRigidBodyCollider2`; `addPolygon`/`addSvg` replace the source's `addShapelyGeometry`/`addSvg` (zero new dependencies, see below) |
| `grid_blocked_boundary_condition_solver2.js` | `grid_blocked_boundary_condition_solver2.py` | `createGridBlockedBoundaryConditionSolver2` -- velocity-field constraints from colliders + closed domain boundaries, the largest single file in this port |
| `advection_solver2.js` | (no counterpart -- ported from jet/fluid-engine-dev instead, see below) | `createSemiLagrangianAdvectionSolver2` -- semi-Lagrangian advection with monotonic cubic interpolation and boundary handling built into the back-trace |
| `external_force_solver2.js` | (no counterpart -- original code, see below) | `createExternalForceSolver2` -- applies an arbitrary caller-supplied force *function* (not just jet's own hardcoded-constant gravity) to a velocity grid |
| `grid_pressure_solver2.js` | (no counterpart -- ported from jet/fluid-engine-dev instead, see below) | `createGridPressureSolver2` -- pressure projection via `linalg`'s multigrid-preconditioned CG, with Dirichlet (fixed-value) pressure cells generalized beyond jet's own hardcoded-zero "air" cells |
| `sdf_inflow_outflow2.js` | (no counterpart -- original code, concept from mantaflow, see below) | `createSDFInflow2`/`createSDFOutflow2`/`createSDFFuelSource2` -- inflow/outflow/fuel-source as reusable, SDF-based scene objects, architecturally parallel to `sdf_collider2.js`'s own colliders; `createOutflowPressureDirichlet2`/`combineDirichlet` helpers |
| `grid_outflow_solver2.js` | (no counterpart -- original code, concept from mantaflow, see below) | `createGridOutflowSolver2` -- the velocity (convective boundary condition) and scalar-field-cleanup parts of what an outflow object does each frame |
| `grid_solver2.js` | `grid_solver2.py` | `createGridSolver2` -- now a concrete orchestrator wiring external forces, pressure projection, and advection together every frame (jet's own established stage order), with optional inflow/outflow objects and a collider; viscosity stays a no-op, explicitly deferred |
| `grid_adaptive_timestep2.js` | (no counterpart -- ported from jet/fluid-engine-dev instead, see below) | `createGridAdaptiveTimeStep2` -- CFL-based adaptive dt for a `FaceCenteredGrid2`; the grid-specific wiring on top of `time`'s solver-agnostic substep math and `linalg`'s GPU max-magnitude reduction |
| `grid_smoke_solver2.js` | (no counterpart -- ported from jet/fluid-engine-dev instead, see below) | `createGridSmokeSolver2` -- a reusable smoke/fire solver: buoyancy + density/temperature advection and decay, composed on top of `createGridSolver2` |
| `vorticity_confinement2.js` | (no counterpart -- original code, concept from mantaflow, see below) | `createVorticityConfinement2` -- compensates for semi-Lagrangian advection's own numerical dissipation by pushing fluid toward already-concentrated vorticity |
| `grid_fire_solver2.js` | (no counterpart -- ported from mantaflow instead, see below) | `createGridFireSolver2` -- a real fuel/combustion solver (fuel burns down, drives density/temperature), decoupled from any specific velocity solver -- composes into whichever one the caller is already using via a plain `force(pos)` |
| `velocity_damping2.js` | (no counterpart -- original code, see below) | `createVelocityDamping2` -- a uniform per-frame velocity decay, pluggable into `createGridSolver2`'s own `computeViscosity` stage hook with zero changes to that file |
| `grid_flip_solver2.js` | (no counterpart -- ported from mantaflow instead, see below) | `createGridFlipSolver2` -- a 2D FLIP (fluid-implicit-particle) liquid solver: fixed particle count, GPU-atomic particle-to-grid scatter, FLIP/PIC blended velocity update, `options.collider` (irregular/multiple/moving obstacles, via the pre-existing SDF-collider stack plus a new particle-side push-out) -- the first particle-based solver in this port; `computeFlipBoxSeed` -- seeds a rectangular box of particles |

### `advection_solver2.js` -- semi-Lagrangian advection, monotonic cubic interpolation

**Not ported from the Python source** -- `grid_solver2.py` never got past an abstract `computeAdvection` hook, so per the user's own "align with jet where reasonable" policy, this is read directly from [jet/fluid-engine-dev](https://github.com/doyubkim/fluid-engine-dev) (MIT license, Doyub Kim) instead: `semi_lagrangian2.h`/`.cpp` for the back-trace algorithm, `math_utils.h`'s `monotonicCatmullRom` for the interpolation itself (Fedkiw, Stam & Jensen's clamped-Catmull-Rom scheme from "Visual Simulation of Smoke", SIGGRAPH 2001, per explicit request), and `array_samplers2-inl.h`'s `CubicArraySampler2` for the 2D tensor-product structure. See [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) for exactly what carries over versus what's new here.

Two pieces, matching jet's own separation of a shared back-trace from an overridable sampler (folded into one factory here, matching this port's factory-over-inheritance convention):
- **`backTrace`**: adaptive-substep 2nd-order midpoint (RK2) integration tracing a point backward through the velocity field for `dt` -- **this is where boundary handling lives**, not a separate pass: after each substep, if the segment would cross into a collider (its SDF changes sign), the traced point is clamped to the approximate crossing point (linear interpolation in SDF value) and tracing stops there. tsl_array_n has no native dynamic-count while-loop, so the adaptive substep count (CFL-driven, data-dependent) uses a bounded `Loop()` with an early `Break()` once the remaining time is exhausted or a crossing is found -- the exact pattern already proven in `tsl_array_n/examples/04-julia/main.js`.
- **The final value lookup** at the traced-back position uses the new monotonic cubic samplers in `grid_math.js` (not linear) -- velocity sampling *during* backTrace's own RK2 steps still uses the velocity grid's existing (bilinear) `sample()`, matching jet's own behavior (only the final lookup benefits from the higher order).

`createSemiLagrangianAdvectionSolver2({ velocityGrid, collider?, dt, maxSubsteps?, order? })` returns `{ advectFaceCentered2(input, output), advectScalar2(input, output) }` -- `dt` accepts a plain number (baked in as a constant) or a node such as an `array0('float')`'s own callable reference (kept live across dispatches, the same pattern `linalg.js`'s `alpha`/`beta` scalars already rely on), so a real CFL-adaptive solver can share one `dt` field across every stage that needs it. Scoped to `FaceCenteredGrid2` (velocity self-advection) and `ScalarGrid2` (density/temperature) for this pass; `CollocatedVectorGrid2` isn't used anywhere in this port yet, so it's skipped for now.

`order` (default `1`, or `2` for MacCormack) -- found by directly comparing this port's own solver against mantaflow's (`source/plugin/advection.cpp`), matching mantaflow's own exact option name/values; see [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) for the full attribution. Order 1 is this file's original single-kernel step, byte-for-byte unchanged. Order 2 adds a second, backward-in-time trace that estimates how much error the forward step introduced, corrects for half of it, then clamps the correction to the locally-observed range of the original field (mantaflow's own "clampMode 2" -- explicitly marked in its own code comment as the variant *"recommended in Andy's paper"*, Selle/Fedkiw/Kim/Liu/Rossignac's *"An Unconditionally Stable MacCormack Method"*, over its own more complex clampMode 1) -- falling back to the plain forward value if no valid neighbor was found or the correction would overshoot. Both the forward and backward-in-time traces reuse this file's own existing `backTrace` unchanged in spirit (a new `direction` parameter flips its two sub-step formulas' own sign, provably inert at its own default), so MacCormack gets the exact same adaptive-substep/boundary-crossing-clamp behavior as order 1, not a separate, simpler trace. Costs 3 dispatches per advected field instead of 1 (forward, backward, correct+clamp) -- a real, inherent cost of the algorithm, not this port's own overhead.

Verified with 8 vitest structural tests (4 original + 4 for order 2, including a collider case), plus two live examples.
- `examples/08-cubic-interpolation/` checks `collocatedCubicValueAtPosition2` against an independent plain-JS reference at several non-grid-aligned query positions (including one exactly on a grid point, and one out of bounds). It read back wrong (effectively zero) results in this project's dev sandbox -- but re-running `examples/00-grid-math/`'s *already real-hardware-confirmed* plain bilinear `collocatedValueAtPosition2` (the direct linear sibling, same file) showed it currently fails identically in that same sandbox, a strong same-sandbox baseline for "known fallback limitation, not a new bug". **Run by the user on real WebGPU hardware, all 5 query points matched the independent reference (max |diff| < 1e-4), including the exact-grid-point case** -- confirmed correct.
- `examples/09-advection/` checks a constant-velocity exact-1-cell shift (no collider) and a wall-collider case. **Run by the user on real WebGPU hardware**: the shift test passed outright, confirming backTrace's core RK2 midpoint integration and velocity sampling. The wall-collider test initially failed *for real* (not a sandbox artifact) -- it read back the wall's own sentinel value, meaning the traced point had leaked all the way through instead of stopping at the surface. Root-caused with a plain-JS trace of the exact algorithm: jet's own trigger condition (`phi0*phi1 < 0`) misses a crossing whenever a substep happens to land exactly on the boundary first (phi0 becomes exactly 0, and `0 * anything` is never negative) -- a real edge case in the ported algorithm (present in jet's own C++ too, just far likelier to trigger with grid-aligned colliders and round velocity/dt values). Fixed by triggering on `phi1 <= 0` instead (the substep's endpoint being at or inside the solid); re-verified with the same plain-JS trace to now clamp one substep earlier, exactly at the wall's surface. **Re-run by the user on real WebGPU hardware after the fix: both tests pass**, with the wall-collider test's output coming back as *exactly* 1.0000 (the fluid value) -- landing precisely on the wall's own grid point means the cubic sampler's `f=0` case reconstructs that point's stored value exactly, regardless of the wall value still sitting in the stencil's other taps.

`order: 2` was verified against these same two scenarios directly (real WebGPU hardware, via the console, reusing `examples/09-advection/`'s own exact-answer construction): the wall-collider test came back *exactly* the same (1.0000, zero leakage) -- confirming MacCormack's extra backward trace inherits the same boundary-crossing safety as order 1, not a separate, unguarded path. The constant-velocity shift test matched exactly at every interior cell, but the domain's own rightmost cell came back `6.5` instead of the exact-shift construction's own expected `6` -- explained, not a bug: that cell's own backward-in-time trace (used only to *estimate* the forward step's error) lands one full cell past the domain's own edge, where index-clamping reads back a plausible-but-not-identical value, and clampMode 2's own "soft" range check (mantaflow's own recommended variant, see above) doesn't reject a `6.5` result since it still falls within the locally-observed `[6,7]` range -- a known, literature-consistent characteristic of MacCormack + soft clamping specifically at boundaries where the deliberately-exact test construction has no real analogue, not present anywhere in the tested interior. `advectFaceCentered2` (velocity self-advection) was also spot-checked with a non-uniform velocity field: fully finite, physically bounded output tracking the input's own shape.
`examples/16-karman-vortex-street/`'s own new "MacCormack advection" checkbox (bakes `order` into the advection kernels at construction time, so unlike the other two checkboxes this one reloads the page to apply) ran 2700+ frames with zero non-finite values, and produced clearly-alternating vortex shedding -- the original target phenomenon this whole example was built to show -- markedly more distinct than without it.

### `external_force_solver2.js` -- pluggable force-field functions

**Original code, not a port.** The user asked for a force-field mechanism supporting two input methods: a hand-written function, or a VDB file. Investigating the most viable JS VDB library (`mjurczyk/openvdb`, MIT-licensed, real npm package with a clean per-position `getValue(pos)` API) found a real gap directly in its source: its buffer-decoding table (`math/memory.js`'s `floatingPointPrecisionLUT`) only has byte-size entries for scalar types -- `vec3i`/`vec3s`/`vec3d` are recognized as type-name strings but have no decoding entry, so parsing a genuinely vector-valued VDB (needed for a force field) would likely error or produce garbage. Raised with the user, who asked to defer VDB entirely and focus only on the hand-written-function path for now.

`createExternalForceSolver2({ velocityGrid, force, dt })` -- `force: (pos) => vec2` is a plain TSL-node-returning function the caller authors, same idiom as `advection_solver2.js`'s `sampleVelocity`/`sampleBoundary`. Called once per face-centered velocity sample (u-faces and v-faces separately, at their own staggered positions), added to that face's velocity component scaled by `dt`: `velocity += force(pos) * dt` -- directly generalizing jet's own `GridFluidSolver2::computeGravity` (which only ever applies a *constant* gravity vector) to an arbitrary position-dependent force. No default force and no separate "constant gravity" convenience wrapper -- the user explicitly wants to author this themselves, and the trivial constant case is already a one-line closure (`(pos) => vec2(0, -9.8)`). The architecture is shaped so a VDB-backed force source, whenever it's built later, plugs in as just another function of this same shape, with zero changes needed here.

Verified with 3 vitest structural tests, plus `examples/10-external-forces/`: a constant force and a position-dependent custom force, both checked against exact hand-computed expected values (no interpolation involved anywhere in this solver, unlike advection). **Confirmed correct even in this project's dev sandbox** (not just on real WebGPU) -- this solver only ever reads/writes the velocity grid's own fields via `.addAssign()` (a "self-touch" pattern already established elsewhere in this project to be reliable even on the WebGL2 fallback), with no cross-field read of a separately-populated field anywhere, so it doesn't hit the fallback limitation that's affected numeric verification everywhere else in this port.

`force(pos)` can also react to anything else the caller's closure captures -- time, pointer, keyboard, or any other live value -- with zero support needed from this solver itself; see `interaction/` below and `examples/11-interactive-forces/`.

### `grid_pressure_solver2.js` -- pressure projection, Dirichlet-aware MGPCG

**Not ported from the Python source** -- same situation as `advection_solver2.js`: `grid_solver2.py`'s own `computePressure` never got past an abstract hook. Read directly from jet/fluid-engine-dev's `grid_single_phase_pressure_solver2.h`/`.cpp` instead. The user asked for a pressure solver supporting a caller-settable "target pressure," built on the existing MGPCG infrastructure (`linalg/linalg.js` + `linalg/multigrid.js`), fully GPU -- clarified via `AskUserQuestion` to mean Dirichlet (fixed-value) pressure cells, e.g. an open boundary or a "pressure vent," plus (a follow-up request) a mouse-interactive way to set one.

`createGridPressureSolver2({ resolution, gridSpacing, origin?, dirichlet?, multigrid?, tolerance?, maxIterations?, atomicScale? })` returns `{ project(inputVelocity, outputVelocity), pressure, b, diagnostics }` (`b` is the divergence RHS field CG actually solves against, `diagnostics.converged` is updated after every `project()` dispatch, `atomicScale` is forwarded to the internal CG solver -- all three exposed purely for diagnostics/tuning, added while root-causing a real-hardware pressure-solve failure in `examples/14-stable-fluids/`; see this file's own "createPreconditionedConjugateGradientSolver" section above and that example's own section below for the full story). `dirichlet: (pos) => { active, target }` is a plain hand-written function, same idiom as `external_force_solver2.js`'s `force` -- omit entirely for a pure zero-flux/Neumann domain. Internally builds a Dirichlet mask/target field from it (re-evaluated every `project()` dispatch, so a live/moving mouse-driven region works, not just a fixed one at construction time), solves `Laplacian(p) = divergence(u*)` via `multigrid.js`'s newly-added `dirichletMask` option (see below) wrapped in `createPreconditionedConjugateGradientSolver`, then corrects velocity by the pressure gradient. `project()` supports (and the orchestrator below actually uses) in-place operation -- safe because the correction step only ever touches its own velocity index plus a cross-*field* read of the pressure grid, no neighbor-velocity read, unlike advection.

**A real, easy-to-get-backwards sign flip, independently derived and hand-verified before relying on it**: jet's own pressure solver builds the *negated* Laplacian as its own internal matrix and correspondingly corrects velocity with a `+`. This port's own `multigrid.js`/`createLaplacianOperator`, already shipped and used since examples 04-07, computes the *standard* (non-negated) Laplacian instead. Reusing jet's own correction sign with this port's own operator would silently un-project instead of project -- verified concretely on a hand-worked 3-cell 1D case (`u*=[0,1,1,0]` gives `p=[0,1,2]`; a `-` sign correction gives the correctly divergence-free `[0,0,0,0]`, jet's literal `+` gives `[0,2,2,0]`, not divergence-free). `b` needed no sign change, only the correction step. Full derivation is in the file's own header comment.

`multigrid.js` itself gained an optional trailing `dirichletMask` parameter (threaded through `laplacianAt`/`laplacianDiagonalAt`/`buildRelaxKernel`/`buildResidualKernel`/`createLaplacianOperator`/`createMultigridPreconditioner`'s own `options.dirichletMask`), fully backward-compatible -- every existing caller passes none. A masked cell's row becomes the identity (`A@p=p`) instead of the normal stencil; neighbors need no special-casing at all (a Dirichlet neighbor's value is already correct wherever read). Applied at the finest level only, never coarsened -- a masked cell's relax update discards its incoming value entirely (`current + (b-current)/1 = b`), so the always-masked level-0 post-correction relax pass resets it to exactly `b(I)` regardless of what an unaware coarser level did in between; see `multigrid.js`'s own header comment for the full argument. jet's own hardcoded-zero "air" cells generalize here to an arbitrary caller-supplied target -- true Dirichlet, not a single constant.

Verified with vitest structural tests (`multigrid.js`'s own + a new `test/grid_pressure_solver2.test.js`) plus two live checks. **The Dirichlet mask mechanism itself, confirmed even in this dev sandbox** (added to `examples/06-multigrid-preconditioner/`, which needs no atomics at all): one cell pinned to a sentinel target value reads back as exactly that target after one 4-level V-cycle -- matched exactly. The full pipeline (`examples/13-interactive-pressure/`) needs MGPCG's GPU-atomic dot product like every CG-based example in this port, and in this dev sandbox hits a *different* WebGL2-fallback-only failure than the atomics compile error examples 04/05/07 hit: the shader actually compiles, but dispatching it logs `GL_INVALID_OPERATION: glDrawArraysInstanced: Not enough space in bound transform feedback buffers` (most likely this pipeline's aggregate number of simultaneously-bound storage buffers -- CG's own several scratch fields plus this solver's own mask/target fields plus velocity -- exceeding a driver limit no earlier, smaller example in this port needed to worry about). No JS exception is thrown either way (confirmed via manual frame stepping) -- **needs the user's real WebGPU hardware to confirm**, same as every other MGPCG-based piece of this port.

### `grid_solver2.js` -- concrete orchestrator

Previously a pure hooks-forwarder ported from `grid_solver2.py`'s abstract base class (all methods `pass`) -- now rewritten as a concrete factory per the user's own request to string together all existing modules into a complete grid solver (viscosity explicitly deferred, to be revisited later). `createGridSolver2({ velocityGrid, gridSpacing, origin?, force?, dirichlet?, collider?, inflows?, outflows?, closedDomainBoundaryFlag?, dt, advection?, pressure?, ...hooks })` builds real `external_force_solver2.js`/`grid_pressure_solver2.js`/`advection_solver2.js`/`grid_blocked_boundary_condition_solver2.js`/`grid_outflow_solver2.js` instances internally and calls them via `onAdvanceTimeStep(dt)` (now `async`, since pressure's CG solve needs an `await`) in jet's own established stage order: external forces -> viscosity (no-op, deferred) -> pressure -> advection. The original hook parameters (`computeExternalForces` etc.) are kept, repurposed as full-stage *overrides* -- supplying one replaces this file's own concrete default for that stage entirely, matching the Python source's original subclass-override spirit while making the un-overridden path actually do something out of the box.

**A second real, independently-confirmed finding while building this**: self-advecting `velocityGrid` onto itself directly (rather than through a scratch copy) is a genuine GPU race, not a sandbox artifact -- confirmed by reading jet's own `GridFluidSolver2::computeAdvection` directly, which *always* clones velocity before advecting for exactly this reason (`advectFaceCentered2` samples its input at arbitrary back-traced positions per invocation; same-buffer input/output gives no ordering guarantee across GPU threads). `grid_solver2.js` therefore keeps a scratch `velocityPrev` grid, copied from `velocityGrid` (`array_utils.js`'s existing `createCopyKernel2`) immediately before every advect dispatch -- this is unrelated to the "single writer per field" WebGL2-fallback-only quirk noted above (that one is about *multiple kernel objects* writing one field across separate dispatches; this is a same-kernel, same-dispatch read/write race, real on any backend).

**A third real bug, this one confirmed on real WebGPU hardware, not caught before shipping**: the very first version of this file never called `grid_blocked_boundary_condition_solver2.js`'s `constrainVelocity()` at all. The user reported an early interactive demo built on this file "not working well" on real hardware -- noisy, scattered dye instead of a smooth plume, plus a bright artifact along one domain edge. Root-caused by reading jet's `GridFluidSolver2::onAdvanceTimeStep`/`applyBoundaryCondition` directly: jet re-applies its boundary-condition solver's `constrainVelocity()` after *every* stage that touches velocity (gravity, viscosity, pressure, advection), not just once per frame. Without it, velocity at the domain edges is entirely unconstrained -- forces and self-advection can push it to arbitrary values there with nothing ever resetting it toward the correct closed-wall condition, contaminating the whole domain over many frames. Fixed by constructing a `createGridBlockedBoundaryConditionSolver2` internally (this factory's new `collider` option is forwarded to it; omit it for an empty domain -- the closed-domain-boundary part still applies either way) and calling `constrainVelocity()` from inside each default stage that changes velocity, matching jet's placement (inside each concrete stage, not the outer loop, so a caller-supplied stage *override* is responsible for its own boundary handling too). One asymmetry worth knowing: a collider-less construction needs no renderer (matches every other option here), but passing a *real* `collider` makes `grid_blocked_boundary_condition_solver2.js`'s own `setCollider()` dispatch a kernel immediately (building `blockMarker`), so that specific combination needs `tsl_array_n.init()` to have already run.

`gridSpacing`/`origin` must be passed as plain-number arrays here, duplicating what was already given to `createFaceCenteredGrid2` when building `velocityGrid` -- a known, accepted wart (`FaceCenteredGrid2.gridSpacing` is a TSL vec2 *node* with no way to read the original numbers back out of it), not fixed in this pass.

Verified with a rewritten `test/grid_solver2.test.js` (construction requires `velocityGrid` now; hook-override ordering, still async-aware) plus `examples/13-interactive-pressure/`, this port's first fully-orchestrated interactive demo: click/drag pins a circular Dirichlet pressure region (source or, with Shift held, sink) while the same wind/arrow-key external force and real self-advection all run together every frame -- two side-by-side canvases (velocity, color-coded as in example 11; pressure, grayscale). Same real-hardware-only status as `grid_pressure_solver2.js` above, for the same reason (MGPCG).

`examples/14-stable-fluids/` is the roadmap's final "concrete demo scenario" stage -- currently a **fully autonomous, deterministic solver-stability test**, with no mouse/keyboard input at all: a small fixed region near the left wall applies a constant rightward push and injects dye, every frame, forever, exercising external forces + Dirichlet-aware MGPCG pressure projection + real self-advection + closed-domain boundary conditions (all via `createGridSolver2`) with zero nondeterminism. It logs a periodic full diagnostic readout to the console (velocity/pressure/dye min/max, every 30 frames by default) plus a per-frame non-finite guard that freezes the loop and reports the exact frame number the moment any field goes non-finite -- deliberately more instrumented than a normal example, so a real-hardware failure can be diagnosed from actual data rather than another guess.

This file went through three real-hardware iterations before landing here, each surfacing something worth remembering:
1. **A hand-tuned "rising smoke" demo**, with a fixed-region constant upward force standing in for buoyancy. Stalled after a few frames on real hardware; the user asked directly whether gravity or thermodynamics had actually been built (they hadn't -- `constant.js`'s `createGravity()` exists but is wired into no force function anywhere, and there's no temperature field anywhere in this port) and diagnosed the likely cause themselves: a *fixed* push region stops lifting dye once it's risen past that region, unlike real buoyancy, which scales with a transported temperature/density field. Investigating this demo also surfaced a real, independently-confirmed bug in `grid_solver2.js` itself -- see that file's own section above for the boundary-condition-constraint finding.
2. **Jos Stam's classic mouse-driven "Stable Fluids" (SIGGRAPH 1999) test**, per the user's own suggestion, replacing the smoke demo wholesale rather than patching it: click/drag would push fluid in the swipe direction and inject dye, both via a short-radius splat (same idiom as `grid_pressure_solver2.js`'s own Dirichlet region). `interaction/pointer.js` only tracks position, not velocity, so this version computed its own frame-to-frame pointer velocity by reading `pointer.position`'s live CPU-mirror array directly (`.node.value.array`, the same escape hatch `linalg.js`'s `dotAccum.node.toAtomic()` already uses) and diffing against the previous frame. Reported on real hardware as an instant, total blackout on pointer release -- diagnosed (not independently confirmed, MGPCG doesn't compile in this dev sandbox) as most likely NaN, since a canvas silently clamps non-finite values to black and a single NaN velocity cell can poison the whole pressure field within a few multigrid iterations. Addressed defensively (clamped the per-frame pointer-delta magnitude, tightened the pressure solve's tolerance/iteration cap) rather than confirming one exact fix.
3. Still failing after that defensive pass, **the user asked directly for the current, non-interactive version** to isolate whether the core pipeline is stable at all, independent of any mouse-timing/input variable. This is also where the current diagnostic-readout/non-finite-guard machinery was generalized.
4. That autonomous version's own real-hardware diagnostic readout gave the actual answer: `b` (divergence) was finite and well-posed at frame 0 (`sum~=0`, as this deliberately-closed domain's solvability condition requires), yet pressure was *already* 100% non-finite that same frame -- ruling out gradual drift and pointing at the very first pressure `solve()` call itself. Root-caused as a genuine bug in the CG solver, not this example: a singular (pure-Neumann, no-Dirichlet-anchor) operator's search direction can drift into its own null space, driving `p.Ap` toward an exact zero that this port's own fixed-point atomic quantization makes especially easy to actually hit -- see `linalg.js`'s new `isDegenerateDot` section above for the full mechanism and the fix. **Confirmed fixed on real hardware**: after the fix, this example ran 1000+ frames with every field staying finite throughout, dye visibly flowing -- `diagnostics.converged` reads `false` for the first couple of frames (a cold start from an all-zero pressure field genuinely needs more than `maxIterations` to converge from scratch) then `true` continuously once the flow settles and CG starts each frame already close to the answer (`pressureGrid.data` persists across frames -- a warm start, not a fresh cold one every time). One further, not-yet-acted-on observation from that same run: velocity magnitude grows substantially over 1000+ frames (plausible early vortex shedding off the fixed push region) -- expected here, not a new bug, since this scene is a closed box (no outflow) with no viscosity, so injected kinetic energy has nowhere to drain except the advection/pressure discretization's own numerical dissipation; worth watching for in `examples/15-flow-past-cylinder/` below too, since that scene still has no viscosity even though it does add real outflow.

### `sdf_inflow_outflow2.js` / `grid_outflow_solver2.js` -- inflow/outflow as reusable SDF-based scene objects

Added directly at the user's own explicit request, in service of a "flow past a cylinder" scenario they specified: inflow at one wall, outflow at another, a uniform whole-domain force, and a circular collider. An initial draft (a per-wall bitmask baked directly into `grid_blocked_boundary_condition_solver2.js`/`grid_solver2.js`) was rejected in favor of this more general shape: **inflow/outflow as first-class, SDF-based scene objects, architecturally parallel to `sdf_collider2.js`'s own colliders** -- arbitrary shape, arbitrary placement (not hardcoded to "the whole left/right wall"), user-creatable and customizable, matching the same `grid`/`sample`/`gradient`/`isInside`/`addPolygon` interface a caller already knows from colliders (both factories *compose* `createSDFStaticCollider2` directly, reuse rather than duplication).

What an outflow object's presence actually does each frame turned out to be **three independent mechanisms**, read directly from [mantaflow](https://github.com/thunil/mantaflow) (Apache License 2.0, Tobias Pfaff & Nils Thuerey -- fetched via the GitHub API, no local checkout in this repo; see [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)) after a first, narrower read of a single source file had suggested it was just one:
1. **Pressure**: a zero-pressure ghost boundary -- reproduced with **zero changes to the pressure solver itself**, just the caller marking a thin ring near the outflow as Dirichlet-pinned to 0 via the *existing* `dirichlet` mechanism (`createOutflowPressureDirichlet2`). Confirmed at the exact code level: `multigrid.js`'s `laplacianAt`/`laplacianDiagonalAt` override a masked cell's whole row unconditionally, and `grid_math.js`'s `faceCenteredDivergenceAtCenter2` (what builds the pressure solver's RHS) has no edge-case branching at all, so marking any cell Dirichlet-zero works identically everywhere in the domain, boundary or interior.
2. **Velocity**: mantaflow does *not* hard-clamp this to 0 -- every `isOutflow` reference in its own `pressure.cpp` is inside a comment reading "don't change velocities in outflow cells". Instead it uses a convective/radiation boundary condition (Orlanski-style): average a local "bulk velocity", then extrapolate via `(vel - velPrev) / factor + vel(upstream neighbor)`. `grid_outflow_solver2.js`'s `applyOutflowVelocityBC()` adapts this from mantaflow's own axis-aligned, per-cell neighbor search to this port's SDF-based, arbitrarily-oriented outflow objects: the "upstream" direction comes from the outflow SDF's own gradient (the same technique collider normals already use), and both the bulk velocity and the upstream sample use `FaceCenteredGrid2`'s own existing bilinear `.sample(pos)`. Writes into a scratch destination array before copying back, the same read/write-race precaution `grid_solver2.js` already uses for self-advection.
3. **Scalar-field cleanup**: this is the part that actually corresponds to "the fluid disappears here" (confirmed directly with the user after an initial guess that the *pressure* treatment alone meant this) -- mantaflow's `resetOutflow` zeros a caller-supplied scalar field (typically density) within outflow cells every step, alongside particle/level-set bookkeeping this port has no equivalent for (a pure Eulerian grid method, no particles/free-surface). `clearOutflowScalarField(scalarGrid)` generalizes this to *any* caller-supplied `ScalarGrid2` (dye, in every example -- never a library-owned concept here).

Two deliberate extensions beyond mantaflow's own literal mechanism: the SDF-shaped (not wall-only) placement already mentioned, and inflow's settable `mode` -- `'set'` (mantaflow's own hard-override behavior) or `'add'` (superimpose an inflow's velocity onto whatever is already there, requested directly by the user, with no mantaflow equivalent). Inflow's own velocity-forcing (`grid_blocked_boundary_condition_solver2.js`'s new `inflows` constructor parameter and `applyInflow()`) forces both the boundary-normal MAC faces (two layers deep, mantaflow's own trick -- makes the first interior cell's divergence exactly 0 by construction, not dependent on CG convergence) *and* the tangential component along the same boundary row (an original addition, avoiding a silent-footgun where a caller's tangential velocity would otherwise be ignored).

**A genuinely subtle ordering interaction, worth understanding before changing either mechanism**: `grid_solver2.js`'s `defaultComputeAdvection` runs `applyOutflowVelocityBC()` *before* `boundarySolver.constrainVelocity()` (which does the closed-wall zeroing, then `applyInflow()`). This means a wall an outflow object overlaps *must* be excluded from `closedDomainBoundaryFlag` (now a `createGridSolver2` option too) -- otherwise the closed-wall zeroing would silently undo the outflow velocity extrapolation every single frame. Inflow doesn't have this problem (`applyInflow()` always runs last, after the closed-wall zeroing, in the same `constrainVelocity()` call), so excluding an inflow's own wall from the flag is optional, for clarity only.

Verified with new structural vitest coverage (`test/sdf_inflow_outflow2.test.js`, `test/grid_outflow_solver2.test.js`, extended `test/grid_blocked_boundary_condition_solver2.test.js`/`test/grid_solver2.test.js`) -- 149 tests total. `examples/15-flow-past-cylinder/` (new) exercises all of this together, plus a *real*, non-null `collider` through `createGridSolver2` for the first time ever (structurally wired since the pressure/orchestration work, but never actually run this way before): inflow at the left wall, outflow at the right, a uniform whole-domain rightward force, a circular collider at the domain's center (diameter = 1/3 of the domain's side length), dye injected near the inflow so any wake behind the cylinder is visible, plus a second canvas showing pressure. Two easy-to-miss wiring points confirmed while building it: the top-level `collider` option is *not* auto-forwarded to advection (`advection: { collider }` is required separately), and dye's own *separate* advection solver (dye isn't part of `createGridSolver2`'s scope at all) needs `collider` passed to its own construction too, or dye would visibly pass through the cylinder while velocity correctly flows around it.

**Root-caused and confirmed fixed on real WebGPU hardware**: the pressure solve genuinely, unboundedly diverged (not merely overflowed -- reducing `atomicScale` just moved the same divergence to a different int32 ceiling) starting within the first ~10-15 CG iterations, for this scene's combination of inflow, a whole-domain force, and pressure projection. Root cause: the example's own pressure config used `multigrid: { numberOfLevels: 1 }` (plain red-black SOR relaxation only, no actual coarse-grid correction) -- an inadequate preconditioner at this grid size (64x64, considerably larger and stiffer than any grid `numberOfLevels:1` had previously been exercised against). Switching to `numberOfLevels: 4` (matching `examples/06-07`'s own confirmed-stable configuration) alone resolves it: confirmed stable (all-finite velocity/pressure/divergence, low residual, a clean pinwheel-shaped pressure pattern around the cylinder matching classic bluff-body flow, pressure trending toward 0 near the outflow wall as predicted) over 1000+ real-hardware frames. Two other real, genuine bugs were found and fixed along the way while root-causing this (neither was the actual root cause, but both are real and worth keeping fixed): an outflow polygon padded only 1 cell past the domain edge could flip the outflow SDF's own gradient direction near the padding's far edge, and `multigrid.js`'s Dirichlet-mask handling made the Laplacian operator matrix asymmetric (see that file's own header comment) -- both are documented in detail at their own fix sites.

`applyOutflowVelocityBC()` (the convective/radiation velocity extrapolation, mantaflow's own "part 2") had its own separate instability, independent of the `numberOfLevels` fix above -- confirmed root-caused and fixed, two distinct real bugs, neither previously caught because no earlier example placed a query point at a grid's own physical edge or ran outflow's velocity extrapolation for long enough to expose the second one:
- **`grid_math.js`'s `bilinearGradientAtPosition2`**: the outflow SDF grid is cell-centered at this port's own resolution (cell centers at `x=0.5..63.5` for `N=64`), while the U-face positions this extrapolation samples run half a cell further, to `x=64` exactly at the domain's own right edge. The shared `bilinearCoordsAndWeights2` index pair this function used to reuse is allowed to collapse both indices to the same column there (correct for a *sampled value*, clamp-to-boundary -- wrong for a *gradient*, which then degenerates to 0 in that axis). Confirmed via an independent plain-JS reference implementation of the exact same formula: the old code gave `gx=0` at the domain edge, the fix (a dedicated, never-collapsing index pair plus a `[0,1]`-clamped blend weight, see the function's own header comment) gives the correct non-zero value, matching the interior case exactly wherever nothing has changed.
- **`grid_outflow_solver2.js`'s own `factor` computation**: confirmed by reading mantaflow's `advection.cpp` a second time, directly at the exact call site -- `applyOutflowBC` doesn't pass its own `timeStep` straight through to `extrapolateVelConvectiveBC`, it passes `max(1.0, timeStep*4)` instead. For any CFL-bounded simulation dt (this example's `1/30` included), that floor always wins, so mantaflow's own `factor` is a *constant* `~1.0`, completely decoupled from the real dt. An earlier version of this file used the real `dt` directly -- `factor = dt * max(1, bulkVel)` with `dt~=0.033` amplifies `(vel - velPrev)` by roughly 1/0.033 ~= 30x before adding the upstream sample, which is enough on its own to diverge every frame regardless of the upstream direction being correct. Fixed via `OUTFLOW_TIMESTEP_FLOOR`/`OUTFLOW_TIMESTEP_SCALE`, matching mantaflow's own literal safety margin.

The `factor`/timestep fix above is real and confirmed correct in isolation, but **a later investigation (building `examples/16-karman-vortex-street/`) found that neither this fix nor a since-reverted attempt at fixing `upstreamPt`'s own sign eliminated a separate long-run instability**, deterministic given a fresh page load, that persisted even with `applyOutflowVelocityBC()` disabled entirely (proving the root cause wasn't confined to that one mechanism). Both a sign-reversal hypothesis and an integer-boundary-alignment hypothesis for it were seriously tested and directly disproved by real-hardware regression experiments, not just reasoned about -- see `grid_outflow_solver2.js`'s own header comment for the full history. **This instability has since been root-caused and fixed**: the actual cause was entirely outside this mechanism -- an asymmetric red-black relaxation schedule in `multigrid.js`'s own `createMultigridPreconditioner` (pre- and post-smoothing used the same color order instead of reversed ones, breaking the symmetry a V-cycle needs to be a valid PCG preconditioner), found by comparing against mantaflow's own multigrid solver. Confirmed via multi-thousand-frame real-hardware runs at the time, and independently re-confirmed in a later session with fresh, deliberately-instrumented long runs: example 15 held at a stable, non-growing peak velocity (~28, at the outflow's own outer boundary) through frame 5300+; example 16 reached the same characteristic plateau by roughly frame 2000 and held it through frame 5750+ -- both with the full outflow mechanism (pressure Dirichlet ring, convective velocity extrapolation, and scalar cleanup) active throughout, `converged`/`rejected` diagnostics healthy the whole time, zero non-finite values. An earlier, now-superseded claim of "1500+ frames, reaching a steady fixed point, no known open issues" (made before the bug above was found) was retracted at the time for being based on a test run that wasn't representative -- that retraction was correct for what was known then, but is itself now superseded by the root-cause fix and the longer, more recent confirmation above.

### `examples/16-karman-vortex-street/` -- a visual (not physically-controlled) vortex-shedding demo

Built directly on example 15's own pipeline, retuned to actually show alternating shedding: a longer, 2.5:1 domain (160x64, not 15's own square 64x64) so a wake has downstream room to develop and shed more than once; a much smaller cylinder relative to the domain (radius 4, a ~12.5% blockage ratio, versus 15's own deliberately large 1/3-of-domain-height obstacle); and the cylinder deliberately offset a few cells off the domain's own vertical centerline -- the standard trick for seeding the asymmetry an inviscid, perfectly-symmetric setup would otherwise preserve forever (confirmed directly: a centered version of this same scene ran 880+ frames with no sign of shedding at all). This port has no viscosity model or vorticity confinement, so any shedding this example shows is driven by the semi-Lagrangian scheme's own numerical dissipation, not a controlled Reynolds number -- "does this look like a vortex street" is the right bar, not "does it match a specific Re's Strouhal number."

Ships at the library default (`outflowVelocityBC: true`, the full convective velocity extrapolation active) -- an earlier version of this file shipped `false` as a workaround for the long-run instability described above; once that instability was root-caused and fixed (the multigrid relaxation-schedule bug, not this mechanism), the workaround was no longer needed and was removed. Confirmed on real hardware to show a real, visually convincing wake, and to run stably (peak velocity plateauing, not growing; `converged`/`rejected` diagnostics healthy; zero non-finite values) through at least 5750 frames -- see the inflow/outflow section above for the full history and the fresh confirmation numbers.

Also demonstrates `grid_adaptive_timestep2.js` (below) via an "adaptive dt (CFL)" checkbox -- off by default (the exact previously-confirmed-stable fixed-`1/30` behavior, unchanged), on to drive the sim through `createGridAdaptiveTimeStep2` instead, with a live substep-count readout on the `#perf` overlay.

### `grid_adaptive_timestep2.js` -- CFL-based adaptive dt

Grid-specific wiring on top of two solver-agnostic pieces: `linalg`'s `createMaxAbsReducer` (a GPU `atomicMax` reduction, generalizing `linalg.js`'s own atomic-dot-product machinery from `atomicAdd`) and `time`'s `computeAdaptiveSubSteps` (a plain-number port of jet/fluid-engine-dev's `GridFluidSolver2::cfl()`/`numberOfSubTimeSteps()`, see below) -- `createGridAdaptiveTimeStep2({ velocityGrid, gridSpacing, dt, targetDt, courantNumber?, maxSubSteps?, atomicScale? })` returns `{ update, state }`; `update()` reduces the velocity field's current max magnitude, computes how many equal substeps `targetDt` needs, writes the resulting smaller dt into the shared `dt` node, and returns the substep count for the caller's own loop:

```js
const numSubSteps = await adaptiveTimeStep.update();
for ( let i = 0; i < numSubSteps; i ++ ) await solver.onAdvanceTimeStep();
```

**`dt` must be a live `array0('float')` node, not a plain number** -- `grid_solver2.js`'s `onAdvanceTimeStep(timeStepInSeconds)` passes its own argument to each stage, but every default stage function ignores it; the `dt` actually baked into the force/advection kernels is whatever was captured once, at `createGridSolver2()` **construction** time. `createGridAdaptiveTimeStep2` throws a clear error if given a plain number rather than silently doing nothing. A second, easy-to-miss requirement, found the hard way while wiring this into `examples/16-karman-vortex-street/`: a live `dt` field must be *invoked* (`dtField()`) before being passed to `createGridSolver2({ dt: dtField() })`/`createSemiLagrangianAdvectionSolver2` -- passing the callable field itself throws (`dt.toVar is not a function`, from inside `advection_solver2.js`'s `backTrace`); see `external_force_solver2.js`'s/`advection_solver2.js`'s own `options.dt` comments, corrected after this was found.

Verified with structural vitest coverage (`test/cfl.test.js` -- pure numbers, no GPU at all; `test/reduction.test.js`/`test/grid_adaptive_timestep2.test.js` -- construction only) plus a real-hardware check on `examples/16`: the reducer's own max-velocity reading matched an independent CPU-side computation from a raw `dataU`/`dataV` readback; a deliberately strict `courantNumber` produced a real multi-substep count (3, for that run's actual velocity) that was then driven through `onAdvanceTimeStep()` that many times with no non-finite values afterward; 300+ further frames with the checkbox on (default `courantNumber`, substep count staying at 1 for this scene's actual velocity range -- correct given jet's own generous default, not a bug, confirmed by cross-checking the formula against the measured velocity directly) showed no instability; toggling the checkbox off cleanly falls back to the fixed-`1/30` behavior.

### `grid_smoke_solver2.js` -- a reusable smoke/fire solver

The first "content" solver this port has built (as opposed to plumbing: advection, pressure, boundary
conditions) -- the user asked for one explicitly reusable, unlike every existing example's own dye,
which every single one hand-rolls its own advection/decay/injection for. Read directly from
jet/fluid-engine-dev's own `GridSmokeSolver2` (no Python source exists, same situation as
`advection_solver2.js`/`grid_pressure_solver2.js` above), which extends `GridFluidSolver2` with
density + temperature fields, a buoyancy force, and decay. jet does this via class inheritance; this
port has none, so `createGridSmokeSolver2({ velocityGrid, ... })` builds the same shape by composition
instead -- it constructs its own internal `createGridSolver2` (buoyancy folded into that factory's
existing `force` option, composed with any caller-supplied extra force) plus its own density/
temperature advection and decay, and returns `{ onAdvanceTimeStep, velocityGrid, density, temperature, solver }`.

Buoyancy: `f = buoyancyDensityFactor*density + buoyancyTemperatureFactor*(temperature - ambientTemperature)`,
applied along an `up` vector -- jet's own formula and default constants (`-0.000625`/`5.0`) carried over
exactly. `ambientTemperature` defaults to a fixed constant (`0`), not jet's own live domain-averaged
temperature -- deliberately, to avoid one more per-frame GPU reduction + readback (this project's own
CG performance investigation found that kind of synchronization to be a real cost on real hardware; see
`docs/perf-investigation-cg-gpu-resident-alpha-beta.md`). Diffusion is not ported at all, matching this
port's own already-deferred viscosity *and* matching jet's own default (`0.0`, i.e. off, unless a
caller explicitly sets a diffusion coefficient *and* solver -- neither exists in jet's own default
construction either).

**A real subtlety, worth understanding before touching this file**: density/temperature use this port's
established 4-field ping-pong (two "state" + two "raw advected scratch" fields, alternated by frame
parity -- the exact shape every existing example already hand-rolls for dye), since a storage field
needs exactly one permanent writer kernel on this project's WebGL2-fallback dev sandbox. But
`createExternalForceSolver2`'s own `force(pos)` closure is invoked exactly once, at construction time,
to build its kernel's node graph -- not re-invoked every frame. A buoyancy force naively reading one
fixed ping-pong slot would read stale data every other frame. Fixed with a single live
`array0('float')` parity flag (toggled via `.fromArray()` every frame, the same
already-built-kernel-reads-a-live-node pattern this port already relies on for `dt`/`alpha`/`beta`/
`simTimeUniform` elsewhere), driving a `select()` *inside* the once-built buoyancy kernel to read
whichever slot is currently active -- zero extra dispatches (a plain CPU buffer write plus a
branchless GPU `select`, not a kernel), unlike an alternative "copy to one stable field" design, which
would cost 2 extra dispatches every frame. Reading "whichever slot is currently active" means buoyancy
sees the *previous* frame's fully-advected-and-injected result -- this matches jet's own
forces-before-advection operator-splitting order exactly, not a deviation.

Density/temperature use `createCellCenteredScalarGrid2` (the proper half-cell-offset variant), not the
plain `createScalarGrid2` every dye example uses -- dye's own half-cell sampling error is imperceptible
for a passively-advected visual field, but density/temperature feed back into the buoyancy *force*, so
the sample position actually matters here. Each state field exposes its own `sample(pos)` (the same
`collocatedValueAtPosition2` wrapper `sdf_collider2.js` already establishes as precedent).

**Source injection is deliberately not built in** -- no emitter abstraction exists anywhere in this
port yet, and jet's own `GridSmokeSolver2` doesn't have one built in either. A caller builds their own
injection kernel(s) against `density.stateA`/`stateB` (mirroring every existing example's own
`createInjectKernel(rawAdvectedGrid, stateGrid)` dye pattern exactly), called once per frame after
`onAdvanceTimeStep()`, choosing A or B via `density.current === density.stateA`.

**"Fire" is not a separate physical model** -- there is no fire/combustion reference anywhere in jet or
mantaflow (confirmed via `grep -ril "fire|combustion|flame|fuel"` across jet's entire source tree, zero
hits). This is a deliberate scope decision: "fire" here means parameterizing and rendering the *same*
density+temperature solver (a hot, bright source plus a temperature-driven color ramp at render time),
not a reaction-front/combustion model -- a real one (mantaflow's own fire plugin, or Nguyen/Fedkiw/
Jensen's *"Physically Based Modeling and Animation of Fire"*, SIGGRAPH 2002) would be a fundamentally
larger, differently-shaped undertaking with no existing reference in this project.

Verified with structural vitest coverage (`test/grid_smoke_solver2.test.js` -- construction, returned
shape, plain-number and live-node tunables) plus a real-hardware check on the new
`examples/17-smoke-fire/`: a heated source's density centroid rose from y≈7 (at the source) to y≈62
over 350 frames with buoyancy at its default strength, then held that height (source's own injection
balancing the domain's top-wall outflow) through frame 1050+ with zero non-finite values throughout;
with `buoyancyTemperatureFactor` live-set to 0 (the demo's own "disable buoyancy" checkbox), the same
source's centroid stayed at y≈5.5 (essentially pinned at the source) through frame 450+, directly
isolating buoyancy's own contribution from plain advection/decay. A real-hardware screenshot after 250
frames shows a convincing rising, billowing plume with a white-hot core fading through orange to gray
smoke -- both canvases (density-only and the fire-colored blend) visually confirm the same structure.

`examples/18-explosion/` -- built on the exact same solver, no new library code -- swaps `examples/17`'s
own continuous small source for a single one-shot burst instead: a large, hot, dense disc plus a brief
outward velocity impulse, applied once (a "detonate again" button re-triggers it without reloading),
then nothing further -- no per-frame injection at all. What follows is entirely buoyancy and momentum
acting on that one initial condition. Real-hardware screenshots show a clean, symmetric vortex-ring
rollup within the first 90 frames (the textbook "mushroom cap" cross-section, the same instability real
starting-plume/explosion simulations rely on) that continues to rise and pass through the top outflow
through frame 240+, zero non-finite values throughout -- an emergent consequence of the existing
solver, not anything special-cased to produce that shape.

While investigating the stalled smoke demo, the user separately reported (on `examples/12-interactive-advection/`, unrelated to `grid_solver2.js`) a "long streak" visual artifact: dye reaching the domain edge, then moving under the force field, appeared to get "copied out and dragged into a very long region." Confirmed this wasn't an advection/boundary-clamping bug first (an independent plain-JS trace of the *oscillating-wind-alone* case did not reproduce any streak), then traced it to that example's own pointer-attraction term (inherited from `examples/11-interactive-forces/`) having *no* distance falloff at all (`normalize(...)` gives a constant-magnitude pull regardless of range) -- combined with a domain edge acting as a wall with no pressure to prevent pile-up, holding the pointer down could drag edge-accumulated dye all the way across the domain in a sustained smear. Not a bug in either example (a question, not a bug report -- 11/12 were not modified); the second iteration above deliberately used a radius-based falloff instead to avoid the same trap, and the current, non-interactive iteration sidesteps the question entirely by not using the pointer at all.

```js
import { interaction } from 'fluxflow';
```

| File | Corresponding Python source | Contents |
|---|---|---|
| `pointer.js` | (no counterpart, original code) | `createPointerUniform(element)` -- tracks pointer (mouse/touch/pen) position and button state over a DOM element |
| `keyboard.js` | (no counterpart, original code) | `createKeyboardUniform(keys)` -- tracks which of a caller-chosen set of keys are currently held down |

**Original code, not a port** -- there's no Python-source or jet counterpart to generalize; this exists purely so a hand-written `external_force_solver2.js` force function can react to something other than position. Both factories are thin DOM-event wrappers around the exact same "live-updatable GPU scalar" pattern already used elsewhere in this port (`linalg.js`'s `alpha`/`beta`, `advection_solver2.js`/`external_force_solver2.js`'s own `dt`): a plain `tsl_array_n.array0(...)` field, written from JS via `.fromArray()` on each relevant DOM event, and read live inside any kernel via the field's own callable reference (`pointer.isDown()`, `keyboard.fields.ArrowUp()`, etc.) -- no rebuild needed when the value changes. A time-varying force needs no wrapper at all: any plain `array0('float')` the caller updates themselves each frame already works the same way (see `examples/11-interactive-forces/main.js`'s own `timeField`), so there's no `createTimeUniform()` here -- it would be a one-line wrapper around what the caller can already write directly.

- `createPointerUniform(element)` returns `{ position, isDown, dispose() }`. `position` is normalized to `[0,1] x [0,1]` relative to `element`'s bounding box, with Y flipped to be up-positive (matching this port's grid/world-space convention, not the DOM's own down-positive screen Y) -- mapping that into a specific simulation's own coordinates (e.g. `.mul(gridSize)`) is left to the caller, since only they know their canvas-to-domain mapping. Listens for `pointerdown`/`pointermove` on `element` and `pointerup` on `window` (so releasing outside the element while dragging still clears `isDown`).
- `createKeyboardUniform(keys)` returns `{ fields: { [key]: array0('float') }, dispose() }`, one field per caller-chosen key name (matching `KeyboardEvent.key`, e.g. `'w'` or `'ArrowUp'`) -- deliberately per-key rather than a hardcoded WASD-style directional vec2, so any key (not just movement keys) can drive a force. Single-character keys are matched case-insensitively; multi-character key names are matched exactly.

Both are browser-only (DOM events) with no vitest coverage, matching `grid/svg_utils.js`'s precedent -- verified live instead via `examples/11-interactive-forces/`, which combines all three (time, pointer, keyboard) into one `force(pos)` closure: a time-varying oscillating wind, pointer-drag attraction, and arrow-key push, visualized as a live-updating 2D color field (no advection/pressure yet, so the grid is cleared and the force reapplied fresh every frame rather than accumulating). The underlying mechanism (each of `timeField`/`pointer`/`keyboard`'s fields is written via `.fromArray()` from JS, then read -- but never written -- by a GPU kernel) hits the *exact* already-documented WebGL2-fallback limitation described below ("reading a different field that already has data, from inside a kernel" -- confirmed here to also cover CPU-`fromArray()`-populated fields, not just GPU-kernel-populated ones): in this dev sandbox the force's time/pointer/keyboard-dependent terms read back as if frozen at their first-ever value. Isolated testing confirmed the TSL graph itself is wired correctly (a minimal standalone kernel reading a live field via `sin(field())` does pick up the live value, at least on some dispatches -- the failure is intermittent, not deterministic, consistent with this project's other confirmed fallback-only gaps). **Run by the user on real WebGPU hardware, the example works correctly end-to-end**: the field visibly oscillates over time, is attracted toward the pointer on drag, and responds to arrow-key input -- confirming the fallback-only diagnosis above and that `interaction/`'s live-uniform mechanism is sound on real hardware, same as every other fallback-only gap in this port.

`examples/12-interactive-advection/` combines this module with `advection_solver2.js`: dye painted at the pointer is carried around by the same interactive (time/pointer/keyboard-reactive) velocity field, transported frame to frame via `advectScalar2` rather than just visualizing the instantaneous velocity -- a more intuitive "does this look like fluid" check than example 11's color-coded velocity. Building it surfaced a real, previously-unknown constraint on this dev sandbox's WebGL2 fallback backend, not specific to interaction/advection at all -- see the "single writer per field" tradeoff below.

### `vorticity_confinement2.js` -- compensating for advection's own numerical dissipation

Found by directly comparing this port's smoke/fire solver against mantaflow's own -- mantaflow has
this (`extforces.cpp`'s `vorticityConfinement`/`KnConfForce`); jet/fluid-engine-dev does not, even
though both trace the technique to the same Fedkiw/Stam/Jensen SIGGRAPH 2001 paper jet's own
`GridSmokeSolver2` already cites. It compensates for a real, previously-documented limitation
(`examples/16-karman-vortex-street/`'s own header comment: "no viscosity model... and no vorticity
confinement either") by pushing fluid toward regions where vorticity is already concentrated, directly
counteracting how semi-Lagrangian advection smooths vortical structures away over time.

`createVorticityConfinement2({ velocityGrid, gridSpacing, strength })` returns `{ update, force }` --
deliberately a standalone, reusable utility, not built into `createGridSolver2`/`createGridSmokeSolver2`
(vorticity confinement is generic, not smoke-specific, and this port's existing composable-force
convention -- `force: (pos) => a(pos).add(b(pos))`, already used by `grid_smoke_solver2.js` to combine
buoyancy with a caller's own extra force -- already gives any caller a clean way to add this on top of
whatever solver they're using). `update()` must be called once per rendered frame, *before*
`onAdvanceTimeStep()` (the same sequencing `grid_adaptive_timestep2.js`'s own `update()` already
established) -- it computes curl and the confinement force from whichever velocity the *previous*
frame finished with. `force(pos)` is a plain `(pos) => vec2` closure, composable exactly like buoyancy.
Reuses `grid_math.js`'s already-existing `faceCenteredCurlAtCenter2` (built for `examples/16`'s own
vorticity visualization) and `scalarGradient2` directly -- the two hardest pieces already existed and
were already tested, so the whole feature is one new scratch field (raw signed curl) plus two kernels.

The 2D formula was re-derived directly from mantaflow's own general N-D form (`eta =
normalize(grad(|curl|))`, `force = strength * cross(eta, curl)`) rather than copied -- expanding the 3D
cross product with a 2D flow's curl vector `(0,0,curl)` gives `force = strength * (eta.y*curl,
-eta.x*curl)`, matching the standard 2D vorticity-confinement formula in the graphics literature. Also
adds one guard mantaflow's own code doesn't have: `normalize(grad)` is undefined (0/0) wherever the
vorticity-magnitude gradient is exactly zero, so below a small epsilon the confinement force is zeroed
at that cell instead of propagating a NaN direction -- the same defensive pattern already established
elsewhere in this port for this exact class of risk (`isDegenerateDot`, `bilinearGradientAtPosition2`'s
edge-case fix, `EXTRAPOLATED_VELOCITY_CLAMP`). No canonical default `strength` exists to port --
mantaflow's own Python-exposed default is `0` (off), so this port does the same; `examples/16`'s and
`examples/18-explosion/`'s own header comments document the values tuned empirically against them
(`0.15` too subtle to see clearly, `1.5` visibly overdriven, `0.5` a good balance for both scenes).

Verified with structural vitest coverage (`test/vorticity_confinement2.test.js`) plus real-hardware
checks: on `examples/18-explosion/`, `1.5` produced a visibly denser tangle of secondary curls
(confirming the mechanism has real, tunable effect, not just "doesn't crash") while `0.5` stayed dense
and turbulent without looking overdriven, stable through frame 500+ with zero non-finite values. On
`examples/16-karman-vortex-street/` (the new "vorticity confinement" checkbox, off by default), the
wake stayed visibly sharp -- clean, well-defined bands rather than a blurred one -- through frame 2500+
with it on, with zero regression when left off.

### `grid_fire_solver2.js` -- a real fuel/combustion solver, decoupled from any specific velocity solver

`grid_smoke_solver2.js`'s own "fire" is explicitly not combustion (see its own section above) -- this
is the real thing, ported from mantaflow's `fire.cpp` (`KnProcessBurn`/`processBurn`,
`KnUpdateFlame`/`updateFlame`; see [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)), built at the
user's own later explicit request with two requirements: usable with *any* velocity solver the caller
already has (not hard-wired to build its own `createGridSolver2` the way `createGridSmokeSolver2`
does), and SDF-based fuel input, architecturally parallel to collider/inflow/outflow.

`createGridFireSolver2({ velocityGrid, fuelSources?, dt, ... })` treats `velocityGrid` as read-only
input -- sampled for advecting its own 4 fields (fuel, react, density, temperature) and for buoyancy,
never constructed or written to. It returns `{ onAdvanceTimeStep, force, fuel, react, density,
temperature }` -- `force(pos)` is a plain `(pos) => vec2` the caller composes into *whichever* solver
they're using (`createGridSolver2` directly, `createGridSmokeSolver2`, or something fully custom),
exactly the same composable-force contract `vorticity_confinement2.js` already established above. The
caller must call their own chosen solver's `onAdvanceTimeStep()` first, then this file's own second,
every frame -- this solver's own advection needs that frame's already-updated velocity, and its own
buoyancy (same live-`parityFlag`-select trick as `grid_smoke_solver2.js`'s own) reads the *previous*
frame's density/temperature regardless, matching jet's own forces-before-advection operator-splitting
order.

The burn step itself, ported directly from `KnProcessBurn`: fuel burns down at a constant
`burningRate` (clamped to >=0); `react` (this batch of fuel's own remaining reaction potential) scales
down in exact proportion to how much of the fuel present that step was just consumed, and
`flame = sqrt(react)`; how much fuel was consumed drives both smoke emission (added to density, more
so as the fuel supply nears exhaustion -- mantaflow's own "guttering candle gets smokier" formula) and,
wherever `flame>0`, temperature is set to a lerp between `ignitionTemp` and `maxTemp` by `flame` --
left untouched wherever no reaction is happening that step, matching `KnProcessBurn`'s own
`if (heat && flame)` guard exactly. `burningRate`/`flameSmoke`/`ignitionTemp`/`maxTemp` default to
mantaflow's own literal values (`0.75`/`1.0`/`1.25`/`1.75`). Not carried over: mantaflow's own optional
colored-smoke mixing (no colored-smoke concept exists anywhere else in this port), and `flame` as its
own ping-ponged GPU field (it's a pure function of `react`, cheaper recomputed once at render time from
an already-read-back array than advected as a 5th field).

`createSDFFuelSource2` (`sdf_inflow_outflow2.js`, a new sibling next to `createSDFInflow2`/
`createSDFOutflow2` in that same file) mirrors `createSDFInflow2` exactly -- same SDF machinery, same
`mode: 'set'|'add'` semantics -- just injecting a scalar `fuel` amount instead of a velocity vector.
Fuel-source injection kernels are built one per (source, ping-pong slot) pair and dispatched every
single frame in a fixed order, right after that slot's own rightful advect-then-copy writer -- the
same shape `grid_blocked_boundary_condition_solver2.js`'s own `buildInflowKernels`/`applyInflow()`
already established (a real, long-proven-on-real-hardware precedent that more than one kernel object
touching a field is fine as long as every one of them dispatches consistently, every frame, from the
very first one) and the same lesson `examples/18-explosion`'s own burst-injection fix confirmed the
hard way this session (a one-off, pre-loop write is what actually broke on this project's WebGL2-
fallback backend; a consistently-ordered per-frame pair did not). Setting `react=1` alongside fuel
(regardless of `mode`) mirrors mantaflow's own scene-level convention of injecting fuel and react
together at a source -- fresh fuel always means full reaction potential.

Verified with structural vitest coverage (`test/grid_fire_solver2.test.js`, `test/sdf_inflow_outflow2.test.js`'s
own new `createSDFFuelSource2` cases) plus a real-hardware check on the new `examples/19-fuel-fire/`:
builds a *plain* `createGridSolver2` (not `createGridSmokeSolver2`) and composes the fire solver's own
`force` into it, proving the decoupling actually works rather than just describing it. A small
continuous fuel source near the bottom produced a genuine, self-sustaining flame -- fuel visibly
burning down and being topped back up by the source every frame, density/temperature rising from the
burn itself, no non-finite values -- confirmed stable over 1700+ real-hardware frames. **A real tuning
finding, not a library bug**: an initially-chosen `buoyancyTemperatureFactor` (22, scaled up naively
from `grid_smoke_solver2.js`'s own default to compensate for mantaflow's own much smaller temperature
range) looked fine for several hundred frames but was a genuine, still-growing instability -- u/v's own
per-frame *sum* grew steadily in magnitude with no sign of saturating, into the tens of thousands, even
though individual cell values and `converged`/`rejected` still looked superficially fine. Root cause:
unlike `examples/17`'s own modest continuous source or `examples/18`'s own single burst (which decays
away, nothing added after frame 0), this scene's fuel source keeps burning *forever*, continuously
adding buoyant energy every frame with nothing but this port's own numerical (not physical) dissipation
to remove it -- a continuous heat source needs a much gentler buoyancy coefficient than a one-shot one.
Reduced to `6` (close to the library's own default of `5.0`), re-confirmed over the same 1700+ frames: a
genuine, bounded, large-scale oscillation instead (the plume's own net horizontal momentum swings from
roughly -19500 back through zero to +2300 and reverses again, like a slow real-world plume sway; its net
vertical momentum settles into a stable plateau, the expected steady-state balance between continuous
buoyant injection and continuous outflow drainage) -- not a runaway.

### `velocity_damping2.js` -- a uniform velocity decay, plugged into `createGridSolver2`'s existing (empty) viscosity stage

Added at the user's own explicit request for `examples/19-fuel-fire/`, after they caught -- from their
own real-hardware screenshots, not this port's own testing -- a large-scale, slowly-reversing sideways
recirculation dominating that scene's entire canvas once its own fuel source was widened and its own
buoyancy softened for a thicker plume "stem". Asked directly how to weigh "thicker stem" against
"doesn't look chaotic," the user chose to invest in a real fix rather than tune around the symptom
further.

This port has no true Navier-Stokes viscosity model at all (`grid_solver2.js`'s own header comment:
"Viscosity stays a no-op -- explicitly deferred... not built yet") -- but that same file already carries
a `computeViscosity` stage hook, unused until now, in jet's own established stage order (external
forces -> viscosity -> pressure -> advection). `createVelocityDamping2` is the first thing plugged into
it, with zero changes needed to `grid_solver2.js` itself: `createGridSolver2`'s own `options.
computeViscosity` fully *replaces* the default no-op stage (not called alongside it), so a caller passes
a small closure that dispatches this file's own `applyDamping()` and then re-applies the boundary
condition (`boundarySolver.constrainVelocity()`) exactly the way every other built-in stage already
does -- `grid_solver2.js`'s own header comment already documents this as a caller override's own
responsibility, learned from an earlier real bug in that same file.

**Why a uniform decay (`field *= 1 - damping`, applied once to the whole velocity field, exactly the
existing `smokeDecay`/`temperatureDecay` idea already used for scalar fields in `grid_smoke_solver2.js`/
`grid_fire_solver2.js`) instead of a real Laplacian viscosity term**: a true diffusion-based viscosity
damps *small*-scale features fastest and the largest-scale mode in the domain slowest -- exactly
backwards from this problem, where the offending structure *is* the single largest-scale mode (a
recirculation cell spanning nearly the whole canvas). A uniform decay damps every scale equally,
including that one directly, is unconditionally stable (no diffusion-solve CFL-like constraint to worry
about), and costs a single per-cell multiply.

Verified directly on `examples/19-fuel-fire/`: with `dampingCoefficient: 0.02`, a fresh 3700+-frame
real-hardware run went from u's own per-frame *sum* swinging into the tens of thousands and staying
one-signed for thousands of frames at a stretch, to staying within roughly +/-300 for the entire run,
while v's own sum settled into a smooth, non-oscillating plateau instead of an oscillating one --
`converged` turned mostly `true` (up from mostly `false`) and framerate roughly doubled, both consistent
with the pressure solver having a far easier, calmer velocity field to project each frame. Visually: a
straight, symmetric, thick rising column with natural small-scale turbulent detail inside it, not a
single dominant vortex consuming the canvas -- the actual "thicker but not chaotic" result the user
asked for.

### `grid_flip_solver2.js` -- a 2D FLIP liquid solver, this port's first particle-based solver

Every other solver in this port evolves fields defined *on* the grid; FLIP (fluid-implicit-particle)
is a fundamentally different, hybrid particle+grid method -- the standard technique for liquid
simulation (splashing/pooling free-surface flows) -- built at the user's own direct request,
referencing mantaflow the way every other ported feature in this project has (see
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) for the full attribution and what carries over versus
what's this port's own adaptation).

Scope for this first version deliberately matches mantaflow's own simplest reference scene,
`scenes/flip01_simple.py` ("very simple flip without level set and without any particle resampling"):
a fixed particle count, seeded once via `computeFlipBoxSeed`, for the solver instance's whole lifetime
(no spawn/kill primitive exists -- `tsl_array_n`'s own arrays have no resize mechanism); no free-surface
level set (fluid/empty classification is purely live particle occupancy). Density resampling *within*
that fixed budget was added later -- see `options.resample` below -- once real-hardware use surfaced a
compaction problem the v1 scope didn't cover. `createGridFlipSolver2` therefore owns velocity and
pressure directly (building `createGridBlockedBoundaryConditionSolver2`/`createGridPressureSolver2`
itself, the same primitives `createGridSolver2` builds on) rather than composing into a caller-supplied
solver the way `grid_fire_solver2.js`/`velocity_damping2.js` do -- FLIP rewrites velocity outright
every frame (particle-to-grid scatter, "P2G"), it doesn't just contribute a force/decay term to an
otherwise-normal self-advecting solver.

Per frame: particles advect forward through the current velocity field (reusing
`advection_solver2.js`'s own existing back-trace machinery, already real-hardware-validated via
MacCormack's own forward-tracing step -- no new integrator needed); each particle scatters its own
velocity onto its nearby grid faces via a real **GPU atomic scatter** (one atomic accumulator per
face, since multiple particles can target the same face at once -- the first genuinely scatter-shaped
kernel in this whole port, distinct from every earlier one-thread-one-output-cell kernel); a snapshot
of the just-rebuilt velocity is kept; cells containing at least one particle are marked fluid, feeding
this port's own pre-existing general Dirichlet pressure mechanism (the same one
`sdf_inflow_outflow2.js`'s own outflow treatment already established) with a mask recomputed fresh
every frame instead of a static SDF; gravity and pressure projection run as usual; then each particle
reads back the grid's own velocity *change* from this whole step (the actual "FLIP" part -- keeps each
particle's own accumulated momentum/noise) blended with a little of the grid's own new velocity
directly (PIC, for stability) -- mantaflow's own 97%/3% default blend, exposed as `flipRatio`.

Demonstrated in `examples/20-flip-dam-break/`: a box of particles in one bottom corner of a closed box,
released under gravity -- the classic dam-break starting condition, matching `flip01_simple.py`'s own
first commented-out scene option. `p2gAtomicScale` (the P2G scatter's own fixed-point atomic-encoding
scale) is a per-instance option, not a shared global default, following this same port's own hard
lesson from `grid_pressure_solver2.js`'s own `maxPlausiblePressure` regression -- a "safe" atomic-
reduction magnitude tuned against one scene does not reliably transfer to a differently-scaled one.

**Collider/obstacle interaction** (`options.collider`, added after the initial v1 above) turned out to
need far less new machinery than it might look like: the grid-side treatment (velocity blocking,
no-flux + friction projection, pressure's own indirect dependence on already-collider-consistent
velocity faces) is entirely the pre-existing `createGridBlockedBoundaryConditionSolver2`/
`grid_pressure_solver2.js` machinery every other solver in this port already uses, just never
threaded through FLIP's own constructor before now; particle advection's own tunneling-prevention
clamp reuses `advection_solver2.js`'s existing `collider` option the same way. The one genuinely new
piece is a particle-side push-out kernel, ported from mantaflow's own `pushOutofObs`/`knPushOutofObs`
(see [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)) -- nothing previously corrected a *particle's*
own position against a collider, only grid velocity was protected. `sdf_collider2.js` also gained a
small `invert` option on `addPolygon`/`addPolygons`, needed to model an irregularly-shaped *container*
(fluid inside, solid outside) rather than a floating obstacle (fluid outside, solid inside), since the
domain's own `closedDomainBoundaryFlag` only supports a rectangular outer boundary. A genuinely moving
collider (`createSDFRigidBodyCollider2`) needs one extra per-frame call the caller makes directly --
`collider.update(dt)` then `boundarySolver.setCollider(collider, ...)` -- since that collider's own
`velocityAt()` bakes its position into the built kernel graph at construction time; `setCollider()`'s
existing rebuild-on-every-call mechanism (pre-existing, previously only exercised for occasional
swaps) is reused as the workaround, verified here for the first time under continuous every-frame
motion. Demonstrated in three separately-isolated examples, each real-hardware-verified on its own:
`examples/21-flip-irregular-container/` (a hand-authored wavy-basin polygon, inverted), `examples/
22-flip-multiple-colliders/` (two fixed pillars unioned into one collider via the pre-existing
`addPolygons([...])`, no new code needed for this scenario at all), and `examples/23-flip-moving-
collider/` (a translating + rotating paddle sweeping through a resting pool).

**Density resampling** (`options.resample`, added after the above) closes a gap the v1 scope left open
on purpose, found not by design review but by watching a real scene run long enough: `examples/20-flip-
dam-break/`'s own settled puddle visibly lost footprint over time, root-caused (via an isolated from-
rest control scene showing zero drift, versus the dam-break scene's own occupied-cell count falling
~17% over 700 frames and still worsening) to particle motion alone gradually clumping into denser and
denser cells with nothing to push back. mantaflow's own reference solves this with `adjustNumber`
(kill excess bulk-region particles, reseed under-min cells from the level set, called every frame in its
own fuller scenes -- see [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)); this port has no spawn/kill
primitive and no level set to reseed from, so it ports the *effect*, not the mechanism: an over-full
cell's excess particle is **relocated** directly into an under-full cell's own center (picking up the
local velocity there via the same `faceCenteredValueAtPosition2` helper G2P already uses) instead of
being destroyed and replaced. The relocation itself is a small atomic-list dance -- one pass counts
particles per cell, a second pushes every over-threshold particle's own index into a shared donor pool
(`atomicAdd`'s *return value* used as a claimed slot, the one genuinely new GPU primitive in this whole
port, isolated-tested on real hardware before being trusted here), a third lets each under-full cell
claim donors off that pool via the existing `Loop()`/`Break()` bounded-loop idiom already proven in
`advection_solver2.js`'s own `backTrace`. A recipient cell also needs at least 2 of its 4 orthogonal
neighbors already at/above `minParticlesPerCell` before it's eligible -- found necessary by testing,
not designed in up front: without it, isolated single-particle specks far from the fluid body got
reinforced every frame too, causing the fluid's own apparent footprint to runaway-*grow* instead of
shrink, the opposite failure. Defaults (`minParticlesPerCell: 4`, `maxParticlesPerCell: 8`) match
mantaflow's own `minParticles = 2^dim`/`maxParticles = 2x` convention for this port's existing
`particlesPerCellAxis=2` seeding density; `enabled: true` by default, since every existing FLIP example
gets this fix automatically through the shared solver. See `grid_flip_solver2.js`'s own header comment
("Particle resampling") for the full story including what's accepted as out of scope (no jitter on
relocation; a starved donor pool under-filling a recipient is expected, matching mantaflow's own
best-effort behavior).

**Velocity damping** (`options.velocityDamping`, added after the above) answers a real user report that
the simulation itself looked too energetic (described as looking like tumbling lava, not water), not just
a rendering complaint. Checked mantaflow's own `flip.cpp` directly rather than guessing at a fix: it has no
velocity-viscosity mechanism at all (its only smoothing kernels post-process a level-set surface this
port doesn't have), and both of its own reference scenes use `flipRatio=0.97` -- identical to this port's
own already-matching default -- so the energetic behavior isn't a mismatch against mantaflow's own
reference, it's inherent to FLIP at a high flip ratio in general (each particle carries its own velocity
forward with very little per-step numerical dissipation). Confirmed quantitatively before fixing anything:
a real-hardware run of `examples/20-flip-dam-break/` showed average particle speed oscillating between
~5-10 units/sec (peaks repeatedly spiking to 27-46) across a full 11-second window with no decay trend at
all. With no mantaflow mechanism to port, this reuses the *idea* already proven in `velocity_damping2.js`
(a uniform per-frame decay -- damps every length scale equally, unlike a real Laplacian viscosity which
spares the largest-scale mode, and needs no stencil or CFL-like constraint) but not that file's own
implementation, since it decays a grid's own `dataU`/`dataV` and the noise here is carried by particles
instead -- implemented as one extra multiply inside the existing G2P kernel, right after the `flipRatio`
blend. `velocityDamping: 0.02` (the same magnitude already validated for an analogous problem in
`grid_fire_solver2.js`, not picked arbitrarily) was confirmed on real hardware to settle the same
dam-break scene cleanly and monotonically to near-rest by roughly 8 seconds, instead of oscillating
indefinitely -- verified across all four FLIP examples, including the continuously-forced moving-collider
scene, with no new non-finite values or pressure-solve rejections introduced.

The user's own next, direct follow-up question -- confirming this is an artificial, not physically
derived, mechanism, and asking for it to be safely user-adjustable -- led to two further additions.
First, a **hard safety clamp inside the solver itself**: `velocityDamping` is unconditionally clamped to
`[0,1]` right where it's read in the G2P kernel, regardless of what any caller passes in. This isn't
cosmetic -- a negative value would *amplify* velocity every frame instead of damping it, a genuine
divergence risk, so the clamp is enforced in the kernel graph itself, not left to caller discipline.
Confirmed for real on real hardware, not just by inspection: deliberately feeding `-1` in for 90 frames
produced completely ordinary, bounded behavior (equivalent to the clamp flooring it to 0), and feeding
`10` in produced a clean, stable all-zero velocity field (equivalent to the clamp ceiling it to 1, "reset
every frame") -- neither diverged or went non-finite. Second, `options.velocityDamping` already accepted
a live node in place of a plain number (this port's own "number or node" convention, same as `dt`) --
`examples/20-flip-dam-break/` now exercises this for real with an actual `<input type="range">` control
(0 to 0.1, a *useful* exploration window, well inside the solver's own hard `[0,1]` bound) bound to a live
`array0('float')`, verified on real hardware to change the running simulation's own decay rate immediately
on interaction, no reload or kernel rebuild needed -- the same already-established live-uniform pattern
`interaction/pointer.js`/`keyboard.js` and `examples/16-karman-vortex-street/`'s own force controls use.

### `grid_flip_solver2.js` -- `carryConcentration`, a dye carried on the particles

An optional per-particle scalar -- dye, ink, a tracer, a second miscible liquid's mixing fraction --
carried through an ordinary free-surface FLIP simulation. Off by default; when off, none of its fields or
kernels are created at all.

**Transport is exact, and that is the point.** A grid-advected scalar picks up numerical diffusion from
every semi-Lagrangian lookup, so a dye filament smears whether or not you asked it to. A particle simply
carries its value, so the only blending is the blending you configure. `examples/26-dye-free-surface/`
states this as a measurement rather than an impression: over 480 real-WebGPU frames of two dyed columns
collapsing into each other, total dye stayed constant to the digit and the fraction of *partially* mixed
particles stayed at **0.00%** with `mixing` at zero -- every colour boundary in that scene is genuine
transport. This is also what Houdini does (dye there is a per-particle `Cd` attribute rather than a
solver), which is where the confirmation for the design came from.

Two optional controls decide the ending, both defaulting to off:
`mixing` lerps each particle toward the mean concentration of the particles sharing its cell, so the dye
softens and eventually goes uniform; `fade` decays it toward zero so the dye disappears instead. `mixing`
is a per-frame convex blend -- phenomenological and frame-rate dependent, **not** a discretised diffusion
coefficient. A physical treatment of genuinely mixing fluids models a per-component drift velocity
instead (Ren et al. 2014; Yang et al. 2015, both SPH and therefore design references rather than
something to port); that is the upgrade path.

**Why the dye lives here and not only in the two-phase solver**, which can also carry a concentration
*and* couple it to density: that solver is all-fluid, so its domain is a sealed box, and a sealed box
completely full of incompressible liquid can only circulate -- there is no free surface to rise or fall.
A dye scene built there is stable, correct, and visually almost inert, which was demonstrated rather than
assumed (`examples/25-dye-injection/` is kept precisely as that honest comparison, and its header records
the measurements and two wrong turns). A free surface removes the constraint: liquid that can slosh,
break and fold is what stretches a dye blob into filaments. The trade is that this solver has no
variable-density coupling, so the dye is a passive tracer -- carried and drawn, but exerting nothing. Dye
whose weight drives the flow is the two-phase solver's job. The two scenes together are the honest
statement of that trade-off rather than either being "the" answer.

One non-obvious interaction, taken from Houdini's users rather than discovered here: particle resampling
relocates particles, and a relocated particle carries its concentration to its new home. SideFX's own
forums repeatedly report reseeding diluting a carried colour attribute into mush under shearing, with
users disabling it to keep a sharp boundary. The same applies here, so a scene chasing crisp filaments
should set `resample: { enabled: false }` -- as `examples/26-dye-free-surface/` does.

### `grid_two_phase_flip_solver2.js` -- a two-phase (liquid + gas) FLIP solver, where the air pushes back

Every liquid solver above this one is single-phase: particles are the liquid, and every cell without a
particle in it is "air" -- a Dirichlet `p = 0` void with no dynamics at all. That is the standard
free-surface simplification and it is a good one, but it means air can never do anything. No rising
bubble, no pocket of air trapped under a breaking wave, no air-driven splash.

This solver simulates both phases. Each particle carries a phase tag (liquid or gas), both phases live
in one shared velocity field, and they are coupled through a **variable-density pressure projection**.
The consequence worth stating first, because it is the thing most likely to be disbelieved:
**there is no buoyancy force anywhere in this solver.** Gravity is applied uniformly to every face for
both phases, exactly as in the single-phase solver. The bubble rises purely because the projection that
follows knows the gas is lighter. There is no buoyancy coefficient to tune, and adding one would be
double-counting.

Unusually for this port, this is **not a port of any open-source solver** -- because after checking both
of this project's C++ references directly, neither has one. mantaflow does have a ghost-fluid pressure
path (`ghostFluidHelper`/`ApplyGhostFluidDiagonal`/`knCorrectVelocityGhostFluid` in `pressure.cpp`), but
reading it shows it is the *free-surface* GFM -- liquid versus `isEmpty`, placing `p = 0` at a sub-cell
position from a level set, with no second phase carrying its own density. jet has
`GridSinglePhasePressureSolver2` and `GridFractionalSinglePhasePressureSolver2`, single-phase as both
names say. So the algorithm comes from papers -- Kang/Fedkiw/Liu 2000 and Hong & Kim 2005 for the
variable-density formulation, Boyd & Bridson's **MultiFLIP** (ACM TOG 2012) for the phase-tagged-particle
two-phase FLIP design, Bridson's book for the discrete face-averaged stencil -- and the code is this
port's own. See [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) for the full citation block and for the
one piece that genuinely *is* ported from mantaflow (below).

**The pressure equation.** `grid_pressure_solver2.js`'s existing convention folds `dt/rho` into `p`,
giving `Laplacian(p) = div(u*)`, `u = u* - grad(p)`. With a spatially varying density that becomes
`div(beta grad(p)) = div(u*)`, `u = u* - beta grad(p)`, where `beta = rho_liquid/rho` -- exactly 1 in the
liquid and the full density ratio in the gas. Supporting it needed a new `faceWeights` option on
`multigrid.js` and `grid_pressure_solver2.js` (original to this port, see below), and `beta` is stored on
**faces** rather than in cells specifically so the operator stays symmetric by construction: two adjacent
cells read the same single array element for the coupling between them, rather than each computing its
own average that merely ought to agree. PCG depends on that symmetry, and `multigrid.js`'s own header
comment records what the last non-symmetric operator in this port did.

**The singular system, and the one piece that IS ported from mantaflow.** Once the air is simulated, a
closed domain has no Dirichlet cell left anywhere, so the pressure system is pure Neumann -- singular,
with pressure defined only up to an additive constant. It does not blow up in one frame the way a sign
error does; it drifts, quietly, until it trips `maxPlausiblePressure` and every solve starts getting
rejected. mantaflow hits exactly this case (`CountEmptyCells(flags) == 0`) and pins one cell's row to the
identity via `fixPressure`, preferring top centre `(sizeX/2, sizeY-1)`. That is ported directly and is the
`pressurePin: 'topCenter'` default -- and it needed no new machinery at all, since this port's existing
`dirichlet` option already means "identity row, this target", which *is* `fixPressure`.

**Particle resampling had to be rebuilt per-phase.** `grid_flip_solver2.js`'s resampler relocates a
particle from an over-full cell into an under-full one. Reused verbatim here that is a physics bug, not a
quality issue: it would relocate a *liquid* particle into a cell that is under-full because it is *gas*,
teleporting mass straight across the interface. So over/under detection stays on the total particle count
(that part is about sampling density and is phase-agnostic), but the donor pool is split by phase and each
under-full cell claims a donor of the phase it should be getting -- its own majority, or its neighbors'
majority if it has no particles to have a majority of.

**Verification.** Verified twice over, and the two passes caught completely different classes of problem
-- see "Two-phase: confirmed on real WebGPU" below for the real-hardware run and the three bugs it found
that the tests could not. The discretization itself is verified without a GPU at all:
`test/variable_density_projection.test.js` reimplements the exact same stencil in plain JS -- same sign
convention, same Neumann edge treatment, same Dirichlet elimination, same MAC face indexing, same
`beta grad(p)` correction -- and checks the properties that actually pin it down: the operator is
symmetric (with a negative control showing a per-*cell* beta would not be), `laplacianDiagonalAt` agrees
with the true diagonal of `laplacianAt`, the projection drives max divergence from ~1.15 to ~1e-13 (with a
negative control showing an unweighted correction leaves it above 0.1), a uniform-density scene reduces
exactly to the constant-coefficient solve, and -- the important one -- a buried gas bubble ends up moving
**up** under uniform gravity with no buoyancy term anywhere. This is the same plain-JS-reference technique
`multigrid.js`'s header comment credits for catching its earlier constant-diagonal bug, applied up front
this time rather than after the fact.

Demonstrated in `examples/24-two-phase-bubble-rise/`: a tank of liquid with a gas layer above the water
line and a gas bubble released at the bottom. The density-ratio slider is a direct test of the "no
buoyancy force" claim -- at 1:400 the bubble tears upward and breaks the surface, at 1:2 it barely drifts,
because the projection has almost no density contrast left to act on.

**Known limits of this first version**, all deliberate and all recorded in the file's own header comment:
one shared velocity field rather than MultiFLIP's two loosely-coupled per-phase fields (so some momentum
bleeds across the interface); no level set, so no sub-cell ghost-fluid interface and no surface tension;
no MultiFLIP-style particle-position anti-mixing; and the multigrid preconditioner is still
constant-coefficient at every level (`faceWeights` applies at level 0 only, exactly as `dirichletMask`
already did), so it preconditions this system less well the larger the density ratio -- which is the
direct reason `gasDensity` defaults to 0.01, a 100:1 ratio, rather than real air/water's ~816:1.

### `multigrid.js` / `grid_pressure_solver2.js` -- `faceWeights`, variable-coefficient support

A partial walk-back of `multigrid.js`'s own decision-1 constant-coefficient scope cut, added for the
solver above. `faceWeights` is an array of per-axis accessors where `faceWeights[axis](...I)` is the
coefficient on the **lower** face of cell `I` along `axis` -- which is exactly MAC face-array indexing
(`dataSizeU = [resX+1, resY]`), so a caller passes its existing face grids straight in with no new layout.
Fully backward-compatible in the strong sense: absent the option, not a single extra node is emitted and
every existing caller's kernel graph is unchanged, which matters because several shipped scenes are tuned
against specific `atomicScale`/`maxPlausiblePressure` magnitudes that a stray `mul(1.0)` has no business
perturbing. Applied at the finest level only, same as `dirichletMask` and for the same reason (no
per-level operator storage exists to hold restricted coarse coefficients); the cost is convergence, not
correctness, and it grows with the coefficient ratio.

## Current state: `noise`

```js
import { noise } from 'fluxflow';
```

| File | Corresponding Python source | Contents |
|---|---|---|
| `noise.js` | `noise.py` | `perlinNoise3d(P)`, `simplexNoise3d(v)`, `cellular3d(P)` -- ported from [WebGL-Noise](https://github.com/ashima/webgl-noise) (MIT), via the Python `fluxflow` project; see [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) |

Meant for use inside a `kernel()`/`func()` body, or composed into `grid_math.js`-style helpers -- e.g. as a forcing/initial-condition field for a future solver, or just as a general parallel-compute building block. The source's `ENABLE_COMPLEX_VERSION` branch in `cellular3d` (a hardcoded-`False`, never-toggled flag) was dropped as dead code -- only the real F1+F2 branch is ported; see the comment in `noise.js`. `permute`/`taylorInvSqrt` are also exported (small building blocks, in case a future 4th noise variant wants them); `mod289`/`mod7`/`fade` stay internal-only, alongside the four trivial one-line Taichi wrappers (`floor`/`fract`/`abs`/`dot`) which weren't ported at all -- TSL's own equivalents are used directly at each call site instead.

Verified by `examples/03-noise/`: renders all three as a 2D grayscale slice. Unlike `grid_math.js`'s functions, these don't read any other already-populated field (each thread only computes from its own position), so — unlike `examples/00-grid-math/` — this one is expected to (and does) render correctly even on this dev sandbox's WebGL2 fallback, not just on real WebGPU.

## Current state: `linalg`

```js
import { linalg } from 'fluxflow';
```

| File | Corresponding Python source | Contents |
|---|---|---|
| `linalg.js` | `linalg.py` | `createConjugateGradientSolver(applyOperator, b, x, options?)` -- matrix-free conjugate gradient, ported from Taichi Lang's own `matrixfree_cg.py` via the Python `fluxflow` project; see [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md). Also `createPreconditionedConjugateGradientSolver(applyOperator, applyPreconditioner, b, x, options?)` -- **original code, not a port** (see below) |
| `reduction.js` | (no counterpart -- original code, see below) | `createMaxAbsReducer(fields, options?)` -- GPU max-magnitude reduction via `atomicMax`, generalizing `linalg.js`'s own atomic-dot-product (`atomicAdd`) machinery; feeds `grid_adaptive_timestep2.js`'s CFL computation |

Meant for the pressure-projection step of a future solver (`Ax=b` where `A` is a matrix-free Laplacian-like stencil operator, `b` the divergence, `x` the pressure). Real structural differences from the source, driven by platform gaps, not style -- see the file's own header comment for the full reasoning:
- **GPU-side reduction via WebGPU atomics, not a CPU sum**: Taichi's `result += p[I]*q[I]` inside a kernel is a genuine parallel reduction the compiler handles on-device. tsl_array_n has no reduction primitive of its own, but three.js TSL exposes real `atomicAdd` on a storage buffer marked `.toAtomic()`, so both dot products this solver needs (`r.r`, `p.Ap`) run as every thread atomically adding its own per-cell product into one shared accumulator -- a genuine GPU-side reduction. Since WGSL atomics only exist for `atomic<i32>`/`atomic<u32>` (never float), each product is scaled by a configurable `atomicScale` (default 65536) and rounded to a fixed-point int before the add, then divided back after reading the single accumulated int back to the CPU -- trading exact float precision and some int32 headroom for turning an O(N) CPU-bound reduction + O(N)-element transfer into an O(N) GPU-bound reduction + a single-int transfer. This also means the solver currently only supports scalar `'float'` fields (a per-component accumulator would be needed for vector types, not attempted since nothing here needs it). `solve()` is still `async`, since the convergence check is inherently CPU-side and still needs one small readback per iteration either way.
- **"Create once, solve many times"**: the source dynamically allocates and destroys its scratch fields (`p`/`r`/`Ap`/`Ax`/`alpha`/`beta`) on every `mfcg(...)` call; tsl_array_n has no field-disposal mechanism at all yet, so a solver meant to run every frame (as a pressure solve would) needs its scratch allocated once, not per-call. `createConjugateGradientSolver(applyOperator, b, x, options?)` builds everything once and returns `{ solve(tol, maxiter) }` -- callers should create one solver per `(b, x)` pair they intend to reuse across frames, matching this port's established `array_utils.js`/boundary-condition-solver convention, not the source's single-call shape.
- **`applyOperator(input, output) => dispatcher`**: a factory, called by this module exactly twice (bound to `(x, Ax)` and to this solver's own `(p, Ap)` scratch) -- replaces the source's `LinearOperator`/single flexible `matvec_kernel`, since tsl_array_n kernels bind to concrete fields at construction time. Same shape as `createCopyKernel2` in `array_utils.js`.
- Generic over 1D/2D/3D shapes (matching the source's own `ti.i`/`ti.ij`/`ti.ijk` branching), capped at 3D for the usual reason (a WebGPU dispatch is inherently <=3D).
- The source's final success check compares the raw (squared) residual against `tol` directly, while every check inside the loop compares `sqrt(residual)` against `tol` -- preserved as-is (see the comment in `linalg.js` for why this is harmless for realistic tolerances, not "fixed").
- **Periodic true-residual recomputation**, added after the fact per the user's own request to align this port's design with [jet/fluid-engine-dev](https://github.com/doyubkim/fluid-engine-dev) (a local C++ reference, MIT license, already the source for several `grid/` files above) where reasonable. Neither the Python source nor Taichi's own upstream do this: every `RESIDUAL_RECOMPUTE_INTERVAL` (50) iterations, and whenever the tracked residual grows between iterations, `solve()` recomputes the true `r = b - Ax` from scratch instead of the cheaper incremental `r -= alpha*Ap` -- correcting the floating-point drift the incremental form accumulates over many iterations. Ported directly from jet's own `pcg()`; see [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) for the attribution. Every `sqrt(...)` convergence check is also now guarded with `Math.abs()` first (matching jet's own "workaround for negative zero"), since a tiny negative residual from atomic quantization noise would otherwise produce `NaN` and silently prevent convergence from ever being detected.

Verified with 9 vitest structural/validation tests, plus a live check. This project's dev sandbox has no real WebGPU adapter (falls back to WebGLBackend) and WebGPU atomics have no GLSL equivalent, so `examples/04-conjugate-gradient/` cannot even compile its shader there -- the browser console shows a vertex shader compile error (`ERROR: 0:68: '&' : syntax error`, on the WGSL pointer syntax `atomicAdd(&x, ...)` that the WebGL2 fallback's node builder never learned a GLSL translation for), a *confirmed* fallback-only failure with no ambiguity left to isolate, unlike earlier sandbox mysteries in this project. **Run by the user on real WebGPU hardware, the example converges to the expected exact answer**: solving `A=diag(1..8)`, `b=[1,...,1]`, it reports `x = [1.0000, 0.4999, 0.3335, 0.2498, 0.2001, 0.1666, 0.1429, 0.1250]` against an expected `[1.0000, 0.5000, 0.3333, 0.2500, 0.2000, 0.1667, 0.1429, 0.1250]` -- the ~1e-4 per-element deviation is consistent with the atomic dot product's fixed-point quantization noise (see `linalg.js`'s header comment), well within this example's 1e-3 comparison tolerance. Both the GPU atomic reduction and the CG iteration built on top of it are confirmed correct on real WebGPU.

### `createPreconditionedConjugateGradientSolver` -- preconditioning support

**Not a port.** This project's own `linalg.py` flags preconditioning as explicit future work in its own header comment, and Taichi Lang's own upstream `matrixfree_cg.py` (checked directly) has no preconditioned CG either -- only the plain CG already ported above, plus an unrelated BiCGSTAB solver. This is an original implementation of the standard preconditioned CG algorithm (see e.g. Shewchuk's classic "An Introduction to the Conjugate Gradient Method Without the Agonizing Pain"), sharing every architectural convention above (GPU-atomic reduction, create-once-solve-many, `applyOperator`-as-factory, 1D/2D/3D-generic, periodic true-residual recomputation) plus one new one:
- **`applyPreconditioner(input, output) => dispatcher`**: same factory idiom as `applyOperator`, called exactly once (bound to `(r, z)`, the preconditioned residual). `z` replaces `r` as the driver of `p`/`alpha`/`beta` (`p = z + beta*p`, etc.), while convergence is still checked against the *true* residual `r.r`, not `r.z` -- a preconditioner can scale `z` arbitrarily relative to `r`, so `r.z` alone isn't a faithful "how close is Ax to b" signal the way it is in the unpreconditioned case. This means one extra GPU-atomic dot product (`r.z`, alongside `r.r`) and one extra `applyPreconditioner` dispatch per iteration, versus plain CG -- a real added cost per iteration, worth it only if the preconditioner meaningfully cuts the number of iterations needed for the caller's actual `A`.

Verified with 9 vitest structural/validation tests, plus a live check. `examples/05-preconditioned-conjugate-gradient/` pairs the same `A = diag(1..8)` operator with its own *exact* Jacobi preconditioner (`M^-1 = diag(1, 1/2, ..., 1/8)`, i.e. `M = A` exactly for this diagonal system) -- a deliberately strong test case, since a perfect preconditioner makes CG converge to the exact answer in a single iteration, so any bug in how `r.z` (as opposed to `r.r`) drives `alpha`/`beta`/`p` would very likely show up as stagnation or divergence well before the 20-iteration cap. Running it in this dev sandbox confirms the exact same fallback-only shader compile failure as the plain-CG example (now for two distinct atomic-dot kernels, `r.r` and `r.z`). **Run by the user on real WebGPU hardware, it converges to an exact match** (`x = [1.0000, 0.5000, 0.3333, 0.2500, 0.2000, 0.1667, 0.1429, 0.1250]`, matching the expected answer to all 4 reported decimal places) -- even tighter than the plain-CG example's ~1e-4 deviation, consistent with this test case's single-iteration convergence accumulating far less atomic quantization noise. Both the preconditioning logic and the underlying GPU-atomic reduction are confirmed correct on real WebGPU.

### `isDegenerateDot` -- guarding against a degenerate CG denominator

Added while root-causing a real-hardware failure in `examples/14-stable-fluids/`: pressure came back 100% non-finite from the very first `project()` call, despite that call's own divergence RHS (`b`) being completely finite and well-posed. Root cause: that scene's domain has no `dirichlet` option at all (a deliberate pure-closed-box test, see that example's own section below), so the Laplacian CG solves against is a pure Neumann operator -- singular, with the constant field in its null space. CG never reduces a residual's null-space component (`A@constant=0` exactly, invisible to every dot product CG computes), so the search direction `p` can drift to be dominated by it over enough iterations -- at which point `Ap` collapses toward 0 everywhere, and `p.Ap` (`alpha`'s own denominator) heads toward an exact zero. jet's own reference `pcg()` (`cg-inl.h`, plain double-precision CPU arithmetic, no equivalent guard) would only hit a *truly* exact zero here in a rare floating-point coincidence -- but this port's own GPU-atomic dot product quantizes every dot product to a fixed-point integer before summing (see `atomicScale` above), so a small-but-nonzero `p.Ap` can round all the way down to the integer 0 well before `p` is anywhere near purely null-space-aligned: a second, much easier way to hit the same exact-zero division, unique to this port's own reduction strategy.

Fixed by `isDegenerateDot(value, scale)` (exported from `linalg.js` for direct unit-testing, since it's a pure function of two numbers, no GPU needed): a denominator whose magnitude is under half the atomic accumulator's own quantization step (`0.5/scale` -- the smallest gap between two representable readback values) is indistinguishable from an exact 0 no matter what produced it, and dividing by it risks `Infinity`/`NaN` with no way to recover. Both `solve()` functions now check every division site (`alpha`'s `pAp`, `beta`'s `oldRTr`/`oldRZ`) against this floor and break out cleanly -- keeping whatever `x` already holds -- instead of dividing by (near-)zero. This changes nothing for a well-conditioned, non-singular system with O(1)-magnitude values (examples 04/05/07's own confirmed-correct runs stay well above this floor throughout), only guards the genuinely degenerate case.

`grid_pressure_solver2.js` now also exposes `diagnostics.converged` (updated after every `project()` dispatch) and an `atomicScale` pass-through option, specifically because a divergence field's natural magnitude (`examples/14-stable-fluids/` measured `[-0.0245, 0.0485]`) is meaningfully smaller than the O(1) values `DEFAULT_ATOMIC_DOT_SCALE` was tuned against -- a caller can tell a `false` (stopped early, whether via this guard or hitting `maxIterations`) apart from a genuine bug, and retune `atomicScale` for their own problem's actual value range if early stops turn out to be frequent.

### `MAX_PAP_GROWTH_FACTOR` -- a real gap in the per-step guards, found via `examples/23-flip-moving-collider/`

`MAX_BETA_MAGNITUDE`/`MAX_ALPHA_MAGNITUDE` each bound a single iteration's own ratio; neither bounds how far `p` (and therefore `x`) has drifted from a *solve's own* starting scale across several iterations. Found by directly instrumenting `createPreconditionedConjugateGradientSolver`'s own loop (temporarily, removed after use) while root-causing real pressure-solve rejections in that example: rejected frames traced back to `pAp` swinging over the course of a single 100-iteration solve between ordinary magnitudes and values in the tens of millions -- including going *negative*, mathematically impossible for a genuinely SPD operator, and the actual signature of the atomic accumulator's own int32 encoding overflowing (`pAp * atomicScale` crossing +-2.1 billion) partway through the solve, well past where any single-iteration guard would have caught it. Fixed two ways: `MAX_PAP_GROWTH_FACTOR` (1e8) is a new, deliberately generous *ratio* check -- `pAp` compared against `initRTr`, this solve's own starting energy scale, not a shared absolute constant (this project has hit the "one global magnitude constant doesn't transfer across scenes" mistake enough times now, see `maxPlausiblePressure`'s own story below, that a scale-invariant check was worth the extra design step) -- confirmed via an explicit real-hardware A/B comparison on `examples/20-flip-dam-break/` to be safe alongside every existing scene, not just harmless in theory (45/300 rejected frames with the guard active vs. 54/300 with it disabled, on otherwise-identical code). The example's own real fix was tuning `pressure.atomicScale` down to 1 for that scene specifically (see that example's own header comment for the full investigation) -- this guard is a genuine, if smaller, additional safety margin on top, for every scene, not a substitute for correct per-scene tuning.

### `createMultigridPreconditioner` -- geometric multigrid V-cycle

Ported from [jet/fluid-engine-dev](https://github.com/doyubkim/fluid-engine-dev) (MIT license, Doyub Kim -- a local copy read directly, the same upstream already credited for several `grid/` files) at the user's explicit request to align future fluxflow work with that library where reasonable; see [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) for exactly what's ported versus generalized. A much stronger preconditioner than the toy exact-diagonal Jacobi one above -- almost certainly what the Python `fluxflow` source's own header comment anticipated ("later, this will be extended into preconditioning version of matrix free cg... for example - multi-grid preconditioning cg"). Plugs directly into `createPreconditionedConjugateGradientSolver` above with no changes to that function at all: `createMultigridPreconditioner(shape, gridSpacing, options?)` returns an `applyPreconditioner`-compatible `(input, output) => dispatcher`.

Two real differences from jet's own MGPCG, both deliberate (see `multigrid.js`'s header comment for the full reasoning):
- **Constant-coefficient, no collider**: jet's version is variable-coefficient and collider-aware, backed by an explicit per-cell matrix built from the actual domain/collider at every grid level -- building that doesn't exist anywhere in this port yet and would be a substantially larger undertaking on its own. This targets the standard constant-coefficient Poisson/Laplacian on a plain rectangular domain instead, with a fixed zero-flux (Neumann-like) boundary treatment matching `grid/grid_math.js`'s own `scalarLaplacian2` convention.
- **Dimension-generic (1D/2D/3D), not 2D-only**: jet's restriction (1/8-3/8-3/8-1/8 full-weighting) and correction (bilinear 1/4-3/4) transfer formulas are written for exactly 2 axes; this generalizes both to the outer product of the same per-axis 1D filter across however many axes `shape.length` has, matching `createConjugateGradientSolver`'s own dimension-genericity rather than hardcoding to 2D.

The V-cycle itself (relax/restrict/correct/residual) is all local-stencil kernels -- no reduction anywhere, unlike the CG solvers, so `applyPreconditioner`'s returned dispatcher is synchronous, not `async`.

**A real bug was found and fixed while verifying this**: the relax step originally divided by a single constant diagonal everywhere, but a boundary cell has fewer valid neighbors than an interior one, so its true diagonal coefficient is smaller in magnitude -- using the interior value at boundaries over-corrected every sweep and compounded into real divergence over many iterations. Fixed by computing the diagonal per-cell (`laplacianDiagonalAt`). The corrected algorithm was independently verified via a plain-JS, float64, no-GPU reference implementation of the same formula, which converges cleanly (residual norm 64 -> ~0.32 over 30 iterations on a 16x16 test case, with the expected early transient increase characteristic of red-black Gauss-Seidel from a zero initial guess, not a red flag).

Verified with 8 vitest structural tests (mirroring the CG solvers' style, including one that actually builds every level's kernels for a multi-level 2D shape), plus live checks. Verification here was unusually involved -- worth documenting the full picture:
- `examples/06-multigrid-preconditioner/` (standalone, no CG wrapper) needs no atomics at all, so it doesn't hit this project's atomic/WebGL2-fallback wall -- but the GPU dispatch pattern didn't reliably match the proven-correct JS reference above in this dev sandbox, most likely another instance of this project's well-established WebGL2-fallback unreliability (a different specific mechanism than the atomics one). **Run by the user on real WebGPU hardware**, the residual ratio after one V-cycle is `0.0256` for `numberOfLevels:1` (plain relax) versus `0.0135` for `numberOfLevels:4` (the full V-cycle) -- the multi-level version genuinely outperforming plain relaxation on this low-frequency test case confirms `restrict()`/`correct()`/the recursion itself are working, not just `relax()`.
- `examples/07-multigrid-preconditioned-cg/` (the full pipeline, a real 2D Poisson problem with a manufactured known solution rather than the 1D diagonal toy case) hits the atomic wall the same way `examples/04-`/`05-` do in this sandbox. Running it here also surfaced a genuine bug in the example's *own* verification code, worth remembering generally: `x.toArray()` came back empty (0 elements) in this sandbox, and `[].every(...)` is vacuously `true` in JavaScript regardless of the predicate -- the original check silently reported a false "converged" pass instead of the empty-readback failure it actually was. Fixed here (and defensively in `examples/04-`/`05-` too, which had the same latent issue) by checking the array length before `.every()`. **Run by the user on real WebGPU hardware, it converges to the expected exact answer** (`succeeded=true`, max deviation from `xExpected` under `1e-2`) -- the full pipeline (GPU-atomic CG reduction + multigrid preconditioner together) is confirmed correct, not just each piece in isolation.

## Current state: `time`

```js
import { time } from 'fluxflow';
```

| File | Corresponding Python source | Contents |
|---|---|---|
| `cfl.js` | (no counterpart -- ported from jet/fluid-engine-dev instead, see below) | `computeAdaptiveSubSteps(maxVelocityMagnitude, minGridSpacing, frameDt, options?)` -- CFL substep-count formula, a plain function of plain numbers with zero GPU/TSL dependency |

Solver-agnostic on purpose: this function doesn't know or care whether `maxVelocityMagnitude` came from a grid velocity field's reduction (`linalg.createMaxAbsReducer`, used by `grid.createGridAdaptiveTimeStep2`) or, eventually, a FLIP solver's own particle-velocity reduction. Confirmed this split is real, not speculative, by reading jet/fluid-engine-dev directly: `PicSolver2` (FLIP's own base class) *inherits* `GridFluidSolver2`'s `numberOfSubTimeSteps()`/`cfl()` unchanged for its own per-frame substep count, adding only a *separate* inner substep loop for particle-position integration (RK2 midpoint) on top -- a future FLIP solver here is expected to do the same: reuse this function for the outer per-frame count, add its own particle-integration substepping alongside it.

Ported formula: `cfl = maxVelocityMagnitude * frameDt / minGridSpacing`, `numSubSteps = max(1, ceil(cfl / courantNumber))`, `subDt = frameDt / numSubSteps`. `courantNumber` defaults to 5, matching jet's own `GridFluidSolver2::_maxCfl` default -- safe well above 1 specifically because this port's advection is semi-Lagrangian (unconditionally stable regardless of CFL, unlike an explicit forward-Euler scheme); a larger `courantNumber` trades advection accuracy (more interpolation smoothing per larger substep) for fewer substeps, not stability. `maxSubSteps` (default 32) is **not** in jet's own reference -- a defensive cap added here, same spirit as `linalg.js`'s `MAX_ALPHA_MAGNITUDE`/`MAX_BETA_MAGNITUDE` circuit breakers, so a velocity spike can't silently demand hundreds of substeps and stall a frame.

`grid.createGridAdaptiveTimeStep2` (`src/grid/grid_adaptive_timestep2.js`, see the `grid` section above) is the grid-specific consumer: it reduces a `FaceCenteredGrid2`'s current max velocity magnitude (`linalg.createMaxAbsReducer`), calls this function, and writes the resulting per-substep `dt` into a live `array0('float')` node shared with `createGridSolver2({ dt })`. That file's own header comment documents one deliberate simplification versus jet's own C++ loop (`PhysicsAnimation::advanceTimeStep`): jet recomputes CFL fresh before *every* substep (cheap on a synchronous CPU simulation); this port computes the substep count *once* per `update()` call and takes that many *equal* substeps instead, trading jet's finer-grained adaptivity for one GPU readback per rendered frame rather than one per substep -- this project's own CG performance investigation (`docs/perf-investigation-cg-gpu-resident-alpha-beta.md`) found GPU/CPU synchronization points to be a real, non-trivial cost on real hardware, so that tradeoff is deliberate.

Verified with pure-number vitest coverage (`test/cfl.test.js` -- no GPU needed at all, unlike almost everything else in this port) plus real-hardware confirmation via `examples/16-karman-vortex-street/`'s own adaptive-dt checkbox (see the `grid_adaptive_timestep2.js` writeup above for what was specifically checked).

## Key tradeoffs made during this port

- **No double precision**: WGSL/WebGPU compute has no native f64, so this is fixed to float(f32) only; the source's `initConstant()` orchestration (needed only to support switching precision) is therefore unnecessary too.
- **`grid_math.py`'s `@ti.func`s all became plain JS functions** (building/composing TSL node graphs) instead of `tsl_array_n.func()` -- these functions often need to return several named values (e.g. `bilinearCoordsAndWeights2` returns 8), which doesn't fit `func()`/`Fn`'s single-destructured-array calling convention.
- **`vectorGradient2`/`vectorGradientAtPosition2`'s mat2 element order was confirmed wrong on real WebGPU, and fixed.** Taichi's `tm.mat2(a,b,c,d)` is row-major (row0=(a,b), row1=(c,d)); TSL's `mat2(a,b,c,d)` turned out to be column-major (column0=(a,b), column1=(c,d)) -- a direct argument-order translation of the source produced a transposed Jacobian. Confirmed live via `examples/00-grid-math/` on real WebGPU hardware and fixed by swapping the middle two constructor arguments; see the comment in `grid_math.js`.
- **Kernels bind to a concrete field at construction time**, with no support for "call the same kernel rebound to a different field" -- as a result, `grid_blocked_boundary_condition_solver2.js`'s API shape deliberately diverges from the source: the constructor takes a fixed `velocity` (FaceCenteredGrid2) directly, and `constrainVelocity()` no longer takes velocity as a per-call argument the way the source does. The collider, however, can still be swapped mid-lifetime (`setCollider()`), which rebuilds every collider-dependent kernel when it's called.
- **Zero new dependencies for the SDF collider's polygon/SVG rasterization**: `addPolygon`/`addPolygons` use hand-rolled point-in-polygon + point-to-boundary distance (matching the semantics of shapely's `boundary.distance`+`contains`/`touches`); multiple shapes are combined via SDF pointwise min (no real polygon boolean union needed); `addSvg` samples along the path using the browser's native `SVGPathElement.getPointAtLength()`, instead of the Python-only `svg.path`.
- **The `VertexCentered*` grids' dataSize doesn't carry over the source's "keep (0,0) when resolution=(0,0)" defensive branch** -- `tsl_array_n.array2()` itself rejects zero-length dimensions, and nothing in `grid/` actually exercises that branch (confirmed via grep).
- **A storage field needs exactly one permanent writer kernel on this project's dev sandbox (WebGL2 fallback)** -- found while building `examples/12-interactive-advection/`, which originally ping-ponged dye between two `ScalarGrid2`s the same way `array_utils.js`'s `createExtrapolateToRegion2` ping-pongs its own `outputField` (two fixed-direction `tsl_array_n.kernel()`s both writing the same field, alternated by parity). That pattern reliably throws `TypeError: dualAttributeData.switchBuffers is not a function` from inside `WebGLBackend.compute()` once *both* writer kernels have each dispatched at least once. Three.js's WebGL2-fallback compute emulation (`WebGLAttributeUtils.js`'s `DualAttributeData`, a transform-feedback ping-pong buffer pair) evidently assumes each storage buffer has a single owning compute pipeline as its write target; a second, independently-built pipeline that also writes it ends up with something other than a `DualAttributeData` in its own `transformBuffers` list. Restructured `examples/12-interactive-advection/` to give every field exactly one permanent writer (four dye fields instead of two: two "freshly advected" scratch fields plus two "current state" fields, each written by a single dedicated kernel) -- no crash across dozens of frames after that change. Whether this is fixable, a real three.js bug, or just a hard rule to design around on this backend is undetermined; `createExtrapolateToRegion2` technically has the same two-writers-one-field shape but has apparently never been driven through a real dispatch loop that would trigger it. Worth checking for if a future ping-pong design on this backend throws the same error.
- **`frictionCoefficient`/`closedDomainBoundaryFlag` are plain mutable properties** (`collider.frictionCoefficient = x`); the corresponding setter methods in the source (`setFriectionCoefficient`/`setClosedDomainBoundaryFlag`) weren't carried over, which is more natural plain-JS idiom. **One thing to watch for**: everywhere `frictionCoefficient` gets read inside a kernel, its value is baked into the node graph as a constant at kernel **build** time, not re-read on every dispatch -- this matches the source's own Taichi-side behavior (reading a plain Python attribute inside a Taichi `@ti.kernel` is also compile-time-constant-folded, not a new limitation introduced by this port), but if "change the friction coefficient at runtime and have an already-built kernel pick it up immediately" is ever needed, it has to become an `array0`/`uniform` instead. `SDFRigidBodyCollider2.velocityAt()` (which reads `currentPosition`/`linearVelocity`) has the same architectural limitation -- see the detailed comment in `sdf_collider2.js`.

## Two-phase: confirmed on real WebGPU, and the three bugs that found

`grid_two_phase_flip_solver2.js` and its `faceWeights` support were written in an environment with no
WebGPU adapter, then run for real. Recording this in full because the outcome is the whole argument for
why "the tests pass" is not the same as "it works": **the plain-JS reference tests were all green, and the
solver was still broken in three separate ways, every one of them invisible to a structural test.**

1. **12 storage buffers in one compute stage; the guaranteed limit is 8.** The donor-claiming kernel bound
   two count grids, two donor pools, four atomic cursors, positions, velocities and both velocity
   components. `maxStorageBuffersPerShaderStage` is 8, so pipeline creation was rejected outright and the
   kernel silently did nothing. Plenty of real hardware raises that limit, which is exactly what makes it
   dangerous — it is a portability bug that a good GPU hides. Fixed by folding the four cursors into one
   array, the two donor pools into one array filled from both ends, and moving the eligibility test and
   phase choice into their own kernel: 12 bindings down to 7.
2. **`select()` over an `atomicLoad()` result generates invalid code.** This produced
   `THREE.TSL: Invalid generated code, expected a "int"`, and it was isolated with a standalone probe
   rather than guessed at, because the construct looks completely ordinary. `pick.select( atomicLoad( a ),
   atomicLoad( b ) )` emits errors; the same thing with `.toInt()` on each side is clean. Values derived
   from a bad select inherit the problem, which is why two such selects produced four errors. The fix
   resolves the phase choice with a real `If`/`Else` instead, so no `select` touches an atomic at all.
   Note how narrow this is: an atomic result in a *comparison* is fine, and an `atomicAdd` result used as
   an index into a *non-atomic* array is fine — both are things `grid_flip_solver2.js` has always done.
3. **The pressure solve blew up at any real density ratio.** With the two bugs above fixed the solver ran,
   but `converged` was false every frame and `rejected` true from frame 60 — meaning the projection was
   being thrown away and the velocity field never actually projected. Bisecting on the density ratio was
   decisive: at 1:1 (`beta` = 1 everywhere) it converged with sane pressure, at 1:100 pressure went
   1023-cells-out-of-1024 non-finite within ten frames. Two causes, both configuration rather than
   formulation:
   - `pressure.atomicScale`. `Ap` carries a factor of `beta`, and `beta` **is** the density ratio, so the
     safe fixed-point scale shrinks in proportion to it. Measured on the same scene: 256 → non-finite by
     frame 10, 16 → non-finite by frame 19, **1 → stable**. Start at roughly the library default divided
     by the density ratio.
   - `multigrid.numberOfLevels`. The pressure solver's default of 1 is plain relaxation with no
     coarse-grid correction, which every single-phase scene here gets away with and a variable-coefficient
     system does not. The two-phase solver now derives its own default from the resolution (up to 4 levels,
     backing off so it can never reject an odd grid size) rather than inheriting that one.

After those fixes, `examples/24-two-phase-bubble-rise/` was run for **780 frames** at 64x64 with 16384
particles: `rejected=false` on every single frame, no non-finite value at any point, velocities bounded
throughout (peak ~27, settling to ~4-8), and mean gas height climbing 49.8 → 53.7 before flattening — the
bubble rising, breaking the surface, merging into the air layer, and the tank settling. A separate
bubble-tracking run measured the bubble particles specifically: mean height 4.34 → 17.32, i.e. from the
floor to the water line, monotonically. Nothing anywhere applies a buoyancy force.

`converged` does read false on most frames and that is expected here rather than a fault: the multigrid
preconditioner is still constant-coefficient (see `multigrid.js` decision 4), so it preconditions this
system less well the higher the density ratio goes. `rejected` is the diagnostic that matters, and it
stayed false throughout.

Still open, and honestly so: this was run on a software adapter (SwiftShader), so it confirms correctness
and stability but says nothing useful about performance; and the longest run so far is 780 frames, where
`examples/20-flip-dam-break/`'s own particle-drift problem only became visible after ~700.

## Verified on real WebGPU

Both `examples/00-grid-math/` (`grid_math.js` numeric verification) and `examples/01-boundary-condition/` (boundary-condition-solver dispatch smoke test) have now been run on the user's real desktop browser (`init()` reporting `backend: WebGPUBackend`, not a fallback):

- `00-grid-math/`: bilinear interpolation and `scalarGradient2` both read back exactly the expected values. `vectorGradient2`'s mat2 test was the one genuine bug this surfaced (see above) -- now fixed and green.
- `01-boundary-condition/`: the full pipeline (collider rasterization -> constructing the boundary solver -> `constrainVelocity()` genuinely dispatching a whole set of kernels -> switching via `setCollider(null,...)`) runs end to end without throwing and reads back plausible values (e.g. the closed left domain boundary correctly zeroed, the rest matching the seeded uniform inflow). This is a real dispatch on real hardware, not just "doesn't throw" -- though the small hand-checked sample isn't an exhaustive proof of the solver's full physical correctness across every branch (no-flux projection, extrapolation, blocked boundaries) either.

This also confirms the dev sandbox's own limitation (no real WebGPU adapter, `init()` falls back to `WebGLBackend`) was exactly that -- a fallback-only artifact, not a bug in this port. In-sandbox, the first two `00-grid-math/` tests read back correctly but the third read back all zeros; investigated at the time to "reading a different field that already has data, from inside a kernel" not working on that fallback backend, regardless of whether the data came from `fromArray()` or another kernel. That's now the fourth confirmed instance of a fallback-only limitation in this project (after the `Loop()` counter, `array0` multi-thread shared reads, and this port's own GPU-round-trip self-touch case in `examples/02-flow-around-shape/`).

## Dependencies

- [tsl_array_n](../tsl_array_n) (peerDependency, linked to the local package within this workspace)
- [three.js](https://threejs.org/) `>=0.180.0` (peerDependency)

## Development

```bash
npm test -w fluxflow         # vitest -- graph construction / pure-CPU geometry / hook orchestration, no real GPU needed
npm run dev -w fluxflow      # vite dev server, runs examples/
```

## License

[Apache License 2.0](LICENSE) © 2026 bert wang -- matches the license of the Python `fluxflow` project this was ported from.

`src/grid/`, `src/noise/`, and `src/linalg/` are ported from a separate project (`D:\OneDrive\04_lib_fluxflow`, also Apache License 2.0), parts of which trace further back to [fluid-engine-dev](https://github.com/doyubkim/fluid-engine-dev) (MIT), [WebGL-Noise](https://github.com/ashima/webgl-noise) (MIT), and [Taichi Lang](https://github.com/taichi-dev/taichi) (Apache-2.0) respectively -- see [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) for the full provenance chain and per-file mapping.
