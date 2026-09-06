# Third-Party Notices

fluxflow does not vendor or bundle any third-party source code as a runtime
dependency — the published package would ship only the files under `src/`.
This notice covers two things: (1) third-party packages fluxflow requires or
uses during development, and (2) the provenance of `src/grid/` and
`src/noise/`, both of which are ports of code from other projects, not
written from scratch.

## Runtime dependency

### three.js

- **Relationship:** required peer dependency (`three >= 0.180.0`), same as `tsl_array_n`. Not bundled.
- **License:** MIT
- **Homepage:** https://threejs.org/
- **Repository:** https://github.com/mrdoob/three.js
- **License text:** https://github.com/mrdoob/three.js/blob/dev/LICENSE

## Development-only dependency

Used to develop and preview this package's examples and tests; never published
or distributed as part of the `fluxflow` package.

### Vite

- **License:** MIT
- **Homepage:** https://vite.dev/
- **Repository:** https://github.com/vitejs/vite
- **License text:** https://github.com/vitejs/vite/blob/main/LICENSE

### Vitest

- **License:** MIT
- **Homepage:** https://vitest.dev/
- **Repository:** https://github.com/vitest-dev/vitest
- **License text:** https://github.com/vitest-dev/vitest/blob/main/LICENSE

## Provenance of `src/grid/`

`src/grid/` is a JavaScript/TSL port of the `grid/` folder from a separate,
private Python project of the same author, **fluxflow (Python)**, located at
`D:\OneDrive\04_lib_fluxflow` — not the same repository as this one, despite
the shared name. That project is itself licensed under the **Apache License
2.0**, and its own `THIRD-PARTY-NOTICES.txt` records that several of the files
being ported here were themselves originally ported into that project from
**fluid-engine-dev**, a C++ library, under the **MIT License**. So the actual
chain for most of this folder is:

```
fluid-engine-dev (C++, MIT, Doyub Kim)
  -> fluxflow (Python/Taichi, Apache-2.0, bert wang)
    -> fluxflow (this package, JS/TSL, Apache-2.0, bert wang)
```

This package is itself licensed under the Apache License 2.0, same as its
Python counterpart — see `LICENSE`.

Every file below carries a header comment in its `src/` form crediting the
Python file it was ported from and describing what changed, satisfying the
Apache License's "state that you changed the file" condition at the file
level; this notice covers the rest.

### fluid-engine-dev (MIT) — via fluxflow (Python)

- **Source:** https://github.com/doyubkim/fluid-engine-dev
- **License:** MIT

Files in this package derived from `fluid-engine-dev` (through the Python
`fluxflow` project's own `grid/constant.py`, `grid_data2.py`, `grid_math.py`,
`grid_solver2.py`, `sdf_collider2.py` — exactly the set that Python project's
own `THIRD-PARTY-NOTICES.txt` lists under this same upstream):

- `src/grid/constant.js`
- `src/grid/grid_data2.js`
- `src/grid/grid_math.js`
- `src/grid/grid_solver2.js`
- `src/grid/sdf_collider2.js`

> The MIT License (MIT)
>
> Copyright (c) 2018 Doyub Kim
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

### fluxflow (Python) (Apache License 2.0)

- **Source:** `D:\OneDrive\04_lib_fluxflow` (private, same author, separate repository)
- **License:** Apache License 2.0

All of `src/grid/` and `src/noise/` is ported from that project (the five
`grid/` files above, plus `array_utils.py`, `level_set_utils.py`,
`grid_blocked_boundary_condition_solver2.py`, and `noise/noise.py` — the
first three of those are not separately attributed to `fluid-engine-dev` in
that project's own notices and are treated here as that project's own
original work; `noise/noise.py` has its own separate upstream, WebGL-Noise,
covered in its own section below). `src/grid/polygon_sdf.js` and
`src/grid/svg_utils.js` have no Python counterpart — they're new code written
for this port to replace two Python-only dependencies (`shapely`, `svg.path`)
that have no browser equivalent; see `sdf_collider2.js`'s header comment for
the approach.

>                                  Apache License
>                            Version 2.0, January 2004
>                         http://www.apache.org/licenses/
>
>    TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
>
>    1. Definitions.
>
>       "License" shall mean the terms and conditions for use, reproduction,
>       and distribution as defined by Sections 1 through 9 of this document.
>
>       "Licensor" shall mean the copyright owner or entity authorized by
>       the copyright owner that is granting the License.
>
>       "Legal Entity" shall mean the union of the acting entity and all
>       other entities that control, are controlled by, or are under common
>       control with that entity. For the purposes of this definition,
>       "control" means (i) the power, direct or indirect, to cause the
>       direction or management of such entity, whether by contract or
>       otherwise, or (ii) ownership of fifty percent (50%) or more of the
>       outstanding shares, or (iii) beneficial ownership of such entity.
>
>       "You" (or "Your") shall mean an individual or Legal Entity
>       exercising permissions granted by this License.
>
>       "Source" form shall mean the preferred form for making modifications,
>       including but not limited to software source code, documentation
>       source, and configuration files.
>
>       "Object" form shall mean any form resulting from mechanical
>       transformation or translation of a Source form, including but
>       not limited to compiled object code, generated documentation,
>       and conversions to other media types.
>
>       "Work" shall mean the work of authorship, whether in Source or
>       Object form, made available under the License, as indicated by a
>       copyright notice that is included in or attached to the work
>       (an example is provided in the Appendix below).
>
>       "Derivative Works" shall mean any work, whether in Source or Object
>       form, that is based on (or derived from) the Work and for which the
>       editorial revisions, annotations, elaborations, or other modifications
>       represent, as a whole, an original work of authorship. For the purposes
>       of this License, Derivative Works shall not include works that remain
>       separable from, or merely link (or bind by name) to the interfaces of,
>       the Work and Derivative Works thereof.
>
>       "Contribution" shall mean any work of authorship, including
>       the original version of the Work and any modifications or additions
>       to that Work or Derivative Works thereof, that is intentionally
>       submitted to Licensor for inclusion in the Work by the copyright owner
>       or by an individual or Legal Entity authorized to submit on behalf of
>       the copyright owner. For the purposes of this definition, "submitted"
>       means any form of electronic, verbal, or written communication sent
>       to the Licensor or its representatives, including but not limited to
>       communication on electronic mailing lists, source code control systems,
>       and issue tracking systems that are managed by, or on behalf of, the
>       Licensor for the purpose of discussing and improving the Work, but
>       excluding communication that is conspicuously marked or otherwise
>       designated in writing by the copyright owner as "Not a Contribution."
>
>       "Contributor" shall mean Licensor and any individual or Legal Entity
>       on behalf of whom a Contribution has been received by Licensor and
>       subsequently incorporated within the Work.
>
>    2. Grant of Copyright License. Subject to the terms and conditions of
>       this License, each Contributor hereby grants to You a perpetual,
>       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
>       copyright license to reproduce, prepare Derivative Works of,
>       publicly display, publicly perform, sublicense, and distribute the
>       Work and such Derivative Works in Source or Object form.
>
>    3. Grant of Patent License. Subject to the terms and conditions of
>       this License, each Contributor hereby grants to You a perpetual,
>       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
>       (except as stated in this section) patent license to make, have made,
>       use, offer to sell, sell, import, and otherwise transfer the Work,
>       where such license applies only to those patent claims licensable
>       by such Contributor that are necessarily infringed by their
>       Contribution(s) alone or by combination of their Contribution(s)
>       with the Work to which such Contribution(s) was submitted. If You
>       institute patent litigation against any entity (including a
>       cross-claim or counterclaim in a lawsuit) alleging that the Work
>       or a Contribution incorporated within the Work constitutes direct
>       or contributory patent infringement, then any patent licenses
>       granted to You under this License for that Work shall terminate
>       as of the date such litigation is filed.
>
>    4. Redistribution. You may reproduce and distribute copies of the
>       Work or Derivative Works thereof in any medium, with or without
>       modifications, and in Source or Object form, provided that You
>       meet the following conditions:
>
>       (a) You must give any other recipients of the Work or
>           Derivative Works a copy of this License; and
>
>       (b) You must cause any modified files to carry prominent notices
>           stating that You changed the files; and
>
>       (c) You must retain, in the Source form of any Derivative Works
>           that You distribute, all copyright, patent, trademark, and
>           attribution notices from the Source form of the Work,
>           excluding those notices that do not pertain to any part of
>           the Derivative Works; and
>
>       (d) If the Work includes a "NOTICE" text file as part of its
>           distribution, then any Derivative Works that You distribute must
>           include a readable copy of the attribution notices contained
>           within such NOTICE file, excluding those notices that do not
>           pertain to any part of the Derivative Works, in at least one
>           of the following places: within a NOTICE text file distributed
>           as part of the Derivative Works; within the Source form or
>           documentation, if provided along with the Derivative Works; or,
>           within a display generated by the Derivative Works, if and
>           wherever such third-party notices normally appear. The contents
>           of the NOTICE file are for informational purposes only and
>           do not modify the License. You may add Your own attribution
>           notices within Derivative Works that You distribute, alongside
>           or as an addendum to the NOTICE text from the Work, provided
>           that such additional attribution notices cannot be construed
>           as modifying the License.
>
>       You may add Your own copyright statement to Your modifications and
>       may provide additional or different license terms and conditions
>       for use, reproduction, or distribution of Your modifications, or
>       for any such Derivative Works as a whole, provided Your use,
>       reproduction, and distribution of the Work otherwise complies with
>       the conditions stated in this License.
>
>    5. Submission of Contributions. Unless You explicitly state otherwise,
>       any Contribution intentionally submitted for inclusion in the Work
>       by You to the Licensor shall be under the terms and conditions of
>       this License, without any additional terms or conditions.
>       Notwithstanding the above, nothing herein shall supersede or modify
>       the terms of any separate license agreement you may have executed
>       with Licensor regarding such Contributions.
>
>    6. Trademarks. This License does not grant permission to use the trade
>       names, trademarks, service marks, or product names of the Licensor,
>       except as required for reasonable and customary use in describing the
>       origin of the Work and reproducing the content of the NOTICE file.
>
>    7. Disclaimer of Warranty. Unless required by applicable law or
>       agreed to in writing, Licensor provides the Work (and each
>       Contributor provides its Contributions) on an "AS IS" BASIS,
>       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
>       implied, including, without limitation, any warranties or conditions
>       of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
>       PARTICULAR PURPOSE. You are solely responsible for determining the
>       appropriateness of using or redistributing the Work and assume any
>       risks associated with Your exercise of permissions under this License.
>
>    8. Limitation of Liability. In no event and under no legal theory,
>       whether in tort (including negligence), contract, or otherwise,
>       unless required by applicable law (such as deliberate and grossly
>       negligent acts) or agreed to in writing, shall any Contributor be
>       liable to You for damages, including any direct, indirect, special,
>       incidental, or consequential damages of any character arising as a
>       result of this License or out of the use or inability to use the
>       Work (including but not limited to damages for loss of goodwill,
>       work stoppage, computer failure or malfunction, or any and all
>       other commercial damages or losses), even if such Contributor
>       has been advised of the possibility of such damages.
>
>    9. Accepting Warranty or Additional Liability. While redistributing
>       the Work or Derivative Works thereof, You may choose to offer,
>       and charge a fee for, acceptance of support, warranty, indemnity,
>       or other liability obligations and/or rights consistent with this
>       License. However, in accepting such obligations, You may act only
>       on Your own behalf and on Your sole responsibility, not on behalf
>       of any other Contributor, and only if You agree to indemnify,
>       defend, and hold each Contributor harmless for any liability
>       incurred by, or claims asserted against, such Contributor by reason
>       of your accepting any such warranty or additional liability.
>
>    END OF TERMS AND CONDITIONS

### jet/fluid-engine-dev (MIT) — `src/grid/advection_solver2.js`, direct, no Python intermediary

`src/grid/advection_solver2.js` (semi-Lagrangian advection with monotonic
cubic interpolation) has no Python source to port from at all -- the
Python `fluxflow` project's own `grid_solver2.py` never got past an
abstract `computeAdvection` hook (see `grid_solver2.js`'s own header
comment). This is instead read directly from **jet/fluid-engine-dev** (the
same local copy already used for the `linalg/` derivations) --
`semi_lagrangian2.h`/`.cpp` (the back-trace algorithm, including its
boundary handling), `math_utils.h`'s `monotonicCatmullRom` (Fedkiw, Stam &
Jensen's clamped-Catmull-Rom scheme, "Visual Simulation of Smoke",
SIGGRAPH 2001), and `array_samplers2-inl.h`'s `CubicArraySampler2` (the 2D
tensor-product structure built on top of the 1D formula). Same direct
chain as the `linalg/` entries above (no Python intermediary):

```
fluid-engine-dev (C++, MIT, Doyub Kim)
  -> fluxflow (this package, JS/TSL, Apache-2.0, bert wang)
```

- **Source:** https://github.com/doyubkim/fluid-engine-dev/blob/master/include/jet/semi_lagrangian2.h (back-trace + boundary handling), .../math_utils.h (monotonicCatmullRom), .../detail/array_samplers2-inl.h (CubicArraySampler2)
- **License:** MIT — full text already reproduced above, under "fluid-engine-dev (MIT) — via fluxflow (Python)"; not repeated a second time.

What carries over essentially unchanged: the back-trace algorithm
(adaptive-substep RK2 midpoint integration, with the boundary-crossing
clamp built directly into it) and the `monotonicCatmullRom` formula
itself, including its exact monotonicity clamp. What's new here (this
project's own code, not jet's): the `Loop()`/`Break()`-based
bounded-iteration translation of jet's dynamic-count `while` loop
(tsl_array_n has no native while-loop), and folding jet's
base-class/override-class split (`SemiLagrangian2` vs.
`CubicSemiLagrangian2`) into a single factory function, matching this
whole port's factory-over-inheritance convention.

### mantaflow (Apache License 2.0) — inflow/outflow, `src/grid/sdf_inflow_outflow2.js` + `src/grid/grid_outflow_solver2.js`

At the user's own explicit request: inflow/outflow as first-class,
SDF-based scene objects (`createSDFInflow2`/`createSDFOutflow2`), and what
their presence actually does to the simulation each frame, is derived from
**mantaflow** (https://github.com/thunil/mantaflow, Copyright 2018 the
mantaflow team, Tobias Pfaff & Nils Thuerey), fetched and read directly via
the GitHub API (no local checkout in this repo). Same license as this
whole package (Apache License 2.0), so no second license-text block is
needed:

```
mantaflow (C++, Apache License 2.0, Tobias Pfaff & Nils Thuerey)
  -> fluxflow (this package, JS/TSL, Apache-2.0, bert wang)
```

- **Source:** https://github.com/thunil/mantaflow/blob/master/source/plugin/extforces.cpp (`setOpenBound`, `resetOutflow`, `setInflowBcs`/`KnSetInflow`), .../pressure.cpp (`MakeRhs`, `knCorrectVelocity`), .../advection.cpp (`getBulkVel`, `extrapolateVelConvectiveBC`, `copyChangedVels`, `applyOutflowBC`)
- **License:** Apache License 2.0 — full text already reproduced above, under "fluxflow (Python) (Apache License 2.0)"; not repeated a second time.

mantaflow's own outflow treatment turned out to be three genuinely
independent mechanisms (confirmed by reading all three source files
directly, after an initial single-file read had suggested it was just one)
— each is credited separately below, since what carries over differs for
each:

1. **Pressure** (`extforces.cpp`'s `setOpenBound`, `pressure.cpp`'s
   `MakeRhs`/`knCorrectVelocity`): the *concept* carries over --
   non-fluid cells get no pressure equation of their own, and any
   neighboring fluid cell treats that neighbor's pressure as exactly 0.
   No literal code carries over at all: this port's own pre-existing,
   more general Dirichlet-pressure mechanism (`grid_pressure_solver2.js`'s
   `dirichlet`, built on `multigrid.js`'s `dirichletMask`) already
   reproduces this exactly, with zero changes to the pressure solver
   itself -- see `sdf_inflow_outflow2.js`'s own header comment for the
   full derivation.
2. **Velocity** (`advection.cpp`'s `getBulkVel`/
   `extrapolateVelConvectiveBC`/`copyChangedVels`/`applyOutflowBC`): the
   convective/radiation boundary condition *formula* carries over fairly
   directly -- averaging a local "bulk velocity", then extrapolating via
   `(vel - velPrev) / factor + vel(upstream neighbor)` where
   `factor = timeStep * max(1, bulkVel[component])`. A detail easy to miss
   on a first read, confirmed real by re-reading `advection.cpp` a second
   time directly at the exact call site: mantaflow's own `applyOutflowBC`
   does NOT pass its own `timeStep` argument straight through into
   `extrapolateVelConvectiveBC` -- it passes `max(1.0, timeStep*4)`
   instead, so `factor`'s own "timeStep" is a floor of 1.0 for any
   ordinary (CFL-bounded, i.e. well under 0.25) simulation dt, not the raw
   dt itself. This port's own `grid_outflow_solver2.js` (`
   OUTFLOW_TIMESTEP_FLOOR`/`OUTFLOW_TIMESTEP_SCALE`) reproduces this exact
   floor -- an earlier version used the raw simulation dt directly, which
   is NOT a faithful port of this formula despite looking like one, and
   was confirmed as a real, independent contributor to a real-hardware
   divergence (a raw dt of ~0.033 amplifies `vel - velPrev` roughly 30x
   more than mantaflow's own floor allows). Adapted from mantaflow's own
   axis-aligned, per-cell-index neighbor search (built for its own
   always-a-literal-wall outflow regions) to this port's SDF-based,
   arbitrarily-oriented outflow objects: the "upstream" direction is read
   from the outflow SDF's own gradient (the same technique
   `grid_blocked_boundary_condition_solver2.js`'s existing
   `noFluxProjectionU`/`noFluxProjectionV` already use for collider
   normals) instead of a fixed compass direction, and both the bulk
   velocity and the upstream neighbor's velocity are read via
   `FaceCenteredGrid2`'s own existing bilinear `.sample(pos)`
   (`grid_data2.js`) instead of mantaflow's own discrete stencil
   averaging.
3. **Scalar-field cleanup** (`extforces.cpp`'s `resetOutflow`): the
   concept (zero a scalar field within outflow cells every step) carries
   over. What's dropped: everything specific to mantaflow's own
   particle/level-set machinery (`resetOutflow` also kills FLIP/PIC
   particles and resets a level-set `phi` -- this port has neither, being
   a pure Eulerian grid method with no free-surface tracking). What's
   generalized: `clearOutflowScalarField(scalarGrid)` accepts *any*
   caller-supplied `ScalarGrid2` (dye, in every existing example -- never
   a library-owned concept in this port) rather than one hardcoded
   density field.

Two deliberate extensions beyond mantaflow's own literal mechanism, for
both inflow and outflow: SDF-shaped (not wall-only) placement -- an inflow
object can represent a source anywhere in the domain (e.g. a circular
"fountain"), not just a literal wall, and an outflow object can cover an
arbitrary-shaped portion of a wall, not "the whole wall or nothing" the
way mantaflow's own `openBound` string works; and inflow's settable
`mode` (`'set'`, matching mantaflow's own `KnSetInflow` hard-override
behavior, or `'add'`, superimposing onto whatever velocity is already
there -- requested directly by the user, with no equivalent in
mantaflow's own literal code).

### mantaflow (Apache License 2.0) — symmetric multigrid relaxation, `src/linalg/multigrid.js`

A second, separate consultation of mantaflow (same project as above,
different source file), at the user's own explicit suggestion, while
tracking down a long-run divergence that only manifested once a Dirichlet
pressure mask was active over many frames (first surfaced via the
inflow/outflow mechanism above). No code is copied from mantaflow here --
what carries over is a design principle, confirmed by reading mantaflow's
own multigrid solver directly:

```
mantaflow (C++, Apache License 2.0, Tobias Pfaff & Nils Thuerey)
  -> fluxflow (this package, JS/TSL, Apache-2.0, bert wang)
```

- **Source:** https://github.com/thunil/mantaflow/blob/master/source/multigrid.cpp (`GridMg::doVCycle`, `GridMg::smoothGS`), .../multigrid.h (`GridMg::setSmoothing`)
- **License:** Apache License 2.0 — full text already reproduced above, under "fluxflow (Python) (Apache License 2.0)"; not repeated a second time.

`GridMg::doVCycle` calls its own `smoothGS(level, reversedOrder)` with
`reversedOrder=false` for every pre-smoothing repetition and `true` for
every post-smoothing one, specifically so the two-color (red-black)
sweep's own color order reverses between them -- keeping the whole
V-cycle a symmetric linear operator, a standard requirement for using
multigrid as a preconditioner inside conjugate gradient (an asymmetric
preconditioner can produce a search direction that grows without bound
across CG iterations, confirmed directly on real hardware as this port's
own actual root cause). `multigrid.js`'s own `relax()` had no such
reversal before this -- see that function's own header comment for the
full investigation and fix. Only the *principle* (reverse color order for
symmetry) carries over; the implementation is this port's own, adapted to
its own dimension-generic (1D/2D/3D), non-Galerkin multigrid design,
which has no direct structural equivalent to mantaflow's own
topology-aware coarse-grid generation.

## Provenance of `src/noise/`

`src/noise/noise.js` is a JavaScript/TSL port of `noise/noise.py` from the
same Python `fluxflow` project referenced above (Apache License 2.0). That
file's own header comment records that it is itself ported from
**WebGL-Noise**, under the **MIT License**:

```
WebGL-Noise (C++/GLSL, MIT, Ashima Arts & Stefan Gustavson)
  -> fluxflow (Python/Taichi, Apache-2.0, bert wang)
    -> fluxflow (this package, JS/TSL, Apache-2.0, bert wang)
```

### WebGL-Noise (MIT) — via fluxflow (Python)

- **Source:** https://github.com/ashima/webgl-noise
- **License:** MIT

Files in this package derived from WebGL-Noise (through the Python
`fluxflow` project's `noise/noise.py`):

- `src/noise/noise.js`

> Copyright (C) 2011 by Ashima Arts (Simplex noise)
>
> Copyright (C) 2011-2016 by Stefan Gustavson (Classic noise and others)
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in
> all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
> THE SOFTWARE.

This file is also, like everything under `src/grid/`, a derivative work of
the Python `fluxflow` project itself (Apache License 2.0) — see that
project's own license text already reproduced above, under "fluxflow
(Python) (Apache License 2.0)"; that section's terms apply here too, this
isn't repeated a second time.

## Provenance of `src/linalg/`

`src/linalg/linalg.js`'s `createConjugateGradientSolver` function is a
JavaScript/TSL port of `linalg/linalg.py`'s `mfcg` function, from the same
Python `fluxflow` project (Apache License 2.0). That file's own header
comment says it is "almost identical to the original Taichi source code",
ported directly from **Taichi Lang** itself
(`python/taichi/linalg/matrixfree_cg.py`), also under the **Apache License
2.0** — so, unlike `fluid-engine-dev` and `WebGL-Noise` above, this is a
same-license derivation (Taichi's own Apache-2.0 code, via a project that is
itself Apache-2.0), not a second license to reproduce:

```
Taichi Lang (Python, Apache-2.0, the Taichi team)
  -> fluxflow (Python/Taichi, Apache-2.0, bert wang)
    -> fluxflow (this package, JS/TSL, Apache-2.0, bert wang)
```

- **Source:** https://github.com/taichi-dev/taichi/blob/master/python/taichi/linalg/matrixfree_cg.py
- **License:** Apache License 2.0 — full text already reproduced above, under "fluxflow (Python) (Apache License 2.0)"; not repeated a second time.

Functions in this package derived from Taichi Lang (through the Python
`fluxflow` project's `linalg/linalg.py`):

- `src/linalg/linalg.js`'s `createConjugateGradientSolver`

The same file's `createPreconditionedConjugateGradientSolver` is **not**
part of this derivation — checked directly, neither the Python `fluxflow`
project (whose own header comment flags preconditioning as future work)
nor Taichi Lang's actual upstream `matrixfree_cg.py` (which has only plain
CG and an unrelated BiCGSTAB solver) have a preconditioned CG to derive
from. It's an original implementation of the standard, textbook
preconditioned CG algorithm, built on the same GPU-atomic-reduction
infrastructure as the function above; see that function's own header
comment in `linalg.js` for the detailed reasoning.

### jet/fluid-engine-dev (MIT) — direct, no Python intermediary

Both `solve()` functions' periodic true-residual recomputation (see
`RESIDUAL_RECOMPUTE_INTERVAL`'s own comment in `linalg.js`) is ported
directly from **jet/fluid-engine-dev**'s own `pcg()`
(`include/jet/detail/cg-inl.h`) — a local copy at
`D:\OneDrive\02_library\cpp\jet\fluid-engine-dev` was read directly for
this. Unlike the `grid/` files above (which derive from fluid-engine-dev
*through* the Python `fluxflow` project, which had already ported them),
this one skips the Python intermediary entirely, the same way the Taichi
Lang derivation above does:

```
fluid-engine-dev (C++, MIT, Doyub Kim)
  -> fluxflow (this package, JS/TSL, Apache-2.0, bert wang)
```

- **Source:** https://github.com/doyubkim/fluid-engine-dev/blob/master/include/jet/detail/cg-inl.h
- **License:** MIT — full text already reproduced above, under "fluid-engine-dev (MIT) — via fluxflow (Python)"; not repeated a second time.

Only this one technique is derived from jet here — the surrounding
solve() functions (variable naming, factory shape, GPU-atomic reduction,
etc.) are this port's own, as described above.

### jet/fluid-engine-dev (MIT) — `src/linalg/multigrid.js`

`src/linalg/multigrid.js`'s geometric multigrid V-cycle preconditioner is
also read directly from jet/fluid-engine-dev (the same local copy as
above) — `include/jet/mg.h`, `detail/mg-inl.h`, `fdm_mg_solver2.cpp`,
`fdm_mg_linear_system2.cpp`, and `fdm_gauss_seidel_solver2.cpp`. Same
direct chain as immediately above (no Python intermediary):

```
fluid-engine-dev (C++, MIT, Doyub Kim)
  -> fluxflow (this package, JS/TSL, Apache-2.0, bert wang)
```

- **Source:** https://github.com/doyubkim/fluid-engine-dev/blob/master/include/jet/mg.h (V-cycle), .../fdm_mg_linear_system2.cpp (restriction/correction formulas), .../fdm_gauss_seidel_solver2.cpp (red-black relax formula)
- **License:** MIT — full text already reproduced above, under "fluid-engine-dev (MIT) — via fluxflow (Python)"; not repeated a second time.

What's ported directly, and what's this port's own generalization (both
described in full in `multigrid.js`'s own header comment):

- The V-cycle structure (relax / restrict / recurse / correct / relax
  again) and the red-black relax formula carry over essentially unchanged.
- The restriction (separable 1/8-3/8-3/8-1/8 full-weighting) and
  correction (bilinear 1/4-3/4) transfer formulas are jet's own, but
  **generalized** from jet's 2D-only formulas (an explicit x/y outer
  product) to the outer product of the same per-axis 1D filter across
  however many axes `shape.length` has (matching jet exactly in the 2D
  case, extending the same idea to 1D and 3D) -- a generalization, not a
  literal port, since jet itself has no dimension-generic version of
  these to derive from directly.
- Unlike jet's own MGPCG (variable-coefficient, collider-aware, backed by
  an explicit per-cell matrix), this file targets the standard
  constant-coefficient Poisson/Laplacian on a plain rectangular domain
  with a fixed zero-flux boundary treatment -- a deliberate scope cut, not
  an oversight; see `multigrid.js`'s own header comment for the reasoning.
- The optional `dirichletMask` parameter (`laplacianAt`/
  `laplacianDiagonalAt`/`createLaplacianOperator`/
  `createMultigridPreconditioner`, added for `grid_pressure_solver2.js`
  below) generalizes jet's own hardcoded-zero "air" cell to an arbitrary
  per-cell fixed value, via an identity-row substitution applied at the
  finest level only -- this mechanism (the mask parameter itself, the
  identity-row substitution, and the single-level-only design, justified
  by a relax-formula argument in `multigrid.js`'s own header comment) is
  this port's own design; jet's own MGPCG takes a different approach
  entirely (an explicit, variable-coefficient per-cell matrix rebuilt at
  every level), out of scope here per the point above.

### jet/fluid-engine-dev (MIT) — `src/grid/grid_pressure_solver2.js`, direct, no Python intermediary

`src/grid/grid_pressure_solver2.js` (pressure projection) has no Python
source to port from at all -- the Python `fluxflow` project's own
`grid_solver2.py` never got past an abstract `computePressure` hook (same
situation as `advection_solver2.js` above). Read directly from jet's
`grid_single_phase_pressure_solver2.h`/`.cpp`. Same direct chain as the
other jet-sourced `src/grid/`/`src/linalg/` entries (no Python
intermediary):

```
fluid-engine-dev (C++, MIT, Doyub Kim)
  -> fluxflow (this package, JS/TSL, Apache-2.0, bert wang)
```

- **Source:** https://github.com/doyubkim/fluid-engine-dev/blob/master/include/jet/grid_single_phase_pressure_solver2.h, .../grid_single_phase_pressure_solver2.cpp
- **License:** MIT — full text already reproduced above, under "fluid-engine-dev (MIT) — via fluxflow (Python)"; not repeated a second time.

What carries over from jet: the fluid/pinned cell classification and the
overall per-cell stencil shape (an identity row for a pinned cell, a
divergence-based row for a fluid cell, a pinned neighbor contributing to a
fluid cell's row exactly like a normal fluid neighbor would). What's this
port's own generalization, not jet's: jet's own "air" cells are hardcoded
to exactly 0 (an open-boundary/atmosphere assumption); this generalizes
that to an arbitrary caller-supplied target value per cell (true
Dirichlet, not a single constant), and jet's own "boundary" (interior
solid/collider) category is not ported at all, matching `multigrid.js`'s
existing no-collider scope cut -- this file only knows about "fluid" and
"Dirichlet-pinned" cells, not a third "solid" one. Also this port's own:
the velocity-correction formula's overall *sign* is flipped relative to
jet's own literal code, a deliberate adjustment to stay consistent with
this port's pre-existing `createLaplacianOperator` sign convention rather
than jet's own (jet's own internal Laplacian is negated relative to this
port's) -- see this file's own header comment for the full derivation.
The matrix-free GPU realization (reusing `multigrid.js`'s existing
machinery with a mask, instead of jet's own `FdmMatrix2`/sparse-row code)
is original.

`src/grid/grid_solver2.js`'s own per-frame orchestration is also modeled
directly on jet's C++, beyond the hook names/shape already attributed
above (through `grid_solver2.py`): the exact stage order (external forces,
viscosity, pressure, advection) and, more specifically, the pattern of
re-applying the domain-boundary/collider velocity constraint
(`createGridBlockedBoundaryConditionSolver2`'s own `constrainVelocity()`,
already ported and attributed above) after *every* stage that changes
velocity, not just once per frame, is read directly from
`GridFluidSolver2::onAdvanceTimeStep`/`applyBoundaryCondition`
(`grid_fluid_solver2.cpp`) -- found and fixed after this orchestrator's
first version omitted it entirely, which let boundary velocity drift
unconstrained over many frames (see the file's own header comment for
the user-reported symptom and root cause).

- **Source:** https://github.com/doyubkim/fluid-engine-dev/blob/master/include/jet/grid_fluid_solver2.h, .../grid_fluid_solver2.cpp
- **License:** MIT — full text already reproduced above, under "fluid-engine-dev (MIT) — via fluxflow (Python)"; not repeated a second time.

## Design inspiration (not a code dependency)

### Taichi Lang

The Python `fluxflow` project (and therefore this port's algorithms) is
written against [Taichi Lang](https://github.com/taichi-dev/taichi)'s
programming model. Beyond the direct `linalg/linalg.js` derivation above, no
further Taichi source code itself is used or copied here — this is a credit
for the language/tool, not a licensing dependency for Taichi itself (the
actual ported algorithms are attributed above, to fluid-engine-dev, to
WebGL-Noise, and to the Python `fluxflow` project). Taichi is distributed
under the Apache License 2.0
(https://github.com/taichi-dev/taichi/blob/master/LICENSE).
