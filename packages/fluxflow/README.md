# fluxflow

Browser-side GPU fluid simulation, built on [tsl_array_n](../tsl_array_n), ported from a [Taichi Lang](https://www.taichi-lang.org/) fluid-simulation library (`D:\OneDrive\04_lib_fluxflow`). The end goal is real-time browser fluid visualization and interaction paired with three.js.

> **Status**: the Python source's `grid/` folder (MAC-grid data structures + numeric helpers + SDF colliders + boundary-condition solver), `noise/` folder (Perlin/Simplex/cellular noise), and `linalg/` folder (matrix-free conjugate gradient) have all been ported. There's no pressure-projection/advection/viscosity solver itself yet (`grid_solver2.js` is just an empty, hook-pluggable skeleton) -- `linalg`'s CG solver is meant to become that solver's pressure step later, but isn't wired into anything yet (its own numerics are confirmed correct on real WebGPU, see below). This port's goal is the "foundation layer", not an end-to-end solver.

## Current state: `grid`

```js
import { grid } from 'fluxflow';
```

| File | Corresponding Python source | Contents |
|---|---|---|
| `constant.js` | `constant.py` | Direction flags, `FLOAT_TYPE`, `createGravity()`/`setGravity()` (`array0('float')`, each simulation builds its own, not a module-level singleton) |
| `grid_math.js` | `grid_math.py` | Bilinear sampling / gradient / divergence / curl / laplacian, plain JS functions (building TSL node graphs), not `tsl_array_n.func()` |
| `level_set_utils.js` | `level_set_utils.py` | SDF's `isInsideSdf`/`fractionInsideSdf` |
| `array_utils.js` | `array_utils.py` | `createCopyKernel2`/`createExtrapolateToRegion2`, factory functions: build a field-bound kernel once, the returned function is the repeatedly-callable dispatcher |
| `grid_data2.js` | `grid_data2.py` | `ScalarGrid2`/`CellCenteredScalarGrid2`/`VertexCenteredScalarGrid2`/`CollocatedVectorGrid2`/`CellCenteredVectorGrid2`/`VertexCenteredVectorGrid`/`FaceCenteredGrid2`, factory functions thinly wrapping `tsl_array_n.array2()` |
| `polygon_sdf.js` | (no counterpart, replaces shapely) | Pure CPU geometry: point-in-polygon, point-to-polygon-boundary distance, polygon-union SDF (pointwise min), polygon centroid/translate/rotate |
| `svg_utils.js` | (no counterpart, replaces svg.path) | Samples vertices along a path using the browser's native `SVGPathElement` API, browser-only |
| `sdf_collider2.js` | `sdf_collider2.py` | `createSDFStaticCollider2`/`createSDFRigidBodyCollider2`; `addPolygon`/`addSvg` replace the source's `addShapelyGeometry`/`addSvg` (zero new dependencies, see below) |
| `grid_blocked_boundary_condition_solver2.js` | `grid_blocked_boundary_condition_solver2.py` | `createGridBlockedBoundaryConditionSolver2` -- velocity-field constraints from colliders + closed domain boundaries, the largest single file in this port |
| `grid_solver2.js` | `grid_solver2.py` | `createGridSolver2`, a pure hooks skeleton; a concrete solver is out of scope for this pass |

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
| `linalg.js` | `linalg.py` | `createConjugateGradientSolver(applyOperator, b, x)` -- matrix-free conjugate gradient, ported from Taichi Lang's own `matrixfree_cg.py` via the Python `fluxflow` project; see [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) |

Meant for the pressure-projection step of a future solver (`Ax=b` where `A` is a matrix-free Laplacian-like stencil operator, `b` the divergence, `x` the pressure). Real structural differences from the source, driven by platform gaps, not style -- see the file's own header comment for the full reasoning:
- **GPU-side reduction via WebGPU atomics, not a CPU sum**: Taichi's `result += p[I]*q[I]` inside a kernel is a genuine parallel reduction the compiler handles on-device. tsl_array_n has no reduction primitive of its own, but three.js TSL exposes real `atomicAdd` on a storage buffer marked `.toAtomic()`, so both dot products this solver needs (`r.r`, `p.Ap`) run as every thread atomically adding its own per-cell product into one shared accumulator -- a genuine GPU-side reduction. Since WGSL atomics only exist for `atomic<i32>`/`atomic<u32>` (never float), each product is scaled by a configurable `atomicScale` (default 65536) and rounded to a fixed-point int before the add, then divided back after reading the single accumulated int back to the CPU -- trading exact float precision and some int32 headroom for turning an O(N) CPU-bound reduction + O(N)-element transfer into an O(N) GPU-bound reduction + a single-int transfer. This also means the solver currently only supports scalar `'float'` fields (a per-component accumulator would be needed for vector types, not attempted since nothing here needs it). `solve()` is still `async`, since the convergence check is inherently CPU-side and still needs one small readback per iteration either way.
- **"Create once, solve many times"**: the source dynamically allocates and destroys its scratch fields (`p`/`r`/`Ap`/`Ax`/`alpha`/`beta`) on every `mfcg(...)` call; tsl_array_n has no field-disposal mechanism at all yet, so a solver meant to run every frame (as a pressure solve would) needs its scratch allocated once, not per-call. `createConjugateGradientSolver(applyOperator, b, x, options?)` builds everything once and returns `{ solve(tol, maxiter) }` -- callers should create one solver per `(b, x)` pair they intend to reuse across frames, matching this port's established `array_utils.js`/boundary-condition-solver convention, not the source's single-call shape.
- **`applyOperator(input, output) => dispatcher`**: a factory, called by this module exactly twice (bound to `(x, Ax)` and to this solver's own `(p, Ap)` scratch) -- replaces the source's `LinearOperator`/single flexible `matvec_kernel`, since tsl_array_n kernels bind to concrete fields at construction time. Same shape as `createCopyKernel2` in `array_utils.js`.
- Generic over 1D/2D/3D shapes (matching the source's own `ti.i`/`ti.ij`/`ti.ijk` branching), capped at 3D for the usual reason (a WebGPU dispatch is inherently <=3D).
- The source's final success check compares the raw (squared) residual against `tol` directly, while every check inside the loop compares `sqrt(residual)` against `tol` -- preserved as-is (see the comment in `linalg.js` for why this is harmless for realistic tolerances, not "fixed").

Verified with 8 vitest structural/validation tests, plus a live check. This project's dev sandbox has no real WebGPU adapter (falls back to WebGLBackend) and WebGPU atomics have no GLSL equivalent, so `examples/04-conjugate-gradient/` cannot even compile its shader there -- the browser console shows a vertex shader compile error (`ERROR: 0:68: '&' : syntax error`, on the WGSL pointer syntax `atomicAdd(&x, ...)` that the WebGL2 fallback's node builder never learned a GLSL translation for), a *confirmed* fallback-only failure with no ambiguity left to isolate, unlike earlier sandbox mysteries in this project. **Run by the user on real WebGPU hardware, the example converges to the expected exact answer**: solving `A=diag(1..8)`, `b=[1,...,1]`, it reports `x = [1.0000, 0.4999, 0.3335, 0.2498, 0.2001, 0.1666, 0.1429, 0.1250]` against an expected `[1.0000, 0.5000, 0.3333, 0.2500, 0.2000, 0.1667, 0.1429, 0.1250]` -- the ~1e-4 per-element deviation is consistent with the atomic dot product's fixed-point quantization noise (see `linalg.js`'s header comment), well within this example's 1e-3 comparison tolerance. Both the GPU atomic reduction and the CG iteration built on top of it are confirmed correct on real WebGPU.

## Key tradeoffs made during this port

- **No double precision**: WGSL/WebGPU compute has no native f64, so this is fixed to float(f32) only; the source's `initConstant()` orchestration (needed only to support switching precision) is therefore unnecessary too.
- **`grid_math.py`'s `@ti.func`s all became plain JS functions** (building/composing TSL node graphs) instead of `tsl_array_n.func()` -- these functions often need to return several named values (e.g. `bilinearCoordsAndWeights2` returns 8), which doesn't fit `func()`/`Fn`'s single-destructured-array calling convention.
- **`vectorGradient2`/`vectorGradientAtPosition2`'s mat2 element order was confirmed wrong on real WebGPU, and fixed.** Taichi's `tm.mat2(a,b,c,d)` is row-major (row0=(a,b), row1=(c,d)); TSL's `mat2(a,b,c,d)` turned out to be column-major (column0=(a,b), column1=(c,d)) -- a direct argument-order translation of the source produced a transposed Jacobian. Confirmed live via `examples/00-grid-math/` on real WebGPU hardware and fixed by swapping the middle two constructor arguments; see the comment in `grid_math.js`.
- **Kernels bind to a concrete field at construction time**, with no support for "call the same kernel rebound to a different field" -- as a result, `grid_blocked_boundary_condition_solver2.js`'s API shape deliberately diverges from the source: the constructor takes a fixed `velocity` (FaceCenteredGrid2) directly, and `constrainVelocity()` no longer takes velocity as a per-call argument the way the source does. The collider, however, can still be swapped mid-lifetime (`setCollider()`), which rebuilds every collider-dependent kernel when it's called.
- **Zero new dependencies for the SDF collider's polygon/SVG rasterization**: `addPolygon`/`addPolygons` use hand-rolled point-in-polygon + point-to-boundary distance (matching the semantics of shapely's `boundary.distance`+`contains`/`touches`); multiple shapes are combined via SDF pointwise min (no real polygon boolean union needed); `addSvg` samples along the path using the browser's native `SVGPathElement.getPointAtLength()`, instead of the Python-only `svg.path`.
- **The `VertexCentered*` grids' dataSize doesn't carry over the source's "keep (0,0) when resolution=(0,0)" defensive branch** -- `tsl_array_n.array2()` itself rejects zero-length dimensions, and nothing in `grid/` actually exercises that branch (confirmed via grep).
- **`frictionCoefficient`/`closedDomainBoundaryFlag` are plain mutable properties** (`collider.frictionCoefficient = x`); the corresponding setter methods in the source (`setFriectionCoefficient`/`setClosedDomainBoundaryFlag`) weren't carried over, which is more natural plain-JS idiom. **One thing to watch for**: everywhere `frictionCoefficient` gets read inside a kernel, its value is baked into the node graph as a constant at kernel **build** time, not re-read on every dispatch -- this matches the source's own Taichi-side behavior (reading a plain Python attribute inside a Taichi `@ti.kernel` is also compile-time-constant-folded, not a new limitation introduced by this port), but if "change the friction coefficient at runtime and have an already-built kernel pick it up immediately" is ever needed, it has to become an `array0`/`uniform` instead. `SDFRigidBodyCollider2.velocityAt()` (which reads `currentPosition`/`linearVelocity`) has the same architectural limitation -- see the detailed comment in `sdf_collider2.js`.

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
