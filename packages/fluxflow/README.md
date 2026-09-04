# fluxflow

Browser-side GPU fluid simulation, built on [tsl_array_n](../tsl_array_n), ported from a [Taichi Lang](https://www.taichi-lang.org/) fluid-simulation library (`D:\OneDrive\04_lib_fluxflow`). The end goal is real-time browser fluid visualization and interaction paired with three.js.

> **Status**: the Python source's `grid/` folder (MAC-grid data structures + numeric helpers + SDF colliders + boundary-condition solver) has been fully ported. There's no pressure-projection/advection/viscosity solver itself yet (`grid_solver2.js` is just an empty, hook-pluggable skeleton), so this can't run a complete fluid simulation yet -- this port's goal is the "foundation layer", not an end-to-end solver.

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

`src/grid/` is ported from a separate project (`D:\OneDrive\04_lib_fluxflow`, also Apache License 2.0), part of which traces further back to [fluid-engine-dev](https://github.com/doyubkim/fluid-engine-dev) (MIT) -- see [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) for the full provenance chain and per-file mapping.
