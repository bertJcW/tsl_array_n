# tsl_array_n

Wraps three.js's [TSL](https://github.com/mrdoob/three.js/wiki/Three.js-Shading-Language) compute capabilities with [Taichi Lang](https://www.taichi-lang.org/)'s mental model, making GPU parallel compute (WebGPU compute) in the browser easier to write. A **general-purpose parallel-compute tool**, not tied to any specific domain -- domain-specific libraries such as fluid simulation are built on top of this one and published separately.

> **Status**: early scaffolding stage. `init()` / `arrayN()` / `array2()` / `array3()` / `kernel()` / `func()` have all landed and been verified.

## Why

TSL can already write GPU compute (`Fn` + `.compute()` + `instancedArray`), but the native way of writing it needs a fair amount of boilerplate: manually setting up a renderer, manually managing storage-buffer indices, manually doing CPU<->GPU data transfer. Taichi Lang hides all of that away in Python -- `ti.field` declares data, `@ti.kernel` writes parallel logic, and the outermost `for` loop is automatically parallelized. tsl_array_n wants to bring that experience to three.js/TSL, as a **general-purpose** parallel-compute foundation library.

## Concept mapping

| Taichi | Native TSL | tsl_array_n |
|---|---|---|
| `ti.init(arch=ti.gpu)` | manual `new WebGPURenderer()` + boilerplate | `init()` ✅ implemented |
| `ti.field(dtype, shape)` | `instancedArray(count, type)` | `arrayN(type, shape)` ✅ implemented |
| (no equivalent) | -- | `array2(type, w, h)` / `array3(type, w, h, d)` ✅ implemented, tsl_array_n's own sugar, equivalent to `arrayN` with the corresponding shape |
| `ti.field(dtype, shape=())` | manual uniform | `array0(type)` ✅ implemented, equivalent to `arrayN(type, [])`; users with a GLSL background can also use `uniform(value)` directly (a re-export of TSL's native `uniform` -- either works) |
| `@ti.kernel` + auto-parallel for | `Fn(()=>{...})().compute(count)` | `kernel(shape, fn)` ✅ implemented |
| `@ti.func` | `Fn(fn)` | `func` ✅ implemented (a straight re-export of `Fn` itself; see "Known limitations" below for its calling convention) |
| `field.to_numpy()`/`from_numpy()` | manual readback | `.toArray()`/`.fromArray()` ✅ implemented |

## Current state: `init()`

The recommended way to call this is the namespace-import style, `import * as tsl_array_n` (matching `import * as THREE from 'three'`), to avoid collisions with same-named exports from other packages (generic names like `init`/`field` are collision-prone):

```js
import * as tsl_array_n from 'tsl_array_n';

const renderer = await tsl_array_n.init(); // detects navigator.gpu, creates a WebGPURenderer, attaches it to document.body
```

Default import and named import are also supported:

```js
import tsl_array_n from 'tsl_array_n';           // tsl_array_n.init()
import { init } from 'tsl_array_n';         // init() (more prone to colliding with other packages' exports, not recommended)
```

An existing canvas / container / renderer can also be passed in:

```js
await tsl_array_n.init( { container: document.querySelector( '#app' ) } );
await tsl_array_n.init( { canvas: myCanvas } );
await tsl_array_n.init( { renderer: myExistingWebGPURenderer } );
```

## Current state: data

`arrayN(type, shape)` declares a block of GPU-resident storage data. `shape` can be a number (1D), an array of any length (2D/3D/higher dimensions, internally flattened with a unified strides algorithm, no special-casing per dimension), or an empty array `[]` (0-D, see below). `array0`/`array2`/`array3` are sugar for a few common shapes:

```js
const scalar = tsl_array_n.array0( 'vec2' );               // equivalent to arrayN('vec2', []), 0-D, a single element
const grid   = tsl_array_n.array2( 'float', 128, 128 );     // equivalent to arrayN('float', [128, 128])
const cloud  = tsl_array_n.arrayN( 'vec3', 4096 );          // 1D, 4096 vec3s
const volume = tsl_array_n.array3( 'float', 32, 32, 32 );   // a 32³ voxel volume
```

`array0(type)` corresponds to Taichi's 0-D field (`ti.field(dtype, shape=())`) -- a single value, but living on the GPU: readable inside `kernel()`/`func()`, and cheaply updatable per frame from the CPU side (`fromArray()`), commonly used to hold a parameter that changes over time without wanting to recompile the kernel every frame. It isn't a real uniform buffer in the WebGPU sense (under the hood it's still a length-1 storage buffer), but it's conceptually fully unified with `arrayN`/`array2`/`array3` -- the same `.fromArray()`/`.toArray()`/call-it-directly-to-get-a-GPU-element API, nothing extra to learn. No index is passed when accessing it: `scalar()` gets the GPU element, `scalar.at()` is always `0`.

If you're more familiar with native GLSL/TSL style, `uniform(value)` is also available directly -- a straight re-export of TSL's own `uniform`, a genuine WebGPU uniform (not a storage buffer), updated via `.value = x` instead of `.fromArray()`:

```js
const c = tsl_array_n.uniform( new THREE.Vector2( 0.7885, 0 ) ); // or tsl_array_n.array0('vec2'), equivalent effect
c.value.set( newX, newY );                                   // update every frame
```

The two are conceptually equivalent (both are "a single value living on the GPU, cheaply updatable per frame from the CPU"); which one to pick is purely a matter of taste -- `array0` is consistent with the rest of the library's data API, `uniform` is closer to native TSL/GLSL style.

`grid.shape` / `grid.count` are the shape and total element count. The field itself can be called directly as a function to access a GPU element -- `grid(i, j, ...)` returns the TSL element at position `(i,j,...)` (readable and writable, used together with `kernel()`/`func()`, taking node-valued indices); `grid.at(i, j, ...)` is a separate thing, pure CPU-side index math that converts multi-dimensional indices into a flat index (a plain number, with the first dimension varying fastest: `index = i + j*width + ...`) -- the two serve different purposes and don't affect each other.

```js
const p = grid( 1, 2 );      // a GPU element, for reading/writing inside a kernel
grid.at( 1, 2 );              // === 1 + 2*width, a plain number, for CPU-side use
```

`toArray()` / `fromArray()` handle CPU<->GPU data transfer:

```js
grid.fromArray( someArray );        // writes into the CPU-side buffer
const data = await grid.toArray();  // asynchronously reads back a typed array
```

> **Known limitations**: the GPU-side storage buffer is lazily created -- it's only actually allocated on the GPU once a compute/render pass has genuinely used it. That means calling `fromArray()` and then immediately `toArray()`, with no kernel having written data in between, reads back nothing (throws on WebGPU, silently returns an empty array on the WebGL fallback). Closing that loop needs `kernel()` (see below), which has been verified to work.

## Current state: `kernel()` / `func()`

`kernel(shape, fn)` corresponds to the "automatically parallel" outermost for loop inside `@ti.kernel`. `shape` determines how many times it dispatches, and the indices `fn` receives are already unflattened nodes (not plain numbers) derived from the GPU-side `instanceIndex`, so the field itself can be called directly to read/write:

```js
const grid = tsl_array_n.array2( 'float', 4, 4 );

const fill = tsl_array_n.kernel( grid.shape, ( x, y ) => {

	grid( x, y ).assign( x.add( y.mul( 4 ) ).toFloat() );

} );

fill();  // each call dispatches one parallel pass; can be called repeatedly, e.g. inside a render loop
```

`shape` is a free parameter independent of any field, and doesn't need to match some field's full shape -- for example, to parallelize over just the first dimension and get a flat index:

```js
tsl_array_n.kernel( grid.shape[ 0 ], ( x ) => { /* x: 0..width-1 */ } );
```

This behaves the same as Taichi's `for i in range(width)` -- the dispatch count is automatically clamped to `[0, width)`, never going out of bounds or touching other rows.

`func` is just TSL's `Fn`, re-exported under a Taichi-style name (`tsl_array_n.func === Fn`, not even a wrapper function), used to write small device-side functions that can be called from inside a kernel:

```js
const flatten = tsl_array_n.func( ( [ x, y ] ) => x.add( y.mul( 4 ) ) );
```

> **Known limitations**: when a function wrapped by `Fn()` is called, its signature is fixed as "a single destructured array parameter + a builder", not a one-to-one mapping of call arguments to parameters -- so a multi-parameter `func()` must be written as `func( ( [ x, y ] ) => ... )`, not `func( ( x, y ) => ... )` (in the latter, `y` actually gets bound to the internal builder object, and calling it produces a confusing error like `y.mul is not a function`). This is TSL's own calling convention -- `func` is `Fn` itself, with no extra handling added. **`kernel(shape, fn)`'s own callback is not affected** -- that one is invoked via plain JS argument spreading, so writing `(x, y) => ...` there works normally.

## Current state: `Loop()` / `Break()` / `Continue()` / `If()`

Straight re-exports of TSL's control-flow primitives, used to write loops inside a kernel/func that genuinely execute sequentially within each thread -- many algorithms (N-body, and fluid-sim-style cases where each particle needs to scan a set of neighbors) need this, since `kernel(shape, fn)`'s own auto-parallelism only covers the outermost level:

```js
import { float } from 'three/tsl';

const prefixSum = tsl_array_n.kernel( n, ( idx ) => {

	const sum = float( 0 ).toVar();

	tsl_array_n.Loop( idx.add( 1 ), ( { i } ) => {   // the loop count can be a node that differs per thread (here, idx+1)

		sum.addAssign( values( i ) );

	} );

	sums( idx ).assign( sum );

} );
```

`Loop(count, ({i}) => {...})`'s callback also follows a "single destructured parameter" convention (this time an object `{i}`, not the array style used by `func`/`Fn`); nested loops are written as `Loop(n, m, ({i, j}) => {...})`. `If(condition, fn)` pairs with `Break()`/`Continue()` for conditional early-exit/skip, used exactly the same way as native TSL -- `tsl_array_n` doesn't add anything on top.

> **Confirmed on real WebGPU**: `examples/03-loop/` has been run on a real `WebGPUBackend` -- the prefix sum, `Break`/`Continue`/`If`, and a minimal repro are all correct (`[1,3,6,10,15]`, etc.). The sandbox environment this library is developed in (no real WebGPU adapter, `init()` falls back to `WebGLBackend`) once produced wrong results when actually tested on that fallback backend -- after investigation this was confirmed not to be a tsl_array_n problem (a minimal repro using raw native `three/tsl`, not going through tsl_array_n at all, fails the same way in that environment), but a limitation of that WebGL2 fallback backend's own interaction between `Loop()` and storage buffers. **Conclusion: `Loop()`/`Break()`/`Continue()`/`If()` need real WebGPU; the WebGL2 fallback under `allowFallback: true` cannot be trusted for them** (this is consistent with the existing known limitation that "the WebGL2 fallback doesn't support compute shaders" -- the earlier simple `kernel()` scenarios happening to work on the fallback was just a coincidence of not hitting this particular gap).

> **`array0` shared reads across multiple threads have also been confirmed on real WebGPU**: `examples/04-julia/` (the animated version, where `c` is a `array0('vec2')` updated every frame, and every thread in a 512x512 kernel reads that same `array0` element) morphs continuously and correctly on real hardware. On the dev sandbox's WebGL2 fallback, this combination -- multiple threads concurrently reading the same `array0` element, read from inside a `Loop()` -- once read back all zeros; same as the `Loop()` case above, confirmed to be a limitation of the fallback backend itself, not a tsl_array_n problem.

## Development

```bash
npm install
npm test        # vitest, tests arrayN/array2/array3's index math + callable fields, no real GPU needed
npm run dev
```

Open `examples/00-context/` to confirm "WebGPU ready"; `examples/01-array/` to confirm arrayN/array2/array3 construction, `.at()`, `fromArray()`; `examples/02-kernel/` to confirm `kernel()`/`func()` dispatch, single-axis dispatch, and the full fromArray->kernel->toArray loop; `examples/03-loop/` to confirm `Loop()`/`Break()`/`Continue()` (on real WebGPU; see the known limitations above). `examples/04-julia/`, `examples/05-nbody/`, `examples/06-julia-uniform/` are three visual demos that exercise the whole API against real algorithms (two Julia-set fractals -- demonstrating dynamic parameters via `array0` and `uniform` respectively -- plus an N-body gravity simulation).

## Dependencies

- [three.js](https://threejs.org/) `>=0.180.0` (peerDependency, needs a version that includes TSL / WebGPURenderer)
- The browser needs WebGPU support (the compute part is WebGPU-only; the WebGL2 fallback doesn't support compute shaders)

## About fluxflow

This library is only responsible for wrapping TSL's parallel-compute capabilities to be as pleasant to use as Taichi, and isn't tied to any particular application domain. A library built on top of it for fluid simulation is a separate project, called **fluxflow**.

## License

[MIT](LICENSE) © 2026 bert wang

See [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) for third-party dependency license information.
