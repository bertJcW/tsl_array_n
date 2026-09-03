# tslify

用 [Taichi Lang](https://www.taichi-lang.org/) 的心智模型封装 three.js 的 [TSL](https://github.com/mrdoob/three.js/wiki/Three.js-Shading-Language) 计算能力，让在浏览器里写 GPU 并行计算（WebGPU compute）更简单。**通用并行计算工具**，不针对任何特定领域——流体解算之类的专用库会建立在这个库之上，单独发布。

> **状态**：早期搭建阶段。`init()` / `arrayN()` / `array2()` / `array3()` / `kernel()` / `func()` 都已落地并验证过。

## 为什么

TSL 已经能写 GPU 计算（`Fn` + `.compute()` + `instancedArray`），但原生写法需要不少样板代码：手动建 renderer、手动管理 storage buffer 下标、手动做 CPU↔GPU 数据搬运。Taichi Lang 在 Python 里把这些都藏了起来——`ti.field` 声明数据，`@ti.kernel` 写并行逻辑，最外层 `for` 循环自动并行。tslify 想把这套体验搬到 three.js/TSL 上，作为一个**通用**的并行计算基础库。

## 概念对照

| Taichi | 原生 TSL | tslify |
|---|---|---|
| `ti.init(arch=ti.gpu)` | 手动 `new WebGPURenderer()` + 样板 | `init()` ✅ 已实现 |
| `ti.field(dtype, shape)` | `instancedArray(count, type)` | `arrayN(type, shape)` ✅ 已实现 |
| （无对应） | — | `array2(type, w, h)` / `array3(type, w, h, d)` ✅ 已实现，tslify 独有的糖，等价于对应形状的 `arrayN` |
| `ti.field(dtype, shape=())` | 手动 uniform | `array0(type)` ✅ 已实现，等价于 `arrayN(type, [])`；GLSL 背景的用户也可以直接用 `uniform(value)`（TSL 原生 `uniform` 的重导出，两者随便选） |
| `@ti.kernel` + 自动并行 for | `Fn(()=>{...})().compute(count)` | `kernel(shape, fn)` ✅ 已实现 |
| `@ti.func` | `Fn(fn)` | `func` ✅ 已实现（`Fn` 本身的重导出，调用约定见下方「已知边界」） |
| `field.to_numpy()`/`from_numpy()` | 手动 readback | `.toArray()`/`.fromArray()` ✅ 已实现 |

## 现状：`init()`

推荐用 `import * as tslify` 这种命名空间写法调用（跟 `import * as THREE from 'three'` 一致），避免和其它包的同名导出（`init`、`field` 之类很通用的名字）冲突：

```js
import * as tslify from 'tslify';

const renderer = await tslify.init(); // 检测 navigator.gpu、创建 WebGPURenderer、挂到 document.body
```

也支持 default import 和具名 import：

```js
import tslify from 'tslify';           // tslify.init()
import { init } from 'tslify';         // init()（更容易和别的包撞名，不太推荐）
```

也可以传入已有的 canvas / 容器 / renderer：

```js
await tslify.init( { container: document.querySelector( '#app' ) } );
await tslify.init( { canvas: myCanvas } );
await tslify.init( { renderer: myExistingWebGPURenderer } );
```

## 现状：数据

`arrayN(type, shape)` 声明一段 GPU 存储数据，`shape` 可以是数字（1D）、任意长度的数组（2D/3D/更高维，内部用统一的 strides 算法拍平，不为每个维度写特判），或者空数组 `[]`（0 维，见下）。`array0`/`array2`/`array3` 是几种常见形状的糖：

```js
const scalar = tslify.array0( 'vec2' );               // 等价于 arrayN('vec2', [])，0 维，单个元素
const grid   = tslify.array2( 'float', 128, 128 );     // 等价于 arrayN('float', [128, 128])
const cloud  = tslify.arrayN( 'vec3', 4096 );          // 1D，4096 个 vec3
const volume = tslify.array3( 'float', 32, 32, 32 );   // 32³ 体素
```

`array0(type)` 对标 Taichi 的 0 维 field（`ti.field(dtype, shape=())`）——一个单值，但活在 GPU 上，可以在 `kernel()`/`func()` 里读、也能从 CPU 侧每帧廉价更新（`fromArray()`），常用来装一个会随时间变化、又不想每帧重新编译 kernel 的参数。它不是 WebGPU 意义上真正的 uniform buffer（底层还是长度为 1 的 storage buffer），但概念上和 `arrayN`/`array2`/`array3` 完全统一——`.fromArray()`/`.toArray()`/直接调用取 GPU element 这套 API 不用另学一遍。访问时不传下标：`scalar()` 拿 GPU element，`scalar.at()` 恒等于 `0`。

如果更熟悉 GLSL/TSL 原生写法，`uniform(value)` 也直接可用——原样重导出 TSL 的 `uniform`，是真正的 WebGPU uniform（不是 storage buffer），更新方式是 `.value = x` 而不是 `.fromArray()`：

```js
const c = tslify.uniform( new THREE.Vector2( 0.7885, 0 ) ); // 或 tslify.array0('vec2')，效果等价
c.value.set( newX, newY );                                   // 每帧更新
```

两者概念等价（都是"活在 GPU 上、可从 CPU 每帧更新的单值"），选哪个纯粹看个人习惯——`array0` 跟库里其它数据 API 统一，`uniform` 更贴近原生 TSL/GLSL 的写法。

`grid.shape` / `grid.count` 是形状和总元素数。field 本身可以直接当函数调用来访问 GPU 元素——`grid(i, j, ...)` 返回 `(i,j,...)` 位置的 TSL element（可读可写，配合 `kernel()`/`func()` 用，吃的是 node 下标）；`grid.at(i, j, ...)` 是另一件事，纯 CPU 侧下标数学，把多维下标换算成扁平 index（一个 number，第一维变化最快：`index = i + j*width + ...`），两者分工不同、互不影响。

```js
const p = grid( 1, 2 );      // GPU element，kernel 里读写用
grid.at( 1, 2 );              // === 1 + 2*width，纯数字，CPU 侧用
```

`toArray()` / `fromArray()` 做 CPU↔GPU 数据搬运：

```js
grid.fromArray( someArray );        // 写入 CPU 侧缓冲区
const data = await grid.toArray();  // 异步读回一个 typed array
```

> **已知边界**：GPU 侧的 storage buffer 是惰性创建的——只有 compute/render pass 真正用过它之后才会在 GPU 上分配。也就是说单独 `fromArray()` 之后立刻 `toArray()`，中间没有 kernel 写过数据的话是读不到东西的（WebGPU 上会抛错，WebGL fallback 上是静默返回空数组）。这个闭环需要 `kernel()`（见下文），已经验证通过。

## 现状：`kernel()` / `func()`

`kernel(shape, fn)` 对标 `@ti.kernel` 里那个"自动并行"的最外层 for 循环。`shape` 决定 dispatch 多少次，`fn` 拿到的下标是已经从 GPU 侧 `instanceIndex` 反解好的 node（不是 plain number），可以直接调用 field 本身来读写：

```js
const grid = tslify.array2( 'float', 4, 4 );

const fill = tslify.kernel( grid.shape, ( x, y ) => {

	grid( x, y ).assign( x.add( y.mul( 4 ) ).toFloat() );

} );

fill();  // 每次调用派发一次并行 pass，可以放进 render loop 里反复调
```

`shape` 是独立于任何 field 的自由参数，不需要跟某个 field 的完整形状对齐——比如只想按第一维并行、拿到扁平下标：

```js
tslify.kernel( grid.shape[ 0 ], ( x ) => { /* x: 0..width-1 */ } );
```

效果跟 Taichi 的 `for i in range(width)` 一样，dispatch 次数自动卡在 `[0, width)`，不会越界，也不会碰到其它行。

`func` 就是 TSL 的 `Fn`，重导出成 Taichi 风格的名字而已（`tslify.func === Fn`，连包装函数都没有），用来写可以被 kernel 内部调用的设备侧小函数：

```js
const flatten = tslify.func( ( [ x, y ] ) => x.add( y.mul( 4 ) ) );
```

> **已知边界**：`Fn()` 包出来的函数被调用时，签名固定是"一个 destructure 数组参数 + builder"，不是把调用参数一个个对应过去——所以多参数的 `func()` 必须写成 `func( ( [ x, y ] ) => ... )`，不能写 `func( ( x, y ) => ... )`（后者 `y` 实际会绑定到内部 builder 对象，调用会报出 `y.mul is not a function` 这类奇怪错误）。这是 TSL 自己的调用约定，`func` 就是 `Fn` 本身，没有任何额外处理。**`kernel(shape, fn)` 的回调不受影响**——那是用普通 JS 展开调用的，正常写 `(x, y) => ...` 就行。

## 现状：`Loop()` / `Break()` / `Continue()` / `If()`

原样重导出 TSL 的控制流原语，用来在 kernel/func 内部写"每个线程内部真的顺序执行"的循环——很多算法（N-body、流体这类每个粒子要扫描一组邻居的场景）都需要这个，`kernel(shape, fn)` 本身的自动并行只覆盖最外层：

```js
import { float } from 'three/tsl';

const prefixSum = tslify.kernel( n, ( idx ) => {

	const sum = float( 0 ).toVar();

	tslify.Loop( idx.add( 1 ), ( { i } ) => {   // 循环次数可以是逐线程不同的 node（这里是 idx+1）

		sum.addAssign( values( i ) );

	} );

	sums( idx ).assign( sum );

} );
```

`Loop(count, ({i}) => {...})` 的回调也是"单个 destructure 参数"的约定（这次是对象 `{i}`，不是 `func`/`Fn` 那种数组），嵌套循环写 `Loop(n, m, ({i, j}) => {...})`。`If(condition, fn)` 配合 `Break()`/`Continue()` 做条件跳出/跳过，用法和原生 TSL 一样，`tslify` 没有加任何东西。

> **已在真实 WebGPU 验证**：`examples/03-loop/` 在真实 `WebGPUBackend` 上跑，前缀和、`Break`/`Continue`/`If`、最小复现全部正确（`[1,3,6,10,15]` 等）。开发这个库用的 sandbox 环境（没有真实 WebGPU 适配器，`init()` 会 fallback 到 `WebGLBackend`）曾经在这个 fallback 后端上实测出错误结果——排查后确认不是 tslify 的问题（不经过 tslify、纯原生 `three/tsl` 的最小复现在那个环境下同样出错），而是那个 WebGL2 fallback 后端自己在 `Loop()` + storage buffer 交互上的限制。**结论：`Loop()`/`Break()`/`Continue()`/`If()` 需要真实 WebGPU，`allowFallback: true` 下的 WebGL2 fallback 不可信**（这跟"WebGL2 fallback 不支持 compute shader"这条已知限制是一致的，之前 `kernel()` 的简单场景之所以能在 fallback 上跑通，只是巧合地没踩到这个坑）。

> **`array0` 多线程共读也已在真实 WebGPU 验证**：`examples/04-julia/`（动画版，`c` 用 `array0('vec2')` 每帧更新，512×512 kernel 里每个线程都读同一个 `array0` 元素）在真机上动画连续正确变形。开发 sandbox 的 WebGL2 fallback 上曾经在这个"多线程并发读同一个 array0 元素 + 在 Loop() 内部读"的组合下全部读到 0——和上面 `Loop()` 那条一样，确认是 fallback 后端自己的限制，不是 tslify 的问题。

## 开发

```bash
npm install
npm test        # vitest，测 arrayN/array2/array3 的下标数学 + 可调用 field，不需要真 GPU
npm run dev
```

打开 `examples/00-context/` 确认 "WebGPU ready"，`examples/01-array/` 确认 arrayN/array2/array3 的构造、`.at()`、`fromArray()`，`examples/02-kernel/` 确认 `kernel()`/`func()` 派发、单轴 dispatch、完整的 fromArray→kernel→toArray 闭环，`examples/03-loop/` 确认 `Loop()`/`Break()`/`Continue()`（真实 WebGPU 下；见上面的已知边界）。`examples/04-julia/`、`examples/05-nbody/`、`examples/06-julia-uniform/` 是三个用真实算法把整套 API 过一遍的可视化 demo（Julia 分形×2——分别用 `array0` 和 `uniform` 演示动态参数——外加 N-body 引力模拟）。

## 依赖

- [three.js](https://threejs.org/) `>=0.180.0`（peerDependency，需要包含 TSL / WebGPURenderer 的版本）
- 浏览器需要支持 WebGPU（计算部分是 WebGPU-only，WebGL2 fallback 不支持 compute shader）

## 关于 fluxflow

这个库只负责把 TSL 的并行计算能力封装得像 Taichi 一样好用，不绑定任何应用领域。基于它做流体解算的库会单独立项，叫 **fluxflow**。

## License

[MIT](LICENSE) © 2026 bert wang

第三方依赖的许可证信息见 [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)。
