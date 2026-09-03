# tslify

用 [Taichi Lang](https://www.taichi-lang.org/) 的心智模型封装 three.js 的 [TSL](https://github.com/mrdoob/three.js/wiki/Three.js-Shading-Language) 计算能力，让在浏览器里写 GPU 并行计算（WebGPU compute）更简单。**通用并行计算工具**，不针对任何特定领域——流体解算之类的专用库会建立在这个库之上，单独发布。

> **状态**：早期搭建阶段。目前 `init()` / `arrayN()` / `array2()` / `array3()` 已落地，`kernel()` / `func()` 还在开发中。

## 为什么

TSL 已经能写 GPU 计算（`Fn` + `.compute()` + `instancedArray`），但原生写法需要不少样板代码：手动建 renderer、手动管理 storage buffer 下标、手动做 CPU↔GPU 数据搬运。Taichi Lang 在 Python 里把这些都藏了起来——`ti.field` 声明数据，`@ti.kernel` 写并行逻辑，最外层 `for` 循环自动并行。tslify 想把这套体验搬到 three.js/TSL 上，作为一个**通用**的并行计算基础库。

## 概念对照

| Taichi | 原生 TSL | tslify |
|---|---|---|
| `ti.init(arch=ti.gpu)` | 手动 `new WebGPURenderer()` + 样板 | `init()` ✅ 已实现 |
| `ti.field(dtype, shape)` | `instancedArray(count, type)` | `arrayN(type, shape)` ✅ 已实现 |
| （无对应） | — | `array2(type, w, h)` / `array3(type, w, h, d)` ✅ 已实现，tslify 独有的糖，等价于对应形状的 `arrayN` |
| `@ti.kernel` + 自动并行 for | `Fn(()=>{...})().compute(count)` | `kernel(shape, fn)` 🚧 开发中 |
| `@ti.func` | `Fn(fn)` | `func(fn)` 🚧 开发中 |
| `field.to_numpy()`/`from_numpy()` | 手动 readback | `.toArray()`/`.fromArray()` ✅ 已实现（`toArray()` 要配合 `kernel()` 才能读到有意义的数据，见下文） |

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

`arrayN(type, shape)` 声明一段 GPU 存储数据，`shape` 可以是数字（1D）或任意长度的数组（2D/3D/更高维，内部用统一的 strides 算法拍平，不为每个维度写特判）。`array2`/`array3` 是 2D/3D 这两种常见、且是 WebGPU dispatch 本身能对应到的形状的糖：

```js
const grid   = tslify.array2( 'float', 128, 128 );   // 等价于 arrayN('float', [128, 128])
const cloud  = tslify.arrayN( 'vec3', 4096 );         // 1D，4096 个 vec3
const volume = tslify.array3( 'float', 32, 32, 32 );  // 32³ 体素
```

`grid.shape` / `grid.count` 是形状和总元素数；`grid.at(i, j, ...)` 把多维下标换算成扁平 index（第一维变化最快：`index = i + j*width + ...`），提供给之后的 `kernel()` 用来访问元素。

`toArray()` / `fromArray()` 做 CPU↔GPU 数据搬运：

```js
grid.fromArray( someArray );        // 写入 CPU 侧缓冲区
const data = await grid.toArray();  // 异步读回一个 typed array
```

> **已知边界**：GPU 侧的 storage buffer 是惰性创建的——只有 compute/render pass 真正用过它之后才会在 GPU 上分配。也就是说单独 `fromArray()` 之后立刻 `toArray()`，中间没有 kernel 写过数据的话是读不到东西的（WebGPU 上会抛错，当前验证到的 WebGL fallback 上是静默返回空数组）。完整闭环要等 `kernel()` 接上后才能验证。

## 开发

```bash
npm install
npm test        # vitest，测 arrayN/array2/array3 的下标数学，不需要真 GPU
npm run dev
```

打开 `examples/00-context/` 确认 "WebGPU ready"，打开 `examples/01-array/` 确认 arrayN/array2/array3 的构造、`.at()`、`fromArray()` 都通过。

## 依赖

- [three.js](https://threejs.org/) `>=0.180.0`（peerDependency，需要包含 TSL / WebGPURenderer 的版本）
- 浏览器需要支持 WebGPU（计算部分是 WebGPU-only，WebGL2 fallback 不支持 compute shader）

## 关于 fluxflow

这个库只负责把 TSL 的并行计算能力封装得像 Taichi 一样好用，不绑定任何应用领域。基于它做流体解算的库会单独立项，叫 **fluxflow**。

## License

[MIT](LICENSE) © 2026 bert wang

第三方依赖的许可证信息见 [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)。
