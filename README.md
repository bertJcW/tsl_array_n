# tslify

用 [Taichi Lang](https://www.taichi-lang.org/) 的心智模型封装 three.js 的 [TSL](https://github.com/mrdoob/three.js/wiki/Three.js-Shading-Language) 计算能力，让在浏览器里写 GPU 并行计算（WebGPU compute）更简单。**通用并行计算工具**，不针对任何特定领域——流体解算之类的专用库会建立在这个库之上，单独发布。

> **状态**：早期搭建阶段。目前只有 `init()` 落地，`field()` / `kernel()` / `func()` 还在开发中。

## 为什么

TSL 已经能写 GPU 计算（`Fn` + `.compute()` + `instancedArray`），但原生写法需要不少样板代码：手动建 renderer、手动管理 storage buffer 下标、手动做 CPU↔GPU 数据搬运。Taichi Lang 在 Python 里把这些都藏了起来——`ti.field` 声明数据，`@ti.kernel` 写并行逻辑，最外层 `for` 循环自动并行。tslify 想把这套体验搬到 three.js/TSL 上，作为一个**通用**的并行计算基础库。

## 概念对照

| Taichi | 原生 TSL | tslify |
|---|---|---|
| `ti.init(arch=ti.gpu)` | 手动 `new WebGPURenderer()` + 样板 | `init()` ✅ 已实现 |
| `ti.field(dtype, shape)` | `instancedArray(count, type)` | `field(type, shape)` 🚧 开发中 |
| `@ti.kernel` + 自动并行 for | `Fn(()=>{...})().compute(count)` | `kernel(shape, fn)` 🚧 开发中 |
| `@ti.func` | `Fn(fn)` | `func(fn)` 🚧 开发中 |
| `field.to_numpy()`/`from_numpy()` | 手动 readback | `field.toArray()`/`fromArray()` 🚧 开发中 |

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

## 开发

```bash
npm install
npm run dev
```

然后打开 `examples/00-context/`，确认页面显示 "WebGPU ready"。

## 依赖

- [three.js](https://threejs.org/) `>=0.180.0`（peerDependency，需要包含 TSL / WebGPURenderer 的版本）
- 浏览器需要支持 WebGPU（计算部分是 WebGPU-only，WebGL2 fallback 不支持 compute shader）

## 关于 fluxflow

这个库只负责把 TSL 的并行计算能力封装得像 Taichi 一样好用，不绑定任何应用领域。基于它做流体解算的库会单独立项，叫 **fluxflow**。
