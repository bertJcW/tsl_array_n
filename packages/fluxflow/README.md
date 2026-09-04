# fluxflow

浏览器端 GPU 流体解算，基于 [tsl_array_n](../tsl_array_n) 构建，从一个 [Taichi Lang](https://www.taichi-lang.org/) 写的流体解算库（`D:\OneDrive\04_lib_fluxflow`）移植而来。配合 three.js 做网页端实时流体可视化和交互是最终目标。

> **状态**：Python 源码 `grid/` 文件夹（MAC 网格数据结构 + 数值 helper + SDF collider + 边界条件求解器）已经全部搬完。还没有压力投影/对流/粘性求解器本身（`grid_solver2.js` 只是一个可插 hook 的空骨架），所以还不能跑一个完整的流体模拟——这次移植的目标是"地基层"，不是端到端的求解器。

## 现状：`grid`

```js
import { grid } from 'fluxflow';
```

| 文件 | 对应 Python 源码 | 内容 |
|---|---|---|
| `constant.js` | `constant.py` | 方向 flag、`FLOAT_TYPE`、`createGravity()`/`setGravity()`（`array0('float')`，每个模拟各建各的，不是模块级单例） |
| `grid_math.js` | `grid_math.py` | 双线性采样/梯度/散度/旋度/拉普拉斯，普通 JS 函数（构建 TSL 节点图），不是 `tsl_array_n.func()` |
| `level_set_utils.js` | `level_set_utils.py` | SDF 的 `isInsideSdf`/`fractionInsideSdf` |
| `array_utils.js` | `array_utils.py` | `createCopyKernel2`/`createExtrapolateToRegion2`，工厂函数：构建一次绑定好 field 的 kernel，返回的函数才是可反复调用的 dispatcher |
| `grid_data2.js` | `grid_data2.py` | `ScalarGrid2`/`CellCenteredScalarGrid2`/`VertexCenteredScalarGrid2`/`CollocatedVectorGrid2`/`CellCenteredVectorGrid2`/`VertexCenteredVectorGrid`/`FaceCenteredGrid2`，工厂函数薄封装 `tsl_array_n.array2()` |
| `polygon_sdf.js` | （无对应，替代 shapely） | 纯 CPU 几何：point-in-polygon、点到多边形边界距离、多边形并集 SDF（pointwise min）、多边形质心/平移/旋转 |
| `svg_utils.js` | （无对应，替代 svg.path） | 浏览器原生 `SVGPathElement` API 沿路径采样顶点，只能在浏览器跑 |
| `sdf_collider2.js` | `sdf_collider2.py` | `createSDFStaticCollider2`/`createSDFRigidBodyCollider2`，`addPolygon`/`addSvg` 替代源码的 `addShapelyGeometry`/`addSvg`（零新依赖，见下） |
| `grid_blocked_boundary_condition_solver2.js` | `grid_blocked_boundary_condition_solver2.py` | `createGridBlockedBoundaryConditionSolver2`——碰撞体+封闭边界的速度场约束，这次移植里最大的一个文件 |
| `grid_solver2.js` | `grid_solver2.py` | `createGridSolver2`，纯 hooks 骨架，具体求解器不在这次范围内 |

## 移植时的关键取舍

- **不支持双精度**：WGSL/WebGPU compute 没有原生 f64，固定只用 float(f32)；源码里为了配合精度切换才需要的 `initConstant()` 编排也就不需要了。
- **`grid_math.py` 的 `@ti.func` 全部改成普通 JS 函数**（构建/组合 TSL 节点图），不用 `tsl_array_n.func()`——这些函数经常要返回好几个具名的值（比如 `bilinearCoordsAndWeights2` 返回8个量），`func()`/`Fn` 的单参数数组解构调用约定不适合。
- **`vectorGradient2`/`vectorGradientAtPosition2` 的 mat2 元素顺序还未经实机数值验证**（Taichi `tm.mat2` 和 TSL `mat2` 的填充顺序 row-major/column-major 是否一致，直译过去可能是转置的）。
- **kernel 在构造时绑定具体 field**，不支持"同一个 kernel 换绑不同 field 调用"——`grid_blocked_boundary_condition_solver2.js` 因此在 API 形状上跟源码有意偏离：构造函数直接吃一个固定的 `velocity`（FaceCenteredGrid2），`constrainVelocity()` 不再像源码那样每次调用传入 velocity。collider 则支持中途换（`setCollider()`），换的时候会重新构建所有依赖 collider 的 kernel。
- **SDF collider 的多边形/SVG 光栅化，零新依赖**：`addPolygon`/`addPolygons` 用手写 point-in-polygon + 点到边界距离（跟 shapely 的 `boundary.distance`+`contains`/`touches` 语义对应）；多个形状合并用 SDF pointwise min（不需要真正的多边形布尔并集）；`addSvg` 用浏览器原生 `SVGPathElement.getPointAtLength()` 沿路径采样，不用 Python 专用的 `svg.path`。
- **`VertexCentered*` 网格的 dataSize 没有照搬源码"resolution=(0,0) 时保持(0,0)"这个防御分支**——`tsl_array_n.array2()` 本身不接受 0 长度维度，且 `grid/` 里从没实际用到这个分支（grep 确认过）。
- **`frictionCoefficient`/`closedDomainBoundaryFlag` 是普通可变属性**（`collider.frictionCoefficient = x`），源码里对应的 setter 方法（`setFriectionCoefficient`/`setClosedDomainBoundaryFlag`）没有照搬，纯 JS 惯用法上更自然。**但要注意**：`frictionCoefficient` 在 kernel 内部被读取的地方，值是在 kernel **构建**时烤进节点图的常量，不是每次 dispatch 都重新读——这跟源码 Taichi 端的行为一致（Taichi 的 `@ti.kernel` 里读取普通 Python 属性也是编译期常量化，不是这次移植引入的新限制），但如果以后要支持"运行时改摩擦系数、且已构建的 kernel 立刻生效"，需要把它换成 `array0`/`uniform`。`SDFRigidBodyCollider2.velocityAt()`（读 `currentPosition`/`linearVelocity`）也有同样的架构性限制，见 `sdf_collider2.js` 里的详细注释。

## 已知边界（sandbox 环境限制，不是这次移植的代码问题）

`examples/00-grid-math/`（`grid_math.js` 数值验证）和 `examples/01-boundary-condition/`（边界条件求解器冒烟测试）都在这个开发 sandbox（没有真实 WebGPU 适配器，`init()` fallback 到 `WebGLBackend`）里跑过：

- `01-boundary-condition/` 全绿——完整链路（collider 栅格化→构造边界求解器→`constrainVelocity()` 真实派发一整套 kernel→`setCollider(null,...)` 切换）跑通不抛错，这本身有价值（能抓到 API 用错/参数个数不对/方法名拼错），但**读回来的数字不是正确性验证**（见下一条）。
- `00-grid-math/` 三个数值测试全部读回 0——排查过（见 `examples/00-grid-math/main.js` 头部注释）：单线程 kernel assign 一个常量能正确写回，问题具体是"kernel 内部读取另一个已有数据的 field"这个操作本身在这个 fallback 后端读不到正确值，不管数据是 `fromArray()` 写的还是另一个 kernel 写的。这是 tsl_array_n 历史上第三次遇到"sandbox WebGL2 fallback 在多 kernel 场景下给出错误结果，真实 WebGPU 上确认没问题"（前两次是 `Loop()` 计数器和 `array0` 多线程共读）——大概率同一类环境限制，但**这次还没有实机 WebGPU 复核**。

**`grid_math.js` 的数值正确性（包括 mat2 顺序问题）、`grid_blocked_boundary_condition_solver2.js` 的实际物理行为，都还没有得到确认，需要在真实 WebGPU 环境跑一遍这两个 example 才能确认。**

## 依赖

- [tsl_array_n](../tsl_array_n)（peerDependency，工作区内联到本地包）
- [three.js](https://threejs.org/) `>=0.180.0`（peerDependency）

## 开发

```bash
npm test -w fluxflow        # vitest，图构建/纯CPU几何/hooks编排，不需要真 GPU
npm run dev -w fluxflow      # vite dev server，跑 examples/
```

## License

[Apache License 2.0](LICENSE) © 2026 bert wang——跟移植来源的 Python `fluxflow` 项目保持一致的协议。

`src/grid/` 移植自另一个项目（`D:\OneDrive\04_lib_fluxflow`，同样是 Apache License 2.0），其中一部分文件再往上追溯到 [fluid-engine-dev](https://github.com/doyubkim/fluid-engine-dev)（MIT）——完整的来源链条和各文件的对应关系见 [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)。
