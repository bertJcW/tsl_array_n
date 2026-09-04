# tsl-kernel monorepo

这个仓库是一个 npm workspaces monorepo，每个子目录是一个独立的包，**各自license 不同，仓库整体没有统一的 license**（见根目录 [LICENSE](LICENSE)）：

| 包 | 说明 | License | 状态 |
|---|---|---|---|
| [`packages/tsl_array_n`](packages/tsl_array_n) | 用 Taichi Lang 的心智模型封装 three.js TSL 计算能力的通用 GPU 并行计算库 | MIT | 已发布 [npm](https://www.npmjs.com/package/tsl_array_n) |
| [`packages/fluxflow`](packages/fluxflow) | 基于 `tsl_array_n` 的浏览器端 GPU 流体解算库，从一个 Taichi Lang 写的流体库移植而来 | Apache-2.0 | 早期阶段——`grid/`（网格数据结构+边界条件求解器）已移植，压力投影/对流/粘性等完整求解器还没做，`private: true` 未发布 |

各包的详细说明、以及各自的第三方依赖归属，见各自目录下的 README / LICENSE / THIRD-PARTY-NOTICES.md。

## 开发

```bash
npm install              # 安装全部子包依赖，自动软链 workspace 内部依赖
npm run dev               # 启动 tsl_array_n 的 vite dev server（等价于 npm run dev -w tsl_array_n）
npm test                  # 跑所有子包的测试
```

单独对某个子包操作，用 `-w <包名>`，例如 `npm test -w tsl_array_n`。

## 为什么是 monorepo

`fluxflow` 开发阶段需要频繁联调 `tsl_array_n`（改完源码立刻能在 fluxflow 里测，不用先发布）。子包之间只通过包名互相 import（`import * as tsl_array_n from 'tsl_array_n'`），不用跨包相对路径引用彼此的 `src/`——这样等某个子包成熟后，可以直接把它的目录拆成独立仓库，代码不需要改动。
