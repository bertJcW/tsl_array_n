# tsl-kernel monorepo

This repository is an npm workspaces monorepo, where each subdirectory is an independent package -- **each with its own license; there is no single license covering the repo as a whole** (see the root [LICENSE](LICENSE)):

| Package | Description | License | Status |
|---|---|---|---|
| [`packages/tsl_array_n`](packages/tsl_array_n) | A general-purpose GPU parallel-compute library wrapping three.js TSL's compute capabilities with Taichi Lang's mental model | MIT | Published on [npm](https://www.npmjs.com/package/tsl_array_n) |
| [`packages/fluxflow`](packages/fluxflow) | A browser-side GPU fluid-simulation library built on `tsl_array_n`, ported from a Taichi Lang fluid library | Apache-2.0 | Early stage -- `grid/` (grid data structures + boundary-condition solver) has been ported; a complete solver (pressure projection/advection/viscosity) hasn't been built yet; `private: true`, unpublished |

See each package's own README / LICENSE / THIRD-PARTY-NOTICES.md for details and third-party attributions.

## Development

```bash
npm install               # installs all sub-package dependencies, auto-symlinks intra-workspace deps
npm run dev                # starts tsl_array_n's vite dev server (equivalent to npm run dev -w tsl_array_n)
npm test                   # runs every sub-package's tests
```

To act on a single sub-package, use `-w <package name>`, e.g. `npm test -w tsl_array_n`.

## Why a monorepo

`fluxflow` needs to iterate against `tsl_array_n` frequently during development (test against source changes immediately, without publishing first). Sub-packages only import each other by package name (`import * as tsl_array_n from 'tsl_array_n'`), never via cross-package relative paths into each other's `src/` -- so once a sub-package matures, its directory can be split into its own repository directly, with no code changes needed.
