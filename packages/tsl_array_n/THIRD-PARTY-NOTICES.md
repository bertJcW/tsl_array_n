# Third-Party Notices

tsl_array_n does not vendor or bundle any third-party source code — the published
package ships only the files under `src/`. This notice lists the third-party
packages tsl_array_n requires or uses during development, for transparency.

## Runtime dependency

### three.js

- **Relationship:** required peer dependency (`three >= 0.180.0`). Not bundled —
  consumers install their own copy. tsl_array_n's `init()` imports `WebGPURenderer`
  from `three/webgpu` at runtime.
- **License:** MIT
- **Homepage:** https://threejs.org/
- **Repository:** https://github.com/mrdoob/three.js
- **License text:** https://github.com/mrdoob/three.js/blob/dev/LICENSE

## Development-only dependency

Used to develop and preview this repository's examples; never published or
distributed as part of the `tsl_array_n` package.

### Vite

- **License:** MIT
- **Homepage:** https://vite.dev/
- **Repository:** https://github.com/vitejs/vite
- **License text:** https://github.com/vitejs/vite/blob/main/LICENSE

## Design inspiration (not a code dependency)

### Taichi Lang

tsl_array_n's API (`init`/`field`/`kernel`/`func`) is deliberately modeled after
[Taichi Lang](https://github.com/taichi-dev/taichi)'s programming model
(`ti.init`/`ti.field`/`@ti.kernel`/`@ti.func`). No Taichi source code is used —
this is a credit for the design, not a licensing dependency. Taichi itself is
distributed under the Apache License 2.0
(https://github.com/taichi-dev/taichi/blob/master/LICENSE).
