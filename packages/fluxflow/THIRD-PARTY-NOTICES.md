# Third-Party Notices

fluxflow does not vendor or bundle any third-party source code as a runtime
dependency — the published package would ship only the files under `src/`.
This notice covers two things: (1) third-party packages fluxflow requires or
uses during development, and (2) the provenance of `src/grid/` and
`src/noise/`, both of which are ports of code from other projects, not
written from scratch.

## Runtime dependency

### three.js

- **Relationship:** required peer dependency (`three >= 0.180.0`), same as `tsl_array_n`. Not bundled.
- **License:** MIT
- **Homepage:** https://threejs.org/
- **Repository:** https://github.com/mrdoob/three.js
- **License text:** https://github.com/mrdoob/three.js/blob/dev/LICENSE

## Development-only dependency

Used to develop and preview this package's examples and tests; never published
or distributed as part of the `fluxflow` package.

### Vite

- **License:** MIT
- **Homepage:** https://vite.dev/
- **Repository:** https://github.com/vitejs/vite
- **License text:** https://github.com/vitejs/vite/blob/main/LICENSE

### Vitest

- **License:** MIT
- **Homepage:** https://vitest.dev/
- **Repository:** https://github.com/vitest-dev/vitest
- **License text:** https://github.com/vitest-dev/vitest/blob/main/LICENSE

## Provenance of `src/grid/`

`src/grid/` is a JavaScript/TSL port of the `grid/` folder from a separate,
private Python project of the same author, **fluxflow (Python)**, located at
`D:\OneDrive\04_lib_fluxflow` — not the same repository as this one, despite
the shared name. That project is itself licensed under the **Apache License
2.0**, and its own `THIRD-PARTY-NOTICES.txt` records that several of the files
being ported here were themselves originally ported into that project from
**fluid-engine-dev**, a C++ library, under the **MIT License**. So the actual
chain for most of this folder is:

```
fluid-engine-dev (C++, MIT, Doyub Kim)
  -> fluxflow (Python/Taichi, Apache-2.0, bert wang)
    -> fluxflow (this package, JS/TSL, Apache-2.0, bert wang)
```

This package is itself licensed under the Apache License 2.0, same as its
Python counterpart — see `LICENSE`.

Every file below carries a header comment in its `src/` form crediting the
Python file it was ported from and describing what changed, satisfying the
Apache License's "state that you changed the file" condition at the file
level; this notice covers the rest.

### fluid-engine-dev (MIT) — via fluxflow (Python)

- **Source:** https://github.com/doyubkim/fluid-engine-dev
- **License:** MIT

Files in this package derived from `fluid-engine-dev` (through the Python
`fluxflow` project's own `grid/constant.py`, `grid_data2.py`, `grid_math.py`,
`grid_solver2.py`, `sdf_collider2.py` — exactly the set that Python project's
own `THIRD-PARTY-NOTICES.txt` lists under this same upstream):

- `src/grid/constant.js`
- `src/grid/grid_data2.js`
- `src/grid/grid_math.js`
- `src/grid/grid_solver2.js`
- `src/grid/sdf_collider2.js`

> The MIT License (MIT)
>
> Copyright (c) 2018 Doyub Kim
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

### fluxflow (Python) (Apache License 2.0)

- **Source:** `D:\OneDrive\04_lib_fluxflow` (private, same author, separate repository)
- **License:** Apache License 2.0

All of `src/grid/` and `src/noise/` is ported from that project (the five
`grid/` files above, plus `array_utils.py`, `level_set_utils.py`,
`grid_blocked_boundary_condition_solver2.py`, and `noise/noise.py` — the
first three of those are not separately attributed to `fluid-engine-dev` in
that project's own notices and are treated here as that project's own
original work; `noise/noise.py` has its own separate upstream, WebGL-Noise,
covered in its own section below). `src/grid/polygon_sdf.js` and
`src/grid/svg_utils.js` have no Python counterpart — they're new code written
for this port to replace two Python-only dependencies (`shapely`, `svg.path`)
that have no browser equivalent; see `sdf_collider2.js`'s header comment for
the approach.

>                                  Apache License
>                            Version 2.0, January 2004
>                         http://www.apache.org/licenses/
>
>    TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
>
>    1. Definitions.
>
>       "License" shall mean the terms and conditions for use, reproduction,
>       and distribution as defined by Sections 1 through 9 of this document.
>
>       "Licensor" shall mean the copyright owner or entity authorized by
>       the copyright owner that is granting the License.
>
>       "Legal Entity" shall mean the union of the acting entity and all
>       other entities that control, are controlled by, or are under common
>       control with that entity. For the purposes of this definition,
>       "control" means (i) the power, direct or indirect, to cause the
>       direction or management of such entity, whether by contract or
>       otherwise, or (ii) ownership of fifty percent (50%) or more of the
>       outstanding shares, or (iii) beneficial ownership of such entity.
>
>       "You" (or "Your") shall mean an individual or Legal Entity
>       exercising permissions granted by this License.
>
>       "Source" form shall mean the preferred form for making modifications,
>       including but not limited to software source code, documentation
>       source, and configuration files.
>
>       "Object" form shall mean any form resulting from mechanical
>       transformation or translation of a Source form, including but
>       not limited to compiled object code, generated documentation,
>       and conversions to other media types.
>
>       "Work" shall mean the work of authorship, whether in Source or
>       Object form, made available under the License, as indicated by a
>       copyright notice that is included in or attached to the work
>       (an example is provided in the Appendix below).
>
>       "Derivative Works" shall mean any work, whether in Source or Object
>       form, that is based on (or derived from) the Work and for which the
>       editorial revisions, annotations, elaborations, or other modifications
>       represent, as a whole, an original work of authorship. For the purposes
>       of this License, Derivative Works shall not include works that remain
>       separable from, or merely link (or bind by name) to the interfaces of,
>       the Work and Derivative Works thereof.
>
>       "Contribution" shall mean any work of authorship, including
>       the original version of the Work and any modifications or additions
>       to that Work or Derivative Works thereof, that is intentionally
>       submitted to Licensor for inclusion in the Work by the copyright owner
>       or by an individual or Legal Entity authorized to submit on behalf of
>       the copyright owner. For the purposes of this definition, "submitted"
>       means any form of electronic, verbal, or written communication sent
>       to the Licensor or its representatives, including but not limited to
>       communication on electronic mailing lists, source code control systems,
>       and issue tracking systems that are managed by, or on behalf of, the
>       Licensor for the purpose of discussing and improving the Work, but
>       excluding communication that is conspicuously marked or otherwise
>       designated in writing by the copyright owner as "Not a Contribution."
>
>       "Contributor" shall mean Licensor and any individual or Legal Entity
>       on behalf of whom a Contribution has been received by Licensor and
>       subsequently incorporated within the Work.
>
>    2. Grant of Copyright License. Subject to the terms and conditions of
>       this License, each Contributor hereby grants to You a perpetual,
>       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
>       copyright license to reproduce, prepare Derivative Works of,
>       publicly display, publicly perform, sublicense, and distribute the
>       Work and such Derivative Works in Source or Object form.
>
>    3. Grant of Patent License. Subject to the terms and conditions of
>       this License, each Contributor hereby grants to You a perpetual,
>       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
>       (except as stated in this section) patent license to make, have made,
>       use, offer to sell, sell, import, and otherwise transfer the Work,
>       where such license applies only to those patent claims licensable
>       by such Contributor that are necessarily infringed by their
>       Contribution(s) alone or by combination of their Contribution(s)
>       with the Work to which such Contribution(s) was submitted. If You
>       institute patent litigation against any entity (including a
>       cross-claim or counterclaim in a lawsuit) alleging that the Work
>       or a Contribution incorporated within the Work constitutes direct
>       or contributory patent infringement, then any patent licenses
>       granted to You under this License for that Work shall terminate
>       as of the date such litigation is filed.
>
>    4. Redistribution. You may reproduce and distribute copies of the
>       Work or Derivative Works thereof in any medium, with or without
>       modifications, and in Source or Object form, provided that You
>       meet the following conditions:
>
>       (a) You must give any other recipients of the Work or
>           Derivative Works a copy of this License; and
>
>       (b) You must cause any modified files to carry prominent notices
>           stating that You changed the files; and
>
>       (c) You must retain, in the Source form of any Derivative Works
>           that You distribute, all copyright, patent, trademark, and
>           attribution notices from the Source form of the Work,
>           excluding those notices that do not pertain to any part of
>           the Derivative Works; and
>
>       (d) If the Work includes a "NOTICE" text file as part of its
>           distribution, then any Derivative Works that You distribute must
>           include a readable copy of the attribution notices contained
>           within such NOTICE file, excluding those notices that do not
>           pertain to any part of the Derivative Works, in at least one
>           of the following places: within a NOTICE text file distributed
>           as part of the Derivative Works; within the Source form or
>           documentation, if provided along with the Derivative Works; or,
>           within a display generated by the Derivative Works, if and
>           wherever such third-party notices normally appear. The contents
>           of the NOTICE file are for informational purposes only and
>           do not modify the License. You may add Your own attribution
>           notices within Derivative Works that You distribute, alongside
>           or as an addendum to the NOTICE text from the Work, provided
>           that such additional attribution notices cannot be construed
>           as modifying the License.
>
>       You may add Your own copyright statement to Your modifications and
>       may provide additional or different license terms and conditions
>       for use, reproduction, or distribution of Your modifications, or
>       for any such Derivative Works as a whole, provided Your use,
>       reproduction, and distribution of the Work otherwise complies with
>       the conditions stated in this License.
>
>    5. Submission of Contributions. Unless You explicitly state otherwise,
>       any Contribution intentionally submitted for inclusion in the Work
>       by You to the Licensor shall be under the terms and conditions of
>       this License, without any additional terms or conditions.
>       Notwithstanding the above, nothing herein shall supersede or modify
>       the terms of any separate license agreement you may have executed
>       with Licensor regarding such Contributions.
>
>    6. Trademarks. This License does not grant permission to use the trade
>       names, trademarks, service marks, or product names of the Licensor,
>       except as required for reasonable and customary use in describing the
>       origin of the Work and reproducing the content of the NOTICE file.
>
>    7. Disclaimer of Warranty. Unless required by applicable law or
>       agreed to in writing, Licensor provides the Work (and each
>       Contributor provides its Contributions) on an "AS IS" BASIS,
>       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
>       implied, including, without limitation, any warranties or conditions
>       of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
>       PARTICULAR PURPOSE. You are solely responsible for determining the
>       appropriateness of using or redistributing the Work and assume any
>       risks associated with Your exercise of permissions under this License.
>
>    8. Limitation of Liability. In no event and under no legal theory,
>       whether in tort (including negligence), contract, or otherwise,
>       unless required by applicable law (such as deliberate and grossly
>       negligent acts) or agreed to in writing, shall any Contributor be
>       liable to You for damages, including any direct, indirect, special,
>       incidental, or consequential damages of any character arising as a
>       result of this License or out of the use or inability to use the
>       Work (including but not limited to damages for loss of goodwill,
>       work stoppage, computer failure or malfunction, or any and all
>       other commercial damages or losses), even if such Contributor
>       has been advised of the possibility of such damages.
>
>    9. Accepting Warranty or Additional Liability. While redistributing
>       the Work or Derivative Works thereof, You may choose to offer,
>       and charge a fee for, acceptance of support, warranty, indemnity,
>       or other liability obligations and/or rights consistent with this
>       License. However, in accepting such obligations, You may act only
>       on Your own behalf and on Your sole responsibility, not on behalf
>       of any other Contributor, and only if You agree to indemnify,
>       defend, and hold each Contributor harmless for any liability
>       incurred by, or claims asserted against, such Contributor by reason
>       of your accepting any such warranty or additional liability.
>
>    END OF TERMS AND CONDITIONS

## Provenance of `src/noise/`

`src/noise/noise.js` is a JavaScript/TSL port of `noise/noise.py` from the
same Python `fluxflow` project referenced above (Apache License 2.0). That
file's own header comment records that it is itself ported from
**WebGL-Noise**, under the **MIT License**:

```
WebGL-Noise (C++/GLSL, MIT, Ashima Arts & Stefan Gustavson)
  -> fluxflow (Python/Taichi, Apache-2.0, bert wang)
    -> fluxflow (this package, JS/TSL, Apache-2.0, bert wang)
```

### WebGL-Noise (MIT) — via fluxflow (Python)

- **Source:** https://github.com/ashima/webgl-noise
- **License:** MIT

Files in this package derived from WebGL-Noise (through the Python
`fluxflow` project's `noise/noise.py`):

- `src/noise/noise.js`

> Copyright (C) 2011 by Ashima Arts (Simplex noise)
>
> Copyright (C) 2011-2016 by Stefan Gustavson (Classic noise and others)
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in
> all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
> THE SOFTWARE.

This file is also, like everything under `src/grid/`, a derivative work of
the Python `fluxflow` project itself (Apache License 2.0) — see that
project's own license text already reproduced above, under "fluxflow
(Python) (Apache License 2.0)"; that section's terms apply here too, this
isn't repeated a second time.

## Design inspiration (not a code dependency)

### Taichi Lang

The Python `fluxflow` project (and therefore this port's algorithms) is
written against [Taichi Lang](https://github.com/taichi-dev/taichi)'s
programming model. No Taichi source code itself is used or copied here — this
is a credit for the language/tool, not a licensing dependency for Taichi
itself (the actual ported algorithms are attributed above, to fluid-engine-dev
and to the Python `fluxflow` project). Taichi is distributed under the Apache
License 2.0 (https://github.com/taichi-dev/taichi/blob/master/LICENSE).
