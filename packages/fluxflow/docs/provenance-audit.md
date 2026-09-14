# Provenance audit: is anything copied, and is anything unattributed?

An audit of `packages/fluxflow` against every source it cites, run on
2026-09-15. Two questions:

1. **Is there code-level copying** from any of the projects this package
   was written against?
2. **Is anything used but missing from `THIRD-PARTY-NOTICES.md`?**

**Finding: no infringement, and no unattributed source.** The evidence, the
method, and — importantly — the limits of what could be checked are below.

---

## Method

Where a source was available locally or fetchable, the check is textual
rather than a reading of intentions: both corpora are tokenised
(lowercased, punctuation stripped), cut into overlapping **10-token
n-grams**, and intersected. Ten tokens is long enough that natural
coincidence is rare and short enough to catch a copied sentence or a
transliterated expression.

| source | licence | available | how checked |
| --- | --- | --- | --- |
| jet / fluid-engine-dev | MIT | **locally** (948 files, 4.5 MB) | full n-gram intersection |
| fluxflow (Python), the direct ancestor | Apache-2.0 | **locally** (520 files, 2.6 MB) | full n-gram intersection |
| mantaflow | Apache-2.0 | not locally | two most specific claims fetched from upstream and compared by hand |
| OpenFOAM | **GPL-3.0** | not locally, deliberately | absence check on the doc that cites it |
| Houdini | proprietary | n/a | architecture reference only; no code claim to check |
| three.js, Vite, Vitest | MIT | dependencies | notices check |

---

## 1. Code-level similarity

### jet / fluid-engine-dev — one match, and it is mathematics

Across 4.5 MB of C++ and 39 JavaScript files, the intersection is
**exactly one 10-gram**, in `src/grid/grid_pressure_solver2.js`:

```
u0 i 1 j u i 1 j invh x
```

That is a finite-difference index expression — `(u(i+1,j) - u(i,j)) * invHx`,
the discrete divergence. It is the formula, not a way of writing it, and it
would be written the same way by anyone discretising the same operator. jet
is MIT and attributed regardless.

### fluxflow (Python) — matches expected, licensed, and attributed

This is the **declared parent**: the package is a port of it, it is
Apache-2.0 (the same licence), and it has its own `THIRD-PARTY-NOTICES.md`
entry. Sharing text with it is the point, not a problem.

| file | shared 10-grams with |
| --- | --- |
| `src/noise/noise.js` | `noise.py` (96) |
| `src/grid/grid_math.js` | `grid_math.py` (83) |
| `src/linalg/multigrid.js` | `linalg.py` (9) |
| `src/linalg/linalg.js` | `linalg.py` (3) |
| `src/grid/grid_blocked_boundary_condition_solver2.js` | (1) |

Worth singling out what the `noise.js` matches actually are. They include:

```
copyright c 2011 by ashima arts simplex noise copyright c 2011
```

That is the **Ashima WebGL-Noise copyright header being carried through**,
which is exactly the required behaviour for MIT-licensed material passing
through two ports. The audit finding here is that attribution survived the
journey, not that something leaked.

The `grid_math.js` matches are the same function ported (`i0c j0c i1c j1c
w00 w10 w01 w11` — the bilinear weight names) plus its explanatory prose.
The `multigrid.js` matches are a carried-over design comment.

### mantaflow — the most-cited source, and the one with a gap

mantaflow is referenced 127 times across 12 source files and has six
`THIRD-PARTY-NOTICES.md` entries — more than any other source. **No local
copy exists**, so a full intersection was not possible. Instead the two
most specific claims were fetched from upstream and compared by hand.

**The FLIP/PIC velocity blend.** mantaflow's
`knMapLinearMACGridToVec3_FLIP`:

```cpp
pvel[idx] = flipRatio * (pvel[idx] + delta) + (1.0 - flipRatio) * v;
```

against `src/grid/grid_flip_solver2.js`:

```js
const blended = flipVel.mul( flipRatioNode ).add( newVel.mul( float( 1 ).sub( flipRatioNode ) ) );
```

Same formula, and it is **neither project's formula** — it is the PIC/FLIP
blend from Zhu & Bridson (2005), which the notices already cite as the
academic source. Different names, different structure, different language.

**The symmetric smoother, the most specific claim in the notices.**
mantaflow's `GridMg::smoothGS(int l, bool reversedOrder)` builds a
colour-offset table for 2D/3D and 5/7/9/27-point stencils, divides the grid
into 2×2 blocks for threading, and computes `reversedOrder ?
colorOffs.size()-1-c : c`. fluxflow's `relax(queue, level, iterations,
reversed)` pushes two pre-built dispatchers into a GPU submission queue in
one order or the other. **The idea is the same and is credited by name; no
line of it is shared.** mantaflow is Apache-2.0, so even a closer borrowing
would be licit with the attribution that is already there.

*Limit of this check: two functions out of a large project. The remaining
mantaflow claims rest on the notices' own descriptions rather than on a
textual comparison.* Given mantaflow's licence is Apache-2.0 — the same as
this package's — the legal exposure of that gap is low; the correctness of
the attributions is what it leaves unverified.

### OpenFOAM — the one that actually matters, and it is clean

OpenFOAM is **GPL-3.0**, which is incompatible with this package's
Apache-2.0. The stated rule was methods only: no source read into code, no
transliteration, everything restated as mathematics and in this project's
own prose.

`docs/openfoam-two-phase-flow.md` (506 lines) contains **zero** lines
matching `void|scalar|volScalarField|fvm::|forAll|#include` — i.e. no C++
at all, not even a snippet. The two source files that mention OpenFOAM cite
it as a method reference beside the original papers (Rusche 2002 for the
reduced pressure, Rudman 1998 for momentum-consistent transport), which is
what the notices describe.

The constraint was respected.

---

## 2. Attribution completeness

### Every source named in `src/` has a notices entry

| name in source | files | notices |
| --- | --- | --- |
| mantaflow | 12 | ✓ (6 entries) |
| jet / fluid-engine-dev | 11 | ✓ (6 entries) |
| Stam | 6 | ✓ (academic, via jet and mantaflow entries) |
| Taichi | 7 | ✓ |
| Houdini | 2 | ✓ |
| OpenFOAM | 2 | ✓ |
| Bridson, Brackbill, Rusche, Rudman, Zalesak | 1–2 each | ✓ (academic references entry) |
| Ashima (WebGL-Noise) | 1 | ✓ |

No orphans.

### Dependencies

`three`, `tsl_array_n`, `vite`, `vitest`. three.js, Vite and Vitest each
have MIT entries; `tsl_array_n` is the sibling package in this repository
under the same licence, and is referenced in the notices.

### Apache-2.0 §4(b) — "state that You changed the files"

Of 39 source files, 22 carry an explicit port-or-change statement. Of the
remaining 17, five are `index.js` barrels with no code, and the rest either
declare originality or declare a *non*-relationship in wording a keyword
search misses:

- `external_force_solver2.js`: "No Python or jet source to port here beyond
  the general shape"
- `velocity_damping2.js`: "New file, no direct Python/jet/mantaflow source"
- `grid_adaptive_timestep2.js`: "Deliberate simplification versus jet's own
  `PhysicsAnimation::advanceTimeStep`"

`float_guards.js`, `profiling.js`, `polygon_sdf.js`, `svg_utils.js`,
`reduction.js` and `interaction/*` reference no external source at all and
are original to this package.

---

## What this audit does not establish

Stated plainly, because an audit that hides its gaps is worth less than one
that does not:

- **mantaflow was checked at two points, not exhaustively.** No local copy
  exists. The licence is compatible and the attributions are extensive, so
  the risk is to the accuracy of the attributions rather than to the right
  to use the material.
- **OpenFOAM was checked by absence, not by comparison.** That is the
  correct check for the rule that was set ("no code read"), but it cannot
  prove a negative about material that may have been paraphrased from
  memory. The mitigation already in place is that every OpenFOAM-adjacent
  method in the code cites the *original paper* rather than OpenFOAM.
- **n-grams catch copying, not independent reinvention of the same
  algorithm** — nor should they. Shared algorithms with cited sources are
  the normal and licit state of a port, and that is what this package is.

## Reproducing it

The n-gram comparison is a short script: tokenise both corpora, build
10-gram sets per file, intersect, report. It was run against
`D:/OneDrive/02_library/cpp/jet` and `D:/OneDrive/04_lib_fluxflow`. Point
it at a local mantaflow checkout to close the gap above.
