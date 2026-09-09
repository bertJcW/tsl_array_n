# Archived research: bubbles via Constraint Bubbles (Houdini's approach)

**Status: archived, not implemented.** Recorded here so the investigation does not
have to be repeated when this direction is revived. The active two-phase work moved
to miscible mixing (two liquids / dye) instead -- see
`src/grid/grid_two_phase_flip_solver2.js`.

## The finding

Houdini does **not** do MultiFLIP-style two-phase for bubbles. Its FLIP Solver has a
parameter called **Enforce Air Incompressibility**, and SideFX's own documentation
describes it (paraphrased -- see the licensing note below on why this is not quoted
verbatim) as: it does not solve for velocity inside the air at all, it only constrains
the liquid from compressing or expanding that air volume. The docs also warn that with
dynamically resized fields the enforcement may fail to *detect the enclosed air
region*, which is the tell that the mechanism is per-air-region rather than per-cell.

That is **Constraint Bubbles** (Goldade & Batty): add one constraint per *disconnected*
air region enforcing zero net flux across its whole surface. The paper says the method
was incorporated into commercial fluid animation software; the matching Houdini feature
plus SideFX's support of the authors' research makes Houdini the near-certain referent,
though the paper does not name it.

## Why it is attractive

| | MultiFLIP-style (what this port built) | Constraint Bubbles |
|---|---|---|
| Air | simulated with particles | not simulated at all |
| Particles | whole domain | liquid only |
| Pressure system | variable-coefficient, conditioning degrades with density ratio | stays constant-coefficient, sparse, SPD |
| Cost | markedly higher | paper reports <10% over a free-surface solve |
| New machinery needed | variable-coefficient multigrid | connected-component labelling of air regions |

The connected-component labelling is the one genuinely hard part on a GPU, and is the
reason this was not attempted yet.

## Sources (all usable, all requiring attribution when used)

- Goldade, R. & Batty, C. *Constraint Bubbles: Adding Efficient Zero-Density Bubbles to
  Incompressible Free Surface Flow.* arXiv:1711.11470.
- Goldade, R., Wang, Y., Aanjaneya, M. & Batty, C. *Constraint Bubbles and Affine
  Regions: Reduced Fluid Models for Efficient Immersed Bubbles and Flexible Spatial
  Coarsening.* ACM TOG 39(4) (SIGGRAPH 2020).
- **Source code, MIT:** https://github.com/rgoldade/ReducedFluids -- reference
  implementation of the above. MIT is compatible with this package's Apache-2.0, same
  as the existing jet upstream. Attribution would go in `THIRD-PARTY-NOTICES.md` if any
  of it is used.
- **Also MIT, separately useful:** https://github.com/rgoldade/GeometricMultigridPressureSolver
  -- potentially relevant to the constant-coefficient coarse-level limitation recorded
  in `src/linalg/multigrid.js` (decision 4), independently of bubbles.
- Christopher Batty's variational/ghost-fluid reference implementations:
  https://github.com/christopherbatty

## Licensing note (applies to everything Houdini-derived in this repo)

Houdini itself is proprietary; being able to *see* an HDA's internals or read the node
documentation is not a licence to copy either. What is taken from Houdini anywhere in
this project is **architectural information only** -- which stages exist, in what order,
and what they are called -- which is idea, not expression, and is reimplemented from
scratch here. Specifically NOT used: any HDK source, any verbatim documentation text,
and any node-network transliterated node-by-node. Documentation wording is paraphrased
rather than quoted for this reason. See the "Design and architecture references"
section of `THIRD-PARTY-NOTICES.md`.
