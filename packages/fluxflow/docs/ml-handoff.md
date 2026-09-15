# ML work: state, what to run on real hardware, and what comes next

**Purpose of this file.** The machine-learning work in `src/ml/` was built
in a container with **no GPU** (`/dev/dri` absent), so every claim about
*speed* is still unmeasured and two documents currently carry estimates
where they should carry measurements. This file is the handoff: what
exists, exactly what to run on a machine with real WebGPU, what to look
for, and where the numbers go when they come back.

Written for someone (or some session) picking this up cold. Nothing here
assumes the conversation that produced it.

Branch: `claude/fluid-simulation-ml-research-58so76`.

---

## 0. TL;DR

| | state |
| --- | --- |
| `docs/machine-learning-fluid-research.md` | Survey of the field, filtered through this package's cost model. Complete. |
| `src/ml/` — inference kernels | Built. 59 tests pass. **No ML library, no dependency added.** |
| `src/ml/unet.js` + `examples/31-null-net-probe/` | Built. **Never run.** Prices the candidate shape for a learned pressure solve. |
| `src/ml/superres.js` + `examples/32-superres-smoke/` | Built. **Never run.** Coarse simulation, fine render. Untrained but useful (degrades to bicubic exactly). |
| Training | **Does not exist.** No autodiff, no optimiser, no loss, no weights. Plan in section 5. |

Three things need a GPU. They are in section 3, in priority order.

---

## 1. Why this needs your machine

The container had no GPU at all, so WebGPU there would fall back to a
software rasteriser and every timing would be meaningless. Separately,
this package has an established history of the **WebGL2 fallback** giving
wrong answers on repeated same-buffer dispatch/readback cycles — see
`examples/06-multigrid-preconditioner/`'s header comment, which records a
full investigation into exactly that. So:

> **If the page reports a backend that is not `WebGPUBackend`, stop. The
> numbers are not evidence of anything.** Both pages print the backend at
> the top for this reason.

---

## 2. Running it

```bash
# from the repo root
npm install
npm test                                  # 439 tests, all should pass
npm run dev -w fluxflow                   # vite, serves packages/fluxflow
```

Then in a **WebGPU-capable browser** (Chrome/Edge current, or Firefox on
Windows ≥ 141):

- `http://localhost:5173/examples/31-null-net-probe/`
- `http://localhost:5173/examples/32-superres-smoke/`
- `http://localhost:5173/examples/15-flow-past-cylinder/?profile=1`

---

## 3. The three runs, in priority order

### Run 1 — the null-net probe (`examples/31-null-net-probe/`)

**The question.** A solver step here is ~1416 dispatches, a
data-dependent iteration count, and four-plus host round trips at ~1.1 ms
each. A feed-forward network of the candidate shape is 12 dispatches, one
submission, zero round trips. Does that translate into wall-clock time?

The network has **random weights**. It computes nonsense deliberately —
what is being priced is the *shape*, and a shape can be priced before any
training run exists.

**What the page does**, in order:
1. Checks the GPU forward pass against a float64 JavaScript reference on
   the same weights. Without this the timings below are timings of an
   unknown computation.
2. Checks that two forward passes are bit-identical (there are no atomics
   in the module, so they must be).
3. Times the forward pass, batched and unbatched.
4. Times one MGPCG solve on the same 64×64 grid, **with the defaults
   `grid_pressure_solver2.js` actually ships** (GPU-resident scalars and
   setup, batched iterations, residual check every 4) rather than
   `solve()`'s own slower parameter defaults.
5. Prints the ratio.

**What to look for:**

| check | pass |
| --- | --- |
| backend | `WebGPUBackend` |
| GPU forward vs float64 reference | relative diff < 1e-4, all finite |
| two passes bit-identical | ✓ |
| every solve converged | `20/20`, `stoppedBy: none` |

**The verdict, and it is stated on the page too:**

- **≥ 2× faster than the solve** → the shape is affordable and the
  question becomes accuracy. Family A of the research document gets a
  budget.
- **< 2×** → not enough headroom. A trained network has to buy accuracy
  with its speed, and family A gives accuracy up (an unbounded residual
  against a liquid this repo has already measured collapsing — see
  `docs/project-history.md` Debugging #5). Record it as a negative result
  in the research document and stop.

**Record:** the whole output block, verbatim.

---

### Run 2 — super-resolution (`examples/32-superres-smoke/`)

**The question.** Two, actually, and only one of them is about
machine learning.

The 96×128 smoke scene shown at 384×512 four ways: nearest (what ships
today — a grid-resolution canvas scaled by CSS `image-rendering:
pixelated`), bilinear, monotonic bicubic, and the network.

**Panel 4 is pixel-identical to panel 3 by construction.** The network's
weights are zero and its residual is added to a bicubic base, so it
computes bicubic exactly. That identity is a **correctness check on the
GPU pipeline**, not a super-resolution demo: it exercises the trunk
convolutions, the `factor²` head, the sub-pixel rearrange's index
arithmetic, the half-cell sample shift and the residual add all at once.

**What to look for:**

| check | pass |
| --- | --- |
| backend | `WebGPUBackend` |
| `network ≡ bicubic` line | ✓, max \|diff\| < 1e-6 |
| panels 1 vs 2/3 | the visible question — is the classical upgrade worth shipping on its own? |

A failing identity means a bug in one of the five things listed above, and
it is the most useful failure in this whole handoff because it localises
to a small amount of code.

**The non-ML decision this run settles.** If bicubic (panel 3) looks
materially better than nearest (panel 1), that is a shippable improvement
**today**, with one dispatch and no machine learning at all —
`ml.createClassicalUpsampler2`. The survey assumed bicubic was already the
baseline; it is not, nearest is.

**Record:** a screenshot of the four panels, and the timing block.

> **Do not read the per-arm timings as application costs.** Each includes a
> 384×512 GPU→CPU readback, because that is how every example in this
> package draws (readback → `ImageData` → `putImageData`). A real renderer
> would keep the field on the GPU and sample it as a texture. The readback,
> not the network, is most of what those numbers measure. They are valid
> for comparing the four arms against each other and invalid for anything
> else.

---

### Run 3 — re-measure the frame (`examples/15-flow-past-cylinder/?profile=1`)

**Why.** `docs/machine-learning-fluid-research.md` Part 0 says the GPU is
"~3–7% busy", and that is an **estimate**, derived from a 2026-09-13
measurement taken before four optimisations landed (CG-iteration
submission batching, stage batching, the interval-4 residual check,
`seedScalarsKernel`). None of those reduce GPU arithmetic — they remove
host overhead — so the numerator held while the denominator shrank and the
percentage went **up**. By how much is unknown.

It is the denominator for every efficiency claim in that document,
including Run 1's.

**How:** the page already carries the harness. In the console:

```js
await window.__fluxflowProbe.profile( 60 )
```

Worth folding into the same session, since both were measured before
`seedScalarsKernel` and may have moved:

- is a host round trip still **1.1 ms**?
- is a CG iteration still **0.861 ms**?

**Record:** `dispatchesPerFrame`, `submissionsPerFrame`, GPU time per
frame, wall per frame, and the phase breakdown.

---

## 4. Where the numbers go when they come back

| number from | replaces |
| --- | --- |
| Run 3 | `docs/machine-learning-fluid-research.md` Part 0, "The percentage moved" — replace the ~3–7% estimate with the measurement, and drop the "has not been re-measured" paragraph |
| Run 1 | same document, Part 2 ("The shape, now built rather than estimated") gains its wall-clock row; Part 3 shortlist item 1 changes from "built and unrun" to the result |
| Run 1 verdict | if < 2×, say so plainly in Part 3 and mark family A closed |
| Run 2 | `README.md`'s `ml` section, super-resolution subsection; and Part 3 shortlist item 6 |

Every one of those places currently says, explicitly, that the figure is an
estimate and unmeasured. Keep that discipline: if a number is replaced,
say what it replaced and why it moved. This repo's documents are a record
of corrections, not a marketing page — two of the corrections already in
`machine-learning-fluid-research.md` are to its own earlier drafts.

---

## 5. What exists in `src/ml/`

**No machine-learning library.** Not PyTorch, not ONNX Runtime Web, not
TensorFlow.js, not WebNN, no WASM runtime, no pretrained weights, no
training data. Every import resolves to `tsl_array_n`, `three/tsl`, or
another file in this package, and no `package.json` changed. The reasoning
is in `layers.js`'s header: a second inference runtime is a second device
and buffer world, and this package's cost is submissions and dispatches,
so a convolution and a Laplacian have to be able to share a command buffer.
Provenance is recorded in `THIRD-PARTY-NOTICES.md` under "Provenance of
`src/ml/`".

| file | what |
| --- | --- |
| `layers.js` | conv2d, full-weighting restriction, bilinear upsample, field↔feature-map joins — TSL compute kernels |
| `unet.js` | `createUNet2` — the candidate shape for a learned pressure solve |
| `superres.js` | `createSuperResolver2`, `createClassicalUpsampler2` — the display path |
| `reference.js` | float64 JavaScript implementations of all of the above, plus seeded init, PyTorch layout conversion, and a plan executor |

**Exact, computed from the built networks** (these are not estimates):

| | `createUNet2([64,64], 16ch, 3 levels)` | `createSuperResolver2([96,128], 4×, 16ch, 3 layers)` |
| --- | --- | --- |
| dispatches | 12 | 5 |
| submissions | 1 | 1 |
| host round trips | 0 | 0 |
| parameters | 14,225 (55.6 KiB) | 7,120 (27.8 KiB) |
| arithmetic | 51.9 MFLOP | 86.7 MMAC |

Two design decisions are load-bearing and easy to undo by accident:

1. **The transfer pair is adjoint** (`R = Pᵀ/4` in 2D). `unet.js` uses
   full-weighting restriction, **not** average pooling, because average
   pooling paired with bilinear upsampling is not adjoint, and a symmetric
   operator is what keeps the shape usable as a CG preconditioner rather
   than only as a filter. `multigrid.js`'s comments record the
   confirmed-on-hardware divergence bug this codebase already paid for
   when a V-cycle was not symmetric. `test/ml.test.js` verifies it
   directly on square and non-square grids.
2. **Super-resolution degrades to bicubic exactly.** The network predicts
   a residual added to the classical upsample *inside the rearrange
   kernel*, so zero weights give bicubic pixel for pixel — tested at every
   pixel for both bases. That is what makes it shippable untrained, makes
   training purely additive, and makes a half-trained network read as
   "bicubic plus garbage" rather than as "garbage".

**Everything here is inference.** There is no `backward`, no `gradient`,
no `optimizer`, no loss anywhere in the folder. Weights come from
`randomize()` (seeded, meaningless) or `loadWeights()` (the seam for
offline training).

---

## 6. Training plan — super-resolution

Six phases. Each has one measurement that can kill it, in the order that
kills it cheapest.

### Phase 0 — prove there is anything to win *(half a day, no GPU needed)*

Take a high-resolution smoke sequence, downsample it to 96×128, upsample
back to 384×512 with the **existing** bicubic, and compute PSNR/SSIM
against the original.

If bicubic is already at ~40 dB there is almost nothing left for a network
to take, and the direction should be downgraded before any framework is
installed. Reuses code that already exists; needs no training.

**This is the step most likely to kill the whole thing, so it goes first.**

### Phase 1 — data generation *(1–2 days)*

Add a `?dump=` mode to an example that writes the density field per frame.
One real decision, and it is the substantive one:

| how the low-resolution input is produced | the problem |
| --- | --- |
| **(a) downsample a high-resolution simulation** | The standard SR setup, what tempoGAN did. But the real input at inference is **not** a downsampled fine field — it is an independently-run coarse simulation, with its own numerical diffusion and missing vortices. Different distribution. |
| **(b) pair a coarse and a fine simulation** | Right distribution, but turbulence is chaotic: they diverge after tens of frames and stop being pairs. |

**Start with (a).** It is tractable and it is the literature's choice;
quantify the distribution gap in phase 5 rather than trying to solve it
first.

Target ~2000 frames across several initial conditions (source position,
buoyancy factor). 384×512 is 16× the cells of 96×128 — fine, this is
offline.

### Phase 2 — offline PyTorch training *(2–3 days)*

A `.py` **outside this package**; no dependency enters the repo. It must
mirror the inference path item for item:

- clamp padding, not zero
- sub-pixel head emitting `factor²` channels, linear (no activation)
- **the residual base must be this package's monotonic bicubic**, not
  `F.interpolate`

That last one is the likeliest way to waste a week. If the training-time
base differs from the inference-time base, the network learns a residual
against a different function and is simply wrong once loaded. Either port
`monotonicCubic1d` into the training script or precompute the base into
the dataset.

Loss: L1 or MSE to start.

### Phase 3 — export and load *(half a day)*

`fromPyTorchConv2dWeights()` and `loadWeights()` already exist. Add an
export script and one cross-framework check: same input, PyTorch output vs
browser GPU output, compared element by element. This is the same class of
check `forwardReference` already does, extended across frameworks.

### Phase 4 — static quality

Add a fifth panel to `examples/32-superres-smoke/`. Acceptance:

- PSNR/SSIM gain over bicubic (phase 0 gave the ceiling)
- per-frame cost still within budget
- a blind look with the labels off — can you pick the network?

### Phase 5 — temporal coherence *(the hard half)*

The phase-4 network **will flicker**. That is not a risk, it is tempoGAN's
central finding: detail synthesised per frame is uncorrelated between
frames, which reads as boiling.

A simulation is better placed than video here, because the velocity field
is right there: advect the previous high-resolution output forward and
feed it as an extra input channel. `inChannels` is already the seam;
`advection_solver2.js` has the semi-Lagrangian backtrace. Add a temporal
term to the loss. Acceptance: frame-to-frame difference energy over ~500
frames, against bicubic's.

### Two expectations to set before starting

**MSE-trained super-resolution will look like slightly sharper bicubic.**
That is not undertraining — the MSE-optimal answer is blurry. The striking
results in this literature come from **adversarial** training, which is
where tempoGAN spent essentially all of its complexity and which is a
different order of difficulty to stabilise. There is a decision point
after phase 4: accept the modest gain, or go adversarial. Do phases 0–4
first regardless, because without that baseline and those metrics there is
no way to tell whether the adversarial version is better.

**A smoke density field is an easy super-resolution target.** It is
already smooth, and bicubic is already decent. The real value in this
direction is not reconstructing missing pixels — it is *synthesising
turbulent detail that was never simulated*, which is a generative problem,
not a reconstruction one. Phase 0's number tells you where that line sits.

---

## 7. Notes for whoever picks this up

- `npm test` from the repo root runs both packages: 439 tests, all
  passing at the time of writing (`ml.test.js` 33, `ml_superres.test.js`
  26).
- The GPU kernels are covered **structurally** only, per this package's
  established convention (`test/multigrid.test.js`'s header explains why).
  The arithmetic is pinned against `reference.js`, and the two examples
  close the loop on hardware.
- `reference.js` executes the `plan` that `createUNet2` /
  `createSuperResolver2` emit, rather than a hand-transcription of the
  architecture. Keep it that way: a transcribed reference drifts the first
  time the architecture changes, and then disagrees for a reason that looks
  like a GPU bug.
- Two corrections are already recorded in
  `docs/machine-learning-fluid-research.md` against its own earlier
  drafts — a unit error (frame versus solver step) and a stale-by-four-
  optimisations percentage. Do not quietly fix a number; say what moved
  and why.
