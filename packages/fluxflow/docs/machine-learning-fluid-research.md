# Machine learning for fluid solving: what exists, and what fits this package

The question this answers: **where can machine learning buy efficiency or
quality in a computer-graphics fluid solver, and which of those directions
survive this package's measured cost model?**

Two halves. The first is a survey of the field, organised by *what the
network replaces* rather than by architecture, because what it replaces is
what decides whether it is usable here. The second is the filter: this
package has an unusual cost model, already measured in detail, and that
measurement disqualifies most of the standard pitches and promotes one
that is rarely the headline.

---

## Part 0 — the constraint that filters everything

Before any of the literature, the measurements that decide which of it
applies — and a warning about their dates, because this repo's numbers move
fast and the first draft of this document got this wrong.

### The numbers, with their provenance

All from `docs/perf-investigation-cg-gpu-resident-alpha-beta.md`, example 15
(`flow-past-cylinder`, 64×64), driver paused:

| quantity | measured | date | still current? |
| --- | --- | --- | --- |
| dispatches **per solver step** | **1416 – 1434** | 2026-09-13 | yes — nothing since removes dispatches |
| dispatches **per rendered frame** | 740 – 813 | 2026-09-13 | yes, but see the unit warning below |
| `renderer.compute()` calls per frame | 259 – 278 | 2026-09-13 | **no** — CG-iteration batching removed 232 submissions/step after this |
| **GPU compute time per frame** | **0.21 – 0.40 ms** | 2026-09-13 | yes, in absolute terms |
| cost of one submission | 38.79 µs *on an idle queue* | 2026-09-13 | yes, with the caveat below |
| cost of one host round trip | **1.1 ms** | 2026-09-14 | yes |
| clean natural solver step | 18.6 – 22.4 ms | 2026-09-14 | **no** — two later changes |
| solver step after `seedScalarsKernel` | 14.8 – 26.4 ms | 2026-09-15 | most recent figure |

**A frame is not a step**, and this document's first draft conflated them.
The drivers take several solver steps per rendered frame in some scenes and
a fraction of one in others; the doc's own worked example is scene 28, where
a 16.6 ms rendered frame carried ~27 submissions against 359 per solver
step. Everything below is stated per *step*, which is the unit the solver
actually has.

### The percentage moved, and it moved against the easy conclusion

The first draft said "the GPU is idle for 99% of the frame". That figure was
taken from 2026-09-13, and **at least four optimisations landed after it**:
CG-iteration submission batching (1.15×–1.56× per step), stage batching
(6%), the residual check going periodic at interval 4 (1.24×), and
`seedScalarsKernel` removing the two setup round trips (1.144×/1.297×).

None of those make the GPU do less arithmetic. They remove host overhead.
So the numerator — 0.21–0.40 ms of actual GPU execution — is unchanged,
while the denominator shrank. **The GPU-busy share has therefore gone up,
not down.**

Roughly, and this is an estimate rather than a measurement: a step is
~1416 dispatches against the frame harness's 740–813, so ~0.4–0.8 ms of GPU
compute per step; a step is now ~15–20 ms. That puts the GPU at **~3–7%
busy, not ~1%**.

**This has not been re-measured, and it should be.** It needs real WebGPU
hardware, which the container this was written in does not have (`/dev/dri`
is absent, so any WebGPU here would be a software rasteriser and the
timings would be meaningless). It is the second measurement on the
shortlist in Part 3.

### The better framing, which the step accounting already provides

The 2026-09-14 interventional accounting is a cleaner basis for the ML
argument than any GPU-utilisation percentage, because it says where a step
goes rather than how busy a device is:

| component | ms | share |
| --- | --- | --- |
| non-pressure stages (advection, forces, dye, boundary) | 1.4 | **6%** |
| CG iterations (0.861 ms × 12) | 10.3 | **46%** |
| pressure-solve fixed cost | ~9.1 | **~40%** |

Fitted at **0.861 ms per CG iteration**, R² = 0.996, with the intercept
agreeing with the directly measured zero-iteration point to 1%.

That fixed 40% is the part worth staring at. It is, per the doc's own
inventory, roughly **four host round trips and one full V-cycle** outside
the CG loop — and a host round trip was re-measured directly at **1.1 ms**,
correcting an earlier 0.2–0.4 ms estimate by 3–5×. The document's own
verdict on it: "Nothing has ever targeted it — every optimisation in this
document attacks per-iteration cost."

### What that reorders

- **Machine learning that makes the arithmetic cheaper is worth a few
  per cent here — call it 3–7%, pending re-measurement.** That is five
  times more than the first draft claimed and still not the lever. Almost
  every ML-for-CFD paper is written against the opposite cost model: a
  large 3D grid where the solve is genuinely compute-bound and a coarser
  approximation is the whole point. That pitch does not transfer to a
  64×64 browser grid.
- **Machine learning that changes the *shape* of the computation is worth
  the step.** A neural network is feed-forward: a fixed number of layers,
  a fixed number of dispatches, no data-dependent loop length, and — the
  part that matters most here — **no convergence check, therefore no host
  round trip.** FluidNet's own framing of its advantage is exactly this:
  ConvNets have "fixed computational complexity and latency, unlike exact
  iterative solvers."

The correction to the percentage does not weaken that second point; it
**strengthens** it. A feed-forward network does not merely replace the 46%
of the step that is CG iterations. It also removes most of the untargeted
40%, because that 40% is dominated by round trips at 1.1 ms each and by the
initial preconditioner V-cycle — none of which a network has. The thing
nothing in this repo has yet attacked is precisely the thing this family of
methods deletes by construction.

So the interesting question for this package is not "can a network
approximate the pressure solve more cheaply than MGPCG". It is: **can a
network replace ~1416 dispatches per step, a data-dependent iteration
count, and ~four-plus host round trips at 1.1 ms each, with ~10–20
dispatches, 1–2 submissions and zero round trips?** That is a structural
question, and it has a very different answer.

### One honest caveat about submissions

The 38.79 µs per submission is measured **on an idle queue**, and this
matters: when the batching change removed ~10 of the remaining ~13
submissions per frame, frame time did not move. The doc's own explanation
is that in a real frame most of a submission's cost overlaps with GPU
execution already queued. So "submissions × 38.79 µs" is an upper bound on
what removing them can buy, not a prediction. The per-step frozen-workload
measurement (1.15×–1.56×) is the one that held up; the per-frame one did
not.

Three further constraints, all of them already established in this repo:

1. **A free-surface liquid does not tolerate residual divergence.**
   Logged in `docs/project-history.md`, Debugging #5: non-converged frames
   left max|div| of 0.1–0.4 against 0.01–0.03 on converged frames, and
   occupied cells collapsed from 1536 to 340–700. Anything that gives up
   the convergence guarantee needs a wrapper that restores it.
2. **Learned constants do not transfer across this package's operators.**
   Also measured: the damped-Jacobi damping factor that was free on the
   masked operator cost 1.40x on the plain one, and undamped Jacobi
   "does not converge at all" on the variable-density operator the liquid
   scenes use. A network trained on one operator is a constant of exactly
   the same shape, only bigger. This is the single most likely way for an
   ML direction to look good in a demo and fail in `examples/24`.
3. **There is no offline training pipeline, and this is a browser
   library.** Weights ship in the bundle; inference happens on arbitrary
   consumer hardware; there is no data-generation cluster. Directions that
   need a large supervised dataset are research projects, not features.
   Directions with *unsupervised* losses — where the loss is computed from
   the operator itself rather than from reference solutions — sidestep
   this entirely, and that turns out to be most of the promising ones.

---

## Part 1 — the survey, by what the network replaces

### A. The pressure solve

The largest literature, and the one aimed squarely at this package's
largest cost. Five distinct levels of invasiveness, in increasing order of
how much they can break.

#### A1. Replace the solve outright

**FluidNet** (Tompson, Schlachter, Sprechmann & Perlin, ICML 2017) is the
canonical version: standard operator splitting, and a ConvNet with a
tailored architecture replaces the linear solve. Trained unsupervised, with
a loss that minimises long-term velocity divergence rather than matching a
reference pressure — explicitly prioritising "a low divergence flow
(conforming to boundary conditions) over recovering the exact pressure or
velocity". Real-time 2D and 3D smoke, and it generalises to unseen
geometry.

The shape is ideal for this package — fixed latency, fixed dispatch count,
no readback. The correctness is not. The network produces an approximately
divergence-free field with no bound on the residual, which is the exact
condition under which this project has already measured its liquid
collapse. **Viable for smoke and fire; ruled out for the free-surface
examples unless wrapped by a guarantee-restoring outer loop.**

#### A2. Neural preconditioner inside a Krylov solver

The one that keeps the guarantee. A preconditioner changes the *rate* of
convergence, not the fixed point: a bad preconditioner costs iterations,
it cannot produce a wrong answer, and the convergence test is still doing
its job.

- **Lan et al., ICML 2024** — [A Neural-Preconditioned Poisson Solver for
  Mixed Dirichlet and Neumann Boundary
  Conditions](https://proceedings.mlr.press/v235/lan24a.html). Directly
  relevant: *mixed Dirichlet and Neumann* is exactly this package's
  situation (`grid_pressure_solver2.js` generalises jet's hardcoded-zero
  air cells into a Dirichlet-aware solve). A lightweight architecture with
  spatially varying convolution kernels, built for fast inference. Reported
  fastest on 95.6% of the systems tested, accounting for 98.0% of total
  solve time, beating algebraic multigrid, incomplete Cholesky, DCDM and
  FluidNet; and notably frugal — 1.5 GiB for 128³ grids against FluidNet's
  5.2 GiB and DCDM's 8.5 GiB.
- **Neural Preconditioning Operator (NPO, 2025)** — algebraic-multigrid
  principles in a transformer architecture, for uniform and irregular
  meshes. The transformer is the wrong shape for a 64×64 browser budget,
  but it is where the accuracy frontier is.
- **GNN-accelerated algebraic multigrid (2026)** — a graph network
  accelerating an AMG pressure solver; same family, mesh-general.

**The caveat to check first.** Conjugate gradients requires a symmetric
positive-definite preconditioner. An arbitrary CNN is neither. The
literature handles this by constraining the architecture, by symmetrising,
or by moving to a flexible Krylov method (FGMRES) that tolerates a varying
preconditioner. This package's CG already carries load-bearing guards
(degenerate denominator, magnitude, sign flip) and a sticky stop code —
a non-SPD preconditioner would trip them, which is the good failure mode,
but it means the architecture choice is not free.

#### A3. Learned search directions

**DCDM** — [A Deep Conjugate Direction Method for Iteratively Solving
Linear Systems](https://arxiv.org/abs/2205.10763) (Kaneda, Akar, Chen,
Trevino Kala, Hyde & Teran, ICML 2023). Keeps CG's structure but uses a
CNN to choose search directions, approximating the action of A⁻¹ up to an
arbitrary constant. Trained **unsupervised**, with a loss equal to the L²
difference between an input and the system matrix times the network
evaluation — i.e. ‖b − A·net(b)‖, which needs **no ground-truth
solutions at all**, only the operator. Reduces the residual to a given
tolerance in a small number of iterations, *independent of problem size*.

The unsupervised loss is the important detail for this repo. It means the
training data is the operator this package already has, evaluated on
fields this package already produces. No reference solver, no dataset, no
pipeline.

#### A4. Learned initial guess / warm start

The cheapest and safest of all. A network predicts the starting `x₀` for
the existing solver; everything downstream is unchanged. A bad prediction
costs iterations and nothing else. Recent framing: **NOWS — Neural
Operator Warm Starts for Accelerating Iterative Solvers** (2025).

For this package the payoff is directly in the measured currency: fewer CG
iterations means fewer V-cycles, fewer dispatches, fewer submissions, and
fewer per-iteration readbacks. And there is an obvious
non-ML baseline to beat first, which is warm-starting from the previous
frame's pressure; if that is not already done, it should be measured
before any network is trained.

#### A5. Learned components inside the existing multigrid

Smallest blast radius in the whole survey: keep the V-cycle, learn the
constants inside it.

- [Learning to Optimize Multigrid PDE
  Solvers](https://arxiv.org/abs/1902.10248) (Greenfeld et al., ICML 2019)
  — learns a mapping from a family of parameterised PDEs to prolongation
  operators, trained once for the class with an unsupervised loss.
- [Deep Multigrid](https://arxiv.org/pdf/1711.03825) — learning
  prolongation and restriction matrices.
- [Learning optimal multigrid smoothers via neural
  networks](https://arxiv.org/abs/2102.12071) (2021) — CNN smoothers
  trained on small problems with a loss derived from multigrid convergence
  theory, then applied to large problems of the same class.
- [Learning Relaxation for Multigrid](https://arxiv.org/pdf/2207.11255)
  (2022).
- [Learning Neural PDE Solvers with Convergence
  Guarantees](https://arxiv.org/html/1906.01200) — learns a correction to
  Jacobi that provably preserves the fixed point: if it converges, it
  converges to the correct solution.

That last one is the conceptually cleanest fit for this package's problem,
because it is the *guarantee* that the free-surface measurement says is
non-negotiable.

**But price it against the measurement first.** This repo has already
established that the V-cycle is **0.155 ms of a ~20 ms step**. Making the
V-cycle better cannot win more than 0.155 ms unless it reduces the
*iteration count*, and iteration count is where the readbacks and
submissions live. Any A5 proposal has to be argued in iterations, not in
V-cycle quality.

---

### B. Learned correction to a coarse solver ("solver in the loop")

The highest-quality-per-cost family in the field, and the one with the
best stability record.

- **Kochkov, Smith, Alieva, Wang, Brenner & Hoyer, PNAS 2021** —
  [Machine learning–accelerated computational fluid
  dynamics](https://www.pnas.org/doi/pdf/10.1073/pnas.2101784118). The
  network does not replace the solver; it improves approximations *inside*
  it, effectively super-resolving the missing subgrid detail. Result:
  **8–10× coarser in each dimension at the same accuracy, 40–80× speedup**,
  stable over long simulations, and generalising to different flow
  conditions — explicitly contrasted with black-box ML approaches that do
  neither.
- **Um, Brand, Fei, Holl & Thuerey, NeurIPS 2020** —
  [Solver-in-the-Loop](https://github.com/tum-pbs/Solver-in-the-Loop).
  The methodological result underneath: training the network *with the
  differentiable solver in the loop*, so it sees its own accumulated error,
  significantly outperforms one-step supervised training, and yields
  "stable rollouts of several hundred recurrent evaluation steps". This is
  the standard answer to the recurring failure mode of every learned
  simulator — one-step-accurate, rollout-divergent.

**Verdict for this package: right idea, wrong bottleneck, for now.** B buys
*resolution*. This package is not resolution-bound at 64×64 — it is
dispatch-bound, with the GPU busy for only a few per cent of a step. B
becomes the most
interesting direction in the survey the moment the dispatch problem is
solved and the grid can grow; before that, an 8× coarsening of a grid that
is already tiny buys nothing. It also requires a differentiable solver,
which this package does not have (see G).

---

### C. Super-resolution and detail synthesis — quality, not efficiency

Operates on the *output* field, after the solve. It preserves no
invariant, so it cannot break the solver: the worst case is that it looks
wrong.

- **Chu & Thuerey, SIGGRAPH 2017** — data-driven synthesis of smoke flows
  with CNN-based feature descriptors; patch-based detail lookup.
- **tempoGAN** (Xie, Franz, Chu & Thuerey, SIGGRAPH 2018) —
  [a temporally coherent, volumetric GAN for super-resolution fluid
  flow](https://arxiv.org/pdf/1801.09710). The contribution is the
  *temporal* discriminator: without it, per-frame super-resolution
  flickers. Reported small-scale features "at least as detailed as the
  ground truth reference".
- **[Dynamic Upsampling of Smoke through Dictionary-based
  Learning](https://arxiv.org/pdf/1910.09166)** (2019) — a cheaper,
  non-GAN alternative.
- **Diffusion-based super-resolution** is where this moved, 2024 onward:
  [CoNFiLD](https://www.nature.com/articles/s41467-024-54712-1) (Nature
  Communications 2024, conditional neural field latent diffusion for
  spatiotemporal turbulence), and spectrum-decomposed diffusion SR that
  generates the high-wavenumber content a coarse solve is missing. The
  stated motivation is exactly tempoGAN's weakness — GANs trained on
  isolated snapshots lack temporal coherence.

**Shape warning.** A diffusion model is an iterative denoiser: tens of
network evaluations per frame, therefore tens-to-hundreds of dispatches.
That is the wrong shape for this package's budget, and it is worth being
blunt about it — diffusion is the most exciting part of this sub-field and
the least usable here. **A single-pass generator (tempoGAN-style, or a
plain feed-forward CNN) is the right shape:** one forward pass, a handful
of dispatches, zero readbacks.

For this repo this is the safest quality win available, and the examples
it applies to already exist: `17-smoke-fire`, `18-explosion`,
`19-fuel-fire`, `25-dye-injection`. A learned upsampler on the dye/density
field changes nothing about pressure, divergence, or volume.

---

### D. Reduced-order models and latent-space simulation

The largest speedups in the entire field, bought with the largest
restriction.

- **Deep Fluids** (Kim, Azevedo, Thuerey, Kim, Gross & Solenthaler,
  Eurographics 2019) — a generative CNN over a parameterised family of
  simulations. Two properties worth noting: the architecture generates
  **divergence-free velocity by construction** (it predicts a stream
  function and takes its curl, so incompressibility is structural rather
  than learned), and the reported numbers are **up to 700× faster than
  re-simulating** with compression up to **1300×**. A second network
  integrates forward in the latent space.
- **Latent Space Physics** (Wiewel, Becher & Thuerey, 2019) and **Latent
  Space Subdivision** (2020) — LSTM/temporal prediction in a learned
  latent space.

The divergence-free-by-construction trick (curl of a predicted stream
function) is the most portable idea in this section, and it is *not*
confined to ROMs — any network that outputs a 2D velocity field can be
made exactly divergence-free this way, which is a genuinely different
proposition from FluidNet's "low divergence by loss function". Worth
flagging as a design primitive independent of family D.

**Verdict for this package: incompatible with its purpose.** A ROM is fast
because it only knows the trajectories it was trained on. `src/interaction/`
exists so a user's pointer and keyboard can drive arbitrary force
functions; that is precisely the out-of-distribution input a ROM cannot
take. Viable only for a "canned background effect" product, which is not
what this is.

---

### E. Neural surrogates for the whole step

Replace the solver. The most-published family, the least usable one here.

**Eulerian / operator learning.** Fourier Neural Operators, U-Net
surrogates, PDE-Refiner, transformer operators. Since 2024 this has
consolidated into *PDE foundation models*:
[Poseidon](https://arxiv.org/abs/2405.19101) (NeurIPS 2024 — a multiscale
operator transformer with time-conditioned layer norms, pretrained on
fluid-dynamics governing equations, evaluated on 15 downstream tasks, open
sourced), **DPOT** (Fourier attention, up to **0.5B parameters**, 10+ PDE
datasets, 100k+ trajectories), PROSE-FD, MoE-POT, UniFluids.

**Lagrangian / particles.**
- [Lagrangian Fluid Simulation with Continuous
  Convolutions](https://openreview.net/forum?id=B1lDoJSYDH) (Ummenhofer,
  Prantl, Thuerey & Koltun, ICLR 2020). The one with the right
  computational shape: **no explicit graph**, spatial convolutions extended
  to the continuous domain as the differentiable operation relating
  particles to neighbours. Simulates different materials, generalises to
  arbitrary collision geometry, and is reported to beat prior formulations
  on both accuracy *and speed*.
- Graph network simulators (GNS, MeshGraphNets) and their descendants —
  DualFluidNet, FluidFormer (2026), NeuralDEM (real-time industrial
  particulate flows).

**Verdict.** Three disqualifiers, in order: model size (a 0.5B-parameter
model is not shipping in a browser bundle), rollout stability (the
recurring failure across the family, and the reason Solver-in-the-Loop
exists), and boundary generalisation (this package's examples include
`21-flip-irregular-container` and `23-flip-moving-collider`, which is
exactly the axis these models are weakest on).

The exception worth keeping on the list is Ummenhofer's continuous
convolution for the FLIP examples: it is local, fixed-cost per step, has
no global solve, and therefore no readback. It replaces the solver, so it
inherits A1's volume-loss risk — but for a FLIP liquid the 12,000-frame
volume/occupancy run in `docs/long-run-stability.md` is exactly the acceptance criterion that would
settle it.

---

### F. Secondary-detail models

The only family in the survey with **no solver risk at all**, because
nothing in the solver's invariants depends on its output.

- **Um, Hu & Thuerey, CGF 2018** — [Liquid Splash Modeling with Neural
  Networks](https://arxiv.org/abs/1704.04456). A **classifier plus a
  velocity modifier**: the network learns when a FLIP particle should
  become a splash droplet and what velocity to give it, trained from
  physically parameterised high-resolution reference simulations. The
  stated result is that it yields splash detail "much more efficiently
  than finer discretizations" — which is the whole point: it buys the look
  of a finer grid without the grid.
- [Efficient learning representation of noise-reduced foam effects with
  convolutional denoising
  networks](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0275117)
  (PLOS One 2022) — foam particles via screen projection, with a CNN
  removing the resulting noise.

For reference on what this buys in production: Houdini's
[whitewater](https://www.sidefx.com/docs/houdini/fluid/whitewater.html)
system is the hand-authored equivalent, and it is a standard part of every
liquid shot.

This package has FLIP (`examples/20`–`23`) and no whitewater. The network
is small, per-particle, local, and one dispatch.

---

### G. Learned control and inverse problems

Where "quality" means *art direction* rather than *more detail*.

- Differentiable simulation plus gradient descent to hit keyframes:
  minimise the discrepancy between simulated and target density at each
  keyframe with a regulariser on control-force magnitude. Recent:
  [Hierarchical Differentiable Fluid
  Simulation](https://onlinelibrary.wiley.com/doi/10.1111/cgf.70226) (CGF
  2025), which attacks the standing bottleneck — grid-based differentiable
  simulation's memory consumption constrains the optimisation resolution.
- Neural smoke stylization and colour transfer; sketch-driven design
  ([DualSmoke](https://arxiv.org/pdf/2208.10906)).

**Verdict: not near-term here, and worth saying why.** All of it requires a
differentiable solver. This package's solver is hand-written TSL compute
kernels with no autodiff anywhere in `tsl_array_n`. Building adjoints for
advection, projection and the MGPCG is a larger project than any of the
directions above, and it would be a prerequisite for both G *and* the
solver-in-the-loop training in B.

---

### H. Rendering-side ML

Routinely confused with simulation ML, and for a real-time product it is
often the better answer to the same complaint.

Denoising, neural upsampling (DLSS-class temporal super-resolution),
neural volumetric rendering, Gaussian-splatting-based fluid rendering, and
inverse rendering to recover fluid motion from video. None of it touches
the solver. The games industry's actual answer to "make this look
higher-resolution" is usually here, not in the simulation.

A useful reality check on per-frame network cost in a shipping engine:
Unreal Engine 5's Neural Network Engine (NNE) with
[TensorRT for RTX](https://developer.nvidia.com/blog/speed-up-unreal-engine-nne-inference-with-nvidia-tensorrt-for-rtx-runtime/)
completes neural inference in **3.8 ms at 1080p on an RTX 5090**, 1.5× over
DirectML. That is the budget a AAA engine accepts for a full-screen
network. This package's whole solver step is ~15–20 ms on a 64×64 grid —
so a
*small* network is affordable here by a wide margin, and a full-screen one
is not.

---

## Part 2 — what would actually run in this browser

Three ways to get a network executing, and the choice is forced by where
the network sits.

**1. Hand-written convolution kernels in TSL, on `tsl_array_n` arrays.**
The network's weights are just arrays; a 3×3 convolution is a
stencil kernel, which is the exact thing this package already writes
dozens of. Everything stays in the same buffer space, there are no copies,
there is no second runtime in the bundle, and — decisively — **the
dispatch count is under direct control and the dispatches can join the
existing batching machinery.** For anything in the inner loop (families A
and B), this is the only viable option.

**2. ONNX Runtime Web with the WebGPU execution provider.** Real and
mature, and it does support keeping data on the GPU:
`ort.Tensor.fromGpuBuffer()` builds a tensor from an existing WebGPU
storage buffer, IO binding avoids GPU↔CPU copies, output buffers can be
pre-allocated, and the underlying Dawn instance/adapter/device can be
shared across sessions. So a zero-copy handoff is *possible in principle*.
Whether three.js's WebGPU renderer will surrender its device and raw
buffers to ORT cleanly is an open question that needs a spike, not an
assumption. Reasonable for a post-process (families C, F, H); risky for
the inner loop.

**3. TensorFlow.js WebGPU backend / WebNN.** Same trade as 2 with less
control. Mentioned for completeness.

### The shape, now built rather than estimated

`src/ml/` implements this, and these numbers come from the built network
(`createUNet2({ shape: [64,64], channels: 16, levels: 3 })`), not from
arithmetic on the back of an envelope:

| | |
| --- | --- |
| dispatches per forward pass | **12** — 8 convolutions, 2 restrictions, 2 upsamples |
| submissions per forward pass | **1**, through `tsl_array_n.createBatch` |
| host round trips | **0** |
| parameters | 14,225 = **55.6 KiB** float32 |
| arithmetic | 25.95 MMAC = **51.9 MFLOP** per pass |

The first draft of this section estimated "≈ 0.2 GFLOP per solve" from a
channel-doubling U-Net. The built network is **four times cheaper** than
that, because its channel width is constant across levels — a choice made
to avoid the projection convolutions that channel-doubling forces at every
level change, which would have cost four more dispatches. The dispatch
estimate ("~10–20") held.

Against that: ~1416 dispatches per solver step, a data-dependent iteration
count, and the four-plus host round trips at 1.1 ms each that make up the
untargeted 40%.

**What is still entirely unknown is the wall-clock time.** 51.9 MFLOP is
nothing in isolation, but at 64×64 the tensors are small enough that a
naive TSL convolution may be latency-bound rather than arithmetic-bound,
and none of the counts above say anything about that. This repo's own
history is a list of confidently predicted wins that measured flat or
negative (the GPU-resident alpha/beta rounds; the `mg-clear` fold; damped
Jacobi), so a parameter count is not evidence about speed.

The probe that settles it is built and unrun — see shortlist item 1.

---

## Part 3 — the shortlist for fluxflow, ranked

Ordered by evidence-per-unit-risk, not by how interesting the paper is.

### 1. The null-net probe — measurement, not machine learning

**Built. `src/ml/` and `examples/31-null-net-probe/`. Not yet run — it
needs real WebGPU hardware.**

The candidate shape exists now as actual TSL kernels: convolution,
full-weighting restriction, bilinear upsample, and a U-Net composing them,
with **random weights**. It computes nonsense, which is the point — what is
being priced is the shape, and a shape can be priced before any training
run exists.

What is already exact, computed from the built network rather than
estimated:

| | `createUNet2({ shape: [64,64], channels: 16, levels: 3 })` |
| --- | --- |
| dispatches per forward pass | **12** |
| submissions per forward pass | **1** (batched through `tsl_array_n.createBatch`) |
| host round trips | **0** |
| parameters | 14,225 = **55.6 KiB** float32 |
| arithmetic | 25.95 MMAC = **51.9 MFLOP** per pass |

That last row retires the estimate this document carried earlier. The
first draft guessed "~0.2 GFLOP"; the built network is 51.9 MFLOP, four
times less, because the channel width is constant across levels rather
than doubling. The dispatch estimate ("~10–20") held.

The probe page reports three things, in order of how much they decide:
**correctness** (the GPU pass against a float64 JavaScript reference on the
same weights — without which the timing is the timing of an unknown
computation), **cost** (dispatches, submissions, encode time and GPU time
per pass, batched and unbatched), and **the comparison** (the same
measurements for one MGPCG solve on the same grid, run with
`grid_pressure_solver2.js`'s shipped defaults rather than `solve()`'s own
slower parameter defaults — the perf document records example 15 springing
exactly that trap).

Nothing about speed is known yet. When the page is run, its numbers replace
the estimates in Part 0 and Part 2, and the acceptance threshold is stated
on the page: **under 2× is a negative result**, because a trained network
has to buy accuracy with its speed and family A gives accuracy up.

### 2. Re-measure the frame, because Part 0's percentage is stale

The GPU-busy share quoted throughout this document (~3–7%) is an estimate
derived from a 2026-09-13 measurement taken before four optimisations
landed. The direction of the staleness is known — host overhead came out,
GPU compute did not, so the share rose — but the magnitude is not.

It is one run of the existing instrumented harness on real hardware:
`renderer.compute()` patched, driver paused, GPU timestamps, reported per
step. It costs an afternoon and it is the denominator for every efficiency
claim below, including item 1's.

Worth folding into the same run: whether the 1.1 ms host round trip and the
0.861 ms per CG iteration still hold on the current defaults, since both
were measured before `seedScalarsKernel`.

### 3. Learned initial guess for the pressure solve (A4)

Strictly safe: CG converges to the same answer regardless, so a bad guess
costs iterations and nothing else. Pays directly in the measured currency —
fewer iterations is fewer V-cycles, fewer submissions, and fewer of the
per-iteration readbacks, which the step accounting put at 52% of a step
before the check interval was widened to 4.

**The bar is higher than it looks, and this is the honest part.** The
obvious non-ML warm start is already in place: `pressureGrid` persists
across frames and CG starts from last frame's solution, with newly-Dirichlet
cells re-seeded to their target first (`grid_pressure_solver2.js`, the
"seed x at newly-Dirichlet cells" block). So a learned initial guess is not
competing against a zero start — it is competing against temporal
coherence, which is free and already very good on a smoothly evolving
field. The case where it could still win is the case where the existing
warm start is *worst*: frames where the fluid configuration changes
abruptly — a newly-wet region, a moving collider, an impact — which is
exactly `examples/23` and `examples/28`. That narrows the claim
considerably, and it should be measured on those scenes rather than on
example 15.

Acceptance: median CG iterations per frame down; max|div| unchanged;
`docs/long-run-stability.md`'s 12,000-frame volume/occupancy run flat.

### 4. Learned preconditioner or learned V-cycle components (A2/A5)

Keeps the convergence guarantee, which the free-surface measurement says
is non-negotiable. DCDM's unsupervised loss ‖b − A·net(b)‖ means the
training signal is the operator this package already has — no dataset, no
reference solver.

Two things to settle before building: whether the architecture preserves
the SPD-ness CG requires (or whether the solver moves to a flexible Krylov
method), and **whether one network covers both the plain and the
variable-density operator** — the damped-Jacobi result says constants do
not transfer between them, and a network is a large constant.

### 5. A splash / whitewater network for the FLIP examples (F)

Pure quality upside, zero solver risk, small local network, one dispatch.
Um et al.'s classifier-plus-velocity-modifier formulation is directly
portable to `grid_flip_solver2.js`'s particles. The nearest thing to free
in this whole document.

### 6. Single-pass super-resolution for the smoke/fire/dye examples (C)

A post-process on the density/dye field, no invariant to preserve. Use a
feed-forward generator, **not** a diffusion model — the iteration count is
the whole objection. tempoGAN's lesson applies regardless of architecture:
without an explicitly temporal term, per-frame upsampling flickers.

### Explicitly not now

- **Full CNN replacement of the pressure solve (A1)** — unbounded residual
  divergence against a measured liquid collapse. Reconsider only for the
  smoke/fire examples, where GPU Gems 38 already established that
  approximate projection is visually acceptable.
- **Latent-space ROM (D)** — incompatible with `src/interaction/`.
- **Foundation-model surrogates (E)** — wrong size by three orders of
  magnitude.
- **Differentiable control (G)** and **solver-in-the-loop training (B)** —
  both blocked behind a differentiable solver that does not exist.

---

## Part 4 — the risks that apply to all of it

- **Operator transfer.** Stated three times above because it is the most
  likely failure: this repo has *already measured* that a tuned constant
  free on one operator costs 1.40× on another and diverges on a third.
  Any learned component must be tested on the variable-density operator
  the liquid scenes use, not just the plain Laplacian.
- **float32.** `examples/28-drop-into-pool`'s convergence investigation
  found float32 running out of digits rather than iterations. Inference is
  float32 too, so this is neutral in itself — but a learned
  preconditioner's benefit can be masked entirely by the same precision
  floor, and a measurement that does not separate the two will be
  misattributed.
- **Generalisation to unseen boundaries.** The recurring caveat in every
  paper that reports it honestly. `21-flip-irregular-container` and
  `23-flip-moving-collider` are the tests.
- **Bundle size and cold start.** Weights ship to the browser. A network
  large enough to be good may be too large to download.
- **Test determinism.** The existing test suite compares numbers. A
  network in the loop makes every downstream expectation weight-dependent,
  and the weights become part of the fixture.
- **Provenance.** `docs/provenance-audit.md` exists because this repo
  tracks where code came from. Pretrained weights, training data and
  reference implementations carry licences too, and a model trained on
  another project's simulation output inherits that project's terms.

---

## Sources

Surveys
- Wang et al., [Physics-based fluid simulation in computer graphics:
  Survey, research trends, and
  challenges](https://link.springer.com/article/10.1007/s41095-023-0368-y),
  Computational Visual Media 2024, 10(5): 803–858
- [Recent Advances on Machine Learning for Computational Fluid Dynamics: A
  Survey](https://arxiv.org/pdf/2408.12171)

Pressure solve
- Tompson, Schlachter, Sprechmann & Perlin, [Accelerating Eulerian Fluid
  Simulation With Convolutional
  Networks](http://proceedings.mlr.press/v70/tompson17a/tompson17a.pdf),
  ICML 2017 · [project page](https://google.github.io/FluidNet/) ·
  [code](https://github.com/google/FluidNet)
- Lan et al., [A Neural-Preconditioned Poisson Solver for Mixed Dirichlet
  and Neumann Boundary
  Conditions](https://proceedings.mlr.press/v235/lan24a.html), ICML 2024
- Kaneda, Akar, Chen, Trevino Kala, Hyde & Teran, [A Deep Conjugate
  Direction Method for Iteratively Solving Linear
  Systems](https://arxiv.org/abs/2205.10763), ICML 2023 ·
  [code](https://github.com/ayano721/2023_DCDM)
- [Neural Preconditioning Operator for Efficient PDE
  Solves](https://arxiv.org/html/2502.01337v2) (2025)
- [NOWS: Neural Operator Warm Starts for Accelerating Iterative
  Solvers](https://arxiv.org/pdf/2511.02481)
- [Acceleration of an algebraic multigrid pressure solver using graph
  neural networks](https://arxiv.org/html/2606.19251v1)
- [A machine learning based solver for pressure Poisson
  equations](https://www.sciencedirect.com/science/article/pii/S2095034922000423)

Multigrid components
- Greenfeld et al., [Learning to Optimize Multigrid PDE
  Solvers](https://arxiv.org/abs/1902.10248), ICML 2019
- [Deep Multigrid: learning prolongation and restriction
  matrices](https://arxiv.org/pdf/1711.03825)
- [Learning optimal multigrid smoothers via neural
  networks](https://arxiv.org/abs/2102.12071)
- [Learning Relaxation for Multigrid](https://arxiv.org/pdf/2207.11255)
- [Learning Neural PDE Solvers with Convergence
  Guarantees](https://arxiv.org/html/1906.01200)

Learned correction to a coarse solver
- Kochkov, Smith, Alieva, Wang, Brenner & Hoyer, [Machine
  learning–accelerated computational fluid
  dynamics](https://www.pnas.org/doi/pdf/10.1073/pnas.2101784118), PNAS
  2021
- Um, Brand, Fei, Holl & Thuerey, [Solver-in-the-Loop: Learning from
  Differentiable Physics to Interact with Iterative
  PDE-Solvers](https://papers.nips.cc/paper/2020/file/43e4e6a6f341e00671e123714de019a8-Paper.pdf),
  NeurIPS 2020 · [code](https://github.com/tum-pbs/Solver-in-the-Loop)

Super-resolution and detail
- Xie, Franz, Chu & Thuerey, [tempoGAN: A Temporally Coherent, Volumetric
  GAN for Super-resolution Fluid Flow](https://arxiv.org/pdf/1801.09710),
  SIGGRAPH 2018
- [Dynamic Upsampling of Smoke through Dictionary-based
  Learning](https://arxiv.org/pdf/1910.09166)
- [Conditional neural field latent diffusion model for generating
  spatiotemporal
  turbulence](https://www.nature.com/articles/s41467-024-54712-1), Nature
  Communications 2024
- [Neural Differentiable Modeling with Diffusion-Based Super-resolution for
  Two-Dimensional Spatiotemporal
  Turbulence](https://arxiv.org/pdf/2406.20047)

Reduced-order and latent space
- Kim, Azevedo, Thuerey, Kim, Gross & Solenthaler, [Deep Fluids: A
  Generative Network for Parameterized Fluid
  Simulations](https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.13619),
  Eurographics 2019

Surrogates and operators
- Ummenhofer, Prantl, Thuerey & Koltun, [Lagrangian Fluid Simulation with
  Continuous Convolutions](https://openreview.net/forum?id=B1lDoJSYDH),
  ICLR 2020
- [Poseidon: Efficient Foundation Models for
  PDEs](https://arxiv.org/abs/2405.19101), NeurIPS 2024
- [DualFluidNet: an Attention-based Dual-pipeline Network for Fluid
  Simulation](https://arxiv.org/pdf/2312.16867)
- [FluidFormer: Transformer with Continuous Convolution for Particle-based
  Fluid Simulation](https://arxiv.org/pdf/2508.01537)
- [NeuralDEM — Real-time Simulation of Industrial Particulate
  Flows](https://arxiv.org/pdf/2411.09678)
- Deng et al., [Fluid Simulation on Neural Flow
  Maps](https://dl.acm.org/doi/10.1145/3618392), SIGGRAPH Asia 2023

Secondary detail
- Um, Hu & Thuerey, [Liquid Splash Modeling with Neural
  Networks](https://arxiv.org/abs/1704.04456), CGF 2018
- [Efficient learning representation of noise-reduced foam effects with
  convolutional denoising
  networks](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0275117),
  PLOS One 2022
- [Houdini whitewater](https://www.sidefx.com/docs/houdini/fluid/whitewater.html)

Control
- [Hierarchical Differentiable Fluid
  Simulation](https://onlinelibrary.wiley.com/doi/10.1111/cgf.70226), CGF
  2025
- [Efficient Solver for Spacetime Control of
  Smoke](https://doi.org/10.1145/3072959.3016963)
- [DualSmoke: Sketch-Based Smoke Illustration
  Design](https://arxiv.org/pdf/2208.10906)

Browser inference
- [ONNX Runtime Web — WebGPU execution
  provider](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)
- [ONNX Runtime Web unleashes generative AI in the browser using
  WebGPU](https://opensource.microsoft.com/blog/2024/02/29/onnx-runtime-web-unleashes-generative-ai-in-the-browser-using-webgpu/)
- [Speed Up Unreal Engine NNE Inference with NVIDIA TensorRT for
  RTX](https://developer.nvidia.com/blog/speed-up-unreal-engine-nne-inference-with-nvidia-tensorrt-for-rtx-runtime/)

Internal, for every measurement quoted:
`docs/perf-investigation-cg-gpu-resident-alpha-beta.md`,
`docs/project-history.md`, `docs/long-run-stability.md`,
`docs/realtime-fluid-tools-research.md`, `src/profiling.js`.
