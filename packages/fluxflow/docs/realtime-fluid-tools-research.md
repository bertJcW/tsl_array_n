# How the real-time fluid tools do it: TouchDesigner and LiquiGen

The question this answers: **TouchDesigner and JangaFX's LiquiGen both solve
fluid in real time. Is that a code-level difference, or is it because they
get to use a native GPU API and this package does not?**

The answer is code-level, and this package already holds the measurement
that settles it. What follows is the evidence, what each tool is actually
documented to do, where WebGPU genuinely *is* behind native, and why the
cheapest-looking fix is not available to a liquid solver.

---

## The short answer

**Native API access is worth at most ~2.5% here, and that number is
measured, not estimated.**

`src/profiling.js` reports CPU-side command encoding at **2.28 ms of an
89.8 ms frame** on `examples/15-flow-past-cylinder/`. Encoding is the
entire cost of "talking to the GPU through this API rather than that one":
building command buffers, validating, submitting. A native API can make
that cheaper. It cannot make it smaller than zero, so the whole
TSL-versus-native question is bounded by 2.5% of the frame.

What the other 97.5% is spent on is a different question with a different
answer, and it is the one that separates these tools from this package:

| | dispatches per frame | GPU-to-CPU round trips per frame |
| --- | --- | --- |
| GPU Gems 38 / TouchDesigner-style Jacobi | **~45** | **0** |
| fluxflow MGPCG (measured) | **502** | 12-14 (was 36-42) |

> **Superseded in part (2026-09-13).** The dispatch count below is an
> undercount: `profiling.js` only wrapped the kernels built through
> `buildElementwiseKernel`, and roughly 15% of a frame's dispatches are
> built with `tsl_array_n.kernel` directly. Counting `renderer.compute()`
> itself gives ~800 dispatches *and* ~272 submissions per frame, and the
> closing section of this document has the measurements. The encoding
> figure stands.

Same hardware, same physics, same API would not close that. It is an
algorithm difference.

---

## TouchDesigner

Well documented, and the documentation is specific.

Derivative's own fluid-simulation material states the implementation is
["Fast Fluid Dynamics Simulation based on Jos Stam's paper and NVIDIA's GPU
Gems Chapter 38"](https://derivative.ca/community-post/tutorial/fluid-simulation/63745),
written in GLSL. Community solvers on the forum describe the same lineage,
[moved from CUDA to GLSL](https://forum.derivative.ca/t/2d-fluid-simulation-on-the-gpu/8307)
so the work happens in shaders.

So the algorithm is not proprietary -- it is
[GPU Gems Chapter 38](https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu),
and that chapter is explicit about the two things that matter here:

- The pressure Poisson equation is solved by **Jacobi iteration**, chosen
  in the chapter's own words "because of its simplicity and easy
  implementation".
- The iteration count is **fixed**: "typically 40 to 80", with a warning
  not to go below 20. **There is no convergence check.** Nothing in the
  loop reads a residual, because nothing in the loop needs a number on the
  host.

The consequence is the important part. A Jacobi sweep is one dispatch that
reads a texture and writes a texture. Forty of them plus advection,
divergence and gradient subtraction is **roughly 45 dispatches and zero
readbacks per frame**. The host issues the same command list every frame
and never waits for an answer.

TouchDesigner also ships **NVIDIA Flex** (particle-based) and **NVIDIA
Flow** (sparse grid) as integrations. Those are vendor libraries, not
TouchDesigner's own solver, and they are the answer to "how does
TouchDesigner do 3D smoke and particle liquid" rather than "how does its
fluid TOP work".

---

## LiquiGen (JangaFX)

Directly comparable to this package, and much less documented.

What is public:

- LiquiGen uses a **PIC/FLIP solver** with whitewater (spray, foam,
  bubbles), real-time meshing, and a real-time path tracer
  ([JangaFX](https://jangafx.com/software/liquigen),
  [CG Channel](https://www.cgchannel.com/2025/07/jangafx-releases-liquigen-1-0/)).
  That is the same solver family as `grid_flip_solver2.js`.
- EmberGen 2.0's roadmap describes a **sparse simulation system** --
  simulate only where there are active voxels, inside a virtual
  8192³ domain, with roughly 200 million active voxels on a 6 GB GPU and
  ~2 billion on 48 GB ([roadmap](https://jangafx.com/roadmap)).
- The company's stated origin is "why can't the GPU handle this?", with
  simulation and rendering merged into
  [a single real-time process](https://www.dell.com/en-us/blog/reimagining-vfx-creation-jangafx-s-path-to-real-time-simulation/)
  rather than the sequential CPU pipeline of older tools.

**What is not public: their pressure solver.** No iteration count, no
solver family, no statement about host synchronisation, and no GPU API is
named in any of the material surveyed -- the roadmap, the product pages,
the CEO interview, or the release coverage. Anything specific about how
LiquiGen's projection works would be a guess, and is not offered here.

Two things are worth stating without guessing:

1. **They are not solving this package's problem on this package's
   hardware budget.** These are desktop tools targeting RTX 4090 / RTX 6000
   class GPUs, with a bespoke engine and years of work behind it. A browser
   package on arbitrary hardware is a different target.
2. **Sparsity is an algorithmic win, not an API win.** "Simulate only where
   there is fluid" is available in WebGPU too. It is on the same axis as
   everything else below: do less work, not the same work through a faster
   door.

---

## Where WebGPU genuinely is behind native

The honest other half. These are real gaps, and none of them is where this
package's time goes:

- **Subgroup / wave intrinsics** arrived only recently (Chrome 125), and
  native APIs have had them for years
  ([Unity's WebGPU limitations](https://docs.unity3d.com/6000.2/Documentation/Manual/WebGPU-limitations.html)).
  This is the one that is *specifically* relevant here: a dot product is a
  reduction, and reductions are exactly what subgroup operations
  accelerate. `createDotReducer` currently partitions by lane and sums
  partials on the host.
- **No async compute**, so independent work cannot overlap the way it can
  on a native queue.
- **No CUDA-ecosystem equivalents** -- there is no cuDNN/CUTLASS-grade
  tuned kernel library behind wgpu.

If the frame were compute-bound on a large grid, these would matter. It is
not: the grid is 64×64, which is 4096 cells, and 4096 cells is microseconds
of arithmetic. A 40 ms frame on 4096 cells is not a hardware limit, an API
limit, or a shader-quality limit. **It is 502 dispatches and a dozen
pipeline drains.**

---

## The actual mechanism, and the trade underneath it

MGPCG converges in far fewer iterations than Jacobi -- that is what it is
for. The standard framing is that multigrid preconditioning "greatly
reduces the number of iterations required to converge, at the expense of
increasing the cost of each iteration". Measured here, that expense is
concrete:

- One V-cycle is **~36 dispatches** (4 levels, 2 down / 2 up smoothing).
- Plus ~7 CG vector operations per iteration.
- Times 12-14 iterations = **~500 dispatches**.

Jacobi's cost per iteration is **one** dispatch. So:

|  | iterations | dispatches / iteration | total |
| --- | --- | --- | --- |
| Jacobi (GPU Gems 38) | 40-80 | 1 | 40-80 |
| MGPCG (fluxflow) | 12-14 | ~43 | ~500 |

MGPCG needs a fifth of the iterations and issues **ten times the
dispatches**. On a grid big enough for each dispatch to be real work, that
trade is clearly correct. On a 64×64 grid where a dispatch is almost pure
overhead, it inverts.

And CG carries a second, structural cost Jacobi does not have. **CG is
defined by global inner products** -- alpha and beta are scalars derived
from `p·Ap` and `r·z`, and the algorithm cannot take its next step without
them. That is why this package had three GPU-to-CPU round trips per
iteration and, after moving the arithmetic onto the GPU, still has one.
Jacobi has none, ever, because there is no global scalar anywhere in it.

**Real-time tools buy zero synchronisation by giving up the convergence
guarantee.** A fixed 40-iteration Jacobi solve does not produce a
divergence-free field. It produces one that looks right.

---

## Why this package cannot simply copy it

This is the part that stops the obvious conclusion from being the right
one, and it is measured rather than argued.

GPU Gems 38 is **smoke in a closed box**. Smoke tolerates residual
divergence: the error shows up as slightly wrong swirl, which nobody can
identify as wrong. A **free-surface liquid does not tolerate it**. Residual
divergence is a per-frame compressibility error, it accumulates as volume
change, and this package has the failure logged in detail:

> At `atomicScale: 256`, only about half of frames converged, and
> non-converged frames left max|div| of 0.1-0.4 over the fluid cells
> against 0.01-0.03 on converged frames. Occupied cells collapsed from 1536
> to 340-700. With the reduction fixed: 100% converged, max|div| < 0.005,
> occupied cells flat at 1536 for 2000 frames.

-- `docs/project-history.md`, Debugging #5

So "stop checking convergence" is not a free choice here. It is the exact
condition under which this solver's liquid has already been measured to
collapse. The same conclusion arrived independently from the performance
side: the predictive stop-test schedule was reverted because the extra
iterations it caused cost back everything the skipped round trips saved.

LiquiGen is a FLIP liquid and is real-time, so the combination is clearly
achievable. How they achieve it is not published.

---

## What this suggests for fluxflow

The lever is **dispatches per frame and synchronisations per frame**, not
the graphics API. Three directions follow, in order of how well the
existing measurements support them:

1. **Make the V-cycle cheaper in dispatches rather than making the solver
   stronger.** 36 dispatches to smooth 4096 cells across 4 levels is the
   anomaly. The coarse levels are tiny -- the single-workgroup coarse
   sweep already collapsed the coarsest level from ~20 dispatches to 1 for
   1.10x, and levels 1 and 2 are candidates for the same treatment.
2. **Jacobi or damped-Jacobi as the smoother inside the V-cycle**, which is
   one dispatch per sweep instead of two (red-black needs one per colour).
   Costs smoothing quality per sweep; the question is whether two Jacobi
   sweeps beat one red-black sweep at the same dispatch count. Measurable
   with the existing paired harness.
3. **A fixed-iteration outer loop with a periodic, not per-iteration,
   convergence check** -- explicitly *not* the reverted predictive
   schedule, which skipped checks while keeping the iteration count
   adaptive. This is the opposite: fix the iteration count so the frame
   cost is constant, and check convergence rarely enough to be free, with
   the circuit breaker catching what slips through. The risk is exactly the
   volume loss quoted above, so it would need a long-run volume test as
   its acceptance criterion, not a frame-time number.

None of these requires leaving WebGPU. The 2.28 ms encoding measurement
says that leaving WebGPU is worth 2.5% at the absolute ceiling, and all
three of the above are worth considerably more than that.

---

## Sources

- [Derivative -- Fluid Simulation tutorial](https://derivative.ca/community-post/tutorial/fluid-simulation/63745)
- [Derivative forum -- 2D fluid simulation on the GPU](https://forum.derivative.ca/t/2d-fluid-simulation-on-the-gpu/8307)
- [GPU Gems 3, Chapter 38 -- Fast Fluid Dynamics Simulation on the GPU](https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu)
- [JangaFX -- LiquiGen](https://jangafx.com/software/liquigen)
- [JangaFX -- roadmap](https://jangafx.com/roadmap)
- [CG Channel -- JangaFX releases LiquiGen 1.0](https://www.cgchannel.com/2025/07/jangafx-releases-liquigen-1-0/)
- [Dell -- Reimagining VFX Creation: JangaFX's Path to Real-Time Simulation](https://www.dell.com/en-us/blog/reimagining-vfx-creation-jangafx-s-path-to-real-time-simulation/)
- [Unity -- Limitations of the WebGPU graphics API](https://docs.unity3d.com/6000.2/Documentation/Manual/WebGPU-limitations.html)
- [Interactive & Immersive HQ -- NVIDIA Flex in TouchDesigner](https://interactiveimmersive.io/blog/touchdesigner-3d/quick-start-guide-nvidia-flex-in-touchdesigner/)

Internal, for the measurements quoted:
`docs/perf-investigation-cg-gpu-resident-alpha-beta.md`,
`docs/project-history.md`, `src/profiling.js`.
