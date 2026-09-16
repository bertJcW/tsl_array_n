# Where the remaining time goes, what other WebGPU projects do about it, and what to try next

Written 2026-09-16, after the two batching passes recorded in
`perf-investigation-cg-gpu-resident-alpha-beta.md`. Every number here was
measured on this machine, in the browser, against raw WebGPU as a control.
The point of the document is to stop the next optimisation from being chosen
by intuition.

## The current bill for one solver step (example 15, 64x64, ~17 CG iterations)

| | ms/step | note |
| --- | --- | --- |
| **GPU compute** | **0.10** | 0.54% of the step |
| three.js host-side dispatch machinery | ~6 | 67 `compute()` calls, 999 dispatches |
| the same work through raw WebGPU | ~1.1 | 0.56 us/dispatch + 4.44 us/submission |
| serial `mapAsync` waits | ~11 | 5-6 of them, ~2.3 ms each |
| everything else (JS in fluxflow, readback marshalling) | ~1 | |
| **total** | **~18.4** | was 24.5 before today's batching |

Two costs are left, and they are not the same kind of thing.

1. **Host bookkeeping above the WebGPU API.** ~5 ms/step of pure overhead:
   three.js re-resolves node state, bindings and pipelines on every dispatch
   (~3.5 us/dispatch) and pays ~33 us per `compute()` call, against 0.56 us
   and 4.44 us for the WebGPU calls those resolve into.
2. **Serial latency.** A `mapAsync` readback costs ~2.3-3.4 ms whether it
   reads 16 bytes or 64 KB, and the CG loop takes 5-6 of them one after
   another because each answers "should I stop?" before the next batch can
   be encoded.

## What the ecosystem does about each

**Nobody fixes the readback; everyone avoids it.** The standard answer in
real-time work is to never ask the GPU a question the host has to wait for:
fixed iteration counts with no convergence test at all (what TouchDesigner
and LiquiGen do -- see `realtime-fluid-tools-research.md`), or a readback
that is deliberately one or more frames late, so the wait overlaps the next
frame's work. The measurement that makes this the right framing here is that
eight concurrent maps cost what one costs (3.06 ms vs 3.45 ms): the cost is
*per wait*, not per read, so it is only ever paid for a dependency the host
inserted.

**The host-overhead answer is "record once, replay".**

- **ONNX Runtime Web** exposes `enableGraphCapture`, which records the
  WebGPU command sequence on the first run and replays it on later ones,
  explicitly to remove per-run CPU overhead. It requires static shapes --
  exactly the condition a fixed-size fluid solver satisfies.
- **Babylon.js** has *snapshot rendering*: record one frame's draw calls,
  replay the recording afterwards. Its documentation is careful to say the
  win is **JavaScript-side only, GPU time is unchanged** -- the same shape
  as our bill.
- **Render bundles** are the browser-level version of this, and they
  re-bind the same resources rather than snapshotting their contents, so
  dynamic data still flows.

The catch for us: **WebGPU has no compute bundle.** `GPURenderBundle` covers
render passes only; the gap is a known, long-standing request
(gpuweb#4138, gpuweb#1971), and command buffers are single-submission by
specification. So in a browser the recordable part is not the command buffer
-- it is everything *above* it, which is precisely where our ~5 ms sits.

## STATUS, added after the fact (2026-09-16)

**All four proposals below were built and measured the same day this was
written.** The document is kept in its original "what to try next" form,
because a plan that is rewritten after the results is no longer evidence
about how good the reasoning was. What each one actually did:

| | proposal | predicted | measured |
| --- | --- | --- | --- |
| A | prepared-dispatch fast path | up to 1.4x | **1.22x** (ex 15), 1.08x (ex 20), nothing on ex 28 |
| B | fold the circuit breaker's read | ~0.8 ms, ~4% | **1.10x** (ex 28), ~1.045x (ex 15) |
| C | optimistic continuation of the stop test | "hides ~1 ms per wait" | **1.37-1.81x** (ex 15) — much better than predicted, because the waits pipeline into each other |
| D | GPU-side stop test, unlocked by A | qualitative | **1.24-1.54x** (ex 20) on top of C |

Two things the plan got wrong, both recorded in
`perf-investigation-cg-gpu-resident-alpha-beta.md`:

- **C was underestimated.** The plan modelled the saving as one batch's
  encode time. It is larger: consecutive waits overlap each other, so each
  block settles at roughly a third of the full latency.
- **D's precondition was not the one named here.** The plan said D needed
  A, to make wasted iterations cheap. What it actually needed was for the
  *iterate to freeze* at convergence, and for the residual being tested to
  be the true `b - Ax` every iteration -- and that second requirement, once
  measured, turned out to be 1.15-1.43x *faster* than the recompute
  interval of 50 it replaced, not a cost at all.

## Proposals, in the order their measured value suggests

### A. A prepared-dispatch fast path in tsl_array_n (biggest, hardest)

Cache, per kernel, what three.js re-derives every call: the
`GPUComputePipeline`, the bind groups, and the workgroup counts. Encode a
whole batch with a thin loop of `setPipeline` / `setBindGroup` /
`dispatchWorkgroups` into one encoder and submit once.

- **Ceiling:** ~6 ms -> ~1.1 ms per step on example 15, i.e. up to **1.4x**
  on top of today's batching -- and more on scenes with more dispatches.
- **The hard part is uniforms.** Skipping `bindings.updateForCompute` skips
  the uniform upload. A kernel with no per-frame uniform can take the fast
  path unconditionally; one with (say) `dt` needs its uniform written
  directly with `queue.writeBuffer`, or needs three's update called for that
  binding only. The first cut should take the fast path only for kernels
  that declare no dynamic uniform, and fall back otherwise.
- **Invalidation:** any rebuild (a new collider, a resize, a device loss)
  must drop the cache. `createBatch` plans already have this shape, and
  `grid_blocked_boundary_condition_solver2.js`'s `plansGeneration` is the
  precedent.
- **How to test it:** a boolean on `tsl_array_n` so both paths exist in one
  build; paired round-robin A/B within one run (the harness used today);
  and the hash check -- 150 steps from a fresh load, u/v/pressure hashed,
  must match the slow path exactly.

### B. Fold the circuit breaker's read into the read the solver already does

The circuit breaker costs one whole round trip of its own -- `phases` prices
`pressure-badcells-read` at 0.80 ms/step -- and it reads a single integer.
The CG loop is already reading a small scalar buffer every fourth iteration,
so the count could be written into a spare slot of that same buffer and come
back with a read that is happening anyway.

What it needs: the bad-cell reduction dispatched as part of the last
submission before a scalars read, and a way for the pressure solver to hand
`cg.solve` that dispatch plus the slot to find the answer in. That is real
coupling between `linalg.js` and a caller's guard, so it should be an
explicit, optional parameter rather than something the solver knows about.

Note what does **not** work, since the obvious versions were checked against
the code: the post-loop `solve-final-read` only runs when the loop exits on
an iteration that skipped its read, so there is usually no second read to
merge with; and the corrections cannot be encoded optimistically ahead of
the answer, because projection is in place (`project( grid, grid )`) and
re-running a correction would apply the gradient twice.

- **Expected:** ~0.8 ms/step, about 4%. **Risk:** low for correctness (the
  same value is computed and checked), moderate for API tidiness.
- **Test:** paired A/B, and `diagnostics.rejected` must fire on exactly the
  same frames -- run example 28 for 300 steps and compare the rejection
  sets, not just the counts.

### C. Optimistic continuation of the stop test

Issue the stop-test readback **without awaiting it**, encode the next batch
of four iterations, and only then await. The extra iterations are harmless
when convergence has already happened -- today's interval sweep showed 4 and
8 both converge 100%, so an overshoot of one batch is inside the safe range
(16 and 32 are not: 47% and 39%).

- **Expected:** hides the ~1 ms of encode+GPU behind each of 4-5 waits.
- **Risk:** medium. The iterate must not be driven past the point where the
  recursive residual stops tracking the true one, which is exactly what the
  interval sweep found at 16.
- **Test:** paired A/B for speed; convergence rate and mean iterations over
  300 steps on examples 15, 26 and 28 must not move.

### D. A stop test that never leaves the GPU (needs A first)

The sticky stop code in `linalg.js` already makes post-convergence
iterations no-ops on the GPU. If the host stopped asking, the loop could run
a fixed budget and read once at the end -- the cost being the *encoding* of
the wasted iterations. At today's ~0.24 ms of encode per iteration that
trade is a wash, which is why it has not been built. After A it is ~0.02 ms,
and it becomes the obvious design. `dispatchWorkgroupsIndirect` with a
GPU-written workgroup count of zero would remove the wasted GPU work too.

- **Sequencing note:** this is why A comes first. D is not independently
  worth building; it is unlocked by A.

### E. Everything the measurements say *not* to do

- **Pooling staging buffers**: a fresh staging buffer per read costs the
  same as a persistent one (2.84 vs 2.70 ms). Not the bottleneck.
- **Reading fewer bytes**: 16 B and 64 KB are within 12%. Not the
  bottleneck.
- **Raising `residualCheckInterval` further**: swept today at 4/8/16/32/64;
  past 8 the solve degrades (47% converged at 16) for no time saved.
- **Chasing GPU-side kernel efficiency**: the GPU is busy 0.5-0.8% of the
  step. A kernel twice as fast saves 0.05 ms.

## How any of these gets accepted

The same bar the batching passes met, because this project has retracted
numbers before that were measured a different way:

1. **Paired within one run.** Both arms interleaved, phase alternated, on
   the same scene state -- a scene that drifts makes across-run comparison
   meaningless.
2. **An arm that is the old behaviour, not an old build.** Today's A/B
   re-split the merged passes at the `renderer.compute` boundary, so both
   arms ran identical kernels in identical order.
3. **Bit identity, not plausibility.** 150 steps from a fresh load, hash
   u, v and pressure, and require equality. Anything that legitimately
   changes the arithmetic (C and D do -- they change *when* the loop stops)
   instead has to show convergence rate, mean iterations and rejection
   count unchanged over 300+ steps on examples 15, 26 and 28.
4. **Report the ratio and both absolute times**, so a later reader can see
   whether the baseline moved.

## Sources

- ONNX Runtime Web, WebGPU EP graph capture:
  https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html
- Babylon.js snapshot rendering:
  https://doc.babylonjs.com/setup/support/webGPU/webGPUOptimization/webGPUSnapshotRendering
- Render bundle best practices (Brandon Jones):
  https://toji.dev/webgpu-best-practices/render-bundles.html
- No compute bundles / command-buffer reuse: gpuweb issues
  https://github.com/gpuweb/gpuweb/issues/4138 and
  https://github.com/gpuweb/gpuweb/issues/1971
