// New file, no direct Python/jet/mantaflow source -- a simple uniform
// velocity-decay "viscosity" stage, added at the user's own explicit
// request after they found a real (non-numerical) problem in
// examples/19-fuel-fire/: widening its fuel source and softening its
// buoyancy for a thicker plume "stem" (that file's own header comment)
// gave a large-scale, slowly-reversing sideways recirculation -- already
// present in some form in this scene from the very start (a buoyant
// plume rising from a small, perfectly symmetric source in a tall,
// mostly-closed domain is an unstable equilibrium, like a pencil balanced
// on its tip: any infinitesimal asymmetry, even pure floating-point
// rounding noise, eventually tips it one way) -- much more time to grow
// and dominate the whole canvas before exiting through the domain's own
// small top-wall outflow, confirmed directly from the user's own
// real-hardware screenshots (u.sum staying one-signed for thousands of
// frames at a stretch instead of reversing every few hundred).
//
// This port has no real Navier-Stokes viscosity model at all
// (grid_solver2.js's own header comment: "Viscosity stays a no-op --
// explicitly deferred... not built yet"), but that same file's own
// computeViscosity stage hook (part of jet's own established
// force -> viscosity -> pressure -> advection order) is already fully
// composable with zero changes to that file -- this is the first thing
// plugged into it.
//
// *** Why a uniform per-frame decay, not a real Laplacian viscosity ***
//
// A true viscosity term (diffusing velocity via its own Laplacian) mostly
// damps *small*-scale features -- sharp gradients decay fastest under
// diffusion, and a smooth, domain-spanning recirculation cell is the
// *lowest*-frequency mode in the grid, the one real diffusion damps
// slowest of all. That's exactly backwards from this problem, where the
// offending structure is the single largest-scale mode in the domain. A
// uniform decay (multiply every velocity component by the same constant
// each frame) damps every scale equally, including that large-scale mode
// directly -- simpler, cheaper (a single per-cell multiply, no stencil),
// and unconditionally stable (no CFL-like constraint a real diffusion
// solve would need to worry about). Directly analogous to this port's own
// already-established smokeDecay/temperatureDecay convention
// (grid_smoke_solver2.js, grid_fire_solver2.js) -- same idea, applied to
// velocity instead of a scalar field (array_utils.js's own
// createDecayKernel2, a pure in-place sibling of those solvers' own
// raw-to-state decay kernels).
//
// *** Boundary conditions after damping ***
//
// grid_solver2.js's own header comment already documents this lesson
// directly: "a caller-supplied stage override is responsible for its own
// boundary handling." A uniform decay can't introduce a *new* boundary
// violation on its own in a domain with no inflows (closed-wall/collider
// velocities are already 0 there, and 0 times any constant is still 0) --
// but calling boundarySolver.constrainVelocity() after damping anyway,
// matching every other stage's own convention exactly, keeps this
// correct even for a future caller with a live inflow, where decaying its
// own prescribed non-zero velocity *would* otherwise drift it away from
// that inflow's own value.
//
// *** Usage ***
//
// options.computeViscosity on createGridSolver2 *replaces* the default
// no-op stage entirely, it isn't called alongside it (see that file's own
// header comment) -- and it has to be handed in as a plain function at
// construction time, before createGridSolver2 has returned its own
// boundarySolver. Resolved with a plain forward-reference closure
// variable, assigned right after construction, since computeViscosity
// itself isn't actually *called* until some later frame's own
// onAdvanceTimeStep():
//
//   let boundarySolver;
//   const damping = createVelocityDamping2( { velocityGrid, dampingCoefficient: 0.02 } );
//   const solver = createGridSolver2( {
//     ...,
//     computeViscosity: () => { damping.applyDamping(); boundarySolver.constrainVelocity(); }
//   } );
//   boundarySolver = solver.boundarySolver;

import { createDecayKernel2 } from './array_utils.js';

// options.velocityGrid: a FaceCenteredGrid2 (grid_data2.js) -- decayed in
// place, every call to applyDamping().
// options.dampingCoefficient: plain number or an already-invoked live node
// (this port's own "number or node" convention) -- fraction of velocity
// removed each call, e.g. 0.02 removes 2% every time applyDamping() runs.
// 0 (the default) is a genuine no-op, safe to leave wired in permanently
// and tune later without touching the wiring itself.
export function createVelocityDamping2( { velocityGrid, dampingCoefficient = 0 } ) {

	const decayU = createDecayKernel2( velocityGrid.dataU, dampingCoefficient );
	const decayV = createDecayKernel2( velocityGrid.dataV, dampingCoefficient );

	function applyDamping() {

		decayU();
		decayV();

	}

	return { applyDamping };

}
