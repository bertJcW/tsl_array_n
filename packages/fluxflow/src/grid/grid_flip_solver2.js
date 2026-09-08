// 2D FLIP (fluid-implicit-particle) liquid solver, read directly from
// mantaflow's own `source/plugin/flip.cpp` (`mapPartsToMAC`,
// `markFluidCells`, `flipVelocityUpdate`, `sampleFlagsWithParticles`,
// `extrapolateMACFromWeight`/`extrapolateMACSimple` -- Apache-2.0, Tobias
// Pfaff & Nils Thuerey; see ../../THIRD-PARTY-NOTICES.md) and its own
// simplest reference scene, `scenes/flip01_simple.py` ("very simple flip
// without level set and without any particle resampling"), fetched and
// read directly rather than assumed. First particle-based solver in this
// port -- every other solver so far evolves fields defined *on* the grid;
// FLIP moves independent particles through it instead.
//
// *** Scope, deliberately matching flip01_simple.py's own minimum, not
// mantaflow's full feature surface ***
//
// Fixed particle count, seeded once via computeFlipBoxSeed (below), for the
// solver's whole lifetime -- tsl_array_n's own arrays are fixed-size at
// construction (confirmed by reading packages/tsl_array_n/src/array.js
// directly: instancedArray(count,type), no resize/dispose exists anywhere),
// so mantaflow's own literal kill+spawn resampling (adjustNumber) would need
// new platform capability this port doesn't have. Density resampling within
// that fixed budget *is* implemented, though -- see the "Particle
// resampling" section below for why it turned out to be necessary, not
// optional. No free-surface level set (a genuinely
// new grid/reinitialization subsystem, unrelated to the SDF colliders
// already built) -- "is this cell fluid" is decided purely by live particle
// occupancy (mantaflow's own markFluidCells), which is simpler and
// sufficient for the physics (a level set in mantaflow is mainly for
// smoother rendering/reseeding, not required for the pressure solve
// itself).
//
// *** Orchestration: owns velocity+pressure directly, does not wrap or
// compose into createGridSolver2 ***
//
// createGridSolver2's own four stage hooks assume self-advected velocity
// (a velocityPrev clone + semi-Lagrangian back-trace, built unconditionally
// at construction) -- every other solver in this port self-advects; FLIP
// never does, it *rebuilds* velocity from particles every frame (P2G), and
// has a particle-position-integration step with no analogue in any hook.
// This factory therefore builds createGridBlockedBoundaryConditionSolver2
// and createGridPressureSolver2 directly, the same primitives
// createGridSolver2 itself builds on, one level down -- not a decoupled
// add-on like grid_fire_solver2.js/velocity_damping2.js (those only
// *contribute* a force/decay to someone else's solver; FLIP owns and
// rewrites velocity outright).
//
// *** The one genuinely new mechanism: GPU-atomic particle-to-grid
// scatter, with a real gotcha found and fixed before this file was written,
// not discovered the hard way inside the full loop ***
//
// mantaflow's own mapPartsToMAC scatters each particle's velocity onto the
// MAC grid via a bilinear-weighted "add my contribution to each of my 4
// nearest faces" step, then divides by an accumulated weight -- CPU-side
// (setInterpolated), safe there because mantaflow has no concurrency to
// worry about. On this port's own GPU-parallel-per-particle-thread model,
// multiple particles can target the *same* face at once, so this needs a
// real scatter-add: one atomic accumulator *per face* (not the single
// shared 0-D accumulator every existing atomic reduction in this port uses
// -- linalg.js's own CG dot product, grid_pressure_solver2.js's own bad-
// cell counter), scattered into by a per-thread-*computed* index. Verified
// directly against tsl_array_n's own array.js before relying on it:
// arrayN(type,shape) always does `instancedArray(count,type)` with an
// already-flattened count, identically whether shape is `[]` (the existing
// 0-D case) or `[nx,ny]` -- `.toAtomic()` marks that same flat node either
// way, so atomicAdd(accum(i,j), value) is structurally the same operation
// as every existing atomicAdd(accum(), value) in this port, just indexed.
// **A real gotcha, caught by an isolated real-hardware test before this
// file was written, not assumed**: a *later*, same-frame kernel reading an
// atomic-marked node's current value via an ordinary node reference (no
// different from reading any other field) compiles without error but
// silently reads back 0 -- confirmed on real WebGPU hardware. The correct
// read is `atomicLoad(node)` (three.js TSL's own export, re-exported from
// `three/tsl` alongside `atomicAdd`) -- every read of a P2G accumulator
// below uses it explicitly.
//
// *** Particle advection reuses an already real-hardware-validated code
// path, not a new integrator ***
//
// advection_solver2.js's own internal trace(startPos, direction) closure
// already supports direction=-1 (forward integration through velocity) --
// its own header comment says so directly, and this exact direction is
// already dispatched on real hardware today, inside advectFaceCentered2's
// own MacCormack backward-correction step. trace() takes a plain position
// with no assumption it's a grid-cell center, so reusing it for a
// particle's own arbitrary position needs zero new integration math --
// this file's only change to that one is returning `trace` alongside the
// two field-advection functions it already returned. RK2 (this port's own
// established substep scheme) over mantaflow's own literal RK4 is a
// deliberate choice: reuse of an already-proven path over a second,
// differently-ordered integrator's own new verification burden.
//
// *** markFluidCells feeds this port's own pre-existing Dirichlet
// mechanism, not a new liquid-specific pressure code path ***
//
// mantaflow's own pressure solve treats "no fluid here" cells as a zero-
// pressure ghost boundary for their fluid neighbors -- exactly the
// mechanism sdf_inflow_outflow2.js's own createOutflowPressureDirichlet2
// already reproduces for outflow (see that file's own header comment for
// the full derivation). grid_pressure_solver2.js's own `dirichlet(pos)`
// callback is already re-evaluated fresh on *every* project() dispatch
// (confirmed by reading dispatchBuildSystem directly), so a mask
// recomputed from live particle occupancy each frame -- instead of a
// static SDF, or a slowly-moving pointer vent -- is already exactly what
// that contract supports. No new pressure-solver capability needed.
//
// *** Collider/obstacle interaction -- threads into two already-existing,
// already real-hardware-validated mechanisms, plus one genuinely new
// particle-side piece ***
//
// Grid-side: options.collider (optional, an SDFStaticCollider2/
// SDFRigidBodyCollider2, sdf_collider2.js) is forwarded straight to
// createGridBlockedBoundaryConditionSolver2's own constructor -- exactly
// the same option, same meaning, as createGridSolver2's own `collider`
// (grid_solver2.js). That solver's constrainVelocity() -- already called
// below in the same three places every other solver in this port calls it
// (after a stage that touches velocity) -- already implements the *entire*
// grid-side collider treatment (mark collider faces, set them to
// collider.velocityAt, free-slip extrapolate, no-flux + friction
// projection, re-apply at collider/fluid boundaries) internally, gated
// purely on whether a collider was passed in. So threading it through here
// needed zero new logic of its own. Pressure needs no new logic either:
// grid_pressure_solver2.js has no solid/obstacle category at all (see that
// file's own header comment) -- a collider's effect on pressure is already
// entirely indirect everywhere in this port, via constrainVelocity()
// zeroing/setting collider-adjacent velocity *faces* immediately before
// project()'s divergence computation reads velocityGrid. One
// FLIP-specific wrinkle worth naming plainly: fluidMask's own dirichlet()
// (below) treats *any* unoccupied cell, including a collider's own
// interior once particles are correctly excluded from it, as Dirichlet-
// zero, same as any other empty/air cell -- a Dirichlet approximation of
// what would ideally be a Neumann/no-flux solid condition, but the *same*
// level of approximation this port's pressure solver already accepts
// everywhere else (it has no true solid-cell category to do better with),
// not a new gap introduced here.
//
// Particle advection: options.collider is also forwarded to this file's
// own createSemiLagrangianAdvectionSolver2 call below -- that factory's
// own `collider` option (advection_solver2.js) already makes its internal
// trace()/backTrace() clamp a traced substep to the collider surface if it
// would cross into solid (confirmed already exercised on real hardware via
// examples/09-advection's own wall-collider case). trace() takes a plain
// position with no assumption it's a grid-cell center, so this "just
// works" for particle advection too -- no new code, just passing the same
// option through to an already-collider-aware factory.
//
// Particle push-out: the one piece genuinely missing anywhere in this
// port before now -- nothing previously corrected a *particle's own
// position* against a collider, only grid velocity was ever protected.
// Ported directly from mantaflow's own pushOutofObs/knPushOutofObs
// (source/plugin/flip.cpp, re-fetched and read directly to confirm the
// exact formula, not assumed from memory): if a particle's own SDF sample
// is below a threshold (default 0, matching the Python wrapper's own
// default), push its position along the SDF's own (normalized) gradient
// by (thresh - v + shift) -- a first-order linear correction, exact only
// for a locally-planar SDF, matching mantaflow's own single-pass (no
// iteration/re-check) behavior exactly, not strengthened beyond it. Pure
// position correction, no velocity write at all, matching upstream:
// mantaflow relies entirely on the grid-side velocity BC above to keep
// velocity physically consistent, and the particle picks up a
// collider-consistent velocity again on the very next frame's G2P. Only
// built when a real collider is passed (mirrors grid_blocked_boundary_
// condition_solver2.js's own rebuildColliderKernels()'s `if (!collider)`
// pattern -- collider.sample/gradient don't exist to call on `undefined`).
// Dispatched in onAdvanceTimeStep() right after advectParticles(), before
// P2G -- cleans up whatever the advection clamp's own first-order
// approximation didn't already catch, before particle velocities get
// scattered onto the grid.
//
// Moving colliders specifically: createSDFRigidBodyCollider2's own
// velocityAt(point) bakes currentPosition/currentAngle into the returned
// TSL graph at kernel-*build* time (a documented limitation in that file's
// own header comment) -- collider.update(dt) alone does not make an
// already-built kernel see the new pose. grid_blocked_boundary_condition_
// solver2.js already has the workaround, just never exercised for
// *continuous* per-frame motion before now: every collider-dependent
// kernel is rebuilt inside rebuildColliderKernels(), which reruns on every
// setCollider() call. So a caller wanting continuous motion just needs to
// call boundarySolver.setCollider(rigidCollider, ...) every frame, right
// after rigidCollider.update(dt) -- boundarySolver is already exposed on
// this factory's own returned object today, so this needed zero FLIP-
// internal orchestration change; see examples/23-flip-moving-collider/
// for where that per-frame call actually happens. Note collider.sample/
// gradient (used by both the advection clamp above and pushOutOfCollider)
// need no rebuild ever -- they read a live texture, safe to bake into a
// kernel graph once -- only the *velocity*-based mechanisms need the
// per-frame setCollider() call.
//
// *** Concurrency shapes new to this codebase, both used below, each
// individually reasoned through rather than assumed safe ***
//
// (1) markFluidCells: multiple particle-threads may write the identical
// constant 1 to the same cell -- a tolerable same-value race (a single-
// word write of an agreed value can't tear), not a data hazard, but a
// genuinely new *shape* for this port (every existing multi-writer case
// here has been sequential kernel dispatches, not concurrent same-frame
// threads targeting the same cell). (2) The P2G atomic scatter itself
// (see above). Both are standard, well-understood GPU patterns, verified
// as described above where verification was actually possible before
// writing this file -- but this file is new enough, and atomics-only code
// is untestable on this port's own WebGL2-fallback sandbox at all (it
// cannot compile atomicAdd/atomicLoad), that real-hardware verification via
// claude-in-chrome is the *only* way any of this has actually been checked,
// not merely reasoned about.
//
// *** Particle resampling -- found necessary by testing, not built
// speculatively; the effect of mantaflow's own adjustNumber, adapted to
// this port's fixed-capacity particle arrays ***
//
// The initial scope note above ("no particle resampling") turned out to be
// a real gap, not just a v1 simplification: a user-reported, then directly
// investigated, finding on examples/20-flip-dam-break/ showed particle
// density drifting over hundreds of frames -- occupied-cell count falling
// ~17%, one cell's own particle count climbing from the seeded 4 up to 17,
// *still worsening* at frame 700, not leveling off. Isolated by testing, not
// assumed: a from-rest control scene (a flat resting pool, zero initial
// velocity) showed *zero* drift over the same 700 frames -- the compaction
// is tied specifically to dynamic particle motion (advection/P2G/G2P
// through a moving fluid), not to gravity+pressure+closed-domain alone.
// Confirmed against mantaflow's own source directly (source/plugin/flip.cpp,
// re-fetched): its own markFluidCells is *also* pure particle-occupancy,
// same as this port -- not the difference. The real difference is
// adjustNumber, called every frame in mantaflow's own fuller scenes
// (scenes/flip02_surface.py), which kills excess particles from over-full
// bulk cells and reseeds under-full ones -- this port's own v1 explicitly
// didn't port it, for the fixed-array reason stated above, and the drift is
// the direct, essentially inevitable consequence of that gap.
//
// This implements the *effect* of adjustNumber without needing array
// resize: instead of killing a particle and spawning a new one, an excess
// particle in an over-full cell is *relocated* directly into an under-full
// cell -- same fixed particle budget throughout, just redistributed. Four
// kernels, run once per frame:
// (1) countPerCellKernel: one thread per particle, atomicAdd into
// cellParticleCount -- same cell-index arithmetic markFluidCellsKernel
// already uses (factored out as cellIndexOf() below, shared by both).
// (2) buildDonorPoolKernel: one thread per particle -- if its own cell's
// count exceeds maxParticlesPerCell, it pushes its own particle index into
// donorPool via atomicAdd on donorPushCursor, using the atomic op's own
// *return value* as the claimed slot (the classic GPU "claim a unique slot"
// pattern) -- every over-threshold particle volunteers as a candidate;
// nothing moves yet, so volunteering costs nothing if never claimed.
// **This is the one genuinely new primitive in this file**: every existing
// atomic use elsewhere (P2G, CG dot products, bad-pressure-cell counting)
// only uses the side effect, never the returned pre-increment value --
// confirmed safe first via an isolated real-hardware scratch test (N
// threads each claim a slot, verified afterward that every slot 0..N-1 was
// written exactly once, no collisions, at both N=256 and N=4000) before
// being relied on here, the same discipline the original P2G scatter got.
// (3) claimDonorsKernel: one thread per *cell* -- if its own count is below
// minParticlesPerCell, it claims up to (minParticlesPerCell - count) donors
// via Loop(minParticlesPerCell, ...) (this file's own first use of the
// bounded-loop-with-early-Break idiom already real-hardware-proven in
// advection_solver2.js's own backTrace), each claim doing its own atomicAdd
// on donorPopCursor and stopping once either the cell has enough or the pop
// cursor exceeds donorPushCursor's own live value (read via atomicLoad
// directly inside this kernel -- no CPU readback needed at all, simpler
// than the original plan's own JS-side-count draft). Each successful claim
// overwrites the donor's own positions() entry to the recipient cell's
// center and velocities() via faceCenteredValueAtPosition2 sampled there
// (the same helper g2pUpdate already uses) -- the relocated particle picks
// up the local fluid velocity at its new home, not whatever it had at the
// old one. No jitter (unlike computeFlipBoxSeed's own initial seeding) --
// this only touches a small minority of particles most frames, not the
// whole field, so a new GPU-random primitive isn't worth adding for it.
// Run right after computeUFluidAdjacent()/computeVFluidAdjacent() (mirrors
// mantaflow's own placement: right after markFluidCells, before
// gravity/pressure/FLIP-update) and before applyGravityU/V(). Because
// relocation changes particle *positions*, the fluidMask/uFluidAdjacent/
// vFluidAdjacent computed just before it are stale for the recipient cells
// -- clearFluidMask()/markFluidCellsKernel()/computeUFluidAdjacent()/
// computeVFluidAdjacent() are called a second time immediately after, so
// this same frame's pressure solve sees the corrected distribution instead
// of lagging a frame. All four already existed; this just calls them twice.
//
// options.resample ({ enabled=true, minParticlesPerCell=4,
// maxParticlesPerCell=8 }) -- defaults match mantaflow's own reference
// (minParticles = 2^dim, i.e. exactly this port's own existing
// particlesPerCellAxis=2 convention; maxParticles = 2x that). A
// shared-solver fix, not per-example -- every FLIP example benefits once
// this lands, no per-example changes needed.
//
// *** Velocity damping -- the user found the simulation itself too
// energetic ("looks like tumbling lava, not water"), asked whether
// mantaflow exposes a viscosity-like dissipation knob ***
//
// Checked mantaflow's own flip.cpp directly rather than guessing: it has
// no velocity-viscosity/damping mechanism at all (the only smoothing
// kernels, knSmoothGrid/knSmoothGridNeg, post-process a *level-set*
// surface reconstruction this port doesn't have -- they never touch
// velocity). Both of mantaflow's own reference scenes
// (flip01_simple.py, flip02_surface.py) call flipVelocityUpdate with
// flipRatio=0.97, identical to this port's own already-matching default
// -- so the energetic behavior isn't a value mismatch against mantaflow's
// own reference, it's an inherent property of FLIP at a high flip ratio in
// general (each particle carries its own velocity forward frame to frame,
// only lightly blended toward the grid's own pressure-projected value, so
// per-particle noise/energy has very little numerical dissipation to
// remove it). Confirmed quantitatively before fixing anything, not
// assumed: a real-hardware checkpointed run of this file's own dam-break
// scene (examples/20-flip-dam-break/) showed average particle speed
// oscillating between ~5-10 units/sec (peaks repeatedly spiking to
// 27-46) across a full 330-frame/11-second window with no decay trend at
// all -- not settling, not a rendering artifact.
//
// No mantaflow mechanism exists to port, so this reuses the *idea*
// already proven in velocity_damping2.js (a uniform per-frame velocity
// decay -- unlike a real Laplacian viscosity, damps every length scale
// equally rather than sparing the largest one, and needs no stencil or
// CFL-like stability constraint; see that file's own header comment for
// the full reasoning) -- but not that file's own implementation directly,
// since it decays a FaceCenteredGrid2's own dataU/dataV (grid-space),
// while the noise here is carried by *particles*. Implemented instead as
// one extra multiply inside this file's own already-existing G2P kernel
// (buildG2PUpdate, right after the flipRatio blend, before the velocity
// clamp) -- no new kernel dispatch. `velocityDamping: 0.02` (matching the
// exact magnitude round 12 already validated for the same class of
// problem in grid_fire_solver2.js, not picked arbitrarily) was then
// confirmed effective on real hardware: the same dam-break scene now
// decays cleanly and monotonically from the initial collapse to
// near-rest (avgSpeed<1, maxSpeed 2-3) by roughly 8 seconds (frame
// 240-270), instead of oscillating indefinitely. A shared-solver default,
// like flipRatio and resample above -- every existing FLIP example gets
// this automatically.
//
// *** Risks, named plainly (collider interaction specifically -- see the
// section above this one for the rest) ***
//
// 1. Real-hardware-only verification, same as every other atomics-touching
// mechanism in this file (see immediately above) -- none of this is
// checkable on the WebGL2 fallback.
// 2. The Dirichlet-vs-Neumann pressure approximation this file already
// documents above (fluidMask's own dirichlet() treats a collider's interior
// as just another empty/air cell, not a true no-flux solid) is inherited
// from grid_pressure_solver2.js's own pre-existing, project-wide scope cut
// -- not new here, but worth restating in one place a caller is likely to
// look.
// 3. pushOutOfCollider is a single first-order linear correction per frame
// (thresh - v + shift along the SDF gradient), exactly matching mantaflow's
// own pushOutofObs -- not iterated, not re-checked. Exact only for a
// locally-planar SDF. A particle that penetrates deeply in a single frame
// (a large dt, a fast-moving collider, or a thin/sharply-curved feature)
// may not be fully resolved by one push -- combined with the advection
// clamp's own similar first-order limitation, a fast enough or thin enough
// collider can still let a particle tunnel through in one frame. Inherited
// from upstream mantaflow, not a regression introduced by this port.
// 4. Collider geometry (static or moving) should stay clear of the domain's
// own rectangular boundary. pushOutOfCollider has no domain-bounds re-clamp
// of its own (it relies on advectParticles's own pre-existing clamp, which
// runs earlier in the same frame) -- an obstacle or basin wall placed too
// close to, or a moving collider that reaches, the domain edge makes this
// solver's own closedDomainBoundaryFlag treatment and the collider's own
// velocity treatment fight over the same faces (confirmed on real hardware:
// examples/23-flip-moving-collider/'s own header comment records a real
// instance -- persistently elevated pressure-solve rejections once a fast
// paddle reached the domain's own wall). This is an example-authoring
// constraint, not something enforced in code, matching this port's existing
// "trust the caller to build a sane scene" convention elsewhere.
// 5. Resampling's own donor pool is sized maxParticles (an upper bound that
// can never be exceeded, since at most every particle could volunteer) --
// but a recipient cell's own claim is capped by however many donors
// actually exist *this frame*, not by how many it asked for. If the whole
// field is genuinely short on excess particles (e.g. very early in a run,
// before any cell has exceeded maxParticlesPerCell yet), some under-full
// cells simply stay under-full that frame rather than being force-filled --
// matching mantaflow's own adjustNumber, which reseeds from a level set
// rather than manufacturing particles from nothing either.
// 6. Relocating a particle to a cell center with no jitter (see above)
// means a frame with many simultaneous relocations into the *same* cell
// could momentarily place particles exactly on top of each other -- P2G's
// own bilinear scatter handles coincident particles fine (it's just a sum),
// but this is a real, accepted simplification, not a proven-harmless one;
// flagged rather than silently assumed away.

import * as tsl_array_n from 'tsl_array_n';
import { vec2, float, int, floor, round, clamp, max, min, atomicAdd, atomicLoad, If, Loop, Break } from 'three/tsl';
import { createGridBlockedBoundaryConditionSolver2 } from './grid_blocked_boundary_condition_solver2.js';
import { createGridPressureSolver2 } from './grid_pressure_solver2.js';
import { createSemiLagrangianAdvectionSolver2 } from './advection_solver2.js';
import { createCopyKernel2, createExtrapolateToRegion2 } from './array_utils.js';
import { bilinearCoordsAndWeights2, collocatedValueAtPosition2, faceCenteredValueAtPosition2 } from './grid_math.js';
import { DEFAULT_ATOMIC_DOT_SCALE } from '../linalg/linalg.js';

// Last-resort circuit breaker on a *particle's* own velocity, mirroring
// grid_blocked_boundary_condition_solver2.js's own MAX_VELOCITY_COMPONENT
// (that one bounds the grid; this one bounds particles, which don't
// otherwise pass through that clamp at all). Not tuned against any
// specific scene yet -- generous on purpose, same "catch a genuine
// runaway, don't bound normal physical variation" spirit.
const MAX_PARTICLE_VELOCITY = 500;

function numberOrNode( value ) {

	return typeof value === 'number' ? float( value ) : value;

}

// Seeds a rectangular box with particlesPerCellAxis^2 particles per grid
// cell, jittered within their own sub-cell (mantaflow's own
// sampleShapeWithParticles does the same, via a Box shape -- this is the
// axis-aligned-rectangle-only special case, sufficient for a dam-break/
// falling-block scene; ../../examples/20-flip-dam-break/ is exactly this).
// Pure JS, no TSL/renderer involved at all -- directly unit-testable, and
// usable before or after createGridFlipSolver2's own construction.
// options.boxMin/boxMax: [x,y] world-space corners. gridSpacingX/Y: this
// solver's own gridSpacing (sub-cell size is derived from it, not an
// independent parameter). particlesPerCellAxis: particle count per cell is
// this squared (2 -> 4 particles/cell, mantaflow's own 2D default is 3).
// jitter: fraction of a sub-cell's own size to randomly offset each
// particle by, avoiding a perfectly regular (and therefore both visually
// artificial and dynamically atypical -- see grid_fire_solver2.js's own
// symmetry-breaking precedent for why a perfectly regular initial
// condition is worth avoiding on its own) starting lattice.
export function computeFlipBoxSeed( { boxMin, boxMax, gridSpacingX, gridSpacingY, particlesPerCellAxis = 2, jitter = 0.2 } ) {

	const [ minX, minY ] = boxMin;
	const [ maxX, maxY ] = boxMax;

	const cellsX = Math.max( 1, Math.round( ( maxX - minX ) / gridSpacingX ) );
	const cellsY = Math.max( 1, Math.round( ( maxY - minY ) / gridSpacingY ) );
	const subSpacingX = gridSpacingX / particlesPerCellAxis;
	const subSpacingY = gridSpacingY / particlesPerCellAxis;

	const positionsList = [];

	for ( let cellJ = 0; cellJ < cellsY; cellJ ++ ) {

		for ( let cellI = 0; cellI < cellsX; cellI ++ ) {

			for ( let subJ = 0; subJ < particlesPerCellAxis; subJ ++ ) {

				for ( let subI = 0; subI < particlesPerCellAxis; subI ++ ) {

					const jx = ( Math.random() * 2 - 1 ) * jitter * subSpacingX;
					const jy = ( Math.random() * 2 - 1 ) * jitter * subSpacingY;

					const x = minX + cellI * gridSpacingX + ( subI + 0.5 ) * subSpacingX + jx;
					const y = minY + cellJ * gridSpacingY + ( subJ + 0.5 ) * subSpacingY + jy;

					positionsList.push( x, y );

				}

			}

		}

	}

	const count = positionsList.length / 2;

	return {
		count,
		positionsArray: new Float32Array( positionsList ),
		velocitiesArray: new Float32Array( count * 2 ) // zeroed -- particles start at rest
	};

}

// options.velocityGrid: required -- a FaceCenteredGrid2 (grid_data2.js)
// this solver owns and rewrites every frame (unlike grid_fire_solver2.js's
// read-only use of the same type -- see this file's own header comment on
// why FLIP can't be decoupled the same way).
// options.maxParticles: required -- fixed for this solver's whole lifetime
// (see this file's own header comment on why). Construct with
// computeFlipBoxSeed's own `count`, then seed positions/velocities
// explicitly afterward (see the returned `positions`/`velocities` fields'
// own comment below) -- matching sdf_collider2.js's own "construct, then
// populate" convention rather than baking scene geometry into this factory.
// options.dt: plain number or live node (this port's own "number or node"
// convention throughout).
// options.gravity: [x,y], plain numbers -- a constant per-frame
// acceleration added directly to velocity, mantaflow's own addGravity.
// options.flipRatio: FLIP/PIC blend for the G2P velocity update (see
// buildG2PUpdate below) -- mantaflow's own scene default is 0.97 (97%
// FLIP -- keep each particle's own momentum/noise -- 3% PIC -- damp it a
// little for stability); matched here as this solver's own default too.
// options.velocityDamping: uniform per-frame fraction of each particle's
// own post-blend velocity removed in the same G2P kernel, right after the
// flipRatio blend -- see this file's own header comment above ("Velocity
// damping") for why this exists and how its default was chosen; 0 is a
// genuine no-op, safe to leave wired in permanently. Accepts a plain number
// or an already-invoked live node (this port's own "number or node"
// convention, same as dt/flipRatio) -- pass a live array0('float')'s own
// reference for a runtime-adjustable control, e.g. a UI slider (see
// examples/20-flip-dam-break/'s own control panel). Unconditionally
// clamped to [0,1] inside the kernel regardless of what's passed in --
// negative values would amplify velocity every frame instead of damping
// it, so this can't be pushed into an unsafe range from outside.
// options.p2gAtomicScale: fixed-point scale for the P2G scatter's own
// atomic accumulators (see linalg.js's DEFAULT_ATOMIC_DOT_SCALE comment for
// the general mechanism) -- a **per-instance option, not a shared global
// default**, following this same project's own hard lesson from
// grid_pressure_solver2.js's own maxPlausiblePressure regression: a "safe"
// atomic-reduction magnitude tuned against one scene does not reliably
// transfer to a differently-scaled one. **Confirmed on real hardware, not
// just anticipated**: the very first real-hardware run of examples/20-
// flip-dam-break/ hit a genuine single-frame pressure blowup (into the
// hundred-thousands) traced to `options.pressure.atomicScale` (a separate,
// pre-existing parameter forwarded straight through to
// createGridPressureSolver2's own CG solve -- distinct from this file's
// own p2gAtomicScale above, which never showed a problem) being left at
// its own library default -- too large for that scene's own gravity-
// driven divergence magnitude despite a small grid. Fixed there by tuning
// `pressure.atomicScale` down explicitly, same as `p2gAtomicScale`'s own
// reasoning above -- see that example's own header comment for the full
// story. Any new FLIP scene should expect to tune *both* independently,
// not assume either one is universal.
// options.weightEpsilon: a P2G-accumulated face is only trusted (used
// directly, rather than left for extrapolation to fill in) once its own
// accumulated weight exceeds this -- mirrors mantaflow's own implicit
// "stomp small values in weight to zero" step (knSafeDivReal).
// options.closedDomainBoundaryFlag: optional, forwarded to boundarySolver's
// own mutable property (grid_solver2.js's own established convention) --
// omit to keep every wall closed, matching flip01_simple.py's own domain.
// options.collider: optional SDFStaticCollider2/SDFRigidBodyCollider2
// (sdf_collider2.js) -- forwarded to both createGridBlockedBoundaryCondition
// Solver2 (grid-side velocity treatment) and createSemiLagrangianAdvection
// Solver2 (particle-trace tunneling clamp), and gates construction of the
// new pushOutOfCollider kernel (particle-side position correction). See
// this file's own header comment above for the full design. Like
// createGridSolver2's own same-named option, a *real* (non-null) collider
// here makes construction dispatch a kernel immediately (blockMarker) --
// needs tsl_array_n.init() to have already run.
// options.colliderPushThresh/colliderPushShift: forwarded to
// pushOutOfCollider (mantaflow's own pushOutofObs thresh/shift) -- default
// 0/0, matching that function's own Python-wrapper defaults exactly.
// Meaningless without options.collider.
// options.pressure: forwarded to createGridPressureSolver2 (e.g.
// { multigrid, tolerance, maxIterations, atomicScale, maxPlausiblePressure })
// -- `dirichlet` itself is NOT forwardable here, this factory builds its
// own from live particle occupancy (see dirichlet() below).
// options.resample: { enabled=true, minParticlesPerCell=4,
// maxParticlesPerCell=8 } -- see this file's own header comment above (the
// "Particle resampling" section) for the full design and why it's needed,
// not just a tuning knob. Defaults match mantaflow's own reference
// (scenes/flip02_surface.py) and this port's own existing
// computeFlipBoxSeed default (particlesPerCellAxis=2 -> 4 particles/cell).
export function createGridFlipSolver2( {
	velocityGrid,
	gridSpacing = [ 1, 1 ],
	origin = [ 0, 0 ],
	maxParticles,
	dt,
	gravity = [ 0, -9.81 ],
	flipRatio = 0.97,
	velocityDamping = 0.02,
	p2gAtomicScale = DEFAULT_ATOMIC_DOT_SCALE,
	weightEpsilon = 1e-4,
	closedDomainBoundaryFlag,
	collider,
	colliderPushThresh = 0,
	colliderPushShift = 0,
	resample = {},
	pressure = {}
} = {} ) {

	if ( ! velocityGrid ) {

		throw new Error( 'createGridFlipSolver2: options.velocityGrid is required.' );

	}

	if ( ! maxParticles ) {

		throw new Error( 'createGridFlipSolver2: options.maxParticles is required.' );

	}

	const {
		enabled: resampleEnabled = true,
		minParticlesPerCell = 4,
		maxParticlesPerCell = 8
	} = resample;

	const [ resolutionX, resolutionY ] = velocityGrid.resolution;
	const [ gridSpacingX, gridSpacingY ] = gridSpacing;
	const [ originX, originY ] = origin;

	const gridSpacingNode = vec2( gridSpacingX, gridSpacingY );
	const originNode = vec2( originX, originY );
	const cellCenterOrigin = originNode.add( gridSpacingNode.mul( 0.5 ) );

	const dataSizeU = velocityGrid.dataSizeU;
	const dataSizeV = velocityGrid.dataSizeV;
	const uCount = dataSizeU[ 0 ] * dataSizeU[ 1 ];
	const vCount = dataSizeV[ 0 ] * dataSizeV[ 1 ];

	const boundarySolver = createGridBlockedBoundaryConditionSolver2(
		velocityGrid, resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY, collider
	);
	if ( closedDomainBoundaryFlag !== undefined ) boundarySolver.closedDomainBoundaryFlag = closedDomainBoundaryFlag;

	// Particles -- fixed-capacity, explicitly zeroed here (this port's own
	// established "explicit clear, don't rely on implicit zero-init"
	// convention, e.g. grid_data2.js's own zeroScalarField2). A caller MUST
	// seed both via .fromArray() right after construction (see this
	// factory's own header comment) -- left zeroed, every particle would
	// sit at this solver's own origin corner, which is a valid but useless
	// starting state, not an error, so no defensive check is added for it.
	const positions = tsl_array_n.arrayN( 'vec2', maxParticles );
	const velocities = tsl_array_n.arrayN( 'vec2', maxParticles );
	positions.fromArray( new Float32Array( maxParticles * 2 ) );
	velocities.fromArray( new Float32Array( maxParticles * 2 ) );

	// Fluid occupancy, cell-centered, float-valued so it plugs directly
	// into collocatedValueAtPosition2 below -- matches grid_pressure_
	// solver2.js's own dirichletMaskField convention exactly (a 0/1 flag
	// stored as float, not int, purely so the same bilinear-sample helper
	// already used throughout this port can read it without a separate
	// int-field code path).
	const fluidMask = tsl_array_n.arrayN( 'float', [ resolutionX, resolutionY ] );
	fluidMask.fromArray( new Float32Array( resolutionX * resolutionY ) );

	const clearFluidMask = tsl_array_n.kernel( [ resolutionX, resolutionY ], ( i, j ) => {

		fluidMask( i, j ).assign( 0 );

	} );

	// Shared by markFluidCellsKernel below and the resample kernels further
	// down (countPerCellKernel/buildDonorPoolKernel) -- the same "which cell
	// is this continuous position in" formula, factored out once it needed
	// a third caller.
	function cellIndexOf( pos ) {

		const cellF = pos.sub( originNode ).div( gridSpacingNode );
		const i = max( 0, min( int( floor( cellF.x ) ), resolutionX - 1 ) );
		const j = max( 0, min( int( floor( cellF.y ) ), resolutionY - 1 ) );
		return { i, j };

	}

	const markFluidCellsKernel = tsl_array_n.kernel( maxParticles, ( p ) => {

		const { i, j } = cellIndexOf( positions( p ) );

		// Same-value race, not a data hazard -- see this file's own header
		// comment.
		fluidMask( i, j ).assign( 1 );

	} );

	function dirichlet( pos ) {

		const mask = collocatedValueAtPosition2( fluidMask, gridSpacingNode, cellCenterOrigin, pos, [ resolutionX, resolutionY ] );
		return { active: mask.lessThan( 0.5 ), target: float( 0 ) };

	}

	const pressureSolver = createGridPressureSolver2( {
		resolution: [ resolutionX, resolutionY ], gridSpacing, origin, dirichlet, ...pressure
	} );
	const projectDispatch = pressureSolver.project( velocityGrid, velocityGrid );

	// ---- P2G: one atomic accumulator pair per face component -- see this
	// file's own header comment for why this shape, and for the atomicLoad
	// gotcha every read below works around explicitly. ----

	const uNumerAccum = tsl_array_n.arrayN( 'int', dataSizeU );
	uNumerAccum.node.toAtomic();
	const uDenomAccum = tsl_array_n.arrayN( 'int', dataSizeU );
	uDenomAccum.node.toAtomic();
	const vNumerAccum = tsl_array_n.arrayN( 'int', dataSizeV );
	vNumerAccum.node.toAtomic();
	const vDenomAccum = tsl_array_n.arrayN( 'int', dataSizeV );
	vDenomAccum.node.toAtomic();

	const zeroU = new Int32Array( uCount );
	const zeroV = new Int32Array( vCount );

	function resetAccumulators() {

		uNumerAccum.fromArray( zeroU );
		uDenomAccum.fromArray( zeroU );
		vNumerAccum.fromArray( zeroV );
		vDenomAccum.fromArray( zeroV );

	}

	const uWeightValid = tsl_array_n.arrayN( 'int', dataSizeU );
	const vWeightValid = tsl_array_n.arrayN( 'int', dataSizeV );

	const p2gScaleNode = numberOrNode( p2gAtomicScale );
	const weightEpsilonNode = numberOrNode( weightEpsilon );

	// component: 'x' or 'y' -- a plain JS string, branched on once at
	// kernel-*build* time (not a GPU conditional), producing two
	// statically-specialized kernels (scatterU only ever reads .x,
	// scatterV only ever reads .y), same as every other "one kernel per
	// component" pair already established throughout this port.
	function buildScatter( component, dataOrigin, size, numerAccum, denomAccum ) {

		return tsl_array_n.kernel( maxParticles, ( p ) => {

			const value = component === 'x' ? velocities( p ).x : velocities( p ).y;
			const { i0c, j0c, i1c, j1c, w00, w10, w01, w11 } =
				bilinearCoordsAndWeights2( positions( p ), dataOrigin, velocityGrid.gridSpacing, size );

			const corners = [ [ i0c, j0c, w00 ], [ i1c, j0c, w10 ], [ i0c, j1c, w01 ], [ i1c, j1c, w11 ] ];

			for ( const [ i, j, w ] of corners ) {

				atomicAdd( numerAccum( i, j ), round( value.mul( w ).mul( p2gScaleNode ) ).toInt() );
				atomicAdd( denomAccum( i, j ), round( w.mul( p2gScaleNode ) ).toInt() );

			}

		} );

	}

	const scatterU = buildScatter( 'x', velocityGrid.dataOriginU, dataSizeU, uNumerAccum, uDenomAccum );
	const scatterV = buildScatter( 'y', velocityGrid.dataOriginV, dataSizeV, vNumerAccum, vDenomAccum );

	// One thread per *destination* face -- no atomics here at all, a plain
	// self-touch (matching this port's own established safe-in-place
	// shape), just reading the now-fully-accumulated atomic buffers via
	// atomicLoad(). Also writes weightValid, consumed by the pre-pressure
	// extrapolation pass below.
	function buildFinalize( dataComponent, size, numerAccum, denomAccum, weightValid ) {

		return tsl_array_n.kernel( size, ( i, j ) => {

			const numer = atomicLoad( numerAccum( i, j ) ).toFloat().div( p2gScaleNode );
			const denom = atomicLoad( denomAccum( i, j ) ).toFloat().div( p2gScaleNode );

			If( denom.greaterThan( weightEpsilonNode ), () => {

				dataComponent( i, j ).assign( numer.div( denom ) );
				weightValid( i, j ).assign( 1 );

			} ).Else( () => {

				dataComponent( i, j ).assign( 0 );
				weightValid( i, j ).assign( 0 );

			} );

		} );

	}

	const finalizeU = buildFinalize( velocityGrid.dataU, dataSizeU, uNumerAccum, uDenomAccum, uWeightValid );
	const finalizeV = buildFinalize( velocityGrid.dataV, dataSizeV, vNumerAccum, vDenomAccum, vWeightValid );

	// ---- extrapolation: createExtrapolateToRegion2 (array_utils.js)
	// reused exactly as grid_blocked_boundary_condition_solver2.js's own
	// extrapolateU/V already do -- see this file's own header comment.
	// Two passes per mantaflow's own extrapolateMACFromWeight (pre-
	// pressure, from P2G's own weight validity) vs extrapolateMACSimple
	// (post-pressure, from live fluid occupancy). ----

	const extrapolateWeightU = createExtrapolateToRegion2( velocityGrid.dataU, uWeightValid, velocityGrid.dataU, dataSizeU );
	const extrapolateWeightV = createExtrapolateToRegion2( velocityGrid.dataV, vWeightValid, velocityGrid.dataV, dataSizeV );

	const uFluidAdjacent = tsl_array_n.arrayN( 'int', dataSizeU );
	const vFluidAdjacent = tsl_array_n.arrayN( 'int', dataSizeV );

	// A U-face is "adjacent to fluid" if either of its two neighboring
	// cells (clamped at the domain edge, same idiom bilinearCoordsAndWeights2
	// itself already uses) is currently marked fluid.
	const computeUFluidAdjacent = tsl_array_n.kernel( dataSizeU, ( i, j ) => {

		const leftIdx = max( 0, min( i.sub( 1 ), resolutionX - 1 ) );
		const rightIdx = max( 0, min( i, resolutionX - 1 ) );
		const anyFluid = fluidMask( leftIdx, j ).greaterThan( 0.5 ).or( fluidMask( rightIdx, j ).greaterThan( 0.5 ) );

		uFluidAdjacent( i, j ).assign( anyFluid.select( int( 1 ), int( 0 ) ) );

	} );

	const computeVFluidAdjacent = tsl_array_n.kernel( dataSizeV, ( i, j ) => {

		const downIdx = max( 0, min( j.sub( 1 ), resolutionY - 1 ) );
		const upIdx = max( 0, min( j, resolutionY - 1 ) );
		const anyFluid = fluidMask( i, downIdx ).greaterThan( 0.5 ).or( fluidMask( i, upIdx ).greaterThan( 0.5 ) );

		vFluidAdjacent( i, j ).assign( anyFluid.select( int( 1 ), int( 0 ) ) );

	} );

	const extrapolatePressureU = createExtrapolateToRegion2( velocityGrid.dataU, uFluidAdjacent, velocityGrid.dataU, dataSizeU );
	const extrapolatePressureV = createExtrapolateToRegion2( velocityGrid.dataV, vFluidAdjacent, velocityGrid.dataV, dataSizeV );

	// ---- particle resampling: relocates excess particles from over-full
	// cells into under-full ones, within the same fixed particle budget --
	// see this file's own header comment (the "Particle resampling"
	// section) for the full design and why it's needed. Only built when
	// resampleEnabled, matching pushOutOfCollider's own "only build what a
	// disabled option can't otherwise use" convention. ----

	let resamplePass = null;

	if ( resampleEnabled ) {

		const cellParticleCount = tsl_array_n.arrayN( 'int', [ resolutionX, resolutionY ] );
		cellParticleCount.node.toAtomic();

		const donorPool = tsl_array_n.arrayN( 'int', maxParticles );
		const donorPushCursor = tsl_array_n.array0( 'int' );
		donorPushCursor.node.toAtomic();
		const donorPopCursor = tsl_array_n.array0( 'int' );
		donorPopCursor.node.toAtomic();

		const zeroCells = new Int32Array( resolutionX * resolutionY );
		const zeroOne = new Int32Array( [ 0 ] );

		function resetResampleBuffers() {

			cellParticleCount.fromArray( zeroCells );
			donorPushCursor.fromArray( zeroOne );
			donorPopCursor.fromArray( zeroOne );

		}

		const countPerCellKernel = tsl_array_n.kernel( maxParticles, ( p ) => {

			const { i, j } = cellIndexOf( positions( p ) );
			atomicAdd( cellParticleCount( i, j ), 1 );

		} );

		const maxParticlesPerCellNode = int( maxParticlesPerCell );

		const buildDonorPoolKernel = tsl_array_n.kernel( maxParticles, ( p ) => {

			const { i, j } = cellIndexOf( positions( p ) );
			const count = atomicLoad( cellParticleCount( i, j ) );

			If( count.greaterThan( maxParticlesPerCellNode ), () => {

				const slot = atomicAdd( donorPushCursor(), 1 );

				// Sized maxParticles, an upper bound that can never be
				// exceeded (at most every particle could volunteer) -- this
				// guard is defensive, not expected to ever trip.
				If( slot.lessThan( maxParticles ), () => {

					donorPool( slot ).assign( p );

				} );

			} );

		} );

		const minParticlesPerCellNode = int( minParticlesPerCell );

		// Guards against a real failure mode found by testing, not
		// anticipated: reseeding *every* under-full cell -- including a
		// truly isolated single-stray-particle speck, far from the fluid's
		// own bulk -- actively inflates that speck into a full
		// minParticlesPerCell-strength cell every frame, growing the
		// fluid's own apparent footprint well past its real extent (a
		// first version without this guard showed occupiedCells climbing
		// past 1800 on examples/20-flip-dam-break/, the opposite failure
		// from the one this whole mechanism exists to fix). mantaflow's own
        // adjustNumber avoids this by reseeding from a level set (a
		// *smoothed* union of nearby particles, not raw per-cell presence);
		// this port has no level set, so this is a local, GPU-cheap
		// approximation of the same distinction: only treat an under-full
		// cell as a genuine recipient if at least 2 of its own 4-connected
		// neighbors are themselves already at or above minParticlesPerCell
		// -- true for a real puddle's own edge/surface cells (which sit
		// against the bulk), false for an isolated droplet sitting alone.
		function neighborFluidCount( i, j ) {

			const left = atomicLoad( cellParticleCount( max( 0, i.sub( 1 ) ), j ) );
			const right = atomicLoad( cellParticleCount( min( resolutionX - 1, i.add( 1 ) ), j ) );
			const down = atomicLoad( cellParticleCount( i, max( 0, j.sub( 1 ) ) ) );
			const up = atomicLoad( cellParticleCount( i, min( resolutionY - 1, j.add( 1 ) ) ) );

			return left.greaterThanEqual( minParticlesPerCellNode ).select( int( 1 ), int( 0 ) )
				.add( right.greaterThanEqual( minParticlesPerCellNode ).select( int( 1 ), int( 0 ) ) )
				.add( down.greaterThanEqual( minParticlesPerCellNode ).select( int( 1 ), int( 0 ) ) )
				.add( up.greaterThanEqual( minParticlesPerCellNode ).select( int( 1 ), int( 0 ) ) );

		}

		const MIN_FLUID_NEIGHBORS = 2;

		// One thread per cell -- claims up to (minParticlesPerCell - count)
		// donors via a bounded Loop with early Break, same idiom
		// advection_solver2.js's own backTrace already established.
		// donorPushCursor's own live value is read directly via atomicLoad
		// inside this kernel -- no CPU readback needed, it's a live atomic
		// buffer, not a value this function needs to branch on in JS.
		const claimDonorsKernel = tsl_array_n.kernel( [ resolutionX, resolutionY ], ( i, j ) => {

			const count = atomicLoad( cellParticleCount( i, j ) );

			If( count.lessThan( minParticlesPerCellNode ).and( neighborFluidCount( i, j ).greaterThanEqual( MIN_FLUID_NEIGHBORS ) ), () => {

				const donorCount = atomicLoad( donorPushCursor() );
				const needed = minParticlesPerCellNode.sub( count );
				const claimed = int( 0 ).toVar();

				Loop( minParticlesPerCell, () => {

					If( claimed.greaterThanEqual( needed ), () => {

						Break();

					} );

					const claimSlot = atomicAdd( donorPopCursor(), 1 );

					If( claimSlot.lessThan( donorCount ), () => {

						const donorIdx = donorPool( claimSlot );
						const newPos = cellCenterOrigin.add( vec2( i, j ).mul( gridSpacingNode ) );

						positions( donorIdx ).assign( newPos );
						velocities( donorIdx ).assign( faceCenteredValueAtPosition2(
							velocityGrid.dataU, velocityGrid.dataV, velocityGrid.gridSpacing,
							velocityGrid.dataOriginU, velocityGrid.dataOriginV, newPos, dataSizeU, dataSizeV
						) );

						claimed.addAssign( 1 );

					} ).Else( () => {

						// No more donors available this frame -- matches
						// mantaflow's own adjustNumber, which reseeds from a
						// level set rather than manufacturing particles from
						// nothing; this cell just stays under-full a bit
						// longer, corrected on a later frame instead.
						Break();

					} );

				} );

			} );

		} );

		resamplePass = function resample() {

			resetResampleBuffers();
			countPerCellKernel();
			buildDonorPoolKernel();
			claimDonorsKernel();

			// Relocation changed particle positions -- the fluidMask/
			// uFluidAdjacent/vFluidAdjacent computed just before this pass
			// are now stale for the recipient cells; recompute so this same
			// frame's pressure solve sees the corrected distribution.
			clearFluidMask();
			markFluidCellsKernel();
			computeUFluidAdjacent();
			computeVFluidAdjacent();

		};

	}

	// ---- gravity: mantaflow's own addGravity, this port's own established
	// force-application shape (a plain self-touch addAssign, matching
	// external_force_solver2.js's own applyExternalForces). ----

	const dtNode = numberOrNode( dt );
	const gravityNode = vec2( gravity[ 0 ], gravity[ 1 ] );

	const applyGravityU = tsl_array_n.kernel( dataSizeU, ( i, j ) => {

		velocityGrid.dataU( i, j ).addAssign( gravityNode.x.mul( dtNode ) );

	} );

	const applyGravityV = tsl_array_n.kernel( dataSizeV, ( i, j ) => {

		velocityGrid.dataV( i, j ).addAssign( gravityNode.y.mul( dtNode ) );

	} );

	// ---- velOld snapshot + G2P (FLIP/PIC blend): mantaflow's own
	// mapPartsToMAC copies velOld right after P2G (before extrapolation/
	// gravity/pressure ever touch it), and flipVelocityUpdate later reads
	// both this and the post-pressure velocityGrid at each particle's own
	// position -- delta = new-old is what actually carries the pressure
	// solve's own correction back onto each particle, matching FLIP's
	// whole point (transport the grid's own *change*, not its raw value,
	// so each particle keeps its own accumulated momentum/noise instead of
	// being smoothed toward the grid every frame the way pure PIC would). ----

	const oldDataU = tsl_array_n.arrayN( 'float', dataSizeU );
	const oldDataV = tsl_array_n.arrayN( 'float', dataSizeV );
	oldDataU.fromArray( new Float32Array( uCount ) );
	oldDataV.fromArray( new Float32Array( vCount ) );

	const snapshotOldU = createCopyKernel2( velocityGrid.dataU, oldDataU, dataSizeU );
	const snapshotOldV = createCopyKernel2( velocityGrid.dataV, oldDataV, dataSizeV );

	const flipRatioNode = numberOrNode( flipRatio );
	// clamp(...,0,1): 0 keeps the FLIP-blended velocity untouched (matches
	// mantaflow's own reference exactly), 1 resets it to zero every frame
	// (maximum sane damping -- fully overdamped/frozen-looking, but still a
	// clean "fraction of velocity removed" reading). Negative values would
	// *amplify* velocity every frame instead of damping it -- a genuine
	// divergence risk -- so this clamp is enforced here unconditionally,
	// regardless of what a caller (including a live UI slider bound to this
	// option, see examples/20-flip-dam-break/'s own control panel) passes in.
	const velocityDampingNode = clamp( numberOrNode( velocityDamping ), float( 0 ), float( 1 ) );

	function clampParticleVelocity( v ) {

		const isNaN = v.notEqual( v );
		return isNaN.select( vec2( 0 ), clamp( v, vec2( - MAX_PARTICLE_VELOCITY ), vec2( MAX_PARTICLE_VELOCITY ) ) );

	}

	const g2pUpdate = tsl_array_n.kernel( maxParticles, ( p ) => {

		const pos = positions( p );
		const newVel = faceCenteredValueAtPosition2( velocityGrid.dataU, velocityGrid.dataV, velocityGrid.gridSpacing, velocityGrid.dataOriginU, velocityGrid.dataOriginV, pos, dataSizeU, dataSizeV );
		const oldVel = faceCenteredValueAtPosition2( oldDataU, oldDataV, velocityGrid.gridSpacing, velocityGrid.dataOriginU, velocityGrid.dataOriginV, pos, dataSizeU, dataSizeV );

		const delta = newVel.sub( oldVel );
		const flipVel = velocities( p ).add( delta );
		const blended = flipVel.mul( flipRatioNode ).add( newVel.mul( float( 1 ).sub( flipRatioNode ) ) );
		const damped = blended.mul( float( 1 ).sub( velocityDampingNode ) );

		velocities( p ).assign( clampParticleVelocity( damped ) );

	} );

	// ---- particle advection: reuses advection_solver2.js's own already
	// real-hardware-validated trace(pos,-1) -- see this file's own header
	// comment. Domain-edge handling is a clamp (matching this port's own
	// established "clamp, don't lose data" pattern,
	// MAX_VELOCITY_COMPONENT/EXTRAPOLATED_VELOCITY_CLAMP) -- with a fixed
	// particle count and no resampling, deleting a particle isn't even a
	// coherent option in this design. ----

	const advectionSolver = createSemiLagrangianAdvectionSolver2( { velocityGrid, collider, dt } );

	const CLAMP_EPSILON = 1e-4;
	const minPos = originNode.add( vec2( CLAMP_EPSILON ) );
	const maxPos = originNode.add( vec2( resolutionX * gridSpacingX, resolutionY * gridSpacingY ) ).sub( vec2( CLAMP_EPSILON ) );

	const advectParticles = tsl_array_n.kernel( maxParticles, ( p ) => {

		const traced = advectionSolver.trace( positions( p ), -1 );
		positions( p ).assign( clamp( traced, minPos, maxPos ) );

	} );

	// ---- particle push-out: ported from mantaflow's own pushOutofObs/
	// knPushOutofObs -- see this file's own header comment for the full
	// derivation and the exact upstream formula this mirrors. Only built
	// when a real collider is present (collider.sample/gradient don't
	// exist to call otherwise), matching grid_blocked_boundary_condition_
	// solver2.js's own rebuildColliderKernels()'s null-collider early exit. ----

	const pushThreshNode = numberOrNode( colliderPushThresh );
	const pushShiftNode = numberOrNode( colliderPushShift );

	const pushOutOfCollider = collider ? tsl_array_n.kernel( maxParticles, ( p ) => {

		const pos = positions( p );
		const v = collider.sample( pos );

		If( v.lessThan( pushThreshNode ), () => {

			const g = collider.gradient( pos );

			If( g.length().greaterThan( 0 ), () => {

				positions( p ).assign( pos.add( g.normalize().mul( pushThreshNode.sub( v ).add( pushShiftNode ) ) ) );

			} );

		} );

	} ) : null;

	// solver.onAdvanceTimeStep() (the caller's own responsibility, this
	// solver takes no arguments the way every other solver in this port's
	// own onAdvanceTimeStep() doesn't either) -- stage order matches
	// mantaflow's own scenes/flip01_simple.py exactly: advect particles ->
	// P2G -> snapshot velOld -> extrapolate (weight) -> mark fluid cells ->
	// gravity -> pressure -> extrapolate (fluid) -> G2P. boundarySolver.
	// constrainVelocity() calls added beyond mantaflow's own more minimal
	// wall-BC placement -- matching this port's own established "constrain
	// after every stage that touches velocity" discipline
	// (grid_solver2.js's own defaultCompute* stages already do this).
	// pushOutOfCollider() (when a collider is present) runs right after
	// advection, before P2G -- see this file's own header comment.
	// resamplePass() (when resampling is enabled) runs right after fluid-
	// cell bookkeeping, before gravity -- mirrors mantaflow's own
	// adjustNumber placement; see this file's own header comment (the
	// "Particle resampling" section) for the full design.
	async function onAdvanceTimeStep() {

		advectParticles();
		if ( pushOutOfCollider ) pushOutOfCollider();

		resetAccumulators();
		scatterU();
		scatterV();
		finalizeU();
		finalizeV();

		snapshotOldU();
		snapshotOldV();

		extrapolateWeightU();
		extrapolateWeightV();
		boundarySolver.constrainVelocity();

		clearFluidMask();
		markFluidCellsKernel();
		computeUFluidAdjacent();
		computeVFluidAdjacent();

		if ( resamplePass ) resamplePass();

		applyGravityU();
		applyGravityV();
		boundarySolver.constrainVelocity();

		await projectDispatch();
		boundarySolver.constrainVelocity();

		extrapolatePressureU();
		extrapolatePressureV();

		g2pUpdate();

	}

	return {
		onAdvanceTimeStep,
		positions, velocities, fluidMask,
		pressure: pressureSolver.pressure,
		boundarySolver, pressureSolver
	};

}
