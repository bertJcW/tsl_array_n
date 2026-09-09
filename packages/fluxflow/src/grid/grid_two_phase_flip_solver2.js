// A 2D two-component FLIP solver: two fluids sharing one velocity field,
// coupled through a variable-density pressure projection, with a continuous
// concentration carried on each particle.
//
// It covers two quite different-looking cases with the same machinery, and it
// is worth knowing which one you are reading about:
//
//   * **Two phases (liquid + gas).** An extreme density ratio and a
//     concentration that only ever takes the values 0 or 1. This is what the
//     file was built for and what most of the comments below discuss; it is
//     this port's first solver in which the *air* is simulated too, rather
//     than being a void the liquid happens to move through.
//   * **Miscible mixing (dye in water, two liquids).** A mild or zero density
//     ratio and a concentration that varies continuously and is allowed to
//     blend. See "Miscible mixing", below. This case is much better behaved
//     numerically -- a mild density ratio keeps the pressure system
//     well-conditioned, which is exactly what the extreme ratio does not.
//
// Set `mixing`/`fade` and a mild `ambientDensity`/`componentDensity` pair for
// the second; leave them at their defaults for the first.
//
// ============================================================
// What "two-phase" changes, and why it needs a new file
// ============================================================
//
// grid_flip_solver2.js (which this file otherwise follows closely -- read
// that file first, most of the particle machinery here is the same idea)
// is single-phase: particles ARE the liquid, and every cell without a
// particle in it is "air", modelled as a Dirichlet `p = 0` region with no
// dynamics whatsoever. That is the standard free-surface simplification,
// and it is a very good one for most liquid shots -- but it means air can
// never push back. No rising bubble, no pocket of air trapped under a
// breaking wave, no air-driven splash: the air isn't there to do any of it.
//
// Two-phase treats liquid and gas as two incompressible fluids sharing one
// velocity field, coupled through a **variable-density pressure
// projection**. The single most important consequence, and the thing worth
// checking first if this ever looks wrong: **buoyancy is not a force term
// anywhere in this file.** Nothing adds an upward push to the gas. Gravity
// is applied uniformly to every face, exactly as in the single-phase
// solver, and the bubble rises purely because the projection that follows
// knows the gas is lighter. If you go looking for a buoyancy coefficient
// to tune, there isn't one, and adding one would be double-counting.
//
// ============================================================
// Where the algorithm comes from
// ============================================================
//
// See ../../THIRD-PARTY-NOTICES.md for the full attribution block. In
// short, and unusually for this port: **there was no open-source two-phase
// solver to port from.** Both of this project's existing upstreams were
// checked directly rather than assumed:
//
//   * **mantaflow** (Apache-2.0) does have a ghost-fluid pressure path --
//     `ghostFluidHelper` / `ApplyGhostFluidDiagonal` /
//     `knCorrectVelocityGhostFluid` in source/plugin/pressure.cpp -- but
//     reading it shows it is the *free-surface* ghost fluid method
//     (liquid vs. `isEmpty`, placing `p = 0` at a sub-cell position
//     interpolated from a level set). There is no second phase with its
//     own density anywhere in it. Worth recording so nobody re-checks:
//     "mantaflow has GFM" is true and is NOT the thing this file needs.
//   * **jet/fluid-engine-dev** (MIT) has `GridSinglePhasePressureSolver2`
//     and `GridFractionalSinglePhasePressureSolver2` -- single-phase, as
//     both names say. Nothing to take.
//
// So the algorithm here comes from papers, and the code is this port's
// own:
//
//   * Brackbill & Ruppel 1986, "FLIP: A method for adaptively zoned,
//     particle-in-cell calculations of fluid flows in two dimensions",
//     J. Comput. Phys. 65(2) -- FLIP itself.
//   * Zhu & Bridson 2005, "Animating Sand as a Fluid", ACM TOG 24(3) --
//     FLIP for incompressible flow, and the PIC/FLIP blend `flipRatio`
//     interpolates between.
//   * Kang, Fedkiw & Liu 2000, "A Boundary Condition Capturing Method for
//     Multiphase Incompressible Flow", J. Sci. Comput. 15(3) -- the
//     variable-density pressure Poisson formulation this file solves.
//   * Hong & Kim 2005, "Discontinuous Fluids", ACM TOG 24(3)
//     (SIGGRAPH 2005) -- the first graphics use of that formulation for
//     two-phase liquid/gas with bubbles.
//   * **Boyd & Bridson 2012, "MultiFLIP for Energetic Two-Phase Fluid
//     Simulation", ACM TOG 31(2)** -- the closest reference to what this
//     file is: two-phase FLIP with one phase bit per particle. See
//     "What this deliberately does NOT do", below, for the parts of
//     MultiFLIP this first version leaves out.
//   * Bridson, "Fluid Simulation for Computer Graphics", 2nd ed.,
//     ch. "Variable Density Solves" -- the discrete face-averaged-density
//     stencil used here rather than a sub-cell ghost-fluid one.
//
// ============================================================
// The pressure equation
// ============================================================
//
// grid_pressure_solver2.js's existing convention folds `dt/rho` into `p`,
// giving `Laplacian(p) = div(u*)`, `u = u* - grad(p)`. With a density that
// varies in space that generalizes to
//
//     div( beta grad(p) ) = div(u*)          u = u* - beta grad(p)
//
// where `beta = rho_liquid / rho`, so beta is exactly 1 in the liquid and
// `rho_liquid/rho_gas` (large) in the gas. Normalizing on the *liquid*
// rather than the gas is deliberate and not arbitrary: it keeps beta = 1
// through the bulk of a typical scene, which is what the multigrid
// preconditioner -- still constant-coefficient, see below -- implicitly
// assumes, so the preconditioner is accurate exactly where most of the
// domain is.
//
// beta lives on **faces**, not cells: `betaU(i,j)` is the lower-x face of
// cell (i,j), `betaV(i,j)` the lower-y face, which is precisely MAC
// indexing (`dataSizeU = [resX+1, resY]`) and precisely what multigrid.js's
// own `faceWeights[axis]` parameter is defined to mean. Storing it per face
// is what keeps the operator symmetric: the coupling between two adjacent
// cells is one shared array element read from both sides, not two averages
// that merely ought to agree. PCG needs that symmetry -- multigrid.js's own
// header comment records what happened here once when an operator wasn't
// symmetric, and it was not a graceful failure.
//
// Face density is the plain arithmetic average of the two adjacent cell
// densities (Bridson's variable-density stencil), and cell density is
// itself linear in the cell's own liquid particle fraction:
//
//     rho_cell = rho_gas + (rho_liquid - rho_gas) * liquidFraction
//     rho_face = (rho_A + rho_B) / 2
//     beta_face = rho_liquid / rho_face
//
// A *sharp* (fraction thresholded at 0.5) density would be closer to the
// ghost-fluid papers above; the smooth version is used here because the
// interface is only ever known to particle-sampling accuracy anyway (there
// is no level set in this port), and a smoothly-varying beta is markedly
// kinder to the constant-coefficient preconditioner. This is a graphics
// tradeoff, chosen knowingly, not an oversight.
//
// ============================================================
// The singular system, and mantaflow's answer to it
// ============================================================
//
// *** Read this before removing `pressurePin`. ***
//
// The single-phase solver always had a Dirichlet region for free: the air.
// Once the air is a simulated phase, a closed domain has **no Dirichlet
// cell anywhere** -- the pressure system becomes pure Neumann, and pure
// Neumann is singular: p and p + c are equally valid solutions, so the
// null-space component is free to wander. It does not diverge in one
// dramatic frame the way a sign error does; it drifts, quietly, until it
// trips grid_pressure_solver2.js's own `maxPlausiblePressure` circuit
// breaker and the solve starts getting silently rejected every frame.
//
// mantaflow hits exactly this case and handles it in
// `solvePressureSystem`: when `CountEmptyCells(flags) == 0` -- literally
// "there is no air region" -- it calls `fixPressure(fixPidx, 0, ...)`,
// which rewrites one cell's row to the identity and pins it to zero. Its
// preferred cell is `(sizeX/2, sizeY-1)`, top centre, walking two cells
// down and then anywhere if that one isn't usable.
//
// That is ported here directly, and it needed no new machinery at all:
// this port's existing `dirichlet` option on grid_pressure_solver2.js
// already means "identity row, this target value", which *is*
// `fixPressure`. `pressurePin` defaults to mantaflow's own top-centre
// position. mantaflow's fallback walk isn't ported -- there is no
// "unusable" cell to walk away from here, since in two-phase every cell is
// fluid by construction, which is the whole reason we are in this branch.
//
// mantaflow's other option for the same problem, `enforceCompatibility`
// (subtract the mean divergence from the RHS), is deliberately not ported:
// it needs a whole-grid reduction per frame, and this port's reductions go
// through linalg.js's fixed-point atomic accumulator, whose scale is
// exactly the kind of per-scene-tuned quantity this project has repeatedly
// learned not to add without a specific need. Pinning one cell costs
// nothing and solves the same problem.
//
// ============================================================
// pressure.atomicScale: the one option you MUST set per scene
// ============================================================
//
// Found the hard way on the first real-WebGPU run, and worth reading before
// filing a bug about this solver exploding.
//
// linalg.js accumulates every CG dot product as a fixed-point integer scaled
// by `atomicScale`; too large a scale for a scene's actual magnitudes
// overflows the accumulator and corrupts the whole reduction. This port has
// hit that before (see examples/20-flip-dam-break/'s own header comment),
// but two-phase makes it structural rather than incidental: `Ap` here
// carries a factor of beta, and beta IS the density ratio, so the safe scale
// shrinks roughly in proportion to it. The library default
// (DEFAULT_ATOMIC_DOT_SCALE, 65536) is wrong here by about that factor.
//
// Measured, on real hardware, same 32x32 scene, at a 100:1 ratio:
//
//     atomicScale: 256  ->  pressure 1023/1024 cells non-finite by frame 10
//     atomicScale: 16   ->  non-finite by frame 19
//     atomicScale: 1    ->  stable, pressure bounded ~3.5, velocities ~4
//
// So: start at roughly DEFAULT_ATOMIC_DOT_SCALE divided by the density
// ratio, and go down from there. This is deliberately NOT baked in as a
// changed default, following the same reasoning
// grid_pressure_solver2.js's own header comment records for
// maxPlausiblePressure: a reduction magnitude tuned against one scene does
// not reliably transfer to a differently-scaled one, and a wrong shared
// default is worse than an explicit per-scene one. `numberOfLevels` IS
// defaulted differently here (see below) because that one has no such
// scene-dependence -- a deeper V-cycle is simply a better preconditioner.
//
// ============================================================
// Miscible mixing: dye in water, and two liquids
// ============================================================
//
// The two-phase reading above treats concentration as a phase *tag*: a
// particle is liquid or it is gas, and the two never blend. Nothing in the
// solver actually requires that. Concentration is a continuous [0,1] value
// carried per particle, the cell density is a linear blend across it, and the
// projection never sees anything but the resulting density -- so a
// concentration of 0.3 is as meaningful as 0 or 1, and always was.
//
// What miscible mixing adds is the ability for that value to *change*:
//
//   * `mixing` blends each particle's concentration toward the mean of the
//     fluid immediately around it, so a sharp dye boundary gradually softens
//     and a stirred scene eventually goes uniform.
//   * `fade` decays concentration toward zero, so the dye disappears instead
//     of blending in.
//
// Both default to 0 and cost nothing when off (the pass is not even
// dispatched). Both are documented in full where they are implemented.
//
// **Set `ambientDensity`/`componentDensity` to the same value and you get a
// passive tracer**: the dye is carried by the flow and rendered, but has no
// dynamical effect at all, because the density field is then uniform and beta
// is 1 everywhere -- the projection reduces exactly to the constant-density
// one. Give them a small difference instead (say 1.0 and 1.05) and the dye
// starts to drive the flow: a heavier dye sinks and plumes, a lighter one
// rises, entirely through the same variable-density mechanism the gas bubble
// uses, just gently. That difference is the single most useful dial in a dye
// scene, and it is a real physical parameter rather than a look control.
//
// **What this is not.** `mixing` is a per-frame convex blend, not a
// discretised diffusion coefficient: it is frame-rate dependent and it is not
// the Fickian diffusion a physical miscible-fluid model would solve. The
// literature's proper treatment of genuinely mixing fluids models a drift
// velocity per component -- see Ren et al., "Multiple-fluid SPH Simulation
// Using a Mixture Model" (ACM TOG 33(6), SIGGRAPH Asia 2014) and its follow-up
// Yang et al., "Fast Multiple-fluid Simulation Using Helmholtz Free Energy"
// (ACM TOG 34(6), 2015), surveyed in Liu et al., "Visual Simulation of
// Multiple Fluids in Computer Graphics: A State-of-the-Art Report" (JCST
// 33(3), 2018). Those are SPH formulations, so they are a design reference
// here rather than something to port; a drift-velocity mixture model is the
// upgrade path if the phenomenological blend is ever not enough. See
// ../../THIRD-PARTY-NOTICES.md.
//
// **Carried on the particle, not advected on the grid** -- a deliberate choice
// with a real consequence. A grid-advected scalar picks up numerical diffusion
// from every semi-Lagrangian lookup, which for a dye filament means it smears
// out whether or not you asked it to. A particle simply carries its value
// exactly, so the only blending that happens is the blending you configured.
// This is also what Houdini does (dye is a per-particle `Cd` attribute rather
// than a solver), which is where the confirmation for this design came from --
// see THIRD-PARTY-NOTICES.md, "Design and architecture references".
//
// ============================================================
// What this deliberately does NOT do (first version)
// ============================================================
//
// * **No separate per-phase velocity fields.** MultiFLIP's central trick is
//   two loosely-coupled velocity fields, so liquid momentum near the
//   interface doesn't bleed into the air and vice versa. This uses one
//   shared field, which is simpler, cheaper, and does let some momentum
//   cross the interface. It is the most likely thing to want next.
// * **No level set, so no sub-cell ghost fluid and no surface tension.**
//   The interface is known only as accurately as the particles sample it.
//   Surface tension in Hong & Kim / mantaflow both need an interface
//   curvature this port has no way to compute yet.
// * **No particle separation at the interface.** MultiFLIP explicitly
//   adjusts particle positions to stop the phases inter-penetrating.
//   Per-phase resampling (below) helps, but it is not the same mechanism.
// * **Constant-coefficient multigrid preconditioning** (multigrid.js
//   decision 4): the V-cycle preconditions the variable-coefficient system
//   with a constant-coefficient approximation. Still a valid preconditioner
//   -- PCG requires symmetric-positive-definite, NOT accurate -- but a
//   worse one the larger the density ratio, so iteration counts climb with
//   it. This is the direct reason `gasDensity` defaults to 0.01 (a 100:1
//   ratio) rather than real air/water's ~1:816.
//
// ============================================================
// Particle resampling: why the single-phase pass could not be reused
// ============================================================
//
// grid_flip_solver2.js's resampler relocates a particle out of an over-full
// cell into an under-full one. Reused verbatim here it would be a genuine
// physics bug, not a quality issue: it would happily relocate a *liquid*
// particle into a cell that is under-full because it is *gas*, teleporting
// mass straight across the interface and changing both cells' densities.
//
// So the pass here keeps the single-phase version's over/under detection on
// the **total** particle count (that part is about sampling density and is
// phase-agnostic), but splits the donor pool by phase and gives each
// under-full cell a donor of the phase that cell should be getting -- its
// own majority phase, or, if it has no particles at all to have a majority
// of, its neighbors' majority. Everything else -- the atomicAdd-return-value
// slot claiming, the bounded `Loop()`/`Break()` donor claiming, the
// requirement that a recipient have at least 2 well-populated orthogonal
// neighbors before it is eligible (which exists to stop isolated specks
// being reinforced into a growing blob) -- is the same mechanism, and
// grid_flip_solver2.js's own header comment is the reference for all of it.

import * as tsl_array_n from 'tsl_array_n';
import { vec2, float, int, floor, round, clamp, max, min, atomicAdd, atomicLoad, If, Loop, Break } from 'three/tsl';
import { createGridBlockedBoundaryConditionSolver2 } from './grid_blocked_boundary_condition_solver2.js';
import { createGridPressureSolver2 } from './grid_pressure_solver2.js';
import { createSemiLagrangianAdvectionSolver2 } from './advection_solver2.js';
import { createCopyKernel2, createExtrapolateToRegion2 } from './array_utils.js';
import { bilinearCoordsAndWeights2, collocatedValueAtPosition2, faceCenteredValueAtPosition2 } from './grid_math.js';
import { DEFAULT_ATOMIC_DOT_SCALE } from '../linalg/linalg.js';

// Same last-resort particle speed bound as the single-phase solver. It
// matters more here: a gas face's own pressure correction is scaled by
// beta, which is the density ratio (100x by default), so the light phase
// genuinely does accelerate far harder than anything the single-phase
// solver ever produced. That is correct physics, but it puts the gas much
// closer to a CFL violation, and this clamp is what keeps one bad frame
// from becoming a permanent one.
const MAX_PARTICLE_VELOCITY = 500;

// Fixed-point scale for accumulating a *continuous* per-particle concentration
// into an integer atomic (see the counting kernel). Unlike the CG solver's own
// atomicScale this one needs no per-scene tuning and is safe by construction:
// concentration is bounded to [0,1] and a cell holds a handful of particles, so
// the worst-case accumulated integer is a few times this value -- five orders
// of magnitude clear of int32. It is an option only so a caller who wants finer
// quantisation than 1/4096 can ask for it.
const DEFAULT_CONCENTRATION_ATOMIC_SCALE = 4096;

// Absolute floor on either component's density. Densities in this port's scenes
// are O(1), so this never binds in practice -- it exists only so that a zero or
// negative value reaching the kernel cannot produce an infinite or
// sign-flipped beta. Deliberately absolute rather than a ratio of the other
// density, so it means the same thing whichever component is heavier.
const MIN_DENSITY = 1e-4;

// Phase tag values. Stored as float rather than int purely so the particle
// arrays stay uniform with positions/velocities and can be read back with
// one toArray() in a renderer; only the >0.5 test is ever used.
export const PHASE_GAS = 0;
export const PHASE_LIQUID = 1;

function numberOrNode( value ) {

	return typeof value === 'number' ? float( value ) : value;

}

// Deepest V-cycle this resolution can actually support, capped at 4.
// multigrid.js's computeLevelShapes throws unless every axis divides evenly
// by 2^(levels-1), so this walks up only as far as that allows -- a fixed
// default would make the solver reject ordinary grid sizes outright.
const MAX_DEFAULT_MULTIGRID_LEVELS = 4;

function defaultMultigridLevels( shape ) {

	let levels = 1;

	while ( levels < MAX_DEFAULT_MULTIGRID_LEVELS && shape.every( ( n ) => n % ( 2 ** levels ) === 0 ) ) {

		levels ++;

	}

	return levels;

}

// Fills an axis-aligned box with particles at a fixed per-cell density and
// tags each one's phase from a plain JS predicate -- so a caller writes
// `isLiquid: ( [ x, y ] ) => y < waterLine` (or a circle test for a bubble,
// or both) and gets a ready two-phase initial condition.
//
// The important difference from computeFlipBoxSeed: for two-phase the box
// is normally the **whole domain**, because both phases have to actually
// exist to interact. Particle count therefore scales with the entire grid,
// not with the liquid's own footprint -- a 64x64 domain at the default
// 2x2 per cell is 16384 particles, where the equivalent single-phase
// dam-break scene needed ~4000.
export function computeTwoPhaseBoxSeed( {
	boxMin, boxMax,
	gridSpacingX, gridSpacingY,
	particlesPerCellAxis = 2,
	jitter = 0.2,
	isLiquid = () => true,
	// Continuous form of `isLiquid`, and the one to use for a dye scene: return
	// a concentration in [0,1] rather than a boolean. Takes precedence when
	// given. `isLiquid` remains for the binary two-phase case, where it is the
	// clearer thing to write.
	concentrationAt
} = {} ) {

	const [ minX, minY ] = boxMin;
	const [ maxX, maxY ] = boxMax;

	const cellsX = Math.max( 1, Math.round( ( maxX - minX ) / gridSpacingX ) );
	const cellsY = Math.max( 1, Math.round( ( maxY - minY ) / gridSpacingY ) );

	const subSpacingX = gridSpacingX / particlesPerCellAxis;
	const subSpacingY = gridSpacingY / particlesPerCellAxis;

	const positionsList = [];
	const phasesList = [];

	for ( let cellJ = 0; cellJ < cellsY; cellJ ++ ) {

		for ( let cellI = 0; cellI < cellsX; cellI ++ ) {

			for ( let subJ = 0; subJ < particlesPerCellAxis; subJ ++ ) {

				for ( let subI = 0; subI < particlesPerCellAxis; subI ++ ) {

					const jx = ( Math.random() * 2 - 1 ) * jitter * subSpacingX;
					const jy = ( Math.random() * 2 - 1 ) * jitter * subSpacingY;

					const x = minX + cellI * gridSpacingX + ( subI + 0.5 ) * subSpacingX + jx;
					const y = minY + cellJ * gridSpacingY + ( subJ + 0.5 ) * subSpacingY + jy;

					positionsList.push( x, y );
					phasesList.push( concentrationAt
						? Math.min( 1, Math.max( 0, concentrationAt( [ x, y ] ) ) )
						: ( isLiquid( [ x, y ] ) ? PHASE_LIQUID : PHASE_GAS ) );

				}

			}

		}

	}

	const count = positionsList.length / 2;
	const phasesArray = Float32Array.from( phasesList );
	let liquidCount = 0, concentrationSum = 0;
	for ( const p of phasesArray ) {

		if ( p > 0.5 ) liquidCount ++;
		concentrationSum += p;

	}

	return {
		count,
		positionsArray: new Float32Array( positionsList ),
		velocitiesArray: new Float32Array( count * 2 ), // zeroed -- particles start at rest
		phasesArray,
		// liquidCount/gasCount are the binary reading and stay exact for a
		// two-phase seed; meanConcentration is the one that stays meaningful
		// when the seed is continuous.
		liquidCount,
		gasCount: count - liquidCount,
		meanConcentration: count > 0 ? concentrationSum / count : 0
	};

}

export function createGridTwoPhaseFlipSolver2( {
	velocityGrid,
	gridSpacing = [ 1, 1 ],
	origin = [ 0, 0 ],
	maxParticles,
	dt,
	gravity = [ 0, - 9.81 ],
	liquidDensity = 1,
	gasDensity = 0.01,
	// Neutral aliases for the two densities, preferred for anything that is not
	// literally liquid-and-gas. `concentration` runs 0 -> 1 across the two
	// components, so `ambientDensity` is the density at concentration 0 and
	// `componentDensity` the density at concentration 1. For a dye scene those
	// read as "clear water" and "dyed water"; for the original gas/liquid case
	// they are exactly gasDensity and liquidDensity. Same fields either way.
	ambientDensity,
	componentDensity,
	// Miscible-mixing controls, both no-ops at their defaults so every existing
	// caller is unaffected. See "Miscible mixing" in the header comment.
	mixing = 0,
	fade = 0,
	concentrationAtomicScale = DEFAULT_CONCENTRATION_ATOMIC_SCALE,
	flipRatio = 0.97,
	gasFlipRatio = 0.90,
	velocityDamping = 0.02,
	p2gAtomicScale = DEFAULT_ATOMIC_DOT_SCALE,
	weightEpsilon = 1e-4,
	closedDomainBoundaryFlag,
	collider,
	colliderPushThresh = 0,
	colliderPushShift = 0,
	pressurePin = 'topCenter',
	resample = {},
	pressure = {}
} = {} ) {

	if ( ambientDensity !== undefined ) gasDensity = ambientDensity;
	if ( componentDensity !== undefined ) liquidDensity = componentDensity;

	if ( ! velocityGrid ) {

		throw new Error( 'createGridTwoPhaseFlipSolver2: options.velocityGrid is required.' );

	}

	if ( ! maxParticles ) {

		throw new Error( 'createGridTwoPhaseFlipSolver2: options.maxParticles is required.' );

	}

	// Both densities may be a plain number or a live node. Only a number can be
	// range-checked here; a node's invariants are enforced in the kernel graph
	// instead (see where the density nodes are built), which is the stronger
	// place for them anyway.
	if ( typeof liquidDensity === 'number' && ! ( liquidDensity > 0 ) ) {

		throw new Error( `createGridTwoPhaseFlipSolver2: the concentration-1 density (liquidDensity / componentDensity) must be > 0, got ${ liquidDensity }.` );

	}

	// gasDensity follows this port's own "number or node" convention (same as
	// dt, flipRatio, velocityDamping), so a scene can put it on a slider and
	// change the density ratio -- and therefore how hard the gas floats -- on
	// a running simulation. A node can't be range-checked in JS, so the
	// invariants are enforced in the kernel graph instead, below; that is the
	// stronger place for them regardless, and is the pattern
	// grid_flip_solver2.js's own velocityDamping clamp already established.
	if ( typeof gasDensity === 'number' && ! ( gasDensity > 0 ) ) {

		throw new Error( `createGridTwoPhaseFlipSolver2: the concentration-0 density (gasDensity / ambientDensity) must be > 0, got ${ gasDensity }.` );

	}

	// *** Ordering is required by the gas/liquid VOCABULARY, not by the solver. ***
	//
	// Under the two-phase reading, "gas" heavier than "liquid" means the two
	// tags are named backwards, which silently inverts every buoyancy result
	// instead of failing -- worth rejecting outright. But that is a fact about
	// the words, not about the mathematics: nothing downstream needs the
	// concentration-1 component to be the heavier one, and a dye *lighter* than
	// the water it is injected into is perfectly ordinary. So the check applies
	// only when the caller used the gas/liquid names; the neutral
	// ambientDensity/componentDensity pair carries no ordering claim and is not
	// checked.
	const usedPhaseVocabulary = ambientDensity === undefined && componentDensity === undefined;

	if ( usedPhaseVocabulary && typeof gasDensity === 'number' && typeof liquidDensity === 'number' && gasDensity > liquidDensity ) {

		throw new Error( `createGridTwoPhaseFlipSolver2: gasDensity (${ gasDensity }) must not exceed liquidDensity (${ liquidDensity }) -- the two phase tags would then be named backwards, which silently inverts every buoyancy result rather than failing. If you genuinely want the concentration-1 component to be the lighter one, use the neutral ambientDensity/componentDensity names instead, which carry no ordering claim.` );

	}

	const {
		enabled: resampleEnabled = true,
		minParticlesPerCell = 3,
		maxParticlesPerCell = 8
	} = resample;

	const [ resolutionX, resolutionY ] = velocityGrid.resolution;
	const [ gridSpacingX, gridSpacingY ] = gridSpacing;
	const [ originX, originY ] = origin;

	const gridSpacingNode = vec2( gridSpacingX, gridSpacingY );
	const originNode = vec2( originX, originY );

	const dataSizeU = velocityGrid.dataSizeU;
	const dataSizeV = velocityGrid.dataSizeV;
	const cellShape = [ resolutionX, resolutionY ];
	const cellCount = resolutionX * resolutionY;
	const uCount = dataSizeU[ 0 ] * dataSizeU[ 1 ];
	const vCount = dataSizeV[ 0 ] * dataSizeV[ 1 ];

	const boundarySolver = createGridBlockedBoundaryConditionSolver2(
		velocityGrid, resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY, collider
	);

	if ( closedDomainBoundaryFlag !== undefined ) boundarySolver.closedDomainBoundaryFlag = closedDomainBoundaryFlag;

	// ---------------------------------------------------------------- particles

	const positions = tsl_array_n.arrayN( 'vec2', maxParticles );
	const velocities = tsl_array_n.arrayN( 'vec2', maxParticles );
	const phase = tsl_array_n.arrayN( 'float', maxParticles );

	positions.fromArray( new Float32Array( maxParticles * 2 ) );
	velocities.fromArray( new Float32Array( maxParticles * 2 ) );
	phase.fromArray( new Float32Array( maxParticles ) ); // all gas until a caller seeds

	function cellIndexOf( pos ) {

		const cellF = pos.sub( originNode ).div( gridSpacingNode );
		const i = max( 0, min( int( floor( cellF.x ) ), resolutionX - 1 ) );
		const j = max( 0, min( int( floor( cellF.y ) ), resolutionY - 1 ) );
		return { i, j };

	}

	// `phase` is a *continuous* concentration in [0,1] -- 0 is the ambient
	// component, 1 the injected one. The binary liquid/gas case is just the
	// special case of only ever storing exactly 0 or 1.
	const concentrationOf = ( p ) => clamp( phase( p ), float( 0 ), float( 1 ) );

	// Used only where a genuinely binary decision is unavoidable: bucketing a
	// donor particle into one of the two resample pools. Everything physical
	// reads concentrationOf() instead.
	const isLiquidParticle = ( p ) => phase( p ).greaterThan( 0.5 );

	// ---------------------------------------------------------------- phase counting -> density -> beta

	// One counting pass per frame feeds BOTH the density field (which the
	// projection needs) and the resampler (which runs at the end of the same
	// frame). Particle positions don't change in between, so recounting for
	// the resampler would be pure duplicated work.
	// Accumulates the SUM OF CONCENTRATIONS in a cell, in fixed point -- not a
	// count of particles over a threshold. That difference is the whole of what
	// makes miscible mixing work: a cell that is half-way dyed has to be able to
	// report 0.5, and a thresholded count can only ever report a ratio of whole
	// particles. The binary phase case still falls out exactly, since a
	// concentration of exactly 0 or 1 sums to the same thing either way.
	const concentrationAccum = tsl_array_n.arrayN( 'int', cellShape );
	concentrationAccum.node.toAtomic();
	const totalCellCount = tsl_array_n.arrayN( 'int', cellShape );
	totalCellCount.node.toAtomic();

	const zeroCells = new Int32Array( cellCount );

	function resetCellCounts() {

		concentrationAccum.fromArray( zeroCells );
		totalCellCount.fromArray( zeroCells );

	}

	const concentrationScaleNode = float( concentrationAtomicScale );

	const countPhasesKernel = tsl_array_n.kernel( maxParticles, ( p ) => {

		const { i, j } = cellIndexOf( positions( p ) );
		atomicAdd( totalCellCount( i, j ), 1 );
		atomicAdd( concentrationAccum( i, j ), round( concentrationOf( p ).mul( concentrationScaleNode ) ).toInt() );

	} );

	const liquidFraction = tsl_array_n.arrayN( 'float', cellShape );
	const density = tsl_array_n.arrayN( 'float', cellShape );
	liquidFraction.fromArray( new Float32Array( cellCount ) );
	// Initial contents are irrelevant -- both fields are fully recomputed at
	// the top of every frame, before anything reads them. Filled with the
	// neutral (uniform-liquid) values anyway so a caller that reads them
	// before the first step gets something meaningful rather than zeros.
	density.fromArray( new Float32Array( cellCount ).fill( liquidDensity ) );

	// Both densities are floored in the kernel graph regardless of what any
	// caller passes in, and the floor is not cosmetic: a density of 0 makes
	// beta infinite and takes the whole pressure system with it, and a negative
	// one makes the operator indefinite, which CG does not converge slowly on
	// -- it breaks down. The floor is absolute rather than a ratio of the other
	// density, so it stays meaningful whichever of the two is larger.
	const minDensityNode = float( MIN_DENSITY );
	const densityAtZeroNode = max( numberOrNode( gasDensity ), minDensityNode );
	const densityAtOneNode = max( numberOrNode( liquidDensity ), minDensityNode );

	// beta = referenceDensity / rho, and the reference is the HEAVIER of the two
	// components rather than a fixed one of them. That keeps beta <= 1
	// everywhere and equal to 1 through the heavy bulk of a scene, which is
	// what the constant-coefficient multigrid preconditioner implicitly assumes
	// -- so the preconditioner stays accurate exactly where most of the domain
	// is, regardless of which component the caller happened to call "1".
	//
	// For the original gas/liquid case this is exactly the previous behaviour
	// (the liquid IS the max), so nothing about those scenes changes.
	const referenceDensityNode = max( densityAtZeroNode, densityAtOneNode );

	// Kept under the old name because a lot of the code below reads better with
	// it, and in the gas/liquid case it is the same value it always was.
	const liquidDensityNode = referenceDensityNode;
	const gasDensityNode = densityAtZeroNode;

	const computeDensityKernel = tsl_array_n.kernel( cellShape, ( i, j ) => {

		const total = atomicLoad( totalCellCount( i, j ) ).toFloat();
		const concentrationSum = atomicLoad( concentrationAccum( i, j ) ).toFloat().div( concentrationScaleNode );

		// A cell with no particles at all is treated as pure gas rather than
		// as "whatever it was last frame". With the whole domain seeded this
		// is rare and transient, but it does happen, and gas is the right
		// reading of a momentarily empty cell: an empty pocket should be
		// something the liquid can collapse into, not a heavy region that
		// shoves it away.
		// Clamped because the fixed-point round trip can land a hair outside
		// [0,1], and everything downstream (the density lerp, then beta, then
		// the operator's coefficients) assumes it is inside.
		const mean = concentrationSum.div( max( total, float( 1 ) ) );
		const fraction = total.greaterThan( 0.5 ).select( clamp( mean, float( 0 ), float( 1 ) ), float( 0 ) );

		liquidFraction( i, j ).assign( fraction );
		density( i, j ).assign( densityAtZeroNode.add( densityAtOneNode.sub( densityAtZeroNode ).mul( fraction ) ) );

	} );

	// beta on faces. Index clamping at the domain edge makes an edge face
	// average its one real neighbor with itself, i.e. just that cell's own
	// density -- the value is unused by the operator there (a domain-edge
	// term is masked to zero) but must still be finite, since TSL evaluates
	// both sides of a select.
	const betaU = tsl_array_n.arrayN( 'float', dataSizeU );
	const betaV = tsl_array_n.arrayN( 'float', dataSizeV );
	betaU.fromArray( new Float32Array( uCount ).fill( 1 ) );
	betaV.fromArray( new Float32Array( vCount ).fill( 1 ) );

	const clampI = ( i ) => max( 0, min( i, resolutionX - 1 ) );
	const clampJ = ( j ) => max( 0, min( j, resolutionY - 1 ) );

	const computeBetaU = tsl_array_n.kernel( dataSizeU, ( i, j ) => {

		const rhoFace = density( clampI( i.sub( 1 ) ), j ).add( density( clampI( i ), j ) ).mul( 0.5 );
		betaU( i, j ).assign( referenceDensityNode.div( max( rhoFace, minDensityNode ) ) );

	} );

	const computeBetaV = tsl_array_n.kernel( dataSizeV, ( i, j ) => {

		const rhoFace = density( i, clampJ( j.sub( 1 ) ) ).add( density( i, clampJ( j ) ) ).mul( 0.5 );
		betaV( i, j ).assign( referenceDensityNode.div( max( rhoFace, minDensityNode ) ) );

	} );

	// ---------------------------------------------------------------- pressure

	// mantaflow's own zeroPressureFixing position, ported -- see the header
	// comment. `pressurePin: null` opts out entirely, which leaves the
	// system singular; that is only ever correct if the caller has arranged
	// a Dirichlet region some other way (an open outflow, say).
	const pinCell = pressurePin === 'topCenter'
		? [ Math.floor( resolutionX / 2 ), resolutionY - 1 ]
		: pressurePin;

	if ( pinCell && ( ! Array.isArray( pinCell ) || pinCell.length !== 2 ) ) {

		throw new Error( `createGridTwoPhaseFlipSolver2: pressurePin must be 'topCenter', null, or an [i, j] cell index, got ${ JSON.stringify( pressurePin ) }.` );

	}

	let dirichlet;

	if ( pinCell ) {

		const [ pinI, pinJ ] = pinCell;

		dirichlet = function dirichletAt( pos ) {

			const cellF = pos.sub( originNode ).div( gridSpacingNode );
			const i = int( floor( cellF.x ) );
			const j = int( floor( cellF.y ) );

			return { active: i.equal( int( pinI ) ).and( j.equal( int( pinJ ) ) ), target: float( 0 ) };

		};

	}

	// *** A default that had to change after the first real-WebGPU run ***
	//
	// grid_pressure_solver2.js's own multigrid default is `numberOfLevels: 1`
	// -- which is plain red-black relaxation with no coarse-grid correction
	// at all, i.e. not really multigrid (this port's README has said as much
	// since examples/06). Every single-phase scene here gets away with it.
	// A variable-coefficient system does not: at a 100:1 density ratio,
	// measured on real hardware, `numberOfLevels: 1` produced a pressure
	// field that went 1023-cells-out-of-1024 non-finite within ten frames,
	// while the same scene at `numberOfLevels: 4` stayed bounded (pressure
	// peaking around 5) for the whole run and produced a cleanly rising
	// bubble. So this solver picks its own default rather than inheriting
	// that one.
	//
	// Derived from the resolution instead of hardcoded, because
	// computeLevelShapes throws unless every axis divides by 2^(levels-1) --
	// a hardcoded 4 would make this solver refuse perfectly ordinary grid
	// sizes. A caller's own `pressure.multigrid.numberOfLevels` still wins.
	const multigridOptions = { numberOfLevels: defaultMultigridLevels( cellShape ), ...( pressure.multigrid ?? {} ) };

	const pressureSolver = createGridPressureSolver2( {
		resolution: cellShape, gridSpacing, origin,
		dirichlet,
		faceWeights: { u: betaU, v: betaV },
		...pressure,
		multigrid: multigridOptions
	} );

	const projectDispatch = pressureSolver.project( velocityGrid, velocityGrid );

	// ---------------------------------------------------------------- P2G

	// Volume-weighted, NOT mass-weighted -- deliberately, and this is the
	// one place where "the obvious physical thing" is the wrong call. A
	// mass-weighted scatter at a 100:1 density ratio lets a single liquid
	// particle outvote every gas particle sharing an interface face, so the
	// gas near the interface simply inherits the liquid's velocity and stops
	// behaving like a separate phase at all. The density difference belongs
	// in the projection (where it is handled exactly), not smeared into the
	// transfer weights.
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

	const extrapolateWeightU = createExtrapolateToRegion2( velocityGrid.dataU, uWeightValid, velocityGrid.dataU, dataSizeU );
	const extrapolateWeightV = createExtrapolateToRegion2( velocityGrid.dataV, vWeightValid, velocityGrid.dataV, dataSizeV );

	// Note what is NOT here, compared to grid_flip_solver2.js: that solver
	// additionally extrapolates velocity into faces not adjacent to any
	// fluid cell, because its pressure solve leaves those faces meaningless.
	// Here every cell is fluid, so every face is solved for. The step has
	// nothing left to do and its absence is the point, not an omission.

	// ---------------------------------------------------------------- gravity

	const dtNode = numberOrNode( dt );
	const gravityNode = vec2( gravity[ 0 ], gravity[ 1 ] );

	// Uniform on every face, both phases, no density weighting anywhere.
	// See the header comment: this plus the variable-density projection IS
	// the buoyancy model.
	const applyGravityU = tsl_array_n.kernel( dataSizeU, ( i, j ) => {

		velocityGrid.dataU( i, j ).addAssign( gravityNode.x.mul( dtNode ) );

	} );

	const applyGravityV = tsl_array_n.kernel( dataSizeV, ( i, j ) => {

		velocityGrid.dataV( i, j ).addAssign( gravityNode.y.mul( dtNode ) );

	} );

	// ---------------------------------------------------------------- G2P

	const oldDataU = tsl_array_n.arrayN( 'float', dataSizeU );
	const oldDataV = tsl_array_n.arrayN( 'float', dataSizeV );
	oldDataU.fromArray( new Float32Array( uCount ) );
	oldDataV.fromArray( new Float32Array( vCount ) );

	const snapshotOldU = createCopyKernel2( velocityGrid.dataU, oldDataU, dataSizeU );
	const snapshotOldV = createCopyKernel2( velocityGrid.dataV, oldDataV, dataSizeV );

	const flipRatioNode = clamp( numberOrNode( flipRatio ), float( 0 ), float( 1 ) );
	const gasFlipRatioNode = clamp( numberOrNode( gasFlipRatio ), float( 0 ), float( 1 ) );
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

		// Per-phase blend. The gas defaults to a lower (more PIC) ratio for
		// a concrete reason rather than by taste: FLIP's characteristic
		// noise is velocity the particles carry that the grid never damps,
		// and the light phase both accelerates hardest under the projection
		// (its beta is the density ratio) and has the least inertia to
		// resist that noise. It is where FLIP misbehaves first, so it gets
		// the more dissipative end of the same blend the liquid uses.
		// Interpolated by concentration rather than switched on a threshold: with
		// a continuous concentration a hard switch would put a visible seam
		// through the middle of a mixing region, where nothing physical changes
		// abruptly. At concentration 0 or 1 this is identical to a switch.
		const c = concentrationOf( p );
		const ratio = gasFlipRatioNode.add( flipRatioNode.sub( gasFlipRatioNode ).mul( c ) );
		const blended = flipVel.mul( ratio ).add( newVel.mul( float( 1 ).sub( ratio ) ) );
		const damped = blended.mul( float( 1 ).sub( velocityDampingNode ) );

		velocities( p ).assign( clampParticleVelocity( damped ) );

	} );

	// ---------------------------------------------------------------- miscible mixing / fade

	// Two optional per-particle operations on the carried concentration, both
	// no-ops at their defaults. Together they are what turns this from a
	// two-phase solver into a miscible one: `mixing` lets a dye blend into what
	// surrounds it, `fade` lets it disappear.
	//
	// Deliberately a SEPARATE kernel rather than extra lines inside g2pUpdate,
	// for a concrete reason rather than tidiness: g2pUpdate already binds seven
	// storage buffers (positions, velocities, phase, and four velocity fields),
	// and the guaranteed per-stage limit is eight. Folding the concentration
	// field in there would sit exactly on the limit, one binding away from the
	// failure this file already hit once on real hardware -- see bug 1 in the
	// resample section. This kernel binds three.
	//
	// **mixing** lerps each particle's concentration toward the mean
	// concentration of the fluid around it, sampled from the same cell-averaged
	// field the density is built from. That is a particle-side stand-in for
	// diffusion: it needs no stencil, no extra field, and no stability
	// condition of its own (it is a convex blend, so it can only ever move a
	// value *between* existing values -- it cannot overshoot, and at mixing=1
	// it just adopts the local mean).
	//
	// It is worth being explicit that this is a phenomenological control, not a
	// discretised diffusion coefficient: `mixing` is a per-frame blend factor,
	// so its effect depends on frame rate, and it is not the Fickian diffusion
	// a physical miscible-fluid model (Ren et al. 2014's mixture model, say)
	// would solve for. It is here because it produces the look -- dye that
	// gradually loses its edges and eventually goes uniform -- at essentially
	// no cost. A real drift-velocity mixture model is the upgrade path.
	//
	// **fade** decays concentration toward zero, so the dye disappears rather
	// than blending in. Same shape as grid_smoke_solver2.js's own `smokeDecay`,
	// and the same reasoning: a scene often wants the tracer gone eventually
	// without wanting the fluid to stop moving.
	//
	// Both are clamped in the kernel graph regardless of what a caller passes,
	// following the precedent grid_flip_solver2.js's velocityDamping set: a
	// negative `fade` would AMPLIFY concentration every frame and a negative
	// `mixing` would push each particle *away* from its surroundings, and both
	// are divergence, not merely a bad-looking result.
	//
	// *** Interaction with resampling -- a warning taken from Houdini ***
	//
	// SideFX's own documentation and user forums repeatedly report that FLIP
	// particle reseeding dilutes a carried colour attribute into uniform mush
	// under shearing, and that users switch reseeding off when they need a
	// sharp boundary between two coloured fluids. This port's `resample` pass
	// does the same kind of thing -- it relocates particles, and a relocated
	// particle carries its concentration to its new home. So a scene chasing
	// crisp dye filaments should expect to set `resample: { enabled: false }`,
	// and one chasing smooth blending can leave it on and let it help. Recorded
	// because it is a real, non-obvious coupling between two features that look
	// unrelated. See THIRD-PARTY-NOTICES.md, "Design and architecture
	// references", for the attribution.
	const mixingNode = clamp( numberOrNode( mixing ), float( 0 ), float( 1 ) );
	const fadeNode = clamp( numberOrNode( fade ), float( 0 ), float( 1 ) );
	const cellCenterOrigin = originNode.add( gridSpacingNode.mul( 0.5 ) );

	const mixConcentration = tsl_array_n.kernel( maxParticles, ( p ) => {

		const local = collocatedValueAtPosition2( liquidFraction, gridSpacingNode, cellCenterOrigin, positions( p ), cellShape );
		const blended = concentrationOf( p ).add( local.sub( concentrationOf( p ) ).mul( mixingNode ) );
		const faded = blended.mul( float( 1 ).sub( fadeNode ) );

		phase( p ).assign( clamp( faded, float( 0 ), float( 1 ) ) );

	} );

	// Skipped entirely when neither control is on, so the default path costs
	// nothing -- not even a dispatch.
	const concentrationPassEnabled = ! ( mixing === 0 && fade === 0 );

	// ---------------------------------------------------------------- advection

	const advectionSolver = createSemiLagrangianAdvectionSolver2( { velocityGrid, collider, dt } );

	const CLAMP_EPSILON = 1e-4;
	const minPos = originNode.add( vec2( CLAMP_EPSILON ) );
	const maxPos = originNode.add( vec2( resolutionX * gridSpacingX, resolutionY * gridSpacingY ) ).sub( vec2( CLAMP_EPSILON ) );

	const advectParticles = tsl_array_n.kernel( maxParticles, ( p ) => {

		const traced = advectionSolver.trace( positions( p ), - 1 );
		positions( p ).assign( clamp( traced, minPos, maxPos ) );

	} );

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

	// ---------------------------------------------------------------- per-phase resampling

	let resamplePass = null;

	if ( resampleEnabled ) {

		// *** Two real bugs found the first time this ran on actual WebGPU,
		// both of them in this pass, and both invisible to every structural
		// test -- worth recording in full, since the shape of each is the
		// kind a reader would otherwise reintroduce. ***
		//
		// 1. **The per-stage storage-buffer limit.** The first version of
		//    this pass bound twelve separate storage buffers in the
		//    donor-claiming kernel (two count grids, two donor pools, four
		//    atomic cursors, positions, velocities, and the two velocity
		//    components). WebGPU's *guaranteed* limit is eight per shader
		//    stage (`maxStorageBuffersPerShaderStage`), so this failed
		//    outright with "The number of storage buffers (12) in the Compute
		//    stage exceeds the maximum per-stage limit (8)" -- pipeline
		//    creation rejected, the kernel silently doing nothing every
		//    frame. Plenty of real hardware raises that limit, which is
		//    exactly what makes it dangerous: it is a portability bug that a
		//    good GPU hides. The rewrite below gets the claim kernel down to
		//    seven bindings and is written to stay there -- if a future
		//    change needs another field in this kernel, fold it into an
		//    existing one rather than adding a binding.
		//
		//    The three things that bought the headroom: the four separate
		//    `array0` cursors became one four-element array; the two donor
		//    pools became one array filled from both ends (liquid upward
		//    from 0, gas downward from the top); and the eligibility test
		//    plus the phase choice moved out into their own kernel, which
		//    writes a single `recipientNeed` field that encodes both.
		//
		// 2. **`min()` does not preserve int-ness, and an array index must be
		//    an int.** Bounding a claimed slot with `min( slot, int( n ) )`
		//    and using the result as an array index produced two
		//    "THREE.TSL: Invalid generated code, expected a int" errors --
		//    one per clamped index. Every bound in this pass is therefore
		//    written as a `lessThan(...).select(...)` instead, which does
		//    preserve the type. Note the contrast with `cellIndexOf` above,
		//    where `max`/`min` around an already-`int()`-wrapped expression
		//    is fine -- so this is not a blanket "never use min on ints",
		//    it is specifically that min's *result* is not safe to index
		//    with.
		//
		// `recipientNeed` encodes eligibility and phase in one int: 0 means
		// "not eligible", a positive n means "needs n more liquid particles",
		// a negative n means "needs n more gas particles". That packing is
		// what lets the claim kernel read one field instead of two count
		// grids, which is what keeps it under the binding limit.
		const recipientNeed = tsl_array_n.arrayN( 'int', cellShape );
		recipientNeed.fromArray( new Int32Array( cellCount ) );

		// One pool, filled from both ends. Each half is capped at
		// floor(maxParticles/2) so the two ends can never meet, which makes
		// every pool index provably in range without a runtime check. A
		// donor beyond that cap is simply dropped -- this pass is
		// best-effort by design (an under-filled recipient is expected and
		// accepted, matching grid_flip_solver2.js and mantaflow's own
		// adjustNumber), so a cap that can only bite in a pathologically
		// clumped frame costs nothing real.
		const donorPool = tsl_array_n.arrayN( 'int', maxParticles );
		const poolHalf = Math.max( 1, Math.floor( maxParticles / 2 ) );
		const poolHalfNode = int( poolHalf );
		const poolTopNode = int( maxParticles - 1 );

		// [ pushLiquid, pushGas, popLiquid, popGas ] -- one binding instead
		// of four array0s. See bug 1 above.
		const CURSOR_PUSH_LIQUID = 0;
		const CURSOR_PUSH_GAS = 1;
		const CURSOR_POP_LIQUID = 2;
		const CURSOR_POP_GAS = 3;
		const cursors = tsl_array_n.arrayN( 'int', 4 );
		cursors.node.toAtomic();

		const zeroCursors = new Int32Array( 4 );

		function resetResampleBuffers() {

			cursors.fromArray( zeroCursors );

		}

		const maxParticlesPerCellNode = int( maxParticlesPerCell );
		const minParticlesPerCellNode = int( minParticlesPerCell );

		function neighborWellPopulatedCount( i, j ) {

			const left = atomicLoad( totalCellCount( max( 0, i.sub( 1 ) ), j ) );
			const right = atomicLoad( totalCellCount( min( resolutionX - 1, i.add( 1 ) ), j ) );
			const down = atomicLoad( totalCellCount( i, max( 0, j.sub( 1 ) ) ) );
			const up = atomicLoad( totalCellCount( i, min( resolutionY - 1, j.add( 1 ) ) ) );

			return left.greaterThanEqual( minParticlesPerCellNode ).select( int( 1 ), int( 0 ) )
				.add( right.greaterThanEqual( minParticlesPerCellNode ).select( int( 1 ), int( 0 ) ) )
				.add( down.greaterThanEqual( minParticlesPerCellNode ).select( int( 1 ), int( 0 ) ) )
				.add( up.greaterThanEqual( minParticlesPerCellNode ).select( int( 1 ), int( 0 ) ) );

		}

		// Which phase a cell should be topped up WITH: its own majority if it
		// has any particles to have a majority of, otherwise its four
		// orthogonal neighbors' majority. Without the neighbor fallback an
		// entirely empty cell would default to one fixed phase and slowly
		// seed that phase into every void that opens up -- the same
		// runaway-growth failure grid_flip_solver2.js's own MIN_FLUID_
		// NEIGHBORS guard exists for, wearing a different hat.
		// *** A unit mismatch that would have been silent: read before editing. ***
		//
		// This test used to compare an integer particle COUNT against another
		// integer particle count, and "majority" was simply `liquid*2 > total`.
		// Now that the accumulator holds a FIXED-POINT SUM OF CONCENTRATIONS
		// instead, the two sides are no longer in the same units -- the sum is
		// scaled up by concentrationAtomicScale (4096 by default). Left
		// uncorrected, `sum*2 > total` is true for essentially any cell with a
		// trace of concentration in it, so every under-full cell would ask for a
		// dye-rich donor and the whole domain would slowly stain itself. It
		// would have looked like a plausible physical effect, which is what
		// makes it worth this comment.
		//
		// Descaling to a mean concentration and comparing against 0.5 puts both
		// sides back in the same units and restores the original meaning.
		function recipientWantsLiquid( i, j ) {

			const total = atomicLoad( totalCellCount( i, j ) ).toFloat();
			const sum = atomicLoad( concentrationAccum( i, j ) ).toFloat().div( concentrationScaleNode );

			const neighborTotal = atomicLoad( totalCellCount( max( 0, i.sub( 1 ) ), j ) )
				.add( atomicLoad( totalCellCount( min( resolutionX - 1, i.add( 1 ) ), j ) ) )
				.add( atomicLoad( totalCellCount( i, max( 0, j.sub( 1 ) ) ) ) )
				.add( atomicLoad( totalCellCount( i, min( resolutionY - 1, j.add( 1 ) ) ) ) )
				.toFloat();

			const neighborSum = atomicLoad( concentrationAccum( max( 0, i.sub( 1 ) ), j ) )
				.add( atomicLoad( concentrationAccum( min( resolutionX - 1, i.add( 1 ) ), j ) ) )
				.add( atomicLoad( concentrationAccum( i, max( 0, j.sub( 1 ) ) ) ) )
				.add( atomicLoad( concentrationAccum( i, min( resolutionY - 1, j.add( 1 ) ) ) ) )
				.toFloat().div( concentrationScaleNode );

			const ownMean = sum.div( max( total, float( 1 ) ) );
			const neighborMean = neighborSum.div( max( neighborTotal, float( 1 ) ) );

			return total.greaterThan( 0.5 ).select(
				ownMean.greaterThan( 0.5 ),
				neighborMean.greaterThan( 0.5 )
			);

		}

		// 3 bindings: the two count grids in, recipientNeed out.
		const computeRecipientNeedKernel = tsl_array_n.kernel( cellShape, ( i, j ) => {

			const count = atomicLoad( totalCellCount( i, j ) );
			const eligible = count.lessThan( minParticlesPerCellNode )
				.and( neighborWellPopulatedCount( i, j ).greaterThanEqual( 2 ) );
			const needed = minParticlesPerCellNode.sub( count );
			const signed = recipientWantsLiquid( i, j ).select( needed, needed.negate() );

			recipientNeed( i, j ).assign( eligible.select( signed, int( 0 ) ) );

		} );

		// 5 bindings: positions, phase, totalCellCount, donorPool, cursors.
		// Over/under detection is on the TOTAL count -- that part is about
		// particle sampling density, which both phases share a grid for --
		// and only which end of the pool a donor lands in depends on phase.
		const buildDonorPoolKernel = tsl_array_n.kernel( maxParticles, ( p ) => {

			const { i, j } = cellIndexOf( positions( p ) );
			const count = atomicLoad( totalCellCount( i, j ) );

			If( count.greaterThan( maxParticlesPerCellNode ), () => {

				If( isLiquidParticle( p ), () => {

					const slot = atomicAdd( cursors( CURSOR_PUSH_LIQUID ), 1 );
					If( slot.lessThan( poolHalfNode ), () => {

						donorPool( slot ).assign( p );

					} );

				} ).Else( () => {

					const slot = atomicAdd( cursors( CURSOR_PUSH_GAS ), 1 );
					If( slot.lessThan( poolHalfNode ), () => {

						donorPool( poolTopNode.sub( slot ) ).assign( p );

					} );

				} );

			} );

		} );

		// 7 bindings: recipientNeed, cursors, donorPool, positions,
		// velocities, dataU, dataV. See bug 1 above before adding an eighth.
		//
		// *** Bug 3, and the reason this reads as two near-duplicate branches
		// instead of one branch-free body: `select()` over an `atomicLoad()`
		// result generates invalid code. ***
		//
		// This was the other half of the "expected a int" errors, and it was
		// isolated with a standalone probe rather than guessed at, because the
		// failing construct looks completely ordinary:
		//
		//     pick.select( atomicLoad( a( 0 ) ), atomicLoad( a( 1 ) ) )   // 2 errors
		//     pick.select( atomicLoad( a( 0 ) ).toInt(),
		//                  atomicLoad( a( 1 ) ).toInt() )                 // clean
		//
		// An atomic load's result does not carry a plain `int` type through
		// `select`, and every value derived from the bad select inherits the
		// problem -- which is why two such selects produced four errors, not
		// two. Note how narrow this is: an atomic result used directly in a
		// *comparison* is fine (grid_flip_solver2.js has always done
		// `claimSlot.lessThan( donorCount )`), and an `atomicAdd` result used
		// directly as an index into a *non-atomic* array is fine (that file
		// indexes its donor pool exactly that way). It is specifically
		// `select` over the atomic result that breaks.
		//
		// `.toInt()` does fix the load case, but the phase choice here is
		// resolved with a real `If`/`Else` instead, because the same kernel also
		// needs to select over two `atomicAdd` results -- and bumping both
		// cursors to then throw one away was always wasteful bookkeeping that
		// only existed to keep the body branch-free. Branching costs nothing in
		// bindings (TSL binds every buffer the kernel names either way), so the
		// loop body is factored into a JS helper and instantiated twice at
		// graph-build time: same code, two clean branches, no select over any
		// atomic anywhere.
		function buildClaimLoop( pushCursor, popCursor, poolIndexOf, needed, claimed, i, j ) {

			// A push cursor keeps counting past the cap it stopped writing at, so
			// it is an over-count rather than a length -- clamped here (via
			// select over plain ints, which is fine) so that `claimSlot <
			// available` is an honest in-range test.
			const pushed = atomicLoad( cursors( pushCursor ) );
			const available = pushed.lessThan( poolHalfNode ).select( pushed.toInt(), poolHalfNode );

			Loop( minParticlesPerCell, () => {

				If( claimed.greaterThanEqual( needed ), () => {

					Break();

				} );

				const claimSlot = atomicAdd( cursors( popCursor ), 1 );

				If( claimSlot.lessThan( available ), () => {

					// claimSlot < available <= poolHalf, so the liquid mapping lands
					// in [0, poolHalf) and the gas mapping in
					// (maxParticles-1-poolHalf, maxParticles-1] -- disjoint and in
					// range by construction, so the index needs no clamp of its own.
					const donorIdx = donorPool( poolIndexOf( claimSlot ) );
					const newPos = originNode.add( gridSpacingNode.mul( 0.5 ) ).add( vec2( i, j ).mul( gridSpacingNode ) );

					positions( donorIdx ).assign( newPos );
					velocities( donorIdx ).assign( faceCenteredValueAtPosition2(
						velocityGrid.dataU, velocityGrid.dataV, velocityGrid.gridSpacing,
						velocityGrid.dataOriginU, velocityGrid.dataOriginV, newPos, dataSizeU, dataSizeV
					) );

					claimed.addAssign( 1 );

				} ).Else( () => {

					Break();

				} );

			} );

		}

		const claimDonorsKernel = tsl_array_n.kernel( cellShape, ( i, j ) => {

			const need = recipientNeed( i, j );

			If( need.notEqual( int( 0 ) ), () => {

				const wantsLiquid = need.greaterThan( int( 0 ) );
				// A select over two plain ints, not over an atomic -- fine.
				const needed = wantsLiquid.select( need, need.negate() );
				const claimed = int( 0 ).toVar();

				If( wantsLiquid, () => {

					buildClaimLoop( CURSOR_PUSH_LIQUID, CURSOR_POP_LIQUID, ( slot ) => slot, needed, claimed, i, j );

				} ).Else( () => {

					buildClaimLoop( CURSOR_PUSH_GAS, CURSOR_POP_GAS, ( slot ) => poolTopNode.sub( slot ), needed, claimed, i, j );

				} );

			} );

		} );

		resamplePass = function resamplePassNow() {

			resetResampleBuffers();
			computeRecipientNeedKernel();
			buildDonorPoolKernel();
			claimDonorsKernel();

		};

	}


	// ---------------------------------------------------------------- frame

	async function onAdvanceTimeStep() {

		advectParticles();
		if ( pushOutOfCollider ) pushOutOfCollider();

		// Counts first: the density field they feed has to describe where the
		// particles are NOW, i.e. after advection, not where they were when
		// last frame's pressure was solved.
		resetCellCounts();
		countPhasesKernel();
		computeDensityKernel();
		computeBetaU();
		computeBetaV();

		resetAccumulators();
		scatterU();
		scatterV();
		finalizeU();
		finalizeV();
		// Snapshot order is grid_flip_solver2.js's, deliberately unchanged:
		// the FLIP delta baseline is taken straight after P2G and BEFORE
		// extrapolation and the wall constraint. Everything the grid does
		// after this point (gravity, the projection, the constraint) is what
		// the particles are supposed to pick up as a delta; the extrapolated
		// values in weightless faces are invented rather than solved for, and
		// must not be handed to particles as though the fluid had produced
		// them. That ordering is real-hardware-validated in the single-phase
		// solver and is not the place to improvise.
		snapshotOldU();
		snapshotOldV();

		extrapolateWeightU();
		extrapolateWeightV();

		boundarySolver.constrainVelocity();

		applyGravityU();
		applyGravityV();
		boundarySolver.constrainVelocity();

		await projectDispatch();
		boundarySolver.constrainVelocity();

		g2pUpdate();

		// After g2p, before resampling: the concentration field this reads was
		// built from the same particle positions still in effect, and resampling
		// is what would move particles out from under it.
		if ( concentrationPassEnabled ) mixConcentration();

		if ( resamplePass ) resamplePass();

	}

	return {
		onAdvanceTimeStep,
		positions, velocities, phase,
		// `concentration` is the same array as `phase`, under the name that
		// actually describes it once the value is continuous. Both are exposed
		// so neither reading of the solver has to use the other's vocabulary.
		concentration: phase,
		liquidFraction, density,
		betaU, betaV,
		pressure: pressureSolver.pressure,
		pinCell,
		boundarySolver, pressureSolver
	};

}
