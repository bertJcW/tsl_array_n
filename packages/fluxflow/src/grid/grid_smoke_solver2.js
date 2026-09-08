// A reusable smoke/fire solver, built on top of createGridSolver2 -- not a
// port of any Python source (grid_solver2.py never got past an abstract
// hook, same situation as advection_solver2.js/grid_pressure_solver2.js),
// read directly from jet/fluid-engine-dev's own GridSmokeSolver2
// (grid_smoke_solver2.h/.cpp, MIT, Doyub Kim -- see
// ../../THIRD-PARTY-NOTICES.md for the attribution), which itself cites
// Fedkiw, Stam & Jensen's "Visual Simulation of Smoke" (SIGGRAPH 2001).
// jet's own GridSmokeSolver2 extends GridFluidSolver2 by inheritance
// (density/temperature fields, a buoyancy force, decay); this port has no
// inheritance, so the same shape is built by composition instead -- this
// factory internally constructs its own createGridSolver2 (with buoyancy
// folded into the force option) plus its own density/temperature
// advection, matching this whole port's established "call the base
// factory, then compose" convention.
//
// *** Fire is NOT a separate physical model -- there is no fire/combustion
// reference anywhere in this port's own jet/mantaflow sources (confirmed
// via grep across jet's own include/src trees, zero hits) ***
//
// This is a deliberate scope decision, not an oversight: "fire" here means
// parameterizing and rendering the SAME density+temperature solver (a hot,
// bright source, a temperature-driven color ramp at render time), not a
// second reaction-front/combustion model. A real combustion solver
// (mantaflow's own fire plugin, or Nguyen/Fedkiw/Jensen's own "Physically
// Based Modeling and Animation of Fire", SIGGRAPH 2002) is a genuinely
// different, much larger undertaking with no reference in this project --
// worth reconsidering explicitly if that level of physical fidelity is
// ever actually wanted, not something this file attempts.
//
// *** A real architectural subtlety: buoyancy reading a ping-ponged field
// through a kernel that's only ever built once ***
//
// Density/temperature use this port's established 4-field ping-pong (two
// "state" fields + two "raw advected scratch" fields, alternated by frame
// parity) -- the same shape every existing example already hand-rolls for
// dye -- because a storage field needs exactly one permanent writer kernel
// on this project's WebGL2-fallback dev sandbox (see array_utils.js's own
// header comment). But createExternalForceSolver2's own force(pos) closure
// (external_force_solver2.js) is invoked exactly once, at construction
// time, to build its kernel's node graph -- not re-invoked every frame.
// A buoyancy force reading one fixed ping-pong slot would therefore read
// stale data every other frame. Fixed with a single live array0('float')
// parity flag (toggled via .fromArray() every frame, the same
// already-built-kernel-reads-a-live-node pattern this port already relies
// on for dt/alpha/beta/simTimeUniform elsewhere), driving a select()
// *inside* the once-built buoyancy kernel to read whichever slot is
// currently active -- zero extra dispatches per frame (a plain CPU buffer
// write plus a branchless GPU select, not a kernel), unlike an alternative
// "copy to one stable field" design, which would cost 2 extra dispatches
// every frame -- this project's own CG performance investigation (see
// ../../docs/perf-investigation-cg-gpu-resident-alpha-beta.md) found extra
// dispatches to be a real, non-trivial cost on real hardware.
//
// Buoyancy reading "whichever slot is currently flagged active" means it
// reads the *previous* frame's fully-advected-and-caller-injected result --
// this matches jet's own operator-splitting order exactly (forces, which
// includes buoyancy, always run before advection within the same frame in
// PhysicsAnimation/GridFluidSolver2's own stage sequence), not a deviation.
//
// *** Density/temperature use createCellCenteredScalarGrid2, not the plain
// createScalarGrid2 every dye example uses ***
//
// Dye's own half-cell sampling-position error (createScalarGrid2 has no
// half-cell offset) is imperceptible for a passively-advected visual
// field. Density/temperature feed back into the buoyancy *force*, so
// getting the sample position right actually matters here -- matching
// jet's own CellCenteredScalarGrid2 choice precisely.
//
// *** Source injection is deliberately not built in ***
//
// No emitter abstraction exists anywhere in this port yet, and jet's own
// GridSmokeSolver2 doesn't have one built in either (sources are added
// externally, via separate GridEmitter objects, in jet's fuller API). A
// caller builds their own injection kernel(s) against density.stateA/
// stateB (mirroring every existing example's own createInjectKernel(
// rawAdvectedGrid, stateGrid) dye pattern exactly), called once per frame
// after onAdvanceTimeStep(), choosing A or B via
// density.current === density.stateA. Keeps this factory focused on
// buoyancy + advect + decay + velocity/pressure orchestration only.

import * as tsl_array_n from 'tsl_array_n';
import { vec2, float } from 'three/tsl';
import { createAdvectedScalarField } from './array_utils.js';
import { createSemiLagrangianAdvectionSolver2 } from './advection_solver2.js';
import { createGridSolver2 } from './grid_solver2.js';

function numberOrNode( value ) {

	return typeof value === 'number' ? float( value ) : value;

}

// options.velocityGrid/gridSpacing/origin/collider/inflows/outflows/
// closedDomainBoundaryFlag/dt/advection/pressure: forwarded to the
// internal createGridSolver2 call unchanged, same meaning as that
// factory's own options.
// options.force: optional EXTRA force (e.g. wind), composed with buoyancy
// into one function -- createGridSolver2's own force option only accepts
// one function (its computeExternalForces override is a full-stage
// replacement, not an additive hook), so this is the intended way to add
// a caller-supplied force on top of buoyancy, not a workaround.
// options.buoyancySmokeDensityFactor/buoyancyTemperatureFactor: jet's own
// defaults (-0.000625, 5.0) -- negative density factor means denser smoke
// sinks, positive temperature factor means hotter gas rises. Plain number
// or an already-invoked live node (same "number or node" convention as dt
// elsewhere in this port), so a caller can wire either to a live UI
// control the same way examples/16 already does for its own force
// strength.
// options.ambientTemperature: plain number or live node, default 0 --
// deliberately NOT jet's own live domain-averaged temperature (which would
// need a new GPU sum reduction plus one more per-frame readback, exactly
// the per-frame sync cost this project's own CG performance investigation
// found expensive). A fixed ambient constant is the right default for the
// common case; a live-averaged variant would be a natural future addition
// to reduction.js (a createSumReducer sibling to createMaxAbsReducer), not
// built here.
// options.smokeDecay/temperatureDecay: jet's own defaults (0.001, 0.001).
// options.up: [x,y], default [0,1] (jet's own default when no gravity
// vector is set) -- baked into the buoyancy kernel as a constant at
// construction time, same documented "kernels bind at build time"
// limitation as frictionCoefficient elsewhere in this port (a 2D scene's
// own fixed "which way is up", not something that needs to vary frame to
// frame).
export function createGridSmokeSolver2( {
	velocityGrid,
	gridSpacing = [ 1, 1 ],
	origin = [ 0, 0 ],
	collider,
	inflows,
	outflows,
	closedDomainBoundaryFlag,
	force,
	dt,
	buoyancySmokeDensityFactor = -0.000625,
	buoyancyTemperatureFactor = 5.0,
	ambientTemperature = 0,
	smokeDecay = 0.001,
	temperatureDecay = 0.001,
	up = [ 0, 1 ],
	advection = {},
	pressure = {}
} = {} ) {

	if ( ! velocityGrid ) {

		throw new Error( 'createGridSmokeSolver2: options.velocityGrid is required.' );

	}

	const [ resolutionX, resolutionY ] = velocityGrid.resolution;
	const [ gridSpacingX, gridSpacingY ] = gridSpacing;
	const [ originX, originY ] = origin;

	const density = createAdvectedScalarField( resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY );
	const temperature = createAdvectedScalarField( resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY );

	// 0 = state-A active, 1 = state-B active -- see this file's own header
	// comment for why buoyancy needs this instead of just closing over a
	// fixed field reference.
	const parityFlag = tsl_array_n.array0( 'float' );
	parityFlag.fromArray( new Float32Array( [ 0 ] ) );
	let activeIsA = true; // plain JS mirror, for the caller-facing getters below -- always known locally, no readback needed

	const upNode = vec2( up[ 0 ], up[ 1 ] );
	const buoyancyDensityFactorNode = numberOrNode( buoyancySmokeDensityFactor );
	const buoyancyTemperatureFactorNode = numberOrNode( buoyancyTemperatureFactor );
	const ambientTemperatureNode = numberOrNode( ambientTemperature );

	function buoyancyForce( pos ) {

		const activeIsAFlag = parityFlag().equal( 0 );
		const densitySample = activeIsAFlag.select( density.stateA.sample( pos ), density.stateB.sample( pos ) );
		const temperatureSample = activeIsAFlag.select( temperature.stateA.sample( pos ), temperature.stateB.sample( pos ) );

		const fBuoy = buoyancyDensityFactorNode.mul( densitySample )
			.add( buoyancyTemperatureFactorNode.mul( temperatureSample.sub( ambientTemperatureNode ) ) );

		return upNode.mul( fBuoy );

	}

	const combinedForce = force ? ( pos ) => buoyancyForce( pos ).add( force( pos ) ) : buoyancyForce;

	const solver = createGridSolver2( {
		velocityGrid, gridSpacing, origin, force: combinedForce, collider, inflows, outflows, closedDomainBoundaryFlag, dt, advection, pressure
	} );

	// order: reuses the SAME `advection` options object already forwarded
	// to the internal createGridSolver2 call above (for velocity's own
	// self-advection) -- so a caller's single `advection: { order: 2 }`
	// option applies MacCormack to density/temperature too, not just
	// velocity. Nothing else from `advection` (e.g. maxSubsteps) is
	// reused here on purpose: this solver's own dt/collider are already
	// its own explicit options, not meant to be overridden through the
	// nested `advection` object the way createGridSolver2's own advection
	// stage is configured.
	const scalarAdvectionSolver = createSemiLagrangianAdvectionSolver2( { velocityGrid: solver.velocityGrid, dt, collider, order: advection.order } );

	const advectDensityAtoB = scalarAdvectionSolver.advectScalar2( density.stateA, density.rawB );
	const advectDensityBtoA = scalarAdvectionSolver.advectScalar2( density.stateB, density.rawA );
	const advectTemperatureAtoB = scalarAdvectionSolver.advectScalar2( temperature.stateA, temperature.rawB );
	const advectTemperatureBtoA = scalarAdvectionSolver.advectScalar2( temperature.stateB, temperature.rawA );

	function buildDecayKernel( rawField, stateField, decay ) {

		const decayNode = numberOrNode( decay );

		return tsl_array_n.kernel( stateField.dataSize, ( i, j ) => {

			stateField.data( i, j ).assign( rawField.data( i, j ).mul( float( 1 ).sub( decayNode ) ) );

		} );

	}

	const decayDensityB = buildDecayKernel( density.rawB, density.stateB, smokeDecay );
	const decayDensityA = buildDecayKernel( density.rawA, density.stateA, smokeDecay );
	const decayTemperatureB = buildDecayKernel( temperature.rawB, temperature.stateB, temperatureDecay );
	const decayTemperatureA = buildDecayKernel( temperature.rawA, temperature.stateA, temperatureDecay );

	// Buoyancy (reads whichever slot parityFlag currently marks active) ->
	// pressure -> advect velocity (createGridSolver2's own default stage
	// order), then density/temperature are advected through this frame's
	// just-updated velocity field and decayed -- matching jet's own
	// "advectable data" timing (everything advects together, once
	// velocity's own forces/pressure for this frame are already applied).
	async function onAdvanceTimeStep() {

		await solver.onAdvanceTimeStep();

		if ( activeIsA ) {

			advectDensityAtoB();
			decayDensityB();
			advectTemperatureAtoB();
			decayTemperatureB();

		} else {

			advectDensityBtoA();
			decayDensityA();
			advectTemperatureBtoA();
			decayTemperatureA();

		}

		activeIsA = ! activeIsA;
		parityFlag.fromArray( new Float32Array( [ activeIsA ? 0 : 1 ] ) );

	}

	return {
		onAdvanceTimeStep,
		velocityGrid: solver.velocityGrid,
		density: { stateA: density.stateA, stateB: density.stateB, get current() { return activeIsA ? density.stateA : density.stateB; } },
		temperature: { stateA: temperature.stateA, stateB: temperature.stateB, get current() { return activeIsA ? temperature.stateA : temperature.stateB; } },
		solver
	};

}
