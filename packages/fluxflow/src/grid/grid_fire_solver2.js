// A real fuel/combustion solver -- unlike grid_smoke_solver2.js's own
// "fire" (explicitly just a parameterized/colored density+temperature
// field, no reaction model, see that file's own header comment), this one
// actually burns a fuel field down over time and drives smoke/heat output
// from that reaction. Ported from mantaflow's own fire.cpp plugin
// (KnProcessBurn/processBurn, KnUpdateFlame/updateFlame -- Apache-2.0,
// Tobias Pfaff & Nils Thuerey; see ../../THIRD-PARTY-NOTICES.md for the
// full attribution), fetched and read directly rather than assumed.
//
// *** Decoupled from any specific velocity solver, per the user's own
// explicit request ***
//
// grid_smoke_solver2.js builds its own internal createGridSolver2 and owns
// the whole velocity/pressure/force orchestration. This solver does not --
// options.velocityGrid is read-only input (sampled for advecting this
// solver's own 4 fields, and for buoyancy), never written to, and this
// file never constructs a createGridSolver2 at all. Instead it returns
// `force(pos)`, a plain (pos)=>vec2 the caller composes into *whichever*
// velocity solver they're already using (createGridSolver2 directly,
// createGridSmokeSolver2, or something fully custom) -- exactly the same
// composable-force contract vorticity_confinement2.js already established
// this project: a solver that adds a physical effect without owning the
// simulation it's added to. This means a caller MUST call their own
// chosen solver's onAdvanceTimeStep() first, then this file's own
// onAdvanceTimeStep() second, every frame -- this solver's own advection
// step needs that frame's already-updated velocity, and its own
// buoyancyForce reads the *previous* frame's density/temperature (see the
// same timing note in grid_smoke_solver2.js's own header comment; it
// applies identically here, for the identical reason).
//
// *** The actual combustion math, read directly from fire.cpp's own
// KnProcessBurn (not reconstructed from a paper or assumed) ***
//
// Per cell, each frame: fuel burns down at a constant `burningRate`
// (clamped to >=0); `react` (0..1, "how much reaction potential this
// batch of fuel has left") scales down in exact proportion to how much of
// the *original* fuel present this step is now gone (`react *=
// newFuel/origFuel`), and is forced to 0 if there was next to no fuel to
// begin with; `flame = sqrt(react)` (mantaflow's own smoothing -- a
// linear react produces a flame that visually "holds" near full brightness
// longer, only tailing off sharply near the very end). How much fuel was
// just consumed this step drives both how much smoke gets emitted
// (density increases, more so as the fuel supply nears exhaustion -- the
// same "guttering candle gets smokier" real-world behavior mantaflow's
// own formula reproduces) and, wherever flame>0, this cell's own
// temperature is set to a lerp between `ignitionTemp` and `maxTemp` by
// `flame` -- a cell with react=0 (no reaction happening, or none ever
// started) has its temperature left untouched by this step, exactly
// matching fire.cpp's own `if (heat && flame)` guard.
//
// *** Scope cut from fire.cpp: colored smoke (its own optional
// red/green/blue mixing) is not ported ***
//
// No colored-smoke concept exists anywhere else in this port -- adding one
// just for this file would be new, unrequested scope, not a port of
// something this project already has a use for. Worth reconsidering
// explicitly if colored fire/smoke is ever actually wanted.
//
// *** flame is not a 5th ping-ponged GPU field ***
//
// flame = sqrt(react) is a pure function of react, needed only for
// visualization -- cheaper to compute once at render/draw time from the
// already-read-back react array (the same place examples/17-18 already
// compute their own hot-ramp color from density/temperature client-side)
// than to advect+decay+burn a whole extra field for it.
//
// *** Fuel injection reuses this project's own just-proven-safe
// multi-writer-per-field pattern ***
//
// Each fuel source's own injection kernels are built once per (source,
// ping-pong slot) pair and dispatched every single frame, in a fixed
// order, right after that slot's own rightful advect-then-copy writer --
// exactly grid_blocked_boundary_condition_solver2.js's own
// buildInflowKernels/applyInflow() shape (a real, long-proven-on-real-
// hardware precedent for "more than one kernel object touching a field is
// fine, as long as every one of them dispatches consistently, every
// frame, from the very first one"), and the same lesson this session's
// examples/18-explosion fix already confirmed the hard way (a one-off,
// pre-loop write is what actually broke on this project's WebGL2-fallback
// backend -- a consistently-ordered per-frame pair did not).

import * as tsl_array_n from 'tsl_array_n';
import { vec2, float, max, sqrt, clamp } from 'three/tsl';
import { createAdvectedScalarField, createCopyKernel2 } from './array_utils.js';
import { createSemiLagrangianAdvectionSolver2 } from './advection_solver2.js';
import { isInsideSdf } from './level_set_utils.js';

const EPSILON = 1e-6;

function numberOrNode( value ) {

	return typeof value === 'number' ? float( value ) : value;

}

// One (applyFuelA, applyFuelB, applyReactA, applyReactB) kernel quartet per
// fuel source -- see this file's own header comment for why one kernel per
// (source, slot) pair, not a single combined kernel, is the pattern this
// project has already confirmed safe. Setting react=1 alongside fuel
// (regardless of `mode`) mirrors mantaflow's own scene-level convention of
// injecting fuel and react together at a source: fresh fuel always means
// full reaction potential, matching KnProcessBurn's own reading of react
// as "fraction of *this batch's* original fuel remaining", not an
// absolute quantity tied to the fuel value itself.
function buildFuelSourceKernels( fuelSource, fuelState, reactState ) {

	function buildApplyFuel( stateField ) {

		return tsl_array_n.kernel( stateField.dataSize, ( i, j ) => {

			const pos = stateField.dataPosition( i, j );

			tsl_array_n.If( isInsideSdf( fuelSource.sample( pos ) ), () => {

				if ( fuelSource.mode === 'add' ) stateField.data( i, j ).addAssign( fuelSource.fuel );
				else stateField.data( i, j ).assign( fuelSource.fuel );

			} );

		} );

	}

	function buildApplyReact( stateField ) {

		return tsl_array_n.kernel( stateField.dataSize, ( i, j ) => {

			const pos = stateField.dataPosition( i, j );

			tsl_array_n.If( isInsideSdf( fuelSource.sample( pos ) ), () => {

				stateField.data( i, j ).assign( float( 1 ) );

			} );

		} );

	}

	return {
		applyFuelA: buildApplyFuel( fuelState.stateA ),
		applyFuelB: buildApplyFuel( fuelState.stateB ),
		applyReactA: buildApplyReact( reactState.stateA ),
		applyReactB: buildApplyReact( reactState.stateB )
	};

}

// The burn step itself -- ported directly from KnProcessBurn (see this
// file's own header comment for the full derivation), operating in place
// on whichever ping-pong slot is "current" for this frame (a pure
// self-touch per cell: reads and writes only its own (i,j), never a
// neighbor -- the same safe-in-place shape already established by e.g.
// grid_blocked_boundary_condition_solver2.js's own clampVelocityU).
function buildBurnKernel( fuelState, densityState, reactState, temperatureState, burningRateNode, flameSmokeNode, ignitionTempNode, maxTempNode, dtNode ) {

	return tsl_array_n.kernel( fuelState.dataSize, ( i, j ) => {

		const origFuel = fuelState.data( i, j ).toVar();
		const origSmoke = densityState.data( i, j ).toVar();

		const newFuel = max( origFuel.sub( burningRateNode.mul( dtNode ) ), float( 0 ) ).toVar();
		fuelState.data( i, j ).assign( newFuel );

		const flame = float( 0 ).toVar();

		tsl_array_n.If( origFuel.greaterThan( EPSILON ), () => {

			const newReact = reactState.data( i, j ).mul( newFuel.div( origFuel ) );
			reactState.data( i, j ).assign( newReact );
			flame.assign( sqrt( newReact ) );

		} ).Else( () => {

			reactState.data( i, j ).assign( float( 0 ) );

		} );

		// Smoke emission grows as fuel nears exhaustion (mantaflow's own
		// "guttering candle gets smokier" formula) -- `smokeEmitBase`
		// only depends on how much fuel was originally *present* this
		// step, `smokeEmit` scales that by how much was actually
		// *consumed* this step.
		const smokeEmitBase = origFuel.lessThan( 1 ).select( float( 1 ).sub( origFuel ).mul( 0.5 ), float( 0 ) );
		const smokeEmit = smokeEmitBase.add( 0.5 ).mul( origFuel.sub( newFuel ) ).mul( 0.1 ).mul( flameSmokeNode );
		densityState.data( i, j ).assign( clamp( origSmoke.add( smokeEmit ), float( 0 ), float( 1 ) ) );

		// Matches fire.cpp's own `if (heat && flame)` guard exactly --
		// temperature is left untouched wherever no reaction is
		// happening this step (flame==0), not reset to some baseline.
		tsl_array_n.If( flame.greaterThan( 0 ), () => {

			temperatureState.data( i, j ).assign( float( 1 ).sub( flame ).mul( ignitionTempNode ).add( flame.mul( maxTempNode ) ) );

		} );

	} );

}

// options.velocityGrid: required -- read-only input this solver advects
// its own fields through and samples for buoyancy; never constructed or
// mutated here (see this file's own header comment on decoupling).
// options.collider: optional, forwarded to this solver's own internal
// advection solver (its boundary-crossing clamp) -- NOT to any velocity
// solver, since this file doesn't have one of its own.
// options.fuelSources: one createSDFFuelSource2(...) (sdf_inflow_outflow2.js)
// or an array of them -- omit for a solver with no fuel input at all
// (e.g. a scene that seeds fuel once directly via fuel.stateA.data.fromArray()
// the same way examples/18-explosion seeds its own one-shot burst).
// options.advection: forwarded to the internal scalar advection solver
// (e.g. { order: 2, maxSubsteps }) -- same MacCormack pass-through
// grid_smoke_solver2.js already established.
// options.burningRate/flameSmoke/ignitionTemp/maxTemp: mantaflow's own
// literal fire.cpp defaults (0.75, 1.0, 1.25, 1.75).
// options.buoyancySmokeDensityFactor/buoyancyTemperatureFactor/
// ambientTemperature/smokeDecay/temperatureDecay/up: same meaning and
// defaults as grid_smoke_solver2.js's own options of the same names --
// this solver's density/temperature are driven by combustion instead of a
// caller-supplied source, but still feed buoyancy/decay identically.
export function createGridFireSolver2( {
	velocityGrid,
	gridSpacing = [ 1, 1 ],
	origin = [ 0, 0 ],
	collider,
	fuelSources,
	dt,
	advection = {},
	burningRate = 0.75,
	flameSmoke = 1.0,
	ignitionTemp = 1.25,
	maxTemp = 1.75,
	buoyancySmokeDensityFactor = -0.000625,
	buoyancyTemperatureFactor = 5.0,
	ambientTemperature = 0,
	smokeDecay = 0.001,
	temperatureDecay = 0.001,
	up = [ 0, 1 ]
} = {} ) {

	if ( ! velocityGrid ) {

		throw new Error( 'createGridFireSolver2: options.velocityGrid is required.' );

	}

	const [ resolutionX, resolutionY ] = velocityGrid.resolution;
	const [ gridSpacingX, gridSpacingY ] = gridSpacing;
	const [ originX, originY ] = origin;

	const fuel = createAdvectedScalarField( resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY );
	const react = createAdvectedScalarField( resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY );
	const density = createAdvectedScalarField( resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY );
	const temperature = createAdvectedScalarField( resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY );

	// 0 = state-A active, 1 = state-B active -- same live-select trick as
	// grid_smoke_solver2.js's own parityFlag, for the same reason: force()
	// below is built once but must still read fresh state every frame.
	const parityFlag = tsl_array_n.array0( 'float' );
	parityFlag.fromArray( new Float32Array( [ 0 ] ) );
	let activeIsA = true;

	const upNode = vec2( up[ 0 ], up[ 1 ] );
	const buoyancyDensityFactorNode = numberOrNode( buoyancySmokeDensityFactor );
	const buoyancyTemperatureFactorNode = numberOrNode( buoyancyTemperatureFactor );
	const ambientTemperatureNode = numberOrNode( ambientTemperature );

	function force( pos ) {

		const activeIsAFlag = parityFlag().equal( 0 );
		const densitySample = activeIsAFlag.select( density.stateA.sample( pos ), density.stateB.sample( pos ) );
		const temperatureSample = activeIsAFlag.select( temperature.stateA.sample( pos ), temperature.stateB.sample( pos ) );

		const fBuoy = buoyancyDensityFactorNode.mul( densitySample )
			.add( buoyancyTemperatureFactorNode.mul( temperatureSample.sub( ambientTemperatureNode ) ) );

		return upNode.mul( fBuoy );

	}

	const scalarAdvectionSolver = createSemiLagrangianAdvectionSolver2( { velocityGrid, dt, collider, order: advection.order, maxSubsteps: advection.maxSubsteps } );

	const advectFuelAtoB = scalarAdvectionSolver.advectScalar2( fuel.stateA, fuel.rawB );
	const advectFuelBtoA = scalarAdvectionSolver.advectScalar2( fuel.stateB, fuel.rawA );
	const advectReactAtoB = scalarAdvectionSolver.advectScalar2( react.stateA, react.rawB );
	const advectReactBtoA = scalarAdvectionSolver.advectScalar2( react.stateB, react.rawA );
	const advectDensityAtoB = scalarAdvectionSolver.advectScalar2( density.stateA, density.rawB );
	const advectDensityBtoA = scalarAdvectionSolver.advectScalar2( density.stateB, density.rawA );
	const advectTemperatureAtoB = scalarAdvectionSolver.advectScalar2( temperature.stateA, temperature.rawB );
	const advectTemperatureBtoA = scalarAdvectionSolver.advectScalar2( temperature.stateB, temperature.rawA );

	// fuel/react have no independent decay -- they're consumed by
	// burningRate inside the burn kernel instead, so raw->state is a
	// plain copy (createCopyKernel2, array_utils.js). density/temperature
	// keep the same decay-kernel shape grid_smoke_solver2.js already uses.
	const copyFuelRawBToStateB = createCopyKernel2( fuel.rawB.data, fuel.stateB.data, fuel.stateB.dataSize );
	const copyFuelRawAToStateA = createCopyKernel2( fuel.rawA.data, fuel.stateA.data, fuel.stateA.dataSize );
	const copyReactRawBToStateB = createCopyKernel2( react.rawB.data, react.stateB.data, react.stateB.dataSize );
	const copyReactRawAToStateA = createCopyKernel2( react.rawA.data, react.stateA.data, react.stateA.dataSize );

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

	const fuelSourceList = ! fuelSources ? [] : ( Array.isArray( fuelSources ) ? fuelSources : [ fuelSources ] );
	const fuelSourceKernels = fuelSourceList.map( ( source ) => buildFuelSourceKernels( source, fuel, react ) );

	function applyFuelSources( targetIsA ) {

		for ( const k of fuelSourceKernels ) {

			if ( targetIsA ) {

				k.applyFuelA();
				k.applyReactA();

			} else {

				k.applyFuelB();
				k.applyReactB();

			}

		}

	}

	const dtNode = numberOrNode( dt );
	const burningRateNode = numberOrNode( burningRate );
	const flameSmokeNode = numberOrNode( flameSmoke );
	const ignitionTempNode = numberOrNode( ignitionTemp );
	const maxTempNode = numberOrNode( maxTemp );

	const burnA = buildBurnKernel( fuel.stateA, density.stateA, react.stateA, temperature.stateA, burningRateNode, flameSmokeNode, ignitionTempNode, maxTempNode, dtNode );
	const burnB = buildBurnKernel( fuel.stateB, density.stateB, react.stateB, temperature.stateB, burningRateNode, flameSmokeNode, ignitionTempNode, maxTempNode, dtNode );

	// Advect all 4 fields (old state -> new raw) -> copy/decay raw -> new
	// state -> inject fuel sources into the new state -> burn the new
	// state in place -> flip parity. See this file's own header comment
	// for why the call-order contract with the caller's own velocity
	// solver matters (this must run *after* it, each frame).
	async function onAdvanceTimeStep() {

		if ( activeIsA ) {

			advectFuelAtoB();
			advectReactAtoB();
			advectDensityAtoB();
			advectTemperatureAtoB();

			copyFuelRawBToStateB();
			copyReactRawBToStateB();
			decayDensityB();
			decayTemperatureB();

			applyFuelSources( false );

			burnB();

		} else {

			advectFuelBtoA();
			advectReactBtoA();
			advectDensityBtoA();
			advectTemperatureBtoA();

			copyFuelRawAToStateA();
			copyReactRawAToStateA();
			decayDensityA();
			decayTemperatureA();

			applyFuelSources( true );

			burnA();

		}

		activeIsA = ! activeIsA;
		parityFlag.fromArray( new Float32Array( [ activeIsA ? 0 : 1 ] ) );

	}

	return {
		onAdvanceTimeStep,
		force,
		fuel: { stateA: fuel.stateA, stateB: fuel.stateB, get current() { return activeIsA ? fuel.stateA : fuel.stateB; } },
		react: { stateA: react.stateA, stateB: react.stateB, get current() { return activeIsA ? react.stateA : react.stateB; } },
		density: { stateA: density.stateA, stateB: density.stateB, get current() { return activeIsA ? density.stateA : density.stateB; } },
		temperature: { stateA: temperature.stateA, stateB: temperature.stateB, get current() { return activeIsA ? temperature.stateA : temperature.stateB; } }
	};

}
