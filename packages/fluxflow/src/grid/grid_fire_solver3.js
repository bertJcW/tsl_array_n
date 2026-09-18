// 3D counterpart of grid_fire_solver2.js -- same mantaflow-derived
// KnProcessBurn combustion math, same velocity-solver-decoupled "return
// force(pos), caller composes it into whichever solver they're already
// using" contract, same scope cuts (no colored smoke, no separate flame
// field). See that file's own extensive header comment for the full
// derivation and design reasoning, none of which is dimension-specific
// beyond buoyancy's up vector now being vec3.

import * as tsl_array_n from 'tsl_array_n';
import { vec3, float, max, sqrt, clamp } from 'three/tsl';
import { createAdvectedScalarField3, createCopyKernel3 } from './array_utils3.js';
import { createSemiLagrangianAdvectionSolver3 } from './advection_solver3.js';
import { isInsideSdf } from './level_set_utils.js';

const EPSILON = 1e-6;

function numberOrNode( value ) {

	return typeof value === 'number' ? float( value ) : value;

}

// See grid_fire_solver2.js's own buildFuelSourceKernels header comment.
function buildFuelSourceKernels( fuelSource, fuelState, reactState ) {

	function buildApplyFuel( stateField ) {

		return tsl_array_n.kernel( stateField.dataSize, ( i, j, k ) => {

			const pos = stateField.dataPosition( i, j, k );

			tsl_array_n.If( isInsideSdf( fuelSource.sample( pos ) ), () => {

				if ( fuelSource.mode === 'add' ) stateField.data( i, j, k ).addAssign( fuelSource.fuel );
				else stateField.data( i, j, k ).assign( fuelSource.fuel );

			} );

		} );

	}

	function buildApplyReact( stateField ) {

		return tsl_array_n.kernel( stateField.dataSize, ( i, j, k ) => {

			const pos = stateField.dataPosition( i, j, k );

			tsl_array_n.If( isInsideSdf( fuelSource.sample( pos ) ), () => {

				stateField.data( i, j, k ).assign( float( 1 ) );

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

// See grid_fire_solver2.js's own buildBurnKernel header comment for the
// full KnProcessBurn derivation -- unchanged, one more index.
function buildBurnKernel( fuelState, densityState, reactState, temperatureState, burningRateNode, flameSmokeNode, ignitionTempNode, maxTempNode, dtNode ) {

	return tsl_array_n.kernel( fuelState.dataSize, ( i, j, k ) => {

		const origFuel = fuelState.data( i, j, k ).toVar();
		const origSmoke = densityState.data( i, j, k ).toVar();

		const newFuel = max( origFuel.sub( burningRateNode.mul( dtNode ) ), float( 0 ) ).toVar();
		fuelState.data( i, j, k ).assign( newFuel );

		const flame = float( 0 ).toVar();

		tsl_array_n.If( origFuel.greaterThan( EPSILON ), () => {

			const newReact = reactState.data( i, j, k ).mul( newFuel.div( origFuel ) );
			reactState.data( i, j, k ).assign( newReact );
			flame.assign( sqrt( newReact ) );

		} ).Else( () => {

			reactState.data( i, j, k ).assign( float( 0 ) );

		} );

		const smokeEmitBase = origFuel.lessThan( 1 ).select( float( 1 ).sub( origFuel ).mul( 0.5 ), float( 0 ) );
		const smokeEmit = smokeEmitBase.add( 0.5 ).mul( origFuel.sub( newFuel ) ).mul( 0.1 ).mul( flameSmokeNode );
		densityState.data( i, j, k ).assign( clamp( origSmoke.add( smokeEmit ), float( 0 ), float( 1 ) ) );

		tsl_array_n.If( flame.greaterThan( 0 ), () => {

			temperatureState.data( i, j, k ).assign( float( 1 ).sub( flame ).mul( ignitionTempNode ).add( flame.mul( maxTempNode ) ) );

		} );

	} );

}

// options: see grid_fire_solver2.js's own createGridFireSolver2 header
// comment for the full parameter list's meaning -- unchanged, resolution/
// gridSpacing/origin/up are now triples ("up" default [0,1,0]).
export function createGridFireSolver3( {
	velocityGrid,
	gridSpacing = [ 1, 1, 1 ],
	origin = [ 0, 0, 0 ],
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
	up = [ 0, 1, 0 ]
} = {} ) {

	if ( ! velocityGrid ) {

		throw new Error( 'createGridFireSolver3: options.velocityGrid is required.' );

	}

	const [ resolutionX, resolutionY, resolutionZ ] = velocityGrid.resolution;
	const [ gridSpacingX, gridSpacingY, gridSpacingZ ] = gridSpacing;
	const [ originX, originY, originZ ] = origin;

	const fuel = createAdvectedScalarField3( resolutionX, resolutionY, resolutionZ, gridSpacingX, gridSpacingY, gridSpacingZ, originX, originY, originZ );
	const react = createAdvectedScalarField3( resolutionX, resolutionY, resolutionZ, gridSpacingX, gridSpacingY, gridSpacingZ, originX, originY, originZ );
	const density = createAdvectedScalarField3( resolutionX, resolutionY, resolutionZ, gridSpacingX, gridSpacingY, gridSpacingZ, originX, originY, originZ );
	const temperature = createAdvectedScalarField3( resolutionX, resolutionY, resolutionZ, gridSpacingX, gridSpacingY, gridSpacingZ, originX, originY, originZ );

	// 0 = state-A active, 1 = state-B active -- same live-select trick as
	// grid_smoke_solver3.js's own parityFlag.
	const parityFlag = tsl_array_n.array0( 'float' );
	parityFlag.fromArray( new Float32Array( [ 0 ] ) );
	let activeIsA = true;

	const upNode = vec3( up[ 0 ], up[ 1 ], up[ 2 ] );
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

	const scalarAdvectionSolver = createSemiLagrangianAdvectionSolver3( { velocityGrid, dt, collider, order: advection.order, maxSubsteps: advection.maxSubsteps } );

	const advectFuelAtoB = scalarAdvectionSolver.advectScalar3( fuel.stateA, fuel.rawB );
	const advectFuelBtoA = scalarAdvectionSolver.advectScalar3( fuel.stateB, fuel.rawA );
	const advectReactAtoB = scalarAdvectionSolver.advectScalar3( react.stateA, react.rawB );
	const advectReactBtoA = scalarAdvectionSolver.advectScalar3( react.stateB, react.rawA );
	const advectDensityAtoB = scalarAdvectionSolver.advectScalar3( density.stateA, density.rawB );
	const advectDensityBtoA = scalarAdvectionSolver.advectScalar3( density.stateB, density.rawA );
	const advectTemperatureAtoB = scalarAdvectionSolver.advectScalar3( temperature.stateA, temperature.rawB );
	const advectTemperatureBtoA = scalarAdvectionSolver.advectScalar3( temperature.stateB, temperature.rawA );

	// fuel/react have no independent decay -- see grid_fire_solver2.js's
	// own comment on this same line.
	const copyFuelRawBToStateB = createCopyKernel3( fuel.rawB.data, fuel.stateB.data, fuel.stateB.dataSize );
	const copyFuelRawAToStateA = createCopyKernel3( fuel.rawA.data, fuel.stateA.data, fuel.stateA.dataSize );
	const copyReactRawBToStateB = createCopyKernel3( react.rawB.data, react.stateB.data, react.stateB.dataSize );
	const copyReactRawAToStateA = createCopyKernel3( react.rawA.data, react.stateA.data, react.stateA.dataSize );

	function buildDecayKernel( rawField, stateField, decay ) {

		const decayNode = numberOrNode( decay );

		return tsl_array_n.kernel( stateField.dataSize, ( i, j, k ) => {

			stateField.data( i, j, k ).assign( rawField.data( i, j, k ).mul( float( 1 ).sub( decayNode ) ) );

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

	// See grid_fire_solver2.js's own onAdvanceTimeStep header comment for
	// the full stage-order and caller-contract reasoning.
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
