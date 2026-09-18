// 3D counterpart of grid_smoke_solver2.js -- same "compose createGridSolver3
// with buoyancy + density/temperature advect/decay" design, same 4-field
// ping-pong, same live-parity-flag buoyancy trick, same fire-is-not-a-
// separate-model and source-injection-is-external scope decisions. See that
// file's own extensive header comment for the full reasoning behind all of
// this -- none of it is dimension-specific beyond buoyancy's up vector now
// being vec3 and jet's own 3D default up being +Y (0,1,0), not 2D's (0,1).

import * as tsl_array_n from 'tsl_array_n';
import { vec3, float } from 'three/tsl';
import { createAdvectedScalarField3 } from './array_utils3.js';
import { createSemiLagrangianAdvectionSolver3 } from './advection_solver3.js';
import { createGridSolver3 } from './grid_solver3.js';

function numberOrNode( value ) {

	return typeof value === 'number' ? float( value ) : value;

}

// options: see grid_smoke_solver2.js's own createGridSmokeSolver2 header
// comment for the full parameter list's meaning -- unchanged, resolution/
// gridSpacing/origin/up are now triples ("up" default [0,1,0], jet's own
// 3D default gravity/up axis).
export function createGridSmokeSolver3( {
	velocityGrid,
	gridSpacing = [ 1, 1, 1 ],
	origin = [ 0, 0, 0 ],
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
	up = [ 0, 1, 0 ],
	advection = {},
	pressure = {}
} = {} ) {

	if ( ! velocityGrid ) {

		throw new Error( 'createGridSmokeSolver3: options.velocityGrid is required.' );

	}

	const [ resolutionX, resolutionY, resolutionZ ] = velocityGrid.resolution;
	const [ gridSpacingX, gridSpacingY, gridSpacingZ ] = gridSpacing;
	const [ originX, originY, originZ ] = origin;

	const density = createAdvectedScalarField3( resolutionX, resolutionY, resolutionZ, gridSpacingX, gridSpacingY, gridSpacingZ, originX, originY, originZ );
	const temperature = createAdvectedScalarField3( resolutionX, resolutionY, resolutionZ, gridSpacingX, gridSpacingY, gridSpacingZ, originX, originY, originZ );

	// 0 = state-A active, 1 = state-B active -- see grid_smoke_solver2.js's
	// own header comment for why buoyancy needs this instead of just
	// closing over a fixed field reference.
	const parityFlag = tsl_array_n.array0( 'float' );
	parityFlag.fromArray( new Float32Array( [ 0 ] ) );
	let activeIsA = true; // plain JS mirror, for the caller-facing getters below -- always known locally, no readback needed

	const upNode = vec3( up[ 0 ], up[ 1 ], up[ 2 ] );
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

	const solver = createGridSolver3( {
		velocityGrid, gridSpacing, origin, force: combinedForce, collider, inflows, outflows, closedDomainBoundaryFlag, dt, advection, pressure
	} );

	// order: reuses the SAME `advection` options object already forwarded
	// to the internal createGridSolver3 call above -- see
	// grid_smoke_solver2.js's own comment on this same line for why.
	const scalarAdvectionSolver = createSemiLagrangianAdvectionSolver3( { velocityGrid: solver.velocityGrid, dt, collider, order: advection.order } );

	const advectDensityAtoB = scalarAdvectionSolver.advectScalar3( density.stateA, density.rawB );
	const advectDensityBtoA = scalarAdvectionSolver.advectScalar3( density.stateB, density.rawA );
	const advectTemperatureAtoB = scalarAdvectionSolver.advectScalar3( temperature.stateA, temperature.rawB );
	const advectTemperatureBtoA = scalarAdvectionSolver.advectScalar3( temperature.stateB, temperature.rawA );

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

	// Buoyancy (reads whichever slot parityFlag currently marks active) ->
	// pressure -> advect velocity (createGridSolver3's own default stage
	// order), then density/temperature are advected through this frame's
	// just-updated velocity field and decayed -- matching jet's own
	// "advectable data" timing.
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
