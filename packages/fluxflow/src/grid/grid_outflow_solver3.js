// 3D counterpart of grid_outflow_solver2.js -- same three-part outflow
// mechanism (pressure Dirichlet ghost, convective velocity extrapolation,
// scalar-field cleanup) read from mantaflow, see that file's own header
// comment for the full derivation, the OUTFLOW_TIMESTEP_FLOOR/SCALE
// porting-detail history, and the EXTRAPOLATED_VELOCITY_CLAMP safety
// margin -- none of it dimension-specific beyond the extra W component.

import * as tsl_array_n from 'tsl_array_n';
import { float, max, clamp } from 'three/tsl';
import { isInsideSdf } from './level_set_utils.js';
import { createOutflowPressureDirichlet3 } from './sdf_inflow_outflow3.js';

const OUTFLOW_TIMESTEP_FLOOR = 1;
const OUTFLOW_TIMESTEP_SCALE = 4;
const EXTRAPOLATED_VELOCITY_CLAMP = 1e3;

// options: see grid_outflow_solver2.js's own createGridOutflowSolver2
// header comment for velocityGrid/velocityPrev/outflows/dt/applyVelocityBC
// -- unchanged conventions, one more staggered component.
export function createGridOutflowSolver3( { velocityGrid, velocityPrev, outflows, dt, applyVelocityBC = true } ) {

	const list = Array.isArray( outflows ) ? outflows : [ outflows ];
	const dtNode = typeof dt === 'number' ? float( dt ) : dt;

	function computeOutflowFactor( bulkVelComponent ) {

		const factorTimeStep = max( float( OUTFLOW_TIMESTEP_FLOOR ), dtNode.mul( OUTFLOW_TIMESTEP_SCALE ) );
		return factorTimeStep.mul( max( float( 1 ), bulkVelComponent ) );

	}

	const uDst = tsl_array_n.arrayN( 'float', velocityGrid.dataSizeU );
	const vDst = tsl_array_n.arrayN( 'float', velocityGrid.dataSizeV );
	const wDst = tsl_array_n.arrayN( 'float', velocityGrid.dataSizeW );

	function buildOutflowVelocityKernels( outflow ) {

		function makeExtrapolate( dataSize, positionFn, gridSpacingComponent, component, dst, dataComponent ) {

			return tsl_array_n.kernel( dataSize, ( i, j, k ) => {

				const pt = positionFn( i, j, k );

				tsl_array_n.If( isInsideSdf( outflow.sample( pt ) ), () => {

					const g = outflow.gradient( pt );

					tsl_array_n.If( g.length().greaterThan( 0 ), () => {

						const n = g.normalize();
						const upstreamPt = pt.sub( n.mul( gridSpacingComponent ) );

						const bulkVel = velocityGrid.sample( pt );
						const factor = computeOutflowFactor( bulkVel[ component ] );

						const upstreamVel = velocityGrid.sample( upstreamPt );
						const prevVel = velocityPrev.sample( pt );
						const current = dataComponent( i, j, k );

						const extrapolated = current.sub( prevVel[ component ] ).div( factor ).add( upstreamVel[ component ] );
						dst( i, j, k ).assign( clamp( extrapolated, - EXTRAPOLATED_VELOCITY_CLAMP, EXTRAPOLATED_VELOCITY_CLAMP ) );

					} ).Else( () => {

						dst( i, j, k ).assign( dataComponent( i, j, k ) );

					} );

				} );

			} );

		}

		function makeCopy( dataSize, positionFn, dst, dataComponent ) {

			return tsl_array_n.kernel( dataSize, ( i, j, k ) => {

				const pt = positionFn( i, j, k );

				tsl_array_n.If( isInsideSdf( outflow.sample( pt ) ), () => {

					dataComponent( i, j, k ).assign( dst( i, j, k ) );

				} );

			} );

		}

		const extrapolateU = makeExtrapolate( velocityGrid.dataSizeU, velocityGrid.uPosition, velocityGrid.gridSpacing.x, 'x', uDst, velocityGrid.dataU );
		const copyU = makeCopy( velocityGrid.dataSizeU, velocityGrid.uPosition, uDst, velocityGrid.dataU );
		const extrapolateV = makeExtrapolate( velocityGrid.dataSizeV, velocityGrid.vPosition, velocityGrid.gridSpacing.y, 'y', vDst, velocityGrid.dataV );
		const copyV = makeCopy( velocityGrid.dataSizeV, velocityGrid.vPosition, vDst, velocityGrid.dataV );
		const extrapolateW = makeExtrapolate( velocityGrid.dataSizeW, velocityGrid.wPosition, velocityGrid.gridSpacing.z, 'z', wDst, velocityGrid.dataW );
		const copyW = makeCopy( velocityGrid.dataSizeW, velocityGrid.wPosition, wDst, velocityGrid.dataW );

		return { extrapolateU, copyU, extrapolateV, copyV, extrapolateW, copyW };

	}

	const velocityKernels = list.map( buildOutflowVelocityKernels );

	function applyOutflowVelocityBC() {

		if ( ! applyVelocityBC ) return;

		for ( const { extrapolateU, copyU, extrapolateV, copyV, extrapolateW, copyW } of velocityKernels ) {

			extrapolateU();
			copyU();
			extrapolateV();
			copyV();
			extrapolateW();
			copyW();

		}

	}

	function clearOutflowScalarField( scalarGrid ) {

		const dispatchers = list.map( ( outflow ) => tsl_array_n.kernel( scalarGrid.dataSize, ( i, j, k ) => {

			const pt = scalarGrid.dataPosition( i, j, k );

			tsl_array_n.If( isInsideSdf( outflow.sample( pt ) ), () => {

				scalarGrid.data( i, j, k ).assign( 0 );

			} );

		} ) );

		return function clear() {

			for ( const dispatch of dispatchers ) dispatch();

		};

	}

	return {
		dirichlet: createOutflowPressureDirichlet3( list ),
		applyOutflowVelocityBC,
		clearOutflowScalarField
	};

}
