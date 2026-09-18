// 3D counterpart of advection_solver2.js. See that file's own header
// comment for the full backTrace/MacCormack design -- none of it is
// dimension-specific (pt/vel0/midPt etc. are opaque vector nodes, and
// .sub()/.mul()/length() work identically on vec3), so backTrace below is
// a verbatim copy, not a re-derivation. What genuinely changes is the
// per-field logic: a third staggered component (W) for face-centered
// advection, 8-neighbour (not 4) local min/max gathering for MacCormack's
// clamp, and grid_math3.js's tricubic samplers in place of grid_math.js's
// bicubic ones.

import * as tsl_array_n from 'tsl_array_n';
import { float, min, max, length, ceil, abs, If, Loop, Break } from 'three/tsl';
import { collocatedCubicValueAtPosition3, faceCenteredCubicValueAtPosition3, trilinearCoordsAndWeights3 } from './grid_math3.js';

const EPSILON = 1e-6;

// Verbatim copy of advection_solver2.js's backTrace -- see that file's own
// header comment for the algorithm, the direction parameter, and the
// phi1<=0 boundary-crossing fix.
function backTrace( sampleVelocity, sampleBoundary, startPos, dt, h, maxSubsteps, direction = 1 ) {

	const pt = startPos.toVar();
	const remainingT = dt.toVar();

	Loop( maxSubsteps, () => {

		If( remainingT.lessThanEqual( EPSILON ), () => {

			Break();

		} );

		const vel0 = sampleVelocity( pt );
		const numSubSteps = max( ceil( length( vel0 ).mul( remainingT ).div( h ) ), 1 );
		const subDt = remainingT.div( numSubSteps );

		const midPt = pt.sub( vel0.mul( subDt.mul( 0.5 ) ).mul( direction ) );
		const midVel = sampleVelocity( midPt );
		const nextPt = pt.sub( midVel.mul( subDt ).mul( direction ) );

		const phi0 = sampleBoundary( pt );
		const phi1 = sampleBoundary( nextPt );

		If( phi1.lessThanEqual( 0 ), () => {

			const w = abs( phi1 ).div( abs( phi0 ).add( abs( phi1 ) ) );
			pt.assign( pt.mul( w ).add( nextPt.mul( float( 1 ).sub( w ) ) ) );
			remainingT.assign( 0 );
			Break();

		} ).Else( () => {

			remainingT.subAssign( subDt );
			pt.assign( nextPt );

		} );

	} );

	return pt;

}

// options: see advection_solver2.js's own createSemiLagrangianAdvectionSolver2
// header comment for velocityGrid/collider/dt/maxSubsteps/order -- unchanged
// in meaning, one dimension wider.
export function createSemiLagrangianAdvectionSolver3( { velocityGrid, collider, dt, maxSubsteps = 32, order = 1 } ) {

	const dtNode = typeof dt === 'number' ? float( dt ) : dt;
	const h = min( velocityGrid.gridSpacing.x, min( velocityGrid.gridSpacing.y, velocityGrid.gridSpacing.z ) );

	function sampleVelocity( pos ) {

		return velocityGrid.sample( pos );

	}

	function sampleBoundary( pos ) {

		return collider ? collider.sample( pos ) : float( 1 );

	}

	function trace( startPos, direction = 1 ) {

		return backTrace( sampleVelocity, sampleBoundary, startPos, dtNode, h, maxSubsteps, direction );

	}

	// Order-2 only: see advection_solver2.js's own gatherLocalMinMax header
	// comment -- 8 neighbours (trilinearCoordsAndWeights3's corners)
	// instead of 4.
	function gatherLocalMinMax( data, dataOrigin, gridSpacing, shape, positionFn, tracedPos ) {

		const { i0c, j0c, k0c, i1c, j1c, k1c } = trilinearCoordsAndWeights3( tracedPos, dataOrigin, gridSpacing, shape );
		const neighbors = [
			[ i0c, j0c, k0c ], [ i1c, j0c, k0c ], [ i0c, j1c, k0c ], [ i1c, j1c, k0c ],
			[ i0c, j0c, k1c ], [ i1c, j0c, k1c ], [ i0c, j1c, k1c ], [ i1c, j1c, k1c ]
		];

		let minv = float( 1e30 );
		let maxv = float( -1e30 );
		let haveValid = null;

		for ( const [ ni, nj, nk ] of neighbors ) {

			const valid = sampleBoundary( positionFn( ni, nj, nk ) ).greaterThan( 0 );
			const value = data( ni, nj, nk );

			minv = valid.select( min( minv, value ), minv );
			maxv = valid.select( max( maxv, value ), maxv );
			haveValid = haveValid === null ? valid : haveValid.or( valid );

		}

		return { minv, maxv, haveValid };

	}

	function clampCorrection( correctedValue, fwdValue, data, dataOrigin, gridSpacing, shape, positionFn, tracedPos ) {

		const { minv, maxv, haveValid } = gatherLocalMinMax( data, dataOrigin, gridSpacing, shape, positionFn, tracedPos );
		const inRange = correctedValue.greaterThanEqual( minv ).and( correctedValue.lessThanEqual( maxv ) );

		return haveValid.and( inRange ).select( correctedValue, fwdValue );

	}

	// input, output: FaceCenteredGrid3 (grid_data3.js).
	function advectFaceCentered3( input, output ) {

		function sampleComponent( component ) {

			return ( pos ) => faceCenteredCubicValueAtPosition3(
				input.dataU, input.dataV, input.dataW, input.gridSpacing,
				input.dataOriginU, input.dataOriginV, input.dataOriginW, trace( pos ),
				input.dataSizeU, input.dataSizeV, input.dataSizeW
			)[ component ];

		}

		const dispatchU1 = tsl_array_n.kernel( input.dataSizeU, ( i, j, k ) => {

			const pos = input.uPosition( i, j, k );

			If( sampleBoundary( pos ).greaterThan( 0 ), () => {

				output.dataU( i, j, k ).assign( sampleComponent( 'x' )( pos ) );

			} );

		} );

		const dispatchV1 = tsl_array_n.kernel( input.dataSizeV, ( i, j, k ) => {

			const pos = input.vPosition( i, j, k );

			If( sampleBoundary( pos ).greaterThan( 0 ), () => {

				output.dataV( i, j, k ).assign( sampleComponent( 'y' )( pos ) );

			} );

		} );

		const dispatchW1 = tsl_array_n.kernel( input.dataSizeW, ( i, j, k ) => {

			const pos = input.wPosition( i, j, k );

			If( sampleBoundary( pos ).greaterThan( 0 ), () => {

				output.dataW( i, j, k ).assign( sampleComponent( 'z' )( pos ) );

			} );

		} );

		if ( order === 1 ) return tsl_array_n.createBatch( [ dispatchU1, dispatchV1, dispatchW1 ] );

		// order 2 (MacCormack) -- see advection_solver2.js's own header
		// comment for the algorithm. fwdU/fwdV/fwdW/bwdU/bwdV/bwdW: one
		// scratch FaceCenteredGrid3-shaped triple each, same reasoning as
		// the 2D version's fwdU/fwdV pair (faceCenteredCubicValueAtPosition3
		// needs all three components together to interpolate correctly even
		// when only one component's own result is kept).
		const fwdU = tsl_array_n.arrayN( 'float', input.dataSizeU );
		const fwdV = tsl_array_n.arrayN( 'float', input.dataSizeV );
		const fwdW = tsl_array_n.arrayN( 'float', input.dataSizeW );
		const bwdU = tsl_array_n.arrayN( 'float', input.dataSizeU );
		const bwdV = tsl_array_n.arrayN( 'float', input.dataSizeV );
		const bwdW = tsl_array_n.arrayN( 'float', input.dataSizeW );

		function makeForward( dataSize, positionFn, component, out, selfData ) {

			return tsl_array_n.kernel( dataSize, ( i, j, k ) => {

				const pos = positionFn( i, j, k );

				If( sampleBoundary( pos ).greaterThan( 0 ), () => {

					const tracedPos = trace( pos );
					out( i, j, k ).assign( faceCenteredCubicValueAtPosition3(
						input.dataU, input.dataV, input.dataW, input.gridSpacing,
						input.dataOriginU, input.dataOriginV, input.dataOriginW, tracedPos,
						input.dataSizeU, input.dataSizeV, input.dataSizeW
					)[ component ] );

				} ).Else( () => {

					out( i, j, k ).assign( selfData( i, j, k ) );

				} );

			} );

		}

		function makeBackward( dataSize, positionFn, component, out, fwdSelfData ) {

			return tsl_array_n.kernel( dataSize, ( i, j, k ) => {

				const pos = positionFn( i, j, k );

				If( sampleBoundary( pos ).greaterThan( 0 ), () => {

					const tracedPos = trace( pos, -1 );
					out( i, j, k ).assign( faceCenteredCubicValueAtPosition3(
						fwdU, fwdV, fwdW, input.gridSpacing,
						input.dataOriginU, input.dataOriginV, input.dataOriginW, tracedPos,
						input.dataSizeU, input.dataSizeV, input.dataSizeW
					)[ component ] );

				} ).Else( () => {

					out( i, j, k ).assign( fwdSelfData( i, j, k ) );

				} );

			} );

		}

		const dispatchForwardU = makeForward( input.dataSizeU, input.uPosition, 'x', fwdU, input.dataU );
		const dispatchForwardV = makeForward( input.dataSizeV, input.vPosition, 'y', fwdV, input.dataV );
		const dispatchForwardW = makeForward( input.dataSizeW, input.wPosition, 'z', fwdW, input.dataW );

		const dispatchBackwardU = makeBackward( input.dataSizeU, input.uPosition, 'x', bwdU, fwdU );
		const dispatchBackwardV = makeBackward( input.dataSizeV, input.vPosition, 'y', bwdV, fwdV );
		const dispatchBackwardW = makeBackward( input.dataSizeW, input.wPosition, 'z', bwdW, fwdW );

		function makeCorrectAndClamp( dataSize, positionFn, out, fwd, bwd, selfData, dataOrigin ) {

			return tsl_array_n.kernel( dataSize, ( i, j, k ) => {

				const pos = positionFn( i, j, k );

				If( sampleBoundary( pos ).greaterThan( 0 ), () => {

					const tracedPos = trace( pos );
					const fwdValue = fwd( i, j, k );
					const correctedValue = fwdValue.add( selfData( i, j, k ).sub( bwd( i, j, k ) ).mul( 0.5 ) );
					const clamped = clampCorrection( correctedValue, fwdValue, selfData, dataOrigin, input.gridSpacing, dataSize, positionFn, tracedPos );

					out( i, j, k ).assign( clamped );

				} );

			} );

		}

		const dispatchCorrectAndClampU = makeCorrectAndClamp( input.dataSizeU, input.uPosition, output.dataU, fwdU, bwdU, input.dataU, input.dataOriginU );
		const dispatchCorrectAndClampV = makeCorrectAndClamp( input.dataSizeV, input.vPosition, output.dataV, fwdV, bwdV, input.dataV, input.dataOriginV );
		const dispatchCorrectAndClampW = makeCorrectAndClamp( input.dataSizeW, input.wPosition, output.dataW, fwdW, bwdW, input.dataW, input.dataOriginW );

		return tsl_array_n.createBatch( [
			dispatchForwardU, dispatchForwardV, dispatchForwardW,
			dispatchBackwardU, dispatchBackwardV, dispatchBackwardW,
			dispatchCorrectAndClampU, dispatchCorrectAndClampV, dispatchCorrectAndClampW
		] );

	}

	// input, output: ScalarGrid3 (grid_data3.js).
	function advectScalar3( input, output ) {

		const dispatchOrder1 = tsl_array_n.kernel( input.dataSize, ( i, j, k ) => {

			const pos = input.dataPosition( i, j, k );

			If( sampleBoundary( pos ).greaterThan( 0 ), () => {

				const tracedPos = trace( pos );
				output.data( i, j, k ).assign(
					collocatedCubicValueAtPosition3( input.data, input.gridSpacing, input.dataOrigin, tracedPos, input.dataSize )
				);

			} );

		} );

		if ( order === 1 ) return dispatchOrder1;

		const fwd = tsl_array_n.arrayN( 'float', input.dataSize );
		const bwd = tsl_array_n.arrayN( 'float', input.dataSize );

		const dispatchForward = tsl_array_n.kernel( input.dataSize, ( i, j, k ) => {

			const pos = input.dataPosition( i, j, k );

			If( sampleBoundary( pos ).greaterThan( 0 ), () => {

				const tracedPos = trace( pos );
				fwd( i, j, k ).assign( collocatedCubicValueAtPosition3( input.data, input.gridSpacing, input.dataOrigin, tracedPos, input.dataSize ) );

			} ).Else( () => {

				fwd( i, j, k ).assign( input.data( i, j, k ) );

			} );

		} );

		const dispatchBackward = tsl_array_n.kernel( input.dataSize, ( i, j, k ) => {

			const pos = input.dataPosition( i, j, k );

			If( sampleBoundary( pos ).greaterThan( 0 ), () => {

				const tracedPos = trace( pos, -1 );
				bwd( i, j, k ).assign( collocatedCubicValueAtPosition3( fwd, input.gridSpacing, input.dataOrigin, tracedPos, input.dataSize ) );

			} ).Else( () => {

				bwd( i, j, k ).assign( fwd( i, j, k ) );

			} );

		} );

		const dispatchCorrectAndClamp = tsl_array_n.kernel( input.dataSize, ( i, j, k ) => {

			const pos = input.dataPosition( i, j, k );

			If( sampleBoundary( pos ).greaterThan( 0 ), () => {

				const tracedPos = trace( pos );
				const fwdValue = fwd( i, j, k );
				const correctedValue = fwdValue.add( input.data( i, j, k ).sub( bwd( i, j, k ) ).mul( 0.5 ) );
				const clamped = clampCorrection( correctedValue, fwdValue, input.data, input.dataOrigin, input.gridSpacing, input.dataSize, input.dataPosition, tracedPos );

				output.data( i, j, k ).assign( clamped );

			} );

		} );

		return tsl_array_n.createBatch( [ dispatchForward, dispatchBackward, dispatchCorrectAndClamp ] );

	}

	return { advectFaceCentered3, advectScalar3, trace };

}
