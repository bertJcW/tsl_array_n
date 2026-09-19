// 3D counterpart of grid_outflow_solver2.js -- same three-part outflow
// mechanism (pressure Dirichlet ghost, convective velocity extrapolation,
// scalar-field cleanup). See that file's own header comment for parts 1 and
// 3 and for where the mechanism comes from; part 2, the velocity boundary
// condition, is NOT the same here and the rest of this comment is why.
//
// *** Part 2 is the standard convective (Orlanski) form, not mantaflow's
// own algebraic arrangement of it, and it takes its upstream direction the
// opposite way round from grid_outflow_solver2.js ***
//
// The 2D file transcribes mantaflow's `(vel - velPrev)/factor +
// vel(upstream)` and reads its upstream direction off the outflow SDF's own
// gradient as `pt - normalize(gradient) * spacing`. Both of those are
// wrong here, and the second one is wrong in 2D too -- it survives there
// because nothing has yet pushed it hard enough.
//
// The direction first. sandbox/outflow-gradient/ samples an outflow slab's
// own SDF and gradient along the axis, for the exact slabs
// examples/16-karman-vortex-street/ and
// examples/35-karman-vortex-street-3d/ build. The gradient is (-1, 0, 0) at
// every point, in 2D and 3D alike: phi decreases going downstream, so the
// gradient points back toward the fluid. That is not an accident of those
// two scenes, it follows from the padding convention every outflow in this
// package uses -- the slab is extended a long way past the domain edge (see
// any example's own OUTER_MARGIN) precisely so that the nearest surface to
// any point inside it is the fluid-facing one. So `pt - n * h` samples
// DOWNSTREAM of the face, and at a domain-edge outflow two cells deep that
// means the two exit faces sample each other and the outermost one samples
// itself after clamping. Neither of them is coupled to the interior at all.
//
// Measured, on examples/35-karman-vortex-street-3d/: the exit faces carry
// 1.15 while the inflow is fixed at 2.00, i.e. barely half the mass
// entering the domain is leaving it, and the pressure field absorbs the
// rest until it is a smooth global ramp and the solve gives up. The scene
// blew up at frame 532, deterministically, five runs out of five.
//
// The formula second, and this is why flipping the sign alone does not fix
// it -- tried, measured, worse. `(current - prevVel)/factor + upstream`
// adds an unbounded increment: whatever advection changed this frame comes
// back amplified by 1/factor on top of the upstream value, every frame.
// With the direction corrected the exit does track the interior (2.02
// against an interior 2.0 at frame 200) and then still drifts, because the
// increment compounds -- that run stopped converging by frame 300 and never
// recovered after frame 425.
//
// What is here instead is the textbook convective boundary condition,
// du/dt + U du/dx = 0 on the outflow boundary, discretised explicitly:
//
//     u <- u + c * (u_upstream - u),   c = clamp(U * dt / h, 0, 1)
//
// with U the bulk velocity's component along the OUTWARD normal, floored at
// zero so that inflow through an outflow face does nothing. That is a
// convex combination of two values the field already holds, so it cannot
// amplify anything, cannot exceed the range of its own inputs, and needs no
// previous-velocity term at all. At c = 1 it is a pure upwind copy and at
// c = 0 it leaves the face alone; this scene runs at c = 0.1, i.e. the exit
// relaxes toward the interior over about ten frames.
//
// This is a real difference from the 2D file rather than a port detail, and
// it is deliberately not applied there in the same change: examples 15 and
// 16 are long-run stable as they stand, and changing a boundary condition
// under them wants its own verification rather than riding along with this
// one.

import * as tsl_array_n from 'tsl_array_n';
import { float, max, clamp } from 'three/tsl';
import { isInsideSdf } from './level_set_utils.js';
import { createOutflowPressureDirichlet3 } from './sdf_inflow_outflow3.js';

// A defensive backstop, kept from the earlier formulation even though the
// update below is a convex combination and so cannot leave the range of its
// own inputs. It costs one instruction and it is the kind of guard this
// package has needed before.
const EXTRAPOLATED_VELOCITY_CLAMP = 1e3;

// options: see grid_outflow_solver2.js's own createGridOutflowSolver2
// header comment for velocityGrid/velocityPrev/outflows/dt/applyVelocityBC
// -- unchanged conventions, one more staggered component.
export function createGridOutflowSolver3( { velocityGrid, velocityPrev, outflows, dt, applyVelocityBC = true } ) {

	const list = Array.isArray( outflows ) ? outflows : [ outflows ];
	const dtNode = typeof dt === 'number' ? float( dt ) : dt;

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

						// n points from inside the outflow region toward its
						// nearest surface, which this package's own padding
						// convention makes the fluid-facing one -- so the
						// upstream neighbour is at `pt + n * h`, not `pt - n * h`.
						// See this file's own header comment for the measurement.
						const n = g.normalize();
						const upstreamPt = pt.add( n.mul( gridSpacingComponent ) );

						// The convection speed is the bulk velocity along the
						// OUTWARD normal, so `-n`, floored at zero: a face with
						// flow coming back in through it is left alone rather
						// than pulled toward a neighbour it is not downstream of.
						const bulkVel = velocityGrid.sample( pt );
						const outwardSpeed = max( float( 0 ), bulkVel.dot( n.negate() ) );
						const courant = clamp( outwardSpeed.mul( dtNode ).div( gridSpacingComponent ), float( 0 ), float( 1 ) );

						const upstreamVel = velocityGrid.sample( upstreamPt );
						const current = dataComponent( i, j, k );

						const relaxed = current.add( upstreamVel[ component ].sub( current ).mul( courant ) );
						dst( i, j, k ).assign( clamp( relaxed, - EXTRAPOLATED_VELOCITY_CLAMP, EXTRAPOLATED_VELOCITY_CLAMP ) );

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
