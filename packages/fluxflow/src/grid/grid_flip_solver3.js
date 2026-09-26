// 3D counterpart of grid_flip_solver2.js -- same mantaflow-derived FLIP
// design (mapPartsToMAC / markFluidCells / flipVelocityUpdate /
// sampleFlagsWithParticles / extrapolateMACFromWeight/Simple), same scope
// (fixed particle count, no level set, particle occupancy alone decides
// fluid cells), same reduced-pressure substitution, same variable-density
// coupling, same particle resampling, same velocity damping, same carried
// concentration (dye). See that file's own extensive header comment for
// the full derivation and design reasoning behind every mechanism named
// above -- essentially none of it is dimension-specific math, it is
// dimension-specific *shape*: a third staggered velocity component (W), a
// third position/velocity axis per particle (vec3, not vec2), 8-corner
// trilinear P2G/G2P weights instead of 4-corner bilinear, and 6-face-
// neighbour occupancy/adjacency checks instead of 4. Every place this file
// deviates from a literal mechanical extension of the 2D file is called
// out at that spot below.
//
// *** The one judgement call with no 2D precedent to mirror: the
// resampling guard's MIN_FLUID_NEIGHBORS threshold ***
//
// grid_flip_solver2.js's own claimDonorsKernel only reseeds an under-full
// cell when at least 2 of its 4 face-connected neighbours already meet
// minParticlesPerCell -- a value grid_flip_solver2.js's own header comment
// says was found necessary by testing (an unguarded version measurably
// grew a scene's apparent fluid footprint on real hardware), not derived
// from first principles. There is no 3D measurement to carry that number
// forward from, so this file scales it by the same "half of the
// neighbours" ratio the 2D value represents (2 of 4) applied to 3D's 6
// face neighbours, i.e. 3 of 6 -- a judgement call, not a verified
// constant, flagged here rather than silently presented as equally
// well-tested. Worth re-measuring against a real 3D scene the same way
// grid_flip_solver2.js's own value was, if resampling ever misbehaves in
// 3D.

import * as tsl_array_n from 'tsl_array_n';
import { vec3, float, int, floor, round, clamp, max, min, atomicAdd, atomicLoad, If, Loop, Break } from 'three/tsl';
import { createGridBlockedBoundaryConditionSolver3 } from './grid_blocked_boundary_condition_solver3.js';
import { createGridPressureSolver3 } from './grid_pressure_solver3.js';
import { createSemiLagrangianAdvectionSolver3 } from './advection_solver3.js';
import { createCopyKernel3, createExtrapolateToRegion3 } from './array_utils3.js';
import { trilinearCoordsAndWeights3, collocatedValueAtPosition3, faceCenteredValueAtPosition3 } from './grid_math3.js';
import { DEFAULT_ATOMIC_DOT_SCALE } from '../linalg/linalg.js';
import { createInclusivePrefixSum } from '../linalg/prefix_sum.js';
import { isNonFinite, isNonFinite3 } from '../float_guards.js';
import { createJitterRandom, DEFAULT_JITTER_SEED } from './jitter_random.js';

// See grid_flip_solver2.js's own header comment for the meaning of both
// constants -- unchanged.
const PRESSURE_BOUND_HEADROOM = 8;

const MAX_PARTICLE_VELOCITY = 500;

// See grid_flip_solver2.js's own DEFAULT_CONCENTRATION_ATOMIC_SCALE
// comment -- unchanged.
const DEFAULT_CONCENTRATION_ATOMIC_SCALE = 4096;

// See grid_flip_solver2.js's own MIN_DENSITY comment -- unchanged.
const MIN_DENSITY = 1e-4;

function numberOrNode( value ) {

	return typeof value === 'number' ? float( value ) : value;

}

// See computeFlipBoxSeed's own header comment (grid_flip_solver2.js) for
// the full design -- unchanged except particlesPerCellAxis^3 particles per
// cell (was ^2), jittered in 3 axes.
// options.boxMin/boxMax: [x,y,z] world-space corners. gridSpacingX/Y/Z:
// this solver's own gridSpacing.
export function computeFlipBoxSeed3( { boxMin, boxMax, gridSpacingX, gridSpacingY, gridSpacingZ, particlesPerCellAxis = 2, jitter = 0.2, randomSeed = DEFAULT_JITTER_SEED } ) {

	const [ minX, minY, minZ ] = boxMin;
	const [ maxX, maxY, maxZ ] = boxMax;

	const cellsX = Math.max( 1, Math.round( ( maxX - minX ) / gridSpacingX ) );
	const cellsY = Math.max( 1, Math.round( ( maxY - minY ) / gridSpacingY ) );
	const cellsZ = Math.max( 1, Math.round( ( maxZ - minZ ) / gridSpacingZ ) );
	const subSpacingX = gridSpacingX / particlesPerCellAxis;
	const subSpacingY = gridSpacingY / particlesPerCellAxis;
	const subSpacingZ = gridSpacingZ / particlesPerCellAxis;

	// Deterministic by default -- see jitter_random.js for what this
	// replaced and the measurements that found it.
	const random = createJitterRandom( randomSeed );

	const positionsList = [];

	for ( let cellK = 0; cellK < cellsZ; cellK ++ ) {

		for ( let cellJ = 0; cellJ < cellsY; cellJ ++ ) {

			for ( let cellI = 0; cellI < cellsX; cellI ++ ) {

				for ( let subK = 0; subK < particlesPerCellAxis; subK ++ ) {

					for ( let subJ = 0; subJ < particlesPerCellAxis; subJ ++ ) {

						for ( let subI = 0; subI < particlesPerCellAxis; subI ++ ) {

							const jx = ( random() * 2 - 1 ) * jitter * subSpacingX;
							const jy = ( random() * 2 - 1 ) * jitter * subSpacingY;
							const jz = ( random() * 2 - 1 ) * jitter * subSpacingZ;

							const x = minX + cellI * gridSpacingX + ( subI + 0.5 ) * subSpacingX + jx;
							const y = minY + cellJ * gridSpacingY + ( subJ + 0.5 ) * subSpacingY + jy;
							const z = minZ + cellK * gridSpacingZ + ( subK + 0.5 ) * subSpacingZ + jz;

							positionsList.push( x, y, z );

						}

					}

				}

			}

		}

	}

	const count = positionsList.length / 3;

	return {
		count,
		positionsArray: new Float32Array( positionsList ),
		velocitiesArray: new Float32Array( count * 3 ) // zeroed -- particles start at rest
	};

}

// options: see grid_flip_solver2.js's own createGridFlipSolver2 header
// comment for the full parameter list's meaning -- unchanged, resolution/
// gridSpacing/origin/gravity are now triples.
export function createGridFlipSolver3( {
	velocityGrid,
	gridSpacing = [ 1, 1, 1 ],
	origin = [ 0, 0, 0 ],
	maxParticles,
	dt,
	gravity = [ 0, -9.81, 0 ],
	reducedPressure = true,
	massWeightedTransfer = false,
	applyForces,
	maxDt,
	flipRatio = 0.97,
	velocityDamping = 0.02,
	p2gAtomicScale = DEFAULT_ATOMIC_DOT_SCALE,
	weightEpsilon = 1e-4,
	closedDomainBoundaryFlag,
	collider,
	colliderPushThresh = 0,
	colliderPushShift = 0,
	resample = {},
	pressure = {},
	// ---------------------------------------------------------------- dye
	carryConcentration = false,
	mixing = 0,
	fade = 0,
	concentrationAtomicScale = DEFAULT_CONCENTRATION_ATOMIC_SCALE,
	ambientDensity,
	componentDensity
} = {} ) {

	if ( ! velocityGrid ) {

		throw new Error( 'createGridFlipSolver3: options.velocityGrid is required.' );

	}

	if ( ! maxParticles ) {

		throw new Error( 'createGridFlipSolver3: options.maxParticles is required.' );

	}

	const {
		enabled: resampleEnabled = true,
		minParticlesPerCell = 4,
		maxParticlesPerCell = 8
	} = resample;

	const [ resolutionX, resolutionY, resolutionZ ] = velocityGrid.resolution;
	const [ gridSpacingX, gridSpacingY, gridSpacingZ ] = gridSpacing;
	const [ originX, originY, originZ ] = origin;

	const gridSpacingNode = vec3( gridSpacingX, gridSpacingY, gridSpacingZ );
	const originNode = vec3( originX, originY, originZ );
	const cellCenterOrigin = originNode.add( gridSpacingNode.mul( 0.5 ) );

	const dtNode = numberOrNode( dt );
	const gravityNode = vec3( gravity[ 0 ], gravity[ 1 ], gravity[ 2 ] );
	const cellShape = [ resolutionX, resolutionY, resolutionZ ];
	const cellCount = resolutionX * resolutionY * resolutionZ;

	const boundarySolver = createGridBlockedBoundaryConditionSolver3(
		velocityGrid, resolutionX, resolutionY, resolutionZ, gridSpacingX, gridSpacingY, gridSpacingZ, originX, originY, originZ, collider
	);
	if ( closedDomainBoundaryFlag !== undefined ) boundarySolver.closedDomainBoundaryFlag = closedDomainBoundaryFlag;

	const dataSizeU = velocityGrid.dataSizeU;
	const dataSizeV = velocityGrid.dataSizeV;
	const dataSizeW = velocityGrid.dataSizeW;
	const uCount = dataSizeU[ 0 ] * dataSizeU[ 1 ] * dataSizeU[ 2 ];
	const vCount = dataSizeV[ 0 ] * dataSizeV[ 1 ] * dataSizeV[ 2 ];
	const wCount = dataSizeW[ 0 ] * dataSizeW[ 1 ] * dataSizeW[ 2 ];

	// See grid_flip_solver2.js's own comment on this same line -- unchanged
	// reasoning, "explicit clear, don't rely on implicit zero-init".
	const positions = tsl_array_n.arrayN( 'vec3', maxParticles );
	const velocities = tsl_array_n.arrayN( 'vec3', maxParticles );
	positions.fromArray( new Float32Array( maxParticles * 3 ) );
	velocities.fromArray( new Float32Array( maxParticles * 3 ) );

	const fluidMask = tsl_array_n.arrayN( 'float', cellShape );
	fluidMask.fromArray( new Float32Array( cellCount ) );

	const clearFluidMask = tsl_array_n.kernel( cellShape, ( i, j, k ) => {

		fluidMask( i, j, k ).assign( 0 );

	} );

	// Shared by markFluidCellsKernel below and the resample kernels further
	// down -- same "which cell is this continuous position in" formula as
	// grid_flip_solver2.js's own cellIndexOf, one more axis.
	function cellIndexOf( pos ) {

		const cellF = pos.sub( originNode ).div( gridSpacingNode );
		const i = max( 0, min( int( floor( cellF.x ) ), resolutionX - 1 ) );
		const j = max( 0, min( int( floor( cellF.y ) ), resolutionY - 1 ) );
		const k = max( 0, min( int( floor( cellF.z ) ), resolutionZ - 1 ) );
		return { i, j, k };

	}

	const markFluidCellsKernel = tsl_array_n.kernel( maxParticles, ( p ) => {

		const { i, j, k } = cellIndexOf( positions( p ) );

		// Same-value race, not a data hazard -- see grid_flip_solver2.js's
		// own header comment.
		fluidMask( i, j, k ).assign( 1 );

	} );

	// ---- reduced pressure: see grid_flip_solver2.js's own dirichlet()
	// header comment for the full derivation -- unchanged (gravityNode.dot(pos)
	// is a vec3 dot, same formula, one more axis contributes to it).
	const reducedPressureEnabled = reducedPressure !== false;

	function dirichlet( pos ) {

		const mask = collocatedValueAtPosition3( fluidMask, gridSpacingNode, cellCenterOrigin, pos, cellShape );
		const target = reducedPressureEnabled ? gravityNode.dot( pos ).mul( dtNode ).negate() : float( 0 );

		return { active: mask.lessThan( 0.5 ), target };

	}

	// ---------------------------------------------------------------- variable density
	// See grid_flip_solver2.js's own header comment (right above the same
	// declaration) for the full design -- unchanged, one more staggered
	// component (betaW).
	const densityCouplingEnabled = ambientDensity !== undefined && componentDensity !== undefined;

	if ( densityCouplingEnabled && ! carryConcentration ) {

		throw new Error( 'createGridFlipSolver3: ambientDensity/componentDensity require carryConcentration: true -- there is no concentration field to build a density from otherwise.' );

	}

	if ( densityCouplingEnabled ) {

		for ( const [ name, value ] of [ [ 'ambientDensity', ambientDensity ], [ 'componentDensity', componentDensity ] ] ) {

			if ( typeof value === 'number' && ! ( value > 0 ) ) {

				throw new Error( `createGridFlipSolver3: ${ name } must be > 0, got ${ value }.` );

			}

		}

	}

	let cellDensity = null;
	let betaU = null;
	let betaV = null;
	let betaW = null;
	let computeBetaU = null;
	let computeBetaV = null;
	let computeBetaW = null;
	let densityAtZeroNode = null;
	let densityAtOneNode = null;
	let referenceDensityNode = null;

	if ( densityCouplingEnabled ) {

		const minDensityNode = float( MIN_DENSITY );
		densityAtZeroNode = max( numberOrNode( ambientDensity ), minDensityNode );
		densityAtOneNode = max( numberOrNode( componentDensity ), minDensityNode );
		referenceDensityNode = max( densityAtZeroNode, densityAtOneNode );

		cellDensity = tsl_array_n.arrayN( 'float', cellShape );
		cellDensity.fromArray( new Float32Array( cellCount ) );

		betaU = tsl_array_n.arrayN( 'float', dataSizeU );
		betaV = tsl_array_n.arrayN( 'float', dataSizeV );
		betaW = tsl_array_n.arrayN( 'float', dataSizeW );
		betaU.fromArray( new Float32Array( uCount ).fill( 1 ) );
		betaV.fromArray( new Float32Array( vCount ).fill( 1 ) );
		betaW.fromArray( new Float32Array( wCount ).fill( 1 ) );

		const clampI = ( i ) => max( 0, min( i, resolutionX - 1 ) );
		const clampJ = ( j ) => max( 0, min( j, resolutionY - 1 ) );
		const clampK = ( k ) => max( 0, min( k, resolutionZ - 1 ) );

		// See grid_flip_solver2.js's own faceDensity comment -- unchanged
		// one-sided-at-the-surface reasoning, one more axis.
		function faceDensity( iA, jA, kA, iB, jB, kB ) {

			const rhoA = cellDensity( iA, jA, kA );
			const rhoB = cellDensity( iB, jB, kB );
			const aFluid = fluidMask( iA, jA, kA ).greaterThan( 0.5 );
			const bFluid = fluidMask( iB, jB, kB ).greaterThan( 0.5 );

			const bothFluid = rhoA.add( rhoB ).mul( 0.5 );
			const oneFluid = aFluid.select( rhoA, rhoB );

			return aFluid.and( bFluid ).select(
				bothFluid,
				aFluid.or( bFluid ).select( oneFluid, referenceDensityNode )
			);

		}

		computeBetaU = tsl_array_n.kernel( dataSizeU, ( i, j, k ) => {

			const rhoFace = faceDensity( clampI( i.sub( 1 ) ), j, k, clampI( i ), j, k );
			betaU( i, j, k ).assign( referenceDensityNode.div( max( rhoFace, minDensityNode ) ) );

		} );

		computeBetaV = tsl_array_n.kernel( dataSizeV, ( i, j, k ) => {

			const rhoFace = faceDensity( i, clampJ( j.sub( 1 ) ), k, i, clampJ( j ), k );
			betaV( i, j, k ).assign( referenceDensityNode.div( max( rhoFace, minDensityNode ) ) );

		} );

		computeBetaW = tsl_array_n.kernel( dataSizeW, ( i, j, k ) => {

			const rhoFace = faceDensity( i, j, clampK( k.sub( 1 ) ), i, j, clampK( k ) );
			betaW( i, j, k ).assign( referenceDensityNode.div( max( rhoFace, minDensityNode ) ) );

		} );

	}

	// ---- a plausibility bound derived from the scene -- see
	// grid_flip_solver2.js's own header comment for the full reasoning.
	// domainExtent now sums all 3 axes.
	//
	// *** Confirmed on real hardware: this formula only bounds the
	// HYDROSTATIC part of the pressure field, and a violent enough 3D
	// splash routinely exceeds it -- this is a real, measured limitation,
	// not a hypothetical one. ***
	//
	// The derivation (see the header comment above dirichlet()) is exact
	// for the reduced-pressure substitution's own air-cell Dirichlet
	// target, which scales with `dt * |g| * domainExtent` alone -- it has
	// no velocity/dynamic-pressure term at all. A dam-break block hitting
	// the floor generates real dynamic (impact) pressure on top of that,
	// and this formula has no way to know about it.
	//
	// Root-caused via a real-hardware diagnostic harness (many paired runs
	// at N=8 and N=20, varying maxPlausiblePressure/preconditioner/
	// resample/reducedPressure/atomicScale independently) after a user
	// report of a periodic "collapse then explode" cycle in
	// examples/33-flip-dam-break-3d/: the derived bound (~38-94 for those
	// scenes) rejected pressure solves whose true peak was legitimately
	// higher than that, and each rejection reverts to a stale pressure
	// snapshot while the simulation keeps evolving -- so the *next* frame's
	// divergence is larger still, is rejected again, and the cycle
	// compounds into the visible blowup. Ruling hypotheses in and out, not
	// just plausible: disabling resampling and setting `atomicScale` (a
	// no-op in the current dot-product implementation -- see linalg.js's
	// own header comment, decision 1) left the pattern unchanged.
	// Loosening maxPlausiblePressure alone measurably reduced but did not
	// eliminate it (1.73% rejected over a 3000-frame run at
	// maxPlausiblePressure: 5000 vs the derived ~94 for a 20^3 domain) --
	// a real, recurring, genuine CG convergence difficulty independent of
	// the bound remained.
	//
	// *** The actual remaining cause, found afterward: this solver's
	// pressure defaults to numberOfLevels:1 (see multigrid.js's own header
	// comment on that default), i.e. no real coarse-grid V-cycle at all --
	// an EARLIER attempt to test a higher level count used numberOfLevels:4
	// on a 20^3 domain and hit grid_pressure_solver3.js's own construction-
	// time divisibility guard (20 is not divisible by 2^(4-1)=8), which
	// read at the time as "raising the multigrid level count doesn't fix
	// it" -- it hadn't actually been tested. numberOfLevels:3 (the most a
	// 20^3 domain evenly supports, since 20 is divisible by 2^(3-1)=4) DOES
	// fix it: 0/3000 rejected with maxPlausiblePressure left at the same
	// 5000, confirmed again at 0/12000 over the same full-length run the
	// original bug was measured against, and does not need
	// maxPlausiblePressure loosened any further than that first fix
	// already established. This solver's own irregular, splash-driven
	// fluidMask topology (a violent collapse creates many small,
	// disconnected fluid/air regions every frame) is, in the end, simply
	// harder to precondition than the finest-level-only smoother alone can
	// handle -- not a bound problem, a preconditioner-strength one. ***
	//
	// No universal fix is applied here (this derivation still activates by
	// default, per its own existing behavior below) -- a caller whose scene
	// generates significant dynamic pressure (fast impacts, not just
	// hydrostatic settling) should measure their own scene's real peak and
	// pass an explicit `pressure.maxPlausiblePressure`, and a caller whose
	// scene develops a similarly irregular Dirichlet topology (a violent
	// free-surface breakup, not just a mild one) should consider raising
	// `pressure.multigrid.numberOfLevels` past its own default of 1, sized
	// to what their own domain resolution evenly supports. See
	// examples/33-flip-dam-break-3d/'s own header comment for the numbers
	// that scene measured for both.
	const domainExtent = resolutionX * gridSpacingX + resolutionY * gridSpacingY + resolutionZ * gridSpacingZ;
	const gravityMagnitude = Math.hypot( gravity[ 0 ], gravity[ 1 ], gravity[ 2 ] );
	const boundDt = typeof maxDt === 'number' ? maxDt : ( typeof dt === 'number' ? dt : undefined );

	const derivedMaxPlausiblePressure = ( boundDt !== undefined && reducedPressureEnabled && gravityMagnitude > 0 )
		? PRESSURE_BOUND_HEADROOM * boundDt * gravityMagnitude * domainExtent
		: undefined;

	const pressureSolver = createGridPressureSolver3( {
		resolution: [ resolutionX, resolutionY, resolutionZ ], gridSpacing, origin, dirichlet,
		...( densityCouplingEnabled ? { faceWeights: { u: betaU, v: betaV, w: betaW } } : {} ),
		...( derivedMaxPlausiblePressure !== undefined ? { maxPlausiblePressure: derivedMaxPlausiblePressure } : {} ),
		...pressure
	} );
	const projectDispatch = pressureSolver.project( velocityGrid, velocityGrid );

	// ---- P2G: one atomic accumulator pair per face component -- see
	// grid_flip_solver2.js's own header comment for the full design.

	const uNumerAccum = tsl_array_n.arrayN( 'int', dataSizeU );
	uNumerAccum.node.toAtomic();
	const uDenomAccum = tsl_array_n.arrayN( 'int', dataSizeU );
	uDenomAccum.node.toAtomic();
	const vNumerAccum = tsl_array_n.arrayN( 'int', dataSizeV );
	vNumerAccum.node.toAtomic();
	const vDenomAccum = tsl_array_n.arrayN( 'int', dataSizeV );
	vDenomAccum.node.toAtomic();
	const wNumerAccum = tsl_array_n.arrayN( 'int', dataSizeW );
	wNumerAccum.node.toAtomic();
	const wDenomAccum = tsl_array_n.arrayN( 'int', dataSizeW );
	wDenomAccum.node.toAtomic();

	const zeroU = new Int32Array( uCount );
	const zeroV = new Int32Array( vCount );
	const zeroW = new Int32Array( wCount );

	function resetAccumulators() {

		uNumerAccum.fromArray( zeroU );
		uDenomAccum.fromArray( zeroU );
		vNumerAccum.fromArray( zeroV );
		vDenomAccum.fromArray( zeroV );
		wNumerAccum.fromArray( zeroW );
		wDenomAccum.fromArray( zeroW );

	}

	const uWeightValid = tsl_array_n.arrayN( 'int', dataSizeU );
	const vWeightValid = tsl_array_n.arrayN( 'int', dataSizeV );
	const wWeightValid = tsl_array_n.arrayN( 'int', dataSizeW );

	const p2gScaleNode = numberOrNode( p2gAtomicScale );
	const weightEpsilonNode = numberOrNode( weightEpsilon );

	let concentration = null;

	if ( carryConcentration ) {

		concentration = tsl_array_n.arrayN( 'float', maxParticles );
		concentration.fromArray( new Float32Array( maxParticles ) );

	}

	// ---- particle mass -- see grid_flip_solver2.js's own massWeight header
	// comment (the long one, "off by default...") for the full measurement
	// and why this option does not work -- unchanged reasoning, kept for
	// the same one-URL-parameter re-run purpose.
	const massWeightingEnabled = massWeightedTransfer && densityCouplingEnabled && carryConcentration;

	function massWeight( p ) {

		if ( ! massWeightingEnabled ) return null;

		const c = clamp( concentration( p ), float( 0 ), float( 1 ) );
		const rho = densityAtZeroNode.add( densityAtOneNode.sub( densityAtZeroNode ).mul( c ) );

		return rho.div( referenceDensityNode );

	}

	// component: 'x', 'y' or 'z' -- see grid_flip_solver2.js's own
	// buildScatter comment. 8 corners (trilinear) instead of 4 (bilinear).
	function buildScatter( component, dataOrigin, size, numerAccum, denomAccum ) {

		return tsl_array_n.kernel( maxParticles, ( p ) => {

			const value = component === 'x' ? velocities( p ).x : component === 'y' ? velocities( p ).y : velocities( p ).z;
			const { i0c, j0c, k0c, i1c, j1c, k1c, w000, w100, w010, w110, w001, w101, w011, w111 } =
				trilinearCoordsAndWeights3( positions( p ), dataOrigin, velocityGrid.gridSpacing, size );

			const corners = [
				[ i0c, j0c, k0c, w000 ], [ i1c, j0c, k0c, w100 ],
				[ i0c, j1c, k0c, w010 ], [ i1c, j1c, k0c, w110 ],
				[ i0c, j0c, k1c, w001 ], [ i1c, j0c, k1c, w101 ],
				[ i0c, j1c, k1c, w011 ], [ i1c, j1c, k1c, w111 ]
			];
			const mass = massWeight( p );

			for ( const [ i, j, k, wRaw ] of corners ) {

				const w = mass ? wRaw.mul( mass ) : wRaw;

				atomicAdd( numerAccum( i, j, k ), round( value.mul( w ).mul( p2gScaleNode ) ).toInt() );
				atomicAdd( denomAccum( i, j, k ), round( w.mul( p2gScaleNode ) ).toInt() );

			}

		} );

	}

	const scatterU = buildScatter( 'x', velocityGrid.dataOriginU, dataSizeU, uNumerAccum, uDenomAccum );
	const scatterV = buildScatter( 'y', velocityGrid.dataOriginV, dataSizeV, vNumerAccum, vDenomAccum );
	const scatterW = buildScatter( 'z', velocityGrid.dataOriginW, dataSizeW, wNumerAccum, wDenomAccum );

	// One thread per destination face -- see grid_flip_solver2.js's own
	// buildFinalize comment.
	function buildFinalize( dataComponent, size, numerAccum, denomAccum, weightValid ) {

		return tsl_array_n.kernel( size, ( i, j, k ) => {

			const numer = atomicLoad( numerAccum( i, j, k ) ).toFloat().div( p2gScaleNode );
			const denom = atomicLoad( denomAccum( i, j, k ) ).toFloat().div( p2gScaleNode );

			If( denom.greaterThan( weightEpsilonNode ), () => {

				dataComponent( i, j, k ).assign( numer.div( denom ) );
				weightValid( i, j, k ).assign( 1 );

			} ).Else( () => {

				dataComponent( i, j, k ).assign( 0 );
				weightValid( i, j, k ).assign( 0 );

			} );

		} );

	}

	const finalizeU = buildFinalize( velocityGrid.dataU, dataSizeU, uNumerAccum, uDenomAccum, uWeightValid );
	const finalizeV = buildFinalize( velocityGrid.dataV, dataSizeV, vNumerAccum, vDenomAccum, vWeightValid );
	const finalizeW = buildFinalize( velocityGrid.dataW, dataSizeW, wNumerAccum, wDenomAccum, wWeightValid );

	// ---- extrapolation -- see grid_flip_solver2.js's own comment. Two
	// passes per mantaflow's own extrapolateMACFromWeight/Simple.

	const extrapolateWeightU = createExtrapolateToRegion3( velocityGrid.dataU, uWeightValid, velocityGrid.dataU, dataSizeU );
	const extrapolateWeightV = createExtrapolateToRegion3( velocityGrid.dataV, vWeightValid, velocityGrid.dataV, dataSizeV );
	const extrapolateWeightW = createExtrapolateToRegion3( velocityGrid.dataW, wWeightValid, velocityGrid.dataW, dataSizeW );

	const uFluidAdjacent = tsl_array_n.arrayN( 'int', dataSizeU );
	const vFluidAdjacent = tsl_array_n.arrayN( 'int', dataSizeV );
	const wFluidAdjacent = tsl_array_n.arrayN( 'int', dataSizeW );

	// A face is "adjacent to fluid" if either of its two neighbouring
	// cells (clamped at the domain edge) is currently marked fluid -- see
	// grid_flip_solver2.js's own comment, one more axis.
	const computeUFluidAdjacent = tsl_array_n.kernel( dataSizeU, ( i, j, k ) => {

		const loIdx = max( 0, min( i.sub( 1 ), resolutionX - 1 ) );
		const hiIdx = max( 0, min( i, resolutionX - 1 ) );
		const anyFluid = fluidMask( loIdx, j, k ).greaterThan( 0.5 ).or( fluidMask( hiIdx, j, k ).greaterThan( 0.5 ) );

		uFluidAdjacent( i, j, k ).assign( anyFluid.select( int( 1 ), int( 0 ) ) );

	} );

	const computeVFluidAdjacent = tsl_array_n.kernel( dataSizeV, ( i, j, k ) => {

		const loIdx = max( 0, min( j.sub( 1 ), resolutionY - 1 ) );
		const hiIdx = max( 0, min( j, resolutionY - 1 ) );
		const anyFluid = fluidMask( i, loIdx, k ).greaterThan( 0.5 ).or( fluidMask( i, hiIdx, k ).greaterThan( 0.5 ) );

		vFluidAdjacent( i, j, k ).assign( anyFluid.select( int( 1 ), int( 0 ) ) );

	} );

	const computeWFluidAdjacent = tsl_array_n.kernel( dataSizeW, ( i, j, k ) => {

		const loIdx = max( 0, min( k.sub( 1 ), resolutionZ - 1 ) );
		const hiIdx = max( 0, min( k, resolutionZ - 1 ) );
		const anyFluid = fluidMask( i, j, loIdx ).greaterThan( 0.5 ).or( fluidMask( i, j, hiIdx ).greaterThan( 0.5 ) );

		wFluidAdjacent( i, j, k ).assign( anyFluid.select( int( 1 ), int( 0 ) ) );

	} );

	const extrapolatePressureU = createExtrapolateToRegion3( velocityGrid.dataU, uFluidAdjacent, velocityGrid.dataU, dataSizeU );
	const extrapolatePressureV = createExtrapolateToRegion3( velocityGrid.dataV, vFluidAdjacent, velocityGrid.dataV, dataSizeV );
	const extrapolatePressureW = createExtrapolateToRegion3( velocityGrid.dataW, wFluidAdjacent, velocityGrid.dataW, dataSizeW );

	// ---- particle resampling -- see grid_flip_solver2.js's own header
	// comment (the "Particle resampling" section) for the full design.

	let resamplePass = null;

	if ( resampleEnabled ) {

		const cellParticleCount = tsl_array_n.arrayN( 'int', cellShape );
		cellParticleCount.node.toAtomic();

		// *** Ranks from a prefix sum, not slots from an atomic cursor ***
		//
		// Both cursors this replaces did exactly what they were verified to
		// do -- hand every claimant a unique slot -- and uniqueness is not
		// reproducibility. Which particle got slot 0 depended on which thread
		// reached the atomic first, so each run teleported a different set of
		// particles and the trajectories parted. Measured on
		// examples/33-flip-dam-break-3d/: four 12,000-step runs of one build
		// gave two different verdicts, with sampled fields already differing
		// by frame 300. A scene that cannot be run twice cannot be bisected,
		// which makes every future measurement of it weaker.
		//
		// A rank defined by index is reproducible. donorRank is the prefix sum
		// of a 0/1 donor flag, so a donor's slot is the number of donors
		// before it in particle order; claimBase is the prefix sum of each
		// cell's need, so a cell's slice of the pool is fixed by cell order.
		// Nothing in either depends on execution order.
		//
		// What this changes and what it does not: the donor SET, the receiving
		// cells, and what happens to a relocated particle are all exactly as
		// before. Only the assignment between them becomes canonical -- lower
		// particle index donates first, lower cell index claims first. The
		// trajectory therefore differs from the old code's, necessarily; it is
		// the same algorithm with its one arbitrary choice made the same way
		// every time.
		const donorPool = tsl_array_n.arrayN( 'int', maxParticles );
		const donorFlag = tsl_array_n.arrayN( 'int', maxParticles );
		const cellNeed = tsl_array_n.arrayN( 'int', cellCount );

		const donorScan = createInclusivePrefixSum( maxParticles, donorFlag );
		const needScan = createInclusivePrefixSum( cellCount, cellNeed );

		const zeroCells = new Int32Array( cellCount );

		// donorFlag and cellNeed are written in full by their own kernels
		// every frame, so only the atomic accumulator needs clearing.
		function resetResampleBuffers() {

			cellParticleCount.fromArray( zeroCells );

		}

		// Cell arrays here are 3-D but a prefix sum is 1-D, so cellNeed and
		// its scan are flat and indexed the same way every other flattening
		// in this package is.
		const cellFlatIndex = ( i, j, k ) => i.add( j.mul( resolutionX ) ).add( k.mul( resolutionX * resolutionY ) );

		const countPerCellKernel = tsl_array_n.kernel( maxParticles, ( p ) => {

			const { i, j, k } = cellIndexOf( positions( p ) );
			atomicAdd( cellParticleCount( i, j, k ), 1 );

		} );

		const maxParticlesPerCellNode = int( maxParticlesPerCell );

		// Who volunteers -- unchanged, and order-independent already, since
		// it only reads counts that integer atomics accumulated exactly.
		const markDonorsKernel = tsl_array_n.kernel( maxParticles, ( p ) => {

			const { i, j, k } = cellIndexOf( positions( p ) );
			const count = atomicLoad( cellParticleCount( i, j, k ) );

			donorFlag( p ).assign( count.greaterThan( maxParticlesPerCellNode ).select( int( 1 ), int( 0 ) ) );

		} );

		// Where each volunteer goes: its rank among the volunteers before it.
		// The inclusive scan is 1-based, hence the subtraction.
		const scatterDonorsKernel = tsl_array_n.kernel( maxParticles, ( p ) => {

			If( donorFlag( p ).equal( int( 1 ) ), () => {

				donorPool( donorScan.result( p ).sub( 1 ) ).assign( p );

			} );

		} );

		const minParticlesPerCellNode = int( minParticlesPerCell );

		// See this file's own header comment for why MIN_FLUID_NEIGHBORS is
		// 3 of 6 here rather than a directly-carried-over number -- 6-face-
		// connected neighbours instead of 2D's 4.
		function neighborFluidCount( i, j, k ) {

			const xLo = atomicLoad( cellParticleCount( max( 0, i.sub( 1 ) ), j, k ) );
			const xHi = atomicLoad( cellParticleCount( min( resolutionX - 1, i.add( 1 ) ), j, k ) );
			const yLo = atomicLoad( cellParticleCount( i, max( 0, j.sub( 1 ) ), k ) );
			const yHi = atomicLoad( cellParticleCount( i, min( resolutionY - 1, j.add( 1 ) ), k ) );
			const zLo = atomicLoad( cellParticleCount( i, j, max( 0, k.sub( 1 ) ) ) );
			const zHi = atomicLoad( cellParticleCount( i, j, min( resolutionZ - 1, k.add( 1 ) ) ) );

			return xLo.greaterThanEqual( minParticlesPerCellNode ).select( int( 1 ), int( 0 ) )
				.add( xHi.greaterThanEqual( minParticlesPerCellNode ).select( int( 1 ), int( 0 ) ) )
				.add( yLo.greaterThanEqual( minParticlesPerCellNode ).select( int( 1 ), int( 0 ) ) )
				.add( yHi.greaterThanEqual( minParticlesPerCellNode ).select( int( 1 ), int( 0 ) ) )
				.add( zLo.greaterThanEqual( minParticlesPerCellNode ).select( int( 1 ), int( 0 ) ) )
				.add( zHi.greaterThanEqual( minParticlesPerCellNode ).select( int( 1 ), int( 0 ) ) );

		}

		const MIN_FLUID_NEIGHBORS = 3;

		// How many each under-filled cell wants. Split out of the claim so the
		// scan below has something to sum; the condition is unchanged.
		const markNeedKernel = tsl_array_n.kernel( cellShape, ( i, j, k ) => {

			const count = atomicLoad( cellParticleCount( i, j, k ) );
			const wants = count.lessThan( minParticlesPerCellNode )
				.and( neighborFluidCount( i, j, k ).greaterThanEqual( MIN_FLUID_NEIGHBORS ) );

			cellNeed( cellFlatIndex( i, j, k ) ).assign( wants.select( minParticlesPerCellNode.sub( count ), int( 0 ) ) );

		} );

		const claimDonorsKernel = tsl_array_n.kernel( cellShape, ( i, j, k ) => {

			const flat = cellFlatIndex( i, j, k );
			const needed = cellNeed( flat );

			If( needed.greaterThan( int( 0 ) ), () => {

				// This cell's own slice of the pool: everything earlier cells
				// asked for comes first. No cursor, no contention, and the
				// same answer on every run -- when donors are scarce the lower
				// cell index wins, deterministically, where before it was
				// whichever thread arrived first.
				const base = needScan.result( flat ).sub( needed );
				const totalDonors = donorScan.result( maxParticles - 1 );
				const claimed = int( 0 ).toVar();

				Loop( minParticlesPerCell, () => {

					If( claimed.greaterThanEqual( needed ), () => {

						Break();

					} );

					const claimSlot = base.add( claimed );

					If( claimSlot.lessThan( totalDonors ), () => {

						const donorIdx = donorPool( claimSlot );
						const newPos = cellCenterOrigin.add( vec3( i, j, k ).mul( gridSpacingNode ) );

						positions( donorIdx ).assign( newPos );
						velocities( donorIdx ).assign( faceCenteredValueAtPosition3(
							velocityGrid.dataU, velocityGrid.dataV, velocityGrid.dataW, velocityGrid.gridSpacing,
							velocityGrid.dataOriginU, velocityGrid.dataOriginV, velocityGrid.dataOriginW, newPos, dataSizeU, dataSizeV, dataSizeW
						) );

						claimed.addAssign( 1 );

					} ).Else( () => {

						Break();

					} );

				} );

			} );

		} );

		// See grid_flip_solver2.js's own resampleBatch comment for why this
		// is one submission rather than seven.
		const resampleBatch = tsl_array_n.createBatch( [
			countPerCellKernel,
			markDonorsKernel,
			...donorScan.kernels,
			scatterDonorsKernel,
			markNeedKernel,
			...needScan.kernels,
			claimDonorsKernel,

			clearFluidMask,
			markFluidCellsKernel,
			computeUFluidAdjacent,
			computeVFluidAdjacent,
			computeWFluidAdjacent
		] );

		resamplePass = function resample() {

			resetResampleBuffers();
			resampleBatch();

		};

	}

	// ---- gravity -- see grid_flip_solver2.js's own comment.

	const applyGravityU = tsl_array_n.kernel( dataSizeU, ( i, j, k ) => {

		velocityGrid.dataU( i, j, k ).addAssign( gravityNode.x.mul( dtNode ) );

	} );

	const applyGravityV = tsl_array_n.kernel( dataSizeV, ( i, j, k ) => {

		velocityGrid.dataV( i, j, k ).addAssign( gravityNode.y.mul( dtNode ) );

	} );

	const applyGravityW = tsl_array_n.kernel( dataSizeW, ( i, j, k ) => {

		velocityGrid.dataW( i, j, k ).addAssign( gravityNode.z.mul( dtNode ) );

	} );

	// ---- reduced-pressure replacement for the three kernels above -- see
	// grid_flip_solver2.js's own header comment for the full derivation.
	let applyReducedGravityU = null;
	let applyReducedGravityV = null;
	let applyReducedGravityW = null;

	if ( reducedPressureEnabled && densityCouplingEnabled ) {

		const clampI2 = ( i ) => max( 0, min( i, resolutionX - 1 ) );
		const clampJ2 = ( j ) => max( 0, min( j, resolutionY - 1 ) );
		const clampK2 = ( k ) => max( 0, min( k, resolutionZ - 1 ) );

		applyReducedGravityU = tsl_array_n.kernel( dataSizeU, ( i, j, k ) => {

			const pos = originNode.add( vec3( float( i ).mul( gridSpacingNode.x ), float( j ).add( 0.5 ).mul( gridSpacingNode.y ), float( k ).add( 0.5 ).mul( gridSpacingNode.z ) ) );
			const dRho = cellDensity( clampI2( i ), j, k ).sub( cellDensity( clampI2( i.sub( 1 ) ), j, k ) ).div( gridSpacingNode.x );
			const term = betaU( i, j, k ).mul( dtNode ).div( referenceDensityNode ).mul( gravityNode.dot( pos ) ).mul( dRho );

			velocityGrid.dataU( i, j, k ).subAssign( term );

		} );

		applyReducedGravityV = tsl_array_n.kernel( dataSizeV, ( i, j, k ) => {

			const pos = originNode.add( vec3( float( i ).add( 0.5 ).mul( gridSpacingNode.x ), float( j ).mul( gridSpacingNode.y ), float( k ).add( 0.5 ).mul( gridSpacingNode.z ) ) );
			const dRho = cellDensity( i, clampJ2( j ), k ).sub( cellDensity( i, clampJ2( j.sub( 1 ) ), k ) ).div( gridSpacingNode.y );
			const term = betaV( i, j, k ).mul( dtNode ).div( referenceDensityNode ).mul( gravityNode.dot( pos ) ).mul( dRho );

			velocityGrid.dataV( i, j, k ).subAssign( term );

		} );

		applyReducedGravityW = tsl_array_n.kernel( dataSizeW, ( i, j, k ) => {

			const pos = originNode.add( vec3( float( i ).add( 0.5 ).mul( gridSpacingNode.x ), float( j ).add( 0.5 ).mul( gridSpacingNode.y ), float( k ).mul( gridSpacingNode.z ) ) );
			const dRho = cellDensity( i, j, clampK2( k ) ).sub( cellDensity( i, j, clampK2( k.sub( 1 ) ) ) ).div( gridSpacingNode.z );
			const term = betaW( i, j, k ).mul( dtNode ).div( referenceDensityNode ).mul( gravityNode.dot( pos ) ).mul( dRho );

			velocityGrid.dataW( i, j, k ).subAssign( term );

		} );

	}

	// ---- velOld snapshot + G2P -- see grid_flip_solver2.js's own header
	// comment.

	const oldDataU = tsl_array_n.arrayN( 'float', dataSizeU );
	const oldDataV = tsl_array_n.arrayN( 'float', dataSizeV );
	const oldDataW = tsl_array_n.arrayN( 'float', dataSizeW );
	oldDataU.fromArray( new Float32Array( uCount ) );
	oldDataV.fromArray( new Float32Array( vCount ) );
	oldDataW.fromArray( new Float32Array( wCount ) );

	const snapshotOldU = createCopyKernel3( velocityGrid.dataU, oldDataU, dataSizeU );
	const snapshotOldV = createCopyKernel3( velocityGrid.dataV, oldDataV, dataSizeV );
	const snapshotOldW = createCopyKernel3( velocityGrid.dataW, oldDataW, dataSizeW );

	const flipRatioNode = numberOrNode( flipRatio );
	const velocityDampingNode = clamp( numberOrNode( velocityDamping ), float( 0 ), float( 1 ) );

	function clampParticleVelocity( v ) {

		return isNonFinite3( v ).select( vec3( 0 ), clamp( v, vec3( - MAX_PARTICLE_VELOCITY ), vec3( MAX_PARTICLE_VELOCITY ) ) );

	}

	const g2pUpdate = tsl_array_n.kernel( maxParticles, ( p ) => {

		const pos = positions( p );
		const newVel = faceCenteredValueAtPosition3( velocityGrid.dataU, velocityGrid.dataV, velocityGrid.dataW, velocityGrid.gridSpacing, velocityGrid.dataOriginU, velocityGrid.dataOriginV, velocityGrid.dataOriginW, pos, dataSizeU, dataSizeV, dataSizeW );
		const oldVel = faceCenteredValueAtPosition3( oldDataU, oldDataV, oldDataW, velocityGrid.gridSpacing, velocityGrid.dataOriginU, velocityGrid.dataOriginV, velocityGrid.dataOriginW, pos, dataSizeU, dataSizeV, dataSizeW );

		const delta = newVel.sub( oldVel );
		const flipVel = velocities( p ).add( delta );
		const blended = flipVel.mul( flipRatioNode ).add( newVel.mul( float( 1 ).sub( flipRatioNode ) ) );
		const damped = blended.mul( float( 1 ).sub( velocityDampingNode ) );

		velocities( p ).assign( clampParticleVelocity( damped ) );

	} );

	// ---- particle advection -- see grid_flip_solver2.js's own header
	// comment.

	const advectionSolver = createSemiLagrangianAdvectionSolver3( { velocityGrid, collider, dt } );

	const CLAMP_EPSILON = 1e-4;
	const minPos = originNode.add( vec3( CLAMP_EPSILON ) );
	const maxPos = originNode.add( vec3( resolutionX * gridSpacingX, resolutionY * gridSpacingY, resolutionZ * gridSpacingZ ) ).sub( vec3( CLAMP_EPSILON ) );

	const advectParticles = tsl_array_n.kernel( maxParticles, ( p ) => {

		const previous = positions( p );
		const traced = advectionSolver.trace( previous, -1 );
		const safe = isNonFinite3( traced ).select( previous, traced );
		positions( p ).assign( clamp( safe, minPos, maxPos ) );

	} );

	// ---- particle push-out -- see grid_flip_solver2.js's own header
	// comment for the full derivation.

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

	// ---------------------------------------------------------------- carried concentration (dye)
	// See grid_flip_solver2.js's own header comment for the full design --
	// unchanged, cellShape is now 3D.
	let cellConcentration = null;
	let concentrationGridPass = null;
	let concentrationParticlePass = null;

	if ( carryConcentration ) {

		cellConcentration = tsl_array_n.arrayN( 'float', cellShape );
		cellConcentration.fromArray( new Float32Array( cellCount ) );

		const concentrationAccum = tsl_array_n.arrayN( 'int', cellShape );
		concentrationAccum.node.toAtomic();
		const concentrationCount = tsl_array_n.arrayN( 'int', cellShape );
		concentrationCount.node.toAtomic();

		const zeroConcentrationCells = new Int32Array( cellCount );
		const concentrationScaleNode = float( concentrationAtomicScale );
		const mixingNode = clamp( numberOrNode( mixing ), float( 0 ), float( 1 ) );
		const fadeNode = clamp( numberOrNode( fade ), float( 0 ), float( 1 ) );
		const concentrationOf = ( p ) => clamp( concentration( p ), float( 0 ), float( 1 ) );

		const accumulateConcentration = tsl_array_n.kernel( maxParticles, ( p ) => {

			const { i, j, k } = cellIndexOf( positions( p ) );
			atomicAdd( concentrationCount( i, j, k ), 1 );
			atomicAdd( concentrationAccum( i, j, k ), round( concentrationOf( p ).mul( concentrationScaleNode ) ).toInt() );

		} );

		const resolveCellConcentration = tsl_array_n.kernel( cellShape, ( i, j, k ) => {

			const total = atomicLoad( concentrationCount( i, j, k ) ).toFloat();
			const sum = atomicLoad( concentrationAccum( i, j, k ) ).toFloat().div( concentrationScaleNode );
			const mean = sum.div( max( total, float( 1 ) ) );

			const c = total.greaterThan( 0.5 ).select( clamp( mean, float( 0 ), float( 1 ) ), float( 0 ) );
			cellConcentration( i, j, k ).assign( c );

			if ( densityCouplingEnabled ) {

				cellDensity( i, j, k ).assign( densityAtZeroNode.add( densityAtOneNode.sub( densityAtZeroNode ).mul( c ) ) );

			}

		} );

		// See grid_flip_solver2.js's own mixAndFade header comment for the
		// full "why zero mixing is not automatically a no-op" NaN-propagation
		// story -- unchanged, isNonFinite guard preserved.
		const mixAndFade = tsl_array_n.kernel( maxParticles, ( p ) => {

			const sampled = collocatedValueAtPosition3( cellConcentration, gridSpacingNode, cellCenterOrigin, positions( p ), cellShape );
			const c = concentrationOf( p );

			const local = isNonFinite( sampled ).select( c, sampled );
			const blended = c.add( local.sub( c ).mul( mixingNode ) );

			concentration( p ).assign( clamp( blended.mul( float( 1 ).sub( fadeNode ) ), float( 0 ), float( 1 ) ) );

		} );

		concentrationGridPass = function concentrationGridPassNow() {

			concentrationAccum.fromArray( zeroConcentrationCells );
			concentrationCount.fromArray( zeroConcentrationCells );
			accumulateConcentration();
			resolveCellConcentration();

			if ( densityCouplingEnabled ) {

				computeBetaU();
				computeBetaV();
				computeBetaW();

			}

		};

		concentrationParticlePass = mixAndFade;

	}

	// Runtime-switchable -- see grid_flip_solver2.js's own comment.
	const settings = { batchStages: true };

	// See grid_flip_solver2.js's own buildStageSet/stage-list header
	// comment for the full batching design -- unchanged.
	function buildStageSet( stages ) {

		return {
			batched: tsl_array_n.createBatch( stages ),
			unbatched: () => { for ( const stage of stages ) if ( stage ) stage(); }
		};

	}

	const preP2GStages = buildStageSet( [
		advectParticles,
		pushOutOfCollider
	] );

	const afterP2GList = [
		scatterU,
		scatterV,
		scatterW,
		finalizeU,
		finalizeV,
		finalizeW,

		snapshotOldU,
		snapshotOldV,
		snapshotOldW,

		extrapolateWeightU,
		extrapolateWeightV,
		extrapolateWeightW,
		boundarySolver.constrainVelocity,

		clearFluidMask,
		markFluidCellsKernel,
		computeUFluidAdjacent,
		computeVFluidAdjacent,
		computeWFluidAdjacent,

		resamplePass,

		concentrationGridPass
	];

	if ( reducedPressureEnabled ) {

		afterP2GList.push( applyReducedGravityU, applyReducedGravityV, applyReducedGravityW );

	} else {

		afterP2GList.push( applyGravityU, applyGravityV, applyGravityW );

	}

	afterP2GList.push( applyForces, boundarySolver.constrainVelocity );

	const afterP2GStages = buildStageSet( afterP2GList );

	const postProjectStages = buildStageSet( [
		boundarySolver.constrainVelocity,

		extrapolatePressureU,
		extrapolatePressureV,
		extrapolatePressureW,

		g2pUpdate,

		concentrationParticlePass
	] );

	function runStages( stageSet ) {

		( settings.batchStages ? stageSet.batched : stageSet.unbatched )();

	}

	async function onAdvanceTimeStep() {

		runStages( preP2GStages );

		resetAccumulators();

		runStages( afterP2GStages );

		await projectDispatch();

		runStages( postProjectStages );

	}

	return {
		onAdvanceTimeStep,
		settings,
		positions, velocities, fluidMask,
		concentration, cellConcentration, cellDensity,
		pressure: pressureSolver.pressure,
		// Exposed for measurement, as in grid_solver3.js: a conservation
		// check has to read the velocity field this solver projects, and
		// rebuilding it outside would measure something else.
		velocityGrid,
		boundarySolver, pressureSolver
	};

}
