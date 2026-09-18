// 3D counterpart of grid_pressure_solver2.js. See that file's own header
// comment for the full jet-derived design: the sign-flip derivation (A =
// +Laplacian, so the correction step needs u = u* - grad(p), not jet's own
// literal +), the Dirichlet-neighbour RHS correction, the last-resort
// circuit breaker and its "snapshot only after the check passes"
// invariant, and the variable-density faceWeights option -- none of that
// reasoning is dimension-specific, and none of it changes here.
//
// What DOES change: `linalg.js`/`multigrid.js` underneath this file were
// already dimension-generic (confirmed directly by examples/31-conjugate-
// gradient-3d/, a real 3D Poisson solve against an analytic solution, and
// by the buildCoarseSweepKernel bug that example's first run found and
// fixed in multigrid.js itself) -- so createLaplacianOperator/
// createMultigridPreconditioner/createPreconditionedConjugateGradientSolver
// are called exactly as grid_pressure_solver2.js already calls them, just
// with a 3-element shape/gridSpacing. Everything that changes here is
// grid_math3.js's own face-centred divergence (a third component, dataW)
// and neighbourDirichletContribution's Dirichlet-neighbour sum (six
// neighbours instead of four -- one more axis).

import * as tsl_array_n from 'tsl_array_n';
import { float, int, max, min, atomicAdd, If } from 'three/tsl';
import { createCellCenteredScalarGrid3 } from './grid_data3.js';
import { faceCenteredDivergenceAtCenter3 } from './grid_math3.js';
import { createCopyKernel3 } from './array_utils3.js';
import { createLaplacianOperator, createMultigridPreconditioner, createJacobiPreconditioner, createIdentityPreconditioner } from '../linalg/multigrid.js';
import { createPreconditionedConjugateGradientSolver } from '../linalg/linalg.js';
import { isNonFiniteOrAbove } from '../float_guards.js';
import { instrumentDispatch, timePhase } from '../profiling.js';

// Same default and the same reasoning as grid_pressure_solver2.js's own
// DEFAULT_MAX_PLAUSIBLE_PRESSURE -- see that file's header comment for the
// real-hardware history behind it (a per-instance option, not a tightened
// global default).
const DEFAULT_MAX_PLAUSIBLE_PRESSURE = 1e6;

// options: see grid_pressure_solver2.js's own createGridPressureSolver2
// header comment for the full parameter list -- unchanged meanings,
// resolution/gridSpacing/origin are now triples.
//
// options.colliderWeights: { u, v, w } -- a STATIC collider's own per-face
// open-area fraction (sdf_collider3.js's own computeFaceWeights), the
// fix for the Karman-vortex-street instability jet/fluid-engine-dev's own
// GridFractionalSinglePhasePressureSolver3 pointed at: this project's
// pressure Laplacian used to be entirely collider-blind (a partially-open
// face was rounded to fully open or fully closed by
// grid_blocked_boundary_condition_solver3.js's own binary marker, with no
// way to represent anything in between), relying on that binary marker
// alone to keep velocity out of solid -- exactly the "Lego blocks" approach
// jet's own GridFluidSolver3 does NOT default to. colliderWeights bakes the
// real fractional occupancy into the pressure system's own coefficients
// instead, the way jet's own default solver does.
//
// Deliberately kept separate from options.faceWeights (that option's own
// meaning -- variable-density beta -- is unchanged): the two are combined
// multiplicatively for SYSTEM building (the Laplacian and the
// Dirichlet-neighbour RHS term, where both are legitimately just face
// conductance multipliers), but the velocity CORRECTION step below treats
// them differently -- faceWeights still SCALES the gradient (matches jet's
// own applyPressureGradient using 1/rho as a multiplier), while
// colliderWeights only GATES it (skip the correction entirely when the
// face is fully blocked, apply it in full otherwise) -- matching jet's own
// applyPressureGradient using its collider weight purely as a `> 0` check,
// never as a scale factor on the correction itself.
export function createGridPressureSolver3( {
	resolution, gridSpacing, origin = [ 0, 0, 0 ],
	dirichlet,
	faceWeights,
	colliderWeights,
	multigrid = {},
	preconditioner = 'multigrid',
	tolerance = 1e-6,
	maxIterations = 100,
	relativeTolerance = true,
	residualRecomputeInterval = 1,
	verifyConvergence = true,
	gpuResidentSetup = true,
	residualCheckInterval = 4,
	gpuResidentScalars = true,
	batchIterations = true,
	atomicScale,
	maxPlausiblePressure = DEFAULT_MAX_PLAUSIBLE_PRESSURE
} = {} ) {

	const settings = { residualCheckInterval, gpuResidentScalars, batchIterations, preconditioner, maxIterations, tolerance, checkBadCells: true, badCellsRideAlong: true, optimisticStopTest: true, gpuStopTest: true, fuseVcycleIntoIteration: true, fuseChunkIntoOneSubmission: false, gpuResidentSetup, relativeTolerance, residualRecomputeInterval, verifyConvergence };

	const [ resolutionX, resolutionY, resolutionZ ] = resolution;
	const [ gridSpacingX, gridSpacingY, gridSpacingZ ] = gridSpacing;
	const [ originX, originY, originZ ] = origin;
	const shape = resolution;

	const pressureGrid = createCellCenteredScalarGrid3( resolutionX, resolutionY, resolutionZ, gridSpacingX, gridSpacingY, gridSpacingZ, originX, originY, originZ );
	const b = tsl_array_n.arrayN( 'float', shape );

	let dirichletMaskField, dirichletTargetField, updateDirichletFields, dirichletMask;
	// The caller's own Dirichlet cells, without the enclosed-cell guard's
	// additions -- the only part of the mask the multigrid preconditioner
	// coarsens. See its own options.coarseDirichletMask comment for why a
	// collider-enclosed cell must not be coarsened.
	let ventMaskField, coarseDirichletMask;

	// A cell every one of whose six faces has zero collider weight is fully
	// enclosed by solid -- entirely decoupled from the rest of the system
	// (every neighbour sees zero conductance to it too, since a face weight
	// is shared between the two cells it separates). Left alone, that's a
	// singular (all-zero) row in the Laplacian: no diagonal, no coupling,
	// nothing for CG to solve. jet/fluid-engine-dev's own buildSingleSystem
	// guards against exactly this with "row.center = 1, b = 0" when a cell's
	// total coupling is ~0 -- this reuses the pre-existing dirichletMask
	// identity-row mechanism to the same end (pin p = 0, arbitrary but
	// harmless since the cell is provably disconnected from every neighbour
	// either way).
	function totalColliderWeight( i, j, k ) {

		return colliderWeights.u( i, j, k ).add( colliderWeights.u( i.add( 1 ), j, k ) )
			.add( colliderWeights.v( i, j, k ) ).add( colliderWeights.v( i, j.add( 1 ), k ) )
			.add( colliderWeights.w( i, j, k ) ).add( colliderWeights.w( i, j, k.add( 1 ) ) );

	}

	if ( dirichlet || colliderWeights ) {

		dirichletMaskField = tsl_array_n.arrayN( 'float', shape );
		dirichletTargetField = tsl_array_n.arrayN( 'float', shape );
		ventMaskField = tsl_array_n.arrayN( 'float', shape );

		updateDirichletFields = tsl_array_n.kernel( shape, ( i, j, k ) => {

			const enclosed = colliderWeights ? totalColliderWeight( i, j, k ).lessThan( 1e-6 ) : null;

			if ( dirichlet ) {

				const pos = pressureGrid.dataPosition( i, j, k );
				const { active, target } = dirichlet( pos );

				const finalActive = enclosed ? active.or( enclosed ) : active;
				const finalTarget = enclosed ? active.select( target, float( 0 ) ) : target;

				dirichletMaskField( i, j, k ).assign( finalActive.select( float( 1 ), float( 0 ) ) );
				dirichletTargetField( i, j, k ).assign( finalTarget );
				ventMaskField( i, j, k ).assign( active.select( float( 1 ), float( 0 ) ) );

			} else {

				dirichletMaskField( i, j, k ).assign( enclosed.select( float( 1 ), float( 0 ) ) );
				dirichletTargetField( i, j, k ).assign( float( 0 ) );
				ventMaskField( i, j, k ).assign( float( 0 ) );

			}

		} );

		dirichletMask = ( i, j, k ) => dirichletMaskField( i, j, k ).greaterThan( 0.5 );
		coarseDirichletMask = ( i, j, k ) => ventMaskField( i, j, k ).greaterThan( 0.5 );

	}

	// System-building weight: faceWeights (density beta) and colliderWeights
	// (collider open-area fraction) combined multiplicatively, one
	// accessor per axis -- see this function's own header comment for why
	// they're combined here but NOT in the velocity-correction step below.
	// systemFaceWeights is what gets forwarded to multigrid.js's own
	// dimension-generic [axis] form (one entry wider than
	// grid_pressure_solver2.js's own faceWeightAccessors); faceWeights and
	// colliderWeights themselves stay as separately-named locals because
	// the correction step still needs each on its own.
	function combineAxisWeight( axis ) {

		const fw = faceWeights ? faceWeights[ axis ] : undefined;
		const cw = colliderWeights ? colliderWeights[ axis ] : undefined;

		if ( fw && cw ) return ( i, j, k ) => fw( i, j, k ).mul( cw( i, j, k ) );
		return fw || cw;

	}

	const systemFaceWeights = { u: combineAxisWeight( 'u' ), v: combineAxisWeight( 'v' ), w: combineAxisWeight( 'w' ) };
	const hasSystemFaceWeights = systemFaceWeights.u || systemFaceWeights.v || systemFaceWeights.w;
	const faceWeightAccessors = hasSystemFaceWeights ? [ systemFaceWeights.u, systemFaceWeights.v, systemFaceWeights.w ] : undefined;
	const applyLaplacian = createLaplacianOperator( shape, gridSpacing, { dirichletMask, faceWeights: faceWeightAccessors } );

	const preconditionerBuilders = {
		multigrid: createMultigridPreconditioner( shape, gridSpacing, { ...multigrid, dirichletMask, coarseDirichletMask, faceWeights: faceWeightAccessors } ),
		jacobi: createJacobiPreconditioner( shape, gridSpacing, { dirichletMask, faceWeights: faceWeightAccessors } ),
		none: createIdentityPreconditioner( shape )
	};

	if ( ! Object.prototype.hasOwnProperty.call( preconditionerBuilders, preconditioner ) ) {

		throw new Error( `grid_pressure_solver3: unknown preconditioner '${ preconditioner }' (expected ${ Object.keys( preconditionerBuilders ).join( ', ' ) }).` );

	}

	function applyPreconditioner( input, output ) {

		const built = {};

		for ( const name of Object.keys( preconditionerBuilders ) ) {

			built[ name ] = preconditionerBuilders[ name ]( input, output );

		}

		function dispatchSelectedPreconditioner() {

			built[ settings.preconditioner ]();

		}

		dispatchSelectedPreconditioner.forms = built.multigrid.forms;
		dispatchSelectedPreconditioner.isFusable = () => settings.preconditioner === 'multigrid';

		return dispatchSelectedPreconditioner;

	}

	applyPreconditioner.settings = preconditionerBuilders.multigrid.settings;

	const refreshPreconditionerMask = preconditionerBuilders.multigrid.refreshCoarseLevels;

	const cg = createPreconditionedConjugateGradientSolver( applyLaplacian, applyPreconditioner, b, pressureGrid.data, { atomicScale } );

	const diagnostics = { converged: null, rejected: false, iterations: null, stoppedBy: null };

	const pressureSnapshot = tsl_array_n.arrayN( 'float', shape );
	pressureSnapshot.fromArray( new Float32Array( shape.reduce( ( a, n ) => a * n, 1 ) ) );
	const snapshotPressure = instrumentDispatch( 'pressure-snapshot', createCopyKernel3( pressureGrid.data, pressureSnapshot, shape ) );
	const restorePressure = instrumentDispatch( 'pressure-restore', createCopyKernel3( pressureSnapshot, pressureGrid.data, shape ) );

	const badCountAccum = tsl_array_n.array0( 'int' );
	badCountAccum.node.toAtomic();

	const countBadPressureCells = tsl_array_n.kernel( shape, ( i, j, k ) => {

		const value = pressureGrid.data( i, j, k );
		const isBad = isNonFiniteOrAbove( value, maxPlausiblePressure );

		atomicAdd( badCountAccum(), isBad.select( int( 1 ), int( 0 ) ) );

	} );

	const badCountZero = new Int32Array( [ 0 ] );

	async function countBadPressureCellsNow() {

		badCountAccum.fromArray( badCountZero );
		countBadPressureCells();
		const [ count ] = await badCountAccum.toArray();
		return count;

	}

	cg.setReadCompanion( {

		dispatch: () => {

			if ( settings.checkBadCells !== true || settings.badCellsRideAlong !== true ) return;

			badCountAccum.fromArray( badCountZero );
			countBadPressureCells();

		},

		read: () => {

			if ( settings.checkBadCells !== true || settings.badCellsRideAlong !== true ) return null;

			return badCountAccum.toArray().then( ( values ) => values[ 0 ] );

		}

	} );

	// input/output: FaceCenteredGrid3 (grid_data3.js).
	function project( input, output ) {

		if ( input.resolution.join() !== resolution.join() ) {

			throw new Error( `createGridPressureSolver3: project() input resolution [${ input.resolution }] does not match constructed resolution [${ resolution }].` );

		}

		// See grid_pressure_solver2.js's own neighbourDirichletContribution
		// header comment for the full derivation -- unchanged, six
		// neighbours (one more axis pair) instead of four.
		function neighbourDirichletContribution( i, j, k ) {

			if ( ! dirichletMask ) return float( 0 );

			const invHx2 = 1 / ( gridSpacing[ 0 ] * gridSpacing[ 0 ] );
			const invHy2 = 1 / ( gridSpacing[ 1 ] * gridSpacing[ 1 ] );
			const invHz2 = 1 / ( gridSpacing[ 2 ] * gridSpacing[ 2 ] );

			const term = ( ni, nj, nk, inBounds, weight, invH2 ) => {

				const isDir = dirichletMaskField( ni, nj, nk ).greaterThan( 0.5 );
				const contribution = weight.mul( dirichletTargetField( ni, nj, nk ) ).mul( invH2 );

				return inBounds.and( isDir ).select( contribution, float( 0 ) );

			};

			const one = float( 1 );
			const lo = ( n ) => max( 0, n );
			const hiI = ( n ) => min( n, resolutionX - 1 );
			const hiJ = ( n ) => min( n, resolutionY - 1 );
			const hiK = ( n ) => min( n, resolutionZ - 1 );

			const wLower = ( axis ) => {

				if ( ! hasSystemFaceWeights ) return one;
				return axis === 'x' ? systemFaceWeights.u( i, j, k ) : axis === 'y' ? systemFaceWeights.v( i, j, k ) : systemFaceWeights.w( i, j, k );

			};

			const wUpper = ( axis ) => {

				if ( ! hasSystemFaceWeights ) return one;
				return axis === 'x' ? systemFaceWeights.u( i.add( 1 ), j, k ) : axis === 'y' ? systemFaceWeights.v( i, j.add( 1 ), k ) : systemFaceWeights.w( i, j, k.add( 1 ) );

			};

			return term( lo( i.sub( 1 ) ), j, k, i.greaterThan( 0 ), wLower( 'x' ), invHx2 )
				.add( term( hiI( i.add( 1 ) ), j, k, i.lessThan( resolutionX - 1 ), wUpper( 'x' ), invHx2 ) )
				.add( term( i, lo( j.sub( 1 ) ), k, j.greaterThan( 0 ), wLower( 'y' ), invHy2 ) )
				.add( term( i, hiJ( j.add( 1 ) ), k, j.lessThan( resolutionY - 1 ), wUpper( 'y' ), invHy2 ) )
				.add( term( i, j, lo( k.sub( 1 ) ), k.greaterThan( 0 ), wLower( 'z' ), invHz2 ) )
				.add( term( i, j, hiK( k.add( 1 ) ), k.lessThan( resolutionZ - 1 ), wUpper( 'z' ), invHz2 ) );

		}

		// The RHS divergence for a face that colliderWeights has partly
		// closed must be scaled by that SAME open-area fraction, or the
		// system is inconsistent: the LHS Laplacian already only lets
		// `weight` of that face's pressure gradient act (via
		// systemFaceWeights, above), but the raw, unweighted face velocity
		// would still claim the cell needs to shed/gain its FULL nominal
		// flux through a face that's mostly closed -- a real mismatch this
		// port's own real-hardware testing found necessary (the collider
		// otherwise explodes almost immediately, far faster than before
		// colliderWeights existed at all, regardless of preconditioner).
		// Deliberately colliderWeights-only, NOT combined with the density
		// faceWeights option: `faceWeights` scales the pressure GRADIENT in
		// the correction step (1/rho), not the velocity itself, so the
		// divergence of the raw velocity field it's fed is already correct
		// as-is for that option -- see this file's own colliderWeights
		// header comment for why the two options aren't interchangeable.
		function weightedDivergence( i, j, k ) {

			if ( ! colliderWeights ) return faceCenteredDivergenceAtCenter3( input.dataU, input.dataV, input.dataW, pressureGrid.gridSpacing, i, j, k );

			const leftU = input.dataU( i, j, k ).mul( colliderWeights.u( i, j, k ) );
			const rightU = input.dataU( i.add( 1 ), j, k ).mul( colliderWeights.u( i.add( 1 ), j, k ) );
			const bottomV = input.dataV( i, j, k ).mul( colliderWeights.v( i, j, k ) );
			const topV = input.dataV( i, j.add( 1 ), k ).mul( colliderWeights.v( i, j.add( 1 ), k ) );
			const backW = input.dataW( i, j, k ).mul( colliderWeights.w( i, j, k ) );
			const frontW = input.dataW( i, j, k.add( 1 ) ).mul( colliderWeights.w( i, j, k.add( 1 ) ) );

			return rightU.sub( leftU ).div( pressureGrid.gridSpacing.x )
				.add( topV.sub( bottomV ).div( pressureGrid.gridSpacing.y ) )
				.add( frontW.sub( backW ).div( pressureGrid.gridSpacing.z ) );

		}

		const dispatchBuildSystem = tsl_array_n.kernel( shape, ( i, j, k ) => {

			const divergence = weightedDivergence( i, j, k );

			if ( dirichletMask ) {

				const isDirichlet = dirichletMask( i, j, k );
				const target = dirichletTargetField( i, j, k );
				const dirichletNeighbourRhs = neighbourDirichletContribution( i, j, k );

				b( i, j, k ).assign( isDirichlet.select( target.negate(), divergence.sub( dirichletNeighbourRhs ) ) );

				If( isDirichlet, () => {

					pressureGrid.data( i, j, k ).assign( target );

				} );

			} else {

				b( i, j, k ).assign( divergence );

			}

		} );

		// See grid_pressure_solver2.js's own dispatchCorrectU/V header
		// comment -- unchanged, one more component.
		const dispatchCorrectU = tsl_array_n.kernel( input.dataSizeU, ( m, j, k ) => {

			If( m.greaterThan( 0 ).and( m.lessThan( resolutionX ) ), () => {

				const gradient = pressureGrid.data( m, j, k ).sub( pressureGrid.data( m.sub( 1 ), j, k ) ).div( pressureGrid.gridSpacing.x );
				const scaled = faceWeights ? gradient.mul( faceWeights.u( m, j, k ) ) : gradient;
				const gated = colliderWeights ? scaled.mul( colliderWeights.u( m, j, k ).greaterThan( 0 ).select( float( 1 ), float( 0 ) ) ) : scaled;
				output.dataU( m, j, k ).assign( input.dataU( m, j, k ).sub( gated ) );

			} );

		} );

		const dispatchCorrectV = tsl_array_n.kernel( input.dataSizeV, ( i, m, k ) => {

			If( m.greaterThan( 0 ).and( m.lessThan( resolutionY ) ), () => {

				const gradient = pressureGrid.data( i, m, k ).sub( pressureGrid.data( i, m.sub( 1 ), k ) ).div( pressureGrid.gridSpacing.y );
				const scaled = faceWeights ? gradient.mul( faceWeights.v( i, m, k ) ) : gradient;
				const gated = colliderWeights ? scaled.mul( colliderWeights.v( i, m, k ).greaterThan( 0 ).select( float( 1 ), float( 0 ) ) ) : scaled;
				output.dataV( i, m, k ).assign( input.dataV( i, m, k ).sub( gated ) );

			} );

		} );

		const dispatchCorrectW = tsl_array_n.kernel( input.dataSizeW, ( i, j, m ) => {

			If( m.greaterThan( 0 ).and( m.lessThan( resolutionZ ) ), () => {

				const gradient = pressureGrid.data( i, j, m ).sub( pressureGrid.data( i, j, m.sub( 1 ) ) ).div( pressureGrid.gridSpacing.z );
				const scaled = faceWeights ? gradient.mul( faceWeights.w( i, j, m ) ) : gradient;
				const gated = colliderWeights ? scaled.mul( colliderWeights.w( i, j, m ).greaterThan( 0 ).select( float( 1 ), float( 0 ) ) ) : scaled;
				output.dataW( i, j, m ).assign( input.dataW( i, j, m ).sub( gated ) );

			} );

		} );

		const buildSystemBatch = tsl_array_n.createBatch(
			updateDirichletFields ? [ updateDirichletFields, dispatchBuildSystem ] : [ dispatchBuildSystem ]
		);
		const finishRejected = tsl_array_n.createBatch( [ restorePressure, dispatchCorrectU, dispatchCorrectV, dispatchCorrectW ] );
		const finishAccepted = tsl_array_n.createBatch( [ snapshotPressure, dispatchCorrectU, dispatchCorrectV, dispatchCorrectW ] );

		return async function dispatch() {

			buildSystemBatch();
			// The multigrid preconditioner keeps a Dirichlet mask, and
			// face weights where there are any, per coarse level --
			// coarsened from what buildSystemBatch just refreshed. See
			// multigrid.js's own level-mask comment for why the coarse
			// levels need them at all. Dispatched once per solve, here,
			// rather than inside the V-cycle, which runs once per CG
			// iteration.
			if ( refreshPreconditionerMask ) refreshPreconditionerMask();
			cg.settings.optimisticStopTest = settings.optimisticStopTest === true;
			cg.settings.gpuStopTest = settings.gpuStopTest === true;
			cg.settings.fuseVcycleIntoIteration = settings.fuseVcycleIntoIteration === true;
			cg.settings.fuseChunkIntoOneSubmission = settings.fuseChunkIntoOneSubmission === true;
			diagnostics.converged = await timePhase( 'pressure-cg-solve', () => cg.solve(
				settings.tolerance, settings.maxIterations,
				settings.residualCheckInterval, settings.gpuResidentScalars, settings.batchIterations,
				settings.gpuResidentSetup, settings.relativeTolerance,
				settings.residualRecomputeInterval, settings.verifyConvergence
			) );

			diagnostics.iterations = cg.state ? cg.state.iterations : null;
			diagnostics.residual = cg.state ? Math.sqrt( Math.abs( cg.state.residualSquared ) ) : null;
			diagnostics.stoppedBy = cg.state ? cg.state.stoppedBy : null;

			const ridingCount = settings.badCellsRideAlong ? cg.state.companionValue : null;

			diagnostics.rejected = settings.checkBadCells
				? ( ridingCount !== null && ridingCount !== undefined
					? ridingCount > 0
					: ( await timePhase( 'pressure-badcells-read', () => countBadPressureCellsNow() ) ) > 0 )
				: false;

			if ( diagnostics.rejected ) finishRejected();
			else finishAccepted();

		};

	}

	settings.multigrid = preconditionerBuilders.multigrid.settings;

	return { project, pressure: pressureGrid, b, diagnostics, settings };

}
