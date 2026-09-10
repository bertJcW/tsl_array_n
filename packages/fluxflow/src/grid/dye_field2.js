// A dye field carried on its own grid, finer than the velocity grid.
//
// *** Why this exists, and why the dye stopped riding on particles ***
//
// grid_flip_solver2.js's `carryConcentration` puts the dye on the particles.
// Transport is then exact -- a particle carries its value and nothing
// diffuses it -- which is the right answer for a *phase*: two liquids that
// do not mix, where the whole point is that the boundary stays razor sharp.
//
// It is the wrong representation for dye, and that was measured rather than
// argued. In examples/28-drop-into-pool/ the dye is about 420 particles, and
// 420 particles is the entire information content of the dye field however
// it is drawn: rendering the same frame as accumulated coverage at four
// times the grid resolution, instead of one opaque disc per particle,
// produced a picture almost indistinguishable from the disc version. No
// renderer can show filaments that the representation does not contain.
//
// What dye actually looks like is thin structure well below the spacing of
// the liquid particles, and the way to get that is to stop tying the dye's
// resolution to the liquid's. A dye field is a *passive scalar*: it needs no
// pressure solve, no projection, no particles. It costs a couple of
// elementwise kernels per frame, so it can afford to be several times finer
// than the velocity grid it is advected by -- coarse velocity, fine dye.
// That split is standard practice for exactly this reason.
//
// The advection solver needed no changes at all to support it:
// advection_solver2.js's `advectScalar2` expresses everything in the scalar
// grid's own terms (its `dataSize`, its `dataPosition`, its `gridSpacing`)
// and only ever samples the velocity by world position, so handing it a grid
// with a different resolution simply works.
//
// *** What this deliberately is not ***
//
// Passive. The dye does not affect the flow: no density coupling, no
// buoyancy. That is not a simplification to be apologised for, it is what
// dye is -- a drop of food colouring is within a percent or two of the
// density of the water it is dropped into, and everything visible about it
// comes from being stirred, not from weighing something different. A dyed
// liquid heavy enough to drive its own plume is a different scene and
// belongs on the particle-carried concentration, which couples to density
// already.

import * as tsl_array_n from 'tsl_array_n';
import { float, clamp } from 'three/tsl';
import { createCellCenteredScalarGrid2 } from './grid_data2.js';
import { collocatedValueAtPosition2 } from './grid_math.js';
import { createCopyKernel2 } from './array_utils.js';
import { createSemiLagrangianAdvectionSolver2 } from './advection_solver2.js';

// Explicit diffusion is stable while `D dt / h^2` stays under 1/4 in 2D.
// The kernel clamps to this rather than to the limit itself, so a caller who
// asks for more diffusion than the step can carry gets the most it can be
// given instead of an instability. Diffusing faster than that needs an
// implicit solve, which a dye field does not justify.
const MAX_DIFFUSION_NUMBER = 0.2;

function numberOrNode( value ) {

	return typeof value === 'number' ? float( value ) : value;

}

/**
 * Creates a dye field on a grid `subdivisions` times finer than the velocity
 * grid, advected by that velocity every step.
 *
 * velocityGrid: the FaceCenteredGrid2 driving the flow.
 * resolution/gridSpacing/origin: the *velocity* grid's, in plain numbers --
 *   same convention as every other factory here, and the fine grid is
 *   derived from them.
 * subdivisions: dye cells per velocity cell, per axis. 3 gives nine dye
 *   cells per velocity cell.
 * dt: number or live node, shared with the solver.
 * diffusion: physical diffusivity, in (grid units)^2 per unit time. 0 (the
 *   default) transports with no diffusion beyond what the advection scheme
 *   itself introduces.
 * liquidMask: optional { field, resolution } -- a coarse 0/1 field (a FLIP
 *   solver's own `fluidMask` is exactly this) used to keep dye out of the
 *   air. Without it, dye thrown above a free surface by a splash stays there
 *   as a stationary smear, because the velocity outside the liquid is
 *   extrapolated rather than solved.
 * order: forwarded to the advection solver. Defaults to 2 (MacCormack),
 *   because the whole reason for a fine grid is to keep thin structure and
 *   first-order semi-Lagrangian advection is what would smear it away.
 */
export function createDyeField2( {
	velocityGrid,
	resolution,
	gridSpacing = [ 1, 1 ],
	origin = [ 0, 0 ],
	subdivisions = 3,
	dt,
	diffusion = 0,
	liquidMask,
	collider,
	order = 2,
	maxSubsteps
} ) {

	if ( ! Number.isInteger( subdivisions ) || subdivisions < 1 ) {

		throw new Error( `createDyeField2: subdivisions must be a positive integer, got ${ subdivisions }.` );

	}

	const [ coarseX, coarseY ] = resolution;
	const fineX = coarseX * subdivisions;
	const fineY = coarseY * subdivisions;
	const fineHx = gridSpacing[ 0 ] / subdivisions;
	const fineHy = gridSpacing[ 1 ] / subdivisions;
	const fineShape = [ fineX, fineY ];
	const fineCount = fineX * fineY;

	const field = createCellCenteredScalarGrid2( fineX, fineY, fineHx, fineHy, origin[ 0 ], origin[ 1 ] );
	const scratch = createCellCenteredScalarGrid2( fineX, fineY, fineHx, fineHy, origin[ 0 ], origin[ 1 ] );

	const zeros = new Float32Array( fineCount );
	field.data.fromArray( zeros );
	scratch.data.fromArray( zeros );

	const advectionSolver = createSemiLagrangianAdvectionSolver2( { velocityGrid, collider, dt, order, maxSubsteps } );
	const dispatchAdvect = advectionSolver.advectScalar2( field, scratch );
	const copyBack = createCopyKernel2( scratch.data, field.data, fineShape );

	// ---- optional: keep dye inside the liquid.
	let dispatchMask = null;

	if ( liquidMask ) {

		const maskShape = liquidMask.resolution ?? resolution;
		const maskSpacing = [ gridSpacing[ 0 ], gridSpacing[ 1 ] ];
		const maskOrigin = [ origin[ 0 ] + maskSpacing[ 0 ] * 0.5, origin[ 1 ] + maskSpacing[ 1 ] * 0.5 ];

		dispatchMask = tsl_array_n.kernel( fineShape, ( i, j ) => {

			const pos = field.dataPosition( i, j );
			const inside = collocatedValueAtPosition2( liquidMask.field, maskSpacing, maskOrigin, pos, maskShape );

			// A soft gate rather than a hard one: the coarse mask's own
			// bilinear edge is a velocity cell wide, and multiplying by it
			// fades the dye out across that edge instead of stamping a
			// staircase of the coarse grid onto a field whose entire purpose
			// is to be finer than that grid.
			field.data( i, j ).mulAssign( clamp( inside, float( 0 ), float( 1 ) ) );

		} );

	}

	// ---- optional: real diffusion, with a real coefficient.
	//
	// Distinct from grid_flip_solver2.js's own `mixing`, which blends a
	// particle toward its cell's mean -- that is a blend factor per frame,
	// so its effect depends on the grid and on the frame rate and it has no
	// units. This is `dc/dt = D laplacian(c)`, so a given D means the same
	// spreading whatever the resolution or the step.
	let dispatchDiffuse = null;
	const diffusionIsLive = typeof diffusion !== 'number';

	if ( diffusionIsLive || diffusion > 0 ) {

		const dtNode = numberOrNode( dt );
		const diffusionNode = numberOrNode( diffusion );

		// Isotropic in practice here (the fine grid inherits the velocity
		// grid's aspect), but written per axis so it stays correct if that
		// ever stops being true.
		const invHx2 = 1 / ( fineHx * fineHx );
		const invHy2 = 1 / ( fineHy * fineHy );

		dispatchDiffuse = tsl_array_n.kernel( fineShape, ( i, j ) => {

			const c = field.data( i, j );

			// Clamped index reads: a zero-gradient (no-flux) edge, which is
			// the right boundary for a dye that cannot leave the domain.
			const iLo = i.sub( 1 ).max( 0 );
			const iHi = i.add( 1 ).min( fineX - 1 );
			const jLo = j.sub( 1 ).max( 0 );
			const jHi = j.add( 1 ).min( fineY - 1 );

			const lapX = field.data( iLo, j ).add( field.data( iHi, j ) ).sub( c.mul( 2 ) ).mul( invHx2 );
			const lapY = field.data( i, jLo ).add( field.data( i, jHi ) ).sub( c.mul( 2 ) ).mul( invHy2 );

			// The stability number is clamped, not the coefficient -- see
			// MAX_DIFFUSION_NUMBER.
			const step = clamp( diffusionNode.mul( dtNode ), float( 0 ), float( MAX_DIFFUSION_NUMBER / Math.max( invHx2, invHy2 ) ) );

			scratch.data( i, j ).assign( c.add( lapX.add( lapY ).mul( step ) ) );

		} );

	}

	/**
	 * One frame of dye transport: advect, then optionally diffuse, then
	 * optionally clip back inside the liquid.
	 */
	function step() {

		dispatchAdvect();
		copyBack();

		if ( dispatchDiffuse ) {

			dispatchDiffuse();
			copyBack();

		}

		if ( dispatchMask ) dispatchMask();

	}

	function clear() {

		field.data.fromArray( zeros );

	}

	return {
		field,
		resolution: fineShape,
		gridSpacing: [ fineHx, fineHy ],
		subdivisions,
		step,
		clear,
		advect: () => {

			dispatchAdvect();
			copyBack();

		}
	};

}

/**
 * CPU-side helper: a Float32Array for a dye field of `resolution`, filled by
 * evaluating `valueAt(x, y)` at each cell's own world centre.
 *
 * Seeding is a caller-side concern (same reasoning as computeFlipBoxSeed's
 * own): it needs no GPU, it happens once, and every scene wants a different
 * shape. This exists so the common case is not fifteen lines of index
 * arithmetic in every example.
 */
export function computeDyeSeed( { resolution, gridSpacing, origin = [ 0, 0 ], valueAt } ) {

	const [ nx, ny ] = resolution;
	const out = new Float32Array( nx * ny );

	for ( let j = 0; j < ny; j ++ ) {

		const y = origin[ 1 ] + ( j + 0.5 ) * gridSpacing[ 1 ];

		for ( let i = 0; i < nx; i ++ ) {

			const x = origin[ 0 ] + ( i + 0.5 ) * gridSpacing[ 0 ];
			out[ i + nx * j ] = valueAt( x, y );

		}

	}

	return out;

}
