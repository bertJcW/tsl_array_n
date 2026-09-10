// Surface tension as a Continuum Surface Force, on a staggered MAC grid.
//
// The defining physics of two liquids that do not mix, and the thing this
// port has been missing for it. Without surface tension an "oil" blob has
// nothing holding it together: it does not stay round, it does not resist
// being stretched into a filament, droplets do not coalesce, and the
// interface frays into individual particles. Every one of those is a
// property of sigma, not of the density ratio the solver already has.
//
// Method: Continuum Surface Force (J. U. Brackbill, D. B. Kothe & C.
// Zemach, "A continuum method for modeling surface tension", J. Comput.
// Phys. 100, 1992). The interfacial force is spread over the finite
// thickness of the phase field rather than applied on a reconstructed
// surface, which is what makes it implementable on a plain grid:
//
//     f = sigma * kappa * grad(c)          [force per unit volume]
//     kappa = -div(n_hat),  n_hat = grad(c) / |grad(c)|
//
// Derived below for this port's own staggered layout rather than
// transcribed from any implementation -- see ../../docs/openfoam-two-phase-
// flow.md for why that distinction is load-bearing here. On a MAC grid the
// derivation is actually more natural than on a collocated one: n_hat's
// normal component wants to live exactly where a face velocity lives, and
// the divergence that turns it into a curvature then lands exactly on a cell
// centre.
//
// *** Sign, which is easy to get backwards and expensive to notice ***
//
// Take a round blob of c = 1 sitting in c = 0. grad(c) points inward, toward
// the blob, so n_hat does too. The divergence of an inward-pointing radial
// unit field is -1/r in 2D, so kappa = -div(n_hat) = +1/r, and
// f = sigma * kappa * grad(c) points inward as well: the force squeezes the
// blob and raises the pressure inside it. That is the correct behaviour, and
// it is checkable rather than a matter of opinion -- the pressure jump
// across the interface of a 2D blob at rest must be sigma/R (Young-Laplace),
// which is what examples/29-static-droplet/ measures.
//
// *** deltaN, and why the normalisation needs it ***
//
// grad(c) is zero everywhere except within the interface's own few cells, so
// normalising it is 0/0 across almost the whole domain. The standard
// treatment is a small additive stabiliser in the denominator, which makes
// n_hat harmlessly zero away from the interface instead of undefined. It has
// the units of a gradient, so it is scaled by the grid spacing rather than
// being an absolute constant -- an absolute one would mean something
// different at every resolution, which is exactly the kind of hidden
// per-scene constant this port has been removing.
//
// *** What this does not do ***
//
// No wall contact angle. A real implementation rotates n_hat at solid
// boundaries to impose a prescribed angle, which is what makes a droplet
// bead up or spread on a surface. Left out deliberately for a first pass:
// it is a boundary treatment with its own vocabulary, and none of the scenes
// this is being built for touch a wall with an interface yet.

import * as tsl_array_n from 'tsl_array_n';
import { float, max, min, sqrt } from 'three/tsl';

// Additive stabiliser for the |grad(c)| normalisation, as a fraction of a
// "one across one cell" gradient -- so it scales with the grid instead of
// being an absolute number that means something different at every
// resolution. Small enough to be irrelevant inside the interface, where
// |grad(c)| is of order 1/h.
const NORMAL_EPSILON_FRACTION = 1e-6;

function numberOrNode( value ) {

	return typeof value === 'number' ? float( value ) : value;

}

/**
 * Builds the surface-tension stage for one velocity grid and one phase
 * field.
 *
 * velocityGrid: FaceCenteredGrid2, modified in place by apply().
 * phase: a cell-centred accessor, c(i, j), varying 0 to 1 across the
 *   interface. A FLIP solver's `cellConcentration` is exactly this shape;
 *   so is its `fluidMask` for a liquid/air surface.
 * resolution/gridSpacing/origin: the velocity grid's own, in plain numbers,
 *   matching every other factory here.
 * sigma: surface tension coefficient, number or live node.
 * dt: number or live node, shared with the solver.
 * faceWeights: optional { u, v } beta accessors, as passed to
 *   grid_pressure_solver2.js. Surface tension is a force, so it accelerates
 *   a face by f/rho, and beta = rho_ref/rho_face is exactly the factor that
 *   converts. Omit for constant density.
 * referenceDensity: the density beta is relative to. Only consulted when
 *   faceWeights is given; defaults to 1.
 */
export function createSurfaceTension2( {
	velocityGrid,
	phase,
	resolution,
	gridSpacing = [ 1, 1 ],
	sigma,
	dt,
	faceWeights,
	referenceDensity = 1,
	curvatureSmoothing = 2
} ) {

	const [ resolutionX, resolutionY ] = resolution;
	const [ hx, hy ] = gridSpacing;
	const cellShape = [ resolutionX, resolutionY ];
	const dataSizeU = velocityGrid.dataSizeU;
	const dataSizeV = velocityGrid.dataSizeV;

	const sigmaNode = numberOrNode( sigma );
	const dtNode = numberOrNode( dt );
	const referenceDensityNode = numberOrNode( referenceDensity );
	const deltaN = float( NORMAL_EPSILON_FRACTION / Math.min( hx, hy ) );

	// Clamped index helpers -- a zero-gradient read outside the domain,
	// which is the right edge treatment for a phase field: the interface
	// meets a wall at whatever angle it arrives, since no contact angle is
	// imposed (see this file's own header).
	const ci = ( i ) => max( 0, min( i, resolutionX - 1 ) );
	const cj = ( j ) => max( 0, min( j, resolutionY - 1 ) );

	// ---- the field the curvature is measured from.
	//
	// *** Curvature is the weak part of CSF, and this is the standard
	// mitigation. Measured here, not assumed. ***
	//
	// The phase field a particle solver produces is a per-cell occupancy
	// average, so its interface is one or two cells wide and noticeably
	// lumpy -- particles land where they land. Differencing that twice, which
	// is what a curvature is, amplifies the lumpiness badly. The first
	// version of this module took the curvature straight from the raw field
	// and examples/29-static-droplet/ measured the pressure jump at 0.43 to
	// 0.67 of the Young-Laplace value, wandering between those over a few
	// hundred frames: the right sign and the right order, with a curvature
	// estimate too rough to be worth much.
	//
	// Smoothing the field before differentiating it is the usual answer, and
	// it is applied *only* to the normals and the curvature. The force still
	// uses the original grad(c), so the force stays located on the real
	// interface rather than being spread across the smoothed one -- smoothing
	// the force as well would trade a curvature error for an interface that
	// is thicker than it should be.
	const smoothingPasses = Math.max( 0, Math.floor( curvatureSmoothing ) );
	let phaseForCurvature = phase;
	let dispatchSmooth = null;

	if ( smoothingPasses > 0 ) {

		const smoothA = tsl_array_n.arrayN( 'float', cellShape );
		const smoothB = tsl_array_n.arrayN( 'float', cellShape );
		const zeros = new Float32Array( resolutionX * resolutionY );
		smoothA.fromArray( zeros );
		smoothB.fromArray( zeros );

		// A 1-2-1 kernel per axis, applied as a single 3x3 pass -- the
		// cheapest filter that is isotropic enough not to imprint the grid
		// axes onto a curvature.
		const buildPass = ( src, dst ) => tsl_array_n.kernel( cellShape, ( i, j ) => {

			const iL = ci( i.sub( 1 ) ), iR = ci( i.add( 1 ) );
			const jB = cj( j.sub( 1 ) ), jT = cj( j.add( 1 ) );

			const centre = src( i, j ).mul( 4 );
			const edges = src( iL, j ).add( src( iR, j ) ).add( src( i, jB ) ).add( src( i, jT ) ).mul( 2 );
			const corners = src( iL, jB ).add( src( iR, jB ) ).add( src( iL, jT ) ).add( src( iR, jT ) );

			dst( i, j ).assign( centre.add( edges ).add( corners ).div( 16 ) );

		} );

		// First pass reads the caller's field, the rest ping-pong.
		const first = buildPass( phase, smoothA );
		const forward = buildPass( smoothA, smoothB );
		const backward = buildPass( smoothB, smoothA );

		dispatchSmooth = () => {

			first();

			for ( let pass = 1; pass < smoothingPasses; pass ++ ) {

				if ( pass % 2 === 1 ) forward(); else backward();

			}

			// An odd number of extra passes leaves the result in smoothB, so
			// copy it back to keep the read below unconditional.
			if ( smoothingPasses > 1 && ( smoothingPasses - 1 ) % 2 === 1 ) backward();

		};

		phaseForCurvature = smoothA;

	}

	// The unit interface normal, stored per face and per component: nHatU
	// holds the x component on u-faces, nHatV the y component on v-faces.
	// Storing each component only where it is naturally defined is what
	// keeps the curvature below a plain difference of stored values rather
	// than an interpolation of interpolations.
	const nHatU = tsl_array_n.arrayN( 'float', dataSizeU );
	const nHatV = tsl_array_n.arrayN( 'float', dataSizeV );
	const curvature = tsl_array_n.arrayN( 'float', cellShape );

	nHatU.fromArray( new Float32Array( dataSizeU[ 0 ] * dataSizeU[ 1 ] ) );
	nHatV.fromArray( new Float32Array( dataSizeV[ 0 ] * dataSizeV[ 1 ] ) );
	curvature.fromArray( new Float32Array( resolutionX * resolutionY ) );

	// A u-face's own normal component is a plain difference of the two cells
	// it separates -- second-order accurate and centred exactly on the face,
	// with no interpolation at all. The tangential component it needs for
	// the magnitude has to be interpolated, and is: the average of the two
	// neighbouring cells' own centred y-derivatives.
	const computeNHatU = tsl_array_n.kernel( dataSizeU, ( i, j ) => {

		const iL = ci( i.sub( 1 ) );
		const iR = ci( i );

		const gx = phaseForCurvature( iR, j ).sub( phaseForCurvature( iL, j ) ).div( hx );

		const gyL = phaseForCurvature( iL, cj( j.add( 1 ) ) ).sub( phaseForCurvature( iL, cj( j.sub( 1 ) ) ) ).div( 2 * hy );
		const gyR = phaseForCurvature( iR, cj( j.add( 1 ) ) ).sub( phaseForCurvature( iR, cj( j.sub( 1 ) ) ) ).div( 2 * hy );
		const gy = gyL.add( gyR ).mul( 0.5 );

		const magnitude = sqrt( gx.mul( gx ).add( gy.mul( gy ) ) );

		nHatU( i, j ).assign( gx.div( magnitude.add( deltaN ) ) );

	} );

	const computeNHatV = tsl_array_n.kernel( dataSizeV, ( i, j ) => {

		const jB = cj( j.sub( 1 ) );
		const jT = cj( j );

		const gy = phaseForCurvature( i, jT ).sub( phaseForCurvature( i, jB ) ).div( hy );

		const gxB = phaseForCurvature( ci( i.add( 1 ) ), jB ).sub( phaseForCurvature( ci( i.sub( 1 ) ), jB ) ).div( 2 * hx );
		const gxT = phaseForCurvature( ci( i.add( 1 ) ), jT ).sub( phaseForCurvature( ci( i.sub( 1 ) ), jT ) ).div( 2 * hx );
		const gx = gxB.add( gxT ).mul( 0.5 );

		const magnitude = sqrt( gx.mul( gx ).add( gy.mul( gy ) ) );

		nHatV( i, j ).assign( gy.div( magnitude.add( deltaN ) ) );

	} );

	// kappa = -div(n_hat), which on a staggered grid is exactly a difference
	// of the stored face values -- the reason for storing them per face.
	const computeCurvature = tsl_array_n.kernel( cellShape, ( i, j ) => {

		const divergence = nHatU( i.add( 1 ), j ).sub( nHatU( i, j ) ).div( hx )
			.add( nHatV( i, j.add( 1 ) ).sub( nHatV( i, j ) ).div( hy ) );

		curvature( i, j ).assign( divergence.negate() );

	} );

	// f = sigma * kappa * grad(c), as a velocity increment: dt * f / rho.
	// The face's own curvature is the average of the two cells it separates,
	// which pairs it with the same grad(c) difference the normal above used.
	//
	// Domain-edge faces are left alone, matching grid_pressure_solver2.js's
	// own correction step: an interface pressed against a wall is a contact
	// angle problem, and this file does not do contact angles.
	const applyU = tsl_array_n.kernel( dataSizeU, ( i, j ) => {

		const inInterior = i.greaterThan( 0 ).and( i.lessThan( resolutionX ) );

		const iL = ci( i.sub( 1 ) );
		const iR = ci( i );

		const gradC = phase( iR, j ).sub( phase( iL, j ) ).div( hx );
		const kappaFace = curvature( iL, j ).add( curvature( iR, j ) ).mul( 0.5 );

		const acceleration = sigmaNode.mul( kappaFace ).mul( gradC )
			.mul( faceWeights ? faceWeights.u( i, j ).div( referenceDensityNode ) : float( 1 ).div( referenceDensityNode ) );

		velocityGrid.dataU( i, j ).addAssign( inInterior.select( acceleration.mul( dtNode ), float( 0 ) ) );

	} );

	const applyV = tsl_array_n.kernel( dataSizeV, ( i, j ) => {

		const inInterior = j.greaterThan( 0 ).and( j.lessThan( resolutionY ) );

		const jB = cj( j.sub( 1 ) );
		const jT = cj( j );

		const gradC = phase( i, jT ).sub( phase( i, jB ) ).div( hy );
		const kappaFace = curvature( i, jB ).add( curvature( i, jT ) ).mul( 0.5 );

		const acceleration = sigmaNode.mul( kappaFace ).mul( gradC )
			.mul( faceWeights ? faceWeights.v( i, j ).div( referenceDensityNode ) : float( 1 ).div( referenceDensityNode ) );

		velocityGrid.dataV( i, j ).addAssign( inInterior.select( acceleration.mul( dtNode ), float( 0 ) ) );

	} );

	/**
	 * One frame of surface tension: normals, curvature, then the velocity
	 * increment. Call it in the force stage, before the pressure projection
	 * -- like gravity, it is a force, and the projection is what turns it
	 * into the pressure jump it is supposed to produce.
	 */
	function apply() {

		if ( dispatchSmooth ) dispatchSmooth();

		computeNHatU();
		computeNHatV();
		computeCurvature();
		applyU();
		applyV();

	}

	return { apply, curvature, nHatU, nHatV };

}
