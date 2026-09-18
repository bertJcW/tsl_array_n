// 3D counterpart of surface_tension2.js -- same Continuum Surface Force
// method (Brackbill, Kothe & Zemach 1992), same staggered-grid derivation
// (a face's own curvature is a plain difference of per-face-stored unit-
// normal components -- see that file's own header comment for the sign
// derivation and the deltaN/curvatureSmoothing reasoning, none of which is
// dimension-specific). What changes: each face now needs TWO tangential
// gradient components averaged (not one) for its normal's magnitude, and
// the smoothing pre-pass is one axis wider.
//
// *** The one deliberate simplification versus a mechanical 2D extension ***
//
// 2D's smoothing pass is a single 3x3 (1-2-1)-per-axis kernel, one
// dispatch. The 3D equivalent is a 3x3x3 (1-2-1)-per-axis kernel, 27 taps
// with weights (8/4/2/1 by corner/edge/face/centre distance) that are easy
// to get subtly wrong by hand. Since a 2D (1,2,1)-per-axis blur is exactly
// two separable 1D (1,2,1)/4 passes (blur along x, then along y), this
// file builds the 3D one the same way -- three separable 1D passes instead
// of one fused 3D kernel. More dispatches per smoothing pass (3 instead of
// 1), same mathematical result, and no 27-term weight table to
// hand-verify.

import * as tsl_array_n from 'tsl_array_n';
import { float, max, min, sqrt } from 'three/tsl';

const NORMAL_EPSILON_FRACTION = 1e-6;

function numberOrNode( value ) {

	return typeof value === 'number' ? float( value ) : value;

}

// See surface_tension2.js's own createSurfaceTension2 header comment for
// the full parameter list -- unchanged conventions, resolution/gridSpacing
// are now triples. phase: a cell-centred accessor c(i,j,k).
export function createSurfaceTension3( {
	velocityGrid,
	phase,
	resolution,
	gridSpacing = [ 1, 1, 1 ],
	sigma,
	dt,
	faceWeights,
	referenceDensity = 1,
	curvatureSmoothing = 2
} ) {

	const [ resolutionX, resolutionY, resolutionZ ] = resolution;
	const [ hx, hy, hz ] = gridSpacing;
	const cellShape = [ resolutionX, resolutionY, resolutionZ ];
	const dataSizeU = velocityGrid.dataSizeU;
	const dataSizeV = velocityGrid.dataSizeV;
	const dataSizeW = velocityGrid.dataSizeW;

	const sigmaNode = numberOrNode( sigma );
	const dtNode = numberOrNode( dt );
	const referenceDensityNode = numberOrNode( referenceDensity );
	const deltaN = float( NORMAL_EPSILON_FRACTION / Math.min( hx, hy, hz ) );

	const ci = ( i ) => max( 0, min( i, resolutionX - 1 ) );
	const cj = ( j ) => max( 0, min( j, resolutionY - 1 ) );
	const ck = ( k ) => max( 0, min( k, resolutionZ - 1 ) );

	// ---- the field the curvature is measured from (see this file's own
	// header comment for why this is 3 separable 1D passes, not one fused
	// 3D kernel).
	const smoothingPasses = Math.max( 0, Math.floor( curvatureSmoothing ) );
	let phaseForCurvature = phase;
	let dispatchSmooth = null;

	if ( smoothingPasses > 0 ) {

		const bufA = tsl_array_n.arrayN( 'float', cellShape );
		const bufB = tsl_array_n.arrayN( 'float', cellShape );
		const zeros = new Float32Array( resolutionX * resolutionY * resolutionZ );
		bufA.fromArray( zeros );
		bufB.fromArray( zeros );

		const blurX = ( src, dst ) => tsl_array_n.kernel( cellShape, ( i, j, k ) => {

			const v = src( ci( i.sub( 1 ) ), j, k ).add( src( ci( i.add( 1 ) ), j, k ) ).add( src( i, j, k ).mul( 2 ) );
			dst( i, j, k ).assign( v.div( 4 ) );

		} );

		const blurY = ( src, dst ) => tsl_array_n.kernel( cellShape, ( i, j, k ) => {

			const v = src( i, cj( j.sub( 1 ) ), k ).add( src( i, cj( j.add( 1 ) ), k ) ).add( src( i, j, k ).mul( 2 ) );
			dst( i, j, k ).assign( v.div( 4 ) );

		} );

		const blurZ = ( src, dst ) => tsl_array_n.kernel( cellShape, ( i, j, k ) => {

			const v = src( i, j, ck( k.sub( 1 ) ) ).add( src( i, j, ck( k.add( 1 ) ) ) ).add( src( i, j, k ).mul( 2 ) );
			dst( i, j, k ).assign( v.div( 4 ) );

		} );

		// One "smoothing pass" is 3 dispatches (x, y, z), ping-ponging
		// through bufA/bufB. First pass reads the caller's own field.
		const firstX = blurX( phase, bufA );
		const firstY = blurY( bufA, bufB );
		const firstZ = blurZ( bufB, bufA );

		const nextX = blurX( bufA, bufB );
		const nextY = blurY( bufB, bufA );
		const nextZ = blurZ( bufA, bufB );

		const nextX2 = blurX( bufB, bufA );
		const nextY2 = blurY( bufA, bufB );
		const nextZ2 = blurZ( bufB, bufA );

		dispatchSmooth = () => {

			firstX(); firstY(); firstZ(); // -> bufA

			for ( let pass = 1; pass < smoothingPasses; pass ++ ) {

				if ( pass % 2 === 1 ) { nextX(); nextY(); nextZ(); } // bufA -> bufB
				else { nextX2(); nextY2(); nextZ2(); } // bufB -> bufA

			}

			// Land back in bufA regardless of parity, so the read below is
			// unconditional -- same reasoning as surface_tension2.js's own
			// end-of-loop copy-back for an odd extra-pass count.
			if ( smoothingPasses > 1 && ( smoothingPasses - 1 ) % 2 === 1 ) { nextX2(); nextY2(); nextZ2(); }

		};

		phaseForCurvature = bufA;

	}

	// Per-face, per-component unit normal -- nHatU holds the x component on
	// u-faces, nHatV the y component on v-faces, nHatW the z component on
	// w-faces (one dimension wider than 2D's nHatU/nHatV pair).
	const nHatU = tsl_array_n.arrayN( 'float', dataSizeU );
	const nHatV = tsl_array_n.arrayN( 'float', dataSizeV );
	const nHatW = tsl_array_n.arrayN( 'float', dataSizeW );
	const curvature = tsl_array_n.arrayN( 'float', cellShape );

	nHatU.fromArray( new Float32Array( dataSizeU[ 0 ] * dataSizeU[ 1 ] * dataSizeU[ 2 ] ) );
	nHatV.fromArray( new Float32Array( dataSizeV[ 0 ] * dataSizeV[ 1 ] * dataSizeV[ 2 ] ) );
	nHatW.fromArray( new Float32Array( dataSizeW[ 0 ] * dataSizeW[ 1 ] * dataSizeW[ 2 ] ) );
	curvature.fromArray( new Float32Array( resolutionX * resolutionY * resolutionZ ) );

	// A u-face's own normal (x) component is a plain difference of the two
	// cells it separates. The magnitude needs BOTH tangential components
	// (y and z) too, each the average of the two neighbouring cells' own
	// centred derivatives along that axis -- one more averaged pair than
	// 2D's single tangential component.
	const computeNHatU = tsl_array_n.kernel( dataSizeU, ( i, j, k ) => {

		const iL = ci( i.sub( 1 ) );
		const iR = ci( i );

		const gx = phaseForCurvature( iR, j, k ).sub( phaseForCurvature( iL, j, k ) ).div( hx );

		const gyL = phaseForCurvature( iL, cj( j.add( 1 ) ), k ).sub( phaseForCurvature( iL, cj( j.sub( 1 ) ), k ) ).div( 2 * hy );
		const gyR = phaseForCurvature( iR, cj( j.add( 1 ) ), k ).sub( phaseForCurvature( iR, cj( j.sub( 1 ) ), k ) ).div( 2 * hy );
		const gy = gyL.add( gyR ).mul( 0.5 );

		const gzL = phaseForCurvature( iL, j, ck( k.add( 1 ) ) ).sub( phaseForCurvature( iL, j, ck( k.sub( 1 ) ) ) ).div( 2 * hz );
		const gzR = phaseForCurvature( iR, j, ck( k.add( 1 ) ) ).sub( phaseForCurvature( iR, j, ck( k.sub( 1 ) ) ) ).div( 2 * hz );
		const gz = gzL.add( gzR ).mul( 0.5 );

		const magnitude = sqrt( gx.mul( gx ).add( gy.mul( gy ) ).add( gz.mul( gz ) ) );

		nHatU( i, j, k ).assign( gx.div( magnitude.add( deltaN ) ) );

	} );

	const computeNHatV = tsl_array_n.kernel( dataSizeV, ( i, j, k ) => {

		const jB = cj( j.sub( 1 ) );
		const jT = cj( j );

		const gy = phaseForCurvature( i, jT, k ).sub( phaseForCurvature( i, jB, k ) ).div( hy );

		const gxB = phaseForCurvature( ci( i.add( 1 ) ), jB, k ).sub( phaseForCurvature( ci( i.sub( 1 ) ), jB, k ) ).div( 2 * hx );
		const gxT = phaseForCurvature( ci( i.add( 1 ) ), jT, k ).sub( phaseForCurvature( ci( i.sub( 1 ) ), jT, k ) ).div( 2 * hx );
		const gx = gxB.add( gxT ).mul( 0.5 );

		const gzB = phaseForCurvature( i, jB, ck( k.add( 1 ) ) ).sub( phaseForCurvature( i, jB, ck( k.sub( 1 ) ) ) ).div( 2 * hz );
		const gzT = phaseForCurvature( i, jT, ck( k.add( 1 ) ) ).sub( phaseForCurvature( i, jT, ck( k.sub( 1 ) ) ) ).div( 2 * hz );
		const gz = gzB.add( gzT ).mul( 0.5 );

		const magnitude = sqrt( gx.mul( gx ).add( gy.mul( gy ) ).add( gz.mul( gz ) ) );

		nHatV( i, j, k ).assign( gy.div( magnitude.add( deltaN ) ) );

	} );

	const computeNHatW = tsl_array_n.kernel( dataSizeW, ( i, j, k ) => {

		const kB = ck( k.sub( 1 ) );
		const kF = ck( k );

		const gz = phaseForCurvature( i, j, kF ).sub( phaseForCurvature( i, j, kB ) ).div( hz );

		const gxB = phaseForCurvature( ci( i.add( 1 ) ), j, kB ).sub( phaseForCurvature( ci( i.sub( 1 ) ), j, kB ) ).div( 2 * hx );
		const gxF = phaseForCurvature( ci( i.add( 1 ) ), j, kF ).sub( phaseForCurvature( ci( i.sub( 1 ) ), j, kF ) ).div( 2 * hx );
		const gx = gxB.add( gxF ).mul( 0.5 );

		const gyB = phaseForCurvature( i, cj( j.add( 1 ) ), kB ).sub( phaseForCurvature( i, cj( j.sub( 1 ) ), kB ) ).div( 2 * hy );
		const gyF = phaseForCurvature( i, cj( j.add( 1 ) ), kF ).sub( phaseForCurvature( i, cj( j.sub( 1 ) ), kF ) ).div( 2 * hy );
		const gy = gyB.add( gyF ).mul( 0.5 );

		const magnitude = sqrt( gx.mul( gx ).add( gy.mul( gy ) ).add( gz.mul( gz ) ) );

		nHatW( i, j, k ).assign( gz.div( magnitude.add( deltaN ) ) );

	} );

	// kappa = -div(n_hat) -- on a staggered grid, exactly a difference of
	// the stored per-face values, one term per axis.
	const computeCurvature = tsl_array_n.kernel( cellShape, ( i, j, k ) => {

		const divergence = nHatU( i.add( 1 ), j, k ).sub( nHatU( i, j, k ) ).div( hx )
			.add( nHatV( i, j.add( 1 ), k ).sub( nHatV( i, j, k ) ).div( hy ) )
			.add( nHatW( i, j, k.add( 1 ) ).sub( nHatW( i, j, k ) ).div( hz ) );

		curvature( i, j, k ).assign( divergence.negate() );

	} );

	// f = sigma * kappa * grad(c), as a velocity increment: dt * f / rho.
	// The face's own curvature is the average of the two cells it
	// separates. Domain-edge faces are left alone (no contact angle -- see
	// this file's own header comment). Written out explicitly per
	// component, same as surface_tension2.js's own applyU/applyV, rather
	// than a generic component-indexed helper -- this is the numerically
	// load-bearing part of the file, and explicit code is easier to check
	// against the 2D original term-by-term than a cleverer abstraction
	// would be.
	const applyU = tsl_array_n.kernel( dataSizeU, ( i, j, k ) => {

		const inInterior = i.greaterThan( 0 ).and( i.lessThan( resolutionX ) );

		const iL = ci( i.sub( 1 ) );
		const iR = ci( i );

		const gradC = phase( iR, j, k ).sub( phase( iL, j, k ) ).div( hx );
		const kappaFace = curvature( iL, j, k ).add( curvature( iR, j, k ) ).mul( 0.5 );

		const acceleration = sigmaNode.mul( kappaFace ).mul( gradC )
			.mul( faceWeights ? faceWeights.u( i, j, k ).div( referenceDensityNode ) : float( 1 ).div( referenceDensityNode ) );

		velocityGrid.dataU( i, j, k ).addAssign( inInterior.select( acceleration.mul( dtNode ), float( 0 ) ) );

	} );

	const applyV = tsl_array_n.kernel( dataSizeV, ( i, j, k ) => {

		const inInterior = j.greaterThan( 0 ).and( j.lessThan( resolutionY ) );

		const jB = cj( j.sub( 1 ) );
		const jT = cj( j );

		const gradC = phase( i, jT, k ).sub( phase( i, jB, k ) ).div( hy );
		const kappaFace = curvature( i, jB, k ).add( curvature( i, jT, k ) ).mul( 0.5 );

		const acceleration = sigmaNode.mul( kappaFace ).mul( gradC )
			.mul( faceWeights ? faceWeights.v( i, j, k ).div( referenceDensityNode ) : float( 1 ).div( referenceDensityNode ) );

		velocityGrid.dataV( i, j, k ).addAssign( inInterior.select( acceleration.mul( dtNode ), float( 0 ) ) );

	} );

	const applyW = tsl_array_n.kernel( dataSizeW, ( i, j, k ) => {

		const inInterior = k.greaterThan( 0 ).and( k.lessThan( resolutionZ ) );

		const kB = ck( k.sub( 1 ) );
		const kF = ck( k );

		const gradC = phase( i, j, kF ).sub( phase( i, j, kB ) ).div( hz );
		const kappaFace = curvature( i, j, kB ).add( curvature( i, j, kF ) ).mul( 0.5 );

		const acceleration = sigmaNode.mul( kappaFace ).mul( gradC )
			.mul( faceWeights ? faceWeights.w( i, j, k ).div( referenceDensityNode ) : float( 1 ).div( referenceDensityNode ) );

		velocityGrid.dataW( i, j, k ).addAssign( inInterior.select( acceleration.mul( dtNode ), float( 0 ) ) );

	} );

	function apply() {

		if ( dispatchSmooth ) dispatchSmooth();

		computeNHatU();
		computeNHatV();
		computeNHatW();
		computeCurvature();
		applyU();
		applyV();
		applyW();

	}

	return { apply, curvature, nHatU, nHatV, nHatW };

}
