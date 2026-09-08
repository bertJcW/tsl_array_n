// A plain-JS reference implementation of the variable-density (two-phase)
// pressure projection that multigrid.js's `options.faceWeights` and
// grid_pressure_solver2.js's `options.faceWeights` implement on the GPU,
// plus the properties that discretization has to have to be correct at all.
//
// Why this file exists, and why it isn't just more structural tests:
// the GPU version of this stencil cannot be run in this dev/CI environment
// (the surrounding PCG solve needs real WebGPU atomics -- see
// reduction.test.js's own header comment), and it is exactly the kind of
// code where a sign or an index being one off produces something that
// still runs, still looks like a fluid, and is silently wrong. multigrid.js
// already carries a long header comment about a real sign bug of precisely
// that shape, found only after it had shipped. So the stencil is
// re-implemented here, in a form that CAN run without a GPU, and checked
// against the properties that actually pin it down.
//
// This is the same technique multigrid.js's own header comment credits for
// catching its earlier constant-diagonal bug ("caught by an independent
// plain-JS reference implementation of the same formula") -- applied up
// front this time rather than after the fact.
//
// The reference below MUST stay in step with laplacianAt /
// laplacianDiagonalAt in ../src/linalg/multigrid.js and with
// dispatchCorrectU/V in ../src/grid/grid_pressure_solver2.js. It mirrors:
// the +Laplacian (non-negated) sign convention, the drop-the-term-at-a-
// domain-edge Neumann treatment, the Dirichlet-neighbor elimination, the
// negated identity row for a Dirichlet cell, MAC face indexing for beta,
// and the `u = u* - beta grad(p)` correction.

import { describe, it, expect } from 'vitest';

const NX = 8, NY = 8, HX = 1, HY = 1;
const idx = ( i, j ) => i + NX * j;
const uFace = ( i, j ) => i + ( NX + 1 ) * j;
const vFace = ( i, j ) => i + NX * j;

// beta = rhoLiquid / rhoFace, with rhoFace the arithmetic average of the two
// adjacent cells -- and stored per FACE, which is what makes the operator
// symmetric (both neighbors read the same single value).
function makeBeta( rhoCell, rhoLiquid = 1 ) {
	const bu = new Float64Array( ( NX + 1 ) * NY );
	const bv = new Float64Array( NX * ( NY + 1 ) );
	const rhoAt = ( i, j ) => rhoCell[ idx(
		Math.min( NX - 1, Math.max( 0, i ) ),
		Math.min( NY - 1, Math.max( 0, j ) )
	) ];
	for ( let j = 0; j < NY; j ++ ) for ( let i = 0; i <= NX; i ++ ) {
		bu[ uFace( i, j ) ] = rhoLiquid / ( 0.5 * ( rhoAt( i - 1, j ) + rhoAt( i, j ) ) );
	}
	for ( let j = 0; j <= NY; j ++ ) for ( let i = 0; i < NX; i ++ ) {
		bv[ vFace( i, j ) ] = rhoLiquid / ( 0.5 * ( rhoAt( i, j - 1 ) + rhoAt( i, j ) ) );
	}
	return { bu, bv };
}

// Mirrors laplacianAt( field, spacing, shape, I, dirichletMask, faceWeights ).
function applyA( p, beta, isDirichlet ) {
	const out = new Float64Array( NX * NY );
	for ( let j = 0; j < NY; j ++ ) for ( let i = 0; i < NX; i ++ ) {
		const c = idx( i, j );
		if ( isDirichlet( i, j ) ) { out[ c ] = - p[ c ]; continue; } // negated identity row
		const center = p[ c ];
		let sum = 0;
		{
			let dLower = i > 0 ? center - p[ idx( i - 1, j ) ] : 0;
			let dUpper = i < NX - 1 ? p[ idx( i + 1, j ) ] - center : 0;
			if ( i > 0 && isDirichlet( i - 1, j ) ) dLower = center;
			if ( i < NX - 1 && isDirichlet( i + 1, j ) ) dUpper = - center;
			sum += ( dUpper * beta.bu[ uFace( i + 1, j ) ] - dLower * beta.bu[ uFace( i, j ) ] ) / ( HX * HX );
		}
		{
			let dLower = j > 0 ? center - p[ idx( i, j - 1 ) ] : 0;
			let dUpper = j < NY - 1 ? p[ idx( i, j + 1 ) ] - center : 0;
			if ( j > 0 && isDirichlet( i, j - 1 ) ) dLower = center;
			if ( j < NY - 1 && isDirichlet( i, j + 1 ) ) dUpper = - center;
			sum += ( dUpper * beta.bv[ vFace( i, j + 1 ) ] - dLower * beta.bv[ vFace( i, j ) ] ) / ( HY * HY );
		}
		out[ c ] = sum;
	}
	return out;
}

// Mirrors laplacianDiagonalAt( spacing, shape, I, dirichletMask, faceWeights )
// -- note it deliberately does NOT drop a Dirichlet *neighbor*'s slot, only a
// genuinely absent one, matching that function's own documented behavior.
function diagonalA( i, j, beta, isDirichlet ) {
	if ( isDirichlet( i, j ) ) return - 1;
	const x = ( i > 0 ? beta.bu[ uFace( i, j ) ] : 0 ) + ( i < NX - 1 ? beta.bu[ uFace( i + 1, j ) ] : 0 );
	const y = ( j > 0 ? beta.bv[ vFace( i, j ) ] : 0 ) + ( j < NY - 1 ? beta.bv[ vFace( i, j + 1 ) ] : 0 );
	return x * ( - 1 / ( HX * HX ) ) + y * ( - 1 / ( HY * HY ) );
}

function cg( beta, b, isDirichlet, iters = 5000 ) {
	const n = NX * NY;
	const x = new Float64Array( n );
	const r = b.slice();
	const pv = r.slice();
	let rr = r.reduce( ( a, v ) => a + v * v, 0 );
	for ( let k = 0; k < iters && rr > 1e-26; k ++ ) {
		const Ap = applyA( pv, beta, isDirichlet );
		let pAp = 0;
		for ( let i = 0; i < n; i ++ ) pAp += pv[ i ] * Ap[ i ];
		if ( Math.abs( pAp ) < 1e-30 ) break;
		const alpha = rr / pAp;
		for ( let i = 0; i < n; i ++ ) { x[ i ] += alpha * pv[ i ]; r[ i ] -= alpha * Ap[ i ]; }
		let rrNew = 0;
		for ( let i = 0; i < n; i ++ ) rrNew += r[ i ] * r[ i ];
		const bta = rrNew / rr;
		for ( let i = 0; i < n; i ++ ) pv[ i ] = r[ i ] + bta * pv[ i ];
		rr = rrNew;
	}
	return { x, residual: Math.sqrt( rr ) };
}

const divergence = ( u, v, i, j ) =>
	( u[ uFace( i + 1, j ) ] - u[ uFace( i, j ) ] ) / HX +
	( v[ vFace( i, j + 1 ) ] - v[ vFace( i, j ) ] ) / HY;

// Mirrors grid_pressure_solver2.js: b = divergence for a fluid cell,
// -target for a Dirichlet cell; correction is u = u* - beta grad(p).
function project( u, v, beta, isDirichlet ) {
	const b = new Float64Array( NX * NY );
	for ( let j = 0; j < NY; j ++ ) for ( let i = 0; i < NX; i ++ ) {
		b[ idx( i, j ) ] = isDirichlet( i, j ) ? - 0 : divergence( u, v, i, j );
	}
	const { x: p, residual } = cg( beta, b, isDirichlet );
	for ( let j = 0; j < NY; j ++ ) for ( let i = 1; i < NX; i ++ ) {
		u[ uFace( i, j ) ] -= beta.bu[ uFace( i, j ) ] * ( p[ idx( i, j ) ] - p[ idx( i - 1, j ) ] ) / HX;
	}
	for ( let j = 1; j < NY; j ++ ) for ( let i = 0; i < NX; i ++ ) {
		v[ vFace( i, j ) ] -= beta.bv[ vFace( i, j ) ] * ( p[ idx( i, j ) ] - p[ idx( i, j - 1 ) ] ) / HY;
	}
	return { p, residual };
}

// The scene every test below shares: liquid in the bottom half, gas above,
// and a gas bubble buried in the liquid -- i.e. the shape of what
// examples/24-two-phase-bubble-rise/ actually runs. Pin is mantaflow's own
// zeroPressureFixing position, top centre.
const RHO_L = 1, RHO_G = 0.01;
const rho = new Float64Array( NX * NY );
for ( let j = 0; j < NY; j ++ ) for ( let i = 0; i < NX; i ++ ) {
	const isBubble = ( i - 4 ) ** 2 + ( j - 2 ) ** 2 <= 1;
	rho[ idx( i, j ) ] = ( j >= NY / 2 || isBubble ) ? RHO_G : RHO_L;
}
const beta = makeBeta( rho, RHO_L );
const isDirichlet = ( i, j ) => i === Math.floor( NX / 2 ) && j === NY - 1;

function randomDivergentField() {
	const u = new Float64Array( ( NX + 1 ) * NY );
	const v = new Float64Array( NX * ( NY + 1 ) );
	let s = 12345;
	const rnd = () => ( s = ( s * 1103515245 + 12345 ) % 2147483648 ) / 2147483648 - 0.5;
	// Domain-boundary faces stay 0 -- no flux through a closed wall, which is
	// also what makes the whole-domain divergence sum to zero (the system's
	// compatibility condition).
	for ( let j = 0; j < NY; j ++ ) for ( let i = 1; i < NX; i ++ ) u[ uFace( i, j ) ] = rnd();
	for ( let j = 1; j < NY; j ++ ) for ( let i = 0; i < NX; i ++ ) v[ vFace( i, j ) ] = rnd();
	return { u, v };
}

describe( 'variable-density projection (reference for multigrid faceWeights)', () => {

	it( 'the operator is symmetric -- the property PCG actually requires', () => {

		// Built explicitly by applying A to each unit vector, so this tests
		// the coded stencil rather than restating the intended formula.
		const n = NX * NY;
		const columns = [];
		for ( let k = 0; k < n; k ++ ) {
			const e = new Float64Array( n );
			e[ k ] = 1;
			columns.push( applyA( e, beta, isDirichlet ) );
		}

		let worst = 0;
		for ( let a = 0; a < n; a ++ ) for ( let b = 0; b < n; b ++ ) {
			worst = Math.max( worst, Math.abs( columns[ a ][ b ] - columns[ b ][ a ] ) );
		}

		expect( worst ).toBeLessThan( 1e-12 );

	} );

	it( 'storing beta per CELL instead of per FACE would break that symmetry', () => {

		// The negative control for the test above: this is the tempting
		// alternative (each cell scales its own row by its own beta), written
		// out to show it is genuinely not equivalent. If this ever starts
		// passing, the symmetry test above has stopped testing anything.
		const n = NX * NY;
		const applyCellScaled = ( p ) => {
			const out = new Float64Array( n );
			for ( let j = 0; j < NY; j ++ ) for ( let i = 0; i < NX; i ++ ) {
				const c = idx( i, j );
				const bc = RHO_L / rho[ c ];
				let sum = 0;
				if ( i > 0 ) sum += ( p[ idx( i - 1, j ) ] - p[ c ] ) * bc;
				if ( i < NX - 1 ) sum += ( p[ idx( i + 1, j ) ] - p[ c ] ) * bc;
				if ( j > 0 ) sum += ( p[ idx( i, j - 1 ) ] - p[ c ] ) * bc;
				if ( j < NY - 1 ) sum += ( p[ idx( i, j + 1 ) ] - p[ c ] ) * bc;
				out[ c ] = sum;
			}
			return out;
		};

		const columns = [];
		for ( let k = 0; k < n; k ++ ) {
			const e = new Float64Array( n );
			e[ k ] = 1;
			columns.push( applyCellScaled( e ) );
		}
		let worst = 0;
		for ( let a = 0; a < n; a ++ ) for ( let b = 0; b < n; b ++ ) {
			worst = Math.max( worst, Math.abs( columns[ a ][ b ] - columns[ b ][ a ] ) );
		}

		expect( worst ).toBeGreaterThan( 1 );

	} );

	it( 'laplacianDiagonalAt agrees with the true diagonal of laplacianAt', () => {

		// The two are separate functions that must describe the same matrix;
		// multigrid.js's relax sweep divides by the diagonal, so a mismatch
		// over-corrects every sweep rather than merely converging slowly.
		let worst = 0;
		for ( let j = 0; j < NY; j ++ ) for ( let i = 0; i < NX; i ++ ) {
			const e = new Float64Array( NX * NY );
			e[ idx( i, j ) ] = 1;
			const column = applyA( e, beta, isDirichlet );
			worst = Math.max( worst, Math.abs( column[ idx( i, j ) ] - diagonalA( i, j, beta, isDirichlet ) ) );
		}

		expect( worst ).toBeLessThan( 1e-12 );

	} );

	it( 'the projection makes the velocity field divergence-free', () => {

		const { u, v } = randomDivergentField();

		let before = 0;
		for ( let j = 0; j < NY; j ++ ) for ( let i = 0; i < NX; i ++ ) {
			before = Math.max( before, Math.abs( divergence( u, v, i, j ) ) );
		}
		expect( before ).toBeGreaterThan( 0.1 ); // the input really is divergent

		project( u, v, beta, isDirichlet );

		let after = 0;
		for ( let j = 0; j < NY; j ++ ) for ( let i = 0; i < NX; i ++ ) {
			if ( isDirichlet( i, j ) ) continue; // the pinned cell is not solved for
			after = Math.max( after, Math.abs( divergence( u, v, i, j ) ) );
		}
		expect( after ).toBeLessThan( 1e-8 );

	} );

	it( 'correcting with an UNWEIGHTED gradient after a weighted solve does not', () => {

		// Negative control for the test above: the operator and the
		// correction are one derivation, and using `u* - grad(p)` with a
		// `div(beta grad(p))` solve leaves a field that is not
		// divergence-free at all. This is the failure mode the correction
		// step's own comment in grid_pressure_solver2.js warns about.
		const { u, v } = randomDivergentField();
		const b = new Float64Array( NX * NY );
		for ( let j = 0; j < NY; j ++ ) for ( let i = 0; i < NX; i ++ ) {
			b[ idx( i, j ) ] = isDirichlet( i, j ) ? - 0 : divergence( u, v, i, j );
		}
		const { x: p } = cg( beta, b, isDirichlet );

		for ( let j = 0; j < NY; j ++ ) for ( let i = 1; i < NX; i ++ ) {
			u[ uFace( i, j ) ] -= ( p[ idx( i, j ) ] - p[ idx( i - 1, j ) ] ) / HX;
		}
		for ( let j = 1; j < NY; j ++ ) for ( let i = 0; i < NX; i ++ ) {
			v[ vFace( i, j ) ] -= ( p[ idx( i, j ) ] - p[ idx( i, j - 1 ) ] ) / HY;
		}

		let after = 0;
		for ( let j = 0; j < NY; j ++ ) for ( let i = 0; i < NX; i ++ ) {
			if ( isDirichlet( i, j ) ) continue;
			after = Math.max( after, Math.abs( divergence( u, v, i, j ) ) );
		}

		expect( after ).toBeGreaterThan( 0.1 );

	} );

	it( 'buoyancy emerges from the density jump alone -- no buoyancy force anywhere', () => {

		// The single most important property of the whole two-phase solver,
		// and the one a reader is most likely to disbelieve: gravity is
		// applied UNIFORMLY to every face, nothing adds an upward term, and
		// the buried gas bubble still ends up moving up.
		const dt = 0.05, G = - 9.81;
		const u = new Float64Array( ( NX + 1 ) * NY );
		const v = new Float64Array( NX * ( NY + 1 ) );
		for ( let j = 1; j < NY; j ++ ) for ( let i = 0; i < NX; i ++ ) v[ vFace( i, j ) ] = G * dt;

		project( u, v, beta, isDirichlet );

		const bubbleV = ( v[ vFace( 4, 2 ) ] + v[ vFace( 4, 3 ) ] ) / 2;
		const liquidV = ( v[ vFace( 1, 2 ) ] + v[ vFace( 1, 3 ) ] ) / 2;

		expect( bubbleV ).toBeGreaterThan( 0 );                          // the bubble rises
		expect( Math.abs( liquidV ) ).toBeLessThan( Math.abs( bubbleV ) ); // the liquid does not

	} );

	it( 'a uniform-density scene reduces exactly to the constant-coefficient solve', () => {

		// beta == 1 on every face is what guarantees the shipped
		// single-phase scenes are unaffected by any of this.
		const uniform = makeBeta( new Float64Array( NX * NY ).fill( 1 ), 1 );
		for ( const face of [ uniform.bu, uniform.bv ] ) {
			for ( const value of face ) expect( value ).toBeCloseTo( 1, 12 );
		}

		const { u, v } = randomDivergentField();
		project( u, v, uniform, isDirichlet );

		let after = 0;
		for ( let j = 0; j < NY; j ++ ) for ( let i = 0; i < NX; i ++ ) {
			if ( isDirichlet( i, j ) ) continue;
			after = Math.max( after, Math.abs( divergence( u, v, i, j ) ) );
		}
		expect( after ).toBeLessThan( 1e-8 );

	} );

	it( 'beta is >= 1 everywhere and largest in the gas', () => {

		let liquidBeta = Infinity, gasBeta = 0, smallest = Infinity;
		for ( const value of beta.bu ) smallest = Math.min( smallest, value );
		for ( const value of beta.bv ) smallest = Math.min( smallest, value );
		for ( let j = 0; j < 2; j ++ ) liquidBeta = Math.min( liquidBeta, beta.bu[ uFace( 1, j ) ] );
		for ( let j = 5; j < 7; j ++ ) gasBeta = Math.max( gasBeta, beta.bu[ uFace( 1, j ) ] );

		expect( smallest ).toBeCloseTo( 1, 12 );                 // normalized on the liquid
		expect( liquidBeta ).toBeCloseTo( 1, 12 );
		expect( gasBeta ).toBeCloseTo( RHO_L / RHO_G, 6 );       // the full density ratio

	} );

} );
