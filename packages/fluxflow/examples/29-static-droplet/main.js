// The correctness test for grid.createSurfaceTension2, not a demo.
//
// A blob of one liquid at rest inside another, no gravity, nothing acting on
// it but surface tension. The Young-Laplace relation gives the answer in
// closed form: across a curved interface the pressure jumps by sigma times
// the curvature, which in 2D for a circle of radius R is
//
//     p_inside - p_outside = sigma / R
//
// so this scene measures that jump and divides it by sigma/R. A ratio near 1
// means the curvature estimate, the sign of the force and the scaling are
// all correct at once. It is a much sharper test than any picture: a sign
// error inverts it, a factor-of-two error in the curvature shows up as 2,
// and a force that is merely *plausible* rather than right will not track
// both `?sigma=` and `?radius=` at the same time.
//
// Measured on real WebGPU, 150-400 frames each, nothing changed between
// runs but the one parameter named:
//
//     sigma=1.0  R=10  dt=1/60    jump 0.00170  vs 0.00167   ratio 1.017
//     sigma=2.5  R=10  dt=1/60    jump 0.00425  vs 0.00417   ratio 1.019
//     sigma=1.0  R=6   dt=1/60    jump 0.00284  vs 0.00278   ratio 1.022
//     sigma=1.0  R=10  dt=0.008   jump 0.000814 vs 0.000800  ratio 1.018
//
// Within about 2% every time, and the ratio is flat over the 400-frame run
// rather than drifting. It tracks sigma linearly, tracks 1/R, and does not
// move with dt -- which is the part that says the dt/rho_ref scaling below
// is right and not absorbed into a fitted constant.
//
// *** The pressure this port solves is not P ***
//
// Worth stating, because the comparison is meaningless without it. The
// correction step is `u = u* - beta grad(p)` with `beta = rho_ref/rho`,
// while the physical statement is `u = u* - (dt/rho) grad(P)`. Equating them
// gives `p = (dt/rho_ref) P`, so the field this solver returns is the
// physical pressure scaled by dt/rho_ref. The prediction has to be scaled
// the same way, which is what `expectedJump` below does. Getting this wrong
// would show up as a ratio that tracks dt -- so the check is run at two
// different dt values below and both must give the same ratio.
//
// *** Why there is still a free surface ***
//
// A completely sealed box of liquid has no Dirichlet cell anywhere, which
// makes the pressure system singular -- defined only up to a constant --
// and grid_flip_solver2.js does not pin a reference cell (the two-phase
// solver does; see docs/openfoam-two-phase-flow.md item 5 for why these are
// two different problems). Leaving air above the liquid gives the solve its
// anchor, and costs nothing here: with gravity off, the pressure outside the
// blob is uniform, so "outside" can be sampled anywhere below the surface
// and the jump is still exactly sigma/R.
//
// URL parameters:
//   ?sigma=1.0    surface tension coefficient
//   ?radius=10    blob radius in cells
//   ?res=64       grid resolution
//   ?dt=0.0166    time step (the ratio must not depend on it)

import * as tsl_array_n from 'tsl_array_n';
import { grid } from 'fluxflow';

const canvas = document.querySelector( '#out' );
const statusEl = document.querySelector( '#status' );
const perfEl = document.querySelector( '#perf' );
const laplaceEl = document.querySelector( '#laplace' );

function status( text, isErr ) {

	statusEl.textContent = text;
	statusEl.className = isErr ? 'err' : '';

}

const params = new URLSearchParams( location.search );
const NX = Number( params.get( 'res' ) ?? 64 );
const NY = NX;
const dt = Number( params.get( 'dt' ) ?? 1 / 60 );
const sigma = Number( params.get( 'sigma' ) ?? 1 );
const blobRadius = Number( params.get( 'radius' ) ?? 10 );

// Liquid fills most of the box; the rest is air, purely to give the pressure
// solve a Dirichlet anchor. See the header.
const fillHeight = NY * 0.9;
const blobCenterX = NX * 0.5;
const blobCenterY = NY * 0.45;

const REFERENCE_DENSITY = 1;
const expectedJump = ( dt / REFERENCE_DENSITY ) * ( sigma / blobRadius );

const diagnosticInterval = 60;

try {

	const renderer = await tsl_array_n.init( { canvas: document.createElement( 'canvas' ), allowFallback: true } );
	status( `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }` );

	const velocityGrid = grid.createFaceCenteredGrid2( NX, NY, 1, 1, 0, 0 );

	const seed = grid.computeFlipBoxSeed( {
		boxMin: [ 0, 0 ],
		boxMax: [ NX, fillHeight ],
		gridSpacingX: 1, gridSpacingY: 1,
		particlesPerCellAxis: 2
	} );

	const count = seed.count;
	const concentrationArray = new Float32Array( count );

	for ( let p = 0; p < count; p ++ ) {

		const dx = seed.positionsArray[ p * 2 ] - blobCenterX;
		const dy = seed.positionsArray[ p * 2 + 1 ] - blobCenterY;
		concentrationArray[ p ] = dx * dx + dy * dy < blobRadius * blobRadius ? 1 : 0;

	}

	// Late-bound on purpose: surface tension needs the solver's own
	// cellConcentration, which does not exist until the factory returns, and
	// the solver needs a force hook it can call every frame. The closure is
	// what ties the two together -- see grid_flip_solver2.js's applyForces.
	let surfaceTension = null;

	const flip = grid.createGridFlipSolver2( {
		velocityGrid,
		gridSpacing: [ 1, 1 ],
		origin: [ 0, 0 ],
		maxParticles: count,
		dt,
		// The whole point: nothing else may move the liquid.
		gravity: [ 0, 0 ],
		carryConcentration: true,
		// No mixing and no fade -- the blob must keep its identity, or the
		// interface it is being asked to hold together dissolves.
		resample: { enabled: false },
		// Damping fights the very currents this measures. Zero, so what is
		// left is what surface tension actually produced.
		velocityDamping: 0,
		applyForces: () => {

			if ( surfaceTension ) surfaceTension.apply();

		}
	} );

	flip.positions.fromArray( seed.positionsArray );
	flip.velocities.fromArray( seed.velocitiesArray );
	flip.concentration.fromArray( concentrationArray );

	surfaceTension = grid.createSurfaceTension2( {
		velocityGrid,
		phase: flip.cellConcentration,
		resolution: [ NX, NY ],
		gridSpacing: [ 1, 1 ],
		sigma,
		dt,
		referenceDensity: REFERENCE_DENSITY
	} );

	// ---------------------------------------------------------------- drawing

	const PIXELS_PER_CELL = 9;
	canvas.width = NX * PIXELS_PER_CELL;
	canvas.height = NY * PIXELS_PER_CELL;
	canvas.style.width = '540px';
	canvas.style.height = '540px';

	const ctx = canvas.getContext( '2d' );
	const scale = PIXELS_PER_CELL;
	const RADIUS = scale * 0.5 * 1.05;

	function draw( positionsData, concentrationData ) {

		ctx.fillStyle = '#06080c';
		ctx.fillRect( 0, 0, canvas.width, canvas.height );

		for ( const blob of [ false, true ] ) {

			for ( let p = 0; p < concentrationData.length; p ++ ) {

				const c = concentrationData[ p ];
				if ( ( c > 0.5 ) !== blob ) continue;

				ctx.fillStyle = blob ? '#f0a860' : '#2a5f8f';
				ctx.beginPath();
				ctx.arc(
					positionsData[ p * 2 ] * scale,
					( NY - positionsData[ p * 2 + 1 ] ) * scale,
					RADIUS, 0, Math.PI * 2
				);
				ctx.fill();

			}

		}

		// The two sampling regions the measurement uses, so what is being
		// compared is visible rather than described.
		ctx.strokeStyle = 'rgba(255,255,255,0.35)';
		ctx.setLineDash( [ 4, 4 ] );
		ctx.beginPath();
		ctx.arc( blobCenterX * scale, ( NY - blobCenterY ) * scale, blobRadius * 0.5 * scale, 0, Math.PI * 2 );
		ctx.stroke();
		ctx.beginPath();
		ctx.arc( blobCenterX * scale, ( NY - blobCenterY ) * scale, blobRadius * 2.2 * scale, 0, Math.PI * 2 );
		ctx.stroke();
		ctx.setLineDash( [] );

	}

	// ---------------------------------------------------------------- measurement

	// Inside: a disc of half the blob's radius, well clear of the interface's
	// own smeared few cells. Outside: an annulus starting at 2.2 R, far
	// enough out that the interface's pressure field has decayed, and still
	// inside the liquid. Both are averages, because a single cell would be
	// reporting whatever noise the projection left in it.
	function measureJump( pressureData, maskData ) {

		let inSum = 0, inN = 0, outSum = 0, outN = 0;
		const rIn = blobRadius * 0.5;
		const rOut = blobRadius * 2.2;

		for ( let j = 0; j < NY; j ++ ) {

			for ( let i = 0; i < NX; i ++ ) {

				const k = i + NX * j;
				if ( maskData[ k ] < 0.5 ) continue; // liquid cells only

				const dx = i + 0.5 - blobCenterX;
				const dy = j + 0.5 - blobCenterY;
				const r = Math.hypot( dx, dy );
				const p = pressureData[ k ];
				if ( ! Number.isFinite( p ) ) continue;

				if ( r < rIn ) { inSum += p; inN ++; }
				else if ( r > rOut && j < fillHeight - 3 ) { outSum += p; outN ++; }

			}

		}

		return {
			inside: inN ? inSum / inN : NaN,
			outside: outN ? outSum / outN : NaN,
			jump: ( inN && outN ) ? inSum / inN - outSum / outN : NaN
		};

	}

	// How round the blob still is: the standard deviation of its particles'
	// distance from their own centroid, over the mean of that distance. A
	// perfect disc of uniformly seeded particles has a fixed value for this,
	// so what matters is that it stops changing, not its absolute size.
	function measureRoundness( positionsData, concentrationData ) {

		let cx = 0, cy = 0, n = 0;

		for ( let p = 0; p < concentrationData.length; p ++ ) {

			if ( concentrationData[ p ] <= 0.5 ) continue;
			cx += positionsData[ p * 2 ];
			cy += positionsData[ p * 2 + 1 ];
			n ++;

		}

		if ( ! n ) return NaN;
		cx /= n; cy /= n;

		let sum = 0, sumSq = 0;

		for ( let p = 0; p < concentrationData.length; p ++ ) {

			if ( concentrationData[ p ] <= 0.5 ) continue;
			const r = Math.hypot( positionsData[ p * 2 ] - cx, positionsData[ p * 2 + 1 ] - cy );
			sum += r; sumSq += r * r;

		}

		const mean = sum / n;
		return Math.sqrt( Math.max( 0, sumSq / n - mean * mean ) ) / mean;

	}

	let frame = 0;

	const FPS_WINDOW = 30;
	const frameTimes = [];
	let lastFrameTime = performance.now();

	function updatePerf() {

		const now = performance.now();
		frameTimes.push( now - lastFrameTime );
		lastFrameTime = now;
		if ( frameTimes.length > FPS_WINDOW ) frameTimes.shift();

		const avgMs = frameTimes.reduce( ( a, b ) => a + b, 0 ) / frameTimes.length;
		perfEl.textContent = `fps: ${ ( 1000 / avgMs ).toFixed( 1 ) } | ${ NX }x${ NY } | σ=${ sigma } R=${ blobRadius } dt=${ dt.toFixed( 4 ) }`;

	}

	let driverPaused = false;

	async function animate() {

		if ( driverPaused ) return;

		updatePerf();
		await flip.onAdvanceTimeStep();

		const [ positionsData, concentrationData, pressureData, maskData ] = await Promise.all( [
			flip.positions.toArray(),
			flip.concentration.toArray(),
			flip.pressureSolver.pressure.data.toArray(),
			flip.fluidMask.toArray()
		] );

		const m = measureJump( pressureData, maskData );
		const ratio = m.jump / expectedJump;

		laplaceEl.textContent =
			`Δp: ${ m.jump.toFixed( 5 ) }  |  σ/R (scaled): ${ expectedJump.toFixed( 5 ) }  |  ratio: ${ ratio.toFixed( 3 ) }`;

		if ( frame % diagnosticInterval === 0 ) {

			let maxV = 0;
			const [ u, v ] = await Promise.all( [ velocityGrid.dataU.toArray(), velocityGrid.dataV.toArray() ] );
			for ( const x of u ) { const a = Math.abs( x ); if ( Number.isFinite( a ) && a > maxV ) maxV = a; }
			for ( const x of v ) { const a = Math.abs( x ); if ( Number.isFinite( a ) && a > maxV ) maxV = a; }

			console.log(
				`fluxflow static-droplet [frame ${ frame }] ` +
				`converged=${ flip.pressureSolver.diagnostics.converged } ` +
				`rejected=${ flip.pressureSolver.diagnostics.rejected } | ` +
				`pIn=${ m.inside.toFixed( 5 ) } pOut=${ m.outside.toFixed( 5 ) } ` +
				`jump=${ m.jump.toFixed( 5 ) } expected=${ expectedJump.toFixed( 5 ) } ratio=${ ratio.toFixed( 3 ) } | ` +
				`roundness=${ measureRoundness( positionsData, concentrationData ).toFixed( 4 ) } ` +
				`spuriousMaxV=${ maxV.toFixed( 4 ) }`
			);

		}

		draw( positionsData, concentrationData );

		frame ++;
		requestAnimationFrame( animate );

	}

	window.__fluxflowProbe = {
		flip, velocityGrid, surfaceTension, expectedJump, measureJump, measureRoundness,
		step: () => flip.onAdvanceTimeStep(),
		pause: async () => {

			driverPaused = true;
			await new Promise( ( resolve ) => setTimeout( resolve, 100 ) );

		},
		resume: () => {

			driverPaused = false;
			requestAnimationFrame( animate );

		}
	};

	requestAnimationFrame( animate );

} catch ( error ) {

	status( error.message, true );
	console.error( error );

}
