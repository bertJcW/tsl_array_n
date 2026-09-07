// A "mushroom cloud" explosion demo, per the user's own explicit request,
// built entirely on top of the reusable grid.createGridSmokeSolver2 (see
// examples/17-smoke-fire/main.js's own header comment for the solver's
// design, and packages/fluxflow/src/grid/grid_smoke_solver2.js for the
// implementation) -- no new library code, this file is scene design only.
//
// *** What's actually different from examples/17-smoke-fire/, and why ***
//
// examples/17 continuously re-injects a small source every frame (a
// steady flame). This is the opposite: ONE large, hot, dense disc plus a
// brief outward velocity impulse, applied exactly once (detonate(), called
// on load and again on the "detonate again" button), then nothing further
// -- no per-frame source at all. Everything that happens afterward is the
// solver's own buoyancy + advection acting on that single initial
// condition, not scripted.
//
// The mushroom-cap shape this reliably produces (if it does) is a real,
// well-documented fluid-dynamics phenomenon, not something drawn or
// special-cased here: a large buoyant blob released into cooler, denser
// surrounding fluid rolls up into a toroidal vortex ring as it rises
// (the same starting-plume/rising-thermal mechanism classical CFD
// demonstrations and real explosion simulations both rely on) -- an
// emergent consequence of this solver's existing buoyancy+advection, not
// a new mechanism. If it does NOT look convincingly mushroom-shaped on
// real hardware, the fix belongs in this file's own tuning (burst size/
// strength/domain proportions), not in grid_smoke_solver2.js itself.
//
// Domain: taller and wider than examples/17 (128x160, vs. 96x128) to give
// the cap room to roll up and spread before either leaving through the
// top outflow or hitting a side wall. Same top-outflow-only boundary
// setup as examples/17, for the same reason (smoke needs somewhere to
// actually go, and an outflow's own Dirichlet-zero pressure ring avoids
// the closed-box pure-Neumann null-space issue examples/14 needed
// isDegenerateDot for).

import * as tsl_array_n from 'tsl_array_n';
import { float, max, sqrt } from 'three/tsl';
import { grid } from 'fluxflow';

const densityCanvas = document.querySelector( '#outDensity' );
const fireCanvas = document.querySelector( '#outFire' );
const statusEl = document.querySelector( '#status' );
const perfEl = document.querySelector( '#perf' );
const detonateButton = document.querySelector( '#detonate' );

function status( text, isErr ) {

	statusEl.textContent = text;
	statusEl.className = isErr ? 'err' : '';

}

const NX = 128;
const NY = 160;
const dt = 1 / 30;

const burstCenterX = NX / 2;
const burstCenterY = 16;
const burstRadius = 14; // large and dense -- a one-shot mushroom needs real mass and heat to roll up, not a wisp
const burstDensity = 1;
const burstTemperature = 10; // well above examples/17's own continuous-source value (4), tuned against buoyancyTemperatureFactor below
const burstVelocityStrength = 9; // outward radial impulse magnitude at the burst's own center, tapering to 0 at burstRadius -- the literal "blast", on top of (and independent from) thermal buoyancy
const buoyancyTemperatureFactor = 6.5; // a bit stronger than grid_smoke_solver2.js's own default (5.0) -- a punchier rise suits this scene better than a slow-drifting one
const diagnosticInterval = 30;
const DRAW_INTERVAL = 2;

const OUTER_MARGIN = 1000; // same padding technique as examples/15-18's own makeWallStripPolygon precedent -- see examples/15-flow-past-cylinder/main.js's own header comment for the real gradient-direction bug this guards against

function makeTopWallStripPolygon( innerY, outerY ) {

	return [ [ - OUTER_MARGIN, innerY ], [ NX + OUTER_MARGIN, innerY ], [ NX + OUTER_MARGIN, outerY ], [ - OUTER_MARGIN, outerY ] ];

}

try {

	const renderer = await tsl_array_n.init( { canvas: document.createElement( 'canvas' ), allowFallback: true } );
	status( `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }` );

	const velocityGrid = grid.createFaceCenteredGrid2( NX, NY, 1, 1, 0, 0 );

	const outflow = grid.createSDFOutflow2( NX, NY, 1, 1, 0, 0 );
	outflow.addPolygon( makeTopWallStripPolygon( NY - 2, NY + OUTER_MARGIN ) );

	const smoke = grid.createGridSmokeSolver2( {
		velocityGrid,
		gridSpacing: [ 1, 1 ],
		origin: [ 0, 0 ],
		outflows: outflow,
		closedDomainBoundaryFlag: grid.DIRECTION_ALL & ~grid.DIRECTION_UP,
		dt,
		buoyancyTemperatureFactor,
		pressure: { multigrid: { numberOfLevels: 4 }, tolerance: 1e-4, maxIterations: 60 }
	} );

	// One-shot burst kernels -- density/temperature only ever written into
	// stateA (the ping-pong slot that's always active immediately after
	// construction; see grid_smoke_solver2.js's own header comment for
	// why the *other* slot needs no separate write -- the first
	// onAdvanceTimeStep() call advects stateA's own contents into it
	// naturally). Velocity is written directly into velocityGrid's own
	// dataU/dataV -- no ping-pong at this level, this *is* the field
	// createGridSolver2 reads/advects every frame.
	const setDensityBurst = tsl_array_n.kernel( smoke.density.stateA.dataSize, ( i, j ) => {

		const pos = smoke.density.stateA.dataPosition( i, j );
		const dx = pos.x.sub( burstCenterX );
		const dy = pos.y.sub( burstCenterY );
		const inBurst = dx.mul( dx ).add( dy.mul( dy ) ).lessThan( burstRadius * burstRadius );

		smoke.density.stateA.data( i, j ).assign( inBurst.select( float( burstDensity ), float( 0 ) ) );

	} );

	const setTemperatureBurst = tsl_array_n.kernel( smoke.temperature.stateA.dataSize, ( i, j ) => {

		const pos = smoke.temperature.stateA.dataPosition( i, j );
		const dx = pos.x.sub( burstCenterX );
		const dy = pos.y.sub( burstCenterY );
		const inBurst = dx.mul( dx ).add( dy.mul( dy ) ).lessThan( burstRadius * burstRadius );

		smoke.temperature.stateA.data( i, j ).assign( inBurst.select( float( burstTemperature ), float( 0 ) ) );

	} );

	// Outward radial push, strongest at the burst's own center and
	// tapering linearly to 0 at burstRadius -- a rough, deliberately
	// simple stand-in for a real blast wave's own falloff shape, not a
	// physically-derived one. `max(dist, 0.001)` avoids a literal 0/0 at
	// the exact center (a real cell will essentially never land exactly
	// there, but a query *position*, unlike a cell index, can).
	const setVelocityBurstU = tsl_array_n.kernel( velocityGrid.dataSizeU, ( i, j ) => {

		const pos = velocityGrid.uPosition( i, j );
		const dx = pos.x.sub( burstCenterX );
		const dy = pos.y.sub( burstCenterY );
		const dist = sqrt( dx.mul( dx ).add( dy.mul( dy ) ) );
		const inBurst = dist.lessThan( burstRadius );
		const falloff = max( float( 1 ).sub( dist.div( burstRadius ) ), float( 0 ) );
		const push = falloff.mul( burstVelocityStrength ).mul( dx.div( max( dist, 0.001 ) ) );

		velocityGrid.dataU( i, j ).addAssign( inBurst.select( push, float( 0 ) ) );

	} );

	const setVelocityBurstV = tsl_array_n.kernel( velocityGrid.dataSizeV, ( i, j ) => {

		const pos = velocityGrid.vPosition( i, j );
		const dx = pos.x.sub( burstCenterX );
		const dy = pos.y.sub( burstCenterY );
		const dist = sqrt( dx.mul( dx ).add( dy.mul( dy ) ) );
		const inBurst = dist.lessThan( burstRadius );
		const falloff = max( float( 1 ).sub( dist.div( burstRadius ) ), float( 0 ) );
		const push = falloff.mul( burstVelocityStrength ).mul( dy.div( max( dist, 0.001 ) ) );

		velocityGrid.dataV( i, j ).addAssign( inBurst.select( push, float( 0 ) ) );

	} );

	let frame = 0;
	let nanDetected = false;

	function detonate() {

		smoke.density.stateA.clear();
		smoke.density.stateB.clear();
		smoke.temperature.stateA.clear();
		smoke.temperature.stateB.clear();
		velocityGrid.dataU.fromArray( new Float32Array( velocityGrid.dataSizeU[ 0 ] * velocityGrid.dataSizeU[ 1 ] ) );
		velocityGrid.dataV.fromArray( new Float32Array( velocityGrid.dataSizeV[ 0 ] * velocityGrid.dataSizeV[ 1 ] ) );

		setDensityBurst();
		setTemperatureBurst();
		setVelocityBurstU();
		setVelocityBurstV();

		frame = 0;
		nanDetected = false;
		status( `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }` );

	}

	detonateButton.addEventListener( 'click', detonate );

	detonate();

	const densityCtx = densityCanvas.getContext( '2d' );
	const densityImage = densityCtx.createImageData( NX, NY );
	const fireCtx = fireCanvas.getContext( '2d' );
	const fireImage = fireCtx.createImageData( NX, NY );

	function clamp01( v ) {

		return Math.min( 1, Math.max( 0, v ) );

	}

	function flippedPixelIndex( i, j ) {

		return ( ( NY - 1 - j ) * NX + i ) * 4;

	}

	function drawDensity( data ) {

		for ( let j = 0; j < NY; j ++ ) {

			for ( let i = 0; i < NX; i ++ ) {

				const v = data[ i + NX * j ];
				const pixel = flippedPixelIndex( i, j );
				const bright = 255 * clamp01( v );

				densityImage.data[ pixel ] = bright;
				densityImage.data[ pixel + 1 ] = bright;
				densityImage.data[ pixel + 2 ] = bright;
				densityImage.data[ pixel + 3 ] = 255;

			}

		}

		densityCtx.putImageData( densityImage, 0, 0 );

	}

	// Same hand-rolled hot-color ramp as examples/17-smoke-fire/main.js --
	// see that file's own comment for why this is deliberately stylized,
	// not a radiometric blackbody calculation.
	const HOT_RAMP = [
		[ 0, 0, 0 ], [ 0.6, 0.05, 0 ], [ 1, 0.35, 0 ], [ 1, 0.8, 0.1 ], [ 1, 1, 1 ]
	];

	function hotRampColor( t ) {

		const clamped = clamp01( t );
		const scaled = clamped * ( HOT_RAMP.length - 1 );
		const idx = Math.min( HOT_RAMP.length - 2, Math.floor( scaled ) );
		const frac = scaled - idx;
		const a = HOT_RAMP[ idx ];
		const b = HOT_RAMP[ idx + 1 ];

		return [
			a[ 0 ] + ( b[ 0 ] - a[ 0 ] ) * frac,
			a[ 1 ] + ( b[ 1 ] - a[ 1 ] ) * frac,
			a[ 2 ] + ( b[ 2 ] - a[ 2 ] ) * frac
		];

	}

	function drawFire( densityData, temperatureData ) {

		for ( let j = 0; j < NY; j ++ ) {

			for ( let i = 0; i < NX; i ++ ) {

				const idx = i + NX * j;
				const density = clamp01( densityData[ idx ] );
				const temperature = temperatureData[ idx ] / burstTemperature;
				const [ hr, hg, hb ] = hotRampColor( temperature );
				const pixel = flippedPixelIndex( i, j );

				const smokeGray = density * 0.5;
				const heat = clamp01( temperature );

				fireImage.data[ pixel ] = 255 * clamp01( smokeGray * ( 1 - heat ) + hr * density );
				fireImage.data[ pixel + 1 ] = 255 * clamp01( smokeGray * ( 1 - heat ) + hg * density );
				fireImage.data[ pixel + 2 ] = 255 * clamp01( smokeGray * ( 1 - heat ) + hb * density );
				fireImage.data[ pixel + 3 ] = 255;

			}

		}

		fireCtx.putImageData( fireImage, 0, 0 );

	}

	// Same non-finite-aware diagnostic pair as every other example.
	function summarize( arr ) {

		let lo = Infinity, hi = -Infinity, nonFiniteCount = 0, sum = 0;

		for ( let i = 0; i < arr.length; i ++ ) {

			const v = arr[ i ];

			if ( ! Number.isFinite( v ) ) {

				nonFiniteCount ++;
				continue;

			}

			if ( v < lo ) lo = v;
			if ( v > hi ) hi = v;
			sum += v;

		}

		return { lo, hi, sum, nonFiniteCount, length: arr.length };

	}

	function fmt( label, s ) {

		if ( s.length === 0 ) return `${ label } EMPTY-READBACK`;
		if ( s.nonFiniteCount > 0 ) return `${ label } ${ s.nonFiniteCount }/${ s.length } NON-FINITE (finite range [${ s.lo.toFixed( 4 ) }, ${ s.hi.toFixed( 4 ) }])`;
		return `${ label } [${ s.lo.toFixed( 4 ) }, ${ s.hi.toFixed( 4 ) }] sum=${ s.sum.toFixed( 4 ) }`;

	}

	async function logDiagnostics( frameNumber, densityData, temperatureData ) {

		const [ uData, vData ] = await Promise.all( [
			velocityGrid.dataU.toArray(),
			velocityGrid.dataV.toArray()
		] );

		console.log(
			`fluxflow explosion [frame ${ frameNumber }] ` +
			`converged=${ smoke.solver.pressureSolver.diagnostics.converged } rejected=${ smoke.solver.pressureSolver.diagnostics.rejected } | ` +
			`${ fmt( 'density', summarize( densityData ) ) } | ` +
			`${ fmt( 'temperature', summarize( temperatureData ) ) } | ` +
			`${ fmt( 'u', summarize( uData ) ) } | ` +
			`${ fmt( 'v', summarize( vData ) ) }`
		);

	}

	function checkForNonFinite( data, frameNumber ) {

		if ( nanDetected ) return;

		for ( let i = 0; i < data.length; i ++ ) {

			if ( ! Number.isFinite( data[ i ] ) ) {

				nanDetected = true;
				status( `non-finite density value detected at frame ${ frameNumber } (index ${ i }, value ${ data[ i ] })`, true );
				console.error( `fluxflow explosion: non-finite density value at frame ${ frameNumber }, index ${ i }:`, data[ i ] );
				return;

			}

		}

	}

	const FPS_WINDOW = 30;
	const frameTimes = [];
	let lastFrameTime = performance.now();

	function updatePerf() {

		const now = performance.now();
		frameTimes.push( now - lastFrameTime );
		lastFrameTime = now;
		if ( frameTimes.length > FPS_WINDOW ) frameTimes.shift();

		const avgMs = frameTimes.reduce( ( a, b ) => a + b, 0 ) / frameTimes.length;
		perfEl.textContent = `fps: ${ ( 1000 / avgMs ).toFixed( 1 ) }`;

	}

	async function animate() {

		updatePerf();

		await smoke.onAdvanceTimeStep();

		if ( ! nanDetected && frame % DRAW_INTERVAL === 0 ) {

			const [ densityData, temperatureData ] = await Promise.all( [
				smoke.density.current.data.toArray(),
				smoke.temperature.current.data.toArray()
			] );

			checkForNonFinite( densityData, frame );

			if ( ! nanDetected && frame % diagnosticInterval === 0 ) await logDiagnostics( frame, densityData, temperatureData );

			if ( ! nanDetected ) {

				drawDensity( densityData );
				drawFire( densityData, temperatureData );

			}

		}

		frame ++;
		requestAnimationFrame( animate );

	}

	requestAnimationFrame( animate );

} catch ( error ) {

	status( error.message, true );
	console.error( error );

}
