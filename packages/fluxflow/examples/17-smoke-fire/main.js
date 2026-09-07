// A reusable smoke/fire solver, per the user's own explicit request for
// one users can reuse (not another hand-rolled-per-example scene) --
// grid.createGridSmokeSolver2 (packages/fluxflow/src/grid/
// grid_smoke_solver2.js, new this round). See that file's own header
// comment for the full design: buoyancy + density/temperature advection +
// decay ported from jet/fluid-engine-dev's own GridSmokeSolver2, "fire" is
// a deliberate parameterization/rendering choice (a hot source + a
// temperature-driven color ramp here), not a separate combustion model --
// there is no fire/combustion reference anywhere in this port's own jet
// source.
//
// Domain: a tall (96x128), otherwise closed box with an outflow strip
// along the *top* wall only -- smoke needs somewhere to actually leave, or
// it just piles up against a closed ceiling forever (and a fully closed
// domain also risks the pure-Neumann null-space issue examples/14 needed
// isDegenerateDot for -- an outflow's own Dirichlet-zero pressure ring
// avoids that the same way examples/15-16 already do). No inflow and no
// collider here -- unlike examples/15-16, this scene's own motion is
// entirely buoyancy-driven (a hot, dense-ish source rising through still
// air), not an imposed cross-flow.
//
// Source: injected every frame in a small disc near the bottom-center,
// exactly mirroring every existing example's own dye-injection idiom
// (createInjectKernel(rawAdvectedGrid, stateGrid), applied here to BOTH
// density and temperature, chosen per frame via smoke.density.current ===
// smoke.density.stateA -- grid_smoke_solver2.js's own header comment
// explains why source injection isn't built into the solver itself).
//
// The "disable temperature buoyancy" checkbox exists specifically to
// verify buoyancy is doing real work, not just watched by eye: with it
// checked, buoyancyTemperatureFactor is live-set to 0 (a plain number
// baked into the solver's own kernel at construction time would NOT be
// toggleable afterward -- this is why it's constructed as a live
// array0('float') node here instead, the same "number or node" convention
// dt/etc. already use throughout this port), so the plume should still
// spread from diffusion-free semi-Lagrangian advection/numerical
// dissipation but visibly stop *rising*.

import * as tsl_array_n from 'tsl_array_n';
import { float, max } from 'three/tsl';
import { grid } from 'fluxflow';

const densityCanvas = document.querySelector( '#outDensity' );
const fireCanvas = document.querySelector( '#outFire' );
const statusEl = document.querySelector( '#status' );
const perfEl = document.querySelector( '#perf' );
const buoyancyOffCheckbox = document.querySelector( '#buoyancyOff' );

function status( text, isErr ) {

	statusEl.textContent = text;
	statusEl.className = isErr ? 'err' : '';

}

const NX = 96;
const NY = 128;
const dt = 1 / 30;
const sourceX = NX / 2;
const sourceY = 6; // near the bottom
const sourceRadius = 6;
const sourceDensity = 1;
const sourceTemperature = 4; // well above ambient (0) -- see buoyancyTemperatureFactor's own default (5.0), tuned against this
const diagnosticInterval = 30;
const DRAW_INTERVAL = 2;

// Same padding technique as examples/15-16's own makeWallStripPolygon --
// see examples/15-flow-past-cylinder/main.js's own header comment for the
// real, confirmed-on-real-hardware gradient-direction bug this guards
// against (an outflow polygon padded only a cell or two past the domain
// edge can make two adjacent faces sample each other as their own
// "upstream" reference instead of both pointing back toward the fluid).
const OUTER_MARGIN = 1000;

function makeTopWallStripPolygon( innerY, outerY ) {

	return [ [ - OUTER_MARGIN, innerY ], [ NX + OUTER_MARGIN, innerY ], [ NX + OUTER_MARGIN, outerY ], [ - OUTER_MARGIN, outerY ] ];

}

try {

	const renderer = await tsl_array_n.init( { canvas: document.createElement( 'canvas' ), allowFallback: true } );
	status( `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }` );

	const velocityGrid = grid.createFaceCenteredGrid2( NX, NY, 1, 1, 0, 0 );

	const outflow = grid.createSDFOutflow2( NX, NY, 1, 1, 0, 0 );
	outflow.addPolygon( makeTopWallStripPolygon( NY - 2, NY + OUTER_MARGIN ) ); // fluid-facing edge at y=NY-2, padding falls away upward

	// Live node, not a plain number -- see this file's own header comment
	// on the "disable buoyancy" checkbox for why.
	const buoyancyTemperatureFactorField = tsl_array_n.array0( 'float' );
	const DEFAULT_BUOYANCY_TEMPERATURE_FACTOR = 5.0; // grid_smoke_solver2.js's own default, restated here since a live node needs its own initial value written explicitly
	buoyancyTemperatureFactorField.fromArray( new Float32Array( [ DEFAULT_BUOYANCY_TEMPERATURE_FACTOR ] ) );

	buoyancyOffCheckbox.addEventListener( 'change', () => {

		const value = buoyancyOffCheckbox.checked ? 0 : DEFAULT_BUOYANCY_TEMPERATURE_FACTOR;
		buoyancyTemperatureFactorField.fromArray( new Float32Array( [ value ] ) );

	} );

	const smoke = grid.createGridSmokeSolver2( {
		velocityGrid,
		gridSpacing: [ 1, 1 ],
		origin: [ 0, 0 ],
		outflows: outflow,
		closedDomainBoundaryFlag: grid.DIRECTION_ALL & ~grid.DIRECTION_UP,
		dt,
		buoyancyTemperatureFactor: buoyancyTemperatureFactorField(),
		// numberOfLevels: real multigrid coarsening, matching every other
		// MGPCG-based example in this port -- numberOfLevels:1 was
		// confirmed inadequate at comparable grid sizes elsewhere
		// (examples/15's own header comment).
		pressure: { multigrid: { numberOfLevels: 4 }, tolerance: 1e-4, maxIterations: 60 }
	} );

	// Explicit clear -- grid_smoke_solver2.js's own header comment on why
	// this isn't done internally (needs tsl_array_n.init(), which has
	// already run by this point in every example, but not necessarily
	// during a structural vitest construction).
	smoke.density.stateA.clear();
	smoke.density.stateB.clear();
	smoke.temperature.stateA.clear();
	smoke.temperature.stateB.clear();

	// Source injection, mirroring every existing example's own
	// createInjectKernel(rawAdvectedGrid, stateGrid) dye pattern exactly --
	// built once per (state field, target value) pair, dispatched
	// whichever is "about to become current" each frame. Density decays
	// toward 0 already (grid_smoke_solver2.js's own decay step, run just
	// before this); this only needs to raise it back up within the source
	// disc, same max(decayed, sourceValue) idiom as dye.
	function createSourceKernel( stateGrid, sourceValue ) {

		return tsl_array_n.kernel( stateGrid.dataSize, ( i, j ) => {

			const pos = stateGrid.dataPosition( i, j );
			const dx = pos.x.sub( sourceX );
			const dy = pos.y.sub( sourceY );
			const inSource = dx.mul( dx ).add( dy.mul( dy ) ).lessThan( sourceRadius * sourceRadius );

			stateGrid.data( i, j ).assign( max( stateGrid.data( i, j ), inSource.select( float( sourceValue ), float( 0 ) ) ) );

		} );

	}

	const injectDensityA = createSourceKernel( smoke.density.stateA, sourceDensity );
	const injectDensityB = createSourceKernel( smoke.density.stateB, sourceDensity );
	const injectTemperatureA = createSourceKernel( smoke.temperature.stateA, sourceTemperature );
	const injectTemperatureB = createSourceKernel( smoke.temperature.stateB, sourceTemperature );

	const densityCtx = densityCanvas.getContext( '2d' );
	const densityImage = densityCtx.createImageData( NX, NY );
	const fireCtx = fireCanvas.getContext( '2d' );
	const fireImage = fireCtx.createImageData( NX, NY );

	function clamp01( v ) {

		return Math.min( 1, Math.max( 0, v ) );

	}

	// canvas Y is down-positive, this grid's Y is up-positive -- flip rows,
	// same convention every other example's own drawing function already uses.
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

	// A hand-rolled, deliberately-not-radiometrically-accurate "hot ramp"
	// (black -> red -> orange -> yellow -> white as t rises from 0 to 1) --
	// the standard stylized fire-color gradient used across countless
	// real-time demos, not a blackbody-spectrum calculation. Blended with
	// density (dim smoke gray where cool, bright ramp color where hot) so
	// the source reads as a glowing core with smoke drifting/cooling above
	// it, the same single-scene "smoke below, fire above the source" look
	// this file's own header comment describes.
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
				const temperature = temperatureData[ idx ] / sourceTemperature; // normalize against the source's own peak
				const [ hr, hg, hb ] = hotRampColor( temperature );
				const pixel = flippedPixelIndex( i, j );

				// Smoke gray where cool, the hot ramp's own color where
				// warm -- blended by temperature itself, not just added,
				// so a cool, dense region still reads as gray smoke
				// rather than a dim red tint.
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

	// Same non-finite-aware diagnostic pair as every other example -- see
	// examples/14-stable-fluids/main.js's own header comment for why a
	// naive min/max scan silently hides NaN instead of reporting it.
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
			`fluxflow smoke-fire [frame ${ frameNumber }] ` +
			`converged=${ smoke.solver.pressureSolver.diagnostics.converged } rejected=${ smoke.solver.pressureSolver.diagnostics.rejected } | ` +
			`${ fmt( 'density', summarize( densityData ) ) } | ` +
			`${ fmt( 'temperature', summarize( temperatureData ) ) } | ` +
			`${ fmt( 'u', summarize( uData ) ) } | ` +
			`${ fmt( 'v', summarize( vData ) ) }`
		);

	}

	let frame = 0;
	let nanDetected = false;

	function checkForNonFinite( data, frameNumber ) {

		if ( nanDetected ) return;

		for ( let i = 0; i < data.length; i ++ ) {

			if ( ! Number.isFinite( data[ i ] ) ) {

				nanDetected = true;
				status( `non-finite density value detected at frame ${ frameNumber } (index ${ i }, value ${ data[ i ] })`, true );
				console.error( `fluxflow smoke-fire: non-finite density value at frame ${ frameNumber }, index ${ i }:`, data[ i ] );
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

		const usingA = smoke.density.current === smoke.density.stateA;

		if ( usingA ) { injectDensityA(); injectTemperatureA(); }
		else { injectDensityB(); injectTemperatureB(); }

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
		if ( ! nanDetected ) requestAnimationFrame( animate );

	}

	requestAnimationFrame( animate );

} catch ( error ) {

	status( error.message, true );
	console.error( error );

}
