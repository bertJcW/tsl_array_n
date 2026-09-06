// Full solver demo: external forces + pressure projection (MGPCG,
// Dirichlet-aware) + advection, wired together via createGridSolver2 (see
// grid_solver2.js). Click/drag pins a circular region's pressure to a
// target value (a "pressure source"); hold Shift while dragging to flip
// it to a sink instead. The resulting pressure gradient visibly perturbs
// the velocity field once projected, demonstrating both new pieces
// (grid_pressure_solver2.js and grid_solver2.js's orchestration) at once.
//
// Unlike examples/11-interactive-forces/ and
// examples/12-interactive-advection/, velocity here is real physics end
// to end: it genuinely accumulates and self-advects frame to frame
// (grid_solver2.js handles both internally), not recomputed fresh every
// frame.
//
// The pointer is only tracked over the velocity canvas -- both canvases
// are the same resolution and display size, so dragging over the
// pressure canvas instead simply won't register a position. A dual-canvas
// pointer would need interaction/pointer.js itself to change (e.g. accept
// several elements sharing one position); not built here, out of scope
// for this demo.
//
// In this dev sandbox: constructs and runs every frame without throwing,
// but hits yet another WebGL2-fallback-only limitation, distinct from the
// atomics compile-time error examples 04/05/07 hit -- this pipeline's
// shader actually *compiles*, but dispatching it logs
// "GL_INVALID_OPERATION: glDrawArraysInstanced: Not enough space in bound
// transform feedback buffers" (a driver/hardware limit on simultaneously
// bound transform-feedback attributes, most likely exceeded here simply
// because the full pipeline -- CG's own several scratch fields plus this
// solver's own b/mask/target fields plus velocity -- has more storage
// buffers in flight at once than any single earlier example in this port
// needed). No JS exception is thrown either way (confirmed via manual
// frame stepping), so the numbers this produces in-sandbox should not be
// trusted -- needs the user's real WebGPU hardware to confirm, same as
// every other MGPCG-based example here.

import * as tsl_array_n from 'tsl_array_n';
import { vec2, sin, float, length } from 'three/tsl';
import { grid, interaction } from 'fluxflow';

const velocityCanvas = document.querySelector( '#outVelocity' );
const pressureCanvas = document.querySelector( '#outPressure' );
const statusEl = document.querySelector( '#status' );

function status( text, isErr ) {

	statusEl.textContent = text;
	statusEl.className = isErr ? 'err' : '';

}

const N = 48;
const dt = 1 / 30;
const dirichletRadius = 4; // grid cells
const dirichletStrength = 15;
const pressureColorScale = dirichletStrength;
const velocityColorScale = 5;

try {

	const renderer = await tsl_array_n.init( { canvas: document.createElement( 'canvas' ), allowFallback: true } );
	status( `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }` );

	const velocityGrid = grid.createFaceCenteredGrid2( N, N, 1, 1, 0, 0 );
	const cellVelocity = tsl_array_n.arrayN( 'vec2', [ N, N ] );

	const timeField = tsl_array_n.array0( 'float' );
	timeField.fromArray( new Float32Array( [ 0 ] ) );

	const pointer = interaction.createPointerUniform( velocityCanvas );
	const keyboard = interaction.createKeyboardUniform( [ 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Shift' ] );

	// time-varying wind + arrow-key push -- same wind/key idiom as
	// examples 11/12, minus their own pointer-attraction term specifically
	// (the pointer sets pressure here instead, to avoid double-duty
	// confusion in one demo).
	function force( pos ) {

		const wind = vec2( sin( timeField() ), 0 ).mul( 1.5 );

		const keyPush = vec2(
			keyboard.fields.ArrowRight().sub( keyboard.fields.ArrowLeft() ),
			keyboard.fields.ArrowUp().sub( keyboard.fields.ArrowDown() )
		).mul( 3 );

		return wind.add( keyPush );

	}

	// a circular region around the pointer becomes Dirichlet while
	// pressed; Shift flips the target's sign (source vs. sink).
	function dirichlet( pos ) {

		const pointerSimPos = pointer.position().mul( float( N ) );
		const dist = length( pos.sub( pointerSimPos ) );
		const active = pointer.isDown().greaterThan( 0.5 ).and( dist.lessThan( dirichletRadius ) );
		const sign = keyboard.fields.Shift().greaterThan( 0.5 ).select( float( -1 ), float( 1 ) );

		return { active, target: sign.mul( dirichletStrength ) };

	}

	const solver = grid.createGridSolver2( {
		velocityGrid,
		gridSpacing: [ 1, 1 ],
		origin: [ 0, 0 ],
		force,
		dirichlet,
		dt,
		pressure: { multigrid: { numberOfLevels: 4 }, tolerance: 1e-4, maxIterations: 40 }
	} );

	const computeCellVelocity = tsl_array_n.kernel( [ N, N ], ( i, j ) => {

		cellVelocity( i, j ).assign( grid.faceCenteredValueAtCellCenter2( velocityGrid.dataU, velocityGrid.dataV, i, j ) );

	} );

	const velocityCtx = velocityCanvas.getContext( '2d' );
	const velocityImage = velocityCtx.createImageData( N, N );
	const pressureCtx = pressureCanvas.getContext( '2d' );
	const pressureImage = pressureCtx.createImageData( N, N );

	function clamp01( v ) {

		return Math.min( 1, Math.max( 0, v ) );

	}

	function drawVelocity( data ) {

		for ( let j = 0; j < N; j ++ ) {

			for ( let i = 0; i < N; i ++ ) {

				const src = ( i + N * j ) * 2;
				const vx = data[ src ];
				const vy = data[ src + 1 ];

				// canvas Y is down-positive, this grid's Y is up-positive --
				// flip rows so the image matches the simulation's own orientation
				const pixel = ( ( N - 1 - j ) * N + i ) * 4;

				velocityImage.data[ pixel ] = 255 * clamp01( 0.5 + 0.5 * ( vx / velocityColorScale ) );
				velocityImage.data[ pixel + 1 ] = 255 * clamp01( 0.5 + 0.5 * ( vy / velocityColorScale ) );
				velocityImage.data[ pixel + 2 ] = 128;
				velocityImage.data[ pixel + 3 ] = 255;

			}

		}

		velocityCtx.putImageData( velocityImage, 0, 0 );

	}

	function drawPressure( data ) {

		for ( let j = 0; j < N; j ++ ) {

			for ( let i = 0; i < N; i ++ ) {

				const p = data[ i + N * j ];
				const pixel = ( ( N - 1 - j ) * N + i ) * 4;
				const bright = 255 * clamp01( 0.5 + 0.5 * ( p / pressureColorScale ) );

				pressureImage.data[ pixel ] = bright;
				pressureImage.data[ pixel + 1 ] = bright;
				pressureImage.data[ pixel + 2 ] = bright;
				pressureImage.data[ pixel + 3 ] = 255;

			}

		}

		pressureCtx.putImageData( pressureImage, 0, 0 );

	}

	let simTime = 0;

	async function animate() {

		simTime += dt;
		timeField.fromArray( new Float32Array( [ simTime ] ) );

		await solver.onAdvanceTimeStep( dt );
		computeCellVelocity();

		const velocityData = await cellVelocity.toArray();
		const pressureData = await solver.pressure.data.toArray();

		drawVelocity( velocityData );
		drawPressure( pressureData );

		requestAnimationFrame( animate );

	}

	requestAnimationFrame( animate );

} catch ( error ) {

	status( error.message, true );
	console.error( error );

}
