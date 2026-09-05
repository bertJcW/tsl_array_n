// Demonstrates external_force_solver2.js's `force: (pos) => vec2` reacting
// to time, pointer, and keyboard -- all three are just tsl_array_n
// array0 fields the caller updates from JS each frame (time: a plain
// array0('float') this file manages itself directly, no wrapper needed;
// pointer/keyboard: interaction/pointer.js and interaction/keyboard.js,
// which exist purely to wire up the DOM event listeners, a genuinely
// non-trivial bit of bookkeeping unlike time's plain one-liner) --
// referenced directly inside the force function's own closure, no special
// support needed from external_force_solver2.js itself for any of this.
//
// Visualization only -- no advection/pressure/viscosity yet (not built),
// so velocity is reset to zero and this frame's force is reapplied fresh
// every frame (`velocityGrid.clear()` then `applyExternalForces()`),
// rather than accumulating -- this shows the *current* force field
// directly, not a physically-evolving simulation.
//
// Color mapping: red/blue = x-velocity (blue negative, red positive),
// green/blue = y-velocity, gray = ~zero. Cell-center velocities come from
// grid_math.js's existing faceCenteredValueAtCellCenter2 (already used
// elsewhere in this port), no new physics code needed for this readout.

import * as tsl_array_n from 'tsl_array_n';
import { vec2, sin, normalize, float } from 'three/tsl';
import { grid, interaction } from 'fluxflow';

const canvas = document.querySelector( '#out' );
const statusEl = document.querySelector( '#status' );

function status( text, isErr ) {

	statusEl.textContent = text;
	statusEl.className = isErr ? 'err' : '';

}

const N = 32;
const dt = 1; // a visualization scale factor here, not a physical time-step -- see header comment
const colorScale = 5;

try {

	// offscreen canvas for the renderer, same as tsl_array_n's own
	// examples/04-julia -- this file draws its own output via 2D canvas on
	// the *visible* #out element instead.
	const renderer = await tsl_array_n.init( { canvas: document.createElement( 'canvas' ), allowFallback: true } );
	status( `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }` );

	const velocityGrid = grid.createFaceCenteredGrid2( N, N, 1, 1, 0, 0 );
	const cellVelocity = tsl_array_n.arrayN( 'vec2', [ N, N ] );

	const timeField = tsl_array_n.array0( 'float' ); // just a plain live scalar -- no wrapper needed, unlike pointer/keyboard
	timeField.fromArray( new Float32Array( [ 0 ] ) );

	const pointer = interaction.createPointerUniform( canvas );
	const keyboard = interaction.createKeyboardUniform( [ 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight' ] );

	function force( pos ) {

		// 1. time-varying wind, oscillating left-right
		const wind = vec2( sin( timeField() ), 0 ).mul( 1.5 );

		// 2. attraction toward the pointer, only while pressed -- pointer's
		// normalized [0,1] canvas position mapped into this grid's own
		// simulation coordinates ([0,N] x [0,N], since gridSpacing=1, origin=0)
		const pointerSimPos = pointer.position().mul( float( N ) );
		const towardPointer = normalize( pointerSimPos.sub( pos ) ).mul( pointer.isDown() ).mul( 4 );

		// 3. arrow-key directional push
		const keyPush = vec2(
			keyboard.fields.ArrowRight().sub( keyboard.fields.ArrowLeft() ),
			keyboard.fields.ArrowUp().sub( keyboard.fields.ArrowDown() )
		).mul( 3 );

		return wind.add( towardPointer ).add( keyPush );

	}

	const solver = grid.createExternalForceSolver2( { velocityGrid, force, dt } );

	const computeCellVelocity = tsl_array_n.kernel( [ N, N ], ( i, j ) => {

		cellVelocity( i, j ).assign( grid.faceCenteredValueAtCellCenter2( velocityGrid.dataU, velocityGrid.dataV, i, j ) );

	} );

	const ctx = canvas.getContext( '2d' );
	const image = ctx.createImageData( N, N );

	function draw( data ) {

		for ( let j = 0; j < N; j ++ ) {

			for ( let i = 0; i < N; i ++ ) {

				const src = ( i + N * j ) * 2;
				const vx = data[ src ];
				const vy = data[ src + 1 ];

				// canvas Y is down-positive, this grid's Y is up-positive --
				// flip rows so the image matches the simulation's own orientation
				const pixel = ( ( N - 1 - j ) * N + i ) * 4;

				image.data[ pixel ] = 255 * Math.min( 1, Math.max( 0, 0.5 + 0.5 * ( vx / colorScale ) ) );
				image.data[ pixel + 1 ] = 255 * Math.min( 1, Math.max( 0, 0.5 + 0.5 * ( vy / colorScale ) ) );
				image.data[ pixel + 2 ] = 128;
				image.data[ pixel + 3 ] = 255;

			}

		}

		ctx.putImageData( image, 0, 0 );

	}

	let simTime = 0;

	async function animate() {

		simTime += dt / 30; // wind's own oscillation speed -- unrelated to the force-application dt above
		timeField.fromArray( new Float32Array( [ simTime ] ) );

		velocityGrid.clear();
		solver.applyExternalForces();
		computeCellVelocity();

		const data = await cellVelocity.toArray();
		draw( data );

		requestAnimationFrame( animate );

	}

	requestAnimationFrame( animate );

} catch ( error ) {

	status( error.message, true );
	console.error( error );

}
