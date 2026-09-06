// Combines external_force_solver2.js's interactive force field (time +
// pointer + keyboard, same design as examples/11-interactive-forces/) with
// advection_solver2.js's semi-Lagrangian advection: dye painted at the
// pointer is carried around by a velocity field that reacts to
// time/mouse/keyboard, visualized directly. A transported dye field is a
// far more intuitive "does this look like fluid" check than the raw
// color-coded velocity examples/11-interactive-forces/ shows.
//
// Velocity itself is *not* self-advected here -- like example 11, it's
// recomputed fresh from the force function every frame (clear() then
// applyExternalForces()), not accumulated. Only dye is advected through
// that instantaneous velocity field. Self-advecting velocity too is a
// natural next step once viscosity/pressure exist, but isn't needed to
// exercise external forces and advection together, and would double the
// amount of ping-ponged state for a visual test that doesn't need it.
//
// Dye state lives in stateA/stateB (ping-ponged, each a full ScalarGrid2,
// since advectScalar2's `input` needs dataPosition/gridSpacing/etc.) plus
// two plain scratch fields, rawAdvectedA/rawAdvectedB (only ever need to
// be `.data`-assignable, since nothing but this file's own inject kernel
// ever reads them). This is four fields, not two, for a real reason found
// while building this: a single field written by *two different*
// tsl_array_n.kernel() objects (e.g. one field that both an advect kernel
// and a separate self-touch inject/decay kernel both write to) reliably
// crashes this project's dev sandbox with
// "dualAttributeData.switchBuffers is not a function" once both kernels
// have each dispatched at least once -- three.js's WebGL2-fallback
// transform-feedback emulation (see WebGLAttributeUtils.js's
// DualAttributeData) evidently assumes each storage buffer has exactly
// one owning compute pipeline as its write target, an assumption this
// project's own array_utils.js createExtrapolateToRegion2 also happens to
// violate (stepAtoB/stepBtoA both write the same outputField) but never
// hit in practice, apparently because nothing has driven it through a
// real dispatch loop like this before. Restructured so every field here
// has exactly one permanent writer kernel: advectAtoB only ever writes
// rawAdvectedB, injectAtoB only ever writes stateB (reading rawAdvectedB,
// not touching it in place) -- and the mirror image for the other
// direction. Whether this is fixable on this backend at all, or whether
// it's simply a hard rule to design around here, is undetermined; the
// single-writer-per-field structure below sidesteps it either way and is
// no less correct than the two-field version would have been.

import * as tsl_array_n from 'tsl_array_n';
import { vec2, sin, normalize, float, length, clamp, max } from 'three/tsl';
import { grid, interaction } from 'fluxflow';

const canvas = document.querySelector( '#out' );
const statusEl = document.querySelector( '#status' );

function status( text, isErr ) {

	statusEl.textContent = text;
	statusEl.className = isErr ? 'err' : '';

}

const N = 64;
const dt = 0.5; // a visualization scale factor, not a physical time-step -- see examples/11's own header comment
const injectRadius = 3; // grid cells
const dyeDecay = 0.995; // per-frame multiplicative fade

try {

	const renderer = await tsl_array_n.init( { canvas: document.createElement( 'canvas' ), allowFallback: true } );
	status( `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }` );

	const velocityGrid = grid.createFaceCenteredGrid2( N, N, 1, 1, 0, 0 );

	// persistent dye state, ping-ponged -- see header comment for why this
	// is four fields, not two.
	const stateA = grid.createScalarGrid2( N, N, 1, 1, 0, 0 );
	const stateB = grid.createScalarGrid2( N, N, 1, 1, 0, 0 );
	stateA.clear();
	stateB.clear();

	// scratch: this frame's freshly-advected (not yet injected/decayed)
	// dye, before it becomes next frame's state. Plain array2 fields
	// wrapped in a `{ data }` shape -- all advectScalar2 needs of its
	// `output` argument.
	const rawAdvectedA = { data: tsl_array_n.arrayN( 'float', [ N, N ] ) };
	const rawAdvectedB = { data: tsl_array_n.arrayN( 'float', [ N, N ] ) };

	const timeField = tsl_array_n.array0( 'float' );
	timeField.fromArray( new Float32Array( [ 0 ] ) );

	const pointer = interaction.createPointerUniform( canvas );
	const keyboard = interaction.createKeyboardUniform( [ 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight' ] );

	// identical to examples/11-interactive-forces/'s own force(pos) -- same
	// three ingredients (time-varying wind, pointer attraction while
	// pressed, arrow-key push), reused here to drive the velocity field
	// that dye gets advected through.
	function force( pos ) {

		const wind = vec2( sin( timeField() ), 0 ).mul( 1.5 );

		const pointerSimPos = pointer.position().mul( float( N ) );
		const towardPointer = normalize( pointerSimPos.sub( pos ) ).mul( pointer.isDown() ).mul( 4 );

		const keyPush = vec2(
			keyboard.fields.ArrowRight().sub( keyboard.fields.ArrowLeft() ),
			keyboard.fields.ArrowUp().sub( keyboard.fields.ArrowDown() )
		).mul( 3 );

		return wind.add( towardPointer ).add( keyPush );

	}

	const forceSolver = grid.createExternalForceSolver2( { velocityGrid, force, dt } );

	// one advection solver, bound once to the single velocityGrid -- since
	// that grid's *identity* never changes (only its contents, recomputed
	// fresh each frame above), trace()'s already-built kernels correctly
	// pick up this frame's velocity with no rebuild needed, the same
	// live-field pattern used throughout this port.
	const advectionSolver = grid.createSemiLagrangianAdvectionSolver2( { velocityGrid, dt } );

	// each of these four dispatchers is the one and only permanent writer
	// of its target field -- see header comment.
	const advectAtoB = advectionSolver.advectScalar2( stateA, rawAdvectedB );
	const advectBtoA = advectionSolver.advectScalar2( stateB, rawAdvectedA );

	// reads `rawAdvected` (this frame's fresh advection result), paints
	// dye near the pointer into it (a linear falloff splat, clamped to
	// [0,1]) while pressed, decays it, and writes the result into
	// `state` -- a cross-field copy-with-modification, not self-touch, so
	// `state`'s only writer stays this one kernel, never the advect kernel
	// that targets `rawAdvected` instead.
	function createInjectKernel( rawAdvectedGrid, stateGrid ) {

		return tsl_array_n.kernel( stateGrid.dataSize, ( i, j ) => {

			const pos = stateGrid.dataPosition( i, j );
			const raw = rawAdvectedGrid.data( i, j );

			tsl_array_n.If( pointer.isDown().greaterThan( 0.5 ), () => {

				const pointerSimPos = pointer.position().mul( float( N ) );
				const dist = length( pos.sub( pointerSimPos ) );
				const splat = clamp( float( 1 ).sub( dist.div( injectRadius ) ) );
				stateGrid.data( i, j ).assign( max( raw.mul( dyeDecay ), splat ) );

			} ).Else( () => {

				stateGrid.data( i, j ).assign( raw.mul( dyeDecay ) );

			} );

		} );

	}

	const injectAtoB = createInjectKernel( rawAdvectedB, stateB );
	const injectBtoA = createInjectKernel( rawAdvectedA, stateA );

	const ctx = canvas.getContext( '2d' );
	const image = ctx.createImageData( N, N );

	function draw( data ) {

		for ( let j = 0; j < N; j ++ ) {

			for ( let i = 0; i < N; i ++ ) {

				const v = data[ i + N * j ];

				// canvas Y is down-positive, this grid's Y is up-positive --
				// flip rows so the image matches the simulation's own orientation
				const pixel = ( ( N - 1 - j ) * N + i ) * 4;
				const bright = 255 * Math.min( 1, Math.max( 0, v ) );

				image.data[ pixel ] = bright;
				image.data[ pixel + 1 ] = bright;
				image.data[ pixel + 2 ] = bright;
				image.data[ pixel + 3 ] = 255;

			}

		}

		ctx.putImageData( image, 0, 0 );

	}

	let simTime = 0;
	let frame = 0;

	async function animate() {

		simTime += dt / 15; // wind's own oscillation speed -- unrelated to the force-application dt above
		timeField.fromArray( new Float32Array( [ simTime ] ) );

		velocityGrid.clear();
		forceSolver.applyExternalForces();

		let currentState;

		if ( frame % 2 === 0 ) {

			advectAtoB();
			injectAtoB();
			currentState = stateB;

		} else {

			advectBtoA();
			injectBtoA();
			currentState = stateA;

		}

		const data = await currentState.data.toArray();
		draw( data );

		frame ++;
		requestAnimationFrame( animate );

	}

	requestAnimationFrame( animate );

} catch ( error ) {

	status( error.message, true );
	console.error( error );

}
