// Tracks pointer (mouse/touch/pen) position and button state on a DOM
// element, exposed as tsl_array_n array0 fields so a hand-written force
// function (external_force_solver2.js's `force: (pos) => vec2`) can read
// them directly -- e.g. `pointer.position()` for an attraction/repulsion
// target, `pointer.isDown()` to gate the effect to only while pressed.
//
// No new mechanism here beyond what array0 + .fromArray() already gives
// any GPU-resident scalar (the same live-updatable-uniform pattern
// linalg.js's alpha/beta and advection_solver2.js's dt already use) --
// this is just the DOM event wiring needed to keep such a field in sync
// with the pointer, which is genuinely more than a one-liner (unlike a
// plain time value, which needs no wrapper at all: any
// `tsl_array_n.array0('float')` the caller updates themselves already
// works).
//
// Browser-only (DOM events) -- no vitest coverage, same as
// grid/svg_utils.js; verified live instead, see
// examples/11-interactive-forces/.

import * as tsl_array_n from 'tsl_array_n';

// element: the DOM element to track the pointer over (e.g. a canvas).
// Position is normalized to [0,1] x [0,1] relative to that element's
// bounding box, Y flipped so it's up-positive (matching this project's
// grid/world-space convention, not the DOM's own down-positive screen Y)
// -- map it into simulation coordinates yourself inside your own force
// function (e.g. `pointer.position().mul(gridSize)`), since only the
// caller knows how their canvas maps to their simulation domain.
export function createPointerUniform( element ) {

	const position = tsl_array_n.array0( 'vec2' );
	const isDown = tsl_array_n.array0( 'float' );

	position.fromArray( new Float32Array( [ 0.5, 0.5 ] ) );
	isDown.fromArray( new Float32Array( [ 0 ] ) );

	function updatePosition( event ) {

		const rect = element.getBoundingClientRect();
		const nx = ( event.clientX - rect.left ) / rect.width;
		const ny = ( event.clientY - rect.top ) / rect.height;
		position.fromArray( new Float32Array( [ nx, 1 - ny ] ) );

	}

	function onPointerMove( event ) {

		updatePosition( event );

	}

	function onPointerDown( event ) {

		updatePosition( event );
		isDown.fromArray( new Float32Array( [ 1 ] ) );

	}

	function onPointerUp() {

		isDown.fromArray( new Float32Array( [ 0 ] ) );

	}

	element.addEventListener( 'pointermove', onPointerMove );
	element.addEventListener( 'pointerdown', onPointerDown );
	// listens on window, not just `element`, so releasing outside the
	// element while dragging still clears isDown
	window.addEventListener( 'pointerup', onPointerUp );

	function dispose() {

		element.removeEventListener( 'pointermove', onPointerMove );
		element.removeEventListener( 'pointerdown', onPointerDown );
		window.removeEventListener( 'pointerup', onPointerUp );

	}

	return { position, isDown, dispose };

}
