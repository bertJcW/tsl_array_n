// Tracks which of a caller-chosen set of keys are currently held down,
// each exposed as its own tsl_array_n array0('float') (1 while held, 0
// otherwise) so a hand-written force function
// (external_force_solver2.js's `force: (pos) => vec2`) can read them
// directly and combine them into whatever it wants -- a directional
// force, a toggle, anything. Deliberately per-key rather than a
// hardcoded WASD-style directional vec2, so any key (not just movement
// keys) can drive a force.
//
// Browser-only (DOM events) -- no vitest coverage, same as
// grid/svg_utils.js; verified live instead, see
// examples/11-interactive-forces/.

import * as tsl_array_n from 'tsl_array_n';

// keys: array of key names to track, matching KeyboardEvent.key values
// (e.g. ['w','a','s','d'] or ['ArrowUp','ArrowDown']) -- single-character
// keys are matched case-insensitively (so 'w' also catches Shift+W),
// multi-character key names (like 'ArrowUp') are matched exactly.
// Returns { fields: { [key]: array0('float') }, dispose() }.
export function createKeyboardUniform( keys ) {

	const fields = {};
	const held = {};

	for ( const key of keys ) {

		fields[ key ] = tsl_array_n.array0( 'float' );
		fields[ key ].fromArray( new Float32Array( [ 0 ] ) );
		held[ key ] = false;

	}

	function normalizeKey( event ) {

		return event.key.length === 1 ? event.key.toLowerCase() : event.key;

	}

	function onKeyDown( event ) {

		const key = normalizeKey( event );

		if ( key in fields && ! held[ key ] ) {

			held[ key ] = true;
			fields[ key ].fromArray( new Float32Array( [ 1 ] ) );

		}

	}

	function onKeyUp( event ) {

		const key = normalizeKey( event );

		if ( key in fields ) {

			held[ key ] = false;
			fields[ key ].fromArray( new Float32Array( [ 0 ] ) );

		}

	}

	window.addEventListener( 'keydown', onKeyDown );
	window.addEventListener( 'keyup', onKeyUp );

	function dispose() {

		window.removeEventListener( 'keydown', onKeyDown );
		window.removeEventListener( 'keyup', onKeyUp );

	}

	return { fields, dispose };

}
