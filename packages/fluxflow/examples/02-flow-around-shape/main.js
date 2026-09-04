// A test of "what can the existing grid code do visually", not a proper
// example/tutorial.
//
// This is not a real fluid solve (no pressure-projection/advection solver --
// grid_solver2.js is just an empty shell) -- what's happening here is two
// smaller things, both of which the existing grid/ modules can genuinely do:
//   1. Uses createSDFRigidBodyCollider2 (a real fluxflow collider object) to
//      drive a rotating star-shaped collider, with its SDF rendered as a
//      heatmap -- a direct showcase of polygon_sdf.js's polygon SDF
//      rasterization.
//   2. A lightweight, pure-CPU flow-around-obstacle particle effect: uses
//      polygon_sdf.js's exported polygonSignedDistance to numerically
//      estimate a gradient direction, pushing particles away as they get
//      close to the shape -- a classic "potential flow" style visual trick,
//      not an actual Navier-Stokes solve, but it reads as fluid flowing
//      around an obstacle. Deliberately pure CPU: a few hundred particles is
//      too small a scale to benefit from GPU parallelism, and forcing it
//      through a kernel would just add complexity.
//
// The shape's rotated pose is computed independently here (calling the same
// rotatePolygon/translatePolygon/polygonCentroid from polygon_sdf.js that
// the collider uses internally), rather than reading the collider's
// internal state directly -- sdf_collider2.js doesn't expose the posed
// polygon on its own, and canvas rendering here uses per-pixel/per-cell CPU
// values, which don't need (and wouldn't be a good fit for) the "read the
// GPU field back via toArray()" path anyway (the resolutions don't match,
// and per the GPU round-trip check below, that path isn't reliable in this
// sandbox environment to begin with). collider.update(dt) is also genuinely
// called, proving that the real object's own rotational kinematics actually
// run (unlike the structural tests in sdf_collider2.test.js, this is live,
// called continuously across many frames).

import * as tsl_array_n from 'tsl_array_n';
import { grid } from 'fluxflow';

const canvas = document.getElementById( 'canvas' );
const ctx = canvas.getContext( '2d' );
const statusEl = document.getElementById( 'status' );

function setStatus( html ) {

	statusEl.innerHTML = html;

}

// ------------------------------------------------------------
// world coordinates <-> canvas pixel coordinates: the world is
// [-5,5]x[-5,5], the canvas is 480x480, y axis flipped (world y points up,
// canvas y points down)

const WORLD_HALF = 5;
const SCALE = canvas.width / ( WORLD_HALF * 2 );

function worldToCanvas( x, y ) {

	return [ ( x + WORLD_HALF ) * SCALE, ( WORLD_HALF - y ) * SCALE ];

}

// ------------------------------------------------------------
// five-pointed star polygon (local coordinates, centered at the origin)

function makeStarPolygon( outerRadius, innerRadius, points ) {

	const verts = [];

	for ( let i = 0; i < points * 2; i ++ ) {

		const r = i % 2 === 0 ? outerRadius : innerRadius;
		const angle = ( i / ( points * 2 ) ) * Math.PI * 2 - Math.PI / 2;
		verts.push( [ r * Math.cos( angle ), r * Math.sin( angle ) ] );

	}

	return verts;

}

const starLocal = makeStarPolygon( 1.6, 0.65, 5 );
const shapeCenter = [ 0, 0 ];
const angularVelocity = 0.6; // rad/s

try {

	const renderer = await tsl_array_n.init( { allowFallback: true, canvas: document.createElement( 'canvas' ) } );

	// A real fluxflow collider -- 8x8 is just a shape for this GPU-side
	// field, this demo's visual rendering doesn't read it (see the file
	// header comment), but it's used below for a real GPU round-trip check
	const collider = grid.createSDFRigidBodyCollider2(
		starLocal.map( ( [ x, y ] ) => [ x + shapeCenter[ 0 ], y + shapeCenter[ 1 ] ] ),
		8, 8, 1.25, 1.25, -5, -5,
		[ 0, 0 ], angularVelocity
	);

	let gpuStatusHtml = '';

	try {

		// Touch this field: first use a kernel that copies it to itself
		// (making sure the GPU-side buffer has genuinely been used by a
		// compute pass -- storage buffers are lazily created, see
		// tsl_array_n's known-limitations note), then toArray() to read it back
		const touch = tsl_array_n.kernel( collider.grid.dataSize, ( i, j ) => {

			collider.grid.data( i, j ).assign( collider.grid.data( i, j ) );

		} );
		touch();

		const readback = Array.from( await collider.grid.data.toArray() );
		const looksReal = readback.some( ( v ) => v !== 0 );

		gpuStatusHtml = looksReal
			? `<span class="ok">✓ GPU SDF field round-trip read back nonzero values (backend: ${ renderer.backend?.constructor?.name })</span>`
			: `<span class="err">✗ GPU SDF field round-trip read back all zeros -- a known fallback limitation of this sandbox environment (see the README); the visualization below doesn't depend on this result, it computes directly on the CPU</span>`;

	} catch ( error ) {

		gpuStatusHtml = `<span class="err">✗ GPU round-trip check threw: ${ error.message }</span>`;

	}

	setStatus( `<span class="ok">✓ init() + createSDFRigidBodyCollider2() constructed successfully</span>\n${ gpuStatusHtml }\nrunning…` );

	// ------------------------------------------------------------
	// particle system: pure CPU, uses polygon_sdf.js's polygonSignedDistance
	// to numerically estimate a gradient, pushing particles away as they
	// approach the shape -- not an actual fluid-equation solve, a classic
	// "potential flow" style visual approximation

	const PARTICLE_COUNT = 400;
	const FLOW_SPEED = 2.2;
	const INFLUENCE_RADIUS = 1.6;

	const TRAIL_LENGTH = 10;

	function spawnParticle() {

		const x = - WORLD_HALF - Math.random() * 1.5;
		const y = ( Math.random() - 0.5 ) * WORLD_HALF * 2;

		return { x, y, trail: Array.from( { length: TRAIL_LENGTH }, () => [ x, y ] ) };

	}

	const particles = Array.from( { length: PARTICLE_COUNT }, () => {

		const p = spawnParticle();
		p.x = - WORLD_HALF + Math.random() * WORLD_HALF * 2; // fill the whole frame on the very first frame, instead of waiting for particles to stream in from the left
		p.trail = Array.from( { length: TRAIL_LENGTH }, () => [ p.x, p.y ] );
		return p;

	} );

	function sdfGradient( x, y, polygon, eps ) {

		const dx = ( grid.polygonSignedDistance( x + eps, y, polygon ) - grid.polygonSignedDistance( x - eps, y, polygon ) ) / ( 2 * eps );
		const dy = ( grid.polygonSignedDistance( x, y + eps, polygon ) - grid.polygonSignedDistance( x, y - eps, polygon ) ) / ( 2 * eps );
		const len = Math.hypot( dx, dy ) || 1;
		return [ dx / len, dy / len ];

	}

	// ------------------------------------------------------------
	// SDF heatmap: warm color inside, fading to cool color outside based on
	// distance, with a bright highlighted contour near the boundary (d~=0)

	const HEATMAP_CELLS = 80;
	const cellSize = canvas.width / HEATMAP_CELLS;

	function sdfColor( d ) {

		if ( Math.abs( d ) < 0.06 ) return '#fef08a'; // boundary contour line

		if ( d < 0 ) {

			const t = Math.min( 1, - d / 1.5 );
			const r = Math.round( 120 + 100 * t );
			const g = Math.round( 40 + 20 * t );
			return `rgb(${ r},${ g },40)`;

		}

		const t = Math.min( 1, d / 4 );
		const r = Math.round( 10 + 10 * t );
		const g = Math.round( 20 + 60 * ( 1 - t ) );
		const b = Math.round( 40 + 150 * ( 1 - t ) );
		return `rgb(${ r },${ g },${ b })`;

	}

	let currentAngle = 0;
	let lastTime = performance.now();

	function frame( now ) {

		const dt = Math.min( 0.05, ( now - lastTime ) / 1000 );
		lastTime = now;

		currentAngle += angularVelocity * dt;

		// The same math the collider uses internally (rotatePolygon/
		// translatePolygon exported from polygon_sdf.js), computed
		// independently here for per-pixel rendering
		const posedStar = grid.rotatePolygon( starLocal, currentAngle, [ 0, 0 ] )
			.map( ( [ x, y ] ) => [ x + shapeCenter[ 0 ], y + shapeCenter[ 1 ] ] );

		// Also genuinely drives the real collider object, proving its own
		// kinematics actually run (not used for rendering)
		collider.update( dt );

		// ---- draw the heatmap ----
		for ( let cy = 0; cy < HEATMAP_CELLS; cy ++ ) {

			const wy = WORLD_HALF - ( cy + 0.5 ) * ( WORLD_HALF * 2 / HEATMAP_CELLS );

			for ( let cx = 0; cx < HEATMAP_CELLS; cx ++ ) {

				const wx = - WORLD_HALF + ( cx + 0.5 ) * ( WORLD_HALF * 2 / HEATMAP_CELLS );
				const d = grid.polygonSignedDistance( wx, wy, posedStar );

				ctx.fillStyle = sdfColor( d );
				ctx.fillRect( cx * cellSize, cy * cellSize, cellSize + 0.5, cellSize + 0.5 );

			}

		}

		// ---- draw the flow-around particles: each particle trails a short
		// fading trail (TRAIL_LENGTH historical positions), which reads as
		// "flowing" even in a still screenshot better than a single frame's
		// tiny line segment would ----
		ctx.lineWidth = 1.5;

		for ( const p of particles ) {

			const d = grid.polygonSignedDistance( p.x, p.y, posedStar );

			if ( d < -0.05 || p.x > WORLD_HALF + 1.5 ) {

				const fresh = spawnParticle();
				p.x = fresh.x; p.y = fresh.y; p.trail = fresh.trail;
				continue;

			}

			let vx = FLOW_SPEED, vy = 0;

			if ( d < INFLUENCE_RADIUS ) {

				const [ gx, gy ] = sdfGradient( p.x, p.y, posedStar, 0.05 );
				const t = Math.max( 0, 1 - d / INFLUENCE_RADIUS );
				const strength = t * t;

				vx = FLOW_SPEED * ( 1 - strength ) + gx * strength * FLOW_SPEED;
				vy = gy * strength * FLOW_SPEED;

			}

			p.x += vx * dt;
			p.y += vy * dt;

			p.trail.push( [ p.x, p.y ] );
			p.trail.shift();

			for ( let i = 1; i < p.trail.length; i ++ ) {

				const [ x0, y0 ] = worldToCanvas( ...p.trail[ i - 1 ] );
				const [ x1, y1 ] = worldToCanvas( ...p.trail[ i ] );
				const alpha = ( i / p.trail.length ) * 0.6;

				ctx.strokeStyle = `rgba(210,235,255,${ alpha })`;
				ctx.beginPath();
				ctx.moveTo( x0, y0 );
				ctx.lineTo( x1, y1 );
				ctx.stroke();

			}

		}

		requestAnimationFrame( frame );

	}

	requestAnimationFrame( frame );

} catch ( error ) {

	setStatus( `<span class="err">✗ failed — ${ error.message }</span>` );
	console.error( error );

}
