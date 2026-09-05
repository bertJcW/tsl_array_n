// Verifies createSemiLagrangianAdvectionSolver2 (grid/advection_solver2.js)
// two ways:
//
// 1. Constant velocity, no collider: dt is chosen so the exact shift is
//    one whole grid cell (v=(1,0), dt=1, gridSpacing=1) -- the traced-back
//    position then lands exactly on a source grid point (f=0 in the
//    cubic sampler), so the expected result is exact, no interpolation
//    error to reason about: output(i,j) should equal the source ramp
//    field's value at (i-1,j), clamped to the domain edge at i=0.
// 2. A collider: a "wall" occupying x in [0,3] on the same 8x8 domain,
//    velocity flowing *into* the wall fast enough that the naive
//    (unclamped) back-trace target would land deep inside it. The wall's
//    own source density is set to a deliberately large sentinel value
//    (100) that should never show up in the fluid region if backTrace's
//    boundary clamping is doing its job -- this is an indirect check
//    (backTrace itself is module-private), but it only needs the public
//    advectScalar2 API and still meaningfully exercises the clamping
//    branch specifically.
//
// CONFIRMED on real WebGPU hardware: test 1 (constant velocity, no
// collider) passed outright -- backTrace's core RK2 midpoint integration
// and velocity sampling are correct. Test 2 (the wall collider) initially
// failed *for real* (not a sandbox artifact): it read back the wall's own
// sentinel value (100), meaning the traced point had leaked all the way
// through the wall rather than being clamped at its surface.
//
// Root-caused with a plain-JS trace of backTrace's exact algorithm for
// this scenario (v=(3,0), dt=1, wall at x in [0,3], query at x=5): the
// substep landing at x=3 has phi0=1 (still outside) but the *next*
// substep's own phi0 is then exactly 0 (already sitting right on the
// boundary) with phi1=-1 (its endpoint, one cell into the wall). jet's
// own trigger condition, `phi0*phi1 < 0`, evaluates to `0 < 0` here --
// false -- so this crossing goes completely undetected and tracing
// marches straight through, landing at x=2 (inside the wall, hence
// reading its sentinel). This is a genuine edge case in the ported
// algorithm (not unique to this port -- jet's own C++ has the identical
// condition), just far more likely to actually trigger with grid-aligned
// colliders and round velocity/dt values than with generic ones. Fixed by
// triggering on `phi1 <= 0` instead (the substep's *endpoint* being at or
// inside the solid, regardless of exactly where phi0 sits) -- re-verified
// with the same plain-JS trace: it now clamps one substep earlier, to
// exactly x=3 (the wall's own surface), the correct behavior.
//
// CONFIRMED on real WebGPU hardware, after the fix: both tests pass.
// Test 2's output at x=5 came back as *exactly* 1.0000 (the fluid value),
// not merely "not 100" -- landing precisely on the wall's own grid point
// means the cubic sampler's f=0 case reconstructs that point's stored
// value exactly, regardless of the wall value still sitting in the
// stencil's other taps.

import * as tsl_array_n from 'tsl_array_n';
import { grid } from 'fluxflow';

const pre = document.querySelector( '#status pre' );
const lines = [];

function log( label, ok, detail ) {

	const cls = ok ? 'ok' : 'err';
	const mark = ok ? '✓' : '✗';
	lines.push( `<span class="${ cls }">${ mark } ${ label }${ detail ? ' — ' + detail : '' }</span>` );
	pre.innerHTML = lines.join( '\n' );

}

const N = 8;

try {

	const renderer = await tsl_array_n.init( { allowFallback: true } );
	log( 'init()', true, `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }` );

	// ---- Test 1: constant velocity, no collider, exact 1-cell shift ----

	const velocityGrid = grid.createFaceCenteredGrid2( N, N, 1, 1, 0, 0 );
	velocityGrid.dataU.fromArray( new Float32Array( velocityGrid.dataSizeU[ 0 ] * velocityGrid.dataSizeU[ 1 ] ).fill( 1 ) );
	velocityGrid.dataV.fromArray( new Float32Array( velocityGrid.dataSizeV[ 0 ] * velocityGrid.dataSizeV[ 1 ] ) );

	const density = grid.createScalarGrid2( N, N, 1, 1, 0, 0 );
	const densityArray = new Float32Array( N * N );
	for ( let j = 0; j < N; j ++ ) for ( let i = 0; i < N; i ++ ) densityArray[ i + N * j ] = i;
	density.data.fromArray( densityArray );

	const output = grid.createScalarGrid2( N, N, 1, 1, 0, 0 );

	const solver = grid.createSemiLagrangianAdvectionSolver2( { velocityGrid, dt: 1.0 } );
	solver.advectScalar2( density, output )();

	const result = Array.from( await output.data.toArray() );
	const expected = [];
	for ( let j = 0; j < N; j ++ ) for ( let i = 0; i < N; i ++ ) expected.push( Math.max( i - 1, 0 ) );

	const matches = result.length === expected.length && result.every( ( v, i ) => Math.abs( v - expected[ i ] ) < 1e-4 );

	log(
		'constant velocity (1,0), dt=1 -- exact 1-cell shift',
		matches,
		matches ? 'all cells matched (max |diff| < 1e-4)' : `row 0: got [${ result.slice( 0, N ).map( v => v.toFixed( 2 ) ) }], expected [${ expected.slice( 0, N ) }]`
	);

	// ---- Test 2: a wall collider, velocity flowing into it ----

	const velocityGrid2 = grid.createFaceCenteredGrid2( N, N, 1, 1, 0, 0 );
	velocityGrid2.dataU.fromArray( new Float32Array( velocityGrid2.dataSizeU[ 0 ] * velocityGrid2.dataSizeU[ 1 ] ).fill( 3 ) );
	velocityGrid2.dataV.fromArray( new Float32Array( velocityGrid2.dataSizeV[ 0 ] * velocityGrid2.dataSizeV[ 1 ] ) );

	const collider = grid.createSDFStaticCollider2( N, N, 1, 1, 0, 0 );
	collider.addPolygons( [ [ [ 0, 0 ], [ 3, 0 ], [ 3, N ], [ 0, N ] ] ] ); // wall occupying x in [0,3]

	const density2 = grid.createScalarGrid2( N, N, 1, 1, 0, 0 );
	const densityArray2 = new Float32Array( N * N );
	for ( let j = 0; j < N; j ++ ) {

		for ( let i = 0; i < N; i ++ ) {

			densityArray2[ i + N * j ] = i < 3 ? 100 : 1; // sentinel inside the wall, plain 1 in the fluid

		}

	}

	density2.data.fromArray( densityArray2 );

	const output2 = grid.createScalarGrid2( N, N, 1, 1, 0, 0 );

	const solver2 = grid.createSemiLagrangianAdvectionSolver2( { velocityGrid: velocityGrid2, collider, dt: 1.0 } );
	solver2.advectScalar2( density2, output2 )();

	const result2 = Array.from( await output2.data.toArray() );
	const queryIndex = 5; // x=5, fluid; naive (unclamped) backtrace target x=5-3*1=2, inside the wall
	const queryValue = result2[ queryIndex ]; // row j=0

	log(
		'wall collider -- back-trace clamped, did not pick up the wall\'s sentinel value',
		queryValue < 50,
		`output at x=5 = ${ queryValue?.toFixed?.( 4 ) } (want < 50, i.e. not near the wall's sentinel 100)`
	);

} catch ( error ) {

	log( 'failed', false, error.message );
	console.error( error );

}
