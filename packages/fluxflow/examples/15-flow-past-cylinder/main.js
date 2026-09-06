// The classic "flow past a cylinder" CFD benchmark, per the user's own
// explicit spec: inflow at the left wall, outflow at the right wall, a
// uniform force across the *whole* domain pushing right (not a small
// region, unlike examples/14-stable-fluids/'s own force), and a circular
// collider centered in the domain with diameter = 1/3 of the domain's
// side length. This is the first example in this port to exercise a
// *real*, non-null collider through createGridSolver2 end to end on real
// hardware (it's been structurally wired since the pressure/orchestration
// work, but never tested this way before), and the first to use the new
// inflow/outflow mechanism at all (grid_blocked_boundary_condition_
// solver2.js's applyInflow, grid_outflow_solver2.js) -- see
// sdf_inflow_outflow2.js's and grid_outflow_solver2.js's own header
// comments for the full mantaflow-derived design.
//
// *** A genuinely subtle ordering interaction between outflow and
// closedDomainBoundaryFlag, worth understanding before changing either ***
//
// grid_solver2.js's defaultComputeAdvection calls, in order: advect ->
// outflowSolver.applyOutflowVelocityBC() -> boundarySolver.constrainVelocity()
// (which runs projectClosedDomainBoundary() then applyInflow()). This means
// if the right wall were still included in closedDomainBoundaryFlag,
// projectClosedDomainBoundary() would zero it right back out *after*
// applyOutflowVelocityBC() just computed a careful convective extrapolation
// there -- silently undoing outflow's own velocity treatment every single
// frame. Excluding DIRECTION_RIGHT below is therefore *functionally
// required*, not just documentation. The left wall (inflow) doesn't have
// this problem -- applyInflow() runs *after* projectClosedDomainBoundary()
// in the same constrainVelocity() call, so it always wins regardless of
// whether DIRECTION_LEFT is excluded; left is excluded below anyway, purely
// for clarity.
//
// Two easy-to-miss wiring points, confirmed by reading grid_solver2.js
// directly: (1) the top-level `collider` option is *not* auto-forwarded to
// advection -- `advection: { collider }` below is required, or the
// self-advected velocity field would sample straight through the cylinder;
// (2) dye's own advection here is a *separate* solver from grid_solver2.js's
// internal one (dye isn't part of that orchestrator's scope at all, same
// as examples/14-stable-fluids/), so *that* solver also needs `collider`
// passed to its own construction, or dye would visibly flow through the
// cylinder while velocity correctly flows around it.
//
// Dye uses the same four-field (state + rawAdvected) x2 ping-pong pattern
// as examples/12-14 (a single field written by two different
// tsl_array_n.kernel() objects crashes this dev sandbox's WebGL2 fallback
// backend -- see examples/12-interactive-advection/'s own header comment).
// solver.outflowSolver.clearOutflowScalarField(...) is built once per dye
// field (matching this port's established "build once, dispatch
// repeatedly" convention) and called after each frame's own inject step,
// so dye that reaches the outflow strip actually vanishes there instead of
// just piling up against the right wall.
//
// Expectations, set accurately: no viscosity model exists in this port yet
// (computeViscosity stays a no-op, per grid_solver2.js's own header
// comment) -- any wake/shedding behind the cylinder is governed by the
// semi-Lagrangian scheme's own numerical dissipation (grid resolution +
// cubic-interpolation truncation error), not a controlled Reynolds number.
// "Does this look like flow past a cylinder" is the right bar to check
// against, not "does it reproduce a textbook Karman vortex street at a
// specific Re."
//
// A second, concrete, checkable prediction of this whole design: unlike
// examples/14-stable-fluids/'s deliberately fully-closed (pure-Neumann,
// singular) pressure domain -- whose CG solve needed this session's own
// isDegenerateDot fix to stay finite at all -- this scenario's outflow
// Dirichlet-zero ring gives the Laplacian a genuine, non-removable anchor
// (the constant field is no longer in its null space, see
// sdf_inflow_outflow2.js's own header comment). Confirmed on real WebGPU
// hardware: pressure visibly trends toward 0 near the right wall on the
// second canvas below, and a clean pinwheel-shaped high/low pressure
// pattern forms around the cylinder, matching classic bluff-body flow.
//
// In this dev sandbox: hits the same atomics compile-time wall as every
// other MGPCG-based example in this port (WGSL atomics have no GLSL
// translation on the WebGL2 fallback backend). CONFIRMED correct and
// stable on real WebGPU hardware (1500+ frames, all-finite velocity/
// pressure/divergence, reaching a steady fixed point) with inflow, all
// three parts of outflow (pressure, convective velocity extrapolation,
// and scalar cleanup), the collider, and the whole-domain force all
// active together, at their default settings -- see
// grid_math.js's own bilinearGradientAtPosition2 and
// grid_outflow_solver2.js's own OUTFLOW_TIMESTEP_FLOOR/SCALE for the two
// real bugs that were found and fixed to get here.

import * as tsl_array_n from 'tsl_array_n';
import { vec2, float, max } from 'three/tsl';
import { grid } from 'fluxflow';

const dyeCanvas = document.querySelector( '#outDye' );
const pressureCanvas = document.querySelector( '#outPressure' );
const statusEl = document.querySelector( '#status' );

function status( text, isErr ) {

	statusEl.textContent = text;
	statusEl.className = isErr ? 'err' : '';

}

const N = 64;
const dt = 1 / 30;
const pushStrength = 0.3; // uniform, every cell -- not a small region
const inflowSpeed = 2;
const cylinderRadius = N / 6; // diameter = N/3, per the user's own spec
const cylinderCenterX = N / 2;
const cylinderCenterY = N / 2;
const dyeSourceX = 3; // inject near the inflow wall, in a band matching the cylinder's own height
const dyeSourceHalfWidth = 4;
const injectionDensity = 1;
const dyeDecay = 0.995;
const pressureColorScale = 2;
const diagnosticInterval = 30; // frames between console readouts

function makeCirclePolygon( cx, cy, radius, segments ) {

	const verts = [];

	for ( let i = 0; i < segments; i ++ ) {

		const angle = ( i / segments ) * Math.PI * 2;
		verts.push( [ cx + radius * Math.cos( angle ), cy + radius * Math.sin( angle ) ] );

	}

	return verts;

}

// A thin strip hugging one wall. The three sides *other* than the one
// fluid-facing edge (`innerX`) are pushed a large distance away (`OUTER_MARGIN`)
// -- *not* just past the domain's own boundary by a cell or two, which is
// what an earlier version of this file did and which caused a real,
// confirmed-on-real-hardware bug: grid_outflow_solver2.js's own velocity
// extrapolation reads the outflow SDF's *gradient* to find which direction
// is "upstream" (toward the fluid), and that gradient points toward
// whichever polygon edge happens to be geometrically nearest. With only a
// 1-cell margin past the domain edge, faces past the strip's own midpoint
// found the *outer* (padding) edge nearer than the inner (fluid-facing)
// one -- flipping the gradient direction there, so two adjacent faces
// could each sample the *other* as its own "upstream" reference instead of
// both consistently pointing back toward the true fluid interior. That
// mutual, circular reference compounds every frame (each face's new value
// partly depends on the other's old one), and was confirmed on real
// WebGPU hardware to blow up past 1e34 within the very first frame.
// Pushing the non-fluid-facing edges this far away guarantees the
// fluid-facing edge is the nearest one -- and therefore the gradient
// direction is consistent -- for every point actually inside the real
// simulation domain, not just near it.
const OUTER_MARGIN = 1000;

function makeWallStripPolygon( innerX, outerX ) {

	return [ [ innerX, - OUTER_MARGIN ], [ outerX, - OUTER_MARGIN ], [ outerX, N + OUTER_MARGIN ], [ innerX, N + OUTER_MARGIN ] ];

}

try {

	const renderer = await tsl_array_n.init( { canvas: document.createElement( 'canvas' ), allowFallback: true } );
	status( `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' }` );

	const velocityGrid = grid.createFaceCenteredGrid2( N, N, 1, 1, 0, 0 );

	// dye state, ping-ponged -- see this file's own header comment for why
	// this is four fields, not two.
	const stateA = grid.createScalarGrid2( N, N, 1, 1, 0, 0 );
	const stateB = grid.createScalarGrid2( N, N, 1, 1, 0, 0 );
	stateA.clear();
	stateB.clear();

	const rawAdvectedA = { data: tsl_array_n.arrayN( 'float', [ N, N ] ) };
	const rawAdvectedB = { data: tsl_array_n.arrayN( 'float', [ N, N ] ) };

	const collider = grid.createSDFStaticCollider2( N, N, 1, 1, 0, 0 );
	collider.addPolygon( makeCirclePolygon( cylinderCenterX, cylinderCenterY, cylinderRadius, 48 ) );

	const inflow = grid.createSDFInflow2( N, N, 1, 1, 0, 0, { velocity: [ inflowSpeed, 0 ], mode: 'set' } );
	inflow.addPolygon( makeWallStripPolygon( 2, - OUTER_MARGIN ) ); // fluid-facing edge at x=2, padding falls away to the left

	const outflow = grid.createSDFOutflow2( N, N, 1, 1, 0, 0 );
	outflow.addPolygon( makeWallStripPolygon( N - 2, N + OUTER_MARGIN ) ); // fluid-facing edge at x=N-2, padding falls away to the right

	// uniform, whole-domain rightward push -- every cell, per the user's
	// own explicit correction (a force confined to a small region produces
	// no real animation once the region's own local flow settles).
	function force() {

		return vec2( pushStrength, 0 );

	}

	// right wall excluded: functionally required whenever outflow is
	// enabled, see this file's own header comment. Left excluded too when
	// inflow is enabled, for clarity only (inflow always wins there
	// regardless).
	const closedDomainBoundaryFlag = grid.DIRECTION_ALL & ~grid.DIRECTION_RIGHT & ~grid.DIRECTION_LEFT;

	const solver = grid.createGridSolver2( {
		velocityGrid,
		gridSpacing: [ 1, 1 ],
		origin: [ 0, 0 ],
		force,
		collider,
		inflows: inflow,
		outflows: outflow,
		closedDomainBoundaryFlag,
		dt,
		advection: { collider }, // NOT automatic -- see this file's own header comment
		// atomicScale reduced well below linalg.js's own default (65536) --
		// confirmed via real-hardware diagnostic logging that the default
		// overflows int32 in the atomic dot-product accumulator on a grid
		// this size (64x64=4096 cells, far more than any grid the default
		// was previously confirmed against): pAp/oldRZ/newRZ kept reading
		// back suspiciously close to +/-2^31/65536~=32768, the exact
		// signature of the accumulated integer saturating at int32's own
		// range. See linalg.js's own DEFAULT_ATOMIC_DOT_SCALE comment.
		// numberOfLevels: 4 (real multigrid coarsening, matching examples/
		// 06-07's own confirmed-stable configuration) is load-bearing, not
		// a tuning nicety -- confirmed on real WebGPU hardware that
		// numberOfLevels: 1 (plain red-black SOR relaxation only, no
		// coarse-grid correction) is *not* an adequate preconditioner at
		// this grid size: the outer PCG loop's own pAp/rTr genuinely
		// diverged (not merely a representation-level overflow -- reducing
		// atomicScale just moved the same divergence to a different
		// int32 ceiling) starting within the first ~10-15 iterations, for
		// this scene's inflow+force+pressure combination. This is the root
		// cause of the original "page loads and just sits there, frozen"
		// report -- switching back to numberOfLevels: 4 alone (no other
		// change) resolves it, confirmed stable (all-finite, low residual)
		// over 1000+ real-hardware frames.
		pressure: { multigrid: { numberOfLevels: 4 }, tolerance: 1e-5, maxIterations: 100, atomicScale: 1024 }
	} );

	// dye's own advection, bound to the solver's already-projected
	// velocityGrid -- separate from grid_solver2.js's internal advection
	// (which only self-advects velocity), since dye isn't part of that
	// orchestrator's scope at all. `collider` passed again here -- see
	// this file's own header comment on why this is a second, separate
	// requirement from the top-level `collider` option above.
	const dyeAdvectionSolver = grid.createSemiLagrangianAdvectionSolver2( { velocityGrid: solver.velocityGrid, dt, collider } );

	const advectAtoB = dyeAdvectionSolver.advectScalar2( stateA, rawAdvectedB );
	const advectBtoA = dyeAdvectionSolver.advectScalar2( stateB, rawAdvectedA );

	// injects dye in a band near the inflow wall, matching the cylinder's
	// own height, so the wake directly behind it is visible rather than
	// flooding the whole left edge with dye.
	function createInjectKernel( rawAdvectedGrid, stateGrid ) {

		return tsl_array_n.kernel( stateGrid.dataSize, ( i, j ) => {

			const pos = stateGrid.dataPosition( i, j );
			const raw = rawAdvectedGrid.data( i, j );

			const inSource = pos.x.lessThan( dyeSourceX ).and( pos.y.sub( cylinderCenterY ).abs().lessThan( dyeSourceHalfWidth ) );
			const sourceInjection = inSource.select( float( injectionDensity ), float( 0 ) );

			stateGrid.data( i, j ).assign( max( raw.mul( dyeDecay ), sourceInjection ) );

		} );

	}

	const injectAtoB = createInjectKernel( rawAdvectedB, stateB );
	const injectBtoA = createInjectKernel( rawAdvectedA, stateA );

	// part 3 of what an outflow does (grid_outflow_solver2.js) -- clears
	// dye that reaches the outflow strip, built once per dye field
	// (matching this port's established "build once, dispatch repeatedly"
	// convention).
	const clearOutflowA = solver.outflowSolver.clearOutflowScalarField( stateA );
	const clearOutflowB = solver.outflowSolver.clearOutflowScalarField( stateB );

	const dyeCtx = dyeCanvas.getContext( '2d' );
	const dyeImage = dyeCtx.createImageData( N, N );
	const pressureCtx = pressureCanvas.getContext( '2d' );
	const pressureImage = pressureCtx.createImageData( N, N );

	function clamp01( v ) {

		return Math.min( 1, Math.max( 0, v ) );

	}

	// canvas Y is down-positive, this grid's Y is up-positive -- flip rows
	// so the image matches the simulation's own orientation (both draw
	// functions below share this convention).
	function flippedPixelIndex( i, j ) {

		return ( ( N - 1 - j ) * N + i ) * 4;

	}

	function drawDye( data ) {

		for ( let j = 0; j < N; j ++ ) {

			for ( let i = 0; i < N; i ++ ) {

				const v = data[ i + N * j ];
				const pixel = flippedPixelIndex( i, j );
				const bright = 255 * clamp01( v );

				dyeImage.data[ pixel ] = bright;
				dyeImage.data[ pixel + 1 ] = bright;
				dyeImage.data[ pixel + 2 ] = bright;
				dyeImage.data[ pixel + 3 ] = 255;

			}

		}

		dyeCtx.putImageData( dyeImage, 0, 0 );

		// overlay the cylinder's own outline directly (cheap, analytic --
		// no extra GPU readback needed since its center/radius are already
		// known in plain JS) so it's clear where the collider actually is
		// relative to the dye pattern.
		dyeCtx.strokeStyle = '#f87171';
		dyeCtx.lineWidth = 1;
		dyeCtx.beginPath();
		dyeCtx.arc( cylinderCenterX, N - cylinderCenterY, cylinderRadius, 0, Math.PI * 2 );
		dyeCtx.stroke();

	}

	function drawPressure( data ) {

		for ( let j = 0; j < N; j ++ ) {

			for ( let i = 0; i < N; i ++ ) {

				const p = data[ i + N * j ];
				const pixel = flippedPixelIndex( i, j );
				const bright = 255 * clamp01( 0.5 + 0.5 * ( p / pressureColorScale ) );

				pressureImage.data[ pixel ] = bright;
				pressureImage.data[ pixel + 1 ] = bright;
				pressureImage.data[ pixel + 2 ] = bright;
				pressureImage.data[ pixel + 3 ] = 255;

			}

		}

		pressureCtx.putImageData( pressureImage, 0, 0 );

	}

	// same non-finite-aware diagnostic pair as examples/14-stable-fluids/
	// -- see that file's own header comment for why a naive min/max scan
	// silently hides NaN instead of reporting it.
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

	async function logDiagnostics( frameNumber, dyeData, pressureData ) {

		const [ uData, vData, bData ] = await Promise.all( [
			velocityGrid.dataU.toArray(),
			velocityGrid.dataV.toArray(),
			solver.pressureSolver.b.toArray()
		] );

		console.log(
			`fluxflow flow-past-cylinder [frame ${ frameNumber }] ` +
			`converged=${ solver.pressureSolver.diagnostics.converged } | ` +
			`${ fmt( 'dye', summarize( dyeData ) ) } | ` +
			`${ fmt( 'u', summarize( uData ) ) } | ` +
			`${ fmt( 'v', summarize( vData ) ) } | ` +
			`${ fmt( 'p', summarize( pressureData ) ) } | ` +
			`${ fmt( 'b(divergence)', summarize( bData ) ) }`
		);

	}

	let frame = 0;
	let nanDetected = false;

	// cheap: reuses the array already read back for drawing, no extra GPU
	// readback -- catches the same kind of failure logDiagnostics reports
	// on, just every frame instead of every diagnosticInterval frames.
	function checkForNonFinite( data, frameNumber ) {

		if ( nanDetected ) return;

		for ( let i = 0; i < data.length; i ++ ) {

			if ( ! Number.isFinite( data[ i ] ) ) {

				nanDetected = true;
				status( `non-finite dye value detected at frame ${ frameNumber } (index ${ i }, value ${ data[ i ] }) -- see this file's own header comment`, true );
				console.error( `fluxflow flow-past-cylinder: non-finite dye value at frame ${ frameNumber }, index ${ i }:`, data[ i ] );
				return;

			}

		}

	}

	async function animate() {

		await solver.onAdvanceTimeStep( dt );

		let currentState;

		if ( frame % 2 === 0 ) {

			advectAtoB();
			injectAtoB();
			clearOutflowB();
			currentState = stateB;

		} else {

			advectBtoA();
			injectBtoA();
			clearOutflowA();
			currentState = stateA;

		}

		const [ dyeData, pressureData ] = await Promise.all( [
			currentState.data.toArray(),
			solver.pressure.data.toArray()
		] );

		checkForNonFinite( dyeData, frame );

		if ( ! nanDetected && frame % diagnosticInterval === 0 ) await logDiagnostics( frame, dyeData, pressureData );

		if ( ! nanDetected ) {

			drawDye( dyeData );
			drawPressure( pressureData );

		}

		frame ++;
		if ( ! nanDetected ) requestAnimationFrame( animate );

	}

	requestAnimationFrame( animate );

} catch ( error ) {

	status( error.message, true );
	console.error( error );

}
