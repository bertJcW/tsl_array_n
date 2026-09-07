// Vorticity confinement -- read directly from mantaflow's own
// vorticityConfinement/KnConfForce (source/plugin/extforces.cpp, Apache
// License 2.0, see ../../THIRD-PARTY-NOTICES.md for the attribution),
// itself an implementation of Fedkiw, Stam & Jensen's "Visual Simulation
// of Smoke" (SIGGRAPH 2001) -- the same paper jet/fluid-engine-dev's own
// GridSmokeSolver2 already cites, though jet itself never implements this
// specific technique. Compensates for the numerical dissipation of
// vortical structures inherent to semi-Lagrangian advection by pushing
// fluid toward regions where vorticity is already concentrated -- exactly
// the limitation examples/16-karman-vortex-street/'s own header comment
// already calls out ("no viscosity model... and no vorticity confinement
// either"), and directly relevant to keeping examples/18-explosion/'s own
// vortex-ring rollup sharp for longer instead of smoothing out.
//
// Deliberately a standalone, reusable utility -- not built into
// createGridSmokeSolver2 or createGridSolver2. Vorticity confinement is
// generic (mantaflow's own version lives in the general extforces.cpp,
// not a smoke-specific file), and this port's existing composable-force
// convention (`force: (pos) => a(pos).add(b(pos))`, already used by
// grid_smoke_solver2.js to combine buoyancy with a caller's own extra
// force) already gives any caller -- smoke, Karman street, or anything
// else -- a clean way to add this on top, with zero changes needed to
// grid_solver2.js or grid_smoke_solver2.js.
//
// *** The formula, re-derived here rather than just copied ***
//
// mantaflow's own KnConfForce (general N-D form): eta = normalize(grad),
// force = strength * cross(eta, curl), where grad = grad(|curl|). In 2D,
// curl (vorticity) is a scalar -- the z-component of the full 3D curl --
// and eta is a 2D vector. Expanding the 3D cross product with
// curl3D=(0,0,w) and eta3D=(ex,ey,0): eta3D x curl3D = (ey*w, -ex*w, 0).
// So the 2D force is `strength * (eta.y*curl, -eta.x*curl)`, matching the
// standard 2D vorticity-confinement formula in the graphics literature --
// verified by direct derivation, not assumed from memory.
//
// *** A degenerate case mantaflow's own code does not guard, but this
// port does, matching its own established pattern ***
//
// normalize(grad) is undefined (0/0) wherever the vorticity-magnitude
// gradient is exactly zero (a locally uniform-|vorticity| region, or an
// extremum) -- mantaflow's own KnConfForce calls normalize(grad) with no
// visible zero-check. This port has repeatedly added a guard beyond what
// a reference does for exactly this class of risk (linalg.js's own
// isDegenerateDot, grid_math.js's own bilinearGradientAtPosition2 fix,
// grid_outflow_solver2.js's own EXTRAPOLATED_VELOCITY_CLAMP) -- same
// treatment here: below GRADIENT_EPSILON, the confinement force is zero
// at that cell instead of computing a NaN direction (also physically
// sensible -- no gradient means no preferred push direction).
//
// *** No canonical default `strength` to port ***
//
// Unlike jet's buoyancy constants (real, tuned, literally carried over),
// mantaflow's own Python-exposed default is `strength=0` -- i.e.
// mantaflow ships this *off* by default and expects every scene to tune
// its own value for its own grid scale/spacing. This port does the same:
// no invented "universal" constant here, callers must supply their own
// tuned value (see examples/16-karman-vortex-street/ and
// examples/18-explosion/ for real, on-real-hardware-tuned starting
// points).

import * as tsl_array_n from 'tsl_array_n';
import { vec2, float, abs, length, max } from 'three/tsl';
import { createCellCenteredScalarGrid2 } from './grid_data2.js';
import { collocatedValueAtPosition2, scalarGradient2, faceCenteredCurlAtCenter2 } from './grid_math.js';

const GRADIENT_EPSILON = 1e-6;

function withSample( grid ) {

	return {
		...grid,
		sample( pos ) {

			return collocatedValueAtPosition2( grid.data, grid.gridSpacing, grid.dataOrigin, pos, grid.resolution );

		}
	};

}

function numberOrNode( value ) {

	return typeof value === 'number' ? float( value ) : value;

}

// options.velocityGrid: the FaceCenteredGrid2 to compute vorticity from
// and apply the confinement force to.
// options.gridSpacing: plain-number array, same convention/duplication as
// createGridSolver2's own options.gridSpacing (FaceCenteredGrid2's own
// gridSpacing is a TSL node, not recoverable as plain numbers).
// options.strength: plain number or an already-invoked live node (the
// "number or node" convention used throughout this port for dt/etc.) --
// no default; see this file's own header comment for why.
// Returns { update, force }. update() must be called once per rendered
// frame, *before* onAdvanceTimeStep() (the same sequencing
// grid_adaptive_timestep2.js's own update() already establishes) -- it
// computes curl and the confinement force from whichever velocity the
// *previous* frame finished with, so this frame's force stage applies a
// force derived from already-settled velocity, not a half-updated one
// (matching how grid_smoke_solver2.js's own buoyancy reads last frame's
// density/temperature -- not a new timing convention). force(pos) is a
// plain (pos) => vec2 closure, composable with any other force the same
// way buoyancy already is.
export function createVorticityConfinement2( { velocityGrid, gridSpacing, strength } ) {

	const [ resolutionX, resolutionY ] = velocityGrid.resolution;
	const [ gridSpacingX, gridSpacingY ] = gridSpacing;
	const shape = [ resolutionX, resolutionY ];
	const strengthNode = numberOrNode( strength );
	// scalarGradient2's own gridSpacing parameter must be a TSL vec2 node
	// (it divides a vec2 by it internally) -- not the plain [x,y] array
	// this factory's own options.gridSpacing convention uses everywhere
	// else, matching velocityGrid.gridSpacing's own node type.
	const gridSpacingNode = vec2( gridSpacingX, gridSpacingY );

	// Signed curl, cell-centered -- a bare arrayN (no continuous-position
	// sampling ever needed on this one, only ever read by discrete
	// neighbor index from computeConfinementForce below).
	const curlField = tsl_array_n.arrayN( 'float', shape );

	const computeCurl = tsl_array_n.kernel( shape, ( i, j ) => {

		curlField( i, j ).assign( faceCenteredCurlAtCenter2( velocityGrid.dataU, velocityGrid.dataV, velocityGrid.gridSpacing, i, j, shape ) );

	} );

	// forceX/forceY: createCellCenteredScalarGrid2 + a local sample(pos)
	// wrapper (sdf_collider2.js's own established precedent) -- needed
	// because force(pos) below must evaluate at arbitrary continuous
	// velocity-face positions, the same reason grid_smoke_solver2.js's own
	// density/temperature fields get the same wrapper.
	function buildForceComponent() {

		return withSample( createCellCenteredScalarGrid2( resolutionX, resolutionY, gridSpacingX, gridSpacingY, 0, 0 ) );

	}

	const forceX = buildForceComponent();
	const forceY = buildForceComponent();

	// scalarGradient2's own `data` parameter is a plain callable, not
	// necessarily a materialized field -- handing it this wrapper computes
	// grad(|curl|) directly, with no separate "materialize |curl|" scratch
	// field needed.
	function absCurl( i, j ) {

		return abs( curlField( i, j ) );

	}

	const computeConfinementForce = tsl_array_n.kernel( shape, ( i, j ) => {

		const grad = scalarGradient2( absCurl, gridSpacingNode, i, j, shape );
		const gradLen = length( grad );
		const degenerate = gradLen.lessThan( GRADIENT_EPSILON );
		const eta = grad.div( max( gradLen, GRADIENT_EPSILON ) ); // safe to evaluate even in the degenerate branch -- degenerate.select() below is what actually zeroes the result
		const curl = curlField( i, j );

		forceX.data( i, j ).assign( degenerate.select( float( 0 ), strengthNode.mul( eta.y ).mul( curl ) ) );
		forceY.data( i, j ).assign( degenerate.select( float( 0 ), strengthNode.mul( eta.x.negate() ).mul( curl ) ) );

	} );

	function update() {

		computeCurl();
		computeConfinementForce();

	}

	function force( pos ) {

		return vec2( forceX.sample( pos ), forceY.sample( pos ) );

	}

	return { update, force };

}
