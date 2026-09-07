// CFL (Courant-Friedrichs-Lewy) substep-count computation, ported from
// jet/fluid-engine-dev's own GridFluidSolver2::cfl()/numberOfSubTimeSteps()
// (grid_fluid_solver2.cpp, MIT license, Doyub Kim -- see
// ../../THIRD-PARTY-NOTICES.md for the attribution). A plain function of
// plain numbers, with zero GPU/TSL dependency -- deliberately: this is the
// one piece of "how many substeps does this frame need" that doesn't care
// whether the caller is a grid (Eulerian) solver or a future FLIP (hybrid
// particle-grid) one, only that it already reduced its own state down to a
// single max-velocity-magnitude number. jet's own PicSolver2 (FLIP's base
// class) confirms this split is real, not speculative: it *inherits*
// GridFluidSolver2's numberOfSubTimeSteps()/cfl() unchanged for its own
// per-frame substep count, and only adds a *separate*, additional inner
// substep loop for particle-position integration on top (pic_solver2.cpp,
// RK2 midpoint rule) -- a FLIP solver built on this port later is expected
// to do the same: reuse this function for the outer per-frame count, add
// its own particle-integration substepping alongside it.

// maxVelocityMagnitude/minGridSpacing/frameDt: plain numbers. The caller is
// responsible for producing maxVelocityMagnitude however makes sense for
// its own solver (e.g. grid/reduction.js's createMaxAbsReducer over a
// FaceCenteredGrid2's dataU/dataV, or a future FLIP solver's own reduction
// over particle velocities) -- this function doesn't know or care which.
//
// courantNumber: default 5, matching jet's own GridFluidSolver2 default
// (grid_fluid_solver2.h, `_maxCfl = 5.0`). Safe meaningfully above 1
// specifically because this port's advection is semi-Lagrangian
// (unconditionally stable regardless of CFL, unlike an explicit
// forward-Euler scheme) -- a larger courantNumber trades advection
// accuracy (each substep's backward trace covers more grid cells, so more
// interpolation smoothing per substep) for fewer, larger substeps, not
// stability.
//
// maxSubSteps: NOT in jet's own reference -- a defensive cap added here,
// same spirit as this port's own MAX_ALPHA_MAGNITUDE/MAX_BETA_MAGNITUDE
// circuit breakers (linalg.js), so a velocity spike (or a caller-supplied
// bad frameDt/courantNumber) can't silently demand hundreds of substeps
// and stall a frame.
export function computeAdaptiveSubSteps( maxVelocityMagnitude, minGridSpacing, frameDt, options = {} ) {

	const courantNumber = options.courantNumber ?? 5;
	const maxSubSteps = options.maxSubSteps ?? 32;

	const cfl = maxVelocityMagnitude * frameDt / minGridSpacing;
	const numSubSteps = Math.min( maxSubSteps, Math.max( 1, Math.ceil( cfl / courantNumber ) ) );

	return { numSubSteps, subDt: frameDt / numSubSteps };

}
