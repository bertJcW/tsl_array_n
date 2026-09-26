// *** A scene that seeds itself from Math.random cannot be run twice ***
//
// The jitter this feeds is wanted: particles left on an exact lattice produce
// visible artefacts, and every FLIP implementation breaks the lattice up. What
// was not wanted is where the numbers came from. Every seeding function in
// this package called Math.random(), so every run of a scene started from a
// different initial condition -- which makes a long run unrepeatable, a
// regression unbisectable, and a verdict a sample of one.
//
// Measured before the cause was found, while the suspicion was on GPU atomics:
// examples/33-flip-dam-break-3d/ gave two different verdicts across four
// 12,000-step runs, and the far cheaper createGridFlipSolver3 check on
// examples/32-grid-solver-3d/ -- 864 particles, a closed 8x8x8 domain, 60
// steps -- reported a peak particle speed of 3.532, 5.020 and 5.396 on three
// consecutive runs. That is the whole of it: not a race, not an atomic, not
// the solver. The initial condition.
//
// mulberry32, chosen because it is four lines, carries no state beyond one
// uint32, and is exactly reproducible on every engine. Seeding functions
// default their seed to a constant, so a scene is reproducible unless its
// author asks otherwise; pass randomSeed to vary it deliberately.

/**
 * A deterministic [0,1) generator. Same seed, same sequence, always.
 *
 * @param {number} seed any 32-bit integer
 * @returns {Function} random() in [0,1)
 */
export function createJitterRandom( seed ) {

	let state = seed >>> 0;

	return function random() {

		state = ( state + 0x6D2B79F5 ) >>> 0;
		let t = state;
		t = Math.imul( t ^ ( t >>> 15 ), t | 1 );
		t ^= t + Math.imul( t ^ ( t >>> 7 ), t | 61 );
		return ( ( t ^ ( t >>> 14 ) ) >>> 0 ) / 4294967296;

	};

}

/** The seed every seeding function in this package uses unless told otherwise. */
export const DEFAULT_JITTER_SEED = 0x9E3779B9;
