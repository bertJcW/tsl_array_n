// Ported from grid_solver2.py. The source is a pure abstract base class (all
// method bodies are pass), designed for concrete solvers to subclass and
// override each compute* method. There's no class-inheritance tradition on
// the JS side (this whole port uses factory functions throughout), so the
// equivalent here is "dependency injection": each stage is passed in as a
// hook parameter, defaulting to a no-op when omitted -- exactly the same
// effect as "a subclass only overrides the methods it needs, the rest stay
// pass".
//
// A concrete solver (the actual pressure-projection/advection/viscosity
// implementation) is out of scope for this port (this pass only covers the
// grid/ foundation layer), so this file has limited value on its own --
// it's here purely to keep the grid/ folder's port complete.

const NOOP = () => {};

export function createGridSolver2( hooks = {} ) {

	const {
		computeExternalForces = NOOP,
		computeViscosity = NOOP,
		computePressure = NOOP,
		computeAdvection = NOOP,
		beginAdvanceTimeStep = NOOP,
		endAdvanceTimeStep = NOOP
	} = hooks;

	function onAdvanceTimeStep( timeStepInSeconds ) {

		beginAdvanceTimeStep( timeStepInSeconds );

		computeExternalForces( timeStepInSeconds );
		computeViscosity( timeStepInSeconds );
		computePressure( timeStepInSeconds );
		computeAdvection( timeStepInSeconds );

		endAdvanceTimeStep( timeStepInSeconds );

	}

	return { onAdvanceTimeStep };

}
