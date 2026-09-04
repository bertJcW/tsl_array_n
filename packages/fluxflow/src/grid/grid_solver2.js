// 移植自 grid_solver2.py。源码是个纯抽象基类（全是 pass 方法体），设计给具体求解器
// 继承、重写各个 compute* 方法。JS 这边没有类继承的传统（这次移植全程用工厂函数），
// 对应的写法是"依赖注入"：把各个阶段当 hooks 参数传进来，不传的阶段默认空操作——
// 效果和"子类只重写用得到的方法、其它保持 pass"完全一样。
//
// 具体求解器（真正的压力投影/对流/粘性实现）不在这次移植范围内（这次只做 grid/
// 这一层的地基），这个文件本身价值有限，纯粹是为了把 grid/ 文件夹搬完整。

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
