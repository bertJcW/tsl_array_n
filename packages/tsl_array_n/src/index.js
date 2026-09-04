import { init, isSupported, getRenderer } from './context.js';
import { arrayN, array0, array2, array3 } from './array.js';
import { kernel } from './kernel.js';
import { func } from './func.js';
import { Loop, Break, Continue, If } from './control.js';
import { uniform } from './uniform.js';

export { init, isSupported, getRenderer };
export { arrayN, array0, array2, array3 };
export { kernel };
export { func };
export { Loop, Break, Continue, If };
export { uniform };

export default {
	init, isSupported, getRenderer,
	arrayN, array0, array2, array3,
	kernel,
	func,
	Loop, Break, Continue, If,
	uniform
};
