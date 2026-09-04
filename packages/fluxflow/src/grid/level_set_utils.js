// 移植自 level_set_utils.py。

import { float } from 'three/tsl';

export function isInsideSdf( phi ) {

	return phi.lessThan( 0 );

}

// 对应源码的 if/elif/elif 链——三个分支互斥、都只是纯值选择（没有副作用/提前
// 退出），直译成嵌套 select() 而不是 If()（源码注释也提到 Taichi 的 @ti.func
// 不支持运行时分支里提前 return，只能末尾 return 一次——TSL 这里更彻底，select()
// 干脆没有语句概念，只有表达式）。
// 第二层只查 inside0 就够（不用再 and not inside1）：能走到这层说明第一个分支
// (inside0 and inside1) 已经为 false，这时 inside0 为真就意味着 inside1 必然
// 为假；第三层同理，能走到这层说明前两层都不成立，此时 inside1 为真就意味着
// inside0 必然为假。
export function fractionInsideSdf( phi0, phi1 ) {

	const inside0 = isInsideSdf( phi0 );
	const inside1 = isInsideSdf( phi1 );

	return inside0.and( inside1 ).select(
		float( 1 ),
		inside0.select(
			phi0.div( phi0.sub( phi1 ) ),
			inside1.select(
				phi1.div( phi1.sub( phi0 ) ),
				float( 0 )
			)
		)
	);

}
