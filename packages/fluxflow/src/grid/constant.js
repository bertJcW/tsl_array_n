// 移植自 constant.py。
//
// DTYPE 的可切换精度（setPrecision(use_double)）在 WGSL/WebGPU compute 里没有
// 对应物（没有原生 f64），不搬——固定只支持 float(f32)。
//
// initConstant() 在源码里存在，是因为 GRAVITY 要等 setPrecision 定好 DTYPE 之后
// 才能以正确的精度分配 field，所以需要一个"先调用 initConstant 再用 GRAVITY"的
// 编排步骤。JS 这边没有精度切换，不需要等待任何东西，GRAVITY 什么时候要用什么时候
// 建就行，initConstant 这层编排本身就不需要了——不是漏搬，是它存在的前提没了。
//
// GRAVITY 本身也不做成模块级单例（源码里 GRAVITY 是整个 constant.py 模块共享的
// 全局 field）：JS 这边每个 solver/模拟各自建一个，避免一个页面上同时跑多个模拟时
// 共享一份可变全局状态。

import * as tsl_array_n from 'tsl_array_n';

export const FLOAT_TYPE = 'float';

export const DIRECTION_NONE  = 0;
export const DIRECTION_LEFT  = 1 << 0;
export const DIRECTION_RIGHT = 1 << 1;
export const DIRECTION_DOWN  = 1 << 2;
export const DIRECTION_UP    = 1 << 3;
export const DIRECTION_ALL   = DIRECTION_LEFT | DIRECTION_RIGHT | DIRECTION_DOWN | DIRECTION_UP;

export const DEFAULT_GRAVITY = -9.81;

// 对应源码 initConstant() 里 GRAVITY = ti.field(...); GRAVITY[None] = -9.81 这两行。
export function createGravity( value = DEFAULT_GRAVITY ) {

	const gravity = tsl_array_n.array0( 'float' );
	gravity.fromArray( new Float32Array( [ value ] ) );
	return gravity;

}

// 对应源码 setGravity(newGravity)——源码改的是隐式的模块级 GRAVITY，这里要传入
// 具体是哪个 gravity（因为不再是单例）。
export function setGravity( gravity, newValue ) {

	gravity.fromArray( new Float32Array( [ newValue ] ) );

}
