// 移植自 grid_data2.py。
//
// 源码用类继承表达"共享构造逻辑 + 少量差异"（CellCenteredVectorGrid2 extends
// CollocatedVectorGrid2，只在 __init__ 末尾多做一步 dataOrigin 偏移）。这里改用
// 工厂函数——跟 tsl_array_n 自己（arrayN/kernel/func 都是闭包函数，不是 class）
// 保持同一个风格。"继承"用 { ...baseGrid, 覆盖的字段 } 展开表达：调用base工厂拿到
// 完整对象，再覆盖需要变的那几个字段，效果跟子类 __init__ 里再赋值一次等价。
//
// resolution/dataSize 是普通 JS 数组（纯 CPU 侧数据，形状信息），gridSpacing/
// dataOrigin/origin 是真正的 TSL vec2 节点（构造时built一次，之后在任意 kernel/func
// 里直接引用同一个节点——TSL 节点不需要"在 kernel 内部"构造，这跟 tsl_array_n 自己
// uniform()/array0() 的用法是一回事）。
//
// .clear() 全部是真正预先 build 好、绑定了具体 field 的 kernel（tsl_array_n 现有
// kernel(shape,fn) 的既有能力边界——构造时闭包绑定，不是每次调用重新 build），
// 构造时另外用 fromArray(全零 typed array) 顺手把初始值置零（CPU 侧写入，不需要
// 等 tsl_array_n.init() 建好 renderer，用来对应源码 __init__ 末尾调用 self.clear()
// 那一步——避免依赖"构造这个 grid 时 init() 是否已经跑过"这个时序假设）。

import * as tsl_array_n from 'tsl_array_n';
import { vec2, float } from 'three/tsl';
import { faceCenteredValueAtPosition2 } from './grid_math.js';

function dataPositionFn( dataOrigin, gridSpacing ) {

	return function dataPosition( i, j ) {

		return dataOrigin.add( gridSpacing.mul( vec2( i.toFloat(), j.toFloat() ) ) );

	};

}

function zeroScalarField2( sizeX, sizeY ) {

	const data = tsl_array_n.array2( 'float', sizeX, sizeY );
	data.fromArray( new Float32Array( sizeX * sizeY ) );
	return data;

}

function zeroVectorField2( sizeX, sizeY ) {

	const data = tsl_array_n.array2( 'vec2', sizeX, sizeY );
	data.fromArray( new Float32Array( sizeX * sizeY * 2 ) );
	return data;

}

// vertex-centered 变体的 dataSize：每个维度 +1。源码 VertexCenteredScalarGrid2/
// VertexCenteredVectorGrid 的 dataSize property 在 resolution 严格等于 (0,0) 时
// 会保留 (0,0)（防御性分支）——这里没有照搬：grid/ 整个文件夹里没有任何地方真的用
// (0,0) 构造过这两个类（grep 过，纯 property getter 里的防御分支，从没被触发过），
// 而且 tsl_array_n.array2() 本身就不接受 0 长度的维度（会直接抛错），没法造出一个
// 真正 0×0 的 field——所以这个分支在这次移植的目标平台上既没有被用到的先例，
// 也没有对应的底层能力，就不搬了。以后如果真的需要"未配置的占位 grid"这种概念，
// 应该在 JS 层用 null/undefined 表达，而不是造一个 0×0 的 field。
function vertexDataSize( resolutionX, resolutionY ) {

	return [ resolutionX + 1, resolutionY + 1 ];

}

// ------------------------------------------------------------
// collocated vector grid

export function createCollocatedVectorGrid2( resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY ) {

	const resolution = [ resolutionX, resolutionY ];
	const gridSpacing = vec2( gridSpacingX, gridSpacingY );
	const dataOrigin = vec2( originX, originY );
	const dataSize = resolution;

	const data = zeroVectorField2( dataSize[ 0 ], dataSize[ 1 ] );
	const clear = tsl_array_n.kernel( dataSize, ( i, j ) => {

		data( i, j ).assign( vec2( 0 ) );

	} );

	return {
		resolution, gridSpacing, dataOrigin, dataSize, data, clear,
		dataPosition: dataPositionFn( dataOrigin, gridSpacing )
	};

}

// ------------------------------------------------------------
// cell centered vector grid

export function createCellCenteredVectorGrid2( resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY ) {

	const grid = createCollocatedVectorGrid2( resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY );
	const dataOrigin = grid.dataOrigin.add( grid.gridSpacing.mul( 0.5 ) );

	return {
		...grid,
		dataOrigin,
		dataPosition: dataPositionFn( dataOrigin, grid.gridSpacing )
	};

}

// ------------------------------------------------------------
// vertex centered vector grid

export function createVertexCenteredVectorGrid2( resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY ) {

	const resolution = [ resolutionX, resolutionY ];
	const gridSpacing = vec2( gridSpacingX, gridSpacingY );
	const dataOrigin = vec2( originX, originY );
	const dataSize = vertexDataSize( resolutionX, resolutionY );

	const data = zeroVectorField2( dataSize[ 0 ], dataSize[ 1 ] );
	const clear = tsl_array_n.kernel( dataSize, ( i, j ) => {

		data( i, j ).assign( vec2( 0 ) );

	} );

	return {
		resolution, gridSpacing, dataOrigin, dataSize, data, clear,
		dataPosition: dataPositionFn( dataOrigin, gridSpacing )
	};

}

// ------------------------------------------------------------
// face centered grid (MAC grid / staggered grid)

export function createFaceCenteredGrid2( resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY ) {

	const resolution = [ resolutionX, resolutionY ];
	const gridSpacing = vec2( gridSpacingX, gridSpacingY );
	const origin = vec2( originX, originY );

	const dataSizeU = [ resolutionX + 1, resolutionY ];
	const dataSizeV = [ resolutionX, resolutionY + 1 ];

	const dataOriginU = origin.add( vec2( 0, gridSpacing.y.mul( 0.5 ) ) );
	const dataOriginV = origin.add( vec2( gridSpacing.x.mul( 0.5 ), 0 ) );

	const dataU = zeroScalarField2( dataSizeU[ 0 ], dataSizeU[ 1 ] );
	const dataV = zeroScalarField2( dataSizeV[ 0 ], dataSizeV[ 1 ] );

	// 源码 clear() 是一个 kernel 里两个 top-level for 循环（dataU 一个、dataV 一个）
	// ——Taichi 支持一个 kernel 里多个并行 for，tsl_array_n 的 kernel(shape,fn) 是
	// 一个 shape 对应一次 dispatch，dataU/dataV 形状还不一样，天然没法塞进同一个
	// kernel。改成两个 kernel，clear() 依次调用两个——效果等价，只是从"一个 kernel
	// 两段循环"变成"两个 kernel 依次 dispatch"。
	const clearU = tsl_array_n.kernel( dataSizeU, ( i, j ) => { dataU( i, j ).assign( float( 0 ) ); } );
	const clearV = tsl_array_n.kernel( dataSizeV, ( i, j ) => { dataV( i, j ).assign( float( 0 ) ); } );

	function clear() {

		clearU();
		clearV();

	}

	const uPosition = dataPositionFn( dataOriginU, gridSpacing );
	const vPosition = dataPositionFn( dataOriginV, gridSpacing );

	// 双线性采样 u/v 两个分量、合成一个 vec2 速度——薄封装，真正的数学在 grid_math
	// 里，这里不重复写
	function sample( pos ) {

		return faceCenteredValueAtPosition2(
			dataU, dataV, gridSpacing,
			dataOriginU, dataOriginV, pos,
			dataSizeU, dataSizeV
		);

	}

	return {
		resolution, gridSpacing, origin,
		dataSizeU, dataSizeV, dataOriginU, dataOriginV,
		dataU, dataV, clear, uPosition, vPosition, sample
	};

}

// ------------------------------------------------------------
// scalar grid

export function createScalarGrid2( resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY ) {

	const resolution = [ resolutionX, resolutionY ];
	const gridSpacing = vec2( gridSpacingX, gridSpacingY );
	const dataOrigin = vec2( originX, originY );
	const dataSize = resolution;

	const data = zeroScalarField2( dataSize[ 0 ], dataSize[ 1 ] );
	const clear = tsl_array_n.kernel( dataSize, ( i, j ) => {

		data( i, j ).assign( float( 0 ) );

	} );

	return {
		resolution, gridSpacing, dataOrigin, dataSize, data, clear,
		dataPosition: dataPositionFn( dataOrigin, gridSpacing )
	};

}

// ------------------------------------------------------------
// cell centered scalar grid

export function createCellCenteredScalarGrid2( resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY ) {

	const grid = createScalarGrid2( resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY );
	const dataOrigin = grid.dataOrigin.add( grid.gridSpacing.mul( 0.5 ) );

	return {
		...grid,
		dataOrigin,
		dataPosition: dataPositionFn( dataOrigin, grid.gridSpacing )
	};

}

// ------------------------------------------------------------
// vertex centered scalar grid

export function createVertexCenteredScalarGrid2( resolutionX, resolutionY, gridSpacingX, gridSpacingY, originX, originY ) {

	const resolution = [ resolutionX, resolutionY ];
	const gridSpacing = vec2( gridSpacingX, gridSpacingY );
	const dataOrigin = vec2( originX, originY );
	const dataSize = vertexDataSize( resolutionX, resolutionY );

	const data = zeroScalarField2( dataSize[ 0 ], dataSize[ 1 ] );
	const clear = tsl_array_n.kernel( dataSize, ( i, j ) => {

		data( i, j ).assign( float( 0 ) );

	} );

	return {
		resolution, gridSpacing, dataOrigin, dataSize, data, clear,
		dataPosition: dataPositionFn( dataOrigin, gridSpacing )
	};

}
