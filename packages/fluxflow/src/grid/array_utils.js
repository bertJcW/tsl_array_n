// 移植自 array_utils.py。
//
// 跟源码的调用方式不同：Taichi 的 @ti.kernel 函数本身"调用即派发"（Taichi 在背后
// 按参数类型签名缓存编译结果，每次调用只是重新派发）。tsl_array_n 的 kernel(shape,fn)
// 是"构建一次、返回可复用的 dispatch 函数"，构建时就要闭包绑定好具体的 field
// （tsl_array_n 现有能力边界，见移植计划决策4，不是这次移植引入的限制）。所以这里
// 两个函数都是"工厂"：调用一次构建（绑定好 field 的） kernel，返回的函数才是真正
// 每帧可以反复调用的 dispatcher——不要以为 createCopyKernel2(...) 这个调用本身
// 就把数据拷贝了，还要再调用一次它的返回值。

import * as tsl_array_n from 'tsl_array_n';
import { int, float } from 'three/tsl';

export function createCopyKernel2( src, dst, shape = src.shape ) {

	return tsl_array_n.kernel( shape, ( i, j ) => {

		dst( i, j ).assign( src( i, j ) );

	} );

}

// 单轮扩散：validSrc=1(有效)的格子原样标记为有效；validSrc=0(无效)的格子看
// 上下左右几个邻居里有几个 validSrc=1，把它们的 output 值取平均写进自己的 output，
// 一个有效邻居都没有就保持无效。
//
// 竞争条件检查：本函数在同一次 dispatch 里既读也写 output——但只有 validSrc[i,j]==0
// 的线程会写 output(i,j)，而被读取的邻居值全部来自 validSrc[neighbor]!=0 的格子，
// 这类格子这一轮必然走"已有效"分支、不会写 output。也就是说"会被读的格子"和
// "会写的格子"这一轮互不重叠，可以放心并行，不需要额外的原子操作或双缓冲 output。
function createExtrapolateStepKernel2( output, validSrc, validDst, shape ) {

	const [ nx, ny ] = shape;

	return tsl_array_n.kernel( shape, ( i, j ) => {

		tsl_array_n.If( validSrc( i, j ).notEqual( 0 ), () => {

			validDst( i, j ).assign( 1 );

		} ).Else( () => {

			const total = float( 0 ).toVar();
			const count = int( 0 ).toVar();

			tsl_array_n.If( i.add( 1 ).lessThan( nx ).and( validSrc( i.add( 1 ), j ).notEqual( 0 ) ), () => {

				total.addAssign( output( i.add( 1 ), j ) );
				count.addAssign( 1 );

			} );

			tsl_array_n.If( i.greaterThan( 0 ).and( validSrc( i.sub( 1 ), j ).notEqual( 0 ) ), () => {

				total.addAssign( output( i.sub( 1 ), j ) );
				count.addAssign( 1 );

			} );

			tsl_array_n.If( j.add( 1 ).lessThan( ny ).and( validSrc( i, j.add( 1 ) ).notEqual( 0 ) ), () => {

				total.addAssign( output( i, j.add( 1 ) ) );
				count.addAssign( 1 );

			} );

			tsl_array_n.If( j.greaterThan( 0 ).and( validSrc( i, j.sub( 1 ) ).notEqual( 0 ) ), () => {

				total.addAssign( output( i, j.sub( 1 ) ) );
				count.addAssign( 1 );

			} );

			tsl_array_n.If( count.greaterThan( 0 ), () => {

				output( i, j ).assign( total.div( count.toFloat() ) );
				validDst( i, j ).assign( 1 );

			} ).Else( () => {

				validDst( i, j ).assign( 0 );

			} );

		} );

	} );

}

// 把 validField=1(有效)区域的值向 =0(无效)区域扩散，每迭代一次往外扩一圈格子。
// 返回一个 run(numberOfIterations=5) 函数——validA/validB 两个内部 scratch buffer
// 和两个方向的 step kernel（A→B / B→A）在这里构建一次，之后每次调用 run() 都复用，
// 不重新构建 kernel 图（对应源码 validA, validB = validB, validA 的 ping-pong，
// 只是这里 kernel 不能像 Python 函数一样换绑参数，改成两个固定方向的 kernel 按
// 奇偶轮流调用，效果等价）。
export function createExtrapolateToRegion2( inputField, validField, outputField, shape = validField.shape ) {

	const copyInputToOutput = outputField !== inputField
		? createCopyKernel2( inputField, outputField, shape )
		: null;

	const validA = tsl_array_n.arrayN( 'int', shape );
	const validB = tsl_array_n.arrayN( 'int', shape );
	const copyValidToA = createCopyKernel2( validField, validA, shape );

	const stepAtoB = createExtrapolateStepKernel2( outputField, validA, validB, shape );
	const stepBtoA = createExtrapolateStepKernel2( outputField, validB, validA, shape );

	return function run( numberOfIterations = 5 ) {

		if ( copyInputToOutput ) copyInputToOutput();

		copyValidToA();

		for ( let iter = 0; iter < numberOfIterations; iter ++ ) {

			if ( iter % 2 === 0 ) stepAtoB();
			else stepBtoA();

		}

	};

}
