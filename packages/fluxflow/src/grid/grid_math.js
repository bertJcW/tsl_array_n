// 移植自 grid_math.py。这里全部是普通 JS 函数（构建/组合 TSL 节点图），不用
// tsl_array_n.func() ——这些函数经常要返回好几个值（比如 bilinearCoordsAndWeights2
// 要返回 8 个量）、参数也都是具名的，而 func()（裸 Fn 重导出）的调用约定是"单个
// 数组解构参数"，硬套上去既别扭又容易踩坑（见 tsl_array_n README 的已知边界）。
// 这些函数只在 JS 图构建阶段被别的函数/kernel 调用，不需要是"设备侧可调用子程序"，
// 普通 JS 函数直接返回/组合节点就够用。
//
// 命名上把源码 snake_case+camelCase 混用统一成纯 camelCase（比如
// collocated_valueAtPosition2 -> collocatedValueAtPosition2），并顺手把源码里
// "colloacted"的拼写错误改成"collocated"——纯命名调整，数值逻辑逐行对应源码。

import { int, float, vec2, mat2, min, max, floor } from 'three/tsl';

// ------------------------------------------------------------
// interpolation

export function faceCenteredValueAtCellCenter2( dataU, dataV, i, j ) {

	return vec2(
		dataU( i, j ).add( dataU( i.add( 1 ), j ) ),
		dataV( i, j ).add( dataV( i, j.add( 1 ) ) )
	).mul( 0.5 );

}

// 给定连续位置，找到周围 4 个格点下标（clamp 到边界内，落在网格外的位置也能采样到
// 边界值）和它们的双线性权重，顺序 (i0,j0) (i1,j0) (i0,j1) (i1,j1)
export function bilinearCoordsAndWeights2( pos, dataOrigin, gridSpacing, shape ) {

	const [ nx, ny ] = shape;

	const gridPos = pos.sub( dataOrigin ).div( gridSpacing );
	const i0 = int( floor( gridPos.x ) );
	const j0 = int( floor( gridPos.y ) );
	const fx = gridPos.x.sub( i0.toFloat() );
	const fy = gridPos.y.sub( j0.toFloat() );

	const i0c = max( 0, min( i0, nx - 1 ) );
	const i1c = max( 0, min( i0.add( 1 ), nx - 1 ) );
	const j0c = max( 0, min( j0, ny - 1 ) );
	const j1c = max( 0, min( j0.add( 1 ), ny - 1 ) );

	const gx = float( 1 ).sub( fx );
	const gy = float( 1 ).sub( fy );

	const w00 = gx.mul( gy );
	const w10 = fx.mul( gy );
	const w01 = gx.mul( fy );
	const w11 = fx.mul( fy );

	return { i0c, j0c, i1c, j1c, w00, w10, w01, w11 };

}

// collocated（标量或向量都行）field 在连续位置的双线性采样
export function collocatedValueAtPosition2( data, gridSpacing, dataOrigin, pos, shape ) {

	const { i0c, j0c, i1c, j1c, w00, w10, w01, w11 } = bilinearCoordsAndWeights2( pos, dataOrigin, gridSpacing, shape );

	return data( i0c, j0c ).mul( w00 )
		.add( data( i1c, j0c ).mul( w10 ) )
		.add( data( i0c, j1c ).mul( w01 ) )
		.add( data( i1c, j1c ).mul( w11 ) );

}

// face-centered (MAC) 速度场在连续位置的双线性采样
// dataOriginU/dataOriginV 是 u-face/v-face 各自的交错原点
export function faceCenteredValueAtPosition2( dataU, dataV, gridSpacing, dataOriginU, dataOriginV, pos, shapeU, shapeV ) {

	const u = collocatedValueAtPosition2( dataU, gridSpacing, dataOriginU, pos, shapeU );
	const v = collocatedValueAtPosition2( dataV, gridSpacing, dataOriginV, pos, shapeV );

	return vec2( u, v );

}

// ------------------------------------------------------------
// gradient

// shape 是 field 的形状；在格点上求梯度
export function scalarGradient2( data, gridSpacing, i, j, shape ) {

	const [ nx, ny ] = shape;
	const center = data( i, j );

	const left  = i.greaterThan( 0 ).select( data( i.sub( 1 ), j ), center );
	const right = i.add( 1 ).lessThan( nx ).select( data( i.add( 1 ), j ), center );
	const down  = j.greaterThan( 0 ).select( data( i, j.sub( 1 ) ), center );
	const up    = j.add( 1 ).lessThan( ny ).select( data( i, j.add( 1 ) ), center );

	return vec2( right.sub( left ), up.sub( down ) ).mul( 0.5 ).div( gridSpacing );

}

// shape 是 field 的形状；在格点上求梯度（Jacobian）
export function vectorGradient2( data, gridSpacing, i, j, shape ) {

	const [ nx, ny ] = shape;
	const center = data( i, j );

	const left  = i.greaterThan( 0 ).select( data( i.sub( 1 ), j ), center );
	const right = i.add( 1 ).lessThan( nx ).select( data( i.add( 1 ), j ), center );
	const down  = j.greaterThan( 0 ).select( data( i, j.sub( 1 ) ), center );
	const up    = j.add( 1 ).lessThan( ny ).select( data( i, j.add( 1 ) ), center );

	const gradX = vec2( right.x.sub( left.x ), up.x.sub( down.x ) ).mul( 0.5 ).div( gridSpacing );
	const gradY = vec2( right.y.sub( left.y ), up.y.sub( down.y ) ).mul( 0.5 ).div( gridSpacing );

	// 注意：mat2(a,b,c,d) 的元素填充顺序（row-major/column-major）在 Taichi
	// 的 tm.mat2 和 TSL 的 mat2 之间还没做过数值验证是否一致——这里按源码参数顺序
	// 直译。用之前先拿一个非对称的向量场（比如 (2x+5y, 0)，对称场看不出转置）实跑
	// 验证一下 Jacobian 有没有被转置，不要直接信任这个函数。
	return mat2(
		gradX.x, gradX.y,
		gradY.x, gradY.y
	);

}

// 标量场在连续位置的梯度：双线性混合 4 个周围格点各自的离散梯度
export function scalarGradientAtPosition2( data, gridSpacing, dataOrigin, pos, shape ) {

	const { i0c, j0c, i1c, j1c, w00, w10, w01, w11 } = bilinearCoordsAndWeights2( pos, dataOrigin, gridSpacing, shape );

	return scalarGradient2( data, gridSpacing, i0c, j0c, shape ).mul( w00 )
		.add( scalarGradient2( data, gridSpacing, i1c, j0c, shape ).mul( w10 ) )
		.add( scalarGradient2( data, gridSpacing, i0c, j1c, shape ).mul( w01 ) )
		.add( scalarGradient2( data, gridSpacing, i1c, j1c, shape ).mul( w11 ) );

}

// 标量场在连续位置的梯度：对 collocatedValueAtPosition2 采样用的同一个双线性曲面
// 解析求导（不是像 scalarGradientAtPosition2 那样混合 4 个角点各自的离散梯度）——
// 更便宜（只读 4 个角点值一次），且跟 collocatedValueAtPosition2 在该点返回的值
// 精确对应。需要梯度跟采样值精确匹配时用这个（比如 SDF 的法向量）；想要在相邻格子间
// 多一点平滑时用 scalarGradientAtPosition2。
export function bilinearGradientAtPosition2( data, gridSpacing, dataOrigin, pos, shape ) {

	const { i0c, j0c, i1c, j1c, w10, w01, w11 } = bilinearCoordsAndWeights2( pos, dataOrigin, gridSpacing, shape );

	const fx = w10.add( w11 );
	const fy = w01.add( w11 );

	const v00 = data( i0c, j0c );
	const v10 = data( i1c, j0c );
	const v01 = data( i0c, j1c );
	const v11 = data( i1c, j1c );

	const gx = float( 1 ).sub( fy ).mul( v10.sub( v00 ) ).add( fy.mul( v11.sub( v01 ) ) ).div( gridSpacing.x );
	const gy = float( 1 ).sub( fx ).mul( v01.sub( v00 ) ).add( fx.mul( v11.sub( v10 ) ) ).div( gridSpacing.y );

	return vec2( gx, gy );

}

// 向量场在连续位置的梯度（Jacobian）
export function vectorGradientAtPosition2( data, gridSpacing, dataOrigin, pos, shape ) {

	const { i0c, j0c, i1c, j1c, w00, w10, w01, w11 } = bilinearCoordsAndWeights2( pos, dataOrigin, gridSpacing, shape );

	return vectorGradient2( data, gridSpacing, i0c, j0c, shape ).mul( w00 )
		.add( vectorGradient2( data, gridSpacing, i1c, j0c, shape ).mul( w10 ) )
		.add( vectorGradient2( data, gridSpacing, i0c, j1c, shape ).mul( w01 ) )
		.add( vectorGradient2( data, gridSpacing, i1c, j1c, shape ).mul( w11 ) );

}

// ------------------------------------------------------------
// divergence

// data 是 2D 向量场，gridSpacing 是向量；在格点上求散度
export function collocatedDivergence2( data, gridSpacing, i, j, shape ) {

	const [ nx, ny ] = shape;
	const center = data( i, j );

	const left  = i.greaterThan( 0 ).select( data( i.sub( 1 ), j ).x, center.x );
	const right = i.add( 1 ).lessThan( nx ).select( data( i.add( 1 ), j ).x, center.x );
	const down  = j.greaterThan( 0 ).select( data( i, j.sub( 1 ) ).y, center.y );
	const up    = j.add( 1 ).lessThan( ny ).select( data( i, j.add( 1 ) ).y, center.y );

	return right.sub( left ).mul( 0.5 ).div( gridSpacing.x )
		.add( up.sub( down ).mul( 0.5 ).div( gridSpacing.y ) );

}

// dataU/dataV 是两个标量场；在每个 cell center 求散度
export function faceCenteredDivergenceAtCenter2( dataU, dataV, gridSpacing, i, j ) {

	const leftU   = dataU( i, j );
	const rightU  = dataU( i.add( 1 ), j );
	const bottomV = dataV( i, j );
	const topV    = dataV( i, j.add( 1 ) );

	return rightU.sub( leftU ).div( gridSpacing.x )
		.add( topV.sub( bottomV ).div( gridSpacing.y ) );

}

// collocated 向量场在连续位置的散度
export function collocatedDivergenceAtPosition2( data, gridSpacing, dataOrigin, pos, shape ) {

	const { i0c, j0c, i1c, j1c, w00, w10, w01, w11 } = bilinearCoordsAndWeights2( pos, dataOrigin, gridSpacing, shape );

	return collocatedDivergence2( data, gridSpacing, i0c, j0c, shape ).mul( w00 )
		.add( collocatedDivergence2( data, gridSpacing, i1c, j0c, shape ).mul( w10 ) )
		.add( collocatedDivergence2( data, gridSpacing, i0c, j1c, shape ).mul( w01 ) )
		.add( collocatedDivergence2( data, gridSpacing, i1c, j1c, shape ).mul( w11 ) );

}

// face-centered (MAC) 向量场在连续位置的散度
// cellCenterOrigin 是 cell-center 布局的原点（dataOrigin + 0.5*gridSpacing），
// shape 是 cell-center 分辨率 (nx, ny)
export function faceCenteredDivergenceAtPosition2( dataU, dataV, gridSpacing, cellCenterOrigin, pos, shape ) {

	const { i0c, j0c, i1c, j1c, w00, w10, w01, w11 } = bilinearCoordsAndWeights2( pos, cellCenterOrigin, gridSpacing, shape );

	return faceCenteredDivergenceAtCenter2( dataU, dataV, gridSpacing, i0c, j0c ).mul( w00 )
		.add( faceCenteredDivergenceAtCenter2( dataU, dataV, gridSpacing, i1c, j0c ).mul( w10 ) )
		.add( faceCenteredDivergenceAtCenter2( dataU, dataV, gridSpacing, i0c, j1c ).mul( w01 ) )
		.add( faceCenteredDivergenceAtCenter2( dataU, dataV, gridSpacing, i1c, j1c ).mul( w11 ) );

}

// ------------------------------------------------------------
// curl

// data 是 2D 向量场，gridSpacing 是向量；在格点上求旋度
export function collocatedCurl2( data, gridSpacing, i, j, shape ) {

	const [ nx, ny ] = shape;
	const center = data( i, j );

	const left  = i.greaterThan( 0 ).select( data( i.sub( 1 ), j ), center );
	const right = i.add( 1 ).lessThan( nx ).select( data( i.add( 1 ), j ), center );
	const down  = j.greaterThan( 0 ).select( data( i, j.sub( 1 ) ), center );
	const up    = j.add( 1 ).lessThan( ny ).select( data( i, j.add( 1 ) ), center );

	const fxYm = down.x;
	const fxYp = up.x;
	const fyXm = left.y;
	const fyXp = right.y;

	return fyXp.sub( fyXm ).div( gridSpacing.x )
		.sub( fxYp.sub( fxYm ).div( gridSpacing.y ) )
		.mul( 0.5 );

}

// dataU/dataV 是两个标量场；在每个 cell center 求旋度
export function faceCenteredCurlAtCenter2( dataU, dataV, gridSpacing, i, j, shape ) {

	const [ nx, ny ] = shape;

	const left  = faceCenteredValueAtCellCenter2( dataU, dataV, max( i.sub( 1 ), 0 ), j );
	const right = faceCenteredValueAtCellCenter2( dataU, dataV, min( i.add( 1 ), nx - 1 ), j );
	const up    = faceCenteredValueAtCellCenter2( dataU, dataV, i, min( j.add( 1 ), ny - 1 ) );
	const down  = faceCenteredValueAtCellCenter2( dataU, dataV, i, max( j.sub( 1 ), 0 ) );

	const fxYm = down.x;
	const fxYp = up.x;
	const fyXm = left.y;
	const fyXp = right.y;

	return fyXp.sub( fyXm ).div( gridSpacing.x )
		.sub( fxYp.sub( fxYm ).div( gridSpacing.y ) )
		.mul( 0.5 );

}

// collocated 向量场在连续位置的旋度
export function collocatedCurlAtPosition2( data, gridSpacing, dataOrigin, pos, shape ) {

	const { i0c, j0c, i1c, j1c, w00, w10, w01, w11 } = bilinearCoordsAndWeights2( pos, dataOrigin, gridSpacing, shape );

	return collocatedCurl2( data, gridSpacing, i0c, j0c, shape ).mul( w00 )
		.add( collocatedCurl2( data, gridSpacing, i1c, j0c, shape ).mul( w10 ) )
		.add( collocatedCurl2( data, gridSpacing, i0c, j1c, shape ).mul( w01 ) )
		.add( collocatedCurl2( data, gridSpacing, i1c, j1c, shape ).mul( w11 ) );

}

// face-centered (MAC) 向量场在连续位置的旋度
// cellCenterOrigin/shape：见 faceCenteredDivergenceAtPosition2
export function faceCenteredCurlAtPosition2( dataU, dataV, gridSpacing, cellCenterOrigin, pos, shape ) {

	const { i0c, j0c, i1c, j1c, w00, w10, w01, w11 } = bilinearCoordsAndWeights2( pos, cellCenterOrigin, gridSpacing, shape );

	return faceCenteredCurlAtCenter2( dataU, dataV, gridSpacing, i0c, j0c, shape ).mul( w00 )
		.add( faceCenteredCurlAtCenter2( dataU, dataV, gridSpacing, i1c, j0c, shape ).mul( w10 ) )
		.add( faceCenteredCurlAtCenter2( dataU, dataV, gridSpacing, i0c, j1c, shape ).mul( w01 ) )
		.add( faceCenteredCurlAtCenter2( dataU, dataV, gridSpacing, i1c, j1c, shape ).mul( w11 ) );

}

// ------------------------------------------------------------
// laplacian

// shape 是 field 的形状；在格点上求拉普拉斯
export function scalarLaplacian2( data, gridSpacing, i, j, shape ) {

	const [ nx, ny ] = shape;
	const center = data( i, j );
	const zero = float( 0 );

	const dleft  = i.greaterThan( 0 ).select( center.sub( data( i.sub( 1 ), j ) ), zero );
	const dright = i.lessThan( nx - 1 ).select( data( i.add( 1 ), j ).sub( center ), zero );
	const dup    = j.lessThan( ny - 1 ).select( data( i, j.add( 1 ) ).sub( center ), zero );
	const ddown  = j.greaterThan( 0 ).select( center.sub( data( i, j.sub( 1 ) ) ), zero );

	return dright.sub( dleft ).div( gridSpacing.x.mul( gridSpacing.x ) )
		.add( dup.sub( ddown ).div( gridSpacing.y.mul( gridSpacing.y ) ) );

}

// 标量场在连续位置的拉普拉斯
export function scalarLaplacianAtPosition2( data, gridSpacing, dataOrigin, pos, shape ) {

	const { i0c, j0c, i1c, j1c, w00, w10, w01, w11 } = bilinearCoordsAndWeights2( pos, dataOrigin, gridSpacing, shape );

	return scalarLaplacian2( data, gridSpacing, i0c, j0c, shape ).mul( w00 )
		.add( scalarLaplacian2( data, gridSpacing, i1c, j0c, shape ).mul( w10 ) )
		.add( scalarLaplacian2( data, gridSpacing, i0c, j1c, shape ).mul( w01 ) )
		.add( scalarLaplacian2( data, gridSpacing, i1c, j1c, shape ).mul( w11 ) );

}

// shape 是 field 的形状；在格点上求拉普拉斯
export function vectorLaplacian2( data, gridSpacing, i, j, shape ) {

	const [ nx, ny ] = shape;
	const center = data( i, j );
	const zero = vec2( 0 );

	const dleft  = i.greaterThan( 0 ).select( center.sub( data( i.sub( 1 ), j ) ), zero );
	const dright = i.lessThan( nx - 1 ).select( data( i.add( 1 ), j ).sub( center ), zero );
	const dup    = j.lessThan( ny - 1 ).select( data( i, j.add( 1 ) ).sub( center ), zero );
	const ddown  = j.greaterThan( 0 ).select( center.sub( data( i, j.sub( 1 ) ) ), zero );

	return dright.sub( dleft ).div( gridSpacing.x.mul( gridSpacing.x ) )
		.add( dup.sub( ddown ).div( gridSpacing.y.mul( gridSpacing.y ) ) );

}

// 向量场在连续位置的拉普拉斯
export function vectorLaplacianAtPosition2( data, gridSpacing, dataOrigin, pos, shape ) {

	const { i0c, j0c, i1c, j1c, w00, w10, w01, w11 } = bilinearCoordsAndWeights2( pos, dataOrigin, gridSpacing, shape );

	return vectorLaplacian2( data, gridSpacing, i0c, j0c, shape ).mul( w00 )
		.add( vectorLaplacian2( data, gridSpacing, i1c, j0c, shape ).mul( w10 ) )
		.add( vectorLaplacian2( data, gridSpacing, i0c, j1c, shape ).mul( w01 ) )
		.add( vectorLaplacian2( data, gridSpacing, i1c, j1c, shape ).mul( w11 ) );

}
