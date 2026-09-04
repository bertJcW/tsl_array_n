// 纯 CPU 侧几何计算，不碰任何 TSL 节点——给 sdf_collider2.js 的 addPolygon()/addSvg()
// 用，把多边形顶点数组栅格化成 SDF 网格数据。
//
// 对应源码 sdf_collider2.py 里对 shapely 的用法（boundary.distance + contains/touches
// 判内外、unary_union 合并多个形状），但不依赖 shapely 或任何 JS 几何库——都是很成熟
// 的小算法，手写更省事：
//   - 点在多边形内：标准射线法（point-in-polygon by ray casting）
//   - 点到多边形边界的最短距离：逐条边求点到线段距离，取最小值——直接对应
//     shapely 的 boundary.distance()（是到边界折线的距离，不是到"面"的距离，
//     这也是为什么 shapely 版本要用 boundary.distance 而不是 geom.distance）
//   - 多个多边形的"并集"SDF：逐点取 min，不需要真正的多边形布尔运算——SDF 并集
//     在数学上就是 pointwise min，图形学标准做法，比手写多边形布尔运算简单可靠
//
// 多边形用普通顶点数组表示：[[x0,y0],[x1,y1],...]（隐式闭合，最后一个点和第一个点
// 之间自动算一条边，不需要重复给一次首点）。

function pointInPolygon( x, y, ring ) {

	let inside = false;

	for ( let i = 0, j = ring.length - 1; i < ring.length; j = i ++ ) {

		const [ xi, yi ] = ring[ i ];
		const [ xj, yj ] = ring[ j ];

		const intersect = ( ( yi > y ) !== ( yj > y ) ) &&
			( x < ( xj - xi ) * ( y - yi ) / ( yj - yi ) + xi );

		if ( intersect ) inside = ! inside;

	}

	return inside;

}

function pointToSegmentDistance( x, y, x1, y1, x2, y2 ) {

	const dx = x2 - x1;
	const dy = y2 - y1;
	const lengthSq = dx * dx + dy * dy;

	let t = lengthSq === 0 ? 0 : ( ( x - x1 ) * dx + ( y - y1 ) * dy ) / lengthSq;
	t = Math.max( 0, Math.min( 1, t ) );

	const px = x1 + t * dx;
	const py = y1 + t * dy;

	return Math.hypot( x - px, y - py );

}

function pointToRingBoundaryDistance( x, y, ring ) {

	let minDist = Infinity;

	for ( let i = 0, j = ring.length - 1; i < ring.length; j = i ++ ) {

		const d = pointToSegmentDistance( x, y, ring[ j ][ 0 ], ring[ j ][ 1 ], ring[ i ][ 0 ], ring[ i ][ 1 ] );
		if ( d < minDist ) minDist = d;

	}

	return minDist;

}

// 单个多边形在 (x,y) 处的有符号距离：内部为负，外部为正（跟源码 host_sdf 的符号约定一致）
export function polygonSignedDistance( x, y, ring ) {

	const dist = pointToRingBoundaryDistance( x, y, ring );
	return pointInPolygon( x, y, ring ) ? - dist : dist;

}

// 多个多边形的并集 SDF：逐点取 min
export function polygonsSignedDistance( x, y, rings ) {

	let minDist = Infinity;

	for ( const ring of rings ) {

		const d = polygonSignedDistance( x, y, ring );
		if ( d < minDist ) minDist = d;

	}

	return minDist;

}

// 多边形的面积加权质心（标准公式，不是顶点坐标平均——跟 shapely .centroid 的定义
// 一致，SDFRigidBodyCollider2 用它当旋转轴心）
export function polygonCentroid( ring ) {

	let area = 0, cx = 0, cy = 0;

	for ( let i = 0, j = ring.length - 1; i < ring.length; j = i ++ ) {

		const [ xi, yi ] = ring[ i ];
		const [ xj, yj ] = ring[ j ];

		const cross = xj * yi - xi * yj;
		area += cross;
		cx += ( xj + xi ) * cross;
		cy += ( yj + yi ) * cross;

	}

	area *= 0.5;
	const scale = 1 / ( 6 * area );

	return [ cx * scale, cy * scale ];

}

export function translatePolygon( ring, dx, dy ) {

	return ring.map( ( [ x, y ] ) => [ x + dx, y + dy ] );

}

export function rotatePolygon( ring, angle, origin ) {

	const cos = Math.cos( angle );
	const sin = Math.sin( angle );
	const [ ox, oy ] = origin;

	return ring.map( ( [ x, y ] ) => {

		const dx = x - ox;
		const dy = y - oy;

		return [ ox + dx * cos - dy * sin, oy + dx * sin + dy * cos ];

	} );

}
