// 浏览器专用（需要 document/DOM），不能在 vitest/Node 环境跑，只能靠 live 示例验证。
//
// 源码 addSvg() 用 Python 专用的 svg.path 库解析 <path> 的 d 属性、沿路径采样出
// 顶点。这里改用浏览器原生的 SVGPathElement.getTotalLength()/getPointAtLength()——
// 效果一样（沿弧长等距采样 N 个点），零依赖。
//
// 一个已知细节：getPointAtLength() 需要 <path> 元素处于一个真正跑过布局的 document
// 里才可靠——用 DOMParser().parseFromString() 单独解析出来的游离 document 不一定
// 触发布局（不同浏览器实现不一定一致）。所以这里把解析出来的 SVG 临时挂到当前页面
// 的 DOM 上（视觉上隐藏），读完几何信息立刻摘掉，不会造成可见的副作用或内存泄漏。

export function parseSvgToPolygons( svgString, { samples = 100, scale = 1, offsetX = 0, offsetY = 0 } = {} ) {

	const container = document.createElement( 'div' );
	container.style.position = 'absolute';
	container.style.visibility = 'hidden';
	container.style.pointerEvents = 'none';
	container.innerHTML = svgString;
	document.body.appendChild( container );

	try {

		const pathElements = Array.from( container.querySelectorAll( 'path' ) );

		return pathElements.map( ( pathEl ) => {

			const length = pathEl.getTotalLength();
			const points = [];

			for ( let i = 0; i < samples; i ++ ) {

				const t = samples > 1 ? length * i / ( samples - 1 ) : 0;
				const pt = pathEl.getPointAtLength( t );
				points.push( [ pt.x * scale + offsetX, pt.y * scale + offsetY ] );

			}

			return points;

		} );

	} finally {

		document.body.removeChild( container );

	}

}
