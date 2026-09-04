// Browser-only (needs document/DOM), can't run in vitest/Node -- only
// verifiable via a live example.
//
// The source's addSvg() uses the Python-only svg.path library to parse a
// <path>'s d attribute and sample vertices along it. This uses the browser's
// native SVGPathElement.getTotalLength()/getPointAtLength() instead -- same
// effect (N points sampled at even arc-length intervals), zero dependencies.
//
// One known detail: getPointAtLength() is only reliable when the <path>
// element belongs to a document that has actually gone through layout --
// a detached document parsed on its own via DOMParser().parseFromString()
// doesn't necessarily trigger layout (browser implementations aren't
// consistent about this). So the parsed SVG is temporarily attached to the
// current page's DOM here (visually hidden), and removed again as soon as
// its geometry has been read -- no visible side effect or memory leak.

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
