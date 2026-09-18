// Temporary diagnostic: let the scene run, then say WHERE the field is
// largest -- which x slice, which cell, which component -- so that "it
// blows up" can be attached to the inflow, the obstacle, the outflow or
// the walls rather than to the scene as a whole.
import { chromium } from 'playwright-core';

const URL = 'http://localhost:5200/examples/35-karman-vortex-street-3d/';
const SECONDS = Number( process.argv[ 2 ] ?? 30 );

const browser = await chromium.launch( { headless: true, executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
	args: [ '--headless=new', '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=d3d11', '--ignore-gpu-blocklist', '--disable-gpu-sandbox', '--disable-background-timer-throttling', '--disable-backgrounding-occluded-windows', '--disable-renderer-backgrounding', '--disable-features=CalculateNativeWinOcclusion' ] } );
const page = await browser.newPage();
page.on( 'pageerror', ( e ) => console.log( '[pageerror]', e.message ) );
await page.goto( URL, { waitUntil: 'load' } );
await page.waitForFunction( () => window.__fluxflowProbe, undefined, { timeout: 60000 } );
await new Promise( ( r ) => setTimeout( r, SECONDS * 1000 ) );

const out = await page.evaluate( async () => {

	const probe = window.__fluxflowProbe;
	probe.stop();
	await new Promise( ( r ) => setTimeout( r, 200 ) );

	const g = probe.velocityGrid;
	const [ NUx, NUy, NUz ] = g.dataSizeU, [ NVx, NVy, NVz ] = g.dataSizeV, [ NWx, NWy, NWz ] = g.dataSizeW;
	const u = await g.dataU.toArray(), v = await g.dataV.toArray(), w = await g.dataW.toArray();
	const p = await probe.solver.pressure.data.toArray();

	function scan( arr, sx, sy, sz, label ) {

		let max = 0, at = null;
		const slice = new Array( sx ).fill( 0 );
		for ( let k = 0; k < sz; k ++ ) for ( let j = 0; j < sy; j ++ ) for ( let i = 0; i < sx; i ++ ) {

			const a = Math.abs( arr[ i + sx * j + sx * sy * k ] );
			if ( a > slice[ i ] ) slice[ i ] = a;
			if ( a > max ) { max = a; at = [ i, j, k ]; }

		}
		return { label, max, at, slice };

	}

	return {
		u: scan( u, NUx, NUy, NUz, 'u' ),
		v: scan( v, NVx, NVy, NVz, 'v' ),
		w: scan( w, NWx, NWy, NWz, 'w' ),
		p: scan( p, NUx - 1, NUy, NUz, 'p' ),
		frame: probe.frame
	};

} );

console.log( `after ${ out.frame } frames` );
for ( const c of [ out.u, out.v, out.w, out.p ] ) console.log( `  max |${ c.label }| = ${ c.max.toFixed( 2 ) } at ${ c.at }` );
console.log( 'per-x-slice max |u| / |v| / |w| / |p|:' );
for ( let i = 0; i < out.u.slice.length; i ++ ) {

	console.log( `  i=${ String( i ).padStart( 2 ) }  ${ out.u.slice[ i ].toFixed( 1 ).padStart( 7 ) } ${ ( out.v.slice[ i ] ?? 0 ).toFixed( 1 ).padStart( 7 ) } ${ ( out.w.slice[ i ] ?? 0 ).toFixed( 1 ).padStart( 7 ) } ${ ( out.p.slice[ i ] ?? 0 ).toFixed( 1 ).padStart( 9 ) }` );

}

await browser.close();
