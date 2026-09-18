import { chromium } from 'playwright-core';

const URL = 'http://localhost:5190/examples/35-karman-vortex-street-3d/';
const DURATION_MS = 10 * 60 * 1000; // 10 minutes, well past the ~4700-frame / ~7 min prior failure point
const POLL_MS = 20000;

const browser = await chromium.launch( {
	headless: true,
	executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
	args: [
		'--headless=new',
		'--enable-unsafe-webgpu',
		'--enable-features=Vulkan',
		'--use-angle=d3d11',
		'--ignore-gpu-blocklist',
		'--disable-gpu-sandbox',
		'--disable-background-timer-throttling',
		'--disable-backgrounding-occluded-windows',
		'--disable-renderer-backgrounding',
		'--disable-features=CalculateNativeWinOcclusion'
	]
} );

const page = await browser.newPage();
page.on( 'console', ( msg ) => {

	const text = msg.text();
	if ( text.includes( 'error' ) || text.includes( 'Error' ) ) console.log( '[console]', text );

} );
page.on( 'pageerror', ( err ) => console.log( '[pageerror]', err.message ) );

await page.goto( URL, { waitUntil: 'load' } );

await page.waitForFunction( () => window.__fluxflowProbe && window.__fluxflowProbe.frame > 5, undefined, { timeout: 30000 } );

const backend = await page.evaluate( () => window.__fluxflowProbe.renderer.backend?.constructor?.name );
console.log( 'backend:', backend );
if ( backend !== 'WebGPUBackend' ) {

	console.log( 'FATAL: not running on WebGPUBackend, aborting -- results would be meaningless (WebGL2 fallback, not real hardware verification).' );
	await browser.close();
	process.exit( 1 );

}

console.log( 'scene started, running for', DURATION_MS / 1000, 's...' );

const start = Date.now();

while ( Date.now() - start < DURATION_MS ) {

	await new Promise( ( r ) => setTimeout( r, POLL_MS ) );

	const report = await page.evaluate( async () => {

		const probe = window.__fluxflowProbe;
		const s = probe.stats();
		const grid = probe.velocityGrid;

		const [ u, v, w ] = await Promise.all( [ grid.dataU.toArray(), grid.dataV.toArray(), grid.dataW.toArray() ] );

		let maxU = 0, maxV = 0, maxW = 0, allFinite = true;
		let sumInflow = 0, sumMid = 0, sumOutflow = 0;

		const NX = grid.dataSizeU[ 0 ], NY = grid.dataSizeU[ 1 ], NZ = grid.dataSizeU[ 2 ];
		const midI = NX >> 1;

		for ( let i = 0; i < NX; i ++ ) {

			for ( let j = 0; j < NY; j ++ ) {

				for ( let k = 0; k < NZ; k ++ ) {

					const idx = i + NX * j + NX * NY * k;
					const val = u[ idx ];
					if ( ! Number.isFinite( val ) ) allFinite = false;
					else maxU = Math.max( maxU, Math.abs( val ) );

					if ( i === 0 ) sumInflow += val;
					if ( i === midI ) sumMid += val;
					if ( i === NX - 1 ) sumOutflow += val;

				}

			}

		}

		for ( const val of v ) { if ( ! Number.isFinite( val ) ) allFinite = false; else maxV = Math.max( maxV, Math.abs( val ) ); }
		for ( const val of w ) { if ( ! Number.isFinite( val ) ) allFinite = false; else maxW = Math.max( maxW, Math.abs( val ) ); }

		return { frame: probe.frame, ...s, maxU, maxV, maxW, allFinite, sumInflow, sumMid, sumOutflow };

	} );

	console.log(
		`frame=${ report.frame } elapsed=${ report.elapsedSeconds.toFixed( 1 ) }s fps=${ report.fps.toFixed( 1 ) } ` +
		`rejected=${ report.rejected } converged=${ report.converged } notConverged=${ report.notConverged } ` +
		`maxU=${ report.maxU.toFixed( 2 ) } maxV=${ report.maxV.toFixed( 2 ) } maxW=${ report.maxW.toFixed( 2 ) } ` +
		`allFinite=${ report.allFinite } inflow=${ report.sumInflow.toFixed( 1 ) } mid=${ report.sumMid.toFixed( 1 ) } outflow=${ report.sumOutflow.toFixed( 1 ) }`
	);

}

await browser.close();
console.log( 'done' );
