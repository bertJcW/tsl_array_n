// Temporary diagnostic: record every frame's pressure-solve diagnostics
// from frame 0, by intercepting the probe's assignment before the scene's
// first rAF runs. Polling from outside cannot do this -- the scene runs at
// several hundred frames a second when nothing reads back from it, so by
// the time a poll lands it is already thousands of frames in.
import { chromium } from 'playwright-core';

const URL = process.argv[ 2 ] ?? 'http://localhost:5190/examples/35-karman-vortex-street-3d/';
const SECONDS = Number( process.argv[ 3 ] ?? 20 );
const SAFE = process.argv[ 4 ] === 'safe';

const browser = await chromium.launch( { headless: true, executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
	args: [ '--headless=new', '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=d3d11', '--ignore-gpu-blocklist', '--disable-gpu-sandbox', '--disable-background-timer-throttling', '--disable-backgrounding-occluded-windows', '--disable-renderer-backgrounding', '--disable-features=CalculateNativeWinOcclusion' ] } );
const page = await browser.newPage();
page.on( 'pageerror', ( e ) => console.log( '[pageerror]', e.message ) );

await page.addInitScript( ( safe ) => {

	let stored;
	let counter = 0;
	window.__history = [];

	Object.defineProperty( window, '__fluxflowProbe', {
		configurable: true,
		get: () => stored,
		set: ( probe ) => {

			stored = probe;
			const solver = probe.solver;

			if ( safe ) Object.assign( solver.pressureSolver.settings, {
				optimisticStopTest: false, gpuStopTest: false, fuseVcycleIntoIteration: false,
				residualCheckInterval: 1, gpuResidentScalars: false, batchIterations: false, gpuResidentSetup: false
			} );

			const original = solver.onAdvanceTimeStep;

			solver.onAdvanceTimeStep = async ( dt ) => {

				await original( dt );
				const d = solver.pressureSolver.diagnostics;
				if ( window.__history.length < 20000 ) window.__history.push( [ counter ++, d.converged === true ? 1 : 0, d.iterations, d.residual, d.rejected ? 1 : 0, d.stoppedBy ] );

			};

		}
	} );

}, SAFE );

await page.goto( URL, { waitUntil: 'load' } );
await new Promise( ( r ) => setTimeout( r, SECONDS * 1000 ) );

const out = await page.evaluate( async () => {

	const probe = window.__fluxflowProbe;
	if ( probe.stop ) probe.stop();
	const u = await probe.velocityGrid.dataU.toArray();
	let maxU = 0;
	for ( const x of u ) { const a = Math.abs( x ); if ( a > maxU ) maxU = a; }
	return { history: window.__history, frames: window.__history.length, maxU };

} );

console.log( `recorded ${ out.history.length } frames (probe at frame ${ out.frames }), final maxU=${ out.maxU.toFixed( 2 ) }` );

let firstBad = out.history.findIndex( ( h ) => h[ 1 ] === 0 );
console.log( `first non-converged frame: index ${ firstBad }` );

console.log( 'converged map (1 = converged), 100 frames per line:' );
for ( let i = 0; i < out.history.length; i += 100 ) console.log( '  ' + out.history.slice( i, i + 100 ).map( ( h ) => h[ 1 ] ).join( '' ) );

const from = Math.max( 0, firstBad - 12 );
for ( const h of out.history.slice( from, firstBad + 14 ) ) {

	console.log( `  frame=${ String( h[ 0 ] ).padStart( 5 ) } conv=${ h[ 1 ] } iters=${ String( h[ 2 ] ).padStart( 4 ) } res=${ h[ 3 ]?.toExponential( 3 ) } rej=${ h[ 4 ] } stoppedBy=${ h[ 5 ] }` );

}

const converged = out.history.filter( ( h ) => h[ 1 ] === 1 ).length;
console.log( `converged ${ converged } / ${ out.history.length }` );

await browser.close();
