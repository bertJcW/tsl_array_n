// Temporary: load the scene from scratch several times and watch the first
// N frames of each, to tell "stable" from "survives its opening transient
// most of the time". Records every frame's solver diagnostics, and the
// velocity maximum every 20 frames, which is cheap enough not to pace the
// loop the way a per-frame readback does.
import { chromium } from 'playwright-core';

const URL = 'http://localhost:5200/examples/35-karman-vortex-street-3d/';
const RUNS = Number( process.argv[ 2 ] ?? 5 );
const FRAMES = Number( process.argv[ 3 ] ?? 900 );

const browser = await chromium.launch( { headless: true, executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
	args: [ '--headless=new', '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=d3d11', '--ignore-gpu-blocklist', '--disable-gpu-sandbox', '--disable-background-timer-throttling', '--disable-backgrounding-occluded-windows', '--disable-renderer-backgrounding', '--disable-features=CalculateNativeWinOcclusion' ] } );

for ( let run = 0; run < RUNS; run ++ ) {

	const page = await browser.newPage();
	page.on( 'pageerror', ( e ) => console.log( '[pageerror]', e.message ) );

	await page.addInitScript( ( frames ) => {

		let stored, counter = 0;
		window.__trace = [];
		window.__done = false;

		Object.defineProperty( window, '__fluxflowProbe', {
			configurable: true,
			get: () => stored,
			set: ( probe ) => {

				stored = probe;
				const solver = probe.solver;
				const original = solver.onAdvanceTimeStep;

				solver.onAdvanceTimeStep = async ( dt ) => {

					await original( dt );
					const n = counter ++;
					if ( n >= frames ) { window.__done = true; return; }

					const d = solver.pressureSolver.diagnostics;
					const row = { n, rejected: d.rejected ? 1 : 0, stoppedBy: d.stoppedBy, converged: d.converged === true ? 1 : 0 };

					if ( n % 20 === 0 ) {

						const u = await probe.velocityGrid.dataU.toArray();
						let maxU = 0;
						for ( const x of u ) { const a = Math.abs( x ); if ( a > maxU ) maxU = a; }
						row.maxU = maxU;

					}

					window.__trace.push( row );

				};

			}
		} );

	}, FRAMES );

	await page.goto( URL, { waitUntil: 'load' } );
	await page.waitForFunction( () => window.__done === true, undefined, { timeout: 180000 } ).catch( () => {} );

	const trace = await page.evaluate( () => window.__trace );
	await page.close();

	const samples = trace.filter( ( r ) => r.maxU !== undefined );
	const rejected = trace.filter( ( r ) => r.rejected ).map( ( r ) => r.n );
	const broke = trace.filter( ( r ) => r.stoppedBy && r.stoppedBy !== 'none' ).map( ( r ) => r.n );
	const last = samples[ samples.length - 1 ];
	// the clamp this scene's own velocity guard holds at
	const blown = samples.find( ( r ) => r.maxU > 90 );

	console.log(
		`run ${ run }: ${ trace.length } frames, converged ${ trace.filter( ( r ) => r.converged ).length }, ` +
		`rejected ${ rejected.length }${ rejected.length ? ` (first at frame ${ rejected[ 0 ] }, last ${ rejected[ rejected.length - 1 ] })` : '' }, ` +
		`CG breakdowns ${ broke.length }${ broke.length ? ` (first at ${ broke[ 0 ] })` : '' }, ` +
		`final maxU ${ last ? last.maxU.toFixed( 2 ) : '?' }` +
		( blown ? `  *** CLAMPED from frame ${ blown.n } ***` : '' )
	);

	console.log( '  maxU every 20 frames: ' + samples.map( ( r ) => r.maxU.toFixed( 0 ) ).join( ' ' ) );

}

await browser.close();
