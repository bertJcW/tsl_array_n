// Capture one frame's pressure system to disk, so that the thing that goes
// wrong can be re-run in seconds instead of a few hundred frames of
// simulation.
//
// What gets written is everything the system is made of and nothing about
// how it was reached: the grid, the Dirichlet and vent masks, the
// collider's face weights, the right-hand side the solver built, and the
// pressure it produced. Rebuilding the operator from those is
// sandbox/stalled-system/'s job.
//
// usage: node export_system.mjs <url> <frame> [outDir]

import { chromium } from 'playwright-core';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';

const URL = process.argv[ 2 ] ?? 'http://localhost:5200/examples/35-karman-vortex-street-3d/';
const AT = Number( process.argv[ 3 ] ?? 118 );
const OUT = process.argv[ 4 ] ?? 'packages/fluxflow/sandbox/stalled-system/data';

const browser = await chromium.launch( {
	headless: true,
	executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
	args: [
		'--headless=new', '--enable-unsafe-webgpu', '--enable-features=Vulkan',
		'--use-angle=d3d11', '--ignore-gpu-blocklist', '--disable-gpu-sandbox',
		'--disable-background-timer-throttling', '--disable-backgrounding-occluded-windows',
		'--disable-renderer-backgrounding', '--disable-features=CalculateNativeWinOcclusion'
	]
} );

const page = await browser.newPage();
page.on( 'pageerror', ( e ) => console.log( '[pageerror]', e.message ) );

await page.addInitScript( ( at ) => {

	let stored, counter = 0;
	window.__frozen = false;

	Object.defineProperty( window, '__fluxflowProbe', {
		configurable: true,
		get: () => stored,
		set: ( probe ) => {

			stored = probe;
			const solver = probe.solver;
			const original = solver.onAdvanceTimeStep;

			solver.onAdvanceTimeStep = async ( dt ) => {

				if ( window.__frozen ) return;
				await original( dt );
				if ( counter ++ >= at ) { window.__frozen = true; probe.stop && probe.stop(); }

			};

		}
	} );

}, AT );

await page.goto( URL, { waitUntil: 'load' } );
await page.waitForFunction( () => window.__frozen === true, undefined, { timeout: 600000 } );

const captured = await page.evaluate( async () => {

	const probe = window.__fluxflowProbe;
	await new Promise( ( r ) => setTimeout( r, 300 ) );

	const solver = probe.solver;
	const ps = solver.pressureSolver;
	const g = probe.velocityGrid;
	const fw = solver.colliderFaceWeights && solver.colliderFaceWeights.fields;

	const read = async ( field ) => ( field ? Array.from( await field.toArray() ) : null );

	return {
		resolution: g.resolution,
		dataSizeU: g.dataSizeU, dataSizeV: g.dataSizeV, dataSizeW: g.dataSizeW,
		diagnostics: { ...ps.diagnostics },
		settings: { ...ps.settings, multigrid: { ...ps.settings.multigrid } },
		arrays: {
			b: await read( ps.b ),
			pressure: await read( solver.pressure.data ),
			dirichletMask: await read( ps.dirichletMask ),
			ventMask: await read( ps.ventMask ),
			weightU: fw ? await read( fw.u ) : null,
			weightV: fw ? await read( fw.v ) : null,
			weightW: fw ? await read( fw.w ) : null
		}
	};

} );

await browser.close();

await mkdir( OUT, { recursive: true } );

const written = [];
for ( const [ name, values ] of Object.entries( captured.arrays ) ) {

	if ( values === null ) { written.push( `${ name }: absent` ); continue; }
	const file = path.join( OUT, `${ name }.bin` );
	await writeFile( file, Buffer.from( new Float32Array( values ).buffer ) );
	written.push( `${ name }: ${ values.length } floats` );

}

const meta = {
	source: URL,
	frame: AT,
	resolution: captured.resolution,
	gridSpacing: [ 1, 1, 1 ],
	dataSizeU: captured.dataSizeU, dataSizeV: captured.dataSizeV, dataSizeW: captured.dataSizeW,
	diagnostics: captured.diagnostics,
	multigrid: captured.settings.multigrid,
	tolerance: captured.settings.tolerance,
	maxIterations: captured.settings.maxIterations
};

await writeFile( path.join( OUT, 'meta.json' ), JSON.stringify( meta, null, '\t' ) + '\n' );

console.log( `captured frame ${ AT } from ${ URL }` );
console.log( `  the solve there: converged=${ captured.diagnostics.converged } iterations=${ captured.diagnostics.iterations } residual=${ captured.diagnostics.residual } stoppedBy=${ captured.diagnostics.stoppedBy }` );
console.log( '  ' + written.join( '\n  ' ) );
console.log( `  written to ${ OUT }` );
