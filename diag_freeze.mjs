// Temporary: stop the scene at an exact frame, then re-solve that one
// frozen pressure system several ways, restoring velocity and pressure
// between arms so every arm sees the identical system. Distinguishes "this
// frame needs more iterations" from "this frame is not solvable by this
// preconditioner at any iteration count".
import { chromium } from 'playwright-core';

const URL = 'http://localhost:5200/examples/35-karman-vortex-street-3d/';
const AT = Number( process.argv[ 2 ] ?? 524 );

const browser = await chromium.launch( { headless: true, executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
	args: [ '--headless=new', '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=d3d11', '--ignore-gpu-blocklist', '--disable-gpu-sandbox', '--disable-background-timer-throttling', '--disable-backgrounding-occluded-windows', '--disable-renderer-backgrounding', '--disable-features=CalculateNativeWinOcclusion' ] } );
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
				if ( counter ++ >= at ) { window.__frozen = true; probe.stop(); }

			};

		}
	} );

}, AT );

await page.goto( URL, { waitUntil: 'load' } );
await page.waitForFunction( () => window.__frozen === true, undefined, { timeout: 180000 } );

const out = await page.evaluate( async () => {

	const probe = window.__fluxflowProbe;
	await new Promise( ( r ) => setTimeout( r, 300 ) );

	const solver = probe.solver, ps = solver.pressureSolver, g = probe.velocityGrid;
	const u0 = await g.dataU.toArray(), v0 = await g.dataV.toArray(), w0 = await g.dataW.toArray();
	const p0 = await solver.pressure.data.toArray();
	const dispatch = ps.project( g, g );

	let bNorm = 0;
	const results = [];

	for ( const [ preconditioner, maxIterations ] of [
		[ 'multigrid', 100 ], [ 'multigrid', 600 ], [ 'multigrid', 3000 ],
		[ 'jacobi', 3000 ], [ 'none', 3000 ], [ 'none', 20000 ]
	] ) {

		g.dataU.fromArray( u0 ); g.dataV.fromArray( v0 ); g.dataW.fromArray( w0 );
		solver.pressure.data.fromArray( p0 );
		ps.settings.preconditioner = preconditioner;
		ps.settings.maxIterations = maxIterations;

		await dispatch();

		if ( bNorm === 0 ) {

			const b = await ps.b.toArray();
			let s = 0;
			for ( const x of b ) s += x * x;
			bNorm = Math.sqrt( s );

		}

		const d = ps.diagnostics;
		results.push( { preconditioner, maxIterations, converged: d.converged, iterations: d.iterations, residual: d.residual, stoppedBy: d.stoppedBy, rejected: d.rejected } );

	}

	return { bNorm, results };

} );

console.log( `frozen at frame ${ AT }, ||b|| = ${ out.bNorm.toExponential( 3 ) }` );
for ( const r of out.results ) console.log(
	`  ${ r.preconditioner.padEnd( 10 ) } cap ${ String( r.maxIterations ).padStart( 5 ) }  converged=${ String( r.converged ).padEnd( 5 ) } ` +
	`iters ${ String( r.iterations ).padStart( 5 ) }  residual ${ r.residual?.toExponential( 2 ) }  rel ${ ( r.residual / out.bNorm ).toExponential( 2 ) }  stoppedBy ${ r.stoppedBy }`
);

await browser.close();
