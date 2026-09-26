// Is this scene's pressure solve STALLED, SLOW, or actually DIVERGING?
//
// The three look identical from outside -- `converged: false` and the whole
// iteration budget spent -- and they call for completely different responses.
// examples/17-smoke-fire/, 18 and 19 converge on 0 of 12,001 steps while
// leaving a projection residual of 2e-5 relative to what they were asked to
// remove: fine by the health probe's 1e-2 bar, and nowhere near the 1e-6 their
// own tolerance demands.
//
// What this found on example 17 is the third case: the residual GROWS with the
// iteration count, 4.12e-4 at 60 iterations to 7.45e-1 at 3000. CG cannot do
// that on a symmetric positive-definite system with an SPD preconditioner, so
// the sweep runs every preconditioner the solver offers -- whether it happens
// with all of them or only one is the difference between blaming the operator
// and blaming the V-cycle.
//
// Every arm restores the velocity and pressure the frame arrived with, so all
// of them see the identical system; otherwise each arm inherits the previous
// arm's correction and nothing is comparable.
//
// usage: node diag_iterations.mjs <url> [freezeAtFrame]

import { chromium } from 'playwright-core';

const URL = process.argv[ 2 ] ?? 'http://localhost:5200/examples/17-smoke-fire/';
const AT = Number( process.argv[ 3 ] ?? 400 );
const BUDGETS = [ 60, 300, 3000 ];
const PRECONDITIONERS = [ 'multigrid', 'jacobi', 'none' ];

const browser = await chromium.launch( {
	headless: true,
	executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
	args: [ '--headless=new', '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=d3d11',
		'--ignore-gpu-blocklist', '--disable-gpu-sandbox', '--disable-background-timer-throttling',
		'--disable-backgrounding-occluded-windows', '--disable-renderer-backgrounding',
		'--disable-features=CalculateNativeWinOcclusion' ]
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
				if ( counter ++ >= at ) { window.__frozen = true; if ( probe.stop ) probe.stop(); }

			};

		}
	} );

}, AT );

await page.goto( URL, { waitUntil: 'load' } );
await page.waitForFunction( () => window.__fluxflowProbe !== undefined, undefined, { timeout: 120000 } );
const backend = await page.evaluate( () => window.__fluxflowProbe?.renderer?.backend?.constructor?.name ?? 'unknown' );
console.log( `backend: ${ backend }` );
if ( backend !== 'WebGPUBackend' ) { console.log( 'FATAL: not WebGPU' ); await browser.close(); process.exit( 1 ); }

await page.waitForFunction( () => window.__frozen === true, undefined, { timeout: 600000 } );

const out = await page.evaluate( async ( [ budgets, preconditioners ] ) => {

	const probe = window.__fluxflowProbe;
	await new Promise( ( r ) => setTimeout( r, 300 ) );

	// The smoke and fire solvers own a grid solver rather than being one.
	const solver = probe.solver;
	const inner = solver.pressureSolver ? solver : ( solver.solver ?? solver );
	const ps = inner.pressureSolver;
	const g = probe.velocityGrid;
	const dims = g.dataSizeU.length;

	const u0 = await g.dataU.toArray();
	const v0 = await g.dataV.toArray();
	const w0 = dims === 3 ? await g.dataW.toArray() : null;
	const p0 = await ps.pressure.data.toArray();

	const dispatch = ps.project( g, g );
	const saved = { ...ps.settings };

	const results = [];
	let bNorm = 0;

	for ( const preconditioner of preconditioners ) {

		for ( const maxIterations of budgets ) {

			g.dataU.fromArray( u0 );
			g.dataV.fromArray( v0 );
			if ( w0 ) g.dataW.fromArray( w0 );
			ps.pressure.data.fromArray( p0 );

			ps.settings.maxIterations = maxIterations;
			ps.settings.preconditioner = preconditioner;
			await dispatch();

			if ( bNorm === 0 ) {

				const b = await ps.b.toArray();
				let s = 0;
				for ( const x of b ) s += x * x;
				bNorm = Math.sqrt( s );

			}

			const d = ps.diagnostics;
			results.push( {
				preconditioner, maxIterations,
				converged: d.converged, iterations: d.iterations,
				residual: d.residual,
				relative: bNorm > 0 ? d.residual / bNorm : null,
				stoppedBy: d.stoppedBy
			} );

		}

	}

	Object.assign( ps.settings, saved );

	return { dims, bNorm, tolerance: saved.tolerance, relativeTolerance: saved.relativeTolerance, results };

}, [ BUDGETS, PRECONDITIONERS ] );

await browser.close();

const demanded = out.relativeTolerance ? out.tolerance * out.bNorm : out.tolerance;
console.log( `\n${ URL }` );
console.log( `frozen at frame ${ AT }; ${ out.dims }D; |b| = ${ out.bNorm.toExponential( 3 ) }; tolerance ${ out.tolerance } (${ out.relativeTolerance ? 'relative' : 'absolute' })` );
console.log( `the stop test therefore demands residual < ${ demanded.toExponential( 3 ) }\n` );

console.log( 'precond      budget   used   converged   residual    residual/|b|   stopped by' );
for ( const r of out.results ) {

	console.log(
		`${ r.preconditioner.padEnd( 10 ) }   ${ String( r.maxIterations ).padStart( 6 ) }   ${ String( r.iterations ).padStart( 4 ) }   ` +
		`${ String( r.converged ).padStart( 9 ) }   ${ r.residual.toExponential( 2 ).padStart( 8 ) }    ` +
		`${ r.relative.toExponential( 2 ).padStart( 9 ) }   ${ r.stoppedBy }`
	);

}

console.log( '' );
for ( const preconditioner of PRECONDITIONERS ) {

	const arms = out.results.filter( ( r ) => r.preconditioner === preconditioner );
	const first = arms[ 0 ], last = arms[ arms.length - 1 ];
	const gain = first.relative / last.relative;
	const verdict = gain > 3
		? 'SLOW: the residual keeps falling, so the budget is what binds'
		: ( gain < 0.5 ? 'DIVERGING: more iterations make the residual WORSE' : 'STALLED: more iterations buy nothing' );
	console.log( `${ preconditioner.padEnd( 10 ) } ${ first.maxIterations } -> ${ last.maxIterations }: ${ gain.toExponential( 2 ) }x  => ${ verdict }` );

}
