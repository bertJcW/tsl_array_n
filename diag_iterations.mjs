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
// Both scalar paths, because they turned out not to agree on what convergence
// means: the host path accepted relativeTolerance and tested the absolute
// residual against it until 2026-09-26. Running the same frozen system down
// both is the A/B that says whether a floor belongs to the system or to a stop
// test, and it costs one more arm.
const PATHS = [ true, false ];

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

const out = await page.evaluate( async ( [ budgets, preconditioners, paths ] ) => {

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

	for ( const gpuResidentScalars of paths ) for ( const preconditioner of preconditioners ) {

		for ( const maxIterations of budgets ) {

			g.dataU.fromArray( u0 );
			g.dataV.fromArray( v0 );
			if ( w0 ) g.dataW.fromArray( w0 );
			ps.pressure.data.fromArray( p0 );

			ps.settings.maxIterations = maxIterations;
			ps.settings.preconditioner = preconditioner;
			ps.settings.gpuResidentScalars = gpuResidentScalars;
			await dispatch();

			if ( bNorm === 0 ) {

				const b = await ps.b.toArray();
				let s = 0;
				for ( const x of b ) s += x * x;
				bNorm = Math.sqrt( s );

			}

			const d = ps.diagnostics;
			results.push( {
				preconditioner, maxIterations, path: gpuResidentScalars ? 'gpu' : 'host',
				converged: d.converged, iterations: d.iterations,
				residual: d.residual,
				relative: bNorm > 0 ? d.residual / bNorm : null,
				stoppedBy: d.stoppedBy
			} );

		}

	}

	// *** WHERE the unsolved part is ***
	//
	// The residual left after a projection IS the divergence the projection
	// failed to remove, so it can be read off the corrected velocity field
	// without the solver exposing r. One more solve at the shipped budget, then
	// a per-row profile: a V-cycle that has stopped helping has stopped helping
	// SOMEWHERE, and a coarse grid's blind spot is a place, not a scalar.
	let divProfile = null;

	{
		g.dataU.fromArray( u0 );
		g.dataV.fromArray( v0 );
		if ( w0 ) g.dataW.fromArray( w0 );
		ps.pressure.data.fromArray( p0 );
		ps.settings.maxIterations = 60;
		ps.settings.preconditioner = 'multigrid';
		ps.settings.gpuResidentScalars = true;
		await dispatch();

		const [ NUx, NUy ] = g.dataSizeU;
		const [ NVx, NVy ] = g.dataSizeV;
		const NX = NUx - 1, NY = NUy;
		const u = await g.dataU.toArray();
		const v = await g.dataV.toArray();
		const U = ( i, j ) => u[ i + NUx * j ];
		const V = ( i, j ) => v[ i + NVx * j ];

		// *** Pinned cells are excluded, and the first version of this did not
		// exclude them ***
		//
		// A vent is a region of pinned pressure: the solver does not make those
		// cells divergence-free and never promised to. Measured without the
		// exclusion, this reported the entire residual as living at j=126 --
		// the first of example 17's two pinned rows -- at 1.01e+1 against
		// ~2e-5 everywhere else, which reads as a dramatic localisation and is
		// just the vent doing what a vent does. Same mistake as the one
		// solver_health.mjs has a long comment about, made again in a fresh
		// tool on the same day.
		const mask = ps.dirichletMask ? await ps.dirichletMask.toArray() : null;
		const pinnedAt = mask ? ( i, j ) => mask[ i + NX * j ] > 0.5 : () => false;

		const rows = new Array( NY ).fill( 0 );
		const cells = [];
		let excluded = 0;
		for ( let j = 0; j < NY; j ++ ) for ( let i = 0; i < NX; i ++ ) {

			if ( pinnedAt( i, j ) ) { excluded ++; continue; }

			const d = Math.abs( U( i + 1, j ) - U( i, j ) + V( i, j + 1 ) - V( i, j ) );
			if ( d > rows[ j ] ) rows[ j ] = d;
			cells.push( { i, j, d } );

		}
		cells.sort( ( a, b ) => b.d - a.d );
		divProfile = { NX, NY, rows, excluded, top: cells.slice( 0, 6 ) };
	}

	Object.assign( ps.settings, saved );

	return { dims, bNorm, tolerance: saved.tolerance, relativeTolerance: saved.relativeTolerance, results, divProfile };

}, [ BUDGETS, PRECONDITIONERS, PATHS ] );

await browser.close();

const demanded = out.relativeTolerance ? out.tolerance * out.bNorm : out.tolerance;
console.log( `\n${ URL }` );
console.log( `frozen at frame ${ AT }; ${ out.dims }D; |b| = ${ out.bNorm.toExponential( 3 ) }; tolerance ${ out.tolerance } (${ out.relativeTolerance ? 'relative' : 'absolute' })` );
console.log( `the stop test therefore demands residual < ${ demanded.toExponential( 3 ) }\n` );

console.log( 'path   precond      budget   used   converged   residual    residual/|b|   stopped by' );
for ( const r of out.results ) {

	console.log(
		`${ r.path.padEnd( 4 ) }   ${ r.preconditioner.padEnd( 10 ) }   ${ String( r.maxIterations ).padStart( 6 ) }   ${ String( r.iterations ).padStart( 4 ) }   ` +
		`${ String( r.converged ).padStart( 9 ) }   ${ r.residual.toExponential( 2 ).padStart( 8 ) }    ` +
		`${ r.relative.toExponential( 2 ).padStart( 9 ) }   ${ r.stoppedBy }`
	);

}

if ( out.divProfile ) {

	const { NX, NY, rows, excluded, top } = out.divProfile;
	console.log( `
where the divergence the projection could not remove ends up (multigrid, 60 iterations, ${ NX }x${ NY }, ${ excluded } pinned cells excluded):` );
	console.log( `  worst cells: ${ top.map( ( c ) => `(${ c.i },${ c.j }) ${ c.d.toExponential( 2 ) }` ).join( ', ' ) }` );

	// One line per band of rows, so a concentration near the vent or the source
	// is visible without printing 128 numbers.
	const bands = 8, per = Math.ceil( NY / bands );
	const band = [];
	for ( let b = 0; b < bands; b ++ ) {

		let worst = 0;
		for ( let j = b * per; j < Math.min( NY, ( b + 1 ) * per ); j ++ ) worst = Math.max( worst, rows[ j ] );
		band.push( `j ${ String( b * per ).padStart( 3 ) }-${ String( Math.min( NY, ( b + 1 ) * per ) - 1 ).padStart( 3 ) }: ${ worst.toExponential( 2 ) }` );

	}
	for ( const line of band ) console.log( '  ' + line );

}

console.log( '' );
for ( const path of [ 'gpu', 'host' ] ) for ( const preconditioner of PRECONDITIONERS ) {

	const arms = out.results.filter( ( r ) => r.preconditioner === preconditioner && r.path === path );
	const first = arms[ 0 ], last = arms[ arms.length - 1 ];
	const gain = first.relative / last.relative;
	const verdict = gain > 3
		? 'SLOW: the residual keeps falling, so the budget is what binds'
		: ( gain < 0.5 ? 'DIVERGING: more iterations make the residual WORSE' : 'STALLED: more iterations buy nothing' );
	console.log( `${ path.padEnd( 4 ) } ${ preconditioner.padEnd( 10 ) } ${ first.maxIterations } -> ${ last.maxIterations }: ${ gain.toExponential( 2 ) }x  => ${ verdict }` );

}
