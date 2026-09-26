// Is this scene's own multigrid preconditioner symmetric?
//
// PCG needs one. Without it the method has no descent guarantee, and the way
// that shows up is unkind: a slightly asymmetric V-cycle is a GOOD
// preconditioner for the first few dozen iterations and then diverges, so from
// outside it reads as "does not converge" -- indistinguishable from converging
// too slowly, and calling for the opposite response.
//
// diag_iterations.mjs found exactly that on examples/17-smoke-fire/: with
// multigrid the residual grows from 4.12e-4 at 60 iterations to 7.45e-1 at
// 3000, while with jacobi or with no preconditioner at all the same frozen
// system converges normally. That clears the operator and accuses the V-cycle,
// and this measures the accusation directly instead of inferring it: form
// (Mx,y) and (x,My) for random x and y on the scene's own system. Symmetry
// means they agree.
//
// It also checks that (Mr,r) keeps one sign across draws, which is the other
// half of what PCG needs. The sign itself is not the point -- this package's
// Poisson operator is the negative Laplacian, so consistently negative is
// correct -- a CHANGE of sign between draws is.
//
// usage: node diag_symmetry.mjs <url> [freezeAtFrame]

import { chromium } from 'playwright-core';

const URL = process.argv[ 2 ] ?? 'http://localhost:5200/examples/17-smoke-fire/';
const AT = Number( process.argv[ 3 ] ?? 400 );

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

const out = await page.evaluate( async () => {

	const probe = window.__fluxflowProbe;
	await new Promise( ( r ) => setTimeout( r, 300 ) );

	const solver = probe.solver;
	const inner = solver.pressureSolver ? solver : ( solver.solver ?? solver );
	const ps = inner.pressureSolver;

	if ( ! ps.buildPreconditioner ) return { error: 'this pressure solver does not expose buildPreconditioner' };

	// No new allocations, and no need for a handle on tsl_array_n from in
	// here: the solver already owns two fields of exactly the right shape.
	// They are saved and put back, so the frozen frame is left as it was and
	// a later arm of the same run still sees the same system.
	const inputField = ps.b;
	const outputField = ps.pressure.data;
	const b0 = await inputField.toArray();
	const p0 = await outputField.toArray();
	const cells = b0.length;
	const shape = ps.dirichletMask?.shape ?? [ cells ];

	let seed = 987654321;
	const rand = () => { seed = ( seed * 1103515245 + 12345 ) & 0x7fffffff; return seed / 0x7fffffff - 0.5; };
	const dot = ( a, b ) => { let s = 0; for ( let i = 0; i < a.length; i ++ ) s += a[ i ] * b[ i ]; return s; };

	const M = ps.buildPreconditioner( inputField, outputField );

	const apply = async ( values ) => {

		inputField.fromArray( Float32Array.from( values ) );
		outputField.fromArray( new Float32Array( cells ) );
		M();
		return Array.from( await outputField.toArray() );

	};

	const x = Array.from( { length: cells }, rand );
	const y = Array.from( { length: cells }, rand );
	const Mx = await apply( x );
	const My = await apply( y );

	const left = dot( Mx, y );
	const right = dot( x, My );
	const relative = Math.abs( left - right ) / Math.max( Math.abs( left ), Math.abs( right ), 1e-30 );

	const quads = [];
	for ( let draw = 0; draw < 5; draw ++ ) {

		const r = Array.from( { length: cells }, rand );
		const Mr = await apply( r );
		quads.push( dot( Mr, r ) );

	}

	let pinned = 0;
	if ( ps.dirichletMask ) { const m = await ps.dirichletMask.toArray(); for ( const v of m ) if ( v > 0.5 ) pinned ++; }

	inputField.fromArray( b0 );
	outputField.fromArray( p0 );

	return {
		shape, cells, pinned,
		levels: ps.settings?.multigrid?.numberOfLevels ?? null,
		left, right, relative, quads
	};

} );

await browser.close();

if ( out.error ) { console.log( `\ncould not measure: ${ out.error }` ); process.exit( 1 ); }

console.log( `\n${ URL }` );
console.log( `frozen at frame ${ AT }; shape ${ out.shape.join( 'x' ) } (${ out.cells } cells); ${ out.pinned } pinned cells; ${ out.levels ?? '?' } multigrid levels\n` );

const symmetric = out.relative < 1e-4;
console.log( `${ symmetric ? '\u2713' : '\u2717' } V-cycle symmetric -- (Mx,y) = ${ out.left.toExponential( 6 ) }, (x,My) = ${ out.right.toExponential( 6 ) }, rel diff ${ out.relative.toExponential( 2 ) }` );

const negative = out.quads.every( ( q ) => q < 0 );
const positive = out.quads.every( ( q ) => q > 0 );
console.log( `${ negative || positive ? '\u2713' : '\u2717' } (Mr,r) keeps one sign -- ${ out.quads.map( ( q ) => q.toExponential( 2 ) ).join( ', ' ) }` );

console.log( symmetric
	? '\nThe preconditioner is symmetric on this system, so PCG diverging here is not its asymmetry.'
	: '\nThe preconditioner is NOT symmetric on this system. PCG has no descent guarantee with it, which is why more iterations make the residual worse.' );

process.exit( symmetric && ( negative || positive ) ? 0 : 2 );
