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
	let bProfile = null;
	const results = [];
	let worst = null;

	const fast = { optimisticStopTest: true, gpuStopTest: true, fuseVcycleIntoIteration: true, residualCheckInterval: 4, gpuResidentScalars: true, gpuResidentSetup: true, batchIterations: true };
	const plain = { optimisticStopTest: false, gpuStopTest: false, fuseVcycleIntoIteration: false, residualCheckInterval: 1, gpuResidentScalars: false, gpuResidentSetup: false, batchIterations: false };

	for ( const [ preconditioner, maxIterations, mode ] of [
		[ 'multigrid', 100, fast ], [ 'multigrid', 3000, fast ],
		[ 'multigrid', 100, plain ], [ 'multigrid', 3000, plain ],
		[ 'multigrid', 3000, { ...fast, fuseVcycleIntoIteration: false } ],
		[ 'multigrid', 3000, { ...fast, gpuStopTest: false, optimisticStopTest: false } ],
		[ 'jacobi', 3000, fast ], [ 'none', 3000, fast ]
	] ) {

		g.dataU.fromArray( u0 ); g.dataV.fromArray( v0 ); g.dataW.fromArray( w0 );
		solver.pressure.data.fromArray( p0 );
		Object.assign( ps.settings, mode );
		ps.settings.preconditioner = preconditioner;
		ps.settings.maxIterations = maxIterations;

		await dispatch();

		if ( bNorm === 0 ) {

			const b = await ps.b.toArray();
			let s = 0;
			for ( const x of b ) s += x * x;
			bNorm = Math.sqrt( s );

			// WHERE the right-hand side is large says which stage put the
			// divergence there: the inflow band, the collider's own faces,
			// the vent, or the interior.
			const NX = 48, NY = 24, NZ = 24;
			const slice = new Array( NX ).fill( 0 );
			const cells = [];
			for ( let k = 0; k < NZ; k ++ ) for ( let j = 0; j < NY; j ++ ) for ( let i = 0; i < NX; i ++ ) {

				const value = Math.abs( b[ i + NX * j + NX * NY * k ] );
				if ( value > slice[ i ] ) slice[ i ] = value;
				cells.push( { i, j, k, b: b[ i + NX * j + NX * NY * k ] } );

			}
			cells.sort( ( a, c ) => Math.abs( c.b ) - Math.abs( a.b ) );
			bProfile = { slice, top: cells.slice( 0, 10 ) };

		}

		const d = ps.diagnostics;
		results.push( { preconditioner, maxIterations, mode: mode === fast ? 'fast' : mode === plain ? 'all off' : ( mode.fuseVcycleIntoIteration === false ? 'no fused V-cycle' : 'no gpu stop test' ), converged: d.converged, iterations: d.iterations, residual: d.residual, stoppedBy: d.stoppedBy, rejected: d.rejected } );

		// After the best-equipped arm, find out WHERE the unsolved part is:
		// the divergence left in the corrected velocity is the residual.
		if ( preconditioner === 'multigrid' && maxIterations === 3000 ) {

			const [ NUx, NUy, NUz ] = g.dataSizeU, [ NVx, NVy ] = g.dataSizeV, [ NWx, NWy ] = g.dataSizeW;
			const u = await g.dataU.toArray(), v = await g.dataV.toArray(), w = await g.dataW.toArray();
			const p = await solver.pressure.data.toArray();
			const b = await ps.b.toArray();
			const NX = NUx - 1;
			const U = ( i, j, k ) => u[ i + NUx * j + NUx * NUy * k ];
			const V = ( i, j, k ) => v[ i + NVx * j + NVx * NVy * k ];
			const W = ( i, j, k ) => w[ i + NWx * j + NWx * NWy * k ];

			const cells = [];
			for ( let k = 0; k < NUz; k ++ ) for ( let j = 0; j < NUy; j ++ ) for ( let i = 0; i < NX; i ++ ) {

				const div = ( U( i + 1, j, k ) - U( i, j, k ) ) + ( V( i, j + 1, k ) - V( i, j, k ) ) + ( W( i, j, k + 1 ) - W( i, j, k ) );
				cells.push( { i, j, k, div, b: b[ i + NX * j + NX * NUy * k ], p: p[ i + NX * j + NX * NUy * k ] } );

			}

			cells.sort( ( a, c ) => Math.abs( c.div ) - Math.abs( a.div ) );
			let l2 = 0;
			for ( const c of cells ) l2 += c.div * c.div;
			worst = { top: cells.slice( 0, 12 ), l2: Math.sqrt( l2 ) };

		}

	}

	return { bNorm, bProfile, results, worst };

} );

console.log( `frozen at frame ${ AT }, ||b|| = ${ out.bNorm.toExponential( 3 ) }` );
for ( const r of out.results ) console.log(
	`  ${ r.preconditioner.padEnd( 10 ) } ${ String( r.mode ).padEnd( 17 ) } cap ${ String( r.maxIterations ).padStart( 5 ) }  converged=${ String( r.converged ).padEnd( 5 ) } ` +
	`iters ${ String( r.iterations ).padStart( 5 ) }  residual ${ r.residual?.toExponential( 2 ) }  rel ${ ( r.residual / out.bNorm ).toExponential( 2 ) }  stoppedBy ${ r.stoppedBy }`
);

if ( out.bProfile ) {

	console.log( '' );
	console.log( 'max |b| per x slice: ' + out.bProfile.slice.map( ( v, i ) => `${ i }:${ v.toFixed( 1 ) }` ).join( ' ' ) );
	console.log( 'largest |b| cells: ' + out.bProfile.top.map( ( c ) => `(${ c.i },${ c.j },${ c.k })=${ c.b.toFixed( 2 ) }` ).join( '  ' ) );

}

if ( out.worst ) {

	console.log( '' );
	console.log( `after MGPCG x3000, divergence left in the corrected velocity: L2 ${ out.worst.l2.toExponential( 3 ) }` );
	console.log( '  worst cells (i,j,k)        div            b            p' );
	for ( const c of out.worst.top ) console.log(
		`  (${ String( c.i ).padStart( 2 ) },${ String( c.j ).padStart( 2 ) },${ String( c.k ).padStart( 2 ) })  ` +
		`${ c.div.toExponential( 3 ).padStart( 12 ) } ${ c.b.toExponential( 3 ).padStart( 12 ) } ${ c.p.toExponential( 3 ).padStart( 12 ) }` );

}

await browser.close();
