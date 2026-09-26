// Freeze a 2D scene at one frame and write its pressure system to disk, so the
// same system can be solved somewhere the GPU is not involved.
//
// The 3D counterpart is export_system.mjs, written for
// sandbox/stalled-system/. This one exists for a narrower question: the 2D
// smoke scenes floor at a relative residual near 1.4e-5 with the multigrid
// preconditioner, on both scalar paths, while the same shape and mask built
// clean in sandbox/vcycle-floor/ reaches 5.58e-7 in sixteen iterations. Four
// candidates are already eliminated (the stop test, the operator, the shape and
// mask, and any geometric blind spot), which leaves the scene's own
// right-hand side -- and to put THAT on trial it has to leave the scene.
//
// What is written: b, the Dirichlet mask, the pressure the frame arrived with,
// the shape and spacing, and what the GPU solver itself reported, so a CPU run
// has something to be compared against rather than admired alone.
//
// usage: node export_system2.mjs [url] [frame] [outDir]

import { chromium } from 'playwright-core';
import { mkdirSync, writeFileSync } from 'node:fs';

const URL = process.argv[ 2 ] ?? 'http://localhost:5200/examples/17-smoke-fire/';
const AT = Number( process.argv[ 3 ] ?? 400 );
const OUT = process.argv[ 4 ] ?? 'system-2d';

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
	const g = probe.velocityGrid;

	// Re-solve once at the shipped budget so the b and the diagnostics on disk
	// describe the same solve, rather than b from this frame and numbers from
	// whatever the last one happened to do.
	const u0 = await g.dataU.toArray();
	const v0 = await g.dataV.toArray();
	const p0 = await ps.pressure.data.toArray();
	const dispatch = ps.project( g, g );

	g.dataU.fromArray( u0 );
	g.dataV.fromArray( v0 );
	ps.pressure.data.fromArray( p0 );
	await dispatch();

	const d = ps.diagnostics;
	const b = await ps.b.toArray();
	const mask = ps.dirichletMask ? await ps.dirichletMask.toArray() : null;
	const pressure = await ps.pressure.data.toArray();

	return {
		shape: ps.dirichletMask ? ps.dirichletMask.shape : null,
		// gridSpacing on a face-centred grid is a TSL node, not a number, and
		// JSON.stringify walks it into its own graph. Only the numeric value
		// matters here, and every 2D scene in this package builds its grid with
		// unit spacing, so read it if it is readable and record 1 otherwise --
		// stated rather than assumed, since the CPU operator divides by it.
		gridSpacing: [ Number( g.gridSpacing?.x?.value ?? g.gridSpacing?.x ) || 1, Number( g.gridSpacing?.y?.value ?? g.gridSpacing?.y ) || 1 ],
		hasWeights: Boolean( inner.colliderFaceWeights ),
		settings: {
			tolerance: ps.settings.tolerance,
			relativeTolerance: ps.settings.relativeTolerance,
			maxIterations: ps.settings.maxIterations,
			preconditioner: ps.settings.preconditioner,
			levels: ps.settings.multigrid?.numberOfLevels ?? null
		},
		gpu: { converged: d.converged, iterations: d.iterations, residual: d.residual, stoppedBy: d.stoppedBy },
		b: Array.from( b ),
		mask: mask ? Array.from( mask ) : null,
		pressureBefore: Array.from( p0 ),
		pressureAfter: Array.from( pressure )
	};

} );

await browser.close();

mkdirSync( OUT, { recursive: true } );

const f32 = ( name, values ) => writeFileSync( `${ OUT }/${ name }.bin`, Buffer.from( Float32Array.from( values ).buffer ) );
f32( 'b', out.b );
if ( out.mask ) f32( 'dirichletMask', out.mask );
f32( 'pressureBefore', out.pressureBefore );
f32( 'pressureAfter', out.pressureAfter );

const bNorm = Math.sqrt( out.b.reduce( ( a, x ) => a + x * x, 0 ) );
let pinned = 0;
if ( out.mask ) for ( const v of out.mask ) if ( v > 0.5 ) pinned ++;

const meta = {
	url: URL, frame: AT, shape: out.shape, gridSpacing: out.gridSpacing,
	hasWeights: out.hasWeights, settings: out.settings, gpu: out.gpu,
	bNorm, pinned, cells: out.b.length
};
writeFileSync( `${ OUT }/meta.json`, JSON.stringify( meta, null, '\t' ) );

console.log( `\n${ OUT }/ written` );
console.log( `  ${ out.shape?.join( 'x' ) ?? '?' }, ${ out.b.length } cells, ${ pinned } pinned, |b| = ${ bNorm.toExponential( 4 ) }` );
console.log( `  collider face weights: ${ out.hasWeights ? 'yes -- the CPU reference must apply them too' : 'no -- a constant-coefficient operator is the whole of it' }` );
console.log( `  solver settings: ${ JSON.stringify( out.settings ) }` );
console.log( `  what the GPU reported for this solve: ${ JSON.stringify( out.gpu ) }, residual/|b| = ${ ( out.gpu.residual / bNorm ).toExponential( 2 ) }` );
