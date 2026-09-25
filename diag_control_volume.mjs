// Where is the right surface to measure conservation on? mantaflow excludes
// outflow cells from the pressure correction, so the velocities there are
// by design not constrained by anything -- which makes the domain's own
// outermost faces the wrong control surface. This compares the flux across
// the fluid/vent interface against the flux across the domain boundary.
import { chromium } from 'playwright-core';

const URL = process.argv[ 2 ];
const SECONDS = Number( process.argv[ 3 ] ?? 40 );

const browser = await chromium.launch( { headless: true, executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
	args: [ '--headless=new', '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=d3d11', '--ignore-gpu-blocklist', '--disable-gpu-sandbox', '--disable-background-timer-throttling', '--disable-backgrounding-occluded-windows', '--disable-renderer-backgrounding', '--disable-features=CalculateNativeWinOcclusion' ] } );
const page = await browser.newPage();
page.on( 'pageerror', ( e ) => console.log( '[pageerror]', e.message ) );
await page.goto( URL, { waitUntil: 'load' } );
await page.waitForFunction( () => window.__fluxflowProbe !== undefined, undefined, { timeout: 60000 } );
await new Promise( ( r ) => setTimeout( r, SECONDS * 1000 ) );

const out = await page.evaluate( async () => {

	const probe = window.__fluxflowProbe;
	const solver = probe.solver;
	const inner = solver.pressureSolver ? solver : ( solver.solver ?? solver );
	const g = probe.velocityGrid;

	const [ NUx, NUy ] = g.dataSizeU;
	const [ NVx, NVy ] = g.dataSizeV;
	const NX = NUx - 1, NY = NUy;

	const u = await g.dataU.toArray();
	const v = await g.dataV.toArray();
	const vent = inner.pressureSolver.ventMask ? await inner.pressureSolver.ventMask.toArray() : null;

	const U = ( i, j ) => u[ i + NUx * j ];
	const V = ( i, j ) => v[ i + NVx * j ];
	const VENT = ( i, j ) => ( vent ? vent[ i + NX * j ] > 0.5 : false );

	// the fluid region: cells that are not vent cells
	let firstVentColumn = NX, firstVentRow = NY;
	for ( let j = 0; j < NY; j ++ ) for ( let i = 0; i < NX; i ++ ) {

		if ( ! VENT( i, j ) ) continue;
		if ( i < firstVentColumn ) firstVentColumn = i;
		if ( j < firstVentRow ) firstVentRow = j;

	}

	const columnFlux = ( i ) => { let q = 0; for ( let j = 0; j < NUy; j ++ ) q += U( i, j ); return q; };
	const rowFlux = ( j ) => { let q = 0; for ( let i = 0; i < NVx; i ++ ) q += V( i, j ); return q; };

	return {
		hasVent: Boolean( vent ),
		firstVentColumn: firstVentColumn < NX ? firstVentColumn : null,
		firstVentRow: firstVentRow < NY ? firstVentRow : null,
		inletColumn: columnFlux( 0 ),
		domainRightColumn: columnFlux( NX ),
		fluidRightColumn: firstVentColumn < NX ? columnFlux( firstVentColumn ) : null,
		domainTopRow: rowFlux( NY ),
		fluidTopRow: firstVentRow < NY ? rowFlux( firstVentRow ) : null,
		domainBottomRow: rowFlux( 0 ),
		// the rows around a top vent, whose fluid-facing edge these scenes
		// put at y = NY-2
		rowNYminus3: rowFlux( NY - 3 ),
		rowNYminus2: rowFlux( NY - 2 ),
		rowNYminus1: rowFlux( NY - 1 ),
		rowNY: rowFlux( NY ),
		NY, NX,
		colInlet: columnFlux( 0 ),
		colQuarter: columnFlux( Math.floor( NX * 0.25 ) ),
		colHalf: columnFlux( Math.floor( NX * 0.5 ) ),
		colNXminus3: columnFlux( NX - 3 ),
		colNXminus2: columnFlux( NX - 2 ),
		colNXminus1: columnFlux( NX - 1 ),
		colNX: columnFlux( NX )
	};

} );

await browser.close();
console.log( JSON.stringify( out, null, '  ' ) );
