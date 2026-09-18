// Temporary: capture the field's per-x profile at chosen frames around the
// frame-535 event, computed in the page so the frames are exact.
import { chromium } from 'playwright-core';

const URL = 'http://localhost:5200/examples/35-karman-vortex-street-3d/';
const AT = ( process.argv[ 2 ] ?? '470,510,525,532,536,540,560,700' ).split( ',' ).map( Number );

const browser = await chromium.launch( { headless: true, executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
	args: [ '--headless=new', '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=d3d11', '--ignore-gpu-blocklist', '--disable-gpu-sandbox', '--disable-background-timer-throttling', '--disable-backgrounding-occluded-windows', '--disable-renderer-backgrounding', '--disable-features=CalculateNativeWinOcclusion' ] } );
const page = await browser.newPage();
page.on( 'pageerror', ( e ) => console.log( '[pageerror]', e.message ) );

await page.addInitScript( ( at ) => {

	let stored, counter = 0;
	const want = new Set( at );
	window.__snaps = [];
	window.__done = false;
	const lastWanted = Math.max( ...at );

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
				if ( n > lastWanted ) { window.__done = true; return; }
				if ( ! want.has( n ) ) return;

				const g = probe.velocityGrid;
				const [ NUx, NUy, NUz ] = g.dataSizeU;
				const u = await g.dataU.toArray();
				const p = await solver.pressure.data.toArray();
				const NX = NUx - 1;

				const uSlice = new Array( NUx ).fill( 0 );
				for ( let k = 0; k < NUz; k ++ ) for ( let j = 0; j < NUy; j ++ ) for ( let i = 0; i < NUx; i ++ ) {
					const a = Math.abs( u[ i + NUx * j + NUx * NUy * k ] );
					if ( a > uSlice[ i ] ) uSlice[ i ] = a;
				}

				const pSlice = new Array( NX ).fill( 0 );
				for ( let k = 0; k < NUz; k ++ ) for ( let j = 0; j < NUy; j ++ ) for ( let i = 0; i < NX; i ++ ) {
					const a = Math.abs( p[ i + NX * j + NX * NUy * k ] );
					if ( a > pSlice[ i ] ) pSlice[ i ] = a;
				}

				const d = solver.pressureSolver.diagnostics;
				window.__snaps.push( { n, uSlice, pSlice, converged: d.converged, iterations: d.iterations, residual: d.residual, stoppedBy: d.stoppedBy, rejected: d.rejected } );

			};

		}
	} );

}, AT );

await page.goto( URL, { waitUntil: 'load' } );
await page.waitForFunction( () => window.__done === true, undefined, { timeout: 180000 } ).catch( () => {} );

const snaps = await page.evaluate( () => window.__snaps );
await browser.close();

for ( const s of snaps ) console.log( `frame ${ String( s.n ).padStart( 4 ) }: converged=${ s.converged } iters=${ String( s.iterations ).padStart( 5 ) } residual=${ s.residual?.toExponential( 2 ) } stoppedBy=${ s.stoppedBy } rejected=${ s.rejected }` );

console.log( '' );
console.log( 'max |u| per x face:' );
console.log( '  x   ' + snaps.map( ( s ) => String( s.n ).padStart( 9 ) ).join( '' ) );
for ( let i = 0; i < snaps[ 0 ].uSlice.length; i ++ ) {

	console.log( '  ' + String( i ).padStart( 2 ) + '  ' + snaps.map( ( s ) => s.uSlice[ i ].toFixed( 2 ).padStart( 9 ) ).join( '' ) );

}

console.log( '' );
console.log( 'max |p| per x cell:' );
console.log( '  x   ' + snaps.map( ( s ) => String( s.n ).padStart( 9 ) ).join( '' ) );
for ( let i = 0; i < snaps[ 0 ].pSlice.length; i ++ ) {

	console.log( '  ' + String( i ).padStart( 2 ) + '  ' + snaps.map( ( s ) => s.pSlice[ i ].toFixed( 2 ).padStart( 9 ) ).join( '' ) );

}
