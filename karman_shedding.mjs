// Is this a vortex street, or a steady wake plus boundary
// reflections? Records transverse velocity at fixed probe points in the
// wake, every Nth frame, and reports whether the signal oscillates -- sign
// changes, period, and the Strouhal number that follows from it.
import { chromium } from 'playwright-core';

const URL = 'http://localhost:5200/examples/35-karman-vortex-street-3d/';
const SECONDS = Number( process.argv[ 2 ] ?? 240 );
const EVERY = Number( process.argv[ 3 ] ?? 3 );

const browser = await chromium.launch( { headless: true, executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
	args: [ '--headless=new', '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=d3d11', '--ignore-gpu-blocklist', '--disable-gpu-sandbox', '--disable-background-timer-throttling', '--disable-backgrounding-occluded-windows', '--disable-renderer-backgrounding', '--disable-features=CalculateNativeWinOcclusion' ] } );
const page = await browser.newPage();
page.on( 'pageerror', ( e ) => console.log( '[pageerror]', e.message ) );

await page.addInitScript( ( every ) => {

	let stored, counter = 0;
	window.__series = [];

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
				if ( n % every !== 0 || window.__series.length >= 4000 ) return;

				const g = probe.velocityGrid;
				const v = await g.dataV.toArray();
				const [ NVx, NVy ] = g.dataSizeV;
				const at = ( i, j, k ) => v[ i + NVx * j + NVx * NVy * k ];
				// y = 9 is the rod's own centre line, z = 12 the mid plane.
				window.__series.push( [ n, at( 16, 9, 12 ), at( 22, 9, 12 ), at( 30, 9, 12 ), at( 40, 9, 12 ) ] );

			};

		}
	} );

}, EVERY );

await page.goto( URL, { waitUntil: 'load' } );
await new Promise( ( r ) => setTimeout( r, SECONDS * 1000 ) );

const series = await page.evaluate( () => window.__series );
await browser.close();

// Cross-correlation between probes: a shed vortex convects DOWNSTREAM, so
// the far probe must lag the near one by a positive number of frames, at a
// speed near the free stream's. A reflection off the outflow travels the
// other way and would show a negative lag.
function bestLag( a, b, maxLag ) {

	const mean = ( x ) => x.reduce( ( p, q ) => p + q, 0 ) / x.length;
	const ma = mean( a ), mb = mean( b );
	let best = 0, bestScore = - Infinity;

	for ( let lag = - maxLag; lag <= maxLag; lag ++ ) {

		let sum = 0, n = 0;
		for ( let i = 0; i < a.length; i ++ ) {

			const j = i + lag;
			if ( j < 0 || j >= b.length ) continue;
			sum += ( a[ i ] - ma ) * ( b[ j ] - mb );
			n ++;

		}
		const score = n > 20 ? sum / n : - Infinity;
		if ( score > bestScore ) { bestScore = score; best = lag; }

	}

	return best;

}

const LABELS = [ 'x=16 (1D behind)', 'x=22 (2D behind)', 'x=30 (3.3D behind)', 'x=40 (5D, near outflow)' ];
const DT = 0.05, D = 6, U = 2;

console.log( `${ series.length } samples, every ${ EVERY } frames, up to frame ${ series.length ? series[ series.length - 1 ][ 0 ] : 0 }` );

for ( let c = 1; c <= 4; c ++ ) {

	// discard the opening transient
	const raw = series.slice( Math.floor( series.length * 0.3 ) ).map( ( r ) => r[ c ] );
	if ( raw.length < 20 ) continue;

	const mean = raw.reduce( ( a, b ) => a + b, 0 ) / raw.length;
	const dev = raw.map( ( x ) => x - mean );
	const std = Math.sqrt( dev.reduce( ( a, b ) => a + b * b, 0 ) / dev.length );

	// crossings of the mean, counted only when the excursion is real
	let crossings = 0, last = 0;
	for ( const d of dev ) {

		const s = d > 0.25 * std ? 1 : d < - 0.25 * std ? - 1 : 0;
		if ( s !== 0 && last !== 0 && s !== last ) crossings ++;
		if ( s !== 0 ) last = s;

	}

	const spanFrames = raw.length * EVERY;
	const periodFrames = crossings > 1 ? ( 2 * spanFrames ) / crossings : NaN;
	const strouhal = Number.isFinite( periodFrames ) ? ( D / U ) / ( periodFrames * DT ) : NaN;

	console.log(
		`${ LABELS[ c - 1 ].padEnd( 24 ) } mean ${ mean.toFixed( 3 ).padStart( 7 ) }  std ${ std.toFixed( 3 ).padStart( 6 ) }  ` +
		`sign changes ${ String( crossings ).padStart( 3 ) }  period ~${ Number.isFinite( periodFrames ) ? periodFrames.toFixed( 0 ) : '-' } frames  St ~${ Number.isFinite( strouhal ) ? strouhal.toFixed( 3 ) : '-' }`
	);

	// a coarse trace, so the shape is visible rather than inferred
	const step = Math.max( 1, Math.floor( dev.length / 100 ) );
	let trace = '';
	for ( let i = 0; i < dev.length; i += step ) trace += dev[ i ] > 0.5 * std ? '+' : dev[ i ] < - 0.5 * std ? '-' : '.';
	console.log( '  ' + trace );

}

{

	const tail = series.slice( Math.floor( series.length * 0.3 ) );
	const col = ( c ) => tail.map( ( r ) => r[ c ] );
	const pairs = [ [ 2, 3, 22, 30 ], [ 3, 4, 30, 40 ], [ 2, 4, 22, 40 ] ];

	console.log( '' );
	console.log( 'phase lag between probes (positive = the downstream probe follows):' );

	for ( const [ ca, cb, xa, xb ] of pairs ) {

		const lagSamples = bestLag( col( ca ), col( cb ), Math.floor( 300 / EVERY ) );
		const lagFrames = lagSamples * EVERY;
		const speed = lagFrames !== 0 ? ( xb - xa ) / ( lagFrames * DT ) : NaN;
		console.log( `  x=${ xa } -> x=${ xb }: ${ lagFrames >= 0 ? '+' : '' }${ lagFrames } frames, implying ${ Number.isFinite( speed ) ? speed.toFixed( 2 ) : '-' } units/time (free stream is ${ U })` );

	}

}
