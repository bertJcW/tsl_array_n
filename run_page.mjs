// Load a self-checking page and print what it says -- once it has finished
// saying it.
//
// *** This used to wait a fixed 15 seconds, and that silently truncated
// evidence ***
//
// sandbox/poisson-3d-dirichlet/ checks V-cycle symmetry at 1, 2, 3 and 4
// levels against two mask configurations: eight rows. Fifteen seconds got
// three of them, and the output of a partial run looks exactly like the output
// of a complete one -- a list of ticks. Reading that list led to the written
// conclusion that the sandbox only tested 1 and 2 levels and that the missing
// 3- and 4-level cases were the coverage gap behind T3. They were never
// missing. The harness was cutting them off.
//
// So this waits for the page to stop producing output rather than for a clock,
// and says which of the two ended the run. A result that hit the cap is
// labelled PARTIAL, because a truncated pass is not a pass.
//
// usage: node run_page.mjs <url> [maxSeconds] [settleSeconds]

import { chromium } from 'playwright-core';

const URL = process.argv[ 2 ];
const MAX_MS = Number( process.argv[ 3 ] ?? 300 ) * 1000;
const SETTLE_MS = Number( process.argv[ 4 ] ?? 4 ) * 1000;
const POLL_MS = 500;

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
page.on( 'console', ( m ) => { const t = m.text(); if ( /error|Error/.test( t ) ) console.log( '[console]', t.slice( 0, 300 ) ); } );

await page.goto( URL, { waitUntil: 'load' } );

const started = Date.now();
let text = '';
let lastChange = Date.now();
let settled = false;

while ( Date.now() - started < MAX_MS ) {

	await new Promise( ( r ) => setTimeout( r, POLL_MS ) );

	const now = await page.evaluate( () => document.body.innerText ).catch( () => text );
	if ( now !== text ) { text = now; lastChange = Date.now(); continue; }

	// A page that has not changed for SETTLE_MS is taken to be finished. Some
	// of these pages run a rAF loop and never settle, which is what the cap is
	// for -- and why the cap being reached is reported rather than ignored.
	if ( Date.now() - lastChange >= SETTLE_MS && text.trim().length > 0 ) { settled = true; break; }

}

const elapsed = ( ( Date.now() - started ) / 1000 ).toFixed( 1 );
await browser.close();

console.log( text );

const failures = ( text.match( /✗/g ) ?? [] ).length;
const passes = ( text.match( /✓/g ) ?? [] ).length;

console.log( `\n---` );
if ( settled ) {

	console.log( `page finished after ${ elapsed }s: ${ passes } passed, ${ failures } failed` );

} else {

	console.log( `PARTIAL: still producing output when the ${ MAX_MS / 1000 }s cap was reached (${ passes } passed, ${ failures } failed SO FAR).` );
	console.log( `Nothing above is a complete result. Re-run with a larger cap: node run_page.mjs ${ URL } <seconds>` );

}

process.exit( failures > 0 || ! settled ? 2 : 0 );
