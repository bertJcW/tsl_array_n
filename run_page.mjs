import { chromium } from 'playwright-core';
const URL = process.argv[ 2 ];
const WAIT = Number( process.argv[ 3 ] ?? 15000 );
const browser = await chromium.launch( { headless: true, executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
	args: [ '--headless=new', '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=d3d11', '--ignore-gpu-blocklist', '--disable-gpu-sandbox', '--disable-background-timer-throttling', '--disable-backgrounding-occluded-windows', '--disable-renderer-backgrounding', '--disable-features=CalculateNativeWinOcclusion' ] } );
const page = await browser.newPage();
page.on( 'pageerror', ( e ) => console.log( '[pageerror]', e.message ) );
page.on( 'console', ( m ) => { const t = m.text(); if ( /error|Error/.test( t ) ) console.log( '[console]', t.slice( 0, 300 ) ); } );
await page.goto( URL, { waitUntil: 'load' } );
await new Promise( ( r ) => setTimeout( r, WAIT ) );
console.log( await page.evaluate( () => document.body.innerText ) );
await browser.close();
