// Every drivable example, 12,000 solver steps each, judged rather than watched.
//
// docs/long-run-stability.md records a 12,000-step run from 2026-09-15 and
// then disowns half of it: its `converged` column was measured before the
// convergence bug was fixed, and the defaults it names no longer exist. Its
// closing line is "a full 12,000-step re-run is the outstanding item". This
// is that re-run, with two differences that matter.
//
// The first is the instrument. That run reported counters -- converged,
// rejected, non-finite, peak pressure -- and counters are what the 3D
// investigation found cannot decide the question: examples/16 converged on
// every single frame while carrying 13.89x its inflow out of the outlet, and
// examples/35 settled after blowing up into a state whose maximum velocity
// was a perfectly reasonable 17.5. So each scene here is run through
// solver_health.mjs, which judges conservation laws and exits 0 or 2. The
// counters are still collected, beside the verdict, as evidence to explain.
//
// The second is coverage. That run drove 9 scenes; 15 expose a probe now,
// including all three 3D ones. Every example that does not appear in the
// results is listed with the reason, so the gap is visible rather than
// implied -- see docs/verification.md's third rule.
//
// usage: node long_run.mjs [frames] [sampleEvery] [only-substring ...]

import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';

const FRAMES = Number( process.argv[ 2 ] ?? 12000 );
const SAMPLE = Number( process.argv[ 3 ] ?? 100 );
const ONLY = process.argv.slice( 4 );

const BASE = 'http://localhost:5200/examples';
const OUT = 'long-run-logs';

// Everything with a probe, in the order the examples are numbered. The note
// is what this scene is here to stress, so a failure has a subject.
const SCENES = [
	[ '14-stable-fluids', 'a fully closed autonomous domain -- nothing drives it and nothing leaves' ],
	[ '15-flow-past-cylinder', 'inflow/outflow with a collider' ],
	[ '16-karman-vortex-street', 'unsteady wake, the 2D outflow rewrite' ],
	[ '17-smoke-fire', 'buoyancy, no inlet, top vent' ],
	[ '18-explosion', 'buoyancy with a violent transient' ],
	[ '19-fuel-fire', 'combustion source term' ],
	[ '20-flip-dam-break', 'free surface' ],
	[ '21-flip-irregular-container', 'free surface against an SDF container' ],
	[ '22-flip-multiple-colliders', 'several colliders at once' ],
	[ '23-flip-moving-collider', 'a collider that moves every step' ],
	[ '24-two-phase-bubble-rise', 'the two-phase solver: air simulated rather than a void' ],
	[ '25-dye-injection', 'the two-phase solver in miscible mode' ],
	[ '26-dye-free-surface', 'dye advected on a free surface' ],
	[ '28-drop-into-pool', 'the iteration-cap case of the 2026-09-15 run' ],
	[ '29-static-droplet', 'surface tension against Young-Laplace' ],
	[ '33-flip-dam-break-3d', '3D free surface' ],
	[ '34-smoke-plume-3d', '3D buoyancy with a vent' ],
	[ '35-karman-vortex-street-3d', 'the scene the 3D investigation was about' ]
];

// Pages with no probe to drive. Named rather than silently absent: a
// self-checking page has no long run to do, which is a different statement
// from "not tested".
const NOT_DRIVEN = [
	[ '00-13, 27, 30, 31, 32', 'self-checking pages: they verify against a known answer on load and finish. Layer 2, not layer 3, and run_page.mjs drives them' ]
];

// Every scene that has a solver to drive now has a probe. The three that did
// not -- 14, 24 and 25 -- were listed here as open gaps, and closing them was
// worth it immediately: 25-dye-injection turns out to be BROKEN, leaving 15x
// the divergence the projection was asked to remove, which nothing could see
// because nothing could drive it.


mkdirSync( OUT, { recursive: true } );

const scenes = ONLY.length ? SCENES.filter( ( [ n ] ) => ONLY.some( ( o ) => n.includes( o ) ) ) : SCENES;

console.log( `${ scenes.length } scenes x ${ FRAMES } steps = ${ ( scenes.length * FRAMES ).toLocaleString() } solver steps, sampled every ${ SAMPLE }` );
console.log( `logs in ${ OUT }/\n` );

const results = [];
const runStart = Date.now();

for ( const [ name, note ] of scenes ) {

	const started = Date.now();
	process.stdout.write( `${ name.padEnd( 30 ) } ` );

	const log = await run( `${ BASE }/${ name }/` );
	const seconds = ( Date.now() - started ) / 1000;
	writeFileSync( `${ OUT }/${ name }.log`, log.text );

	const r = parse( log, name, note, seconds );
	results.push( r );

	console.log( `${ r.verdict.padEnd( 18 ) } ${ r.frames } frames  ${ seconds.toFixed( 0 ) }s  ${ ( r.frames / seconds ).toFixed( 0 ) } fps` );
	if ( r.detail ) console.log( `${ ' '.repeat( 31 ) }${ r.detail }` );

}

function run( url ) {

	return new Promise( ( resolve ) => {

		// Long enough for 12,000 steps on the slowest 3D scene. The page is
		// left to run at its own rate rather than driven step by step: what a
		// long run is asking about is the scene as it actually runs.
		const child = spawn( process.execPath, [ 'solver_health.mjs', url, String( FRAMES ), String( SAMPLE ) ], {
			env: { ...process.env, SOLVER_HEALTH_TIMEOUT: String( 3 * 3600 * 1000 ) }
		} );

		let text = '';
		child.stdout.on( 'data', ( d ) => { text += d; } );
		child.stderr.on( 'data', ( d ) => { text += d; } );
		child.on( 'close', ( code ) => resolve( { text, code } ) );

	} );

}

function parse( log, name, note, seconds ) {

	const line = ( re ) => ( log.text.match( re ) ?? [] )[ 0 ];
	const grab = ( re, i = 1 ) => ( log.text.match( re ) ?? [] )[ i ];

	const verdictLine = line( /VERDICT: [^\n]*/ ) ?? ( log.text.includes( 'FATAL: not WebGPU' ) ? 'NOT WEBGPU' : 'NO VERDICT' );
	const verdict = verdictLine.replace( /^VERDICT: /, '' ).split( /,| -- | over /, 1 )[ 0 ].trim();

	return {
		name, note, seconds, code: log.code,
		verdict,
		verdictLine,
		detail: ( log.text.match( /VERDICT: [^\n]*\n([^\n]*)/ ) ?? [] )[ 1 ]?.trim() ?? '',
		outOfScope: /note: (the solved region moves|with no inlet)/.test( log.text ),
		frames: Number( grab( /^(\d+) frames, sampled/m ) ?? 0 ),
		converged: Number( grab( /solver counters: (\d+) converged/ ) ?? 0 ),
		rejected: Number( grab( /(\d+) rejected/ ) ?? 0 ),
		breakdowns: Number( grab( /(\d+) CG breakdowns/ ) ?? 0 ),
		residualMedian: grab( /median ([\d.e+-]+), 90th/ ) ?? '',
		residualWorst: grab( /worst ([\d.e+-]+)/ ) ?? '',
		backend: grab( /backend: (\w+)/ ) ?? ''
	};

}

// ---- the report -------------------------------------------------------

const totalSteps = results.reduce( ( a, r ) => a + r.frames, 0 );
const wall = ( Date.now() - runStart ) / 1000;

const md = [];
md.push( `| example | verdict | steps | converged | rejected | CG breakdowns | residual median / worst | what it stresses |` );
md.push( `| --- | --- | --- | --- | --- | --- | --- | --- |` );
for ( const r of results ) {

	const v = r.verdict === 'HEALTHY' ? ( r.outOfScope ? '**HEALTHY** (narrow)' : '**HEALTHY**' ) : `**${ r.verdict }**`;
	md.push( `| ${ r.name } | ${ v } | ${ r.frames.toLocaleString() } | ${ r.converged.toLocaleString() } | ${ r.rejected } | ${ r.breakdowns } | ${ r.residualMedian } / ${ r.residualWorst } | ${ r.note } |` );

}
md.push( '' );
md.push( `Not driven, and why:` );
md.push( '' );
for ( const [ what, why ] of NOT_DRIVEN ) md.push( `- **${ what }** -- ${ why }` );

const broken = results.filter( ( r ) => r.code !== 0 );
const narrow = results.filter( ( r ) => r.code === 0 && r.outOfScope );

console.log( `\n${ '='.repeat( 72 ) }` );
console.log( `${ totalSteps.toLocaleString() } solver steps across ${ results.length } scenes, ${ ( wall / 60 ).toFixed( 1 ) } minutes of wall clock` );
console.log( `${ results.length - broken.length } healthy (${ narrow.length } of them narrow passes), ${ broken.length } not` );
for ( const r of broken ) console.log( `  ${ r.name }: ${ r.verdictLine }` );

writeFileSync( `${ OUT }/summary.md`, md.join( '\n' ) + '\n' );
writeFileSync( `${ OUT }/summary.json`, JSON.stringify( { FRAMES, SAMPLE, totalSteps, wallSeconds: wall, results }, null, '\t' ) );
console.log( `\ntable written to ${ OUT }/summary.md` );

process.exit( broken.length ? 2 : 0 );
