import * as tslify from 'tslify';
import { float, vec3, positionLocal } from 'three/tsl';
import {
	Scene,
	PerspectiveCamera,
	CircleGeometry,
	InstancedMesh,
	MeshBasicNodeMaterial,
	Color
} from 'three/webgpu';

const statusEl = document.querySelector( '#status' );

function status( text, ok ) {

	statusEl.textContent = text;
	statusEl.className = ok === undefined ? '' : ( ok ? 'ok' : 'err' );

}

try {

	const renderer = await tslify.init( { allowFallback: true } );
	status( `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' } — setting up…` );

	const N = 1024;
	const G = 0.05;
	const softening = 1.0;
	const dt = 0.08;

	const positions = tslify.arrayN( 'vec3', N );
	const velocities = tslify.arrayN( 'vec3', N );

	// seed: random points in a sphere, at rest — mutual gravity does the rest.
	const radius = 25;
	const posSeed = new Float32Array( N * 3 );

	for ( let i = 0; i < N; i ++ ) {

		let x, y, z;

		do {

			x = ( Math.random() * 2 - 1 );
			y = ( Math.random() * 2 - 1 );
			z = ( Math.random() * 2 - 1 );

		} while ( x * x + y * y + z * z > 1 );

		posSeed[ i * 3 + 0 ] = x * radius;
		posSeed[ i * 3 + 1 ] = y * radius;
		posSeed[ i * 3 + 2 ] = z * radius;

	}

	positions.fromArray( posSeed );
	velocities.fromArray( new Float32Array( N * 3 ) ); // zero

	const step = tslify.kernel( N, ( i ) => {

		const myPos = positions( i ).toConst();
		const acc = vec3( 0, 0, 0 ).toVar();

		tslify.Loop( N, ( { i: j } ) => {

			tslify.If( j.equal( i ), () => {

				tslify.Continue();

			} );

			const diff = positions( j ).sub( myPos );
			const distSq = diff.dot( diff ).add( softening * softening );
			const dist = distSq.sqrt();
			const invDist3 = float( 1 ).div( dist.mul( distSq ) );

			acc.addAssign( diff.mul( G ).mul( invDist3 ) );

		} );

		const newVel = velocities( i ).add( acc.mul( dt ) );

		velocities( i ).assign( newVel );
		positions( i ).assign( myPos.add( newVel.mul( dt ) ) );

	} );

	// --- rendering: N instances of a tiny circle, each instance's position driven straight
	// from the compute buffer via instanceIndex. (Points geometry doesn't establish real GPU
	// instancing, so a storage-instanced buffer's toAttribute() can't vary per-vertex there —
	// InstancedMesh does establish real instancing, matching what the buffer expects.)

	const scene = new Scene();
	scene.background = new Color( 0x000000 );

	const camera = new PerspectiveCamera( 60, window.innerWidth / window.innerHeight, 0.1, 1000 );
	camera.position.set( 0, 0, 90 );
	camera.lookAt( 0, 0, 0 );

	const geometry = new CircleGeometry( 0.4, 8 );
	const material = new MeshBasicNodeMaterial();
	material.positionNode = positionLocal.add( positions.node.toAttribute() ); // local circle shape + per-instance particle offset
	material.color = new Color( 0x66ccff );

	const points = new InstancedMesh( geometry, material, N );
	points.frustumCulled = false;
	scene.add( points );

	window.addEventListener( 'resize', () => {

		camera.aspect = window.innerWidth / window.innerHeight;
		camera.updateProjectionMatrix();
		renderer.setSize( window.innerWidth, window.innerHeight );

	} );

	let frame = 0;

	renderer.setAnimationLoop( () => {

		step();
		renderer.render( scene, camera );
		frame ++;

	} );

	status( `backend: ${ renderer.backend?.constructor?.name ?? 'unknown' } — running, N=${ N }`, true );

	window.__tslifyNBodyDebug = { positions, frameCount: () => frame };

} catch ( error ) {

	status( error.message, false );
	console.error( error );

}
