import * as tslify from 'tslify';

const status = document.getElementById( 'status' );

try {

	const renderer = await tslify.init();

	status.textContent = `WebGPU ready ✓ (backend: ${ renderer.backend?.constructor?.name ?? 'unknown' })`;
	status.classList.add( 'ok' );

} catch ( error ) {

	status.textContent = error.message;
	status.classList.add( 'err' );

}
