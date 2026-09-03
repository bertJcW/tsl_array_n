import * as tsl_array_n from 'tsl_array_n';

const status = document.getElementById( 'status' );

try {

	const renderer = await tsl_array_n.init();

	status.textContent = `WebGPU ready ✓ (backend: ${ renderer.backend?.constructor?.name ?? 'unknown' })`;
	status.classList.add( 'ok' );

} catch ( error ) {

	status.textContent = error.message;
	status.classList.add( 'err' );

}
