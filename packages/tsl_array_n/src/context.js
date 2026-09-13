import { WebGPURenderer } from 'three/webgpu';

let currentRenderer = null;

export function isSupported() {

	return typeof navigator !== 'undefined' && !! navigator.gpu;

}

// *** Why init() builds the device itself instead of letting three.js do it ***
//
// `requestDevice()` with no `requiredLimits` gets WebGPU's *default* limits,
// and those are downlevel values rather than what the adapter offers.
// Measured on the development machine (RTX 5060 Ti, Chrome 152): the adapter
// offers maxStorageBuffersPerShaderStage = 16 and
// maxComputeInvocationsPerWorkgroup = 1024, while the device three.js builds
// gets 8 and 256. A kernel that binds nine storage buffers then fails at
// pipeline creation with
//
//     Invalid BindGroupLayout ... While validating binding counts
//
// reported as an uncaptured GPU error on the first dispatch, which leaves that
// whole pass silently undone -- this is what fluxflow's
// examples/16-karman-vortex-street did, and it is a hard failure that looks
// like nothing at all from the JavaScript side.
//
// Nothing here asks for a limit the machine has not got: every value requested
// is the adapter's own, so the request is satisfiable by construction. Pass
// `adapterLimits: false` to init() to go back to three.js's own device.
async function createDeviceWithAdapterLimits( powerPreference ) {

	try {

		const adapter = await navigator.gpu.requestAdapter(
			powerPreference === undefined ? {} : { powerPreference }
		);

		if ( adapter === null ) return null;

		// Limits live as accessors on the prototype, so enumerate both the
		// instance and its prototype rather than trusting Object.entries.
		const names = [];

		for ( const source of [ adapter.limits, Object.getPrototypeOf( adapter.limits ) ] ) {

			for ( const name of Object.getOwnPropertyNames( source ) ) {

				if ( typeof adapter.limits[ name ] === 'number' && names.includes( name ) === false ) names.push( name );

			}

		}

		const requiredLimits = {};

		for ( const name of names ) requiredLimits[ name ] = adapter.limits[ name ];

		return await adapter.requestDevice( {
			requiredFeatures: [ ...adapter.features ],
			requiredLimits
		} );

	} catch ( error ) {

		// An adapter that cannot answer this falls back to three.js's own
		// device, which is what every earlier version of this package used.
		return null;

	}

}

export async function init( options = {} ) {

	if ( ! isSupported() ) {

		throw new Error(
			'tsl_array_n requires WebGPU, but navigator.gpu is not available in this browser. ' +
			'Use a recent Chrome or Edge over https:// or http://localhost.'
		);

	}

	const { renderer, canvas, container, width, height, allowFallback = false, adapterLimits = true, ...rendererOptions } = options;

	let target;

	if ( renderer ) {

		target = renderer;

	} else {

		// See createDeviceWithAdapterLimits above: without this the device
		// carries WebGPU's downlevel defaults, and a kernel binding more than
		// eight storage buffers silently fails to dispatch.
		const device = adapterLimits ? await createDeviceWithAdapterLimits( rendererOptions.powerPreference ) : null;

		target = new WebGPURenderer( { canvas, antialias: true, ...rendererOptions, ...( device === null ? {} : { device } ) } );

		const targetWidth = width ?? ( container ? container.clientWidth : window.innerWidth );
		const targetHeight = height ?? ( container ? container.clientHeight : window.innerHeight );

		target.setPixelRatio( window.devicePixelRatio );
		target.setSize( targetWidth, targetHeight );

		if ( ! canvas ) {

			( container ?? document.body ).appendChild( target.domElement );

		}

	}

	await target.init();

	if ( ! allowFallback && target.backend?.isWebGPUBackend !== true ) {

		throw new Error(
			'tsl_array_n: this browser reports navigator.gpu, but the WebGPU adapter request failed and three.js ' +
			`fell back to "${ target.backend?.constructor?.name ?? 'unknown backend' }". GPU compute needs a real ` +
			'WebGPU backend. Pass { allowFallback: true } to init() to use this renderer anyway (compute will not work).'
		);

	}

	currentRenderer = target;

	return currentRenderer;

}

export function getRenderer() {

	if ( ! currentRenderer ) {

		throw new Error( 'tsl_array_n: call init() before using field() / kernel().' );

	}

	return currentRenderer;

}
