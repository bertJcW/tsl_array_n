import { WebGPURenderer } from 'three/webgpu';

let currentRenderer = null;

export function isSupported() {

	return typeof navigator !== 'undefined' && !! navigator.gpu;

}

export async function init( options = {} ) {

	if ( ! isSupported() ) {

		throw new Error(
			'tslify requires WebGPU, but navigator.gpu is not available in this browser. ' +
			'Use a recent Chrome or Edge over https:// or http://localhost.'
		);

	}

	const { renderer, canvas, container, width, height, allowFallback = false, ...rendererOptions } = options;

	let target;

	if ( renderer ) {

		target = renderer;

	} else {

		target = new WebGPURenderer( { canvas, antialias: true, ...rendererOptions } );

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
			'tslify: this browser reports navigator.gpu, but the WebGPU adapter request failed and three.js ' +
			`fell back to "${ target.backend?.constructor?.name ?? 'unknown backend' }". GPU compute needs a real ` +
			'WebGPU backend. Pass { allowFallback: true } to init() to use this renderer anyway (compute will not work).'
		);

	}

	currentRenderer = target;

	return currentRenderer;

}

export function getRenderer() {

	if ( ! currentRenderer ) {

		throw new Error( 'tslify: call init() before using field() / kernel().' );

	}

	return currentRenderer;

}
