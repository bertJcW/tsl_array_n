import { defineConfig } from 'vite';
import { fileURLToPath, URL } from 'node:url';

export default defineConfig( {
	resolve: {
		alias: {
			'tsl_array_n': fileURLToPath( new URL( './src/index.js', import.meta.url ) )
		}
	}
} );
