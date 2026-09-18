// 3D counterpart of sdf_inflow_outflow2.js. Same family of thin wrappers
// around a collider-shaped SDF object -- see that file's own header
// comment for the full mantaflow-derived reasoning (zero-pressure ghost
// boundary via the pressure solver's existing Dirichlet mechanism, why
// outflow/fuel-source are separately named siblings of inflow) -- none of
// it is dimension-specific except the velocity vector's own width.

import { vec3, float } from 'three/tsl';
import { createSDFStaticCollider3 } from './sdf_collider3.js';
import { isInsideSdf } from './level_set_utils.js';

// See sdf_inflow_outflow2.js's own createSDFInflow2 header comment for
// velocity/mode's full semantics -- unchanged, velocity is now [vx,vy,vz]
// or a live vec3 node.
export function createSDFInflow3( resolutionX, resolutionY, resolutionZ, gridSpacingX, gridSpacingY, gridSpacingZ, originX, originY, originZ, options = {} ) {

	const sdf = createSDFStaticCollider3( resolutionX, resolutionY, resolutionZ, gridSpacingX, gridSpacingY, gridSpacingZ, originX, originY, originZ );
	const { velocity = [ 0, 0, 0 ], mode = 'set' } = options;
	const velocityNode = Array.isArray( velocity ) ? vec3( velocity[ 0 ], velocity[ 1 ], velocity[ 2 ] ) : velocity;

	return { ...sdf, velocity: velocityNode, mode };

}

export function createSDFOutflow3( resolutionX, resolutionY, resolutionZ, gridSpacingX, gridSpacingY, gridSpacingZ, originX, originY, originZ ) {

	return createSDFStaticCollider3( resolutionX, resolutionY, resolutionZ, gridSpacingX, gridSpacingY, gridSpacingZ, originX, originY, originZ );

}

// See sdf_inflow_outflow2.js's own createSDFFuelSource2 header comment --
// unchanged, a scalar injection (not a vector), so nothing here actually
// depends on dimension at all beyond the SDF grid underneath it.
export function createSDFFuelSource3( resolutionX, resolutionY, resolutionZ, gridSpacingX, gridSpacingY, gridSpacingZ, originX, originY, originZ, options = {} ) {

	const sdf = createSDFStaticCollider3( resolutionX, resolutionY, resolutionZ, gridSpacingX, gridSpacingY, gridSpacingZ, originX, originY, originZ );
	const { fuel = 1, mode = 'set' } = options;
	const fuelNode = typeof fuel === 'number' ? float( fuel ) : fuel;

	return { ...sdf, fuel: fuelNode, mode };

}

export function createOutflowPressureDirichlet3( outflows, { target = 0 } = {} ) {

	const list = Array.isArray( outflows ) ? outflows : [ outflows ];
	const targetNode = typeof target === 'number' ? float( target ) : target;

	return function dirichlet( pos ) {

		let active = null;

		for ( const outflow of list ) {

			const inside = isInsideSdf( outflow.sample( pos ) );
			active = active ? active.or( inside ) : inside;

		}

		return { active, target: targetNode };

	};

}

// Dimension-independent (operates only on abstract dirichlet(pos)
// functions) -- kept as its own copy for this file's self-containedness
// rather than importing sdf_inflow_outflow2.js's identical one, matching
// this port's established "no shared state between the 2D and 3D files"
// convention. See sdf_inflow_outflow2.js's own combineDirichlet header
// comment for the "b wins on overlap" reasoning. Named with the "3" suffix
// (unlike its logic-identical 2D sibling) so the two can both be reached
// through grid/index.js's barrel export -- an unsuffixed name here would
// collide with sdf_inflow_outflow2.js's own export in that shared
// namespace and silently disappear from it (ES module ambiguous star-
// export resolution, not an error, just a missing binding).
export function combineDirichlet3( a, b ) {

	if ( ! a ) return b;
	if ( ! b ) return a;

	return function dirichlet( pos ) {

		const resultA = a( pos );
		const resultB = b( pos );

		return {
			active: resultA.active.or( resultB.active ),
			target: resultB.active.select( resultB.target, resultA.target )
		};

	};

}
