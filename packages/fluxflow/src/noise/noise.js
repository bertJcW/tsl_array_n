// Ported from noise/noise.py, which is itself ported from WebGL-Noise
// (https://github.com/ashima/webgl-noise), MIT License:
//   Copyright (C) 2011 by Ashima Arts (Simplex noise)
//   Copyright (C) 2011-2016 by Stefan Gustavson (Classic noise and others)
// See ../../THIRD-PARTY-NOTICES.md for the full license text.
//
// Plain JS functions building TSL node graphs, not tsl_array_n.func() --
// same reasoning as grid_math.js: these are meant to be called from inside
// a kernel/func body during JS graph construction, not to be genuine
// device-side callable subroutines, and func()/Fn's single-destructured-
// array calling convention would only get in the way here.
//
// The four trivial one-line Taichi wrappers (floor/fract/abs/dot just
// forwarding to tm.floor/tm.fract/ti.abs/x.dot(y)) aren't ported as
// separate functions -- TSL's own floor/fract/abs functions and the .dot()
// node method are used directly at each call site instead, since wrapping
// them again here would be pure indirection.
//
// Function names get an explicit "3d" suffix (perlinNoise3d, simplexNoise3d,
// alongside the source's own already-suffixed cellular3d) for consistency --
// the source is inconsistent about this (perlin_noise/simplex_noise have no
// suffix, cellular3d does), and all three are in fact 3D-only.
//
// cellular3d drops the source's `ENABLE_COMPLEX_VERSION` branch entirely:
// that flag is hardcoded `False` and never toggled anywhere in the source,
// so the `if ti.static(ENABLE_COMPLEX_VERSION):` branch never actually
// compiles into anything -- it's dead code. Only the `else` branch (the
// real F1+F2 sorting network) is ported.

import { floor, fract, abs, min, max, step, sqrt, mix, vec2, vec3, vec4 } from 'three/tsl';

function mod289( x ) {

	return x.sub( floor( x.mul( 1 / 289 ) ).mul( 289 ) );

}

function mod7( x ) {

	return x.sub( floor( x.mul( 1 / 7 ) ).mul( 7 ) );

}

export function permute( x ) {

	return mod289( x.mul( 34 ).add( 10 ).mul( x ) );

}

export function taylorInvSqrt( r ) {

	return r.mul( -0.85373472095314 ).add( 1.79284291400159 );

}

function fade( t ) {

	return t.mul( t ).mul( t ).mul( t.mul( t.mul( 6 ).sub( 15 ) ).add( 10 ) );

}

// Classic (improved) Perlin noise. P: vec3 position, returns a float in
// roughly [-1, 1].
export function perlinNoise3d( P ) {

	let Pi0 = floor( P );
	let Pi1 = Pi0.add( vec3( 1, 1, 1 ) );
	Pi0 = mod289( Pi0 );
	Pi1 = mod289( Pi1 );

	const Pf0 = fract( P );
	const Pf1 = Pf0.sub( vec3( 1, 1, 1 ) );

	const ix  = vec4( Pi0.x, Pi1.x, Pi0.x, Pi1.x );
	const iy  = vec4( Pi0.y, Pi0.y, Pi1.y, Pi1.y );
	const iz0 = vec4( Pi0.z, Pi0.z, Pi0.z, Pi0.z );
	const iz1 = vec4( Pi1.z, Pi1.z, Pi1.z, Pi1.z );

	const ixy  = permute( permute( ix ).add( iy ) );
	const ixy0 = permute( ixy.add( iz0 ) );
	const ixy1 = permute( ixy.add( iz1 ) );

	let gx0 = ixy0.mul( 1 / 7 );
	let gy0 = fract( floor( gx0 ).mul( 1 / 7 ) ).sub( 0.5 );
	gx0 = fract( gx0 );

	const gz0 = vec4( 0.5, 0.5, 0.5, 0.5 ).sub( abs( gx0 ) ).sub( abs( gy0 ) );
	const sz0 = step( gz0, vec4( 0, 0, 0, 0 ) );
	gx0 = gx0.sub( sz0.mul( step( 0, gx0 ).sub( 0.5 ) ) );
	gy0 = gy0.sub( sz0.mul( step( 0, gy0 ).sub( 0.5 ) ) );

	let gx1 = ixy1.mul( 1 / 7 );
	let gy1 = fract( floor( gx1 ).mul( 1 / 7 ) ).sub( 0.5 );
	gx1 = fract( gx1 );

	const gz1 = vec4( 0.5, 0.5, 0.5, 0.5 ).sub( abs( gx1 ) ).sub( abs( gy1 ) );
	const sz1 = step( gz1, vec4( 0, 0, 0, 0 ) );
	gx1 = gx1.sub( sz1.mul( step( 0, gx1 ).sub( 0.5 ) ) );
	gy1 = gy1.sub( sz1.mul( step( 0, gy1 ).sub( 0.5 ) ) );

	const g000 = vec3( gx0.x, gy0.x, gz0.x );
	const g100 = vec3( gx0.y, gy0.y, gz0.y );
	const g010 = vec3( gx0.z, gy0.z, gz0.z );
	const g110 = vec3( gx0.w, gy0.w, gz0.w );
	const g001 = vec3( gx1.x, gy1.x, gz1.x );
	const g101 = vec3( gx1.y, gy1.y, gz1.y );
	const g011 = vec3( gx1.z, gy1.z, gz1.z );
	const g111 = vec3( gx1.w, gy1.w, gz1.w );

	const norm0 = taylorInvSqrt( vec4( g000.dot( g000 ), g010.dot( g010 ), g100.dot( g100 ), g110.dot( g110 ) ) );
	const norm1 = taylorInvSqrt( vec4( g001.dot( g001 ), g011.dot( g011 ), g101.dot( g101 ), g111.dot( g111 ) ) );

	const n000 = norm0.x.mul( g000.dot( Pf0 ) );
	const n010 = norm0.y.mul( g010.dot( vec3( Pf0.x, Pf1.y, Pf0.z ) ) );
	const n100 = norm0.z.mul( g100.dot( vec3( Pf1.x, Pf0.y, Pf0.z ) ) );
	const n110 = norm0.w.mul( g110.dot( vec3( Pf1.x, Pf1.y, Pf0.z ) ) );
	const n001 = norm1.x.mul( g001.dot( vec3( Pf0.x, Pf0.y, Pf1.z ) ) );
	const n011 = norm1.y.mul( g011.dot( vec3( Pf0.x, Pf1.y, Pf1.z ) ) );
	const n101 = norm1.z.mul( g101.dot( vec3( Pf1.x, Pf0.y, Pf1.z ) ) );
	const n111 = norm1.w.mul( g111.dot( Pf1 ) );

	const fadeXYZ = fade( Pf0 );
	const nZ  = mix( vec4( n000, n100, n010, n110 ), vec4( n001, n101, n011, n111 ), fadeXYZ.z );
	const nYZ = mix( vec2( nZ.x, nZ.y ), vec2( nZ.z, nZ.w ), fadeXYZ.y );
	const nXYZ = mix( nYZ.x, nYZ.y, fadeXYZ.x );

	return nXYZ.mul( 2.2 );

}

// Simplex noise. v: vec3 position, returns a float in roughly [-1, 1].
export function simplexNoise3d( v ) {

	const C = vec2( 1 / 6, 1 / 3 );
	const D = vec4( 0, 0.5, 1, 2 );

	// first corner
	const i  = floor( v.add( v.dot( vec3( C.y, C.y, C.y ) ) ) );
	const x0 = v.sub( i ).add( i.dot( vec3( C.x, C.x, C.x ) ) );

	// other corners
	const g = step( vec3( x0.y, x0.z, x0.x ), vec3( x0.x, x0.y, x0.z ) );
	const l = g.mul( -1 ).add( 1 );
	const i1 = min( vec3( g.x, g.y, g.z ), vec3( l.z, l.x, l.y ) );
	const i2 = max( vec3( g.x, g.y, g.z ), vec3( l.z, l.x, l.y ) );

	// x0 = x0 - 0.0 + 0.0 * C.xxx
	// x1 = x0 - i1  + 1.0 * C.xxx
	// x2 = x0 - i2  + 2.0 * C.xxx
	// x3 = x0 - 1.0 + 3.0 * C.xxx
	const x1 = x0.sub( i1 ).add( vec3( C.x, C.x, C.x ) );
	const x2 = x0.sub( i2 ).add( vec3( C.y, C.y, C.y ) ); // 2.0*C.x = 1/3 = C.y
	const x3 = x0.sub( vec3( D.y, D.y, D.y ) ); // -1.0+3.0*C.x = -0.5 = -D.y

	// permutations
	const iMod = mod289( i );
	const permuteZ  = permute( iMod.z.add( vec4( 0, i1.z, i2.z, 1 ) ) );
	const permuteZY = permute( permuteZ.add( iMod.y ).add( vec4( 0, i1.y, i2.y, 1 ) ) );
	const p = permute( permuteZY.add( iMod.x ).add( vec4( 0, i1.x, i2.x, 1 ) ) );

	// Gradients: 7x7 points over a square, mapped onto an octahedron.
	// The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
	const nRecip7 = 0.142857142857; // 1.0/7.0
	const ns = vec3( D.w, D.y, D.z ).mul( nRecip7 ).sub( vec3( D.x, D.z, D.x ) );

	const j = p.sub( floor( p.mul( ns.z ).mul( ns.z ) ).mul( 49 ) ); // mod(p, 7*7)

	const x_ = floor( j.mul( ns.z ) );
	const y_ = floor( j.sub( x_.mul( 7 ) ) ); // mod(j, N)

	const x = x_.mul( ns.x ).add( vec4( ns.y, ns.y, ns.y, ns.y ) );
	const y = y_.mul( ns.x ).add( vec4( ns.y, ns.y, ns.y, ns.y ) );
	const h = abs( x ).add( abs( y ) ).mul( -1 ).add( 1 );

	const b0 = vec4( x.x, x.y, y.x, y.y );
	const b1 = vec4( x.z, x.w, y.z, y.w );

	// vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0
	// vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0
	const s0 = floor( b0 ).mul( 2 ).add( 1 );
	const s1 = floor( b1 ).mul( 2 ).add( 1 );
	const sh = step( h, vec4( 0, 0, 0, 0 ) ).negate();

	const a0 = vec4( b0.x, b0.z, b0.y, b0.w ).add( vec4( s0.x, s0.z, s0.y, s0.w ).mul( vec4( sh.x, sh.x, sh.y, sh.y ) ) );
	const a1 = vec4( b1.x, b1.z, b1.y, b1.w ).add( vec4( s1.x, s1.z, s1.y, s1.w ).mul( vec4( sh.z, sh.z, sh.w, sh.w ) ) );

	let p0 = vec3( a0.x, a0.y, h.x );
	let p1 = vec3( a0.z, a0.w, h.y );
	let p2 = vec3( a1.x, a1.y, h.z );
	let p3 = vec3( a1.z, a1.w, h.w );

	// normalize gradients
	const norm = taylorInvSqrt( vec4( p0.dot( p0 ), p1.dot( p1 ), p2.dot( p2 ), p3.dot( p3 ) ) );
	p0 = p0.mul( norm.x );
	p1 = p1.mul( norm.y );
	p2 = p2.mul( norm.z );
	p3 = p3.mul( norm.w );

	// mix final noise value -- note m gets squared twice (once explicitly,
	// once again via the m.mul(m) in the dot() call below), matching the
	// source's own "m = m*m" reassignment followed by "dot(m*m, ...)" --
	// the combined effect is an m^4 falloff weight, which is the standard
	// simplex-noise formula, not a source typo.
	let m = max( vec4( x0.dot( x0 ), x1.dot( x1 ), x2.dot( x2 ), x3.dot( x3 ) ).mul( -1 ).add( 0.5 ), 0 );
	m = m.mul( m );

	return m.mul( m ).dot( vec4( p0.dot( x0 ), p1.dot( x1 ), p2.dot( x2 ), p3.dot( x3 ) ) ).mul( 105 );

}

// Cellular (Voronoi) noise. P: vec3 position, returns vec2(F1, F2), the
// real (already square-rooted) distance to the nearest and 2nd-nearest
// feature points.
export function cellular3d( P ) {

	const K = 0.142857142857; // 1/7
	const Ko = 0.428571428571; // 1/2-K/2
	const K2 = 0.020408163265306; // 1/(7*7)
	const Kz = 0.166666666667; // 1/6
	const Kzo = 0.416666666667; // 1/2-1/6*2
	const jitter = 1; // smaller jitter gives a more regular pattern

	function offsetFromP( p ) {

		const ox = fract( p.mul( K ) ).sub( Ko );
		const oy = mod7( floor( p.mul( K ) ) ).mul( K ).sub( Ko );
		const oz = floor( p.mul( K2 ) ).mul( Kz ).sub( Kzo ); // p < 289 guaranteed

		return { ox, oy, oz };

	}

	const Pi = mod289( floor( P ) );
	const Pf = fract( P ).sub( 0.5 );
	const Pfx = Pf.x.add( vec3( 1, 0, -1 ) );
	const Pfy = Pf.y.add( vec3( 1, 0, -1 ) );
	const Pfz = Pf.z.add( vec3( 1, 0, -1 ) );

	const p = permute( Pi.x.add( vec3( -1, 0, 1 ) ) );
	const p1 = permute( p.add( Pi.y ).sub( 1 ) );
	const p2 = permute( p.add( Pi.y ) );
	const p3 = permute( p.add( Pi.y ).add( 1 ) );

	const p11 = permute( p1.add( Pi.z ).sub( 1 ) );
	const p12 = permute( p1.add( Pi.z ) );
	const p13 = permute( p1.add( Pi.z ).add( 1 ) );

	const p21 = permute( p2.add( Pi.z ).sub( 1 ) );
	const p22 = permute( p2.add( Pi.z ) );
	const p23 = permute( p2.add( Pi.z ).add( 1 ) );

	const p31 = permute( p3.add( Pi.z ).sub( 1 ) );
	const p32 = permute( p3.add( Pi.z ) );
	const p33 = permute( p3.add( Pi.z ).add( 1 ) );

	const o11 = offsetFromP( p11 );
	const o12 = offsetFromP( p12 );
	const o13 = offsetFromP( p13 );
	const o21 = offsetFromP( p21 );
	const o22 = offsetFromP( p22 );
	const o23 = offsetFromP( p23 );
	const o31 = offsetFromP( p31 );
	const o32 = offsetFromP( p32 );
	const o33 = offsetFromP( p33 );

	const dx11 = Pfx.add( o11.ox.mul( jitter ) );
	const dy11 = Pfy.x.add( o11.oy.mul( jitter ) );
	const dz11 = Pfz.x.add( o11.oz.mul( jitter ) );

	const dx12 = Pfx.add( o12.ox.mul( jitter ) );
	const dy12 = Pfy.x.add( o12.oy.mul( jitter ) );
	const dz12 = Pfz.y.add( o12.oz.mul( jitter ) );

	const dx13 = Pfx.add( o13.ox.mul( jitter ) );
	const dy13 = Pfy.x.add( o13.oy.mul( jitter ) );
	const dz13 = Pfz.z.add( o13.oz.mul( jitter ) );

	const dx21 = Pfx.add( o21.ox.mul( jitter ) );
	const dy21 = Pfy.y.add( o21.oy.mul( jitter ) );
	const dz21 = Pfz.x.add( o21.oz.mul( jitter ) );

	const dx22 = Pfx.add( o22.ox.mul( jitter ) );
	const dy22 = Pfy.y.add( o22.oy.mul( jitter ) );
	const dz22 = Pfz.y.add( o22.oz.mul( jitter ) );

	const dx23 = Pfx.add( o23.ox.mul( jitter ) );
	const dy23 = Pfy.y.add( o23.oy.mul( jitter ) );
	const dz23 = Pfz.z.add( o23.oz.mul( jitter ) );

	const dx31 = Pfx.add( o31.ox.mul( jitter ) );
	const dy31 = Pfy.z.add( o31.oy.mul( jitter ) );
	const dz31 = Pfz.x.add( o31.oz.mul( jitter ) );

	const dx32 = Pfx.add( o32.ox.mul( jitter ) );
	const dy32 = Pfy.z.add( o32.oy.mul( jitter ) );
	const dz32 = Pfz.y.add( o32.oz.mul( jitter ) );

	const dx33 = Pfx.add( o33.ox.mul( jitter ) );
	const dy33 = Pfy.z.add( o33.oy.mul( jitter ) );
	const dz33 = Pfz.z.add( o33.oz.mul( jitter ) );

	let d11 = dx11.mul( dx11 ).add( dy11.mul( dy11 ) ).add( dz11.mul( dz11 ) );
	let d12 = dx12.mul( dx12 ).add( dy12.mul( dy12 ) ).add( dz12.mul( dz12 ) );
	let d13 = dx13.mul( dx13 ).add( dy13.mul( dy13 ) ).add( dz13.mul( dz13 ) );
	let d21 = dx21.mul( dx21 ).add( dy21.mul( dy21 ) ).add( dz21.mul( dz21 ) );
	let d22 = dx22.mul( dx22 ).add( dy22.mul( dy22 ) ).add( dz22.mul( dz22 ) );
	let d23 = dx23.mul( dx23 ).add( dy23.mul( dy23 ) ).add( dz23.mul( dz23 ) );
	let d31 = dx31.mul( dx31 ).add( dy31.mul( dy31 ) ).add( dz31.mul( dz31 ) );
	let d32 = dx32.mul( dx32 ).add( dy32.mul( dy32 ) ).add( dz32.mul( dz32 ) );
	let d33 = dx33.mul( dx33 ).add( dy33.mul( dy33 ) ).add( dz33.mul( dz33 ) );

	// Sort out both F1 and F2 (ENABLE_COMPLEX_VERSION's F1-only branch
	// dropped, see the file header comment)
	const d1a = min( d11, d12 );
	d12 = max( d11, d12 );
	d11 = min( d1a, d13 ); // smallest now not in d12 or d13
	d13 = max( d1a, d13 );
	d12 = min( d12, d13 ); // 2nd smallest now not in d13

	const d2a = min( d21, d22 );
	d22 = max( d21, d22 );
	d21 = min( d2a, d23 ); // smallest now not in d22 or d23
	d23 = max( d2a, d23 );
	d22 = min( d22, d23 ); // 2nd smallest now not in d23

	const d3a = min( d31, d32 );
	d32 = max( d31, d32 );
	d31 = min( d3a, d33 ); // smallest now not in d32 or d33
	d33 = max( d3a, d33 );
	d32 = min( d32, d33 ); // 2nd smallest now not in d33

	const da = min( d11, d21 );
	d21 = max( d11, d21 );

	// d11 needs partial-component (swizzle) writes from here on, so it
	// becomes a genuine mutable TSL variable -- every other d** above is a
	// plain JS binding rebound to a new whole-vector value each time, which
	// is enough for TSL node graphs (no in-place mutation needed for those)
	const d11v = min( da, d31 ).toVar(); // smallest now in d11
	d31 = max( da, d31 ); // 2nd smallest now not in d31

	d11v.xy = d11v.x.lessThan( d11v.y ).select( d11v.xy, d11v.yx );
	d11v.xz = d11v.x.lessThan( d11v.z ).select( d11v.xz, d11v.zx ); // d11.x now smallest

	d12 = min( d12, d21 ); // 2nd smallest now not in d21
	d12 = min( d12, d22 ); // nor in d22
	d12 = min( d12, d31 ); // nor in d31
	d12 = min( d12, d32 ); // nor in d32
	d11v.yz = min( d11v.yz, d12.xy ); // nor in d12.yz
	d11v.y = min( d11v.y, d12.z ); // only two more to go
	d11v.y = min( d11v.y, d11v.z ); // done

	return sqrt( d11v.xy ); // F1, F2

}
