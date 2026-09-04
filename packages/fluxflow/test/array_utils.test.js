// 只验证 kernel 图能不能构建成功、返回的是可调用的 dispatcher——不实际调用
// dispatcher（那需要 tsl_array_n.init() 建立的真实 renderer，vitest/Node 环境
// 没有），真正的扩散/拷贝数值行为要等 live 示例里用 toArray() 验证。

import { describe, it, expect } from 'vitest';
import * as tsl_array_n from 'tsl_array_n';
import { createCopyKernel2, createExtrapolateToRegion2 } from '../src/grid/array_utils.js';

describe( 'array_utils', () => {

	it( 'createCopyKernel2 returns a reusable dispatcher', () => {

		const src = tsl_array_n.array2( 'float', 4, 4 );
		const dst = tsl_array_n.array2( 'float', 4, 4 );

		const copy = createCopyKernel2( src, dst );

		expect( typeof copy ).toBe( 'function' );

	} );

	it( 'createCopyKernel2 infers shape from src when not given explicitly', () => {

		const src = tsl_array_n.array2( 'float', 3, 5 );
		const dst = tsl_array_n.array2( 'float', 3, 5 );

		expect( () => createCopyKernel2( src, dst ) ).not.toThrow();

	} );

	it( 'createExtrapolateToRegion2 returns a run() function, in-place (output === input)', () => {

		const field = tsl_array_n.array2( 'float', 4, 4 );
		const valid = tsl_array_n.array2( 'int', 4, 4 );

		const run = createExtrapolateToRegion2( field, valid, field );

		expect( typeof run ).toBe( 'function' );

	} );

	it( 'createExtrapolateToRegion2 returns a run() function, out-of-place (separate output field)', () => {

		const input = tsl_array_n.array2( 'float', 4, 4 );
		const valid = tsl_array_n.array2( 'int', 4, 4 );
		const output = tsl_array_n.array2( 'float', 4, 4 );

		const run = createExtrapolateToRegion2( input, valid, output );

		expect( typeof run ).toBe( 'function' );

	} );

} );
