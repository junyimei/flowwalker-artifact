
/* Copyright 2010-2014 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * The source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * The Licensed Deliverables contained herein are PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and are being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  THEY ARE
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and are provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

/* ====================================================================
 * Copyright (2024) Bytedance Ltd. and/or its affiliates
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ====================================================================
 */
#pragma once

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "util.cuh"
#include <stdio.h>
#include <stdlib.h>

typedef struct curandStateArr
{
    curandState_t state_arr[BLOCK_SIZE];
} curandStateArr;

typedef struct myrandState
{
    unsigned int d, v[5];
} myrandState;

typedef struct myrandStateArr
{
    unsigned int d[BLOCK_SIZE];
    unsigned int v[5][BLOCK_SIZE];
} myrandStateArr;

typedef struct wgrandState
{
    unsigned long long next_random;
} wgrandState;

typedef struct wgrandStateArr
{
    wgrandState state_arr[BLOCK_SIZE];
} wgrandStateArr;

template <int N>
QUALIFIERS void __curand_matvec_inplace_arr(myrandStateArr *state, unsigned int *matrix)
{
    int tid = threadIdx.x;
    unsigned int result[N] = {0};
    for (int i = 0; i < N; i++)
    {
#ifdef __CUDA_ARCH__
#pragma unroll 16
#endif
        for (int j = 0; j < 32; j++)
        {
            if (state->v[i][tid] & (1 << j))
            {
                for (int k = 0; k < N; k++)
                {
                    result[k] ^= matrix[N * (i * 32 + j) + k];
                }
            }
        }
    }
    for (int i = 0; i < N; i++)
    {
        state->v[i][tid] = result[i];
    }
}

template <typename T, int N>
QUALIFIERS void arr_skipahead_sequence_inplace(unsigned long long x, T *state)
{
    int matrix_num = 0;
    while (x)
    {
        for (unsigned int t = 0; t < (x & PRECALC_BLOCK_MASK); t++)
        {
#ifdef __CUDA_ARCH__
            __curand_matvec_inplace_arr<N>(state, precalc_xorwow_matrix[matrix_num]);
#else
            __curand_matvec_inplace_arr<N>(state, precalc_xorwow_matrix_host[matrix_num]);
#endif
        }
        x >>= PRECALC_BLOCK_SIZE;
        matrix_num++;
    }
    /* No update of state->d needed, guaranteed to be a multiple of 2^32 */
}

template <typename T, int N>
QUALIFIERS void arr_skipahead_inplace(const unsigned long long x, T *state)
{
    int tid = threadIdx.x;
    unsigned long long p = x;
    int matrix_num = 0;
    while (p)
    {
        for (unsigned int t = 0; t < (p & PRECALC_BLOCK_MASK); t++)
        {
#ifdef __CUDA_ARCH__
            __curand_matvec_inplace_arr<N>(state, precalc_xorwow_offset_matrix[matrix_num]);
#else
            __curand_matvec_inplace_arr<N>(state, precalc_xorwow_offset_matrix_host[matrix_num]);
#endif
        }
        p >>= PRECALC_BLOCK_SIZE;
        matrix_num++;
    }
    state->d[tid] += 362437 * (unsigned int)x;
}

QUALIFIERS void NextValue(wgrandState *state)
{
    state->next_random = state->next_random * (unsigned long long)13173779397737131ULL + 1023456798976543201ULL;
}

QUALIFIERS unsigned int myrand(myrandState *state)
{
    unsigned int t;
    t = (state->v[0] ^ (state->v[0] >> 2));
    state->v[0] = state->v[1];
    state->v[1] = state->v[2];
    state->v[2] = state->v[3];
    state->v[3] = state->v[4];
    state->v[4] = (state->v[4] ^ (state->v[4] << 4)) ^ (t ^ (t << 1));
    state->d += 362437;
    return state->v[4] + state->d;
}
QUALIFIERS unsigned int myrand(myrandStateArr *state)
{
    int tid = threadIdx.x;
    unsigned int t;
    t = (state->v[0][tid] ^ (state->v[0][tid] >> 2));
    state->v[0][tid] = state->v[1][tid];
    state->v[1][tid] = state->v[2][tid];
    state->v[2][tid] = state->v[3][tid];
    state->v[3][tid] = state->v[4][tid];
    state->v[4][tid] = (state->v[4][tid] ^ (state->v[4][tid] << 4)) ^ (t ^ (t << 1));
    state->d[tid] += 362437;
    return state->v[4][tid] + state->d[tid];
}

QUALIFIERS unsigned int myrand(curandState *state)
{
    return curand(state);
}

QUALIFIERS unsigned int myrand(curandStateArr *state)
{
    int tid = threadIdx.x;
    return curand(&state->state_arr[tid]);
}

QUALIFIERS unsigned int myrand(wgrandState *state)
{
    unsigned int ret_value = (unsigned int)(state->next_random & 0x7fffffffULL);
    NextValue(state);
    return ret_value;
}

QUALIFIERS unsigned int myrand(wgrandStateArr *state)
{
    int tid = threadIdx.x;
    unsigned int ret_value = (unsigned int)(state->state_arr[tid].next_random & 0x7fffffffULL);
    NextValue(&state->state_arr[tid]);
    return ret_value;
}

QUALIFIERS float myrand_uniform(myrandState *state)
{
    return _curand_uniform(myrand(state));
}

QUALIFIERS float myrand_uniform(myrandStateArr *state)
{
    return _curand_uniform(myrand(state));
}

QUALIFIERS float myrand_uniform(curandState *state)
{
    return _curand_uniform(curand(state));
}

QUALIFIERS float myrand_uniform(curandStateArr *state)
{
    int tid = threadIdx.x;
    return _curand_uniform(curand(&state->state_arr[tid]));
}

QUALIFIERS float myrand_uniform(wgrandState *state)
{
    float max = 1.0f, min = 0.0f;
    unsigned int value = state->next_random & 0xffffff;
    auto ret_value = (float)value;
    ret_value /= 0xffffffL;
    ret_value *= (max - min);
    ret_value += min;
    NextValue(state);
    return ret_value;
}

QUALIFIERS float myrand_uniform(wgrandStateArr *state)
{
    int tid = threadIdx.x;
    float max = 1.0f, min = 0.0f;
    unsigned int value = state->state_arr[tid].next_random & 0xffffff;
    auto ret_value = (float)value;
    ret_value /= 0xffffffL;
    ret_value *= (max - min);
    ret_value += min;
    NextValue(&state->state_arr[tid]);
    return ret_value;
}

QUALIFIERS void myrand_init(unsigned long long seed,
                            unsigned long long subsequence,
                            unsigned long long offset,
                            myrandState *state)
{
    // Break up seed, apply salt
    // Constants are arbitrary nonzero values
    unsigned int s0 = ((unsigned int)seed) ^ 0xaad26b49UL;
    unsigned int s1 = (unsigned int)(seed >> 32) ^ 0xf7dcefddUL;
    // Simple multiplication to mix up bits
    // Constants are arbitrary odd values
    unsigned int t0 = 1099087573UL * s0;
    unsigned int t1 = 2591861531UL * s1;
    state->d = 6615241 + t1 + t0;
    state->v[0] = 123456789UL + t0;
    state->v[1] = 362436069UL ^ t0;
    state->v[2] = 521288629UL + t1;
    state->v[3] = 88675123UL ^ t1;
    state->v[4] = 5783321UL + t0;

    _skipahead_sequence_inplace<myrandState, 5>(subsequence, state);
    _skipahead_inplace<myrandState, 5>(offset, state);
}

QUALIFIERS void myrand_init(unsigned long long seed,
                            unsigned long long subsequence,
                            unsigned long long offset,
                            myrandStateArr *state)
{
    int tid = threadIdx.x;
    // Break up seed, apply salt
    // Constants are arbitrary nonzero values
    unsigned int s0 = ((unsigned int)seed) ^ 0xaad26b49UL;
    unsigned int s1 = (unsigned int)(seed >> 32) ^ 0xf7dcefddUL;
    // Simple multiplication to mix up bits
    // Constants are arbitrary odd values
    unsigned int t0 = 1099087573UL * s0;
    unsigned int t1 = 2591861531UL * s1;
    state->d[tid] = 6615241 + t1 + t0;
    state->v[0][tid] = 123456789UL + t0;
    state->v[1][tid] = 362436069UL ^ t0;
    state->v[2][tid] = 521288629UL + t1;
    state->v[3][tid] = 88675123UL ^ t1;
    state->v[4][tid] = 5783321UL + t0;

    arr_skipahead_sequence_inplace<myrandStateArr, 5>(subsequence, state);
    arr_skipahead_inplace<myrandStateArr, 5>(offset, state);
}

QUALIFIERS void myrand_init(unsigned long long seed,
                            unsigned long long subsequence,
                            unsigned long long offset,
                            curandStateArr *state)
{
    int tid = threadIdx.x;
    curand_init(seed, subsequence, offset, &state->state_arr[tid]);
}

QUALIFIERS void myrand_init(unsigned long long seed,
                            unsigned long long subsequence,
                            unsigned long long offset,
                            wgrandState *state)
{
    state->next_random = seed + subsequence + offset;
    state->next_random ^= state->next_random >> 33U;
    state->next_random *= 0xff51afd7ed558ccdUL;
    state->next_random ^= state->next_random >> 33U;
    state->next_random *= 0xc4ceb9fe1a85ec53UL;
    state->next_random ^= state->next_random >> 33U;
}

QUALIFIERS void myrand_init(unsigned long long seed,
                            unsigned long long subsequence,
                            unsigned long long offset,
                            wgrandStateArr *state)
{
    int tid = threadIdx.x;
    state->state_arr[tid].next_random = seed + subsequence + offset;
    state->state_arr[tid].next_random ^= state->state_arr[tid].next_random >> 33U;
    state->state_arr[tid].next_random *= 0xff51afd7ed558ccdUL;
    state->state_arr[tid].next_random ^= state->state_arr[tid].next_random >> 33U;
    state->state_arr[tid].next_random *= 0xc4ceb9fe1a85ec53UL;
    state->state_arr[tid].next_random ^= state->state_arr[tid].next_random >> 33U;
}
