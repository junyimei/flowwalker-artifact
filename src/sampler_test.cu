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
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <gflags/gflags.h>
#include <stdio.h>

#include <algorithm>
#include <iterator>
#include <map>

#include "app.cuh"
#include "sampler.cuh"
#include "util.cuh"

#define TEST_BLOCK_SIZE 512
// #define AVG

DEFINE_int32(groupsize, 32, "The number of threads in a sampler");
DEFINE_int32(gran, 32, "The number of elements processed by each sampler.");
DEFINE_int32(type, 0, "Sample type");

__global__ void init_state(curandState* state) {
  /*
    init random number generator state
  */
  int gid = (blockDim.x * blockIdx.x) + threadIdx.x;
  curand_init(1337, gid, 0, state + gid);
}

__global__ void init_state(myrandStateArr* state) {
  /*
    init random number generator state
  */
  int gid = (blockDim.x * blockIdx.x) + threadIdx.x;
  // curand_init(1337, gid, 0, state + gid);
  myrand_init(1337, gid, 0, state + blockIdx.x);
}

__global__ void thread_myrand(weight_t* weights, int size, vtx_t* res,
                              int group_size, int granularity, int* task_list) {
  // int total_threads = blockDim.x * gridDim.x;
  int tid = threadIdx.x;
  int gid = tid + blockDim.x * blockIdx.x;

  int worker_per_blk = BLOCK_SIZE / group_size;
  int worker_id = gid / group_size;

  int num_tasks = size / granularity;
  __shared__ myrandStateArr state;
  myrand_init(1337, gid, 0, &state);
  // __shared__ curandState shared_state[BLOCK_SIZE];

  for (int i = worker_id; i < num_tasks; i += gridDim.x * worker_per_blk) {
    int task_id = task_list[i];
    int selected_id = -1;
    weight_t* local_weights = weights + task_id * granularity;

    selected_id = test_thread_sampler(local_weights, granularity, &state);

    if (tid % group_size == 0) res[task_id] = selected_id;
    __syncthreads();
  }
}

#ifdef AVG
__global__ void warp_myrand(weight_t* weights, int size, vtx_t* res,
                            int group_size, int granularity, int* task_list,
                            u64* clk)
#else
__global__ void warp_myrand(weight_t* weights, int size, vtx_t* res,
                            int group_size, int granularity, int* task_list)
#endif
{
  // int total_threads = blockDim.x * gridDim.x;
  int tid = threadIdx.x;
  int gid = tid + blockDim.x * blockIdx.x;

  int worker_per_blk = BLOCK_SIZE / group_size;
  int worker_id = gid / group_size;

  int num_tasks = size / granularity;
  __shared__ myrandStateArr state;
  myrand_init(1337, gid, 0, &state);
  // __shared__ curandState shared_state[BLOCK_SIZE];

  for (int i = worker_id; i < num_tasks; i += gridDim.x * worker_per_blk) {
    int task_id = task_list[i];
    int selected_id = -1;
    weight_t* local_weights = weights + task_id * granularity;

#ifdef AVG
    selected_id =
        test_warp_sampler(local_weights, granularity, &state, clk + task_id);
#else
    selected_id = test_warp_sampler(local_weights, granularity, &state);
#endif

    if (tid % group_size == 0) res[task_id] = selected_id;
    __syncthreads();
  }
}

#ifdef AVG
__global__ void block_myrand(weight_t* weights, int size, vtx_t* res,
                             int group_size, int granularity, int* task_list,
                             u64* clk)
#else
__global__ void block_myrand(weight_t* weights, int size, vtx_t* res,
                             int group_size, int granularity, int* task_list)
#endif
{
  // int total_threads = blockDim.x * gridDim.x;
  int tid = threadIdx.x;
  int gid = tid + blockDim.x * blockIdx.x;

  int worker_per_blk = BLOCK_SIZE / group_size;
  int worker_id = gid / group_size;

  int num_tasks = size / granularity;
  __shared__ myrandStateArr state;
  myrand_init(1337, gid, 0, &state);
  // __shared__ curandState shared_state[BLOCK_SIZE];

  for (int i = worker_id; i < num_tasks; i += gridDim.x * worker_per_blk) {
    int task_id = task_list[i];
    int selected_id = -1;
    weight_t* local_weights = weights + task_id * granularity;

#ifdef AVG
    selected_id =
        test_block_sampler(local_weights, granularity, &state, clk + task_id);
#else
    selected_id = test_block_sampler(local_weights, granularity, &state);
#endif

    if (tid % group_size == 0) res[task_id] = selected_id;
    __syncthreads();
  }
}

__global__ void warp_oneloop(weight_t* weights, int size, vtx_t* res,
                             int group_size, int granularity, int* task_list) {
  // int total_threads = blockDim.x * gridDim.x;
  int tid = threadIdx.x;
  int gid = tid + blockDim.x * blockIdx.x;

  int worker_per_blk = BLOCK_SIZE / group_size;
  int worker_id = gid / group_size;

  int num_tasks = size / granularity;
  __shared__ myrandStateArr state;
  myrand_init(1337, gid, 0, &state);
  // __shared__ curandState shared_state[BLOCK_SIZE];

  for (int i = worker_id; i < num_tasks; i += gridDim.x * worker_per_blk) {
    int task_id = task_list[i];
    int selected_id = -1;
    weight_t* local_weights = weights + task_id * granularity;

    selected_id = test_warp_sampler_oneloop(local_weights, granularity, &state);

    if (tid % group_size == 0) res[task_id] = selected_id;
    __syncthreads();
  }
}

__global__ void block_oneloop(weight_t* weights, int size, vtx_t* res,
                              int group_size, int granularity, int* task_list) {
  // int total_threads = blockDim.x * gridDim.x;
  int tid = threadIdx.x;
  int gid = tid + blockDim.x * blockIdx.x;

  int worker_per_blk = BLOCK_SIZE / group_size;
  int worker_id = gid / group_size;

  int num_tasks = size / granularity;
  __shared__ myrandStateArr state;
  myrand_init(1337, gid, 0, &state);
  // __shared__ curandState shared_state[BLOCK_SIZE];

  for (int i = worker_id; i < num_tasks; i += gridDim.x * worker_per_blk) {
    int task_id = task_list[i];
    int selected_id = -1;
    weight_t* local_weights = weights + task_id * granularity;

    selected_id =
        test_block_sampler_oneloop(local_weights, granularity, &state);

    if (tid % group_size == 0) res[task_id] = selected_id;
    __syncthreads();
  }
}

__global__ void thread_curand(weight_t* weights, int size, vtx_t* res,
                              int group_size, int granularity, int* task_list) {
  // int total_threads = blockDim.x * gridDim.x;
  int tid = threadIdx.x;
  int gid = threadIdx.x + blockDim.x * blockIdx.x;

  int worker_per_blk = BLOCK_SIZE / group_size;
  int worker_id = gid / group_size;

  int num_tasks = size / granularity;
  // __shared__ myrandStateArr state;
  // myrand_init(1337, gid, 0, &state);
  __shared__ curandState state[BLOCK_SIZE];
  curand_init(1337, gid, 0, &state[tid]);

  for (int i = worker_id; i < num_tasks; i += gridDim.x * worker_per_blk) {
    int task_id = task_list[i];
    int selected_id = -1;
    weight_t* local_weights = weights + task_id * granularity;

    selected_id = test_thread_sampler(local_weights, granularity, state + tid);

    if (threadIdx.x % group_size == 0) res[task_id] = selected_id;
    __syncthreads();
  }
}

__global__ void warp_curand(weight_t* weights, int size, vtx_t* res,
                            int group_size, int granularity, int* task_list) {
  // int total_threads = blockDim.x * gridDim.x;
  int tid = threadIdx.x;
  int gid = threadIdx.x + blockDim.x * blockIdx.x;

  int worker_per_blk = BLOCK_SIZE / group_size;
  int worker_id = gid / group_size;

  int num_tasks = size / granularity;
  // __shared__ myrandStateArr state;
  // myrand_init(1337, gid, 0, &state);
  __shared__ curandState state[BLOCK_SIZE];
  curand_init(1337, gid, 0, &state[tid]);

  for (int i = worker_id; i < num_tasks; i += gridDim.x * worker_per_blk) {
    int task_id = task_list[i];
    int selected_id = -1;
    weight_t* local_weights = weights + task_id * granularity;

    selected_id = test_warp_sampler(local_weights, granularity, state + tid);

    if (threadIdx.x % group_size == 0) res[task_id] = selected_id;
    __syncthreads();
  }
}

__global__ void block_curand(weight_t* weights, int size, vtx_t* res,
                             int group_size, int granularity, int* task_list) {
  // int total_threads = blockDim.x * gridDim.x;
  int tid = threadIdx.x;
  int gid = threadIdx.x + blockDim.x * blockIdx.x;

  int worker_per_blk = BLOCK_SIZE / group_size;
  int worker_id = gid / group_size;

  int num_tasks = size / granularity;
  // __shared__ myrandStateArr state;
  // myrand_init(1337, gid, 0, &state);
  __shared__ curandState state[BLOCK_SIZE];
  curand_init(1337, gid, 0, &state[tid]);

  for (int i = worker_id; i < num_tasks; i += gridDim.x * worker_per_blk) {
    int task_id = task_list[i];
    int selected_id = -1;
    weight_t* local_weights = weights + task_id * granularity;

    selected_id = test_block_sampler(local_weights, granularity, state + tid);

    if (threadIdx.x % group_size == 0) res[task_id] = selected_id;
    __syncthreads();
  }
}

__global__ void its_warp(weight_t* weights, int size, vtx_t* res,
                         int group_size, int granularity, int* task_list,
                         float* prob) {
  // int total_threads = blockDim.x * gridDim.x;
  int tid = threadIdx.x;
  int gid = tid + blockDim.x * blockIdx.x;

  int worker_per_blk = BLOCK_SIZE / group_size;
  int worker_id = gid / group_size;

  int num_tasks = size / granularity;
  __shared__ myrandStateArr state;
  myrand_init(1337, gid, 0, &state);
  // __shared__ curandState shared_state[BLOCK_SIZE];

  for (int i = worker_id; i < num_tasks; i += gridDim.x * worker_per_blk) {
    int task_id = task_list[i];
    int selected_id = -1;
    weight_t* local_weights = weights + task_id * granularity;
    float* local_prob = prob + task_id * granularity;

    selected_id =
        its_warp_sampler_direct(local_weights, granularity, &state, local_prob);

    if (tid % group_size == 0) res[task_id] = selected_id;
    __syncthreads();
  }
}

__global__ void its_block(weight_t* weights, int size, vtx_t* res,
                          int group_size, int granularity, int* task_list,
                          float* prob) {
  // int total_threads = blockDim.x * gridDim.x;
  int tid = threadIdx.x;
  int gid = tid + blockDim.x * blockIdx.x;

  int worker_per_blk = BLOCK_SIZE / group_size;
  int worker_id = gid / group_size;

  int num_tasks = size / granularity;
  __shared__ myrandStateArr state;
  myrand_init(1337, gid, 0, &state);
  // __shared__ curandState shared_state[BLOCK_SIZE];

  for (int i = worker_id; i < num_tasks; i += gridDim.x * worker_per_blk) {
    int task_id = task_list[i];
    int selected_id = -1;
    weight_t* local_weights = weights + task_id * granularity;
    float* local_prob = prob + task_id * granularity;

    selected_id = its_block_sampler_direct(local_weights, granularity, &state,
                                           local_prob);

    if (tid % group_size == 0) res[task_id] = selected_id;
    __syncthreads();
  }
}

__global__ void alias_warp(weight_t* weights, int size, vtx_t* res,
                           int group_size, int granularity, int* task_list,
                           float* prob, vtx_t* alias, vtx_t* large,
                           vtx_t* small) {
  // int total_threads = blockDim.x * gridDim.x;
  int tid = threadIdx.x;
  int gid = tid + blockDim.x * blockIdx.x;

  int worker_per_blk = BLOCK_SIZE / group_size;
  int worker_id = gid / group_size;

  int num_tasks = size / granularity;
  __shared__ myrandStateArr state;
  myrand_init(1337, gid, 0, &state);
  // __shared__ curandState shared_state[BLOCK_SIZE];

  for (int i = worker_id; i < num_tasks; i += gridDim.x * worker_per_blk) {
    int task_id = task_list[i];
    int selected_id = -1;
    weight_t* local_weights = weights + task_id * granularity;
    float* local_prob = prob + task_id * granularity;
    vtx_t* local_alias = alias + task_id * granularity;
    vtx_t* local_large = large + task_id * granularity;
    vtx_t* local_small = small + task_id * granularity;

    selected_id = naive_alias_warp_sampler(local_weights, granularity, &state,
                                           local_alias, local_prob, local_large,
                                           local_small);

    if (tid % group_size == 0) res[task_id] = selected_id;
    __syncthreads();
  }
}

__global__ void alias_block(weight_t* weights, int size, vtx_t* res,
                            int group_size, int granularity, int* task_list,
                            float* prob, vtx_t* alias, vtx_t* large,
                            vtx_t* small) {
  // int total_threads = blockDim.x * gridDim.x;
  int tid = threadIdx.x;
  int gid = tid + blockDim.x * blockIdx.x;

  int worker_per_blk = BLOCK_SIZE / group_size;
  int worker_id = gid / group_size;

  int num_tasks = size / granularity;
  __shared__ myrandStateArr state;
  myrand_init(1337, gid, 0, &state);
  // __shared__ curandState shared_state[BLOCK_SIZE];

  for (int i = worker_id; i < num_tasks; i += gridDim.x * worker_per_blk) {
    int task_id = task_list[i];
    int selected_id = -1;
    weight_t* local_weights = weights + task_id * granularity;
    float* local_prob = prob + task_id * granularity;
    vtx_t* local_alias = alias + task_id * granularity;
    vtx_t* local_large = large + task_id * granularity;
    vtx_t* local_small = small + task_id * granularity;

    selected_id = naive_alias_block_sampler(local_weights, granularity, &state,
                                            local_alias, local_prob,
                                            local_large, local_small);

    if (tid % group_size == 0) res[task_id] = selected_id;
    __syncthreads();
  }
}

__global__ void thread_globalrand(weight_t* weights, int size, vtx_t* res,
                                  int group_size, int granularity,
                                  int* task_list, curandState* state) {
  // int total_threads = blockDim.x * gridDim.x;
  int tid = threadIdx.x;
  int gid = tid + blockDim.x * blockIdx.x;

  int worker_per_blk = BLOCK_SIZE / group_size;
  int worker_id = gid / group_size;

  int num_tasks = size / granularity;

  for (int i = worker_id; i < num_tasks; i += gridDim.x * worker_per_blk) {
    int task_id = task_list[i];
    int selected_id = -1;
    weight_t* local_weights = weights + task_id * granularity;

    selected_id = test_thread_sampler(local_weights, granularity, state + gid);

    if (threadIdx.x % group_size == 0) res[task_id] = selected_id;
    __syncthreads();
  }
}
__global__ void warp_globalrand(weight_t* weights, int size, vtx_t* res,
                                int group_size, int granularity, int* task_list,
                                curandState* state) {
  // int total_threads = blockDim.x * gridDim.x;
  int tid = threadIdx.x;
  int gid = tid + blockDim.x * blockIdx.x;

  int worker_per_blk = BLOCK_SIZE / group_size;
  int worker_id = gid / group_size;

  int num_tasks = size / granularity;

  for (int i = worker_id; i < num_tasks; i += gridDim.x * worker_per_blk) {
    int task_id = task_list[i];
    int selected_id = -1;
    weight_t* local_weights = weights + task_id * granularity;

    selected_id = test_warp_sampler(local_weights, granularity, state + gid);

    if (threadIdx.x % group_size == 0) res[task_id] = selected_id;
    __syncthreads();
  }
}
__global__ void block_globalrand(weight_t* weights, int size, vtx_t* res,
                                 int group_size, int granularity,
                                 int* task_list, curandState* state) {
  // int total_threads = blockDim.x * gridDim.x;
  int tid = threadIdx.x;
  int gid = tid + blockDim.x * blockIdx.x;

  int worker_per_blk = BLOCK_SIZE / group_size;
  int worker_id = gid / group_size;

  int num_tasks = size / granularity;

  for (int i = worker_id; i < num_tasks; i += gridDim.x * worker_per_blk) {
    int task_id = task_list[i];
    int selected_id = -1;
    weight_t* local_weights = weights + task_id * granularity;

    selected_id = test_block_sampler(local_weights, granularity, state + gid);

    if (threadIdx.x % group_size == 0) res[task_id] = selected_id;
    __syncthreads();
  }
}

__global__ void thread_globalmyrand(weight_t* weights, int size, vtx_t* res,
                                    int group_size, int granularity,
                                    int* task_list, myrandStateArr* state) {
  // int total_threads = blockDim.x * gridDim.x;
  // int tid = threadIdx.x;
  int bid = blockIdx.x;
  int gid = threadIdx.x + blockDim.x * blockIdx.x;

  int worker_per_blk = BLOCK_SIZE / group_size;
  int worker_id = gid / group_size;

  int num_tasks = size / granularity;

  for (int i = worker_id; i < num_tasks; i += gridDim.x * worker_per_blk) {
    int task_id = task_list[i];
    int selected_id = -1;
    weight_t* local_weights = weights + task_id * granularity;

    selected_id = test_thread_sampler(local_weights, granularity, state + bid);

    if (threadIdx.x % group_size == 0) res[task_id] = selected_id;
    __syncthreads();
  }
}

__global__ void warp_globalmyrand(weight_t* weights, int size, vtx_t* res,
                                  int group_size, int granularity,
                                  int* task_list, myrandStateArr* state) {
  // int total_threads = blockDim.x * gridDim.x;
  // int tid = threadIdx.x;
  int bid = blockIdx.x;
  int gid = threadIdx.x + blockDim.x * blockIdx.x;

  int worker_per_blk = BLOCK_SIZE / group_size;
  int worker_id = gid / group_size;

  int num_tasks = size / granularity;

  for (int i = worker_id; i < num_tasks; i += gridDim.x * worker_per_blk) {
    int task_id = task_list[i];
    int selected_id = -1;
    weight_t* local_weights = weights + task_id * granularity;

    selected_id = test_warp_sampler(local_weights, granularity, state + bid);

    if (threadIdx.x % group_size == 0) res[task_id] = selected_id;
    __syncthreads();
  }
}
__global__ void block_globalmyrand(weight_t* weights, int size, vtx_t* res,
                                   int group_size, int granularity,
                                   int* task_list, myrandStateArr* state) {
  // int total_threads = blockDim.x * gridDim.x;
  // int tid = threadIdx.x;
  int bid = blockIdx.x;
  int gid = threadIdx.x + blockDim.x * blockIdx.x;

  int worker_per_blk = BLOCK_SIZE / group_size;
  int worker_id = gid / group_size;

  int num_tasks = size / granularity;

  for (int i = worker_id; i < num_tasks; i += gridDim.x * worker_per_blk) {
    int task_id = task_list[i];
    int selected_id = -1;
    weight_t* local_weights = weights + task_id * granularity;

    selected_id = test_block_sampler(local_weights, granularity, state + bid);

    if (threadIdx.x % group_size == 0) res[task_id] = selected_id;
    __syncthreads();
  }
}

#ifdef AVG
__global__ void rjs_warp(weight_t* weights, int size, vtx_t* res,
                         int group_size, int granularity, int* task_list,
                         u64* clk)
#else
__global__ void rjs_warp(weight_t* weights, int size, vtx_t* res,
                         int group_size, int granularity, int* task_list)
#endif
{
  // int total_threads = blockDim.x * gridDim.x;
  int tid = threadIdx.x;
  int gid = tid + blockDim.x * blockIdx.x;

  int worker_per_blk = BLOCK_SIZE / group_size;
  int worker_id = gid / group_size;

  int num_tasks = size / granularity;
  __shared__ myrandStateArr state;
  myrand_init(1337, gid, 0, &state);
  // __shared__ curandState shared_state[BLOCK_SIZE];

  for (int i = worker_id; i < num_tasks; i += gridDim.x * worker_per_blk) {
    int task_id = task_list[i];
    int selected_id = -1;
    weight_t* local_weights = weights + task_id * granularity;

#ifdef AVG
    selected_id =
        rjs_warp_sampler(local_weights, granularity, &state, clk + task_id);
#else
    selected_id = rjs_warp_sampler(local_weights, granularity, &state);
#endif

    if (tid % group_size == 0) res[task_id] = selected_id;
    __syncthreads();
  }
}

#ifdef AVG
__global__ void rjs_block(weight_t* weights, int size, vtx_t* res,
                          int group_size, int granularity, int* task_list,
                          u64* clk)
#else
__global__ void rjs_block(weight_t* weights, int size, vtx_t* res,
                          int group_size, int granularity, int* task_list)
#endif
{
  // int total_threads = blockDim.x * gridDim.x;
  int tid = threadIdx.x;
  int gid = tid + blockDim.x * blockIdx.x;

  int worker_per_blk = BLOCK_SIZE / group_size;
  int worker_id = gid / group_size;

  int num_tasks = size / granularity;
  __shared__ myrandStateArr state;
  myrand_init(1337, gid, 0, &state);
  // __shared__ curandState shared_state[BLOCK_SIZE];

  for (int i = worker_id; i < num_tasks; i += gridDim.x * worker_per_blk) {
    int task_id = task_list[i];
    int selected_id = -1;
    weight_t* local_weights = weights + task_id * granularity;
#ifdef AVG
    selected_id =
        rjs_block_sampler(local_weights, granularity, &state, clk + task_id);
#else
    selected_id = rjs_block_sampler(local_weights, granularity, &state);
#endif

    if (tid % group_size == 0) res[task_id] = selected_id;
    __syncthreads();
  }
}

__global__ void warp_wgrand(weight_t* weights, int size, vtx_t* res,
                            int group_size, int granularity, int* task_list) {
  // int total_threads = blockDim.x * gridDim.x;
  int tid = threadIdx.x;
  int gid = threadIdx.x + blockDim.x * blockIdx.x;

  int worker_per_blk = BLOCK_SIZE / group_size;
  int worker_id = gid / group_size;

  int num_tasks = size / granularity;
  // __shared__ myrandStateArr state;
  // myrand_init(1337, gid, 0, &state);
  __shared__ wgrandState state[BLOCK_SIZE];
  myrand_init(1337, gid, 0, &state[tid]);

  for (int i = worker_id; i < num_tasks; i += gridDim.x * worker_per_blk) {
    int task_id = task_list[i];
    int selected_id = -1;
    weight_t* local_weights = weights + task_id * granularity;

    selected_id = test_warp_sampler(local_weights, granularity, state + tid);

    if (threadIdx.x % group_size == 0) res[task_id] = selected_id;
    __syncthreads();
  }
}

__global__ void block_wgrand(weight_t* weights, int size, vtx_t* res,
                             int group_size, int granularity, int* task_list) {
  // int total_threads = blockDim.x * gridDim.x;
  int tid = threadIdx.x;
  int gid = threadIdx.x + blockDim.x * blockIdx.x;

  int worker_per_blk = BLOCK_SIZE / group_size;
  int worker_id = gid / group_size;

  int num_tasks = size / granularity;
  // __shared__ myrandStateArr state;
  // myrand_init(1337, gid, 0, &state);
  __shared__ wgrandState state[BLOCK_SIZE];
  myrand_init(1337, gid, 0, &state[tid]);

  for (int i = worker_id; i < num_tasks; i += gridDim.x * worker_per_blk) {
    int task_id = task_list[i];
    int selected_id = -1;
    weight_t* local_weights = weights + task_id * granularity;

    selected_id = test_block_sampler(local_weights, granularity, state + tid);

    if (threadIdx.x % group_size == 0) res[task_id] = selected_id;
    __syncthreads();
  }
}

__global__ void get_exp(weight_t* weights, int size, int granularity,
                        int* task_list, float* exp) {
  int tid = threadIdx.x;
  int gid = tid + blockDim.x * blockIdx.x;

  int num_tasks = size / granularity;
  __shared__ myrandStateArr state;
  myrand_init(1337, gid, 0, &state);
  // __shared__ curandState shared_state[BLOCK_SIZE];

  for (int i = gid; i < num_tasks; i += gridDim.x * BLOCK_SIZE) {
    int task_id = task_list[i];
    weight_t* local_weights = weights + task_id * granularity;

    float max_w = 0;
    float sum_w = 0;
    for (int j = 0; j < granularity; ++j) {
      sum_w += local_weights[j];
      max_w = max(max_w, local_weights[j]);
    }

    exp[task_id] = (granularity * max_w) / sum_w;
  }
}

void test_sampler_helper(weight_t* d_weights, int size, vtx_t* d_res,
                         int num_blocks, int group_size, int sample_type,
                         int granularity) {
  printf(
      "\n\n-----------------------------------------------------------------"
      "----------------------------------------\n");
  printf(
      "Test config: {Num of Elements: %d, Number of Thread Blocks: %d, "
      "Block Size: %d, "
      "Number of Threads in a Sampler: %d, Sampler Type: %d, Number of "
      "Elements Processed "
      "by a Sampler: %d, Number of tasks: %d}\n",
      size, num_blocks, BLOCK_SIZE, group_size, sample_type, granularity,
      size / granularity);

#ifdef AVG
  float fq = get_clk();
  u64* clk;
  cudaMallocManaged(&clk, sizeof(u64) * size);
  cudaMemset(clk, 0, sizeof(u64) * size);
#endif
  // Create CUDA events
  vtx_t* h_res = new vtx_t[size];
  float* prob;
  vtx_t *alias, *large, *small;
  curandState* state;
  myrandStateArr* myrand_state;

  warm_up_gpu<<<num_blocks, BLOCK_SIZE>>>();
  cudaDeviceSynchronize();

  int num_tasks = size / granularity;
  int* task_list = new int[num_tasks];
  for (int i = 0; i < num_tasks; ++i) {
    task_list[i] = i;
  }
  std::random_shuffle(task_list, task_list + num_tasks);
  int* d_task_list;
  cudaMalloc(reinterpret_cast<void**>(&d_task_list), sizeof(int) * num_tasks);
  cudaMemcpy(d_task_list, task_list, sizeof(int) * num_tasks,
             cudaMemcpyHostToDevice);
  delete[] task_list;

  if (sample_type == 5 || sample_type == 6) {
    cudaMalloc(&prob, sizeof(float) * size);
  } else if (sample_type == 7 || sample_type == 8) {
    cudaMalloc(&prob, sizeof(float) * size);
    cudaMalloc(&alias, sizeof(vtx_t) * size);
    cudaMalloc(&large, sizeof(vtx_t) * size);
    cudaMalloc(&small, sizeof(vtx_t) * size);
  } else if (sample_type >= 30) {
    // printf("================\n");
    // printf("size=%llu\n", sizeof(myrandStateArr));
    cudaMalloc(&myrand_state, sizeof(myrandStateArr) * num_blocks);

    init_state<<<num_blocks, BLOCK_SIZE>>>(myrand_state);
    cudaDeviceSynchronize();
  } else if (sample_type >= 20 && sample_type <= 30) {
    // printf("234234\n");
    cudaMalloc(&state, sizeof(curandState) * num_blocks * BLOCK_SIZE);
    init_state<<<num_blocks, BLOCK_SIZE>>>(state);
    cudaDeviceSynchronize();
  }
  float* exp;
  cudaMallocManaged(&exp, sizeof(float) * num_tasks);
  get_exp<<<num_blocks, BLOCK_SIZE>>>(d_weights, size, granularity, d_task_list,
                                      exp);

  cudaDeviceSynchronize();
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);

  if (sample_type == 0) {
    thread_myrand<<<num_blocks, BLOCK_SIZE>>>(
        d_weights, size, d_res, group_size, granularity, d_task_list);
  } else if (sample_type == 1) {
#ifdef AVG
    warp_myrand<<<num_blocks, BLOCK_SIZE>>>(d_weights, size, d_res, group_size,
                                            granularity, d_task_list, clk);
#else
    warp_myrand<<<num_blocks, BLOCK_SIZE>>>(d_weights, size, d_res, group_size,
                                            granularity, d_task_list);
#endif
  } else if (sample_type == 2) {
#ifdef AVG
    block_myrand<<<num_blocks, BLOCK_SIZE>>>(d_weights, size, d_res, group_size,
                                             granularity, d_task_list, clk);
#else
    block_myrand<<<num_blocks, BLOCK_SIZE>>>(d_weights, size, d_res, group_size,
                                             granularity, d_task_list);
#endif
  } else if (sample_type == 3) {
    warp_oneloop<<<num_blocks, BLOCK_SIZE>>>(d_weights, size, d_res, group_size,
                                             granularity, d_task_list);
  } else if (sample_type == 4) {
    block_oneloop<<<num_blocks, BLOCK_SIZE>>>(
        d_weights, size, d_res, group_size, granularity, d_task_list);
  } else if (sample_type == 5) {
    its_warp<<<num_blocks, BLOCK_SIZE>>>(d_weights, size, d_res, group_size,
                                         granularity, d_task_list, prob);
  } else if (sample_type == 6) {
    its_block<<<num_blocks, BLOCK_SIZE>>>(d_weights, size, d_res, group_size,
                                          granularity, d_task_list, prob);
  } else if (sample_type == 7) {
    alias_warp<<<num_blocks, BLOCK_SIZE>>>(d_weights, size, d_res, group_size,
                                           granularity, d_task_list, prob,
                                           alias, large, small);
  } else if (sample_type == 8) {
    alias_block<<<num_blocks, BLOCK_SIZE>>>(d_weights, size, d_res, group_size,
                                            granularity, d_task_list, prob,
                                            alias, large, small);
  } else if (sample_type == 10) {
    thread_curand<<<num_blocks, BLOCK_SIZE>>>(
        d_weights, size, d_res, group_size, granularity, d_task_list);
  } else if (sample_type == 11) {
    warp_curand<<<num_blocks, BLOCK_SIZE>>>(d_weights, size, d_res, group_size,
                                            granularity, d_task_list);
  } else if (sample_type == 12) {
    block_curand<<<num_blocks, BLOCK_SIZE>>>(d_weights, size, d_res, group_size,
                                             granularity, d_task_list);
  } else if (sample_type == 13) {
#ifdef AVG
    rjs_warp<<<num_blocks, BLOCK_SIZE>>>(d_weights, size, d_res, group_size,
                                         granularity, d_task_list, clk);
#else
    rjs_warp<<<num_blocks, BLOCK_SIZE>>>(d_weights, size, d_res, group_size,
                                         granularity, d_task_list);
#endif
  } else if (sample_type == 14) {
    // u64 *cnt;
    // cudaMallocManaged(&cnt, sizeof(u64) * size);
    // cudaMemset(cnt, 0, sizeof(u64) * size);
#ifdef AVG
    rjs_block<<<num_blocks, BLOCK_SIZE>>>(d_weights, size, d_res, group_size,
                                          granularity, d_task_list, clk);
#else
    // rjs_block<<<num_blocks, BLOCK_SIZE>>>(d_weights, size, d_res, group_size,
    // granularity, d_task_list, cnt);
    rjs_block<<<num_blocks, BLOCK_SIZE>>>(d_weights, size, d_res, group_size,
                                          granularity, d_task_list);
#endif
    // cudaDeviceSynchronize();

    // u64 exp_avg = 0, cnt_avg = 0;
    // for (int i = 0; i < num_tasks; i++)
    // {
    //     exp_avg += exp[i];
    //     cnt_avg += cnt[i];
    // }
    // printf("exp_avg=%.3f, cnt_avg=%.3f\n", (double)exp_avg / num_tasks,
    // (double)cnt_avg / num_tasks);
  } else if (sample_type == 20) {
    thread_globalrand<<<num_blocks, BLOCK_SIZE>>>(
        d_weights, size, d_res, group_size, granularity, d_task_list, state);
  } else if (sample_type == 21) {
    warp_globalrand<<<num_blocks, BLOCK_SIZE>>>(
        d_weights, size, d_res, group_size, granularity, d_task_list, state);
  } else if (sample_type == 22) {
    block_globalrand<<<num_blocks, BLOCK_SIZE>>>(
        d_weights, size, d_res, group_size, granularity, d_task_list, state);
  } else if (sample_type == 30) {
    thread_globalmyrand<<<num_blocks, BLOCK_SIZE>>>(d_weights, size, d_res,
                                                    group_size, granularity,
                                                    d_task_list, myrand_state);
  } else if (sample_type == 31) {
    warp_globalmyrand<<<num_blocks, BLOCK_SIZE>>>(d_weights, size, d_res,
                                                  group_size, granularity,
                                                  d_task_list, myrand_state);
  } else if (sample_type == 32) {
    block_globalmyrand<<<num_blocks, BLOCK_SIZE>>>(d_weights, size, d_res,
                                                   group_size, granularity,
                                                   d_task_list, myrand_state);
  } else if (sample_type == 41) {
    warp_wgrand<<<num_blocks, BLOCK_SIZE>>>(d_weights, size, d_res, group_size,
                                            granularity, d_task_list);
  } else if (sample_type == 42) {
    block_wgrand<<<num_blocks, BLOCK_SIZE>>>(d_weights, size, d_res, group_size,
                                             granularity, d_task_list);
  }

  cudaDeviceSynchronize();

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float elapsed_time = 0;
  cudaEventElapsedTime(&elapsed_time, start, stop);

  cudaMemcpy(h_res, d_res, sizeof(int) * size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

#ifdef AVG
  u64 time_sum = 0;
  for (int i = 0; i < num_tasks; i++) {
    time_sum += clk[i];
  }
  printf("Avg time: %.6f ms\n", static_cast<double>(time_sum) / num_tasks / fq);
#endif
  printf("Elapsed time: %.3f ms\n", elapsed_time);
}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  // Test config.
  // total 2GB
  int size = 512 * 1024 * 1024;  // the number of elements.
  // int size = FLAGS_gran;
  int res_size = 512 * 1024 * 1024;
  // int res_size = FLAGS_gran;

  int device;
  cudaDeviceProp prop;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  int n_sm = prop.multiProcessorCount;
  int b_per_sm = 1024 / BLOCK_SIZE;
  int block_num = n_sm * b_per_sm * 2;

  // Allocate device memory
  weight_t* d_weight_array;
  vtx_t* d_res;

  cudaMalloc(&d_weight_array, sizeof(weight_t) * size);
  cudaMalloc(&d_res, sizeof(vtx_t) * res_size);

  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
  // curandGenerate(gen, d_element_array, size);
  curandGenerateUniform(gen, d_weight_array, size);
  // curandGenerateLogNormal(gen, d_weight_array, size, 0, 2);

  // Test warp_sampler as an example
  int group_size = FLAGS_groupsize;  // The number of threads in a sampler.
  int sample_type = FLAGS_type;
  int granularity =
      FLAGS_gran;  // The number of elements processed by each sampler.

  test_sampler_helper(d_weight_array, size, d_res, block_num, group_size,
                      sample_type, granularity);

  cudaDeviceSynchronize();
  // Free device memory
  cudaFree(d_weight_array);
  cudaFree(d_res);

  return 0;
}
