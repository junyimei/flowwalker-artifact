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
#include <algorithm>
#include <random>
#include <vector>

#include "gpu_queue.cuh"
#include "gpu_task.cuh"
#include "myrand.cuh"
#include "sampler.cuh"
#include "util.cuh"
// #define ONELOOP
// #define MICRO_BENCH
#define TASK_NUM 64
#define THRESHOLD 1024
enum walk_mode {
  wp,     // warp only static
  wb,     // warp and block static
  twb,    // thread, warp and block
  queue,  // queue
  tw,     // thread and warp static
  twb_dynamic,
  wb_dynamic,
  w_dynamic,
  b_dynamic,
  // wb_dynamic_curand,
  wb_curand,
  wb_dprs,
  wb_dynamic_dprs,
  wb_curand_zprs
};
/*
Walk functions
*/
double walk_test(vtx_t*& result_pool_ptr, gpu_graph* graph,
                 vtx_t* start_points,  // NOLINT
                 int max_depth, int num_walkers, walk_mode type,
                 int* schema = NULL, int schema_len = 0);

double walk_batch(vtx_t*& result_pool, gpu_graph* graph,
                  vtx_t* start_points,  // NOLINT
                  int max_depth, int num_walkers, int batch_size,
                  int* schema = NULL, int schema_len = 0);

/*
Walkers
*/
#ifdef MICRO_BENCH
template <typename walker_t>
__global__ void walker_w(walker_t* walker, vtx_t* start_points,
                         vtx_t* result_pool, TaskAssignments* assign,
                         metrics* counter)
#else
template <typename walker_t>
__global__ void walker_w(walker_t* walker, vtx_t* start_points,
                         vtx_t* result_pool, TaskAssignments* assign)
#endif
{
#ifdef MICRO_BENCH
  counter->block_begin(blockIdx.x);
#endif
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int lid = threadIdx.x % WARP_SIZE;
  int gid = (blockDim.x * blockIdx.x) + tid;
  int wid = threadIdx.x / WARP_SIZE;

  gpu_graph* graph = walker->graph;
  int max_depth = walker->max_depth;

  __shared__ myrandStateArr state;
  myrand_init(1337, gid, 0, &state);

  __shared__ Task tasks[BLOCK_SIZE];
  __shared__ int num_active_tasks;
  __shared__ int walk_id;

  int num_walks = assign[bid].end - assign[bid].begin;

  int task_start = assign[bid].begin;

  vtx_t* local_start_points = start_points + task_start;
  vtx_t* local_result_pool = result_pool + task_start * max_depth;

  if (tid < num_walks) {
    local_result_pool[(u64)tid * max_depth] = local_start_points[tid];
    tasks[tid].init(graph, local_start_points, tid);
  } else {
    tasks[tid].length = -1;
  }

  if (tid == 0) {
    walk_id = min(BLOCK_SIZE, num_walks);
    num_active_tasks = num_walks;
  }
  __syncthreads();

#ifdef MICRO_BENCH
  counter->sample_begin(bid);
#endif
  while (num_active_tasks > 0) {
    for (int local_task_id = wid; local_task_id < BLOCK_SIZE;
         local_task_id += WARP_PER_BLK) {
      Task& task = tasks[local_task_id];
      if (task.length >= 0) {
#ifdef MICRO_BENCH
        u64 start = clock64();
#endif
#ifdef ONELOOP
        vtx_t selected = sampler_warp_oneloop<walker_t, myrandStateArr>(
            walker, &task, &state);
        // __syncwarp(FULL_WARP_MASK);
#else
        vtx_t selected =
            sampler_warp<walker_t, myrandStateArr>(walker, &task, &state);
#endif
#ifdef MICRO_BENCH
        counter->add_warp(bid, start, task.degree);
#endif
        if (lid == 0) {
          // early stop
          if (selected == -1) {
            task.length = max_depth;
          } else {
            selected = graph->adjncy[task.neighbor_offset + selected];
            local_result_pool[(u64)task.walker_id * max_depth + task.length] =
                selected;
            task.update(graph, selected);
          }

          // task.print();
        }
      }
      __syncwarp(FULL_WARP_MASK);
    }
    __syncthreads();
    // remove finished tasks and add fetch new tasks
    if (walker->is_stop(tasks[tid].length, &state)) {
      atomicAdd(&num_active_tasks, -1);
      int local_walk_id = atomicAdd(&walk_id, 1);
      if (local_walk_id < num_walks) {
        local_result_pool[(u64)local_walk_id * max_depth] =
            local_start_points[local_walk_id];
        tasks[tid].init(graph, local_start_points, local_walk_id);
      } else {
        tasks[tid].length = -1;
      }
    }
    __syncthreads();
  }
#ifdef MICRO_BENCH
  counter->block_end(bid);
#endif
}

#ifdef MICRO_BENCH
template <typename walker_t>
__global__ void walker_wb(walker_t* walker, vtx_t* start_points,
                          vtx_t* result_pool, TaskAssignments* assign,
                          metrics* counter)
#else
template <typename walker_t>
__global__ void walker_wb(walker_t* walker, vtx_t* start_points,
                          vtx_t* result_pool, TaskAssignments* assign)
#endif
{
#ifdef MICRO_BENCH
  counter->block_begin(blockIdx.x);
#endif
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int lid = threadIdx.x % WARP_SIZE;
  int gid = (blockDim.x * blockIdx.x) + tid;
  int wid = threadIdx.x / WARP_SIZE;

  gpu_graph* graph = walker->graph;
  int max_depth = walker->max_depth;

  __shared__ myrandStateArr state;
  myrand_init(1337, gid, 0, &state);

  __shared__ Task tasks[BLOCK_SIZE];
  __shared__ int num_active_tasks;

  __shared__ int large_tasks[BLOCK_SIZE];
  __shared__ int block_task_count;

  __shared__ int walk_id;

  int num_walks = assign[bid].end - assign[bid].begin;

  int task_start = assign[bid].begin;

  vtx_t* local_start_points = start_points + task_start;
  vtx_t* local_result_pool = result_pool + task_start * max_depth;

  if (tid < num_walks) {
    local_result_pool[(u64)tid * max_depth] = local_start_points[tid];
    tasks[tid].init(graph, local_start_points, tid);
  } else {
    tasks[tid].length = -1;
  }

  if (tid == 0) {
    walk_id = min(BLOCK_SIZE, num_walks);
    num_active_tasks = num_walks;
    block_task_count = 0;
  }
  __syncthreads();
#ifdef MICRO_BENCH
  counter->sample_begin(bid);
#endif
  while (num_active_tasks > 0) {
    for (int local_task_id = wid; local_task_id < BLOCK_SIZE;
         local_task_id += WARP_PER_BLK) {
      Task& task = tasks[local_task_id];
      if (task.length >= 0) {
        if (task.degree <= THRESHOLD) {
#ifdef MICRO_BENCH
          u64 start = clock64();
#endif
#ifdef ONELOOP
          vtx_t selected = sampler_warp_oneloop<walker_t, myrandStateArr>(
              walker, &task, &state);
#else
          vtx_t selected =
              sampler_warp<walker_t, myrandStateArr>(walker, &task, &state);
#endif
#ifdef MICRO_BENCH
          counter->add_warp(bid, start, task.degree);
#endif
          if (lid == 0) {
            if (selected == -1) {
              task.length = max_depth;
            } else {
              selected = graph->adjncy[task.neighbor_offset + selected];
              local_result_pool[(u64)task.walker_id * max_depth + task.length] =
                  selected;
              task.update(graph, selected);
            }
          }
        } else {
          if (lid == 0) {
            int index = atomicAdd(&block_task_count, 1);
            large_tasks[index] = local_task_id;
          }
        }
      }
      __syncwarp(FULL_WARP_MASK);
    }
    __syncthreads();

    for (int i = 0; i < block_task_count; i++) {
      Task& task = tasks[large_tasks[i]];
#ifdef MICRO_BENCH
      u64 start = clock64();
#endif
#ifdef ONELOOP
      vtx_t selected = sampler_block_oneloop<walker_t, myrandStateArr>(
          walker, &task, &state);
#else
      vtx_t selected =
          sampler_block<walker_t, myrandStateArr>(walker, &task, &state);
#endif
#ifdef MICRO_BENCH
      counter->add_block(bid, start, task.degree);
#endif
      if (tid == 0) {
        if (selected == -1) {
          task.length = max_depth;
        } else {
          selected = graph->adjncy[task.neighbor_offset + selected];
          local_result_pool[(u64)task.walker_id * max_depth + task.length] =
              selected;
          task.update(graph, selected);
        }
      }
      __syncthreads();
    }

    // remove finished tasks and add fetch new tasks
    if (walker->is_stop(tasks[tid].length, &state)) {
      atomicAdd(&num_active_tasks, -1);
      int local_walk_id = atomicAdd(&walk_id, 1);
      if (local_walk_id < num_walks) {
        local_result_pool[(u64)local_walk_id * max_depth] =
            local_start_points[local_walk_id];
        tasks[tid].init(graph, local_start_points, local_walk_id);
      } else {
        tasks[tid].length = -1;
      }
    }
    __syncthreads();
    if (tid == 0) {
      block_task_count = 0;
    }
    __syncthreads();
  }
#ifdef MICRO_BENCH
  counter->block_end(bid);
#endif
}

#ifdef MICRO_BENCH
template <typename walker_t>
__global__ void walker_twb(walker_t* walker, vtx_t* start_points,
                           vtx_t* result_pool, TaskAssignments* assign,
                           metrics* counter)
#else
template <typename walker_t>
__global__ void walker_twb(walker_t* walker, vtx_t* start_points,
                           vtx_t* result_pool, TaskAssignments* assign)
#endif
{
#ifdef MICRO_BENCH
  counter->block_begin(blockIdx.x);
#endif
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int lid = threadIdx.x % WARP_SIZE;
  int gid = (blockDim.x * blockIdx.x) + tid;
  int wid = threadIdx.x / WARP_SIZE;

  gpu_graph* graph = walker->graph;
  int max_depth = walker->max_depth;

  __shared__ myrandStateArr state;
  myrand_init(1337, gid, 0, &state);

  __shared__ Task tasks[BLOCK_SIZE];
  __shared__ int num_active_tasks;

  __shared__ int large_tasks[BLOCK_SIZE];
  __shared__ int warp_task_count;
  __shared__ int block_task_count;

  __shared__ int walk_id;

  int num_walks = assign[bid].end - assign[bid].begin;

  int task_start = assign[bid].begin;

  vtx_t* local_start_points = start_points + task_start;
  vtx_t* local_result_pool = result_pool + task_start * max_depth;

  if (tid < num_walks) {
    local_result_pool[(u64)tid * max_depth] = local_start_points[tid];
    tasks[tid].init(graph, local_start_points, tid);
  } else {
    tasks[tid].length = -1;
  }

  if (tid == 0) {
    walk_id = min(BLOCK_SIZE, num_walks);
    num_active_tasks = num_walks;
    block_task_count = 0;
    warp_task_count = 0;
  }
  __syncthreads();
#ifdef MICRO_BENCH
  counter->sample_begin(bid);
#endif
  while (num_active_tasks > 0) {
    int local_task_id = tid;
    Task& task = tasks[local_task_id];
    if (task.length >= 0) {
      if (task.degree <= 16) {
#ifdef MICRO_BENCH
        u64 start = clock64();
#endif
        vtx_t selected =
            sampler_thread<walker_t, myrandStateArr>(walker, &task, &state);

#ifdef MICRO_BENCH
        counter->add_thread(bid, start, task.degree);
#endif
        if (selected == -1) {
          task.length = max_depth;
        } else {
          selected = graph->adjncy[task.neighbor_offset + selected];
          local_result_pool[(u64)task.walker_id * max_depth + task.length] =
              selected;
          task.update(graph, selected);
        }

        // task.print();
      } else if (task.degree <= THRESHOLD) {
        int index = atomicAdd(&warp_task_count, 1);
        large_tasks[index] = local_task_id;
      } else {
        int index = atomicAdd(&block_task_count, 1);
        large_tasks[BLOCK_SIZE - index - 1] = local_task_id;
      }
    }

    __syncthreads();
    // __syncwarp(FULL_WARP_MASK);

    for (int i = wid; i < warp_task_count; i += WARP_PER_BLK) {
      Task& h_task = tasks[large_tasks[i]];
#ifdef MICRO_BENCH
      u64 start = clock64();
#endif
#ifdef ONELOOP
      vtx_t selected = sampler_warp_oneloop<walker_t, myrandStateArr>(
          walker, &h_task, &state);
#else
      vtx_t selected =
          sampler_warp<walker_t, myrandStateArr>(walker, &h_task, &state);
#endif
#ifdef MICRO_BENCH
      counter->add_warp(bid, start, h_task.degree);
#endif
      if (lid == 0) {
        if (selected == -1) {
          h_task.length = max_depth;
        } else {
          selected = graph->adjncy[h_task.neighbor_offset + selected];
          local_result_pool[(u64)h_task.walker_id * max_depth + h_task.length] =
              selected;
          h_task.update(graph, selected);
        }
      }
      __syncwarp(FULL_WARP_MASK);
    }
    __syncthreads();
    for (int i = 0; i < block_task_count; i++) {
      Task& h_task = tasks[large_tasks[BLOCK_SIZE - i - 1]];
#ifdef MICRO_BENCH
      u64 start = clock64();
#endif
#ifdef ONELOOP
      vtx_t selected = sampler_block_oneloop<walker_t, myrandStateArr>(
          walker, &h_task, &state);
#else
      vtx_t selected =
          sampler_block<walker_t, myrandStateArr>(walker, &h_task, &state);
#endif
#ifdef MICRO_BENCH
      counter->add_block(bid, start, h_task.degree);
#endif
      if (tid == 0) {
        if (selected == -1) {
          h_task.length = max_depth;
        } else {
          selected = graph->adjncy[h_task.neighbor_offset + selected];
          local_result_pool[(u64)h_task.walker_id * max_depth + h_task.length] =
              selected;
          h_task.update(graph, selected);
        }
      }
      __syncthreads();
    }

    // remove finished tasks and add fetch new tasks
    if (walker->is_stop(tasks[tid].length, &state)) {
      atomicAdd(&num_active_tasks, -1);
      int local_walk_id = atomicAdd(&walk_id, 1);
      if (local_walk_id < num_walks) {
        local_result_pool[(u64)local_walk_id * max_depth] =
            local_start_points[local_walk_id];
        tasks[tid].init(graph, local_start_points, local_walk_id);
      } else {
        tasks[tid].length = -1;
      }
    }
    __syncthreads();
    if (tid == 0) {
      block_task_count = 0;
      warp_task_count = 0;
    }
    __syncthreads();
  }
#ifdef MICRO_BENCH
  counter->block_end(bid);
#endif
}

#ifdef MICRO_BENCH
template <typename walker_t>
__global__ void walker_queue(walker_t* walker, queue_gpu<Task>* walker_queue,
                             queue_gpu<Task>* high_queue, vtx_t* start_points,
                             vtx_t* result_pool, int walker_num,
                             metrics* counter)
#else
template <typename walker_t>
__global__ void walker_queue(walker_t* walker, queue_gpu<Task>* walker_queue,
                             queue_gpu<Task>* high_queue, vtx_t* start_points,
                             vtx_t* result_pool, int walker_num)
#endif
{
#ifdef MICRO_BENCH
  int bid = blockIdx.x;
  counter->block_begin(bid);
#endif
  int tid = threadIdx.x;

  int lid = threadIdx.x % WARP_SIZE;
  int gid = (blockDim.x * blockIdx.x) + tid;
  int wid = threadIdx.x / WARP_SIZE;

  gpu_graph* graph = walker->graph;
  int max_depth = walker->max_depth;

  __shared__ myrandStateArr state;
  myrand_init(1337, gid, 0, &state);

  for (uint i = gid; i < walker_num; i += gridDim.x * blockDim.x) {
    walker_queue[0].push(Task(graph, start_points, i));
    result_pool[(u64)i * max_depth] = start_points[i];
  }
  __threadfence();
#ifdef MICRO_BENCH
  counter->sample_begin(bid);
#endif
  for (int i = 0; i < max_depth; i++) {
    // __threadfence();
    __threadfence_block();
    __shared__ Task task[WARP_PER_BLK];
    __shared__ bool not_empty[WARP_PER_BLK];

    not_empty[wid] = false;

    if (lid == 0) {
      not_empty[wid] = walker_queue[i].pop(&task[wid]);
    }
    __syncwarp(FULL_WARP_MASK);

    while (not_empty[wid]) {
      if (task[wid].degree <= THRESHOLD) {
#ifdef MICRO_BENCH
        u64 start = clock64();
#endif
#ifdef ONELOOP
        vtx_t selected = sampler_warp_oneloop<walker_t, myrandStateArr>(
            walker, &task[wid], &state);
#else
        vtx_t selected =
            sampler_warp<walker_t, myrandStateArr>(walker, &task[wid], &state);
#endif
#ifdef MICRO_BENCH
        counter->add_warp(bid, start, task[wid].degree);
#endif
        if (lid == 0) {
          if (selected == -1) {
            task[wid].length = max_depth;
          } else {
            selected = graph->adjncy[task[wid].neighbor_offset + selected];
            result_pool[(u64)task[wid].walker_id * max_depth +
                        task[wid].length] = selected;
            task[wid].update(graph, selected);
          }
          if (!walker->is_stop(task[wid].length, &state)) {
            walker_queue[task[wid].length].push(task[wid]);
          }
        }
      } else {
        if (lid == 0) high_queue[i].push(task[wid]);
      }
      __syncwarp(FULL_WARP_MASK);
      if (lid == 0) {
        not_empty[wid] = walker_queue[i].pop(&task[wid]);
      }
      __syncwarp(FULL_WARP_MASK);
    }

    __syncthreads();

    __shared__ Task h_task;
    __shared__ bool h_not_empty;
    if (tid == 0) {
      h_not_empty = high_queue[i].pop(&h_task);
    }
    __syncthreads();

    while (h_not_empty) {
#ifdef MICRO_BENCH
      u64 start = clock64();
#endif
#ifdef ONELOOP
      vtx_t selected = sampler_block_oneloop<walker_t, myrandStateArr>(
          walker, &h_task, &state);
#else
      vtx_t selected =
          sampler_block<walker_t, myrandStateArr>(walker, &h_task, &state);
#endif
#ifdef MICRO_BENCH
      counter->add_block(bid, start, h_task.degree);
#endif
      __syncthreads();
      if (tid == 0) {
        if (selected == -1) {
          h_task.length = max_depth;
        } else {
          selected = graph->adjncy[h_task.neighbor_offset + selected];
          result_pool[(u64)h_task.walker_id * max_depth + h_task.length] =
              selected;
          h_task.update(graph, selected);
        }
        if (!walker->is_stop(h_task.length, &state)) {
          walker_queue[h_task.length].push(h_task);
        }
      }
      __syncthreads();

      if (tid == 0) {
        h_not_empty = high_queue[i].pop(&h_task);
      }
      __syncthreads();
    }
    __syncthreads();
  }
#ifdef MICRO_BENCH
  counter->block_end(bid);
#endif
}

#ifdef MICRO_BENCH
template <typename walker_t>
__global__ void walker_tw(walker_t* walker, vtx_t* start_points,
                          vtx_t* result_pool, TaskAssignments* assign,
                          metrics* counter)
#else
template <typename walker_t>
__global__ void walker_tw(walker_t* walker, vtx_t* start_points,
                          vtx_t* result_pool, TaskAssignments* assign)
#endif
{
#ifdef MICRO_BENCH
  counter->block_begin(blockIdx.x);
#endif
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int lid = threadIdx.x % WARP_SIZE;
  int gid = (blockDim.x * blockIdx.x) + tid;
  int wid = threadIdx.x / WARP_SIZE;

  gpu_graph* graph = walker->graph;
  int max_depth = walker->max_depth;

  __shared__ myrandStateArr state;
  myrand_init(1337, gid, 0, &state);

  __shared__ Task tasks[BLOCK_SIZE];
  __shared__ int num_active_tasks;

  __shared__ int large_tasks[BLOCK_SIZE];
  __shared__ int warp_task_count;

  __shared__ int walk_id;

  int num_walks = assign[bid].end - assign[bid].begin;

  int task_start = assign[bid].begin;

  vtx_t* local_start_points = start_points + task_start;
  vtx_t* local_result_pool = result_pool + task_start * max_depth;

  if (tid < num_walks) {
    local_result_pool[(u64)tid * max_depth] = local_start_points[tid];
    tasks[tid].init(graph, local_start_points, tid);
  } else {
    tasks[tid].length = -1;
  }

  if (tid == 0) {
    walk_id = min(BLOCK_SIZE, num_walks);
    num_active_tasks = num_walks;
    warp_task_count = 0;
  }
  __syncthreads();
#ifdef MICRO_BENCH
  counter->sample_begin(bid);
#endif
  while (num_active_tasks > 0) {
    int local_task_id = tid;
    Task& task = tasks[local_task_id];
    if (task.length >= 0) {
#ifdef MICRO_BENCH
      u64 start = clock64();
#endif
      if (task.degree <= 16) {
        vtx_t selected =
            sampler_thread<walker_t, myrandStateArr>(walker, &task, &state);
#ifdef MICRO_BENCH
        counter->add_thread(bid, start, task.degree);
#endif

        if (selected == -1) {
          task.length = max_depth;
        } else {
          selected = graph->adjncy[task.neighbor_offset + selected];
          local_result_pool[(u64)task.walker_id * max_depth + task.length] =
              selected;
          task.update(graph, selected);
        }
      } else {
        int index = atomicAdd(&warp_task_count, 1);
        large_tasks[index] = local_task_id;
      }
    }

    __syncthreads();

    for (int i = wid; i < warp_task_count; i += WARP_PER_BLK) {
      Task& h_task = tasks[large_tasks[i]];
#ifdef MICRO_BENCH
      u64 start = clock64();
#endif
#ifdef ONELOOP
      vtx_t selected = sampler_warp_oneloop<walker_t, myrandStateArr>(
          walker, &h_task, &state);
#else
      vtx_t selected =
          sampler_warp<walker_t, myrandStateArr>(walker, &h_task, &state);
#endif
#ifdef MICRO_BENCH
      counter->add_warp(bid, start, h_task.degree);
#endif
      if (lid == 0) {
        if (selected == -1) {
          h_task.length = max_depth;
        } else {
          selected = graph->adjncy[h_task.neighbor_offset + selected];
          local_result_pool[(u64)h_task.walker_id * max_depth + h_task.length] =
              selected;
          h_task.update(graph, selected);
        }
      }
      __syncwarp(FULL_WARP_MASK);
    }
    __syncthreads();

    // remove finished tasks and add fetch new tasks
    if (walker->is_stop(tasks[tid].length, &state)) {
      atomicAdd(&num_active_tasks, -1);
      int local_walk_id = atomicAdd(&walk_id, 1);
      if (local_walk_id < num_walks) {
        local_result_pool[(u64)local_walk_id * max_depth] =
            local_start_points[local_walk_id];
        tasks[tid].init(graph, local_start_points, local_walk_id);
      } else {
        tasks[tid].length = -1;
      }
    }
    __syncthreads();
    if (tid == 0) {
      warp_task_count = 0;
    }
    __syncthreads();
  }
#ifdef MICRO_BENCH
  counter->block_end(bid);
#endif
}

#ifdef MICRO_BENCH
template <typename walker_t>
__global__ void walker_twb_dynamic(walker_t* walker, vtx_t* start_points,
                                   int* start_pointer, vtx_t* result_pool,
                                   int walker_num, metrics* counter)
#else
template <typename walker_t>
__global__ void walker_twb_dynamic(walker_t* walker, vtx_t* start_points,
                                   int* start_pointer, vtx_t* result_pool,
                                   int walker_num)
#endif
{
#ifdef MICRO_BENCH
  int bid = blockIdx.x;
  counter->block_begin(bid);
#endif
  int tid = threadIdx.x;
  int lid = threadIdx.x % WARP_SIZE;
  int gid = (blockDim.x * blockIdx.x) + tid;
  int wid = threadIdx.x / WARP_SIZE;

  gpu_graph* graph = walker->graph;
  int max_depth = walker->max_depth;

  __shared__ myrandStateArr state;
  myrand_init(1337, gid, 0, &state);

  __shared__ Task tasks[TASK_NUM];
  __shared__ int num_active_tasks;

  __shared__ int large_tasks[TASK_NUM];
  __shared__ int warp_task_count;
  __shared__ int block_task_count;

  if (tid == 0) {
    num_active_tasks = 0;
    warp_task_count = 0;
    block_task_count = 0;
  }
  __syncthreads();

  if (tid < TASK_NUM && tid < walker_num) {
    int old_size = atomicAdd(start_pointer, 1);
    if (old_size < walker_num) {
      vtx_t sp = start_points[old_size];
      result_pool[(u64)old_size * max_depth] = sp;
      tasks[tid].init(graph, sp, old_size);
      atomicAdd(&num_active_tasks, 1);
    } else {
      tasks[tid].length = -1;
    }
  } else if (tid < TASK_NUM && tid >= walker_num) {
    tasks[tid].length = -1;
  }

  __syncthreads();
#ifdef MICRO_BENCH
  counter->sample_begin(bid);
#endif
  while (num_active_tasks > 0) {
    if (tid < TASK_NUM) {
      int local_task_id = tid;
      Task& task = tasks[local_task_id];
      if (task.length >= 0) {
        if (task.degree <= 16) {
#ifdef MICRO_BENCH
          u64 start = clock64();
#endif
          vtx_t selected =
              sampler_thread<walker_t, myrandStateArr>(walker, &task, &state);

#ifdef MICRO_BENCH
          counter->add_thread(bid, start, task.degree);
#endif
          if (selected == -1) {
            task.length = max_depth;
          } else {
            selected = graph->adjncy[task.neighbor_offset + selected];
            result_pool[(u64)task.walker_id * max_depth + task.length] =
                selected;
            task.update(graph, selected);
          }

          // task.print();
        } else if (task.degree <= THRESHOLD) {
          int index = atomicAdd(&warp_task_count, 1);
          large_tasks[index] = local_task_id;
        } else {
          int index = atomicAdd(&block_task_count, 1);
          large_tasks[TASK_NUM - index - 1] = local_task_id;
        }
      }
    }

    __syncthreads();

    for (int i = wid; i < warp_task_count; i += WARP_PER_BLK) {
      Task& h_task = tasks[large_tasks[i]];
#ifdef MICRO_BENCH
      u64 start = clock64();
#endif
#ifdef ONELOOP
      vtx_t selected = sampler_warp_oneloop<walker_t, myrandStateArr>(
          walker, &h_task, &state);
#else
      vtx_t selected =
          sampler_warp<walker_t, myrandStateArr>(walker, &h_task, &state);
#endif
#ifdef MICRO_BENCH
      counter->add_warp(bid, start, h_task.degree);
#endif
      if (lid == 0) {
        if (selected == -1) {
          h_task.length = max_depth;
        } else {
          selected = graph->adjncy[h_task.neighbor_offset + selected];
          result_pool[(u64)h_task.walker_id * max_depth + h_task.length] =
              selected;
          h_task.update(graph, selected);
        }
      }
      __syncwarp(FULL_WARP_MASK);
    }
    __syncthreads();
    for (int i = 0; i < block_task_count; i++) {
      Task& h_task = tasks[large_tasks[TASK_NUM - i - 1]];
#ifdef MICRO_BENCH
      u64 start = clock64();
#endif
#ifdef ONELOOP
      vtx_t selected = sampler_block_oneloop<walker_t, myrandStateArr>(
          walker, &h_task, &state);
#else
      vtx_t selected =
          sampler_block<walker_t, myrandStateArr>(walker, &h_task, &state);
#endif
#ifdef MICRO_BENCH
      counter->add_block(bid, start, h_task.degree);
#endif
      if (tid == 0) {
        if (selected == -1) {
          h_task.length = max_depth;
        } else {
          selected = graph->adjncy[h_task.neighbor_offset + selected];
          result_pool[(u64)h_task.walker_id * max_depth + h_task.length] =
              selected;
          h_task.update(graph, selected);
        }
      }
      __syncthreads();
    }

    // remove finished tasks and add fetch new tasks
    if (tid < TASK_NUM && walker->is_stop(tasks[tid].length, &state)) {
      int old_size = atomicAdd(start_pointer, 1);
      if (old_size < walker_num) {
        vtx_t sp = start_points[old_size];
        result_pool[(u64)old_size * max_depth] = sp;
        tasks[tid].init(graph, sp, old_size);
      } else {
        atomicAdd(&num_active_tasks, -1);
        tasks[tid].length = -1;
      }
    }
    __syncthreads();
    if (tid == 0) {
      block_task_count = 0;
      warp_task_count = 0;
    }
    __syncthreads();
  }
#ifdef MICRO_BENCH
  counter->block_end(bid);
#endif
}

#ifdef MICRO_BENCH
template <typename walker_t>
__global__ void walker_wb_dynamic(walker_t* walker, vtx_t* start_points,
                                  int* start_pointer, vtx_t* result_pool,
                                  int walker_num, metrics* counter)
#else
template <typename walker_t>
__global__ void walker_wb_dynamic(walker_t* walker, vtx_t* start_points,
                                  int* start_pointer, vtx_t* result_pool,
                                  int walker_num)
#endif
{
#ifdef MICRO_BENCH
  int bid = blockIdx.x;
  counter->block_begin(bid);
#endif
  int tid = threadIdx.x;
  int lid = threadIdx.x % WARP_SIZE;
  int gid = (blockDim.x * blockIdx.x) + tid;
  int wid = threadIdx.x / WARP_SIZE;

  gpu_graph* graph = walker->graph;
  int max_depth = walker->max_depth;

  __shared__ myrandStateArr state;
  myrand_init(1337, gid, 0, &state);

  __shared__ Task tasks[TASK_NUM];
  __shared__ int num_active_tasks;

  __shared__ int large_tasks[TASK_NUM];
  __shared__ int block_task_count;

  if (tid == 0) {
    num_active_tasks = 0;
    block_task_count = 0;
  }
  __syncthreads();

  if (tid < TASK_NUM && tid < walker_num) {
    int old_size = atomicAdd(start_pointer, 1);
    if (old_size < walker_num) {
      vtx_t sp = start_points[old_size];
      result_pool[(u64)old_size * max_depth] = sp;
      tasks[tid].init(graph, sp, old_size);
      atomicAdd(&num_active_tasks, 1);
    } else {
      tasks[tid].length = -1;
    }
  } else if (tid < TASK_NUM && tid >= walker_num) {
    tasks[tid].length = -1;
  }

  __syncthreads();

#ifdef MICRO_BENCH
  counter->sample_begin(bid);
#endif
  while (num_active_tasks > 0) {
    for (int local_task_id = wid; local_task_id < TASK_NUM;
         local_task_id += WARP_PER_BLK) {
      Task& task = tasks[local_task_id];
      if (task.length >= 0) {
        // if (task.degree <= 32768)
        if (task.degree <= THRESHOLD) {
#ifdef MICRO_BENCH
          u64 start = clock64();
#endif
#ifdef ONELOOP
          vtx_t selected = sampler_warp_oneloop<walker_t, myrandStateArr>(
              walker, &task, &state);
#else
          vtx_t selected =
              sampler_warp<walker_t, myrandStateArr>(walker, &task, &state);
#endif
#ifdef MICRO_BENCH
          counter->add_warp(bid, start, task.degree);
#endif
          if (lid == 0) {
            if (selected == -1) {
              task.length = max_depth;
            } else {
              selected = graph->adjncy[task.neighbor_offset + selected];
              result_pool[(u64)task.walker_id * max_depth + task.length] =
                  selected;
              task.update(graph, selected);
            }
          }
        } else {
          if (lid == 0) {
            int index = atomicAdd(&block_task_count, 1);
            large_tasks[index] = local_task_id;
          }
        }
      }
      __syncwarp(FULL_WARP_MASK);
    }
    __syncthreads();
    // __syncwarp(FULL_WARP_MASK);

    for (int i = 0; i < block_task_count; i++) {
      Task& h_task = tasks[large_tasks[i]];
#ifdef MICRO_BENCH
      u64 start = clock64();
#endif
#ifdef ONELOOP
      vtx_t selected = sampler_block_oneloop<walker_t, myrandStateArr>(
          walker, &h_task, &state);
#else
      vtx_t selected =
          sampler_block<walker_t, myrandStateArr>(walker, &h_task, &state);
#endif
#ifdef MICRO_BENCH
      counter->add_block(bid, start, h_task.degree);
#endif
      if (tid == 0) {
        if (selected == -1) {
          h_task.length = max_depth;
        } else {
          selected = graph->adjncy[h_task.neighbor_offset + selected];
          result_pool[(u64)h_task.walker_id * max_depth + h_task.length] =
              selected;
          h_task.update(graph, selected);
        }
      }
      __syncthreads();
    }

    // remove finished tasks and add fetch new tasks
    if (tid < TASK_NUM && walker->is_stop(tasks[tid].length, &state)) {
      int old_size = atomicAdd(start_pointer, 1);
      if (old_size < walker_num) {
        vtx_t sp = start_points[old_size];
        result_pool[(u64)old_size * max_depth] = sp;
        tasks[tid].init(graph, sp, old_size);
      } else {
        atomicAdd(&num_active_tasks, -1);
        tasks[tid].length = -1;
      }
    }
    __syncthreads();
    if (tid == 0) {
      block_task_count = 0;
      // warp_task_count = 0;
    }
    __syncthreads();
  }
#ifdef MICRO_BENCH
  counter->block_end(bid);
#endif
}

#ifdef MICRO_BENCH
template <typename walker_t>
__global__ void walker_w_dynamic(walker_t* walker, vtx_t* start_points,
                                 int* start_pointer, vtx_t* result_pool,
                                 int walker_num, metrics* counter)
#else
template <typename walker_t>
__global__ void walker_w_dynamic(walker_t* walker, vtx_t* start_points,
                                 int* start_pointer, vtx_t* result_pool,
                                 int walker_num)
#endif
{
#ifdef MICRO_BENCH
  int bid = blockIdx.x;
  counter->block_begin(bid);
#endif
  int tid = threadIdx.x;
  int lid = threadIdx.x % WARP_SIZE;
  int gid = (blockDim.x * blockIdx.x) + tid;
  int wid = threadIdx.x / WARP_SIZE;

  gpu_graph* graph = walker->graph;
  int max_depth = walker->max_depth;

  __shared__ myrandStateArr state;
  myrand_init(1337, gid, 0, &state);

  __shared__ Task tasks[TASK_NUM];
  __shared__ int num_active_tasks;

  if (tid == 0) {
    num_active_tasks = 0;
  }
  __syncthreads();

  if (tid < TASK_NUM && tid < walker_num) {
    int old_size = atomicAdd(start_pointer, 1);
    if (old_size < walker_num) {
      vtx_t sp = start_points[old_size];
      result_pool[(u64)old_size * max_depth] = sp;
      tasks[tid].init(graph, sp, old_size);
      atomicAdd(&num_active_tasks, 1);
    } else {
      tasks[tid].length = -1;
    }
  } else if (tid < TASK_NUM && tid >= walker_num) {
    tasks[tid].length = -1;
  }

  __syncthreads();

#ifdef MICRO_BENCH
  counter->sample_begin(bid);
#endif
  while (num_active_tasks > 0) {
    for (int local_task_id = wid; local_task_id < TASK_NUM;
         local_task_id += WARP_PER_BLK) {
      Task& task = tasks[local_task_id];
      if (task.length >= 0) {
#ifdef MICRO_BENCH
        u64 start = clock64();
#endif
#ifdef ONELOOP
        vtx_t selected = sampler_warp_oneloop<walker_t, myrandStateArr>(
            walker, &task, &state);
#else
        vtx_t selected =
            sampler_warp<walker_t, myrandStateArr>(walker, &task, &state);
#endif
#ifdef MICRO_BENCH
        counter->add_warp(bid, start, task.degree);
#endif
        if (lid == 0) {
          if (selected == -1) {
            task.length = max_depth;
          } else {
            selected = graph->adjncy[task.neighbor_offset + selected];
            result_pool[(u64)task.walker_id * max_depth + task.length] =
                selected;
            task.update(graph, selected);
          }
        }
      }
      __syncwarp(FULL_WARP_MASK);
    }
    __syncthreads();

    // remove finished tasks and add fetch new tasks
    if (tid < TASK_NUM && walker->is_stop(tasks[tid].length, &state)) {
      int old_size = atomicAdd(start_pointer, 1);
      if (old_size < walker_num) {
        vtx_t sp = start_points[old_size];
        result_pool[(u64)old_size * max_depth] = sp;
        tasks[tid].init(graph, sp, old_size);
      } else {
        atomicAdd(&num_active_tasks, -1);
        tasks[tid].length = -1;
      }
    }
    __syncthreads();
  }
#ifdef MICRO_BENCH
  counter->block_end(bid);
#endif
}

#ifdef MICRO_BENCH
template <typename walker_t>
__global__ void walker_b_dynamic(walker_t* walker, vtx_t* start_points,
                                 int* start_pointer, vtx_t* result_pool,
                                 int walker_num, metrics* counter)
#else
template <typename walker_t>
__global__ void walker_b_dynamic(walker_t* walker, vtx_t* start_points,
                                 int* start_pointer, vtx_t* result_pool,
                                 int walker_num)
#endif
{
#ifdef MICRO_BENCH
  int bid = blockIdx.x;
  counter->block_begin(bid);
#endif
  int tid = threadIdx.x;
  int gid = (blockDim.x * blockIdx.x) + tid;

  gpu_graph* graph = walker->graph;
  int max_depth = walker->max_depth;

  __shared__ myrandStateArr state;
  myrand_init(1337, gid, 0, &state);

  __shared__ Task tasks[TASK_NUM];
  __shared__ int num_active_tasks;

  if (tid == 0) {
    num_active_tasks = 0;
  }
  __syncthreads();

  if (tid < TASK_NUM && tid < walker_num) {
    int old_size = atomicAdd(start_pointer, 1);
    if (old_size < walker_num) {
      vtx_t sp = start_points[old_size];
      result_pool[(u64)old_size * max_depth] = sp;
      tasks[tid].init(graph, sp, old_size);
      atomicAdd(&num_active_tasks, 1);
    } else {
      tasks[tid].length = -1;
    }
  } else if (tid < TASK_NUM && tid >= walker_num) {
    tasks[tid].length = -1;
  }

  __syncthreads();
#ifdef MICRO_BENCH
  counter->sample_begin(bid);
#endif
  while (num_active_tasks > 0) {
    for (int local_task_id = 0; local_task_id < TASK_NUM; local_task_id++) {
      Task& task = tasks[local_task_id];
      if (task.length >= 0) {
#ifdef MICRO_BENCH
        u64 start = clock64();
#endif
#ifdef ONELOOP
        vtx_t selected = sampler_block_oneloop<walker_t, myrandStateArr>(
            walker, &task, &state);
#else
        vtx_t selected =
            sampler_block<walker_t, myrandStateArr>(walker, &task, &state);
#endif
#ifdef MICRO_BENCH
        counter->add_block(bid, start, task.degree);
#endif
        if (tid == 0) {
          if (selected == -1) {
            task.length = max_depth;
          } else {
            selected = graph->adjncy[task.neighbor_offset + selected];
            result_pool[(u64)task.walker_id * max_depth + task.length] =
                selected;
            task.update(graph, selected);
          }
        }
      }
      __syncwarp(FULL_WARP_MASK);
    }
    __syncthreads();

    // remove finished tasks and add fetch new tasks
    if (tid < TASK_NUM && walker->is_stop(tasks[tid].length, &state)) {
      int old_size = atomicAdd(start_pointer, 1);
      if (old_size < walker_num) {
        vtx_t sp = start_points[old_size];
        result_pool[(u64)old_size * max_depth] = sp;
        tasks[tid].init(graph, sp, old_size);
      } else {
        atomicAdd(&num_active_tasks, -1);
        tasks[tid].length = -1;
      }
    }
    __syncthreads();
  }
#ifdef MICRO_BENCH
  counter->block_end(bid);
#endif
}

template <typename walker_t>
__global__ void walker_wb_dynamic_zprs(walker_t* walker, vtx_t* start_points,
                                       int* start_pointer, vtx_t* result_pool,
                                       int walker_num) {
#ifdef MICRO_BENCH
  int bid = blockIdx.x;
  counter->block_begin(bid);
#endif
  int tid = threadIdx.x;
  int lid = threadIdx.x % WARP_SIZE;
  int gid = (blockDim.x * blockIdx.x) + tid;
  int wid = threadIdx.x / WARP_SIZE;

  gpu_graph* graph = walker->graph;
  int max_depth = walker->max_depth;

  __shared__ myrandStateArr state;
  myrand_init(1337, gid, 0, &state);

  __shared__ Task tasks[TASK_NUM];
  __shared__ int num_active_tasks;

  __shared__ int large_tasks[TASK_NUM];
  __shared__ int block_task_count;

  if (tid == 0) {
    num_active_tasks = 0;
    block_task_count = 0;
  }
  __syncthreads();

  if (tid < TASK_NUM && tid < walker_num) {
    int old_size = atomicAdd(start_pointer, 1);
    if (old_size < walker_num) {
      vtx_t sp = start_points[old_size];
      result_pool[(u64)old_size * max_depth] = sp;
      tasks[tid].init(graph, sp, old_size);
      atomicAdd(&num_active_tasks, 1);
    } else {
      tasks[tid].length = -1;
    }
  } else if (tid < TASK_NUM && tid >= walker_num) {
    tasks[tid].length = -1;
  }

  __syncthreads();

#ifdef MICRO_BENCH
  counter->sample_begin(bid);
#endif
  while (num_active_tasks > 0) {
    for (int local_task_id = wid; local_task_id < TASK_NUM;
         local_task_id += WARP_PER_BLK) {
      Task& task = tasks[local_task_id];
      if (task.length >= 0) {
        if (task.degree <= THRESHOLD) {
#ifdef MICRO_BENCH
          u64 start = clock64();
#endif

          vtx_t selected =
              sampler_warp<walker_t, myrandStateArr>(walker, &task, &state);
#ifdef MICRO_BENCH
          counter->add_warp(bid, start, task.degree);
#endif
          if (lid == 0) {
            if (selected == -1) {
              task.length = max_depth;
            } else {
              // printf("walker:%d,len=%d,label=%d\n", task.walker_id,
              // task.length, graph->edge_label[task.neighbor_offset +
              // selected]);
              selected = graph->adjncy[task.neighbor_offset + selected];
              result_pool[(u64)task.walker_id * max_depth + task.length] =
                  selected;

              task.update(graph, selected);
            }
          }
        } else {
          if (lid == 0) {
            int index = atomicAdd(&block_task_count, 1);
            large_tasks[index] = local_task_id;
          }
        }
      }
      __syncwarp(FULL_WARP_MASK);
    }
    __syncthreads();

    for (int i = 0; i < block_task_count; i++) {
      Task& h_task = tasks[large_tasks[i]];
#ifdef MICRO_BENCH
      u64 start = clock64();
#endif
      vtx_t selected =
          sampler_block<walker_t, myrandStateArr>(walker, &h_task, &state);

#ifdef MICRO_BENCH
      counter->add_block(bid, start, h_task.degree);
#endif
      if (tid == 0) {
        if (selected == -1) {
          h_task.length = max_depth;
        } else {
          selected = graph->adjncy[h_task.neighbor_offset + selected];
          result_pool[(u64)h_task.walker_id * max_depth + h_task.length] =
              selected;
          h_task.update(graph, selected);
        }
      }
      __syncthreads();
    }

    // remove finished tasks and add fetch new tasks
    if (tid < TASK_NUM && walker->is_stop(tasks[tid].length, &state)) {
      int old_size = atomicAdd(start_pointer, 1);
      if (old_size < walker_num) {
        vtx_t sp = start_points[old_size];
        result_pool[(u64)old_size * max_depth] = sp;
        tasks[tid].init(graph, sp, old_size);
      } else {
        atomicAdd(&num_active_tasks, -1);
        tasks[tid].length = -1;
      }
    }
    __syncthreads();
    if (tid == 0) {
      block_task_count = 0;
    }
    __syncthreads();
  }
#ifdef MICRO_BENCH
  counter->block_end(bid);
#endif
}

template <typename walker_t>
__global__ void walker_wb_dynamic_dprs(walker_t* walker, vtx_t* start_points,
                                       int* start_pointer, vtx_t* result_pool,
                                       int walker_num) {
#ifdef MICRO_BENCH
  int bid = blockIdx.x;
  counter->block_begin(bid);
#endif
  int tid = threadIdx.x;
  int lid = threadIdx.x % WARP_SIZE;
  int gid = (blockDim.x * blockIdx.x) + tid;
  int wid = threadIdx.x / WARP_SIZE;

  gpu_graph* graph = walker->graph;
  int max_depth = walker->max_depth;

  __shared__ myrandStateArr state;
  myrand_init(1337, gid, 0, &state);

  __shared__ Task tasks[TASK_NUM];
  __shared__ int num_active_tasks;

  __shared__ int large_tasks[TASK_NUM];
  __shared__ int block_task_count;

  if (tid == 0) {
    num_active_tasks = 0;
    block_task_count = 0;
  }
  __syncthreads();

  if (tid < TASK_NUM && tid < walker_num) {
    int old_size = atomicAdd(start_pointer, 1);
    if (old_size < walker_num) {
      vtx_t sp = start_points[old_size];
      result_pool[(u64)old_size * max_depth] = sp;
      tasks[tid].init(graph, sp, old_size);
      atomicAdd(&num_active_tasks, 1);
    } else {
      tasks[tid].length = -1;
    }
  } else if (tid < TASK_NUM && tid >= walker_num) {
    tasks[tid].length = -1;
  }

  __syncthreads();
  // if (tid == 0)
  //     printf("walker num=%d,bid=%d,active tasks:%d\n", walker_num, bid,
  //     num_active_tasks);
#ifdef MICRO_BENCH
  counter->sample_begin(bid);
#endif
  while (num_active_tasks > 0) {
    for (int local_task_id = wid; local_task_id < TASK_NUM;
         local_task_id += WARP_PER_BLK) {
      Task& task = tasks[local_task_id];
      if (task.length >= 0) {
        if (task.degree <= THRESHOLD) {
#ifdef MICRO_BENCH
          u64 start = clock64();
#endif

          vtx_t selected = sampler_warp_oneloop<walker_t, myrandStateArr>(
              walker, &task, &state);
#ifdef MICRO_BENCH
          counter->add_warp(bid, start, task.degree);
#endif
          if (lid == 0) {
            // printf("selected:%d,len:%d\n", selected, task.length);

            if (selected == -1) {
              task.length = max_depth;
            } else {
              selected = graph->adjncy[task.neighbor_offset + selected];
              result_pool[(u64)task.walker_id * max_depth + task.length] =
                  selected;
              task.update(graph, selected);
            }
          }
        } else {
          if (lid == 0) {
            int index = atomicAdd(&block_task_count, 1);
            large_tasks[index] = local_task_id;
          }
        }
      }
      __syncwarp(FULL_WARP_MASK);
    }
    __syncthreads();

    for (int i = 0; i < block_task_count; i++) {
      Task& h_task = tasks[large_tasks[i]];
#ifdef MICRO_BENCH
      u64 start = clock64();
#endif

      vtx_t selected = sampler_block_oneloop<walker_t, myrandStateArr>(
          walker, &h_task, &state);

#ifdef MICRO_BENCH
      counter->add_block(bid, start, h_task.degree);
#endif
      if (tid == 0) {
        if (selected == -1) {
          h_task.length = max_depth;
        } else {
          selected = graph->adjncy[h_task.neighbor_offset + selected];
          result_pool[(u64)h_task.walker_id * max_depth + h_task.length] =
              selected;
          h_task.update(graph, selected);
        }
      }
      __syncthreads();
    }

    // remove finished tasks and add fetch new tasks
    if (tid < TASK_NUM && walker->is_stop(tasks[tid].length, &state)) {
      int old_size = atomicAdd(start_pointer, 1);
      if (old_size < walker_num) {
        vtx_t sp = start_points[old_size];
        result_pool[(u64)old_size * max_depth] = sp;
        tasks[tid].init(graph, sp, old_size);
      } else {
        atomicAdd(&num_active_tasks, -1);
        tasks[tid].length = -1;
      }
    }
    __syncthreads();
    if (tid == 0) {
      block_task_count = 0;
    }
    __syncthreads();
  }
#ifdef MICRO_BENCH
  counter->block_end(bid);
#endif
}

// plain
template <typename walker_t>
__global__ void walker_wb_dprs_curand(walker_t* walker, vtx_t* start_points,
                                      vtx_t* result_pool,
                                      TaskAssignments* assign,
                                      curandState* state) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int lid = threadIdx.x % WARP_SIZE;
  int gid = (blockDim.x * blockIdx.x) + tid;
  int wid = threadIdx.x / WARP_SIZE;

  gpu_graph* graph = walker->graph;
  int max_depth = walker->max_depth;

  curand_init(1337, gid, 0, state + gid);

  __shared__ Task tasks[BLOCK_SIZE];
  __shared__ int num_active_tasks;

  __shared__ int large_tasks[BLOCK_SIZE];
  __shared__ int block_task_count;

  __shared__ int walk_id;

  int num_walks = assign[bid].end - assign[bid].begin;

  int task_start = assign[bid].begin;

  vtx_t* local_start_points = start_points + task_start;
  vtx_t* local_result_pool = result_pool + task_start * max_depth;

  if (tid < num_walks) {
    local_result_pool[(u64)tid * max_depth] = local_start_points[tid];
    tasks[tid].init(graph, local_start_points, tid);
  } else {
    tasks[tid].length = -1;
  }

  if (tid == 0) {
    walk_id = min(BLOCK_SIZE, num_walks);
    num_active_tasks = num_walks;
    block_task_count = 0;
  }
  __syncthreads();

  while (num_active_tasks > 0) {
    for (int local_task_id = wid; local_task_id < BLOCK_SIZE;
         local_task_id += WARP_PER_BLK) {
      Task& task = tasks[local_task_id];
      if (task.length >= 0) {
        if (task.degree <= THRESHOLD) {
          vtx_t selected = sampler_warp_oneloop<walker_t, curandState>(
              walker, &task, state + gid);

          if (lid == 0) {
            if (selected == -1) {
              task.length = max_depth;
            } else {
              selected = graph->adjncy[task.neighbor_offset + selected];
              local_result_pool[(u64)task.walker_id * max_depth + task.length] =
                  selected;
              task.update(graph, selected);
            }
          }
        } else {
          if (lid == 0) {
            int index = atomicAdd(&block_task_count, 1);
            large_tasks[index] = local_task_id;
          }
        }
      }
      __syncwarp(FULL_WARP_MASK);
    }
    __syncthreads();

    for (int i = 0; i < block_task_count; i++) {
      Task& task = tasks[large_tasks[i]];

      vtx_t selected = sampler_block_oneloop<walker_t, curandState>(
          walker, &task, state + gid);

      if (tid == 0) {
        if (selected == -1) {
          task.length = max_depth;
        } else {
          selected = graph->adjncy[task.neighbor_offset + selected];
          local_result_pool[(u64)task.walker_id * max_depth + task.length] =
              selected;
          task.update(graph, selected);
        }
      }
      __syncthreads();
    }

    // remove finished tasks and add fetch new tasks
    if (walker->is_stop(tasks[tid].length, state + gid)) {
      atomicAdd(&num_active_tasks, -1);
      int local_walk_id = atomicAdd(&walk_id, 1);
      if (local_walk_id < num_walks) {
        local_result_pool[(u64)local_walk_id * max_depth] =
            local_start_points[local_walk_id];
        tasks[tid].init(graph, local_start_points, local_walk_id);
      } else {
        tasks[tid].length = -1;
      }
    }
    __syncthreads();
    if (tid == 0) {
      block_task_count = 0;
    }
    __syncthreads();
  }
}

template <typename walker_t>
__global__ void walker_wb_zprs_curand(walker_t* walker, vtx_t* start_points,
                                      vtx_t* result_pool,
                                      TaskAssignments* assign,
                                      curandState* state) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int lid = threadIdx.x % WARP_SIZE;
  int gid = (blockDim.x * blockIdx.x) + tid;
  int wid = threadIdx.x / WARP_SIZE;

  gpu_graph* graph = walker->graph;
  int max_depth = walker->max_depth;

  curand_init(1337, gid, 0, state + gid);

  __shared__ Task tasks[BLOCK_SIZE];
  __shared__ int num_active_tasks;

  __shared__ int large_tasks[BLOCK_SIZE];
  __shared__ int block_task_count;

  __shared__ int walk_id;

  int num_walks = assign[bid].end - assign[bid].begin;

  int task_start = assign[bid].begin;

  vtx_t* local_start_points = start_points + task_start;
  vtx_t* local_result_pool = result_pool + task_start * max_depth;

  if (tid < num_walks) {
    local_result_pool[(u64)tid * max_depth] = local_start_points[tid];
    tasks[tid].init(graph, local_start_points, tid);
  } else {
    tasks[tid].length = -1;
  }

  if (tid == 0) {
    walk_id = min(BLOCK_SIZE, num_walks);
    num_active_tasks = num_walks;
    block_task_count = 0;
  }
  __syncthreads();

  while (num_active_tasks > 0) {
    for (int local_task_id = wid; local_task_id < BLOCK_SIZE;
         local_task_id += WARP_PER_BLK) {
      Task& task = tasks[local_task_id];
      if (task.length >= 0) {
        if (task.degree <= THRESHOLD) {
          vtx_t selected =
              sampler_warp<walker_t, curandState>(walker, &task, state + gid);

          if (lid == 0) {
            if (selected == -1) {
              task.length = max_depth;
            } else {
              selected = graph->adjncy[task.neighbor_offset + selected];
              local_result_pool[(u64)task.walker_id * max_depth + task.length] =
                  selected;
              task.update(graph, selected);
            }
          }
        } else {
          if (lid == 0) {
            int index = atomicAdd(&block_task_count, 1);
            large_tasks[index] = local_task_id;
          }
        }
      }
      __syncwarp(FULL_WARP_MASK);
    }
    __syncthreads();

    for (int i = 0; i < block_task_count; i++) {
      Task& task = tasks[large_tasks[i]];

      vtx_t selected =
          sampler_block<walker_t, curandState>(walker, &task, state + gid);

      if (tid == 0) {
        if (selected == -1) {
          task.length = max_depth;
        } else {
          selected = graph->adjncy[task.neighbor_offset + selected];
          local_result_pool[(u64)task.walker_id * max_depth + task.length] =
              selected;
          task.update(graph, selected);
        }
      }
      __syncthreads();
    }

    // remove finished tasks and add fetch new tasks
    if (walker->is_stop(tasks[tid].length, state + gid)) {
      atomicAdd(&num_active_tasks, -1);
      int local_walk_id = atomicAdd(&walk_id, 1);
      if (local_walk_id < num_walks) {
        local_result_pool[(u64)local_walk_id * max_depth] =
            local_start_points[local_walk_id];
        tasks[tid].init(graph, local_start_points, local_walk_id);
      } else {
        tasks[tid].length = -1;
      }
    }
    __syncthreads();
    if (tid == 0) {
      block_task_count = 0;
    }
    __syncthreads();
  }
}

// plain+rng
template <typename walker_t>
__global__ void walker_wb_dprs(walker_t* walker, vtx_t* start_points,
                               vtx_t* result_pool, TaskAssignments* assign) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int lid = threadIdx.x % WARP_SIZE;
  int gid = (blockDim.x * blockIdx.x) + tid;
  int wid = threadIdx.x / WARP_SIZE;

  gpu_graph* graph = walker->graph;
  int max_depth = walker->max_depth;

  __shared__ myrandStateArr state;
  myrand_init(1337, gid, 0, &state);

  __shared__ Task tasks[BLOCK_SIZE];
  __shared__ int num_active_tasks;

  __shared__ int large_tasks[BLOCK_SIZE];
  __shared__ int block_task_count;

  __shared__ int walk_id;

  int num_walks = assign[bid].end - assign[bid].begin;

  int task_start = assign[bid].begin;

  vtx_t* local_start_points = start_points + task_start;
  vtx_t* local_result_pool = result_pool + task_start * max_depth;

  if (tid < num_walks) {
    local_result_pool[(u64)tid * max_depth] = local_start_points[tid];
    tasks[tid].init(graph, local_start_points, tid);
  } else {
    tasks[tid].length = -1;
  }

  if (tid == 0) {
    walk_id = min(BLOCK_SIZE, num_walks);
    num_active_tasks = num_walks;
    block_task_count = 0;
  }
  __syncthreads();

  while (num_active_tasks > 0) {
    for (int local_task_id = wid; local_task_id < BLOCK_SIZE;
         local_task_id += WARP_PER_BLK) {
      Task& task = tasks[local_task_id];
      if (task.length >= 0) {
        if (task.degree <= THRESHOLD) {
          vtx_t selected = sampler_warp_oneloop<walker_t, myrandStateArr>(
              walker, &task, &state);

          if (lid == 0) {
            if (selected == -1) {
              task.length = max_depth;
            } else {
              selected = graph->adjncy[task.neighbor_offset + selected];
              local_result_pool[(u64)task.walker_id * max_depth + task.length] =
                  selected;
              task.update(graph, selected);
            }
          }
        } else {
          if (lid == 0) {
            int index = atomicAdd(&block_task_count, 1);
            large_tasks[index] = local_task_id;
          }
        }
      }
      __syncwarp(FULL_WARP_MASK);
    }
    __syncthreads();

    for (int i = 0; i < block_task_count; i++) {
      Task& task = tasks[large_tasks[i]];

      vtx_t selected = sampler_block_oneloop<walker_t, myrandStateArr>(
          walker, &task, &state);

      if (tid == 0) {
        if (selected == -1) {
          task.length = max_depth;
        } else {
          selected = graph->adjncy[task.neighbor_offset + selected];
          local_result_pool[(u64)task.walker_id * max_depth + task.length] =
              selected;
          task.update(graph, selected);
        }
      }
      __syncthreads();
    }

    // remove finished tasks and add fetch new tasks
    if (walker->is_stop(tasks[tid].length, &state)) {
      atomicAdd(&num_active_tasks, -1);
      int local_walk_id = atomicAdd(&walk_id, 1);
      if (local_walk_id < num_walks) {
        local_result_pool[(u64)local_walk_id * max_depth] =
            local_start_points[local_walk_id];
        tasks[tid].init(graph, local_start_points, local_walk_id);
      } else {
        tasks[tid].length = -1;
      }
    }
    __syncthreads();
    if (tid == 0) {
      block_task_count = 0;
    }
    __syncthreads();
  }
}

template <typename walker_t>
__global__ void walker_thread(walker_t* walker, vtx_t* start_points,
                              int* start_pointer, vtx_t* result_pool,
                              int walker_num) {
#ifdef MICRO_BENCH
  int bid = blockIdx.x;
  counter->block_begin(bid);
#endif
  int tid = threadIdx.x;
  int gid = (blockDim.x * blockIdx.x) + tid;

  gpu_graph* graph = walker->graph;
  int max_depth = walker->max_depth;

  __shared__ curandState state[BLOCK_SIZE];
  curand_init(1337, gid, 0, state + tid);

  __shared__ Task tasks[TASK_NUM];
  __shared__ int num_active_tasks;

  if (tid == 0) {
    num_active_tasks = 0;
  }
  __syncthreads();

  if (tid < TASK_NUM && tid < walker_num) {
    int old_size = atomicAdd(start_pointer, 1);
    if (old_size < walker_num) {
      vtx_t sp = start_points[old_size];
      result_pool[(u64)old_size * max_depth] = sp;
      tasks[tid].init(graph, sp, old_size);
      atomicAdd(&num_active_tasks, 1);
    } else {
      tasks[tid].length = -1;
    }
  }

  __syncthreads();

  while (num_active_tasks > 0) {
    for (int local_task_id = tid; local_task_id < TASK_NUM;
         local_task_id += BLOCK_SIZE) {
      Task& task = tasks[local_task_id];
      if (task.length >= 0) {
        vtx_t selected =
            sampler_thread<walker_t, curandState>(walker, &task, state + tid);

        if (selected == -1) {
          task.length = max_depth;
        } else {
          selected = graph->adjncy[task.neighbor_offset + selected];
          result_pool[(u64)task.walker_id * max_depth + task.length] = selected;

          task.update(graph, selected);
        }
      }
    }
    __syncthreads();

    // remove finished tasks and add fetch new tasks
    if (tid < TASK_NUM && walker->is_stop(tasks[tid].length, state + tid)) {
      int old_size = atomicAdd(start_pointer, 1);
      if (old_size < walker_num) {
        vtx_t sp = start_points[old_size];
        result_pool[(u64)old_size * max_depth] = sp;
        tasks[tid].init(graph, sp, old_size);
      } else {
        atomicAdd(&num_active_tasks, -1);
        tasks[tid].length = -1;
      }
    }
    __syncthreads();
  }
}

template <typename walker_t>
__global__ void walker_wb_dynamic_rjs(walker_t* walker, vtx_t* start_points,
                                      int* start_pointer, vtx_t* result_pool,
                                      int walker_num) {
#ifdef MICRO_BENCH
  int bid = blockIdx.x;
  counter->block_begin(bid);
#endif
  int tid = threadIdx.x;
  int lid = threadIdx.x % WARP_SIZE;
  int gid = (blockDim.x * blockIdx.x) + tid;
  int wid = threadIdx.x / WARP_SIZE;

  gpu_graph* graph = walker->graph;
  int max_depth = walker->max_depth;

  __shared__ myrandStateArr state;
  myrand_init(1337, gid, 0, &state);

  __shared__ Task tasks[TASK_NUM];
  __shared__ int num_active_tasks;

  __shared__ int large_tasks[TASK_NUM];
  __shared__ int block_task_count;

  if (tid == 0) {
    num_active_tasks = 0;
    block_task_count = 0;
  }
  __syncthreads();

  if (tid < TASK_NUM && tid < walker_num) {
    int old_size = atomicAdd(start_pointer, 1);
    if (old_size < walker_num) {
      vtx_t sp = start_points[old_size];
      result_pool[(u64)old_size * max_depth] = sp;
      tasks[tid].init(graph, sp, old_size);
      atomicAdd(&num_active_tasks, 1);
    } else {
      tasks[tid].length = -1;
    }
  } else if (tid < TASK_NUM && tid >= walker_num) {
    tasks[tid].length = -1;
  }

  __syncthreads();

#ifdef MICRO_BENCH
  counter->sample_begin(bid);
#endif
  while (num_active_tasks > 0) {
    for (int local_task_id = wid; local_task_id < TASK_NUM;
         local_task_id += WARP_PER_BLK) {
      Task& task = tasks[local_task_id];
      if (task.length >= 0) {
        if (task.degree <= THRESHOLD) {
#ifdef MICRO_BENCH
          u64 start = clock64();
#endif

          vtx_t selected =
              sampler_rjs_warp<walker_t, myrandStateArr>(walker, &task, &state);
#ifdef MICRO_BENCH
          counter->add_warp(bid, start, task.degree);
#endif
          if (lid == 0) {
            if (selected == -1) {
              task.length = max_depth;
            } else {
              selected = graph->adjncy[task.neighbor_offset + selected];
              result_pool[(u64)task.walker_id * max_depth + task.length] =
                  selected;
              task.update(graph, selected);
            }
          }
        } else {
          if (lid == 0) {
            int index = atomicAdd(&block_task_count, 1);
            large_tasks[index] = local_task_id;
          }
        }
      }
      __syncwarp(FULL_WARP_MASK);
    }
    __syncthreads();

    for (int i = 0; i < block_task_count; i++) {
      Task& h_task = tasks[large_tasks[i]];
#ifdef MICRO_BENCH
      u64 start = clock64();
#endif

      vtx_t selected =
          sampler_rjs_block<walker_t, myrandStateArr>(walker, &h_task, &state);

#ifdef MICRO_BENCH
      counter->add_block(bid, start, h_task.degree);
#endif
      if (tid == 0) {
        if (selected == -1) {
          h_task.length = max_depth;
        } else {
          selected = graph->adjncy[h_task.neighbor_offset + selected];
          result_pool[(u64)h_task.walker_id * max_depth + h_task.length] =
              selected;
          h_task.update(graph, selected);
        }
      }
      __syncthreads();
    }

    // remove finished tasks and add fetch new tasks
    if (tid < TASK_NUM && walker->is_stop(tasks[tid].length, &state)) {
      int old_size = atomicAdd(start_pointer, 1);
      if (old_size < walker_num) {
        vtx_t sp = start_points[old_size];
        result_pool[(u64)old_size * max_depth] = sp;
        tasks[tid].init(graph, sp, old_size);
      } else {
        atomicAdd(&num_active_tasks, -1);
        tasks[tid].length = -1;
      }
    }
    __syncthreads();
    if (tid == 0) {
      block_task_count = 0;
    }
    __syncthreads();
  }
#ifdef MICRO_BENCH
  counter->block_end(bid);
#endif
}
