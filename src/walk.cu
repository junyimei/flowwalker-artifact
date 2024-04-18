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
#include <cub/cub.cuh>
#include <time.h>
#include <gflags/gflags.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <typeinfo>
#include <fstream>
#include <iostream>
#include <map>
#include "gpu_graph.cuh"
#include "app.cuh"
#include "util.cuh"
#include "gpu_task.cuh"
#include "gpu_queue.cuh"
#include "walker.cuh"

DECLARE_double(tp);
DECLARE_double(p);
DECLARE_double(q);
DECLARE_bool(deepwalk);
DECLARE_bool(ppr);
DECLARE_bool(node2vec);
DECLARE_bool(metapath);
DECLARE_bool(syn);
DECLARE_bool(dprs);

__global__ void init_taskassignment(TaskAssignments *assign, int num_walkers, int block_num)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int task_per_block = num_walkers / block_num;
    int remainder = num_walkers % block_num;
    if (tid == 0)
    {
        if (bid < remainder)
        {
            task_per_block++;
            assign[bid].begin = task_per_block * bid;
        }
        else
        {
            assign[bid].begin = task_per_block * bid + remainder;
        }
        assign[bid].end = assign[bid].begin + task_per_block;
    }
}

__global__ void init_taskassignment_batch(TaskAssignments **assign, int num_walkers, int block_num, int batch_size)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    for (int i = 0; i < num_walkers; i += batch_size)
    {
        int b_num_walkers = min(batch_size, num_walkers - i);
        int task_per_block = b_num_walkers / block_num;
        int remainder = b_num_walkers % block_num;
        if (tid == 0)
        {
            if (bid < remainder)
            {
                task_per_block++;
                assign[i][bid].begin = i + task_per_block * bid;
            }
            else
            {
                assign[i][bid].begin = i + task_per_block * bid + remainder;
            }
            assign[i][bid].end = i + assign[i][bid].begin + task_per_block;
        }
    }
}

int get_block_num()
{
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    int n_sm = prop.multiProcessorCount;
    int b_per_sm = 1024 / BLOCK_SIZE;
    return n_sm * b_per_sm;
}

int get_block_num(int num_seeds)
{
    return (num_seeds + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

template <typename T>
T *get_device_ptr(T *host_ptr, u64 n)
{

    T *device_ptr;
    CUDA_RT_CALL(cudaMalloc(&device_ptr, (u64)sizeof(T) * n));
    CUDA_RT_CALL(cudaMemcpy(device_ptr, host_ptr, (u64)sizeof(T) * n, cudaMemcpyDefault));
    return device_ptr;
}

template <typename T>
T *get_device_ptr(u64 n, int init_val)
{
    T *device_ptr;
    CUDA_RT_CALL(cudaMalloc(&device_ptr, (u64)sizeof(T) * n));
    CUDA_RT_CALL(cudaMemset(device_ptr, init_val, (u64)sizeof(T) * n));
    return device_ptr;
}

template <typename T>
T *get_device_ptr_2d(u64 row, u64 col, int init_val)
{
    T **device_ptr;
    CUDA_RT_CALL(cudaMalloc(&device_ptr, (u64)sizeof(T) * row * col));
    CUDA_RT_CALL(cudaMemset(device_ptr, init_val, (u64)sizeof(T) * row * col));
    return device_ptr;
}

template <typename T>
T *get_device_ptr_um(T *host_ptr, u64 n)
{

    T *device_ptr;
    CUDA_RT_CALL(cudaMallocManaged(&device_ptr, (u64)sizeof(T) * n));
    CUDA_RT_CALL(cudaMemcpy(device_ptr, host_ptr, (u64)sizeof(T) * n, cudaMemcpyDefault));
    return device_ptr;
}

template <typename T>
T *get_device_ptr_um(u64 n, int init_val)
{
    T *device_ptr;
    CUDA_RT_CALL(cudaMallocManaged(&device_ptr, (u64)sizeof(T) * n));
    CUDA_RT_CALL(cudaMemset(device_ptr, init_val, (u64)sizeof(T) * n));
    return device_ptr;
}

queue_gpu<Task> *get_queuearr(int max_depth, int num_walkers)
{
    queue_gpu<Task> *walker_queue = new queue_gpu<Task>[max_depth];

    for (int i = 0; i < max_depth; i++)
    {
        walker_queue[i].init(num_walkers);
    }
    queue_gpu<Task> *walker_queue_ptr;
    CUDA_RT_CALL(cudaMalloc(&walker_queue_ptr, (u64)max_depth * sizeof(queue_gpu<Task>)));

    CUDA_RT_CALL(cudaMemcpy(walker_queue_ptr, walker_queue, (u64)max_depth * sizeof(queue_gpu<Task>), cudaMemcpyDefault));

    delete[] walker_queue;
    return walker_queue_ptr;
}

template <typename T>
queue_gpu<T> *get_queue(int num_walkers)
{
    queue_gpu<T> *walker_queue = new queue_gpu<T>;
    walker_queue->init(num_walkers);
    queue_gpu<T> *walker_queue_ptr = get_device_ptr<queue_gpu<T>>(walker_queue, 1);
    delete[] walker_queue;
    return walker_queue_ptr;
}

__global__ void init_state(myrandStateArr *state)
{

    //  init random number generator state
    int gid = (blockDim.x * blockIdx.x) + threadIdx.x;
    myrand_init(1337, gid, 0, state + blockIdx.x);
}

myrandStateArr *get_mystate(int num_blocks)
{
    myrandStateArr *state_ptr;
    CUDA_RT_CALL(cudaMalloc(&state_ptr, num_blocks * sizeof(myrandStateArr)));
    init_state<<<num_blocks, BLOCK_SIZE>>>(state_ptr);
    return state_ptr;
}

template <typename walker_t>
double timing(walker_t *walker_ptr, vtx_t *start_points_ptr, vtx_t *result_pool_ptr,
              int block_num, int num_walkers, int max_depth, walk_mode type = wb_dynamic)
{
    printf("========start timing\n");
#ifdef MICRO_BENCH
    metrics *h_counter = new metrics(block_num);
    metrics *d_counter = get_device_ptr<metrics>(h_counter, 1);
#endif

    double start_time, total_time;

    if (type == queue)
    {
        queue_gpu<Task> *walker_queue_ptr = get_queuearr(max_depth, num_walkers);
        queue_gpu<Task> *high_queue_ptr = get_queuearr(max_depth, num_walkers);
        start_time = wtime();
#ifdef MICRO_BENCH
        walker_queue<<<block_num, BLOCK_SIZE>>>(walker_ptr, walker_queue_ptr,
                                                high_queue_ptr, start_points_ptr, result_pool_ptr, num_walkers, d_counter);
#else
        walker_queue<<<block_num, BLOCK_SIZE>>>(walker_ptr, walker_queue_ptr,
                                                high_queue_ptr, start_points_ptr, result_pool_ptr, num_walkers);
#endif
        CUDA_RT_CALL(cudaDeviceSynchronize());
        total_time = wtime() - start_time;
    }
    else if (type == twb_dynamic)
    {
        int *start_pointer = get_device_ptr<int>(1, 0);
        start_time = wtime();
#ifdef MICRO_BENCH
        walker_twb_dynamic<<<block_num, BLOCK_SIZE>>>(walker_ptr, start_points_ptr, start_pointer, result_pool_ptr, num_walkers, d_counter);
#else
        walker_twb_dynamic<<<block_num, BLOCK_SIZE>>>(walker_ptr, start_points_ptr, start_pointer, result_pool_ptr, num_walkers);
#endif
        CUDA_RT_CALL(cudaDeviceSynchronize());
        total_time = wtime() - start_time;
    }
    else if (type == wb_dynamic)
    {
        int *start_pointer = get_device_ptr<int>(1, 0);

        start_time = wtime();
        if (FLAGS_dprs == true)
        {
#ifdef MICRO_BENCH
            walker_wb_dynamic_dprs<<<block_num, BLOCK_SIZE>>>(walker_ptr, start_points_ptr, start_pointer, result_pool_ptr, num_walkers, d_counter);
#else
            walker_wb_dynamic_dprs<<<block_num, BLOCK_SIZE>>>(walker_ptr, start_points_ptr, start_pointer, result_pool_ptr, num_walkers);
#endif
        }
        else
        {
#ifdef MICRO_BENCH
            walker_wb_dynamic_zprs<<<block_num, BLOCK_SIZE>>>(walker_ptr, start_points_ptr, start_pointer, result_pool_ptr, num_walkers, d_counter);
#else
            walker_wb_dynamic_zprs<<<block_num, BLOCK_SIZE>>>(walker_ptr, start_points_ptr, start_pointer, result_pool_ptr, num_walkers);
#endif
        }
        CUDA_RT_CALL(cudaDeviceSynchronize());
        total_time = wtime() - start_time;
    }
    else if (type == wb_dynamic_dprs)
    {
        int *start_pointer = get_device_ptr<int>(1, 0);

        start_time = wtime();
        walker_wb_dynamic_dprs<<<block_num, BLOCK_SIZE>>>(walker_ptr, start_points_ptr, start_pointer, result_pool_ptr, num_walkers);
        CUDA_RT_CALL(cudaDeviceSynchronize());
        total_time = wtime() - start_time;
    }
    else if (type == w_dynamic)
    {
        int *start_pointer = get_device_ptr<int>(1, 0);
        start_time = wtime();
#ifdef MICRO_BENCH
        walker_w_dynamic<<<block_num, BLOCK_SIZE>>>(walker_ptr, start_points_ptr, start_pointer, result_pool_ptr, num_walkers, d_counter);
#else
        walker_w_dynamic<<<block_num, BLOCK_SIZE>>>(walker_ptr, start_points_ptr, start_pointer, result_pool_ptr, num_walkers);
#endif
        CUDA_RT_CALL(cudaDeviceSynchronize());
        total_time = wtime() - start_time;
    }
    else if (type == b_dynamic)
    {
        int *start_pointer = get_device_ptr<int>(1, 0);
        start_time = wtime();
#ifdef MICRO_BENCH
        walker_b_dynamic<<<block_num, BLOCK_SIZE>>>(walker_ptr, start_points_ptr, start_pointer, result_pool_ptr, num_walkers, d_counter);
#else
        walker_b_dynamic<<<block_num, BLOCK_SIZE>>>(walker_ptr, start_points_ptr, start_pointer, result_pool_ptr, num_walkers);
#endif
        CUDA_RT_CALL(cudaDeviceSynchronize());
        total_time = wtime() - start_time;
    }
    else
    {
        TaskAssignments *assign_ptr = get_device_ptr<TaskAssignments>(block_num, 0);
        init_taskassignment<<<block_num, BLOCK_SIZE>>>(assign_ptr, num_walkers, block_num);
        CUDA_RT_CALL(cudaDeviceSynchronize());
        if (type == wp)
        {
            start_time = wtime();
#ifdef MICRO_BENCH
            walker_w<<<block_num, BLOCK_SIZE>>>(walker_ptr, start_points_ptr, result_pool_ptr, assign_ptr, d_counter);
#else
            walker_w<<<block_num, BLOCK_SIZE>>>(walker_ptr, start_points_ptr, result_pool_ptr, assign_ptr);
#endif
            CUDA_RT_CALL(cudaDeviceSynchronize());
            total_time = wtime() - start_time;
        }
        else if (type == wb)
        {
            start_time = wtime();
#ifdef MICRO_BENCH
            walker_wb<<<block_num, BLOCK_SIZE>>>(walker_ptr, start_points_ptr, result_pool_ptr, assign_ptr, d_counter);
#else
            walker_wb<<<block_num, BLOCK_SIZE>>>(walker_ptr, start_points_ptr, result_pool_ptr, assign_ptr);
#endif
            CUDA_RT_CALL(cudaDeviceSynchronize());
            total_time = wtime() - start_time;
        }
        else if (type == twb)
        {
            start_time = wtime();
#ifdef MICRO_BENCH
            walker_twb<<<block_num, BLOCK_SIZE>>>(walker_ptr, start_points_ptr, result_pool_ptr, assign_ptr, d_counter);
#else
            walker_twb<<<block_num, BLOCK_SIZE>>>(walker_ptr, start_points_ptr, result_pool_ptr, assign_ptr);
#endif
            CUDA_RT_CALL(cudaDeviceSynchronize());
            total_time = wtime() - start_time;
        }
        else if (type == tw)
        {
            start_time = wtime();
#ifdef MICRO_BENCH
            walker_tw<<<block_num, BLOCK_SIZE>>>(walker_ptr, start_points_ptr, result_pool_ptr, assign_ptr, d_counter);
#else
            walker_tw<<<block_num, BLOCK_SIZE>>>(walker_ptr, start_points_ptr, result_pool_ptr, assign_ptr);
#endif

            CUDA_RT_CALL(cudaDeviceSynchronize());
            total_time = wtime() - start_time;
        }
        else if (type == wb_curand)
        {
            curandState *state_ptr;
            CUDA_RT_CALL(cudaMalloc(&state_ptr, block_num * BLOCK_SIZE * sizeof(curandState)));
            start_time = wtime();
            walker_wb_dprs_curand<<<block_num, BLOCK_SIZE>>>(walker_ptr, start_points_ptr, result_pool_ptr, assign_ptr, state_ptr);
            CUDA_RT_CALL(cudaDeviceSynchronize());
            total_time = wtime() - start_time;
        }
        else if (type == wb_dprs)
        {
            start_time = wtime();
            walker_wb_dprs<<<block_num, BLOCK_SIZE>>>(walker_ptr, start_points_ptr, result_pool_ptr, assign_ptr);
            CUDA_RT_CALL(cudaDeviceSynchronize());
            total_time = wtime() - start_time;
        }
        else if (type == wb_curand_zprs)
        {
            curandState *state_ptr;
            CUDA_RT_CALL(cudaMalloc(&state_ptr, block_num * BLOCK_SIZE * sizeof(curandState)));
            start_time = wtime();
            walker_wb_zprs_curand<<<block_num, BLOCK_SIZE>>>(walker_ptr, start_points_ptr, result_pool_ptr, assign_ptr, state_ptr);
            CUDA_RT_CALL(cudaDeviceSynchronize());
            total_time = wtime() - start_time;
        }
    }

#ifdef MICRO_BENCH
    CUDA_RT_CALL(cudaMemcpy(h_counter, d_counter, sizeof(metrics), cudaMemcpyDeviceToHost));
    h_counter->print_all(get_clk());
#endif
    return total_time * 1000;
}

// #define TEST_RJS

template <typename walker_t>
double timing_batch_async(walker_t *walker_ptr, vtx_t *start_points, vtx_t *result_pool, int batch_size, int num_walkers, int max_depth, int block_num)
{
    cudaStream_t *streams = new cudaStream_t[2];
    for (int i = 0; i < 2; i++)
    {
        CUDA_RT_CALL(cudaStreamCreate(&streams[i]));
    }
    vtx_t *start_points_ptr = get_device_ptr<vtx_t>((u64)batch_size * 2, 0);
    vtx_t *result_pool_ptr = get_device_ptr<vtx_t>((u64)batch_size * max_depth * 2, -1);

    int *start_pointer = get_device_ptr<int>(2, 0);
    printf("========start timing\n");
    double start_time, total_time;
    start_time = wtime();

    for (int i = 0; i < num_walkers; i += batch_size * 2)
    {
        int j = i + batch_size;
        int batch_num1 = min(batch_size, num_walkers - i);
        int batch_num2 = min(batch_size, num_walkers - j);

        CUDA_RT_CALL(cudaMemcpyAsync(start_points_ptr, start_points + i, (u64)sizeof(vtx_t) * batch_num1, cudaMemcpyHostToDevice, streams[0]));
        CUDA_RT_CALL(cudaMemsetAsync(result_pool_ptr, -1, (u64)sizeof(vtx_t) * batch_num1 * max_depth, streams[0]));
        CUDA_RT_CALL(cudaMemsetAsync(start_pointer, 0, sizeof(int), streams[0]));
#ifndef TEST_RJS
        if (FLAGS_node2vec)
        {
            walker_wb_dynamic_dprs<<<block_num, BLOCK_SIZE, 0, streams[0]>>>(walker_ptr, start_points_ptr, start_pointer, result_pool_ptr, batch_num1);
        }
        else
        {
            walker_wb_dynamic_zprs<<<block_num, BLOCK_SIZE, 0, streams[0]>>>(walker_ptr, start_points_ptr, start_pointer, result_pool_ptr, batch_num1);
        }
#else
        walker_wb_dynamic_rjs<<<block_num, BLOCK_SIZE, 0, streams[0]>>>(walker_ptr, start_points_ptr, start_pointer, result_pool_ptr, batch_num1);
#endif

        CUDA_RT_CALL(cudaMemcpyAsync(result_pool + (u64)i * max_depth, result_pool_ptr, (u64)sizeof(vtx_t) * batch_num1 * max_depth, cudaMemcpyDeviceToHost, streams[0]));

        if (batch_num2 > 0)
        {
            vtx_t *b_start_points_ptr = start_points_ptr + batch_size;
            vtx_t *b_result_pool_ptr = result_pool_ptr + (u64)batch_size * max_depth;

            CUDA_RT_CALL(cudaMemcpyAsync(b_start_points_ptr, start_points + j, (u64)sizeof(vtx_t) * batch_num2, cudaMemcpyHostToDevice, streams[1]));
            CUDA_RT_CALL(cudaMemsetAsync(b_result_pool_ptr, -1, (u64)sizeof(vtx_t) * batch_num2 * max_depth, streams[1]));
            CUDA_RT_CALL(cudaMemsetAsync(start_pointer + 1, 0, sizeof(int), streams[1]));
#ifndef TEST_RJS
            if (FLAGS_node2vec)
            {
                walker_wb_dynamic_dprs<<<block_num, BLOCK_SIZE, 0, streams[1]>>>(walker_ptr, b_start_points_ptr, start_pointer + 1, b_result_pool_ptr, batch_num2);
            }
            else
            {
                walker_wb_dynamic_zprs<<<block_num, BLOCK_SIZE, 0, streams[1]>>>(walker_ptr, b_start_points_ptr, start_pointer + 1, b_result_pool_ptr, batch_num2);
            }
#else
            walker_wb_dynamic_rjs<<<block_num, BLOCK_SIZE, 0, streams[1]>>>(walker_ptr, b_start_points_ptr, start_pointer + 1, b_result_pool_ptr, batch_num2);
#endif
            CUDA_RT_CALL(cudaMemcpyAsync(result_pool + (u64)j * max_depth, b_result_pool_ptr, (u64)sizeof(vtx_t) * batch_num2 * max_depth, cudaMemcpyDeviceToHost, streams[1]));
        }
    }
    for (int i = 0; i < 2; i++)
    {
        CUDA_RT_CALL(cudaStreamSynchronize(streams[i]));
    }
    total_time = wtime() - start_time;

    for (int i = 0; i < 2; i++)
    {
        CUDA_RT_CALL(cudaStreamDestroy(streams[i]));
    }

    return total_time * 1000;
}

template <typename walker_t>
double timing_batch_sync(walker_t *walker_ptr, vtx_t *start_points, vtx_t *result_pool, int batch_size, int num_walkers, int max_depth, int block_num)
{

    int stream_num = (num_walkers % batch_size == 0) ? num_walkers / batch_size : num_walkers / batch_size + 1;
    vtx_t *start_points_ptr = get_device_ptr<vtx_t>(batch_size, 0);
    vtx_t *result_pool_ptr = get_device_ptr<vtx_t>((u64)batch_size * max_depth, -1);

    int *start_pointer = get_device_ptr<int>(1, 0);
    printf("========start timing\n");
    double start_time, compute_time, total_time, transfer_s, transfer_time;

    compute_time = 0;
    transfer_time = 0;

    total_time = wtime();
    for (int i = 0; i < stream_num; i++)
    {
        int batch_num = min(batch_size, num_walkers - i * batch_size);
        // printf("batch:%d,batch num:%d\n", i, batch_num);
        CUDA_RT_CALL(cudaMemcpy(start_points_ptr, start_points + (u64)i * batch_size, sizeof(vtx_t) * batch_num, cudaMemcpyHostToDevice));
        CUDA_RT_CALL(cudaMemset(result_pool_ptr, -1, (u64)sizeof(vtx_t) * batch_num * max_depth));
        CUDA_RT_CALL(cudaMemset(start_pointer, 0, sizeof(int)));

        start_time = wtime();
        if (FLAGS_node2vec)
        {
            walker_wb_dynamic_dprs<<<block_num, BLOCK_SIZE>>>(walker_ptr, start_points_ptr, start_pointer, result_pool_ptr, batch_num);
        }
        else
        {
            walker_wb_dynamic_zprs<<<block_num, BLOCK_SIZE>>>(walker_ptr, start_points_ptr, start_pointer, result_pool_ptr, batch_num);
        }

        CUDA_RT_CALL(cudaDeviceSynchronize());
        compute_time += wtime() - start_time;

        transfer_s = wtime();
        CUDA_RT_CALL(cudaMemcpy(result_pool + (u64)i * batch_size * max_depth, result_pool_ptr, (u64)sizeof(vtx_t) * batch_num * max_depth, cudaMemcpyDeviceToHost));
        transfer_time += wtime() - transfer_s;
    }
    CUDA_RT_CALL(cudaDeviceSynchronize());
    total_time = wtime() - total_time;

    printf("compute time=%.6f ms, transfer time=%.6f ms\n", compute_time * 1000, transfer_time * 1000);
    return total_time * 1000;
}

// double walk_test(gpu_graph *graph, vtx_t *start_points, int max_depth, int num_walkers, walk_mode type, int *schema, int schema_len)
double walk_test(vtx_t *&result_pool_ptr, gpu_graph *graph, vtx_t *start_points, int max_depth, int num_walkers, walk_mode type, int *schema, int schema_len)
{
    LOG("%s\n", __FUNCTION__);
    double total_time;

    gpu_graph *graph_ptr = get_device_ptr<gpu_graph>(graph, 1);
    vtx_t *start_points_ptr = get_device_ptr<vtx_t>(start_points, num_walkers);
    result_pool_ptr = get_device_ptr<vtx_t>((u64)num_walkers * max_depth, -1);

    // int sm_count = get_block_num(1);
    // int block_num = get_block_num(num_walkers);
    int block_num = get_block_num() * 2;

    if (FLAGS_deepwalk)
    {
        Deepwalk *walker = new Deepwalk(graph_ptr, max_depth);
        Deepwalk *walker_ptr = get_device_ptr<Deepwalk>(walker, 1);
        total_time = timing<Deepwalk>(walker_ptr, start_points_ptr, result_pool_ptr, block_num, num_walkers, max_depth, type);
    }
    else if (FLAGS_ppr)
    {
        PPR *walker = new PPR(graph_ptr, max_depth, FLAGS_tp);
        PPR *walker_ptr = get_device_ptr<PPR>(walker, 1);
        total_time = timing<PPR>(walker_ptr, start_points_ptr, result_pool_ptr, block_num, num_walkers, max_depth, type);
    }
    else if (FLAGS_node2vec)
    {
        Node2vec *walker = new Node2vec(graph_ptr, max_depth, FLAGS_p, FLAGS_q);
        Node2vec *walker_ptr = get_device_ptr<Node2vec>(walker, 1);
        total_time = timing<Node2vec>(walker_ptr, start_points_ptr, result_pool_ptr, block_num, num_walkers, max_depth, type);
    }
    else if (FLAGS_metapath)
    {
        int *schema_ptr = get_device_ptr<int>(schema, schema_len);
        Metapath *walker = new Metapath(graph_ptr, max_depth, schema_ptr, schema_len);
        Metapath *walker_ptr = get_device_ptr<Metapath>(walker, 1);
        total_time = timing<Metapath>(walker_ptr, start_points_ptr, result_pool_ptr, block_num, num_walkers, max_depth, type);
    }
    else
    {
        printf("Please choose a walk mode\n");
        exit(0);
    }
    CUDA_RT_CALL(cudaDeviceSynchronize());

    LOG("grid:%d,block:%d,sampling time:%.6f ms\n", block_num, BLOCK_SIZE, total_time);

    return total_time;
}

double walk_batch(vtx_t *&result_pool, gpu_graph *graph, vtx_t *start_points, int max_depth, int num_walkers, int batch_size, int *schema, int schema_len)
{
    LOG("%s\n", __FUNCTION__);
    double total_time;

    gpu_graph *graph_ptr = get_device_ptr<gpu_graph>(graph, 1);

    // vtx_t *result_pool;
    cudaMallocHost(&result_pool, (u64)num_walkers * max_depth * sizeof(vtx_t));

    // int sm_count = get_block_num(1);
    // int block_num = get_block_num(min(batch_size, num_walkers));
    int block_num = get_block_num() * 2;

    if (FLAGS_deepwalk)
    {
        Deepwalk *walker = new Deepwalk(graph_ptr, max_depth);
        Deepwalk *walker_ptr = get_device_ptr(walker, 1);
        if (FLAGS_syn)
            total_time = timing_batch_sync(walker_ptr, start_points, result_pool, batch_size, num_walkers, max_depth, block_num);
        else
            total_time = timing_batch_async(walker_ptr, start_points, result_pool, batch_size, num_walkers, max_depth, block_num);
    }
    else if (FLAGS_ppr)
    {
        PPR *walker = new PPR(graph_ptr, max_depth, FLAGS_tp);
        PPR *walker_ptr = get_device_ptr<PPR>(walker, 1);
        if (FLAGS_syn)
            total_time = timing_batch_sync(walker_ptr, start_points, result_pool, batch_size, num_walkers, max_depth, block_num);
        else
            total_time = timing_batch_async(walker_ptr, start_points, result_pool, batch_size, num_walkers, max_depth, block_num);
    }
    else if (FLAGS_node2vec)
    {
        Node2vec *walker = new Node2vec(graph_ptr, max_depth, FLAGS_p, FLAGS_q);
        Node2vec *walker_ptr = get_device_ptr<Node2vec>(walker, 1);
        if (FLAGS_syn)
            total_time = timing_batch_sync(walker_ptr, start_points, result_pool, batch_size, num_walkers, max_depth, block_num);
        else
            total_time = timing_batch_async(walker_ptr, start_points, result_pool, batch_size, num_walkers, max_depth, block_num);
    }
    else if (FLAGS_metapath)
    {
        int *schema_ptr = get_device_ptr<int>(schema, schema_len);
        Metapath *walker = new Metapath(graph_ptr, max_depth, schema_ptr, schema_len);
        Metapath *walker_ptr = get_device_ptr<Metapath>(walker, 1);
        if (FLAGS_syn)
            total_time = timing_batch_sync(walker_ptr, start_points, result_pool, batch_size, num_walkers, max_depth, block_num);
        else
            total_time = timing_batch_async(walker_ptr, start_points, result_pool, batch_size, num_walkers, max_depth, block_num);
    }
    else
    {
        printf("Please choose a walk mode\n");
        exit(0);
    }

    CUDA_RT_CALL(cudaDeviceSynchronize());

    LOG("grid:%d,block:%d,sampling time:%.6f ms\n", block_num, BLOCK_SIZE, total_time);

    return total_time;
}