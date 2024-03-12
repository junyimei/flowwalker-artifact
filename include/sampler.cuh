#pragma once
#include <algorithm>
#include <random>
#include <cub/cub.cuh>

#include "app.cuh"
#include "gpu_task.cuh"
#include "myrand.cuh"

template <typename walker_t, typename state_t>
__device__ inline vtx_t sampler_thread(walker_t *walker, Task *task, state_t *state)
{
    vtx_t selected_id = -1;
    vtx_t size = task->degree;

    weight_t weight_sum = 0;
    for (int i = 0; i < size; i++)
    {
        weight_t w = walker->get_weight(task, i);
        weight_sum += w;
        if (w > 0)
        {
            if (selected_id == -1)
                selected_id = 0;
            if (myrand_uniform(state) < w / weight_sum)
            {
                selected_id = i;
            }
        }
    }
    return selected_id;
}

template <typename walker_t, typename state_t>
__device__ inline vtx_t sampler_warp(walker_t *walker, Task *task, state_t *state)
{
    vtx_t size = task->degree;
    int lid = threadIdx.x % WARP_SIZE;

    ll selected_id = -1;

    weight_t weight_sum = 0;

    for (int i = lid; i < size; i += WARP_SIZE)
    {
        weight_t w = walker->get_weight(task, i);
        weight_sum += w;
    }

    // calculate prefix sum
    weight_t local_sum = weight_sum;
    for (int mask = 1; mask < WARP_SIZE; mask <<= 1)
    {
        weight_t w = __shfl_up_sync(FULL_WARP_MASK, weight_sum, mask, WARP_SIZE);
        if (lid >= mask)
            weight_sum += w;
    }

    weight_sum -= local_sum;

    for (int i = lid; i < size; i += WARP_SIZE)
    {
        weight_t w = walker->get_weight(task, i);
        weight_sum += w;

        if (w > 0)
        {
            // printf("walker id=%d, w: %f, i=%d\n", task->walker_id, w, i);
            if (selected_id == -1)
                selected_id = 0;
            if (myrand_uniform(state) < w / weight_sum)
            {

                selected_id = i + (ll)size * lid;
            }
        }
    }

    selected_id = warpReduceMax(selected_id);
    // if (lid == 0)
    //     printf("walker id=%d, selected_id=%lld, len=%d, size=%d,return=%d,%d\n", task->walker_id, selected_id, task->length, size, (int)(selected_id % (ll)size), (-1) % 1);

    return (selected_id == -1) ? -1 : selected_id % size;
}

template <typename walker_t, typename state_t>
__device__ inline vtx_t sampler_block(walker_t *walker, Task *task, state_t *state)
{
    vtx_t size = task->degree;
    int tid = threadIdx.x;

    weight_t weight_sum = 0;
    ll selected_id = -1;

    for (int i = tid; i < size; i += BLOCK_SIZE)
    {
        weight_t w = walker->get_weight(task, i);
        weight_sum += w;
    }

    // Compute the prefix sum for local weight sum.
    typedef cub::BlockScan<weight_t, BLOCK_SIZE> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage_sum;
    BlockScan(temp_storage_sum).ExclusiveSum(weight_sum, weight_sum);

    for (int i = tid; i < size; i += BLOCK_SIZE)
    {
        weight_t w = walker->get_weight(task, i);
        weight_sum += w;
        if (w > 0)
        {
            if (selected_id == -1)
                selected_id = 0;
            float r = myrand_uniform(state);
            if (r < w / weight_sum)
            {

                selected_id = i + (ll)size * tid;
            }
        }
    }

    typedef cub::BlockReduce<vtx_t, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage_reduce;

    __syncthreads();
    ll selected = BlockReduce(temp_storage_reduce).Reduce(selected_id, cub::Max());
    __syncthreads();

    return (selected == -1) ? -1 : selected % size;
}

template <typename walker_t, typename state_t>
__device__ inline vtx_t sampler_warp_oneloop(walker_t *walker, Task *task, state_t *state)
{
    vtx_t size = task->degree;
    int lid = threadIdx.x % WARP_SIZE;

    weight_t w = 0;
    weight_t prev_ite_sum = 0;

    vtx_t selected_id = -1;
    for (int i = lid; i < size + WARP_SIZE - size % WARP_SIZE; i += WARP_SIZE)
    {
        weight_t weight_sum = (i < size) ? walker->get_weight(task, i) : 0;
        w = weight_sum;
        if (w > 0 && selected_id == -1)
            selected_id = 0;
        for (int mask = 1; mask < 32; mask <<= 1)
        {
            weight_t val = __shfl_up_sync(FULL_WARP_MASK, weight_sum, mask, WARP_SIZE);
            if (lid >= mask)
                weight_sum += val;
        }
        weight_sum += prev_ite_sum;
        if (i < size && myrand_uniform(state) < w / weight_sum)
            selected_id = i;
        prev_ite_sum = __shfl_sync(FULL_WARP_MASK, weight_sum, 31, WARP_SIZE);
    }
    selected_id = warpReduceMax(selected_id);

    return selected_id;
}

template <typename walker_t, typename state_t>
__device__ inline vtx_t sampler_block_oneloop(walker_t *walker, Task *task, state_t *state)
{
    vtx_t size = task->degree;
    int tid = threadIdx.x;

    __shared__ weight_t prefix_sum;
    prefix_sum = 0;

    vtx_t selected_id = -1;
    for (int i = tid; i < size + BLOCK_SIZE - size % BLOCK_SIZE; i += BLOCK_SIZE)
    {
        weight_t w = (i < size) ? walker->get_weight(task, i) : 0;
        vtx_t local_selected_id = -1;
        // Load data into shared memory.
        weight_t weight_sum = (tid == 0) ? prefix_sum + w : w;

        // Conduct inclusive prefix sum.
        typedef cub::BlockScan<weight_t, BLOCK_SIZE> BlockScan;
        __shared__ typename BlockScan::TempStorage temp_storage_sum;

        BlockScan(temp_storage_sum).InclusiveSum(weight_sum, weight_sum);

        // Sample one index.
        if (w > 0 && i < size)
        {
            if (local_selected_id == -1)
                local_selected_id = 0;
            if (myrand_uniform(state) < w / weight_sum)
                local_selected_id = i;
        }

        // Max operation.
        typedef cub::BlockReduce<vtx_t, BLOCK_SIZE> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage_reduce;

        selected_id = max(selected_id, BlockReduce(temp_storage_reduce).Reduce(local_selected_id, cub::Max()));

        if (tid == BLOCK_SIZE - 1)
        {
            prefix_sum = weight_sum;
        }
        __syncthreads();
    }
    return selected_id;
}

template <typename walker_t, typename state_t>
__device__ inline vtx_t sampler_rjs_warp(walker_t *walker, Task *task, state_t *state)
{
    vtx_t size = task->degree;
    int lid = threadIdx.x % WARP_SIZE;

    vtx_t selected = -1;
    weight_t local_max_weight = 0;
    for (int i = lid; i < size; i += WARP_SIZE)
    {
        weight_t w = walker->get_weight(task, i);
        local_max_weight = max(local_max_weight, w);
        // if (w > 0 && selected_id == -1)
        //     selected_id = 0;
    }
    __syncwarp();

    weight_t max_weight = warpReduceMax(local_max_weight);
    __syncwarp();
    // vtx_t selected = 0;
    if (lid == 0 && max_weight > 0)
    {
        // max_weight = max_weight;
        vtx_t x;
        weight_t y, prob;
        do
        {
            x = myrand(state) % size;
            y = myrand_uniform(state) * max_weight;
            prob = walker->get_weight(task, x);
        } while (y > prob);
        selected = x;
    }
    __syncwarp();

    return selected;
}

template <typename walker_t, typename state_t>
__device__ inline vtx_t sampler_rjs_block(walker_t *walker, Task *task, state_t *state)
{
    vtx_t size = task->degree;
    int tid = threadIdx.x;
    weight_t local_max_weight = 0;
    vtx_t selected = -1;
    for (int i = tid; i < size; i += BLOCK_SIZE)
    {
        weight_t w = walker->get_weight(task, i);
        local_max_weight = max(local_max_weight, w);
        // if (w > 0 && selected == -1)
        //     selected = 0;
    }
    __syncthreads();
    typedef cub::BlockReduce<weight_t, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage_reduce;

    weight_t max_w = BlockReduce(temp_storage_reduce).Reduce(local_max_weight, cub::Max());
    __syncthreads();

    if (tid == 0 && max_w > 0)
    {
        vtx_t x;
        weight_t y, prob;
        do
        {
            x = myrand(state) % size;
            y = myrand_uniform(state) * max_w;
            prob = walker->get_weight(task, x);

        } while (y > prob);
        selected = x;
    }

    __syncthreads();

    return selected;
}

/*
======================================
Below functions are used for testing
======================================
*/

template <typename T>
__device__ inline vtx_t test_thread_sampler(const weight_t *weights, int size, T *state)
{
    weight_t sum = weights[0];
    vtx_t selected_id = 0;
    for (int i = 1; i < size; i++)
    {
        weight_t w = weights[i];
        sum += w;
        selected_id = myrand_uniform(state) < w / sum ? i : selected_id;
    }
    return selected_id;
}

template <typename T>
__device__ inline vtx_t test_warp_sampler(const weight_t *weights, int size, T *state)
{
    int lid = threadIdx.x % WARP_SIZE;

    vtx_t selected_id = 0;

    weight_t weight_sum = 0;

    for (int i = lid; i < size; i += WARP_SIZE)
    {
        weight_sum += weights[i];
    }

    weight_t local_sum = weight_sum;
    for (int mask = 1; mask < WARP_SIZE; mask <<= 1)
    {
        weight_t w = __shfl_up_sync(FULL_WARP_MASK, weight_sum, mask, WARP_SIZE);
        if (lid >= mask)
            weight_sum += w;
    }

    weight_sum -= local_sum;

    for (int i = lid; i < size; i += WARP_SIZE)
    {
        weight_t w = weights[i];
        weight_sum += w;

        if (myrand_uniform(state) < w / weight_sum)
        {

            selected_id = i + (ll)size * lid;
        }
    }

    selected_id = warpReduceMax(selected_id);

    return selected_id % size;
    // return (selected_id == -1) ? -1 : selected_id % size;
}

template <typename T>
__device__ inline vtx_t test_block_sampler(const weight_t *weights, int size, T *state)
{

    int tid = threadIdx.x;

    // Compute the local sum weight for each thread.
    weight_t weight_sum = 0;
    ll selected_id = 0;
    for (int i = tid; i < size; i += BLOCK_SIZE)
    {
        weight_sum += weights[i];
    }

    // Compute the prefix sum for local weight sum.
    typedef cub::BlockScan<weight_t, BLOCK_SIZE> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage_sum;

    BlockScan(temp_storage_sum).ExclusiveSum(weight_sum, weight_sum);

    for (int i = tid; i < size; i += BLOCK_SIZE)
    {
        weight_t w = weights[i];
        weight_sum += w;

        if (myrand_uniform(state) < w / weight_sum)
        {

            selected_id = i + (ll)size * tid;
        }
    }

    typedef cub::BlockReduce<vtx_t, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage_reduce;

    __syncthreads();
    ll selected = BlockReduce(temp_storage_reduce).Reduce(selected_id, cub::Max());

    __syncthreads();

    return selected % size;
    // return (selected == -1) ? -1 : selected % size;
}

template <typename T>
__device__ inline vtx_t test_warp_sampler(const weight_t *weights, int size, T *state, u64 *clk)
{

    u64 start_clk = clock64();

    int lid = threadIdx.x % WARP_SIZE;

    vtx_t selected_id = 0;

    weight_t weight_sum = 0;

    for (int i = lid; i < size; i += WARP_SIZE)
    {
        weight_sum += weights[i];
    }

    weight_t local_sum = weight_sum;
    for (int mask = 1; mask < WARP_SIZE; mask <<= 1)
    {
        weight_t w = __shfl_up_sync(FULL_WARP_MASK, weight_sum, mask, WARP_SIZE);
        if (lid >= mask)
            weight_sum += w;
    }

    weight_sum -= local_sum;

    for (int i = lid; i < size; i += WARP_SIZE)
    {
        weight_t w = weights[i];
        weight_sum += w;

        if (myrand_uniform(state) < w / weight_sum)
        {

            selected_id = i + (ll)size * lid;
        }
    }

    selected_id = warpReduceMax(selected_id);

    if (lid == 0)
        *clk += clock64() - start_clk;

    return selected_id % size;
}

template <typename T>
__device__ inline vtx_t test_block_sampler(const weight_t *weights, int size, T *state, u64 *clk)
{

    u64 start_clk = clock64();

    int tid = threadIdx.x;

    // Compute the local sum weight for each thread.
    weight_t weight_sum = 0;
    ll selected_id = 0;
    for (int i = tid; i < size; i += BLOCK_SIZE)
    {
        weight_sum += weights[i];
    }

    // Compute the prefix sum for local weight sum.
    typedef cub::BlockScan<weight_t, BLOCK_SIZE> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage_sum;

    BlockScan(temp_storage_sum).ExclusiveSum(weight_sum, weight_sum);

    for (int i = tid; i < size; i += BLOCK_SIZE)
    {
        weight_t w = weights[i];
        weight_sum += w;

        if (myrand_uniform(state) < w / weight_sum)
        {

            selected_id = i + (ll)size * tid;
        }
    }

    typedef cub::BlockReduce<vtx_t, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage_reduce;

    __syncthreads();
    ll selected = BlockReduce(temp_storage_reduce).Reduce(selected_id, cub::Max());

    __syncthreads();

    if (tid == 0)
        *clk += clock64() - start_clk;

    return selected % size;
}

template <typename T>
__device__ inline vtx_t test_warp_sampler_oneloop(const weight_t *weights, int size, T *state)
{
    int lid = threadIdx.x % WARP_SIZE;

    weight_t w = 0;
    weight_t prev_ite_sum = 0;
    vtx_t selected_id = 0;
    for (int i = lid; i < size + WARP_SIZE - size % WARP_SIZE; i += WARP_SIZE)
    {
        weight_t weight_sum = (i < size) ? __ldg(&weights[i]) : 0;
        w = weight_sum;

        for (int mask = 1; mask < 32; mask <<= 1)
        {
            weight_t val = __shfl_up_sync(FULL_WARP_MASK, weight_sum, mask, WARP_SIZE);
            if (lid >= mask)
                weight_sum += val;
        }
        weight_sum += prev_ite_sum;
        if (i < size && myrand_uniform(state) < w / weight_sum)
            selected_id = i;
        prev_ite_sum = __shfl_sync(FULL_WARP_MASK, weight_sum, 31, WARP_SIZE);
    }
    selected_id = warpReduceMax(selected_id);

    return selected_id;
}
/**
 * N: the number of elements, p: the number of threads, x: global memory access latency
 * Time cost: (N/p) * (x + log p + log p)
 */
template <typename T>
__device__ inline vtx_t test_block_sampler_oneloop(const weight_t *weights, int size, T *state)
{
    int tid = threadIdx.x;

    __shared__ weight_t prefix_sum;
    prefix_sum = 0;

    vtx_t selected_id = 0;
    vtx_t local_selected_id = 0;

    for (int i = tid; i < size + BLOCK_SIZE - size % BLOCK_SIZE; i += BLOCK_SIZE)
    {
        weight_t w = (i < size) ? weights[i] : 0;

        // Load data into shared memory.
        weight_t weight_sum = (tid == 0) ? prefix_sum + w : w;

        // Conduct inclusive prefix sum.
        typedef cub::BlockScan<weight_t, BLOCK_SIZE> BlockScan;
        __shared__ typename BlockScan::TempStorage temp_storage_sum;
        BlockScan(temp_storage_sum).InclusiveSum(weight_sum, weight_sum);

        // Sample one index.
        if (w > 0 && i < size)
        {
            if (myrand_uniform(state) < w / weight_sum)
                local_selected_id = i;
        }

        // __syncthreads();

        // __syncthreads();
        if (tid == BLOCK_SIZE - 1)
        {
            prefix_sum = weight_sum;
        }
        __syncthreads();
    }
    // Max operation.
    typedef cub::BlockReduce<vtx_t, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage_reduce;
    selected_id = BlockReduce(temp_storage_reduce).Reduce(local_selected_id, cub::Max());

    return selected_id;
}

template <typename T>
__device__ inline vtx_t test_block_sampler_oneloop_old(const weight_t *weights, int size, T *state)
{
    int tid = threadIdx.x;

    __shared__ weight_t prefix_sum;
    prefix_sum = 0;

    vtx_t selected_id = 0;

    for (int i = tid; i < size + BLOCK_SIZE - size % BLOCK_SIZE; i += BLOCK_SIZE)
    {
        weight_t w = (i < size) ? weights[i] : 0;
        vtx_t local_selected_id = 0;

        // Load data into shared memory.
        weight_t weight_sum = (tid == 0) ? prefix_sum + w : w;

        // Conduct inclusive prefix sum.
        typedef cub::BlockScan<weight_t, BLOCK_SIZE> BlockScan;
        __shared__ typename BlockScan::TempStorage temp_storage_sum;
        BlockScan(temp_storage_sum).InclusiveSum(weight_sum, weight_sum);

        // Sample one index.
        if (w > 0 && i < size)
        {
            if (myrand_uniform(state) < w / weight_sum)
                local_selected_id = i;
        }

        // Max operation.
        typedef cub::BlockReduce<vtx_t, BLOCK_SIZE> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage_reduce;

        // __syncthreads();
        selected_id = max(selected_id, BlockReduce(temp_storage_reduce).Reduce(local_selected_id, cub::Max()));
        // __syncthreads();
        if (tid == BLOCK_SIZE - 1)
        {
            prefix_sum = weight_sum;
        }
        __syncthreads();
    }

    return selected_id;
}

__device__ inline void construct_table_warp(weight_t *weights, int size, vtx_t *alias, weight_t *prob, vtx_t *large, vtx_t *small)
{
    int tid = threadIdx.x;
    int lid = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    __shared__ int large_num[WARP_PER_BLK];
    __shared__ int small_num[WARP_PER_BLK];
    large_num[wid] = 0;
    small_num[wid] = 0;

    weight_t w_sum = 0;
    for (int i = lid; i < size; i += WARP_SIZE)
    {
        w_sum += weights[i];
    }

    for (int offset = 16; offset > 0; offset /= 2)
    {
        w_sum += __shfl_down_sync(FULL_WARP_MASK, w_sum, offset);
    }

    w_sum = __shfl_sync(FULL_WARP_MASK, w_sum, 31, WARP_SIZE);

    for (int i = lid; i < size; i += WARP_SIZE)
    {
        prob[i] = size * weights[i] / w_sum;
        if (prob[i] > 1)
        {
            int old = atomicAdd(&large_num[wid], 1);
            large[old] = i;
        }
        else
        {
            int old = atomicAdd(&small_num[wid], 1);
            small[old] = i;
        }
    }
    __syncwarp(FULL_WARP_MASK);

    while (large_num[wid] > 0 && small_num[wid] > 0)
    {
        __syncwarp(FULL_WARP_MASK);
        bool valid = false;
        int small_old = atomicSub(&small_num[wid], 1);
        int large_old = atomicSub(&large_num[wid], 1);
        if (small_old <= 0)
        {
            atomicAdd(&small_num[wid], 1);
        }
        else
        {
            if (large_old <= 0)
            {
                atomicAdd(&large_num[wid], 1);
            }
            else
            {
                valid = true;
            }
        }
        __syncwarp(FULL_WARP_MASK);
        if (valid)
        {
            int small_idx = small[small_old - 1];
            int large_idx = large[large_old - 1];
            weight_t old_prob = atomicAdd(&prob[large_idx], prob[small_idx] - 1.0);
            if (old_prob + prob[small_idx] - 1.0 < 0)
            {
                // roll back
                atomicAdd(&prob[large_idx], 1.0 - prob[small_idx]);
                small_old = atomicAdd(&small_num[wid], 1);
                small[small_old] = small_idx;
            }
            else
            {
                alias[small_idx] = large_idx;
            }
            if (prob[large_idx] > 1.0)
            {
                large_old = atomicAdd(&large_num[wid], 1);
                large[large_old] = large_idx;
            }
            else if (prob[large_idx] < 1.0)
            {
                small_old = atomicAdd(&small_num[wid], 1);
                small[small_old] = small_idx;
            }
        }
        __syncwarp(FULL_WARP_MASK);
    }
    while (small_num[wid] > 0)
    {
        int small_old = atomicSub(&small_num[wid], 1);
        if (small_old <= 0)
        {
            atomicAdd(&small_num[wid], 1);
        }
        else
        {
            prob[small[small_old - 1]] = 1.0;
        }
        __syncwarp(FULL_WARP_MASK);
    }
    while (large_num[wid] > 0)
    {
        int large_old = atomicSub(&large_num[wid], 1);
        if (large_old <= 0)
        {
            atomicAdd(&large_num[wid], 1);
        }
        else
        {
            prob[large[large_old - 1]] = 1.0;
        }
        __syncwarp(FULL_WARP_MASK);
    }
    __syncwarp(FULL_WARP_MASK);
}

template <typename T>
__device__ inline vtx_t naive_alias_warp_sampler(weight_t *weights, int size, T *state, vtx_t *alias, weight_t *prob, vtx_t *large, vtx_t *small)
{
    // construct alias table

    int tid = threadIdx.x;
    int lid = tid % WARP_SIZE;
    construct_table_warp(weights, size, alias, prob, large, small);
    // selection
    if (lid == 0)
    {
        vtx_t selected_idx = myrand(state) % size;
        float r = myrand_uniform(state);
        if (r < prob[selected_idx])
        {
            return selected_idx;
        }
        else
        {
            return alias[selected_idx];
        }
    }
    return -1;
}

__device__ inline void construct_table_block(weight_t *weights, int size, vtx_t *alias, weight_t *prob, vtx_t *large, vtx_t *small)
{
    int tid = threadIdx.x;
    __shared__ int large_num;
    __shared__ int small_num;
    large_num = 0;
    small_num = 0;
    __syncthreads();
    typedef cub::BlockReduce<weight_t, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage sum_storage;

    weight_t local_sum = 0;
    for (int i = tid; i < size; i += BLOCK_SIZE)
    {
        local_sum += weights[i];
    }
    __shared__ weight_t w_sum;
    weight_t tmp_sum = BlockReduce(sum_storage).Sum(local_sum);
    __syncthreads();
    if (tid == 0)
        w_sum = tmp_sum;

    __syncthreads();

    for (int i = tid; i < size; i += BLOCK_SIZE)
    {
        prob[i] = size * weights[i] / w_sum;
        if (prob[i] > 1)
        {
            int old = atomicAdd(&large_num, 1);
            large[old] = i;
        }
        else
        {
            int old = atomicAdd(&small_num, 1);
            small[old] = i;
        }
    }
    __syncthreads();

    while (large_num > 0 && small_num > 0)
    {
        __syncthreads();

        bool valid = false;
        int small_old = atomicSub(&small_num, 1);
        int large_old = atomicSub(&large_num, 1);
        if (small_old <= 0)
        {
            atomicAdd(&small_num, 1);
        }
        else
        {

            if (large_old <= 0)
            {
                atomicAdd(&large_num, 1);
            }
            else
            {
                valid = true;
            }
        }
        __syncthreads();
        if (valid)
        {
            int small_idx = small[small_old - 1];
            int large_idx = large[large_old - 1];
            weight_t old_prob = atomicAdd(&prob[large_idx], prob[small_idx] - 1.0);
            if (old_prob + prob[small_idx] - 1.0 < 0)
            {
                atomicAdd(&prob[large_idx], 1.0 - prob[small_idx]);
                small_old = atomicAdd(&small_num, 1);
                small[small_old] = small_idx;
            }
            else
            {
                alias[small_idx] = large_idx;
            }
            if (prob[large_idx] > 1.0)
            {
                large_old = atomicAdd(&large_num, 1);
                large[large_old] = large_idx;
            }
            else if (prob[large_idx] < 1.0)
            {
                small_old = atomicAdd(&small_num, 1);
                small[small_old] = small_idx;
            }
        }

        __syncthreads();
    }
    __syncthreads();

    while (small_num > 0)
    {
        int small_old = atomicSub(&small_num, 1);
        if (small_old <= 0)
        {
            atomicAdd(&small_num, 1);
        }
        else
        {
            prob[small[small_old - 1]] = 1.0;
        }
        __syncthreads();
    }
    while (large_num > 0)
    {
        int large_old = atomicSub(&large_num, 1);
        if (large_old <= 0)
        {
            atomicAdd(&large_num, 1);
        }
        else
        {
            prob[large[large_old - 1]] = 1.0;
        }
        __syncthreads();
    }
    __syncthreads();
}

template <typename T>
__device__ inline vtx_t naive_alias_block_sampler(weight_t *weights, int size, T *state, vtx_t *alias, weight_t *prob, vtx_t *large, vtx_t *small)
{
    // construct alias table

    int tid = threadIdx.x;
    construct_table_block(weights, size, alias, prob, large, small);
    // selection
    if (tid == 0)
    {
        vtx_t selected_idx = myrand(state) % size;
        float r = myrand_uniform(state);
        if (r < prob[selected_idx])
        {
            return selected_idx;
        }
        else
        {
            return alias[selected_idx];
        }
    }
    return -1;
}

template <typename T>
__device__ inline vtx_t its_warp_sampler(weight_t *weights, int size, T *state, weight_t *prob)
{
    int tid = threadIdx.x;
    int lid = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;

    int n_item = size / WARP_SIZE;
    int remainder = size % WARP_SIZE;
    vtx_t local_idx = 0;
    weight_t local_sum = 0;
    if (lid < remainder)
    {
        n_item++;
        local_idx = n_item * lid;
    }
    else
    {
        local_idx = n_item * lid + remainder;
    }
    for (int i = 0; i < n_item; i++)
    {
        vtx_t idx = local_idx + i;
        local_sum += weights[idx];
        prob[idx] = local_sum;
    }

    for (int mask = 1; mask < WARP_SIZE; mask <<= 1)
    {
        weight_t w = __shfl_up_sync(FULL_WARP_MASK, local_sum, mask, WARP_SIZE);
        if (lid >= mask)
            local_sum += w;
    }

    __shared__ weight_t all_sum[WARP_PER_BLK];
    // __shared__ weight_t r[WARP_PER_BLK];
    if (lid == WARP_SIZE - 1)
    {
        all_sum[wid] = local_sum + prob[size - 1];
    }
    __syncwarp(FULL_WARP_MASK);

    for (int i = 0; i < n_item; i++)
    {
        uint idx = local_idx + i;
        prob[idx] = (prob[idx] + local_sum) / all_sum[wid];
    }
    __syncwarp(FULL_WARP_MASK);
    // binary search
    vtx_t selected;
    if (lid == 0)
    {
        float r;
        r = myrand_uniform(state);
        selected = binary_search(prob, size, r);
    }

    __syncwarp(FULL_WARP_MASK);
    return selected;
}

template <typename T>
__device__ inline vtx_t its_warp_sampler_direct(weight_t *weights, int size, T *state, weight_t *prob)
{
    int tid = threadIdx.x;
    int lid = tid % WARP_SIZE;

    weight_t prefix_sum = 0;

    for (int i = lid; i < size + WARP_SIZE - size % WARP_SIZE; i += WARP_SIZE)
    {
        weight_t w = (i < size) ? __ldg(&weights[i]) : 0;
        weight_t local_sum = (tid == 0) ? prefix_sum + w : w;
        for (int mask = 1; mask < WARP_SIZE; mask <<= 1)
        {
            weight_t val = __shfl_up_sync(FULL_WARP_MASK, local_sum, mask, WARP_SIZE);
            if (lid >= mask)
                local_sum += val;
        }
        if (i < size)
            prob[i] = local_sum;
        prefix_sum = __shfl_sync(FULL_WARP_MASK, local_sum, 31, WARP_SIZE);
    }
    __syncwarp(FULL_WARP_MASK);
    weight_t all_sum = prob[size - 1];
    __syncwarp(FULL_WARP_MASK);

    for (int i = tid; i < size; i += BLOCK_SIZE)
    {
        prob[i] = prob[i] / all_sum;
    }
    __syncwarp(FULL_WARP_MASK);
    // binary search
    vtx_t selected;
    if (lid == 0)
    {
        float r;
        r = myrand_uniform(state);
        selected = binary_search(prob, size, r);
    }

    __syncwarp(FULL_WARP_MASK);
    return selected;
}

template <typename T>
__device__ inline vtx_t its_block_sampler(weight_t *weights, int size, T *state, weight_t *prob)
{
    int tid = threadIdx.x;

    int n_item = size / BLOCK_SIZE;
    int remainder = size % BLOCK_SIZE;
    vtx_t local_idx = 0;
    weight_t local_sum = 0;
    if (tid < remainder)
    {
        n_item++;
        local_idx = n_item * tid;
    }
    else
    {
        local_idx = n_item * tid + remainder;
    }
    for (int i = 0; i < n_item; i++)
    {
        uint idx = local_idx + i;
        local_sum += weights[idx];
        prob[idx] = local_sum;
    }

    typedef cub::BlockScan<weight_t, BLOCK_SIZE> BlockScan;
    __shared__ typename BlockScan::TempStorage prefix_storage;
    BlockScan(prefix_storage).ExclusiveSum(local_sum, local_sum);

    __shared__ weight_t all_sum;
    if (tid == BLOCK_SIZE - 1)
    {
        all_sum = local_sum + prob[size - 1];
    }
    __syncthreads();
    for (int i = 0; i < n_item; i++)
    {
        vtx_t idx = local_idx + i;
        prob[idx] = (prob[idx] + local_sum) / all_sum;
    }
    __syncthreads();
    // binary search
    vtx_t selected = 0;
    if (tid == 0)
    {
        float r;
        r = myrand_uniform(state);
        selected = binary_search(prob, size, r);
    }

    __syncthreads();

    return selected;
}

template <typename T>
__device__ inline vtx_t its_block_sampler_direct(weight_t *weights, int size, T *state, weight_t *prob)
{
    int tid = threadIdx.x;
    __shared__ weight_t prefix_sum;
    prefix_sum = 0;
    for (int i = tid; i < size + BLOCK_SIZE - size % BLOCK_SIZE; i += BLOCK_SIZE)
    {
        weight_t w = (i < size) ? weights[i] : 0;
        weight_t local_sum = (tid == 0) ? prefix_sum + w : w;
        typedef cub::BlockScan<weight_t, BLOCK_SIZE> BlockScan;
        __shared__ typename BlockScan::TempStorage prefix_storage;
        BlockScan(prefix_storage).ExclusiveSum(local_sum, local_sum);
        if (w > 0 && i < size)
        {
            prob[i] = local_sum + w;
        }
        if (tid == BLOCK_SIZE - 1)
        {
            prefix_sum = local_sum;
        }
    }
    __syncthreads();
    weight_t all_sum = prob[size - 1];
    __syncthreads();
    for (int i = tid; i < size; i += BLOCK_SIZE)
    {
        prob[i] = prob[i] / all_sum;
    }

    __syncthreads();
    // binary search
    vtx_t selected = 0;
    if (tid == 0)
    {
        float r;
        r = myrand_uniform(state);
        selected = binary_search(prob, size, r);
    }

    __syncthreads();

    return selected;
}

template <typename T>
__device__ inline vtx_t rjs_warp_sampler(const weight_t *weights, int size, T *state, u64 *clk)
{
    u64 start_clk = clock64();

    int lid = threadIdx.x % WARP_SIZE;

    weight_t local_max_weight = 0;
    for (int i = lid; i < size; i += WARP_SIZE)
    {
        local_max_weight = max(local_max_weight, weights[i]);
    }
    __syncwarp();

    weight_t max_weight = warpReduceMax(local_max_weight);
    __syncwarp();
    vtx_t selected = 0;
    if (lid == 0)
    {
        // max_weight = max_weight;
        vtx_t x;
        weight_t y, prob;
        do
        {
            x = myrand(state) % size;
            y = myrand_uniform(state) * max_weight;
            prob = weights[x];
        } while (y > prob);
        selected = x;
    }
    __syncwarp();

    if (lid == 0)
        *clk += clock64() - start_clk;
    return selected;
}

template <typename T>
__device__ inline vtx_t rjs_warp_sampler(const weight_t *weights, int size, T *state)
{

    int lid = threadIdx.x % WARP_SIZE;

    weight_t local_max_weight = 0;
    for (int i = lid; i < size; i += WARP_SIZE)
    {
        local_max_weight = max(local_max_weight, weights[i]);
    }
    __syncwarp();

    weight_t max_weight = warpReduceMax(local_max_weight);
    __syncwarp();
    vtx_t selected = 0;
    if (lid == 0)
    {
        // max_weight = max_weight;
        vtx_t x;
        weight_t y, prob;
        do
        {
            x = myrand(state) % size;
            y = myrand_uniform(state) * max_weight;
            prob = weights[x];
        } while (y > prob);
        selected = x;
    }
    __syncwarp();

    return selected;
}

template <typename T>
__device__ inline vtx_t rjs_block_sampler(weight_t *weights, int size, T *state, u64 *clk)
{
    u64 start_clk = clock64();

    int tid = threadIdx.x;
    // __shared__ weight_t max_weight;
    // max_weight = 0;

    weight_t local_max_weight = 0;
    for (int i = tid; i < size; i += BLOCK_SIZE)
    {
        local_max_weight = max(local_max_weight, weights[i]);
    }
    __syncthreads();
    typedef cub::BlockReduce<weight_t, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage_reduce;

    weight_t max_w = BlockReduce(temp_storage_reduce).Reduce(local_max_weight, cub::Max());
    __syncthreads();
    vtx_t selected = 0;

    // int num_rand = 0;
    if (tid == 0)
    {
        // max_weight = max_w;
        vtx_t x;
        weight_t y, prob;
        do
        {
            // num_rand++;
            // atomicAdd(cnt, 1);
            x = myrand(state) % size;
            y = myrand_uniform(state) * max_w;
            prob = weights[x];
        } while (y > prob);
        selected = x;
        // if (task_id == 0)
        //     printf("%d:%f,%d\n", blockIdx.x, max_weight, num_rand);
    }

    __syncthreads();

    if (tid == 0)
        *clk += clock64() - start_clk;

    return selected;
}

template <typename T>
__device__ inline vtx_t rjs_block_sampler(weight_t *weights, int size, T *state)
{
    int tid = threadIdx.x;
    // __shared__ weight_t max_weight;
    // max_weight = 0;

    weight_t local_max_weight = 0;
    for (int i = tid; i < size; i += BLOCK_SIZE)
    {
        local_max_weight = max(local_max_weight, weights[i]);
    }
    __syncthreads();
    typedef cub::BlockReduce<weight_t, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage_reduce;

    weight_t max_w = BlockReduce(temp_storage_reduce).Reduce(local_max_weight, cub::Max());
    __syncthreads();
    vtx_t selected = 0;

    // int num_rand = 0;
    if (tid == 0)
    {
        // max_weight = max_w;
        vtx_t x;
        weight_t y, prob;
        do
        {
            // num_rand++;
            // atomicAdd(cnt, 1);
            x = myrand(state) % size;
            y = myrand_uniform(state) * max_w;
            prob = weights[x];
        } while (y > prob);
        selected = x;
        // if (task_id == 0)
        //     printf("%d:%f,%d\n", blockIdx.x, max_weight, num_rand);
    }

    __syncthreads();

    return selected;
}