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

#include <cooperative_groups.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cooperative_groups/reduce.h>
#include <gflags/gflags.h>
#include <cub/cub.cuh>

#include "graph.cuh"
#include "util.cuh"
#include "gpu_task.cuh"
#include "myrand.cuh"

// #define MICRO_BENCH

/*
Walker apps
*/
class WalkerMeta
{
public:
    int max_depth;
    gpu_graph *graph;
    // vtx_t *result_pool;

public:
    WalkerMeta() {}
    WalkerMeta(gpu_graph *_graph, int _max_depth)
    {
        this->max_depth = _max_depth;
        this->graph = _graph;
    }
    __device__ weight_t get_weight(Task *task, int i);
    __device__ bool is_stop(int len, myrandStateArr *state);
    __device__ bool is_stop(int len, curandState *state);
};

class Deepwalk : public WalkerMeta
{
public:
    Deepwalk() {}
    Deepwalk(gpu_graph *_graph, int _max_depth) : WalkerMeta(_graph, _max_depth)
    {
    }

    __device__ weight_t get_weight(Task *task, int i)
    {
        return graph->adjwgt[task->neighbor_offset + i];
    }
    template <typename state_t>
    __device__ bool is_stop(int len, state_t *state)
    {
        return len >= max_depth;
    }
};

class PPR : public WalkerMeta
{
public:
    float tp;

public:
    PPR() {}
    PPR(gpu_graph *_graph, int _max_depth, float _tp) : WalkerMeta(_graph, _max_depth)
    {
        this->tp = _tp;
    }

    __device__ weight_t get_weight(Task *task, int i)
    {
        return graph->adjwgt[task->neighbor_offset + i];
    }
    template <typename state_t>
    __device__ bool is_stop(int len, state_t *state)
    {
        float r = myrand_uniform(state);
        if (r <= tp)
            return true;
        else
            return len >= max_depth;
    }
};

class Node2vec : public WalkerMeta
{
public:
    float p, q;

public:
    Node2vec() {}
    Node2vec(gpu_graph *_graph, int _max_depth, float _p, float _q) : WalkerMeta(_graph, _max_depth)
    {
        this->p = _p;
        this->q = _q;
    }

    __device__ weight_t get_weight(Task *task, int i)
    {
        weight_t w = graph->adjwgt[task->neighbor_offset + i];
        if (task->prev_vertex == -1)
            return w;
        else
        {
            vtx_t post = graph->adjncy[task->neighbor_offset + i];
            if (post == task->prev_vertex)
            {
                return w / p;
            }
            else if (!graph->check_connect(task->prev_vertex, post))
            {
                return w / q;
            }
            return w;
        }
    }
    template <typename state_t>
    __device__ bool is_stop(int len, state_t *state)
    {
        return len >= max_depth;
    }
};

class Metapath : public WalkerMeta
{
public:
    int *schema;
    int schema_len;

public:
    Metapath() {}
    Metapath(gpu_graph *_graph, int _max_depth, int *_schema, int _schema_len) : WalkerMeta(_graph, _max_depth)
    {
        this->schema = _schema;
        this->schema_len = _schema_len;
    }

    __device__ weight_t get_weight(Task *task, int i)
    {
        edge_t offset = task->neighbor_offset + i;
        return graph->edge_label[offset] == schema[(task->length - 1) % schema_len] ? 1.0 : 0;
    }
    template <typename state_t>
    __device__ bool is_stop(int len, state_t *state)
    {
        return len >= max_depth;
    }
};
