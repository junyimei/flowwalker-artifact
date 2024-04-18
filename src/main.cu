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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <gflags/gflags.h>
#include <omp.h>
#include <cub/cub.cuh>
#include <time.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <map>

#include "graph.cuh"
#include "gpu_graph.cuh"
#include "app.cuh"
#include "walker.cuh"
#include "util.cuh"

using namespace std;

DEFINE_string(input, "../data/wiki-Vote", "input dataset");
DEFINE_string(b_file, "../res/bucket.csv", "degree bucket file");

DEFINE_int32(n, 4000, "sample size");
DEFINE_bool(seq, false, "start from 0 to n-1");
DEFINE_bool(all, false, "start from all nodes");
DEFINE_int32(d, 80, "depth");

DEFINE_bool(umgraph, false, "enable unified memory to store graph.");

DEFINE_bool(deepwalk, false, "deepwalk");
DEFINE_bool(node2vec, false, "node2vec");
DEFINE_bool(ppr, false, "ppr");
DEFINE_bool(metapath, false, "metapath");

DEFINE_bool(dprs, false, "enable direct parallel RS");

DEFINE_double(p, 2.0, "hyper-parameter p for node2vec");
DEFINE_double(q, 0.5, "hyper-parameter q for node2vec");
DEFINE_double(tp, 0.2, "terminate probabiility");
DEFINE_int32(schemalen, 5, "number of labels");
DEFINE_string(schema, "0,1,2,3,4", "metapath schema");

DEFINE_bool(printresult, false, "printresult");
DEFINE_bool(printworkload, false, "printworkload");
DEFINE_bool(save_degree, false, "save degree distribution");

DEFINE_bool(autobatch, false, "use adaptive batch size");
DEFINE_bool(batch, false, "use batch mode");
DEFINE_bool(syn, false, "enable synchronized batch");
DEFINE_uint32(batchsize, 10000000, "batch size");
DEFINE_int32(headroom, 512, "GPU memory headroom for other data structures while using autobatch(MB)");

DEFINE_int32(walkmode, -1, "walkmode:0-static warp,1-static warp block,2-static thread warp block,3-queue,4-static thread warp,5-dynamic thread warp block,6-dynamic warp block,7-dynamic warp,8-dynamic block");

DEFINE_bool(lognorm, false, "lognorm edge weight");

void calculate_workload(int num_walkers, int max_depth, vtx_t *result_pool_ptr, graph *ginst, double total_s, bool is_host = false)
{
    vtx_t *result = result_pool_ptr;
    if (!is_host)
    {
        result = (vtx_t *)malloc((u64)sizeof(vtx_t) * num_walkers * max_depth);
        CUDA_RT_CALL(cudaMemcpy(result, result_pool_ptr, (u64)sizeof(vtx_t) * num_walkers * max_depth, cudaMemcpyDeviceToHost));
    }

    u64 sampled_v = 0;
    u64 sampled_e = 0;
    for (int i = 0; i < num_walkers; i++)
    {
        vtx_t vtx_pre = -1;
        for (int j = 0; j < max_depth; j++)
        {
            u64 res_offset = (u64)i * max_depth + j;
            vtx_t vtx = result[res_offset];
            if (vtx == -1)
            {
                break;
            }
            if (j > 0)
                sampled_v++;
            if (vtx_pre != -1)
                sampled_e += ginst->xadj[vtx_pre + 1] - ginst->xadj[vtx_pre];
            vtx_pre = vtx;
        }
    }
    printf("Total sampled vertex: %llu\n", sampled_v);
    printf("Total sampled edge: %llu,throughput: %f edges per s\n", sampled_e, sampled_e / total_s);
    // free(xadj);
}

void degree_bucket(int num_walkers, int max_depth, vtx_t *result_pool_ptr, graph *ginst, bool is_host = false)
{
    vtx_t *result = result_pool_ptr;
    if (!is_host)
    {
        result = (vtx_t *)malloc((u64)sizeof(vtx_t) * num_walkers * max_depth);
        CUDA_RT_CALL(cudaMemcpy(result, result_pool_ptr, (u64)sizeof(vtx_t) * num_walkers * max_depth, cudaMemcpyDeviceToHost));
    }

    std::map<edge_t, vtx_t> degree_mp;
    for (int i = 0; i < num_walkers; i++)
    {
        vtx_t vtx_pre = -1;
        for (int j = 0; j < max_depth; j++)
        {
            u64 res_offset = (u64)i * max_depth + j;
            vtx_t vtx = result[res_offset];
            if (vtx == -1)
            {
                break;
            }

            if (vtx_pre != -1)
            {
                edge_t degree = ginst->xadj[vtx_pre + 1] - ginst->xadj[vtx_pre];
                degree_mp[degree]++;
            }
            vtx_pre = vtx;
        }
    }
    ofstream degree_bucket_ofs(FLAGS_b_file);
    for (auto i = degree_mp.begin(); i != degree_mp.end(); i++)
    {
        degree_bucket_ofs << i->first << "," << i->second << endl;
    }
    degree_bucket_ofs.close();
    // free(xadj);
}

void print_res(int num_walkers, int max_depth, vtx_t *result_pool, bool is_host = false)
{
    // printf("%p\n", result_pool);
    vtx_t *result = result_pool;
    if (!is_host)
    {
        result = (vtx_t *)malloc((u64)sizeof(vtx_t) * num_walkers * max_depth);
        CUDA_RT_CALL(cudaMemcpy(result, result_pool, (u64)sizeof(vtx_t) * num_walkers * max_depth, cudaMemcpyDeviceToHost));
    }

    u64 sampled = 0;
    // for (int i = 0; i < num_walkers; i++)
    for (int i = 0; i < 100; i++)
    {
        printf("----------------------\nWalker %d:\n", i);
        int len = max_depth;
        for (int j = 0; j < max_depth; j++)
        {
            u64 res_offset = i * max_depth + j;
            if ((int)result[res_offset] == -1)
            {
                len = j;
                break;
            }
            if (j > 0)
                sampled++;
            printf("%d\t", result[res_offset]);
        }
        printf("\nTotal length=%d\n", len);
    }
    printf("Total sampled: %llu\n", sampled);
    // free(result);
}

void print_res_metapath(int num_walkers, int max_depth, vtx_t *result_pool, graph *graph, bool is_host = false)
{
    vtx_t *result = result_pool;
    if (!is_host)
    {
        result = (vtx_t *)malloc((u64)sizeof(vtx_t) * num_walkers * max_depth);
        CUDA_RT_CALL(cudaMemcpy(result, result_pool, (u64)sizeof(vtx_t) * num_walkers * max_depth, cudaMemcpyDeviceToHost));
    }

    u64 sampled = 0;
    for (u64 i = 0; i < num_walkers; i++)
    {
        printf("----------------------\nWalker %llu:\n", i);
        int len = max_depth;
        int pre_vtx = -1;

        for (int j = 0; j < max_depth; j++)
        {
            u64 res_offset = i * max_depth + j;
            vtx_t vtx = result[res_offset];
            if (vtx == -1)
            {
                len = j;
                break;
            }
            if (pre_vtx != -1)
            {
                edge_t pre_offset = graph->xadj[pre_vtx];
                vtx_t pre_degree = graph->xadj[pre_vtx + 1] - pre_offset;
                for (vtx_t k = 0; k < pre_degree; k++)
                {
                    if (graph->adjncy[pre_offset + k] == vtx)
                    {
                        printf("[%d]\t", graph->edge_label[pre_offset + k]);
                        break;
                    }
                }
            }
            if (j > 0)
                sampled++;
            pre_vtx = vtx;
            printf("%d\t", vtx);
        }
        printf("\nTotal length=%d\n", len);
    }
    printf("Total sampled: %llu\n", sampled);
}

int *get_metapath(int &schema_len)
{
    vector<int> v_schema;
    schema_len = 0;
    while (FLAGS_schema.find(",") != string::npos)
    {
        string tmp = FLAGS_schema.substr(0, FLAGS_schema.find(","));
        v_schema.push_back(stoi(tmp));
        FLAGS_schema = FLAGS_schema.substr(FLAGS_schema.find(",") + 1);
    }
    v_schema.push_back(stoi(FLAGS_schema));
    schema_len = v_schema.size();
    FLAGS_schemalen = schema_len;
    int *schema = (int *)malloc(schema_len * sizeof(int));
    for (int i = 0; i < schema_len; i++)
    {
        schema[i] = v_schema[i];
    }
    return schema;
}

void adjust_flags(uint query_num)
{

    if (FLAGS_metapath)
    {
        FLAGS_d = FLAGS_schemalen;
    }
    if (FLAGS_all)
    {
        FLAGS_seq = true;
        FLAGS_batch = true;
        FLAGS_autobatch = true;
        FLAGS_walkmode = -1;
    }

    if (FLAGS_walkmode >= 0)
    {
        FLAGS_batch = false;
        FLAGS_all = false;
    }
    if (FLAGS_autobatch)
    {
        FLAGS_batch = true;

        size_t avail = get_avail_mem();

        int bs = min((avail - FLAGS_headroom * 1024 * 1024) / (4 * (FLAGS_d + 2)), (size_t)INT_MAX);

        assert(bs > 0);
        if (FLAGS_syn == false)
            bs /= 2;
        FLAGS_batchsize = min(bs, query_num);
        printf("batch size=%u\n", FLAGS_batchsize);
    }
}
vtx_t *get_startpoints(graph *ginst, int sample_size)
{
    vtx_t *start_points;
    CUDA_RT_CALL(cudaMallocHost(&start_points, sample_size * sizeof(vtx_t)));

    if (FLAGS_ppr)
    {
        vtx_t idx = ginst->get_maxdegree_offset();
        for (int i = 0; i < sample_size; i++)
        {
            start_points[i] = idx;
        }
    }
    else
    {
        vtx_t num_node = ginst->vert_count;
        srand(0);

        for (int i = 0; i < sample_size; i++)
        {
            if (FLAGS_seq)
                start_points[i] = i;
            else
                start_points[i] = rand() % num_node;
            // start_points[i] = i;
            // printf("start p %d:%u\n",i,start_points[i]);
        }
    }
    return start_points;
}

int main(int argc, char *argv[])
{
    printf("\n---------------------------------------------------------------------------------------------------------\n");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    string adj_file = FLAGS_input + "_xadj.bin";
    string edge_file = FLAGS_input + "_edge.bin";
    string weight_file = FLAGS_input + "_weight.bin";
    string edgelabel_file = FLAGS_input + "_label.bin";
    graph *ginst;

    // if (FLAGS_metapath)
    // {
    ginst = new graph(adj_file.c_str(), edge_file.c_str(), weight_file.c_str(), edgelabel_file.c_str());
    // }
    // else
    // {
    //     ginst = new graph(adj_file.c_str(), edge_file.c_str(), weight_file.c_str(), nullptr);
    // }
    // LOG("Read graph \n");
    printf("Read Graph!\n");
    ginst->print_max_degree();
    // ginst->print_degree();
    // ginst->print_weight();
    gpu_graph *ggraph = new gpu_graph(ginst);

    int schema_len;
    int *schema;
    if (FLAGS_metapath)
    {
        schema = get_metapath(schema_len);
        printf("schema len=%d\n", schema_len);
        for (int i = 0; i < schema_len; i++)
        {
            printf("%d ", schema[i]);
        }
        printf("\n");
    }

    int sample_size = FLAGS_n;

    if (FLAGS_all)
        sample_size = ginst->vert_count;
    adjust_flags(sample_size);
    int depth = FLAGS_d + 1;

    vtx_t *start_points = get_startpoints(ginst, sample_size);

    vtx_t *result_pool = NULL;
    double sample_time = 0;
    if (FLAGS_batch)
    {
        sample_time = walk_batch(result_pool, ggraph, start_points, depth, sample_size, FLAGS_batchsize, schema, schema_len);
    }
    else
    {
        enum walk_mode mode = wb_dynamic;
        if (FLAGS_walkmode >= 0)
        {
            mode = walk_mode(FLAGS_walkmode);
        }
        sample_time = walk_test(result_pool, ggraph, start_points, depth, sample_size, mode, schema, schema_len);
    }

    bool is_host = false;
    if (FLAGS_batch)
        is_host = true;
    if (FLAGS_printresult)
    {
        if (FLAGS_metapath)
        {
            print_res_metapath(sample_size, depth, result_pool, ginst, is_host);
        }
        else
        {
            print_res(sample_size, depth, result_pool, is_host);
        }
    }
    if (FLAGS_printworkload)
    {
        calculate_workload(sample_size, depth, result_pool, ginst, sample_time, is_host);
    }
    if (FLAGS_save_degree)
    {
        degree_bucket(sample_size, depth, result_pool, ginst, is_host);
    }

    delete ginst;
    delete ggraph;

    // delete start_points;
    cudaFreeHost(start_points);

    return 0;
}