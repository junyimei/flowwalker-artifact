#pragma once
#include <algorithm>
#include <random>
// #include <cub/cub.cuh>
#include <vector>

// #include "app.cuh"
// #include "myrand.cuh"
// #include "sampler.cuh"
#include "gpu_graph.cuh"
#include "util.cuh"
// using namespace std;

class Task
{
public:
    int walker_id;          // The id of the walker
    vtx_t current_vertex;   // The current vertex
    vtx_t prev_vertex;      // The previous vertex
    edge_t neighbor_offset; // The offset of the neighbor set
    vtx_t degree;           // The degree of the current vertex
    // int res_offset;         // The offset of the walk in the result pool
    int length; // The current length of the walk, -1 means the task unit is empty.

public:
    __device__ Task() {}
    __device__ Task(gpu_graph *graph, vtx_t *start_points, int _walker_id)
    {
        walker_id = _walker_id;
        prev_vertex = -1;
        current_vertex = start_points[_walker_id];
        // printf("current_vertex=%u\n",current_vertex);
        neighbor_offset = graph->xadj[current_vertex];
        degree = graph->getDegree(current_vertex);
        // res_offset = _walker_id * max_depth + 1;
        length = 1;
    }
    __device__ Task(gpu_graph *graph, vtx_t _current_vertex, int _walker_id)
    {
        walker_id = _walker_id;
        prev_vertex = -1;
        current_vertex = _current_vertex;
        // printf("current_vertex=%u\n",current_vertex);
        neighbor_offset = graph->xadj[current_vertex];
        degree = graph->getDegree(current_vertex);
        // res_offset = _walker_id * max_depth + 1;
        length = 1;
    }
    __device__ void init(gpu_graph *graph, vtx_t *start_points, int _walker_id)
    {
        walker_id = _walker_id;
        prev_vertex = -1;
        current_vertex = start_points[_walker_id];
        // printf("current_vertex=%u\n",current_vertex);
        neighbor_offset = graph->xadj[current_vertex];
        degree = graph->getDegree(current_vertex);
        // res_offset = _walker_id * max_depth + 1;
        length = 1;
    }
    __device__ void init(gpu_graph *graph, vtx_t _current_vertex, int _walker_id)
    {
        walker_id = _walker_id;
        prev_vertex = -1;
        current_vertex = _current_vertex;
        // printf("current_vertex=%u\n",current_vertex);
        neighbor_offset = graph->xadj[current_vertex];
        degree = graph->getDegree(current_vertex);
        // res_offset = _walker_id * max_depth + 1;
        length = 1;
    }

    __device__ void update(gpu_graph *graph, vtx_t selected_id)
    {
        prev_vertex = current_vertex;
        current_vertex = selected_id;
        neighbor_offset = graph->xadj[selected_id];
        degree = graph->getDegree(selected_id);
        // res_offset++;
        length++;
    }
};

typedef struct tagTaskAssignments
{
    int begin;
    int end;
} TaskAssignments;
