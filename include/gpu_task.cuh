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

#include "gpu_graph.cuh"
#include "util.cuh"

class Task {
 public:
  int walker_id;           // The id of the walker
  vtx_t current_vertex;    // The current vertex
  vtx_t prev_vertex;       // The previous vertex
  edge_t neighbor_offset;  // The offset of the neighbor set
  vtx_t degree;            // The degree of the current vertex
  int length;  // The current length of the walk, -1 means the task unit is
               // empty.

 public:
  __device__ Task() {}
  __device__ Task(gpu_graph* graph, vtx_t* start_points, int _walker_id) {
    walker_id = _walker_id;
    prev_vertex = -1;
    current_vertex = start_points[_walker_id];
    neighbor_offset = graph->xadj[current_vertex];
    degree = graph->getDegree(current_vertex);
    length = 1;
  }
  __device__ Task(gpu_graph* graph, vtx_t _current_vertex, int _walker_id) {
    walker_id = _walker_id;
    prev_vertex = -1;
    current_vertex = _current_vertex;
    neighbor_offset = graph->xadj[current_vertex];
    degree = graph->getDegree(current_vertex);
    length = 1;
  }
  __device__ void init(gpu_graph* graph, vtx_t* start_points, int _walker_id) {
    walker_id = _walker_id;
    prev_vertex = -1;
    current_vertex = start_points[_walker_id];
    neighbor_offset = graph->xadj[current_vertex];
    degree = graph->getDegree(current_vertex);
    length = 1;
  }
  __device__ void init(gpu_graph* graph, vtx_t _current_vertex,
                       int _walker_id) {
    walker_id = _walker_id;
    prev_vertex = -1;
    current_vertex = _current_vertex;
    neighbor_offset = graph->xadj[current_vertex];
    degree = graph->getDegree(current_vertex);
    length = 1;
  }

  __device__ void update(gpu_graph* graph, vtx_t selected_id) {
    prev_vertex = current_vertex;
    current_vertex = selected_id;
    neighbor_offset = graph->xadj[selected_id];
    degree = graph->getDegree(selected_id);
    // res_offset++;
    length++;
  }
};

typedef struct tagTaskAssignments {
  int begin;
  int end;
} TaskAssignments;
