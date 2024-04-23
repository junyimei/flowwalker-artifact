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
#include <assert.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <gflags/gflags.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "util.cuh"

DECLARE_bool(metapath);
DECLARE_bool(lognorm);

inline off_t fsize(const char* filename) {
  struct stat st;
  if (stat(filename, &st) == 0) return st.st_size;
  return -1;
}

class graph {
 public:
  edge_t* xadj;
  vtx_t* adjncy;
  weight_t* weight;
  int* edge_label;
  vtx_t vert_count;
  edge_t edge_count;

 public:
  graph() {}
  ~graph() {}
  template <typename T>
  bool load_edge_feature(const char* filename, T* arr) {
    FILE* file = NULL;
    edge_t ret;

    file = fopen(filename, "rb");
    if (file != NULL) {
      // T *tmp_arr = NULL;
      if (posix_memalign(reinterpret_cast<void**>(&arr), getpagesize(),
                         sizeof(T) * edge_count))
        perror("posix_memalign");

      ret = fread(arr, sizeof(T), edge_count, file);

      assert(ret == edge_count);
      fclose(file);
    } else {
      std::cout << "File cannot open: " << filename << std::endl;
      return 0;
    }
    return 1;
  }

  void generate_weight_lognorm(float mean, float stddev) {
    weight_t* tmp_weight;
    cudaMalloc(reinterpret_cast<void**>(&tmp_weight),
               sizeof(weight_t) * edge_count);
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateLogNormal(gen, tmp_weight, edge_count, mean, stddev);
    weight = new weight_t[edge_count];
    cudaMemcpy(weight, tmp_weight, sizeof(weight_t) * edge_count,
               cudaMemcpyDefault);
    cudaFree(tmp_weight);
  }

  bool load_graph(const char* xadj_file, const char* adjncy_file,
                  const char* weight_file, const char* label_file) {
    double tm = wtime();
    FILE* file = NULL;
    edge_t ret;

    vert_count = fsize(xadj_file) / sizeof(vtx_t) - 1 - 2;
    edge_count = fsize(adjncy_file) / sizeof(edge_t);

    file = fopen(xadj_file, "rb");
    if (file != NULL) {
      edge_t* tmp_xadj = NULL;

      if (posix_memalign(reinterpret_cast<void**>(&tmp_xadj), getpagesize(),
                         sizeof(edge_t) * (vert_count + 1)))
        perror("posix_memalign");

      ret = fread(&vert_count, sizeof(vtx_t), 1, file);
      ret = fread(&edge_count, sizeof(edge_t), 1, file);
      ret = fread(tmp_xadj, sizeof(edge_t), vert_count + 1, file);
      assert(ret == vert_count + 1);
      fclose(file);

      std::cout << "Vertex count:" << vert_count
                << " Expected edge count: " << edge_count << "\n";

      assert(tmp_xadj[vert_count] > 0);

      xadj = reinterpret_cast<edge_t*>(tmp_xadj);
    } else {
      std::cout << "xadj file cannot open\n";
      return 0;
    }

    file = fopen(adjncy_file, "rb");
    if (file != NULL) {
      vtx_t* tmp_adjncy = NULL;
      if (posix_memalign(reinterpret_cast<void**>(&tmp_adjncy), getpagesize(),
                         sizeof(vtx_t) * edge_count))
        perror("posix_memalign");

      ret = fread(tmp_adjncy, sizeof(vtx_t), edge_count, file);
      assert(ret == edge_count);
      assert(ret == xadj[vert_count]);
      fclose(file);

      adjncy = reinterpret_cast<vtx_t*>(tmp_adjncy);
    } else {
      std::cout << "CSR file cannot open\n";
      return 0;
    }

    if (FLAGS_metapath) {
      file = fopen(label_file, "rb");
      if (file != NULL) {
        int* tmp_label = NULL;
        if (posix_memalign(reinterpret_cast<void**>(&tmp_label), getpagesize(),
                           sizeof(int) * edge_count))
          perror("posix_memalign");

        ret = fread(tmp_label, sizeof(int), edge_count, file);

        assert(ret == edge_count);
        fclose(file);

        edge_label = reinterpret_cast<int*>(tmp_label);
      } else {
        std::cout << "CSR file cannot open\n";
        return 0;
      }
    } else {
      if (FLAGS_lognorm) {
        generate_weight_lognorm(0.0f, 3.0f);
      } else {
        file = fopen(weight_file, "rb");
        if (file != NULL) {
          weight_t* tmp_weight = NULL;
          if (posix_memalign(reinterpret_cast<void**>(&tmp_weight),
                             getpagesize(), sizeof(weight_t) * edge_count))
            perror("posix_memalign");

          ret = fread(tmp_weight, sizeof(weight_t), edge_count, file);

          assert(ret == edge_count);
          fclose(file);

          weight = reinterpret_cast<weight_t*>(tmp_weight);
        } else {
          std::cout << "File cannot open: " << weight_file << std::endl;
          return 0;
        }
      }
    }

    std::cout << "Graph load (success): " << vert_count << " verts, "
              << edge_count << " edges " << wtime() - tm << " second(s)\n";
    return 1;
  }

  graph(const char* xadj_file, const char* adjncy_file, const char* weight_file,
        const char* label_file) {
    load_graph(xadj_file, adjncy_file, weight_file, label_file);
  }

  void print_degree() {
    for (size_t i = 20000; i < 20100; i++) {
      printf("%d\t", xadj[i + 1] - xadj[i]);
    }
    printf("\n");
  }
  void print_weight() {
    for (size_t i = 20000; i < 20100; i++) {
      printf("%f\t", weight[i]);
    }
    printf("\n");
  }
  void print_max_degree() {
    vtx_t max_degree = 0;
    vtx_t max_idx = 0;
    for (auto i = 0; i < vert_count; i++) {
      vtx_t degree = xadj[i + 1] - xadj[i];
      if (degree > max_degree) {
        max_degree = degree;
        max_idx = i;
      }
    }
    printf("Max degree: %d, idx: %d\n", max_degree, max_idx);
  }
  vtx_t get_maxdegree_offset() {
    vtx_t max_degree = 0;
    vtx_t max_idx = 0;
    for (auto i = 0; i < vert_count; i++) {
      vtx_t degree = xadj[i + 1] - xadj[i];
      if (degree > max_degree) {
        max_degree = degree;
        max_idx = i;
      }
    }
    return max_idx;
  }
};
