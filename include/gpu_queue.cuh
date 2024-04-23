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

#include "gpu_graph.cuh"
#include "util.cuh"

template <class T>
class queue_gpu {
  // Implementation of non-blocking queue on GPU
 public:
  int max_size;
  int size;
  uint head;
  uint tail;
  T* data = nullptr;
  int lock;

 public:
  void init(int _max_size) {
    max_size = _max_size;
    size = 0;
    head = 0;
    tail = 0;
    lock = 0;
    CUDA_RT_CALL(cudaMalloc(&data, sizeof(T) * _max_size));
  }

  void free() {
    if (data != nullptr) CUDA_RT_CALL(cudaFree(data));
  }

  __device__ void push(T item) {
    uint old = atomicInc(&tail, max_size - 1);
    data[old] = item;
    int old_size = atomicAdd(&size, 1);
  }

  __device__ bool pop(T* item) {
    int old_size = atomicSub(&size, 1);
    if (old_size <= 0) {
      atomicAdd(&size, 1);
      return false;
    }
    uint old = atomicInc(&head, max_size - 1);
    *item = data[old];

    return true;
  }
  __device__ bool pop(T* item, int* head_offset) {
    int old_size = atomicSub(&size, 1);
    if (old_size <= 0) {
      atomicAdd(&size, 1);
      return false;
    }
    uint old = atomicInc(&head, max_size - 1);
    *item = data[old];
    *head_offset = old;

    return true;
  }

  __device__ void reset() {
    head = 0;
    tail = 0;
    size = 0;
  }

  __device__ bool empty() {
    if (size <= 0)
      return true;
    else
      return false;
  }

  __device__ void atomic_push(T item) {
    bool flag = false;
    do {
      if ((flag = atomicCAS(&lock, 0, 1)) == 0) {
        if (size < max_size) {
          data[tail] = item;
          tail = (tail + 1) % max_size;
          size++;
        }
      }
      __threadfence();
      if (flag) {
        atomicExch(&lock, 0);
      }
      break;
    } while (!flag);
  }

  __device__ bool atomic_pop(T* item) {
    bool flag = false;
    bool return_flag = false;
    do {
      if ((flag = atomicCAS(&lock, 0, 1)) == 0) {
        if (size > 0) {
          *item = data[head];
          head = (head + 1) % max_size;
          size--;
          return_flag = true;
        }
      }
      __threadfence();
      if (flag) {
        atomicExch(&lock, 0);
      }
    } while (!flag);
    return return_flag;
  }

  __device__ void atomic_push2(T item) {
    while (atomicCAS(&lock, 0, 1) == 1) {
    }
    if (size < max_size) {
      data[tail] = item;
      tail = (tail + 1) % max_size;
      size++;
    }
    atomicExch(&lock, 0);
  }

  __device__ bool atomic_pop2(T* item) {
    while (atomicCAS(&lock, 0, 1) == 1) {
    }
    if (size > 0) {
      *item = data[head];
      head = (head + 1) % max_size;
      size--;
      atomicExch(&lock, 0);
      return true;
    }
    atomicExch(&lock, 0);
    return false;
  }
  __device__ bool atomic_empty() {
    bool flag = false;
    bool return_flag = false;
    do {
      if ((flag = atomicCAS(&lock, 0, 1)) == 0) {
        if (size <= 0) {
          return_flag = true;
        }
      }
      __threadfence();
      if (flag) {
        atomicExch(&lock, 0);
      }
    } while (!flag);
    return return_flag;
  }
  __device__ bool atomic_empty2() {
    while (atomicCAS(&lock, 0, 1) == 1) {
    }
    if (size <= 0) {
      atomicExch(&lock, 0);
      return true;
    }
    atomicExch(&lock, 0);
    return false;
  }
};
