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
#include "util.cuh"
#include "myrand.cuh"
double wtime()
{
  double time[2];
  struct timeval time1;
  gettimeofday(&time1, NULL);

  time[0] = time1.tv_sec;
  time[1] = time1.tv_usec;

  return time[0] + time[1] * 1.0e-6;
}

__device__ uint binary_search(float *prob, int size, float target)
{
  int l = 0;
  int r = size - 1;
  while (l < r)
  {
    int mid = (l + r) / 2;
    if (prob[mid] > target)
    {
      r = mid;
    }
    else
    {
      l = mid + 1;
    }
  }
  return l;
}

__global__ void warm_up_gpu()
{
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  ib += ia + tid;
}

size_t get_avail_mem()
{
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  size_t avail;
  size_t total;
  cudaMemGetInfo(&avail, &total);
  printf("Amount of total memory: %g GB, avail memory: %g GB, take up: %g GB, %g MB, %g KB\n", total / (1024.0 * 1024.0 * 1024.0), avail / (1024.0 * 1024.0 * 1024.0), (total - avail) / (1024.0 * 1024.0 * 1024.0), (total - avail) / (1024.0 * 1024.0), (total - avail) / (1024.0));
  return avail;
}

int get_clk()
{
  int device;
  int peak_clk = 1;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&peak_clk, cudaDevAttrClockRate, device);
  return peak_clk;
}