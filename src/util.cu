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

// size_t get_avail_mem_nvml()
// {
//   nvmlReturn_t result;
//   nvmlDevice_t device;
//   nvmlMemory_t mem_info;
//   result = nvmlInit();
//   result = nvmlDeviceGetHandleByIndex(0, &device);
//   result = nvmlDeviceGetMemoryInfo(device, &mem_info);
//   printf("Amount of total memory: %g GB, avail memory: %g GB, take up: %g GB, %g MB, %g KB\n", mem_info.total / (1024.0 * 1024.0 * 1024.0), mem_info.free / (1024.0 * 1024.0 * 1024.0), mem_info.used / (1024.0 * 1024.0 * 1024.0), mem_info.used / (1024.0 * 1024.0), mem_info.used / (1024.0));

//   result = nvmlShutdown();
//   return mem_info.free;
// }

// https://cse.usf.edu/~kchriste/tools/genzipf.c
__global__ void zipf_gen(float alpha, int n, float *rand_array, int len, float c)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = tid; i < len; i += stride)
  {
    float sum_prob = 0;
    for (int j = 1; j <= n; j++)
    {
      sum_prob += (c / pow((float)j, alpha));
      if (sum_prob >= rand_array[i])
      {
        rand_array[i] = j;
        break;
      }
    }
  }
}

void zipf(float *d_rand_array, int len, float alpha = 1.0, int n = 10)
{
  float c = 0; // Normalization constant

  // Calculate normalization constant on the CPU
  for (int i = 1; i <= n; i++)
  {
    c = c + (1.0 / pow((float)i, alpha));
  }
  c = 1.0 / c;

  zipf_gen<<<BLOCK_SIZE * 4, BLOCK_SIZE>>>(alpha, n, d_rand_array, len, c);
  cudaDeviceSynchronize();
}

// //=========================================================================
// //= Multiplicative LCG for generating uniform(0.0, 1.0) random numbers    =
// //=   - x_n = 7^5*x_(n-1)mod(2^31 - 1)                                    =
// //=   - With x seeded to 1 the 10000th x value should be 1043618065       =
// //=   - From R. Jain, "The Art of Computer Systems Performance Analysis," =
// //=     John Wiley & Sons, 1991. (Page 443, Figure 26.2)                  =
// //=========================================================================
// double rand_val(int seed)
// {
//   const long a = 16807;      // Multiplier
//   const long m = 2147483647; // Modulus
//   const long q = 127773;     // m div a
//   const long r = 2836;       // m mod a
//   static long x;             // Random int value
//   long x_div_q;              // x divided by q
//   long x_mod_q;              // x modulo q
//   long x_new;                // New x value

//   // Set the seed if argument is non-zero and then return zero
//   if (seed > 0)
//   {
//     x = seed;
//     return (0.0);
//   }

//   // RNG using integer arithmetic
//   x_div_q = x / q;
//   x_mod_q = x % q;
//   x_new = (a * x_mod_q) - (r * x_div_q);
//   if (x_new > 0)
//     x = x_new;
//   else
//     x = x_new + m;

//   // Return a random value between 0.0 and 1.0
//   return ((double)x / m);
// }

int get_clk()
{
  int device;
  int peak_clk = 1;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&peak_clk, cudaDevAttrClockRate, device);
  return peak_clk;
}