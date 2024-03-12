#ifndef _GPU_GRAPH_H_
#define _GPU_GRAPH_H_
#include <algorithm>
#include <iostream>
#include <gflags/gflags.h>
#include "util.cuh"
#include "graph.cuh"

DECLARE_bool(metapath);
DECLARE_bool(umgraph);

class gpu_graph
{
public:
	vtx_t vtx_num;				// vertex number
	vtx_t *adjncy;				// edge list
	weight_t *adjwgt = nullptr; // edge weight
	edge_t *xadj;				// vertex list (the offset of edge, csr)
	int *edge_label = nullptr;

public:
	~gpu_graph() {}
	gpu_graph() {}
	gpu_graph(graph *ginst)
	{
		vtx_num = ginst->vert_count;
		ll xadj_sz = sizeof(edge_t) * (ginst->vert_count + 1);
		ll edge_sz = sizeof(vtx_t) * ginst->edge_count;

		ll weight_sz = sizeof(weight_t) * ginst->edge_count;

		if (FLAGS_umgraph)
		{
			CUDA_RT_CALL(cudaMallocManaged(&xadj, xadj_sz));
			CUDA_RT_CALL(cudaMallocManaged(&adjncy, edge_sz));

			if (FLAGS_metapath)
			{
				CUDA_RT_CALL(cudaMallocManaged(&edge_label, edge_sz));
			}
			else
			{
				CUDA_RT_CALL(cudaMallocManaged(&adjwgt, weight_sz));
			}
		}
		else
		{
			CUDA_RT_CALL(cudaMalloc(&xadj, xadj_sz));
			CUDA_RT_CALL(cudaMalloc(&adjncy, edge_sz));

			if (FLAGS_metapath)
			{
				CUDA_RT_CALL(cudaMalloc(&edge_label, edge_sz));
			}
			else
			{
				CUDA_RT_CALL(cudaMalloc(&adjwgt, weight_sz));
			}
		}
		CUDA_RT_CALL(cudaMemcpy(xadj, ginst->xadj, xadj_sz, cudaMemcpyHostToDevice));
		CUDA_RT_CALL(cudaMemcpy(adjncy, ginst->adjncy, edge_sz, cudaMemcpyHostToDevice));

		if (FLAGS_metapath)
		{
			CUDA_RT_CALL(cudaMemcpy(edge_label, ginst->edge_label, edge_sz, cudaMemcpyHostToDevice));
		}
		else
		{
			CUDA_RT_CALL(cudaMemcpy(adjwgt, ginst->weight, weight_sz, cudaMemcpyHostToDevice));
		}
		// printf("copying graph done\n");
	}

	__device__ vtx_t getDegree(vtx_t idx)
	{
		// if(threadIdx.x==0)
		//   printf("xadj size:%u, idx=%u\n",vtx_num,idx);
		return xadj[idx + 1] - xadj[idx];
	}
	__device__ void print_max_degree()
	{
		int gid = (blockDim.x * blockIdx.x) + threadIdx.x;
		if (gid == 0)
		{
			vtx_t max_degree = 0;
			vtx_t max_idx = 0;
			for (auto i = 0; i < vtx_num; i++)
			{
				vtx_t degree = xadj[i + 1] - xadj[i];
				if (degree > max_degree)
				{
					max_degree = degree;
					max_idx = i;
				}
			}
			printf("Max degree: %d, idx: %d\n", max_degree, max_idx);
		}
	}

	__device__ bool binarySearch(vtx_t *arr, vtx_t size, vtx_t target)
	{
		vtx_t l = 0;
		vtx_t r = size - 1;
		while (l <= r)
		{
			vtx_t m = l + (r - l) / 2;

			// Check if x is present at mid
			if (arr[m] == target)
				return true;

			// If x greater, ignore left half
			if (arr[m] < target)
				l = m + 1;

			// If x is smaller, ignore right half
			else
				r = m - 1;
		}

		// If we reach here, then element was not present
		return false;
	}
	__device__ bool CheckConnect(vtx_t src, vtx_t dst)
	{
		// check whether vertex src and dst are connected using binary search
		vtx_t src_degree = getDegree(src);
		vtx_t dst_degree = getDegree(dst);
		if (src_degree < dst_degree)
		{
			return binarySearch(adjncy + xadj[src], src_degree, dst);
		}
		return binarySearch(adjncy + xadj[dst], dst_degree, src);
	}
	__device__ bool check_connect(vtx_t src, vtx_t dst)
	{
		// check whether vertex src and dst are connected using binary search
		vtx_t src_degree = getDegree(src);

		return binarySearch(adjncy + xadj[src], src_degree, dst);
	}
};

#endif
