## FlowWalker Artifact


- [FlowWalker Artifact](#flowwalker-artifact)
- [Code Structure](#code-structure)
- [Software Requirements](#software-requirements)
- [Setup](#setup)
- [Dataset and Pre-processing](#dataset-and-pre-processing)
- [Execution](#execution)
- [Reproducing Results in Paper](#reproducing-results-in-paper)
  - [Environment](#environment)
  - [Overall Comparision (Table 2)](#overall-comparision-table-2)
  - [Sampler Test (Figure 6)](#sampler-test-figure-6)
  - [Ablation Study (Figure 7)](#ablation-study-figure-7)


This is the source code of submission paper **FlowWalker: A Memory-efficient and High-performance GPU-based Dynamic Graph Random Walk Framework**. 

`FlowWalker` is a GPU-based dynamic graph random walk framework. It implements an efficient parallel reservoir sampling method to fully exploit the GPU parallelism and reduce space complexity. Moreover, it employs a sampler-centric paradigm alongside a dynamic scheduling strategy to handle the huge amounts of walking queries. `FlowWalker` stands as a high-throughput as well as a memory-efficient framework that requires no auxiliary data structures in GPU global memory. 

For further details, please refer to our [paper](#) and [technical report](https://arxiv.org/abs/2404.08364). Feel free to email Junyi Mei by meijunyi AT sjtu.edu.cn if you have any questions. We are looking forward to receive your feedbacks.


## Code Structure

```
├── CMakeLists.txt
├── include
│   ├── app.cuh         # define different random walk applications
│   ├── gpu_graph.cuh
│   ├── gpu_queue.cuh
│   ├── gpu_task.cuh    # data structure for sampling queries stored in shared memory
│   ├── graph.cuh
│   ├── myrand.cuh
│   ├── sampler.cuh     # define samplers
│   ├── util.cuh
│   └── walker.cuh      # random walk kernels
├── README.md
├── src
│   ├── main.cu
│   ├── sampler_test.cu # test samplers
│   ├── util.cu
└── └── walk.cu
```

## Software Requirements
To run this code, make sure your system has installed the following libraries properly:

```
CUDA (>=11.6)
CMake (>=3.15)
g++ (>=9.4)
gflags (>= 2.2, can be installed with: apt-get install libgflags-dev)
```

## Setup

To complie this project, enter the root directory add execute:
```
mkdir build && cd build
cmake ..
make -j
```

## Dataset and Pre-processing
The input of FlowWalker is transformed into CSR format. We provide an example graph [wiki-Vote](http://snap.stanford.edu/data/wiki-Vote.html). You can use this graph or download any other dataset you would like to use. Please refer to README.md in the data folder for more details.

## Execution 
Execute DeepWalk with command:
```
cd build
./bin/flowwalker --input ../data/wiki-Vote --deepwalk --n=100
```
This will execute 100 walking queries on the dataset `wiki-Vote`. If you want to execute queries starting from vertices in the graph, you can enable the option `--all`. This will enable the batch processing, the default batch size is $10^7$, you can use `--batchsize=xxx` to modify it. And the option `--autobatch` helps you decide the batch size as described in the paper.

Execute PPR with command:
```
./bin/flowwalker --input ../data/wiki-Vote --ppr --n=100 
```
You can use `--tp=xx` to determine termination probability. Default is `--tp=0.2`.

Execute Node2Vec with command:
```
./bin/flowwalker --input ../data/wiki-Vote --node2vec --n=100 --dprs
```
You can use `--p=xx --q=xx` to determine two hyperparameters of Node2Vec. Default is `--p=2.0 --q=0.5`.

Execute MetaPath with command:
```
./bin/flowwalker --input ../data/wiki-Vote --metapath --n=100
```
You can use `--schema=xxxx` to determine the schema. Default is `--schema=0,1,2,3,4`.

## Reproducing Results in Paper

### Environment
In order to reproduce the results in the paper, we recommend running this program on the server equipped with following hardwares:

| Hardware | Requirement | 
| :-----:| :---- | 
| GPU | A100 (40GB) | 
| Main Memory | $\geq$ 256GB (converting data format for large graphs) | 
| PCI-E | 4.0 × 16, with 31.5GB/s bandwidth |

### Overall Comparision (Table 2)
Table 2 in paper conducts the overall performance of FlowWalker and other baselines on four random walk algorithms. To run FlowWalker, use the following command:
```
./bin/flowwalker --input $PATH_TO_DATA --$APP --all --autobatch
```
`$APP` includes `deepwalk`, `ppr`, `node2vec`, and `metapath`.

### Sampler Test (Figure 6)
Figure 6 tests the performance of various samplers under different sampling sizes. Figure 6(a) compares the running time of ZPRS, ITS, and ALS samplers:
```
./bin/TestSampler.out --groupsize=$SAMPLING_SIZE --gran=32 --type=1  #ZPRS(warp)
./bin/TestSampler.out --groupsize=$SAMPLING_SIZE --gran=512 --type=2 #ZPRS(block)
./bin/TestSampler.out --groupsize=$SAMPLING_SIZE --gran=32 --type=5  #ITS(warp)
./bin/TestSampler.out --groupsize=$SAMPLING_SIZE --gran=512 --type=6 #ITS(block)
./bin/TestSampler.out --groupsize=$SAMPLING_SIZE --gran=32 --type=7  #ALS(warp)
./bin/TestSampler.out --groupsize=$SAMPLING_SIZE --gran=512 --type=8 #ALS(block)
```

Figure 6(b) presents the speedup of RNG optimization.
```
./bin/TestSampler.out --groupsize=$SAMPLING_SIZE --gran=32 --type=1  # optimized (warp)
./bin/TestSampler.out --groupsize=$SAMPLING_SIZE --gran=32 --type=21 # curand (warp)
./bin/TestSampler.out --groupsize=$SAMPLING_SIZE --gran=512 --type=2 # optimized (block)
./bin/TestSampler.out --groupsize=$SAMPLING_SIZE --gran=512 --type=21 # curand (block)
```

Figure 6(c) compares the performance of ZPRS and DPRS.
```
./bin/TestSampler.out --groupsize=$SAMPLING_SIZE --gran=32 --type=1  #ZPRS(warp)
./bin/TestSampler.out --groupsize=$SAMPLING_SIZE --gran=512 --type=2 #ZPRS(block)
./bin/TestSampler.out --groupsize=$SAMPLING_SIZE --gran=32 --type=3  #DPRS(warp)
./bin/TestSampler.out --groupsize=$SAMPLING_SIZE --gran=512 --type=4 #DPRS(block)
```

### Ablation Study (Figure 7)
The command to conduct ablation study of FlowWalker:
```
./bin/flowwalker --input $PATH_TO_DATA --deepwalk --seq --n=1000000 --d=80 --walkmode=$WALKMODE
```
`$WALKMODE` defines FlowWalker with different levels of optimizations:
```
--walkmode=9    : FW
--walkmode=10   : FW + RNG
--walkmode=1    : FW + ZPRS (incorporating RNG)
--walkmode=6    : FW + DS (incorporating RNG+ZPRS)
```