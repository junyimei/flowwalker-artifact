## FlowWalker Artifact

This is the artifact of submission paper **FlowWalker: A Memory-efficient and High-performance GPU-based Dynamic Graph Random Walk Framework**. 

To run this code, make sure your system has installed the following libraries properly:

```
CUDA (>=11.6)
CMake (>=3.15)
g++ (>=9.4)
gflags (>= 2.2, can be installed with: apt-get install libgflags-dev)
```

### Compile

To complie this project, enter the root directory add execute:
```
mkdir build && cd build
cmake ..
make -j
```

### Pre-processing
The input of FlowWalker is transformed into CSR format. We provide an example graph [wiki-Vote](http://snap.stanford.edu/data/wiki-Vote.html). You can use this graph or download any other dataset you would like to use. Please refer to README.md in the data folder for more details.

### Execution 
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
