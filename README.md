# Graph Clustering Algorithms for the GPU

Depends on Kokkos (https://github.com/kokkos/kokkos), KokkosKernels (https://github.com/kokkos/kokkos-kernels), and the Cuda Toolkit (version >12.0).

This is a parallel undirected graph clustering package that implements novel GPU-first formulations of the Louvain and Leiden clustering algorithms.  
Our pLeiden and pLeiden+ implementations are the first to provide the Leiden algorithm's six original guarantees in a parallel setting.

## Usage

### Executables

#### Clustering Programs
Each clustering program has 1 required parameter and 3 optional parameters.
You will invoke each as below:  
`./build/app/<program> <required graph file> <optional additional iteration count> <optional trial count> <optional clustering output file>`  

**pLouvain**: Run the Louvain+ algorithm. Choose this one for fast, high-quality clustering.  
**pLeiden**: Run the Leiden algorithm. Choose this one for intra-cluster connectivity guarantees, but lower quality than pLouvain or pLeiden+.  
**pLeiden+**: Run the Leiden+ algorithm. Use this one with multiple iterations for highest-quality. Intra-cluster connectivity guarantees are delayed until algorithm encounters a stable iteration.  
Each program currently only optimizes for modularity.

##### Parameters
**Graph File**: A graph represented in the Metis file format or the Matrix Market file format. Files ending in `".mtx"` will be interpreted as Matrix Market files, all other file extensions will be interpreted as Metis files.  
**Additional Iteration Count**: Applies the program successively on its own output for the given iteration count.  
**Trial Count**: Performs this number of clustering trials, and returns the best clustering.  
**Clustering Output File**: Writes the clustering, numbered `0` to `c-1` for `c` clusters, to the given file. Line `x` gives the cluster to which vertex `x` belongs.

#### Evolutionary/Memetic Clustering
**meme**: Uses a memetic clustering algorithm inspired by VieClus.  
Initial clustering pool is generated with pLeiden+.  
Recombination is performed by pLouvain.  
Mutation operator is performed with 5 iterations of pLeiden+.  
Mutation operator is performed on the output of the recombination operator.  
Quality is generally superior to VieClus.  
Invoke as:  
`./build/app/meme <required graph file> <required pool size> <required time limit in seconds> <optional clustering output file>`

#### Helpers
**calc_mod**: Takes a graph file and a clustering file as defined above as input. Computes the modularity of the given clustering on the given graph. Invoke as below:  
`./build/app/calc_mod <required graph file> <required clustering file>`

### Library
There is currently no library available to programmatically invoke our clustering methods.

### Input Format
We ignore vertex weights within metis graph files.  
Programs may not work as expected if edge weights are given in metis format.  
We ignore edge weights in matrix market files.  
Robust support for weighted edges is forthcoming.

## Comparison versus State of the Art Parallel Clustering Algorithms
The below images compare our programs with the following state-of-the-art competitors:  
v-Louvain: https://github.com/puzzlef/louvain-communities-cuda  
GALA: https://github.com/LinXi-lx/GALA  
GVE-Louvain: https://github.com/puzzlef/louvain-communities-openmp  
GVE-Leiden: https://github.com/puzzlef/leiden-communities-openmp  
Networkit Louvain  
Networkit Leiden  
Cugraph Louvain  
Cugraph Leiden

GPU programs (pLouvain, pLeiden, pLeiden+, v-Louvain, GALA, cugraph Louvain/Leiden) are run on an Nvidia B200 GPU.  
CPU programs (GVE-Louvain/Leiden, Networkit Louvain/Leiden) are run on an AMD Ryzen 9950x3D CPU.

### Runtime Comparison
![Comparison of Clustering Runtimes](images/runtime_comparison-1.png)
pLouvain is up to 1200x faster than Cugraph Louvain for some graphs.

### Modularity Comparison
![Comparison of Clustering Modularity](images/modularity_comparison-1.png)
Networkit Leiden is off the chart at -0.415.  
pLouvain and pLeiden+ achieve higher quality than each state-of-the-art competitor nearly universally.

## Planned Features
Support for user-specified LambdaCC objective function configurations.

### Possible Features
Python wrapper.  
C++ library.

If you are interested in these potential features, please open an issue to let us know.