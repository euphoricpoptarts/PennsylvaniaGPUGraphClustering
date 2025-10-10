# Graph Clustering Algorithms for the GPU

Depends on Kokkos (https://github.com/kokkos/kokkos), KokkosKernels (https://github.com/kokkos/kokkos-kernels), and the Cuda Toolkit (version >12.0).

This is a parallel undirected graph clustering package that implements novel GPU-first formulations of the Louvain and Leiden clustering algorithms.

## Reasons to Use These Programs

### Fastest GPU Clustering Algorithms
![Comparison of Clustering Runtimes](images/runtime_comparison-1.png)

### Quality Competitive with Memetic Algorithms
![Comparison of Clustering Modularity](images/modularity_comparison-1.png)

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
**Graph File**: A graph represented in the Metis file format.  
**Additional Iteration Count**: Applies the program successively on its own output for the given iteration count.  
**Trial Count**: Performs this number of clustering trials, and returns the best clustering.  
**Clustering Output File**: Writes the clustering, numbered `0` to `c-1` for `c` clusters, to the given file. Line `x` gives the cluster to which vertex `x` belongs.

#### Helpers
**calc_mod**: Takes a graph file and a clustering file as defined above as input. Computes the modularity of the given clustering on the given graph. Invoke as below:  
`./build/app/calc_mod <required graph file> <required clustering file>`

### Library
There is currently no library available to programmatically invoke our clustering methods.  
If you are interested in a c++ library or a python wrapper to these methods, please open an issue to let us know.

### Input Format
We do not yet support vertex weights within metis graph files.

## Planned Features
Support for Matrix Marketplace graph files.  
Support for user-specified LambdaCC objective function configurations.

### Possible Features
Memetic algorithm with our programs.  
Python wrapper.  
C++ library.