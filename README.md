# A K-Way Graph Partitioner for the GPU

Depends on Kokkos (https://github.com/kokkos/kokkos), KokkosKernels (https://github.com/kokkos/kokkos-kernels), and Metis (https://github.com/KarypisLab/METIS).

This is a parallel k-way undirected graph partitioning algorithm that implements the multilevel method. Uses novel GPU implementations of existing coarsening algorithms, and a novel refinement algorithm designed for the GPU. Initial partitioning is performed on extremely coarse graphs using Metis.

## Usage

### Executables

#### Partitioners
Each partitioner executable requires 2 parameters. The first is a graph file in metis format, the second is a config file. Optionally, a third parameter can be used to specify an output file for the partition, and a fourth parameter for runtime statistics in JSON format.  
Although the partitioner itself supports weighted edges and vertices, the import method currently does not support weighted vertices.  
jet: The primary partitioner exe. Coarsening algorithm can be set in config file. Runs on the default device.  
jet\_host: jet but runs on the host device.  
jet\_serial: jet but runs on the host on a single thread.

#### Helpers
pstat: Given a metis graph file, partition file, and k-value, will print out quality information on the partition.

### Library
We do not currently provide an option to compile a library. However, you can import "jet.hpp" into your code to use the partitioner via the "jet\_partitioner::partition" method. Note that this requires you to add our source directory to your include path and also to link our dependencies.

### Input Format
We do not yet support vertex weights within metis graph files.

### Config File format:  
\<Coarsening algorithm\> (0 -> 2-hop)/(1 -> HEC)/(2 -> pure match)/(default -> 2-hop)  
\<Number of parts\>  
\<Partitioning attempts\>  
\<Imbalance value\>  
