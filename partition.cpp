// ***********************************************************************
// 
// Jet: Multilevel Graph Partitioning
//
// Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS). 
// 
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// ************************************************************************
#include "uncoarsen.hpp"
#include "defs.h"
#include "io.hpp"
#include <limits>

using namespace jet_partitioner;

part_vt partition(value_t& edge_cut,
                                  matrix_t g,
                                  wgt_view_t vweights,
                                  bool uniform_ew,
                                  ExperimentLoggerUtil<value_t>& experiment) {
    using ref_t = jet_refiner<matrix_t, part_t>;
    using rfd_t = typename ref_t::refine_data;
    rfd_t rfd;
    ref_t refiner(g, 256);
    part_vt part("cluster assignments", g.numRows());
    Kokkos::parallel_for("set initial assignments", r_policy(0, g.numRows()), KOKKOS_LAMBDA(const ordinal_t i){
        part(i) = i;
    });
    refiner.jet_refine(g, vweights, part, 0, rfd, experiment);
    edge_cut = rfd.cut;
    return part;
}

void degree_weighting(const matrix_t& g, wgt_view_t vweights){
    Kokkos::parallel_for("set v weights", r_policy(0, g.numRows()), KOKKOS_LAMBDA(const ordinal_t i){
        vweights(i) = g.graph.row_map(i + 1) - g.graph.row_map(i);
    });
}

int main(int argc, char **argv) {

    if (argc < 2) {
        std::cerr << "Insufficient number of args provided" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <metis_graph_file> <optional partition_output_filename> <optional metrics_filename>" << std::endl;
        return -1;
    }
    char *filename = argv[1];
    char *part_file = nullptr;
    char *metrics = nullptr;
    if(argc >= 3){
        part_file = argv[2];
    }
    if(argc >= 4){
        metrics = argv[3];
    }

    Kokkos::initialize();
    //must scope kokkos-related data
    //so that it falls out of scope b4 finalize
    {
        matrix_t g;
        bool uniform_ew = false;
        if(!load_metis_graph(g, uniform_ew, filename)) return -1;
        std::cout << "vertices: " << g.numRows() << "; edges: " << g.nnz() / 2 << std::endl;
        wgt_view_t vweights("vertex weights", g.numRows());
        Kokkos::deep_copy(vweights, 1);

        part_vt best_part;
        value_t edgecut = 0;
        ExperimentLoggerUtil<value_t> experiment;
        part_vt part = partition(edgecut, g, vweights, uniform_ew, experiment);

        if(part_file != nullptr) write_part(best_part, part_file);
    }
    Kokkos::finalize();

    return 0;
}
