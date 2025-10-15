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
#include "defs.h"
#include "io_mtx.hpp"
#include <limits>

using namespace jet_community;

using scalar_t = ordinal_t;

double modularity(const matrix_t g, const part_vt labels, const ordinal_t label_count, scalar_t g_degree, double penalty_scale){
    wgt_view_t total("total degree", label_count);
    ordinal_t n = g.numRows();
    scalar_t uncut = 0;
    Kokkos::parallel_reduce("count degrees", r_policy(0, n), KOKKOS_LAMBDA(const ordinal_t i, scalar_t& update){
        scalar_t td = 0;
        ordinal_t l = labels(i);
        for(edge_offset_t j = g.graph.row_map(i); j < g.graph.row_map(i+1); j++){
            td += g.values(j);
            ordinal_t v = g.graph.entries(j);
            if(l == labels(v)) update += g.values(j);
        }
        Kokkos::atomic_add(&total(l), td);
	}, uncut);
    int64_t square_sum = 0;
    Kokkos::parallel_reduce("sum modularity", r_policy(0, label_count), KOKKOS_LAMBDA(const ordinal_t l, int64_t& update){
        int64_t square = total(l);
        square = square*square;
        update += square;
    }, square_sum);
    double inv_gdeg = 1.0 / static_cast<double>(g_degree);
    double pen = penalty_scale*square_sum*inv_gdeg;
    double m = (uncut - pen)*inv_gdeg;
    return m;
}

int main(int argc, char **argv) {

    if (argc < 3) {
        std::cerr << "Insufficient number of args provided" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <metis_graph_file> <part_file>" << std::endl;
        return -1;
    }
    char *filename = argv[1];
    char *part_file = argv[2];

    Kokkos::initialize();
    //must scope kokkos-related data
    //so that it falls out of scope b4 finalize
    {
        matrix_t g;
        bool uniform_ew = false;
        if(!load_graph(g, uniform_ew, filename)) return -1;
        std::cout << "vertices: " << g.numRows() << "; edges: " << g.nnz() / 2 << std::endl;
        wgt_view_t vweights("vertex weights", g.numRows());
        Kokkos::deep_copy(vweights, 1);

        part_vt part = load_part(g.numRows(), part_file);
        std::cout << "Modularity: " << std::setprecision(9) << modularity(g, part, g.numRows(), g.nnz(), 1.0) << std::endl;
    }
    Kokkos::finalize();

    return 0;
}
