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
#include "jet_refiner.hpp"
#include "defs.h"
#include "io.hpp"
#include "contract.hpp"
#include <limits>
#include <queue>

using namespace jet_partitioner;
using ref_t = jet_refiner<matrix_t, part_t>;
using rfd_t = typename ref_t::refine_data;
using stat = part_stat<matrix_t, part_t>;
using contracter_t = contracter<matrix_t>;
using clt = contracter_t::coarse_level_triple;

part_vt connected_comps(matrix_t g, part_vt part){
    ordinal_t n = g.numRows();
    part_vt comp_ids("component ids", n);
    Kokkos::deep_copy(comp_ids, -1);
    ordinal_t t_comps = 0;
    for(ordinal_t i = 0; i < n; i++){
        if(comp_ids(i) != -1) continue;
        std::queue<ordinal_t> q;
        q.push(i);
        comp_ids(i) = t_comps;
        while(!q.empty()){
            ordinal_t u = q.front();
            q.pop();
            for(edge_offset_t j = g.graph.row_map(u); j < g.graph.row_map(u+1); j++){
                ordinal_t v = g.graph.entries(j);
                if(comp_ids(v) == -1 && part(v) == part(u)){
                    comp_ids(v) = t_comps;
                    q.push(v);
                }
            }
        }
        t_comps++;
    }
    std::cout << "Total components: " << t_comps << std::endl;
    return comp_ids;
}

part_vt rec_part(ref_t& refiner, clt c, rfd_t& rfd, wgt_view_t wdeg, int countdown, ExperimentLoggerUtil<value_t>& experiment){
    part_vt part("cluster assignments", c.mtx.numRows());
    Kokkos::parallel_for("set initial assignments", r_policy(0, c.mtx.numRows()), KOKKOS_LAMBDA(const ordinal_t i){
        part(i) = i;
    });
    std::cout << "Pre-refine" << std::endl;
    refiner.jet_refine(c.mtx, wdeg, c.nb_self_loops, part, rfd, experiment);
    stat::relabel(part, rfd);
    if(countdown > 0){
        contracter_t contracter;
        ordinal_t labels = stat::get_total_labels(part);
        clt active_clt = contracter.build_coarse_graph(c, part, labels, experiment);
        wgt_view_t active_wdeg("weighted degree 2", labels);
        Kokkos::deep_copy(active_wdeg, rfd.total_deg);
        part_vt active_part = rec_part(refiner, active_clt, rfd, active_wdeg, countdown - 1, experiment);
        Kokkos::parallel_for("update top level assignments", r_policy(0, c.mtx.numRows()), KOKKOS_LAMBDA(const ordinal_t i){
            part(i) = active_part(part(i));
        });

        labels = stat::get_total_labels(part);
        std::cout << "Total labels " << labels << std::endl;
        labels = stat::get_total_labels(active_part);
        std::cout << "Total labels " << labels << std::endl;
        std::cout << "Post-refine" << std::endl;
        refiner.jet_refine(c.mtx, wdeg, c.nb_self_loops, part, rfd, experiment);
        stat::relabel(part, rfd);
    }
    return part;
}

part_vt rec_part(ref_t& refiner, part_vt part, clt c, rfd_t& rfd, wgt_view_t wdeg, int countdown, ExperimentLoggerUtil<value_t>& experiment){
    std::cout << "Pre-refine" << std::endl;
    refiner.jet_refine(c.mtx, wdeg, c.nb_self_loops, part, rfd, experiment);
    stat::relabel(part, rfd);
    if(countdown > 0){
        contracter_t contracter;
        ordinal_t labels = stat::get_total_labels(part);
        clt active_clt = contracter.build_coarse_graph(c, part, labels, experiment);
        wgt_view_t active_wdeg("weighted degree 2", labels);
        Kokkos::deep_copy(active_wdeg, rfd.total_deg);
        part_vt active_part = rec_part(refiner, active_clt, rfd, active_wdeg, countdown - 1, experiment);
        Kokkos::parallel_for("update top level assignments", r_policy(0, c.mtx.numRows()), KOKKOS_LAMBDA(const ordinal_t i){
            part(i) = active_part(part(i));
        });

        labels = stat::get_total_labels(part);
        std::cout << "Total labels " << labels << std::endl;
        labels = stat::get_total_labels(active_part);
        std::cout << "Total labels " << labels << std::endl;
        std::cout << "Post-refine" << std::endl;
        refiner.jet_refine(c.mtx, wdeg, c.nb_self_loops, part, rfd, experiment);
        stat::relabel(part, rfd);
    }
    return part;
}

part_vt partition(value_t& edge_cut,
                                  matrix_t g,
                                  wgt_view_t vweights,
                                  ExperimentLoggerUtil<value_t>& experiment) {
    rfd_t rfd;
    ref_t refiner(g, 256);
    wgt_view_t nb_self_loops("self loop counter", g.numRows());
    clt c;
    c.mtx = g;
    c.nb_self_loops = nb_self_loops;
    clt active_clt = c;
    Kokkos::fence();
    Kokkos::Timer t;
    part_vt part = rec_part(refiner, c, rfd, vweights, 2, experiment);
    Kokkos::fence();
    part = connected_comps(g, part);
    std::cout << t.seconds() << std::endl;
    ordinal_t labels = stat::get_total_labels(part);
    stat::reset_rfd(g, part, labels, rfd);
    //part = rec_part(refiner, part, c, rfd, vweights, 2, experiment);
    //part = connected_comps(g, part);
    std::cout << "Total labels " << labels << std::endl;
    double modularity = stat::modularity(g, part, labels, g.nnz());
    std::cout << "Modularity: " << std::setprecision(6) << modularity << std::endl;
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

    Kokkos::initialize(argc, argv);
    //must scope kokkos-related data
    //so that it falls out of scope b4 finalize
    {
        matrix_t g;
        bool uniform_ew = false;
        if(!load_metis_graph(g, uniform_ew, filename)) return -1;
        std::cout << "vertices: " << g.numRows() << "; edges: " << g.nnz() / 2 << std::endl;
        wgt_view_t vweights("vertex weights", g.numRows());
        degree_weighting(g, vweights);
        //Kokkos::deep_copy(vweights, 1);

        part_vt best_part;
        value_t edgecut = 0;
        ExperimentLoggerUtil<value_t> experiment;
        part_vt part = partition(edgecut, g, vweights, experiment);

        if(part_file != nullptr) write_part(best_part, part_file);
    }
    Kokkos::finalize();

    return 0;
}
