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
#include "io.hpp"
#include "io_mtx.hpp"
#include "memory_store.hpp"
#include "cluster_data.h"
#include "ExperimentLoggerUtil.hpp"
#include "clustering_methods.hpp"
#include "weighted_graph.h"
#include "parse_args.hpp"
#include "vertex_weighting.hpp"
#include "objective_helpers.hpp"
#include <queue>
#include <memory>

using namespace jet_community;
using rfd_t = cluster_data<matrix_t>;
using wg_t = weighted_graph<matrix_t>;
using mem_t = memory_store<matrix_t>;
using cm_t = clustering_methods<matrix_t>;

// checks how many components graph has after "deleting" cut edges of part_d
void connected_comps(matrix_t g, part_vt part_d){
    ordinal_t n = g.numRows();
    part_mt comp_ids("component ids", n);
    auto part = Kokkos::create_mirror_view(part_d);
    Kokkos::deep_copy(part, part_d);
    Kokkos::deep_copy(comp_ids, -1);
    auto row_map = Kokkos::create_mirror_view(g.graph.row_map);
    Kokkos::deep_copy(row_map, g.graph.row_map);
    auto entries = Kokkos::create_mirror_view(g.graph.entries);
    Kokkos::deep_copy(entries, g.graph.entries);
    ordinal_t t_comps = 0;
    for(ordinal_t i = 0; i < n; i++){
        if(comp_ids(i) != -1) continue;
        std::queue<ordinal_t> q;
        q.push(i);
        comp_ids(i) = t_comps;
        while(!q.empty()){
            ordinal_t u = q.front();
            q.pop();
            for(edge_offset_t j = row_map(u); j < row_map(u+1); j++){
                ordinal_t v = entries(j);
                if(comp_ids(v) == -1 && part(v) == part(u)){
                    comp_ids(v) = t_comps;
                    q.push(v);
                }
            }
        }
        t_comps++;
    }
    std::cout << "Total components: " << t_comps << std::endl;
    // return comp_ids;
}

part_vt run_clustering(const wg_t wg,
                    double& obj,
                    const cluster_args args,
                    ExperimentLoggerUtil<value_t>& experiment) {
    std::unique_ptr<rfd_t> rfd = get_objective(wg, args);
    mem_t mem(wg.mtx, *rfd);
    Kokkos::fence();
    Kokkos::Timer iteration;
    std::cout << std::setprecision(6);
    part_vt constraint;
#ifdef LEIDEN
    part_vt part = cm_t::leiden_part<false>(mem, wg, *rfd, experiment, constraint);
#else
    part_vt part = cm_t::louvain_part<false>(mem, wg, *rfd, experiment, constraint);
#endif
    Kokkos::fence();
    double time = iteration.seconds();
    std::cout << "Cluster time: " << time << "; " << *rfd << std::endl;
    iteration.reset();
    for(int i = 0; i < args.n_successive_iterations; i++){
        constraint = part;
#ifdef LEIDEN
        part = cm_t::leiden_part<true>(mem, wg, *rfd, experiment, constraint);
#else
        part = cm_t::louvain_part<true>(mem, wg, *rfd, experiment, constraint);
#endif
        Kokkos::fence();
        time = iteration.seconds();
        std::cout << "Cluster time: " << time << "; " << *rfd << std::endl;
        iteration.reset();
    }
    experiment.setModularity(rfd->get_objective());
    experiment.setEdgeCut((wg.mtx.nnz() - rfd->uncut) / 2);
    obj = rfd->get_objective();
    return part;
}

int main(int argc, char **argv) {

    const cluster_args args = parse_cluster_args(argc, argv);
    if(!args.valid) return -1;

    Kokkos::initialize(argc, argv);
    //must scope kokkos-related data
    //so that it falls out of scope b4 finalize
    {
        matrix_t g;
        bool uniform_ew = false;
        if(!load_graph(g, uniform_ew, args.graph_file.c_str())) return -1;
        std::cout << "Vertex Count: " << g.numRows() << "; Undirected Edge Count: " << g.nnz() / 2 << std::endl;
        if(!uniform_ew) uniform_ew = sanitize_edge_weights(g, args);
        std::cout << std::endl;

        wg_t wg;
        wg.mtx = g;
        wg.vtx_w = get_vtx_weights(g, args);
        wg.edge_uniform = uniform_ew;
        part_vt best_clusters;
        double best_mod = -std::numeric_limits<double>::infinity();
        Kokkos::fence();
        for(int i = 0; i < args.n_trials; i++){
            ExperimentLoggerUtil<value_t> experiment;
            double mod;
            Kokkos::Timer total_time;
            part_vt clusters = run_clustering(wg, mod, args, experiment);
            std::cout << "Total time: " << total_time.seconds() << std::endl;
            experiment.addMeasurement(Measurement::Total, total_time.seconds());
            std::cout << std::endl;
            if(mod > best_mod){
                best_mod = mod;
                best_clusters = clusters;
            }
            if(args.metrics_file.size() > 0){
                experiment.log(args.metrics_file.c_str(), i == 0, (i+1) == args.n_trials);
            }
        }

        std::cout << std::setprecision(9) << "Best objective found: " << best_mod << std::endl;
        if(args.output_file.size() > 0){
            std::cout << "Writing best clustering to " << args.output_file << std::endl;
            write_part(best_clusters, args.output_file.c_str());
        }
    }
    Kokkos::finalize();

    return 0;
}
