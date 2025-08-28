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
#include "memory_store.hpp"
#include "cluster_data.h"
#include "part_stat.hpp"
#include "ExperimentLoggerUtil.hpp"
#include "hec.hpp"
#include <limits>
#include <queue>

using namespace jet_community;
using ref_t = jet_refiner<matrix_t>;
using vtx_vt = typename ref_t::vtx_vt;
using rfd_t = cluster_data<matrix_t>;
using pstat = part_stat<matrix_t, part_t>;
using contracter_t = contracter<matrix_t>;
using clt = contracter_t::coarse_level_triple;
using mem_t = memory_store<matrix_t>;
using hec_t = coarsen_heuristics<matrix_t, part_t>;

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

matrix_t constraint_graph(const matrix_t& g, const part_vt& constraint, vtx_view_t scratch){
    edge_view_t row_map("row map", g.numRows() + 1);
    Kokkos::parallel_for("mark", policy(g.numRows(), Kokkos::AUTO), KOKKOS_LAMBDA(const member& t){
        ordinal_t i = t.league_rank();
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [&] (const edge_offset_t& j, gain_t& update){
            ordinal_t v = g.graph.entries(j);
            if(constraint(v) == constraint(i)){
                scratch(j) = 1;
                update++;
            } else {
                scratch(j) = 0;
            }
        }, row_map(i));
    });
    edge_offset_t nnz = 0;
    Kokkos::parallel_scan("scan offsets", r_policy(0, g.numRows() + 1), KOKKOS_LAMBDA(const ordinal_t i, edge_offset_t& update, const bool final){
        edge_offset_t val = row_map(i);
        if(final){
            row_map(i) = update;
        }
        update += val;
    }, nnz);
    vtx_view_t entries(Kokkos::ViewAllocateWithoutInitializing("entries"), nnz);
    wgt_view_t values(Kokkos::ViewAllocateWithoutInitializing("entries"), nnz);
    Kokkos::parallel_scan("stream compaction", r_policy(0, g.nnz()), KOKKOS_LAMBDA(const edge_offset_t j, edge_offset_t& insert, const bool final){
        if(scratch(j) == 1){
            if(final){
                entries(insert) = g.graph.entries(j);
                values(insert) = g.values(j);
            }
            insert++;
        }
    });
    matrix_t cg("constraint graph", g.numRows(), g.numRows(), nnz, values, row_map, entries);
    return cg;
}

static void coarsen_vtx_w(wgt_view_t in, wgt_view_t out, vtx_view_t map){
    Kokkos::parallel_for("set v weights", r_policy(0, in.extent(0)), KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t c = map(i);
        Kokkos::atomic_add(&out(c), in(i));
    });
}

static void downsample(vtx_vt in, vtx_vt out, vtx_view_t map){
    Kokkos::parallel_for("set v weights", r_policy(0, in.extent(0)), KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t c = map(i);
        out(c) = in(i);
    });
}

part_vt leiden_part(mem_t& mem, clt top, rfd_t& rfd, ExperimentLoggerUtil<value_t>& experiment){
    std::vector<clt> levels;
    std::vector<part_vt> parts;
    levels.push_back(top);
    double aggregate = 0;
    ref_t refiner;
    while(true) {
        clt c = levels[levels.size() - 1];
        part_vt part("cluster assignments", c.mtx.numRows());
        Kokkos::parallel_for("set initial assignments", r_policy(0, c.mtx.numRows()), KOKKOS_LAMBDA(const ordinal_t x){
            part(x) = x;
        });
        if(levels.size() == 1) refiner.jet_refine<true, false>(c.mtx, c.wdeg, part, rfd, true, mem, part);
        else refiner.jet_refine<false, false>(c.mtx, c.wdeg, part, rfd, true, mem, part);
        part = hec_t::coarsen_HEC(c.mtx, c.wdeg, part, mem, rfd);
        parts.push_back(part);
        if(rfd.label_count < c.mtx.numRows()){
            Kokkos::Timer t;
            contracter_t contracter;
            clt next_clt;
            if(levels.size() == 1) next_clt = contracter.build_coarse_graph<true>(c, part, rfd.label_count, mem);
            else next_clt = contracter.build_coarse_graph<false>(c, part, rfd.label_count, mem);
            next_clt.wdeg = wgt_view_t("weighted degree 2", rfd.label_count);
            coarsen_vtx_w(c.wdeg, next_clt.wdeg, part);

            rfd.update(next_clt.mtx, next_clt.wdeg);

            levels.push_back(next_clt);
            aggregate += t.seconds();
        } else {
            break;
        }
    }

    // std::cout << "Aggregation time: " << aggregate << "s" << std::endl;
    experiment.addMeasurement(Measurement::Contract, aggregate);
    // std::cout << rfd.obj << std::endl;
    // std::cout << rfd.label_count << std::endl;
    
    for(int i = levels.size() - 2; i >= 0; i--){
        clt c = levels[i];
        part_vt coarse_part = parts[i + 1];
        part_vt part = parts[i];
        Kokkos::parallel_for("update top level assignments", r_policy(0, c.mtx.numRows()), KOKKOS_LAMBDA(const ordinal_t x){
            part(x) = coarse_part(part(x));
        });
        // refiner.jet_refine<false>(c.mtx, c.wdeg, part, rfd, false, mem);
    }
    return parts[0];
}

template <bool constrained>
part_vt louvain_part(mem_t& mem, clt top, rfd_t& rfd, ExperimentLoggerUtil<value_t>& experiment, vtx_vt constraint){
    std::vector<clt> levels;
    std::vector<part_vt> parts;
    levels.push_back(top);
    double aggregate = 0;
    ref_t refiner;
    while(true) {
        clt c = levels[levels.size() - 1];
        // std::cout << "Pre-refine" << std::endl;
        part_vt part("cluster assignments", c.mtx.numRows());
        Kokkos::parallel_for("set initial assignments", r_policy(0, c.mtx.numRows()), KOKKOS_LAMBDA(const ordinal_t x){
            part(x) = x;
        });
        if(levels.size() == 1) refiner.jet_refine<true, constrained>(c.mtx, c.wdeg, part, rfd, true, mem, constraint);
        else refiner.jet_refine<false, constrained>(c.mtx, c.wdeg, part, rfd, true, mem, constraint);
        parts.push_back(part);
        if(rfd.label_count < c.mtx.numRows()){
            Kokkos::Timer t;
            contracter_t contracter;
            clt next_clt;
            if(levels.size() == 1) next_clt = contracter.build_coarse_graph<true>(c, part, rfd.label_count, mem);
            else next_clt = contracter.build_coarse_graph<false>(c, part, rfd.label_count, mem);
            next_clt.wdeg = wgt_view_t("weighted degree 2", rfd.label_count);
            Kokkos::deep_copy(next_clt.wdeg, rfd.total_deg);
            levels.push_back(next_clt);

            if(constrained) {
                vtx_vt next_constraint("next constraint", rfd.label_count);
                downsample(constraint, next_constraint, part);
                constraint = next_constraint;
            }

            aggregate += t.seconds();
        } else {
            break;
        }
    }

    // std::cout << "Post coarsen obj: " << rfd.obj << std::endl;
    if(levels.size() > 1){
        // last level has the same partition as previous level
        // so refining this level on the uncoarsening pass
        // would not integrate any coarse information
        levels.pop_back();
        parts.pop_back();
    }
    
    // levels.size()-2 so that (i+1) is in bounds
    for(int i = levels.size() - 2; i >= 0; i--){
        clt c = levels[i];
        part_vt coarse_part = parts[i + 1];
        part_vt part = parts[i];
        Kokkos::parallel_for("update top level assignments", r_policy(0, c.mtx.numRows()), KOKKOS_LAMBDA(const ordinal_t x){
            part(x) = coarse_part(part(x));
        });
        if(i == 0) refiner.jet_refine<true, false>(c.mtx, c.wdeg, part, rfd, false, mem, constraint);
        else refiner.jet_refine<false, false>(c.mtx, c.wdeg, part, rfd, false, mem, constraint);
    }

    experiment.addMeasurement(Measurement::Contract, aggregate);
    // std::cout << "Post uncoarsen obj: " << rfd.obj << std::endl;
    return parts[0];
}

part_vt partition(matrix_t g,
                    wgt_view_t vweights,
                    double& mod,
                    ExperimentLoggerUtil<value_t>& experiment) {
    rfd_t rfd(g, vweights, 1.0, true);
    mem_t mem(g);
    clt c;
    c.mtx = g;
    c.wdeg = vweights;
    clt active_clt = c;
    Kokkos::fence();
    Kokkos::Timer t;
    std::cout << std::setprecision(6);
    part_vt constraint;
    part_vt part = louvain_part<false>(mem, c, rfd, experiment, constraint);
    double time = t.seconds();
    std::cout << "Cluster time: " << time << std::endl;
    std::cout << "Cut: " << rfd.cut / 2 << std::endl;
    t.reset();
    for(int i = 0; i < 5; i++){
        rfd.update(g, vweights);
        constraint = part;
        part = louvain_part<true>(mem, c, rfd, experiment, constraint);
        time = t.seconds();
        std::cout << "Cluster time: " << time << std::endl;
        std::cout << "Cut: " << rfd.cut / 2 << std::endl;
        t.reset();
    }
    experiment.setModularity(rfd.obj);
    experiment.addMeasurement(Measurement::Total, time);
    experiment.setEdgeCut(rfd.cut / 2);
    std::cout << "Modularity: " << rfd.obj << std::endl;
    // if(false){
    //     double obj = rfd.obj;
    //     do {
    //         obj = rfd.obj;
    //         // connected_comps(g, part);
    //         // c.mtx = constraint_graph(g, part, refiner.get_entries_view());
    //         Kokkos::Timer x;
    //         part = leiden_part(mem, part, c, rfd, experiment);
    //         std::cout << x.seconds() << std::endl;
    //     } while(obj < rfd.obj);
    // }
    mod = rfd.obj;
    // connected_comps(g, part);
    // ordinal_t labels = pstat::get_total_labels(part);
    // std::cout << "Total labels " << labels << std::endl;
    // double modularity = pstat::modularity(g, part, labels, g.nnz());
    // std::cout << "Modularity: " << modularity << std::endl;
    // wgt_view_t vtx_w("vertex weights", g.numRows());
    // Kokkos::deep_copy(vtx_w, 1);
    // wgt_view_t part_sizes = pstat::get_part_sizes(g, vtx_w, part, labels);
    // std::cout << "Largest part: " << static_cast<double>(pstat::largest_part_size(part_sizes)) / static_cast<double>(g.numRows()) << std::endl;
    // edge_cut = pstat::get_total_cut(g, part);
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
        std::cerr << "Usage: " << argv[0] << " <metis_graph_file> <optional partition_output_filename> <optional metrics output filename>" << std::endl;
        return -1;
    }
    char *filename = argv[1];
    char *part_file = nullptr;
    if(argc >= 3){
        part_file = argv[2];
    }
    char *metrics_file = nullptr;
    if(argc >= 4){
        metrics_file = argv[3];
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

        int iters = 21;
        part_vt best_part;
        double best_mod = -1;
        for(int i = 0; i < iters; i++){
            ExperimentLoggerUtil<value_t> experiment;
            double mod;
            part_vt part = partition(g, vweights, mod, experiment);
            if(mod > best_mod){
                best_mod = mod;
                best_part = part;
            }
            if(metrics_file != nullptr){
                experiment.log(metrics_file, i == 0, (i+1) == iters);
            }
        }

        if(part_file != nullptr) write_part(best_part, part_file);
    }
    Kokkos::finalize();

    return 0;
}
