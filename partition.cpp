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
#include <limits>
#include <queue>

using namespace jet_partitioner;
using ref_t = jet_refiner<matrix_t, part_t>;
using rfd_t = typename ref_t::refine_data;
using pstat = part_stat<matrix_t, part_t>;
using contracter_t = contracter<matrix_t>;
using clt = contracter_t::coarse_level_triple;
using mem_t = memory_store<matrix_t, part_t>;

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

part_vt rec_part(mem_t& mem, clt top, rfd_t& rfd, ExperimentLoggerUtil<value_t>& experiment){
    std::vector<clt> levels;
    std::vector<part_vt> parts;
    levels.push_back(top);
    rfd.init = false;
    double aggregate = 0;
    ref_t refiner;
    while(true) {
        clt c = levels[levels.size() - 1];
        // std::cout << "Pre-refine" << std::endl;
        part_vt part("cluster assignments", c.mtx.numRows());
        Kokkos::parallel_for("set initial assignments", r_policy(0, c.mtx.numRows()), KOKKOS_LAMBDA(const ordinal_t x){
            part(x) = x;
        });
        refiner.jet_refine(c.mtx, c.wdeg, part, rfd, true, mem, experiment);
        parts.push_back(part);
        if(rfd.label_count < c.mtx.numRows()){
            Kokkos::Timer t;
            contracter_t contracter;
            clt next_clt = contracter.build_coarse_graph(c, part, rfd.label_count, mem, experiment);
            next_clt.wdeg = wgt_view_t("weighted degree 2", rfd.label_count);
            Kokkos::deep_copy(next_clt.wdeg, rfd.total_deg);
            levels.push_back(next_clt);
            aggregate += t.seconds();
        } else {
            break;
        }
    }

    std::cout << "Aggregation time: " << aggregate << "s" << std::endl;
    std::cout << rfd.mod << std::endl;
    
    for(int i = levels.size() - 2; i >= 0; i--){
        clt c = levels[i];
        part_vt coarse_part = parts[i + 1];
        part_vt part = parts[i];
        Kokkos::parallel_for("update top level assignments", r_policy(0, c.mtx.numRows()), KOKKOS_LAMBDA(const ordinal_t x){
            part(x) = coarse_part(part(x));
        });
        refiner.jet_refine(c.mtx, c.wdeg, part, rfd, false, mem, experiment);
    }
    return parts[0];
}

part_vt rec_part(mem_t& mem, clt top, rfd_t& rfd, part_vt constraint, ExperimentLoggerUtil<value_t>& experiment){
    std::vector<clt> levels;
    std::vector<part_vt> parts;
    levels.push_back(top);
    rfd.init = false;
    double aggregate = 0;
    bool stop = false;
    ref_t refiner;
    while(true) {
        clt c = levels[levels.size() - 1];
        // std::cout << "Pre-refine" << std::endl;
        part_vt part("cluster assignments", c.mtx.numRows());
        Kokkos::parallel_for("set initial assignments", r_policy(0, c.mtx.numRows()), KOKKOS_LAMBDA(const ordinal_t x){
            part(x) = x;
        });
        matrix_t cg = constraint_graph(c.mtx, constraint, mem.cd_mem.conn_entries);
        refiner.jet_refine(cg, c.wdeg, part, rfd, true, mem, experiment);
        parts.push_back(part);
        if(rfd.label_count < c.mtx.numRows()){
            Kokkos::Timer t;
            contracter_t contracter;
            clt next_clt = contracter.build_coarse_graph(c, part, rfd.label_count, mem, experiment);
            next_clt.wdeg = wgt_view_t("weighted degree 2", rfd.label_count);
            part_vt next_constraint("active constraint", rfd.label_count);
            Kokkos::parallel_for("set initial assignments", r_policy(0, c.mtx.numRows()), KOKKOS_LAMBDA(const ordinal_t x){
                next_constraint(part(x)) = constraint(x);
            });
            constraint = next_constraint;
            Kokkos::deep_copy(next_clt.wdeg, rfd.total_deg);
            levels.push_back(next_clt);
            aggregate += t.seconds();
        } else if(!stop) {
            Kokkos::deep_copy(constraint, 0);
            parts.pop_back();
            stop = true;
            continue;
        } else {
            break;
        }
    }

    std::cout << "Aggregation time: " << aggregate << "s" << std::endl;
    std::cout << rfd.mod << std::endl;
    
    for(int i = levels.size() - 2; i >= 0; i--){
        clt c = levels[i];
        part_vt coarse_part = parts[i + 1];
        part_vt part = parts[i];
        Kokkos::parallel_for("update top level assignments", r_policy(0, c.mtx.numRows()), KOKKOS_LAMBDA(const ordinal_t x){
            part(x) = coarse_part(part(x));
        });
        refiner.jet_refine(c.mtx, c.wdeg, part, rfd, false, mem, experiment);
    }
    return parts[0];
}

part_vt partition(value_t& edge_cut,
                                  matrix_t g,
                                  wgt_view_t vweights,
                                  ExperimentLoggerUtil<value_t>& experiment) {
    rfd_t rfd;
    mem_t mem(g);
    clt c;
    c.mtx = g;
    c.wdeg = vweights;
    clt active_clt = c;
    Kokkos::fence();
    Kokkos::Timer t;
    std::cout << std::setprecision(6);
    part_vt part = rec_part(mem, c, rfd, experiment);
    std::cout << t.seconds() << std::endl;
    if(false){
        double obj = rfd.mod;
        do {
            obj = rfd.mod;
            // connected_comps(g, part);
            // c.mtx = constraint_graph(g, part, refiner.get_entries_view());
            Kokkos::Timer x;
            part = rec_part(mem, c, rfd, part, experiment);
            std::cout << x.seconds() << std::endl;
        } while(obj < rfd.mod);
    }
    Kokkos::fence();
    std::cout << t.seconds() << std::endl;
    // connected_comps(g, part);
    ordinal_t labels = pstat::get_total_labels(part);
    std::cout << "Total labels " << labels << std::endl;
    double modularity = pstat::modularity(g, part, labels, g.nnz());
    std::cout << "Modularity: " << modularity << std::endl;
    wgt_view_t vtx_w("vertex weights", g.numRows());
    Kokkos::deep_copy(vtx_w, 1);
    wgt_view_t part_sizes = pstat::get_part_sizes(g, vtx_w, part, labels);
    std::cout << "Largest part: " << static_cast<double>(pstat::largest_part_size(part_sizes)) / static_cast<double>(g.numRows()) << std::endl;
    edge_cut = pstat::get_total_cut(g, part);
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
        std::cerr << "Usage: " << argv[0] << " <metis_graph_file> <optional partition_output_filename>" << std::endl;
        return -1;
    }
    char *filename = argv[1];
    // char *part_file = nullptr;
    // if(argc >= 3){
    //     part_file = argv[2];
    // }

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

        value_t edgecut = 0;
        ExperimentLoggerUtil<value_t> experiment;
        for(int i = 0; i < 1; i++){
            part_vt part = partition(edgecut, g, vweights, experiment);
        }

        // if(part_file != nullptr) write_part(part, part_file);
    }
    Kokkos::finalize();

    return 0;
}
