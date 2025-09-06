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
#pragma once
#include <limits>
#include <Kokkos_Core.hpp>
#include "KokkosSparse_CrsMatrix.hpp"
#include "Kokkos_UnorderedMap.hpp"
#include "memory_store.hpp"
#include "cluster_data.h"

namespace jet_community {

template<class crsMat, typename part_t>
class coarsen_heuristics {
public:
    // define internal types
    using matrix_t = crsMat;
    using exec_space = typename matrix_t::execution_space;
    using Device = typename matrix_t::device_type;
    using ordinal_t = typename matrix_t::ordinal_type;
    using edge_offset_t = typename matrix_t::size_type;
    using scalar_t = typename matrix_t::value_type;
    using vtx_vt = typename Kokkos::View<ordinal_t*, Device>;
    using wgt_vt = typename Kokkos::View<scalar_t*, Device>;
    using part_vt = typename Kokkos::View<part_t*, Device>;
    using policy_t = typename Kokkos::RangePolicy<exec_space>;
    using team_policy_t = typename Kokkos::TeamPolicy<exec_space>;
    using member = typename team_policy_t::member_type;
    using mem_t = memory_store<matrix_t>;
    using refine_data = cluster_data<matrix_t>;
    // there is a problem edge-case in kokkos with MaxLoc that can be triggered rarely for any input graph
    // the problem will be fixed soon, use MaxFirstLoc in meantime
    using argmax_reducer_t = Kokkos::MaxFirstLoc<uint32_t, edge_offset_t, Device>;
    using argmax_t = typename argmax_reducer_t::value_type;
    using hasher_t = Kokkos::pod_hash<ordinal_t>;
    static constexpr ordinal_t ORD_MAX = std::numeric_limits<ordinal_t>::max();
    static constexpr bool is_host_space = std::is_same<typename exec_space::memory_space, typename Kokkos::DefaultHostExecutionSpace::memory_space>::value;

    static void ensure_gamma_connectivity(part_vt vcmap, const ordinal_t n, const vtx_vt hn, const wgt_vt wdeg, mem_t& mem, const refine_data& rfd) {

        vtx_vt dist = Kokkos::subview(mem.s_mem.vtx2, std::make_pair(static_cast<ordinal_t>(0), n));
        Kokkos::deep_copy(exec_space(), dist, 0);

        Kokkos::parallel_for("find dist", policy_t(0, n), KOKKOS_LAMBDA(ordinal_t i) {
            int d = 0;

            ordinal_t now = i;

            while(vcmap(now) != now){
                now = hn(now);
                d++;
            }

            if(now != i) dist(i) = d;
            Kokkos::atomic_max(&dist(now), d);
        });

        // create a contiguous range where we can lookup the size of the BFS prior to each level
        ordinal_t t_dist = 0;
        Kokkos::parallel_scan("scan dists", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
            if(vcmap(i) == i){
                ordinal_t val = dist(i) + 1;
                if(final){
                    dist(i) = update;
                }
                update += val;
            }
        }, t_dist);

        wgt_vt penalty = mem.p_mem.pvals;
        wgt_vt total_size = Kokkos::subview(mem.p_mem.pvals_clone, std::make_pair(static_cast<ordinal_t>(0), t_dist));
        Kokkos::deep_copy(exec_space(), total_size, 0);
        Kokkos::parallel_for("add wdeg", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i){
            int d = dist(i);
            ordinal_t c = vcmap(i);
            if(c == i) d = 0;
            int offset = dist(c) + d;
            penalty(i) = Kokkos::atomic_fetch_add(&total_size(offset), wdeg(i));
        });
        Kokkos::parallel_scan("scan total_size", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, scalar_t& update, const bool final){
            scalar_t val = total_size(i);
            if(final){
                total_size(i) = update;
            }
            update += val;
        });
        float gamma = rfd.get_penalty_modifier();
        auto score = mem.p_mem.obj_persistent;
        Kokkos::parallel_for("add wdeg", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i){
            int d = dist(i);
            ordinal_t c = vcmap(i);
            if(c == i) d = 0;
            int offset = dist(c);
            scalar_t prior_penalty = total_size(offset + d) - total_size(offset);
            // ordering by penalty within each tree now represents a pseudo BFS
            scalar_t total_penalty = penalty(i) + prior_penalty;
            float scaled_penalty = gamma*wdeg(i)*total_penalty;
            if(score(i) < scaled_penalty) {
                // break the tree here
                vcmap(i) = i;
            }
        });
        Kokkos::parallel_for("find new root", policy_t(0, n), KOKKOS_LAMBDA(ordinal_t i) {
            ordinal_t now = i;

            while(vcmap(now) != now){
                now = hn(now);
            }

            if(now != i) vcmap(i) = vcmap(now);
        });
    }

    static ordinal_t contigitize_clusters(part_vt vcmap, const ordinal_t n) {
        ordinal_t nc = 0;
        Kokkos::parallel_scan("assign contiguous id", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t u, ordinal_t& update, const bool final){
            if(vcmap(u) == u){
                if(final){
                    vcmap(u) = update;
                }
                update++;
            } else if(final){
                vcmap(u) = vcmap(u) + n;
            }
        }, nc);
        Kokkos::parallel_for("propagate ids", policy_t(0, n), KOKKOS_LAMBDA(ordinal_t u) {
            if(vcmap(u) >= n) {
                ordinal_t c_id = vcmap(u) - n;
                vcmap(u) = vcmap(c_id);
            }
        });
        return nc;
    }

    //hn is a list of vertices such that vertex i wants to aggregate with vertex hn(i)
    static void find_trees(part_vt vcmap, const ordinal_t n, const vtx_vt hn) {

        // compute connected components on the graph induced by hn
        // in this kernel we ignore edges that go towards a higher ordinal vertex
        Kokkos::parallel_for("connected components part 1", policy_t(0, n), KOKKOS_LAMBDA(ordinal_t i) {
            ordinal_t now = i;
            ordinal_t then = hn(i);

            if(now == then){
                vcmap(i) = i;
                return;
            }
            
            // pointer-chasing
            hasher_t hash;
            while(hash(now) > hash(then)){
                now = then;
                then = hn(then);
            }

            if(now != i){
                vcmap(i) = now;
                // other vertices in path will get here, except for now
                vcmap(now) = now;
            }
            
        });

        // in this kernel we consider the edges ignored by the previous kernel
        // in order to connect vertices left unassigned by previous kernel
        Kokkos::parallel_for("connected components part 2", policy_t(0, n), KOKKOS_LAMBDA(ordinal_t i) {
            if(vcmap(i) != ORD_MAX) return;
            ordinal_t now = i;
            ordinal_t then = hn(i);

            // pointer-chasing in the opposite direction
            hasher_t hash;
            while(vcmap(now) == ORD_MAX && hash(now) < hash(then)){
                now = then;
                then = hn(then);
            }

            vcmap(i) = vcmap(now);
        });
    }

    static part_vt coarsen_HEC(const matrix_t& g,
        const wgt_vt& wdeg,
        const part_vt& constraint,
        mem_t& mem,
        const refine_data& rfd,
        int& coarse_vtx_count) {

        ordinal_t n = g.numRows();
        vtx_vt hn = Kokkos::subview(mem.s_mem.vtx1, std::make_pair(static_cast<ordinal_t>(0), n));
        part_vt vcmap("vcmap", n);
        Kokkos::deep_copy(exec_space(), vcmap, ORD_MAX);

        float gamma = rfd.get_penalty_modifier();
        auto score = mem.p_mem.obj_persistent;
        Kokkos::parallel_for("Heaviest HN", team_policy_t(n, Kokkos::AUTO), KOKKOS_LAMBDA(const member & thread) {
            ordinal_t i = thread.league_rank();
            edge_offset_t end = g.graph.row_map(i + 1);
            edge_offset_t start = g.graph.row_map(i);
            float multi = gamma*wdeg(i);
            typename Kokkos::MaxLoc<float,edge_offset_t,Device>::value_type argmax{0, end};
            // get neighbor maximizing objective
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(thread, start, end), [=](const edge_offset_t idx, Kokkos::ValLocScalar<float,edge_offset_t>& local) {
                ordinal_t v = g.graph.entries(idx);
                if(constraint(i) != constraint(v)) return;
                scalar_t wgt = g.values(idx);
                float val = static_cast<float>(wgt) - multi*wdeg(v);
                if(val >= local.val){
                    local.val = val;
                    local.loc = idx;
                }
            
            }, Kokkos::MaxLoc<float, edge_offset_t,Device>(argmax));
            Kokkos::single(Kokkos::PerTeam(thread), [=](){
                if(argmax.loc >= start && argmax.loc < end && argmax.val >= 0){
                    ordinal_t h = g.graph.entries(argmax.loc);
                    score(i) = g.values(argmax.loc);
                    hn(i) = h;
                } else {
                    hn(i) = i;
                    score(i) = 0;
                }
            });
        });
        find_trees(vcmap, n, hn);
        ensure_gamma_connectivity(vcmap, n, hn, wdeg, mem, rfd);
        coarse_vtx_count = contigitize_clusters(vcmap, n);

        return vcmap;
    }

    
};

}
