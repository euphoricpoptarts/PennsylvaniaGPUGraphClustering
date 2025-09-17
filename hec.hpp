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
#include <random>
#include <Kokkos_Core.hpp>
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosSparse_SortCrs.hpp"
#include "Kokkos_UnorderedMap.hpp"
#include "memory_store.hpp"
#include "cluster_data.h"

namespace jet_community {

template<class crsMat, typename part_t>
class leidenR {
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
    static constexpr ordinal_t split = 1000000000;

    static void ensure_gamma_connectivity(const matrix_t g, part_vt vcmap, part_vt constraint, const vtx_vt order, const ordinal_t n, const wgt_vt wdeg, const wgt_vt total_deg, mem_t& mem, const refine_data& rfd) {

        vtx_vt row_map = Kokkos::subview(mem.p_mem.row_map, std::make_pair(static_cast<ordinal_t>(0), n + 1));
        vtx_vt store_atom = Kokkos::subview(mem.p_mem.entries, std::make_pair(static_cast<ordinal_t>(0), n));
        vtx_vt ids = Kokkos::subview(mem.s_mem.vtx1, std::make_pair(static_cast<ordinal_t>(0), n));
        vtx_vt sort_order = Kokkos::subview(mem.s_mem.vtx2, std::make_pair(static_cast<ordinal_t>(0), n));
        Kokkos::deep_copy(exec_space(), row_map, 0);

        Kokkos::parallel_for("compute cluster cardinality", policy_t(0, n), KOKKOS_LAMBDA(ordinal_t i) {
            ordinal_t c = vcmap(i);
            store_atom(i) = Kokkos::atomic_fetch_add(&row_map(c), 1);
        });

        Kokkos::parallel_scan("scan offsets", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
            ordinal_t val = row_map(i);
            if(final) row_map(i) = update;
            update += val;
            if(final && i + 1 == n){
                row_map(n) = update;
            }
        });

        Kokkos::parallel_for("insert", policy_t(0, n), KOKKOS_LAMBDA(ordinal_t i) {
            ordinal_t c = vcmap(i);
            ordinal_t insert = row_map(c) + store_atom(i);
            ids(insert) = i;
            sort_order(insert) = order(i);
        });

        // sort_order is unneeded after this
        KokkosSparse::sort_crs_matrix<exec_space, vtx_vt, vtx_vt, vtx_vt>(exec_space(), row_map, sort_order, ids);

        wgt_vt total_size = mem.p_mem.pvals;
        wgt_vt inner_conn = Kokkos::subview(mem.s_mem.vtx2, std::make_pair(static_cast<ordinal_t>(0), n));
        wgt_vt pvals = Kokkos::subview(mem.p_mem.pvals_clone, std::make_pair(static_cast<ordinal_t>(0), n));
        wgt_vt outer_conn("outer conn", n);
        // gets the total size of each cluster during construction by order of ids view
        Kokkos::parallel_scan("scan total_size", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t x, scalar_t& update, const bool final){
            ordinal_t i = ids(x);
            scalar_t val = wdeg(i);
            if(final){
                total_size(x) = update;
            }
            update += val;
        });
        float gamma = rfd.get_penalty_modifier();
        vtx_vt order1 = mem.p_mem.order1;
        ordinal_t big_begin = mem.p_mem.offset_large;
        vtx_vt small_vtx = Kokkos::subview(order1, std::make_pair(static_cast<ordinal_t>(0), big_begin));
        vtx_vt large_vtx = Kokkos::subview(order1, std::make_pair(big_begin, n));
        Kokkos::parallel_for("compute inner_conn", policy_t(0, big_begin), KOKKOS_LAMBDA(const ordinal_t x) {
            ordinal_t i = small_vtx(x);
            edge_offset_t end = g.graph.row_map(i + 1);
            edge_offset_t start = g.graph.row_map(i);
            scalar_t result = 0;
            for(edge_offset_t idx = start; idx < end; idx++) {
                ordinal_t v = g.graph.entries(idx);
                if(vcmap(i) == vcmap(v) && order(v) < order(i)) result += g.values(idx);
            }
            inner_conn(i) = result;
        });
        Kokkos::parallel_for("compute inner_conn", team_policy_t(n - big_begin, Kokkos::AUTO), KOKKOS_LAMBDA(const member & thread) {
            ordinal_t i = large_vtx(thread.league_rank());
            edge_offset_t end = g.graph.row_map(i + 1);
            edge_offset_t start = g.graph.row_map(i);
            scalar_t result = 0;
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(thread, start, end), [=](const edge_offset_t idx, scalar_t& update) {
                ordinal_t v = g.graph.entries(idx);
                if(vcmap(i) == vcmap(v) && order(v) < order(i)) update += g.values(idx);
            }, result);
            inner_conn(i) = result;
        });

        vtx_vt breakers("breakers", n);
        Kokkos::deep_copy(exec_space(), breakers, n);

        // break clusters if vertex i not well connect to cluster
        Kokkos::parallel_for("find breakpoints (part 1)", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t x){
            ordinal_t i = ids(x);
            ordinal_t c = vcmap(i);
            ordinal_t c_begin = row_map(c);
            scalar_t prev_size = total_size(x) - total_size(c_begin);
            if(inner_conn(i) < gamma*wdeg(i)*prev_size){
                // break cluster on x
                Kokkos::atomic_min(&breakers(c), x);
            }
        });

        // gets the total outward connectivity of each cluster during construction by order of ids view
        Kokkos::parallel_scan("scan outward connectivity (inclusive)", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t x, scalar_t& update, const bool final){
            ordinal_t i = ids(x);
            // *2 to account for edges previously outside cluster that are now within it
            scalar_t val = pvals(i) - 2*inner_conn(i);
            update += val;
            if(final){
                outer_conn(x) = update;
            }
        });

        // break cluster if cluster not well connected to rest of constraint cluster
        Kokkos::parallel_for("find breakpoints (part 2)", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t x){
            ordinal_t i = ids(x);
            ordinal_t c = vcmap(i);
            ordinal_t c_begin = row_map(c);
            scalar_t curr_size = total_size(x) + wdeg(i) - total_size(c_begin);
            scalar_t other_size = total_deg(constraint(i)) - curr_size;
            scalar_t outer = outer_conn(x);
            if(c_begin > 0) outer -= outer_conn(c_begin - 1);
            if(outer < gamma*curr_size*other_size){
                // break cluster after x
                Kokkos::atomic_min(&breakers(c), x + 1);
            }
        });

        Kokkos::parallel_for("break clusters", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t x){
            ordinal_t i = ids(x);
            ordinal_t c = vcmap(i);
            if(x >= breakers(c)){
                vcmap(i) = i;
            }
        });
    }

    // reassigns cluster labels into a contiguous range beginning from 0
    static ordinal_t contigitize_clusters(part_vt vcmap, const ordinal_t n) {
        ordinal_t nc = 0;
        Kokkos::parallel_scan("assign contiguous id", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t u, ordinal_t& update, const bool final){
            // every cluster with label "u" must contain vertex "u"
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
    static void find_trees(part_vt vcmap, const ordinal_t n, const vtx_vt hn, vtx_vt order) {

        // compute connected components on the forest induced by hn
        // in this kernel we ignore edges that go towards a higher ordinal vertex
        Kokkos::parallel_for("connected components part 1", policy_t(0, n), KOKKOS_LAMBDA(ordinal_t i) {
            ordinal_t now = i;
            ordinal_t then = hn(i);

            if(now == then){
                vcmap(i) = i;
                return;
            }
            
            // pointer-chasing towards lowest order
            while(order(now) > order(then)){
                now = then;
                then = hn(then);
            }

            if(now != i){
                vcmap(i) = now;
                // other vertices in path will get here, except for now
                vcmap(now) = now;
            }
            
        });

        // flip the ordering for remaining unclustered vertices
        // this also makes them higher ordered then all previously clustered vertices
        Kokkos::parallel_for("update order", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i){
            if(vcmap(i) == ORD_MAX) order(i) = 2*split - order(i);
        });

        // in this kernel we consider the edges ignored by the previous kernel
        // in order to connect vertices left unassigned by previous kernel
        Kokkos::parallel_for("connected components part 2", policy_t(0, n), KOKKOS_LAMBDA(ordinal_t i) {
            if(vcmap(i) != ORD_MAX) return;
            ordinal_t now = i;
            ordinal_t then = hn(i);

            while(vcmap(now) == ORD_MAX && order(now) > order(then)){
                now = then;
                then = hn(then);
            }

            vcmap(i) = vcmap(now);
        });
    }

    template <bool uniform>
    static part_vt coarsen_leidenR(const matrix_t& g,
        const wgt_vt& wdeg,
        const part_vt& constraint,
        mem_t& mem,
        const refine_data& rfd,
        int& coarse_vtx_count) {

        ordinal_t n = g.numRows();
        float gamma = rfd.get_penalty_modifier();
        vtx_vt hn = Kokkos::subview(mem.s_mem.vtx1, std::make_pair(static_cast<ordinal_t>(0), n));
        part_vt vcmap("vcmap", n);
        Kokkos::deep_copy(exec_space(), vcmap, ORD_MAX);
        vtx_vt order("order", n);

        vtx_vt well_conn("well connected", n);
        const wgt_vt pvals = Kokkos::subview(mem.p_mem.pvals_clone, std::make_pair(static_cast<ordinal_t>(0), n));
        const wgt_vt total_deg = rfd.total_deg;
        Kokkos::parallel_for("determine well connected", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i){
            ordinal_t c = constraint(i);
            scalar_t wd = wdeg(i);
            // is vertex 'i' well connected to the rest of constraint cluster 'c'?
            if(pvals(i) >= gamma*wd*(total_deg(c) - wd)) well_conn(i) = 1;
            else well_conn(i) = 0;
        });

        std::random_device rd;
        if(uniform){
            // randomizations is only strictly necessary on top level graph
            // randomization on successive levels is empirically detrimental to objective quality
            ordinal_t seed = rd();
            Kokkos::parallel_for("Heaviest HN", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i) {
                edge_offset_t end = g.graph.row_map(i + 1);
                edge_offset_t start = g.graph.row_map(i);
                ordinal_t width = end - start;
                float multi = gamma*wdeg(i);
                hasher_t hash;
                ordinal_t jx = hash(seed + i);
                // 0.001 chance to self-aggregate
                if(jx % 1000 == 0 || well_conn(i) == 0){
                    hn(i) = i;
                    return;
                }
                jx = start + Kokkos::abs(jx % width);
                // choose random satisfactory neighbor
                for(edge_offset_t j = jx; j < end; j++){
                    ordinal_t v = g.graph.entries(j);
                    if(constraint(i) != constraint(v)) continue;
                    if(well_conn(v) == 0) continue;
                    scalar_t wgt = g.values(j);
                    if(wgt >= multi*wdeg(v)){
                        hn(i) = v;
                        return;
                    }
                }
                for(edge_offset_t j = start; j < jx; j++){
                    ordinal_t v = g.graph.entries(j);
                    if(constraint(i) != constraint(v)) continue;
                    if(well_conn(v) == 0) continue;
                    scalar_t wgt = g.values(j);
                    if(wgt >= multi*wdeg(v)){
                        hn(i) = v;
                        return;
                    }
                }
                hn(i) = i;
            });
        } else {
            Kokkos::parallel_for("Heaviest HN", team_policy_t(n, Kokkos::AUTO), KOKKOS_LAMBDA(const member & thread) {
                ordinal_t i = thread.league_rank();
                edge_offset_t end = g.graph.row_map(i + 1);
                edge_offset_t start = g.graph.row_map(i);
                float multi = gamma*wdeg(i);
                if(well_conn(i) == 0){
                    hn(i) = i;
                    return;
                }
                typename Kokkos::MaxLoc<float,edge_offset_t,Device>::value_type argmax{0, end};
                // get neighbor maximizing objective
                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(thread, start, end), [=](const edge_offset_t idx, Kokkos::ValLocScalar<float,edge_offset_t>& local) {
                    ordinal_t v = g.graph.entries(idx);
                    if(constraint(i) != constraint(v)) return;
                    if(well_conn(v) == 0) return;
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
                        hn(i) = h;
                    } else {
                        hn(i) = i;
                    }
                });
            });
        }
        ordinal_t seed = rd();
        // this random ordering determines the spanning trees
        // and the insertion order into each cluster
        Kokkos::parallel_for("set order", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i){
            hasher_t hash;
            ordinal_t o = hash(i + seed);
            order(i) = Kokkos::abs(o % split);
        });
        Kokkos::parallel_for("unselect equal order edges", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i){
            ordinal_t h = hn(i);
            // if two vertices have the same ordering and one points to the other
            // just unselect that edge
            // sidenote: if two vertices have the same ordering but neither points to the other
            // than the sort which comes later will be the tie-breaker
            if(order(i) == order(h)) hn(i) = i;
        });
        find_trees(vcmap, n, hn, order);
        ensure_gamma_connectivity(g, vcmap, constraint, order, n, wdeg, rfd.total_deg, mem, rfd);
        coarse_vtx_count = contigitize_clusters(vcmap, n);

        return vcmap;
    }

    
};

}
