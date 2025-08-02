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

    //hn is a list of vertices such that vertex i wants to aggregate with vertex hn(i)
    static ordinal_t parallel_map_construct(part_vt vcmap, const ordinal_t n, const vtx_vt hn, mem_t& mem) {

        ordinal_t perm_length = n;
        //construct mapping using heaviest edges
        ordinal_t iter = 0;
        vtx_vt curr_perm;
        while (perm_length > 0) {
            vtx_vt in_deg = Kokkos::subview(mem.s_mem.zeros1, std::make_pair(static_cast<ordinal_t>(0), n));
            Kokkos::parallel_for(policy_t(0, perm_length), KOKKOS_LAMBDA(ordinal_t i) {
                ordinal_t u = perm_length == n ? i : curr_perm(i);
                ordinal_t v = hn(u);
                if(u == v) {
                    return;
                }
                hasher_t hash;
                int condition = hash(u + iter) < hash(v + iter);
                // need to enforce an ordering condition to break cycles
                if((condition) && vcmap(v) == ORD_MAX) Kokkos::atomic_add(&in_deg(v), 1);
            });
            Kokkos::parallel_for(policy_t(0, perm_length), KOKKOS_LAMBDA(ordinal_t i) {
                ordinal_t u = perm_length == n ? i : curr_perm(i);
                if(in_deg(u) > 0){
                    in_deg(u) = 0;
                    return;
                }
                ordinal_t v = hn(u);
                if(u == v) {
                    vcmap(u) = u;
                    return;
                }
                ordinal_t cv = v < u ? v : u;
                hasher_t hash;
                int condition = hash(u + iter) < hash(v + iter);
                if(!condition) return;
                if (Kokkos::atomic_compare_exchange(&vcmap(v), ORD_MAX, ORD_MAX - 1) == ORD_MAX) {
                    vcmap(u) = cv;
                    vcmap(v) = cv;
                }
                else {
                    if (vcmap(v) < n) {
                        vcmap(u) = vcmap(v);
                    }
                    else {
                        vcmap(u) = ORD_MAX;
                    }
                }
            });
            vtx_vt vtx3 = mem.p_mem.order1;
            Kokkos::parallel_scan(policy_t(0, perm_length), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final) {
                ordinal_t u = perm_length == n ? i : curr_perm(i);
                if (vcmap(u) >= n) {
                    if(final){
                        vtx3(update) = u;
                    }
                    update++;
                }
            }, perm_length);
            vtx_vt next_perm = Kokkos::subview(vtx3, std::make_pair(static_cast<ordinal_t>(0), perm_length));
            curr_perm = Kokkos::subview(mem.s_mem.vtx2, std::make_pair(static_cast<ordinal_t>(0), perm_length));
            Kokkos::deep_copy(exec_space(), curr_perm, next_perm);
            iter++;
        }
        ordinal_t nc = 0;
        Kokkos::parallel_scan("assign aggregates", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t u, ordinal_t& update, const bool final){
            if(vcmap(u) == u){
                if(final){
                    vcmap(u) = update;
                }
                update++;
            } else if(final){
                vcmap(u) = vcmap(u) + n;
            }
        }, nc);
        Kokkos::parallel_for("propagate aggregates", policy_t(0, n), KOKKOS_LAMBDA(ordinal_t u) {
            if(vcmap(u) >= n) {
                ordinal_t c_id = vcmap(u) - n;
                vcmap(u) = vcmap(c_id);
            }
        });
        return nc;
    }

    static part_vt coarsen_HEC(const matrix_t& g,
        const wgt_vt& vtx_w,
        const part_vt& constraint,
        mem_t& mem,
        refine_data& rfd) {

        ordinal_t n = g.numRows();
        vtx_vt hn = Kokkos::subview(mem.s_mem.vtx1, std::make_pair(static_cast<ordinal_t>(0), n));
        part_vt vcmap("vcmap", n);
        Kokkos::deep_copy(exec_space(), vcmap, ORD_MAX);

        float gamma = 1.0 / static_cast<float>(rfd.g_deg);
        Kokkos::parallel_for("Heaviest HN", team_policy_t(n, Kokkos::AUTO), KOKKOS_LAMBDA(const member & thread) {
            ordinal_t i = thread.league_rank();
            edge_offset_t end = g.graph.row_map(i + 1);
            edge_offset_t start = g.graph.row_map(i);
            float multi = gamma*vtx_w(i);
            typename Kokkos::MaxLoc<float,edge_offset_t,Device>::value_type argmax{0, end};
            // get neighbor maximizing objective
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(thread, start, end), [=](const edge_offset_t idx, Kokkos::ValLocScalar<float,edge_offset_t>& local) {
                ordinal_t v = g.graph.entries(idx);
                if(constraint(i) != constraint(v)) return;
                scalar_t wgt = g.values(idx);
                float val = static_cast<float>(wgt) - multi*vtx_w(v);
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
        rfd.label_count = parallel_map_construct(vcmap, n, hn, mem);
        rfd.total_deg = wgt_vt("coarse total degree", rfd.label_count);
        wgt_vt total_deg = rfd.total_deg;
        Kokkos::parallel_for("add degree", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i){
            ordinal_t label = vcmap(i);
            Kokkos::atomic_add(&total_deg(label), vtx_w(i));
        });

        return vcmap;
    }

    
};

}
