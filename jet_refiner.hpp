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
#include <type_traits>
#include <limits>
#include <iostream>
#include <iomanip>
#include <Kokkos_Core.hpp>
#include "KokkosSparse_CrsMatrix.hpp"
#include "ExperimentLoggerUtil.hpp"
#include "memory_store.hpp"
#include "part_stat.hpp"

namespace jet_community {

template<class crsMat, typename part_t>
class jet_refiner {
public:

    //helper for getting gain_t
    template<typename T>
    struct type_identity {
        typedef T type;
    };

    // define internal types
    using matrix_t = crsMat;
    using exec_space = typename matrix_t::execution_space;
    using mem_space = typename matrix_t::memory_space;
    using Device = typename matrix_t::device_type;
    using ordinal_t = typename matrix_t::ordinal_type;
    using edge_offset_t = typename matrix_t::size_type;
    using scalar_t = typename matrix_t::value_type;
    // need some trickery because make_signed is undefined for floating point types
    using gain_t = typename std::conditional_t<std::is_signed_v<scalar_t>, type_identity<scalar_t>, std::make_signed<scalar_t>>::type;
    using vtx_view_t = Kokkos::View<ordinal_t*, Device>;
    using vtx_svt = Kokkos::View<ordinal_t, Device>;
    using wgt_view_t = Kokkos::View<scalar_t*, Device>;
    using edge_view_t = Kokkos::View<edge_offset_t*, Device>;
    using gain_vt = Kokkos::View<gain_t*, Device>;
    using gain_svt = Kokkos::View<gain_t, Device>;
    using vtx_pin_st = Kokkos::View<ordinal_t, Kokkos::SharedHostPinnedSpace>;
    using gain_pin_vt = Kokkos::View<gain_t*, Kokkos::SharedHostPinnedSpace>;
    using gain_pin_st = Kokkos::View<gain_t, Kokkos::SharedHostPinnedSpace>;
    using part_vt = Kokkos::View<part_t*, Device>;
    using part_svt = Kokkos::View<part_t, Device>;
    using obj_vt = Kokkos::View<float*, Device>;
    using edge_subview_t = Kokkos::View<edge_offset_t, Device>;
    using policy_t = Kokkos::RangePolicy<exec_space>;
    using team_policy_t = Kokkos::TeamPolicy<exec_space>;
    using dyn_policy_t = Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>, exec_space>;
    using dyn_team_policy_t = Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>, exec_space>;
    using member = typename team_policy_t::member_type;
    using stat = part_stat<matrix_t, part_t>;
    using refine_data = typename stat::refine_data;
    using argmax_reducer_t = Kokkos::MaxFirstLoc<float, edge_offset_t, Device>;
    using argmax_t = typename argmax_reducer_t::value_type;
    using mem_t = memory_store<matrix_t, part_t>;
    using cdata_t = typename mem_t::conn_data;
    static constexpr ordinal_t ORD_MAX = std::numeric_limits<ordinal_t>::max();
    static constexpr gain_t GAIN_MIN = std::numeric_limits<gain_t>::lowest();
    static constexpr bool is_host_space = std::is_same<typename exec_space::memory_space, typename Kokkos::DefaultHostExecutionSpace::memory_space>::value;
    static constexpr part_t NULL_PART = -1;
    static constexpr part_t HASH_RECLAIM = -2;
    static constexpr part_t NO_MOVE = -3;

    static KOKKOS_INLINE_FUNCTION uint32_t hash(uint32_t x) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        return x;
    }

struct problem {
    matrix_t g;
    wgt_view_t vtx_w;
    wgt_view_t wdeg;
    bool use_team = true;
};

void copy_refine_data(refine_data& lhs, refine_data& rhs){
    Kokkos::deep_copy(exec_space(), lhs.total_deg, rhs.total_deg);
    lhs.g_deg = rhs.g_deg;
    lhs.cut = rhs.cut;
    lhs.init = rhs.init;
    lhs.mod = rhs.mod;
    lhs.label_count = rhs.label_count;
}

refine_data clone_refine_data(refine_data& rhs){
    refine_data clone;
    clone.total_deg = gain_vt(Kokkos::ViewAllocateWithoutInitializing("total degree of clusters"), rhs.total_deg.extent(0));
    copy_refine_data(clone, rhs);
    return clone;
}

void relabel_contiguously(part_vt labels, refine_data& rfd, mem_t& mem){
	ordinal_t n = labels.extent(0);
    ordinal_t initial_count = rfd.label_count;
	vtx_view_t used = Kokkos::subview(mem.s_mem.vtx1, std::make_pair((ordinal_t)0, initial_count));
    Kokkos::deep_copy(exec_space(), used, 0);
    // some vertices can have zero degree so some labels can have zero total degree
    // therefore we can't use rfd.total_deg to determine which labels are in use
	Kokkos::parallel_for("mark labels", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i){
		used(labels(i)) = 1;
	});
    ordinal_t t_labels = 0;
	Kokkos::parallel_scan("count labels", policy_t(0, initial_count), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
		if(used(i) > 0){
			if(final) used(i) = update;
			update++;
		}
	}, t_labels);
	Kokkos::parallel_for("relabel", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i){
		labels(i) = used(labels(i));
	});
    gain_vt total_deg("new total degree", t_labels);
    Kokkos::parallel_for("relabel degrees", policy_t(0, initial_count), KOKKOS_LAMBDA(const ordinal_t i){
		if(rfd.total_deg(i) > 0){
            ordinal_t relabeled = used(i);
            total_deg(relabeled) = rfd.total_deg(i);
        }
	});
    rfd.total_deg = total_deg;
    rfd.label_count = t_labels;
}

//determines which vertices (if any) should be moved to another part to decrease cutsize
//8 kernels, 2 device-host syncs
template <bool uniform>
vtx_view_t jet_lp(const problem& prob, const matrix_t& c_graph, const part_vt& part, const refine_data& rfd, mem_t& mem, float filter_ratio, bool skip_lock){
    const matrix_t& g = prob.g;
    ordinal_t n = g.numRows();
    ordinal_t num_pos = 0;
    part_vt dest_part = mem.p_mem.dest_part;
    obj_vt save_gains = mem.p_mem.gain_persistent;
    vtx_view_t lock_bit = mem.p_mem.lock_bit;
    gain_vt total_deg = rfd.total_deg;
    gain_vt wdeg = prob.wdeg;
    part_vt dest_cache = mem.p_mem.dest_cache;
    cdata_t& cdata = mem.cd_mem;
    part_vt conn_table_sizes = cdata.conn_table_sizes;
    gain_vt pvals = mem.p_mem.pvals;
    float inv_2m = 1.0 / static_cast<float>(rfd.g_deg);
    ordinal_t cutoff = 128;
    vtx_view_t vtx1 = mem.s_mem.vtx1;
    vtx_view_t vtx2 = mem.s_mem.vtx2;
    vtx_view_t vtx3 = mem.s_mem.vtx3;
    Kokkos::parallel_scan("filter out locked and find large tables", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
        part_t cache = dest_cache(i);
        if(cache == NULL_PART && lock_bit(i) == 0 && conn_table_sizes(i) > cutoff){
            if(final){
                vtx2(update) = i;
            }
            update++;
        } else if(final){
            if(lock_bit(i)) {
                dest_part(i) = NO_MOVE;
            } else if(cache != NULL_PART) {
                dest_part(i) = cache;
            }
        }
    }, num_pos);
    vtx_view_t large_tables = Kokkos::subview(vtx2, std::make_pair(static_cast<ordinal_t>(0), num_pos));
    Kokkos::parallel_for("select destination part (small tables)", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i){
        if(!(dest_cache(i) == NULL_PART && lock_bit(i) == 0) || conn_table_sizes(i) > cutoff){
            return;
        }
        part_t best = NO_MOVE;
        float wd = wdeg(i);
        float multi = wd*inv_2m;
        part_t p = part(i);
        float p_conn = pvals(i) - (total_deg(p) - wd)*multi;
        // b_conn must be at least this value to pass filter
        float b_conn = p_conn - filter_ratio*(p_conn);
        edge_offset_t start = c_graph.graph.row_map(i);
        edge_offset_t end = c_graph.graph.row_map(i+1);
        //finds potential destination as most connected part excluding p
        for(edge_offset_t j = start; j < end; j++){
            gain_t j_val = c_graph.values(j);
            if(j_val > 0 && j_val >= b_conn){
                part_t px = c_graph.graph.entries(j);
                float j_conn = j_val - static_cast<float>(total_deg(px))*multi;
                if(j_conn >= b_conn){
                    b_conn = j_conn;
                    best = px;
                }
            }
        }
        float gain = 0;
        if(best != NO_MOVE){
            // vertices must pass this filter in order to be considered further
            gain = b_conn - p_conn;
        }
        save_gains(i) = gain;
        dest_cache(i) = best;
        //a vertex is not considered further if best == p
        dest_part(i) = best;
    });
    Kokkos::parallel_for("select destination part (large tables)", team_policy_t(num_pos, Kokkos::AUTO), KOKKOS_LAMBDA(const member& t){
        ordinal_t i = large_tables(t.league_rank());
        float wd = wdeg(i);
        float multi = wd*inv_2m;
        edge_offset_t start = c_graph.graph.row_map(i);
        edge_offset_t end = c_graph.graph.row_map(i+1);
        part_t p = part(i);
        float p_conn = pvals(i) - (total_deg(p) - wd)*multi;
        // b_conn must be at least this value to pass filter
        float limit = p_conn - filter_ratio*(p_conn);
        argmax_t am{-100000.0, end};
        //finds potential destination as most connected part excluding p
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, start, end), [=](const edge_offset_t j, argmax_t& local){
            gain_t j_val = c_graph.values(j);
            if(j_val > 0 && j_val >= limit && j_val > local.val){
                part_t px = c_graph.graph.entries(j);
                float j_conn = j_val - static_cast<float>(total_deg(px))*multi;
                if(j_conn > local.val && j_conn >= limit){
                    local.val = j_conn;
                    local.loc = j;
                }
            }
        }, argmax_reducer_t(am));
        Kokkos::single(Kokkos::PerTeam(t), [=](){
            float gain = 0;
            part_t best = NO_MOVE;
            if(am.loc >= start && am.loc < end){
                float b_conn = am.val;
                best = c_graph.graph.entries(am.loc);
                gain = b_conn - p_conn;
            }
            save_gains(i) = gain;
            dest_cache(i) = best;
            //a vertex is not considered further if best == p
            dest_part(i) = best;
        });
    });
    //need to store the pre-afterburn gains into a separate view
    //than savegains, because we write new values into it that may not be overwritten
    //if a vertex has its best neighbor cached
    obj_vt pregain = mem.s_mem.obj1;
    //write all unlocked vertices that passed the above filter into an unordered list
    //output count of such vertices into num_pos
    vtx_view_t swap_scratch = vtx3;
    Kokkos::parallel_scan("filter potentially viable moves", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
        part_t best = dest_part(i);
        if(best != NO_MOVE && lock_bit(i) == 0){
            if(final){
                swap_scratch(update) = i;
                pregain(i) = save_gains(i);
            }
            update++;
        } else if(final){
            pregain(i) = -100000.0;
            lock_bit(i) = 0;
        }
    }, mem.s_mem.scan_host);
    exec_space().fence();
    num_pos = mem.s_mem.scan_host();
    //truncate scratch views by num_pos
    vtx_view_t pos_moves = Kokkos::subview(swap_scratch, std::make_pair(static_cast<ordinal_t>(0), num_pos));
    //in this kernel every potential move from the previous filters
    //is reevaluated by considering the effect of the other potential moves
    //a move is considered to occur before another according to their potential gains
    //and the vertex ids
    ordinal_t big = 0;
    cutoff = 128;
    Kokkos::parallel_scan("filter out locked and find large tables", policy_t(0, num_pos), KOKKOS_LAMBDA(const ordinal_t x, ordinal_t& update, const bool final){
        ordinal_t i = pos_moves(x);
        if(g.graph.row_map(i+1) - g.graph.row_map(i) > cutoff){
            if(final){
                vtx1(update) = i;
            }
            update++;
        } else if(final){
            vtx1(n - 1 + update - x) = i;
        }
    }, big);
    vtx_view_t big_rows = Kokkos::subview(vtx1, std::make_pair(static_cast<ordinal_t>(0), big));
    vtx_view_t small_rows = Kokkos::subview(vtx1, std::make_pair(n - (num_pos - big), n));
    float eps = 0.1;
    Kokkos::parallel_for("afterburner heuristic", team_policy_t(big, 256), KOKKOS_LAMBDA(const member& t){
        float change = 0;
        ordinal_t i = big_rows(t.league_rank());
        part_t best = dest_part(i);
        part_t p = part(i);
        float wd = prob.wdeg(i);
        float multi = wd*inv_2m;
        float igain = pregain(i);
        ordinal_t hi = hash(i);
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [&](const edge_offset_t j, float& update){
            ordinal_t v = g.graph.entries(j);
            float vgain = pregain(v);
            //adjust local gain if v has higher priority than i
            if((vgain - igain) >= eps || (abs(vgain - igain) < eps && static_cast<ordinal_t>(hash(v)) < hi)){
                part_t vpart = dest_part(v);
                scalar_t wgt;
                if constexpr(uniform) wgt = 1;
                else wgt = g.values(j);
                float q = static_cast<float>(wgt) - multi*prob.wdeg(v);
                update -= (vpart == p) ? q : 0;
                update += (vpart == best) ? q : 0;
                vpart = part(v);
                update += (vpart == p) ? q : 0;
                update -= (vpart == best) ? q : 0;
            }
        }, change);
        Kokkos::single(Kokkos::PerTeam(t), [&](){
            if(igain + change >= 0){
                lock_bit(i) = 1;
            }
        });
    });
    Kokkos::parallel_for("afterburner heuristic", policy_t(0, num_pos - big), KOKKOS_LAMBDA(const ordinal_t& x){
        float change = 0;
        ordinal_t i = small_rows(x);
        part_t best = dest_part(i);
        part_t p = part(i);
        float wd = prob.wdeg(i);
        float multi = wd*inv_2m;
        float igain = pregain(i);
        ordinal_t hi = hash(i);
        for(edge_offset_t j = g.graph.row_map(i); j < g.graph.row_map(i + 1); j++){
            ordinal_t v = g.graph.entries(j);
            float vgain = pregain(v);
            //adjust local gain if v has higher priority than i
            if((vgain - igain) >= eps || (abs(vgain - igain) < eps && static_cast<ordinal_t>(hash(v)) < hi)){
                part_t vpart = dest_part(v);
                scalar_t wgt;
                if constexpr(uniform) wgt = 1;
                else wgt = g.values(j);
                float q = static_cast<float>(wgt) - multi*prob.wdeg(v);
                change -= (vpart == p) ? q : 0;
                change += (vpart == best) ? q : 0;
                vpart = part(v);
                change += (vpart == p) ? q : 0;
                change -= (vpart == best) ? q : 0;
            }
        }
        if(igain + change >= 0){
            lock_bit(i) = 1;
        }
    });
    vtx_view_t swaps2 = Kokkos::subview(vtx2, std::make_pair(static_cast<ordinal_t>(0), num_pos));
    //scan all vertices that passed the post filter
    Kokkos::parallel_scan("filter beneficial moves", policy_t(0, num_pos), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
        if(lock_bit(pos_moves(i))){
            if(final){
                swaps2(update) = pos_moves(i);
                // don't maintain locks in coarsening phase
                if(skip_lock) lock_bit(pos_moves(i)) = 0;
            }
            update++;
        }
    }, mem.s_mem.scan_host);
    exec_space().fence();
    num_pos = mem.s_mem.scan_host();
    pos_moves = Kokkos::subview(swaps2, std::make_pair(static_cast<ordinal_t>(0), num_pos));
    return pos_moves;
}

// updates datastructures assuming a "large" number of vertices are moved
template <bool uniform>
void update_large(const problem& prob, const part_vt part, const vtx_view_t swaps, mem_t& mem){
    const matrix_t& g = prob.g;
    ordinal_t total_moves = swaps.extent(0);
    vtx_view_t swap_bit = mem.s_mem.zeros1;
    cdata_t& cdata = mem.cd_mem;
    ordinal_t total = 0;
    ordinal_t cutoff = 32;
    vtx_view_t vtx1 = mem.s_mem.vtx1;
    vtx_view_t dest_cache = mem.p_mem.dest_cache;
    Kokkos::parallel_for("mark", policy_t(0, total_moves), KOKKOS_LAMBDA(const ordinal_t x){
        ordinal_t i = swaps(x);
        swap_bit(i) = 1;
    });
    // usually, but not always, faster than directly marking adjacencies of moved vertices, since we can break out of the loop
    // also can't use a team here because we want to be able to exit the loop early
    Kokkos::parallel_for("check adjacent", policy_t(0, g.numRows()), KOKKOS_LAMBDA(const ordinal_t i){
        if(swap_bit(i) == 1) return;
        //mark adjacent vertices
        edge_offset_t limit = g.graph.row_map(i) + cutoff;
        if(g.graph.row_map(i+1) < limit) limit = g.graph.row_map(i+1);
        for(edge_offset_t j = g.graph.row_map(i); j < limit; j++){
            ordinal_t v = g.graph.entries(j);
            if(swap_bit(v) == 1){
                swap_bit(i) = 2;
                break;
            }
        }
    });
    Kokkos::parallel_scan("collect vtx to be checked", policy_t(0, g.numRows()), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
        if(swap_bit(i) == 0){
            ordinal_t degree = g.graph.row_map(i+1) - g.graph.row_map(i);
            if(degree >= cutoff){
                if(final){
                    vtx1(update) = i;
                }
                update++;
            }
        }
    }, total);
    Kokkos::parallel_for("check adjacent (large rows)", team_policy_t(total, Kokkos::AUTO), KOKKOS_LAMBDA(const member& t){
        //mark adjacent vertices
        ordinal_t marked = 0;
        ordinal_t i = vtx1(t.league_rank());
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i+1)), [=](const edge_offset_t j, ordinal_t& update){
            if(update == 0){
                ordinal_t v = g.graph.entries(j);
                if(swap_bit(v) == 1){
                    update++;
                }
            }
        }, marked);
        if(marked > 0){
            swap_bit(i) = 2;
        }
    });
    Kokkos::parallel_scan("collect vtx to be updated", policy_t(0, g.numRows()), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
        if(swap_bit(i)){
            ordinal_t degree = g.graph.row_map(i+1) - g.graph.row_map(i);
            if(degree >= cutoff){
                if(final){
                    vtx1(update) = i;
                    // reset to zero for next iteration
                    swap_bit(i) = 0;
                    dest_cache(i) = NULL_PART;
                }
                update++;
            }
        }
    }, total);
    vtx_view_t affected = Kokkos::subview(vtx1, std::make_pair(static_cast<ordinal_t>(0), total));
    gain_vt pvals = mem.p_mem.pvals;
    int max_size = 512;
    //recompute conn tables for each vertex adjacent to a moved vertex
    Kokkos::parallel_for("reset conn DS", team_policy_t(total, Kokkos::AUTO).set_scratch_size(0, Kokkos::PerTeam(max_size*sizeof(gain_t) + max_size*sizeof(part_t))), KOKKOS_LAMBDA(const member& t){
        const ordinal_t i = affected(t.league_rank());
        edge_offset_t g_start = cdata.conn_offsets(i);
        edge_offset_t g_end = cdata.conn_offsets(i + 1);
        part_t size = g_end - g_start;
        part_t* s_conn_entries;
        gain_t* s_conn_vals;
        if(size < max_size){
            s_conn_entries = (part_t*) t.team_shmem().get_shmem(sizeof(part_t) * size);
            s_conn_vals = (gain_t*) t.team_shmem().get_shmem(sizeof(gain_t) * size);
        } else {
            s_conn_entries = cdata.conn_entries.data() + g_start;
            s_conn_vals = cdata.conn_vals.data() + g_start;
        }
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 0, size), [&] (const edge_offset_t& j) {
            s_conn_entries[j] = NULL_PART;
            s_conn_vals[j] = 0;
        });
        part_t p_i = part(i);
        t.team_barrier();
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [&] (const edge_offset_t& j, gain_t& update){
            ordinal_t v = g.graph.entries(j);
            gain_t wgt;
            if constexpr(uniform) wgt = 1;
            else wgt = g.values(j);
            part_t p = part(v);
            if(p == p_i){
                update += wgt;
                return;
            }
            part_t p_o = hash(p) % static_cast<uint32_t>(size);
            bool success = false;
            while(!success){
                part_t px = s_conn_entries[p_o];
                while(px != p && px != NULL_PART){
                    p_o = (p_o + 1) % size;
                    px = s_conn_entries[p_o];
                }
                if(px == p){
                    success = true;
                } else {
                    px = Kokkos::atomic_compare_exchange(s_conn_entries + p_o, NULL_PART, p);
                    if(px == p || px == NULL_PART){
                        success = true;
                    } else {
                        p_o = (p_o + 1) % size;
                    }
                }
            }
            Kokkos::atomic_add(s_conn_vals + p_o, wgt);
        }, pvals(i));
        if(size < max_size){
            Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g_start, g_end), [&] (const edge_offset_t& j) {
                cdata.conn_entries(j) = s_conn_entries[j - g_start];
                cdata.conn_vals(j) = s_conn_vals[j - g_start];
            });
        }
    });
    Kokkos::parallel_scan("collect vtx to be updated", policy_t(0, g.numRows()), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
        if(swap_bit(i)){
            ordinal_t degree = g.graph.row_map(i+1) - g.graph.row_map(i);
            if(degree < cutoff){
                if(final){
                    vtx1(update) = i;
                    // reset to zero for next iteration
                    swap_bit(i) = 0;
                    dest_cache(i) = NULL_PART;
                }
                update++;
            }
        }
    }, total);
    affected = Kokkos::subview(vtx1, std::make_pair(static_cast<ordinal_t>(0), total));
    Kokkos::parallel_for("reset conn DS", policy_t(0, total), KOKKOS_LAMBDA(const ordinal_t x){
        const ordinal_t i = affected(x);
        edge_offset_t g_start = cdata.conn_offsets(i);
        edge_offset_t g_end = cdata.conn_offsets(i + 1);
        for(edge_offset_t j = g_start; j < g_end; j++) {
            cdata.conn_entries(j) = NULL_PART;
        }
        for(edge_offset_t j = g_start; j < g_end; j++) {
            cdata.conn_vals(j) = 0;
        }
        part_t size = g_end - g_start;
        part_t* s_conn_entries = cdata.conn_entries.data() + g_start;
        gain_t* s_conn_vals = cdata.conn_vals.data() + g_start;
        gain_t update = 0;
        part_t p_i = part(i);
        for(edge_offset_t j = g.graph.row_map(i); j < g.graph.row_map(i + 1); j++) {
            ordinal_t v = g.graph.entries(j);
            gain_t wgt;
            if constexpr(uniform) wgt = 1;
            else wgt = g.values(j);
            part_t p = part(v);
            if(p == p_i){
                update += wgt;
                continue;
            }
            while(j + 1 < g.graph.row_map(i+1) && p == part(g.graph.entries(j+1))){
                j++;
                if constexpr(uniform) wgt += 1;
                else wgt += g.values(j);
            }
            part_t p_o = hash(p) % static_cast<uint32_t>(size);
            part_t px = s_conn_entries[p_o];
            while(px != p && px != NULL_PART){
                p_o = (p_o + 1) % size;
                px = s_conn_entries[p_o];
            }
            if(px != p){
                s_conn_entries[p_o] = p;
            }
            s_conn_vals[p_o] += wgt;
        }
        pvals(i) = update;
    });
}

//update datastructures assuming a "small" number of vertices are moved
//2 kernels, 0 device-host syncs
template <bool uniform>
void update_small(const problem& prob, const part_vt part, const vtx_view_t swaps, const part_vt dest_part, mem_t& mem){
    const matrix_t& g = prob.g;
    ordinal_t total_moves = swaps.extent(0);
    vtx_view_t dest_cache = mem.p_mem.dest_cache;
    gain_vt pvals = mem.p_mem.pvals;
    cdata_t& cdata = mem.cd_mem;
    Kokkos::parallel_for("update conns (subtract) (high degree)", team_policy_t(total_moves, Kokkos::AUTO), KOKKOS_LAMBDA(const member& t){
        ordinal_t i = swaps(t.league_rank());
        part_t p = part(i);
        //subtract i's contribution to p connectivity for adjacent vertices
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [=] (const edge_offset_t j){
            ordinal_t v = g.graph.entries(j);
            gain_t wgt;
            if constexpr(uniform) wgt = 1;
            else wgt = g.values(j);
            if(p == part(v)){
                Kokkos::atomic_add(&pvals(v), -wgt);
                return;
            }
            edge_offset_t v_start = cdata.conn_offsets(v);
            part_t v_size = cdata.conn_table_sizes(v);
            part_t p_o = hash(p) % static_cast<uint32_t>(v_size);
            //v is always adjacent to p because it is adjacent to i which is in p
            while(cdata.conn_entries(v_start + p_o) != p){
                p_o = (p_o + 1) % v_size;
            }
            //DO NOT USE ATOMIC_ADD_FETCH HERE IT IS WAY SLOWER
            gain_t x = Kokkos::atomic_fetch_add(&cdata.conn_vals(v_start + p_o), -wgt);
            //parts have locked locations if v_size == k (even when not originally allocated to size k)
            if(x == wgt){
                //free this gain slot
                cdata.conn_entries(v_start + p_o) = HASH_RECLAIM;
            }
        });
    });

    // remove the new cluster id from hashmap, and insert old cluster id into hashmap
    Kokkos::parallel_for("swap pval and bval", policy_t(0, total_moves), KOKKOS_LAMBDA(const ordinal_t x){
        ordinal_t i = swaps(x);
        // remove best (new cluster) from the hashmap
        part_t best = dest_part(i);
        // insert p (old cluster) into the hashmap
        part_t p = part(i);
        part(i) = best;
        dest_part(i) = p;
        part_t size = cdata.conn_table_sizes(i);
        edge_offset_t offset = cdata.conn_offsets(i);
        part_t b_hash = hash(best) % static_cast<uint32_t>(size);
        gain_t old_val = pvals(i);
        pvals(i) = 0;
        // find new cluster's connection strength, and set into pval
        for(part_t q = 0; q < size; q++){
            part_t p_i = (b_hash + q) % size;
            if(cdata.conn_entries(offset + p_i) == best){
                pvals(i) = cdata.conn_vals(offset + p_i);
                // delete from hashmap
                cdata.conn_vals(offset + p_i) = 0;
                cdata.conn_entries(offset + p_i) = HASH_RECLAIM;
                break;
            } else if(cdata.conn_entries(offset + p_i) == NULL_PART){
                break;
            }
        }
        // DO NOT insert old cluster into hashmap if it has no connection to vertex
        if(old_val == 0) return;
        bool success = false;
        part_t p_o = hash(p) % static_cast<uint32_t>(size);
        // insert p into conn table
        // needs to find either HASH_RECLAIM or NULL_PART to make insertion
        while(!success){
            part_t px = cdata.conn_entries(offset + p_o);
            while(px > NULL_PART){
                p_o = (p_o + 1) % size;
                px = cdata.conn_entries(offset + p_o);
            }
            if(px <= NULL_PART) {
                cdata.conn_entries(offset + p_o) = p;
                success = true;
            }
        }
        cdata.conn_vals(offset + p_o) = old_val;
    });

    Kokkos::parallel_for("update conns (add) (high degree)", team_policy_t(total_moves, Kokkos::AUTO), KOKKOS_LAMBDA(const member& t){
        ordinal_t i = swaps(t.league_rank());
        //part contains new part at this point
        part_t best = part(i);
        //add i's contribution to best connectivity for adjacent vertices
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [=] (const edge_offset_t j){
            ordinal_t v = g.graph.entries(j);
            gain_t wgt;
            if constexpr(uniform) wgt = 1;
            else wgt = g.values(j);
            dest_cache(v) = NULL_PART;
            if(best == part(v)){
                Kokkos::atomic_add(&pvals(v), wgt);
                return;
            }
            edge_offset_t v_start = cdata.conn_offsets(v);
            part_t v_size = cdata.conn_table_sizes(v);
            part_t p_o = hash(best) % static_cast<uint32_t>(v_size);
            bool success = false;
            //check if best in conn table
            //can only determine best is absent if NULL_PART is found or v_size reached
            for(part_t q = 0; q < v_size; q++){
                part_t p_i = (p_o + q) % v_size;
                part_t px = cdata.conn_entries(v_start + p_i);
                if(px == best){
                    success = true;
                    p_o = p_i;
                    break;
                } else if(px == NULL_PART){
                    break;
                }
            }
            //insert best into conn table
            //needs to find either HASH_RECLAIM or NULL_PART to make insertion
            while(!success){
                part_t px = cdata.conn_entries(v_start + p_o);
                while(px != best && px > NULL_PART){
                    p_o = (p_o + 1) % v_size;
                    px = cdata.conn_entries(v_start + p_o);
                }
                if(px == best){
                    success = true;
                } else {
                    part_t orig = NULL_PART;
                    if(cdata.conn_entries(v_start + p_o) == HASH_RECLAIM) orig = HASH_RECLAIM;
                    //don't care if this thread succeeds if another thread succeeds with the same value
                    Kokkos::atomic_compare_exchange(&cdata.conn_entries(v_start + p_o), orig, best);
                    if(cdata.conn_entries(v_start + p_o) == best){
                        success = true;
                    } else {
                        p_o = (p_o + 1) % v_size;
                    }
                }
            }
            Kokkos::atomic_add(&cdata.conn_vals(v_start + p_o), wgt);
        });
    });
}

KOKKOS_INLINE_FUNCTION
static gain_t lookup(const part_t* keys, const gain_t* vals, const part_t& target, const part_t& size){
    part_t start = hash(target) % static_cast<uint32_t>(size);
    for(part_t q = 0; q < size; q++){
        part_t p_i = (start + q) % size;
        if(keys[p_i] == target){
            return vals[p_i];
        } else if(keys[p_i] == NULL_PART){
            return 0;
        }
    }
    return 0;
}

//perform swaps, update gains, and compute change to cut and imbalance
//4 kernels, 1 device-host syncs
template <bool uniform>
void perform_moves(const problem& prob, part_vt part, const vtx_view_t swaps, mem_t& mem, refine_data& curr_state, bool use_big){
    const wgt_view_t& wdeg = prob.wdeg;
    vtx_view_t dest_part = mem.p_mem.dest_part;
    ordinal_t total_moves = swaps.extent(0);
    cdata_t& cdata = mem.cd_mem;
    gain_vt pvals = mem.p_mem.pvals;
    vtx_view_t dest_cache = mem.p_mem.dest_cache;
    Kokkos::parallel_for("update total deg", policy_t(0, total_moves), KOKKOS_LAMBDA(const ordinal_t& x){
        ordinal_t i = swaps(x);
        part_t best = dest_part(i);
        part_t p = part(i);
        dest_cache(i) = NULL_PART;
        Kokkos::atomic_add(&curr_state.total_deg(p), -wdeg(i));
        Kokkos::atomic_add(&curr_state.total_deg(best), wdeg(i));
    });
    // this works well for large vertex swap counts
    // perhaps the old approach could be useful for small vertex swap counts (specifically during the uncoarsening pass)
    Kokkos::parallel_reduce("count cutsize change part1", policy_t(0, prob.g.numRows()), KOKKOS_LAMBDA(const ordinal_t& i, gain_t& gain_update){
        gain_update += pvals(i);
    }, mem.s_mem.cut_change1);
    //change part assignments and update part sizes
    if(!cdata.init || use_big || total_moves >= prob.g.numRows() * 0.1){
        // update cluster ids before updating datastructures
        Kokkos::parallel_for("update parts", policy_t(0, total_moves), KOKKOS_LAMBDA(const ordinal_t x){
            ordinal_t i = swaps(x);
            part_t best = dest_part(i);
            part(i) = best;
        });
        if(!cdata.init){
            init_conn_graph<uniform>(prob.g, part, mem);
            Kokkos::deep_copy(exec_space(), dest_cache, NULL_PART);
        } else {
            update_large<uniform>(prob, part, swaps, mem);
        }
    } else {
        // cluster ids updated inside this function
        update_small<uniform>(prob, part, swaps, dest_part, mem);
    }
    Kokkos::parallel_reduce("count cutsize change part2", policy_t(0, prob.g.numRows()), KOKKOS_LAMBDA(const ordinal_t& i, gain_t& gain_update){
        gain_update += pvals(i);
    }, mem.s_mem.cut_change2);
    exec_space().fence();
    // cut change is equal to newly covered edge count
    int64_t cut_change = mem.s_mem.cut_change2() - mem.s_mem.cut_change1();
    curr_state.cut -= cut_change;
}

//initialize conn hash tables for each vertex
template <bool uniform>
void init_conn_graph(const matrix_t& g, const part_vt& part, mem_t& mem){
    cdata_t& cdata = mem.cd_mem;
    cdata.init = true;
    Kokkos::deep_copy(exec_space(), cdata.conn_vals, 0);
    Kokkos::deep_copy(exec_space(), cdata.conn_entries, NULL_PART);
    ordinal_t cutoff = 32;
    ordinal_t total = 0;
    ordinal_t n = g.numRows();
    vtx_view_t vtx1 = mem.s_mem.vtx1;
    gain_vt pvals = mem.p_mem.pvals;
    Kokkos::parallel_scan("collect vtx to be updated", policy_t(0, g.numRows()), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
        ordinal_t degree = g.graph.row_map(i+1) - g.graph.row_map(i);
        if(degree >= cutoff){
            if(final){
                vtx1(update) = i;
            }
            update++;
        } else if(final){
            vtx1(n - 1 - (i - update)) = i;
        }
    }, total);
    vtx_view_t big = Kokkos::subview(vtx1, std::make_pair(static_cast<ordinal_t>(0), total));
    Kokkos::parallel_for("init conn DS", team_policy_t(total, Kokkos::AUTO), KOKKOS_LAMBDA(const member& t){
        ordinal_t i = big(t.league_rank());
        edge_offset_t g_start = cdata.conn_offsets(i);
        edge_offset_t g_end = cdata.conn_offsets(i + 1);
        part_t size = g_end - g_start;
        part_t* s_conn_entries = cdata.conn_entries.data() + g_start;
        gain_t* s_conn_vals = cdata.conn_vals.data() + g_start;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [&] (const edge_offset_t& j, gain_t& update){
            ordinal_t v = g.graph.entries(j);
            gain_t wgt;
            if constexpr(uniform) wgt = 1;
            else wgt = g.values(j);
            part_t p = part(v);
            if(p == part(i)){
                update += wgt;
                return;
            }
            part_t p_o = hash(p) % static_cast<uint32_t>(size);
            bool success = false;
            while(!success){
                part_t px = s_conn_entries[p_o];
                while(px != p && px != NULL_PART){
                    p_o = (p_o + 1) % size;
                    px = s_conn_entries[p_o];
                }
                if(px == p){
                    success = true;
                } else {
                    Kokkos::atomic_compare_exchange(s_conn_entries + p_o, NULL_PART, p);
                    if(s_conn_entries[p_o] == p){
                        success = true;
                    } else {
                        p_o = (p_o + 1) % size;
                    }
                }
            }
            Kokkos::atomic_add(s_conn_vals + p_o, wgt);
        }, pvals(i));
    });
    vtx_view_t small = Kokkos::subview(vtx1, std::make_pair(total, n));
    Kokkos::parallel_for("init conn DS", policy_t(0, n - total), KOKKOS_LAMBDA(const ordinal_t x){
        ordinal_t i = small(x);
        edge_offset_t g_start = cdata.conn_offsets(i);
        edge_offset_t g_end = cdata.conn_offsets(i + 1);
        part_t size = g_end - g_start;
        part_t* s_conn_entries = cdata.conn_entries.data() + g_start;
        gain_t* s_conn_vals = cdata.conn_vals.data() + g_start;
        gain_t update = 0;
        for(edge_offset_t j = g.graph.row_map(i); j < g.graph.row_map(i + 1); j++){
            ordinal_t v = g.graph.entries(j);
            gain_t wgt;
            if constexpr(uniform) wgt = 1;
            else wgt = g.values(j);
            part_t p = part(v);
            if(p == part(i)){
                update += wgt;
                continue;
            }
            part_t p_o = hash(p) % static_cast<uint32_t>(size);
            bool success = false;
            while(!success){
                part_t px = s_conn_entries[p_o];
                while(px != p && px != NULL_PART){
                    p_o = (p_o + 1) % size;
                    px = s_conn_entries[p_o];
                }
                if(px == p){
                    success = true;
                } else {
                    s_conn_entries[p_o] = p;
                    success = true;
                }
            }
            s_conn_vals[p_o] += wgt;
        }
        pvals(i) = update;
    });
}

//initializes datastructures
void truncate_and_init_mem(mem_t& mem, problem& prob, int label_count, bool top){
    const matrix_t g = prob.g;
    ordinal_t n = g.numRows();
    cdata_t& cdata = mem.cd_mem;
    cdata.init = false;
    cdata.conn_offsets = Kokkos::subview(cdata.conn_offsets, std::make_pair(static_cast<ordinal_t>(0), n + 1));
    cdata.conn_table_sizes = Kokkos::subview(cdata.conn_table_sizes, std::make_pair(static_cast<ordinal_t>(0), n));
    Kokkos::parallel_for("comp conn row size", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t& i){
        ordinal_t degree = g.graph.row_map(i + 1) - g.graph.row_map(i);
        if(!top) degree *= 1.2;
        if(degree > label_count) degree = label_count;
        cdata.conn_offsets(i) = degree;
        cdata.conn_table_sizes(i) = degree;
    });
    edge_offset_t gain_size = 0;
    Kokkos::parallel_scan("comp conn offsets", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t& i, edge_offset_t& update, const bool final){
        edge_offset_t x = cdata.conn_offsets(i);
        if(final){
            cdata.conn_offsets(i) = update;
        }
        update += x;
        if(final && i + 1 == n){
            cdata.conn_offsets(n) = update;
        }
    }, gain_size);
    cdata.conn_vals = Kokkos::subview(cdata.conn_vals, std::make_pair(static_cast<edge_offset_t>(0), gain_size));
    cdata.conn_entries = Kokkos::subview(cdata.conn_entries, std::make_pair(static_cast<edge_offset_t>(0), gain_size));
    cdata.c_graph = matrix_t("conn graph", g.numRows(), g.numRows(), gain_size, cdata.conn_vals, cdata.conn_offsets, cdata.conn_entries);
    Kokkos::deep_copy(exec_space(), mem.p_mem.pvals, 0);
    Kokkos::deep_copy(exec_space(), mem.p_mem.dest_cache, NULL_PART);
    Kokkos::deep_copy(exec_space(), mem.p_mem.lock_bit, 0);
}

template <bool uniform>
void jet_refine(const matrix_t g, wgt_view_t wdeg, part_vt best_part, refine_data& best_state, bool is_initial, mem_t& input_mem, ExperimentLoggerUtil<scalar_t>& experiment){
    Kokkos::Timer y;
    // contains reusable memory to avoid repeated allocations/deallocations
    // some of this memory contains state information
    mem_t mem(input_mem, g);
    // initialize metadata
    if(!best_state.init){
        best_state.mod = -1.0;
        best_state.total_deg = gain_vt("total degree of clusters", g.numRows());
        best_state.g_deg = stat::sum(wdeg);
        best_state.cut = best_state.g_deg;
        best_state.label_count = g.numRows();
        Kokkos::deep_copy(best_state.total_deg, wdeg);
        best_state.init = true;
    }
    problem prob;
    prob.g = g;
    prob.wdeg = wdeg;
    prob.use_team = (g.nnz() / g.numRows() >= 8);
    refine_data curr_state = clone_refine_data(best_state);
    part_vt part = mem.p_mem.part;
    Kokkos::deep_copy(exec_space(), part, best_part);
    truncate_and_init_mem(mem, prob, best_state.label_count, best_state.g_deg == g.nnz());
    if(!is_initial){
        init_conn_graph<uniform>(g, part, mem);
    }
    int iter_count = 0;
    Kokkos::fence();
    Kokkos::Timer iter_t;
    float filter_ratio = 0.75;
    bool use_big = true;
    int big_limit = 2;
    int limit = 6;
    bool skip = true;
    if(!is_initial) big_limit = 0;
    int count = 0;
    while(count++ < limit){
        iter_count++;
        if(iter_count > big_limit) use_big = false;
        vtx_view_t moves;
        matrix_t c_graph = mem.cd_mem.c_graph;
        if(!mem.cd_mem.init){
            // use the input graph in place of the conn graph
            c_graph = g;
        }
        moves = jet_lp<uniform>(prob, c_graph, part, curr_state, mem, filter_ratio, skip);
        if(moves.extent(0) == 0) break;
        perform_moves<uniform>(prob, part, moves, mem, curr_state, use_big);
        curr_state.mod = stat::modularity(curr_state.g_deg, curr_state.cut, curr_state.total_deg);
        // std::cout << "Cut: " << curr_state.cut << "; Modularity: " << std::setprecision(6) << curr_state.mod << "; Labels: " << stat::total_labels(curr_state.total_deg) << std::endl;
        //copy current partition and relevant data to output partition if following conditions pass
        if(curr_state.mod > best_state.mod){
            copy_refine_data(best_state, curr_state);
            Kokkos::deep_copy(exec_space(), best_part, part);
        }
    }
    Kokkos::fence();
    relabel_contiguously(best_part, best_state, mem);
    //divide cut by 2 because each cut edge is counted from both sides
    typename ExperimentLoggerUtil<scalar_t>::CoarseLevel cl(best_state.cut / 2, 0, g.nnz(), g.numRows(), y.seconds(), iter_t.seconds(), iter_count, iter_count);
    // std::cout << "Avg iteration time: " << (iter_t.seconds() / iter_count) << std::endl;
    experiment.addCoarseLevel(cl);
    y.reset();
    iter_t.reset();
}
};

}
