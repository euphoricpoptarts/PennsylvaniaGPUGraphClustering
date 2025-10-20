#pragma once
#include <type_traits>
#include <Kokkos_Core.hpp>
#include "KokkosSparse_CrsMatrix.hpp"
#include "cluster_data.h"

// this struct contains almost all auxiliary memory used by the algorithm
// this allows for efficient reuse of memory
template <class crsMat>
struct memory_store {

    //helper for getting gain_t
    template<typename T>
    struct type_identity {
        typedef T type;
    };

    // define internal types
    using matrix_t = crsMat;
    using Device = typename matrix_t::device_type;
    using ordinal_t = typename matrix_t::ordinal_type;
    using edge_offset_t = typename matrix_t::size_type;
    using scalar_t = typename matrix_t::value_type;
    // need some trickery because make_signed is undefined for floating point types
    using gain_t = typename std::conditional_t<std::is_signed_v<scalar_t>, type_identity<scalar_t>, std::make_signed<scalar_t>>::type;
    using vtx_vt = Kokkos::View<ordinal_t*, Device>;
    using edge_vt = Kokkos::View<edge_offset_t*, Device>;
    using gain_vt = Kokkos::View<gain_t*, Device>;
    using gain_svt = Kokkos::View<gain_t, Device>;
    using vtx_pin_st = Kokkos::View<ordinal_t, Kokkos::SharedHostPinnedSpace>;
    using gain_pin_vt = Kokkos::View<gain_t*, Kokkos::SharedHostPinnedSpace>;
    using gain_pin_st = Kokkos::View<gain_t, Kokkos::SharedHostPinnedSpace>;
    using obj_vt = Kokkos::View<float*, Device>;

    // this struct contains memory which either requires initialization or some degree of persistence
    struct persistent {
        edge_vt row_map;
        gain_vt vals;
        vtx_vt entries;
        vtx_vt cluster_sizes;
        obj_vt obj_persistent;
        gain_vt pvals, pvals_clone;
        vtx_vt part, dest_part;
        vtx_vt order1, order2;
        vtx_vt dest_cache;
        ordinal_t offset_mid, offset_large;

        persistent(const matrix_t largest){
            ordinal_t n = largest.numRows();
            vals = gain_vt(Kokkos::ViewAllocateWithoutInitializing("vals"), largest.nnz()*1.2);
            entries = vtx_vt(Kokkos::ViewAllocateWithoutInitializing("entries"), largest.nnz()*1.2);
            cluster_sizes = vtx_vt(Kokkos::ViewAllocateWithoutInitializing("cluster table sizes"), n);
            row_map = edge_vt(Kokkos::ViewAllocateWithoutInitializing("row map"), n + 1);
            obj_persistent = obj_vt(Kokkos::ViewAllocateWithoutInitializing("gain persistent"), n);
            pvals = gain_vt(Kokkos::ViewAllocateWithoutInitializing("p vals"), n);
            pvals_clone = gain_vt(Kokkos::ViewAllocateWithoutInitializing("p vals clone"), n);
            dest_part = vtx_vt(Kokkos::ViewAllocateWithoutInitializing("destination scratch"), n);
            part = vtx_vt(Kokkos::ViewAllocateWithoutInitializing("part scratch"), n);
            dest_cache = vtx_vt(Kokkos::ViewAllocateWithoutInitializing("best connected part for each vertex"), n);
            order1 = vtx_vt(Kokkos::ViewAllocateWithoutInitializing("vtx ordering 1"), n);
            order2 = vtx_vt(Kokkos::ViewAllocateWithoutInitializing("vtx ordering 2"), n);
        }
    };

    // this struct contains memory which can be used as-is
    struct scratch {
        vtx_vt vtx1, vtx2, zeros1;
        vtx_pin_st scan_host, pin_host;
        gain_pin_st cut_change1, cut_change2;
        gain_pin_vt reduce_locs;

        scratch(const ordinal_t n) {
            vtx1 = vtx_vt(Kokkos::ViewAllocateWithoutInitializing("vtx scratch 1"), n);
            vtx2 = vtx_vt(Kokkos::ViewAllocateWithoutInitializing("vtx scratch 2"), n);
            zeros1 = vtx_vt("zeros 1", n);
            scan_host = vtx_pin_st("scan host");
            pin_host = vtx_pin_st("pin host");
            reduce_locs = gain_pin_vt("reduce to here", 2);
            cut_change1 = Kokkos::subview(reduce_locs, 0);
            cut_change2 = Kokkos::subview(reduce_locs, 1);
        }
    };

    persistent p_mem;
    scratch s_mem;
    cluster_data<matrix_t> spare_cluster_data;

    memory_store(const matrix_t largest, cluster_data<matrix_t>& clone_target) :
        p_mem(largest), 
        s_mem(largest.numRows()),
        spare_cluster_data(clone_target) {}
};