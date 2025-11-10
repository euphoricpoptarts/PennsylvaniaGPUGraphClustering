#pragma once
#include <type_traits>
#include <Kokkos_Core.hpp>
#include "KokkosSparse_CrsMatrix.hpp"
#include "cluster_data.h"

// this struct contains almost all auxiliary memory used by the algorithm
// this allows for efficient reuse of memory
template <class crsMat>
struct memory_store {

    // define internal types
    using matrix_t = crsMat;
    using Device = typename matrix_t::device_type;
    using ordinal_t = typename matrix_t::ordinal_type;
    using edge_offset_t = typename matrix_t::size_type;
    using scalar_t = typename matrix_t::value_type;
    using vtx_vt = Kokkos::View<ordinal_t*, Device>;
    using edge_vt = Kokkos::View<edge_offset_t*, Device>;
    using wgt_vt = Kokkos::View<scalar_t*, Device>;
    using vtx_pin_st = Kokkos::View<ordinal_t, Kokkos::SharedHostPinnedSpace>;
    using edge_pin_st = Kokkos::View<edge_offset_t, Kokkos::SharedHostPinnedSpace>;
    using gain_pin_vt = Kokkos::View<gain_t*, Kokkos::SharedHostPinnedSpace>;
    using gain_pin_st = Kokkos::View<gain_t, Kokkos::SharedHostPinnedSpace>;
    using obj_vt = Kokkos::View<float*, Device>;

    // this struct contains memory which either requires initialization or some degree of persistence
    struct persistent {
        edge_vt row_map;
        wgt_vt vals;
        vtx_vt entries;
        vtx_vt cluster_sizes;
        obj_vt obj_persistent;
        wgt_vt pvals, pvals_clone;
        vtx_vt part, dest_part;
        vtx_vt order1, order2;
        vtx_vt dest_cache;
        ordinal_t last_scan_mid, last_scan_large;
        ordinal_t offset_mid, offset_large;

        persistent(const matrix_t largest){
            ordinal_t n = largest.numRows();
            vals = wgt_vt(Kokkos::ViewAllocateWithoutInitializing("vals"), largest.nnz()*1.2);
            entries = vtx_vt(Kokkos::ViewAllocateWithoutInitializing("entries"), largest.nnz()*1.2);
            cluster_sizes = vtx_vt(Kokkos::ViewAllocateWithoutInitializing("cluster table sizes"), n);
            row_map = edge_vt(Kokkos::ViewAllocateWithoutInitializing("row map"), n + 1);
            obj_persistent = obj_vt(Kokkos::ViewAllocateWithoutInitializing("gain persistent"), n);
            pvals = wgt_vt(Kokkos::ViewAllocateWithoutInitializing("p vals"), n);
            pvals_clone = wgt_vt(Kokkos::ViewAllocateWithoutInitializing("p vals clone"), n);
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
        edge_pin_st edge_scan_host;

        scratch(const ordinal_t n) {
            vtx1 = vtx_vt(Kokkos::ViewAllocateWithoutInitializing("vtx scratch 1"), n);
            vtx2 = vtx_vt(Kokkos::ViewAllocateWithoutInitializing("vtx scratch 2"), n);
            zeros1 = vtx_vt("zeros 1", n);
            scan_host = vtx_pin_st("scan host");
            edge_scan_host = edge_pin_st("edge scan host");
            pin_host = vtx_pin_st("pin host");
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