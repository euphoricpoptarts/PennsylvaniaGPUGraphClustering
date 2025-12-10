#pragma once
#include <Kokkos_Core.hpp>
#include "KokkosSparse_CrsMatrix.hpp"
#include "memory_store.hpp"

namespace jet_community {

template<class crsMat>
class ordering {
public:

    // define internal types
    using matrix_t = crsMat;
    using exec_space = typename matrix_t::execution_space;
    using Device = typename matrix_t::device_type;
    using ordinal_t = typename matrix_t::ordinal_type;
    using vtx_vt = Kokkos::View<ordinal_t*, exec_space>;
    using policy_t = Kokkos::RangePolicy<exec_space>;
    using mem_t = memory_store<matrix_t>;
    static constexpr ordinal_t MID_CUTOFF = 32;
    static constexpr ordinal_t LARGE_CUTOFF = 128;
    static constexpr ordinal_t MASSIVE_CUTOFF = 15000;

// all this stuff is adapted from the kokkos wiki
struct wrapper {
    static const int N = 5;
    ordinal_t value[N];

    KOKKOS_INLINE_FUNCTION
    wrapper() {
        init();
    }

    KOKKOS_INLINE_FUNCTION
    wrapper(const wrapper& rhs) {
        for(ordinal_t i = 0; i < N; i++) value[i] = rhs.value[i];
    }

    KOKKOS_INLINE_FUNCTION
    void init() {
        for(ordinal_t i = 0; i < N; i++) value[i] = 0;
    }
};

template <class Space>
struct SumMyArray {
    public:
    // Required
    typedef SumMyArray reducer;
    typedef wrapper value_type;
    typedef Kokkos::View<value_type*, Space, Kokkos::MemoryUnmanaged>
        result_view_type;

    private:
    value_type& value;

    public:
    KOKKOS_INLINE_FUNCTION
    SumMyArray(value_type& _value) : value(_value) {}

    // Required
    KOKKOS_INLINE_FUNCTION
    void join(value_type& dest, const value_type& src) const {
        for(ordinal_t i = 0; i < dest.N; i++) dest.value[i] += src.value[i];
    }

    KOKKOS_INLINE_FUNCTION
    void init(value_type& val) const {
        val.init();
    }

    KOKKOS_INLINE_FUNCTION
    value_type& reference() const { return value; }

    KOKKOS_INLINE_FUNCTION
    result_view_type view() const { return result_view_type(&value, 1); }

    KOKKOS_INLINE_FUNCTION
    bool references_scalar() const { return true; }
};

struct ScanMyArray {
    public:
    // Required
    typedef ScanMyArray reducer;
    typedef wrapper value_type;

    private:
    matrix_t g;
    vtx_vt order1, order2;
    wrapper offsets;

    public:
    ScanMyArray(matrix_t _g,
        vtx_vt _order1,
        vtx_vt _order2,
        wrapper _offsets) : 
        g(_g),
        order1(_order1),
        order2(_order2),
        offsets(_offsets) {}

    // Required
    KOKKOS_INLINE_FUNCTION
    void join(value_type& dest, const value_type& src) const {
        for(ordinal_t i = 0; i < dest.N; i++) dest.value[i] += src.value[i];
    }

    KOKKOS_INLINE_FUNCTION
    void init(value_type& val) const {
        val.init();
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const ordinal_t i, wrapper& update, const bool final) const {
        ordinal_t degree = g.graph.row_map(i + 1) - g.graph.row_map(i);
        if(degree < LARGE_CUTOFF){
            if(final) order1(offsets.value[0] + update.value[0]) = i;
            update.value[0]++;
        } else if(degree < MASSIVE_CUTOFF){
            if(final) order1(offsets.value[1] + update.value[1]) = i;
            update.value[1]++;
        } else {
            if(final) {
                order1(offsets.value[4] + update.value[4]) = i;
                order2(offsets.value[4] + update.value[4]) = i;
            }
            update.value[4]++;
        }
        if(degree < MID_CUTOFF){
            if(final) order2(offsets.value[2] + update.value[2]) = i;
            update.value[2]++;
        } else if(degree < MASSIVE_CUTOFF){
            if(final) order2(offsets.value[3] + update.value[3]) = i;
            update.value[3]++;
        }
    }
};

// buckets vertices by degree, whilst maintaining natural order within each bucket
// generates two such orderings, only differing in the cutoff separating the first and second buckets
static void generate_orderings(mem_t& mem, const matrix_t& g) {
    ordinal_t n = g.numRows();
    vtx_vt order1 = mem.o_mem.order1;
    vtx_vt order2 = mem.o_mem.order2;
    wrapper bucket_sizes;
    typedef SumMyArray<Kokkos::HostSpace> ArraySumResult;

    Kokkos::parallel_reduce("sum bucket sizes", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, wrapper& update) {
        ordinal_t degree = g.graph.row_map(i + 1) - g.graph.row_map(i);
        if(degree < LARGE_CUTOFF){
            update.value[0]++;
        } else if(degree < MASSIVE_CUTOFF){
            update.value[1]++;
        } else {
            update.value[4]++;
        }
        if(degree < MID_CUTOFF){
            update.value[2]++;
        } else if(degree < MASSIVE_CUTOFF){
            update.value[3]++;
        }
    }, ArraySumResult(bucket_sizes));
    wrapper offsets(bucket_sizes);
    offsets.value[0] = 0;
    offsets.value[1] = bucket_sizes.value[0];
    offsets.value[2] = 0;
    offsets.value[3] = bucket_sizes.value[2];
    offsets.value[4] = bucket_sizes.value[0] + bucket_sizes.value[1];
    mem.o_mem.offset_large = offsets.value[1];
    mem.o_mem.offset_mid = offsets.value[3];
    mem.o_mem.offset_large2 = offsets.value[4];
    mem.o_mem.offset_mid2 = offsets.value[4];
    Kokkos::parallel_scan("generate orders", policy_t(0, n), ScanMyArray(g, order1, order2, offsets));
}

};

}