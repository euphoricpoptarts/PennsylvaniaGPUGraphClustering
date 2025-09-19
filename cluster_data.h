#pragma once
#include <Kokkos_Core.hpp>
#include <iostream>

template <typename matrix_t>
struct cluster_data {
    using Device = typename matrix_t::device_type;
    using scalar_t = typename matrix_t::value_type;
    using ordinal_t = typename matrix_t::ordinal_type;
    using wgt_vt = Kokkos::View<scalar_t*, Device>;
    using exec_space = typename matrix_t::execution_space;
    using policy_t = Kokkos::RangePolicy<exec_space>;

    // metadata that is preserved between levels in the clustering scheme
    wgt_vt total_deg;
    scalar_t g_deg = 0;
    scalar_t v_total = 0;
    scalar_t cut = 0;
    double obj = -1.0;
    ordinal_t label_count;

    // metadata needed between local move iterations
    scalar_t last_pval = 0;

    // objective scaling
    double penalty_scale = 1.0;

    static scalar_t sum(const wgt_vt wdeg){
        scalar_t result = 0;
        Kokkos::parallel_reduce("sum view", policy_t(0, wdeg.size()), KOKKOS_LAMBDA(const ordinal_t i, scalar_t& update){
            update += wdeg(i);
        }, result);
        return result;
    }

    cluster_data(const matrix_t g, const wgt_vt wdeg, double _penalty_scale, bool uniform) {
        // this is not the true objective for a singleton clustering
        // but we should find a better one regardless so it doesn't matter
        obj = -1.0;
        total_deg = wgt_vt("total degree of clusters", g.numRows());
        if(uniform) g_deg = g.nnz();
        else g_deg = sum(g.values);
        v_total = g.numRows();
        cut = g_deg;
        label_count = g.numRows();
        Kokkos::deep_copy(total_deg, wdeg);
        penalty_scale = _penalty_scale;
    }

    void update(const matrix_t g, const wgt_vt wdeg) {
        total_deg = wgt_vt("total deg", g.numRows());
        Kokkos::deep_copy(total_deg, wdeg);
        cut = sum(g.values);
        label_count = g.numRows();
        update_objective();
    }

    void copy(const cluster_data& rhs){
        wgt_vt td_lhs = Kokkos::subview(total_deg, std::make_pair((ordinal_t)0, rhs.label_count));
        wgt_vt td_rhs = Kokkos::subview(rhs.total_deg, std::make_pair((ordinal_t)0, rhs.label_count));
        Kokkos::deep_copy(exec_space(), td_lhs, td_rhs);
        g_deg = rhs.g_deg;
        cut = rhs.cut;
        v_total = rhs.v_total;
        obj = rhs.obj;
        label_count = rhs.label_count;
        penalty_scale = rhs.penalty_scale;
    }

    cluster_data(const cluster_data& rhs){
        total_deg = wgt_vt(Kokkos::ViewAllocateWithoutInitializing("total degree of clusters"), rhs.total_deg.extent(0));
        copy(rhs);
    }

    double get_penalty_modifier() const {
        double inv_gdeg = 1.0 / static_cast<double>(g_deg);
        double modifier = penalty_scale * inv_gdeg;
        return modifier;
    }

    void update_objective() {
        // avoid implicit capture of "this"
        wgt_vt total = total_deg;
        long long square_sum = 0;
        Kokkos::parallel_reduce("sum of squares", policy_t(0, label_count), KOKKOS_LAMBDA(const ordinal_t l, long long& update){
            long long c_size = total(l);
            update += c_size*c_size;
        }, square_sum);
        double inv_gdeg = 1.0 / static_cast<double>(g_deg);
        double penalty_factor = penalty_scale*inv_gdeg*inv_gdeg;
        double m = 1.0 - static_cast<double>(cut) * inv_gdeg;
        m -= static_cast<double>(square_sum)*penalty_factor;
        // std::cout << "Objective " << m << std::endl;
        obj = m;
    }

    friend std::ostream& operator<<(std::ostream& os, const cluster_data& cd) {
        os << "Cut: " << cd.cut / 2;
        os << " Modularity: " << cd.obj;
        os << " Labels: " << cd.label_count;
        return os;
    }
};