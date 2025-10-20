#include "local_mover.hpp"
#include "contract.hpp"
#include "memory_store.hpp"
#include "cluster_data.h"
#include "ExperimentLoggerUtil.hpp"
#include "leidenR.hpp"
#include "weighted_graph.h"

namespace jet_community {

template<class crsMat>
class clustering_methods {
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
    using policy_t = typename Kokkos::RangePolicy<exec_space>;
    using team_policy_t = typename Kokkos::TeamPolicy<exec_space>;
    using member = typename team_policy_t::member_type;
    using mem_t = memory_store<matrix_t>;
    using refine_data = cluster_data<matrix_t>;
    using contracter_t = contracter<matrix_t>;
    using wg_t = weighted_graph<matrix_t>;
    using rfd_t = cluster_data<matrix_t>;
    using lr_t = leidenR<matrix_t, ordinal_t>;
    using lm_t = local_move_heuristic<matrix_t>;

    static void coarsen_vtx_w(wgt_view_t in, wgt_view_t out, vtx_view_t map){
        Kokkos::parallel_for("set v weights", policy_t(0, in.extent(0)), KOKKOS_LAMBDA(const ordinal_t i){
            ordinal_t c = map(i);
            Kokkos::atomic_add(&out(c), in(i));
        });
    }

    static void downsample(vtx_vt in, vtx_vt out, vtx_view_t map){
        Kokkos::parallel_for("set v weights", policy_t(0, in.extent(0)), KOKKOS_LAMBDA(const ordinal_t i){
            ordinal_t c = map(i);
            out(c) = in(i);
        });
    }

    template <bool improve>
    static vtx_vt leiden_part(mem_t& mem, wg_t top, rfd_t& rfd, ExperimentLoggerUtil<value_t>& experiment, vtx_vt input){
        std::vector<wg_t> levels;
        std::vector<vtx_vt> parts;
        levels.push_back(top);
        double aggregate = 0;
        lm_t local_mover;
        vtx_vt part("cluster assignments", top.mtx.numRows());
        if(!improve){
            Kokkos::parallel_for("set initial assignments", r_policy(0, top.mtx.numRows()), KOKKOS_LAMBDA(const ordinal_t x){
                part(x) = x;
            });
        } else Kokkos::deep_copy(part, input);
        while(true) {
            wg_t c = levels[levels.size() - 1];
            // std::cout << "num coarse vertices: " << c.mtx.numRows() << "; edges: " << c.mtx.nnz() << std::endl;
            double old_obj = rfd.obj;
            if(levels.size() == 1) local_mover.template local_move<true, false>(c, part, rfd, !improve, mem, part);
            else local_mover.template local_move<false, false>(c, part, rfd, false, mem, part);
            if(old_obj == rfd.obj){
                if(levels.size() == 1) local_mover.template local_move_strict<true, false>(c, part, rfd, !improve, mem, part);
                else local_mover.template local_move_strict<false, false>(c, part, rfd, false, mem, part);
            }
            if(rfd.label_count == c.mtx.numRows()){
                parts.push_back(part);
                break;
            }
            vtx_vt louv = part;
            int coarse_vtx_count = 0;
            vtx_vt coarse_map;
            if(levels.size() == 1) coarse_map = lr_t::template coarsen_leidenR<true>(c, louv, mem, rfd, coarse_vtx_count);
            else coarse_map = lr_t::template coarsen_leidenR<false>(c, louv, mem, rfd, coarse_vtx_count);
            parts.push_back(coarse_map);
            if(coarse_vtx_count < c.mtx.numRows()){
                Kokkos::Timer t;
                contracter_t contracter;
                wg_t next_level;
                if(levels.size() == 1) next_level = contracter.template build_coarse_graph<true>(c, coarse_map, coarse_vtx_count, mem);
                else next_level = contracter.template build_coarse_graph<false>(c, coarse_map, coarse_vtx_count, mem);
                next_level.vtx_w = wgt_view_t("weighted degree 2", coarse_vtx_count);
                coarsen_vtx_w(c.vtx_w, next_level.vtx_w, coarse_map);

                part = vtx_vt("cluster assignments coarse", coarse_vtx_count);
                downsample(louv, part, coarse_map);

                levels.push_back(next_level);
                aggregate += t.seconds();
            } else {
                // avoid creating new graph if leidenR didn't contract any vertices
                // which may happen with astronomically low probability for any input clustering
                // or if input clustering is very bad
                levels.push_back(c);
            }
        }

        // std::cout << "Aggregation time: " << aggregate << "s" << std::endl;
        experiment.addMeasurement(Measurement::Contract, aggregate);
        // std::cout << rfd.obj << std::endl;
        // std::cout << rfd.label_count << std::endl;
        int64_t t_nnz = 0;
        for(const wg_t& level : levels){
            t_nnz += level.mtx.nnz();
        }
        experiment.setTotalNnz(t_nnz);
        experiment.setLevelCount(levels.size());
        for(int i = levels.size() - 2; i >= 0; i--){
            wg_t c = levels[i];
            vtx_vt coarse_part = parts[i + 1];
            vtx_vt fine_part = parts[i];
            Kokkos::parallel_for("update top level assignments", r_policy(0, c.mtx.numRows()), KOKKOS_LAMBDA(const ordinal_t x){
                fine_part(x) = coarse_part(fine_part(x));
            });
    #ifdef LEIDEN_PLUS
            if(i == 0) local_mover.template local_move<true, false>(c, fine_part, rfd, false, mem, part);
            else local_mover.template local_move<false, false>(c, fine_part, rfd, false, mem, part);
    #endif
        }
        return parts[0];
    }

    template <bool constrained>
    static vtx_vt louvain_part(mem_t& mem, wg_t top, rfd_t& rfd, ExperimentLoggerUtil<value_t>& experiment, vtx_vt constraint){
        if(constrained) rfd.reset(top.mtx, top.vtx_w);
        std::vector<wg_t> levels;
        std::vector<vtx_vt> parts;
        levels.push_back(top);
        double aggregate = 0;
        lm_t local_mover;
        bool drop_constraint = constrained;
        while(true) {
            wg_t c = levels[levels.size() - 1];
            // std::cout << "Pre-refine" << std::endl;
            vtx_vt part("cluster assignments", c.mtx.numRows());
            Kokkos::parallel_for("set initial assignments", r_policy(0, c.mtx.numRows()), KOKKOS_LAMBDA(const ordinal_t x){
                part(x) = x;
            });
            if(levels.size() == 1) local_mover.template local_move<true, constrained>(c, part, rfd, true, mem, constraint);
            else local_mover.template local_move<false, constrained>(c, part, rfd, true, mem, constraint);
            // the user clearly cares about quality if they are doing multiple iterations
            if(constrained && rfd.label_count == c.mtx.numRows()){
                if(levels.size() == 1) local_mover.template local_move_strict<true, constrained>(c, part, rfd, true, mem, constraint);
                else local_mover.template local_move_strict<false, constrained>(c, part, rfd, true, mem, constraint);
            }
            parts.push_back(part);
            if(rfd.label_count < c.mtx.numRows()){
                Kokkos::Timer t;
                contracter_t contracter;
                wg_t next_level;
                if(levels.size() == 1) next_level = contracter.template build_coarse_graph<true>(c, part, rfd.label_count, mem);
                else next_level = contracter.template build_coarse_graph<false>(c, part, rfd.label_count, mem);
                wgt_view_t td_rfd = Kokkos::subview(rfd.total_deg, std::make_pair((ordinal_t)0, rfd.label_count));
                next_level.vtx_w = wgt_view_t("next level vtx weights", rfd.label_count);
                Kokkos::deep_copy(next_level.vtx_w, td_rfd);
                levels.push_back(next_level);

                if(constrained) {
                    vtx_vt next_constraint("next constraint", rfd.label_count);
                    downsample(constraint, next_constraint, part);
                    constraint = next_constraint;
                }

                aggregate += t.seconds();
            } else if(drop_constraint) {
                Kokkos::deep_copy(constraint, 0);
                parts.pop_back();
                drop_constraint = false;
            } else {
                break;
            }
        }
        
        int64_t t_nnz = 0;
        for(const wg_t& level : levels){
            t_nnz += level.mtx.nnz();
        }
        experiment.setTotalNnz(t_nnz);
        experiment.setLevelCount(levels.size());

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
            wg_t c = levels[i];
            vtx_vt coarse_part = parts[i + 1];
            vtx_vt part = parts[i];
            Kokkos::parallel_for("update top level assignments", r_policy(0, c.mtx.numRows()), KOKKOS_LAMBDA(const ordinal_t x){
                part(x) = coarse_part(part(x));
            });
            if(i == 0) local_mover.template local_move<true, false>(c, part, rfd, false, mem, constraint);
            else local_mover.template local_move<false, false>(c, part, rfd, false, mem, constraint);
        }

        experiment.addMeasurement(Measurement::Contract, aggregate);
        // std::cout << "Post uncoarsen obj: " << rfd.obj << std::endl;
        return parts[0];
    }
};

}