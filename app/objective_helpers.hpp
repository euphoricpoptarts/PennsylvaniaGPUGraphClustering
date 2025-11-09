#include "parse_args.hpp"
#include "weighted_graph.h"
#include "cluster_data.h"

namespace jet_community {
    
    std::unique_ptr<cluster_data<matrix_t>> get_objective(const weighted_graph<matrix_t> wg, const base_args args){
        switch(args.obj_type){
            case Objective::Modularity:
                return std::make_unique<modularity<matrix_t>>(wg.mtx, wg.vtx_w, args.lambda_multiplier, true);
            case Objective::WModularity:
                return std::make_unique<modularity<matrix_t>>(wg.mtx, wg.vtx_w, args.lambda_multiplier, wg.edge_uniform);
            case Objective::NLCC:
                return std::make_unique<normalized_lcc<matrix_t>>(wg.mtx, wg.vtx_w, args.lambda_multiplier, wg.edge_uniform);
            case Objective::CPM:
                return std::make_unique<constant_potts<matrix_t>>(wg.mtx, wg.vtx_w, args.lambda_multiplier);
            default:
                return std::make_unique<cluster_data<matrix_t>>(wg.mtx, wg.vtx_w, args.lambda_multiplier);
        }
    }

    bool uses_edge_weights(const base_args args) {
        if(args.obj_type == Objective::Modularity || args.obj_type == Objective::CPM){
            return true;
        }
        return false;
    }
}