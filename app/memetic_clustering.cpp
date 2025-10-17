#include "defs.h"
#include "io.hpp"
#include "io_mtx.hpp"
#include "contract.hpp"
#include "memory_store.hpp"
#include "cluster_data.h"
#include "ExperimentLoggerUtil.hpp"
#include "clustering_methods.hpp"

using namespace jet_community;
using rfd_t = cluster_data<matrix_t>;
using contracter_t = contracter<matrix_t>;
using clt = typename contracter_t::coarse_level_triple;
using mem_t = memory_store<matrix_t>;
using cm_t = clustering_methods<matrix_t>;

part_vt intersection_cluster(part_vt c1, part_vt c2, int l2){

    ordinal_t n = c1.extent(0);
    part_vt htable("hash table", n);
    Kokkos::deep_copy(htable, -1);

    part_vt out("output constraint", n);
    Kokkos::parallel_for("insert products", r_policy(0, n), KOKKOS_LAMBDA(const ordinal_t i) {
        ordinal_t a = c1(i);
        ordinal_t b = c2(i);
        ordinal_t h = a*l2 + b;
        ordinal_t x = h % n;
        while(true){
            Kokkos::atomic_compare_exchange(&htable(x), -1, h);
            if(htable(x) == h){
                break;
            } else {
                x = (x + 1) % n;
            }
        }
        out(i) = x;
    });
    part_vt hval("hash vals", n);
    Kokkos::parallel_scan("get new ids", r_policy(0, n), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
        if(htable(i) != -1){
            if(final){
                hval(i) = update;
            }
            update++;
        }
    });
    Kokkos::parallel_for("set ids", r_policy(0, n), KOKKOS_LAMBDA(const ordinal_t i){
        out(i) = hval(out(i));
    });
    return out;
}

struct clustering {
    part_vt clusters;
    double obj;
    int labels;
};

value_t get_cut_diff(matrix_t g, part_vt c1, part_vt c2){
    value_t cut = 0;
    // Kokkos::parallel_reduce("count cut", r_policy(0, g.numRows()), KOKKOS_LAMBDA(const ordinal_t i, value_t& update){
    //     for(edge_offset_t j = g.graph.row_map(i); j < g.graph.row_map(i+1); j++){
    //         ordinal_t v = g.graph.entries(j);
    //         if(c1[i] != c1[v] && c2[i] == c2[v]) update += g.values(j);
    //         else if(c1[i] == c1[v] && c2[i] != c2[v]) update += g.values(j);
    //     }
    // }, cut);
    Kokkos::parallel_reduce("count cut", policy(g.numRows(), Kokkos::AUTO), KOKKOS_LAMBDA(const member& t, value_t& global_update){
        ordinal_t i = t.league_rank();
        value_t local_sum = 0;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i+1)), [&](const edge_offset_t j, value_t& update){
            ordinal_t v = g.graph.entries(j);
            if(c1[i] != c1[v] && c2[i] == c2[v]) update += g.values(j);
            else if(c1[i] == c1[v] && c2[i] != c2[v]) update += g.values(j);
        }, local_sum);
        Kokkos::single(Kokkos::PerTeam(t), [&](){
            global_update += local_sum;
        });
    }, cut);
    return cut;
}

part_vt meme_cluster(matrix_t g,
                    wgt_view_t vweights,
                    int pop_size,
                    int time_limit) {

    rfd_t rfd(g, vweights, 1.0, true);
    mem_t mem(g, rfd);
    clt top;
    top.mtx = g;
    top.wdeg = vweights;
    std::vector<clustering> pop;
    ExperimentLoggerUtil<value_t> dummy;
    std::cout << std::setprecision(9);
    for(int i = 0; i < pop_size; i++){
        part_vt dummy_constraint;
        part_vt c = cm_t::leiden_part<false>(mem, top, rfd, dummy, dummy_constraint);
        clustering y;
        y.clusters = c;
        y.obj = rfd.obj;
        y.labels = rfd.label_count;
        std::cout << "Adding clustering with obj: " << rfd.obj << std::endl;
        pop.push_back(y);
        rfd.update(g, vweights);
    }

    std::random_device r;
    std::default_random_engine e1(r());
    std::uniform_int_distribution<int> uniform_dist1(0, pop_size - 1);
    std::uniform_int_distribution<int> uniform_dist2(0, pop_size - 2);
    std::uniform_int_distribution<int> uniform_dist3(0, g.nnz());
    double best = 0;
    int e = 0;
    Kokkos::Timer t;
    while(t.seconds() < time_limit) {
        for(int i = 0; i < pop_size; i++){
            int choice1 = uniform_dist1(e1);
            int choice2 = uniform_dist2(e1);
            choice2 = (choice1 + choice2) % pop_size;
            int p1 = pop[choice1].obj > pop[choice2].obj ? choice1 : choice2;
            int p2 = p1;
            while(p2 == p1){
                choice1 = uniform_dist1(e1);
                choice2 = uniform_dist2(e1);
                choice2 = (choice1 + choice2) % pop_size;
                p2 = pop[choice1].obj > pop[choice2].obj ? choice1 : choice2;
            }
            clustering c1 = pop[p1]; // choose parent 1
            clustering c2 = pop[p2]; // choose parent 2
            part_vt constraint = intersection_cluster(c1.clusters, c2.clusters, c2.labels);
            // create offspring
            part_vt c3 = cm_t::louvain_part<true>(mem, top, rfd, dummy, constraint);
            std::cout << "Parent 1 obj: " << c1.obj << "; Parent 2 obj: " << c2.obj << "; Offspring obj: " << rfd.obj;
            for(int x = 0; x < 5; x++){
                c3 = cm_t::leiden_part<true>(mem, top, rfd, dummy, c3);
            }
            std::cout << "; Post leiden obj: " << rfd.obj;
            if(rfd.obj > best){
                best = rfd.obj;
            }

            // replace worst with c3
            int am = -1;
            double obj_max = rfd.obj;
            double min_diff = g.nnz();
            for(int p = 0; p < pop_size; p++){
                double obj = pop[p].obj;
                if(obj < obj_max){
                    // value_t diff = uniform_dist3(e1);
                    value_t diff = get_cut_diff(g, c3, pop[p].clusters);
                    if(diff < min_diff){
                        min_diff = diff;
                        am = p;
                    }
                }
            }
            if(am != -1){
                pop[am].clusters = c3;
                pop[am].obj = rfd.obj;
                pop[am].labels = rfd.label_count;
            } else {
                min_diff = 0;
            }
            std::cout << "; Min diff: " << min_diff << std::endl;
            // std::cout << std::endl;
            rfd.update(g, vweights);
        }
        std::cout << "Epoch " << e << " best objective: " << best << std::endl;
        e++;
    }
    int am = -1;
    double obj_max = 0;
    for(int p = 0; p < pop_size; p++){
        double obj = pop[p].obj;
        if(obj > obj_max){
            obj_max = obj;
            am = p;
        }
    }
    std::cout << "Final objective: " << obj_max << std::endl;
    return pop[am].clusters;
}

void degree_weighting(const matrix_t& g, wgt_view_t vweights){
    Kokkos::parallel_for("set v weights", r_policy(0, g.numRows()), KOKKOS_LAMBDA(const ordinal_t i){
        vweights(i) = g.graph.row_map(i + 1) - g.graph.row_map(i);
    });
}

int main(int argc, char **argv) {

    if (argc < 4) {
        std::cerr << "Insufficient number of args provided" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <graph_file> <pop size> <time limit in seconds> <optional clustering_output_filename>" << std::endl;
        return -1;
    }
    char *filename = argv[1];
    int pop_size = atoi(argv[2]);
    if(pop_size < 10){
        std::cout << "WARNING: Population size given as " << pop_size << " < 10. Setting population size to 10." << std::endl;
        pop_size = 10;
    }
    int time_limit = atoi(argv[3]);
    if(time_limit < 10){
        std::cout << "WARNING: Time limit given as " << time_limit << "s < 10s. Setting time limit to 10s." << std::endl;
        time_limit = 10;
    }
    char *clusters_file = nullptr;
    if(argc >= 5){
        clusters_file = argv[4];
    }

    Kokkos::initialize(argc, argv);
    //must scope kokkos-related data
    //so that it falls out of scope b4 finalize
    {
        matrix_t g;
        bool uniform_ew = false;
        if(!load_graph(g, uniform_ew, filename)) return -1;
        std::cout << "vertices: " << g.numRows() << "; edges: " << g.nnz() / 2 << std::endl;
        wgt_view_t vweights("vertex weights", g.numRows());
        degree_weighting(g, vweights);
        //Kokkos::deep_copy(vweights, 1);

        part_vt best_clusters = meme_cluster(g, vweights, pop_size, time_limit);
        if(clusters_file != nullptr){
            std::cout << "Writing best clustering to " << clusters_file << std::endl;
            write_part(best_clusters, clusters_file);
        } 
    }
    Kokkos::finalize();

    return 0;
}