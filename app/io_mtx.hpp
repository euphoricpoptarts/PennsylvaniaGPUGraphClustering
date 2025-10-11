#pragma once
#include "defs.h"
#include <filesystem>
#include <sstream>
#include <string>
#include <iostream>
#include <fstream>
#include "io.hpp"

namespace jet_community {

bool load_mtx_graph(matrix_t& g, bool& uniform_ew, const char *fname) {
    Kokkos::Timer t;
    std::ifstream infp(fname);
    if (!infp.is_open()) {
        std::cerr << "FATAL ERROR: Could not open Matrix Market graph file " << fname << std::endl;
        return false;
    }
    size_t sz = std::filesystem::file_size(fname);
    char* s = new char[sz + 1];
    std::cout << "Reading " << sz << " bytes from " << fname << std::endl;
    infp.read(s, sz);
    infp.close();
    //append an endline to end of file in case one doesn't exist
    //needed to prevent parser from overshooting end of buffer
    if(s[sz - 1] != '\n'){
        s[sz] = '\n';
        sz++;
    }
    const char* f = s;
    const char* fmax = s + sz;
    size_t header[4] = {0, 0, 0, 0};
    //ignore commented lines
    while(*f == '%') next_line(f);
    while(!isdigit(*f)) f++;
    //read header data
    for(int i = 0; i < 4; i++){
        header[i] = fast_atoi<size_t>(f);
        while(!isdigit(*f)){
            if(*f == '\n'){
                //end for loop
                i = 4;
                f++;
                break;
            }
            f++;
        }
    }
    bool has_ew = false;
    ordinal_t n = header[0];
    edge_offset_t m = header[2];
    std::vector<ordinal_t> src(m);
    std::vector<ordinal_t> dst(m);

    vtx_view_t entries(Kokkos::ViewAllocateWithoutInitializing("entries"), m*2);
    vtx_mirror_t entries_m = Kokkos::create_mirror_view(entries);
    edge_view_t row_map(Kokkos::ViewAllocateWithoutInitializing("row_map"), n + 1);
    edge_mirror_t row_map_m = Kokkos::create_mirror_view(row_map);
    Kokkos::deep_copy(row_map_m, 0);
    wgt_view_t values(Kokkos::ViewAllocateWithoutInitializing("values"), 2*m);
    wgt_mirror_t values_m;
    if(has_ew){
        values_m = Kokkos::create_mirror_view(values);
    }
    edge_offset_t edges_read = 0;
    bool is_dest = false;
    //ready edge information
    while(f < fmax){
        //increment past whitespace
        while(f < fmax && !isdigit(*f)){
            //ignore commented lines
            if(*f == '%'){
                next_line(f);
                continue;
            }
            if(*f == '\n'){
                if(!is_dest){
                    std::cerr << "FATAL ERROR: Line with exactly one integer detected" << std::endl;
                    return false;
                }
            }
            f++;
        }
        if(f >= fmax) break;
        //fast_atoi also increments past numeric chars
        ordinal_t v = fast_atoi<ordinal_t>(f);
        //subtract 1 to convert to 0-indexed
        v--;
        row_map_m(v)++;
        if(!is_dest){
            src[edges_read] = v;
            is_dest = true;
        } else {
            dst[edges_read++] = v;
            is_dest = false;
            next_line(f);
        }
    }
    delete[] s;
    if(edges_read != m){
        std::cerr << "FATAL ERROR: Mismatch between expected and actual line/nonzero count in metis file" << std::endl;
        std::cerr << "Read " << edges_read << " nonzeros" << std::endl;
        std::cerr << "Nonzeros expected: " << m << std::endl;
        return false;
    }
    edge_offset_t sum = 0;
    for(ordinal_t i = 0; i <= n; i++){
        ordinal_t d = row_map_m(i);
        row_map_m(i) = sum;
        sum += d;
    }
    std::vector<ordinal_t> counters(n, 0);
    for(edge_offset_t j = 0; j < edges_read; j++){
        ordinal_t u = src[j];
        ordinal_t v = dst[j];
        edge_offset_t u_offset = row_map_m(u) + counters[u]++;
        edge_offset_t v_offset = row_map_m(v) + counters[v]++;
        entries_m(u_offset) = v;
        entries_m(v_offset) = u;
    }
    Kokkos::deep_copy(row_map, row_map_m);
    Kokkos::deep_copy(entries, entries_m);
    if(has_ew){
        Kokkos::deep_copy(values, values_m);
        uniform_ew = false;
    } else {
        Kokkos::deep_copy(values, 1);
        uniform_ew = true;
    }
    graph_t g_graph(entries, row_map);
    g = matrix_t("input graph", n, values, g_graph);
    std::cout << "Read graph from " << fname << " in " << std::setprecision(3) << t.seconds() << "s" << std::endl;
    return true;
}


bool load_graph(matrix_t& g, bool& uniform_ew, const char* fname) {
    std::filesystem::path p(fname);

    const std::string MTX_SUFFIX = ".mtx";
    if(p.extension().compare(MTX_SUFFIX) == 0){
        return load_mtx_graph(g, uniform_ew, fname);
    } else {
        return load_metis_graph(g, uniform_ew, fname);
    }
}

}
