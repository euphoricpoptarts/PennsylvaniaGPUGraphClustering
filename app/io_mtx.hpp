#pragma once
#include "defs.h"
#include <filesystem>
#include <sstream>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <functional>
#include "io.hpp"

namespace jet_community {

struct mm_meta {
    bool is_coordinate;
    bool is_pattern;
    bool is_integer;
    bool is_symmetric;
};

mm_meta get_mm_metadata(const char* f, const char* fmax){
    const int max_fields = 5;
    const std::string PATTERN = "pattern";
    const std::string INTEGER = "integer";
    const std::string COORDINATE = "coordinate";
    const std::string SYMMETRIC = "symmetric";
    std::string fields[max_fields];
    int field_num = 0;
    const char* begin = f;
    // tracks if we are inside an alphanumeric sequence
    bool state_machine = false;
    while(f < fmax) {
        char c = *f;
        if(isalnum(c)){
            if(!state_machine){
                begin = f;
            }
            state_machine = true;
        } else {
            if(state_machine){
                if(field_num < max_fields){
                    fields[field_num++].assign(begin, f);
                }
            }
            state_machine = false;
        }
        f++;
        if(c == '\n') break;
    }
    mm_meta data;
    data.is_coordinate = (COORDINATE == fields[2]);
    data.is_pattern = (PATTERN == fields[3]);
    data.is_integer = (INTEGER == fields[3]);
    data.is_symmetric = (SYMMETRIC == fields[4]);
    return data;
}

uint64_t HASH_EMPTY = std::numeric_limits<uint64_t>::max();

// basic linear-probing hash table
bool is_new_entry(std::vector<uint64_t>& htable, uint64_t x){
    std::hash<uint64_t> hash_f;
    size_t cap = htable.size();
    size_t mod = cap - 1;
    size_t i = hash_f(x) & mod;
    while(true){
        if(htable[i] == HASH_EMPTY){
            htable[i] = x;
            return true;
        } else if(htable[i] == x){
            return false;
        }
        i = (i + 1) & mod;
    }
}

enum class ParseState { SRC, DST, WGT };

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
    std::cout << "Parsing file as matrix market format" << std::endl;
    //append an endline to end of file in case one doesn't exist
    //needed to prevent parser from overshooting end of buffer
    if(s[sz - 1] != '\n'){
        s[sz] = '\n';
        sz++;
    }
    const char* f = s;
    const char* fmax = s + sz;
    mm_meta metadata = get_mm_metadata(f, fmax);
    if(!metadata.is_coordinate) {
        std::cerr << "FATAL ERROR: Importing of non-coordinate format matrix market files is not currently supported!" << std::endl;
        return false;
    }
    uniform_ew = true;
    if(!metadata.is_pattern) {
        if(metadata.is_integer){
            uniform_ew = false;
        } else {
            std::cout << "WARNING: Non-integer edge weights given in matrix market file. This is not currently supported. The given weights will be ignored." << std::endl;
        }
    }
    if(!metadata.is_symmetric) {
        std::cout << "INFO: Non-symmetric matrix market file detected. The graph will be automatically symmetrized." << std::endl;
        if(!uniform_ew) {
            uniform_ew = true;
            std::cout << "WARNING: Edge weights given in matrix market file. This is not supported for non-symmetric file formats. The given weights will be ignored." << std::endl;
        }
    }
    size_t sub_header[4] = {0, 0, 0, 0};
    //ignore commented lines
    while(*f == '%') next_line(f);
    while(!isdigit(*f)) f++;
    //read header data
    for(int i = 0; i < 4; i++){
        sub_header[i] = fast_atoi<size_t>(f);
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
    ordinal_t n = sub_header[0];
    if(sub_header[0] != sub_header[1]){
        std::cerr << "FATAL ERROR: Bipartite file format detected. This format is not currently supported!" << std::endl;
        return false;
    }
    edge_offset_t m = sub_header[2];
    std::vector<ordinal_t> src(m);
    std::vector<ordinal_t> dst(m);
    std::vector<value_t> val;
    if(!uniform_ew) val.resize(m);

    edge_offset_t edges_read = 0;
    ParseState state = ParseState::SRC;
    //read edge information
    while(f < fmax){
        //increment past whitespace
        while(f < fmax && !isdigit(*f)){
            //ignore commented lines
            if(*f == '%'){
                next_line(f);
                continue;
            }
            if(*f == '\n'){
                if(state != ParseState::SRC){
                    std::cerr << "FATAL ERROR: Line with exactly one value detected" << std::endl;
                    return false;
                }
            }
            f++;
        }
        if(f >= fmax) break;
        //fast_atoi also increments past numeric chars
        ordinal_t v = 0;
        value_t w = 0;
        //subtract 1 to convert to 0-indexed
        if(state != ParseState::WGT) v = fast_atoi<ordinal_t>(f) - 1;
        else w = fast_atoi<value_t>(f);
        if(state == ParseState::SRC){
            src[edges_read] = v;
            state = ParseState::DST;
        } else if(state == ParseState::DST) {
            dst[edges_read] = v;
            if(uniform_ew) {
                state = ParseState::SRC;
                edges_read++;
                next_line(f);
            } else {
                state = ParseState::WGT;
            }
        } else {
            val[edges_read++] = w;
            state = ParseState::SRC;
            next_line(f);
        }
    }
    delete[] s;
    if(edges_read != m){
        std::cerr << "FATAL ERROR: Mismatch between expected and actual line/nonzero count in matrix market file" << std::endl;
        std::cerr << "Read " << edges_read << " nonzeros" << std::endl;
        std::cerr << "Nonzeros expected: " << m << std::endl;
        return false;
    }
    std::cout << "Finished reading data from " << fname << std::endl;
    std::cout << "Beginning construction of graph" << std::endl;

    int self_loops = 0;
    int zero_weight = 0;
    edge_view_t row_map(Kokkos::ViewAllocateWithoutInitializing("row_map"), n + 1);
    edge_mirror_t row_map_m = Kokkos::create_mirror_view(row_map);
    Kokkos::deep_copy(row_map_m, 0);
    std::vector<uint64_t> unique_edges;
    if(!metadata.is_symmetric){
        size_t cap = 2;
        while(cap < (size_t)m) cap <<= 1;
        if(cap < (size_t)(1.1*m)) cap <<= 1;
        // hashtable expects a power of 2 capacity
        unique_edges.assign(cap, HASH_EMPTY);
    }
    // count edges per vertex
    for(edge_offset_t j = 0; j < edges_read; j++){
        ordinal_t u = src[j];
        ordinal_t v = dst[j];
        if(u == v){
            self_loops++;
            continue;
        }
        if(!uniform_ew){
            if(val[j] == 0){
                zero_weight++;
                // ignore this edge in a later loop
                src[j] = dst[j];
                continue;
            }
        }
        if(!metadata.is_symmetric){
            uint64_t less = (u < v) ? u : v;
            uint64_t more = (u > v) ? u : v;
            uint64_t signature = less*n + more;
            if(!is_new_entry(unique_edges, signature)){
                // ignore this edge in a later loop
                src[j] = dst[j];
                continue;
            }
        }
        row_map_m(u)++;
        row_map_m(v)++;
    }
    unique_edges.clear();
    if(self_loops > 0){
        std::cout << "WARNING: Ignoring " << self_loops << " self loop edges." << std::endl;
    }
    if(zero_weight > 0){
        std::cout << "WARNING: Ignoring " << zero_weight << " zero weight edges." << std::endl;
    }
    edge_offset_t sum = 0;
    // compute row map
    for(ordinal_t i = 0; i <= n; i++){
        ordinal_t d = row_map_m(i);
        row_map_m(i) = sum;
        sum += d;
    }

    // must adjust for replicated edges and self loops that we ignored
    m = sum / 2;
    vtx_view_t entries(Kokkos::ViewAllocateWithoutInitializing("entries"), m*2);
    vtx_mirror_t entries_m = Kokkos::create_mirror_view(entries);
    wgt_view_t values(Kokkos::ViewAllocateWithoutInitializing("values"), 2*m);
    wgt_mirror_t values_m;
    if(!uniform_ew){
        values_m = Kokkos::create_mirror_view(values);
    }

    std::vector<ordinal_t> counters(n, 0);
    // write edges to entries
    for(edge_offset_t j = 0; j < edges_read; j++){
        ordinal_t u = src[j];
        ordinal_t v = dst[j];
        if(u == v) continue;
        edge_offset_t u_offset = row_map_m(u) + counters[u]++;
        edge_offset_t v_offset = row_map_m(v) + counters[v]++;
        entries_m(u_offset) = v;
        entries_m(v_offset) = u;
        if(!uniform_ew){
            values_m(u_offset) = val[j];
            values_m(v_offset) = val[j];
        }
    }
    Kokkos::deep_copy(row_map, row_map_m);
    Kokkos::deep_copy(entries, entries_m);
    if(!uniform_ew){
        Kokkos::deep_copy(values, values_m);
    } else {
        Kokkos::deep_copy(values, 1);
    }

    graph_t g_graph(entries, row_map);
    g = matrix_t("input graph", n, values, g_graph);
    std::cout << "Read and constructed graph from " << fname << " in " << std::setprecision(3) << t.seconds() << "s" << std::endl;
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
