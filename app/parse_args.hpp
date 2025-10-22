#include <unordered_set>
#include <unordered_map>
#include <string>

enum class Objective { Modularity, CPM, WModularity, NLCC, Base};

static std::unordered_map<std::string, Objective> obj_map = {
    {"Mod", Objective::Modularity},
    {"WMod", Objective::WModularity},
    {"CPM", Objective::CPM},
    {"NLCC", Objective::NLCC},
    {"LCC", Objective::Base}
};

struct cluster_args {
    int n_trials = 1;
    int n_successive_iterations = 0;
    std::string graph_file;
    std::string output_file;
    std::string vw_file;
    double lambda_multiplier = 1.0;
    Objective obj_type = Objective::Modularity;
    bool valid;
};

struct meme_args {
    int pop_size = 0;
    int time_limit_seconds = 0;
    std::string graph_file;
    std::string output_file;
    Objective obj_type = Objective::Modularity;
    bool valid;
};

void print_configuration(cluster_args args){
    std::cout << "Program configuration: " << std::endl;
    std::cout << "-- Input Graph File: " << args.graph_file << std::endl;
    if(args.vw_file.size() > 0) std::cout << "-- Input Vertex Weights File: " << args.vw_file << std::endl;
    else {
        std::cout << "-- Vertex Weighting Method: ";
        switch(args.obj_type){
            case Objective::Modularity:
                std::cout << "Degree" << std::endl;
                break;
            case Objective::WModularity:
                std::cout << "Weighted Degree" << std::endl;
                break;
            default:
                std::cout << "Unit Uniform" << std::endl;
        }
    }
    std::cout << "-- Successive Iteration Count: " << args.n_successive_iterations << std::endl;
    std::cout << "-- Total trial count: " << args.n_trials << std::endl;
    std::cout << "-- Objective Type: ";
    switch(args.obj_type){
        case Objective::Modularity:
            std::cout << "Modularity" << std::endl;
            break;
        case Objective::WModularity:
            std::cout << "Weighted Modularity" << std::endl;
            break;
        case Objective::CPM:
            std::cout << "Constant-Potts Model" << std::endl;
            break;
        case Objective::NLCC:
            std::cout << "Normalized LambdaCC" << std::endl;
            break;
        case Objective::Base:
            std::cout << "Base LambdaCC" << std::endl;
    }
    std::cout << "-- Lambda Multiplier: " << args.lambda_multiplier << std::endl;
    if(args.output_file.size() > 0) std::cout << "-- Output Clusters File: " << args.output_file << std::endl;
}

cluster_args parse_args(int argc, char** argv){
    std::unordered_set<std::string> valid_options = {"-i", "-o", "-trials", "-ex_iters", "-objective", "-lambda", "-vtx_weights"};
    std::unordered_map<std::string, std::string> required_options = {
        {"-i", "FATAL ERROR: Required graph input file not given. Please specify with argument '-i <graph file>'"}
    };
    std::unordered_map<std::string, std::string> given_options;
    
    cluster_args args;
    args.valid = true;
    bool is_opt_name = true;
    std::string curr_opt_name;
    // parse options
    for(int i = 1; i < argc; i++){
        std::string opt = argv[i];
        if(is_opt_name){
            if(valid_options.count(opt) != 0){
                curr_opt_name = opt;
                is_opt_name = false;
            } else {
                std::cerr << "FATAL ERROR: Invalid option name: " << opt << std::endl;
                args.valid = false;
                return args;
            }
        } else {
            given_options[curr_opt_name] = opt;
            is_opt_name = true;
        }
    }
    if(!is_opt_name){
        std::cerr << "FATAL ERROR: Value not provided for option '" << curr_opt_name << "'" << std::endl;
        args.valid = false;
        return args;
    }
    for(const auto& [req,msg] : required_options){
        if(given_options.count(req) == 0){
            std::cerr << msg << std::endl;
            args.valid = false;
        }
    }
    args.graph_file = given_options["-i"];
    if(given_options.count("-o") != 0){
        args.output_file = given_options["-o"];
    }
    if(given_options.count("-trials") != 0){
        args.n_trials = std::stoi(given_options["-trials"]);
        if(args.n_trials < 1) args.n_trials = 1;
    }
    if(given_options.count("-ex_iters") != 0){
        args.n_successive_iterations = std::stoi(given_options["-ex_iters"]);
        if(args.n_successive_iterations < 0) args.n_successive_iterations = 0;
    }
    if(given_options.count("-objective") != 0){
        std::string obj_specifier = given_options["-objective"];
        if(obj_map.count(obj_specifier) != 0){
            args.obj_type = obj_map[obj_specifier];
        } else {
            std::cerr << "FATAL ERROR: Invalid objective name '" << obj_specifier << "'!" << std::endl;
            args.valid = false;
        }
    }
    if(given_options.count("-lambda") != 0){
        args.lambda_multiplier = std::stod(given_options["-lambda"]);
    }
    if(given_options.count("-vtx_weights") != 0){
        args.vw_file = given_options["-vtx_weights"];
    }
    print_configuration(args);
    if(args.vw_file.size() > 0) {
        if(args.obj_type != Objective::NLCC && args.obj_type != Objective::Base){
            args.valid = false;
            std::cerr << "FATAL ERROR: Vertex weights file is only supported with objectives Normalized LambdaCC and Base LambdaCC!" << std::endl;
        }
    }
    return args;
}