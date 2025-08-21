using JSON
using DelimitedFiles
using NaturalCapitalIndex

if length(ARGS) > 0
    config_file = ARGS[1]
else
    error("Must provide config file argument")
end

args = JSON.parsefile(config_file)
workspace = Base.Filesystem.joinpath(args["working_dir"], "packages")
country_file = args["country_list_file"]

df, header = readdlm(country_file, ',', header=true);

country_list = df[:, 2];

objectives = ["net_econ_value", "biodiversity", "carbon", "nitrate_cancer_cases"];
minimization_frontier_objectives = ["ag_value", "biodiversity", "carbon", "nitrate_cancer_cases"];

for country in country_list
    println(country)
    try
        do_nci(workspace, country, objectives, 5000, 1000,
            objectives_to_minimize=[4],
            minimization_frontier_objectives=minimization_frontier_objectives,
            suffix="");
    catch e
        open("runtime_error.log", "a") do file
            println(file, country, e)
        end
    end
end


