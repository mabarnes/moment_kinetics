using Pkg, TOML

repo_dir = dirname(dirname(dirname(@__FILE__)))
local_preferences_filename = joinpath(repo_dir, "LocalPreferences.toml")
local_preferences = TOML.parsefile(local_preferences_filename)
mk_preferences = local_preferences["moment_kinetics"]
batch_system = mk_preferences["batch_system"]

if mk_preferences["use_plots"] == "y"
    println("Setting up plots_post_processing")

    if batch_system
        touch(joinpath("plots_post_processing", "Project.toml"))
        Pkg.activate("plots_post_processing")

        include("add_dependencies_to_project.jl")
        Pkg.add("Plots")
        Pkg.develop(path=joinpath("plots_post_processing", "plots_post_processing"))
        Pkg.precompile()

        if mk_preferences["submit_precompilation"] == "y"
            run(`precompile-plots-post-processing-submit.sh`)
        end
    else
        Pkg.add("Plots")
        Pkg.develop(path=joinpath("plots_post_processing", "plots_post_processing"))
    end
else
    if !batch_system
        # If plots_post_processing and dependencies have been added previously, remove
        # them
        try
            Pkg.rm("plots_post_processing")
        catch
        end
        try
            Pkg.rm("Plots")
        catch
        end
    end
end
