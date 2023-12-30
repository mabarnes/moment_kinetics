using Pkg, TOML

repo_dir = dirname(dirname(dirname(@__FILE__)))
local_preferences_filename = joinpath(repo_dir, "LocalPreferences.toml")
local_preferences = TOML.parsefile(local_preferences_filename)
mk_preferences = local_preferences["moment_kinetics"]
batch_system = mk_preferences["batch_system"]

if mk_preferences["use_makie"] == "y"
    println("Setting up makie_post_processing")

    if batch_system
        touch(joinpath("makie_post_processing", "Project.toml"))
        Pkg.activate("makie_post_processing")

        include("add_dependencies_to_project.jl")
        Pkg.add(["Makie", "CairoMakie"])
        Pkg.develop(path=joinpath("makie_post_processing", "makie_post_processing"))
        Pkg.precompile()

        if mk_preferences["submit_precompilation"] == "y"
            run(`precompile-makie-post-processing-submit.sh`)
        end
    else
        Pkg.add(["Makie", "CairoMakie"])
        Pkg.develop(path=joinpath("makie_post_processing", "makie_post_processing"))
    end
else
    if !batch_system
        # If makie_post_processing and dependencies have been added previously, remove
        # them
        try
            Pkg.rm("makie_post_processing")
        catch
        end
        try
            Pkg.rm("Makie")
        catch
        end
        try
            Pkg.rm("CairoMakie")
        catch
        end
    end
end
