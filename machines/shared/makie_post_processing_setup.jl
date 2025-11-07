using Pkg, TOML

repo_dir = dirname(Pkg.project().path)
local_preferences_filename = joinpath(repo_dir, "LocalPreferences.toml")
local_preferences = TOML.parsefile(local_preferences_filename)
mk_preferences = local_preferences["moment_kinetics"]
batch_system = mk_preferences["batch_system"]

if mk_preferences["use_makie"] == "y"
    println("Setting up makie_post_processing")

    if batch_system || mk_preferences["separate_postproc_projects"] == "y"
        touch(joinpath("makie_post_processing", "Project.toml"))
        Pkg.activate("makie_post_processing")
        project_dir = dirname(Pkg.project().path)

        include("add_dependencies_to_project.jl")
        Pkg.add(["Makie", "CairoMakie"])
        Pkg.develop(path=joinpath("makie_post_processing", "makie_post_processing"))
        Pkg.precompile()
    else
        Pkg.add(["Makie", "CairoMakie"])
        Pkg.develop(path=joinpath("makie_post_processing", "makie_post_processing"))
    end
end

if !batch_system && (mk_preferences["use_makie"] == "n" ||
                     mk_preferences["separate_postproc_projects"] == "y")
    # If makie_post_processing and dependencies have been added previously, remove
    # them
    Pkg.activate(".")
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
