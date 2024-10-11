using Pkg

repo_dir = dirname(dirname(@__FILE__))
Pkg.develop([PackageSpec(path=joinpath(repo_dir, "moment_kinetics")),
             PackageSpec(path=joinpath(repo_dir, "makie_post_processing", "makie_post_processing")),
             PackageSpec(path=joinpath(repo_dir, "plots_post_processing", "plots_post_processing"))])
Pkg.instantiate()

using Documenter
using moment_kinetics, makie_post_processing, plots_post_processing

makedocs(
    sitename = "moment_kinetics",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    modules = [moment_kinetics, makie_post_processing, plots_post_processing],
)

if get(ENV, "CI", nothing) == "true"
    # Documenter can also automatically deploy documentation to gh-pages.
    # See "Hosting Documentation" and deploydocs() in the Documenter manual
    # for more information.
    deploydocs(
        repo = "github.com/mabarnes/moment_kinetics",
        push_preview = true,
    )
end
