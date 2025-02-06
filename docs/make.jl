using Pkg

Pkg.instantiate()

using Documenter, UUIDs
using moment_kinetics, makie_post_processing, plots_post_processing

makedocs(
    sitename = "moment_kinetics",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true",
                             size_threshold = 1000000,
                             size_threshold_warn = 500000,
                             # Use the following horrible incantation to get the version
                             # of moment_kinetics. moment_kinetics is the package with the
                             # UUID being used here. We need to do this because the
                             # Project.toml in the top-level directory is user-generated
                             # and does not have a version, but this is the Project.toml
                             # that would be used by default by Documenter.jl.
                             inventory_version = Pkg.dependencies()[UUID("b5ff72cc-06fc-4161-ad14-dba1c22ed34e")].version,
                            ),
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
