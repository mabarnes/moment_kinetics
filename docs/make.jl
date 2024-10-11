using Pkg

Pkg.instantiate()

using Documenter
using moment_kinetics, makie_post_processing, plots_post_processing

makedocs(
    sitename = "moment_kinetics",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true",
                             size_threshold = 1000000,
                             size_threshold_warn = 500000,
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
