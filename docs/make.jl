using Documenter
using moment_kinetics

makedocs(
    sitename = "moment_kinetics",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    modules = [moment_kinetics]
)

if get(ENV, "CI", nothing) == "true"
    # Documenter can also automatically deploy documentation to gh-pages.
    # See "Hosting Documentation" and deploydocs() in the Documenter manual
    # for more information.
    deploydocs(
        repo = "github.com/johnomotani/moment_kinetics",
        push_preview = true,
    )
end
