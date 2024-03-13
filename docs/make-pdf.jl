"""
Build the documentation as a pdf using LaTeX

Note this requires LuaTeX to be installed, e.g. on Ubuntu or Mint
```
sudo apt install texlive-luatex
```
It may also need the `texlive-latex-extra` [JTO: already had this installed, so haven't
tested without it].
"""

using Pkg

repo_dir = dirname(dirname(@__FILE__))
Pkg.develop([PackageSpec(path=joinpath(repo_dir, "moment_kinetics")),
             PackageSpec(path=joinpath(repo_dir, "makie_post_processing", "makie_post_processing")),
             PackageSpec(path=joinpath(repo_dir, "plots_post_processing", "plots_post_processing"))])
Pkg.instantiate()

using Documenter
using Glob
using moment_kinetics, makie_post_processing, plots_post_processing

doc_files = glob("src/*.md")

# Remove the src/ prefix
doc_files = [basename(s) for s ∈ doc_files]

if get(ENV, "CI", nothing) == "true"
    latex_kwargs = (platform = "docker",)
else
    latex_kwargs = ()
end

makedocs(
    sitename = "momentkinetics",
    format = Documenter.LaTeX(; latex_kwargs...),
    modules = [moment_kinetics, makie_post_processing, plots_post_processing],
    authors = "M. Barnes, J.T. Omotani, M. Hardman",
    pages = doc_files
)
