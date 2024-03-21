`moment_kinetics` documentation
===============================

The docs are built automatically when a pull request is merged into the `master` branch on `github.com/mabarnes/moment_kinetics`. 

To build a local version, run `julia --project make.jl` in this directory. To see the output, open `build/index.html` with a web browser.

The docs are built using [Documenter.jl](https://juliadocs.github.io/Documenter.jl/stable/).

The docs are written in Markdown, in files in the `docs/src/` subdirectory. `index.md` contains the home page and contents, and there is a file for each module (with `docs/src/foo.md` corresponding to the module defined in `src/foo.jl`). Each module page contains at minimum the docs auto-generated from the docstrings in the Julia source code - additional content can be added in the `*.md` files as needed. For extended syntax for documenting Julia code, and including LaTeX-syntax expressions, see the [Documenter.jl online documentation](https://juliadocs.github.io/Documenter.jl/stable/).

Docstrings should be formatted following the [guidelines for Julia documentation](https://docs.julialang.org/en/v1/manual/documentation/#man-documentation).
