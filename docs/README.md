`moment_kinetics` documentation
===============================

The docs are built automatically when a pull request is merged into the `master` branch on `github.com/mabarnes/moment_kinetics`. 

To build a local version, run `julia --project make.jl` in this directory. To see the output, open `build/index.html` with a web browser. It may be necessary when editing the doc pages to rebuild many times. Then it is more convenient to keep a REPL session open from `julia --project` and just keep running `julia> include("make.jl")` as this avoids repeated compilation of code.

It is also possible to build a pdf version of the documentation by replacing `make.jl` with `make-pdf.jl`. This requires LuaTex to be installed e.g. on Ubuntu or Mint
```
sudo apt install texlive-luatex
```
It may also need the `texlive-latex-extra` [JTO already had this installed, so hasn't tested without it].

The docs are built using [Documenter.jl](https://juliadocs.github.io/Documenter.jl/stable/).

The docs are written in Markdown, in files in the `docs/src/` subdirectory. `index.md` contains the home page and contents, there are several hand-written documentation pages `docs/src/input_options.md`, etc., and there is a file for each module (with `docs/src/zz_foo.md` corresponding to the module defined in `src/foo.jl`). Each module page contains at minimum the docs auto-generated from the docstrings in the Julia source code - additional content can be added in the `*.md` files as needed. The `zz_` prefix for the module pages is so that the pages are ordered nicely in the sidebar of the docs - in the sidebar the entries are ordered by filename, so using the `zz_` prefix for module pages ensures they are found together below the hand-written pages. For extended syntax for documenting Julia code, and including LaTeX-syntax expressions, see the [Documenter.jl online documentation](https://juliadocs.github.io/Documenter.jl/stable/).

Docstrings should be formatted following the [guidelines for Julia documentation](https://docs.julialang.org/en/v1/manual/documentation/#man-documentation).
