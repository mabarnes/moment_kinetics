"""
Build the documentation as a pdf using LaTeX

Note this requires LuaTeX to be installed, e.g. on Ubuntu or Mint
```
sudo apt install texlive-luatex
```
It may also need the `texlive-latex-extra` [JTO: already had this installed, so haven't
tested without it].
"""

using Documenter
using moment_kinetics

makedocs(
    sitename = "momentkinetics",
    format = Documenter.LaTeX(),
    modules = [moment_kinetics],
    authors = "M. Barnes, J.T. Omotani, M. Hardman",
    pages = ["moment_kinetic_equations.md"]
)
