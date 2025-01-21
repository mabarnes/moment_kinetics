`plots_post_processing`
=======================

As of 20/1/2025, importing `Plots` in the Github Actions CI causes an error, so
for now we skip building the documentation for `plots_post_processing` as we
cannot import the module without its `Plots` dependency. When this is fixed,
remove the quadruple-backticks that comment out the `@autodocs` block below, in
`docs/src/zz_plot_sequence.md` and in `docs/src/zz_plot_MMS_sequence.md`, and
uncomment the lines  with `plots_post_processing` in `docs/make.jl`,
`docs/make-pdf.jl`, and `docs/Project.toml`.
````
```@autodocs
Modules = [plots_post_processing, plots_post_processing.post_processing_input, plots_post_processing.shared_utils]
```
````
