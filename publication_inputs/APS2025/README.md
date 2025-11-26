* Input files for the simulations analysed by scripts in this directory are in
  `publication_inputs/2D-instability`. These inputs may be updated in future,
  so to reproduce plots it might be necessary to revert those inputs to the Git
  commit used for this analysis.
* To produce plots, first run the necessary simulations (see the script), then
  run (from the top level of the `moment_kinetics` repo)
  `publication_inputs/APS2025/poster-plots-APS2025.jl`.
    * To make publication-quality plots, check the resolution setting (near the
      top of the script) is set to high quality, for the default .png output,
      or swap the output format to .pdf (if adapting the script to make figures
      for a paper).
* To make the summary document, use LyX to open
  `publication_inputs/APS2025/poster-plots-APS2025.lyx` and export a pdf.
