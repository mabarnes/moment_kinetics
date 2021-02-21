# moment_kinetics
1) To install dependencies, run 'julia -e "import Pkg; Pkg.add(\"PackageCompiler\"); Pkg.add(\"TimerOutputs\"); Pkg.add(\"NCDatasets\"); Pkg.add(\"FFTW\"); Pkg.add(\"Plots\"); Pkg.add(\"LsqFit\"); Pkg.add(\"OrderedCollections\"); Pkg.add(\"Glob\"); Pkg.add(\"NaturalSort\")"'.
2) To pre-compile a static image that includes a few of the external packages required for post-processing, run 'julia precompile.jl'.
3) Create a subdirectory to store run output, 'mkdir runs'.
4) To run julia with optimization, type 'julia -O3 -Jmoment_kinetics.so driver.jl'.  Input options can be specified in moment_kinetics_input.jl. If running a scan, it can be parallelised by passing the number of processors as an argument.
5) To make plots and calculate frequencies/growth rates, type 'julia -Jmoment_kinetics.so post_processing.jl'.  Input options for post-processing can be specified in post_processing_input.jl.
