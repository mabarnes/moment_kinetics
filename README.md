# moment_kinetics
0) Ensure that the Julia version is >= 1.6.1 by doing `julia --version` at command line. 
1) To install dependencies, run 'julia -e "import Pkg; Pkg.add(\"PackageCompiler\"); Pkg.add(\"TimerOutputs\"); Pkg.add(\"NCDatasets\"); Pkg.add(\"FFTW\"); Pkg.add(\"Plots\"); Pkg.add(\"LsqFit\"); Pkg.add(\"OrderedCollections\"); Pkg.add(\"Glob\"); Pkg.add(\"NaturalSort\"); Pkg.add(\"SpecialFunctions\"); Pkg.add(\"Roots\")"'.
2) To pre-compile a static image that includes a few of the external packages required for post-processing, run 'julia precompile.jl'.
3) Create a subdirectory to store run output, 'mkdir runs'.
4) To run julia with optimization, type 'julia -O3 -Jmoment_kinetics.so moment_kinetics.jl'.  Input options can be specified in moment_kinetics_input.jl.
5) To make plots and calculate frequencies/growth rates, type 'julia -Jmoment_kinetics.so post_processing.jl'. Pass the directory to process as a command line argument. Input options for post-processing can be specified in post_processing_input.jl.
4b) Parameter scans or performance tests can be performed by running driver.jl. If running a scan, it can be parallelised by passing the number of processors as an argument. Scan options are set in scan_inputs.jl.
5b) Post processing can be done for several directories at once using 'julia -Jmoment_kinetics.so post_processing_driver.jl'. Pass the directories to process as command line arguments. Optionally pass a number as the first argument to parallelise post processing of different directories. Input options for post-processing can be specified in post_processing_input.jl.
