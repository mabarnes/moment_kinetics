# moment_kinetics
1) To pre-compile a static image that includes a few of the external packages required for post-processing, run 'julia precompile.jl'.
2) To run julia with optimization, type 'julia -O3 -Jmoment_kinetics.so moment_kinetics.jl'.  Input options can be specified in moment_kinetics_input.jl.
3) To make plots and calculate frequencies/growth rates, type 'julia -Jmoment_kinetics.so post_processing.jl'.  Input options for post-processing can be specified in post_processing_input.jl.