using Pkg

# Activate the moment_kinetics package
Pkg.activate(".")

using PackageCompiler

packages = [:moment_kinetics, :PackageCompiler, :ArgParse, :CairoMakie, :Combinatorics, :DelimitedFiles, :FFTW, :Glob, :HDF5, :IJulia, :LinearAlgebra, :LsqFit, :MPI, :NaturalSort, :NCDatasets, :OrderedCollections, :Primes, :Roots, :SHA, :SpecialFunctions, :Statistics, :TOML, :TimerOutputs]

# Create the sysimage 'makie_postproc.so' in the base moment_kinetics source directory
# with both moment_kinetics and the dependencies listed above precompiled.
# Warning: editing the code will not affect what runs when using this .so, you
# need to re-precompile if you change anything.
create_sysimage(packages;
                sysimage_path="makie_postproc.so",
                precompile_execution_file="util/precompile_makie_plots.jl",
                include_transitive_dependencies=false, # This is needed to make MPI work, see https://github.com/JuliaParallel/MPI.jl/issues/518
               )
