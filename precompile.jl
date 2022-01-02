using Pkg

# Activate the moment_kinetics package
Pkg.activate(".")

using PackageCompiler

packages = [:moment_kinetics, :PackageCompiler, :ArgParse, :Combinatorics, :DelimitedFiles, :FFTW, :Glob, :IJulia, :Interpolations, :LinearAlgebra, :LsqFit, :MPI, :NaturalSort, :NCDatasets, :OrderedCollections, :Plots, :Primes, :Roots, :SHA, :SpecialFunctions, :Statistics, :TOML, :TimerOutputs]

# Create the sysimage 'moment_kinetics.so' in the base moment_kinetics source directory
# with both moment_kinetics and the dependencies listed above precompiled.
# Warning: editing the code will not affect what runs when using this .so, you
# need to re-precompile if you change anything.
create_sysimage(packages;
                sysimage_path="moment_kinetics.so",
                precompile_execution_file="util/precompile_run.jl",
                include_transitive_dependencies=false, # This is needed to make MPI work, see https://github.com/JuliaParallel/MPI.jl/issues/518
               )
