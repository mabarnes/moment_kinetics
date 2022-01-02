using Pkg

# Activate the moment_kinetics package
Pkg.activate(".")

using PackageCompiler

packages = [:PackageCompiler, :ArgParse, :Combinatorics, :DelimitedFiles, :FFTW, :Glob, :IJulia, :Interpolations, :LinearAlgebra, :LsqFit, :MPI, :NaturalSort, :NCDatasets, :OrderedCollections, :Plots, :Primes, :Roots, :SHA, :SpecialFunctions, :Statistics, :TOML, :TimerOutputs]

# create the sysimage 'dependencies.so' in the base moment_kinetics source directory
# with the above pre-compiled packages
create_sysimage(packages; sysimage_path="dependencies.so", precompile_execution_file="util/precompile_run_short.jl")
