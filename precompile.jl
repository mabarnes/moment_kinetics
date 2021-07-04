using Pkg

# Activate the moment_kinetics package
Pkg.activate(".")

using PackageCompiler

packages = [:moment_kinetics, :PackageCompiler, :TimerOutputs, :NCDatasets, :FFTW, :Plots, :LsqFit, :OrderedCollections, :Glob, :NaturalSort, :SpecialFunctions, :Roots, :TOML]

# Create the sysimage 'moment_kinetics.so' in the base moment_kinetics source directory
# with both moment_kinetics and the dependencies listed above precompiled.
# Warning: editing the code will not affect what runs when using this .so, you
# need to re-precompile if you change anything.
create_sysimage(packages; sysimage_path="moment_kinetics.so", precompile_execution_file="util/precompile_run.jl")
