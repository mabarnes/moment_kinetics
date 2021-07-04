using Pkg

# Activate the moment_kinetics package
Pkg.activate(".")

using PackageCompiler

packages = [:PackageCompiler, :TimerOutputs, :NCDatasets, :FFTW, :Plots, :LsqFit, :OrderedCollections, :Glob, :NaturalSort, :SpecialFunctions, :Roots, :Revise]

# create the sysimage 'moment_kinetics.so' in the base moment_kinetics source directory
# with the above pre-compiled packages
create_sysimage(packages; sysimage_path="dependencies.so", precompile_execution_file="util/precompile_run.jl")
