using Pkg

# Activate the moment_kinetics package
Pkg.activate(".")

using PackageCompiler

using TOML
project_file = TOML.parsefile("Project.toml")
deps = (Symbol(d) for d ∈ keys(project_file["deps"]))

packages = [:moment_kinetics, :PackageCompiler, deps...]
println("precompling $packages")

# Create the sysimage 'moment_kinetics.so' in the base moment_kinetics source directory
# with both moment_kinetics and the dependencies listed above precompiled.
# Warning: editing the code will not affect what runs when using this .so, you
# need to re-precompile if you change anything.
create_sysimage(packages;
                sysimage_path="moment_kinetics.so",
                include_transitive_dependencies=false, # This is needed to make MPI work, see https://github.com/JuliaParallel/MPI.jl/issues/518
                sysimage_build_args=`-O3 --check-bounds=no`, # Assume if we are precompiling we want an optimized, production build
               )
