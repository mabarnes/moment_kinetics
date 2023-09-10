using Pkg

# Activate the moment_kinetics package
Pkg.activate(".")

using PackageCompiler

# Create the sysimage 'moment_kinetics.so' in the base moment_kinetics source directory
# with both moment_kinetics and the dependencies listed above precompiled.
# Warning: editing the code will not affect what runs when using this .so, you
# need to re-precompile if you change anything.
create_sysimage(; sysimage_path="moment_kinetics.so",
                include_transitive_dependencies=false, # This is needed to make MPI work, see https://github.com/JuliaParallel/MPI.jl/issues/518
                sysimage_build_args=`-O3 --check-bounds=no`, # Assume if we are precompiling we want an optimized, production build
               )
