using Pkg

# Activate the moment_kinetics package
Pkg.activate(".")

using PackageCompiler

# create the sysimage 'dependencies.so' in the base moment_kinetics source directory
# with the above pre-compiled packages
create_sysimage(; sysimage_path="dependencies.so",
                precompile_execution_file="util/precompile_run_short.jl",
                include_transitive_dependencies=false, # This is needed to make MPI work, see https://github.com/JuliaParallel/MPI.jl/issues/518
                sysimage_build_args=`-O3 --check-bounds=no`, # Assume if we are precompiling we want an optimized, production build
               )
