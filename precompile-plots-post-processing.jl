using Pkg

using PackageCompiler

# Create the sysimage 'makie_postproc.so' in the base moment_kinetics source directory
# with both moment_kinetics and the dependencies listed above precompiled.
# Warning: editing the code will not affect what runs when using this .so, you
# need to re-precompile if you change anything.
create_sysimage(; sysimage_path="plots_postproc.so",
                precompile_execution_file="util/precompile_plots_plots.jl",
                include_transitive_dependencies=false, # This is needed to make MPI work, see https://github.com/JuliaParallel/MPI.jl/issues/518
                sysimage_build_args=`-O3`, # Assume if we are precompiling we want an optimized, production build. For post-processing, we probably do not want `--check-bounds=no` even if we start using it again for simulations.
               )
