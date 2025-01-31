using Pkg

using PackageCompiler

# Create the sysimage 'makie_postproc.so' in the base moment_kinetics source directory
# with both moment_kinetics and the dependencies listed above precompiled.
# Warning: editing the code will not affect what runs when using this .so, you
# need to re-precompile if you change anything.
create_sysimage(; sysimage_path="makie_postproc.so",
                precompile_execution_file="util/precompile_makie_plots.jl",
                include_transitive_dependencies=false, # This is needed to make MPI work, see https://github.com/JuliaParallel/MPI.jl/issues/518
                sysimage_build_args=`-O3`, # Assume if we are precompiling we want an optimized, production build. No `--check-bounds=no` because Makie doesn't like it https://github.com/MakieOrg/Makie.jl/issues/3132
               )
