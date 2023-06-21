using Pkg

# Ensure matplotlib is installed in the system Python
#####################################################

println("\n** Installing matplotlib into the system Python\n")
run(`pip install matplotlib`)


# Instantiate packages so we can use MPIPreferences below
#########################################################

println("\n** Getting dependencies\n")
Pkg.instantiate()
Pkg.resolve()


# HDF5 setup
############

println("\n** Setting up to use system HDF5\n")
ENV["JULIA_HDF5_PATH"] = ENV["HDF5_DIR"] # system hdf5
Pkg.build()


# MPI setup
###########

println("\n** Setting up to use system MPI\n")
using MPIPreferences

# For GNU compilers.
# Working using the versions as of 19/6/2023, with the following modules loaded:
#     1) craype/2.7.19              5) cray-libsci/22.12.1.1   9) cray-python/3.9.13.1
#     2) cray-dsmml/0.2.2           6) PrgEnv-cray/8.3.3      10) cray-hdf5-parallel/1.12.2.1
#     3) libfabric/1.12.1.2.2.0.0   7) gcc/11.2.0
#     4) craype-network-ofi         8) cray-mpich/8.1.23
MPIPreferences.use_system_binary(library_names="libmpi_gnu_91", mpiexec="srun")

# For Cray compilers.
# As of 19/6/2023 does not seem to work (jobscript-precompile hangs while running
# precompile.jl, with no terminal output). Had the following modules loaded:
#     1) cce/15.0.0                 5) craype-network-ofi      9) cray-python/3.9.13.1
#     2) craype/2.7.19              6) cray-mpich/8.1.23      10) cray-hdf5-parallel/1.12.2.1
#     3) cray-dsmml/0.2.2           7) cray-libsci/22.12.1.1
#     4) libfabric/1.12.1.2.2.0.0   8) PrgEnv-cray/8.3.3
#MPIPreferences.use_system_binary(library_names="libmpi_cray", mpiexec="srun")


# Force exit so Julia must be restarted
#######################################

println()
println("************************************************************")
println("Julia must be restarted to use the updated MPI, exiting now.")
println("************************************************************")
exit(0)
