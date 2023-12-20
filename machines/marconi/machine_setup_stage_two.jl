# Instantiate packages so we can use MPIPreferences below
#########################################################

using Pkg

println("\n** Getting dependencies\n")
Pkg.instantiate()
Pkg.resolve()


repo_dir = dirname(dirname(dirname(@__FILE__)))
artifact_dir = joinpath(repo_dir, "machines", "artifacts")

# HDF5 setup
############

println("\n** Setting up to use custom compiled HDF5\n")
hdf5_dir = joinpath(artifact_dir, "hdf5-build/")
using HDF5
HDF5.API.set_libraries!(joinpath(hdf5_dir, "libhdf5.so"), joinpath(hdf5_dir, "libhdf5_hl.so"))


# MPI setup
###########

println("\n** Setting up to use system MPI\n")
using MPIPreferences

MPIPreferences.use_system_binary()


# Force exit so Julia must be restarted
#######################################

println()
println("************************************************************")
println("Julia must be restarted to use the updated MPI, exiting now.")
println("************************************************************")
exit(0)
