# Ensure matplotlib is installed in the system Python
#####################################################

println("\n** Installing matplotlib into the system Python\n")
run(`pip3 install --user matplotlib`)


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
ENV["JULIA_HDF5_PATH"] = joinpath(artifact_dir, "hdf5-build/")
Pkg.build()


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
