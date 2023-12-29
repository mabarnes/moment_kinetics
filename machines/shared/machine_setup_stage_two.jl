using Pkg, TOML


"""
    get_input_with_path_completion(message=nothing)

Print `message` and get user input using the `read` utility from bash to allow
path-completion.

Solution adapted from
https://discourse.julialang.org/t/collecting-all-output-from-shell-commands/15592/2
"""
function get_input_with_path_completion(message=nothing)
    if message !== nothing
        println(message)
    end

    # The `out` object is used by `pipeline()` to capture output from the shell command.
    out = Pipe()

    # Use Julia's shell command functionality to actually run bash - not sure how to use
    # `read` and get output from it without using bash.
    run(pipeline(`bash -c "read -e -p '> ' USERINPUT; echo \$USERINPUT"`, stdout=out))

    # Need to close `out.in` to be able to read from `out`.
    close(out.in)

    # chomp removes the trailing '\n'. 'String()' converts Vector{Char} to a String.
    input = chomp(String(read(out)))

    return input
end


repo_dir = dirname(dirname(dirname(@__FILE__)))
local_settings = TOML.parsefile(joinpath(repo_dir, "LocalPreferences.toml"))["moment_kinetics"]
machine = local_settings["machine"]
machine_dir = joinpath(repo_dir, "machines", machine)
machine_settings_filename = joinpath(machine_dir, "machine_settings.toml")
if isfile(machine_settings_filename)
    machine_settings = TOML.parsefile(machine_settings_filename)
else
    machine_settings = Dict{String,Any}()
end


# Instantiate packages so we can use MPIPreferences below
#########################################################

println("\n** Getting dependencies\n")
Pkg.instantiate()
Pkg.resolve()


# MPI setup
###########

println("\n** Setting up to use system MPI\n")
using MPIPreferences

if "mpi_library_names" ∈ keys(machine_settings) || "mpiexec" ∈ keys(machine_settings)
    MPIPreferences.use_system_binary(library_names=machine_settings["mpi_library_names"],
                                     mpiexec=machine_settings["mpiexec"])
else
    # If settings for MPI library are not given explicitly, then auto-detection by
    # MPIPreferences.use_system_binary() should work.
    MPIPreferences.use_system_binary()
end


# HDF5 setup
############

println("\n** Setting up to use system HDF5\n")

if machine_settings["hdf5_library_setting"] == "system"
    hdf5_dir = ENV["HDF5_DIR"] # system hdf5
    using HDF5
    HDF5.API.set_libraries!(joinpath(hdf5_dir, "libhdf5.so"),
                            joinpath(hdf5_dir, "libhdf5_hl.so"))
elseif machine_settings["hdf5_library_setting"] == "download"
    artifact_dir = joinpath(repo_dir, "machines", "artifacts")
    hdf5_dir = joinpath(artifact_dir, "hdf5-build/")
    using HDF5
    HDF5.API.set_libraries!(joinpath(hdf5_dir, "libhdf5.so"),
                            joinpath(hdf5_dir, "libhdf5_hl.so"))
elseif machine_settings["hdf5_library_setting"] == "prompt"
    # Prompt user to select what HDF5 to use
    local_hdf5_install_dir = joinpath("machines", "artifacts", "hdf5-build", "lib")
    if isdir(local_hdf5_install_dir)
        local_hdf5_install_dir = realpath(local_hdf5_install_dir)
        # We have downloaded and compiled HDF5, so link that
        hdf5_dir = local_hdf5_install_dir
        hdf5_lib = joinpath(local_hdf5_install_dir, "libhdf5.so")
        hdf5_lib_hl = joinpath(local_hdf5_install_dir, "libhdf5_hl.so")
    else
        println("\n** Setting up to use system HDF5\n")

        default_hdf5_dir = get(ENV, "HDF5_DIR", "") # try to find a path to a system hdf5, may not work on all systems

        using TOML
        repo_dir = dirname(dirname(dirname(@__FILE__)))
        local_preferences_filename = joinpath(repo_dir, "LocalPreferences.toml")
        if ispath(local_preferences_filename)
            local_preferences = TOML.parsefile(local_preferences_filename)
        else
            local_preferences = Dict{String,Any}()
        end
        mk_preferences = get(local_preferences, "moment_kinetics", Dict{String,String}())
        println("mk_preferences ", mk_preferences)
        default_hdf5_dir = get(mk_preferences, "hdf5_dir", default_hdf5_dir)

        hdf5_dir = ""
        hdf5_lib = ""
        hdf5_lib_hl = ""
        while true
            global hdf5_dir, hdf5_lib, hdf5_lib_hl
            hdf5_dir = get_input_with_path_completion(
                "\nAn HDF5 installation compiled with your system MPI is required to use\n"
                * "parallel I/O. Enter the directory where the libhdf5.so and libhdf5_hl.so are\n"
                * "located (enter 'default' to use the Julia-provided HDF5, which does not\n"
                * "support parallel I/O): [$default_hdf5_dir]")

            if hdf5_dir == ""
                hdf5_dir = default_hdf5_dir
            end

            if hdf5_dir == "default"
                break
            end

            if isdir(hdf5_dir)
                hdf5_dir = realpath(hdf5_dir)
            end
            hdf5_lib = joinpath(hdf5_dir, "libhdf5.so")
            hdf5_lib_hl = joinpath(hdf5_dir, "libhdf5_hl.so")
            if isfile(hdf5_lib) && isfile(hdf5_lib_hl)
                break
            else
                # Remove trailing slash if it exists so that we can print a single trailing slash
                # consistently
                hdf5_dir = rstrip(hdf5_dir, '/')
                print("HDF5 libraries not found in '$hdf5_dir/'.")
                if !isfile(hdf5_lib)
                    print(" $hdf5_lib does not exist.")
                end
                if !isfile(hdf5_lib_hl)
                    print(" $hdf5_lib_hl does not exist.")
                end
            end
        end

        mk_preferences["hdf5_dir"] = hdf5_dir
        open(local_preferences_filename, "w") do io
            TOML.print(io, local_preferences, sorted=true)
        end
    end
    using HDF5
    if hdf5_dir == "default"
        HDF5.API.set_libraries!()
    else
        HDF5.API.set_libraries!(hdf5_lib, hdf5_lib_hl)
    end
else
    error("Unrecognized setting "
          * "hdf5_library_setting=$(machine_settings["hdf5_library_setting"])")
end


# Force exit so Julia must be restarted
#######################################

println()
println("************************************************************")
println("Julia must be restarted to use the updated MPI, exiting now.")
println("************************************************************")
exit(0)
