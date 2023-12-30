using Pkg, TOML

if abspath(PROGRAM_FILE) == @__FILE__
    prompt_for_hdf5 = true
    repo_dir = dirname(dirname(dirname(@__FILE__)))
    local_preferences_filename = joinpath(repo_dir, "LocalPreferences.toml")
    local_preferences = TOML.parsefile(local_preferences_filename)
    mk_preferences = local_preferences["moment_kinetics"]
else
    prompt_for_hdf5 = false
end

machine = mk_preferences["machine"]
machine_dir = joinpath(repo_dir, "machines", machine)
machine_settings_filename = joinpath(machine_dir, "machine_settings.toml")
if isfile(machine_settings_filename)
    machine_settings = TOML.parsefile(machine_settings_filename)
else
    machine_settings = Dict{String,Any}()
end


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


to_rm = String[]
if mk_preferences["use_netcdf"] == "n"
    push!(to_rm, "NCDatasets")
end
if mk_preferences["enable_mms"] == "n"
    push!(to_rm, "Symbolics", "IfElse")
end
for p ∈ to_rm
    # If `p` was not previously installed, then Pkg.rm(p) with throw an error. We only
    # need to remove if p was previously installed, so it is OK to ignore the error.
    try
        Pkg.rm(p)
    catch
    end
end
Pkg.add(["HDF5", "MPI", "MPIPreferences", "PackageCompiler", "SpecialFunctions"])


# Instantiate packages so we can use MPIPreferences below
#########################################################

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
    hdf5_dir = joinpath(artifact_dir, "hdf5-build", "lib")
    using HDF5
    HDF5.API.set_libraries!(joinpath(hdf5_dir, "libhdf5.so"),
                            joinpath(hdf5_dir, "libhdf5_hl.so"))
elseif machine_settings["hdf5_library_setting"] == "prompt"
    # Prompt user to select what HDF5 to use
    if mk_preferences["build_hdf5"] == "y"
        local_hdf5_install_dir = joinpath("machines", "artifacts", "hdf5-build", "lib")
        local_hdf5_install_dir = realpath(local_hdf5_install_dir)
        # We have downloaded and compiled HDF5, so link that
        hdf5_dir = local_hdf5_install_dir
        hdf5_lib = joinpath(local_hdf5_install_dir, "libhdf5.so")
        hdf5_lib_hl = joinpath(local_hdf5_install_dir, "libhdf5_hl.so")
    elseif !prompt_for_hdf5
        hdf5_dir = mk_preferences("hdf5_dir")
        if hdf5_dir != "default"
            hdf5_lib = joinpath(hdf5_dir, "libhdf5.so")
            hdf5_lib_hl = joinpath(hdf5_dir, "libhdf5_hl.so")
        end
    else
        println("\n** Setting up to use system HDF5\n")

        default_hdf5_dir = get(ENV, "HDF5_DIR", "") # try to find a path to a system hdf5, may not work on all systems

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
    end

    # Reload local_preferences and mk_preferences as they may have been modified by MPI
    # setup
    local_preferences_filename = joinpath(repo_dir, "LocalPreferences.toml")
    local_preferences = TOML.parsefile(local_preferences_filename)
    mk_preferences = local_preferences["moment_kinetics"]

    mk_preferences["hdf5_dir"] = hdf5_dir

    # Delete any existing preferences for HDF5 and HDF5.jll because they may prevent
    # `using HDF5` if the libraries do not exist.
    pop!(local_preferences, "HDF5", nothing)
    pop!(local_preferences, "HDF5_jll", nothing)

    open(local_preferences_filename, "w") do io
        TOML.print(io, local_preferences, sorted=true)
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


Pkg.develop(path="moment_kinetics")
Pkg.precompile()


# It seems to be important to add the dependencies for MMS before the ones for NetCDF (as
# of 30/12/2023).  Don't understand why that should be true.
if mk_preferences["enable_mms"] == "y"
    Pkg.add(["Symbolics", "IfElse"])
    Pkg.precompile()
end

if mk_preferences["use_netcdf"] == "y"
    Pkg.add("NCDatasets")
    Pkg.precompile()
end
