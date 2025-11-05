using Pkg, TOML

if abspath(PROGRAM_FILE) == @__FILE__
    prompt_for_lib_paths = true
    repo_dir = dirname(Pkg.project().path)
    project_dir = repo_dir
    local_preferences_filename = joinpath(repo_dir, "LocalPreferences.toml")
    local_preferences = TOML.parsefile(local_preferences_filename)
    mk_preferences = local_preferences["moment_kinetics"]
else
    prompt_for_lib_paths = false
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
to_add = ["HDF5", "MPI", "MPIPreferences", "PackageCompiler", "SpecialFunctions"]
if !mk_preferences["batch_system"] && mk_preferences["use_revise"] == "y"
    push!(to_add, "Revise")
end
Pkg.add(to_add)


# Instantiate packages so we can use MPIPreferences below
#########################################################

Pkg.instantiate()
Pkg.resolve()


# MPI setup
###########

if mk_preferences["use_system_mpi"] == "y"
    println("\n** Setting up to use system MPI\n")
    using MPIPreferences

    if "mpi_library_names" ∈ keys(machine_settings) || "mpiexec" ∈ keys(machine_settings)
        MPIPreferences.use_system_binary(library_names=machine_settings["mpi_library_names"],
                                         mpiexec=machine_settings["mpiexec"])
    elseif Sys.isapple()
        # On macOS, MPIPreferences.use_system_binary() does not automatically find the MPI
        # library when MPI was installed with homebrew, so prompt the user for the library
        # path instead.
        # ?? Could we attempt to auto-detect the MPI library before prompting the user??
        if prompt_for_lib_paths
            try
                # See if MPIPreferences can auto-detect the system MPI library path
                MPIPreferences.use_system_binary()
            catch
                println("Failed to auto-detect path of MPI library...")

                local mpi_library_path

                default_mpi_library_path = get(mk_preferences, "mpi_library_path", "")
                mpi_library_path = get_input_with_path_completion(
                    "\nEnter the full path to your MPI library (e.g. something like "
                    * "'libmpi.dylib'): [$default_mpi_library_path]")
                if mpi_library_path == ""
                    mpi_library_path = default_mpi_library_path
                end

                MPIPreferences.use_system_binary(library_names=mpi_library_path)

                global mk_preferences, local_preferences

                # Just got the value for the setting, now write it to LocalPreferences.toml,
                # but first reload the preferences from the LocalPreferences.toml file so that
                # we don't overwrite the values that MPIPreferences has set.
                local_preferences = TOML.parsefile(local_preferences_filename)
                mk_preferences = local_preferences["moment_kinetics"]
                mk_preferences["mpi_library_path"] = mpi_library_path
                open(local_preferences_filename, "w") do io
                    TOML.print(io, local_preferences, sorted=true)
                end
                # Re-read local_preferences file, so we can modify it again below, keeping the
                # changes here
                local_preferences = TOML.parsefile(local_preferences_filename)
                mk_preferences = local_preferences["moment_kinetics"]
            end
        else
            if "mpi_library_path" ∈ keys(mk_preferences)
                mpi_library_path = mk_preferences["mpi_library_path"]
                MPIPreferences.use_system_binary(library_names=mpi_library_path)
            else
                # Must have auto-detected MPI library before, so do the same here
                MPIPreferences.use_system_binary()
            end
        end
    else
        # If settings for MPI library are not given explicitly, then auto-detection by
        # MPIPreferences.use_system_binary() should work.
        MPIPreferences.use_system_binary()
    end
else
    using MPI
    MPI.install_mpiexecjl(; destdir=project_dir, force=true)
end


# HDF5 setup
############

function get_hdf5_lib_names(dirname)
    if Sys.isapple()
        libhdf5_name = joinpath(dirname, "libhdf5.dylib")
        libhdf5_hl_name = joinpath(dirname, "libhdf5_hl.dylib")
    else
        libhdf5_name = joinpath(dirname, "libhdf5.so")
        libhdf5_hl_name = joinpath(dirname, "libhdf5_hl.so")
    end
    return libhdf5_name, libhdf5_hl_name
end

if mk_preferences["use_system_mpi"] == "y"
    # Only need to do this if using 'system MPI'. If we are using the Julia-provided MPI,
    # then the Julia-provided HDF5 is already MPI-enabled
    println("\n** Setting up to use system HDF5\n")

    if machine_settings["hdf5_library_setting"] == "system"
	if "HDF5_LIB" ∈ keys(ENV)
            hdf5_dir = ENV["HDF5_LIB"] # system hdf5
        else
            hdf5_dir = joinpath(ENV["HDF5_DIR"], "lib") # system hdf5
        end
        using HDF5
        HDF5.API.set_libraries!(get_hdf5_lib_names(hdf5_dir)...)
    elseif machine_settings["hdf5_library_setting"] == "download"
        artifact_dir = joinpath(repo_dir, "machines", "artifacts")
        hdf5_dir = joinpath(artifact_dir, "hdf5-build", "lib")
        using HDF5
        HDF5.API.set_libraries!(get_hdf5_lib_names(hdf5_dir)...)
    elseif machine_settings["hdf5_library_setting"] == "prompt"
        # Prompt user to select what HDF5 to use
        if mk_preferences["build_hdf5"] == "y"
            local_hdf5_install_dir = joinpath("machines", "artifacts", "hdf5-build", "lib")
            local_hdf5_install_dir = realpath(local_hdf5_install_dir)
            # We have downloaded and compiled HDF5, so link that
            hdf5_dir = local_hdf5_install_dir
            hdf5_lib, hdf5_lib_hl = get_hdf5_lib_names(local_hdf5_install_dir)
        elseif !prompt_for_lib_paths
            hdf5_dir = mk_preferences["hdf5_dir"]
            if hdf5_dir != "default"
                hdf5_lib, hdf5_lib_hl = get_hdf5_lib_names(hdf5_dir)
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
                    * "parallel I/O. Enter the directory where the libhdf5.so and\n"
                    * "libhdf5_hl.so (or libhdf5.dylib and libhdf5_hl.dylib on macOS)\n"
                    * "are located (enter 'default' to use the Julia-provided HDF5, which\n"
                    * "is not compatible with using the system MPI): [$default_hdf5_dir]")

                if hdf5_dir == ""
                    hdf5_dir = default_hdf5_dir
                end

                if hdf5_dir == "default"
                    break
                end

                if isdir(hdf5_dir)
                    hdf5_dir = realpath(hdf5_dir)
                end
                hdf5_lib, hdf5_lib_hl = get_hdf5_lib_names(hdf5_dir)
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
        local_preferences_filename = joinpath(project_dir, "LocalPreferences.toml")
        local_preferences = TOML.parsefile(local_preferences_filename)
        if abspath(PROGRAM_FILE) == @__FILE__
            # Only need to do this for the top-level project. When adding dependencies to
            # makie_post_processing or plots_post_processing, do not need to set "hdf5_dir" in
            # `mk_preferences` to go in the `moment_kinetics` section - this only needs to be
            # done for the top-level project.
            mk_preferences = local_preferences["moment_kinetics"]

            mk_preferences["hdf5_dir"] = hdf5_dir
        end

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
