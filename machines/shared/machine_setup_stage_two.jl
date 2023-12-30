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
local_preferences = TOML.parsefile(joinpath(repo_dir, "LocalPreferences.toml"))
mk_preferences = local_preferences["moment_kinetics"]
machine = mk_preferences["machine"]
machine_dir = joinpath(repo_dir, "machines", machine)
machine_settings_filename = joinpath(machine_dir, "machine_settings.toml")
if isfile(machine_settings_filename)
    machine_settings = TOML.parsefile(machine_settings_filename)
else
    machine_settings = Dict{String,Any}()
end

batch_system = mk_preferences["batch_system"]

if mk_preferences["use_plots"] == "y"
    python_venv_path = joinpath(repo_dir, "machines", "artifacts", "mk_venv")
    activate_path = joinpath(python_venv_path, "bin", "activate")
    run(`bash -c "python -m venv --system-site-packages $python_venv_path; source $activate_path; PYTHONNOUSERSITE=1 pip install matplotlib"`)
    if batch_system
        open("julia.env", "a") do io
            println(io, "source $activate_path")
        end
    else
        bin_path = joinpath(repo_dir, "bin", "julia")
        contents = readlines(bin_path)
        open(bin_path, "w") do io
            println(io, contents[1])
            println(io, "source $activate_path")
            for line ∈ contents[2:end]
                println(io, line)
            end
        end
    end
end

to_add = String["HDF5", "MPI", "MPIPreferences", "SpecialFunctions"]
to_rm = String[]
if mk_preferences["use_netcdf"] == "y"
    push!(to_add, "NCDatasets")
else
    push!(to_rm, "NCDatasets")
end
if mk_preferences["enable_mms"] == "y"
    push!(to_add, "Symbolics", "IfElse")
else
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
Pkg.add(to_add)


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


Pkg.develop(path="moment_kinetics")
Pkg.precompile()


if batch_system
  # Make symlinks to batch job submission scripts
  symlink("precompile-submit.sh", joinpath("machines", "shared", "precompile-submit.sh"))
  symlink("submit-run.sh", joinpath("machines", "shared", "submit-run.sh"))
  symlink("submit-restart.sh", joinpath("machines", "shared", "submit-restart.sh"))
  if mk_preferences["use_makie"]
      symlink("precompile-makie-post-processing-submit.sh",
              joinpath("machines", "shared",
                       "precompile-makie-post-processing-submit.sh"))
  end
  if mk_preferences["use_plots"]
      symlink("precompile-plots-post-processing-submit.sh",
              joinpath("machines", "shared",
                       "precompile-plots-post-processing-submit.sh"))
  end

  if mk_preferences["submit_precompilation"] == "y"
      run(`precompile-submit.sh`)
  end
end


# Force exit so Julia must be restarted
#######################################

println()
println("************************************************************")
println("Julia must be restarted to use the updated MPI, exiting now.")
println("************************************************************")
exit(0)
