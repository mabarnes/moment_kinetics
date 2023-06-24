"""
Functions to help setting up on known machines
"""
module machine_setup

export machine_setup_moment_kinetics

using TOML

# Default settings for the arguments to machine_setup_moment_kinetics(), set like this
# so that they can be passed to the bash script `machine_setup.sh`
default_settings = Dict{String,Dict{String,String}}()
default_settings["base"] = Dict{String,String}()
default_settings["archer"] = default_settings["base"]
default_settings["marconi"] = default_settings["base"]

"""
    machine_setup_moment_kinetics(machine::String,
                                  account::String,
                                  julia_directory::String;
                                  no_force_exit::Bool=false,
                                  interactive::Bool=true)

Do setup for a known `machine`:
* On clusters that use a module system, provide `julia.env` at the top level of the
  moment_kinetics repo.

  Call
  ```shell
  source julia.env
  ```
  to get the correct modules for running moment_kinetics, either on the command line (to
  get them for the current session) or in your `.bashrc` (to get them by default). Note
  that this calls `module purge` so will remove any currently loaded modules when it is
  run.
* Run setup commands for MPI and HDF5 which ensure the correct, system-provided
  libraries are used.
* Makes a symlink to the Julia exeutable used to run this command at `bin/julia` under
  the moment_kinetics repo, so that setup and job submission scripts can use a known
  relative path.
  !!! note
      If you change the Julia executable, e.g. to update to a new verison, you will need
      to either replace the symlink `<moment_kinetics>/bin/julia` by hand, or re-run
      this function using the new executable.

`julia_directory` gives the location of the directory (usually called `.julia`) where
Julia installs files, saves settings, etc. `julia_directory` must be passed if this
directory should be in a non-default location (i.e. not `\$HOME/.julia/`). The value is
used to set `JULIA_DEPOT_PATH` in the `julia.env` file, so that this setting is
propagated to the environment on the compute nodes.

Usually it is necessary for Julia to be restarted after running this function to ensure
the correct MPI is linked, etc. so the function will force Julia to exit. If for some
reason this is not desired (e.g. when debugging), pass `no_force_exit=true`.

The `interactive` argument exists so that when this function is called from another
script, terminal output with instructions for the next step can be disabled.

Currently supported machines:
* `"archer"` - the UK supercomputer [ARCHER2](https://www.archer2.ac.uk/)
* `"marconi"` - the EUROfusion supercomputer
    [Marconi](https://wiki.u-gov.it/confluence/display/SCAIUS/UG3.1%3A+MARCONI+UserGuide)

Notes:
* The settings created by this function are saved in LocalPreferences.toml (using the
  `Preferences.jl` package). It might sometimes be useful to edit these by hand (e.g.
  the `account` setting if this needs to be changed.): it is fine to do this.
"""
function machine_setup_moment_kinetics(machine::String,
                                       account::String,
                                       julia_directory::String;
                                       no_force_exit::Bool=false,
                                       interactive::Bool=true)

    repo_dir = dirname(dirname(dirname(@__FILE__)))

    # Common operations that only depend on the name of `machine`
    #############################################################

    println("\n** Creating `julia.env` for environment setup\n")
    envname = joinpath(repo_dir, "julia.env")
    # Read the template
    template = read("machines/$machine/julia.env")
    # Write julia.env, overwriting if it already exists
    ispath(envname) && rm(envname)
    open(envname, "w") do io
        write(io, template)
        if julia_directory != ""
            println("\n** Setting JULIA_DEPOT_PATH=$julia_directory in `julia.env`\n")
            println(io, "\nexport JULIA_DEPOT_PATH=$julia_directory")
        end
    end

    # Make a local link to the Julia binary so scripts in the repo can find it
    println("\n** Making a symlink to the julia executable at bin/julia\n")
    bindir = joinpath(repo_dir, "bin")
    mkpath(bindir)
    julia_executable_name = joinpath(bindir, "julia")
    islink(julia_executable_name) && rm(julia_executable_name)
    symlink(joinpath(Sys.BINDIR, "julia"), julia_executable_name)

    # Write these preferences into a [moment_kinetics] section in LocalPreferences.toml
    #
    # Load and re-write LocalPreferences.toml directly here to avoid needing to import
    # the Preferences.jl package, which would need to be installed (TOML.jl is available
    # as part of the Julia system). This is a bit hacky, but hopefully no need to do
    # anything fancy here!
    println("\n** Adding system-specific settings for moment_kinetics to LocalPreferences.toml\n")
    local_preferences_filename = joinpath(repo_dir, "LocalPreferences.toml")
    if ispath(local_preferences_filename)
        local_preferences = TOML.parsefile(local_preferences_filename)
    else
        local_preferences = Dict{String,Any}()
    end
    # Always overwrite any existing preferences, to get a fresh setup
    mk_preferences = local_preferences["moment_kinetics"] = Dict{String,Any}()
    mk_preferences["machine"] = machine
    mk_preferences["account"] = account
    open(local_preferences_filename, "w") do io
        TOML.print(io, local_preferences, sorted=true)
    end

    # If it is necessary to run a shell script to compile dependencies, set
    # this flag to true.
    compile_dependencies_relative_path = joinpath("machines", "shared",
                                                  "compile_dependencies.sh")
    compile_dependencies_path = joinpath(repo_dir,
                                         compile_dependencies_relative_path)
    needs_compile_dependencies = false

    # A second stage of setup may be needed after restarting Julia on some machines.
    # If it is, set `needs_second_stage = true` in the machine-specific case below.
    second_stage_relative_path = joinpath("machines", "shared",
                                          "machine_setup_stage_two.jl")
    second_stage_path = joinpath(repo_dir, second_stage_relative_path)
    needs_second_stage = false

    # Set this flag to true in the machine-specific branch below to require a
    # non-empty `account` setting
    needs_account = false

    if machine == "archer"
        needs_account = true
        if julia_directory == ""
            error("On ARCHER2, the `julia_directory` setting is required, because the "
                  * "default location for the `.julia` directory is in your home "
                  * "directory and the `/home/` filesystem is not available on the "
                  * "compute nodes.")
        end

        # Need to set JULIA_DEPOT_PATH so the `.julia` directory is on the /work
        # filesystem (where it can be used on compute nodes, unlike /home) before
        # setting up MPI and HDF5, so a second stage is required for archer.
        needs_second_stage = true
    elseif machine == "marconi"
        needs_account = true

        # For marconi, need to run a script to compile HDF5
        needs_compile_dependencies = true

        # Second stage is required for marconi to set up HDF5 and MPI
        needs_second_stage = true
    else
        error("Unsupported machine '$machine'")
    end

    if needs_account && account == ""
        error("For machine=\"$machine\" it is required to pass a value for the "
              * "`account` argument.")
    end

    if needs_compile_dependencies
        # Remove link if it exists already
        islink(compile_dependencies_path) && rm(compile_dependencies_path)

        symlink(joinpath("..", machine, "compile_dependencies.sh"), compile_dependencies_path)

        # Create directory to download/compile dependencies in
        artifact_path = joinpath("machines", "artifacts")
        mkpath(artifact_path)

        if interactive
            println()
            println("***********************************************************************")
            println("To compile dependencies run:")
            println("    \$ machines/shared/compile_dependencies.sh")
            println("***********************************************************************")
        end
    end

    if needs_second_stage
        # Remove link if it exists already
        islink(second_stage_path) && rm(second_stage_path)

        symlink(joinpath("..", machine, "machine_setup_stage_two.jl"), second_stage_path)

        if interactive
            println()
            println("***********************************************************************")
            println("To complete setup, first `source julia.env` (so that `JULIA_DEPOT_PATH`")
            println("is set correctly, then start Julia again (you can now use ")
            println("`bin/julia --project`) and to complete the setup run:")
            println("    julia> include(\"$second_stage_relative_path\")")
            println("***********************************************************************")
        end
    end

    if !no_force_exit
        exit(0)
    end

    return nothing
end

end

using .machine_setup

if abspath(PROGRAM_FILE) == @__FILE__
    # Allow the command to be called as a script.
    # Don't want to add dependencies that would need to be installed, so very basic
    # handling of arguments

    if "-h" ∈ ARGS || "--help" ∈ ARGS
        println("Script to set up moment_kinetics to run on a cluster.")
        println()
        println("The settings requested correspond to the arguments to ")
        println("`machine_setup_moment_kinetics()`. See the function docstring:")
        println()
        println(@doc machine_setup_moment_kinetics)
        exit(9)
    elseif "-d" ∈ ARGS || "--defaults" ∈ ARGS
        # Print out the default values for arguments for this machine
        i = "-d" ∈ ARGS ? findfirst(x -> x=="-d", ARGS) :
                          findfirst(x -> x=="--defaults", ARGS)
        if length(ARGS) <= i
            println("Must pass a machine name after `-d` or `--defaults`")
            exit(1)
        end
        machine = ARGS[i+1]

        known_machines = sort(collect(keys(machine_setup.default_settings)))
        # Remove "base" which is a dummy entry
        known_machines = filter(x -> x ≠ "base", known_machines)
        if !(machine ∈ keys(machine_setup.default_settings))
            println(stderr, "Error: machine \"$machine\" not recognised")
            println(stderr, "       Known machines are $known_machines\n")
            exit(1)
        end

        d = machine_setup.default_settings[machine]
        println()
        exit(0)
    end

    # Get function arguments from ARGS
    machine_setup_moment_kinetics(ARGS...; interactive=false)

    exit(0)
end
