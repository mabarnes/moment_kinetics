"""
Functions to help setting up on known machines
"""
module machine_setup

export machine_setup_moment_kinetics

using Pkg
using TOML

# Default settings for the arguments to machine_setup_moment_kinetics(), set like this
# so that they can be passed to the bash script `machine_setup.sh`
default_settings = Dict{String,Dict{String,String}}()
default_settings["base"] = Dict("account"=>"",
                                "default_run_time"=>"24:00:00",
                                "default_nodes"=>"1",
                                "default_postproc_time"=>"1:00:00",
                                "default_postproc_memory"=>"64G",
                                "default_partition"=>"",
                                "default_qos"=>"",
                                "submit_precompilation"=>"y",
                                "use_makie"=>"n",
                                "use_plots"=>"n",
                                "separate_postproc_projects"=>"n",
                                "use_system_mpi"=>"y",
                                "use_netcdf"=>"n",
                                "enable_mms"=>"n",
                                "use_revise"=>"n")
# No batch system steup for "generic-pc"
default_settings["generic-pc"] = merge(default_settings["base"],
                                   Dict("default_run_time"=>"0:00:00",
                                        "default_nodes"=>"0",
                                        "default_postproc_time"=>"0:00:00",
                                        "default_postproc_memory"=>"0",
                                        "use_makie"=>"y",
                                        "use_revise"=>"y"))
default_settings["generic-batch"] = deepcopy(default_settings["base"])
default_settings["archer"] = merge(default_settings["base"],
                                   Dict("default_partition"=>"standard",
                                        "default_qos"=>"standard"))
default_settings["pitagora"] = merge(default_settings["base"],
                                     Dict("default_partition"=>"dcgp_fua_prod",
                                          "default_postproc_time"=>"0:30:00",
                                          "default_qos"=>"normal"))
"""
    get_user_input(possible_values, default_value)

Prompt for user input. If the user enters nothing, return `default_value`. Check that the
entered value is one of `possible_values`, if not prompt again.
"""
function get_user_input(possible_values, default_value)
    setting = default_value
    while true
        if possible_values[1] == default_value
            print("[", possible_values[1], "]")
        else
            print(possible_values[1])
        end
        for x ∈ possible_values[2:end]
            if x == default_value
                print("/[", x, "]")
            else
                print("/$x")
            end
        end
        println(":")
        print("> ")
        input = readline()
        if input == ""
            break
        elseif input ∈ possible_values
            setting = input
            break
        end
    end
    return setting
end

"""
    get_setting(setting_name, message, machine, local_defaults,
                possible_values=nothing)

Prompt the user to set a setting called `setting_name` after printing `message`. Default
value is read from `local_defaults` if it exists there (which it will do if it has been
set before, as then it is stored in `LocalPreferences.toml`), or from sensible defaults in
the `machine` section of `default_settings` otherwise.
"""
function get_setting(setting_name, message, machine, local_defaults,
                     possible_values=nothing)
    # Get default value
    default_value = get(local_defaults, setting_name, default_settings[machine][setting_name])

    if possible_values === nothing
        println("$message\n[$default_value]:")
        print("> ")
        setting = readline()
        if setting == ""
            setting = default_value
        end
    else
        println(message)
        setting = get_user_input(possible_values, default_value)
    end

    println("\nUsing $setting_name=$setting\n")
    local_defaults[setting_name] = setting

    return setting
end

"""
    machine_setup_moment_kinetics(machine::String; ; no_force_exit::Bool=false,
                                  interactive::Bool=true)

Do setup for a known `machine`, prompting the user for various settings (with defaults set
to sensible values - if the script has been run before, the defaults are the previously
used values):
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
* Makes a symlink to, or a bash script that calls, the Julia executable used to run this
  command at `bin/julia` under the moment_kinetics repo, so that setup and job submission
  scripts can use a known relative path.
  !!! note
      If you change the Julia executable, e.g. to update to a new verison, you will need
      to either replace the symlink `<moment_kinetics>/bin/julia` or edit the bash script
      at `<moment_kinetics>/bin/julia` by hand, or re-run this function using the new
      executable.

Usually it is necessary for Julia to be restarted after running this function to run Julia
with the correct `JULIA_DEPOT_PATH`, etc. so the function will force Julia to exit. If for
some reason this is not desired (e.g. when debugging), pass `no_force_exit=true`.

The `interactive` argument exists so that when this function is called from another
script, terminal output with instructions for the next step can be disabled.

Currently supported machines:
* `"generic-pc"` - A generic personal computer (i.e. laptop or desktop machine).. Set up
    for interactive use, rather than for submitting jobs to a batch queue.
* `"generic-batch"` - A generic cluster using a batch queue. Requires some manual setup
    first, see `machines/generic-batch-template/README.md`.
* `"archer"` - the UK supercomputer [ARCHER2](https://www.archer2.ac.uk/)
* "pitagora" - the EUROfusion supercomputer
  [Pitagora](https://docs.hpc.cineca.it/hpc/pitagora.html)

!!! note
    The settings created by this function are saved in LocalPreferences.toml. It might
    sometimes be useful to edit these by hand (e.g.  the `account` setting if this needs
    to be changed.): it is fine to do this.
"""
function machine_setup_moment_kinetics(machine::String; no_force_exit::Bool=false,
                                       interactive::Bool=true)

    repo_dir = dirname(dirname(dirname(@__FILE__)))

    # Get defaults from LocalPreferences.toml if possible
    if isfile("LocalPreferences.toml")
        local_preferences = TOML.parsefile("LocalPreferences.toml")
        if "moment_kinetics" ∈ keys(local_preferences)
            mk_preferences = local_preferences["moment_kinetics"]
        else
            mk_preferences = local_preferences["moment_kinetics"] = Dict{String,Any}()
        end
    else
        local_preferences = Dict{String,Any}()
        mk_preferences = local_preferences["moment_kinetics"] = Dict{String,Any}()
    end

    mk_preferences["machine"] = machine

    # Common operations that only depend on the name of `machine`
    #############################################################

    if machine == "generic-pc"
        batch_system = false
    else
        batch_system = true
    end
    mk_preferences["batch_system"] = batch_system

    # Get some settings
    if haskey(ENV, "JULIA_DEPOT_PATH")
        julia_directory = ENV["JULIA_DEPOT_PATH"]
    else
        julia_directory = ""
    end
    mk_preferences["julia_directory"] = julia_directory
    if batch_system
        get_setting("default_run_time",
                    "Enter the default value for the time limit for simulation jobs",
                    machine, mk_preferences)
        get_setting("default_nodes",
                    "Enter the default value for the number of nodes for a run",
                    machine, mk_preferences)
        get_setting("default_postproc_time",
                    "Enter the default value for the time limit for post-processing jobs",
                    machine, mk_preferences)
        get_setting("default_postproc_memory",
                    "Enter the default value for the memory requested for post-processing jobs",
                    machine, mk_preferences)
        get_setting("default_partition",
                    "Enter the default value for the partition for simulation jobs",
                    machine, mk_preferences)
        get_setting("default_qos",
                    "Enter the default value for the QOS for simulation jobs",
                    machine, mk_preferences)
        get_setting("account",
                    "Enter the account code used to submit jobs",
                    machine, mk_preferences)
        get_setting("submit_precompilation",
                    "Do you want to submit a serial (or debug) job to precompile, creating the\n"
                    * "moment_kinetics.so image (this is required in order to use the job submission\n"
                    * "scripts and templates provided)?\n",
                    machine, mk_preferences, ["y", "n"])
    end
    get_setting("use_makie",
                "Would you like to set up makie_post_processing?",
                machine, mk_preferences, ["y", "n"])
    get_setting("use_plots",
                "Would you like to set up plots_post_processing?",
                machine, mk_preferences, ["y", "n"])
    if !batch_system
        get_setting("separate_postproc_projects",
                    "Would you like to set up separate packages for post processing (this might\n"
                    * "be useful if you want to use different optimization flags for runs and\n"
                    * "post-processing for example)?",
                    machine, mk_preferences, ["y", "n"])
    end
    get_setting("use_system_mpi",
                "Normally you probably want to use the system-provided MPI library. However\n"
                * "occasionally it can be useful to use the Julia-provided MPI instead. If you\n"
                * "choose the Julia-provided MPI, a link to `mpiexecjl` will be installed to the\n"
                * "project directory, which you should use instead of `mpirun`/`mpiexec` when\n"
                * "you want to run an MPI job - i.e. use `./mpiexecjl -np 4 julia ...` instead\n"
                * "of `mpirun -np 4 julia ...`.\n"
                * "Do you want to use the system-provided MPI?",
                machine, mk_preferences, ["y", "n"])
    get_setting("use_netcdf",
                "Would you like to enable optional NetCDF I/O (warning: using NetCDF sometimes\n"
                * "causes errors when using a local or system install of HDF5)?",
                machine, mk_preferences, ["y", "n"])
    get_setting("enable_mms",
                "Would you like to enable MMS testing?",
                machine, mk_preferences, ["y", "n"])
    if !batch_system
        get_setting("use_revise",
                    "Would you like to automatically use Revise.jl (so that you do not "
                    * "need to restart julia after editing code)?",
                    machine, mk_preferences, ["y", "n"])
        if mk_preferences["use_revise"] == "y"
            Pkg.add("Revise")

            # Check that `using Revise` is in the startup.jl
            if julia_directory == ""
                depot_directory = DEPOT_PATH[1]
            else
                depot_directory = julia_directory
            end
            # Ensure that the config/ subdirectory exists in depot_directory
            config_path = joinpath(depot_directory, "config")
            mkpath(config_path)
            # Ensure startup.jl is a file
            startup_path = joinpath(config_path, "startup.jl")
            touch(startup_path)
            result = run(`grep "using Revise" $startup_path`, wait=false)
            if !success(result)
                println("Adding `using Revise` to $startup_path")
                # When initialising a new copy of the repo, if Revise is not installed yet
                # having just `using Revise` in the startup.jl would cause an error, so
                # guard it with a try/catch.
                open(startup_path, "a") do io
                    println(io, "\ntry")
                    println(io, "    using Revise")
                    println(io, "catch")
                    println(io, "    println(\"Warning: failed to load Revise\")")
                    println(io, "end")
                end
            end
        end
    end

    # Write these preferences into a [moment_kinetics] section in LocalPreferences.toml
    #
    # Load and re-write LocalPreferences.toml directly here to avoid needing to import
    # the Preferences.jl package, which would need to be installed (TOML.jl is available
    # as part of the Julia system). This is a bit hacky, but hopefully no need to do
    # anything fancy here!
    println("\n** Adding system-specific settings for moment_kinetics to LocalPreferences.toml\n")
    open("LocalPreferences.toml", "w") do io
        TOML.print(io, local_preferences, sorted=true)
    end

    if batch_system
        # Only use julia.env for a batch system as for an interactive system there are no
        # modules to set up and source'ing julia.env is mildly inconvenient.
        println("\n** Creating `julia.env` for environment setup\n")
        envname = joinpath(repo_dir, "julia.env")
        # Read the template
        template = read("machines/$machine/julia.env")
        # Write julia.env, overwriting if it already exists
        ispath(envname) && rm(envname)
        open(envname, "w") do io
            write(io, template)

            # Don't do the following on ARCHER2 because the depot has to be copied onto
            # the compute notes within batch jobs, but sometimes scripts want to use the
            # julia.env on the login nodes.
            if machine != "archer" && julia_directory != ""
                println("\n** Setting JULIA_DEPOT_PATH=$julia_directory in `julia.env`\n")
                println(io, "\nexport JULIA_DEPOT_PATH=$julia_directory")
            end
        end
    end

    bindir = joinpath(repo_dir, "bin")
    mkpath(bindir)
    julia_executable_name = joinpath(bindir, "julia")
    if batch_system || (julia_directory == "" && mk_preferences["use_plots"] == "n")
        # Make a local link to the Julia binary so scripts in the repo can find it
        println("\n** Making a symlink to the julia executable at bin/julia\n")
        islink(julia_executable_name) && rm(julia_executable_name)
        symlink(joinpath(Sys.BINDIR, "julia"), julia_executable_name)
    else
        # Make a script to run julia, including the JULIA_DEPOT_PATH so that we can avoid
        # needing the julia.env setup
        open(julia_executable_name, "w") do io
            println(io, "#!/usr/bin/env bash")
            if julia_directory != ""
                println(io, "export JULIA_DEPOT_PATH=$julia_directory")
            end
            julia_path = joinpath(Sys.BINDIR, "julia")
            println(io, "$julia_path \"\$@\"")
        end
        function make_executable!(file)
            user_permissions = uperm(file)
            group_permissions = gperm(file)
            other_permissions = operm(file)

            # Change each permissions field to be executable
            user_permissions = user_permissions | 0x01
            group_permissions = group_permissions | 0x01
            other_permissions = other_permissions | 0x01

            permissions = user_permissions * UInt16(0o100) +
                          group_permissions * UInt16(0o10) +
                          other_permissions * UInt16(0o1)
            chmod(file, permissions)

            return nothing
        end
        make_executable!(julia_executable_name)
    end

    # If it is necessary to run a shell script to compile dependencies, set
    # this flag to true.
    compile_dependencies_relative_path = joinpath("machines", "shared",
                                                  "compile_dependencies.sh")
    compile_dependencies_path = joinpath(repo_dir,
                                         compile_dependencies_relative_path)

    # Set this flag to true in the machine-specific branch below to require a
    # non-empty `account` setting
    needs_account = false

    if machine == "generic-pc"
    elseif machine == "generic-batch"
        needs_account = true
    elseif machine == "archer"
        needs_account = true
        if julia_directory == ""
            error("On ARCHER2, the `julia_directory` setting is required, because the "
                  * "default location for the `.julia` directory is in your home "
                  * "directory and the `/home/` filesystem is not available on the "
                  * "compute nodes.")
        end
    elseif machine == "pitagora"
        needs_account = true
    else
        error("Unsupported machine '$machine'")
    end

    if needs_account && mk_preferences["account"] == ""
        error("For machine=\"$machine\" it is required to pass a value for the "
              * "`account` argument.")
    end

    if isfile(joinpath("machines", machine, "compile_dependencies.sh"))
        # Remove link if it exists already
        islink(compile_dependencies_path) && rm(compile_dependencies_path)

        symlink(joinpath("..", machine, "compile_dependencies.sh"), compile_dependencies_path)

        if interactive
            println()
            println("***********************************************************************")
            println("To compile dependencies run:")
            println("    \$ machines/shared/compile_dependencies.sh")
            println("***********************************************************************")
        end
    end

    if interactive
        println()
        println("***********************************************************************")
        println("To complete setup, first `source julia.env` if it exists (so that")
        println("`JULIA_DEPOT_PATH` is set correctly), then start Julia again (you can")
        println("now use `bin/julia --project`) and to complete the setup run:")
        println("    julia> include(\"$second_stage_relative_path\")")
        println("***********************************************************************")
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
        println("Runs `machine_setup_moment_kinetics()`. See the function docstring:")
        println()
        println(@doc machine_setup_moment_kinetics)
        exit(9)
    end

    machine_setup_moment_kinetics(ARGS[1]; interactive=false)

    exit(0)
end
