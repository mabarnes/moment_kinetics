"""
Functions to help setting up on known machines
"""
module machine_setup

export machine_setup_moment_kinetics

using TOML

# Default settings for the arguments to machine_setup_moment_kinetics(), set like this
# so that they can be passed to the bash script `machine_setup.sh`
default_settings = Dict{String,Dict{String,String}}()
default_settings["base"] = Dict("default_run_time"=>"24:00:00",
                                "default_nodes"=>"1",
                                "default_postproc_time"=>"1:00:00",
                                "default_postproc_memory"=>"64G",
                                "default_partition"=>"",
                                "default_qos"=>"",
                                "use_makie"=>"1",
                                "use_plots"=>"1")
# No batch system steup for "generic-pc"
default_settings["generic-pc"] = merge(default_settings["base"],
                                   Dict("default_run_time"=>"0:00:00",
                                        "default_nodes"=>"0",
                                        "default_postproc_time"=>"0:00:00",
                                        "default_postproc_memory"=>"0",
                                        "use_makie"=>"0"))
default_settings["archer"] = merge(default_settings["base"],
                                   Dict("default_partition"=>"standard",
                                        "default_qos"=>"standard"))
default_settings["marconi"] = merge(default_settings["base"],
                                    Dict("default_partition"=>"skl_fua_prod",
                                         "default_qos"=>"normal"))


"""
    machine_setup_moment_kinetics(machine::String,
                                  account::String,
                                  julia_directory::String,
                                  default_run_time::String,
                                  default_nodes::String,
                                  default_postproc_time::String,
                                  default_postproc_memory::String,
                                  default_partition::String;
                                  default_qos::String;
                                  use_makie::String;
                                  use_plots::String;
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
used to set `JULIA_DEPOT_PATH` in the `julia.env` file or `bin/julia` script , so that
this setting is propagated to the environment on the compute nodes.

Usually it is necessary for Julia to be restarted after running this function to ensure
the correct MPI is linked, etc. so the function will force Julia to exit. If for some
reason this is not desired (e.g. when debugging), pass `no_force_exit=true`.

The `interactive` argument exists so that when this function is called from another
script, terminal output with instructions for the next step can be disabled.

The remaining arguments can be used to change the default settings for jobs submitted
using the provided `submit-run.sh` script. These settings are read by the scripts from
`LocalPreferences.toml` and the values can safely be edited in that file without
re-running this function (if you want to). The arguments are:
* `default_run_time` is the maximum run time for the simulation, in the format expected
  by `sbatch --time`, e.g. `"24:00:00"` for 24 hours, 0 minutes, 0 seconds.
* `default_nodes` is the default number of nodes to use for a simulation run. Note that
  post-processing always runs in serial (using a serial or debug queue if available).
* `default_postproc_time` is the maximum run time for the post-processing job, in the
  format expected by `sbatch --time`, e.g. `"1:00:00"` for 1 hours, 0 minutes, 0
  seconds.
* `default_postproc_memory` is the memory requested for the post-processing job, in the
  format expected by `sbatch --mem`, e.g. `"64G"` for 64GB.
* `default_partition` is the default 'partition' passed to `sbatch --partition`. See your
  cluster's documentation for possible values. The default will be the standard queue,
  which charges towards the budget of your allocation. You might sometimes want, for
  example, to change this to a debug queue if one is available.
* `default_qos` is the default 'quality of service' passed to `sbatch --qos`. See your
  cluster's documentation for possible values. The default will be the standard queue,
  which charges towards the budget of your allocation. You might want, for example, to
  change this to a free, low-priority queue if one is available.
* `use_makie` indicates whether makie_post_processing has been enabled ("0" means yes, "1"
  means no).
* `use_plots` indicates whether plots_post_processing has been enabled ("0" means yes, "1"
  means no).

Currently supported machines:
* `"generic-pc"` - A generic personal computer. Set up for interactive use, rather than
    for submitting jobs to a batch queue.
* `"archer"` - the UK supercomputer [ARCHER2](https://www.archer2.ac.uk/)
* `"marconi"` - the EUROfusion supercomputer
    [Marconi](https://wiki.u-gov.it/confluence/display/SCAIUS/UG3.1%3A+MARCONI+UserGuide)

!!! note
    The settings created by this function are saved in LocalPreferences.toml (using the
    `Preferences.jl` package). It might sometimes be useful to edit these by hand (e.g.
    the `account` setting if this needs to be changed.): it is fine to do this.
"""
function machine_setup_moment_kinetics(machine::String,
                                       account::String,
                                       julia_directory::String,
                                       default_run_time::String,
                                       default_nodes::String,
                                       default_postproc_time::String,
                                       default_postproc_memory::String,
                                       default_partition::String,
                                       default_qos::String,
                                       use_makie::String,
                                       use_plots::String;
                                       no_force_exit::Bool=false,
                                       interactive::Bool=true)

    repo_dir = dirname(dirname(dirname(@__FILE__)))

    # Common operations that only depend on the name of `machine`
    #############################################################

    if machine == "generic-pc"
        batch_system = false
    else
        batch_system = true
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
            if julia_directory != ""
                println("\n** Setting JULIA_DEPOT_PATH=$julia_directory in `julia.env`\n")
                println(io, "\nexport JULIA_DEPOT_PATH=$julia_directory")
            end
        end
    end

    bindir = joinpath(repo_dir, "bin")
    mkpath(bindir)
    julia_executable_name = joinpath(bindir, "julia")
    if batch_system || julia_directory == ""
        # Make a local link to the Julia binary so scripts in the repo can find it
        println("\n** Making a symlink to the julia executable at bin/julia\n")
        islink(julia_executable_name) && rm(julia_executable_name)
        symlink(joinpath(Sys.BINDIR, "julia"), julia_executable_name)
    else
        # Make a script to run julia, including the JULIA_DEPOT_PATH so that we can avoid
        # needing the julia.env setup
        open(julia_executable_name, "w") do io
            println(io, "#!/bin/bash")
            println(io, "export JULIA_DEPOT_PATH=$julia_directory")
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
    mk_preferences = local_preferences["moment_kinetics"] = Dict{String,String}()
    mk_preferences["julia_directory"] = julia_directory
    mk_preferences["default_run_time"] = default_run_time
    mk_preferences["default_nodes"] = default_nodes
    mk_preferences["default_postproc_time"] = default_postproc_time
    mk_preferences["default_postproc_memory"] = default_postproc_memory
    mk_preferences["default_partition"] = default_partition
    mk_preferences["default_qos"] = default_qos
    mk_preferences["account"] = account
    mk_preferences["use_makie"] = use_makie
    mk_preferences["use_plots"] = use_plots
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

    if machine == "generic-pc"
        # For generic-pc, run compile_dependencies.sh script to optionally download and
        # compile HDF5
        needs_compile_dependencies = true

        needs_second_stage = true
    elseif machine == "archer"
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
            println("To complete setup, first `source julia.env` if it exists (so that")
            println("`JULIA_DEPOT_PATH` is set correctly), then start Julia again (you can")
            println("now use `bin/julia --project`) and to complete the setup run:")
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
using TOML

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

        d = deepcopy(machine_setup.default_settings[machine])
        # default setting for "julia_directory" is the JULIA_DEPOT_PATH environment
        # variable
        d["julia_directory"] = get(ENV, "JULIA_DEPOT_PATH", "")
        # No default for "account".
        d["account"] = ""

        # If settings have already been saved (i.e. machine_setup.sh has already been
        # run), then use the previous settings as the default this time.
        # Use TOML.parsefile() to read the existing preferences to avoid depending on the
        # Preferences package at this point (because we might want to set
        # JULIA_DEPOT_PATH, but not have set it yet).
        if ispath("LocalPreferences.toml")
            existing_settings = get(TOML.parsefile("LocalPreferences.toml"),
                                    "moment_kinetics", Dict{String, String}())
        else
            existing_settings = Dict{String, String}()
        end
        for setting ∈ ("default_run_time", "default_nodes", "default_postproc_time",
                       "default_postproc_memory", "default_partition",
                       "default_partition", "account", "julia_directory", "use_makie",
                       "use_plots")
            d[setting] = get(existing_settings, setting, d[setting])
        end

        println("\"", d["default_run_time"], "\" \"",d["default_nodes"], "\" \"",
                d["default_postproc_time"], "\" \"", d["default_postproc_memory"],
                "\" \"", d["default_partition"], "\" \"", d["default_qos"], "\" \"",
                d["account"], "\" \"", d["julia_directory"], "\" \"", d["use_makie"],
                "\" \"", d["use_plots"], "\"")
        exit(0)
    end

    # Get function arguments from ARGS
    machine_setup_moment_kinetics(ARGS...; interactive=false)

    exit(0)
end
