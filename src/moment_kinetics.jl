"""
"""
module moment_kinetics

export run_moment_kinetics, restart_moment_kinetics

using MPI

# Include submodules from other source files
# Note that order of includes matters - things used in one module must already
# be defined
include("command_line_options.jl")
include("debugging.jl")
include("type_definitions.jl")
include("communication.jl")
include("moment_kinetics_structs.jl")
include("looping.jl")
include("array_allocation.jl")
include("interpolation.jl")
include("calculus.jl")
include("clenshaw_curtis.jl")
include("chebyshev.jl")
include("finite_differences.jl")
include("quadrature.jl")
include("hermite_spline_interpolation.jl")
include("file_io.jl")
include("input_structs.jl")
include("coordinates.jl")
include("velocity_moments.jl")
include("em_fields.jl")
include("bgk.jl")
include("initial_conditions.jl")
include("moment_constraints.jl")
include("semi_lagrange.jl")
include("advection.jl")
include("vpa_advection.jl")
include("z_advection.jl")
include("r_advection.jl")
include("charge_exchange.jl")
include("ionization.jl")
include("continuity.jl")
include("energy_equation.jl")
include("force_balance.jl")
include("source_terms.jl")
include("numerical_dissipation.jl")
include("analysis.jl")
include("load_data.jl")
include("post_processing_input.jl")
include("post_processing.jl")
include("time_advance.jl")

include("moment_kinetics_input.jl")
include("scan_input.jl")

using TimerOutputs
using TOML

using .file_io: setup_file_io, finish_file_io, reload_evolving_fields!
using .file_io: write_data_to_ascii, write_data_to_binary
using .command_line_options: get_options
using .communication
using .communication: _block_synchronize
using .coordinates: define_coordinate
using .debugging
using .initial_conditions: init_pdf_and_moments, enforce_boundary_conditions!
using .looping
using .moment_constraints: hard_force_moment_constraints!
using .moment_kinetics_input: mk_input, run_type, performance_test
using .time_advance: setup_time_advance!, time_advance!

@debug_detect_redundant_block_synchronize using ..communication: debug_detect_redundant_is_active

"""
main function that contains all of the content of the program
"""
function run_moment_kinetics(to::TimerOutput, input_dict=Dict())
    try
        # set up all the structs, etc. needed for a run
        mk_state = setup_moment_kinetics(input_dict)

        try
            # solve the 1+1D kinetic equation to advance f in time by nstep time steps
            if run_type == performance_test
                @timeit to "time_advance" time_advance!(mk_state...)
            else
                time_advance!(mk_state...)
            end
        finally

            # clean up i/o and communications
            # last 2 elements of mk_state are `io` and `cdf`
            cleanup_moment_kinetics!(mk_state[end-1:end]...)
        end

        if block_rank[] == 0 && run_type == performance_test
            # Print the timing information if this is a performance test
            display(to)
            println()
        end
    catch e
        # Stop code from hanging when running on multiple processes if only one of them
        # throws an error
        if global_size[] > 1
            println("Abort called on rank $(block_rank[]) due to error. Error message "
                    * "was:\n", e)
            #MPI.Abort(comm_world, 1)
        end

        rethrow(e)
    end

    return nothing
end

"""
overload which takes a filename and loads input
"""
function run_moment_kinetics(to::TimerOutput, input_filename::String)
    return run_moment_kinetics(to, TOML.parsefile(input_filename))
end

"""
overload with no TimerOutput arguments
"""
function run_moment_kinetics(input)
    return run_moment_kinetics(TimerOutput(), input)
end

"""
overload which gets the input file name from command line arguments
"""
function run_moment_kinetics()
    inputfile = get_options()["inputfile"]
    if inputfile == nothing
        run_moment_kinetics(Dict())
    else
        run_moment_kinetics(inputfile)
    end
end

"""
Append a number to the filename, to get a new, non-existing filename to backup the file
to.
"""
function get_backup_filename(filename)
    counter = 1
    basename, extension = splitext(filename)
    backup_name = ""
    while true
        backup_name = "$(basename)_$(counter)$(extension)"
        if !isfile(backup_name)
            break
        end
        counter += 1
    end
    backup_name == "" && error("Failed to find a name for backup file.")
    return backup_name
end

"""
Restart moment kinetics from an existing run. Space/velocity-space resolution in the
input must be the same as for the original run.
"""
function restart_moment_kinetics(restart_filename::String, input_filename::String,
                                 time_index::Int=-1)
    restart_moment_kinetics(restart_filename, TOML.parsefile(input_filename),
                            time_index)
    return nothing
end
function restart_moment_kinetics()
    options = get_options()
    inputfile = options["inputfile"]
    if inputfile === nothing
        error("Must pass input file as first argument to restart a run.")
    end
    restartfile = options["restartfile"]
    if restartfile === nothing
        error("Must pass output file to restart from as second argument.")
    end
    time_index = options["restart-time-index"]

    restart_moment_kinetics(restartfile, inputfile, time_index)

    return nothing
end
function restart_moment_kinetics(restart_filename::String, input_dict::Dict,
                                 time_index::Int=-1)
    try
        # Move the output file being restarted from to make sure it doesn't get
        # overwritten.
        backup_filename = get_backup_filename(restart_filename)
        global_rank[] == 0 && mv(restart_filename, backup_filename)

        # Set up all the structs, etc. needed for a run.
        pdf, scratch, code_time, t_input, vpa, z, r, vpa_spectral, z_spectral,
        r_spectral, moments, fields, vpa_advect, z_advect, r_advect, vpa_SL, z_SL, r_SL,
        composition, collisions, num_diss_params, advance, scratch_dummy_sr, io, cdf =
        setup_moment_kinetics(input_dict, backup_filename=backup_filename,
                              restart_time_index=time_index)

        try
            time_advance!(pdf, scratch, code_time, t_input, vpa, z, r, vpa_spectral,
                          z_spectral, r_spectral, moments, fields, vpa_advect, z_advect,
                          r_advect, vpa_SL, z_SL, r_SL, composition, collisions,
                          num_diss_params, advance, scratch_dummy_sr, io, cdf)
        finally
            # clean up i/o and communications
            # last 2 elements of mk_state are `io` and `cdf`
            cleanup_moment_kinetics!(io, cdf)
        end
    catch e
        # Stop code from hanging when running on multiple processes if only one of them
        # throws an error
        if global_size[] > 1
            println("Abort called on rank $(block_rank[]) due to error. Error message "
                    * "was:\n", e)
            #MPI.Abort(comm_world, 1)
        end

        rethrow(e)
    end

    return nothing
end

"""
Perform all the initialization steps for a run.

If `backup_filename` is `nothing`, set up for a regular run; if a filename is passed,
reload data from time index given by `restart_time_index` for a restart.
"""
function setup_moment_kinetics(input_dict::Dict; backup_filename=nothing,
                               restart_time_index=-1)
    # Set up MPI
    initialize_comms!()

    input = mk_input(input_dict)
    # obtain input options from moment_kinetics_input.jl
    # and check input to catch errors
    run_name, output_dir, evolve_moments, t_input, z_input, r_input, vpa_input,
        composition, species, collisions, drive_input, num_diss_params = input
    # initialize z grid and write grid point locations to file
    z, z_spectral = define_coordinate(z_input, composition)
    # initialize r grid and write grid point locations to file
    r, r_spectral = define_coordinate(r_input, composition)
    # initialize vpa grid and write grid point locations to file
    vpa, vpa_spectral = define_coordinate(vpa_input, composition)
    # Create loop range variables for shared-memory-parallel loops
    looping.setup_loop_ranges!(block_rank[], block_size[]; s=composition.n_species, r=r.n,
                               z=z.n, vpa=vpa.n)
    # initialize f(z,vpa) and the lowest three v-space moments (density(z), upar(z) and ppar(z)),
    # each of which may be evolved separately depending on input choices.
    pdf, moments = init_pdf_and_moments(vpa, z, r, vpa_spectral, composition, species,
                                        t_input.n_rk_stages, evolve_moments,
                                        collisions.ionization)
    # initialize time variable
    code_time = 0.
    # create arrays and do other work needed to setup
    # the main time advance loop -- including normalisation of f by density if requested
    moments, fields, vpa_advect, z_advect, r_advect, vpa_SL, z_SL, r_SL, scratch,
        advance, scratch_dummy_sr = setup_time_advance!(pdf, vpa, z, r, z_spectral,
            composition, drive_input, moments, t_input, collisions, species,
            num_diss_params)

    if backup_filename !== nothing
        # Have done unnecessary initialisation of pdf and moments, which is overwritten
        # here
        code_time = reload_evolving_fields!(pdf, moments, backup_filename,
                                            restart_time_index, composition, r, z, vpa)
        _block_synchronize()
    end

    # setup i/o
    io, cdf = setup_file_io(output_dir, run_name, vpa, z, r, composition, collisions,
                            moments.evolve_density, moments.evolve_upar,
                            moments.evolve_ppar)
    # write initial data to ascii files
    write_data_to_ascii(pdf.norm, moments, fields, vpa, z, r, code_time, composition.n_species, io)
    # write initial data to binary file (netcdf)
    write_data_to_binary(pdf.norm, moments, fields, code_time, composition.n_species, cdf, 1)

    begin_s_r_z_region()

    return pdf, scratch, code_time, t_input, vpa, z, r, vpa_spectral, z_spectral, r_spectral, moments,
           fields, vpa_advect, z_advect, r_advect, vpa_SL, z_SL, r_SL, composition,
           collisions, num_diss_params, advance, scratch_dummy_sr, io, cdf
end

"""
Clean up after a run
"""
function cleanup_moment_kinetics!(io::Union{file_io.ios,Nothing},
                                  cdf::Union{file_io.netcdf_info,Nothing})
    @debug_detect_redundant_block_synchronize begin
        # Disable check for redundant _block_synchronize() during finalization, as this
        # only runs once so any failure is not important.
        debug_detect_redundant_is_active[] = false
    end

    begin_serial_region()

    # finish i/o
    finish_file_io(io, cdf)

    # clean up MPI objects
    finalize_comms!()

    return nothing
end

end
