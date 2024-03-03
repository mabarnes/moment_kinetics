"""
"""
module moment_kinetics

export run_moment_kinetics

using MPI

# Include submodules from other source files
# Note that order of includes matters - things used in one module must already
# be defined
include("../../machines/shared/machine_setup.jl") # Included so Documenter.jl can find its docs
include("check_so_newer_than_code.jl")
include("command_line_options.jl")
include("constants.jl")
include("debugging.jl")
include("type_definitions.jl")
include("communication.jl")
include("moment_kinetics_structs.jl")
include("looping.jl")
include("array_allocation.jl")
include("interpolation.jl")
include("calculus.jl")
include("clenshaw_curtis.jl")
include("gauss_legendre.jl")
include("chebyshev.jl")
include("finite_differences.jl")
include("quadrature.jl")
include("hermite_spline_interpolation.jl")
include("derivatives.jl")
include("input_structs.jl")
include("reference_parameters.jl")
include("coordinates.jl")
include("file_io.jl")
include("geo.jl")
include("velocity_moments.jl")
include("velocity_grid_transforms.jl")
include("gyroaverages.jl")
include("em_fields.jl")
include("bgk.jl")
include("manufactured_solns.jl") # MRH Here?
include("external_sources.jl")
include("initial_conditions.jl")
include("moment_constraints.jl")
include("fokker_planck_test.jl")
include("fokker_planck_calculus.jl")
include("fokker_planck.jl")
include("advection.jl")
include("vpa_advection.jl")
include("z_advection.jl")
include("r_advection.jl")
include("vperp_advection.jl")
include("neutral_r_advection.jl")
include("neutral_z_advection.jl")
include("neutral_vz_advection.jl")
include("charge_exchange.jl")
include("ionization.jl")
include("krook_collisions.jl")
include("continuity.jl")
include("energy_equation.jl")
include("force_balance.jl")
include("source_terms.jl")
include("numerical_dissipation.jl")
include("moment_kinetics_input.jl")
include("load_data.jl")
include("parameter_scans.jl")
include("analysis.jl")
include("utils.jl")
include("time_advance.jl")

using TimerOutputs
using Dates
using Glob
using Primes

using .file_io: setup_file_io, finish_file_io
using .file_io: write_data_to_ascii
using .file_io: write_moments_data_to_binary, write_dfns_data_to_binary
using .command_line_options: get_options
using .communication
using .communication: _block_synchronize
using .debugging
using .external_sources
using .input_structs
using .initial_conditions: allocate_pdf_and_moments, init_pdf_and_moments!,
                           enforce_boundary_conditions!
using .load_data: reload_evolving_fields!
using .looping
using .moment_constraints: hard_force_moment_constraints!
using .looping: debug_setup_loop_ranges_split_one_combination!
using .moment_kinetics_input: mk_input, read_input_file
using .time_advance: setup_time_advance!, time_advance!
using .type_definitions: mk_int
using .utils: to_minutes

@debug_detect_redundant_block_synchronize using ..communication: debug_detect_redundant_is_active

"""
main function that contains all of the content of the program
"""
function run_moment_kinetics(to::Union{TimerOutput,Nothing}, input_dict=Dict();
                             restart=false, restart_time_index=-1)

    if global_rank[] == 0
        # Check that, if we are using a custom compiled system image that includes
        # moment_kinetics, the system image is newer than the source code files (if there
        # are changes made to the source code since the system image was compiled, they
        # will not affect the current run). Prints a warning if any code files are newer
        # than the system image.
        check_so_newer_than_code()
    end

    mk_state = nothing
    try
        # set up all the structs, etc. needed for a run
        mk_state = setup_moment_kinetics(input_dict; restart=restart,
                                         restart_time_index=restart_time_index)

        # solve the 1+1D kinetic equation to advance f in time by nstep time steps
        if to === nothing
            time_advance!(mk_state...)
        else
            @timeit to "time_advance" time_advance!(mk_state...)
        end

        # clean up i/o and communications
        # last 3 elements of mk_state are ascii_io, io_moments, and io_dfns
        cleanup_moment_kinetics!(mk_state[end-2:end]...)

        if block_rank[] == 0 && to !== nothing
            # Print the timing information if this is a performance test
            display(to)
            println()
        end
    catch e
        # Stop code from hanging when running on multiple processes if only one of them
        # throws an error
        if global_size[] > 1
            println("$(typeof(e)) on process $(global_rank[]):")
            showerror(stdout, e)
            display(stacktrace(catch_backtrace()))
            flush(stdout)
            flush(stderr)
            MPI.Abort(comm_world, 1)
        else
            # Error almost certainly occured before cleanup. If running in serial we can
            # still finalise file I/O
            cleanup_moment_kinetics!(mk_state[end-2:end]...)
        end

        rethrow(e)
    end

    return nothing
end

"""
overload which takes a filename and loads input
"""
function run_moment_kinetics(to::Union{TimerOutput,Nothing}, input_filename::String;
                             restart=false, restart_time_index=-1)
    return run_moment_kinetics(to, read_input_file(input_filename); restart=restart,
                               restart_time_index=restart_time_index)
end

"""
overload with no TimerOutput arguments
"""
function run_moment_kinetics(input; restart=false, restart_time_index=-1)
    return run_moment_kinetics(nothing, input; restart=restart,
                               restart_time_index=restart_time_index)
end

"""
overload which gets the input file name from command line arguments
"""
function run_moment_kinetics()
    options = get_options()
    inputfile = options["inputfile"]
    restart = options["restart"]
    if options["restartfile"] !== nothing
        restart = options["restartfile"]
    end
    restart_time_index = options["restart-time-index"]
    if inputfile === nothing
        this_input = Dict()
    else
        this_input = inputfile
    end
    run_moment_kinetics(this_input; restart=restart,
                        restart_time_index=restart_time_index)
end

"""
Append a number to the filename, to get a new, non-existing filename to backup the file
to.
"""
function get_backup_filename(filename)
    if !isfile(filename)
        error("Requested to restart from $filename, but this file does not exist")
    end
    counter = 1
    temp, extension = splitext(filename)
    extension = extension[2:end]
    temp, iblock_or_type = splitext(temp)
    iblock_or_type = iblock_or_type[2:end]
    iblock = nothing
    basename = nothing
    type = nothing
    if iblock_or_type == "dfns"
        iblock = nothing
        type = iblock_or_type
        basename = temp
        parallel_io = true
    else
        # Filename had an iblock, so we are not using parallel I/O, but actually want to
        # use the iblock for this block, not necessarily for the exact file that was
        # passed.
        iblock = iblock_index[]
        basename, type = splitext(temp)
        type = type[2:end]
        parallel_io = false
    end
    if type != "dfns"
        error("Must pass the '.dfns.h5' output file for restarting. Got $filename.")
    end
    backup_dfns_filename = ""
    if parallel_io
        # Using parallel I/O
        while true
            backup_dfns_filename = "$(basename)_$(counter).$(type).$(extension)"
            if !isfile(backup_dfns_filename)
                break
            end
            counter += 1
        end
        # Create dfns_filename here even though it is the filename passed in, as
        # parallel_io=false branch needs to get the right `iblock` for this block.
        dfns_filename = "$(basename).dfns.$(extension)"
        moments_filename = "$(basename).moments.$(extension)"
        backup_moments_filename = "$(basename)_$(counter).moments.$(extension)"
    else
        while true
            backup_dfns_filename = "$(basename)_$(counter).$(type).$(iblock).$(extension)"
            if !isfile(backup_dfns_filename)
                break
            end
            counter += 1
        end
        # Create dfns_filename here even though it is almost the filename passed in, in
        # order to get the right `iblock` for this block.
        dfns_filename = "$(basename).dfns.$(iblock).$(extension)"
        moments_filename = "$(basename).moments.$(iblock).$(extension)"
        backup_moments_filename = "$(basename)_$(counter).moments.$(iblock).$(extension)"
    end
    backup_dfns_filename == "" && error("Failed to find a name for backup file.")
    backup_prefix_iblock = ("$(basename)_$(counter)", iblock)
    original_prefix_iblock = (basename, iblock)
    return dfns_filename, backup_dfns_filename, parallel_io, moments_filename,
           backup_moments_filename, backup_prefix_iblock, original_prefix_iblock
end

"""
Perform all the initialization steps for a run.

If `backup_filename` is `nothing`, set up for a regular run; if a filename is passed,
reload data from time index given by `restart_time_index` for a restart.

`debug_loop_type` and `debug_loop_parallel_dims` are used to force specific set ups for
parallel loop ranges, and are only used by the tests in `debug_test/`.
"""
function setup_moment_kinetics(input_dict::AbstractDict;
        restart::Union{Bool,AbstractString}=false, restart_time_index::mk_int=-1,
        debug_loop_type::Union{Nothing,NTuple{N,Symbol} where N}=nothing,
        debug_loop_parallel_dims::Union{Nothing,NTuple{N,Symbol} where N}=nothing)

    setup_start_time = now()

    # Set up MPI
    initialize_comms!()

    if global_rank[] == 0
        println("Starting setup   ", Dates.format(now(), dateformat"H:MM:SS"))
        flush(stdout)
    end

    input = mk_input(input_dict; save_inputs_to_txt=true, ignore_MPI=false)
    # obtain input options from moment_kinetics_input.jl
    # and check input to catch errors
    io_input, evolve_moments, t_input, z, z_spectral, r, r_spectral, vpa, vpa_spectral,
        vperp, vperp_spectral, gyrophase, gyrophase_spectral, vz, vz_spectral, vr,
        vr_spectral, vzeta, vzeta_spectral, composition, species, collisions, geometry,
        gyroavs, drive_input, external_source_settings, num_diss_params,
        manufactured_solns_input = input

    # Create loop range variables for shared-memory-parallel loops
    if debug_loop_type === nothing
        # Non-debug case used for all simulations
        looping.setup_loop_ranges!(block_rank[], block_size[];
                                   s=composition.n_ion_species,
                                   sn=composition.n_neutral_species,
                                   r=r.n, z=z.n, vperp=vperp.n, vpa=vpa.n,
                                   vzeta=vzeta.n, vr=vr.n, vz=vz.n)
    else
        if debug_loop_parallel_dims === nothing
            error("debug_loop_parallel_dims must not be `nothing` when debug_loop_type "
                  * "is not `nothing`.")
        end
        # Debug initialisation only used by tests in `debug_test/`
        debug_setup_loop_ranges_split_one_combination!(
            block_rank[], block_size[], debug_loop_type, debug_loop_parallel_dims...;
            s=composition.n_ion_species, sn=composition.n_neutral_species, r=r.n, z=z.n,
            vperp=vperp.n, vpa=vpa.n, vzeta=vzeta.n, vr=vr.n, vz=vz.n)
    end

    # Allocate arrays and create the pdf and moments structs
    pdf, moments, boundary_distributions =
        allocate_pdf_and_moments(composition, r, z, vperp, vpa, vzeta, vr, vz,
                                 evolve_moments, collisions, external_source_settings,
                                 num_diss_params)

    if restart === false
        restarting = false
        # initialize f(z,vpa) and the lowest three v-space moments (density(z), upar(z) and ppar(z)),
        # each of which may be evolved separately depending on input choices.
        init_pdf_and_moments!(pdf, moments, boundary_distributions, geometry,
                              composition, r, z, vperp, vpa, vzeta, vr, vz,
                              vpa_spectral, vz_spectral, species,
                              external_source_settings, manufactured_solns_input)
        # initialize time variable
        code_time = 0.
        previous_runs_info = nothing
    else
        restarting = true

        run_name = input_dict["run_name"]
        base_directory = get(input_dict, "base_directory", "runs")
        output_dir = joinpath(base_directory, run_name)
        if restart === true
            run_name = input_dict["run_name"]
            io_settings = get(input_dict, "output", Dict{String,Any}())
            binary_format = get(io_settings, "binary_format", hdf5)
            if binary_format === hdf5
                ext = "h5"
            elseif binary_format === netcdf
                ext = "cdf"
            else
                error("Unrecognized binary_format '$binary_format'")
            end
            restart_filename_pattern = joinpath(output_dir, run_name * ".dfns*." * ext)
            restart_filename_glob = glob(restart_filename_pattern)
            if length(restart_filename_glob) == 0
                error("No output file to restart from found matching the pattern "
                      * "$restart_filename_pattern")
            end
            restart_filename = restart_filename_glob[1]
        else
            restart_filename = restart
        end

        # Move the output file being restarted from to make sure it doesn't get
        # overwritten.
        dfns_filename, backup_dfns_filename, parallel_io, moments_filename,
        backup_moments_filename, backup_prefix_iblock, original_prefix_iblock =
            get_backup_filename(restart_filename)

        # Ensure every process got the filenames and checked files exist before moving
        # files
        MPI.Barrier(comm_world)

        if abspath(output_dir) == abspath(dirname(dfns_filename))
            # Only move the file if it is in our current run directory. Otherwise we are
            # restarting from another run, and will not be overwriting the file.
            if (parallel_io && global_rank[] == 0) || (!parallel_io && block_rank[] == 0)
                mv(dfns_filename, backup_dfns_filename)
                mv(moments_filename, backup_moments_filename)
            end
        else
            # Reload from dfns_filename without moving the file
            backup_prefix_iblock = original_prefix_iblock
        end

        # Ensure files have been moved before any process tries to read from them
        MPI.Barrier(comm_world)

        # Reload pdf and moments from an existing output file
        code_time, previous_runs_info, restart_time_index =
            reload_evolving_fields!(pdf, moments, boundary_distributions,
                                    backup_prefix_iblock, restart_time_index,
                                    composition, geometry, r, z, vpa, vperp, vzeta, vr,
                                    vz)

        # Re-initialize the source amplitude here instead of loading it from the restart
        # file so that we can change the settings between restarts.
        initialize_external_source_amplitude!(moments, external_source_settings, vperp,
                                              vzeta, vr, composition.n_neutral_species)

        _block_synchronize()
    end
    # create arrays and do other work needed to setup
    # the main time advance loop -- including normalisation of f by density if requested

    moments, fields, spectral_objects, advect_objects,
    scratch, advance, fp_arrays, scratch_dummy, manufactured_source_list =
        setup_time_advance!(pdf, vz, vr, vzeta, vpa, vperp, z, r, vz_spectral,
            vr_spectral, vzeta_spectral, vpa_spectral, vperp_spectral, z_spectral,
            r_spectral, composition, drive_input, moments, t_input, collisions, species,
            geometry, gyroavs, boundary_distributions, external_source_settings, num_diss_params,
            manufactured_solns_input, restarting)

    # This is the closest we can get to the end time of the setup before writing it to the
    # output file
    setup_end_time = now()
    time_for_setup = to_minutes(setup_end_time - setup_start_time)
    # setup i/o
    ascii_io, io_moments, io_dfns = setup_file_io(io_input, boundary_distributions, vz,
        vr, vzeta, vpa, vperp, z, r, composition, collisions, moments.evolve_density,
        moments.evolve_upar, moments.evolve_ppar, external_source_settings, input_dict,
        restart_time_index, previous_runs_info, time_for_setup)
    # write initial data to ascii files
    write_data_to_ascii(moments, fields, vpa, vperp, z, r, code_time,
        composition.n_ion_species, composition.n_neutral_species, ascii_io)
    # write initial data to binary files

    write_moments_data_to_binary(moments, fields, code_time, composition.n_ion_species,
        composition.n_neutral_species, io_moments, 1, 0.0, r, z)
    write_dfns_data_to_binary(pdf.charged.norm, pdf.neutral.norm, moments, fields,
         code_time, composition.n_ion_species, composition.n_neutral_species, io_dfns, 1,
         0.0, r, z, vperp, vpa, vzeta, vr, vz)

    begin_s_r_z_vperp_region()

    return pdf, scratch, code_time, t_input, vz, vr, vzeta, vpa, vperp, gyrophase, z, r,
           moments, fields, spectral_objects, advect_objects,
           composition, collisions, geometry, gyroavs, boundary_distributions,
           external_source_settings, num_diss_params, advance, fp_arrays, scratch_dummy,
           manufactured_source_list, ascii_io, io_moments, io_dfns
end

"""
Clean up after a run
"""
function cleanup_moment_kinetics!(ascii_io, io_moments, io_dfns)
    @debug_detect_redundant_block_synchronize begin
        # Disable check for redundant _block_synchronize() during finalization, as this
        # only runs once so any failure is not important.
        debug_detect_redundant_is_active[] = false
    end

    begin_serial_region()

    # finish i/o
    finish_file_io(ascii_io, io_moments, io_dfns)

    @serial_region begin
        if global_rank[] == 0
            println("finished file io         ",
               Dates.format(now(), dateformat"H:MM:SS"))
        end
    end

    # clean up MPI objects
    finalize_comms!()

    return nothing
end

end
