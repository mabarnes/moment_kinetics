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
include("runge_kutta.jl")
include("reference_parameters.jl")
include("coordinates.jl")
include("file_io.jl")
include("geo.jl")
include("velocity_moments.jl")
include("velocity_grid_transforms.jl")
include("electron_fluid_equations.jl")
include("em_fields.jl")
include("bgk.jl")
include("manufactured_solns.jl") # MRH Here?
include("external_sources.jl")
include("moment_constraints.jl")
include("fokker_planck_test.jl")
include("fokker_planck_calculus.jl")
include("fokker_planck.jl")
include("advection.jl")
include("vpa_advection.jl")
include("z_advection.jl")
include("r_advection.jl")
include("vperp_advection.jl")
include("electron_z_advection.jl")
include("electron_vpa_advection.jl")
include("neutral_r_advection.jl")
include("neutral_z_advection.jl")
include("neutral_vz_advection.jl")
include("boundary_conditions.jl")
include("charge_exchange.jl")
include("ionization.jl")
include("krook_collisions.jl")
include("continuity.jl")
include("energy_equation.jl")
include("force_balance.jl")
include("source_terms.jl")
include("numerical_dissipation.jl")
include("moment_kinetics_input.jl")
include("utils.jl")
include("load_data.jl")
include("analysis.jl")
include("electron_kinetic_equation.jl")
include("initial_conditions.jl")
include("parameter_scans.jl")
include("time_advance.jl")

using TimerOutputs
using Dates
using Glob
using Primes

using .file_io: setup_file_io, finish_file_io
using .file_io: write_data_to_ascii
using .file_io: write_all_moments_data_to_binary, write_all_dfns_data_to_binary
using .command_line_options: get_options
using .communication
using .communication: _block_synchronize
using .debugging
using .external_sources
using .input_structs
using .initial_conditions: allocate_pdf_and_moments, init_pdf_and_moments!
using .load_data: reload_evolving_fields!
using .looping
using .moment_constraints: hard_force_moment_constraints!
using .looping: debug_setup_loop_ranges_split_one_combination!
using .moment_kinetics_input: mk_input, read_input_file
using .time_advance: setup_time_advance!, time_advance!
using .type_definitions: mk_int
using .utils: to_minutes, get_default_restart_filename,
              get_prefix_iblock_and_move_existing_file
using .em_fields: setup_em_fields
using .time_advance: setup_dummy_and_buffer_arrays
using .time_advance: allocate_advection_structs

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

        if global_rank[] == 0 && to !== nothing
            # Print the timing information if this is a performance test
            display(to)
            println()
        end
    catch e
        # Stop code from hanging when running on multiple processes if only one of them
        # throws an error
        if global_size[] > 1
            println("$(typeof(e)) on process $(global_rank[]):")
            showerror(stdout, e, catch_backtrace())
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
        drive_input, external_source_settings, num_diss_params,
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

    # create the "fields" structure that contains arrays
    # for the electrostatic potential phi and the electromagnetic fields
    fields = setup_em_fields(z.n, r.n, drive_input.force_phi, drive_input.amplitude,
                             drive_input.frequency, drive_input.force_Er_zero_at_wall)

    # Allocate arrays and create the pdf and moments structs
    pdf, moments, boundary_distributions =
        allocate_pdf_and_moments(composition, r, z, vperp, vpa, vzeta, vr, vz,
                                 evolve_moments, collisions, external_source_settings,
                                 num_diss_params, t_input)

    # create structs containing the information needed to treat advection in z, r, vpa, vperp, and vz
    # for ions, electrons and neutrals
    # NB: the returned advection_structs are yet to be initialized
    advection_structs = allocate_advection_structs(composition, z, r, vpa, vperp, vz, vr, vzeta)

    # setup dummy arrays & buffer arrays for z r MPI                             
    n_neutral_species_alloc = max(1, composition.n_neutral_species)
    scratch_dummy = setup_dummy_and_buffer_arrays(r.n, z.n, vpa.n, vperp.n, vz.n, vr.n, vzeta.n, 
        composition.n_ion_species, n_neutral_species_alloc)

    if restart === false
        restarting = false
        # initialize f(z,vpa) and the lowest three v-space moments (density(z), upar(z) and ppar(z)),
        # each of which may be evolved separately depending on input choices.
        init_pdf_and_moments!(pdf, moments, fields, boundary_distributions, geometry,
                              composition, r, z, vperp, vpa, vzeta, vr, vz,
                              z_spectral, r_spectral, vperp_spectral, vpa_spectral,
                              vz_spectral, species, collisions, external_source_settings,
                              manufactured_solns_input, scratch_dummy, t_input,
                              num_diss_params, advection_structs, io_input, input_dict)
        # initialize time variable
        code_time = 0.
        dt = nothing
        dt_before_last_fail = nothing
        electron_dt = nothing
        electron_dt_before_last_fail = nothing
        previous_runs_info = nothing
        restart_had_kinetic_electrons = false
    else
        restarting = true

        if restart === true
            restart_filename = get_default_restart_filename(io_input, "dfns")
        else
            restart_filename = restart
        end

        backup_prefix_iblock = get_prefix_iblock_and_move_existing_file(restart_filename,
                                                                        io_input.output_dir)

        # Reload pdf and moments from an existing output file
        code_time, dt, dt_before_last_fail, electron_dt, electron_dt_before_last_fail,
        previous_runs_info, restart_time_index, restart_had_kinetic_electrons =
            reload_evolving_fields!(pdf, moments, boundary_distributions,
                                    backup_prefix_iblock, restart_time_index,
                                    composition, geometry, r, z, vpa, vperp, vzeta, vr,
                                    vz; run_directory=io_input.output_dir)

        begin_serial_region()
        @serial_region begin
            @. moments.electron.temp = moments.electron.vth^2
        end
        if composition.electron_physics == kinetic_electrons
            begin_r_z_vperp_vpa_region()
            @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
                pdf.electron.pdf_before_ion_timestep[ivpa,ivperp,iz,ir] =
                    pdf.electron.norm[ivpa,ivperp,iz,ir]
            end
        end

        # Re-initialize the source amplitude here instead of loading it from the restart
        # file so that we can change the settings between restarts.
        initialize_external_source_amplitude!(moments, external_source_settings, vperp,
                                              vzeta, vr, composition.n_neutral_species)

        _block_synchronize()
    end

    # Broadcast code_time from the root process of each shared-memory block (on which it
    # might have been loaded from a restart file).
    code_time = MPI.Bcast(code_time, 0, comm_block[])

    # create arrays and do other work needed to setup
    # the main time advance loop -- including normalisation of f by density if requested

    moments, fields, spectral_objects, scratch, advance, t_params, fp_arrays,
    manufactured_source_list =
        setup_time_advance!(pdf, fields, vz, vr, vzeta, vpa, vperp, z, r, vz_spectral,
            vr_spectral, vzeta_spectral, vpa_spectral, vperp_spectral, z_spectral,
            r_spectral, composition, drive_input, moments, t_input, code_time, dt,
            dt_before_last_fail, electron_dt, electron_dt_before_last_fail, collisions,
            species, geometry, boundary_distributions, external_source_settings,
            num_diss_params, manufactured_solns_input, advection_structs, scratch_dummy,
            io_input, restarting, restart_had_kinetic_electrons, input_dict)

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
    write_data_to_ascii(moments, fields, z, r, code_time, composition.n_ion_species,
        composition.n_neutral_species, ascii_io)
    # write initial data to binary files

    write_all_moments_data_to_binary(moments, fields, code_time,
        composition.n_ion_species, composition.n_neutral_species, io_moments, 1, 0.0,
        t_params, r, z)
    write_all_dfns_data_to_binary(pdf, moments, fields, code_time,
        composition.n_ion_species, composition.n_neutral_species, io_dfns, 1, 0.0,
        t_params, r, z, vperp, vpa, vzeta, vr, vz)

    begin_s_r_z_vperp_region()

    return pdf, scratch, code_time, t_params, vz, vr, vzeta, vpa, vperp, gyrophase, z, r,
           moments, fields, spectral_objects, advection_structs,
           composition, collisions, geometry, boundary_distributions,
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
