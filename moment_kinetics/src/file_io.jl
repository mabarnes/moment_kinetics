"""
"""
module file_io

export input_option_error
export get_group
export open_output_file, open_ascii_output_file
export setup_io_input
export setup_file_io, finish_file_io
export write_data_to_ascii

using ..communication
using ..coordinates: coordinate
using ..debugging
using ..input_structs
using ..looping
using ..timer_utils
using ..timer_utils: timer_names_all_ranks_moments, timer_names_all_ranks_dfns,
                     TimerNamesDict, SortedDict
using ..moment_kinetics_structs: scratch_pdf, em_fields_struct
using ..type_definitions: mk_float, mk_int

# Import moment_kinetics so we can refer to it in docstrings
import ..moment_kinetics

using LibGit2
using MPI
using Pkg
using UUIDs

@debug_shared_array using ..communication: DebugMPISharedArray

function __init__()
    try
        # Try to load the NCDatasets package.  If the package is not installed, then
        # NetCDF I/O will not be available.
        Base.require(Main, :NCDatasets)
    catch
        # Do nothing
    end
end

"""
Container for I/O settings
"""
Base.@kwdef struct io_input_struct
    run_name::String
    base_directory::String
    ascii_output::Bool
    binary_format::binary_format_type
    parallel_io::Bool
    run_id::String
    output_dir::String
    write_error_diagnostics::Bool
    write_steady_state_diagnostics::Bool
    write_electron_error_diagnostics::Bool
    write_electron_steady_state_diagnostics::Bool
    display_timing_info::Bool
end

"""
structure containing the various input/output streams
"""
struct ascii_ios{T <: Union{IOStream,Nothing}}
    # corresponds to the ascii file to which the distribution function is written
    ff::T
    # corresponds to the ascii file to which velocity space moments of the
    # distribution function such as density and pressure are written
    moments_ion::T
    moments_electron::T
    moments_neutral::T
    # corresponds to the ascii file to which electromagnetic fields
    # such as the electrostatic potential are written
    fields::T
end

"""
structure containing the data/metadata needed for binary file i/o
moments & fields only
"""
struct io_moments_info{Tfile, Ttime, Tphi, Tmomi, Tmome, Tmomn, Tchodura_lower,
                       Tchodura_upper, Texti1, Texti2, Texti3, Texti4,
                       Texti5, Textn1, Textn2, Textn3, Textn4, Textn5, Texte1, Texte2,
                       Texte3, Texte4, Tconstri, Tconstrn, Tconstre, Tint, Telectrontime,
                       Telectronint, Tnldiagnostics}
    # file identifier for the binary file to which data is written
    fid::Tfile
    # handle for the time variable
    time::Ttime
    # handle for the electrostatic potential variable
    phi::Tphi
    # handle for the radial electric field variable
    Er::Tphi
    # handle for the z electric field variable
    Ez::Tphi
    # handle for the ion species density
    density::Tmomi
    # low-order approximation, used to diagnose timestepping error
    density_loworder::Union{Tmomi,Nothing}
    # start of the last timestep before output, used to measure steady state residual
    density_start_last_timestep::Union{Tmomi,Nothing}
    # handle for the ion species parallel flow
    parallel_flow::Tmomi
    # low-order approximation, used to diagnose timestepping error
    parallel_flow_loworder::Union{Tmomi,Nothing}
    # start of the last timestep before output, used to measure steady state residual
    parallel_flow_start_last_timestep::Union{Tmomi,Nothing}
    # handle for the ion species pressure
    pressure::Tmomi
    # low-order approximation, used to diagnose timestepping error
    pressure_loworder::Union{Tmomi,Nothing}
    # start of the last timestep before output, used to measure steady state residual
    pressure_start_last_timestep::Union{Tmomi,Nothing}
    # handle for the ion species parallel pressure
    parallel_pressure::Tmomi
    # handle for the ion species perpendicular pressure
    perpendicular_pressure::Tmomi
    # handle for the ion species parallel heat flux
    parallel_heat_flux::Tmomi
    # handle for the ion species thermal speed
    thermal_speed::Tmomi
    # handle for the ion species entropy production
    entropy_production::Tmomi
    # handle for chodura diagnostic (lower)
    chodura_integral_lower::Tchodura_lower
    # handle for chodura diagnostic (upper)
    chodura_integral_upper::Tchodura_upper
    # handle for the electron species density
    electron_density::Tmome
    # low-order approximation, used to diagnose timestepping error
    electron_density_loworder::Union{Tmome,Nothing}
    # start of the last timestep before output, used to measure steady state residual
    electron_density_start_last_timestep::Union{Tmome,Nothing}
    # handle for the electron species parallel flow
    electron_parallel_flow::Tmome
    # low-order approximation, used to diagnose timestepping error
    electron_parallel_flow_loworder::Union{Tmome,Nothing}
    # start of the last timestep before output, used to measure steady state residual
    electron_parallel_flow_start_last_timestep::Union{Tmome,Nothing}
    # handle for the electron species pressure
    electron_pressure::Tmome
    # low-order approximation, used to diagnose timestepping error
    electron_pressure_loworder::Union{Tmome,Nothing}
    # start of the last timestep before output, used to measure steady state residual
    electron_pressure_start_last_timestep::Union{Tmome,Nothing}
    # handle for the electron species parallel pressure
    electron_parallel_pressure::Tmome
    # handle for the electron species parallel heat flux
    electron_parallel_heat_flux::Tmome
    # handle for the electron species thermal speed
    electron_thermal_speed::Tmome

    # handle for the neutral species density
    density_neutral::Tmomn
    # low-order approximation, used to diagnose timestepping error
    density_neutral_loworder::Union{Tmomn,Nothing}
    # start of the last timestep before output, used to measure steady state residual
    density_neutral_start_last_timestep::Union{Tmomn,Nothing}
    uz_neutral::Tmomn
    # low-order approximation, used to diagnose timestepping error
    uz_neutral_loworder::Union{Tmomn,Nothing}
    # start of the last timestep before output, used to measure steady state residual
    uz_neutral_start_last_timestep::Union{Tmomn,Nothing}
    p_neutral::Tmomn
    # low-order approximation, used to diagnose timestepping error
    p_neutral_loworder::Union{Tmomn,Nothing}
    # start of the last timestep before output, used to measure steady state residual
    p_neutral_start_last_timestep::Union{Tmomn,Nothing}
    pz_neutral::Tmomn
    qz_neutral::Tmomn
    thermal_speed_neutral::Tmomn

    # handles for external source variables
    external_source_amplitude::Texti1
    external_source_T_array::Texti1
    external_source_density_amplitude::Texti2
    external_source_momentum_amplitude::Texti3
    external_source_pressure_amplitude::Texti4
    external_source_controller_integral::Texti5
    external_source_neutral_amplitude::Textn1
    external_source_neutral_T_array::Textn1
    external_source_neutral_density_amplitude::Textn2
    external_source_neutral_momentum_amplitude::Textn3
    external_source_neutral_pressure_amplitude::Textn4
    external_source_neutral_controller_integral::Textn5
    external_source_electron_amplitude::Texte1
    external_source_electron_T_array::Texte1
    external_source_electron_density_amplitude::Texte2
    external_source_electron_momentum_amplitude::Texte3
    external_source_electron_pressure_amplitude::Texte4

    # handles for constraint coefficients
    ion_constraints_A_coefficient::Tconstri
    ion_constraints_B_coefficient::Tconstri
    ion_constraints_C_coefficient::Tconstri
    neutral_constraints_A_coefficient::Tconstrn
    neutral_constraints_B_coefficient::Tconstrn
    neutral_constraints_C_coefficient::Tconstrn
    electron_constraints_A_coefficient::Tconstre
    electron_constraints_B_coefficient::Tconstre
    electron_constraints_C_coefficient::Tconstre

    # cumulative wall clock time taken by the run
    time_for_run::Ttime
    # cumulative number of timesteps taken
    step_counter::Tint
    # current timestep size
    dt::Ttime
    # size of last timestep before output, used for some diagnostics
    previous_dt::Ttime
    # cumulative number of timestep failures
    failure_counter::Tint
    # Last successful timestep before most recent timestep failure, used by adaptve
    # timestepping algorithm
    dt_before_last_fail::Ttime
    # cumulative number of electron pseudo-timesteps taken
    electron_step_counter::Telectronint
    # cumulative electron pseudo-time
    electron_cumulative_pseudotime::Telectrontime
    # current electron pseudo-timestep size
    electron_dt::Telectrontime
    # size of last electron pseudo-timestep before the output was written
    electron_previous_dt::Telectrontime
    # cumulative number of electron pseudo-timestep failures
    electron_failure_counter::Telectronint
    # Last successful timestep before most recent electron pseudo-timestep failure, used
    # by adaptive timestepping algorithm
    electron_dt_before_last_fail::Telectrontime
    # Variables recording diagnostic information about non-linear solvers (used for
    # implicit parts of timestep). These are stored in nested NamedTuples so that we can
    # write diagnostics generically for as many nonlinear solvers as are created.
    nl_solver_diagnostics::Tnldiagnostics

    # Settings for I/O
    io_input::io_input_struct
 end

"""
structure containing the data/metadata needed for binary file i/o
distribution function data only
"""
struct io_dfns_info{Tfile, Tfi, Tfe, Tfn, Tmoments}
    # file identifier for the binary file to which data is written
    fid::Tfile
    # handle for the ion species distribution function variable
    f::Tfi
    # low-order approximation to ion species distribution function, used to diagnose timestepping error
    f_loworder::Union{Tfi,Nothing}
    # ion species distribution function at the start of the last timestep before output, used to measure steady state residual
    f_start_last_timestep::Union{Tfi,Nothing}
    # handle for the electron distribution function variable
    f_electron::Tfe
    # low-order approximation to electron distribution function, used to diagnose timestepping error
    f_electron_loworder::Union{Tfe,Nothing}
    # electron distribution function at the start of the last timestep before output, used to measure steady state residual
    f_electron_start_last_timestep::Union{Tfe,Nothing}
    # handle for the neutral species distribution function variable
    f_neutral::Tfn
    # low-order approximation to neutral species distribution function, used to diagnose timestepping error
    f_neutral_loworder::Union{Tfn,Nothing}
    # neutral species distribution function at the start of the last timestep before output, used to measure steady state residual
    f_neutral_start_last_timestep::Union{Tfn,Nothing}

    # Settings for I/O
    io_input::io_input_struct

    # Handles for moment variables
    io_moments::Tmoments
end

"""
structure containing the data/metadata needed for binary file i/o
for electron initialization
"""
struct io_initial_electron_info{Tfile, Tfe, Tmom, Texte1, Texte2, Texte3, Texte4,
                                Tconstr, Telectrontime, Telectronint}
    # file identifier for the binary file to which data is written
    fid::Tfile
    time::Telectrontime
    # handle for the electron distribution function variable
    f_electron::Tfe
    # low-order approximation, used to diagnose timestepping error
    f_electron_loworder::Union{Tfe,Nothing}
    # start of the last timestep before output, used to measure steady state residual
    f_electron_start_last_timestep::Union{Tfe,Nothing}
    # handle for the electron density variable
    electron_density::Union{Tmom,Nothing}
    # low-order approximation, used to diagnose timestepping error
    electron_density_loworder::Union{Tmom,Nothing}
    # start of the last timestep before output, used to measure steady state residual
    electron_density_start_last_timestep::Union{Tmom,Nothing}
    # handle for the electron parallel flow variable
    electron_parallel_flow::Union{Tmom,Nothing}
    # low-order approximation, used to diagnose timestepping error
    electron_parallel_flow_loworder::Union{Tmom,Nothing}
    # start of the last timestep before output, used to measure steady state residual
    electron_parallel_flow_start_last_timestep::Union{Tmom,Nothing}
    # handle for the electron pressure variable
    electron_pressure::Tmom
    # low-order approximation, used to diagnose timestepping error
    electron_pressure_loworder::Union{Tmom,Nothing}
    # start of the last timestep before output, used to measure steady state residual
    electron_pressure_start_last_timestep::Union{Tmom,Nothing}
    # handle for the electron parallel pressure variable
    electron_parallel_pressure::Tmom
    # handle for the electron parallel heat flux variable
    electron_parallel_heat_flux::Tmom
    # handle for the electron thermal speed variable
    electron_thermal_speed::Tmom
    # handles for external source terms
    external_source_electron_amplitude::Texte1
    external_source_electron_T_array::Texte1
    external_source_electron_density_amplitude::Texte2
    external_source_electron_momentum_amplitude::Texte3
    external_source_electron_pressure_amplitude::Texte4
    # handles for constraint coefficients
    electron_constraints_A_coefficient::Tconstr
    electron_constraints_B_coefficient::Tconstr
    electron_constraints_C_coefficient::Tconstr
    # Electrostatic potential stored to save the value set by the electron boundary
    # condition.
    phi::Tmom
    # cumulative number of electron pseudo-timesteps taken
    electron_step_counter::Telectronint
    # local electron pseudo-time
    electron_local_pseudotime::Telectrontime
    # cumulative electron pseudo-time
    electron_cumulative_pseudotime::Telectrontime
    # current residual for the electron pseudo-timestepping loop
    electron_residual::Telectrontime
    # current electron pseudo-timestep size
    electron_dt::Telectrontime
    # size of last electron pseudo-timestep before the output was written
    electron_previous_dt::Telectrontime
    # cumulative number of electron pseudo-timestep failures
    electron_failure_counter::Telectronint
    # Last successful timestep before most recent electron pseudo-timestep failure, used
    # by adaptve timestepping algorithm
    electron_dt_before_last_fail::Telectrontime

    # Settings for I/O
    io_input::io_input_struct
end

"""
Read the settings for I/O
"""
function setup_io_input(input_dict, timestepping_section, warn_unexpected::Bool;
                        ignore_MPI=false, write_output=true)
    io_settings = set_defaults_and_check_section!(
        input_dict, "output", warn_unexpected;
        run_name="",
        base_directory="runs",
        ascii_output=false,
        binary_format=hdf5,
        parallel_io="",
        display_timing_info=true,
       )
    if io_settings["run_name"] == ""
        error("When passing a Dict directly for input, it is required to set `run_name` "
              * "in the `[output]` section")
    end
    if io_settings["parallel_io"] == ""
        io_settings["parallel_io"] = io_has_parallel(Val(io_settings["binary_format"]))
    end
    # Make copy of the section to avoid modifying the passed-in Dict
    io_settings = copy(io_settings)
    run_id = string(uuid4())
    if !ignore_MPI
        # Communicate run_id to all blocks
        # Need to convert run_id to a Vector{Char} for MPI
        run_id_chars = [run_id...]
        MPI.Bcast!(run_id_chars, 0, comm_world)
        run_id = string(run_id_chars...)
    end
    io_settings["run_id"] = run_id
    io_settings["output_dir"] = joinpath(io_settings["base_directory"], io_settings["run_name"])
    io_settings["write_error_diagnostics"] = timestepping_section["write_error_diagnostics"]
    io_settings["write_steady_state_diagnostics"] = timestepping_section["write_steady_state_diagnostics"]
    io_settings["write_electron_error_diagnostics"] = timestepping_section["electron_t_input"]["write_error_diagnostics"]
    io_settings["write_electron_steady_state_diagnostics"] = timestepping_section["electron_t_input"]["write_steady_state_diagnostics"]

    # Create output_dir if it does not exist.
    if !(ignore_MPI || !write_output)
        if global_rank[] == 0
            mkpath(io_settings["output_dir"])
        end
        @_block_synchronize()
    end

    return io_input_struct(; (Symbol(k) => v for (k,v) ∈ io_settings)...)
end

"""
    io_has_parallel(Val(binary_format))

Test if the backend supports parallel I/O.

`binary_format` should be one of the values of the `binary_format_type` enum
"""
function io_has_parallel end

function io_has_implementation(::Val)
    # This method will be overridden by more specific versions (some only if an extension
    # is loaded).
    return false
end

"""
    check_io_implementation(binary_format)

Check that an implementation is available for `binary_format`, raising an error if not.
"""
function check_io_implementation(binary_format)
    if !io_has_implementation(Val(binary_format))
        if binary_format == netcdf
            error("NCDatasets is not installed, cannot use NetCDF I/O. Re-run "
                  * "machines/machine-setup.sh and activate NetCDF, or install "
                  * "NCDatasets.")
        else
            error("No implementation available for binary format $binary_format")
        end
    end
    return nothing
end

"""
open the necessary output files
"""
function setup_file_io(io_input, vz, vr, vzeta, vpa, vperp, z, r, composition, collisions,
                       evolve_density, evolve_upar, evolve_p, external_source_settings,
                       manufactured_source_list, input_dict, restart_time_index,
                       previous_runs_info, time_for_setup, t_params, nl_solver_params)

    @begin_serial_region()
    @serial_region begin
        # Only read/write from first process in each 'block'

        # check to see if output_dir exists in the current directory
        # if not, create it
        isdir(io_input.output_dir) || mkdir(io_input.output_dir)
        out_prefix = joinpath(io_input.output_dir, io_input.run_name)

        if io_input.ascii_output
            ff_io = open_ascii_output_file(out_prefix, "f_vs_t")
            mom_ion_io = open_ascii_output_file(out_prefix, "moments_ion_vs_t")
            mom_eon_io = open_ascii_output_file(out_prefix, "moments_electron_vs_t")
            mom_ntrl_io = open_ascii_output_file(out_prefix, "moments_neutral_vs_t")
            fields_io = open_ascii_output_file(out_prefix, "fields_vs_t")
            ascii = ascii_ios(ff_io, mom_ion_io, mom_eon_io, mom_ntrl_io, fields_io)
        else
            ascii = ascii_ios(nothing, nothing, nothing, nothing, nothing)
        end

        io_moments = setup_moments_io(out_prefix, io_input, vz, vr, vzeta, vpa, vperp, r,
                                      z, composition, collisions, evolve_density,
                                      evolve_upar, evolve_p, external_source_settings,
                                      manufactured_source_list, input_dict,
                                      comm_inter_block[], restart_time_index,
                                      previous_runs_info, time_for_setup, t_params,
                                      nl_solver_params)
        io_dfns = setup_dfns_io(out_prefix, io_input, r, z, vperp, vpa, vzeta, vr, vz,
                                composition, collisions, evolve_density, evolve_upar,
                                evolve_p, external_source_settings,
                                manufactured_source_list, restart_time_index, input_dict,
                                comm_inter_block[], previous_runs_info, time_for_setup,
                                t_params, nl_solver_params)

        return ascii, io_moments, io_dfns
    end
    # For other processes in the block, return objects with just the input.
    return nothing, (io_input=io_input,), (io_input=io_input,)
end

"""
open output file to save the initial electron pressure and distribution function
"""
function setup_electron_io(io_input, vpa, vperp, z, r, composition, collisions,
                           evolve_density, evolve_upar, evolve_p,
                           external_source_settings, t_params, input_dict,
                           restart_time_index, previous_runs_info, prefix_label;
                           ir=nothing)
    @begin_serial_region()
    @serial_region begin
        # Only read/write from first process in each 'block'

        # check to see if output_dir exists in the current directory
        # if not, create it
        isdir(io_input.output_dir) || mkdir(io_input.output_dir)
        out_prefix = joinpath(io_input.output_dir, io_input.run_name)

        run_id = io_input.run_id
        parallel_io = io_input.parallel_io
        io_comm = comm_inter_block[]

        electrons_prefix = string(out_prefix, ".$prefix_label")
        if !parallel_io
            electrons_prefix *= ".$(iblock_index[])"
        end
        if ir !== nothing
            electrons_prefix *= ".ir$ir"
        end
        fid, file_info = open_output_file(electrons_prefix, io_input, io_comm)

        # write a header to the output file
        add_attribute!(fid, "file_info",
                       "Output initial electron state from the moment_kinetics code")
        add_attribute!(fid, "pdf_electron_converged", false)

        # write some overview information to the output file
        write_overview!(fid, composition, collisions, parallel_io, evolve_density,
                        evolve_upar, evolve_p, -1.0)

        # write provenance tracking information to the output file
        write_provenance_tracking_info!(fid, parallel_io, run_id, restart_time_index,
                                        input_dict, previous_runs_info)

        # write the input settings
        write_input!(fid, input_dict, parallel_io)

        ### define coordinate dimensions ###
        if ir === nothing
            io_r = r
        else
            io_r = (n=1, n_global=1, ngrid=1, name="r", irank=0, nrank=1, L=r.L,
                    grid=[r.grid[ir]], wgts=[r.wgts[ir]], discretization=r.discretization,
                    finite_difference_option=r.finite_difference_option,
                    cheb_option=r.cheb_option, bc=r.bc,
                    element_spacing_option=r.element_spacing_option)
        end
        define_io_coordinates!(fid, nothing, nothing, nothing, vpa, vperp, z, io_r,
                               parallel_io)

        ### create variables for time-dependent quantities ###
        dynamic = create_io_group(fid, "dynamic_data", description="time evolving variables")

        # create groups to save timing data in
        timing = create_io_group(fid, "timing_data",
                                 description="timing data to check run-time performance")
        for i ∈ 0:global_size[]-1
            create_io_group(timing, "rank$i", description="timing data for MPI rank $i")
        end

        io_pseudotime = create_dynamic_variable!(dynamic, "time", mk_float; parallel_io=parallel_io,
                                                 description="pseudotime used for electron initialization")
        io_local_pseudotime = create_dynamic_variable!(dynamic, "electron_local_pseudotime", mk_float; parallel_io=parallel_io,
                                                       description="pseudotime within a single pseudotimestepping loop")
        io_electron_residual = create_dynamic_variable!(dynamic, "electron_residual", mk_float; parallel_io=parallel_io,
                                                        description="residual for electron pseudotimestepping loop")
        io_f_electron = create_dynamic_variable!(dynamic, "f_electron", mk_float, vpa,
                                                 vperp, z, io_r;
                                                 parallel_io=parallel_io,
                                                 description="electron distribution function")
        if io_input.write_electron_error_diagnostics
            io_f_electron_loworder =
                create_dynamic_variable!(dynamic, "f_electron_loworder", mk_float,
                                         vpa, vperp, z, io_r,
                                         parallel_io=parallel_io,
                                         description="low-order approximation to electron distribution function, used to diagnose timestepping error")
        else
            io_f_electron_loworder = nothing
        end
        if io_input.write_electron_steady_state_diagnostics
            io_f_electron_start_last_timestep =
                create_dynamic_variable!(dynamic, "f_electron_start_last_timestep",
                                         mk_float, vpa, vperp, z, io_r,
                                         parallel_io=parallel_io,
                                         description="electron distribution function at the start of the last electron pseudo-timestep before output, used to measure steady state residual")
        else
            io_f_electron_start_last_timestep = nothing
        end

        io_electron_density, io_electron_density_loworder,
        io_electron_density_start_last_timestep, io_electron_upar,
        io_electron_upar_loworder, io_electron_upar_start_last_timestep, io_electron_ppar,
        io_electron_ppar_loworder, io_electron_ppar_start_last_timestep, io_electron_qpar,
        io_electron_vth, external_source_electron_amplitude,
        external_source_electron_T_array, external_source_electron_density_amplitude,
        external_source_electron_momentum_amplitude,
        external_source_electron_pressure_amplitude,
        electron_constraints_A_coefficient, electron_constraints_B_coefficient,
        electron_constraints_C_coefficient, io_electron_step_counter, io_electron_dt,
        io_electron_previous_dt, io_electron_failure_counter,
        io_electron_dt_before_last_fail =
            define_dynamic_electron_moment_variables!(fid, r, z, parallel_io,
                                                      external_source_settings,
                                                      evolve_density, evolve_upar,
                                                      evolve_p, kinetic_electrons,
                                                      t_params,
                                                      io_input.write_electron_error_diagnostics,
                                                      io_input.write_electron_steady_state_diagnostics,
                                                      ir; electron_only_io=true)

        io_phi = create_dynamic_variable!(dynamic, "phi", mk_float, z, io_r;
                                          parallel_io=parallel_io,
                                          description="electrostatic potential",
                                          units="T_ref/e")

        close(fid)

        return file_info
    end
    # For other processes in the block, return an object with just the input.
    return (io_input=io_input,)
end

"""
Get the `file_info` for an existing electron I/O file
"""
function get_electron_io_info(io_input, prefix_label)
    out_prefix = joinpath(io_input.output_dir, io_input.run_name)
    electrons_prefix = string(out_prefix, ".$prefix_label")
    if io_input.binary_format == hdf5
        filename = string(electrons_prefix, ".h5")
    elseif io_input.binary_format == netcdf
        filename = string(electrons_prefix, ".cdf")
    else
        error("Unrecognized binary_format=$(io_input.binary_format)")
    end

    return (filename, io_input, comm_inter_block[])
end

"""
Reopen an existing initial electron output file to append more data
"""
function reopen_initial_electron_io(file_info, ir)
    if (ir === nothing && block_rank[] == 0) || anyzv_subblock_rank[] == 0
        # Only read/write from first process in each 'block' (for 'initial_electron' I/O)
        # or anyzv subblock (for debug I/O that is written independently for each `ir`).

        filename, io_input, io_comm = file_info
        if ir !== nothing
            prefix, suffix = splitext(filename)
            filename = prefix * ".ir$ir" * suffix
        end
        fid = reopen_output_file(filename, io_input, io_comm)
        dyn = get_group(fid, "dynamic_data")

        variable_list = get_variable_keys(dyn)
        function getvar(name)
            if name ∈ variable_list
                return dyn[name]
            else
                return nothing
            end
        end
        return io_initial_electron_info(fid, getvar("time"), getvar("f_electron"),
                                        getvar("f_electron_loworder"),
                                        getvar("f_electron_start_last_timestep"),
                                        getvar("electron_density"),
                                        getvar("electron_density_loworder"),
                                        getvar("electron_density_start_last_timestep"),
                                        getvar("electron_parallel_flow"),
                                        getvar("electron_parallel_flow_loworder"),
                                        getvar("electron_parallel_flow_start_last_timestep"),
                                        getvar("electron_pressure"),
                                        getvar("electron_pressure_loworder"),
                                        getvar("electron_pressure_start_last_timestep"),
                                        getvar("electron_parallel_pressure"),
                                        getvar("electron_parallel_heat_flux"),
                                        getvar("electron_thermal_speed"),
                                        getvar("external_source_electron_amplitude"),
                                        getvar("external_source_electron_T_array"),
                                        getvar("external_source_electron_density_amplitude"),
                                        getvar("external_source_electron_momentum_amplitude"),
                                        getvar("external_source_electron_pressure_amplitude"),
                                        getvar("electron_constraints_A_coefficient"),
                                        getvar("electron_constraints_B_coefficient"),
                                        getvar("electron_constraints_C_coefficient"),
                                        getvar("phi"),
                                        getvar("electron_step_counter"),
                                        getvar("electron_local_pseudotime"),
                                        getvar("electron_cumulative_pseudotime"),
                                        getvar("electron_residual"),
                                        getvar("electron_dt"),
                                        getvar("electron_previous_dt"),
                                        getvar("electron_failure_counter"),
                                        getvar("electron_dt_before_last_fail"),
                                        io_input)
    end

    # For processes other than the root process of each shared-memory group...
    return nothing
end


"""
Get a (sub-)group from a file or group
"""
function get_group end

"""
Test if a member of a (sub-)group is a group
"""
function is_group end

"""
Get names of all subgroups
"""
function get_subgroup_keys end

"""
Get names of all variables
"""
function get_variable_keys end

"""
    write_single_value!(file_or_group, name,
                        data::Union{Number, AbstractString, AbstractArray{T,N}},
                        coords::Union{coordinate,mk_int,NamedTuple}...; parallel_io,
                        description=nothing, units=nothing,
                        overwrite=false) where {T,N}

Write a single variable to a file or group. If a description or units are passed, add as
attributes of the variable.

If `overwrite=true` is passed, overwrite the variable if it already exists in the file.
Note that when overwriting a `String` variable, the new `String` must have exactly the
same length as the original `String`.
"""
function write_single_value! end

# Convert Enum values to String to be written to file
function write_single_value!(file_or_group, name, data::Enum; kwargs...)
    return write_single_value!(file_or_group, name, string(data); kwargs...)
end

"""
write some overview information for the simulation to the binary file
"""
function write_overview!(fid, composition, collisions, parallel_io, evolve_density,
                         evolve_upar, evolve_p, time_for_setup)
    @serial_region begin
        overview = create_io_group(fid, "overview")
        write_single_value!(overview, "nspecies", composition.n_species,
                            parallel_io=parallel_io,
                            description="total number of evolved plasma species")
        write_single_value!(overview, "n_ion_species", composition.n_ion_species,
                            parallel_io=parallel_io,
                            description="number of evolved ion species")
        write_single_value!(overview, "n_neutral_species", composition.n_neutral_species,
                            parallel_io=parallel_io,
                            description="number of evolved neutral species")
        write_single_value!(overview, "T_e", composition.T_e, parallel_io=parallel_io,
                            description="fixed electron temperature")
        write_single_value!(overview, "charge_exchange_frequency",
                            collisions.reactions.charge_exchange_frequency, parallel_io=parallel_io,
                            description="quantity related to the charge exchange frequency")
        write_single_value!(overview, "ionization_frequency",
                            collisions.reactions.ionization_frequency, parallel_io=parallel_io,
                            description="quantity related to the ionization frequency")
        write_single_value!(overview, "evolve_density", evolve_density,
                            parallel_io=parallel_io,
                            description="is density evolved separately from the distribution function?")
        write_single_value!(overview, "evolve_upar", evolve_upar,
                            parallel_io=parallel_io,
                            description="is parallel flow evolved separately from the distribution function?")
        write_single_value!(overview, "evolve_p", evolve_p, parallel_io=parallel_io,
                            description="is parallel pressure evolved separately from the distribution function?")
        write_single_value!(overview, "nrank", global_size[],
                            parallel_io=parallel_io,
                            description="Number of MPI ranks used for run.")
        write_single_value!(overview, "block_size", block_size[],
                            parallel_io=parallel_io,
                            description="Number of MPI ranks in each shared-memory block.")
        write_single_value!(overview, "parallel_io", parallel_io,
                            parallel_io=parallel_io,
                            description="is parallel I/O being used?")
        write_single_value!(overview, "time_for_setup", time_for_setup,
                            parallel_io=parallel_io,
                            description="time taken for setup of moment_kinetics (excluding file I/O)",
                            units="minutes")
    end
    return nothing
end

"""
Write time-independent information about manufactured solutions
"""
function write_manufactured_solutions!(fid, parallel_io, manufactured_source_list, vz, vr,
                                       vzeta, vpa, vperp, z, r, dfns::Bool)
    if manufactured_source_list === nothing
        # Not using manufactured solutions
        return nothing
    end

    @serial_region begin
        manufactured_solutions = create_io_group(fid, "manufactured_solutions")

        write_single_value!(manufactured_solutions, "Source_i_expression",
                            manufactured_source_list.Source_i_expression;
                            parallel_io=parallel_io,
                            description="Symbolic expression for ion manufactured source.")

        if :Source_i_array ∈ keys(manufactured_source_list) && length(manufactured_source_list.Source_i_array) > 0
            write_single_value!(manufactured_solutions, "Source_i_array",
                                manufactured_source_list.Source_i_array, vpa, vperp, r, z;
                                parallel_io=parallel_io,
                                description="Time-independent ion manufactured source array.")
        end

        write_single_value!(manufactured_solutions, "Source_n_expression",
                            manufactured_source_list.Source_n_expression;
                            parallel_io=parallel_io,
                            description="Symbolic expression for neutral manufactured source.")

        if :Source_n_array ∈ keys(manufactured_source_list) && length(manufactured_source_list.Source_n_array) > 0
            write_single_value!(manufactured_solutions, "Source_n_array",
                                manufactured_source_list.Source_n_array, vz, vr, vzeta, r, z;
                                parallel_io=parallel_io,
                                description="Time-independent neutral manufactured source array.")
        end
    end

    return nothing
end

"""
Write provenance tracking information, to allow runs to be reproduced.
"""
function write_provenance_tracking_info!(fid, parallel_io, run_id, restart_time_index,
                                         input_dict, previous_runs_info)

    @serial_region begin
        provenance_tracking = create_io_group(fid, "provenance_tracking")

        write_single_value!(provenance_tracking, "run_id", run_id,
                            parallel_io=parallel_io,
                            description="Unique identifier for the run")

        write_single_value!(provenance_tracking, "restart_time_index", restart_time_index,
                            parallel_io=parallel_io,
                            description="Index of the previous run from which this run " *
                                        "was restarted (if this value is negative, the " *
                                        "run is not a restart)")

        # Convert input_dict into a TOML-formatted string so that we can store it in a
        # single variable.
        io_buffer = IOBuffer()
        options_to_TOML(io_buffer, input_dict)
        input_string = String(take!(io_buffer))
        write_single_value!(provenance_tracking, "input", input_string,
                            parallel_io=parallel_io,
                            description="Input for the run, in TOML format")

        # Record the total number of MPI ranks, as this is required in addition to the
        # input to determine how many processes were in each shared-memory 'block'.
        write_single_value!(provenance_tracking, "n_mpi_ranks", global_size[],
                            parallel_io=parallel_io,
                            description="Total number of MPI ranks used for the run")

        # Get current git hash for code
        project_dir = dirname(dirname(dirname(@__FILE__)))
        repo = GitRepo(project_dir)
        git_commit_hash = string(LibGit2.GitHash(LibGit2.peel(LibGit2.GitCommit, LibGit2.head(repo))))
        if LibGit2.isdirty(repo)
            # Use a shell command to get the 'git diff' because it seems to be complicated
            # (if not impossible) to get this using LibGit2.
            # Use `setenv()` to run the command in `project_dir` without changing the
            # current working firectory.
            # Use `read()` rather than `run()` so that the command returns the terminal
            # output.
            # Finally need to convert the output to String as `read()` returns a
            # Vector{UInt8}.
            git_diff = String(read(setenv(`git diff`; dir=project_dir)))
        else
            git_diff = ""
        end
        write_single_value!(provenance_tracking, "git_commit_hash", git_commit_hash,
                            parallel_io=parallel_io,
                            description="git commit hash of moment_kinetics when this run was performed")
        write_single_value!(provenance_tracking, "git_diff", git_diff,
                            parallel_io=parallel_io,
                            description="`git diff` of moment_kinetics when this run was performed")

        # Get information on all installed packages
        dependencies = string(Pkg.dependencies())
        write_single_value!(provenance_tracking, "dependencies", dependencies,
                            parallel_io=parallel_io,
                            description="Information about all dependency packages (output of `Pkg.dependencies()`)")

        if previous_runs_info !== nothing
            for (i, info) ∈ enumerate(previous_runs_info)
                section = create_io_group(provenance_tracking, "previous_run_$i")
                write_Dict_to_section(section, info, parallel_io)
            end
            previous_run_ids = [""]
            n_previous_runs = 1
        end
    end
    return nothing
end

"""
    write_Dict_to_section(section_io, section_dict, parallel_io)

Write the contents of `section_dict` into the I/O group `section_io`.

Any nested Dicts in `section_dict` are written to subsections.

All the keys in `section_dict` (and any nested Dicts) should be Strings.

`parallel_io` is a Bool indicating whether parallel I/O is being used.
"""
function write_Dict_to_section(section_io, section_dict, parallel_io)
    for (key, value) ∈ section_dict
        if isa(value, AbstractDict)
            subsection_io = create_io_group(section_io, key)
            write_Dict_to_section(subsection_io, value, parallel_io)
        else
            write_single_value!(section_io, key, value, parallel_io=parallel_io)
        end
    end
end

"""
Save info from the dict with input settings to the output file

Note: assumes all keys in `input_dict` are strings.
"""
function write_input!(fid, input_dict, parallel_io)
    @serial_region begin
        input_io = create_io_group(fid, "input")
        write_Dict_to_section(input_io, input_dict, parallel_io)
    end
end

"""
Define coords group for coordinate information in the output file and write information
about spatial and velocity space coordinate grids
"""
function define_io_coordinates!(fid, vz, vr, vzeta, vpa, vperp, z, r, parallel_io)
    @serial_region begin
        # create the "coords" group that will contain coordinate information
        coords = create_io_group(fid, "coords")
        if z !== nothing
            # create the "z" sub-group of "coords" that will contain z coordinate info,
            # including total number of grid points and grid point locations
            define_io_coordinate!(coords, z, "z", "spatial coordinate z", parallel_io)
        end
        if r !== nothing
            # create the "r" sub-group of "coords" that will contain r coordinate info,
            # including total number of grid points and grid point locations
            define_io_coordinate!(coords, r, "r", "spatial coordinate r", parallel_io)
        end

        if parallel_io
            # Parallel I/O produces a single file, so effectively a 'single block'

            # Write variable recording the index of the block within the global domain
            # decomposition
            write_single_value!(coords, "iblock", 0, parallel_io=parallel_io,
                                description="index of this zr block")

            # Write variable recording the total number of blocks in the global domain
            # decomposition
            write_single_value!(coords, "nblocks", 1, parallel_io=parallel_io,
                                description="number of zr blocks")
        else
            # Write a separate file for each block

            # Write variable recording the index of the block within the global domain
            # decomposition
            write_single_value!(coords, "iblock", iblock_index[], parallel_io=parallel_io,
                                description="index of this zr block")

            # Write variable recording the total number of blocks in the global domain
            # decomposition
            write_single_value!(coords, "nblocks", global_size[]÷block_size[],
                                parallel_io=parallel_io, description="number of zr blocks")
        end

        if vz !== nothing
            # create the "vz" sub-group of "coords" that will contain vz coordinate info,
            # including total number of grid points and grid point locations
            define_io_coordinate!(coords, vz, "vz", "velocity coordinate v_z", parallel_io)
        end
        if vr !== nothing
            # create the "vr" sub-group of "coords" that will contain vr coordinate info,
            # including total number of grid points and grid point locations
            define_io_coordinate!(coords, vr, "vr", "velocity coordinate v_r", parallel_io)
        end
        if vzeta !== nothing
            # create the "vzeta" sub-group of "coords" that will contain vzeta coordinate info,
            # including total number of grid points and grid point locations
            define_io_coordinate!(coords, vzeta, "vzeta", "velocity coordinate v_zeta",
                                  parallel_io)
        end
        if vpa !== nothing
            # create the "vpa" sub-group of "coords" that will contain vpa coordinate info,
            # including total number of grid points and grid point locations
            define_io_coordinate!(coords, vpa, "vpa", "velocity coordinate v_parallel",
                                  parallel_io)
        end
        if vperp !== nothing
            # create the "vperp" sub-group of "coords" that will contain vperp coordinate info,
            # including total number of grid points and grid point locations
            define_io_coordinate!(coords, vperp, "vperp", "velocity coordinate v_perp",
                                  parallel_io)
        end
    end

    return nothing
end

"""
define a sub-group for each code coordinate and write to output file
"""
function define_io_coordinate!(parent, coord, coord_name, description, parallel_io)
    @serial_region begin
        # create the "group" sub-group of "parent" that will contain coord_str coordinate info
        group = create_io_group(parent, coord_name, description=description)

        if parallel_io
            # When using parallel I/O, write n_global as n_local because the file is as if
            # it had been produced by a serial run.
            # This is a bit of a hack and should probably be removed when
            # post_processing.jl is updated to be compatible with that.
            write_single_value!(group, "n_local", coord.n_global; parallel_io=parallel_io,
                                description="number of local $coord_name grid points")
        else
            # write the number of local grid points for this coordinate to variable
            # "n_local" within "coords/coord_name" group
            write_single_value!(group, "n_local", coord.n; parallel_io=parallel_io,
                                description="number of local $coord_name grid points")
        end

        # write the number of points within each element for this coordinate to variable
        # "ngrid" within "coords/coord_name" group
        write_single_value!(group, "ngrid", coord.ngrid; parallel_io=parallel_io,
                            description="number of points in each element in $coord_name")

        # write the number of global grid points for this coordinate to variable "n_local"
        # within "coords/coord_name" group
        write_single_value!(group, "n_global", coord.n_global; parallel_io=parallel_io,
                            description="total number of $coord_name grid points")

        if parallel_io
            # write the rank as if whole file was written by rank-0
            write_single_value!(group, "irank", 0, parallel_io=parallel_io,
                                description="rank of this block in the $(coord.name) grid communicator")
            write_single_value!(group, "nrank", 1, parallel_io=parallel_io,
                                description="number of ranks in the $(coord.name) grid communicator")
        else
            # write the rank in the coord-direction of this process
            write_single_value!(group, "irank", coord.irank, parallel_io=parallel_io,
                                description="rank of this block in the $(coord.name) grid communicator")
            write_single_value!(group, "nrank", coord.nrank, parallel_io=parallel_io,
                                description="number of ranks in the $(coord.name) grid communicator")
        end
        # Record the local size of the coordinate, as this will be the chunk size used by
        # parallel I/O (see hdf5_get_fixed_dim_sizes() in file_io_hdf5.jl).
        if coord.nrank == 1
            write_single_value!(group, "chunk_size", coord.n, parallel_io=parallel_io,
                                description="chunk size of blocks in the $(coord.name) grid communicator")
        else
            write_single_value!(group, "chunk_size", coord.n - 1, parallel_io=parallel_io,
                                description="chunk size of blocks in the $(coord.name) grid communicator")
        end

        # write the global length of this coordinate to variable "L"
        # within "coords/coord_name" group
        write_single_value!(group, "L", coord.L; parallel_io=parallel_io,
                            description="box length in $coord_name")

        # write the locations of this coordinate's grid points to variable "grid" within "coords/coord_name" group
        write_single_value!(group, "grid", coord.grid, coord; parallel_io=parallel_io,
                            description="$coord_name values sampled by the $coord_name grid")

        # write the integration weights attached to each coordinate grid point
        write_single_value!(group, "wgts", coord.wgts, coord; parallel_io=parallel_io,
                            description="integration weights associated with the $coord_name grid points")

        # write the discretization option for the coordinate
        write_single_value!(group, "discretization", coord.discretization;
                            parallel_io=parallel_io,
                            description="discretization used for $coord_name")

        # write the finite-difference option for the coordinate
        write_single_value!(group, "finite_difference_option",
                            coord.finite_difference_option; parallel_io=parallel_io,
                            description="type of finite difference for $coord_name, if used")

        write_single_value!(group, "cheb_option", coord.cheb_option; parallel_io=parallel_io,
                            description="type of chebyshev differentiation used for $coord_name, if used")

        # write the boundary condition for the coordinate
        write_single_value!(group, "bc", coord.bc; parallel_io=parallel_io,
                            description="boundary condition for $coord_name")

        # write the element spacing option for the coordinate
        write_single_value!(group, "element_spacing_option", coord.element_spacing_option; parallel_io=parallel_io,
                            description="element_spacing_option for $coord_name")

        return group
    end

    # For processes other than the root process of each shared-memory group...
    return nothing
end

"""
    create_dynamic_variable!(file_or_group, name, type,
                             coords::Union{coordinate,NamedTuple}...; parallel_io,
                             description=nothing, units=nothing)

Create a time-evolving variable in `file_or_group` named `name` of type `type`.

`coords` are the coordinates corresponding to the dimensions of the array, in the order of
the array dimensions - they may be either `coordinate` structs or `NamedTuple`s that
contain at least the fields `name`, `n`.

A description and/or units can be added with the keyword arguments.

`parallel_io` is a Bool specifying whether parallel I/O is being used.
"""
function create_dynamic_variable! end

"""
define dynamic (time-evolving) moment variables for writing to the hdf5 file
"""
function define_dynamic_moment_variables!(fid, n_ion_species, n_neutral_species,
                                          r::coordinate, z::coordinate, io_input,
                                          external_source_settings, evolve_density,
                                          evolve_upar, evolve_p, electron_physics,
                                          t_params, nl_solver_params)
    @serial_region begin
        parallel_io = io_input.parallel_io
        dynamic = create_io_group(fid, "dynamic_data", description="time evolving variables")
        timing = create_io_group(fid, "timing_data",
                                 description="timing data to check run-time performance")

        io_time = create_dynamic_variable!(dynamic, "time", mk_float; parallel_io=parallel_io,
                                           description="simulation time")

        io_phi, io_Er, io_Ez =
            define_dynamic_em_field_variables!(fid, r, z, parallel_io)

        io_density, io_density_loworder, io_density_start_last_timestep, io_upar,
        io_upar_loworder, io_upar_start_last_timestep, io_p, io_p_loworder,
        io_p_start_last_timestep, io_ppar, io_pperp, io_qpar, io_vth, io_dSdt,
        external_source_amplitude, external_source_T_array,
        external_source_density_amplitude, external_source_momentum_amplitude,
        external_source_pressure_amplitude, external_source_controller_integral,
        io_chodura_lower, io_chodura_upper, ion_constraints_A_coefficient,
        ion_constraints_B_coefficient, ion_constraints_C_coefficient =
            define_dynamic_ion_moment_variables!(fid, n_ion_species, r, z, parallel_io,
                                                 external_source_settings, evolve_density,
                                                 evolve_upar, evolve_p,
                                                 io_input.write_error_diagnostics,
                                                 io_input.write_steady_state_diagnostics)

        io_electron_density, io_electron_density_loworder,
        io_electron_density_start_last_timestep, io_electron_upar,
        io_electron_upar_loworder, io_electron_upar_start_last_timestep, io_electron_p,
        io_electron_p_loworder, io_electron_p_start_last_timestep, io_electron_ppar,
        io_electron_qpar, io_electron_vth, external_source_electron_amplitude,
        external_source_electron_T_array, external_source_electron_density_amplitude,
        external_source_electron_momentum_amplitude,
        external_source_electron_pressure_amplitude,
        electron_constraints_A_coefficient, electron_constraints_B_coefficient,
        electron_constraints_C_coefficient, io_electron_step_counter,
        io_electron_cumulative_pseudotime, io_electron_dt, io_electron_previous_dt,
        io_electron_failure_counter, io_electron_dt_before_last_fail =
            define_dynamic_electron_moment_variables!(fid, r, z, parallel_io,
                                                      external_source_settings,
                                                      evolve_density, evolve_upar,
                                                      evolve_p, electron_physics,
                                                      t_params.electron,
                                                      io_input.write_error_diagnostics,
                                                      io_input.write_steady_state_diagnostics)

        io_density_neutral, io_density_neutral_loworder,
        io_density_neutral_start_last_timestep, io_uz_neutral, io_uz_neutral_loworder,
        io_uz_neutral_start_last_timestep, io_p_neutral, io_p_neutral_loworder,
        io_p_neutral_start_last_timestep, io_pz_neutral, io_qz_neutral,
        io_thermal_speed_neutral, external_source_neutral_amplitude,
        external_source_neutral_T_array, external_source_neutral_density_amplitude,
        external_source_neutral_momentum_amplitude,
        external_source_neutral_pressure_amplitude,
        external_source_neutral_controller_integral, neutral_constraints_A_coefficient,
        neutral_constraints_B_coefficient, neutral_constraints_C_coefficient =
            define_dynamic_neutral_moment_variables!(fid, n_neutral_species, r, z,
                                                     parallel_io,
                                                     external_source_settings,
                                                     evolve_density, evolve_upar,
                                                     evolve_p,
                                                     io_input.write_error_diagnostics,
                                                     io_input.write_steady_state_diagnostics)

        io_time_for_run = create_dynamic_variable!(
            timing, "time_for_run", mk_float; parallel_io=parallel_io,
            description="cumulative wall clock time for run (excluding setup)",
            units="minutes")

        io_step_counter = create_dynamic_variable!(
            dynamic, "step_counter", mk_int; parallel_io=parallel_io,
            description="cumulative number of timesteps for the run")

        io_dt = create_dynamic_variable!(
            dynamic, "dt", mk_float; parallel_io=parallel_io,
            description="current timestep size")

        io_previous_dt = create_dynamic_variable!(
            dynamic, "previous_dt", mk_float; parallel_io=parallel_io,
            description="size of the last timestep before the output")

        io_failure_counter = create_dynamic_variable!(
            dynamic, "failure_counter", mk_int; parallel_io=parallel_io,
            description="cumulative number of timestep failures for the run")

        dynamic_keys = collect(keys(dynamic))
        for failure_var ∈ keys(t_params.failure_caused_by)
            create_dynamic_variable!(
                dynamic, "failure_caused_by_$failure_var", mk_int;
                parallel_io=parallel_io,
                description="cumulative count of how many times $failure_var caused "
                            * "a timestep failure for the run")
        end
        for limit_var ∈ keys(t_params.limit_caused_by)
            create_dynamic_variable!(
                dynamic, "limit_caused_by_$limit_var", mk_int;
                parallel_io=parallel_io,
                description="cumulative count of how many times $limit_var limited "
                            * "the timestep for the run")
        end

        io_dt_before_last_fail = create_dynamic_variable!(
            dynamic, "dt_before_last_fail", mk_float; parallel_io=parallel_io,
            description="Last successful timestep before most recent timestep failure, "
                        * "used by adaptve timestepping algorithm")

        io_nl_solver_diagnostics = NamedTuple(
            term=>(n_solves=create_dynamic_variable!(
                                dynamic, "$(term)_n_solves", mk_int; parallel_io=parallel_io,
                                description="Number of nonlinear solves for $term"),
                   nonlinear_iterations=create_dynamic_variable!(
                                            dynamic, "$(term)_nonlinear_iterations", mk_int;
                                            parallel_io=parallel_io,
                                            description="Number of nonlinear iterations for $term"),
                   linear_iterations=create_dynamic_variable!(
                                         dynamic, "$(term)_linear_iterations", mk_int;
                                         parallel_io=parallel_io,
                                         description="Number of linear iterations for $term"),
                   precon_iterations=create_dynamic_variable!(
                                         dynamic, "$(term)_precon_iterations", mk_int;
                                         parallel_io=parallel_io,
                                         description="Number of preconditioner iterations for $term"),
                  )
            for (term, params) ∈ pairs(nl_solver_params) if params !== nothing)

        return io_moments_info(fid, io_time, io_phi, io_Er, io_Ez, io_density,
                               io_density_loworder, io_density_start_last_timestep,
                               io_upar, io_upar_loworder, io_upar_start_last_timestep,
                               io_p, io_p_loworder, io_p_start_last_timestep, io_ppar,
                               io_pperp, io_qpar, io_vth, io_dSdt, io_chodura_lower,
                               io_chodura_upper, io_electron_density,
                               io_electron_density_loworder,
                               io_electron_density_start_last_timestep, io_electron_upar,
                               io_electron_upar_loworder,
                               io_electron_upar_start_last_timestep, io_electron_p,
                               io_electron_p_loworder, io_electron_p_start_last_timestep,
                               io_electron_ppar, io_electron_qpar, io_electron_vth,
                               io_density_neutral, io_density_neutral_loworder,
                               io_density_neutral_start_last_timestep, io_uz_neutral,
                               io_uz_neutral_loworder, io_uz_neutral_start_last_timestep,
                               io_p_neutral, io_p_neutral_loworder,
                               io_p_neutral_start_last_timestep, io_pz_neutral,
                               io_qz_neutral, io_thermal_speed_neutral,
                               external_source_amplitude,
                               external_source_T_array, external_source_density_amplitude,
                               external_source_momentum_amplitude,
                               external_source_pressure_amplitude,
                               external_source_controller_integral,
                               external_source_neutral_amplitude,
                               external_source_neutral_T_array,
                               external_source_neutral_density_amplitude,
                               external_source_neutral_momentum_amplitude,
                               external_source_neutral_pressure_amplitude,
                               external_source_neutral_controller_integral,
                               external_source_electron_amplitude,
                               external_source_electron_T_array,
                               external_source_electron_density_amplitude,
                               external_source_electron_momentum_amplitude,
                               external_source_electron_pressure_amplitude,
                               ion_constraints_A_coefficient,
                               ion_constraints_B_coefficient,
                               ion_constraints_C_coefficient,
                               neutral_constraints_A_coefficient,
                               neutral_constraints_B_coefficient,
                               neutral_constraints_C_coefficient,
                               electron_constraints_A_coefficient,
                               electron_constraints_B_coefficient,
                               electron_constraints_C_coefficient,
                               io_time_for_run, io_step_counter, io_dt, io_previous_dt,
                               io_failure_counter, io_dt_before_last_fail,
                               io_electron_step_counter,
                               io_electron_cumulative_pseudotime, io_electron_dt,
                               io_electron_previous_dt, io_electron_failure_counter,
                               io_electron_dt_before_last_fail, io_nl_solver_diagnostics,
                               io_input)
    end

    # For processes other than the root process of each shared-memory group...
    return nothing
end

"""
define dynamic (time-evolving) electromagnetic field variables for writing to the hdf5
file
"""
function define_dynamic_em_field_variables!(fid, r::coordinate, z::coordinate,
                                            parallel_io)

    dynamic = get_group(fid, "dynamic_data")

    # io_phi is the handle referring to the electrostatic potential phi
    io_phi = create_dynamic_variable!(dynamic, "phi", mk_float, z, r;
                                      parallel_io=parallel_io,
                                      description="electrostatic potential",
                                      units="T_ref/e")
    # io_Er is the handle for the radial component of the electric field
    io_Er = create_dynamic_variable!(dynamic, "Er", mk_float, z, r;
                                     parallel_io=parallel_io,
                                     description="radial electric field",
                                     units="T_ref/e L_ref")
    # io_Ez is the handle for the zed component of the electric field
    io_Ez = create_dynamic_variable!(dynamic, "Ez", mk_float, z, r;
                                     parallel_io=parallel_io,
                                     description="vertical electric field",
                                     units="T_ref/e L_ref")

    return io_phi, io_Er, io_Ez
end

"""
define dynamic (time-evolving) ion moment variables for writing to the hdf5 file
"""
function define_dynamic_ion_moment_variables!(fid, n_ion_species, r::coordinate,
        z::coordinate, parallel_io, external_source_settings, evolve_density, evolve_upar,
        evolve_p, write_error_diagnostics, write_steady_state_diagnostics)

    dynamic = get_group(fid, "dynamic_data")
    ion_species_coord = (name="ion_species", n=n_ion_species)

    # io_density is the handle for the ion particle density
    io_density = create_dynamic_variable!(dynamic, "density", mk_float, z, r,
                                          ion_species_coord; parallel_io=parallel_io,
                                          description="ion species density",
                                          units="n_ref")
    if write_error_diagnostics
        io_density_loworder =
            create_dynamic_variable!(dynamic, "density_loworder", mk_float, z, r,
                                     ion_species_coord; parallel_io=parallel_io,
                                     description="low-order approximation to ion species density, used to diagnose timestepping error",
                                     units="n_ref")
    else
        io_density_loworder = nothing
    end
    if write_steady_state_diagnostics
        io_density_start_last_timestep =
            create_dynamic_variable!(dynamic, "density_start_last_timestep", mk_float, z, r,
                                     ion_species_coord; parallel_io=parallel_io,
                                     description="ion species density at the start of the last timestep before output, used to measure steady state residual",
                                     units="n_ref")
    else
        io_density_start_last_timestep = nothing
    end

    # io_upar is the handle for the ion parallel flow density
    io_upar = create_dynamic_variable!(dynamic, "parallel_flow", mk_float, z, r,
                                       ion_species_coord; parallel_io=parallel_io,
                                       description="ion species parallel flow",
                                       units="c_ref = sqrt(T_ref/m_ref)")
    if write_error_diagnostics
        io_upar_loworder =
            create_dynamic_variable!(dynamic, "parallel_flow_loworder", mk_float, z, r,
                                     ion_species_coord; parallel_io=parallel_io,
                                     description="low-order approximation to ion species parallel flow, used to diagnose timestepping error",
                                     units="c_ref = sqrt(T_ref/m_ref)")
    else
        io_upar_loworder = nothing
    end
    if write_steady_state_diagnostics
        io_upar_start_last_timestep =
            create_dynamic_variable!(dynamic, "parallel_flow_start_last_timestep",
                                     mk_float, z, r, ion_species_coord;
                                     parallel_io=parallel_io,
                                     description="ion species parallel flow at the start of the last timestep before output, used to measure steady state residual",
                                     units="c_ref = sqrt(T_ref/m_ref)")
    else
        io_upar_start_last_timestep = nothing
    end

    # io_p is the handle for the ion pressure
    io_p = create_dynamic_variable!(dynamic, "pressure", mk_float, z, r,
                                    ion_species_coord; parallel_io=parallel_io,
                                    description="ion species pressure",
                                    units="n_ref*T_ref")
    if write_error_diagnostics
        io_p_loworder =
            create_dynamic_variable!(dynamic, "pressure_loworder", mk_float, z, r,
                                     ion_species_coord; parallel_io=parallel_io,
                                     description="low-order approximation to ion species pressure, used to diagnose timestepping error",
                                     units="n_ref*T_ref")
    else
        io_p_loworder = nothing
    end
    if write_steady_state_diagnostics
        io_p_start_last_timestep =
            create_dynamic_variable!(dynamic, "pressure_start_last_timestep",
                                     mk_float, z, r, ion_species_coord;
                                     parallel_io=parallel_io,
                                     description="ion species pressure at the start of the last timestep before output, used to measure steady state residual",
                                     units="n_ref*T_ref")
    else
        io_p_start_last_timestep = nothing
    end

    # io_ppar is the handle for the ion parallel pressure
    io_ppar = create_dynamic_variable!(dynamic, "parallel_pressure", mk_float, z, r,
                                       ion_species_coord; parallel_io=parallel_io,
                                       description="ion species parallel pressure",
                                       units="n_ref*T_ref")

    # io_pperp is the handle for the ion parallel pressure
    io_pperp = create_dynamic_variable!(dynamic, "perpendicular_pressure", mk_float, z, r,
                                        ion_species_coord; parallel_io=parallel_io,
                                        description="ion species perpendicular pressure",
                                        units="n_ref*T_ref")

    # io_qpar is the handle for the ion parallel heat flux
    io_qpar = create_dynamic_variable!(dynamic, "parallel_heat_flux", mk_float, z, r,
                                       ion_species_coord; parallel_io=parallel_io,
                                       description="ion species parallel heat flux",
                                       units="n_ref*T_ref*c_ref")

    # io_vth is the handle for the ion thermal speed
    io_vth = create_dynamic_variable!(dynamic, "thermal_speed", mk_float, z, r,
                                      ion_species_coord; parallel_io=parallel_io,
                                      description="ion species thermal speed",
                                      units="c_ref")

    # io_dSdt is the handle for the entropy production (due to collisions)
    io_dSdt = create_dynamic_variable!(dynamic, "entropy_production", mk_float, z, r,
                                      ion_species_coord; parallel_io=parallel_io,
                                      description="ion species entropy production",
                                      units="")

    ion_source_settings = external_source_settings.ion
    if any(x -> x.active, ion_source_settings)
        n_sources = (name="n_ion_sources", n=length(ion_source_settings))
        external_source_amplitude = create_dynamic_variable!(
            dynamic, "external_source_amplitude", mk_float, z, r, n_sources;
            parallel_io=parallel_io, description="Amplitude of the external source for ions",
            units="n_ref/c_ref^3*c_ref/L_ref")
        external_source_T_array = create_dynamic_variable!(
            dynamic, "external_source_T_array", mk_float, z, r, n_sources;
            parallel_io=parallel_io, description="Temperature of the external source for ions",
            units="T_ref")
        if evolve_density
            external_source_density_amplitude = create_dynamic_variable!(
                dynamic, "external_source_density_amplitude", mk_float, z, r, n_sources;
                parallel_io=parallel_io, description="Amplitude of the external density source for ions",
                units="n_ref*c_ref/L_ref")
        else
            external_source_density_amplitude = nothing
        end
        if evolve_upar
            external_source_momentum_amplitude = create_dynamic_variable!(
                dynamic, "external_source_momentum_amplitude", mk_float, z, r, n_sources;
                parallel_io=parallel_io, description="Amplitude of the external momentum source for ions",
                units="m_ref*n_ref*c_ref*c_ref/L_ref")
        else
            external_source_momentum_amplitude = nothing
        end
        if evolve_p
            external_source_pressure_amplitude = create_dynamic_variable!(
                dynamic, "external_source_pressure_amplitude", mk_float, z, r, n_sources;
                parallel_io=parallel_io, description="Amplitude of the external pressure source for ions",
                units="n_ref*T_ref*c_ref/L_ref")
        else
            external_source_pressure_amplitude = nothing
        end
        if any(x -> x.PI_density_controller_I != 0.0 && x.source_type ∈ 
                    ("density_profile_control", "density_midpoint_control"), ion_source_settings)
            if any(x -> x.source_type == "density_profile_control", ion_source_settings)
                external_source_controller_integral = create_dynamic_variable!(
                    dynamic, "external_source_controller_integral", mk_float, z, r, n_sources;
                    parallel_io=parallel_io,
                    description="Integral term for the PID controller of the external source for ions")
            else
                r_midpoint = (name="midpoint_controller_r", n=1)
                z_midpoint = (name="midpoint_controller_z", n=1)
                external_source_controller_integral = create_dynamic_variable!(
                    dynamic, "external_source_controller_integral", mk_float, r_midpoint, z_midpoint, n_sources;
                    parallel_io=parallel_io,
                    description="Integral term for the PID controller of the external source for ions")
            end
        else
            external_source_controller_integral = nothing
        end
    else
        external_source_amplitude = nothing
        external_source_T_array = nothing
        external_source_density_amplitude = nothing
        external_source_momentum_amplitude = nothing
        external_source_pressure_amplitude = nothing
        external_source_controller_integral = nothing
    end

    if parallel_io || z.irank == 0
        # io_chodura_lower is the handle for the ion thermal speed
        io_chodura_lower = create_dynamic_variable!(dynamic, "chodura_integral_lower", mk_float, r,
                                                    ion_species_coord;
                                                    parallel_io=parallel_io,
                                                    description="Generalised Chodura integral lower sheath entrance",
                                                    units="c_ref")
    else
        io_chodura_lower = nothing
    end
    if parallel_io || z.irank == z.nrank - 1
        # io_chodura_upper is the handle for the ion thermal speed
        io_chodura_upper = create_dynamic_variable!(dynamic, "chodura_integral_upper", mk_float, r,
                                                    ion_species_coord;
                                                    parallel_io=parallel_io,
                                                    description="Generalised Chodura integral upper sheath entrance",
                                                    units="c_ref")
    else
        io_chodura_upper = nothing
    end

    if evolve_density || evolve_upar || evolve_p
        ion_constraints_A_coefficient =
            create_dynamic_variable!(dynamic, "ion_constraints_A_coefficient", mk_float,
                                     z, r, ion_species_coord; parallel_io=parallel_io,
                                     description="'A' coefficient enforcing density constraint for ions")
        ion_constraints_B_coefficient =
            create_dynamic_variable!(dynamic, "ion_constraints_B_coefficient", mk_float,
                                     z, r, ion_species_coord; parallel_io=parallel_io,
                                     description="'B' coefficient enforcing flow constraint for ions")
        ion_constraints_C_coefficient =
            create_dynamic_variable!(dynamic, "ion_constraints_C_coefficient", mk_float,
                                     z, r, ion_species_coord; parallel_io=parallel_io,
                                     description="'C' coefficient enforcing pressure constraint for ions")
    else
           ion_constraints_A_coefficient = nothing
           ion_constraints_B_coefficient = nothing
           ion_constraints_C_coefficient = nothing
    end

    return io_density, io_density_loworder, io_density_start_last_timestep, io_upar,
           io_upar_loworder, io_upar_start_last_timestep, io_p, io_p_loworder,
           io_p_start_last_timestep, io_ppar, io_pperp, io_qpar, io_vth, io_dSdt,
           external_source_amplitude, external_source_T_array, external_source_density_amplitude,
           external_source_momentum_amplitude, external_source_pressure_amplitude,
           external_source_controller_integral, io_chodura_lower, io_chodura_upper,
           ion_constraints_A_coefficient, ion_constraints_B_coefficient,
           ion_constraints_C_coefficient
end

"""
define dynamic (time-evolving) electron moment variables for writing to the hdf5 file
"""
function define_dynamic_electron_moment_variables!(fid, r::coordinate, z::coordinate,
        parallel_io, external_source_settings, evolve_density, evolve_upar, evolve_p,
        electron_physics, t_params, write_error_diagnostics,
        write_steady_state_diagnostics, ir=nothing; electron_only_io=false)

    dynamic = get_group(fid, "dynamic_data")

    if ir === nothing
        io_r = r
    else
        io_r = (n=1, n_global=1, ngrid=1, name="r", irank=0, nrank=1, L=r.L,
                grid=[r.grid[ir]], wgts=[r.wgts[ir]], discretization=r.discretization,
                finite_difference_option=r.finite_difference_option,
                cheb_option=r.cheb_option, bc=r.bc,
                element_spacing_option=r.element_spacing_option)
    end

    if !electron_only_io
        # io_density is the handle for the ion particle density
        io_electron_density = create_dynamic_variable!(dynamic, "electron_density", mk_float, z, io_r;
                                                       parallel_io=parallel_io,
                                                       description="electron species density",
                                                       units="n_ref")
        if write_error_diagnostics
            io_electron_density_loworder =
                create_dynamic_variable!(dynamic, "electron_density_loworder", mk_float, z, io_r;
                                         parallel_io=parallel_io,
                                         description="low-order approximation to electron species density, used to diagnose timestepping error",
                                         units="n_ref")
        else
            io_electron_density_loworder = nothing
        end
        if write_steady_state_diagnostics
            io_electron_density_start_last_timestep =
                create_dynamic_variable!(dynamic, "electron_density_start_last_timestep",
                                         mk_float, z, io_r; parallel_io=parallel_io,
                                         description="electron species density at the start of the last timestep before output, used to measure steady state residual",
                                         units="n_ref")
        else
            io_electron_density_start_last_timestep = nothing
        end

        # io_electron_upar is the handle for the electron parallel flow density
        io_electron_upar = create_dynamic_variable!(dynamic, "electron_parallel_flow", mk_float, z, io_r;
                                                    parallel_io=parallel_io,
                                                    description="electron species parallel flow",
                                                    units="c_ref = sqrt(T_ref/mi)")
        if write_error_diagnostics
            io_electron_upar_loworder =
                create_dynamic_variable!(dynamic, "electron_parallel_flow_loworder", mk_float, z,
                                         io_r; parallel_io=parallel_io,
                                         description="low-order approximation to electron species parallel flow, used to diagnose timestepping error",
                                         units="c_ref = sqrt(T_ref/mi)")
        else
            io_electron_upar_loworder = nothing
        end
        if write_steady_state_diagnostics
            io_electron_upar_start_last_timestep =
                create_dynamic_variable!(dynamic, "electron_parallel_flow_start_last_timestep",
                                         mk_float, z, io_r; parallel_io=parallel_io,
                                         description="electron species parallel flow at the start of the last timestep before output, used to measure steady state residual",
                                         units="c_ref = sqrt(T_ref/mi)")
        else
            io_electron_upar_start_last_timestep = nothing
        end
    else
        io_electron_density = nothing
        io_electron_density_loworder = nothing
        io_electron_density_start_last_timestep = nothing
        io_electron_upar = nothing
        io_electron_upar_loworder = nothing
        io_electron_upar_start_last_timestep = nothing
    end

    # io_electron_ppar is the handle for the electron parallel pressure
    io_electron_p = create_dynamic_variable!(dynamic, "electron_pressure", mk_float, z, io_r;
                                             parallel_io=parallel_io,
                                             description="electron species pressure",
                                             units="n_ref*T_ref")
    if write_error_diagnostics
        io_electron_p_loworder =
            create_dynamic_variable!(dynamic, "electron_pressure_loworder", mk_float,
                                     z, io_r; parallel_io=parallel_io,
                                     description="low-order approximation to electron species pressure, used to diagnose timestepping error",
                                     units="n_ref*T_ref")
    else
        io_electron_p_loworder = nothing
    end
    if write_steady_state_diagnostics
        io_electron_p_start_last_timestep =
            create_dynamic_variable!(dynamic,
                                     "electron_pressure_start_last_timestep",
                                     mk_float, z, io_r; parallel_io=parallel_io,
                                     description="electron species pressure at the start of the last timestep before output, used to measure steady state residual",
                                     units="n_ref*T_ref")
    else
        io_electron_p_start_last_timestep = nothing
    end

    io_electron_ppar = create_dynamic_variable!(dynamic, "electron_parallel_pressure", mk_float, z, io_r;
                                       parallel_io=parallel_io,
                                       description="electron species parallel pressure",
                                       units="n_ref*T_ref")

    # io_electron_qpar is the handle for the electron parallel heat flux
    io_electron_qpar = create_dynamic_variable!(dynamic, "electron_parallel_heat_flux", mk_float, z, io_r;
                                                parallel_io=parallel_io,
                                                description="electron species parallel heat flux",
                                                units="n_ref*T_ref*c_ref")

    # io_electron_vth is the handle for the electron thermal speed
    io_electron_vth = create_dynamic_variable!(dynamic, "electron_thermal_speed", mk_float, z, io_r;
                                               parallel_io=parallel_io,
                                               description="electron species thermal speed",
                                               units="c_ref")

    electron_source_settings = external_source_settings.electron
    if any(x -> x.active, electron_source_settings)
        n_sources = (name="n_electron_sources", n=length(electron_source_settings))
        external_source_electron_amplitude = create_dynamic_variable!(
            dynamic, "external_source_electron_amplitude", mk_float, z, io_r, n_sources;
            parallel_io=parallel_io, description="Amplitude of the external source for electrons",
            units="n_ref/c_ref^3*c_ref/L_ref")
        external_source_electron_T_array = create_dynamic_variable!(
            dynamic, "external_source_electron_T_array", mk_float, z, io_r, n_sources;
            parallel_io=parallel_io, description="Temperature of the external source for electrons",
            units="T_ref")
        external_source_electron_density_amplitude = create_dynamic_variable!(
            dynamic, "external_source_electron_density_amplitude", mk_float, z, io_r, n_sources;
            parallel_io=parallel_io, description="Amplitude of the external density source for electrons",
            units="n_ref*c_ref/L_ref")
        external_source_electron_momentum_amplitude = create_dynamic_variable!(
            dynamic, "external_source_electron_momentum_amplitude", mk_float, z, io_r, n_sources;
            parallel_io=parallel_io, description="Amplitude of the external momentum source for electrons",
            units="m_ref*n_ref*c_ref*c_ref/L_ref")
        external_source_electron_pressure_amplitude = create_dynamic_variable!(
            dynamic, "external_source_electron_pressure_amplitude", mk_float, z, io_r, n_sources;
            parallel_io=parallel_io, description="Amplitude of the external pressure source for electrons",
            units="n_ref*T_ref*c_ref/L_ref")
    else
        external_source_electron_amplitude = nothing
        external_source_electron_T_array = nothing
        external_source_electron_density_amplitude = nothing
        external_source_electron_momentum_amplitude = nothing
        external_source_electron_pressure_amplitude = nothing
    end

    electron_constraints_A_coefficient =
        create_dynamic_variable!(dynamic, "electron_constraints_A_coefficient", mk_float, z, io_r;
                                 parallel_io=parallel_io,
                                 description="'A' coefficient enforcing density constraint for electrons")
    electron_constraints_B_coefficient =
        create_dynamic_variable!(dynamic, "electron_constraints_B_coefficient", mk_float, z, io_r;
                                 parallel_io=parallel_io,
                                 description="'B' coefficient enforcing flow constraint for electrons")
    electron_constraints_C_coefficient =
        create_dynamic_variable!(dynamic, "electron_constraints_C_coefficient", mk_float, z, io_r;
                                 parallel_io=parallel_io,
                                 description="'C' coefficient enforcing pressure constraint for electrons")

    if electron_physics ∈ (kinetic_electrons, kinetic_electrons_with_temperature_equation)
        io_electron_step_counter = create_dynamic_variable!(
            dynamic, "electron_step_counter", mk_int, io_r; parallel_io=parallel_io,
            description="cumulative number of electron pseudo-timesteps for the run")

        io_electron_cumulative_pseudotime = create_dynamic_variable!(
            dynamic, "electron_cumulative_pseudotime", mk_float, io_r; parallel_io=parallel_io,
            description="cumulative electron pseudo-time")

        io_electron_dt = create_dynamic_variable!(
            dynamic, "electron_dt", mk_float, io_r; parallel_io=parallel_io,
            description="current electron pseudo-timestep size")

        io_electron_previous_dt = create_dynamic_variable!(
            dynamic, "electron_previous_dt", mk_float, io_r; parallel_io=parallel_io,
            description="size of last electron pseudo-timestep before output was written")

        io_electron_failure_counter = create_dynamic_variable!(
            dynamic, "electron_failure_counter", mk_int, io_r; parallel_io=parallel_io,
            description="cumulative number of electron pseudo-timestep failures for the run")

        for failure_var ∈ keys(t_params.failure_caused_by)
            create_dynamic_variable!(
                dynamic, "electron_failure_caused_by_$failure_var", mk_int, io_r;
                parallel_io=parallel_io,
                description="cumulative count of how many times $failure_var caused an "
                            * "electron pseudo-timestep failure for the run")
        end

        for limit_var ∈ keys(t_params.limit_caused_by)
            create_dynamic_variable!(
                dynamic, "electron_limit_caused_by_$limit_var", mk_int, io_r;
                parallel_io=parallel_io,
                description="cumulative count of how many times $limit_var limited the "
                            * "electron pseudo-timestep for the run")
        end

        io_electron_dt_before_last_fail = create_dynamic_variable!(
            dynamic, "electron_dt_before_last_fail", mk_float, io_r; parallel_io=parallel_io,
            description="Last successful electron pseudo-timestep before most recent "
                        * "electron pseudo-timestep failure, used by adaptve "
                        * "timestepping algorithm")
    else
        io_electron_step_counter = nothing
        io_electron_cumulative_pseudotime = nothing
        io_electron_dt = nothing
        io_electron_previous_dt = nothing
        io_electron_failure_counter = nothing
        io_electron_dt_before_last_fail = nothing
    end

    return io_electron_density, io_electron_density_loworder,
           io_electron_density_start_last_timestep, io_electron_upar,
           io_electron_upar_loworder, io_electron_upar_start_last_timestep,
           io_electron_p, io_electron_p_loworder, io_electron_p_start_last_timestep,
           io_electron_ppar, io_electron_qpar, io_electron_vth,
           external_source_electron_amplitude, external_source_electron_T_array,
           external_source_electron_density_amplitude,
           external_source_electron_momentum_amplitude,
           external_source_electron_pressure_amplitude,
           electron_constraints_A_coefficient, electron_constraints_B_coefficient,
           electron_constraints_C_coefficient, io_electron_step_counter,
           io_electron_cumulative_pseudotime, io_electron_dt, io_electron_previous_dt,
           io_electron_failure_counter, io_electron_dt_before_last_fail
end

"""
define dynamic (time-evolving) neutral moment variables for writing to the hdf5 file
"""
function define_dynamic_neutral_moment_variables!(fid, n_neutral_species, r::coordinate,
        z::coordinate, parallel_io, external_source_settings, evolve_density, evolve_upar,
        evolve_p, write_error_diagnostics, write_steady_state_diagnostics)

    dynamic = get_group(fid, "dynamic_data")
    neutral_species_coord = (name="neutral_species", n=n_neutral_species)

    # io_density_neutral is the handle for the neutral particle density
    io_density_neutral = create_dynamic_variable!(dynamic, "density_neutral", mk_float, z,
                                                  r, neutral_species_coord;
                                                  parallel_io=parallel_io,
                                                  description="neutral species density",
                                                  units="n_ref")
    if write_error_diagnostics
        io_density_neutral_loworder =
            create_dynamic_variable!(dynamic, "density_neutral_loworder", mk_float, z, r,
                                     neutral_species_coord; parallel_io=parallel_io,
                                     description="low-order approximation to neutral species density, used to diagnose timestepping error",
                                     units="n_ref")
    else
        io_density_neutral_loworder = nothing
    end
    if write_steady_state_diagnostics
        io_density_neutral_start_last_timestep =
            create_dynamic_variable!(dynamic, "density_neutral_start_last_timestep", mk_float, z, r,
                                     neutral_species_coord; parallel_io=parallel_io,
                                     description="neutral species density at the start of the last timestep before output, used to measure steady state residual",
                                     units="n_ref")
    else
        io_density_neutral_start_last_timestep = nothing
    end

    # io_uz_neutral is the handle for the neutral z momentum density
    io_uz_neutral = create_dynamic_variable!(dynamic, "uz_neutral", mk_float, z, r,
                                             neutral_species_coord;
                                             parallel_io=parallel_io,
                                             description="neutral species mean z velocity",
                                             units="c_ref = sqrt(T_ref/mi)")
    if write_error_diagnostics
        io_uz_neutral_loworder =
            create_dynamic_variable!(dynamic, "uz_neutral_loworder", mk_float, z, r,
                                     neutral_species_coord; parallel_io=parallel_io,
                                     description="low-order approximation to neutral species mean z velocity, used to diagnose timestepping error",
                                     units="c_ref = sqrt(T_ref/mi)")
    else
        io_uz_neutral_loworder = nothing
    end
    if write_steady_state_diagnostics
        io_uz_neutral_start_last_timestep =
            create_dynamic_variable!(dynamic, "uz_neutral_start_last_timestep", mk_float,
                                     z, r, neutral_species_coord;
                                     parallel_io=parallel_io,
                                     description="neutral species mean z velocity at the start of the last timestep before output, used to measure steady state residual",
                                     units="c_ref = sqrt(T_ref/mi)")
    else
        io_uz_neutral_start_last_timestep = nothing
    end

    # io_p_neutral is the handle for the neutral species pressure
    io_p_neutral = create_dynamic_variable!(dynamic, "p_neutral", mk_float, z, r,
                                             neutral_species_coord;
                                             parallel_io=parallel_io,
                                             description="neutral species pressure",
                                             units="n_ref*T_ref")
    if write_error_diagnostics
        io_p_neutral_loworder =
            create_dynamic_variable!(dynamic, "p_neutral_loworder", mk_float, z, r,
                                     neutral_species_coord; parallel_io=parallel_io,
                                     description="low-order approximation to neutral species pressure, used to diagnose timestepping error",
                                     units="n_ref*T_ref")
    else
        io_p_neutral_loworder = nothing
    end
    if write_steady_state_diagnostics
        io_p_neutral_start_last_timestep =
            create_dynamic_variable!(dynamic, "p_neutral_start_last_timestep", mk_float,
                                     z, r, neutral_species_coord;
                                     parallel_io=parallel_io,
                                     description="neutral species pressure at the start of the last timestep before output, used to measure steady state residual",
                                     units="n_ref*T_ref")
    else
        io_p_neutral_start_last_timestep = nothing
    end

    # io_pz_neutral is the handle for the neutral species zz pressure
    io_pz_neutral = create_dynamic_variable!(dynamic, "pz_neutral", mk_float, z, r,
                                             neutral_species_coord;
                                             parallel_io=parallel_io,
                                             description="neutral species mean zz pressure",
                                             units="n_ref*T_ref")

    # io_qz_neutral is the handle for the neutral z heat flux
    io_qz_neutral = create_dynamic_variable!(dynamic, "qz_neutral", mk_float, z, r,
                                             neutral_species_coord;
                                             parallel_io=parallel_io,
                                             description="neutral species z heat flux",
                                             units="n_ref*T_ref*c_ref")

    # io_thermal_speed_neutral is the handle for the neutral thermal speed
    io_thermal_speed_neutral = create_dynamic_variable!(
        dynamic, "thermal_speed_neutral", mk_float, z, r, neutral_species_coord;
        parallel_io=parallel_io, description="neutral species thermal speed",
        units="c_ref")

    neutral_source_settings = external_source_settings.neutral
    if n_neutral_species > 0 && any(x -> x.active, neutral_source_settings)
        n_sources = (name="n_neutral_sources", n=length(neutral_source_settings))
        external_source_neutral_amplitude = create_dynamic_variable!(
            dynamic, "external_source_neutral_amplitude", mk_float, z, r, n_sources;
            parallel_io=parallel_io, description="Amplitude of the external source for neutrals",
            units="n_ref/c_ref^3*c_ref/L_ref")
        external_source_neutral_T_array = create_dynamic_variable!(
            dynamic, "external_source_neutral_T_array", mk_float, z, r, n_sources;
            parallel_io=parallel_io, description="Temperature of the external source for neutrals",
            units="T_ref")
        if evolve_density
            external_source_neutral_density_amplitude = create_dynamic_variable!(
                dynamic, "external_source_neutral_density_amplitude", mk_float, z, r, n_sources;
                parallel_io=parallel_io, description="Amplitude of the external density source for neutrals",
                units="n_ref*c_ref/L_ref")
        else
            external_source_neutral_density_amplitude = nothing
        end
        if evolve_upar
            external_source_neutral_momentum_amplitude = create_dynamic_variable!(
                dynamic, "external_source_neutral_momentum_amplitude", mk_float, z, r, n_sources;
                parallel_io=parallel_io, description="Amplitude of the external momentum source for neutrals",
                units="m_ref*n_ref*c_ref*c_ref/L_ref")
        else
            external_source_neutral_momentum_amplitude = nothing
        end
        if evolve_p
            external_source_neutral_pressure_amplitude = create_dynamic_variable!(
                dynamic, "external_source_neutral_pressure_amplitude", mk_float, z, r, n_sources;
                parallel_io=parallel_io, description="Amplitude of the external pressure source for neutrals",
                units="n_ref*T_ref*c_ref/L_ref")
        else
            external_source_neutral_pressure_amplitude = nothing
        end
        if any(x -> x.PI_density_controller_I != 0.0 && x.source_type ∈ 
                    ("density_profile_control", "density_midpoint_control"), neutral_source_settings)
            if any(x -> x.source_type == "density_profile_control", neutral_source_settings)
                external_source_neutral_controller_integral = create_dynamic_variable!(
                    dynamic, "external_source_neutral_controller_integral", mk_float, z, r, n_sources;
                    parallel_io=parallel_io,
                    description="Integral term for the PID controller of the external source for neutrals")
            else
                external_source_neutral_controller_integral = create_dynamic_variable!(
                    dynamic, "external_source_neutral_controller_integral", mk_float;
                    parallel_io=parallel_io,
                    description="Integral term for the PID controller of the external source for neutrals")
            end
        else
            external_source_neutral_controller_integral = nothing
        end
    else
        external_source_neutral_amplitude = nothing
        external_source_neutral_T_array = nothing
        external_source_neutral_density_amplitude = nothing
        external_source_neutral_momentum_amplitude = nothing
        external_source_neutral_pressure_amplitude = nothing
        external_source_neutral_controller_integral = nothing
    end

    if evolve_density || evolve_upar || evolve_p
        neutral_constraints_A_coefficient =
            create_dynamic_variable!(dynamic, "neutral_constraints_A_coefficient",
                                     mk_float, z, r, neutral_species_coord;
                                     parallel_io=parallel_io,
                                     description="'A' coefficient enforcing density constraint for neutrals")
        neutral_constraints_B_coefficient =
            create_dynamic_variable!(dynamic, "neutral_constraints_B_coefficient",
                                     mk_float, z, r, neutral_species_coord;
                                     parallel_io=parallel_io,
                                     description="'B' coefficient enforcing flow constraint for neutrals")
        neutral_constraints_C_coefficient =
            create_dynamic_variable!(dynamic, "neutral_constraints_C_coefficient",
                                     mk_float, z, r, neutral_species_coord;
                                     parallel_io=parallel_io,
                                     description="'C' coefficient enforcing pressure constraint for neutrals")
    else
           neutral_constraints_A_coefficient = nothing
           neutral_constraints_B_coefficient = nothing
           neutral_constraints_C_coefficient = nothing
    end

    return io_density_neutral, io_density_neutral_loworder,
           io_density_neutral_start_last_timestep, io_uz_neutral, io_uz_neutral_loworder,
           io_uz_neutral_start_last_timestep, io_p_neutral, io_p_neutral_loworder,
           io_p_neutral_start_last_timestep, io_pz_neutral, io_qz_neutral,
           io_thermal_speed_neutral, external_source_neutral_amplitude,
           external_source_neutral_T_array, external_source_neutral_density_amplitude,
           external_source_neutral_momentum_amplitude,
           external_source_neutral_pressure_amplitude,
           external_source_neutral_controller_integral, neutral_constraints_A_coefficient,
           neutral_constraints_B_coefficient, neutral_constraints_C_coefficient
end

"""
define dynamic (time-evolving) distribution function variables for writing to the output
file
"""
function define_dynamic_dfn_variables!(fid, r, z, vperp, vpa, vzeta, vr, vz, composition,
                                       io_input, external_source_settings,
                                       evolve_density, evolve_upar, evolve_p, t_params,
                                       nl_solver_params)

    @serial_region begin
        parallel_io = io_input.parallel_io
        io_moments = define_dynamic_moment_variables!(fid, composition.n_ion_species,
                                                      composition.n_neutral_species, r, z,
                                                      io_input,
                                                      external_source_settings,
                                                      evolve_density, evolve_upar,
                                                      evolve_p,
                                                      composition.electron_physics,
                                                      t_params, nl_solver_params)

        dynamic = get_group(fid, "dynamic_data")
        ion_species_coord = (name="ion_species", n=composition.n_ion_species)

        # io_f is the handle for the ion pdf
        io_f = create_dynamic_variable!(dynamic, "f", mk_float, vpa, vperp, z, r,
                                        ion_species_coord; parallel_io=parallel_io,
                                        description="ion species distribution function")
        if io_input.write_error_diagnostics
            io_f_loworder = create_dynamic_variable!(dynamic, "f_loworder", mk_float, vpa,
                                                     vperp, z, r, ion_species_coord;
                                                     parallel_io=parallel_io,
                                                     description="low-order approximation to ion species distribution function, used to diagnose timestepping error")
        else
            io_f_loworder = nothing
        end
        if io_input.write_steady_state_diagnostics
            io_f_start_last_timestep =
                create_dynamic_variable!(dynamic, "f_start_last_timestep", mk_float, vpa,
                                         vperp, z, r, ion_species_coord;
                                         parallel_io=parallel_io,
                                         description="ion species distribution function at the start of the last timestep before output, used to measure steady state residual")
        else
            io_f_start_last_timestep = nothing
        end

        if composition.electron_physics ∈ (kinetic_electrons,
                                           kinetic_electrons_with_temperature_equation)
            # io_f_electron is the handle for the electron pdf
            io_f_electron = create_dynamic_variable!(dynamic, "f_electron", mk_float, vpa,
                                                     vperp, z, r;
                                                     parallel_io=parallel_io,
                                                     description="electron distribution function")
            if io_input.write_error_diagnostics
                io_f_electron_loworder =
                    create_dynamic_variable!(dynamic, "f_electron_loworder", mk_float,
                                             vpa, vperp, z, r;
                                             parallel_io=parallel_io,
                                             description="low-order approximation to electron distribution function, used to diagnose timestepping error")
            else
                io_f_electron_loworder = nothing
            end
            if io_input.write_steady_state_diagnostics
                io_f_electron_start_last_timestep =
                    create_dynamic_variable!(dynamic, "f_electron_start_last_timestep",
                                             mk_float, vpa, vperp, z, r;
                                             parallel_io=parallel_io,
                                             description="electron distribution function at the start of the last electron pseudo-timestep before output, used to measure steady state residual")
            else
                io_f_electron_start_last_timestep = nothing
            end
        else
            io_f_electron = nothing
            io_f_electron_loworder = nothing
            io_f_electron_start_last_timestep = nothing
        end

        neutral_species_coord = (name="neutral_species", n=composition.n_neutral_species)

        # io_f_neutral is the handle for the neutral pdf
        io_f_neutral = create_dynamic_variable!(dynamic, "f_neutral", mk_float, vz, vr, vzeta, z, r,
                                                neutral_species_coord;
                                                parallel_io=parallel_io,
                                                description="neutral species distribution function")
        if io_input.write_error_diagnostics
            io_f_neutral_loworder =
                create_dynamic_variable!(dynamic, "f_neutral_loworder", mk_float, vz, vr,
                                         vzeta, z, r, neutral_species_coord;
                                         parallel_io=parallel_io,
                                         description="low-order approximation to neutral species distribution function, used to diagnose timestepping error")
        else
            io_f_neutral_loworder = nothing
        end
        if io_input.write_steady_state_diagnostics
            io_f_neutral_start_last_timestep =
                create_dynamic_variable!(dynamic, "f_neutral_start_last_timestep",
                                         mk_float, vz, vr, vzeta, z, r,
                                         neutral_species_coord; parallel_io=parallel_io,
                                         description="neutral species distribution function at the start of the last timestep before output, used to measure steady state residual")
        else
            io_f_neutral_start_last_timestep = nothing
        end

        return io_dfns_info(fid, io_f, io_f_loworder, io_f_start_last_timestep,
                            io_f_electron, io_f_electron_loworder,
                            io_f_electron_start_last_timestep, io_f_neutral,
                            io_f_neutral_loworder, io_f_neutral_start_last_timestep,
                            io_input, io_moments)
    end

    # For processes other than the root process of each shared-memory group...
    return nothing
end

"""
Add an attribute to a file, group or variable
"""
function add_attribute! end

"""
Modify an attribute to a file, group or variable
"""
function modify_attribute! end

"""
Low-level function to open a binary output file

Each implementation (HDF5, NetCDF, etc.) defines a method of this function to open a file
of the corresponding type.
"""
function open_output_file_implementation end

"""
Open an output file, selecting the backend based on io_option
"""
function open_output_file(prefix, io_input, io_comm)
    check_io_implementation(io_input.binary_format)

    return open_output_file_implementation(Val(io_input.binary_format), prefix, io_input,
                                           io_comm)
end

"""
Re-open an existing output file, selecting the backend based on io_option
"""
function reopen_output_file(filename, io_input, io_comm)
    prefix, format_string = splitext(filename)
    if format_string == ".h5"
        check_io_implementation(hdf5)
        return open_output_file_implementation(Val(hdf5), prefix, io_input, io_comm,
                                               "r+")[1]
    elseif format_string == ".cdf"
        check_io_implementation(netcdf)
        return open_output_file_implementation(Val(netcdf), prefix, io_input, io_comm,
                                               "a")[1]
    else
        error("Unsupported I/O format $binary_format")
    end
end

"""
setup file i/o for moment variables
"""
function setup_moments_io(prefix, io_input, vz, vr, vzeta, vpa, vperp, r, z,
                          composition, collisions, evolve_density, evolve_upar,
                          evolve_p, external_source_settings, manufactured_source_list,
                          input_dict, io_comm, restart_time_index, previous_runs_info,
                          time_for_setup, t_params, nl_solver_params)
    @serial_region begin
        moments_prefix = string(prefix, ".moments")
        parallel_io = io_input.parallel_io
        if !parallel_io
            moments_prefix *= ".$(iblock_index[])"
        end
        fid, file_info = open_output_file(moments_prefix, io_input, io_comm)

        # write a header to the output file
        add_attribute!(fid, "file_info", "Output moments data from the moment_kinetics code")

        # write some overview information to the output file
        write_overview!(fid, composition, collisions, parallel_io, evolve_density,
                        evolve_upar, evolve_p, time_for_setup)

        write_manufactured_solutions!(fid, parallel_io, manufactured_source_list, vz, vr,
                                      vzeta, vpa, vperp, z, r, false)

        # write provenance tracking information to the output file
        write_provenance_tracking_info!(fid, parallel_io, io_input.run_id,
                                        restart_time_index, input_dict,
                                        previous_runs_info)

        # write the input settings
        write_input!(fid, input_dict, parallel_io)

        ### define coordinate dimensions ###
        define_io_coordinates!(fid, vz, vr, vzeta, vpa, vperp, z, r, parallel_io)

        ### create variables for time-dependent quantities and store them ###
        ### in a struct for later access ###
        io_moments = define_dynamic_moment_variables!(
            fid, composition.n_ion_species, composition.n_neutral_species, r, z,
            io_input, external_source_settings, evolve_density, evolve_upar,
            evolve_p, composition.electron_physics, t_params, nl_solver_params)

        close(fid)

        return file_info
    end

    # Should not be called processes other than the root process of each shared-memory
    # group...
    error("setup_moments_io() called by non-block-root block_rank[]=$(block_rank[])")
end

"""
Reopen an existing moments output file to append more data
"""
function reopen_moments_io(file_info)
    @serial_region begin
        filename, io_input, io_comm = file_info
        fid = reopen_output_file(filename, io_input, io_comm)
        dyn = get_group(fid, "dynamic_data")
        timing = get_group(fid, "timing_data")

        variable_list = get_variable_keys(dyn)
        function getvar(name)
            if name ∈ variable_list
                return dyn[name]
            elseif name == "nl_solver_diagnostics"
                nl_names = (name for name ∈ variable_list
                            if occursin("_nonlinear_iterations", name))
                nl_prefixes = (split(name, "_nonlinear_iterations")[1]
                               for name ∈ nl_names)
                return NamedTuple(Symbol(term)=>(n_solves=dyn["$(term)_n_solves"],
                                                 nonlinear_iterations=dyn["$(term)_nonlinear_iterations"],
                                                 linear_iterations=dyn["$(term)_linear_iterations"],
                                                 precon_iterations=dyn["$(term)_precon_iterations"])
                                  for term ∈ nl_prefixes)
            else
                return nothing
            end
        end
        return io_moments_info(fid, getvar("time"), getvar("phi"), getvar("Er"),
                               getvar("Ez"), getvar("density"),
                               getvar("density_loworder"),
                               getvar("density_start_last_timestep"),
                               getvar("parallel_flow"), getvar("parallel_flow_loworder"),
                               getvar("parallel_flow_start_last_timestep"),
                               getvar("pressure"),
                               getvar("pressure_loworder"),
                               getvar("pressure_start_last_timestep"),
                               getvar("parallel_pressure"),
                               getvar("perpendicular_pressure"),
                               getvar("parallel_heat_flux"),
                               getvar("thermal_speed"), getvar("entropy_production"),
                               getvar("chodura_integral_lower"),
                               getvar("chodura_integral_upper"),
                               getvar("electron_density"),
                               getvar("electron_density_loworder"),
                               getvar("electron_density_start_last_timestep"),
                               getvar("electron_parallel_flow"),
                               getvar("electron_parallel_flow_loworder"),
                               getvar("electron_parallel_flow_start_last_timestep"),
                               getvar("electron_pressure"),
                               getvar("electron_pressure_loworder"),
                               getvar("electron_pressure_start_last_timestep"),
                               getvar("electron_parallel_pressure"),
                               getvar("electron_parallel_heat_flux"),
                               getvar("electron_thermal_speed"),
                               getvar("density_neutral"),
                               getvar("density_neutral_loworder"),
                               getvar("density_neutral_start_last_timestep"),
                               getvar("uz_neutral"), getvar("uz_neutral_loworder"),
                               getvar("uz_neutral_start_last_timestep"),
                               getvar("p_neutral"), getvar("p_neutral_loworder"),
                               getvar("p_neutral_start_last_timestep"),
                               getvar("pz_neutral"), getvar("qz_neutral"),
                               getvar("thermal_speed_neutral"),
                               getvar("external_source_amplitude"),
                               getvar("external_source_T_array"),
                               getvar("external_source_density_amplitude"),
                               getvar("external_source_momentum_amplitude"),
                               getvar("external_source_pressure_amplitude"),
                               getvar("external_source_controller_integral"),
                               getvar("external_source_neutral_amplitude"),
                               getvar("external_source_neutral_T_array"),
                               getvar("external_source_neutral_density_amplitude"),
                               getvar("external_source_neutral_momentum_amplitude"),
                               getvar("external_source_neutral_pressure_amplitude"),
                               getvar("external_source_neutral_controller_integral"),
                               getvar("external_source_electron_amplitude"),
                               getvar("external_source_electron_T_array"),
                               getvar("external_source_electron_density_amplitude"),
                               getvar("external_source_electron_momentum_amplitude"),
                               getvar("external_source_electron_pressure_amplitude"),
                               getvar("ion_constraints_A_coefficient"),
                               getvar("ion_constraints_B_coefficient"),
                               getvar("ion_constraints_C_coefficient"),
                               getvar("neutral_constraints_A_coefficient"),
                               getvar("neutral_constraints_B_coefficient"),
                               getvar("neutral_constraints_C_coefficient"),
                               getvar("electron_constraints_A_coefficient"),
                               getvar("electron_constraints_B_coefficient"),
                               getvar("electron_constraints_C_coefficient"),
                               timing["time_for_run"], getvar("step_counter"),
                               getvar("dt"), getvar("previous_dt"), getvar("failure_counter"),
                               getvar("dt_before_last_fail"),getvar("electron_step_counter"),
                               getvar("electron_cumulative_pseudotime"),
                               getvar("electron_dt"), getvar("electron_previous_dt"),
                               getvar("electron_failure_counter"),
                               getvar("electron_dt_before_last_fail"),
                               getvar("nl_solver_diagnostics"), io_input)
    end

    # Should not be called processes other than the root process of each shared-memory
    # group...
    error("reopen_moments_io() called by non-block-root block_rank[]=$(block_rank[])")
end

"""
setup file i/o for distribution function variables
"""
function setup_dfns_io(prefix, io_input, r, z, vperp, vpa, vzeta, vr, vz, composition,
                       collisions, evolve_density, evolve_upar, evolve_p,
                       external_source_settings, manufactured_source_list,
                       restart_time_index, input_dict, io_comm, previous_runs_info,
                       time_for_setup, t_params, nl_solver_params; is_debug=false)

    @serial_region begin
        dfns_prefix = string(prefix, ".dfns")
        parallel_io = io_input.parallel_io
        if !parallel_io
            dfns_prefix *= ".$(iblock_index[])"
        end
        fid, file_info = open_output_file(dfns_prefix, io_input, io_comm)

        # write a header to the output file
        add_attribute!(fid, "file_info",
                       "Output distribution function data from the moment_kinetics code")

        # write some overview information to the output file
        write_overview!(fid, composition, collisions, parallel_io, evolve_density,
                        evolve_upar, evolve_p, time_for_setup)

        write_manufactured_solutions!(fid, parallel_io, manufactured_source_list, vz, vr,
                                      vzeta, vpa, vperp, z, r, true)

        # write provenance tracking information to the output file
        write_provenance_tracking_info!(fid, parallel_io, io_input.run_id,
                                        restart_time_index, input_dict,
                                        previous_runs_info)

        # write the input settings
        write_input!(fid, input_dict, parallel_io)

        ### define coordinate dimensions ###
        define_io_coordinates!(fid, vz, vr, vzeta, vpa, vperp, z, r, parallel_io)

        ### create variables for time-dependent quantities and store them ###
        ### in a struct for later access ###
        io_dfns = define_dynamic_dfn_variables!(
            fid, r, z, vperp, vpa, vzeta, vr, vz, composition, io_input,
            external_source_settings, evolve_density, evolve_upar, evolve_p, t_params,
            nl_solver_params)

        if is_debug
            # create the "istage" variable, used to identify the rk stage where
            # `write_debug_data_to_binary()` was called.
            dynamic = get_group(fid, "dynamic_data")
            io_istage = create_dynamic_variable!(dynamic, "istage", mk_int;
                                                 parallel_io=parallel_io,
                                                 description="RK istage")
            # create the "label" variable, used to identify the
            # `write_debug_data_to_binary()` call-site
            io_label = create_dynamic_variable!(dynamic, "label", String;
                                                parallel_io=parallel_io,
                                                description="call-site label")
        end

        close(fid)

        return file_info
    end

    # Should not be called processes other than the root process of each shared-memory
    # group...
    error("setup_dfns_io() called by non-block-root block_rank[]=$(block_rank[])")
end

"""
Reopen an existing distribution-functions output file to append more data
"""
function reopen_dfns_io(file_info)
    @serial_region begin
        filename, io_input, io_comm = file_info
        fid = reopen_output_file(filename, io_input, io_comm)
        dyn = get_group(fid, "dynamic_data")
        timing = get_group(fid, "timing_data")

        variable_list = get_variable_keys(dyn)
        function getvar(name)
            if name ∈ variable_list
                return dyn[name]
            elseif name == "nl_solver_diagnostics"
                nl_names = (name for name ∈ variable_list
                            if occursin("_nonlinear_iterations", name))
                nl_prefixes = (split(name, "_nonlinear_iterations")[1]
                               for name ∈ nl_names)
                return NamedTuple(Symbol(term)=>(n_solves=dyn["$(term)_n_solves"],
                                                 nonlinear_iterations=dyn["$(term)_nonlinear_iterations"],
                                                 linear_iterations=dyn["$(term)_linear_iterations"],
                                                 precon_iterations=dyn["$(term)_precon_iterations"])
                                  for term ∈ nl_prefixes)
            else
                return nothing
            end
        end
        io_moments = io_moments_info(fid, getvar("time"), getvar("phi"), getvar("Er"),
                                     getvar("Ez"), getvar("density"),
                                     getvar("density_loworder"),
                                     getvar("density_start_last_timestep"),
                                     getvar("parallel_flow"),
                                     getvar("parallel_flow_loworder"),
                                     getvar("parallel_flow_start_last_timestep"),
                                     getvar("pressure"),
                                     getvar("pressure_loworder"),
                                     getvar("pressure_start_last_timestep"),
                                     getvar("parallel_pressure"),
                                     getvar("perpendicular_pressure"),
                                     getvar("parallel_heat_flux"),
                                     getvar("thermal_speed"),
                                     getvar("entropy_production"),
                                     getvar("chodura_integral_lower"),
                                     getvar("chodura_integral_upper"),
                                     getvar("electron_density"),
                                     getvar("electron_density_loworder"),
                                     getvar("electron_density_start_last_timestep"),
                                     getvar("electron_parallel_flow"),
                                     getvar("electron_parallel_flow_loworder"),
                                     getvar("electron_parallel_flow_start_last_timestep"),
                                     getvar("electron_pressure"),
                                     getvar("electron_pressure_loworder"),
                                     getvar("electron_pressure_start_last_timestep"),
                                     getvar("electron_parallel_pressure"),
                                     getvar("electron_parallel_heat_flux"),
                                     getvar("electron_thermal_speed"),
                                     getvar("density_neutral"),
                                     getvar("density_neutral_loworder"),
                                     getvar("density_neutral_start_last_timestep"),
                                     getvar("uz_neutral"), getvar("uz_neutral_loworder"),
                                     getvar("uz_neutral_start_last_timestep"),
                                     getvar("p_neutral"), getvar("p_neutral_loworder"),
                                     getvar("p_neutral_start_last_timestep"),
                                     getvar("pz_neutral"), getvar("qz_neutral"),
                                     getvar("thermal_speed_neutral"),
                                     getvar("external_source_amplitude"),
                                     getvar("external_source_T_array"),
                                     getvar("external_source_density_amplitude"),
                                     getvar("external_source_momentum_amplitude"),
                                     getvar("external_source_pressure_amplitude"),
                                     getvar("external_source_controller_integral"),
                                     getvar("external_source_neutral_amplitude"),
                                     getvar("external_source_neutral_T_array"),
                                     getvar("external_source_neutral_density_amplitude"),
                                     getvar("external_source_neutral_momentum_amplitude"),
                                     getvar("external_source_neutral_pressure_amplitude"),
                                     getvar("external_source_neutral_controller_integral"),
                                     getvar("external_source_electron_amplitude"),
                                     getvar("external_source_electron_T_array"),
                                     getvar("external_source_electron_density_amplitude"),
                                     getvar("external_source_electron_momentum_amplitude"),
                                     getvar("external_source_electron_pressure_amplitude"),
                                     getvar("ion_constraints_A_coefficient"),
                                     getvar("ion_constraints_B_coefficient"),
                                     getvar("ion_constraints_C_coefficient"),
                                     getvar("neutral_constraints_A_coefficient"),
                                     getvar("neutral_constraints_B_coefficient"),
                                     getvar("neutral_constraints_C_coefficient"),
                                     getvar("electron_constraints_A_coefficient"),
                                     getvar("electron_constraints_B_coefficient"),
                                     getvar("electron_constraints_C_coefficient"),
                                     timing["time_for_run"], getvar("step_counter"),
                                     getvar("dt"), getvar("previous_dt"),
                                     getvar("failure_counter"),
                                     getvar("dt_before_last_fail"),
                                     getvar("electron_step_counter"),
                                     getvar("electron_cumulative_pseudotime"),
                                     getvar("electron_dt"),
                                     getvar("electron_previous_dt"),
                                     getvar("electron_failure_counter"),
                                     getvar("electron_dt_before_last_fail"),
                                     getvar("nl_solver_diagnostics"), io_input)

        return io_dfns_info(fid, getvar("f"), getvar("f_loworder"),
                            getvar("f_start_last_timestep"), getvar("f_electron"),
                            getvar("f_electron_loworder"),
                            getvar("f_electron_start_last_timestep"), getvar("f_neutral"),
                            getvar("f_neutral_loworder"),
                            getvar("f_neutral_start_last_timestep"), io_input, io_moments)
    end

    # Should not be called processes other than the root process of each shared-memory
    # group...
    error("reopen_dfns_io() called by non-block-root block_rank[]=$(block_rank[])")
end

"""
    append_to_dynamic_var(io_var, data, t_idx, parallel_io, coords...; only_root=false)

Append `data` to the dynamic variable `io_var`. The time-index of the data being appended
is `t_idx`. `parallel_io` indicates whether parallel I/O is being used. `coords...` is
used to get the ranges to write from/to (needed for parallel I/O) - the entries in the
`coords` tuple can be either `coordinate` instances or integers (for an integer `n` the
range is `1:n`).

If `only_root=true` is passed, the data is only written once - from the global root
process if parallel I/O is being used (if parallel I/O is not used, this has no effect as
each file is only written by one process).
"""
function append_to_dynamic_var end

function append_to_dynamic_var(data::Nothing, args...; kwargs...)
    # Variable was not created to save, so nothing to do.
    return nothing
end

@debug_shared_array begin
    function append_to_dynamic_var(data::DebugMPISharedArray, args...; kwargs...)
        return append_to_dynamic_var(data.data, args...; kwargs...)
    end
end

"""
write time-dependent moments data for ions, electrons and neutrals to the binary output
file
"""
@timeit global_timer write_all_moments_data_to_binary(
                         scratch, moments, fields, n_ion_species, n_neutral_species,
                         io_or_file_info_moments, t_idx, time_for_run, t_params,
                         nl_solver_params, r, z, dfns=false; timing_data=true) = begin

    io_moments = io_or_file_info_moments
    @serial_region begin
        # Only read/write from first process in each 'block'

        if isa(io_or_file_info_moments, io_moments_info)
            io_moments = io_or_file_info_moments
            closefile = false
        else
            io_moments = reopen_moments_io(io_or_file_info_moments)
            closefile = true
        end

        parallel_io = io_moments.io_input.parallel_io
        dynamic = get_group(io_moments.fid, "dynamic_data")

        # add the time for this time slice to the hdf5 file
        append_to_dynamic_var(io_moments.time, t_params.t[], t_idx, parallel_io)

        write_em_fields_data_to_binary(fields, io_moments, t_idx, r, z)

        write_ion_moments_data_to_binary(scratch, moments, n_ion_species, t_params,
                                         io_moments, t_idx, r, z)

        write_electron_moments_data_to_binary(scratch, moments, t_params,
                                              t_params.electron, io_moments, t_idx, r, z)

        write_neutral_moments_data_to_binary(scratch, moments, n_neutral_species,
                                             t_params, io_moments, t_idx, r, z)

        append_to_dynamic_var(io_moments.time_for_run, time_for_run, t_idx, parallel_io)
        append_to_dynamic_var(io_moments.step_counter, t_params.step_counter[], t_idx, parallel_io)
        append_to_dynamic_var(io_moments.dt, t_params.dt_before_output[], t_idx, parallel_io)
        append_to_dynamic_var(io_moments.previous_dt, t_params.previous_dt[], t_idx, parallel_io)
        append_to_dynamic_var(io_moments.failure_counter, t_params.failure_counter[], t_idx, parallel_io)
        dynamic_varnames = collect(keys(dynamic))
        for (k,v) ∈ pairs(t_params.failure_caused_by)
            if "failure_caused_by_$k" ∉ dynamic_varnames
                continue
            end
            io_var = dynamic["failure_caused_by_$k"]
            append_to_dynamic_var(io_var, v, t_idx, parallel_io; only_root=true)
        end
        for (k,v) ∈ pairs(t_params.limit_caused_by)
            if "limit_caused_by_$k" ∉ dynamic_varnames
                continue
            end
            io_var = dynamic["limit_caused_by_$k"]
            append_to_dynamic_var(io_var, v, t_idx, parallel_io; only_root=true)
        end
        append_to_dynamic_var(io_moments.dt_before_last_fail,
                              t_params.dt_before_last_fail[], t_idx, parallel_io)
        for (k,v) ∈ pairs(nl_solver_params)
            if v === nothing
                continue
            end
            append_to_dynamic_var(io_moments.nl_solver_diagnostics[k].n_solves,
                                  v.n_solves[], t_idx, parallel_io)
            append_to_dynamic_var(io_moments.nl_solver_diagnostics[k].nonlinear_iterations,
                                  v.nonlinear_iterations[], t_idx, parallel_io)
            append_to_dynamic_var(io_moments.nl_solver_diagnostics[k].linear_iterations,
                                  v.linear_iterations[], t_idx, parallel_io)
            append_to_dynamic_var(io_moments.nl_solver_diagnostics[k].precon_iterations,
                                  v.precon_iterations[], t_idx, parallel_io)
        end
    end

    if timing_data
        write_timing_data(io_moments, t_idx, dfns)
    end

    @serial_region begin
        closefile && close(io_moments.fid)
    end

    return nothing
end

function write_timing_data(io_moments, t_idx, dfns=false)
    # Timers are created separately on each MPI rank. The timers created on each rank may
    # not be exactly the same - for example some might be created only on processes that
    # contain a domain boundary, or on the root process of each shared-memory block. Also,
    # the timers are only created at run-time, and we would rather not have to maintian a
    # hard-coded list of every timer (which would make it inconvenient to add new timers
    # or debug timers).
    #
    # It is most convenient for every process in `comm_world` to know the complete list
    # of timers that exist on any rank, so we have to gather this list. The list is
    # stored in `timer_names_all_ranks_moments` and `timer_names_all_ranks_dfns`. We
    # store two separate lists because moments and dfns might not be written at the same
    # times, so some timers may be newly added in one but not the other, and so need to
    # be tracked separately.
    #
    # In order to make sure that every process deals with the same variable at the same
    # time, we store the variable names for each rank in a nested SortedDict (with the
    # same nesting structure as the `global_timer` on each rank). The sort applied to
    # the keys of a SortedDict means that we can be sure to iterate through the names in
    # the same order on every process.
    #
    # The timing data for every process in a shared-memory block is collected to the
    # root process of the block to be written to file.
    #
    # In order to avoid communicating many strings at every output step, we first check
    # how new timers there are on each rank, that are not in the global list of timers.
    # The number of new variables is gathered, and if there are no new variables, no
    # communication of variable names needs to be done. If communication is needed, only
    # the names of the new variables need to be gathered.
    #
    # Once the lists of variable names have been updated, the timing data is gathered onto
    # the root process of each shared-memory block. The data is communicated as vectors of
    # integers, with the order of the entries being determined by the order of the nested
    # SortedDict.
    #
    # Each entry in `timer_names_all_ranks_moments` and `timer_names_all_ranks_dfns` is
    # a SortedDict, even if it does not yet contain any other entries, because
    # sub-timers might be added at any point.

    if dfns
        timer_names_all_ranks = timer_names_all_ranks_dfns
    else
        timer_names_all_ranks = timer_names_all_ranks_moments
    end

    # Collect the names that have not been used on any rank before this call.
    unique_new_names = String[]

    if block_rank[] == 0
        io_group = get_group(io_moments.fid, "timing_data")
    end

    # Find any new timer names on this process.
    function get_new_timer_names()
        new_timer_names = String[]
        function get_names_inner(this_timer, timer_names_subdict, prefix)
            names_subdict_keys = keys(timer_names_subdict)
            inner_timers_keys = collect(keys(this_timer.inner_timers))
            # Sort keys to ensure that the list created by this function has a
            # deterministic order.
            sort!(inner_timers_keys)
            for k ∈ inner_timers_keys
                this_name = prefix == "" ? k : prefix * ";" * k
                if k ∉ names_subdict_keys
                    push!(new_timer_names, this_name)
                    subsubdict = TimerNamesDict()
                else
                    subsubdict = timer_names_subdict[k]
                end
                get_names_inner(this_timer[k], subsubdict, this_name)
            end
            return nothing
        end
        get_names_inner(global_timer, timer_names_all_ranks, "")
        return new_timer_names
    end

    # Add the new timer names in to `timer_names_all_ranks_moments` or
    # `timer_names_all_ranks_dfns`. Note that there may be duplicate names in
    # new_timer_names, and these will be ignored.
    function add_new_timer_names!(new_timer_names)
        for n ∈ new_timer_names
            this_dict_all_ranks = timer_names_all_ranks
            split_name = split(n, ";")
            n_levels = length(split_name)
            for (i, level) ∈ enumerate(split_name)
                if level ∉ keys(this_dict_all_ranks)
                    this_dict_all_ranks[level] = TimerNamesDict()
                    if i == n_levels
                        # New variable that has not been used on any rank before.
                        push!(unique_new_names, n)
                    end
                end
                this_dict_all_ranks = this_dict_all_ranks[level]
            end
        end
        return nothing
    end

    # Create variables in the output file for any new timers. This needs to be called
    # simultaneously by all processes in `comm_inter_block[]`.
    function create_new_timer_io_variables!(new_timer_names, timer_group, parallel_io)
        for n ∈ new_timer_names
            create_dynamic_variable!(io_group, "time:" * n, mk_int,
                                     (name="rank", n=global_size[]);
                                     parallel_io=parallel_io)
            create_dynamic_variable!(io_group, "ncalls:" * n, mk_int,
                                     (name="rank", n=global_size[]);
                                     parallel_io=parallel_io)
            create_dynamic_variable!(io_group, "allocs:" * n, mk_int,
                                     (name="rank", n=global_size[]);
                                     parallel_io=parallel_io)
        end
        return nothing
    end

    # Once all the names are known, use this function to collect all the data from timers
    # on this process into arrays to be communicated.
    function get_data_from_timers()
        times = mk_int[]
        ncalls = mk_int[]
        allocs = mk_int[]
        function walk_through_timers(names_dict, timer)
            if timer === nothing
                # Timer not found on this rank, so set to 0
                push!(times, 0.0)
                push!(ncalls, 0.0)
                push!(allocs, 0.0)
            else
                push!(times, timer.accumulated_data.time)
                push!(ncalls, timer.accumulated_data.ncalls)
                push!(allocs, timer.accumulated_data.allocs)
            end

            # Note that here we have to get the order of entries from names_dict (which is
            # a SortedDict) to ensure that the order of each list is consistent between
            # all different processes.
            for (sub_name, sub_dict) ∈ pairs(names_dict)
                if timer === nothing || sub_name ∉ keys(timer.inner_timers)
                    walk_through_timers(sub_dict, nothing)
                else
                    walk_through_timers(sub_dict, timer[sub_name])
                end
            end
        end
        for name ∈ keys(timer_names_all_ranks)
            walk_through_timers(timer_names_all_ranks[name], global_timer[name])
        end
        return times, ncalls, allocs
    end

    parallel_io = io_moments.io_input.parallel_io
    if parallel_io
        comm = comm_world
        comm_size = global_size[]
        comm_rank = global_rank[]
    else
        comm = comm_block[]
        comm_size = block_size[]
        comm_rank = block_rank[]
    end

    # First count how many new timer names all processes
    new_names_this_rank = get_new_timer_names()
    n_new_names = Ref(length(new_names_this_rank))
    MPI.Allreduce!(n_new_names, +, comm)

    # Allgather new names onto all ranks - if parallel_io=true this is all ranks in
    # comm_world, if parallel_io=false it is all ranks in comm_block[] (as each block
    # writes output independently).
    if n_new_names[] > 0
        # Pack all new names into a single string for communication.
        new_names_string = string((s * "&" for s ∈ new_names_this_rank)...)

        # Get the sizes of the per-rank strings that need to be gathered
        string_sizes = Vector{mk_int}(undef, comm_size)
        string_sizes[comm_rank+1] = length(new_names_string)
        string_buffer = MPI.UBuffer(string_sizes, 1)
        MPI.Allgather!(string_buffer, comm)

        # Gather the strings
        gathered_char_vector = Vector{Char}(undef, sum(string_sizes))
        local_start_index = sum(string_sizes[1:comm_rank]) + 1
        local_end_index = local_start_index - 1 + string_sizes[comm_rank+1]
        gathered_char_vector[local_start_index:local_end_index] .= [new_names_string...]
        gathered_buffer = MPI.VBuffer(gathered_char_vector, string_sizes)
        MPI.Allgatherv!(gathered_buffer, comm)

        # The string will end with a "&", so we need to slice off the final element, which
        # will be an empty string.
        all_new_names = split(string(gathered_char_vector...), "&")[1:end-1]

        # Add the new names to timer_names_per_rank
        add_new_timer_names!(all_new_names)

        if block_rank[] == 0
            create_new_timer_io_variables!(unique_new_names, io_group, parallel_io)
        end
    end

    # Collect the timing data onto the root process of each block
    times_data, ncalls_data, allocs_data = get_data_from_timers()
    n_timers = length(times_data)
    gathered_times_data = MPI.Gather(times_data, comm_block[]; root=0)
    gathered_ncalls_data = MPI.Gather(ncalls_data, comm_block[]; root=0)
    gathered_allocs_data = MPI.Gather(allocs_data, comm_block[]; root=0)

    if block_rank[] == 0
        gathered_times_data = reshape(gathered_times_data, n_timers, block_size[])
        gathered_ncalls_data = reshape(gathered_ncalls_data, n_timers, block_size[])
        gathered_allocs_data = reshape(gathered_allocs_data, n_timers, block_size[])

        # Write the timer variables

        # We iterate through the variables in the same order as they were packed into
        # the array in `get_data_from_timers()`, so `counter` gets the corresponding
        # data from the flattened array that was communicated.
        counter = 1

        # Write data for all processes in the shared memory block, which must be
        # unpacked from the arrays that were communicated.
        if t_idx < 0
            # The top-level timer (usually "moment_kinetics" was probably the
            # first one created. It definitely exists, and should have been
            # written at each timestep.
            # If we got the length of `time:$this_name`, the variable might have
            # the wrong length (e.g. if it has only just been created and has
            # length 1).
            length_check_var = io_group["time:" * first(keys(timer_names_all_ranks))]
            this_t_idx = size(length_check_var, ndims(length_check_var))
        else
            this_t_idx = t_idx
        end
        if parallel_io
            timer_coord = (local_io_range=1:block_size[],
                           global_io_range=global_rank[]+1:global_rank[]+block_size[])
        else
            timer_coord = (local_io_range=1:block_size[],
                           global_io_range=1:block_size[])
        end
        function write_level(names_dict, this_name)
            io_time = io_group["time:" * this_name]
            io_ncalls = io_group["ncalls:" * this_name]
            io_allocs = io_group["allocs:" * this_name]
            @views append_to_dynamic_var(io_time, gathered_times_data[counter,:],
                                         this_t_idx, parallel_io, timer_coord)
            @views append_to_dynamic_var(io_ncalls, gathered_ncalls_data[counter,:],
                                         this_t_idx, parallel_io, timer_coord)
            @views append_to_dynamic_var(io_allocs, gathered_allocs_data[counter,:],
                                         this_t_idx, parallel_io, timer_coord)
            counter += 1
            for (sub_name, sub_dict) ∈ pairs(names_dict)
                write_level(sub_dict, this_name * ";" * sub_name)
            end
        end

        for top_level_name ∈ keys(timer_names_all_ranks)
            write_level(timer_names_all_ranks[top_level_name], top_level_name)
        end
    end

    # Pick a fixed size for "global_timer_string" so that we can overwrite the variable
    # without needing to resize it.
    global_timer_string_size = 10000 # 100 characters x 100 lines seems like a reasonable maximum size.
    global_timer_description = "Formatted representation of global_timer"
    if global_rank[] == 0 || (block_rank[] == 0 && !parallel_io)
        if t_idx > 1 || t_idx == -1
            if t_idx == -1
                top_level = nothing
            else
                if "moment_kinetics" ∈ keys(timer_names_all_ranks)
                    top_level = ("moment_kinetics", "time_advance! step", "ssp_rk!")
                    this_dict = timer_names_all_ranks

                    # Check all the expected levels are present, otherwise just set
                    # top_level=nothing.
                    for n ∈ top_level
                        if n ∉ keys(this_dict)
                            top_level = nothing
                            break
                        end
                        this_dict = this_dict[n]
                    end
                else
                    # If `time_advance!()` was called in a non-standard way (i.e. not by
                    # `run_moment_kinetics()`), the actual timers might be different. In
                    # this case, skip trying to pretty-up the "global_timer_string".
                    top_level = nothing
                end
            end
            # Write a formatted string showing the TimerOutput data, that replicates what
            # was printed to the terminal, for a quick look.
            string_to_write = format_global_timer(; show_output=false,
                                                    top_level=top_level)

            # Ensure `string_to_write` is no longer than `global_timer_string_size`.
            string_to_write = string_to_write[1:min(length(string_to_write), global_timer_string_size)]
            # Ensure `string_to_write` is at least as long as `global_timer_string_size`.
            # Do this way instead of using `rpad()` because `rpad()` measures the length
            # using `textwidth()` rather than a raw character count, whereas we want a
            # fixed number of ASCII characters to write to the output file.
            string_to_write = string_to_write * ' '^(global_timer_string_size - length(string_to_write))

            write_single_value!(get_group(io_moments.fid, "timing_data"),
                                "global_timer_string", string_to_write;
                                parallel_io=parallel_io,
                                description=global_timer_description,
                                overwrite=true)
        end
    elseif block_rank[] == 0
        if t_idx > 1 || t_idx == -1
            # Although only global_rank[]==0 needs to write "global_timer_string" when we
            # are using parallel I/O, other ranks in `comm_inter_block[]` must also call
            # `write_single_value!() so that the variable in the HDF5 file can be created.
            # These other ranks do not actually write data though, so it does not matter
            # what is passed to the `data` argument of `write_single_value!()` (as
            # long as it is a string with the right length).
            string_to_write = " " ^ global_timer_string_size
            write_single_value!(get_group(io_moments.fid, "timing_data"),
                                "global_timer_string", string_to_write;
                                parallel_io=parallel_io,
                                description=global_timer_description,
                                overwrite=true)
        end
    end

    return nothing
end

"""
    write_final_timing_data_to_binary(io_or_file_info_moments)

Write the timing data in [`moment_kinetics.timer_utils.global_timer`](@ref) to the output
file. Needs to be called after exiting from the `@timeit` block so that all timers are
finalised properly.
"""
function write_final_timing_data_to_binary(io_or_file_info_moments, io_or_file_info_dfns)
    io_moments = io_or_file_info_moments
    io_dfns_moments = io_or_file_info_dfns
    @serial_region begin
        # Only read/write from first process in each 'block'

        if isa(io_or_file_info_moments, io_moments_info)
            io_moments = io_or_file_info_moments
            closefile = false
        else
            io_moments = reopen_moments_io(io_or_file_info_moments)
            closefile = true
        end

        if isa(io_or_file_info_dfns, io_dfns_info)
            io_dfns = io_or_file_info_dfns
            closefile = false
        else
            io_dfns = reopen_dfns_io(io_or_file_info_dfns)
            closefile = true
        end
        io_dfns_moments = io_dfns.io_moments
    end

    write_timing_data(io_moments, -1)
    write_timing_data(io_dfns_moments, -1, true)

    @serial_region begin
        closefile && close(io_moments.fid)
        closefile && close(io_dfns.fid)
    end
    return nothing
end

"""
write time-dependent EM fields data to the binary output file

Note: should only be called from within a function that (re-)opens the output file.
"""
function write_em_fields_data_to_binary(fields, io_moments::io_moments_info, t_idx, r, z)
    @serial_region begin
        # Only read/write from first process in each 'block'

        parallel_io = io_moments.io_input.parallel_io

        # add the electrostatic potential and electric field components at this time slice to the hdf5 file
        append_to_dynamic_var(io_moments.phi, fields.phi, t_idx, parallel_io, z, r)
        append_to_dynamic_var(io_moments.Er, fields.Er, t_idx, parallel_io, z, r)
        append_to_dynamic_var(io_moments.Ez, fields.Ez, t_idx, parallel_io, z, r)
    end

    return nothing
end

"""
write time-dependent moments data for ions to the binary output file

Note: should only be called from within a function that (re-)opens the output file.
"""
function write_ion_moments_data_to_binary(scratch, moments, n_ion_species, t_params,
                                          io_moments::io_moments_info, t_idx, r, z)
    @serial_region begin
        # Only read/write from first process in each 'block'

        parallel_io = io_moments.io_input.parallel_io

        # add the density data at this time slice to the output file
        append_to_dynamic_var(io_moments.density, scratch[t_params.n_rk_stages+1].density,
                              t_idx, parallel_io, z, r, n_ion_species)
        # If options were not set to select the following outputs, then the io variables
        # will be `nothing` and nothing will be written.
        append_to_dynamic_var(io_moments.density_loworder, scratch[2].density, t_idx,
                              parallel_io, z, r, n_ion_species)
        append_to_dynamic_var(io_moments.density_start_last_timestep, scratch[1].density,
                              t_idx, parallel_io, z, r, n_ion_species)

        append_to_dynamic_var(io_moments.parallel_flow,
                              scratch[t_params.n_rk_stages+1].upar, t_idx, parallel_io, z,
                              r, n_ion_species)
        # If options were not set to select the following outputs, then the io variables
        # will be `nothing` and nothing will be written.
        append_to_dynamic_var(io_moments.parallel_flow_loworder, scratch[2].upar, t_idx,
                              parallel_io, z, r, n_ion_species)
        append_to_dynamic_var(io_moments.parallel_flow_start_last_timestep,
                              scratch[1].upar, t_idx, parallel_io, z, r, n_ion_species)

        append_to_dynamic_var(io_moments.pressure, scratch[t_params.n_rk_stages+1].p,
                              t_idx, parallel_io, z, r, n_ion_species)
        # If options were not set to select the following outputs, then the io variables
        # will be `nothing` and nothing will be written.
        append_to_dynamic_var(io_moments.pressure_loworder, scratch[2].p,
                              t_idx, parallel_io, z, r, n_ion_species)
        append_to_dynamic_var(io_moments.pressure_start_last_timestep, scratch[1].p,
                              t_idx, parallel_io, z, r, n_ion_species)

        append_to_dynamic_var(io_moments.parallel_pressure, moments.ion.ppar, t_idx,
                              parallel_io, z, r, n_ion_species)

        append_to_dynamic_var(io_moments.perpendicular_pressure, moments.ion.pperp, t_idx,
                              parallel_io, z, r, n_ion_species)

        append_to_dynamic_var(io_moments.parallel_heat_flux, moments.ion.qpar, t_idx,
                              parallel_io, z, r, n_ion_species)
        append_to_dynamic_var(io_moments.thermal_speed, moments.ion.vth, t_idx,
                              parallel_io, z, r, n_ion_species)
        append_to_dynamic_var(io_moments.entropy_production, moments.ion.dSdt, t_idx,
                              parallel_io, z, r, n_ion_species)
        if z.irank == 0 # lower wall 
            append_to_dynamic_var(io_moments.chodura_integral_lower,
                                  moments.ion.chodura_integral_lower, t_idx,
                                  parallel_io, r, n_ion_species)
        elseif io_moments.chodura_integral_lower !== nothing
            append_to_dynamic_var(io_moments.chodura_integral_lower,
                                  moments.ion.chodura_integral_lower, t_idx,
                                  parallel_io, 0, n_ion_species)
        end
        if z.irank == z.nrank - 1 # upper wall
            append_to_dynamic_var(io_moments.chodura_integral_upper,
                                  moments.ion.chodura_integral_upper, t_idx,
                                  parallel_io, r, n_ion_species)
        elseif io_moments.chodura_integral_upper !== nothing
            append_to_dynamic_var(io_moments.chodura_integral_upper,
                                  moments.ion.chodura_integral_upper, t_idx,
                                  parallel_io, 0, n_ion_species)
        end
        if io_moments.external_source_amplitude !== nothing
            n_sources = size(moments.ion.external_source_amplitude)[3]
            append_to_dynamic_var(io_moments.external_source_amplitude,
                                  moments.ion.external_source_amplitude, t_idx,
                                  parallel_io, z, r, n_sources)
            append_to_dynamic_var(io_moments.external_source_T_array,
                                  moments.ion.external_source_T_array, t_idx,
                                  parallel_io, z, r, n_sources)
            if moments.evolve_density
                append_to_dynamic_var(io_moments.external_source_density_amplitude,
                                      moments.ion.external_source_density_amplitude,
                                      t_idx, parallel_io, z, r, n_sources)
            end
            if moments.evolve_upar
                append_to_dynamic_var(io_moments.external_source_momentum_amplitude,
                                      moments.ion.external_source_momentum_amplitude,
                                      t_idx, parallel_io, z, r, n_sources)
            end
            if moments.evolve_p
                append_to_dynamic_var(io_moments.external_source_pressure_amplitude,
                                      moments.ion.external_source_pressure_amplitude,
                                      t_idx, parallel_io, z, r, n_sources)
            end
        end
        if io_moments.external_source_controller_integral !== nothing
            n_sources = size(moments.ion.external_source_amplitude)[3]
            if size(moments.ion.external_source_controller_integral) == (1,1, n_sources)
                append_to_dynamic_var(io_moments.external_source_controller_integral,
                                      moments.ion.external_source_controller_integral,
                                      t_idx, parallel_io, 1, 1, n_sources)
            else
                append_to_dynamic_var(io_moments.external_source_controller_integral,
                                      moments.ion.external_source_controller_integral,
                                      t_idx, parallel_io, z, r, n_sources)
            end
        end
        if moments.evolve_density || moments.evolve_upar || moments.evolve_p
            append_to_dynamic_var(io_moments.ion_constraints_A_coefficient,
                                  moments.ion.constraints_A_coefficient, t_idx,
                                  parallel_io, z, r, n_ion_species)
            append_to_dynamic_var(io_moments.ion_constraints_B_coefficient,
                                  moments.ion.constraints_B_coefficient, t_idx,
                                  parallel_io, z, r, n_ion_species)
            append_to_dynamic_var(io_moments.ion_constraints_C_coefficient,
                                  moments.ion.constraints_C_coefficient, t_idx,
                                  parallel_io, z, r, n_ion_species)
        end
    end

    return nothing
end

"""
write time-dependent moments data for electrons to the binary output file

Note: should only be called from within a function that (re-)opens the output file.
"""
function write_electron_moments_data_to_binary(scratch, moments, t_params, electron_t_params,
                                               io_moments::Union{io_moments_info,io_initial_electron_info},
                                               t_idx, r, z, ir=nothing)
    if (ir === nothing && block_rank[] == 0) || anyzv_subblock_rank[] == 0
        # Only read/write from first process in each 'block' (for 'initial_electron' I/O)
        # or anyzv subblock (for debug I/O that is written independently for each `ir`).

        parallel_io = io_moments.io_input.parallel_io
        dynamic = get_group(io_moments.fid, "dynamic_data")

        function get_from_ir(x::AbstractMatrix)
            if ir === nothing
                return x
            else
                return @view x[:,ir:ir]
            end
        end
        function get_from_ir(x::AbstractArray{T,3} where T)
            if ir === nothing
                return x
            else
                return @view x[:,ir:ir,:]
            end
        end

        if io_moments.electron_density !== nothing
            append_to_dynamic_var(io_moments.electron_density,
                                  get_from_ir(scratch[t_params.n_rk_stages+1].electron_density),
                                  t_idx, parallel_io, z, r)
            # If options were not set to select the following outputs, then the io variables
            # will be `nothing` and nothing will be written.
            append_to_dynamic_var(io_moments.electron_density_loworder,
                                  get_from_ir(scratch[2].electron_density), t_idx,
                                  parallel_io, z, r)
            append_to_dynamic_var(io_moments.electron_density_start_last_timestep,
                                  get_from_ir(scratch[1].electron_density), t_idx,
                                  parallel_io, z, r)
        end

        if io_moments.electron_parallel_flow !== nothing
            append_to_dynamic_var(io_moments.electron_parallel_flow,
                                  get_from_ir(scratch[t_params.n_rk_stages+1].electron_upar),
                                  t_idx, parallel_io, z, r)
            # If options were not set to select the following outputs, then the io variables
            # will be `nothing` and nothing will be written.
            append_to_dynamic_var(io_moments.electron_parallel_flow_loworder,
                                  get_from_ir(scratch[2].electron_upar), t_idx,
                                  parallel_io, z, r)
            append_to_dynamic_var(io_moments.electron_parallel_flow_start_last_timestep,
                                  get_from_ir(scratch[1].electron_upar), t_idx,
                                  parallel_io, z, r)
        end

        append_to_dynamic_var(io_moments.electron_pressure,
                              get_from_ir(scratch[t_params.n_rk_stages+1].electron_p),
                              t_idx, parallel_io, z, r)
        # If options were not set to select the following outputs, then the io variables
        # will be `nothing` and nothing will be written.
        append_to_dynamic_var(io_moments.electron_pressure_loworder,
                              get_from_ir(scratch[2].electron_p), t_idx, parallel_io, z,
                              r)
        append_to_dynamic_var(io_moments.electron_pressure_start_last_timestep,
                              get_from_ir(scratch[1].electron_p), t_idx, parallel_io, z,
                              r)

        append_to_dynamic_var(io_moments.electron_parallel_pressure,
                              get_from_ir(moments.electron.ppar), t_idx, parallel_io, z,
                              r)

        append_to_dynamic_var(io_moments.electron_parallel_heat_flux,
                              get_from_ir(moments.electron.qpar), t_idx, parallel_io, z,
                              r)
        append_to_dynamic_var(io_moments.electron_thermal_speed,
                              get_from_ir(moments.electron.vth), t_idx, parallel_io, z, r)
        if io_moments.external_source_electron_amplitude !== nothing
            n_sources = size(moments.electron.external_source_amplitude)[3]
            append_to_dynamic_var(io_moments.external_source_electron_amplitude,
                                  get_from_ir(moments.electron.external_source_amplitude),
                                  t_idx, parallel_io, z, r, n_sources)
            append_to_dynamic_var(io_moments.external_source_electron_T_array,
                                  get_from_ir(moments.electron.external_source_T_array),
                                  t_idx, parallel_io, z, r, n_sources)
            append_to_dynamic_var(io_moments.external_source_electron_density_amplitude,
                                  get_from_ir(moments.electron.external_source_density_amplitude),
                                  t_idx, parallel_io, z, r, n_sources)
            append_to_dynamic_var(io_moments.external_source_electron_momentum_amplitude,
                                  get_from_ir(moments.electron.external_source_momentum_amplitude),
                                  t_idx, parallel_io, z, r, n_sources)
            append_to_dynamic_var(io_moments.external_source_electron_pressure_amplitude,
                                  get_from_ir(moments.electron.external_source_pressure_amplitude),
                                  t_idx, parallel_io, z, r, n_sources)
        end
        append_to_dynamic_var(io_moments.electron_constraints_A_coefficient,
                              get_from_ir(moments.electron.constraints_A_coefficient),
                              t_idx, parallel_io, z, r)
        append_to_dynamic_var(io_moments.electron_constraints_B_coefficient,
                              get_from_ir(moments.electron.constraints_B_coefficient),
                              t_idx, parallel_io, z, r)
        append_to_dynamic_var(io_moments.electron_constraints_C_coefficient,
                              get_from_ir(moments.electron.constraints_C_coefficient),
                              t_idx, parallel_io, z, r)

        if electron_t_params !== nothing
            # Save timestepping info

            function get_from_ir_1d(s)
                if ir === nothing
                    return s
                else
                    return @view s[ir:ir]
                end
            end

            append_to_dynamic_var(io_moments.electron_step_counter,
                                  get_from_ir_1d(electron_t_params.step_counter), t_idx,
                                  parallel_io, r)
            append_to_dynamic_var(io_moments.electron_cumulative_pseudotime,
                                  get_from_ir_1d(electron_t_params.t), t_idx, parallel_io,
                                  r)
            # We don't write `electron_t_params.dt_before_output` here because either the
            # electrons advance with the ion timestep and electron_dt does not matter, or
            # the electrons advance inside a pseudotimestepping loop on each ion timestep
            # stage. In the latter case, output is only ever written after the end of the
            # pseudotimestepping loop, at an ion output step, and dt_before_output is not
            # set by `adaptive_timestep_update_t_params!()`, so it should not be written
            # here. Instead it is correct to write just `electron_t_params.dt`, as the
            # electron dt is never shortened to hit an exact output time (which is why
            # dt_before_output is needed for the ions).
            append_to_dynamic_var(io_moments.electron_dt,
                                  get_from_ir_1d(electron_t_params.dt),
                                  t_idx, parallel_io, r)
            append_to_dynamic_var(io_moments.electron_previous_dt,
                                  get_from_ir_1d(electron_t_params.previous_dt), t_idx,
                                  parallel_io, r)
            append_to_dynamic_var(io_moments.electron_failure_counter,
                                  get_from_ir_1d(electron_t_params.failure_counter),
                                  t_idx, parallel_io, r)
            dynamic_keys = collect(keys(dynamic))

            if ir === nothing
                # When writing all r-indices, `only_root` indicates the global root
                # process.
                only_root = true
            else
                # When writing a single r-ind.x, pass an MPI communicator for `only_root`
                # to indicate that we write only from the root of the anyzv subblock.
                only_root = comm_anysv_subblock[]
            end

            for (k,v) ∈ pairs(electron_t_params.failure_caused_by)
                # Only write these variables if they were created in the output file,
                # because sometimes (e.g. for debug_io=true) they are not needed.
                if k ∈ dynamic_keys
                    io_var = dynamic["electron_failure_caused_by_$k"]
                    append_to_dynamic_var(io_var, get_from_ir_1d(v), t_idx, parallel_io,
                                          r; only_root=only_root)
                end
            end
            for (k,v) ∈ pairs(electron_t_params.limit_caused_by)
                # Only write these variables if they were created in the output file,
                # because sometimes (e.g. for debug_io=true) they are not needed.
                if k ∈ dynamic_keys
                    io_var = dynamic["electron_limit_caused_by_$k"]
                    append_to_dynamic_var(io_var, get_from_ir_1d(v), t_idx, parallel_io,
                                          r; only_root=only_root)
                end
            end
            append_to_dynamic_var(io_moments.electron_dt_before_last_fail,
                                  get_from_ir_1d(electron_t_params.dt_before_last_fail),
                                  t_idx, parallel_io, r)
        end
    end

    return nothing
end

"""
write time-dependent moments data for neutrals to the binary output file

Note: should only be called from within a function that (re-)opens the output file.
"""
function write_neutral_moments_data_to_binary(scratch, moments, n_neutral_species,
                                              t_params, io_moments::io_moments_info,
                                              t_idx, r, z)
    if n_neutral_species ≤ 0
        return nothing
    end

    @serial_region begin
        # Only read/write from first process in each 'block'

        parallel_io = io_moments.io_input.parallel_io

        append_to_dynamic_var(io_moments.density_neutral,
                              scratch[t_params.n_rk_stages+1].density_neutral, t_idx,
                              parallel_io, z, r, n_neutral_species)
        # If options were not set to select the following outputs, then the io variables
        # will be `nothing` and nothing will be written.
        append_to_dynamic_var(io_moments.density_neutral_loworder,
                              scratch[2].density_neutral, t_idx, parallel_io, z, r,
                              n_neutral_species)
        append_to_dynamic_var(io_moments.density_neutral_start_last_timestep,
                              scratch[1].density_neutral, t_idx, parallel_io, z, r,
                              n_neutral_species)

        append_to_dynamic_var(io_moments.uz_neutral,
                              scratch[t_params.n_rk_stages+1].uz_neutral, t_idx,
                              parallel_io, z, r, n_neutral_species)
        # If options were not set to select the following outputs, then the io variables
        # will be `nothing` and nothing will be written.
        append_to_dynamic_var(io_moments.uz_neutral_loworder,
                              scratch[2].uz_neutral, t_idx, parallel_io, z, r,
                              n_neutral_species)
        append_to_dynamic_var(io_moments.uz_neutral_start_last_timestep,
                              scratch[1].uz_neutral, t_idx, parallel_io, z, r,
                              n_neutral_species)

        append_to_dynamic_var(io_moments.p_neutral,
                              scratch[t_params.n_rk_stages+1].p_neutral, t_idx,
                              parallel_io, z, r, n_neutral_species)
        # If options were not set to select the following outputs, then the io variables
        # will be `nothing` and nothing will be written.
        append_to_dynamic_var(io_moments.p_neutral_loworder,
                              scratch[2].p_neutral, t_idx, parallel_io, z, r,
                              n_neutral_species)
        append_to_dynamic_var(io_moments.p_neutral_start_last_timestep,
                              scratch[1].p_neutral, t_idx, parallel_io, z, r,
                              n_neutral_species)

        append_to_dynamic_var(io_moments.pz_neutral, moments.neutral.pz, t_idx,
                              parallel_io, z, r, n_neutral_species)

        append_to_dynamic_var(io_moments.qz_neutral, moments.neutral.qz, t_idx,
                              parallel_io, z, r, n_neutral_species)
        append_to_dynamic_var(io_moments.thermal_speed_neutral, moments.neutral.vth,
                              t_idx, parallel_io, z, r, n_neutral_species)
        if io_moments.external_source_neutral_amplitude !== nothing
            n_sources = size(moments.neutral.external_source_amplitude)[3]
            append_to_dynamic_var(io_moments.external_source_neutral_amplitude,
                                  moments.neutral.external_source_amplitude, t_idx,
                                  parallel_io, z, r, n_sources)
            append_to_dynamic_var(io_moments.external_source_neutral_T_array,
                                  moments.neutral.external_source_T_array, t_idx,
                                  parallel_io, z, r, n_sources)
            if moments.evolve_density
                append_to_dynamic_var(io_moments.external_source_neutral_density_amplitude,
                                      moments.neutral.external_source_density_amplitude,
                                      t_idx, parallel_io, z, r, n_sources)
            end
            if moments.evolve_upar
                append_to_dynamic_var(io_moments.external_source_neutral_momentum_amplitude,
                                      moments.neutral.external_source_momentum_amplitude,
                                      t_idx, parallel_io, z, r, n_sources)
            end
            if moments.evolve_p
                append_to_dynamic_var(io_moments.external_source_neutral_pressure_amplitude,
                                      moments.neutral.external_source_pressure_amplitude,
                                      t_idx, parallel_io, z, r, n_sources)
            end
        end
        if io_moments.external_source_neutral_controller_integral !== nothing
            n_sources = size(moments.neutral.external_source_amplitude)[3]
            if size(moments.neutral.external_source_neutral_controller_integral) == (1,1, n_sources)
                append_to_dynamic_var(io_moments.external_source_neutral_controller_integral,
                                      moments.neutral.external_source_controller_integral,
                                      t_idx, parallel_io, 1, 1, n_sources)
            else
                append_to_dynamic_var(io_moments.external_source_neutral_controller_integral,
                                      moments.neutral.external_source_controller_integral,
                                      t_idx, parallel_io, z, r, n_sources)
            end
        end
        if moments.evolve_density || moments.evolve_upar || moments.evolve_p
            append_to_dynamic_var(io_moments.neutral_constraints_A_coefficient,
                                  moments.neutral.constraints_A_coefficient, t_idx,
                                  parallel_io, z, r, n_neutral_species)
            append_to_dynamic_var(io_moments.neutral_constraints_B_coefficient,
                                  moments.neutral.constraints_B_coefficient, t_idx,
                                  parallel_io, z, r, n_neutral_species)
            append_to_dynamic_var(io_moments.neutral_constraints_C_coefficient,
                                  moments.neutral.constraints_C_coefficient, t_idx,
                                  parallel_io, z, r, n_neutral_species)
        end
    end

    return nothing
end

"""
write time-dependent distribution function data for ions, electrons and neutrals to the
binary output file
"""
@timeit global_timer write_all_dfns_data_to_binary(
                         scratch, scratch_electron, moments, fields, n_ion_species,
                         n_neutral_species, io_or_file_info_dfns, t_idx, time_for_run,
                         t_params, nl_solver_params, r, z, vperp, vpa, vzeta, vr,
                         vz; is_debug=false, label=nothing, istage=nothing) = begin
    io_dfns = nothing
    io_dfns_moments = io_or_file_info_dfns
    closefile = true
    @serial_region begin
        # Only read/write from first process in each 'block'

        if isa(io_or_file_info_dfns, io_dfns_info)
            io_dfns = io_or_file_info_dfns
            closefile = false
        else
            io_dfns = reopen_dfns_io(io_or_file_info_dfns)
            closefile = true
        end

        io_dfns_moments = io_dfns.io_moments

        if is_debug
            # Figure out the current length of the debug file
            dynamic = get_group(io_dfns.fid, "dynamic_data")
            parallel_io = io_dfns.io_input.parallel_io
            t_idx = length(dynamic["time"]) + 1
            append_to_dynamic_var(dynamic["istage"], istage, t_idx, parallel_io)
            append_to_dynamic_var(dynamic["label"], label, t_idx, parallel_io)
        end
    end

    # Write the moments for this time slice to the output file.
    # This also updates the time.
    write_all_moments_data_to_binary(scratch, moments, fields, n_ion_species,
                                     n_neutral_species, io_dfns_moments, t_idx,
                                     time_for_run, t_params, nl_solver_params, r, z, true;
                                     timing_data=!is_debug)

    @serial_region begin
        # add the distribution function data at this time slice to the output file
        write_ion_dfns_data_to_binary(scratch, t_params, n_ion_species, io_dfns, t_idx, r,
                                      z, vperp, vpa)
        if t_params.kinetic_electron_solver ∈ (implicit_time_evolving, explicit_time_evolving) || scratch_electron !== nothing
            write_electron_dfns_data_to_binary(scratch, scratch_electron, t_params,
                                               io_dfns, t_idx, r, z, vperp, vpa)
        end
        write_neutral_dfns_data_to_binary(scratch, t_params, n_neutral_species, io_dfns,
                                          t_idx, r, z, vzeta, vr, vz)

        closefile && close(io_dfns.fid)
    end
    return nothing
end

"""
    write_debug_data_to_binary(this_scratch, moments, fields, composition, t_params,
                               r, z, vperp, vpa, vzeta, vr, vz, label, istage)

If `t_params.debug_io` represents an output file (rather than being `nothing`), write the
state contained in `this_scratch`, `moments`, and `fields` to that output file. `label` is
a String identifying the location this function was called from (for reference when
debugging). `istage` should be the Runge-Kutta stage that this function was called from
(when called from within the loop over Runge-Kutta stages).
"""
function write_debug_data_to_binary(this_scratch, this_scratch_electron, moments, fields,
                                    composition, t_params, r, z, vperp, vpa, vzeta, vr,
                                    vz, label, istage)
    if t_params.debug_io === nothing
        # Not using debug IO, so nothing to do. When t_params.debug_io is `nothing`, this
        # is known at compile time, so the compiler will optimise away
        # `write_debug_data_to_binary()` because it knows it is a no-op.
        return nothing
    end

    @begin_serial_region()

    # write_all_dfns_data_to_binary() expects `scratch` to be a Vector of length
    # n_rk_stages+1.
    scratch = [this_scratch for i ∈ 1:t_params.n_rk_stages+1]
    if this_scratch_electron !== nothing
        scratch_electron = [this_scratch_electron for i ∈ 1:t_params.electron.n_rk_stages+1]
    else
        scratch_electron = nothing
    end

    write_all_dfns_data_to_binary(scratch, scratch_electron, moments, fields,
                                  composition.n_ion_species,
                                  composition.n_neutral_species, t_params.debug_io,
                                  nothing, 0.0, t_params, (), r, z, vperp, vpa, vzeta, vr,
                                  vz; is_debug=true, label=label, istage=istage)

    # This call shouldn't be necessary, as the next @begin_*_region() call following the
    # return from this functions should synchronize, but synchronize here anyway just to
    # be on the safe side - this will only affect runs with debug_io=true set, so
    # performance is not a concern.
    @_block_synchronize()

    return nothing
end

"""
write time-dependent distribution function data for ions to the binary output file

Note: should only be called from within a function that (re-)opens the output file.
"""
function write_ion_dfns_data_to_binary(scratch, t_params, n_ion_species,
                                       io_dfns::io_dfns_info, t_idx, r, z, vperp, vpa)
    @serial_region begin
        # Only read/write from first process in each 'block'

        parallel_io = io_dfns.io_input.parallel_io

        append_to_dynamic_var(io_dfns.f, scratch[t_params.n_rk_stages+1].pdf, t_idx,
                              parallel_io, vpa, vperp, z, r, n_ion_species)
        # If options were not set to select the following outputs, then the io variables
        # will be `nothing` and nothing will be written.
        append_to_dynamic_var(io_dfns.f_loworder, scratch[2].pdf, t_idx,
                              parallel_io, vpa, vperp, z, r, n_ion_species)
        append_to_dynamic_var(io_dfns.f_start_last_timestep, scratch[1].pdf, t_idx,
                              parallel_io, vpa, vperp, z, r, n_ion_species)
    end
    return nothing
end

"""
write time-dependent distribution function data for electrons to the binary output file

Note: should only be called from within a function that (re-)opens the output file.
"""
function write_electron_dfns_data_to_binary(scratch, scratch_electron, t_params,
                                            io_dfns::Union{io_dfns_info,io_initial_electron_info},
                                            t_idx, r, z, vperp, vpa, ir=nothing)
    if (ir === nothing && block_rank[] == 0) || anyzv_subblock_rank[] == 0
        # Only read/write from first process in each 'block' (for 'initial_electron' I/O)
        # or anyzv subblock (for debug I/O that is written independently for each `ir`).

        parallel_io = io_dfns.io_input.parallel_io

        if io_dfns.f_electron !== nothing
            if t_params.kinetic_electron_solver ∈ (implicit_time_evolving, explicit_time_evolving) || scratch_electron === nothing
                n_rk_stages = t_params.n_rk_stages
                this_scratch = scratch
            elseif t_params.electron === nothing
                # t_params is the t_params for electron timestepping
                n_rk_stages = t_params.n_rk_stages
                this_scratch = scratch_electron
            else
                n_rk_stages = t_params.electron.n_rk_stages
                this_scratch = scratch_electron
            end

            function get_from_ir(f)
                if ir === nothing
                    return f
                else
                    return @view f[:,:,:,ir:ir]
                end
            end

            append_to_dynamic_var(io_dfns.f_electron,
                                  get_from_ir(this_scratch[n_rk_stages+1].pdf_electron),
                                  t_idx, parallel_io, vpa, vperp, z, r)
            # If options were not set to select the following outputs, then the io
            # variables will be `nothing` and nothing will be written.
            append_to_dynamic_var(io_dfns.f_electron_loworder,
                                  get_from_ir(this_scratch[2].pdf_electron),
                                  t_idx, parallel_io, vpa, vperp, z, r)
            append_to_dynamic_var(io_dfns.f_electron_start_last_timestep,
                                  get_from_ir(this_scratch[1].pdf_electron),
                                  t_idx, parallel_io, vpa, vperp, z, r)
        end
    end
    return nothing
end

"""
write time-dependent distribution function data for neutrals to the binary output file

Note: should only be called from within a function that (re-)opens the output file.
"""
function write_neutral_dfns_data_to_binary(scratch, t_params, n_neutral_species,
                                           io_dfns::io_dfns_info, t_idx, r, z, vzeta, vr,
                                           vz)
    @serial_region begin
        # Only read/write from first process in each 'block'

        parallel_io = io_dfns.io_input.parallel_io

        if n_neutral_species > 0
            append_to_dynamic_var(io_dfns.f_neutral,
                                  scratch[t_params.n_rk_stages+1].pdf_neutral, t_idx,
                                  parallel_io, vz, vr, vzeta, z, r, n_neutral_species)
            # If options were not set to select the following outputs, then the io
            # variables will be `nothing` and nothing will be written.
            append_to_dynamic_var(io_dfns.f_neutral_loworder, scratch[2].pdf_neutral,
                                  t_idx, parallel_io, vz, vr, vzeta, z, r,
                                  n_neutral_species)
            append_to_dynamic_var(io_dfns.f_neutral_start_last_timestep,
                                  scratch[1].pdf_neutral, t_idx, parallel_io, vz, vr,
                                  vzeta, z, r, n_neutral_species)
        end
    end
    return nothing
end

"""
    write_electron_state(scratch_electron, moments, t_params, io_initial_electron,
                         t_idx, local_pseudotime, electron_residual, r, z, vperp, vpa;
                         pdf_electron_converged=false)

Write the electron state to an output file.
"""
function write_electron_state(scratch_electron, moments, phi::AbstractMatrix{mk_float},
                              t_params, io_or_file_info_initial_electron, t_idx,
                              local_pseudotime, electron_residual, r, z, vperp, vpa;
                              pdf_electron_converged=false, ir=nothing)

    if (ir === nothing && block_rank[] == 0) || anyzv_subblock_rank[] == 0
        # Only read/write from first process in each 'block' (for 'initial_electron' I/O)
        # or anyzv subblock (for debug I/O that is written independently for each `ir`).

        if isa(io_or_file_info_initial_electron, io_dfns_info)
            io_initial_electron = io_or_file_info_initial_electron
            closefile = false
        else
            io_initial_electron = reopen_initial_electron_io(io_or_file_info_initial_electron, ir)
            closefile = true
        end

        parallel_io = io_initial_electron.io_input.parallel_io

        if ir === nothing
            io_r = r
        else
            # Writing a single r-index in each output file, so create a 'fake'
            # r-coordinate.
            io_r = (local_io_range=1:1, global_io_range=1:1)
        end

        function get_from_ir(x::AbstractVector)
            if ir === nothing
                return x
            else
                return @view x[ir:ir]
            end
        end
        function get_from_ir(x::AbstractMatrix)
            if ir === nothing
                return x
            else
                return @view x[:,ir:ir]
            end
        end

        # add the pseudo-time for this time slice to the hdf5 file
        if ir === nothing
            t = t_params.t[1]
        else
            t = t_params.t[ir]
        end
        append_to_dynamic_var(io_initial_electron.time, t, t_idx, parallel_io)
        append_to_dynamic_var(io_initial_electron.electron_local_pseudotime,
                              local_pseudotime, t_idx, parallel_io)
        append_to_dynamic_var(io_initial_electron.electron_cumulative_pseudotime,
                              get_from_ir(t_params.t), t_idx, parallel_io, io_r)
        append_to_dynamic_var(io_initial_electron.electron_residual, electron_residual,
                              t_idx, parallel_io)

        # Save phi to keep the boundary values that are imposed on the sheath-entrance
        # boundary points by the electron boundary condition.
        append_to_dynamic_var(io_initial_electron.phi, get_from_ir(phi), t_idx,
                              parallel_io, z, io_r)

        write_electron_dfns_data_to_binary(nothing, scratch_electron, t_params,
                                           io_initial_electron, t_idx, io_r, z, vperp,
                                           vpa, ir)

        write_electron_moments_data_to_binary(scratch_electron, moments, t_params,
                                              t_params, io_initial_electron, t_idx, io_r,
                                              z, ir)

        if pdf_electron_converged
            modify_attribute!(io_initial_electron.fid, "pdf_electron_converged",
                              pdf_electron_converged)
        end

        closefile && close(io_initial_electron.fid)
    end

    return nothing
end

"""
close all opened output files
"""
function finish_file_io(ascii_io::Union{ascii_ios,Nothing},
                        binary_moments::Union{io_moments_info,Tuple,NamedTuple,Nothing},
                        binary_dfns::Union{io_dfns_info,Tuple,NamedTuple,Nothing})
    @serial_region begin
        # Only read/write from first process in each 'block'

        if ascii_io !== nothing
            # get the fields in the ascii_ios struct
            ascii_io_fields = fieldnames(typeof(ascii_io))
            for x ∈ ascii_io_fields
                io = getfield(ascii_io, x)
                if io !== nothing
                    close(io)
                end
            end
        end
        if binary_moments !== nothing && !isa(binary_moments, Tuple)
            close(binary_moments.fid)
        end
        if binary_dfns !== nothing && !isa(binary_dfns, Tuple)
            close(binary_dfns.fid)
        end
    end
    return nothing
end

"""
close output files for electron initialization
"""
function finish_electron_io(
        binary_initial_electron::Union{io_initial_electron_info,Tuple,Nothing,Bool,NamedTuple})

    @serial_region begin
        # Only read/write from first process in each 'block'

        if (binary_initial_electron !== nothing && !isa(binary_initial_electron, Tuple)
            && !isa(binary_initial_electron, Bool))

            close(binary_initial_electron.fid)
        end
    end
    return nothing
end

# Include the non-optional implementations of binary I/O functions
include("file_io_hdf5.jl")

"""
"""
@timeit global_timer write_data_to_ascii(pdf, moments, fields, vz, vr, vzeta, vpa, vperp,
                                         z, r, t, n_ion_species, n_neutral_species,
                                         ascii_io::Union{ascii_ios,Nothing}) = begin
    if ascii_io === nothing || ascii_io.moments_ion === nothing
        # ascii I/O is disabled
        return nothing
    end

    if r.n > 1 || vperp.n > 1 || vzeta.n > 1 || vr.n > 1
        error("Ascii I/O is only implemented for 1D1V case")
    end
    if vz.n != vpa.n
        error("ASCII I/O is only implemented when vz.n($(vz.n))==vpa.n($(vpa.n))")
    end
    if n_neutral_species != n_ion_species
        error("ASCII I/O is only implemented when n_neutral_species($(n_neutral_species))==n_ion_species($(n_ion_species))")
    end

    @serial_region begin
        # Only read/write from first process in each 'block'

        @views write_f_ascii(pdf, z, vpa, t, ascii_io.ff)
        write_moments_ion_ascii(moments.ion, z, r, t, n_ion_species, ascii_io.moments_ion)
        write_moments_electron_ascii(moments.electron, z, r, t, ascii_io.moments_electron)
        if n_neutral_species > 0
            @views write_moments_neutral_ascii(moments.neutral, z, r, t, n_neutral_species, ascii_io.moments_neutral)
        end
        write_fields_ascii(fields, z, r, t, ascii_io.fields)
    end
    return nothing
end

"""
write the function f(z,vpa) at this time slice
"""
function write_f_ascii(f, z, vpa, t, ascii_io)
    @serial_region begin
        # Only read/write from first process in each 'block'

        @inbounds begin
            #n_species = size(f,3)
            #for is ∈ 1:n_species
                for i ∈ 1:z.n
                    for j ∈ 1:vpa.n
                        println(ascii_io,"t: ", t, "   z: ", z.grid[i],
                            "  vpa: ", vpa.grid[j], "   fion: ", f.ion.norm[j,1,i,1,1],
                            "   fneutral: ", f.neutral.norm[j,1,1,i,1,1])
                    end
                    println(ascii_io)
                end
                println(ascii_io)
            #end
            #println(ascii_io)
        end
    end
    return nothing
end

"""
write moments of the ion species distribution function f at this time slice
"""
function write_moments_ion_ascii(mom, z, r, t, n_species, ascii_io)
    @serial_region begin
        # Only read/write from first process in each 'block'

        @inbounds begin
            for is ∈ 1:n_species
                for ir ∈ 1:r.n
                    for iz ∈ 1:z.n
                        println(ascii_io,"t: ", t, "   species: ", is, "   r: ", r.grid[ir], "   z: ", z.grid[iz],
                            "  dens: ", mom.dens[iz,ir,is], "   upar: ", mom.upar[iz,ir,is],
                            "   ppar: ", mom.ppar[iz,ir,is], "   qpar: ", mom.qpar[iz,ir,is])
                    end
                end
            end
        end
        println(ascii_io,"")
    end
    return nothing
end

"""
write moments of the ion species distribution function f at this time slice
"""
function write_moments_electron_ascii(mom, z, r, t, ascii_io)
    @serial_region begin
        # Only read/write from first process in each 'block'
    
        @inbounds begin
            for ir ∈ 1:r.n
                for iz ∈ 1:z.n
                    println(ascii_io,"t: ", t, "   r: ", r.grid[ir], "   z: ", z.grid[iz],
                            "  dens: ", mom.dens[iz,ir], "   upar: ", mom.upar[iz,ir],
                            "   ppar: ", mom.ppar[iz,ir], "   qpar: ", mom.qpar[iz,ir])
                end
            end
        end
        println(ascii_io,"")
    end
    return nothing
end

"""
write moments of the neutral species distribution function f_neutral at this time slice
"""
function write_moments_neutral_ascii(mom, z, r, t, n_species, ascii_io)
    @serial_region begin
        # Only read/write from first process in each 'block'

        @inbounds begin
            for is ∈ 1:n_species
                for ir ∈ 1:r.n
                    for iz ∈ 1:z.n
                        println(ascii_io,"t: ", t, "   species: ", is, "   r: ", r.grid[ir], "   z: ", z.grid[iz],
                            "  dens: ", mom.dens[iz,ir,is], "   uz: ", mom.uz[iz,ir,is],
                            "   ur: ", mom.ur[iz,ir,is], "   uzeta: ", mom.uzeta[iz,ir,is],
                            "   pz: ", mom.pz[iz,ir,is])
                    end
                end
            end
        end
        println(ascii_io,"")
    end
    return nothing
end

"""
write electrostatic potential at this time slice
"""
function write_fields_ascii(flds, z, r, t, ascii_io)
    @serial_region begin
        # Only read/write from first process in each 'block'

        @inbounds begin
            for ir ∈ 1:r.n
                for iz ∈ 1:z.n
                    println(ascii_io,"t: ", t, "   r: ", r.grid[ir],"   z: ", z.grid[iz], "  phi: ", flds.phi[iz,ir],
                            " Ez: ", flds.Ez[iz,ir])
                end
            end
        end
        println(ascii_io,"")
    end
    return nothing
end

"""
accepts an option name which has been identified as problematic and returns
an appropriate error message
"""
function input_option_error(option_name, input)
    msg = string("'",input,"'")
    msg = string(msg, " is not a valid ", option_name)
    error(msg)
    return nothing
end

"""
opens an output file with the requested prefix and extension
and returns the corresponding io stream (identifier)
"""
function open_ascii_output_file(prefix, ext)
    str = string(prefix,".",ext)
    return io = open(str,"w")
end

end
