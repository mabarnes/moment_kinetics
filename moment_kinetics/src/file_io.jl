"""
"""
module file_io

export input_option_error
export get_group
export open_output_file, open_ascii_output_file
export setup_file_io, finish_file_io
export write_data_to_ascii

using ..communication
using ..coordinates: coordinate
using ..debugging
using ..input_structs
using ..looping
using ..moment_kinetics_structs: scratch_pdf, em_fields_struct
using ..type_definitions: mk_float, mk_int

using LibGit2
using MPI
using Pkg
using TOML

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
                       Texte3, Texte4, Tconstri, Tconstrn, Tconstre, Tint, Tfailcause,
                       Telectrontime, Telectronint, Telectronfailcause}
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
    # handle for the ion species parallel flow
    parallel_flow::Tmomi
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
    # handle for the electron species parallel flow
    electron_parallel_flow::Tmome
    # handle for the electron species parallel pressure
    electron_parallel_pressure::Tmome
    # handle for the electron species parallel heat flux
    electron_parallel_heat_flux::Tmome
    # handle for the electron species thermal speed
    electron_thermal_speed::Tmome

    # handle for the neutral species density
    density_neutral::Tmomn
    uz_neutral::Tmomn
    pz_neutral::Tmomn
    qz_neutral::Tmomn
    thermal_speed_neutral::Tmomn

    # handles for external source variables
    external_source_amplitude::Texti1
    external_source_density_amplitude::Texti2
    external_source_momentum_amplitude::Texti3
    external_source_pressure_amplitude::Texti4
    external_source_controller_integral::Texti5
    external_source_neutral_amplitude::Textn1
    external_source_neutral_density_amplitude::Textn2
    external_source_neutral_momentum_amplitude::Textn3
    external_source_neutral_pressure_amplitude::Textn4
    external_source_neutral_controller_integral::Textn5
    external_source_electron_amplitude::Texte1
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
    # cumulative number of timestep failures
    failure_counter::Tint
    # cumulative count of which variable caused a timstep failure
    failure_caused_by::Tfailcause
    # cumulative count of which factors limited the timestep at each step
    limit_caused_by::Tfailcause
    # Last successful timestep before most recent timestep failure, used by adaptve
    # timestepping algorithm
    dt_before_last_fail::Ttime
    # cumulative number of electron pseudo-timesteps taken
    electron_step_counter::Telectronint
    # current electron pseudo-timestep size
    electron_dt::Telectrontime
    # cumulative number of electron pseudo-timestep failures
    electron_failure_counter::Telectronint
    # cumulative count of which variable caused a electron pseudo-timstep failure
    electron_failure_caused_by::Telectronfailcause
    # cumulative count of which factors limited the electron pseudo-timestep at each step
    electron_limit_caused_by::Telectronfailcause
    # Last successful timestep before most recent electron pseudo-timestep failure, used
    # by adaptve timestepping algorithm
    electron_dt_before_last_fail::Telectrontime

    # Use parallel I/O?
    parallel_io::Bool
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
    # handle for the electron distribution function variable
    f_electron::Tfe
    # handle for the neutral species distribution function variable
    f_neutral::Tfn

    # Use parallel I/O?
    parallel_io::Bool

    # Handles for moment variables
    io_moments::Tmoments
end

"""
structure containing the data/metadata needed for binary file i/o
for electron initialization
"""
struct io_initial_electron_info{Tfile, Ttime, Tfe, Tmom, Texte1, Texte2, Texte3, Texte4,
                                Tconstr, Telectrontime, Telectronint, Telectronfailcause}
    # file identifier for the binary file to which data is written
    fid::Tfile
    # handle for the pseudotime variable
    pseudotime::Ttime
    # handle for the electron distribution function variable
    f_electron::Tfe
    # handle for the electron density variable
    electron_density::Tmom
    # handle for the electron parallel flow variable
    electron_parallel_flow::Tmom
    # handle for the electron parallel pressure variable
    electron_parallel_pressure::Tmom
    # handle for the electron parallel heat flux variable
    electron_parallel_heat_flux::Tmom
    # handle for the electron thermal speed variable
    electron_thermal_speed::Tmom
    # handles for external source terms
    external_source_electron_amplitude::Texte1
    external_source_electron_density_amplitude::Texte2
    external_source_electron_momentum_amplitude::Texte3
    external_source_electron_pressure_amplitude::Texte4
    # handles for constraint coefficients
    electron_constraints_A_coefficient::Tconstr
    electron_constraints_B_coefficient::Tconstr
    electron_constraints_C_coefficient::Tconstr
    # cumulative number of electron pseudo-timesteps taken
    electron_step_counter::Telectronint
    # current electron pseudo-timestep size
    electron_dt::Telectrontime
    # cumulative number of electron pseudo-timestep failures
    electron_failure_counter::Telectronint
    # cumulative count of which variable caused a electron pseudo-timstep failure
    electron_failure_caused_by::Telectronfailcause
    # cumulative count of which factors limited the electron pseudo-timestep at each step
    electron_limit_caused_by::Telectronfailcause
    # Last successful timestep before most recent electron pseudo-timestep failure, used
    # by adaptve timestepping algorithm
    electron_dt_before_last_fail::Telectrontime

    # Use parallel I/O?
    parallel_io::Bool
end

"""
    io_has_parallel(Val(binary_format))

Test if the backend supports parallel I/O.

`binary_format` should be one of the values of the `binary_format_type` enum
"""
function io_has_parallel end

function io_has_implementation(binary_format::binary_format_type)
    if binary_format == hdf5
        return true
    elseif binary_format == netcdf
        netcdf_ext = Base.get_extension(@__MODULE__, :file_io_netcdf)
        return netcdf_ext !== nothing
    else
        error("Unrecognised binary format $binary_format")
    end
end

"""
    check_io_implementation(binary_format)

Check that an implementation is available for `binary_format`, raising an error if not.
"""
function check_io_implementation(binary_format)
    if !io_has_implementation(binary_format)
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
function setup_file_io(io_input, boundary_distributions, vz, vr, vzeta, vpa, vperp, z, r,
                       composition, collisions, evolve_density, evolve_upar, evolve_ppar,
                       external_source_settings, input_dict, restart_time_index,
                       previous_runs_info, time_for_setup)
    begin_serial_region()
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

        run_id = io_input.run_id

        io_moments = setup_moments_io(out_prefix, io_input.binary_format, vz, vr, vzeta,
                                      vpa, vperp, r, z, composition, collisions,
                                      evolve_density, evolve_upar, evolve_ppar,
                                      external_source_settings, input_dict,
                                      io_input.parallel_io, comm_inter_block[], run_id,
                                      restart_time_index, previous_runs_info,
                                      time_for_setup)
        io_dfns = setup_dfns_io(out_prefix, io_input.binary_format,
                                boundary_distributions, r, z, vperp, vpa, vzeta, vr, vz,
                                composition, collisions, evolve_density, evolve_upar,
                                evolve_ppar, external_source_settings, input_dict,
                                io_input.parallel_io, comm_inter_block[], run_id,
                                restart_time_index, previous_runs_info, time_for_setup)

        return ascii, io_moments, io_dfns
    end
    # For other processes in the block, return (nothing, nothing, nothing)
    return nothing, nothing, nothing
end

"""
open output file to save the initial electron pressure and distribution function
"""
function setup_electron_io(io_input, vpa, vperp, z, r, composition, collisions,
                           evolve_density, evolve_upar, evolve_ppar,
                           external_source_settings, input_dict, restart_time_index,
                           previous_runs_info, prefix_label)
    begin_serial_region()
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
        fid, file_info = open_output_file(electrons_prefix, io_input.binary_format,
                                          parallel_io, io_comm)

        # write a header to the output file
        add_attribute!(fid, "file_info",
                       "Output initial electron state from the moment_kinetics code")

        # write some overview information to the output file
        write_overview!(fid, composition, collisions, parallel_io, evolve_density,
                        evolve_upar, evolve_ppar, -1.0)

        # write provenance tracking information to the output file
        write_provenance_tracking_info!(fid, parallel_io, run_id, restart_time_index,
                                        input_dict, previous_runs_info)

        # write the input settings
        write_input!(fid, input_dict, parallel_io)

        ### define coordinate dimensions ###
        define_io_coordinates!(fid, nothing, nothing, nothing, vpa, vperp, z, r,
                               parallel_io)

        ### create variables for time-dependent quantities ###
        dynamic = create_io_group(fid, "dynamic_data", description="time evolving variables")
        io_pseudotime = create_dynamic_variable!(dynamic, "time", mk_float; parallel_io=parallel_io,
                                                 description="pseudotime used for electron initialization")
        io_f_electron = create_dynamic_variable!(dynamic, "f_electron", mk_float, vpa,
                                                 vperp, z, r;
                                                 parallel_io=parallel_io,
                                                 description="electron distribution function")

        io_electron_density, io_electron_upar, io_electron_ppar, io_electron_qpar,
        io_electron_vth, external_source_electron_amplitude,
        external_source_electron_density_amplitude,
        external_source_electron_momentum_amplitude,
        external_source_electron_pressure_amplitude,
        electron_constraints_A_coefficient, electron_constraints_B_coefficient,
        electron_constraints_C_coefficient, io_electron_step_counter, io_electron_dt,
        io_electron_failure_counter, io_electron_failure_caused_by,
        io_electron_limit_caused_by, io_electron_dt_before_last_fail =
            define_dynamic_electron_moment_variables!(fid, r, z, parallel_io,
                                                      external_source_settings,
                                                      evolve_density, evolve_upar,
                                                      evolve_ppar, kinetic_electrons)

        close(fid)

        return file_info
    end
    # For other processes in the block, return nothing
    return nothing
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

    return (filename, io_input.parallel_io, comm_inter_block[])
end

"""
Reopen an existing initial electron output file to append more data
"""
function reopen_initial_electron_io(file_info)
    @serial_region begin
        filename, parallel_io, io_comm = file_info
        fid = reopen_output_file(filename, parallel_io, io_comm)
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
                                        getvar("electron_density"),
                                        getvar("electron_parallel_flow"),
                                        getvar("electron_parallel_pressure"),
                                        getvar("electron_parallel_heat_flux"),
                                        getvar("electron_thermal_speed"),
                                        getvar("external_source_electron_amplitude"),
                                        getvar("external_source_electron_density_amplitude"),
                                        getvar("external_source_electron_momentum_amplitude"),
                                        getvar("external_source_electron_pressure_amplitude"),
                                        getvar("electron_constraints_A_coefficient"),
                                        getvar("electron_constraints_B_coefficient"),
                                        getvar("electron_constraints_C_coefficient"),
                                        getvar("electron_step_counter"),
                                        getvar("electron_dt"),
                                        getvar("electron_failure_counter"),
                                        getvar("electron_failure_caused_by"),
                                        getvar("electron_limit_caused_by"),
                                        getvar("electron_dt_before_last_fail"),
                                        parallel_io)
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
    write_single_value!(file_or_group, name, value; description=nothing, units=nothing)

Write a single variable to a file or group. If a description or units are passed, add as
attributes of the variable.
"""
function write_single_value! end

"""
write some overview information for the simulation to the binary file
"""
function write_overview!(fid, composition, collisions, parallel_io, evolve_density,
                         evolve_upar, evolve_ppar, time_for_setup)
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
                            collisions.charge_exchange, parallel_io=parallel_io,
                            description="quantity related to the charge exchange frequency")
        write_single_value!(overview, "ionization_frequency", collisions.ionization,
                            parallel_io=parallel_io,
                            description="quantity related to the ionization frequency")
        write_single_value!(overview, "evolve_density", evolve_density,
                            parallel_io=parallel_io,
                            description="is density evolved separately from the distribution function?")
        write_single_value!(overview, "evolve_upar", evolve_upar,
                            parallel_io=parallel_io,
                            description="is parallel flow evolved separately from the distribution function?")
        write_single_value!(overview, "evolve_ppar", evolve_ppar,
                            parallel_io=parallel_io,
                            description="is parallel pressure evolved separately from the distribution function?")
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
        TOML.print(io_buffer, input_dict)
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
Write the distributions that may be used for boundary conditions to the output file
"""
function write_boundary_distributions!(fid, boundary_distributions, parallel_io,
                                       composition, z, vperp, vpa, vzeta, vr, vz)
    @serial_region begin
        boundary_distributions_io = create_io_group(fid, "boundary_distributions")

        write_single_value!(boundary_distributions_io, "pdf_rboundary_ion_left",
            boundary_distributions.pdf_rboundary_ion[:,:,:,1,:], vpa, vperp, z,
            parallel_io=parallel_io, n_ion_species=composition.n_ion_species,
            description="Initial ion-particle pdf at left radial boundary")
        write_single_value!(boundary_distributions_io, "pdf_rboundary_ion_right",
            boundary_distributions.pdf_rboundary_ion[:,:,:,2,:], vpa, vperp, z,
            parallel_io=parallel_io, n_ion_species=composition.n_ion_species,
            description="Initial ion-particle pdf at right radial boundary")
        write_single_value!(boundary_distributions_io, "pdf_rboundary_neutral_left",
            boundary_distributions.pdf_rboundary_neutral[:,:,:,:,1,:], vz, vr, vzeta, z,
            parallel_io=parallel_io, n_neutral_species=composition.n_neutral_species,
            description="Initial neutral-particle pdf at left radial boundary")
        write_single_value!(boundary_distributions_io, "pdf_rboundary_neutral_right",
            boundary_distributions.pdf_rboundary_neutral[:,:,:,:,2,:], vz, vr, vzeta, z,
            parallel_io=parallel_io, n_neutral_species=composition.n_neutral_species,
            description="Initial neutral-particle pdf at right radial boundary")
    end
    return nothing
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
        write_single_value!(group, "fd_option", coord.fd_option; parallel_io=parallel_io,
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
    create_dynamic_variable!(file_or_group, name, type, coords::coordinate...;
                             n_ion_species=1, n_neutral_species=1,
                             diagnostic_var_size=nothing, description=nothing,
                             units=nothing)

Create a time-evolving variable in `file_or_group` named `name` of type `type`. `coords`
are the coordinates corresponding to the dimensions of the array, in the order of the
array dimensions. The species dimension does not have a `coordinate`, so the number of
species is passed as `nspecies`. A description and/or units can be added with the keyword
arguments.

If a Tuple giving an array size is passed to `diagnostic_var_size`, a 'diagnostic'
variable is created - i.e. one that does not depend on the coordinates, so is assumed to
be the same on all processes and only needs to be written from the root process (for each
output file).
"""
function create_dynamic_variable! end

"""
define dynamic (time-evolving) moment variables for writing to the hdf5 file
"""
function define_dynamic_moment_variables!(fid, n_ion_species, n_neutral_species,
                                          r::coordinate, z::coordinate, parallel_io,
                                          external_source_settings, evolve_density,
                                          evolve_upar, evolve_ppar, electron_physics)
    @serial_region begin
        dynamic = create_io_group(fid, "dynamic_data", description="time evolving variables")

        io_time = create_dynamic_variable!(dynamic, "time", mk_float; parallel_io=parallel_io,
                                           description="simulation time")

        io_phi, io_Er, io_Ez =
            define_dynamic_em_field_variables!(fid, r, z, parallel_io)

        io_density, io_upar, io_ppar, io_pperp, io_qpar, io_vth, io_dSdt,
        external_source_amplitude, external_source_density_amplitude,
        external_source_momentum_amplitude, external_source_pressure_amplitude,
        external_source_controller_integral, io_chodura_lower, io_chodura_upper,
        ion_constraints_A_coefficient, ion_constraints_B_coefficient,
        ion_constraints_C_coefficient =
            define_dynamic_ion_moment_variables!(fid, n_ion_species, r, z, parallel_io,
                                                 external_source_settings, evolve_density,
                                                 evolve_upar, evolve_ppar)

        io_electron_density, io_electron_upar, io_electron_ppar, io_electron_qpar,
        io_electron_vth, external_source_electron_amplitude,
        external_source_electron_density_amplitude,
        external_source_electron_momentum_amplitude,
        external_source_electron_pressure_amplitude,
        electron_constraints_A_coefficient, electron_constraints_B_coefficient,
        electron_constraints_C_coefficient, io_electron_step_counter, io_electron_dt,
        io_electron_failure_counter, io_electron_failure_caused_by,
        io_electron_limit_caused_by, io_electron_dt_before_last_fail =
            define_dynamic_electron_moment_variables!(fid, r, z, parallel_io,
                                                      external_source_settings,
                                                      evolve_density, evolve_upar,
                                                      evolve_ppar, electron_physics)

        io_density_neutral, io_uz_neutral, io_pz_neutral, io_qz_neutral,
        io_thermal_speed_neutral, external_source_neutral_amplitude,
        external_source_neutral_density_amplitude,
        external_source_neutral_momentum_amplitude,
        external_source_neutral_pressure_amplitude,
        external_source_neutral_controller_integral, neutral_constraints_A_coefficient,
        neutral_constraints_B_coefficient, neutral_constraints_C_coefficient =
            define_dynamic_neutral_moment_variables!(fid, n_neutral_species, r, z,
                                                     parallel_io,
                                                     external_source_settings,
                                                     evolve_density, evolve_upar,
                                                     evolve_ppar)

        io_time_for_run = create_dynamic_variable!(
            dynamic, "time_for_run", mk_float; parallel_io=parallel_io,
            description="cumulative wall clock time for run (excluding setup)",
            units="minutes")

        io_step_counter = create_dynamic_variable!(
            dynamic, "step_counter", mk_int; parallel_io=parallel_io,
            description="cumulative number of timesteps for the run")

        io_dt = create_dynamic_variable!(
            dynamic, "dt", mk_float; parallel_io=parallel_io,
            description="current timestep size")

        io_failure_counter = create_dynamic_variable!(
            dynamic, "failure_counter", mk_int; parallel_io=parallel_io,
            description="cumulative number of timestep failures for the run")

        n_failure_vars = 1 + evolve_density + evolve_upar + evolve_ppar
        if n_neutral_species > 0
            n_failure_vars *= 2
        end
        if electron_physics ∈ (braginskii_fluid, kinetic_electrons)
            n_failure_vars += 1
        end
        io_failure_caused_by = create_dynamic_variable!(
            dynamic, "failure_caused_by", mk_int; diagnostic_var_size=n_failure_vars,
            parallel_io=parallel_io,
            description="cumulative count of how many times each variable caused a "
                        * "timestep failure for the run")
        n_limit_vars = 5 + 2
        if n_neutral_species > 0
            n_limit_vars += 2
        end
        io_limit_caused_by = create_dynamic_variable!(
            dynamic, "limit_caused_by", mk_int; diagnostic_var_size=n_limit_vars,
            parallel_io=parallel_io,
            description="cumulative count of how many times each factor limited the "
                        * "timestep for the run")

        io_dt_before_last_fail = create_dynamic_variable!(
            dynamic, "dt_before_last_fail", mk_float; parallel_io=parallel_io,
            description="Last successful timestep before most recent timestep failure, "
                        * "used by adaptve timestepping algorithm")

        return io_moments_info(fid, io_time, io_phi, io_Er, io_Ez, io_density, io_upar,
                               io_ppar, io_pperp, io_qpar, io_vth, io_dSdt, io_chodura_lower, io_chodura_upper,
                               io_electron_density, io_electron_upar, io_electron_ppar, io_electron_qpar, io_electron_vth,
                               io_density_neutral, io_uz_neutral,
                               io_pz_neutral, io_qz_neutral, io_thermal_speed_neutral,
                               external_source_amplitude,
                               external_source_density_amplitude,
                               external_source_momentum_amplitude,
                               external_source_pressure_amplitude,
                               external_source_controller_integral,
                               external_source_neutral_amplitude,
                               external_source_neutral_density_amplitude,
                               external_source_neutral_momentum_amplitude,
                               external_source_neutral_pressure_amplitude,
                               external_source_neutral_controller_integral,
                               external_source_electron_amplitude,
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
                               io_time_for_run, io_step_counter, io_dt,
                               io_failure_counter, io_failure_caused_by,
                               io_limit_caused_by, io_dt_before_last_fail,
                               io_electron_step_counter, io_electron_dt,
                               io_electron_failure_counter, io_electron_failure_caused_by,
                               io_electron_limit_caused_by,
                               io_electron_dt_before_last_fail, parallel_io)
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
        evolve_ppar)

    dynamic = get_group(fid, "dynamic_data")

    # io_density is the handle for the ion particle density
    io_density = create_dynamic_variable!(dynamic, "density", mk_float, z, r;
                                          n_ion_species=n_ion_species,
                                          parallel_io=parallel_io,
                                          description="ion species density",
                                          units="n_ref")

    # io_upar is the handle for the ion parallel flow density
    io_upar = create_dynamic_variable!(dynamic, "parallel_flow", mk_float, z, r;
                                       n_ion_species=n_ion_species,
                                       parallel_io=parallel_io,
                                       description="ion species parallel flow",
                                       units="c_ref = sqrt(2*T_ref/mi)")

    # io_ppar is the handle for the ion parallel pressure
    io_ppar = create_dynamic_variable!(dynamic, "parallel_pressure", mk_float, z, r;
                                       n_ion_species=n_ion_species,
                                       parallel_io=parallel_io,
                                       description="ion species parallel pressure",
                                       units="n_ref*T_ref")

    # io_pperp is the handle for the ion parallel pressure
    io_pperp = create_dynamic_variable!(dynamic, "perpendicular_pressure", mk_float, z, r;
                                        n_ion_species=n_ion_species,
                                        parallel_io=parallel_io,
                                        description="ion species perpendicular pressure",
                                        units="n_ref*T_ref")

    # io_qpar is the handle for the ion parallel heat flux
    io_qpar = create_dynamic_variable!(dynamic, "parallel_heat_flux", mk_float, z, r;
                                       n_ion_species=n_ion_species,
                                       parallel_io=parallel_io,
                                       description="ion species parallel heat flux",
                                       units="n_ref*T_ref*c_ref")

    # io_vth is the handle for the ion thermal speed
    io_vth = create_dynamic_variable!(dynamic, "thermal_speed", mk_float, z, r;
                                      n_ion_species=n_ion_species,
                                      parallel_io=parallel_io,
                                      description="ion species thermal speed",
                                      units="c_ref")

    # io_dSdt is the handle for the entropy production (due to collisions)
    io_dSdt = create_dynamic_variable!(dynamic, "entropy_production", mk_float, z, r;
                                      n_ion_species=n_ion_species,
                                      parallel_io=parallel_io,
                                      description="ion species entropy production",
                                      units="")

    ion_source_settings = external_source_settings.ion
    if ion_source_settings.active
        external_source_amplitude = create_dynamic_variable!(
            dynamic, "external_source_amplitude", mk_float, z, r;
            parallel_io=parallel_io, description="Amplitude of the external source for ions",
            units="n_ref/c_ref^3*c_ref/L_ref")
        if evolve_density
            external_source_density_amplitude = create_dynamic_variable!(
                dynamic, "external_source_density_amplitude", mk_float, z, r;
                parallel_io=parallel_io, description="Amplitude of the external density source for ions",
                units="n_ref*c_ref/L_ref")
        else
            external_source_density_amplitude = nothing
        end
        if evolve_upar
            external_source_momentum_amplitude = create_dynamic_variable!(
                dynamic, "external_source_momentum_amplitude", mk_float, z, r;
                parallel_io=parallel_io, description="Amplitude of the external momentum source for ions",
                units="m_ref*n_ref*c_ref*c_ref/L_ref")
        else
            external_source_momentum_amplitude = nothing
        end
        if evolve_ppar
            external_source_pressure_amplitude = create_dynamic_variable!(
                dynamic, "external_source_pressure_amplitude", mk_float, z, r;
                parallel_io=parallel_io, description="Amplitude of the external pressure source for ions",
                units="m_ref*n_ref*c_ref^2*c_ref/L_ref")
        else
            external_source_pressure_amplitude = nothing
        end
        if ion_source_settings.PI_density_controller_I != 0.0 &&
                ion_source_settings.source_type ∈ ("density_profile_control", "density_midpoint_control")
            if ion_source_settings.source_type == "density_profile_control"
                external_source_controller_integral = create_dynamic_variable!(
                    dynamic, "external_source_controller_integral", mk_float, z, r;
                    parallel_io=parallel_io,
                    description="Integral term for the PID controller of the external source for ions")
            else
                external_source_controller_integral = create_dynamic_variable!(
                    dynamic, "external_source_controller_integral", mk_float;
                    parallel_io=parallel_io,
                    description="Integral term for the PID controller of the external source for ions")
            end
        else
            external_source_controller_integral = nothing
        end
    else
        external_source_amplitude = nothing
        external_source_density_amplitude = nothing
        external_source_momentum_amplitude = nothing
        external_source_pressure_amplitude = nothing
        external_source_controller_integral = nothing
    end

    if parallel_io || z.irank == 0
        # io_chodura_lower is the handle for the ion thermal speed
        io_chodura_lower = create_dynamic_variable!(dynamic, "chodura_integral_lower", mk_float, r;
                                          n_ion_species=n_ion_species,
                                          parallel_io=parallel_io,
                                          description="Generalised Chodura integral lower sheath entrance",
                                          units="c_ref")
    else
        io_chodura_lower = nothing
    end
    if parallel_io || z.irank == z.nrank - 1
        # io_chodura_upper is the handle for the ion thermal speed
        io_chodura_upper = create_dynamic_variable!(dynamic, "chodura_integral_upper", mk_float, r;
                                          n_ion_species=n_ion_species,
                                          parallel_io=parallel_io,
                                          description="Generalised Chodura integral upper sheath entrance",
                                          units="c_ref")
    else
        io_chodura_upper = nothing
    end

    if evolve_density || evolve_upar || evolve_ppar
        ion_constraints_A_coefficient =
            create_dynamic_variable!(dynamic, "ion_constraints_A_coefficient", mk_float, z, r;
                                   n_ion_species=n_ion_species,
                                   parallel_io=parallel_io,
                                   description="'A' coefficient enforcing density constraint for ions")
        ion_constraints_B_coefficient =
            create_dynamic_variable!(dynamic, "ion_constraints_B_coefficient", mk_float, z, r;
                                   n_ion_species=n_ion_species,
                                   parallel_io=parallel_io,
                                   description="'B' coefficient enforcing flow constraint for ions")
        ion_constraints_C_coefficient =
            create_dynamic_variable!(dynamic, "ion_constraints_C_coefficient", mk_float, z, r;
                                   n_ion_species=n_ion_species,
                                   parallel_io=parallel_io,
                                   description="'C' coefficient enforcing pressure constraint for ions")
    else
           ion_constraints_A_coefficient = nothing
           ion_constraints_B_coefficient = nothing
           ion_constraints_C_coefficient = nothing
    end

    return io_density, io_upar, io_ppar, io_pperp, io_qpar, io_vth, io_dSdt,
           external_source_amplitude, external_source_density_amplitude,
           external_source_momentum_amplitude, external_source_pressure_amplitude,
           external_source_controller_integral, io_chodura_lower, io_chodura_upper,
           ion_constraints_A_coefficient, ion_constraints_B_coefficient,
           ion_constraints_C_coefficient
end

"""
define dynamic (time-evolving) electron moment variables for writing to the hdf5 file
"""
function define_dynamic_electron_moment_variables!(fid, r::coordinate, z::coordinate,
        parallel_io, external_source_settings, evolve_density, evolve_upar, evolve_ppar,
        electron_physics)

    dynamic = get_group(fid, "dynamic_data")

    # io_density is the handle for the ion particle density
    io_electron_density = create_dynamic_variable!(dynamic, "electron_density", mk_float, z, r;
                                          parallel_io=parallel_io,
                                          description="electron species density",
                                          units="n_ref")

    # io_electron_upar is the handle for the electron parallel flow density
    io_electron_upar = create_dynamic_variable!(dynamic, "electron_parallel_flow", mk_float, z, r;
                                       parallel_io=parallel_io,
                                       description="electron species parallel flow",
                                       units="c_ref = sqrt(2*T_ref/mi)")

    # io_electron_ppar is the handle for the electron parallel pressure
    io_electron_ppar = create_dynamic_variable!(dynamic, "electron_parallel_pressure", mk_float, z, r;
                                       parallel_io=parallel_io,
                                       description="electron species parallel pressure",
                                       units="n_ref*T_ref")

    # io_electron_qpar is the handle for the electron parallel heat flux
    io_electron_qpar = create_dynamic_variable!(dynamic, "electron_parallel_heat_flux", mk_float, z, r;
                                       parallel_io=parallel_io,
                                       description="electron species parallel heat flux",
                                       units="n_ref*T_ref*c_ref")

    # io_electron_vth is the handle for the electron thermal speed
    io_electron_vth = create_dynamic_variable!(dynamic, "electron_thermal_speed", mk_float, z, r;
                                      parallel_io=parallel_io,
                                      description="electron species thermal speed",
                                      units="c_ref")

    electron_source_settings = external_source_settings.electron
    if electron_source_settings.active
        external_source_electron_amplitude = create_dynamic_variable!(
            dynamic, "external_source_electron_amplitude", mk_float, z, r;
            parallel_io=parallel_io, description="Amplitude of the external source for electrons",
            units="n_ref/c_ref^3*c_ref/L_ref")
        external_source_electron_density_amplitude = create_dynamic_variable!(
            dynamic, "external_source_electron_density_amplitude", mk_float, z, r;
            parallel_io=parallel_io, description="Amplitude of the external density source for electrons",
            units="n_ref*c_ref/L_ref")
        external_source_electron_momentum_amplitude = create_dynamic_variable!(
            dynamic, "external_source_electron_momentum_amplitude", mk_float, z, r;
            parallel_io=parallel_io, description="Amplitude of the external momentum source for electrons",
            units="m_ref*n_ref*c_ref*c_ref/L_ref")
        external_source_electron_pressure_amplitude = create_dynamic_variable!(
            dynamic, "external_source_electron_pressure_amplitude", mk_float, z, r;
            parallel_io=parallel_io, description="Amplitude of the external pressure source for electrons",
            units="m_ref*n_ref*c_ref^2*c_ref/L_ref")
    else
        external_source_electron_amplitude = nothing
        external_source_electron_density_amplitude = nothing
        external_source_electron_momentum_amplitude = nothing
        external_source_electron_pressure_amplitude = nothing
    end

    electron_constraints_A_coefficient =
        create_dynamic_variable!(dynamic, "electron_constraints_A_coefficient", mk_float, z, r;
                               parallel_io=parallel_io,
                               description="'A' coefficient enforcing density constraint for electrons")
    electron_constraints_B_coefficient =
        create_dynamic_variable!(dynamic, "electron_constraints_B_coefficient", mk_float, z, r;
                               parallel_io=parallel_io,
                               description="'B' coefficient enforcing flow constraint for electrons")
    electron_constraints_C_coefficient =
        create_dynamic_variable!(dynamic, "electron_constraints_C_coefficient", mk_float, z, r;
                               parallel_io=parallel_io,
                               description="'C' coefficient enforcing pressure constraint for electrons")

    if electron_physics == kinetic_electrons
        io_electron_step_counter = create_dynamic_variable!(
            dynamic, "electron_step_counter", mk_int; parallel_io=parallel_io,
            description="cumulative number of electron pseudo-timesteps for the run")

        io_electron_dt = create_dynamic_variable!(
            dynamic, "electron_dt", mk_float; parallel_io=parallel_io,
            description="current electron pseudo-timestep size")

        io_electron_failure_counter = create_dynamic_variable!(
            dynamic, "electron_failure_counter", mk_int; parallel_io=parallel_io,
            description="cumulative number of electron pseudo-timestep failures for the run")

        n_failure_vars = 1 + 1
        io_electron_failure_caused_by = create_dynamic_variable!(
            dynamic, "electron_failure_caused_by", mk_int;
            diagnostic_var_size=n_failure_vars, parallel_io=parallel_io,
            description="cumulative count of how many times each variable caused an "
                        * "electron pseudo-timestep failure for the run")

        n_limit_vars = 5 + 2
        io_electron_limit_caused_by = create_dynamic_variable!(
            dynamic, "electron_limit_caused_by", mk_int; diagnostic_var_size=n_limit_vars,
            parallel_io=parallel_io,
            description="cumulative count of how many times each factor limited the "
                        * "electron pseudo-timestep for the run")

        io_electron_dt_before_last_fail = create_dynamic_variable!(
            dynamic, "electron_dt_before_last_fail", mk_float; parallel_io=parallel_io,
            description="Last successful electron pseudo-timestep before most recent "
                        * "electron pseudo-timestep failure, used by adaptve "
                        * "timestepping algorithm")
    else
        io_electron_step_counter = nothing
        io_electron_dt = nothing
        io_electron_failure_counter = nothing
        io_electron_failure_caused_by = nothing
        io_electron_limit_caused_by = nothing
        io_electron_dt_before_last_fail = nothing
    end

    return io_electron_density, io_electron_upar, io_electron_ppar, io_electron_qpar,
           io_electron_vth, external_source_electron_amplitude,
           external_source_electron_density_amplitude,
           external_source_electron_momentum_amplitude,
           external_source_electron_pressure_amplitude,
           electron_constraints_A_coefficient, electron_constraints_B_coefficient,
           electron_constraints_C_coefficient, io_electron_step_counter, io_electron_dt,
           io_electron_failure_counter, io_electron_failure_caused_by,
           io_electron_limit_caused_by, io_electron_dt_before_last_fail
end

"""
define dynamic (time-evolving) neutral moment variables for writing to the hdf5 file
"""
function define_dynamic_neutral_moment_variables!(fid, n_neutral_species, r::coordinate,
        z::coordinate, parallel_io, external_source_settings, evolve_density, evolve_upar,
        evolve_ppar)

    dynamic = get_group(fid, "dynamic_data")

    # io_density_neutral is the handle for the neutral particle density
    io_density_neutral = create_dynamic_variable!(dynamic, "density_neutral", mk_float, z, r;
                                                  n_neutral_species=n_neutral_species,
                                                  parallel_io=parallel_io,
                                                  description="neutral species density",
                                                  units="n_ref")

    # io_uz_neutral is the handle for the neutral z momentum density
    io_uz_neutral = create_dynamic_variable!(dynamic, "uz_neutral", mk_float, z, r;
                                             n_neutral_species=n_neutral_species,
                                             parallel_io=parallel_io,
                                             description="neutral species mean z velocity",
                                             units="c_ref = sqrt(2*T_ref/mi)")

    # io_pz_neutral is the handle for the neutral species zz pressure
    io_pz_neutral = create_dynamic_variable!(dynamic, "pz_neutral", mk_float, z, r;
                                             n_neutral_species=n_neutral_species,
                                             parallel_io=parallel_io,
                                             description="neutral species mean zz pressure",
                                             units="n_ref*T_ref")

    # io_qz_neutral is the handle for the neutral z heat flux
    io_qz_neutral = create_dynamic_variable!(dynamic, "qz_neutral", mk_float, z, r;
                                             n_neutral_species=n_neutral_species,
                                             parallel_io=parallel_io,
                                             description="neutral species z heat flux",
                                             units="n_ref*T_ref*c_ref")

    # io_thermal_speed_neutral is the handle for the neutral thermal speed
    io_thermal_speed_neutral = create_dynamic_variable!(
        dynamic, "thermal_speed_neutral", mk_float, z, r;
        n_neutral_species=n_neutral_species,
        parallel_io=parallel_io, description="neutral species thermal speed",
        units="c_ref")

    neutral_source_settings = external_source_settings.neutral
    if n_neutral_species > 0 && neutral_source_settings.active
        external_source_neutral_amplitude = create_dynamic_variable!(
            dynamic, "external_source_neutral_amplitude", mk_float, z, r;
            parallel_io=parallel_io, description="Amplitude of the external source for neutrals",
            units="n_ref/c_ref^3*c_ref/L_ref")
        if evolve_density
            external_source_neutral_density_amplitude = create_dynamic_variable!(
                dynamic, "external_source_neutral_density_amplitude", mk_float, z, r;
                parallel_io=parallel_io, description="Amplitude of the external density source for neutrals",
                units="n_ref*c_ref/L_ref")
        else
            external_source_neutral_density_amplitude = nothing
        end
        if evolve_upar
            external_source_neutral_momentum_amplitude = create_dynamic_variable!(
                dynamic, "external_source_neutral_momentum_amplitude", mk_float, z, r;
                parallel_io=parallel_io, description="Amplitude of the external momentum source for neutrals",
                units="m_ref*n_ref*c_ref*c_ref/L_ref")
        else
            external_source_neutral_momentum_amplitude = nothing
        end
        if evolve_ppar
            external_source_neutral_pressure_amplitude = create_dynamic_variable!(
                dynamic, "external_source_neutral_pressure_amplitude", mk_float, z, r;
                parallel_io=parallel_io, description="Amplitude of the external pressure source for neutrals",
                units="m_ref*n_ref*c_ref^2*c_ref/L_ref")
        else
            external_source_neutral_pressure_amplitude = nothing
        end
        if neutral_source_settings.PI_density_controller_I != 0.0 &&
                neutral_source_settings.source_type ∈ ("density_profile_control", "density_midpoint_control")
            if neutral_source_settings.source_type == "density_profile_control"
                external_source_neutral_controller_integral = create_dynamic_variable!(
                    dynamic, "external_source_neutral_controller_integral", mk_float, z, r;
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
        external_source_neutral_density_amplitude = nothing
        external_source_neutral_momentum_amplitude = nothing
        external_source_neutral_pressure_amplitude = nothing
        external_source_neutral_controller_integral = nothing
    end

    if evolve_density || evolve_upar || evolve_ppar
        neutral_constraints_A_coefficient =
            create_dynamic_variable!(dynamic, "neutral_constraints_A_coefficient", mk_float, z, r;
                                   n_neutral_species=n_neutral_species,
                                   parallel_io=parallel_io,
                                   description="'A' coefficient enforcing density constraint for neutrals")
        neutral_constraints_B_coefficient =
            create_dynamic_variable!(dynamic, "neutral_constraints_B_coefficient", mk_float, z, r;
                                   n_neutral_species=n_neutral_species,
                                   parallel_io=parallel_io,
                                   description="'B' coefficient enforcing flow constraint for neutrals")
        neutral_constraints_C_coefficient =
            create_dynamic_variable!(dynamic, "neutral_constraints_C_coefficient", mk_float, z, r;
                                   n_neutral_species=n_neutral_species,
                                   parallel_io=parallel_io,
                                   description="'C' coefficient enforcing pressure constraint for neutrals")
    else
           neutral_constraints_A_coefficient = nothing
           neutral_constraints_B_coefficient = nothing
           neutral_constraints_C_coefficient = nothing
    end

    return io_density_neutral, io_uz_neutral, io_pz_neutral, io_qz_neutral,
           io_thermal_speed_neutral, external_source_neutral_amplitude,
           external_source_neutral_density_amplitude,
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
                                       parallel_io, external_source_settings,
                                       evolve_density, evolve_upar, evolve_ppar)

    @serial_region begin
        io_moments = define_dynamic_moment_variables!(fid, composition.n_ion_species,
                                                      composition.n_neutral_species, r, z,
                                                      parallel_io,
                                                      external_source_settings,
                                                      evolve_density, evolve_upar,
                                                      evolve_ppar,
                                                      composition.electron_physics)

        dynamic = get_group(fid, "dynamic_data")

        # io_f is the handle for the ion pdf
        io_f = create_dynamic_variable!(dynamic, "f", mk_float, vpa, vperp, z, r;
                                        n_ion_species=composition.n_ion_species,
                                        parallel_io=parallel_io,
                                        description="ion species distribution function")

        if composition.electron_physics == kinetic_electrons
            # io_f_electron is the handle for the electron pdf
            io_f_electron = create_dynamic_variable!(dynamic, "f_electron", mk_float, vpa,
                                                     vperp, z, r;
                                                     parallel_io=parallel_io,
                                                     description="electron distribution function")
        else
            io_f_electron = nothing
        end

        # io_f_neutral is the handle for the neutral pdf
        io_f_neutral = create_dynamic_variable!(dynamic, "f_neutral", mk_float, vz, vr, vzeta, z, r;
                                                n_neutral_species=composition.n_neutral_species,
                                                parallel_io=parallel_io,
                                                description="neutral species distribution function")

        return io_dfns_info(fid, io_f, io_f_electron, io_f_neutral, parallel_io, io_moments)
    end

    # For processes other than the root process of each shared-memory group...
    return nothing
end

"""
Add an attribute to a file, group or variable
"""
function add_attribute! end

"""
Low-level function to open a binary output file

Each implementation (HDF5, NetCDF, etc.) defines a method of this function to open a file
of the corresponding type.
"""
function open_output_file_implementation end

"""
Open an output file, selecting the backend based on io_option
"""
function open_output_file(prefix, binary_format, parallel_io, io_comm)
    check_io_implementation(binary_format)

    return open_output_file_implementation(Val(binary_format), prefix, parallel_io,
                                           io_comm)
end

"""
Re-open an existing output file, selecting the backend based on io_option
"""
function reopen_output_file(filename, parallel_io, io_comm)
    prefix, format_string = splitext(filename)
    if format_string == ".h5"
        check_io_implementation(hdf5)
        return open_output_file_implementation(Val(hdf5), prefix, parallel_io, io_comm,
                                               "r+")[1]
    elseif format_string == ".cdf"
        check_io_implementation(netcdf)
        return open_output_file_implementation(Val(netcdf), prefix, parallel_io, io_comm,
                                               "a")[1]
    else
        error("Unsupported I/O format $binary_format")
    end
end

"""
setup file i/o for moment variables
"""
function setup_moments_io(prefix, binary_format, vz, vr, vzeta, vpa, vperp, r, z,
                          composition, collisions, evolve_density, evolve_upar,
                          evolve_ppar, external_source_settings, input_dict, parallel_io,
                          io_comm, run_id, restart_time_index, previous_runs_info,
                          time_for_setup)
    @serial_region begin
        moments_prefix = string(prefix, ".moments")
        if !parallel_io
            moments_prefix *= ".$(iblock_index[])"
        end
        fid, file_info = open_output_file(moments_prefix, binary_format, parallel_io, io_comm)

        # write a header to the output file
        add_attribute!(fid, "file_info", "Output moments data from the moment_kinetics code")

        # write some overview information to the output file
        write_overview!(fid, composition, collisions, parallel_io, evolve_density,
                        evolve_upar, evolve_ppar, time_for_setup)

        # write provenance tracking information to the output file
        write_provenance_tracking_info!(fid, parallel_io, run_id, restart_time_index,
                                        input_dict, previous_runs_info)

        # write the input settings
        write_input!(fid, input_dict, parallel_io)

        ### define coordinate dimensions ###
        define_io_coordinates!(fid, vz, vr, vzeta, vpa, vperp, z, r, parallel_io)

        ### create variables for time-dependent quantities and store them ###
        ### in a struct for later access ###
        io_moments = define_dynamic_moment_variables!(
            fid, composition.n_ion_species, composition.n_neutral_species, r, z,
            parallel_io, external_source_settings, evolve_density, evolve_upar,
            evolve_ppar, composition.electron_physics)

        close(fid)

        return file_info
    end

    # For processes other than the root process of each shared-memory group...
    return nothing
end

"""
Reopen an existing moments output file to append more data
"""
function reopen_moments_io(file_info)
    @serial_region begin
        filename, parallel_io, io_comm = file_info
        fid = reopen_output_file(filename, parallel_io, io_comm)
        dyn = get_group(fid, "dynamic_data")

        variable_list = get_variable_keys(dyn)
        function getvar(name)
            if name ∈ variable_list
                return dyn[name]
            else
                return nothing
            end
        end
        return io_moments_info(fid, getvar("time"), getvar("phi"), getvar("Er"),
                               getvar("Ez"), getvar("density"), getvar("parallel_flow"),
                               getvar("parallel_pressure"), getvar("perpendicular_pressure"),
                               getvar("parallel_heat_flux"),
                               getvar("thermal_speed"), getvar("entropy_production"),
                               getvar("chodura_integral_lower"),
                               getvar("chodura_integral_upper"),
                               getvar("electron_density"),
                               getvar("electron_parallel_flow"),
                               getvar("electron_parallel_pressure"),
                               getvar("electron_parallel_heat_flux"),
                               getvar("electron_thermal_speed"),
                               getvar("density_neutral"), getvar("uz_neutral"),
                               getvar("pz_neutral"), getvar("qz_neutral"),
                               getvar("thermal_speed_neutral"),
                               getvar("external_source_amplitude"),
                               getvar("external_source_density_amplitude"),
                               getvar("external_source_momentum_amplitude"),
                               getvar("external_source_pressure_amplitude"),
                               getvar("external_source_controller_integral"),
                               getvar("external_source_neutral_amplitude"),
                               getvar("external_source_neutral_density_amplitude"),
                               getvar("external_source_neutral_momentum_amplitude"),
                               getvar("external_source_neutral_pressure_amplitude"),
                               getvar("external_source_neutral_controller_integral"),
                               getvar("external_source_electron_amplitude"),
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
                               getvar("time_for_run"), getvar("step_counter"),
                               getvar("dt"), getvar("failure_counter"),
                               getvar("failure_caused_by"), getvar("limit_caused_by"),
                               getvar("dt_before_last_fail"),getvar("electron_step_counter"),
                               getvar("electron_dt"), getvar("electron_failure_counter"),
                               getvar("electron_failure_caused_by"),
                               getvar("electron_limit_caused_by"),
                               getvar("electron_dt_before_last_fail"), parallel_io)
    end

    # For processes other than the root process of each shared-memory group...
    return nothing
end

"""
setup file i/o for distribution function variables
"""
function setup_dfns_io(prefix, binary_format, boundary_distributions, r, z, vperp, vpa,
                       vzeta, vr, vz, composition, collisions, evolve_density,
                       evolve_upar, evolve_ppar, external_source_settings, input_dict,
                       parallel_io, io_comm, run_id, restart_time_index,
                       previous_runs_info, time_for_setup)

    @serial_region begin
        dfns_prefix = string(prefix, ".dfns")
        if !parallel_io
            dfns_prefix *= ".$(iblock_index[])"
        end
        fid, file_info = open_output_file(dfns_prefix, binary_format, parallel_io, io_comm)

        # write a header to the output file
        add_attribute!(fid, "file_info",
                       "Output distribution function data from the moment_kinetics code")

        # write some overview information to the output file
        write_overview!(fid, composition, collisions, parallel_io, evolve_density,
                        evolve_upar, evolve_ppar, time_for_setup)

        # write provenance tracking information to the output file
        write_provenance_tracking_info!(fid, parallel_io, run_id, restart_time_index,
                                        input_dict, previous_runs_info)

        # write the input settings
        write_input!(fid, input_dict, parallel_io)

        # write the distributions that may be used for boundary conditions to the output
        # file
        write_boundary_distributions!(fid, boundary_distributions, parallel_io,
                                      composition, z, vperp, vpa, vzeta, vr, vz)

        ### define coordinate dimensions ###
        define_io_coordinates!(fid, vz, vr, vzeta, vpa, vperp, z, r, parallel_io)

        ### create variables for time-dependent quantities and store them ###
        ### in a struct for later access ###
        io_dfns = define_dynamic_dfn_variables!(
            fid, r, z, vperp, vpa, vzeta, vr, vz, composition, parallel_io,
            external_source_settings, evolve_density, evolve_upar, evolve_ppar)

        close(fid)

        return file_info
    end

    # For processes other than the root process of each shared-memory group...
    return nothing
end

"""
Reopen an existing distribution-functions output file to append more data
"""
function reopen_dfns_io(file_info)
    @serial_region begin
        filename, parallel_io, io_comm = file_info
        fid = reopen_output_file(filename, parallel_io, io_comm)
        dyn = get_group(fid, "dynamic_data")

        variable_list = get_variable_keys(dyn)
        function getvar(name)
            if name ∈ variable_list
                return dyn[name]
            else
                return nothing
            end
        end
        io_moments = io_moments_info(fid, getvar("time"), getvar("phi"), getvar("Er"),
                                     getvar("Ez"), getvar("density"),
                                     getvar("parallel_flow"), getvar("parallel_pressure"),
                                     getvar("perpendicular_pressure"),
                                     getvar("parallel_heat_flux"), getvar("thermal_speed"),
                                     getvar("entropy_production"), getvar("chodura_integral_lower"),
                                     getvar("chodura_integral_upper"),
                                     getvar("electron_density"),
                                     getvar("electron_parallel_flow"),
                                     getvar("electron_parallel_pressure"),
                                     getvar("electron_parallel_heat_flux"),
                                     getvar("electron_thermal_speed"),
                                     getvar("density_neutral"), getvar("uz_neutral"),
                                     getvar("pz_neutral"), getvar("qz_neutral"),
                                     getvar("thermal_speed_neutral"),
                                     getvar("external_source_amplitude"),
                                     getvar("external_source_density_amplitude"),
                                     getvar("external_source_momentum_amplitude"),
                                     getvar("external_source_pressure_amplitude"),
                                     getvar("external_source_controller_integral"),
                                     getvar("external_source_neutral_amplitude"),
                                     getvar("external_source_neutral_density_amplitude"),
                                     getvar("external_source_neutral_momentum_amplitude"),
                                     getvar("external_source_neutral_pressure_amplitude"),
                                     getvar("external_source_neutral_controller_integral"),
                                     getvar("external_source_electron_amplitude"),
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
                                     getvar("time_for_run"), getvar("step_counter"),
                                     getvar("dt"), getvar("failure_counter"),
                                     getvar("failure_caused_by"),
                                     getvar("limit_caused_by"),
                                     getvar("dt_before_last_fail"),
                                     getvar("electron_step_counter"),
                                     getvar("electron_dt"),
                                     getvar("electron_failure_counter"),
                                     getvar("electron_failure_caused_by"),
                                     getvar("electron_limit_caused_by"),
                                     getvar("electron_dt_before_last_fail"), parallel_io)

        return io_dfns_info(fid, getvar("f"), getvar("f_electron"), getvar("f_neutral"),
                            parallel_io, io_moments)
    end

    # For processes other than the root process of each shared-memory group...
    return nothing
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

@debug_shared_array begin
    function append_to_dynamic_var(data::DebugMPISharedArray, args...; kwargs...)
        return append_to_dynamic_var(data.data, args...; kwargs...)
    end
end

"""
write time-dependent moments data for ions, electrons and neutrals to the binary output
file
"""
function write_all_moments_data_to_binary(moments, fields, t, n_ion_species,
                                          n_neutral_species, io_or_file_info_moments,
                                          t_idx, time_for_run, t_params, r, z)
    @serial_region begin
        # Only read/write from first process in each 'block'

        if isa(io_or_file_info_moments, io_moments_info)
            io_moments = io_or_file_info_moments
            closefile = false
        else
            io_moments = reopen_moments_io(io_or_file_info_moments)
            closefile = true
        end

        parallel_io = io_moments.parallel_io

        # add the time for this time slice to the hdf5 file
        append_to_dynamic_var(io_moments.time, t, t_idx, parallel_io)

        write_em_fields_data_to_binary(fields, io_moments, t_idx, r, z)

        write_ion_moments_data_to_binary(moments, n_ion_species, io_moments, t_idx, r, z)

        write_electron_moments_data_to_binary(moments, t_params.electron, io_moments,
                                              t_idx, r, z)

        write_neutral_moments_data_to_binary(moments, n_neutral_species, io_moments,
                                             t_idx, r, z)

        append_to_dynamic_var(io_moments.time_for_run, time_for_run, t_idx, parallel_io)
        append_to_dynamic_var(io_moments.step_counter, t_params.step_counter[], t_idx, parallel_io)
        append_to_dynamic_var(io_moments.dt, t_params.dt_before_output[], t_idx, parallel_io)
        append_to_dynamic_var(io_moments.failure_counter, t_params.failure_counter[], t_idx, parallel_io)
        append_to_dynamic_var(io_moments.failure_caused_by, t_params.failure_caused_by,
                              t_idx, parallel_io, length(t_params.failure_caused_by);
                              only_root=true)
        append_to_dynamic_var(io_moments.limit_caused_by, t_params.limit_caused_by, t_idx,
                              parallel_io, length(t_params.limit_caused_by);
                              only_root=true)
        append_to_dynamic_var(io_moments.dt_before_last_fail,
                              t_params.dt_before_last_fail[], t_idx, parallel_io)

        closefile && close(io_moments.fid)
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

        parallel_io = io_moments.parallel_io

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
function write_ion_moments_data_to_binary(moments, n_ion_species,
                                          io_moments::io_moments_info, t_idx, r, z)
    @serial_region begin
        # Only read/write from first process in each 'block'

        parallel_io = io_moments.parallel_io

        # add the density data at this time slice to the output file
        append_to_dynamic_var(io_moments.density, moments.ion.dens, t_idx,
                              parallel_io, z, r, n_ion_species)
        append_to_dynamic_var(io_moments.parallel_flow, moments.ion.upar, t_idx,
                              parallel_io, z, r, n_ion_species)
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
            append_to_dynamic_var(io_moments.external_source_amplitude,
                                  moments.ion.external_source_amplitude, t_idx,
                                  parallel_io, z, r)
            if moments.evolve_density
                append_to_dynamic_var(io_moments.external_source_density_amplitude,
                                      moments.ion.external_source_density_amplitude,
                                      t_idx, parallel_io, z, r)
            end
            if moments.evolve_upar
                append_to_dynamic_var(io_moments.external_source_momentum_amplitude,
                                      moments.ion.external_source_momentum_amplitude,
                                      t_idx, parallel_io, z, r)
            end
            if moments.evolve_ppar
                append_to_dynamic_var(io_moments.external_source_pressure_amplitude,
                                      moments.ion.external_source_pressure_amplitude,
                                      t_idx, parallel_io, z, r)
            end
        end
        if io_moments.external_source_controller_integral !== nothing
            if size(moments.ion.external_source_controller_integral) == (1,1)
                append_to_dynamic_var(io_moments.external_source_controller_integral,
                                      moments.ion.external_source_controller_integral[1,1],
                                      t_idx, parallel_io)
            else
                append_to_dynamic_var(io_moments.external_source_controller_integral,
                                      moments.ion.external_source_controller_integral,
                                      t_idx, parallel_io, z, r)
            end
        end
        if moments.evolve_density || moments.evolve_upar || moments.evolve_ppar
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
function write_electron_moments_data_to_binary(moments, t_params,
                                               io_moments::Union{io_moments_info,io_initial_electron_info},
                                               t_idx, r, z)
    @serial_region begin
        # Only read/write from first process in each 'block'

        parallel_io = io_moments.parallel_io

        append_to_dynamic_var(io_moments.electron_density, moments.electron.dens, t_idx,
                              parallel_io, z, r)
        append_to_dynamic_var(io_moments.electron_parallel_flow, moments.electron.upar,
                              t_idx, parallel_io, z, r)
        append_to_dynamic_var(io_moments.electron_parallel_pressure,
                              moments.electron.ppar, t_idx, parallel_io, z, r)
        append_to_dynamic_var(io_moments.electron_parallel_heat_flux,
                              moments.electron.qpar, t_idx, parallel_io, z, r)
        append_to_dynamic_var(io_moments.electron_thermal_speed, moments.electron.vth,
                              t_idx, parallel_io, z, r)
        if io_moments.external_source_electron_amplitude !== nothing
            append_to_dynamic_var(io_moments.external_source_electron_amplitude,
                                  moments.electron.external_source_amplitude, t_idx,
                                  parallel_io, z, r)
            append_to_dynamic_var(io_moments.external_source_electron_density_amplitude,
                                  moments.electron.external_source_density_amplitude,
                                  t_idx, parallel_io, z, r)
            append_to_dynamic_var(io_moments.external_source_electron_momentum_amplitude,
                                  moments.electron.external_source_momentum_amplitude,
                                  t_idx, parallel_io, z, r)
            append_to_dynamic_var(io_moments.external_source_electron_pressure_amplitude,
                                  moments.electron.external_source_pressure_amplitude,
                                  t_idx, parallel_io, z, r)
        end
        append_to_dynamic_var(io_moments.electron_constraints_A_coefficient,
                              moments.electron.constraints_A_coefficient, t_idx,
                              parallel_io, z, r)
        append_to_dynamic_var(io_moments.electron_constraints_B_coefficient,
                              moments.electron.constraints_B_coefficient, t_idx,
                              parallel_io, z, r)
        append_to_dynamic_var(io_moments.electron_constraints_C_coefficient,
                              moments.electron.constraints_C_coefficient, t_idx,
                              parallel_io, z, r)

        if t_params !== nothing
            # Save timestepping info
            append_to_dynamic_var(io_moments.electron_step_counter, t_params.step_counter[], t_idx, parallel_io)
            append_to_dynamic_var(io_moments.electron_dt, t_params.dt_before_output[], t_idx, parallel_io)
            append_to_dynamic_var(io_moments.electron_failure_counter, t_params.failure_counter[], t_idx, parallel_io)
            append_to_dynamic_var(io_moments.electron_failure_caused_by, t_params.failure_caused_by,
                                  t_idx, parallel_io, length(t_params.failure_caused_by);
                                  only_root=true)
            append_to_dynamic_var(io_moments.electron_limit_caused_by, t_params.limit_caused_by, t_idx,
                                  parallel_io, length(t_params.limit_caused_by);
                                  only_root=true)
            append_to_dynamic_var(io_moments.electron_dt_before_last_fail,
                                  t_params.dt_before_last_fail[], t_idx, parallel_io)
        end
    end

    return nothing
end

"""
write time-dependent moments data for neutrals to the binary output file

Note: should only be called from within a function that (re-)opens the output file.
"""
function write_neutral_moments_data_to_binary(moments, n_neutral_species,
                                              io_moments::io_moments_info, t_idx, r, z)
    if n_neutral_species ≤ 0
        return nothing
    end

    @serial_region begin
        # Only read/write from first process in each 'block'

        parallel_io = io_moments.parallel_io

        append_to_dynamic_var(io_moments.density_neutral, moments.neutral.dens, t_idx,
                              parallel_io, z, r, n_neutral_species)
        append_to_dynamic_var(io_moments.uz_neutral, moments.neutral.uz, t_idx,
                              parallel_io, z, r, n_neutral_species)
        append_to_dynamic_var(io_moments.pz_neutral, moments.neutral.pz, t_idx,
                              parallel_io, z, r, n_neutral_species)
        append_to_dynamic_var(io_moments.qz_neutral, moments.neutral.qz, t_idx,
                              parallel_io, z, r, n_neutral_species)
        append_to_dynamic_var(io_moments.thermal_speed_neutral, moments.neutral.vth,
                              t_idx, parallel_io, z, r, n_neutral_species)
        if io_moments.external_source_neutral_amplitude !== nothing
            append_to_dynamic_var(io_moments.external_source_neutral_amplitude,
                                  moments.neutral.external_source_amplitude, t_idx,
                                  parallel_io, z, r)
            if moments.evolve_density
                append_to_dynamic_var(io_moments.external_source_neutral_density_amplitude,
                                      moments.neutral.external_source_density_amplitude,
                                      t_idx, parallel_io, z, r)
            end
            if moments.evolve_upar
                append_to_dynamic_var(io_moments.external_source_neutral_momentum_amplitude,
                                      moments.neutral.external_source_momentum_amplitude,
                                      t_idx, parallel_io, z, r)
            end
            if moments.evolve_ppar
                append_to_dynamic_var(io_moments.external_source_neutral_pressure_amplitude,
                                      moments.neutral.external_source_pressure_amplitude,
                                      t_idx, parallel_io, z, r)
            end
        end
        if io_moments.external_source_neutral_controller_integral !== nothing
            if size(moments.neutral.external_source_neutral_controller_integral) == (1,1)
                append_to_dynamic_var(io_moments.external_source_neutral_controller_integral,
                                      moments.neutral.external_source_controller_integral[1,1],
                                      t_idx, parallel_io)
            else
                append_to_dynamic_var(io_moments.external_source_neutral_controller_integral,
                                      moments.neutral.external_source_controller_integral,
                                      t_idx, parallel_io, z, r)
            end
        end
        if moments.evolve_density || moments.evolve_upar || moments.evolve_ppar
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
function write_all_dfns_data_to_binary(pdf, moments, fields, t, n_ion_species,
                                       n_neutral_species, io_or_file_info_dfns, t_idx,
                                       time_for_run, t_params, r, z, vperp, vpa, vzeta,
                                        vr, vz)
    @serial_region begin
        # Only read/write from first process in each 'block'

        if isa(io_or_file_info_dfns, io_dfns_info)
            io_dfns = io_or_file_info_dfns
            closefile = false
        else
            io_dfns = reopen_dfns_io(io_or_file_info_dfns)
            closefile = true
        end

        # Write the moments for this time slice to the output file.
        # This also updates the time.
        write_all_moments_data_to_binary(moments, fields, t, n_ion_species,
                                         n_neutral_species, io_dfns.io_moments, t_idx,
                                         time_for_run, t_params, r, z)

        # add the distribution function data at this time slice to the output file
        write_ion_dfns_data_to_binary(pdf.ion.norm, n_ion_species, io_dfns, t_idx, r, z,
                                      vperp, vpa)
        if pdf.electron !== nothing
            write_electron_dfns_data_to_binary(pdf.electron.norm, io_dfns, t_idx, r, z,
                                               vperp, vpa)
        end
        write_neutral_dfns_data_to_binary(pdf.neutral.norm, n_neutral_species, io_dfns,
                                          t_idx, r, z, vzeta, vr, vz)

        closefile && close(io_dfns.fid)
    end
    return nothing
end

"""
write time-dependent distribution function data for ions to the binary output file

Note: should only be called from within a function that (re-)opens the output file.
"""
function write_ion_dfns_data_to_binary(ff, n_ion_species, io_dfns::io_dfns_info, t_idx, r,
                                       z, vperp, vpa)
    @serial_region begin
        # Only read/write from first process in each 'block'

        parallel_io = io_dfns.parallel_io

        append_to_dynamic_var(io_dfns.f, ff, t_idx, parallel_io, vpa, vperp, z, r,
                              n_ion_species)
    end
    return nothing
end

"""
write time-dependent distribution function data for electrons to the binary output file

Note: should only be called from within a function that (re-)opens the output file.
"""
function write_electron_dfns_data_to_binary(ff_electron,
                                            io_dfns::Union{io_dfns_info,io_initial_electron_info},
                                            t_idx, r, z, vperp, vpa)
    @serial_region begin
        # Only read/write from first process in each 'block'

        parallel_io = io_dfns.parallel_io

        if io_dfns.f_electron !== nothing
            append_to_dynamic_var(io_dfns.f_electron, ff_electron, t_idx, parallel_io,
                                  vpa, vperp, z, r)
        end
    end
    return nothing
end

"""
write time-dependent distribution function data for neutrals to the binary output file

Note: should only be called from within a function that (re-)opens the output file.
"""
function write_neutral_dfns_data_to_binary(ff_neutral, n_neutral_species,
                                           io_dfns::io_dfns_info, t_idx, r, z, vzeta, vr,
                                           vz)
    @serial_region begin
        # Only read/write from first process in each 'block'

        parallel_io = io_dfns.parallel_io

        if n_neutral_species > 0
            append_to_dynamic_var(io_dfns.f_neutral, ff_neutral, t_idx, parallel_io, vz,
                                  vr, vzeta, z, r, n_neutral_species)
        end
    end
    return nothing
end

"""
    write_electron_state(pdf, moments, t_params, t, io_initial_electron,
                         t_idx, r, z, vperp, vpa)

Write the electron state to an output file.
"""
function write_electron_state(pdf, moments, t_params, t, io_or_file_info_initial_electron,
                              t_idx, r, z, vperp, vpa)

    @serial_region begin
        # Only read/write from first process in each 'block'

        if isa(io_or_file_info_initial_electron, io_dfns_info)
            io_initial_electron = io_or_file_info_initial_electron
            closefile = false
        else
            io_initial_electron = reopen_initial_electron_io(io_or_file_info_initial_electron)
            closefile = true
        end

        parallel_io = io_initial_electron.parallel_io

        # add the pseudo-time for this time slice to the hdf5 file
        append_to_dynamic_var(io_initial_electron.pseudotime, t, t_idx, parallel_io)

        write_electron_dfns_data_to_binary(pdf, io_initial_electron, t_idx, r, z, vperp,
                                           vpa)

        write_electron_moments_data_to_binary(moments, t_params, io_initial_electron,
                                              t_idx, r, z)

        closefile && close(io_initial_electron.fid)
    end

    return nothing
end

"""
close all opened output files
"""
function finish_file_io(ascii_io::Union{ascii_ios,Nothing},
                        binary_moments::Union{io_moments_info,Tuple,Nothing},
                        binary_dfns::Union{io_dfns_info,Tuple,Nothing})
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
        binary_initial_electron::Union{io_initial_electron_info,Tuple,Nothing,Bool})

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
function write_data_to_ascii(pdf, moments, fields, vpa, vperp, z, r, t, n_ion_species,
                             n_neutral_species, ascii_io::Union{ascii_ios,Nothing})
    if ascii_io === nothing || ascii_io.moments_ion === nothing
        # ascii I/O is disabled
        return nothing
    end

    @serial_region begin
        # Only read/write from first process in each 'block'

        write_f_ascii(pdf, z, vpa, t, ascii_io.ff)
        write_moments_ion_ascii(moments.ion, z, r, t, n_ion_species, ascii_io.moments_ion)
        write_moments_electron_ascii(moments.electron, z, r, t, ascii_io.moments_electron)
        if n_neutral_species > 0
            write_moments_neutral_ascii(moments.neutral, z, r, t, n_neutral_species, ascii_io.moments_neutral)
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
                for j ∈ 1:vpa.n
                    for i ∈ 1:z.n
                        println(ascii_io,"t: ", t, "   z: ", z.grid[i],
                            "  vpa: ", vpa.grid[j], "   fion: ", f.ion.norm[i,j,1], 
                            "   fneutral: ", f.neutral.norm[i,j,1])
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

"""
An nc_info instance that may be initialised for writing debug output

This is a non-const module variable, so does cause type instability, but it is only used
for debugging (from `debug_dump()`) so performance is not critical.
"""
debug_output_file = nothing

"""
Global counter for calls to debug_dump
"""
const debug_output_counter = Ref(1)

"""
    debug_dump(ff, dens, upar, ppar, phi, t; istage=0, label="")
    debug_dump(fvec::scratch_pdf, fields::em_fields_struct, t; istage=0, label="")

Dump variables into a NetCDF file for debugging

Intended to be called more frequently than `write_data_to_binary()`, possibly several
times within a timestep, so includes a `label` argument to identify the call site.

Writes to a file called `debug_output.h5` in the current directory.

Can either be called directly with the arrays to be dumped (fist signature), or using
`scratch_pdf` and `em_fields_struct` structs.

`nothing` can be passed to any of the positional arguments (if they are unavailable at a
certain point in the code, or just not interesting). `t=nothing` will set `t` to the
value saved in the previous call (or 0.0 on the first call). Passing `nothing` to the
other arguments will set that array to `0.0` for this call (need to write some value so
all the arrays have the same length, with an entry for each call to `debug_dump()`).
"""
function debug_dump end
function debug_dump(vz::coordinate, vr::coordinate, vzeta::coordinate, vpa::coordinate,
                    vperp::coordinate, z::coordinate, r::coordinate, t::mk_float;
                    evolve_density, evolve_upar, evolve_ppar,
                    ff=nothing, dens=nothing, upar=nothing, ppar=nothing, pperp=nothing, qpar=nothing,
                    vth=nothing,
                    ff_neutral=nothing, dens_neutral=nothing, uz_neutral=nothing,
                    #ur_neutral=nothing, uzeta_neutral=nothing,
                    pz_neutral=nothing,
                    #pr_neutral=nothing, pzeta_neutral=nothing,
                    qz_neutral=nothing,
                    #qr_neutral=nothing, qzeta_neutral=nothing,
                    vth_neutral=nothing,
                    phi=nothing, Er=nothing, Ez=nothing,
                    istage=0, label="")
    global debug_output_file

    # Only read/write from first process in each 'block'
    _block_synchronize()
    @serial_region begin
        if debug_output_file === nothing
            # Open the file the first time`debug_dump()` is called

            debug_output_counter[] = 1

            (nvpa, nvperp, nz, nr, n_species) = size(ff)
            prefix = "debug_output.$(iblock_index[])"
            filename = string(prefix, ".h5")
            # if a file with the requested name already exists, remove it
            isfile(filename) && rm(filename)
            # create the new NetCDF file
            fid = open_output_file_hdf5(prefix)
            # write a header to the NetCDF file
            add_attribute!(fid, "file_info",
                           "This is a file containing debug output from the moment_kinetics code")

            ### define coordinate dimensions ###
            define_io_coordinates!(fid, vz, vr, vzeta, vpa, vperp, z, r, false)

            ### create variables for time-dependent quantities and store them ###
            ### in a struct for later access ###
            io_moments = define_dynamic_moment_variables!(fid, composition.n_ion_species,
                                                          composition.n_neutral_species,
                                                          r, z, false,
                                                          external_source_settings,
                                                          evolve_density, evolve_upar,
                                                          evolve_ppar,
                                                          composition.electron_physics)
            io_dfns = define_dynamic_dfn_variables!(
                fid, r, z, vperp, vpa, vzeta, vr, vz, composition.n_ion_species,
                composition.n_neutral_species, false, external_source_settings,
                evolve_density, evolve_upar, evolve_ppar)

            # create the "istage" variable, used to identify the rk stage where
            # `debug_dump()` was called
            dynamic = fid["dynamic_data"]
            io_istage = create_dynamic_variable!(dynamic, "istage", mk_int;
                                                 parallel_io=parallel_io,
                                                 description="rk istage")
            # create the "label" variable, used to identify the `debug_dump()` call-site
            io_label = create_dynamic_variable!(dynamic, "label", String;
                                                parallel_io=parallel_io,
                                                description="call-site label")

            # create a struct that stores the variables and other info needed for
            # writing to the netcdf file during run-time
            debug_output_file = (fid=fid, moments=io_moments, dfns=io_dfns,
                                 istage=io_istage, label=io_label)
        end

        # add the time for this time slice to the netcdf file
        if t === nothing
            if debug_output_counter[] == 1
                debug_output_file.moments.time[debug_output_counter[]] = 0.0
            else
                debug_output_file.moments.time[debug_output_counter[]] =
                debug_output_file.moments.time[debug_output_counter[]-1]
            end
        else
            debug_output_file.moments.time[debug_output_counter[]] = t
        end
        # add the rk istage for this call to the netcdf file
        debug_output_file.istage[debug_output_counter[]] = istage
        # add the label for this call to the netcdf file
        debug_output_file.label[debug_output_counter[]] = label
        # add the distribution function data at this time slice to the netcdf file
        if ff === nothing
            debug_output_file.dfns.ion_f[:,:,:,:,:,debug_output_counter[]] = 0.0
        else
            debug_output_file.dfns.ion_f[:,:,:,:,:,debug_output_counter[]] = ff
        end
        # add the moments data at this time slice to the netcdf file
        if dens === nothing
            debug_output_file.moments.density[:,:,:,debug_output_counter[]] = 0.0
        else
            debug_output_file.moments.density[:,:,:,debug_output_counter[]] = dens
        end
        if upar === nothing
            debug_output_file.moments.parallel_flow[:,:,:,debug_output_counter[]] = 0.0
        else
            debug_output_file.moments.parallel_flow[:,:,:,debug_output_counter[]] = upar
        end
        if ppar === nothing
            debug_output_file.moments.parallel_pressure[:,:,:,debug_output_counter[]] = 0.0
        else
            debug_output_file.moments.parallel_pressure[:,:,:,debug_output_counter[]] = ppar
        end
        if pperp === nothing
            debug_output_file.moments.perpendicular_pressure[:,:,:,debug_output_counter[]] = 0.0
        else
            debug_output_file.moments.perpendicular_pressure[:,:,:,debug_output_counter[]] = pperp
        end
        if qpar === nothing
            debug_output_file.moments.parallel_heat_flux[:,:,:,debug_output_counter[]] = 0.0
        else
            debug_output_file.moments.parallel_heat_flux[:,:,:,debug_output_counter[]] = qpar
        end
        if vth === nothing
            debug_output_file.moments.thermal_speed[:,:,:,debug_output_counter[]] = 0.0
        else
            debug_output_file.moments.thermal_speed[:,:,:,debug_output_counter[]] = vth
        end

        # add the neutral distribution function data at this time slice to the netcdf file
        if ff_neutral === nothing
            debug_output_file.dfns.f_neutral[:,:,:,:,:,:,debug_output_counter[]] = 0.0
        else
            debug_output_file.dfns.f_neutral[:,:,:,:,:,:,debug_output_counter[]] = ff_neutral
        end
        # add the neutral moments data at this time slice to the netcdf file
        if dens === nothing
            debug_output_file.moments.density_neutral[:,:,:,debug_output_counter[]] = 0.0
        else
            debug_output_file.moments.density_neutral[:,:,:,debug_output_counter[]] = dens_neutral
        end
        if uz_neutral === nothing
            debug_output_file.moments.uz_neutral[:,:,:,debug_output_counter[]] = 0.0
        else
            debug_output_file.moments.uz_neutral[:,:,:,debug_output_counter[]] = uz_neutral
        end
        if pz_neutral === nothing
            debug_output_file.moments.pz_neutral[:,:,:,debug_output_counter[]] = 0.0
        else
            debug_output_file.moments.pz_neutral[:,:,:,debug_output_counter[]] = pz_neutral
        end
        if qz_neutral === nothing
            debug_output_file.moments.qz_neutral[:,:,:,debug_output_counter[]] = 0.0
        else
            debug_output_file.moments.qz_neutral[:,:,:,debug_output_counter[]] = qz_neutral
        end
        if vth_neutral === nothing
            debug_output_file.moments.thermal_speed_neutral[:,:,:,debug_output_counter[]] = 0.0
        else
            debug_output_file.moments.thermal_speed_neutral[:,:,:,debug_output_counter[]] = vth_neutral
        end

        # add the electrostatic potential data at this time slice to the netcdf file
        if phi === nothing
            debug_output_file.moments.phi[:,:,debug_output_counter[]] = 0.0
        else
            debug_output_file.moments.phi[:,:,debug_output_counter[]] = phi
        end
        if Er === nothing
            debug_output_file.moments.Er[:,:,debug_output_counter[]] = 0.0
        else
            debug_output_file.moments.Er[:,:,debug_output_counter[]] = Er
        end
        if Ez === nothing
            debug_output_file.moments.Ez[:,:,debug_output_counter[]] = 0.0
        else
            debug_output_file.moments.Ez[:,:,debug_output_counter[]] = Ez
        end
    end

    debug_output_counter[] += 1

    _block_synchronize()

    return nothing
end
function debug_dump(fvec::Union{scratch_pdf,Nothing},
                    fields::Union{em_fields_struct,Nothing}, vz, vr, vzeta, vpa, vperp, z,
                    r, t; istage=0, label="")
    if fvec === nothing
        pdf = nothing
        density = nothing
        upar = nothing
        ppar = nothing
        pperp = nothing
        pdf_neutral = nothing
        density_neutral = nothing
    else
        pdf = fvec.pdf
        density = fvec.density
        upar = fvec.upar
        ppar = fvec.ppar
        pperp = fvec.pperp
        pdf_neutral = fvec.pdf_neutral
        density_neutral = fvec.density_neutral
    end
    if fields === nothing
        phi = nothing
        Er = nothing
        Ez = nothing
    else
        phi = fields.phi
        Er = fields.Er
        Ez = fields.Ez
    end
    return debug_dump(vz, vr, vzeta, vpa, vperp, z, r, t; ff=pdf, dens=density, upar=upar,
                      ppar=ppar, pperp=pperp, ff_neutral=pdf_neutral, dens_neutral=density_neutral,
                      phi=phi, Er=Er, Ez=Ez, t, istage=istage, label=label)
end

end
