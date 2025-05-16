"""
"""
module load_data

export open_readonly_output_file
export load_fields_data
export load_ion_moments_data
export load_electron_moments_data
export load_neutral_particle_moments_data
export load_pdf_data
export load_neutral_pdf_data
export load_coordinate_data
export load_time_data
export load_block_data
export load_rank_data
export load_species_data
export read_distributed_zr_data!

using ..array_allocation: allocate_float, allocate_int
using ..calculus: derivative!
using ..communication
using ..continuity: continuity_equation!, neutral_continuity_equation!
using ..coordinates: coordinate, define_coordinate
using ..electron_fluid_equations: electron_energy_equation!
using ..electron_vpa_advection: update_electron_speed_vpa!
using ..electron_z_advection: update_electron_speed_z!
using ..energy_equation: energy_equation!, neutral_energy_equation!
using ..file_io: check_io_implementation, get_group, get_subgroup_keys, get_variable_keys
using ..force_balance: force_balance!, neutral_force_balance!
using ..input_structs
using ..interpolation: interpolate_to_grid_1d!
using ..krook_collisions
using ..collision_frequencies: get_collision_frequency_ii, get_collision_frequency_ee,
                                get_collision_frequency_ei
using ..looping
using ..moment_kinetics_input: mk_input
using ..neutral_vz_advection: update_speed_neutral_vz!
using ..neutral_z_advection: update_speed_neutral_z!
using ..type_definitions: mk_float, mk_int, OptionsDict
using ..utils: get_CFL!, get_minimum_CFL_z, get_minimum_CFL_vpa, get_minimum_CFL_neutral_z,
               get_minimum_CFL_neutral_vz, enum_from_string
using ..vpa_advection: update_speed_vpa!
using ..z_advection: update_speed_z!

using Glob
using HDF5
using MPI
using OrderedCollections: OrderedDict

const timestep_diagnostic_variables = ("time_for_run", "step_counter", "dt",
                                       "failure_counter", "failure_caused_by",
                                       "steps_per_output", "failures_per_output",
                                       "failure_caused_by_per_output",
                                       "average_successful_dt", "electron_step_counter",
                                       "electron_dt", "electron_failure_counter",
                                       "electron_failure_caused_by",
                                       "electron_steps_per_ion_step",
                                       "electron_steps_per_output",
                                       "electron_failures_per_output",
                                       "electron_failure_caused_by_per_output",
                                       "electron_average_successful_dt")
const em_variables = ("phi", "Er", "Ez")
const ion_moment_variables = ("density", "parallel_flow", "pressure", "parallel_pressure",
                              "thermal_speed", "temperature", "parallel_heat_flux",
                              "collision_frequency_ii", "sound_speed", "mach_number",
                              "total_energy", "total_energy_flux")
const ion_moment_gradient_variables = ("ddens_dz", "ddens_dz_upwind", "dupar_dz",
                                       "dupar_dz_upwind", "dp_dz", "dp_dz_upwind",
                                       "dppar_dz", "dppar_dz_upwind", "dvth_dz", "dT_dz",
                                       "dqpar_dz")
const electron_moment_variables = ("electron_density", "electron_parallel_flow",
                                   "electron_pressure", "electron_parallel_pressure",
                                   "electron_thermal_speed", "electron_temperature",
                                   "electron_parallel_heat_flux",
                                   "collision_frequency_ee", "collision_frequency_ei")
const electron_moment_gradient_variables = ("electron_ddens_dz", "electron_dupar_dz",
                                            "electron_dp_dz", "electron_dvth_dz",
                                            "electron_dT_dz", "electron_dqpar_dz")
const neutral_moment_variables = ("density_neutral", "uz_neutral", "p_neutral",
                                  "pz_neutral", "thermal_speed_neutral",
                                  "temperature_neutral", "qz_neutral",
                                  "total_energy_neutral", "total_energy_flux_neutral")
const neutral_moment_gradient_variables = ("neutral_ddens_dz", "neutral_ddens_dz_upwind",
                                           "neutral_duz_dz", "neutral_duz_dz_upwind",
                                           "neutral_dp_dz", "neutral_dp_dz_upwind",
                                           "neutral_dpz_dz", "neutral_dvth_dz",
                                           "neutral_dT_dz", "neutral_dqz_dz")
const ion_source_variables = ("external_source_amplitude", "external_source_density_amplitude",
                                "external_source_momentum_amplitude", "external_source_pressure_amplitude",
                                "external_source_controller_integral")
const neutral_source_variables = ("external_source_neutral_amplitude", "external_source_neutral_density_amplitude",
                                "external_source_neutral_momentum_amplitude", "external_source_neutral_pressure_amplitude",
                                "external_source_neutral_controller_integral")
const electron_source_variables = ("external_source_electron_amplitude", "external_source_electron_density_amplitude",
                                "external_source_electron_momentum_amplitude", "external_source_electron_pressure_amplitude")
const all_moment_variables = tuple(em_variables..., ion_moment_variables...,
                                   electron_moment_variables...,
                                   neutral_moment_variables...,
                                   ion_moment_gradient_variables...,
                                   electron_moment_gradient_variables...,
                                   neutral_moment_gradient_variables...,
                                   ion_source_variables..., 
                                   electron_source_variables...,
                                   neutral_source_variables...)
const ion_dfn_variables = ("f",)
const electron_dfn_variables = ("f_electron",)
const neutral_dfn_variables = ("f_neutral",)
const all_dfn_variables = tuple(ion_dfn_variables..., electron_dfn_variables...,
                                neutral_dfn_variables...)

const ion_variables = tuple(ion_moment_variables..., ion_dfn_variables)
const neutral_variables = tuple(neutral_moment_variables..., neutral_dfn_variables)
const all_variables = tuple(all_moment_variables..., all_dfn_variables...)

function open_file_to_read end
function open_file_to_read(::Val{hdf5}, filename)
    return h5open(filename, "r")
end

function get_attribute end
function get_attribute(file_or_group_or_var::Union{HDF5.H5DataStore,HDF5.Dataset}, name)
    return attrs(file_or_group_or_var)[name]
end

"""
"""
function open_readonly_output_file(run_name, ext; iblock=0, printout=false)
    possible_names = (
        string(run_name, ".", ext, ".h5"),
        string(run_name, ".", ext, ".", iblock, ".h5"),
        string(run_name, ".", ext, ".cdf"),
        string(run_name, ".", ext, ".", iblock, ".cdf"),
    )
    existing_files = Tuple(f for f in possible_names if isfile(f))
    exists_count = length(existing_files)
    if exists_count == 0
        error("None of $possible_names exist, cannot open output")
    elseif exists_count > 1
        error("Multiple files present, do not know which to open: $existing_files")
    end

    # Have checked there is only one filename in existing_files
    filename = existing_files[1]

    if splitext(filename)[2] == ".h5"
        if printout
            print("Opening ", filename, " to read HDF5 data...")
        end
        # open the HDF5 file with given filename for reading
        check_io_implementation(hdf5)
        fid = open_file_to_read(Val(hdf5), filename)
    else
        if printout
            print("Opening ", filename, " to read NetCDF data...")
        end

        # open the netcdf file with given filename for reading
        check_io_implementation(netcdf)
        fid = open_file_to_read(Val(netcdf), filename)
    end
    if printout
        println("done.")
    end
    return fid
end

function get_nranks(run_name,nblocks,description)
    z_nrank = 0
    r_nrank = 0
    for iblock in 0:nblocks-1
        fid = open_readonly_output_file(run_name,description,iblock=iblock, printout=false)
        z_irank, r_irank = load_rank_data(fid,printout=false)
        z_nrank = max(z_irank,z_nrank)
        r_nrank = max(r_irank,r_nrank)
        close(fid)
    end
    r_nrank = r_nrank + 1
    z_nrank = z_nrank + 1
    return z_nrank, r_nrank
end

"""
Load a single variable from a file
"""
function load_variable end
function load_variable(file_or_group::HDF5.H5DataStore, name::String)
    # This overload deals with cases where fid is an HDF5 `File` or `Group` (`H5DataStore`
    # is the abstract super-type for both
    try
        return read(file_or_group[name])
    catch
        println("An error occured while loading $name")
        rethrow()
    end
end

"""
Load a slice of a single variable from a file
"""
function load_slice end
function load_slice(file_or_group::HDF5.H5DataStore, name::String, slices_or_indices...)
    # This overload deals with cases where fid is an HDF5 `File` or `Group` (`H5DataStore`
    # is the abstract super-type for both
    try
        return file_or_group[name][slices_or_indices...]
    catch
        println("An error occured while loading $name")
        rethrow()
    end
end

"""
    read_Dict_from_section(file_or_group, section_name; ignore_subsections=false)

Read information from `section_name` in `file_or_group`, returning a Dict.

By default, any subsections are included as nested Dicts. If `ignore_subsections=true`
they are ignored.
"""
function read_Dict_from_section(file_or_group, section_name; ignore_subsections=false)
    # Function that can be called recursively to read nested Dicts from sub-groups in
    # the output file
    section_io = get_group(file_or_group, section_name)
    section = OptionsDict()

    for key ∈ get_variable_keys(section_io)
        section[key] = load_variable(section_io, key)
    end
    if !ignore_subsections
        for key ∈ get_subgroup_keys(section_io)
            section[key] = read_Dict_from_section(section_io, key)
        end
    end

    return section
end

"""
Load saved input settings
"""
function load_input(fid)
    return read_Dict_from_section(fid, "input")
end

"""
    load_coordinate_data(fid, name; printout=false, irank=nothing, nrank=nothing,
                         run_directory=nothing, ignore_MPI=true)

Load data for the coordinate `name` from a file-handle `fid`.

Returns (`coord`, `spectral`, `chunk_size`). `coord` is a `coordinate` object. `spectral`
is the object used to implement the discretization in this coordinate. `chunk_size` is the
size of chunks in this coordinate that was used when writing to the output file.

If `printout` is set to `true` a message will be printed when this function is called.

If `irank` and `nrank` are passed, then the `coord` and `spectral` objects returned will
be set up for the parallelisation specified by `irank` and `nrank`, rather than the one
implied by the output file.

Unless `ignore_MPI=false` is passed, the returned coordinates will be created without
shared memory scratch arrays (`ignore_MPI=true` will be passed through to
[`define_coordinate`](@ref)).
"""
function load_coordinate_data(fid, name; printout=false, irank=nothing, nrank=nothing,
                              run_directory=nothing, ignore_MPI=true)
    if printout
        println("Loading $name coordinate data...")
    end

    overview = get_group(fid, "overview")
    parallel_io = load_variable(overview, "parallel_io")

    coords_group = get_group(fid, "coords")
    if name ∈ get_subgroup_keys(coords_group)
        coord_group = get_group(coords_group, name)
    else
        return nothing, nothing, nothing
    end

    input = OptionsDict()
    ngrid = load_variable(coord_group, "ngrid")
    input["ngrid"] = ngrid
    n_local = load_variable(coord_group, "n_local")
    n_global = load_variable(coord_group, "n_global")
    grid = load_variable(coord_group, "grid")
    wgts = load_variable(coord_group, "wgts")

    if n_global == 1 && ngrid == 1
        nelement_global = 1
    else
        nelement_global = (n_global-1) ÷ (ngrid-1)
    end

    if irank === nothing && nrank === nothing
        irank = load_variable(coord_group, "irank")
        if "nrank" in keys(coord_group)
            nrank = load_variable(coord_group, "nrank")
        else
            # Workaround for older output files that did not save nrank
            if name ∈ ("r", "z")
                nrank = max(n_global - 1, 1) ÷ max(n_local - 1, 1)
            else
                nrank = 1
            end
        end

        if n_local == 1 && ngrid == 1
            nelement_local = 1
        else
            nelement_local = (n_local-1) ÷ (ngrid-1)
        end
    else
        # Want to create coordinate with a specific `nrank` and `irank`. Need to
        # calculate `nelement_local` consistent with `nrank`, which might be different now
        # than in the original simulation.
        # Note `n_local` is only (possibly) used to calculate the `chunk_size`. It
        # probably makes most sense for that to be the same as the original simulation, so
        # do not recalculate `n_local` here.
        irank === nothing && error("When `nrank` is passed, `irank` must also be passed")
        nrank === nothing && error("When `irank` is passed, `nrank` must also be passed")

        if nelement_global % nrank != 0
            error("Can only load coordinate with new `nrank` that divides "
                  * "nelement_global=$nelement_global exactly.")
        end
        nelement_local = nelement_global ÷ nrank
    end
    if "chunk_size" ∈ coord_group
        chunk_size = load_variable(coord_group, "chunk_size")
    else
        # Workaround for older output files that did not save chunk_size.
        # Sub-optimal for runs that used parallel I/O.
        if nrank == 1
            chunk_size = n_global
        else
            chunk_size = n_local - 1
        end
    end
    input["nelement"] = nelement_global
    input["nelement_local"] = nelement_local
    # L = global box length
    input["L"] = load_variable(coord_group, "L")
    input["discretization"] = load_variable(coord_group, "discretization")
    if "finite_difference_option" ∈ keys(coord_group)
        input["finite_difference_option"] = load_variable(coord_group, "finite_difference_option")
    else
        # Older output file
        input["finite_difference_option"] = load_variable(coord_group, "fd_option")
    end
    if "cheb_option" ∈ keys(coord_group)
        input["cheb_option"] = load_variable(coord_group, "cheb_option")
    else
        # Old output file
        input["cheb_option"] = "FFT"
    end
    input["bc"] = load_variable(coord_group, "bc")
    if "element_spacing_option" ∈ keys(coord_group)
        input["element_spacing_option"] = load_variable(coord_group, "element_spacing_option")
    else
        input["element_spacing_option"] = "uniform"
    end

    coord, spectral = define_coordinate(OptionsDict(name => input), name;
                                        parallel_io=parallel_io,
                                        run_directory=run_directory,
                                        ignore_MPI=ignore_MPI, irank=irank, nrank=nrank,
                                        comm=MPI.COMM_NULL)

    return coord, spectral, chunk_size
end

function load_run_info_history(fid)
    provenance_tracking = get_group(fid, "provenance_tracking")

    last_run_info = read_Dict_from_section(fid, "provenance_tracking";
                                           ignore_subsections=true)

    counter = 0
    pt_keys = keys(provenance_tracking)
    while "previous_run_$(counter+1)" ∈ pt_keys
        counter += 1
    end
    previous_runs_info =
        tuple((read_Dict_from_section(provenance_tracking, "previous_run_$i")
               for i ∈ 1:counter)...,
              last_run_info)

    return previous_runs_info
end

"""
"""
function load_species_data(fid; printout=false)
    if printout
        print("Loading species data...")
    end

    overview = get_group(fid, "overview")
    n_ion_species = load_variable(overview, "n_ion_species")
    n_neutral_species = load_variable(overview, "n_neutral_species")

    if printout
        println("done.")
    end

    return n_ion_species, n_neutral_species
end

"""
"""
function load_mk_options(fid)
    overview = get_group(fid, "overview")

    evolve_density = load_variable(overview, "evolve_density")
    evolve_upar = load_variable(overview, "evolve_upar")
    if "evolve_p" ∈ keys(overview)
        evolve_p = load_variable(overview, "evolve_p")
    else
        # Older output file, before option was renamed
        evolve_p = load_variable(overview, "evolve_ppar")
    end

    return evolve_density, evolve_upar, evolve_p
end

"""
If a tuple is given for `fid`, concatenate the "time" output from each file in the tuple
"""
function load_time_data(fid; printout=false)
    if printout
        print("Loading time data...")
    end

    if !isa(fid, Tuple)
        fid = (fid,)
    end

    group = get_group(first(fid), "dynamic_data")
    time = load_variable(group, "time")
    restarts_nt = [length(time)]
    for f ∈ fid[2:end]
        group = get_group(f, "dynamic_data")
        # Skip first point as this is a duplicate of the last point of the previous
        # restart.
        this_time = load_variable(group, "time")
        push!(restarts_nt, length(this_time))
        time = vcat(time, this_time[2:end])
    end
    ntime = length(time)

    if printout
        println("done.")
    end

    return ntime, time, restarts_nt
end

"""
"""
function load_block_data(fid; printout=false)
    if printout
        print("Loading block data...")
    end

    coords = get_group(fid, "coords")
    nblocks = load_variable(coords, "nblocks")
    iblock = load_variable(coords, "iblock")

    if printout
        println("done.")
    end

    return  nblocks, iblock
end

"""
"""
function load_rank_data(fid; printout=false)
    if printout
        print("Loading rank data...")
    end

    coords = get_group(fid, "coords")
    z_irank = load_variable(get_group(coords, "z"), "irank")
    r_irank = load_variable(get_group(coords, "r"), "irank")
    z_nrank = load_variable(get_group(coords, "z"), "nrank")
    r_nrank = load_variable(get_group(coords, "r"), "nrank")
    
    if printout
        println("done.")
    end

    return z_irank, z_nrank, r_irank, r_nrank
end

"""
"""
function load_fields_data(fid; printout=false)
    if printout
        print("Loading fields data...")
    end

    group = get_group(fid, "dynamic_data")

    # Read electrostatic potential
    phi = load_variable(group, "phi")

    # Read radial electric field
    Er = load_variable(group, "Er")

    # Read z electric field
    Ez = load_variable(group, "Ez")

    if printout
        println("done.")
    end

    return phi, Er, Ez
end

"""
"""
function load_ion_moments_data(fid; printout=false, extended_moments = false)
    if printout
        print("Loading ion velocity moments data...")
    end

    group = get_group(fid, "dynamic_data")

    # Read ion species density
    density = load_variable(group, "density")

    # Read ion species parallel flow
    parallel_flow = load_variable(group, "parallel_flow")

    # Read ion species pressure
    pressure = load_variable(group, "pressure")

    # Read ion species parallel pressure
    parallel_pressure = load_variable(group, "parallel_pressure")

    # Read ion_species parallel heat flux
    parallel_heat_flux = load_variable(group, "parallel_heat_flux")

    # Read ion species thermal speed
    thermal_speed = load_variable(group, "thermal_speed")

    if extended_moments
        # Read ion species perpendicular pressure
        perpendicular_pressure = load_variable(group, "perpendicular_pressure")

        # Read ion species entropy_production
        entropy_production = load_variable(group, "entropy_production")
    end

    if printout
        println("done.")
    end
    if extended_moments
        return density, parallel_flow, pressure, parallel_pressure, perpendicular_pressure, parallel_heat_flux, thermal_speed, entropy_production
    else
        return density, parallel_flow, pressure, parallel_pressure, parallel_heat_flux, thermal_speed
    end
end

"""
"""
function load_electron_moments_data(fid; printout=false)
    if printout
        print("Loading electron velocity moments data...")
    end

    group = get_group(fid, "dynamic_data")

    # Read electron pressure
    pressure = load_variable(group, "electron_pressure")

    # Read electron parallel pressure
    parallel_pressure = load_variable(group, "electron_parallel_pressure")

    # Read electron parallel heat flux
    parallel_heat_flux = load_variable(group, "electron_parallel_heat_flux")

    # Read electron thermal speed
    thermal_speed = load_variable(group, "electron_thermal_speed")

    if printout
        println("done.")
    end

    return pressure, parallel_pressure, parallel_heat_flux, thermal_speed
end

function load_neutral_particle_moments_data(fid; printout=false)
    if printout
        print("Loading neutral particle velocity moments data...")
    end

    group = get_group(fid, "dynamic_data")

    # Read neutral species density
    neutral_density = load_variable(group, "density_neutral")

    # Read neutral species uz
    neutral_uz = load_variable(group, "uz_neutral")

    # Read neutral species p
    neutral_p = load_variable(group, "p_neutral")

    # Read neutral species pz
    neutral_pz = load_variable(group, "pz_neutral")

    # Read neutral species qz
    neutral_qz = load_variable(group, "qz_neutral")

    # Read neutral species thermal speed
    neutral_thermal_speed = load_variable(group, "thermal_speed_neutral")

    if printout
        println("done.")
    end

    return neutral_density, neutral_uz, neutral_p, neutral_pz, neutral_qz,
           neutral_thermal_speed
end

"""
"""
function load_pdf_data(fid; printout=false)
    if printout
        print("Loading ion particle distribution function data...")
    end

    group = get_group(fid, "dynamic_data")

    # Read ion distribution function
    pdf = load_variable(group, "f")

    if printout
        println("done.")
    end

    return pdf
end
"""
"""
function load_neutral_pdf_data(fid; printout=false)
    if printout
        print("Loading neutral particle distribution function data...")
    end

    group = get_group(fid, "dynamic_data")

    # Read neutral distribution function
    neutral_pdf = load_variable(group, "f_neutral")

    if printout
        println("done.")
    end

    return neutral_pdf
end

"""
Reload pdf and moments from an existing output file.
"""
function reload_evolving_fields!(pdf, moments, fields, boundary_distributions,
                                 restart_prefix_iblock, time_index, composition, geometry,
                                 r, z, vpa, vperp, vzeta, vr, vz)
    code_time = 0.0
    dt = Ref(-Inf)
    dt_before_last_fail = Ref(Inf)
    electron_dt = Ref(-Inf)
    electron_dt_before_last_fail = Ref(Inf)
    previous_runs_info = nothing
    restart_electron_physics = nothing
    @begin_serial_region()
    @serial_region begin
        fid = open_readonly_output_file(restart_prefix_iblock[1], "dfns";
                                        iblock=restart_prefix_iblock[2])
        try # finally to make sure to close file0
            overview = get_group(fid, "overview")
            dynamic = get_group(fid, "dynamic_data")
            parallel_io = load_variable(overview, "parallel_io")
            if time_index < 0
                time_index, _, _ = load_time_data(fid)
            end
            restart_evolve_density, restart_evolve_upar, restart_evolve_p =
                load_mk_options(fid)

            restart_input = load_input(fid)

            previous_runs_info = load_run_info_history(fid)

            restart_n_ion_species, restart_n_neutral_species = load_species_data(fid)
            restart_r, restart_r_spectral, restart_z, restart_z_spectral, restart_vperp,
                restart_vperp_spectral, restart_vpa, restart_vpa_spectral, restart_vzeta,
                restart_vzeta_spectral, restart_vr,restart_vr_spectral, restart_vz,
                restart_vz_spectral = load_restart_coordinates(fid, r, z, vperp, vpa,
                                                               vzeta, vr, vz, parallel_io)

            # Test whether any interpolation is needed
            interpolation_needed = OrderedDict(
                x.name => (restart_x !== nothing
                           && (x.n != restart_x.n
                               || !all(isapprox.(x.grid, restart_x.grid))))
                for (x, restart_x) ∈ ((z, restart_z), (r, restart_r),
                                      (vperp, restart_vperp), (vpa, restart_vpa),
                                      (vzeta, restart_vzeta), (vr, restart_vr),
                                      (vz, restart_vz)))

            neutral_1V = (vzeta.n_global == 1 && vr.n_global == 1)
            restart_neutral_1V = ((restart_vzeta === nothing
                                   || restart_vzeta.n_global == 1)
                                  && (restart_vr === nothing || restart_vr.n_global == 1))
            if any(geometry.bzeta .!= 0.0) && ((neutral_1V && !restart_neutral_1V) ||
                                               (!neutral_1V && restart_neutral_1V))
                # One but not the other of the run being restarted from and this run are
                # 1V, but the interpolation below does not allow for vz and vpa being in
                # different directions. Therefore interpolation between 1V and 3V cases
                # only works (at the moment!) if bzeta=0.
                error("Interpolation between 1V and 3V neutrals not yet supported when "
                      * "bzeta!=0.")
            end

            code_time = load_slice(dynamic, "time", time_index)

            r_range, z_range, vperp_range, vpa_range, vzeta_range, vr_range, vz_range =
                get_reload_ranges(parallel_io, restart_r, restart_z, restart_vperp,
                                  restart_vpa, restart_vzeta, restart_vr, restart_vz)

            fields.phi .= reload_electron_moment("phi", dynamic, time_index, r, z,
                                                 r_range, z_range, restart_r,
                                                 restart_r_spectral, restart_z,
                                                 restart_z_spectral, interpolation_needed)
            moments.ion.dens .= reload_moment("density", dynamic, time_index, r, z,
                                              r_range, z_range, restart_r,
                                              restart_r_spectral, restart_z,
                                              restart_z_spectral, interpolation_needed)
            moments.ion.upar .= reload_moment("parallel_flow", dynamic, time_index, r, z,
                                              r_range, z_range, restart_r,
                                              restart_r_spectral, restart_z,
                                              restart_z_spectral, interpolation_needed)
            moments.ion.p .= reload_moment("pressure", dynamic, time_index, r,
                                           z, r_range, z_range, restart_r,
                                           restart_r_spectral, restart_z,
                                           restart_z_spectral, interpolation_needed)
            moments.ion.ppar .= reload_moment("parallel_pressure", dynamic, time_index, r,
                                              z, r_range, z_range, restart_r,
                                              restart_r_spectral, restart_z,
                                              restart_z_spectral, interpolation_needed)
            moments.ion.pperp .= reload_moment("perpendicular_pressure", dynamic,
                                               time_index, r, z, r_range, z_range,
                                               restart_r, restart_r_spectral, restart_z,
                                               restart_z_spectral, interpolation_needed)
            moments.ion.qpar .= reload_moment("parallel_heat_flux", dynamic, time_index,
                                              r, z, r_range, z_range, restart_r,
                                              restart_r_spectral, restart_z,
                                              restart_z_spectral, interpolation_needed)
            moments.ion.vth .= reload_moment("thermal_speed", dynamic, time_index, r, z,
                                             r_range, z_range, restart_r,
                                             restart_r_spectral, restart_z,
                                             restart_z_spectral, interpolation_needed)
            moments.ion.dSdt .= reload_moment("entropy_production", dynamic, time_index,
                                              r, z, r_range, z_range, restart_r,
                                              restart_r_spectral, restart_z,
                                              restart_z_spectral, interpolation_needed)
            if moments.evolve_density || moments.evolve_upar || moments.evolve_p
                if "ion_constraints_A_coefficient" ∈ keys(dynamic)
                    moments.ion.constraints_A_coefficient .=
                        reload_moment("ion_constraints_A_coefficient", dynamic,
                                      time_index, r, z, r_range, z_range, restart_r,
                                      restart_r_spectral, restart_z, restart_z_spectral,
                                      interpolation_needed)
                elseif moments.ion.constraints_A_coefficient !== nothing
                    moments.ion.constraints_A_coefficient .= 0.0
                end
                if "ion_constraints_B_coefficient" ∈ keys(dynamic)
                    moments.ion.constraints_B_coefficient .=
                        reload_moment("ion_constraints_B_coefficient", dynamic,
                                      time_index, r, z, r_range, z_range, restart_r,
                                      restart_r_spectral, restart_z, restart_z_spectral,
                                      interpolation_needed)
                elseif moments.ion.constraints_B_coefficient !== nothing
                    moments.ion.constraints_B_coefficient .= 0.0
                end
                if "ion_constraints_C_coefficient" ∈ keys(dynamic)
                    moments.ion.constraints_C_coefficient .=
                        reload_moment("ion_constraints_C_coefficient", dynamic,
                                      time_index, r, z, r_range, z_range, restart_r,
                                      restart_r_spectral, restart_z, restart_z_spectral,
                                      interpolation_needed)
                elseif moments.ion.constraints_C_coefficient !== nothing
                    moments.ion.constraints_C_coefficient .= 0.0
                end
            end
            if z.irank == 0
                if "chodura_integral_lower" ∈ keys(dynamic)
                    moments.ion.chodura_integral_lower .= load_slice(dynamic, "chodura_integral_lower",
                                                                     r_range, :, time_index)
                else
                    moments.ion.chodura_integral_lower .= 0.0
                end
            end
            if z.irank == z.nrank - 1
                if "chodura_integral_upper" ∈ keys(dynamic)
                    moments.ion.chodura_integral_upper .= load_slice(dynamic, "chodura_integral_upper",
                                                                     r_range, :, time_index)
                else
                    moments.ion.chodura_integral_upper .= 0.0
                end
            end

            if "external_source_controller_integral" ∈ get_variable_keys(dynamic)
                if length(moments.ion.external_source_controller_integral) == 1
                    moments.ion.external_source_controller_integral .=
                        load_slice(dynamic, "external_source_controller_integral", time_index)
            elseif size(moments.ion.external_source_controller_integral)[1] > 1 ||
                    size(moments.ion.external_source_controller_integral)[2] > 1 
                    moments.ion.external_source_controller_integral .=
                        reload_moment("external_source_controller_integral", dynamic,
                                      time_index, r, z, r_range, z_range, restart_r,
                                      restart_r_spectral, restart_z, restart_z_spectral,
                                      interpolation_needed)
                end
            end

            pdf.ion.norm .= reload_ion_pdf(dynamic, time_index, moments, r, z, vperp, vpa, r_range,
                                           z_range, vperp_range, vpa_range, restart_r,
                                           restart_r_spectral, restart_z,
                                           restart_z_spectral, restart_vperp,
                                           restart_vperp_spectral, restart_vpa,
                                           restart_vpa_spectral, interpolation_needed,
                                           restart_evolve_density, restart_evolve_upar,
                                           restart_evolve_p)
            boundary_distributions_io = get_group(fid, "boundary_distributions")

            boundary_distributions.pdf_rboundary_ion[:,:,:,1,:] .=
                reload_ion_boundary_pdf(boundary_distributions_io,
                                        "pdf_rboundary_ion_left", 1, moments, z, vperp,
                                        vpa, z_range, vperp_range, vpa_range, restart_z,
                                        restart_z_spectral, restart_vperp,
                                        restart_vperp_spectral, restart_vpa,
                                        restart_vpa_spectral, interpolation_needed,
                                        restart_evolve_density, restart_evolve_upar,
                                        restart_evolve_p)
            boundary_distributions.pdf_rboundary_ion[:,:,:,2,:] .=
                reload_ion_boundary_pdf(boundary_distributions_io,
                                        "pdf_rboundary_ion_right", r.n, moments, z, vperp,
                                        vpa, z_range, vperp_range, vpa_range, restart_z,
                                        restart_z_spectral, restart_vperp,
                                        restart_vperp_spectral, restart_vpa,
                                        restart_vpa_spectral, interpolation_needed,
                                        restart_evolve_density, restart_evolve_upar,
                                        restart_evolve_p)

            moments.electron.dens .= reload_electron_moment("electron_density", dynamic,
                                                            time_index, r, z, r_range,
                                                            z_range, restart_r,
                                                            restart_r_spectral, restart_z,
                                                            restart_z_spectral,
                                                            interpolation_needed)
            moments.electron.upar .= reload_electron_moment("electron_parallel_flow",
                                                            dynamic, time_index, r, z,
                                                            r_range, z_range, restart_r,
                                                            restart_r_spectral, restart_z,
                                                            restart_z_spectral,
                                                            interpolation_needed)
            moments.electron.p .= reload_electron_moment("electron_pressure",
                                                         dynamic, time_index, r, z,
                                                         r_range, z_range, restart_r,
                                                         restart_r_spectral, restart_z,
                                                         restart_z_spectral,
                                                         interpolation_needed)
            moments.electron.ppar .= reload_electron_moment("electron_parallel_pressure",
                                                            dynamic, time_index, r, z,
                                                            r_range, z_range, restart_r,
                                                            restart_r_spectral, restart_z,
                                                            restart_z_spectral,
                                                            interpolation_needed)
            moments.electron.qpar .= reload_electron_moment("electron_parallel_heat_flux",
                                                            dynamic, time_index, r, z,
                                                            r_range, z_range, restart_r,
                                                            restart_r_spectral, restart_z,
                                                            restart_z_spectral,
                                                            interpolation_needed)
            moments.electron.vth .= reload_electron_moment("electron_thermal_speed",
                                                           dynamic, time_index, r, z,
                                                           r_range, z_range, restart_r,
                                                           restart_r_spectral, restart_z,
                                                           restart_z_spectral,
                                                           interpolation_needed)
            if "electron_constraints_A_coefficient" ∈ keys(dynamic)
                moments.electron.constraints_A_coefficient .=
                    reload_electron_moment("electron_constraints_A_coefficient", dynamic,
                                           time_index, r, z, r_range, z_range, restart_r,
                                           restart_r_spectral, restart_z,
                                           restart_z_spectral, interpolation_needed)
            else
                moments.electron.constraints_A_coefficient .= 0.0
            end
            if "electron_constraints_B_coefficient" ∈ keys(dynamic)
                moments.electron.constraints_B_coefficient .=
                    reload_electron_moment("electron_constraints_B_coefficient", dynamic,
                                           time_index, r, z, r_range, z_range, restart_r,
                                           restart_r_spectral, restart_z,
                                           restart_z_spectral, interpolation_needed)
            else
                moments.electron.constraints_B_coefficient .= 0.0
            end
            if "electron_constraints_C_coefficient" ∈ keys(dynamic)
                moments.electron.constraints_C_coefficient .=
                    reload_electron_moment("electron_constraints_C_coefficient", dynamic,
                                           time_index, r, z, r_range, z_range, restart_r,
                                           restart_r_spectral, restart_z,
                                           restart_z_spectral, interpolation_needed)
            else
                moments.electron.constraints_C_coefficient .= 0.0
            end

            # For now, electrons are always fully moment_kinetic
            restart_electron_evolve_density, restart_electron_evolve_upar,
                restart_electron_evolve_p = true, true, true
            electron_evolve_density, electron_evolve_upar, electron_evolve_p =
                true, true, true
            # Input is written to output files with all defaults filled in, and
            # restart_input is read from a previous output file.
            # restart_input["composition"]["electron_physics"] should always exist, even
            # if it was set from a default, so we do not have to check the keys to see
            # whether it exists.
            restart_electron_physics = enum_from_string(electron_physics_type,
                                                        restart_input["composition"]["electron_physics"])
            if pdf.electron !== nothing &&
                    restart_electron_physics ∈ (kinetic_electrons,
                                                kinetic_electrons_with_temperature_equation)
                pdf.electron.norm .=
                    reload_electron_pdf(dynamic, time_index, moments, r, z, vperp, vpa,
                                        r_range, z_range, vperp_range, vpa_range,
                                        restart_r, restart_r_spectral, restart_z,
                                        restart_z_spectral, restart_vperp,
                                        restart_vperp_spectral, restart_vpa,
                                        restart_vpa_spectral, interpolation_needed,
                                        restart_evolve_density, restart_evolve_upar,
                                        restart_evolve_p)
            elseif pdf.electron !== nothing
                # The electron distribution function will be initialized later
                pdf.electron.norm .= 0.0
            end

            if composition.n_neutral_species > 0
                moments.neutral.dens .= reload_moment("density_neutral", dynamic,
                                                      time_index, r, z, r_range, z_range,
                                                      restart_r, restart_r_spectral,
                                                      restart_z, restart_z_spectral,
                                                      interpolation_needed)
                moments.neutral.uz .= reload_moment("uz_neutral", dynamic, time_index, r,
                                                    z, r_range, z_range, restart_r,
                                                    restart_r_spectral, restart_z,
                                                    restart_z_spectral,
                                                    interpolation_needed)
                moments.neutral.p .= reload_moment("p_neutral", dynamic, time_index, r,
                                                    z, r_range, z_range, restart_r,
                                                    restart_r_spectral, restart_z,
                                                    restart_z_spectral,
                                                    interpolation_needed)
                moments.neutral.pz .= reload_moment("pz_neutral", dynamic, time_index, r,
                                                    z, r_range, z_range, restart_r,
                                                    restart_r_spectral, restart_z,
                                                    restart_z_spectral,
                                                    interpolation_needed)
                moments.neutral.qz .= reload_moment("qz_neutral", dynamic, time_index, r,
                                                    z, r_range, z_range, restart_r,
                                                    restart_r_spectral, restart_z,
                                                    restart_z_spectral,
                                                    interpolation_needed)
                moments.neutral.vth .= reload_moment("thermal_speed_neutral", dynamic,
                                                     time_index, r, z, r_range, z_range,
                                                     restart_r, restart_r_spectral,
                                                     restart_z, restart_z_spectral,
                                                     interpolation_needed)
                if moments.evolve_density || moments.evolve_upar || moments.evolve_p
                    if "neutral_constraints_A_coefficient" ∈ keys(dynamic)
                        moments.neutral.constraints_A_coefficient .=
                            reload_moment("neutral_constraints_A_coefficient", dynamic,
                                          time_index, r, z, r_range, z_range, restart_r,
                                          restart_r_spectral, restart_z, restart_z_spectral,
                                          interpolation_needed)
                    elseif moments.neutral.constraints_A_coefficient !== nothing
                        moments.neutral.constraints_A_coefficient .= 0.0
                    end
                    if "neutral_constraints_B_coefficient" ∈ keys(dynamic)
                        moments.neutral.constraints_B_coefficient .=
                            reload_moment("neutral_constraints_B_coefficient", dynamic,
                                          time_index, r, z, r_range, z_range, restart_r,
                                          restart_r_spectral, restart_z, restart_z_spectral,
                                          interpolation_needed)
                    elseif moments.neutral.constraints_B_coefficient !== nothing
                        moments.neutral.constraints_B_coefficient .= 0.0
                    end
                    if "neutral_constraints_C_coefficient" ∈ keys(dynamic)
                        moments.neutral.constraints_C_coefficient .=
                            reload_moment("neutral_constraints_C_coefficient", dynamic,
                                          time_index, r, z, r_range, z_range, restart_r,
                                          restart_r_spectral, restart_z, restart_z_spectral,
                                          interpolation_needed)
                    elseif moments.neutral.constraints_C_coefficient !== nothing
                        moments.neutral.constraints_C_coefficient .= 0.0
                    end
                end

                if "external_source_neutral_controller_integral" ∈ get_variable_keys(dynamic) &&
                        length(moments.neutral.external_source_controller_integral) == 1
                    moments.neutral.external_source_controller_integral .=
                        load_slice(dynamic,
                                   "external_source_neutral_controller_integral",
                                   time_index)
                elseif length(moments.neutral.external_source_controller_integral) > 1
                    moments.neutral.external_source_controller_integral .=
                        reload_moment("external_source_neutral_controller_integral",
                                      dynamic, time_index, r, z, r_range, z_range,
                                      restart_r, restart_r_spectral, restart_z,
                                      restart_z_spectral, interpolation_needed)
                end

                pdf.neutral.norm .=
                    reload_neutral_pdf(dynamic, time_index, moments, r, z, vzeta, vr, vz,
                                       r_range, z_range, vzeta_range, vr_range, vz_range,
                                       restart_r, restart_r_spectral, restart_z,
                                       restart_z_spectral, restart_vzeta,
                                       restart_vzeta_spectral, restart_vr,
                                       restart_vr_spectral, restart_vz,
                                       restart_vz_spectral, interpolation_needed,
                                       restart_evolve_density, restart_evolve_upar,
                                       restart_evolve_p)

                boundary_distributions.pdf_rboundary_neutral[:,:,:,:,1,:] .=
                    reload_neutral_boundary_pdf(boundary_distributions_io,
                                                "pdf_rboundary_neutral_left", 1, moments,
                                                z, vzeta, vr, vz, z_range, vzeta_range,
                                                vr_range, vz_range, restart_z,
                                                restart_z_spectral, restart_vzeta,
                                                restart_vzeta_spectral, restart_vr,
                                                restart_vr_spectral, restart_vz,
                                                restart_vz_spectral, interpolation_needed,
                                                restart_evolve_density,
                                                restart_evolve_upar, restart_evolve_p)
                boundary_distributions.pdf_rboundary_neutral[:,:,:,:,2,:] .=
                    reload_neutral_boundary_pdf(boundary_distributions_io,
                                                "pdf_rboundary_neutral_right", r.n,
                                                moments, z, vzeta, vr, vz, z_range,
                                                vzeta_range, vr_range, vz_range,
                                                restart_z, restart_z_spectral,
                                                restart_vzeta, restart_vzeta_spectral,
                                                restart_vr, restart_vr_spectral,
                                                restart_vz, restart_vz_spectral,
                                                interpolation_needed,
                                                restart_evolve_density,
                                                restart_evolve_upar, restart_evolve_p)
            end

            if "dt" ∈ keys(dynamic)
                # If "dt" is not present, the file being restarted from is an older
                # one that did not have an adaptive timestep, so just leave the value
                # of "dt" from the input file.
                dt[] = load_slice(dynamic, "dt", time_index)
            end
            if "dt_before_last_fail" ∈ keys(dynamic)
                # If "dt_before_last_fail" is not present, the file being
                # restarted from is an older one that did not have an adaptive
                # timestep, so just leave the value of "dt_before_last_fail" from
                # the input file.
                dt_before_last_fail[] = load_slice(dynamic, "dt_before_last_fail",
                                                time_index)
            end
            if "electron_dt" ∈ keys(dynamic)
                # The algorithm for electron pseudo-timestepping actually starts each
                # solve using t_params.electron.previous_dt[], so "electron_previous_dt"
                # is the thing to load.
                electron_dt[] = load_slice(dynamic, "electron_previous_dt", time_index)
            end
            if "electron_dt_before_last_fail" ∈ keys(dynamic)
                electron_dt_before_last_fail[] =
                    load_slice(dynamic, "electron_dt_before_last_fail", time_index)
            end
        finally
            close(fid)
        end
    end
    moments.ion.dens_updated .= true
    moments.ion.upar_updated .= true
    moments.ion.p_updated .= true
    moments.ion.qpar_updated .= true
    moments.electron.dens_updated[] = true
    moments.electron.upar_updated[] = true
    moments.electron.p_updated[] = true
    moments.electron.qpar_updated[] = true
    moments.neutral.dens_updated .= true
    moments.neutral.uz_updated .= true
    moments.neutral.p_updated .= true
    moments.neutral.qz_updated .= true

    restart_electron_physics = MPI.bcast(restart_electron_physics, 0, comm_block[])
    MPI.Bcast!(dt, comm_block[])
    MPI.Bcast!(dt_before_last_fail, comm_block[])
    MPI.Bcast!(electron_dt, comm_block[])
    MPI.Bcast!(electron_dt_before_last_fail, comm_block[])

    if dt[] == -Inf
        dt = nothing
    else
        dt = dt[]
    end
    if electron_dt[] == -Inf
        electron_dt = nothing
    else
        electron_dt = electron_dt[]
    end

    return code_time, dt, dt_before_last_fail[], electron_dt,
           electron_dt_before_last_fail[], previous_runs_info, time_index,
           restart_electron_physics
end

"""
Reload electron pdf and moments from an existing output file.
"""
function reload_electron_data!(pdf, moments, t_params, restart_prefix_iblock, time_index,
                               geometry, r, z, vpa, vperp, vzeta, vr, vz)
    code_time = Ref(0.0)
    pdf_electron_converged = Ref(false)
    previous_runs_info = nothing
    @begin_serial_region()
    @serial_region begin
        fid = open_readonly_output_file(restart_prefix_iblock[1], "initial_electron";
                                        iblock=restart_prefix_iblock[2])
        try # finally to make sure to close file0
            overview = get_group(fid, "overview")
            dynamic = get_group(fid, "dynamic_data")
            parallel_io = load_variable(overview, "parallel_io")
            if time_index < 0
                time_index, _, _ = load_time_data(fid)
            end
            #restart_evolve_density, restart_evolve_upar, restart_evolve_p =
            #    load_mk_options(fid)
            # For now, electrons are always fully moment_kinetic
            restart_evolve_density, restart_evolve_upar, restart_evolve_p = true, true,
                                                                               true
            evolve_density, evolve_upar, evolve_p = true, true, true

            previous_runs_info = load_run_info_history(fid)

            restart_n_ion_species, restart_n_neutral_species = load_species_data(fid)
            restart_r, restart_r_spectral, restart_z, restart_z_spectral, restart_vperp,
                restart_vperp_spectral, restart_vpa, restart_vpa_spectral, restart_vzeta,
                restart_vzeta_spectral, restart_vr,restart_vr_spectral, restart_vz,
                restart_vz_spectral = load_restart_coordinates(fid, r, z, vperp, vpa,
                                                               vzeta, vr, vz, parallel_io)

            # Test whether any interpolation is needed
            interpolation_needed = OrderedDict(
                x.name => x.n != restart_x.n || !all(isapprox.(x.grid, restart_x.grid))
                for (x, restart_x) ∈ ((z, restart_z), (r, restart_r),
                                      (vperp, restart_vperp), (vpa, restart_vpa)))

            code_time[] = load_slice(dynamic, "time", time_index)

            pdf_electron_converged[] = get_attribute(fid, "pdf_electron_converged")

            r_range, z_range, vperp_range, vpa_range, vzeta_range, vr_range, vz_range =
                get_reload_ranges(parallel_io, restart_r, restart_z, restart_vperp,
                                  restart_vpa, restart_vzeta, restart_vr, restart_vz)

            moments.electron.upar_updated[] = true
            moments.electron.p .=
                reload_electron_moment("electron_pressure", dynamic, time_index,
                                       r, z, r_range, z_range, restart_r,
                                       restart_r_spectral, restart_z, restart_z_spectral,
                                       interpolation_needed)
            moments.electron.p_updated[] = true
            moments.electron.ppar .=
                reload_electron_moment("electron_parallel_pressure", dynamic, time_index,
                                       r, z, r_range, z_range, restart_r,
                                       restart_r_spectral, restart_z, restart_z_spectral,
                                       interpolation_needed)
            moments.electron.qpar .=
                reload_electron_moment("electron_parallel_heat_flux", dynamic, time_index,
                                       r, z, r_range, z_range, restart_r,
                                       restart_r_spectral, restart_z, restart_z_spectral,
                                       interpolation_needed)
            moments.electron.qpar_updated[] = true
            moments.electron.vth .=
                reload_electron_moment("electron_thermal_speed", dynamic, time_index, r,
                                       z, r_range, z_range, restart_r, restart_r_spectral,
                                       restart_z, restart_z_spectral,
                                       interpolation_needed)

            pdf.electron.norm .=
                reload_electron_pdf(dynamic, time_index, moments, r, z, vperp, vpa,
                                    r_range, z_range, vperp_range, vpa_range, restart_r,
                                    restart_r_spectral, restart_z, restart_z_spectral,
                                    restart_vperp, restart_vperp_spectral, restart_vpa,
                                    restart_vpa_spectral, interpolation_needed,
                                    restart_evolve_density, restart_evolve_upar,
                                    restart_evolve_p)

            new_dt = load_slice(dynamic, "electron_dt", time_index)
            if new_dt > 0.0
                # if the reloaded electron_dt was 0.0, then the previous run would not
                # have been using kinetic electrons, so we only use the value if it is
                # positive
                t_params.dt[] = new_dt
            end
            t_params.dt_before_last_fail[] =
                load_slice(dynamic, "electron_dt_before_last_fail", time_index)
        finally
            close(fid)
        end
    end

    # Broadcast dt, dt_before_last_fail, code_time and pdf_electron_converged from the
    # root process of each shared-memory block (on which it might have been loaded from a
    # restart file).
    MPI.Bcast!(t_params.dt, comm_block[]; root=0)
    MPI.Bcast!(t_params.dt_before_last_fail, comm_block[]; root=0)
    MPI.Bcast!(code_time, comm_block[]; root=0)
    MPI.Bcast!(pdf_electron_converged, comm_block[]; root=0)

    t_params.previous_dt[] = t_params.dt[]
    t_params.dt_before_output[] = t_params.dt[]

    return code_time[], pdf_electron_converged[], previous_runs_info, time_index
end

function load_restart_coordinates(fid, r, z, vperp, vpa, vzeta, vr, vz, parallel_io)
    if parallel_io
        restart_z, restart_z_spectral, _ =
            load_coordinate_data(fid, "z"; irank=z.irank, nrank=z.nrank)
        restart_r, restart_r_spectral, _ =
            load_coordinate_data(fid, "r"; irank=r.irank, nrank=r.nrank)
        restart_vperp, restart_vperp_spectral, _ =
            load_coordinate_data(fid, "vperp"; irank=vperp.irank, nrank=vperp.nrank)
        restart_vpa, restart_vpa_spectral, _ =
            load_coordinate_data(fid, "vpa"; irank=vpa.irank, nrank=vpa.nrank)
        restart_vzeta, restart_vzeta_spectral, _ =
            load_coordinate_data(fid, "vzeta"; irank=vzeta.irank, nrank=vzeta.nrank)
        restart_vr, restart_vr_spectral, _ =
            load_coordinate_data(fid, "vr"; irank=vr.irank, nrank=vr.nrank)
        restart_vz, restart_vz_spectral, _ =
            load_coordinate_data(fid, "vz"; irank=vz.irank, nrank=vz.nrank)
    else
        restart_z, restart_z_spectral, _ =
            load_coordinate_data(fid, "z")
        restart_r, restart_r_spectral, _ =
            load_coordinate_data(fid, "r")
        restart_vperp, restart_vperp_spectral, _ =
            load_coordinate_data(fid, "vperp")
        restart_vpa, restart_vpa_spectral, _ =
            load_coordinate_data(fid, "vpa")
        restart_vzeta, restart_vzeta_spectral, _ =
            load_coordinate_data(fid, "vzeta")
        restart_vr, restart_vr_spectral, _ =
            load_coordinate_data(fid, "vr")
        restart_vz, restart_vz_spectral, _ =
            load_coordinate_data(fid, "vz")

        if restart_r.nrank != r.nrank
            error("Not using parallel I/O, and distributed MPI layout has "
                  * "changed: now r.nrank=$(r.nrank), but we are trying to "
                  * "restart from files ith restart_r.nrank=$(restart_r.nrank).")
        end
        if restart_z.nrank != z.nrank
            error("Not using parallel I/O, and distributed MPI layout has "
                  * "changed: now z.nrank=$(z.nrank), but we are trying to "
                  * "restart from files ith restart_z.nrank=$(restart_z.nrank).")
        end
        if restart_vperp.nrank != vperp.nrank
            error("Not using parallel I/O, and distributed MPI layout has "
                  * "changed: now vperp.nrank=$(vperp.nrank), but we are trying to "
                  * "restart from files ith restart_vperp.nrank=$(restart_vperp.nrank).")
        end
        if restart_vpa.nrank != vpa.nrank
            error("Not using parallel I/O, and distributed MPI layout has "
                  * "changed: now vpa.nrank=$(vpa.nrank), but we are trying to "
                  * "restart from files ith restart_vpa.nrank=$(restart_vpa.nrank).")
        end
        if restart_vzeta.nrank != vzeta.nrank
            error("Not using parallel I/O, and distributed MPI layout has "
                  * "changed: now vzeta.nrank=$(vzeta.nrank), but we are trying to "
                  * "restart from files ith restart_vzeta.nrank=$(restart_vzeta.nrank).")
        end
        if restart_vr.nrank != vr.nrank
            error("Not using parallel I/O, and distributed MPI layout has "
                  * "changed: now vr.nrank=$(vr.nrank), but we are trying to "
                  * "restart from files ith restart_vr.nrank=$(restart_vr.nrank).")
        end
        if restart_vz.nrank != vz.nrank
            error("Not using parallel I/O, and distributed MPI layout has "
                  * "changed: now vz.nrank=$(vz.nrank), but we are trying to "
                  * "restart from files ith restart_vz.nrank=$(restart_vz.nrank).")
        end
    end

    return restart_r, restart_r_spectral, restart_z, restart_z_spectral, restart_vperp,
           restart_vperp_spectral, restart_vpa, restart_vpa_spectral, restart_vzeta,
           restart_vzeta_spectral, restart_vr,restart_vr_spectral, restart_vz,
           restart_vz_spectral
end

function get_reload_ranges(parallel_io, restart_r, restart_z, restart_vperp, restart_vpa,
                           restart_vzeta, restart_vr, restart_vz)
    if parallel_io
        function get_range(coord)
            if coord === nothing
                return 1:0
            elseif coord.irank == coord.nrank - 1
                return coord.global_io_range
            else
                # Need to modify the range to load the end-point that is duplicated on
                # the next process
                this_range = coord.global_io_range
                return this_range.start:(this_range.stop+1)
            end
        end
        r_range = get_range(restart_r)
        z_range = get_range(restart_z)
        vperp_range = get_range(restart_vperp)
        vpa_range = get_range(restart_vpa)
        vzeta_range = get_range(restart_vzeta)
        vr_range = get_range(restart_vr)
        vz_range = get_range(restart_vz)
    else
        r_range = (:)
        z_range = (:)
        vperp_range = (:)
        vpa_range = (:)
        vzeta_range = (:)
        vr_range = (:)
        vz_range = (:)
    end
    return r_range, z_range, vperp_range, vpa_range, vzeta_range, vr_range, vz_range
end

function reload_moment(var_name, dynamic, time_index, r, z, r_range, z_range, restart_r,
                       restart_r_spectral, restart_z, restart_z_spectral,
                       interpolation_needed)
    moment = load_slice(dynamic, var_name, z_range, r_range, :, time_index)
    orig_nz, orig_nr, nspecies = size(moment)
    if interpolation_needed["r"]
        new_moment = allocate_float(orig_nz, r.n, nspecies)
        for is ∈ 1:nspecies, iz ∈ 1:orig_nz
            @views interpolate_to_grid_1d!(new_moment[iz,:,is], r.grid,
                                           moment[iz,:,is], restart_r,
                                           restart_r_spectral)
        end
        moment = new_moment
    end
    if interpolation_needed["z"]
        new_moment = allocate_float(z.n, r.n, nspecies)
        for is ∈ 1:nspecies, ir ∈ 1:r.n
            @views interpolate_to_grid_1d!(new_moment[:,ir,is], z.grid,
                                           moment[:,ir,is], restart_z,
                                           restart_z_spectral)
        end
        moment = new_moment
    end
    return moment
end

function reload_electron_moment(var_name, dynamic, time_index, r, z, r_range, z_range,
                                restart_r, restart_r_spectral, restart_z,
                                restart_z_spectral, interpolation_needed)
    moment = load_slice(dynamic, var_name, z_range, r_range, time_index)
    orig_nz, orig_nr = size(moment)
    if interpolation_needed["r"]
        new_moment = allocate_float(orig_nz, r.n)
        for iz ∈ 1:orig_nz
            @views interpolate_to_grid_1d!(new_moment[iz,:], r.grid, moment[iz,:],
                                           restart_r, restart_r_spectral)
        end
        moment = new_moment
    end
    if interpolation_needed["z"]
        new_moment = allocate_float(z.n, r.n)
        for ir ∈ 1:r.n
            @views interpolate_to_grid_1d!(new_moment[:,ir], z.grid, moment[:,ir],
                                           restart_z, restart_z_spectral)
        end
        moment = new_moment
    end
    return moment
end

function reload_ion_pdf(dynamic, time_index, moments, r, z, vperp, vpa, r_range, z_range,
                        vperp_range, vpa_range, restart_r, restart_r_spectral, restart_z,
                        restart_z_spectral, restart_vperp, restart_vperp_spectral,
                        restart_vpa, restart_vpa_spectral, interpolation_needed,
                        restart_evolve_density, restart_evolve_upar, restart_evolve_p)

    this_pdf = load_slice(dynamic, "f", vpa_range, vperp_range, z_range,
        r_range, :, time_index)
    orig_nvpa, orig_nvperp, orig_nz, orig_nr, nspecies = size(this_pdf)
    if interpolation_needed["r"]
        new_pdf = allocate_float(orig_nvpa, orig_nvperp, orig_nz, r.n, nspecies)
        for is ∈ 1:nspecies, iz ∈ 1:orig_nz, ivperp ∈ 1:orig_nvperp,
            ivpa ∈ 1:orig_nvpa
            @views interpolate_to_grid_1d!(
                       new_pdf[ivpa,ivperp,iz,:,is], r.grid,
                       this_pdf[ivpa,ivperp,iz,:,is], restart_r,
                       restart_r_spectral)
        end
        this_pdf = new_pdf
    end
    if interpolation_needed["z"]
        new_pdf = allocate_float(orig_nvpa, orig_nvperp, z.n, r.n, nspecies)
        for is ∈ 1:nspecies, ir ∈ 1:r.n, ivperp ∈ 1:orig_nvperp,
            ivpa ∈ 1:orig_nvpa
            @views interpolate_to_grid_1d!(
                       new_pdf[ivpa,ivperp,:,ir,is], z.grid,
                       this_pdf[ivpa,ivperp,:,ir,is], restart_z,
                       restart_z_spectral)
        end
        this_pdf = new_pdf
    end

    # Current moment-kinetic implementation is only 1V, so no need to handle a
    # normalised vperp coordinate. This will need to change when 2V
    # moment-kinetics is implemented.
    if interpolation_needed["vperp"]
        new_pdf = allocate_float(orig_nvpa, vperp.n, z.n, r.n, nspecies)
        for is ∈ 1:nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n, ivpa ∈ 1:orig_nvpa
            @views interpolate_to_grid_1d!(
                       new_pdf[ivpa,:,iz,ir,is], vperp.grid,
                       this_pdf[ivpa,:,iz,ir,is], restart_vperp,
                       restart_vperp_spectral)
        end
        this_pdf = new_pdf
    end

    if (
        (moments.evolve_density == restart_evolve_density &&
         moments.evolve_upar == restart_evolve_upar && moments.evolve_p ==
         restart_evolve_p)
        || (!moments.evolve_upar && !restart_evolve_upar &&
            !moments.evolve_p && !restart_evolve_p)
       )
        # No chages to velocity-space coordinates, so just interpolate from
        # one grid to the other
        if interpolation_needed["vpa"]
            new_pdf = allocate_float(vpa.n, vperp.n, z.n, r.n, nspecies)
            for is ∈ 1:nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
                @views interpolate_to_grid_1d!(
                           new_pdf[:,ivperp,iz,ir,is], vpa.grid,
                           this_pdf[:,ivperp,iz,ir,is], restart_vpa,
                           restart_vpa_spectral)
            end
            this_pdf = new_pdf
        end
    elseif (!moments.evolve_upar && !moments.evolve_p &&
            restart_evolve_upar && !restart_evolve_p)
        # vpa = new_wpa = old_wpa + upar
        # => old_wpa = new_wpa - upar
        new_pdf = allocate_float(vpa.n, vperp.n, z.n, r.n, nspecies)
        for is ∈ nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
            restart_vpa_vals = vpa.grid .- moments.ion.upar[iz,ir,is]
            @views interpolate_to_grid_1d!(
                       new_pdf[:,ivperp,iz,ir,is], restart_vpa_vals,
                       this_pdf[:,ivperp,iz,ir,is], restart_vpa,
                       restart_vpa_spectral)
        end
        this_pdf = new_pdf
    elseif (!moments.evolve_upar && !moments.evolve_p &&
            !restart_evolve_upar && restart_evolve_p)
        # vpa = new_wpa = old_wpa*vth
        # => old_wpa = new_wpa/vth
        new_pdf = allocate_float(vpa.n, vperp.n, z.n, r.n, nspecies)
        for is ∈ nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
            restart_vpa_vals = vpa.grid ./ moments.ion.vth[iz,ir,is]
            @views interpolate_to_grid_1d!(
                       new_pdf[:,ivperp,iz,ir,is], restart_vpa_vals,
                       this_pdf[:,ivperp,iz,ir,is], restart_vpa,
                       restart_vpa_spectral)
        end
        this_pdf = new_pdf
    elseif (!moments.evolve_upar && !moments.evolve_p &&
            restart_evolve_upar && restart_evolve_p)
        # vpa = new_wpa = old_wpa*vth + upar
        # => old_wpa = (new_wpa - upar)/vth
        new_pdf = allocate_float(vpa.n, vperp.n, z.n, r.n, nspecies)
        for is ∈ nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
            restart_vpa_vals =
            @. (vpa.grid - moments.ion.upar[iz,ir,is]) / moments.ion.vth[iz,ir,is]
            @views interpolate_to_grid_1d!(
                       new_pdf[:,ivperp,iz,ir,is], restart_vpa_vals,
                       this_pdf[:,ivperp,iz,ir,is], restart_vpa,
                       restart_vpa_spectral)
        end
        this_pdf = new_pdf
    elseif (moments.evolve_upar && !moments.evolve_p &&
            !restart_evolve_upar && !restart_evolve_p)
        # vpa = new_wpa + upar = old_wpa
        # => old_wpa = new_wpa + upar
        new_pdf = allocate_float(vpa.n, vperp.n, z.n, r.n, nspecies)
        for is ∈ nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
            restart_vpa_vals = vpa.grid .+ moments.ion.upar[iz,ir,is]
            @views interpolate_to_grid_1d!(
                       new_pdf[:,ivperp,iz,ir,is], restart_vpa_vals,
                       this_pdf[:,ivperp,iz,ir,is], restart_vpa,
                       restart_vpa_spectral)
        end
        this_pdf = new_pdf
    elseif (moments.evolve_upar && !moments.evolve_p &&
            !restart_evolve_upar && restart_evolve_p)
        # vpa = new_wpa + upar = old_wpa*vth
        # => old_wpa = (new_wpa + upar)/vth
        new_pdf = allocate_float(vpa.n, vperp.n, z.n, r.n, nspecies)
        for is ∈ nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
            restart_vpa_vals =
            @. (vpa.grid + moments.ion.upar[iz,ir,is]) / moments.ion.vth[iz,ir,is]
            @views interpolate_to_grid_1d!(
                       new_pdf[:,ivperp,iz,ir,is], restart_vpa_vals,
                       this_pdf[:,ivperp,iz,ir,is], restart_vpa,
                       restart_vpa_spectral)
        end
        this_pdf = new_pdf
    elseif (moments.evolve_upar && !moments.evolve_p &&
            restart_evolve_upar && restart_evolve_p)
        # vpa = new_wpa + upar = old_wpa*vth + upar
        # => old_wpa = new_wpa/vth
        new_pdf = allocate_float(vpa.n, vperp.n, z.n, r.n, nspecies)
        for is ∈ nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
            restart_vpa_vals = vpa.grid ./ moments.ion.vth[iz,ir,is]
            @views interpolate_to_grid_1d!(
                       new_pdf[:,ivperp,iz,ir,is], restart_vpa_vals,
                       this_pdf[:,ivperp,iz,ir,is], restart_vpa,
                       restart_vpa_spectral)
        end
        this_pdf = new_pdf
    elseif (!moments.evolve_upar && moments.evolve_p &&
            !restart_evolve_upar && !restart_evolve_p)
        # vpa = new_wpa*vth = old_wpa
        # => old_wpa = new_wpa*vth
        new_pdf = allocate_float(vpa.n, vperp.n, z.n, r.n, nspecies)
        for is ∈ nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
            restart_vpa_vals = vpa.grid .* moments.ion.vth[iz,ir,is]
            @views interpolate_to_grid_1d!(
                       new_pdf[:,ivperp,iz,ir,is], restart_vpa_vals,
                       this_pdf[:,ivperp,iz,ir,is], restart_vpa,
                       restart_vpa_spectral)
        end
        this_pdf = new_pdf
    elseif (!moments.evolve_upar && moments.evolve_p &&
            restart_evolve_upar && !restart_evolve_p)
        # vpa = new_wpa*vth = old_wpa + upar
        # => old_wpa = new_wpa*vth - upar/vth
        new_pdf = allocate_float(vpa.n, vperp.n, z.n, r.n, nspecies)
        for is ∈ nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
            restart_vpa_vals = @. vpa.grid * moments.ion.vth[iz,ir,is] - moments.ion.upar[iz,ir,is]
            @views interpolate_to_grid_1d!(
                       new_pdf[:,ivperp,iz,ir,is], restart_vpa_vals,
                       this_pdf[:,ivperp,iz,ir,is], restart_vpa,
                       restart_vpa_spectral)
        end
        this_pdf = new_pdf
    elseif (!moments.evolve_upar && moments.evolve_p &&
            restart_evolve_upar && restart_evolve_p)
        # vpa = new_wpa*vth = old_wpa*vth + upar
        # => old_wpa = new_wpa - upar/vth
        new_pdf = allocate_float(vpa.n, vperp.n, z.n, r.n, nspecies)
        for is ∈ nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
            restart_vpa_vals =
            @. vpa.grid - moments.ion.upar[iz,ir,is]/moments.ion.vth[iz,ir,is]
            @views interpolate_to_grid_1d!(
                       new_pdf[:,ivperp,iz,ir,is], restart_vpa_vals,
                       this_pdf[:,ivperp,iz,ir,is], restart_vpa,
                       restart_vpa_spectral)
        end
        this_pdf = new_pdf
    elseif (moments.evolve_upar && moments.evolve_p &&
            !restart_evolve_upar && !restart_evolve_p)
        # vpa = new_wpa*vth + upar = old_wpa
        # => old_wpa = new_wpa*vth + upar
        new_pdf = allocate_float(vpa.n, vperp.n, z.n, r.n, nspecies)
        for is ∈ nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
            restart_vpa_vals = @. vpa.grid * moments.ion.vth[iz,ir,is] + moments.ion.upar[iz,ir,is]
            @views interpolate_to_grid_1d!(
                       new_pdf[:,ivperp,iz,ir,is], restart_vpa_vals,
                       this_pdf[:,ivperp,iz,ir,is], restart_vpa,
                       restart_vpa_spectral)
        end
        this_pdf = new_pdf
    elseif (moments.evolve_upar && moments.evolve_p &&
            restart_evolve_upar && !restart_evolve_p)
        # vpa = new_wpa*vth + upar = old_wpa + upar
        # => old_wpa = new_wpa*vth
        new_pdf = allocate_float(vpa.n, vperp.n, z.n, r.n, nspecies)
        for is ∈ nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
            restart_vpa_vals = vpa.grid .* moments.ion.vth[iz,ir,is]
            @views interpolate_to_grid_1d!(
                       new_pdf[:,ivperp,iz,ir,is], restart_vpa_vals,
                       this_pdf[:,ivperp,iz,ir,is], restart_vpa,
                       restart_vpa_spectral)
        end
        this_pdf = new_pdf
    elseif (moments.evolve_upar && moments.evolve_p &&
            !restart_evolve_upar && restart_evolve_p)
        # vpa = new_wpa*vth + upar = old_wpa*vth
        # => old_wpa = new_wpa + upar/vth
        new_pdf = allocate_float(vpa.n, vperp.n, z.n, r.n, nspecies)
        for is ∈ nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
            restart_vpa_vals =
            @. vpa.grid + moments.ion.upar[iz,ir,is] / moments.ion.vth[iz,ir,is]
            @views interpolate_to_grid_1d!(
                       new_pdf[:,ivperp,iz,ir,is], restart_vpa_vals,
                       this_pdf[:,ivperp,iz,ir,is], restart_vpa,
                       restart_vpa_spectral)
        end
        this_pdf = new_pdf
    else
        # This should never happen, as all combinations of evolve_* options
        # should be handled above.
        error("Unsupported combination of moment-kinetic options:"
              * " evolve_density=", moments.evolve_density
              * " evolve_upar=", moments.evolve_upar
              * " evolve_p=", moments.evolve_p
              * " restart_evolve_density=", restart_evolve_density
              * " restart_evolve_upar=", restart_evolve_upar
              * " restart_evolve_p=", restart_evolve_p)
    end
    if moments.evolve_density && !restart_evolve_density
        # Need to normalise by density
        for is ∈ nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n
            this_pdf[:,:,iz,ir,is] ./= moments.ion.dens[iz,ir,is]
        end
    elseif !moments.evolve_density && restart_evolve_density
        # Need to unnormalise by density
        for is ∈ nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n
            this_pdf[:,:,iz,ir,is] .*= moments.ion.dens[iz,ir,is]
        end
    end
    if moments.evolve_p && !restart_evolve_p
        # Need to normalise by vth
        for is ∈ nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n
            this_pdf[:,:,iz,ir,is] .*= moments.ion.vth[iz,ir,is]
        end
    elseif !moments.evolve_p && restart_evolve_p
        # Need to unnormalise by vth
        for is ∈ nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n
            this_pdf[:,:,iz,ir,is] ./= moments.ion.vth[iz,ir,is]
        end
    end

    return this_pdf
end

function reload_ion_boundary_pdf(boundary_distributions_io, var_name, ir, moments, z,
                                 vperp, vpa, z_range, vperp_range, vpa_range, restart_z,
                                 restart_z_spectral, restart_vperp,
                                 restart_vperp_spectral, restart_vpa,
                                 restart_vpa_spectral, interpolation_needed,
                                 restart_evolve_density, restart_evolve_upar,
                                 restart_evolve_p)
    this_pdf = load_slice(boundary_distributions_io, var_name, vpa_range,
        vperp_range, z_range, :)
    orig_nvpa, orig_nvperp, orig_nz, nspecies = size(this_pdf)
    if interpolation_needed["z"]
        new_pdf = allocate_float(orig_nvpa, orig_nvperp, z.n, nspecies)
        for is ∈ 1:nspecies, ivperp ∈ 1:orig_nvperp,
            ivpa ∈ 1:orig_nvpa
            @views interpolate_to_grid_1d!(
                                           new_pdf[ivpa,ivperp,:,is], z.grid,
                                           this_pdf[ivpa,ivperp,:,is], restart_z,
                                           restart_z_spectral)
        end
        this_pdf = new_pdf
    end

    # Current moment-kinetic implementation is only 1V, so no need to handle a
    # normalised vperp coordinate. This will need to change when 2V
    # moment-kinetics is implemented.
    if interpolation_needed["vperp"]
        new_pdf = allocate_float(orig_nvpa, vperp.n, z.n, nspecies)
        for is ∈ 1:nspecies, iz ∈ 1:z.n, ivpa ∈ 1:orig_nvpa
            @views interpolate_to_grid_1d!(
                                           new_pdf[ivpa,:,iz,is], vperp.grid,
                                           this_pdf[ivpa,:,iz,is], restart_vperp,
                                           restart_vperp_spectral)
        end
        this_pdf = new_pdf
    end

    if (
        (moments.evolve_density == restart_evolve_density &&
         moments.evolve_upar == restart_evolve_upar && moments.evolve_p ==
         restart_evolve_p)
        || (!moments.evolve_upar && !restart_evolve_upar &&
            !moments.evolve_p && !restart_evolve_p)
       )
        # No chages to velocity-space coordinates, so just interpolate from
        # one grid to the other
        if interpolation_needed["vpa"]
            new_pdf = allocate_float(vpa.n, vperp.n, z.n, nspecies)
            for is ∈ 1:nspecies, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
                @views interpolate_to_grid_1d!(
                                               new_pdf[:,ivperp,iz,is], vpa.grid,
                                               this_pdf[:,ivperp,iz,is], restart_vpa,
                                               restart_vpa_spectral)
            end
            this_pdf = new_pdf
        end
    elseif (!moments.evolve_upar && !moments.evolve_p &&
            restart_evolve_upar && !restart_evolve_p)
        # vpa = new_wpa = old_wpa + upar
        # => old_wpa = new_wpa - upar
        new_pdf = allocate_float(vpa.n, vperp.n, z.n, nspecies)
        for is ∈ nspecies, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
            restart_vpa_vals = vpa.grid .- moments.ion.upar[iz,ir,is]
            @views interpolate_to_grid_1d!(
                                           new_pdf[:,ivperp,iz,is], restart_vpa_vals,
                                           this_pdf[:,ivperp,iz,is], restart_vpa, restart_vpa_spectral)
        end
        this_pdf = new_pdf
    elseif (!moments.evolve_upar && !moments.evolve_p &&
            !restart_evolve_upar && restart_evolve_p)
        # vpa = new_wpa = old_wpa*vth
        # => old_wpa = new_wpa/vth
        new_pdf = allocate_float(vpa.n, vperp.n, z.n, nspecies)
        for is ∈ nspecies, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
            restart_vpa_vals = vpa.grid ./ moments.ion.vth[iz,ir,is]
            @views interpolate_to_grid_1d!(
                                           new_pdf[:,ivperp,iz,is], restart_vpa_vals,
                                           this_pdf[:,ivperp,iz,is], restart_vpa, restart_vpa_spectral)
        end
        this_pdf = new_pdf
    elseif (!moments.evolve_upar && !moments.evolve_p &&
            restart_evolve_upar && restart_evolve_p)
        # vpa = new_wpa = old_wpa*vth + upar
        # => old_wpa = (new_wpa - upar)/vth
        new_pdf = allocate_float(vpa.n, vperp.n, z.n, nspecies)
        for is ∈ nspecies, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
            restart_vpa_vals =
            @. (vpa.grid - moments.ion.upar[iz,ir,is]) /
            moments.ion.vth[iz,ir,is]
            @views interpolate_to_grid_1d!(
                                           new_pdf[:,ivperp,iz,is], restart_vpa_vals,
                                           this_pdf[:,ivperp,iz,is], restart_vpa, restart_vpa_spectral)
        end
        this_pdf = new_pdf
    elseif (moments.evolve_upar && !moments.evolve_p &&
            !restart_evolve_upar && !restart_evolve_p)
        # vpa = new_wpa + upar = old_wpa
        # => old_wpa = new_wpa + upar
        new_pdf = allocate_float(vpa.n, vperp.n, z.n, nspecies)
        for is ∈ nspecies, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
            restart_vpa_vals = vpa.grid .+ moments.ion.upar[iz,ir,is]
            @views interpolate_to_grid_1d!(
                                           new_pdf[:,ivperp,iz,is], restart_vpa_vals,
                                           this_pdf[:,ivperp,iz,is], restart_vpa, restart_vpa_spectral)
        end
        this_pdf = new_pdf
    elseif (moments.evolve_upar && !moments.evolve_p &&
            !restart_evolve_upar && restart_evolve_p)
        # vpa = new_wpa + upar = old_wpa*vth
        # => old_wpa = (new_wpa + upar)/vth
        new_pdf = allocate_float(vpa.n, vperp.n, z.n, nspecies)
        for is ∈ nspecies, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
            restart_vpa_vals =
            @. (vpa.grid + moments.ion.upar[iz,ir,is]) /
            moments.ion.vth
            @views interpolate_to_grid_1d!(
                                           new_pdf[:,ivperp,iz,is], restart_vpa_vals,
                                           this_pdf[:,ivperp,iz,is], restart_vpa, restart_vpa_spectral)
        end
        this_pdf = new_pdf
    elseif (moments.evolve_upar && !moments.evolve_p &&
            restart_evolve_upar && restart_evolve_p)
        # vpa = new_wpa + upar = old_wpa*vth + upar
        # => old_wpa = new_wpa/vth
        new_pdf = allocate_float(vpa.n, vperp.n, z.n, nspecies)
        for is ∈ nspecies, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
            restart_vpa_vals = vpa.grid ./ moments.ion.vth[iz,ir,is]
            @views interpolate_to_grid_1d!(
                                           new_pdf[:,ivperp,iz,is], restart_vpa_vals,
                                           this_pdf[:,ivperp,iz,is], restart_vpa, restart_vpa_spectral)
        end
        this_pdf = new_pdf
    elseif (!moments.evolve_upar && moments.evolve_p &&
            !restart_evolve_upar && !restart_evolve_p)
        # vpa = new_wpa*vth = old_wpa
        # => old_wpa = new_wpa*vth
        new_pdf = allocate_float(vpa.n, vperp.n, z.n, nspecies)
        for is ∈ nspecies, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
            restart_vpa_vals = vpa.grid .* moments.ion.vth[iz,ir,is]
            @views interpolate_to_grid_1d!(
                                           new_pdf[:,ivperp,iz,is], restart_vpa_vals,
                                           this_pdf[:,ivperp,iz,is], restart_vpa, restart_vpa_spectral)
        end
        this_pdf = new_pdf
    elseif (!moments.evolve_upar && moments.evolve_p &&
            restart_evolve_upar && !restart_evolve_p)
        # vpa = new_wpa*vth = old_wpa + upar
        # => old_wpa = new_wpa*vth - upar/vth
        new_pdf = allocate_float(vpa.n, vperp.n, z.n, nspecies)
        for is ∈ nspecies, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
            restart_vpa_vals = @. vpa.grid * moments.ion.vth[iz,ir,is] -
            moments.ion.upar[iz,ir,is]
            @views interpolate_to_grid_1d!(
                                           new_pdf[:,ivperp,iz,is], restart_vpa_vals,
                                           this_pdf[:,ivperp,iz,is], restart_vpa, restart_vpa_spectral)
        end
        this_pdf = new_pdf
    elseif (!moments.evolve_upar && moments.evolve_p &&
            restart_evolve_upar && restart_evolve_p)
        # vpa = new_wpa*vth = old_wpa*vth + upar
        # => old_wpa = new_wpa - upar/vth
        new_pdf = allocate_float(vpa.n, vperp.n, z.n, nspecies)
        for is ∈ nspecies, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
            restart_vpa_vals =
            @. vpa.grid -
            moments.ion.upar[iz,ir,is]/moments.ion.vth[iz,ir,is]
            @views interpolate_to_grid_1d!(
                                           new_pdf[:,ivperp,iz,is], restart_vpa_vals,
                                           this_pdf[:,ivperp,iz,is], restart_vpa, restart_vpa_spectral)
        end
        this_pdf = new_pdf
    elseif (moments.evolve_upar && moments.evolve_p &&
            !restart_evolve_upar && !restart_evolve_p)
        # vpa = new_wpa*vth + upar = old_wpa
        # => old_wpa = new_wpa*vth + upar
        new_pdf = allocate_float(vpa.n, vperp.n, z.n, nspecies)
        for is ∈ nspecies, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
            restart_vpa_vals = @. vpa.grid * moments.ion.vth[iz,ir,is] +
            moments.ion.upar[iz,ir,is]
            @views interpolate_to_grid_1d!(
                                           new_pdf[:,ivperp,iz,is], restart_vpa_vals,
                                           this_pdf[:,ivperp,iz,is], restart_vpa, restart_vpa_spectral)
        end
        this_pdf = new_pdf
    elseif (moments.evolve_upar && moments.evolve_p &&
            restart_evolve_upar && !restart_evolve_p)
        # vpa = new_wpa*vth + upar = old_wpa + upar
        # => old_wpa = new_wpa*vth
        new_pdf = allocate_float(vpa.n, vperp.n, z.n, nspecies)
        for is ∈ nspecies, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
            restart_vpa_vals = vpa.grid .* moments.ion.vth[iz,ir,is]
            @views interpolate_to_grid_1d!(
                                           new_pdf[:,ivperp,iz,is], restart_vpa_vals,
                                           this_pdf[:,ivperp,iz,is], restart_vpa, restart_vpa_spectral)
        end
        this_pdf = new_pdf
    elseif (moments.evolve_upar && moments.evolve_p &&
            !restart_evolve_upar && restart_evolve_p)
        # vpa = new_wpa*vth + upar = old_wpa*vth
        # => old_wpa = new_wpa + upar/vth
        new_pdf = allocate_float(vpa.n, vperp.n, z.n, nspecies)
        for is ∈ nspecies, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
            restart_vpa_vals =
            @. vpa.grid +
            moments.ion.upar[iz,ir,is] / moments.ion.vth[iz,ir,is]
            @views interpolate_to_grid_1d!(
                                           new_pdf[:,ivperp,iz,is], restart_vpa_vals,
                                           this_pdf[:,ivperp,iz,is], restart_vpa, restart_vpa_spectral)
        end
        this_pdf = new_pdf
    else
        # This should never happen, as all combinations of evolve_* options
        # should be handled above.
        error("Unsupported combination of moment-kinetic options:"
              * " evolve_density=", moments.evolve_density
              * " evolve_upar=", moments.evolve_upar
              * " evolve_p=", moments.evolve_p
              * " restart_evolve_density=", restart_evolve_density
              * " restart_evolve_upar=", restart_evolve_upar
              * " restart_evolve_p=", restart_evolve_p)
    end
    if moments.evolve_density && !restart_evolve_density
        # Need to normalise by density
        for is ∈ nspecies, iz ∈ 1:z.n
            this_pdf[:,:,iz,is] ./= moments.ion.dens[iz,ir,is]
        end
    elseif !moments.evolve_density && restart_evolve_density
        # Need to unnormalise by density
        for is ∈ nspecies, iz ∈ 1:z.n
            this_pdf[:,:,iz,is] .*= moments.ion.dens[iz,ir,is]
        end
    end
    if moments.evolve_p && !restart_evolve_p
        # Need to normalise by vth
        for is ∈ nspecies, iz ∈ 1:z.n
            this_pdf[:,:,iz,is] .*= moments.ion.vth[iz,ir,is]
        end
    elseif !moments.evolve_p && restart_evolve_p
        # Need to unnormalise by vth
        for is ∈ nspecies, iz ∈ 1:z.n
            this_pdf[:,:,iz,is] ./= moments.ion.vth[iz,ir,is]
        end
    end

    return this_pdf
end

function reload_electron_pdf(dynamic, time_index, moments, r, z, vperp, vpa, r_range,
                             z_range, vperp_range, vpa_range, restart_r,
                             restart_r_spectral, restart_z, restart_z_spectral,
                             restart_vperp, restart_vperp_spectral, restart_vpa,
                             restart_vpa_spectral, interpolation_needed,
                             restart_evolve_density, restart_evolve_upar,
                             restart_evolve_p)

    # Currently, electrons are always fully moment-kinetic
    evolve_density = true
    evolve_upar = true
    evolve_p = true

    this_pdf = load_slice(dynamic, "f_electron", vpa_range, vperp_range, z_range, r_range,
                          time_index)
    orig_nvpa, orig_nvperp, orig_nz, orig_nr = size(this_pdf)
    if interpolation_needed["r"]
        new_pdf = allocate_float(orig_nvpa, orig_nvperp, orig_nz, r.n)
        for iz ∈ 1:orig_nz, ivperp ∈ 1:orig_nvperp,
            ivpa ∈ 1:orig_nvpa
            @views interpolate_to_grid_1d!(
                       new_pdf[ivpa,ivperp,iz,:], r.grid,
                       this_pdf[ivpa,ivperp,iz,:], restart_r,
                       restart_r_spectral)
        end
        this_pdf = new_pdf
    end
    if interpolation_needed["z"]
        new_pdf = allocate_float(orig_nvpa, orig_nvperp, z.n, r.n)
        for ir ∈ 1:r.n, ivperp ∈ 1:orig_nvperp,
            ivpa ∈ 1:orig_nvpa
            @views interpolate_to_grid_1d!(
                       new_pdf[ivpa,ivperp,:,ir], z.grid,
                       this_pdf[ivpa,ivperp,:,ir], restart_z,
                       restart_z_spectral)
        end
        this_pdf = new_pdf
    end

    # Current moment-kinetic implementation is only 1V, so no need to handle a
    # normalised vperp coordinate. This will need to change when 2V
    # moment-kinetics is implemented.
    if interpolation_needed["vperp"]
        new_pdf = allocate_float(orig_nvpa, vperp.n, z.n, r.n)
        for ir ∈ 1:r.n, iz ∈ 1:z.n, ivpa ∈ 1:orig_nvpa
            @views interpolate_to_grid_1d!(
                       new_pdf[ivpa,:,iz,ir], vperp.grid,
                       this_pdf[ivpa,:,iz,ir], restart_vperp,
                       restart_vperp_spectral)
        end
        this_pdf = new_pdf
    end

    if (
        (evolve_density == restart_evolve_density &&
         evolve_upar == restart_evolve_upar && evolve_p ==
         restart_evolve_p)
        || (!evolve_upar && !restart_evolve_upar &&
            !evolve_p && !restart_evolve_p)
       )
        # No chages to velocity-space coordinates, so just interpolate from
        # one grid to the other
        if interpolation_needed["vpa"]
            new_pdf = allocate_float(vpa.n, vperp.n, z.n, r.n)
            for ir ∈ 1:r.n, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
                @views interpolate_to_grid_1d!(
                           new_pdf[:,ivperp,iz,ir], vpa.grid,
                           this_pdf[:,ivperp,iz,ir], restart_vpa,
                           restart_vpa_spectral)
            end
            this_pdf = new_pdf
        end
    elseif (!evolve_upar && !evolve_p &&
            restart_evolve_upar && !restart_evolve_p)
        # vpa = new_wpa = old_wpa + upar
        # => old_wpa = new_wpa - upar
        new_pdf = allocate_float(vpa.n, vperp.n, z.n, r.n)
        for ir ∈ 1:r.n, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
            restart_vpa_vals = vpa.grid .- moments.electron.upar[iz,ir]
            @views interpolate_to_grid_1d!(
                       new_pdf[:,ivperp,iz,ir], restart_vpa_vals,
                       this_pdf[:,ivperp,iz,ir], restart_vpa,
                       restart_vpa_spectral)
        end
        this_pdf = new_pdf
    elseif (!evolve_upar && !evolve_p &&
            !restart_evolve_upar && restart_evolve_p)
        # vpa = new_wpa = old_wpa*vth
        # => old_wpa = new_wpa/vth
        new_pdf = allocate_float(vpa.n, vperp.n, z.n, r.n)
        for ir ∈ 1:r.n, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
            restart_vpa_vals = vpa.grid ./ moments.electron.vth[iz,ir]
            @views interpolate_to_grid_1d!(
                       new_pdf[:,ivperp,iz,ir], restart_vpa_vals,
                       this_pdf[:,ivperp,iz,ir], restart_vpa,
                       restart_vpa_spectral)
        end
        this_pdf = new_pdf
    elseif (!evolve_upar && !evolve_p &&
            restart_evolve_upar && restart_evolve_p)
        # vpa = new_wpa = old_wpa*vth + upar
        # => old_wpa = (new_wpa - upar)/vth
        new_pdf = allocate_float(vpa.n, vperp.n, z.n, r.n)
        for ir ∈ 1:r.n, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
            restart_vpa_vals =
            @. (vpa.grid - moments.electron.upar[iz,ir]) /
                moments.electron.vth[iz,ir]
            @views interpolate_to_grid_1d!(
                       new_pdf[:,ivperp,iz,ir], restart_vpa_vals,
                       this_pdf[:,ivperp,iz,ir], restart_vpa,
                       restart_vpa_spectral)
        end
        this_pdf = new_pdf
    elseif (evolve_upar && !evolve_p &&
            !restart_evolve_upar && !restart_evolve_p)
        # vpa = new_wpa + upar = old_wpa
        # => old_wpa = new_wpa + upar
        new_pdf = allocate_float(vpa.n, vperp.n, z.n, r.n)
        for ir ∈ 1:r.n, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
            restart_vpa_vals = vpa.grid .+ moments.electron.upar[iz,ir]
            @views interpolate_to_grid_1d!(
                       new_pdf[:,ivperp,iz,ir], restart_vpa_vals,
                       this_pdf[:,ivperp,iz,ir], restart_vpa,
                       restart_vpa_spectral)
        end
        this_pdf = new_pdf
    elseif (evolve_upar && !evolve_p &&
            !restart_evolve_upar && restart_evolve_p)
        # vpa = new_wpa + upar = old_wpa*vth
        # => old_wpa = (new_wpa + upar)/vth
        new_pdf = allocate_float(vpa.n, vperp.n, z.n, r.n)
        for ir ∈ 1:r.n, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
            restart_vpa_vals =
            @. (vpa.grid + moments.electron.upar[iz,ir]) /
                moments.electron.vth
            @views interpolate_to_grid_1d!(
                       new_pdf[:,ivperp,iz,ir], restart_vpa_vals,
                       this_pdf[:,ivperp,iz,ir], restart_vpa,
                       restart_vpa_spectral)
        end
        this_pdf = new_pdf
    elseif (evolve_upar && !evolve_p &&
            restart_evolve_upar && restart_evolve_p)
        # vpa = new_wpa + upar = old_wpa*vth + upar
        # => old_wpa = new_wpa/vth
        new_pdf = allocate_float(vpa.n, vperp.n, z.n, r.n)
        for ir ∈ 1:r.n, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
            restart_vpa_vals = vpa.grid ./ moments.electron.vth[iz,ir]
            @views interpolate_to_grid_1d!(
                       new_pdf[:,ivperp,iz,ir], restart_vpa_vals,
                       this_pdf[:,ivperp,iz,ir], restart_vpa,
                       restart_vpa_spectral)
        end
        this_pdf = new_pdf
    elseif (!evolve_upar && evolve_p &&
            !restart_evolve_upar && !restart_evolve_p)
        # vpa = new_wpa*vth = old_wpa
        # => old_wpa = new_wpa*vth
        new_pdf = allocate_float(vpa.n, vperp.n, z.n, r.n)
        for ir ∈ 1:r.n, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
            restart_vpa_vals = vpa.grid .* moments.electron.vth[iz,ir]
            @views interpolate_to_grid_1d!(
                       new_pdf[:,ivperp,iz,ir], restart_vpa_vals,
                       this_pdf[:,ivperp,iz,ir], restart_vpa,
                       restart_vpa_spectral)
        end
        this_pdf = new_pdf
    elseif (!evolve_upar && evolve_p &&
            restart_evolve_upar && !restart_evolve_p)
        # vpa = new_wpa*vth = old_wpa + upar
        # => old_wpa = new_wpa*vth - upar/vth
        new_pdf = allocate_float(vpa.n, vperp.n, z.n, r.n)
        for ir ∈ 1:r.n, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
            restart_vpa_vals = @. vpa.grid * moments.electron.vth[iz,ir] -
                                  moments.electron.upar[iz,ir]
            @views interpolate_to_grid_1d!(
                       new_pdf[:,ivperp,iz,ir], restart_vpa_vals,
                       this_pdf[:,ivperp,iz,ir], restart_vpa,
                       restart_vpa_spectral)
        end
        this_pdf = new_pdf
    elseif (!evolve_upar && evolve_p &&
            restart_evolve_upar && restart_evolve_p)
        # vpa = new_wpa*vth = old_wpa*vth + upar
        # => old_wpa = new_wpa - upar/vth
        new_pdf = allocate_float(vpa.n, vperp.n, z.n, r.n)
        for ir ∈ 1:r.n, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
            restart_vpa_vals =
            @. vpa.grid -
            moments.electron.upar[iz,ir]/moments.electron.vth[iz,ir]
            @views interpolate_to_grid_1d!(
                       new_pdf[:,ivperp,iz,ir], restart_vpa_vals,
                       this_pdf[:,ivperp,iz,ir], restart_vpa,
                       restart_vpa_spectral)
        end
        this_pdf = new_pdf
    elseif (evolve_upar && evolve_p &&
            !restart_evolve_upar && !restart_evolve_p)
        # vpa = new_wpa*vth + upar = old_wpa
        # => old_wpa = new_wpa*vth + upar
        new_pdf = allocate_float(vpa.n, vperp.n, z.n, r.n)
        for ir ∈ 1:r.n, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
            restart_vpa_vals = @. vpa.grid * moments.electron.vth[iz,ir] +
                                  moments.electron.upar[iz,ir]
            @views interpolate_to_grid_1d!(
                       new_pdf[:,ivperp,iz,ir], restart_vpa_vals,
                       this_pdf[:,ivperp,iz,ir], restart_vpa,
                       restart_vpa_spectral)
        end
        this_pdf = new_pdf
    elseif (evolve_upar && evolve_p &&
            restart_evolve_upar && !restart_evolve_p)
        # vpa = new_wpa*vth + upar = old_wpa + upar
        # => old_wpa = new_wpa*vth
        new_pdf = allocate_float(vpa.n, vperp.n, z.n, r.n)
        for ir ∈ 1:r.n, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
            restart_vpa_vals = vpa.grid .* moments.electron.vth[iz,ir]
            @views interpolate_to_grid_1d!(
                       new_pdf[:,ivperp,iz,ir], restart_vpa_vals,
                       this_pdf[:,ivperp,iz,ir], restart_vpa,
                       restart_vpa_spectral)
        end
        this_pdf = new_pdf
    elseif (evolve_upar && evolve_p &&
            !restart_evolve_upar && restart_evolve_p)
        # vpa = new_wpa*vth + upar = old_wpa*vth
        # => old_wpa = new_wpa + upar/vth
        new_pdf = allocate_float(vpa.n, vperp.n, z.n, r.n)
        for ir ∈ 1:r.n, iz ∈ 1:z.n, ivperp ∈ 1:vperp.n
            restart_vpa_vals =
            @. vpa.grid +
            moments.electron.upar[iz,ir] / moments.electron.vth[iz,ir]
            @views interpolate_to_grid_1d!(
                       new_pdf[:,ivperp,iz,ir], restart_vpa_vals,
                       this_pdf[:,ivperp,iz,ir], restart_vpa,
                       restart_vpa_spectral)
        end
        this_pdf = new_pdf
    else
        # This should never happen, as all combinations of evolve_* options
        # should be handled above.
        error("Unsupported combination of moment-kinetic options:"
              * " evolve_density=", evolve_density
              * " evolve_upar=", evolve_upar
              * " evolve_p=", evolve_p
              * " restart_evolve_density=", restart_evolve_density
              * " restart_evolve_upar=", restart_evolve_upar
              * " restart_evolve_p=", restart_evolve_p)
    end
    if evolve_density && !restart_evolve_density
        # Need to normalise by density
        for ir ∈ 1:r.n, iz ∈ 1:z.n
            this_pdf[:,:,iz,ir] ./= moments.electron.dens[iz,ir]
        end
    elseif !evolve_density && restart_evolve_density
        # Need to unnormalise by density
        for ir ∈ 1:r.n, iz ∈ 1:z.n
            this_pdf[:,:,iz,ir] .*= moments.electron.dens[iz,ir]
        end
    end
    if evolve_p && !restart_evolve_p
        # Need to normalise by vth
        for ir ∈ 1:r.n, iz ∈ 1:z.n
            this_pdf[:,:,iz,ir] .*= moments.electron.vth[iz,ir]
        end
    elseif !evolve_p && restart_evolve_p
        # Need to unnormalise by vth
        for ir ∈ 1:r.n, iz ∈ 1:z.n
            this_pdf[:,:,iz,ir] ./= moments.electron.vth[iz,ir]
        end
    end

    return this_pdf
end

function reload_neutral_pdf(dynamic, time_index, moments, r, z, vzeta, vr, vz, r_range,
                            z_range, vzeta_range, vr_range, vz_range, restart_r,
                            restart_r_spectral, restart_z, restart_z_spectral,
                            restart_vzeta, restart_vzeta_spectral, restart_vr,
                            restart_vr_spectral, restart_vz, restart_vz_spectral,
                            interpolation_needed, restart_evolve_density,
                            restart_evolve_upar, restart_evolve_p)
    this_pdf = load_slice(dynamic, "f_neutral", vz_range, vr_range,
                          vzeta_range, z_range, r_range, :, time_index)
    orig_nvz, orig_nvr, orig_nvzeta, orig_nz, orig_nr, nspecies =
        size(this_pdf)
    if interpolation_needed["r"]
        new_pdf = allocate_float(orig_nvz, orig_nvr, orig_nvzeta, orig_nz,
                                 r.n, nspecies)
        for is ∈ 1:nspecies, iz ∈ 1:orig_nz, ivzeta ∈ 1:orig_nvzeta,
                ivr ∈ 1:orig_nvr, ivz ∈ 1:orig_nvz
            @views interpolate_to_grid_1d!(
                new_pdf[ivz,ivr,ivzeta,iz,:,is], r.grid,
                this_pdf[ivz,ivr,ivzeta,iz,:,is], restart_r,
                restart_r_spectral)
        end
        this_pdf = new_pdf
    end
    if interpolation_needed["z"]
        new_pdf = allocate_float(orig_nvz, orig_nvr, orig_nvzeta, z.n,
                                 r.n, nspecies)
        for is ∈ 1:nspecies, ir ∈ 1:r.n, ivzeta ∈ 1:orig_nvzeta,
                ivr ∈ 1:orig_nvr, ivz ∈ 1:orig_nvz
            @views interpolate_to_grid_1d!(
                new_pdf[ivz,ivr,ivzeta,:,ir,is], z.grid,
                this_pdf[ivz,ivr,ivzeta,:,ir,is], restart_z,
                restart_z_spectral)
        end
        this_pdf = new_pdf
    end

    # Current moment-kinetic implementation is only 1V, so no need
    # to handle normalised vzeta or vr coordinates. This will need
    # to change when/if 3V moment-kinetics is implemented.
    if interpolation_needed["vzeta"]
        new_pdf = allocate_float(orig_nvz, orig_nvr, vzeta.n, z.n, r.n,
                                 nspecies)
        for is ∈ 1:nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n, ivr ∈ 1:orig_nvr,
                ivz ∈ 1:orig_nvz
            @views interpolate_to_grid_1d!(
                new_pdf[ivz,ivr,:,iz,ir,is], vzeta.grid,
                this_pdf[ivz,ivr,:,iz,ir,is], restart_vzeta,
                restart_vzeta_spectral)
        end
        this_pdf = new_pdf
    end
    if interpolation_needed["vr"]
        new_pdf = allocate_float(orig_nvz, vr.n, vzeta.n, z.n, r.n,
                                 nspecies)
        for is ∈ 1:nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n,
                ivzeta ∈ 1:vzeta.n, ivz ∈ 1:orig_nvz
            @views interpolate_to_grid_1d!(
                new_pdf[ivz,:,ivzeta,iz,ir,is], vr.grid,
                this_pdf[ivz,:,ivzeta,iz,ir,is], restart_vr,
                restart_vr_spectral)
        end
        this_pdf = new_pdf
    end

    if (
        (moments.evolve_density == restart_evolve_density &&
         moments.evolve_upar == restart_evolve_upar &&
         moments.evolve_p == restart_evolve_p)
        || (!moments.evolve_upar && !restart_evolve_upar &&
            !moments.evolve_p && !restart_evolve_p)
       )
        # No chages to velocity-space coordinates, so just interpolate from
        # one grid to the other
        if interpolation_needed["vz"]
            new_pdf = allocate_float(vz.n, vr.n, vzeta.n, z.n, r.n, nspecies)
            for is ∈ 1:nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n, ivr ∈ 1:vr.n,
                    ivzeta ∈ 1:vzeta.n
                @views interpolate_to_grid_1d!(
                    new_pdf[:,ivr,ivzeta,iz,ir,is], vz.grid,
                    this_pdf[:,ivr,ivzeta,iz,ir,is], restart_vz,
                    restart_vz_spectral)
            end
            this_pdf = new_pdf
        end
    elseif (!moments.evolve_upar && !moments.evolve_p &&
            restart_evolve_upar && !restart_evolve_p)
        # vpa = new_wpa = old_wpa + upar
        # => old_wpa = new_wpa - upar
        new_pdf = allocate_float(vz.n, vr.n, vzeta.n, z.n, r.n, nspecies)
        for is ∈ nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n, ivzeta ∈ 1:vzeta.n,
                ivr ∈ 1:vr.n
            restart_vz_vals = vz.grid .- moments.neutral.uz[iz,ir,is]
            @views interpolate_to_grid_1d!(
                new_pdf[:,ivr,ivzeta,iz,ir,is], restart_vz_vals,
                this_pdf[:,ivr,ivzeta,iz,ir,is], restart_vz,
                restart_vz_spectral)
        end
        this_pdf = new_pdf
    elseif (!moments.evolve_upar && !moments.evolve_p &&
            !restart_evolve_upar && restart_evolve_p)
        # vpa = new_wpa = old_wpa*vth
        # => old_wpa = new_wpa/vth
        new_pdf = allocate_float(vz.n, vr.n, vzeta.n, z.n, r.n, nspecies)
        for is ∈ nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n, ivr ∈ 1:vr.n,
                ivzeta ∈ 1:vzeta.n
            restart_vz_vals = vz.grid ./ moments.neutral.vth[iz,ir,is]
            @views interpolate_to_grid_1d!(
                new_pdf[:,ivr,ivzeta,iz,ir,is], restart_vz_vals,
                this_pdf[:,ivr,ivzeta,iz,ir,is], restart_vz,
                restart_vz_spectral)
        end
        this_pdf = new_pdf
    elseif (!moments.evolve_upar && !moments.evolve_p &&
            restart_evolve_upar && restart_evolve_p)
        # vpa = new_wpa = old_wpa*vth + upar
        # => old_wpa = (new_wpa - upar)/vth
        new_pdf = allocate_float(vz.n, vr.n, vzeta.n, z.n, r.n, nspecies)
        for is ∈ nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n, ivr ∈ 1:vr.n,
                ivzeta ∈ 1:vzeta.n
            restart_vz_vals =
                @. (vz.grid - moments.neutral.uz[iz,ir,is]) /
                   moments.neutral.vth[iz,ir,is]
            @views interpolate_to_grid_1d!(
                new_pdf[:,ivr,ivzeta,iz,ir,is], restart_vz_vals,
                this_pdf[:,ivr,ivzeta,iz,ir,is], restart_vz,
                restart_vz_spectral)
        end
        this_pdf = new_pdf
    elseif (moments.evolve_upar && !moments.evolve_p &&
            !restart_evolve_upar && !restart_evolve_p)
        # vpa = new_wpa + upar = old_wpa
        # => old_wpa = new_wpa + upar
        new_pdf = allocate_float(vz.n, vr.n, vzeta.n, z.n, r.n, nspecies)
        for is ∈ nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n, ivr ∈ 1:vr.n,
                ivzeta ∈ 1:vzeta.n
            restart_vz_vals = vz.grid .+ moments.neutral.uz[iz,ir,is]
            @views interpolate_to_grid_1d!(
                new_pdf[:,ivr,ivzeta,iz,ir,is], restart_vz_vals,
                this_pdf[:,ivr,ivzeta,iz,ir,is], restart_vz, restart_vz_spectral)
        end
        this_pdf = new_pdf
    elseif (moments.evolve_upar && !moments.evolve_p &&
            !restart_evolve_upar && restart_evolve_p)
        # vpa = new_wpa + upar = old_wpa*vth
        # => old_wpa = (new_wpa + upar)/vth
        new_pdf = allocate_float(vz.n, vr.n, vzeta.n, z.n, r.n, nspecies)
        for is ∈ nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n, ivr ∈ 1:vr.n,
                ivzeta ∈ 1:vzeta.n
            restart_vz_vals =
                @. (vz.grid + moments.neutral.uz[iz,ir,is]) /
                    moments.neutral.vth[iz,ir,is]
            @views interpolate_to_grid_1d!(
                new_pdf[:,ivr,ivzeta,iz,ir,is], restart_vz_vals,
                this_pdf[:,ivr,ivzeta,iz,ir,is], restart_vz,
                restart_vz_spectral)
        end
        this_pdf = new_pdf
    elseif (moments.evolve_upar && !moments.evolve_p &&
            restart_evolve_upar && restart_evolve_p)
        # vpa = new_wpa + upar = old_wpa*vth + upar
        # => old_wpa = new_wpa/vth
        new_pdf = allocate_float(vz.n, vr.n, vzeta.n, z.n, r.n, nspecies)
        for is ∈ nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n, ivr ∈ 1:vr.n,
                ivzeta ∈ 1:vzeta.n
            restart_vz_vals = vz.grid ./ moments.neutral.vth[iz,ir,is]
            @views interpolate_to_grid_1d!(
                new_pdf[:,ivr,ivzeta,iz,ir,is], restart_vz_vals,
                this_pdf[:,ivr,ivzeta,iz,ir,is], restart_vz,
                restart_vz_spectral)
        end
        this_pdf = new_pdf
    elseif (!moments.evolve_upar && moments.evolve_p &&
            !restart_evolve_upar && !restart_evolve_p)
        # vpa = new_wpa*vth = old_wpa
        # => old_wpa = new_wpa*vth
        new_pdf = allocate_float(vz.n, vr.n, vzeta.n, z.n, r.n, nspecies)
        for is ∈ nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n, ivr ∈ 1:vr.n,
                ivzeta ∈ 1:vzeta.n
            restart_vz_vals = vz.grid .* moments.neutral.vth[iz,ir,is]
            @views interpolate_to_grid_1d!(
                new_pdf[:,ivr,ivzeta,iz,ir,is], restart_vz_vals,
                this_pdf[:,ivr,ivzeta,iz,ir,is], restart_vz,
                restart_vz_spectral)
        end
        this_pdf = new_pdf
    elseif (!moments.evolve_upar && moments.evolve_p &&
            restart_evolve_upar && !restart_evolve_p)
        # vpa = new_wpa*vth = old_wpa + upar
        # => old_wpa = new_wpa*vth - upar/vth
        new_pdf = allocate_float(vz.n, vr.n, vzeta.n, z.n, r.n, nspecies)
        for is ∈ nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n, ivr ∈ 1:vr.n,
                ivzeta ∈ 1:vzeta.n
            restart_vz_vals =
                @. vz.grid * moments.neutral.vth[iz,ir,is] -
                   moments.neutral.upar[iz,ir,is]
            @views interpolate_to_grid_1d!(
                new_pdf[:,ivr,ivzeta,iz,ir,is], restart_vz_vals,
                this_pdf[:,ivr,ivzeta,iz,ir,is], restart_vz,
                restart_vz_spectral)
        end
        this_pdf = new_pdf
    elseif (!moments.evolve_upar && moments.evolve_p &&
            restart_evolve_upar && restart_evolve_p)
        # vpa = new_wpa*vth = old_wpa*vth + upar
        # => old_wpa = new_wpa - upar/vth
        new_pdf = allocate_float(vz.n, vr.n, vzeta.n, z.n, r.n, nspecies)
        for is ∈ nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n, ivr ∈ 1:vr.n,
                ivzeta ∈ 1:vzeta.n
            restart_vz_vals =
                @. vz.grid -
                   moments.neutral.uz[iz,ir,is]/moments.neutral.vth[iz,ir,is]
            @views interpolate_to_grid_1d!(
                new_pdf[:,ivr,ivzeta,iz,ir,is], restart_vz_vals,
                this_pdf[:,ivr,ivzeta,iz,ir,is], restart_vz,
                restart_vz_spectral)
        end
        this_pdf = new_pdf
    elseif (moments.evolve_upar && moments.evolve_p &&
            !restart_evolve_upar && !restart_evolve_p)
        # vpa = new_wpa*vth + upar = old_wpa
        # => old_wpa = new_wpa*vth + upar
        new_pdf = allocate_float(vz.n, vr.n, vzeta.n, z.n, r.n, nspecies)
        for is ∈ nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n, ivr ∈ 1:vr.n,
                ivzeta ∈ 1:vzeta.n
            restart_vz_vals =
                @. vz.grid * moments.neutral.vth[iz,ir,is] +
                   moments.neutral.uz[iz,ir,is]
            @views interpolate_to_grid_1d!(
                new_pdf[:,ivr,ivzeta,iz,ir,is], restart_vz_vals,
                this_pdf[:,ivr,ivzeta,iz,ir,is], restart_vz,
                restart_vz_spectral)
        end
        this_pdf = new_pdf
    elseif (moments.evolve_upar && moments.evolve_p &&
            restart_evolve_upar && !restart_evolve_p)
        # vpa = new_wpa*vth + upar = old_wpa + upar
        # => old_wpa = new_wpa*vth
        new_pdf = allocate_float(vz.n, vr.n, vzeta.n, z.n, r.n, nspecies)
        for is ∈ nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n, ivr ∈ 1:vr.n,
                ivzeta ∈ 1:vzeta.n
            restart_vz_vals = vz.grid .* moments.neutral.vth[iz,ir,is]
            @views interpolate_to_grid_1d!(
                new_pdf[:,ivr,ivzeta,iz,ir,is], restart_vz_vals,
                this_pdf[:,ivr,ivzeta,iz,ir,is], restart_vz,
                restart_vz_spectral)
        end
        this_pdf = new_pdf
    elseif (moments.evolve_upar && moments.evolve_p &&
            !restart_evolve_upar && restart_evolve_p)
        # vpa = new_wpa*vth + upar = old_wpa*vth
        # => old_wpa = new_wpa + upar/vth
        new_pdf = allocate_float(vz.n, vr.n, vzeta.n, z.n, r.n, nspecies)
        for is ∈ nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n, ivr ∈ 1:vr.n,
                ivzeta ∈ 1:vzeta.n
            restart_vz_vals =
                @. vz.grid +
                   moments.neutral.uz[iz,ir,is]/moments.neutral.vth[iz,ir,is]
            @views interpolate_to_grid_1d!(
                new_pdf[:,ivr,ivzeta,iz,ir,is], restart_vz_vals,
                this_pdf[:,ivr,ivzeta,iz,ir,is], restart_vz,
                restart_vz_spectral)
        end
        this_pdf = new_pdf
    else
        # This should never happen, as all combinations of evolve_* options
        # should be handled above.
        error("Unsupported combination of moment-kinetic options:"
              * " evolve_density=", moments.evolve_density
              * " evolve_upar=", moments.evolve_upar
              * " evolve_p=", moments.evolve_p
              * " restart_evolve_density=", restart_evolve_density
              * " restart_evolve_upar=", restart_evolve_upar
              * " restart_evolve_p=", restart_evolve_p)
    end
    if moments.evolve_density && !restart_evolve_density
        # Need to normalise by density
        for is ∈ nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n
            this_pdf[:,:,:,iz,ir,is] ./= moments.neutral.dens[iz,ir,is]
        end
    elseif !moments.evolve_density && restart_evolve_density
        # Need to unnormalise by density
        for is ∈ nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n
            this_pdf[:,:,:,iz,ir,is] .*= moments.neutral.dens[iz,ir,is]
        end
    end
    if moments.evolve_p && !restart_evolve_p
        # Need to normalise by vth
        for is ∈ nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n
            this_pdf[:,:,:,iz,ir,is] .*= moments.neutral.vth[iz,ir,is]
        end
    elseif !moments.evolve_p && restart_evolve_p
        # Need to unnormalise by vth
        for is ∈ nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n
            this_pdf[:,:,:,iz,ir,is] ./= moments.neutral.vth[iz,ir,is]
        end
    end

    return this_pdf
end

function reload_neutral_boundary_pdf(boundary_distributions_io, var_name, ir, moments, z,
                                     vzeta, vr, vz, z_range, vzeta_range, vr_range,
                                     vz_range, restart_z, restart_z_spectral,
                                     restart_vzeta, restart_vzeta_spectral, restart_vr,
                                     restart_vr_spectral, restart_vz, restart_vz_spectral,
                                     interpolation_needed, restart_evolve_density,
                                     restart_evolve_upar, restart_evolve_p)
    this_pdf = load_slice(boundary_distributions_io, var_name, vz_range,
                          vr_range, vzeta_range, z_range, :)
    orig_nvz, orig_nvr, orig_nvzeta, orig_nz, nspecies = size(this_pdf)
    if interpolation_needed["z"]
        new_pdf = allocate_float(orig_nvz, orig_nvr, orig_nvzeta, z.n,
                                 nspecies)
        for is ∈ 1:nspecies, ivzeta ∈ 1:orig_nvzeta, ivr ∈ 1:orig_nvr,
                ivz ∈ 1:orig_nvz
            @views interpolate_to_grid_1d!(
                new_pdf[ivz,ivr,ivzeta,:,is], z.grid,
                this_pdf[ivz,ivr,ivzeta,:,is], restart_z,
                restart_z_spectral)
        end
        this_pdf = new_pdf
    end

    # Current moment-kinetic implementation is only 1V, so no need
    # to handle normalised vzeta or vr coordinates. This will need
    # to change when/if 3V moment-kinetics is implemented.
    if interpolation_needed["vzeta"]
        new_pdf = allocate_float(orig_nvz, orig_nvr, vzeta.n, z.n,
                                 nspecies)
        for is ∈ 1:nspecies, iz ∈ 1:z.n, ivr ∈ 1:orig_nvr,
                ivz ∈ 1:orig_nvz
            @views interpolate_to_grid_1d!(
                new_pdf[ivz,ivr,:,iz,is], vzeta.grid,
                this_pdf[ivz,ivr,:,iz,is], restart_vzeta,
                restart_vzeta_spectral)
        end
        this_pdf = new_pdf
    end
    if interpolation_needed["vr"]
        new_pdf = allocate_float(orig_nvz, vr.n, vzeta.n, z.n, nspecies)
        for is ∈ 1:nspecies, iz ∈ 1:z.n, ivzeta ∈ 1:vzeta.n,
                ivz ∈ 1:orig_nvz
            @views interpolate_to_grid_1d!(
                new_pdf[ivz,:,ivzeta,iz,is], vr.grid,
                this_pdf[ivz,:,ivzeta,iz,is], restart_vr,
                restart_vr_spectral)
        end
        this_pdf = new_pdf
    end

    if (
        (moments.evolve_density == restart_evolve_density &&
         moments.evolve_upar == restart_evolve_upar && moments.evolve_p ==
         restart_evolve_p)
        || (!moments.evolve_upar && !restart_evolve_upar &&
            !moments.evolve_p && !restart_evolve_p)
       )
        # No chages to velocity-space coordinates, so just interpolate from
        # one grid to the other
        if interpolation_needed["vz"]
            new_pdf = allocate_float(vz.n, vr.n, vzeta.n, z.n, nspecies)
            for is ∈ 1:nspecies, iz ∈ 1:z.n, ivr ∈ 1:vr.n,
                    ivzeta ∈ 1:vzeta.n
                @views interpolate_to_grid_1d!(
                    new_pdf[:,ivr,ivzeta,iz,is], vz.grid,
                    this_pdf[:,ivr,ivzeta,iz,is], restart_vz,
                    restart_vz_spectral)
            end
            this_pdf = new_pdf
        end
    elseif (!moments.evolve_upar && !moments.evolve_p &&
            restart_evolve_upar && !restart_evolve_p)
        # vpa = new_wpa = old_wpa + upar
        # => old_wpa = new_wpa - upar
        new_pdf = allocate_float(vz.n, vr.n, vzeta.n, z.n, nspecies)
        for is ∈ nspecies, iz ∈ 1:z.n, ivzeta ∈ 1:vzeta.n, ivr ∈ 1:vr.n
            restart_vz_vals = vz.grid .- moments.neutral.uz[iz,ir,is]
            @views interpolate_to_grid_1d!(
                new_pdf[:,ivr,ivzeta,iz,is], restart_vz_vals,
                this_pdf[:,ivr,ivzeta,iz,is], restart_vz,
                restart_vz_spectral)
        end
        this_pdf = new_pdf
    elseif (!moments.evolve_upar && !moments.evolve_p &&
            !restart_evolve_upar && restart_evolve_p)
        # vpa = new_wpa = old_wpa*vth
        # => old_wpa = new_wpa/vth
        new_pdf = allocate_float(vz.n, vr.n, vzeta.n, z.n, nspecies)
        for is ∈ nspecies, iz ∈ 1:z.n, ivr ∈ 1:vr.n, ivzeta ∈ 1:vzeta.n
            restart_vz_vals = vz.grid ./ moments.neutral.vth[iz,ir,is]
            @views interpolate_to_grid_1d!(
                new_pdf[:,ivr,ivzeta,iz,is], restart_vz_vals,
                this_pdf[:,ivr,ivzeta,iz,is], restart_vz,
                restart_vz_spectral)
        end
        this_pdf = new_pdf
    elseif (!moments.evolve_upar && !moments.evolve_p &&
            restart_evolve_upar && restart_evolve_p)
        # vpa = new_wpa = old_wpa*vth + upar
        # => old_wpa = (new_wpa - upar)/vth
        new_pdf = allocate_float(vz.n, vr.n, vzeta.n, z.n, nspecies)
        for is ∈ nspecies, iz ∈ 1:z.n, ivr ∈ 1:vr.n, ivzeta ∈ 1:vzeta.n
            restart_vz_vals =
                @. (vz.grid - moments.neutral.uz[iz,ir,is]) /
                   moments.neutral.vth[iz,ir,is]
            @views interpolate_to_grid_1d!(
                new_pdf[:,ivr,ivzeta,iz,is], restart_vz_vals,
                this_pdf[:,ivr,ivzeta,iz,is], restart_vz,
                restart_vz_spectral)
        end
        this_pdf = new_pdf
    elseif (moments.evolve_upar && !moments.evolve_p &&
            !restart_evolve_upar && !restart_evolve_p)
        # vpa = new_wpa + upar = old_wpa
        # => old_wpa = new_wpa + upar
        new_pdf = allocate_float(vz.n, vr.n, vzeta.n, z.n, nspecies)
        for is ∈ nspecies, iz ∈ 1:z.n, ivr ∈ 1:vr.n, ivzeta ∈ 1:vzeta.n
            restart_vz_vals = vz.grid .+
                              moments.neutral.uz[iz,ir,is]
            @views interpolate_to_grid_1d!(
                new_pdf[:,ivr,ivzeta,iz,is], restart_vz_vals,
                this_pdf[:,ivr,ivzeta,iz,is], restart_vz,
                restart_vz_spectral)
        end
        this_pdf = new_pdf
    elseif (moments.evolve_upar && !moments.evolve_p &&
            !restart_evolve_upar && restart_evolve_p)
        # vpa = new_wpa + upar = old_wpa*vth
        # => old_wpa = (new_wpa + upar)/vth
        new_pdf = allocate_float(vz.n, vr.n, vzeta.n, z.n, nspecies)
        for is ∈ nspecies, iz ∈ 1:z.n, ivr ∈ 1:vr.n, ivzeta ∈ 1:vzeta.n
            restart_vz_vals =
                @. (vz.grid + moments.neutral.uz[iz,ir,is]) /
                   moments.neutral.vth
            @views interpolate_to_grid_1d!(
                new_pdf[:,ivr,ivzeta,iz,is], restart_vz_vals,
                this_pdf[:,ivr,ivzeta,iz,is], restart_vz,
                restart_vz_spectral)
        end
        this_pdf = new_pdf
    elseif (moments.evolve_upar && !moments.evolve_p &&
            restart_evolve_upar && restart_evolve_p)
        # vpa = new_wpa + upar = old_wpa*vth + upar
        # => old_wpa = new_wpa/vth
        new_pdf = allocate_float(vz.n, vr.n, vzeta.n, z.n, nspecies)
        for is ∈ nspecies, iz ∈ 1:z.n, ivr ∈ 1:vr.n, ivzeta ∈ 1:vzeta.n
            restart_vz_vals = vz.grid ./ moments.neutral.vth[iz,ir,is]
            @views interpolate_to_grid_1d!(
                new_pdf[:,ivr,ivzeta,iz,is], restart_vz_vals,
                this_pdf[:,ivr,ivzeta,iz,is], restart_vz,
                restart_vz_spectral)
        end
        this_pdf = new_pdf
    elseif (!moments.evolve_upar && moments.evolve_p &&
            !restart_evolve_upar && !restart_evolve_p)
        # vpa = new_wpa*vth = old_wpa
        # => old_wpa = new_wpa*vth
        new_pdf = allocate_float(vz.n, vr.n, vzeta.n, z.n, nspecies)
        for is ∈ nspecies, iz ∈ 1:z.n, ivr ∈ 1:vr.n, ivzeta ∈ 1:vzeta.n
            restart_vz_vals = vz.grid .*
                              moments.neutral.vth[iz,ir,is]
            @views interpolate_to_grid_1d!(
                new_pdf[:,ivr,ivzeta,iz,is], restart_vz_vals,
                this_pdf[:,ivr,ivzeta,iz,is], restart_vz,
                restart_vz_spectral)
        end
        this_pdf = new_pdf
    elseif (!moments.evolve_upar && moments.evolve_p &&
            restart_evolve_upar && !restart_evolve_p)
        # vpa = new_wpa*vth = old_wpa + upar
        # => old_wpa = new_wpa*vth - upar/vth
        new_pdf = allocate_float(vz.n, vr.n, vzeta.n, z.n, nspecies)
        for is ∈ nspecies, iz ∈ 1:z.n, ivr ∈ 1:vr.n, ivzeta ∈ 1:vzeta.n
            restart_vz_vals =
                @. vz.grid * moments.neutral.vth[iz,ir,is] -
                   moments.neutral.upar[iz,ir,is]
            @views interpolate_to_grid_1d!(
                new_pdf[:,ivr,ivzeta,iz,is], restart_vz_vals,
                this_pdf[:,ivr,ivzeta,iz,is], restart_vz,
                restart_vz_spectral)
        end
        this_pdf = new_pdf
    elseif (!moments.evolve_upar && moments.evolve_p &&
            restart_evolve_upar && restart_evolve_p)
        # vpa = new_wpa*vth = old_wpa*vth + upar
        # => old_wpa = new_wpa - upar/vth
        new_pdf = allocate_float(vz.n, vr.n, vzeta.n, z.n, nspecies)
        for is ∈ nspecies, iz ∈ 1:z.n, ivr ∈ 1:vr.n, ivzeta ∈ 1:vzeta.n
            restart_vz_vals =
                @. vz.grid -
                   moments.neutral.uz[iz,ir,is]/moments.neutral.vth[iz,ir,is]
            @views interpolate_to_grid_1d!(
                new_pdf[:,ivr,ivzeta,iz,is], restart_vz_vals,
                this_pdf[:,ivr,ivzeta,iz,is], restart_vz,
                restart_vz_spectral)
        end
        this_pdf = new_pdf
    elseif (moments.evolve_upar && moments.evolve_p &&
            !restart_evolve_upar && !restart_evolve_p)
        # vpa = new_wpa*vth + upar = old_wpa
        # => old_wpa = new_wpa*vth + upar
        new_pdf = allocate_float(vz.n, vr.n, vzeta.n, z.n, nspecies)
        for is ∈ nspecies, iz ∈ 1:z.n, ivr ∈ 1:vr.n, ivzeta ∈ 1:vzeta.n
            restart_vz_vals =
                @. vz.grid * moments.neutral.vth[iz,ir,is] +
                   moments.neutral.uz[iz,ir,is]
            @views interpolate_to_grid_1d!(
                new_pdf[:,ivr,ivzeta,iz,is], restart_vz_vals,
                this_pdf[:,ivr,ivzeta,iz,is], restart_vz,
                restart_vz_spectral)
        end
        this_pdf = new_pdf
    elseif (moments.evolve_upar && moments.evolve_p &&
            restart_evolve_upar && !restart_evolve_p)
        # vpa = new_wpa*vth + upar = old_wpa + upar
        # => old_wpa = new_wpa*vth
        new_pdf = allocate_float(vz.n, vr.n, vzeta.n, z.n, nspecies)
        for is ∈ nspecies, iz ∈ 1:z.n, ivr ∈ 1:vr.n, ivzeta ∈ 1:vzeta.n
            restart_vz_vals = vz.grid .* moments.neutral.vth[iz,ir,is]
            @views interpolate_to_grid_1d!(
                new_pdf[:,ivr,ivzeta,iz,is], restart_vz_vals,
                this_pdf[:,ivr,ivzeta,iz,is], restart_vz,
                restart_vz_spectral)
        end
        this_pdf = new_pdf
    elseif (moments.evolve_upar && moments.evolve_p &&
            !restart_evolve_upar && restart_evolve_p)
        # vpa = new_wpa*vth + upar = old_wpa*vth
        # => old_wpa = new_wpa + upar/vth
        new_pdf = allocate_float(vz.n, vr.n, vzeta.n, z.n, nspecies)
        for is ∈ nspecies, iz ∈ 1:z.n, ivr ∈ 1:vr.n, ivzeta ∈ 1:vzeta.n
            restart_vz_vals =
                @. vz.grid +
                   moments.neutral.uz[iz,ir,is]/moments.neutral.vth[iz,ir,is]
            @views interpolate_to_grid_1d!(
                new_pdf[:,ivr,ivzeta,iz,is], restart_vz_vals,
                this_pdf[:,ivr,ivzeta,iz,is], restart_vz,
                restart_vz_spectral)
        end
    else
        # This should never happen, as all combinations of evolve_* options
        # should be handled above.
        error("Unsupported combination of moment-kinetic options:"
              * " evolve_density=", moments.evolve_density
              * " evolve_upar=", moments.evolve_upar
              * " evolve_p=", moments.evolve_p
              * " restart_evolve_density=", restart_evolve_density
              * " restart_evolve_upar=", restart_evolve_upar
              * " restart_evolve_p=", restart_evolve_p)
    end
    if moments.evolve_density && !restart_evolve_density
        # Need to normalise by density
        for is ∈ nspecies, iz ∈ 1:z.n
            this_pdf[:,:,:,iz,is] ./= moments.neutral.dens[iz,ir,is]
        end
    elseif !moments.evolve_density && restart_evolve_density
        # Need to unnormalise by density
        for is ∈ nspecies, iz ∈ 1:z.n
            this_pdf[:,:,:,iz,is] .*= moments.neutral.dens[iz,ir,is]
        end
    end
    if moments.evolve_p && !restart_evolve_p
        # Need to normalise by vth
        for is ∈ nspecies, iz ∈ 1:z.n
            this_pdf[:,:,:,iz,is] .*= moments.neutral.vth[iz,ir,is]
        end
    elseif !moments.evolve_p && restart_evolve_p
        # Need to unnormalise by vth
        for is ∈ nspecies, iz ∈ 1:z.n
            this_pdf[:,:,:,iz,is] ./= moments.neutral.vth[iz,ir,is]
        end
    end

    return this_pdf
end

"""
Read a slice of an ion distribution function

run_names is a tuple. If it has more than one entry, this means that there are multiple
restarts (which are sequential in time), so concatenate the data from each entry together.

The slice to take is specified by the keyword arguments.
"""
function load_distributed_ion_pdf_slice(run_names::Tuple, nblocks::Tuple, t_range,
                                        n_species::mk_int, r::coordinate, z::coordinate,
                                        vperp::coordinate, vpa::coordinate; is=nothing,
                                        ir=nothing, iz=nothing, ivperp=nothing,
                                        ivpa=nothing)
    # dimension of pdf is [vpa,vperp,z,r,species,t]

    result_dims = mk_int[]
    if ivpa === nothing
        ivpa = 1:vpa.n_global
        push!(result_dims, vpa.n_global)
    elseif !isa(ivpa, mk_int)
        push!(result_dims, length(ivpa))
    end
    if ivperp === nothing
        ivperp = 1:vperp.n_global
        push!(result_dims, vperp.n_global)
    elseif !isa(ivperp, mk_int)
        push!(result_dims, length(ivperp))
    end
    if iz === nothing
        iz = 1:z.n_global
        push!(result_dims, z.n_global)
    elseif isa(iz, mk_int)
        push!(result_dims, 1)
    else
        push!(result_dims, length(iz))
    end
    if ir === nothing
        ir = 1:r.n_global
        push!(result_dims, r.n_global)
    elseif isa(ir, mk_int)
        push!(result_dims, 1)
    else
        push!(result_dims, length(ir))
    end
    if is === nothing
        is = 1:n_species
        push!(result_dims, n_species)
    elseif !isa(is, mk_int)
        push!(result_dims, length(is))
    else
        push!(result_dims, 1)
    end
    push!(result_dims, length(t_range))

    f_global = allocate_float(result_dims...)

    local_tind_start = 1
    local_tind_end = -1
    global_tind_start = 1
    global_tind_end = -1
    for (run_name, nb) in zip(run_names, nblocks)
        for iblock in 0:nb-1
            fid = open_readonly_output_file(run_name, "dfns", iblock=iblock, printout=false)

            z_irank, z_nrank, r_irank, r_nrank = load_rank_data(fid)

            # max index set to avoid double assignment of repeated points
            # nr/nz if irank = nrank-1, (nr-1)/(nz-1) otherwise
            imax_r = (r_irank == r.nrank - 1 ? r.n : r.n - 1)
            imax_z = (z_irank == z.nrank - 1 ? z.n : z.n - 1)
            local_r_range = 1:imax_r
            local_z_range = 1:imax_z
            global_r_range = iglobal_func(1, r_irank, r.n):iglobal_func(imax_r, r_irank, r.n)
            global_z_range = iglobal_func(1, z_irank, z.n):iglobal_func(imax_z, z_irank, z.n)

            if ir !== nothing && !any(i ∈ global_r_range for i in ir)
                # No data for the slice on this rank
                continue
            elseif isa(ir, StepRange)
                # Note that `findfirst(test, array)` returns the index `i` of the first
                # element of `array` for which `test(array[i])` is `true`.
                # `findlast()` similarly finds the index of the last element...
                start_ind = findfirst(i -> i>=ir.start, global_r_range)
                start = global_r_range[start_ind]
                stop_ind = findlast(i -> i<=ir.stop, global_r_range)
                stop = global_r_range[stop_ind]
                local_r_range = (local_r_range.start + start - global_r_range.start):ir.step:(local_r_range.stop + stop - global_r_range.stop)
                global_r_range = findfirst(i->i ∈ global_r_range, ir):findlast(i->i ∈ global_r_range, ir)
            elseif isa(ir, UnitRange)
                start_ind = findfirst(i -> i>=ir.start, global_r_range)
                start = global_r_range[start_ind]
                stop_ind = findlast(i -> i<=ir.stop, global_r_range)
                stop = global_r_range[stop_ind]
                local_r_range = (local_r_range.start + start - global_r_range.start):(local_r_range.stop + stop - global_r_range.stop)
                global_r_range = findfirst(i->i ∈ global_r_range, ir):findlast(i->i ∈ global_r_range, ir)
            elseif isa(ir, mk_int)
                local_r_range = ir - (global_r_range.start - 1)
                global_r_range = ir
            end
            if iz !== nothing && !any(i ∈ global_z_range for i in iz)
                # No data for the slice on this rank
                continue
            elseif isa(iz, StepRange)
                # Note that `findfirst(test, array)` returns the index `i` of the first
                # element of `array` for which `test(array[i])` is `true`.
                # `findlast()` similarly finds the index of the last element...
                start_ind = findfirst(i -> i>=iz.start, global_z_range)
                start = global_z_range[start_ind]
                stop_ind = findlast(i -> i<=iz.stop, global_z_range)
                stop = global_z_range[stop_ind]
                local_z_range = (local_z_range.start + start - global_z_range.start):iz.step:(local_z_range.stop + stop - global_z_range.stop)
                global_z_range = findfirst(i->i ∈ global_z_range, iz):findlast(i->i ∈ global_z_range, iz)
            elseif isa(iz, UnitRange)
                start_ind = findfirst(i -> i>=iz.start, global_z_range)
                start = global_z_range[start_ind]
                stop_ind = findlast(i -> i<=iz.stop, global_z_range)
                stop = global_z_range[stop_ind]
                local_z_range = (local_z_range.start + start - global_z_range.start):(local_z_range.stop + stop - global_z_range.stop)
                global_z_range = findfirst(i->i ∈ global_z_range, iz):findlast(i->i ∈ global_z_range, iz)
            elseif isa(iz, mk_int)
                local_z_range = iz - (global_z_range.start - 1)
                global_z_range = iz
            end

            f_local_slice = load_pdf_data(fid)

            if local_tind_start > 1
                # The run being loaded is a restart (as local_tind_start=1 for the first
                # run), so skip the first point, as this is a duplicate of the last point
                # of the previous restart
                skip_first = 1
            else
                skip_first = 0
            end
            ntime_local = size(f_local_slice, ndims(f_local_slice)) - skip_first
            local_tind_end = local_tind_start + ntime_local - 1
            local_t_range = collect(it - local_tind_start + 1 + skip_first
                                    for it ∈ t_range
                                    if local_tind_start <= it <= local_tind_end)
            global_tind_end = global_tind_start + length(local_t_range) - 1

            f_global_slice = selectdim(f_global, ndims(f_global),
                                       global_tind_start:global_tind_end)

            # Note: use selectdim() and get the dimension from thisdim because the actual
            # number of dimensions in f_global_slice, f_local_slice is different depending
            # on which combination of ivpa, ivperp, iz, ir, and is was passed.
            thisdim = ndims(f_local_slice) - 5
            f_local_slice = selectdim(f_local_slice, thisdim, ivpa)

            thisdim = ndims(f_local_slice) - 4
            f_local_slice = selectdim(f_local_slice, thisdim, ivperp)

            thisdim = ndims(f_local_slice) - 3
            if isa(iz, mk_int)
                f_global_slice = selectdim(f_global_slice, thisdim, 1)
                f_local_slice = selectdim(f_local_slice, thisdim,
                                          ilocal_func(iz, z_irank, z.n))
            else
                f_global_slice = selectdim(f_global_slice, thisdim, global_z_range)
                f_local_slice = selectdim(f_local_slice, thisdim, local_z_range)
            end

            thisdim = ndims(f_local_slice) - 2
            if isa(ir, mk_int)
                f_global_slice = selectdim(f_global_slice, thisdim, 1)
                f_local_slice = selectdim(f_local_slice, thisdim,
                                          ilocal_func(ir, r_irank, r.n))
            else
                f_global_slice = selectdim(f_global_slice, thisdim, global_r_range)
                f_local_slice = selectdim(f_local_slice, thisdim, local_r_range)
            end

            thisdim = ndims(f_local_slice) - 1
            f_global_slice = selectdim(f_global_slice, thisdim, is)
            f_local_slice = selectdim(f_local_slice, thisdim, is)

            # Select time slice
            thisdim = ndims(f_local_slice)
            f_local_slice = selectdim(f_local_slice, thisdim, local_t_range)

            f_global_slice .= f_local_slice
            close(fid)
        end
        local_tind_start = local_tind_end + 1
        global_tind_start = global_tind_end + 1
    end

    if isa(iz, mk_int)
        thisdim = ndims(f_global) - 3
        f_global = selectdim(f_global, thisdim, 1)
    end
    if isa(ir, mk_int)
        thisdim = ndims(f_global) - 2
        f_global = selectdim(f_global, thisdim, 1)
    end
    if isa(is, mk_int)
        thisdim = ndims(f_global) - 1
        f_global = selectdim(f_global, thisdim, 1)
    end
    if isa(t_range, mk_int)
        thisdim = ndims(f_global)
        f_global = selectdim(f_global, thisdim, 1)
    end

    return f_global
end

"""
Read a slice of an electron distribution function

run_names is a tuple. If it has more than one entry, this means that there are multiple
restarts (which are sequential in time), so concatenate the data from each entry together.

The slice to take is specified by the keyword arguments.
"""
function load_distributed_electron_pdf_slice(run_names::Tuple, nblocks::Tuple, t_range,
                                             n_species::mk_int, r::coordinate,
                                             z::coordinate, vperp::coordinate,
                                             vpa::coordinate; ir=nothing, iz=nothing,
                                             ivperp=nothing, ivpa=nothing)
    # dimension of pdf is [vpa,vperp,z,r,t]

    result_dims = mk_int[]
    if ivpa === nothing
        ivpa = 1:vpa.n_global
        push!(result_dims, vpa.n_global)
    elseif !isa(ivpa, mk_int)
        push!(result_dims, length(ivpa))
    end
    if ivperp === nothing
        ivperp = 1:vperp.n_global
        push!(result_dims, vperp.n_global)
    elseif !isa(ivperp, mk_int)
        push!(result_dims, length(ivperp))
    end
    if iz === nothing
        iz = 1:z.n_global
        push!(result_dims, z.n_global)
    elseif isa(iz, mk_int)
        push!(result_dims, 1)
    else
        push!(result_dims, length(iz))
    end
    if ir === nothing
        ir = 1:r.n_global
        push!(result_dims, r.n_global)
    elseif isa(ir, mk_int)
        push!(result_dims, 1)
    else
        push!(result_dims, length(ir))
    end
    push!(result_dims, length(t_range))

    f_global = allocate_float(result_dims...)

    local_tind_start = 1
    local_tind_end = -1
    global_tind_start = 1
    global_tind_end = -1
    for (run_name, nb) in zip(run_names, nblocks)
        for iblock in 0:nb-1
            fid = open_readonly_output_file(run_name, "dfns", iblock=iblock, printout=false)

            z_irank, z_nrank, r_irank, r_nrank = load_rank_data(fid)

            # max index set to avoid double assignment of repeated points
            # nr/nz if irank = nrank-1, (nr-1)/(nz-1) otherwise
            imax_r = (r_irank == r.nrank - 1 ? r.n : r.n - 1)
            imax_z = (z_irank == z.nrank - 1 ? z.n : z.n - 1)
            local_r_range = 1:imax_r
            local_z_range = 1:imax_z
            global_r_range = iglobal_func(1, r_irank, r.n):iglobal_func(imax_r, r_irank, r.n)
            global_z_range = iglobal_func(1, z_irank, z.n):iglobal_func(imax_z, z_irank, z.n)

            if ir !== nothing && !any(i ∈ global_r_range for i in ir)
                # No data for the slice on this rank
                continue
            elseif isa(ir, StepRange)
                # Note that `findfirst(test, array)` returns the index `i` of the first
                # element of `array` for which `test(array[i])` is `true`.
                # `findlast()` similarly finds the index of the last element...
                start_ind = findfirst(i -> i>=ir.start, global_r_range)
                start = global_r_range[start_ind]
                stop_ind = findlast(i -> i<=ir.stop, global_r_range)
                stop = global_r_range[stop_ind]
                local_r_range = (local_r_range.start + start - global_r_range.start):ir.step:(local_r_range.stop + stop - global_r_range.stop)
                global_r_range = findfirst(i->i ∈ global_r_range, ir):findlast(i->i ∈ global_r_range, ir)
            elseif isa(ir, UnitRange)
                start_ind = findfirst(i -> i>=ir.start, global_r_range)
                start = global_r_range[start_ind]
                stop_ind = findlast(i -> i<=ir.stop, global_r_range)
                stop = global_r_range[stop_ind]
                local_r_range = (local_r_range.start + start - global_r_range.start):(local_r_range.stop + stop - global_r_range.stop)
                global_r_range = findfirst(i->i ∈ global_r_range, ir):findlast(i->i ∈ global_r_range, ir)
            elseif isa(ir, mk_int)
                local_r_range = ir - (global_r_range.start - 1)
                global_r_range = ir
            end
            if iz !== nothing && !any(i ∈ global_z_range for i in iz)
                # No data for the slice on this rank
                continue
            elseif isa(iz, StepRange)
                # Note that `findfirst(test, array)` returns the index `i` of the first
                # element of `array` for which `test(array[i])` is `true`.
                # `findlast()` similarly finds the index of the last element...
                start_ind = findfirst(i -> i>=iz.start, global_z_range)
                start = global_z_range[start_ind]
                stop_ind = findlast(i -> i<=iz.stop, global_z_range)
                stop = global_z_range[stop_ind]
                local_z_range = (local_z_range.start + start - global_z_range.start):iz.step:(local_z_range.stop + stop - global_z_range.stop)
                global_z_range = findfirst(i->i ∈ global_z_range, iz):findlast(i->i ∈ global_z_range, iz)
            elseif isa(iz, UnitRange)
                start_ind = findfirst(i -> i>=iz.start, global_z_range)
                start = global_z_range[start_ind]
                stop_ind = findlast(i -> i<=iz.stop, global_z_range)
                stop = global_z_range[stop_ind]
                local_z_range = (local_z_range.start + start - global_z_range.start):(local_z_range.stop + stop - global_z_range.stop)
                global_z_range = findfirst(i->i ∈ global_z_range, iz):findlast(i->i ∈ global_z_range, iz)
            elseif isa(iz, mk_int)
                local_z_range = iz - (global_z_range.start - 1)
                global_z_range = iz
            end

            f_local_slice = load_pdf_data(fid)

            if local_tind_start > 1
                # The run being loaded is a restart (as local_tind_start=1 for the first
                # run), so skip the first point, as this is a duplicate of the last point
                # of the previous restart
                skip_first = 1
            else
                skip_first = 0
            end
            ntime_local = size(f_local_slice, ndims(f_local_slice)) - skip_first
            local_tind_end = local_tind_start + ntime_local - 1
            local_t_range = collect(it - local_tind_start + 1 + skip_first
                                    for it ∈ t_range
                                    if local_tind_start <= it <= local_tind_end)
            global_tind_end = global_tind_start + length(local_t_range) - 1

            f_global_slice = selectdim(f_global, ndims(f_global),
                                       global_tind_start:global_tind_end)

            # Note: use selectdim() and get the dimension from thisdim because the actual
            # number of dimensions in f_global_slice, f_local_slice is different depending
            # on which combination of ivpa, ivperp, iz, ir, and is was passed.
            thisdim = ndims(f_local_slice) - 5
            f_local_slice = selectdim(f_local_slice, thisdim, ivpa)

            thisdim = ndims(f_local_slice) - 4
            f_local_slice = selectdim(f_local_slice, thisdim, ivperp)

            thisdim = ndims(f_local_slice) - 3
            if isa(iz, mk_int)
                f_global_slice = selectdim(f_global_slice, thisdim, 1)
                f_local_slice = selectdim(f_local_slice, thisdim,
                                          ilocal_func(iz, z_irank, z.n))
            else
                f_global_slice = selectdim(f_global_slice, thisdim, global_z_range)
                f_local_slice = selectdim(f_local_slice, thisdim, local_z_range)
            end

            thisdim = ndims(f_local_slice) - 2
            if isa(ir, mk_int)
                f_global_slice = selectdim(f_global_slice, thisdim, 1)
                f_local_slice = selectdim(f_local_slice, thisdim,
                                          ilocal_func(ir, r_irank, r.n))
            else
                f_global_slice = selectdim(f_global_slice, thisdim, global_r_range)
                f_local_slice = selectdim(f_local_slice, thisdim, local_r_range)
            end

            thisdim = ndims(f_local_slice) - 1
            f_global_slice = selectdim(f_global_slice, thisdim)
            f_local_slice = selectdim(f_local_slice, thisdim)

            # Select time slice
            thisdim = ndims(f_local_slice)
            f_local_slice = selectdim(f_local_slice, thisdim, local_t_range)

            f_global_slice .= f_local_slice
            close(fid)
        end
        local_tind_start = local_tind_end + 1
        global_tind_start = global_tind_end + 1
    end

    if isa(iz, mk_int)
        thisdim = ndims(f_global) - 3
        f_global = selectdim(f_global, thisdim, 1)
    end
    if isa(ir, mk_int)
        thisdim = ndims(f_global) - 2
        f_global = selectdim(f_global, thisdim, 1)
    end
    if isa(t_range, mk_int)
        thisdim = ndims(f_global)
        f_global = selectdim(f_global, thisdim, 1)
    end

    return f_global
end

"""
Read a slice of a neutral distribution function

run_names is a tuple. If it has more than one entry, this means that there are multiple
restarts (which are sequential in time), so concatenate the data from each entry together.

The slice to take is specified by the keyword arguments.
"""
function load_distributed_neutral_pdf_slice(run_names::Tuple, nblocks::Tuple, t_range,
                                            n_species::mk_int, r::coordinate,
                                            z::coordinate, vzeta::coordinate,
                                            vr::coordinate, vz::coordinate; isn=nothing,
                                            ir=nothing, iz=nothing, ivzeta=nothing,
                                            ivr=nothing, ivz=nothing)
    # dimension of pdf is [vpa,vperp,z,r,species,t]

    result_dims = mk_int[]
    if ivz === nothing
        ivz = 1:vz.n_global
        push!(result_dims, vz.n_global)
    elseif !isa(ivz, mk_int)
        push!(result_dims, length(ivz))
    end
    if ivr === nothing
        ivr = 1:vr.n_global
        push!(result_dims, vr.n_global)
    elseif !isa(ivr, mk_int)
        push!(result_dims, length(ivr))
    end
    if ivzeta === nothing
        ivzeta = 1:vzeta.n_global
        push!(result_dims, vzeta.n_global)
    elseif !isa(ivzeta, mk_int)
        push!(result_dims, length(ivzeta))
    end
    if iz === nothing
        iz = 1:z.n_global
        push!(result_dims, z.n_global)
    elseif isa(iz, mk_int)
        push!(result_dims, 1)
    else
        push!(result_dims, length(iz))
    end
    if ir === nothing
        ir = 1:r.n_global
        push!(result_dims, r.n_global)
    elseif isa(ir, mk_int)
        push!(result_dims, 1)
    else
        push!(result_dims, length(ir))
    end
    if isn === nothing
        isn = 1:n_species
        push!(result_dims, n_species)
    elseif !isa(isn, mk_int)
        push!(result_dims, length(isn))
    else
        push!(result_dims, 1)
    end
    push!(result_dims, length(t_range))

    f_global = allocate_float(result_dims...)

    local_tind_start = 1
    local_tind_end = -1
    global_tind_start = 1
    global_tind_end = -1
    for (run_name, nb) in zip(run_names, nblocks)
        for iblock in 0:nb-1
            fid = open_readonly_output_file(run_name, "dfns", iblock=iblock, printout=false)

            z_irank, z_nrank, r_irank, r_nrank = load_rank_data(fid)

            # max index set to avoid double assignment of repeated points
            # nr/nz if irank = nrank-1, (nr-1)/(nz-1) otherwise
            imax_r = (r_irank == r.nrank - 1 ? r.n : r.n - 1)
            imax_z = (z_irank == z.nrank - 1 ? z.n : z.n - 1)
            local_r_range = 1:imax_r
            local_z_range = 1:imax_z
            global_r_range = iglobal_func(1, r_irank, r.n):iglobal_func(imax_r, r_irank, r.n)
            global_z_range = iglobal_func(1, z_irank, z.n):iglobal_func(imax_z, z_irank, z.n)

            if ir !== nothing && !any(i ∈ global_r_range for i in ir)
                # No data for the slice on this rank
                continue
            elseif isa(ir, StepRange)
                # Note that `findfirst(test, array)` returns the index `i` of the first
                # element of `array` for which `test(array[i])` is `true`.
                # `findlast()` similarly finds the index of the last element...
                start_ind = findfirst(i -> i>=ir.start, global_r_range)
                start = global_r_range[start_ind]
                stop_ind = findlast(i -> i<=ir.stop, global_r_range)
                stop = global_r_range[stop_ind]
                local_r_range = (local_r_range.start + start - global_r_range.start):ir.step:(local_r_range.stop + stop - global_r_range.stop)
                global_r_range = findfirst(i->i ∈ global_r_range, ir):findlast(i->i ∈ global_r_range, ir)
            elseif isa(ir, UnitRange)
                start_ind = findfirst(i -> i>=ir.start, global_r_range)
                start = global_r_range[start_ind]
                stop_ind = findlast(i -> i<=ir.stop, global_r_range)
                stop = global_r_range[stop_ind]
                local_r_range = (local_r_range.start + start - global_r_range.start):(local_r_range.stop + stop - global_r_range.stop)
                global_r_range = findfirst(i->i ∈ global_r_range, ir):findlast(i->i ∈ global_r_range, ir)
            elseif isa(ir, mk_int)
                local_r_range = ir - (global_r_range.start - 1)
                global_r_range = ir
            end
            if iz !== nothing && !any(i ∈ global_z_range for i in iz)
                # No data for the slice on this rank
                continue
            elseif isa(iz, StepRange)
                # Note that `findfirst(test, array)` returns the index `i` of the first
                # element of `array` for which `test(array[i])` is `true`.
                # `findlast()` similarly finds the index of the last element...
                start_ind = findfirst(i -> i>=iz.start, global_z_range)
                start = global_z_range[start_ind]
                stop_ind = findlast(i -> i<=iz.stop, global_z_range)
                stop = global_z_range[stop_ind]
                local_z_range = (local_z_range.start + start - global_z_range.start):iz.step:(local_z_range.stop + stop - global_z_range.stop)
                global_z_range = findfirst(i->i ∈ global_z_range, iz):findlast(i->i ∈ global_z_range, iz)
            elseif isa(iz, UnitRange)
                start_ind = findfirst(i -> i>=iz.start, global_z_range)
                start = global_z_range[start_ind]
                stop_ind = findlast(i -> i<=iz.stop, global_z_range)
                stop = global_z_range[stop_ind]
                local_z_range = (local_z_range.start + start - global_z_range.start):(local_z_range.stop + stop - global_z_range.stop)
                global_z_range = findfirst(i->i ∈ global_z_range, iz):findlast(i->i ∈ global_z_range, iz)
            elseif isa(iz, mk_int)
                local_z_range = iz - (global_z_range.start - 1)
                global_z_range = iz
            end

            f_local_slice = load_neutral_pdf_data(fid)

            if local_tind_start > 1
                # The run being loaded is a restart (as local_tind_start=1 for the first
                # run), so skip the first point, as this is a duplicate of the last point
                # of the previous restart
                skip_first = 1
            else
                skip_first = 0
            end
            ntime_local = size(f_local_slice, ndims(f_local_slice)) - skip_first
            local_tind_end = local_tind_start + ntime_local - 1
            local_t_range = collect(it - local_tind_start + 1 + skip_first
                                    for it ∈ t_range
                                    if local_tind_start <= it <= local_tind_end)
            global_tind_end = global_tind_start + length(local_t_range) - 1

            f_global_slice = selectdim(f_global, ndims(f_global),
                                       global_tind_start:global_tind_end)

            # Note: use selectdim() and get the dimension from thisdim because the actual
            # number of dimensions in f_global_slice, f_local_slice is different depending
            # on which combination of ivpa, ivperp, iz, ir, and is was passed.
            thisdim = ndims(f_local_slice) - 6
            f_local_slice = selectdim(f_local_slice, thisdim, ivz)

            thisdim = ndims(f_local_slice) - 5
            f_local_slice = selectdim(f_local_slice, thisdim, ivr)

            thisdim = ndims(f_local_slice) - 4
            f_local_slice = selectdim(f_local_slice, thisdim, ivzeta)

            thisdim = ndims(f_local_slice) - 3
            if isa(iz, mk_int)
                f_global_slice = selectdim(f_global_slice, thisdim, 1)
                f_local_slice = selectdim(f_local_slice, thisdim,
                                          ilocal_func(iz, z_irank, z.n))
            else
                f_global_slice = selectdim(f_global_slice, thisdim, global_z_range)
                f_local_slice = selectdim(f_local_slice, thisdim, local_z_range)
            end

            thisdim = ndims(f_local_slice) - 2
            if isa(ir, mk_int)
                f_global_slice = selectdim(f_global_slice, thisdim, 1)
                f_local_slice = selectdim(f_local_slice, thisdim,
                                          ilocal_func(ir, r_irank, r.n))
            else
                f_global_slice = selectdim(f_global_slice, thisdim, global_r_range)
                f_local_slice = selectdim(f_local_slice, thisdim, local_r_range)
            end

            thisdim = ndims(f_local_slice) - 1
            f_global_slice = selectdim(f_global_slice, thisdim, isn)
            f_local_slice = selectdim(f_local_slice, thisdim, isn)

            # Select time slice
            thisdim = ndims(f_local_slice)
            f_local_slice = selectdim(f_local_slice, thisdim, local_t_range)

            f_global_slice .= f_local_slice
            close(fid)
        end
        local_tind_start = local_tind_end + 1
        global_tind_start = global_tind_end + 1
    end

    if isa(iz, mk_int)
        thisdim = ndims(f_global) - 3
        f_global = selectdim(f_global, thisdim, 1)
    end
    if isa(ir, mk_int)
        thisdim = ndims(f_global) - 2
        f_global = selectdim(f_global, thisdim, 1)
    end
    if isa(isn, mk_int)
        thisdim = ndims(f_global) - 1
        f_global = selectdim(f_global, thisdim, 1)
    end
    if isa(t_range, mk_int)
        thisdim = ndims(f_global)
        f_global = selectdim(f_global, thisdim, 1)
    end

    return f_global
end

function iglobal_func(ilocal,irank,nlocal)
    if irank == 0
        iglobal = ilocal
    elseif irank > 0 && ilocal >= 1 && ilocal <= nlocal
        iglobal = ilocal + irank*(nlocal - 1)
    else
        error("ERROR: Invalid call to iglobal_func. ilocal=$ilocal, irank=$irank, "
              * "nlocal=$nlocal")
    end
    return iglobal
end

function ilocal_func(iglobal,irank,nlocal)
    return iglobal - irank*(nlocal - 1)
end

"""
    get_run_info_no_setup(run_dir...; itime_min=1, itime_max=0, itime_skip=1, dfns=false,
                          initial_electron=false)
    get_run_info_no_setup((run_dir, restart_index)...; itime_min=1, itime_max=0,
                          itime_skip=1, dfns=false, initial_electron=false)

Get file handles and other info for a single run

`run_dir` is either the directory to read output from (whose name should be the
`run_name`), or a moment_kinetics binary output file. If a file is passed, it is only used
to infer the directory and `run_name`, so it is possible for example to pass a
`.moments.h5` output file and also `dfns=true` and the `.dfns.h5` file will be the one
actually opened (as long as it exists).

`restart_index` can be given by passing a Tuple, e.g. `("runs/example", 42)` as the
positional argument. It specifies which restart to read if there are multiple restarts. If
no `restart_index` is given or if `nothing` is passed, read all restarts and concatenate
them. An integer value reads the restart with that index - `-1` indicates the latest
restart (which does not have an index).

Several runs can be loaded at the same time by passing multiple positional arguments. Each
argument can be a String `run_dir` giving a directory to read output from or a Tuple
`(run_dir, restart_index)` giving both a directory and a restart index (it is allowed to
mix Strings and Tuples in a call).

By default load data from moments files, pass `dfns=true` to load from distribution
functions files, or `initial_electron=true` and `dfns=true` to load from initial electron
state files, or `electron_debug=true` and `dfns=true` to load from electron debug files.

The `itime_min`, `itime_max` and `itime_skip` options can be used to select only a slice
of time points when loading data. In `makie_post_process` these options are read from the
input (if they are set) before `get_run_info_no_setup()` is called, so that the `run_info`
returned can be passed to
`makie_post_processing.setup_makie_post_processing_input!()`, to be used for
defaults for the remaining options. If either `itime_min` or `itime_max` are ≤0, their
values are used as offsets from the final time index of the run.
"""
function get_run_info_no_setup(run_dir::Union{AbstractString,Tuple{AbstractString,Union{Int,Nothing}}}...;
                               itime_min=1, itime_max=0, itime_skip=1, dfns=false,
                               initial_electron=false, electron_debug=false)
    if length(run_dir) == 0
        error("No run_dir passed")
    end
    if initial_electron && !dfns
        error("When `initial_electron=true` is passed, `dfns=true` must also be passed")
    end
    if electron_debug && !dfns
        error("When `electron_debug=true` is passed, `dfns=true` must also be passed")
    end
    if length(run_dir) > 1
        run_info = Tuple(get_run_info_no_setup(r; itime_min=itime_min,
                                               itime_max=itime_max, itime_skip=itime_skip,
                                               dfns=dfns,
                                               initial_electron=initial_electron,
                                               electron_debug=electron_debug)
                         for r ∈ run_dir)
        return run_info
    end

    this_run_dir = run_dir[1]
    if isa(this_run_dir, Tuple)
        if length(this_run_dir) != 2
            error("When a Tuple is passed for run_dir, expect it to have length 2. Got "
                  * "$this_run_dir")
        end
        this_run_dir, restart_index = this_run_dir
    else
        restart_index = nothing
    end

    if !isa(this_run_dir, AbstractString) || !isa(restart_index, Union{Int,Nothing})
        error("Expected all `run_dir` arguments to be `String` or `(String, Int)` or "
              * "`(String, Nothing)`. Got $run_dir")
    end

    if isfile(this_run_dir)
        # this_run_dir is actually a filename. Assume it is a moment_kinetics output file
        # and infer the directory and the run_name from the filename.

        filename = basename(this_run_dir)
        this_run_dir = dirname(this_run_dir)

        if occursin(".moments.", filename)
            run_name = split(filename, ".moments.")[1]
        elseif occursin(".dfns.", filename)
            run_name = split(filename, ".dfns.")[1]
        elseif occursin(".initial_electron.", filename)
            run_name = split(filename, ".initial_electron.")[1]
        elseif occursin(".electron_debug.", filename)
            run_name = split(filename, ".electron_debug.")[1]
        else
            error("Cannot recognise '$this_run_dir/$filename' as a moment_kinetics output file")
        end
    elseif isdir(this_run_dir)
        # Normalise by removing any trailing slash - with a slash basename() would return an
        # empty string
        this_run_dir = rstrip(this_run_dir, '/')

        run_name = basename(this_run_dir)
    else
        error("$this_run_dir does not exist")
    end

    base_prefix = joinpath(this_run_dir, run_name)
    if restart_index === nothing
        # Find output files from all restarts in the directory
        counter = 1
        run_prefixes = Vector{String}()
        if initial_electron
            while true
                # Test if output files exist for this value of counter
                prefix_with_count = base_prefix * "_$counter"
                if length(glob(basename(prefix_with_count) * ".initial_elctron*.h5", dirname(prefix_with_count))) > 0 ||
                    length(glob(basename(prefix_with_count) * ".initial_elctron*.cdf", dirname(prefix_with_count))) > 0

                    push!(run_prefixes, prefix_with_count)
                else
                    # No more output files found
                    break
                end
                counter += 1
            end
        else
            while true
                # Test if output files exist for this value of counter
                prefix_with_count = base_prefix * "_$counter"
                if length(glob(basename(prefix_with_count) * ".dfns*.h5", dirname(prefix_with_count))) > 0 ||
                    length(glob(basename(prefix_with_count) * ".dfns*.cdf", dirname(prefix_with_count))) > 0 ||
                    length(glob(basename(prefix_with_count) * ".moments*.h5", dirname(prefix_with_count))) > 0 ||
                    length(glob(basename(prefix_with_count) * ".moments*.cdf", dirname(prefix_with_count))) > 0

                    push!(run_prefixes, prefix_with_count)
                else
                    # No more output files found
                    break
                end
                counter += 1
            end
        end
        # Add the final run which does not have a '_$counter' suffix
        push!(run_prefixes, base_prefix)
        run_prefixes = tuple(run_prefixes...)
    elseif restart_index == -1
        run_prefixes = (base_prefix,)
    elseif restart_index > 0
        run_prefixes = (base_prefix * "_$restart_index",)
    else
        error("Invalid restart_index=$restart_index")
    end

    if initial_electron
        ext = "initial_electron"
    elseif electron_debug
        ext = "electron_debug"
    elseif dfns
        ext = "dfns"
    else
        ext = "moments"
    end

    has_data = all(length(glob(basename(p) * ".$ext*.h5", dirname(p))) > 0 ||
                   length(glob(basename(p) * ".$ext*.cdf", dirname(p))) > 0
                   for p ∈ run_prefixes)
    if !has_data
        println("No $ext data found for $run_prefixes, skipping $ext")
        return nothing
    end

    fids0 = Tuple(open_readonly_output_file(r, ext, printout=false)
                         for r ∈ run_prefixes)
    nblocks = Tuple(load_block_data(f)[1] for f ∈ fids0)
    if all(n == 1 for n ∈ nblocks)
        # Did not use distributed memory, or used parallel_io
        parallel_io = true
    else
        parallel_io = false
    end

    nt_unskipped, time, restarts_nt = load_time_data(fids0)
    if itime_min <= 0
        itime_min = nt_unskipped + itime_min
    end
    if itime_max <= 0
        itime_max = nt_unskipped + itime_max
    end
    time = time[itime_min:itime_skip:itime_max]
    nt = length(time)

    # Get input and coordinates from the final restart
    file_final_restart = fids0[end]

    input = load_input(file_final_restart)

    # obtain input options from moment_kinetics_input.jl
    # and check input to catch errors
    io_input, evolve_moments, t_input, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
        composition, species, collisions, geometry, drive_input, external_source_settings,
        num_diss_params, manufactured_solns_input = mk_input(input; warn_unexpected=true)

    n_ion_species, n_neutral_species = load_species_data(file_final_restart)
    evolve_density, evolve_upar, evolve_p = load_mk_options(file_final_restart)

    z_local, z_local_spectral, z_chunk_size =
        load_coordinate_data(file_final_restart, "z")
    r_local, r_local_spectral, r_chunk_size =
        load_coordinate_data(file_final_restart, "r")
    r, r_spectral, z, z_spectral = construct_global_zr_coords(r_local, z_local)

    vperp, vperp_spectral, vperp_chunk_size =
        load_coordinate_data(file_final_restart, "vperp")
    vpa, vpa_spectral, vpa_chunk_size =
        load_coordinate_data(file_final_restart, "vpa")

    if n_neutral_species > 0
        vzeta, vzeta_spectral, vzeta_chunk_size =
            load_coordinate_data(file_final_restart, "vzeta")
        vr, vr_spectral, vr_chunk_size =
            load_coordinate_data(file_final_restart, "vr")
        vz, vz_spectral, vz_chunk_size =
            load_coordinate_data(file_final_restart, "vz")
    else
        dummy_adv_input = advection_input("default", 1.0, 0.0, 0.0)
        dummy_comm = MPI.COMM_NULL
        dummy_input = OptionsDict("dummy" => OptionsDict())
        vzeta, vzeta_spectral = define_coordinate(dummy_input, "dummy"; ignore_MPI = true)
        vzeta_chunk_size = 1
        vr, vr_spectral = define_coordinate(dummy_input, "dummy"; ignore_MPI = true)
        vr_chunk_size = 1
        vz, vz_spectral = define_coordinate(dummy_input, "dummy"; ignore_MPI = true)
        vz_chunk_size = 1
    end

    overview = get_group(fids0[1], "overview")
    nrank = load_variable(overview, "nrank")
    block_size = load_variable(overview, "block_size")

    # Get variable names just from the first restart, for simplicity
    variable_names = get_variable_keys(get_group(fids0[1], "dynamic_data"))
    if initial_electron
        evolving_variables = ("f_electron",)
    else
        evolving_variables = ["f"]
        if evolve_density
            push!(evolving_variables, "density")
        end
        if evolve_upar
            push!(evolving_variables, "parallel_flow")
        end
        if evolve_p
            push!(evolving_variables, "pressure")
        end
        if composition.electron_physics ∈ (kinetic_electrons,
                                           kinetic_electrons_with_temperature_equation)
            push!(evolving_variables, "f_electron")
        end
        if composition.electron_physics ∈ (braginskii_fluid, kinetic_electrons,
                                           kinetic_electrons_with_temperature_equation)
            push!(evolving_variables, "electron_pressure")
        end
        if composition.n_neutral_species > 0
            push!(evolving_variables, "f_neutral")
            if evolve_density
                push!(evolving_variables, "density_neutral")
            end
            if evolve_upar
                push!(evolving_variables, "uz_neutral")
            end
            if evolve_p
                push!(evolving_variables, "p_neutral")
            end
        end
        evolving_variables = Tuple(evolving_variables)
    end

    # Assume the timing variables are the same in every restart - this may not always be
    # true, and might cause errors if some variables are missing for restarts after the
    # first.
    timing_group = get_group(fids0[1], "timing_data")
    timing_variable_names = collect(k for k in keys(timing_group)
                                    if startswith(k, "time:") || startswith(k, "ncalls:") ||
                                       startswith(k, "allocs:"))

    if parallel_io
        files = fids0
    else
        # Don't keep open files as read_distributed_zr_data!(), etc. open the files
        # themselves
        files = run_prefixes
        for f ∈ fids0
            close(f)
        end
    end

    run_info = (run_name=run_name, run_prefix=base_prefix, parallel_io=parallel_io,
                ext=ext, nblocks=nblocks, files=files, input=input,
                n_ion_species=n_ion_species, n_neutral_species=n_neutral_species,
                evolve_moments=evolve_moments, t_input=t_input, composition=composition,
                species=species, collisions=collisions, geometry=geometry,
                drive_input=drive_input, num_diss_params=num_diss_params,
                external_source_settings=external_source_settings,
                evolve_density=evolve_density, evolve_upar=evolve_upar,
                evolve_p=evolve_p,
                manufactured_solns_input=manufactured_solns_input, nt=nt,
                nt_unskipped=nt_unskipped, restarts_nt=restarts_nt, itime_min=itime_min,
                itime_skip=itime_skip, itime_max=itime_max, time=time, r=r, z=z,
                vperp=vperp, vpa=vpa, vzeta=vzeta, vr=vr, vz=vz, r_local=r_local,
                z_local=z_local, r_spectral=r_spectral, z_spectral=z_spectral,
                vperp_spectral=vperp_spectral, vpa_spectral=vpa_spectral,
                vzeta_spectral=vzeta_spectral, vr_spectral=vr_spectral,
                vz_spectral=vz_spectral, nrank=nrank, block_size=block_size,
                r_chunk_size=r_chunk_size, z_chunk_size=z_chunk_size,
                vperp_chunk_size=vperp_chunk_size, vpa_chunk_size=vpa_chunk_size,
                vzeta_chunk_size=vzeta_chunk_size, vr_chunk_size=vr_chunk_size,
                vz_chunk_size=vz_chunk_size, variable_names=variable_names,
                evolving_variables=evolving_variables,
                timing_variable_names=timing_variable_names, dfns=dfns)

    return run_info
end

"""
    close_run_info(run_info)

Close all the files in a run_info NamedTuple.
"""
function close_run_info(run_info)
    if run_info === nothing
        return nothing
    end
    if !run_info.parallel_io
        # Files are not kept open, so nothing to do
        return nothing
    end

    for f ∈ run_info.files
        close(f)
    end

    return nothing
end

"""
    postproc_load_variable(run_info, variable_name; it=nothing, is=nothing,
                           ir=nothing, iz=nothing, ivperp=nothing, ivpa=nothing,
                           ivzeta=nothing, ivr=nothing, ivz=nothing, group=nothing)

Load a variable

`run_info` is the information about a run returned by
`makie_post_processing.get_run_info()`.

`variable_name` is the name of the variable to load.

The keyword arguments `it`, `is`, `ir`, `iz`, `ivperp`, `ivpa`, `ivzeta`, `ivr`, and `ivz`
can be set to an integer or a range (e.g. `3:8` or `3:2:8`) to select subsets of the data.
Only the data for the subset requested will be loaded from the output file (mostly - when
loading fields or moments from runs which used `parallel_io = false`, the full array will
be loaded and then sliced).

If a variable is found in a group other than "dynamic_data", the group name should be
passed to the `group` argument.
"""
function postproc_load_variable(run_info, variable_name; it=nothing, is=nothing,
                                ir=nothing, iz=nothing, ivperp=nothing, ivpa=nothing,
                                ivzeta=nothing, ivr=nothing, ivz=nothing, group=nothing)

    nt = run_info.nt

    if it === nothing
        it = run_info.itime_min:run_info.itime_skip:run_info.itime_max
    elseif isa(it, mk_int)
        nt = 1
        it = collect(run_info.itime_min:run_info.itime_skip:run_info.itime_max)[it]
    else
        nt = length(it)
    end
    if is === nothing
        # Can't use 'n_species' in a similar way to the way we treat other dims, because
        # we don't know here if the variable is for ions or neutrals.
        # Use Colon operator `:` when slice argument is `nothing` as when we pass that as
        # an 'index', it selects the whole dimension. Brackets are needed around the `:`
        # when assigning it to variables, etc. to avoid an error "LoadError: syntax:
        # newline not allowed after ":" used for quoting".
        is = (:)
    elseif isa(is, mk_int)
        nspecies = 1
    else
        nspecies = length(is)
    end
    if ir === nothing
        nr = run_info.r.n
        ir = 1:nr
    elseif isa(ir, mk_int)
        nr = 1
    else
        nr = length(ir)
    end
    if iz === nothing
        nz = run_info.z.n
        iz = 1:nz
    elseif isa(iz, mk_int)
        nz = 1
    else
        nz = length(iz)
    end
    if ivperp === nothing
        if :vperp ∈ keys(run_info)
            # v-space coordinates only present if run_info contains distribution functions
            nvperp = run_info.vperp === nothing ? 1 : run_info.vperp.n
            ivperp = 1:nvperp
        else
            nvperp = nothing
            ivperp = nothing
        end
    elseif isa(ivperp, mk_int)
        nvperp = 1
    else
        nvperp = length(ivperp)
    end
    if ivpa === nothing
        if :vpa ∈ keys(run_info)
            # v-space coordinates only present if run_info contains distribution functions
            nvpa = run_info.vpa === nothing ? 1 : run_info.vpa.n
            ivpa = 1:nvpa
        else
            nvpa = nothing
            ivpa = nothing
        end
    elseif isa(ivpa, mk_int)
        nvpa = 1
    else
        nvpa = length(ivpa)
    end
    if ivzeta === nothing
        if :vzeta ∈ keys(run_info)
            # v-space coordinates only present if run_info contains distribution functions
            nvzeta = run_info.vzeta === nothing ? 1 : run_info.vzeta.n
            ivzeta = 1:nvzeta
        else
            nvzeta = nothing
            ivzeta = nothing
        end
    elseif isa(ivzeta, mk_int)
        nvzeta = 1
    else
        nvzeta = length(ivzeta)
    end
    if ivr === nothing
        if :vr ∈ keys(run_info)
            # v-space coordinates only present if run_info contains distribution functions
            nvr = run_info.vr === nothing ? 1 : run_info.vr.n
            ivr = 1:nvr
        else
            nvr = nothing
            ivr = nothing
        end
    elseif isa(ivr, mk_int)
        nvr = 1
    else
        nvr = length(ivr)
    end
    if ivz === nothing
        if :vz ∈ keys(run_info)
            # v-space coordinates only present if run_info contains distribution functions
            nvz = run_info.vz === nothing ? 1 : run_info.vz.n
            ivz = 1:nvz
        else
            nvz = nothing
            ivz = nothing
        end
    elseif isa(ivz, mk_int)
        nvz = 1
    else
        nvz = length(ivz)
    end

    if run_info.parallel_io
        if group === nothing
            group = "dynamic_data"
        end

        # Get HDF5/NetCDF variables directly and load slices
        variable = Tuple(get_group(f, group)[variable_name]
                         for f ∈ run_info.files)
        nd = ndims(variable[1])

        if nd == 1
            # Time-dependent scalar with dimensions (t)
            # Might be an mk_int, so handle type carefully
            dims = Vector{mk_int}()
            !isa(it, mk_int) && push!(dims, nt)
            vartype = typeof(variable[1][1])
            if vartype == mk_int
                result = allocate_int(dims...)
            elseif vartype == mk_float
                result = allocate_float(dims...)
            else
                result = Array{vartype}(undef, dims...)
            end
        elseif nd == 2
            # Time-dependent vector with dimensions (unique_dim,t).
            # The dimension that is not time is not a coordinate, so do not give any
            # option to slice it - always just return the full length.
            # Might be an mk_int, so handle type carefully
            dims = Vector{mk_int}()
            push!(dims, size(variable[1], 1))
            !isa(it, mk_int) && push!(dims, nt)
            vartype = typeof(variable[1][1,1])
            if vartype == mk_int
                result = allocate_int(dims...)
            elseif vartype == mk_float
                result = allocate_float(dims...)
            else
                result = Array{vartype}(undef, dims...)
            end
        elseif nd == 3
            # EM variable with dimensions (z,r,t)
            dims = Vector{mk_int}()
            !isa(iz, mk_int) && push!(dims, nz)
            !isa(ir, mk_int) && push!(dims, nr)
            !isa(it, mk_int) && push!(dims, nt)
            result = allocate_float(dims...)
        elseif nd == 4
            # moment variable with dimensions (z,r,s,t)
            # Get nspecies from the variable, not from run_info, because it might be
            # either ion or neutral
            dims = Vector{mk_int}()
            !isa(iz, mk_int) && push!(dims, nz)
            !isa(ir, mk_int) && push!(dims, nr)
            if is === (:)
                nspecies = size(variable[1], 3)
                push!(dims, nspecies)
            elseif !isa(is, mk_int)
                push!(dims, nspecies)
            end
            !isa(it, mk_int) && push!(dims, nt)
            result = allocate_float(dims...)
        elseif nd == 5
            # electron distribution function variable with dimensions (vpa,vperp,z,r,t)
            dims = Vector{mk_int}()
            !isa(ivpa, mk_int) && push!(dims, nvpa)
            !isa(ivperp, mk_int) && push!(dims, nvperp)
            !isa(iz, mk_int) && push!(dims, nz)
            !isa(ir, mk_int) && push!(dims, nr)
            !isa(it, mk_int) && push!(dims, nt)
            result = allocate_float(dims...)
        elseif nd == 6
            # ion distribution function variable with dimensions (vpa,vperp,z,r,s,t)
            nspecies = size(variable[1], 5)
            dims = Vector{mk_int}()
            !isa(ivpa, mk_int) && push!(dims, nvpa)
            !isa(ivperp, mk_int) && push!(dims, nvperp)
            !isa(iz, mk_int) && push!(dims, nz)
            !isa(ir, mk_int) && push!(dims, nr)
            if is === (:)
                push!(dims, nspecies)
            elseif !isa(is, mk_int)
                push!(dims, nspecies)
            end
            !isa(it, mk_int) && push!(dims, nt)
            result = allocate_float(dims...)
        elseif nd == 7
            # neutral distribution function variable with dimensions (vz,vr,vzeta,z,r,s,t)
            nspecies = size(variable[1], 6)
            dims = Vector{mk_int}()
            !isa(ivz, mk_int) && push!(dims, nvz)
            !isa(ivr, mk_int) && push!(dims, nvr)
            !isa(ivzeta, mk_int) && push!(dims, nvzeta)
            !isa(iz, mk_int) && push!(dims, nz)
            !isa(ir, mk_int) && push!(dims, nr)
            if is === (:)
                push!(dims, nspecies)
            elseif !isa(is, mk_int)
                push!(dims, nspecies)
            end
            !isa(it, mk_int) && push!(dims, nt)
            result = allocate_float(dims...)
        else
            error("Unsupported number of dimensions ($nd) for '$variable_name'.")
        end

        local_it_start = 1
        global_it_start = 1
        for v ∈ variable
            # For restarts, the first time point is a duplicate of the last time
            # point of the previous restart. Use `offset` to skip this point.
            offset = local_it_start == 1 ? 0 : 1
            local_nt = size(v, nd) - offset
            local_it_end = local_it_start+local_nt-1

            if isa(it, mk_int)
                tind = it - local_it_start + 1
                if tind < 1
                    error("Trying to select time index before the beginning of this "
                          * "restart, should have finished already")
                elseif tind <= local_nt
                    # tind is within this restart's time range, so get result
                    if nd == 1
                        result .= v[tind]
                    elseif nd == 2
                        result .= v[:,tind]
                    elseif nd == 3
                        result .= v[iz,ir,tind]
                    elseif nd == 4
                        result .= v[iz,ir,is,tind]
                    elseif nd == 5
                        result .= v[ivpa,ivperp,iz,ir,tind]
                    elseif nd == 6
                        result .= v[ivpa,ivperp,iz,ir,is,tind]
                    elseif nd == 7
                        result .= v[ivz,ivr,ivzeta,iz,ir,is,tind]
                    else
                        error("Unsupported combination nd=$nd, ir=$ir, iz=$iz, ivperp=$ivperp "
                              * "ivpa=$ivpa, ivzeta=$ivzeta, ivr=$ivr, ivz=$ivz.")
                    end

                    # Already got the data for `it`, so end loop
                    break
                end
            else
                tinds = collect(i - local_it_start + 1 + offset for i ∈ it
                                if local_it_start <= i <= local_it_end)
                # Convert tinds to slice, as we know the spacing is constant
                if length(tinds) != 0
                    # There is some data in this file
                    if length(tinds) > 1
                        tstep = tinds[2] - tinds[begin]
                    else
                        tstep = 1
                    end
                    tinds = tinds[begin]:tstep:tinds[end]
                    global_it_end = global_it_start + length(tinds) - 1

                    if nd == 1
                        selectdim(result, ndims(result), global_it_start:global_it_end) .= v[tinds]
                    elseif nd == 2
                        selectdim(result, ndims(result), global_it_start:global_it_end) .= v[:,tinds]
                    elseif nd == 3
                        selectdim(result, ndims(result), global_it_start:global_it_end) .= v[iz,ir,tinds]
                    elseif nd == 4
                        selectdim(result, ndims(result), global_it_start:global_it_end) .= v[iz,ir,is,tinds]
                    elseif nd == 5
                        selectdim(result, ndims(result), global_it_start:global_it_end) .= v[ivpa,ivperp,iz,ir,tinds]
                    elseif nd == 6
                        selectdim(result, ndims(result), global_it_start:global_it_end) .= v[ivpa,ivperp,iz,ir,is,tinds]
                    elseif nd == 7
                        selectdim(result, ndims(result), global_it_start:global_it_end) .= v[ivz,ivr,ivzeta,iz,ir,is,tinds]
                    else
                        error("Unsupported combination nd=$nd, ir=$ir, iz=$iz, ivperp=$ivperp "
                              * "ivpa=$ivpa, ivzeta=$ivzeta, ivr=$ivr, ivz=$ivz.")
                    end

                    global_it_start = global_it_end + 1
                end
            end

            local_it_start = local_it_end + 1
        end
    else
        # Use existing distributed I/O loading functions
        diagnostic_variable = false
        if variable_name ∈ tuple(em_variables..., electron_moment_variables...)
            nd = 3
        elseif variable_name ∈ electron_dfn_variables
            nd = 5
        elseif variable_name ∈ ion_dfn_variables
            nd = 6
        elseif variable_name ∈ neutral_dfn_variables
            nd = 7
        elseif variable_name ∈ tuple(ion_moment_variables..., neutral_moment_variables...)
            # Ion or neutral moment variable
            nd = 4
        else
            # Diagnostic variable that does not depend on coordinates, and should be the
            # same in every output file (so can just read from the first one).
            diagnostic_variable = true
        end

        if diagnostic_variable
            if group === nothing
                group = "dynamic_data"
            end
            fid = open_readonly_output_file(run_info.files[1], run_info.ext, iblock=0)
            this_group = get_group(fid, group)
            result = load_variable(this_group, variable_name)
            result = selectdim(result, ndims(result), it)
        elseif nd == 3
            result = allocate_float(run_info.z.n, run_info.r.n, run_info.nt)
            read_distributed_zr_data!(result, variable_name, run_info.files,
                                      run_info.ext, run_info.nblocks, run_info.z_local.n,
                                      run_info.r_local.n, run_info.itime_skip,
                                      group=group)
            result = result[iz,ir,it]
        elseif nd == 4
            # If we ever have neutrals included but n_neutral_species != n_ion_species,
            # then this will fail - in that case would need some way to specify that we
            # need to read a neutral moment variable rather than an ion moment variable
            # here.
            result = allocate_float(run_info.z.n, run_info.r.n, run_info.n_ion_species,
                                    run_info.nt)
            read_distributed_zr_data!(result, variable_name, run_info.files,
                                      run_info.ext, run_info.nblocks, run_info.z_local.n,
                                      run_info.r_local.n, run_info.itime_skip,
                                      group=group)
            result = result[iz,ir,is,it]
        elseif nd === 5
            if group !== nothing
                error("group argument not supported when reading electron pdf with "
                      * "parallel_io=false. Got group=$group")
            end
            result = load_distributed_electron_pdf_slice(run_info.files, run_info.nblocks,
                                                         it, run_info.n_ion_species,
                                                         run_info.r_local,
                                                         run_info.z_local, run_info.vperp,
                                                         run_info.vpa; ir=ir, iz=iz,
                                                         ivperp=ivperp, ivpa=ivpa)
        elseif nd === 6
            if group !== nothing
                error("group argument not supported when reading ion pdf with "
                      * "parallel_io=false. Got group=$group")
            end
            result = load_distributed_ion_pdf_slice(run_info.files, run_info.nblocks, it,
                                                    run_info.n_ion_species,
                                                    run_info.r_local, run_info.z_local,
                                                    run_info.vperp, run_info.vpa;
                                                    is=(is === (:) ? nothing : is),
                                                    ir=ir, iz=iz, ivperp=ivperp,
                                                    ivpa=ivpa)
        elseif nd === 7
            if group !== nothing
                error("group argument not supported when reading neutral pdf with "
                      * "parallel_io=false. Got group=$group")
            end
            result = load_distributed_neutral_pdf_slice(run_info.files, run_info.nblocks,
                                                        it, run_info.n_ion_species,
                                                        run_info.r_local,
                                                        run_info.z_local, run_info.vzeta,
                                                        run_info.vr, run_info.vz;
                                                        isn=(is === (:) ? nothing : is),
                                                        ir=ir, iz=iz, ivzeta=ivzeta,
                                                        ivr=ivr, ivz=ivz)
        end
    end

    return result
end

"""
Load all the moment variables, returning them in a NamedTuple.

Intended for internal use inside [`get_variable`](@ref).
"""
function _get_all_moment_variables(run_info; it=nothing, kwargs...)
    if isa(it, Integer)
        # Actually load a length-1 slice so that we can use the results in a loop over
        # `it`.
        it = it:it
    end
    pairs = Pair{Symbol,Any}[]
    for v ∈ all_moment_variables
        println("getting ", v)
        try
            push!(pairs, Symbol(v)=>get_variable(run_info, v; it=it, kwargs...))
        catch e
            if !isa(e, KeyError)
                rethrow()
            end
        end
    end
    return (; pairs...)
end

"""
Create fake scratch and moments structs at time index `it` from the variables in
`all_moments`.
"""
function _get_fake_moments_fields_scratch(all_moments, it; ion_extra::Tuple=(),
                                          electron_extra::Tuple=(),
                                          neutral_extra::Tuple=())
    function make_struct(; kwargs...)
        function get_var(variable_name_or_array)
            if isa(variable_name_or_array, Symbol)
                var = all_moments[variable_name_or_array]
            else
                var = variable_name_or_array
            end
            return selectdim(var, ndims(var), it)
        end
        return (; (field_name=>get_var(variable_name_or_array)
                   for (field_name, variable_name_or_array) ∈ kwargs
                   if !isa(variable_name_or_array, Symbol)
                      || variable_name_or_array ∈ keys(all_moments)
                  )...)
    end

    ion_moments = make_struct(; dens=:density, upar=:parallel_flow,
        p=:pressure, ppar=:parallel_pressure, qpar=:parallel_heat_flux,
        vth=:thermal_speed, temp=:temperature, ddens_dz=:ddens_dz,
        ddens_dz_upwind=:ddens_dz_upwind, dupar_dz=:dupar_dz,
        dupar_dz_upwind=:dupar_dz_upwind, dp_dz=:dp_dz, dp_dz_upwind=:dp_dz_upwind,
        dppar_dz=:dppar_dz, dppar_dz_upwind=:dppar_dz_upwind, dvth_dz=:dvth_dz,
        dT_dz=:dT_dz, dqpar_dz=:dqpar_dz,
        external_source_amplitude=:external_source_amplitude,
        external_source_density_amplitude=:external_source_density_amplitude,
        external_source_momentum_amplitude=:external_source_momentum_amplitude,
        external_source_pressure_amplitude=:external_source_pressure_amplitude,
        external_source_controller_integral=:external_source_controller_integral,
        ion_extra...)

    electron_moments = make_struct(; dens=:electron_density, upar=:electron_parallel_flow,
        p=:electron_pressure, ppar=:electron_parallel_pressure,
        qpar=:electron_parallel_heat_flux, vth=:electron_thermal_speed,
        temp=:electron_temperature, ddens_dz=:electron_ddens_dz,
        dupar_dz=:electron_dupar_dz, dppar_dz=:electron_dppar_dz,
        dvth_dz=:electron_dvth_dz, dT_dz=:electron_dT_dz, dqpar_dz=:electron_dqpar_dz,
        external_source_amplitude=:external_source_electron_amplitude,
        external_source_density_amplitude=:external_source_electron_density_amplitude,
        external_source_momentum_amplitude=:external_source_electron_momentum_amplitude,
        external_source_pressure_amplitude=:external_source_electron_pressure_amplitude,
        external_source_controller_integral=:external_electron_source_controller_integral,
        electron_extra...)

    neutral_moments = make_struct(; dens=:density_neutral, uz=:uz_neutral, p=:p_neutral,
        pz=:pz_neutral, qz=:qz_neutral, vth=:thermal_speed_neutral,
        temp=:temperature_neutral, ddens_dz=:neutral_ddens_dz,
        ddens_dz_upwind=:neutral_ddens_dz_upwind, duz_dz=:neutral_duz_dz,
        duz_dz_upwind=:neutral_duz_dz_upwind, dp_dz=:neutral_dp_dz,
        dp_dz_upwind=:neutral_dp_dz_upwind, dpz_dz=:neutral_dpz_dz,
        dvth_dz=:neutral_dvth_dz, dT_dz=:neutral_dT_dz, dqz_dz=:neutral_dqz_dz,
        external_source_amplitude=:external_source_neutral_amplitude,
        external_source_density_amplitude=:external_source_neutral_density_amplitude,
        external_source_momentum_amplitude=:external_source_neutral_momentum_amplitude,
        external_source_pressure_amplitude=:external_source_neutral_pressure_amplitude,
        external_source_controller_integral=:external_source_neutral_controller_integral,
        neutral_extra...)

    moments = (; ion=ion_moments, electron=electron_moments, neutral=neutral_moments)

    fields = make_struct(; phi=:phi, Er=:Er, Ez=:Ez)

    fvec = make_struct(; density=:density, upar=:parallel_flow, p=:pressure,
        electron_density=:electron_density, electron_upar=:electron_parallel_flow,
        electron_p=:electron_pressure, electron_temp=:electron_temperature,
        density_neutral=:density_neutral, uz_neutral=:uz_neutral, p_neutral=:p_neutral)

    return moments, fields, fvec
end

"""
    get_variable(run_info::Tuple, variable_name; kwargs...)
    get_variable(run_info, variable_name; kwargs...)

Get an array (or Tuple of arrays, if `run_info` is a Tuple) of the data for
`variable_name` from `run_info`.

Some derived variables need to be calculated from the saved output, not just loaded from
file (with `postproc_load_variable`). This function takes care of that calculation, and
handles the case where `run_info` is a Tuple (which `postproc_load_data` does not handle).

`kwargs...` are passed through to `postproc_load_variable()`.
"""
function get_variable end

function get_variable(run_info::Tuple, variable_name; kwargs...)
    return Tuple(get_variable(ri, variable_name; kwargs...) for ri ∈ run_info)
end

function get_variable(run_info, variable_name; normalize_advection_speed_shape=true,
                      kwargs...)

    # Set up loop macros for serial operation, in case they are used by any functions
    # below.
    looping.setup_loop_ranges!(0, 1;
                               s=run_info.composition.n_ion_species,
                               sn=run_info.composition.n_neutral_species, r=run_info.r.n,
                               z=run_info.z.n, vperp=run_info.vperp.n, vpa=run_info.vpa.n,
                               vzeta=run_info.vzeta.n, vr=run_info.vr.n, vz=run_info.vz.n)

    # Select a slice of an time-series sized variable
    function select_slice_of_variable(variable::AbstractVector; it=nothing,
                                      is=nothing, ir=nothing, iz=nothing, ivperp=nothing,
                                      ivpa=nothing, ivzeta=nothing, ivr=nothing,
                                      ivz=nothing)
        if it !== nothing
            variable = selectdim(variable, 1, kwargs[:it])
        end

        return variable
    end

    # Select a slice of an ion distribution function sized variable
    function select_slice_of_variable(variable::AbstractArray{T,6} where T; it=nothing,
                                      is=nothing, ir=nothing, iz=nothing, ivperp=nothing,
                                      ivpa=nothing, ivzeta=nothing, ivr=nothing,
                                      ivz=nothing)
        if it !== nothing
            variable = selectdim(variable, 6, kwargs[:it])
        end
        if is !== nothing
            variable = selectdim(variable, 5, kwargs[:is])
        end
        if ir !== nothing
            variable = selectdim(variable, 4, kwargs[:ir])
        end
        if iz !== nothing
            variable = selectdim(variable, 3, kwargs[:iz])
        end
        if ivperp !== nothing
            variable = selectdim(variable, 2, kwargs[:ivperp])
        end
        if ivpa !== nothing
            variable = selectdim(variable, 1, kwargs[:ivpa])
        end

        return variable
    end

    # Select a slice of an electron distribution function sized variable
    function select_slice_of_variable(variable::AbstractArray{T,5} where T; it=nothing,
                                      is=nothing, ir=nothing, iz=nothing, ivperp=nothing,
                                      ivpa=nothing, ivzeta=nothing, ivr=nothing,
                                      ivz=nothing)
        if it !== nothing
            variable = selectdim(variable, 5, kwargs[:it])
        end
        if ir !== nothing
            variable = selectdim(variable, 4, kwargs[:ir])
        end
        if iz !== nothing
            variable = selectdim(variable, 3, kwargs[:iz])
        end
        if ivperp !== nothing
            variable = selectdim(variable, 2, kwargs[:ivperp])
        end
        if ivpa !== nothing
            variable = selectdim(variable, 1, kwargs[:ivpa])
        end

        return variable
    end

    # Select a slice of a neutral distribution function sized variable
    function select_slice_of_variable(variable::AbstractArray{T,7} where T; it=nothing,
                                      is=nothing, ir=nothing, iz=nothing, ivperp=nothing,
                                      ivpa=nothing, ivzeta=nothing, ivr=nothing,
                                      ivz=nothing)
        if it !== nothing
            variable = selectdim(variable, 7, kwargs[:it])
        end
        if is !== nothing
            variable = selectdim(variable, 6, kwargs[:is])
        end
        if ir !== nothing
            variable = selectdim(variable, 5, kwargs[:ir])
        end
        if iz !== nothing
            variable = selectdim(variable, 4, kwargs[:iz])
        end
        if ivzeta !== nothing
            variable = selectdim(variable, 3, kwargs[:ivzeta])
        end
        if ivr !== nothing
            variable = selectdim(variable, 2, kwargs[:ivr])
        end
        if ivz !== nothing
            variable = selectdim(variable, 1, kwargs[:ivz])
        end

        return variable
    end

    # Get a 'per step' value from a saved 'cumulative' value. E.g. 'iterations per step'
    # from a saved 'cumulative total iterations'
    function get_per_step_from_cumulative_variable(run_info, varname::AbstractString;
                                                   kwargs...)
        variable = get_variable(run_info, varname; kwargs...)
        tdim = ndims(variable)
        for i ∈ size(variable, tdim):-1:2
            selectdim(variable, tdim, i) .-= selectdim(variable, tdim, i-1)
        end

        # Per-step count does not make sense for the first step, so make sure element-1 is
        # zero.
        selectdim(variable, tdim, 1) .= zero(first(variable))

        # Assume cumulative variables always increase, so if any value in the 'per-step'
        # variable is negative, it is because there was a restart where the cumulative
        # variable started over
        variable .= max.(variable, zero(first(variable)))

        return variable
    end

    function get_z_derivative_of_loaded_variable(x)
        dx_dz = similar(x)
        if :iz ∈ keys(kwargs) && kwargs[:iz] !== nothing
            error("Cannot take z-derivative when iz!==nothing")
        end
        if :is ∈ keys(kwargs) && isa(kwargs[:is], mk_int) && :ir ∈ keys(kwargs) && isa(kwargs[:ir], mk_int)
            for it ∈ 1:size(dx_dz, 2)
                @views derivative!(dx_dz[:,it], x[:,it], run_info.z, run_info.z_spectral)
            end
        elseif :ir ∈ keys(kwargs) && isa(kwargs[:ir], mk_int)
            for it ∈ 1:size(dx_dz, 3), is ∈ 1:size(dx_dz, 2)
                @views derivative!(dx_dz[:,is,it], x[:,is,it], run_info.z,
                                   run_info.z_spectral)
            end
        elseif :is ∈ keys(kwargs) && isa(kwargs[:is], mk_int)
            for it ∈ 1:size(dx_dz, 3), ir ∈ 1:run_info.r.n
                @views derivative!(dx_dz[:,is,it], x[:,is,it], run_info.z,
                                   run_info.z_spectral)
            end
        else
            for it ∈ 1:size(dx_dz, 4), is ∈ 1:size(dx_dz, 3), ir ∈ 1:run_info.r.n
                @views derivative!(dx_dz[:,ir,is,it], x[:,ir,is,it], run_info.z,
                                   run_info.z_spectral)
            end
        end
        return dx_dz
    end

    function get_upwind_z_derivative(x, upar)
        dx_dz = similar(x)
        if :iz ∈ keys(kwargs) && kwargs[:iz] !== nothing
            error("Cannot take z-derivative when iz!==nothing")
        end
        if :is ∈ keys(kwargs) && isa(kwargs[:is], mk_int) && :ir ∈ keys(kwargs) && isa(kwargs[:ir], mk_int)
            for it ∈ 1:size(dx_dz, 2)
                @views derivative!(dx_dz[:,it], x[:,it], run_info.z, .-upar[:,it],
                                   run_info.z_spectral)
            end
        elseif :ir ∈ keys(kwargs) && isa(kwargs[:ir], mk_int)
            for it ∈ 1:size(dx_dz, 3), is ∈ 1:size(dx_dz, 2)
                @views derivative!(dx_dz[:,is,it], x[:,is,it], run_info.z,
                                   .-upar[:,is,it], run_info.z_spectral)
            end
        elseif :is ∈ keys(kwargs) && isa(kwargs[:is], mk_int)
            for it ∈ 1:size(dx_dz, 3), ir ∈ 1:run_info.r.n
                @views derivative!(dx_dz[:,ir,it], x[:,ir,it], run_info.z,
                                   .-upar[:,ir,it], run_info.z_spectral)
            end
        else
            for it ∈ 1:size(dx_dz, 4), is ∈ 1:size(dx_dz, 3), ir ∈ 1:run_info.r.n
                @views derivative!(dx_dz[:,ir,is,it], x[:,ir,is,it], run_info.z,
                                   .-upar[:,ir,is,it], run_info.z_spectral)
            end
        end
        return dx_dz
    end

    function get_electron_z_derivative(x)
        dx_dz = similar(x)
        if :iz ∈ keys(kwargs) && kwargs[:iz] !== nothing
            error("Cannot take z-derivative when iz!==nothing")
        end
        if :ir ∈ keys(kwargs) && isa(kwargs[:ir], mk_int)
            for it ∈ 1:size(dx_dz, 2)
                @views derivative!(dx_dz[:,it], x[:,it], run_info.z,
                                   run_info.z_spectral)
            end
        else
            for it ∈ 1:size(dx_dz, 3), ir ∈ 1:run_info.r.n
                @views derivative!(dx_dz[:,ir,it], x[:,ir,it], run_info.z,
                                   run_info.z_spectral)
            end
        end
        return dx_dz
    end

    if variable_name == "temperature"
        vth = get_variable(run_info, "thermal_speed"; kwargs...)
        variable = 0.5 * vth.^2
    elseif variable_name == "ddens_dz"
        variable = get_z_derivative(run_info, "density"; kwargs...)
    elseif variable_name == "ddens_dz_upwind"
        n = get_variable(run_info, "density"; kwargs...)
        upar = get_variable(run_info, "parallel_flow"; kwargs...)
        variable = get_upwind_z_derivative(n, upar)
    elseif variable_name == "dupar_dz"
        variable = get_z_derivative(run_info, "parallel_flow"; kwargs...)
    elseif variable_name == "dupar_dz_upwind"
        upar = get_variable(run_info, "parallel_flow"; kwargs...)
        variable = get_upwind_z_derivative(upar, upar)
    elseif variable_name == "dp_dz"
        variable = get_z_derivative(run_info, "pressure"; kwargs...)
    elseif variable_name == "dp_dz_upwind"
        p = get_variable(run_info, "pressure"; kwargs...)
        upar = get_variable(run_info, "parallel_flow"; kwargs...)
        variable = get_upwind_z_derivative(p, upar)
    elseif variable_name == "dppar_dz"
        variable = get_z_derivative(run_info, "parallel_pressure"; kwargs...)
    elseif variable_name == "dppar_dz_upwind"
        ppar = get_variable(run_info, "parallel_pressure"; kwargs...)
        upar = get_variable(run_info, "parallel_flow"; kwargs...)
        variable = get_upwind_z_derivative(ppar, upar)
    elseif variable_name == "dvth_dz"
        variable = get_z_derivative(run_info, "thermal_speed"; kwargs...)
    elseif variable_name == "dT_dz"
        variable = get_z_derivative(run_info, "temperature"; kwargs...)
    elseif variable_name == "dqpar_dz"
        variable = get_z_derivative(run_info, "parallel_heat_flux"; kwargs...)
    elseif variable_name == "electron_ddens_dz"
        n = get_variable(run_info, "electron_density"; kwargs...)
        variable = get_electron_z_derivative(n)
    elseif variable_name == "electron_dupar_dz"
        upar = get_variable(run_info, "electron_parallel_flow"; kwargs...)
        variable = get_electron_z_derivative(upar)
    elseif variable_name == "electron_dp_dz"
        p = get_variable(run_info, "electron_pressure"; kwargs...)
        variable = get_electron_z_derivative(p)
    elseif variable_name == "electron_dppar_dz"
        ppar = get_variable(run_info, "electron_parallel_pressure"; kwargs...)
        variable = get_electron_z_derivative(ppar)
    elseif variable_name == "electron_dvth_dz"
        vth = get_variable(run_info, "electron_thermal_speed"; kwargs...)
        variable = get_electron_z_derivative(vth)
    elseif variable_name == "electron_dT_dz"
        T = get_variable(run_info, "electron_temperature"; kwargs...)
        variable = get_electron_z_derivative(T)
    elseif variable_name == "electron_dqpar_dz"
        qpar = get_variable(run_info, "electron_parallel_heat_flux"; kwargs...)
        variable = get_electron_z_derivative(qpar)
    elseif variable_name == "neutral_ddens_dz"
        variable = get_z_derivative(run_info, "density_neutral"; kwargs...)
    elseif variable_name == "neutral_ddens_dz_upwind"
        n = get_variable(run_info, "density_neutral"; kwargs...)
        uz = get_variable(run_info, "uz_neutral"; kwargs...)
        variable = get_upwind_z_derivative(n, uz)
    elseif variable_name == "neutral_duz_dz"
        variable = get_z_derivative(run_info, "uz_neutral"; kwargs...)
    elseif variable_name == "neutral_duz_dz_upwind"
        uz = get_variable(run_info, "uz_neutral"; kwargs...)
        variable = get_upwind_z_derivative(uz, uz)
    elseif variable_name == "neutral_dp_dz"
        variable = get_z_derivative(run_info, "p_neutral"; kwargs...)
    elseif variable_name == "neutral_dp_dz_upwind"
        p = get_variable(run_info, "p_neutral"; kwargs...)
        uz = get_variable(run_info, "uz_neutral"; kwargs...)
        variable = get_upwind_z_derivative(p, uz)
    elseif variable_name == "neutral_dpz_dz"
        variable = get_z_derivative(run_info, "pz_neutral"; kwargs...)
    elseif variable_name == "neutral_dvth_dz"
        variable = get_z_derivative(run_info, "thermal_speed_neutral"; kwargs...)
    elseif variable_name == "neutral_dT_dz"
        variable = get_z_derivative(run_info, "temperature_neutral"; kwargs...)
    elseif variable_name == "neutral_dqz_dz"
        variable = get_z_derivative(run_info, "qz_neutral"; kwargs...)
    elseif variable_name == "ddens_dt"
        all_moments = _get_all_moment_variables(run_info; kwargs...)
        variable = similar(all_moments.density)
        # Define function here to minimise effect type instability due to
        # get_all_moment_variables returning NamedTuples
        function get_ddens_dt!(variable, all_moments)
            nt = size(variable, ndims(variable))
            dummy = similar(variable, size(variable)[1:end-1])
            for tind ∈ 1:nt
                moments, fields, fvec =
                    _get_fake_moments_fields_scratch(all_moments, tind;
                                                     ion_extra=(:ddens_dt=>variable,))
                # Dummy first argument, because we actually want the 'side effect' of
                # filling the time derivative array in `moments`.
                continuity_equation!(dummy, fvec, moments, run_info.composition, 0.0,
                                     run_info.z_spectral,
                                     run_info.collisions.reactions.ionization_frequency,
                                     run_info.external_source_settings.ion,
                                     run_info.num_diss_params)
            end
        end
        get_ddens_dt!(variable, all_moments)
    elseif variable_name == "dnupar_dt"
        all_moments = _get_all_moment_variables(run_info; kwargs...)
        variable = similar(all_moments.parallel_flow)
        # Define function here to minimise effect type instability due to
        # get_all_moment_variables returning NamedTuples
        function get_dnupar_dt!(variable, all_moments)
            nt = size(variable, ndims(variable))
            dummy = similar(variable, size(variable)[1:end-1])
            for tind ∈ 1:nt
                moments, fields, fvec =
                    _get_fake_moments_fields_scratch(all_moments, tind;
                                                     ion_extra=(:dnupar_dt=>variable,))
                # Dummy first argument, because we actually want the 'side effect' of
                # filling the time derivative array in `moments`.
                force_balance!(dummy, fvec.density, fvec, moments, fields,
                               run_info.collisions, 0.0, run_info.z_spectral,
                               run_info.composition, run_info.geometry,
                               run_info.external_source_settings.ion,
                               run_info.num_diss_params)
            end
        end
        get_dnupar_dt!(variable, all_moments)
    elseif variable_name == "dupar_dt"
        dn_dt = get_variable(run_info, "ddens_dt"; kwargs...)
        dnupar_dt = get_variable(run_info, "dnupar_dt"; kwargs...)
        n = get_variable(run_info, "density"; kwargs...)
        upar = get_variable(run_info, "parallel_flow"; kwargs...)
        variable = @. dnupar_dt / n - upar / n * dn_dt
    elseif variable_name == "dp_dt"
        all_moments = _get_all_moment_variables(run_info; kwargs...)
        variable = similar(all_moments.pressure)
        # Define function here to minimise effect type instability due to
        # get_all_moment_variables returning NamedTuples
        function get_dp_dt!(variable, all_moments)
            nt = size(variable, ndims(variable))
            dummy = similar(variable, size(variable)[1:end-1])
            for tind ∈ 1:nt
                moments, fields, fvec =
                    _get_fake_moments_fields_scratch(all_moments, tind;
                                                     ion_extra=(:dp_dt=>variable,))
                # Dummy first argument, because we actually want the 'side effect' of
                # filling the time derivative array in `moments`.
                energy_equation!(dummy, fvec, moments, run_info.collisions, 0.0,
                                 run_info.z_spectral, run_info.composition,
                                 run_info.external_source_settings.ion,
                                 run_info.num_diss_params)
            end
        end
        get_dp_dt!(variable, all_moments)
    elseif variable_name == "dvth_dt"
        dn_dt = get_variable(run_info, "ddens_dt"; kwargs...)
        dp_dt = get_variable(run_info, "dp_dt"; kwargs...)
        n = get_variable(run_info, "density"; kwargs...)
        p = get_variable(run_info, "pressure"; kwargs...)
        vth = get_variable(run_info, "thermal_speed"; kwargs...)
        variable = @. 0.5 * vth * (dp_dt / p - dn_dt / n)
    elseif variable_name == "electron_dp_dt"
        all_moments = _get_all_moment_variables(run_info; kwargs...)
        variable = similar(all_moments.electron_pressure)
        # Define function here to minimise effect type instability due to
        # get_all_moment_variables returning NamedTuples
        function get_electron_dp_dt!(variable, all_moments)
            nt = size(variable, ndims(variable))
            dummy = similar(variable, size(variable)[1:end-1])
            for tind ∈ 1:nt
                moments, fields, fvec =
                    _get_fake_moments_fields_scratch(all_moments, tind;
                                                     electron_extra=(:dp_dt=>variable,))
                # Dummy first argument, because we actually want the 'side effect' of
                # filling the time derivative array in `moments`.
                electron_energy_equation!(dummy, moments.electron.dens,
                                          moments.electron.p, moments.electron.dens,
                                          moments.electron.upar, moments.electron.ppar,
                                          moments.ion.dens, moments.ion.upar,
                                          moments.ion.p, moments.neutral.dens,
                                          moments.neutral.uz, moments.neutral.p,
                                          moments.neutral.p, moments.electron,
                                          run_info.collisions, 0.0, run_info.composition,
                                          run_info.external_source_settings.electron,
                                          run_info.num_diss_params, run_info.r,
                                          run_info.z)
            end
        end
        get_electron_dp_dt!(variable, all_moments)
    elseif variable_name == "electron_dvth_dt"
        # Note that this block neglects any contribution of dn/dt to dvth/dt because the
        # operator splitting between implicit/explicit operators in the code means that
        # when dvth/dt is calculated for electrons, the (ion) density does not change in
        # the same step, so does not contribute to the dvth/dt used internally. This
        # post-processing function replicates that behaviour, guessing that that will
        # usually be what we want, e.g. if we want recalculate the coefficients used in
        # the electron kinetic equation.
        dp_dt = get_variable(run_info, "electron_dp_dt"; kwargs...)
        n = get_variable(run_info, "electron_density"; kwargs...)
        p = get_variable(run_info, "electron_pressure"; kwargs...)
        vth = get_variable(run_info, "electron_thermal_speed"; kwargs...)
        variable = @. 0.5 * vth * dp_dt / p
    elseif variable_name == "neutral_ddens_dt"
        all_moments = _get_all_moment_variables(run_info; kwargs...)
        variable = similar(all_moments.density)
        # Define function here to minimise effect type instability due to
        # get_all_moment_variables returning NamedTuples
        function get_neutral_ddens_dt!(variable, all_moments)
            nt = size(variable, ndims(variable))
            dummy = similar(variable, size(variable)[1:end-1])
            for tind ∈ 1:nt
                moments, fields, fvec =
                    _get_fake_moments_fields_scratch(all_moments, tind;
                                                     neutral_extra=(:ddens_dt=>variable,))
                # Dummy first argument, because we actually want the 'side effect' of
                # filling the time derivative array in `moments`.
                neutral_continuity_equation!(dummy, fvec, moments, run_info.composition,
                                             0.0, run_info.z_spectral,
                                             run_info.collisions.reactions.ionization_frequency,
                                             run_info.external_source_settings.neutral,
                                             run_info.num_diss_params)
            end
        end
        get_neutral_ddens_dt!(variable, all_moments)
    elseif variable_name == "neutral_dnuz_dt"
        all_moments = _get_all_moment_variables(run_info; kwargs...)
        variable = similar(all_moments.uz_neutral)
        # Define function here to minimise effect type instability due to
        # get_all_moment_variables returning NamedTuples
        function get_neutral_dnuz_dt!(variable, all_moments)
            nt = size(variable, ndims(variable))
            dummy = similar(variable, size(variable)[1:end-1])
            for tind ∈ 1:nt
                moments, fields, fvec =
                    _get_fake_moments_fields_scratch(all_moments, tind;
                                                     neutral_extra=(:dnuz_dt=>variable,))
                # Dummy first argument, because we actually want the 'side effect' of
                # filling the time derivative array in `moments`.
                neutral_force_balance!(dummy, fvec.density_neutral, fvec, moments, fields,
                                       run_info.collisions, 0.0, run_info.z_spectral,
                                       run_info.composition, run_info.geometry,
                                       run_info.external_source_settings.neutral,
                                       run_info.num_diss_params)
            end
        end
        get_neutral_dnuz_dt!(variable, all_moments)
    elseif variable_name == "neutral_duz_dt"
        dn_dt = get_variable(run_info, "neutral_ddens_dt"; kwargs...)
        dnuz_dt = get_variable(run_info, "neutral_dnuz_dt"; kwargs...)
        n = get_variable(run_info, "density_neutral"; kwargs...)
        uz = get_variable(run_info, "uz_neutral"; kwargs...)
        variable = @. dnuz_dt / n - uz / n * dn_dt
    elseif variable_name == "neutral_dp_dt"
        all_moments = _get_all_moment_variables(run_info; kwargs...)
        variable = similar(all_moments.p_neutral)
        # Define function here to minimise effect type instability due to
        # get_all_moment_variables returning NamedTuples
        function get_neutral_duz_dt!(variable, all_moments)
            nt = size(variable, ndims(variable))
            dummy = similar(variable, size(variable)[1:end-1])
            for tind ∈ 1:nt
                moments, fields, fvec =
                    _get_fake_moments_fields_scratch(all_moments, tind;
                                                     neutral_extra=(:dp_dt=>variable,))
                # Dummy first argument, because we actually want the 'side effect' of
                # filling the time derivative array in `moments`.
                neutral_energy_equation!(dummy, fvec, moments, run_info.collisions, 0.0,
                                         run_info.z_spectral, run_info.composition,
                                         run_info.external_source_settings.neutral,
                                         run_info.num_diss_params)
            end
        end
        get_neutral_duz_dt!(variable, all_moments)
    elseif variable_name == "neutral_dvth_dt"
        dn_dt = get_variable(run_info, "neutral_ddens_dt"; kwargs...)
        dp_dt = get_variable(run_info, "neutral_dp_dt"; kwargs...)
        n = get_variable(run_info, "density_neutral"; kwargs...)
        p = get_variable(run_info, "p_neutral"; kwargs...)
        vth = get_variable(run_info, "thermal_speed_neutral"; kwargs...)
        variable = @. 0.5 * vth * (dp_dt / p - dn_dt / n)
    elseif variable_name == "mfp"
        vth = get_variable(run_info, "thermal_speed"; kwargs...)
        nu_ii = get_variable(run_info, "collision_frequency_ii"; kwargs...)
        variable = vth ./ nu_ii
    elseif variable_name == "L_T"
        dT_dz = get_variable(run_info, "dT_dz"; kwargs...)
        temp = get_variable(run_info, "temperature"; kwargs...)
        # We define gradient lengthscale of T as LT^-1 = dln(T)/dz (ignore negative sign
        # tokamak convention as we're only concerned with comparing magnitudes)
        variable = abs.(temp .* dT_dz.^(-1))
        # flat points in temperature have diverging LT, so ignore those with NaN
        # using a hard coded 10.0 tolerance for now
        variable[variable .> 10.0] .= NaN
    elseif variable_name == "L_n"
        ddens_dz = get_variable(run_info, "ddens_dz"; kwargs...)
        n = get_variable(run_info, "density"; kwargs...)
        # We define gradient lengthscale of n as Ln^-1 = dln(n)/dz (ignore negative sign
        # tokamak convention as we're only concerned with comparing magnitudes)
        variable = abs.(n .* ddens_dz.^(-1))
        # flat points in temperature have diverging Ln, so ignore those with NaN
        # using a hard coded 10.0 tolerance for now
        variable[variable .> 10.0] .= NaN
    elseif variable_name == "L_upar"
        dupar_dz = get_variable(run_info, "dupar_dz"; kwargs...)
        upar = get_variable(run_info, "parallel_flow"; kwargs...)
        # We define gradient lengthscale of upar as Lupar^-1 = dln(upar)/dz (ignore negative sign
        # tokamak convention as we're only concerned with comparing magnitudes)
        variable = abs.(upar .* dupar_dz.^(-1))
        # flat points in temperature have diverging Lupar, so ignore those with NaN
        # using a hard coded 10.0 tolerance for now
        variable[variable .> 10.0] .= NaN
    elseif variable_name == "coll_krook_heat_flux"
        n = get_variable(run_info, "density"; kwargs...)
        vth = get_variable(run_info, "thermal_speed"; kwargs...)
        dT_dz = get_variable(run_info, "dT_dz"; kwargs...)
        if run_info.vperp.n == 1
            Krook_vth = sqrt(3.0) * vth
            adjust_1V = 1.0 / sqrt(3.0)
        else
            Krook_vth = vth
            adjust_1V = 1.0
        end
        Krook_nu_ii = get_collision_frequency_ii(run_info.collisions, n, Krook_vth)
        variable = @. -(1/2) * 3/2 * n * Krook_vth^2 * 3 * dT_dz / Krook_nu_ii
    elseif variable_name == "collision_frequency_ii"
        n = get_variable(run_info, "density"; kwargs...)
        vth = get_variable(run_info, "thermal_speed"; kwargs...)
        variable = get_collision_frequency_ii(run_info.collisions, n, vth)
    elseif variable_name == "collision_frequency_ee"
        n = get_variable(run_info, "electron_density"; kwargs...)
        vth = get_variable(run_info, "electron_thermal_speed"; kwargs...)
        variable = get_collision_frequency_ee(run_info.collisions, n, vth)
    elseif variable_name == "collision_frequency_ei"
        n = get_variable(run_info, "electron_density"; kwargs...)
        vth = get_variable(run_info, "electron_thermal_speed"; kwargs...)
        variable = get_collision_frequency_ei(run_info.collisions, n, vth)
    elseif variable_name == "electron_temperature"
        vth = get_variable(run_info, "electron_thermal_speed"; kwargs...)
        variable = 0.5 * run_info.composition.me_over_mi .* vth.^2
    elseif variable_name == "temperature_neutral"
        vth = get_variable(run_info, "thermal_speed_neutral"; kwargs...)
        variable = 0.5 * vth.^2
    elseif variable_name == "sound_speed"
        T_e = run_info.composition.T_e
        T_i = get_variable(run_info, "temperature"; kwargs...)

        # Adiabatic index. Not too clear what value should be (see e.g. [Riemann 1991,
        # below eq. (39)], or discussion of Bohm criterion in Stangeby's book.
        gamma = 1.0 # 3.0

        variable = @. sqrt((T_e + gamma*T_i))
    elseif variable_name == "mach_number"
        upar = get_variable(run_info, "parallel_flow"; kwargs...)
        cs = get_variable(run_info, "sound_speed"; kwargs...)
        variable = upar ./ cs
    elseif variable_name == "total_energy"
        p = get_variable(run_info, "pressure"; kwargs...)
        upar = get_variable(run_info, "parallel_flow"; kwargs...)
        n = get_variable(run_info, "density"; kwargs...)

        variable = @. 1.5 * p + 0.5*n*upar^2
    elseif variable_name == "total_energy_neutral"
        p = get_variable(run_info, "p_neutral"; kwargs...)
        upar = get_variable(run_info, "uz_neutral"; kwargs...)
        n = get_variable(run_info, "density_neutral"; kwargs...)

        # Factor of 3/2 in front of 1/2*n*vth^2*upar because this in 1V - would be 5/2
        # for 2V/3V cases.
        variable = @. 1.5 * p + 0.5*n*upar^2
    elseif variable_name == "total_energy_flux"
        if run_info.vperp.n > 1
            qpar = get_variable(run_info, "parallel_heat_flux"; kwargs...)
            vth = get_variable(run_info, "thermal_speed"; kwargs...)
            upar = get_variable(run_info, "parallel_flow"; kwargs...)
            n = get_variable(run_info, "density"; kwargs...)

            variable = @. qpar + 1.25*n*vth^2*upar + 0.5*n*upar^3
        else
            qpar = get_variable(run_info, "parallel_heat_flux"; kwargs...)
            vth = get_variable(run_info, "thermal_speed"; kwargs...)
            upar = get_variable(run_info, "parallel_flow"; kwargs...)
            n = get_variable(run_info, "density"; kwargs...)

            # Factor of 3/2 in front of 1/2*n*vth^2*upar because this in 1V - would be 5/2
            # for 2V/3V cases.
            variable = @. qpar + 0.75*n*vth^2*upar + 0.5*n*upar^3
        end
    elseif variable_name == "total_energy_flux_neutral"
        if run_info.vzeta.n > 1 || run_info.vr.n > 1
            qpar = get_variable(run_info, "qz_neutral"; kwargs...)
            vth = get_variable(run_info, "thermal_speed_neutral"; kwargs...)
            upar = get_variable(run_info, "uz_neutral"; kwargs...)
            n = get_variable(run_info, "density_neutral"; kwargs...)

            variable = @. qpar + 1.25*n*vth^2*upar + 0.5*n*upar^3
        else
            qpar = get_variable(run_info, "qz_neutral"; kwargs...)
            vth = get_variable(run_info, "thermal_speed_neutral"; kwargs...)
            upar = get_variable(run_info, "uz_neutral"; kwargs...)
            n = get_variable(run_info, "density_neutral"; kwargs...)

            # Factor of 3/2 in front of 1/2*n*vth^2*upar because this in 1V - would be 5/2
            # for 2V/3V cases.
            variable = @. qpar + 0.75*n*vth^2*upar + 0.5*n*upar^3
        end
    elseif variable_name == "z_advect_speed"
        # update_speed_z!() requires all dimensions to be present, so do *not* pass kwargs
        # to get_variable() in this case. Instead select a slice of the result.
        upar = get_variable(run_info, "parallel_flow")
        vth = get_variable(run_info, "thermal_speed")
        nz, nr, nspecies, nt = size(upar)
        nvperp = run_info.vperp.n
        nvpa = run_info.vpa.n

        speed = allocate_float(nz, nvpa, nvperp, nr, nspecies, nt)
        Er = get_variable(run_info, "Er")
        gEr = allocate_float(nvperp, nz, nr, nspecies, nt)
        for it ∈ 1:nt, is ∈ 1:nspecies, ir ∈ 1:nr, iz ∈ 1:nz
            # Don't support gyroaveraging here (yet)
            gEr[:,iz,ir,is,it] .= Er[iz,ir,it]
        end

        setup_distributed_memory_MPI(1,1,1,1)
        setup_loop_ranges!(0, 1; s=nspecies, sn=run_info.n_neutral_species, r=nr, z=nz,
                           vperp=nvperp, vpa=nvpa, vzeta=run_info.vzeta.n,
                           vr=run_info.vr.n, vz=run_info.vz.n)
        for it ∈ 1:nt, is ∈ 1:nspecies
            @begin_serial_region()
            # Only need some struct with a 'speed' variable
            advect = (speed=@view(speed[:,:,:,:,is,it]),)
            # Only need Er
            fields = (gEr=@view(gEr[:,:,:,is,it]),)
            @views update_speed_z!(advect, upar[:,:,is,it], vth[:,:,is,it],
                                   run_info.evolve_upar, run_info.evolve_p, fields,
                                   run_info.vpa, run_info.vperp, run_info.z, run_info.r,
                                   run_info.time[it], run_info.geometry, is)
        end

        # Horrible hack so that we can get the speed back without rearranging the
        # dimensions, if we want that to pass it to a utility function from the main code
        # (e.g. to calculate a CFL limit).
        if normalize_advection_speed_shape
            variable = allocate_float(nvpa, nvperp, nz, nr, nspecies, nt)
            for it ∈ 1:nt, is ∈ 1:nspecies, ir ∈ 1:nr, iz ∈ 1:nz, ivperp ∈ 1:nvperp, ivpa ∈ 1:nvpa
                variable[ivpa,ivperp,iz,ir,is,it] = speed[iz,ivpa,ivperp,ir,is,it]
            end
            variable = select_slice_of_variable(variable; kwargs...)
        else
            variable = speed
            if :it ∈ keys(kwargs)
                variable = selectdim(variable, 6, kwargs[:it])
            end
            if :is ∈ keys(kwargs)
                variable = selectdim(variable, 5, kwargs[:is])
            end
            if :ir ∈ keys(kwargs)
                variable = selectdim(variable, 4, kwargs[:ir])
            end
            if :ivperp ∈ keys(kwargs)
                variable = selectdim(variable, 3, kwargs[:ivperp])
            end
            if :ivpa ∈ keys(kwargs)
                variable = selectdim(variable, 2, kwargs[:ivpa])
            end
            if :iz ∈ keys(kwargs)
                variable = selectdim(variable, 1, kwargs[:iz])
            end
        end
    elseif variable_name == "vpa_advect_speed"
        density = get_variable(run_info, "density"; kwargs...)
        upar = get_variable(run_info, "parallel_flow"; kwargs...)
        p = get_variable(run_info, "pressure"; kwargs...)
        vth = get_variable(run_info, "thermal_speed"; kwargs...)
        dupar_dz = similar(upar)
        dupar_dz = get_z_derivative_of_loaded_variable(upar)
        dp_dz = similar(p)
        dp_dz = get_z_derivative_of_loaded_variable(p)
        dvth_dz = similar(vth)
        dvth_dz = get_z_derivative_of_loaded_variable(vth)
        dqpar_dz = get_z_derivative(run_info, "parallel_heat_flux"; kwargs...)
        dupar_dt = get_variable(run_info, "dupar_dt"; kwargs...)
        dvth_dt = get_variable(run_info, "dvth_dt"; kwargs...)
        if any(x -> x.active, run_info.external_source_settings.ion)
            n_sources = length(run_info.external_source_settings.ion)
            external_source_amplitude = get_variable(run_info, "external_source_amplitude")
            if run_info.evolve_density
                external_source_density_amplitude = get_variable(run_info, "external_source_density_amplitude")
            else
                external_source_density_amplitude = zeros(0,0,n_sources,run_info.nt)
            end
            if run_info.evolve_upar
                external_source_momentum_amplitude = get_variable(run_info, "external_source_momentum_amplitude")
            else
                external_source_momentum_amplitude = zeros(0,0,n_sources,run_info.nt)
            end
            if run_info.evolve_p
                external_source_pressure_amplitude = get_variable(run_info, "external_source_pressure_amplitude")
            else
                external_source_pressure_amplitude = zeros(0,0,n_sources,run_info.nt)
            end
        else
            n_sources = 0
            external_source_amplitude = zeros(0,0,n_sources,run_info.nt)
            external_source_density_amplitude = zeros(0,0,n_sources,run_info.nt)
            external_source_momentum_amplitude = zeros(0,0,n_sources,run_info.nt)
            external_source_pressure_amplitude = zeros(0,0,n_sources,run_info.nt)
        end

        nz, nr, nspecies, nt = size(vth)
        nvperp = run_info.vperp.n
        nvpa = run_info.vpa.n

        # update_speed_z!() requires all dimensions to be present, so do *not* pass kwargs
        # to get_variable() in this case. Instead select a slice of the result.
        Ez = get_variable(run_info, "Ez")
        gEz = allocate_float(nvperp, nz, nr, nspecies, nt)
        for it ∈ 1:nt, is ∈ 1:nspecies, ir ∈ 1:nr, iz ∈ 1:nz
            # Don't support gyroaveraging here (yet)
            gEz[:,iz,ir,is,it] .= Ez[iz,ir,it]
        end

        speed=allocate_float(nvpa, nvperp, nz, nr, nspecies, nt)
        setup_distributed_memory_MPI(1,1,1,1)
        setup_loop_ranges!(0, 1; s=nspecies, sn=run_info.n_neutral_species, r=nr, z=nz,
                           vperp=nvperp, vpa=nvpa, vzeta=run_info.vzeta.n,
                           vr=run_info.vr.n, vz=run_info.vz.n)
        
        # Use neutrals for fvec calculation in moment_kinetic version only when 
        # n_neutrals != 0
        if run_info.n_neutral_species != 0
            density_neutral = get_variable(run_info, "density_neutral")
            uz_neutral = get_variable(run_info, "uz_neutral")
            p_neutral = get_variable(run_info, "p_neutral")
        end

        for it ∈ 1:nt
            @begin_serial_region()
            # Only need some struct with a 'speed' variable
            advect = [(speed=@view(speed[:,:,:,:,is,it]),) for is ∈ 1:nspecies]
            # Only need Ez
            fields = (gEz=@view(gEz[:,:,:,:,it]),)
            @views moments = (ion=(dp_dz=dp_dz[:,:,:,it],
                                   dupar_dz=dupar_dz[:,:,:,it],
                                   dvth_dz=dvth_dz[:,:,:,it],
                                   dqpar_dz=dqpar_dz[:,:,:,it],
                                   vth=vth[:,:,:,it],
                                   dupar_dt=dupar_dt[:,:,:,it],
                                   dvth_dt=dvth_dt[:,:,:,it],
                                   external_source_amplitude=external_source_amplitude[:,:,:,it],
                                   external_source_density_amplitude=external_source_density_amplitude[:,:,:,it],
                                   external_source_momentum_amplitude=external_source_momentum_amplitude[:,:,:,it],
                                   external_source_pressure_amplitude=external_source_pressure_amplitude[:,:,:,it]),
                             evolve_density=run_info.evolve_density,
                             evolve_upar=run_info.evolve_upar,
                             evolve_p=run_info.evolve_p)
            if run_info.n_neutral_species != 0
                @views fvec = (density=density[:,:,:,it],
                               upar=upar[:,:,:,it],
                               p=p[:,:,:,it],
                               density_neutral=density_neutral[:,:,:,it],
                               uz_neutral=uz_neutral[:,:,:,it],
                               p_neutral=p_neutral[:,:,:,it])
            else
                @views fvec = (density=density[:,:,:,it],
                               upar=upar[:,:,:,it],
                               p=p[:,:,:,it])
            end
            @views update_speed_vpa!(advect, fields, fvec, moments, run_info.vpa,
                                     run_info.vperp, run_info.z, run_info.r,
                                     run_info.composition, run_info.collisions,
                                     run_info.external_source_settings.ion,
                                     run_info.time[it], run_info.geometry)
        end

        variable = speed
        variable = select_slice_of_variable(variable; kwargs...)
     elseif variable_name == "electron_z_advect_speed"
        # update_speed_z!() requires all dimensions to be present, so do *not* pass kwargs
        # to get_variable() in this case. Instead select a slice of the result.
        upar = get_variable(run_info, "electron_parallel_flow")
        vth = get_variable(run_info, "electron_thermal_speed")
        nz, nr, nt = size(upar)
        nvperp = run_info.vperp.n
        nvpa = run_info.vpa.n

        speed = allocate_float(nz, nvpa, nvperp, nr, nt)

        setup_distributed_memory_MPI(1,1,1,1)
        setup_loop_ranges!(0, 1; s=run_info.n_ion_species, sn=run_info.n_neutral_species,
                           r=nr, z=nz,
                           vperp=(run_info.vperp === nothing ? 1 : run_info.vperp.n),
                           vpa=(run_info.vpa === nothing ? 1 : run_info.vpa.n),
                           vzeta=(run_info.vzeta === nothing ? 1 : run_info.vzeta.n),
                           vr=(run_info.vr === nothing ? 1 : run_info.vr.n),
                           vz=(run_info.vz === nothing ? 1 : run_info.vz.n))
        for it ∈ 1:nt
            @begin_serial_region()
            # Only need some struct with a 'speed' variable
            advect = (speed=@view(speed[:,:,:,:,it]),)
            for ir ∈ 1:run_info.r.n
                @views update_electron_speed_z!(advect, upar[:,ir,it], vth[:,ir,it],
                                                run_info.vpa.grid, ir)
            end
        end

        # Horrible hack so that we can get the speed back without rearranging the
        # dimensions, if we want that to pass it to a utility function from the main code
        # (e.g. to calculate a CFL limit).
        if normalize_advection_speed_shape
            variable = allocate_float(nvpa, nvperp, nz, nr, nspecies, nt)
            for it ∈ 1:nt, ir ∈ 1:nr, iz ∈ 1:nz, ivperp ∈ 1:nvperp, ivpa ∈ 1:nvpa
                variable[ivpa,ivperp,iz,ir,it] = speed[iz,ivpa,ivperp,ir,it]
            end
            variable = select_slice_of_variable(variable; kwargs...)
        else
            variable = speed
            if :it ∈ keys(kwargs)
                variable = selectdim(variable, 5, kwargs[:it])
            end
            if :ir ∈ keys(kwargs)
                variable = selectdim(variable, 4, kwargs[:ir])
            end
            if :ivperp ∈ keys(kwargs)
                variable = selectdim(variable, 3, kwargs[:ivperp])
            end
            if :ivpa ∈ keys(kwargs)
                variable = selectdim(variable, 2, kwargs[:ivpa])
            end
            if :iz ∈ keys(kwargs)
                variable = selectdim(variable, 1, kwargs[:iz])
            end
        end
    elseif variable_name == "electron_vpa_advect_speed"
        # update_speed_z!() requires all dimensions to be present, so do *not* pass kwargs
        # to get_variable() in this case. Instead select a slice of the result.
        density = get_variable(run_info, "electron_density"; kwargs...)
        upar = get_variable(run_info, "electron_parallel_flow"; kwargs...)
        p = get_variable(run_info, "electron_pressure"; kwargs...)
        ppar = get_variable(run_info, "electron_parallel_pressure"; kwargs...)
        vth = get_variable(run_info, "electron_thermal_speed"; kwargs...)
        dp_dz = similar(p)
        dp_dz = get_z_derivative_of_loaded_variable(p)
        dppar_dz = similar(ppar)
        dppar_dz = get_z_derivative_of_loaded_variable(ppar)
        dvth_dz = similar(vth)
        dvth_dz = get_z_derivative_of_loaded_variable(vth)
        dqpar_dz = get_z_derivative(run_info, "electron_parallel_heat_flux"; kwargs...)
        if any(x -> x.active, run_info.external_source_settings.electron)
            n_sources = length(run_info.external_source_settings.electron)
            external_source_amplitude = get_variable(run_info, "external_source_electron_amplitude"; kwargs...)
            external_source_density_amplitude = get_variable(run_info, "external_source_electron_density_amplitude"; kwargs...)
            external_source_momentum_amplitude = get_variable(run_info, "external_source_electron_momentum_amplitude"; kwargs...)
            external_source_pressure_amplitude = get_variable(run_info, "external_source_electron_pressure_amplitude"; kwargs...)
        else
            n_sources = 0
            external_source_amplitude = zeros(0,0,n_sources,run_info.nt)
            external_source_density_amplitude = zeros(0,0,n_sources,run_info.nt)
            external_source_momentum_amplitude = zeros(0,0,n_sources,run_info.nt)
            external_source_pressure_amplitude = zeros(0,0,n_sources,run_info.nt)
        end

        nz, nr, nt = size(vth)
        nvperp = run_info.vperp.n
        nvpa = run_info.vpa.n

        speed=allocate_float(nvpa, nvperp, nz, nr, nt)
        setup_distributed_memory_MPI(1,1,1,1)
        setup_loop_ranges!(0, 1; s=run_info.n_ion_species, sn=run_info.n_neutral_species,
                           r=nr, z=nz,
                           vperp=(run_info.vperp === nothing ? 1 : run_info.vperp.n),
                           vpa=(run_info.vpa === nothing ? 1 : run_info.vpa.n),
                           vzeta=(run_info.vzeta === nothing ? 1 : run_info.vzeta.n),
                           vr=(run_info.vr === nothing ? 1 : run_info.vr.n),
                           vz=(run_info.vz === nothing ? 1 : run_info.vz.n))
        for it ∈ 1:nt
            @begin_serial_region()
            # Only need some struct with a 'speed' variable
            advect = (speed=@view(speed[:,:,:,:,it]),)
            moments = (electron=(ppar=ppar[:,:,it],
                                 vth=vth[:,:,it],
                                 dp_dz=dp_dz[:,:,it],
                                 dppar_dz=dppar_dz[:,:,it],
                                 dqpar_dz=dqpar_dz[:,:,it],
                                 dvth_dz=dvth_dz[:,:,it],
                                 external_source_amplitude=external_source_amplitude[:,:,:,it],
                                 external_source_density_amplitude=external_source_density_amplitude[:,:,:,it],
                                 external_source_momentum_amplitude=external_source_momentum_amplitude[:,:,:,it],
                                 external_source_pressure_amplitude=external_source_pressure_amplitude[:,:,:,it]),)
            @views update_electron_speed_vpa!(advect, density[:,:,it], upar[:,:,it],
                                              p[:,:,it], moments,
                                              run_info.composition.me_over_mi,
                                              run_info.vpa.grid,
                                              run_info.external_source_settings.electron)
        end

        variable = speed
        variable = select_slice_of_variable(variable; kwargs...)
    elseif variable_name == "neutral_z_advect_speed"
        # update_speed_z!() requires all dimensions to be present, so do *not* pass kwargs
        # to get_variable() in this case. Instead select a slice of the result.
        uz = get_variable(run_info, "parallel_flow")
        vth = get_variable(run_info, "thermal_speed_neutral")
        nz, nr, nspecies, nt = size(uz)
        nvzeta = run_info.vzeta.n
        nvr = run_info.vr.n
        nvz = run_info.vz.n

        speed = allocate_float(nz, nvz, nvr, nvzeta, nr, nspecies, nt)

        setup_distributed_memory_MPI(1,1,1,1)
        setup_loop_ranges!(0, 1; s=nspecies, sn=run_info.n_neutral_species, r=nr, z=nz,
                           vperp=run_info.vperp.n, vpa=run_info.vpa.n, vzeta=nvzeta,
                           vr=nvr, vz=nvz)
        for it ∈ 1:nt, isn ∈ 1:nspecies
            @begin_serial_region()
            # Only need some struct with a 'speed' variable
            advect = (speed=@view(speed[:,:,:,:,:,isn,it]),)
            @views update_speed_neutral_z!(advect, uz[:,:,:,it], vth[:,:,:,it],
                                           run_info.evolve_upar, run_info.evolve_p,
                                           run_info.vz, run_info.vr, run_info.vzeta,
                                           run_info.z, run_info.r, run_info.time[it])
        end

        # Horrible hack so that we can get the speed back without rearranging the
        # dimensions, if we want that to pass it to a utility function from the main code
        # (e.g. to calculate a CFL limit).
        if normalize_advection_speed_shape
            variable = allocate_float(nvz, nvr, nvzeta, nz, nr, nspecies, nt)
            for it ∈ 1:nt, isn ∈ 1:nspecies, ir ∈ 1:nr, iz ∈ 1:nz, ivzeta ∈ 1:nvzeta, ivr ∈ 1:nvr, ivz ∈ 1:nvz
                variable[ivz,ivr,ivzeta,iz,ir,isn,it] = speed[iz,ivz,ivr,ivzeta,ir,isn,it]
            end
            variable = select_slice_of_variable(variable; kwargs...)
        else
            variable = speed
            if :it ∈ keys(kwargs)
                variable = selectdim(variable, 7, kwargs[:it])
            end
            if :is ∈ keys(kwargs)
                variable = selectdim(variable, 6, kwargs[:is])
            end
            if :ir ∈ keys(kwargs)
                variable = selectdim(variable, 5, kwargs[:ir])
            end
            if :ivzeta ∈ keys(kwargs)
                variable = selectdim(variable, 4, kwargs[:ivzeta])
            end
            if :ivr ∈ keys(kwargs)
                variable = selectdim(variable, 3, kwargs[:ivr])
            end
            if :ivz ∈ keys(kwargs)
                variable = selectdim(variable, 2, kwargs[:ivz])
            end
            if :iz ∈ keys(kwargs)
                variable = selectdim(variable, 1, kwargs[:iz])
            end
        end
    elseif variable_name == "neutral_vz_advect_speed"
        # update_speed_z!() requires all dimensions to be present, so do *not* pass kwargs
        # to get_variable() in this case. Instead select a slice of the result.
        Ez = get_variable(run_info, "Ez"; kwargs...)
        density = get_variable(run_info, "density"; kwargs...)
        upar = get_variable(run_info, "parallel_flow"; kwargs...)
        p = get_variable(run_info, "pressure"; kwargs...)
        density_neutral = get_variable(run_info, "density_neutral"; kwargs...)
        uz_neutral = get_variable(run_info, "uz_neutral"; kwargs...)
        p_neutral = get_variable(run_info, "p_neutral"; kwargs...)
        vth = get_variable(run_info, "thermal_speed_neutral"; kwargs...)
        duz_dz = get_z_derivative_of_loaded_variable(uz_neutral)
        dp_dz = similar(p_neutral)
        dp_dz = get_z_derivative_of_loaded_variable(p_neutral)
        dvth_dz = similar(vth)
        dvth_dz = get_z_derivative_of_loaded_variable(vth)
        dqz_dz = get_z_derivative(run_info, "qz_neutral"; kwargs...)
        dp_dt = get_variable(run_info, "p_neutral")
        duz_dt = get_variable(run_info, "uz_neutral")
        dvth_dt = get_variable(run_info, "thermal_speed_neutral")
        if any(x -> x.active, run_info.external_source_settings.neutral)
            n_sources = length(run_info.external_source_settings.neutral)
            external_source_amplitude = get_variable(run_info, "external_source_neutral_amplitude"; kwargs...)
            if run_info.evolve_density
                external_source_density_amplitude = get_variable(run_info, "external_source_neutral_density_amplitude"; kwargs...)
            else
                external_source_density_amplitude = zeros(0,0,n_sources,run_info.nt)
            end
            if run_info.evolve_upar
                external_source_momentum_amplitude = get_variable(run_info, "external_source_neutral_momentum_amplitude"; kwargs...)
            else
                external_source_momentum_amplitude = zeros(0,0,n_sources,run_info.nt)
            end
            if run_info.evolve_p
                external_source_pressure_amplitude = get_variable(run_info, "external_source_neutral_pressure_amplitude"; kwargs...)
            else
                external_source_pressure_amplitude = zeros(0,0,n_sources,run_info.nt)
            end
        else
            n_sources = 0
            external_source_amplitude = zeros(0,0,n_sources,run_info.nt)
            external_source_density_amplitude = zeros(0,0,n_sources,run_info.nt)
            external_source_momentum_amplitude = zeros(0,0,n_sources,run_info.nt)
            external_source_pressure_amplitude = zeros(0,0,n_sources,run_info.nt)
        end

        nz, nr, nspecies, nt = size(vth)
        nvzeta = run_info.vzeta.n
        nvr = run_info.vr.n
        nvz = run_info.vz.n
        speed = allocate_float(nvz, nvr, nvzeta, nz, nr, nspecies, nt)

        setup_distributed_memory_MPI(1,1,1,1)
        setup_loop_ranges!(0, 1; s=nspecies, sn=run_info.n_neutral_species, r=nr, z=nz,
                           vperp=run_info.vperp.n, vpa=run_info.vpa.n, vzeta=nvzeta,
                           vr=nvr, vz=nvz)
        for it ∈ 1:nt
            @begin_serial_region()
            # Only need some struct with a 'speed' variable
            advect = [(speed=@view(speed[:,:,:,:,:,isn,it]),) for isn ∈ 1:nspecies]
            # Don't actually use `fields` at the moment
            fields = nothing
            @views fvec = (density=density[:,:,:,it],
                           upar=upar[:,:,:,it],
                           p=p[:,:,:,it],
                           density_neutral=density_neutral[:,:,:,it],
                           uz_neutral=uz_neutral[:,:,:,it],
                           p_neutral=p_neutral[:,:,:,it])
            @views moments = (neutral=(dp_dz=dp_dz[:,:,:,it],
                                       duz_dz=duz_dz[:,:,:,it],
                                       dvth_dz=dvth_dz[:,:,:,it],
                                       dqz_dz=dqz_dz[:,:,:,it],
                                       vth=vth[:,:,:,it],
                                       dp_dt=dp_dt[:,:,:,it],
                                       duz_dt=duz_dt[:,:,:,it],
                                       dvth_dt=dvth_dt[:,:,:,it],
                                       external_source_amplitude=external_source_amplitude[:,:,:,it],
                                       external_source_density_amplitude=external_source_density_amplitude[:,:,:,it],
                                       external_source_momentum_amplitude=external_source_momentum_amplitude[:,:,:,it],
                                       external_source_pressure_amplitude=external_source_pressure_amplitude[:,:,:,it]),
                             evolve_density=run_info.evolve_density,
                             evolve_upar=run_info.evolve_upar,
                             evolve_p=run_info.evolve_p)
            @views update_speed_neutral_vz!(advect, fields, fvec, moments,
                                            run_info.vz, run_info.vr, run_info.vzeta,
                                            run_info.z, run_info.r, run_info.composition,
                                            run_info.collisions,
                                            run_info.external_source_settings.neutral)
        end

        variable = speed
        variable = select_slice_of_variable(variable; kwargs...)
    elseif variable_name == "steps_per_output"
        variable = get_per_step_from_cumulative_variable(run_info, "step_counter"; kwargs...)
    elseif variable_name == "failures_per_output"
        variable = get_per_step_from_cumulative_variable(run_info, "failure_counter"; kwargs...)
    elseif variable_name == "failure_caused_by_per_output"
        variable = get_per_step_from_cumulative_variable(run_info, "failure_caused_by"; kwargs...)
    elseif variable_name == "limit_caused_by_per_output"
        variable = get_per_step_from_cumulative_variable(run_info, "limit_caused_by"; kwargs...)
    elseif variable_name == "average_successful_dt"
        steps_per_output = get_variable(run_info, "steps_per_output"; kwargs...)
        failures_per_output = get_variable(run_info, "failures_per_output"; kwargs...)
        successful_steps_per_output = steps_per_output - failures_per_output

        delta_t = copy(run_info.time)
        for i ∈ length(delta_t):-1:2
            delta_t[i] -= delta_t[i-1]
        end

        variable = delta_t ./ successful_steps_per_output
        for i ∈ eachindex(successful_steps_per_output)
            if successful_steps_per_output[i] == 0
                variable[i] = 0.0
            end
        end
        if successful_steps_per_output[1] == 0
            # Don't want a meaningless Inf...
            variable[1] = 0.0
        end
    elseif variable_name == "electron_steps_per_ion_step"
        electron_steps_per_output = get_variable(run_info, "electron_steps_per_output"; kwargs...)
        ion_steps_per_output = get_variable(run_info, "steps_per_output"; kwargs...)
        variable = electron_steps_per_output ./ ion_steps_per_output
    elseif variable_name == "electron_steps_per_output"
        variable = get_per_step_from_cumulative_variable(run_info, "electron_step_counter"; kwargs...)
    elseif variable_name == "electron_failures_per_output"
        variable = get_per_step_from_cumulative_variable(run_info, "electron_failure_counter"; kwargs...)
    elseif variable_name == "electron_failure_caused_by_per_output"
        variable = get_per_step_from_cumulative_variable(run_info, "electron_failure_caused_by"; kwargs...)
    elseif variable_name == "electron_limit_caused_by_per_output"
        variable = get_per_step_from_cumulative_variable(run_info, "electron_limit_caused_by"; kwargs...)
    elseif variable_name == "electron_average_successful_dt"
        electron_steps_per_output = get_variable(run_info, "electron_steps_per_output"; kwargs...)
        electron_failures_per_output = get_variable(run_info, "electron_failures_per_output"; kwargs...)
        electron_successful_steps_per_output = electron_steps_per_output - electron_failures_per_output
        electron_pseudotime = get_variable(run_info, "electron_cumulative_pseudotime"; kwargs...)

        delta_t = copy(electron_pseudotime)
        for i ∈ length(delta_t):-1:2
            delta_t[i] -= delta_t[i-1]
        end

        variable = delta_t ./ electron_successful_steps_per_output
        for i ∈ eachindex(electron_successful_steps_per_output)
            if electron_successful_steps_per_output[i] == 0
                variable[i] = 0.0
            end
        end
        if electron_successful_steps_per_output[1] == 0
            # Don't want a meaningless Inf...
            variable[1] = 0.0
        end
    elseif variable_name == "CFL_ion_z"
        # update_speed_z!() requires all dimensions to be present, so do *not* pass kwargs
        # to get_variable() in this case. Instead select a slice of the result.
        speed = get_variable(run_info, "z_advect_speed";
                             normalize_advection_speed_shape=false)
        nz, nvpa, nvperp, nr, nspecies, nt = size(speed)
        CFL = similar(speed)
        for it ∈ 1:nt
            @views get_CFL!(CFL[:,:,:,:,:,it], speed[:,:,:,:,:,it], run_info.z)
        end

        variable = allocate_float(nvpa, nvperp, nz, nr, nspecies, nt)
        for it ∈ 1:nt, is ∈ 1:nspecies, ir ∈ 1:nr, iz ∈ 1:nz, ivperp ∈ 1:nvperp, ivpa ∈ 1:nvpa
            variable[ivpa,ivperp,iz,ir,is,it] = CFL[iz,ivpa,ivperp,ir,is,it]
        end
        variable = select_slice_of_variable(variable; kwargs...)
    elseif variable_name == "CFL_ion_vpa"
        # update_speed_z!() requires all dimensions to be present, so do *not* pass kwargs
        # to get_variable() in this case. Instead select a slice of the result.
        speed = get_variable(run_info, "vpa_advect_speed")
        nt = size(speed, 6)
        CFL = similar(speed)
        for it ∈ 1:nt
            @views get_CFL!(CFL[:,:,:,:,:,it], speed[:,:,:,:,:,it], run_info.vpa)
        end

        variable = CFL
        variable = select_slice_of_variable(variable; kwargs...)
    elseif variable_name == "CFL_electron_z"
        # update_speed_z!() requires all dimensions to be present, so do *not* pass kwargs
        # to get_variable() in this case. Instead select a slice of the result.
        speed = get_variable(run_info, "electron_z_advect_speed";
                             normalize_advection_speed_shape=false)
        nz, nvpa, nvperp, nr, nt = size(speed)
        CFL = similar(speed)
        for it ∈ 1:nt
            @views get_CFL!(CFL[:,:,:,:,it], speed[:,:,:,:,it], run_info.z)
        end

        variable = allocate_float(nvpa, nvperp, nz, nr, nt)
        for it ∈ 1:nt, ir ∈ 1:nr, iz ∈ 1:nz, ivperp ∈ 1:nvperp, ivpa ∈ 1:nvpa
            variable[ivpa,ivperp,iz,ir,it] = CFL[iz,ivpa,ivperp,ir,it]
        end
        variable = select_slice_of_variable(variable; kwargs...)
    elseif variable_name == "CFL_electron_vpa"
        # update_speed_z!() requires all dimensions to be present, so do *not* pass kwargs
        # to get_variable() in this case. Instead select a slice of the result.
        speed = get_variable(run_info, "electron_vpa_advect_speed")
        nt = size(speed, 5)
        CFL = similar(speed)
        for it ∈ 1:nt
            @views get_CFL!(CFL[:,:,:,:,it], speed[:,:,:,:,it], run_info.vpa)
        end

        variable = CFL
        variable = select_slice_of_variable(variable; kwargs...)
    elseif variable_name == "CFL_neutral_z"
        # update_speed_z!() requires all dimensions to be present, so do *not* pass kwargs
        # to get_variable() in this case. Instead select a slice of the result.
        speed = get_variable(run_info, "neutral_z_advect_speed";
                             normalize_advection_speed_shape=false)
        nz, nvz, nvr, nvzeta, nr, nspecies, nt = size(speed)
        CFL = similar(speed)
        for it ∈ 1:nt
            @views get_CFL!(CFL[:,:,:,:,:,:,it], speed[:,:,:,:,:,:,it], run_info.z)
        end

        variable = allocate_float(nvz, nvr, nvzeta, nz, nr, nspecies, nt)
        for it ∈ 1:nt, is ∈ 1:nspecies, ir ∈ 1:nr, iz ∈ 1:nz, ivzeta ∈ 1:nvzeta, ivr ∈ 1:nvr, ivz ∈ 1:nvz
            variable[ivz,ivr,ivzeta,iz,ir,is,it] = CFL[iz,ivz,ivr,ivzeta,ir,is,it]
        end
        variable = select_slice_of_variable(variable; kwargs...)
    elseif variable_name == "CFL_neutral_vz"
        # update_speed_z!() requires all dimensions to be present, so do *not* pass kwargs
        # to get_variable() in this case. Instead select a slice of the result.
        speed = get_variable(run_info, "neutral_vz_advect_speed")
        nt = size(speed, 7)
        CFL = similar(speed)
        for it ∈ 1:nt
            @views get_CFL!(CFL[:,:,:,:,:,:,it], speed[:,:,:,:,:,:,it], run_info.vz)
        end

        variable = CFL
        variable = select_slice_of_variable(variable; kwargs...)
    elseif variable_name == "minimum_CFL_ion_z"
        # update_speed_z!() requires all dimensions to be present, so do *not* pass kwargs
        # to get_variable() in this case. Instead select a slice of the result.
        speed = get_variable(run_info, "z_advect_speed";
                             normalize_advection_speed_shape=false)
        nt = size(speed, 6)
        nspecies = size(speed, 5)
        variable = allocate_float(nt)
        @begin_serial_region()
        for it ∈ 1:nt
            min_CFL = Inf
            for is ∈ 1:nspecies
                min_CFL = min(min_CFL, get_minimum_CFL_z(@view(speed[:,:,:,:,is,it]), run_info.z))
            end
            variable[it] = min_CFL
        end
        variable = select_slice_of_variable(variable; kwargs...)
    elseif variable_name == "minimum_CFL_ion_vpa"
        # update_speed_z!() requires all dimensions to be present, so do *not* pass kwargs
        # to get_variable() in this case. Instead select a slice of the result.
        speed = get_variable(run_info, "vpa_advect_speed")
        nt = size(speed, 6)
        nspecies = size(speed, 5)
        variable = allocate_float(nt)
        @begin_serial_region()
        for it ∈ 1:nt
            min_CFL = Inf
            for is ∈ 1:nspecies
                min_CFL = min(min_CFL, get_minimum_CFL_vpa(@view(speed[:,:,:,:,is,it]), run_info.vpa))
            end
            variable[it] = min_CFL
        end
        variable = select_slice_of_variable(variable; kwargs...)
    elseif variable_name == "minimum_CFL_electron_z"
        # update_speed_z!() requires all dimensions to be present, so do *not* pass kwargs
        # to get_variable() in this case. Instead select a slice of the result.
        speed = get_variable(run_info, "electron_z_advect_speed";
                             normalize_advection_speed_shape=false)
        nt = size(speed, 5)
        variable = allocate_float(nt)
        @begin_serial_region()
        for it ∈ 1:nt
            min_CFL = get_minimum_CFL_z(@view(speed[:,:,:,:,it]), run_info.z)
            variable[it] = min_CFL
        end
        variable = select_slice_of_variable(variable; kwargs...)
    elseif variable_name == "minimum_CFL_electron_vpa"
        # update_speed_z!() requires all dimensions to be present, so do *not* pass kwargs
        # to get_variable() in this case. Instead select a slice of the result.
        speed = get_variable(run_info, "electron_vpa_advect_speed")
        nt = size(speed, 5)
        variable = allocate_float(nt)
        @begin_serial_region()
        for it ∈ 1:nt
            min_CFL = get_minimum_CFL_vpa(@view(speed[:,:,:,:,it]), run_info.vpa)
            variable[it] = min_CFL
        end
        variable = select_slice_of_variable(variable; kwargs...)
    elseif variable_name == "minimum_CFL_neutral_z"
        # update_speed_z!() requires all dimensions to be present, so do *not* pass kwargs
        # to get_variable() in this case. Instead select a slice of the result.
        speed = get_variable(run_info, "neutral_z_advect_speed";
                             normalize_advection_speed_shape=false)
        nt = size(speed, 7)
        nspecies = size(speed, 6)
        variable = allocate_float(nt)
        @begin_serial_region()
        for it ∈ 1:nt
            min_CFL = Inf
            for isn ∈ 1:nspecies
                min_CFL = min(min_CFL, get_minimum_CFL_neutral_z(@view(speed[:,:,:,:,:,isn,it]), run_info.z))
            end
            variable[it] = min_CFL
        end
        variable = select_slice_of_variable(variable; kwargs...)
    elseif variable_name == "minimum_CFL_neutral_vz"
        # update_speed_z!() requires all dimensions to be present, so do *not* pass kwargs
        # to get_variable() in this case. Instead select a slice of the result.
        speed = get_variable(run_info, "neutral_vz_advect_speed")
        nt = size(speed, 7)
        nspecies = size(speed, 6)
        variable = allocate_float(nt)
        @begin_serial_region()
        for it ∈ 1:nt
            min_CFL = Inf
            for isn ∈ 1:nspecies
                min_CFL = min(min_CFL, get_minimum_CFL_neutral_vz(@view(speed[:,:,:,:,:,isn,it]), run_info.vz))
            end
            variable[it] = min_CFL
        end
        variable = select_slice_of_variable(variable; kwargs...)
    elseif occursin("_timestep_error", variable_name)
        prefix = split(variable_name, "_timestep_error")[1]
        full_order = get_variable(run_info, prefix; kwargs...)
        low_order = get_variable(run_info, prefix * "_loworder"; kwargs...)
        variable = low_order .- full_order
    elseif occursin("_timestep_residual", variable_name)
        prefix = split(variable_name, "_timestep_residual")[1]
        full_order = get_variable(run_info, prefix; kwargs...)
        low_order = get_variable(run_info, prefix * "_loworder"; kwargs...)
        if prefix == "pdf_electron"
            rtol = run_info.input["electron_timestepping"]["rtol"]
            atol = run_info.input["electron_timestepping"]["atol"]
        else
            rtol = run_info.input["timestepping"]["rtol"]
            atol = run_info.input["timestepping"]["atol"]
        end
        variable = @. (low_order - full_order) / (rtol * abs(full_order) + atol)
    elseif occursin("_steady_state_residual", variable_name)
        prefix = split(variable_name, "_steady_state_residual")[1]
        end_step = get_variable(run_info, prefix; kwargs...)
        begin_step = get_variable(run_info, prefix * "_start_last_timestep"; kwargs...)
        if prefix == "f_electron"
            dt = get_variable(run_info, "electron_previous_dt"; kwargs...)
        else
            dt = get_variable(run_info, "previous_dt"; kwargs...)
        end
        dt = reshape(dt, ones(mk_int, ndims(end_step)-1)..., length(dt))
        for i ∈ eachindex(dt)
            if dt[i] ≤ 0.0
                dt[i] = Inf
            end
        end
        variable = (end_step .- begin_step) ./ dt
    elseif occursin("_nonlinear_iterations_per_solve", variable_name)
        prefix = split(variable_name, "_nonlinear_iterations_per_solve")[1]
        nl_nsolves = get_per_step_from_cumulative_variable(
            run_info, "$(prefix)_n_solves"; kwargs...)
        nl_iterations = get_per_step_from_cumulative_variable(
            run_info, "$(prefix)_nonlinear_iterations"; kwargs...)
        variable = nl_iterations ./ nl_nsolves
    elseif occursin("_linear_iterations_per_nonlinear_iteration", variable_name)
        prefix = split(variable_name, "_linear_iterations_per_nonlinear_iteration")[1]
        nl_iterations = get_per_step_from_cumulative_variable(
            run_info, "$(prefix)_nonlinear_iterations"; kwargs...)
        nl_linear_iterations = get_per_step_from_cumulative_variable(
            run_info, "$(prefix)_linear_iterations"; kwargs...)
        variable = nl_linear_iterations ./ nl_iterations
    elseif occursin("_precon_iterations_per_linear_iteration", variable_name)
        prefix = split(variable_name, "_precon_iterations_per_linear_iteration")[1]
        nl_linear_iterations = get_per_step_from_cumulative_variable(
            run_info, "$(prefix)_linear_iterations"; kwargs...)
        nl_precon_iterations = get_per_step_from_cumulative_variable(
            run_info, "$(prefix)_precon_iterations"; kwargs...)
        variable = nl_precon_iterations ./ nl_linear_iterations
    elseif endswith(variable_name, "_per_step") && variable_name ∉ run_info.variable_names
        # If "_per_step" is appended to a variable name, assume it is a cumulative
        # variable, and get the per-step version.
        variable =
            get_per_step_from_cumulative_variable(run_info,
                                                  split(variable_name, "_per_step")[1];
                                                  kwargs...)
    else
        variable = postproc_load_variable(run_info, variable_name; kwargs...)
    end

    return variable
end

"""
    get_z_derivative(run_info, variable_name; kwargs...)

Get (i.e. load or calculate) `variable_name` from `run_info` and calculate its
z-derivative. Returns the z-derivative

`kwargs...` are passed through to `get_variable()`.
"""
function get_z_derivative(run_info, variable_name; kwargs...)
    variable = get_variable(run_info, variable_name; kwargs...)
    z_deriv = similar(variable)

    if ndims(variable) == 3
        # EM field
        nz, nr, nt = size(variable)
        for it ∈ 1:nt, ir ∈ 1:nr
            @views derivative!(z_deriv[:,ir,it], variable[:,ir,it], run_info.z,
                               run_info.z_spectral)
        end
    elseif ndims(variable) == 4
        # Moment variable (ion or neutral)
        nz, nr, nspecies, nt = size(variable)
        for it ∈ 1:nt, is ∈ 1:nspecies, ir ∈ 1:nr
            @views derivative!(z_deriv[:,ir,is,it], variable[:,ir,is,it], run_info.z,
                               run_info.z_spectral)
        end
    elseif ndims(variable) == 6
        # Ion distribution function
        nvpa, nvperp, nz, nr, nspecies, nt = size(variable)
        for it ∈ 1:nt, is ∈ 1:nspecies, ir ∈ 1:nr, ivperp ∈ 1:nvperp, ivpa ∈ 1:nvpa
            @views derivative!(z_deriv[ivpa,ivperp,:,ir,is,it],
                               variable[ivpa,ivperp,:,ir,is,it], run_info.z,
                               run_info.z_spectral)
        end
    elseif ndims(variable) == 7
        # Neutral distribution function
        nvz, nvr, nvzeta, nz, nr, nspecies, nt = size(variable)
        for it ∈ 1:nt, is ∈ 1:nspecies, ir ∈ 1:nr, ivzeta ∈ 1:nvzeta, ivr ∈ 1:nvr, ivz ∈ 1:nvz
            @views derivative!(z_deriv[ivz,ivr,ivzeta,:,ir,is,it],
                               variable[ivz,ivr,ivzeta,:,ir,is,it], run_info.z,
                               run_info.z_spectral)
        end
    else
        error("Unsupported number of dimensions ($(ndims(variable))) for $variable_name")
    end

    return z_deriv
end

"""
Read data which is a function of (z,r,t) or (z,r,species,t)

run_names is a tuple. If it has more than one entry, this means that there are multiple
restarts (which are sequential in time), so concatenate the data from each entry together.
"""
function read_distributed_zr_data!(var::Array{mk_float,N}, var_name::String,
   run_names::Tuple, file_key::String, nblocks::Tuple,
   nz_local::mk_int, nr_local::mk_int, iskip::mk_int; group=nothing) where N
    # dimension of var is [z,r,species,t]

    if group === nothing
        group = "dynamic_data"
    end

    local_tind_start = 1
    local_tind_end = -1
    global_tind_start = 1
    global_tind_end = -1
    for (run_name, nb) in zip(run_names, nblocks)
        for iblock in 0:nb-1
            fid = open_readonly_output_file(run_name,file_key,iblock=iblock,printout=false)
            this_group = get_group(fid, group)
            var_local = load_variable(this_group, var_name)

            ntime_local = size(var_local, N)

            # offset is the amount we have to skip at the beginning of this restart to
            # line up properly with having outputs every iskip since the beginning of the
            # first restart.
            # Note: use rem(x,y,RoundDown) here because this gives a result that's
            # definitely between 0 and y, whereas rem(x,y) or mod(x,y) give negative
            # results for negative x.
            offset = rem(1 - (local_tind_start-1), iskip, RoundDown)
            if offset == 0
                # Actually want offset in the range [1,iskip], so correct if rem()
                # returned 0
                offset = iskip
            end
            if local_tind_start > 1
                # The run being loaded is a restart (as local_tind_start=1 for the first
                # run), so skip the first point, as this is a duplicate of the last point
                # of the previous restart
                offset += 1
            end

            local_tind_end = local_tind_start + ntime_local - 1
            global_tind_end = global_tind_start + length(offset:iskip:ntime_local) - 1

            z_irank, z_nrank, r_irank, r_nrank = load_rank_data(fid)

            # min index set to avoid double assignment of repeated points
            # 1 if irank = 0, 2 otherwise
            imin_r = min(1,r_irank) + 1
            imin_z = min(1,z_irank) + 1
            for ir_local in imin_r:nr_local
                for iz_local in imin_z:nz_local
                    ir_global = iglobal_func(ir_local,r_irank,nr_local)
                    iz_global = iglobal_func(iz_local,z_irank,nz_local)
                    if N == 4
                        var[iz_global,ir_global,:,global_tind_start:global_tind_end] .= var_local[iz_local,ir_local,:,offset:iskip:end]
                    elseif N == 3
                        var[iz_global,ir_global,global_tind_start:global_tind_end] .= var_local[iz_local,ir_local,offset:iskip:end]
                    else
                        error("Unsupported number of dimensions: $N")
                    end
                end
            end
            close(fid)
        end
        local_tind_start = local_tind_end + 1
        global_tind_start = global_tind_end + 1
    end
end

"""
"""
function construct_global_zr_coords(r_local, z_local; ignore_MPI=true)

    function make_global_input(coord_local)
        return OptionsDict(coord_local.name => OptionsDict("ngrid" => coord_local.ngrid,
                                                           "nelement" => coord_local.nelement_global,
                                                           "nelement_local" => coord_local.nelement_global,
                                                           "L" => coord_local.L,
                                                           "discretization" => coord_local.discretization,
                                                           "cheb_option" => coord_local.cheb_option,
                                                           "bc" => coord_local.bc,
                                                           "element_spacing_option" => coord_local.element_spacing_option,),
                          )
    end

    r_global, r_global_spectral = define_coordinate(make_global_input(r_local), "r"; ignore_MPI=ignore_MPI)
    z_global, z_global_spectral = define_coordinate(make_global_input(z_local), "z"; ignore_MPI=ignore_MPI)

    return r_global, r_global_spectral, z_global, z_global_spectral
end

end
