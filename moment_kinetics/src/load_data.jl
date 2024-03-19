"""
"""
module load_data

export open_readonly_output_file
export load_fields_data
export load_pdf_data
export load_neutral_pdf_data
export load_coordinate_data
export load_time_data
export load_block_data
export load_rank_data
export load_species_data
export read_distributed_zr_data!

using ..array_allocation: allocate_float
using ..coordinates: coordinate, define_coordinate
using ..file_io: check_io_implementation, get_group, get_subgroup_keys, get_variable_keys
using ..input_structs: advection_input, grid_input, hdf5, netcdf
using ..interpolation: interpolate_to_grid_1d!
using ..krook_collisions
using ..looping
using ..moment_kinetics_input: mk_input
using ..type_definitions: mk_float, mk_int

using Glob
using HDF5
using MPI

const em_variables = ("phi", "Er", "Ez")
const ion_moment_variables = ("density", "parallel_flow", "parallel_pressure",
                              "thermal_speed", "temperature", "parallel_heat_flux",
                              "collision_frequency_ii", "collision_frequency_ee",
                              "collision_frequency_ei", "sound_speed", "mach_number")
const electron_moment_variables = ("electron_density", "electron_parallel_flow",
                                   "electron_parallel_pressure", "electron_thermal_speed",
                                   "electron_temperature", "electron_parallel_heat_flux")
const neutral_moment_variables = ("density_neutral", "uz_neutral", "pz_neutral",
                                  "thermal_speed_neutral", "temperature_neutral",
                                  "qz_neutral")
const all_moment_variables = tuple(em_variables..., ion_moment_variables...,
                                   electron_moment_variables...,
                                   neutral_moment_variables...)

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
    section = Dict{String,Any}()

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
    load_coordinate_data(fid, name; printout=false, irank=nothing, nrank=nothing)

Load data for the coordinate `name` from a file-handle `fid`.

Returns (`coord`, `spectral`, `chunk_size`). `coord` is a `coordinate` object. `spectral`
is the object used to implement the discretization in this coordinate. `chunk_size` is the
size of chunks in this coordinate that was used when writing to the output file.

If `printout` is set to `true` a message will be printed when this function is called.

If `irank` and `nrank` are passed, then the `coord` and `spectral` objects returned will
be set up for the parallelisation specified by `irank` and `nrank`, rather than the one
implied by the output file.

If `ignore_MPI=true` is passed, the returned coordinates will be created without shared
memory scratch arrays (`ignore_MPI=true` will be passed through to
[`define_coordinate`](@ref)).
"""
function load_coordinate_data(fid, name; printout=false, irank=nothing, nrank=nothing,
                              ignore_MPI=false)
    if printout
        println("Loading $name coordinate data...")
    end

    overview = get_group(fid, "overview")
    parallel_io = load_variable(overview, "parallel_io")

    coord_group = get_group(get_group(fid, "coords"), name)

    ngrid = load_variable(coord_group, "ngrid")
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
    # L = global box length
    L = load_variable(coord_group, "L")
    discretization = load_variable(coord_group, "discretization")
    fd_option = load_variable(coord_group, "fd_option")
    if "cheb_option" ∈ keys(coord_group)
        cheb_option = load_variable(coord_group, "cheb_option")
    else
        # Old output file
        cheb_option = "FFT"
    end
    bc = load_variable(coord_group, "bc")
    if "element_spacing_option" ∈ keys(coord_group)
        element_spacing_option = load_variable(coord_group, "element_spacing_option")
    else
        element_spacing_option = "uniform"
    end
    # Define input to create coordinate struct
    input = grid_input(name, ngrid, nelement_global, nelement_local, nrank, irank, L,
                       discretization, fd_option, cheb_option, bc, advection_input("", 0.0, 0.0, 0.0),
                       MPI.COMM_NULL, element_spacing_option)

    coord, spectral = define_coordinate(input, parallel_io; ignore_MPI=ignore_MPI)

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
    evolve_ppar = load_variable(overview, "evolve_ppar")

    return evolve_density, evolve_upar, evolve_ppar
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
        return density, parallel_flow, parallel_pressure, perpendicular_pressure, parallel_heat_flux, thermal_speed, entropy_production
    else
        return density, parallel_flow, parallel_pressure, parallel_heat_flux, thermal_speed
    end
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

    # Read neutral species pz
    neutral_pz = load_variable(group, "pz_neutral")

    # Read neutral species qz
    neutral_qz = load_variable(group, "qz_neutral")

    # Read neutral species thermal speed
    neutral_thermal_speed = load_variable(group, "thermal_speed_neutral")

    if printout
        println("done.")
    end

    return neutral_density, neutral_uz, neutral_pz, neutral_qz, neutral_thermal_speed
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
function reload_evolving_fields!(pdf, moments, boundary_distributions, restart_prefix_iblock,
                                 time_index, composition, geometry, r, z, vpa, vperp,
                                 vzeta, vr, vz)
    code_time = 0.0
    previous_runs_info = nothing
    restart_had_kinetic_electrons = false
    begin_serial_region()
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
            restart_evolve_density, restart_evolve_upar, restart_evolve_ppar =
                load_mk_options(fid)

            previous_runs_info = load_run_info_history(fid)

            restart_n_ion_species, restart_n_neutral_species = load_species_data(fid)
            restart_r, restart_r_spectral, restart_z, restart_z_spectral, restart_vperp,
                restart_vperp_spectral, restart_vpa, restart_vpa_spectral, restart_vzeta,
                restart_vzeta_spectral, restart_vr,restart_vr_spectral, restart_vz,
                restart_vz_spectral = load_restart_coordinates(fid, r, z, vperp, vpa,
                                                               vzeta, vr, vz, parallel_io)

            # Test whether any interpolation is needed
            interpolation_needed = Dict(
                x.name => x.n != restart_x.n || !all(isapprox.(x.grid, restart_x.grid))
                for (x, restart_x) ∈ ((z, restart_z), (r, restart_r),
                                      (vperp, restart_vperp), (vpa, restart_vpa),
                                      (vzeta, restart_vzeta), (vr, restart_vr),
                                      (vz, restart_vz)))

            neutral_1V = (vzeta.n_global == 1 && vr.n_global == 1)
            restart_neutral_1V = (restart_vzeta.n_global == 1 && restart_vr.n_global == 1)
            if any(geometry.bzeta .!= 0.0) && ((neutral1V && !restart_neutral_1V) ||
                                               (!neutral1V && restart_neutral_1V))
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

            moments.ion.dens .= reload_moment("density", dynamic, time_index, r, z,
                                              r_range, z_range, restart_r,
                                              restart_r_spectral, restart_z,
                                              restart_z_spectral, interpolation_needed)
            moments.ion.dens_updated .= true
            moments.ion.upar .= reload_moment("parallel_flow", dynamic, time_index, r, z,
                                              r_range, z_range, restart_r,
                                              restart_r_spectral, restart_z,
                                              restart_z_spectral, interpolation_needed)
            moments.ion.upar_updated .= true
            moments.ion.ppar .= reload_moment("parallel_pressure", dynamic, time_index, r,
                                              z, r_range, z_range, restart_r,
                                              restart_r_spectral, restart_z,
                                              restart_z_spectral, interpolation_needed)
            moments.ion.ppar_updated .= true
            moments.ion.qpar .= reload_moment("parallel_heat_flux", dynamic, time_index,
                                              r, z, r_range, z_range, restart_r,
                                              restart_r_spectral, restart_z,
                                              restart_z_spectral, interpolation_needed)
            moments.ion.qpar_updated .= true
            moments.ion.vth .= reload_moment("thermal_speed", dynamic, time_index, r, z,
                                             r_range, z_range, restart_r,
                                             restart_r_spectral, restart_z,
                                             restart_z_spectral, interpolation_needed)
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

            if "external_source_controller_integral" ∈ get_variable_keys(dynamic) &&
                    length(moments.ion.external_source_controller_integral) == 1
                moments.ion.external_source_controller_integral .=
                    load_slice(dynamic, "external_source_controller_integral", time_index)
            elseif length(moments.ion.external_source_controller_integral) > 1
                moments.ion.external_source_controller_integral .=
                    reload_moment("external_source_controller_integral", dynamic,
                                  time_index, r, z, r_range, z_range, restart_r,
                                  restart_r_spectral, restart_z, restart_z_spectral,
                                  interpolation_needed)
            end

            pdf.ion.norm .= reload_ion_pdf(dynamic, time_index, moments, r, z, vperp, vpa, r_range,
                                           z_range, vperp_range, vpa_range, restart_r,
                                           restart_r_spectral, restart_z,
                                           restart_z_spectral, restart_vperp,
                                           restart_vperp_spectral, restart_vpa,
                                           restart_vpa_spectral, interpolation_needed,
                                           restart_evolve_density, restart_evolve_upar,
                                           restart_evolve_ppar)
            boundary_distributions_io = get_group(fid, "boundary_distributions")

            boundary_distributions.pdf_rboundary_ion[:,:,:,1,:] .=
                reload_ion_boundary_pdf(boundary_distributions_io,
                                        "pdf_rboundary_ion_left", 1, moments, z, vperp,
                                        vpa, z_range, vperp_range, vpa_range, restart_z,
                                        restart_z_spectral, restart_vperp,
                                        restart_vperp_spectral, restart_vpa,
                                        restart_vpa_spectral, interpolation_needed,
                                        restart_evolve_density, restart_evolve_upar,
                                        restart_evolve_ppar)
            boundary_distributions.pdf_rboundary_ion[:,:,:,2,:] .=
                reload_ion_boundary_pdf(boundary_distributions_io,
                                        "pdf_rboundary_ion_right", r.n, moments, z, vperp,
                                        vpa, z_range, vperp_range, vpa_range, restart_z,
                                        restart_z_spectral, restart_vperp,
                                        restart_vperp_spectral, restart_vpa,
                                        restart_vpa_spectral, interpolation_needed,
                                        restart_evolve_density, restart_evolve_upar,
                                        restart_evolve_ppar)

            moments.electron.dens .= reload_electron_moment("electron_density", dynamic,
                                                            time_index, r, z, r_range,
                                                            z_range, restart_r,
                                                            restart_r_spectral, restart_z,
                                                            restart_z_spectral,
                                                            interpolation_needed)
            moments.electron.dens_updated[] = true
            moments.electron.upar .= reload_electron_moment("electron_parallel_flow",
                                                            dynamic, time_index, r, z,
                                                            r_range, z_range, restart_r,
                                                            restart_r_spectral, restart_z,
                                                            restart_z_spectral,
                                                            interpolation_needed)
            moments.electron.upar_updated[] = true
            moments.electron.ppar .= reload_electron_moment("electron_parallel_pressure",
                                                            dynamic, time_index, r, z,
                                                            r_range, z_range, restart_r,
                                                            restart_r_spectral, restart_z,
                                                            restart_z_spectral,
                                                            interpolation_needed)
            moments.electron.ppar_updated[] = true
            moments.electron.qpar .= reload_electron_moment("electron_parallel_heat_flux",
                                                            dynamic, time_index, r, z,
                                                            r_range, z_range, restart_r,
                                                            restart_r_spectral, restart_z,
                                                            restart_z_spectral,
                                                            interpolation_needed)
            moments.electron.qpar_updated[] = true
            moments.electron.vth .= reload_electron_moment("electron_thermal_speed",
                                                           dynamic, time_index, r, z,
                                                           r_range, z_range, restart_r,
                                                           restart_r_spectral, restart_z,
                                                           restart_z_spectral,
                                                           interpolation_needed)

            # For now, electrons are always fully moment_kinetic
            restart_electron_evolve_density, restart_electron_evolve_upar,
                restart_electron_evolve_ppar = true, true, true
            electron_evolve_density, electron_evolve_upar, electron_evolve_ppar =
                true, true, true
            restart_had_kinetic_electrons = ("f_electron" ∈ keys(dynamic))
            if pdf.electron !== nothing && restart_had_kinetic_electrons
                pdf.electron.norm .=
                    reload_electron_pdf(dynamic, time_index, moments, r, z, vperp, vpa,
                                        r_range, z_range, vperp_range, vpa_range,
                                        restart_r, restart_r_spectral, restart_z,
                                        restart_z_spectral, restart_vperp,
                                        restart_vperp_spectral, restart_vpa,
                                        restart_vpa_spectral, interpolation_needed,
                                        restart_evolve_density, restart_evolve_upar,
                                        restart_evolve_ppar)
            end

            if composition.n_neutral_species > 0
                moments.neutral.dens .= reload_moment("density_neutral", dynamic,
                                                      time_index, r, z, r_range, z_range,
                                                      restart_r, restart_r_spectral,
                                                      restart_z, restart_z_spectral,
                                                      interpolation_needed)
                moments.neutral.dens_updated .= true
                moments.neutral.uz .= reload_moment("uz_neutral", dynamic, time_index, r,
                                                    z, r_range, z_range, restart_r,
                                                    restart_r_spectral, restart_z,
                                                    restart_z_spectral,
                                                    interpolation_needed)
                moments.neutral.uz_updated .= true
                moments.neutral.pz .= reload_moment("pz_neutral", dynamic, time_index, r,
                                                    z, r_range, z_range, restart_r,
                                                    restart_r_spectral, restart_z,
                                                    restart_z_spectral,
                                                    interpolation_needed)
                moments.neutral.pz_updated .= true
                moments.neutral.qz .= reload_moment("qz_neutral", dynamic, time_index, r,
                                                    z, r_range, z_range, restart_r,
                                                    restart_r_spectral, restart_z,
                                                    restart_z_spectral,
                                                    interpolation_needed)
                moments.neutral.qz_updated .= true
                moments.neutral.vth .= reload_moment("thermal_speed_neutral", dynamic,
                                                     time_index, r, z, r_range, z_range,
                                                     restart_r, restart_r_spectral,
                                                     restart_z, restart_z_spectral,
                                                     interpolation_needed)

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
                                       restart_evolve_ppar)

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
                                                restart_evolve_upar, restart_evolve_ppar)
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
                                                restart_evolve_upar, restart_evolve_ppar)
            end
        finally
            close(fid)
        end
    end

    return code_time, previous_runs_info, time_index, restart_had_kinetic_electrons
end

"""
Reload electron pdf and moments from an existing output file.
"""
function reload_electron_data!(pdf, moments, restart_prefix_iblock, time_index, geometry,
                               r, z, vpa, vperp, vzeta, vr, vz)
    code_time = 0.0
    previous_runs_info = nothing
    begin_serial_region()
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
            #restart_evolve_density, restart_evolve_upar, restart_evolve_ppar =
            #    load_mk_options(fid)
            # For now, electrons are always fully moment_kinetic
            restart_evolve_density, restart_evolve_upar, restart_evolve_ppar = true, true,
                                                                               true
            evolve_density, evolve_upar, evolve_ppar = true, true, true

            previous_runs_info = load_run_info_history(fid)

            restart_n_ion_species, restart_n_neutral_species = load_species_data(fid)
            restart_r, restart_r_spectral, restart_z, restart_z_spectral, restart_vperp,
                restart_vperp_spectral, restart_vpa, restart_vpa_spectral, restart_vzeta,
                restart_vzeta_spectral, restart_vr,restart_vr_spectral, restart_vz,
                restart_vz_spectral = load_restart_coordinates(fid, r, z, vperp, vpa,
                                                               vzeta, vr, vz, parallel_io)

            # Test whether any interpolation is needed
            interpolation_needed = Dict(
                x.name => x.n != restart_x.n || !all(isapprox.(x.grid, restart_x.grid))
                for (x, restart_x) ∈ ((z, restart_z), (r, restart_r),
                                      (vperp, restart_vperp), (vpa, restart_vpa)))

            neutral_1V = (vzeta.n_global == 1 && vr.n_global == 1)
            restart_neutral_1V = (restart_vzeta.n_global == 1 && restart_vr.n_global == 1)
            if geometry.bzeta != 0.0 && ((neutral1V && !restart_neutral_1V) ||
                                         (!neutral1V && restart_neutral_1V))
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

            moments.electron.dens .=
                reload_electron_moment("electron_density", dynamic, time_index, r, z,
                                       r_range, z_range, restart_r, restart_r_spectral,
                                       restart_z, restart_z_spectral,
                                       interpolation_needed)
            moments.electron.dens_updated[] = true
            moments.electron.upar .=
                reload_electron_moment("electron_parallel_flow", dynamic, time_index, r,
                                       z, r_range, z_range, restart_r, restart_r_spectral,
                                       restart_z, restart_z_spectral,
                                       interpolation_needed)
            moments.electron.upar_updated[] = true
            moments.electron.ppar .=
                reload_electron_moment("electron_parallel_pressure", dynamic, time_index,
                                       r, z, r_range, z_range, restart_r,
                                       restart_r_spectral, restart_z, restart_z_spectral,
                                       interpolation_needed)
            moments.electron.ppar_updated[] = true
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
                                    restart_evolve_ppar)
        finally
            close(fid)
        end
    end

    return code_time, previous_runs_info, time_index
end

function load_restart_coordinates(fid, r, z, vperp, vpa, vzeta, vr, vz, parallel_io)
    if parallel_io
        restart_z, restart_z_spectral, _ =
            load_coordinate_data(fid, "z"; irank=z.irank, nrank=z.nrank, ignore_MPI=true)
        restart_r, restart_r_spectral, _ =
            load_coordinate_data(fid, "r"; irank=r.irank, nrank=r.nrank, ignore_MPI=true)
        restart_vperp, restart_vperp_spectral, _ =
            load_coordinate_data(fid, "vperp"; irank=vperp.irank, nrank=vperp.nrank, ignore_MPI=true)
        restart_vpa, restart_vpa_spectral, _ =
            load_coordinate_data(fid, "vpa"; irank=vpa.irank, nrank=vpa.nrank, ignore_MPI=true)
        restart_vzeta, restart_vzeta_spectral, _ =
            load_coordinate_data(fid, "vzeta"; irank=vzeta.irank, nrank=vzeta.nrank, ignore_MPI=true)
        restart_vr, restart_vr_spectral, _ =
            load_coordinate_data(fid, "vr"; irank=vr.irank, nrank=vr.nrank, ignore_MPI=true)
        restart_vz, restart_vz_spectral, _ =
            load_coordinate_data(fid, "vz"; irank=vz.irank, nrank=vz.nrank, ignore_MPI=true)
    else
        restart_z, restart_z_spectral, _ = load_coordinate_data(fid, "z"; ignore_MPI=true)
        restart_r, restart_r_spectral, _ = load_coordinate_data(fid, "r"; ignore_MPI=true)
        restart_vperp, restart_vperp_spectral, _ =
            load_coordinate_data(fid, "vperp"; ignore_MPI=true)
        restart_vpa, restart_vpa_spectral, _ = load_coordinate_data(fid, "vpa"; ignore_MPI=true)
        restart_vzeta, restart_vzeta_spectral, _ =
            load_coordinate_data(fid, "vzeta"; ignore_MPI=true)
        restart_vr, restart_vr_spectral, _ = load_coordinate_data(fid, "vr"; ignore_MPI=true)
        restart_vz, restart_vz_spectral, _ = load_coordinate_data(fid, "vz"; ignore_MPI=true)

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
            if coord.irank == coord.nrank - 1
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
                        restart_evolve_density, restart_evolve_upar, restart_evolve_ppar)

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
         moments.evolve_upar == restart_evolve_upar && moments.evolve_ppar ==
         restart_evolve_ppar)
        || (!moments.evolve_upar && !restart_evolve_upar &&
            !moments.evolve_ppar && !restart_evolve_ppar)
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
    elseif (!moments.evolve_upar && !moments.evolve_ppar &&
            restart_evolve_upar && !restart_evolve_ppar)
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
    elseif (!moments.evolve_upar && !moments.evolve_ppar &&
            !restart_evolve_upar && restart_evolve_ppar)
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
    elseif (!moments.evolve_upar && !moments.evolve_ppar &&
            restart_evolve_upar && restart_evolve_ppar)
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
    elseif (moments.evolve_upar && !moments.evolve_ppar &&
            !restart_evolve_upar && !restart_evolve_ppar)
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
    elseif (moments.evolve_upar && !moments.evolve_ppar &&
            !restart_evolve_upar && restart_evolve_ppar)
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
    elseif (moments.evolve_upar && !moments.evolve_ppar &&
            restart_evolve_upar && restart_evolve_ppar)
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
    elseif (!moments.evolve_upar && moments.evolve_ppar &&
            !restart_evolve_upar && !restart_evolve_ppar)
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
    elseif (!moments.evolve_upar && moments.evolve_ppar &&
            restart_evolve_upar && !restart_evolve_ppar)
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
    elseif (!moments.evolve_upar && moments.evolve_ppar &&
            restart_evolve_upar && restart_evolve_ppar)
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
    elseif (moments.evolve_upar && moments.evolve_ppar &&
            !restart_evolve_upar && !restart_evolve_ppar)
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
    elseif (moments.evolve_upar && moments.evolve_ppar &&
            restart_evolve_upar && !restart_evolve_ppar)
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
    elseif (moments.evolve_upar && moments.evolve_ppar &&
            !restart_evolve_upar && restart_evolve_ppar)
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
              * " evolve_ppar=", moments.evolve_ppar
              * " restart_evolve_density=", restart_evolve_density
              * " restart_evolve_upar=", restart_evolve_upar
              * " restart_evolve_ppar=", restart_evolve_ppar)
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
    if moments.evolve_ppar && !restart_evolve_ppar
        # Need to normalise by vth
        for is ∈ nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n
            this_pdf[:,:,iz,ir,is] .*= moments.ion.vth[iz,ir,is]
        end
    elseif !moments.evolve_ppar && restart_evolve_ppar
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
                                 restart_evolve_ppar)
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
         moments.evolve_upar == restart_evolve_upar && moments.evolve_ppar ==
         restart_evolve_ppar)
        || (!moments.evolve_upar && !restart_evolve_upar &&
            !moments.evolve_ppar && !restart_evolve_ppar)
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
    elseif (!moments.evolve_upar && !moments.evolve_ppar &&
            restart_evolve_upar && !restart_evolve_ppar)
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
    elseif (!moments.evolve_upar && !moments.evolve_ppar &&
            !restart_evolve_upar && restart_evolve_ppar)
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
    elseif (!moments.evolve_upar && !moments.evolve_ppar &&
            restart_evolve_upar && restart_evolve_ppar)
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
    elseif (moments.evolve_upar && !moments.evolve_ppar &&
            !restart_evolve_upar && !restart_evolve_ppar)
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
    elseif (moments.evolve_upar && !moments.evolve_ppar &&
            !restart_evolve_upar && restart_evolve_ppar)
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
    elseif (moments.evolve_upar && !moments.evolve_ppar &&
            restart_evolve_upar && restart_evolve_ppar)
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
    elseif (!moments.evolve_upar && moments.evolve_ppar &&
            !restart_evolve_upar && !restart_evolve_ppar)
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
    elseif (!moments.evolve_upar && moments.evolve_ppar &&
            restart_evolve_upar && !restart_evolve_ppar)
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
    elseif (!moments.evolve_upar && moments.evolve_ppar &&
            restart_evolve_upar && restart_evolve_ppar)
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
    elseif (moments.evolve_upar && moments.evolve_ppar &&
            !restart_evolve_upar && !restart_evolve_ppar)
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
    elseif (moments.evolve_upar && moments.evolve_ppar &&
            restart_evolve_upar && !restart_evolve_ppar)
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
    elseif (moments.evolve_upar && moments.evolve_ppar &&
            !restart_evolve_upar && restart_evolve_ppar)
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
              * " evolve_ppar=", moments.evolve_ppar
              * " restart_evolve_density=", restart_evolve_density
              * " restart_evolve_upar=", restart_evolve_upar
              * " restart_evolve_ppar=", restart_evolve_ppar)
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
    if moments.evolve_ppar && !restart_evolve_ppar
        # Need to normalise by vth
        for is ∈ nspecies, iz ∈ 1:z.n
            this_pdf[:,:,iz,is] .*= moments.ion.vth[iz,ir,is]
        end
    elseif !moments.evolve_ppar && restart_evolve_ppar
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
                             restart_evolve_ppar)

    # Currently, electrons are always fully moment-kinetic
    evolve_density = true
    evolve_upar = true
    evolve_ppar = true

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
         evolve_upar == restart_evolve_upar && evolve_ppar ==
         restart_evolve_ppar)
        || (!evolve_upar && !restart_evolve_upar &&
            !evolve_ppar && !restart_evolve_ppar)
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
    elseif (!evolve_upar && !evolve_ppar &&
            restart_evolve_upar && !restart_evolve_ppar)
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
    elseif (!evolve_upar && !evolve_ppar &&
            !restart_evolve_upar && restart_evolve_ppar)
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
    elseif (!evolve_upar && !evolve_ppar &&
            restart_evolve_upar && restart_evolve_ppar)
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
    elseif (evolve_upar && !evolve_ppar &&
            !restart_evolve_upar && !restart_evolve_ppar)
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
    elseif (evolve_upar && !evolve_ppar &&
            !restart_evolve_upar && restart_evolve_ppar)
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
    elseif (evolve_upar && !evolve_ppar &&
            restart_evolve_upar && restart_evolve_ppar)
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
    elseif (!evolve_upar && evolve_ppar &&
            !restart_evolve_upar && !restart_evolve_ppar)
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
    elseif (!evolve_upar && evolve_ppar &&
            restart_evolve_upar && !restart_evolve_ppar)
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
    elseif (!evolve_upar && evolve_ppar &&
            restart_evolve_upar && restart_evolve_ppar)
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
    elseif (evolve_upar && evolve_ppar &&
            !restart_evolve_upar && !restart_evolve_ppar)
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
    elseif (evolve_upar && evolve_ppar &&
            restart_evolve_upar && !restart_evolve_ppar)
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
    elseif (evolve_upar && evolve_ppar &&
            !restart_evolve_upar && restart_evolve_ppar)
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
              * " evolve_ppar=", evolve_ppar
              * " restart_evolve_density=", restart_evolve_density
              * " restart_evolve_upar=", restart_evolve_upar
              * " restart_evolve_ppar=", restart_evolve_ppar)
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
    if evolve_ppar && !restart_evolve_ppar
        # Need to normalise by vth
        for ir ∈ 1:r.n, iz ∈ 1:z.n
            this_pdf[:,:,iz,ir] .*= moments.electron.vth[iz,ir]
        end
    elseif !evolve_ppar && restart_evolve_ppar
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
                            restart_evolve_upar, restart_evolve_ppar)
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
         moments.evolve_ppar == restart_evolve_ppar)
        || (!moments.evolve_upar && !restart_evolve_upar &&
            !moments.evolve_ppar && !restart_evolve_ppar)
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
    elseif (!moments.evolve_upar && !moments.evolve_ppar &&
            restart_evolve_upar && !restart_evolve_ppar)
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
    elseif (!moments.evolve_upar && !moments.evolve_ppar &&
            !restart_evolve_upar && restart_evolve_ppar)
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
    elseif (!moments.evolve_upar && !moments.evolve_ppar &&
            restart_evolve_upar && restart_evolve_ppar)
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
    elseif (moments.evolve_upar && !moments.evolve_ppar &&
            !restart_evolve_upar && !restart_evolve_ppar)
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
    elseif (moments.evolve_upar && !moments.evolve_ppar &&
            !restart_evolve_upar && restart_evolve_ppar)
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
    elseif (moments.evolve_upar && !moments.evolve_ppar &&
            restart_evolve_upar && restart_evolve_ppar)
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
    elseif (!moments.evolve_upar && moments.evolve_ppar &&
            !restart_evolve_upar && !restart_evolve_ppar)
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
    elseif (!moments.evolve_upar && moments.evolve_ppar &&
            restart_evolve_upar && !restart_evolve_ppar)
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
    elseif (!moments.evolve_upar && moments.evolve_ppar &&
            restart_evolve_upar && restart_evolve_ppar)
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
    elseif (moments.evolve_upar && moments.evolve_ppar &&
            !restart_evolve_upar && !restart_evolve_ppar)
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
    elseif (moments.evolve_upar && moments.evolve_ppar &&
            restart_evolve_upar && !restart_evolve_ppar)
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
    elseif (moments.evolve_upar && moments.evolve_ppar &&
            !restart_evolve_upar && restart_evolve_ppar)
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
              * " evolve_ppar=", moments.evolve_ppar
              * " restart_evolve_density=", restart_evolve_density
              * " restart_evolve_upar=", restart_evolve_upar
              * " restart_evolve_ppar=", restart_evolve_ppar)
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
    if moments.evolve_ppar && !restart_evolve_ppar
        # Need to normalise by vth
        for is ∈ nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n
            this_pdf[:,:,:,iz,ir,is] .*= moments.neutral.vth[iz,ir,is]
        end
    elseif !moments.evolve_ppar && restart_evolve_ppar
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
                                     restart_evolve_upar, restart_evolve_ppar)
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
         moments.evolve_upar == restart_evolve_upar && moments.evolve_ppar ==
         restart_evolve_ppar)
        || (!moments.evolve_upar && !restart_evolve_upar &&
            !moments.evolve_ppar && !restart_evolve_ppar)
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
    elseif (!moments.evolve_upar && !moments.evolve_ppar &&
            restart_evolve_upar && !restart_evolve_ppar)
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
    elseif (!moments.evolve_upar && !moments.evolve_ppar &&
            !restart_evolve_upar && restart_evolve_ppar)
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
    elseif (!moments.evolve_upar && !moments.evolve_ppar &&
            restart_evolve_upar && restart_evolve_ppar)
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
    elseif (moments.evolve_upar && !moments.evolve_ppar &&
            !restart_evolve_upar && !restart_evolve_ppar)
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
    elseif (moments.evolve_upar && !moments.evolve_ppar &&
            !restart_evolve_upar && restart_evolve_ppar)
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
    elseif (moments.evolve_upar && !moments.evolve_ppar &&
            restart_evolve_upar && restart_evolve_ppar)
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
    elseif (!moments.evolve_upar && moments.evolve_ppar &&
            !restart_evolve_upar && !restart_evolve_ppar)
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
    elseif (!moments.evolve_upar && moments.evolve_ppar &&
            restart_evolve_upar && !restart_evolve_ppar)
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
    elseif (!moments.evolve_upar && moments.evolve_ppar &&
            restart_evolve_upar && restart_evolve_ppar)
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
    elseif (moments.evolve_upar && moments.evolve_ppar &&
            !restart_evolve_upar && !restart_evolve_ppar)
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
    elseif (moments.evolve_upar && moments.evolve_ppar &&
            restart_evolve_upar && !restart_evolve_ppar)
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
    elseif (moments.evolve_upar && moments.evolve_ppar &&
            !restart_evolve_upar && restart_evolve_ppar)
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
              * " evolve_ppar=", moments.evolve_ppar
              * " restart_evolve_density=", restart_evolve_density
              * " restart_evolve_upar=", restart_evolve_upar
              * " restart_evolve_ppar=", restart_evolve_ppar)
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
    if moments.evolve_ppar && !restart_evolve_ppar
        # Need to normalise by vth
        for is ∈ nspecies, iz ∈ 1:z.n
            this_pdf[:,:,:,iz,is] .*= moments.neutral.vth[iz,ir,is]
        end
    elseif !moments.evolve_ppar && restart_evolve_ppar
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

`run_dir` is the directory to read output from.

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
state files.

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
                               initial_electron=false)
    if length(run_dir) == 0
        error("No run_dir passed")
    end
    if initial_electron && !dfns
        error("When `initial_electron=true` is passed, `dfns=true` must also be passed")
    end
    if length(run_dir) > 1
        run_info = Tuple(get_run_info_no_setup(r; itime_min=itime_min,
                                               itime_max=itime_max, itime_skip=itime_skip,
                                               dfns=dfns,
                                               initial_electron=initial_electron)
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

    if !isdir(this_run_dir)
        error("$this_run_dir is not a directory")
    end

    # Normalise by removing any trailing slash - with a slash basename() would return an
    # empty string
    this_run_dir = rstrip(this_run_dir, '/')

    run_name = basename(this_run_dir)
    base_prefix = joinpath(this_run_dir, run_name)
    if restart_index === nothing
        # Find output files from all restarts in the directory
        counter = 1
        run_prefixes = Vector{String}()
        while true
            # Test if output files exist for this value of counter
            prefix_with_count = base_prefix * "_$counter"
            if length(glob(basename(prefix_with_count) * ".*.h5", dirname(prefix_with_count))) > 0 ||
                length(glob(basename(prefix_with_count) * ".*.cdf", dirname(prefix_with_count))) > 0

                push!(run_prefixes, prefix_with_count)
            else
                # No more output files found
                break
            end
            counter += 1
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
        num_diss_params, manufactured_solns_input = mk_input(input)

    n_ion_species, n_neutral_species = load_species_data(file_final_restart)
    evolve_density, evolve_upar, evolve_ppar = load_mk_options(file_final_restart)

    z_local, z_local_spectral, z_chunk_size =
        load_coordinate_data(file_final_restart, "z"; ignore_MPI=true)
    r_local, r_local_spectral, r_chunk_size =
        load_coordinate_data(file_final_restart, "r"; ignore_MPI=true)
    r, r_spectral, z, z_spectral = construct_global_zr_coords(r_local, z_local;
                                                              ignore_MPI=true)

    if dfns
        vperp, vperp_spectral, vperp_chunk_size =
            load_coordinate_data(file_final_restart, "vperp"; ignore_MPI=true)
        vpa, vpa_spectral, vpa_chunk_size =
            load_coordinate_data(file_final_restart, "vpa"; ignore_MPI=true)

        if n_neutral_species > 0
            vzeta, vzeta_spectral, vzeta_chunk_size =
                load_coordinate_data(file_final_restart, "vzeta"; ignore_MPI=true)
            vr, vr_spectral, vr_chunk_size =
                load_coordinate_data(file_final_restart, "vr"; ignore_MPI=true)
            vz, vz_spectral, vz_chunk_size =
                load_coordinate_data(file_final_restart, "vz"; ignore_MPI=true)
        else
            dummy_adv_input = advection_input("default", 1.0, 0.0, 0.0)
            dummy_comm = MPI.COMM_NULL
            dummy_input = grid_input("dummy", 1, 1, 1, 1, 0, 1.0,
                                     "chebyshev_pseudospectral", "", "", "periodic",
                                     dummy_adv_input, dummy_comm, "uniform")
            vzeta, vzeta_spectral = define_coordinate(dummy_input)
            vzeta_chunk_size = 1
            vr, vr_spectral = define_coordinate(dummy_input)
            vr_chunk_size = 1
            vz, vz_spectral = define_coordinate(dummy_input)
            vz_chunk_size = 1
        end
    end

    if parallel_io
        files = fids0
    else
        # Don't keep open files as read_distributed_zr_data!(), etc. open the files
        # themselves
        files = run_prefixes
    end

    if dfns
        run_info = (run_name=run_name, run_prefix=base_prefix, parallel_io=parallel_io,
                    ext=ext, nblocks=nblocks, files=files, input=input,
                    n_ion_species=n_ion_species, n_neutral_species=n_neutral_species,
                    evolve_moments=evolve_moments, composition=composition,
                    species=species, collisions=collisions, geometry=geometry,
                    drive_input=drive_input, num_diss_params=num_diss_params,
                    evolve_density=evolve_density, evolve_upar=evolve_upar,
                    evolve_ppar=evolve_ppar,
                    manufactured_solns_input=manufactured_solns_input, nt=nt,
                    nt_unskipped=nt_unskipped, restarts_nt=restarts_nt,
                    itime_min=itime_min, itime_skip=itime_skip, itime_max=itime_max,
                    time=time, r=r, z=z, vperp=vperp, vpa=vpa, vzeta=vzeta, vr=vr, vz=vz,
                    r_local=r_local, z_local=z_local, r_spectral=r_spectral,
                    z_spectral=z_spectral, vperp_spectral=vperp_spectral,
                    vpa_spectral=vpa_spectral, vzeta_spectral=vzeta_spectral,
                    vr_spectral=vr_spectral, vz_spectral=vz_spectral,
                    r_chunk_size=r_chunk_size, z_chunk_size=z_chunk_size,
                    vperp_chunk_size=vperp_chunk_size, vpa_chunk_size=vpa_chunk_size,
                    vzeta_chunk_size=vzeta_chunk_size, vr_chunk_size=vr_chunk_size,
                    vz_chunk_size=vz_chunk_size, dfns=dfns)
    else
        run_info = (run_name=run_name, run_prefix=base_prefix, parallel_io=parallel_io,
                    ext=ext, nblocks=nblocks, files=files, input=input,
                    n_ion_species=n_ion_species, n_neutral_species=n_neutral_species,
                    evolve_moments=evolve_moments, composition=composition,
                    species=species, collisions=collisions, geometry=geometry,
                    drive_input=drive_input, num_diss_params=num_diss_params,
                    evolve_density=evolve_density, evolve_upar=evolve_upar,
                    evolve_ppar=evolve_ppar,
                    manufactured_solns_input=manufactured_solns_input, nt=nt,
                    nt_unskipped=nt_unskipped, restarts_nt=restarts_nt,
                    itime_min=itime_min, itime_skip=itime_skip, itime_max=itime_max,
                    time=time, r=r, z=z, r_local=r_local, z_local=z_local,
                    r_spectral=r_spectral, z_spectral=z_spectral,
                    r_chunk_size=r_chunk_size, z_chunk_size=z_chunk_size, dfns=dfns)
    end

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
                           ivzeta=nothing, ivr=nothing, ivz=nothing)

Load a variable

`run_info` is the information about a run returned by
`makie_post_processing.get_run_info()`.

`variable_name` is the name of the variable to load.

The keyword arguments `it`, `is`, `ir`, `iz`, `ivperp`, `ivpa`, `ivzeta`, `ivr`, and `ivz`
can be set to an integer or a range (e.g. `3:8` or `3:2:8`) to select subsets of the data.
Only the data for the subset requested will be loaded from the output file (mostly - when
loading fields or moments from runs which used `parallel_io = false`, the full array will
be loaded and then sliced).
"""
function postproc_load_variable(run_info, variable_name; it=nothing, is=nothing,
                                ir=nothing, iz=nothing, ivperp=nothing, ivpa=nothing,
                                ivzeta=nothing, ivr=nothing, ivz=nothing)
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
            nvperp = run_info.vperp.n
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
            nvpa = run_info.vpa.n
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
            nvzeta = run_info.vzeta.n
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
            nvr = run_info.vr.n
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
            nvz = run_info.vz.n
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
        # Get HDF5/NetCDF variables directly and load slices
        variable = Tuple(get_group(f, "dynamic_data")[variable_name]
                         for f ∈ run_info.files)
        nd = ndims(variable[1])

        if nd == 3
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
                nspecies = size(variable[1], 3)
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
                nspecies = size(variable[1], 3)
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
                    if nd == 3
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

                    if nd == 3
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
        if variable_name ∈ em_variables
            nd = 3
        elseif variable_name ∈ electron_dfn_variables
            nd = 5
        elseif variable_name ∈ ion_dfn_variables
            nd = 6
        elseif variable_name ∈ neutral_dfn_variables
            nd = 7
        else
            # Ion or neutral moment variable
            nd = 4
        end

        if nd == 3
            result = allocate_float(run_info.z.n, run_info.r.n, run_info.nt)
            read_distributed_zr_data!(result, variable_name, run_info.files,
                                      run_info.ext, run_info.nblocks, run_info.z_local.n,
                                      run_info.r_local.n, run_info.itime_skip)
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
                                      run_info.r_local.n, run_info.itime_skip)
            result = result[iz,ir,is,it]
        elseif nd === 5
            result = load_distributed_electron_pdf_slice(run_info.files, run_info.nblocks,
                                                         it, run_info.n_ion_species,
                                                         run_info.r_local,
                                                         run_info.z_local, run_info.vperp,
                                                         run_info.vpa; ir=ir, iz=iz,
                                                         ivperp=ivperp, ivpa=ivpa)
        elseif nd === 6
            result = load_distributed_ion_pdf_slice(run_info.files, run_info.nblocks, it,
                                                    run_info.n_ion_species,
                                                    run_info.r_local, run_info.z_local,
                                                    run_info.vperp, run_info.vpa;
                                                    is=(is === (:) ? nothing : is),
                                                    ir=ir, iz=iz, ivperp=ivperp,
                                                    ivpa=ivpa)
        elseif nd === 7
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

function get_variable(run_info, variable_name; kwargs...)
    if variable_name == "temperature"
        vth = postproc_load_variable(run_info, "thermal_speed"; kwargs...)
        variable = vth.^2
    elseif variable_name == "collision_frequency_ii"
        n = postproc_load_variable(run_info, "density"; kwargs...)
        vth = postproc_load_variable(run_info, "thermal_speed"; kwargs...)
        variable = get_collision_frequency_ii(run_info.collisions, n, vth)
    elseif variable_name == "collision_frequency_ee"
        n = postproc_load_variable(run_info, "electron_density"; kwargs...)
        vth = postproc_load_variable(run_info, "electron_thermal_speed"; kwargs...)
        variable = get_collision_frequency_ee(run_info.collisions, n, vth)
    elseif variable_name == "collision_frequency_ei"
        n = postproc_load_variable(run_info, "electron_density"; kwargs...)
        vth = postproc_load_variable(run_info, "electron_thermal_speed"; kwargs...)
        variable = get_collision_frequency_ei(run_info.collisions, n, vth)
    elseif variable_name == "temperature_neutral"
        vth = postproc_load_variable(run_info, "thermal_speed_neutral"; kwargs...)
        variable = vth.^2
    elseif variable_name == "sound_speed"
        T_e = run_info.composition.T_e
        T_i = get_variable(run_info, "temperature"; kwargs...)

        # Adiabatic index. Not too clear what value should be (see e.g. [Riemann 1991,
        # below eq. (39)], or discussion of Bohm criterion in Stangeby's book.
        gamma = 3.0

        # Factor of 0.5 needed because temperatures are normalised to mi*cref^2, not Tref
        variable = @. sqrt(0.5*(T_e + gamma*T_i))
    elseif variable_name == "mach_number"
        upar = get_variable(run_info, "parallel_flow"; kwargs...)
        cs = get_variable(run_info, "sound_speed"; kwargs...)
        variable = upar ./ cs
    else
        variable = postproc_load_variable(run_info, variable_name; kwargs...)
    end

    return variable
end

"""
Read data which is a function of (z,r,t) or (z,r,species,t)

run_names is a tuple. If it has more than one entry, this means that there are multiple
restarts (which are sequential in time), so concatenate the data from each entry together.
"""
function read_distributed_zr_data!(var::Array{mk_float,N}, var_name::String,
   run_names::Tuple, file_key::String, nblocks::Tuple,
   nz_local::mk_int,nr_local::mk_int,iskip::mk_int) where N
    # dimension of var is [z,r,species,t]

    local_tind_start = 1
    local_tind_end = -1
    global_tind_start = 1
    global_tind_end = -1
    for (run_name, nb) in zip(run_names, nblocks)
        for iblock in 0:nb-1
            fid = open_readonly_output_file(run_name,file_key,iblock=iblock,printout=false)
            group = get_group(fid, "dynamic_data")
            var_local = load_variable(group, var_name)

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
function construct_global_zr_coords(r_local, z_local; ignore_MPI=false)

    function make_global_input(coord_local)
        return grid_input(coord_local.name, coord_local.ngrid,
            coord_local.nelement_global, coord_local.nelement_global, 1, 0, coord_local.L,
            coord_local.discretization, coord_local.fd_option, coord_local.cheb_option, coord_local.bc,
            coord_local.advection, MPI.COMM_NULL, coord_local.element_spacing_option)
    end

    r_global, r_global_spectral = define_coordinate(make_global_input(r_local);
                                                    ignore_MPI=ignore_MPI)
    z_global, z_global_spectral = define_coordinate(make_global_input(z_local);
                                                    ignore_MPI=ignore_MPI)

    return r_global, r_global_spectral, z_global, z_global_spectral
end

end
