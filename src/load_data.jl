"""
"""
module load_data

export open_readonly_output_file
export load_fields_data
export load_charged_particle_moments_data
export load_neutral_particle_moments_data
export load_pdf_data
export load_neutral_pdf_data
export load_coordinate_data
export load_time_data
export load_block_data
export load_rank_data
export load_species_data

using ..array_allocation: allocate_float
using ..coordinates: coordinate, define_coordinate
using ..file_io: get_group, get_subgroup_keys, get_variable_keys
using ..input_structs: advection_input, grid_input
using ..interpolation: interpolate_to_grid_1d!
using ..looping
using ..type_definitions: mk_int

using HDF5
using MPI
using NCDatasets

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
        fid = h5open(filename, "r")
    else
        if printout
            print("Opening ", filename, " to read NetCDF data...")
        end
        # open the netcdf file with given filename for reading
        fid = NCDataset(filename, "r")
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
function load_variable() end
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
function load_variable(file_or_group::NCDataset, name::String)
    # This overload deals with cases where fid is a NetCDF `Dataset` (which could be a
    # file or a group).
    try
        if size(file_or_group[name].var) == ()
            var = file_or_group[name].var[]
        else
            var = file_or_group[name].var[:]
        end
        if isa(var, Char)
            var = (var == Char(true))
        end
        return var
    catch
        println("An error occured while loading $name")
        rethrow()
    end
end

"""
Load a slice of a single variable from a file
"""
function load_slice() end
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
function load_slice(file_or_group::NCDataset, name::String, slices_or_indices...)
    # This overload deals with cases where fid is a NetCDF `Dataset` (which could be a
    # file or a group).
    try
        var = file_or_group[name].var[slices_or_indices...]
        return var
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
"""
function load_coordinate_data(fid, name; printout=false, irank=nothing, nrank=nothing)
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
    bc = load_variable(coord_group, "bc")

    # Define input to create coordinate struct
    input = grid_input(name, ngrid, nelement_global, nelement_local, nrank, irank, L,
                       discretization, fd_option, bc, advection_input("", 0.0, 0.0, 0.0),
                       MPI.COMM_NULL)

    coord, spectral = define_coordinate(input, parallel_io)

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
    
    if printout
        println("done.")
    end

    return z_irank, r_irank
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
function load_charged_particle_moments_data(fid; printout=false)
    if printout
        print("Loading charged particle velocity moments data...")
    end

    group = get_group(fid, "dynamic_data")

    # Read charged species density
    density = load_variable(group, "density")

    # Read charged species parallel flow
    parallel_flow = load_variable(group, "parallel_flow")

    # Read charged species parallel pressure
    parallel_pressure = load_variable(group, "parallel_pressure")

    # Read charged_species parallel heat flux
    parallel_heat_flux = load_variable(group, "parallel_heat_flux")

    # Read charged species thermal speed
    thermal_speed = load_variable(group, "thermal_speed")

    if printout
        println("done.")
    end

    return density, parallel_flow, parallel_pressure, parallel_heat_flux, thermal_speed
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
        print("Loading charged particle distribution function data...")
    end

    group = get_group(fid, "dynamic_data")

    # Read charged distribution function
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
                                 time_index, composition, r, z, vpa, vperp, vzeta, vr, vz)
    code_time = 0.0
    previous_runs_info = nothing
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

            # Test whether any interpolation is needed
            interpolation_needed = Dict(
                x.name => x.n != restart_x.n || !all(isapprox.(x.grid, restart_x.grid))
                for (x, restart_x) ∈ ((z, restart_z), (r, restart_r),
                                      (vperp, restart_vperp), (vpa, restart_vpa),
                                      (vzeta, restart_vzeta), (vr, restart_vr),
                                      (vz, restart_vz)))

            code_time = load_slice(dynamic, "time", time_index)

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

            function load_moment(var_name)
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

            moments.charged.dens .= load_moment("density")
            moments.charged.dens_updated .= true
            moments.charged.upar .= load_moment("parallel_flow")
            moments.charged.upar_updated .= true
            moments.charged.ppar .= load_moment("parallel_pressure")
            moments.charged.ppar_updated .= true
            moments.charged.qpar .= load_moment("parallel_heat_flux")
            moments.charged.qpar_updated .= true
            moments.charged.vth .= load_moment("thermal_speed")

            if "external_source_controller_integral" ∈ get_variable_keys(dynamic) &&
                    length(moments.charged.external_source_controller_integral) == 1
                moments.charged.external_source_controller_integral .=
                    load_slice(dynamic, "external_source_controller_integral", time_index)
            elseif length(moments.charged.external_source_controller_integral) > 1
                moments.charged.external_source_controller_integral .=
                    load_moment("external_source_controller_integral")
            end

            function load_charged_pdf()
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
                        restart_vpa_vals = vpa.grid .- moments.charged.upar[iz,ir,is]
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
                        restart_vpa_vals = vpa.grid ./ moments.charged.vth[iz,ir,is]
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
                            @. (vpa.grid - moments.charged.upar[iz,ir,is]) /
                               moments.charged.vth[iz,ir,is]
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
                        restart_vpa_vals = vpa.grid .+ moments.charged.upar[iz,ir,is]
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
                            @. (vpa.grid + moments.charged.upar[iz,ir,is]) /
                               moments.charged.vth
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
                        restart_vpa_vals = vpa.grid ./ moments.charged.vth
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
                        restart_vpa_vals = vpa.grid .* moments.charged.vth[iz,ir,is]
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
                        restart_vpa_vals = @. vpa.grid * moments.charged.vth[iz,ir,is] -
                                              moments.charged.upar[iz,ir,is]
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
                            @. vpa.grid -
                               moments.charged.upar[iz,ir,is]/moments.charged.vth[iz,ir,is]
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
                        restart_vpa_vals = @. vpa.grid * moments.charged.vth[iz,ir,is] +
                                              moments.charged.upar[iz,ir,is]
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
                        restart_vpa_vals = vpa.grid .* moments.charged.vth[iz,ir,is]
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
                            @. vpa.grid +
                               moments.charged.upar[iz,ir,is] / moments.charged.vth[iz,ir,is]
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
                        this_pdf[:,:,iz,ir,is] ./= moments.charged.dens[iz,ir,is]
                    end
                elseif !moments.evolve_density && restart_evolve_density
                    # Need to unnormalise by density
                    for is ∈ nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n
                        this_pdf[:,:,iz,ir,is] .*= moments.charged.dens[iz,ir,is]
                    end
                end
                if moments.evolve_ppar && !restart_evolve_ppar
                    # Need to normalise by vth
                    for is ∈ nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n
                        this_pdf[:,:,iz,ir,is] .*= moments.charged.vth[iz,ir,is]
                    end
                elseif !moments.evolve_ppar && restart_evolve_ppar
                    # Need to unnormalise by vth
                    for is ∈ nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n
                        this_pdf[:,:,iz,ir,is] ./= moments.charged.vth[iz,ir,is]
                    end
                end

                return this_pdf
            end

            pdf.charged.norm .= load_charged_pdf()

            boundary_distributions_io = get_group(fid, "boundary_distributions")

            function load_charged_boundary_pdf(var_name)
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
                        restart_vpa_vals = vpa.grid .- moments.charged.upar[iz,is]
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
                        restart_vpa_vals = vpa.grid ./ moments.charged.vth[iz,is]
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
                            @. (vpa.grid - moments.charged.upar[iz,is]) /
                               moments.charged.vth[iz,is]
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
                        restart_vpa_vals = vpa.grid .+ moments.charged.upar[iz,is]
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
                            @. (vpa.grid + moments.charged.upar[iz,is]) /
                               moments.charged.vth
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
                        restart_vpa_vals = vpa.grid ./ moments.charged.vth
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
                        restart_vpa_vals = vpa.grid .* moments.charged.vth[iz,is]
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
                        restart_vpa_vals = @. vpa.grid * moments.charged.vth[iz,is] -
                                              moments.charged.upar[iz,is]
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
                               moments.charged.upar[iz,is]/moments.charged.vth[iz,is]
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
                        restart_vpa_vals = @. vpa.grid * moments.charged.vth[iz,is] +
                                              moments.charged.upar[iz,is]
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
                        restart_vpa_vals = vpa.grid .* moments.charged.vth[iz,is]
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
                               moments.charged.upar[iz,is] / moments.charged.vth[iz,is]
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
                        this_pdf[:,:,iz,is] ./= moments.charged.dens[iz,is]
                    end
                elseif !moments.evolve_density && restart_evolve_density
                    # Need to unnormalise by density
                    for is ∈ nspecies, iz ∈ 1:z.n
                        this_pdf[:,:,iz,is] .*= moments.charged.dens[iz,is]
                    end
                end
                if moments.evolve_ppar && !restart_evolve_ppar
                    # Need to normalise by vth
                    for is ∈ nspecies, iz ∈ 1:z.n
                        this_pdf[:,:,iz,is] .*= moments.charged.vth[iz,is]
                    end
                elseif !moments.evolve_ppar && restart_evolve_ppar
                    # Need to unnormalise by vth
                    for is ∈ nspecies, iz ∈ 1:z.n
                        this_pdf[:,:,iz,is] ./= moments.charged.vth[iz,is]
                    end
                end

                return this_pdf
            end

            boundary_distributions.pdf_rboundary_charged[:,:,:,1,:] .=
                load_charged_boundary_pdf("pdf_rboundary_charged_left")
            boundary_distributions.pdf_rboundary_charged[:,:,:,2,:] .=
                load_charged_boundary_pdf("pdf_rboundary_charged_right")

            if composition.n_neutral_species > 0
                moments.neutral.dens .= load_moment("density_neutral")
                moments.neutral.dens_updated .= true
                moments.neutral.uz .= load_moment("uz_neutral")
                moments.neutral.uz_updated .= true
                moments.neutral.pz .= load_moment("pz_neutral")
                moments.neutral.pz_updated .= true
                moments.neutral.qz .= load_moment("qz_neutral")
                moments.neutral.qz_updated .= true
                moments.neutral.vth .= load_moment("thermal_speed_neutral")

                if "external_source_neutral_controller_integral" ∈ get_variable_keys(dynamic) &&
                        length(moments.neutral.external_source_controller_integral) == 1
                    moments.neutral.external_source_controller_integral .=
                        load_slice(dynamic,
                                   "external_source_neutral_controller_integral",
                                   time_index)
                elseif length(moments.neutral.external_source_controller_integral) > 1
                    moments.neutral.external_source_controller_integral .=
                        load_moment("external_source_neutral_controller_integral")
                end

                function load_neutral_pdf()
                    this_pdf = load_slice(dynamic, "f_neutral", vz_range, vr_range,
                                          vzeta_range, z_range, r_range, :, time_index)
                    orig_nvz, orig_nvr, orig_nvzeta, orig_nz, orig_nr, nspecies =
                        size(this_pdf)
                    if interpolation_needed["r"]
                        new_pdf = allocate_float(orig_nvz, orig_nvr, orig_nvzeta, orig_nz,
                                                 r.n, nspecies)
                        for is ∈ 1:nspecies, iz ∈ 1:orig_nz, ivzeta ∈ 1:orig_nnzeta,
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
                        for is ∈ 1:nspecies, ir ∈ 1:r.n, iz ∈ 1:z.n, ivr ∈ 1:orig_ivr,
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
                                ivzeta ∈ 1:orig_ivzeta, ivz ∈ 1:orig_nvz
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
                                   moments.neutral.vth
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
                            restart_vz_vals = vz.grid ./ moments.neutral.vth
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

                pdf.neutral.norm .= load_neutral_pdf()

                function load_neutral_boundary_pdf(var_name)
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
                        for is ∈ 1:nspecies, iz ∈ 1:z.n, ivr ∈ 1:orig_ivr,
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
                        for is ∈ 1:nspecies, iz ∈ 1:z.n, ivzeta ∈ 1:orig_ivzeta,
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
                            restart_vz_vals = vz.grid .- moments.neutral.uz[iz,is]
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
                            restart_vz_vals = vz.grid ./ moments.neutral.vth[iz,is]
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
                                @. (vz.grid - moments.neutral.uz[iz,is]) /
                                   moments.neutral.vth[iz,is]
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
                                              moments.neutral.uz[iz,is]
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
                                @. (vz.grid + moments.neutral.uz[iz,is]) /
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
                            restart_vz_vals = vz.grid ./ moments.neutral.vth
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
                                              moments.neutral.vth[iz,is]
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
                                @. vz.grid * moments.neutral.vth[iz,is] -
                                   moments.neutral.upar[iz,is]
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
                                   moments.neutral.uz[iz,is]/moments.neutral.vth[iz,is]
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
                                @. vz.grid * moments.neutral.vth[iz,is] +
                                   moments.neutral.uz[iz,is]
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
                            restart_vz_vals = vz.grid .* moments.neutral.vth[iz,is]
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
                                   moments.neutral.uz[iz,is]/moments.neutral.vth[iz,is]
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
                            this_pdf[:,:,:,iz,is] ./= moments.neutral.dens[iz,is]
                        end
                    elseif !moments.evolve_density && restart_evolve_density
                        # Need to unnormalise by density
                        for is ∈ nspecies, iz ∈ 1:z.n
                            this_pdf[:,:,:,iz,is] .*= moments.neutral.dens[iz,is]
                        end
                    end
                    if moments.evolve_ppar && !restart_evolve_ppar
                        # Need to normalise by vth
                        for is ∈ nspecies, iz ∈ 1:z.n
                            this_pdf[:,:,:,iz,is] .*= moments.neutral.vth[iz,is]
                        end
                    elseif !moments.evolve_ppar && restart_evolve_ppar
                        # Need to unnormalise by vth
                        for is ∈ nspecies, iz ∈ 1:z.n
                            this_pdf[:,:,:,iz,is] ./= moments.neutral.vth[iz,is]
                        end
                    end

                    return this_pdf
                end

                boundary_distributions.pdf_rboundary_neutral[:,:,:,:,1,:] .=
                    load_neutral_boundary_pdf("pdf_rboundary_neutral_left")
                boundary_distributions.pdf_rboundary_neutral[:,:,:,:,2,:] .=
                    load_neutral_boundary_pdf("pdf_rboundary_neutral_right")
            end
        finally
            close(fid)
        end
    end

    return code_time, previous_runs_info, time_index
end

"""
Read a slice of an ion distribution function

run_names is a tuple. If it has more than one entry, this means that there are multiple
restarts (which are sequential in time), so concatenate the data from each entry together.

The slice to take is specified by the keyword arguments.
"""
function load_distributed_charged_pdf_slice(run_names::Tuple, nblocks::Tuple, t_range,
                                            n_species::mk_int, r::coordinate,
                                            z::coordinate, vperp::coordinate,
                                            vpa::coordinate; is=nothing, ir=nothing,
                                            iz=nothing, ivperp=nothing, ivpa=nothing)
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

            z_irank, r_irank = load_rank_data(fid)

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

            z_irank, r_irank = load_rank_data(fid)

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

end
