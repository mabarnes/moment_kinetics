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
Load saved input settings
"""
function load_input(fid)
    function read_dict(io, section_name)
        # Function that can be called recursively to read nested Dicts from sub-groups in
        # the output file
        section_io = get_group(io, section_name)
        section = Dict{String,Any}()

        for key ∈ get_variable_keys(section_io)
            section[key] = load_variable(section_io, key)
        end
        for key ∈ get_subgroup_keys(section_io)
            section[key] = read_dict(section_io, key)
        end

        return section
    end

    return read_dict(fid, "input")
end

"""
Load data for a coordinate
"""
function load_coordinate_data(fid, name; printout=false)
    if printout
        println("Loading $name coordinate data...")
    end

    coord_group = get_group(get_group(fid, "coords"), name)

    ngrid = load_variable(coord_group, "ngrid")
    n_local = load_variable(coord_group, "n_local")
    n_global = load_variable(coord_group, "n_global")
    grid = load_variable(coord_group, "grid")
    wgts = load_variable(coord_group, "wgts")
    irank = load_variable(coord_group, "irank")
    # L = global box length
    L = load_variable(coord_group, "L")
    discretization = load_variable(coord_group, "discretization")
    fd_option = load_variable(coord_group, "fd_option")
    bc = load_variable(coord_group, "bc")

    nelement_local = nothing
    if n_local == 1 && ngrid == 1
        nelement_local = 1
    else
        nelement_local = (n_local-1) ÷ (ngrid-1)
    end
    if n_global == 1 && ngrid == 1
        nelement_global = 1
    else
        nelement_global = (n_global-1) ÷ (ngrid-1)
    end

    # Define input to create coordinate struct
    # Some dummy inputs, at least for now: nrank=0
    input = grid_input(name, ngrid, nelement_global, nelement_local, 0, irank, L,
                       discretization, fd_option, bc, advection_input("", 0.0, 0.0, 0.0),
                       MPI.COMM_NULL)

    coord, spectral = define_coordinate(input)

    return coord, spectral
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

            if parallel_io
                restart_n_ion_species, restart_n_neutral_species = load_species_data(fid)
                restart_z, _ = load_coordinate_data(fid, "z")
                restart_r, _ = load_coordinate_data(fid, "r")
                restart_vperp, _ = load_coordinate_data(fid, "vperp")
                restart_vpa, _ = load_coordinate_data(fid, "vpa")
                restart_vzeta, _ = load_coordinate_data(fid, "vzeta")
                restart_vr, _ = load_coordinate_data(fid, "vr")
                restart_vz, _ = load_coordinate_data(fid, "vz")
                if (restart_n_ion_species != composition.n_ion_species ||
                    restart_n_neutral_species != composition.n_neutral_species ||
                    restart_z.n != z.n_global || restart_r.n != r.n_global ||
                    restart_vperp.n_global != vperp.n_global ||
                    restart_vpa.n != vpa.n_global || restart_vzeta.n != vzeta.n_global ||
                    restart_vr.n != vr.n_global || restart_vz.n != vz.n_global)

                    error("Dimensions of restart file and input do not match.\n" *
                          "Restart file was n_ion_species=$restart_n_ion_species, " *
                          "n_neutral_species=$restart_n_neutral_species, nr=$(restart_r.n), " *
                          "nz=$(restart_z.n), nvperp=$(restart_vperp.n), nvpa=$(restart_vpa.n).\n" *
                          "nvzeta=$(restart_vzeta.n), nvr=$(restart_vr.n), nvz=$(restart_vz.n)." *
                          "Input file gave n_ion_species=$(composition.n_ion_species), " *
                          "n_neutral_species=$(composition.n_neutral_species), nr=$(r.n), " *
                          "nz=$(z.n), nvperp=$(vperp.n), nvpa=$(vpa.n), nvzeta=$(vzeta.n), " *
                          "nvr=$(vr.n), nvz=$(vz.n).")
                end

                code_time = load_slice(dynamic, "time", time_index)

                function get_range(coord)
                    if coord.irank == coord.nrank - 1
                        return coord.global_io_range
                    else
                        # Need to modify the range to load the end-point that is duplicated on
                        # the next process
                        r = coord.global_io_range
                        return r.start:(r.stop+1)
                    end
                end
                r_range = get_range(r)
                z_range = get_range(z)
                vperp_range = get_range(vperp)
                vpa_range = get_range(vpa)
                vzeta_range = get_range(vzeta)
                vr_range = get_range(vr)
                vz_range = get_range(vz)

                pdf.charged.norm .= load_slice(dynamic, "f", vpa_range, vperp_range,
                                               z_range, r_range, :, time_index)
                moments.charged.dens .= load_slice(dynamic, "density", z_range, r_range,
                                                   :, time_index)
                moments.charged.dens_updated .= true
                moments.charged.upar .= load_slice(dynamic, "parallel_flow", z_range,
                                                   r_range, :, time_index)
                moments.charged.upar_updated .= true
                moments.charged.ppar .= load_slice(dynamic, "parallel_pressure", z_range,
                                                   r_range, :, time_index)
                moments.charged.ppar_updated .= true
                moments.charged.qpar .= load_slice(dynamic, "parallel_heat_flux", z_range,
                                                   r_range, :, time_index)
                moments.charged.qpar_updated .= true
                moments.charged.vth .= load_slice(dynamic, "thermal_speed", z_range,
                                                  r_range, :, time_index)

                boundary_distributions_io = get_group(fid, "boundary_distributions")
                boundary_distributions.pdf_rboundary_charged[:,:,:,1,:] .=
                load_slice(boundary_distributions_io, "pdf_rboundary_charged_left",
                           vpa_range, vperp_range, z_range, :)
                boundary_distributions.pdf_rboundary_charged[:,:,:,2,:] .=
                load_slice(boundary_distributions_io, "pdf_rboundary_charged_right",
                           vpa_range, vperp_range, z_range, :)

                if composition.n_neutral_species > 0
                    pdf.neutral.norm .= load_slice(dynamic, "f_neutral", vz_range,
                                                   vr_range, vzeta_range, z_range,
                                                   r_range, :, time_index)
                    moments.neutral.dens .= load_slice(dynamic, "density_neutral",
                                                       z_range, r_range, :, time_index)
                    moments.neutral.dens_updated .= true
                    moments.neutral.uz .= load_slice(dynamic, "uz_neutral", z_range,
                                                     r_range, :, time_index)
                    moments.neutral.uz_updated .= true
                    moments.neutral.pz .= load_slice(dynamic, "pz_neutral", z_range,
                                                     r_range, :, time_index)
                    moments.neutral.pz_updated .= true
                    moments.neutral.qz .= load_slice(dynamic, "qz_neutral", z_range,
                                                     r_range, :, time_index)
                    moments.neutral.qz_updated .= true
                    moments.neutral.vth .= load_slice(dynamic, "thermal_speed", z_range,
                                                      r_range, :, time_index)

                    boundary_distributions.pdf_rboundary_neutral[:,:,:,:,1,:] .=
                    load_slice(boundary_distributions_io, "pdf_rboundary_neutral_left",
                               vz_range, vr_range, vzeta_range, z_range, :)
                    boundary_distributions.pdf_rboundary_neutral[:,:,:,:,2,:] .=
                    load_slice(boundary_distributions_io, "pdf_rboundary_neutral_right",
                               vz_range, vr_range, vzeta_range, z_range, :)
                end
            else
                restart_n_ion_species, restart_n_neutral_species = load_species_data(fid)
                restart_z, _ = load_coordinate_data(fid, "z")
                restart_r, _ = load_coordinate_data(fid, "r")
                restart_vperp, _ = load_coordinate_data(fid, "vperp")
                restart_vpa, _ = load_coordinate_data(fid, "vpa")
                restart_vzeta, _ = load_coordinate_data(fid, "vzeta")
                restart_vr, _ = load_coordinate_data(fid, "vr")
                restart_vz, _ = load_coordinate_data(fid, "vz")
                if (restart_n_ion_species != composition.n_ion_species ||
                    restart_n_neutral_species != composition.n_neutral_species ||
                    restart_z.n != z.n || restart_r.n != r.n || restart_vperp.n != vperp.n ||
                    restart_vpa.n != vpa.n || restart_vzeta.n != vzeta.n ||
                    restart_vr.n != vr.n || restart_vz.n != vz.n)

                    error("Dimensions of restart file and input do not match.\n" *
                          "Restart file was n_ion_species=$restart_n_ion_species, " *
                          "n_neutral_species=$restart_n_neutral_species, nr=$(restart_r.n), " *
                          "nz=$(restart_z.n), nvperp=$(restart_vperp.n), nvpa=$(restart_vpa.n).\n" *
                          "nvzeta=$(restart_vzeta.n), nvr=$(restart_vr.n), nvz=$(restart_vz.n)." *
                          "Input file gave n_ion_species=$(composition.n_ion_species), " *
                          "n_neutral_species=$(composition.n_neutral_species), nr=$(r.n), " *
                          "nz=$(z.n), nvperp=$(vperp.n), nvpa=$(vpa.n), nvzeta=$(vzeta.n), " *
                          "nvr=$(vr.n), nvz=$(vz.n).")
                end

                code_time = load_slice(dynamic, "time", time_index)

                pdf.charged.norm .= load_slice(dynamic, "f", :, :, :, :, :, time_index)
                moments.charged.dens .= load_slice(dynamic, "density", :, :, :,
                                                   time_index)
                moments.charged.dens_updated .= true
                moments.charged.upar .= load_slice(dynamic, "parallel_flow", :, :, :,
                                                   time_index)
                moments.charged.upar_updated .= true
                moments.charged.ppar .= load_slice(dynamic, "parallel_pressure", :, :, :,
                                                   time_index)
                moments.charged.ppar_updated .= true
                moments.charged.qpar .= load_slice(dynamic, "parallel_heat_flux", :, :, :,
                                                   time_index)
                moments.charged.qpar_updated .= true
                moments.charged.vth .= load_slice(dynamic, "thermal_speed", :, :, :,
                                                  time_index)

                boundary_distributions_io = get_group(fid, "boundary_distributions")
                boundary_distributions.pdf_rboundary_charged[:,:,:,1,:] .=
                load_variable(boundary_distributions_io, "pdf_rboundary_charged_left")
                boundary_distributions.pdf_rboundary_charged[:,:,:,2,:] .=
                load_variable(boundary_distributions_io, "pdf_rboundary_charged_right")

                if composition.n_neutral_species > 0
                    pdf.neutral.norm .= load_slice(dynamic, "f_neutral", :, :, :, :, :, :,
                                                   time_index)
                    moments.neutral.dens .= load_slice(dynamic, "density_neutral", :, :,
                                                       :, time_index)
                    moments.neutral.dens_updated .= true
                    moments.neutral.uz .= load_slice(dynamic, "uz_neutral", :, :, :,
                                                     time_index)
                    moments.neutral.uz_updated .= true
                    moments.neutral.pz .= load_slice(dynamic, "pz_neutral", :, :, :,
                                                     time_index)
                    moments.neutral.pz_updated .= true
                    moments.neutral.qz .= load_slice(dynamic, "qz_neutral", :, :, :,
                                                     time_index)
                    moments.neutral.qz_updated .= true
                    moments.neutral.vth .= load_slice(dynamic, "thermal_speed", :, :, :,
                                                      time_index)

                    boundary_distributions.pdf_rboundary_neutral[:,:,:,:,1,:] .=
                    load_variable(boundary_distributions_io, "pdf_rboundary_neutral_left")
                    boundary_distributions.pdf_rboundary_neutral[:,:,:,:,2,:] .=
                    load_variable(boundary_distributions_io, "pdf_rboundary_neutral_right")
                end
            end
        finally
            close(fid)
        end
    end

    return code_time
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

            # min index set to avoid double assignment of repeated points
            # 1 if irank = 0, 2 otherwise
            imin_r = min(1,r_irank) + 1
            imin_z = min(1,z_irank) + 1
            local_r_range = imin_r:r.n
            local_z_range = imin_z:z.n
            global_r_range = iglobal_func(imin_r, r_irank, r.n):iglobal_func(r.n, r_irank, r.n)
            global_z_range = iglobal_func(imin_z, z_irank, z.n):iglobal_func(z.n, z_irank, z.n)

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
            f_global_slice = selectdim(f_global_slice, thisdim, 1)
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

            # min index set to avoid double assignment of repeated points
            # 1 if irank = 0, 2 otherwise
            imin_r = min(1,r_irank) + 1
            imin_z = min(1,z_irank) + 1
            local_r_range = imin_r:r.n
            local_z_range = imin_z:z.n
            global_r_range = iglobal_func(imin_r, r_irank, r.n):iglobal_func(r.n, r_irank, r.n)
            global_z_range = iglobal_func(imin_z, z_irank, z.n):iglobal_func(z.n, z_irank, z.n)

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
            f_global_slice = selectdim(f_global_slice, thisdim, 1)
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
    elseif irank > 0 && ilocal > 1
        iglobal = ilocal + irank*(nlocal - 1)
    else
        println("ERROR: Invalid call to iglobal_func")
    end
    return iglobal
end

function ilocal_func(iglobal,irank,nlocal)
    return iglobal - irank*(nlocal - 1)
end

end
