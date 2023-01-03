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

using HDF5
using NCDatasets

"""
"""
function open_readonly_output_file(run_name, ext; iblock=0, printout=false)
    # create the HDF5 filename from the given run_name
    # and the shared-memory block index
    hdf5_filename = string(run_name, ".", iblock,".", ext, ".h5")
    if isfile(hdf5_filename)
        if printout
            print("Opening ", hdf5_filename, " to read HDF5 data...")
        end
        # open the HDF5 file with given filename for reading
        fid = h5open(hdf5_filename, "r")
    else
        # create the netcdf filename from the given run_name
        # and the shared-memory block index
        netcdf_filename = string(run_name, ".", iblock,".", ext,  ".cdf")

        if printout
            print("Opening ", netcdf_filename, " to read NetCDF data...")
        end
        # open the netcdf file with given filename for reading
        fid = NCDataset(netcdf_filename, "r")
    end
    if printout
        println("done.")
    end
    return fid
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
        return file_or_group[name].var[:]
    catch
        println("An error occured while loading $name")
        rethrow()
    end
end

"""
Get a (sub-)group from a file or group
"""
function get_group() end
function get_group(file_or_group::HDF5.H5DataStore, name::String)
    # This overload deals with cases where fid is an HDF5 `File` or `Group` (`H5DataStore`
    # is the abstract super-type for both
    try
        return file_or_group[name]
    catch
        println("An error occured while opening the $name group")
        rethrow()
    end
end
function get_group(file_or_group::NCDataset, name::String)
    # This overload deals with cases where fid is a NetCDF `Dataset` (which could be a
    # file or a group).
    try
        return file_or_group.group[name]
    catch
        println("An error occured while opening the $name group")
        rethrow()
    end
end

"""
Load data for a coordinate
"""
function load_coordinate_data(fid, name; printout=false)
    if printout
        println("Loading $name coordinate data...")
    end

    coord_group = get_group(get_group(fid, "coords"), name)

    n_local = load_variable(coord_group, "n_local")
    n_global = load_variable(coord_group, "n_global")
    grid = load_variable(coord_group, "grid")
    wgts = load_variable(coord_group, "wgts")
    # L = global box length
    L = load_variable(coord_group, "L")

    return n_local, n_global, grid, wgts, L
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
function load_time_data(fid; printout=false)
    if printout
        print("Loading time data...")
    end

    group = get_group(fid, "dynamic_data")
    time = load_variable(group, "time")
    ntime = length(time)

    if printout
        println("done.")
    end

    return  ntime, time
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
    
    evolve_ppar = false

    if printout
        println("done.")
    end

    return density, parallel_flow, parallel_pressure, parallel_heat_flux, thermal_speed, evolve_ppar
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

end
