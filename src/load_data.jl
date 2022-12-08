"""
"""
module load_data

export open_readonly_output_file
export load_fields_data
export load_charged_particle_moments_data
export load_neutral_particle_moments_data
export load_pdf_data
export load_neutral_pdf_data
export load_neutral_velocity_coordinate_data
export load_charged_velocity_coordinate_data
export load_time_data
export load_block_data
export load_rank_data
export load_global_zr_coordinate_data
export load_local_zr_coordinate_data
export load_species_data

using HDF5
using NCDatasets

"""
"""
function open_readonly_output_file(run_name, ext; iblock=0, printout=true)
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
            print("Opening ", filename, " to read NetCDF data...")
        end
        # open the netcdf file with given filename for reading
        fid = NCDataset(netcdf_filename,"a")
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
    return read(file_or_group[name])
end
function load_variable(file_or_group::NCDataset, name::String)
    # This overload deals with cases where fid is a NetCDF `Dataset` (which could be a
    # file or a group).
    return file_or_group[name].var[:]
end

"""
Get a (sub-)group from a file or group
"""
function get_group() end
function get_group(file_or_group::HDF5.H5DataStore, name::String)
    # This overload deals with cases where fid is an HDF5 `File` or `Group` (`H5DataStore`
    # is the abstract super-type for both
    return file_or_group[name]
end
function get_group(file_or_group::NCDataset, name::String)
    # This overload deals with cases where fid is a NetCDF `Dataset` (which could be a
    # file or a group).
    return file_or_group.group[name]
end

"""
Load data for spatial coordinates

Velocity space coordinate data handled separately as it may not be present in all output
files.
"""
function load_local_zr_coordinate_data(fid; printout=false)
    if printout
        print("Loading coordinate data...")
    end

    coords_group = get_group(fid, "coords")

    # Get r coordinate quantities
    r_group = get_group(coords_group, "r")
    nr = load_variable(r_group, "npts")
    r = load_variable(r_group, "grid")
    r_wgts = load_variable(r_group, "wgts")
    # Lr = global r box length
    Lr = load_variable(r_group, "L")

    # Get z coordinate quantities
    z_group = get_group(coords_group, "z")
    nz = load_variable(z_group, "npts")
    z = load_variable(z_group, "grid")
    z_wgts = load_variable(z_group, "wgts")
    # Lz = global z box length
    Lz = load_variable(z_group, "L")

    if printout
        println("done.")
    end
    return nz, z, z_wgts, Lz, nr, r, r_wgts, Lr
end

function load_charged_velocity_coordinate_data(fid)
    print("Loading coordinate data...")
    # define a handle for the vperp coordinate
    cdfvar = fid["vperp"]
    # get the number of vperp grid points
    nvperp = length(cdfvar)
    # load the data for vperp
    vperp = cdfvar.var[:]
    # get the weights associated with the vperp coordinate
    cdfvar = fid["vperp_wgts"]
    vperp_wgts = cdfvar.var[:]

    # define a handle for the vpa coordinate
    cdfvar = fid["vpa"]
    # get the number of vpa grid points
    nvpa = length(cdfvar)
    # load the data for vpa
    vpa = cdfvar.var[:]
    # get the weights associated with the vpa coordinate
    cdfvar = fid["vpa_wgts"]
    vpa_wgts = cdfvar.var[:]
    println("done.")

    return nvpa, vpa, vpa_wgts, nvperp, vperp, vperp_wgts
end

"""
"""
function load_species_data(fid)
    print("Loading species data...")
    # get the lengths of the ion species dimension
    n_ion_species = fid.dim["n_ion_species"]
    # get the lengths of the neutral species dimension
    n_neutral_species = fid.dim["n_neutral_species"]
    println("done.")
    return n_ion_species, n_neutral_species
end

"""
"""
function load_time_data(fid)
    print("Loading time data...")

    group = get_group(fid, "dynamic_data")
    time = load_variable(group, "time")
    ntime = length(time)

    println("done.")

    return  ntime, time
end

"""
"""
function load_block_data(fid)
    print("Loading block data...")
    cdfvar = fid["nblocks"]
    nblocks = cdfvar.var[]
    
    cdfvar = fid["iblock"]
    iblock = cdfvar.var[]
    println("done.")
    return  nblocks, iblock
end

"""
"""
function load_rank_data(fid; printout=true)
    if printout
        print("Loading rank data...")
    end
    cdfvar = fid["z_irank"]
    z_irank = cdfvar.var[]
    
    cdfvar = fid["r_irank"]
    r_irank = cdfvar.var[]
    if printout
        println("done.")
    end
    return z_irank, r_irank
end

function load_global_zr_coordinate_data(fid)
    print("Loading process data...")
    cdfvar = fid["nz_global"]
    nz_global = cdfvar.var[]
    
    cdfvar = fid["nr_global"]
    nr_global = cdfvar.var[]
    println("done.")
    return  nz_global, nr_global
end

"""
"""

function load_neutral_velocity_coordinate_data(fid)
    print("Loading neutral coordinate data...")
    # define a handle for the vz coordinate
    cdfvar = fid["vz"]
    # get the number of vz grid points
    nvz = length(cdfvar)
    # load the data for vz
    vz = cdfvar.var[:]
    # get the weights associated with the vz coordinate
    cdfvar = fid["vz_wgts"]
    vz_wgts = cdfvar.var[:]
    
    # define a handle for the vr coordinate
    cdfvar = fid["vr"]
    # get the number of vr grid points
    nvr = length(cdfvar)
    # load the data for vr
    vr = cdfvar.var[:]
    # get the weights associated with the vr coordinate
    cdfvar = fid["vr_wgts"]
    vr_wgts = cdfvar.var[:]

    # define a handle for the vzeta coordinate
    cdfvar = fid["vzeta"]
    # get the number of vzeta grid points
    nvzeta = length(cdfvar)
    # load the data for vzeta
    vzeta = cdfvar.var[:]
    # get the weights associated with the vzeta coordinate
    cdfvar = fid["vzeta_wgts"]
    vzeta_wgts = cdfvar.var[:]

    println("done.")

    return nvz, vz, vz_wgts, nvr, vr, vr_wgts, nvzeta, vzeta, vzeta_wgts
end
"""
"""
function load_fields_data(fid)
    print("Loading fields data...")

    group = get_group(fid, "dynamic_data")

    # Read electrostatic potential
    phi = load_variable(group, "phi")

    # Read radial electric field
    Er = load_variable(group, "Er")

    # Read z electric field
    Ez = load_variable(group, "Ez")

    println("done.")

    return phi, Er, Ez
end

"""
"""
function load_charged_particle_moments_data(fid)
    print("Loading charged particle velocity moments data...")

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

    println("done.")

    return density, parallel_flow, parallel_pressure, parallel_heat_flux, thermal_speed, evolve_ppar
end

function load_neutral_particle_moments_data(fid)
    print("Loading neutral particle velocity moments data...")

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

    println("done.")

    return neutral_density, neutral_uz, neutral_pz, neutral_qz, neutral_thermal_speed
end

"""
"""
function load_pdf_data(fid)
    print("Loading charged particle distribution function data...")

    group = get_group(fid, "dynamic_data")

    # Read charged distribution function
    pdf = load_variable(group, "f")

    println("done.")

    return pdf
end
"""
"""
function load_neutral_pdf_data(fid)
    print("Loading neutral particle distribution function data...")

    group = get_group(fid, "dynamic_data")

    # Read neutral distribution function
    neutral_pdf = load_variable(group, "f_neutral")

    println("done.")

    return neutral_pdf
end

end
