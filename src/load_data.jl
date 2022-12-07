"""
"""
module load_data

export open_output_file
export load_coordinate_data
export load_fields_data
export load_charged_particle_moments_data
export load_neutral_particle_moments_data
export load_pdf_data
export load_neutral_pdf_data
export load_neutral_coordinate_data
export load_time_data

using HDF5
using NCDatasets

"""
"""
function open_output_file(run_name, ext; iblock=0)
    # create the HDF5 filename from the given run_name
    # and the shared-memory block index
    hdf5_filename = string(run_name, ".", iblock,".", ext, ".h5")
    if isfile(hdf5_filename)
        print("Opening ", hdf5_filename, " to read HDF5 data...")
        # open the HDF5 file with given filename for reading
        fid = h5open(hdf5_filename, "r")
    else
        # create the netcdf filename from the given run_name
        # and the shared-memory block index
        netcdf_filename = string(run_name, ".", iblock,".", ext,  ".cdf")

        print("Opening ", netcdf_filename, " to read NetCDF data...")
        # open the netcdf file with given filename for reading
        fid = NCDataset(netcdf_filename,"a")
    end
    println("done.")

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
Load data for spatial coordinates and numbers of species

Velocity space coordinate data handled separately as it may not be present in all output
files.
"""
function load_coordinate_data(fid)
    print("Loading coordinate data...")

    coords_group = get_group(fid, "coords")

    # Get r coordinate quantities
    r_group = get_group(coords_group, "r")
    nr = load_variable(r_group, "npts")
    r = load_variable(r_group, "grid")
    r_wgts = load_variable(r_group, "wgts")
    # Lr = r box length
    Lr = r[end]-r[1]
    
    # Get z coordinate quantities
    z_group = get_group(coords_group, "z")
    nz = load_variable(z_group, "npts")
    z = load_variable(z_group, "grid")
    z_wgts = load_variable(z_group, "wgts")
    # Lz = z box length
    Lz = z[end]-z[1]

    overview_group = get_group(fid, "overview")
    n_ion_species = load_variable(overview_group, "n_ion_species")
    n_neutral_species = load_variable(overview_group, "n_neutral_species")
    println("done.")

    return nz, z, z_wgts, Lz, nr, r, r_wgts, Lr, n_ion_species, n_neutral_species
end

"""
Load data for velocity space coordinates
"""
function load_vspace_coordinate_data(fid)
    print("Loading velocity space coordinate data...")

    coords_group = get_group(fid, "coords")

    # Get vperp coordinate quantities
    vperp_group = get_group(coords_group, "vperp")
    nvperp = load_variable(vperp_group, "npts")
    vperp = load_variable(vperp_group, "grid")
    vperp_wgts = load_variable(vperp_group, "wgts")
    # Lvperp = vperp box length
    Lvperp = vperp[end]-vperp[1]

    # Get vpa coordinate quantities
    vpa_group = get_group(coords_group, "vpa")
    nvpa = load_variable(vpa_group, "npts")
    vpa = load_variable(vpa_group, "grid")
    vpa_wgts = load_variable(vpa_group, "wgts")
    # Lvpa = vpa box length
    Lvpa = vpa[end]-vpa[1]
    
    return nvpa, vpa, vpa_wgts, nvperp, vperp, vperp_wgts
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

function load_neutral_coordinate_data(fid)
    print("Loading neutral coordinate data...")

    coords_group = get_group(fid, "coords")

    # Get vz coordinate quantities
    vz_group = get_group(coords_group, "vz")
    nvz = load_variable(vz_group, "npts")
    vz = load_variable(vz_group, "grid")
    vz_wgts = load_variable(vz_group, "wgts")
    # Lvz = vz box length
    Lvz = vz[end]-vz[1]

    # Get vr coordinate quantities
    vr_group = get_group(coords_group, "vr")
    nvr = load_variable(vr_group, "npts")
    vr = load_variable(vr_group, "grid")
    vr_wgts = load_variable(vr_group, "wgts")
    # Lvr = vr box length
    Lvr = vr[end]-vr[1]

    # Get vzeta coordinate quantities
    vzeta_group = get_group(coords_group, "vzeta")
    nvzeta = load_variable(vzeta_group, "npts")
    vzeta = load_variable(vzeta_group, "grid")
    vzeta_wgts = load_variable(vzeta_group, "wgts")
    # Lvzeta = vzeta box length
    Lvzeta = vzeta[end]-vzeta[1]

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
