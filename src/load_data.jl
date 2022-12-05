"""
"""
module load_data

export open_netcdf_file
export load_coordinate_data
export load_fields_data
export load_charged_particle_moments_data
export load_neutral_particle_moments_data
export load_pdf_data
export load_neutral_pdf_data
export load_neutral_coordinate_data
export load_time_data

using NCDatasets

"""
"""
function open_netcdf_file(run_name, ext; iblock=0)
    # create the netcdf filename from the given run_name
    # and the shared-memory block index
    filename = string(run_name, ".", iblock,".", ext,  ".cdf")

    print("Opening ", filename, " to read NetCDF data...")
    # open the netcdf file with given filename for reading
    fid = NCDataset(filename,"a")
    println("done.")

    #dimnames = keys(fid.dim)
    #println(dimnames)
    #println(fid.dim["n_neutral_species"])

    return fid
end

"""
"""
function load_coordinate_data(fid)
    print("Loading coordinate data...")
    # define a handle for the r coordinate
    cdfvar = fid["r"]
    # get the number of r grid points
    nr = length(cdfvar)
    # load the data for r
    r = cdfvar.var[:]
    # get the weights associated with the r coordinate
    cdfvar = fid["r_wgts"]
    r_wgts = cdfvar.var[:]
    # Lr = r box length
    Lr = r[end]-r[1]
    
    # define a handle for the z coordinate
    cdfvar = fid["z"]
    # get the number of z grid points
    nz = length(cdfvar)
    # load the data for z
    z = cdfvar.var[:]
    # get the weights associated with the z coordinate
    cdfvar = fid["z_wgts"]
    z_wgts = cdfvar.var[:]
    # Lz = z box length
    Lz = z[end]-z[1]

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
    
    # get the lengths of the ion species dimension
    n_ion_species = fid.dim["n_ion_species"]
    # get the lengths of the neutral species dimension
    n_neutral_species = fid.dim["n_neutral_species"]
    println("done.")

    return nvpa, vpa, vpa_wgts, nvperp, vperp, vperp_wgts, nz, z, z_wgts, Lz, nr, r, r_wgts, Lr, n_ion_species, n_neutral_species
end

"""
"""
function load_time_data(fid)
    print("Loading time data...")
    # define a handle for the time coordinate
    cdfvar = fid["time"]
    # get the number of time grid points
    ntime = length(cdfvar)
    # load the data for time
    time = cdfvar.var[:]
    println("done.")

    return  ntime, time
end

function load_neutral_coordinate_data(fid)
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
    # define a handle for the electrostatic potential
    cdfvar = fid["phi"]
    # load the electrostatic potential data
    phi = cdfvar.var[:,:,:]
    # define a handle for the radial electric field
    cdfvar = fid["Er"]
    # load the radial electric field data
    Er = cdfvar.var[:,:,:]
    # define a handle for the z electric field
    cdfvar = fid["Ez"]
    # load the z electric field data
    Ez = cdfvar.var[:,:,:]
    println("done.")
    return phi, Er, Ez
end

"""
"""
function load_charged_particle_moments_data(fid)
    print("Loading charged particle velocity moments data...")
    # define a handle for the charged species density
    cdfvar = fid["density"]
    # load the charged species density data
    density = cdfvar.var[:,:,:,:]
    # define a handle for the charged species parallel flow
    cdfvar = fid["parallel_flow"]
    # load the charged species parallel flow data
    parallel_flow = cdfvar.var[:,:,:,:]
    # define a handle for the charged species parallel pressure
    cdfvar = fid["parallel_pressure"]
    # load the charged species parallel pressure data
    parallel_pressure = cdfvar.var[:,:,:,:]
    # define a handle for the charged species parallel heat flux
    cdfvar = fid["parallel_heat_flux"]
    # load the charged species parallel heat flux data
    parallel_heat_flux = cdfvar.var[:,:,:,:]
    # define a handle for the charged species thermal speed
    cdfvar = fid["thermal_speed"]
    # load the charged species thermal speed data
    thermal_speed = cdfvar.var[:,:,:,:]
    
    evolve_ppar = false
    println("done.")
    return density, parallel_flow, parallel_pressure, parallel_heat_flux, thermal_speed, evolve_ppar
end

function load_neutral_particle_moments_data(fid)
    print("Loading neutral particle velocity moments data...")
    cdfvar = fid["density_neutral"]
    # load the neutral species density data
    neutral_density = cdfvar.var[:,:,:,:]
    cdfvar = fid["uz_neutral"]
    # load the neutral species uz data
    neutral_uz = cdfvar.var[:,:,:,:]
    cdfvar = fid["pz_neutral"]
    # load the neutral species pz data
    neutral_pz = cdfvar.var[:,:,:,:]
    cdfvar = fid["qz_neutral"]
    # load the neutral species qz data
    neutral_qz = cdfvar.var[:,:,:,:]
    cdfvar = fid["thermal_speed_neutral"]
    # load the neutral species thermal_speed data
    neutral_thermal_speed = cdfvar.var[:,:,:,:]
    println("done.")
    return neutral_density, neutral_uz, neutral_pz, neutral_qz, neutral_thermal_speed
end

"""
"""
function load_pdf_data(fid)
    print("Loading charged particle distribution function data...")
    # define a handle for the distribution function
    cdfvar = fid["f"]
    # load the distribution function data
    pdf = cdfvar.var[:,:,:,:,:,:]
    println("done.")
    return pdf
end
"""
"""
function load_neutral_pdf_data(fid)
    print("Loading neutral particle distribution function data...")
    # define a handle for the distribution function
    cdfvar = fid["f_neutral"]
    # load the distribution function data
    neutral_pdf = cdfvar.var[:,:,:,:,:,:]
    println("done.")
    return neutral_pdf
end

end
