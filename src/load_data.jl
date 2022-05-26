"""
"""
module load_data

export open_netcdf_file
export load_coordinate_data
export load_fields_data
export load_moments_data
export load_pdf_data

using ..coordinates: define_coordinate
using ..input_structs: grid_input

using NCDatasets

"""
"""
function open_netcdf_file(run_name)
    # create the netcdf filename from the given run_name
    filename = string(run_name, ".cdf")

    print("Opening ", filename, " to read NetCDF data...")
    # open the netcdf file with given filename for reading
    fid = NCDataset(filename,"a")
    println("done.")

    return fid
end

"""
"""
function load_coordinate_data(fid)
    print("Loading coordinate data...")

    function load_coordinate(name::String)
        # define a handle for the coordinate
        cdfvar = fid[name]
        # get the number of grid points
        n = length(cdfvar)
        # load the data for grid point positions
        grid = cdfvar.var[:]
        # get the weights associated with the coordinate
        cdfvar = fid["$(name)_wgts"]
        wgts = cdfvar.var[:]
        # Lr = box length for coordinate
        L = grid[end]-grid[1]

        cdfvar = fid["$(name)_ngrid"]
        ngrid = cdfvar.var[:]

        cdfvar = fid["$(name)_nelement"]
        nelement = cdfvar.var[:]

        cdfvar = fid["$(name)_discretization"]
        discretization = cdfvar.var[:]

        cdfvar = fid["$(name)_fd_option"]
        fd_option = cdfvar.var[:]

        cdfvar = fid["$(name)_bc"]
        bc = cdfvar.var[:]

        input = grid_input(name, ngrid, nelement, L, discretization, fd_option, bc, nothing)

        coord = define_coordinate(input)

        # grid is recreated in define_coordinate, so check it is consistent with the
        # saved grid positions
        @assert isapprox(grid, coord.grid, rtol=1.e-14)

        return coord
    end

    r = load_coordinate("r")
    z = load_coordinate("z")
    vperp = load_coordinate("vperp")
    vpa = load_coordinate("vpa")

    # define a handle for the time coordinate
    cdfvar = fid["time"]
    # get the number of time grid points
    ntime = length(cdfvar)
    # load the data for time
    time = cdfvar.var[:]
    println("done.")

    return vpa, vperp, z, r, ntime, time
end

"""
"""
function load_fields_data(fid)
    print("Loading fields data...")
    # define a handle for the electrostatic potential
    cdfvar = fid["phi"]
    # load the electrostatic potential data
    phi = cdfvar.var[:,:,:]
    println("done.")
    return phi
end

"""
"""
function load_moments_data(fid)
    print("Loading velocity moments data...")
    # define a handle for the species density
    cdfvar = fid["density"]
    # load the species density data
    density = cdfvar.var[:,:,:,:]
    # define a handle for the species parallel flow
    cdfvar = fid["parallel_flow"]
    # load the species parallel flow data
    parallel_flow = cdfvar.var[:,:,:,:]
    # define a handle for the species parallel pressure
    cdfvar = fid["parallel_pressure"]
    # load the species parallel pressure data
    parallel_pressure = cdfvar.var[:,:,:,:]
    # define a handle for the species parallel heat flux
    cdfvar = fid["parallel_heat_flux"]
    # load the species parallel heat flux data
    parallel_heat_flux = cdfvar.var[:,:,:,:]
    # define a handle for the species thermal speed
    cdfvar = fid["thermal_speed"]
    # load the species thermal speed data
    thermal_speed = cdfvar.var[:,:,:,:]
    # define the number of species
    n_species = size(cdfvar,3)
    # define a handle for the flag indicating if the parallel pressure should be separately advanced
    cdfvar = fid["evolve_ppar"]
    # load the parallel pressure evolution flag
    evolve_ppar_int = cdfvar.var[:]
    if evolve_ppar_int[1] == 1
        evolve_ppar = true
    else
        evolve_ppar = false
    end
    println("done.")
    return density, parallel_flow, parallel_pressure, parallel_heat_flux, thermal_speed, n_species, evolve_ppar
end

"""
"""
function load_pdf_data(fid)
    print("Loading distribution function data...")
    # define a handle for the distribution function
    cdfvar = fid["f"]
    # load the distribution function data
    pdf = cdfvar.var[:,:,:,:,:,:]
    println("done.")
    return pdf
end

end
