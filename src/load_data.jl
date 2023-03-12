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

using ..coordinates: define_coordinate
using ..file_io: get_group
using ..input_structs: advection_input, grid_input

using HDF5
using MPI
using NCDatasets

"""
"""
function open_readonly_output_file(run_name, ext; iblock=0, printout=false)
    possible_names = (
        string(run_name, ".", ext, ".h5"),
        string(run_name, ".", iblock,".", ext, ".h5"),
        string(run_name, ".", ext, ".cdf"),
        string(run_name, ".", iblock,".", ext, ".cdf"),
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
