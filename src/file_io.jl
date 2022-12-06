"""
"""
module file_io

export input_option_error
export open_output_file
export setup_file_io, finish_file_io
export write_data_to_ascii
export write_data_to_netcdf, write_data_to_hdf5

using HDF5
using NCDatasets
using ..communication: _block_synchronize, iblock_index
using ..debugging
using ..looping
using ..moment_kinetics_structs: scratch_pdf, em_fields_struct
using ..type_definitions: mk_float, mk_int

"""
structure containing the various input/output streams
"""
struct ascii_ios{T <: Union{IOStream,Nothing}}
    # corresponds to the ascii file to which the distribution function is written
    #ff::T
    # corresponds to the ascii file to which velocity space moments of the
    # distribution function such as density and pressure are written
    moments_charged::T
    moments_neutral::T
    # corresponds to the ascii file to which electromagnetic fields
    # such as the electrostatic potential are written
    fields::T
end

"""
structure containing the data/metadata needed for netcdf file i/o
moments & fields only 
"""
struct netcdf_moments_info{Ttime, Tphi, Tmomi, Tmomn}
    # file identifier for the netcdf file to which data is written
    fid::NCDataset
    # handle for the time variable
    time::Ttime
    # handle for the electrostatic potential variable
    phi::Tphi
    # handle for the radial electric field variable
    Er::Tphi
    # handle for the z electric field variable
    Ez::Tphi
    # handle for the charged species density
    density::Tmomi
    # handle for the charged species parallel flow
    parallel_flow::Tmomi
    # handle for the charged species parallel pressure
    parallel_pressure::Tmomi
    # handle for the charged species parallel heat flux
    parallel_heat_flux::Tmomi
    # handle for the charged species thermal speed
    thermal_speed::Tmomi
    
    # handle for the neutral species density
    density_neutral::Tmomn
    uz_neutral::Tmomn
    pz_neutral::Tmomn
    qz_neutral::Tmomn
    thermal_speed_neutral::Tmomn

end

"""
structure containing the data/metadata needed for netcdf file i/o
distribution function data only 
"""
struct netcdf_dfns_info{Ttime, Tfi, Tfn}
    # file identifier for the netcdf file to which data is written
    fid::NCDataset
    # handle for the time variable
    time::Ttime
    # handle for the charged species distribution function variable
    f::Tfi
    # handle for the neutral species distribution function variable
    f_neutral::Tfn
    
end

"""
structure containing the data/metadata needed for hdf5 file i/o
moments & fields only
"""
struct hdf5_moments_info{Ttime, Tphi, Tmomi, Tmomn}
     # file identifier for the netcdf file to which data is written
    fid::HDF5.File
    # handle for the time variable
    time::Ttime
    # handle for the electrostatic potential variable
    phi::Tphi
    # handle for the radial electric field variable
    Er::Tphi
    # handle for the z electric field variable
    Ez::Tphi
    # handle for the charged species density
    density::Tmomi
    # handle for the charged species parallel flow
    parallel_flow::Tmomi
    # handle for the charged species parallel pressure
    parallel_pressure::Tmomi
    # handle for the charged species parallel heat flux
    parallel_heat_flux::Tmomi
    # handle for the charged species thermal speed
    thermal_speed::Tmomi

    # handle for the neutral species density
    density_neutral::Tmomn
    uz_neutral::Tmomn
    pz_neutral::Tmomn
    qz_neutral::Tmomn
    thermal_speed_neutral::Tmomn
 end

"""
structure containing the data/metadata needed for hdf5 file i/o
distribution function data only
"""
struct hdf5_dfns_info{Ttime, Tfi, Tfn}
     # file identifier for the netcdf file to which data is written
    fid::HDF5.File
    # handle for the time variable
    time::Ttime
    # handle for the charged species distribution function variable
    f::Tfi
    # handle for the neutral species distribution function variable
    f_neutral::Tfn
 end

"""
open the necessary output files
"""
function setup_file_io(io_input, vz, vr, vzeta, vpa, vperp, z, r, composition, collisions)
    begin_serial_region()
    @serial_region begin
        # Only read/write from first process in each 'block'

        # check to see if output_dir exists in the current directory
        # if not, create it
        isdir(io_input.output_dir) || mkdir(io_input.output_dir)
        out_prefix = string(io_input.output_dir, "/", io_input.run_name, ".", iblock_index[])
        if io_input.ascii_output
            #ff_io = open_output_file(out_prefix, "f_vs_t")
            mom_chrg_io = open_output_file(out_prefix, "moments_charged_vs_t")
            mom_ntrl_io = open_output_file(out_prefix, "moments_neutral_vs_t")
            fields_io = open_output_file(out_prefix, "fields_vs_t")
            ascii = ascii_ios(mom_chrg_io, mom_ntrl_io, fields_io)
        else
            ascii = ascii_ios(nothing, nothing, nothing)
        end
        cdf_moments = setup_moments_netcdf_io(out_prefix, r, z, vperp, vpa, vzeta, vr, vz, composition, collisions)
        cdf_dfns = setup_dfns_netcdf_io(out_prefix, r, z, vperp, vpa, vzeta, vr, vz, composition, collisions)
        h5_moments = setup_moments_hdf5_io(out_prefix, r, z, vperp, vpa, vzeta, vr, vz, composition, collisions)
        h5_dfns = setup_dfns_hdf5_io(out_prefix, r, z, vperp, vpa, vzeta, vr, vz, composition, collisions)
        return ascii, cdf_moments, cdf_dfns, h5_moments, hy_dfns
    end
    # For other processes in the block, return (nothing, nothing, nothing)
    return nothing, nothing, nothing
end

# Define the steps for creating a NetCDF file in utility functions so that they can be
# shared between `setup_netcdf_io()` and `debug_dump()`
"""
    define_dimensions!(fid, nvz, nvr, nvzeta, nvpa, nvperp, nz, nr, n_species, n_ion_species=nothing,
                       n_neutral_species=nothing)

Define dimensions for an output file.
"""
function define_dimensions!(fid, nvz, nvr, nvzeta, nvpa, nvperp, nz, nr, n_species,
            n_ion_species=nothing, n_neutral_species=nothing)
    # define the vz dimension
    defDim(fid, "nvz", nvz)
    # define the vr dimension
    defDim(fid, "nvr", nvr)
    # define the vzeta dimension
    defDim(fid, "nvzeta", nvzeta)
    # define the vpa dimension
    defDim(fid, "nvpa", nvpa)
    # define the vperp dimension
    defDim(fid, "nvperp", nvperp)
    # define the z dimension
    defDim(fid, "nz", nz)
    # define the r dimension
    defDim(fid, "nr", nr)
    # define the species dimension
    defDim(fid, "n_species", n_species)
    if n_ion_species !== nothing
        # define the ion species dimension
        defDim(fid, "n_ion_species", n_ion_species)
    end
    if n_neutral_species !== nothing
        # define the neutral species dimension
        defDim(fid, "n_neutral_species", n_neutral_species)
    end
    # define the time dimension, with an expandable size (denoted by Inf)
    defDim(fid, "ntime", Inf)

    return nothing
end

"""
write some overview information for the simulation to the hdf5 file
"""
function write_overview_hdf5!(fid, composition, collisions)
    overview = create_group(fid, "overview")
    overview["nspecies"] = composition.n_species
    attributes(overview)["nspecies"] = "total number of evolved plasma species"
    overview["n_ion_species"] = composition.n_ion_species
    attributes(overview)["n_ion_species"] = "number of evolved ion species"
    overview["n_neutral_species"] = composition.n_neutral_species
    attributes(overview)["n_neutral_species"] = "number of evolved neutral species"
    overview["T_e"] = composition.T_e
    attributes(overview)["T_e"] = "fixed electron temperature"
    overview["charge_exchange_frequency"] = collisions.charge_exchange
    attributes(overview)["charge_exchange_frequency"] = "quantity related to the charge exchange frequency"
end

"""
Define coords group for coordinate information in the hdf5 file and write information
about spatial coordinate grids
"""
function define_spatial_coordinates_hdf5!(fid, z, r)
    # create the "coords" group that will contain coordinate information
    coords = create_group(fid, "coords")
    # create the "z" sub-group of "coords" that will contain z coordinate info,
    # including total number of grid points and grid point locations
    z_h5 = define_coordinate_hdf5!(coords, z, "z", "spatial coordinate z")
    # create the "r" sub-group of "coords" that will contain r coordinate info,
    # including total number of grid points and grid point locations
    r_h5 = define_coordinate_hdf5!(coords, r, "r", "spatial coordinate r")

    return coords
end

"""
Add to coords group in hdf5 file information about vspace coordinate grids
"""
function add_vspace_coordinates_hdf5!(coords, vz, vr, vzeta, vpa, vperp)
    # create the "vz" sub-group of "coords" that will contain vz coordinate info,
    # including total number of grid points and grid point locations
    vz_h5 = define_coordinate_hdf5!(coords, vz, "vz", "velocity coordinate v_z")
    # create the "vr" sub-group of "coords" that will contain vr coordinate info,
    # including total number of grid points and grid point locations
    vr_h5 = define_coordinate_hdf5!(coords, vr, "vr", "velocity coordinate v_r")
    # create the "vzeta" sub-group of "coords" that will contain vzeta coordinate info,
    # including total number of grid points and grid point locations
    vzeta_h5 = define_coordinate_hdf5!(coords, vzeta, "vzeta", "velocity coordinate v_zeta")
    # create the "vpa" sub-group of "coords" that will contain vpa coordinate info,
    # including total number of grid points and grid point locations
    vpa_h5 = define_coordinate_hdf5!(coords, vpa, "vpa", "velocity coordinate v_parallel")
    # create the "vperp" sub-group of "coords" that will contain vperp coordinate info,
    # including total number of grid points and grid point locations
    vperp_h5 = define_coordinate_hdf5!(coords, vperp, "vperp", "velocity coordinate v_perp")

    return nothing
end

"""
define a sub-group for each code coordinate and write to hdf5 file
"""
function define_coordinate_hdf5!(parent, coord, coord_name, descriptor)
    # create the "group" sub-group of "parent" that will contain coord_str coordinate info
    group = create_group(parent, coord_name)
    attributes(parent)[coord_name] = descriptor
    # write the number of grid points for this coordinate to variable "npts" within "coords/coord_name" group
    group["npts"] = coord.n
    attributes(group)["npts"] = string("total number of ", coord_name, " grid points")
    # write the locations of this coordinate's grid points to variable "grid" within "coords/coord_name" group
    group["grid"] = coord.grid
    attributes(group)["grid"] = string(coord_name, " values sampled by the ", coord_name, " grid")
    # write the integration weights attached to each coordinate grid point
    group["wgts"] = coord.wgts
    attributes(group)["wgts"] = string("integration weights associated with the ", coord_name, " grid points")
    return group
end

"""
given a tuple, reduced_dims, containing all dimensions except the time dimension,
return chunk_dims tuple that indicates the data chunk written to hdf5 file each write
and the dims tuple which also contains the max size of the dataset, accounting for multiple
time slices
"""
function hdf5_dynamic_dims(reduced_dims)
    # chunk_dims is a tuple indicating the data chunk size to be written each step
    chunk_dims = (reduced_dims..., 1)
    # dims contains the initial allocated data size in chunk_dims and the maximum
    # data size in the second argument; the -1 indicates that the time index is
    # effectively unlimited (as large as the largest unsigned integer value).
    # the time index will be dynamically extended as more data is written to file
    dims = (chunk_dims, (reduced_dims...,-1))

    return chunk_dims, dims
end

"""
define dynamic (time-evolving) moment variables for writing to the hdf5 file
"""
function define_dynamic_moment_variables_hdf5!(fid, nz, nr, n_ion_species,
                                               n_neutral_species)
    dynamic = create_group(fid, "dynamic_data")
    # create the time variable initially to have one element but allow it to be expanded
    # indefinitely (up to the largest unsigned integer in size)
    h5_time = create_dataset(dynamic, "time", mk_float, ((1,),(-1,)), chunk=(1,))

    # reduced_dims is a tuple containing all dimensions for the relevant data aside from time
    reduced_dims = (nz, nr)
    # given the tuple reduced_dims that contains all dimensions except the time dimension,
    # return chunk_dims tuple that indicates the data chunk written to hdf5 file each write
    # and the dims tuple which also contains the max size of the dataset, accounting for
    # multiple time slices
    chunk_dims, dims = hdf5_dynamic_dims(reduced_dims)
    # # chunk_dims is a tuple indicating the data chunk size to be written each step
    # chunk_dims = (reduced_dims..., 1)
    # # dims contains the initial allocated data size in chunk_dims and the maximum
    # # data size in the second argument; the -1 indicates that the time index is
    # # effectively unlimited (as large as the largest unsigned integer value).
    # # the time index will be dynamically extended as more data is written to file
    # dims = (chunk_dims, (reduced_dims...,-1))
    # h5_phi is the handle referring to the electrostatic potential phi
    h5_phi = create_dataset(dynamic, "phi", mk_float, dims, chunk=chunk_dims)
    # h5_Er is the handle for the radial component of the electric field
    h5_Er = create_dataset(dynamic, "Er", mk_float, dims, chunk=chunk_dims)
    # h5_Ez is the handle for the zed component of the electric field
    h5_Ez = create_dataset(dynamic, "Ez", mk_float, dims, chunk=chunk_dims)

    # reduced_dims is a tuple containing all dimensions for the relevant data aside from time
    reduced_dims = (nz, nr, n_ion_species)
    # given the tuple reduced_dims that contains all dimensions except the time dimension,
    # return chunk_dims tuple that indicates the data chunk written to hdf5 file each write
    # and the dims tuple which also contains the max size of the dataset, accounting for
    # multiple time slices
    chunk_dims, dims = hdf5_dynamic_dims(reduced_dims)
    # h5_density is the handle for the ion particle density
    h5_density = create_dataset(dynamic, "density", mk_float, dims, chunk=chunk_dims)
    # h5_upar is the handle for the ion parallel flow density
    h5_upar = create_dataset(dynamic, "parallel_flow", mk_float, dims, chunk=chunk_dims)
    # h5_ppar is the handle for the ion parallel pressure
    h5_ppar = create_dataset(dynamic, "parallel_pressure", mk_float, dims, chunk=chunk_dims)
    # h5_qpar is the handle for the ion parallel heat flux
    h5_qpar = create_dataset(dynamic, "parallel_heat_flux", mk_float, dims, chunk=chunk_dims)
    # h5_vth is the handle for the ion thermal speed
    h5_vth = create_dataset(dynamic, "thermal_speed", mk_float, dims, chunk=chunk_dims)

    # reduced_dims is a tuple containing all dimensions for the relevant data aside from time
    reduced_dims = (nz, nr, n_neutral_species)
    # given the tuple reduced_dims that contains all dimensions except the time dimension,
    # return chunk_dims tuple that indicates the data chunk written to hdf5 file each write
    # and the dims tuple which also contains the max size of the dataset, accounting for
    # multiple time slices
    chunk_dims, dims = hdf5_dynamic_dims(reduced_dims)
    # h5_density_neutral is the handle for the neutral particle density
    h5_density_neutral = create_dataset(dynamic, "density_neutral", mk_float, dims, chunk=chunk_dims)
    # h5_uz_neutral is the handle for the neutral z momentum density
    h5_uz_neutral = create_dataset(dynamic, "uz_neutral", mk_float, dims, chunk=chunk_dims)
    # h5_pz_neutral is the handle for the neutral species zz pressure
    h5_pz_neutral = create_dataset(dynamic, "pz_neutral", mk_float, dims, chunk=chunk_dims)
    # h5_qz_neutral is the handle for the neutral z heat flux
    h5_qz_neutral = create_dataset(dynamic, "qz_neutral", mk_float, dims, chunk=chunk_dims)
    # h5_thermal_speed_neutral is the handle for the neutral thermal speed
    h5_thermal_speed_neutral = create_dataset(dynamic, "thermal_speed_neutral", mk_float, dims, chunk=chunk_dims)

    return h5_time, h5_phi, h5_Er, h5_Ez, h5_density, h5_upar, h5_ppar, h5_qpar, h5_vth,
        h5_density_neutral, h5_uz_neutral, h5_pz_neutral, h5_qz_neutral,
        h5_thermal_speed_neutral
end

"""
define dynamic (time-evolving) distribution function variables for writing to the hdf5
file
"""
function define_dynamic_dfn_variables_hdf5!(fid, nz, nr, nvz, nvr, nvzeta, nvpa, nvperp,
                                            n_ion_species, n_neutral_species)
    dynamic = create_group(fid, "dynamic_data")
    # create the time variable initially to have one element but allow it to be expanded
    # indefinitely (up to the largest unsigned integer in size)
    h5_time = create_dataset(dynamic, "time", mk_float, ((1,),(-1,)), chunk=(1,))

    # reduced_dims is a tuple containing all dimensions for the relevant data aside from time
    reduced_dims = (nvpa, nvperp, nz, nr, n_ion_species)
    # given the tuple reduced_dims that contains all dimensions except the time dimension,
    # return chunk_dims tuple that indicates the data chunk written to hdf5 file each write
    # and the dims tuple which also contains the max size of the dataset, accounting for
    # multiple time slices
    chunk_dims, dims = hdf5_dynamic_dims(reduced_dims)
    # h5_f is the handle for the ion pdf
    h5_f = create_dataset(dynamic, "f", mk_float, dims, chunk=chunk_dims)

    # reduced_dims is a tuple containing all dimensions for the relevant data aside from time
    reduced_dims = (nvz, nvr, nvzeta, nz, nr, n_neutral_species)
    # given the tuple reduced_dims that contains all dimensions except the time dimension,
    # return chunk_dims tuple that indicates the data chunk written to hdf5 file each write
    # and the dims tuple which also contains the max size of the dataset, accounting for
    # multiple time slices
    chunk_dims, dims = hdf5_dynamic_dims(reduced_dims)
    # h5_f_neutral is the handle for the neutral pdf
    h5_f_neutral = create_dataset(dynamic, "f_neutral", mk_float, dims, chunk=chunk_dims)

    return h5_time, h5_f, h5_f_neutral
end

function extend_time_index!(h5, t_idx)
    for var in h5.fid["dynamic_data"]
        dims = size(var)
        nd = ndims(var)
        dims_mod = (first(dims,nd-1)..., t_idx)
        HDF5.set_extent_dims(var, dims_mod)
    end
    return nothing
end

"""
    define_static_variables!(vz,vr,vzeta,vpa,vperp,z,r,composition,collisions,evolve_ppar)

Define static (i.e. time-independent) variables for an output file.
"""
function define_static_variables!(fid,vz,vr,vzeta,vpa,vperp,z,r,composition,collisions)
    # create and write the "r" variable to file
    varname = "r"
    attributes = Dict("description" => "radial coordinate")
    dims = ("nr",)
    vartype = mk_float
    var = defVar(fid, varname, vartype, dims, attrib=attributes)
    var[:] = r.grid
    # create and write the "r_wgts" variable to file
    varname = "r_wgts"
    attributes = Dict("description" => "integration weights for radial coordinate")
    vartype = mk_float
    var = defVar(fid, varname, vartype, dims, attrib=attributes)
    var[:] = r.wgts
    # create and write the "z" variable to file
    varname = "z"
    attributes = Dict("description" => "parallel coordinate")
    dims = ("nz",)
    vartype = mk_float
    var = defVar(fid, varname, vartype, dims, attrib=attributes)
    var[:] = z.grid
    # create and write the "z_wgts" variable to file
    varname = "z_wgts"
    attributes = Dict("description" => "integration weights for parallel coordinate")
    vartype = mk_float
    var = defVar(fid, varname, vartype, dims, attrib=attributes)
    var[:] = z.wgts
    # create and write the "vperp" variable to file
    varname = "vperp"
    attributes = Dict("description" => "parallel velocity")
    dims = ("nvperp",)
    vartype = mk_float
    var = defVar(fid, varname, vartype, dims, attrib=attributes)
    var[:] = vperp.grid
    # create and write the "vperp_wgts" variable to file
    varname = "vperp_wgts"
    attributes = Dict("description" => "integration weights for parallel velocity coordinate")
    vartype = mk_float
    var = defVar(fid, varname, vartype, dims, attrib=attributes)
    var[:] = vperp.wgts
    # create and write the "vpa" variable to file
    varname = "vpa"
    attributes = Dict("description" => "parallel velocity")
    dims = ("nvpa",)
    vartype = mk_float
    var = defVar(fid, varname, vartype, dims, attrib=attributes)
    var[:] = vpa.grid
    # create and write the "vpa_wgts" variable to file
    varname = "vpa_wgts"
    attributes = Dict("description" => "integration weights for parallel velocity coordinate")
    vartype = mk_float
    var = defVar(fid, varname, vartype, dims, attrib=attributes)
    var[:] = vpa.wgts
    # create and write the "vzeta" variable to file
    varname = "vzeta"
    attributes = Dict("description" => "parallel velocity")
    dims = ("nvzeta",)
    vartype = mk_float
    var = defVar(fid, varname, vartype, dims, attrib=attributes)
    var[:] = vzeta.grid
    # create and write the "vzeta_wgts" variable to file
    varname = "vzeta_wgts"
    attributes = Dict("description" => "integration weights for parallel velocity coordinate")
    vartype = mk_float
    var = defVar(fid, varname, vartype, dims, attrib=attributes)
    var[:] = vzeta.wgts
    # create and write the "vr" variable to file
    varname = "vr"
    attributes = Dict("description" => "parallel velocity")
    dims = ("nvr",)
    vartype = mk_float
    var = defVar(fid, varname, vartype, dims, attrib=attributes)
    var[:] = vr.grid
    # create and write the "vr_wgts" variable to file
    varname = "vr_wgts"
    attributes = Dict("description" => "integration weights for parallel velocity coordinate")
    vartype = mk_float
    var = defVar(fid, varname, vartype, dims, attrib=attributes)
    var[:] = vr.wgts
    # create and write the "vz" variable to file
    varname = "vz"
    attributes = Dict("description" => "parallel velocity")
    dims = ("nvz",)
    vartype = mk_float
    var = defVar(fid, varname, vartype, dims, attrib=attributes)
    var[:] = vz.grid
    # create and write the "vz_wgts" variable to file
    varname = "vz_wgts"
    attributes = Dict("description" => "integration weights for parallel velocity coordinate")
    vartype = mk_float
    var = defVar(fid, varname, vartype, dims, attrib=attributes)
    var[:] = vz.wgts
    # create and write the "T_e" variable to file
    varname = "T_e"
    attributes = Dict("description" => "electron temperature")
    dims = ()
    vartype = mk_float
    var = defVar(fid, varname, vartype, dims, attrib=attributes)
    var[:] = composition.T_e
    # create and write the "charge_exchange_frequency" variable to file
    varname = "charge_exchange_frequency"
    attributes = Dict("description" => "charge exchange collision frequency")
    dims = ()
    vartype = mk_float
    var = defVar(fid, varname, vartype, dims, attrib=attributes)
    var[:] = collisions.charge_exchange

    return nothing
end

"""
    define_dynamic_variables!(fid)

Define dynamic (i.e. time-evolving) variables for an output file.
"""
function define_dynamic_moment_variables!(fid)
    # create the "time" variable
    varname = "time"
    attributes = Dict("description" => "time")
    dims = ("ntime",)
    vartype = mk_float
    cdf_time = defVar(fid, varname, vartype, dims, attrib=attributes)
    # create variables that are floats with data in the z and time dimensions
    vartype = mk_float
    dims = ("nz","nr","ntime")
    # create the "phi" variable, which will contain the electrostatic potential
    varname = "phi"
    attributes = Dict("description" => "electrostatic potential",
                      "units" => "T_ref/e")
    cdf_phi = defVar(fid, varname, vartype, dims, attrib=attributes)
    # create the "Er" variable, which will contain the radial electric field
    varname = "Er"
    attributes = Dict("description" => "electrostatic potential",
                      "units" => "T_ref/e L_ref")
    cdf_Er = defVar(fid, varname, vartype, dims, attrib=attributes)
    # create the "Ez" variable, which will contain the electric field along z
    varname = "Ez"
    attributes = Dict("description" => "electrostatic potential",
                      "units" => "T_ref/e L_ref")
    cdf_Ez = defVar(fid, varname, vartype, dims, attrib=attributes)
    # create variables that are floats with data in the z, ion species and time dimensions
    vartype = mk_float
    dims = ("nz","nr","n_ion_species","ntime")
    # create the "density" variable, which will contain the charged species densities
    varname = "density"
    attributes = Dict("description" => "charged species density",
                      "units" => "n_ref")
    cdf_density = defVar(fid, varname, vartype, dims, attrib=attributes)
    # create the "parallel_flow" variable, which will contain the charged species parallel flows
    varname = "parallel_flow"
    attributes = Dict("description" => "charged species parallel flow",
                      "units" => "c_ref = sqrt(2*T_ref/mi)")
    cdf_upar = defVar(fid, varname, vartype, dims, attrib=attributes)
    # create the "parallel_pressure" variable, which will contain the charged species parallel pressures
    varname = "parallel_pressure"
    attributes = Dict("description" => "charged species parallel pressure",
                      "units" => "n_ref*T_ref")
    cdf_ppar = defVar(fid, varname, vartype, dims, attrib=attributes)
    # create the "parallel_heat_flux" variable, which will contain the charged species parallel heat fluxes
    varname = "parallel_heat_flux"
    attributes = Dict("description" => "charged species parallel heat flux",
                      "units" => "n_ref*T_ref*c_ref")
    cdf_qpar = defVar(fid, varname, vartype, dims, attrib=attributes)
    # create the "thermal_speed" variable, which will contain the charged  species thermal speed
    varname = "thermal_speed"
    attributes = Dict("description" => "charged species thermal speed",
                      "units" => "c_ref")
    cdf_vth = defVar(fid, varname, vartype, dims, attrib=attributes)
    
    # create the "density_neutral" variable, which will contain the neutral species densities
    dims = ("nz","nr","n_neutral_species","ntime")
    varname = "density_neutral"
    attributes = Dict("description" => "neutral species density",
                      "units" => "n_ref")
    cdf_density_neutral = defVar(fid, varname, vartype, dims, attrib=attributes)
    varname = "uz_neutral"
    attributes = Dict("description" => "neutral species mean z velocity",
                      "units" => "c_ref = sqrt(2*T_ref/mi)")
    cdf_uz_neutral = defVar(fid, varname, vartype, dims, attrib=attributes)
    varname = "pz_neutral"
    attributes = Dict("description" => "neutral species zz pressure",
                      "units" => "n_ref*T_ref")
    cdf_pz_neutral = defVar(fid, varname, vartype, dims, attrib=attributes)
    varname = "qz_neutral"
    attributes = Dict("description" => "neutral species z heat flux",
                      "units" => "n_ref*T_ref*c_ref")
    cdf_qz_neutral = defVar(fid, varname, vartype, dims, attrib=attributes)
    varname = "thermal_speed_neutral"
    attributes = Dict("description" => "neutral species thermal speed",
                      "units" => "c_ref")
    cdf_vth_neutral = defVar(fid, varname, vartype, dims, attrib=attributes)
    
    
    return cdf_time, cdf_phi, cdf_Er, cdf_Ez, cdf_density, cdf_upar, cdf_ppar, cdf_qpar, cdf_vth, 
       cdf_density_neutral, cdf_uz_neutral, cdf_pz_neutral, cdf_qz_neutral, cdf_vth_neutral
end

"""
    define_dynamic_variables!(fid)

Define dynamic (i.e. time-evolving) variables for an output file.
"""
function define_dynamic_dfn_variables!(fid)
    # create the "time" variable
    varname = "time"
    attributes = Dict("description" => "time")
    dims = ("ntime",)
    vartype = mk_float
    cdf_time = defVar(fid, varname, vartype, dims, attrib=attributes)
    # create the "f" variable
    varname = "f"
    attributes = Dict("description" => "charged species distribution function")
    vartype = mk_float
    dims = ("nvpa","nvperp","nz","nr","n_ion_species","ntime")
    cdf_f = defVar(fid, varname, vartype, dims, attrib=attributes)
    # create variables that are floats with data in the z, neutral species and time dimensions
    # create the "f_neutral" variable
    varname = "f_neutral"
    attributes = Dict("description" => "neutral species distribution function")
    vartype = mk_float
    dims = ("nvz","nvr","nvzeta","nz","nr","n_neutral_species","ntime")
    cdf_f_neutral = defVar(fid, varname, vartype, dims, attrib=attributes)
    return cdf_time, cdf_f, cdf_f_neutral
end

"""
setup file i/o for netcdf
moment variables only 
"""
function setup_moments_netcdf_io(prefix, r, z, vperp, vpa, vzeta, vr, vz, composition, collisions)
    # the netcdf file will be given by output_dir/run_name with .cdf appended
    filename = string(prefix,".moments.cdf")
    # if a netcdf file with the requested name already exists, remove it
    isfile(filename) && rm(filename)
    # create the new NetCDF file
    fid = NCDataset(filename,"c")
    # write a header to the NetCDF file
    fid.attrib["file_info"] = "This is a NetCDF file containing output data from the moment_kinetics code"
    ### define coordinate dimensions ###
    define_dimensions!(fid, vz.n, vr.n, vzeta.n, vpa.n, vperp.n, z.n, r.n, composition.n_species,
                       composition.n_ion_species, composition.n_neutral_species)
    ### create and write static variables to file ###
    define_static_variables!(fid,vz,vr,vzeta,vpa,vperp,z,r,composition,collisions)
    ### create variables for time-dependent quantities and store them ###
    ### in a struct for later access ###
    cdf_time, cdf_phi, cdf_Er, cdf_Ez, cdf_density, cdf_upar, cdf_ppar, cdf_qpar, cdf_vth, cdf_density_neutral, cdf_uz_neutral, cdf_pz_neutral, cdf_qz_neutral, cdf_vth_neutral =
        define_dynamic_moment_variables!(fid)
        define_dynamic_variables!(fid)

    # create a struct that stores the variables and other info needed for
    # writing to the netcdf file during run-time
    return netcdf_moments_info(fid, cdf_time, cdf_phi, cdf_Er, cdf_Ez, cdf_density, cdf_upar,
                       cdf_ppar, cdf_qpar, cdf_vth, cdf_density_neutral,
                       cdf_uz_neutral, cdf_pz_neutral, cdf_qz_neutral, cdf_vth_neutral)
end

"""
setup file i/o for netcdf
dfn variables only 
"""
function setup_dfns_netcdf_io(prefix, r, z, vperp, vpa, vzeta, vr, vz, composition, collisions)
    # the netcdf file will be given by output_dir/run_name with _dfns.cdf appended
    filename = string(prefix,".dfns.cdf")
    # if a netcdf file with the requested name already exists, remove it
    isfile(filename) && rm(filename)
    # create the new NetCDF file
    fid = NCDataset(filename,"c")
    # write a header to the NetCDF file
    fid.attrib["file_info"] = "This is a NetCDF file containing output distribution function data from the moment_kinetics code"
    ### define coordinate dimensions ###
    define_dimensions!(fid, vz.n, vr.n, vzeta.n, vpa.n, vperp.n, z.n, r.n, composition.n_species,
                       composition.n_ion_species, composition.n_neutral_species)
    ### create and write static variables to file ###
    define_static_variables!(fid,vz,vr,vzeta,vpa,vperp,z,r,composition,collisions)
    ### create variables for time-dependent quantities and store them ###
    ### in a struct for later access ###
    cdf_time, cdf_f, cdf_f_neutral = define_dynamic_dfn_variables!(fid)

    # create a struct that stores the variables and other info needed for
    # writing to the netcdf file during run-time
    return netcdf_dfns_info(fid, cdf_time, cdf_f, cdf_f_neutral)
end

"""
setup file i/o for hdf5
moment variables only
"""
function setup_moments_hdf5_io(prefix, r, z, vperp, vpa, vzeta, vr, vz, composition, collisions)
    # the netcdf file will be given by output_dir/run_name with .cdf appended
    filename = string(prefix,".moments.h5")
    # if a netcdf file with the requested name already exists, remove it
    isfile(filename) && rm(filename)
    # create the new NetCDF file
    fid = h5open(filename,"cw")
    # write a header to the hdf5 file
    #fid.attrib["file_info"] = "This is a NetCDF file containing output data from the moment_kinetics code"
    # write some overview information to the hdf5 file
    write_overview_hdf5!(fid, composition, collisions)
    ### define coordinate dimensions ###
    define_spatial_coordinates_hdf5!(fid, z, r)
    # ### create and write static variables to file ###
    # define_static_variables!(fid,vz,vr,vzeta,vpa,vperp,z,r,composition,collisions)
    # ### create variables for time-dependent quantities and store them ###
    # ### in a struct for later access ###
    h5_time, h5_phi, h5_Er, h5_Ez, h5_density, h5_upar, h5_ppar, h5_qpar, h5_vth,
        h5_density_neutral, h5_uz_neutral, h5_pz_neutral, h5_qz_neutral, h5_vth_neutral =
        define_dynamic_moment_variables_hdf5!(fid, z.n, r.n, composition.n_ion_species,
                                              composition.n_neutral_species)

    # create a struct that stores the variables and other info needed for
    # writing to the netcdf file during run-time
    return hdf5_moments_info(fid, h5_time, h5_phi, h5_Er, h5_Ez, h5_density, h5_upar,
                             h5_ppar, h5_qpar, h5_vth, h5_density_neutral, h5_uz_neutral,
                             h5_pz_neutral, h5_qz_neutral, h5_vth_neutral)
end

"""
setup file i/o for hdf5
dfn variables only
"""
function setup_dfn_hdf5_io(prefix, r, z, vperp, vpa, vzeta, vr, vz, composition, collisions)
    # the netcdf file will be given by output_dir/run_name with .cdf appended
    filename = string(prefix,".dfns.h5")
    # if a netcdf file with the requested name already exists, remove it
    isfile(filename) && rm(filename)
    # create the new NetCDF file
    fid = h5open(filename,"cw")
    # write a header to the hdf5 file
    #fid.attrib["file_info"] = "This is a NetCDF file containing output data from the moment_kinetics code"
    # write some overview information to the hdf5 file
    write_overview_hdf5!(fid, composition, collisions)
    ### define coordinate dimensions ###
    coords_group = define_spatial_coordinates_hdf5!(fid, z, r)
    add_vspace_coordinates_hdf5!(coords_group, vz, vr, vzeta, vpa, vperp)
    # ### create and write static variables to file ###
    # define_static_variables!(fid,vz,vr,vzeta,vpa,vperp,z,r,composition,collisions)
    # ### create variables for time-dependent quantities and store them ###
    # ### in a struct for later access ###
    h5_time, h5_f, h5_f_neutral = define_dynamic_dfn_variables_hdf5!(fid, z.n, r.n, vz.n,
        vr.n, vzeta.n, vpa.n, vperp.n, composition.n_ion_species,
        composition.n_neutral_species)

    # create a struct that stores the variables and other info needed for
    # writing to the netcdf file during run-time
    return hdf5_dfns_info(fid, h5_time, h5_f, h5_f_neutral)
end

"""
close all opened output files
"""
function finish_file_io(ascii_io, cdf_moments, cdf_dfns, hdf5)
    @serial_region begin
        # Only read/write from first process in each 'block'

        # get the fields in the ascii_ios struct
        ascii_io_fields = fieldnames(typeof(ascii_io))
        for x ∈ ascii_io_fields
            io = getfield(ascii_io, x)
            if io !== nothing
                close(io)
            end
        end
        close(hdf5.fid)
    end
    return nothing
end

"""
"""
function write_data_to_ascii(moments, fields, vpa, vperp, z, r, t, n_ion_species,
                             n_neutral_species, ascii_io)
    if ascii_io.moments_charged === nothing
        # ascii I/O is disabled
        return nothing
    end

    @serial_region begin
        # Only read/write from first process in each 'block'

        #write_f_ascii(ff, z, vpa, t, ascii_io.ff)
        write_moments_charged_ascii(moments.charged, z, r, t, n_ion_species, ascii_io.moments_charged)
        if n_neutral_species > 0
            write_moments_neutral_ascii(moments.neutral, z, r, t, n_neutral_species, ascii_io.moments_neutral)
        end
        write_fields_ascii(fields, z, r, t, ascii_io.fields)
    end
    return nothing
end

"""
write the function f(z,vpa) at this time slice
"""
function write_f_ascii(f, z, vpa, t, ascii_io)
    @serial_region begin
        # Only read/write from first process in each 'block'

        @inbounds begin
            n_species = size(f,3)
            for is ∈ 1:n_species
                for j ∈ 1:vpa.n
                    for i ∈ 1:z.n
                        println(ascii_io,"t: ", t, "   spec: ", is, ",   z: ", z.grid[i],
                            ",  vpa: ", vpa.grid[j], ",   f: ", f[i,j,is])
                    end
                    println(ascii_io)
                end
                println(ascii_io)
            end
            println(ascii_io)
        end
    end
    return nothing
end

"""
write moments of the charged species distribution function f at this time slice
"""
function write_moments_charged_ascii(mom, z, r, t, n_species, ascii_io)
    @serial_region begin
        # Only read/write from first process in each 'block'

        @inbounds begin
            for is ∈ 1:n_species
                for ir ∈ 1:r.n
                    for iz ∈ 1:z.n
                        println(ascii_io,"t: ", t, "   species: ", is, "   r: ", r.grid[ir], "   z: ", z.grid[iz],
                            "  dens: ", mom.dens[iz,ir,is], "   upar: ", mom.upar[iz,ir,is],
                            "   ppar: ", mom.ppar[iz,ir,is], "   qpar: ", mom.qpar[iz,ir,is])
                    end
                end
            end
        end
        println(ascii_io,"")
    end
    return nothing
end

"""
write moments of the neutral species distribution function f_neutral at this time slice
"""
function write_moments_neutral_ascii(mom, z, r, t, n_species, ascii_io)
    @serial_region begin
        # Only read/write from first process in each 'block'

        @inbounds begin
            for is ∈ 1:n_species
                for ir ∈ 1:r.n
                    for iz ∈ 1:z.n
                        println(ascii_io,"t: ", t, "   species: ", is, "   r: ", r.grid[ir], "   z: ", z.grid[iz],
                            "  dens: ", mom.dens[iz,ir,is], "   uz: ", mom.uz[iz,ir,is],
                            "   ur: ", mom.ur[iz,ir,is], "   uzeta: ", mom.uzeta[iz,ir,is])
                    end
                end
            end
        end
        println(ascii_io,"")
    end
    return nothing
end

"""
write electrostatic potential at this time slice
"""
function write_fields_ascii(flds, z, r, t, ascii_io)
    @serial_region begin
        # Only read/write from first process in each 'block'

        @inbounds begin
            for ir ∈ 1:r.n
                for iz ∈ 1:z.n
                    println(ascii_io,"t: ", t, "   r: ", r.grid[ir],"   z: ", z.grid[iz], "  phi: ", flds.phi[iz,ir])
                end
            end
        end
        println(ascii_io,"")
    end
    return nothing
end

"""
write time-dependent data to the netcdf file
moments data only 
"""
function write_moments_data_to_netcdf(moments, fields, t, n_ion_species, n_neutral_species, cdf, t_idx)
    @serial_region begin
        # Only read/write from first process in each 'block'

        # add the time for this time slice to the netcdf file
        cdf.time[t_idx] = t
        # add the electrostatic potential data at this time slice to the netcdf file
        cdf.phi[:,:,t_idx] = fields.phi
        cdf.Er[:,:,t_idx] = fields.Er
        cdf.Ez[:,:,t_idx] = fields.Ez
        # add the density data at this time slice to the netcdf file
        cdf.density[:,:,:,t_idx] = moments.charged.dens
        cdf.parallel_flow[:,:,:,t_idx] = moments.charged.upar
        cdf.parallel_pressure[:,:,:,t_idx] = moments.charged.ppar
        cdf.parallel_heat_flux[:,:,:,t_idx] = moments.charged.qpar
        cdf.thermal_speed[:,:,:,t_idx] = moments.charged.vth
        if n_neutral_species > 0
            cdf.density_neutral[:,:,:,t_idx] = moments.neutral.dens
            cdf.uz_neutral[:,:,:,t_idx] = moments.neutral.uz
            cdf.pz_neutral[:,:,:,t_idx] = moments.neutral.pz
            cdf.qz_neutral[:,:,:,t_idx] = moments.neutral.qz
            cdf.thermal_speed_neutral[:,:,:,t_idx] = moments.neutral.vth
        end
    end
    return nothing
end

"""
write time-dependent data to the netcdf file
dfns data only 
"""
function write_dfns_data_to_netcdf(ff, ff_neutral, t, n_ion_species, n_neutral_species, cdf, t_idx)
    @serial_region begin
        # Only read/write from first process in each 'block'

        # add the time for this time slice to the netcdf file
        cdf.time[t_idx] = t
        # add the distribution function data at this time slice to the netcdf file
        cdf.f[:,:,:,:,:,t_idx] = ff
        # add the electrostatic potential data at this time slice to the netcdf file
        if n_neutral_species > 0
            cdf.f_neutral[:,:,:,:,:,:,t_idx] = ff_neutral
        end
    end
    return nothing
end
@debug_shared_array begin
    # Special versions when using DebugMPISharedArray to avoid implicit conversion to
    # Array, which is forbidden.
    function write_dfns_data_to_binary(ff::DebugMPISharedArray, ff_neutral::DebugMPISharedArray,
            t, n_ion_species, n_neutral_species, cdf::netcdf_dfns_info, t_idx)
        @serial_region begin
            # Only read/write from first process in each 'block'

            # add the time for this time slice to the netcdf file
            cdf.time[t_idx] = t
            # add the distribution function data at this time slice to the netcdf file
            cdf.f[:,:,:,:,:,t_idx] = ff.data
            # add the electrostatic potential data at this time slice to the netcdf file
            if n_neutral_species > 0
                cdf.f_neutral[:,:,:,:,:,:,t_idx] = ff_neutral.data
            end
        end
        return nothing
    end
end
@debug_shared_array begin
    # Special versions when using DebugMPISharedArray to avoid implicit conversion to
    # Array, which is forbidden.
    function write_moments_data_to_binary(moments, fields, t, n_ion_species,
            n_neutral_species, cdf::netcdf_moments_info, t_idx)
        @serial_region begin
            # Only read/write from first process in each 'block'

            # add the time for this time slice to the netcdf file
            cdf.time[t_idx] = t
            # add the electrostatic potential data at this time slice to the netcdf file
            cdf.phi[:,:,t_idx] = fields.phi.data
            cdf.Er[:,:,t_idx] = fields.Er.data
            cdf.Ez[:,:,t_idx] = fields.Ez.data
            # add the density data at this time slice to the netcdf file
            cdf.density[:,:,:,t_idx] = moments.charged.dens.data
            cdf.parallel_flow[:,:,:,t_idx] = moments.charged.upar.data
            cdf.parallel_pressure[:,:,:,t_idx] = moments.charged.ppar.data
            cdf.parallel_heat_flux[:,:,:,t_idx] = moments.charged.qpar.data
            cdf.thermal_speed[:,:,:,t_idx] = moments.charged.vth.data
            if n_neutral_species > 0
                cdf.density_neutral[:,:,:,t_idx] = moments.neutral.dens.data
                cdf.uz_neutral[:,:,:,t_idx] = moments.neutral.uz.data
                cdf.pz_neutral[:,:,:,t_idx] = moments.neutral.pz.data
                cdf.qz_neutral[:,:,:,t_idx] = moments.neutral.qz.data
                cdf.thermal_speed_neutral[:,:,:,t_idx] = moments.neutral.vth.data
            end
        end
        return nothing
    end
end

"""
write time-dependent data to the hdf5 file
moments data only
"""
function write_moments_data_to_hdf5(moments, fields, t, n_ion_species, n_neutral_species,
        h5::hdf5_moments_info, t_idx)
    @serial_region begin
        # Only read/write from first process in each 'block'

        # extend the size of the time dimension for all time-evolving quantities
        extend_time_index!(h5, t_idx)
        # add the time for this time slice to the hdf5 file
        h5.time[t_idx] = t
        # add the electrostatic potential and electric field components at this time slice to the hdf5 file
        h5.phi[:,:,t_idx] = fields.phi
        h5.Er[:,:,t_idx] = fields.Er
        h5.Ez[:,:,t_idx] = fields.Ez
        # add the density data at this time slice to the netcdf file
        h5.density[:,:,:,t_idx] = moments.charged.dens
        h5.parallel_flow[:,:,:,t_idx] = moments.charged.upar
        h5.parallel_pressure[:,:,:,t_idx] = moments.charged.ppar
        h5.parallel_heat_flux[:,:,:,t_idx] = moments.charged.qpar
        h5.thermal_speed[:,:,:,t_idx] = moments.charged.vth
        if n_neutral_species > 0
            h5.density_neutral[:,:,:,t_idx] = moments.neutral.dens
            h5.uz_neutral[:,:,:,t_idx] = moments.neutral.uz
            h5.pz_neutral[:,:,:,t_idx] = moments.neutral.pz
            h5.qz_neutral[:,:,:,t_idx] = moments.neutral.qz
            h5.thermal_speed_neutral[:,:,:,t_idx] = moments.neutral.vth
        end
    end
    return nothing
end

"""
write time-dependent data to the hdf5 file
dfns data only
"""
function write_dfns_data_to_hdf5(ff, ff_neutral, moments, fields, t, n_ion_species,
                                    n_neutral_species, h5, t_idx)
    @serial_region begin
        # Only read/write from first process in each 'block'

        # extend the size of the time dimension for all time-evolving quantities
        extend_time_index!(h5, t_idx)
        # add the time for this time slice to the hdf5 file
        h5.time[t_idx] = t
        # add the distribution function data at this time slice to the netcdf file
        h5.f[:,:,:,:,:,t_idx] = ff
        if n_neutral_species > 0
            h5.f_neutral[:,:,:,:,:,:,t_idx] = ff_neutral
        end
    end
    return nothing
end

"""
accepts an option name which has been identified as problematic and returns
an appropriate error message
"""
function input_option_error(option_name, input)
    msg = string("'",input,"'")
    msg = string(msg, " is not a valid ", option_name)
    error(msg)
    return nothing
end

"""
opens an output file with the requested prefix and extension
and returns the corresponding io stream (identifier)
"""
function open_output_file(prefix, ext)
    str = string(prefix,".",ext)
    return io = open(str,"w")
end

"""
An nc_info instance that may be initialised for writing debug output

This is a non-const module variable, so does cause type instability, but it is only used
for debugging (from `debug_dump()`) so performance is not critical.
"""
debug_output_file = nothing

"""
Global counter for calls to debug_dump
"""
const debug_output_counter = Ref(1)

"""
    debug_dump(ff, dens, upar, ppar, phi, t; istage=0, label="")
    debug_dump(fvec::scratch_pdf, fields::em_fields_struct, t; istage=0, label="")

Dump variables into a NetCDF file for debugging

Intended to be called more frequently than `write_data_to_netcdf()`, possibly several
times within a timestep, so includes a `label` argument to identify the call site.

Writes to a file called `debug_output.cdf` in the current directory.

Can either be called directly with the arrays to be dumped (fist signature), or using
`scratch_pdf` and `em_fields_struct` structs.

`nothing` can be passed to any of the positional arguments (if they are unavailable at a
certain point in the code, or just not interesting). `t=nothing` will set `t` to the
value saved in the previous call (or 0.0 on the first call). Passing `nothing` to the
other arguments will set that array to `0.0` for this call (need to write some value so
all the arrays have the same length, with an entry for each call to `debug_dump()`).
"""
function debug_dump end
function debug_dump(ff, dens, upar, ppar, phi, t; istage=0, label="")
    global debug_output_file

    # Only read/write from first process in each 'block'
    original_loop_region = loop_ranges[].parallel_dims
    begin_serial_region()
    @serial_region begin
        if debug_output_file === nothing
            # Open the file the first time`debug_dump()` is called

            debug_output_counter[] = 1

            (nvpa, nvperp, nz, nr, n_species) = size(ff)
            # the netcdf file will be given by output_dir/run_name with .cdf appended
            filename = string("debug_output.cdf")
            # if a netcdf file with the requested name already exists, remove it
            isfile(filename) && rm(filename)
            # create the new NetCDF file
            fid = NCDataset(filename,"c")
            # write a header to the NetCDF file
            fid.attrib["file_info"] = "This is a NetCDF file containing debug output from the moment_kinetics code"
            ### define coordinate dimensions ###
            define_dimensions!(fid, nvpa, nvperp, nz, nr, n_species)
            ### create variables for time-dependent quantities and store them ###
            ### in a struct for later access ###
            cdf_time, cdf_f, cdf_phi, cdf_density, cdf_upar, cdf_ppar, cdf_qpar, cdf_vth =
                define_dynamic_variables!(fid)

            # create the "istage" variable, used to identify the rk stage where
            # `debug_dump()` was called
            varname = "istage"
            attributes = Dict("description" => "rk istage")
            dims = ("ntime",)
            vartype = mk_int
            cdf_istage = defVar(fid, varname, vartype, dims, attrib=attributes)
            # create the "label" variable, used to identify the `debug_dump()` call-site
            varname = "label"
            attributes = Dict("description" => "call-site label")
            dims = ("ntime",)
            vartype = String
            cdf_label = defVar(fid, varname, vartype, dims, attrib=attributes)

            # create a struct that stores the variables and other info needed for
            # writing to the netcdf file during run-time
            debug_output_file = (fid=fid, time=cdf_time, f=cdf_f, phi=cdf_phi,
                                 density=cdf_density, parallel_flow=cdf_upar,
                                 parallel_pressure=cdf_ppar,
                                 parallel_heat_flux=cdf_qpar, thermal_speed=cdf_vth,
                                 istage=cdf_istage, label=cdf_label)
        end

        # add the time for this time slice to the netcdf file
        if t === nothing
            if debug_output_counter[] == 1
                debug_output_file.time[debug_output_counter[]] = 0.0
            else
                debug_output_file.time[debug_output_counter[]] =
                debug_output_file.time[debug_output_counter[]-1]
            end
        else
            debug_output_file.time[debug_output_counter[]] = t
        end
        # add the rk istage for this call to the netcdf file
        debug_output_file.istage[debug_output_counter[]] = istage
        # add the label for this call to the netcdf file
        debug_output_file.label[debug_output_counter[]] = label
        # add the distribution function data at this time slice to the netcdf file
        if ff === nothing
            debug_output_file.f[:,:,:,:,:,debug_output_counter[]] .= 0.0
        else
            debug_output_file.f[:,:,:,:,:,debug_output_counter[]] = ff
        end
        # add the electrostatic potential data at this time slice to the netcdf file
        if phi === nothing
            debug_output_file.phi[:,:,debug_output_counter[]] .= 0.0
        else
            debug_output_file.phi[:,:,debug_output_counter[]] = phi
        end
        # add the moments data at this time slice to the netcdf file
        if dens === nothing
            debug_output_file.density[:,:,:,debug_output_counter[]] .= 0.0
        else
            debug_output_file.density[:,:,:,debug_output_counter[]] = dens
        end
        if upar === nothing
            debug_output_file.parallel_flow[:,:,:,debug_output_counter[]] .= 0.0
        else
            debug_output_file.parallel_flow[:,:,:,debug_output_counter[]] = upar
        end
        if ppar === nothing
            debug_output_file.parallel_pressure[:,:,:,debug_output_counter[]] .= 0.0
        else
            debug_output_file.parallel_pressure[:,:,:,debug_output_counter[]] = ppar
        end
    end

    debug_output_counter[] += 1

    # hacky work-around to restore original region
    _block_synchronize()
    loop_ranges[] = looping.loop_ranges_store[original_loop_region]

    return nothing
end
function debug_dump(fvec::Union{scratch_pdf,Nothing},
                    fields::Union{em_fields_struct,Nothing}, t; istage=0, label="")
    if fvec === nothing
        pdf = nothing
        density = nothing
        upar = nothing
        ppar = nothing
    else
        pdf = fvec.pdf
        density = fvec.density
        upar = fvec.upar
        ppar = fvec.ppar
    end
    if fields === nothing
        phi = nothing
    else
        phi = fields.phi
    end
    return debug_dump(pdf, density, upar, ppar, phi, t; istage=istage, label=label)
end

end
