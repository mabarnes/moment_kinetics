# No separate module defined here as this file is included within the file_io module

using HDF5

"""
structure containing the data/metadata needed for hdf5 file i/o
moments & fields only
"""
struct hdf5_moments_info{Ttime, Tphi, Tmomi, Tmomn} <: io_moments_info
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
struct hdf5_dfns_info{Ttime, Tfi, Tfn} <: io_dfns_info
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
function setup_dfns_hdf5_io(prefix, r, z, vperp, vpa, vzeta, vr, vz, composition, collisions)
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
write time-dependent data to the hdf5 file
moments data only
"""
function write_moments_data_to_binary(moments, fields, t, n_ion_species,
                                      n_neutral_species, h5::hdf5_moments_info, t_idx)
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
function write_dfns_data_to_binary(ff, ff_neutral, t, n_ion_species, n_neutral_species,
                                   h5::hdf5_dfns_info, t_idx)
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

@debug_shared_array begin
    # Special versions when using DebugMPISharedArray to avoid implicit conversion to
    # Array, which is forbidden.
    function write_dfns_data_to_binary(ff::DebugMPISharedArray, ff_neutral::DebugMPISharedArray,
            t, n_ion_species, n_neutral_species, h5::hdf5_dfns_info, t_idx)
        @serial_region begin
            # Only read/write from first process in each 'block'

            # add the time for this time slice to the netcdf file
            h5.time[t_idx] = t
            # add the distribution function data at this time slice to the netcdf file
            h5.f[:,:,:,:,:,t_idx] = ff.data
            # add the electrostatic potential data at this time slice to the netcdf file
            if n_neutral_species > 0
                h5.f_neutral[:,:,:,:,:,:,t_idx] = ff_neutral.data
            end
        end
        return nothing
    end
end

@debug_shared_array begin
    # Special versions when using DebugMPISharedArray to avoid implicit conversion to
    # Array, which is forbidden.
    function write_moments_data_to_binary(moments, fields, t, n_ion_species,
            n_neutral_species, h5::hdf5_moments_info, t_idx)
        @serial_region begin
            # Only read/write from first process in each 'block'

            # add the time for this time slice to the netcdf file
            h5.time[t_idx] = t
            # add the electrostatic potential data at this time slice to the netcdf file
            h5.phi[:,:,t_idx] = fields.phi.data
            h5.Er[:,:,t_idx] = fields.Er.data
            h5.Ez[:,:,t_idx] = fields.Ez.data
            # add the density data at this time slice to the netcdf file
            h5.density[:,:,:,t_idx] = moments.charged.dens.data
            h5.parallel_flow[:,:,:,t_idx] = moments.charged.upar.data
            h5.parallel_pressure[:,:,:,t_idx] = moments.charged.ppar.data
            h5.parallel_heat_flux[:,:,:,t_idx] = moments.charged.qpar.data
            h5.thermal_speed[:,:,:,t_idx] = moments.charged.vth.data
            if n_neutral_species > 0
                h5.density_neutral[:,:,:,t_idx] = moments.neutral.dens.data
                h5.uz_neutral[:,:,:,t_idx] = moments.neutral.uz.data
                h5.pz_neutral[:,:,:,t_idx] = moments.neutral.pz.data
                h5.qz_neutral[:,:,:,t_idx] = moments.neutral.qz.data
                h5.thermal_speed_neutral[:,:,:,t_idx] = moments.neutral.vth.data
            end
        end
        return nothing
    end
end

