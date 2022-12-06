# No separate module defined here as this file is included within the file_io module

"""
structure containing the data/metadata needed for netcdf file i/o
moments & fields only 
"""
struct netcdf_moments_info{Ttime, Tphi, Tmomi, Tmomn} <: io_moments_info
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
struct netcdf_dfns_info{Ttime, Tfi, Tfn} <: io_dfns_info
    # file identifier for the netcdf file to which data is written
    fid::NCDataset
    # handle for the time variable
    time::Ttime
    # handle for the charged species distribution function variable
    f::Tfi
    # handle for the neutral species distribution function variable
    f_neutral::Tfn
    
end

# Define the steps for creating a NetCDF file in utility functions so that they can be
# shared between `setup_netcdf_io()` and `debug_dump()`
"""
    define_spatial_dimensions_netcdf!(fid, nz, nr, n_species, n_ion_species=nothing,
        n_neutral_species=nothing)

Define spatial dimensions for an output file.
"""
function define_spatial_dimensions_netcdf!(fid, nz, nr, n_species, n_ion_species=nothing,
        n_neutral_species=nothing)
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
    define_vspace_dimensions_netcdf!(fid, nvz, nvr, nvzeta, nvpa, nvperp)

Define velocity space dimensions for an output file.
"""
function define_vspace_dimensions_netcdf!(fid, nvz, nvr, nvzeta, nvpa, nvperp)
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

    return nothing
end

"""
    define_static_variables_netcdf!(fid, z, r, composition, collisions)

Define static (i.e. time-independent) variables for an output file.
"""
function define_static_variables_netcdf!(fid, z, r, composition, collisions)
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
    define_static_vspace_variables_netcdf!(fid, vz, vr, vzeta, vpa, vperp)

Define static (i.e. time-independent) variables for an output file.
"""
function define_static_vspace_variables_netcdf!(fid, vz, vr, vzeta, vpa, vperp)
    # create and write the "vperp" variable to file
    varname = "vperp"
    attributes = Dict("description" => "perpendicular velocity")
    dims = ("nvperp",)
    vartype = mk_float
    var = defVar(fid, varname, vartype, dims, attrib=attributes)
    var[:] = vperp.grid
    # create and write the "vperp_wgts" variable to file
    varname = "vperp_wgts"
    attributes = Dict("description" => "integration weights for perpendicular velocity coordinate")
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

    return nothing
end

"""
    define_dynamic_moment_variables_netcdf!(fid)

Define dynamic (i.e. time-evolving) variables for an output file.
"""
function define_dynamic_moment_variables_netcdf!(fid)
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
    define_dynamic_variables_netcdf!(fid)

Define dynamic (i.e. time-evolving) variables for an output file.
"""
function define_dynamic_dfn_variables_netcdf!(fid)
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
    define_spatial_dimensions_netcdf!(fid, z.n, r.n, composition.n_species,
        composition.n_ion_species, composition.n_neutral_species)
    ### create and write static variables to file ###
    define_static_variables_netcdf!(fid,z,r,composition,collisions)
    ### create variables for time-dependent quantities and store them ###
    ### in a struct for later access ###
    cdf_time, cdf_phi, cdf_Er, cdf_Ez, cdf_density, cdf_upar, cdf_ppar, cdf_qpar, cdf_vth,
    cdf_density_neutral, cdf_uz_neutral, cdf_pz_neutral, cdf_qz_neutral, cdf_vth_neutral =
        define_dynamic_moment_variables_netcdf!(fid)

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
    define_spatial_dimensions_netcdf!(fid, z.n, r.n, composition.n_species,
        composition.n_ion_species, composition.n_neutral_species)
    define_vspace_dimensions_netcdf!(fid, vz.n, vr.n, vzeta.n, vpa.n, vperp.n)
    ### create and write static variables to file ###
    define_static_variables_netcdf!(fid,z,r,composition,collisions)
    define_static_vspace_variables_netcdf!(fid,vz,vr,vzeta,vpa,vperp)
    ### create variables for time-dependent quantities and store them ###
    ### in a struct for later access ###
    cdf_time, cdf_f, cdf_f_neutral = define_dynamic_dfn_variables_netcdf!(fid)

    # create a struct that stores the variables and other info needed for
    # writing to the netcdf file during run-time
    return netcdf_dfns_info(fid, cdf_time, cdf_f, cdf_f_neutral)
end

"""
write time-dependent data to the netcdf file
moments data only 
"""
function write_moments_data_to_binary(moments, fields, t, n_ion_species, n_neutral_species, cdf::netcdf_moments_info, t_idx)
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
function write_dfns_data_to_binary(ff, ff_neutral, t, n_ion_species, n_neutral_species,
                                   cdf::netcdf_dfns_info, t_idx)
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

