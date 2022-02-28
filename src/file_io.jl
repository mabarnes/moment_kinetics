"""
"""
module file_io

export input_option_error
export open_output_file
export setup_file_io, finish_file_io
export write_data_to_ascii
export write_data_to_binary

using NCDatasets
using ..looping
using ..type_definitions: mk_float, mk_int

"""
structure containing the various input/output streams
"""
struct ios
    # corresponds to the ascii file to which the distribution function is written
    #ff::IOStream
    # corresponds to the ascii file to which velocity space moments of the
    # distribution function such as density and pressure are written
    moments::IOStream
    # corresponds to the ascii file to which electromagnetic fields
    # such as the electrostatic potential are written
    fields::IOStream
end

# Use this long-winded type (found by using `typeof(v)` where `v` is a variable
# returned from `NCDatasets.defVar()`) because compiler does not seem to be
# able to pick up the return types of `defVar()` at compile time, so without
# using it the result returned from `setup_file_io()` is not a concrete type.
nc_var_type{N} = Union{
   NCDatasets.CFVariable{mk_float, N,
                         NCDatasets.Variable{mk_float, N, NCDatasets.NCDataset},
                         NCDatasets.Attributes{NCDatasets.NCDataset{Nothing}},
                         NamedTuple{(:fillvalue, :scale_factor, :add_offset,
                                     :calendar, :time_origin, :time_factor),
                                    NTuple{6, Nothing}}},
   NCDatasets.CFVariable{mk_float, N,
                         NCDatasets.Variable{mk_float, N,
                                             NCDatasets.NCDataset{Nothing}},
                         NCDatasets.Attributes{NCDatasets.NCDataset{Nothing}},
                         NamedTuple{(:fillvalue, :scale_factor, :add_offset,
                                     :calendar, :time_origin, :time_factor),
                                    NTuple{6, Nothing}}}}

"""
structure containing the data/metadata needed for netcdf file i/o
"""
struct netcdf_info
    # file identifier for the netcdf file to which data is written
    fid::NCDataset
    # handle for the time variable
    time::nc_var_type{1}
    # handle for the distribution function variable
    f::nc_var_type{5}
    # handle for the electrostatic potential variable
    phi::nc_var_type{3}
    # handle for the species density
    density::nc_var_type{4}
    # handle for the species parallel flow
    parallel_flow::nc_var_type{4}
    # handle for the species parallel pressure
    parallel_pressure::nc_var_type{4}
    # handle for the species parallel heat flux
    parallel_heat_flux::nc_var_type{4}
    # handle for the species thermal speed
    thermal_speed::nc_var_type{4}
end

"""
open the necessary output files
"""
function setup_file_io(output_dir, run_name, vpa, z, r, composition,
                       collisions, evolve_ppar)
    begin_serial_region()
    @serial_region begin
        # Only read/write from first process in each 'block'

        # check to see if output_dir exists in the current directory
        # if not, create it
        isdir(output_dir) || mkdir(output_dir)
        out_prefix = string(output_dir, "/", run_name)
        #ff_io = open_output_file(out_prefix, "f_vs_t")
        mom_io = open_output_file(out_prefix, "moments_vs_t")
        fields_io = open_output_file(out_prefix, "fields_vs_t")
        cdf = setup_netcdf_io(out_prefix, r, z, vpa, composition, collisions,
                              evolve_ppar)
        #return ios(ff_io, mom_io, fields_io), cdf
        return ios(mom_io, fields_io), cdf
    end
    # For other processes in the block, return (nothing, nothing)
    return nothing, nothing
end

"""
setup file i/o for netcdf
"""
function setup_netcdf_io(prefix, r, z, vpa, composition, collisions, evolve_ppar)
    # the netcdf file will be given by output_dir/run_name with .cdf appended
    filename = string(prefix,".cdf")
    # if a netcdf file with the requested name already exists, remove it
    isfile(filename) && rm(filename)
    # create the new NetCDF file
    fid = NCDataset(filename,"c")
    # write a header to the NetCDF file
    fid.attrib["file_info"] = "This is a NetCDF file containing output data from the moment_kinetics code"
    ### define coordinate dimensions ###
    # define the vpa dimension
    defDim(fid, "nvpa", vpa.n)
    # define the z dimension
    defDim(fid, "nz", z.n)
    # define the r dimension
    defDim(fid, "nr", r.n)
    # define the species dimension
    defDim(fid, "n_species", composition.n_species)
    # define the ion species dimension
    defDim(fid, "n_ion_species", composition.n_ion_species)
    # define the neutral species dimension
    defDim(fid, "n_neutral_species", composition.n_neutral_species)
    # define the time dimension, with an expandable size (denoted by Inf)
    defDim(fid, "ntime", Inf)
    ### create and write static variables to file ###
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
    # create and write the "evolve_ppar" variable to file
    varname = "evolve_ppar"
    attributes = Dict("description" => "flag indicating if the parallel pressure is separately evolved")
    vartype = mk_int
    dims = ("n_species",)
    var = defVar(fid, varname, vartype, dims, attrib=attributes)
    var[:] = evolve_ppar
    ### create variables for time-dependent quantities and store them ###
    ### in a struct for later access ###
    # create the "time" variable
    varname = "time"
    attributes = Dict("description" => "time")
    dims = ("ntime",)
    vartype = mk_float
    cdf_time = defVar(fid, varname, vartype, dims, attrib=attributes)
    # create the "f" variable
    varname = "f"
    attributes = Dict("description" => "distribution function")
    vartype = mk_float
    dims = ("nvpa","nz","nr","n_species","ntime")
    cdf_f = defVar(fid, varname, vartype, dims, attrib=attributes)
    # create variables that are floats with data in the z and time dimensions
    vartype = mk_float
    dims = ("nz","nr","ntime")
    # create the "phi" variable, which will contain the electrostatic potential
    varname = "phi"
    attributes = Dict("description" => "electrostatic potential",
                      "units" => "Te/e")
    cdf_phi = defVar(fid, varname, vartype, dims, attrib=attributes)
    # create variables that are floats with data in the z, species and time dimensions
    vartype = mk_float
    dims = ("nz","nr","n_species","ntime")
    # create the "density" variable, which will contain the species densities
    varname = "density"
    attributes = Dict("description" => "species density",
                      "units" => "Ne")
    cdf_density = defVar(fid, varname, vartype, dims, attrib=attributes)
    # create the "parallel_flow" variable, which will contain the species parallel flows
    varname = "parallel_flow"
    attributes = Dict("description" => "species parallel flow",
                      "units" => "sqrt(2*Te/ms)")
    cdf_upar = defVar(fid, varname, vartype, dims, attrib=attributes)
    # create the "parallel_pressure" variable, which will contain the species parallel pressures
    varname = "parallel_pressure"
    attributes = Dict("description" => "species parallel pressure",
                      "units" => "Ne*Te")
    cdf_ppar = defVar(fid, varname, vartype, dims, attrib=attributes)
    # create the "parallel_heat_flux" variable, which will contain the species parallel heat fluxes
    varname = "parallel_heat_flux"
    attributes = Dict("description" => "species parallel heat flux",
                      "units" => "Ne*Te*vth")
    cdf_qpar = defVar(fid, varname, vartype, dims, attrib=attributes)
    # create the "thermal_speed" variable, which will contain the species thermal speed
    varname = "thermal_speed"
    attributes = Dict("description" => "species thermal speed",
                      "units" => "vth")
    cdf_vth = defVar(fid, varname, vartype, dims, attrib=attributes)

    # create a struct that stores the variables and other info needed for
    # writing to the netcdf file during run-time
    return netcdf_info(fid, cdf_time, cdf_f, cdf_phi, cdf_density, cdf_upar,
                       cdf_ppar, cdf_qpar, cdf_vth)
end

"""
close all opened output files
"""
function finish_file_io(io, cdf)
    @serial_region begin
        # Only read/write from first process in each 'block'

        # get the fields in the ios struct
        io_fields = fieldnames(typeof(io))
        for i ∈ 1:length(io_fields)
            close(getfield(io, io_fields[i]))
        end
        close(cdf.fid)
    end
    return nothing
end

"""
"""
function write_data_to_ascii(ff, moments, fields, vpa, z, r, t, n_species, io)
    @serial_region begin
        # Only read/write from first process in each 'block'

        #write_f_ascii(ff, z, vpa, t, io.ff)
        write_moments_ascii(moments, z, r, t, n_species, io.moments)
        write_fields_ascii(fields, z, r, t, io.fields)
    end
    return nothing
end

"""
write the function f(z,vpa) at this time slice
"""
function write_f_ascii(f, z, vpa, t, io)
    @serial_region begin
        # Only read/write from first process in each 'block'

        @inbounds begin
            n_species = size(f,3)
            for is ∈ 1:n_species
                for j ∈ 1:vpa.n
                    for i ∈ 1:z.n
                        println(io,"t: ", t, "   spec: ", is, ",   z: ", z.grid[i],
                            ",  vpa: ", vpa.grid[j], ",   f: ", f[i,j,is])
                    end
                    println(io)
                end
                println(io)
            end
            println(io)
        end
    end
    return nothing
end

"""
write moments of the distribution function f(z,vpa) at this time slice
"""
function write_moments_ascii(mom, z, r, t, n_species, io)
    @serial_region begin
        # Only read/write from first process in each 'block'

        @inbounds begin
            for is ∈ 1:n_species
                for ir ∈ 1:r.n
                    for iz ∈ 1:z.n
                        println(io,"t: ", t, "   species: ", is, "   r: ", r.grid[ir], "   z: ", z.grid[iz],
                            "  dens: ", mom.dens[iz,ir,is], "   upar: ", mom.upar[iz,ir,is],
                            "   ppar: ", mom.ppar[iz,ir,is], "   qpar: ", mom.qpar[iz,ir,is])
                    end
                end
            end
        end
        println(io,"")
    end
    return nothing
end

"""
write electrostatic potential at this time slice
"""
function write_fields_ascii(flds, z, r, t, io)
    @serial_region begin
        # Only read/write from first process in each 'block'

        @inbounds begin
            for ir ∈ 1:r.n
                for iz ∈ 1:z.n
                    println(io,"t: ", t, "   r: ", r.grid[ir],"   z: ", z.grid[iz], "  phi: ", flds.phi[iz,ir])
                end
            end
        end
        println(io,"")
    end
    return nothing
end

"""
write time-dependent data to the netcdf file
"""
function write_data_to_binary(ff, moments, fields, t, n_species, cdf, t_idx)
    @serial_region begin
        # Only read/write from first process in each 'block'

        # add the time for this time slice to the netcdf file
        cdf.time[t_idx] = t
        # add the distribution function data at this time slice to the netcdf file
        cdf.f[:,:,:,:,t_idx] = ff
        # add the electrostatic potential data at this time slice to the netcdf file
        cdf.phi[:,:,t_idx] = fields.phi
        # add the density data at this time slice to the netcdf file
        for is ∈ 1:n_species
            cdf.density[:,:,:,t_idx] = moments.dens
            cdf.parallel_flow[:,:,:,t_idx] = moments.upar
            cdf.parallel_pressure[:,:,:,t_idx] = moments.ppar
            cdf.parallel_heat_flux[:,:,:,t_idx] = moments.qpar
            cdf.thermal_speed[:,:,:,t_idx] = moments.vth
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

end
