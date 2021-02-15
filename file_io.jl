module file_io

export input_option_error
export open_output_file
export setup_file_io, finish_file_io
export write_data_to_ascii
export write_data_to_binary

using NCDatasets
using type_definitions: mk_float

# structure containing the various input/output streams
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
# structure containing the data/metadata needed for netcdf file i/o
struct netcdf_info{t_type, zvpast_type, zt_type, zst_type}
    # file identifier for the netcdf file to which data is written
    fid::NCDataset
    # handle for the time variable
    time::t_type
    # handle for the distribution function variable
    f::zvpast_type
    # handle for the electrostatic potential variable
    phi::zt_type
    # handle for the species density
    density::zst_type
    # handle for the species parallel flow
    parallel_flow::zst_type
    # handle for the species parallel pressure
    parallel_pressure::zst_type
end
# open the necessary output files
function setup_file_io(output_dir, run_name, z, vpa, composition)
    # check to see if output_dir exists in the current directory
    # if not, create it
    isdir(output_dir) || mkdir(output_dir)
    out_prefix = string(output_dir, "/", run_name)
    #ff_io = open_output_file(out_prefix, "f_vs_t")
    mom_io = open_output_file(out_prefix, "moments_vs_t")
    fields_io = open_output_file(out_prefix, "fields_vs_t")
    cdf = setup_netcdf_io(out_prefix, z, vpa, composition)
    #return ios(ff_io, mom_io, fields_io), cdf
    return ios(mom_io, fields_io), cdf
end
# setup file i/o for netcdf
function setup_netcdf_io(prefix, z, vpa, composition)
    # the netcdf file will be given by output_dir/run_name with .cdf appended
    filename = string(prefix,".cdf")
    # if a netcdf file with the requested name already exists, remove it
    isfile(filename) && rm(filename)
    # create the new NetCDF file
    fid = NCDataset(filename,"c")
    # write a header to the NetCDF file
    fid.attrib["file_info"] = "This is a NetCDF file containing output data from the moment_kinetics code"
    ### define coordinate dimensions ###
    # define the z dimension
    defDim(fid, "nz", z.n)
    # define the vpa dimension
    defDim(fid, "nvpa", vpa.n)
    # define the species dimension
    defDim(fid, "n_species", composition.n_species)
    # define the ion species dimension
    defDim(fid, "n_ion_species", composition.n_ion_species)
    # define the neutral species dimension
    defDim(fid, "n_neutral_species", composition.n_neutral_species)
    # define the time dimension, with an expandable size (denoted by Inf)
    defDim(fid, "ntime", Inf)
    ### create and write static variables to file ###
    # create and write the "z" variable to file
    varname = "z"
    attributes = Dict("description" => "parallel coordinate")
    dims = ("nz",)
    vartype = mk_float
    var = defVar(fid, varname, vartype, dims, attrib=attributes)
    var[:] = z.grid
    # create and write the "vpa" variable to file
    varname = "vpa"
    attributes = Dict("description" => "parallel velocity")
    dims = ("nvpa",)
    vartype = mk_float
    var = defVar(fid, varname, vartype, dims, attrib=attributes)
    var[:] = vpa.grid
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
    dims = ("nz","nvpa","n_species","ntime")
    cdf_f = defVar(fid, varname, vartype, dims, attrib=attributes)
    # create variables that are floats with data in the z and time dimensions
    vartype = mk_float
    dims = ("nz","ntime")
    # create the "phi" variable, which will contain the electrostatic potential
    varname = "phi"
    attributes = Dict("description" => "electrostatic potential",
                      "units" => "Te/e")
    cdf_phi = defVar(fid, varname, vartype, dims, attrib=attributes)
    # create variables that are floats with data in the z, species and time dimensions
    vartype = mk_float
    dims = ("nz","n_species","ntime")
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
    # create a struct that stores the variables and other info needed for
    # writing to the netcdf file during run-time
    t_type = typeof(cdf_time)
    zvpast_type = typeof(cdf_f)
    zt_type = typeof(cdf_phi)
    zst_type = typeof(cdf_density)
    return netcdf_info{t_type, zvpast_type, zt_type, zst_type}(fid, cdf_time, cdf_f,
        cdf_phi, cdf_density, cdf_upar, cdf_ppar)
end
# close all opened output files
function finish_file_io(io, cdf)
    # get the fields in the ios struct
    io_fields = fieldnames(typeof(io))
    for i ∈ 1:length(io_fields)
        close(getfield(io, io_fields[i]))
    end
    close(cdf.fid)
    return nothing
end
function write_data_to_ascii(ff, moments, fields, z, vpa, t, n_species, io)
    #write_f_ascii(ff, z, vpa, t, io.ff)
    write_moments_ascii(moments, z, t, n_species, io.moments)
    write_fields_ascii(fields, z, t, io.fields)
end
# write the function f(z,vpa) at this time slice
function write_f_ascii(f, z, vpa, t, io)
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
    return nothing
end
# write moments of the distribution function f(z,vpa) at this time slice
function write_moments_ascii(mom, z, t, n_species, io)
    @inbounds begin
        for is ∈ 1:n_species
            for i ∈ 1:z.n
                println(io,"t: ", t, "   species: ", is, "   z: ", z.grid[i],
                    "  dens: ", mom.dens[i,is], "   upar: ", mom.upar[i,is],
                    "   ppar: ", mom.ppar[i,is])
            end
        end
    end
    println(io,"")
    return nothing
end
# write electrostatic potential at this time slice
function write_fields_ascii(flds, z, t, io)
    @inbounds begin
        for i ∈ 1:z.n
            println(io,"t: ", t, ",   z: ", z.grid[i], "  phi: ", flds.phi[i])
        end
    end
    println(io,"")
    return nothing
end
# write time-dependent data to the netcdf file
function write_data_to_binary(ff, moments, fields, t, n_species, cdf, t_idx)
    # add the time for this time slice to the netcdf file
    cdf.time[t_idx] = t
    # add the distribution function data at this time slice to the netcdf file
    cdf.f[:,:,:,t_idx] = ff
    # add the electrostatic potential data at this time slice to the netcdf file
    cdf.phi[:,t_idx] = fields.phi
    # add the density data at this time slice to the netcdf file
    for is ∈ 1:n_species
        cdf.density[:,:,t_idx] = moments.dens
        cdf.parallel_flow[:,:,t_idx] = moments.upar
        cdf.parallel_pressure[:,:,t_idx] = moments.ppar
    end
end
# accepts an option name which has been identified as problematic and returns
# an appropriate error message
function input_option_error(option_name, input)
    msg = string("'",input,"'")
    msg = string(msg, " is not a valid ", option_name)
    error(msg)
    return nothing
end
# opens an output file with the requested prefix and extension
# and returns the corresponding io stream (identifier)
function open_output_file(prefix, ext)
    str = string(prefix,".",ext)
    return io = open(str,"w")
end

end
