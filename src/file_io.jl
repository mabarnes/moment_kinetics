"""
"""
module file_io

export input_option_error
export open_output_file
export setup_file_io, finish_file_io
export write_data_to_ascii
export write_data_to_binary

using NCDatasets
using ..communication: _block_synchronize
using ..looping
using ..moment_kinetics_structs: scratch_pdf, em_fields_struct
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

"""
structure containing the data/metadata needed for netcdf file i/o
"""
struct netcdf_info{Ttime, Tfi, Tphi, Tmom}
    # file identifier for the netcdf file to which data is written
    fid::NCDataset
    # handle for the time variable
    time::Ttime
    # handle for the distribution function variable
    f::Tfi
    # handle for the electrostatic potential variable
    phi::Tphi
    # handle for the species density
    density::Tmom
    # handle for the species parallel flow
    parallel_flow::Tmom
    # handle for the species parallel pressure
    parallel_pressure::Tmom
    # handle for the species parallel heat flux
    parallel_heat_flux::Tmom
    # handle for the species thermal speed
    thermal_speed::Tmom
end

"""
open the necessary output files
"""
function setup_file_io(output_dir, run_name, vpa, z, r, composition,
                       collisions, evolve_density, evolve_upar, evolve_ppar)
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
                              evolve_density, evolve_upar, evolve_ppar)
        #return ios(ff_io, mom_io, fields_io), cdf
        return ios(mom_io, fields_io), cdf
    end
    # For other processes in the block, return (nothing, nothing)
    return nothing, nothing
end

# Define the steps for creating a NetCDF file in utility functions so that they can be
# shared between `setup_netcdf_io()` and `debug_dump()`
"""
    define_dimensions!(fid, nvpa, nvperp, nz, nr, n_species, n_ion_species=nothing,
                       n_neutral_species=nothing)

Define dimensions for an output file.
"""
function define_dimensions!(fid, nvpa, nz, nr, n_species, n_ion_species=nothing,
                            n_neutral_species=nothing)
    # define the vpa dimension
    defDim(fid, "nvpa", nvpa)
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
    define_static_variables!(vpa,vperp,z,r,composition,collisions,evolve_density,evolve_upar,evolve_ppar)

Define static (i.e. time-independent) variables for an output file.
"""
function define_static_variables!(fid,vpa,z,r,composition,collisions,evolve_density,evolve_upar,evolve_ppar)
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
    # create and write the "evolve_density" variable to file
    varname = "evolve_density"
    attributes = Dict("description" => "flag indicating if the density is separately evolved")
    vartype = mk_int
    dims = ("n_species",)
    var = defVar(fid, varname, vartype, dims, attrib=attributes)
    var[:] = evolve_density
    # create and write the "evolve_upar" variable to file
    varname = "evolve_upar"
    attributes = Dict("description" => "flag indicating if the parallel flow is separately evolved")
    vartype = mk_int
    dims = ("n_species",)
    var = defVar(fid, varname, vartype, dims, attrib=attributes)
    var[:] = evolve_upar
    # create and write the "evolve_ppar" variable to file
    varname = "evolve_ppar"
    attributes = Dict("description" => "flag indicating if the parallel pressure is separately evolved")
    vartype = mk_int
    dims = ("n_species",)
    var = defVar(fid, varname, vartype, dims, attrib=attributes)
    var[:] = evolve_ppar

    return nothing
end

"""
    define_dynamic_variables!(fid)

Define dynamic (i.e. time-evolving) variables for an output file.
"""
function define_dynamic_variables!(fid)
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

    return cdf_time, cdf_f, cdf_phi, cdf_density, cdf_upar, cdf_ppar, cdf_qpar, cdf_vth
end

"""
setup file i/o for netcdf
"""
function setup_netcdf_io(prefix, r, z, vpa, composition, collisions, evolve_density,
                         evolve_upar, evolve_ppar)
    # the netcdf file will be given by output_dir/run_name with .cdf appended
    filename = string(prefix,".cdf")
    # if a netcdf file with the requested name already exists, remove it
    isfile(filename) && rm(filename)
    # create the new NetCDF file
    fid = NCDataset(filename,"c")
    # write a header to the NetCDF file
    fid.attrib["file_info"] = "This is a NetCDF file containing output data from the moment_kinetics code"
    ### define coordinate dimensions ###
    define_dimensions!(fid, vpa.n, z.n, r.n, composition.n_species,
                       composition.n_ion_species, composition.n_neutral_species)
    ### create and write static variables to file ###
    define_static_variables!(fid,vpa,z,r,composition,collisions,evolve_density,evolve_upar,evolve_ppar)
    ### create variables for time-dependent quantities and store them ###
    ### in a struct for later access ###
    cdf_time, cdf_f, cdf_phi, cdf_density, cdf_upar, cdf_ppar, cdf_qpar, cdf_vth =
        define_dynamic_variables!(fid)

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
        cdf.f[:,:,:,:,t_idx] .= ff
        # add the electrostatic potential data at this time slice to the netcdf file
        cdf.phi[:,:,t_idx] .= fields.phi
        # add the density data at this time slice to the netcdf file
        for is ∈ 1:n_species
            cdf.density[:,:,:,t_idx] .= moments.dens
            cdf.parallel_flow[:,:,:,t_idx] .= moments.upar
            cdf.parallel_pressure[:,:,:,t_idx] .= moments.ppar
            cdf.parallel_heat_flux[:,:,:,t_idx] .= moments.qpar
            cdf.thermal_speed[:,:,:,t_idx] .= moments.vth
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
Reload pdf and moments from an existing output file.
"""
function reload_evolving_fields!(pdf, moments, restart_filename, time_index,
                                 composition, r, z, vpa)
    code_time = 0.0
    begin_serial_region()
    @serial_region begin
        fid = NCDataset(restart_filename,"r")
        try
            if time_index < 0
                time_index = fid.dim["ntime"]
            end
            restart_n_species = fid.dim["n_species"]
            restart_nr = fid.dim["nr"]
            restart_nz = fid.dim["nz"]
            restart_nvpa = fid.dim["nvpa"]
            if restart_n_species != composition.n_species || restart_nr != r.n ||
                restart_nz != z.n || restart_nvpa != vpa.n

                error("Dimensions of restart file and input do not match.\n" *
                      "Restart file was n_species=$restart_n_species, nr=$restart_nr, " *
                      "nz=$restart_nz, nvpa=$restart_nvpa.\n" *
                      "Input file gave  n_species=$(composition.n_species), nr=$(r.n), " *
                      "nz=$(z.n), nvpa=$(vpa.n).")
            end

            code_time = fid["time"].var[time_index]
            pdf.norm .= fid["f"].var[:,:,:,:,time_index]
            moments.dens .= fid["density"].var[:,:,:,time_index]
            moments.upar .= fid["parallel_flow"].var[:,:,:,time_index]
            moments.ppar .= fid["parallel_pressure"].var[:,:,:,time_index]
            moments.qpar .= fid["parallel_heat_flux"].var[:,:,:,time_index]
            moments.vth .= fid["thermal_speed"].var[:,:,:,time_index]
        finally
            close(fid)
        end
    end

    return code_time
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

Intended to be called more frequently than `write_data_to_binary()`, possibly several
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

            (nvpa, nz, nr, n_species) = size(ff)
            # the netcdf file will be given by output_dir/run_name with .cdf appended
            filename = string("debug_output.cdf")
            # if a netcdf file with the requested name already exists, remove it
            isfile(filename) && rm(filename)
            # create the new NetCDF file
            fid = NCDataset(filename,"c")
            # write a header to the NetCDF file
            fid.attrib["file_info"] = "This is a NetCDF file containing debug output from the moment_kinetics code"
            ### define coordinate dimensions ###
            define_dimensions!(fid, nvpa, nz, nr, n_species)
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
            debug_output_file.f[:,:,:,:,debug_output_counter[]] .= 0.0
        else
            debug_output_file.f[:,:,:,:,debug_output_counter[]] = ff
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
