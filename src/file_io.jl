"""
"""
module file_io

export input_option_error
export open_output_file, open_ascii_output_file
export setup_file_io, finish_file_io
export write_data_to_ascii
export write_data_to_netcdf, write_data_to_hdf5

using ..communication: _block_synchronize, iblock_index, block_size, global_size
using ..coordinates: coordinate
using ..debugging
using ..input_structs
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
structure containing the data/metadata needed for binary file i/o
moments & fields only
"""
struct io_moments_info{Tfile, Ttime, Tphi, Tmomi, Tmomn}
     # file identifier for the binary file to which data is written
    fid::Tfile
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
structure containing the data/metadata needed for binary file i/o
distribution function data only
"""
struct io_dfns_info{Tfile, Ttime, Tfi, Tfn}
     # file identifier for the binary file to which data is written
    fid::Tfile
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

        io_moments = setup_moments_io(out_prefix, io_input.binary_format, r, z,
                                      composition, collisions)
        io_dfns = setup_dfns_io(out_prefix, io_input.binary_format, r, z, vperp, vpa,
                                vzeta, vr, vz, composition, collisions)

        return ascii, io_moments, io_dfns
    end
    # For other processes in the block, return (nothing, nothing, nothing)
    return nothing, nothing, nothing
end

"""
    write_single_value!(file_or_group, name, value; description=nothing)

Write a single variable to a file or group. If a description is passed, add as an
attribute of the variable.
"""
function write_single_value!() end

"""
write some overview information for the simulation to the hdf5 file
"""
function write_overview!(fid, composition, collisions)
    overview = create_io_group(fid, "overview")
    write_single_value!(overview, "nspecies", composition.n_species,
                        description="total number of evolved plasma species")
    write_single_value!(overview, "n_ion_species", composition.n_ion_species,
                        description="number of evolved ion species")
    write_single_value!(overview, "n_neutral_species", composition.n_neutral_species,
                        description="number of evolved neutral species")
    write_single_value!(overview, "T_e", composition.T_e,
                        description="fixed electron temperature")
    write_single_value!(overview, "charge_exchange_frequency", collisions.charge_exchange,
                        description="quantity related to the charge exchange frequency")
    write_single_value!(overview, "ionization_frequency", collisions.ionization,
                        description="quantity related to the ionization frequency")
end

"""
Define coords group for coordinate information in the output file and write information
about spatial coordinate grids
"""
function define_spatial_coordinates!(fid, z, r)
    # create the "coords" group that will contain coordinate information
    coords = create_io_group(fid, "coords")
    # create the "z" sub-group of "coords" that will contain z coordinate info,
    # including total number of grid points and grid point locations
    define_coordinate!(coords, z, "z", "spatial coordinate z")
    # create the "r" sub-group of "coords" that will contain r coordinate info,
    # including total number of grid points and grid point locations
    define_coordinate!(coords, r, "r", "spatial coordinate r")

    # Write variable recording the index of the block within the global domain
    # decomposition
    write_single_value!(coords, "iblock", iblock_index[],
                        description="index of this zr block")

    # Write variable recording the total number of blocks in the global domain
    # decomposition
    write_single_value!(coords, "nblocks", global_size[]÷block_size[],
                        description="number of zr blocks")

    return coords
end

"""
Add to coords group in output file information about vspace coordinate grids
"""
function add_vspace_coordinates!(coords, vz, vr, vzeta, vpa, vperp)
    # create the "vz" sub-group of "coords" that will contain vz coordinate info,
    # including total number of grid points and grid point locations
    define_coordinate!(coords, vz, "vz", "velocity coordinate v_z")
    # create the "vr" sub-group of "coords" that will contain vr coordinate info,
    # including total number of grid points and grid point locations
    define_coordinate!(coords, vr, "vr", "velocity coordinate v_r")
    # create the "vzeta" sub-group of "coords" that will contain vzeta coordinate info,
    # including total number of grid points and grid point locations
    define_coordinate!(coords, vzeta, "vzeta", "velocity coordinate v_zeta")
    # create the "vpa" sub-group of "coords" that will contain vpa coordinate info,
    # including total number of grid points and grid point locations
    define_coordinate!(coords, vpa, "vpa", "velocity coordinate v_parallel")
    # create the "vperp" sub-group of "coords" that will contain vperp coordinate info,
    # including total number of grid points and grid point locations
    define_coordinate!(coords, vperp, "vperp", "velocity coordinate v_perp")

    return nothing
end

"""
define a sub-group for each code coordinate and write to output file
"""
function define_coordinate!(parent, coord, coord_name, description)
    # create the "group" sub-group of "parent" that will contain coord_str coordinate info
    group = create_io_group(parent, coord_name, description=description)

    # write the number of local grid points for this coordinate to variable "n_local"
    # within "coords/coord_name" group
    write_single_value!(group, "n_local", coord.n;
                        description="number of local $coord_name grid points")

    # write the number of global grid points for this coordinate to variable "n_local"
    # within "coords/coord_name" group
    write_single_value!(group, "n_global", coord.n_global;
                        description="total number of $coord_name grid points")

    # write the rank in the coord-direction of this process
    write_single_value!(group, "irank", coord.irank,
                        description="rank of this block in the $(coord.name) grid communicator")

    # write the global length of this coordinate to variable "L"
    # within "coords/coord_name" group
    write_single_value!(group, "L", coord.L;
                        description="box length in $coord_name")

    # write the locations of this coordinate's grid points to variable "grid" within "coords/coord_name" group
    write_single_value!(group, "grid", coord.grid, coord;
                        description="$coord_name values sampled by the $coord_name grid")

    # write the integration weights attached to each coordinate grid point
    write_single_value!(group, "wgts", coord.wgts, coord;
                        description="integration weights associated with the $coord_name grid points")

    return group
end

"""
    create_dynamic_variable!(file_or_group, name, type, coords::coordinate...;
                             nspecies=1, description=nothing, units=nothing)

Create a time-evolving variable in `file_or_group` named `name` of type `type`. `coords`
are the coordinates corresponding to the dimensions of the array, in the order of the
array dimensions. The species dimension does not have a `coordinate`, so the number of
species is passed as `nspecies`. A description and/or units can be added with the keyword
arguments.
"""
function create_dynamic_variable!() end

"""
define dynamic (time-evolving) moment variables for writing to the hdf5 file
"""
function define_dynamic_moment_variables!(fid, n_ion_species, n_neutral_species,
                                          r::coordinate, z::coordinate)
    dynamic = create_io_group(fid, "dynamic_data", description="time evolving variables")

    io_time = create_dynamic_variable!(dynamic, "time", mk_float; description="simulation time")

    # io_phi is the handle referring to the electrostatic potential phi
    io_phi = create_dynamic_variable!(dynamic, "phi", mk_float, z, r;
                                      description="electrostatic potential",
                                      units="T_ref/e")
    # io_Er is the handle for the radial component of the electric field
    io_Er = create_dynamic_variable!(dynamic, "Er", mk_float, z, r;
                                     description="radial electric field",
                                     units="T_ref/e L_ref")
    # io_Ez is the handle for the zed component of the electric field
    io_Ez = create_dynamic_variable!(dynamic, "Ez", mk_float, z, r;
                                     description="vertical electric field",
                                     units="T_ref/e L_ref")

    # io_density is the handle for the ion particle density
    io_density = create_dynamic_variable!(dynamic, "density", mk_float, z, r;
                                          n_ion_species=n_ion_species,
                                          description="charged species density",
                                          units="n_ref")

    # io_upar is the handle for the ion parallel flow density
    io_upar = create_dynamic_variable!(dynamic, "parallel_flow", mk_float, z, r;
                                       n_ion_species=n_ion_species,
                                       description="charged species parallel flow",
                                       units="c_ref = sqrt(2*T_ref/mi)")

    # io_ppar is the handle for the ion parallel pressure
    io_ppar = create_dynamic_variable!(dynamic, "parallel_pressure", mk_float, z, r;
                                       n_ion_species=n_ion_species,
                                       description="charged species parallel pressure",
                                       units="n_ref*T_ref")

    # io_qpar is the handle for the ion parallel heat flux
    io_qpar = create_dynamic_variable!(dynamic, "parallel_heat_flux", mk_float, z, r;
                                       n_ion_species=n_ion_species,
                                       description="charged species parallel heat flux",
                                       units="n_ref*T_ref*c_ref")

    # io_vth is the handle for the ion thermal speed
    io_vth = create_dynamic_variable!(dynamic, "thermal_speed", mk_float, z, r;
                                      n_ion_species=n_ion_species,
                                      description="charged species thermal speed",
                                      units="c_ref")

    # io_density_neutral is the handle for the neutral particle density
    io_density_neutral = create_dynamic_variable!(dynamic, "density_neutral", mk_float, z, r;
                                                  n_neutral_species=n_neutral_species,
                                                  description="neutral species density",
                                                  units="n_ref")

    # io_uz_neutral is the handle for the neutral z momentum density
    io_uz_neutral = create_dynamic_variable!(dynamic, "uz_neutral", mk_float, z, r;
                                             n_neutral_species=n_neutral_species,
                                             description="neutral species mean z velocity",
                                             units="c_ref = sqrt(2*T_ref/mi)")

    # io_pz_neutral is the handle for the neutral species zz pressure
    io_pz_neutral = create_dynamic_variable!(dynamic, "pz_neutral", mk_float, z, r;
                                             n_neutral_species=n_neutral_species,
                                             description="neutral species mean zz pressure",
                                             units="n_ref*T_ref")

    # io_qz_neutral is the handle for the neutral z heat flux
    io_qz_neutral = create_dynamic_variable!(dynamic, "qz_neutral", mk_float, z, r;
                                             n_neutral_species=n_neutral_species,
                                             description="neutral species z heat flux",
                                             units="n_ref*T_ref*c_ref")

    # io_thermal_speed_neutral is the handle for the neutral thermal speed
    io_thermal_speed_neutral = create_dynamic_variable!(
        dynamic, "thermal_speed_neutral", mk_float, z, r;
        n_neutral_species=n_neutral_species,
        description="neutral species thermal speed", units="c_ref")

    return io_moments_info(fid, io_time, io_phi, io_Er, io_Ez, io_density, io_upar,
                           io_ppar, io_qpar, io_vth, io_density_neutral, io_uz_neutral,
                           io_pz_neutral, io_qz_neutral, io_thermal_speed_neutral)
end

"""
define dynamic (time-evolving) distribution function variables for writing to the output
file
"""
function define_dynamic_dfn_variables!(fid, r, z, vperp, vpa, vzeta, vr, vz,
                                       n_ion_species, n_neutral_species)

    dynamic = create_io_group(fid, "dynamic_data", description="time evolving variables")

    io_time = create_dynamic_variable!(dynamic, "time", mk_float; description="simulation time")

    # io_f is the handle for the ion pdf
    io_f = create_dynamic_variable!(dynamic, "f", mk_float, vpa, vperp, z, r;
                                    n_ion_species=n_ion_species,
                                    description="charged species distribution function")

    # io_f_neutral is the handle for the neutral pdf
    io_f_neutral = create_dynamic_variable!(dynamic, "f_neutral", mk_float, vz, vr, vzeta, z, r;
                                            n_neutral_species=n_neutral_species,
                                            description="neutral species distribution function")

    return io_dfns_info(fid, io_time, io_f, io_f_neutral)
end

"""
Add an attribute to a file, group or variable
"""
function add_attribute!() end

"""
Open an output file, selecting the backend based on io_option
"""
function open_output_file(prefix, binary_format)
    if binary_format == hdf5
        return open_output_file_hdf5(prefix)
    elseif binary_format == netcdf
        return open_output_file_netcdf(prefix)
    else
        error("Unsupported I/O format $binary_format")
    end
end

"""
setup file i/o for moment variables
"""
function setup_moments_io(prefix, binary_format, r, z, composition, collisions)
    fid = open_output_file(string(prefix, ".moments"), binary_format)

    # write a header to the output file
    add_attribute!(fid, "file_info", "Output moments data from the moment_kinetics code")

    # write some overview information to the output file
    write_overview!(fid, composition, collisions)

    ### define coordinate dimensions ###
    define_spatial_coordinates!(fid, z, r)

    ### create variables for time-dependent quantities and store them ###
    ### in a struct for later access ###
    io_moments = define_dynamic_moment_variables!(
        fid, composition.n_ion_species, composition.n_neutral_species, r, z)

    return io_moments
end

"""
setup file i/o for distribution function variables
"""
function setup_dfns_io(prefix, binary_format, r, z, vperp, vpa, vzeta, vr, vz, composition,
                       collisions)

    fid = open_output_file(string(prefix, ".dfns"), binary_format)

    # write a header to the output file
    add_attribute!(fid, "file_info",
                   "Output distribution function data from the moment_kinetics code")

    # write some overview information to the hdf5 file
    write_overview!(fid, composition, collisions)

    ### define coordinate dimensions ###
    coords_group = define_spatial_coordinates!(fid, z, r)
    add_vspace_coordinates!(coords_group, vz, vr, vzeta, vpa, vperp)

    ### create variables for time-dependent quantities and store them ###
    ### in a struct for later access ###
    io_dfns = define_dynamic_dfn_variables!(
        fid, r, z, vperp, vpa, vzeta, vr, vz, composition.n_ion_species,
        composition.n_neutral_species)

    return io_dfns
end

"""
    append_to_dynamic_var(io_var, data, t_idx)

Append `data` to the dynamic variable `io_var`. The time-index of the data being appended
is `t_idx`. 
"""
function append_to_dynamic_var() end

"""
write time-dependent moments data to the binary output file
"""
function write_moments_data_to_binary(moments, fields, t, n_ion_species,
                                      n_neutral_species, io_moments, t_idx)
    @serial_region begin
        # Only read/write from first process in each 'block'

        # add the time for this time slice to the hdf5 file
        append_to_dynamic_var(io_moments.time, t, t_idx)

        # add the electrostatic potential and electric field components at this time slice to the hdf5 file
        append_to_dynamic_var(io_moments.phi, fields.phi, t_idx)
        append_to_dynamic_var(io_moments.Er, fields.Er, t_idx)
        append_to_dynamic_var(io_moments.Ez, fields.Ez, t_idx)

        # add the density data at this time slice to the output file
        append_to_dynamic_var(io_moments.density, moments.charged.dens, t_idx)
        append_to_dynamic_var(io_moments.parallel_flow, moments.charged.upar, t_idx)
        append_to_dynamic_var(io_moments.parallel_pressure, moments.charged.ppar, t_idx)
        append_to_dynamic_var(io_moments.parallel_heat_flux, moments.charged.qpar, t_idx)
        append_to_dynamic_var(io_moments.thermal_speed, moments.charged.vth, t_idx)
        if n_neutral_species > 0
            append_to_dynamic_var(io_moments.density_neutral, moments.neutral.dens, t_idx)
            append_to_dynamic_var(io_moments.uz_neutral, moments.neutral.uz, t_idx)
            append_to_dynamic_var(io_moments.pz_neutral, moments.neutral.pz, t_idx)
            append_to_dynamic_var(io_moments.qz_neutral, moments.neutral.qz, t_idx)
            append_to_dynamic_var(io_moments.thermal_speed_neutral, moments.neutral.vth, t_idx)
        end
    end
    return nothing
end

"""
write time-dependent distribution function data to the binary output file
"""
function write_dfns_data_to_binary(ff, ff_neutral, t, n_ion_species, n_neutral_species,
                                   io_dfns, t_idx)
    @serial_region begin
        # Only read/write from first process in each 'block'

        # add the time for this time slice to the hdf5 file
        append_to_dynamic_var(io_dfns.time, t, t_idx)

        # add the distribution function data at this time slice to the output file
        append_to_dynamic_var(io_dfns.f, ff, t_idx)
        if n_neutral_species > 0
            append_to_dynamic_var(io_dfns.f_neutral, ff_neutral, t_idx)
        end
    end
    return nothing
end

@debug_shared_array begin
    # Special versions when using DebugMPISharedArray to avoid implicit conversion to
    # Array, which is forbidden.
    function write_moments_data_to_binary(moments, fields, t, n_ion_species,
            n_neutral_species, io_moments, t_idx)
        @serial_region begin
            # Only read/write from first process in each 'block'

            # add the time for this time slice to the hdf5 file
            append_to_dynamic_var(io_moments.time, t, t_idx)

            # add the electrostatic potential and electric field components at this time slice to the hdf5 file
            append_to_dynamic_var(io_moments.phi.data, field.phi, t_idx)
            append_to_dynamic_var(io_moments.Er.data, field.Er, t_idx)
            append_to_dynamic_var(io_moments.Ez.data, field.Ez, t_idx)

            # add the density data at this time slice to the output file
            append_to_dynamic_var(io_moments.density.data, moments.charged.dens, t_idx)
            append_to_dynamic_var(io_moments.parallel_flow.data, moments.charged.upar, t_idx)
            append_to_dynamic_var(io_moments.parallel_pressure.data, moments.charged.ppar, t_idx)
            append_to_dynamic_var(io_moments.parallel_heat_flux.data, moments.charged.qpar, t_idx)
            append_to_dynamic_var(io_moments.thermal_speed, moments.charged.vth.data, t_idx)
            if n_neutral_species > 0
                append_to_dynamic_var(io_moments.density_neutral, moments.neutral.dens.data, t_idx)
                append_to_dynamic_var(io_moments.uz_neutral, moments.neutral.uz.data, t_idx)
                append_to_dynamic_var(io_moments.pz_neutral, moments.neutral.pz.data, t_idx)
                append_to_dynamic_var(io_moments.qz_neutral, moments.neutral.qz.data, t_idx)
                append_to_dynamic_var(io_moments.thermal_speed_neutral, moments.neutral.vth.data, t_idx)
            end
        end
        return nothing
    end
end

@debug_shared_array begin
    # Special versions when using DebugMPISharedArray to avoid implicit conversion to
    # Array, which is forbidden.
    function write_dfns_data_to_binary(ff::DebugMPISharedArray, ff_neutral::DebugMPISharedArray,
            t, n_ion_species, n_neutral_species, h5::hdf5_dfns_info, t_idx)
        @serial_region begin
            # Only read/write from first process in each 'block'

            # add the time for this time slice to the hdf5 file
            append_to_dynamic_var(io_dfns.time, t, t_idx)

            # add the distribution function data at this time slice to the output file
            append_to_dynamic_var(io_dfns.f, ff.data, t_idx)
            if n_neutral_species > 0
                append_to_dynamic_var(io_dfns.f_neutral, ff_neutral.data, t_idx)
            end
        end
        return nothing
    end
end

"""
close all opened output files
"""
function finish_file_io(ascii_io::Union{ascii_ios,Nothing},
                        binary_moments::Union{io_moments_info,Nothing},
                        binary_dfns::Union{io_dfns_info,Nothing})
    @serial_region begin
        # Only read/write from first process in each 'block'

        if ascii_io !== nothing
            # get the fields in the ascii_ios struct
            ascii_io_fields = fieldnames(typeof(ascii_io))
            for x ∈ ascii_io_fields
                io = getfield(ascii_io, x)
                if io !== nothing
                    close(io)
                end
            end
        end
        if binary_moments !== nothing
            close(binary_moments.fid)
        end
        if binary_dfns !== nothing
            close(binary_dfns.fid)
        end
    end
    return nothing
end

# Include the possible implementations of binary I/O functions
include("file_io_netcdf.jl")
include("file_io_hdf5.jl")

"""
"""
function write_data_to_ascii(moments, fields, vpa, vperp, z, r, t, n_ion_species,
                             n_neutral_species, ascii_io::Union{ascii_ios,Nothing})
    if ascii_io === nothing || ascii_io.moments_charged === nothing
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
function open_ascii_output_file(prefix, ext)
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
            define_spatial_dimensions_netcdf!(fid, nz, nr, n_species)
            define_vspace_dimensions_netcdf!(fid, nvpa, nvperp)
            ### create variables for time-dependent quantities and store them ###
            ### in a struct for later access ###
            cdf_time, cdf_phi, cdf_Er, cdf_Ez, cdf_density, cdf_upar, cdf_ppar, cdf_qpar, cdf_vth,
            cdf_density_neutral, cdf_uz_neutral, cdf_pz_neutral, cdf_qz_neutral, cdf_vth_neutral =
                define_dynamic_moment_variables!(fid)
            cdf_dfns_time, cdf_charged_f, cdf_neutral_f =
                define_dynamic_dfns_variables_netcdf!(fid)

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
            debug_output_file = (fid=fid, time=cdf_time, charged_f=cdf_charged_f,
                                 phi=cdf_phi, Er=cdf_Er, Ez=cdf_Ez, density=cdf_density,
                                 parallel_flow=cdf_upar, parallel_pressure=cdf_ppar,
                                 parallel_heat_flux=cdf_qpar, thermal_speed=cdf_vth,
                                 neutral_f=cdf_neutral_f,
                                 density_neutral=cdf_density_neutral,
                                 uz_neutral=cdf_uz_neutral, pz_neutral=cdf_pz_neutral,
                                 qz_neutral=cdf_qz_neutral,
                                 thermal_speed_neutral=cdf_vth_neutral, istage=cdf_istage,
                                 label=cdf_label)
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
            debug_output_file.charged_f[:,:,:,:,:,debug_output_counter[]] .= 0.0
        else
            debug_output_file.charged_f[:,:,:,:,:,debug_output_counter[]] = ff
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
