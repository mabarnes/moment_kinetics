module file_io

export input_option_error
export open_output_file
export setup_file_io, finish_file_io
export write_f, write_moments, write_fields

# structure containing the various input/output streams
struct ios
    # corresponds to the ascii file to which the distribution function is written
    ff::IOStream
    # corresponds to the ascii file to which velocity space moments of the
    # distribution function such as density and pressure are written
    moments::IOStream
    # corresponds to the ascii file to which electromagnetic fields
    # such as the electrostatic potential are written
    fields::IOStream
end
# open the necessary output files
function setup_file_io(run_name)
    ff_io = open_output_file(run_name, "f_vs_t")
    mom_io = open_output_file(run_name, "moments_vs_t")
    fields_io = open_output_file(run_name, "fields_vs_t")
    return ios(ff_io, mom_io, fields_io)
end
# close all opened output files
function finish_file_io(io)
    # get the fields in the ios struct
    io_fields = fieldnames(typeof(io))
    for i ∈ 1:length(io_fields)
        close(getfield(io, io_fields[i]))
    end
    return nothing
end
# write the function f(z,vpa) at this time slice
function write_f(f, z, vpa, t, io)
    @inbounds begin
        for j ∈ 1:vpa.n
            for i ∈ 1:z.n
                println(io,"t: ", t, ",   z: ", z.grid[i], ",  vpa: ", vpa.grid[j],
                    ",   f: ", f[i,j,1])
            end
            println(io)
        end
    end
    println(io,"")
    return nothing
end
# write moments of the distribution function f(z,vpa) at this time slice
function write_moments(mom, z, t, io)
    @inbounds begin
        for i ∈ 1:z.n
            println(io,"t: ", t, ",   z: ", z.grid[i], "  dens: ", mom.dens[i],
                ",   ppar: ", mom.ppar[i])
        end
    end
    println(io,"")
    return nothing
end
# write electrostatic potential at this time slice
function write_fields(flds, z, t, io)
    @inbounds begin
        for i ∈ 1:z.n
            println(io,"t: ", t, ",   z: ", z.grid[i], "  phi: ", flds.phi[i])
        end
    end
    println(io,"")
    return nothing
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
