module moment_kinetics

export run_moment_kinetics

using MPI

# Include submodules from other source files
# Note that order of includes matters - things used in one module must already
# be defined
include("command_line_options.jl")
include("debugging.jl")
include("type_definitions.jl")
include("communication.jl")
include("looping.jl")
include("array_allocation.jl")
include("interpolation.jl")
include("clenshaw_curtis.jl")
include("chebyshev.jl")
include("finite_differences.jl")
include("quadrature.jl")
include("calculus.jl")
include("file_io.jl")
include("input_structs.jl")
include("coordinates.jl")
include("velocity_moments.jl")
include("em_fields.jl")
include("bgk.jl")
include("initial_conditions.jl")
include("semi_lagrange.jl")
include("advection.jl")
include("vpa_advection.jl")
include("z_advection.jl")
include("r_advection.jl")
include("charge_exchange.jl")
include("ionization.jl")
include("continuity.jl")
include("energy_equation.jl")
include("force_balance.jl")
include("source_terms.jl")
include("time_advance.jl")

include("moment_kinetics_input.jl")
include("scan_input.jl")

include("analysis.jl")
include("load_data.jl")
include("post_processing_input.jl")
include("post_processing.jl")

using TimerOutputs
using TOML

using .file_io: setup_file_io, finish_file_io
using .file_io: write_data_to_ascii, write_data_to_binary
using .command_line_options: get_options
using .communication
using .coordinates: define_coordinate
using .debugging
using .initial_conditions: init_pdf_and_moments
using .looping
using .moment_kinetics_input: mk_input, run_type, performance_test
using .time_advance: setup_time_advance!, time_advance!

@debug_detect_redundant_block_synchronize using ..communication: debug_detect_redundant_is_active

# main function that contains all of the content of the program
function run_moment_kinetics(to::TimerOutput, input_dict=Dict())
    # set up all the structs, etc. needed for a run
    mk_state = setup_moment_kinetics(input_dict)

    # solve the 1+1D kinetic equation to advance f in time by nstep time steps
    if run_type == performance_test
        @timeit to "time_advance" time_advance!(mk_state...)
    else
        time_advance!(mk_state...)
    end

    # clean up i/o and communications
    # last 2 elements of mk_state are `io` and `cdf`
    cleanup_moment_kinetics!(mk_state[end-1:end]...)

    if block_rank[] == 0 && run_type == performance_test
        # Print the timing information if this is a performance test
        display(to)
        println()
    end

    return nothing
end

# overload which takes a filename and loads input
function run_moment_kinetics(to::TimerOutput, input_filename::String)
    return run_moment_kinetics(to, TOML.parsefile(input_filename))
end
# overloads with no TimerOutput arguments
function run_moment_kinetics(input)
    return run_moment_kinetics(TimerOutput(), input)
end
function run_moment_kinetics()
    inputfile = get_options()["inputfile"]
    if inputfile == nothing
        run_moment_kinetics(Dict())
    else
        run_moment_kinetics(inputfile)
    end
end

# Perform all the initialization steps for a run.
function setup_moment_kinetics(input_dict::Dict)
    
    #print("got to here 1 \n")

    
    # Set up MPI
    initialize_comms!()

    #print("got to here 2 \n")
    
    input = mk_input(input_dict)
    # obtain input options from moment_kinetics_input.jl
    # and check input to catch errors
    run_name, output_dir, evolve_moments, t_input, z_input, r_input, vpa_input,
        composition, species, collisions, drive_input = input
    # initialize z grid and write grid point locations to file
    z = define_coordinate(z_input, composition)
    # initialize r grid and write grid point locations to file
    r = define_coordinate(r_input, composition)
    # initialize vpa grid and write grid point locations to file
    vpa = define_coordinate(vpa_input, composition)
    # Create loop range variables for shared-memory-parallel loops
    looping.setup_loop_ranges!(block_rank[], block_size[]; s=composition.n_species, r=r.n,
                               z=z.n, vpa=vpa.n)
    # initialize f(z,vpa) and the lowest three v-space moments (density(z), upar(z) and ppar(z)),
    # each of which may be evolved separately depending on input choices.
    
    #print("got to here 3 \n")
    
    pdf, moments = init_pdf_and_moments(vpa, z, r, composition, species, t_input.n_rk_stages, evolve_moments)
   
    #print("got to here 4 \n")
   
    # initialize time variable
    code_time = 0.
    # create arrays and do other work needed to setup
    # the main time advance loop -- including normalisation of f by density if requested
    vpa_spectral, z_spectral, r_spectral, moments, fields, vpa_advect, z_advect, r_advect,
        vpa_SL, z_SL, r_SL, scratch, advance = setup_time_advance!(pdf, vpa, z, r, composition,
        drive_input, moments, t_input, collisions, species)
    
    
    #print("got to here 5 \n")
    # setup i/o
    io, cdf = setup_file_io(output_dir, run_name, vpa, z, r, composition, collisions,
                            moments.evolve_ppar)
    # write initial data to ascii files
    write_data_to_ascii(pdf.unnorm, moments, fields, vpa, z, r, code_time, composition.n_species, io)
    # write initial data to binary file (netcdf)
    write_data_to_binary(pdf.unnorm, moments, fields, code_time, composition.n_species, cdf, 1)

    begin_s_r_z_region()


    #print("got to here 6 \n")

    return pdf, scratch, code_time, t_input, vpa, z, r, vpa_spectral, z_spectral, r_spectral, moments,
           fields, vpa_advect, z_advect, r_advect, vpa_SL, z_SL, r_SL, composition, collisions, advance,
           io, cdf
end

# Clean up after a run
function cleanup_moment_kinetics!(io::Union{file_io.ios,Nothing},
                                  cdf::Union{file_io.netcdf_info,Nothing})
    @debug_detect_redundant_block_synchronize begin
        # Disable check for redundant _block_synchronize() during finalization, as this
        # only runs once so any failure is not important.
        debug_detect_redundant_is_active[] = false
    end

    begin_serial_region()

    # finish i/o
    finish_file_io(io, cdf)

    # clean up MPI objects
    finalize_comms!()

    return nothing
end

end
