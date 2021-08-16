module moment_kinetics

export run_moment_kinetics

# Include submodules from other source files
# Note that order of includes matters - things used in one module must already
# be defined
include("type_definitions.jl")
include("optimization.jl")
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
include("charge_exchange.jl")
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
using .coordinates: define_coordinate
using .initial_conditions: init_pdf_and_moments
using .moment_kinetics_input: mk_input, run_type, performance_test
using .time_advance: setup_time_advance!, time_advance!

# main function that contains all of the content of the program
function run_moment_kinetics(to, input_dict=Dict())
    input = mk_input(input_dict)
    # obtain input options from moment_kinetics_input.jl
    # and check input to catch errors
    run_name, output_dir, evolve_moments, t_input, z_input, vpa_input,
        composition, species, charge_exchange_frequency, drive_input = input
    # initialize z grid and write grid point locations to file
    z = define_coordinate(z_input)
    # initialize vpa grid and write grid point locations to file
    vpa = define_coordinate(vpa_input)
    # initialize f(z,vpa) and the lowest three v-space moments (density(z), upar(z) and ppar(z)),
    # each of which may be evolved separately depending on input choices.
    pdf, moments = init_pdf_and_moments(vpa, z, composition, species, t_input.n_rk_stages, evolve_moments)
    # initialize time variable
    code_time = 0.
    # create arrays and do other work needed to setup
    # the main time advance loop -- including normalisation of f by density if requested
    vpa_spectral, z_spectral, moments, fields, vpa_advect, z_advect,
        vpa_SL, z_SL, scratch, advance = setup_time_advance!(pdf, vpa, z, composition,
        drive_input, moments, t_input, charge_exchange_frequency, species)
    # setup i/o
    io, cdf = setup_file_io(output_dir, run_name, vpa[1], z[1], composition, charge_exchange_frequency,
                            moments.evolve_ppar)
    # write initial data to ascii files
    write_data_to_ascii(pdf.unnorm, moments, fields, vpa[1], z[1], code_time, composition.n_species, io)
    # write initial data to binary file (netcdf)
    write_data_to_binary(pdf.unnorm, moments, fields, code_time, composition.n_species, cdf, 1)
    # solve the 1+1D kinetic equation to advance f in time by nstep time steps
    if run_type == performance_test
        @timeit to "time_advance" time_advance!(pdf, scratch, code_time, t_input,
            vpa, z, vpa_spectral, z_spectral, moments, fields,
            vpa_advect, z_advect, vpa_SL, z_SL, composition, charge_exchange_frequency,
            advance, io, cdf)
    else
        time_advance!(pdf, scratch, code_time, t_input, vpa, z,
            vpa_spectral, z_spectral, moments, fields,
            vpa_advect, z_advect, vpa_SL, z_SL, composition, charge_exchange_frequency,
            advance, io, cdf)
    end
    # finish i/o
    finish_file_io(io, cdf)
    return nothing
end

# overload which takes a filename and loads input
function run_moment_kinetics(to, input_filename::String)
    return run_moment_kinetics(to, TOML.parsefile(input_filename))
end

end
