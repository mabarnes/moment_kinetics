# add the current directory to the path where the code looks for external modules
push!(LOAD_PATH, ".")

module moment_kinetics

export run_moment_kinetics

using TimerOutputs

using file_io: setup_file_io, finish_file_io
using file_io: write_data_to_ascii, write_data_to_binary
using coordinates: define_coordinate
using initial_conditions: init_f
using moment_kinetics_input: run_type
using moment_kinetics_input: performance_test
using time_advance: setup_time_advance!, time_advance!

# main function that contains all of the content of the program
function run_moment_kinetics(to, input)
    # obtain input options from moment_kinetics_input.jl
    # and check input to catch errors
    run_name, output_dir, evolve_moments, t_input, z_input, vpa_input,
        composition, species, charge_exchange_frequency, drive_input = input
    # initialize z grid and write grid point locations to file
    z = define_coordinate(z_input)
    # initialize vpa grid and write grid point locations to file
    vpa = define_coordinate(vpa_input)
    # initialize f(z).  note that ff initialised here satisfies ∫dvpa f(z,vpa) = n(z)
    # if evolve_moments.density = true, it will be normalsied by n(z) later
    ff = init_f(z, vpa, composition, species, t_input.n_rk_stages)
    # initialize time variable
    code_time = 0.
    # create arrays and do other work needed to setup
    # the main time advance loop -- including normalisation of f by density if requested
    z_spectral, vpa_spectral, moments, fields, z_advect, vpa_advect,
        z_SL, vpa_SL, scratch, advance = setup_time_advance!(ff, z, vpa, composition,
        drive_input, evolve_moments, t_input, charge_exchange_frequency)
    # setup i/o
    io, cdf = setup_file_io(output_dir, run_name, z, vpa, composition, charge_exchange_frequency)
    # write initial data to ascii files
    write_data_to_ascii(ff, moments, fields, z, vpa, code_time, composition.n_species, io)
    # write initial data to binary file (netcdf)
    write_data_to_binary(ff, moments, fields, code_time, composition.n_species, cdf, 1)
    # solve the 1+1D kinetic equation to advance f in time by nstep time steps
    if run_type == performance_test
        @timeit to "time_advance" time_advance!(ff, scratch, code_time, t_input,
            z, vpa, z_spectral, vpa_spectral, moments, fields,
            z_advect, vpa_advect, z_SL, vpa_SL, composition, charge_exchange_frequency,
            advance, io, cdf)
    else
        time_advance!(ff, scratch, code_time, t_input, z, vpa,
            z_spectral, vpa_spectral, moments, fields,
            z_advect, vpa_advect, z_SL, vpa_SL, composition, charge_exchange_frequency,
            advance, io, cdf)
    end
    # finish i/o
    finish_file_io(io, cdf)
    return nothing
end

end

# provide option of running from command line via 'julia moment_kinetics.jl'
if abspath(PROGRAM_FILE) == @__FILE__
    using TimerOutputs
    using moment_kinetics_input: mk_input

    to = TimerOutput
    input = mk_input()
    moment_kinetics.run_moment_kinetics(to, input)
end
