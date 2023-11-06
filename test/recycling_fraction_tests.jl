module RecyclingFraction

# Regression test using wall boundary conditions, with recycling fraction less than 1 and
# a plasma source. Runs for a while and then checks phi profile against saved reference
# output.

include("setup.jl")

using Base.Filesystem: tempname
using MPI
using TimerOutputs

using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.input_structs: grid_input, advection_input
using moment_kinetics.interpolation: interpolate_to_grid_z
using moment_kinetics.load_data: open_readonly_output_file
using moment_kinetics.load_data: load_fields_data,
                                 load_pdf_data, load_time_data,
                                 load_species_data

# Create a temporary directory for test output
test_output_directory = get_MPI_tempdir()

# default inputs for tests
test_input = Dict("n_ion_species" => 1,
                  "n_neutral_species" => 1,
                  "boltzmann_electron_response" => true,
                  "run_name" => "full-f",
                  "base_directory" => test_output_directory,
                  "evolve_moments_density" => false,
                  "evolve_moments_parallel_flow" => false,
                  "evolve_moments_parallel_pressure" => false,
                  "evolve_moments_conservation" => false,
                  "recycling_fraction" => 0.5,
                  "krook_collisions" => true,
                  "T_e" => 0.2,
                  "T_wall" => 0.1,
                  "initial_density1" => 1.0,
                  "initial_temperature1" => 1.0,
                  "z_IC_option1" => "gaussian",
                  "z_IC_density_amplitude1" => 0.001,
                  "z_IC_density_phase1" => 0.0,
                  "z_IC_upar_amplitude1" => 1.0,
                  "z_IC_upar_phase1" => 0.0,
                  "z_IC_temperature_amplitude1" => 0.0,
                  "z_IC_temperature_phase1" => 0.0,
                  "vpa_IC_option1" => "gaussian",
                  "vpa_IC_density_amplitude1" => 1.0,
                  "vpa_IC_density_phase1" => 0.0,
                  "vpa_IC_upar_amplitude1" => 0.0,
                  "vpa_IC_upar_phase1" => 0.0,
                  "vpa_IC_temperature_amplitude1" => 0.0,
                  "vpa_IC_temperature_phase1" => 0.0,
                  "initial_density2" => 1.0,
                  "initial_temperature2" => 1.0,
                  "z_IC_option2" => "gaussian",
                  "z_IC_density_amplitude2" => 0.001,
                  "z_IC_density_phase2" => 0.0,
                  "z_IC_upar_amplitude2" => -1.0,
                  "z_IC_upar_phase2" => 0.0,
                  "z_IC_temperature_amplitude2" => 0.0,
                  "z_IC_temperature_phase2" => 0.0,
                  "vpa_IC_option2" => "gaussian",
                  "vpa_IC_density_amplitude2" => 1.0,
                  "vpa_IC_density_phase2" => 0.0,
                  "vpa_IC_upar_amplitude2" => 0.0,
                  "vpa_IC_upar_phase2" => 0.0,
                  "vpa_IC_temperature_amplitude2" => 0.0,
                  "vpa_IC_temperature_phase2" => 0.0,
                  "charge_exchange_frequency" => 0.75,
                  "ionization_frequency" => 0.5,
                  "constant_ionization_rate" => false,
                  "nstep" => 1000,
                  "dt" => 1.0e-4,
                  "nwrite" => 1000,
                  "n_rk_stages" => 4,
                  "split_operators" => false,
                  "r_ngrid" => 1,
                  "r_nelement" => 1,
                  "z_ngrid" => 9,
                  "z_nelement" => 8,
                  "z_bc" => "wall",
                  "z_discretization" => "chebyshev_pseudospectral",
                  "z_element_spacing_option" => "sqrt",
                  "vpa_ngrid" => 10,
                  "vpa_nelement" => 15,
                  "vpa_L" => 18.0,
                  "vpa_bc" => "zero",
                  "vpa_discretization" => "chebyshev_pseudospectral",
                  "vz_ngrid" => 10,
                  "vz_nelement" => 15,
                  "vz_L" => 18.0,
                  "vz_bc" => "zero",
                  "vz_discretization" => "chebyshev_pseudospectral",
                  "ion_source" => Dict("active" => true,
                                       "z_profile" => "gaussian",
                                       "z_width" => 0.125,
                                       "source_strength" => 2.0,
                                       "source_T" => 2.0))


test_input_split1 = merge(test_input,
                          Dict("run_name" => "split1",
                               "evolve_moments_density" => true,
                               "evolve_moments_conservation" => true))
test_input_split2 = merge(test_input_split1,
                          Dict("run_name" => "split2",
                               "evolve_moments_parallel_flow" => true))
test_input_split3 = merge(test_input_split2,
                          Dict("run_name" => "split3",
                               "evolve_moments_parallel_pressure" => true,
                               "numerical_dissipation" => Dict{String,Any}("force_minimum_pdf_value" => 0.0)))

"""
Run a test for a single set of parameters
"""
function run_test(test_input, expected_phi; args...)
    # by passing keyword arguments to run_test, args becomes a Tuple of Pairs which can be
    # used to update the default inputs

    # Convert keyword arguments to a unique name
    name = test_input["run_name"]
    if length(args) > 0
        name = string(name, "_", (string(k, "-", v, "_") for (k, v) in args)...)

        # Remove trailing "_"
        name = chop(name)
    end

    # Provide some progress info
    println("    - testing ", name)

    # Convert dict from symbol keys to String keys
    modified_inputs = Dict(String(k) => v for (k, v) in args)

    # Update default inputs with values to be changed
    input = merge(test_input, modified_inputs)

    input["run_name"] = name

    # Suppress console output while running
    phi = undef
    quietoutput() do
        # run simulation
        run_moment_kinetics(input)
    end

    if global_rank[] == 0
        quietoutput() do
            # Load and analyse output
            #########################

            path = joinpath(realpath(input["base_directory"]), name, name)

            # open the netcdf file and give it the handle 'fid'
            fid = open_readonly_output_file(path,"moments")

            # load species, time coordinate data
            n_ion_species, n_neutral_species = load_species_data(fid)
            ntime, time = load_time_data(fid)
            n_ion_species, n_neutral_species = load_species_data(fid)
            
            # load fields data
            phi_zrt, Er_zrt, Ez_zrt = load_fields_data(fid)

            close(fid)
            
            phi = phi_zrt[:,1,:]
        end

        # Regression test
        actual_phi = phi[begin:3:end, end]
        if expected_phi == nothing
            # Error: no expected input provided
            println("data tested would be: ", actual_phi)
            @test false
        else
            @test isapprox(actual_phi, expected_phi, rtol=1.e-14, atol=1.e-15)
        end
    end
end

function runtests()

    @testset "Recycling fraction" verbose=use_verbose begin
        println("Recycling fraction tests")

        @testset "Full-f" begin
            run_test(test_input,
                     [-0.05499288668923642, -0.017610447066356092, -0.0014497230450292054,
                      0.0015713106015958053, 0.0021153221201727283, 0.00135154586425295,
                      0.0013626547300678799, 0.003653592144195716, 0.00672151562009703,
                      0.014857207950835708, 0.03452385151240508, 0.03591016289984108,
                      0.02229102871737884, 0.007447997216451657, 0.00505099606227552,
                      0.0016937650707449176, 0.0013469420674100871, 0.0016965410643657965,
                      0.002562353505582182, -6.33366212813045e-5, -0.00969571716777773,
                      -0.048688980279053266])
        end
        @testset "Split 1" begin
            run_test(test_input_split1,
                     [-0.05479385362313513, -0.017535505402861598, -0.001471824975450105,
                      0.0016368028167570122, 0.002097476960724949, 0.0013534479492176561,
                      0.0013561386240175473, 0.003653749578548698, 0.006719973898101642,
                      0.014855703778346607, 0.03452400416817504, 0.03590889139838701,
                      0.022290971814958923, 0.0074469188093168, 0.005048816543623428,
                      0.0016968659971732512, 0.0013266650464258426, 0.0017028444376377347,
                      0.0025344688564808227, -0.00018702558321354208,
                      -0.009661131226016069, -0.048448362188207916])
        end
        @testset "Split 2" begin
            run_test(test_input_split2,
                     [-0.055555691893139914, -0.020145180423245236, 0.001182129287268245,
                      0.002193138222766261, 0.001944121337024971, 0.0011789363569829913,
                      0.0013514285031174852, 0.0037355306549218033, 0.00672369539198614,
                      0.014826903274076, 0.03454936283464004, 0.03587040869161285,
                      0.022277731128342124, 0.007403053124167948, 0.005121534871576419,
                      0.0017463615798537773, 0.001145272466843452, 0.0014049855368865622,
                      0.002275541369035829, 0.0016780270704040375, -0.008381042083345462,
                      -0.05005526304897398])
        end
        @testset "Split 3" begin
            run_test(test_input_split3,
                     [-0.05290583816069098, -0.021497388904574324, -0.002477762532005264,
                      0.001455878854656788, 0.0016388093325101834, 0.0030245421957292365,
                      0.0019329001985405718, 0.0032039812401694263, 0.0068711220110731054,
                      0.014512282612483555, 0.03456423108187791, 0.03564369601953716,
                      0.022146796498517047, 0.007965168282176983, 0.005155553848018062,
                      0.001379839079378397, 0.001669612083430523, 0.00211947053128546,
                      0.002764018750225307, 0.0010824086517704319, -0.014991374022468888,
                      -0.04805193118546115])
        end
    end
end

end # RecyclingFraction


using .RecyclingFraction

RecyclingFraction.runtests()
