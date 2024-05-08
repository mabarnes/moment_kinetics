module BraginskiiElectronsIMEX

# Regression test using wall boundary conditions, with recycling fraction less than 1 and
# a plasma source. Runs for a while and then checks phi profile against saved reference
# output.

include("setup.jl")

using Base.Filesystem: tempname
using MPI

using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.input_structs: grid_input, advection_input
using moment_kinetics.interpolation: interpolate_to_grid_z
using moment_kinetics.load_data: open_readonly_output_file
using moment_kinetics.load_data: load_fields_data,
                                 load_pdf_data, load_time_data,
                                 load_species_data

# default inputs for tests
test_input = Dict("n_ion_species" => 1,
                  "n_neutral_species" => 1,
                  "electron_physics" => "braginskii_fluid",
                  "run_name" => "braginskii-electrons-imex",
                  "evolve_moments_density" => true,
                  "evolve_moments_parallel_flow" => true,
                  "evolve_moments_parallel_pressure" => true,
                  "evolve_moments_conservation" => true,
                  "recycling_fraction" => 0.5,
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
                  "timestepping" => Dict{String,Any}("type" => "KennedyCarpenterARK324",
                                                     "nstep" => 100,
                                                     "dt" => 1.0e-5,
                                                     "minimum_dt" => 5.e-6,
                                                     "nwrite" => 100),
                  "r_ngrid" => 1,
                  "r_nelement" => 1,
                  "z_ngrid" => 5,
                  "z_nelement" => 16,
                  "z_bc" => "wall",
                  "z_discretization" => "chebyshev_pseudospectral",
                  "z_element_spacing_option" => "sqrt",
                  "vpa_ngrid" => 6,
                  "vpa_nelement" => 31,
                  "vpa_L" => 18.0,
                  "vpa_bc" => "zero",
                  "vpa_discretization" => "chebyshev_pseudospectral",
                  "vz_ngrid" => 6,
                  "vz_nelement" => 31,
                  "vz_L" => 18.0,
                  "vz_bc" => "zero",
                  "vz_discretization" => "chebyshev_pseudospectral",
                  "ion_source" => Dict("active" => true,
                                       "z_profile" => "gaussian",
                                       "z_width" => 0.125,
                                       "source_strength" => 2.0,
                                       "source_T" => 2.0),
                  "ion_numerical_dissipation" => Dict{String,Any}("force_minimum_pdf_value" => 0.0,
                                                                  "vpa_dissipation_coefficient" => 1e-2),
                  "neutral_numerical_dissipation" => Dict{String,Any}("force_minimum_pdf_value" => 0.0,
                                                                      "vz_dissipation_coefficient" => 1e-2))

if global_size[] > 2 && global_size[] % 2 == 0
    # Test using distributed-memory
    test_input["z_nelement_local"] = test_input["z_nelement"] ÷ 2
end

"""
Run a test for a single set of parameters
"""
function run_test(test_input, expected_phi; rtol=4.e-14, atol=1.e-15, args...)
    # by passing keyword arguments to run_test, args becomes a Tuple of Pairs which can be
    # used to update the default inputs

    # Make a copy to make sure nothing modifies the input Dicts defined in this test
    # script.
    test_input = deepcopy(test_input)

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
            @test isapprox(actual_phi, expected_phi, rtol=rtol, atol=atol)
        end
    end
end

function runtests()
    # Create a temporary directory for test output
    test_output_directory = get_MPI_tempdir()

    expected_results = [3.0, 3.4312405783969817, 3.6672032157110688, 3.8986711186885126,
                        4.300726176582548, 4.751419246046722, 5.174798022240745,
                        5.673403293519727, 6.2113762678636055, 6.585223545483835,
                        6.787338093050003, 6.800854619194591, 6.502479916930342,
                        6.070728566432454, 5.625825781093318, 5.050215164940833,
                        4.417039506095468, 3.931944879496679, 3.5660918681240577,
                        3.192708043666926, 2.6965813995980894, 1.8135891820742627]

    @testset "Braginskii electron IMEX timestepping" verbose=use_verbose begin
        println("Braginskii electron IMEX timestepping tests")

        @testset "Split 3" begin
            test_input["base_directory"] = test_output_directory
            run_test(test_input, expected_results)
        end
        @long @testset "Check other timestep - $type" for
                type ∈ ("KennedyCarpenterARK437",)

            timestep_check_input = deepcopy(test_input)
            timestep_check_input["base_directory"] = test_output_directory
            timestep_check_input["run_name"] = type
            timestep_check_input["timestepping"]["type"] = type
            run_test(timestep_check_input, expected_results, rtol=2.e-4, atol=1.e-10)
        end
    end

    if global_rank[] == 0
        # Delete output directory to avoid using too much disk space
        rm(realpath(test_output_directory); recursive=true)
    end
end

end # BraginskiiElectronsIMEX


using .BraginskiiElectronsIMEX

BraginskiiElectronsIMEX.runtests()
