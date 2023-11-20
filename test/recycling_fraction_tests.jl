module RecyclingFraction

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
                  "boltzmann_electron_response" => true,
                  "run_name" => "full-f",
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
                               "dt" => 1.0e-5,
                               "z_nelement" => 16,
                               "vpa_nelement" => 31,
                               "vz_nelement" => 31,
                               "evolve_moments_parallel_pressure" => true,
                               "numerical_dissipation" => Dict{String,Any}("force_minimum_pdf_value" => 0.0, "vpa_dissipation_coefficient" => 1e-2)))

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
            @test isapprox(actual_phi, expected_phi, rtol=4.e-14, atol=1.e-15)
        end
    end
end

function runtests()
    # Create a temporary directory for test output
    test_output_directory = get_MPI_tempdir()

    @testset "Recycling fraction" verbose=use_verbose begin
        println("Recycling fraction tests")

        @testset "Full-f" begin
            test_input["base_directory"] = test_output_directory
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
            test_input_split1["base_directory"] = test_output_directory
            run_test(test_input_split1,
                     [-0.054793853738618496, -0.017535475032013862,
                      -0.0014718402826481662, 0.0016368065803215382, 0.002097475822421603,
                      0.001353447830403315, 0.001356138437924921, 0.0036537497347573,
                      0.006719973928112565, 0.014855703760316889, 0.03452400419220982,
                      0.03590889137214591, 0.022290971843531463, 0.007446918804323913,
                      0.005048816472156039, 0.0016968661957691385, 0.0013266658105610114,
                      0.0017028442360018413, 0.002534466861251151,
                      -0.00018703865529355897, -0.009661145065079906,
                      -0.0484483682752969])
        end
        @testset "Split 2" begin
            test_input_split2["base_directory"] = test_output_directory
            run_test(test_input_split2,
                     [-0.05555568198447252, -0.020145183717956348, 0.001182118478411508,
                      0.002193148323751635, 0.0019441188563940751, 0.0011789368818662881,
                      0.0013514249605048384, 0.003735531583031493, 0.006723696092974834,
                      0.014826903180374499, 0.03454936277756109, 0.03587040875737859,
                      0.022277731154827392, 0.007403052912240603, 0.00512153431160143,
                      0.0017463637584066217, 0.0011452779397062784, 0.0014049872146431029,
                      0.0022755389057580316, 0.0016780234234311344, -0.008381041468024259,
                      -0.05005526194222513])
        end
        @testset "Split 3" begin
            test_input_split3["base_directory"] = test_output_directory
            run_test(test_input_split3,
                     [-0.036205375991650725, -0.030483334021285433, -0.028961568619094404,
                      -0.028550383934166465, -0.02551672335720456, -0.021976119708577647,
                      -0.01982001937014411, -0.01331564927923702, -0.00984100255121529,
                      -0.005266490060020825, 0.0021494114844098316, 0.004620296275317165,
                      0.011509404776589328, 0.01757377252325957, 0.02014859036576961,
                      0.027122126647315926, 0.030505809525427197, 0.034043759795000156,
                      0.03932240322253646, 0.04089804092628224, 0.04436256082283185,
                      0.04582461258085377, 0.0457025256980273, 0.04312136181903663,
                      0.04054653135540802, 0.03787132328029428, 0.03223719811392133,
                      0.030105212408583878, 0.024827994199332723, 0.018595982530248478,
                      0.016209527148134187, 0.008754940562653064, 0.0040079860524162405,
                      3.89264740137833e-5, -0.007642430261913982, -0.010258137085572222,
                      -0.015541799469166076, -0.021192018291797773, -0.022784703489569562,
                      -0.026873219344096318, -0.028749404798656616, -0.029220744790456707,
                      -0.032303083015072])
        end
    end

    if global_rank[] == 0
        # Delete output directory to avoid using too much disk space
        rm(realpath(test_output_directory); recursive=true)
    end
end

end # RecyclingFraction


using .RecyclingFraction

RecyclingFraction.runtests()
