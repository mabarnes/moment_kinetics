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
                  "timestepping" => Dict{String,Any}("nstep" => 1000,
                                                     "dt" => 1.0e-4,
                                                     "nwrite" => 1000,
                                                     "split_operators" => false),
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
                               "z_nelement" => 16,
                               "vpa_nelement" => 31,
                               "vz_nelement" => 31,
                               "evolve_moments_parallel_pressure" => true,
                               "numerical_dissipation" => Dict{String,Any}("force_minimum_pdf_value" => 0.0, "vpa_dissipation_coefficient" => 1e-2)))
test_input_split3["timestepping"] = merge(test_input_split3["timestepping"],
                                           Dict("dt" => 1.0e-5))

# default inputs for adaptive timestepping tests
test_input_adaptive = merge(test_input,
                            Dict{String,Any}("run_name" => "adaptive full-f",
                                             "z_ngrid" => 5,
                                             "z_nelement" => 16,
                                             "vpa_ngrid" => 6,
                                             "vpa_nelement" => 31,
                                             "vz_ngrid" => 6,
                                             "vz_nelement" => 31))
# Note, use excessively conservative timestepping settings here, because
# we want to avoid any timestep failures in the test. If failures
# occur, the number or when exactly they occur could depend on the
# round-off error, which could make the results less reproducible (even
# though the difference should be negligible compared to the
# discretization error of the simulation).
test_input_adaptive["timestepping"] = merge(test_input_adaptive["timestepping"],
                                            Dict{String,Any}("type" => "Fekete4(3)",
                                                             "nstep" => 5000,
                                                             "dt" => 1.0e-5,
                                                             "minimum_dt" => 1.0e-5,
                                                             "CFL_prefactor" => 1.0,
                                                             "step_update_prefactor" => 0.5,
                                                             "nwrite" => 1000,
                                                             "high_precision_error_sum" => true,
                                                             "split_operators" => false))

test_input_adaptive_split1 = merge(test_input_adaptive,
                                   Dict("run_name" => "adaptive split1",
                                        "evolve_moments_density" => true,
                                        "evolve_moments_conservation" => true))
test_input_adaptive_split2 = merge(test_input_adaptive_split1,
                                   Dict("run_name" => "adaptive split2",
                                        "evolve_moments_parallel_flow" => true))
test_input_adaptive_split2["timestepping"] = merge(test_input_adaptive_split2["timestepping"],
                                                   Dict{String,Any}("step_update_prefactor" => 0.4))
test_input_adaptive_split3 = merge(test_input_adaptive_split2,
                                   Dict("run_name" => "adaptive split3",
                                        "evolve_moments_parallel_pressure" => true,
                                        "numerical_dissipation" => Dict{String,Any}("force_minimum_pdf_value" => 0.0,
                                                                                    "vpa_dissipation_coefficient" => 1e-2)))
# The initial conditions seem to make the split3 case hard to advance without any
# failures. In a real simulation, would just set the minimum_dt higher to try to get
# through this without crashing. For this test, want the timestep to adapt (not just sit
# at minimum_dt), so just set a very small timestep.
test_input_adaptive_split3["timestepping"] = merge(test_input_adaptive_split3["timestepping"],
                                                   Dict{String,Any}("dt" => 1.0e-7,
                                                                    "rtol" => 2.0e-4,
                                                                    "atol" => 2.0e-10,
                                                                    "minimum_dt" => 1.0e-7,
                                                                    "step_update_prefactor" => 0.064))

"""
Run a test for a single set of parameters
"""
function run_test(test_input, expected_phi; rtol=4.e-14, atol=1.e-15, args...)
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
            @test isapprox(actual_phi, expected_phi, rtol=rtol, atol=atol)
        end
    end
end

function runtests()
    # Create a temporary directory for test output
    test_output_directory = get_MPI_tempdir()

    @testset "Recycling fraction" verbose=use_verbose begin
        println("Recycling fraction tests")

        @long @testset "Full-f" begin
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
        @long @testset "Split 1" begin
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
        @long @testset "Split 2" begin
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
        @long @testset "Split 3" begin
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

        @testset "Adaptive timestep - full-f" begin
            test_input_adaptive["base_directory"] = test_output_directory
            run_test(test_input_adaptive,
                     [-0.0440638810983187, -0.021804577592742472, -0.01261897866636822,
                      -0.010636815255843361, -0.006914657999716591, -4.249944880749747e-5,
                      0.006086779933068045, 0.01190043024535959, 0.019617322354987612,
                      0.029847932036078002, 0.03944936313055367, 0.04251143549751657,
                      0.031884499662286334, 0.02118429309190258, 0.015186796843450617,
                      0.009232551395383617, 0.0011522268975610983, -0.005815896057890807,
                      -0.009277831066572732, -0.011461822379560897, -0.02057695651221416,
                      -0.03801829623432516])
        end
        @testset "Adaptive timestep - split 1" begin
            test_input_adaptive_split1["base_directory"] = test_output_directory
            run_test(test_input_adaptive_split1,
                     [-0.044040657899710525, -0.02171631884930434, -0.012547515180557226,
                      -0.010655908101276736, -0.0069170772573878304,
                      -4.355393962794202e-5, 0.006086534259249292, 0.011901727290552297,
                      0.019619284803989313, 0.029848807021314223, 0.03944756553690864,
                      0.042513574213181055, 0.031888958551607754, 0.02118509026798706,
                      0.015188240953123551, 0.00923342235524323, 0.0011506251426590692,
                      -0.005819714223988892, -0.009273870379420018, -0.01155660020408171,
                      -0.02040618061333261, -0.03800833501382964])
        end
        @testset "Adaptive timestep - split 2" begin
            test_input_adaptive_split2["base_directory"] = test_output_directory
            run_test(test_input_adaptive_split2,
                     [-0.04458973581961622, -0.022746972290027208, -0.012948917687917337,
                      -0.010874684145894823, -0.006983552559746659, -6.010006205129479e-5,
                      0.006027491017442355, 0.011975929643130948, 0.019628146001631465,
                      0.02987625619844453, 0.03947012098924943, 0.04249759681422653,
                      0.03185107707012637, 0.021153660966099008, 0.01515958423642362,
                      0.009165359600335187, 0.0011740543476613805, -0.005842751853425141,
                      -0.009109262734493492, -0.01102014206733999, -0.020920404037822794,
                      -0.03755374133774351])
        end
        @testset "Adaptive timestep - split 3" begin
            test_input_adaptive_split3["base_directory"] = test_output_directory
            run_test(test_input_adaptive_split3,
                     [-0.034629893674047986, -0.03201102395873595, -0.027146091249773203,
                      -0.020930812236513833, -0.010156481871961498, 0.0027832606290649417,
                      0.012831753103965483, 0.02209013079063234, 0.033028487927273226,
                      0.04152538920797391, 0.04537564199541445, 0.04623977802174481,
                      0.04254552381541882, 0.03480879451618848, 0.027074396992535412,
                      0.017880292949103555, 0.004778337433968685, -0.007768102112507204,
                      -0.01629918332920463, -0.02414003022582283, -0.03156787896364437,
                      -0.03417555449482148])
        end

        @long @testset "Check other timestep - $type" for
                type ∈ ("RKF5(4)", "Fekete10(4)", "Fekete6(4)", "Fekete4(2)", "SSPRK3",
                        "SSPRK2", "SSPRK1")

            timestep_check_input = deepcopy(test_input_adaptive)
            timestep_check_input["base_directory"] = test_output_directory
            timestep_check_input["run_name"] = type
            timestep_check_input["timestepping"]["type"] = type
            run_test(timestep_check_input,
                     [-0.0440638810983187, -0.021804577592742472, -0.01261897866636822,
                      -0.010636815255843361, -0.006914657999716591, -4.249944880749747e-5,
                      0.006086779933068045, 0.01190043024535959, 0.019617322354987612,
                      0.029847932036078002, 0.03944936313055367, 0.04251143549751657,
                      0.031884499662286334, 0.02118429309190258, 0.015186796843450617,
                      0.009232551395383617, 0.0011522268975610983, -0.005815896057890807,
                      -0.009277831066572732, -0.011461822379560897, -0.02057695651221416,
                      -0.03801829623432516], rtol=8.e-4, atol=1.e-10)
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
