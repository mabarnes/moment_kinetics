module RecyclingFraction

# Regression test using wall boundary conditions, with recycling fraction less than 1 and
# a plasma source. Runs for a while and then checks phi profile against saved reference
# output.

include("setup.jl")

using Base.Filesystem: tempname
using MPI

using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.input_structs: grid_input, advection_input, merge_dict_with_kwargs!
using moment_kinetics.interpolation: interpolate_to_grid_z
using moment_kinetics.load_data: get_run_info_no_setup, close_run_info,
                                 postproc_load_variable

# default inputs for tests
test_input = Dict("composition" => Dict{String,Any}("n_ion_species" => 1,
                                                  "n_neutral_species" => 1,
                                                  "electron_physics" => "boltzmann_electron_response",
                                                  "T_e" => 0.2,
                                                  "T_wall" => 0.1,
                                                  "recycling_fraction" => 0.5),
                "ion_species_1" => Dict{String,Any}("initial_density" => 1.0,
                                                    "initial_temperature" => 1.0),
                "z_IC_ion_species_1" => Dict{String,Any}("initialization_option" => "gaussian",
                                                         "density_amplitude" => 0.0,
                                                         "density_phase" => 0.0,
                                                         "upar_amplitude" => 1.0,
                                                         "upar_phase" => 0.0,
                                                         "temperature_amplitude" => 0.0,
                                                         "temperature_phase" => 0.0),
                "vpa_IC_ion_species_1" => Dict{String,Any}("initialization_option" => "gaussian",
                                                         "density_amplitude" => 1.0,
                                                         "density_phase" => 0.0,
                                                         "upar_amplitude" => 0.0,
                                                         "upar_phase" => 0.0,
                                                         "temperature_amplitude" => 0.0,
                                                         "temperature_phase" => 0.0),
                "neutral_species_1" => Dict{String,Any}("initial_density" => 1.0,
                                                        "initial_temperature" => 1.0),
                "z_IC_neutral_species_1" => Dict{String,Any}("initialization_option" => "gaussian",
                                                             "density_amplitude" => 0.001,
                                                             "density_phase" => 0.0,
                                                             "upar_amplitude" => -1.0,
                                                             "upar_phase" => 0.0,
                                                             "temperature_amplitude" => 0.0,
                                                             "temperature_phase" => 0.0),  
                "vpa_IC_neutral_species_1" => Dict{String,Any}("initialization_option" => "gaussian",
                                                             "density_amplitude" => 1.0,
                                                             "density_phase" => 0.0,
                                                             "upar_amplitude" => 0.0,
                                                             "upar_phase" => 0.0,
                                                             "temperature_amplitude" => 0.0,
                                                             "temperature_phase" => 0.0),  
                  "run_name" => "full-f",
                  "evolve_moments_density" => false,
                  "evolve_moments_parallel_flow" => false,
                  "evolve_moments_parallel_pressure" => false,
                  "evolve_moments_conservation" => false,
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

if global_size[] > 2 && global_size[] % 2 == 0
    # Test using distributed-memory
    test_input["z_nelement_local"] = test_input["z_nelement"] ÷ 2
end

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
                               "ion_numerical_dissipation" => Dict{String,Any}("force_minimum_pdf_value" => 0.0, "vpa_dissipation_coefficient" => 1e-2),
                               "neutral_numerical_dissipation" => Dict{String,Any}("force_minimum_pdf_value" => 0.0, "vz_dissipation_coefficient" => 1e-2)))
test_input_split3["timestepping"] = merge(test_input_split3["timestepping"],
                                           Dict("dt" => 1.0e-5,
                                                "write_error_diagnostics" => true,
                                                "write_steady_state_diagnostics" => true))

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

    # Make a copy to make sure nothing modifies the input Dicts defined in this test
    # script.
    input = deepcopy(test_input)

    # Convert keyword arguments to a unique name
    name = input["run_name"]
    if length(args) > 0
        name = string(name, "_", (string(k, "-", v, "_") for (k, v) in args)...)

        # Remove trailing "_"
        name = chop(name)
    end

    # Provide some progress info
    println("    - testing ", name)

    # Update default inputs with values to be changed
    merge_dict_with_kwargs!(input; args...)
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

            path = joinpath(realpath(input["base_directory"]), name)

            # open the output file(s)
            run_info = get_run_info_no_setup(path)

            # load fields data
            phi_zrt = postproc_load_variable(run_info, "phi")

            close_run_info(run_info)
            
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
                     [-0.05519530428547719, -0.017715187591731293, -0.0014094765667960986,
                      0.0017408709303110715, 0.002364329303626605, 0.0015912944705669578,
                      0.0015964146438650865, 0.003860702183595992, 0.0069126570648780075,
                      0.01502802246799623, 0.034672817945651656, 0.03605806530524313,
                      0.022451501088277735, 0.007636465002105951, 0.005249626321396431,
                      0.0019202967839667788, 0.0015870957754252823, 0.0019420461064341924,
                      0.0027769433764546388, 2.482219694607524e-5, -0.009668963817923988,
                      -0.04888254078430223])
        end
        @long @testset "Split 1" begin
            test_input_split1["base_directory"] = test_output_directory
            run_test(test_input_split1,
                     [-0.05499683305021959, -0.017648671323931234, -0.001435044896193346,
                      0.0018073884147499259, 0.0023450881700708397, 0.0015955143089305631,
                      0.001589815317774866, 0.003860937118209949, 0.006911057359417227,
                      0.015026521129765527, 0.03467295602711963, 0.03605680131391841,
                      0.022451426419324128, 0.007635385849475049, 0.005247408512658831,
                      0.0019234631747149433, 0.001566853129890402, 0.001949947092934496,
                      0.0027475042305556622, -0.00010906536252042205,
                      -0.00962961346763738, -0.04864884428378774])
        end
        @long @testset "Split 2" begin
            test_input_split2["base_directory"] = test_output_directory
            run_test(test_input_split2,
                     [-0.05584608895693617, -0.020285311011810747, 0.0013246162852513857,
                      0.002408198079080654, 0.002193404660625343, 0.0014165984310586497,
                      0.0015838582299817442, 0.003942456292519074, 0.006915806487555444,
                      0.014996822639406769, 0.034698460163972725, 0.03601812331030096,
                      0.022438422037486003, 0.007592137358805067, 0.00532114704663586,
                      0.001973382574270663, 0.0013804707387267182, 0.0016443777257862315,
                      0.0025134094388913966, 0.0018832456170377893, -0.008404304571565805,
                      -0.05034925353831177])
        end
        @long @testset "Split 3" begin
            test_input_split3["base_directory"] = test_output_directory
            run_test(test_input_split3,
                     [-0.03620705983288495, -0.030483526856225397, -0.028960441350906176,
                      -0.028549218778503995, -0.025515599608030678, -0.021975115062752498,
                      -0.019818943505064867, -0.013314608790987136, -0.009839994543852062,
                      -0.005265524793578627, 0.002150328580191541, 0.004621192779726743,
                      0.011510249894025814, 0.017574569324021832, 0.020149366401796907,
                      0.027122843852491103, 0.03050649889203747, 0.03404441833358536,
                      0.039323018405068834, 0.04089864462026069, 0.04436314083820065,
                      0.04582518382395237, 0.045703097564838854, 0.04312195015009901,
                      0.04054713854267327, 0.0378719503058148, 0.03223787080558438,
                      0.030105904373564214, 0.024828730387096765, 0.01859677066598083,
                      0.01621033424329937, 0.008755805694319756, 0.004008885194932725,
                      3.98551863685712e-5, -0.007641449438487893, -0.010257112218851392,
                      -0.01554078023638837, -0.021190999288843403, -0.022783488732253034,
                      -0.026872207781674047, -0.028748355604856834, -0.02921966520151288,
                      -0.03230397503173283])
        end

        fullf_expected_output = [-0.04413552960518834, -0.021828861958540703,
                                 -0.012581752434237016, -0.010579192578141765,
                                 -0.0068512759586887885, 2.5839426347419376e-5,
                                 0.006155432086970659, 0.011968120005188723,
                                 0.019681941313251225, 0.029904657995405693,
                                 0.03949771002617614, 0.04255717403165738,
                                 0.031939444000420925, 0.02124792913154375,
                                 0.015253505295222548, 0.00930078535463162,
                                 0.0012208189549702839, -0.005751247690841759,
                                 -0.009217769053947429, -0.011407780388153488,
                                 -0.020596064125157174, -0.03809484046514718]
        @testset "Adaptive timestep - full-f" begin
            test_input_adaptive["base_directory"] = test_output_directory
            run_test(test_input_adaptive,
                     fullf_expected_output, rtol=6.0e-4, atol=2.0e-12)
        end
        @testset "Adaptive timestep - split 1" begin
            test_input_adaptive_split1["base_directory"] = test_output_directory
            run_test(test_input_adaptive_split1,
                     [-0.04411241514173686, -0.02173832960900953, -0.012507206800103619,
                      -0.010594605015685111, -0.006851937496868262, 2.4307749760781248e-5,
                      0.006153573001701797, 0.011966908336801292, 0.019680600194907028,
                      0.02990265026906549, 0.039494202889752195, 0.04255801055036077,
                      0.031941240640856954, 0.02124538117173497, 0.0152520119972139,
                      0.009299543598122585, 0.001218486350949803, -0.005753814631808573,
                      -0.009211138614098327, -0.011500666056267622, -0.020424831739003606,
                      -0.03808534246490289], rtol=6.0e-4, atol=2.0e-12)
        end
        @testset "Adaptive timestep - split 2" begin
            test_input_adaptive_split2["base_directory"] = test_output_directory
            run_test(test_input_adaptive_split2,
                     [-0.044658773996679106, -0.022770640350132876, -0.01291279676887995,
                      -0.010818472056813256, -0.00692137979236985, 8.260915374129437e-6,
                      0.006095505380954945, 0.012043966021961394, 0.01969312249006842,
                      0.02993329149668162, 0.03951863202813308, 0.04254329784045647,
                      0.031905905757383245, 0.0212173464030042, 0.015225863469416798,
                      0.00923409202948059, 0.0012431576067072942, -0.005777971895837924,
                      -0.009047333684784373, -0.010964221005155143, -0.020937032832434074,
                      -0.03762542957657465], rtol=6.0e-4, atol=2.0e-12)
        end
        @testset "Adaptive timestep - split 3" begin
            test_input_adaptive_split3["base_directory"] = test_output_directory
            run_test(test_input_adaptive_split3,
                     [-0.03462849854420647, -0.03201042148267427, -0.027145441768761876,
                      -0.020930163084065857, -0.010155888649583823, 0.0027838306741690046,
                      0.012832285048924648, 0.022090624745954916, 0.033028935673319444,
                      0.041525790733438206, 0.04537601028144605, 0.046240136469284064,
                      0.042545918106145095, 0.03480923464250141, 0.02707487186844719,
                      0.017880803893474347, 0.004778898480075153, -0.007767501511791003,
                      -0.016298557311046468, -0.0241393991660673, -0.03156718517969005,
                      -0.03417441352897386], rtol=6.0e-4, atol=2.0e-12)
        end

        @long @testset "Check other timestep - $type" for
                type ∈ ("RKF5(4)", "Fekete10(4)", "Fekete6(4)", "Fekete4(2)", "SSPRK3",
                        "SSPRK2", "SSPRK1")

            timestep_check_input = deepcopy(test_input_adaptive)
            timestep_check_input["base_directory"] = test_output_directory
            timestep_check_input["run_name"] = type
            timestep_check_input["timestepping"]["type"] = type
            run_test(timestep_check_input,
                     fullf_expected_output, rtol=8.e-4, atol=1.e-10)
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
