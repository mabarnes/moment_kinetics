"""
Module for testing the 1D ITG helical field case.

After running the simulation, the growth rate is calculated for the 
single initialised mode using a DFFT or something.

Growth rate in code units (check page 26 of FP4):

γ = k_z * ρ_i * v_ti/(2*sqrt(L_B * L_T)) - D * k_z^2

in units in the code, this is 

γ = k_z/(sqrt(L_B * L_T)) - D * k_z^2

"""

module ITG_1D

# Test generated from TOML input files

include("setup.jl")

using Base.Filesystem: tempname
using MPI

using moment_kinetics.interpolation: interpolate_to_grid_z
using moment_kinetics.load_data: get_run_info_no_setup, close_run_info,
                                 postproc_load_variable, get_variable
using moment_kinetics.utils: merge_dict_with_kwargs!
using moment_kinetics.analysis: get_growth_rate_of_box_mode_1D

# default inputs for tests
test_input = OptionsDict(
 "output" => OptionsDict(
     "display_timing_info" => false
    ),
 "r" => OptionsDict(
     "ngrid" => 1,
     "nelement" => 1
    ),
 "vperp" => OptionsDict(
     "ngrid" => 5,
     "discretization" => "chebyshev_pseudospectral",
     "nelement" => 5,
     "L" => 6.0,
     "bc" => "zero"
    ),
 "evolve_moments" => OptionsDict(
     "pressure" => false,
     "density" => false,
     "moments_conservation" => false,
     "parallel_flow" => false
    ),
 "ion_species_1" => OptionsDict(
     "initial_temperature" => 1.0,
     "initial_density" => 1.0,
     "L_T" => -0.005
    ),
 "krook_collisions" => OptionsDict(
     "use_krook" => true,
     "frequency_option" => "reference_parameters"
    ),
 "vpa" => OptionsDict(
     "ngrid" => 6,
     "discretization" => "chebyshev_pseudospectral",
     "nelement" => 10,
     "L" => 9.0,
     "bc" => "zero"
    ),
 "geometry" => OptionsDict(
     "pitch" => 0.0,
     "option" => "1D-Helical-ITG",
     "rhostar" => 1.0,
     "dBdr_constant" => 0.05
    ),
 "z" => OptionsDict(
     "ngrid" => 5,
     "discretization" => "chebyshev_pseudospectral",
     "nelement" => 16,
     "L" => 1.0,
     "bc" => "periodic"
    ),
 "vpa_IC_ion_species_1" => OptionsDict(
     "initialization_option" => "gaussian",
     "density_amplitude" => 1.0,
     "temperature_amplitude" => 0.0,
     "density_phase" => 0.0,
     "upar_amplitude" => 0.0,
     "temperature_phase" => 0.0,
     "upar_phase" => 0.0
    ),
 "composition" => OptionsDict(
     "T_e" => 1.0,
     "electron_physics" => "boltzmann_electron_response",
     "n_ion_species" => 1,
     "n_neutral_species" => 0
    ),
 "ion_numerical_dissipation" => OptionsDict(
     "z_dissipation_coefficient" => 0.1,
     "z_dissipation_degree" => 2
    ),
 "z_IC_ion_species_1" => OptionsDict(
     "initialization_option" => "sinusoid",
     "wavenumber" => 1.0,
     "density_amplitude" => 1.0e-6,
     "temperature_amplitude" => 0.0,
     "density_phase" => 0.0,
     "upar_amplitude" => 0.0,
     "temperature_phase" => 0.0,
     "upar_phase" => 0.0
    ),
 "timestepping" => OptionsDict(
     "nstep" => 800,
     "steady_state_residual" => true,
     "dt" => 0.0006,
     "nwrite" => 20,
     "type" => "SSPRK4",
     "nwrite_dfns" => 20,
     "print_nT_live" => true
    )
)

test_input_2 = recursive_merge(test_input,
                               OptionsDict(
 "ion_species_1" => OptionsDict(
     "L_T" => -0.01
    )
))
test_input_3 = recursive_merge(test_input,
                               OptionsDict(
 "timestepping" => OptionsDict(
     "dt" => 0.0007
    ),
 "ion_species_1" => OptionsDict(
     "L_T" => -0.015
    )
))
# Here choose the names for each test
test_input = recursive_merge(test_input,
                               OptionsDict("output" => OptionsDict("run_name" => "L_B 20, L_T 0.005")))
test_input_2 = recursive_merge(test_input_2,
                               OptionsDict("output" => OptionsDict("run_name" => "L_B 20, L_T 0.01")))
test_input_3 = recursive_merge(test_input_3,
                               OptionsDict("output" => OptionsDict("run_name" => "L_B 20, L_T 0.015")))

"""
Run a test for a single set of parameters
"""
function run_test(test_input, expected_phi, growth_rate; rtol=5.e-6, atol=1.e-15, args...)
    # by passing keyword arguments to run_test, args becomes a Tuple of Pairs which can be
    # used to update the default inputs

    # Make a copy to make sure nothing modifies the input Dicts defined in this test
    # script.
    input = deepcopy(test_input)

    # Convert keyword arguments to a unique name
    name = input["output"]["run_name"]
    if length(args) > 0
        name = string(name, "_", (string(k, "-", v, "_") for (k, v) in args)...)

        # Remove trailing "_"
        name = chop(name)
    end

    # Provide some progress info
    println("    - testing ", name)

    # Update default inputs with values to be changed
    merge_dict_with_kwargs!(input; args...)
    input["output"]["run_name"] = name

    # Suppress console output while running
    phi = nothing
    γ = nothing
    quietoutput() do
        # run simulation
        run_moment_kinetics(input)
    end

    if global_rank[] == 0
        quietoutput() do
            # Load and analyse output
            #########################

            path = joinpath(realpath(input["output"]["base_directory"]), name)

            # open the output file(s)
            run_info = get_run_info_no_setup(path)

            # load fields data
            phi_zrt = postproc_load_variable(run_info, "phi")
            phi = phi_zrt[:,1,:]

            γ = get_growth_rate_of_box_mode_1D(run_info)

            close_run_info(run_info)
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

        # test that growth rate of box scale mode was within tolerance of original
        @test isapprox(γ, growth_rate, rtol=rtol, atol=atol)

    end
end

function runtests()
    # Create a temporary directory for test output
    test_output_directory = get_MPI_tempdir()

    @testset "ITG_1D tests" verbose=use_verbose begin
        println("ITG_1D tests")
        @testset "try_large_LB_1" begin
            test_input["output"]["base_directory"] = test_output_directory
            run_test(test_input,
                     [-0.00092136560596960582, -0.0010873935972280562, -0.00096911839965229112, 
                     -0.00094855521648896869, -0.00066429192750337446, -0.000364794120938439, 
                     -7.8941331344729244e-05, 0.00020108376961124316, 0.00048279003480120991, 
                     0.00082833025735707993, 0.00088130271493059578, 0.0010325850954454058, 
                     0.0009389583236784877, 0.00091385323512876428, 0.00070541091448252341, 
                     0.00058100662074831778, 0.00028053083952395478, -5.0845478544697865e-05, 
                     -0.00026420773472161935, -0.00052938018405801414, -0.00071385591685403649, 
                     -0.00099852553565971613], [15.899808589067652])
        end
        @testset "try_large_LB_2" begin
            test_input_2["output"]["base_directory"] = test_output_directory
            run_test(test_input_2,
                     [-7.4855860391547925e-05, -7.7189112103790371e-05, -7.4252375253183473e-05, 
                     -6.7441393010133044e-05, -5.3276475488905454e-05, -3.4711970060149843e-05, 
                     -1.8980385105405269e-05, -2.9245223328020922e-06, 1.7022131525353316e-05, 
                     3.3504134937201261e-05, 4.2536956975883595e-05, 4.7941182742410966e-05, 
                     4.8972889967607691e-05, 4.3010983225942717e-05, 3.4226276208098745e-05, 
                     2.2368276847320008e-05, 3.3089653558193284e-06, -1.7658987956091874e-05, 
                     -3.3391821339161166e-05, -4.7885565007146585e-05, -6.3572933052972796e-05, 
                     -7.3755996241030075e-05], [10.079216090727092])
        end
        @testset "try_large_LB_3" begin
            test_input_3["output"]["base_directory"] = test_output_directory
            run_test(test_input_3,
                     [-4.8013926991666265e-05, -4.9715053149726605e-05, -4.8520164714074109e-05, 
                     -4.525685525189694e-05, -3.8148306012431515e-05, -2.8600287378195704e-05, 
                     -2.0388907394155408e-05, -1.191271062243366e-05, -1.2361240704927428e-06, 
                     7.7647038399472831e-06, 1.2842734767631775e-05, 1.6053305234619468e-05, 
                     1.710150985915883e-05, 1.4436958686440702e-05, 1.0115812188642918e-05, 
                     4.1064147810250505e-06, -5.7505531071005723e-06, -1.6774972395660353e-05, 
                     -2.515366948502484e-05, -3.2963600121591584e-05, -4.1565798542815808e-05, 
                     -4.7355801258764636e-05], [7.496243247066792])
        end
    end
    if global_rank[] == 0
        # Delete output directory to avoid using too much disk space
        rm(realpath(test_output_directory); recursive=true)
    end
end

end

using .ITG_1D

ITG_1D.runtests()
