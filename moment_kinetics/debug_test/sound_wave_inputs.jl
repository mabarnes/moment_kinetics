test_type = "sound_wave"

# default inputs for tests
test_input_finite_difference_1D1V = Dict(
     "run_name" => "finite_difference_1D1V",
     "n_ion_species" => 2,
     "n_neutral_species" => 2,
     "boltzmann_electron_response" => true,
     "base_directory" => test_output_directory,
     "evolve_moments_density" => false,
     "evolve_moments_parallel_flow" => false,
     "evolve_moments_parallel_pressure" => false,
     "evolve_moments_conservation" => true,
     "T_e" => 1.0,
     "initial_density1" => 0.5,
     "initial_temperature1" => 1.0,
     "initial_density2" => 0.5,
     "initial_temperature2" => 1.0,
     "z_IC_option1" => "sinusoid",
     "z_IC_density_amplitude1" => 0.001,
     "z_IC_density_phase1" => 0.0,
     "z_IC_upar_amplitude1" => 0.0,
     "z_IC_upar_phase1" => 0.0,
     "z_IC_temperature_amplitude1" => 0.0,
     "z_IC_temperature_phase1" => 0.0,
     "z_IC_option2" => "sinusoid",
     "z_IC_density_amplitude2" => 0.001,
     "z_IC_density_phase2" => 0.0,
     "z_IC_upar_amplitude2" => 0.0,
     "z_IC_upar_phase2" => 0.0,
     "z_IC_temperature_amplitude2" => 0.0,
     "z_IC_temperature_phase2" => 0.0,
     "charge_exchange_frequency" => 2*π*0.1,
     "ionization_frequency" => 0.0,
     "timestepping" => Dict{String,Any}("nstep" => 3,
                                        "dt" => 1.e-8,
                                        "nwrite" => 2,
                                        "n_rk_stages" => 2,
                                        "split_operators" => false),
     "z_ngrid" => 4,
     "z_nelement" => 1,
     "z_bc" => "periodic",
     "z_discretization" => "finite_difference",
     "r_ngrid" => 1,
     "r_nelement" => 1,
     "vpa_ngrid" => 4,
     "vpa_nelement" => 1,
     "vpa_L" => 8.0,
     "vpa_bc" => "periodic",
     "vpa_discretization" => "finite_difference",
     "vperp_ngrid" => 1,
     "vperp_nelement" => 1,
     "vz_ngrid" => 4,
     "vz_nelement" => 1,
     "vz_L" => 8.0,
     "vz_bc" => "periodic",
     "vz_discretization" => "finite_difference",
     "vzeta_ngrid" => 1,
     "vzeta_nelement" => 1,
     "vr_ngrid" => 1,
     "vr_nelement" => 1)

test_input_finite_difference_1D1V_split_1_moment =
    merge(test_input_finite_difference_1D1V,
          Dict("run_name" => "finite_difference_1D1V_split_1_moment",
               "evolve_moments_density" => true))

test_input_finite_difference_1D1V_split_2_moments =
    merge(test_input_finite_difference_1D1V_split_1_moment,
          Dict("run_name" => "finite_difference_1D1V_split_2_moments",
               "evolve_moments_parallel_flow" => true))

test_input_finite_difference_1D1V_split_3_moments =
    merge(test_input_finite_difference_1D1V_split_2_moments,
          Dict("run_name" => "finite_difference_1D1V_split_3_moments",
               "evolve_moments_parallel_pressure" => true))

test_input_finite_difference_cx0_1D1V =
    merge(test_input_finite_difference_1D1V,
          Dict("run_name" => "finite_difference_cx0_1D1V",
               "charge_exchange_frequency" => 0.0))

test_input_finite_difference_cx0_1D1V_split_1_moment =
    merge(test_input_finite_difference_cx0_1D1V,
          Dict("run_name" => "finite_difference_cx0_1D1V_split_1_moment",
               "evolve_moments_density" => true))

test_input_finite_difference_cx0_1D1V_split_2_moments =
    merge(test_input_finite_difference_cx0_1D1V_split_1_moment,
          Dict("run_name" => "finite_difference_cx0_1D1V_split_2_moments",
               "evolve_moments_parallel_flow" => true))

test_input_finite_difference_cx0_1D1V_split_3_moments =
    merge(test_input_finite_difference_cx0_1D1V_split_2_moments,
          Dict("run_name" => "finite_difference_cx0_1D1V_split_3_moments",
               "evolve_moments_parallel_pressure" => true))

test_input_finite_difference =
    merge(test_input_finite_difference_1D1V,
          Dict("run_name" => "finite_difference",
               "r_ngrid" => 4,
               "r_nelement" => 1,
               "r_discretization" => "finite_difference",
               "vperp_ngrid" => 4,
               "vperp_nelement" => 1,
               "vperp_discretization" => "finite_difference",
               "vz_ngrid" => 4,
               "vz_nelement" => 1,
               "vz_discretization" => "finite_difference",
               "vr_ngrid" => 4,
               "vr_nelement" => 1,
               "vr_discretization" => "finite_difference",
               "vzeta_ngrid" => 4,
               "vzeta_nelement" => 1,
               "vzeta_discretization" => "finite_difference"))

test_input_finite_difference_split_1_moment =
    merge(test_input_finite_difference,
          Dict("run_name" => "finite_difference_split_1_moment",
               "evolve_moments_density" => true))

test_input_finite_difference_split_2_moments =
    merge(test_input_finite_difference_split_1_moment,
          Dict("run_name" => "finite_difference_split_2_moments",
               "evolve_moments_parallel_flow" => true))

test_input_finite_difference_split_3_moments =
    merge(test_input_finite_difference_split_2_moments,
          Dict("run_name" => "finite_difference_split_3_moments",
               "evolve_moments_parallel_pressure" => true))

test_input_finite_difference_cx0 =
    merge(test_input_finite_difference,
          Dict("run_name" => "finite_difference_cx0",
               "charge_exchange_frequency" => 0.0))

test_input_finite_difference_cx0_split_1_moment =
    merge(test_input_finite_difference_cx0,
          Dict("run_name" => "finite_difference_cx0_split_1_moment",
               "evolve_moments_density" => true))

test_input_finite_difference_cx0_split_2_moments =
    merge(test_input_finite_difference_cx0_split_1_moment,
          Dict("run_name" => "finite_difference_cx0_split_2_moments",
               "evolve_moments_parallel_flow" => true))

test_input_finite_difference_cx0_split_3_moments =
    merge(test_input_finite_difference_cx0_split_2_moments,
          Dict("run_name" => "finite_difference_cx0_split_3_moments",
               "evolve_moments_parallel_pressure" => true))

test_input_chebyshev = merge(test_input_finite_difference,
                             Dict("run_name" => "chebyshev_pseudospectral",
                                  "r_discretization" => "chebyshev_pseudospectral",
                                  "r_ngrid" => 3,
                                  "r_nelement" => 1,
                                  "z_discretization" => "chebyshev_pseudospectral",
                                  "z_ngrid" => 3,
                                  "z_nelement" => 2,
                                  "vperp_discretization" => "chebyshev_pseudospectral",
                                  "vperp_ngrid" => 3,
                                  "vperp_nelement" => 1,
                                  "vpa_discretization" => "chebyshev_pseudospectral",
                                  "vpa_ngrid" => 3,
                                  "vpa_nelement" => 2,
                                  "vz_discretization" => "chebyshev_pseudospectral",
                                  "vz_ngrid" => 3,
                                  "vz_nelement" => 2,
                                  "vr_discretization" => "chebyshev_pseudospectral",
                                  "vr_ngrid" => 3,
                                  "vr_nelement" => 1,
                                  "vzeta_discretization" => "chebyshev_pseudospectral",
                                  "vzeta_ngrid" => 3,
                                  "vzeta_nelement" => 1))

test_input_chebyshev_split_1_moment =
    merge(test_input_chebyshev,
          Dict("run_name" => "chebyshev_pseudospectral_split_1_moment",
               "evolve_moments_density" => true))

test_input_chebyshev_split_2_moments =
    merge(test_input_chebyshev_split_1_moment,
          Dict("run_name" => "chebyshev_pseudospectral_split_2_moments",
               "evolve_moments_parallel_flow" => true))

test_input_chebyshev_split_3_moments =
    merge(test_input_chebyshev_split_2_moments,
          Dict("run_name" => "chebyshev_pseudospectral_split_3_moments",
               "evolve_moments_parallel_pressure" => true))

test_input_chebyshev_cx0 =
    merge(test_input_chebyshev,
          Dict("run_name" => "chebyshev_pseudospectral_cx0",
               "charge_exchange_frequency" => 0.0))

test_input_chebyshev_cx0_split_1_moment =
    merge(test_input_chebyshev_cx0,
          Dict("run_name" => "chebyshev_pseudospectral_cx0_split_1_moment",
               "evolve_moments_density" => true))

test_input_chebyshev_cx0_split_2_moments =
    merge(test_input_chebyshev_cx0_split_1_moment,
          Dict("run_name" => "chebyshev_pseudospectral_cx0_split_2_moments",
               "evolve_moments_parallel_flow" => true))

test_input_chebyshev_cx0_split_3_moments =
    merge(test_input_chebyshev_cx0_split_2_moments,
          Dict("run_name" => "chebyshev_pseudospectral_cx0_split_3_moments",
               "evolve_moments_parallel_pressure" => true))

test_input_chebyshev_1D1V =
    merge(test_input_finite_difference_1D1V,
          Dict("run_name" => "chebyshev_pseudospectral_1D1V",
               "z_discretization" => "chebyshev_pseudospectral",
               "z_ngrid" => 3,
               "z_nelement" => 2,
               "vpa_discretization" => "chebyshev_pseudospectral",
               "vpa_ngrid" => 3,
               "vpa_nelement" => 2,
               "vz_discretization" => "chebyshev_pseudospectral",
               "vz_ngrid" => 3,
               "vz_nelement" => 2))

test_input_chebyshev_1D1V_split_1_moment =
    merge(test_input_chebyshev_1D1V,
          Dict("run_name" => "chebyshev_pseudospectral_1D1V_split_1_moment",
               "evolve_moments_density" => true))

test_input_chebyshev_1D1V_split_2_moments =
    merge(test_input_chebyshev_1D1V_split_1_moment,
          Dict("run_name" => "chebyshev_pseudospectral_1D1V_split_2_moments",
               "evolve_moments_parallel_flow" => true))

test_input_chebyshev_1D1V_split_3_moments =
    merge(test_input_chebyshev_1D1V_split_2_moments,
          Dict("run_name" => "chebyshev_pseudospectral_1D1V_split_3_moments",
               "evolve_moments_parallel_pressure" => true, "runtime_plots" => true))

test_input_chebyshev_cx0_1D1V =
    merge(test_input_chebyshev_1D1V,
          Dict("run_name" => "chebyshev_pseudospectral_cx0_1D1V",
               "charge_exchange_frequency" => 0.0))

test_input_chebyshev_cx0_1D1V_split_1_moment =
    merge(test_input_chebyshev_cx0_1D1V,
          Dict("run_name" => "chebyshev_pseudospectral_cx0_1D1V_split_1_moment",
               "evolve_moments_density" => true))

test_input_chebyshev_cx0_1D1V_split_2_moments =
    merge(test_input_chebyshev_cx0_1D1V_split_1_moment,
          Dict("run_name" => "chebyshev_pseudospectral_cx0_1D1V_split_2_moments",
               "evolve_moments_parallel_flow" => true))

test_input_chebyshev_cx0_1D1V_split_3_moments =
    merge(test_input_chebyshev_cx0_1D1V_split_2_moments,
          Dict("run_name" => "chebyshev_pseudospectral_cx0_1D1V_split_3_moments",
               "evolve_moments_parallel_pressure" => true))

test_input_list = [
     #test_input_finite_difference,
     #test_input_finite_difference_split_1_moment,
     #test_input_finite_difference_split_2_moments,
     #test_input_finite_difference_split_3_moments,
     #test_input_finite_difference_cx0,
     #test_input_finite_difference_cx0_split_1_moment,
     #test_input_finite_difference_cx0_split_2_moments,
     #test_input_finite_difference_cx0_split_3_moments,
     #test_input_finite_difference_1D1V,
     #test_input_finite_difference_1D1V_split_1_moment,
     #test_input_finite_difference_1D1V_split_2_moments,
     #test_input_finite_difference_1D1V_split_3_moments,
     #test_input_finite_difference_cx0_1D1V,
     #test_input_finite_difference_cx0_1D1V_split_1_moment,
     #test_input_finite_difference_cx0_1D1V_split_2_moments,
     #test_input_finite_difference_cx0_1D1V_split_3_moments,
     test_input_chebyshev,
     #test_input_chebyshev_split_1_moment,
     #test_input_chebyshev_split_2_moments,
     #test_input_chebyshev_split_3_moments,
     test_input_chebyshev_cx0,
     #test_input_chebyshev_cx0_split_1_moment,
     #test_input_chebyshev_cx0_split_2_moments,
     #test_input_chebyshev_cx0_split_3_moments,
     test_input_chebyshev_1D1V,
     test_input_chebyshev_1D1V_split_1_moment,
     test_input_chebyshev_1D1V_split_2_moments,
     test_input_chebyshev_1D1V_split_3_moments,
     test_input_chebyshev_cx0_1D1V,
     test_input_chebyshev_cx0_1D1V_split_1_moment,
     test_input_chebyshev_cx0_1D1V_split_2_moments,
     test_input_chebyshev_cx0_1D1V_split_3_moments,
    ]
