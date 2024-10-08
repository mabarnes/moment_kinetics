# Useful parameters
const z_L = 1.0 # always 1 in normalized units?
const vpa_L = 8.0

# The expected output
struct expected_data
    z::Array{mk_float, 1}
    vpa::Array{mk_float, 1}
    phi::Array{mk_float, 2}
    n_ion::Array{mk_float, 2}
    n_neutral::Array{mk_float, 2}
    upar_ion::Array{mk_float, 2}
    upar_neutral::Array{mk_float, 2}
    ppar_ion::Array{mk_float, 2}
    ppar_neutral::Array{mk_float, 2}
    f_ion::Array{mk_float, 3}
    f_neutral::Array{mk_float, 3}
end

# Use very small number of points in vpa_expected to reduce the amount of entries we
# need to store. First and last entries are within the grid (rather than at the ends) in
# order to get non-zero values.
# Note: in the arrays of numbers for expected data, space-separated entries have to stay
# on the same line.
const expected =
  (
   z=[z for z in range(-0.5 * z_L, 0.5 * z_L, length=11)],
   vpa=[vpa for vpa in range(-0.2 * vpa_L, 0.2 * vpa_L, length=3)],
   phi=[-1.386282080324426 -1.2382641646968997; -1.2115129555832849 -1.130635565831393;
        -0.8609860698164498 -0.872637046489647; -0.5494724768120176 -0.5903597644911374;
        -0.35345976364887166 -0.37552658974835484; -0.28768207245186167 -0.2921445534164449;
        -0.353459763648872 -0.3755265897483555; -0.5494724768120175 -0.5903597644911376;
        -0.8609860698164502 -0.8726370464896476; -1.2115129555832849 -1.1306355658313922;
        -1.3862820803244258 -1.2382641646968997],
   n_ion=[0.2500030702177186 0.2898869775083742; 0.2977473631375158 0.3228278662412625;
          0.42274585818529853 0.417848119539277; 0.5772542465450629 0.5541281150892785;
          0.7022542481909738 0.6869277664245242; 0.7499999999999394 0.7466605958319346;
          0.7022542481909738 0.6869277664245237; 0.577254246545063 0.5541281150892783;
          0.42274585818529864 0.41784811953927686; 0.2977473631375159 0.32282786624126253;
          0.2500030702177185 0.2898869775083743],
   n_neutral=[0.7499999999999382 0.7736769553678673; 0.7022542481909748 0.7056866352169496;
              0.5772542465450632 0.5582977481633454; 0.4227458581852985 0.40969188756651037;
              0.29774736313751604 0.30539644783353687; 0.2500030702177186 0.268198658560817;
              0.29774736313751604 0.305396447833537; 0.42274585818529836 0.4096918875665103;
              0.5772542465450631 0.5582977481633457; 0.7022542481909745 0.7056866352169494;
              0.7499999999999383 0.7736769553678673],
   upar_ion=[-2.7135787559953277e-17 -6.299214622140781e-17; -9.321028970172899e-18 -0.1823721921091055;
             -2.8374879811351724e-18 -0.19657035490893093; 1.2124327390522635e-17 -0.11139486685283827;
             3.6525788403693063e-17 -0.033691837771623996; -2.0930856430671915e-17 4.84147091991613e-17;
             8.753545920086251e-18 0.033691837771624024; 1.1293771270243255e-17 0.11139486685283813;
             1.3739171132886587e-17 0.19657035490893102; -6.840453743089351e-18 0.18237219210910513;
             -2.7135787559953277e-17 -4.656897959900552e-17],
   upar_neutral=[6.5569385065066925e-18 7.469475342067322e-17; 1.1054500872839027e-17 -0.036209130454625794;
                 -3.241833393685864e-17 -0.00915544640981337; -3.617637280460899e-17 0.05452268209340691;
                 4.417578961284041e-17 0.07606644718003618; 4.9354467746194965e-17 4.452343983947504e-17;
                 6.573091229872379e-18 -0.07606644718003616; 2.989662686945165e-17 -0.05452268209340687;
                 -3.1951996361666834e-17 0.009155446409813412; -4.395464518158184e-18 0.03620913045462582;
                 6.5569385065066925e-18 7.150569974151354e-17],
   ppar_ion=[0.18749999999999992 0.2328164829490338; 0.20909325514551116 0.21912575009260987;
             0.24403180771238264 0.20822611102296495; 0.24403180771238278 0.21506741942934832;
             0.2090932551455113 0.22097085045011763; 0.1875 0.22119050467096843;
             0.20909325514551128 0.2209708504501176; 0.2440318077123828 0.2150674194293483;
             0.24403180771238256 0.20822611102296476; 0.20909325514551116 0.21912575009260982;
             0.18749999999999992 0.2328164829490338],
   ppar_neutral=[0.18750000000000003 0.2480244331470989; 0.20909325514551122 0.2440075646485762;
                 0.24403180771238286 0.22861256884534023; 0.24403180771238278 0.20588932618946498;
                 0.20909325514551144 0.19263633346696638; 0.18749999999999992 0.19091848744561835;
                 0.20909325514551141 0.19263633346696654; 0.2440318077123828 0.20588932618946482;
                 0.24403180771238286 0.22861256884534029; 0.20909325514551114 0.24400756464857642;
                 0.18750000000000006 0.24802443314709893],
   f_ion=[0.0370462360994826 0.04059927063892091 0.0428431419871786 0.030398267195914062 0.01236045902698859 0.006338529470383425 0.012360459026988587 0.030398267195914028 0.04284314198717859 0.0405992706389209 0.0370462360994826;
          0.20411991941198782 0.25123395823993105 0.3934413727192304 0.6277900619432855 0.9100364506661008 1.0606601717796504 0.910036450666101 0.6277900619432859 0.39344137271923046 0.25123395823993094 0.20411991941198776;
          0.0370462360994826 0.04059927063892091 0.0428431419871786 0.030398267195914062 0.01236045902698859 0.006338529470383425 0.012360459026988587 0.030398267195914028 0.04284314198717859 0.0405992706389209 0.0370462360994826;;;
          0.05392403019146985 0.06057819609646438 0.03676744157455075 0.013740507879552622 0.010777319583092297 0.019330359159894384 0.027982173790396116 0.027603104735767332 0.02667986700464528 0.035654512254837005 0.05392403019146984;
          0.21177720235387912 0.24902901234066305 0.3729377138332225 0.596281539172339 0.8870867512643452 1.0533860567375264 0.887086751264345 0.5962815391723388 0.3729377138332225 0.24902901234066285 0.21177720235387912;
          0.053924030191469796 0.035654512254837074 0.02667986700464531 0.02760310473576733 0.02798217379039615 0.019330359159894287 0.010777319583092311 0.013740507879552624 0.03676744157455069 0.060578196096464365 0.05392403019146979],
   f_neutral=[0.0063385294703834595 0.012360459026988546 0.030398267195914108 0.04284314198717859 0.040599270638920985 0.03704623609948259 0.040599270638920965 0.0428431419871786 0.030398267195914094 0.012360459026988546 0.006338529470383456;
              1.0606601717796493 0.9100364506661016 0.6277900619432857 0.3934413727192303 0.2512339582399308 0.20411991941198754 0.2512339582399307 0.3934413727192301 0.6277900619432853 0.9100364506661016 1.0606601717796487;
              0.0063385294703834595 0.012360459026988546 0.030398267195914108 0.04284314198717859 0.040599270638920985 0.03704623609948259 0.040599270638920965 0.0428431419871786 0.030398267195914094 0.012360459026988546 0.006338529470383456;;;
              0.024285034070612683 0.04071236753946936 0.04190483876050118 0.036374533667106086 0.0369234055803037 0.04165072188572372 0.03672486160719089 0.019283695804388743 0.008424202166370513 0.010011778724856858 0.02428503407061268;
              1.05300288883775 0.9036794996386066 0.6251037063201776 0.39552476644559265 0.25711045639921726 0.2113940344541052 0.25711045639921726 0.39552476644559253 0.6251037063201775 0.9036794996386066 1.0530028888377503;
              0.024285034070612672 0.01001177872485688 0.00842420216637052 0.019283695804388705 0.03672486160719087 0.04165072188572364 0.0369234055803037 0.036374533667106045 0.041904838760501203 0.040712367539469434 0.02428503407061266])

# default inputs for tests
test_input_finite_difference = OptionsDict("composition" => OptionsDict("n_ion_species" => 1,
                                                                        "n_neutral_species" => 1,
                                                                        "electron_physics" => "boltzmann_electron_response",
                                                                        "T_e" => 1.0),
                                           "ion_species_1" => OptionsDict("initial_density" => 0.5,
                                                                          "initial_temperature" => 1.0),
                                           "z_IC_ion_species_1" => OptionsDict("initialization_option" => "sinusoid",
                                                                               "density_amplitude" => 0.5,
                                                                               "density_phase" => 0.0,
                                                                               "upar_amplitude" => 0.0,
                                                                               "upar_phase" => 0.0,
                                                                               "temperature_amplitude" => 0.5,
                                                                               "temperature_phase" => mk_float(π)),
                                           "neutral_species_1" => OptionsDict("initial_density" => 0.5,
                                                                              "initial_temperature" => 1.0),
                                           "z_IC_neutral_species_1" => OptionsDict("initialization_option" => "sinusoid",
                                                                                   "density_amplitude" => 0.5,
                                                                                   "density_phase" => mk_float(π),
                                                                                   "upar_amplitude" => 0.0,
                                                                                   "upar_phase" => 0.0,
                                                                                   "temperature_amplitude" => 0.5,
                                                                                   "temperature_phase" => 0.0),    
                                           "output" => OptionsDict("run_name" => "finite_difference"),
                                           "evolve_moments" => OptionsDict("density" => false,
                                                                           "parallel_flow" => false,
                                                                           "parallel_pressure" => false,
                                                                           "moments_conservation" => true),
                                           "reactions" => OptionsDict("charge_exchange_frequency" => 2*π*0.1,
                                                                      "ionization_frequency" => 0.0),
                                           "timestepping" => OptionsDict("nstep" => 100,
                                                                         "dt" => 0.001,
                                                                         "nwrite" => 100,
                                                                         "nwrite_dfns" => 100,
                                                                         "split_operators" => false),
                                           "r" => OptionsDict("ngrid" => 1,
                                                              "nelement" => 1,
                                                              "bc" => "periodic",
                                                              "discretization" => "finite_difference"),
                                           "z" => OptionsDict("ngrid" => 100,
                                                              "nelement" => 1,
                                                              "bc" => "periodic",
                                                              "discretization" => "finite_difference"),
                                           "vpa" => OptionsDict("ngrid" => 400,
                                                                "nelement" => 1,
                                                                "L" => vpa_L,
                                                                "bc" => "periodic",
                                                                "discretization" => "finite_difference"),
                                           "vz" => OptionsDict("ngrid" => 400,
                                                               "nelement" => 1,
                                                               "L" => vpa_L,
                                                               "bc" => "periodic",
                                                               "discretization" => "finite_difference"),
                                          )

test_input_finite_difference_split_1_moment =
    recursive_merge(test_input_finite_difference,
                    OptionsDict("output" => OptionsDict("run_name" => "finite_difference_split_1_moment"),
                                "evolve_moments" => OptionsDict("density" => true)))

test_input_finite_difference_split_2_moments =
    recursive_merge(test_input_finite_difference_split_1_moment,
                    OptionsDict("output" => OptionsDict("run_name" => "finite_difference_split_2_moments"),
                                "evolve_moments" => OptionsDict("parallel_flow" => true)))

test_input_finite_difference_split_3_moments =
    recursive_merge(test_input_finite_difference_split_2_moments,
                    OptionsDict("output" => OptionsDict("run_name" => "finite_difference_split_3_moments"),
                                "evolve_moments" => OptionsDict("parallel_pressure" => true),
                                "vpa" => OptionsDict("L" => 12.0),
                                "vz" => OptionsDict("L" => 12.0),
                               ))

test_input_chebyshev = recursive_merge(test_input_finite_difference,
                                       OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral"),
                                                   "z" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                                                      "ngrid" => 9,
                                                                      "nelement" => 4),
                                                   "vpa" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                                                        "ngrid" => 17,
                                                                        "nelement" => 8),
                                                   "vz" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                                                       "ngrid" => 17,
                                                                       "nelement" => 8)),
                                      )

if global_size[] > 2 && global_size[] % 2 == 0
    # Test using distributed-memory
    test_input_chebyshev["z"]["nelement_local"] = test_input_chebyshev["z"]["nelement"] ÷ 2
end

test_input_chebyshev_split_1_moment =
    recursive_merge(test_input_chebyshev,
                    OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral_split_1_moment"),
                                "evolve_moments" => OptionsDict("density" => true)))

test_input_chebyshev_split_2_moments =
    recursive_merge(test_input_chebyshev_split_1_moment,
                    OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral_split_2_moments"),
                                "evolve_moments" => OptionsDict("parallel_flow" => true)))

test_input_chebyshev_split_3_moments =
    recursive_merge(test_input_chebyshev_split_2_moments,
                    OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral_split_3_moments"),
                                "evolve_moments" => OptionsDict("parallel_pressure" => true),
                                "vpa" => OptionsDict("L" => 12.0),
                                "vz" => OptionsDict("L" => 12.0),
                               ))
