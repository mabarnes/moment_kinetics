# Useful parameters
const z_L = 1.0 # always 1 in normalized units?
const vpa_L = sqrt(2) * 8.0

# Use very small number of points in vpa_expected to reduce the amount of entries we
# need to store. First and last entries are within the grid (rather than at the ends) in
# order to get non-zero values.
# Note: in the arrays of numbers for expected data, space-separated entries have to stay
# on the same line.
const expected =
  (
   z=[z for z in range(-0.5 * z_L, 0.5 * z_L, length=11)],
   vpa=[vpa for vpa in range(-0.2 * vpa_L, 0.2 * vpa_L, length=3)],
   phi=[-1.3862943611198908 -1.2382718388327256; -1.2115183691070186 -1.1306406580148494;
        -0.860986322750322 -0.872638240787864; -0.5494724738535592 -0.5903597696325978;
        -0.3534597636915602 -0.37552658452432697; -0.28768207245178135 -0.2921445457254851;
        -0.35345976369156035 -0.37552658452432724; -0.5494724738535595 -0.5903597696325981;
        -0.860986322750322 -0.8726382407878633; -1.2115183691070182 -1.13064065801485;
        -1.3862943611198906 -1.2382718388327256],
   n_ion=[0.25 0.2898847528848707; 0.29774575181738067 0.3228262223281206;
          0.42274575172974493 0.4178476204984502; 0.5772542482702547 0.5541281122300437;
          0.7022542481826191 0.6869277699984507; 0.7499999999999997 0.7466606015744714;
          0.7022542481826191 0.6869277699984505; 0.5772542482702547 0.5541281122300435;
          0.42274575172974477 0.4178476204984502; 0.2977457518173806 0.32282622232812047;
          0.25 0.28988475288487076],
   n_neutral=[0.7499999999999998 0.7736769684350007; 0.7022542481826196 0.7056866417564308;
              0.5772542482702552 0.5582978049742762; 0.42274575172974477 0.4096916252480846;
              0.29774575181738083 0.30539475187745263; 0.24999999999999992 0.2681959319511268;
              0.2977457518173809 0.3053947518774524; 0.42274575172974477 0.4096916252480848;
              0.5772542482702551 0.5582978049742763; 0.7022542481826195 0.70568664175643;
              0.7499999999999997 0.7736769684350008],
   upar_ion=sqrt(2) .* [5.1613576111288053e-17 -2.123545480084421e-16; 9.511571498528484e-18 -0.18237573505356336;
             3.467873011283547e-17 -0.19657153937328775; 3.1268365433125454e-17 -0.11139479678823913;
             -7.85680226742028e-17 -0.03369179943909123; 3.0405822071006615e-17 -7.142520624427162e-17;
             -4.6330440535052126e-17 0.033691799439091306; 7.978031114543488e-18 0.11139479678823902;
             -9.42922075719305e-18 0.1965715393732874; 4.905041261528767e-18 0.18237573505356316;
             5.161357611128805e-17 -1.98079721522979e-16],
   upar_neutral=sqrt(2) .* [-5.25681128612874e-18 -8.138292557219318e-18; -2.1410327957680203e-17 -0.03620919086900061;
                 -3.323217164475685e-17 -0.009155710507522782; 4.966838011974837e-17 0.054522829689858866;
                 6.9299624525258045e-18 0.07606842244489476; -1.391317598583754e-17 1.1238771957348959e-16;
                 4.654198359544338e-19 -0.07606842244489485; 2.1256669382510767e-17 -0.05452282968985885;
                 2.1970070136117582e-17 0.009155710507522738; -8.263186521090134e-18 0.036209190869000665;
                 -5.256811286128741e-18 -7.622565185438491e-18],
   ppar_ion=2 .* [0.18750000000000008 0.23281690067278377; 0.20909325514551103 0.2191254545255386;
             0.24403180771238261 0.20822592737403486; 0.24403180771238275 0.21506759664463654;
             0.2090932551455112 0.22097089988531768; 0.1875 0.22119051348454336;
             0.2090932551455112 0.2209708998853175; 0.24403180771238273 0.21506759664463654;
             0.24403180771238245 0.20822592737403486; 0.20909325514551103 0.21912545452553867;
             0.18750000000000008 0.23281690067278374],
   ppar_neutral=2 .* [0.18750000000000006 0.24802447121079244; 0.20909325514551114 0.24400764814887374;
                 0.24403180771238295 0.22861292459492588; 0.2440318077123829 0.2058895439942422;
                 0.20909325514551147 0.1926358795649716; 0.18750000000000006 0.19091788167327758;
                 0.20909325514551147 0.19263587956497163; 0.24403180771238278 0.20588954399424197;
                 0.2440318077123828 0.2286129245949259; 0.2090932551455111 0.24400764814887368;
                 0.18750000000000006 0.24802447121079246],
   f_ion=(1 / sqrt(2 * π)) .* [0.03704633061445434 0.040599341664601864 0.04284314970873867 0.030398267056148856 0.012360459027428135 0.006338529470381567 0.012360459027428118 0.030398267056148832 0.04284314970873862 0.04059934166460186 0.03704633061445434;
          0.2041161581761397 0.2512319184948212 0.3934412241088816 0.6277900647663583 0.9100364506644036 1.0606601717797792 0.9100364506644041 0.6277900647663586 0.3934412241088818 0.25123191849482107 0.20411615817613982;
          0.03704633061445434 0.040599341664601864 0.04284314970873867 0.030398267056148856 0.012360459027428135 0.006338529470381567 0.012360459027428118 0.030398267056148832 0.04284314970873862 0.04059934166460186 0.03704633061445434;;;
          0.053924384494760155 0.06057845439557781 0.036767488220049674 0.013740511163603739 0.010777324650110652 0.0193303624862901 0.02798217512766377 0.027603098839369202 0.026679828448322784 0.035654591961970045 0.05392438449476015;
          0.21177343692536066 0.24902654604956825 0.3729373156851441 0.5962815376599825 0.8870867557870747 1.0533860766907464 0.8870867557870745 0.5962815376599826 0.3729373156851444 0.249026546049568 0.21177343692536063;
          0.053924384494760155 0.03565459196197003 0.026679828448322816 0.027603098839369188 0.027982175127663798 0.01933036248629009 0.010777324650110682 0.01374051116360376 0.03676748822004965 0.06057845439557782 0.05392438449476016],
   f_neutral=(1 / sqrt(2 * π)) .* [0.006338529470381577 0.012360459027428073 0.03039826705614889 0.042843149708738676 0.04059934166460194 0.03704633061445434 0.040599341664601926 0.04284314970873865 0.03039826705614885 0.012360459027428083 0.006338529470381584;
              1.0606601717797781 0.9100364506644045 0.6277900647663583 0.39344122410888155 0.2512319184948211 0.20411615817613965 0.2512319184948211 0.3934412241088815 0.6277900647663583 0.9100364506644045 1.0606601717797786;
              0.006338529470381577 0.012360459027428073 0.03039826705614889 0.042843149708738676 0.04059934166460194 0.03704633061445434 0.040599341664601926 0.04284314970873865 0.03039826705614885 0.012360459027428083 0.006338529470381584;;;
              0.024285046706911284 0.040712372422460126 0.041904880509743704 0.036374621413627864 0.036923488429025 0.04165074679893949 0.03672486041712861 0.019283695999620983 0.008424202760799221 0.010011785865586771 0.02428504670691128;
              1.0530028930305488 0.9036794951365155 0.6251037108251192 0.3955246177555597 0.25710841311808186 0.21139025326516536 0.25710841311808175 0.39552461775555947 0.6251037108251194 0.9036794951365155 1.053002893030549;
              0.024285046706911263 0.010011785865586764 0.008424202760799245 0.019283695999620955 0.03672486041712864 0.041650746798939486 0.036923488429025014 0.03637462141362787 0.04190488050974369 0.040712372422460195 0.024285046706911267])

# default inputs for tests
test_input_finite_difference = OptionsDict("composition" => OptionsDict("n_ion_species" => 1,
                                                                        "n_neutral_species" => 1,
                                                                        "electron_physics" => "boltzmann_electron_response",
                                                                        "T_e" => 1.0),
                                           "ion_species_1" => OptionsDict("initial_density" => 0.5,
                                                                          "initial_temperature" => 0.3333333333333333),
                                           "z_IC_ion_species_1" => OptionsDict("initialization_option" => "sinusoid",
                                                                               "density_amplitude" => 0.5,
                                                                               "density_phase" => 0.0,
                                                                               "upar_amplitude" => 0.0,
                                                                               "upar_phase" => 0.0,
                                                                               "temperature_amplitude" => 0.5,
                                                                               "temperature_phase" => mk_float(π)),
                                           "neutral_species_1" => OptionsDict("initial_density" => 0.5,
                                                                              "initial_temperature" => 0.3333333333333333),
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
                                                                           "pressure" => false,
                                                                           "moments_conservation" => true),
                                           "reactions" => OptionsDict("charge_exchange_frequency" => 0.8885765876316732,
                                                                      "ionization_frequency" => 0.0),
                                           "timestepping" => OptionsDict("nstep" => 100,
                                                                         "dt" => 0.0007071067811865475,
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
                                "evolve_moments" => OptionsDict("pressure" => true),
                                "vpa" => OptionsDict("L" => 20.784609690826528),
                                "vz" => OptionsDict("L" => 20.784609690826528),
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
                                "evolve_moments" => OptionsDict("pressure" => true),
                                "vpa" => OptionsDict("L" => 20.784609690826528),
                                "vz" => OptionsDict("L" => 20.784609690826528),
                               ))
