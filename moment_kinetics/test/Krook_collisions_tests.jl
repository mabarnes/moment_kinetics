module KrookCollisionsTests

# Test for Krook collision operator, based on NonlinearSoundWave test

include("setup.jl")

using Base.Filesystem: tempname

using moment_kinetics.load_data: open_readonly_output_file, load_coordinate_data,
                                 load_species_data, load_fields_data,
                                 load_ion_moments_data, load_pdf_data,
                                 load_neutral_particle_moments_data,
                                 load_neutral_pdf_data, load_time_data, load_species_data
using moment_kinetics.interpolation: interpolate_to_grid_z, interpolate_to_grid_vpa
using moment_kinetics.type_definitions: mk_float
using moment_kinetics.utils: merge_dict_with_kwargs!

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
     phi=[-1.3862943611198908 -1.2382457888717142; -1.2115183691070186 -1.1306196421498953;
          -0.860986322750322 -0.8726520952244605; -0.5494724738535592 -0.5904511211246796;
          -0.3534597636915602 -0.37557956070712833; -0.28768207245178135 -0.29192141692451096;
          -0.35345976369156035 -0.37557956070712845; -0.5494724738535595 -0.5904511211246797;
          -0.860986322750322 -0.8726520952244602; -1.2115183691070182 -1.1306196421498949;
          -1.3862943611198906 -1.2382457888717144],
     n_ion=[0.25 0.2898923044697399; 0.29774575181738067 0.3228330050536253;
            0.42274575172974493 0.4178418298065724; 0.5772542482702547 0.5540775176682499;
            0.7022542481826191 0.686891421264091; 0.7499999999999997 0.7468272216474126;
            0.7022542481826191 0.6868914212640906; 0.5772542482702547 0.5540775176682496;
            0.42274575172974477 0.4178418298065727; 0.2977457518173806 0.3228330050536256;
            0.25 0.28989230446973996],
     n_neutral=[0.7499999999999998 0.7736771072214454; 0.7022542481826196 0.705686714081075;
                0.5772542482702552 0.5582976228204487; 0.42274575172974477 0.40969113303014065;
                0.29774575181738083 0.30539471647964855; 0.24999999999999992 0.26819707572106;
                0.2977457518173809 0.3053947164796483; 0.42274575172974477 0.4096911330301402;
                0.5772542482702551 0.5582976228204486; 0.7022542481826195 0.705686714081075;
                0.7499999999999997 0.7736771072214454],
     upar_ion=sqrt(2) .* [5.1613576111288053e-17 1.7824283715661693e-16; 9.511571498528484e-18 -0.18246293837210678;
               3.467873011283547e-17 -0.19666573102552073; 3.1268365433125454e-17 -0.11128036402813761;
               -7.85680226742028e-17 -0.03317981976875931; 3.0405822071006615e-17 -4.294965262347655e-17;
               -4.6330440535052126e-17 0.03317981976875943; 7.978031114543488e-18 0.11128036402813778;
               -9.42922075719305e-18 0.19666573102552032; 4.905041261528767e-18 0.18246293837210653;
               5.161357611128805e-17 1.6437677472493942e-16],
     upar_neutral=sqrt(2) .* [-5.25681128612874e-18 8.998878031629687e-17; -2.1410327957680203e-17 -0.036209944927282176;
                   -3.323217164475685e-17 -0.009156342266174351; 4.966838011974837e-17 0.05452637956260735;
                   6.9299624525258045e-18 0.07608073432552652; -1.391317598583754e-17 -1.2322635316655561e-16;
                   4.654198359544338e-19 -0.0760807343255267; 2.1256669382510767e-17 -0.05452637956260749;
                   2.1970070136117582e-17 0.009156342266174294; -8.263186521090134e-18 0.036209944927282155;
                   -5.256811286128741e-18 8.865786794476116e-17],
     p_ion=(2/3) .* [0.18750000000000008 0.23302774299822557; 0.20909325514551103 0.21936769787415103;
               0.24403180771238261 0.20856277641657595; 0.24403180771238275 0.2154268106106535;
               0.2090932551455112 0.2206184395140329; 0.1875 0.21979740311704793;
               0.2090932551455112 0.22061843951403276; 0.24403180771238273 0.21542681061065366;
               0.24403180771238245 0.20856277641657608; 0.20909325514551103 0.21936769787415114;
               0.18750000000000008 0.23302774299822557],
     p_neutral=(2/3) .* [0.18750000000000006 0.24802927638746003; 0.20909325514551114 0.2440126345473401;
                   0.24403180771238295 0.2286179898140519; 0.2440318077123829 0.20589247230117314;
                   0.20909325514551147 0.19263091603124238; 0.18750000000000006 0.1909059116301176;
                   0.20909325514551147 0.19263091603124244; 0.24403180771238278 0.20589247230117316;
                   0.2440318077123828 0.22861798981405207; 0.2090932551455111 0.24401263454734015;
                   0.18750000000000006 0.24802927638746008],
     f_ion=(1 / sqrt(2 * π)) .* [0.03704633061445434 0.040599341664601864 0.04284314970873867 0.030398267056148856 0.012360459027428135 0.006338529470381567 0.012360459027428118 0.030398267056148832 0.04284314970873862 0.04059934166460186 0.03704633061445434;
            0.2041161581761397 0.2512319184948212 0.3934412241088816 0.6277900647663583 0.9100364506644036 1.0606601717797792 0.9100364506644041 0.6277900647663586 0.3934412241088818 0.25123191849482107 0.20411615817613982;
            0.03704633061445434 0.040599341664601864 0.04284314970873867 0.030398267056148856 0.012360459027428135 0.006338529470381567 0.012360459027428118 0.030398267056148832 0.04284314970873862 0.04059934166460186 0.03704633061445434;;;
            0.05390003952306185 0.060668903632275895 0.037468714568864685 0.014783443991530101 0.010917696554920737 0.018422975159286537 0.027170955317800925 0.027269141882307273 0.026567530213466868 0.03561275246167625 0.053900039523061834;
            0.21183314018321575 0.24917189857143243 0.37345407790645385 0.597221918072211 0.8859681875060791 1.0485989094790735 0.8859681875060792 0.5972219180722116 0.3734540779064541 0.24917189857143285 0.21183314018321578;
            0.05390003952306177 0.0356127524616763 0.026567530213466896 0.02726914188230729 0.027170955317800928 0.018422975159286543 0.010917696554920774 0.014783443991530103 0.037468714568864664 0.060668903632275964 0.053900039523061806],
     f_neutral=(1 / sqrt(2 * π)) .* [0.006338529470381577 0.012360459027428073 0.03039826705614889 0.042843149708738676 0.04059934166460194 0.03704633061445434 0.040599341664601926 0.04284314970873865 0.03039826705614885 0.012360459027428083 0.006338529470381584;
                1.0606601717797781 0.9100364506644045 0.6277900647663583 0.39344122410888155 0.2512319184948211 0.20411615817613965 0.2512319184948211 0.3934412241088815 0.6277900647663583 0.9100364506644045 1.0606601717797786;
                0.006338529470381577 0.012360459027428073 0.03039826705614889 0.042843149708738676 0.04059934166460194 0.03704633061445434 0.040599341664601926 0.04284314970873865 0.03039826705614885 0.012360459027428083 0.006338529470381584;;;
                0.024284901305335987 0.04071460848299759 0.04191393294584067 0.03638224538383874 0.03692291383553763 0.04164451713956093 0.03671950659707693 0.0192811926383855 0.008423252942010866 0.010011399856724638 0.02428490130533598;
                1.053003364698727 0.9036809824213016 0.6251085384477069 0.39553074816501577 0.2571014809014902 0.21136781088988876 0.2571014809014902 0.3955307481650158 0.6251085384477073 0.9036809824213019 1.0530033646987271;
                0.024284901305336 0.01001139985672466 0.00842325294201086 0.01928119263838556 0.03671950659707701 0.041644517139560876 0.03692291383553763 0.036382245383838704 0.041913932945840685 0.040714608482997565 0.024284901305336015])

# default inputs for tests
test_input_full_f = OptionsDict("composition" => OptionsDict("n_ion_species" => 1,
                                                             "n_neutral_species" => 1,
                                                             "electron_physics" => "boltzmann_electron_response",
                                                             "T_e" => 1.0,
                                                             "T_wall" => 0.3333333333333333),
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
                                "output" => OptionsDict("run_name" => "full_f"),
                                "evolve_moments" => OptionsDict("density" => false,
                                                                "parallel_flow" => false,
                                                                "pressure" => false,
                                                                "moments_conservation" => true),
                                "krook_collisions" => OptionsDict("use_krook" => true,
                                                                  "frequency_option" => "reference_parameters"),
                                "reactions" => OptionsDict("charge_exchange_frequency" => 0.8885765876316732,
                                                           "ionization_frequency" => 0.0),
                                "timestepping" => OptionsDict("nstep" => 100,
                                                              "dt" => 0.0007071067811865475,
                                                              "nwrite" => 100,
                                                              "nwrite_dfns" => 100,
                                                              "split_operators" => false),
                                "r" => OptionsDict("ngrid" => 1,
                                                   "nelement" => 1,
                                                   "discretization" => "chebyshev_pseudospectral"),
                                "z" => OptionsDict("ngrid" => 9,
                                                   "nelement" => 4,
                                                   "bc" => "periodic",
                                                   "discretization" => "chebyshev_pseudospectral"),
                                "vpa" => OptionsDict("ngrid" => 17,
                                                     "nelement" => 8,
                                                     "L" => vpa_L,
                                                     "bc" => "periodic",
                                                     "discretization" => "chebyshev_pseudospectral"),
                                "vz" => OptionsDict("ngrid" => 17,
                                                    "nelement" => 8,
                                                    "L" => vpa_L,
                                                    "bc" => "periodic",
                                                    "discretization" => "chebyshev_pseudospectral"),
                               )

test_input_split_1_moment =
    recursive_merge(test_input_full_f,
                    OptionsDict("output" => OptionsDict("run_name" => "split_1_moment"),
                                "evolve_moments" => OptionsDict("density" => true)))

test_input_split_2_moments =
    recursive_merge(test_input_split_1_moment,
                    OptionsDict("output" => OptionsDict("run_name" => "split_2_moments"),
                                "evolve_moments" => OptionsDict("parallel_flow" => true)))

test_input_split_3_moments =
    recursive_merge(test_input_split_2_moments,
                    OptionsDict("output" => OptionsDict("run_name" => "split_3_moments"),
                                "evolve_moments" => OptionsDict("pressure" => true),
                                "vpa" => OptionsDict("L" => 20.784609690826528),
                                "vz" => OptionsDict("L" => 20.784609690826528),
                               ))


"""
Run a sound-wave test for a single set of parameters
"""
# Note 'name' should not be shared by any two tests in this file
function run_test(test_input, rtol, atol; args...)
    # by passing keyword arguments to run_test, args becomes a Dict which can be used to
    # update the default inputs

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
    quietoutput() do
        # run simulation
        run_moment_kinetics(input)
    end

    phi = nothing
    n_ion = nothing
    upar_ion = nothing
    p_ion = nothing
    f_ion = nothing
    n_neutral = nothing
    upar_neutral = nothing
    p_neutral = nothing
    f_neutral = nothing
    z, z_spectral = nothing, nothing
    vpa, vpa_spectral = nothing, nothing

    if global_rank[] == 0
        quietoutput() do

            # Load and analyse output
            #########################

            path = joinpath(realpath(input["output"]["base_directory"]), name, name)

            # open the netcdf file containing moments data and give it the handle 'fid'
            fid = open_readonly_output_file(path, "moments")

            # load species, time coordinate data
            n_ion_species, n_neutral_species = load_species_data(fid)
            ntime, time = load_time_data(fid)
            n_ion_species, n_neutral_species = load_species_data(fid)
            
            # load fields data
            phi_zrt, Er_zrt, Ez_zrt = load_fields_data(fid)

            # load velocity moments data
            n_ion_zrst, upar_ion_zrst, p_ion_zrst, ppar_ion_zrst, qpar_ion_zrst, v_t_ion_zrst = load_ion_moments_data(fid)
            n_neutral_zrst, upar_neutral_zrst, p_neutral_zrst, pz_neutral_zrst, qpar_neutral_zrst, v_t_neutral_zrst = load_neutral_particle_moments_data(fid)
            z, z_spectral = load_coordinate_data(fid, "z"; ignore_MPI=true)

            close(fid)
            
            # open the netcdf file containing pdf data
            fid = open_readonly_output_file(path, "dfns")
            
            # load particle distribution function (pdf) data
            f_ion_vpavperpzrst = load_pdf_data(fid)
            f_neutral_vzvrvzetazrst = load_neutral_pdf_data(fid)
            vpa, vpa_spectral = load_coordinate_data(fid, "vpa"; ignore_MPI=true)

            close(fid)
            
            phi = phi_zrt[:,1,:]
            n_ion = n_ion_zrst[:,1,:,:]
            upar_ion = upar_ion_zrst[:,1,:,:]
            p_ion = p_ion_zrst[:,1,:,:]
            qpar_ion = qpar_ion_zrst[:,1,:,:]
            v_t_ion = v_t_ion_zrst[:,1,:,:]
            f_ion = f_ion_vpavperpzrst[:,1,:,1,:,:]
            n_neutral = n_neutral_zrst[:,1,:,:]
            upar_neutral = upar_neutral_zrst[:,1,:,:]
            p_neutral = p_neutral_zrst[:,1,:,:]
            qpar_neutral = qpar_neutral_zrst[:,1,:,:]
            v_t_neutral = v_t_neutral_zrst[:,1,:,:]
            f_neutral = f_neutral_vzvrvzetazrst[:,1,1,:,1,:,:]

            # Unnormalize f
            if input["evolve_moments"]["density"]
                for it ∈ 1:length(time), is ∈ 1:n_ion_species, iz ∈ 1:z.n
                    f_ion[:,iz,is,it] .*= n_ion[iz,is,it]
                end
                for it ∈ 1:length(time), isn ∈ 1:n_neutral_species, iz ∈ 1:z.n
                    f_neutral[:,iz,isn,it] .*= n_neutral[iz,isn,it]
                end
            end
            if input["evolve_moments"]["pressure"]
                for it ∈ 1:length(time), is ∈ 1:n_ion_species, iz ∈ 1:z.n
                    f_ion[:,iz,is,it] ./= v_t_ion[iz,is,it]
                end
                for it ∈ 1:length(time), isn ∈ 1:n_neutral_species, iz ∈ 1:z.n
                    f_neutral[:,iz,isn,it] ./= v_t_neutral[iz,isn,it]
                end
            end
        end

        # Test against values interpolated onto 'expected' grid which is fairly coarse no we
        # do not have to save too much data in this file

        # Use commented-out lines to get the test data to put in `expected`
        #newgrid_phi = cat(interpolate_to_grid_z(expected.z, phi[:, 1], z, z_spectral),
        #                   interpolate_to_grid_z(expected.z, phi[:, 2], z, z_spectral);
        #                   dims=2)
        #println("phi ", size(newgrid_phi))
        #println(newgrid_phi)
        #println()
        #newgrid_n_ion = cat(interpolate_to_grid_z(expected.z, n_ion[:, :, 1], z, z_spectral)[:,1],
        #                        interpolate_to_grid_z(expected.z, n_ion[:, :, 2], z, z_spectral)[:,1];
        #                        dims=2)
        #println("n_ion ", size(newgrid_n_ion))
        #println(newgrid_n_ion)
        #println()
        #newgrid_n_neutral = cat(interpolate_to_grid_z(expected.z, n_neutral[:, :, 1], z, z_spectral)[:,1],
        #                        interpolate_to_grid_z(expected.z, n_neutral[:, :, 2], z, z_spectral)[:,1];
        #                        dims=2)
        #println("n_neutral ", size(newgrid_n_neutral))
        #println(newgrid_n_neutral)
        #println()
        #newgrid_upar_ion = cat(interpolate_to_grid_z(expected.z, upar_ion[:, :, 1], z, z_spectral)[:,1],
        #                           interpolate_to_grid_z(expected.z, upar_ion[:, :, 2], z, z_spectral)[:,1];
        #                           dims=2)
        #println("upar_ion ", size(newgrid_upar_ion))
        #println(newgrid_upar_ion)
        #println()
        #newgrid_upar_neutral = cat(interpolate_to_grid_z(expected.z, upar_neutral[:, :, 1], z, z_spectral)[:,1],
        #                           interpolate_to_grid_z(expected.z, upar_neutral[:, :, 2], z, z_spectral)[:,1];
        #                           dims=2)
        #println("upar_neutral ", size(newgrid_upar_neutral))
        #println(newgrid_upar_neutral)
        #println()
        #newgrid_p_ion = cat(interpolate_to_grid_z(expected.z, p_ion[:, :, 1], z, z_spectral)[:,1],
        #                    interpolate_to_grid_z(expected.z, p_ion[:, :, 2], z, z_spectral)[:,1];
        #                    dims=2)
        #println("p_ion ", size(newgrid_p_ion))
        #println(newgrid_p_ion)
        #println()
        #newgrid_p_neutral = cat(interpolate_to_grid_z(expected.z, p_neutral[:, :, 1], z, z_spectral)[:,1],
        #                        interpolate_to_grid_z(expected.z, p_neutral[:, :, 2], z, z_spectral)[:,1];
        #                        dims=2)
        #println("p_neutral ", size(newgrid_p_neutral))
        #println(newgrid_p_neutral)
        #println()
        #newgrid_f_ion = cat(interpolate_to_grid_vpa(expected.vpa, interpolate_to_grid_z(expected.z, f_ion[:, :, :, 1], z, z_spectral), vpa, vpa_spectral)[:,:,1],
        #                        interpolate_to_grid_vpa(expected.vpa, interpolate_to_grid_z(expected.z, f_ion[:, :, :, 2], z, z_spectral), vpa, vpa_spectral)[:,:,1];
        #                        dims=3)
        #println("f_ion ", size(newgrid_f_ion))
        #println(newgrid_f_ion)
        #println()
        #newgrid_f_neutral = cat(interpolate_to_grid_vpa(expected.vpa, interpolate_to_grid_z(expected.z, f_neutral[:, :, :, 1], z, z_spectral), vpa, vpa_spectral)[:,:,1],
        #                        interpolate_to_grid_vpa(expected.vpa, interpolate_to_grid_z(expected.z, f_neutral[:, :, :, 2], z, z_spectral), vpa, vpa_spectral)[:,:,1];
        #                        dims=3)
        #println("f_neutral ", size(newgrid_f_neutral))
        #println(newgrid_f_neutral)
        #println()
        function test_values(tind)
            @testset "tind=$tind" begin
                newgrid_phi = interpolate_to_grid_z(expected.z, phi[:, tind], z, z_spectral)
                @test isapprox(expected.phi[:, tind], newgrid_phi, rtol=rtol)

                # Check ion particle moments and f
                ######################################

                newgrid_n_ion = interpolate_to_grid_z(expected.z, n_ion[:, :, tind], z, z_spectral)
                @test isapprox(expected.n_ion[:, tind], newgrid_n_ion[:,1], rtol=rtol)

                newgrid_upar_ion = interpolate_to_grid_z(expected.z, upar_ion[:, :, tind], z, z_spectral)
                @test isapprox(expected.upar_ion[:, tind], newgrid_upar_ion[:,1], rtol=rtol, atol=atol)

                newgrid_p_ion = interpolate_to_grid_z(expected.z, p_ion[:, :, tind], z, z_spectral)
                @test isapprox(expected.p_ion[:, tind], newgrid_p_ion[:,1], rtol=rtol)

                newgrid_vth_ion = @. sqrt(2.0*newgrid_p_ion/newgrid_n_ion)
                newgrid_f_ion = interpolate_to_grid_z(expected.z, f_ion[:, :, :, tind], z, z_spectral)
                temp = newgrid_f_ion
                newgrid_f_ion = fill(NaN, length(expected.vpa),
                                         size(newgrid_f_ion, 2),
                                         size(newgrid_f_ion, 3),
                                         size(newgrid_f_ion, 4))
                for iz ∈ 1:length(expected.z)
                    wpa = copy(expected.vpa)
                    if input["evolve_moments"]["parallel_flow"]
                        wpa .-= newgrid_upar_ion[iz,1]
                    end
                    if input["evolve_moments"]["pressure"]
                        wpa ./= newgrid_vth_ion[iz,1]
                    end
                    newgrid_f_ion[:,iz,1] = interpolate_to_grid_vpa(wpa, temp[:,iz,1], vpa, vpa_spectral)
                end
                @test isapprox(expected.f_ion[:, :, tind], newgrid_f_ion[:,:,1], rtol=rtol)

                # Check neutral particle moments and f
                ######################################

                newgrid_n_neutral = interpolate_to_grid_z(expected.z, n_neutral[:, :, tind], z, z_spectral)
                @test isapprox(expected.n_neutral[:, tind], newgrid_n_neutral[:,:,1], rtol=rtol)

                newgrid_upar_neutral = interpolate_to_grid_z(expected.z, upar_neutral[:, :, tind], z, z_spectral)
                @test isapprox(expected.upar_neutral[:, tind], newgrid_upar_neutral[:,:,1], rtol=rtol, atol=atol)

                newgrid_p_neutral = interpolate_to_grid_z(expected.z, p_neutral[:, :, tind], z, z_spectral)
                @test isapprox(expected.p_neutral[:, tind], newgrid_p_neutral[:,:,1], rtol=rtol)

                newgrid_vth_neutral = @. sqrt(2.0*newgrid_p_neutral/newgrid_n_neutral)
                newgrid_f_neutral = interpolate_to_grid_z(expected.z, f_neutral[:, :, :, tind], z, z_spectral)
                temp = newgrid_f_neutral
                newgrid_f_neutral = fill(NaN, length(expected.vpa),
                                         size(newgrid_f_neutral, 2),
                                         size(newgrid_f_neutral, 3),
                                         size(newgrid_f_neutral, 4))
                for iz ∈ 1:length(expected.z)
                    wpa = copy(expected.vpa)
                    if input["evolve_moments"]["parallel_flow"]
                        wpa .-= newgrid_upar_neutral[iz,1]
                    end
                    if input["evolve_moments"]["pressure"]
                        wpa ./= newgrid_vth_neutral[iz,1]
                    end
                    newgrid_f_neutral[:,iz,1] = interpolate_to_grid_vpa(wpa, temp[:,iz,1], vpa, vpa_spectral)
                end
                @test isapprox(expected.f_neutral[:, :, tind], newgrid_f_neutral[:,:,1], rtol=rtol)
            end
        end

        # Test initial values
        test_values(1)

        # Test final values
        test_values(2)
    end
end


function runtests()
    # Create a temporary directory for test output
    test_output_directory = get_MPI_tempdir()

    @testset "Krook collisions" verbose=use_verbose begin
        println("Krook collisions tests")

        # Benchmark data is taken from this run (full-f with no splitting)
        @testset "full-f" begin
            test_input_full_f["output"]["base_directory"] = test_output_directory
            run_test(test_input_full_f, 1.e-10, 3.e-16)
        end
        @testset "split 1" begin
            test_input_split_1_moment["output"]["base_directory"] = test_output_directory
            run_test(test_input_split_1_moment, 1.e-3, 1.e-15)
        end
        @testset "split 2" begin
            test_input_split_2_moments["output"]["base_directory"] = test_output_directory
            run_test(test_input_split_2_moments, 1.e-3, 1.e-15)
        end
        @testset "split 3" begin
            test_input_split_3_moments["output"]["base_directory"] = test_output_directory
            run_test(test_input_split_3_moments, 1.e-3, 1.e-15)
        end
    end

    if global_rank[] == 0
        # Delete output directory to avoid using too much disk space
        rm(realpath(test_output_directory); recursive=true)
    end
end

end # KrookCollisionsTests


using .KrookCollisionsTests

KrookCollisionsTests.runtests()
