module KrookCollisionsTests

# Test for Krook collision operator, based on NonlinearSoundWave test

include("setup.jl")

using Base.Filesystem: tempname

using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.input_structs: grid_input, advection_input
using moment_kinetics.load_data: open_readonly_output_file, load_coordinate_data,
                                 load_species_data, load_fields_data,
                                 load_charged_particle_moments_data, load_pdf_data,
                                 load_neutral_particle_moments_data,
                                 load_neutral_pdf_data, load_time_data, load_species_data
using moment_kinetics.interpolation: interpolate_to_grid_z, interpolate_to_grid_vpa
using moment_kinetics.type_definitions: mk_float

# Useful parameters
const z_L = 1.0 # always 1 in normalized units?
const vpa_L = 8.0

# Use very small number of points in vpa_expected to reduce the amount of entries we
# need to store. First and last entries are within the grid (rather than at the ends) in
# order to get non-zero values.
# Note: in the arrays of numbers for expected data, space-separated entries have to stay
# on the same line.
const expected =
    (
     z=[z for z in range(-0.5 * z_L, 0.5 * z_L, length=11)],
     vpa=[vpa for vpa in range(-0.2 * vpa_L, 0.2 * vpa_L, length=3)],
     phi=[-1.386282080324426 -1.2382381134436695; -1.2115129555832849 -1.1306145497660034;
          -0.8609860698164498 -0.8726509017405432; -0.5494724768120176 -0.5904511161957423;
          -0.35345976364887166 -0.37557956583926283; -0.28768207245186167 -0.2919214243915014;
          -0.353459763648872 -0.3755795658392631; -0.5494724768120175 -0.5904511161957432;
          -0.8609860698164502 -0.8726509017405427; -1.2115129555832849 -1.1306145497660032;
          -1.3862820803244258 -1.2382381134436695],
     n_charged=[0.2500030702177186 0.28989452952580286; 0.2977473631375158 0.32283464906590775;
                0.42274585818529853 0.41784232850006636; 0.5772542465450629 0.5540775204094593;
                0.7022542481909738 0.6868914177534788; 0.7499999999999394 0.7468272160708606;
                0.7022542481909738 0.6868914177534787; 0.577254246545063 0.554077520409459;
                0.42274585818529864 0.4178423285000665; 0.2977473631375159 0.3228346490659078;
                0.2500030702177185 0.2898945295258028],
     n_neutral=[0.7499999999999382 0.7736770941648626; 0.7022542481909748 0.7056867075427516;
                0.5772542465450632 0.5582975660019874; 0.4227458581852985 0.4096913953484598;
                0.29774736313751604 0.3053964124252619; 0.2500030702177186 0.2681998023548167;
                0.29774736313751604 0.3053964124252619; 0.42274585818529836 0.4096913953484599;
                0.5772542465450631 0.5582975660019875; 0.7022542481909745 0.7056867075427524;
                0.7499999999999383 0.7736770941648626],
     upar_charged=[-2.7135787559953277e-17 -1.6845791254993525e-16; -9.321028970172899e-18 -0.18245939812953485;
                   -2.8374879811351724e-18 -0.19666454846377826; 1.2124327390522635e-17 -0.11128043369942339;
                   3.6525788403693063e-17 -0.03317985705380149; -2.0930856430671915e-17 4.720175801869314e-17;
                   8.753545920086251e-18 0.033179857053801595; 1.1293771270243255e-17 0.11128043369942343;
                   1.3739171132886587e-17 0.19666454846377784; -6.840453743089351e-18 0.18245939812953468;
                   -2.7135787559953277e-17 -1.9129596434811267e-16],
     upar_neutral=[6.5569385065066925e-18 8.08747058038406e-18; 1.1054500872839027e-17 -0.03620988455458174;
                   -3.241833393685864e-17 -0.009156078199383568; -3.617637280460899e-17 0.05452623197292568;
                   4.417578961284041e-17 0.07607875911384775; 4.9354467746194965e-17 1.635044638743921e-16;
                   6.573091229872379e-18 -0.0760787591138477; 2.989662686945165e-17 -0.05452623197292564;
                   -3.1951996361666834e-17 0.009156078199383685; -4.395464518158184e-18 0.03620988455458165;
                   6.5569385065066925e-18 1.8232586069007834e-18],
     ppar_charged=[0.18749999999999992 0.23302732230115558; 0.20909325514551116 0.21936799130257528;
                   0.24403180771238264 0.20856296024163393; 0.24403180771238278 0.2154266357557397;
                   0.2090932551455113 0.2206183912107678; 0.1875 0.21979739387340663;
                   0.20909325514551128 0.22061839121076784; 0.2440318077123828 0.21542663575573945;
                   0.24403180771238256 0.20856296024163395; 0.20909325514551116 0.2193679913025754;
                   0.18749999999999992 0.23302732230115553],
     ppar_neutral=[0.18750000000000003 0.2480292382671593; 0.20909325514551122 0.24401255100297964;
                   0.24403180771238286 0.22861763406831279; 0.24403180771238278 0.2058922545451891;
                   0.20909325514551144 0.1926313699453636; 0.18749999999999992 0.19090651730415983;
                   0.20909325514551141 0.19263136994536365; 0.2440318077123828 0.20589225454518903;
                   0.24403180771238286 0.2286176340683127; 0.20909325514551114 0.24401255100297964;
                   0.18750000000000006 0.24802923826715936],
     f_charged=[0.0370462360994826 0.04059927063892091 0.0428431419871786 0.030398267195914062 0.01236045902698859 0.006338529470383425 0.012360459026988587 0.030398267195914028 0.04284314198717859 0.0405992706389209 0.0370462360994826;
                0.20411991941198782 0.25123395823993105 0.3934413727192304 0.6277900619432855 0.9100364506661008 1.0606601717796504 0.910036450666101 0.6277900619432859 0.39344137271923046 0.25123395823993094 0.20411991941198776;
                0.0370462360994826 0.04059927063892091 0.0428431419871786 0.030398267195914062 0.01236045902698859 0.006338529470383425 0.012360459026988587 0.030398267195914028 0.04284314198717859 0.0405992706389209 0.0370462360994826;;;
                0.0538996852594264 0.06066864433237418 0.03746866696438989 0.014783440166032301 0.010917691665145668 0.018422971878502774 0.027170953411068444 0.027269146560166702 0.026567569739750264 0.035612674100528624 0.05389968525942639;
                0.2118369019176154 0.24917436308523389 0.37345448114678914 0.5972219245577428 0.8859681860177208 1.0485988935814787 0.8859681860177204 0.5972219245577435 0.37345448114678825 0.24917436308523389 0.2118369019176155;
                0.05389968525942635 0.03561267410052869 0.02656756973975021 0.02726914656016675 0.027170953411068514 0.018422971878502753 0.01091769166514568 0.014783440166032254 0.037468666964389795 0.060668644332374164 0.05389968525942635],
     f_neutral=[0.0063385294703834595 0.012360459026988546 0.030398267195914108 0.04284314198717859 0.040599270638920985 0.03704623609948259 0.040599270638920965 0.0428431419871786 0.030398267195914094 0.012360459026988546 0.006338529470383456;
                1.0606601717796493 0.9100364506661016 0.6277900619432857 0.3934413727192303 0.2512339582399308 0.20411991941198754 0.2512339582399307 0.3934413727192301 0.6277900619432853 0.9100364506661016 1.0606601717796487;
                0.0063385294703834595 0.012360459026988546 0.030398267195914108 0.04284314198717859 0.040599270638920985 0.03704623609948259 0.040599270638920965 0.0428431419871786 0.030398267195914094 0.012360459026988546 0.006338529470383456;;;
                0.0242848886629411 0.04071460358290305 0.04191389118981371 0.03638215764882266 0.03692283098105331 0.04164449216999481 0.03671950776850948 0.01928119243573099 0.008423252360063483 0.010011392733734206 0.02428488866294109;
                1.0530033604430462 0.9036809869030653 0.6251085339983469 0.3955308968816375 0.25710352416286547 0.21137159186144025 0.25710352416286547 0.3955308968816377 0.6251085339983473 0.9036809869030653 1.0530033604430464;
                0.024284888662941113 0.010011392733734206 0.008423252360063494 0.019281192435730943 0.036719507768509525 0.041644492169994836 0.03692283098105331 0.03638215764882269 0.04191389118981368 0.04071460358290303 0.024284888662941134])

# default inputs for tests
test_input_full_f = Dict("n_ion_species" => 1,
                         "n_neutral_species" => 1,
                         "boltzmann_electron_response" => true,
                         "run_name" => "full_f",
                         "evolve_moments_density" => false,
                         "evolve_moments_parallel_flow" => false,
                         "evolve_moments_parallel_pressure" => false,
                         "evolve_moments_conservation" => true,
                         "krook_collisions" => Dict{String,Any}("use_krook" => true,"frequency_option" => "reference_parameters"),
                         "T_e" => 1.0,
                         "initial_density1" => 0.5,
                         "initial_temperature1" => 1.0,
                         "initial_density2" => 0.5,
                         "initial_temperature2" => 1.0,
                         "z_IC_option1" => "sinusoid",
                         "z_IC_density_amplitude1" => 0.5,
                         "z_IC_density_phase1" => 0.0,
                         "z_IC_upar_amplitude1" => 0.0,
                         "z_IC_upar_phase1" => 0.0,
                         "z_IC_temperature_amplitude1" => 0.5,
                         "z_IC_temperature_phase1" => mk_float(π),
                         "z_IC_option2" => "sinusoid",
                         "z_IC_density_amplitude2" => 0.5,
                         "z_IC_density_phase2" => mk_float(π),
                         "z_IC_upar_amplitude2" => 0.0,
                         "z_IC_upar_phase2" => 0.0,
                         "z_IC_temperature_amplitude2" => 0.5,
                         "z_IC_temperature_phase2" => 0.0,
                         "charge_exchange_frequency" => 2*π*0.1,
                         "ionization_frequency" => 0.0,
                         "nstep" => 100,
                         "dt" => 0.001,
                         "nwrite" => 100,
                         "nwrite_dfns" => 100,
                         "use_semi_lagrange" => false,
                         "n_rk_stages" => 4,
                         "split_operators" => false,
                         "r_ngrid" => 1,
                         "r_nelement" => 1,
                         "r_bc" => "periodic",
                         "r_discretization" => "chebyshev_pseudospectral",
                         "z_ngrid" => 9,
                         "z_nelement" => 4,
                         "z_bc" => "periodic",
                         "z_discretization" => "chebyshev_pseudospectral",
                         "vpa_ngrid" => 17,
                         "vpa_nelement" => 8,
                         "vpa_L" => vpa_L,
                         "vpa_bc" => "periodic",
                         "vpa_discretization" => "chebyshev_pseudospectral",
                         "vz_ngrid" => 17,
                         "vz_nelement" => 8,
                         "vz_L" => vpa_L,
                         "vz_bc" => "periodic",
                         "vz_discretization" => "chebyshev_pseudospectral")

test_input_split_1_moment =
    merge(test_input_full_f,
          Dict("run_name" => "split_1_moment",
               "evolve_moments_density" => true))

test_input_split_2_moments =
    merge(test_input_split_1_moment,
          Dict("run_name" => "split_2_moments",
               "evolve_moments_parallel_flow" => true))

test_input_split_3_moments =
    merge(test_input_split_2_moments,
          Dict("run_name" => "split_3_moments",
               "evolve_moments_parallel_pressure" => true,
               "vpa_L" => 12.0, "vz_L" => 12.0))


"""
Run a sound-wave test for a single set of parameters
"""
# Note 'name' should not be shared by any two tests in this file
function run_test(test_input, rtol, atol; args...)
    # by passing keyword arguments to run_test, args becomes a Dict which can be used to
    # update the default inputs

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
    quietoutput() do
        # run simulation
        run_moment_kinetics(input)
    end

    phi = nothing
    n_charged = nothing
    upar_charged = nothing
    ppar_charged = nothing
    f_charged = nothing
    n_neutral = nothing
    upar_neutral = nothing
    ppar_neutral = nothing
    f_neutral = nothing
    z, z_spectral = nothing, nothing
    vpa, vpa_spectral = nothing, nothing

    if global_rank[] == 0
        quietoutput() do

            # Load and analyse output
            #########################

            path = joinpath(realpath(input["base_directory"]), name, name)

            # open the netcdf file containing moments data and give it the handle 'fid'
            fid = open_readonly_output_file(path, "moments")

            # load species, time coordinate data
            n_ion_species, n_neutral_species = load_species_data(fid)
            ntime, time = load_time_data(fid)
            n_ion_species, n_neutral_species = load_species_data(fid)
            
            # load fields data
            phi_zrt, Er_zrt, Ez_zrt = load_fields_data(fid)

            # load velocity moments data
            n_charged_zrst, upar_charged_zrst, ppar_charged_zrst, qpar_charged_zrst, v_t_charged_zrst = load_charged_particle_moments_data(fid)
            n_neutral_zrst, upar_neutral_zrst, ppar_neutral_zrst, qpar_neutral_zrst, v_t_neutral_zrst = load_neutral_particle_moments_data(fid)
            z, z_spectral = load_coordinate_data(fid, "z")

            close(fid)
            
            # open the netcdf file containing pdf data
            fid = open_readonly_output_file(path, "dfns")
            
            # load particle distribution function (pdf) data
            f_charged_vpavperpzrst = load_pdf_data(fid)
            f_neutral_vzvrvzetazrst = load_neutral_pdf_data(fid)
            vpa, vpa_spectral = load_coordinate_data(fid, "vpa")

            close(fid)
            
            phi = phi_zrt[:,1,:]
            n_charged = n_charged_zrst[:,1,:,:]
            upar_charged = upar_charged_zrst[:,1,:,:]
            ppar_charged = ppar_charged_zrst[:,1,:,:]
            qpar_charged = qpar_charged_zrst[:,1,:,:]
            v_t_charged = v_t_charged_zrst[:,1,:,:]
            f_charged = f_charged_vpavperpzrst[:,1,:,1,:,:]
            n_neutral = n_neutral_zrst[:,1,:,:]
            upar_neutral = upar_neutral_zrst[:,1,:,:]
            ppar_neutral = ppar_neutral_zrst[:,1,:,:]
            qpar_neutral = qpar_neutral_zrst[:,1,:,:]
            v_t_neutral = v_t_neutral_zrst[:,1,:,:]
            f_neutral = f_neutral_vzvrvzetazrst[:,1,1,:,1,:,:]

            # Unnormalize f
            if input["evolve_moments_density"]
                for it ∈ 1:length(time), is ∈ 1:n_ion_species, iz ∈ 1:z.n
                    f_charged[:,iz,is,it] .*= n_charged[iz,is,it]
                end
                for it ∈ 1:length(time), isn ∈ 1:n_neutral_species, iz ∈ 1:z.n
                    f_neutral[:,iz,isn,it] .*= n_neutral[iz,isn,it]
                end
            end
            if input["evolve_moments_parallel_pressure"]
                for it ∈ 1:length(time), is ∈ 1:n_ion_species, iz ∈ 1:z.n
                    f_charged[:,iz,is,it] ./= v_t_charged[iz,is,it]
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
        #newgrid_n_charged = cat(interpolate_to_grid_z(expected.z, n_charged[:, :, 1], z, z_spectral)[:,1],
        #                        interpolate_to_grid_z(expected.z, n_charged[:, :, 2], z, z_spectral)[:,1];
        #                        dims=2)
        #println("n_charged ", size(newgrid_n_charged))
        #println(newgrid_n_charged)
        #println()
        #newgrid_n_neutral = cat(interpolate_to_grid_z(expected.z, n_neutral[:, :, 1], z, z_spectral)[:,1],
        #                        interpolate_to_grid_z(expected.z, n_neutral[:, :, 2], z, z_spectral)[:,1];
        #                        dims=2)
        #println("n_neutral ", size(newgrid_n_neutral))
        #println(newgrid_n_neutral)
        #println()
        #newgrid_upar_charged = cat(interpolate_to_grid_z(expected.z, upar_charged[:, :, 1], z, z_spectral)[:,1],
        #                           interpolate_to_grid_z(expected.z, upar_charged[:, :, 2], z, z_spectral)[:,1];
        #                           dims=2)
        #println("upar_charged ", size(newgrid_upar_charged))
        #println(newgrid_upar_charged)
        #println()
        #newgrid_upar_neutral = cat(interpolate_to_grid_z(expected.z, upar_neutral[:, :, 1], z, z_spectral)[:,1],
        #                           interpolate_to_grid_z(expected.z, upar_neutral[:, :, 2], z, z_spectral)[:,1];
        #                           dims=2)
        #println("upar_neutral ", size(newgrid_upar_neutral))
        #println(newgrid_upar_neutral)
        #println()
        #newgrid_ppar_charged = cat(interpolate_to_grid_z(expected.z, ppar_charged[:, :, 1], z, z_spectral)[:,1],
        #                           interpolate_to_grid_z(expected.z, ppar_charged[:, :, 2], z, z_spectral)[:,1];
        #                           dims=2)
        #println("ppar_charged ", size(newgrid_ppar_charged))
        #println(newgrid_ppar_charged)
        #println()
        #newgrid_ppar_neutral = cat(interpolate_to_grid_z(expected.z, ppar_neutral[:, :, 1], z, z_spectral)[:,1],
        #                           interpolate_to_grid_z(expected.z, ppar_neutral[:, :, 2], z, z_spectral)[:,1];
        #                           dims=2)
        #println("ppar_neutral ", size(newgrid_ppar_neutral))
        #println(newgrid_ppar_neutral)
        #println()
        #newgrid_f_charged = cat(interpolate_to_grid_vpa(expected.vpa, interpolate_to_grid_z(expected.z, f_charged[:, :, :, 1], z, z_spectral), vpa, vpa_spectral)[:,:,1],
        #                        interpolate_to_grid_vpa(expected.vpa, interpolate_to_grid_z(expected.z, f_charged[:, :, :, 2], z, z_spectral), vpa, vpa_spectral)[:,:,1];
        #                        dims=4)
        #println("f_charged ", size(newgrid_f_charged))
        #println(newgrid_f_charged)
        #println()
        #newgrid_f_neutral = cat(interpolate_to_grid_vpa(expected.vpa, interpolate_to_grid_z(expected.z, f_neutral[:, :, :, 1], z, z_spectral), vpa, vpa_spectral)[:,:,1],
        #                        interpolate_to_grid_vpa(expected.vpa, interpolate_to_grid_z(expected.z, f_neutral[:, :, :, 2], z, z_spectral), vpa, vpa_spectral)[:,:,1];
        #                        dims=4)
        #println("f_neutral ", size(newgrid_f_neutral))
        #println(newgrid_f_neutral)
        #println()
        function test_values(tind)
            @testset "tind=$tind" begin
                newgrid_phi = interpolate_to_grid_z(expected.z, phi[:, tind], z, z_spectral)
                @test isapprox(expected.phi[:, tind], newgrid_phi, rtol=rtol)

                # Check charged particle moments and f
                ######################################

                newgrid_n_charged = interpolate_to_grid_z(expected.z, n_charged[:, :, tind], z, z_spectral)
                @test isapprox(expected.n_charged[:, tind], newgrid_n_charged[:,1], rtol=rtol)

                newgrid_upar_charged = interpolate_to_grid_z(expected.z, upar_charged[:, :, tind], z, z_spectral)
                @test isapprox(expected.upar_charged[:, tind], newgrid_upar_charged[:,1], rtol=rtol, atol=atol)

                newgrid_ppar_charged = interpolate_to_grid_z(expected.z, ppar_charged[:, :, tind], z, z_spectral)
                @test isapprox(expected.ppar_charged[:, tind], newgrid_ppar_charged[:,1], rtol=rtol)

                newgrid_vth_charged = @. sqrt(2.0*newgrid_ppar_charged/newgrid_n_charged)
                newgrid_f_charged = interpolate_to_grid_z(expected.z, f_charged[:, :, :, tind], z, z_spectral)
                temp = newgrid_f_charged
                newgrid_f_charged = fill(NaN, length(expected.vpa),
                                         size(newgrid_f_charged, 2),
                                         size(newgrid_f_charged, 3),
                                         size(newgrid_f_charged, 4))
                for iz ∈ 1:length(expected.z)
                    wpa = copy(expected.vpa)
                    if input["evolve_moments_parallel_flow"]
                        wpa .-= newgrid_upar_charged[iz,1]
                    end
                    if input["evolve_moments_parallel_pressure"]
                        wpa ./= newgrid_vth_charged[iz,1]
                    end
                    newgrid_f_charged[:,iz,1] = interpolate_to_grid_vpa(wpa, temp[:,iz,1], vpa, vpa_spectral)
                end
                @test isapprox(expected.f_charged[:, :, tind], newgrid_f_charged[:,:,1], rtol=rtol)

                # Check neutral particle moments and f
                ######################################

                newgrid_n_neutral = interpolate_to_grid_z(expected.z, n_neutral[:, :, tind], z, z_spectral)
                @test isapprox(expected.n_neutral[:, tind], newgrid_n_neutral[:,:,1], rtol=rtol)

                newgrid_upar_neutral = interpolate_to_grid_z(expected.z, upar_neutral[:, :, tind], z, z_spectral)
                @test isapprox(expected.upar_neutral[:, tind], newgrid_upar_neutral[:,:,1], rtol=rtol, atol=atol)

                newgrid_ppar_neutral = interpolate_to_grid_z(expected.z, ppar_neutral[:, :, tind], z, z_spectral)
                @test isapprox(expected.ppar_neutral[:, tind], newgrid_ppar_neutral[:,:,1], rtol=rtol)

                newgrid_vth_neutral = @. sqrt(2.0*newgrid_ppar_neutral/newgrid_n_neutral)
                newgrid_f_neutral = interpolate_to_grid_z(expected.z, f_neutral[:, :, :, tind], z, z_spectral)
                temp = newgrid_f_neutral
                newgrid_f_neutral = fill(NaN, length(expected.vpa),
                                         size(newgrid_f_neutral, 2),
                                         size(newgrid_f_neutral, 3),
                                         size(newgrid_f_neutral, 4))
                for iz ∈ 1:length(expected.z)
                    wpa = copy(expected.vpa)
                    if input["evolve_moments_parallel_flow"]
                        wpa .-= newgrid_upar_neutral[iz,1]
                    end
                    if input["evolve_moments_parallel_pressure"]
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
            test_input_full_f["base_directory"] = test_output_directory
            run_test(test_input_full_f, 1.e-10, 3.e-16)
        end
        @testset "split 1" begin
            test_input_split_1_moment["base_directory"] = test_output_directory
            run_test(test_input_split_1_moment, 1.e-3, 1.e-15)
        end
        @testset "split 2" begin
            test_input_split_2_moments["base_directory"] = test_output_directory
            run_test(test_input_split_2_moments, 1.e-3, 1.e-15)
        end
        @testset "split 3" begin
            test_input_split_3_moments["base_directory"] = test_output_directory
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
