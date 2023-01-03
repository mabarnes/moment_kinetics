module NonlinearSoundWaveTests

include("setup.jl")

using Base.Filesystem: tempname
using TimerOutputs

using moment_kinetics.chebyshev: setup_chebyshev_pseudospectral
using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.input_structs: grid_input, advection_input
using moment_kinetics.load_data: open_readonly_output_file, load_coordinate_data, load_species_data,
                                 load_fields_data, load_charged_particle_moments_data, load_pdf_data,
                                 load_neutral_particle_moments_data, load_neutral_pdf_data, load_time_data,
                                 load_species_data
using moment_kinetics.interpolation: interpolate_to_grid_z, interpolate_to_grid_vpa
using moment_kinetics.type_definitions: mk_float

const analytical_rtol = 3.e-2
const regression_rtol = 2.e-8

# Create a temporary directory for test output
test_output_directory = tempname()
mkpath(test_output_directory)

# Useful parameters
const z_L = 1.0 # always 1 in normalized units?
const vpa_L = 8.0

# The expected output
struct expected_data
    z::Array{mk_float, 1}
    vpa::Array{mk_float, 1}
    phi::Array{mk_float, 2}
    n_charged::Array{mk_float, 2}
    n_neutral::Array{mk_float, 2}
    upar_charged::Array{mk_float, 2}
    upar_neutral::Array{mk_float, 2}
    ppar_charged::Array{mk_float, 2}
    ppar_neutral::Array{mk_float, 2}
    f_charged::Array{mk_float, 3}
    f_neutral::Array{mk_float, 3}
end

# Use very small number of points in vpa_expected to reduce the amount of entries we
# need to store. First and last entries are within the grid (rather than at the ends) in
# order to get non-zero values.
# Note: in the arrays of numbers for expected data, space-separated entries have to stay
# on the same line.
const expected =
  expected_data(
   [z for z in range(-0.5 * z_L, 0.5 * z_L, length=11)],
   [vpa for vpa in range(-0.2 * vpa_L, 0.2 * vpa_L, length=3)],
   # Expected phi:
   [-1.3862943611198908 -1.2383045470759624; -1.2115160487200705 -1.1306858544584915;
    -0.8609385840190196 -0.8726873717254309; -0.5495323843850888 -0.5902920285798641;
    -0.3534143971698439 -0.3756206843360002; -0.2876820724517811 -0.2919957809820923;
    -0.3534143971698439 -0.3756206843360003; -0.5495323843850888 -0.5902920285798642;
    -0.8609385840190196 -0.8726873717254305; -1.2115160487200705 -1.1306858544584917;
    -1.3862943611198908 -1.2383045470759624],
   # Expected n_charged:
   [0.25 0.2898752714189248; 0.2977455383009383 0.3227988955943189;
    0.42274604116408965 0.4178422059225094; 0.5772539588359104 0.5541536347014877;
    0.7022544616990617 0.6868683064113692; 0.75 0.7467716866098297;
    0.7022544616990617 0.6868683064113691; 0.5772539588359104 0.5541536347014876;
    0.42274604116408965 0.41784220592250954; 0.2977455383009383 0.3227988955943189;
    0.25 0.28987527141892466],
   # Expected n_neutral:
   [0.75 0.7737996678005374; 0.7022544616990617 0.7056147526068444;
    0.5772539588359104 0.558319990049417; 0.42274604116408965 0.40968196865824635;
    0.29774553830093836 0.3053756664897497; 0.25 0.26816543361490336;
    0.29774553830093836 0.3053756664897497; 0.4227460411640896 0.40968196865824663;
    0.5772539588359104 0.5583199900494169; 0.7022544616990617 0.7056147526068444;
    0.7499999999999999 0.7737996678005375],
   # Expected upar_charged:
   [0.0 -7.19910242530375e-17; 0.0 -0.18232118939493933; 0.0 -0.19672400115275973;
    0.0 -0.11125439582747795; 0.0 -0.03374761848109975; 0.0 -3.0357660829594124e-17;
    0.0 0.03374761848109988; 0.0 0.11125439582747787; 0.0 0.19672400115275954;
    0.0 0.1823211893949391; 0.0 -7.979727989493313e-17],
   # Expected upar_neutral:
   [0.0 2.3527187142935446e-17; 0.0 -0.0361818709266271; 0.0 -0.00922636838029161;
    0.0 0.05457230487205319; 0.0 0.07610770982473092; 0.0 -1.1167282376600696e-17;
    0.0 -0.07610770982473074; 0.0 -0.054572304872053286; 0.0 0.009226368380291661;
    0.0 0.03618187092662704; 0.0 3.155028321932818e-17],
   # Expected ppar_charged:
   [0.1875 0.2327894016408319; 0.20909100943488423 0.22988088802604845;
    0.24403280042122125 0.22432000010325867; 0.2440328004212213 0.2220384013448133;
    0.20909100943488423 0.22159376338740927; 0.18750000000000006 0.2213757069926422;
    0.2090910094348842 0.2215937633874092; 0.2440328004212212 0.22203840134481326;
    0.2440328004212212 0.22432000010325867; 0.2090910094348842 0.2298808880260482;
    0.18749999999999994 0.23278940164083192],
   # Expected ppar_neutral:
   [0.1875000000000001 0.24825663604647846; 0.20909100943488423 0.24474944179969235;
    0.2440328004212213 0.2287762657201228; 0.2440328004212213 0.20702913756159222;
    0.2090910094348841 0.19445093198867122; 0.18749999999999994 0.19084845746775503;
    0.20909100943488398 0.19445093198867128; 0.2440328004212213 0.20702913756159214;
    0.2440328004212212 0.22877626572012272; 0.20909100943488418 0.24474944179969224;
    0.18750000000000006 0.24825663604647852],
   # Expected f_charged:
   [0.0370462360994826 0.04056128509273146 0.04289169811317834 0.030368915327672278 0.012353622350339346 0.006338529470383422 0.01235362235033933 0.03036891532767225 0.0428916981131783 0.04056128509273143 0.03704623609948261;
    0.20411991941198765 0.2511561329105551 0.39355562262094196 0.6276758497903184 0.910082733302134 1.0606601717796504 0.9100827333021344 0.627675849790319 0.393555622620942 0.2511561329105552 0.20411991941198787;
    0.0370462360994826 0.04056128509273146 0.04289169811317834 0.030368915327672278 0.012353622350339346 0.006338529470383422 0.01235362235033933 0.03036891532767225 0.0428916981131783 0.04056128509273143 0.03704623609948261;;;
    0.053941018353262433 0.0605545923385808 0.03683391041516729 0.013633122226762785 0.010808508800713386 0.019345197462736118 0.027958745982156707 0.027628128337785936 0.02665929672138348 0.03561333548228102 0.05394101835326243;
    0.2117759342183267 0.24890460364947747 0.37312606860180475 0.596074140742547 0.8872166613189386 1.0533874362221456 0.8872166613189383 0.596074140742547 0.3731260686018043 0.24890460364947742 0.2117759342183268;
    0.05394101835326246 0.03561333548228096 0.026659296721383476 0.02762812833778592 0.02795874598215675 0.019345197462736125 0.010808508800713384 0.013633122226762792 0.0368339104151672 0.060554592338580765 0.053941018353262475],
   # Expected f_neutral:
   [0.006338529470383451 0.01235362235033934 0.030368915327672327 0.042891698113178334 0.04056128509273143 0.03704623609948261 0.04056128509273146 0.04289169811317831 0.03036891532767233 0.012353622350339327 0.006338529470383458;
    1.0606601717796487 0.9100827333021346 0.6276758497903172 0.39355562262094196 0.25115613291055505 0.2041199194119877 0.2511561329105551 0.393555622620942 0.6276758497903171 0.9100827333021345 1.060660171779649;
    0.006338529470383451 0.01235362235033934 0.030368915327672327 0.042891698113178334 0.04056128509273143 0.03704623609948261 0.04056128509273146 0.04289169811317831 0.03036891532767233 0.012353622350339327 0.006338529470383458;;;
    0.024302660398203713 0.04068112767994812 0.04194789083678011 0.036333231314850035 0.03689201427179055 0.04167458209630273 0.036666413774663884 0.019366777784571958 0.008337639230774864 0.009995913913465715 0.02430266039820372;
    1.0530041569733002 0.9037244247465048 0.6249909697965033 0.3956376124151513 0.25703391714537 0.2113926549694852 0.25703391714537016 0.39563761241515105 0.624990969796503 0.9037244247465047 1.0530041569733009;
    0.024302660398203686 0.00999591391346571 0.008337639230774855 0.019366777784571982 0.03666641377466386 0.0416745820963027 0.036892014271790535 0.03633323131485009 0.04194789083678017 0.04068112767994807 0.02430266039820367])

# default inputs for tests
test_input_finite_difference = Dict("n_ion_species" => 1,
                                    "n_neutral_species" => 1,
                                    "boltzmann_electron_response" => true,
                                    "run_name" => "finite_difference",
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
                                    "z_IC_density_amplitude1" => 0.5,
                                    "z_IC_density_phase1" => 0.0,
                                    "z_IC_upar_amplitude1" => 0.0,
                                    "z_IC_upar_phase1" => 0.0,
                                    "z_IC_temperature_amplitude1" => 0.5,
                                    "z_IC_temperature_phase1" => π,
                                    "z_IC_option2" => "sinusoid",
                                    "z_IC_density_amplitude2" => 0.5,
                                    "z_IC_density_phase2" => π,
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
                                    "r_discretization" => "finite_difference",
                                    "z_ngrid" => 100,
                                    "z_nelement" => 1,
                                    "z_bc" => "periodic",
                                    "z_discretization" => "finite_difference",
                                    "vpa_ngrid" => 400,
                                    "vpa_nelement" => 1,
                                    "vpa_L" => vpa_L,
                                    "vpa_bc" => "periodic",
                                    "vpa_discretization" => "finite_difference")

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

test_input_chebyshev = merge(test_input_finite_difference,
                             Dict("run_name" => "chebyshev_pseudospectral",
                                  "z_discretization" => "chebyshev_pseudospectral",
                                  "z_ngrid" => 9,
                                  "z_nelement" => 2,
                                  "vpa_discretization" => "chebyshev_pseudospectral",
                                  "vpa_ngrid" => 17,
                                  "vpa_nelement" => 8))

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


# Not actually used in the tests, but needed for first argument of run_moment_kinetics
to = TimerOutput()

"""
Run a sound-wave test for a single set of parameters
"""
# Note 'name' should not be shared by any two tests in this file
function run_test(test_input, rtol; args...)
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
    phi = undef
    n_charged = undef
    upar_charged = undef
    ppar_charged = undef
    f_charged = undef
    n_neutral = undef
    upar_neutral = undef
    ppar_neutral = undef
    f_neutral = undef
    quietoutput() do
        # run simulation
        run_moment_kinetics(to, input)
    end

    if global_rank[] == 0
        quietoutput() do

            # Load and analyse output
            #########################

            path = joinpath(realpath(input["base_directory"]), name, name)

            # open the netcdf file containing moments data and give it the handle 'fid'
            fid = open_readonly_output_file(path, "moments")

            # load space-time coordinate data
            nz, nz_global, z, z_wgts, Lz = load_coordinate_data(fid, "z")
            nr, nr_global, r, r_wgts, Lr = load_coordinate_data(fid, "r")
            n_ion_species, n_neutral_species = load_species_data(fid)
            ntime, time = load_time_data(fid)
            n_ion_species, n_neutral_species = load_species_data(fid)
            
            # load fields data
            phi_zrt, Er_zrt, Ez_zrt = load_fields_data(fid)

            # load velocity moments data
            n_charged_zrst, upar_charged_zrst, ppar_charged_zrst, qpar_charged_zrst, v_t_charged_zrst, evolve_ppar = load_charged_particle_moments_data(fid)
            n_neutral_zrst, upar_neutral_zrst, ppar_neutral_zrst, qpar_neutral_zrst, v_t_neutral_zrst = load_neutral_particle_moments_data(fid)

            close(fid)
            
            # open the netcdf file containing pdf data
            fid = open_readonly_output_file(path, "dfns")
            
            # load particle distribution function (pdf) data
            f_charged_vpavperpzrst = load_pdf_data(fid)
            f_neutral_vzvrvzetazrst = load_neutral_pdf_data(fid)

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
        end

        # Create coordinates
        #
        # create the 'input' struct containing input info needed to create a coordinate
        # adv_input not actually used in this test so given values unimportant
        adv_input = advection_input("default", 1.0, 0.0, 0.0)
        nrank_per_block = 0 # dummy value
		irank = 0 # dummy value
		comm = false # dummy value 
		input = grid_input("coord", test_input["z_ngrid"], test_input["z_nelement"], 
                           test_input["z_nelement"], nrank_per_block, irank,
						   z_L, test_input["z_discretization"], "",
                           "periodic", #test_input["z_bc"],
                           adv_input,comm)
        z = define_coordinate(input)
        if test_input["z_discretization"] == "chebyshev_pseudospectral"
            z_spectral = setup_chebyshev_pseudospectral(z)
        else
            z_spectral = false
        end
        input = grid_input("coord", test_input["vpa_ngrid"], test_input["vpa_nelement"],
                           test_input["vpa_nelement"], nrank_per_block, irank,
						   vpa_L, test_input["vpa_discretization"], "",
                           test_input["vpa_bc"], adv_input, comm)
        vpa = define_coordinate(input)
        if test_input["vpa_discretization"] == "chebyshev_pseudospectral"
            vpa_spectral = setup_chebyshev_pseudospectral(vpa)
        else
            vpa_spectral = false
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
                @test isapprox(expected.upar_charged[:, tind], newgrid_upar_charged[:,1], rtol=rtol)

                newgrid_ppar_charged = interpolate_to_grid_z(expected.z, ppar_charged[:, :, tind], z, z_spectral)
                @test isapprox(expected.ppar_charged[:, tind], newgrid_ppar_charged[:,1], rtol=rtol)

                newgrid_f_charged = interpolate_to_grid_z(expected.z, f_charged[:, :, :, tind], z, z_spectral)
                newgrid_f_charged = interpolate_to_grid_vpa(expected.vpa, newgrid_f_charged, vpa, vpa_spectral)
                @test isapprox(expected.f_charged[:, :, tind], newgrid_f_charged[:,:,1], rtol=rtol)

                # Check neutral particle moments and f
                ######################################

                newgrid_n_neutral = interpolate_to_grid_z(expected.z, n_neutral[:, :, tind], z, z_spectral)
                @test isapprox(expected.n_neutral[:, tind], newgrid_n_neutral[:,:,1], rtol=rtol)

                newgrid_upar_neutral = interpolate_to_grid_z(expected.z, upar_neutral[:, :, tind], z, z_spectral)
                @test isapprox(expected.upar_neutral[:, tind], newgrid_upar_neutral[:,:,1], rtol=rtol)

                newgrid_ppar_neutral = interpolate_to_grid_z(expected.z, ppar_neutral[:, :, tind], z, z_spectral)
                @test isapprox(expected.ppar_neutral[:, tind], newgrid_ppar_neutral[:,:,1], rtol=rtol)

                newgrid_f_neutral = interpolate_to_grid_z(expected.z, f_neutral[:, :, :, tind], z, z_spectral)
                newgrid_f_neutral = interpolate_to_grid_vpa(expected.vpa, newgrid_f_neutral, vpa, vpa_spectral)
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
    @testset "nonlinear sound wave" verbose=use_verbose begin
        println("nonlinear sound wave tests")

        # finite difference
        @testset "FD base" begin
            run_test(test_input_finite_difference, 1.1e-3)
        end
        #@testset "FD split 1" begin
        #    run_test(test_input_finite_difference_split_1_moment, 1.e-3)
        #end
        #@testset_skip "grids need shift/scale for collisions" "FD split 2" begin
        #    run_test(test_input_finite_difference_split_2_moments, 1.e-3)
        #end
        #@testset_skip "grids need shift/scale for collisions" "FD split 3" begin
        #    run_test(test_input_finite_difference_split_3_moments, 1.e-3)
        #end

        # Chebyshev pseudospectral
        # Benchmark data is taken from this run (Chebyshev with no splitting)
        @testset "Chebyshev base" begin
            run_test(test_input_chebyshev, 1.e-10)
        end
        #@testset "Chebyshev split 1" begin
        #    run_test(test_input_chebyshev_split_1_moment, 1.e-3)
        #end
        #@testset_skip "grids need shift/scale for collisions" "Chebyshev split 2" begin
        #    run_test(test_input_chebyshev_split_2_moments, 1.e-3)
        #end
        #@testset_skip "grids need shift/scale for collisions" "Chebyshev split 3" begin
        #    run_test(test_input_chebyshev_split_3_moments, 1.e-3)
        #end
    end
end

end # NonlinearSoundWaveTests


using .NonlinearSoundWaveTests

NonlinearSoundWaveTests.runtests()
