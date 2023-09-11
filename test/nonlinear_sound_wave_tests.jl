module NonlinearSoundWaveTests

include("setup.jl")

using Base.Filesystem: tempname
using MPI
using TimerOutputs

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
   [-1.3862820803244256 -1.2383045504753758; -1.211510602668698 -1.1306858553168957;
    -0.860938418534854 -0.8726873701297669; -0.5495322983936358 -0.5902920278548919;
    -0.3534144494723056 -0.3756206847277757; -0.2876820724518619 -0.2919957813382994;
    -0.35341444947230544 -0.37562068472777577; -0.5495322983936355 -0.5902920278548919;
    -0.8609384185348539 -0.8726873701297669; -1.2115106026686981 -1.130685855316896;
    -1.3862820803244256 -1.2383045504753758],
   # Expected n_charged:
   [0.2500030702177184 0.2898752704335188; 0.2977471383217195 0.3227988953227183;
    0.42274614626845974 0.417842206578383; 0.5772539714051019 0.5541536351162784;
    0.702254450621661 0.686868306132489; 0.7499999999999392 0.7467716863438243;
    0.702254450621661 0.6868683061324891; 0.577253971405102 0.5541536351162781;
    0.42274614626845963 0.41784220657838306; 0.29774713832171945 0.3227988953227184;
    0.25000307021771856 0.2898752704335188],
   # Expected n_neutral:
   [0.7499999999999392 0.7737996616909211; 0.702254450621661 0.7056147469533546;
    0.5772539714051019 0.5583199869826109; 0.42274614626845974 0.4096819689829928;
    0.29774713832171956 0.30537567265010457; 0.2500030702177185 0.26816544412496246;
    0.2977471383217197 0.30537567265010435; 0.4227461462684595 0.4096819689829924;
    0.5772539714051017 0.5583199869826102; 0.7022544506216611 0.7056147469533546;
    0.7499999999999394 0.7737996616909211],
   # Expected upar_charged:
   [1.1971912119126474e-17 -2.0968470015869656e-16;
    -5.818706134342973e-17 -0.18232119131671534;
    9.895531571141618e-17 -0.1967239995126128;
    -8.38774587436108e-18 -0.11125439467488389;
    -1.7717259792356293e-17 -0.033747618153236424;
    -3.143783114880992e-17 9.84455572616838e-17;
    -2.499642278253839e-17 0.03374761815323648;
    -2.9272523290371316e-17 0.1112543946748839;
    3.346728365577734e-17 0.19672399951261313; -8.193702949354942e-17 0.18232119131671523;
    1.1971912119126474e-17 -2.0187639060277426e-16],
   # Expected upar_neutral:
   [-3.143783114880993e-17 -3.003240017784847e-17;
    -1.7717259792356296e-17 -0.03618184473095593;
    -8.38774587436108e-18 -0.009226347310827927;
    9.895531571141618e-17 0.054572308562824384; -5.818706134342965e-17 0.0761077077764258;
    1.1971912119126477e-17 2.1033522146218786e-16;
    4.747047511913174e-17 -0.07610770777642595;
    6.558391323223502e-18 -0.05457230856282445;
    2.2213638810498713e-17 0.009226347310827925;
    -2.7413075842225616e-17 0.036181844730955835;
    -3.143783114880993e-17 -2.810175715538666e-17],
   # Expected ppar_charged:
   [0.18749999999999997 0.23278940073547755; 0.20909100943488423 0.21912527958959363;
    0.24403280042122125 0.20817795270356831; 0.2440328004212212 0.21516422119834766;
    0.20909100943488412 0.2208129180125869; 0.18750000000000003 0.2213757117801786;
    0.2090910094348841 0.22081291801258685; 0.244032800421221 0.21516422119834752;
    0.24403280042122122 0.2081779527035683; 0.2090910094348842 0.21912527958959355;
    0.18749999999999992 0.23278940073547752],
   # Expected ppar_neutral:
   [0.18750000000000003 0.2482565420443467; 0.20909100943488412 0.24384477300624322;
    0.2440328004212212 0.228696297221881; 0.24403280042122125 0.20584407392704468;
    0.20909100943488423 0.19265693636741701; 0.18749999999999994 0.19084861718939317;
    0.2090910094348842 0.19265693636741688; 0.24403280042122116 0.20584407392704462;
    0.2440328004212211 0.228696297221881; 0.2090910094348841 0.2438447730062432;
    0.18750000000000003 0.24825654204434672],
   # Expected f_charged:
   [0.03704623609948259 0.04056128509273146 0.04289169811317835 0.030368915327672292 0.01235362235033934 0.0063385294703834204 0.012353622350339327 0.030368915327672247 0.04289169811317828 0.04056128509273145 0.0370462360994826;
    0.20411991941198782 0.251156132910555 0.3935556226209418 0.6276758497903185 0.9100827333021343 1.06066017177965 0.9100827333021342 0.6276758497903192 0.3935556226209421 0.25115613291055494 0.2041199194119877;
    0.03704623609948259 0.04056128509273146 0.04289169811317835 0.030368915327672292 0.01235362235033934 0.0063385294703834204 0.012353622350339327 0.030368915327672247 0.04289169811317828 0.04056128509273145 0.0370462360994826;;;
    0.05394101807537287 0.060554592436498814 0.036833910331906125 0.013633122209675089 0.010808508772375046 0.019345197472213343 0.027958746006592806 0.027628128813266543 0.026659296935378614 0.035613334811632306 0.05394101807537285;
    0.21177593449262566 0.24890460398430211 0.3731260689313831 0.5960741409510352 0.8872166610615642 1.0533874354116926 0.8872166610615648 0.5960741409510364 0.37312606893138234 0.24890460398430203 0.21177593449262566;
    0.053941018075372965 0.03561333481163235 0.026659296935378617 0.027628128813266564 0.02795874600659282 0.019345197472213388 0.010808508772375068 0.013633122209675051 0.03683391033190611 0.06055459243649881 0.05394101807537297],
   # Expected f_neutral:
   [0.006338529470383422 0.012353622350339336 0.030368915327672292 0.04289169811317835 0.04056128509273145 0.03704623609948259 0.04056128509273144 0.04289169811317833 0.03036891532767228 0.012353622350339327 0.00633852947038342;
    1.0606601717796502 0.9100827333021341 0.6276758497903185 0.3935556226209418 0.25115613291055494 0.2041199194119877 0.25115613291055505 0.3935556226209418 0.6276758497903185 0.9100827333021344 1.0606601717796504;
    0.006338529470383422 0.012353622350339336 0.030368915327672292 0.04289169811317835 0.04056128509273145 0.03704623609948259 0.04056128509273144 0.04289169811317833 0.03036891532767228 0.012353622350339327 0.00633852947038342;;;
    0.02430266037826662 0.040681127671090396 0.04194789083035397 0.036333231317680646 0.03689201427762576 0.04167458210401646 0.03666641377414817 0.019366777795459547 0.00833763923889437 0.009995913879120842 0.024302660378266627;
    1.0530041566990045 0.9037244245778953 0.6249909697155338 0.39563761233111516 0.2570339174716525 0.21139265577993924 0.25703391747165233 0.3956376123311148 0.6249909697155333 0.903724424577895 1.053004156699005;
    0.0243026603782666 0.009995913879120834 0.00833763923889439 0.01936677779545955 0.036666413774148206 0.041674582104016505 0.036892014277625805 0.036333231317680584 0.04194789083035393 0.04068112767109046 0.024302660378266613])

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
                                    "r_discretization" => "finite_difference",
                                    "z_ngrid" => 100,
                                    "z_nelement" => 1,
                                    "z_bc" => "periodic",
                                    "z_discretization" => "finite_difference",
                                    "vpa_ngrid" => 400,
                                    "vpa_nelement" => 1,
                                    "vpa_L" => vpa_L,
                                    "vpa_bc" => "periodic",
                                    "vpa_discretization" => "finite_difference",
                                    "vz_ngrid" => 400,
                                    "vz_nelement" => 1,
                                    "vz_L" => vpa_L,
                                    "vz_bc" => "periodic",
                                    "vz_discretization" => "finite_difference")

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
                                  "vpa_nelement" => 8,
                                  "vz_discretization" => "chebyshev_pseudospectral",
                                  "vz_ngrid" => 17,
                                  "vz_nelement" => 8))

test_input_chebyshev_split_1_moment =
    merge(test_input_chebyshev,
          Dict("run_name" => "chebyshev_pseudospectral_split_1_moment",
               "evolve_moments_density" => true,
               "z_nelement" => 4))

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
function run_test(test_input, rtol, atol, upar_rtol=nothing; args...)
    # by passing keyword arguments to run_test, args becomes a Dict which can be used to
    # update the default inputs

    if upar_rtol === nothing
        upar_rtol = rtol
    end

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

        # Create coordinates
        #
        # create the 'input' struct containing input info needed to create a coordinate
        # adv_input not actually used in this test so given values unimportant
        adv_input = advection_input("default", 1.0, 0.0, 0.0)
        nrank_per_block = 0 # dummy value
		irank = 0 # dummy value
		comm = MPI.COMM_NULL # dummy value
        element_spacing_option = "uniform"
		input = grid_input("coord", test_input["z_ngrid"], test_input["z_nelement"], 
                           test_input["z_nelement"], nrank_per_block, irank,
						   z_L, test_input["z_discretization"], "",
                           "periodic", #test_input["z_bc"],
                           adv_input,comm, element_spacing_option)
        z, z_spectral = define_coordinate(input)
        input = grid_input("coord", test_input["vpa_ngrid"], test_input["vpa_nelement"],
                           test_input["vpa_nelement"], nrank_per_block, irank,
						   vpa_L, test_input["vpa_discretization"], "",
                           test_input["vpa_bc"], adv_input, comm, element_spacing_option)
        vpa, vpa_spectral = define_coordinate(input)

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
                @test isapprox(expected.upar_charged[:, tind], newgrid_upar_charged[:,1], rtol=upar_rtol, atol=atol)

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
                @test isapprox(expected.upar_neutral[:, tind], newgrid_upar_neutral[:,:,1], rtol=upar_rtol, atol=atol)

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
            run_test(test_input_finite_difference, 1.e-3, 1.e-11, 2.e-3)
        end
        @testset "FD split 1" begin
            run_test(test_input_finite_difference_split_1_moment, 1.e-3, 1.e-11)
        end
        @testset_skip "grids need shift/scale for collisions" "FD split 2" begin
            run_test(test_input_finite_difference_split_2_moments, 1.e-3, 1.e-11)
        end
        @testset_skip "grids need shift/scale for collisions" "FD split 3" begin
            run_test(test_input_finite_difference_split_3_moments, 1.e-3, 1.e-11)
        end

        # Chebyshev pseudospectral
        # Benchmark data is taken from this run (Chebyshev with no splitting)
        @testset "Chebyshev base" begin
            run_test(test_input_chebyshev, 1.e-10, 3.e-16)
        end
        @testset "Chebyshev split 1" begin
            run_test(test_input_chebyshev_split_1_moment, 1.e-3, 1.e-15)
        end
        @testset_skip "grids need shift/scale for collisions" "Chebyshev split 2" begin
            run_test(test_input_chebyshev_split_2_moments, 1.e-3, 1.e-15)
        end
        @testset_skip "grids need shift/scale for collisions" "Chebyshev split 3" begin
            run_test(test_input_chebyshev_split_3_moments, 1.e-3, 1.e-15)
        end
    end
end

end # NonlinearSoundWaveTests


using .NonlinearSoundWaveTests

NonlinearSoundWaveTests.runtests()
