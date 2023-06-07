module NonlinearSoundWaveTests

include("setup.jl")

using Base.Filesystem: tempname
using TimerOutputs

using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.input_structs: grid_input, advection_input
using moment_kinetics.load_data: open_netcdf_file, load_coordinate_data,
                                 load_fields_data, load_moments_data, load_pdf_data
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
    n::Array{mk_float, 3}
    upar::Array{mk_float, 3}
    ppar::Array{mk_float, 3}
    f::Array{mk_float, 4}
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
   # Expected n:
   [0.2500030702177184 0.7499999999999392; 0.2977471383217195 0.702254450621661;
    0.42274614626845974 0.5772539714051019; 0.5772539714051019 0.42274614626845974;
    0.702254450621661 0.29774713832171956; 0.7499999999999392 0.2500030702177185;
    0.702254450621661 0.2977471383217197; 0.577253971405102 0.4227461462684595;
    0.42274614626845963 0.5772539714051017; 0.29774713832171945 0.7022544506216611;
    0.25000307021771856 0.7499999999999394;;; 0.2898752704335188 0.7737996616909211;
    0.3227988953227183 0.7056147469533546; 0.417842206578383 0.5583199869826109;
    0.5541536351162784 0.4096819689829928; 0.686868306132489 0.30537567265010457;
    0.7467716863438243 0.26816544412496246; 0.6868683061324891 0.30537567265010435;
    0.5541536351162781 0.4096819689829924; 0.41784220657838306 0.5583199869826102;
    0.3227988953227184 0.7056147469533546; 0.2898752704335188 0.7737996616909211],
   # Expected upar:
   [1.1971912119126474e-17 -3.143783114880993e-17;
    -5.818706134342973e-17 -1.7717259792356296e-17;
    9.895531571141618e-17 -8.38774587436108e-18;
    -8.38774587436108e-18 9.895531571141618e-17;
    -1.7717259792356293e-17 -5.818706134342965e-17;
    -3.143783114880992e-17 1.1971912119126477e-17;
    -2.499642278253839e-17 4.747047511913174e-17;
    -2.9272523290371316e-17 6.558391323223502e-18;
    3.346728365577734e-17 2.2213638810498713e-17;
    -8.193702949354942e-17 -2.7413075842225616e-17;
    1.1971912119126474e-17 -3.143783114880993e-17;;;
    -2.0968470015869656e-16 -3.003240017784847e-17;
    -0.18232119131671534 -0.03618184473095593; -0.1967239995126128 -0.009226347310827927;
    -0.11125439467488389 0.054572308562824384; -0.033747618153236424 0.0761077077764258;
    9.84455572616838e-17 2.1033522146218786e-16; 0.03374761815323648 -0.07610770777642595;
    0.1112543946748839 -0.05457230856282445; 0.19672399951261313 0.009226347310827925;
    0.18232119131671523 0.036181844730955835;
    -2.0187639060277426e-16 -2.810175715538666e-17],
   # Expected ppar:
   [0.18749999999999997 0.18750000000000003; 0.20909100943488423 0.20909100943488412;
    0.24403280042122125 0.2440328004212212; 0.2440328004212212 0.24403280042122125;
    0.20909100943488412 0.20909100943488423; 0.18750000000000003 0.18749999999999994;
    0.2090910094348841 0.2090910094348842; 0.244032800421221 0.24403280042122116;
    0.24403280042122122 0.2440328004212211; 0.2090910094348842 0.2090910094348841;
    0.18749999999999992 0.18750000000000003;;; 0.23278940073547755 0.2482565420443467;
    0.21912527958959363 0.24384477300624322; 0.20817795270356831 0.228696297221881;
    0.21516422119834766 0.20584407392704468; 0.2208129180125869 0.19265693636741701;
    0.2213757117801786 0.19084861718939317; 0.22081291801258685 0.19265693636741688;
    0.21516422119834752 0.20584407392704462; 0.2081779527035683 0.228696297221881;
    0.21912527958959355 0.2438447730062432; 0.23278940073547752 0.24825654204434672],
   # Expected f:
   [0.03704623609948259 0.04056128509273146 0.04289169811317835 0.030368915327672292 0.01235362235033934 0.0063385294703834204 0.012353622350339327 0.030368915327672247 0.04289169811317828 0.04056128509273145 0.0370462360994826;
    0.20411991941198782 0.251156132910555 0.3935556226209418 0.6276758497903185 0.9100827333021343 1.06066017177965 0.9100827333021342 0.6276758497903192 0.3935556226209421 0.25115613291055494 0.2041199194119877;
    0.03704623609948259 0.04056128509273146 0.04289169811317835 0.030368915327672292 0.01235362235033934 0.0063385294703834204 0.012353622350339327 0.030368915327672247 0.04289169811317828 0.04056128509273145 0.0370462360994826;;;
    0.006338529470383422 0.012353622350339336 0.030368915327672292 0.04289169811317835 0.04056128509273145 0.03704623609948259 0.04056128509273144 0.04289169811317833 0.03036891532767228 0.012353622350339327 0.00633852947038342;
    1.0606601717796502 0.9100827333021341 0.6276758497903185 0.3935556226209418 0.25115613291055494 0.2041199194119877 0.25115613291055505 0.3935556226209418 0.6276758497903185 0.9100827333021344 1.0606601717796504;
    0.006338529470383422 0.012353622350339336 0.030368915327672292 0.04289169811317835 0.04056128509273145 0.03704623609948259 0.04056128509273144 0.04289169811317833 0.03036891532767228 0.012353622350339327 0.00633852947038342;;;;
    0.05394101807537287 0.060554592436498814 0.036833910331906125 0.013633122209675089 0.010808508772375046 0.019345197472213343 0.027958746006592806 0.027628128813266543 0.026659296935378614 0.035613334811632306 0.05394101807537285;
    0.21177593449262566 0.24890460398430211 0.3731260689313831 0.5960741409510352 0.8872166610615642 1.0533874354116926 0.8872166610615648 0.5960741409510364 0.37312606893138234 0.24890460398430203 0.21177593449262566;
    0.053941018075372965 0.03561333481163235 0.026659296935378617 0.027628128813266564 0.02795874600659282 0.019345197472213388 0.010808508772375068 0.013633122209675051 0.03683391033190611 0.06055459243649881 0.05394101807537297;;;
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
    phi = undef
    n = undef
    upar = undef
    ppar = undef
    f = undef
    quietoutput() do
        # run simulation
        run_moment_kinetics(to, input)
    end

    if global_rank[] == 0
        quietoutput() do

            # Load and analyse output
            #########################

            path = joinpath(realpath(input["base_directory"]), name, name)

            # open the netcdf file and give it the handle 'fid'
            fid = open_netcdf_file(path)

            # load space-time coordinate data
            nvpa, vpa, vpa_wgts, nz, z, z_wgts, Lz, nr, r, r_wgts, Lr, ntime, time = load_coordinate_data(fid)

            # load fields data
            phi_zrt = load_fields_data(fid)

            # load velocity moments data
            n_zrst, upar_zrst, ppar_zrst, qpar_zrst, v_t_zrst, n_species, evolve_ppar = load_moments_data(fid)

            # load particle distribution function (pdf) data
            f_vpazrst = load_pdf_data(fid)

            close(fid)
            
            phi = phi_zrt[:,1,:]
            n, upar, ppar, qpar, v_t = n_zrst[:,1,:,:], upar_zrst[:,1,:,:], ppar_zrst[:,1,:,:], qpar_zrst[:,1,:,:], v_t_zrst[:,1,:,:]
            f = f_vpazrst[:,:,1,:,:]

            # Unnormalize f
            if input["evolve_moments_density"]
                for it ∈ 1:length(time), is ∈ 1:n_species, iz ∈ 1:nz
                    f[:,iz,is,it] .*= n[iz,is,it]
                end
            end
            if input["evolve_moments_parallel_pressure"]
                for it ∈ 1:length(time), is ∈ 1:n_species, iz ∈ 1:nz
                    vth = sqrt(2.0*ppar[iz,is,it]/n[iz,is,it])
                    f[:,iz,is,it] ./= vth
                end
            end
        end

        # Create coordinates
        #
        # create the 'input' struct containing input info needed to create a coordinate
        # adv_input not actually used in this test so given values unimportant
        adv_input = advection_input("default", 1.0, 0.0, 0.0)
        input = grid_input("coord", test_input["z_ngrid"], test_input["z_nelement"],
                           z_L, test_input["z_discretization"], "",
                           "periodic", #test_input["z_bc"],
                           adv_input)
        z, z_spectral = define_coordinate(input)
        input = grid_input("coord", test_input["vpa_ngrid"], test_input["vpa_nelement"],
                           vpa_L, test_input["vpa_discretization"], "",
                           test_input["vpa_bc"], adv_input)
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
        #newgrid_n = cat(interpolate_to_grid_z(expected.z, n[:, :, 1], z, z_spectral),
        #                   interpolate_to_grid_z(expected.z, n[:, :, 2], z, z_spectral);
        #                   dims=3)
        #println("n ", size(newgrid_n))
        #println(newgrid_n)
        #println()
        #newgrid_upar = cat(interpolate_to_grid_z(expected.z, upar[:, :, 1], z, z_spectral),
        #                   interpolate_to_grid_z(expected.z, upar[:, :, 2], z, z_spectral);
        #                   dims=3)
        #println("upar ", size(newgrid_upar))
        #println(newgrid_upar)
        #println()
        #newgrid_ppar = cat(interpolate_to_grid_z(expected.z, ppar[:, :, 1], z, z_spectral),
        #                   interpolate_to_grid_z(expected.z, ppar[:, :, 2], z, z_spectral);
        #                   dims=3)
        #println("ppar ", size(newgrid_ppar))
        #println(newgrid_ppar)
        #println()
        #newgrid_f = cat(interpolate_to_grid_vpa(expected.vpa, interpolate_to_grid_z(expected.z, f[:, :, :, 1], z, z_spectral), vpa, vpa_spectral),
        #                interpolate_to_grid_vpa(expected.vpa, interpolate_to_grid_z(expected.z, f[:, :, :, 2], z, z_spectral), vpa, vpa_spectral);
        #                dims=4)
        #println("f ", size(newgrid_f))
        #println(newgrid_f)
        #println()
        function test_values(tind)
            @testset "tind=$tind" begin
                newgrid_phi = interpolate_to_grid_z(expected.z, phi[:, tind], z, z_spectral)
                @test isapprox(expected.phi[:, tind], newgrid_phi, rtol=rtol)

                newgrid_n = interpolate_to_grid_z(expected.z, n[:, :, tind], z, z_spectral)
                @test isapprox(expected.n[:, :, tind], newgrid_n, rtol=rtol)

                newgrid_upar = interpolate_to_grid_z(expected.z, upar[:, :, tind], z, z_spectral)
                @test isapprox(expected.upar[:, :, tind], newgrid_upar, rtol=rtol, atol=atol)

                newgrid_ppar = interpolate_to_grid_z(expected.z, ppar[:, :, tind], z, z_spectral)
                @test isapprox(expected.ppar[:, :, tind], newgrid_ppar, rtol=rtol)

                newgrid_f = interpolate_to_grid_z(expected.z, f[:, :, :, tind], z, z_spectral)
                newgrid_f = interpolate_to_grid_vpa(expected.vpa, newgrid_f, vpa, vpa_spectral)
                @test isapprox(expected.f[:, :, :, tind], newgrid_f, rtol=rtol)
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
            run_test(test_input_finite_difference, 1.e-3, 1.e-11)
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
            run_test(test_input_chebyshev, 1.e-10, 0.0)
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
