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
   [-1.3862943611198908 -1.2383045463629818; -1.2115160487200705 -1.13068585398684;
    -0.8609385840190196 -0.8726873715298277; -0.5495323843850888 -0.5902920285734169;
    -0.3534143971698439 -0.3756206847517388; -0.2876820724517811 -0.2919957813057993;
    -0.3534143971698439 -0.3756206847517386; -0.5495323843850888 -0.5902920285734168;
    -0.8609385840190196 -0.8726873715298276; -1.2115160487200705 -1.13068585398684;
    -1.3862943611198906 -1.2383045463629814],
   # Expected n:
   [0.2500030702177184 0.7499999999999392; 0.2977471383217195 0.702254450621661;
    0.42274614626845974 0.5772539714051019; 0.5772539714051019 0.42274614626845974;
    0.702254450621661 0.29774713832171956; 0.7499999999999392 0.2500030702177185;
    0.702254450621661 0.2977471383217197; 0.577253971405102 0.4227461462684595;
    0.42274614626845963 0.5772539714051017; 0.29774713832171945 0.7022544506216611;
    0.25000307021771856 0.7499999999999394;;; 0.2898752716256001 0.7737996617058853;
    0.3227988957484685 0.7056147469649242; 0.4178422060017743 0.5583199869733192;
    0.5541536347070377 0.4096819689742094; 0.6868683061250952 0.30537567264957566;
    0.7467716863680942 0.26816544412495724; 0.6868683061250951 0.30537567264957544;
    0.5541536347070379 0.4096819689742097; 0.41784220600177435 0.5583199869733193;
    0.3227988957484684 0.7056147469649247; 0.28987527162560023 0.7737996617058853],
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
    -4.87890977618477e-17 2.47198095326695e-17; -0.18232118950261103 -0.03618184472105245;
    -0.19672400124405012 -0.009226347353430713; -0.111254395822785 0.054572308523873916;
    -0.03374761828134262 0.07610770777087446; 3.903127820947816e-17 6.873841773558098e-17;
    0.03374761828134244 -0.07610770777087458; 0.11125439582278493 -0.054572308523873875;
    0.19672400124405037 0.00922634735343074;
    0.18232118950261114 0.03618184472105236;
    -5.463810853260936e-17 3.570936628678348e-17],
   # Expected ppar:
   [0.18749999999999997 0.18750000000000003; 0.20909100943488423 0.20909100943488412;
    0.24403280042122125 0.2440328004212212; 0.2440328004212212 0.24403280042122125;
    0.20909100943488412 0.20909100943488423; 0.18750000000000003 0.18749999999999994;
    0.2090910094348841 0.2090910094348842; 0.244032800421221 0.24403280042122116;
    0.24403280042122122 0.2440328004212211; 0.2090910094348842 0.2090910094348841;
    0.18749999999999992 0.18750000000000003;;; 0.23278940185673783 0.24825654209861486;
    0.2191252807139984 0.24384477304984092; 0.20817795230311276 0.2286962972103614;
    0.21516422027442467 0.20584407390260756; 0.22081291792546837 0.19265693636401973;
    0.221375711749034 0.19084861718892623; 0.22081291792546837 0.19265693636401973;
    0.21516422027442472 0.2058440739026075; 0.2081779523031126 0.22869629721036128;
    0.21912528071399817 0.24384477304984079; 0.23278940185673783 0.24825654209861486],
   # Expected f:
   [0.03704623609948259 0.04056128509273146 0.04289169811317835 0.030368915327672292 0.01235362235033934 0.0063385294703834204 0.012353622350339327 0.030368915327672247 0.04289169811317828 0.04056128509273145 0.0370462360994826;
    0.20411991941198782 0.251156132910555 0.3935556226209418 0.6276758497903185 0.9100827333021343 1.06066017177965 0.9100827333021342 0.6276758497903192 0.3935556226209421 0.25115613291055494 0.2041199194119877;
    0.03704623609948259 0.04056128509273146 0.04289169811317835 0.030368915327672292 0.01235362235033934 0.0063385294703834204 0.012353622350339327 0.030368915327672247 0.04289169811317828 0.04056128509273145 0.0370462360994826;;;
    0.006338529470383422 0.012353622350339336 0.030368915327672292 0.04289169811317835 0.04056128509273145 0.03704623609948259 0.04056128509273144 0.04289169811317833 0.03036891532767228 0.012353622350339327 0.00633852947038342;
    1.0606601717796502 0.9100827333021341 0.6276758497903185 0.3935556226209418 0.25115613291055494 0.2041199194119877 0.25115613291055505 0.3935556226209418 0.6276758497903185 0.9100827333021344 1.0606601717796504;
    0.006338529470383422 0.012353622350339336 0.030368915327672292 0.04289169811317835 0.04056128509273145 0.03704623609948259 0.04056128509273144 0.04289169811317833 0.03036891532767228 0.012353622350339327 0.00633852947038342;;;;
    0.053941018332391975 0.06055459233234704 0.03683391041400562 0.01363312221008983 0.010808508795685883 0.019345197457361595 0.027958745991323922 0.027628128348009626 0.026659296739883084 0.035613335487055406 0.053941018332391996;
    0.21177593456447602 0.24890460386660612 0.3731260686685075 0.5960741408603063 0.8872166610479093 1.0533874354116672 0.8872166610479104 0.5960741408603063 0.37312606866850717 0.24890460386660582 0.21177593456447602;
    0.05394101833239206 0.035613335487055454 0.02665929673988311 0.027628128348009623 0.027958745991323905 0.01934519745736161 0.010808508795685876 0.01363312221008982 0.036833910414005626 0.060554592332346995 0.053941018332392066;;;
    0.02430266039287735 0.04068112767108886 0.041947890832283334 0.03633323131760558 0.03689201427794907 0.04167458210370478 0.036666413774383615 0.019366777779091234 0.008337639225557955 0.00999591390830718 0.024302660392877363;
    1.0530041566271542 0.9037244245663942 0.6249909697337432 0.39563761233144573 0.25703391747291604 0.2113926557799654 0.25703391747291604 0.39563761233144573 0.624990969733743 0.9037244245663943 1.0530041566271546;
    0.024302660392877318 0.009995913908307178 0.008337639225557981 0.019366777779091245 0.0366664137743837 0.04167458210370474 0.03689201427794906 0.03633323131760558 0.04194789083228321 0.04068112767108885 0.024302660392877307])

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
