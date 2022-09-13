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
   [-1.3862943611198908 -1.2383045469542675; -1.2115160487200705 -1.1306858543409086;
    -0.8609385840190196 -0.8726873716115217; -0.5495323843850888 -0.590292028506168;
    -0.3534143971698439 -0.3756206845796865; -0.2876820724517811 -0.2919957810948679;
    -0.3534143971698439 -0.3756206845796866; -0.5495323843850888 -0.5902920285061679;
    -0.8609385840190196 -0.872687371611522; -1.2115160487200705 -1.1306858543409086;
    -1.3862943611198908 -1.238304546954267],
   # Expected n:
   cat(
     [0.25 0.75; 0.2977455383009383 0.7022544616990617;
      0.42274604116408965 0.5772539588359104; 0.5772539588359104 0.42274604116408965;
      0.7022544616990617 0.29774553830093836; 0.75 0.25;
      0.7022544616990617 0.29774553830093836; 0.5772539588359104 0.4227460411640896;
      0.42274604116408965 0.5772539588359104; 0.2977455383009383 0.7022544616990617;
      0.25 0.7499999999999999],
     [0.2898752714542009 0.7737996619185367; 0.32279889563424924 0.7056147470829168;
      0.41784220596750293 0.558319986991169; 0.554153634744472 0.4096819689562005;
      0.6868683062431302 0.30537567253162184; 0.7467716865256118 0.2681654439125427;
      0.68686830624313 0.30537567253162184; 0.5541536347444718 0.4096819689562005;
      0.41784220596750293 0.5583199869911688; 0.32279889563424913 0.7056147470829169;
      0.28987527145420106 0.7737996619185364], dims=3),
   # Expected upar:
   cat(
     [0.0 0.0; 0.0 0.0; 0.0 0.0; 0.0 0.0; 0.0 0.0; 0.0 0.0; 0.0 0.0; 0.0 0.0; 0.0 0.0;
      0.0 0.0; 0.0 0.0],
     [4.5102810375396984e-17 -9.86623976961809e-18; -0.18232118975621942 -0.036181844668896304;
      -0.1967240013780399 -0.0092263473252106; -0.11125439591412994 0.054572308563824605;
      -0.033747618389833936 0.07610770790787731; 1.734723475976807e-17 -4.456070928915423e-17;
      0.033747618389834116 -0.07610770790787723; 0.1112543959141299 -0.05457230856382462;
      0.19672400137804 0.009226347325210557; 0.18232118975621936 0.036181844668896186;
      5.312590645178972e-17 -7.26415455565288e-18], dims=3),
   # Expected ppar:
   cat(
     [0.1875 0.18750000000000006; 0.20909100943488423 0.20909100943488423;
      0.24403280042122125 0.2440328004212213; 0.2440328004212213 0.24403280042122125;
      0.20909100943488423 0.20909100943488423; 0.18750000000000006 0.1875;
      0.2090910094348842 0.20909100943488415; 0.2440328004212212 0.24403280042122127;
      0.2440328004212212 0.24403280042122122; 0.2090910094348842 0.2090910094348842;
      0.18749999999999994 0.18750000000000006],
     [0.2327894018948567 0.2482565421461269; 0.2191252806860825 0.243844773085044;
      0.20817795225960353 0.22869629722135848; 0.21516422029106266 0.20584407388961623;
      0.22081291795382973 0.19265693632654532; 0.22137571175436813 0.1908486171435662;
      0.22081291795382976 0.1926569363265453; 0.2151642202910627 0.20584407388961612;
      0.2081779522596035 0.2286962972213586; 0.21912528068608247 0.24384477308504401;
      0.23278940189485675 0.24825654214612694], dims=3),
   # Expected f:
   cat(
     cat(
       [0.03704623609948259 0.04056128509273144 0.04289169811317831 0.030368915327672306 0.01235362235033934 0.006338529470383427 0.01235362235033934 0.03036891532767224 0.042891698113178243 0.04056128509273144 0.0370462360994826;
        0.20411991941198782 0.25115613291055505 0.3935556226209419 0.6276758497903187 0.9100827333021342 1.06066017177965 0.9100827333021346 0.6276758497903189 0.3935556226209421 0.25115613291055505 0.20411991941198776;
        0.03704623609948259 0.04056128509273144 0.04289169811317831 0.030368915327672306 0.01235362235033934 0.006338529470383427 0.01235362235033934 0.03036891532767224 0.042891698113178243 0.04056128509273144 0.0370462360994826],
       [0.006338529470383427 0.012353622350339344 0.030368915327672306 0.04289169811317831 0.04056128509273147 0.03704623609948259 0.040561285092731464 0.042891698113178334 0.0303689153276723 0.012353622350339336 0.006338529470383412;
        1.06066017177965 0.9100827333021344 0.6276758497903187 0.3935556226209419 0.25115613291055494 0.20411991941198782 0.251156132910555 0.3935556226209419 0.6276758497903188 0.9100827333021342 1.0606601717796498;
        0.006338529470383427 0.012353622350339344 0.030368915327672306 0.04289169811317831 0.04056128509273147 0.03704623609948259 0.040561285092731464 0.042891698113178334 0.0303689153276723 0.012353622350339336 0.006338529470383412],
       dims=3),
     cat(
       [0.05394101835693815 0.06055459235026534 0.03683391042482269 0.013633122220571711 0.010808508798576085 0.019345197453545 0.02795874598821946 0.027628128341034865 0.026659296723991397 0.035613335483824025 0.053941018356938174;
        0.21177593422804294 0.24890460367837086 0.373126068648311 0.5960741408475303 0.8872166611651076 1.0533874357493165 0.8872166611651073 0.59607414084753 0.37312606864831166 0.24890460367837103 0.21177593422804286;
        0.05394101835693814 0.035613335483824046 0.026659296723991335 0.027628128341034907 0.027958745988219506 0.019345197453544963 0.010808508798576057 0.013633122220571742 0.03683391042482251 0.060554592350265356 0.05394101835693816],
       [0.02430266039522943 0.040681127671453016 0.041947890830957894 0.036333231315383414 0.03689201427541417 0.041674582102116095 0.03666641377427506 0.019366777780393175 0.008337639227436133 0.009995913911026824 0.024302660395229415;
        1.0530041569635882 0.9037244247204006 0.6249909697362629 0.3956376123287154 0.257033917317933 0.21139265544231367 0.25703391731793307 0.3956376123287153 0.6249909697362623 0.9037244247204006 1.0530041569635877;
        0.02430266039522943 0.009995913911026829 0.008337639227436145 0.019366777780393144 0.036666413774275104 0.04167458210211608 0.03689201427541415 0.036333231315383414 0.041947890830957915 0.04068112767145302 0.02430266039522944],
       dims=3), dims=4))

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
                @test isapprox(expected.upar[:, :, tind], newgrid_upar, rtol=rtol)

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
            run_test(test_input_finite_difference, 1.e-3)
        end
        @testset "FD split 1" begin
            run_test(test_input_finite_difference_split_1_moment, 1.e-3)
        end
        @testset_skip "grids need shift/scale for collisions" "FD split 2" begin
            run_test(test_input_finite_difference_split_2_moments, 1.e-3)
        end
        @testset_skip "grids need shift/scale for collisions" "FD split 3" begin
            run_test(test_input_finite_difference_split_3_moments, 1.e-3)
        end

        # Chebyshev pseudospectral
        # Benchmark data is taken from this run (Chebyshev with no splitting)
        @testset "Chebyshev base" begin
            run_test(test_input_chebyshev, 1.e-10)
        end
        @testset "Chebyshev split 1" begin
            run_test(test_input_chebyshev_split_1_moment, 1.e-3)
        end
        @testset_skip "grids need shift/scale for collisions" "Chebyshev split 2" begin
            run_test(test_input_chebyshev_split_2_moments, 1.e-3)
        end
        @testset_skip "grids need shift/scale for collisions" "Chebyshev split 3" begin
            run_test(test_input_chebyshev_split_3_moments, 1.e-3)
        end
    end
end

end # NonlinearSoundWaveTests


using .NonlinearSoundWaveTests

NonlinearSoundWaveTests.runtests()
