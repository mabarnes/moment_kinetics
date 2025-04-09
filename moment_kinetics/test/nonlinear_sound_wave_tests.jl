module NonlinearSoundWaveTests

include("setup.jl")

using Base.Filesystem: tempname

using moment_kinetics.interpolation: interpolate_to_grid_z, interpolate_to_grid_vpa
using moment_kinetics.load_data: get_run_info_no_setup, close_run_info,
                                 postproc_load_variable
using moment_kinetics.type_definitions: mk_float
using moment_kinetics.utils: merge_dict_with_kwargs!

const analytical_rtol = 3.e-2
const regression_rtol = 2.e-8

include("nonlinear_sound_wave_inputs_and_expected_data.jl")

"""
Run a sound-wave test for a single set of parameters
"""
# Note 'name' should not be shared by any two tests in this file
function run_test(test_input, rtol, atol, upar_rtol=nothing; args...)
    # by passing keyword arguments to run_test, args becomes a Dict which can be used to
    # update the default inputs

    # Make a copy to make sure nothing modifies the input Dicts defined in this test
    # script.
    input = deepcopy(test_input)

    if upar_rtol === nothing
        upar_rtol = rtol
    end

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
    ppar_ion = nothing
    f_ion = nothing
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

            path = joinpath(realpath(input["output"]["base_directory"]), name)

            # open the output file(s)
            run_info = get_run_info_no_setup(path; dfns=true)

            # load species, time coordinate data
            n_ion_species = run_info.composition.n_ion_species
            n_neutral_species = run_info.composition.n_neutral_species
            ntime = run_info.nt
            time = run_info.time
            
            # load fields data
            phi_zrt = postproc_load_variable(run_info, "phi")
            Er_zrt = postproc_load_variable(run_info, "Er")
            Ez_zrt = postproc_load_variable(run_info, "Ez")

            # load velocity moments data
            n_ion_zrst = postproc_load_variable(run_info, "density")
            upar_ion_zrst = postproc_load_variable(run_info, "parallel_flow")
            ppar_ion_zrst = postproc_load_variable(run_info, "parallel_pressure")
            qpar_ion_zrst = postproc_load_variable(run_info, "parallel_heat_flux")
            v_t_ion_zrst = postproc_load_variable(run_info, "thermal_speed")
            n_neutral_zrst = postproc_load_variable(run_info, "density_neutral")
            upar_neutral_zrst = postproc_load_variable(run_info, "uz_neutral")
            ppar_neutral_zrst = postproc_load_variable(run_info, "pz_neutral")
            qpar_neutral_zrst = postproc_load_variable(run_info, "qz_neutral")
            v_t_neutral_zrst = postproc_load_variable(run_info, "thermal_speed_neutral")
            z = run_info.z
            z_spectral = run_info.z_spectral

            # load particle distribution function (pdf) data
            f_ion_vpavperpzrst = postproc_load_variable(run_info, "f")
            f_neutral_vzvrvzetazrst = postproc_load_variable(run_info, "f_neutral")
            vpa = run_info.vpa
            vpa_spectral = run_info.vpa_spectral

            close_run_info(run_info)
            
            phi = phi_zrt[:,1,:]
            n_ion = n_ion_zrst[:,1,:,:]
            upar_ion = upar_ion_zrst[:,1,:,:]
            ppar_ion = ppar_ion_zrst[:,1,:,:]
            qpar_ion = qpar_ion_zrst[:,1,:,:]
            v_t_ion = v_t_ion_zrst[:,1,:,:]
            f_ion = f_ion_vpavperpzrst[:,1,:,1,:,:]
            n_neutral = n_neutral_zrst[:,1,:,:]
            upar_neutral = upar_neutral_zrst[:,1,:,:]
            ppar_neutral = ppar_neutral_zrst[:,1,:,:]
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
            if input["evolve_moments"]["parallel_pressure"]
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
        #newgrid_ppar_ion = cat(interpolate_to_grid_z(expected.z, ppar_ion[:, :, 1], z, z_spectral)[:,1],
        #                           interpolate_to_grid_z(expected.z, ppar_ion[:, :, 2], z, z_spectral)[:,1];
        #                           dims=2)
        #println("ppar_ion ", size(newgrid_ppar_ion))
        #println(newgrid_ppar_ion)
        #println()
        #newgrid_ppar_neutral = cat(interpolate_to_grid_z(expected.z, ppar_neutral[:, :, 1], z, z_spectral)[:,1],
        #                           interpolate_to_grid_z(expected.z, ppar_neutral[:, :, 2], z, z_spectral)[:,1];
        #                           dims=2)
        #println("ppar_neutral ", size(newgrid_ppar_neutral))
        #println(newgrid_ppar_neutral)
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
                @test isapprox(expected.upar_ion[:, tind], newgrid_upar_ion[:,1], rtol=upar_rtol, atol=atol)

                newgrid_ppar_ion = interpolate_to_grid_z(expected.z, ppar_ion[:, :, tind], z, z_spectral)
                @test isapprox(expected.ppar_ion[:, tind], newgrid_ppar_ion[:,1], rtol=rtol)

                newgrid_vth_ion = @. sqrt(2.0*newgrid_ppar_ion/newgrid_n_ion)
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
                    if input["evolve_moments"]["parallel_pressure"]
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
                @test isapprox(expected.upar_neutral[:, tind], newgrid_upar_neutral[:,:,1], rtol=upar_rtol, atol=atol)

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
                    if input["evolve_moments"]["parallel_flow"]
                        wpa .-= newgrid_upar_neutral[iz,1]
                    end
                    if input["evolve_moments"]["parallel_pressure"]
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

    @testset "nonlinear sound wave" verbose=use_verbose begin
        println("nonlinear sound wave tests")

        # finite difference
        @testset "FD base" begin
            test_input_finite_difference["output"]["base_directory"] = test_output_directory
            run_test(test_input_finite_difference, 1.e-3, 1.e-11, 2.e-3)
        end
        @testset "FD split 1" begin
            test_input_finite_difference_split_1_moment["output"]["base_directory"] = test_output_directory
            run_test(test_input_finite_difference_split_1_moment, 1.e-3, 1.e-11)
        end
        @testset "FD split 2" begin
            test_input_finite_difference_split_2_moments["output"]["base_directory"] = test_output_directory
            run_test(test_input_finite_difference_split_2_moments, 1.e-3, 1.e-11)
        end
        @testset "FD split 3" begin
            test_input_finite_difference_split_3_moments["output"]["base_directory"] = test_output_directory
            run_test(test_input_finite_difference_split_3_moments, 1.e-3, 1.e-11)
        end

        # Chebyshev pseudospectral
        # Benchmark data is taken from this run (Chebyshev with no splitting)
        @testset "Chebyshev base" begin
            test_input_chebyshev["output"]["base_directory"] = test_output_directory
            run_test(test_input_chebyshev, 1.e-10, 3.e-16)
        end
        @testset "Chebyshev split 1" begin
            test_input_chebyshev_split_1_moment["output"]["base_directory"] = test_output_directory
            run_test(test_input_chebyshev_split_1_moment, 1.e-3, 1.e-15)
        end
        @testset "Chebyshev split 2" begin
            test_input_chebyshev_split_2_moments["output"]["base_directory"] = test_output_directory
            run_test(test_input_chebyshev_split_2_moments, 1.e-3, 1.e-15)
        end
        @testset "Chebyshev split 3" begin
            test_input_chebyshev_split_3_moments["output"]["base_directory"] = test_output_directory
            run_test(test_input_chebyshev_split_3_moments, 1.e-3, 1.e-15)
        end
    end

    if global_rank[] == 0
        # Delete output directory to avoid using too much disk space
        rm(realpath(test_output_directory); recursive=true)
    end
end

end # NonlinearSoundWaveTests


using .NonlinearSoundWaveTests

NonlinearSoundWaveTests.runtests()
