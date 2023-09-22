module NonlinearSoundWaveTests

include("setup.jl")

using Base.Filesystem: tempname
using MPI
using TimerOutputs

using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.input_structs: grid_input, advection_input
using moment_kinetics.load_data: open_readonly_output_file, load_coordinate_data,
                                 load_species_data, load_fields_data,
                                 load_charged_particle_moments_data, load_pdf_data,
                                 load_neutral_particle_moments_data,
                                 load_neutral_pdf_data, load_time_data, load_species_data
using moment_kinetics.interpolation: interpolate_to_grid_z, interpolate_to_grid_vpa
using moment_kinetics.type_definitions: mk_float

const analytical_rtol = 3.e-2
const regression_rtol = 2.e-8

# Create a temporary directory for test output
test_output_directory = tempname()
mkpath(test_output_directory)

include("nonlinear_sound_wave_inputs_and_expected_data.jl")

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
    quietoutput() do
        # run simulation
        run_moment_kinetics(to, input)
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
                @test isapprox(expected.upar_charged[:, tind], newgrid_upar_charged[:,1], rtol=upar_rtol, atol=atol)

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
    @testset "nonlinear sound wave" verbose=use_verbose begin
        println("nonlinear sound wave tests")

        # finite difference
        @testset "FD base" begin
            run_test(test_input_finite_difference, 1.e-3, 1.e-11, 2.e-3)
        end
        @testset "FD split 1" begin
            run_test(test_input_finite_difference_split_1_moment, 1.e-3, 1.e-11)
        end
        @testset "FD split 2" begin
            run_test(test_input_finite_difference_split_2_moments, 1.e-3, 1.e-11)
        end
        @testset "FD split 3" begin
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
        @testset "Chebyshev split 2" begin
            run_test(test_input_chebyshev_split_2_moments, 1.e-3, 1.e-15)
        end
        @testset "Chebyshev split 3" begin
            run_test(test_input_chebyshev_split_3_moments, 1.e-3, 1.e-15)
        end
    end
end

end # NonlinearSoundWaveTests


using .NonlinearSoundWaveTests

NonlinearSoundWaveTests.runtests()
