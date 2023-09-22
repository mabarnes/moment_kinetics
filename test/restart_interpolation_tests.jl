module RestartInterpolationTests

# Test for restart interpolation, based on NonlinearSoundWave test

include("setup.jl")

using Base.Filesystem: tempname
using MPI

using moment_kinetics.communication
using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.file_io: io_has_parallel
using moment_kinetics.input_structs: grid_input, advection_input, hdf5
using moment_kinetics.load_data: open_readonly_output_file, load_coordinate_data,
                                 load_species_data, load_fields_data,
                                 load_charged_particle_moments_data, load_pdf_data,
                                 load_neutral_particle_moments_data,
                                 load_neutral_pdf_data, load_time_data, load_species_data
using moment_kinetics.interpolation: interpolate_to_grid_z, interpolate_to_grid_vpa
using moment_kinetics.type_definitions: mk_float

# Create a temporary directory for test output
test_output_directory = tempname()
mkpath(test_output_directory)

include("nonlinear_sound_wave_inputs_and_expected_data.jl")

base_input = copy(test_input_chebyshev)
base_input["base_directory"] = test_output_directory
base_input["nstep"] = 50
base_input["nwrite"] = 50
base_input["nwrite_dfns"] = 50
if global_size[] > 1 && global_size[] % 2 == 0
    # Test using distributed-memory
    base_input["z_nelement"] /= 2
end
base_input["output"] = Dict{String,Any}("parallel_io" => false)

restart_test_input_chebyshev =
    merge(base_input,
          Dict("run_name" => "restart_chebyshev_pseudospectral",
               "z_ngrid" => 17, "z_nelement" => 2,
               "vpa_ngrid" => 9, "vpa_nelement" => 32,
               "vz_ngrid" => 9, "vz_nelement" => 32))

restart_test_input_chebyshev_split_1_moment =
    merge(restart_test_input_chebyshev,
          Dict("run_name" => "restart_chebyshev_pseudospectral_split_1_moment",
               "evolve_moments_density" => true))

restart_test_input_chebyshev_split_2_moments =
    merge(restart_test_input_chebyshev_split_1_moment,
          Dict("run_name" => "restart_chebyshev_pseudospectral_split_2_moments",
               "evolve_moments_parallel_flow" => true))

restart_test_input_chebyshev_split_3_moments =
    merge(restart_test_input_chebyshev_split_2_moments,
          Dict("run_name" => "restart_chebyshev_pseudospectral_split_3_moments",
               "evolve_moments_parallel_pressure" => true,
               "vpa_L" => 1.5*vpa_L, "vz_L" => 1.5*vpa_L))

"""
Run a sound-wave test for a single set of parameters
"""
# Note 'name' should not be shared by any two tests in this file
function run_test(test_input, message, rtol, atol; kwargs...)
    # by passing keyword arguments to run_test, kwargs becomes a Tuple of Pairs which can be used to
    # update the default inputs

    parallel_io = test_input["output"]["parallel_io"]
    # Convert keyword arguments to a unique name
    name = test_input["run_name"]
    if length(kwargs) > 0
        name = string(name, "_", (string(k, "-", v, "_") for (k, v) in kwargs)...)

        # Remove trailing "_"
        name = chop(name)
    end
    if parallel_io
        name *= "_parallel-io"
    end

    # Provide some progress info
    println("    - testing ", message)

    # Convert from Tuple of Pairs with symbol keys to Dict with String keys
    modified_inputs = Dict(String(k) => v for (k, v) in kwargs)

    # Update default inputs with values to be changed
    input = merge(test_input, modified_inputs)

    input["run_name"] = name

    # Suppress console output while running
    quietoutput() do
        # run simulation
        if parallel_io
            restart_filename = joinpath(base_input["base_directory"],
                                        base_input["run_name"],
                                        base_input["run_name"] * ".dfns.h5")
        else
            restart_filename = joinpath(base_input["base_directory"],
                                        base_input["run_name"],
                                        base_input["run_name"] * ".dfns.0.h5")
        end
        run_moment_kinetics(input; restart=restart_filename)
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

        newgrid_phi = interpolate_to_grid_z(expected.z, phi[:, end], z, z_spectral)
        @test isapprox(expected.phi[:, end], newgrid_phi, rtol=rtol)

        # Check charged particle moments and f
        ######################################

        newgrid_n_charged = interpolate_to_grid_z(expected.z, n_charged[:, :, end], z, z_spectral)
        @test isapprox(expected.n_charged[:, end], newgrid_n_charged[:,1], rtol=rtol)

        newgrid_upar_charged = interpolate_to_grid_z(expected.z, upar_charged[:, :, end], z, z_spectral)
        @test isapprox(expected.upar_charged[:, end], newgrid_upar_charged[:,1], rtol=rtol, atol=atol)

        newgrid_ppar_charged = interpolate_to_grid_z(expected.z, ppar_charged[:, :, end], z, z_spectral)
        @test isapprox(expected.ppar_charged[:, end], newgrid_ppar_charged[:,1], rtol=rtol)

        newgrid_vth_charged = @. sqrt(2.0*newgrid_ppar_charged/newgrid_n_charged)
        newgrid_f_charged = interpolate_to_grid_z(expected.z, f_charged[:, :, :, end], z, z_spectral)
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
        @test isapprox(expected.f_charged[:, :, end], newgrid_f_charged[:,:,1], rtol=rtol)

        # Check neutral particle moments and f
        ######################################

        newgrid_n_neutral = interpolate_to_grid_z(expected.z, n_neutral[:, :, end], z, z_spectral)
        @test isapprox(expected.n_neutral[:, end], newgrid_n_neutral[:,1], rtol=rtol)

        newgrid_upar_neutral = interpolate_to_grid_z(expected.z, upar_neutral[:, :, end], z, z_spectral)
        @test isapprox(expected.upar_neutral[:, end], newgrid_upar_neutral[:,1], rtol=rtol, atol=atol)

        newgrid_ppar_neutral = interpolate_to_grid_z(expected.z, ppar_neutral[:, :, end], z, z_spectral)
        @test isapprox(expected.ppar_neutral[:, end], newgrid_ppar_neutral[:,1], rtol=rtol)

        newgrid_vth_neutral = @. sqrt(2.0*newgrid_ppar_neutral/newgrid_n_neutral)
        newgrid_f_neutral = interpolate_to_grid_z(expected.z, f_neutral[:, :, :, end], z, z_spectral)
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
        @test isapprox(expected.f_neutral[:, :, end], newgrid_f_neutral[:,:,1], rtol=rtol)
    end
end


function runtests()
    function do_tests(label)
        # Only testing Chebyshev discretization because interpolation not yet implemented
        # for finite-difference

        parallel_io = base_input["output"]["parallel_io"]

        base_input_evolve_density = merge(base_input,
                                          Dict("evolve_moments_density" => true))
        base_input_evolve_upar = merge(base_input_evolve_density,
                                       Dict("evolve_moments_parallel_flow" => true,
                                            "vpa_L" => 1.5*vpa_L, "vz_L" => 1.5*vpa_L))
        base_input_evolve_ppar = merge(base_input_evolve_upar,
                                       Dict("evolve_moments_parallel_pressure" => true,
                                            "vpa_L" => 1.5*vpa_L, "vz_L" => 1.5*vpa_L))

        for (base, base_label) ∈ ((base_input, "full-f"),
                                  (base_input_evolve_density, "split 1"),
                                  (base_input_evolve_upar, "split 2"),
                                  (base_input_evolve_ppar, "split 3"))
            # Base run, from which tests are restarted
            # Suppress console output while running
            quietoutput() do
                # run simulation
                run_moment_kinetics(base)
            end

            # Benchmark data is taken from this run (full-f with no splitting)
            message = "restart full-f from $base_label$label"
            @testset "$message" begin
                run_test(restart_test_input_chebyshev, message, 1.e-3, 1.e-15)
            end
            message = "restart split 1 from $base_label$label"
            @testset "$message" begin
                run_test(restart_test_input_chebyshev_split_1_moment, message, 1.e-3, 1.e-15)
            end
            message = "restart split 2 from $base_label$label"
            @testset "$message" begin
                run_test(restart_test_input_chebyshev_split_2_moments, message, 1.e-3, 1.e-15)
            end
            message = "restart split 3 from $base_label$label"
            @testset "$message" begin
                run_test(restart_test_input_chebyshev_split_3_moments, message, 1.e-3, 1.e-15)
            end
        end
    end

    @testset "restart interpolation" verbose=use_verbose begin
        println("restart interpolation tests")
        do_tests("")

        if io_has_parallel(Val(hdf5))
            orig_base_input = copy(base_input)
            # Also test not using parallel_io
            base_input["output"]["parallel_io"] = true
            base_input["run_name"] *= "_parallel_io"
            do_tests(", parallel I/O")
            global base_input = orig_base_input
        end
    end
end

end # RestartInterpolationTests


using .RestartInterpolationTests

RestartInterpolationTests.runtests()
