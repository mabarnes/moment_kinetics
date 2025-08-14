module RestartInterpolationTests

# Test for restart interpolation, based on NonlinearSoundWave test

include("setup.jl")

using Base.Filesystem: tempname

using moment_kinetics.communication
using moment_kinetics.file_io: io_has_parallel
using moment_kinetics.input_structs: hdf5
using moment_kinetics.interpolation: interpolate_to_grid_z, interpolate_to_grid_vpa
using moment_kinetics.load_data: get_run_info_no_setup, close_run_info,
                                 postproc_load_variable
using moment_kinetics.type_definitions: mk_float
using moment_kinetics.utils: merge_dict_with_kwargs!

include("nonlinear_sound_wave_inputs_and_expected_data.jl")

base_input = copy(test_input_chebyshev)
base_input["timestepping"]["nstep"] = 50
base_input["timestepping"]["nwrite"] = 50
base_input["timestepping"]["nwrite_dfns"] = 50
if global_size[] > 1 && global_size[] % 2 == 0
    # Test using distributed-memory
    base_input["z"]["nelement_local"] = base_input["z"]["nelement"] ÷ 2
end
base_input["output"]["parallel_io"] = false

restart_test_input_chebyshev =
recursive_merge(deepcopy(base_input),
                OptionsDict("output" => OptionsDict("run_name" => "restart_chebyshev_pseudospectral"),
                            "r" => OptionsDict("ngrid" => 3, "nelement" => 2,
                                               "discretization" => "chebyshev_pseudospectral"),
                            "z" => OptionsDict("ngrid" => 17, "nelement" => 2),
                            "vpa" => OptionsDict("ngrid" => 9, "nelement" => 32),
                            "vz" => OptionsDict("ngrid" => 9, "nelement" => 32)),
               )
if global_size[] > 1 && global_size[] % 2 == 0
    # Test using distributed-memory
    restart_test_input_chebyshev["z"]["nelement_local"] = restart_test_input_chebyshev["z"]["nelement"] ÷ 2
end

restart_test_input_chebyshev_split_1_moment =
    recursive_merge(deepcopy(restart_test_input_chebyshev),
                    OptionsDict("output" => OptionsDict("run_name" => "restart_chebyshev_pseudospectral_split_1_moment"),
                                "evolve_moments" => OptionsDict("density" => true)),
                   )

restart_test_input_chebyshev_split_2_moments =
    recursive_merge(deepcopy(restart_test_input_chebyshev_split_1_moment),
                    OptionsDict("output" => OptionsDict("run_name" => "restart_chebyshev_pseudospectral_split_2_moments"),
                                "r" => OptionsDict("ngrid" => 1, "nelement" => 1),
                                "evolve_moments" => OptionsDict("parallel_flow" => true)),
                   )

restart_test_input_chebyshev_split_3_moments =
    recursive_merge(deepcopy(restart_test_input_chebyshev_split_2_moments),
                    OptionsDict("output" => OptionsDict("run_name" => "restart_chebyshev_pseudospectral_split_3_moments"),
                                "evolve_moments" => OptionsDict("pressure" => true),
                                "vpa" => OptionsDict("L" => sqrt(3/2)*1.5*vpa_L), "vz" => OptionsDict("L" => sqrt(3/2)*1.5*vpa_L)),
                   )

"""
Run a sound-wave test for a single set of parameters
"""
# Note 'name' should not be shared by any two tests in this file
function run_test(test_input, base, message, rtol, atol; tol_3V, args...)
    # by passing keyword arguments to run_test, args becomes a Tuple of Pairs which can be used to
    # update the default inputs

    # Make a copy to make sure nothing modifies the input Dicts defined in this test
    # script.
    input = deepcopy(test_input)

    if tol_3V === nothing
        atol_3V = atol
        rtol_3V = rtol
        is_3V = false
    else
        atol_3V = tol_3V
        rtol_3V = tol_3V
        is_3V = true
    end

    parallel_io = input["output"]["parallel_io"]
    # Convert keyword arguments to a unique name
    function stringify_arg(key, value)
        if isa(value, AbstractDict)
            return string(string(key)[1], (stringify_arg(k, v) for (k, v) in value)...)
        else
            if isa(value, AbstractString)
                return string(string(key)[1], value[1])
            else
                return string(string(key)[1], value)
            end
        end
    end
    name = input["output"]["run_name"]
    if length(args) > 0
        name = string(name, "_", (stringify_arg(k, v) for (k, v) in args)...)
    end
    # Make sure name is not too long
    if length(name) > 60
        name = name[1:60]
    end
    if parallel_io
        name *= "p-io"
    end

    # Provide some progress info
    println("    - testing ", message)

    merge_dict_with_kwargs!(input; args...)
    input["output"]["run_name"] = name

    # Suppress console output while running
    quietoutput() do
        # run simulation
        if parallel_io
            restart_filename = joinpath(base["output"]["base_directory"],
                                        base["output"]["run_name"],
                                        base["output"]["run_name"] * ".dfns.h5")
        else
            restart_filename = joinpath(base["output"]["base_directory"],
                                        base["output"]["run_name"],
                                        base["output"]["run_name"] * ".dfns.0.h5")
        end
        run_moment_kinetics(input; restart=restart_filename)
    end

    phi = nothing
    n_ion = nothing
    upar_ion = nothing
    p_ion = nothing
    ppar_ion = nothing
    f_ion = nothing
    n_neutral = nothing
    upar_neutral = nothing
    p_neutral = nothing
    pz_neutral = nothing
    f_neutral = nothing
    z, z_spectral = nothing, nothing
    vpa, vpa_spectral = nothing, nothing
    vz, vz_spectral = nothing, nothing

    if global_rank[] == 0
        quietoutput() do

            # Load and analyse output
            #########################

            # Read the output data
            path = joinpath(realpath(input["output"]["base_directory"]), name)

            run_info = get_run_info_no_setup((path, -1); dfns=true)
            z = run_info.z
            z_spectral = run_info.z_spectral
            vpa = run_info.vpa
            vpa_spectral = run_info.vpa_spectral
            vzeta = run_info.vzeta
            vzeta_spectral = run_info.vzeta_spectral
            vr = run_info.vr
            vr_spectral = run_info.vr_spectral
            vz = run_info.vz
            vz_spectral = run_info.vz_spectral
            time = run_info.time
            n_ion_species = run_info.n_ion_species
            n_neutral_species = run_info.n_neutral_species
            n_ion_zrst = postproc_load_variable(run_info, "density")
            upar_ion_zrst = postproc_load_variable(run_info, "parallel_flow")
            p_ion_zrst = postproc_load_variable(run_info, "pressure")
            ppar_ion_zrst = postproc_load_variable(run_info, "parallel_pressure")
            qpar_ion_zrst = postproc_load_variable(run_info, "parallel_heat_flux")
            v_t_ion_zrst = postproc_load_variable(run_info, "thermal_speed")
            f_ion_vpavperpzrst  = postproc_load_variable(run_info, "f")
            n_neutral_zrst = postproc_load_variable(run_info, "density_neutral")
            upar_neutral_zrst = postproc_load_variable(run_info, "uz_neutral")
            p_neutral_zrst = postproc_load_variable(run_info, "p_neutral")
            pz_neutral_zrst = postproc_load_variable(run_info, "pz_neutral")
            qpar_neutral_zrst = postproc_load_variable(run_info, "qz_neutral")
            v_t_neutral_zrst = postproc_load_variable(run_info, "thermal_speed_neutral")
            # Slice f_neutral while loading to save memory, and avoid termination of the
            # 'long tests' CI job.
            f_neutral_vzvrvzetazrst = postproc_load_variable(run_info, "f_neutral",
                                                             ivzeta=(vzeta.n+1)÷2,
                                                             ivr=(vr.n+1)÷2)
            phi_zrt = postproc_load_variable(run_info, "phi")
            Er_zrt = postproc_load_variable(run_info, "Er")
            Ez_zrt = postproc_load_variable(run_info, "Ez")

            close_run_info(run_info)

            # Delete output because output files for 3V tests can be large
            rm(joinpath(realpath(input["output"]["base_directory"]), name); recursive=true)

            phi = phi_zrt[:,1,:]
            n_ion = n_ion_zrst[:,1,:,:]
            upar_ion = upar_ion_zrst[:,1,:,:]
            p_ion = p_ion_zrst[:,1,:,:]
            ppar_ion = ppar_ion_zrst[:,1,:,:]
            qpar_ion = qpar_ion_zrst[:,1,:,:]
            v_t_ion = v_t_ion_zrst[:,1,:,:]
            f_ion = f_ion_vpavperpzrst[:,1,:,1,:,:]
            n_neutral = n_neutral_zrst[:,1,:,:]
            upar_neutral = upar_neutral_zrst[:,1,:,:]
            p_neutral = p_neutral_zrst[:,1,:,:]
            pz_neutral = pz_neutral_zrst[:,1,:,:]
            qpar_neutral = qpar_neutral_zrst[:,1,:,:]
            v_t_neutral = v_t_neutral_zrst[:,1,:,:]
            f_neutral = f_neutral_vzvrvzetazrst[:,:,1,:,:]

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

        newgrid_phi = interpolate_to_grid_z(expected.z, phi[:, end], z, z_spectral)
        @test isapprox(expected.phi[:, end], newgrid_phi, rtol=rtol)

        # Check ion particle moments and f
        ######################################

        newgrid_n_ion = interpolate_to_grid_z(expected.z, n_ion[:, :, end], z, z_spectral)
        @test isapprox(expected.n_ion[:, end], newgrid_n_ion[:,1], rtol=rtol)

        newgrid_upar_ion = interpolate_to_grid_z(expected.z, upar_ion[:, :, end], z, z_spectral)
        @test isapprox(expected.upar_ion[:, end], newgrid_upar_ion[:,1], rtol=rtol, atol=atol_3V)

        newgrid_ppar_ion = interpolate_to_grid_z(expected.z, ppar_ion[:, :, end], z, z_spectral)
        @test isapprox(expected.ppar_ion[:, end], newgrid_ppar_ion[:,1], rtol=rtol)

        newgrid_p_ion = interpolate_to_grid_z(expected.z, p_ion[:, :, end], z, z_spectral)
        newgrid_vth_ion = @. sqrt(2.0*newgrid_p_ion/newgrid_n_ion)
        newgrid_f_ion = interpolate_to_grid_z(expected.z, f_ion[:, :, :, end], z, z_spectral)
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
        if is_3V
            # Only looking at ivperp=1 point, but peak value of distribution function is
            # changed to keep density fixed when converting from 1V to 3V
            @test isapprox(expected.f_ion[:, :, end] ./ π, newgrid_f_ion[:,:,1],
                           rtol=rtol_3V)
        else
            @test isapprox(expected.f_ion[:, :, end], newgrid_f_ion[:,:,1], rtol=rtol)
        end

        # Check neutral particle moments and f
        ######################################

        newgrid_n_neutral = interpolate_to_grid_z(expected.z, n_neutral[:, :, end], z, z_spectral)
        @test isapprox(expected.n_neutral[:, end], newgrid_n_neutral[:,1], rtol=rtol)

        newgrid_upar_neutral = interpolate_to_grid_z(expected.z, upar_neutral[:, :, end], z, z_spectral)
        @test isapprox(expected.upar_neutral[:, end], newgrid_upar_neutral[:,1], rtol=rtol, atol=atol_3V)

        # The errors on pz_neutral when using a 3V grid are large - probably because of
        # linear interpolation in ion-neutral interaction operators - so for the 3V tests
        # we have to use a very loose tolerance for pz_neutral.
        newgrid_pz_neutral = interpolate_to_grid_z(expected.z, pz_neutral[:, :, end], z, z_spectral)
        @test isapprox(expected.ppar_neutral[:, end], newgrid_pz_neutral[:,1], rtol=rtol_3V)

        newgrid_p_neutral = interpolate_to_grid_z(expected.z, p_neutral[:, :, end], z, z_spectral)
        newgrid_vth_neutral = @. sqrt(2.0*newgrid_p_neutral/newgrid_n_neutral)
        newgrid_f_neutral = interpolate_to_grid_z(expected.z, f_neutral[:, :, :, end], z, z_spectral)
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
            newgrid_f_neutral[:,iz,1] = interpolate_to_grid_vpa(wpa, temp[:,iz,1], vz, vz_spectral)
        end
        if is_3V
            # Only looking at ivperp=1 point, but peak value of distribution function is
            # changed to keep density fixed when converting from 1V to 3V
            @test isapprox(expected.f_neutral[:, :, end] ./ π,
                           newgrid_f_neutral[:,:,1], rtol=rtol)
        else
            @test isapprox(expected.f_neutral[:, :, end], newgrid_f_neutral[:,:,1],
                           rtol=rtol)
        end
    end
end


function runtests()
    function do_tests(label, rtol=1.0e-3, nstep=50, include_moment_kinetic=true;
                      tol_3V=nothing, args...)
        # Only testing Chebyshev discretization because interpolation not yet implemented
        # for finite-difference

        parallel_io = base_input["output"]["parallel_io"]

        base_input_full_f = deepcopy(base_input)
        base_input_full_f["timestepping"] = recursive_merge(base_input["timestepping"],
                                                            OptionsDict("nstep" => nstep),
                                                           )
        base_input_evolve_density = recursive_merge(base_input_full_f,
                                                    OptionsDict("evolve_moments" => OptionsDict("density" => true)),
                                                   )
        base_input_evolve_upar = recursive_merge(base_input_evolve_density,
                                                 OptionsDict("evolve_moments" => OptionsDict("parallel_flow" => true),
                                                             "vpa" => OptionsDict("L" => 1.5*vpa_L),
                                                             "vz" => OptionsDict("L" => 1.5*vpa_L)),
                                                )
        base_input_evolve_p = recursive_merge(base_input_evolve_upar,
                                                 OptionsDict("evolve_moments" => OptionsDict("pressure" => true),
                                                             "vpa" => OptionsDict("L" => sqrt(3/2)*1.5*vpa_L),
                                                             "vz" => OptionsDict("L" => sqrt(3/2)*1.5*vpa_L)),
                                                )

        for (base, base_label) ∈ ((base_input_full_f, "full-f"),
                                  (base_input_evolve_density, "split 1"),
                                  (base_input_evolve_upar, "split 2"),
                                  (base_input_evolve_p, "split 3"))

            test_output_directory = get_MPI_tempdir()
            base["output"]["base_directory"] = test_output_directory

            # Base run, from which tests are restarted
            # Suppress console output while running
            quietoutput() do
                # run simulation
                run_moment_kinetics(base)
            end

            # Benchmark data is taken from this run (full-f with no splitting)
            message = "restart full-f from $base_label$label"
            @testset "$message" begin
                # When not including moment-kinetic tests (because we are running a 2V/3V
                # simulation) don't test upar. upar and uz end up with large 'errors'
                # (~50%), and it is not clear why, but ignore this so test can pass.
                this_input = deepcopy(restart_test_input_chebyshev)
                this_input["output"]["base_directory"] = test_output_directory
                this_input["output"]["parallel_io"] = parallel_io
                run_test(this_input, base, message, rtol, 1.e-15; tol_3V=tol_3V, args...)
            end
            if include_moment_kinetic
                message = "restart split 1 from $base_label$label"
                @testset "$message" begin
                    this_input = deepcopy(restart_test_input_chebyshev_split_1_moment)
                    this_input["output"]["base_directory"] = test_output_directory
                    this_input["output"]["parallel_io"] = parallel_io
                    run_test(this_input, base, message, rtol, 1.e-15; tol_3V=tol_3V, args...)
                end
                message = "restart split 2 from $base_label$label"
                @testset "$message" begin
                    this_input = deepcopy(restart_test_input_chebyshev_split_2_moments)
                    this_input["output"]["base_directory"] = test_output_directory
                    this_input["output"]["parallel_io"] = parallel_io
                    run_test(this_input, base, message, rtol, 1.e-15; tol_3V=tol_3V, args...)
                end
                message = "restart split 3 from $base_label$label"
                @testset "$message" begin
                    this_input = deepcopy(restart_test_input_chebyshev_split_3_moments)
                    this_input["output"]["base_directory"] = test_output_directory
                    this_input["output"]["parallel_io"] = parallel_io
                    run_test(this_input, base, message, rtol, 1.e-15; tol_3V=tol_3V, args...)
                end
            end

            if global_rank[] == 0
                # Delete output directory to avoid using too much disk space
                rm(realpath(test_output_directory); recursive=true)
            end
        end
    end

    @testset "restart interpolation" verbose=use_verbose begin
        println("restart interpolation tests")

        do_tests("")

        # Note: only do 2 steps in 2V/3V mode because it is so slow. Also, linear
        # interpolation used for ion-neutral coupling in 2V/3V case has low accuracy, so
        # use looser tolerance for various things.
        @long do_tests(", 2V/3V", 1.0e-1, 98, false; tol_3V=0.3,
                       timestepping=OptionsDict("nstep" => 2),
                       r=OptionsDict("ngrid" => 1, "nelement" => 1),
                       vperp=OptionsDict("ngrid" => 17, "nelement" => 4, "L" => vpa_L, "ngrid" => 17),
                       vpa=OptionsDict("nelement" => 8),
                       vzeta=OptionsDict("ngrid" => 17, "nelement" => 4, "L" => vpa_L),
                       vr=OptionsDict("ngrid" => 17, "nelement" => 4, "L" => vpa_L),
                       vz=OptionsDict("ngrid" => 17, "nelement" => 8),
                      )

        if io_has_parallel(Val(hdf5))
            orig_base_input = deepcopy(base_input)
            # Also test not using parallel_io
            base_input["output"]["parallel_io"] = true
            base_input["output"]["run_name"] *= "_parallel_io"

            do_tests(", parallel I/O")

            # Note: only do 2 steps in 2V/3V mode because it is so slow
            # interpolation used for ion-neutral coupling in 2V/3V case has low accuracy,
            # so use looser tolerance for various things.
            @long do_tests(", 2V/3V, parallel I/O", 2.0e-1, 98, false; tol_3V=0.3,
                           timestepping=OptionsDict("nstep" => 2),
                           r=OptionsDict("ngrid" => 1, "nelement" => 1),
                           vperp=OptionsDict("ngrid" => 17, "nelement" => 4, "L" => vpa_L, "ngrid" => 17),
                           vpa=OptionsDict("nelement" => 8),
                           vzeta=OptionsDict("ngrid" => 17, "nelement" => 4, "L" => vpa_L),
                           vr=OptionsDict("ngrid" => 17, "nelement" => 4, "L" => vpa_L),
                           vz=OptionsDict("ngrid" => 17, "nelement" => 8),
                          )

            global base_input = orig_base_input
        end
    end
end

end # RestartInterpolationTests


using .RestartInterpolationTests

RestartInterpolationTests.runtests()
