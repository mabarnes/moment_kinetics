module RegriddingTests
include("setup.jl")

using moment_kinetics: setup_moment_kinetics
using moment_kinetics.load_data: regrid_ion_pdf, regrid_electron_pdf,
                                 regrid_neutral_pdf

const input_2V = OptionsDict("output" => OptionsDict("run_name" => "regridding_test"),
                             "composition" => OptionsDict("n_ion_species" => 1,
                                                          "n_neutral_species" => 0,
                                                          "electron_physics" => "boltzmann_electron_response"),
                             "vpa" => OptionsDict("ngrid" => 5,
                                                  "L" => 10.0,
                                                  "nelement" => 8,
                                                 ),
                             "vperp" => OptionsDict("ngrid" => 5,
                                                    "nelement" => 4,
                                                    "L" => 5.0,
                                                   ),
                             "vzeta" => OptionsDict("ngrid" => 5,
                                                    "L" => 10.0,
                                                    "nelement" => 9,
                                                   ),
                             "vr" => OptionsDict("ngrid" => 5,
                                                 "L" => 10.0,
                                                 "nelement" => 10,
                                                ),
                             "vz" => OptionsDict("ngrid" => 5,
                                                 "L" => 10.0,
                                                 "nelement" => 11,
                                                ),
                             "z" => OptionsDict("ngrid" => 1,
                                                "nelement" => 1,
                                               ),
                             "r" => OptionsDict("ngrid" => 1,
                                                "nelement" => 1,
                                               ),
                            )

const input_1V = deepcopy(input_2V)
input_1V["vperp"]["ngrid"] = 1
input_1V["vperp"]["nelement"] = 1
input_1V["vzeta"]["ngrid"] = 1
input_1V["vzeta"]["nelement"] = 1
input_1V["vr"]["ngrid"] = 1
input_1V["vr"]["nelement"] = 1

const n = 1.5
const u = 0.1
const vth_2V = 1.2
const vth_1V = 1.2 / sqrt(3.0)

function get_Maxwellian(vperp, vpa, evolve_density, evolve_upar, evolve_p, is_1V,
                        unit_vth_perp; is_electron=false)
    if evolve_upar && evolve_p
        if is_1V
            vpa_unnorm = reshape(vpa.grid, vpa.n, 1) .* vth_1V .+ u
        else
            vpa_unnorm = reshape(vpa.grid, vpa.n, 1) .* vth_2V .+ u
            vperp_unnorm = reshape(vperp.grid, 1, vperp.n) .* vth_2V
        end
    elseif evolve_upar
        vpa_unnorm = reshape(vpa.grid, vpa.n, 1) .+ u
        vperp_unnorm = reshape(vperp.grid, 1, vperp.n)
    elseif evolve_p
        if is_1V
            vpa_unnorm = reshape(vpa.grid, vpa.n, 1) .* vth_1V
        else
            vpa_unnorm = reshape(vpa.grid, vpa.n, 1) .* vth_2V
            vperp_unnorm = reshape(vperp.grid, 1, vperp.n) .* vth_2V
        end
    else
        vpa_unnorm = reshape(vpa.grid, vpa.n, 1)
        vperp_unnorm = reshape(vperp.grid, 1, vperp.n)
    end
    if is_1V
        f = @. 1.0 / sqrt(π) * exp(-(vpa_unnorm - u)^2 / (vth_1V * sqrt(3.0))^2)
        if !evolve_p
            f ./= (vth_1V * sqrt(3.0))
        else
            f ./= sqrt(3.0)
        end
    else
        if unit_vth_perp
            f = @. 1.0 / π^1.5 * exp(-((vpa_unnorm - u)^2 / vth_2V^2 + vperp_unnorm^2))
        else
            f = @. 1.0 / π^1.5 * exp(-((vpa_unnorm - u)^2 + vperp_unnorm^2) / vth_2V^2)
        end
        if !evolve_p
            if unit_vth_perp
                f ./= vth_2V
            else
                f ./= vth_2V^3
            end
        end
    end
    if !evolve_density
        f .*= n
    end

    # Make f the expected shape for moment_kinetics
    if is_electron
        # electron
        f = reshape(f, size(f)..., 1, 1)
    else
        # ion
        f = reshape(f, size(f)..., 1, 1, 1)
    end

    return f
end

function get_Maxwellian_neutral(vzeta, vr, vz, evolve_density, evolve_upar, evolve_p,
                                is_1V, unit_vth_perp)
    if evolve_upar && evolve_p
        if is_1V
            vz_unnorm = reshape(vz.grid, vz.n, 1, 1) .* vth_1V .+ u
        else
            vz_unnorm = reshape(vz.grid, vz.n, 1, 1) .* vth_2V .+ u
            vzeta_unnorm = reshape(vzeta.grid, 1, 1, vzeta.n) .* vth_2V
            vr_unnorm = reshape(vr.grid, 1, vr.n, 1) .* vth_2V
        end
    elseif evolve_upar
        vz_unnorm = reshape(vz.grid, vz.n, 1, 1) .+ u
        vr_unnorm = reshape(vr.grid, 1, vr.n, 1)
        vzeta_unnorm = reshape(vzeta.grid, 1, 1, vzeta.n)
    elseif evolve_p
        if is_1V
            vz_unnorm = reshape(vz.grid, vz.n, 1, 1) .* vth_1V
        else
            vz_unnorm = reshape(vz.grid, vz.n, 1, 1) .* vth_2V
            vzeta_unnorm = reshape(vzeta.grid, 1, 1, vzeta.n) .* vth_2V
            vr_unnorm = reshape(vr.grid, 1, vr.n, 1) .* vth_2V
        end
    else
        vz_unnorm = reshape(vz.grid, vz.n, 1, 1)
        vzeta_unnorm = reshape(vzeta.grid, 1, 1, vzeta.n)
        vr_unnorm = reshape(vr.grid, 1, vr.n, 1)
    end
    if is_1V
        f = @. 1.0 / sqrt(π) * exp(-(vz_unnorm - u)^2 / (vth_1V * sqrt(3.0))^2)
        if !evolve_p
            f ./= (vth_1V * sqrt(3.0))
        else
            f ./= sqrt(3.0)
        end
    else
        if unit_vth_perp
            f = @. 1.0 / π^1.5 * exp(-((vz_unnorm - u)^2 / vth_2V^2 + vzeta_unnorm^2 + vr_unnorm^2))
        else
            f = @. 1.0 / π^1.5 * exp(-((vz_unnorm - u)^2 + vzeta_unnorm^2 + vr_unnorm^2) / vth_2V^2)
        end
        if !evolve_p
            if unit_vth_perp
                f ./= vth_2V
            else
                f ./= vth_2V^3
            end
        end
    end
    if !evolve_density
        f .*= n
    end

    # Make f the expected shape for moment_kinetics
    f = reshape(f, size(f)..., 1, 1, 1)

    return f
end

function all_tests(; rtol=1.0e-2, atol=5.0e-4)
    mk_state_2V = nothing
    mk_state_1V = nothing

    quietoutput() do
        mk_state_2V = setup_moment_kinetics(input_2V; write_output=false)
        mk_state_1V = setup_moment_kinetics(input_1V; write_output=false)
    end

    pdf_2V, scratch_2V, scratch_implicit_2V, scratch_electron_2V, t_params_2V, vz_2V,
    vr_2V, vzeta_2V, vpa_2V, vperp_2V, gyrophase_2V, z_2V, r_2V, moments_2V, fields_2V,
    spectral_objects_2V, advect_objects_2V, composition_2V, collisions_2V, geometry_2V,
    gyroavs_2V, boundaries_2V, external_source_settings_2V, num_diss_params_2V,
    nl_solver_params_2V, advance_2V, advance_implicit_2V, fp_arrays_2V, scratch_dummy_2V,
    manufactured_source_list_2V, ascii_io_2V, io_moments_2V, io_dfns_2V = mk_state_2V

    pdf_1V, scratch_1V, scratch_implicit_1V, scratch_electron_1V, t_params_1V, vz_1V,
    vr_1V, vzeta_1V, vpa_1V, vperp_1V, gyrophase_1V, z_1V, r_1V, moments_1V, fields_1V,
    spectral_objects_1V, advect_objects_1V, composition_1V, collisions_1V, geometry_1V,
    gyroavs_1V, boundaries_1V, external_source_settings_1V, num_diss_params_1V,
    nl_solver_params_1V, advance_1V, advance_implicit_1V, fp_arrays_1V, scratch_dummy_1V,
    manufactured_source_list_1V, ascii_io_1V, io_moments_1V, io_dfns_1V = mk_state_1V

    # r and z are the same for 1V and 2V
    r = r_2V
    r_spectral = spectral_objects_2V.r_spectral
    z = z_2V
    z_spectral = spectral_objects_2V.z_spectral

    @testset "1V $old_is_1V->$new_is_1V, evolve_density $old_evolve_density->$new_evolve_density, evolve_upar $old_evolve_upar->$new_evolve_upar, evolve_p $old_evolve_p->$new_evolve_p" for
            old_is_1V ∈ (true, false), new_is_1V ∈ (true, false),
            old_evolve_density ∈ (false, true), new_evolve_density ∈ (false, true),
            old_evolve_upar ∈ (false, true), new_evolve_upar ∈ (false, true),
            old_evolve_p ∈ (false, true), new_evolve_p ∈ (false, true),
            force_interp ∈ (false, true) # force_interp sets `interpolation_needed` entries to `true` in the `regrid_*_pdf()` functions even when the grids are the same, to check for errors in more code branches.

        if old_is_1V
            old_vperp = vperp_1V
            old_vperp_spectral = spectral_objects_1V.vperp_spectral
            old_vpa = vpa_1V
            old_vpa_spectral = spectral_objects_1V.vpa_spectral
            old_vzeta = vzeta_1V
            old_vzeta_spectral = spectral_objects_1V.vzeta_spectral
            old_vr = vr_1V
            old_vr_spectral = spectral_objects_1V.vr_spectral
            old_vz = vz_1V
            old_vz_spectral = spectral_objects_1V.vz_spectral
            # When restarting, vth is modified from 1V to 2V (if necessary) after the
            # distribution functions are interpolated, so we should use the 'old' value
            # here.
            vth = vth_1V
        else
            old_vperp = vperp_2V
            old_vperp_spectral = spectral_objects_2V.vperp_spectral
            old_vpa = vpa_2V
            old_vpa_spectral = spectral_objects_2V.vpa_spectral
            old_vzeta = vzeta_2V
            old_vzeta_spectral = spectral_objects_2V.vzeta_spectral
            old_vr = vr_2V
            old_vr_spectral = spectral_objects_2V.vpa_spectral
            old_vz = vz_2V
            old_vz_spectral = spectral_objects_2V.vz_spectral
            # When restarting, vth is modified from 1V to 2V (if necessary) after the
            # distribution functions are interpolated, so we should use the 'old' value
            # here.
            vth = vth_2V
        end

        if new_is_1V
            new_vperp = vperp_1V
            new_vpa = vpa_1V
            new_vzeta = vzeta_1V
            new_vr = vr_1V
            new_vz = vz_1V
        else
            new_vperp = vperp_2V
            new_vpa = vpa_2V
            new_vzeta = vzeta_2V
            new_vr = vr_2V
            new_vz = vz_2V
        end

        moments = (ion=(dens=fill(n, 1, 1, 1), upar=fill(u, 1, 1, 1), vth=fill(vth, 1, 1, 1)),
                   electron=(dens=fill(n, 1, 1), upar=fill(u, 1, 1), vth=fill(vth, 1, 1)),
                   neutral=(dens=fill(n, 1, 1), uz=fill(u, 1, 1), vth=fill(vth, 1, 1)),
                   evolve_density=new_evolve_density, evolve_upar=new_evolve_upar,
                   evolve_p=new_evolve_p)

        @testset "ion" begin
            old_f = get_Maxwellian(old_vperp, old_vpa, old_evolve_density, old_evolve_upar,
                                   old_evolve_p, old_is_1V, false)

            unit_vth_perp = !new_evolve_p && (old_is_1V && !new_is_1V)
            expected_new_f = get_Maxwellian(new_vperp, new_vpa, new_evolve_density,
                                            new_evolve_upar, new_evolve_p, new_is_1V,
                                            unit_vth_perp)

            interp_new_f = regrid_ion_pdf(old_f,
                                          (r=r, z=z, vperp=new_vperp, vpa=new_vpa),
                                          (r=r, r_spectral=r_spectral, z=z,
                                           z_spectral=z_spectral, vperp=old_vperp,
                                           vperp_spectral=old_vperp_spectral, vpa=old_vpa,
                                           vpa_spectral=old_vpa_spectral),
                                          Dict("r"=>false, "z"=>false,
                                               "vperp"=>force_interp || (old_is_1V != new_is_1V),
                                               "vpa"=>force_interp),
                                          moments,
                                          old_evolve_density, old_evolve_upar, old_evolve_p
                                         )

            @test elementwise_isapprox(interp_new_f, expected_new_f; rtol=rtol, atol=atol)
        end

        @testset "electron" begin
            old_f = get_Maxwellian(old_vperp, old_vpa, old_evolve_density, old_evolve_upar,
                                   old_evolve_p, old_is_1V, false, is_electron=true)

            unit_vth_perp = !new_evolve_p && (old_is_1V && !new_is_1V)
            expected_new_f = get_Maxwellian(new_vperp, new_vpa, new_evolve_density,
                                            new_evolve_upar, new_evolve_p, new_is_1V,
                                            unit_vth_perp, is_electron=true)

            interp_new_f = regrid_electron_pdf(old_f,
                                               (r=r, z=z, vperp=new_vperp, vpa=new_vpa),
                                               (r=r, r_spectral=r_spectral, z=z,
                                                z_spectral=z_spectral, vperp=old_vperp,
                                                vperp_spectral=old_vperp_spectral, vpa=old_vpa,
                                                vpa_spectral=old_vpa_spectral),
                                               Dict("r"=>false, "z"=>false,
                                                    "vperp"=>force_interp || (old_is_1V != new_is_1V),
                                                    "vpa"=>force_interp),
                                               moments,
                                               old_evolve_density, old_evolve_upar, old_evolve_p;
                                               allow_unsupported_options=true
                                              )

            @test elementwise_isapprox(interp_new_f, expected_new_f; rtol=rtol, atol=atol)
        end

        @testset "neutral" begin
            old_f = get_Maxwellian_neutral(old_vzeta, old_vr, old_vz, old_evolve_density,
                                           old_evolve_upar, old_evolve_p, old_is_1V,
                                           false)

            unit_vth_perp = !new_evolve_p && (old_is_1V && !new_is_1V)
            expected_new_f = get_Maxwellian_neutral(new_vzeta, new_vr, new_vz,
                                                    new_evolve_density, new_evolve_upar,
                                                    new_evolve_p, new_is_1V,
                                                    unit_vth_perp)

            interp_new_f = regrid_neutral_pdf(old_f,
                                              (r=r, z=z, vzeta=new_vzeta, vr=new_vr, vz=new_vz),
                                              (r=r, r_spectral=r_spectral,
                                               z=z, z_spectral=z_spectral,
                                               vzeta=old_vzeta, vzeta_spectral=old_vzeta_spectral,
                                               vr=old_vr, vr_spectral=old_vr_spectral,
                                               vz=old_vz, vz_spectral=old_vz_spectral),
                                              Dict("r"=>false, "z"=>false,
                                                   "vzeta"=>force_interp || (old_is_1V != new_is_1V),
                                                   "vr"=>force_interp || (old_is_1V != new_is_1V),
                                                   "vz"=>force_interp),
                                              moments,
                                              old_evolve_density, old_evolve_upar, old_evolve_p
                                             )

            @test elementwise_isapprox(interp_new_f, expected_new_f; rtol=rtol, atol=atol)
        end
    end

    return nothing
end

function runtests()
    @testset "regridding" begin
        println("regridding tests")
        all_tests()
    end
end

end # RegriddingTests

using .RegriddingTests
RegriddingTests.runtests()
