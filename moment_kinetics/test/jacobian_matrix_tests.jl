module JacobianMatrixTests

# Tests for construction of Jacobian matrices used for preconditioning

include("setup.jl")

using moment_kinetics: setup_moment_kinetics, cleanup_moment_kinetics!
using moment_kinetics.analysis: vpagrid_to_vpa
using moment_kinetics.array_allocation: allocate_shared_float
using moment_kinetics.boundary_conditions: enforce_v_boundary_condition_local!,
                                           enforce_vperp_boundary_condition!,
                                           skip_f_electron_bc_points_in_Jacobian,
                                           skip_f_electron_bc_points_in_Jacobian_v_solve,
                                           skip_f_electron_bc_points_in_Jacobian_z_solve
using moment_kinetics.calculus: derivative!, second_derivative!, integral
using moment_kinetics.communication
using moment_kinetics.communication: _anyzv_subblock_synchronize
using moment_kinetics.derivatives: derivative_z_anyzv!, derivative_z_pdf_vpavperpz!
using moment_kinetics.electron_fluid_equations: calculate_electron_moments_no_r!,
                                                electron_energy_equation_no_r!,
                                                get_electron_energy_equation_term
using moment_kinetics.electron_kinetic_equation: get_electron_sub_terms,
                                                 get_electron_sub_terms_z_only_Jacobian,
                                                 get_electron_sub_terms_v_only_Jacobian,
                                                 add_contribution_from_pdf_term!,
                                                 get_contribution_from_electron_pdf_term,
                                                 add_dissipation_term!,
                                                 get_electron_dissipation_term,
                                                 get_ion_dt_forcing_of_electron_p_term,
                                                 electron_kinetic_equation_euler_update!,
                                                 enforce_boundary_condition_on_electron_pdf!,
                                                 fill_electron_kinetic_equation_Jacobian!,
                                                 fill_electron_kinetic_equation_v_only_Jacobian!,
                                                 fill_electron_kinetic_equation_z_only_Jacobian_f!,
                                                 fill_electron_kinetic_equation_z_only_Jacobian_p!,
                                                 add_wall_boundary_condition_to_Jacobian!,
                                                 zero_z_boundary_condition_points
using moment_kinetics.electron_vpa_advection: electron_vpa_advection!,
                                              update_electron_speed_vpa!,
                                              get_electron_vpa_advection_term
using moment_kinetics.electron_z_advection: electron_z_advection!,
                                            update_electron_speed_z!,
                                            get_electron_z_advection_term
using moment_kinetics.external_sources: total_external_electron_sources!,
                                        get_total_external_electron_source_term
using moment_kinetics.jacobian_matrices
using moment_kinetics.krook_collisions: electron_krook_collisions!,
                                        get_electron_krook_collisions_term
using moment_kinetics.looping
using moment_kinetics.moment_constraints: electron_implicit_constraint_forcing!,
                                          get_electron_implicit_constraint_forcing_term,
                                          hard_force_moment_constraints!
using moment_kinetics.timer_utils: reset_mk_timers!
using moment_kinetics.type_definitions: mk_float
using moment_kinetics.velocity_moments: calculate_electron_moment_derivatives_no_r!

using LinearAlgebra
using moment_kinetics.StatsBase

# Small parameter used to create perturbations to test Jacobian against
epsilon = 1.0e-6
test_wavenumber = 2.0
dt = 0.2969848480983499
ion_dt = 7.071067811865475e-7
ir = 1
zero = 1.0e-14

# Test input uses `z_bc = "constant"`, which is not a very physically useful option, but
# is useful for testing because:
# * `z_bc = "wall"` would introduce discontinuities in the distribution function which
#   might reduce accuracy and so make it harder to see whether errors are due to a mistake
#   in the matrix construction or just due to discretisation error
# * For `z_bc = "periodic"`, the Jacobian matrices (by design) do not account for the
#   periodicity. This should be fine when they are used as preconditioners, but does
#   introduce errors at the periodic boundaries which would complicate testing.
test_input = OptionsDict("output" => OptionsDict("run_name" => "jacobian_matrix"),
                         "composition" => OptionsDict("n_ion_species" => 1,
                                                      "n_neutral_species" => 1,
                                                      "electron_physics" => "kinetic_electrons",
                                                      "recycling_fraction" => 0.5,
                                                      "T_e" => 0.3333333333333333,
                                                      "T_wall" => 0.1),
                         "evolve_moments" => OptionsDict("density" => true,
                                                         "parallel_flow" => true,
                                                         "pressure" => true,
                                                         "moments_conservation" => true),
                         "ion_species_1" => OptionsDict("initial_density" => 1.0,
                                                        "initial_temperature" => 0.3333333333333333),
                         "z_IC_ion_species_1" => OptionsDict("initialization_option" => "sinusoid",
                                                             "density_amplitude" => 0.1,
                                                             "density_phase" => 3.141592653589793,
                                                             "upar_amplitude" => 0.14142135623730953,
                                                             "upar_phase" => 3.141592653589793,
                                                             "temperature_amplitude" => 0.1,
                                                             "temperature_phase" => 3.141592653589793),
                         "vpa_IC_ion_species_1" => OptionsDict("initialization_option" => "gaussian",
                                                               "density_amplitude" => 1.0,
                                                               "density_phase" => 0.0,
                                                               "upar_amplitude" => 0.0,
                                                               "upar_phase" => 0.0,
                                                               "temperature_amplitude" => 0.0,
                                                               "temperature_phase" => 0.0),
                         "neutral_species_1" => OptionsDict("initial_density" => 1.0,
                                                            "initial_temperature" => 0.3333333333333333),
                         "z_IC_neutral_species_1" => OptionsDict("initialization_option" => "sinusoid",
                                                                 "density_amplitude" => 0.001,
                                                                 "density_phase" => 3.141592653589793,
                                                                 "upar_amplitude" => 0.0,
                                                                 "upar_phase" => 3.141592653589793,
                                                                 "temperature_amplitude" => 0.0,
                                                                 "temperature_phase" => 3.141592653589793),
                         "vz_IC_neutral_species_1" => OptionsDict("initialization_option" => "gaussian",
                                                                  "density_amplitude" => 1.0,
                                                                  "density_phase" => 0.0,
                                                                  "upar_amplitude" => 0.0,
                                                                  "upar_phase" => 0.0,
                                                                  "temperature_amplitude" => 0.0,
                                                                  "temperature_phase" => 0.0),
                         "reactions" => OptionsDict("charge_exchange_frequency" => 1.0606601717798214,
                                                    "ionization_frequency" => 0.0),
                         "r" => OptionsDict("ngrid" => 1,
                                            "nelement" => 1),
                         "z" => OptionsDict("ngrid" => 9,
                                            "nelement" => 16,
                                            "bc" => "constant",
                                            "discretization" => "gausslegendre_pseudospectral"),
                         "vpa" => OptionsDict("ngrid" => 6,
                                              "nelement" => 31,
                                              "L" => 20.784609690826528,
                                              "bc" => "zero",
                                              "discretization" => "gausslegendre_pseudospectral",
                                              "element_spacing_option" => "coarse_tails8.660254037844386"),
                         "vz" => OptionsDict("ngrid" => 6,
                                             "nelement" => 31,
                                             "L" => 20.784609690826528,
                                             "bc" => "zero",
                                             "discretization" => "gausslegendre_pseudospectral",
                                             "element_spacing_option" => "coarse_tails8.660254037844386"),
                         "timestepping" => OptionsDict("type" => "KennedyCarpenterARK324",
                                                       "kinetic_electron_solver" => "implicit_p_implicit_pseudotimestep",
                                                       "kinetic_ion_solver" => "full_explicit_ion_advance",
                                                       "nstep" => 1,
                                                       "dt" => ion_dt,
                                                       "minimum_dt" => 7.071067811865474e-8,
                                                       "rtol" => 0.0001,
                                                       "max_increase_factor_near_last_fail" => 1.001,
                                                       "last_fail_proximity_factor" => 1.1,
                                                       "max_increase_factor" => 1.05,
                                                       "nwrite" => 10000,
                                                       "nwrite_dfns" => 10000,
                                                       "steady_state_residual" => true,
                                                       "converged_residual_value" => 0.0014142135623730952),
                         "electron_timestepping" => OptionsDict("nstep" => 1,
                                                                "dt" => dt,
                                                                "maximum_dt" => 0.7071067811865475,
                                                                "nwrite" => 10000,
                                                                "nwrite_dfns" => 100000,
                                                                "type" => "Fekete4(3)",
                                                                "rtol" => 1.0e-6,
                                                                "atol" => 1.0e-14,
                                                                "minimum_dt" => 7.071067811865475e-11,
                                                                "initialization_residual_value" => 2.5,
                                                                "converged_residual_value" => 0.014142135623730952,
                                                                "constraint_forcing_rate" => 3.282389678267954,
                                                                "include_wall_bc_in_preconditioner" => true),
                         "nonlinear_solver" => OptionsDict("nonlinear_max_iterations" => 100,
                                                           "rtol" => 1.0e-5,
                                                           "atol" => 1.0e-15,
                                                           "preconditioner_update_interval" => 1),
                         "ion_numerical_dissipation" => OptionsDict("vpa_dissipation_coefficient" => 4.242640687119286,
                                                                    "force_minimum_pdf_value" => 0.0),
                         "electron_numerical_dissipation" => OptionsDict("vpa_dissipation_coefficient" => 8.485281374238571,
                                                                         "force_minimum_pdf_value" => 0.0),
                         "neutral_numerical_dissipation" => OptionsDict("vz_dissipation_coefficient" => 0.42426406871192857,
                                                                        "force_minimum_pdf_value" => 0.0),
                         "ion_source_1" => OptionsDict("active" => true,
                                                       "z_profile" => "gaussian",
                                                       "z_width" => 0.125,
                                                       "source_strength" => 0.14142135623730953,
                                                       "source_T" => 2.0),
                         "krook_collisions" => OptionsDict("use_krook" => true),
                        )

function get_mk_state(test_input)
    # Reset timers in case there was a previous run which did not clean them up.
    reset_mk_timers!()

    mk_state = nothing
    quietoutput() do
        mk_state = setup_moment_kinetics(test_input; skip_electron_solve=true)
    end
    return mk_state
end
function cleanup_mk_state!(args...)
    quietoutput() do
        cleanup_moment_kinetics!(args...)
    end
    return nothing
end

function generate_norm_factor(perturbed_residual::AbstractArray{mk_float,3})
    # half-width of the window for moving average
    w = 3
    norm_factor_unsmoothed = mean(abs.(perturbed_residual); dims=3)
    # Smooth the 'norm_factor' with a moving average to avoid problems due to places where
    # norm_factor happens to be (almost) zero
    norm_factor = similar(norm_factor_unsmoothed)
    for i ∈ 1:w
        norm_factor[i,1,1,1] = mean(norm_factor_unsmoothed[1:i+w,1,1,1])
    end
    for i ∈ w+1:size(perturbed_residual, 1)-w
        norm_factor[i,1,1,1] = mean(norm_factor_unsmoothed[i-w:i+w,1,1,1])
    end
    for i ∈ 1:w
        norm_factor[end+1-i,1,1,1] = mean(norm_factor_unsmoothed[end+1-i-w:end,1,1,1])
    end
    return norm_factor
end
function generate_norm_factor(perturbed_residual::AbstractArray{mk_float,1})
    norm_factor_unsmoothed = mean(abs.(perturbed_residual); dims=1)
end

function test_get_pdf_term(test_input::AbstractDict, label::String, get_term::Function,
                           rhs_func!::Function, rtol::mk_float)
    test_input = deepcopy(test_input)
    test_input["output"]["run_name"] *= "_$label"
    println("    - $label")

    @testset "$label" begin
        # Suppress console output while running
        pdf, scratch, scratch_implicit, scratch_electron, t_params, vz, vr, vzeta, vpa,
            vperp, gyrophase, z, r, moments, fields, spectral_objects, advection_structs,
            composition, collisions, geometry, gyroavs, boundary_distributions,
            external_source_settings, num_diss_params, nl_solver_params, advance,
            advance_implicit, fp_arrays, scratch_dummy, manufactured_source_list,
            ascii_io, io_moments, io_dfns = get_mk_state(test_input)

        dens = @view moments.electron.dens[:,ir]
        ddens_dz = @view moments.electron.ddens_dz[:,ir]
        upar = @view moments.electron.upar[:,ir]
        dupar_dz = @view moments.electron.dupar_dz[:,ir]
        p = @view moments.electron.p[:,ir]
        dp_dz = @view moments.electron.dp_dz[:,ir]
        vth = @view moments.electron.vth[:,ir]
        dvth_dz = @view moments.electron.dvth_dz[:,ir]
        qpar = @view moments.electron.qpar[:,ir]
        dqpar_dz = @view moments.electron.dqpar_dz[:,ir]
        ion_dens = @view moments.ion.dens[:,ir]
        ion_upar = @view moments.ion.upar[:,ir]
        z_spectral = spectral_objects.z_spectral
        vperp_spectral = spectral_objects.vperp_spectral
        vpa_spectral = spectral_objects.vpa_spectral
        z_advect = advection_structs.z_advect
        vpa_advect = advection_structs.vpa_advect
        vpa_speed = @view vpa_advect[1].speed[:,:,:,ir]
        me = composition.me_over_mi

        delta_p = allocate_shared_float(z)
        p_amplitude = epsilon * maximum(p)
        f = @view pdf.electron.norm[:,:,:,ir]

        @begin_r_anyzv_region()

        buffer_1 = @view scratch_dummy.buffer_rs_1[ir,1]
        buffer_2 = @view scratch_dummy.buffer_rs_2[ir,1]
        buffer_3 = @view scratch_dummy.buffer_rs_3[ir,1]
        buffer_4 = @view scratch_dummy.buffer_rs_4[ir,1]

        # Reconstruct w_∥^3 moment of g_e from already-calculated qpar
        third_moment = scratch_dummy.buffer_z_1
        dthird_moment_dz = scratch_dummy.buffer_z_2
        @begin_anyzv_z_region()
        @loop_z iz begin
            third_moment[iz] = qpar[iz] / p[iz] / vth[iz]
        end
        derivative_z_anyzv!(dthird_moment_dz, third_moment, buffer_1, buffer_2, buffer_3,
                            buffer_4, z_spectral, z)

        @begin_anyzv_region()
        @anyzv_serial_region begin
            @. delta_p = p_amplitude * sin(2.0*π*test_wavenumber*z.grid/z.L)

            # Make sure initial condition has some z-variation. As f is 'moment kinetic' this
            # means f must have a non-Maxwellian part that varies in z.
            f .*= 1.0 .+ 1.0e-4 .* reshape(vpa.grid.^3, vpa.n, 1, 1) .* reshape(sin.(2.0.*π.*z.grid./z.L), 1, 1, z.n)
        end
        # Ensure initial electron distribution function obeys constraints
        hard_force_moment_constraints!(reshape(f, vpa.n, vperp.n, z.n, 1), moments, vpa, vperp)
        @begin_r_anyzv_region()
        delta_f = allocate_shared_float(vpa, vperp, z)
        f_amplitude = epsilon * maximum(f)
        # Use exp(sin()) in vpa so that perturbation does not have any symmetry that makes
        # low-order moments vanish exactly.
        @begin_anyzv_region()
        @anyzv_serial_region begin
            delta_f .= f_amplitude .*
                       reshape(sin.(2.0.*π.*test_wavenumber.*z.grid./z.L), 1, 1, z.n) .*
                       reshape(exp.(sin.(2.0.*π.*test_wavenumber.*vpa.grid./vpa.L)) .- 1.0, vpa.n, 1, 1) .*
                       f
        end

        if label == "electron_krook_collisions"
            # The actual (electron) upar will be updated and set equal to the ion upar by
            # `calculate_electron_moments_no_r!()`. For this test, we want to artificially
            # keep a difference between electron and ion upar, so use upar_test (which is
            # copied here before ion_upar is modified) in place of the usual upar array.
            upar_test = allocate_shared_float(z)
            @begin_anyzv_region()
            @anyzv_serial_region begin
                upar_test .= @view moments.electron.upar[:,ir]
            end

            # Modify ion_upar to make sure it is different from upar_electron so that the
            # term proportional to (u_i-u_e) gets tested in case it is ever needed.
            @. ion_upar += sin(4.0*π*test_wavenumber*z.grid/z.L)
        else
            upar_test = upar
        end

        pdf_size = length(f)
        p_size = length(p)
        total_size = pdf_size + p_size

        z_speed = @view z_advect[1].speed[:,:,:,ir]

        dpdf_dz = @view scratch_dummy.buffer_vpavperpzr_1[:,:,:,ir]
        @begin_anyzv_vperp_vpa_region()
        update_electron_speed_z!(z_advect[1], upar_test, vth, vpa.grid, ir)
        @loop_vperp_vpa ivperp ivpa begin
            @views z_advect[1].adv_fac[:,ivpa,ivperp,ir] = -z_speed[:,ivpa,ivperp]
        end
        #calculate the upwind derivative
        @views derivative_z_pdf_vpavperpz!(dpdf_dz, f, z_advect[1].adv_fac[:,:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_1[:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_2[:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_3[:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_4[:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_5[:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_6[:,:,ir],
                                           z_spectral, z)

        dpdf_dvpa = @view scratch_dummy.buffer_vpavperpzr_2[:,:,:,ir]
        @begin_anyzv_z_vperp_region()
        update_electron_speed_vpa!(vpa_advect[1], dens, upar, p, moments,
                                   composition.me_over_mi, vpa.grid,
                                   external_source_settings.electron, ir)
        @loop_z_vperp iz ivperp begin
            @views @. vpa_advect[1].adv_fac[:,ivperp,iz,ir] = -vpa_advect[1].speed[:,ivperp,iz,ir]
        end
        #calculate the upwind derivative of the electron pdf w.r.t. wpa
        @loop_z_vperp iz ivperp begin
            @views derivative!(dpdf_dvpa[:,ivperp,iz], f[:,ivperp,iz], vpa,
                               vpa_advect[1].adv_fac[:,ivperp,iz,ir], vpa_spectral)
        end

        d2pdf_dvpa2 = @view scratch_dummy.buffer_vpavperpzr_3[:,:,:,ir]
        @begin_anyzv_z_vperp_region()
        @loop_z_vperp iz ivperp begin
            @views second_derivative!(d2pdf_dvpa2[:,ivperp,iz], f[:,ivperp,iz], vpa,
                                      vpa_spectral)
        end

        zeroth_moment = z.scratch_shared
        first_moment = z.scratch_shared2
        second_moment = z.scratch_shared3
        @begin_anyzv_z_region()
        @loop_z iz begin
            @views zeroth_moment[iz] = integral(f[:,:,iz], vpa.grid, 0, vpa.wgts,
                                                vperp.grid, 0, vperp.wgts)
            @views first_moment[iz] = integral(f[:,:,iz], vpa.grid, 1, vpa.wgts,
                                                vperp.grid, 0, vperp.wgts)
            @views second_moment[iz] = integral((vperp,vpa)->(vpa^2+vperp^2), f[:,:,iz],
                                                vperp, vpa)
        end

        jacobian = create_jacobian_info((; vpa=vpa, vperp=vperp, z=z),
                                        (; vpa=vpa_spectral, vperp=vperp_spectral,
                                         z=z_spectral);
                                        comm=comm_anyzv_subblock[],
                                        synchronize=_anyzv_subblock_synchronize,
                                        electron_pdf=((:anyzv,:z,:vperp,:vpa), (:vpa, :vperp, :z), false),
                                        electron_p=((:anyzv,:z), (:z,), false),
                                        boundary_skip_funcs=(electron_pdf=skip_f_electron_bc_points_in_Jacobian,
                                                             electron_p=nothing))
        jacobian_initialize_identity!(jacobian)

        sub_terms = get_electron_sub_terms(dens, ddens_dz, upar_test, dupar_dz, p, dp_dz,
                                           dvth_dz, zeroth_moment, first_moment,
                                           second_moment, third_moment, dthird_moment_dz,
                                           dqpar_dz, ion_upar, f, dpdf_dz, dpdf_dvpa,
                                           d2pdf_dvpa2, me, moments, collisions,
                                           composition, external_source_settings,
                                           num_diss_params, t_params.electron, ion_dt, z,
                                           vperp, vpa, z_speed, vpa_speed, ir)
        equation_term = get_term(sub_terms)
        add_term_to_Jacobian!(jacobian, :electron_pdf, dt, equation_term, z_speed)

        # Test 'ADI Jacobians' before other tests, because residual_func() may modify some
        # variables (vth, etc.).

        jacobian_ADI_check = create_jacobian_info((; vpa=vpa, vperp=vperp, z=z),
                                                  (; vpa=vpa_spectral,
                                                   vperp=vperp_spectral, z=z_spectral);
                                                  comm=comm_anyzv_subblock[],
                                                  synchronize=_anyzv_subblock_synchronize,
                                                  electron_pdf=((:anyzv,:z,:vperp,:vpa), (:vpa, :vperp, :z), false),
                                                  electron_p=((:anyzv,:z), (:z,), false),
                                                  boundary_skip_funcs=(electron_pdf=skip_f_electron_bc_points_in_Jacobian,
                                                                       electron_p=nothing))
        v_solve_jacobian_ADI_check = create_jacobian_info((; vpa=vpa, vperp=vperp),
                                                          (; vpa=vpa_spectral,
                                                           vperp=vperp_spectral);
                                                          comm=nothing,
                                                          synchronize=nothing,
                                                          electron_pdf=(nothing, (:vpa, :vperp), false),
                                                          electron_p=(nothing, (), false),
                                                          boundary_skip_funcs=(electron_pdf=skip_f_electron_bc_points_in_Jacobian_v_solve,
                                                                               electron_p=nothing))
        z_solve_jacobian_ADI_check = create_jacobian_info((; z=z),
                                                          (; z=z_spectral);
                                                          comm=nothing,
                                                          synchronize=nothing,
                                                          electron_pdf=(nothing, (:z,), false),
                                                          boundary_skip_funcs=(electron_pdf=skip_f_electron_bc_points_in_Jacobian_z_solve,
                                                                               electron_p=nothing))

        @testset "ADI Jacobians - implicit z" begin
            # 'Implicit' and 'explicit' parts of Jacobian should add up to full Jacobian.
            jacobian_initialize_identity!(jacobian_ADI_check)

            v_size = vperp.n * vpa.n

            # Add 'implicit' contribution
            @begin_anyzv_vperp_vpa_region()
            @loop_vperp_vpa ivperp ivpa begin
                this_slice = (ivperp - 1)*vpa.n + ivpa:v_size:(z.n - 1)*v_size + (ivperp - 1)*vpa.n + ivpa

                # We are reusing z_solve_jacobian_ADI_check, so need to zero out its
                # matrix.
                z_solve_jacobian_ADI_check.matrix .= 0.0

                implicit_z_sub_terms = @views get_electron_sub_terms_z_only_Jacobian(
                                                  dens, ddens_dz, upar_test, dupar_dz, p,
                                                  dp_dz, dvth_dz, zeroth_moment,
                                                  first_moment, second_moment,
                                                  third_moment, dthird_moment_dz,
                                                  dqpar_dz, ion_upar, f[ivpa,ivperp,:],
                                                  dpdf_dz[ivpa,ivperp,:],
                                                  dpdf_dvpa[ivpa,ivperp,:],
                                                  d2pdf_dvpa2[ivpa,ivperp,:], me, moments,
                                                  collisions, external_source_settings,
                                                  num_diss_params, t_params.electron,
                                                  ion_dt, z, vperp, vpa,
                                                  z_speed[:,ivpa,ivperp], ir, ivperp,
                                                  ivpa)
                implict_z_term = get_term(implicit_z_sub_terms)
                @views add_term_to_Jacobian!(z_solve_jacobian_ADI_check, :electron_pdf,
                                             dt, implict_z_term, z_speed[:,ivpa,ivperp])

                @views jacobian_ADI_check.matrix[this_slice,this_slice] .+= z_solve_jacobian_ADI_check.matrix
            end
            @_anyzv_subblock_synchronize()

            # Add 'explicit' contribution
            explicit_v_sub_terms = get_electron_sub_terms(
                                       dens, ddens_dz, upar_test, dupar_dz, p, dp_dz,
                                       dvth_dz, zeroth_moment, first_moment,
                                       second_moment, third_moment, dthird_moment_dz,
                                       dqpar_dz, ion_upar, f, dpdf_dz, dpdf_dvpa,
                                       d2pdf_dvpa2, me, moments, collisions, composition,
                                       external_source_settings, num_diss_params,
                                       t_params.electron, ion_dt, z, vperp, vpa, z_speed,
                                       vpa_speed, ir, :explicit_v)
            explicit_v_term = get_term(explicit_v_sub_terms)
            add_term_to_Jacobian!(jacobian_ADI_check, :electron_pdf, dt, explicit_v_term,
                                  z_speed)

            @begin_anyzv_region()
            @anyzv_serial_region begin
                @test elementwise_isapprox(jacobian_ADI_check.matrix, jacobian.matrix;
                                           rtol=1.0e-15,
                                           atol=1.0e-15*max(extrema(jacobian.matrix)...))
            end
            @_anyzv_subblock_synchronize()
        end

        @testset "ADI Jacobians - implicit v" begin
            # 'Implicit' and 'explicit' parts of Jacobian should add up to full Jacobian.
            jacobian_initialize_identity!(jacobian_ADI_check)

            v_size = vperp.n * vpa.n

            # Add 'implicit' contribution
            @begin_anyzv_z_region()
            @loop_z iz begin
                this_slice = collect((iz - 1)*v_size + 1:iz*v_size)
                push!(this_slice, iz + pdf_size)

                # We are reusing v_solve_jacobian_ADI_check, so need to zero out its
                # matrix.
                v_solve_jacobian_ADI_check.matrix .= 0.0

                implicit_v_sub_terms, this_z_speed =
                    get_electron_sub_terms_v_only_Jacobian(
                        dens[iz], ddens_dz[iz], upar_test[iz], dupar_dz[iz], @view(p[iz]),
                        dp_dz[iz], @view(dvth_dz[iz]), @view(zeroth_moment[iz]),
                        @view(first_moment[iz]), @view(second_moment[iz]),
                        @view(third_moment[iz]), dthird_moment_dz[iz],
                        @view(dqpar_dz[iz]), ion_upar[iz], @view(f[:,:,iz]),
                        @view(dpdf_dz[:,:,iz]), @view(dpdf_dvpa[:,:,iz]),
                        @view(d2pdf_dvpa2[:,:,iz]), me, moments, collisions,
                        external_source_settings, num_diss_params, t_params.electron,
                        ion_dt, z, vperp, vpa, @view(z_speed[iz,:,:]),
                        @view(vpa_speed[:,:,iz]), ir, iz)
                implicit_v_term = get_term(implicit_v_sub_terms)
                add_term_to_Jacobian!(v_solve_jacobian_ADI_check, :electron_pdf, dt,
                                      implicit_v_term, this_z_speed)
                @views jacobian_ADI_check.matrix[this_slice,this_slice] .+= v_solve_jacobian_ADI_check.matrix
            end
            @_anyzv_subblock_synchronize()

            # Add 'explicit' contribution
            explicit_z_sub_terms = get_electron_sub_terms(
                                       dens, ddens_dz, upar_test, dupar_dz, p, dp_dz,
                                       dvth_dz, zeroth_moment, first_moment,
                                       second_moment, third_moment, dthird_moment_dz,
                                       dqpar_dz, ion_upar, f, dpdf_dz, dpdf_dvpa,
                                       d2pdf_dvpa2, me, moments, collisions, composition,
                                       external_source_settings, num_diss_params,
                                       t_params.electron, ion_dt, z, vperp, vpa, z_speed,
                                       vpa_speed, ir, :explicit_z)
            explicit_z_term = get_term(explicit_z_sub_terms)
            add_term_to_Jacobian!(jacobian_ADI_check, :electron_pdf, dt, explicit_z_term,
                                  z_speed)

            @begin_anyzv_region()
            @anyzv_serial_region begin
                @test elementwise_isapprox(jacobian_ADI_check.matrix, jacobian.matrix; rtol=1.0e-15, atol=2.0e-15*max(extrema(jacobian.matrix)...))
            end
        end

        function residual_func!(residual, this_f, this_p)
            @begin_anyzv_z_region()
            # Calculate derived moments and derivatives using new_variables
            #
            # For "electron_krook_collisions" upar_test is different from upar. Do not
            # pass in upar_test here, because we want upar_test to stay fixed at its
            # initial value, not be updated to be equal to ion_upar.
            calculate_electron_moments_no_r!(this_f, dens, upar, this_p, ion_dens,
                                             ion_upar, moments, composition, collisions,
                                             r, z, vperp, vpa, ir)
            calculate_electron_moment_derivatives_no_r!(
                moments, dens, upar_test, this_p, scratch_dummy, z, z_spectral,
                num_diss_params.electron.moment_dissipation_coefficient, ir)

            # electron_kinetic_equation_euler_update!() just adds dt*d(g_e)/dt to the
            # electron_pdf member of the first argument, so if we set the electron_pdf member
            # of the first argument to zero, and pass dt=1, then it will evaluate the time
            # derivative, which is the residual for a steady-state solution.
            @begin_anyzv_z_vperp_vpa_region()
            @loop_z_vperp_vpa iz ivperp ivpa begin
                residual[ivpa,ivperp,iz] = f[ivpa,ivperp,iz]
            end
            rhs_func!(; residual, this_f, dens, upar=upar_test, this_p, vth, ion_upar,
                      moments, collisions, composition, z_advect, vpa_advect, z, vperp,
                      vpa, z_spectral, vpa_spectral, external_source_settings,
                      num_diss_params, t_params, scratch_dummy, dt, ir)
            # Now
            #   residual = f_electron_old + dt*RHS(f_electron_newvar)
            # so update to desired residual
            @begin_anyzv_z_vperp_vpa_region()
            @loop_z_vperp_vpa iz ivperp ivpa begin
                residual[ivpa,ivperp,iz] = this_f[ivpa,ivperp,iz] - residual[ivpa,ivperp,iz]
            end

            # Set residual to zero where pdf_electron is determined by boundary conditions.
            if vpa.n > 1
                @begin_anyzv_z_vperp_region()
                @loop_z_vperp iz ivperp begin
                    @views enforce_v_boundary_condition_local!(residual[:,ivperp,iz], vpa.bc,
                                                               vpa_advect[1].speed[:,ivperp,iz,ir],
                                                               num_diss_params.electron.vpa_dissipation_coefficient > 0.0,
                                                               vpa, vpa_spectral)
                end
            end
            if vperp.n > 1
                @begin_anyzv_z_vpa_region()
                enforce_vperp_boundary_condition!(residual, vperp.bc,
                                                  vperp, vperp_spectral, vperp_adv,
                                                  vperp_diffusion, ir)
            end
            if z.bc == "wall"
                error("z_bc = \"wall\" not supported here yet.")
            elseif (z.bc == "constant") && (z.irank == 0 || z.irank == z.nrank - 1)
                # Boundary conditions on incoming part of distribution function. Note
                # that as density, upar, p do not change in this implicit step,
                # f_electron_newvar, f_old, and residual should all be zero at exactly the
                # same set of grid points, so it is reasonable to zero-out `residual` to
                # impose the boundary condition. We impose this after subtracting f_old in
                # case rounding errors, etc. mean that at some point f_old had a different
                # boundary condition cut-off index.
                @begin_anyzv_vperp_vpa_region()
                v_unnorm = vpa.scratch
                zero = 1.0e-14
                if z.irank == 0
                    iz = 1
                    v_unnorm .= vpagrid_to_vpa(vpa.grid, vth[iz], upar_test[iz], true,
                                               true)
                    @loop_vperp_vpa ivperp ivpa begin
                        if v_unnorm[ivpa] > -zero
                            residual[ivpa,ivperp,iz] = 0.0
                        end
                    end
                end
                if z.irank == z.nrank - 1
                    iz = z.n
                    v_unnorm .= vpagrid_to_vpa(vpa.grid, vth[iz], upar_test[iz], true,
                                               true)
                    @loop_vperp_vpa ivperp ivpa begin
                        if v_unnorm[ivpa] < zero
                            residual[ivpa,ivperp,iz] = 0.0
                        end
                    end
                end
            end
            return nothing
        end

        original_residual = allocate_shared_float(vpa, vperp, z)
        perturbed_residual = allocate_shared_float(vpa, vperp, z)

        @testset "δf only" begin
            residual_func!(original_residual, f, p)
            residual_func!(perturbed_residual, f.+delta_f, p)

            @begin_anyzv_region()
            @anyzv_serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[1:pdf_size] .= vec(delta_f)
                residual_update_with_Jacobian = jacobian.matrix * delta_state
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[1:pdf_size]

                # Check p did not get perturbed by the Jacobian
                @test elementwise_isapprox(residual_update_with_Jacobian[pdf_size+1:end],
                                           zeros(p_size); atol=1.0e-15)

                norm_factor = generate_norm_factor(perturbed_residual)
                @test elementwise_isapprox(perturbed_residual ./ norm_factor,
                                           reshape(perturbed_with_Jacobian, vpa.n, vperp.n, z.n) ./ norm_factor;
                                           rtol=0.0, atol=rtol)
            end
        end

        @testset "δp only" begin
            residual_func!(original_residual, f, p)
            residual_func!(perturbed_residual, f, p .+ delta_p)

            @begin_anyzv_region()
            @anyzv_serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[pdf_size+1:end] .= vec(delta_p)
                residual_update_with_Jacobian = jacobian.matrix * delta_state
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[1:pdf_size]

                # Check p did not get perturbed by the Jacobian
                @test elementwise_isapprox(residual_update_with_Jacobian[pdf_size+1:end],
                                           delta_state[pdf_size+1:end]; atol=1.0e-15)

                norm_factor = generate_norm_factor(perturbed_residual)
                @test elementwise_isapprox(perturbed_residual ./ norm_factor,
                                           reshape(perturbed_with_Jacobian, vpa.n, vperp.n, z.n) ./ norm_factor;
                                           rtol=0.0, atol=rtol)
            end
        end

        @testset "δf and δp" begin
            residual_func!(original_residual, f, p)
            residual_func!(perturbed_residual, f.+delta_f, p.+delta_p)

            @begin_anyzv_region()
            @anyzv_serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[1:pdf_size] .= vec(delta_f)
                delta_state[pdf_size+1:end] .= vec(delta_p)
                residual_update_with_Jacobian = jacobian.matrix * delta_state
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[1:pdf_size]

                # Check p did not get perturbed by the Jacobian
                @test elementwise_isapprox(residual_update_with_Jacobian[pdf_size+1:end],
                                           delta_state[pdf_size+1:end]; atol=1.0e-15)

                norm_factor = generate_norm_factor(perturbed_residual)
                @test elementwise_isapprox(perturbed_residual ./ norm_factor,
                                           reshape(perturbed_with_Jacobian, vpa.n, vperp.n, z.n) ./ norm_factor;
                                           rtol=0.0, atol=rtol)
            end
        end

        cleanup_mk_state!(ascii_io, io_moments, io_dfns)
    end

    return nothing
end

function test_get_p_term(test_input::AbstractDict, label::String, get_term::Function,
                         rhs_func!::Function, rtol::mk_float)
    test_input = deepcopy(test_input)
    test_input["output"]["run_name"] *= "_$label"
    println("    - $label")

    @testset "$label" begin
        # Suppress console output while running
        pdf, scratch, scratch_implicit, scratch_electron, t_params, vz, vr, vzeta, vpa,
            vperp, gyrophase, z, r, moments, fields, spectral_objects, advection_structs,
            composition, collisions, geometry, gyroavs, boundary_distributions,
            external_source_settings, num_diss_params, nl_solver_params, advance,
            advance_implicit, fp_arrays, scratch_dummy, manufactured_source_list,
            ascii_io, io_moments, io_dfns = get_mk_state(test_input)

        dens = @view moments.electron.dens[:,ir]
        upar = @view moments.electron.upar[:,ir]
        p = @view moments.electron.p[:,ir]
        ppar = @view moments.electron.ppar[:,ir]
        vth = @view moments.electron.vth[:,ir]
        qpar = @view moments.electron.qpar[:,ir]
        ion_dens = @view moments.ion.dens[:,ir]
        ion_upar = @view moments.ion.upar[:,ir]
        ddens_dz = @view moments.electron.ddens_dz[:,ir]
        dupar_dz = @view moments.electron.dupar_dz[:,ir]
        dp_dz = @view moments.electron.dp_dz[:,ir]
        dvth_dz = @view moments.electron.dvth_dz[:,ir]
        dqpar_dz = @view moments.electron.dqpar_dz[:,ir]
        f = @view pdf.electron.norm[:,:,:,ir]
        z_spectral = spectral_objects.z_spectral
        vperp_spectral = spectral_objects.vperp_spectral
        vpa_spectral = spectral_objects.vpa_spectral
        z_advect = advection_structs.z_advect
        vpa_advect = advection_structs.vpa_advect
        vpa_speed = @view vpa_advect[1].speed[:,:,:,ir]
        me = composition.me_over_mi

        @begin_r_anyzv_region()

        zeroth_moment = z.scratch_shared
        first_moment = z.scratch_shared2
        second_moment = z.scratch_shared3
        @begin_anyzv_z_region()
        @loop_z iz begin
            @views zeroth_moment[iz] = integral(f[:,:,iz], vpa.grid, 0, vpa.wgts,
                                                vperp.grid, 0, vperp.wgts)
            @views first_moment[iz] = integral(f[:,:,iz], vpa.grid, 1, vpa.wgts,
                                                vperp.grid, 0, vperp.wgts)
            @views second_moment[iz] = integral((vperp,vpa)->(vpa^2+vperp^2), f[:,:,iz],
                                                vperp, vpa)
        end

        buffer_1 = @view scratch_dummy.buffer_rs_1[ir,1]
        buffer_2 = @view scratch_dummy.buffer_rs_2[ir,1]
        buffer_3 = @view scratch_dummy.buffer_rs_3[ir,1]
        buffer_4 = @view scratch_dummy.buffer_rs_4[ir,1]

        # Reconstruct w_∥^3 moment of g_e from already-calculated qpar
        third_moment = scratch_dummy.buffer_z_1
        dthird_moment_dz = scratch_dummy.buffer_z_2
        @begin_anyzv_z_region()
        @loop_z iz begin
            third_moment[iz] = qpar[iz] / p[iz] / vth[iz]
        end
        derivative_z_anyzv!(dthird_moment_dz, third_moment, buffer_1, buffer_2, buffer_3,
                            buffer_4, z_spectral, z)

        @begin_anyzv_vperp_vpa_region()
        update_electron_speed_z!(z_advect[1], upar, vth, vpa.grid, ir)
        z_speed = @view z_advect[1].speed[:,:,:,ir]

        delta_p = allocate_shared_float(z)
        p_amplitude = epsilon * maximum(p)
        f = @view pdf.electron.norm[:,:,:,ir]
        @begin_anyzv_region()
        @anyzv_serial_region begin
            @. delta_p = p_amplitude * sin(2.0*π*test_wavenumber*z.grid/z.L)

            # Make sure initial condition has some z-variation. As f is 'moment kinetic' this
            # means f must have a non-Maxwellian part that varies in z.
            f .*= 1.0 .+ 1.0e-4 .* reshape(vpa.grid.^3, vpa.n, 1, 1) .* reshape(sin.(2.0.*π.*z.grid./z.L), 1, 1, z.n)
        end
        # Ensure initial electron distribution function obeys constraints
        hard_force_moment_constraints!(reshape(f, vpa.n, vperp.n, z.n, 1), moments, vpa, vperp)
        @begin_r_anyzv_region()
        delta_f = allocate_shared_float(vpa, vperp, z)
        f_amplitude = epsilon * maximum(f)
        @begin_anyzv_region()
        @anyzv_serial_region begin
            # Use exp(sin()) in vpa so that perturbation does not have any symmetry that makes
            # low-order moments vanish exactly.
            delta_f .= f_amplitude .*
                       reshape(sin.(2.0.*π.*test_wavenumber.*z.grid./z.L), 1, 1, z.n) .*
                       reshape(exp.(sin.(2.0.*π.*test_wavenumber.*vpa.grid./vpa.L)) .- 1.0, vpa.n, 1, 1) .*
                       f
        end

        pdf_size = length(f)
        p_size = length(p)
        total_size = pdf_size + p_size

        dpdf_dz = @view scratch_dummy.buffer_vpavperpzr_1[:,:,:,ir]
        @begin_anyzv_vperp_vpa_region()
        update_electron_speed_z!(z_advect[1], upar, vth, vpa.grid, ir)
        @loop_vperp_vpa ivperp ivpa begin
            @views z_advect[1].adv_fac[:,ivpa,ivperp,ir] = -z_speed[:,ivpa,ivperp]
        end
        #calculate the upwind derivative
        @views derivative_z_pdf_vpavperpz!(dpdf_dz, f, z_advect[1].adv_fac[:,:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_1[:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_2[:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_3[:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_4[:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_5[:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_6[:,:,ir],
                                           z_spectral, z)

        dpdf_dvpa = @view scratch_dummy.buffer_vpavperpzr_2[:,:,:,ir]
        @begin_anyzv_z_vperp_region()
        update_electron_speed_vpa!(vpa_advect[1], dens, upar, p, moments,
                                   composition.me_over_mi, vpa.grid,
                                   external_source_settings.electron, ir)
        @loop_z_vperp iz ivperp begin
            @views @. vpa_advect[1].adv_fac[:,ivperp,iz,ir] = -vpa_advect[1].speed[:,ivperp,iz,ir]
        end
        #calculate the upwind derivative of the electron pdf w.r.t. wpa
        @loop_z_vperp iz ivperp begin
            @views derivative!(dpdf_dvpa[:,ivperp,iz], f[:,ivperp,iz], vpa,
                               vpa_advect[1].adv_fac[:,ivperp,iz,ir], vpa_spectral)
        end

        d2pdf_dvpa2 = @view scratch_dummy.buffer_vpavperpzr_3[:,:,:,ir]
        @begin_anyzv_z_vperp_region()
        @loop_z_vperp iz ivperp begin
            @views second_derivative!(d2pdf_dvpa2[:,ivperp,iz], f[:,ivperp,iz], vpa,
                                      vpa_spectral)
        end

        jacobian = create_jacobian_info((; vpa=vpa, vperp=vperp, z=z),
                                        (; vpa=vpa_spectral, vperp=vperp_spectral,
                                         z=z_spectral);
                                        comm=comm_anyzv_subblock[],
                                        synchronize=_anyzv_subblock_synchronize,
                                        electron_pdf=((:anyzv,:z,:vperp,:vpa), (:vpa, :vperp, :z), false),
                                        electron_p=((:anyzv,:z), (:z,), false),
                                        boundary_skip_funcs=(electron_pdf=skip_f_electron_bc_points_in_Jacobian,
                                                             electron_p=nothing))
        jacobian_initialize_identity!(jacobian)

        sub_terms = get_electron_sub_terms(dens, ddens_dz, upar, dupar_dz, p, dp_dz,
                                           dvth_dz, zeroth_moment, first_moment,
                                           second_moment, third_moment, dthird_moment_dz,
                                           dqpar_dz, ion_upar, f, dpdf_dz, dpdf_dvpa,
                                           d2pdf_dvpa2, me, moments, collisions,
                                           composition, external_source_settings,
                                           num_diss_params, t_params, ion_dt, z, vperp,
                                           vpa, z_speed, vpa_speed, ir)
        equation_term = get_term(sub_terms)
        add_term_to_Jacobian!(jacobian, :electron_p, dt, equation_term, z_speed)

        # Test 'ADI Jacobians' before other tests, because residual_func() may modify some
        # variables (vth, etc.).

        jacobian_ADI_check = create_jacobian_info((; vpa=vpa, vperp=vperp, z=z),
                                                  (; vpa=vpa_spectral,
                                                   vperp=vperp_spectral, z=z_spectral);
                                                  comm=comm_anyzv_subblock[],
                                                  synchronize=_anyzv_subblock_synchronize,
                                                  electron_pdf=((:anyzv,:z,:vperp,:vpa), (:vpa, :vperp, :z), false),
                                                  electron_p=((:anyzv,:z), (:z,), false),
                                                  boundary_skip_funcs=(electron_pdf=skip_f_electron_bc_points_in_Jacobian,
                                                                       electron_p=nothing))
        v_solve_jacobian_ADI_check = create_jacobian_info((; vpa=vpa, vperp=vperp),
                                                          (; vpa=vpa_spectral,
                                                           vperp=vperp_spectral);
                                                          comm=nothing,
                                                          synchronize=nothing,
                                                          electron_pdf=(nothing, (:vpa, :vperp), false),
                                                          electron_p=(nothing, (), false),
                                                          boundary_skip_funcs=(electron_pdf=skip_f_electron_bc_points_in_Jacobian_v_solve,
                                                                               electron_p=nothing))
        z_solve_jacobian_ADI_check = create_jacobian_info((; z=z),
                                                          (; z=z_spectral);
                                                          comm=nothing,
                                                          synchronize=nothing,
                                                          electron_p=(nothing, (:z,), false),
                                                          boundary_skip_funcs=(electron_pdf=skip_f_electron_bc_points_in_Jacobian_z_solve,
                                                                               electron_p=nothing))

        @testset "ADI Jacobians - implicit z" begin
            # 'Implicit' and 'explicit' parts of Jacobian should add up to full Jacobian.
            jacobian_initialize_identity!(jacobian_ADI_check)

            v_size = vperp.n * vpa.n

            @serial_region begin
                this_slice = total_size - z.n + 1:total_size

                # We are reusing z_solve_jacobian_ADI_check, so need to zero out its
                # matrix.
                z_solve_jacobian_ADI_check.matrix .= 0.0

                implicit_z_sub_terms = @views get_electron_sub_terms_z_only_Jacobian(
                                                  dens, ddens_dz, upar, dupar_dz, p,
                                                  dp_dz, dvth_dz, zeroth_moment,
                                                  first_moment, second_moment,
                                                  third_moment, dthird_moment_dz,
                                                  dqpar_dz, ion_upar, f[1,1,:],
                                                  dpdf_dz[1,1,:], dpdf_dvpa[1,1,:],
                                                  d2pdf_dvpa2[1,1,:], me, moments,
                                                  collisions, external_source_settings,
                                                  num_diss_params, t_params, ion_dt, z,
                                                  vperp, vpa, z_speed[:,1,1], ir, 1, 1)
                implict_z_term = get_term(implicit_z_sub_terms)
                add_term_to_Jacobian!(z_solve_jacobian_ADI_check, :electron_p, dt,
                                      implict_z_term)

                @views jacobian_ADI_check.matrix[this_slice,this_slice] .+= z_solve_jacobian_ADI_check.matrix
            end
            @_anyzv_subblock_synchronize()

            # Add 'explicit' contribution
            explicit_v_sub_terms = get_electron_sub_terms(
                                       dens, ddens_dz, upar, dupar_dz, p, dp_dz, dvth_dz,
                                       zeroth_moment, first_moment, second_moment,
                                       third_moment, dthird_moment_dz, dqpar_dz, ion_upar,
                                       f, dpdf_dz, dpdf_dvpa, d2pdf_dvpa2, me, moments,
                                       collisions, composition, external_source_settings,
                                       num_diss_params, t_params, ion_dt, z, vperp, vpa,
                                       z_speed, vpa_speed, ir, :explicit_v)
            explicit_v_term = get_term(explicit_v_sub_terms)
            add_term_to_Jacobian!(jacobian_ADI_check, :electron_p, dt, explicit_v_term,
                                  z_speed)

            @begin_anyzv_region()
            @anyzv_serial_region begin
                @test elementwise_isapprox(jacobian_ADI_check.matrix, jacobian.matrix; rtol=0.0, atol=1.0e-15)
            end
            @_anyzv_subblock_synchronize()
        end

        @testset "ADI Jacobians - implicit v" begin
            # 'Implicit' and 'explicit' parts of Jacobian should add up to full Jacobian.
            jacobian_initialize_identity!(jacobian_ADI_check)

            v_size = vperp.n * vpa.n

            # Add 'implicit' contribution
            @begin_anyzv_z_region()
            @loop_z iz begin
                this_slice = collect((iz - 1)*v_size + 1:iz*v_size)
                push!(this_slice, iz + pdf_size)

                # We are reusing v_solve_jacobian_ADI_check, so need to zero out its
                # matrix.
                v_solve_jacobian_ADI_check.matrix .= 0.0

                implicit_v_sub_terms, this_z_speed =
                    get_electron_sub_terms_v_only_Jacobian(
                        dens[iz], ddens_dz[iz], upar[iz], dupar_dz[iz], @view(p[iz]),
                        dp_dz[iz], @view(dvth_dz[iz]), @view(zeroth_moment[iz]),
                        @view(first_moment[iz]), @view(second_moment[iz]),
                        @view(third_moment[iz]), dthird_moment_dz[iz],
                        @view(dqpar_dz[iz]), ion_upar[iz], @view(f[:,:,iz]),
                        @view(dpdf_dz[:,:,iz]), @view(dpdf_dvpa[:,:,iz]),
                        @view(d2pdf_dvpa2[:,:,iz]), me, moments, collisions,
                        external_source_settings, num_diss_params, t_params, ion_dt, z,
                        vperp, vpa, @view(z_speed[iz,:,:]), @view(vpa_speed[:,:,iz]), ir,
                        iz)
                implicit_v_term = get_term(implicit_v_sub_terms)
                add_term_to_Jacobian!(v_solve_jacobian_ADI_check, :electron_p, dt,
                                      implicit_v_term, this_z_speed)
                jacobian_ADI_check.matrix[this_slice,this_slice] .+= v_solve_jacobian_ADI_check.matrix
            end
            @_anyzv_subblock_synchronize()

            # Add 'explicit' contribution
            explicit_z_sub_terms = get_electron_sub_terms(
                                       dens, ddens_dz, upar, dupar_dz, p, dp_dz, dvth_dz,
                                       zeroth_moment, first_moment, second_moment,
                                       third_moment, dthird_moment_dz, dqpar_dz, ion_upar,
                                       f, dpdf_dz, dpdf_dvpa, d2pdf_dvpa2, me, moments,
                                       collisions, composition, external_source_settings,
                                       num_diss_params, t_params, ion_dt, z, vperp, vpa,
                                       z_speed, vpa_speed, ir, :explicit_z)
            explicit_z_term = get_term(explicit_z_sub_terms)
            add_term_to_Jacobian!(jacobian_ADI_check, :electron_p, dt, explicit_z_term,
                                  z_speed)

            @begin_anyzv_region()
            @anyzv_serial_region begin
                @test elementwise_isapprox(jacobian_ADI_check.matrix, jacobian.matrix;
                                           rtol=0.0,
                                           atol=1.0e-15*max(extrema(jacobian.matrix)...))
            end
        end

        function residual_func!(residual, this_f, this_p)
            @begin_anyzv_z_region()
            # Calculate derived moments and derivatives using new_variables
            calculate_electron_moments_no_r!(this_f, dens, upar, this_p, ion_dens,
                                             ion_upar, moments, composition, collisions,
                                             r, z, vperp, vpa, ir)
            calculate_electron_moment_derivatives_no_r!(
                moments, dens, upar, this_p, scratch_dummy, z, z_spectral,
                num_diss_params.electron.moment_dissipation_coefficient, ir)

            # electron_kinetic_equation_euler_update!() just adds dt*d(g_e)/dt to the
            # electron_pdf member of the first argument, so if we set the electron_pdf member
            # of the first argument to zero, and pass dt=1, then it will evaluate the time
            # derivative, which is the residual for a steady-state solution.
            @begin_anyzv_z_region()
            @loop_z iz begin
                residual[iz] = p[iz]
            end
            @views rhs_func!(; residual, this_p, dens, upar, vth, ppar,
                             ion_dens=moments.ion.dens[:,ir,1],
                             ion_upar=moments.ion.upar[:,ir,1],
                             ion_p=moments.ion.p[:,ir,1],
                             neutral_dens=moments.neutral.dens[:,ir,1],
                             neutral_uz=moments.neutral.uz[:,ir,1],
                             neutral_p=moments.neutral.p[:,ir,1], moments, collisions,
                             composition, z, z_spectral, external_source_settings,
                             num_diss_params, t_params, ion_dt, scratch_dummy, dt, ir)
            # Now
            #   residual = f_electron_old + dt*RHS(f_electron_newvar)
            # so update to desired residual
            @begin_anyzv_z_region()
            @loop_z iz begin
                residual[iz] = this_p[iz] - residual[iz]
            end
        end

        original_residual = allocate_shared_float(z)
        perturbed_residual = allocate_shared_float(z)

        @testset "δf only" begin
            residual_func!(original_residual, f, p)
            residual_func!(perturbed_residual, f.+delta_f, p)

            @begin_anyzv_region()
            @anyzv_serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[1:pdf_size] .= vec(delta_f)
                residual_update_with_Jacobian = jacobian.matrix * delta_state
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[pdf_size+1:end]

                # Check f did not get perturbed by the Jacobian
                @test elementwise_isapprox(residual_update_with_Jacobian[1:pdf_size],
                                           delta_state[1:pdf_size]; atol=1.0e-15)

                if label == "ion_dt_forcing_of_electron_p"
                    # No norm factor, because both perturbed residuals should be zero
                    # here, as delta_f does not affect this term, and `p` is used as
                    # `p_previous_ion_step` in this test, so the residuals are exactly
                    # zero if there is no delta_p.
                    norm_factor = 1.0
                else
                    norm_factor = generate_norm_factor(perturbed_residual)
                end
                @test elementwise_isapprox(perturbed_residual ./ norm_factor,
                                           perturbed_with_Jacobian ./ norm_factor;
                                           rtol=0.0, atol=rtol)
            end
        end

        @testset "δp only" begin
            residual_func!(original_residual, f, p)
            residual_func!(perturbed_residual, f, p .+ delta_p)

            @begin_anyzv_region()
            @anyzv_serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[pdf_size+1:end] .= vec(delta_p)
                residual_update_with_Jacobian = jacobian.matrix * delta_state
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[pdf_size+1:end]

                # Check f did not get perturbed by the Jacobian
                @test elementwise_isapprox(residual_update_with_Jacobian[1:pdf_size],
                                           zeros(pdf_size); atol=1.0e-15)

                norm_factor = generate_norm_factor(perturbed_residual)
                @test elementwise_isapprox(perturbed_residual ./ norm_factor,
                                           perturbed_with_Jacobian ./ norm_factor;
                                           rtol=0.0, atol=rtol)
            end
        end

        @testset "δf and δp" begin
            residual_func!(original_residual, f, p)
            residual_func!(perturbed_residual, f.+delta_f, p.+delta_p)

            @begin_anyzv_region()
            @anyzv_serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[1:pdf_size] .= vec(delta_f)
                delta_state[pdf_size+1:end] .= vec(delta_p)
                residual_update_with_Jacobian = jacobian.matrix * delta_state
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[pdf_size+1:end]

                # Check p did not get perturbed by the Jacobian
                @test elementwise_isapprox(residual_update_with_Jacobian[1:pdf_size],
                                           delta_state[1:pdf_size]; atol=1.0e-15)

                norm_factor = generate_norm_factor(perturbed_residual)
                @test elementwise_isapprox(perturbed_residual ./ norm_factor,
                                           perturbed_with_Jacobian ./ norm_factor;
                                           rtol=0.0, atol=rtol)
            end
        end

        cleanup_mk_state!(ascii_io, io_moments, io_dfns)
    end

    return nothing
end

function test_electron_kinetic_equation(test_input; rtol=(5.0e2*epsilon)^2)

    # Looser rtol for "wall" bc because integral corrections not accounted for in wall bc
    # Jacobian (yet?).
    @testset "electron_kinetic_equation bc=$bc" for (bc, adi_tol) ∈ (("constant", 1.0e-15), ("wall", 1.0e-13))
        println("    - electron_kinetic_equation $bc")
        this_test_input = deepcopy(test_input)
        this_test_input["output"]["run_name"] *= "_electron_kinetic_equation_$bc"
        this_test_input["z"]["bc"] = bc

        # Suppress console output while running
        pdf, scratch, scratch_implicit, scratch_electron, t_params, vz, vr, vzeta, vpa,
            vperp, gyrophase, z, r, moments, fields, spectral_objects, advection_structs,
            composition, collisions, geometry, gyroavs, boundary_distributions,
            external_source_settings, num_diss_params, nl_solver_params, advance,
            advance_implicit, fp_arrays, scratch_dummy, manufactured_source_list,
            ascii_io, io_moments, io_dfns = get_mk_state(this_test_input)

        dens = @view moments.electron.dens[:,ir]
        upar = @view moments.electron.upar[:,ir]
        p = @view moments.electron.p[:,ir]
        vth = @view moments.electron.vth[:,ir]
        qpar = @view moments.electron.qpar[:,ir]
        ion_dens = @view moments.ion.dens[:,ir]
        ion_upar = @view moments.ion.upar[:,ir]
        phi = @view fields.phi[:,ir]
        z_spectral = spectral_objects.z_spectral
        vperp_spectral = spectral_objects.vperp_spectral
        vpa_spectral = spectral_objects.vpa_spectral
        z_advect = advection_structs.z_advect
        vpa_advect = advection_structs.vpa_advect

        delta_p = allocate_shared_float(z)
        p_amplitude = epsilon * maximum(p)
        f = @view pdf.electron.norm[:,:,:,ir]

        @begin_r_anyzv_region()

        @begin_anyzv_region()
        @anyzv_serial_region begin
            @. delta_p = p_amplitude * sin(2.0*π*test_wavenumber*z.grid/z.L)

            # Make sure initial condition has some z-variation. As f is 'moment kinetic' this
            # means f must have a non-Maxwellian part that varies in z.
            f .*= 1.0 .+ 1.0e-4 .* reshape(vpa.grid.^3, vpa.n, 1, 1) .* reshape(sin.(2.0.*π.*z.grid./z.L), 1, 1, z.n)
        end
        # Ensure initial electron distribution function obeys constraints
        hard_force_moment_constraints!(reshape(f, vpa.n, vperp.n, z.n, 1), moments, vpa, vperp)
        @begin_r_anyzv_region()
        delta_f = allocate_shared_float(vpa, vperp, z)
        f_amplitude = epsilon * maximum(f)
        # Use exp(sin()) in vpa so that perturbation does not have any symmetry that makes
        # low-order moments vanish exactly.
        @begin_anyzv_region()
        @anyzv_serial_region begin
            delta_f .= f_amplitude .*
                       reshape(sin.(2.0.*π.*test_wavenumber.*z.grid./z.L), 1, 1, z.n) .*
                       reshape(exp.(sin.(2.0.*π.*test_wavenumber.*vpa.grid./vpa.L)) .- 1.0, vpa.n, 1, 1) .*
                       f
        end

        pdf_size = length(f)
        p_size = length(p)
        total_size = pdf_size + p_size

        # Test 'ADI Jacobians' before other tests, because residual_func() may modify some
        # variables (vth, etc.).

        # Reconstruct w_∥^3 moment of g_e from already-calculated qpar
        buffer_1 = @view scratch_dummy.buffer_rs_1[ir,1]
        buffer_2 = @view scratch_dummy.buffer_rs_2[ir,1]
        buffer_3 = @view scratch_dummy.buffer_rs_3[ir,1]
        buffer_4 = @view scratch_dummy.buffer_rs_4[ir,1]
        third_moment = scratch_dummy.buffer_z_1
        dthird_moment_dz = scratch_dummy.buffer_z_2
        @begin_anyzv_z_region()
        @loop_z iz begin
            third_moment[iz] = qpar[iz] / p[iz] / vth[iz]
        end
        derivative_z_anyzv!(dthird_moment_dz, third_moment, buffer_1, buffer_2, buffer_3,
                            buffer_4, z_spectral, z)

        z_speed = @view z_advect[1].speed[:,:,:,ir]

        dpdf_dz = @view scratch_dummy.buffer_vpavperpzr_1[:,:,:,ir]
        @begin_anyzv_vperp_vpa_region()
        update_electron_speed_z!(z_advect[1], upar, vth, vpa.grid, ir)
        @loop_vperp_vpa ivperp ivpa begin
            @views z_advect[1].adv_fac[:,ivpa,ivperp,ir] = -z_speed[:,ivpa,ivperp]
        end
        #calculate the upwind derivative
        @views derivative_z_pdf_vpavperpz!(dpdf_dz, f, z_advect[1].adv_fac[:,:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_1[:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_2[:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_3[:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_4[:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_5[:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_6[:,:,ir],
                                           z_spectral, z)

        dpdf_dvpa = @view scratch_dummy.buffer_vpavperpzr_2[:,:,:,ir]
        @begin_anyzv_z_vperp_region()
        update_electron_speed_vpa!(vpa_advect[1], dens, upar, p, moments,
                                   composition.me_over_mi, vpa.grid,
                                   external_source_settings.electron, ir)
        @loop_z_vperp iz ivperp begin
            @views @. vpa_advect[1].adv_fac[:,ivperp,iz,ir] = -vpa_advect[1].speed[:,ivperp,iz,ir]
        end
        #calculate the upwind derivative of the electron pdf w.r.t. wpa
        @loop_z_vperp iz ivperp begin
            @views derivative!(dpdf_dvpa[:,ivperp,iz], f[:,ivperp,iz], vpa,
                               vpa_advect[1].adv_fac[:,ivperp,iz,ir], vpa_spectral)
        end
        vpa_speed = @view vpa_advect[1].speed[:,:,:,ir]

        d2pdf_dvpa2 = @view scratch_dummy.buffer_vpavperpzr_3[:,:,:,ir]
        @begin_anyzv_z_vperp_region()
        @loop_z_vperp iz ivperp begin
            @views second_derivative!(d2pdf_dvpa2[:,ivperp,iz], f[:,ivperp,iz], vpa,
                                      vpa_spectral)
        end

        zeroth_moment = z.scratch_shared
        first_moment = z.scratch_shared2
        second_moment = z.scratch_shared3
        @begin_anyzv_z_region()
        @loop_z iz begin
            @views zeroth_moment[iz] = integral(f[:,:,iz], vpa.grid, 0, vpa.wgts,
                                                vperp.grid, 0, vperp.wgts)
            @views first_moment[iz] = integral(f[:,:,iz], vpa.grid, 1, vpa.wgts,
                                                vperp.grid, 0, vperp.wgts)
            @views second_moment[iz] = integral((vperp,vpa)->(vpa^2+vperp^2), f[:,:,iz],
                                                vperp, vpa)
        end

        jacobian = create_jacobian_info((; vpa=vpa, vperp=vperp, z=z),
                                        (; vpa=vpa_spectral, vperp=vperp_spectral,
                                         z=z_spectral);
                                        comm=comm_anyzv_subblock[],
                                        synchronize=_anyzv_subblock_synchronize,
                                        electron_pdf=((:anyzv,:z,:vperp,:vpa), (:vpa, :vperp, :z), false),
                                        electron_p=((:anyzv,:z), (:z,), false),
                                        boundary_skip_funcs=(electron_pdf=skip_f_electron_bc_points_in_Jacobian,
                                                             electron_p=nothing))

        # Calculate jacobian later, so that we can use `jacobian` as a temporary buffer,
        # to avoid allocating too much shared memory for the Github Actions CI servers.
        #fill_electron_kinetic_equation_Jacobian!(
        #    jacobian, f, p, moments, phi, collisions, composition, z, vperp, vpa,
        #    z_spectral, vperp_spectral, vpa_spectral, z_advect, vpa_advect,
        #    scratch_dummy, external_source_settings, num_diss_params,
        #    t_params.electron, ion_dt, ir, true)

        # Test 'ADI Jacobians' before other tests, because residual_func() may modify some
        # variables (vth, etc.).

        jacobian_ADI_check = create_jacobian_info((; vpa=vpa, vperp=vperp, z=z),
                                                  (; vpa=vpa_spectral,
                                                   vperp=vperp_spectral, z=z_spectral);
                                                  comm=comm_anyzv_subblock[],
                                                  synchronize=_anyzv_subblock_synchronize,
                                                  electron_pdf=((:anyzv,:z,:vperp,:vpa), (:vpa, :vperp, :z), false),
                                                  electron_p=((:anyzv,:z), (:z,), false),
                                                  boundary_skip_funcs=(electron_pdf=skip_f_electron_bc_points_in_Jacobian,
                                                                       electron_p=nothing))
        v_solve_jacobian_ADI_check = create_jacobian_info((; vpa=vpa, vperp=vperp),
                                                          (; vpa=vpa_spectral,
                                                           vperp=vperp_spectral);
                                                          comm=nothing,
                                                          synchronize=nothing,
                                                          electron_pdf=(nothing, (:vpa, :vperp), false),
                                                          electron_p=(nothing, (), false),
                                                          boundary_skip_funcs=(electron_pdf=skip_f_electron_bc_points_in_Jacobian_v_solve,
                                                                               electron_p=nothing))
        z_solve_jacobian_ADI_check = create_jacobian_info((; z=z),
                                                          (; z=z_spectral);
                                                          comm=nothing,
                                                          synchronize=nothing,
                                                          electron_pdf=(nothing, (:z,), false),
                                                          boundary_skip_funcs=(electron_pdf=skip_f_electron_bc_points_in_Jacobian_z_solve,
                                                                               electron_p=nothing))

        z_solve_p_jacobian_ADI_check = create_jacobian_info((; z=z),
                                                            (; z=z_spectral);
                                                            comm=nothing,
                                                            synchronize=nothing,
                                                            electron_p=(nothing, (:z,), false),
                                                            boundary_skip_funcs=(electron_pdf=skip_f_electron_bc_points_in_Jacobian_z_solve,
                                                                                 electron_p=nothing))

        @testset "ADI Jacobians - implicit z" begin
            # 'Implicit' and 'explicit' parts of Jacobian should add up to full Jacobian.
            jacobian_initialize_zero!(jacobian_ADI_check)

            v_size = vperp.n * vpa.n

            # Add 'implicit' contribution
            @begin_anyzv_vperp_vpa_region()
            @loop_vperp_vpa ivperp ivpa begin
                this_slice = (ivperp - 1)*vpa.n + ivpa:v_size:(z.n - 1)*v_size + (ivperp - 1)*vpa.n + ivpa
                @views fill_electron_kinetic_equation_z_only_Jacobian_f!(
                    z_solve_jacobian_ADI_check, f[ivpa,ivperp,:], p,
                    dpdf_dz[ivpa,ivperp,:], dpdf_dvpa[ivpa,ivperp,:],
                    d2pdf_dvpa2[ivpa,ivperp,:], z_speed[:,ivpa,ivperp], moments,
                    zeroth_moment, first_moment, second_moment, third_moment,
                    dthird_moment_dz, collisions, composition, z, vperp, vpa, z_spectral,
                    vperp_spectral, vpa_spectral, z_advect, vpa_advect, scratch_dummy,
                    external_source_settings, num_diss_params, t_params.electron, ion_dt,
                    ir, ivperp, ivpa)

                @views jacobian_ADI_check.matrix[this_slice,this_slice] .+= z_solve_jacobian_ADI_check.matrix
            end

            @begin_anyzv_region()
            @anyzv_serial_region begin
                # Add 'implicit' contribution
                this_slice = (pdf_size + 1):total_size
                @views fill_electron_kinetic_equation_z_only_Jacobian_p!(
                    z_solve_p_jacobian_ADI_check, p, f[1,1,:], dpdf_dz[1,1,:],
                    dpdf_dvpa[1,1,:], d2pdf_dvpa2[1,1,:], z_speed[:,1,1], moments,
                    zeroth_moment, first_moment, second_moment, third_moment,
                    dthird_moment_dz, collisions, composition, z, vperp, vpa, z_spectral,
                    vperp_spectral, vpa_spectral, z_advect, vpa_advect, scratch_dummy,
                    external_source_settings, num_diss_params, t_params.electron, ion_dt,
                    ir, true)
                @begin_anyzv_region()
                @views jacobian_ADI_check.matrix[this_slice,this_slice] .+= z_solve_p_jacobian_ADI_check.matrix
            end
            @_anyzv_subblock_synchronize()

            # Add 'explicit' contribution
            # Use jacobian as a temporary buffer here.
            fill_electron_kinetic_equation_Jacobian!(
                jacobian, f, p, moments, phi, collisions, composition, z, vperp, vpa,
                z_spectral, vperp_spectral, vpa_spectral, z_advect, vpa_advect,
                scratch_dummy, external_source_settings, num_diss_params,
                t_params.electron, ion_dt, ir, true, :explicit_v)
            @begin_anyzv_region()
            @anyzv_serial_region begin
                jacobian_ADI_check.matrix .+= jacobian.matrix
            end
            @_anyzv_subblock_synchronize()

            fill_electron_kinetic_equation_Jacobian!(
                jacobian, f, p, moments, phi, collisions, composition, z, vperp, vpa,
                z_spectral, vperp_spectral, vpa_spectral, z_advect, vpa_advect,
                scratch_dummy, external_source_settings, num_diss_params,
                t_params.electron, ion_dt, ir, true)

            @begin_anyzv_region()
            @anyzv_serial_region begin
                # The settings for this test are a bit strange, due to trying to get the
                # finite-difference approximation to the Jacobian to agree with the
                # Jacobian matrix functions without being too messed up by floating-point
                # rounding errors. The result is that some entries in the Jacobian matrix
                # here are O(1.0e5), so it is important to use `rtol` here.
                @test elementwise_isapprox(jacobian_ADI_check.matrix, jacobian.matrix; rtol=adi_tol, atol=1.0e-15)
            end
            @_anyzv_subblock_synchronize()
        end

        @testset "ADI Jacobians - implicit v" begin
            # 'Implicit' and 'explicit' parts of Jacobian should add up to full Jacobian.
            jacobian_initialize_zero!(jacobian_ADI_check)

            v_size = vperp.n * vpa.n

            # Add 'implicit' contribution
            @begin_anyzv_z_region()
            @loop_z iz begin
                this_slice = collect((iz - 1)*v_size + 1:iz*v_size)
                push!(this_slice, iz + pdf_size)
                fill_electron_kinetic_equation_v_only_Jacobian!(
                    v_solve_jacobian_ADI_check, @view(f[:,:,iz]), @view(p[iz]),
                    @view(dpdf_dz[:,:,iz]), @view(dpdf_dvpa[:,:,iz]),
                    @view(d2pdf_dvpa2[:,:,iz]), @view(z_speed[iz,:,:]),
                    @view(vpa_speed[:,:,iz]), moments, @view(zeroth_moment[iz]),
                    @view(first_moment[iz]), @view(second_moment[iz]),
                    @view(third_moment[iz]), dthird_moment_dz[iz], phi[iz], collisions,
                    composition, z, vperp, vpa, z_spectral, vperp_spectral, vpa_spectral,
                    z_advect, vpa_advect, scratch_dummy, external_source_settings,
                    num_diss_params, t_params.electron, ion_dt, ir, iz, true)
                @views jacobian_ADI_check.matrix[this_slice,this_slice] .+= v_solve_jacobian_ADI_check.matrix
            end
            @_anyzv_subblock_synchronize()

            # Add 'explicit' contribution
            # Use jacobian as a temporary buffer here.
            fill_electron_kinetic_equation_Jacobian!(
                jacobian, f, p, moments, phi, collisions, composition, z, vperp, vpa,
                z_spectral, vperp_spectral, vpa_spectral, z_advect, vpa_advect,
                scratch_dummy, external_source_settings, num_diss_params,
                t_params.electron, ion_dt, ir, true, :explicit_z)

            @begin_anyzv_region()
            @anyzv_serial_region begin
                jacobian_ADI_check.matrix .+= jacobian.matrix
            end
            @_anyzv_subblock_synchronize()

            fill_electron_kinetic_equation_Jacobian!(
                jacobian, f, p, moments, phi, collisions, composition, z, vperp, vpa,
                z_spectral, vperp_spectral, vpa_spectral, z_advect, vpa_advect,
                scratch_dummy, external_source_settings, num_diss_params,
                t_params.electron, ion_dt, ir, true)

            @begin_anyzv_region()
            @anyzv_serial_region begin
                # The settings for this test are a bit strange, due to trying to get the
                # finite-difference approximation to the Jacobian to agree with the
                # Jacobian matrix functions without being too messed up by floating-point
                # rounding errors. The result is that some entries in the Jacobian matrix
                # here are O(1.0e5), so it is important to use `rtol` here.
                @test elementwise_isapprox(jacobian_ADI_check.matrix, jacobian.matrix; rtol=10.0*adi_tol, atol=1.0e-13)
            end
        end

        function residual_func!(residual_f, residual_p, this_f, this_p)
            @begin_anyzv_z_region()
            # Calculate derived moments and derivatives using new_variables
            calculate_electron_moments_no_r!(this_f, dens, upar, this_p, ion_dens,
                                             ion_upar, moments, composition, collisions,
                                             r, z, vperp, vpa, ir)
            calculate_electron_moment_derivatives_no_r!(
                moments, dens, upar, this_p, scratch_dummy, z, z_spectral,
                num_diss_params.electron.moment_dissipation_coefficient, ir)

            # electron_kinetic_equation_euler_update!() just adds dt*d(g_e)/dt to the
            # electron_pdf member of the first argument, so if we set the electron_pdf member
            # of the first argument to zero, and pass dt=1, then it will evaluate the time
            # derivative, which is the residual for a steady-state solution.
            @begin_anyzv_z_vperp_vpa_region()
            @loop_z_vperp_vpa iz ivperp ivpa begin
                residual_f[ivpa,ivperp,iz] = f[ivpa,ivperp,iz]
            end
            @begin_anyzv_z_region()
            @loop_z iz begin
                residual_p[iz] = p[iz]
            end
            electron_kinetic_equation_euler_update!(
                (pdf_electron=residual_f, electron_p=residual_p), this_f, this_p, moments,
                z, vperp, vpa, z_spectral, vpa_spectral, z_advect, vpa_advect,
                scratch_dummy, collisions, composition, external_source_settings,
                num_diss_params, t_params.electron, ir; evolve_p=true, ion_dt=ion_dt)
            # Now
            #   residual = f_electron_old + dt*RHS(f_electron_newvar)
            # so update to desired residual
            @begin_anyzv_z_vperp_vpa_region()
            @loop_z_vperp_vpa iz ivperp ivpa begin
                residual_f[ivpa,ivperp,iz] = this_f[ivpa,ivperp,iz] - residual_f[ivpa,ivperp,iz]
            end
            @begin_anyzv_z_region()
            @loop_z iz begin
                residual_p[iz] = this_p[iz] - residual_p[iz]
            end

            # Set residual to zero where pdf_electron is determined by boundary conditions.
            if vpa.n > 1
                @begin_anyzv_z_vperp_region()
                @loop_z_vperp iz ivperp begin
                    @views enforce_v_boundary_condition_local!(residual_f[:,ivperp,iz], vpa.bc,
                                                               vpa_advect[1].speed[:,ivperp,iz,ir],
                                                               num_diss_params.electron.vpa_dissipation_coefficient > 0.0,
                                                               vpa, vpa_spectral)
                end
            end
            if vperp.n > 1
                @begin_anyzv_z_vpa_region()
                enforce_vperp_boundary_condition!(residual_f, vperp.bc,
                                                  vperp, vperp_spectral, vperp_adv,
                                                  vperp_diffusion, ir)
            end
            zero_z_boundary_condition_points(residual_f, z, vpa, moments, ir)
            return nothing
        end

        original_residual_f = allocate_shared_float(vpa, vperp, z)
        original_residual_p = allocate_shared_float(z)
        perturbed_residual_f = allocate_shared_float(vpa, vperp, z)
        perturbed_residual_p = allocate_shared_float(z)
        f_plus_delta_f = allocate_shared_float(vpa, vperp, z)
        f_with_delta_p = allocate_shared_float(vpa, vperp, z)
        @begin_anyzv_z_vperp_vpa_region()
        @loop_z_vperp_vpa iz ivperp ivpa begin
            f_plus_delta_f[ivpa,ivperp,iz] = f[ivpa,ivperp,iz] + delta_f[ivpa,ivperp,iz]
            f_with_delta_p[ivpa,ivperp,iz] = f[ivpa,ivperp,iz]
        end
        p_plus_delta_p = allocate_shared_float(z)
        @begin_anyzv_z_region()
        @loop_z iz begin
            p_plus_delta_p[iz] = p[iz] + delta_p[iz]
        end

        @testset "δf only" begin
            residual_func!(original_residual_f, original_residual_p, f, p)
            residual_func!(perturbed_residual_f, perturbed_residual_p, f_plus_delta_f, p)

            @begin_anyzv_region()
            @anyzv_serial_region begin
                delta_state = zeros(mk_float, total_size)
                # Take this difference rather than using delta_f directly because we need
                # the effect of the boundary condition having been applied to
                # f_plus_delta_f.
                delta_state[1:pdf_size] .= vec(f_plus_delta_f .- f)
                residual_update_with_Jacobian = jacobian.matrix * delta_state
                perturbed_with_Jacobian_f = vec(original_residual_f) .+ residual_update_with_Jacobian[1:pdf_size]
                perturbed_with_Jacobian_p = vec(original_residual_p) .+ residual_update_with_Jacobian[pdf_size+1:end]

                norm_factor_f = generate_norm_factor(perturbed_residual_f)
                @test elementwise_isapprox(perturbed_residual_f ./ norm_factor_f,
                                           reshape(perturbed_with_Jacobian_f, vpa.n, vperp.n, z.n) ./ norm_factor_f;
                                           rtol=0.0, atol=rtol)
                norm_factor_p = generate_norm_factor(perturbed_residual_p)
                @test elementwise_isapprox(perturbed_residual_p ./ norm_factor_p,
                                           perturbed_with_Jacobian_p ./ norm_factor_p;
                                           rtol=0.0, atol=rtol)
            end
        end

        @testset "δp only" begin
            residual_func!(original_residual_f, original_residual_p, f, p)
            residual_func!(perturbed_residual_f, perturbed_residual_p, f_with_delta_p, p_plus_delta_p)

            @begin_anyzv_region()
            @anyzv_serial_region begin
                delta_state = zeros(mk_float, total_size)
                # Take this difference rather than using delta_f directly because we need
                # the effect of the boundary condition having been applied to
                # f_with_delta_p.
                delta_state[1:pdf_size] .= vec(f_with_delta_p .- f)
                delta_state[pdf_size+1:end] .= vec(delta_p)
                residual_update_with_Jacobian = jacobian.matrix * delta_state
                perturbed_with_Jacobian_f = vec(original_residual_f) .+ residual_update_with_Jacobian[1:pdf_size]
                perturbed_with_Jacobian_p = vec(original_residual_p) .+ residual_update_with_Jacobian[pdf_size+1:end]

                norm_factor_f = generate_norm_factor(perturbed_residual_f)
                @test elementwise_isapprox(perturbed_residual_f ./ norm_factor_f,
                                           reshape(perturbed_with_Jacobian_f, vpa.n, vperp.n, z.n) ./ norm_factor_f;
                                           rtol=0.0, atol=rtol)
                norm_factor_p = generate_norm_factor(perturbed_residual_p)
                @test elementwise_isapprox(perturbed_residual_p ./ norm_factor_p,
                                           perturbed_with_Jacobian_p ./ norm_factor_p;
                                           rtol=0.0, atol=rtol)
            end
        end

        @testset "δf and δp" begin
            residual_func!(original_residual_f, original_residual_p, f, p)
            residual_func!(perturbed_residual_f, perturbed_residual_p, f_plus_delta_f, p_plus_delta_p)

            @begin_anyzv_region()
            @anyzv_serial_region begin
                delta_state = zeros(mk_float, total_size)
                # Take this difference rather than using delta_f directly because we need
                # the effect of the boundary condition having been applied to
                # f_plus_delta_f.
                delta_state[1:pdf_size] .= vec(f_plus_delta_f .- f)
                delta_state[pdf_size+1:end] .= vec(delta_p)
                residual_update_with_Jacobian = jacobian.matrix * delta_state
                perturbed_with_Jacobian_f = vec(original_residual_f) .+ residual_update_with_Jacobian[1:pdf_size]
                perturbed_with_Jacobian_p = vec(original_residual_p) .+ residual_update_with_Jacobian[pdf_size+1:end]

                norm_factor_f = generate_norm_factor(perturbed_residual_f)
                @test elementwise_isapprox(perturbed_residual_f ./ norm_factor_f,
                                           reshape(perturbed_with_Jacobian_f, vpa.n, vperp.n, z.n) ./ norm_factor_f;
                                           rtol=0.0, atol=rtol)
                norm_factor_p = generate_norm_factor(perturbed_residual_p)
                @test elementwise_isapprox(perturbed_residual_p ./ norm_factor_p,
                                           perturbed_with_Jacobian_p ./ norm_factor_p;
                                           rtol=0.0, atol=rtol)
            end
        end

        cleanup_mk_state!(ascii_io, io_moments, io_dfns)
    end

    return nothing
end

function test_electron_wall_bc(test_input; atol=(10.0*epsilon)^2)
    test_input = deepcopy(test_input)
    test_input["output"]["run_name"] *= "_electron_wall_bc"
    println("    - electron_wall_bc")

    # This test only affects the end-points in z, so only include those points to avoid an
    # over-optimistic error estimate due the time update matrix for all other z-indices
    # just being the identity.
    test_input["z"]["nelement"] = 1
    test_input["z"]["ngrid"] = 2
    test_input["z"]["bc"] = "wall"

    # Interpolation done during the boundary condition needs to be reasonably accurate for
    # the simplified form (without constraints) that is done in the 'Jacobian matrix' to
    # match the full version, so increase vpa resolution.
    test_input["vpa"]["nelement"] = 256
    test_input["vz"]["nelement"] = 256

    @testset "electron_wall_bc" begin
        # Suppress console output while running
        pdf, scratch, scratch_implicit, scratch_electron, t_params, vz, vr, vzeta, vpa,
            vperp, gyrophase, z, r, moments, fields, spectral_objects, advection_structs,
            composition, collisions, geometry, gyroavs, boundary_distributions,
            external_source_settings, num_diss_params, nl_solver_params, advance,
            advance_implicit, fp_arrays, scratch_dummy, manufactured_source_list,
            ascii_io, io_moments, io_dfns = get_mk_state(test_input)

        dens = @view moments.electron.dens[:,ir]
        upar = @view moments.electron.upar[:,ir]
        p = @view moments.electron.p[:,ir]
        vth = @view moments.electron.vth[:,ir]
        qpar = @view moments.electron.qpar[:,ir]
        ion_dens = @view moments.ion.dens[:,ir]
        ion_upar = @view moments.ion.upar[:,ir]
        ddens_dz = @view moments.electron.ddens_dz[:,ir]
        dp_dz = @view moments.electron.dp_dz[:,ir]
        phi = @view fields.phi[:,ir]
        z_spectral = spectral_objects.z_spectral
        vperp_spectral = spectral_objects.vperp_spectral
        vpa_spectral = spectral_objects.vpa_spectral
        z_advect = advection_structs.z_advect
        vpa_advect = advection_structs.vpa_advect
        me = composition.me_over_mi

        @begin_r_anyzv_region()

        buffer_1 = @view scratch_dummy.buffer_rs_1[ir,1]
        buffer_2 = @view scratch_dummy.buffer_rs_2[ir,1]
        buffer_3 = @view scratch_dummy.buffer_rs_3[ir,1]
        buffer_4 = @view scratch_dummy.buffer_rs_4[ir,1]

        v_size = vperp.n * vpa.n

        # Reconstruct w_∥^3 moment of g_e from already-calculated qpar
        third_moment = scratch_dummy.buffer_z_1
        dthird_moment_dz = scratch_dummy.buffer_z_2
        @begin_anyzv_z_region()
        @loop_z iz begin
            third_moment[iz] = qpar[iz] / p[iz] / vth[iz]
        end
        derivative_z_anyzv!(dthird_moment_dz, third_moment, buffer_1, buffer_2, buffer_3,
                            buffer_4, z_spectral, z)

        @begin_anyzv_vperp_vpa_region()
        update_electron_speed_z!(z_advect[1], upar, vth, vpa.grid, ir)
        z_speed = @view z_advect[1].speed[:,:,:,ir]

        delta_p = allocate_shared_float(z)
        p_amplitude = epsilon * maximum(p)
        f = @view pdf.electron.norm[:,:,:,ir]
        @begin_anyzv_region()
        @anyzv_serial_region begin
            # Make sure initial condition has some z-variation. As f is 'moment kinetic' this
            # means f must have a non-Maxwellian part that varies in z.
            f .*= 1.0 .+ 1.0e-4 .* reshape(vpa.grid.^3, vpa.n, 1, 1) .* reshape(sin.(2.0.*π.*z.grid./z.L), 1, 1, z.n)
        end
        # Ensure initial electron distribution function obeys constraints
        hard_force_moment_constraints!(reshape(f, vpa.n, vperp.n, z.n, 1), moments, vpa, vperp)
        @begin_r_anyzv_region()
        # enforce the boundary condition(s) on the electron pdf
        @views enforce_boundary_condition_on_electron_pdf!(
                   f, phi, vth, upar, z, vperp, vpa, vperp_spectral, vpa_spectral,
                   vpa_advect, moments,
                   num_diss_params.electron.vpa_dissipation_coefficient > 0.0,
                   composition.me_over_mi, ir; bc_constraints=false, update_vcut=false)
        delta_f = allocate_shared_float(vpa, vperp, z)
        f_amplitude = epsilon * maximum(f)
        # Use exp(sin()) in vpa so that perturbation does not have any symmetry that makes
        # low-order moments vanish exactly.
        # For this test have no z-dependence in delta_f so that it does not vanish
        # at the z-boundaries
        @begin_anyzv_region()
        @anyzv_serial_region begin
            @. delta_p = p_amplitude

            delta_f .= f_amplitude .*
                       reshape(exp.(sin.(2.0.*π.*test_wavenumber.*vpa.grid./vpa.L)) .- 1.0, vpa.n, 1, 1) .*
                       f
        end

        pdf_size = length(f)
        p_size = length(p)
        total_size = pdf_size + p_size

        dpdf_dvpa = @view scratch_dummy.buffer_vpavperpzr_2[:,:,:,ir]
        @begin_anyzv_z_vperp_region()
        update_electron_speed_vpa!(vpa_advect[1], dens, upar, p, moments,
                                   composition.me_over_mi, vpa.grid,
                                   external_source_settings.electron, ir)
        @loop_z_vperp iz ivperp begin
            @views @. vpa_advect[1].adv_fac[:,ivperp,iz,ir] = -vpa_advect[1].speed[:,ivperp,iz,ir]
        end
        #calculate the upwind derivative of the electron pdf w.r.t. wpa
        @loop_z_vperp iz ivperp begin
            @views derivative!(dpdf_dvpa[:,ivperp,iz], f[:,ivperp,iz], vpa,
                               vpa_advect[1].adv_fac[:,ivperp,iz,ir], vpa_spectral)
        end

        jacobian = create_jacobian_info((; vpa=vpa, vperp=vperp, z=z),
                                        (; vpa=vpa_spectral, vperp=vperp_spectral,
                                         z=z_spectral);
                                        comm=comm_anyzv_subblock[],
                                        synchronize=_anyzv_subblock_synchronize,
                                        electron_pdf=((:anyzv,:z,:vperp,:vpa), (:vpa, :vperp, :z), false),
                                        electron_p=((:anyzv,:z), (:z,), false),
                                        boundary_skip_funcs=(electron_pdf=skip_f_electron_bc_points_in_Jacobian,
                                                             electron_p=nothing))
        jacobian_initialize_identity!(jacobian)

        add_wall_boundary_condition_to_Jacobian!(
            jacobian, phi, f, p, vth, upar, z, vperp, vpa, vperp_spectral, vpa_spectral,
            vpa_advect, moments, num_diss_params.electron.vpa_dissipation_coefficient, me,
            ir, :all)

        # Test 'ADI Jacobians' before other tests, because residual_func() may modify some
        # variables (vth, etc.).

        jacobian_ADI_check = create_jacobian_info((; vpa=vpa, vperp=vperp, z=z),
                                                  (; vpa=vpa_spectral,
                                                   vperp=vperp_spectral, z=z_spectral);
                                                  comm=comm_anyzv_subblock[],
                                                  synchronize=_anyzv_subblock_synchronize,
                                                  electron_pdf=((:anyzv,:z,:vperp,:vpa), (:vpa, :vperp, :z), false),
                                                  electron_p=((:anyzv,:z), (:z,), false),
                                                  boundary_skip_funcs=(electron_pdf=skip_f_electron_bc_points_in_Jacobian,
                                                                       electron_p=nothing))
        v_solve_jacobian_ADI_check = create_jacobian_info((; vpa=vpa, vperp=vperp),
                                                          (; vpa=vpa_spectral,
                                                           vperp=vperp_spectral);
                                                          comm=nothing,
                                                          synchronize=nothing,
                                                          electron_pdf=(nothing, (:vpa, :vperp), false),
                                                          electron_p=(nothing, (), false),
                                                          boundary_skip_funcs=(electron_pdf=skip_f_electron_bc_points_in_Jacobian_v_solve,
                                                                               electron_p=nothing))

        @testset "ADI Jacobians - implicit z" begin
            # 'Implicit' and 'explicit' parts of Jacobian should add up to full Jacobian.
            jacobian_initialize_identity!(jacobian_ADI_check)

            # There is no 'implicit z' contribution for wall bc

            # Add 'explicit' contribution
            add_wall_boundary_condition_to_Jacobian!(
                jacobian_ADI_check, phi, f, p, vth, upar, z, vperp, vpa, vperp_spectral,
                vpa_spectral, vpa_advect, moments,
                num_diss_params.electron.vpa_dissipation_coefficient, me, ir, :explicit_v)
            @_anyzv_subblock_synchronize()

            @begin_anyzv_region()
            @anyzv_serial_region begin
                @test elementwise_isapprox(jacobian_ADI_check.matrix, jacobian.matrix; rtol=0.0, atol=1.0e-15)
            end
            @_anyzv_subblock_synchronize()
        end

        @testset "ADI Jacobians - implicit v" begin
            # 'Implicit' and 'explicit' parts of Jacobian should add up to full Jacobian.
            jacobian_initialize_identity!(jacobian_ADI_check)

            v_size = vperp.n * vpa.n

            # Add 'implicit' contribution
            @begin_anyzv_z_region()
            @loop_z iz begin
                this_slice = collect((iz - 1)*v_size + 1:iz*v_size)
                push!(this_slice, iz + pdf_size)

                # We are reusing v_solve_jacobian_ADI_check, so need to zero out its
                # matrix.
                v_solve_jacobian_ADI_check.matrix .= 0.0

                @views add_wall_boundary_condition_to_Jacobian!(
                    v_solve_jacobian_ADI_check, phi[iz], f[:,:,iz], p[iz], vth[iz],
                    upar[iz], z, vperp, vpa, vperp_spectral, vpa_spectral, vpa_advect,
                    moments, num_diss_params.electron.vpa_dissipation_coefficient, me, ir,
                    :implicit_v, iz)
                @views jacobian_ADI_check.matrix[this_slice,this_slice] .+= v_solve_jacobian_ADI_check.matrix
            end
            @_anyzv_subblock_synchronize()

            # Add 'explicit' contribution
            add_wall_boundary_condition_to_Jacobian!(
                jacobian_ADI_check, phi, f, p, vth, upar, z, vperp, vpa, vperp_spectral,
                vpa_spectral, vpa_advect, moments,
                num_diss_params.electron.vpa_dissipation_coefficient, me, ir, :explicit_z)
            @_anyzv_subblock_synchronize()

            @begin_anyzv_region()
            @anyzv_serial_region begin
                @test elementwise_isapprox(jacobian_ADI_check.matrix, jacobian.matrix; rtol=0.0, atol=1.0e-15)
            end
        end

        function residual_func!(residual, this_f, this_p)
            @begin_anyzv_z_region()
            # Calculate derived moments and derivatives using new_variables
            calculate_electron_moments_no_r!(this_f, dens, upar, this_p, ion_dens,
                                             ion_upar, moments, composition, collisions,
                                             r, z, vperp, vpa, ir)
            calculate_electron_moment_derivatives_no_r!(
                moments, dens, upar, this_p, scratch_dummy, z, z_spectral,
                num_diss_params.electron.moment_dissipation_coefficient, ir)

            # enforce the boundary condition(s) on the electron pdf
            @views enforce_boundary_condition_on_electron_pdf!(
                       this_f, phi, vth, upar, z, vperp, vpa, vperp_spectral,
                       vpa_spectral, vpa_advect, moments,
                       num_diss_params.electron.vpa_dissipation_coefficient > 0.0,
                       composition.me_over_mi, ir; bc_constraints=false,
                       update_vcut=false)

            # electron_kinetic_equation_euler_update!() just adds dt*d(g_e)/dt to the
            # electron_pdf member of the first argument, so if we set the electron_pdf member
            # of the first argument to zero, and pass dt=1, then it will evaluate the time
            # derivative, which is the residual for a steady-state solution.
            @begin_anyzv_z_vperp_vpa_region()
            @loop_z_vperp_vpa iz ivperp ivpa begin
                residual[ivpa,ivperp,iz] = f[ivpa,ivperp,iz]
            end
            # Now
            #   residual = f_electron_old + dt*RHS(f_electron_newvar)
            # so update to desired residual
            @begin_anyzv_z_vperp_vpa_region()
            @loop_z_vperp_vpa iz ivperp ivpa begin
                residual[ivpa,ivperp,iz] = this_f[ivpa,ivperp,iz] - residual[ivpa,ivperp,iz]
            end

            # Set residual to zero where pdf_electron is determined by boundary conditions.
            if vpa.n > 1
                @begin_anyzv_z_vperp_region()
                @loop_z_vperp iz ivperp begin
                    @views enforce_v_boundary_condition_local!(residual[:,ivperp,iz], vpa.bc,
                                                               vpa_advect[1].speed[:,ivperp,iz,ir],
                                                               num_diss_params.electron.vpa_dissipation_coefficient > 0.0,
                                                               vpa, vpa_spectral)
                end
            end
            if vperp.n > 1
                @begin_anyzv_z_vpa_region()
                enforce_vperp_boundary_condition!(residual, vperp.bc,
                                                  vperp, vperp_spectral, vperp_adv,
                                                  vperp_diffusion, ir)
            end
            if z.bc == "wall"
                zero_z_boundary_condition_points(residual, z, vpa, moments, ir)
            else
                error("Testing wall bc but z_bc != \"wall\".")
            end

            return nothing
        end

        original_residual = allocate_shared_float(vpa, vperp, z)
        perturbed_residual = allocate_shared_float(vpa, vperp, z)
        f_plus_delta_f = allocate_shared_float(vpa, vperp, z)
        f_with_delta_p = allocate_shared_float(vpa, vperp, z)
        @begin_anyzv_z_vperp_vpa_region()
        @loop_z_vperp_vpa iz ivperp ivpa begin
            f_plus_delta_f[ivpa,ivperp,iz] = f[ivpa,ivperp,iz] + delta_f[ivpa,ivperp,iz]
            f_with_delta_p[ivpa,ivperp,iz] = f[ivpa,ivperp,iz]
        end
        p_plus_delta_p = allocate_shared_float(z)
        @begin_anyzv_z_region()
        @loop_z iz begin
            p_plus_delta_p[iz] = p[iz] + delta_p[iz]
        end

        @testset "δf only" begin
            residual_func!(original_residual, f, p)
            residual_func!(perturbed_residual, f_plus_delta_f, p)

            @begin_anyzv_region()
            @anyzv_serial_region begin
                delta_state = zeros(mk_float, total_size)
                # Take this difference rather than using delta_f directly because we need
                # the effect of the boundary condition having been applied to
                # f_plus_delta_f.
                delta_state[1:pdf_size] .= vec(f_plus_delta_f .- f)
                residual_update_with_Jacobian = jacobian.matrix * delta_state
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[1:pdf_size]

                # Check p did not get perturbed by the Jacobian
                @test elementwise_isapprox(residual_update_with_Jacobian[pdf_size+1:end],
                                           zeros(p_size); atol=1.0e-15)

                # Check that something happened, to make sure that for example the
                # residual function and Jacobian don't both just zero out the boundary
                # points.
                @test norm(vec(perturbed_residual) .- perturbed_with_Jacobian) > 1.0e-12

                # If the boundary condition is correctly implemented in the Jacobian, then
                # if f+delta_f obeys the boundary condition, then J*delta_state should
                # give zeros in the boundary points.
                @test elementwise_isapprox(perturbed_residual,
                                           reshape(perturbed_with_Jacobian, vpa.n, vperp.n, z.n);
                                           rtol=0.0, atol=atol)
            end
        end

        @testset "δp only" begin
            residual_func!(original_residual, f, p)
            residual_func!(perturbed_residual, f_with_delta_p, p_plus_delta_p)

            @begin_anyzv_region()
            @anyzv_serial_region begin
                delta_state = zeros(mk_float, total_size)
                # Take this difference rather than using delta_f directly because we need
                # the effect of the boundary condition having been applied to
                # f_with_delta_p.
                delta_state[1:pdf_size] .= vec(f_with_delta_p .- f)
                delta_state[pdf_size+1:end] .= vec(delta_p)
                residual_update_with_Jacobian = jacobian.matrix * delta_state
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[1:pdf_size]

                # Check p did not get perturbed by the Jacobian
                @test elementwise_isapprox(residual_update_with_Jacobian[pdf_size+1:end],
                                           vec(delta_p); atol=1.0e-15)

                # Check that something happened, to make sure that for example the
                # residual function and Jacobian don't both just zero out the boundary
                # points.
                @test norm(vec(perturbed_residual) .- perturbed_with_Jacobian) > 1.0e-12

                # Use an absolute tolerance for this test because if we used a norm_factor
                # like the other tests, it would be zero to machine precision at some
                # points.
                @test elementwise_isapprox(perturbed_residual,
                                           reshape(perturbed_with_Jacobian, vpa.n, vperp.n, z.n);
                                           rtol=0.0, atol=atol)
            end
        end

        @testset "δf and δp" begin
            residual_func!(original_residual, f, p)
            residual_func!(perturbed_residual, f_plus_delta_f, p_plus_delta_p)

            @begin_anyzv_region()
            @anyzv_serial_region begin
                delta_state = zeros(mk_float, total_size)
                # Take this difference rather than using delta_f directly because we need
                # the effect of the boundary condition having been applied to
                # f_plus_delta_f.
                delta_state[1:pdf_size] .= vec(f_plus_delta_f .- f)
                delta_state[pdf_size+1:end] .= vec(delta_p)
                residual_update_with_Jacobian = jacobian.matrix * delta_state
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[1:pdf_size]

                # Check p did not get perturbed by the Jacobian
                @test elementwise_isapprox(residual_update_with_Jacobian[pdf_size+1:end],
                                           vec(delta_p); atol=1.0e-15)

                # Check that something happened, to make sure that for example the
                # residual function and Jacobian don't both just zero out the boundary
                # points.
                @test norm(vec(perturbed_residual) .- perturbed_with_Jacobian) > 1.0e-12

                # Use an absolute tolerance for this test because if we used a norm_factor
                # like the other tests, it would be zero to machine precision at some
                # points.
                @test elementwise_isapprox(perturbed_residual,
                                           reshape(perturbed_with_Jacobian, vpa.n, vperp.n, z.n);
                                           rtol=0.0, atol=atol)
            end
        end

        cleanup_mk_state!(ascii_io, io_moments, io_dfns)
    end

    return nothing
end

function runtests()
    if Sys.isapple() && "CI" ∈ keys(ENV) && global_size[] > 1
        # These tests are too slow in the parallel tests job on macOS, so skip in that
        # case.
        return nothing
    end
    # Create a temporary directory for test output
    test_output_directory = get_MPI_tempdir()
    test_input["output"]["base_directory"] = test_output_directory

    @testset "Jacobian matrix" verbose=use_verbose begin
        println("Jacobian matrix")

        # Quite large multipliers for rtol in these tests, but it is plausible that a
        # nonlinear error (∼epsilon^2) could be multiplied by
        # ∼vth*vpa.L/2∼sqrt(2)*60*6≈500.

        function z_advection_wrapper!(; kwargs...)
            electron_z_advection!(kwargs[:residual], kwargs[:this_f], kwargs[:upar],
                                  kwargs[:vth], kwargs[:z_advect], kwargs[:z],
                                  kwargs[:vpa].grid, kwargs[:z_spectral],
                                  kwargs[:scratch_dummy], kwargs[:dt], kwargs[:ir])
            return nothing
        end
        test_get_pdf_term(test_input, "electron_z_advection",
                          get_electron_z_advection_term, z_advection_wrapper!,
                          (2.5e2*epsilon)^2)

        function vpa_advection_wrapper!(; kwargs...)
            electron_vpa_advection!(kwargs[:residual], kwargs[:this_f], kwargs[:dens],
                                    kwargs[:upar], kwargs[:this_p], kwargs[:moments],
                                    kwargs[:composition], kwargs[:vpa_advect],
                                    kwargs[:vpa], kwargs[:vpa_spectral],
                                    kwargs[:scratch_dummy], kwargs[:dt],
                                    kwargs[:external_source_settings].electron,
                                    kwargs[:ir])
            return nothing
        end
        test_get_pdf_term(test_input, "electron_vpa_advection",
                          get_electron_vpa_advection_term, vpa_advection_wrapper!,
                          (3.0e2*epsilon)^2)

        function contribution_from_electron_pdf_term_wrapper!(; kwargs...)
            add_contribution_from_pdf_term!(kwargs[:residual], kwargs[:this_f],
                                            kwargs[:this_p], kwargs[:dens], kwargs[:upar],
                                            kwargs[:moments], kwargs[:vpa].grid,
                                            kwargs[:z], kwargs[:dt],
                                            kwargs[:external_source_settings].electron,
                                            kwargs[:ir])
            return nothing
        end
        test_get_pdf_term(test_input, "contribution_from_electron_pdf_term",
                          get_contribution_from_electron_pdf_term,
                          contribution_from_electron_pdf_term_wrapper!, (4.0e2*epsilon)^2)

        function contribution_from_electron_dissipation_term!(; kwargs...)
            add_dissipation_term!(kwargs[:residual], kwargs[:this_f],
                                  kwargs[:scratch_dummy], kwargs[:z_spectral], kwargs[:z],
                                  kwargs[:vpa], kwargs[:vpa_spectral],
                                  kwargs[:num_diss_params], kwargs[:dt])
            return nothing
        end
        test_get_pdf_term(test_input, "electron_dissipation_term",
                          get_electron_dissipation_term,
                          contribution_from_electron_dissipation_term!, (1.0e1*epsilon)^2)

        function contribution_from_krook_collisions!(; kwargs...)
            electron_krook_collisions!(kwargs[:residual], kwargs[:this_f], kwargs[:dens],
                                       kwargs[:upar], kwargs[:ion_upar], kwargs[:vth],
                                       kwargs[:collisions], kwargs[:vperp], kwargs[:vpa],
                                       kwargs[:dt])
            return nothing
        end
        test_get_pdf_term(test_input, "electron_krook_collisions",
                          get_electron_krook_collisions_term,
                          contribution_from_krook_collisions!, (2.0e1*epsilon)^2)

        function contribution_from_external_electron_sources!(; kwargs...)
            total_external_electron_sources!(kwargs[:residual], kwargs[:this_f],
                                             kwargs[:dens], kwargs[:upar],
                                             kwargs[:moments], kwargs[:composition],
                                             kwargs[:external_source_settings].electron,
                                             kwargs[:vperp], kwargs[:vpa], kwargs[:dt],
                                             kwargs[:ir])
            return nothing
        end
        test_get_pdf_term(test_input, "external_electron_sources",
                          get_total_external_electron_source_term,
                          contribution_from_external_electron_sources!, (3.0e1*epsilon)^2)

        # For this test where only the 'constraint forcing' term is added to the residual,
        # the residual is exactly zero for the initial condition (because that is
        # constructed to obey the constraints). Therefore the 'perturbed_residual' is
        # non-zero only because of delta_f, which is small, O(epsilon), so 'norm_factor'
        # is also O(epsilon). We therefore use a tolerance of O(epsilon) in this test,
        # unlike the other tests which use a tolerance of O(epsilon^2). Note that in the
        # final test of the full electron kinetic equations, with all terms including this
        # one, we do not have a similar issue, as there the other terms create an O(1)
        # residual for the initial condition, which will then set the 'norm_factor'.
        #
        # We test the Jacobian for these constraint forcing terms using
        # constraint_forcing_rate=O(1), because in these tests we set dt=O(1), so a large
        # coefficient would make the non-linearity large and then it would be hard to
        # distinguish errors from non-linearity (or rounding errors) in
        # `test_electron_kinetic_equation()` that tests the combined effect of all terms
        # in the electron kinetic equation. This test would actually be OK because the
        # ratio of linear to non-linear contributions of this single term does not depend
        # on the size of the coefficient. In the combined test, we are effectively
        # comparing the non-linear error from this term to the residual from other terms,
        # so the coefficient of this term matters there. Even though these settings are
        # not what we would use in a real simulation, they should tell us if the
        # implementation is correct.
        function contribution_from_implicit_constraint_forcing!(; kwargs...)
            electron_implicit_constraint_forcing!(kwargs[:residual], kwargs[:this_f],
                                                  kwargs[:t_params].electron.constraint_forcing_rate,
                                                  kwargs[:vperp], kwargs[:vpa],
                                                  kwargs[:dt], kwargs[:ir])
            return nothing
        end
        test_get_pdf_term(test_input, "implicit_constraint_forcing",
                          get_electron_implicit_constraint_forcing_term,
                          contribution_from_implicit_constraint_forcing!, (2.5e0*epsilon))

        function contribution_from_electron_energy_equation!(; kwargs...)
            electron_energy_equation_no_r!(
                kwargs[:residual], kwargs[:dens], kwargs[:this_p], kwargs[:dens],
                kwargs[:upar], kwargs[:ppar], kwargs[:ion_dens], kwargs[:ion_upar],
                kwargs[:ion_p], kwargs[:neutral_dens], kwargs[:neutral_uz],
                kwargs[:neutral_p], kwargs[:moments].electron, kwargs[:collisions],
                kwargs[:dt], kwargs[:composition],
                kwargs[:external_source_settings].electron, kwargs[:num_diss_params],
                kwargs[:z], kwargs[:ir])
            return nothing
        end
        test_get_p_term(test_input, "electron_energy_equation",
                        get_electron_energy_equation_term,
                        contribution_from_electron_energy_equation!, (6.0e2*epsilon)^2)

        function contribution_from_ion_dt_forcing_of_electron_p!(; kwargs...)
            p_previous_ion_step = kwargs[:moments].electron.p
            residual = kwargs[:residual]
            this_p = kwargs[:this_p]
            ir = kwargs[:ir]
            ion_dt = kwargs[:ion_dt]
            @begin_anyzv_z_region()
            @loop_z iz begin
                # At this point, p_out = p_in + dt*RHS(p_in). Here we add a source/damping
                # term so that in the steady state of the electron pseudo-timestepping
                # iteration,
                #   RHS(p) - (p - p_previous_ion_step) / ion_dt = 0,
                # resulting in a backward-Euler step (as long as the pseudo-timestepping
                # loop converges).
                residual[iz] += -dt * (this_p[iz] - p_previous_ion_step[iz,ir]) / ion_dt
            end
            return nothing
        end
        test_get_p_term(test_input, "ion_dt_forcing_of_electron_p",
                        get_ion_dt_forcing_of_electron_p_term,
                        contribution_from_ion_dt_forcing_of_electron_p!,
                        (1.5e1*epsilon)^2)

        test_electron_wall_bc(test_input)

        test_electron_kinetic_equation(test_input)
    end

    if global_rank[] == 0
        # Delete output directory to avoid using too much disk space
        rm(realpath(test_output_directory); recursive=true)
    end

    return nothing
end

end # JacobianMatrixTests


using .JacobianMatrixTests

JacobianMatrixTests.runtests()
