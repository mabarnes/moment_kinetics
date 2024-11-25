module JacobianMatrixTests

# Tests for construction of Jacobian matrices used for preconditioning

include("setup.jl")

using moment_kinetics: setup_moment_kinetics, cleanup_moment_kinetics!
using moment_kinetics.analysis: vpagrid_to_dzdt
using moment_kinetics.array_allocation: allocate_shared_float
using moment_kinetics.boundary_conditions: enforce_v_boundary_condition_local!,
                                           enforce_vperp_boundary_condition!,
                                           skip_f_electron_bc_points_in_Jacobian
using moment_kinetics.calculus: derivative!
using moment_kinetics.derivatives: derivative_z!, derivative_z_pdf_vpavperpz!
using moment_kinetics.electron_fluid_equations: calculate_electron_qpar_from_pdf_no_r!,
                                                electron_energy_equation_no_r!,
                                                add_electron_energy_equation_to_Jacobian!,
                                                add_electron_energy_equation_to_z_only_Jacobian!,
                                                add_electron_energy_equation_to_v_only_Jacobian!
using moment_kinetics.electron_kinetic_equation: add_contribution_from_pdf_term!,
                                                 add_contribution_from_electron_pdf_term_to_Jacobian!,
                                                 add_contribution_from_electron_pdf_term_to_z_only_Jacobian!,
                                                 add_contribution_from_electron_pdf_term_to_v_only_Jacobian!,
                                                 add_dissipation_term!,
                                                 add_electron_dissipation_term_to_Jacobian!,
                                                 add_electron_dissipation_term_to_v_only_Jacobian!,
                                                 add_ion_dt_forcing_of_electron_ppar_to_Jacobian!,
                                                 add_ion_dt_forcing_of_electron_ppar_to_z_only_Jacobian!,
                                                 add_ion_dt_forcing_of_electron_ppar_to_v_only_Jacobian!,
                                                 electron_kinetic_equation_euler_update!,
                                                 enforce_boundary_condition_on_electron_pdf!,
                                                 fill_electron_kinetic_equation_Jacobian!,
                                                 fill_electron_kinetic_equation_v_only_Jacobian!,
                                                 fill_electron_kinetic_equation_z_only_Jacobian_f!,
                                                 fill_electron_kinetic_equation_z_only_Jacobian_ppar!,
                                                 add_wall_boundary_condition_to_Jacobian!
using moment_kinetics.electron_vpa_advection: electron_vpa_advection!,
                                              update_electron_speed_vpa!,
                                              add_electron_vpa_advection_to_Jacobian!,
                                              add_electron_vpa_advection_to_v_only_Jacobian!
using moment_kinetics.electron_z_advection: electron_z_advection!,
                                            update_electron_speed_z!,
                                            add_electron_z_advection_to_Jacobian!,
                                            add_electron_z_advection_to_z_only_Jacobian!,
                                            add_electron_z_advection_to_v_only_Jacobian!
using moment_kinetics.external_sources: total_external_electron_sources!,
                                        add_total_external_electron_source_to_Jacobian!,
                                        add_total_external_electron_source_to_z_only_Jacobian!,
                                        add_total_external_electron_source_to_v_only_Jacobian!
using moment_kinetics.krook_collisions: electron_krook_collisions!,
                                        add_electron_krook_collisions_to_Jacobian!,
                                        add_electron_krook_collisions_to_z_only_Jacobian!,
                                        add_electron_krook_collisions_to_v_only_Jacobian!
using moment_kinetics.looping
using moment_kinetics.moment_constraints: electron_implicit_constraint_forcing!,
                                          add_electron_implicit_constraint_forcing_to_Jacobian!,
                                          add_electron_implicit_constraint_forcing_to_z_only_Jacobian!,
                                          add_electron_implicit_constraint_forcing_to_v_only_Jacobian!,
                                          hard_force_moment_constraints!
using moment_kinetics.timer_utils: reset_mk_timers!
using moment_kinetics.type_definitions: mk_float
using moment_kinetics.velocity_moments: calculate_electron_moment_derivatives_no_r!,
                                        integrate_over_vspace

using StatsBase

# Small parameter used to create perturbations to test Jacobian against
epsilon = 1.0e-6
test_wavenumber = 2.0
dt = 0.42
ion_dt = 1.0e-6
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
test_input = OptionsDict("output" => OptionsDict("run_name" => "jacobian_matrix",
                                                ),
                         "composition" => OptionsDict("n_ion_species" => 1,
                                                      "n_neutral_species" => 1,
                                                      "electron_physics" => "kinetic_electrons",
                                                      "recycling_fraction" => 0.5,
                                                      "T_e" => 1.0,
                                                      "T_wall" => 0.1,
                                                     ),
                         "evolve_moments" => OptionsDict("density" => true,
                                                         "parallel_flow" => true,
                                                         "parallel_pressure" => true,
                                                         "moments_conservation" => true,
                                                        ),
                         "ion_species_1" => OptionsDict("initial_density" => 1.0,
                                                        "initial_temperature" => 1.0,
                                                       ),
                         "z_IC_ion_species_1" => OptionsDict("initialization_option" => "sinusoid",
                                                             "density_amplitude" => 0.1,
                                                             "density_phase" => mk_float(π),
                                                             "upar_amplitude" => 0.1,
                                                             "upar_phase" => mk_float(π),
                                                             "temperature_amplitude" => 0.1,
                                                             "temperature_phase" => mk_float(π),
                                                            ),
                         "vpa_IC_ion_species_1" => OptionsDict("initialization_option" => "gaussian",
                                                               "density_amplitude" => 1.0,
                                                               "density_phase" => 0.0,
                                                               "upar_amplitude" => 0.0,
                                                               "upar_phase" => 0.0,
                                                               "temperature_amplitude" => 0.0,
                                                               "temperature_phase" => 0.0,
                                                              ),
                         "neutral_species_1" => OptionsDict("initial_density" => 1.0,
                                                            "initial_temperature" => 1.0,
                                                           ),
                         "z_IC_neutral_species_1" => OptionsDict("initialization_option" => "sinusoid",
                                                                 "density_amplitude" => 0.001,
                                                                 "density_phase" => mk_float(π),
                                                                 "upar_amplitude" => 0.0,
                                                                 "upar_phase" => mk_float(π),
                                                                 "temperature_amplitude" => 0.0,
                                                                 "temperature_phase" => mk_float(π),
                                                                ),
                         "vz_IC_neutral_species_1" => OptionsDict("initialization_option" => "gaussian",
                                                                   "density_amplitude" => 1.0,
                                                                   "density_phase" => 0.0,
                                                                   "upar_amplitude" => 0.0,
                                                                   "upar_phase" => 0.0,
                                                                   "temperature_amplitude" => 0.0,
                                                                   "temperature_phase" => 0.0,
                                                                  ),
                         "reactions" => OptionsDict("charge_exchange_frequency" => 0.75,
                                                    "ionization_frequency" => 0.0,
                                                   ),
                         "r" => OptionsDict("ngrid" => 1,
                                            "nelement" => 1,
                                           ),
                         "z" => OptionsDict("ngrid" => 9,
                                            "nelement" => 16,
                                            "bc" => "constant",
                                            "discretization" => "gausslegendre_pseudospectral",
                                           ),
                         "vpa" => OptionsDict("ngrid" => 6,
                                              "nelement" => 31,
                                              "L" => 12.0,
                                              "bc" => "zero",
                                              "discretization" => "gausslegendre_pseudospectral",
                                              "element_spacing_option" => "coarse_tails",
                                             ),
                         "vz" => OptionsDict("ngrid" => 6,
                                             "nelement" => 31,
                                             "L" => 12.0,
                                             "bc" => "zero",
                                             "discretization" => "gausslegendre_pseudospectral",
                                             "element_spacing_option" => "coarse_tails",
                                            ),
                         "timestepping" => OptionsDict("type" => "KennedyCarpenterARK324",
                                                       "implicit_electron_advance" => false,
                                                       "implicit_electron_ppar" => true,
                                                       "implicit_ion_advance" => false,
                                                       "implicit_vpa_advection" => false,
                                                       "nstep" => 1,
                                                       "dt" => ion_dt,
                                                       "minimum_dt" => 1.0e-7,
                                                       "rtol" => 1.0e-4,
                                                       "max_increase_factor_near_last_fail" => 1.001,
                                                       "last_fail_proximity_factor" => 1.1,
                                                       "max_increase_factor" => 1.05,
                                                       "nwrite" => 10000,
                                                       "nwrite_dfns" => 10000,
                                                       "steady_state_residual" => true,
                                                       "converged_residual_value" => 1.0e-3,
                                                      ),
                         "electron_timestepping" => OptionsDict("nstep" => 1,
                                                                "dt" => dt,
                                                                "maximum_dt" => 1.0,
                                                                "nwrite" => 10000,
                                                                "nwrite_dfns" => 100000,
                                                                "type" => "Fekete4(3)",
                                                                "rtol" => 1.0e-6,
                                                                "atol" => 1.0e-14,
                                                                "minimum_dt" => 1.0e-10,
                                                                "initialization_residual_value" => 2.5,
                                                                "converged_residual_value" => 1.0e-2,
                                                                "constraint_forcing_rate" => 2.321,
                                                               ),
                         "nonlinear_solver" => OptionsDict("nonlinear_max_iterations" => 100,
                                                           "rtol" => 1.0e-5,
                                                           "atol" => 1.0e-15,
                                                           "preconditioner_update_interval" => 1,
                                                          ),
                         "ion_numerical_dissipation" => OptionsDict("vpa_dissipation_coefficient" => 1.0e0,
                                                                    "force_minimum_pdf_value" => 0.0,
                                                                   ),
                         "electron_numerical_dissipation" => OptionsDict("vpa_dissipation_coefficient" => 2.0,
                                                                         "force_minimum_pdf_value" => 0.0,
                                                                        ),
                         "neutral_numerical_dissipation" => OptionsDict("vz_dissipation_coefficient" => 1.0e-1,
                                                                        "force_minimum_pdf_value" => 0.0,
                                                                       ),
                         "ion_source_1" => OptionsDict("active" => true,
                                                       "z_profile" => "gaussian",
                                                       "z_width" => 0.125,
                                                       "source_strength" => 0.1,
                                                       "source_T" => 2.0,
                                                      ),
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

# Quite a large multiplier in rtol for this test, but it is plausible that a nonlinear
# error (∼epsilon^2) could be multiplied by ∼vth*vpa.L/2∼sqrt(2)*60*6≈500.
function test_electron_z_advection(test_input; rtol=(2.5e2*epsilon)^2)
    test_input = deepcopy(test_input)
    test_input["output"]["run_name"] *= "_electron_z_advection"
    println("    - electron_z_advection")

    @testset "electron_z_advection" begin
        # Suppress console output while running
        pdf, scratch, scratch_implicit, scratch_electron, t_params, vz, vr, vzeta, vpa,
            vperp, gyrophase, z, r, moments, fields, spectral_objects, advection_structs,
            composition, collisions, geometry, gyroavs, boundary_distributions,
            external_source_settings, num_diss_params, nl_solver_params, advance,
            advance_implicit, fp_arrays, scratch_dummy, manufactured_source_list,
            ascii_io, io_moments, io_dfns = get_mk_state(test_input)

        dens = @view moments.electron.dens[:,ir]
        upar = @view moments.electron.upar[:,ir]
        ppar = @view moments.electron.ppar[:,ir]
        vth = @view moments.electron.vth[:,ir]
        qpar = @view moments.electron.qpar[:,ir]
        z_spectral = spectral_objects.z_spectral
        vpa_spectral = spectral_objects.vpa_spectral
        z_advect = advection_structs.z_advect
        vpa_advect = advection_structs.vpa_advect
        me = composition.me_over_mi

        delta_p = allocate_shared_float(size(ppar)...)
        p_amplitude = epsilon * maximum(ppar)
        f = @view pdf.electron.norm[:,:,:,ir]
        begin_serial_region()
        @serial_region begin
            @. delta_p = p_amplitude * sin(2.0*π*test_wavenumber*z.grid/z.L)

            # Make sure initial condition has some z-variation. As f is 'moment kinetic' this
            # means f must have a non-Maxwellian part that varies in z.
            f .*= 1.0 .+ 1.0e-4 .* reshape(vpa.grid.^3, vpa.n, 1, 1) .* reshape(sin.(2.0.*π.*z.grid./z.L), 1, 1, z.n)
            # Ensure initial electron distribution function obeys constraints
        end
        hard_force_moment_constraints!(reshape(f, vpa.n, vperp.n, z.n, 1), moments, vpa)
        delta_f = allocate_shared_float(size(f)...)
        f_amplitude = epsilon * maximum(f)
        # Use exp(sin()) in vpa so that perturbation does not have any symmetry that makes
        # low-order moments vanish exactly.
        begin_serial_region()
        @serial_region begin
            delta_f .= f_amplitude .*
                       reshape(sin.(2.0.*π.*test_wavenumber.*z.grid./z.L), 1, 1, z.n) .*
                       reshape(exp.(sin.(2.0.*π.*test_wavenumber.*vpa.grid./vpa.L)) .- 1.0, vpa.n, 1, 1) .*
                       f
        end

        pdf_size = length(f)
        p_size = length(ppar)
        total_size = pdf_size + p_size

        z_speed = @view z_advect[1].speed[:,:,:,ir]

        dpdf_dz = @view scratch_dummy.buffer_vpavperpzr_1[:,:,:,ir]
        begin_vperp_vpa_region()
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

        jacobian_matrix = allocate_shared_float(total_size, total_size)
        @serial_region begin
            jacobian_matrix .= 0.0
            for row ∈ 1:total_size
                # Initialise identity matrix
                jacobian_matrix[row,row] = 1.0
            end
        end

        add_electron_z_advection_to_Jacobian!(
            jacobian_matrix, f, dens, upar, ppar, vth, dpdf_dz, me, z, vperp, vpa,
            z_spectral, z_advect, z_speed, scratch_dummy, dt, ir; ppar_offset=pdf_size)

        # Test 'ADI Jacobians' before other tests, because residual_func() may modify some
        # variables (vth, etc.).

        jacobian_matrix_ADI_check = allocate_shared_float(total_size, total_size)

        @testset "ADI Jacobians - implicit z" begin
            # 'Implicit' and 'explicit' parts of Jacobian should add up to full Jacobian.
            begin_serial_region()
            @serial_region begin
                jacobian_matrix_ADI_check .= 0.0
                for row ∈ 1:total_size
                    # Initialise identity matrix
                    jacobian_matrix_ADI_check[row,row] = 1.0
                end
            end

            v_size = vperp.n * vpa.n

            # Add 'implicit' contribution
            begin_vperp_vpa_region()
            @loop_vperp_vpa ivperp ivpa begin
                this_slice = (ivperp - 1)*vpa.n + ivpa:v_size:(z.n - 1)*v_size + (ivperp - 1)*vpa.n + ivpa
                @views add_electron_z_advection_to_z_only_Jacobian!(
                    jacobian_matrix_ADI_check[this_slice,this_slice], f[ivpa,ivperp,:],
                    dens, upar, ppar, vth, dpdf_dz[ivpa,ivperp,:], me, z, vperp, vpa,
                    z_spectral, z_advect, z_speed, scratch_dummy, dt, ir,
                    ivperp, ivpa)
            end

            # Add 'explicit' contribution
            add_electron_z_advection_to_Jacobian!(
                jacobian_matrix_ADI_check, f, dens, upar, ppar, vth, dpdf_dz, me, z,
                vperp, vpa, z_spectral, z_advect, z_speed, scratch_dummy, dt, ir,
                :explicit_v; ppar_offset=pdf_size)

            begin_serial_region()
            @serial_region begin
                @test elementwise_isapprox(jacobian_matrix_ADI_check, jacobian_matrix; rtol=0.0, atol=1.0e-15)
            end
        end

        @testset "ADI Jacobians - implicit v" begin
            # 'Implicit' and 'explicit' parts of Jacobian should add up to full Jacobian.
            begin_serial_region()
            @serial_region begin
                jacobian_matrix_ADI_check .= 0.0
                for row ∈ 1:total_size
                    # Initialise identity matrix
                    jacobian_matrix_ADI_check[row,row] = 1.0
                end
            end

            v_size = vperp.n * vpa.n

            # Add 'implicit' contribution
            begin_z_region()
            @loop_z iz begin
                this_slice = collect((iz - 1)*v_size + 1:iz*v_size)
                push!(this_slice, iz + pdf_size)
                @views add_electron_z_advection_to_v_only_Jacobian!(
                    jacobian_matrix_ADI_check[this_slice,this_slice], f[:,:,iz],
                    dens[iz], upar[iz], ppar[iz], vth[iz], dpdf_dz[:,:,iz], me, z, vperp,
                    vpa, z_spectral, z_advect, z_speed, scratch_dummy, dt, ir, iz)
            end

            # Add 'explicit' contribution
            add_electron_z_advection_to_Jacobian!(
                jacobian_matrix_ADI_check, f, dens, upar, ppar, vth, dpdf_dz, me, z,
                vperp, vpa, z_spectral, z_advect, z_speed, scratch_dummy, dt, ir,
                :explicit_z; ppar_offset=pdf_size)

            begin_serial_region()
            @serial_region begin
                @test elementwise_isapprox(jacobian_matrix_ADI_check, jacobian_matrix; rtol=0.0, atol=1.0e-15)
            end
        end

        function residual_func!(residual, this_f, this_p)
            begin_z_region()
            @loop_z iz begin
                # update the electron thermal speed using the updated electron
                # parallel pressure
                vth[iz] = sqrt(abs(2.0 * this_p[iz] /
                                   (dens[iz] * composition.me_over_mi)))
            end
            # Calculate heat flux and derivatives using new_variables
            calculate_electron_qpar_from_pdf_no_r!(qpar, this_p, vth, this_f, vpa, ir)

            calculate_electron_moment_derivatives_no_r!(
                moments,
                (electron_density=dens,
                 electron_upar=upar,
                 electron_ppar=this_p),
                scratch_dummy, z, z_spectral,
                num_diss_params.electron.moment_dissipation_coefficient, ir)

            # electron_kinetic_equation_euler_update!() just adds dt*d(g_e)/dt to the
            # electron_pdf member of the first argument, so if we set the electron_pdf member
            # of the first argument to zero, and pass dt=1, then it will evaluate the time
            # derivative, which is the residual for a steady-state solution.
            begin_z_vperp_vpa_region()
            @loop_z_vperp_vpa iz ivperp ivpa begin
                residual[ivpa,ivperp,iz] = f[ivpa,ivperp,iz]
            end
            electron_z_advection!(residual, this_f, upar, vth, z_advect, z, vpa.grid,
                                  z_spectral, scratch_dummy, dt, ir)
            # Now
            #   residual = f_electron_old + dt*RHS(f_electron_newvar)
            # so update to desired residual
            begin_z_vperp_vpa_region()
            @loop_z_vperp_vpa iz ivperp ivpa begin
                residual[ivpa,ivperp,iz] = this_f[ivpa,ivperp,iz] - residual[ivpa,ivperp,iz]
            end

            # Set residual to zero where pdf_electron is determined by boundary conditions.
            if vpa.n > 1
                begin_z_vperp_region()
                @loop_z_vperp iz ivperp begin
                    @views enforce_v_boundary_condition_local!(residual[:,ivperp,iz], vpa.bc,
                                                               vpa_advect[1].speed[:,ivperp,iz,ir],
                                                               num_diss_params.electron.vpa_dissipation_coefficient > 0.0,
                                                               vpa, vpa_spectral)
                end
            end
            if vperp.n > 1
                begin_z_vpa_region()
                enforce_vperp_boundary_condition!(residual, vperp.bc,
                                                  vperp, vperp_spectral, vperp_adv,
                                                  vperp_diffusion, ir)
            end
            if z.bc == "wall"
                error("z_bc = \"wall\" not supported here yet.")
            elseif (z.bc == "constant") && (z.irank == 0 || z.irank == z.nrank - 1)
                # Boundary conditions on incoming part of distribution function. Note
                # that as density, upar, ppar do not change in this implicit step,
                # f_electron_newvar, f_old, and residual should all be zero at exactly
                # the same set of grid points, so it is reasonable to zero-out
                # `residual` to impose the boundary condition. We impose this after
                # subtracting f_old in case rounding errors, etc. mean that at some
                # point f_old had a different boundary condition cut-off index.
                begin_vperp_vpa_region()
                v_unnorm = vpa.scratch
                zero = 1.0e-14
                if z.irank == 0
                    iz = 1
                    v_unnorm .= vpagrid_to_dzdt(vpa.grid, vth[iz], upar[iz], true, true)
                    @loop_vperp_vpa ivperp ivpa begin
                        if v_unnorm[ivpa] > -zero
                            residual[ivpa,ivperp,iz] = 0.0
                        end
                    end
                end
                if z.irank == z.nrank - 1
                    iz = z.n
                    v_unnorm .= vpagrid_to_dzdt(vpa.grid, vth[iz], upar[iz], true, true)
                    @loop_vperp_vpa ivperp ivpa begin
                        if v_unnorm[ivpa] < zero
                            residual[ivpa,ivperp,iz] = 0.0
                        end
                    end
                end
            end
            return nothing
        end

        original_residual = allocate_shared_float(size(f)...)
        perturbed_residual = allocate_shared_float(size(f)...)

        @testset "δf only" begin
            residual_func!(original_residual, f, ppar)
            residual_func!(perturbed_residual, f.+delta_f, ppar)

            begin_serial_region()
            @serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[1:pdf_size] .= vec(delta_f)
                residual_update_with_Jacobian = jacobian_matrix * delta_state
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[1:pdf_size]

                # Check ppar did not get perturbed by the Jacobian
                @test elementwise_isapprox(residual_update_with_Jacobian[pdf_size+1:end],
                                           zeros(p_size); atol=1.0e-15)

                norm_factor = generate_norm_factor(perturbed_residual)
                @test elementwise_isapprox(perturbed_residual ./ norm_factor,
                                           reshape(perturbed_with_Jacobian, vpa.n, vperp.n, z.n) ./ norm_factor;
                                           rtol=0.0, atol=rtol)
            end
        end

        @testset "δp only" begin
            residual_func!(original_residual, f, ppar)
            residual_func!(perturbed_residual, f, ppar .+ delta_p)

            begin_serial_region()
            @serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[pdf_size+1:end] .= vec(delta_p)
                residual_update_with_Jacobian = jacobian_matrix * delta_state
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[1:pdf_size]

                # Check ppar did not get perturbed by the Jacobian
                @test elementwise_isapprox(residual_update_with_Jacobian[pdf_size+1:end],
                                           delta_state[pdf_size+1:end]; atol=1.0e-15)

                norm_factor = generate_norm_factor(perturbed_residual)
                @test elementwise_isapprox(perturbed_residual ./ norm_factor,
                                           reshape(perturbed_with_Jacobian, vpa.n, vperp.n, z.n) ./ norm_factor;
                                           rtol=0.0, atol=rtol)
            end
        end

        @testset "δf and δp" begin
            residual_func!(original_residual, f, ppar)
            residual_func!(perturbed_residual, f.+delta_f, ppar.+delta_p)

            begin_serial_region()
            @serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[1:pdf_size] .= vec(delta_f)
                delta_state[pdf_size+1:end] .= vec(delta_p)
                residual_update_with_Jacobian = jacobian_matrix * delta_state
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[1:pdf_size]

                # Check ppar did not get perturbed by the Jacobian
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

function test_electron_vpa_advection(test_input; rtol=(3.0e2*epsilon)^2)
    test_input = deepcopy(test_input)
    test_input["output"]["run_name"] *= "_electron_vpa_advection"
    println("    - electron_vpa_advection")

    @testset "electron_vpa_advection" begin
        # Suppress console output while running
        pdf, scratch, scratch_implicit, scratch_electron, t_params, vz, vr, vzeta, vpa,
            vperp, gyrophase, z, r, moments, fields, spectral_objects, advection_structs,
            composition, collisions, geometry, gyroavs, boundary_distributions,
            external_source_settings, num_diss_params, nl_solver_params, advance,
            advance_implicit, fp_arrays, scratch_dummy, manufactured_source_list,
            ascii_io, io_moments, io_dfns = get_mk_state(test_input)

        dens = @view moments.electron.dens[:,ir]
        upar = @view moments.electron.upar[:,ir]
        ppar = @view moments.electron.ppar[:,ir]
        vth = @view moments.electron.vth[:,ir]
        qpar = @view moments.electron.qpar[:,ir]
        ddens_dz = @view moments.electron.ddens_dz[:,ir]
        dppar_dz = @view moments.electron.dppar_dz[:,ir]
        z_spectral = spectral_objects.z_spectral
        vpa_spectral = spectral_objects.vpa_spectral
        z_advect = advection_structs.z_advect
        vpa_advect = advection_structs.vpa_advect
        me = composition.me_over_mi

        buffer_1 = @view scratch_dummy.buffer_rs_1[ir,1]
        buffer_2 = @view scratch_dummy.buffer_rs_2[ir,1]
        buffer_3 = @view scratch_dummy.buffer_rs_3[ir,1]
        buffer_4 = @view scratch_dummy.buffer_rs_4[ir,1]

        # Reconstruct w_∥^3 moment of g_e from already-calculated qpar
        third_moment = scratch_dummy.buffer_z_1
        dthird_moment_dz = scratch_dummy.buffer_z_2
        begin_z_region()
        @loop_z iz begin
            third_moment[iz] = 0.5 * qpar[iz] / ppar[iz] / vth[iz]
        end
        derivative_z!(dthird_moment_dz, third_moment, buffer_1, buffer_2,
                      buffer_3, buffer_4, z_spectral, z)

        begin_vperp_vpa_region()
        update_electron_speed_z!(z_advect[1], upar, vth, vpa.grid, ir)
        z_speed = @view z_advect[1].speed[:,:,:,ir]

        delta_p = allocate_shared_float(size(ppar)...)
        p_amplitude = epsilon * maximum(ppar)
        f = @view pdf.electron.norm[:,:,:,ir]
        begin_serial_region()
        @serial_region begin
            @. delta_p = p_amplitude * sin(2.0*π*test_wavenumber*z.grid/z.L)

            # Make sure initial condition has some z-variation. As f is 'moment kinetic' this
            # means f must have a non-Maxwellian part that varies in z.
            f .*= 1.0 .+ 1.0e-4 .* reshape(vpa.grid.^3, vpa.n, 1, 1) .* reshape(sin.(2.0.*π.*z.grid./z.L), 1, 1, z.n)
        end
        # Ensure initial electron distribution function obeys constraints
        hard_force_moment_constraints!(reshape(f, vpa.n, vperp.n, z.n, 1), moments, vpa)
        delta_f = allocate_shared_float(size(f)...)
        f_amplitude = epsilon * maximum(f)
        # Use exp(sin()) in vpa so that perturbation does not have any symmetry that makes
        # low-order moments vanish exactly.
        begin_serial_region()
        @serial_region begin
            delta_f .= f_amplitude .*
                       reshape(sin.(2.0.*π.*test_wavenumber.*z.grid./z.L), 1, 1, z.n) .*
                       reshape(exp.(sin.(2.0.*π.*test_wavenumber.*vpa.grid./vpa.L)) .- 1.0, vpa.n, 1, 1) .*
                       f
        end

        pdf_size = length(f)
        p_size = length(ppar)
        total_size = pdf_size + p_size

        dpdf_dvpa = @view scratch_dummy.buffer_vpavperpzr_2[:,:,:,ir]
        begin_z_vperp_region()
        update_electron_speed_vpa!(vpa_advect[1], dens, upar, ppar, moments, vpa.grid,
                                   external_source_settings.electron, ir)
        @loop_z_vperp iz ivperp begin
            @views @. vpa_advect[1].adv_fac[:,ivperp,iz,ir] = -vpa_advect[1].speed[:,ivperp,iz,ir]
        end
        #calculate the upwind derivative of the electron pdf w.r.t. wpa
        @loop_z_vperp iz ivperp begin
            @views derivative!(dpdf_dvpa[:,ivperp,iz], f[:,ivperp,iz], vpa,
                               vpa_advect[1].adv_fac[:,ivperp,iz,ir], vpa_spectral)
        end

        jacobian_matrix = allocate_shared_float(total_size, total_size)
        begin_serial_region()
        @serial_region begin
            jacobian_matrix .= 0.0
            for row ∈ 1:total_size
                # Initialise identity matrix
                jacobian_matrix[row,row] = 1.0
            end
        end

        add_electron_vpa_advection_to_Jacobian!(
            jacobian_matrix, f, dens, upar, ppar, vth, third_moment, dpdf_dvpa, ddens_dz,
            dppar_dz, dthird_moment_dz, moments, me, z, vperp, vpa, z_spectral,
            vpa_spectral, vpa_advect, z_speed, scratch_dummy, external_source_settings,
            dt, ir; ppar_offset=pdf_size)

        # Test 'ADI Jacobians' before other tests, because residual_func() may modify some
        # variables (vth, etc.).

        jacobian_matrix_ADI_check = allocate_shared_float(total_size, total_size)

        @testset "ADI Jacobians - implicit z" begin
            # 'Implicit' and 'explicit' parts of Jacobian should add up to full Jacobian.
            begin_serial_region()
            @serial_region begin
                jacobian_matrix_ADI_check .= 0.0
                for row ∈ 1:total_size
                    # Initialise identity matrix
                    jacobian_matrix_ADI_check[row,row] = 1.0
                end
            end

            # There is no 'implicit z' contribution for vpa advection

            # Add 'explicit' contribution
            add_electron_vpa_advection_to_Jacobian!(
                jacobian_matrix_ADI_check, f, dens, upar, ppar, vth, third_moment,
                dpdf_dvpa, ddens_dz, dppar_dz, dthird_moment_dz, moments, me, z, vperp,
                vpa, z_spectral, vpa_spectral, vpa_advect, z_speed, scratch_dummy,
                external_source_settings, dt, ir, :explicit_v; ppar_offset=pdf_size)

            begin_serial_region()
            @serial_region begin
                @test elementwise_isapprox(jacobian_matrix_ADI_check, jacobian_matrix; rtol=0.0, atol=1.0e-15)
            end
        end

        @testset "ADI Jacobians - implicit v" begin
            # 'Implicit' and 'explicit' parts of Jacobian should add up to full Jacobian.
            begin_serial_region()
            @serial_region begin
                jacobian_matrix_ADI_check .= 0.0
                for row ∈ 1:total_size
                    # Initialise identity matrix
                    jacobian_matrix_ADI_check[row,row] = 1.0
                end
            end

            v_size = vperp.n * vpa.n

            # Add 'implicit' contribution
            begin_z_region()
            @loop_z iz begin
                this_slice = collect((iz - 1)*v_size + 1:iz*v_size)
                push!(this_slice, iz + pdf_size)
                @views add_electron_vpa_advection_to_v_only_Jacobian!(
                    jacobian_matrix_ADI_check[this_slice,this_slice], f[:,:,iz], dens[iz],
                    upar[iz], ppar[iz], vth[iz], third_moment[iz], dpdf_dvpa[:,:,iz],
                    ddens_dz[iz], dppar_dz[iz], dthird_moment_dz[iz], moments, me, z,
                    vperp, vpa, z_spectral, vpa_spectral, vpa_advect, z_speed,
                    scratch_dummy, external_source_settings, dt, ir, iz)
            end

            # Add 'explicit' contribution
            add_electron_vpa_advection_to_Jacobian!(
                jacobian_matrix_ADI_check, f, dens, upar, ppar, vth, third_moment,
                dpdf_dvpa, ddens_dz, dppar_dz, dthird_moment_dz, moments, me, z, vperp,
                vpa, z_spectral, vpa_spectral, vpa_advect, z_speed, scratch_dummy,
                external_source_settings, dt, ir, :explicit_z; ppar_offset=pdf_size)

            begin_serial_region()
            @serial_region begin
                @test elementwise_isapprox(jacobian_matrix_ADI_check, jacobian_matrix; rtol=0.0, atol=1.0e-15)
            end
        end

        function residual_func!(residual, this_f, this_p)
            begin_z_region()
            @loop_z iz begin
                # update the electron thermal speed using the updated electron
                # parallel pressure
                vth[iz] = sqrt(abs(2.0 * this_p[iz] /
                                   (dens[iz] * composition.me_over_mi)))
            end
            # Calculate heat flux and derivatives using new_variables
            calculate_electron_qpar_from_pdf_no_r!(qpar, this_p, vth, this_f, vpa, ir)

            calculate_electron_moment_derivatives_no_r!(
                moments,
                (electron_density=dens,
                 electron_upar=upar,
                 electron_ppar=this_p),
                scratch_dummy, z, z_spectral,
                num_diss_params.electron.moment_dissipation_coefficient, ir)

            # electron_kinetic_equation_euler_update!() just adds dt*d(g_e)/dt to the
            # electron_pdf member of the first argument, so if we set the electron_pdf member
            # of the first argument to zero, and pass dt=1, then it will evaluate the time
            # derivative, which is the residual for a steady-state solution.
            begin_z_vperp_vpa_region()
            @loop_z_vperp_vpa iz ivperp ivpa begin
                residual[ivpa,ivperp,iz] = f[ivpa,ivperp,iz]
            end
            electron_vpa_advection!(residual, this_f, dens, upar, this_p, moments,
                                    vpa_advect, vpa, vpa_spectral, scratch_dummy, dt,
                                    external_source_settings.electron, ir)
            # Now
            #   residual = f_electron_old + dt*RHS(f_electron_newvar)
            # so update to desired residual
            begin_z_vperp_vpa_region()
            @loop_z_vperp_vpa iz ivperp ivpa begin
                residual[ivpa,ivperp,iz] = this_f[ivpa,ivperp,iz] - residual[ivpa,ivperp,iz]
            end

            # Set residual to zero where pdf_electron is determined by boundary conditions.
            if vpa.n > 1
                begin_z_vperp_region()
                @loop_z_vperp iz ivperp begin
                    @views enforce_v_boundary_condition_local!(residual[:,ivperp,iz], vpa.bc,
                                                               vpa_advect[1].speed[:,ivperp,iz,ir],
                                                               num_diss_params.electron.vpa_dissipation_coefficient > 0.0,
                                                               vpa, vpa_spectral)
                end
            end
            if vperp.n > 1
                begin_z_vpa_region()
                enforce_vperp_boundary_condition!(residual, vperp.bc,
                                                  vperp, vperp_spectral, vperp_adv,
                                                  vperp_diffusion, ir)
            end
            if z.bc == "wall"
                error("z_bc = \"wall\" not supported here yet.")
            elseif (z.bc == "constant") && (z.irank == 0 || z.irank == z.nrank - 1)
                # Boundary conditions on incoming part of distribution function. Note
                # that as density, upar, ppar do not change in this implicit step,
                # f_electron_newvar, f_old, and residual should all be zero at exactly
                # the same set of grid points, so it is reasonable to zero-out
                # `residual` to impose the boundary condition. We impose this after
                # subtracting f_old in case rounding errors, etc. mean that at some
                # point f_old had a different boundary condition cut-off index.
                begin_vperp_vpa_region()
                v_unnorm = vpa.scratch
                zero = 1.0e-14
                if z.irank == 0
                    iz = 1
                    v_unnorm .= vpagrid_to_dzdt(vpa.grid, vth[iz], upar[iz], true, true)
                    @loop_vperp_vpa ivperp ivpa begin
                        if v_unnorm[ivpa] > -zero
                            residual[ivpa,ivperp,iz] = 0.0
                        end
                    end
                end
                if z.irank == z.nrank - 1
                    iz = z.n
                    v_unnorm .= vpagrid_to_dzdt(vpa.grid, vth[iz], upar[iz], true, true)
                    @loop_vperp_vpa ivperp ivpa begin
                        if v_unnorm[ivpa] < zero
                            residual[ivpa,ivperp,iz] = 0.0
                        end
                    end
                end
            end
            return nothing
        end

        original_residual = allocate_shared_float(size(f)...)
        perturbed_residual = allocate_shared_float(size(f)...)

        @testset "δf only" begin
            residual_func!(original_residual, f, ppar)
            residual_func!(perturbed_residual, f.+delta_f, ppar)

            begin_serial_region()
            @serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[1:pdf_size] .= vec(delta_f)
                residual_update_with_Jacobian = jacobian_matrix * delta_state
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[1:pdf_size]

                # Check ppar did not get perturbed by the Jacobian
                @test elementwise_isapprox(residual_update_with_Jacobian[pdf_size+1:end],
                                           zeros(p_size); atol=1.0e-15)

                # Divide out the z-average of the magnitude of perturbed_residual from the
                # difference, so that different orders of magnitude at different w_∥ are all
                # tested sensibly, but occasional small values of the residual do not make the
                # test fail.
                # Since we have already normalised, pass `rtol` to `atol` for the comparison.
                norm_factor = generate_norm_factor(perturbed_residual)
                @test elementwise_isapprox(perturbed_residual ./ norm_factor,
                                           reshape(perturbed_with_Jacobian, vpa.n, vperp.n, z.n) ./ norm_factor;
                                           rtol=0.0, atol=rtol)
            end
        end

        @testset "δp only" begin
            residual_func!(original_residual, f, ppar)
            residual_func!(perturbed_residual, f, ppar .+ delta_p)

            begin_serial_region()
            @serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[pdf_size+1:end] .= vec(delta_p)
                residual_update_with_Jacobian = jacobian_matrix * delta_state
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[1:pdf_size]

                # Check ppar did not get perturbed by the Jacobian
                @test elementwise_isapprox(residual_update_with_Jacobian[pdf_size+1:end],
                                           delta_state[pdf_size+1:end]; atol=1.0e-15)

                # Divide out the z-average of the magnitude of perturbed_residual from the
                # difference, so that different orders of magnitude at different w_∥ are all
                # tested sensibly, but occasional small values of the residual do not make the
                # test fail.
                # Since we have already normalised, pass `rtol` to `atol` for the comparison.
                norm_factor = generate_norm_factor(perturbed_residual)
                @test elementwise_isapprox(perturbed_residual ./ norm_factor,
                                           reshape(perturbed_with_Jacobian, vpa.n, vperp.n, z.n) ./ norm_factor;
                                           rtol=0.0, atol=rtol)
            end
        end

        @testset "δf and δp" begin
            residual_func!(original_residual, f, ppar)
            residual_func!(perturbed_residual, f.+delta_f, ppar.+delta_p)

            begin_serial_region()
            @serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[1:pdf_size] .= vec(delta_f)
                delta_state[pdf_size+1:end] .= vec(delta_p)
                residual_update_with_Jacobian = jacobian_matrix * delta_state
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[1:pdf_size]

                # Check ppar did not get perturbed by the Jacobian
                @test elementwise_isapprox(residual_update_with_Jacobian[pdf_size+1:end],
                                           delta_state[pdf_size+1:end]; atol=1.0e-15)

                # Divide out the z-average of the magnitude of perturbed_residual from the
                # difference, so that different orders of magnitude at different w_∥ are all
                # tested sensibly, but occasional small values of the residual do not make the
                # test fail.
                # Since we have already normalised, pass `rtol` to `atol` for the comparison.
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

function test_contribution_from_electron_pdf_term(test_input; rtol=(4.0e2*epsilon)^2)
    test_input = deepcopy(test_input)
    test_input["output"]["run_name"] *= "_contribution_from_electron_pdf_term"
    println("    - contribution_from_electron_pdf_term")

    @testset "contribution_from_electron_pdf_term" begin
        # Suppress console output while running
        pdf, scratch, scratch_implicit, scratch_electron, t_params, vz, vr, vzeta, vpa,
            vperp, gyrophase, z, r, moments, fields, spectral_objects, advection_structs,
            composition, collisions, geometry, gyroavs, boundary_distributions,
            external_source_settings, num_diss_params, nl_solver_params, advance,
            advance_implicit, fp_arrays, scratch_dummy, manufactured_source_list,
            ascii_io, io_moments, io_dfns = get_mk_state(test_input)

        dens = @view moments.electron.dens[:,ir]
        upar = @view moments.electron.upar[:,ir]
        ppar = @view moments.electron.ppar[:,ir]
        vth = @view moments.electron.vth[:,ir]
        qpar = @view moments.electron.qpar[:,ir]
        ddens_dz = @view moments.electron.ddens_dz[:,ir]
        dppar_dz = @view moments.electron.dppar_dz[:,ir]
        dqpar_dz = @view moments.electron.dqpar_dz[:,ir]
        dvth_dz = @view moments.electron.dvth_dz[:,ir]
        z_spectral = spectral_objects.z_spectral
        vpa_spectral = spectral_objects.vpa_spectral
        z_advect = advection_structs.z_advect
        vpa_advect = advection_structs.vpa_advect
        me = composition.me_over_mi

        buffer_1 = @view scratch_dummy.buffer_rs_1[ir,1]
        buffer_2 = @view scratch_dummy.buffer_rs_2[ir,1]
        buffer_3 = @view scratch_dummy.buffer_rs_3[ir,1]
        buffer_4 = @view scratch_dummy.buffer_rs_4[ir,1]

        # Reconstruct w_∥^3 moment of g_e from already-calculated qpar
        third_moment = scratch_dummy.buffer_z_1
        dthird_moment_dz = scratch_dummy.buffer_z_2
        begin_z_region()
        @loop_z iz begin
            third_moment[iz] = 0.5 * qpar[iz] / ppar[iz] / vth[iz]
        end
        derivative_z!(dthird_moment_dz, third_moment, buffer_1, buffer_2,
                      buffer_3, buffer_4, z_spectral, z)

        begin_vperp_vpa_region()
        update_electron_speed_z!(z_advect[1], upar, vth, vpa.grid, ir)
        z_speed = @view z_advect[1].speed[:,:,:,ir]

        delta_p = allocate_shared_float(size(ppar)...)
        p_amplitude = epsilon * maximum(ppar)
        f = @view pdf.electron.norm[:,:,:,ir]
        begin_serial_region()
        @serial_region begin
            @. delta_p = p_amplitude * sin(2.0*π*test_wavenumber*z.grid/z.L)

            # Make sure initial condition has some z-variation. As f is 'moment kinetic' this
            # means f must have a non-Maxwellian part that varies in z.
            f .*= 1.0 .+ 1.0e-4 .* reshape(vpa.grid.^3, vpa.n, 1, 1) .* reshape(sin.(2.0.*π.*z.grid./z.L), 1, 1, z.n)
        end
        # Ensure initial electron distribution function obeys constraints
        hard_force_moment_constraints!(reshape(f, vpa.n, vperp.n, z.n, 1), moments, vpa)
        delta_f = allocate_shared_float(size(f)...)
        f_amplitude = epsilon * maximum(f)
        # Use exp(sin()) in vpa so that perturbation does not have any symmetry that makes
        # low-order moments vanish exactly.
        begin_serial_region()
        @serial_region begin
            delta_f .= f_amplitude .*
                       reshape(sin.(2.0.*π.*test_wavenumber.*z.grid./z.L), 1, 1, z.n) .*
                       reshape(exp.(sin.(2.0.*π.*test_wavenumber.*vpa.grid./vpa.L)) .- 1.0, vpa.n, 1, 1) .*
                       f
        end

        pdf_size = length(f)
        p_size = length(ppar)
        total_size = pdf_size + p_size

        jacobian_matrix = allocate_shared_float(total_size, total_size)
        begin_serial_region()
        @serial_region begin
            jacobian_matrix .= 0.0
            for row ∈ 1:total_size
                # Initialise identity matrix
                jacobian_matrix[row,row] = 1.0
            end
        end

        add_contribution_from_electron_pdf_term_to_Jacobian!(
            jacobian_matrix, f, dens, upar, ppar, vth, third_moment, ddens_dz, dppar_dz,
            dvth_dz, dqpar_dz, dthird_moment_dz, moments, me, external_source_settings, z,
            vperp, vpa, z_spectral, z_speed, scratch_dummy, dt, ir; ppar_offset=pdf_size)

        # Test 'ADI Jacobians' before other tests, because residual_func() may modify some
        # variables (vth, etc.).

        jacobian_matrix_ADI_check = allocate_shared_float(total_size, total_size)

        @testset "ADI Jacobians - implicit z" begin
            # 'Implicit' and 'explicit' parts of Jacobian should add up to full Jacobian.
            begin_serial_region()
            @serial_region begin
                jacobian_matrix_ADI_check .= 0.0
                for row ∈ 1:total_size
                    # Initialise identity matrix
                    jacobian_matrix_ADI_check[row,row] = 1.0
                end
            end

            v_size = vperp.n * vpa.n

            # Add 'implicit' contribution
            begin_vperp_vpa_region()
            @loop_vperp_vpa ivperp ivpa begin
                this_slice = (ivperp - 1)*vpa.n + ivpa:v_size:(z.n - 1)*v_size + (ivperp - 1)*vpa.n + ivpa
                @views add_contribution_from_electron_pdf_term_to_z_only_Jacobian!(
                    jacobian_matrix_ADI_check[this_slice,this_slice], f[ivpa,ivperp,:],
                    dens, upar, ppar, vth, third_moment, ddens_dz, dppar_dz, dvth_dz,
                    dqpar_dz, dthird_moment_dz, moments, me, external_source_settings, z,
                    vperp, vpa, z_spectral, z_speed, scratch_dummy, dt, ir, ivperp, ivpa)
            end

            # Add 'explicit' contribution
            add_contribution_from_electron_pdf_term_to_Jacobian!(
                jacobian_matrix_ADI_check, f, dens, upar, ppar, vth, third_moment,
                ddens_dz, dppar_dz, dvth_dz, dqpar_dz, dthird_moment_dz, moments, me,
                external_source_settings, z, vperp, vpa, z_spectral, z_speed,
                scratch_dummy, dt, ir, :explicit_v; ppar_offset=pdf_size)

            begin_serial_region()
            @serial_region begin
                @test elementwise_isapprox(jacobian_matrix_ADI_check, jacobian_matrix; rtol=0.0, atol=1.0e-13)
            end
        end

        @testset "ADI Jacobians - implicit v" begin
            # 'Implicit' and 'explicit' parts of Jacobian should add up to full Jacobian.
            begin_serial_region()
            @serial_region begin
                jacobian_matrix_ADI_check .= 0.0
                for row ∈ 1:total_size
                    # Initialise identity matrix
                    jacobian_matrix_ADI_check[row,row] = 1.0
                end
            end

            v_size = vperp.n * vpa.n

            # Add 'implicit' contribution
            begin_z_region()
            @loop_z iz begin
                this_slice = collect((iz - 1)*v_size + 1:iz*v_size)
                push!(this_slice, iz + pdf_size)
                @views add_contribution_from_electron_pdf_term_to_v_only_Jacobian!(
                    jacobian_matrix_ADI_check[this_slice,this_slice], f[:,:,iz], dens[iz],
                    upar[iz], ppar[iz], vth[iz], third_moment[iz], ddens_dz[iz],
                    dppar_dz[iz], dvth_dz[iz], dqpar_dz[iz], dthird_moment_dz[iz],
                    moments, me, external_source_settings, z, vperp, vpa, z_spectral,
                    z_speed, scratch_dummy, dt, ir, iz)
            end

            # Add 'explicit' contribution
            add_contribution_from_electron_pdf_term_to_Jacobian!(
                jacobian_matrix_ADI_check, f, dens, upar, ppar, vth, third_moment,
                ddens_dz, dppar_dz, dvth_dz, dqpar_dz, dthird_moment_dz, moments, me,
                external_source_settings, z, vperp, vpa, z_spectral, z_speed,
                scratch_dummy, dt, ir, :explicit_z; ppar_offset=pdf_size)

            begin_serial_region()
            @serial_region begin
                @test elementwise_isapprox(jacobian_matrix_ADI_check, jacobian_matrix; rtol=0.0, atol=1.0e-13)
            end
        end

        function residual_func!(residual, this_f, this_p)
            begin_z_region()
            @loop_z iz begin
                # update the electron thermal speed using the updated electron
                # parallel pressure
                vth[iz] = sqrt(abs(2.0 * this_p[iz] /
                                   (dens[iz] * composition.me_over_mi)))
            end
            # Calculate heat flux and derivatives using new_variables
            calculate_electron_qpar_from_pdf_no_r!(qpar, this_p, vth, this_f, vpa, ir)

            calculate_electron_moment_derivatives_no_r!(
                moments,
                (electron_density=dens,
                 electron_upar=upar,
                 electron_ppar=this_p),
                scratch_dummy, z, z_spectral,
                num_diss_params.electron.moment_dissipation_coefficient, ir)

            # electron_kinetic_equation_euler_update!() just adds dt*d(g_e)/dt to the
            # electron_pdf member of the first argument, so if we set the electron_pdf member
            # of the first argument to zero, and pass dt=1, then it will evaluate the time
            # derivative, which is the residual for a steady-state solution.
            begin_z_vperp_vpa_region()
            @loop_z_vperp_vpa iz ivperp ivpa begin
                residual[ivpa,ivperp,iz] = f[ivpa,ivperp,iz]
            end
            add_contribution_from_pdf_term!(residual, this_f, this_p, dens, upar, moments,
                                            vpa.grid, z, dt,
                                            external_source_settings.electron, ir)
            # Now
            #   residual = f_electron_old + dt*RHS(f_electron_newvar)
            # so update to desired residual
            begin_z_vperp_vpa_region()
            @loop_z_vperp_vpa iz ivperp ivpa begin
                residual[ivpa,ivperp,iz] = this_f[ivpa,ivperp,iz] - residual[ivpa,ivperp,iz]
            end

            # Set residual to zero where pdf_electron is determined by boundary conditions.
            if vpa.n > 1
                begin_z_vperp_region()
                @loop_z_vperp iz ivperp begin
                    @views enforce_v_boundary_condition_local!(residual[:,ivperp,iz], vpa.bc,
                                                               vpa_advect[1].speed[:,ivperp,iz,ir],
                                                               num_diss_params.electron.vpa_dissipation_coefficient > 0.0,
                                                               vpa, vpa_spectral)
                end
            end
            if vperp.n > 1
                begin_z_vpa_region()
                enforce_vperp_boundary_condition!(residual, vperp.bc,
                                                  vperp, vperp_spectral, vperp_adv,
                                                  vperp_diffusion, ir)
            end
            if z.bc == "wall"
                error("z_bc = \"wall\" not supported here yet.")
            elseif (z.bc == "constant") && (z.irank == 0 || z.irank == z.nrank - 1)
                # Boundary conditions on incoming part of distribution function. Note
                # that as density, upar, ppar do not change in this implicit step,
                # f_electron_newvar, f_old, and residual should all be zero at exactly
                # the same set of grid points, so it is reasonable to zero-out
                # `residual` to impose the boundary condition. We impose this after
                # subtracting f_old in case rounding errors, etc. mean that at some
                # point f_old had a different boundary condition cut-off index.
                begin_vperp_vpa_region()
                v_unnorm = vpa.scratch
                zero = 1.0e-14
                if z.irank == 0
                    iz = 1
                    v_unnorm .= vpagrid_to_dzdt(vpa.grid, vth[iz], upar[iz], true, true)
                    @loop_vperp_vpa ivperp ivpa begin
                        if v_unnorm[ivpa] > -zero
                            residual[ivpa,ivperp,iz] = 0.0
                        end
                    end
                end
                if z.irank == z.nrank - 1
                    iz = z.n
                    v_unnorm .= vpagrid_to_dzdt(vpa.grid, vth[iz], upar[iz], true, true)
                    @loop_vperp_vpa ivperp ivpa begin
                        if v_unnorm[ivpa] < zero
                            residual[ivpa,ivperp,iz] = 0.0
                        end
                    end
                end
            end
            return nothing
        end

        original_residual = allocate_shared_float(size(f)...)
        perturbed_residual = allocate_shared_float(size(f)...)

        @testset "δf only" begin
            residual_func!(original_residual, f, ppar)
            residual_func!(perturbed_residual, f.+delta_f, ppar)

            begin_serial_region()
            @serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[1:pdf_size] .= vec(delta_f)
                residual_update_with_Jacobian = jacobian_matrix * delta_state
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[1:pdf_size]

                # Check ppar did not get perturbed by the Jacobian
                @test elementwise_isapprox(residual_update_with_Jacobian[pdf_size+1:end],
                                           zeros(p_size); atol=1.0e-15)

                norm_factor = generate_norm_factor(perturbed_residual)
                @test elementwise_isapprox(perturbed_residual ./ norm_factor,
                                           reshape(perturbed_with_Jacobian, vpa.n, vperp.n, z.n) ./ norm_factor;
                                           rtol=0.0, atol=rtol)
            end
        end

        @testset "δp only" begin
            residual_func!(original_residual, f, ppar)
            residual_func!(perturbed_residual, f, ppar .+ delta_p)

            begin_serial_region()
            @serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[pdf_size+1:end] .= vec(delta_p)
                residual_update_with_Jacobian = jacobian_matrix * delta_state
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[1:pdf_size]

                # Check ppar did not get perturbed by the Jacobian
                @test elementwise_isapprox(residual_update_with_Jacobian[pdf_size+1:end],
                                           delta_state[pdf_size+1:end]; atol=1.0e-15)

                norm_factor = generate_norm_factor(perturbed_residual)
                @test elementwise_isapprox(perturbed_residual ./ norm_factor,
                                           reshape(perturbed_with_Jacobian, vpa.n, vperp.n, z.n) ./ norm_factor;
                                           rtol=0.0, atol=rtol)
            end
        end

        @testset "δf and δp" begin
            residual_func!(original_residual, f, ppar)
            residual_func!(perturbed_residual, f.+delta_f, ppar.+delta_p)

            begin_serial_region()
            @serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[1:pdf_size] .= vec(delta_f)
                delta_state[pdf_size+1:end] .= vec(delta_p)
                residual_update_with_Jacobian = jacobian_matrix * delta_state
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[1:pdf_size]

                # Check ppar did not get perturbed by the Jacobian
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

function test_electron_dissipation_term(test_input; rtol=(3.0e0*epsilon)^2)
    test_input = deepcopy(test_input)
    test_input["output"]["run_name"] *= "_electron_dissipation_term"
    println("    - electron_dissipation_term")

    @testset "electron_dissipation_term" begin
        # Suppress console output while running
        pdf, scratch, scratch_implicit, scratch_electron, t_params, vz, vr, vzeta, vpa,
            vperp, gyrophase, z, r, moments, fields, spectral_objects, advection_structs,
            composition, collisions, geometry, gyroavs, boundary_distributions,
            external_source_settings, num_diss_params, nl_solver_params, advance,
            advance_implicit, fp_arrays, scratch_dummy, manufactured_source_list,
            ascii_io, io_moments, io_dfns = get_mk_state(test_input)

        dens = @view moments.electron.dens[:,ir]
        upar = @view moments.electron.upar[:,ir]
        ppar = @view moments.electron.ppar[:,ir]
        vth = @view moments.electron.vth[:,ir]
        qpar = @view moments.electron.qpar[:,ir]
        z_spectral = spectral_objects.z_spectral
        vpa_spectral = spectral_objects.vpa_spectral
        z_advect = advection_structs.z_advect
        vpa_advect = advection_structs.vpa_advect

        begin_vperp_vpa_region()
        update_electron_speed_z!(z_advect[1], upar, vth, vpa.grid, ir)
        z_speed = @view z_advect[1].speed[:,:,:,ir]

        delta_p = allocate_shared_float(size(ppar)...)
        p_amplitude = epsilon * maximum(ppar)
        f = @view pdf.electron.norm[:,:,:,ir]
        begin_serial_region()
        @serial_region begin
            @. delta_p = p_amplitude * sin(2.0*π*test_wavenumber*z.grid/z.L)

            # Make sure initial condition has some z-variation. As f is 'moment kinetic' this
            # means f must have a non-Maxwellian part that varies in z.
            f .*= 1.0 .+ 1.0e-4 .* reshape(vpa.grid.^3, vpa.n, 1, 1) .* reshape(sin.(2.0.*π.*z.grid./z.L), 1, 1, z.n)
        end
        # Ensure initial electron distribution function obeys constraints
        hard_force_moment_constraints!(reshape(f, vpa.n, vperp.n, z.n, 1), moments, vpa)
        delta_f = allocate_shared_float(size(f)...)
        f_amplitude = epsilon * maximum(f)
        # Use exp(sin()) in vpa so that perturbation does not have any symmetry that makes
        # low-order moments vanish exactly.
        begin_serial_region()
        @serial_region begin
            delta_f .= f_amplitude .*
                       reshape(sin.(2.0.*π.*test_wavenumber.*z.grid./z.L), 1, 1, z.n) .*
                       reshape(exp.(sin.(2.0.*π.*test_wavenumber.*vpa.grid./vpa.L)) .- 1.0, vpa.n, 1, 1) .*
                       f
        end

        pdf_size = length(f)
        p_size = length(ppar)
        total_size = pdf_size + p_size

        jacobian_matrix = allocate_shared_float(total_size, total_size)
        begin_serial_region()
        @serial_region begin
            jacobian_matrix .= 0.0
            for row ∈ 1:total_size
                # Initialise identity matrix
                jacobian_matrix[row,row] = 1.0
            end
        end

        add_electron_dissipation_term_to_Jacobian!(
            jacobian_matrix, f, num_diss_params, z, vperp, vpa, vpa_spectral, z_speed, dt,
            ir)

        # Test 'ADI Jacobians' before other tests, because residual_func() may modify some
        # variables (vth, etc.).

        jacobian_matrix_ADI_check = allocate_shared_float(total_size, total_size)

        @testset "ADI Jacobians - implicit z" begin
            # 'Implicit' and 'explicit' parts of Jacobian should add up to full Jacobian.
            begin_serial_region()
            @serial_region begin
                jacobian_matrix_ADI_check .= 0.0
                for row ∈ 1:total_size
                    # Initialise identity matrix
                    jacobian_matrix_ADI_check[row,row] = 1.0
                end
            end

            # There is no 'implicit z' contribution for electron dissipation

            # Add 'explicit' contribution
            add_electron_dissipation_term_to_Jacobian!(
                jacobian_matrix_ADI_check, f, num_diss_params, z, vperp, vpa,
                vpa_spectral, z_speed, dt, ir, :explicit_v)

            begin_serial_region()
            @serial_region begin
                @test elementwise_isapprox(jacobian_matrix_ADI_check, jacobian_matrix; rtol=0.0, atol=1.0e-15)
            end
        end

        @testset "ADI Jacobians - implicit v" begin
            # 'Implicit' and 'explicit' parts of Jacobian should add up to full Jacobian.
            begin_serial_region()
            @serial_region begin
                jacobian_matrix_ADI_check .= 0.0
                for row ∈ 1:total_size
                    # Initialise identity matrix
                    jacobian_matrix_ADI_check[row,row] = 1.0
                end
            end

            v_size = vperp.n * vpa.n

            # Add 'implicit' contribution
            begin_z_region()
            @loop_z iz begin
                this_slice = collect((iz - 1)*v_size + 1:iz*v_size)
                push!(this_slice, iz + pdf_size)
                @views add_electron_dissipation_term_to_v_only_Jacobian!(
                    jacobian_matrix_ADI_check[this_slice,this_slice], f[:,:,iz],
                    num_diss_params, z, vperp, vpa, vpa_spectral, z_speed, dt, ir, iz)
            end

            # Add 'explicit' contribution
            add_electron_dissipation_term_to_Jacobian!(
                jacobian_matrix_ADI_check, f, num_diss_params, z, vperp, vpa,
                vpa_spectral, z_speed, dt, ir, :explicit_z)

            begin_serial_region()
            @serial_region begin
                @test elementwise_isapprox(jacobian_matrix_ADI_check, jacobian_matrix; rtol=0.0, atol=1.0e-15)
            end
        end

        function residual_func!(residual, this_f, this_p)
            begin_z_region()
            @loop_z iz begin
                # update the electron thermal speed using the updated electron
                # parallel pressure
                vth[iz] = sqrt(abs(2.0 * this_p[iz] /
                                   (dens[iz] * composition.me_over_mi)))
            end
            # Calculate heat flux and derivatives using new_variables
            calculate_electron_qpar_from_pdf_no_r!(qpar, this_p, vth, this_f, vpa, ir)

            calculate_electron_moment_derivatives_no_r!(
                moments,
                (electron_density=dens,
                 electron_upar=upar,
                 electron_ppar=this_p),
                scratch_dummy, z, z_spectral,
                num_diss_params.electron.moment_dissipation_coefficient, ir)

            # electron_kinetic_equation_euler_update!() just adds dt*d(g_e)/dt to the
            # electron_pdf member of the first argument, so if we set the electron_pdf member
            # of the first argument to zero, and pass dt=1, then it will evaluate the time
            # derivative, which is the residual for a steady-state solution.
            begin_z_vperp_vpa_region()
            @loop_z_vperp_vpa iz ivperp ivpa begin
                residual[ivpa,ivperp,iz] = f[ivpa,ivperp,iz]
            end
            add_dissipation_term!(residual, this_f, scratch_dummy, z_spectral, z, vpa,
                                  vpa_spectral, num_diss_params, dt)
            # Now
            #   residual = f_electron_old + dt*RHS(f_electron_newvar)
            # so update to desired residual
            begin_z_vperp_vpa_region()
            @loop_z_vperp_vpa iz ivperp ivpa begin
                residual[ivpa,ivperp,iz] = this_f[ivpa,ivperp,iz] - residual[ivpa,ivperp,iz]
            end

            # Set residual to zero where pdf_electron is determined by boundary conditions.
            if vpa.n > 1
                begin_z_vperp_region()
                @loop_z_vperp iz ivperp begin
                    @views enforce_v_boundary_condition_local!(residual[:,ivperp,iz], vpa.bc,
                                                               vpa_advect[1].speed[:,ivperp,iz,ir],
                                                               num_diss_params.electron.vpa_dissipation_coefficient > 0.0,
                                                               vpa, vpa_spectral)
                end
            end
            if vperp.n > 1
                begin_z_vpa_region()
                enforce_vperp_boundary_condition!(residual, vperp.bc,
                                                  vperp, vperp_spectral, vperp_adv,
                                                  vperp_diffusion, ir)
            end
            if z.bc == "wall"
                error("z_bc = \"wall\" not supported here yet.")
            elseif (z.bc == "constant") && (z.irank == 0 || z.irank == z.nrank - 1)
                # Boundary conditions on incoming part of distribution function. Note
                # that as density, upar, ppar do not change in this implicit step,
                # f_electron_newvar, f_old, and residual should all be zero at exactly
                # the same set of grid points, so it is reasonable to zero-out
                # `residual` to impose the boundary condition. We impose this after
                # subtracting f_old in case rounding errors, etc. mean that at some
                # point f_old had a different boundary condition cut-off index.
                begin_vperp_vpa_region()
                v_unnorm = vpa.scratch
                zero = 1.0e-14
                if z.irank == 0
                    iz = 1
                    v_unnorm .= vpagrid_to_dzdt(vpa.grid, vth[iz], upar[iz], true, true)
                    @loop_vperp_vpa ivperp ivpa begin
                        if v_unnorm[ivpa] > -zero
                            residual[ivpa,ivperp,iz] = 0.0
                        end
                    end
                end
                if z.irank == z.nrank - 1
                    iz = z.n
                    v_unnorm .= vpagrid_to_dzdt(vpa.grid, vth[iz], upar[iz], true, true)
                    @loop_vperp_vpa ivperp ivpa begin
                        if v_unnorm[ivpa] < zero
                            residual[ivpa,ivperp,iz] = 0.0
                        end
                    end
                end
            end
            return nothing
        end

        original_residual = allocate_shared_float(size(f)...)
        perturbed_residual = allocate_shared_float(size(f)...)

        @testset "δf only" begin
            residual_func!(original_residual, f, ppar)
            residual_func!(perturbed_residual, f.+delta_f, ppar)

            begin_serial_region()
            @serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[1:pdf_size] .= vec(delta_f)
                residual_update_with_Jacobian = jacobian_matrix * delta_state
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[1:pdf_size]

                # Check ppar did not get perturbed by the Jacobian
                @test elementwise_isapprox(residual_update_with_Jacobian[pdf_size+1:end],
                                           zeros(p_size); atol=1.0e-15)

                norm_factor = generate_norm_factor(perturbed_residual)
                @test elementwise_isapprox(perturbed_residual ./ norm_factor,
                                           reshape(perturbed_with_Jacobian, vpa.n, vperp.n, z.n) ./ norm_factor;
                                           rtol=0.0, atol=rtol)
            end
        end

        @testset "δp only" begin
            residual_func!(original_residual, f, ppar)
            residual_func!(perturbed_residual, f, ppar .+ delta_p)

            begin_serial_region()
            @serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[pdf_size+1:end] .= vec(delta_p)
                residual_update_with_Jacobian = jacobian_matrix * delta_state
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[1:pdf_size]

                # Check ppar did not get perturbed by the Jacobian
                @test elementwise_isapprox(residual_update_with_Jacobian[pdf_size+1:end],
                                           delta_state[pdf_size+1:end]; atol=1.0e-15)

                norm_factor = generate_norm_factor(perturbed_residual)
                @test elementwise_isapprox(perturbed_residual ./ norm_factor,
                                           reshape(perturbed_with_Jacobian, vpa.n, vperp.n, z.n) ./ norm_factor;
                                           rtol=0.0, atol=rtol)
            end
        end

        @testset "δf and δp" begin
            residual_func!(original_residual, f, ppar)
            residual_func!(perturbed_residual, f.+delta_f, ppar.+delta_p)

            begin_serial_region()
            @serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[1:pdf_size] .= vec(delta_f)
                delta_state[pdf_size+1:end] .= vec(delta_p)
                residual_update_with_Jacobian = jacobian_matrix * delta_state
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[1:pdf_size]

                # Check ppar did not get perturbed by the Jacobian
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

function test_electron_krook_collisions(test_input; rtol=(2.0e1*epsilon)^2)
    test_input = deepcopy(test_input)
    test_input["output"]["run_name"] *= "_electron_krook_collisions"
    println("    - electron_krook_collisions")

    @testset "electron_krook_collisions" begin
        # Suppress console output while running
        pdf, scratch, scratch_implicit, scratch_electron, t_params, vz, vr, vzeta, vpa,
            vperp, gyrophase, z, r, moments, fields, spectral_objects, advection_structs,
            composition, collisions, geometry, gyroavs, boundary_distributions,
            external_source_settings, num_diss_params, nl_solver_params, advance,
            advance_implicit, fp_arrays, scratch_dummy, manufactured_source_list,
            ascii_io, io_moments, io_dfns = get_mk_state(test_input)

        dens = @view moments.electron.dens[:,ir]
        upar = @view moments.electron.upar[:,ir]
        ppar = @view moments.electron.ppar[:,ir]
        vth = @view moments.electron.vth[:,ir]
        qpar = @view moments.electron.qpar[:,ir]
        z_spectral = spectral_objects.z_spectral
        vpa_spectral = spectral_objects.vpa_spectral
        z_advect = advection_structs.z_advect
        vpa_advect = advection_structs.vpa_advect

        # Modify upar_ion to make sure it is different from upar_electron so that the term
        # proportional to (u_i-u_e) gets tested in case it is ever needed.
        upar_ion = @view moments.ion.upar[:,ir,1]
        @. upar_ion += sin(4.0*π*test_wavenumber*z.grid/z.L)

        begin_vperp_vpa_region()
        update_electron_speed_z!(z_advect[1], upar, vth, vpa.grid, ir)
        z_speed = @view z_advect[1].speed[:,:,:,ir]

        delta_p = allocate_shared_float(size(ppar)...)
        p_amplitude = epsilon * maximum(ppar)
        f = @view pdf.electron.norm[:,:,:,ir]
        begin_serial_region()
        @serial_region begin
            @. delta_p = p_amplitude * sin(2.0*π*test_wavenumber*z.grid/z.L)

            # Make sure initial condition has some z-variation. As f is 'moment kinetic' this
            # means f must have a non-Maxwellian part that varies in z.
            f .*= 1.0 .+ 1.0e-4 .* reshape(vpa.grid.^3, vpa.n, 1, 1) .* reshape(sin.(2.0.*π.*z.grid./z.L), 1, 1, z.n)
        end
        # Ensure initial electron distribution function obeys constraints
        hard_force_moment_constraints!(reshape(f, vpa.n, vperp.n, z.n, 1), moments, vpa)
        delta_f = allocate_shared_float(size(f)...)
        f_amplitude = epsilon * maximum(f)
        # Use exp(sin()) in vpa so that perturbation does not have any symmetry that makes
        # low-order moments vanish exactly.
        begin_serial_region()
        @serial_region begin
            delta_f .= f_amplitude .*
                       reshape(sin.(2.0.*π.*test_wavenumber.*z.grid./z.L), 1, 1, z.n) .*
                       reshape(exp.(sin.(2.0.*π.*test_wavenumber.*vpa.grid./vpa.L)) .- 1.0, vpa.n, 1, 1) .*
                       f
        end

        pdf_size = length(f)
        p_size = length(ppar)
        total_size = pdf_size + p_size

        jacobian_matrix = allocate_shared_float(total_size, total_size)
        begin_serial_region()
        @serial_region begin
            jacobian_matrix .= 0.0
            for row ∈ 1:total_size
                # Initialise identity matrix
                jacobian_matrix[row,row] = 1.0
            end
        end

        add_electron_krook_collisions_to_Jacobian!(
            jacobian_matrix, f, dens, upar, ppar, vth, @view(moments.ion.upar[:,ir]),
            collisions, z, vperp, vpa, z_speed, dt, ir; ppar_offset=pdf_size)

        # Test 'ADI Jacobians' before other tests, because residual_func() may modify some
        # variables (vth, etc.).

        jacobian_matrix_ADI_check = allocate_shared_float(total_size, total_size)

        @testset "ADI Jacobians - implicit z" begin
            # 'Implicit' and 'explicit' parts of Jacobian should add up to full Jacobian.
            begin_serial_region()
            @serial_region begin
                jacobian_matrix_ADI_check .= 0.0
                for row ∈ 1:total_size
                    # Initialise identity matrix
                    jacobian_matrix_ADI_check[row,row] = 1.0
                end
            end

            v_size = vperp.n * vpa.n

            # Add 'implicit' contribution
            begin_vperp_vpa_region()
            @loop_vperp_vpa ivperp ivpa begin
                this_slice = (ivperp - 1)*vpa.n + ivpa:v_size:(z.n - 1)*v_size + (ivperp - 1)*vpa.n + ivpa
                @views add_electron_krook_collisions_to_z_only_Jacobian!(
                    jacobian_matrix_ADI_check[this_slice,this_slice], f[ivpa,ivperp,:],
                    dens, upar, ppar, vth, moments.ion.upar[:,ir], collisions, z, vperp,
                    vpa, z_speed, dt, ir, ivperp, ivpa)
            end

            # Add 'explicit' contribution
            add_electron_krook_collisions_to_Jacobian!(
                jacobian_matrix_ADI_check, f, dens, upar, ppar, vth,
                @view(moments.ion.upar[:,ir]), collisions, z, vperp, vpa, z_speed, dt, ir,
                :explicit_v; ppar_offset=pdf_size)

            begin_serial_region()
            @serial_region begin
                @test elementwise_isapprox(jacobian_matrix_ADI_check, jacobian_matrix; rtol=0.0, atol=1.0e-15)
            end
        end

        @testset "ADI Jacobians - implicit v" begin
            # 'Implicit' and 'explicit' parts of Jacobian should add up to full Jacobian.
            begin_serial_region()
            @serial_region begin
                jacobian_matrix_ADI_check .= 0.0
                for row ∈ 1:total_size
                    # Initialise identity matrix
                    jacobian_matrix_ADI_check[row,row] = 1.0
                end
            end

            v_size = vperp.n * vpa.n

            # Add 'implicit' contribution
            begin_z_region()
            @loop_z iz begin
                this_slice = collect((iz - 1)*v_size + 1:iz*v_size)
                push!(this_slice, iz + pdf_size)
                @views add_electron_krook_collisions_to_v_only_Jacobian!(
                    jacobian_matrix_ADI_check[this_slice,this_slice], f[:,:,iz], dens[iz],
                    upar[iz], ppar[iz], vth[iz], moments.ion.upar[iz,ir], collisions, z,
                    vperp, vpa, z_speed, dt, ir, iz)
            end

            # Add 'explicit' contribution
            add_electron_krook_collisions_to_Jacobian!(
                jacobian_matrix_ADI_check, f, dens, upar, ppar, vth,
                @view(moments.ion.upar[:,ir]), collisions, z, vperp, vpa, z_speed, dt, ir,
                :explicit_z; ppar_offset=pdf_size)

            begin_serial_region()
            @serial_region begin
                @test elementwise_isapprox(jacobian_matrix_ADI_check, jacobian_matrix; rtol=0.0, atol=1.0e-15)
            end
        end

        function residual_func!(residual, this_f, this_p)
            begin_z_region()
            @loop_z iz begin
                # update the electron thermal speed using the updated electron
                # parallel pressure
                vth[iz] = sqrt(abs(2.0 * this_p[iz] /
                                   (dens[iz] * composition.me_over_mi)))
            end
            # Calculate heat flux and derivatives using new_variables
            calculate_electron_qpar_from_pdf_no_r!(qpar, this_p, vth, this_f, vpa, ir)

            calculate_electron_moment_derivatives_no_r!(
                moments,
                (electron_density=dens,
                 electron_upar=upar,
                 electron_ppar=this_p),
                scratch_dummy, z, z_spectral,
                num_diss_params.electron.moment_dissipation_coefficient, ir)

            # electron_kinetic_equation_euler_update!() just adds dt*d(g_e)/dt to the
            # electron_pdf member of the first argument, so if we set the electron_pdf member
            # of the first argument to zero, and pass dt=1, then it will evaluate the time
            # derivative, which is the residual for a steady-state solution.
            begin_z_vperp_vpa_region()
            @loop_z_vperp_vpa iz ivperp ivpa begin
                residual[ivpa,ivperp,iz] = f[ivpa,ivperp,iz]
            end
            @views electron_krook_collisions!(residual, this_f, dens, upar,
                                              moments.ion.upar[:,ir], vth, collisions,
                                              vperp, vpa, dt)
            # Now
            #   residual = f_electron_old + dt*RHS(f_electron_newvar)
            # so update to desired residual
            begin_z_vperp_vpa_region()
            @loop_z_vperp_vpa iz ivperp ivpa begin
                residual[ivpa,ivperp,iz] = this_f[ivpa,ivperp,iz] - residual[ivpa,ivperp,iz]
            end

            # Set residual to zero where pdf_electron is determined by boundary conditions.
            if vpa.n > 1
                begin_z_vperp_region()
                @loop_z_vperp iz ivperp begin
                    @views enforce_v_boundary_condition_local!(residual[:,ivperp,iz], vpa.bc,
                                                               vpa_advect[1].speed[:,ivperp,iz,ir],
                                                               num_diss_params.electron.vpa_dissipation_coefficient > 0.0,
                                                               vpa, vpa_spectral)
                end
            end
            if vperp.n > 1
                begin_z_vpa_region()
                enforce_vperp_boundary_condition!(residual, vperp.bc,
                                                  vperp, vperp_spectral, vperp_adv,
                                                  vperp_diffusion, ir)
            end
            if z.bc == "wall"
                error("z_bc = \"wall\" not supported here yet.")
            elseif (z.bc == "constant") && (z.irank == 0 || z.irank == z.nrank - 1)
                # Boundary conditions on incoming part of distribution function. Note
                # that as density, upar, ppar do not change in this implicit step,
                # f_electron_newvar, f_old, and residual should all be zero at exactly
                # the same set of grid points, so it is reasonable to zero-out
                # `residual` to impose the boundary condition. We impose this after
                # subtracting f_old in case rounding errors, etc. mean that at some
                # point f_old had a different boundary condition cut-off index.
                begin_vperp_vpa_region()
                v_unnorm = vpa.scratch
                zero = 1.0e-14
                if z.irank == 0
                    iz = 1
                    v_unnorm .= vpagrid_to_dzdt(vpa.grid, vth[iz], upar[iz], true, true)
                    @loop_vperp_vpa ivperp ivpa begin
                        if v_unnorm[ivpa] > -zero
                            residual[ivpa,ivperp,iz] = 0.0
                        end
                    end
                end
                if z.irank == z.nrank - 1
                    iz = z.n
                    v_unnorm .= vpagrid_to_dzdt(vpa.grid, vth[iz], upar[iz], true, true)
                    @loop_vperp_vpa ivperp ivpa begin
                        if v_unnorm[ivpa] < zero
                            residual[ivpa,ivperp,iz] = 0.0
                        end
                    end
                end
            end
            return nothing
        end

        original_residual = allocate_shared_float(size(f)...)
        perturbed_residual = allocate_shared_float(size(f)...)

        @testset "δf only" begin
            residual_func!(original_residual, f, ppar)
            residual_func!(perturbed_residual, f.+delta_f, ppar)

            begin_serial_region()
            @serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[1:pdf_size] .= vec(delta_f)
                residual_update_with_Jacobian = jacobian_matrix * delta_state
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[1:pdf_size]

                # Check ppar did not get perturbed by the Jacobian
                @test elementwise_isapprox(residual_update_with_Jacobian[pdf_size+1:end],
                                           zeros(p_size); atol=1.0e-15)

                norm_factor = generate_norm_factor(perturbed_residual)
                @test elementwise_isapprox(perturbed_residual ./ norm_factor,
                                           reshape(perturbed_with_Jacobian, vpa.n, vperp.n, z.n) ./ norm_factor;
                                           rtol=0.0, atol=rtol)
            end
        end

        @testset "δp only" begin
            residual_func!(original_residual, f, ppar)
            residual_func!(perturbed_residual, f, ppar .+ delta_p)

            begin_serial_region()
            @serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[pdf_size+1:end] .= vec(delta_p)
                residual_update_with_Jacobian = jacobian_matrix * delta_state
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[1:pdf_size]

                # Check ppar did not get perturbed by the Jacobian
                @test elementwise_isapprox(residual_update_with_Jacobian[pdf_size+1:end],
                                           delta_state[pdf_size+1:end]; atol=1.0e-15)

                norm_factor = generate_norm_factor(perturbed_residual)
                @test elementwise_isapprox(perturbed_residual ./ norm_factor,
                                           reshape(perturbed_with_Jacobian, vpa.n, vperp.n, z.n) ./ norm_factor;
                                           rtol=0.0, atol=rtol)
            end
        end

        @testset "δf and δp" begin
            residual_func!(original_residual, f, ppar)
            residual_func!(perturbed_residual, f.+delta_f, ppar.+delta_p)

            begin_serial_region()
            @serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[1:pdf_size] .= vec(delta_f)
                delta_state[pdf_size+1:end] .= vec(delta_p)
                residual_update_with_Jacobian = jacobian_matrix * delta_state
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[1:pdf_size]

                # Check ppar did not get perturbed by the Jacobian
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

function test_external_electron_source(test_input; rtol=(3.0e1*epsilon)^2)
    test_input = deepcopy(test_input)
    test_input["output"]["run_name"] *= "_external_electron_source"
    println("    - external_electron_source")

    @testset "external_electron_source" begin
        # Suppress console output while running
        pdf, scratch, scratch_implicit, scratch_electron, t_params, vz, vr, vzeta, vpa,
            vperp, gyrophase, z, r, moments, fields, spectral_objects, advection_structs,
            composition, collisions, geometry, gyroavs, boundary_distributions,
            external_source_settings, num_diss_params, nl_solver_params, advance,
            advance_implicit, fp_arrays, scratch_dummy, manufactured_source_list,
            ascii_io, io_moments, io_dfns = get_mk_state(test_input)

        dens = @view moments.electron.dens[:,ir]
        upar = @view moments.electron.upar[:,ir]
        ppar = @view moments.electron.ppar[:,ir]
        vth = @view moments.electron.vth[:,ir]
        qpar = @view moments.electron.qpar[:,ir]
        ddens_dz = @view moments.electron.ddens_dz[:,ir]
        dppar_dz = @view moments.electron.dppar_dz[:,ir]
        z_spectral = spectral_objects.z_spectral
        vpa_spectral = spectral_objects.vpa_spectral
        z_advect = advection_structs.z_advect
        vpa_advect = advection_structs.vpa_advect
        me = composition.me_over_mi

        buffer_1 = @view scratch_dummy.buffer_rs_1[ir,1]
        buffer_2 = @view scratch_dummy.buffer_rs_2[ir,1]
        buffer_3 = @view scratch_dummy.buffer_rs_3[ir,1]
        buffer_4 = @view scratch_dummy.buffer_rs_4[ir,1]

        # Reconstruct w_∥^3 moment of g_e from already-calculated qpar
        third_moment = scratch_dummy.buffer_z_1
        dthird_moment_dz = scratch_dummy.buffer_z_2
        begin_z_region()
        @loop_z iz begin
            third_moment[iz] = 0.5 * qpar[iz] / ppar[iz] / vth[iz]
        end
        derivative_z!(dthird_moment_dz, third_moment, buffer_1, buffer_2,
                      buffer_3, buffer_4, z_spectral, z)

        begin_vperp_vpa_region()
        update_electron_speed_z!(z_advect[1], upar, vth, vpa.grid, ir)
        z_speed = @view z_advect[1].speed[:,:,:,ir]

        delta_p = allocate_shared_float(size(ppar)...)
        p_amplitude = epsilon * maximum(ppar)
        f = @view pdf.electron.norm[:,:,:,ir]
        begin_serial_region()
        @serial_region begin
            @. delta_p = p_amplitude * sin(2.0*π*test_wavenumber*z.grid/z.L)

            # Make sure initial condition has some z-variation. As f is 'moment kinetic' this
            # means f must have a non-Maxwellian part that varies in z.
            f .*= 1.0 .+ 1.0e-4 .* reshape(vpa.grid.^3, vpa.n, 1, 1) .* reshape(sin.(2.0.*π.*z.grid./z.L), 1, 1, z.n)
        end
        # Ensure initial electron distribution function obeys constraints
        hard_force_moment_constraints!(reshape(f, vpa.n, vperp.n, z.n, 1), moments, vpa)
        delta_f = allocate_shared_float(size(f)...)
        f_amplitude = epsilon * maximum(f)
        # Use exp(sin()) in vpa so that perturbation does not have any symmetry that makes
        # low-order moments vanish exactly.
        begin_serial_region()
        @serial_region begin
            delta_f .= f_amplitude .*
                       reshape(sin.(2.0.*π.*test_wavenumber.*z.grid./z.L), 1, 1, z.n) .*
                       reshape(exp.(sin.(2.0.*π.*test_wavenumber.*vpa.grid./vpa.L)) .- 1.0, vpa.n, 1, 1) .*
                       f
        end

        pdf_size = length(f)
        p_size = length(ppar)
        total_size = pdf_size + p_size

        jacobian_matrix = allocate_shared_float(total_size, total_size)
        begin_serial_region()
        @serial_region begin
            jacobian_matrix .= 0.0
            for row ∈ 1:total_size
                # Initialise identity matrix
                jacobian_matrix[row,row] = 1.0
            end
        end

        add_total_external_electron_source_to_Jacobian!(
            jacobian_matrix, f, moments, me, z_speed, external_source_settings.electron,
            z, vperp, vpa, dt, ir; ppar_offset=pdf_size)

        # Test 'ADI Jacobians' before other tests, because residual_func() may modify some
        # variables (vth, etc.).

        jacobian_matrix_ADI_check = allocate_shared_float(total_size, total_size)

        @testset "ADI Jacobians - implicit z" begin
            # 'Implicit' and 'explicit' parts of Jacobian should add up to full Jacobian.
            begin_serial_region()
            @serial_region begin
                jacobian_matrix_ADI_check .= 0.0
                for row ∈ 1:total_size
                    # Initialise identity matrix
                    jacobian_matrix_ADI_check[row,row] = 1.0
                end
            end

            v_size = vperp.n * vpa.n

            # Add 'implicit' contribution
            begin_vperp_vpa_region()
            @loop_vperp_vpa ivperp ivpa begin
                this_slice = (ivperp - 1)*vpa.n + ivpa:v_size:(z.n - 1)*v_size + (ivperp - 1)*vpa.n + ivpa
                @views add_total_external_electron_source_to_z_only_Jacobian!(
                    jacobian_matrix_ADI_check[this_slice,this_slice], f[ivpa,ivperp,:],
                    moments, me, z_speed, external_source_settings.electron, z, vperp,
                    vpa, dt, ir, ivperp, ivpa)
            end

            # Add 'explicit' contribution
            add_total_external_electron_source_to_Jacobian!(
                jacobian_matrix_ADI_check, f, moments, me, z_speed,
                external_source_settings.electron, z, vperp, vpa, dt, ir, :explicit_v;
                ppar_offset=pdf_size)

            begin_serial_region()
            @serial_region begin
                @test elementwise_isapprox(jacobian_matrix_ADI_check, jacobian_matrix; rtol=0.0, atol=1.0e-15)
            end
        end

        @testset "ADI Jacobians - implicit v" begin
            # 'Implicit' and 'explicit' parts of Jacobian should add up to full Jacobian.
            begin_serial_region()
            @serial_region begin
                jacobian_matrix_ADI_check .= 0.0
                for row ∈ 1:total_size
                    # Initialise identity matrix
                    jacobian_matrix_ADI_check[row,row] = 1.0
                end
            end

            v_size = vperp.n * vpa.n

            # Add 'implicit' contribution
            begin_z_region()
            @loop_z iz begin
                this_slice = collect((iz - 1)*v_size + 1:iz*v_size)
                push!(this_slice, iz + pdf_size)
                @views add_total_external_electron_source_to_v_only_Jacobian!(
                    jacobian_matrix_ADI_check[this_slice,this_slice], f[:,:,iz],
                    moments, me, z_speed, external_source_settings.electron, z, vperp,
                    vpa, dt, ir, iz)
            end

            # Add 'explicit' contribution
            add_total_external_electron_source_to_Jacobian!(
                jacobian_matrix_ADI_check, f, moments, me, z_speed,
                external_source_settings.electron, z, vperp, vpa, dt, ir, :explicit_z;
                ppar_offset=pdf_size)

            begin_serial_region()
            @serial_region begin
                @test elementwise_isapprox(jacobian_matrix_ADI_check, jacobian_matrix; rtol=0.0, atol=1.0e-15)
            end
        end

        function residual_func!(residual, this_f, this_p)
            begin_z_region()
            @loop_z iz begin
                # update the electron thermal speed using the updated electron
                # parallel pressure
                vth[iz] = sqrt(abs(2.0 * this_p[iz] /
                                   (dens[iz] * composition.me_over_mi)))
            end
            # Calculate heat flux and derivatives using new_variables
            calculate_electron_qpar_from_pdf_no_r!(qpar, this_p, vth, this_f, vpa, ir)

            calculate_electron_moment_derivatives_no_r!(
                moments,
                (electron_density=dens,
                 electron_upar=upar,
                 electron_ppar=this_p),
                scratch_dummy, z, z_spectral,
                num_diss_params.electron.moment_dissipation_coefficient, ir)

            # electron_kinetic_equation_euler_update!() just adds dt*d(g_e)/dt to the
            # electron_pdf member of the first argument, so if we set the electron_pdf member
            # of the first argument to zero, and pass dt=1, then it will evaluate the time
            # derivative, which is the residual for a steady-state solution.
            begin_z_vperp_vpa_region()
            @loop_z_vperp_vpa iz ivperp ivpa begin
                residual[ivpa,ivperp,iz] = f[ivpa,ivperp,iz]
            end
            total_external_electron_sources!(residual, this_f, dens, upar, moments,
                                             composition,
                                             external_source_settings.electron, vperp,
                                             vpa, dt, ir)
            # Now
            #   residual = f_electron_old + dt*RHS(f_electron_newvar)
            # so update to desired residual
            begin_z_vperp_vpa_region()
            @loop_z_vperp_vpa iz ivperp ivpa begin
                residual[ivpa,ivperp,iz] = this_f[ivpa,ivperp,iz] - residual[ivpa,ivperp,iz]
            end

            # Set residual to zero where pdf_electron is determined by boundary conditions.
            if vpa.n > 1
                begin_z_vperp_region()
                @loop_z_vperp iz ivperp begin
                    @views enforce_v_boundary_condition_local!(residual[:,ivperp,iz], vpa.bc,
                                                               vpa_advect[1].speed[:,ivperp,iz,ir],
                                                               num_diss_params.electron.vpa_dissipation_coefficient > 0.0,
                                                               vpa, vpa_spectral)
                end
            end
            if vperp.n > 1
                begin_z_vpa_region()
                enforce_vperp_boundary_condition!(residual, vperp.bc,
                                                  vperp, vperp_spectral, vperp_adv,
                                                  vperp_diffusion, ir)
            end
            if z.bc == "wall"
                error("z_bc = \"wall\" not supported here yet.")
            elseif (z.bc == "constant") && (z.irank == 0 || z.irank == z.nrank - 1)
                # Boundary conditions on incoming part of distribution function. Note
                # that as density, upar, ppar do not change in this implicit step,
                # f_electron_newvar, f_old, and residual should all be zero at exactly
                # the same set of grid points, so it is reasonable to zero-out
                # `residual` to impose the boundary condition. We impose this after
                # subtracting f_old in case rounding errors, etc. mean that at some
                # point f_old had a different boundary condition cut-off index.
                begin_vperp_vpa_region()
                v_unnorm = vpa.scratch
                zero = 1.0e-14
                if z.irank == 0
                    iz = 1
                    v_unnorm .= vpagrid_to_dzdt(vpa.grid, vth[iz], upar[iz], true, true)
                    @loop_vperp_vpa ivperp ivpa begin
                        if v_unnorm[ivpa] > -zero
                            residual[ivpa,ivperp,iz] = 0.0
                        end
                    end
                end
                if z.irank == z.nrank - 1
                    iz = z.n
                    v_unnorm .= vpagrid_to_dzdt(vpa.grid, vth[iz], upar[iz], true, true)
                    @loop_vperp_vpa ivperp ivpa begin
                        if v_unnorm[ivpa] < zero
                            residual[ivpa,ivperp,iz] = 0.0
                        end
                    end
                end
            end
            return nothing
        end

        original_residual = allocate_shared_float(size(f)...)
        perturbed_residual = allocate_shared_float(size(f)...)

        @testset "δf only" begin
            residual_func!(original_residual, f, ppar)
            residual_func!(perturbed_residual, f.+delta_f, ppar)

            begin_serial_region()
            @serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[1:pdf_size] .= vec(delta_f)
                residual_update_with_Jacobian = jacobian_matrix * delta_state
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[1:pdf_size]

                # Check ppar did not get perturbed by the Jacobian
                @test elementwise_isapprox(residual_update_with_Jacobian[pdf_size+1:end],
                                           zeros(p_size); atol=1.0e-15)

                norm_factor = generate_norm_factor(perturbed_residual)
                @test elementwise_isapprox(perturbed_residual ./ norm_factor,
                                           reshape(perturbed_with_Jacobian, vpa.n, vperp.n, z.n) ./ norm_factor;
                                           rtol=0.0, atol=rtol)
            end
        end

        @testset "δp only" begin
            residual_func!(original_residual, f, ppar)
            residual_func!(perturbed_residual, f, ppar .+ delta_p)

            begin_serial_region()
            @serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[pdf_size+1:end] .= vec(delta_p)
                residual_update_with_Jacobian = jacobian_matrix * delta_state
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[1:pdf_size]

                # Check ppar did not get perturbed by the Jacobian
                @test elementwise_isapprox(residual_update_with_Jacobian[pdf_size+1:end],
                                           delta_state[pdf_size+1:end]; atol=1.0e-15)

                norm_factor = generate_norm_factor(perturbed_residual)
                @test elementwise_isapprox(perturbed_residual ./ norm_factor,
                                           reshape(perturbed_with_Jacobian, vpa.n, vperp.n, z.n) ./ norm_factor;
                                           rtol=0.0, atol=rtol)
            end
        end

        @testset "δf and δp" begin
            residual_func!(original_residual, f, ppar)
            residual_func!(perturbed_residual, f.+delta_f, ppar.+delta_p)

            begin_serial_region()
            @serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[1:pdf_size] .= vec(delta_f)
                delta_state[pdf_size+1:end] .= vec(delta_p)
                residual_update_with_Jacobian = jacobian_matrix * delta_state
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[1:pdf_size]

                # Check ppar did not get perturbed by the Jacobian
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

# For this test where only the 'constraint forcing' term is added to the residual, the
# residual is exactly zero for the initial condition (because that is constructed to obey
# the constraints). Therefore the 'perturbed_residual' is non-zero only because of
# delta_f, which is small, O(epsilon), so 'norm_factor' is also O(epsilon). We therefore
# use a tolerance of O(epsilon) in this test, unlike the other tests which use a tolerance
# of O(epsilon^2). Note that in the final test of the full electron kinetic equations,
# with all terms including this one, we do not have a similar issue, as there the other
# terms create an O(1) residual for the initial condition, which will then set the
# 'norm_factor'.
#
# We test the Jacobian for these constraint forcing terms using
# constraint_forcing_rate=O(1), because in these tests we set dt=O(1), so a large
# coefficient would make the non-linearity large and then it would be hard to distinguish
# errors from non-linearity (or rounding errors) in `test_electron_kinetic_equation()`
# that tests the combined effect of all terms in the electron kinetic equation. This test
# would actually be OK because the ratio of linear to non-linear contributions of this
# single term does not depend on the size of the coefficient. In the combined test, we are
# effectively comparing the non-linear error from this term to the residual from other
# terms, so the coefficient of this term matters there. Even though these settings are not
# what we would use in a real simulation, they should tell us if the implementation is
# correct.
function test_electron_implicit_constraint_forcing(test_input; rtol=(1.5e0*epsilon))
    test_input = deepcopy(test_input)
    test_input["output"]["run_name"] *= "_electron_implicit_constraint_forcing"
    println("    - electron_implicit_constraint_forcing")

    @testset "electron_implicit_constraint_forcing" begin
        # Suppress console output while running
        pdf, scratch, scratch_implicit, scratch_electron, t_params, vz, vr, vzeta, vpa,
            vperp, gyrophase, z, r, moments, fields, spectral_objects, advection_structs,
            composition, collisions, geometry, gyroavs, boundary_distributions,
            external_source_settings, num_diss_params, nl_solver_params, advance,
            advance_implicit, fp_arrays, scratch_dummy, manufactured_source_list,
            ascii_io, io_moments, io_dfns = get_mk_state(test_input)

        dens = @view moments.electron.dens[:,ir]
        upar = @view moments.electron.upar[:,ir]
        ppar = @view moments.electron.ppar[:,ir]
        vth = @view moments.electron.vth[:,ir]
        qpar = @view moments.electron.qpar[:,ir]
        z_spectral = spectral_objects.z_spectral
        vpa_spectral = spectral_objects.vpa_spectral
        z_advect = advection_structs.z_advect
        vpa_advect = advection_structs.vpa_advect

        begin_vperp_vpa_region()
        update_electron_speed_z!(z_advect[1], upar, vth, vpa.grid, ir)
        z_speed = @view z_advect[1].speed[:,:,:,ir]

        delta_p = allocate_shared_float(size(ppar)...)
        p_amplitude = epsilon * maximum(ppar)
        f = @view pdf.electron.norm[:,:,:,ir]
        begin_serial_region()
        @serial_region begin
            @. delta_p = p_amplitude * sin(2.0*π*test_wavenumber*z.grid/z.L)

            # Make sure initial condition has some z-variation. As f is 'moment kinetic' this
            # means f must have a non-Maxwellian part that varies in z.
            f .*= 1.0 .+ 1.0e-4 .* reshape(vpa.grid.^3, vpa.n, 1, 1) .* reshape(sin.(2.0.*π.*z.grid./z.L), 1, 1, z.n)
        end
        # Ensure initial electron distribution function obeys constraints
        hard_force_moment_constraints!(reshape(f, vpa.n, vperp.n, z.n, 1), moments, vpa)
        delta_f = allocate_shared_float(size(f)...)
        f_amplitude = epsilon * maximum(f)
        # Use exp(sin()) in vpa so that perturbation does not have any symmetry that makes
        # low-order moments vanish exactly.
        begin_serial_region()
        @serial_region begin
            delta_f .= f_amplitude .*
                       reshape(sin.(2.0.*π.*test_wavenumber.*z.grid./z.L), 1, 1, z.n) .*
                       reshape(exp.(sin.(2.0.*π.*test_wavenumber.*vpa.grid./vpa.L)) .- 1.0, vpa.n, 1, 1) .*
                       f
        end

        pdf_size = length(f)
        p_size = length(ppar)
        total_size = pdf_size + p_size

        zeroth_moment = z.scratch_shared
        first_moment = z.scratch_shared2
        second_moment = z.scratch_shared3
        begin_z_region()
        vpa_grid = vpa.grid
        vpa_wgts = vpa.wgts
        @loop_z iz begin
            @views zeroth_moment[iz] = integrate_over_vspace(f[:,1,iz], vpa_wgts)
            @views first_moment[iz] = integrate_over_vspace(f[:,1,iz], vpa_grid, vpa_wgts)
            @views second_moment[iz] = integrate_over_vspace(f[:,1,iz], vpa_grid, 2, vpa_wgts)
        end

        jacobian_matrix = allocate_shared_float(total_size, total_size)
        begin_serial_region()
        @serial_region begin
            jacobian_matrix .= 0.0
            for row ∈ 1:total_size
                # Initialise identity matrix
                jacobian_matrix[row,row] = 1.0
            end
        end

        add_electron_implicit_constraint_forcing_to_Jacobian!(
            jacobian_matrix, f, zeroth_moment, first_moment, second_moment, z_speed, z,
            vperp, vpa, t_params.electron.constraint_forcing_rate, dt, ir)

        # Test 'ADI Jacobians' before other tests, because residual_func() may modify some
        # variables (vth, etc.).

        jacobian_matrix_ADI_check = allocate_shared_float(total_size, total_size)

        @testset "ADI Jacobians - implicit z" begin
            # 'Implicit' and 'explicit' parts of Jacobian should add up to full Jacobian.
            begin_serial_region()
            @serial_region begin
                jacobian_matrix_ADI_check .= 0.0
                for row ∈ 1:total_size
                    # Initialise identity matrix
                    jacobian_matrix_ADI_check[row,row] = 1.0
                end
            end

            v_size = vperp.n * vpa.n

            # Add 'implicit' contribution
            begin_vperp_vpa_region()
            @loop_vperp_vpa ivperp ivpa begin
                this_slice = (ivperp - 1)*vpa.n + ivpa:v_size:(z.n - 1)*v_size + (ivperp - 1)*vpa.n + ivpa
                @views add_electron_implicit_constraint_forcing_to_z_only_Jacobian!(
                    jacobian_matrix_ADI_check[this_slice,this_slice], f[ivpa,ivperp,:],
                    zeroth_moment, first_moment, second_moment, z_speed, z, vperp, vpa,
                    t_params.electron.constraint_forcing_rate, dt, ir, ivperp, ivpa)
            end

            # Add 'explicit' contribution
            add_electron_implicit_constraint_forcing_to_Jacobian!(
                jacobian_matrix_ADI_check, f, zeroth_moment, first_moment, second_moment,
                z_speed, z, vperp, vpa, t_params.electron.constraint_forcing_rate, dt, ir,
                :explicit_v)

            begin_serial_region()
            @serial_region begin
                @test elementwise_isapprox(jacobian_matrix_ADI_check, jacobian_matrix; rtol=0.0, atol=1.0e-15)
            end
        end

        @testset "ADI Jacobians - implicit v" begin
            # 'Implicit' and 'explicit' parts of Jacobian should add up to full Jacobian.
            begin_serial_region()
            @serial_region begin
                jacobian_matrix_ADI_check .= 0.0
                for row ∈ 1:total_size
                    # Initialise identity matrix
                    jacobian_matrix_ADI_check[row,row] = 1.0
                end
            end

            v_size = vperp.n * vpa.n

            # Add 'implicit' contribution
            begin_z_region()
            @loop_z iz begin
                this_slice = collect((iz - 1)*v_size + 1:iz*v_size)
                push!(this_slice, iz + pdf_size)
                @views add_electron_implicit_constraint_forcing_to_v_only_Jacobian!(
                    jacobian_matrix_ADI_check[this_slice,this_slice], f[:,:,iz],
                    zeroth_moment[iz], first_moment[iz], second_moment[iz], z_speed, z,
                    vperp, vpa, t_params.electron.constraint_forcing_rate, dt, ir, iz)
            end

            # Add 'explicit' contribution
            add_electron_implicit_constraint_forcing_to_Jacobian!(
                jacobian_matrix_ADI_check, f, zeroth_moment, first_moment, second_moment,
                z_speed, z, vperp, vpa, t_params.electron.constraint_forcing_rate, dt, ir,
                :explicit_z)

            begin_serial_region()
            @serial_region begin
                @test elementwise_isapprox(jacobian_matrix_ADI_check, jacobian_matrix; rtol=0.0, atol=1.0e-15)
            end
        end

        function residual_func!(residual, this_f, this_p)
            begin_z_region()
            @loop_z iz begin
                # update the electron thermal speed using the updated electron
                # parallel pressure
                vth[iz] = sqrt(abs(2.0 * this_p[iz] /
                                   (dens[iz] * composition.me_over_mi)))
            end
            # Calculate heat flux and derivatives using new_variables
            calculate_electron_qpar_from_pdf_no_r!(qpar, this_p, vth, this_f, vpa, ir)

            calculate_electron_moment_derivatives_no_r!(
                moments,
                (electron_density=dens,
                 electron_upar=upar,
                 electron_ppar=this_p),
                scratch_dummy, z, z_spectral,
                num_diss_params.electron.moment_dissipation_coefficient, ir)

            # electron_kinetic_equation_euler_update!() just adds dt*d(g_e)/dt to the
            # electron_pdf member of the first argument, so if we set the electron_pdf member
            # of the first argument to zero, and pass dt=1, then it will evaluate the time
            # derivative, which is the residual for a steady-state solution.
            begin_z_vperp_vpa_region()
            @loop_z_vperp_vpa iz ivperp ivpa begin
                residual[ivpa,ivperp,iz] = f[ivpa,ivperp,iz]
            end
            electron_implicit_constraint_forcing!(residual, this_f,
                                                  t_params.electron.constraint_forcing_rate,
                                                  vpa, dt, ir)
            # Now
            #   residual = f_electron_old + dt*RHS(f_electron_newvar)
            # so update to desired residual
            begin_z_vperp_vpa_region()
            @loop_z_vperp_vpa iz ivperp ivpa begin
                residual[ivpa,ivperp,iz] = this_f[ivpa,ivperp,iz] - residual[ivpa,ivperp,iz]
            end

            # Set residual to zero where pdf_electron is determined by boundary conditions.
            if vpa.n > 1
                begin_z_vperp_region()
                @loop_z_vperp iz ivperp begin
                    @views enforce_v_boundary_condition_local!(residual[:,ivperp,iz], vpa.bc,
                                                               vpa_advect[1].speed[:,ivperp,iz,ir],
                                                               num_diss_params.electron.vpa_dissipation_coefficient > 0.0,
                                                               vpa, vpa_spectral)
                end
            end
            if vperp.n > 1
                begin_z_vpa_region()
                enforce_vperp_boundary_condition!(residual, vperp.bc,
                                                  vperp, vperp_spectral, vperp_adv,
                                                  vperp_diffusion, ir)
            end
            if z.bc == "wall"
                error("z_bc = \"wall\" not supported here yet.")
            elseif (z.bc == "constant") && (z.irank == 0 || z.irank == z.nrank - 1)
                # Boundary conditions on incoming part of distribution function. Note
                # that as density, upar, ppar do not change in this implicit step,
                # f_electron_newvar, f_old, and residual should all be zero at exactly
                # the same set of grid points, so it is reasonable to zero-out
                # `residual` to impose the boundary condition. We impose this after
                # subtracting f_old in case rounding errors, etc. mean that at some
                # point f_old had a different boundary condition cut-off index.
                begin_vperp_vpa_region()
                v_unnorm = vpa.scratch
                zero = 1.0e-14
                if z.irank == 0
                    iz = 1
                    v_unnorm .= vpagrid_to_dzdt(vpa.grid, vth[iz], upar[iz], true, true)
                    @loop_vperp_vpa ivperp ivpa begin
                        if v_unnorm[ivpa] > -zero
                            residual[ivpa,ivperp,iz] = 0.0
                        end
                    end
                end
                if z.irank == z.nrank - 1
                    iz = z.n
                    v_unnorm .= vpagrid_to_dzdt(vpa.grid, vth[iz], upar[iz], true, true)
                    @loop_vperp_vpa ivperp ivpa begin
                        if v_unnorm[ivpa] < zero
                            residual[ivpa,ivperp,iz] = 0.0
                        end
                    end
                end
            end
            return nothing
        end

        original_residual = allocate_shared_float(size(f)...)
        perturbed_residual = allocate_shared_float(size(f)...)

        @testset "δf only" begin
            residual_func!(original_residual, f, ppar)
            residual_func!(perturbed_residual, f.+delta_f, ppar)

            begin_serial_region()
            @serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[1:pdf_size] .= vec(delta_f)
                residual_update_with_Jacobian = jacobian_matrix * delta_state
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[1:pdf_size]

                # Check ppar did not get perturbed by the Jacobian
                @test elementwise_isapprox(residual_update_with_Jacobian[pdf_size+1:end],
                                           zeros(p_size); atol=1.0e-15)

                norm_factor = generate_norm_factor(perturbed_residual)
                @test elementwise_isapprox(perturbed_residual ./ norm_factor,
                                           reshape(perturbed_with_Jacobian, vpa.n, vperp.n, z.n) ./ norm_factor;
                                           rtol=0.0, atol=rtol)
            end
        end

        @testset "δp only" begin
            residual_func!(original_residual, f, ppar)
            residual_func!(perturbed_residual, f, ppar .+ delta_p)

            begin_serial_region()
            @serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[pdf_size+1:end] .= vec(delta_p)
                residual_update_with_Jacobian = jacobian_matrix * delta_state
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[1:pdf_size]

                # Check ppar did not get perturbed by the Jacobian
                @test elementwise_isapprox(residual_update_with_Jacobian[pdf_size+1:end],
                                           delta_state[pdf_size+1:end]; atol=1.0e-15)

                # No norm factor, because both perturbed residuals should be zero here, as
                # delta_p does not affect this term, and `f` (with no `delta_f`) obeys the
                # constraints exactly, so this term vanishes.
                @test elementwise_isapprox(perturbed_residual,
                                           reshape(perturbed_with_Jacobian, vpa.n, vperp.n, z.n);
                                           rtol=0.0, atol=1.0e-15)
            end
        end

        @testset "δf and δp" begin
            residual_func!(original_residual, f, ppar)
            residual_func!(perturbed_residual, f.+delta_f, ppar.+delta_p)

            begin_serial_region()
            @serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[1:pdf_size] .= vec(delta_f)
                delta_state[pdf_size+1:end] .= vec(delta_p)
                residual_update_with_Jacobian = jacobian_matrix * delta_state
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[1:pdf_size]

                # Check ppar did not get perturbed by the Jacobian
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

function test_electron_energy_equation(test_input; rtol=(6.0e2*epsilon)^2)
    test_input = deepcopy(test_input)
    test_input["output"]["run_name"] *= "_electron_energy_equation"
    println("    - electron_energy_equation")

    @testset "electron_energy_equation" begin
        # Suppress console output while running
        pdf, scratch, scratch_implicit, scratch_electron, t_params, vz, vr, vzeta, vpa,
            vperp, gyrophase, z, r, moments, fields, spectral_objects, advection_structs,
            composition, collisions, geometry, gyroavs, boundary_distributions,
            external_source_settings, num_diss_params, nl_solver_params, advance,
            advance_implicit, fp_arrays, scratch_dummy, manufactured_source_list,
            ascii_io, io_moments, io_dfns = get_mk_state(test_input)

        dens = @view moments.electron.dens[:,ir]
        upar = @view moments.electron.upar[:,ir]
        ppar = @view moments.electron.ppar[:,ir]
        vth = @view moments.electron.vth[:,ir]
        qpar = @view moments.electron.qpar[:,ir]
        ddens_dz = @view moments.electron.ddens_dz[:,ir]
        dupar_dz = @view moments.electron.dupar_dz[:,ir]
        dppar_dz = @view moments.electron.dppar_dz[:,ir]
        z_spectral = spectral_objects.z_spectral
        vpa_spectral = spectral_objects.vpa_spectral
        z_advect = advection_structs.z_advect
        vpa_advect = advection_structs.vpa_advect
        me = composition.me_over_mi

        buffer_1 = @view scratch_dummy.buffer_rs_1[ir,1]
        buffer_2 = @view scratch_dummy.buffer_rs_2[ir,1]
        buffer_3 = @view scratch_dummy.buffer_rs_3[ir,1]
        buffer_4 = @view scratch_dummy.buffer_rs_4[ir,1]

        # Reconstruct w_∥^3 moment of g_e from already-calculated qpar
        third_moment = scratch_dummy.buffer_z_1
        dthird_moment_dz = scratch_dummy.buffer_z_2
        begin_z_region()
        @loop_z iz begin
            third_moment[iz] = 0.5 * qpar[iz] / ppar[iz] / vth[iz]
        end
        derivative_z!(dthird_moment_dz, third_moment, buffer_1, buffer_2,
                      buffer_3, buffer_4, z_spectral, z)

        begin_vperp_vpa_region()
        update_electron_speed_z!(z_advect[1], upar, vth, vpa.grid, ir)
        z_speed = @view z_advect[1].speed[:,:,:,ir]

        delta_p = allocate_shared_float(size(ppar)...)
        p_amplitude = epsilon * maximum(ppar)
        f = @view pdf.electron.norm[:,:,:,ir]
        begin_serial_region()
        @serial_region begin
            @. delta_p = p_amplitude * sin(2.0*π*test_wavenumber*z.grid/z.L)

            # Make sure initial condition has some z-variation. As f is 'moment kinetic' this
            # means f must have a non-Maxwellian part that varies in z.
            f .*= 1.0 .+ 1.0e-4 .* reshape(vpa.grid.^3, vpa.n, 1, 1) .* reshape(sin.(2.0.*π.*z.grid./z.L), 1, 1, z.n)
        end
        # Ensure initial electron distribution function obeys constraints
        hard_force_moment_constraints!(reshape(f, vpa.n, vperp.n, z.n, 1), moments, vpa)
        delta_f = allocate_shared_float(size(f)...)
        f_amplitude = epsilon * maximum(f)
        begin_serial_region()
        @serial_region begin
            # Use exp(sin()) in vpa so that perturbation does not have any symmetry that makes
            # low-order moments vanish exactly.
            delta_f .= f_amplitude .*
                       reshape(sin.(2.0.*π.*test_wavenumber.*z.grid./z.L), 1, 1, z.n) .*
                       reshape(exp.(sin.(2.0.*π.*test_wavenumber.*vpa.grid./vpa.L)) .- 1.0, vpa.n, 1, 1) .*
                       f
        end

        pdf_size = length(f)
        p_size = length(ppar)
        total_size = pdf_size + p_size

        jacobian_matrix = allocate_shared_float(total_size, total_size)
        begin_serial_region()
        @serial_region begin
            jacobian_matrix .= 0.0
            for row ∈ 1:total_size
                # Initialise identity matrix
                jacobian_matrix[row,row] = 1.0
            end
        end

        add_electron_energy_equation_to_Jacobian!(
            jacobian_matrix, f, dens, upar, ppar, vth, third_moment, ddens_dz, dupar_dz,
            dppar_dz, dthird_moment_dz, collisions, composition, z, vperp, vpa,
            z_spectral, num_diss_params, dt, ir; ppar_offset=pdf_size)

        # Test 'ADI Jacobians' before other tests, because residual_func() may modify some
        # variables (vth, etc.).

        jacobian_matrix_ADI_check = allocate_shared_float(total_size, total_size)

        @testset "ADI Jacobians - implicit z" begin
            # 'Implicit' and 'explicit' parts of Jacobian should add up to full Jacobian.
            begin_serial_region()
            @serial_region begin
                jacobian_matrix_ADI_check .= 0.0
                for row ∈ 1:total_size
                    # Initialise identity matrix
                    jacobian_matrix_ADI_check[row,row] = 1.0
                end
            end

            v_size = vperp.n * vpa.n

            @serial_region begin
                # Add 'implicit' contribution
                this_slice = total_size - z.n + 1:total_size
                @views add_electron_energy_equation_to_z_only_Jacobian!(
                    jacobian_matrix_ADI_check[this_slice,this_slice], dens, upar, ppar,
                    vth, third_moment, ddens_dz, dupar_dz, dppar_dz, dthird_moment_dz,
                    collisions, composition, z, vperp, vpa, z_spectral, num_diss_params,
                    dt, ir)
            end

            # Add 'explicit' contribution
            add_electron_energy_equation_to_Jacobian!(
                jacobian_matrix_ADI_check, f, dens, upar, ppar, vth, third_moment,
                ddens_dz, dupar_dz, dppar_dz, dthird_moment_dz, collisions, composition,
                z, vperp, vpa, z_spectral, num_diss_params, dt, ir, :explicit_v;
                ppar_offset=pdf_size)

            begin_serial_region()
            @serial_region begin
                @test elementwise_isapprox(jacobian_matrix_ADI_check, jacobian_matrix; rtol=0.0, atol=1.0e-15)
            end
        end

        @testset "ADI Jacobians - implicit v" begin
            # 'Implicit' and 'explicit' parts of Jacobian should add up to full Jacobian.
            begin_serial_region()
            @serial_region begin
                jacobian_matrix_ADI_check .= 0.0
                for row ∈ 1:total_size
                    # Initialise identity matrix
                    jacobian_matrix_ADI_check[row,row] = 1.0
                end
            end

            v_size = vperp.n * vpa.n

            # Add 'implicit' contribution
            begin_z_region()
            @loop_z iz begin
                this_slice = collect((iz - 1)*v_size + 1:iz*v_size)
                push!(this_slice, iz + pdf_size)
                @views add_electron_energy_equation_to_v_only_Jacobian!(
                    jacobian_matrix_ADI_check[this_slice,this_slice], f[:,:,iz], dens[iz],
                    upar[iz], ppar[iz], vth[iz], third_moment[iz], ddens_dz[iz],
                    dupar_dz[iz], dppar_dz[iz], dthird_moment_dz[iz], collisions,
                    composition, z, vperp, vpa, z_spectral, num_diss_params, dt, ir, iz)
            end

            # Add 'explicit' contribution
            add_electron_energy_equation_to_Jacobian!(
                jacobian_matrix_ADI_check, f, dens, upar, ppar, vth, third_moment,
                ddens_dz, dupar_dz, dppar_dz, dthird_moment_dz, collisions, composition,
                z, vperp, vpa, z_spectral, num_diss_params, dt, ir, :explicit_z;
                ppar_offset=pdf_size)

            begin_serial_region()
            @serial_region begin
                @test elementwise_isapprox(jacobian_matrix_ADI_check, jacobian_matrix; rtol=0.0, atol=1.0e-15)
            end
        end

        function residual_func!(residual, this_f, this_p)
            begin_z_region()
            @loop_z iz begin
                # update the electron thermal speed using the updated electron
                # parallel pressure
                vth[iz] = sqrt(abs(2.0 * this_p[iz] /
                                   (dens[iz] * composition.me_over_mi)))
            end
            # Calculate heat flux and derivatives using new_variables
            calculate_electron_qpar_from_pdf_no_r!(qpar, this_p, vth, this_f, vpa, ir)

            calculate_electron_moment_derivatives_no_r!(
                moments,
                (electron_density=dens,
                 electron_upar=upar,
                 electron_ppar=this_p),
                scratch_dummy, z, z_spectral,
                num_diss_params.electron.moment_dissipation_coefficient, ir)

            # electron_kinetic_equation_euler_update!() just adds dt*d(g_e)/dt to the
            # electron_pdf member of the first argument, so if we set the electron_pdf member
            # of the first argument to zero, and pass dt=1, then it will evaluate the time
            # derivative, which is the residual for a steady-state solution.
            begin_z_region()
            @loop_z iz begin
                residual[iz] = ppar[iz]
            end
            @views electron_energy_equation_no_r!(
                       residual, this_p, dens, upar, moments.ion.dens[:,ir],
                       moments.ion.upar[:,ir], moments.ion.ppar[:,ir],
                       moments.neutral.dens[:,ir], moments.neutral.uz[:,ir],
                       moments.neutral.pz[:,ir], moments.electron, collisions, dt,
                       composition, external_source_settings.electron, num_diss_params, z,
                       ir)
            # Now
            #   residual = f_electron_old + dt*RHS(f_electron_newvar)
            # so update to desired residual
            begin_z_region()
            @loop_z iz begin
                residual[iz] = this_p[iz] - residual[iz]
            end
        end

        original_residual = allocate_shared_float(size(ppar)...)
        perturbed_residual = allocate_shared_float(size(ppar)...)

        @testset "δf only" begin
            residual_func!(original_residual, f, ppar)
            residual_func!(perturbed_residual, f.+delta_f, ppar)

            begin_serial_region()
            @serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[1:pdf_size] .= vec(delta_f)
                residual_update_with_Jacobian = jacobian_matrix * delta_state
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[pdf_size+1:end]

                # Check f did not get perturbed by the Jacobian
                @test elementwise_isapprox(residual_update_with_Jacobian[1:pdf_size],
                                           delta_state[1:pdf_size]; atol=1.0e-15)

                norm_factor = generate_norm_factor(perturbed_residual)
                @test elementwise_isapprox(perturbed_residual ./ norm_factor,
                                           perturbed_with_Jacobian ./ norm_factor;
                                           rtol=0.0, atol=rtol)
            end
        end

        @testset "δp only" begin
            residual_func!(original_residual, f, ppar)
            residual_func!(perturbed_residual, f, ppar .+ delta_p)

            begin_serial_region()
            @serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[pdf_size+1:end] .= vec(delta_p)
                residual_update_with_Jacobian = jacobian_matrix * delta_state
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
            residual_func!(original_residual, f, ppar)
            residual_func!(perturbed_residual, f.+delta_f, ppar.+delta_p)

            begin_serial_region()
            @serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[1:pdf_size] .= vec(delta_f)
                delta_state[pdf_size+1:end] .= vec(delta_p)
                residual_update_with_Jacobian = jacobian_matrix * delta_state
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[pdf_size+1:end]

                # Check ppar did not get perturbed by the Jacobian
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

function test_ion_dt_forcing_of_electron_ppar(test_input; rtol=(1.5e1*epsilon)^2)
    test_input = deepcopy(test_input)
    test_input["output"]["run_name"] *= "_ion_dt_forcing_of_electron_ppar"
    println("    - ion_dt_forcing_of_electron_ppar")

    @testset "ion_dt_forcing_of_electron_ppar" begin
        # Suppress console output while running
        pdf, scratch, scratch_implicit, scratch_electron, t_params, vz, vr, vzeta, vpa,
            vperp, gyrophase, z, r, moments, fields, spectral_objects, advection_structs,
            composition, collisions, geometry, gyroavs, boundary_distributions,
            external_source_settings, num_diss_params, nl_solver_params, advance,
            advance_implicit, fp_arrays, scratch_dummy, manufactured_source_list,
            ascii_io, io_moments, io_dfns = get_mk_state(test_input)

        dens = @view moments.electron.dens[:,ir]
        upar = @view moments.electron.upar[:,ir]
        ppar = @view moments.electron.ppar[:,ir]
        vth = @view moments.electron.vth[:,ir]
        qpar = @view moments.electron.qpar[:,ir]
        z_spectral = spectral_objects.z_spectral
        vpa_spectral = spectral_objects.vpa_spectral
        z_advect = advection_structs.z_advect
        vpa_advect = advection_structs.vpa_advect

        begin_vperp_vpa_region()
        update_electron_speed_z!(z_advect[1], upar, vth, vpa.grid, ir)
        z_speed = @view z_advect[1].speed[:,:,:,ir]

        delta_p = allocate_shared_float(size(ppar)...)
        p_amplitude = epsilon * maximum(ppar)
        f = @view pdf.electron.norm[:,:,:,ir]
        begin_serial_region()
        @serial_region begin
            @. delta_p = p_amplitude * sin(2.0*π*test_wavenumber*z.grid/z.L)

            # Make sure initial condition has some z-variation. As f is 'moment kinetic' this
            # means f must have a non-Maxwellian part that varies in z.
            f .*= 1.0 .+ 1.0e-4 .* reshape(vpa.grid.^3, vpa.n, 1, 1) .* reshape(sin.(2.0.*π.*z.grid./z.L), 1, 1, z.n)
        end
        # Ensure initial electron distribution function obeys constraints
        hard_force_moment_constraints!(reshape(f, vpa.n, vperp.n, z.n, 1), moments, vpa)
        delta_f = allocate_shared_float(size(f)...)
        f_amplitude = epsilon * maximum(f)
        # Use exp(sin()) in vpa so that perturbation does not have any symmetry that makes
        # low-order moments vanish exactly.
        begin_serial_region()
        @serial_region begin
            delta_f .= f_amplitude .*
                       reshape(sin.(2.0.*π.*test_wavenumber.*z.grid./z.L), 1, 1, z.n) .*
                       reshape(exp.(sin.(2.0.*π.*test_wavenumber.*vpa.grid./vpa.L)) .- 1.0, vpa.n, 1, 1) .*
                       f
        end

        pdf_size = length(f)
        p_size = length(ppar)
        total_size = pdf_size + p_size

        jacobian_matrix = allocate_shared_float(total_size, total_size)
        begin_serial_region()
        @serial_region begin
            jacobian_matrix .= 0.0
            for row ∈ 1:total_size
                # Initialise identity matrix
                jacobian_matrix[row,row] = 1.0
            end
        end

        add_ion_dt_forcing_of_electron_ppar_to_Jacobian!(
            jacobian_matrix, z, dt, ion_dt, ir; ppar_offset=pdf_size)

        # Test 'ADI Jacobians' before other tests, because residual_func() may modify some
        # variables (vth, etc.).

        jacobian_matrix_ADI_check = allocate_shared_float(total_size, total_size)

        @testset "ADI Jacobians - implicit z" begin
            # 'Implicit' and 'explicit' parts of Jacobian should add up to full Jacobian.
            begin_serial_region()
            @serial_region begin
                jacobian_matrix_ADI_check .= 0.0
                for row ∈ 1:total_size
                    # Initialise identity matrix
                    jacobian_matrix_ADI_check[row,row] = 1.0
                end
            end

            v_size = vperp.n * vpa.n

            @serial_region begin
                # Add 'implicit' contribution
                this_slice = total_size - z.n + 1:total_size
                @views add_ion_dt_forcing_of_electron_ppar_to_z_only_Jacobian!(
                    jacobian_matrix_ADI_check[this_slice,this_slice], z, dt, ion_dt, ir)
            end

            # Add 'explicit' contribution
            add_ion_dt_forcing_of_electron_ppar_to_Jacobian!(
                jacobian_matrix_ADI_check, z, dt, ion_dt, ir, :explicit_v;
                ppar_offset=pdf_size)

            begin_serial_region()
            @serial_region begin
                @test elementwise_isapprox(jacobian_matrix_ADI_check, jacobian_matrix; rtol=0.0, atol=1.0e-15)
            end
        end

        @testset "ADI Jacobians - implicit v" begin
            # 'Implicit' and 'explicit' parts of Jacobian should add up to full Jacobian.
            begin_serial_region()
            @serial_region begin
                jacobian_matrix_ADI_check .= 0.0
                for row ∈ 1:total_size
                    # Initialise identity matrix
                    jacobian_matrix_ADI_check[row,row] = 1.0
                end
            end

            v_size = vperp.n * vpa.n

            # Add 'implicit' contribution
            begin_z_region()
            @loop_z iz begin
                this_slice = collect((iz - 1)*v_size + 1:iz*v_size)
                push!(this_slice, iz + pdf_size)
                @views add_ion_dt_forcing_of_electron_ppar_to_v_only_Jacobian!(
                    jacobian_matrix_ADI_check[this_slice,this_slice], z, dt, ion_dt, ir,
                    iz)
            end

            # Add 'explicit' contribution
            add_ion_dt_forcing_of_electron_ppar_to_Jacobian!(
                jacobian_matrix_ADI_check, z, dt, ion_dt, ir, :explicit_z;
                ppar_offset=pdf_size)

            begin_serial_region()
            @serial_region begin
                @test elementwise_isapprox(jacobian_matrix_ADI_check, jacobian_matrix; rtol=0.0, atol=1.0e-15)
            end
        end

        function residual_func!(residual, this_f, this_p)
            begin_z_region()
            @loop_z iz begin
                # update the electron thermal speed using the updated electron
                # parallel pressure
                vth[iz] = sqrt(abs(2.0 * this_p[iz] /
                                   (dens[iz] * composition.me_over_mi)))
            end
            # Calculate heat flux and derivatives using new_variables
            calculate_electron_qpar_from_pdf_no_r!(qpar, this_p, vth, this_f, vpa, ir)

            calculate_electron_moment_derivatives_no_r!(
                moments,
                (electron_density=dens,
                 electron_upar=upar,
                 electron_ppar=this_p),
                scratch_dummy, z, z_spectral,
                num_diss_params.electron.moment_dissipation_coefficient, ir)

            # electron_kinetic_equation_euler_update!() just adds dt*d(g_e)/dt to the
            # electron_pdf member of the first argument, so if we set the electron_pdf member
            # of the first argument to zero, and pass dt=1, then it will evaluate the time
            # derivative, which is the residual for a steady-state solution.
            begin_z_region()
            @loop_z iz begin
                residual[iz] = ppar[iz]
            end
            ppar_previous_ion_step = moments.electron.ppar
            begin_z_region()
            @loop_z iz begin
                # At this point, ppar_out = ppar_in + dt*RHS(ppar_in). Here we add a
                # source/damping term so that in the steady state of the electron
                # pseudo-timestepping iteration,
                #   RHS(ppar) - (ppar - ppar_previous_ion_step) / ion_dt = 0,
                # resulting in a backward-Euler step (as long as the pseudo-timestepping
                # loop converges).
                residual[iz] += -dt * (this_p[iz] - ppar_previous_ion_step[iz,ir]) / ion_dt
            end
            # Now
            #   residual = f_electron_old + dt*RHS(f_electron_newvar)
            # so update to desired residual
            begin_z_region()
            @loop_z iz begin
                residual[iz] = this_p[iz] - residual[iz]
            end
        end

        original_residual = allocate_shared_float(size(ppar)...)
        perturbed_residual = allocate_shared_float(size(ppar)...)

        @testset "δf only" begin
            residual_func!(original_residual, f, ppar)
            residual_func!(perturbed_residual, f.+delta_f, ppar)

            begin_serial_region()
            @serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[1:pdf_size] .= vec(delta_f)
                residual_update_with_Jacobian = jacobian_matrix * delta_state
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[pdf_size+1:end]

                # Check f did not get perturbed by the Jacobian
                @test elementwise_isapprox(residual_update_with_Jacobian[1:pdf_size],
                                           delta_state[1:pdf_size]; atol=1.0e-15)

                # No norm factor, because both perturbed residuals should be zero here, as
                # delta_f does not affect this term, and `ppar` is used as
                # `ppar_previous_ion_step` in this test, so the residuals are exactly zero if
                # there is no delta_p.
                @test elementwise_isapprox(perturbed_residual,
                                           perturbed_with_Jacobian;
                                           rtol=0.0, atol=1.0e-15)
            end
        end

        @testset "δp only" begin
            residual_func!(original_residual, f, ppar)
            residual_func!(perturbed_residual, f, ppar .+ delta_p)

            begin_serial_region()
            @serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[pdf_size+1:end] .= vec(delta_p)
                residual_update_with_Jacobian = jacobian_matrix * delta_state
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
            residual_func!(original_residual, f, ppar)
            residual_func!(perturbed_residual, f.+delta_f, ppar.+delta_p)

            begin_serial_region()
            @serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[1:pdf_size] .= vec(delta_f)
                delta_state[pdf_size+1:end] .= vec(delta_p)
                residual_update_with_Jacobian = jacobian_matrix * delta_state
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[pdf_size+1:end]

                # Check ppar did not get perturbed by the Jacobian
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
    test_input = deepcopy(test_input)
    test_input["output"]["run_name"] *= "_electron_kinetic_equation"
    println("    - electron_kinetic_equation")

    @testset "electron_kinetic_equation" begin
        # Suppress console output while running
        pdf, scratch, scratch_implicit, scratch_electron, t_params, vz, vr, vzeta, vpa,
            vperp, gyrophase, z, r, moments, fields, spectral_objects, advection_structs,
            composition, collisions, geometry, gyroavs, boundary_distributions,
            external_source_settings, num_diss_params, nl_solver_params, advance,
            advance_implicit, fp_arrays, scratch_dummy, manufactured_source_list,
            ascii_io, io_moments, io_dfns = get_mk_state(test_input)

        dens = @view moments.electron.dens[:,ir]
        upar = @view moments.electron.upar[:,ir]
        ppar = @view moments.electron.ppar[:,ir]
        vth = @view moments.electron.vth[:,ir]
        qpar = @view moments.electron.qpar[:,ir]
        z_spectral = spectral_objects.z_spectral
        vperp_spectral = spectral_objects.vperp_spectral
        vpa_spectral = spectral_objects.vpa_spectral
        z_advect = advection_structs.z_advect
        vpa_advect = advection_structs.vpa_advect

        delta_p = allocate_shared_float(size(ppar)...)
        p_amplitude = epsilon * maximum(ppar)
        f = @view pdf.electron.norm[:,:,:,ir]
        begin_serial_region()
        @serial_region begin
            @. delta_p = p_amplitude * sin(2.0*π*test_wavenumber*z.grid/z.L)

            # Make sure initial condition has some z-variation. As f is 'moment kinetic' this
            # means f must have a non-Maxwellian part that varies in z.
            f .*= 1.0 .+ 1.0e-4 .* reshape(vpa.grid.^3, vpa.n, 1, 1) .* reshape(sin.(2.0.*π.*z.grid./z.L), 1, 1, z.n)
        end
        # Ensure initial electron distribution function obeys constraints
        hard_force_moment_constraints!(reshape(f, vpa.n, vperp.n, z.n, 1), moments, vpa)
        delta_f = allocate_shared_float(size(f)...)
        f_amplitude = epsilon * maximum(f)
        # Use exp(sin()) in vpa so that perturbation does not have any symmetry that makes
        # low-order moments vanish exactly.
        begin_serial_region()
        @serial_region begin
            delta_f .= f_amplitude .*
                       reshape(sin.(2.0.*π.*test_wavenumber.*z.grid./z.L), 1, 1, z.n) .*
                       reshape(exp.(sin.(2.0.*π.*test_wavenumber.*vpa.grid./vpa.L)) .- 1.0, vpa.n, 1, 1) .*
                       f
        end

        pdf_size = length(f)
        p_size = length(ppar)
        total_size = pdf_size + p_size

        jacobian_matrix = allocate_shared_float(total_size, total_size)

        # Calculate this later, so that we can use `jacobian_matrix` as a temporary
        # buffer, to avoid allocating too much shared memory for the Github Actions CI
        # servers.
        #fill_electron_kinetic_equation_Jacobian!(
        #    jacobian_matrix, f, ppar, moments, collisions, composition, z, vperp, vpa,
        #    z_spectral, vperp_spectral, vpa_spectral, z_advect, vpa_advect, scratch_dummy,
        #    external_source_settings, num_diss_params, t_params.electron, ion_dt, ir,
        #    true)

        # Test 'ADI Jacobians' before other tests, because residual_func() may modify some
        # variables (vth, etc.).

        # Reconstruct w_∥^3 moment of g_e from already-calculated qpar
        buffer_1 = @view scratch_dummy.buffer_rs_1[ir,1]
        buffer_2 = @view scratch_dummy.buffer_rs_2[ir,1]
        buffer_3 = @view scratch_dummy.buffer_rs_3[ir,1]
        buffer_4 = @view scratch_dummy.buffer_rs_4[ir,1]
        third_moment = scratch_dummy.buffer_z_1
        dthird_moment_dz = scratch_dummy.buffer_z_2
        begin_z_region()
        @loop_z iz begin
            third_moment[iz] = 0.5 * qpar[iz] / ppar[iz] / vth[iz]
        end
        derivative_z!(dthird_moment_dz, third_moment, buffer_1, buffer_2, buffer_3,
                      buffer_4, z_spectral, z)

        z_speed = @view z_advect[1].speed[:,:,:,ir]

        dpdf_dz = @view scratch_dummy.buffer_vpavperpzr_1[:,:,:,ir]
        begin_vperp_vpa_region()
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
        begin_z_vperp_region()
        update_electron_speed_vpa!(vpa_advect[1], dens, upar, ppar, moments, vpa.grid,
                                   external_source_settings.electron, ir)
        @loop_z_vperp iz ivperp begin
            @views @. vpa_advect[1].adv_fac[:,ivperp,iz,ir] = -vpa_advect[1].speed[:,ivperp,iz,ir]
        end
        #calculate the upwind derivative of the electron pdf w.r.t. wpa
        @loop_z_vperp iz ivperp begin
            @views derivative!(dpdf_dvpa[:,ivperp,iz], f[:,ivperp,iz], vpa,
                               vpa_advect[1].adv_fac[:,ivperp,iz,ir], vpa_spectral)
        end

        zeroth_moment = z.scratch_shared
        first_moment = z.scratch_shared2
        second_moment = z.scratch_shared3
        begin_z_region()
        vpa_grid = vpa.grid
        vpa_wgts = vpa.wgts
        @loop_z iz begin
            @views zeroth_moment[iz] = integrate_over_vspace(f[:,1,iz], vpa_wgts)
            @views first_moment[iz] = integrate_over_vspace(f[:,1,iz], vpa_grid, vpa_wgts)
            @views second_moment[iz] = integrate_over_vspace(f[:,1,iz], vpa_grid, 2, vpa_wgts)
        end

        # Test 'ADI Jacobians' before other tests, because residual_func() may modify some
        # variables (vth, etc.).

        jacobian_matrix_ADI_check = allocate_shared_float(total_size, total_size)
        begin_serial_region()
        @serial_region begin
            # Need to explicitly initialise because
            # fill_electron_kinetic_equation_z_only_Jacobian_f!() and
            # fill_electron_kinetic_equation_z_only_Jacobian_ppar!()
            # only fill the diagonal-in-velocity-indices elements, so when applied to
            # a full matrix they would not initialise every element.
            jacobian_matrix_ADI_check .= 0.0
        end

        @testset "ADI Jacobians - implicit z" begin
            # 'Implicit' and 'explicit' parts of Jacobian should add up to full Jacobian.

            v_size = vperp.n * vpa.n

            # Add 'implicit' contribution
            begin_vperp_vpa_region()
            @loop_vperp_vpa ivperp ivpa begin
                this_slice = (ivperp - 1)*vpa.n + ivpa:v_size:(z.n - 1)*v_size + (ivperp - 1)*vpa.n + ivpa
                @views fill_electron_kinetic_equation_z_only_Jacobian_f!(
                    jacobian_matrix_ADI_check[this_slice,this_slice], f[ivpa,ivperp,:],
                    ppar, dpdf_dz[ivpa,ivperp,:], dpdf_dvpa[ivpa,ivperp,:], z_speed,
                    moments, zeroth_moment, first_moment, second_moment, third_moment,
                    dthird_moment_dz, collisions, composition, z, vperp, vpa, z_spectral,
                    vperp_spectral, vpa_spectral, z_advect, vpa_advect, scratch_dummy,
                    external_source_settings, num_diss_params, t_params.electron, ion_dt,
                    ir, ivperp, ivpa, true)
            end

            @serial_region begin
                # Add 'implicit' contribution
                this_slice = (pdf_size + 1):total_size
                @views fill_electron_kinetic_equation_z_only_Jacobian_ppar!(
                    jacobian_matrix_ADI_check[this_slice,this_slice], ppar, moments,
                    zeroth_moment, first_moment, second_moment, third_moment,
                    dthird_moment_dz, collisions, composition, z, vperp, vpa, z_spectral,
                    vperp_spectral, vpa_spectral, z_advect, vpa_advect, scratch_dummy,
                    external_source_settings, num_diss_params, t_params.electron, ion_dt,
                    ir, true)
            end

            # Add 'explicit' contribution
            # Use jacobian_matrix as a temporary buffer here.
            fill_electron_kinetic_equation_Jacobian!(
                jacobian_matrix, f, ppar, moments, collisions, composition, z, vperp, vpa,
                z_spectral, vperp_spectral, vpa_spectral, z_advect, vpa_advect,
                scratch_dummy, external_source_settings, num_diss_params,
                t_params.electron, ion_dt, ir, true, :explicit_v)
            begin_serial_region()
            @serial_region begin
                jacobian_matrix_ADI_check .+= jacobian_matrix
            end

            fill_electron_kinetic_equation_Jacobian!(
                jacobian_matrix, f, ppar, moments, collisions, composition, z, vperp, vpa,
                z_spectral, vperp_spectral, vpa_spectral, z_advect, vpa_advect, scratch_dummy,
                external_source_settings, num_diss_params, t_params.electron, ion_dt, ir,
                true)

            begin_serial_region()
            @serial_region begin
                # The settings for this test are a bit strange, due to trying to get the
                # finite-difference approximation to the Jacobian to agree with the
                # Jacobian matrix functions without being too messed up by floating-point
                # rounding errors. The result is that some entries in the Jacobian matrix
                # here are O(1.0e5), so it is important to use `rtol` here.
                @test elementwise_isapprox(jacobian_matrix_ADI_check, jacobian_matrix; rtol=1.0e-15, atol=1.0e-15)
            end
        end

        begin_serial_region()
        @serial_region begin
            # Need to explicitly initialise because
            # fill_electron_kinetic_equation_z_only_Jacobian_f!() and
            # fill_electron_kinetic_equation_z_only_Jacobian_ppar!()
            # only fill the diagonal-in-velocity-indices elements, so when applied to
            # a full matrix they would not initialise every element.
            jacobian_matrix_ADI_check .= 0.0
        end

        @testset "ADI Jacobians - implicit v" begin
            # 'Implicit' and 'explicit' parts of Jacobian should add up to full Jacobian.

            v_size = vperp.n * vpa.n

            # Add 'implicit' contribution
            begin_z_region()
            @loop_z iz begin
                this_slice = collect((iz - 1)*v_size + 1:iz*v_size)
                push!(this_slice, iz + pdf_size)
                @views fill_electron_kinetic_equation_v_only_Jacobian!(
                    jacobian_matrix_ADI_check[this_slice,this_slice], f[:,:,iz], ppar[iz],
                    dpdf_dz[:,:,iz], dpdf_dvpa[:,:,iz], z_speed, moments,
                    zeroth_moment[iz], first_moment[iz], second_moment[iz],
                    third_moment[iz], dthird_moment_dz[iz], collisions, composition, z,
                    vperp, vpa, z_spectral, vperp_spectral, vpa_spectral, z_advect,
                    vpa_advect, scratch_dummy, external_source_settings, num_diss_params,
                    t_params.electron, ion_dt, ir, iz, true)
            end

            # Add 'explicit' contribution
            # Use jacobian_matrix as a temporary buffer here.
            fill_electron_kinetic_equation_Jacobian!(
                jacobian_matrix, f, ppar, moments, collisions, composition, z, vperp, vpa,
                z_spectral, vperp_spectral, vpa_spectral, z_advect, vpa_advect,
                scratch_dummy, external_source_settings, num_diss_params,
                t_params.electron, ion_dt, ir, true, :explicit_z)

            begin_serial_region()
            @serial_region begin
                jacobian_matrix_ADI_check .+= jacobian_matrix
            end

            fill_electron_kinetic_equation_Jacobian!(
                jacobian_matrix, f, ppar, moments, collisions, composition, z, vperp, vpa,
                z_spectral, vperp_spectral, vpa_spectral, z_advect, vpa_advect, scratch_dummy,
                external_source_settings, num_diss_params, t_params.electron, ion_dt, ir,
                true)

            begin_serial_region()
            @serial_region begin
                # The settings for this test are a bit strange, due to trying to get the
                # finite-difference approximation to the Jacobian to agree with the
                # Jacobian matrix functions without being too messed up by floating-point
                # rounding errors. The result is that some entries in the Jacobian matrix
                # here are O(1.0e5), so it is important to use `rtol` here.
                @test elementwise_isapprox(jacobian_matrix_ADI_check, jacobian_matrix; rtol=1.0e-13, atol=1.0e-13)
            end
        end

        function residual_func!(residual_f, residual_p, this_f, this_p)
            begin_z_region()
            @loop_z iz begin
                # update the electron thermal speed using the updated electron
                # parallel pressure
                vth[iz] = sqrt(abs(2.0 * this_p[iz] /
                                   (dens[iz] * composition.me_over_mi)))
            end
            # Calculate heat flux and derivatives using new_variables
            calculate_electron_qpar_from_pdf_no_r!(qpar, this_p, vth, this_f, vpa, ir)

            calculate_electron_moment_derivatives_no_r!(
                moments,
                (electron_density=dens,
                 electron_upar=upar,
                 electron_ppar=this_p),
                scratch_dummy, z, z_spectral,
                num_diss_params.electron.moment_dissipation_coefficient, ir)

            # electron_kinetic_equation_euler_update!() just adds dt*d(g_e)/dt to the
            # electron_pdf member of the first argument, so if we set the electron_pdf member
            # of the first argument to zero, and pass dt=1, then it will evaluate the time
            # derivative, which is the residual for a steady-state solution.
            begin_z_vperp_vpa_region()
            @loop_z_vperp_vpa iz ivperp ivpa begin
                residual_f[ivpa,ivperp,iz] = f[ivpa,ivperp,iz]
            end
            begin_z_region()
            @loop_z iz begin
                residual_p[iz] = ppar[iz]
            end
            electron_kinetic_equation_euler_update!(
                residual_f, residual_p, this_f, this_p, moments, z, vperp, vpa,
                z_spectral, vpa_spectral, z_advect, vpa_advect, scratch_dummy, collisions,
                composition, external_source_settings, num_diss_params, t_params.electron,
                ir; evolve_ppar=true, ion_dt=ion_dt)
            # Now
            #   residual = f_electron_old + dt*RHS(f_electron_newvar)
            # so update to desired residual
            begin_z_vperp_vpa_region()
            @loop_z_vperp_vpa iz ivperp ivpa begin
                residual_f[ivpa,ivperp,iz] = this_f[ivpa,ivperp,iz] - residual_f[ivpa,ivperp,iz]
            end
            begin_z_region()
            @loop_z iz begin
                residual_p[iz] = this_p[iz] - residual_p[iz]
            end

            # Set residual to zero where pdf_electron is determined by boundary conditions.
            if vpa.n > 1
                begin_z_vperp_region()
                @loop_z_vperp iz ivperp begin
                    @views enforce_v_boundary_condition_local!(residual_f[:,ivperp,iz], vpa.bc,
                                                               vpa_advect[1].speed[:,ivperp,iz,ir],
                                                               num_diss_params.electron.vpa_dissipation_coefficient > 0.0,
                                                               vpa, vpa_spectral)
                end
            end
            if vperp.n > 1
                begin_z_vpa_region()
                enforce_vperp_boundary_condition!(residual_f, vperp.bc,
                                                  vperp, vperp_spectral, vperp_adv,
                                                  vperp_diffusion, ir)
            end
            if z.bc == "wall"
                error("z_bc = \"wall\" not supported here yet.")
            elseif (z.bc == "constant") && (z.irank == 0 || z.irank == z.nrank - 1)
                # Boundary conditions on incoming part of distribution function. Note
                # that as density, upar, ppar do not change in this implicit step,
                # f_electron_newvar, f_old, and residual should all be zero at exactly
                # the same set of grid points, so it is reasonable to zero-out
                # `residual` to impose the boundary condition. We impose this after
                # subtracting f_old in case rounding errors, etc. mean that at some
                # point f_old had a different boundary condition cut-off index.
                begin_vperp_vpa_region()
                v_unnorm = vpa.scratch
                zero = 1.0e-14
                if z.irank == 0
                    iz = 1
                    v_unnorm .= vpagrid_to_dzdt(vpa.grid, vth[iz], upar[iz], true, true)
                    @loop_vperp_vpa ivperp ivpa begin
                        if v_unnorm[ivpa] > -zero
                            residual_f[ivpa,ivperp,iz] = 0.0
                        end
                    end
                end
                if z.irank == z.nrank - 1
                    iz = z.n
                    v_unnorm .= vpagrid_to_dzdt(vpa.grid, vth[iz], upar[iz], true, true)
                    @loop_vperp_vpa ivperp ivpa begin
                        if v_unnorm[ivpa] < zero
                            residual_f[ivpa,ivperp,iz] = 0.0
                        end
                    end
                end
            end
            return nothing
        end

        original_residual_f = allocate_shared_float(size(f)...)
        original_residual_p = allocate_shared_float(size(ppar)...)
        perturbed_residual_f = allocate_shared_float(size(f)...)
        perturbed_residual_p = allocate_shared_float(size(ppar)...)

        @testset "δf only" begin
            residual_func!(original_residual_f, original_residual_p, f, ppar)
            residual_func!(perturbed_residual_f, perturbed_residual_p, f.+delta_f, ppar)

            begin_serial_region()
            @serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[1:pdf_size] .= vec(delta_f)
                residual_update_with_Jacobian = jacobian_matrix * delta_state
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
            residual_func!(original_residual_f, original_residual_p, f, ppar)
            residual_func!(perturbed_residual_f, perturbed_residual_p, f, ppar.+delta_p)

            begin_serial_region()
            @serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[pdf_size+1:end] .= vec(delta_p)
                residual_update_with_Jacobian = jacobian_matrix * delta_state
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
            residual_func!(original_residual_f, original_residual_p, f, ppar)
            residual_func!(perturbed_residual_f, perturbed_residual_p, f.+delta_f, ppar.+delta_p)

            begin_serial_region()
            @serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[1:pdf_size] .= vec(delta_f)
                delta_state[pdf_size+1:end] .= vec(delta_p)
                residual_update_with_Jacobian = jacobian_matrix * delta_state
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

function test_electron_wall_bc(test_input; atol=(5.0*epsilon)^2)
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
        ppar = @view moments.electron.ppar[:,ir]
        vth = @view moments.electron.vth[:,ir]
        qpar = @view moments.electron.qpar[:,ir]
        ddens_dz = @view moments.electron.ddens_dz[:,ir]
        dppar_dz = @view moments.electron.dppar_dz[:,ir]
        phi = @view fields.phi[:,ir]
        z_spectral = spectral_objects.z_spectral
        vperp_spectral = spectral_objects.vperp_spectral
        vpa_spectral = spectral_objects.vpa_spectral
        z_advect = advection_structs.z_advect
        vpa_advect = advection_structs.vpa_advect
        me = composition.me_over_mi

        buffer_1 = @view scratch_dummy.buffer_rs_1[ir,1]
        buffer_2 = @view scratch_dummy.buffer_rs_2[ir,1]
        buffer_3 = @view scratch_dummy.buffer_rs_3[ir,1]
        buffer_4 = @view scratch_dummy.buffer_rs_4[ir,1]

        v_size = vperp.n * vpa.n

        # Reconstruct w_∥^3 moment of g_e from already-calculated qpar
        third_moment = scratch_dummy.buffer_z_1
        dthird_moment_dz = scratch_dummy.buffer_z_2
        begin_z_region()
        @loop_z iz begin
            third_moment[iz] = 0.5 * qpar[iz] / ppar[iz] / vth[iz]
        end
        derivative_z!(dthird_moment_dz, third_moment, buffer_1, buffer_2,
                      buffer_3, buffer_4, z_spectral, z)

        begin_vperp_vpa_region()
        update_electron_speed_z!(z_advect[1], upar, vth, vpa.grid, ir)
        z_speed = @view z_advect[1].speed[:,:,:,ir]

        delta_p = allocate_shared_float(size(ppar)...)
        p_amplitude = epsilon * maximum(ppar)
        f = @view pdf.electron.norm[:,:,:,ir]
        begin_serial_region()
        @serial_region begin
            # Make sure initial condition has some z-variation. As f is 'moment kinetic' this
            # means f must have a non-Maxwellian part that varies in z.
            f .*= 1.0 .+ 1.0e-4 .* reshape(vpa.grid.^3, vpa.n, 1, 1) .* reshape(sin.(2.0.*π.*z.grid./z.L), 1, 1, z.n)
        end
        # Ensure initial electron distribution function obeys constraints
        hard_force_moment_constraints!(reshape(f, vpa.n, vperp.n, z.n, 1), moments, vpa)
        delta_f = allocate_shared_float(size(f)...)
        f_amplitude = epsilon * maximum(f)
        # Use exp(sin()) in vpa so that perturbation does not have any symmetry that makes
        # low-order moments vanish exactly.
        # For this test have no z-dependence in delta_f so that it does not vanish
        # at the z-boundaries
        begin_serial_region()
        @serial_region begin
            @. delta_p = p_amplitude

            delta_f .= f_amplitude .*
                       reshape(exp.(sin.(2.0.*π.*test_wavenumber.*vpa.grid./vpa.L)) .- 1.0, vpa.n, 1, 1) .*
                       f
        end

        pdf_size = length(f)
        p_size = length(ppar)
        total_size = pdf_size + p_size

        dpdf_dvpa = @view scratch_dummy.buffer_vpavperpzr_2[:,:,:,ir]
        begin_z_vperp_region()
        update_electron_speed_vpa!(vpa_advect[1], dens, upar, ppar, moments, vpa.grid,
                                   external_source_settings.electron, ir)
        @loop_z_vperp iz ivperp begin
            @views @. vpa_advect[1].adv_fac[:,ivperp,iz,ir] = -vpa_advect[1].speed[:,ivperp,iz,ir]
        end
        #calculate the upwind derivative of the electron pdf w.r.t. wpa
        @loop_z_vperp iz ivperp begin
            @views derivative!(dpdf_dvpa[:,ivperp,iz], f[:,ivperp,iz], vpa,
                               vpa_advect[1].adv_fac[:,ivperp,iz,ir], vpa_spectral)
        end

        jacobian_matrix = allocate_shared_float(total_size, total_size)
        begin_serial_region()
        @serial_region begin
            jacobian_matrix .= 0.0
        end
        begin_z_vperp_vpa_region()
        @loop_z_vperp_vpa iz ivperp ivpa begin
            if skip_f_electron_bc_points_in_Jacobian(iz, ivperp, ivpa, z, vperp, vpa, z_speed)
                continue
            end

            # Rows corresponding to pdf_electron
            row = (iz - 1) * v_size + (ivperp - 1) * vpa.n + ivpa

            # Initialise identity matrix in rows corresponding to points that are
            # not set by the boundary condition.
            jacobian_matrix[row,row] = 1.0
        end

        add_wall_boundary_condition_to_Jacobian!(
            jacobian_matrix, phi, f, ppar, vth, upar, z, vperp, vpa, vperp_spectral,
            vpa_spectral, vpa_advect, moments,
            num_diss_params.electron.vpa_dissipation_coefficient, me, ir)

#        # Test 'ADI Jacobians' before other tests, because residual_func() may modify some
#        # variables (vth, etc.).
#
#        jacobian_matrix_ADI_check = allocate_shared_float(total_size, total_size)
#
#        @testset "ADI Jacobians - implicit z" begin
#            # 'Implicit' and 'explicit' parts of Jacobian should add up to full Jacobian.
#            begin_serial_region()
#            @serial_region begin
#                jacobian_matrix_ADI_check .= 0.0
#                for row ∈ 1:total_size
#                    # Initialise identity matrix
#                    jacobian_matrix_ADI_check[row,row] = 1.0
#                end
#            end
#
#            # There is no 'implicit z' contribution for vpa advection
#
#            # Add 'explicit' contribution
#            add_electron_vpa_advection_to_Jacobian!(
#                jacobian_matrix_ADI_check, f, dens, upar, ppar, vth, third_moment,
#                dpdf_dvpa, ddens_dz, dppar_dz, dthird_moment_dz, moments, me, z, vperp,
#                vpa, z_spectral, vpa_spectral, vpa_advect, z_speed, scratch_dummy,
#                external_source_settings, dt, ir, :explicit_v; ppar_offset=pdf_size)
#
#            begin_serial_region()
#            @serial_region begin
#                @test elementwise_isapprox(jacobian_matrix_ADI_check, jacobian_matrix; rtol=0.0, atol=1.0e-15)
#            end
#        end
#
#        @testset "ADI Jacobians - implicit v" begin
#            # 'Implicit' and 'explicit' parts of Jacobian should add up to full Jacobian.
#            begin_serial_region()
#            @serial_region begin
#                jacobian_matrix_ADI_check .= 0.0
#                for row ∈ 1:total_size
#                    # Initialise identity matrix
#                    jacobian_matrix_ADI_check[row,row] = 1.0
#                end
#            end
#
#            v_size = vperp.n * vpa.n
#
#            # Add 'implicit' contribution
#            begin_z_region()
#            @loop_z iz begin
#                this_slice = collect((iz - 1)*v_size + 1:iz*v_size)
#                push!(this_slice, iz + pdf_size)
#                @views add_electron_vpa_advection_to_v_only_Jacobian!(
#                    jacobian_matrix_ADI_check[this_slice,this_slice], f[:,:,iz], dens[iz],
#                    upar[iz], ppar[iz], vth[iz], third_moment[iz], dpdf_dvpa[:,:,iz],
#                    ddens_dz[iz], dppar_dz[iz], dthird_moment_dz[iz], moments, me, z,
#                    vperp, vpa, z_spectral, vpa_spectral, vpa_advect, z_speed,
#                    scratch_dummy, external_source_settings, dt, ir, iz)
#            end
#
#            # Add 'explicit' contribution
#            add_electron_vpa_advection_to_Jacobian!(
#                jacobian_matrix_ADI_check, f, dens, upar, ppar, vth, third_moment,
#                dpdf_dvpa, ddens_dz, dppar_dz, dthird_moment_dz, moments, me, z, vperp,
#                vpa, z_spectral, vpa_spectral, vpa_advect, z_speed, scratch_dummy,
#                external_source_settings, dt, ir, :explicit_z; ppar_offset=pdf_size)
#
#            begin_serial_region()
#            @serial_region begin
#                @test elementwise_isapprox(jacobian_matrix_ADI_check, jacobian_matrix; rtol=0.0, atol=1.0e-15)
#            end
#        end

        function residual_func!(residual, this_f, this_p)
            begin_z_region()
            @loop_z iz begin
                # update the electron thermal speed using the updated electron
                # parallel pressure
                vth[iz] = sqrt(abs(2.0 * this_p[iz] /
                                   (dens[iz] * composition.me_over_mi)))
            end

before = copy(this_f)
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
            begin_z_vperp_vpa_region()
            @loop_z_vperp_vpa iz ivperp ivpa begin
                residual[ivpa,ivperp,iz] = f[ivpa,ivperp,iz]
            end
            # Now
            #   residual = f_electron_old + dt*RHS(f_electron_newvar)
            # so update to desired residual
            begin_z_vperp_vpa_region()
            @loop_z_vperp_vpa iz ivperp ivpa begin
                residual[ivpa,ivperp,iz] = this_f[ivpa,ivperp,iz] - residual[ivpa,ivperp,iz]
            end

            # Set residual to zero where pdf_electron is determined by boundary conditions.
            if vpa.n > 1
                begin_z_vperp_region()
                @loop_z_vperp iz ivperp begin
                    @views enforce_v_boundary_condition_local!(residual[:,ivperp,iz], vpa.bc,
                                                               vpa_advect[1].speed[:,ivperp,iz,ir],
                                                               num_diss_params.electron.vpa_dissipation_coefficient > 0.0,
                                                               vpa, vpa_spectral)
                end
            end
            if vperp.n > 1
                begin_z_vpa_region()
                enforce_vperp_boundary_condition!(residual, vperp.bc,
                                                  vperp, vperp_spectral, vperp_adv,
                                                  vperp_diffusion, ir)
            end
            if z.bc != "wall"
                error("Testing wall bc but z_bc != \"wall\".")
            end

            return nothing
        end

        original_residual = allocate_shared_float(size(f)...)
        perturbed_residual = allocate_shared_float(size(f)...)

        @testset "δf only" begin
            residual_func!(original_residual, f, ppar)
            residual_func!(perturbed_residual, f.+delta_f, ppar)

            begin_serial_region()
            @serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[1:pdf_size] .= vec(delta_f)
                residual_update_with_Jacobian = jacobian_matrix * delta_state
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[1:pdf_size]

                # Check ppar did not get perturbed by the Jacobian
                @test elementwise_isapprox(residual_update_with_Jacobian[pdf_size+1:end],
                                           zeros(p_size); atol=1.0e-15)

                # Use an absolute tolerance for this test because if we used a norm_factor
                # like the other tests, it would be zero to machine precision at some
                # points.
                @test elementwise_isapprox(perturbed_residual,
                                           reshape(perturbed_with_Jacobian, vpa.n, vperp.n, z.n);
                                           rtol=0.0, atol=atol)
            end
        end

        @testset "δp only" begin
            residual_func!(original_residual, f, ppar)
            residual_func!(perturbed_residual, copy(f), ppar.+delta_p)

            begin_serial_region()
            @serial_region begin
                delta_state = zeros(mk_float, total_size)
                delta_state[pdf_size+1:end] .= vec(delta_p)
                residual_update_with_Jacobian = jacobian_matrix * delta_state
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[1:pdf_size]

                # Check ppar did not get perturbed by the Jacobian
                @test elementwise_isapprox(residual_update_with_Jacobian[pdf_size+1:end],
                                           zeros(p_size); atol=1.0e-15)

                # Use an absolute tolerance for this test because if we used a norm_factor
                # like the other tests, it would be zero to machine precision at some
                # points.
                @test elementwise_isapprox(perturbed_residual,
                                           reshape(perturbed_with_Jacobian, vpa.n, vperp.n, z.n);
                                           rtol=0.0, atol=atol)
            end
        end
#        @testset "δp only" begin
#            residual_func!(original_residual, f, ppar)
#            residual_func!(perturbed_residual, f, ppar .+ delta_p)
#
#            begin_serial_region()
#            @serial_region begin
#                delta_state = zeros(mk_float, total_size)
#                delta_state[pdf_size+1:end] .= vec(delta_p)
#                residual_update_with_Jacobian = jacobian_matrix * delta_state
#                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[1:pdf_size]
#
#                # Check ppar did not get perturbed by the Jacobian
#                @test elementwise_isapprox(residual_update_with_Jacobian[pdf_size+1:end],
#                                           delta_state[pdf_size+1:end]; atol=1.0e-15)
#
#                # Divide out the z-average of the magnitude of perturbed_residual from the
#                # difference, so that different orders of magnitude at different w_∥ are all
#                # tested sensibly, but occasional small values of the residual do not make the
#                # test fail.
#                # Since we have already normalised, pass `rtol` to `atol` for the comparison.
#                norm_factor = generate_norm_factor(perturbed_residual)
#                @test elementwise_isapprox(perturbed_residual ./ norm_factor,
#                                           reshape(perturbed_with_Jacobian, vpa.n, vperp.n, z.n) ./ norm_factor;
#                                           rtol=0.0, atol=rtol)
#            end
#        end
#
#        @testset "δf and δp" begin
#            residual_func!(original_residual, f, ppar)
#            residual_func!(perturbed_residual, f.+delta_f, ppar.+delta_p)
#
#            begin_serial_region()
#            @serial_region begin
#                delta_state = zeros(mk_float, total_size)
#                delta_state[1:pdf_size] .= vec(delta_f)
#                delta_state[pdf_size+1:end] .= vec(delta_p)
#                residual_update_with_Jacobian = jacobian_matrix * delta_state
#                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[1:pdf_size]
#
#                # Check ppar did not get perturbed by the Jacobian
#                @test elementwise_isapprox(residual_update_with_Jacobian[pdf_size+1:end],
#                                           delta_state[pdf_size+1:end]; atol=1.0e-15)
#
#                # Divide out the z-average of the magnitude of perturbed_residual from the
#                # difference, so that different orders of magnitude at different w_∥ are all
#                # tested sensibly, but occasional small values of the residual do not make the
#                # test fail.
#                # Since we have already normalised, pass `rtol` to `atol` for the comparison.
#                norm_factor = generate_norm_factor(perturbed_residual)
#                @test elementwise_isapprox(perturbed_residual ./ norm_factor,
#                                           reshape(perturbed_with_Jacobian, vpa.n, vperp.n, z.n) ./ norm_factor;
#                                           rtol=0.0, atol=rtol)
#            end
#        end

        cleanup_mk_state!(ascii_io, io_moments, io_dfns)
    end

    return nothing
end

function runtests()
    # Create a temporary directory for test output
    test_output_directory = get_MPI_tempdir()
    test_input["output"]["base_directory"] = test_output_directory

    @testset "Jacobian matrix" verbose=use_verbose begin
        println("Jacobian matrix")

        test_electron_z_advection(test_input)
        test_electron_vpa_advection(test_input)
        test_contribution_from_electron_pdf_term(test_input)
        test_electron_dissipation_term(test_input)
        test_electron_krook_collisions(test_input)
        test_external_electron_source(test_input)
        test_electron_implicit_constraint_forcing(test_input)
        test_electron_energy_equation(test_input)
        test_ion_dt_forcing_of_electron_ppar(test_input)
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
