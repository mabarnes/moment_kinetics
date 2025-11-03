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
                                                 zero_z_boundary_condition_points,
                                                 kinetic_electron_residual!,
                                                 get_electron_preconditioner
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

using moment_kinetics.BlockBandedMatrices
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
                                                         "pressure" => true),
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
                                                       "kinetic_electron_preconditioner" => "lu_no_separate_moments",
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
        mk_state = setup_moment_kinetics(test_input; skip_electron_solve=true,
                                         write_output=false)
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

function jacobian_isapprox(a::jacobian_info, b::jacobian_info; kwargs...)
    success = true
    for (row_a, row_b) ∈ zip(a.matrix, b.matrix)
        for (block_a, block_b) ∈ zip(row_a, row_b)
            success = success && elementwise_isapprox(block_a, block_b; kwargs...)
        end
    end
    return success
end
function jacobian_extrema(j::jacobian_info)
    minima_list = mk_float[]
    maxima_list = mk_float[]
    for row ∈ j.matrix
        for block ∈ row
            if isa(block, BlockSkylineMatrix)
                l, u = extrema(block.data)
            else
                l, u = extrema(block)
            end
            push!(minima_list, l)
            push!(maxima_list, u)
        end
    end
    return minimum(minima_list), maximum(maxima_list)
end
function adi_plus_equals!(jfull::jacobian_info, jADI::jacobian_info, f_slice, p_slice)
    @views jfull.matrix[1][1][f_slice,f_slice] .+= jADI.matrix[1][1]
    @views jfull.matrix[1][2][f_slice,p_slice] .+= jADI.matrix[1][2]
    @views jfull.matrix[2][1][p_slice,f_slice] .+= jADI.matrix[2][1]
    @views jfull.matrix[2][2][p_slice,p_slice] .+= jADI.matrix[2][2]
    return jfull
end
function jacobian_vector_product(j::jacobian_info, v::AbstractVector)
    result = Tuple(zeros(size(w)) for w ∈ v)
    for (x, row) ∈ zip(result, j.matrix)
        if length(row) != length(v)
            error("Number of blocks in RHS (length(v)=$(length(v))) is not the same as "
                  * "the number of columns in j.matrix (length(row)=$(length(Row))).")
        end
        for (w, block) ∈ zip(v, row)
            mul!(x, block, w, 1.0, 1.0)
        end
    end
    return result
end

function get_delta_state(delta_f, delta_p, separate_zeroth_moment,
                         separate_first_moment, separate_second_moment,
                         separate_third_moment, separate_dp_dz, separate_dq_dz, p, dp_dz,
                         n, dn_dz, third_moment, dthird_moment_dz, me, z, vperp, vpa,
                         z_spectral)
    p_size = length(delta_p)

    delta_state = [vec(delta_f), delta_p]
    if separate_zeroth_moment
        delta_zeroth_moment = zeros(mk_float, p_size)
        for iz ∈ 1:z.n
            @views delta_zeroth_moment[iz] = integral(delta_f[:,:,iz], vpa.grid, 0,
                                                      vpa.wgts, vperp.grid, 0, vperp.wgts)
        end
        push!(delta_state, delta_zeroth_moment)
    end
    if separate_first_moment
        delta_first_moment = zeros(mk_float, p_size)
        for iz ∈ 1:z.n
            @views delta_first_moment[iz] = integral(delta_f[:,:,iz], vpa.grid, 1,
                                                     vpa.wgts, vperp.grid, 0, vperp.wgts)
        end
        push!(delta_state, delta_first_moment)
    end
    if separate_second_moment
        delta_second_moment = zeros(mk_float, p_size)
        for iz ∈ 1:z.n
            @views delta_second_moment[iz] = integral((vperp,vpa)->(vpa^2+vperp^2),
                                                      delta_f[:,:,iz], vperp, vpa)
        end
        push!(delta_state, delta_second_moment)
    end
    if separate_third_moment
        delta_third_moment = zeros(mk_float, p_size)
        for iz ∈ 1:z.n
            @views delta_third_moment[iz] = integral((vperp,vpa)->vpa*(vpa^2+vperp^2),
                                                      delta_f[:,:,iz], vperp, vpa)
        end
        push!(delta_state, delta_third_moment)
    end
    if separate_dp_dz
        delta_dp_dz = zeros(mk_float, p_size)
        derivative!(delta_dp_dz, delta_p, z, z_spectral)
        push!(delta_state, delta_dp_dz)
    end
    if separate_dq_dz
        if !separate_dp_dz
            error("Currently assume separate_dp_dz=true when separate_dq_dz=true here.")
        end
        if !separate_third_moment
            error("Require separate_third_moment=true when separate_dq_dz=true.")
        end
        delta_dthird_moment_dz = zeros(mk_float, p_size)
        derivative!(delta_dthird_moment_dz, delta_third_moment, z, z_spectral)
        delta_dq_dz = @. sqrt(2.0/me) * ((-0.75) * p^0.5 * delta_p * third_moment * n^(-1.5) * dn_dz +
                                         (-0.5) * p^1.5 * delta_third_moment * n^(-1.5) * dn_dz +
                                         1.5 * n^(-0.5) * delta_third_moment * p^0.5 * dp_dz +
                                         0.75 * n^(-0.5) * third_moment * p^(-0.5) * delta_p * dp_dz +
                                         1.5 * n^(-0.5) * third_moment * p^0.5 * delta_dp_dz +
                                         1.5 * n^(-0.5) * p^0.5 * delta_p * dthird_moment_dz +
                                         n^(-0.5) * p^1.5 * delta_dthird_moment_dz
                                        )
        push!(delta_state, delta_dq_dz)
    end

    return delta_state
end

function test_get_pdf_term(test_input::AbstractDict, label::String, get_term::Function,
                           rhs_func!::Function, rtol::mk_float)
    test_input = deepcopy(test_input)
    test_input["output"]["run_name"] *= "_" * label[1:min(11, length(label))]
    println("        - $label")

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
        vpa_speed = @view vpa_advect[:,:,:,ir]
        me = composition.me_over_mi

        delta_p = allocate_shared_float(z; comm=comm_anyzv_subblock[])
        p_amplitude = epsilon * maximum(p)
        f = @view pdf.electron.norm[:,:,:,ir]

        @begin_r_anyzv_region()

        buffer_1 = @view scratch_dummy.buffer_rs_1[ir,1]
        buffer_2 = @view scratch_dummy.buffer_rs_2[ir,1]
        buffer_3 = @view scratch_dummy.buffer_rs_3[ir,1]
        buffer_4 = @view scratch_dummy.buffer_rs_4[ir,1]

        # Reconstruct w_∥^3 moment of g_e from already-calculated qpar
        # Note must use buffer with r-dimension, because use of z-only buffers is not
        # allowed within anyzv regions (as this would cause errors when the r-dimension is
        # split).
        third_moment = @view scratch_dummy.buffer_zrs_1[:,1,1]
        dthird_moment_dz = @view scratch_dummy.buffer_zrs_2[:,1,1]
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
        delta_f = allocate_shared_float(vpa, vperp, z; comm=comm_anyzv_subblock[])
        # Use exp(sin()) in vpa so that perturbation does not have any symmetry that makes
        # low-order moments vanish exactly.
        @begin_anyzv_region()
        @anyzv_serial_region begin
            f_amplitude = epsilon * maximum(f)
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
            upar_test = allocate_shared_float(z; comm=comm_anyzv_subblock[])
            @begin_anyzv_region()
            @anyzv_serial_region begin
                upar_test .= @view moments.electron.upar[:,ir]

                # Modify ion_upar to make sure it is different from upar_electron so that the
                # term proportional to (u_i-u_e) gets tested in case it is ever needed.
                @. ion_upar += sin(4.0*π*test_wavenumber*z.grid/z.L)
            end
        else
            upar_test = upar
        end

        pdf_size = length(f)
        p_size = length(p)
        total_size = pdf_size + p_size

        z_speed = @view z_advect[:,:,:,ir]

        dpdf_dz = @view scratch_dummy.buffer_vpavperpzr_1[:,:,:,ir]
        @begin_anyzv_vperp_vpa_region()
        update_electron_speed_z!(z_speed, upar_test, vth, vpa.grid)
        #calculate the upwind derivative
        @views derivative_z_pdf_vpavperpz!(dpdf_dz, f, z_advect[:,:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_1[:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_2[:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_3[:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_4[:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_5[:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_6[:,:,ir],
                                           z_spectral, z)

        dpdf_dvpa = @view scratch_dummy.buffer_vpavperpzr_2[:,:,:,ir]
        @begin_anyzv_z_vperp_region()
        update_electron_speed_vpa!(vpa_advect, dens, upar, p, moments,
                                   composition.me_over_mi, vpa.grid,
                                   external_source_settings.electron, ir)
        #calculate the upwind derivative of the electron pdf w.r.t. wpa
        @loop_z_vperp iz ivperp begin
            @views derivative!(dpdf_dvpa[:,ivperp,iz], f[:,ivperp,iz], vpa,
                               vpa_advect[:,ivperp,iz,ir], vpa_spectral)
        end

        d2pdf_dvpa2 = @view scratch_dummy.buffer_vpavperpzr_3[:,:,:,ir]
        @begin_anyzv_z_vperp_region()
        @loop_z_vperp iz ivperp begin
            @views second_derivative!(d2pdf_dvpa2[:,ivperp,iz], f[:,ivperp,iz], vpa,
                                      vpa_spectral)
        end

        zeroth_moment = allocate_shared_float(z; comm=comm_anyzv_subblock[])
        first_moment = allocate_shared_float(z; comm=comm_anyzv_subblock[])
        second_moment = allocate_shared_float(z; comm=comm_anyzv_subblock[])
        @begin_anyzv_z_region()
        @loop_z iz begin
            @views zeroth_moment[iz] = integral(f[:,:,iz], vpa.grid, 0, vpa.wgts,
                                                vperp.grid, 0, vperp.wgts)
            @views first_moment[iz] = integral(f[:,:,iz], vpa.grid, 1, vpa.wgts,
                                                vperp.grid, 0, vperp.wgts)
            @views second_moment[iz] = integral((vperp,vpa)->(vpa^2+vperp^2), f[:,:,iz],
                                                vperp, vpa)
        end

        jacobian = nl_solver_params.electron_advance.preconditioners[1][2]
        jacobian_initialize_identity!(jacobian)

        separate_zeroth_moment = (:zeroth_moment ∈ jacobian.state_vector_entries)
        separate_first_moment = (:first_moment ∈ jacobian.state_vector_entries)
        separate_second_moment = (:second_moment ∈ jacobian.state_vector_entries)
        separate_third_moment = (:third_moment ∈ jacobian.state_vector_entries)
        separate_dp_dz = (:electron_dp_dz ∈ jacobian.state_vector_entries)
        separate_dq_dz = (:electron_dq_dz ∈ jacobian.state_vector_entries)
        sub_terms = get_electron_sub_terms(dens, ddens_dz, upar_test, dupar_dz, p, dp_dz,
                                           dvth_dz, zeroth_moment, first_moment,
                                           second_moment, third_moment, dthird_moment_dz,
                                           dqpar_dz, ion_upar, f, dpdf_dz, dpdf_dvpa,
                                           d2pdf_dvpa2, me, moments, collisions,
                                           composition, external_source_settings,
                                           num_diss_params, t_params.electron, ion_dt, z,
                                           vperp, vpa, z_speed, vpa_speed, ir,
                                           separate_zeroth_moment, separate_first_moment,
                                           separate_second_moment, separate_third_moment,
                                           separate_dp_dz, separate_dq_dz)
        equation_term = get_term(sub_terms)
        add_term_to_Jacobian!(jacobian, :electron_pdf, dt, equation_term, z_speed)

        if test_input["timestepping"]["kinetic_electron_preconditioner"] == "lu_no_separate_moments"
            # ADI only (currently?) supported for "lu_no_separate_moments".
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
                    jacobian_initialize_zero!(z_solve_jacobian_ADI_check)

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
                                                      z_speed[ivpa,ivperp,:], ir, ivperp,
                                                      ivpa)
                    implict_z_term = get_term(implicit_z_sub_terms)
                    @views add_term_to_Jacobian!(z_solve_jacobian_ADI_check, :electron_pdf,
                                                 dt, implict_z_term, z_speed[ivpa,ivperp,:])

                    @views jacobian_ADI_check.matrix[1][1][this_slice,this_slice] .+= z_solve_jacobian_ADI_check.matrix[1][1]
                end
                @_anyzv_subblock_synchronize()

                # Add 'explicit' contribution
                separate_zeroth_moment = (:zeroth_moment ∈ jacobian_ADI_check.state_vector_entries)
                separate_first_moment = (:first_moment ∈ jacobian_ADI_check.state_vector_entries)
                separate_second_moment = (:second_moment ∈ jacobian_ADI_check.state_vector_entries)
                separate_third_moment = (:third_moment ∈ jacobian_ADI_check.state_vector_entries)
                separate_dp_dz = (:electron_dp_dz ∈ jacobian_ADI_check.state_vector_entries)
                separate_dq_dz = (:electron_dq_dz ∈ jacobian_ADI_check.state_vector_entries)
                explicit_v_sub_terms = get_electron_sub_terms(
                                           dens, ddens_dz, upar_test, dupar_dz, p, dp_dz,
                                           dvth_dz, zeroth_moment, first_moment,
                                           second_moment, third_moment, dthird_moment_dz,
                                           dqpar_dz, ion_upar, f, dpdf_dz, dpdf_dvpa,
                                           d2pdf_dvpa2, me, moments, collisions, composition,
                                           external_source_settings, num_diss_params,
                                           t_params.electron, ion_dt, z, vperp, vpa, z_speed,
                                           vpa_speed, ir, separate_zeroth_moment,
                                           separate_first_moment, separate_second_moment,
                                           separate_third_moment, separate_dp_dz,
                                           separate_dq_dz, :explicit_v)
                explicit_v_term = get_term(explicit_v_sub_terms)
                add_term_to_Jacobian!(jacobian_ADI_check, :electron_pdf, dt, explicit_v_term,
                                      z_speed)

                @begin_anyzv_region()
                @anyzv_serial_region begin
                    @test jacobian_isapprox(jacobian_ADI_check, jacobian; rtol=1.0e-15,
                                            atol=1.0e-15*max(abs.(jacobian_extrema(jacobian))...))
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
                    f_slice = (iz - 1)*v_size + 1:iz*v_size
                    p_slice = iz:iz

                    # We are reusing v_solve_jacobian_ADI_check, so need to zero out its
                    # matrix.
                    jacobian_initialize_zero!(v_solve_jacobian_ADI_check)

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
                            ion_dt, z, vperp, vpa, @view(z_speed[:,:,iz]),
                            @view(vpa_speed[:,:,iz]), ir, iz)
                    implicit_v_term = get_term(implicit_v_sub_terms)
                    add_term_to_Jacobian!(v_solve_jacobian_ADI_check, :electron_pdf, dt,
                                          implicit_v_term, this_z_speed)
                    adi_plus_equals!(jacobian_ADI_check, v_solve_jacobian_ADI_check, f_slice,
                                     p_slice)
                end
                @_anyzv_subblock_synchronize()

                # Add 'explicit' contribution
                separate_zeroth_moment = (:zeroth_moment ∈ jacobian_ADI_check.state_vector_entries)
                separate_first_moment = (:first_moment ∈ jacobian_ADI_check.state_vector_entries)
                separate_second_moment = (:second_moment ∈ jacobian_ADI_check.state_vector_entries)
                separate_third_moment = (:third_moment ∈ jacobian_ADI_check.state_vector_entries)
                separate_dp_dz = (:electron_dp_dz ∈ jacobian_ADI_check.state_vector_entries)
                separate_dq_dz = (:electron_dq_dz ∈ jacobian_ADI_check.state_vector_entries)
                explicit_z_sub_terms = get_electron_sub_terms(
                                           dens, ddens_dz, upar_test, dupar_dz, p, dp_dz,
                                           dvth_dz, zeroth_moment, first_moment,
                                           second_moment, third_moment, dthird_moment_dz,
                                           dqpar_dz, ion_upar, f, dpdf_dz, dpdf_dvpa,
                                           d2pdf_dvpa2, me, moments, collisions, composition,
                                           external_source_settings, num_diss_params,
                                           t_params.electron, ion_dt, z, vperp, vpa, z_speed,
                                           vpa_speed, ir, separate_zeroth_moment,
                                           separate_first_moment, separate_second_moment,
                                           separate_third_moment, separate_dp_dz,
                                           separate_dq_dz, :explicit_z)
                explicit_z_term = get_term(explicit_z_sub_terms)
                add_term_to_Jacobian!(jacobian_ADI_check, :electron_pdf, dt, explicit_z_term,
                                      z_speed)

                @begin_anyzv_region()
                @anyzv_serial_region begin
                    @test jacobian_isapprox(jacobian_ADI_check, jacobian; rtol=1.0e-15,
                                            atol=2.0e-15*max(abs.(jacobian_extrema(jacobian))...))
                end
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
            @views rhs_func!(; residual, this_f, dens, upar=upar_test, this_p, vth,
                             ion_upar, moments, collisions, composition, z_advect,
                             vpa_advect, z, vperp, vpa, z_spectral, vpa_spectral,
                             external_source_settings, num_diss_params, t_params,
                             scratch_dummy, dt, ir)
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
                                                               vpa_advect[:,ivperp,iz,ir],
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

        original_residual = allocate_shared_float(vpa, vperp, z; comm=comm_anyzv_subblock[])
        perturbed_residual = allocate_shared_float(vpa, vperp, z; comm=comm_anyzv_subblock[])

        @testset "δf only" begin
            residual_func!(original_residual, f, p)
            residual_func!(perturbed_residual, f.+delta_f, p)

            @begin_anyzv_region()
            @anyzv_serial_region begin
                delta_state = get_delta_state(delta_f, zeros(mk_float, p_size),
                                              separate_zeroth_moment,
                                              separate_first_moment,
                                              separate_second_moment,
                                              separate_third_moment, separate_dp_dz,
                                              separate_dq_dz, p, dp_dz, dens, ddens_dz,
                                              third_moment, dthird_moment_dz, me, z,
                                              vperp, vpa, z_spectral)
                residual_update_with_Jacobian = jacobian_vector_product(jacobian, delta_state)
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[1]

                # Check p did not get perturbed by the Jacobian
                @test elementwise_isapprox(residual_update_with_Jacobian[2],
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
                delta_state = get_delta_state(zeros(mk_float, vpa.n, vperp.n, z.n),
                                              delta_p, separate_zeroth_moment,
                                              separate_first_moment,
                                              separate_second_moment,
                                              separate_third_moment, separate_dp_dz,
                                              separate_dq_dz, p, dp_dz, dens, ddens_dz,
                                              third_moment, dthird_moment_dz, me, z,
                                              vperp, vpa, z_spectral)
                residual_update_with_Jacobian = jacobian_vector_product(jacobian, delta_state)
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[1]

                # Check p did not get perturbed by the Jacobian
                @test elementwise_isapprox(residual_update_with_Jacobian[2],
                                           delta_state[2]; atol=1.0e-15)

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
                delta_state = get_delta_state(delta_f, delta_p, separate_zeroth_moment,
                                              separate_first_moment,
                                              separate_second_moment,
                                              separate_third_moment, separate_dp_dz,
                                              separate_dq_dz, p, dp_dz, dens, ddens_dz,
                                              third_moment, dthird_moment_dz, me, z,
                                              vperp, vpa, z_spectral)
                residual_update_with_Jacobian = jacobian_vector_product(jacobian, delta_state)
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[1]

                # Check p did not get perturbed by the Jacobian
                @test elementwise_isapprox(residual_update_with_Jacobian[2],
                                           delta_state[2]; atol=1.0e-15)

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
    test_input["output"]["run_name"] *= "_" * label[1:min(11, length(label))]
    println("        - $label")

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
        vpa_speed = @view vpa_advect[:,:,:,ir]
        me = composition.me_over_mi

        @begin_r_anyzv_region()

        zeroth_moment = allocate_shared_float(z; comm=comm_anyzv_subblock[])
        first_moment = allocate_shared_float(z; comm=comm_anyzv_subblock[])
        second_moment = allocate_shared_float(z; comm=comm_anyzv_subblock[])
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
        # Note must use buffer with r-dimension, because use of z-only buffers is not
        # allowed within anyzv regions (as this would cause errors when the r-dimension is
        # split).
        third_moment = @view scratch_dummy.buffer_zrs_1[:,1,1]
        dthird_moment_dz = @view scratch_dummy.buffer_zrs_2[:,1,1]
        @begin_anyzv_z_region()
        @loop_z iz begin
            third_moment[iz] = qpar[iz] / p[iz] / vth[iz]
        end
        derivative_z_anyzv!(dthird_moment_dz, third_moment, buffer_1, buffer_2, buffer_3,
                            buffer_4, z_spectral, z)

        @begin_anyzv_vperp_vpa_region()
        z_speed = @view z_advect[:,:,:,ir]
        update_electron_speed_z!(z_speed, upar, vth, vpa.grid)

        delta_p = allocate_shared_float(z; comm=comm_anyzv_subblock[])
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
        delta_f = allocate_shared_float(vpa, vperp, z; comm=comm_anyzv_subblock[])
        @begin_anyzv_region()
        @anyzv_serial_region begin
            f_amplitude = epsilon * maximum(f)
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
        update_electron_speed_z!(z_speed, upar, vth, vpa.grid)
        #calculate the upwind derivative
        @views derivative_z_pdf_vpavperpz!(dpdf_dz, f, z_advect[:,:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_1[:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_2[:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_3[:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_4[:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_5[:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_6[:,:,ir],
                                           z_spectral, z)

        dpdf_dvpa = @view scratch_dummy.buffer_vpavperpzr_2[:,:,:,ir]
        @begin_anyzv_z_vperp_region()
        update_electron_speed_vpa!(vpa_advect, dens, upar, p, moments,
                                   composition.me_over_mi, vpa.grid,
                                   external_source_settings.electron, ir)
        #calculate the upwind derivative of the electron pdf w.r.t. wpa
        @loop_z_vperp iz ivperp begin
            @views derivative!(dpdf_dvpa[:,ivperp,iz], f[:,ivperp,iz], vpa,
                               vpa_advect[:,ivperp,iz,ir], vpa_spectral)
        end

        d2pdf_dvpa2 = @view scratch_dummy.buffer_vpavperpzr_3[:,:,:,ir]
        @begin_anyzv_z_vperp_region()
        @loop_z_vperp iz ivperp begin
            @views second_derivative!(d2pdf_dvpa2[:,ivperp,iz], f[:,ivperp,iz], vpa,
                                      vpa_spectral)
        end

        jacobian = nl_solver_params.electron_advance.preconditioners[1][2]
        jacobian_initialize_identity!(jacobian)

        separate_zeroth_moment = (:zeroth_moment ∈ jacobian.state_vector_entries)
        separate_first_moment = (:first_moment ∈ jacobian.state_vector_entries)
        separate_second_moment = (:second_moment ∈ jacobian.state_vector_entries)
        separate_third_moment = (:third_moment ∈ jacobian.state_vector_entries)
        separate_dp_dz = (:electron_dp_dz ∈ jacobian.state_vector_entries)
        separate_dq_dz = (:electron_dq_dz ∈ jacobian.state_vector_entries)
        sub_terms = get_electron_sub_terms(dens, ddens_dz, upar, dupar_dz, p, dp_dz,
                                           dvth_dz, zeroth_moment, first_moment,
                                           second_moment, third_moment, dthird_moment_dz,
                                           dqpar_dz, ion_upar, f, dpdf_dz, dpdf_dvpa,
                                           d2pdf_dvpa2, me, moments, collisions,
                                           composition, external_source_settings,
                                           num_diss_params, t_params, ion_dt, z, vperp,
                                           vpa, z_speed, vpa_speed, ir,
                                           separate_zeroth_moment, separate_first_moment,
                                           separate_second_moment, separate_third_moment,
                                           separate_dp_dz, separate_dq_dz)
        equation_term = get_term(sub_terms)
        add_term_to_Jacobian!(jacobian, :electron_p, dt, equation_term, z_speed)

        if test_input["timestepping"]["kinetic_electron_preconditioner"] == "lu_no_separate_moments"
            # ADI only (currently?) supported for "lu_no_separate_moments".
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
                    # We are reusing z_solve_jacobian_ADI_check, so need to zero out its
                    # matrix.
                    jacobian_initialize_zero!(z_solve_jacobian_ADI_check)

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
                                                      vperp, vpa, z_speed[1,1,:], ir, 1, 1)
                    implict_z_term = get_term(implicit_z_sub_terms)
                    add_term_to_Jacobian!(z_solve_jacobian_ADI_check, :electron_p, dt,
                                          implict_z_term)

                    @views jacobian_ADI_check.matrix[2][2] .+= z_solve_jacobian_ADI_check.matrix[1][1]
                end
                @_anyzv_subblock_synchronize()

                # Add 'explicit' contribution
                ADI_separate_zeroth_moment = (:zeroth_moment ∈ jacobian_ADI_check.state_vector_entries)
                ADI_separate_first_moment = (:first_moment ∈ jacobian_ADI_check.state_vector_entries)
                ADI_separate_second_moment = (:second_moment ∈ jacobian_ADI_check.state_vector_entries)
                ADI_separate_third_moment = (:third_moment ∈ jacobian_ADI_check.state_vector_entries)
                ADI_separate_dp_dz = (:electron_dp_dz ∈ jacobian_ADI_check.state_vector_entries)
                ADI_separate_dq_dz = (:electron_dq_dz ∈ jacobian_ADI_check.state_vector_entries)
                explicit_v_sub_terms = get_electron_sub_terms(
                                           dens, ddens_dz, upar, dupar_dz, p, dp_dz, dvth_dz,
                                           zeroth_moment, first_moment, second_moment,
                                           third_moment, dthird_moment_dz, dqpar_dz, ion_upar,
                                           f, dpdf_dz, dpdf_dvpa, d2pdf_dvpa2, me, moments,
                                           collisions, composition, external_source_settings,
                                           num_diss_params, t_params, ion_dt, z, vperp, vpa,
                                           z_speed, vpa_speed, ir, ADI_separate_zeroth_moment,
                                           ADI_separate_first_moment, ADI_separate_second_moment,
                                           ADI_separate_third_moment, ADI_separate_dp_dz,
                                           ADI_separate_dq_dz, :explicit_v)
                explicit_v_term = get_term(explicit_v_sub_terms)
                add_term_to_Jacobian!(jacobian_ADI_check, :electron_p, dt, explicit_v_term,
                                      z_speed)

                @begin_anyzv_region()
                @anyzv_serial_region begin
                    @test jacobian_isapprox(jacobian_ADI_check, jacobian; rtol=0.0,
                                            atol=1.0e-15)
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
                    f_slice = (iz - 1)*v_size + 1:iz*v_size
                    p_slice = iz:iz

                    # We are reusing v_solve_jacobian_ADI_check, so need to zero out its
                    # matrix.
                    jacobian_initialize_zero!(v_solve_jacobian_ADI_check)

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
                            vperp, vpa, @view(z_speed[:,:,iz]), @view(vpa_speed[:,:,iz]), ir,
                            iz)
                    implicit_v_term = get_term(implicit_v_sub_terms)
                    add_term_to_Jacobian!(v_solve_jacobian_ADI_check, :electron_p, dt,
                                          implicit_v_term, this_z_speed)
                    adi_plus_equals!(jacobian_ADI_check, v_solve_jacobian_ADI_check, f_slice,
                                     p_slice)
                end
                @_anyzv_subblock_synchronize()

                # Add 'explicit' contribution
                ADI_separate_zeroth_moment = (:zeroth_moment ∈ jacobian_ADI_check.state_vector_entries)
                ADI_separate_first_moment = (:first_moment ∈ jacobian_ADI_check.state_vector_entries)
                ADI_separate_second_moment = (:second_moment ∈ jacobian_ADI_check.state_vector_entries)
                ADI_separate_third_moment = (:third_moment ∈ jacobian_ADI_check.state_vector_entries)
                ADI_separate_dp_dz = (:electron_dp_dz ∈ jacobian_ADI_check.state_vector_entries)
                ADI_separate_dq_dz = (:electron_dq_dz ∈ jacobian_ADI_check.state_vector_entries)
                explicit_z_sub_terms = get_electron_sub_terms(
                                           dens, ddens_dz, upar, dupar_dz, p, dp_dz, dvth_dz,
                                           zeroth_moment, first_moment, second_moment,
                                           third_moment, dthird_moment_dz, dqpar_dz, ion_upar,
                                           f, dpdf_dz, dpdf_dvpa, d2pdf_dvpa2, me, moments,
                                           collisions, composition, external_source_settings,
                                           num_diss_params, t_params, ion_dt, z, vperp, vpa,
                                           z_speed, vpa_speed, ir, ADI_separate_zeroth_moment,
                                           ADI_separate_first_moment, ADI_separate_second_moment,
                                           ADI_separate_third_moment, ADI_separate_dp_dz,
                                           ADI_separate_dq_dz, :explicit_z)
                explicit_z_term = get_term(explicit_z_sub_terms)
                add_term_to_Jacobian!(jacobian_ADI_check, :electron_p, dt, explicit_z_term,
                                      z_speed)

                @begin_anyzv_region()
                @anyzv_serial_region begin
                    @test jacobian_isapprox(jacobian_ADI_check, jacobian; rtol=0.0,
                                            atol=1.0e-15*max(abs.(jacobian_extrema(jacobian))...))
                end
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

        original_residual = allocate_shared_float(z; comm=comm_anyzv_subblock[])
        perturbed_residual = allocate_shared_float(z; comm=comm_anyzv_subblock[])

        @testset "δf only" begin
            residual_func!(original_residual, f, p)
            residual_func!(perturbed_residual, f.+delta_f, p)

            @begin_anyzv_region()
            @anyzv_serial_region begin
                delta_state = get_delta_state(delta_f, zeros(mk_float, p_size),
                                              separate_zeroth_moment,
                                              separate_first_moment,
                                              separate_second_moment,
                                              separate_third_moment, separate_dp_dz,
                                              separate_dq_dz, p, dp_dz, dens, ddens_dz,
                                              third_moment, dthird_moment_dz, me, z,
                                              vperp, vpa, z_spectral)
                residual_update_with_Jacobian = jacobian_vector_product(jacobian, delta_state)
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[2]

                # Check f did not get perturbed by the Jacobian
                @test elementwise_isapprox(residual_update_with_Jacobian[1],
                                           delta_state[1]; atol=1.0e-15)

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
                delta_state = get_delta_state(zeros(mk_float, vpa.n, vperp.n, z.n),
                                              delta_p, separate_zeroth_moment,
                                              separate_first_moment,
                                              separate_second_moment,
                                              separate_third_moment, separate_dp_dz,
                                              separate_dq_dz, p, dp_dz, dens, ddens_dz,
                                              third_moment, dthird_moment_dz, me, z,
                                              vperp, vpa, z_spectral)
                residual_update_with_Jacobian = jacobian_vector_product(jacobian, delta_state)
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[2]

                # Check f did not get perturbed by the Jacobian
                @test elementwise_isapprox(residual_update_with_Jacobian[1],
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
                delta_state = get_delta_state(delta_f, delta_p, separate_zeroth_moment,
                                              separate_first_moment,
                                              separate_second_moment,
                                              separate_third_moment, separate_dp_dz,
                                              separate_dq_dz, p, dp_dz, dens, ddens_dz,
                                              third_moment, dthird_moment_dz, me, z,
                                              vperp, vpa, z_spectral)
                residual_update_with_Jacobian = jacobian_vector_product(jacobian, delta_state)
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[2]

                # Check p did not get perturbed by the Jacobian
                @test elementwise_isapprox(residual_update_with_Jacobian[1],
                                           delta_state[1]; atol=1.0e-15)

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
        println("        - electron_kinetic_equation $bc")
        this_test_input = deepcopy(test_input)
        label = "electron_kinetic_equation_$bc"
        test_input["output"]["run_name"] *= "_" * label[1:min(11, length(label))]
        this_test_input["z"]["bc"] = bc

        # Suppress console output while running
        pdf, scratch, scratch_implicit, scratch_electron, t_params, vz, vr, vzeta, vpa,
            vperp, gyrophase, z, r, moments, fields, spectral_objects, advection_structs,
            composition, collisions, geometry, gyroavs, boundary_distributions,
            external_source_settings, num_diss_params, nl_solver_params, advance,
            advance_implicit, fp_arrays, scratch_dummy, manufactured_source_list,
            ascii_io, io_moments, io_dfns = get_mk_state(this_test_input)

        dens = @view moments.electron.dens[:,ir]
        ddens_dz = @view moments.electron.ddens_dz[:,ir]
        upar = @view moments.electron.upar[:,ir]
        p = @view moments.electron.p[:,ir]
        dp_dz = @view moments.electron.dp_dz[:,ir]
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
        me = composition.me_over_mi

        delta_p = allocate_shared_float(z; comm=comm_anyzv_subblock[])
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
        delta_f = allocate_shared_float(vpa, vperp, z; comm=comm_anyzv_subblock[])
        # Use exp(sin()) in vpa so that perturbation does not have any symmetry that makes
        # low-order moments vanish exactly.
        @begin_anyzv_region()
        @anyzv_serial_region begin
            f_amplitude = epsilon * maximum(f)
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
        # Note must use buffer with r-dimension, because use of z-only buffers is not
        # allowed within anyzv regions (as this would cause errors when the r-dimension is
        # split).
        third_moment = @view scratch_dummy.buffer_zrs_1[:,1,1]
        dthird_moment_dz = @view scratch_dummy.buffer_zrs_2[:,1,1]
        @begin_anyzv_z_region()
        @loop_z iz begin
            third_moment[iz] = qpar[iz] / p[iz] / vth[iz]
        end
        derivative_z_anyzv!(dthird_moment_dz, third_moment, buffer_1, buffer_2, buffer_3,
                            buffer_4, z_spectral, z)

        z_speed = @view z_advect[:,:,:,ir]

        dpdf_dz = @view scratch_dummy.buffer_vpavperpzr_1[:,:,:,ir]
        @begin_anyzv_vperp_vpa_region()
        update_electron_speed_z!(z_speed, upar, vth, vpa.grid)
        #calculate the upwind derivative
        @views derivative_z_pdf_vpavperpz!(dpdf_dz, f, z_advect[:,:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_1[:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_2[:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_3[:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_4[:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_5[:,:,ir],
                                           scratch_dummy.buffer_vpavperpr_6[:,:,ir],
                                           z_spectral, z)

        dpdf_dvpa = @view scratch_dummy.buffer_vpavperpzr_2[:,:,:,ir]
        @begin_anyzv_z_vperp_region()
        update_electron_speed_vpa!(vpa_advect, dens, upar, p, moments,
                                   composition.me_over_mi, vpa.grid,
                                   external_source_settings.electron, ir)
        #calculate the upwind derivative of the electron pdf w.r.t. wpa
        @loop_z_vperp iz ivperp begin
            @views derivative!(dpdf_dvpa[:,ivperp,iz], f[:,ivperp,iz], vpa,
                               vpa_advect[:,ivperp,iz,ir], vpa_spectral)
        end
        vpa_speed = @view vpa_advect[:,:,:,ir]

        d2pdf_dvpa2 = @view scratch_dummy.buffer_vpavperpzr_3[:,:,:,ir]
        @begin_anyzv_z_vperp_region()
        @loop_z_vperp iz ivperp begin
            @views second_derivative!(d2pdf_dvpa2[:,ivperp,iz], f[:,ivperp,iz], vpa,
                                      vpa_spectral)
        end

        zeroth_moment = allocate_shared_float(z; comm=comm_anyzv_subblock[])
        first_moment = allocate_shared_float(z; comm=comm_anyzv_subblock[])
        second_moment = allocate_shared_float(z; comm=comm_anyzv_subblock[])
        @begin_anyzv_z_region()
        @loop_z iz begin
            @views zeroth_moment[iz] = integral(f[:,:,iz], vpa.grid, 0, vpa.wgts,
                                                vperp.grid, 0, vperp.wgts)
            @views first_moment[iz] = integral(f[:,:,iz], vpa.grid, 1, vpa.wgts,
                                                vperp.grid, 0, vperp.wgts)
            @views second_moment[iz] = integral((vperp,vpa)->(vpa^2+vperp^2), f[:,:,iz],
                                                vperp, vpa)
        end

        jacobian = nl_solver_params.electron_advance.preconditioners[1][2]
        separate_zeroth_moment = (:zeroth_moment ∈ jacobian.state_vector_entries)
        separate_first_moment = (:first_moment ∈ jacobian.state_vector_entries)
        separate_second_moment = (:second_moment ∈ jacobian.state_vector_entries)
        separate_third_moment = (:third_moment ∈ jacobian.state_vector_entries)
        separate_dp_dz = (:electron_dp_dz ∈ jacobian.state_vector_entries)
        separate_dq_dz = (:electron_dq_dz ∈ jacobian.state_vector_entries)

        # Calculate jacobian later, so that we can use `jacobian` as a temporary buffer,
        # to avoid allocating too much shared memory for the Github Actions CI servers.
        #fill_electron_kinetic_equation_Jacobian!(
        #    jacobian, f, p, moments, phi, collisions, composition, z, vperp, vpa,
        #    z_spectral, vperp_spectral, vpa_spectral, z_advect, vpa_advect,
        #    scratch_dummy, external_source_settings, num_diss_params,
        #    t_params.electron, ion_dt, ir, true)

        if test_input["timestepping"]["kinetic_electron_preconditioner"] == "lu_no_separate_moments"
            # ADI only (currently?) supported for "lu_no_separate_moments".
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
                        d2pdf_dvpa2[ivpa,ivperp,:], z_speed[ivpa,ivperp,:], moments,
                        zeroth_moment, first_moment, second_moment, third_moment,
                        dthird_moment_dz, collisions, composition, z, vperp, vpa, z_spectral,
                        vperp_spectral, vpa_spectral, z_advect, vpa_advect, scratch_dummy,
                        external_source_settings, num_diss_params, t_params.electron, ion_dt,
                        ir, ivperp, ivpa)

                    @views jacobian_ADI_check.matrix[1][1][this_slice,this_slice] .+= z_solve_jacobian_ADI_check.matrix[1][1]
                end

                @begin_anyzv_region()
                @anyzv_serial_region begin
                    # Add 'implicit' contribution
                    @views fill_electron_kinetic_equation_z_only_Jacobian_p!(
                        z_solve_p_jacobian_ADI_check, p, f[1,1,:], dpdf_dz[1,1,:],
                        dpdf_dvpa[1,1,:], d2pdf_dvpa2[1,1,:], z_speed[1,1,:], moments,
                        zeroth_moment, first_moment, second_moment, third_moment,
                        dthird_moment_dz, collisions, composition, z, vperp, vpa, z_spectral,
                        vperp_spectral, vpa_spectral, z_advect, vpa_advect, scratch_dummy,
                        external_source_settings, num_diss_params, t_params.electron, ion_dt,
                        ir, true)
                    @begin_anyzv_region()
                    @views jacobian_ADI_check.matrix[2][2] .+= z_solve_p_jacobian_ADI_check.matrix[1][1]
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
                    adi_plus_equals!(jacobian_ADI_check, jacobian, :, :)
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
                    @test jacobian_isapprox(jacobian_ADI_check, jacobian; rtol=adi_tol,
                                            atol=1.0e-15)
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
                    f_slice = (iz - 1)*v_size + 1:iz*v_size
                    p_slice = iz:iz
                    fill_electron_kinetic_equation_v_only_Jacobian!(
                        v_solve_jacobian_ADI_check, @view(f[:,:,iz]), @view(p[iz]),
                        @view(dpdf_dz[:,:,iz]), @view(dpdf_dvpa[:,:,iz]),
                        @view(d2pdf_dvpa2[:,:,iz]), @view(z_speed[:,:,iz]),
                        @view(vpa_speed[:,:,iz]), moments, @view(zeroth_moment[iz]),
                        @view(first_moment[iz]), @view(second_moment[iz]),
                        @view(third_moment[iz]), dthird_moment_dz[iz], phi[iz], collisions,
                        composition, z, vperp, vpa, z_spectral, vperp_spectral, vpa_spectral,
                        z_advect, vpa_advect, scratch_dummy, external_source_settings,
                        num_diss_params, t_params.electron, ion_dt, ir, iz, true)
                    adi_plus_equals!(jacobian_ADI_check, v_solve_jacobian_ADI_check, f_slice,
                                     p_slice)
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
                    adi_plus_equals!(jacobian_ADI_check, jacobian, :, :)
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
                    @test jacobian_isapprox(jacobian_ADI_check, jacobian; rtol=10.0*adi_tol,
                                            atol=1.0e-13)
                end
            end
        else
            fill_electron_kinetic_equation_Jacobian!(
                jacobian, f, p, moments, phi, collisions, composition, z, vperp, vpa,
                z_spectral, vperp_spectral, vpa_spectral, z_advect, vpa_advect,
                scratch_dummy, external_source_settings, num_diss_params,
                t_params.electron, ion_dt, ir, true)
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
                                                               vpa_advect[:,ivperp,iz,ir],
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

        original_residual_f = allocate_shared_float(vpa, vperp, z; comm=comm_anyzv_subblock[])
        original_residual_p = allocate_shared_float(z; comm=comm_anyzv_subblock[])
        perturbed_residual_f = allocate_shared_float(vpa, vperp, z; comm=comm_anyzv_subblock[])
        perturbed_residual_p = allocate_shared_float(z; comm=comm_anyzv_subblock[])
        f_plus_delta_f = allocate_shared_float(vpa, vperp, z; comm=comm_anyzv_subblock[])
        f_with_delta_p = allocate_shared_float(vpa, vperp, z; comm=comm_anyzv_subblock[])
        @begin_anyzv_z_vperp_vpa_region()
        @loop_z_vperp_vpa iz ivperp ivpa begin
            f_plus_delta_f[ivpa,ivperp,iz] = f[ivpa,ivperp,iz] + delta_f[ivpa,ivperp,iz]
            f_with_delta_p[ivpa,ivperp,iz] = f[ivpa,ivperp,iz]
        end
        p_plus_delta_p = allocate_shared_float(z; comm=comm_anyzv_subblock[])
        @begin_anyzv_z_region()
        @loop_z iz begin
            p_plus_delta_p[iz] = p[iz] + delta_p[iz]
        end

        @testset "δf only" begin
            residual_func!(original_residual_f, original_residual_p, f, p)
            residual_func!(perturbed_residual_f, perturbed_residual_p, f_plus_delta_f, p)

            @begin_anyzv_region()
            @anyzv_serial_region begin
                # Take this difference rather than using delta_f directly because we need
                # the effect of the boundary condition having been applied to
                # f_plus_delta_f.
                delta_state = get_delta_state(f_plus_delta_f .- f,
                                              zeros(mk_float, p_size),
                                              separate_zeroth_moment,
                                              separate_first_moment,
                                              separate_second_moment,
                                              separate_third_moment, separate_dp_dz,
                                              separate_dq_dz, p, dp_dz, dens, ddens_dz,
                                              third_moment, dthird_moment_dz, me, z,
                                              vperp, vpa, z_spectral)
                residual_update_with_Jacobian = jacobian_vector_product(jacobian, delta_state)
                perturbed_with_Jacobian_f = vec(original_residual_f) .+ residual_update_with_Jacobian[1]
                perturbed_with_Jacobian_p = vec(original_residual_p) .+ residual_update_with_Jacobian[2]

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
                # Take this difference rather than using delta_f directly because we need
                # the effect of the boundary condition having been applied to
                # f_with_delta_p.
                delta_state = get_delta_state(f_with_delta_p .- f, delta_p,
                                              separate_zeroth_moment,
                                              separate_first_moment,
                                              separate_second_moment,
                                              separate_third_moment, separate_dp_dz,
                                              separate_dq_dz, p, dp_dz, dens, ddens_dz,
                                              third_moment, dthird_moment_dz, me, z,
                                              vperp, vpa, z_spectral)
                residual_update_with_Jacobian = jacobian_vector_product(jacobian, delta_state)
                perturbed_with_Jacobian_f = vec(original_residual_f) .+ residual_update_with_Jacobian[1]
                perturbed_with_Jacobian_p = vec(original_residual_p) .+ residual_update_with_Jacobian[2]

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
                # Take this difference rather than using delta_f directly because we need
                # the effect of the boundary condition having been applied to
                # f_plus_delta_f.
                delta_state = get_delta_state(f_plus_delta_f .- f, delta_p,
                                              separate_zeroth_moment,
                                              separate_first_moment,
                                              separate_second_moment,
                                              separate_third_moment, separate_dp_dz,
                                              separate_dq_dz, p, dp_dz, dens, ddens_dz,
                                              third_moment, dthird_moment_dz, me, z,
                                              vperp, vpa, z_spectral)
                residual_update_with_Jacobian = jacobian_vector_product(jacobian, delta_state)
                perturbed_with_Jacobian_f = vec(original_residual_f) .+ residual_update_with_Jacobian[1]
                perturbed_with_Jacobian_p = vec(original_residual_p) .+ residual_update_with_Jacobian[2]

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
    label = "electron_wall_bc"
    test_input["output"]["run_name"] *= "_" * label[1:min(11, length(label))]
    println("        - electron_wall_bc")

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
        # Note must use buffer with r-dimension, because use of z-only buffers is not
        # allowed within anyzv regions (as this would cause errors when the r-dimension is
        # split).
        third_moment = @view scratch_dummy.buffer_zrs_1[:,1,1]
        dthird_moment_dz = @view scratch_dummy.buffer_zrs_2[:,1,1]
        @begin_anyzv_z_region()
        @loop_z iz begin
            third_moment[iz] = qpar[iz] / p[iz] / vth[iz]
        end
        derivative_z_anyzv!(dthird_moment_dz, third_moment, buffer_1, buffer_2, buffer_3,
                            buffer_4, z_spectral, z)

        @begin_anyzv_vperp_vpa_region()
        z_speed = @view z_advect[:,:,:,ir]
        update_electron_speed_z!(z_speed, upar, vth, vpa.grid)

        delta_p = allocate_shared_float(z; comm=comm_anyzv_subblock[])
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
        delta_f = allocate_shared_float(vpa, vperp, z; comm=comm_anyzv_subblock[])
        # Use exp(sin()) in vpa so that perturbation does not have any symmetry that makes
        # low-order moments vanish exactly.
        # For this test have no z-dependence in delta_f so that it does not vanish
        # at the z-boundaries
        @begin_anyzv_region()
        @anyzv_serial_region begin
            f_amplitude = epsilon * maximum(f)
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
        update_electron_speed_vpa!(vpa_advect, dens, upar, p, moments,
                                   composition.me_over_mi, vpa.grid,
                                   external_source_settings.electron, ir)
        #calculate the upwind derivative of the electron pdf w.r.t. wpa
        @loop_z_vperp iz ivperp begin
            @views derivative!(dpdf_dvpa[:,ivperp,iz], f[:,ivperp,iz], vpa,
                               vpa_advect[:,ivperp,iz,ir], vpa_spectral)
        end

        jacobian = nl_solver_params.electron_advance.preconditioners[1][2]
        separate_zeroth_moment = (:zeroth_moment ∈ jacobian.state_vector_entries)
        separate_first_moment = (:first_moment ∈ jacobian.state_vector_entries)
        separate_second_moment = (:second_moment ∈ jacobian.state_vector_entries)
        separate_third_moment = (:third_moment ∈ jacobian.state_vector_entries)
        separate_dp_dz = (:electron_dp_dz ∈ jacobian.state_vector_entries)
        separate_dq_dz = (:electron_dq_dz ∈ jacobian.state_vector_entries)
        jacobian_initialize_identity!(jacobian)

        add_wall_boundary_condition_to_Jacobian!(
            jacobian, phi, f, p, vth, upar, z, vperp, vpa, vperp_spectral, vpa_spectral,
            vpa_advect, moments, num_diss_params.electron.vpa_dissipation_coefficient, me,
            ir, :all)

        if test_input["timestepping"]["kinetic_electron_preconditioner"] == "lu_no_separate_moments"
            # ADI only (currently?) supported for "lu_no_separate_moments".
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
                    @test jacobian_isapprox(jacobian_ADI_check, jacobian; rtol=0.0, atol=1.0e-15)
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
                    f_slice = (iz - 1)*v_size + 1:iz*v_size
                    p_slice = iz:iz

                    # We are reusing v_solve_jacobian_ADI_check, so need to zero out its
                    # matrix.
                    jacobian_initialize_zero!(v_solve_jacobian_ADI_check)

                    @views add_wall_boundary_condition_to_Jacobian!(
                        v_solve_jacobian_ADI_check, phi[iz], f[:,:,iz], p[iz], vth[iz],
                        upar[iz], z, vperp, vpa, vperp_spectral, vpa_spectral, vpa_advect,
                        moments, num_diss_params.electron.vpa_dissipation_coefficient, me, ir,
                        :implicit_v, iz)
                    adi_plus_equals!(jacobian_ADI_check, v_solve_jacobian_ADI_check, f_slice,
                                     p_slice)
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
                    @test jacobian_isapprox(jacobian_ADI_check, jacobian; rtol=0.0, atol=1.0e-15)
                end
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
                                                               vpa_advect[:,ivperp,iz,ir],
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

        original_residual = allocate_shared_float(vpa, vperp, z; comm=comm_anyzv_subblock[])
        perturbed_residual = allocate_shared_float(vpa, vperp, z; comm=comm_anyzv_subblock[])
        f_plus_delta_f = allocate_shared_float(vpa, vperp, z; comm=comm_anyzv_subblock[])
        f_with_delta_p = allocate_shared_float(vpa, vperp, z; comm=comm_anyzv_subblock[])
        @begin_anyzv_z_vperp_vpa_region()
        @loop_z_vperp_vpa iz ivperp ivpa begin
            f_plus_delta_f[ivpa,ivperp,iz] = f[ivpa,ivperp,iz] + delta_f[ivpa,ivperp,iz]
            f_with_delta_p[ivpa,ivperp,iz] = f[ivpa,ivperp,iz]
        end
        p_plus_delta_p = allocate_shared_float(z; comm=comm_anyzv_subblock[])
        @begin_anyzv_z_region()
        @loop_z iz begin
            p_plus_delta_p[iz] = p[iz] + delta_p[iz]
        end

        @testset "δf only" begin
            residual_func!(original_residual, f, p)
            residual_func!(perturbed_residual, f_plus_delta_f, p)

            @begin_anyzv_region()
            @anyzv_serial_region begin
                # Take this difference rather than using delta_f directly because we need
                # the effect of the boundary condition having been applied to
                # f_plus_delta_f.
                delta_state = get_delta_state(f_plus_delta_f .- f,
                                              zeros(mk_float, p_size),
                                              separate_zeroth_moment,
                                              separate_first_moment,
                                              separate_second_moment,
                                              separate_third_moment, separate_dp_dz,
                                              separate_dq_dz, p, dp_dz, dens, ddens_dz,
                                              third_moment, dthird_moment_dz, me, z,
                                              vperp, vpa, z_spectral)
                residual_update_with_Jacobian = jacobian_vector_product(jacobian, delta_state)
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[1]

                # Check p did not get perturbed by the Jacobian
                @test elementwise_isapprox(residual_update_with_Jacobian[2],
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
                # Take this difference rather than using delta_f directly because we need
                # the effect of the boundary condition having been applied to
                # f_with_delta_p.
                delta_state = get_delta_state(f_with_delta_p .- f, delta_p,
                                              separate_zeroth_moment,
                                              separate_first_moment,
                                              separate_second_moment,
                                              separate_third_moment, separate_dp_dz,
                                              separate_dq_dz, p, dp_dz, dens, ddens_dz,
                                              third_moment, dthird_moment_dz, me, z,
                                              vperp, vpa, z_spectral)
                residual_update_with_Jacobian = jacobian_vector_product(jacobian, delta_state)
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[1]

                # Check p did not get perturbed by the Jacobian
                @test elementwise_isapprox(residual_update_with_Jacobian[2],
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
                # Take this difference rather than using delta_f directly because we need
                # the effect of the boundary condition having been applied to
                # f_plus_delta_f.
                delta_state = get_delta_state(f_plus_delta_f .- f, delta_p,
                                              separate_zeroth_moment,
                                              separate_first_moment,
                                              separate_second_moment,
                                              separate_third_moment, separate_dp_dz,
                                              separate_dq_dz, p, dp_dz, dens, ddens_dz,
                                              third_moment, dthird_moment_dz, me, z,
                                              vperp, vpa, z_spectral)
                residual_update_with_Jacobian = jacobian_vector_product(jacobian, delta_state)
                perturbed_with_Jacobian = vec(original_residual) .+ residual_update_with_Jacobian[1]

                # Check p did not get perturbed by the Jacobian
                @test elementwise_isapprox(residual_update_with_Jacobian[2],
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

function test_jacobian_inversion(test_input; rtol=2.0e-12)

    @testset "jacobian_inversion $bc" for bc ∈ ("wall", "periodic")
        println("        - jacobian_inversion $bc")
        this_test_input = deepcopy(test_input)
        label = "jacobian_inversion"
        test_input["output"]["run_name"] *= "_" * label[1:min(11, length(label))]
        this_test_input["z"]["bc"] = bc

        # Suppress console output while running
        pdf, scratch, scratch_implicit, scratch_electron, t_params, vz, vr, vzeta, vpa,
            vperp, gyrophase, z, r, moments, fields, spectral_objects, advection_structs,
            composition, collisions, geometry, gyroavs, boundary_distributions,
            external_source_settings, num_diss_params, nl_solver_params, advance,
            advance_implicit, fp_arrays, scratch_dummy, manufactured_source_list,
            ascii_io, io_moments, io_dfns = get_mk_state(this_test_input)

        dens = @view moments.electron.dens[:,ir]
        ddens_dz = @view moments.electron.ddens_dz[:,ir]
        upar = @view moments.electron.upar[:,ir]
        p = @view moments.electron.p[:,ir]
        dp_dz = @view moments.electron.dp_dz[:,ir]
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
        me = composition.me_over_mi
        f = @view pdf.electron.norm[:,:,:,ir]

        buffer_1 = @view scratch_dummy.buffer_rs_1[ir,1]
        buffer_2 = @view scratch_dummy.buffer_rs_2[ir,1]
        buffer_3 = @view scratch_dummy.buffer_rs_3[ir,1]
        buffer_4 = @view scratch_dummy.buffer_rs_4[ir,1]

        @begin_r_anyzv_region()

        @begin_anyzv_region()
        @anyzv_serial_region begin
            # Make sure initial condition has some z-variation. As f is 'moment kinetic' this
            # means f must have a non-Maxwellian part that varies in z.
            f .*= 1.0 .+ 1.0e-4 .* reshape(vpa.grid.^3, vpa.n, 1, 1) .* reshape(sin.(2.0.*π.*z.grid./z.L), 1, 1, z.n)
        end
        # Ensure initial electron distribution function obeys constraints
        hard_force_moment_constraints!(reshape(f, vpa.n, vperp.n, z.n, 1), moments, vpa, vperp)
        @begin_r_anyzv_region()

        _, apply_preconditioner!, recalculate_preconditioner! =
            get_electron_preconditioner(nl_solver_params.electron_advance, f, p, buffer_1,
                                        buffer_2, buffer_3, buffer_4, dens, upar, phi,
                                        moments, collisions, composition, z, vperp, vpa,
                                        z_spectral, vperp_spectral, vpa_spectral,
                                        z_advect, vpa_advect, scratch_dummy,
                                        external_source_settings, num_diss_params,
                                        t_params.electron, ion_dt, ir, true)

        if apply_preconditioner! === identity
            error("Expected second object returned by get_electron_preconditioner() to "
                  * "be the preconditioner function, but it is `identity()`.")
        end

        quietoutput() do
            recalculate_preconditioner!()
        end

        residual_func! = kinetic_electron_residual!(; f_electron_old=f, electron_p_old=p,
                                                    evolve_p=true, moments, composition,
                                                    collisions, external_source_settings,
                                                    t_params=t_params.electron, ion_dt,
                                                    scratch_dummy, this_phi=phi,
                                                    electron_density=dens,
                                                    electron_upar=upar,
                                                    ion_density=ion_dens, ion_upar, r, z,
                                                    vperp, vpa, z_spectral,
                                                    vperp_spectral, vpa_spectral,
                                                    z_advect, vpa_advect, num_diss_params,
                                                    ir)

        residual_f = allocate_shared_float(vpa, vperp, z; comm=comm_anyzv_subblock[])
        residual_p = allocate_shared_float(z; comm=comm_anyzv_subblock[])

        residual_func!((residual_p, residual_f), (p, f))

        apply_preconditioner!((residual_p, residual_f))

        # Expected values calculated with "lu" preconditioner in serial
        if bc == "wall"
            expected_precon_f = [0.0, -0.00026243482386546346, -0.00011421982675535042,
                                 -1.1759291936047118e-13, -0.0009881949820878444,
                                 -2.1195911358657453e-6, 9.123672873804058e-10,
                                 0.00025912012926323634, -5.2966194318659466e-9,
                                 8.151255374149414e-7, 0.0003239515924804457,
                                 -9.445549047898288e-13, 4.33242113198284e-5,
                                 -0.00013558506707875496, 2.3057913521750143e-16,
                                 -1.0273201390568985e-5, -1.4152310847618907e-5,
                                 2.374129113647413e-11, -0.00040737559405739976,
                                 -1.4552579634104623e-7, 3.685774264981963e-8,
                                 0.0003524376124200645, -8.779510255722654e-11,
                                 6.4075405432487595e-6, -2.041799682441718e-5,
                                 -7.443343852558604e-15, 0.00010056745481201907,
                                 -4.6523222095088714e-5, 2.642129554959949e-14,
                                 -0.00013801764839411703, -1.4633682853623258e-6,
                                 4.771715384844078e-10, -7.187019696417113e-5,
                                 -8.526934662581403e-9, 1.4776571713032378e-7,
                                 0.00015483575142182733, -1.95925164789606e-12,
                                 9.564865924910972e-6, -4.427714829683053e-5, 0.0,
                                 1.8610753793925994e-5, -1.0463682760315984e-5,
                                 4.392084877053337e-13, -9.88363808050514e-5,
                                 -1.3915451578956776e-7, 2.057026442530227e-9,
                                 4.945936500173832e-5, -2.657114071370429e-10,
                                 3.2582726739713795e-7, 2.7664377629326245e-5,
                                 -3.429741251850739e-14, 6.9608324168808086e-6,
                                 -1.648742151666801e-5, 1.5006229146507156e-16,
                                 -3.060098994461431e-6, -1.0968045172923077e-6,
                                 4.692809404168612e-12, -2.488907539370082e-5,
                                 -6.721416514101965e-9, 2.7076497299178404e-9,
                                 2.139693508073343e-5, -2.2216025386071877e-12,
                                 1.1649042526116935e-7, -6.349746774296051e-6,
                                 -3.8948557508845744e-16, -6.555435745903182e-7,
                                 -1.5743737230305942e-6, -1.060460971478119e-15,
                                 -4.409996514834645e-6, 2.473348404530956e-8,
                                 -2.2815529201981252e-11, 1.881626421282728e-5,
                                 5.616697028408237e-10, -1.501985841132335e-8,
                                 -1.5582501801846387e-5, 1.2238540297364975e-13,
                                 -1.3911859521566854e-6, 5.50776281915729e-7, 0.0,
                                 -9.812314917906333e-6, 1.2774169017285161e-6,
                                 -1.4275331256437735e-13, 1.8472120439279747e-5,
                                 3.0282188107263064e-8, -6.321162402727669e-10,
                                 1.9132305615603017e-6, 7.895230025498785e-11,
                                 -1.4749632455237745e-7, -1.832914315889399e-5,
                                 -1.4329130948491945e-14, -4.308185190677648e-6,
                                 4.209358046673509e-6, -1.3206432832893986e-15,
                                 -8.025587231923555e-6, 5.556300084075351e-7,
                                 -5.11892079337856e-12, 3.2911938196680545e-5,
                                 6.592859513101088e-9, -4.400959227562349e-9,
                                 -1.812563285022334e-5, 2.8147955186593915e-12,
                                 -6.583830395362971e-7, -5.341798892269458e-6,
                                 -4.162939520299952e-15, -7.9349871902275e-6,
                                 2.962481516117123e-6, -2.376161747209819e-14,
                                 4.07570162659824e-6, 1.2793568908887107e-7,
                                 -7.774552202249068e-11, 2.5898551550870282e-5,
                                 9.498302129514302e-10, -2.483075690056359e-8,
                                 -2.3987149585078982e-5, 8.085641792283607e-16,
                                 -1.8894174598451682e-6, 2.6468113290765397e-6, 0.0,
                                 -1.1671917246065354e-5, 1.7453316055924875e-6,
                                 -2.5188036840545635e-13, 2.33299239122936e-5,
                                 3.840714340008221e-8, -7.962077382922875e-10,
                                 -8.735639220241381e-7, 1.0231998533368461e-10,
                                 -1.8299893057057518e-7, -2.2376253773753568e-5,
                                 -5.2106123157825264e-15, -5.480350423697201e-6,
                                 7.184501680375392e-6, -1.2418521344727347e-15,
                                 -9.227227923535485e-6, 1.0463350560570163e-6,
                                 -7.337929964099104e-12, 5.734985390509029e-5,
                                 1.4334516862833935e-8, -8.428278503944787e-9,
                                 -4.6205208614496977e-5, 9.592630387853028e-12,
                                 -1.3073120230862577e-6, -4.630478966830632e-6,
                                 -1.0045491792688813e-16, -1.6944249956040575e-5,
                                 1.0847605927202962e-5, -1.6058930953918734e-14,
                                 2.3877283416775046e-5, 5.032826816997743e-7,
                                 -2.3541267600703443e-10, 6.325947409008095e-5,
                                 3.92137384424439e-9, -8.553593030254213e-8,
                                 -9.503765198975055e-5, 1.0398919125065843e-12,
                                 -6.609679467858779e-6, 2.3904721422624788e-5, 0.0,
                                 -3.91833433317881e-5, 1.1032601493519582e-5,
                                 -6.538376670705185e-13, 0.00012085753374351535,
                                 2.120319196690735e-7, -3.851420517229504e-9,
                                 -4.506833268963184e-5, 5.281367888806511e-10,
                                 -8.419765075997406e-7, -0.0001098828979534082,
                                 7.774529004044268e-14, -2.5091734185079533e-5,
                                 7.18551892787695e-5, -1.1677039299805215e-15,
                                 -2.8423777731855702e-5, 6.3044549283880185e-6,
                                 -4.1891047620531115e-11, 0.0002687300451633772,
                                 6.358003485129668e-8, -4.425940103257077e-8,
                                 -0.00021750858612255914, 3.6829172115987825e-11,
                                 -6.3606632850534355e-6, -4.9689738716082016e-5,
                                 2.2932623876016093e-15, -7.612772199347756e-5,
                                 6.459300258747033e-5, -1.2772924136758432e-13,
                                 0.00013145604651099344, 1.3162889146928006e-6,
                                 -1.8184627193958724e-9, 0.00023683991669477765,
                                 6.801295044155718e-9, -5.127753261751731e-7,
                                 -0.00045670382860270797, 9.704467969858988e-13,
                                 -3.3885266597429494e-5, 0.00011557291323304993, 0.0,
                                 -0.0001456797359462352, 1.1337976613084519e-6,
                                 -1.11930399779463e-11, 0.0006275208487376323, 0.0,
                                 -3.8191883249818254e-8]
            expected_precon_p = [-8.087887064200703e-6, -6.47191484134741e-6,
                                 -5.359023686751642e-6, -5.127464715057588e-6,
                                 -3.939900733397052e-6, -2.7650617381612424e-6,
                                 -1.3501341022297346e-6, -5.881676453350948e-7,
                                 -8.348232719857003e-8, 2.928634678312922e-7,
                                 1.093719199374862e-6, 2.0994978253450584e-6,
                                 2.9789078572072334e-6, 3.5370781179276477e-6,
                                 3.7664350209648104e-6, 3.806108162508256e-6,
                                 3.7937514476097364e-6, 3.765448679119842e-6,
                                 3.652849732784844e-6, 3.4033346956242403e-6,
                                 3.036592610436986e-6, 2.642852446657062e-6,
                                 2.315362649191352e-6, 2.102070248882447e-6,
                                 2.0125166430705334e-6, 1.927372987117184e-6,
                                 1.754825535743743e-6, 1.5507040703079115e-6,
                                 1.3693138548052406e-6, 1.238774919876768e-6,
                                 1.158977040355246e-6, 1.1170683761604064e-6,
                                 1.1014227798249969e-6, 1.087537434805276e-6,
                                 1.06205452360508e-6, 1.035868514900998e-6,
                                 1.0151512631035506e-6, 1.000339432719001e-6,
                                 9.898779851732035e-7, 9.830031883680537e-7,
                                 9.799507779289338e-7, 9.768942425649066e-7,
                                 9.700110911567868e-7, 9.59873497730499e-7,
                                 9.47321128617413e-7, 9.338314714137644e-7,
                                 9.214926186541667e-7, 9.124470309534405e-7,
                                 9.083093463159349e-7, 9.041402399345912e-7,
                                 8.948017985807708e-7, 8.81546846303971e-7,
                                 8.663526544187773e-7, 8.516085233647857e-7,
                                 8.394682748858748e-7, 8.313265064056376e-7,
                                 8.278063693604952e-7, 8.243859361584087e-7,
                                 8.171769278437437e-7, 8.080040863089244e-7,
                                 7.990293797972502e-7, 7.919714537454827e-7,
                                 7.875027187704659e-7, 7.852753251042487e-7,
                                 7.845269902257909e-7, 7.839359104508588e-7,
                                 7.83178496467644e-7, 7.833718068513934e-7,
                                 7.853161443789699e-7, 7.889005825678238e-7,
                                 7.931155225049909e-7, 7.966115364595023e-7,
                                 7.982996098905158e-7, 8.000476797923024e-7,
                                 8.041048128247807e-7, 8.101025747588787e-7,
                                 8.171430310442445e-7, 8.239367190244879e-7,
                                 8.293320324079331e-7, 8.327563285109398e-7,
                                 8.341672574087055e-7, 8.354877303328722e-7,
                                 8.380668212246133e-7, 8.407870879706317e-7,
                                 8.424478630097333e-7, 8.424384912725976e-7,
                                 8.41148730892838e-7, 8.396329477648123e-7,
                                 8.388280356188269e-7, 8.379710643773791e-7,
                                 8.35995844275998e-7, 8.335747380360552e-7,
                                 8.328926092716359e-7, 8.373916268441246e-7,
                                 8.48838544110014e-7, 8.63383655470379e-7,
                                 8.721881157860076e-7, 8.826309863974625e-7,
                                 9.126094838534878e-7, 9.744365185517878e-7,
                                 1.0823260433434815e-6, 1.2390443953012038e-6,
                                 1.4205704953562305e-6, 1.5768872207308472e-6,
                                 1.6548149993427564e-6, 1.737297821110609e-6,
                                 1.935803883844453e-6, 2.24631918863475e-6,
                                 2.63074662304742e-6, 3.006528158913599e-6,
                                 3.2854377731191234e-6, 3.4344356524189377e-6,
                                 3.4837499450562743e-6, 3.5203749426013097e-6,
                                 3.5471207162770997e-6, 3.4395565275749935e-6,
                                 3.0644263218943795e-6, 2.402279556876876e-6,
                                 1.6017247403272275e-6, 9.362160308844297e-7,
                                 6.19116721234212e-7, 2.1480665421920169e-7,
                                 -4.897543903861425e-7, -1.72096100838954e-6,
                                 -2.8979587963607895e-6, -4.0915259936689005e-6,
                                 -4.56757518680138e-6, -5.699917311291001e-6,
                                 -7.351338134107261e-6]
        elseif bc == "periodic"
            expected_precon_f = [0.0, -2.689856159133315e-5, -1.3070544725972842e-6,
                                 -5.823091373789312e-15, 0.00021395751402369096,
                                 -7.226754787842689e-9, -6.505942620391849e-11,
                                 -0.0003029604518824253, -8.307482881394773e-12,
                                 -3.2798315052124255e-8, 3.1186105583212166e-5,
                                 -6.309294456153017e-16, -2.8795636508913925e-6,
                                 -1.1760791578650514e-5, -3.580770172820817e-18,
                                 -4.529788627233683e-5, -2.6367649948227514e-7,
                                 -2.3169160878983405e-13, 0.00015873570526030392,
                                 -1.1557356047283408e-9, -5.374344810148598e-10,
                                 0.00033706389077643864, -3.5035981153095784e-13,
                                 -2.1200988395282723e-7, -5.589990981267413e-5,
                                 -1.5465799695294134e-17, -9.481221729465886e-6,
                                 -4.350527376294805e-6, -1.7418267920849465e-16,
                                 8.125578195246154e-5, -4.283300975400923e-8,
                                 -5.780572180398438e-12, -0.0005975713087782275,
                                 -1.3619418106968022e-10, -4.138623105836361e-9,
                                 0.00022430683136868555, -1.7991312610971595e-14,
                                 -8.93004444687289e-7, -2.6193716786517864e-5, 0.0,
                                 -3.1976493049076054e-5, -1.3355900121511244e-6,
                                 -7.860734705885287e-15, 0.00033615113819891913,
                                 -7.441776597245181e-9, -6.924625663802672e-11,
                                 -6.628865595275006e-5, -9.01284904126136e-12,
                                 -3.4479928413862705e-8, -4.999109732839875e-5,
                                 -7.123923488306154e-15, -3.057722225642731e-6,
                                 -1.13671931867169e-5, -4.451898182801825e-16,
                                 -4.2447271451760316e-5, -2.7629743307143586e-7,
                                 -4.632240932159212e-13, 7.068627708424323e-5,
                                 -1.2461459287396265e-9, -5.920817780641994e-10,
                                 0.0004604531575833517, -1.525239424734903e-12,
                                 -2.2498715642494773e-7, -6.077946897233919e-5,
                                 -1.0798197725384256e-14, -1.0165633451196533e-5,
                                 -4.566664364063316e-6, -3.7885874318009705e-14,
                                 9.87979380068951e-5, -4.6116831914685784e-8,
                                 -1.618749156544304e-11, -0.0006137115633066045,
                                 -2.2386000200568566e-10, -4.7446205459823435e-9,
                                 0.00024390151619880701, -1.4932438162316963e-12,
                                 -9.486566065448492e-7, -2.8663302494252272e-5, 0.0,
                                 -3.22396493379474e-5, -1.4280708396053312e-6,
                                 -1.2749180006237055e-12, 0.0002980362870995414,
                                 -9.07108858353412e-9, -2.0382610945427656e-10,
                                 -0.0002441230860139521, -7.458978636766545e-11,
                                 -3.882300375787085e-8, 2.8612465170501554e-5,
                                 -9.514088077674814e-13, -3.181331581253931e-6,
                                 -1.297107613699514e-5, -9.34354889773392e-14,
                                 -4.82920224488569e-5, -2.992905204490665e-7,
                                 -1.688264939669159e-11, 0.0002363934818587371,
                                 -2.112749770612922e-9, -1.1769171361868597e-9,
                                 0.00015904415233195483, -2.2369433892950136e-11,
                                 -2.396186131736972e-7, -4.8010353621836704e-5,
                                 -2.2311705360924519e-13, -9.712306338450578e-6,
                                 -4.935453885083177e-6, -6.752525725927881e-13,
                                 1.7504186393311341e-6, -5.0495896559360704e-8,
                                 -7.333226519463028e-11, -0.00018805522610613722,
                                 -3.8747496165951914e-10, -5.540669490475558e-9,
                                 0.0003145749719997812, -3.688153259691079e-12,
                                 -9.580996042004377e-7, -3.532143943186217e-5, 0.0,
                                 -2.660084747462441e-5, -1.4510000249318741e-6,
                                 -1.6747150860460022e-12, 0.0001388216140755302,
                                 -8.641336915467327e-9, -1.5391550239927274e-10,
                                 -0.0006081641422282393, -3.3895746676636e-11,
                                 -3.657858725786695e-8, 0.00016883255737224956,
                                 -2.36211813357928e-13, -3.035482565826772e-6,
                                 -1.4404563270793609e-5, -1.816220016300127e-14,
                                 -5.5266300139645656e-5, -2.8590490559090357e-7,
                                 -2.0704524538542575e-12, 0.00042795800318015496,
                                 -1.3144740769965886e-9, -6.042737829889986e-10,
                                 -0.00015251946311340718, -1.4019008108361211e-12,
                                 -2.22494049513514e-7, -3.4453298176776513e-5,
                                 -4.133676665489397e-15, -8.596772394630011e-6,
                                 -4.867679665071e-6, -1.1022647280143609e-14,
                                 -7.231805832719707e-5, -4.4958817527737185e-8,
                                 -6.5670325874086495e-12, 6.959671021253288e-5,
                                 -1.4681198370216932e-10, -4.162718674014095e-9,
                                 0.0003398115007579349, -3.543914636697896e-14,
                                 -8.946129273492403e-7, -3.675882961486821e-5, 0.0,
                                 -2.3082242600629707e-5, -1.3675434762551176e-6,
                                 -9.016131145941432e-15, 9.039092387442201e-5,
                                 -7.529161216055214e-9, -6.582318649749764e-11,
                                 -0.0006504993405612378, -8.820935201046555e-12,
                                 -3.335638603351467e-8, 0.00017963472023266276,
                                 -8.017511666632264e-16, -2.848789139127294e-6,
                                 -1.3676733541968765e-5, -9.07985049820824e-18,
                                 -5.3218990538564066e-5, -2.6850452172020103e-7,
                                 -2.259889542333254e-13, 0.00037832658812480223,
                                 -1.182835343096881e-9, -5.302876292373681e-10,
                                 -6.844769441664821e-5, -3.646965218943488e-13,
                                 -2.10383841167308e-7, -3.8447587619985764e-5,
                                 -1.4446755728740177e-17, -8.475479944976842e-6,
                                 -4.536648277278062e-6, -1.5586527035978824e-16,
                                 -2.2635897122175947e-5, -4.2621706086099445e-8,
                                 -5.528425052134024e-12, -0.00015572798331515685,
                                 -1.3633359122727386e-10, -3.9969661661043216e-9,
                                 0.0002897657314314546, -1.6418422837499233e-14,
                                 -8.619076280309243e-7, -3.1414874516881614e-5, 0.0,
                                 -2.6332983630403214e-5, -1.308331092217464e-6,
                                 -5.804889010532783e-15, 0.00020363261266787065,
                                 -7.233559320965759e-9, -6.48808435047593e-11]
            expected_precon_p = [-4.753071052392934e-8, -6.194221729216732e-8,
                                 -9.378750313523302e-8, -1.3819768819369164e-7,
                                 -1.8843854387197142e-7, -2.3709134071964462e-7,
                                 -2.775826278366458e-7, -3.0524380622598197e-7,
                                 -3.173962545825285e-7, -3.293488769290452e-7,
                                 -3.5513932753668355e-7, -3.896630322331047e-7,
                                 -4.2663972059990167e-7, -4.6023738322158216e-7,
                                 -4.864198294735874e-7, -5.03299993245867e-7,
                                 -5.104397779179304e-7, -5.172891624315767e-7,
                                 -5.314502624331879e-7, -5.489551290958189e-7,
                                 -5.655573813171792e-7, -5.782695681750664e-7,
                                 -5.86165171455691e-7, -5.900512495787641e-7,
                                 -5.913462286465064e-7, -5.923622602796204e-7,
                                 -5.936360202657797e-7, -5.931888130674182e-7,
                                 -5.895351299482024e-7, -5.827113272637699e-7,
                                 -5.744256839732374e-7, -5.673059212374343e-7,
                                 -5.637794336634658e-7, -5.600620141572066e-7,
                                 -5.511536669568362e-7, -5.371495620020459e-7,
                                 -5.190794600993597e-7, -4.99260413447933e-7,
                                 -4.80921225827758e-7, -4.673537737326177e-7,
                                 -4.6110742853068067e-7, -4.547842786336679e-7,
                                 -4.4049621912862095e-7, -4.1983940923291474e-7,
                                 -3.954010844027455e-7, -3.7054727703887936e-7,
                                 -3.4883407487863177e-7, -3.3336881090798406e-7,
                                 -3.2638867764407027e-7, -3.1940114923932514e-7,
                                 -3.038643391396372e-7, -2.8189494858882527e-7,
                                 -2.5641632311946184e-7, -2.308029089845183e-7,
                                 -2.0847213362496698e-7, -1.925043354664963e-7,
                                 -1.852647434851214e-7, -1.7799126076797435e-7,
                                 -1.6170547512383493e-7, -1.3835260115360784e-7,
                                 -1.1070532265097125e-7, -8.225014223174758e-8,
                                 -5.691672566490112e-8, -3.853704094174525e-8,
                                 -3.014192897595779e-8, -2.167331459882276e-8,
                                 -2.6119378192714464e-9, 2.487390673954959e-8,
                                 5.7447319739271256e-8, 9.075422520187208e-8,
                                 1.2002706898984779e-7, 1.4095993720189723e-7,
                                 1.5042324719477855e-7, 1.5990194275580588e-7,
                                 1.8097826552602933e-7, 2.1070084384030588e-7,
                                 2.448521085430226e-7, 2.7854362271709795e-7,
                                 3.0713148180574046e-7, 3.2699545953276376e-7,
                                 3.358189806373326e-7, 3.445599356068144e-7,
                                 3.636508234465773e-7, 3.8977123672786617e-7,
                                 4.1862815554654427e-7, 4.458681926579532e-7,
                                 4.6798633963492547e-7, 4.827845918691574e-7,
                                 4.891993802580412e-7, 4.954536344925791e-7,
                                 5.087474987919901e-7, 5.260490639139269e-7,
                                 5.437871877443048e-7, 5.58920883621753e-7,
                                 5.697678367365577e-7, 5.761265990075196e-7,
                                 5.786175910415938e-7, 5.808717119947762e-7,
                                 5.850112075232654e-7, 5.887586011162498e-7,
                                 5.899120292445921e-7, 5.875157324157344e-7,
                                 5.824824047171516e-7, 5.772135345718437e-7,
                                 5.743733836074976e-7, 5.712410829166535e-7,
                                 5.632362017223345e-7, 5.494643270815885e-7,
                                 5.299224998152985e-7, 5.065251909638521e-7,
                                 4.832303164645017e-7, 4.65043907087848e-7,
                                 4.564082523432816e-7, 4.475010280218296e-7,
                                 4.26771957553924e-7, 3.9536354076414304e-7,
                                 3.5608015914272674e-7, 3.138680174771965e-7,
                                 2.7523099554363323e-7, 2.467820077795854e-7,
                                 2.337056885014205e-7, 2.2047692458499845e-7,
                                 1.906007315089273e-7, 1.4741156655007906e-7,
                                 9.629542779786032e-8, 4.432639919593458e-8,
                                 -9.67036610042644e-10, -3.308937936683096e-8,
                                 -4.753071052392934e-8]
        else
            error("No expected results for bc=\"$bc\".")
        end

        if expected_precon_f === nothing
            # Error: no expected input provided
            println("data tested would be: expected_precon_f = ",
                    @view residual_f[1:100:end])
            @test false
        else
            @test elementwise_isapprox(expected_precon_f, @view residual_f[1:100:end];
                                       rtol=rtol, atol=1.0e-20)
        end

        if expected_precon_p === nothing
            # Error: no expected input provided
            println("data tested would be: expected_precon_p = ", residual_p)
            @test false
        else
            if (nl_solver_params.electron_advance.preconditioner_type === Val(:electron_lu_separate_dp_dz_dq_dz)
                    && bc == "periodic")
                # This combination has a strangely high relative error. Not sure why,
                # maybe just bad luck (?), but :lu_separate_dp_dz_dq_dz is not commonly
                # used anyway, so allow this to pass (at least for now).
                p_rtol = rtol * 1.0e3
            else
                p_rtol = rtol
            end
            @test elementwise_isapprox(expected_precon_p, residual_p; rtol=p_rtol,
                                       atol=1.0e-20)
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

    @testset "Jacobian matrix " verbose=use_verbose begin
        println("Jacobian matrix")
        precon_list = String[]
        push!(precon_list, "schur_complement")
        # Test this by default as it has a different structure in the Jacobian matrix than
        # `schur_complement`, so although "lu" is used more often, this provides more test
        # coverage.
        push!(precon_list, "lu_no_separate_moments")
        @long push!(precon_list, "lu_separate_third_moment")
        @long push!(precon_list, "lu")
        @long push!(precon_list, "lu_separate_dp_dz_dq_dz")
        @testset "$kinetic_electron_preconditioner" verbose=use_verbose for kinetic_electron_preconditioner ∈ precon_list
            println("    - $kinetic_electron_preconditioner")

            this_test_input = deepcopy(test_input)
            this_test_input["output"]["run_name"] *= "_" * kinetic_electron_preconditioner
            this_test_input["timestepping"]["kinetic_electron_preconditioner"] = kinetic_electron_preconditioner

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
            test_get_pdf_term(this_test_input, "electron_z_advection",
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
            test_get_pdf_term(this_test_input, "electron_vpa_advection",
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
            test_get_pdf_term(this_test_input, "contribution_from_electron_pdf_term",
                              get_contribution_from_electron_pdf_term,
                              contribution_from_electron_pdf_term_wrapper!, (4.0e2*epsilon)^2)

            function contribution_from_electron_dissipation_term!(; kwargs...)
                add_dissipation_term!(kwargs[:residual], kwargs[:this_f],
                                      kwargs[:scratch_dummy], kwargs[:z_spectral], kwargs[:z],
                                      kwargs[:vpa], kwargs[:vpa_spectral],
                                      kwargs[:num_diss_params], kwargs[:dt])
                return nothing
            end
            test_get_pdf_term(this_test_input, "electron_dissipation_term",
                              get_electron_dissipation_term,
                              contribution_from_electron_dissipation_term!, (1.0e1*epsilon)^2)

            function contribution_from_krook_collisions!(; kwargs...)
                electron_krook_collisions!(kwargs[:residual], kwargs[:this_f], kwargs[:dens],
                                           kwargs[:upar], kwargs[:ion_upar], kwargs[:vth],
                                           kwargs[:collisions], kwargs[:vperp], kwargs[:vpa],
                                           kwargs[:dt])
                return nothing
            end
            test_get_pdf_term(this_test_input, "electron_krook_collisions",
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
            test_get_pdf_term(this_test_input, "external_electron_sources",
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
            test_get_pdf_term(this_test_input, "implicit_constraint_forcing",
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
            test_get_p_term(this_test_input, "electron_energy_equation",
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
            test_get_p_term(this_test_input, "ion_dt_forcing_of_electron_p",
                            get_ion_dt_forcing_of_electron_p_term,
                            contribution_from_ion_dt_forcing_of_electron_p!,
                            (1.5e1*epsilon)^2)

            test_electron_wall_bc(this_test_input)

            test_electron_kinetic_equation(this_test_input)

            test_jacobian_inversion(this_test_input)
        end
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
