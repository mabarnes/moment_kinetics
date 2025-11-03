module JacobianMatrixTests

# Tests for construction of Jacobian matrices used for preconditioning

include("setup.jl")
include("jacobian_matrix_expected_data.jl")

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

const regression_f_step = 100

# Small parameter used to create perturbations to test Jacobian against
const epsilon = 1.0e-6
const test_wavenumber = 2.0
const dt = 0.2969848480983499
const ion_dt = 7.071067811865475e-7
const ir = 1
const zero = 1.0e-14

# Test input uses `z_bc = "constant"`, which is not a very physically useful option, but
# is useful for testing because:
# * `z_bc = "wall"` would introduce discontinuities in the distribution function which
#   might reduce accuracy and so make it harder to see whether errors are due to a mistake
#   in the matrix construction or just due to discretisation error
# * For `z_bc = "periodic"`, the Jacobian matrices (by design) do not account for the
#   periodicity. This should be fine when they are used as preconditioners, but does
#   introduce errors at the periodic boundaries which would complicate testing.
const test_input = OptionsDict("output" => OptionsDict("run_name" => "jacobian_matrix"),
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
                           rhs_func!::Function, rtol::mk_float,
                           expected::Union{Nothing,NamedTuple})
    test_input = deepcopy(test_input)
    test_input["output"]["run_name"] *= "_" * label[1:min(11, length(label))]
    println("        - $label")
    if test_input["timestepping"]["kinetic_electron_preconditioner"] == "lu_separate_dp_dz_dq_dz"
        # This preconditioner option has larger errors compared to the regression test
        # values (that were generated from "lu") than the others. Not sure why - maybe
        # there is a bug somewhere, or maybe the extra entries in the matrix just increase
        # rounding errors?
        regression_tol = 1.0e-10
    else
        regression_tol = 1.0e-13
    end

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
                normed_residual = perturbed_residual ./ norm_factor
                normed_with_Jacobian = reshape(perturbed_with_Jacobian, vpa.n, vperp.n, z.n) ./ norm_factor
                @test elementwise_isapprox(normed_residual, normed_with_Jacobian;
                                           rtol=0.0, atol=rtol)

                if expected === nothing
                    @test false
                    println("No stored regression test data. Tested data would be:\n"
                            * "(delta_f_residual=$(normed_residual[1:regression_f_step:end]),\n"
                            * " delta_f_with_Jacobian=$(normed_with_Jacobian[1:regression_f_step:end]),")
                else
                    @test elementwise_isapprox(normed_residual[1:regression_f_step:end],
                                               expected.delta_f_residual; rtol=0.0,
                                               atol=regression_tol)
                    @test elementwise_isapprox(normed_with_Jacobian[1:regression_f_step:end],
                                               expected.delta_f_with_Jacobian; rtol=0.0,
                                               atol=regression_tol)
                end
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
                normed_residual = perturbed_residual ./ norm_factor
                normed_with_Jacobian = reshape(perturbed_with_Jacobian, vpa.n, vperp.n, z.n) ./ norm_factor
                @test elementwise_isapprox(normed_residual, normed_with_Jacobian;
                                           rtol=0.0, atol=rtol)

                if expected === nothing
                    println(" delta_p_residual=$(normed_residual[1:regression_f_step:end]),\n"
                            * " delta_p_with_Jacobian=$(normed_with_Jacobian[1:regression_f_step:end]),")
                else
                    @test elementwise_isapprox(normed_residual[1:regression_f_step:end],
                                               expected.delta_p_residual; rtol=0.0,
                                               atol=regression_tol)
                    @test elementwise_isapprox(normed_with_Jacobian[1:regression_f_step:end],
                                               expected.delta_p_with_Jacobian; rtol=0.0,
                                               atol=regression_tol)
                end
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
                normed_residual = perturbed_residual ./ norm_factor
                normed_with_Jacobian = reshape(perturbed_with_Jacobian, vpa.n, vperp.n, z.n) ./ norm_factor
                @test elementwise_isapprox(normed_residual, normed_with_Jacobian;
                                           rtol=0.0, atol=rtol)

                if expected === nothing
                    println(" both_residual=$(normed_residual[1:regression_f_step:end]),\n"
                            * " both_with_Jacobian=$(normed_with_Jacobian[1:regression_f_step:end]),\n"
                            * ")")
                else
                    @test elementwise_isapprox(normed_residual[1:regression_f_step:end],
                                               expected.both_residual; rtol=0.0,
                                               atol=regression_tol)
                    @test elementwise_isapprox(normed_with_Jacobian[1:regression_f_step:end],
                                               expected.both_with_Jacobian; rtol=0.0,
                                               atol=regression_tol)
                end
            end
        end

        cleanup_mk_state!(ascii_io, io_moments, io_dfns)
    end

    return nothing
end

function test_get_p_term(test_input::AbstractDict, label::String, get_term::Function,
                         rhs_func!::Function, rtol::mk_float,
                         expected::Union{Nothing,NamedTuple})
    test_input = deepcopy(test_input)
    test_input["output"]["run_name"] *= "_" * label[1:min(11, length(label))]
    println("        - $label")
    if test_input["timestepping"]["kinetic_electron_preconditioner"] == "lu_separate_dp_dz_dq_dz"
        # This preconditioner option has larger errors compared to the regression test
        # values (that were generated from "lu") than the others. Not sure why - maybe
        # there is a bug somewhere, or maybe the extra entries in the matrix just increase
        # rounding errors?
        regression_tol = 3.0e-10
    else
        regression_tol = 1.0e-13
    end

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
                normed_residual = perturbed_residual ./ norm_factor
                normed_with_Jacobian = perturbed_with_Jacobian ./ norm_factor
                @test elementwise_isapprox(normed_residual, normed_with_Jacobian;
                                           rtol=0.0, atol=rtol)

                if expected === nothing
                    @test false
                    println("No stored regression test data. Tested data would be:\n"
                            * "(delta_f_residual=$(normed_residual),\n"
                            * " delta_f_with_Jacobian=$(normed_with_Jacobian),")
                else
                    @test elementwise_isapprox(normed_residual,
                                               expected.delta_f_residual; rtol=0.0,
                                               atol=regression_tol)
                    @test elementwise_isapprox(normed_with_Jacobian,
                                               expected.delta_f_with_Jacobian; rtol=0.0,
                                               atol=regression_tol)
                end
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
                normed_residual = perturbed_residual ./ norm_factor
                normed_with_Jacobian = perturbed_with_Jacobian ./ norm_factor
                @test elementwise_isapprox(normed_residual, normed_with_Jacobian;
                                           rtol=0.0, atol=rtol)

                if expected === nothing
                    println(" delta_p_residual=$(normed_residual),\n"
                            * " delta_p_with_Jacobian=$(normed_with_Jacobian),")
                else
                    @test elementwise_isapprox(normed_residual,
                                               expected.delta_p_residual; rtol=0.0,
                                               atol=regression_tol)
                    @test elementwise_isapprox(normed_with_Jacobian,
                                               expected.delta_p_with_Jacobian; rtol=0.0,
                                               atol=regression_tol)
                end
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
                normed_residual = perturbed_residual ./ norm_factor
                normed_with_Jacobian = perturbed_with_Jacobian ./ norm_factor
                @test elementwise_isapprox(normed_residual, normed_with_Jacobian;
                                           rtol=0.0, atol=rtol)

                if expected === nothing
                    println(" both_residual=$(normed_residual),\n"
                            * " both_with_Jacobian=$(normed_with_Jacobian),\n"
                            * ")")
                else
                    @test elementwise_isapprox(normed_residual,
                                               expected.both_residual; rtol=0.0,
                                               atol=regression_tol)
                    @test elementwise_isapprox(normed_with_Jacobian,
                                               expected.both_with_Jacobian; rtol=0.0,
                                               atol=regression_tol)
                end
            end
        end

        cleanup_mk_state!(ascii_io, io_moments, io_dfns)
    end

    return nothing
end

function test_electron_kinetic_equation(test_input; rtol=(5.0e2*epsilon)^2,
                                        expected_constant::Union{Nothing,NamedTuple},
                                        expected_wall::Union{Nothing,NamedTuple})

    # Looser rtol for "wall" bc because integral corrections not accounted for in wall bc
    # Jacobian (yet?).
    @testset "electron_kinetic_equation bc=$bc" for (bc, adi_tol, expected) ∈ (("constant", 1.0e-15, expected_constant),
                                                                               ("wall", 1.0e-13, expected_wall))
        println("        - electron_kinetic_equation $bc")
        this_test_input = deepcopy(test_input)
        label = "electron_kinetic_equation_$bc"
        test_input["output"]["run_name"] *= "_" * label[1:min(11, length(label))]
        this_test_input["z"]["bc"] = bc
        if test_input["timestepping"]["kinetic_electron_preconditioner"] == "lu_separate_dp_dz_dq_dz"
            # This preconditioner option has larger errors compared to the regression test
            # values (that were generated from "lu") than the others. Not sure why - maybe
            # there is a bug somewhere, or maybe the extra entries in the matrix just increase
            # rounding errors?
            regression_tol = 5.0e-10
        else
            regression_tol = 1.0e-13
        end

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
                normed_residual_f = perturbed_residual_f ./ norm_factor_f
                normed_with_Jacobian_f = reshape(perturbed_with_Jacobian_f, vpa.n, vperp.n, z.n) ./ norm_factor_f
                @test elementwise_isapprox(normed_residual_f, normed_with_Jacobian_f;
                                           rtol=0.0, atol=rtol)
                norm_factor_p = generate_norm_factor(perturbed_residual_p)
                normed_residual_p = perturbed_residual_p ./ norm_factor_p
                normed_with_Jacobian_p = perturbed_with_Jacobian_p ./ norm_factor_p
                @test elementwise_isapprox(normed_residual_p, normed_with_Jacobian_p;
                                           rtol=0.0, atol=rtol)

                if expected === nothing
                    @test false
                    println("No stored regression test data. Tested data would be:\n"
                            * "(delta_f_residual_f=$(normed_residual_f[1:regression_f_step:end]),\n"
                            * " delta_f_with_Jacobian_f=$(normed_with_Jacobian_f[1:regression_f_step:end]),\n"
                            * " delta_f_residual_p=$(normed_residual_p),\n"
                            * " delta_f_with_Jacobian_p=$(normed_with_Jacobian_p),")
                else
                    @test elementwise_isapprox(normed_residual_f[1:regression_f_step:end],
                                               expected.delta_f_residual_f; rtol=0.0,
                                               atol=regression_tol)
                    @test elementwise_isapprox(normed_with_Jacobian_f[1:regression_f_step:end],
                                               expected.delta_f_with_Jacobian_f; rtol=0.0,
                                               atol=regression_tol)
                    @test elementwise_isapprox(normed_residual_p,
                                               expected.delta_f_residual_p; rtol=0.0,
                                               atol=regression_tol)
                    @test elementwise_isapprox(normed_with_Jacobian_p,
                                               expected.delta_f_with_Jacobian_p; rtol=0.0,
                                               atol=regression_tol)
                end
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
                normed_residual_f = perturbed_residual_f ./ norm_factor_f
                normed_with_Jacobian_f = reshape(perturbed_with_Jacobian_f, vpa.n, vperp.n, z.n) ./ norm_factor_f
                @test elementwise_isapprox(normed_residual_f, normed_with_Jacobian_f;
                                           rtol=0.0, atol=rtol)
                norm_factor_p = generate_norm_factor(perturbed_residual_p)
                normed_residual_p = perturbed_residual_p ./ norm_factor_p
                normed_with_Jacobian_p = perturbed_with_Jacobian_p ./ norm_factor_p
                @test elementwise_isapprox(normed_residual_p, normed_with_Jacobian_p;
                                           rtol=0.0, atol=rtol)

                if expected === nothing
                    println(" delta_p_residual_f=$(normed_residual_f[1:regression_f_step:end]),\n"
                            * " delta_p_with_Jacobian_f=$(normed_with_Jacobian_f[1:regression_f_step:end]),\n"
                            * " delta_p_residual_p=$(normed_residual_p),\n"
                            * " delta_p_with_Jacobian_p=$(normed_with_Jacobian_p),")
                else
                    @test elementwise_isapprox(normed_residual_f[1:regression_f_step:end],
                                               expected.delta_p_residual_f; rtol=0.0,
                                               atol=regression_tol)
                    @test elementwise_isapprox(normed_with_Jacobian_f[1:regression_f_step:end],
                                               expected.delta_p_with_Jacobian_f; rtol=0.0,
                                               atol=regression_tol)
                    @test elementwise_isapprox(normed_residual_p,
                                               expected.delta_p_residual_p; rtol=0.0,
                                               atol=regression_tol)
                    @test elementwise_isapprox(normed_with_Jacobian_p,
                                               expected.delta_p_with_Jacobian_p; rtol=0.0,
                                               atol=regression_tol)
                end
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
                normed_residual_f = perturbed_residual_f ./ norm_factor_f
                normed_with_Jacobian_f = reshape(perturbed_with_Jacobian_f, vpa.n, vperp.n, z.n) ./ norm_factor_f
                @test elementwise_isapprox(normed_residual_f, normed_with_Jacobian_f;
                                           rtol=0.0, atol=rtol)
                norm_factor_p = generate_norm_factor(perturbed_residual_p)
                normed_residual_p = perturbed_residual_p ./ norm_factor_p
                normed_with_Jacobian_p = perturbed_with_Jacobian_p ./ norm_factor_p
                @test elementwise_isapprox(normed_residual_p, normed_with_Jacobian_p;
                                           rtol=0.0, atol=rtol)

                if expected === nothing
                    println(" both_residual_f=$(normed_residual_f[1:regression_f_step:end]),\n"
                            * " both_with_Jacobian_f=$(normed_with_Jacobian_f[1:regression_f_step:end]),\n"
                            * " both_residual_p=$(normed_residual_p),\n"
                            * " both_with_Jacobian_p=$(normed_with_Jacobian_p),\n"
                            * ")")
                else
                    @test elementwise_isapprox(normed_residual_f[1:regression_f_step:end],
                                               expected.both_residual_f; rtol=0.0,
                                               atol=regression_tol)
                    @test elementwise_isapprox(normed_with_Jacobian_f[1:regression_f_step:end],
                                               expected.both_with_Jacobian_f; rtol=0.0,
                                               atol=regression_tol)
                    @test elementwise_isapprox(normed_residual_p,
                                               expected.both_residual_p; rtol=0.0,
                                               atol=regression_tol)
                    @test elementwise_isapprox(normed_with_Jacobian_p,
                                               expected.both_with_Jacobian_p; rtol=0.0,
                                               atol=regression_tol)
                end
            end
        end

        cleanup_mk_state!(ascii_io, io_moments, io_dfns)
    end

    return nothing
end

function test_electron_wall_bc(test_input; atol=(10.0*epsilon)^2,
                               expected::Union{Nothing,NamedTuple})
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
                reshaped_with_Jacobian = reshape(perturbed_with_Jacobian, vpa.n, vperp.n, z.n)
                @test elementwise_isapprox(perturbed_residual, reshaped_with_Jacobian;
                                           rtol=0.0, atol=atol)

                if expected === nothing
                    @test false
                    println("No stored regression test data. Tested data would be:\n"
                            * "(delta_f_residual=$(perturbed_residual[1:regression_f_step:end]),\n"
                            * " delta_f_with_Jacobian=$(reshaped_with_Jacobian[1:regression_f_step:end]),")
                else
                    @test elementwise_isapprox(perturbed_residual[1:regression_f_step:end],
                                               expected.delta_f_residual; rtol=0.0,
                                               atol=1.0e-13)
                    @test elementwise_isapprox(reshaped_with_Jacobian[1:regression_f_step:end],
                                               expected.delta_f_with_Jacobian; rtol=0.0,
                                               atol=1.0e-13)
                end
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
                reshaped_with_Jacobian = reshape(perturbed_with_Jacobian, vpa.n, vperp.n, z.n)
                @test elementwise_isapprox(perturbed_residual, reshaped_with_Jacobian;
                                           rtol=0.0, atol=atol)

                if expected === nothing
                    println(" delta_p_residual=$(perturbed_residual[1:regression_f_step:end]),\n"
                            * " delta_p_with_Jacobian=$(reshaped_with_Jacobian[1:regression_f_step:end]),")
                else
                    @test elementwise_isapprox(perturbed_residual[1:regression_f_step:end],
                                               expected.delta_p_residual; rtol=0.0,
                                               atol=1.0e-13)
                    @test elementwise_isapprox(reshaped_with_Jacobian[1:regression_f_step:end],
                                               expected.delta_p_with_Jacobian; rtol=0.0,
                                               atol=1.0e-13)
                end
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
                reshaped_with_Jacobian = reshape(perturbed_with_Jacobian, vpa.n, vperp.n, z.n)
                @test elementwise_isapprox(perturbed_residual, reshaped_with_Jacobian;
                                           rtol=0.0, atol=atol)

                if expected === nothing
                    println(" both_residual=$(perturbed_residual[1:regression_f_step:end]),\n"
                            * " both_with_Jacobian=$(reshaped_with_Jacobian[1:regression_f_step:end]),\n"
                            * ")")
                else
                    @test elementwise_isapprox(perturbed_residual[1:regression_f_step:end],
                                               expected.both_residual; rtol=0.0,
                                               atol=1.0e-13)
                    @test elementwise_isapprox(reshaped_with_Jacobian[1:regression_f_step:end],
                                               expected.both_with_Jacobian; rtol=0.0,
                                               atol=1.0e-13)
                end
            end
        end

        cleanup_mk_state!(ascii_io, io_moments, io_dfns)
    end

    return nothing
end

function test_jacobian_inversion(test_input; rtol=2.0e-12)

    @testset "jacobian_inversion $bc" for (bc, expected) ∈ (("wall", expected_jacobian_inversion_wall),
                                                            ("periodic", expected_jacobian_inversion_periodic))
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

        recalculate_preconditioner!()

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

        if expected === nothing
            @test false
            println("No stored regression test data. Tested data would be:\n"
                    * "(expected_precon_f=", @view residual_f[1:regression_f_step:end], ",\n"
                    * " expected_precon_p=", residual_p, ",\n"
                    * ")")
        else
            @test elementwise_isapprox(expected.precon_f,
                                       @view residual_f[1:regression_f_step:end];
                                       rtol=rtol, atol=1.0e-20)

            if (nl_solver_params.electron_advance.preconditioner_type === Val(:electron_lu_separate_dp_dz_dq_dz)
                    && bc == "periodic")
                # This combination has a strangely high relative error. Not sure why,
                # maybe just bad luck (?), but :lu_separate_dp_dz_dq_dz is not commonly
                # used anyway, so allow this to pass (at least for now).
                p_rtol = rtol * 1.0e3
            else
                p_rtol = rtol
            end
            @test elementwise_isapprox(expected.precon_p, residual_p; rtol=p_rtol,
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
                              (2.5e2*epsilon)^2, expected_z)

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
                              (3.0e2*epsilon)^2, expected_vpa)

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
                              contribution_from_electron_pdf_term_wrapper!,
                              (4.0e2*epsilon)^2, expected_pdf_term)

            function contribution_from_electron_dissipation_term!(; kwargs...)
                add_dissipation_term!(kwargs[:residual], kwargs[:this_f],
                                      kwargs[:scratch_dummy], kwargs[:z_spectral], kwargs[:z],
                                      kwargs[:vpa], kwargs[:vpa_spectral],
                                      kwargs[:num_diss_params], kwargs[:dt])
                return nothing
            end
            test_get_pdf_term(this_test_input, "electron_dissipation_term",
                              get_electron_dissipation_term,
                              contribution_from_electron_dissipation_term!,
                              (1.0e1*epsilon)^2, expected_dissipation)

            function contribution_from_krook_collisions!(; kwargs...)
                electron_krook_collisions!(kwargs[:residual], kwargs[:this_f], kwargs[:dens],
                                           kwargs[:upar], kwargs[:ion_upar], kwargs[:vth],
                                           kwargs[:collisions], kwargs[:vperp], kwargs[:vpa],
                                           kwargs[:dt])
                return nothing
            end
            test_get_pdf_term(this_test_input, "electron_krook_collisions",
                              get_electron_krook_collisions_term,
                              contribution_from_krook_collisions!, (2.0e1*epsilon)^2,
                              expected_krook)

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
                              contribution_from_external_electron_sources!,
                              (3.0e1*epsilon)^2, expected_sources)

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
                              contribution_from_implicit_constraint_forcing!,
                              (2.5e0*epsilon), expected_constraint)

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
                            contribution_from_electron_energy_equation!,
                            (6.0e2*epsilon)^2, expected_energy)

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
                            (1.5e1*epsilon)^2, expected_ion_dt)

            test_electron_wall_bc(this_test_input; expected=expected_wall)

            test_electron_kinetic_equation(this_test_input; expected_constant=expected_ke_constant,
                                           expected_wall=expected_ke_wall)

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
