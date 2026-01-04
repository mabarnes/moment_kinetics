module ion_jacobian_terms

export get_ion_preconditioners, fill_ion_kinetic_equation_Jacobian!

using ..energy_equation: get_dvth_dt_expanded_term_evolve_nup
using ..force_balance: get_dupar_dt_expanded_term_evolve_nup
using ..jacobian_matrices
using ..krook_collisions: get_ion_krook_collisions_term_evolve_nup
using ..looping
using ..moment_kinetics_structs
using ..numerical_dissipation: get_ion_dissipation_term_evolve_nup
using ..timer_utils
using ..type_definitions
using ..vpa_advection: get_ion_vpa_advection_term_evolve_nup
using ..vperp_advection: get_ion_vperp_advection_term_evolve_nup
using ..z_advection: get_ion_z_advection_term_evolve_nup

"""
"""
function get_ion_sub_terms_evolve_nup(
             pdf_array::AbstractArray{mk_float,3},
             dpdf_dz_array::AbstractArray{mk_float,3},
             dpdf_dvperp_array::AbstractArray{mk_float,3},
             dpdf_dvpa_array::AbstractArray{mk_float,3},
             d2pdf_dz2_array::AbstractArray{mk_float,3},
             d2pdf_dvperp2_array::AbstractArray{mk_float,3},
             d2pdf_dvpa2_array::AbstractArray{mk_float,3},
             dens_array::AbstractVector{mk_float},
             ddens_dt_array::AbstractVector{mk_float},
             ddens_dr_array::AbstractVector{mk_float},
             ddens_dz_array::AbstractVector{mk_float},
             upar_array::AbstractVector{mk_float},
             dupar_dt_array::AbstractVector{mk_float},
             dupar_dr_array::AbstractVector{mk_float},
             dupar_dz_array::AbstractVector{mk_float},
             vth_array::AbstractVector{mk_float},
             dvth_dt_array::AbstractVector{mk_float},
             dvth_dr_array::AbstractVector{mk_float},
             dvth_dz_array::AbstractVector{mk_float},
             ppar_array::AbstractVector{mk_float},
             dppar_dz_array::AbstractVector{mk_float},
             third_moment_array::AbstractVector{mk_float},
             dthird_moment_dz_array::AbstractVector{mk_float},
             dqpar_dz_array::AbstractVector{mk_float},
             Ez_array::AbstractVector{mk_float},
             collisions, external_sources, geometry, num_diss_params,
             z::coordinate, vperp::coordinate, vpa::coordinate,
             r_speed_array::AbstractArray{mk_float,3},
             alpha_speed_array::AbstractArray{mk_float,3},
             z_speed_array::AbstractArray{mk_float,3},
             vperp_speed_array::AbstractArray{mk_float,3},
             vpa_speed_array::AbstractArray{mk_float,3},
             ir::mk_int,
         )

    mi = 1.0 # Don't allow for multiple ion species yet, so dimensionless mass is always 1.0.

    f = EquationTerm(:ion_pdf, pdf_array; vpa=vpa, vperp=vperp, z=z)
    df_dz = EquationTerm(:ion_pdf, dpdf_dz_array; derivatives=[:z],
                         upwind_speeds=[z_speed_array], vpa=vpa, vperp=vperp, z=z)
    df_dvperp = EquationTerm(:ion_pdf, dpdf_dvperp_array; derivatives=[:vperp],
                             upwind_speeds=[vperp_speed_array], vpa=vpa, vperp=vperp, z=z)
    df_dvpa = EquationTerm(:ion_pdf, dpdf_dvpa_array; derivatives=[:vpa],
                           upwind_speeds=[vpa_speed_array], vpa=vpa, vperp=vperp, z=z)

    n = ConstantTerm(dens_array; z=z)
    dn_dt = ConstantTerm(ddens_dt_array; z=z)
    dn_dr = ConstantTerm(ddens_dr_array; z=z)
    dn_dz = ConstantTerm(ddens_dz_array; z=z)
    upar = ConstantTerm(upar_array; z=z)
    dupar_dr = ConstantTerm(dupar_dr_array; z=z)
    dupar_dz = ConstantTerm(dupar_dz_array; z=z)
    vth = ConstantTerm(vth_array; z=z)
    dvth_dr = ConstantTerm(dvth_dr_array; z=z)
    dvth_dz = ConstantTerm(dvth_dz_array; z=z)
    Ez = ConstantTerm(Ez_array; z=z)

    wpa = ConstantTerm(vpa.grid; vpa=vpa)
    wperp = ConstantTerm(vperp.grid; vperp=vperp)

    if vperp.n == 1
        # ppar does not depend directly on f
        ppar = ConstantTerm(ppar_array; z=z)
        dppar_dz = ConstantTerm(dppar_dz_array; z=z)
        wpa2_moment = NullTerm()
        dwpa2_moment_dz = NullTerm()
        wpa2_moment_constraint_rhs = NullTerm()
    else
        wpa2_moment_integrand_prefactor = wpa^2
        wpa2_moment = EquationTerm(:ion_pdf, wpa2_moment_array;
                                   integrand_coordinates=[vpa,vperp,z],
                                   integrand_prefactor=wpa2_moment_integrand_prefactor,
                                   z=z)
        dwpa2_moment_dz = EquationTerm(:wpa2_moment, dwpa2_moment_dz_array;
                                       derivatives=[:z], z=z)
        wpa2_moment_constraint_rhs = wpa2_moment

        ppar = mi * n * vth^2 * wpa2_moment
        dppar_dz_expanded = mi * (vth^2 * wpa2_moment * dn_dz
                                  + 2 * n * vth * wpa2_moment * dvth_dz
                                  + n * vth^2 * dwpa2_moment_dz)
        dppar_dz = CompoundTerm(dppar_dz_expanded, dppar_dz_array; z=z)
    end
    third_moment_integrand_prefactor = wpa*(wpa^2 + wperp^2)
    third_moment = EquationTerm(:ion_pdf, third_moment_array;
                                integrand_coordinates=[vpa,vperp,z],
                                integrand_prefactor=third_moment_integrand_prefactor,
                                z=z)
    dthird_moment_dz = EquationTerm(:third_moment, dthird_moment_dz_array;
                                    derivatives=[:z], z=z)
    third_moment_constraint_rhs = third_moment

    dqpar_dz_expanded = mi * (vth^3 * third_moment * dn_dz
                              + 3 * n * vth^2 * third_moment * dvth_dz
                              + n * vth^3 * dthird_moment_dz)
    dqpar_dz = CompoundTerm(dqpar_dz_expanded, dqpar_dz_array; z=z)

    bzed = ConstantTerm(@view geometry.bzed[:,ir]; z=z)

    r_speed = ConstantTerm(r_speed_array; vpa=vpa, vperp=vperp, z=z)
    alpha_speed = ConstantTerm(alpha_speed_array; z=z, vpa=vpa, vperp=vperp)
    z_speed = ConstantTerm(z_speed_array; z=z, vpa=vpa, vperp=vperp)

    # As dupar_dt is a CompoundTerm, only need to keep contributions in dupar_dt_expanded
    # that depend on f - any constant terms would not contribute (their contribution is
    # captured completely by dupar_dt_array).
    dupar_dt_expanded = get_dupar_dt_expanded_term_evolve_nup(bzed, mi, n, dppar_dz)
    dupar_dt = CompoundTerm(dupar_dt_expanded, dupar_dt_array; z=z)

    # As dvth_dt is a CompoundTerm, only need to keep contributions in dvth_dt_expanded
    # that depend on f - any constant terms would not contribute (their contribution is
    # captured completely by dvth_dt_array).
    dvth_dt_expanded = get_dvth_dt_expanded_term_evolve_nup(bzed, mi, n, vth, ppar,
                                                            dupar_dz, dqpar_dz)
    dvth_dt = CompoundTerm(dvth_dt_expanded, dvth_dt_array; z=z)

    nvperp = vperp.n

    z_dissipation_coefficient = num_diss_params.z_dissipation_coefficient
    vperp_dissipation_coefficient = num_diss_params.vperp_dissipation_coefficient
    vpa_dissipation_coefficient = num_diss_params.vpa_dissipation_coefficient

    return IonSubTerms(; f, df_dz, df_dvperp, df_dvpa, n, dn_dt, dn_dr, dn_dz, upar,
                       dupar_dt, dupar_dr, dupar_dz, vth, dvth_dt, dvth_dr, dvth_dz,
                       wperp, wpa, bzed, r_speed, alpha_speed, z_speed, nvperp,
                       z_dissipation_coefficient, vperp_dissipation_coefficient,
                       vpa_dissipation_coefficient, collisions, external_sources, z, ir)
end

"""
"""
@timeit global_timer fill_ion_kinetic_equation_Jacobian_evolve_nup!(
                         jacobian::jacobian_info, f::AbstractArray{mk_float,3},
                         scratch_dummy, is, ir,
                        ) = begin

    z_speed = @view z_advect[is].speed[:,:,:,ir]
    alpha_speed = @view alpha_advect[is].speed[:,:,:,ir]

    sub_terms = get_ion_sub_terms_evolve_nup()
    pdf_terms, wpa2_terms, third_moment_terms = get_all_ion_terms_evolve_nup(sub_terms)

    jacobian_initialize_identity!(jacobian)

    # For now (when we have implemented only 2D sims), convert z_speed to the total speed
    # perpendicular to the wall by adding the z-projection of the binormal advection
    # speed, which `alpha_speed` currently is.
    @begin_anyzv_z_vperp_vpa_region()
    @loop_z_vperp_vpa iz ivperp ivpa begin
        z_speed[iz,ivpa,ivperp,ir] += alpha_speed[iz,ivpa,ivperp,ir]
    end

    add_term_to_Jacobian!(jacobian, :ion_pdf, dt, pdf_terms, z_speed)
    if collisions.fkpl.use_fokker_planck
        error("Preconditioner for Fokker-Planck collisions not integrated into "
              * "fill_ion_kinetic_equation_Jacobian!() yet.")
    end

    if vperp.n > 1
        add_term_to_Jacobian!(jacobian, :wpa2_moment, 1.0, wpa2_moment_terms)
    end
    add_term_to_Jacobian!(jacobian, :third_moment, 1.0, third_moment_terms)

    return nothing
end

function get_all_ion_terms_evolve_nup(sub_terms::IonSubTerms)
    pdf_terms += get_ion_z_advection_term_evolve_nup(sub_terms)
    pdf_terms += get_ion_vpa_advection_term_evolve_nup(sub_terms)
    pdf_terms += get_ion_vperp_advection_term_evolve_nup(sub_terms)
    pdf_terms += get_contribution_from_ion_pdf_term_evolve_nup(sub_terms)
    pdf_terms += get_ion_dissipation_term_evolve_nup(sub_terms)
    pdf_terms += get_ion_krook_collisions_term_evolve_nup(sub_terms)
    # Fokker-Planck collisions would be handled by special function, not EquationTerm.
    pdf_terms += get_total_external_ion_source_term_evolve_nup(sub_terms)

    if sub_terms.wpa2_moment_constraint_rhs === nothing
        wpa2_moment_terms = nothing
    else
        wpa2_moment_terms = -sub_terms.wpa2_moment_constraint_rhs
    end
    third_moment_terms = -sub_terms.third_moment_constraint_rhs

    return pdf_terms
end

function get_ion_preconditioners(preconditioner_type, nl_solver_input, coords,
                                 outer_coords, spectral;
                                 boundary_skip_funcs::BSF=nothing) where {BSF}
    coord_sizes = Tuple(isa(c, coordinate) ? c.n : c for c ∈ coords)
    total_size_coords = prod(coord_sizes)
    outer_coord_sizes = Tuple(isa(c, coordinate) ? c.n : c for c ∈ outer_coords)

    if preconditioner_type === Val(:ion_parallel_lu)
        # One constraint for third_moment that feeds pdf into qpar.
        pdf_plus_constraints_size = total_size_coords + coords.z.n
        if coords.vperp.n == 1
            preconditioners = [(lu(sparse(1.0*I, 1, 1)),
                                create_jacobian_info(coords, spectral;
                                                     comm=comm_anyzv_subblock[],
                                                     synchronize=_anyzv_subblock_synchronize,
                                                     boundary_skip_funcs=boundary_skip_funcs.full,
                                                     ion_pdf=((:anyzv,:z,:vperp,:vpa), (:vpa, :vperp, :z), false),
                                                     third_moment=((:anyzv,:z), (:z,), true)),
                                allocate_shared_float(:newton_size=>pdf_plus_p_plus_constraints_size;
                                                      comm=comm_anyzv_subblock[]),
                                allocate_shared_float(:newton_size=>pdf_plus_p_plus_constraints_size;
                                                      comm=comm_anyzv_subblock[]),
                               )
                               for _ ∈ CartesianIndices(reverse(outer_coord_sizes))]
        else
            # 2V run, need another constraint for wpa^2 moment that feeds pdf into ppar.
            pdf_plus_constraints_size += coords.z.n

            preconditioners = [(lu(sparse(1.0*I, 1, 1)),
                                create_jacobian_info(coords, spectral;
                                                     comm=comm_anyzv_subblock[],
                                                     synchronize=_anyzv_subblock_synchronize,
                                                     boundary_skip_funcs=boundary_skip_funcs.full,
                                                     ion_pdf=((:anyzv,:z,:vperp,:vpa), (:vpa, :vperp, :z), false),
                                                     wpa2_moment=((:anyzv,:z), (:z,), true),
                                                     third_moment=((:anyzv,:z), (:z,), true)),
                                allocate_shared_float(:newton_size=>pdf_plus_p_plus_constraints_size;
                                                      comm=comm_anyzv_subblock[]),
                                allocate_shared_float(:newton_size=>pdf_plus_p_plus_constraints_size;
                                                      comm=comm_anyzv_subblock[]),
                               )
                               for _ ∈ CartesianIndices(reverse(outer_coord_sizes))]
        end
        # Initialise input buffers to zero so that constraint equations have zero on the
        # RHS.
        @begin_serial_region()
        @serial_region begin
            for p ∈ preconditioners
                p[3] .= 0.0
            end
        end
    elseif preconditioner_type === Val(:none)
        preconditioners = nothing
    else
        error("Unrecognised preconditioner_type=$preconditioner_type")
    end

    return preconditioners
end

end # ion_jacobian_terms
