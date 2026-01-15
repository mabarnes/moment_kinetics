"""
"""
module vpa_advection

export vpa_advection!
export update_speed_vpa!

using ..advection: advance_f_local!
using ..boundary_conditions: enforce_v_boundary_condition_local!
using ..communication
using ..debugging
using ..looping
using ..moment_constraints: hard_force_moment_constraints!,
                            moment_constraints_on_residual!
using ..timer_utils
using ..moment_kinetics_structs: scratch_pdf, weak_discretization_info
using ..nonlinear_solvers: newton_solve!
using ..velocity_moments: update_derived_moments!, calculate_ion_moment_derivatives!

using ..array_allocation: allocate_float
using ..boundary_conditions: vpagrid_to_vpa
using ..calculus: second_derivative!
using LinearAlgebra
using SparseArrays

"""
"""
@timeit global_timer vpa_advection!(
                         f_out, fvec_in, fields, moments, vpa_advect, r_advect,
                         alpha_advect, z_advect, vpa, vperp, z, r, dt, t, vpa_spectral,
                         composition, collisions, ion_source_settings, geometry) = begin
    return vpa_advection!(f_out, fvec_in, fields, moments, vpa_advect, r_advect,
                          alpha_advect, z_advect, vpa, vperp, z, r, dt, t, vpa_spectral,
                          composition, collisions, ion_source_settings, geometry,
                          Val(moments.evolve_density), Val(moments.evolve_upar),
                          Val(moments.evolve_p))
end
function vpa_advection!(f_out, fvec_in, fields, moments, vpa_advect, r_advect,
                        alpha_advect, z_advect, vpa, vperp, z, r, dt, t, vpa_spectral,
                        composition, collisions, ion_source_settings, geometry,
                        evolve_density::Val, evolve_upar::Val, evolve_p::Val)

    @begin_s_r_z_vperp_region()

    # only have a parallel acceleration term for neutrals if using the peculiar velocity
    # wpar = vpar - upar as a variable; i.e., d(wpar)/dt /=0 for neutrals even though d(vpar)/dt = 0.

    speed_args = get_speed_vpa_inner_args(vpa_advect, fvec_in, moments, fields, r_advect,
                                          alpha_advect, z_advect, geometry, vperp, vpa,
                                          evolve_density, evolve_upar, evolve_p)
    @loop_s_r is ir begin
        speed_args_sr = get_speed_vpa_inner_views_sr(is, ir, speed_args...)
        @loop_z iz begin
            speed_args_z = get_speed_vpa_inner_views_z(iz, speed_args_sr...)
            @loop_vperp ivperp begin
                speed_args_vperp = get_speed_vpa_inner_views_vperp(ivperp, speed_args_z...)
                # calculate the advection speed corresponding to current f
                update_speed_vpa_inner!(speed_args_vperp...)
                @views advance_f_local!(f_out[:,ivperp,iz,ir,is],
                                        fvec_in.pdf[:,ivperp,iz,ir,is],
                                        first(speed_args_vperp), vpa, dt, vpa_spectral)
            end
        end
    end
end

"""
"""
@timeit global_timer implicit_vpa_advection!(
                         f_out, fvec_in, fields, moments, r_advect, alpha_advect,
                         z_advect, vpa_advect, vpa, vperp, z, r, dt, t, r_spectral,
                         z_spectral, vpa_spectral, composition, collisions,
                         ion_source_settings, geometry, nl_solver_params, vpa_diffusion,
                         num_diss_params, gyroavs, scratch_dummy) = begin
    if vperp.n > 1 && (moments.evolve_density || moments.evolve_upar || moments.evolve_p)
        error("Moment constraints in implicit_vpa_advection!() do not support 2V runs yet")
    end

    # calculate the advection speed corresponding to current f
    update_speed_vpa!(vpa_advect, fields, fvec_in, moments, r_advect, alpha_advect,
                      z_advect, vpa, vperp, z, r, composition, collisions,
                      ion_source_settings, t, geometry)

    # Ensure moments are consistent with f_new
    new_scratch = scratch_pdf(f_out, fvec_in.density, fvec_in.upar, fvec_in.ppar,
                              fvec_in.pperp, fvec_in.temp_z_s, fvec_in.pdf_neutral,
                              fvec_in.density_neutral, fvec_in.uz_neutral,
                              fvec_in.pz_neutral)
    update_derived_moments!(new_scratch, moments, vpa, vperp, z, r, composition,
                            r_spectral, geometry, gyroavs, scratch_dummy, z_advect, collisions, false)
    calculate_ion_moment_derivatives!(moments, new_scratch, scratch_dummy, z,
                                      z_spectral,
                                      num_diss_params.ion.moment_dissipation_coefficient)

    @begin_s_r_z_vperp_region()

    coords = (vpa=vpa,)
    vpa_bc = vpa.bc
    minval = num_diss_params.ion.force_minimum_pdf_value
    vpa_dissipation_coefficient = num_diss_params.ion.vpa_dissipation_coefficient
    zero = 1.0e-14
    @loop_s is begin
        @loop_r_z_vperp ir iz ivperp begin
            f_old_no_bc = @view fvec_in.pdf[:,ivperp,iz,ir,is]
            this_f_out = @view f_out[:,ivperp,iz,ir,is]
            speed = @view vpa_advect[:,ivperp,iz,ir,is]

            if z.irank == 0 && iz == 1
                @. vpa.scratch = vpagrid_to_vpa(vpa.grid, moments.ion.vth[iz,ir,is],
                                                fvec_in.upar[iz,ir,is], moments.evolve_p,
                                                moments.evolve_upar)
                icut_lower_z = vpa.n
                for ivpa ∈ vpa.n:-1:1
                    # for left boundary in zed (z = -Lz/2), want
                    # f(z=-Lz/2, v_parallel > 0) = 0
                    if vpa.scratch[ivpa] ≤ zero
                        icut_lower_z = ivpa + 1
                        break
                    end
                end
            end
            if z.irank == z.nrank - 1 && iz == z.n
                @. vpa.scratch = vpagrid_to_vpa(vpa.grid, moments.ion.vth[iz,ir,is],
                                                fvec_in.upar[iz,ir,is], moments.evolve_p,
                                                moments.evolve_upar)
                icut_upper_z = 0
                for ivpa ∈ 1:vpa.n
                    # for right boundary in zed (z = Lz/2), want
                    # f(z=Lz/2, v_parallel < 0) = 0
                    if vpa.scratch[ivpa] ≥ -zero
                        icut_upper_z = ivpa - 1
                        break
                    end
                end
            end

            function apply_bc!(x)
                # Boundary condition
                enforce_v_boundary_condition_local!(x, vpa_bc, speed, vpa_diffusion,
                                                    vpa, vpa_spectral)

                if z.bc == "wall"
                    # Wall boundary conditions. Note that as density, upar, ppar do not
                    # change in this implicit step, f_new, f_old, and residual should all
                    # be zero at exactly the same set of grid points, so it is reasonable
                    # to zero-out `residual` to impose the boundary condition. We impose
                    # this after subtracting f_old in case rounding errors, etc. mean that
                    # at some point f_old had a different boundary condition cut-off
                    # index.
                    if z.irank == 0 && iz == 1
                        x[icut_lower_z:end] .= 0.0
                    end
                    # absolute velocity at right boundary
                    if z.irank == z.nrank - 1 && iz == z.n
                        x[1:icut_upper_z] .= 0.0
                    end
                end
            end

            # Need to apply 'new' boundary conditions to `f_old`, so that by imposing them
            # on `residual`, they are automatically imposed on `f_new`.
            f_old = vpa.scratch7 .= f_old_no_bc
            apply_bc!(f_old)

            #if nl_solver_params.solves_since_precon_update[] ≥ nl_solver_params.preconditioner_update_interval
            #    nl_solver_params.solves_since_precon_update[] = 0

            #    advection_matrix = allocate_float(vpa, vpa)
            #    advection_matrix .= 0.0
            #    for i ∈ 1:vpa.nelement_local
            #        imin = vpa.imin[i] - (i != 1)
            #        imax = vpa.imax[i]
            #        if i == 1
            #            advection_matrix[imin,imin:imax] .+= vpa_spectral.lobatto.Dmat[1,:] ./ vpa.element_scale[i]
            #        else
            #            if speed[imin] < 0.0
            #                advection_matrix[imin,imin:imax] .+= vpa_spectral.lobatto.Dmat[1,:] ./ vpa.element_scale[i]
            #            elseif speed[imin] > 0.0
            #                # Do nothing
            #            else
            #                advection_matrix[imin,imin:imax] .+= 0.5 .* vpa_spectral.lobatto.Dmat[1,:] ./ vpa.element_scale[i]
            #            end
            #        end
            #        advection_matrix[imin+1:imax-1,imin:imax] .+= vpa_spectral.lobatto.Dmat[2:end-1,:] ./ vpa.element_scale[i]
            #        if i == vpa.nelement_local
            #            advection_matrix[imax,imin:imax] .+= vpa_spectral.lobatto.Dmat[end,:] ./ vpa.element_scale[i]
            #        else
            #            if speed[imax] < 0.0
            #                # Do nothing
            #            elseif speed[imax] > 0.0
            #                advection_matrix[imax,imin:imax] .+= vpa_spectral.lobatto.Dmat[end,:] ./ vpa.element_scale[i]
            #            else
            #                advection_matrix[imax,imin:imax] .+= 0.5 .* vpa_spectral.lobatto.Dmat[end,:] ./ vpa.element_scale[i]
            #            end
            #        end
            #    end
            #    # Multiply by advection speed
            #    for i ∈ 1:vpa.n
            #        advection_matrix[i,:] .*= dt * speed[i]
            #    end
            #    for i ∈ 1:vpa.n
            #        advection_matrix[i,i] += 1.0
            #    end

            #    if isa(vpa_spectral, weak_discretization_info)
            #        # This allocates a new matrix - to avoid this would need to pre-allocate a
            #        # suitable buffer somewhere and use `mul!()`.
            #        advection_matrix = vpa_spectral.mass_matrix * advection_matrix
            #        @. advection_matrix -= dt * vpa_dissipation_coefficient * vpa_spectral.K_matrix
            #    elseif vpa_dissipation_coefficient > 0.0
            #        error("Non-weak-form schemes cannot precondition diffusion")
            #    end

            #    # hacky (?) Dirichlet boundary conditions
            #    this_f_out[1] = 0.0
            #    this_f_out[end] = 0.0
            #    advection_matrix[1,:] .= 0.0
            #    advection_matrix[1,1] = 1.0
            #    advection_matrix[end,:] .= 0.0
            #    advection_matrix[end,end] = 1.0

            #    if z.bc == "wall"
            #        if z.irank == 0 && iz == 1
            #            # Set equal df/dt equal to f on points that should be set to zero for
            #            # boundary condition. The vector that the inverse of the advection matrix
            #            # acts on should have zeros there already.
            #            advection_matrix[icut_lower_z:end,icut_lower_z:end] .= 0.0
            #            for i ∈ icut_lower_z:vpa.n
            #                advection_matrix[i,i] = 1.0
            #            end
            #        end
            #        if z.irank == z.nrank - 1 && iz == z.n
            #            # Set equal df/dt equal to f on points that should be set to zero for
            #            # boundary condition. The vector that the inverse of the advection matrix
            #            # acts on should have zeros there already.
            #            # I comes from LinearAlgebra and represents identity matrix
            #            advection_matrix[1:icut_upper_z,1:icut_upper_z] .= 0.0
            #            for i ∈ 1:icut_upper_z
            #                advection_matrix[i,i] = 1.0
            #            end
            #        end
            #    end

            #    advection_matrix = sparse(advection_matrix)
            #    nl_solver_params.preconditioners[ivperp,iz,ir,is] = lu(advection_matrix)
            #end

            #function preconditioner(x)
            #    if isa(vpa_spectral, weak_discretization_info)
            #        # Multiply by mass matrix, storing result in vpa.scratch
            #        mul!(vpa.scratch, vpa_spectral.mass_matrix, x)
            #    end

            #    # Handle boundary conditions
            #    enforce_v_boundary_condition_local!(vpa.scratch, vpa_bc, speed, vpa_diffusion,
            #                                        vpa, vpa_spectral)

            #    if z.bc == "wall"
            #        # Wall boundary conditions. Note that as density, upar, ppar do not
            #        # change in this implicit step, f_new, f_old, and residual should all
            #        # be zero at exactly the same set of grid points, so it is reasonable
            #        # to zero-out `residual` to impose the boundary condition. We impose
            #        # this after subtracting f_old in case rounding errors, etc. mean that
            #        # at some point f_old had a different boundary condition cut-off
            #        # index.
            #        if z.irank == 0 && iz == 1
            #            vpa.scratch[icut_lower_z:end] .= 0.0
#           #             println("at icut_lower_z ", f_new[icut_lower_z], " ", f_old[icut_lower_z])
            #        end
            #        # absolute velocity at right boundary
            #        if z.irank == z.nrank - 1 && iz == z.n
            #            vpa.scratch[1:icut_upper_z] .= 0.0
            #        end
            #    end

            #    # Do LU application on vpa.scratch, storing result in x
            #    ldiv!(x, nl_solver_params.preconditioners[ivperp,iz,ir,is], vpa.scratch)
            #    return nothing
            #end
            left_preconditioner = identity
            right_preconditioner = identity
            #right_preconditioner = preconditioner

            # Define a function whose input is `f_new`, so that when it's output
            # `residual` is zero, f_new is the result of a backward-Euler timestep:
            #   (f_new - f_old) / dt = RHS(f_new)
            # ⇒ f_new - f_old - dt*RHS(f_new) = 0
            function residual_func!(residual, f_new; krylov=false)
                apply_bc!(f_new)
                residual .= f_old
                @views advance_f_local!(residual, f_new, vpa_advect[ivperp,iz,ir,is], vpa,
                                        dt, vpa_spectral)

                if vpa_diffusion
                    second_derivative!(vpa.scratch, f_new, vpa, vpa_spectral)
                    @. residual += dt * vpa_dissipation_coefficient * vpa.scratch
                end

                # Make sure updated f will not contain negative values
                #@. residual = max(residual, minval)

                # Now
                #   residual = f_old + dt*RHS(f_new)
                # so update to desired residual
                @. residual = f_new - residual

                apply_bc!(residual)
            end

            # Buffers
            # Note vpa,scratch is used by advance_f!, so we cannot use it here.
            residual = vpa.scratch2
            delta_x = vpa.scratch3
            rhs_delta = vpa.scratch4
            v = vpa.scratch5
            w = vpa.scratch6

            # Use forward-Euler step for initial guess
            # By passing this_f_out, which is equal to f_old at this point, the 'residual'
            # is
            #   f_new - f_old - dt*RHS(f_old) = -dt*RHS(f_old)
            # so to get a forward-Euler step we have to subtract this 'residual'
            residual_func!(residual, this_f_out)
            this_f_out .-= residual

            success = newton_solve!(this_f_out, residual_func!, residual, delta_x,
                                    rhs_delta, v, w, nl_solver_params, coords=coords,
                                    left_preconditioner=left_preconditioner,
                                    right_preconditioner=right_preconditioner)
            if !success
                return success
            end
        end
    end

    return true
end

"""
calculate the advection speed in the vpa-direction at each grid point
"""
function update_speed_vpa!(vpa_advect, fields, fvec, moments, r_advect, alpha_advect,
                           z_advect, vpa, vperp, z, r, composition, collisions,
                           ion_source_settings, t, geometry)
    return update_speed_vpa!(vpa_advect, fields, fvec, moments, r_advect, alpha_advect,
                             z_advect, vpa, vperp, z, r, composition, collisions,
                             ion_source_settings, t, geometry,
                             Val(moments.evolve_density), Val(moments.evolve_upar),
                             Val(moments.evolve_p))
end
function update_speed_vpa!(vpa_advect, fields, fvec, moments, r_advect, alpha_advect,
                           z_advect, vpa, vperp, z, r, composition, collisions,
                           ion_source_settings, t, geometry, evolve_density::Val,
                           evolve_upar::Val, evolve_p::Val)
    @debug_consistency_checks r.n == size(vpa_advect,4) || throw(BoundsError(vpa_advect))
    @debug_consistency_checks z.n == size(vpa_advect,3) || throw(BoundsError(vpa_advect))
    @debug_consistency_checks vperp.n == size(vpa_advect,2) || throw(BoundsError(vpa_advect))
    @debug_consistency_checks composition.n_ion_species == size(vpa_advect,5) || throw(BoundsError(vpa_advect))
    @debug_consistency_checks vpa.n == size(vpa_advect,1) || throw(BoundsError(vpa_advect))

    @begin_s_r_z_vperp_region()

    # dvpa/dt = Ze/m ⋅ E_parallel - (vperp^2/2B) bz dB/dz
    # magnetic mirror term only supported for standard DK implementation
    speed_args = get_speed_vpa_inner_args(vpa_advect, fvec, moments, fields, r_advect,
                                          alpha_advect, z_advect, geometry, vperp, vpa,
                                          evolve_density, evolve_upar, evolve_p)
    @loop_s_r is ir begin
        speed_args_sr = get_speed_vpa_inner_views_sr(is, ir, speed_args...)
        @loop_z iz begin
            speed_args_z = get_speed_vpa_inner_views_z(iz, speed_args_sr...)
            @loop_vperp ivperp begin
                update_speed_vpa_inner!(get_speed_vpa_inner_views_vperp(ivperp, speed_args_z...)...)
            end
        end
    end

    return nothing
end

@inline function get_speed_vpa_inner_args(vpa_advect, fvec, moments, fields, r_advect,
                                          alpha_advect, z_advect, geometry, vperp, vpa,
                                          evolve_density::Val, evolve_upar::Val,
                                          evolve_p::Val)
    if evolve_p === Val(true)
        return vpa_advect, moments.ion.vth, fields.gEz, geometry.bzed, geometry.Bmag,
               geometry.dBdz, moments.ion.dupar_dt, moments.ion.dupar_dr,
               moments.ion.dupar_dz, moments.ion.dvth_dt, moments.ion.dvth_dr,
               moments.ion.dvth_dz, r_advect, alpha_advect, z_advect, vperp.grid,
               vpa.grid, evolve_density, evolve_upar, evolve_p
    elseif evolve_upar === Val(true)
        return vpa_advect, fields.gEz, geometry.bzed, geometry.Bmag, geometry.dBdz,
               moments.ion.dupar_dt, moments.ion.dupar_dr, moments.ion.dupar_dz,
               r_advect, alpha_advect, z_advect, vperp.grid, evolve_density, evolve_upar,
               evolve_p
    elseif evolve_density === Val(true)
        return vpa_advect, fields.gEz, geometry.bzed, geometry.Bmag, geometry.dBdz,
               vperp.grid, evolve_density, evolve_upar, evolve_p
    else
        return vpa_advect, fields.gEz, geometry.bzed, geometry.dBdz, geometry.Bmag,
               vperp.grid, evolve_density, evolve_upar, evolve_p
    end
end

@inline function get_speed_vpa_inner_views_sr(is, ir, vpa_advect, vth, gEz, bzed, Bmag,
                                              dBdz, dupar_dt, dupar_dr, dupar_dz,
                                              dvth_dt, dvth_dr, dvth_dz, r_advect,
                                              alpha_advect, z_advect, wperp, wpa,
                                              evolve_density::Val{true},
                                              evolve_upar::Val{true}, evolve_p::Val{true})
    return @views vpa_advect[:,:,:,ir,is], vth[:,ir,is], gEz[:,:,ir,is], bzed[:,ir],
                  Bmag[:,ir], dBdz[:,ir], dupar_dt[:,ir,is], dupar_dr[:,ir,is],
                  dupar_dz[:,ir,is], dvth_dt[:,ir,is], dvth_dr[:,ir,is],
                  dvth_dz[:,ir,is], r_advect[:,:,:,ir,is], alpha_advect[:,:,:,ir,is],
                  z_advect[:,:,:,ir,is], wperp, wpa, evolve_density, evolve_upar,
                  evolve_p
end

@inline function get_speed_vpa_inner_views_z(iz, vpa_advect, vth, gEz, bzed, Bmag, dBdz,
                                             dupar_dt, dupar_dr, dupar_dz, dvth_dt,
                                             dvth_dr, dvth_dz, r_advect, alpha_advect,
                                             z_advect, wperp, wpa,
                                             evolve_density::Val{true},
                                             evolve_upar::Val{true}, evolve_p::Val{true})
    return @views vpa_advect[:,:,iz], vth[iz], gEz[:,iz], bzed[iz], Bmag[iz], dBdz[iz],
                  dupar_dt[iz], dupar_dr[iz], dupar_dz[iz], dvth_dt[iz], dvth_dr[iz],
                  dvth_dz[iz], r_advect[:,:,iz], alpha_advect[:,:,iz], z_advect[:,:,iz],
                  wperp, wpa, evolve_density, evolve_upar, evolve_p
end

@inline function get_speed_vpa_inner_views_vperp(ivperp, vpa_advect, vth, gEz, bzed,
                                                 Bmag, dBdz, dupar_dt, dupar_dr,
                                                 dupar_dz, dvth_dt, dvth_dr, dvth_dz,
                                                 r_advect, alpha_advect, z_advect, wperp,
                                                 wpa, evolve_density::Val{true},
                                                 evolve_upar::Val{true},
                                                 evolve_p::Val{true})
    this_wperp = wperp[ivperp]
    mu = 0.5 * this_wperp^2 * vth^2 / Bmag
    return @views vpa_advect[:,ivperp], vth, gEz[ivperp], bzed, dBdz, dupar_dt, dupar_dr,
                  dupar_dz, dvth_dt, dvth_dr, dvth_dz, r_advect[:,ivperp],
                  alpha_advect[:,ivperp], z_advect[:,ivperp], mu, wpa, evolve_density,
                  evolve_upar, evolve_p
end

"""
update the advection speed in the parallel velocity coordinate for the case
where density, flow and pressure are evolved independently from the pdf;
in this case, the parallel velocity coordinate is the normalized peculiar velocity
wpa = (vpa - upar)/vth
"""
function update_speed_vpa_inner!(vpa_advect, vth, gEz, bzed, dBdz, dupar_dt, dupar_dr,
                                 dupar_dz, dvth_dt, dvth_dr, dvth_dz, r_advect,
                                 alpha_advect, z_advect, mu, wpa,
                                 evolve_density::Val{true}, evolve_upar::Val{true},
                                 evolve_p::Val{true})

    @. vpa_advect =
           (bzed * gEz
            - (dupar_dt + r_advect * dupar_dr + (alpha_advect + z_advect) * dupar_dz)
            - wpa * (dvth_dt + r_advect * dvth_dr + (alpha_advect + z_advect) * dvth_dz)
            - (mu * bzed * dBdz)
           ) / vth

    return nothing
end

@inline function get_speed_vpa_inner_views_sr(is, ir, vpa_advect, gEz, bzed, Bmag, dBdz,
                                              dupar_dt, dupar_dr, dupar_dz, r_advect,
                                              alpha_advect, z_advect, vperp,
                                              evolve_density::Val{true},
                                              evolve_upar::Val{true},
                                              evolve_p::Val{false})
    return @views vpa_advect[:,:,:,ir,is], gEz[:,:,ir,is], bzed[:,ir], Bmag[:,ir],
                  dBdz[:,ir], dupar_dt[:,ir,is], dupar_dr[:,ir,is], dupar_dz[:,ir,is],
                  r_advect[:,:,:,ir,is], alpha_advect[:,:,:,ir,is],
                  z_advect[:,:,:,ir,is], vperp, evolve_density, evolve_upar, evolve_p
end

@inline function get_speed_vpa_inner_views_z(iz, vpa_advect, gEz, bzed, Bmag, dBdz,
                                             dupar_dt, dupar_dr, dupar_dz, r_advect,
                                             alpha_advect, z_advect, vperp,
                                             evolve_density::Val{true},
                                             evolve_upar::Val{true},
                                             evolve_p::Val{false})
    return @views vpa_advect[:,:,iz], gEz[:,iz], bzed[iz], Bmag[iz], dBdz[iz],
                  dupar_dt[iz], dupar_dr[iz], dupar_dz[iz], r_advect[:,:,iz],
                  alpha_advect[:,:,iz], z_advect[:,:,iz], vperp, evolve_density,
                  evolve_upar, evolve_p
end

@inline function get_speed_vpa_inner_views_vperp(ivperp, vpa_advect, gEz, bzed, Bmag,
                                                 dBdz, dupar_dt, dupar_dr, dupar_dz,
                                                 r_advect, alpha_advect, z_advect, vperp,
                                                 evolve_density::Val{true},
                                                 evolve_upar::Val{true},
                                                 evolve_p::Val{false})
    mu = 0.5 * vperp[ivperp]^2 / Bmag
    return @views vpa_advect[:,ivperp], gEz[ivperp], bzed, dBdz, dupar_dt, dupar_dr,
                  dupar_dz, r_advect[:,ivperp], alpha_advect[:,ivperp],
                  z_advect[:,ivperp], mu, evolve_density, evolve_upar, evolve_p
end

"""
update the advection speed in the parallel velocity coordinate for the case
where density and flow are evolved independently from the pdf;
in this case, the parallel velocity coordinate is the peculiar velocity
wpa = vpa-upar
"""
function update_speed_vpa_inner!(vpa_advect, gEz, bzed, dBdz, dupar_dt, dupar_dr,
                                 dupar_dz, r_advect, alpha_advect, z_advect, mu,
                                 evolve_density::Val{true}, evolve_upar::Val{true},
                                 evolve_p::Val{false})
    @. vpa_advect =
           (bzed * gEz
            - (dupar_dt + r_advect * dupar_dr + (alpha_advect + z_advect) * dupar_dz)
            - (mu * bzed * dBdz)
           )

    return nothing
end

@inline function get_speed_vpa_inner_views_sr(is, ir, vpa_advect, gEz, bzed, Bmag, dBdz,
                                              vperp, evolve_density::Val{true},
                                              evolve_upar::Val{false},
                                              evolve_p::Val{false})
    return @views vpa_advect[:,:,:,ir,is], gEz[:,:,ir,is], bzed[:,ir], Bmag[:,ir],
                  dBdz[:,ir], vperp, evolve_density, evolve_upar, evolve_p
end

@inline function get_speed_vpa_inner_views_z(iz, vpa_advect, gEz, bzed, Bmag, dBdz,
                                             vperp, evolve_density::Val{true},
                                             evolve_upar::Val{false},
                                             evolve_p::Val{false})
    return @views vpa_advect[:,:,iz], gEz[:,iz], bzed[iz], Bmag[iz], dBdz[iz], vperp,
                  evolve_density, evolve_upar, evolve_p
end

@inline function get_speed_vpa_inner_views_vperp(ivperp, vpa_advect, gEz, bzed, Bmag,
                                                 dBdz, vperp, evolve_density::Val{true},
                                                 evolve_upar::Val{false},
                                                 evolve_p::Val{false})
    mu = 0.5 * vperp[ivperp]^2 / Bmag
    return @views vpa_advect[:,ivperp], gEz[ivperp], bzed, dBdz, mu, evolve_density,
                  evolve_upar, evolve_p
end

"""
update the advection speed in the parallel velocity coordinate for the case
where density is evolved independently from the pdf;
in this case, the parallel velocity coordinate is unchanged.
"""
function update_speed_vpa_inner!(vpa_advect, gEz, bzed, dBdz, mu,
                                 evolve_density::Val{true}, evolve_upar::Val{false},
                                 evolve_p::Val{false})
    @. vpa_advect = bzed * gEz - mu * bzed * dBdz

    return nothing
end

@inline function get_speed_vpa_inner_views_sr(is, ir, vpa_advect, gEz, bzed, dBdz, Bmag,
                                              vperp, evolve_density::Val{false},
                                              evolve_upar::Val{false},
                                              evolve_p::Val{false})
    return @views vpa_advect[:,:,:,ir,is], gEz[:,:,ir,is], bzed[:,ir], dBdz[:,ir],
                  Bmag[:,ir], vperp, evolve_density, evolve_upar, evolve_p
end

@inline function get_speed_vpa_inner_views_z(iz, vpa_advect, gEz, bzed, dBdz, Bmag, vperp,
                                             evolve_density::Val{false},
                                             evolve_upar::Val{false},
                                             evolve_p::Val{false})
    return @views vpa_advect[:,:,iz], gEz[:,iz], bzed[iz], dBdz[iz], Bmag[iz], vperp,
                  evolve_density, evolve_upar, evolve_p
end

@inline function get_speed_vpa_inner_views_vperp(ivperp, vpa_advect, gEz, bzed, dBdz,
                                                 Bmag, vperp, evolve_density::Val{false},
                                                 evolve_upar::Val{false},
                                                 evolve_p::Val{false})
    # mu, the adiabatic invariant
    mu = 0.5 * vperp[ivperp]^2 / Bmag
    return @views vpa_advect[:,ivperp], gEz[ivperp], bzed, dBdz, mu, evolve_density,
                  evolve_upar, evolve_p
end

"""
update the advection speed in the parallel velocity coordinate for the case
where no moments are evolved independently from the pdf. vpa is unchanged.
"""
function update_speed_vpa_inner!(vpa_advect, gEz, bzed, dBdz, mu,
                                 evolve_density::Val{false}, evolve_upar::Val{false},
                                 evolve_p::Val{false})
    @. vpa_advect = bzed * gEz - mu * bzed * dBdz
    return nothing
end

end
