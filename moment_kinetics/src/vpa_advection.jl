"""
"""
module vpa_advection

export vpa_advection!
export update_speed_vpa!

using ..advection: advance_f_local!
using ..boundary_conditions: enforce_v_boundary_condition_local!
using ..communication
using ..looping
using ..moment_constraints: hard_force_moment_constraints!,
                            moment_constraints_on_residual!
using ..timer_utils
using ..moment_kinetics_structs: scratch_pdf, weak_discretization_info
using ..nonlinear_solvers: newton_solve!
using ..velocity_moments: update_derived_moments!, calculate_ion_moment_derivatives!

using ..array_allocation: allocate_float
using ..boundary_conditions: vpagrid_to_dzdt
using ..calculus: second_derivative!
using LinearAlgebra
using SparseArrays

"""
"""
@timeit global_timer vpa_advection!(
                         f_out, fvec_in, fields, moments, advect, vpa, vperp, z, r, dt, t,
                         vpa_spectral, composition, collisions, ion_source_settings,
                         geometry) = begin

    @begin_s_r_z_vperp_region()

    # only have a parallel acceleration term for neutrals if using the peculiar velocity
    # wpar = vpar - upar as a variable; i.e., d(wpar)/dt /=0 for neutrals even though d(vpar)/dt = 0.

    # calculate the advection speed corresponding to current f
    update_speed_vpa!(advect, fields, fvec_in, moments, vpa, vperp, z, r, composition,
                      collisions, ion_source_settings, t, geometry)
    @loop_s is begin
        @loop_r_z_vperp ir iz ivperp begin
            @views advance_f_local!(f_out[:,ivperp,iz,ir,is], fvec_in.pdf[:,ivperp,iz,ir,is],
                                    advect[is], ivperp, iz, ir, vpa, dt, vpa_spectral)
        end
    end
end

"""
"""
@timeit global_timer implicit_vpa_advection!(
                         f_out, fvec_in, fields, moments, z_advect, vpa_advect, vpa,
                         vperp, z, r, dt, t, r_spectral, z_spectral, vpa_spectral,
                         composition, collisions, ion_source_settings, geometry,
                         nl_solver_params, vpa_diffusion, num_diss_params, gyroavs,
                         scratch_dummy) = begin
    if vperp.n > 1 && (moments.evolve_density || moments.evolve_upar || moments.evolve_ppar)
        error("Moment constraints in implicit_vpa_advection!() do not support 2V runs yet")
    end

    # calculate the advection speed corresponding to current f
    update_speed_vpa!(vpa_advect, fields, fvec_in, moments, vpa, vperp, z, r, composition,
                      collisions, ion_source_settings, t, geometry)

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
            speed = @view vpa_advect[is].speed[:,ivperp,iz,ir]

            if z.irank == 0 && iz == 1
                @. vpa.scratch = vpagrid_to_dzdt(vpa.grid, moments.ion.vth[iz,ir,is],
                                                 fvec_in.upar[iz,ir,is],
                                                 moments.evolve_ppar,
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
                @. vpa.scratch = vpagrid_to_dzdt(vpa.grid, moments.ion.vth[iz,ir,is],
                                                 fvec_in.upar[iz,ir,is],
                                                 moments.evolve_ppar,
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

            #    advection_matrix = allocate_float(vpa.n, vpa.n)
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
                advance_f_local!(residual, f_new, vpa_advect[is], ivperp, iz, ir, vpa, dt,
                                 vpa_spectral)

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
function update_speed_vpa!(advect, fields, fvec, moments, vpa, vperp, z, r, composition,
                           collisions, ion_source_settings, t, geometry)
    @boundscheck r.n == size(advect[1].speed,4) || throw(BoundsError(advect))
    @boundscheck z.n == size(advect[1].speed,3) || throw(BoundsError(advect))
    @boundscheck vperp.n == size(advect[1].speed,2) || throw(BoundsError(advect))
    #@boundscheck composition.n_ion_species == size(advect,2) || throw(BoundsError(advect))
    @boundscheck composition.n_ion_species == size(advect,1) || throw(BoundsError(advect))
    @boundscheck vpa.n == size(advect[1].speed,1) || throw(BoundsError(speed))
    if vpa.advection.option == "default"
        # dvpa/dt = Ze/m ⋅ E_parallel - (vperp^2/2B) bz dB/dz
        # magnetic mirror term only supported for standard DK implementation
        update_speed_default!(advect, fields, fvec, moments, vpa, vperp, z, r, composition,
                              collisions, ion_source_settings, t, geometry)
    elseif vpa.advection.option == "constant"
        @begin_serial_region()
        @serial_region begin
            # Not usually used - just run in serial
            # dvpa/dt = constant
            for is ∈ 1:composition.n_ion_species
                update_speed_constant!(advect[is], vpa, 1:vperp.n, 1:z.n, 1:r.n)
            end
        end
    elseif vpa.advection.option == "linear"
        @begin_serial_region()
        @serial_region begin
            # Not usually used - just run in serial
            # dvpa/dt = constant ⋅ (vpa + L_vpa/2)
            for is ∈ 1:composition.n_ion_species
                update_speed_linear!(advect[is], vpa, 1:vperp.n, 1:z.n, 1:r.n)
            end
        end
    end
    return nothing
end

"""
"""
function update_speed_default!(advect, fields, fvec, moments, vpa, vperp, z, r, composition,
                               collisions, ion_source_settings, t, geometry)
    if moments.evolve_ppar && moments.evolve_upar
        update_speed_n_u_p_evolution!(advect, fields, fvec, moments, vpa, z, r,
                                      composition, collisions, ion_source_settings)
    elseif moments.evolve_ppar
        update_speed_n_p_evolution!(advect, fields, fvec, moments, vpa, z, r, composition,
                                    collisions, ion_source_settings)
    elseif moments.evolve_upar
        update_speed_n_u_evolution!(advect, fields, fvec, moments, vpa, z, r, composition,
                                    collisions, ion_source_settings)
    else
        bzed = geometry.bzed
        dBdz = geometry.dBdz
        Bmag = geometry.Bmag
        @inbounds @fastmath begin
            @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
                # mu, the adiabatic invariant
                mu = 0.5*(vperp.grid[ivperp]^2)/Bmag[iz,ir]
                # bzed = B_z/B
                advect[is].speed[ivpa,ivperp,iz,ir] = (0.5*bzed[iz,ir]*fields.gEz[ivperp,iz,ir,is] - 
                                                       mu*bzed[iz,ir]*dBdz[iz,ir])
            end
        end
    end
end

"""
update the advection speed in the parallel velocity coordinate for the case
where density, flow and pressure are evolved independently from the pdf;
in this case, the parallel velocity coordinate is the normalized peculiar velocity
wpahat = (vpa - upar)/vth
"""
function update_speed_n_u_p_evolution!(advect, fields, fvec, moments, vpa, z, r,
                                       composition, collisions, ion_source_settings)
    upar = fvec.upar
    vth = moments.ion.vth
    Ez = fields.Ez
    dupar_dz = moments.ion.dupar_dz
    dvth_dz = moments.ion.dvth_dz
    dupar_dt = moments.ion.dupar_dt
    dvth_dt = moments.ion.dvth_dt
    wpa = vpa.grid
    @loop_s is begin
        speed = advect[is].speed
        @loop_r ir begin
            # update parallel acceleration to account for:
            @loop_z_vperp iz ivperp begin
                @. speed[:,ivperp,iz,ir] =
                    (0.5 * Ez[iz,ir]
                     - (dupar_dt[iz,ir,is] + (vth[iz,ir,is] * wpa + upar[iz,ir,is]) * dupar_dz[iz,ir,is])
                     - wpa * (dvth_dt[iz,ir,is] + (vth[iz,ir,is] * wpa + upar[iz,ir,is]) * dvth_dz[iz,ir,is])
                    ) / vth[iz,ir,is]
            end
        end
    end

    return nothing
end

"""
update the advection speed in the parallel velocity coordinate for the case
where density and pressure are evolved independently from the pdf;
in this case, the parallel velocity coordinate is the normalized velocity
vpahat = vpa/vth
"""
function update_speed_n_p_evolution!(advect, fields, fvec, moments, vpa, z, r,
                                     composition, collisions, ion_source_settings)
    vth = moments.ion.vth
    Ez = fields.Ez
    dvth_dz = moments.ion.dvth_dz
    dvth_dt = moments.ion.dvth_dt
    wpa = vpa.grid
    @loop_s is begin
        speed = advect[is].speed
        @loop_r ir begin
            # update parallel acceleration to account for:
            @loop_z_vperp iz ivperp begin
                @. speed[:,ivperp,iz,ir] =
                    (0.5 * Ez[iz,ir]
                     - wpa * (dvth_dt[iz,ir,is] + vth[iz,ir,is] * wpa * dvth_dz[iz,ir,is])
                    ) / vth[iz,ir,is]
            end
        end
    end

    return nothing
end

"""
update the advection speed in the parallel velocity coordinate for the case
where density and flow are evolved independently from the pdf;
in this case, the parallel velocity coordinate is the peculiar velocity
wpa = vpa-upar
"""
function update_speed_n_u_evolution!(advect, fields, fvec, moments, vpa, z, r,
                                     composition, collisions, ion_source_settings)
    upar = fvec.upar
    Ez = fields.Ez
    dupar_dz = moments.ion.dupar_dz
    dupar_dt = moments.ion.dupar_dt
    wpa = vpa.grid
    @loop_s is begin
        speed = advect[is].speed
        @loop_r ir begin
            # update parallel acceleration to account for:
            @loop_z_vperp iz ivperp begin
                @. speed[:,ivperp,iz,ir] =
                    (0.5 * Ez[iz,ir]
                     - (dupar_dt[iz,ir,is] + (wpa + upar[iz,ir,is]) * dupar_dz[iz,ir,is])
                    )
            end
        end
    end

    return nothing
end

"""
update the advection speed dvpa/dt = constant
"""
function update_speed_constant!(advect, vpa, vperp_range, z_range, r_range)
    #@inbounds @fastmath begin
    for ir ∈ r_range
        for iz ∈ z_range
            for ivperp ∈ vperp_range
                @views advect.speed[:,ivperp,iz,ir] .= vpa.advection.constant_speed
            end
        end
    end
    #end
end

"""
update the advection speed dvpa/dt = const*(vpa + L/2)
"""
function update_speed_linear(advect, vpa, vperp_range, z_range, r_range)
    @inbounds @fastmath begin
        for ir ∈ r_range
            for iz ∈ z_range
                for ivperp ∈ vperp_range
                    @views @. advect.speed[:,ivperp,iz,ir] = vpa.advection.constant_speed*(vpa.grid+0.5*vpa.L)
                end
            end
        end
    end
end

end
