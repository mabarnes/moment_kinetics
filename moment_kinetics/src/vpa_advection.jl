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
using ..moment_kinetics_structs: scratch_pdf, weak_discretization_info
using ..nonlinear_solvers: newton_solve!
using ..velocity_moments: update_derived_moments!, calculate_ion_moment_derivatives!

using ..array_allocation: allocate_float
using ..boundary_conditions: vpagrid_to_dzdt
using ..calculus: second_derivative!
using ..maxwell_diffusion: ion_vpa_maxwell_diffusion_inner!
using LinearAlgebra
using SparseArrays

"""
"""
function vpa_advection!(f_out, fvec_in, fields, moments, advect, vpa, vperp, z, r, dt, t,
                        vpa_spectral, composition, collisions, ion_source_settings, geometry)

    begin_s_r_z_vperp_region()

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
function implicit_vpa_advection!(f_out, fvec_in, fields, moments, z_advect, vpa_advect,
                                 vpa, vperp, z, r, dt, t, r_spectral, z_spectral,
                                 vpa_spectral, composition, collisions,
                                 ion_source_settings, geometry, nl_solver_params,
                                 maxwell_diffusion, vpa_diffusion, num_diss_params,
                                 gyroavs, scratch_dummy)
    evolve_density = Val(moments.evolve_density)
    evolve_upar = Val(moments.evolve_upar)
    evolve_ppar = Val(moments.evolve_ppar)
    if vperp.n > 1 && (moments.evolve_density || moments.evolve_upar || moments.evolve_ppar)
        error("Moment constraints in implicit_vpa_advection!() do not support 2V runs yet")
    end

    # calculate the advection speed corresponding to current f
    update_speed_vpa!(vpa_advect, fields, fvec_in, moments, vpa, vperp, z, r, composition,
                      collisions, ion_source_settings, t, geometry)

    # Ensure moments are consistent with f_new
    new_scratch = scratch_pdf(f_out, fvec_in.density, fvec_in.upar, fvec_in.ppar,
                              fvec_in.pperp, fvec_in.temp_z_s, fvec_in.electron_density,
                              fvec_in.electron_upar, fvec_in.electron_ppar,
                              fvec_in.electron_pperp, fvec_in.electron_temp,
                              fvec_in.pdf_neutral, fvec_in.density_neutral,
                              fvec_in.uz_neutral, fvec_in.pz_neutral)
    update_derived_moments!(new_scratch, moments, vpa, vperp, z, r, composition,
                            r_spectral, geometry, gyroavs, scratch_dummy, z_advect, false)
    calculate_ion_moment_derivatives!(moments, new_scratch, scratch_dummy, z,
                                      z_spectral,
                                      num_diss_params.ion.moment_dissipation_coefficient)

    begin_s_r_z_vperp_region()

    coords = (vpa=vpa,)
    vpa_bc = vpa.bc
    minval = num_diss_params.ion.force_minimum_pdf_value
    n = fvec_in.density
    upar = fvec_in.upar
    vth = moments.ion.vth
    vpa_dissipation_coefficient = num_diss_params.ion.vpa_dissipation_coefficient
    include_vpa_dissipation = (vpa_dissipation_coefficient > 0.0)
    maxwell_D_ii = collisions.mxwl_diff.D_ii
    zero = 1.0e-14

    function get_precon(is, ir, iz, ivperp, speed; icut_lower=2, icut_upper=vpa.n-1)
        if maxwell_diffusion &&
            !(isa(evolve_density, Val{true}) && isa(evolve_upar, Val{true}) &&
              isa(evolve_ppar, Val{true}))
            error("vpa advection preconditioner with maxwell-diffusion is only "
                  * "implemented for fully moment-kinetic case so far")
        end
        if icut_lower < 2
            icut_lower = 2
        end
        if icut_upper > vpa.n - 1
            icut_upper = vpa.n - 1
        end

        if maxwell_diffusion
            maxwell_prefactor = maxwell_D_ii * n[iz,ir,is] / vth[iz,ir,is]^3
        end

        # Dirichlet boundary conditions set the first and last values of the solution
        # to zero, so can remove the first/last rows/columns of the matrix.
        # When there is a 'cutoff index' because we are imposing sheath-edge boundary
        # conditions, more values (all those outside the `icut` index) are
        # zero-ed out, and so removed from the matrix system.

        precon_matrix = allocate_float(vpa.n, vpa.n)
        precon_matrix .= 0.0

        for i ∈ 1:vpa.nelement_local
            imin = vpa.imin[i] - (i != 1)
            imax = vpa.imax[i]

            # vpa advection terms
            if i == 1 && i == vpa.nelement_local
                @views precon_matrix .-= reshape(speed[imin:imax], :, 1) .* vpa_spectral.lobatto.Dmat ./ vpa.element_scale[i]
            elseif i == 1
                @views precon_matrix[imin:imax-1,imin:imax] .-= reshape(speed[imin:imax-1], :, 1) .* vpa_spectral.lobatto.Dmat[1:end-1,1:end] ./ vpa.element_scale[i]
                if speed[imax] < 0.0
                    # Do nothing
                elseif speed[imax] > 0.0
                    @views @. precon_matrix[imax,imin:imax] -= speed[imax] * vpa_spectral.lobatto.Dmat[end,1:end] / vpa.element_scale[i]
                else
                    @views @. precon_matrix[imax,imin:imax] -= 0.5 * speed[imax] * vpa_spectral.lobatto.Dmat[end,1:end] / vpa.element_scale[i]
                end
            elseif i == vpa.nelement_local
                if speed[imin] < 0.0
                    @views @. precon_matrix[imin,imin:imax] -= speed[imin] * vpa_spectral.lobatto.Dmat[1,:] / vpa.element_scale[i]
                elseif speed[imin] > 0.0
                    # Do nothing
                else
                    @views @. precon_matrix[imin,imin:imax] -= 0.5 .* speed[imin] * vpa_spectral.lobatto.Dmat[1,:] / vpa.element_scale[i]
                end
                @views precon_matrix[imin+1:imax,imin:imax] .-= reshape(speed[imin+1:imax], :, 1) .* vpa_spectral.lobatto.Dmat[2:end,:] ./ vpa.element_scale[i]
            else
                if speed[imin] < 0.0
                    @views @. precon_matrix[imin,imin:imax] -= speed[imin] * vpa_spectral.lobatto.Dmat[1,:] / vpa.element_scale[i]
                elseif speed[imin] > 0.0
                    # Do nothing
                else
                    @views @. precon_matrix[imin,imin:imax] -= 0.5 * speed[imin] * vpa_spectral.lobatto.Dmat[1,:] / vpa.element_scale[i]
                end
                @views precon_matrix[imin+1:imax-1,imin:imax] .-= reshape(speed[imin+1:imax-1], :, 1) .* vpa_spectral.lobatto.Dmat[2:end-1,:] ./ vpa.element_scale[i]
                if speed[imax] < 0.0
                    # Do nothing
                elseif speed[imax] > 0.0
                    @views @. precon_matrix[imax,imin:imax] -= speed[imax] * vpa_spectral.lobatto.Dmat[end,:] / vpa.element_scale[i]
                else
                    @views @. precon_matrix[imax,imin:imax] -= 0.5 * speed[imax] * vpa_spectral.lobatto.Dmat[end,:] / vpa.element_scale[i]
                end
            end

            if maxwell_diffusion
                if i == 1 && i == vpa.nelement_local
                    @views precon_matrix .+= maxwell_prefactor .* vpa_spectral.lobatto.Dmat ./ vpa.element_scale[i] .* reshape(vpa.grid[imin:imax], 1, :)
                elseif i == 1
                    @views precon_matrix[imin:imax-1,imin:imax] .+= maxwell_prefactor .* vpa_spectral.lobatto.Dmat[1:end-1,:] ./ vpa.element_scale[i] .* reshape(vpa.grid[imin:imax], 1, :)
                    @views @. precon_matrix[imax,imin:imax] += 0.5 * maxwell_prefactor * vpa_spectral.lobatto.Dmat[end,:] / vpa.element_scale[i] * vpa.grid[imin:imax]
                elseif i == vpa.nelement_local
                    @views @. precon_matrix[imin,imin:imax] += 0.5 .* maxwell_prefactor * vpa_spectral.lobatto.Dmat[1,:] / vpa.element_scale[i] * vpa.grid[imin:imax]
                    @views precon_matrix[imin+1:imax,imin:imax] .+= maxwell_prefactor .* vpa_spectral.lobatto.Dmat[2:end,:] ./ vpa.element_scale[i] .* reshape(vpa.grid[imin:imax], 1, :)
                else
                    @views @. precon_matrix[imin,imin:imax] += 0.5 * maxwell_prefactor * vpa_spectral.lobatto.Dmat[1,:] / vpa.element_scale[i] * vpa.grid[imin:imax]
                    @views precon_matrix[imin+1:imax-1,imin:imax] .+= maxwell_prefactor .* vpa_spectral.lobatto.Dmat[2:end-1,:] ./ vpa.element_scale[i] .* reshape(vpa.grid[imin:imax], 1, :)
                    @views @. precon_matrix[imax,imin:imax] += 0.5 * maxwell_prefactor * vpa_spectral.lobatto.Dmat[end,:] / vpa.element_scale[i] * vpa.grid[imin:imax]
                end
            end
        end

        # Remove first/last row/column, to represent Dirichlet boundary conditions
        precon_matrix = @view precon_matrix[2:end-1,2:end-1]

        if include_vpa_dissipation || maxwell_diffusion
            # This allocates a new matrix - to avoid this would need to pre-allocate a
            # suitable buffer somewhere.
            precon_matrix .= vpa_spectral.mass_matrix[2:end-1,2:end-1] * precon_matrix
        end

        if include_vpa_dissipation || maxwell_diffusion
            @. precon_matrix += vpa_dissipation_coefficient * vpa_spectral.K_matrix[2:end-1,2:end-1]
        end

        if maxwell_diffusion
            precon_matrix .+= maxwell_prefactor * vpa_spectral.K_matrix[2:end-1,2:end-1]
        end

        precon_matrix = @view precon_matrix[icut_lower-1:icut_upper-1,icut_lower-1:icut_upper-1]

        if include_vpa_dissipation || maxwell_diffusion
            @views precon_matrix .=
                vpa_spectral.mass_matrix[icut_lower:icut_upper,icut_lower:icut_upper] .-
                dt .* precon_matrix
        else
            precon_matrix .= Diagonal(ones(icut_upper - icut_lower + 1)) .-
                             dt .* precon_matrix
        end

        precon_lu = lu(sparse(precon_matrix))

        return precon_lu
    end

    @loop_s is begin
        @loop_r_z_vperp ir iz ivperp begin
            f_old_no_bc = @view fvec_in.pdf[:,ivperp,iz,ir,is]
            this_f_out = @view f_out[:,ivperp,iz,ir,is]
            speed = @view vpa_advect[is].speed[:,ivperp,iz,ir]

            if z.irank == 0 && iz == 1
                @. vpa.scratch = vpagrid_to_dzdt(vpa.grid, vth[iz,ir,is],
                                                 fvec_in.upar[iz,ir,is],
                                                 moments.evolve_ppar,
                                                 moments.evolve_upar)
                icut_lower_z = searchsortedlast(vpa.scratch, zero) + 1
            end
            if z.irank == z.nrank - 1 && iz == z.n
                @. vpa.scratch = vpagrid_to_dzdt(vpa.grid, vth[iz,ir,is],
                                                 fvec_in.upar[iz,ir,is],
                                                 moments.evolve_ppar,
                                                 moments.evolve_upar)
                icut_upper_z = searchsortedfirst(vpa.scratch, -zero) - 1
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
            f_old = vpa.scratch9 .= f_old_no_bc
            apply_bc!(f_old)

            if nl_solver_params.stage_counter[] % nl_solver_params.preconditioner_update_interval == 0
                if z.irank == 0 && iz == 1
                    nl_solver_params.preconditioners[ivperp,iz,ir,is] =
                        (get_precon(is, ir, iz, ivperp, speed; icut_upper=icut_lower_z-1),
                         2, icut_lower_z-1)
                elseif z.irank == z.nrank - 1 && iz == z.n
                    nl_solver_params.preconditioners[ivperp,iz,ir,is] =
                        (get_precon(is, ir, iz, ivperp, speed; icut_lower=icut_upper_z+1),
                         icut_upper_z+1, vpa.n-1)
                else
                    nl_solver_params.preconditioners[ivperp,iz,ir,is] =
                        (get_precon(is, ir, iz, ivperp, speed),
                         2, vpa.n-1)
                end
            end

            function preconditioner(x)
                precon_lu, icut_lower, icut_upper =
                    nl_solver_params.preconditioners[ivperp,iz,ir,is]
                if include_vpa_dissipation || maxwell_diffusion
                    @views mul!(vpa.scratch[icut_lower:icut_upper],
                                vpa_spectral.mass_matrix[icut_lower:icut_upper,icut_lower:icut_upper],
                                x[icut_lower:icut_upper])
                end
                @views ldiv!(x[icut_lower:icut_upper], precon_lu,
                             vpa.scratch[icut_lower:icut_upper])
                return nothing
            end

            left_preconditioner = identity
            right_preconditioner = preconditioner

            # Define a function whose input is `f_new`, so that when it's output
            # `residual` is zero, f_new is the result of a backward-Euler timestep:
            #   (f_new - f_old) / dt = RHS(f_new)
            # ⇒ f_new - f_old - dt*RHS(f_new) = 0
            function residual_func!(residual, f_new)
                apply_bc!(f_new)
                residual .= f_old
                advance_f_local!(residual, f_new, vpa_advect[is], ivperp, iz, ir, vpa, dt,
                                 vpa_spectral)

                if include_vpa_dissipation
                    second_derivative!(vpa.scratch, f_new, vpa, vpa_spectral)
                    @. residual += dt * vpa_dissipation_coefficient * vpa.scratch
                end

                if maxwell_diffusion
                    ion_vpa_maxwell_diffusion_inner!(residual, f_new, n[iz,ir,is],
                                                     upar[iz,ir,is], vth[iz,ir,is], vpa,
                                                     vpa_spectral, maxwell_D_ii, dt,
                                                     evolve_density, evolve_upar,
                                                     evolve_ppar)
                end

                # Now
                #   residual = f_old + dt*RHS(f_new)
                # so update to desired residual
                @. residual = f_new - residual

                apply_bc!(residual)
            end

            # Buffers
            # Note vpa,scratch is used by advance_f!, so we cannot use it here.
            residual = vpa.scratch4
            delta_x = vpa.scratch5
            rhs_delta = vpa.scratch6
            v = vpa.scratch7
            w = vpa.scratch8

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

    nl_solver_params.stage_counter[] += 1

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
        begin_serial_region()
        @serial_region begin
            # Not usually used - just run in serial
            # dvpa/dt = constant
            for is ∈ 1:composition.n_ion_species
                update_speed_constant!(advect[is], vpa, 1:vperp.n, 1:z.n, 1:r.n)
            end
        end
    elseif vpa.advection.option == "linear"
        begin_serial_region()
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
        update_speed_n_u_p_evolution!(advect, fvec, moments, vpa, z, r, composition,
                                      collisions, ion_source_settings)
    elseif moments.evolve_ppar
        update_speed_n_p_evolution!(advect, fields, fvec, moments, vpa, z, r, composition,
                                    collisions, ion_source_settings)
    elseif moments.evolve_upar
        update_speed_n_u_evolution!(advect, fvec, moments, vpa, z, r, composition,
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
function update_speed_n_u_p_evolution!(advect, fvec, moments, vpa, z, r, composition,
                                       collisions, ion_source_settings)
    @loop_s is begin
        @loop_r ir begin
            # update parallel acceleration to account for:
            # • parallel derivative of parallel pressure
            # • (wpar/2*ppar)*dqpar/dz
            # • -wpar^2 * d(vth)/dz term
            @loop_z_vperp iz ivperp begin
                @views @. advect[is].speed[:,ivperp,iz,ir] =
                    moments.ion.dppar_dz[iz,ir,is]/(fvec.density[iz,ir,is]*moments.ion.vth[iz,ir,is]) +
                    0.5*vpa.grid*moments.ion.dqpar_dz[iz,ir,is]/fvec.ppar[iz,ir,is] -
                    vpa.grid^2*moments.ion.dvth_dz[iz,ir,is]
            end
        end
    end
    # add in contributions from charge exchange and ionization collisions
    if composition.n_neutral_species > 0 &&
            (abs(collisions.charge_exchange) > 0.0 || abs(collisions.ionization) > 0.0)

        @loop_s is begin
            @loop_r_z_vperp ir iz ivperp begin
                @views @. advect[is].speed[:,ivperp,iz,ir] +=
                    collisions.charge_exchange *
                    (0.5*vpa.grid/fvec.ppar[iz,ir,is]
                     * (fvec.density_neutral[iz,ir,is]*fvec.ppar[iz,ir,is]
                        - fvec.density[iz,ir,is]*fvec.pz_neutral[iz,ir,is]
                        - fvec.density[iz,ir,is]*fvec.density_neutral[iz,ir,is]
                          * (fvec.upar[iz,ir,is]-fvec.uz_neutral[iz,ir,is])^2)
                     - fvec.density_neutral[iz,ir,is]
                       * (fvec.uz_neutral[iz,ir,is]-fvec.upar[iz,ir,is])
                       / moments.ion.vth[iz,ir,is]) +
                    collisions.ionization *
                    (0.5*vpa.grid
                       * (fvec.density_neutral[iz,ir,is]
                          - fvec.density[iz,ir,is]*fvec.pz_neutral[iz,ir,is]
                            / fvec.ppar[iz,ir,is]
                          - fvec.density[iz,ir,is]*fvec.density_neutral[iz,ir,is]
                            * (fvec.uz_neutral[iz,ir,is] - fvec.upar[iz,ir,is])^2
                            / fvec.ppar[iz,ir,is])
                     - fvec.density_neutral[iz,ir,is]
                       * (fvec.uz_neutral[iz,ir,is] - fvec.upar[iz,ir,is])
                       / moments.ion.vth[iz,ir,is])
            end
        end
    end
    if ion_source_settings.active
        source_density_amplitude = moments.ion.external_source_density_amplitude
        source_momentum_amplitude = moments.ion.external_source_momentum_amplitude
        source_pressure_amplitude = moments.ion.external_source_pressure_amplitude
        density = fvec.density
        upar = fvec.upar
        ppar = fvec.ppar
        vth = moments.ion.vth
        vpa_grid = vpa.grid
        @loop_s_r_z is ir iz begin
            term1 = source_density_amplitude[iz,ir] * upar[iz,ir,is]/(density[iz,ir,is]*vth[iz,ir,is])
            term2_over_vpa =
                -0.5 * (source_pressure_amplitude[iz,ir] +
                        2.0 * upar[iz,ir,is] * source_momentum_amplitude[iz,ir]) /
                       ppar[iz,ir,is] +
                0.5 * source_density_amplitude[iz,ir] / density[iz,ir,is]
            @loop_vperp_vpa ivperp ivpa begin
                advect[is].speed[ivpa,ivperp,iz,ir] += term1 + vpa_grid[ivpa] * term2_over_vpa
            end
        end
    end
end

"""
update the advection speed in the parallel velocity coordinate for the case
where density and pressure are evolved independently from the pdf;
in this case, the parallel velocity coordinate is the normalized velocity
vpahat = vpa/vth
"""
function update_speed_n_p_evolution!(advect, fields, fvec, moments, vpa, z, r,
                                     composition, collisions, ion_source_settings)
    @loop_s is begin
        # include contributions common to both ion and neutral species
        @loop_r ir begin
            # update parallel acceleration to account for:
            # • (vpahat/2*ppar)*dqpar/dz
            # • vpahat*(upar/vth-vpahat) * d(vth)/dz term
            # • vpahat*d(upar)/dz
            # • -(1/2)*(dphi/dz)/vthi
            @loop_z_vperp iz ivperp begin
                @views @. advect[is].speed[:,ivperp,iz,ir] = 0.5*vpa.grid*moments.ion.dqpar_dz[iz,ir,is]/fvec.ppar[iz,ir,is] +
                                                             vpa.grid*moments.ion.dvth_dz[iz] * (fvec.upar[iz,ir,is]/moments.vth[iz,ir,is] - vpa.grid) +
                                                             vpa.grid*moments.ion.dupar_dz[iz,ir,is] +
                                                             0.5*fields.Ez[iz,ir]/moments.vth[iz,ir,is]
            end
        end
    end
    # add in contributions from charge exchange and ionization collisions
    if composition.n_neutral_species > 0
        error("suspect the charge exchange and ionization contributions here may be "
              * "wrong because (upar[is]-upar[isp])^2 type terms were missed in the "
              * "energy equation when it was substituted in to derive them.")
        if abs(collisions.charge_exchange + collisions.ionization) > 0.0
            @loop_s is begin
                @loop_r_z_vperp ir iz ivperp begin
                    @views @. advect[is].speed[:,ivperp,iz,ir] += (collisions.charge_exchange + collisions.ionization) *
                            0.5*vpa.grid*fvec.density[iz,ir,is] * (1.0-fvec.pz_neutral[iz,ir,is]/fvec.ppar[iz,ir,is])
                end
            end
        end
    end
    if ion_source_settings.active
        error("External source not implemented for evolving n and ppar case")
    end
end

"""
update the advection speed in the parallel velocity coordinate for the case
where density and flow are evolved independently from the pdf;
in this case, the parallel velocity coordinate is the peculiar velocity
wpa = vpa-upar
"""
function update_speed_n_u_evolution!(advect, fvec, moments, vpa, z, r, composition,
                                     collisions, ion_source_settings)
    @loop_s is begin
        @loop_r ir begin
            # update parallel acceleration to account for:
            # • parallel derivative of parallel pressure
            # • -wpar*dupar/dz
            @loop_z_vperp iz ivperp begin
                @views @. advect[is].speed[:,ivperp,iz,ir] =
                    moments.ion.dppar_dz[iz,ir,is]/fvec.density[iz,ir,is] -
                    vpa.grid*moments.ion.dupar_dz[iz,ir,is]
            end
        end
    end
    # if neutrals present compute contribution to parallel acceleration due to charge exchange
    # and/or ionization collisions betweens ions and neutrals
    if composition.n_neutral_species > 0
        # account for collisional charge exchange friction between ions and neutrals
        if abs(collisions.charge_exchange) > 0.0
            @loop_s is begin
                @loop_r_z_vperp ir iz ivperp begin
                    @views @. advect[is].speed[:,ivperp,iz,ir] -= collisions.charge_exchange*fvec.density_neutral[iz,ir,is]*(fvec.uz_neutral[iz,ir,is]-fvec.upar[iz,ir,is])
                end
            end
        end
        if abs(collisions.ionization) > 0.0
            @loop_s is begin
                @loop_r_z_vperp ir iz ivperp begin
                    @views @. advect[is].speed[:,ivperp,iz,ir] -= collisions.ionization*fvec.density_neutral[iz,ir,is]*(fvec.uz_neutral[iz,ir,is]-fvec.upar[iz,ir,is])
                end
            end
        end
    end
    if ion_source_settings.active
        source_density_amplitude = moments.ion.external_source_density_amplitude
        source_strength = ion_source_settings.source_strength
        r_amplitude = ion_source_settings.r_amplitude
        z_amplitude = ion_source_settings.z_amplitude
        density = fvec.density
        upar = fvec.upar
        vth = moments.ion.vth
        @loop_s_r_z is ir iz begin
            term = source_density_amplitude[iz,ir] * upar[iz,ir,is] / density[iz,ir,is]
            @loop_vperp_vpa ivperp ivpa begin
                advect[is].speed[ivpa,ivperp,iz,ir] += term
            end
        end
    end
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
