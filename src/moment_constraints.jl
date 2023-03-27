"""
Functions for enforcing integral constraints on the normalised distribution function.
Ensures consistency of evolution split into moments and normalised distribution
function.
"""
module moment_constraints

using ..communication: _block_synchronize
using ..initial_conditions: enforce_zero_incoming_bc!
using ..looping
using ..velocity_moments: integrate_over_vspace, update_qpar!

export enforce_moment_constraints!, hard_force_moment_constraints!

"""
    enforce_moment_constraints!(fvec_new, fvec_old, vpa, z, r, composition, moments, dummy_sr)

Force moment constraints to be true for fvec_new if they were true for fvec_old. Should
always be a small correction because the time step is 'small' so fvec_new should only
violate the constraints by a small amount.
"""
function enforce_moment_constraints!(fvec_new, fvec_old, vpa, z, r, composition, moments, dummy_sr)
    # pre-calculate avgdens_ratio so that we don't read fvec_new.density[:,is] on every
    # process in the next loop - that would be an error because different processes
    # write to fvec_new.density[:,is]
    # This loop needs to be @loop_s_r because it fills the (not-shared)
    # dummy_sr buffer to be used within the @loop_s_r below, so the values
    # of is looped over by this process need to be the same.
    # Need to call _block_synchronize() even though loop type does not change because
    # all spatial ranks read fvec_new.density, but it will be written below.
    if any(moments.particle_number_conserved)
        _block_synchronize()
    end
    @loop_s_r is ir begin
        if moments.particle_number_conserved[is]
            @views @. z.scratch = fvec_old.density[:,ir,is] - fvec_new.density[:,ir,is]
            @views dummy_sr[ir,is] = integral(z.scratch, z.wgts)/integral(fvec_old.density[:,ir,is], z.wgts)
        end
    end
    # Need to call _block_synchronize() even though loop type does not change because
    # all spatial ranks read fvec_new.density, but it will be written below.
    if any(moments.particle_number_conserved)
        _block_synchronize()
    end

    @loop_s is begin
        # add a small correction to the density for each species to ensure that
        # that particle number is conserved if it should be;
        # ionisation collisions and net particle flux out of the domain due to, e.g.,
        # a wall BC break particle conservation, in which cases it should not be enforced.
        if moments.particle_number_conserved[is]
            @loop_r ir begin
                avgdens_ratio = dummy_sr[ir,is]
                @loop_z iz begin
                    # update the density with the above factor to ensure particle conservation
                    fvec_new.density[iz,ir,is] += fvec_old.density[iz,ir,is] * avgdens_ratio
                    # update the thermal speed, as the density has changed
                    moments.vth[iz,ir,is] = sqrt(2.0*fvec_new.ppar[iz,ir,is]/fvec_new.density[iz,ir,is])
                end
            end
        end
        @loop_r ir begin
            if moments.evolve_upar && is ∈ composition.ion_species_range && z.bc == "wall"
                # Enforce zero-incoming boundary condition on the old distribution
                # function with the new parallel flow, then force the updated old
                # distribution function to obey the integral constraints exactly, which
                # should be a small correction here as the boundary condition should
                # only modify a few points due to the small change in upar.
                # This procedure ensures that fvec_old obeys both the new boundary
                # conditions and the moment constraints, so that when it is used to
                # update f_new, f_new also does.
                # Note fvec_old is never used after this function, so it is OK to modify
                # it in-place.

                # define a zero that accounts for finite precision
                zero = 1.0e-10

                @views enforce_zero_incoming_bc!(
                    fvec_old.pdf[:,:,ir,is], vpa, fvec_new.density[:,ir,is],
                    fvec_new.upar[:,ir,is], fvec_new.ppar[:,ir,is], moments.evolve_upar,
                    moments.evolve_ppar, zero)
                # Correct fvec_old.pdf in case applying new bc messed up moment
                # constraints
                @views hard_force_moment_constraints!(fvec_old.pdf[:,1,ir,is], moments,
                                                      vpa)
                @views hard_force_moment_constraints!(fvec_old.pdf[:,end,ir,is],
                                                      moments, vpa)
            end
            @loop_z iz begin
                # Create views once to save overhead
                fnew_view = @view(fvec_new.pdf[:,iz,ir,is])
                fold_view = @view(fvec_old.pdf[:,iz,ir,is])

                # first calculate all of the integrals involving the updated pdf fvec_new.pdf
                density_integral = integrate_over_vspace(fnew_view, vpa.wgts)
                if moments.evolve_upar
                    upar_integral = integrate_over_vspace(fnew_view, vpa.grid, vpa.wgts)
                end
                if moments.evolve_ppar
                    ppar_integral = integrate_over_vspace(fnew_view, vpa.grid, 2, vpa.wgts)
                end
                # update the pdf to account for the density-conserving correction
                @. fnew_view += fold_view * (1.0 - density_integral)
                if moments.evolve_upar
                    if !moments.evolve_ppar
                        upar_coefficient = upar_integral /
                            integrate_over_vspace(fold_view, vpa.grid, 2, vpa.wgts)
                    else
                        vpa3_moment = integrate_over_vspace(fold_view, vpa.grid, 3, vpa.wgts)
                        vpa4_moment = integrate_over_vspace(fold_view, vpa.grid, 4, vpa.wgts)
                        ppar_coefficient = (-ppar_integral + 0.5*density_integral +
                                            2.0*upar_integral*vpa3_moment) /
                                            (vpa4_moment - 0.25 - 2.0*vpa3_moment*vpa3_moment)
                        upar_coefficient = 2.0 * (upar_integral + ppar_coefficient * vpa3_moment)
                    end
                    # update the pdf to account for the momentum-conserving correction
                    @. fnew_view -= upar_coefficient * vpa.grid * fold_view
                    if moments.evolve_ppar
                        # update the pdf to account for the energy-conserving correction
                        #@. fnew_view += ppar_coefficient * (vpa.grid^2 - 0.5) * fold_view
                        # Until julia-1.8 is released, prefer x*x to x^2 to avoid
                        # extra allocations when broadcasting.
                        @. fnew_view += ppar_coefficient * (vpa.grid * vpa.grid - 0.5) * fold_view
                    end
                end
            end
        end
    end
    # the pdf, density and thermal speed have been changed so the corresponding parallel heat flux must be updated
    moments.qpar_updated .= false
    update_qpar!(moments.qpar, moments.qpar_updated, fvec_new.density, fvec_new.upar,
                 moments.vth, fvec_new.pdf, vpa, z, r, composition,
                 moments.evolve_density, moments.evolve_upar, moments.evolve_ppar)
end

"""
    hard_force_moment_constraints!(f, moments, vpa)

Force the moment constraints needed for the system being evolved to be applied to `f`.
Not guaranteed to be a small correction, if `f` does not approximately obey the
constraints to start with, but can be useful at initialisation to ensure a consistent
initial state, and when applying boundary conditions.

Note this function assumes the input is given at a single spatial position.
"""
function hard_force_moment_constraints!(f, moments, vpa)
    #if moments.evolve_ppar
    #    I0 = integrate_over_vspace(f, vpa.wgts)
    #    I1 = integrate_over_vspace(f, vpa.grid, vpa.wgts)
    #    I2 = integrate_over_vspace(f, vpa.grid, 2, vpa.wgts)
    #    I3 = integrate_over_vspace(f, vpa.grid, 3, vpa.wgts)
    #    I4 = integrate_over_vspace(f, vpa.grid, 4, vpa.wgts)
    #    A = ((1.0 - 0.5*I2/I4)*(I2 - I3^2/I4) + 0.5*I3/I4*(I1-I2*I3/I4)) /
    #        ((I0 - I2^2/I4)*(I2 - I3^2/I4) - (I1 - I2*I3/I4)^2)
    #    B = -(0.5*I3/I4 + A*(I1 - I2*I3/I4)) / (I2 - I3^2/I4)
    #    C = -(A*I1 + B*I2) / I3
    #    @. f = A*f + B*vpa.grid*f + C*vpa.grid*vpa.grid*f
    #elseif moments.evolve_upar
    #    I0 = integrate_over_vspace(f, vpa.wgts)
    #    I1 = integrate_over_vspace(f, vpa.grid, vpa.wgts)
    #    I2 = integrate_over_vspace(f, vpa.grid, 2, vpa.wgts)
    #    A = 1.0 / (I0 + I1*I1/I2)
    #    B = -I1*A/I2
    #    @. f = A*f + B*vpa.grid*f
    #elseif moments.evolve_density
    #    I0 = integrate_over_vspace(f, vpa.wgts)
    #    @. f = f / I0
    #end

    if moments.evolve_ppar
        I0 = integrate_over_vspace(f, vpa.wgts)
        I1 = integrate_over_vspace(f, vpa.grid, vpa.wgts)
        I2 = integrate_over_vspace(f, vpa.grid, 2, vpa.wgts)
        I3 = integrate_over_vspace(f, vpa.grid, 3, vpa.wgts)
        I4 = integrate_over_vspace(f, vpa.grid, 4, vpa.wgts)

        A = (I3^2 - I2*I4 + 0.5*(I2^2 - I1*I3)) /
            (I0*(I3^2 - I2*I4) + I1*I1*I4 - 2.0*I1*I2*I3 + I2^3)
        B = (0.5*I3 + A*(I1*I4 - I2*I3)) / (I3^2 - I2*I4)
        C = (0.5 - A*I2 -B*I3) / I4

        @. f = A*f + B*vpa.grid*f + C*vpa.grid*vpa.grid*f
    elseif moments.evolve_upar
        I0 = integrate_over_vspace(f, vpa.wgts)
        I1 = integrate_over_vspace(f, vpa.grid, vpa.wgts)
        I2 = integrate_over_vspace(f, vpa.grid, 2, vpa.wgts)

        A = 1.0 / (I0 - I1^2/I2)
        B = -A*I1/I2

        @. f = A*f + B*vpa.grid*f
    elseif moments.evolve_density
        I0 = integrate_over_vspace(f, vpa.wgts)
        @. f = f / I0
    end

    return nothing
end

end
