"""
"""
module force_balance

export force_balance!

using ..calculus: derivative!
using ..looping

"""
use the force balance equation d(nu)/dt + d(ppar + n*upar*upar)/dz =
-(dens/2)*dphi/dz + R*dens_i*dens_n*(upar_n-upar_i)
to update the parallel particle flux dens*upar for each species
"""
function force_balance!(pflx, density, fvec, fields, collisions, vpa, z, r, dt,
                        spectral, composition, num_diss_params)

    begin_s_r_region()

    # account for momentum flux contribution to force balance
    @loop_s is begin
        @loop_r ir begin
            @views force_balance_flux_species!(pflx[:,ir,is], fvec.density[:,ir,is],
                                               fvec.upar[:,ir,is], fvec.ppar[:,ir,is],
                                               z, dt, spectral, num_diss_params)
            if is ∈ composition.ion_species_range
                # account for parallel electric field contribution to force balance
                @views force_balance_Epar_species!(pflx[:,ir,is], fields.phi[:,ir], fvec.density[:,ir,is], z, dt, spectral)
            end
        end
    end
    # if neutrals present account for charge exchange and/or ionization collisions
    if composition.n_neutral_species > 0
        # account for collisional friction between ions and neutrals
        if abs(collisions.charge_exchange) > 0.0
            force_balance_CX!(pflx, fvec.density, fvec.upar, collisions.charge_exchange, composition, z, r, dt)
        end
        # account for ionization collisions
        if abs(collisions.ionization) > 0.0
            force_balance_ionization!(pflx, fvec.density, fvec.upar, collisions.ionization,
                                      composition, z.n, dt)
        end
    end

    @loop_s_r_z is ir iz begin
        # convert from the particle flux to the parallel flow
        pflx[iz,ir,is] /= density[iz,ir,is]
    end
end

"""
use the force balance equation d(mnu)/dt + d(ppar + mnu * u)/dz = ...
to update the momentum flux mnu; this function accounts for the contribution from the
flux term above
"""
function force_balance_flux_species!(pflx, dens, upar, ppar, z, dt, spectral,
                                     num_diss_params)
    # calculate the parallel flux of parallel momentum densitg at the previous time level/RK stage
    derivative!(z.scratch, ppar, z, spectral)

    ##@. z.scratch2 = dens*upar^2
    ## Until julia-1.8 is released, prefer x*x to x^2 to avoid extra allocations when broadcasting.
    #@. z.scratch2 = dens*upar*upar
    ## Use as 'adv_fac' for upwinding
    #@. z.scratch3 = -upar
    #derivative!(z.scratch2, z.scratch2, z, z.scratch3, spectral)

    #@. z.scratch2 = dens*upar
    ## Use as 'adv_fac' for upwinding
    #@. z.scratch3 = -upar
    #derivative!(z.scratch2, z.scratch2, z, z.scratch3, spectral)
    #derivative!(z.scratch3, upar, z, spectral)

    # Use as 'adv_fac' for upwinding
    @. z.scratch3 = -upar
    derivative!(z.scratch2, dens, z, z.scratch3, spectral)

    derivative!(z.scratch3, upar, z, z.scratch3, spectral)

    # update the parallel momentum density to account for the parallel flux of parallel momentum
    #@. pflx = dens*upar - dt*(z.scratch + z.scratch2)
    #@. pflx = dens*upar - dt*(z.scratch + upar*z.scratch2 + dens.*upar*z.scratch3)
    @. pflx = dens*upar - dt*(z.scratch + upar*upar*z.scratch2 + 2.0*dens*upar*z.scratch3)

    # Ad-hoc diffusion to stabilise numerics...
    diffusion_coefficient = num_diss_params.moment_dissipation_coefficient
    if diffusion_coefficient > 0.0
        derivative!(z.scratch, upar, z, spectral, Val(2))
        @. pflx += dt*diffusion_coefficient*z.scratch*dens
    end

    return nothing
end

"""
use the force balance equation d(mnu)/dt + ... = -n*Epar + ...
to update mnu; this function accounts for the contribution from the Epar term
"""
function force_balance_Epar_species!(pflx, phi, dens, z, dt, spectral)
    # calculate the (negative of the) parallel electric field.
    # Done like this because passing in -phi would require a temporary buffer to be allocated.
    # So z.scratch = -Epar
    derivative!(z.scratch, phi, z, spectral)
    # update the parallel momentum density to account for the force from the parallel electric field
    #  pflx += 0.5*dt*Epar*dens
    #  => pflx -= 0.5*dt*(-Epar)*dens
    @. pflx -= 0.5*dt*z.scratch*dens
end

"""
add the contribution to the evolution of the particle flux arising from charge exchange collisions
"""
function force_balance_CX!(pflx, dens, upar, CX_frequency, composition, z, r, dt)
    @loop_s is begin
        @loop_r ir begin
            # include contribution to ion acceleration due to collisional friction with neutrals
            if is ∈ composition.ion_species_range
                for isp ∈ composition.neutral_species_range
                    @views @. pflx[:,ir,is] += dt*CX_frequency*dens[:,ir,is]*dens[:,ir,isp]*(upar[:,ir,isp]-upar[:,ir,is])
                end
            end
            # include contribution to neutral acceleration due to collisional friction with ions
            if is ∈ composition.neutral_species_range
                for isp ∈ composition.ion_species_range
                    @views @. pflx[:,ir,is] += dt*CX_frequency*dens[:,ir,is]*dens[:,ir,isp]*(upar[:,ir,isp]-upar[:,ir,is])
                end
            end
        end
    end
end

"""
add the contribution to the evolution of the particle flux arising from ionization collisions
"""
function force_balance_ionization!(pflx, dens, upar, ionization_frequency, composition, nz, dt)
    @loop_s is begin
        @loop_r ir begin
            # include contribution to ion acceleration due to ionization of neutrals
            if is ∈ composition.ion_species_range
                for isp ∈ composition.neutral_species_range
                    @views @. pflx[:,ir,is] += dt*ionization_frequency*dens[:,ir,is]*dens[:,ir,isp]*upar[:,ir,isp]
                end
            end
            # include contribution to neutral acceleration due to ionizaton
            if is ∈ composition.neutral_species_range
                for isp ∈ composition.ion_species_range
                    @views @. pflx[:,ir,is] -= dt*ionization_frequency*dens[:,ir,isp]*dens[:,ir,is]*upar[:,ir,is]
                end
            end
        end
    end
end

end
