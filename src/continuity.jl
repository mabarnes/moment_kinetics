"""
"""
module continuity

export continuity_equation!

using ..calculus: derivative!
using ..looping

"""
use the continuity equation dn/dt + d(n*upar)/dz to update the density n for all species
"""
function continuity_equation!(dens_out, fvec_in, moments, composition, vpa, z, r, dt,
                              spectral, ionization, num_diss_params)
    # use the continuity equation dn/dt + d(n*upar)/dz to update the density n
    # for each species

    begin_s_r_region()

    @loop_s is begin
        @loop_r ir begin #MRH NOT SURE ABOUT THIS!
            @views continuity_equation_single_species!(dens_out[:,ir,is],
                fvec_in.density[:,ir,:], fvec_in.upar[:,ir,is], z, dt, spectral, ionization, composition, is, num_diss_params)
        end
    end
end

"""
use the continuity equation dn/dt + d(n*upar)/dz to update the density n
"""
function continuity_equation_single_species!(dens_out, dens_in, upar, z, dt, spectral,
                                             ionization, composition, is,
                                             num_diss_params)
    ## calculate the particle flux nu
    #@. z.scratch = dens_in[:,is]*upar
    ## Use as 'adv_fac' for upwinding
    #@. z.scratch3 = -upar
    ## calculate d(nu)/dz, averaging the derivative values at element boundaries
    #derivative!(z.scratch, z.scratch, z, z.scratch3, spectral)
    ##derivative!(z.scratch, z.scratch, z, -upar, spectral)
    ## update the density to account for the divergence of the particle flux
    #@. dens_out -= dt*z.scratch

    # Use as 'adv_fac' for upwinding
    @. z.scratch3 = -upar
    @views derivative!(z.scratch, dens_in[:,is], z, z.scratch3, spectral)
    derivative!(z.scratch2, upar, z, spectral)
    @. dens_out -= dt*(upar*z.scratch + dens_in[:,is]*z.scratch2)

    # update the density to account for ionization collisions;
    # ionization collisions increase the density for ions and decrease the density for neutrals
    if is ∈ composition.ion_species_range
        # NB: could improve efficiency here by calculating total neutral density
        for isn ∈ composition.neutral_species_range
            @. dens_out += dt*ionization*dens_in[:,is]*dens_in[:,isn]
        end
    elseif is ∈ composition.neutral_species_range
        # NB: could improve efficiency here by calculating total ion density
        for isi ∈ composition.ion_species_range
            @. dens_out -= dt*ionization*dens_in[:,is]*dens_in[:,isi]
        end
    end

    # Ad-hoc diffusion to stabilise numerics...
    diffusion_coefficient = num_diss_params.moment_dissipation_coefficient
    if diffusion_coefficient > 0.0
        derivative!(z.scratch, dens_in[:,is], z, spectral, Val(2))
        @. dens_out += dt*diffusion_coefficient*z.scratch
    end

    return nothing
end

end
