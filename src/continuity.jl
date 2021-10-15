module continuity

export continuity_equation!

using ..calculus: derivative!

# use the continuity equation dn/dt + d(n*upar)/dz to update the density n for all species
function continuity_equation!(dens_out, fvec_in, moments, vpa, z, dt, composition,
                              ionization_frequency, spectral)
    # use the continuity equation dn/dt + d(n*upar)/dz to update the density n
    # for each species
    n_species = size(dens_out,2)
    for is ∈ 1:n_species
        @views continuity_equation_single_species!(dens_out[:,is],
            fvec_in.density[:,is], fvec_in.upar[:,is], z, dt, spectral)
    end
    # if neutrals are present and ionization collisions are included,
    # account for their contribution to the change in particle density
    if composition.n_neutral_species > 0 && abs(ionization_frequency) > 0.0
        continuity_equation_ionization!(dens_out, fvec_in.density,
                                       ionization_frequency, composition, dt)
    end
end
# use the continuity equation dn/dt + d(n*upar)/dz to update the density n
function continuity_equation_single_species!(dens_out, dens_in, upar, z, dt, spectral)
    # calculate the particle flux nu
    @. z.scratch = dens_in*upar
    # calculate d(nu)/dz, averaging the derivative values at element boundaries
    derivative!(z.scratch, z.scratch, z, spectral)
    #derivative!(z.scratch, z.scratch, z, -upar, spectral)
    # update the density
    @. dens_out -= dt*z.scratch
end
# account for the change in particle density arising from ionization collisions
function continuity_equation_ionization!(dens_out, dens_in, ionization_frequency,
                                         composition, dt)
    # account for increase in ion density due to ionization
    for isi ∈ 1:composition.n_ion_species
        for is ∈ 1:composition.n_neutral_species
            isn = is + composition.n_ion_species
            @. dens_out[:,isi] += dt*ionization_frequency*dens_in[:,isi]*dens_in[:,isn]
        end
    end
    # account for decrease in neutral density due to collisons
    for is ∈ 1:composition.n_neutral_species
        isn = is + composition.n_ion_species
        for isi ∈ 1:composition.n_ion_species
            @. dens_out[:,isn] -= dt*ionization_frequency*dens_in[:,isi]*dens_in[:,isn]
        end
    end
end

end
