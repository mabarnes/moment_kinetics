module continuity

export continuity_equation!

using ..calculus: derivative!
using ..optimization

# use the continuity equation dn/dt + d(n*upar)/dz to update the density n for all species
function continuity_equation!(dens_out, fvec_in, moments, vpa_vec, z_vec, dt, spectral_vec)
    # use the continuity equation dn/dt + d(n*upar)/dz to update the density n
    # for each species
    n_species = size(dens_out,2)
    for is ∈ 1:n_species
        ithread = threadid()
        vpa = vpa_vec[ithread]
        z = z_vec[ithread]
        spectral = spectral_vec[ithread]
        @views continuity_equation_single_species!(dens_out[:,is],
            fvec_in.density[:,is], fvec_in.upar[:,is], z, dt, spectral)
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

end
