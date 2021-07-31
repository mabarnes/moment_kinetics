module continuity

export continuity_equation!

using ..calculus: derivative!
using ..optimization

# use the continuity equation dn/dt + d(n*upar)/dz to update the density n for all species
function continuity_equation!(dens_out, fvec_in, moments, vpa, z, dt, spectral)
    # use the continuity equation dn/dt + d(n*upar)/dz to update the density n
    # for each species
    n_species = size(dens_out,2)
    for is ∈ 1:n_species
        @views continuity_equation_single_species!(dens_out[:,is],
            fvec_in.density[:,is], fvec_in.upar[:,is], z, dt, spectral)
    end
end
# use the continuity equation dn/dt + d(n*upar)/dz to update the density n
function continuity_equation_single_species!(dens_out, dens_in, upar, z, dt, spectral)
    ithread = Base.Threads.threadid()
    scratch = @view(z.scratch[:,ithread])
    # calculate the particle flux nu
    @. scratch = dens_in*upar
    # calculate d(nu)/dz, averaging the derivative values at element boundaries
    derivative!(scratch, scratch, z, spectral)
    #derivative!(scratch, scratch, z, -upar, spectral)
    # update the density
    @. dens_out -= dt*scratch
end

end
