"""
"""
module continuity

export continuity_equation!

using ..calculus: derivative!
using ..looping

"""
use the continuity equation dn/dt + d(n*upar)/dz to update the density n for all species
"""
function continuity_equation!(dens_out, fvec_in, moments, composition, vpa, z, r, dt, spectral)
    # use the continuity equation dn/dt + d(n*upar)/dz to update the density n
    # for each species
    @loop_s is begin
        @loop_r ir begin #MRH NOT SURE ABOUT THIS!
            @views continuity_equation_single_species!(dens_out[:,ir,is],
                fvec_in.density[:,ir,is], fvec_in.upar[:,ir,is], z, dt, spectral)
        end
    end
end

"""
use the continuity equation dn/dt + d(n*upar)/dz to update the density n
"""
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
