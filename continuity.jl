module continuity

export continuity_equation!

using chebyshev: chebyshev_derivative!, chebyshev_info
using finite_differences: derivative_finite_difference!
using velocity_moments: update_moments!
using advection: set_igrid_ielem

# use the continuity equation dn/dt + d(n*upar)/dz to update the density n for all species
function continuity_equation!(dens_out, fvec_in, moments, z, vpa, dt, spectral)
    # update the parallel flow velocity upar
    update_moments!(moments, fvec_in.pdf, vpa, z.n)
    # use the continuity equation dn/dt + d(n*upar)/dz to update the density n
    # for each species
    n_species = size(dens_out,2)
    for is ∈ 1:n_species
        @views continuity_equation_single_species!(dens_out[:,is],
            fvec_in.density[:,is], moments.upar[:,is], z, dt, spectral)
    end
end
# use the continuity equation dn/dt + d(n*upar)/dz to update the density n
function continuity_equation_single_species!(dens_out, dens_in, upar, z, dt, spectral)
    @. z.scratch = dens_in*upar
    # calculate d(nu)/dz
    derivative!(z.scratch2d, z.scratch, z, -upar, spectral)
    for iz ∈ 1:z.n
        igrid, ielem = set_igrid_ielem(z.igrid[iz], z.ielement[iz], -upar[iz], z.ngrid, z.nelement)
        dens_out[iz] -= dt*z.scratch2d[igrid,ielem]
    end
end
# Chebyshev transform f to get Chebyshev spectral coefficients and use them to calculate f'
function derivative!(df, f, coord, adv_fac, spectral::chebyshev_info)
    chebyshev_derivative!(df, f, spectral, coord)
end
# calculate the derivative of f using finite differences; stored in df
function derivative!(df, f, coord, adv_fac, not_spectral::Bool)
    derivative_finite_difference!(df, f, coord.cell_width, adv_fac,
        coord.bc, coord.fd_option, coord.igrid, coord.ielement)
end

end
