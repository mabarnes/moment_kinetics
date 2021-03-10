module source_terms

export source_terms!

using velocity_moments: update_moments!
using advection: set_igrid_ielem
using chebyshev: chebyshev_info, chebyshev_derivative!
using finite_differences: derivative_finite_difference!

function source_terms!(pdf_out, fvec_in, moments, z, vpa, dt, advection, spectral)
    # update the parallel flow velocity upar
    update_moments!(moments, fvec_in.pdf, vpa, z.n)
    # calculate the source terms due to redefinition of the pdf to split off density
    # and use them to update the pdf
    n_species = size(pdf_out,3)
    for is ∈ 1:n_species
        for ivpa ∈ 1:vpa.n
            @views source_terms_single_species!(pdf_out[:,ivpa,is],
                fvec_in.pdf[:,ivpa,is], fvec_in.density[:,is], moments.upar[:,is],
                z, vpa.grid[ivpa], dt, advection[ivpa,is], spectral)
        end
    end
    return nothing
end
function source_terms_single_species!(pdf_out, pdf_in, dens, upar, z, vpa, dt, advection, spectral)
    # calculate dupar/dz
    derivative!(z.scratch2d, upar, z, advection.adv_fac, spectral)
    for iz ∈ 1:z.n
        igrid, ielem = set_igrid_ielem(z.igrid[iz], z.ielement[iz], advection.adv_fac[iz], z.ngrid, z.nelement)
        pdf_out[iz] += dt*z.scratch2d[igrid,ielem]*pdf_in[iz]
    end
    # calculate ∂ln(n)/∂z
    derivative!(z.scratch2d, log.(dens), z, advection.adv_fac, spectral)
    for iz ∈ 1:z.n
        igrid, ielem = set_igrid_ielem(z.igrid[iz], z.ielement[iz], advection.adv_fac[iz], z.ngrid, z.nelement)
        pdf_out[iz] -= dt*z.scratch2d[igrid,ielem]*(vpa - upar[iz])*pdf_in[iz]
    end
    return nothing
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
