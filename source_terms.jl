module source_terms

export source_terms!

using velocity_moments: update_moments!
using calculus: derivative!

function source_terms!(pdf_out, fvec_in, moments, z, vpa, dt, spectral)
    # update the parallel flow velocity upar
    update_moments!(moments, fvec_in.pdf, vpa, z.n)
    # calculate the source terms due to redefinition of the pdf to split off density,
    # and use them to update the pdf
    n_species = size(pdf_out,3)
    for is ∈ 1:n_species
        for ivpa ∈ 1:vpa.n
            @views source_terms_single_species!(pdf_out[:,ivpa,is], fvec_in.pdf[:,ivpa,is],
                fvec_in.density[:,is], moments.upar[:,is], z, dt, spectral)
        end
    end
    return nothing
end
function source_terms_single_species!(pdf_out, pdf_in, dens, upar, z, dt, spectral)
    # calculate d(n*upar)/dz
    @. z.scratch = dens*upar
    derivative!(z.scratch, z.scratch, z, spectral)
    #derivative!(z.scratch, z.scratch, z, -upar, spectral)
    # update the density
    @. pdf_out += dt*z.scratch*pdf_in/dens
    return nothing
end

end
