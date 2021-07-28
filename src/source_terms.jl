module source_terms

export source_terms!

using ..calculus: derivative!

function source_terms!(pdf_out, fvec_in, moments, vpa, z, dt, spectral, composition, CX_frequency)
    # calculate the source terms due to redefinition of the pdf to split off density,
    # and use them to update the pdf
    #n_species = size(pdf_out,3)
    if moments.evolve_ppar
        for is ∈ 1:composition.n_species
            for ivpa ∈ 1:vpa.n
                @views source_terms_evolve_ppar!(pdf_out[ivpa,:,is], fvec_in.pdf[ivpa,:,is],
                                                 fvec_in.density[:,is], fvec_in.upar[:,is], fvec_in.ppar[:,is],
                                                 moments.vth[:,is], moments.qpar[:,is], z, dt, spectral)
            end
        end
        if composition.n_neutral_species > 0 && abs(CX_frequency) > 0.0
            for ivpa ∈ 1:vpa.n
                @views source_terms_evolve_ppar_CX!(pdf_out[ivpa,:,:], fvec_in.pdf[ivpa,:,:],
                                                    fvec_in.density, fvec_in.ppar, composition,
                                                    CX_frequency, dt)
            end
        end
    elseif moments.evolve_density
        for is ∈ 1:composition.n_species
            for ivpa ∈ 1:vpa.n
                @views source_terms_evolve_density!(pdf_out[ivpa,:,is], fvec_in.pdf[ivpa,:,is],
                                                    fvec_in.density[:,is], fvec_in.upar[:,is], z, dt, spectral)
            end
        end
    end
    return nothing
end
function source_terms_evolve_density!(pdf_out, pdf_in, dens, upar, z, dt, spectral)
    # calculate d(n*upar)/dz
    @. z.scratch = dens*upar
    derivative!(z.scratch, z.scratch, z, spectral)
    #derivative!(z.scratch, z.scratch, z, -upar, spectral)
    # update the density
    @. pdf_out += dt*z.scratch*pdf_in/dens
    return nothing
end
function source_terms_evolve_ppar!(pdf_out, pdf_in, dens, upar, ppar, vth, qpar, z, dt, spectral)
    # calculate dn/dz
    derivative!(z.scratch, dens, z, spectral)
    # update the pdf to account for the density gradient contribution to the source
    @. pdf_out += dt*z.scratch*pdf_in*upar/dens
    # calculate dvth/dz
    derivative!(z.scratch, vth, z, spectral)
    # update the pdf to account for the -g*upar/vth * dvth/dz contribution to the source
    @. pdf_out -= dt*z.scratch*pdf_in*upar/vth
    # calculate dqpar/dz
    derivative!(z.scratch, qpar, z, spectral)
    # update the pdf to account for the parallel heat flux contribution to the source
    @. pdf_out -= 0.5*dt*z.scratch*pdf_in/ppar
    return nothing
end
function source_terms_evolve_ppar_CX!(pdf_out, pdf_in, dens, ppar, composition, CX_frequency, dt)
    for is ∈ 1:composition.n_ion_species
        for isn ∈ 1:composition.n_neutral_species
            isp = composition.n_ion_species + isn
            @views @. pdf_out[:,is] -= 0.5*dt*pdf_in[:,is]*CX_frequency *
                (dens[:,isp]*ppar[:,is]-dens[:,is]*ppar[:,isp])/ppar[:,is]
        end
    end
    for isn ∈ 1:composition.n_neutral_species
        is = composition.n_ion_species + isn
        for isp ∈ 1:composition.n_ion_species
            @views @. pdf_out[:,is] -= 0.5*dt*pdf_in[:,is]*CX_frequency *
                (dens[:,isp]*ppar[:,is]-dens[:,is]*ppar[:,isp])/ppar[:,is]
        end
    end
    return nothing
end

end
