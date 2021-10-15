module source_terms

export source_terms!

using ..calculus: derivative!

function source_terms!(pdf_out, fvec_in, moments, vpa, z, dt, spectral, composition,
                      CX_frequency, ionization_frequency)
    # calculate the source terms due to redefinition of the pdf to split off density,
    # and use them to update the pdf
    #n_species = size(pdf_out,3)
    if moments.evolve_ppar
        for is ∈ 1:composition.n_species
            @views source_terms_evolve_ppar!(pdf_out[:,:,is], fvec_in.pdf[:,:,is],
                                             fvec_in.density[:,is], fvec_in.upar[:,is], fvec_in.ppar[:,is],
                                             moments.vth[:,is], moments.qpar[:,is], z, dt, spectral)
        end
        # obtain the source contributions due to collisions with neutrals
        if composition.n_neutral_species > 0
            # obtain the source contribution due to charge exchange collisions
            if abs(CX_frequency) > 0.0
                source_terms_evolve_ppar_CX!(pdf_out, fvec_in.pdf,
                                            fvec_in.density, fvec_in.ppar, composition,
                                            CX_frequency, dt)
            end
            if abs(ionization_frequency) > 0.0
                @views source_terms_evolve_ppar_ionization!(pdf_out, fvec_in.pdf, fvec_in.density,
                                                            fvec_in.ppar, composition, ionization_frequency, dt, z.n)
            end
        end
    elseif moments.evolve_density
        for is ∈ 1:composition.n_species
            @views source_terms_evolve_density!(pdf_out[:,:,is], fvec_in.pdf[:,:,is],
                                                fvec_in.density[:,is], fvec_in.upar[:,is], z, dt, spectral)
        end
    end
    return nothing
end
function source_terms_evolve_density!(pdf_out, pdf_in, dens, upar, z, dt, spectral)
    # calculate d(n*upar)/dz
    @. z.scratch = dens*upar
    derivative!(z.scratch, z.scratch, z, spectral)
    @. z.scratch *= dt/dens
    #derivative!(z.scratch, z.scratch, z, -upar, spectral)
    # update the density
    nvpa, nz = size(pdf_out)
    for iz ∈ 1:nz, ivpa ∈ 1:nvpa
        pdf_out[ivpa,iz] += pdf_in[ivpa,iz]*z.scratch[iz]
    end
    return nothing
end
function source_terms_evolve_ppar!(pdf_out, pdf_in, dens, upar, ppar, vth, qpar, z, dt, spectral)
    # calculate dn/dz
    derivative!(z.scratch, dens, z, spectral)
    # update the pdf to account for the density gradient contribution to the source
    @. z.scratch *= dt*upar/dens
    # calculate dvth/dz
    derivative!(z.scratch2, vth, z, spectral)
    # update the pdf to account for the -g*upar/vth * dvth/dz contribution to the source
    @. z.scratch -= dt*z.scratch2*upar/vth
    # calculate dqpar/dz
    derivative!(z.scratch2, qpar, z, spectral)
    # update the pdf to account for the parallel heat flux contribution to the source
    @. z.scratch -= 0.5*dt*z.scratch2/ppar

    nvpa, nz = size(pdf_out)
    for iz ∈ 1:nz, ivpa ∈ 1:nvpa
        pdf_out[ivpa,iz] += pdf_in[ivpa,iz]*z.scratch[iz]
    end
    return nothing
end
function source_terms_evolve_ppar_CX!(pdf_out, pdf_in, dens, ppar, composition, CX_frequency, dt)
    for is ∈ 1:composition.n_ion_species
        for isn ∈ 1:composition.n_neutral_species
            isp = composition.n_ion_species + isn
            for iz in 1:size(ppar)[1]
                @views @. pdf_out[:,iz,is] -= 0.5*dt*pdf_in[:,iz,is]*CX_frequency *
                    (dens[iz,isp]*ppar[iz,is]-dens[iz,is]*ppar[iz,isp])/ppar[iz,is]
            end
        end
    end
    for isn ∈ 1:composition.n_neutral_species
        is = composition.n_ion_species + isn
        for isp ∈ 1:composition.n_ion_species
            for iz in 1:size(ppar)[1]
                @views @. pdf_out[:,iz,is] -= 0.5*dt*pdf_in[:,iz,is]*CX_frequency *
                    (dens[iz,isp]*ppar[iz,is]-dens[iz,is]*ppar[iz,isp])/ppar[iz,is]
            end
        end
    end
    return nothing
end
function source_terms_evolve_ppar_ionization!(pdf_out, pdf_in, dens, ppar, composition, ionization_frequency, dt, nz)
    for isi ∈ 1:composition.n_ion_species
        for is ∈ 1:composition.n_neutral_species
            isn = composition.n_ion_species + is
            for iz in 1:nz
                @views @. pdf_out[:,iz,isi] += 0.5*dt*pdf_in[:,iz,isi]*ionization_frequency *
                    dens[iz,isn]*(3.0 - ppar[iz,isn]*dens[iz,isi]/(ppar[iz,isi]*dens[iz,isn]))
            end
        end
    end
    for is ∈ 1:composition.n_neutral_species
        isn = composition.n_ion_species + is
        for isi ∈ 1:composition.n_ion_species
            for iz in 1:nz
                @views @. pdf_out[:,iz,isn] -= dt*pdf_in[:,iz,isn]*ionization_frequency*dens[iz,isi]
            end
        end
    end
    return nothing
end

end
