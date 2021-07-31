module source_terms

export source_terms!

using ..calculus: derivative!
using ..optimization

function source_terms!(pdf_out, fvec_in, moments, vpa, z, dt, spectral, composition, CX_frequency)
    # calculate the source terms due to redefinition of the pdf to split off density,
    # and use them to update the pdf
    #n_species = size(pdf_out,3)
    if moments.evolve_ppar
        for is ∈ 1:composition.n_species
            @views source_terms_evolve_ppar!(pdf_out[:,:,is], fvec_in.pdf[:,:,is],
                                             fvec_in.density[:,is], fvec_in.upar[:,is], fvec_in.ppar[:,is],
                                             moments.vth[:,is], moments.qpar[:,is], z, dt, spectral)
        end
        if composition.n_neutral_species > 0 && abs(CX_frequency) > 0.0
            @views source_terms_evolve_ppar_CX!(pdf_out[:,:,:], fvec_in.pdf[:,:,:],
                                                fvec_in.density, fvec_in.ppar, composition,
                                                CX_frequency, dt)
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
    ithread = Base.Threads.threadid()
    scratch = @view(z.scratch[:,ithread])
    # calculate d(n*upar)/dz
    @. scratch = dens*upar
    derivative!(scratch, scratch, z, spectral)
    #derivative!(scratch, scratch, z, -upar, spectral)
    @. scratch *= dt/dens
    # update the density
    for i in CartesianIndices(pdf_out)
        pdf_out[i] += pdf_in[i]*scratch[i[2]]
    end
    return nothing
end
function source_terms_evolve_ppar!(pdf_out, pdf_in, dens, upar, ppar, vth, qpar, z, dt, spectral)
    ithread = Base.Threads.threadid()
    scratch = @view(z.scratch[:,ithread])
    scratch2 = @view(z.scratch2[:,ithread])
    # calculate dn/dz
    derivative!(scratch, dens, z, spectral)
    # update the pdf to account for the density gradient contribution to the source
    @. scratch *= dt*upar/dens
    # calculate dvth/dz
    derivative!(scratch2, vth, z, spectral)
    # update the pdf to account for the -g*upar/vth * dvth/dz contribution to the source
    @. scratch -= dt*scratch2*upar/vth
    # calculate dqpar/dz
    derivative!(scratch2, qpar, z, spectral)
    # update the pdf to account for the parallel heat flux contribution to the source
    @. scratch -= 0.5*dt*scratch2/ppar

    for i ∈ CartesianIndices(pdf_out)
        pdf_out[i] += pdf_in[i] * scratch[i[2]]
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

end
