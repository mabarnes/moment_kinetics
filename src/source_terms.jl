"""
"""
module source_terms

export source_terms!
export source_terms_manufactured!

using ..calculus: derivative!
using ..looping

"""
calculate the source terms due to redefinition of the pdf to split off density,
and use them to update the pdf
"""
function source_terms!(pdf_out, fvec_in, moments, vpa, vperp, z, r, dt, spectral, composition, CX_frequency)

    begin_s_r_z_region()
    
    #n_species = size(pdf_out,3)
    if moments.evolve_ppar
        @loop_s is begin
            @views source_terms_evolve_ppar!(pdf_out[:,:,:,:,is], fvec_in.pdf[:,:,:,:,is],
                                             fvec_in.density[:,:,is], fvec_in.upar[:,:,is], fvec_in.ppar[:,:,is],
                                             moments.vth[:,:,is], moments.qpar[:,:,is], z, r, dt, spectral)
        end
        if composition.n_neutral_species > 0 && abs(CX_frequency) > 0.0
            @views source_terms_evolve_ppar_CX!(pdf_out[:,:,:,:,:], fvec_in.pdf[:,:,:,:,:],
                                                fvec_in.density, fvec_in.ppar, composition,
                                                CX_frequency, dt, z, r)
        end
    elseif moments.evolve_density
        @loop_s is begin
            @views source_terms_evolve_density!(pdf_out[:,:,:,:,is], fvec_in.pdf[:,:,:,:,is],
                                                fvec_in.density[:,:,is], fvec_in.upar[:,:,is], z, r, dt, spectral)
        end
    end
    return nothing
end

"""
"""
function source_terms_evolve_density!(pdf_out, pdf_in, dens, upar, z, r, dt, spectral)
    # update the density
    nvpa = size(pdf_out, 1)
    @loop_r ir begin
        # calculate d(n*upar)/dz
        @views @. z.scratch = dens[:,ir]*upar[:,ir]
        derivative!(z.scratch, z.scratch, z, spectral)
        @views @. z.scratch *= dt/dens[:,ir]
        #derivative!(z.scratch, z.scratch, z, -upar, spectral)
        @loop_z_vperp_vpa iz ivperp ivpa begin
            pdf_out[ivpa,ivperp,iz,ir] += pdf_in[ivpa,ivperp,iz,ir]*z.scratch[iz]
        end
    end
    return nothing
end

"""
"""
function source_terms_evolve_ppar!(pdf_out, pdf_in, dens, upar, ppar, vth, qpar, z, r, dt, spectral)
    nvpa = size(pdf_out, 1)
    @loop_r ir begin
        # calculate dn/dz
        derivative!(z.scratch, view(dens,:,ir), z, spectral)
        # update the pdf to account for the density gradient contribution to the source
        @views @. z.scratch *= dt*upar[:,ir]/dens[:,ir]
        # calculate dvth/dz
        derivative!(z.scratch2, view(vth,:,ir), z, spectral)
        # update the pdf to account for the -g*upar/vth * dvth/dz contribution to the source
        @views @. z.scratch -= dt*z.scratch2*upar[:,ir]/vth[:,ir]
        # calculate dqpar/dz
        derivative!(z.scratch2, view(qpar,:,ir), z, spectral)
        # update the pdf to account for the parallel heat flux contribution to the source
        @views @. z.scratch -= 0.5*dt*z.scratch2/ppar[:,ir]

        @loop_z_vperp_vpa iz ivperp ivpa begin
            pdf_out[ivpa,ivperp,iz,ir] += pdf_in[ivpa,ivperp,iz,ir]*z.scratch[iz]
        end
    end
    return nothing
end

"""
"""
function source_terms_evolve_ppar_CX!(pdf_out, pdf_in, dens, ppar, composition, CX_frequency, dt, z, r)
    @loop_s is begin
        if is ∈ composition.ion_species_range
            for isp ∈ composition.neutral_species_range
                @loop_r_z_vperp ir iz ivperp begin
                    @views @. pdf_out[:,ivperp,iz,ir,is] -= 0.5*dt*pdf_in[:,ivperp,iz,ir,is]*CX_frequency *
                    (dens[iz,ir,isp]*ppar[iz,ir,is]-dens[iz,ir,is]*ppar[iz,ir,isp])/ppar[iz,ir,is]
                end
            end
        end
        if is ∈ composition.neutral_species_range
            for isp ∈ composition.ion_species_range
                @loop_r_z_vperp ir iz ivperp begin
                    @views @. pdf_out[:,ivperp,iz,ir,is] -= 0.5*dt*pdf_in[:,ivperp,iz,ir,is]*CX_frequency *
                    (dens[iz,ir,isp]*ppar[iz,ir,is]-dens[iz,ir,is]*ppar[iz,ir,isp])/ppar[iz,ir,is]
                end
            end
        end
    end
    return nothing
end

"""
advance the dfn with an arbitrary source function 
"""

function source_terms_manufactured!(pdf_out, fvec_in, moments, vpa, vperp, z, r, t, dt, composition, manufactured_source_list)
    Source_i_func = manufactured_source_list.Source_i_func
    @loop_s is begin
        if is ∈ composition.ion_species_range
            @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
                pdf_out[ivpa,ivperp,iz,ir,is] += dt*Source_i_func(vpa.grid[ivpa],vperp.grid[ivperp],z.grid[iz],r.grid[ir],t)
                # correct O(1) here?
            end
        end
        #if is ∈ composition.neutral_species_range
        #PLACEHOLDER for neutral source
        #end
    end
    
    return nothing
end


end
