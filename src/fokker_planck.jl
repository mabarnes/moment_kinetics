"""
module for including the Full-F Fokker-Planck Collision Operator
"""
module fokker_planck


export init_fokker_planck_collisions
export explicit_fokker_planck_collisions!
export evaluate_RMJ_collision_operator!

using SpecialFunctions: ellipk, ellipe
using ..type_definitions: mk_float, mk_int
using ..array_allocation: allocate_float, allocate_shared_float
using ..communication: Array
using ..velocity_moments: integrate_over_vspace
using ..calculus: derivative!, second_derivative!
using ..looping
"""
a struct of dummy arrays and precalculated coefficients
for the Fokker-Planck collision operator 
"""

struct fokkerplanck_arrays_struct
    elliptic_integral_E_factor::Array{mk_float,2}
    elliptic_integral_K_factor::Array{mk_float,2}
    Rosenbluth_G::Array{mk_float,2}
    Rosenbluth_H::Array{mk_float,2}
    buffer_vpamu_1::Array{mk_float,2}
    buffer_vpamu_2::Array{mk_float,2}
    buffer_vpamu_3::Array{mk_float,2}
    Gamma_vpa::Array{mk_float,2}
    Gamma_mu::Array{mk_float,2}
    #Cssp_result_vpamu::Array{mk_float,2}
end

"""
allocate the required ancilliary arrays 
"""

function allocate_fokkerplanck_arrays(mu,vpa)
    nvpa = vpa.n
    nmu = mu.n
    
    #elliptic_integral_E_factor = allocate_float(nvpa,nmu,nvpa,nmu)
    #elliptic_integral_K_factor = allocate_float(nvpa,nmu,nvpa,nmu)
    elliptic_integral_E_factor = allocate_float(nvpa,nmu)
    elliptic_integral_K_factor = allocate_float(nvpa,nmu)
    Rosenbluth_G = allocate_float(nvpa,nmu)
    Rosenbluth_H = allocate_float(nvpa,nmu)
    buffer_vpamu_1 = allocate_float(nvpa,nmu)
    buffer_vpamu_2 = allocate_float(nvpa,nmu)
    buffer_vpamu_3 = allocate_float(nvpa,nmu)
    Gamma_vpa = allocate_float(nvpa,nmu)
    Gamma_mu = allocate_float(nvpa,nmu)
    #Cssp_result_vpamu = allocate_float(nvpa,nmu)
    
    return fokkerplanck_arrays_struct(elliptic_integral_E_factor,elliptic_integral_K_factor,
                               Rosenbluth_G,Rosenbluth_H,
                               buffer_vpamu_1,buffer_vpamu_2,buffer_vpamu_3,
                               Gamma_vpa, Gamma_mu)
                               #Cssp_result_vpamu)
end


# initialise the elliptic integral factor arrays 
# note the definitions of ellipe & ellipk
# `https://specialfunctions.juliamath.org/stable/functions_list/#SpecialFunctions.ellipe`
# `https://specialfunctions.juliamath.org/stable/functions_list/#SpecialFunctions.ellipk`
# `ellipe(m) = \int^{\pi/2}\_0 \sqrt{ 1 - m \sin^2(\theta)} d \theta`
# `ellipe(k) = \int^{\pi/2}\_0 \frac{1}{\sqrt{ 1 - m \sin^2(\theta)}} d \theta`

function init_elliptic_integral_factors!(elliptic_integral_E_factor, elliptic_integral_K_factor, mu, vpa, Bmag, mu_spectral, mm, prefac_E, prefac_K)
    
    # must loop over vpa, mu, vpa', mu'
    # hence mix of looping macros for unprimed variables 
    # & standard local `for' loop for primed variables
    nmu = mu.n
    nvpa = vpa.n
    zero = 1.0e-10
    for imu in 1:nmu
        for ivpa in 1:nvpa
            for imup in 1:nmu
                for ivpap in 1:nvpa                        
                    # the argument of the elliptic integrals 
                    # mm = 4 (2 B sqrt(mu mu')) / ( (vpa- vpa')^2 + 2 B (sqrt(mu) + sqrt(mu') )^2 )
                    denom = (vpa.grid[ivpa] - vpa.grid[ivpap])^2 + 2.0*Bmag*(sqrt(mu.grid[imu]) + sqrt(mu.grid[imup]))^2 
                    if denom < zero 
                        #then vpa = vpa' = mu' = mu = 0 
                        mm[ivpap,imup] = 0.0
                        prefac_E[ivpap,imup] = 0.0 
                        #prefac_K = 0.0 #  
                    else    
                        mm[ivpap,imup] = 4.0*(2.0*Bmag*sqrt(mu.grid[imu]*mu.grid[imup]))/denom
                        # form ( (vpa- vpa')^2 + 2 B (sqrt(mu) + sqrt(mu') )^2 )^{1/2}
                        prefac_E[ivpap,imup] = sqrt(denom)
                        #prefac_K = 1.0/sqrt(denom)
                    end
                end
            end
            # form ( (vpa- vpa')^2 + 2 B (sqrt(mu) + sqrt(mu') )^2 )^{-1/2} by differentiation to avoid singularity
            # this avoids computing H_s'(vpa,mu=0) with a special method to avoid 1/sqrt(mu') divergence in the integrand
            for ivpap in 1:nvpa
                @views derivative!(mu.scratch, prefac_E[ivpap,:], mu, mu_spectral)
                @. prefac_K[ivpap,:] = mu.scratch/Bmag
            end
            for imup in 1:nmu
                for ivpap in 1:nvpa
                    elliptic_integral_E_factor[ivpap,imup,ivpa,imu] = 2.0*ellipe(mm[ivpap,imup])*prefac_E[ivpap,imup]/pi
                    elliptic_integral_K_factor[ivpap,imup,ivpa,imu] = 2.0*ellipk(mm[ivpap,imup])*prefac_K[ivpap,imup]/pi
                end 
            end 
            
        end
    end

end

"""
Alternative method for calculating the elliptic integral factors that 
avoids storing these factors as a large matrix. This is an important 
consideration for when B = B(r,z) and so the matrices would have to be 
nz*nr*nmu^2*nvpa^2 in size -- larger than the pdf!
Here we calculate the elliptic integrals on the fly, which is expected to 
be slower than the precomputed method at runtime.
"""
function get_elliptic_integral_factors!(elliptic_integral_E_factor, elliptic_integral_K_factor, ivpa, imu, mu, vpa, Bmag, mu_spectral, mm, prefac_E, prefac_K)
    
    # must loop over vpa', mu'
    nmu = mu.n
    nvpa = vpa.n
    zero = 1.0e-10
    for imup in 1:nmu
        for ivpap in 1:nvpa                        
            # the argument of the elliptic integrals 
            # mm = 4 (2 B sqrt(mu mu')) / ( (vpa- vpa')^2 + 2 B (sqrt(mu) + sqrt(mu') )^2 )
            denom = (vpa.grid[ivpa] - vpa.grid[ivpap])^2 + 2.0*Bmag*(sqrt(mu.grid[imu]) + sqrt(mu.grid[imup]))^2 
            if denom < zero 
                #then vpa = vpa' = mu' = mu = 0 
                mm[ivpap,imup] = 0.0
                prefac_E[ivpap,imup] = 0.0 
                #prefac_K = 0.0 #  
            else    
                if ivpap == ivpa && imu == imup
                    mm[ivpap,imup] = 1.0
                else
                    mm[ivpap,imup] = 4.0*(2.0*Bmag*sqrt(mu.grid[imu]*mu.grid[imup]))/denom
                end
                # form ( (vpa- vpa')^2 + 2 B (sqrt(mu) + sqrt(mu') )^2 )^{1/2}
                prefac_E[ivpap,imup] = sqrt(denom)
                #prefac_K = 1.0/sqrt(denom)
            end
        end
    end
    # form ( (vpa- vpa')^2 + 2 B (sqrt(mu) + sqrt(mu') )^2 )^{-1/2} by differentiation to avoid singularity
    # this avoids computing H_s'(vpa,mu=0) with a special method to avoid 1/sqrt(mu') divergence in the integrand
    for ivpap in 1:nvpa
        @views derivative!(mu.scratch, prefac_E[ivpap,:], mu, mu_spectral)
        @. prefac_K[ivpap,:] = mu.scratch/Bmag
    end
    for imup in 1:nmu
        for ivpap in 1:nvpa
            elliptic_integral_E_factor[ivpap,imup] = 2.0*ellipe(mm[ivpap,imup])*prefac_E[ivpap,imup]/pi
            elliptic_integral_K_factor[ivpap,imup] = 2.0*ellipk(mm[ivpap,imup])*prefac_K[ivpap,imup]/pi
        end 
    end 
    #println(elliptic_integral_E_factor)
    #println(elliptic_integral_K_factor)
end

"""
function that initialises the arrays needed for Fokker Planck collisions
"""

function init_fokker_planck_collisions(mu, vpa)
    fokkerplanck_arrays = allocate_fokkerplanck_arrays(mu, vpa)
    #@views init_elliptic_integral_factors!(fokkerplanck_arrays.elliptic_integral_E_factor,
    #                                fokkerplanck_arrays.elliptic_integral_K_factor,
    #                                mu, vpa, Bmag, mu_spectral,
    #                                fokkerplanck_arrays.buffer_vpamu_1,
    #                                fokkerplanck_arrays.buffer_vpamu_2,
    #                                fokkerplanck_arrays.buffer_vpamu_3)
    return fokkerplanck_arrays
end

"""
calculates the (normalised) Rosenbluth potential G 
"""
# G(vpa,mu) = \int^\infty_0 \int^{\infty}_{-\infty} ((vpa- vpa')^2 + (mu + mu'))^{1/2}
#                 * (2 ellipe(mm)/ \pi) F(vpa',mu') (2 mu'/\sqrt{\pi}) d mu' d vpa'


function calculate_Rosenbluth_potentials!(Rosenbluth_G,Rosenbluth_H,fsp_in, mu, mu_spectral, vpa, vpa_spectral, Bmag,
     elliptic_integral_E_factor,elliptic_integral_K_factor,buffer_vpamu_1,buffer_vpamu_2,buffer_vpamu_3)
    
    for imu in 1:mu.n 
        for ivpa in 1:vpa.n
            # compute the elliptic integrals 
            @views get_elliptic_integral_factors!(elliptic_integral_E_factor, elliptic_integral_K_factor, 
                                 ivpa, imu, mu, vpa, Bmag, mu_spectral, buffer_vpamu_1, buffer_vpamu_2, buffer_vpamu_3)
            # G
            @views @. buffer_vpamu_1[:,:] = fsp_in*elliptic_integral_E_factor[:,:]
            @views Rosenbluth_G[ivpa,imu] = integrate_over_vspace(buffer_vpamu_1, vpa.grid, 0, vpa.wgts, mu.grid, 0, mu.wgts, Bmag)
            # H 
            #@views @. buffer_vpamu_1[:,:] = fsp_in*elliptic_integral_K_factor[:,:]
            #@views Rosenbluth_H[ivpa,imu] = integrate_over_vspace(buffer_vpamu_1, vpa.grid, 0, vpa.wgts, mu.grid, 0, mu.wgts, Bmag)
        end
    end
    
    Rosenbluth_H .= 0.0
    for imu in 1:mu.n
        vpa.scratch2 .= 1.0
        @views second_derivative!(vpa.scratch, Rosenbluth_G[:,imu], vpa.scratch2, vpa, vpa_spectral)
        @views @. buffer_vpamu_1[:,imu] = vpa.scratch
        #println(buffer_vpamu_1[:,imu])
        #println(vpa.scratch)
    end 
    for ivpa in 1:vpa.n
        @. mu.scratch2 = 2.0 * mu.grid/ Bmag
        @views second_derivative!(mu.scratch, Rosenbluth_G[ivpa,:], mu.scratch2, mu, mu_spectral)
        @views @. buffer_vpamu_2[ivpa,:] = mu.scratch
        #println(buffer_vpamu_2[ivpa,:])
    end
    @views @. Rosenbluth_H = 0.5*(buffer_vpamu_1 + buffer_vpamu_2)
    #println(Rosenbluth_H[1,2])
    #for imu in 1:mu.n
    #    println(buffer_vpamu_1[:,imu])
    #    println(buffer_vpamu_2[:,imu])
    #    println(Rosenbluth_H[:,imu])
    #end
end
 

"""
returns (normalised) C[Fs,Fs']

"""
#returns (normalised) C[F_s,F_s'] = C[F_s,F_s'](vpa,mu) given inputs
#distribution F_s = F_s(vpa,mu) 
#distribution F_s' = F_s'(vpa,mu) 
#mass m_s 
#mass m_s'
#collision frequency nu_{ss'} = gamma_{ss'} n_{ref} / 2 (m_s)^2 (c_{ref})^3
#with gamma_ss' = 2 pi (Z_s Z_s')^2 e^4 ln \Lambda_{ss'} / (4 pi \epsilon_0)^2 
function evaluate_RMJ_collision_operator!(Cssp_out, fs_in, fsp_in, ms, msp, cfreqssp, 
     mu, vpa, mu_spectral, vpa_spectral, Bmag, fokkerplanck_arrays::fokkerplanck_arrays_struct)
    
    # short names for buffer arrays 
    buffer_1 = fokkerplanck_arrays.buffer_vpamu_1
    buffer_2 = fokkerplanck_arrays.buffer_vpamu_2
    buffer_3 = fokkerplanck_arrays.buffer_vpamu_3
    Rosenbluth_G = fokkerplanck_arrays.Rosenbluth_G
    Rosenbluth_H = fokkerplanck_arrays.Rosenbluth_H
    Gamma_vpa = fokkerplanck_arrays.Gamma_vpa
    Gamma_mu = fokkerplanck_arrays.Gamma_mu
    
    # calculate the Rosenbluth potentials
    # and store in fokkerplanck_arrays_struct
    @views calculate_Rosenbluth_potentials!(Rosenbluth_G,Rosenbluth_H,fsp_in,
     mu, mu_spectral, vpa, vpa_spectral, Bmag,
     fokkerplanck_arrays.elliptic_integral_E_factor,
     fokkerplanck_arrays.elliptic_integral_K_factor,
     buffer_1, buffer_2, buffer_3)
    
    
    nmu = mu.n 
    nvpa = vpa.n 
    # zero Cssp to prepare for addition of collision terms 
    Cssp_out .= 0.0
    # zero fluxes to prepare for their calculation 
    Gamma_vpa .= 0.0
    Gamma_mu  .= 0.0
    
    # Gamma_v|| += d F_s / d vpa * d^2 G_sp / d vpa^2
    for imu in 1:nmu
        vpa.scratch2 .= 1.0 # remove Q argument from second_derivative! as never different from 1?
        @views derivative!(vpa.scratch, fs_in[:,imu], vpa, vpa_spectral)
        @views @. buffer_1[:,imu] = vpa.scratch
        @views second_derivative!(vpa.scratch, Rosenbluth_G[:,imu], vpa.scratch2, vpa, vpa_spectral)
        @views @. buffer_2[:,imu] = vpa.scratch
    end 
    @views @. Gamma_vpa += buffer_1*buffer_2
    
    #  Gamma_v|| += + (2 mu/Bmag) d F_s / d mu * d^2 G_sp / d vpa d mu 
    for imu in 1:nmu
        @views derivative!(vpa.scratch, Rosenbluth_G[:,imu], vpa, vpa_spectral)
        @views @. buffer_1[:,imu] = vpa.scratch
    end 
    for ivpa in 1:nvpa
        @views derivative!(mu.scratch, fs_in[ivpa,:], mu, mu_spectral)
        @views @. buffer_2[ivpa,:] = mu.scratch * (2.0 * mu.grid/Bmag)
        @views derivative!(mu.scratch, buffer_1[ivpa,:], mu, mu_spectral)
        @views @. buffer_3[ivpa,:] = mu.scratch
    end 
    @views @. Gamma_vpa += buffer_2*buffer_3
    
    # Gamma_v|| += - 2 (m_s/m_s') F_s * d H_s' / d vpa
    for imu in 1:nmu
        @views derivative!(vpa.scratch, Rosenbluth_H[:,imu], vpa, vpa_spectral)
        @views @. buffer_1[:,imu] = vpa.scratch
    end 
    @views @. Gamma_vpa -= 2.0*(ms/msp)*fs_in*buffer_1
    
    # compute d Gamma_vpa / d vpa
    for imu in 1:nmu
        @views derivative!(vpa.scratch, Gamma_vpa[:,imu], vpa, vpa_spectral)
        @views @. buffer_1[:,imu] = vpa.scratch
    end 
    # C_ss'[F_s,F_s'] += d Gamma_vpa / d vpa
    @views @. Cssp_out += buffer_1
    
    # Gamma_mu += (2 mu / Bmag) d F_s / d vpa * d^2 G_sp / d vpa d mu
    for imu in 1:nmu
        @views derivative!(vpa.scratch, Rosenbluth_G[:,imu], vpa, vpa_spectral)
        @views @. buffer_1[:,imu] = vpa.scratch
        @views derivative!(vpa.scratch, fs_in[:,imu], vpa, vpa_spectral)
        @views @. buffer_2[:,imu] = vpa.scratch
    end 
    for ivpa in 1:nvpa
        @views derivative!(mu.scratch, buffer_1[ivpa,:], mu, mu_spectral)
        @views @. buffer_3[ivpa,:] = mu.scratch * (2.0 * mu.grid / Bmag)
    end 
    @views @. Gamma_mu += buffer_2*buffer_3
    
    # Gamma_mu += (2 mu / Bmag)^{3/2} d F_s / d mu * d / d mu ( (2 mu / Bmag)^{1/2} d G / d mu)
    #for imu in 1:nmu
    #    @views derivative!(vpa.scratch, fs_in[:,imu], vpa, vpa_spectral)
    #    @views @. buffer_1[:,imu] = vpa.scratch
    #    @views derivative!(vpa.scratch, Rosenbluth_G[:,imu], vpa, vpa_spectral)
    #    @views @. buffer_2[:,imu] = vpa.scratch
    #end 
    for ivpa in 1:nvpa
        @views derivative!(mu.scratch, fs_in[ivpa,:], mu, mu_spectral)
        @views @. buffer_1[ivpa,:] = mu.scratch
        @. mu.scratch2 = sqrt(2.0 * mu.grid/ Bmag)
        @views second_derivative!(mu.scratch, Rosenbluth_G[ivpa,:], mu.scratch2, mu, mu_spectral)
        @views @. buffer_2[ivpa,:] = mu.scratch*(2.0 * mu.grid / Bmag)*sqrt(2.0 * mu.grid/ Bmag)
        
    end 
    @views @. Gamma_mu += buffer_1*buffer_2
    
    # Gamma_mu += - 2 (ms/ms') (2 mu / Bmag)  F_s * d H_s'/ d mu 
    for ivpa in 1:nvpa
        @views derivative!(mu.scratch, Rosenbluth_H[ivpa,:], mu, mu_spectral)
        @views @. buffer_1[ivpa,:] = mu.scratch*(2.0 * mu.grid/ Bmag)
    end 
    @views @. Gamma_mu -= 2.0*(ms/msp)*fs_in*buffer_1
    
    # compute d Gamma_mu / d mu
    for ivpa in 1:nvpa
        @views derivative!(mu.scratch, Gamma_mu[ivpa,:], mu, mu_spectral)
        @views @. buffer_1[ivpa,:] = mu.scratch
    end 
    # C_ss'[F_s,F_s'] += d Gamma_mu / d mu
    @views @. Cssp_out += buffer_1
    
    # multiply by overall collision frequency
    @views @. Cssp_out = cfreqssp*Cssp_out
end 

function explicit_fokker_planck_collisions!(pdf_out,pdf_in,composition,geometry,collisions,dt,fokkerplanck_arrays::fokkerplanck_arrays_struct,
                                             scratch_dummy, r, z, mu, vpa, mu_spectral, vpa_spectral)
    n_ion_species = composition.n_ion_species
    @boundscheck vpa.n == size(pdf_out,1) || throw(BoundsError(pdf_out))
    @boundscheck mu.n == size(pdf_out,2) || throw(BoundsError(pdf_out))
    @boundscheck z.n == size(pdf_out,3) || throw(BoundsError(pdf_out))
    @boundscheck r.n == size(pdf_out,4) || throw(BoundsError(pdf_out))
    @boundscheck n_ion_species == size(pdf_out,5) || throw(BoundsError(pdf_out))
    @boundscheck vpa.n == size(pdf_in,1) || throw(BoundsError(pdf_in))
    @boundscheck mu.n == size(pdf_in,2) || throw(BoundsError(pdf_in))
    @boundscheck z.n == size(pdf_in,3) || throw(BoundsError(pdf_in))
    @boundscheck r.n == size(pdf_in,4) || throw(BoundsError(pdf_in))
    @boundscheck n_ion_species == size(pdf_in,5) || throw(BoundsError(pdf_in))
    Cssp_result_vpamu = scratch_dummy.dummy_vpamu
    mi = 1.0 # generalise this to an Array with size n_ion_species
    cfreqii = collisions.nuii # generalise this to an Array with size (n_ion_species,n_ion_species)
    
    begin_r_z_region()
    # serial in s mu vpa for now 
    for is in 1:n_ion_species
        for isp in 1:n_ion_species       
            @loop_r_z ir iz begin
                Bmag = geometry.Bmag
                @views evaluate_RMJ_collision_operator!(Cssp_result_vpamu,pdf_in[:,:,iz,ir,is],pdf_in[:,:,iz,ir,isp],
                                                        mi, mi, cfreqii, mu, vpa, mu_spectral, vpa_spectral, Bmag,
                                                        fokkerplanck_arrays)
                @views @. pdf_out[:,:,iz,ir,is] += dt*Cssp_result_vpamu[:,:]
            end
        end
    end
end

end
