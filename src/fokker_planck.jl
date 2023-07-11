"""
module for including the Full-F Fokker-Planck Collision Operator
"""
module fokker_planck


export init_fokker_planck_collisions
export explicit_fokker_planck_collisions!
export calculate_Rosenbluth_potentials!
export calculate_collisional_fluxes, calculate_Maxwellian_Rosenbluth_coefficients
export Cflux_vpa_Maxwellian_inputs, Cflux_vperp_Maxwellian_inputs
export calculate_Rosenbluth_H_from_G!

export d2Gdvpa2, dGdvperp, d2Gdvperpdvpa, d2Gdvperp2
export dHdvpa, dHdvperp, Cssp_Maxwellian_inputs
export F_Maxwellian

using SpecialFunctions: ellipk, ellipe, erf
using ..type_definitions: mk_float, mk_int
using ..array_allocation: allocate_float, allocate_shared_float
using ..communication: MPISharedArray
using ..velocity_moments: integrate_over_vspace
using ..calculus: derivative!, second_derivative!
using ..looping
"""
a struct of dummy arrays and precalculated coefficients
for the Fokker-Planck collision operator 
"""

struct fokkerplanck_arrays_struct
    elliptic_integral_E_factor::Array{mk_float,4}
    elliptic_integral_K_factor::Array{mk_float,4}
    Rosenbluth_G::Array{mk_float,2}
    #Rosenbluth_d2Gdvpa2::MPISharedArray{mk_float,2}
    #Rosenbluth_d2Gdvperpdvpa::MPISharedArray{mk_float,2}
    #Rosenbluth_d2Gdvperp2::MPISharedArray{mk_float,2}
    Rosenbluth_H::Array{mk_float,2}
    #Rosenbluth_dHdvpa::MPISharedArray{mk_float,2}
    #Rosenbluth_dHdvperp::MPISharedArray{mk_float,2}
    #Cflux_vpa::MPISharedArray{mk_float,2}
    #Cflux_vperp::MPISharedArray{mk_float,2}
    buffer_vpavperp_1::Array{mk_float,2}
    buffer_vpavperp_2::Array{mk_float,2}
    #Cssp_result_vpavperp::Array{mk_float,2}
end

"""
allocate the required ancilliary arrays 
"""

function allocate_fokkerplanck_arrays(vperp,vpa)
    nvpa = vpa.n
    nvperp = vperp.n
    
    elliptic_integral_E_factor = allocate_float(nvpa,nvperp,nvpa,nvperp)
    elliptic_integral_K_factor = allocate_float(nvpa,nvperp,nvpa,nvperp)
    Rosenbluth_G = allocate_float(nvpa,nvperp)
    #Rosenbluth_d2Gdvpa2 = allocate_shared_float(nvpa,nvperp)
    #Rosenbluth_d2Gdvperpdvpa = allocate_shared_float(nvpa,nvperp)
    #Rosenbluth_d2Gdvperp2 = allocate_shared_float(nvpa,nvperp)
    Rosenbluth_H = allocate_float(nvpa,nvperp)
    #Rosenbluth_dHdvpa = allocate_shared_float(nvpa,nvperp)
    #Rosenbluth_dHdvperp = allocate_shared_float(nvpa,nvperp)
    #Cflux_vpa = allocate_shared_float(nvpa,nvperp)
    #Cflux_vperp = allocate_shared_float(nvpa,nvperp)
    buffer_vpavperp_1 = allocate_float(nvpa,nvperp)
    buffer_vpavperp_2 = allocate_float(nvpa,nvperp)
    #Cssp_result_vpavperp = allocate_float(nvpa,nvperp)
    
    return fokkerplanck_arrays_struct(elliptic_integral_E_factor,elliptic_integral_K_factor,
                               Rosenbluth_G,#Rosenbluth_d2Gdvpa2,Rosenbluth_d2Gdvperpdvpa,Rosenbluth_d2Gdvperp2,
                               Rosenbluth_H,#Rosenbluth_dHdvpa,Rosenbluth_dHdvperp,
                               #Cflux_vpa,Cflux_vperp,
                               buffer_vpavperp_1,buffer_vpavperp_2)
                               #Cssp_result_vpavperp)
end


# initialise the elliptic integral factor arrays 
# note the definitions of ellipe & ellipk
# `https://specialfunctions.juliamath.org/stable/functions_list/#SpecialFunctions.ellipe`
# `https://specialfunctions.juliamath.org/stable/functions_list/#SpecialFunctions.ellipk`
# `ellipe(m) = \int^{\pi/2}\_0 \sqrt{ 1 - m \sin^2(\theta)} d \theta`
# `ellipe(k) = \int^{\pi/2}\_0 \frac{1}{\sqrt{ 1 - m \sin^2(\theta)}} d \theta`

function init_elliptic_integral_factors!(elliptic_integral_E_factor, elliptic_integral_K_factor, vperp, vpa)
    
    # must loop over vpa, vperp, vpa', vperp'
    # hence mix of looping macros for unprimed variables 
    # & standard local `for' loop for primed variables
    nvperp = vperp.n
    nvpa = vpa.n
    zero = 1.0e-10
    for ivperpp in 1:nvperp
        for ivpap in 1:nvpa
            for ivperp in 1:nvperp
                for ivpa in 1:nvpa                        
                    # the argument of the elliptic integrals 
                    # mm = 4 vperp vperp' / ( (vpa- vpa')^2 + (vperp + vperp'))
                    denom = (vpa.grid[ivpa] - vpa.grid[ivpap])^2 + (vperp.grid[ivperp] + vperp.grid[ivperpp])^2 
                    if denom < zero 
                        println("denom = zero ",ivperpp," ",ivpap," ",ivperp," ",ivpa)
                    end
                    #    #then vpa = vpa' = vperp' = vperp = 0 
                    #    mm = 0.0
                    #    prefac = 0.0 # because vperp' wgt = 0 here 
                    #else    
                        mm = 4.0*vperp.grid[ivperp]*vperp.grid[ivperpp]/denom
                        prefac = sqrt(denom)
                    #end
                    #println(mm," ",prefac," ",denom," ",ivperpp," ",ivpap," ",ivperp," ",ivpa)
                    elliptic_integral_E_factor[ivpa,ivperp,ivpap,ivperpp] = 2.0*ellipe(mm)*prefac/pi
                    elliptic_integral_K_factor[ivpa,ivperp,ivpap,ivperpp] = 2.0*ellipk(mm)/(pi*prefac)
                    #println(elliptic_integral_K_factor[ivpa,ivperp,ivpap,ivperpp]," ",mm," ",prefac," ",denom," ",ivperpp," ",ivpap," ",ivperp," ",ivpa)
                    
                end
            end
        end
    end

end

"""
function that initialises the arrays needed for Fokker Planck collisions
"""

function init_fokker_planck_collisions(vperp,vpa;init_integral_factors=false)
    fokkerplanck_arrays = allocate_fokkerplanck_arrays(vperp,vpa)
    if vperp.n > 1 && init_integral_factors
        @views init_elliptic_integral_factors!(fokkerplanck_arrays.elliptic_integral_E_factor,
                                        fokkerplanck_arrays.elliptic_integral_K_factor,
                                        vperp,vpa)
    end
    return fokkerplanck_arrays
end

"""
calculates the (normalised) Rosenbluth potential G 
"""
# G(vpa,vperp) = \int^\infty_0 \int^{\infty}_{-\infty} ((vpa- vpa')^2 + (vperp + vperp'))^{1/2}
#                 * (2 ellipe(mm)/ \pi) F(vpa',vperp') (2 vperp'/\sqrt{\pi}) d vperp' d vpa'


function calculate_Rosenbluth_potentials!(Rosenbluth_G,Rosenbluth_H,fsp_in,
     elliptic_integral_E_factor,elliptic_integral_K_factor,buffer_vpavperp,vperp,vpa)
    
    for ivperp in 1:vperp.n 
        for ivpa in 1:vpa.n
            # G
            @views @. buffer_vpavperp[:,:] = fsp_in*elliptic_integral_E_factor[ivpa,ivperp,:,:]
            @views Rosenbluth_G[ivpa,ivperp] = integrate_over_vspace(buffer_vpavperp, vpa.grid, 0, vpa.wgts, vperp.grid, 0, vperp.wgts)
            # H 
            @views @. buffer_vpavperp[:,:] = fsp_in*elliptic_integral_K_factor[ivpa,ivperp,:,:]
            @views Rosenbluth_H[ivpa,ivperp] = integrate_over_vspace(buffer_vpavperp, vpa.grid, 0, vpa.wgts, vperp.grid, 0, vperp.wgts)
        end
    end

end

"""
Computes the Laplacian of G in vpa vperp coordinates to obtain H
""" 
function calculate_Rosenbluth_H_from_G!(Rosenbluth_H,Rosenbluth_G,vpa,vpa_spectral,vperp,vperp_spectral,buffer_vpavperp_1,buffer_vpavperp_2)
    Rosenbluth_H .= 0.0
    for ivperp in 1:vperp.n
        @views derivative!(vpa.scratch, Rosenbluth_G[:,ivperp], vpa, vpa_spectral)
        @views derivative!(vpa.scratch2, vpa.scratch, vpa, vpa_spectral)
        @views @. buffer_vpavperp_1[:,ivperp] = vpa.scratch2
    end 
    for ivpa in 1:vpa.n
        @views derivative!(vperp.scratch, Rosenbluth_G[ivpa,:], vperp, vperp_spectral)
        @. vperp.scratch = vperp.grid*vperp.scratch
        @views derivative!(vperp.scratch2, vperp.scratch, vperp, vperp_spectral)
        @views @. buffer_vpavperp_2[ivpa,:] = vperp.scratch2/vperp.grid
    end
    @views @. Rosenbluth_H = 0.5*(buffer_vpavperp_1 + buffer_vpavperp_2)
end
 

"""
calculates the (normalised) Rosenbluth potential coefficients d2Gdvpa2, d2Gdvperpdvpa, ..., dHdvperp for a Maxwellian inputs.
"""
function calculate_Maxwellian_Rosenbluth_coefficients(dens,upar,vth,vpa,vperp,ivpa,ivperp,n_ion_species) # Rosenbluth_d2Gdvpa2,Rosenbluth_d2Gdvperpdvpa,Rosenbluth_d2Gdvperp2,Rosenbluth_dHdvpa,Rosenbluth_dHdvperp,
    # zero coefficients prior to looping over s'
    Rosenbluth_d2Gdvpa2 = 0.0
    Rosenbluth_d2Gdvperpdvpa = 0.0
    Rosenbluth_d2Gdvperp2 = 0.0
    Rosenbluth_dHdvpa = 0.0
    Rosenbluth_dHdvperp = 0.0
    
    # fill in value at (ivpa,ivperp)
    for isp in 1:n_ion_species
        Rosenbluth_d2Gdvpa2 += d2Gdvpa2(dens[isp],upar[isp],vth[isp],vpa,vperp,ivpa,ivperp)
        Rosenbluth_d2Gdvperpdvpa += d2Gdvperpdvpa(dens[isp],upar[isp],vth[isp],vpa,vperp,ivpa,ivperp)
        Rosenbluth_d2Gdvperp2 += d2Gdvperp2(dens[isp],upar[isp],vth[isp],vpa,vperp,ivpa,ivperp)
        Rosenbluth_dHdvpa += dHdvpa(dens[isp],upar[isp],vth[isp],vpa,vperp,ivpa,ivperp)
        Rosenbluth_dHdvperp += dHdvperp(dens[isp],upar[isp],vth[isp],vpa,vperp,ivpa,ivperp)
    end
    return Rosenbluth_d2Gdvpa2,Rosenbluth_d2Gdvperpdvpa,Rosenbluth_d2Gdvperp2,Rosenbluth_dHdvpa,Rosenbluth_dHdvperp
end

"""
calculates the collisional fluxes given input F_s and G_sp, H_sp
"""
function calculate_collisional_fluxes(F,dFdvpa,dFdvperp,
                            d2Gdvpa2,d2Gdvperpdvpa,d2Gdvperp2,dHdvpa,dHdvperp,
                            ms,msp)
    # fill in value at (ivpa,ivperp)
    Cflux_vpa = dFdvpa*d2Gdvpa2 + dFdvperp*d2Gdvperpdvpa - 2.0*(ms/msp)*F*dHdvpa
    #Cflux_vpa = dFdvpa*d2Gdvpa2 + dFdvperp*d2Gdvperpdvpa # - 2.0*(ms/msp)*F*dHdvpa
    #Cflux_vpa =  - 2.0*(ms/msp)*F*dHdvpa
    Cflux_vperp = dFdvpa*d2Gdvperpdvpa + dFdvperp*d2Gdvperp2 - 2.0*(ms/msp)*F*dHdvperp
    return Cflux_vpa, Cflux_vperp
end

"""
returns (normalised) C[Fs,Fs']

"""
#returns (normalised) C[F_s,F_s'] = C[F_s,F_s'](vpa,vperp) given inputs
#distribution F_s = F_s(vpa,vperp) 
#distribution F_s' = F_s'(vpa,vperp) 
#mass m_s 
#mass m_s'
#collision frequency nu_{ss'} = gamma_{ss'} n_{ref} / 2 (m_s)^2 (c_{ref})^3
#with gamma_ss' = 2 pi (Z_s Z_s')^2 e^4 ln \Lambda_{ss'} / (4 pi \epsilon_0)^2 
function evaluate_RMJ_collision_operator!(Cssp_out,fs_in,fsp_in,ms,msp,cfreqssp, fokkerplanck_arrays::fokkerplanck_arrays_struct, vperp, vpa, vperp_spectral, vpa_spectral)
    # calculate the Rosenbluth potentials
    # and store in fokkerplanck_arrays_struct
    @views calculate_Rosenbluth_potentials!(fokkerplanck_arrays.Rosenbluth_G,fokkerplanck_arrays.Rosenbluth_H,fsp_in,
     fokkerplanck_arrays.elliptic_integral_E_factor,
     fokkerplanck_arrays.elliptic_integral_K_factor,
     fokkerplanck_arrays.buffer_vpavperp_1,vperp,vpa)
    
    # short names for buffer arrays 
    buffer_1 = fokkerplanck_arrays.buffer_vpavperp_1
    buffer_2 = fokkerplanck_arrays.buffer_vpavperp_2
    Rosenbluth_G = fokkerplanck_arrays.Rosenbluth_G
    Rosenbluth_H = fokkerplanck_arrays.Rosenbluth_H
    nvperp = vperp.n 
    nvpa = vpa.n 
    # zero Cssp to prepare for addition of collision terms 
    Cssp_out .= 0.0
    
    #  + d^2 F_s / d vpa^2 * d^2 G_sp / d vpa^2 
    for ivperp in 1:nvperp
        vpa.scratch2 .= 1.0 # remove Q argument from second_derivative! as never different from 1?
        @views second_derivative!(vpa.scratch, fs_in[:,ivperp], vpa.scratch2, vpa, vpa_spectral)
        @views @. buffer_1[:,ivperp] = vpa.scratch
        @views second_derivative!(vpa.scratch, Rosenbluth_G[:,ivperp], vpa.scratch2, vpa, vpa_spectral)
        @views @. buffer_2[:,ivperp] = vpa.scratch
    end 
    @views @. Cssp_out += buffer_1*buffer_2
    
    #  + 2 d^2 F_s / d vpa d vperp * d^2 G_sp / d vpa d vperp 
    for ivperp in 1:nvperp
        @views derivative!(vpa.scratch, fs_in[:,ivperp], vpa, vpa_spectral)
        @views @. buffer_1[:,ivperp] = vpa.scratch
        @views derivative!(vpa.scratch, Rosenbluth_G[:,ivperp], vpa, vpa_spectral)
        @views @. buffer_2[:,ivperp] = vpa.scratch
    end 
    for ivpa in 1:nvpa
        @views derivative!(vperp.scratch, buffer_1[ivpa,:], vperp, vperp_spectral)
        @views @. buffer_1[ivpa,:] = vperp.scratch
        @views derivative!(vperp.scratch, buffer_2[ivpa,:], vperp, vperp_spectral)
        @views @. buffer_2[ivpa,:] = vperp.scratch
    end 
    @views @. Cssp_out += 2.0*buffer_1*buffer_2
    
    #  + d^2 F_s / d vperp^2 * d^2 G_sp / d vperp^2 
    for ivpa in 1:nvpa
        vperp.scratch2 .= 1.0 # remove Q argument from second_derivative! as never different from 1?
        @views second_derivative!(vperp.scratch, fs_in[ivpa,:], vperp.scratch2, vperp, vperp_spectral)
        @views @. buffer_1[ivpa,:] = vperp.scratch
        @views second_derivative!(vperp.scratch, Rosenbluth_G[ivpa,:], vperp.scratch2, vperp, vperp_spectral)
        @views @. buffer_2[ivpa,:] = vperp.scratch
    end 
    @views @. Cssp_out += buffer_1*buffer_2
    
    #  + ( 1/vperp^2) d F_s / d vperp * d G_sp / d vperp 
    for ivpa in 1:nvpa
        @views derivative!(vperp.scratch, fs_in[ivpa,:], vperp, vperp_spectral)
        @views @. buffer_1[ivpa,:] = vperp.scratch/(vperp.grid^2) # MRH this line causes divide by zero!
        @views derivative!(vperp.scratch, Rosenbluth_G[ivpa,:], vperp, vperp_spectral)
        @views @. buffer_2[ivpa,:] = vperp.scratch
    end 
    @views @. Cssp_out += buffer_1*buffer_2
    
    #  + 2( 1 - ms/msp) * d F_s / d vpa * d H_sp / d vpa 
    for ivperp in 1:nvperp
        @views derivative!(vpa.scratch, fs_in[:,ivperp], vpa, vpa_spectral)
        @views @. buffer_1[:,ivperp] = vpa.scratch
        @views derivative!(vpa.scratch, Rosenbluth_H[:,ivperp], vpa, vpa_spectral)
        @views @. buffer_2[:,ivperp] = vpa.scratch
    end 
    @views @. Cssp_out += 2.0*(1.0 - ms/msp)*buffer_1*buffer_2
    
    #  + 2( 1 - ms/msp) * d F_s / d vperp * d H_sp / d vperp 
    for ivpa in 1:nvpa
        @views derivative!(vperp.scratch, fs_in[ivpa,:], vperp, vperp_spectral)
        @views @. buffer_1[ivpa,:] = vperp.scratch
        @views derivative!(vperp.scratch, Rosenbluth_H[ivpa,:], vperp, vperp_spectral)
        @views @. buffer_2[ivpa,:] = vperp.scratch
    end 
    @views @. Cssp_out += 2.0*(1.0 - ms/msp)*buffer_1*buffer_2
    
    # + (8 ms / \sqrt{\pi} msp ) F_s F_sp 
    @views @. Cssp_out += ((8.0*ms)/(sqrt(pi)*msp))*fs_in*fsp_in
    
    # multiply by overall collision frequency
    @views @. Cssp_out = cfreqssp*Cssp_out
end 

function explicit_fokker_planck_collisions!(pdf_out,pdf_in,composition,collisions,dt,fokkerplanck_arrays::fokkerplanck_arrays_struct,
                                             scratch_dummy, r, z, vperp, vpa, vperp_spectral, vpa_spectral)
    n_ion_species = composition.n_ion_species
    @boundscheck vpa.n == size(pdf_out,1) || throw(BoundsError(pdf_out))
    @boundscheck vperp.n == size(pdf_out,2) || throw(BoundsError(pdf_out))
    @boundscheck z.n == size(pdf_out,3) || throw(BoundsError(pdf_out))
    @boundscheck r.n == size(pdf_out,4) || throw(BoundsError(pdf_out))
    @boundscheck n_ion_species == size(pdf_out,5) || throw(BoundsError(pdf_out))
    @boundscheck vpa.n == size(pdf_in,1) || throw(BoundsError(pdf_in))
    @boundscheck vperp.n == size(pdf_in,2) || throw(BoundsError(pdf_in))
    @boundscheck z.n == size(pdf_in,3) || throw(BoundsError(pdf_in))
    @boundscheck r.n == size(pdf_in,4) || throw(BoundsError(pdf_in))
    @boundscheck n_ion_species == size(pdf_in,5) || throw(BoundsError(pdf_in))
    Cssp_result_vpavperp = scratch_dummy.dummy_vpavperp
    mi = 1.0 # generalise this to an Array with size n_ion_species
    cfreqii = collisions.nuii # generalise this to an Array with size (n_ion_species,n_ion_species)
    
    begin_r_z_region()
    # serial in s vperp vpa for now 
    for is in 1:n_ion_species
        for isp in 1:n_ion_species       
            @loop_r_z ir iz begin
                @views evaluate_RMJ_collision_operator!(Cssp_result_vpavperp,pdf_in[:,:,iz,ir,is],pdf_in[:,:,iz,ir,isp],
                                                        mi, mi, cfreqii, fokkerplanck_arrays, vperp, vpa,
                                                        vperp_spectral, vpa_spectral)
                @views @. pdf_out[:,:,iz,ir,is] += dt*Cssp_result_vpavperp[:,:]
            end
        end
    end
end

function explicit_fokker_planck_collisions_Maxwellian_coefficients!(pdf_out,pdf_in,dens_in,upar_in,vth_in,
                                             composition,collisions,dt,fokkerplanck_arrays::fokkerplanck_arrays_struct,
                                             scratch_dummy, r, z, vperp, vpa, vperp_spectral, vpa_spectral)
    n_ion_species = composition.n_ion_species
    @boundscheck vpa.n == size(pdf_out,1) || throw(BoundsError(pdf_out))
    @boundscheck vperp.n == size(pdf_out,2) || throw(BoundsError(pdf_out))
    @boundscheck z.n == size(pdf_out,3) || throw(BoundsError(pdf_out))
    @boundscheck r.n == size(pdf_out,4) || throw(BoundsError(pdf_out))
    @boundscheck n_ion_species == size(pdf_out,5) || throw(BoundsError(pdf_out))
    @boundscheck vpa.n == size(pdf_in,1) || throw(BoundsError(pdf_in))
    @boundscheck vperp.n == size(pdf_in,2) || throw(BoundsError(pdf_in))
    @boundscheck z.n == size(pdf_in,3) || throw(BoundsError(pdf_in))
    @boundscheck r.n == size(pdf_in,4) || throw(BoundsError(pdf_in))
    @boundscheck n_ion_species == size(pdf_in,5) || throw(BoundsError(pdf_in))
    
    mi = 1.0 # generalise this to an Array with size n_ion_species
    mip = 1.0 # generalise this to an Array with size n_ion_species
    cfreqii = collisions.nuii # generalise this to an Array with size (n_ion_species,n_ion_species)
    fk = fokkerplanck_arrays
    Cssp_result_vpavperp = scratch_dummy.dummy_vpavperp
    pdf_buffer_1 = scratch_dummy.buffer_vpavperpzrs_1
    pdf_buffer_2 = scratch_dummy.buffer_vpavperpzrs_2
    
    # precompute derivatives of the pdfs to benefit from parallelisation
    # d F / d vpa
    begin_s_r_z_vperp_region()
    @loop_s_r_z_vperp is ir iz ivperp begin
        @views derivative!(vpa.scratch, pdf_in[:,ivperp,iz,ir,is], vpa, vpa_spectral)
        @. pdf_buffer_1[:,ivperp,iz,ir,is] = vpa.scratch
    end
    # d F / d vperp
    begin_s_r_z_vpa_region()
    @loop_s_r_z_vpa is ir iz ivpa begin
        @views derivative!(vperp.scratch, pdf_in[ivpa,:,iz,ir,is], vperp, vperp_spectral)
        @. pdf_buffer_2[ivpa,:,iz,ir,is] = vperp.scratch
    end
    
    begin_s_r_z_vperp_vpa_region()
    @loop_s_r_z is ir iz begin
        @loop_vperp_vpa ivperp ivpa begin
            # first compute local (in z,r) Rosenbluth potential coefficients, summing over all s'
            # ((fk.Rosenbluth_d2Gdvpa2[ivpa,ivperp], fk.Rosenbluth_d2Gdvperpdvpa[ivpa,ivperp], 
            #fk.Rosenbluth_d2Gdvperp2[ivpa,ivperp],fk.Rosenbluth_dHdvpa[ivpa,ivperp],
            #fk.Rosenbluth_dHdvperp[ivpa,ivperp])
            ((Rosenbluth_d2Gdvpa2, Rosenbluth_d2Gdvperpdvpa, 
            Rosenbluth_d2Gdvperp2,Rosenbluth_dHdvpa,
            Rosenbluth_dHdvperp) = calculate_Maxwellian_Rosenbluth_coefficients(dens_in[iz,ir,:],
                 upar_in[iz,ir,:],vth_in[iz,ir,:],vpa,vperp,ivpa,ivperp,n_ion_species) )
                 
            # now form the collisional fluxes at this s,z,r
            ( (Cflux_vpa,Cflux_vperp) = calculate_collisional_fluxes(pdf_in[ivpa,ivperp,iz,ir,is],
                    pdf_buffer_1[ivpa,ivperp,iz,ir,is],pdf_buffer_2[ivpa,ivperp,iz,ir,is],
                    Rosenbluth_d2Gdvpa2,Rosenbluth_d2Gdvperpdvpa,
                    Rosenbluth_d2Gdvperp2,Rosenbluth_dHdvpa,Rosenbluth_dHdvperp,
                    mi,mip) )
            
            # now overwrite the buffer arrays with the local values as we no longer need dFdvpa or dFdvperp at s,r,z
            pdf_buffer_1[ivpa,ivperp,iz,ir,is] = Cflux_vpa
            pdf_buffer_2[ivpa,ivperp,iz,ir,is] = Cflux_vperp
        end
        
    end
    
    # now differentiate the fluxes to obtain the explicit operator 
    
    # d Cflux_vpa / d vpa
    begin_s_r_z_vperp_region()
    @loop_s_r_z_vperp is ir iz ivperp begin
        @views derivative!(vpa.scratch, pdf_buffer_1[:,ivperp,iz,ir,is], vpa, vpa_spectral)
        @. pdf_buffer_1[:,ivperp,iz,ir,is] = vpa.scratch
    end
    # (1/vperp) d Cflux_vperp / d vperp
    begin_s_r_z_vpa_region()
    @loop_s_r_z_vpa is ir iz ivpa begin
        @views @. vperp.scratch2 = vperp.grid*pdf_buffer_2[ivpa,:,iz,ir,is]
        @views derivative!(vperp.scratch, vperp.scratch2, vperp, vperp_spectral)
        @. pdf_buffer_2[ivpa,:,iz,ir,is] = vperp.scratch[:]/vperp.grid[:]
    end
    
    # now add the result to the outgoing pdf
    # d F / d t = nu_ii * ( d Cflux_vpa / d vpa + (1/vperp) d Cflux_vperp / d vperp)
    begin_s_r_z_vperp_vpa_region()
    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        pdf_out[ivpa,ivperp,iz,ir,is] += dt*cfreqii*(pdf_buffer_1[ivpa,ivperp,iz,ir,is] + pdf_buffer_1[ivpa,ivperp,iz,ir,is])
    end
end

# below are a series of functions that can be used to test the calculation 
# of the Rosenbluth potentials for a shifted Maxwellian
# or provide an estimate for collisional coefficients 

# 1D derivative functions

function dGdeta(eta::mk_float)
    # d \tilde{G} / d eta
    dGdeta_fac = (1.0/sqrt(pi))*exp(-eta^2)/eta + (1.0 - 0.5/(eta^2))*erf(eta)
    return dGdeta_fac
end

function d2Gdeta2(eta::mk_float)
    # d \tilde{G} / d eta
    d2Gdeta2_fac = erf(eta)/(eta^3) - (2.0/sqrt(pi))*exp(-eta^2)/(eta^2)
    return d2Gdeta2_fac
end

function ddGddeta(eta::mk_float)
    # d / d eta ( (1/ eta) d \tilde{G} d eta 
    ddGddeta_fac = (1.5/(eta^2) - 1.0)*erf(eta)/(eta^2) - (3.0/sqrt(pi))*exp(-eta^2)/(eta^3)
    return ddGddeta_fac
end

function dHdeta(eta::mk_float)
    dHdeta_fac = (2.0/sqrt(pi))*(exp(-eta^2))/eta - erf(eta)/(eta^2)
    return dHdeta_fac
end

# functions of vpa & vperp 
function eta_func(upar::mk_float,vth::mk_float,
             vpa,vperp,ivpa,ivperp)
    speed = sqrt( (vpa.grid[ivpa] - upar)^2 + vperp.grid[ivperp]^2)/vth
    return speed
end

function d2Gdvpa2(dens::mk_float,upar::mk_float,vth::mk_float,
                            vpa,vperp,ivpa,ivperp)
    eta = eta_func(upar,vth,vpa,vperp,ivpa,ivperp)
    fac = dGdeta(eta) + ddGddeta(eta)*((vpa.grid[ivpa] - upar)^2)/(vth^2)
    d2Gdvpa2_fac = fac*dens/(eta*vth)
    return d2Gdvpa2_fac
end

function d2Gdvperpdvpa(dens::mk_float,upar::mk_float,vth::mk_float,
                            vpa,vperp,ivpa,ivperp)
    eta = eta_func(upar,vth,vpa,vperp,ivpa,ivperp)
    fac = ddGddeta(eta)*vperp.grid[ivperp]*(vpa.grid[ivpa] - upar)/(vth^2)
    d2Gdvperpdvpa_fac = fac*dens/(eta*vth)
    return d2Gdvperpdvpa_fac
end

function d2Gdvperp2(dens::mk_float,upar::mk_float,vth::mk_float,
                            vpa,vperp,ivpa,ivperp)
    eta = eta_func(upar,vth,vpa,vperp,ivpa,ivperp)
    fac = dGdeta(eta) + ddGddeta(eta)*(vperp.grid[ivperp]^2)/(vth^2)
    d2Gdvperp2_fac = fac*dens/(eta*vth)
    return d2Gdvperp2_fac
end

function dGdvperp(dens::mk_float,upar::mk_float,vth::mk_float,
                            vpa,vperp,ivpa,ivperp)
    eta = eta_func(upar,vth,vpa,vperp,ivpa,ivperp)
    fac = dGdeta(eta)*vperp.grid[ivperp]*dens/(vth*eta)
    return fac 
end

function dHdvperp(dens::mk_float,upar::mk_float,vth::mk_float,
                            vpa,vperp,ivpa,ivperp)
    eta = eta_func(upar,vth,vpa,vperp,ivpa,ivperp)
    fac = dHdeta(eta)*vperp.grid[ivperp]*dens/(eta*vth^3)
    return fac 
end

function dHdvpa(dens::mk_float,upar::mk_float,vth::mk_float,
                            vpa,vperp,ivpa,ivperp)
    eta = eta_func(upar,vth,vpa,vperp,ivpa,ivperp)
    fac = dHdeta(eta)*(vpa.grid[ivpa]-upar)*dens/(eta*vth^3)
    return fac 
end

function F_Maxwellian(dens::mk_float,upar::mk_float,vth::mk_float,
                        vpa,vperp,ivpa,ivperp)
    eta = eta_func(upar,vth,vpa,vperp,ivpa,ivperp)
    fac = (dens/(vth^3))*exp(-eta^2)
    return fac
end

function dFdvpa_Maxwellian(dens::mk_float,upar::mk_float,vth::mk_float,
                        vpa,vperp,ivpa,ivperp)
    eta = eta_func(upar,vth,vpa,vperp,ivpa,ivperp)
    fac = -2.0*(dens/(vth^4))*((vpa.grid[ivpa] - upar)/vth)*exp(-eta^2)
    return fac
end

function dFdvperp_Maxwellian(dens::mk_float,upar::mk_float,vth::mk_float,
                        vpa,vperp,ivpa,ivperp)
    eta = eta_func(upar,vth,vpa,vperp,ivpa,ivperp)
    fac = -2.0*(dens/(vth^4))*(vperp.grid[ivperp]/vth)*exp(-eta^2)
    return fac
end

function d2Fdvperpdvpa_Maxwellian(dens::mk_float,upar::mk_float,vth::mk_float,
                        vpa,vperp,ivpa,ivperp)
    eta = eta_func(upar,vth,vpa,vperp,ivpa,ivperp)
    fac = 4.0*(dens/(vth^5))*(vperp.grid[ivperp]/vth)*((vpa.grid[ivpa] - upar)/vth)*exp(-eta^2)
    return fac
end

function d2Fdvpa2_Maxwellian(dens::mk_float,upar::mk_float,vth::mk_float,
                        vpa,vperp,ivpa,ivperp)
    eta = eta_func(upar,vth,vpa,vperp,ivpa,ivperp)
    fac = 4.0*(dens/(vth^5))*( ((vpa.grid[ivpa] - upar)/vth)^2 - 0.5 )*exp(-eta^2)
    return fac
end

function d2Fdvperp2_Maxwellian(dens::mk_float,upar::mk_float,vth::mk_float,
                        vpa,vperp,ivpa,ivperp)
    eta = eta_func(upar,vth,vpa,vperp,ivpa,ivperp)
    fac = 4.0*(dens/(vth^5))*((vperp.grid[ivperp]/vth)^2 - 0.5)*exp(-eta^2)
    return fac
end

function Cssp_Maxwellian_inputs(denss::mk_float,upars::mk_float,vths::mk_float,ms::mk_float,
                                denssp::mk_float,uparsp::mk_float,vthsp::mk_float,msp::mk_float,
                                nussp::mk_float,vpa,vperp,ivpa,ivperp)
    
    d2Fsdvpa2 = d2Fdvpa2_Maxwellian(denss,upars,vths,vpa,vperp,ivpa,ivperp)
    d2Fsdvperp2 = d2Fdvperp2_Maxwellian(denss,upars,vths,vpa,vperp,ivpa,ivperp)
    d2Fsdvperpdvpa = d2Fdvperpdvpa_Maxwellian(denss,upars,vths,vpa,vperp,ivpa,ivperp)
    dFsdvperp = dFdvperp_Maxwellian(denss,upars,vths,vpa,vperp,ivpa,ivperp)
    dFsdvpa = dFdvpa_Maxwellian(denss,upars,vths,vpa,vperp,ivpa,ivperp)
    Fs = F_Maxwellian(denss,upars,vths,vpa,vperp,ivpa,ivperp)
    
    d2Gspdvpa2 = d2Gdvpa2(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
    d2Gspdvperp2 = d2Gdvperp2(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
    d2Gspdvperpdvpa = d2Gdvperpdvpa(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
    dGspdvperp = dGdvperp(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
    dHspdvperp = dHdvperp(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
    dHspdvpa = dHdvpa(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
    Fsp = F_Maxwellian(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
    
    ( Cssp_Maxwellian = 
        d2Fsdvpa2*d2Gspdvpa2 + 
        d2Fsdvperp2*d2Gspdvperp2 + 
        2.0*d2Fsdvperpdvpa*d2Gspdvperpdvpa + 
        (1.0/(vperp.grid[ivperp]^2))*dFsdvperp*dGspdvperp +
        2.0*(1.0 - (ms/msp))*(dFsdvpa*dHspdvpa + dFsdvperp*dHspdvperp) +
        (8.0/sqrt(pi))*(ms/msp)*Fs*Fsp ) 
        
    Cssp_Maxwellian *= nussp
    return Cssp_Maxwellian
end

function Cflux_vpa_Maxwellian_inputs(ms::mk_float,denss::mk_float,upars::mk_float,vths::mk_float,
                                     msp::mk_float,denssp::mk_float,uparsp::mk_float,vthsp::mk_float,
                                     vpa,vperp,ivpa,ivperp)
    etap = eta_func(uparsp,vthsp,vpa,vperp,ivpa,ivperp)
    eta = eta_func(upars,vths,vpa,vperp,ivpa,ivperp)
    prefac = -2.0*denss*denssp*exp( -eta^2)/(vthsp*vths^5)
    (fac = (vpa.grid[ivpa]-uparsp)*(d2Gdeta2(etap) + (ms/msp)*((vths/vthsp)^2)*dHdeta(etap)/etap)
             + (uparsp - upars)*( dGdeta(etap) + ((vpa.grid[ivpa]-uparsp)^2/vthsp^2)*ddGddeta(etap) )/etap )
    Cflux = prefac*fac
    #fac *= (ms/msp)*(vths/vthsp)*dHdeta(etap)/etap
    #fac *= d2Gdeta2(etap) 
    return Cflux
end

function Cflux_vperp_Maxwellian_inputs(ms::mk_float,denss::mk_float,upars::mk_float,vths::mk_float,
                                     msp::mk_float,denssp::mk_float,uparsp::mk_float,vthsp::mk_float,
                                     vpa,vperp,ivpa,ivperp)
    etap = eta_func(uparsp,vthsp,vpa,vperp,ivpa,ivperp)
    eta = eta_func(upars,vths,vpa,vperp,ivpa,ivperp)
    prefac = -2.0*(vperp.grid[ivperp])*denss*denssp*exp( -eta^2)/(vthsp*vths^5)
    (fac = (d2Gdeta2(etap) + (ms/msp)*((vths/vthsp)^2)*dHdeta(etap)/etap)
             + ((uparsp - upars)*(vpa.grid[ivpa]-uparsp)/vthsp^2)*ddGddeta(etap)/etap )
    Cflux = prefac*fac
    #fac *= (ms/msp)*(vths/vthsp)*dHdeta(etap)/etap
    #fac *= d2Gdeta2(etap) 
    return Cflux
end

end
