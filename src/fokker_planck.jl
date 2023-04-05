"""
module for including the Full-F Fokker-Planck Collision Operator
"""
module fokker_planck


export init_fokker_planck_collisions
export explicit_fokker_planck_collisions!

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
    elliptic_integral_E_factor::Array{mk_float,4}
    elliptic_integral_K_factor::Array{mk_float,4}
    Rosenbluth_G::Array{mk_float,2}
    Rosenbluth_H::Array{mk_float,2}
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
    Rosenbluth_H = allocate_float(nvpa,nvperp)
    buffer_vpavperp_1 = allocate_float(nvpa,nvperp)
    buffer_vpavperp_2 = allocate_float(nvpa,nvperp)
    #Cssp_result_vpavperp = allocate_float(nvpa,nvperp)
    
    return fokkerplanck_arrays_struct(elliptic_integral_E_factor,elliptic_integral_K_factor,
                               Rosenbluth_G,Rosenbluth_H,
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
                        #then vpa = vpa' = vperp' = vperp = 0 
                        mm = 0.0
                        prefac = 0.0 # because vperp' wgt = 0 here 
                    else    
                        mm = 4.0*vperp.grid[ivperp]*vperp.grid[ivperpp]/denom
                        prefac = sqrt(denom)
                    end
                    elliptic_integral_E_factor[ivpa,ivperp,ivpap,ivperpp] = 2.0*ellipe(mm)*prefac/pi
                    elliptic_integral_K_factor[ivpa,ivperp,ivpap,ivperpp] = 2.0*ellipk(mm)/(pi*prefac)
                end
            end
        end
    end

end

"""
function that initialises the arrays needed for Fokker Planck collisions
"""

function init_fokker_planck_collisions(vperp,vpa)
    fokkerplanck_arrays = allocate_fokkerplanck_arrays(vperp,vpa)
    @views init_elliptic_integral_factors!(fokkerplanck_arrays.elliptic_integral_E_factor,
                                    fokkerplanck_arrays.elliptic_integral_K_factor,
                                    vperp,vpa)
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
    Rosenbluth_H = fokkerplanck_arrays.Rosenbluth_G
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

end
