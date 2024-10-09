"""
Model Diffusion operator - this will act to push the distribution towards
Maxwellian, but purely based on velocity gradients. So need the second derivative of 
the distribution function. Integration constants should not be a problem, i.e. pushing 
the second derivatives of a Maxwellian and our function together should also move their 
values towards each other, as the boundary conditions of both would be the same. 

This operator will mostly come into effect in places where there is ringing in the 
distribution, which can be expected in the grid points near the walls. Fortunately this
region is also where the plasma is most collisional, so having such an operator is also
most valid here.
"""

module maxwell_diffusion

export setup_mxwl_diff_collisions_input, ion_vpa_maxwell_diffusion!, neutral_vz_maxwell_diffusion!

using ..looping
using ..input_structs: mxwl_diff_collisions_input, set_defaults_and_check_section!
using ..calculus: second_derivative!
using ..timer_utils
using ..reference_parameters: get_reference_collision_frequency_ii, setup_reference_parameters

"""
Function for reading Maxwell diffusion operator input parameters. 
Structure the namelist as follows.

[maxwell_diffusion_collisions]
use_maxwell_diffusion = true
D_ii = 1.0
diffusion_coefficient_option = "manual"
"""
function setup_mxwl_diff_collisions_input(toml_input::Dict)
    reference_params = setup_reference_parameters(toml_input)
    # get reference diffusion coefficient, made up of collision frequency and 
    # thermal speed for now. NOTE THAT THIS CONSTANT PRODUCES ERRORS. DO NOT USE
    D_ii_mxwl_diff_default = get_reference_collision_frequency_ii(reference_params)# *
                             #2 * reference_params.Tref/reference_params.mref
    D_nn_mxwl_diff_default = D_ii_mxwl_diff_default
    # read the input toml and specify a sensible default    
    input_section = set_defaults_and_check_section!(toml_input, "maxwell_diffusion_collisions",
       # begin default inputs (as kwargs)
       use_maxwell_diffusion = false,
       D_ii = -1.0,
       D_nn = -1.0,
       diffusion_coefficient_option = "reference_parameters")
       
    # ensure that the diffusion coefficient is consistent with the input option
    diffusion_coefficient_option = input_section["diffusion_coefficient_option"]
    if diffusion_coefficient_option == "reference_parameters"
        input_section["D_ii"] = D_ii_mxwl_diff_default
        input_section["D_nn"] = -1.0 #D_nn_mxwl_diff_default
    elseif diffusion_coefficient_option == "manual" 
        # use the diffusion coefficient from the input file
        # do nothing
    else
        error("Invalid option [maxwell_diffusion_collisions] "
              * "diffusion_coefficient_option=$(diffusion_coefficient_option) passed")
    end
    # finally, ensure prefactor < 0 if use_maxwell_diffusion is false
    # so that prefactor > 0 is the only check required in the rest of the code
    if !input_section["use_maxwell_diffusion"]
        input_section["D_ii"] = -1.0
        input_section["D_nn"] = -1.0
    end
    input = Dict(Symbol(k)=>v for (k,v) in input_section)

    return mxwl_diff_collisions_input(; input...)
end

"""
Calculate the Maxwellian associated with the current ion pdf moments, and then 
subtract this from current pdf. Then take second derivative of this function
to act as the diffusion operator. 
"""
@timeit global_timer ion_vpa_maxwell_diffusion!(
                         f_out, f_in, moments, vpa, vperp, spectral::T_spectral, dt,
                         diffusion_coefficient) where T_spectral = begin
    
    # If negative input (should be -1.0), then none of this diffusion will happen. 
    # This number can be put in as some parameter in the input file called something
    # like 'maxwellian_diffusion_coefficient'
    if diffusion_coefficient <= 0.0 || vpa.n == 1
        return nothing
    end

    if vperp.n > 1 && (moments.evolve_density || moments.evolve_upar || moments.evolve_ppar)
        error("Maxwell diffusion not implemented for 2V moment-kinetic cases yet")
    end

    # Otherwise, build the maxwellian function (which is going to be subtracted from 
    # the current distribution) using the moments of the distribution (so that the 
    # operator itself conserves the moments), and then this result will be the one 
    # whose second derivative will be added to the RHS (i.e. subtracted from the current)
    begin_s_r_z_vperp_region()

    # In what follows, there are eight combinations of booleans (though not all have been
    # fully implemented yet). In line with moment kinetics, the Maxwellian is normalised
    # in the relevant ways:
    # - density: normalise by n 
    # - upar: working in peculiar velocity space, so no upar subtraction from vpa 
    # - ppar: normalisation by vth, in 1D is 1/vth prefactor, and grid is normalised by vth,
    # hence no 1/vth^2 term in the exponent.
    if moments.evolve_density && moments.evolve_upar && moments.evolve_ppar
        @loop_s_r_z_vperp is ir iz ivperp begin
            @views @. vpa.scratch = f_in.pdf[:,ivperp,iz,ir,is] - 
                            exp(-((vpa.grid[:])^2 + (vperp.grid[ivperp])^2) )
            second_derivative!(vpa.scratch2, vpa.scratch, vpa, spectral)
            @views @. f_out[:,ivperp,iz,ir,is] += dt * diffusion_coefficient * vpa.scratch2
        end
    elseif moments.evolve_density && moments.evolve_upar
        @loop_s_r_z_vperp is ir iz ivperp begin
            vth = moments.ion.vth[iz,ir,is]
            @views @. vpa.scratch = f_in.pdf[:,ivperp,iz,ir,is] - 
                            1.0 / vth * exp(- ((vpa.grid[:])^2 + (vperp.grid[ivperp])^2)/(vth^2) )
            second_derivative!(vpa.scratch2, vpa.scratch, vpa, spectral)
            @views @. f_out[:,ivperp,iz,ir,is] += dt * diffusion_coefficient * vpa.scratch2
        end
    elseif moments.evolve_density && moments.evolve_ppar
        @loop_s_r_z_vperp is ir iz ivperp begin
            upar = f_in.upar[iz,ir,is]
            @views @. vpa.scratch = f_in.pdf[:,ivperp,iz,ir,is] - 
                            exp(- ((vpa.grid[:] - upar)^2 + (vperp.grid[ivperp])^2))
            second_derivative!(vpa.scratch2, vpa.scratch, vpa, spectral)
            @views @. f_out[:,ivperp,iz,ir,is] += dt * diffusion_coefficient * vpa.scratch2
        end
    elseif moments.evolve_upar && moments.evolve_ppar
        @loop_s_r_z_vperp is ir iz ivperp begin
            n = f_in.density[iz,ir,is]
            @views @. vpa.scratch = f_in.pdf[:,ivperp,iz,ir,is] - 
                            n * exp(- ((vpa.grid[:])^2 + (vperp.grid[ivperp])^2) )
            second_derivative!(vpa.scratch2, vpa.scratch, vpa, spectral)
            @views @. f_out[:,ivperp,iz,ir,is] += dt * diffusion_coefficient * vpa.scratch2
        end
    elseif moments.evolve_density
        @loop_s_r_z_vperp is ir iz ivperp begin
            vth = moments.ion.vth[iz,ir,is]
            upar = f_in.upar[iz,ir,is]
            @views @. vpa.scratch = f_in.pdf[:,ivperp,iz,ir,is] - 
                            1.0 / vth * exp(- ((vpa.grid[:] - upar)^2 + (vperp.grid[ivperp])^2)/(vth^2) )
            second_derivative!(vpa.scratch2, vpa.scratch, vpa, spectral)
            @views @. f_out[:,ivperp,iz,ir,is] += dt * diffusion_coefficient * vpa.scratch2
        end
    elseif moments.evolve_upar
        @loop_s_r_z_vperp is ir iz ivperp begin
            vth = moments.ion.vth[iz,ir,is]
            n = f_in.density[iz,ir,is]
            @views @. vpa.scratch = f_in.pdf[:,ivperp,iz,ir,is] - 
                            n / vth * exp(- ((vpa.grid[:])^2 + (vperp.grid[ivperp])^2)/(vth^2) )
            second_derivative!(vpa.scratch2, vpa.scratch, vpa, spectral)
            @views @. f_out[:,ivperp,iz,ir,is] += dt * diffusion_coefficient * vpa.scratch2
        end
    elseif moments.evolve_ppar
        @loop_s_r_z_vperp is ir iz ivperp begin
            n = f_in.density[iz,ir,is]
            upar = f_in.upar[iz,ir,is]
            @views @. vpa.scratch = f_in.pdf[:,ivperp,iz,ir,is] - 
                            n * exp(- ((vpa.grid[:] - upar)^2 + (vperp.grid[ivperp])^2) )
            second_derivative!(vpa.scratch2, vpa.scratch, vpa, spectral)
            @views @. f_out[:,ivperp,iz,ir,is] += dt * diffusion_coefficient * vpa.scratch2
        end
    else
        # Drift kinetic version is the only one that currently can support 2V. 
        @loop_s_r_z_vperp is ir iz ivperp begin
            n = f_in.density[iz,ir,is]
            upar = f_in.upar[iz,ir,is]
            vth = moments.ion.vth[iz,ir,is]
            if vperp.n == 1
                vth_prefactor = 1.0 / vth
            else
                vth_prefactor = 1.0 / vth^3
            end
            @views @. vpa.scratch = f_in.pdf[:,ivperp,iz,ir,is] - n * vth_prefactor * 
                            exp(-( ((vpa.grid[:] - upar)^2) + (vperp.grid[ivperp])^2)/(vth^2) )
            second_derivative!(vpa.scratch2, vpa.scratch, vpa, spectral)
            @views @. f_out[:,ivperp,iz,ir,is] += dt * diffusion_coefficient * vpa.scratch2
        end
    end
    return nothing
end

"""
Calculate the Maxwellian associated with the current neutral pdf moments, and then 
subtract this from current pdf. Then take second derivative of this function
to act as the diffusion operator. 
"""
@timeit global_timer neutral_vz_maxwell_diffusion!(
                         f_out, f_in, moments, vzeta, vr, vz, spectral::T_spectral, dt,
                         diffusion_coefficient) where T_spectral = begin
    
    # If negative input (should be -1.0), then none of this diffusion will happen. 
    # This number can be put in as some parameter in the input file called something
    # like 'maxwellian_diffusion_coefficient'
    if diffusion_coefficient <= 0.0 || vz.n == 1
        return nothing
    end

    if vr.n > 1 && (moments.evolve_density || moments.evolve_upar || moments.evolve_ppar)
        error("Maxwell diffusion not implemented for 2V moment-kinetic cases yet")
    end

    # Otherwise, build the maxwellian function (which is going to be subtracted from 
    # the current distribution) using the moments of the distribution (so that the 
    # operator itself conserves the moments), and then this result will be the one 
    # whose second derivative will be added to the RHS (i.e. subtracted from the current pdf)
    begin_sn_r_z_vzeta_vr_region()


    if moments.evolve_ppar && moments.evolve_upar
        # See similar comments in krook_collisions! function. 
        @loop_sn_r_z_vzeta_vr isn ir iz ivzeta ivr begin
            @views @. vz.scratch = f_in.pdf_neutral[:,ivr,ivzeta,iz,ir,isn] - 
                            exp(-((vz.grid[:])^2 + (vr.grid[ivr])^2 + (vzeta.grid[ivzeta])^2) )
            second_derivative!(vz.scratch2, vz.scratch, vz, spectral)
            @views @. f_out[:,ivr,ivzeta,iz,ir,isn] += dt * diffusion_coefficient * vz.scratch2
        end
    elseif moments.evolve_ppar
        @loop_sn_r_z_vzeta_vr isn ir iz ivzeta ivr begin
            vth = moments.neutral.vth[iz,ir,isn]
            uz = f_in.uz_neutral[iz,ir,isn]
            @views @. vz.scratch = f_in.pdf_neutral[:,ivr,ivzeta,iz,ir,isn] - 
                            exp(- ((vz.grid[:] - uz)^2 + (vr.grid[ivr])^2 + (vzeta.grid[ivzeta])^2)/(vth^2) )
            second_derivative!(vz.scratch2, vz.scratch, vz, spectral)
            @views @. f_out[:,ivr,ivzeta,iz,ir,isn] += dt * diffusion_coefficient * vz.scratch2
        end
    elseif moments.evolve_upar
        @loop_sn_r_z_vzeta_vr isn ir iz ivzeta ivr begin
            vth = moments.neutral.vth[iz,ir,isn]
            @views @. vz.scratch = f_in.pdf_neutral[:,ivr,ivzeta,iz,ir,isn] - 
                            1.0 / vth * exp(- ((vz.grid[:])^2 + (vr.grid[ivr])^2 + (vzeta.grid[ivzeta])^2)/(vth^2) )
            second_derivative!(vz.scratch2, vz.scratch, vz, spectral)
            @views @. f_out[:,ivr,ivzeta,iz,ir,isn] += dt * diffusion_coefficient * vz.scratch2
        end
    elseif moments.evolve_density
        @loop_sn_r_z_vzeta_vr isn ir iz ivzeta ivr begin
            vth = moments.neutral.vth[iz,ir,isn]
            uz = f_in.uz_neutral[iz,ir,isn]
            @views @. vz.scratch = f_in.pdf_neutral[:,ivr,ivzeta,iz,ir,isn] - 
                            1.0 / vth * exp(- ((vz.grid[:] - uz)^2 + 
                                            (vr.grid[ivr])^2 + (vzeta.grid[ivzeta])^2)/(vth^2) )
            second_derivative!(vz.scratch2, vz.scratch, vz, spectral)
            @views @. f_out[:,ivr,ivzeta,iz,ir,isn] += dt * diffusion_coefficient * vz.scratch2
        end
    else 
        @loop_sn_r_z_vzeta_vr isn ir iz ivzeta ivr begin
            n = f_in.density_neutral[iz,ir,isn]
            uz = f_in.uz_neutral[iz,ir,isn]
            vth = moments.neutral.vth[iz,ir,isn]
            if vr.n == 1 && vzeta.n == 1
                vth_prefactor = 1.0 / vth
            else
                vth_prefactor = 1.0 / vth^3
            end
            @views @. vz.scratch = f_in.pdf_neutral[:,ivr,ivzeta,iz,ir,isn] - n * vth_prefactor * 
                            exp(-( (vz.grid[:] - uz)^2 + (vr.grid[ivr])^2 + (vzeta.grid[ivzeta])^2)/(vth^2) )
            second_derivative!(vz.scratch2, vz.scratch, vz, spectral)
            @views @. f_out[:,ivr,ivzeta,iz,ir,isn] += dt * diffusion_coefficient * vz.scratch2
        end
    end
    return nothing
end

end # maxwell_diffusion
