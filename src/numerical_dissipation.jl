"""
"""
module numerical_dissipation

export vpa_dissipation!

using ..looping
using ..calculus: derivative!

"""
Add diffusion in the vpa direction to suppress oscillations
"""
function vpa_dissipation!(f_out, fvec_in, moments, vpa, spectral, dt)
    begin_s_r_z_region()

    diffusion_coefficient = -1.0

    if diffusion_coefficient <= 0.0
        return nothing
    end

    @loop_s_r_z is ir iz begin
        # Don't want to dissipate the fluid moments, so divide out the Maxwellian, then
        # diffuse the result, i.e.
        # df/dt += diffusion_coefficient * f_M d2(f/f_M)/dvpa2
        # Store f_M in vpa.scratch
        if moments.evolve_ppar && moments.evolve_upar
            @views @. vpa.scratch = exp(-vpa.grid^2)
        elseif moments.evolve_ppar
            vth = sqrt(2.0*fvec_in.ppar[iz,ir,is]/fvec_in.density[iz,ir,is])
            @views @. vpa.scratch = exp(-(vpa.grid - fvec_in.upar[iz,ir,is]/vth)^2)
        elseif moments.evolve_upar
            vth = sqrt(2.0*fvec_in.ppar[iz,ir,is]/fvec_in.density[iz,ir,is])
            @views @. vpa.scratch = exp(-(vpa.grid/vth)^2)
        elseif moments.evolve_density
            vth = sqrt(2.0*fvec_in.ppar[iz,ir,is]/fvec_in.density[iz,ir,is])
            @views @. vpa.scratch = exp(-((vpa.grid - fvec_in.upar[iz,ir,is])/vth)^2)
        else
            vth = sqrt(2.0*fvec_in.ppar[iz,ir,is]/fvec_in.density[iz,ir,is])
            @views @. vpa.scratch = (fvec_in.density[iz,ir,is] *
                                     exp(-((vpa.grid - fvec_in.upar[iz,ir,is])/vth)^2))
        end
        @views @. vpa.scratch2 = fvec_in.pdf[:,iz,ir,is] / vpa.scratch
        derivative!(vpa.scratch3, vpa.scratch2, vpa, spectral, Val(2))
        @views @. f_out[:,iz,ir,is] += dt * diffusion_coefficient * vpa.scratch *
                                       vpa.scratch3
    end

    return nothing
end

end
