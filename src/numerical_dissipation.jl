"""
"""
module numerical_dissipation

export vpa_boundary_buffer!, vpa_dissipation!, z_dissipation!

using Base.Iterators: flatten

using ..looping
using ..calculus: derivative!

"""
Suppress the distribution function by damping towards a Maxwellian in the last element
before the vpa boundaries, to avoid numerical instabilities there.
"""
function vpa_boundary_buffer!(f_out, fvec_in, moments, vpa, dt)
    damping_rate_prefactor = -0.01 / dt

    if damping_rate_prefactor <= 0.0
        return nothing
    end

    if vpa.nelement > 2
        # Damping rate decays quadratically through the first/last elements
        # Hopefully this makes it smooth...
        # Note vpa is antisymmetric with vpa=0 in the centre of the grid, so the following
        # should work for both ends of the grid.
        @. vpa.scratch = damping_rate_prefactor *
                         (abs(vpa.grid) - abs(vpa.grid[vpa.ngrid])^2) /
                         (abs(vpa.grid[1]) - abs(vpa.grid[vpa.ngrid])^2)

        # Iterate over the first and last element in the vpa dimension
        vpa_inds = flatten((1:vpa.ngrid, vpa.n-vpa.ngrid+1:vpa.n))
    else
        # ≤2 elements, so applying a 'buffer' in the boundary elements would apply it
        # across the whole grid. Instead, hard-code a number of grid points to use as
        # the 'buffer'.
        nbuffer = 16

        @. vpa.scratch = damping_rate_prefactor *
                         (abs(vpa.grid) - abs(vpa.grid[nbuffer])^2) /
                         (abs(vpa.grid[1]) - abs(vpa.grid[nbuffer])^2)

        # Iterate over the first and last element in the vpa dimension
        vpa_inds = flatten((1:nbuffer, vpa.n-nbuffer+1:vpa.n))
    end

    begin_s_r_z_region()

    if moments.evolve_upar && moments.evolve_ppar
        @loop_s_r_z is ir iz begin
            for ivpa ∈ vpa_inds
                f_out[ivpa,iz,ir,is] += dt*vpa.scratch[ivpa]*
                                        (exp(-vpa.grid[ivpa]^2) - fvec_in.pdf[ivpa,iz,ir,is])
            end
        end
    elseif moments.evolve_ppar
        @loop_s_r_z is ir iz begin
            vth = sqrt(2.0*fvec_in.ppar[iz,ir,is]/fvec_in.density[iz,ir,is])
            for ivpa ∈ vpa_inds
                f_out[ivpa,iz,ir,is] += dt*vpa.scratch[ivpa]*
                                        (exp(-(vpa.grid[ivpa] -
                                               fvec_in.upar[iz,ir,is]/vth)^2) -
                                         fvec_in.pdf[ivpa,iz,ir,is])
            end
        end
    elseif moments.evolve_upar
        @loop_s_r_z is ir iz begin
            vth = sqrt(2.0*fvec_in.ppar[iz,ir,is]/fvec_in.density[iz,ir,is])
            for ivpa ∈ vpa_inds
                f_out[ivpa,iz,ir,is] += dt*vpa.scratch[ivpa]*
                                        (exp(-(vpa.grid[ivpa])^2)/vth -
                                         fvec_in.pdf[ivpa,iz,ir,is])
            end
        end
    elseif moments.evolve_density
        @loop_s_r_z is ir iz begin
            vth = sqrt(2.0*fvec_in.ppar[iz,ir,is]/fvec_in.density[iz,ir,is])
            for ivpa ∈ vpa_inds
                f_out[ivpa,iz,ir,is] += dt*vpa.scratch[ivpa]*
                                        (exp(-(vpa.grid[ivpa] -
                                               fvec_in.upar[iz,ir,is])^2)/vth -
                                         fvec_in.pdf[ivpa,iz,ir,is])
            end
        end
    else
        @loop_s_r_z is ir iz begin
            vth = sqrt(2.0*fvec_in.ppar[iz,ir,is]/fvec_in.density[iz,ir,is])
            for ivpa ∈ vpa_inds
                f_out[ivpa,iz,ir,is] += dt*vpa.scratch[ivpa]*
                                        (fvec_in.density[iz,ir,is]/vth*
                                         exp(-(vpa.grid[ivpa] -
                                               fvec_in.upar[iz,ir,is])^2)/vth -
                                         fvec_in.pdf[ivpa,iz,ir,is])
            end
        end
    end

    return nothing
end

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

"""
Add diffusion in the z direction to suppress oscillations
"""
function z_dissipation!(f_out, fvec_in, moments, z, vpa, spectral, dt)
    begin_s_r_vpa_region()

    diffusion_coefficient = -1.0

    if diffusion_coefficient <= 0.0
        return nothing
    end

    @loop_s_r_vpa is ir ivpa begin
        @views derivative!(z.scratch, fvec_in.pdf[ivpa,:,ir,is], z, spectral, Val(2))
        @views @. f_out[ivpa,:,ir,is] += dt * diffusion_coefficient * z.scratch
    end

    return nothing
end

end
