"""
"""
module numerical_dissipation

export setup_numerical_dissipation, vpa_boundary_buffer_decay!,
       vpa_boundary_buffer_diffusion!, vpa_dissipation!, z_dissipation!

using Base.Iterators: flatten

using ..looping
using ..calculus: derivative!
using ..type_definitions: mk_float

Base.@kwdef struct numerical_dissipation_parameters
    vpa_boundary_buffer_damping_rate::mk_float = -1.0
    vpa_boundary_buffer_diffusion_coefficient::mk_float = -1.0
    vpa_dissipation_coefficient::mk_float = -1.0
    z_dissipation_coefficient::mk_float = -1.0
    moment_dissipation_coefficient::mk_float = -1.0
    force_minimum_pdf_value::Union{Nothing,mk_float} = nothing
end

function setup_numerical_dissipation(input_section::Dict)
    input = Dict(Symbol(k)=>v for (k,v) in input_section)

    return numerical_dissipation_parameters(; input...)
end

"""
Suppress the distribution function by damping towards a Maxwellian in the last element
before the vpa boundaries, to avoid numerical instabilities there.

Disabled by default.

The damping rate is set in the input TOML file by the parameter
```
[numerical_dissipation]
vpa_boundary_buffer_damping_rate = 0.1
```
"""
function vpa_boundary_buffer_decay!(f_out, fvec_in, moments, vpa, dt,
                                    num_diss_params::numerical_dissipation_parameters)
    damping_rate_prefactor = num_diss_params.vpa_boundary_buffer_damping_rate

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
Suppress the distribution function by applying diffusion in the last element before the
vpa boundaries, to avoid numerical instabilities there.

Disabled by default.

The maximum diffusion rate in the buffer is set in the input TOML file by the parameter
```
[numerical_dissipation]
vpa_boundary_buffer_diffusion_coefficient = 0.1
```
"""
function vpa_boundary_buffer_diffusion!(f_out, fvec_in, vpa, vpa_spectral, dt,
                                        num_diss_params::numerical_dissipation_parameters)
    diffusion_prefactor = num_diss_params.vpa_boundary_buffer_diffusion_coefficient

    if diffusion_prefactor <= 0.0
        return nothing
    end

    if vpa.nelement > 2
        # Damping rate decays quadratically through the first/last elements
        # Hopefully this makes it smooth...
        # Note vpa is antisymmetric with vpa=0 in the centre of the grid, so the following
        # should work for both ends of the grid.
        @. vpa.scratch = diffusion_prefactor *
        (abs(vpa.grid) - abs(vpa.grid[vpa.ngrid])^2) /
        (abs(vpa.grid[1]) - abs(vpa.grid[vpa.ngrid])^2)

        # Iterate over the first and last element in the vpa dimension
        vpa_inds = flatten((1:vpa.ngrid, vpa.n-vpa.ngrid+1:vpa.n))
    else
        # ≤2 elements, so applying a 'buffer' in the boundary elements would apply it
        # across the whole grid. Instead, hard-code a number of grid points to use as
        # the 'buffer'.
        nbuffer = 16

        @. vpa.scratch = diffusion_prefactor *
        (abs(vpa.grid) - abs(vpa.grid[nbuffer])^2) /
        (abs(vpa.grid[1]) - abs(vpa.grid[nbuffer])^2)

        # Iterate over the first and last element in the vpa dimension
        vpa_inds = flatten((1:nbuffer, vpa.n-nbuffer+1:vpa.n))
    end

    begin_s_r_z_region()

    @loop_s_r_z is ir iz begin
        # Calculate second derivative
        @views derivative!(vpa.scratch2, fvec_in.pdf[:,iz,ir,is], vpa, vpa_spectral,
                           Val(2))
        for ivpa ∈ vpa_inds
            f_out[ivpa,iz,ir,is] += dt*vpa.scratch[ivpa]*vpa.scratch2[ivpa]
        end
    end

    return nothing
end

"""
Try to suppress oscillations near the boundary by ensuring that every point in the final
element is ≤ the innermost value. The distribution function should be decreasing near
the boundaries, so this should be an OK thing to force.

Note: not currently used.
"""
function vpa_boundary_force_decreasing!(f_out, vpa)
    begin_s_r_z_region()

    ngrid = vpa.ngrid
    n = vpa.n
    last_start = n - ngrid + 1
    @loop_s_r_z is ir iz begin
        # First element in vpa
        for ivpa ∈ 1:ngrid-1
            if f_out[ivpa,iz,ir,is] > f_out[ngrid,iz,ir,is]
                f_out[ivpa,iz,ir,is] = f_out[ngrid,iz,ir,is]
            end
        end
        # Last element in vpa
        for ivpa ∈ last_start+1:n
            if f_out[ivpa,iz,ir,is] > f_out[last_start,iz,ir,is]
                f_out[ivpa,iz,ir,is] = f_out[last_start,iz,ir,is]
            end
        end
    end

    return nothing
end

"""
Add diffusion in the vpa direction to suppress oscillations

Disabled by default.

The diffusion coefficient is set in the input TOML file by the parameter
```
[numerical_dissipation]
vpa_dissipation_coefficient = 0.1
```
"""
function vpa_dissipation!(f_out, fvec_in, moments, vpa, spectral::T_spectral, dt,
        num_diss_params::numerical_dissipation_parameters) where T_spectral
    begin_s_r_z_region()

    diffusion_coefficient = num_diss_params.vpa_dissipation_coefficient
    if diffusion_coefficient <= 0.0
        return nothing
    end

    # if T_spectral <: Bool
    #     # Scale diffusion coefficient like square of grid spacing, so convergence will
    #     # be second order accurate despite presence of numerical dissipation.
    #     # Assume constant grid spacing, so all cell_width entries are the same.
    #     diffusion_coefficient *= vpa.cell_width[1]^2
    # else
    #     # Dissipation should decrease with element size at order (ngrid-1) to preserve
    #     # expected convergence of Chebyshev pseudospectral scheme
    #     diffusion_coefficient *= (vpa.L/vpa.nelement)^(vpa.ngrid-1)
    # end

    @loop_s_r_z is ir iz begin
        # # Don't want to dissipate the fluid moments, so divide out the Maxwellian, then
        # # diffuse the result, i.e.
        # # df/dt += diffusion_coefficient * f_M d2(f/f_M)/dvpa2
        # # Store f_M in vpa.scratch
        # if (moments.evolve_ppar || moments.evolve_vth) && moments.evolve_upar
        #     @views @. vpa.scratch = exp(-vpa.grid^2)
        # elseif moments.evolve_ppar || moments.evolve_vth
        #     vth = sqrt(2.0*fvec_in.ppar[iz,ir,is]/fvec_in.density[iz,ir,is])
        #     @views @. vpa.scratch = exp(-(vpa.grid - fvec_in.upar[iz,ir,is]/vth)^2)
        # elseif moments.evolve_upar
        #     vth = sqrt(2.0*fvec_in.ppar[iz,ir,is]/fvec_in.density[iz,ir,is])
        #     @views @. vpa.scratch = exp(-(vpa.grid/vth)^2)
        # elseif moments.evolve_density
        #     vth = sqrt(2.0*fvec_in.ppar[iz,ir,is]/fvec_in.density[iz,ir,is])
        #     @views @. vpa.scratch = exp(-((vpa.grid - fvec_in.upar[iz,ir,is])/vth)^2)
        # else
        #     vth = sqrt(2.0*fvec_in.ppar[iz,ir,is]/fvec_in.density[iz,ir,is])
        #     @views @. vpa.scratch = (fvec_in.density[iz,ir,is] *
        #                              exp(-((vpa.grid - fvec_in.upar[iz,ir,is])/vth)^2))
        # end
        # @views @. vpa.scratch2 = fvec_in.pdf[:,iz,ir,is] / vpa.scratch
        # derivative!(vpa.scratch3, vpa.scratch2, vpa, spectral, Val(2))
        # @views @. f_out[:,iz,ir,is] += dt * diffusion_coefficient * vpa.scratch *
        #                                vpa.scratch3

        @views derivative!(vpa.scratch, fvec_in.pdf[:,iz,ir,is], vpa, spectral, Val(2))
        @views @. f_out[:,iz,ir,is] += dt * diffusion_coefficient * vpa.scratch
    end

    return nothing
end

"""
Add diffusion in the z direction to suppress oscillations

Disabled by default.

The diffusion coefficient is set in the input TOML file by the parameter
```
[numerical_dissipation]
z_dissipation_coefficient = 0.1
```
"""
function z_dissipation!(f_out, fvec_in, moments, z, vpa, spectral::T_spectral, dt,
        num_diss_params::numerical_dissipation_parameters) where T_spectral

    diffusion_coefficient = num_diss_params.z_dissipation_coefficient
    if diffusion_coefficient <= 0.0
        return nothing
    end

    begin_s_r_vpa_region()

    #if T_spectral <: Bool
    #    # Scale diffusion coefficient like square of grid spacing, so convergence will
    #    # be second order accurate despite presence of numerical dissipation.
    #    # Assume constant grid spacing, so all cell_width entries are the same.
    #    diffusion_coefficient *= z.cell_width[1]^2
    #else
    #    # Dissipation should decrease with element size at order (ngrid-1) to preserve
    #    # expected convergence of Chebyshev pseudospectral scheme
    #    diffusion_coefficient *= (z.L/z.nelement)^(z.ngrid-1)
    #end

    #@. z.scratch2 = 1.e-2 * (1.0 - (2.0*z.grid/z.L)^2)
    #diffusion_coefficient = z.scratch2

    @loop_s_r_vpa is ir ivpa begin
        @views derivative!(z.scratch, fvec_in.pdf[ivpa,:,ir,is], z, spectral, Val(2))
        @views @. f_out[ivpa,:,ir,is] += dt * diffusion_coefficient * z.scratch
    end

    return nothing
end

"""
    force_minimum_pdf_value!(f, num_diss_paras::numerical_dissipation_parameters)

Set a minimum value for the pdf-sized array `f`. Any points less than the minimum are
set to the minimum. By default, no minimum is applied. The minimum value can be set by
```
[numerical_dissipation]
force_minimum_pdf_value = 0.0
```
"""
function force_minimum_pdf_value!(f, num_diss_params::numerical_dissipation_parameters)
    minval = num_diss_params.force_minimum_pdf_value

    if minval === nothing
        return nothing
    end

    @loop_s_r_z_vpa is ir iz ivpa begin
        if f[ivpa,iz,ir,is] < minval
            f[ivpa,iz,ir,is] = minval
        end
    end

    return nothing
end

end
