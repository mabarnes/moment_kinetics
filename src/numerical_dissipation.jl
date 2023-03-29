"""
-"""
module numerical_dissipation

export setup_numerical_dissipation
       #vpa_boundary_buffer_decay!,
       #vpa_boundary_buffer_diffusion!, , z_dissipation!
export vpa_dissipation!
export r_dissipation!
export z_dissipation!

using Base.Iterators: flatten

using ..looping
using ..calculus: derivative!, second_derivative!
using ..derivatives: derivative_r!, derivative_z!
using ..type_definitions: mk_float

Base.@kwdef struct numerical_dissipation_parameters
    vpa_boundary_buffer_damping_rate::mk_float = -1.0
    vpa_boundary_buffer_diffusion_coefficient::mk_float = -1.0
    vpa_dissipation_coefficient::mk_float = -1.0
    z_dissipation_coefficient::mk_float = -1.0
    r_dissipation_coefficient::mk_float = -1.0
end

function setup_numerical_dissipation(input_section::Dict)
    input = Dict(Symbol(k)=>v for (k,v) in input_section)

    return numerical_dissipation_parameters(; input...)
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
function vpa_dissipation!(f_out, f_in, vpa, spectral::T_spectral, dt,
        num_diss_params::numerical_dissipation_parameters, z, z_advect) where T_spectral
    
    begin_s_r_z_vperp_region()

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
    @loop_s_r_z_vperp is ir iz ivperp begin
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
        vpa.scratch2 .= 1.0 # placeholder for Q in d / d vpa ( Q d f / d vpa)
        @views second_derivative!(vpa.scratch, f_in[:,ivperp,iz,ir,is], vpa.scratch2, vpa, spectral,
                               iz, z, z_advect[is].speed[iz,:,ivperp,ir])
        @views @. f_out[:,ivperp,iz,ir,is] += dt * diffusion_coefficient * vpa.scratch
    end

    return nothing
end

"""
Add diffusion in the z & r direction to suppress oscillations

Disabled by default or if negative value is set.

The diffusion coefficient is set in the input TOML file by the parameter
```
[numerical_dissipation]
z_dissipation_coefficient = 0.1
r_dissipation_coefficient = 0.1
```

Note that the current distributed-memory compatible
implementation does not impose a penalisation term
on internal or external element boundaries

"""
function z_dissipation!(f_out, f_in, z, z_spectral::T_spectral, dt,
        num_diss_params::numerical_dissipation_parameters, scratch_dummy) where T_spectral

    diffusion_coefficient = num_diss_params.z_dissipation_coefficient
    if diffusion_coefficient <= 0.0
        return nothing
    end

    begin_s_r_vperp_vpa_region()

    # calculate d / d z ( Q d f / d z ) using distributed memory compatible routines
    # first compute d f / d z using centred reconciliation and place in dummy array #1
    derivative_z!(scratch_dummy.buffer_vpavperpzrs_1, f_in[:,:,:,:,:],
					scratch_dummy.buffer_vpavperprs_1, scratch_dummy.buffer_vpavperprs_2,
					scratch_dummy.buffer_vpavperprs_3,scratch_dummy.buffer_vpavperprs_4,
					z_spectral,z)
    # form Q d f / d r and place in dummy array #2
    @loop_s_r_vperp_vpa is ir ivperp ivpa begin
        Q = 1.0 # placeholder for geometrical or velocity space dependent metric coefficient
        @. scratch_dummy.buffer_vpavperpzrs_2[ivpa,ivperp,:,ir,is] =  Q * scratch_dummy.buffer_vpavperpzrs_1[ivpa,ivperp,:,ir,is]
    end
    # compute d / d z ( Q d f / d z ) using centred reconciliation and place in dummy array #1
    derivative_z!(scratch_dummy.buffer_vpavperpzrs_1, scratch_dummy.buffer_vpavperpzrs_2[:,:,:,:,:],
					scratch_dummy.buffer_vpavperprs_1, scratch_dummy.buffer_vpavperprs_2,
					scratch_dummy.buffer_vpavperprs_3,scratch_dummy.buffer_vpavperprs_4,
					z_spectral,z)
    # advance f due to diffusion_coefficient * d / d z ( Q d f / d z )
    @loop_s_r_vperp_vpa is ir ivperp ivpa begin
        @views @. f_out[ivpa,ivperp,:,ir,is] += dt * diffusion_coefficient * scratch_dummy.buffer_vpavperpzrs_1[ivpa,ivperp,:,ir,is]
    end

    return nothing
end

function r_dissipation!(f_out, f_in, r, r_spectral::T_spectral, dt,
        num_diss_params::numerical_dissipation_parameters, scratch_dummy, z, z_advect::T) where {T_spectral, T}

    diffusion_coefficient = num_diss_params.r_dissipation_coefficient
    if diffusion_coefficient <= 0.0
        return nothing
    end

    begin_s_z_vperp_vpa_region()

    # calculate d / d r ( Q d f / d r ) using distributed memory compatible routines
    # first compute d f / d r using centred reconciliation and place in dummy array #1
    derivative_r!(scratch_dummy.buffer_vpavperpzrs_1, f_in[:,:,:,:,:],
					scratch_dummy.buffer_vpavperpzs_1, scratch_dummy.buffer_vpavperpzs_2,
					scratch_dummy.buffer_vpavperpzs_3,scratch_dummy.buffer_vpavperpzs_4,
					r_spectral,r,z,z_advect)
    # form Q d f / d r and place in dummy array #2
    @loop_s_z_vperp_vpa is iz ivperp ivpa begin
        Q = 1.0 # placeholder for geometrical or velocity space dependent metric coefficient
        @. scratch_dummy.buffer_vpavperpzrs_2[ivpa,ivperp,iz,:,is] =  Q * scratch_dummy.buffer_vpavperpzrs_1[ivpa,ivperp,iz,:,is]
    end
    # compute d / d r ( Q d f / d r ) using centred reconciliation and place in dummy array #1
    derivative_r!(scratch_dummy.buffer_vpavperpzrs_1, scratch_dummy.buffer_vpavperpzrs_2[:,:,:,:,:],
					scratch_dummy.buffer_vpavperpzs_1, scratch_dummy.buffer_vpavperpzs_2,
					scratch_dummy.buffer_vpavperpzs_3,scratch_dummy.buffer_vpavperpzs_4,
					r_spectral,r,z,z_advect)
    # advance f due to diffusion_coefficient * d / d r ( Q d f / d r )
    @loop_s_z_vperp_vpa is iz ivperp ivpa begin
        @views @. f_out[ivpa,ivperp,iz,:,is] += dt * diffusion_coefficient * scratch_dummy.buffer_vpavperpzrs_1[ivpa,ivperp,iz,:,is]
    end

    return nothing
end

end
