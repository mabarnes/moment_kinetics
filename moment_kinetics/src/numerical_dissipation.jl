"""
"""
module numerical_dissipation

export setup_numerical_dissipation, vpa_boundary_buffer_decay!,
       vpa_boundary_buffer_diffusion!, force_minimum_pdf_value!, force_minimum_pdf_value_neutral!,
        vpa_dissipation!, vperp_dissipation!, 
        z_dissipation!, r_dissipation!, 
        vz_dissipation_neutral!, z_dissipation_neutral!,
        r_dissipation_neutral! 

using Base.Iterators: flatten

using ..looping
using ..calculus: derivative!, second_derivative!, laplacian_derivative!
using ..derivatives: derivative_r!, derivative_z!, second_derivative_r!,
                     second_derivative_z!
using ..input_structs
using ..timer_utils
using ..type_definitions: mk_float, mk_int

# define individual structs for each species with their particular parameters
Base.@kwdef struct ion_num_diss_params
    vpa_boundary_buffer_damping_rate::mk_float = -1.0
    vpa_boundary_buffer_diffusion_coefficient::mk_float = -1.0
    vpa_dissipation_coefficient::mk_float = -1.0
    vperp_dissipation_coefficient::mk_float = -1.0
    z_dissipation_coefficient::mk_float = -1.0
    z_dissipation_degree::mk_int = 2
    r_dissipation_coefficient::mk_float = -1.0
    moment_dissipation_coefficient::mk_float = -1.0
    force_minimum_pdf_value::mk_float = -Inf
end

Base.@kwdef struct electron_num_diss_params
    vpa_boundary_buffer_damping_rate::mk_float = -1.0
    vpa_boundary_buffer_diffusion_coefficient::mk_float = -1.0
    vpa_dissipation_coefficient::mk_float = -1.0
    vperp_dissipation_coefficient::mk_float = -1.0
    z_dissipation_coefficient::mk_float = -1.0
    r_dissipation_coefficient::mk_float = -1.0
    moment_dissipation_coefficient::mk_float = -1.0
    force_minimum_pdf_value::mk_float = -Inf
end

Base.@kwdef struct neutral_num_diss_params
    vz_dissipation_coefficient::mk_float = -1.0
    z_dissipation_coefficient::mk_float = -1.0
    r_dissipation_coefficient::mk_float = -1.0
    moment_dissipation_coefficient::mk_float = -1.0
    force_minimum_pdf_value::mk_float = -Inf
end

struct numerical_dissipation_parameters
    ion::ion_num_diss_params
    electron::electron_num_diss_params
    neutral::neutral_num_diss_params
end

#############################################################
########### Numerical Dissipation Parameter setup ###########
"""
Define the dissipation parameters for each species, which means
there need to be three sections in each input file that specify
the parameters required of each species, as follows:

```
[ion_numerical_dissipation]
vpa_dissipation_coefficient
...

[electron_numerical_dissipation]
vpa_dissipation_coefficient
...

[neutral_numerical_dissipation]
vz_dissipation_coefficient
...
```

There will still be the -1.0 default parameters.
"""
function setup_numerical_dissipation(input_dict, warn_unexpected::Bool)
    ion_params = set_defaults_and_check_section!(
        input_dict, ion_num_diss_params, warn_unexpected, "ion_numerical_dissipation"
       )
    electron_params = set_defaults_and_check_section!(
        input_dict, electron_num_diss_params, warn_unexpected,
        "electron_numerical_dissipation"
       )
    neutral_params = set_defaults_and_check_section!(
        input_dict, neutral_num_diss_params, warn_unexpected,
        "neutral_numerical_dissipation"
       )

    return numerical_dissipation_parameters(ion_params, electron_params, neutral_params)
end

"""
Suppress the distribution function by damping towards a Maxwellian in the last element
before the vpa boundaries, to avoid numerical instabilities there.

Disabled by default.

The damping rate is set in the input TOML file by the parameter
```
[ion_numerical_dissipation]
vpa_boundary_buffer_damping_rate = 0.1
```
"""
@timeit global_timer vpa_boundary_buffer_decay!(
                         f_out, fvec_in, moments, vpa, dt, damping_rate_prefactor) = begin

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

    @begin_s_r_z_region()

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
[ion_numerical_dissipation]
vpa_boundary_buffer_diffusion_coefficient = 0.1
```
"""
@timeit global_timer vpa_boundary_buffer_diffusion!(
                         f_out, fvec_in, vpa, vpa_spectral, dt,
                         diffusion_prefactor) = begin

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

    @begin_s_r_z_region()

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
@timeit global_timer vpa_boundary_force_decreasing!(f_out, vpa) = begin
    @begin_s_r_z_region()

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
[ion_numerical_dissipation]
vpa_dissipation_coefficient = 0.1
```
"""
@timeit global_timer vpa_dissipation!(
                         f_out, f_in, vpa, spectral, dt, diffusion_coefficient) = begin

    if diffusion_coefficient <= 0.0 || vpa.n == 1
        return nothing
    end

    @begin_s_r_z_vperp_region()

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
        @views second_derivative!(vpa.scratch, f_in[:,ivperp,iz,ir,is], vpa, spectral)
        @views @. f_out[:,ivperp,iz,ir,is] += dt * diffusion_coefficient * vpa.scratch
    end
    return nothing
end

"""
Add diffusion in the vperp direction to suppress oscillations

Disabled by default.

The diffusion coefficient is set in the input TOML file by the parameter
```
[ion_numerical_dissipation]
vperp_dissipation_coefficient = 0.1
```
"""
@timeit global_timer vperp_dissipation!(
                         f_out, f_in, vperp, spectral, dt, diffusion_coefficient) = begin
    
    if diffusion_coefficient <= 0.0 || vperp.n == 1
        return nothing
    end
    
    @begin_s_r_z_vpa_region()

    @loop_s_r_z_vpa is ir iz ivpa begin
        @views laplacian_derivative!(vperp.scratch, f_in[ivpa,:,iz,ir,is], vperp, spectral)
        @views @. f_out[ivpa,:,iz,ir,is] += dt * diffusion_coefficient * vperp.scratch
    end

    return nothing
end

"""
Add diffusion in the z direction to suppress oscillations, with derivatives of 
degree n.

Disabled by default.

The diffusion coefficient and degree are set in the input TOML file by the parameter
```
[ion_numerical_dissipation]
z_dissipation_coefficient = 0.1
z_dissipation_degree = 2
```

Note that the current distributed-memory compatible
implementation does not impose a penalisation term
on internal or external element boundaries

"""
@timeit global_timer z_dissipation!(
                         f_out, f_in, z, z_spectral, dt, diffusion_coefficient,
                         degree, scratch_dummy) = begin

    if diffusion_coefficient <= 0.0 || z.n == 1
        return nothing
    end

    @begin_s_r_vperp_vpa_region()

    if degree == 2
        # calculate d^2 f / d z^2 using distributed memory compatible routines
        second_derivative_z!(scratch_dummy.buffer_vpavperpzrs_1, f_in,
                            scratch_dummy.buffer_vpavperprs_1,
                            scratch_dummy.buffer_vpavperprs_2,
                            scratch_dummy.buffer_vpavperprs_3,scratch_dummy.buffer_vpavperprs_4,
                            z_spectral,z)

    elseif degree == 4
        # calculate d^2 f / d z^2 using distributed memory compatible routines
        second_derivative_z!(scratch_dummy.buffer_vpavperpzrs_2, f_in,
                            scratch_dummy.buffer_vpavperprs_1,
                            scratch_dummy.buffer_vpavperprs_2,
                            scratch_dummy.buffer_vpavperprs_3,scratch_dummy.buffer_vpavperprs_4,
                            z_spectral,z)
        # calculate d^4 f / d z^4
        second_derivative_z!(scratch_dummy.buffer_vpavperpzrs_1, scratch_dummy.buffer_vpavperpzrs_2,
                            scratch_dummy.buffer_vpavperprs_1,
                            scratch_dummy.buffer_vpavperprs_2,
                            scratch_dummy.buffer_vpavperprs_3,scratch_dummy.buffer_vpavperprs_4,
                            z_spectral,z)
    else
        error("Only degree 2 and 4 dissipation implemented, got degree = $degree")
    end

    # advance f due to diffusion_coefficient * d^n f / d z^n where n is the degree of dissipation
    @loop_s_r_vperp_vpa is ir ivperp ivpa begin
        @views @. f_out[ivpa,ivperp,:,ir,is] += dt * diffusion_coefficient * scratch_dummy.buffer_vpavperpzrs_1[ivpa,ivperp,:,ir,is]
    end

    return nothing
end

"""
Add diffusion in the r direction to suppress oscillations

Disabled by default.

The diffusion coefficient is set in the input TOML file by the parameter
```
[ion_numerical_dissipation]
r_dissipation_coefficient = 0.1

```

Note that the current distributed-memory compatible
implementation does not impose a penalisation term
on internal or external element boundaries

"""
@timeit global_timer r_dissipation!(
                         f_out, f_in, r, r_spectral, dt, diffusion_coefficient,
                         scratch_dummy) = begin

    if diffusion_coefficient <= 0.0 || r.n == 1
        return nothing
    end

    @begin_s_z_vperp_vpa_region()

    # calculate d^2 f / d r^2 using distributed memory compatible routines
    second_derivative_r!(scratch_dummy.buffer_vpavperpzrs_1, f_in,
                         scratch_dummy.buffer_vpavperpzs_1,
                         scratch_dummy.buffer_vpavperpzs_2,
                         scratch_dummy.buffer_vpavperpzs_3,scratch_dummy.buffer_vpavperpzs_4,
                         r_spectral,r)
    # advance f due to diffusion_coefficient * d^2 f / d r^2
    @loop_s_z_vperp_vpa is iz ivperp ivpa begin
        @views @. f_out[ivpa,ivperp,iz,:,is] += dt * diffusion_coefficient * scratch_dummy.buffer_vpavperpzrs_1[ivpa,ivperp,iz,:,is]
    end

    return nothing
end

"""
Add diffusion in the vz direction to suppress oscillations for neutrals

Disabled by default.

The diffusion coefficient is set in the input TOML file by the parameter
```
[neutral_numerical_dissipation]
vz_dissipation_coefficient = 0.1
```
"""
@timeit global_timer vz_dissipation_neutral!(
                         f_out, f_in, vz, spectral, dt, diffusion_coefficient) = begin

    if diffusion_coefficient <= 0.0
        return nothing
    end

    @begin_sn_r_z_vzeta_vr_region()

    @loop_sn_r_z_vzeta_vr isn ir iz ivzeta ivr begin
        @views second_derivative!(vz.scratch, f_in[:,ivr,ivzeta,iz,ir,isn], vz, spectral)
        @views @. f_out[:,ivr,ivzeta,iz,ir,isn] += dt * diffusion_coefficient * vz.scratch
    end

    return nothing
end

"""
Add diffusion in the z direction to suppress oscillations for neutrals

Disabled by default.

The diffusion coefficient is set in the input TOML file by the parameter
```
[neutral_numerical_dissipation]
z_dissipation_coefficient = 0.1
```

Note that the current distributed-memory compatible
implementation does not impose a penalisation term
on internal or external element boundaries

"""
@timeit global_timer z_dissipation_neutral!(
                         f_out, f_in, z, z_spectral, dt, diffusion_coefficient,
                         scratch_dummy) = begin

    if diffusion_coefficient <= 0.0
        return nothing
    end

    @begin_sn_r_vzeta_vr_vz_region()

    # calculate d^2 f / d z^2  using distributed memory compatible routines
    second_derivative_z!(scratch_dummy.buffer_vzvrvzetazrsn_1, f_in,
                         scratch_dummy.buffer_vzvrvzetarsn_1,
                         scratch_dummy.buffer_vzvrvzetarsn_2,
                         scratch_dummy.buffer_vzvrvzetarsn_3,
                         scratch_dummy.buffer_vzvrvzetarsn_4, z_spectral, z)
    # advance f due to diffusion_coefficient * d^2 f/ d z^2
    @loop_sn_r_vzeta_vr_vz isn ir ivzeta ivr ivz begin
        @views @. f_out[ivz,ivr,ivzeta,:,ir,isn] += dt * diffusion_coefficient * scratch_dummy.buffer_vzvrvzetazrsn_1[ivz,ivr,ivzeta,:,ir,isn]
    end

    return nothing
end

"""
Add diffusion in the r direction to suppress oscillations for neutrals

Disabled by default.

The diffusion coefficient is set in the input TOML file by the parameter
```
[neutral_numerical_dissipation]
r_dissipation_coefficient = 0.1

```

Note that the current distributed-memory compatible
implementation does not impose a penalisation term
on internal or external element boundaries

"""
@timeit global_timer r_dissipation_neutral!(
                         f_out, f_in, r, r_spectral, dt, diffusion_coefficient,
                         scratch_dummy) = begin

    if diffusion_coefficient <= 0.0 || r.n == 1
        return nothing
    end

    @begin_sn_z_vzeta_vr_vz_region()

    # calculate d^2 f/ d r^2 using distributed memory compatible routines
    second_derivative_r!(scratch_dummy.buffer_vzvrvzetazrsn_1, f_in,
                         scratch_dummy.buffer_vzvrvzetazsn_1,
                         scratch_dummy.buffer_vzvrvzetazsn_2,
                         scratch_dummy.buffer_vzvrvzetazsn_3,scratch_dummy.buffer_vzvrvzetazsn_4,
                         r_spectral,r)
    # advance f due to diffusion_coefficient * d / d r ( Q d f / d r )
    @loop_sn_z_vzeta_vr_vz isn iz ivzeta ivr ivz begin
        @views @. f_out[ivz,ivr,ivzeta,iz,:,isn] += dt * diffusion_coefficient * scratch_dummy.buffer_vzvrvzetazrsn_1[ivz,ivr,ivzeta,iz,:,isn]
    end

    return nothing
end

"""
    force_minimum_pdf_value!(f, minval)

Set a minimum value for the pdf-sized array `f`. Any points less than the minimum are
set to the minimum. By default, no minimum is applied. The minimum value can be set by
```
[ion_numerical_dissipation]
force_minimum_pdf_value = 0.0
```
"""
@timeit global_timer force_minimum_pdf_value!(f, minval) = begin

    if minval == -Inf
        return nothing
    end

    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        if f[ivpa,ivperp,iz,ir,is] < minval
            f[ivpa,ivperp,iz,ir,is] = minval
        end
    end

    return nothing
end

"""
    force_minimum_pdf_value_neutral!(f, minval)

Set a minimum value for the neutral-pdf-sized array `f`. Any points less than the minimum
are set to the minimum. By default, no minimum is applied. The minimum value can be set by
```
[neutral_numerical_dissipation]
force_minimum_pdf_value = 0.0
```
"""
@timeit global_timer force_minimum_pdf_value_neutral!(f, minval) = begin

    if minval === nothing
        return nothing
    end

    @loop_sn_r_z_vzeta_vr_vz isn ir iz ivzeta ivr ivz begin
        if f[ivz,ivr,ivzeta,iz,ir,isn] < minval
            f[ivz,ivr,ivzeta,iz,ir,isn] = minval
        end
    end

    return nothing
end

end
