"""
Runge Kutta timestepping
"""
module runge_kutta

export setup_runge_kutta_coefficients!, rk_update_evolved_moments!,
       rk_update_evolved_moments_electron!, rk_update_evolved_moments_neutral!,
       rk_update_variable!, rk_error_variable!, local_error_norm

using ..array_allocation: allocate_float
using ..communication
using ..looping
using ..type_definitions: mk_float

using MPI
using StatsBase: mean

"""
given the number of Runge Kutta stages that are requested,
returns the needed Runge Kutta coefficients;
e.g., if f is the function to be updated, then
f^{n+1}[stage+1] = rk_coef[1,stage]*f^{n} + rk_coef[2,stage]*f^{n+1}[stage] + rk_coef[3,stage]*(f^{n}+dt*G[f^{n+1}[stage]]
"""
function setup_runge_kutta_coefficients!(type, input_CFL_prefactor, split_operators)
    if type == "RKF5(4)"
        # Embedded 5th order / 4th order Runge-Kutta-Fehlberg method.
        # Note uses the 5th order solution for the time advance, even though the error
        # estimate is for the 4th order solution.
        #
        # Coefficients originate here:
        # https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method,
        # 'COEFFICIENTS FOR RK4(5), FORMULA 2 Table III in Fehlberg'
        #
        # Coefficients converted to the format for moment_kinetics time-stepper using
        # `util/calculate_rk_coeffs.jl`
        rk_coefs = mk_float[3//4 5//8   10469//2197  115//324        121//240    641//1980  11//36   ;
                            1//4 3//32  17328//2197  95//54          33//10      232//165   4//3     ;
                            0    9//32 -32896//2197 -95744//29241   -1408//285  -512//171  -512//171 ;
                            0    0      7296//2197   553475//233928  6591//1520  2197//836  2197//836;
                            0    0      0           -845//4104      -77//40     -56//55    -1        ;
                            0    0      0            0              -11//40      34//55     8//11    ;
                            0    0      0            0               0           2//55     -1        ]
        n_rk_stages = 6
        rk_order = 5
        adaptive = true
        low_storage = false
        if input_CFL_prefactor ≤ 0.0
            CFL_prefactor = 1.0
        else
            CFL_prefactor = input_CFL_prefactor
        end
    elseif type == "Fekete10(4)"
        # Fekete 10-stage 4th-order SSPRK (see comments in util/calculate_rk_coeffs.jl.
        # Note that a 'low storage' implementation of the main method (if not the
        # truncation error estimate) is possible [D.I. Ketcheson, Highly efficient strong
        # stability-preserving Runge–Kutta methods with low-storage implementations, SIAM
        # J. Sci. Comput. 30 (2008) 2113–2136, https://doi.org/10.1137/07070485X,
        # https://www.davidketcheson.info/assets/papers/2008_explicit_ssp.pdf] but
        # would require a particular implementation that does not fit in with the
        # currently-implemented moment_kinetics 'low_storage' code, so we do not take
        # advantage of it yet. If this timestepping scheme turns out to be particularly
        # efficient, a low-storage version could be implemented (which might be
        # particularly important given the large number of stages in this scheme which
        # will lead to high memory usage), with one extra buffer for the truncation error
        # estimate which would need to be updated incrementally at each stage, rather than
        # calculated only at the end of the RK step.
        rk_coefs = mk_float[5//6 0    0    0    3//5  0    0    0    0    -1//2  -1//5;
                            1//6 5//6 0    0    0     0    0    0    0     0      6//5;
                            0    1//6 5//6 0    0     0    0    0    0     0      0   ;
                            0    0    1//6 5//6 0     0    0    0    0     0     -9//5;
                            0    0    0    1//6 1//3  0    0    0    0     0      9//5;
                            0    0    0    0    1//15 5//6 0    0    0     9//10  0   ;
                            0    0    0    0    0     1//6 5//6 0    0     0     -6//5;
                            0    0    0    0    0     0    1//6 5//6 0     0      6//5;
                            0    0    0    0    0     0    0    1//6 5//6  0     -9//5;
                            0    0    0    0    0     0    0    0    1//6  1//2   9//5;
                            0    0    0    0    0     0    0    0    0     1//10 -1   ]
        n_rk_stages = 10
        rk_order = 4
        adaptive = true
        low_storage = false
        if input_CFL_prefactor ≤ 0.0
            CFL_prefactor = 12.0
        else
            CFL_prefactor = input_CFL_prefactor
        end
    elseif type == "Fekete6(4)"
        # Fekete 6-stage 4th-order SSPRK (see comments in util/calculate_rk_coeffs.jl.
        # Note Fekete et al. recommend the 10-stage method rather than this one.
        #rk_coeffs = mk_float[0.6447024483081 0.2386994475333264  0.5474858792272213     0.3762853856474131     0.0                -0.18132326703443313    -0.0017300417984673078;
        #                     0.3552975516919 0.4295138541066736 -6.461498003318411e-14 -1.1871059690804486e-13 0.0                 2.9254376698872875e-14 -0.18902907903375094  ;
        #                     0.0             0.33178669836       0.25530138316744333   -3.352873534367973e-14  0.0                 0.2059808002676668      0.2504712436879622   ;
        #                     0.0             0.0                 0.1972127376054        0.3518900216285391     0.0                 0.4792670116241715     -0.9397479180374522   ;
        #                     0.0             0.0                 0.0                    0.2718245927242        0.5641843457422999  9.986456106503283e-14   1.1993626679930305   ;
        #                     0.0             0.0                 0.0                    0.0                    0.4358156542577     0.3416567872695656     -0.5310335716309745   ;
        #                     0.0             0.0                 0.0                    0.0                    0.0                 0.1544186678729         0.2117066988196524   ]
        # Might as well set to 0 the entries that look like they should be 0 apart from
        # rounding errors.
        rk_coefs = mk_float[0.6447024483081 0.2386994475333264 0.5474858792272213  0.3762853856474131 0.0                -0.18132326703443313    -0.0017300417984673078;
                            0.3552975516919 0.4295138541066736 0.0                 0.0                0.0                 0.0                    -0.18902907903375094  ;
                            0.0             0.33178669836      0.25530138316744333 0.0                0.0                 0.2059808002676668      0.2504712436879622   ;
                            0.0             0.0                0.1972127376054     0.3518900216285391 0.0                 0.4792670116241715     -0.9397479180374522   ;
                            0.0             0.0                0.0                 0.2718245927242    0.5641843457422999  0.0                     1.1993626679930305   ;
                            0.0             0.0                0.0                 0.0                0.4358156542577     0.3416567872695656     -0.5310335716309745   ;
                            0.0             0.0                0.0                 0.0                0.0                 0.1544186678729         0.2117066988196524   ]
        n_rk_stages = 6
        rk_order = 4
        adaptive = true
        low_storage = false
        if input_CFL_prefactor ≤ 0.0
            CFL_prefactor = 8.0
        else
            CFL_prefactor = input_CFL_prefactor
        end
    elseif type == "Fekete4(3)"
        # Fekete 4-stage, 3rd-order SSPRK (see comments in util/calculate_rk_coeffs.jl).
        # Note this is the same as moment_kinetics original 4-stage SSPRK method, with
        # the addition of a truncation error estimate.
        rk_coefs = mk_float[1//2 0    2//3 0    -1//2;
                            0    1//2 1//6 1//2  1   ;
                            1//2 1//2 1//6 1//2 -1//2]
        n_rk_stages = 4
        rk_order = 3
        adaptive = true
        low_storage = true
        if input_CFL_prefactor ≤ 0.0
            CFL_prefactor = 6.0
        else
            CFL_prefactor = input_CFL_prefactor
        end
    elseif type == "Fekete4(2)"
        # Fekete 4-stage 2nd-order SSPRK (see comments in util/calculate_rk_coeffs.jl.
        rk_coefs = mk_float[2//3 0    0    1//4 -1//8 ;
                            1//3 2//3 0    0     3//16;
                            0    1//3 2//3 0     0    ;
                            0    0    1//3 1//2  3//16;
                            0    0    0    1//4 -1//4 ]
        n_rk_stages = 4
        rk_order = 2
        adaptive = true
        low_storage = false
        if input_CFL_prefactor ≤ 0.0
            CFL_prefactor = 7.0
        else
            CFL_prefactor = input_CFL_prefactor
        end
    elseif type == "SSPRK4"
        n_rk_stages = 4
        rk_coefs = allocate_float(3, n_rk_stages)
        rk_coefs .= 0.0
        rk_coefs[1,1] = 0.5
        rk_coefs[3,1] = 0.5
        rk_coefs[2,2] = 0.5
        rk_coefs[3,2] = 0.5
        rk_coefs[1,3] = 2.0/3.0
        rk_coefs[2,3] = 1.0/6.0
        rk_coefs[3,3] = 1.0/6.0
        rk_coefs[2,4] = 0.5
        rk_coefs[3,4] = 0.5
        n_rk_stages = 4
        rk_order = 3
        adaptive = false
        low_storage = true
        CFL_prefactor = NaN
    elseif type == "SSPRK3"
        n_rk_stages = 3
        rk_coefs = allocate_float(3, n_rk_stages)
        rk_coefs .= 0.0
        rk_coefs[3,1] = 1.0
        rk_coefs[1,2] = 0.75
        rk_coefs[3,2] = 0.25
        rk_coefs[1,3] = 1.0/3.0
        rk_coefs[3,3] = 2.0/3.0
        rk_order = 3 # ? Not sure about this order
        adaptive = false
        low_storage = true
        CFL_prefactor = NaN
    elseif type == "SSPRK2"
        n_rk_stages = 2
        rk_coefs = allocate_float(3, n_rk_stages)
        rk_coefs .= 0.0
        rk_coefs[3,1] = 1.0
        rk_coefs[1,2] = 0.5
        rk_coefs[3,2] = 0.5
        rk_order = 2
        adaptive = false
        low_storage = true
        CFL_prefactor = NaN
    elseif type == "SSPRK1"
        n_rk_stages = 1
        rk_coefs = allocate_float(3, n_rk_stages)
        rk_coefs .= 0.0
        rk_coefs[3,1] = 1.0
        rk_order = 1
        adaptive = false
        low_storage = true
        CFL_prefactor = NaN
    else
        error("Unsupported RK timestep method, type=$type\n"
              * "Valid methods are: SSPRK4, SSPRK3, SSPRK2, SSPRK1, RKF5(4), Fekete10(4),"
              * "Fekete6(4), Fekete4(3), Fekete4(2)")
    end

    if split_operators && adaptive
        error("Adaptive timestepping not supported with operator splitting")
    end

    return rk_coefs, n_rk_stages, rk_order, adaptive, low_storage, CFL_prefactor
end

"""
use Runge Kutta to update any ion velocity moments evolved separately from
the pdf
"""
function rk_update_evolved_moments!(scratch, moments, t_params, istage)
    # if separately evolving the particle density, update using RK
    if moments.evolve_density
        rk_update_variable!(scratch, :density, t_params, istage)
    end

    # if separately evolving the parallel flow, update using RK
    if moments.evolve_upar
        rk_update_variable!(scratch, :upar, t_params, istage)
    end

    # if separately evolving the parallel pressure, update using RK;
    if moments.evolve_ppar
        rk_update_variable!(scratch, :ppar, t_params, istage)
    end
end

"""
use Runge Kutta to update any electron velocity moments evolved separately from
the pdf
"""
function rk_update_evolved_moments_electron!(scratch, moments, t_params, istage)
    # For now, electrons always fully moment kinetic, and ppar is the only evolving moment
    # (density and upar are calculated from quasineutrality and ambipolarity constraints).
    rk_update_variable!(scratch, :ppar_electron, t_params, istage)
end

"""
use Runge Kutta to update any neutral-particle velocity moments evolved separately from
the pdf
"""
function rk_update_evolved_moments_neutral!(scratch, moments, t_params, istage)
    # if separately evolving the particle density, update using RK
    if moments.evolve_density
        rk_update_variable!(scratch, :density_neutral, t_params, istage; neutrals=true)
    end

    # if separately evolving the parallel flow, update using RK
    if moments.evolve_upar
        rk_update_variable!(scratch, :uz_neutral, t_params, istage; neutrals=true)
    end

    # if separately evolving the parallel pressure, update using RK;
    if moments.evolve_ppar
        rk_update_variable!(scratch, :pz_neutral, t_params, istage; neutrals=true)
    end
end

"""
Update the variable named `var_symbol` in `scratch` to the current Runge-Kutta stage
`istage`. The current value in `scratch[istage+1]` is the result of the forward-Euler
update, which needs to be corrected using values from previous stages with the Runge-Kutta
coefficients.
"""
function rk_update_variable!(scratch, var_symbol::Symbol, t_params, istage; neutrals=false)
    if t_params.low_storage
        var_arrays = (getfield(scratch[istage+1], var_symbol),
                      getfield(scratch[istage], var_symbol),
                      getfield(scratch[1], var_symbol))
    else
        var_arrays = Tuple(getfield(scratch[i], var_symbol) for i ∈ 1:istage+1)
    end
    rk_coefs = @view t_params.rk_coefs[:,istage]

    if neutrals
        if t_params.low_storage
            rk_update_loop_neutrals_low_storage!(rk_coefs, var_arrays...)
        else
            rk_update_loop_neutrals!(rk_coefs, var_arrays)
        end
    else
        if t_params.low_storage
            rk_update_loop_low_storage!(rk_coefs, var_arrays...)
        else
            rk_update_loop!(rk_coefs, var_arrays)
        end
    end

    return nothing
end

"""
Calculate the estimated truncation error for the variable named `var_symbol`, for adaptive
timestepping methods.

The calculated error is stored in `var_symbol` in `scratch[2]` (as this entry should not
be needed again after the error is calculated).
"""
function rk_error_variable!(scratch, var_symbol::Symbol, t_params; neutrals=false)
    if !t_params.adaptive
        error("rk_error_variable!() should only be called when using adaptive "
              * "timestepping")
    end
    if t_params.low_storage
        var_arrays = (getfield(scratch[end], var_symbol),
                      getfield(scratch[end-1], var_symbol),
                      getfield(scratch[1], var_symbol))
    else
        var_arrays = Tuple(getfield(scratch[i], var_symbol) for i ∈ 1:length(scratch))
    end

    error_coefs = @view t_params.rk_coefs[:,end]

    # The second element of `scratch` is not needed any more for the RK update, so we can
    # overwrite it with the error estimate.
    output = getfield(scratch[2], var_symbol)

    if neutrals
        if t_params.low_storage
            rk_update_loop_neutrals_low_storage!(error_coefs, var_arrays...;
                                                 output=output)
        else
            rk_update_loop_neutrals!(error_coefs, var_arrays; output=output)
        end
    else
        if t_params.low_storage
            rk_update_loop_low_storage!(error_coefs, var_arrays...;
                                        output=output)
        else
            rk_update_loop!(error_coefs, var_arrays; output=output)
        end
    end

    return nothing
end

# Ion distribution function
function rk_update_loop_low_storage!(rk_coefs, new::AbstractArray{mk_float,5},
                                     old::AbstractArray{mk_float,5},
                                     first::AbstractArray{mk_float,5}; output=new)
    @boundscheck length(rk_coefs) == 3

    begin_s_r_z_vperp_vpa_region()
    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        output[ivpa,ivperp,iz,ir,is] = rk_coefs[1]*first[ivpa,ivperp,iz,ir,is] +
                                       rk_coefs[2]*old[ivpa,ivperp,iz,ir,is] +
                                       rk_coefs[3]*new[ivpa,ivperp,iz,ir,is]
    end

    return nothing
end
function rk_update_loop!(rk_coefs,
                         var_arrays::NTuple{N,AbstractArray{mk_float,5}};
                         output=var_arrays[N]) where N
    @boundscheck length(rk_coefs) ≥ N

    begin_s_r_z_vperp_vpa_region()
    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        output[ivpa,ivperp,iz,ir,is] =
            sum(rk_coefs[i] * var_arrays[i][ivpa,ivperp,iz,ir,is] for i ∈ 1:N)
    end

    return nothing
end

# Ion moments
function rk_update_loop_low_storage!(rk_coefs, new::AbstractArray{mk_float,3},
                                     old::AbstractArray{mk_float,3},
                                     first::AbstractArray{mk_float,3}; output=new)
    @boundscheck length(rk_coefs) == 3

    begin_s_r_z_region()
    @loop_s_r_z is ir iz begin
        output[iz,ir,is] = rk_coefs[1]*first[iz,ir,is] +
                           rk_coefs[2]*old[iz,ir,is] +
                           rk_coefs[3]*new[iz,ir,is]
    end

    return nothing
end
function rk_update_loop!(rk_coefs,
                         var_arrays::NTuple{N,AbstractArray{mk_float,3}};
                         output=var_arrays[N]) where N
    @boundscheck length(rk_coefs) ≥ N

    begin_s_r_z_region()
    @loop_s_r_z is ir iz begin
        output[iz,ir,is] = sum(rk_coefs[i] * var_arrays[i][iz,ir,is] for i ∈ 1:N)
    end

    return nothing
end

# Electron distribution function
function rk_update_loop_low_storage!(rk_coefs, new::AbstractArray{mk_float,4},
                                     old::AbstractArray{mk_float,4},
                                     first::AbstractArray{mk_float,4}; output=new)
    @boundscheck length(rk_coefs) == 3

    begin_r_z_vperp_vpa_region()
    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
        output[ivpa,ivperp,iz,ir] = rk_coefs[1]*first[ivpa,ivperp,iz,ir] +
                                    rk_coefs[2]*old[ivpa,ivperp,iz,ir] +
                                    rk_coefs[3]*new[ivpa,ivperp,iz,ir]
    end

    return nothing
end
function rk_update_loop!(rk_coefs,
                         var_arrays::NTuple{N,AbstractArray{mk_float,4}};
                         output=var_arrays[N]) where N
    @boundscheck length(rk_coefs) ≥ N

    begin_r_z_vperp_vpa_region()
    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
        output[ivpa,ivperp,iz,ir] =
            sum(rk_coefs[i] * var_arrays[i][ivpa,ivperp,iz,ir] for i ∈ 1:N)
    end

    return nothing
end

# Electron moments
function rk_update_loop_low_storage!(rk_coefs, new::AbstractArray{mk_float,2},
                                     old::AbstractArray{mk_float,2},
                                     first::AbstractArray{mk_float,2}; output=new)
    @boundscheck length(rk_coefs) == 3

    begin_r_z_region()
    @loop_r_z ir iz begin
        output[iz,ir] = rk_coefs[1]*first[iz,ir] +
                        rk_coefs[2]*old[iz,ir] +
                        rk_coefs[3]*new[iz,ir]
    end

    return nothing
end
function rk_update_loop!(rk_coefs,
                         var_arrays::NTuple{N,AbstractArray{mk_float,2}};
                         output=var_arrays[N]) where N
    @boundscheck length(rk_coefs) ≥ N

    begin_r_z_region()
    @loop_r_z ir iz begin
        output[iz,ir] = sum(rk_coefs[i] * var_arrays[i][iz,ir] for i ∈ 1:N)
    end

    return nothing
end

# Neutral distribution function
function rk_update_loop_neutrals_low_storage!(rk_coefs, new::AbstractArray{mk_float,6},
                                     old::AbstractArray{mk_float,6},
                                     first::AbstractArray{mk_float,6}; output=new)
    @boundscheck length(rk_coefs) == 3

    begin_sn_r_z_vzeta_vr_vz_region()
    @loop_sn_r_z_vzeta_vr_vz isn ir iz ivzeta ivr ivz begin
        output[ivz,ivr,ivzeta,iz,ir,isn] = rk_coefs[1]*first[ivz,ivr,ivzeta,iz,ir,isn] +
                                           rk_coefs[2]*old[ivz,ivr,ivzeta,iz,ir,isn] +
                                           rk_coefs[3]*new[ivz,ivr,ivzeta,iz,ir,isn]
    end

    return nothing
end
function rk_update_loop_neutrals!(rk_coefs,
                                  var_arrays::NTuple{N,AbstractArray{mk_float,6}};
                                  output=var_arrays[N]) where N
    @boundscheck length(rk_coefs) ≥ N

    begin_sn_r_z_vzeta_vr_vz_region()
    @loop_sn_r_z_vzeta_vr_vz isn ir iz ivzeta ivr ivz begin
        output[ivz,ivr,ivzeta,iz,ir,isn] =
            sum(rk_coefs[i] * var_arrays[i][ivz,ivr,ivzeta,iz,ir,isn] for i ∈ 1:N)
    end

    return nothing
end

# Neutral moments
function rk_update_loop_neutrals_low_storage!(rk_coefs, new::AbstractArray{mk_float,3},
                                              old::AbstractArray{mk_float,3},
                                              first::AbstractArray{mk_float,3};
                                              output=new)
    @boundscheck length(rk_coefs) == 3

    begin_sn_r_z_region()
    @loop_sn_r_z isn ir iz begin
        output[iz,ir,isn] = rk_coefs[1]*first[iz,ir,isn] +
                            rk_coefs[2]*old[iz,ir,isn] +
                            rk_coefs[3]*new[iz,ir,isn]
    end

    return nothing
end
function rk_update_loop_neutrals!(rk_coefs,
                                  var_arrays::NTuple{N,AbstractArray{mk_float,3}};
                                  output=var_arrays[N]) where N
    @boundscheck length(rk_coefs) ≥ N

    begin_sn_r_z_region()
    @loop_sn_r_z isn ir iz begin
        output[iz,ir,isn] = sum(rk_coefs[i] * var_arrays[i][iz,ir,isn] for i ∈ 1:N)
    end

    return nothing
end

"""
    local_error_norm(error, f, rtol, atol)
    local_error_norm(error, f, rtol, atol, neutral=false; method="Linf",
                     skip_r_inner=false, skip_z_lower=false, error_sum_zero=0.0)

Maximum error norm in the range owned by this MPI process, given by
```math
\\max(\\frac{|\\mathtt{error}|}{\\mathtt{rtol}*|\\mathtt{f}| + \\mathtt{atol})
```

3 dimensional arrays (which represent moments) are treated as ion moments unless
`neutral=true` is passed.

`method` can be "Linf" (to take the maximum error) or "L2" to take the root-mean-square
(RMS) error.

`skip_r_inner` and `skip_z_lower` can be set to true to skip the contribution from the
inner/lower boundaries, to avoid double-counting those points when using
distributed-memory MPI.

`error_sum_zero` should always have value 0.0, but is included so that different types can
be used for L2sum. For testing, if we want consistency of results when using different
numbers of processes (when the number of processes changes the order of operations in the
sum is changed, which changes the rounding errors) then we have to use higher precision
(i.e. use the Float128 type from the Quadmath package). The type of a 0.0 value can be set
according to the `high_precision_error_sum` option in the `[timestepping]` section, and
stored in a template-typed value in the `t_params` object - when that value is passed in
as the argument to `error_sum_zero`, that type will be used for L2sum, and the type will
be known at compile time, allowing this function to be efficient.
"""
function local_error_norm end

function local_error_norm(error::MPISharedArray{mk_float,2},
                          f::MPISharedArray{mk_float,2}, rtol, atol; method="Linf",
                          skip_r_inner=false, skip_z_lower=false, error_sum_zero=0.0)
    if method == "Linf"
        f_max = -Inf
        @loop_r_z ir iz begin
            error_norm = abs(error[iz,ir]) / (rtol*abs(f[iz,ir]) + atol)
            f_max = max(f_max, error_norm)
        end
        return f_max
    elseif method == "L2"
        L2sum = error_sum_zero
        @loop_r_z ir iz begin
            if (skip_r_inner && ir == 1) || (skip_z_lower && iz == 1)
                continue
            end
            error_norm = (error[iz,ir] / (rtol*abs(f[iz,ir]) + atol))^2
            L2sum += error_norm
        end
        # Will sum results from different processes in shared memory block after returning
        # from this function.
        nz, nr = size(error)
        if skip_r_inner
            nr -= 1
        end
        if skip_z_lower
            nz -= 1
        end
        return L2sum
    else
        error("Unrecognized method '$method'")
    end
end
function local_error_norm(error::MPISharedArray{mk_float,3},
                          f::MPISharedArray{mk_float,3}, rtol, atol, neutral=false;
                          method="Linf", skip_r_inner=false, skip_z_lower=false,
                          error_sum_zero=0.0)
    if method == "Linf"
        f_max = -Inf
        if neutral
            @loop_sn_r_z isn ir iz begin
                error_norm = abs(error[iz,ir,isn]) / (rtol*abs(f[iz,ir,isn]) + atol)
                f_max = max(f_max, error_norm)
            end
        else
            @loop_s_r_z is ir iz begin
                error_norm = abs(error[iz,ir,is]) / (rtol*abs(f[iz,ir,is]) + atol)
                f_max = max(f_max, error_norm)
            end
        end
        return f_max
    elseif method == "L2"
        L2sum = error_sum_zero
        if neutral
            @loop_sn_r_z isn ir iz begin
                if (skip_r_inner && ir == 1) || (skip_z_lower && iz == 1)
                    continue
                end
                error_norm = (error[iz,ir,isn] / (rtol*abs(f[iz,ir,isn]) + atol))^2
                L2sum += error_norm
            end
        else
            @loop_s_r_z is ir iz begin
                if (skip_r_inner && ir == 1) || (skip_z_lower && iz == 1)
                    continue
                end
                error_norm = (error[iz,ir,is] / (rtol*abs(f[iz,ir,is]) + atol))^2
                L2sum += error_norm
            end
        end
        # Will sum results from different processes in shared memory block after returning
        # from this function.
        nz, nr, nspecies = size(error)
        if skip_r_inner
            nr -= 1
        end
        if skip_z_lower
            nz -= 1
        end
        return L2sum
    else
        error("Unrecognized method '$method'")
    end
end
function local_error_norm(error::MPISharedArray{mk_float,4},
                          f::MPISharedArray{mk_float,4}, rtol, atol; method="Linf",
                          skip_r_inner=false, skip_z_lower=false, error_sum_zero=0.0)
    if method == "Linf"
        f_max = -Inf
        @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
            error_norm = abs(error[ivpa,ivperp,iz,ir]) /
                         (rtol*abs(f[ivpa,ivperp,iz,ir]) + atol)
            f_max = max(f_max, error_norm)
        end
        return f_max
    elseif method == "L2"
        L2sum = error_sum_zero
        @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
            if (skip_r_inner && ir == 1) || (skip_z_lower && iz == 1)
                continue
            end
            error_norm = (error[ivpa,ivperp,iz,ir] /
                          (rtol*abs(f[ivpa,ivperp,iz,ir]) + atol))^2
            L2sum += error_norm
        end
        # Will sum results from different processes in shared memory block after returning
        # from this function.
        nvpa, nvperp, nz, nr = size(error)
        if skip_r_inner
            nr -= 1
        end
        if skip_z_lower
            nz -= 1
        end
        return L2sum
    else
        error("Unrecognized method '$method'")
    end
end
function local_error_norm(error::MPISharedArray{mk_float,5},
                          f::MPISharedArray{mk_float,5}, rtol, atol; method="Linf",
                          skip_r_inner=false, skip_z_lower=false, error_sum_zero=0.0)
    if method == "Linf"
        f_max = -Inf
        @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
            error_norm = abs(error[ivpa,ivperp,iz,ir,is]) /
                         (rtol*abs(f[ivpa,ivperp,iz,ir,is]) + atol)
            f_max = max(f_max, error_norm)
        end
        return f_max
    elseif method == "L2"
        L2sum = error_sum_zero
        @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
            if (skip_r_inner && ir == 1) || (skip_z_lower && iz == 1)
                continue
            end
            error_norm = (error[ivpa,ivperp,iz,ir,is] /
                          (rtol*abs(f[ivpa,ivperp,iz,ir,is]) + atol))^2
            L2sum += error_norm
        end
        # Will sum results from different processes in shared memory block after returning
        # from this function.
        nvpa, nvperp, nz, nr, nspecies = size(error)
        if skip_r_inner
            nr -= 1
        end
        if skip_z_lower
            nz -= 1
        end
        return L2sum
    else
        error("Unrecognized method '$method'")
    end
end
function local_error_norm(error::MPISharedArray{mk_float,6},
                          f::MPISharedArray{mk_float,6}, rtol, atol; method="Linf",
                          skip_r_inner=false, skip_z_lower=false, error_sum_zero=0.0)
    if method == "Linf"
        f_max = -Inf
        @loop_sn_r_z_vzeta_vr_vz isn ir iz ivzeta ivr ivz begin
            error_norm = abs(error[ivz,ivr,ivzeta,iz,ir,isn]) /
                         (rtol*abs(f[ivz,ivr,ivzeta,iz,ir,isn]) + atol)
            f_max = max(f_max, error_norm)
        end
        return f_max
    elseif method == "L2"
        L2sum = error_sum_zero
        @loop_sn_r_z_vzeta_vr_vz isn ir iz ivzeta ivr ivz begin
            if (skip_r_inner && ir == 1) || (skip_z_lower && iz == 1)
                continue
            end
            error_norm = (error[ivz,ivr,ivzeta,iz,ir,isn] /
                          (rtol*abs(f[ivz,ivr,ivzeta,iz,ir,isn]) + atol))^2
            L2sum += error_norm
        end
        # Will sum results from different processes in shared memory block after returning
        # from this function.
        return L2sum
    else
        error("Unrecognized method '$method'")
    end
end

"""
    adaptive_timestep_update_t_params!(t_params, CFL_limits, error_norms,
                                       total_points, current_dt, error_norm_method)

Use the calculated `CFL_limits` and `error_norms` to update the timestep in `t_params`.
"""
function adaptive_timestep_update_t_params!(t_params, scratch, t, CFL_limits, error_norms,
                                            total_points, current_dt, error_norm_method;
                                            electron=false)
    # Get global minimum of CFL limits
    CFL_limit = nothing
    this_limit_caused_by = nothing
    @serial_region begin
        # Get maximum error over all blocks
        CFL_limits = MPI.Allreduce(CFL_limits, min, comm_inter_block[])
        CFL_limit_caused_by = argmin(CFL_limits)
        CFL_limit = CFL_limits[CFL_limit_caused_by]
        # Reserve first five entries of t_params.limit_caused_by for accuracy,
        # max_increase_factor, max_increase_factor_near_fail, minimum_dt and maximum_dt
        # limits.
        this_limit_caused_by = CFL_limit_caused_by + 5
    end

    if error_norm_method == "Linf"
        # Get overall maximum error on the shared-memory block
        error_norms = MPI.Reduce(error_norms, max, comm_block[]; root=0)

        error_norm = nothing
        @serial_region begin
            # Get maximum error over all blocks
            error_norms = MPI.Allreduce(error_norms, max, comm_inter_block[])
            error_norm = maximum(error_norms)
        end
        error_norm = MPI.bcast(error_norm, 0, comm_block[])
    elseif error_norm_method == "L2"
        # Get overall maximum error on the shared-memory block
        error_norms = MPI.Reduce(error_norms, +, comm_block[]; root=0)

        error_norm = nothing
        @serial_region begin
            # Get maximum error over all blocks
            error_norms = MPI.Allreduce(error_norms, +, comm_inter_block[])

            # So far `error_norms` is the sum of squares of the errors. Now that summation
            # is finished, need to divide by total number of points and take square-root.
            error_norms .= sqrt.(error_norms ./ total_points)

            # Weight the error from each variable equally by taking the mean, so the
            # larger number of points in the distribution functions does not mean that
            # error on the moments is ignored.
            error_norm = mean(error_norms)
        end

        error_norm = MPI.bcast(error_norm, 0, comm_block[])
    else
        error("Unrecognized error_norm_method '$method'")
    end

    just_completed_output_step = false

    # Use current_dt instead of t_params.dt[] here because we are about to write to
    # the shared-memory variable t_params.dt[] below, and we do not want to add an extra
    # _block_synchronize() call after reading it here.
    if error_norm > 1.0 && current_dt > t_params.minimum_dt
        # Timestep failed, reduce timestep and re-try

        # Set scratch[end] equal to scratch[1] to start the timestep over
        scratch_temp = scratch[t_params.n_rk_stages+1]
        scratch[t_params.n_rk_stages+1] = scratch[1]
        scratch[1] = scratch_temp

        @serial_region begin
            t_params.failure_counter[] += 1

            if t_params.previous_dt[] > 0.0
                # If previous_dt=0, the previous step was also a failure so only update
                # dt_before_last_fail when previous_dt>0
                t_params.dt_before_last_fail[] = t_params.previous_dt[]
            end

            # If we were trying to take a step to the output timestep, dt will be smaller on
            # the re-try, so will not reach the output time.
            t_params.step_to_output[] = false

            # Get new timestep estimate using same formula as for a successful step, but
            # limit decrease to factor 1/2 - this factor should probably be settable!
            t_params.dt[] = max(t_params.dt[] / 2.0,
                                t_params.dt[] * t_params.step_update_prefactor * error_norm^(-1.0/t_params.rk_order))
            t_params.dt[] = max(t_params.dt[], t_params.minimum_dt)

            minimum_dt = 1.e-14
            if t_params.dt[] < minimum_dt
                println("Time advance failed: trying to set dt=$(t_params.dt[]) less than "
                        * "$minimum_dt at t=$t. Ending run.")
                # Set dt negative to signal an error
                t_params.dt[] = -1.0
            end

            # Don't update the simulation time, as this step failed
            t_params.previous_dt[] = 0.0

            # Call the 'cause' of the timestep failure the variable that has the biggest
            # error norm here
            max_error_variable_index = argmax(error_norms)
            t_params.failure_caused_by[max_error_variable_index] += 1

            #println("t=$t, timestep failed, error_norm=$error_norm, error_norms=$error_norms, decreasing timestep to ", t_params.dt[])
        end
    else
        @serial_region begin
            # Save the timestep used to complete this step, this is used to update the
            # simulation time.
            t_params.previous_dt[] = t_params.dt[]

            if t_params.step_to_output[]
                # Completed an output step, reset dt to what it was before it was reduced to reach
                # the output time
                t_params.dt[] = t_params.dt_before_output[]
                t_params.step_to_output[] = false

                if t_params.dt[] > CFL_limit
                    t_params.dt[] = CFL_limit
                end

                just_completed_output_step = true
            else
                # Adjust timestep according to Fehlberg's suggestion
                # (https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method).
                # `step_update_prefactor` is a constant numerical factor to make the estimate
                # of a good value for the next timestep slightly conservative. It defaults to
                # 0.9.
                t_params.dt[] *= t_params.step_update_prefactor * error_norm^(-1.0/t_params.rk_order)

                if t_params.dt[] > CFL_limit
                    t_params.dt[] = CFL_limit
                else
                    this_limit_caused_by = 1
                end

                # Limit so timestep cannot increase by a large factor, which might lead to
                # numerical instability in some cases.
                max_cap_limit_caused_by = 2
                if isinf(t_params.max_increase_factor_near_last_fail)
                    # Not using special timestep limiting near last failed dt value
                    max_cap = t_params.max_increase_factor * t_params.previous_dt[]
                else
                    max_cap = t_params.max_increase_factor * t_params.previous_dt[]
                    slow_increase_threshold = t_params.dt_before_last_fail[] / t_params.last_fail_proximity_factor
                    if t_params.previous_dt[] > t_params.dt_before_last_fail[] * t_params.last_fail_proximity_factor
                        # dt has successfully exceeded the last failed value, so allow it
                        # to increase more quickly again
                        t_params.dt_before_last_fail[] = Inf
                    elseif max_cap > slow_increase_threshold
                        # dt is getting close to last failed value, so increase more
                        # slowly
                        max_cap = max(slow_increase_threshold,
                                      t_params.max_increase_factor_near_last_fail *
                                      t_params.previous_dt[])
                        max_cap_limit_caused_by = 3
                    end
                end
                if t_params.dt[] > max_cap
                    t_params.dt[] = max_cap
                    this_limit_caused_by = max_cap_limit_caused_by
                end

                # Prevent timestep from going below minimum_dt
                if t_params.dt[] < t_params.minimum_dt
                    t_params.dt[] = t_params.minimum_dt
                    this_limit_caused_by = 4
                end

                # Prevent timestep from going above maximum_dt
                if t_params.dt[] > t_params.maximum_dt
                    t_params.dt[] = t_params.maximum_dt
                    this_limit_caused_by = 5
                end

                t_params.limit_caused_by[this_limit_caused_by] += 1

                if (t_params.step_counter[] % 1000 == 0) && global_rank[] == 0
                    prefix = electron ? "electron" : "ion"
                    println("$prefix step ", t_params.step_counter[], ": t=",
                            round(t, sigdigits=6), ", nfail=", t_params.failure_counter[],
                            ", dt=", t_params.dt[])
                end
            end
        end
    end

    @serial_region begin
        current_time = t + t_params.previous_dt[]
        if (!just_completed_output_step
            && (current_time + t_params.dt[] >= t_params.next_output_time[]))

            t_params.dt_before_output[] = t_params.dt[]
            t_params.dt[] = t_params.next_output_time[] - current_time
            t_params.step_to_output[] = true
        end
    end

    # Shared-memory variables have been updated, so synchronize
    _block_synchronize()

    return nothing
end

end # runge_kutta
