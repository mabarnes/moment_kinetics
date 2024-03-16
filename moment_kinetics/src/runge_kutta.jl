"""
Runge Kutta timestepping
"""
module runge_kutta

export setup_runge_kutta_coefficients!, rk_update_evolved_moments!,
       rk_update_evolved_moments_electron!, rk_update_evolved_moments_neutral!,
       rk_update_variable!

using ..array_allocation: allocate_float
using ..looping
using ..type_definitions: mk_float

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
        # Fekete 4-stage, 3rd-order SSPRK (see comments in util/calculate_rk_coeffs.jl.
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

# Ion distribution function
function rk_update_loop_low_storage!(rk_coefs, new::AbstractArray{mk_float,5},
                                     old::AbstractArray{mk_float,5},
                                     first::AbstractArray{mk_float,5})
    @boundscheck length(rk_coefs) == 3

    begin_s_r_z_vperp_vpa_region()
    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        new[ivpa,ivperp,iz,ir,is] = rk_coefs[1]*first[ivpa,ivperp,iz,ir,is] +
                                    rk_coefs[2]*old[ivpa,ivperp,iz,ir,is] +
                                    rk_coefs[3]*new[ivpa,ivperp,iz,ir,is]
    end

    return nothing
end
function rk_update_loop!(rk_coefs,
                         var_arrays::NTuple{N,AbstractArray{mk_float,5}}) where N
    @boundscheck length(rk_coefs) ≥ N

    begin_s_r_z_vperp_vpa_region()
    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        var_arrays[N][ivpa,ivperp,iz,ir,is] =
            sum(rk_coefs[i] * var_arrays[i][ivpa,ivperp,iz,ir,is] for i ∈ 1:N)
    end

    return nothing
end

# Ion moments
function rk_update_loop_low_storage!(rk_coefs, new::AbstractArray{mk_float,3},
                                     old::AbstractArray{mk_float,3},
                                     first::AbstractArray{mk_float,3})
    @boundscheck length(rk_coefs) == 3

    begin_s_r_z_region()
    @loop_s_r_z is ir iz begin
        new[iz,ir,is] = rk_coefs[1]*first[iz,ir,is] +
                        rk_coefs[2]*old[iz,ir,is] +
                        rk_coefs[3]*new[iz,ir,is]
    end

    return nothing
end
function rk_update_loop!(rk_coefs,
                         var_arrays::NTuple{N,AbstractArray{mk_float,3}}) where N
    @boundscheck length(rk_coefs) ≥ N

    begin_s_r_z_region()
    @loop_s_r_z is ir iz begin
        var_arrays[N][iz,ir,is] = sum(rk_coefs[i] * var_arrays[i][iz,ir,is] for i ∈ 1:N)
    end

    return nothing
end

# Electron distribution function
function rk_update_loop_low_storage!(rk_coefs, new::AbstractArray{mk_float,4},
                                     old::AbstractArray{mk_float,4},
                                     first::AbstractArray{mk_float,4})
    @boundscheck length(rk_coefs) == 3

    begin_r_z_vperp_vpa_region()
    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
        new[ivpa,ivperp,iz,ir] = rk_coefs[1]*first[ivpa,ivperp,iz,ir] +
                                 rk_coefs[2]*old[ivpa,ivperp,iz,ir] +
                                 rk_coefs[3]*new[ivpa,ivperp,iz,ir]
    end

    return nothing
end
function rk_update_loop!(rk_coefs,
                         var_arrays::NTuple{N,AbstractArray{mk_float,4}}) where N
    @boundscheck length(rk_coefs) ≥ N

    begin_r_z_vperp_vpa_region()
    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
        var_arrays[N][ivpa,ivperp,iz,ir] =
            sum(rk_coefs[i] * var_arrays[i][ivpa,ivperp,iz,ir] for i ∈ 1:N)
    end

    return nothing
end

# Electron moments
function rk_update_loop_low_storage!(rk_coefs, new::AbstractArray{mk_float,2},
                                     old::AbstractArray{mk_float,2},
                                     first::AbstractArray{mk_float,2})
    @boundscheck length(rk_coefs) == 3

    begin_r_z_region()
    @loop_r_z ir iz begin
        new[iz,ir] = rk_coefs[1]*first[iz,ir] +
                     rk_coefs[2]*old[iz,ir] +
                     rk_coefs[3]*new[iz,ir]
    end

    return nothing
end
function rk_update_loop!(rk_coefs,
                         var_arrays::NTuple{N,AbstractArray{mk_float,2}}) where N
    @boundscheck length(rk_coefs) ≥ N

    begin_r_z_region()
    @loop_r_z ir iz begin
        var_arrays[N][iz,ir] = sum(rk_coefs[i] * var_arrays[i][iz,ir] for i ∈ 1:N)
    end

    return nothing
end

# Neutral distribution function
function rk_update_loop_neutrals_low_storage!(rk_coefs, new::AbstractArray{mk_float,6},
                                     old::AbstractArray{mk_float,6},
                                     first::AbstractArray{mk_float,6})
    @boundscheck length(rk_coefs) == 3

    begin_sn_r_z_vzeta_vr_vz_region()
    @loop_sn_r_z_vzeta_vr_vz isn ir iz ivzeta ivr ivz begin
        new[ivz,ivr,ivzeta,iz,ir,isn] = rk_coefs[1]*first[ivz,ivr,ivzeta,iz,ir,isn] +
                                        rk_coefs[2]*old[ivz,ivr,ivzeta,iz,ir,isn] +
                                        rk_coefs[3]*new[ivz,ivr,ivzeta,iz,ir,isn]
    end

    return nothing
end
function rk_update_loop_neutrals!(rk_coefs,
                                  var_arrays::NTuple{N,AbstractArray{mk_float,6}}) where N
    @boundscheck length(rk_coefs) ≥ N

    begin_sn_r_z_vzeta_vr_vz_region()
    @loop_sn_r_z_vzeta_vr_vz isn ir iz ivzeta ivr ivz begin
        var_arrays[N][ivz,ivr,ivzeta,iz,ir,isn] =
            sum(rk_coefs[i] * var_arrays[i][ivz,ivr,ivzeta,iz,ir,isn] for i ∈ 1:N)
    end

    return nothing
end

# Neutral moments
function rk_update_loop_neutrals_low_storage!(rk_coefs, new::AbstractArray{mk_float,3},
                                              old::AbstractArray{mk_float,3},
                                              first::AbstractArray{mk_float,3})
    @boundscheck length(rk_coefs) == 3

    begin_sn_r_z_region()
    @loop_sn_r_z isn ir iz begin
        new[iz,ir,isn] = rk_coefs[1]*first[iz,ir,isn] +
                         rk_coefs[2]*old[iz,ir,isn] +
                         rk_coefs[3]*new[iz,ir,isn]
    end

    return nothing
end
function rk_update_loop_neutrals!(rk_coefs,
                                  var_arrays::NTuple{N,AbstractArray{mk_float,3}}) where N
    @boundscheck length(rk_coefs) ≥ N

    begin_sn_r_z_region()
    @loop_sn_r_z isn ir iz begin
        var_arrays[N][iz,ir,isn] = sum(rk_coefs[i] * var_arrays[i][iz,ir,isn] for i ∈ 1:N)
    end

    return nothing
end

end # runge_kutta
