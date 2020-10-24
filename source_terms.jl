module source_terms

import array_allocation: allocate_float
import moment_kinetics_input: advection_speed, advection_speed_option

export setup_source
#export update_speed!
export update_advection_factor!
export calculate_explicit_source!
export update_f!
#=
struct advection_speed_info_z
    # this is the peculiar velocity along z
    vpa::Array{Float64,1}
end
struct advection_speed_info_vpa
    # this is the particle distribution function
    ff::Array{Float64,3}
    # fcheby contains the Chebyshev spectral coefficients of ff
    fcheby::Array{Float64,2}
end
function setup_advection_speed_z(vpa)
    return advection_speed_info_z(vpa)
end
function setup_advection_speed_vpa(ff, fcheby)
    return advection_speed_info_vpa(ff, fcheby)
end
=#
# structure containing the basic arrays associated with the
# source terms appearing in the advection equation for each coordinate
struct source_info{ndim}
    # rhs is the sum of the source terms appearing on the righthand side
    # of the equation
    rhs::Array{Float64, ndim}
    # df is the derivative of the distribution function f with respect
    # to the coordinate associated with this set of source terms
    df::Array{Float64, ndim}
    # speed is the component of the advection speed along this coordinate axis
    speed::Array{Float64, ndim}
    # adv_fac is the advection factor that multiplies df in the advective source
    adv_fac::Array{Float64, ndim}
end
# create arrays needed to compute the source term(s)
function setup_source(n)
    # create array for storing the explicit source terms appearing
    # on the righthand side of the equation
    rhs = allocate_float(n)
    # create array for storing ∂f/∂(coordinate)
    df = allocate_float(n)
    # create array for storing the advection coefficient
    adv_fac = allocate_float(n)
    # create array for storing the speed along this coordinate
    speed = allocate_float(n)
    # return source_info struct containing necessary 1D arrays
    return source_info{1}(rhs, df, speed, adv_fac)
end
# create arrays needed to compute the source term(s)
function setup_source(n, m)
    # create array for storing the explicit source terms appearing
    # on the righthand side of the equation
    rhs = allocate_float(n, m)
    # create array for storing ∂f/∂(coordinate)
    df = allocate_float(n, m)
    # create array for storing the advection coefficient
    adv_fac = allocate_float(n, m)
    # create array for storing the speed along this coordinate
    speed = allocate_float(n, m)
    # return source_info struct containing necessary 1D arrays
    return source_info{2}(rhs, df, speed, adv_fac)
end
#=
# calculate the advection speed at each grid point
function update_speed!(speed, coord)
    n = coord.n
    @boundscheck n == length(speed) || throw(BoundsError(speed))
    if advection_speed_option == "constant"
        @inbounds for i ∈ 1:n
            speed[i] = advection_speed
        end
    elseif advection_speed_option == "linear"
        @inbounds for i ∈ 1:n
            speed[i] = advection_speed*(coord.grid[i]+0.5*coord.L)
        end
    end
    return nothing
end
# calculate the advection speed in the z coordinate
function update_speed!(speed, adv::advection_speed_info_z, coord)
    n = coord.n
    @boundscheck n == length(speed) || throw(BoundsError(speed))
    @inbounds for i ∈ 1:n
        speed[i] = adv.vpa[i]
    end
    return nothing
end
# calculate the advection speed in the vpa coordinate
function update_speed!(speed, adv::advection_speed_info_vpa, coord)
    n = coord.n
    @boundscheck n == length(speed) || throw(BoundsError(speed))
    # obtain the parallel pressure
    update_ppar!(ppar, vpa_tmp, ff, vpa, nz)
    # calculate the z derivative of the parallel pressure

    return nothing
end
=#
# calculate the factor appearing in front of f' in the advection term
# at time level n in the frame moving with the approximate characteristic
function update_advection_factor!(adv_fac, speed, SL, n, dt, j)
    @boundscheck n == length(SL.dep_idx) || throw(BoundsError(SL.dep_idx))
    @boundscheck n == length(adv_fac) || throw(BoundsError(adv_fac))
    @boundscheck n == length(speed) || throw(BoundsError(speed))
    @boundscheck n == length(SL.characteristic_speed) ||
        throw(BoundsError(SL.characteristic_speed))
    @inbounds for i ∈ 2:n
        idx = SL.dep_idx[i]
        # only need to calculate advection factor for characteristics
        # that originate within the domain, as zero incoming BC
        # takes care of the rest.
        if idx > 0
            # the effective advection speed appearing in the source
            # is the speed in the frame moving with the approximate
            # characteristic speed v_char
            # NB: need to change v[idx] to v[i] for second iteration of RK
            if j == 1
                adv_fac[i] = -dt*(speed[idx]-SL.characteristic_speed[i])
            elseif j == 2
                adv_fac[i] = -dt*(speed[i]-SL.characteristic_speed[i])
            end
        end
    end
    return nothing
end
# calculate the explicit source terms on the rhs of the equation;
# i.e., -Δt⋅δv⋅f'
function calculate_explicit_source!(rhs, df, adv_fac, dep_idx, n, j)
    # ensure that arrays needed for this function are inbounds
    # to avoid checking multiple times later
    @boundscheck n == length(rhs) || throw(BoundsError(rhs))
    @boundscheck n == length(df) || throw(BoundsError(df))
    @boundscheck n == length(adv_fac) || throw(BoundsError(adv_fac))
    @boundscheck n == length(dep_idx) || throw(BoundsError(dep_idx))
    # calculate the source evaluated at the departure point for the
    # ith characteristic.  note that adv_fac[i] has already
    # been defined so that it corresponds to the advection factor
    # corresponding to the ith characteristic
    if j == 1
        @inbounds for i ∈ 2:n
            idx = dep_idx[i]
            if idx > 0
                rhs[i] = adv_fac[i]*df[idx]
            end
        end
    elseif j == 2
        @inbounds for i ∈ 2:n
            rhs[i] = adv_fac[i]*df[i]
        end
    end
    return nothing
end
# update ff at time level n+1 using an explicit Runge-Kutta method
# along approximate characteristics
function update_f!(ff, rhs, dep_idx, n, j)
    @boundscheck n == size(ff,1) || throw(BoundsError(ff))
    @boundscheck n == length(rhs) || throw(BoundsError(rhs))
    @boundscheck n == length(dep_idx) || throw(BoundsError(dep_idx))

    @inbounds for i ∈ 2:n
        # dep_idx is the index of the departure point for the approximate
        # characteristic passing through grid point i
        # if semi-Lagrange is not used, then dep_idx = i
        idx = dep_idx[i]
        if idx > 0
            ff[i,j+1] = ff[idx,1] + rhs[i]
        else
            # NB: need to re-examine this for case with non-advective terms
            ff[i,j+1] = 0.
        end
    end
    return nothing
end

end
