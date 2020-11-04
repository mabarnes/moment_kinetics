module source_terms

export setup_source
export update_advection_factor!
export calculate_explicit_source!
export update_f!
export update_boundary_indices!

using array_allocation: allocate_float
using moment_kinetics_input: advection_speed, advection_speed_option

# structure containing the basic arrays associated with the
# source terms appearing in the advection equation for each coordinate
mutable struct source_info
    # rhs is the sum of the source terms appearing on the righthand side
    # of the equation
    rhs::Array{Float64, 1}
    # df is the derivative of the distribution function f with respect
    # to the coordinate associated with this set of source terms
    df::Array{Float64, 1}
    # speed is the component of the advection speed along this coordinate axis
    speed::Array{Float64, 1}
    # adv_fac is the advection factor that multiplies df in the advective source
    adv_fac::Array{Float64, 1}
    # upwind_idx is the boundary index for the upwind boundary
    upwind_idx::Int64
    # downwind_idx is the boundary index for the downwind boundary
    downwind_idx::Int64
    # upwind_increment is the index increment used when sweeping in the upwind direction
    upwind_increment::Int64
end
# create arrays needed to compute the source term(s) for a 1D problem
function setup_source(n)
    return setup_source_local(n)
end
# create arrays needed to compute the source term(s) for a 2D problem
function setup_source(n, m)
    # allocate an array containing structures with much of the info needed
    # to do the 1D advection time advance
    source = Array{source_info,1}(undef, m)
    # store all of this information in a structure and return it
    for i ∈ 1:m
        source[i] = setup_source_local(n)
    end
    return source
end
# create arrays needed to compute the source term(s)
function setup_source_local(n)
    # create array for storing the explicit source terms appearing
    # on the righthand side of the equation
    rhs = allocate_float(n)
    # create array for storing ∂f/∂(coordinate)
    df = allocate_float(n)
    # create array for storing the advection coefficient
    adv_fac = allocate_float(n)
    # create array for storing the speed along this coordinate
    speed = allocate_float(n)
    # index for the upwind boundary; will be updated before use so value irrelevant
    upwind_idx = 1
    # index for the downwind boundary; will be updated before use so value irrelevant
    downwind_idx = n
    # index increment used when sweeping in the upwind direction; will be updated before use
    upwind_increment = -1
    # return source_info struct containing necessary 1D/0D arrays
    return source_info(rhs, df, speed, adv_fac, upwind_idx, downwind_idx, upwind_increment)
end
# calculate the grid index correspond to the upwind and downwind boundaries,
# as well as the index increment needed to sweep in the upwind direction
function update_boundary_indices!(source)
    m = size(source,1)
    n = size(source[1].speed,1)
    for j ∈ 1:m
        # for now, assume the speed has the same sign at all grid points
        # so only need to check its value at one location to determine the upwind direction
        if source[j].speed[1] > 0
            source[j].upwind_idx = 1
            source[j].upwind_increment = -1
            source[j].downwind_idx = n
        else
            source[j].upwind_idx = n
            source[j].upwind_increment = 1
            source[j].downwind_idx = 1
        end
    end
    return nothing
end
# calculate the factor appearing in front of f' in the advection term
# at time level n in the frame moving with the approximate characteristic
function update_advection_factor!(adv_fac, speed, upwind_idx, downwind_idx,
    upwind_increment, SL, n, dt, j)
    @boundscheck n == length(SL.dep_idx) || throw(BoundsError(SL.dep_idx))
    @boundscheck n == length(adv_fac) || throw(BoundsError(adv_fac))
    @boundscheck n == length(speed) || throw(BoundsError(speed))
    @boundscheck n == length(SL.characteristic_speed) ||
        throw(BoundsError(SL.characteristic_speed))
    @inbounds for i ∈ upwind_idx-upwind_increment:-upwind_increment:downwind_idx
        idx = SL.dep_idx[i]
        # only need to calculate advection factor for characteristics
        # that originate within the domain, as zero incoming BC
        # takes care of the rest.
        if idx != upwind_idx + upwind_increment
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
function calculate_explicit_source!(rhs, df, adv_fac, up_idx, down_idx, up_incr,
    dep_idx, n, j)
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
        @inbounds for i ∈ up_idx:-up_incr:down_idx
            idx = dep_idx[i]
            if idx != up_idx + up_incr
                rhs[i] = adv_fac[i]*df[idx]
            end
        end
    elseif j == 2
        @inbounds for i ∈ up_idx:-up_incr:down_idx
            rhs[i] = adv_fac[i]*df[i]
        end
    end
    return nothing
end

end
