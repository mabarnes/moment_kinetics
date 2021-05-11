module advection

export setup_advection
export update_advection_factor!
export calculate_explicit_advection!
export update_boundary_indices!
export advance_f_local!

using type_definitions: mk_float, mk_int
using array_allocation: allocate_float
using calculus: derivative!

# structure containing the basic arrays associated with the
# advection terms appearing in the advection equation for each coordinate
mutable struct advection_info
    # rhs is the sum of the advection terms appearing on the righthand side
    # of the equation
    rhs::Array{mk_float, 1}
    # df is the derivative of the distribution function f with respect
    # to the coordinate associated with this set of advection terms
    # it has dimensions of nelement x ngrid_per_element
    df::Array{mk_float, 2}
    # speed is the component of the advection speed along this coordinate axis
    speed::Array{mk_float, 1}
    # if using semi-Lagrange approach,
    # modified_speed is delta / dt, where delta for a given characteristic
    # is the displacement from the arrival point to the
    # (generally off-grid) departure point using the coordinate in which
    # the grid is equally spaced (a re-scaling of the Chebyshev theta coordinate);
    # otherwise, modified_speed = speed
    modified_speed::Array{mk_float,1}
    # adv_fac is the advection factor that multiplies df in the advection term
    adv_fac::Array{mk_float, 1}
    # upwind_idx is the boundary index for the upwind boundary
    upwind_idx::mk_int
    # downwind_idx is the boundary index for the downwind boundary
    downwind_idx::mk_int
    # upwind_increment is the index increment used when sweeping in the upwind direction
    upwind_increment::mk_int
end
# create arrays needed to compute the advection term(s) for a 1D problem
function setup_advection(coord, nspec)
    # allocate an array containing structures with much of the info needed
    # to do the 1D advection time advance
    advection = Array{advection_info,1}(undef, nspec)
    # store all of this information in a structure and return it
    for is ∈ 1:nspec
        advection[is] = setup_advection_local(coord.n, coord.ngrid, coord.nelement)
    end
    return advection
end
# create arrays needed to compute the advection term(s) for a 2D problem
function setup_advection(coord1, coord2, nspec)
    # n and m are the number of unique grid points along coordinates coord1 and coord2
    n = coord1.n
    m = coord2.n
    # allocate an array containing structures with much of the info needed
    # to do the 1D advection time advance
    advection = Array{advection_info,2}(undef, m, nspec)
    # store all of this information in a structure and return it
    for is ∈ 1:nspec
        for i ∈ 1:m
            advection[i,is] = setup_advection_local(coord1.n, coord1.ngrid, coord1.nelement)
        end
    end
    return advection
end
# create arrays needed to compute the advection term(s)
function setup_advection_local(n, ngrid, nelement)
    # create array for storing the explicit advection terms appearing
    # on the righthand side of the equation
    rhs = allocate_float(n)
    # create array for storing ∂f/∂(coordinate)
    # NB: need to store on nelement x ngrid_per_element array, as must keep info
    # about multi-valued derivative at overlapping point at element boundaries
    df = allocate_float(ngrid, nelement)
    # create array for storing the advection coefficient
    adv_fac = allocate_float(n)
    # create array for storing the speed along this coordinate
    speed = allocate_float(n)
    # create array for storing the modified speed along this coordinate
    modified_speed = allocate_float(n)
    # index for the upwind boundary; will be updated before use so value irrelevant
    upwind_idx = 1
    # index for the downwind boundary; will be updated before use so value irrelevant
    downwind_idx = n
    # index increment used when sweeping in the upwind direction; will be updated before use
    upwind_increment = -1
    # return advection_info struct containing necessary 1D/0D arrays
    return advection_info(rhs, df, speed, modified_speed, adv_fac, upwind_idx, downwind_idx, upwind_increment)
end
# calculate the grid index correspond to the upwind and downwind boundaries,
# as well as the index increment needed to sweep in the upwind direction
function update_boundary_indices!(advection)
    m = size(advection,1)
    n = size(advection[1].speed,1)
    for j ∈ 1:m
        # NB: for now, assume the speed has the same sign at all grid points
        # so only need to check its value at one location to determine the upwind direction
        if advection[j].speed[1] > 0
            advection[j].upwind_idx = 1
            advection[j].upwind_increment = -1
            advection[j].downwind_idx = n
        else
            advection[j].upwind_idx = n
            advection[j].upwind_increment = 1
            advection[j].downwind_idx = 1
        end
    end
    return nothing
end
# calculate the factor appearing in front of f' in the advection term
# at time level n in the frame moving with the approximate characteristic
function update_advection_factor!(adv_fac, speed, upwind_idx, downwind_idx,
    upwind_increment, SL, n, dt, j, coord)
    @boundscheck n == length(SL.dep_idx) || throw(BoundsError(SL.dep_idx))
    @boundscheck n == length(adv_fac) || throw(BoundsError(adv_fac))
    @boundscheck n == length(speed) || throw(BoundsError(speed))
    @boundscheck n == length(SL.characteristic_speed) ||
        throw(BoundsError(SL.characteristic_speed))
    #NB: commented out line below needed for bc != periodic?
    #@inbounds for i ∈ upwind_idx-upwind_increment:-upwind_increment:downwind_idx
    #@inbounds begin
        for i ∈ upwind_idx:-upwind_increment:downwind_idx
            idx = SL.dep_idx[i]
            # only need to calculate advection factor for characteristics
            # that originate within the domain, as zero/constant incoming BC
            # takes care of the rest.
            if idx != upwind_idx + upwind_increment
                # the effective advection speed appearing in the advection term
                # is the speed in the frame moving with the approximate
                # characteristic speed v_char
                # NB: need to change v[idx] to v[i] for second iteration of RK
                if j == 1
                    adv_fac[i] = -dt*(speed[idx]-SL.characteristic_speed[i])
                else
                    adv_fac[i] = -dt*(speed[i]-SL.characteristic_speed[i])
                end
            end
        end
    #end
    return nothing
end
# calculate the explicit advection terms on the rhs of the equation;
# i.e., -Δt⋅δv⋅f'
function calculate_explicit_advection!(rhs, df, adv_fac, up_idx, up_incr, dep_idx, n, j)
    # calculate the advection terms evaluated at the departure point for the
    # ith characteristic.  note that adv_fac[i] has already
    # been defined so that it corresponds to the advection factor
    # corresponding to the ith characteristic
    if j == 1
        for i ∈ 1:n
            idx = dep_idx[i]
            if idx != up_idx + up_incr
                rhs[i] = adv_fac[i]*df[idx]
            end
        end
    else
        for i ∈ 1:n
            rhs[i] = adv_fac[i]*df[i]
        end
    end
    return nothing
end
# update the righthand side of the equation to account for 1d advection in this coordinate
function update_rhs!(advection, f_current, SL, coord, dt, j, spectral)
    # calculate the factor appearing in front of df/dcoord in the advection
    # term at time level n in the frame moving with the approximate
    # characteristic
    update_advection_factor!(advection.adv_fac,
        advection.modified_speed, advection.upwind_idx, advection.downwind_idx,
        advection.upwind_increment, SL, coord.n, dt, j, coord)
    # calculate df/dcoord
    derivative!(coord.scratch, f_current, coord, advection.adv_fac, spectral)
    #derivative!(coord.scratch, f_current, coord, spectral)
    # calculate the explicit advection terms on the rhs of the equation;
    # i.e., -Δt⋅δv⋅f'
    calculate_explicit_advection!(advection.rhs, coord.scratch,
        advection.adv_fac, advection.upwind_idx, advection.upwind_increment,
        SL.dep_idx, coord.n, j)
end
# do all the work needed to update f(coord) at a single value of other coords
function advance_f_local!(f_new, f_current, f_old, SL, advection, coord, dt, j, spectral, use_SL)
    # update the rhs of the equation accounting for 1d advection in coord
    update_rhs!(advection, f_current, SL, coord, dt, j, spectral)
    # update ff at time level n+1 using an explicit Runge-Kutta method
    # along approximate characteristics
    update_f!(f_new, f_old, advection.rhs, advection.upwind_idx, advection.downwind_idx,
        advection.upwind_increment, SL.dep_idx, coord.n, coord.bc, use_SL)
end
# update ff at time level n+1 using an explicit Runge-Kutta method
# along approximate characteristics
function update_f!(f_new, f_old, rhs, up_idx, down_idx, up_incr, dep_idx, n, bc, use_SL)
    @boundscheck n == length(f_new) || throw(BoundsError(f_new))
    @boundscheck n == length(rhs) || throw(BoundsError(rhs))
    @boundscheck n == length(dep_idx) || throw(BoundsError(dep_idx))
    @boundscheck n == length(f_old) || throw(BoundsError(f_old))

    if use_SL
        # do not update the upwind boundary, where the constant incoming BC has been imposed
        if bc != "periodic"
            f_new[up_idx] = f_old[up_idx]
            istart = up_idx-up_incr
        else
            istart = up_idx
        end
        #@inbounds for i ∈ up_idx-up_incr:-up_incr:down_idx
        @inbounds for i ∈ up_idx:-up_incr:down_idx
            # dep_idx is the index of the departure point for the approximate
            # characteristic passing through grid point i
            # if semi-Lagrange is not used, then dep_idx = i
            idx = dep_idx[i]
            if idx != up_idx + up_incr
                f_new[i] = f_old[idx] + rhs[i]
            else
                # if departure index is beyond upwind boundary, then
                # set updated value along characteristic equal to the old
                # value at the boundary; i.e., assume f is constant
                # beyond the upwind boundary
                f_new[i] = f_old[up_idx]
            end
        end
    else
        #@inbounds for i ∈ up_idx:-up_incr:down_idx
        for i ∈ up_idx:-up_incr:down_idx
            f_new[i] += rhs[i]
        end
    end
    return nothing
end

end
