module advection

export setup_advection
export update_advection_factor!
export calculate_explicit_advection!
export update_boundary_indices!
export advance_f_local!

using Base.Iterators: take, rest
using NamedDims

using ..type_definitions: mk_float, mk_int
using ..array_allocation: allocate_shared_float, allocate_shared_int, drop_dim
using ..calculus: derivative!
using ..communication: block_rank, block_synchronize, MPISharedArray
using ..coordinates: coordinate

# structure containing the basic arrays associated with the
# advection terms appearing in the advection equation for each coordinate
mutable struct advection_info{D1,D2,D3,N1,N2,N3}
    # rhs is the sum of the advection terms appearing on the righthand side
    # of the equation
    rhs::MPISharedArray{D1, mk_float, N1}
    # df is the derivative of the distribution function f with respect
    # to the coordinate associated with this set of advection terms
    # it has dimensions of nelement x ngrid_per_element
    df::MPISharedArray{D2, mk_float, N2}
    # speed is the component of the advection speed along this coordinate axis
    speed::MPISharedArray{D1, mk_float, N1}
    # if using semi-Lagrange approach,
    # modified_speed is delta / dt, where delta for a given characteristic
    # is the displacement from the arrival point to the
    # (generally off-grid) departure point using the coordinate in which
    # the grid is equally spaced (a re-scaling of the Chebyshev theta coordinate);
    # otherwise, modified_speed = speed
    modified_speed::MPISharedArray{D1, mk_float, N1}
    # adv_fac is the advection factor that multiplies df in the advection term
    adv_fac::MPISharedArray{D1, mk_float, N1}
    # upwind_idx is the boundary index for the upwind boundary
    upwind_idx::MPISharedArray{D3, mk_int, N3}
    # downwind_idx is the boundary index for the downwind boundary
    downwind_idx::MPISharedArray{D3, mk_int, N3}
    # upwind_increment is the index increment used when sweeping in the upwind direction
    upwind_increment::MPISharedArray{D3, mk_int, N3}
end
"""
Create arrays needed to compute the advection term(s) for a 1D problem

Arguments
---------
nspec : mk_int
    Number of species
coords : name=coord::coordinate
    Keyword arguments give the name (key) and coordinate (value) for the phase space
    dimensions. The first keyword argument is the dimension in which advection is
    calculated with the structs created here. The remaining keyword arguments are the
    other dimensions - the order that the other dimensions/coordinates are given in does
    not matter.

Returns
-------
Vector{advection_info}
    Vector of structs containing, for each species, the information and temporary arrays
    needed to calculate advection in the dimension corresponding to the first of the
    coords keyword arguments.
"""
function setup_advection(nspec, ::Val{advection_dim}, advection_coord::coordinate,
                         ::Val{dims}; other_dim_sizes...) where {advection_dim,dims}
    # allocate an array containing structures with much of the info needed
    # to do the 1D advection time advance
    return [setup_advection_per_species(Val(advection_dim), advection_coord, Val(dims);
                                        other_dim_sizes...) for _ ∈ 1:nspec]
end
# Create arrays needed to compute the advection term(s).
# The first coordinate argument is the dimension in which advection is calculated with
# the struct created here. The following arguments are the remaining phase space
# dimensions, and their order does not matter.
function setup_advection_per_species(
        ::Val{advection_dim}, advection_coord::coordinate, ::Val{dims};
        other_dim_sizes...) where {advection_dim,dims}
    advection_dims = Val(NamedDims.expand_dimnames((advection_dim,), dims))
    other_dims = drop_dim(Val(advection_dim), Val(dims))
    advection_gridelement_dims = Val(NamedDims.expand_dimnames((:grid,:element), other_dims))
    # create array for storing the explicit advection terms appearing
    # on the righthand side of the equation
    rhs = allocate_shared_float(advection_dims;
        advection_dim => advection_coord.n, other_dim_sizes...)
    # create array for storing ∂f/∂(coordinate)
    # NB: need to store on nelement x ngrid_per_element array, as must keep info
    # about multi-valued derivative at overlapping point at element boundaries
    df = allocate_shared_float(advection_gridelement_dims;
        grid=advection_coord.ngrid, element=advection_coord.nelement, other_dim_sizes...)
    # create array for storing the advection coefficient
    adv_fac = allocate_shared_float(advection_dims;
        advection_dim => advection_coord.n, other_dim_sizes...)
    # create array for storing the speed along this coordinate
    speed = allocate_shared_float(advection_dims;
        advection_dim => advection_coord.n, other_dim_sizes...)
    # create array for storing the modified speed along this coordinate
    modified_speed = allocate_shared_float(advection_dims;
        advection_dim => advection_coord.n, other_dim_sizes...)
    # index for the upwind boundary; will be updated before use so value irrelevant
    upwind_idx = allocate_shared_int(other_dims; other_dim_sizes...)
    # index for the downwind boundary; will be updated before use so value irrelevant
    downwind_idx = allocate_shared_int(other_dims; other_dim_sizes...)
    # index increment used when sweeping in the upwind direction; will be updated before use
    upwind_increment = allocate_shared_int(other_dims; other_dim_sizes...)
    if block_rank[] == 0
        upwind_idx[:] .= 1
        downwind_idx[:] .= advection_coord.n
        upwind_increment[:] .= -1
    end
    block_synchronize()
    # return advection_info struct containing necessary arrays
    return advection_info(rhs, df, speed, modified_speed, adv_fac, upwind_idx, downwind_idx, upwind_increment)
end

"""
Calculate the grid index correspond to the upwind and downwind boundaries, as well as
the index increment needed to sweep in the upwind direction

Arguments
---------
advection : Vector{advection_info}(n_orthogonal)
    structs containing information on how to advect in a direction. Has as many entries
    as there are grid points in the dimension orthogonal to the advection direction.
orthogonal_coordinate : coordinate
    coordinate struct for the dimension orthogonal to the advection direction, used for
    information needed to iterate over the orthogonal coordinate.
"""
function update_boundary_indices!(advection, orthogonal_coordinate_range)
    n = size(advection.speed,1)
    for j ∈ orthogonal_coordinate_range
        # NB: for now, assume the speed has the same sign at all grid points
        # so only need to check its value at one location to determine the upwind direction
        if advection.speed[1,j] > 0
            advection.upwind_idx[j] = 1
            advection.upwind_increment[j] = -1
            advection.downwind_idx[j] = n
        else
            advection.upwind_idx[j] = n
            advection.upwind_increment[j] = 1
            advection.downwind_idx[j] = 1
        end
    end
    return nothing
end
# calculate the factor appearing in front of f' in the advection term
# at time level n in the frame moving with the approximate characteristic
function update_advection_factor!(adv_fac, speed, upwind_idx, downwind_idx,
    upwind_increment, SL, i_outer, n, dt, j, coord)
    @boundscheck n == size(SL.dep_idx, 1) || throw(BoundsError(SL.dep_idx))
    @boundscheck n == length(adv_fac) || throw(BoundsError(adv_fac))
    @boundscheck n == length(speed) || throw(BoundsError(speed))
    @boundscheck n == size(SL.characteristic_speed, 1) ||
        throw(BoundsError(SL.characteristic_speed))
    #NB: commented out line below needed for bc != periodic?
    #@inbounds for i ∈ upwind_idx-upwind_increment:-upwind_increment:downwind_idx
    #@inbounds begin
    if j == 1
        for i ∈ upwind_idx:-upwind_increment:downwind_idx
            idx = SL.dep_idx[i,i_outer]
            # only need to calculate advection factor for characteristics
            # that originate within the domain, as zero/constant incoming BC
            # takes care of the rest.
            if idx != upwind_idx + upwind_increment
                # the effective advection speed appearing in the advection term
                # is the speed in the frame moving with the approximate
                # characteristic speed v_char
                adv_fac[i] = -dt*(speed[idx]-SL.characteristic_speed[i,i_outer])
            end
        end
    else
        # NB: need to change v[idx] to v[i] for second iteration of RK -
        # otherwise identical to loop in first branch
        for i ∈ upwind_idx:-upwind_increment:downwind_idx
            idx = SL.dep_idx[i,i_outer]
            if idx != upwind_idx + upwind_increment
                adv_fac[i] = -dt*(speed[i]-SL.characteristic_speed[i,i_outer])
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
function update_rhs!(advection, i_outer, f_current, SL, coord, dt, j, spectral)
    # calculate the factor appearing in front of df/dcoord in the advection
    # term at time level n in the frame moving with the approximate
    # characteristic
    @views update_advection_factor!(advection.adv_fac[:,i_outer],
        advection.modified_speed[:,i_outer], advection.upwind_idx[i_outer],
        advection.downwind_idx[i_outer], advection.upwind_increment[i_outer],
        SL, i_outer, coord.n, dt, j, coord)
    # calculate df/dcoord
    @views derivative!(coord.scratch, f_current, coord, advection.adv_fac[:,i_outer], spectral)
    #derivative!(coord.scratch, f_current, coord, spectral)
    # calculate the explicit advection terms on the rhs of the equation;
    # i.e., -Δt⋅δv⋅f'
    @views calculate_explicit_advection!(advection.rhs[:,i_outer], coord.scratch,
        advection.adv_fac[:,i_outer], advection.upwind_idx[i_outer],
        advection.upwind_increment[i_outer], SL.dep_idx[:,i_outer], coord.n, j)
end
# do all the work needed to update f(coord) at a single value of other coords
function advance_f_local!(f_new, f_current, f_old, SL, advection, i_outer, coord, dt, j, spectral, use_SL)
    # update the rhs of the equation accounting for 1d advection in coord
    update_rhs!(advection, i_outer, f_current, SL, coord, dt, j, spectral)
    # update ff at time level n+1 using an explicit Runge-Kutta method
    # along approximate characteristics
    @views update_f!(f_new, f_old, advection.rhs[:,i_outer],
                     advection.upwind_idx[i_outer],
                     advection.downwind_idx[i_outer],
                     advection.upwind_increment[i_outer], SL.dep_idx[:,i_outer], coord.n,
                     coord.bc, use_SL)
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
