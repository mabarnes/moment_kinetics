module semi_lagrange

export setup_semi_lagrange
export update_crossing_times!
export find_departure_points!
export project_characteristics_onto_grid!

using ..type_definitions: mk_float, mk_int
using ..array_allocation: allocate_shared_float, allocate_shared_int
using ..communication: block_rank, MPISharedArray

# structure semi_lagrange_info contains the basic information needed
# to project backwards along approximate characteristics, which
# underpins the semi-Lagrange approach to time advancement
struct semi_lagrange_info{N}
    # crossing_time is the time required to cross a given cell
    # moving at a specified advection speed
    crossing_time::MPISharedArray{mk_float,N}
    # trajectory_time is the cumulative trajectory time along a characteristic
    trajectory_time::MPISharedArray{mk_float,N}
    # dep_pts are the departure points at time level m for characteristic
    # arriving at time level m+1
    dep_pts::MPISharedArray{mk_float,N}
    # dep_idx are the indices of the nearest downwind grid point to
    # the departure point (which is in general between grid points)
    dep_idx::MPISharedArray{mk_int,N}
    # characteristic_speed is the approximate characteristic speed
    characteristic_speed::MPISharedArray{mk_float,N}
    # n_transits is the number of times that the characteristic
    # crosses the upwind bounary; only needed if BC = "periodic"
    n_transits::MPISharedArray{mk_int,N}
end
# create and return a structure containing the arrays needed for the
# semi-Lagrange time advance
function setup_semi_lagrange(dims...)
    # create an array to hold crossing times for each cell
    crossing_time = allocate_shared_float(dims...)
    # create an array to hold cumulative trajectory time for each characteristic
    trajectory_time = allocate_shared_float(dims...)
    # create array for the departure points at time level m
    dep_pts = allocate_shared_float(dims...)
    # create array for the indices of the nearest downwind grid point to
    # the departure point (which is in general between grid points)
    dep_idx = allocate_shared_int(dims...)
    # initialize the departure indices to be the save as the arrival indices
    # this allows for a clean algorithm that can use semi-Lagrange or not
    # without a lot of conditional statements within inner loops
    if block_rank[] == 0
        for i ∈ CartesianIndices(dep_idx)
            dep_idx[i] = i[1]
        end
    end
    # create array containing approximate characteristic speed
    # initialize characteristic speed to zero, in case one wants to
    # do a time advance without use of semi-Lagrange treatment
    characteristic_speed = allocate_shared_float(dims...)
    if block_rank[] == 0
        characteristic_speed .= 0.0
    end
    # create array to contain the number of times the characteristics cross
    # the upwind boundary
    n_transits = allocate_shared_int(dims...)
    if block_rank[] == 0
        n_transits .= 0
    end
    # store all of this information in a structure and return it
    return semi_lagrange_info(crossing_time, trajectory_time, dep_pts, dep_idx,
        characteristic_speed, n_transits)
end
function find_approximate_characteristic!(SL, advection, i_outer, coord, dt)
    # calculate the time required to cross the cell associated with each
    # grid point based on the cell width and advection speed
    @views update_crossing_times!(SL.crossing_time, advection.speed[:,i_outer],
                                  coord.cell_width)
    # integrate backward in time from time level m+1 to level m
    # along approximate characteristics determined by speed profile at level m
    # to obtain departure points.  these will not correspond
    # in general to grid points at time level m
    @views find_departure_points!(SL, i_outer, coord, advection.speed[:,i_outer],
                                  advection.upwind_idx[i_outer],
                                  advection.downwind_idx[i_outer],
                                  advection.upwind_increment[i_outer], dt)
    # redefine v₀ slightly so
    # that departure point corresonds to nearest grid point
    # this avoids the need to do interpolation to obtain
    # function values off the fixed grid in z
    @views project_characteristics_onto_grid!(SL, advection.modified_speed[:,i_outer],
                                              coord, dt, advection.upwind_idx[i_outer],
                                              advection.upwind_increment[i_outer])
end
# obtain the time needed to cross the cell
# assigned to each grid point, given the
# advection speed within the cell
# d is an array contain cell widths, with d[1]=d[n] the distance between the 1st and 2nd grid points
# v is an array containing the speeds at grid points
# NB: assuming constant advection speed within each cell, taken as speed
# NB: at neighboring downwind grid point
# NB: might be better and not much harder to analytically solve
# NB: for crossing time assuming, e.g., linear variation of advection speed
function update_crossing_times!(crossing_time, v, d)
    # d contains the cell width associated with
    # each of the m grid points
    n = length(d)
    # ensure that the arrays containing the crossing time for each cell (t)
    # and the speed associated with each cell (v) are inbounds
    @boundscheck n == length(crossing_time) || throw(BoundsError(crossing_time))
    @boundscheck n == length(v) || throw(BoundsError(v))
    # modify indexing of crossing_time slightly depending on sign
    # of advection speed; this simplifies loops used later
    # to calculate characteristics
    # NB: have assumed here that the advection speed does not change sign
    # NB: when moving along this spatial coordinate
    if v[1] > 0
        # the crossing time for each cell is the width of the cell
        # divided by a measure of the speed of particles within the cell;
        # note that crossing_time[1] = crossing_time[n] = crossing time for the right-most cell;
        @inbounds for i ∈ 2:n
            crossing_time[i] = abs(2.0*d[i-1]/(v[i]+v[i-1]))
            #crossing_time[i] = abs(d[i-1]/v[i])
        end
        # crossing_time[1] should not be needed unless periodic BCs are used (if then)
        crossing_time[1] = crossing_time[n]
    else
        # the crossing time for each cell is the width of the cell
        # divided by a measure of the speed of particles within the cell;
        # note that crossing_time[n] = crossing_time[1] = crossing time for the left-most cell;
        @inbounds for i ∈ 1:n-1
            #crossing_time[i] = abs(d[i]/v[i])
            crossing_time[i] = abs(2.0*d[i]/(v[i]+v[i+1]))
        end
        # crossing_time[n] should not be needed unless periodic BCs are used (if then)
        crossing_time[n] = crossing_time[1]
    end

    return nothing
end
# follow trajectory from each grid point at time level n+1
# backwards in time until Δt has elapsed to find
# the departure points and the indices of the nearest downwind grid points
# to the departure points
# SL.dep_pts: array containing the locations of the departure points
# SL.dep_idx: array containing the indices of the nearest downwind
# grid point to the departure point
# tbound: array used to store the cumulative time spent by a particle moving along its
# characteristic backwards from the downwind boundary.  only needed if BC is periodic
# SL.crossing_time: array containing the amount of time required to cross each cell
# speed: the advection speed at the grid point locations given by coord.grid
# coord.grid: grid point locations
# dt: time step size
#function find_departure_points!(dep, dep_idx, tbound, tcell, v, coord, bc, dt)
function find_departure_points!(SL, coord, speed, upwind_idx, downwind_idx,
    upwind_increment, dt)
    # n is the number of grid points along this coordinate axis
    n = coord.n
    # ensure that all of the arrays used in this function are inbounds
    @boundscheck n == size(SL.crossing_time, 1) || throw(BoundsError(SL.crossing_time))
    @boundscheck n == size(SL.dep_idx, 1) || throw(BoundsError(SL.dep_idx))
    @boundscheck n == size(SL.dep_pts, 1) || throw(BoundsError(SL.dep_pts))
    @boundscheck n == size(speed, 1) || throw(BoundsError(speed))

    @inbounds begin
        SL.n_transits .= 0.0
        if coord.bc != "periodic"
            # if the BC is not periodic, then the departure point for the
            # characteristic terminating at the upwind boundary originates
            # beyond the upwind boundary (and thus off the grid).
            # constant or zero BC will be used to fix the boundary value.
            # set departure point to be at upwind boundary
            SL.dep_pts[upwind_idx,i_outer] = coord.grid[upwind_idx]
            # set the departure index to be just upwind of the upwind boundary;
            # will be used during update of f to impose zero/constant incoming BC
            SL.dep_idx[upwind_idx,i_outer] = upwind_idx + upwind_increment
            iend = upwind_idx-upwind_increment
        else
            # if periodic boundary condition, will be most efficient to store
            # the time taken by a characteristic to pass from the upwind=downwind boundary
            # to each point on the grid (tbound);
            # this is needed by the point at the upwind/downwind
            # boundary and some of this information may also be needed by points downwind
            # of the boundary whose characteristics originate upwind of the upwind boundary.
            @views calculate_time_from_boundary!(coord.scratch,
                SL.crossing_time[:,i_outer], upwind_idx, upwind_increment, downwind_idx,
                dt)
            iend = upwind_idx
        end
        # start with the characteristic at the furthest point downwind
        jstart = downwind_idx
        # ttotal = cumulative time integrating backward along trajectory;
        # should be initialized to zero to start trajectory tracing
        ttotal_in = 0.0
        # sweep upwind and calculate departure points for each characteristic
        #for i ∈ downwind_idx:upwind_increment:upwind_idx-upwind_increment
        for i ∈ downwind_idx:upwind_increment:iend
            # calculate the departure point for the ith characteristic;
            # updates SL.dep_pts, SL.dep_idx, and ttotal_out
            ttotal_out = departure_point!(SL, i_outer, ttotal_in, i, jstart,
                coord.grid, speed, dt, upwind_idx, upwind_increment, downwind_idx,
                coord.bc, coord.scratch)
            # jstart will be the grid point to start the time integration
            # for the characteristic immediately upwind of the ith one.
            # generic case is to start sweeping upwind starting at the
            # departure point for the ith characteristic.
            jstart = SL.dep_idx[i,i_outer]
            # account for cases where departure point is less than one grid
            # spacing away from arrival point
            if ttotal_out < SL.crossing_time[SL.dep_idx[i,i_outer],i_outer]
                # in this case, jstart should be the neighboring upwind point
                # from the departure=arrival point of the ith characteristic
                if upwind_increment < 0
                    jstart = max(i+upwind_increment,iend)
                else
                    jstart = min(i+upwind_increment,iend)
                end
            end
            # the cumulative time spent on the i-1 characteristic in integrating
            # backward from its arrival point to the grid point corresponding
            # to jstart
            ttotal_in = max(0.,ttotal_out - SL.crossing_time[i,i_outer])
        end
    end
    return nothing
end
# calculate the departure point for the ith characteristic;
# overwrites SL.dep_pts[i], SL.dep_idx[i], and returns ...
# note that the dep_idx calculated here is the index of the nearest
# downwind gridpoint to the departure point (which is in general off-grid)
function departure_point!(SL, i_outer, t_in, i, jstart, grid, v, dt, upwind_idx,
    upwind_increment, downwind_idx, bc, tbound)
    # t_in is the time spent by a particle following this ith characteristic
    # in going from the arrival point (grid point i at future time level) to the
    # grid point jstart;
    # note that the jstart grid point is the grid point immediately
    # downwind of the departure point for the characteristic
    # immediately downwind of this (ith) one.
    t_out, dep_pt_found = departure_point_single_transit!(SL, i_outer, t_in, i, jstart,
        grid, v, dt, upwind_idx, upwind_increment)
    # if dep_pt_found = false, then the departure point for this characteristic
    # was not encountered during a single transit of the domain
    if dep_pt_found == false
        @inbounds begin
            if bc != "periodic"
                # set departure point to be the upwind boundary, which is given
                # by incoming boundary condition
                SL.dep_pts[i,i_outer] = grid[upwind_idx]
                # similarly, dep_idx must be set to the upwind boundary index
                SL.dep_idx[i,i_outer] = upwind_idx + upwind_increment
            else
                # if the time taken to cross all grid points is less than dt-t_out,
                # calculate the number of times the domain must be crossed before
                # this is no longer true
                if tbound[upwind_idx] <= (dt - t_out)
                    tmp = 1 + convert(mk_int,fld(dt-t_out, tbound[upwind_idx]))
                    @. SL.n_transits[i:upwind_increment:upwind_idx,i_outer] += tmp
                    # update trajectory time to account for additional transits of domain
                    t_out += (tmp-1)*tbound[upwind_idx]
                else
                    @. SL.n_transits[i:upwind_increment:upwind_idx,i_outer] += 1
                end
                # remainder is the difference between dt and the time taken
                # to go from the arrival point to the downwind boundary
                # after the above n_transits additional transits
                remainder = (dt-t_out) % tbound[upwind_idx]
                # find the departure point and the
                # grid point immediately downwind of the departure point
                for j ∈ downwind_idx+upwind_increment:upwind_increment:upwind_idx
                    if remainder < tbound[j]
                        jmod = j-upwind_increment
                        SL.dep_idx[i,i_outer] = jmod
                        SL.dep_pts[i,i_outer] = grid[jmod] - v[jmod]*(remainder-tbound[SL.dep_idx[i,i_outer]])
                        t_out += tbound[SL.dep_idx[i,i_outer]]
                        break
                    end
                end
            end
        end
    end
    return t_out
end
function departure_point_single_transit!(SL, i_outer, t_in, i, jstart, grid, v, dt, upwind_idx,
    upwind_increment)
    # t_in is the time spent by a particle following this ith characteristic
    # in going from the arrival point (grid point i at future time level) to the
    # grid point jstart;
    # note that the jstart grid point is the grid point immediately
    # downwind of the departure point for the characteristic
    # immediately downwind of this (ith) one.
    t_out = t_in
    @inbounds begin
        for j ∈ jstart:upwind_increment:upwind_idx-upwind_increment
            tmp = t_out + SL.crossing_time[j,i_outer]
            if tmp >= dt
                SL.dep_pts[i,i_outer] = grid[j] - v[j]*(dt-t_out)
                SL.dep_idx[i,i_outer] = j
                dep_pt_found = true
                return t_out, dep_pt_found
            else
                t_out = tmp
            end
        end
    end
    # departure point not found before reaching upwind element boundary
    dep_pt_found = false
    # return the cumulative time elapsed in following the characteristic to the
    # upwind boundary (including time from any previous transits of the domain)
    return t_out, dep_pt_found
end
# if periodic boundary condition, will be most efficient to store
# the time taken by a characteristic to pass from the downwind boundary
# to each point on the grid, as this will be used for all characteristics
# that wrap around from -L/2 to L/2
function calculate_time_from_boundary!(tbound, tcell, upwind_idx, upwind_incr, downwind_idx, dt)
    # tbound is the time spent on a characteristic in getting from
    # the downdiwnd boundary to the ith grid point
    @boundscheck abs(upwind_idx+upwind_incr-downwind_idx) == length(tbound) || throw(BoundsError(tbound))
    @inbounds begin
        # starting at downwind boundary, so no time needed to get there
        tbound[downwind_idx] = 0.
        # sweep upwind, calculating the integrated time out to each grid point
        for i ∈ downwind_idx+upwind_incr:upwind_incr:upwind_idx
            idx = i-upwind_incr
            if upwind_incr > 0
                idxmod = idx
            else
                idxmod = idx
            end
            tbound[i] = tbound[idx] + tcell[idxmod]
            # if the time from the boundary to this grid point is greater
            # than the time step size, then no characteristics will depart
            # further upwind; no need to calculate tbound for points upwind.
            if tbound[i] > dt
                # set tbound at upwind boundary equal to the largest calculated
                # value of tbound; this is used when finding departure points
                # to check if multiple transits of the domain must occur
                # before the trajectory time is larger than dt
                tbound[upwind_idx] = tbound[i]
                break
            end
        end
    end
    return nothing
end
#=
# calculate the departure point for the ith characteristic
# overwrites dep[i], dep_idx[i], and ttotal[i]
# note that the dep_idx calculated here is the index of the nearest
# downwind gridpoint to the departure point (which is in general offgrid)
function departure_point_periodic!(dep, dep_idx, total, i, jstart, tcell, z, v, dt)
    @boundscheck jstart <= length(z) || throw(BoundsError(z))
    @boundscheck jstart <= length(v) || throw(BoundsError(v))
    @boundscheck jstart <= length(tcell) || throw(BoundsError(tcell))
    @inbounds begin
        for j ∈ jstart:-1:2
            tmp = total[i] + tcell[j]
            if tmp >= dt
                dep[i] = z[j] - v[j]*(dt-total[i])
                dep_idx[i] = j
                return nothing
            else
                total[i] = tmp
            end
        end
        j = jstart
        while total[i] < dt
            tmp = total[i] + tcell[j]
            if tmp >= dt
                dep[i] = z[j] - v[j]*(dt-total[i])
                dep_idx[i] = j
                total[i] = dt
            else
                total[i] = tmp
            end
            if j == 2
                j = m
            else
                j -= 1
            end
        end


        # departure point not found before reaching upwind element boundary, so
        # set departure point to be the upwind boundary, which is given
        # by zero incoming boundary condition
        dep[i] = z[1]
        # similarly, dep_idx must be set to the upwind point nearest the boundary
        # that is not itself the boundary
        dep_idx[i] = 1
    end
    return nothing
end
=#
# determine the nearest grid point to each departure point
#function project_characteristics_onto_grid!(dep_idx, dep, vc, z, dz, dt)
function project_characteristics_onto_grid!(SL, i_outer, mod_speed, coord, dt, upwind_idx, upwind_increment)
    # n is the number of grid points along this coordinate axis
    n = coord.n
    # ensure arrays used in this function are inbounds
    @boundscheck n == size(SL.dep_idx, 1) || throw(BoundsError(SL.dep_idx))
    @boundscheck n == size(SL.dep_pts, 1) || throw(BoundsError(SL.dep_pts))
    @boundscheck n == size(mod_speed, 1) || throw(BoundsError(mod_speed))
    @boundscheck n == size(coord.cell_width, 1) || throw(BoundsError(coord.cell_width))
    @boundscheck n == size(SL.characteristic_speed, 1) ||
        throw(BoundsError(SL.characteristic_speed))

    @inbounds for i ∈ 1:n
        idx = SL.dep_idx[i,i_outer]
        if idx == upwind_idx + upwind_increment
            # departure point/index already found to be upwind of upwind boundary;
            # only possible for non-periodic boundary condition, so okay to use
            # upwind boundary as departure point/index;
            # this has already been set when finding departure points
            # similarly, no need to update the characteristic speed,
            # as exact speed from time level n gives a departure point
            # where the function value is the same as the upwind boundary.
            # NB: need to revisit this, but setting characteristic_speed to be
            # NB: the opposite sign to the advection speed
            # NB: for the moment as a way of identifying characteristics
            # NB: originating beyond the grid boundary
            SL.characteristic_speed[i,i_outer] = real(upwind_increment)
            # NB: need to set mod_speed to something sensible here?
        else
            # account for fact that cell_width[1] is first cell, etc.
            # whereas idx is defined to be nearest downwind grid point from
            # departure point; thus the cell width we probe is idx if
            # the advection speed is negative and idx-1 if the advection speed is positive
            if upwind_increment > 0
                idxmod = idx
            else
                idxmod = idx - 1
            end
            if abs(coord.grid[idx]-SL.dep_pts[i,i_outer]) < 0.5*coord.cell_width[idxmod]
                # no need to do anything with dep_idx, as this is actually
                # the closest grid point to the departure point
            else
                # the nearest grid point to the departure point is upwind
                # of the departure point, so modify the dep_idx accordingly;
                # note that it should not be possible for idx to be at the
                # upwind boundary, as it is always the nearest downwind point
                # that is used.  thus, there should be no possibility
                # of having dep_idx beyond the upwind boundary.
                SL.dep_idx[i,i_outer] = idx + upwind_increment
            end
            # the factor wrapped_distance accounts for contribution to the
            # distance travelled by characteristics in wrapping around the domain,
            # possibly multiple times
            wrapped_distance = -upwind_increment*SL.n_transits[i,i_outer]*coord.L
            # arrival_grid is the grid location at the future time level m+1
            # from which we start following the characteristic backwards in time
            arrival_grid = coord.grid[i]
            # departure_grid is the nearest grid point to the departure point
            # of the characteristic at time level m
            departure_grid = coord.grid[SL.dep_idx[i,i_outer]]
            # departure_point is the location of the characteristic at time level m
            # that passes through arrival_grid at time level m+1
            departure_point = SL.dep_pts[i,i_outer]
            # update the characteristic speed to account
            # for the trajectory slope change needed to make departure point
            # be the nearest grid point.
            # despite the fact that the spatially-dependent speed at time
            # level n was used to find the departure point, one can replace
            # this with the constant velocity along the ith characteristic
            # that was needed to arrive at the departure grid point
            #NB: need to worry about the sign of these differences for case
            #NB: with negative advection speed?
            SL.characteristic_speed[i,i_outer] = (wrapped_distance + (arrival_grid-departure_grid))/dt
            # despite the fact that the spatially-dependent speed at time
            # level n was used to find the departure point, one can replace
            # this with the constant velocity along the ith characteristic
            # that was needed to arrive at the departure point
            mod_speed[i] = (wrapped_distance + (arrival_grid-departure_point))/dt
        end
    end
    return nothing
end

end
