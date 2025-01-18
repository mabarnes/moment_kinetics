"""
"""
module advection

export setup_advection
export update_advection_factor!
export calculate_explicit_advection!
export advance_f_local!
export advance_f_df_precomputed!
export advection_info

using ..type_definitions
using ..array_allocation: allocate_shared_float, allocate_shared_int
using ..calculus: derivative!
using ..communication
using ..looping

"""
structure containing the basic arrays associated with the
advection terms appearing in the advection equation for each coordinate
"""
mutable struct advection_info{L,M}
    # rhs is the sum of the advection terms appearing on the righthand side
    # of the equation
    rhs::MPISharedArray{mk_float, L}
    # df is the derivative of the distribution function f with respect
    # to the coordinate associated with this set of advection terms
    # it has dimensions of nelement x ngrid_per_element
    df::MPISharedArray{mk_float, M}
    # speed is the component of the advection speed along this coordinate axis
    speed::MPISharedArray{mk_float, L}
    # adv_fac is the advection factor that multiplies df in the advection term
    adv_fac::MPISharedArray{mk_float, L}
end

"""
create arrays needed to compute the advection term(s) for a 1D problem
"""
function setup_advection(nspec, coords...)
    # allocate an array containing structures with much of the info needed
    # to do the 1D advection time advance
    ncoord = length(coords)
    advection = MKVector{advection_info{ncoord,ncoord+1}}(undef, nspec)
    # store all of this information in a structure and return it
    for is ∈ 1:nspec
        advection[is] = setup_advection_per_species(coords...)
    end
    return advection
end

"""
create arrays needed to compute the advection term(s)
"""
function setup_advection_per_species(coords...)
    # create array for storing the explicit advection terms appearing
    # on the righthand side of the equation
    rhs = allocate_shared_float([coord.n for coord in coords]...)
    # create array for storing ∂f/∂(coordinate)
    # NB: need to store on nelement x ngrid_per_element array, as must keep info
    # about multi-valued derivative at overlapping point at element boundaries
    df = allocate_shared_float(coords[1].ngrid, coords[1].nelement_local,
                               [coord.n for coord in coords[2:end]]...)
    # create array for storing the advection coefficient
    adv_fac = allocate_shared_float([coord.n for coord in coords]...)
    # create array for storing the speed along this coordinate
    speed = allocate_shared_float([coord.n for coord in coords]...)
    # initialise speed to zero so that it can be used safely without
    # introducing NaNs (if left uninitialised) when coordinate speeds
    # are used but the coordinate has only a single point
    # (e.g. dr/dt in dvperp/dt = (vperp/2B)dB/dt, see vperp_advection.jl)
    @serial_region begin
        @. speed = 0.0
    end
    # return advection_info struct containing necessary arrays
    return advection_info(rhs, df, speed, adv_fac)
end

"""
calculate the factor appearing in front of f' in the advection term
at time level n in the frame moving with the approximate characteristic
"""
function update_advection_factor!(adv_fac, speed, n, dt)
    @boundscheck n == length(adv_fac) || throw(BoundsError(adv_fac))
    @boundscheck n == length(speed) || throw(BoundsError(speed))
    for i ∈ 1:n
        adv_fac[i] = -dt*speed[i]
    end
    return nothing
end

"""
calculate the explicit advection terms on the rhs of the equation;
i.e., -Δt⋅δv⋅f'
"""
function calculate_explicit_advection!(rhs, df, adv_fac, n)
    for i ∈ 1:n
        rhs[i] = adv_fac[i]*df[i]
    end
    return nothing
end

"""
update the righthand side of the equation to account for 1d advection in this coordinate
"""
function update_rhs!(advection, i_outer, j_outer, k_outer, f_current, coord, dt, spectral)
    # calculate the factor appearing in front of df/dcoord in the advection
    # term at time level n in the frame moving with the approximate
    # characteristic
    @views update_advection_factor!(advection.adv_fac[:,i_outer,j_outer,k_outer],
        advection.speed[:,i_outer,j_outer,k_outer], coord.n, dt)

    # calculate df/dcoord
    @views derivative!(coord.scratch, f_current, coord, advection.adv_fac[:,i_outer,j_outer,k_outer], spectral)
    #@views derivative!(coord.scratch, f_current, coord, spectral)
    
    #derivative!(coord.scratch, f_current, coord, spectral)
    # calculate the explicit advection terms on the rhs of the equation;
    # i.e., -Δt⋅δv⋅f'
    @views calculate_explicit_advection!(advection.rhs[:,i_outer,j_outer,k_outer], coord.scratch, advection.adv_fac[:,i_outer,j_outer,k_outer], coord.n)
end

function update_rhs!(advection, i_outer, j_outer, k_outer, l_outer, f_current, coord, dt, spectral)
    # calculate the factor appearing in front of df/dcoord in the advection
    # term at time level n in the frame moving with the approximate
    # characteristic
    
    @views update_advection_factor!(advection.adv_fac[:,i_outer,j_outer,k_outer,l_outer],
        advection.speed[:,i_outer,j_outer,k_outer,l_outer], coord.n, dt)
    
    # calculate df/dcoord
    @views derivative!(coord.scratch, f_current, coord, advection.adv_fac[:,i_outer,j_outer,k_outer,l_outer], spectral)
    #@views derivative!(coord.scratch, f_current, coord, spectral)
    
    #derivative!(coord.scratch, f_current, coord, spectral)
    # calculate the explicit advection terms on the rhs of the equation;
    # i.e., -Δt⋅δv⋅f'
    @views calculate_explicit_advection!(advection.rhs[:,i_outer,j_outer,k_outer,l_outer], coord.scratch, advection.adv_fac[:,i_outer,j_outer,k_outer,l_outer], coord.n)
    
end

"""
do all the work needed to update f(coord) at a single value of other coords
"""
function advance_f_local!(f_new, f_current, advection, i_outer, j_outer, k_outer, coord, dt, spectral)
    # update the rhs of the equation accounting for 1d advection in coord
    update_rhs!(advection, i_outer, j_outer, k_outer, f_current, coord, dt, spectral)
    # update f to take into account the explicit advection
    @views update_f!(f_new, advection.rhs[:,i_outer,j_outer,k_outer], coord.n)
end

function advance_f_local!(f_new, f_current, advection, i_outer, j_outer, k_outer, l_outer, coord, dt, spectral)
    # update the rhs of the equation accounting for 1d advection in coord
    update_rhs!(advection, i_outer, j_outer, k_outer, l_outer, f_current, coord, dt, spectral)
    # update f to take into account the explicit advection
    @views update_f!(f_new, advection.rhs[:,i_outer,j_outer,k_outer,l_outer], coord.n)
end

function advance_f_df_precomputed!(f_new, df_current, advection, i_outer, j_outer, k_outer, coord, dt)
    # update the rhs of the equation accounting for 1d advection in coord
    @views calculate_explicit_advection!(advection.rhs[:,i_outer,j_outer,k_outer], df_current, advection.adv_fac[:,i_outer,j_outer,k_outer], coord.n)
	# update f to take into account the explicit advection
    @views update_f!(f_new, advection.rhs[:,i_outer,j_outer,k_outer], coord.n)
end

function advance_f_df_precomputed!(f_new, df_current, advection, i_outer, j_outer, k_outer, l_outer, coord, dt)
    # update the rhs of the equation accounting for 1d advection in coord
    @views calculate_explicit_advection!(advection.rhs[:,i_outer,j_outer,k_outer,l_outer], df_current, advection.adv_fac[:,i_outer,j_outer,k_outer,l_outer], coord.n)
	# update f to take into account the explicit advection
    @views update_f!(f_new, advection.rhs[:,i_outer,j_outer,k_outer,l_outer], coord.n)
end

"""
"""
function update_f!(f_new, rhs, n)
    @boundscheck n == length(f_new) || throw(BoundsError(f_new))
    @boundscheck n == length(rhs) || throw(BoundsError(rhs))
    
    for i ∈ 1:n
        f_new[i] += rhs[i]
    end
    return nothing
end

end
