"""
"""
module advection

export setup_advection
export update_advection_factor!
export calculate_explicit_advection!
export advance_f_local!
export advance_f_df_precomputed!
export advection_info

using ..type_definitions: mk_float, mk_int, MPISharedArray
using ..array_allocation: allocate_shared_float, allocate_shared_int
using ..calculus: derivative!
using ..debugging
using ..looping

"""
structure containing the basic arrays associated with the
advection terms appearing in the advection equation for each coordinate
"""
struct advection_info{L}
    # speed is the component of the advection speed along this coordinate axis
    speed::MPISharedArray{mk_float, L}
end

"""
create arrays needed to compute the advection term(s) for a 1D problem
"""
function setup_advection(nspec, coords...)
    # allocate an array containing structures with much of the info needed
    # to do the 1D advection time advance
    advection = [setup_advection_per_species(coords...) for _ ∈ 1:nspec]
    return advection
end

"""
create arrays needed to compute the advection term(s)
"""
function setup_advection_per_species(coords...)
    # create array for storing the speed along this coordinate
    speed = allocate_shared_float(coords...)
    # initialise speed to zero so that it can be used safely without
    # introducing NaNs (if left uninitialised) when coordinate speeds
    # are used but the coordinate has only a single point
    # (e.g. dr/dt in dvperp/dt = (vperp/2B)dB/dt, see vperp_advection.jl)
    @serial_region begin
        @. speed = 0.0
    end
    # return advection_info struct containing necessary arrays
    return advection_info(speed)
end

"""
do all the work needed to update f(coord) at a single value of other coords
"""
function advance_f_local!(f_new, f_current, advection, i_outer, j_outer, k_outer, coord, dt, spectral)
    df = coord.scratch
    speed = @view advection.speed[:,i_outer,j_outer,k_outer]
    derivative!(df, f_current, coord, speed, spectral)
    @. f_new += -dt * speed * df
end

function advance_f_local!(f_new, f_current, advection, i_outer, j_outer, k_outer, l_outer, coord, dt, spectral)
    df = coord.scratch
    speed = @view advection.speed[:,i_outer,j_outer,k_outer,l_outer]
    derivative!(df, f_current, coord, speed, spectral)
    @. f_new += -dt * speed * df
end

function advance_f_df_precomputed!(f_new, df_current, advection, i_outer, j_outer, k_outer, coord, dt)
    speed = @view advection.speed[:,i_outer,j_outer,k_outer]
    @. f_new += -dt * speed * df_current
end

function advance_f_df_precomputed!(f_new, df_current, advection, i_outer, j_outer, k_outer, l_outer, coord, dt)
    speed = @view advection.speed[:,i_outer,j_outer,k_outer,l_outer]
    @. f_new += -dt * speed * df_current
end

end
