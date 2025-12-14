"""
"""
module advection

export setup_advection
export update_advection_factor!
export calculate_explicit_advection!
export advance_f_local!
export advance_f_df_precomputed!

using ..type_definitions: mk_float, mk_int, MPISharedArray
using ..array_allocation: allocate_shared_float, allocate_shared_int
using ..calculus: derivative!
using ..debugging
using ..looping

"""
create arrays needed to compute the advection term(s) for a 1D problem
"""
function setup_advection(coords...)
    # Allocate the object needed to do the 1D advection time advance.
    # Used to be a struct with several arrays, could be again in future if necessary, but
    # for now only an array for the speed is needed.
    advection_speed = allocate_shared_float(coords...)
    # initialise speed to zero so that it can be used safely without
    # introducing NaNs (if left uninitialised) when coordinate speeds
    # are used but the coordinate has only a single point
    # (e.g. dr/dt in dvperp/dt = (vperp/2B)dB/dt, see vperp_advection.jl)
    @serial_region begin
        @. advection_speed = 0.0
    end
    return advection_speed
end

"""
do all the work needed to update f(coord) at a single value of other coords
"""
function advance_f_local!(f_new, f_current, advect, coord, dt, spectral)
    df = coord.scratch
    derivative!(df, f_current, coord, advect, spectral)
    @. f_new += -dt * advect * df
end

function advance_f_df_precomputed!(f_new, df_current, advect, coord, dt)
    @. f_new += -dt * advect * df_current
end

end
