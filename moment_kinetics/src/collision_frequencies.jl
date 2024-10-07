module collision_frequencies

export get_collision_frequency_ii, get_collision_frequency_ee, get_collision_frequency_ei

using ..reference_parameters: get_reference_collision_frequency_ii,
                              get_reference_collision_frequency_ee,
                              get_reference_collision_frequency_ei
using ..reference_parameters: setup_reference_parameters

"""
    get_collision_frequency_ii(collisions, n, vth)

Calculate the ion-ion collision frequency, depending on the settings/parameters in
`collisions`, for the given density `n` and thermal speed `vth`.

`n` and `vth` may be scalars or arrays, but should have shapes that can be broadcasted
together.
"""
function get_collision_frequency_ii(collisions, n, vth)
    # extract krook options from collisions struct
    colk = collisions.krook
    nuii0 = colk.nuii0
    frequency_option = colk.frequency_option
    if frequency_option ∈ ("reference_parameters", "collisionality_scan")
        return @. nuii0 * n * vth^(-3)
    elseif frequency_option == "manual"
        # Include 0.0*n so that the result gets promoted to an array if n is an array,
        # which hopefully means this function will have a fixed return type given the
        # types of the arguments (we don't want to be 'type unstable' for array inputs by
        # returning a scalar from this branch but an array from the "reference_parameters"
        # branch).
        return @. nuii0 + 0.0 * n
    elseif frequency_option == "none"
        # Include 0.0*n so that the result gets promoted to an array if n is an array,
        # which hopefully means this function will have a fixed return type given the
        # types of the arguments (we don't want to be 'type unstable' for array inputs by
        # returning a scalar from this branch but an array from the "reference_parameters"
        # branch).
        return @. 0.0 * n
    else
        error("Unrecognised option [krook_collisions] "
              * "frequency_option=$(frequency_option)")
    end
end

"""
    get_collision_frequency_ee(collisions, n, vthe)

Calculate the electron-electron collision frequency, depending on the settings/parameters
in `collisions`, for the given density `n` and electron thermal speed `vthe`.

`n` and `vthe` may be scalars or arrays, but should have shapes that can be broadcasted
together.
"""
function get_collision_frequency_ee(collisions, n, vthe)
    # extract krook options from collisions struct
    colk = collisions.krook
    nuee0 = colk.nuee0
    frequency_option = colk.frequency_option
    if frequency_option == "reference_parameters"
        return @. nuee0 * n * vthe^(-3)
    elseif frequency_option == "manual"
        # Include 0.0*n so that the result gets promoted to an array if n is an array,
        # which hopefully means this function will have a fixed return type given the
        # types of the arguments (we don't want to be 'type unstable' for array inputs by
        # returning a scalar from this branch but an array from the "reference_parameters"
        # branch).
        return @. nuee0 + 0.0 * n
    elseif frequency_option == "none"
        # Include 0.0*n so that the result gets promoted to an array if n is an array,
        # which hopefully means this function will have a fixed return type given the
        # types of the arguments (we don't want to be 'type unstable' for array inputs by
        # returning a scalar from this branch but an array from the "reference_parameters"
        # branch).
        return @. 0.0 * n
    else
        error("Unrecognised option [krook_collisions] "
              * "frequency_option=$(frequency_option)")
    end
end

"""
    get_collision_frequency_ei(collisions, n, vthe)

Calculate the electron-electron collision frequency, depending on the settings/parameters
in `collisions`, for the given density `n` and electron thermal speed `vthe`.

`n` and `vthe` may be scalars or arrays, but should have shapes that can be broadcasted
together.
"""
function get_collision_frequency_ei(collisions, n, vthe)
    # extract krook options from collisions struct
    colk = collisions.krook
    nuei0 = colk.nuei0
    frequency_option = colk.frequency_option
    if frequency_option == "reference_parameters"
        return @. nuei0 * n * vthe^(-3)
    elseif frequency_option == "manual"
        # Include 0.0*n so that the result gets promoted to an array if n is an array,
        # which hopefully means this function will have a fixed return type given the
        # types of the arguments (we don't want to be 'type unstable' for array inputs by
        # returning a scalar from this branch but an array from the "reference_parameters"
        # branch).
        return @. nuei0 + 0.0 * n
    elseif frequency_option == "none"
        # Include 0.0*n so that the result gets promoted to an array if n is an array,
        # which hopefully means this function will have a fixed return type given the
        # types of the arguments (we don't want to be 'type unstable' for array inputs by
        # returning a scalar from this branch but an array from the "reference_parameters"
        # branch).
        return @. 0.0 * n
    else
        error("Unrecognised option [krook_collisions] "
              * "frequency_option=$(frequency_option)")
    end
end

end