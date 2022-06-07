"""
Some shared functions used by MMS tests
"""
module MMSTestUtils

export increase_resolution, get_and_check_ngrid, set_ngrid, test_error_series

using moment_kinetics.type_definitions

"""
    increase_resolution(input::Dict, factor)

Increase resolution of simulation by multiplying the numbers of elements `*_nelement` in
the `input` settings by `factor`.
"""
function increase_resolution(input::Dict, nelement)
    result = copy(input)
    result["run_name"] = input["run_name"] * "_$nelement"
    for key ∈ keys(result)
        if occursin("_nelement", key)
            if occursin("v", key)
                result[key] = 4 * nelement
            else
                result[key] = nelement
            end
        end
    end

    return result
end

"""
    get_and_check_ngrid(input::Dict)

Get value of `ngrid` and check that it is the same for all dimensions. `ngrid` needs to
be the same as it sets the convergence order, and we want this to be the same for all
operators.
"""
function get_and_check_ngrid(input::Dict)::mk_int
    ngrid = nothing

    for key ∈ keys(input)
        if occursin("_ngrid", key)
            if ngrid === nothing
                ngrid = input[key]
            else
                if ngrid != input[key]
                    error("*_ngrid should all be the same, but $key=$(input[key]) when "
                          * "we already found ngrid=$ngrid")
                end
            end
        end
    end

    return ngrid
end

"""
    set_ngrid(input::Dict, ngrid::mk_int)

Set value of `ngrid`, the same for all dimensions.
"""
function set_ngrid(input::Dict, ngrid::mk_int)
    for key ∈ keys(input)
        if occursin("_ngrid", key)
            input[key] = ngrid
        end
    end

    return nothing
end

"""
    test_error_series(errors::Vector{mk_float}, resolution_factors::Vector,
                      expected_order, expected_lowest)

Test whether the error norms in `errors` converge as expected with increases in
resolution by `resolution_factors`. `expected_order` is the order p such that the error
is expected to be proportional to h^p. `expected_lowest` is the expected value of the
error at the lowest resolution (used as a regression test).

Note the entries in `errors` and `resolution_factors` should be sorted in increasing
order of `resolution_factors`.
"""
function test_error_series(errors::Vector{mk_float}, resolution_factors::Vector,
                           expected_order, expected_lowest)
    error_factors = errors[1:end-1] ./ errors[2:end]
    expected_factors = resolution_factors[2:end].^expected_order
end

end # MMSTestUtils

using .MMSTestUtils
