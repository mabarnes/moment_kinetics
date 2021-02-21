module scan_input

export mk_scan_inputs

# By default, inputs are combined with an 'inner product', i.e. inputs a,b,c
# are combined as (a[1],b[1],c[1]), (a[2],b[2],c[2]), etc.
# Any inputs named in 'combine_outer' are instead combined with an 'outer
# product', i.e. an entry is created for every value of those inputs combined
# with every combination of the other inputs.
const combine_outer = [:charge_exchange_frequency]

const base_name = "scan"

function mk_scan_inputs()
    scan_inputs = Dict()

    #scan_inputs[:nstep] = (6000,)
    #scan_inputs[:dt] = (0.0005/sqrt(0.5),)
    #scan_inputs[:nwrite] = (20,)

    scan_inputs[(:initial_density, 1)] = Float64[0.0001, 0.25, 0.5, 0.75, 0.9999]
    scan_inputs[(:initial_density, 2)] = 1.0 .- scan_inputs[(:initial_density, 1)]

    #scan_inputs[(:initial_temperature, 1)] = (0.25, 0.5, 0.75, 1.0)
    #scan_inputs[(:initial_temperature, 2)] = scan_inputs[(:initial_density, 1)]

    #scan_inputs[(:z_IC_amplitude, 1)] = (.001,)
    #scan_inputs[(:z_IC_amplitude, 2)] = (.001,)

    scan_inputs[:charge_exchange_frequency] = range(0.0, 2.0*π * 2.0, length=20)

    # Combine inputs into single Vector
    outer_inputs = Dict()
    for x ∈ combine_outer
        outer_inputs[x] = pop!(scan_inputs, x)
    end

    # First combine the 'inner product' inputs
    ##########################################

    # check all inputs have same size
    l = length(collect(values(scan_inputs))[1])
    for (key, value) ∈ scan_inputs
        if length(value) != l
            error("Lengths of all input values should be the same. Expected ", l,
                  "for ", key, " but got ", length(value))
        end
    end

    result = Vector{Dict}(undef, l)
    for i ∈ 1:l
        result[i] = Dict{Any,Any}(key=>value[i] for (key, value) in scan_inputs)
    end

    # Combine 'result' with 'combine_outer' fields
    ##############################################

    # Need to make arrays of keys and values in outer_inputs so that we can
    # access them by index. Dicts don't add items from the beginning!
    outer_keys = collect(keys(outer_inputs))
    outer_values = collect(values(outer_inputs))
    function create_level(i, scan_list)
        # This function recursively builds an outer-product of all the
        # option values in the scan
        if i > length(outer_keys)
            # Done building the scan_list, so just return it, ending the
            # recursion
            return scan_list
        end

        l = length(scan_list) * length(outer_values[i])
        new_scan_inputs = Vector{Dict}(undef, l)
        count = 0
        for partial_dict ∈ scan_list
            for j ∈ 1:length(outer_values[i])
                count = count + 1
                new_dict = copy(partial_dict)
                new_dict[outer_keys[i]] = outer_values[i][j]
                new_scan_inputs[count] = new_dict
            end
        end

        return create_level(i + 1, new_scan_inputs)
    end

    result = create_level(1, result)

    for x in result
        x[:run_name] = mk_name(x)
    end

    println("Running scan:")
    for x in result
        println(x)
    end

    return result
end

function mk_name(input)
    name = base_name

    for (k, v) ∈ input
        if isa(k, Tuple)
            kstring = ""
            for x in k
                kstring = kstring * string(x)
            end
        else
            kstring = string(k)
        end
        name = name * "_" * kstring * "-" * string(v)
    end

    return name
end

end
