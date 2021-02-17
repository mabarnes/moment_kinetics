module scan_input

export mk_scan_inputs

# How to combine input lists. E.g. if we have a,b,c then:
#   - inner has runs with (a[1], b[1], c[1]), (a[2], b[2], c[2]), etc.
#     [all lists must have the same length]
#   - outer has runs with all permutations of elements from a, b and c
#     [lists can be any length]
@enum CombineMethod inner outer
const combine_method = inner

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

    if combine_method == inner
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
            result[i][:run_name] = mk_name(result[i])
        end
    elseif combine_method == outer
        # Need to make arrays of keys and values in scan_inputs so that we can
        # access them by index. Dicts don't add items from the beginning!
        scan_keys = collect(keys(scan_inputs))
        scan_values = collect(values(scan_inputs))
        function create_level(i, scan_list)
            # This function recursively builds an outer-product of all the
            # option values in the scan
            if i > length(scan_keys)
                # Done building the scan_list, so just return it, ending the recursion
                return scan_list
            end

            l = length(scan_list) * length(scan_values[i])
            new_scan_inputs = Vector{Dict}(undef, l)
            count = 0
            for partial_dict ∈ scan_list
                for j ∈ 1:length(scan_values[i])
                    count = count + 1
                    new_dict = copy(partial_dict)
                    new_dict[scan_keys[i]] = scan_values[i][j]
                    new_scan_inputs[count] = new_dict
                end
            end

            return create_level(i + 1, new_scan_inputs)
        end

        result = create_level(1, (Dict(),))
        
        for x in result
            x[:run_name] = mk_name(x)
        end
    else
        error("Unknown combine_method ", combine_method)
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
