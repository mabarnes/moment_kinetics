using moment_kinetics.load_data: get_run_info_no_setup, postproc_load_variable

function convert_to_string(array::Vector{UInt8})
    s = String(array)
    s = replace(s, "\0" => "")
    return s
end

"""
    regression_test_debug_comparison(filenameA, filenameB;
                                     tolerance=1.0e-13,
                                     conversions=Dict{String,Any}(),
                                     print_changed=false)

Compare debug files produced when `debug_io=true` is set in the `[timestepping]` input to
find the first place where any variable differs by more than `tolerance`.

This function is expected to be useful when regression testing a run that is expected to
be identical between two different versions, to identify the place in the code where any
difference in output comes from.

If some variables are expected to be rescaled by constant factors between the runs, pass
the conversion in `conversions`. Variables from `filenameA` that match a key in
`conversions` will be multiplied by the corresponding value. The name can also be
converted if necessary. An entry `"new_name" => ("old_name", c)` in `conversions` means
that `"new_name"` in A will be multiplied by `c` and then compared to `"old_name"` in B.

Any variables in `ignore` will not be compared.

Pass `print_changed=true` to print the array values for any arrays that are different.
"""
function regression_test_debug_comparison(filenameA, filenameB;
                                          tolerance=1.0e-13,
                                          conversions=Dict{String,Any}(),
                                          ignore=(),
                                          print_changed=false)

    A = get_run_info_no_setup(filenameA; dfns=true)
    B = get_run_info_no_setup(filenameB; dfns=true)

    nt_min = min(A.nt, B.nt)
    variable_names = A.variable_names
    conversions_keys = collect(keys(conversions))

    for it ∈ 1:nt_min
        changed_variables = Tuple{String,String}[]
        for v ∈ variable_names
            if startswith(v, "chodura_integral")
                # These are just diagnostic variables, and have a non-standard shape, so
                # skip rather than trying to handle
                continue
            end
            if v ∈ ignore
                continue
            end

            newv = postproc_load_variable(A, v; it=it)
            if v ∈ conversions_keys
                c = conversions[v]
                if c isa Tuple
                    newv .*= c[2]
                    oldname = c[1]
                else
                    newv .*= c
                    oldname = v
                end
            else
                oldname = v
            end
            oldv = postproc_load_variable(B, oldname; it=it)

            # First check that newv and oldv are not equal to catch the case where both
            # are Inf (if both are Inf and they are subtracted to compare to tolerance,
            # the result would be NaN even though they are 'equal'), then compare with
            # negation and ≤ so that if there are any NaNs the variable counts as
            # 'changed', as any comparison with NaN always evaluates to `false`.
            for i ∈ eachindex(newv, oldv)
                if newv[i] != oldv[i] && !(abs(newv[i] - oldv[i]) ≤ tolerance)
                    push!(changed_variables, (v, oldname))
                    break
                end
            end
        end

        if !isempty(changed_variables)
            step = postproc_load_variable(A, "step_counter"; it=it)[1]
            stepB = postproc_load_variable(B, "step_counter"; it=it)[1]
            if step != stepB
                println("Error: checking step=$step in A, but step=$stepB in B")
            end

            istage = postproc_load_variable(A, "istage"; it=it)[1]
            istageB = postproc_load_variable(B, "istage"; it=it)[1]
            if istage != istageB
                println("Error: checking istage=$istage in A, but istage=$istageB in B")
            end

            label = convert_to_string(postproc_load_variable(A, "label"; it=it))
            labelB = convert_to_string(postproc_load_variable(B, "label"; it=it))
            if label != labelB
                println("Error: checking label=$label in A, but label=$labelB in B")
            end

            println("Differences found at step=$step, istage=$istage, label=$label")
            println("$changed_variables")

            if print_changed
                for (vA, vB) ∈ changed_variables
                    valA = postproc_load_variable(A, vA; it=it)
                    if vA ∈ conversions_keys
                        c = conversions[vA]
                        if c isa Tuple
                            valA .*= c[2]
                        else
                            valA .*= c
                        end
                    end
                    valB = postproc_load_variable(B, vB; it=it)

                    println()
                    println("$vA A\n", valA)
                    println("$vB B\n", valB)
                    println("diff\n", valA .- valB)
                end
            end

            return nothing
        end
    end

    println("No differences found")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    regression_test_debug_comparison(ARGS[1], ARGS[2])
end
