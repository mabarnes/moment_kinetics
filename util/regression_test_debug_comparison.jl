using moment_kinetics.load_data: get_run_info_no_setup, postproc_load_variable,
                                 close_run_info, get_group
using moment_kinetics.load_data: load_variable as low_level_load_variable

function convert_to_string(array::AbstractVector{UInt8})
    s = String(array)
    s = replace(s, "\0" => "")
    return s
end

"""
    regression_test_debug_comparison(filenameA, filenameB;
                                     tolerance=1.0e-13,
                                     conversions=Dict{String,Any}(),
                                     ignore=(), print_index_ranges=true,
                                     print_changed=false, print_deltas=false)

Compare debug files produced when `debug_io=true` is set in the `[timestepping]` input to
find the first place where any variable differs by more than `tolerance`.

This function is expected to be useful when regression testing a run that is expected to
be identical between two different versions, to identify the place in the code where any
difference in output comes from.

Prints point in the code where differences were first detected (step_counter, istage, and
label), the variables with differences, and for each variable the minimum/maximum index
for each dimension where a difference was found (note not every point within these index
ranges must have a difference, the aim is to give a quick indication, e.g. if the
differences only occur on one boundary).

If some variables are expected to be rescaled by constant factors between the runs, pass
the conversion in `conversions`. Variables from `filenameA` that match a key in
`conversions` will be multiplied by the corresponding value. The name can also be
converted if necessary. An entry `"new_name" => ("old_name", c)` in `conversions` means
that `"new_name"` in A will be multiplied by `c` and then compared to `"old_name"` in B.

Any variables in `ignore` will not be compared.

To disable printing of index ranges (to reduce the amount of output if that's useful),
pass `print_index_ranges=false`.

Pass `print_changed=true` to print the array values for any arrays that are different.
Pass `print_deltas=true` as well to print the difference between the changed value and the
previous value - represents the contribution added by the term that caused the change.
"""
function regression_test_debug_comparison(filenameA, filenameB;
                                          tolerance=1.0e-13,
                                          conversions=Dict{String,Any}(),
                                          ignore=(), print_index_ranges=true,
                                          print_changed=false, print_deltas=false,
                                          return_changed=false)

    A = get_run_info_no_setup(filenameA; dfns=true)
    B = get_run_info_no_setup(filenameB; dfns=true)

    nt_min = min(A.nt, B.nt)
    variable_names = A.variable_names
    conversions_keys = collect(keys(conversions))

    # first check some time-independent variables, if they exist
    ############################################################

    changed_static_variables = Tuple[]
    function check_static_var(name, group_name="overview")
        if !(group_name ∈ A.groups)
            return nothing
        end
        if name ∈ ignore
            return nothing
        end
        if name ∈ conversions_keys
            c = conversions[name]
            if c isa Tuple
                conversion_factor = c[2]
                old_name = c[1]
            else
                conversion_factor = c
                old_name = name
            end
        else
            conversion_factor = 1
            old_name = name
        end
        newv = nothing
        oldv = nothing
        try
            # Can't use postproc_load_variable() here because that function assumes that
            # the variable is time-dependent.
            newv = low_level_load_variable(get_group(A, group_name), name)
            oldv = low_level_load_variable(get_group(B, group_name), old_name)
        catch
            # If variables were not found, nothing to compare
        finally
            println("checking $group_name:$name")

            if name == old_name
                names_to_save = (group_name, name)
            else
                names_to_save = (group_name, name, old_name)
            end
            if newv isa Number
                newv .*= conversion_factor
                if all(newv[i] .!= oldv[i]) && !all(abs.(newv[i] .- oldv[i]) .≤ tolerance)
                    push!(changed_static_variables, names_to_save)
                end
            elseif conversion_factor != 1
                error("Cannot convert non-numeric $name by conversion_factor=$conversion_factor")
            else
                if newv != oldv
                    push!(changed_static_variables, names_to_save)
                end
            end
        end
    end

    # Quantities from manufactured solutions testing
    check_static_var("Source_i_expression", "manufactured_solutions")
    check_static_var("Source_n_expression", "manufactured_solutions")
    check_static_var("Source_i_array", "manufactured_solutions")
    check_static_var("Source_n_array", "manufactured_solutions")

    if !isempty(changed_static_variables)
        println("Differences found in static variables")
        println("$changed_static_variables")
        if print_changed
            for v ∈ changed_static_variables
                if length(v) ≥ 3 && isa(v[3], String)
                    group_name, vA, vB = v[1:3]
                else
                    group_name = v[1]
                    vA = vB = v[2]
                end
                valA = low_level_load_variable(get_group(A, group_name), vA)
                valB = low_level_load_variable(get_group(B, group_name), vB)
                if vA ∈ conversions_keys
                    c = conversions[vA]
                    if c isa Tuple
                        valA .*= c[2]
                    else
                        valA .*= c
                    end
                end

                println()
                println("$vA A\n", valA)
                println("$vB B\n", valB)
                println("diff\n", valA .- valB)
            end
        end
        if return_changed
            changed_vars = []
            for v ∈ changed_static_variables
                if length(v) ≥ 3 && isa(v[3], String)
                    group_name, vA, vB = v[1:3]
                else
                    group_name = v[1]
                    vA = vB = v[2]
                end
                valA = low_level_load_variable(get_group(A, group_name), vA)
                valB = low_level_load_variable(get_group(B, group_name), vB)
                if vA ∈ conversions_keys
                    c = conversions[vA]
                    if c isa Tuple
                        valA .*= c[2]
                    else
                        valA .*= c
                    end
                end
                push!(changed_vars, (valA, valB))
            end
        else
            changed_vars = nothing
        end

        return changed_vars
    end

    # check the time-dependent variables
    ####################################
    for it ∈ 1:nt_min
        changed_variables = Tuple[]
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
            if print_index_ranges
                maxinds = CartesianIndex((-1 for _ ∈ 1:ndims(newv))...)
                # `length(newv)+1` is definitely bigger than any index of any dimension in
                # `newv`.
                mininds = CartesianIndex((length(newv)+1 for _ ∈ 1:ndims(newv))...)
                for i ∈ CartesianIndices(newv)
                    if newv[i] != oldv[i] && !(abs(newv[i] - oldv[i]) ≤ tolerance)
                        mininds = min(mininds, i)
                        maxinds = max(maxinds, i)
                    end
                end

                if v == oldname
                    names_to_save = (v,)
                else
                    names_to_save = (v, oldname)
                end
                if all(maxinds.I .> 0)
                    # Found one or more differences

                    # Guess dimension types from number of dimensions.
                    if ndims(newv) == 1
                        # Not sure what dimensions are for 1D array
                        push!(changed_variables, (names_to_save..., (i=mininds[1]:maxinds[1],)))
                    elseif ndims(newv) == 2
                        push!(changed_variables, (names_to_save..., (z=mininds[1]:maxinds[1],
                                                                     r=mininds[2]:maxinds[2])))
                    elseif ndims(newv) == 3
                        push!(changed_variables, (names_to_save..., (z=mininds[1]:maxinds[1],
                                                                     r=mininds[2]:maxinds[2],
                                                                     s=mininds[3]:maxinds[3])))
                    elseif ndims(newv) == 4
                        push!(changed_variables, (names_to_save..., (vpa=mininds[1]:maxinds[1],
                                                                     vperp=mininds[2]:maxinds[2],
                                                                     z=mininds[3]:maxinds[3],
                                                                     r=mininds[4]:maxinds[4])))
                    elseif ndims(newv) == 5
                        push!(changed_variables, (names_to_save..., (vpa=mininds[1]:maxinds[1],
                                                                     vperp=mininds[2]:maxinds[2],
                                                                     z=mininds[3]:maxinds[3],
                                                                     r=mininds[4]:maxinds[4],
                                                                     s=mininds[5]:maxinds[5])))
                    elseif ndims(newv) == 6
                        push!(changed_variables, (names_to_save..., (vz=mininds[1]:maxinds[1],
                                                                     vr=mininds[2]:maxinds[2],
                                                                     vzeta=mininds[3]:maxinds[3],
                                                                     z=mininds[4]:maxinds[4],
                                                                     r=mininds[5]:maxinds[5],
                                                                     s=mininds[6]:maxinds[6])))
                    end
                end
            else
                for i ∈ eachindex(newv, oldv)
                    if newv[i] != oldv[i] && !(abs(newv[i] - oldv[i]) ≤ tolerance)
                        push!(changed_variables, names_to_save)
                        break
                    end
                end
            end
        end

        if !isempty(changed_variables)
            step = postproc_load_variable(A, "step_counter"; it=it)[1]
            stepB = postproc_load_variable(B, "step_counter"; it=it)[1]
            if step != stepB
                println("Error: checking step=$step in A, but step=$stepB in B")
            end

            is_debug_files = ("istage" ∈ A.variable_names
                              && "label" ∈ A.variable_names
                              && "istage" ∈ B.variable_names
                              && "label" ∈ B.variable_names)
            if is_debug_files
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
            else
                println("Differences found at step=$step")
            end
            println("$changed_variables")

            if print_changed
                for v ∈ changed_variables
                    if length(v) ≥ 2 && isa(v[2], String)
                        vA, vB = v[1:2]
                    else
                        vA = vB = v[1]
                    end
                    valA = postproc_load_variable(A, vA; it=it)
                    if it > 1 && print_deltas
                        deltaA = valA .- postproc_load_variable(A, vA; it=it-1)
                    end
                    if vA ∈ conversions_keys
                        c = conversions[vA]
                        if c isa Tuple
                            valA .*= c[2]
                            if it > 1 && print_deltas
                                deltaA .*= c[2]
                            end
                        else
                            valA .*= c
                            if it > 1 && print_deltas
                                deltaA .*= c
                            end
                        end
                    end
                    valB = postproc_load_variable(B, vB; it=it)
                    if it > 1 && print_deltas
                        deltaB = valB .- postproc_load_variable(B, vB; it=it-1)
                    end

                    println()
                    println("$vA A\n", valA)
                    println("$vB B\n", valB)
                    println("diff\n", valA .- valB)
                    if it > 1 && print_deltas
                        println("delta $vA A\n", deltaA)
                        println("delta $vB B\n", deltaB)
                        println("diff delta\n", deltaA .- deltaB)
                    end
                end
            end
            if return_changed
                changed_vars = []
                for v ∈ changed_variables
                    if length(v) ≥ 2 && isa(v[2], String)
                        vA, vB = v[1:2]
                    else
                        vA = vB = v[1]
                    end
                    valA = postproc_load_variable(A, vA; it=it)
                    valB = postproc_load_variable(B, vB; it=it)
                    if vA ∈ conversions_keys
                        c = conversions[vA]
                        if c isa Tuple
                            valA .*= c[2]
                            if it > 1 && print_deltas
                                deltaA .*= c[2]
                            end
                        else
                            valA .*= c
                            if it > 1 && print_deltas
                                deltaA .*= c
                            end
                        end
                    end
                    push!(changed_vars, (valA, valB))
                end
            else
                changed_vars = nothing
            end

            close_run_info(A)
            close_run_info(B)
            return changed_vars
        end
    end

    close_run_info(A)
    close_run_info(B)
    println("No differences found")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    regression_test_debug_comparison(ARGS[1], ARGS[2])
end
