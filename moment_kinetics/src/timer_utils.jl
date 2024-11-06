"""
Utilities for timing functions or blocks of code
"""
module timer_utils

export global_timer, @timeit, @timeit_debug, format_global_timer, reset_mk_timers!,
       timeit_debug_enabled

using ..type_definitions: mk_int

using DataStructures: SortedDict
using TimerOutputs

"""
    timeit_debug_enabled()

`@timeit_debug` uses a function defined in the enclosing module called
`timeit_debug_enabled()` to decide whether to include debug timers (included if this
function returns `true`, or not - with zero overhead - if it returns `false`).

To control the debug timers in `moment_kinetics` we define this function once, in
[`timer_utils`](@ref), and import it from there into any other modules that use
@timeit_debug.

To activate debug timers, edit this function so that it returns `true`.
"""
timeit_debug_enabled() = false

"""
Global object used to collect timings of various parts of the code
"""
const global_timer = TimerOutput()

"""
"""
const TimerNamesDict = SortedDict{String,SortedDict,Base.Order.ForwardOrdering}
TimerNamesDict() = TimerNamesDict(Base.Order.ForwardOrdering())

"""
Nested SortedDict containting the names of all timers that have been created on each MPI
rank and added to the moments output file.
"""
const timer_names_per_rank_moments = SortedDict{mk_int,Tuple{TimerNamesDict,Ref{mk_int}}}()

"""
Nested SortedDict containting the names of all timers that have been created on each MPI
rank and added to the dfns output file.
"""
const timer_names_per_rank_dfns = SortedDict{mk_int,Tuple{TimerNamesDict,Ref{mk_int}}}()

"""
    format_global_timer(; show=true, truncate_output=true)

Manipulate a copy of the [`global_timer`](@ref), to remove some things to reduce the
clutter when it is printed.

By default the resulting `TimerOutput` is displayed in the terminal. Pass
`show_output=true` to display the resulting `TimerOutput` in the terminal.

By default, the output is truncated, removing deeply nested timers and timers with very
little time. To include all timers, pass `truncate_output=false`. The threshold for
dropping timers is if their time is less than `threshold` times the total time.

By default, returns a string showing the contents of the `TimerOutput`. When
`show_output=true` is passed, just returns the empty string.
"""
function format_global_timer(; show_output=false, threshold=1.0e-3, truncate_output=true,
                               top_level=nothing)
    # Remove lower-level timers to prevent the terminal output becoming too
    # cluttered.
    if top_level === nothing
        timers_to_print = deepcopy(global_timer)
    else
        this_level = global_timer
        for key in top_level
            this_level = this_level[key]
        end
        timers_to_print = deepcopy(this_level)
    end
    if truncate_output
        try
            empty!(timers_to_print["moment_kinetics"]["setup_moment_kinetics"].inner_timers)
        catch
        end
        try
            empty!(timers_to_print["moment_kinetics"]["time_advance! step"]["write_all_dfns_data_to_binary"].inner_timers)
        catch
        end
        try
            empty!(timers_to_print["moment_kinetics"]["time_advance! step"]["ssp_rk!"]["apply_all_bcs_constraints_update_moments!"].inner_timers)
        catch
        end
        try
            # Note accumulated_data.time is an integer representing time in nanoseconds.
            if timers_to_print["moment_kinetics"]["time_advance! step"]["write_data_to_ascii"].accumulated_data.time < 1.0e6
                # Probably not using ASCII output, so skip printing it.
                pop!(timers_to_print["moment_kinetics"]["time_advance! step"].inner_timers, "write_data_to_ascii")
            end
        catch
        end

        # Remove timers that contribute a very small fraction of the total time, to
        # prevent the terminal output becoming too cluttered.
        if top_level === nothing
            # Need to get a timer that has actually finished timing, as the very top level
            # might still have `accumulated_data.time = 0` when this function is called.
            if length(timers_to_print.inner_timers) == 0
                # There is no inner timer, so fall back to just using `timers_to_print`
                total_time = timers_to_print.accumulated_data.time
            else
                # Usually top_level_name will be "moment_kinetics", but do this in a
                # generic way to make sure it can get something even in non-standard
                # cases.
                top_level_name = first(keys(timers_to_print.inner_timers))
                total_time = timers_to_print[top_level_name].accumulated_data.time
            end
        else
            total_time = timers_to_print.accumulated_data.time
        end
        function remove_short_timers!(to)
            for (key,inner_to) ∈ pairs(to.inner_timers)
                if inner_to.accumulated_data.time < threshold * total_time
                    pop!(to.inner_timers, key)
                else
                    remove_short_timers!(inner_to)
                end
            end
        end
        this_level = timers_to_print
        if top_level !== nothing
            for (i, key) ∈ enumerate(("moment_kinetics", "time_advance! step", "ssp_rk!", "euler_time_advance!"))
                if i ≤ length(top_level)
                    if key == top_level[i]
                        # Already selected they "key" section when we restricted to
                        # `top_level`.
                        continue
                    else
                        # Trying to select a section that is not included in
                        # timers_to_print since we restricted to `top_level`, so nothing
                        # to do.
                        break
                    end
                else
                    this_level = this_level[key]
                end
            end
        end
        remove_short_timers!(this_level)
    end

    if show_output
        show(timers_to_print; sortby=:firstexec)
        println()
        return ""
    else
        string_buffer = IOBuffer()
        # Set the "COLUMNS" environment variable so that the printing of timers_to_print
        # does not truncate to fit in a narrow width (the default would be 80 characters).
        withenv("COLUMNS" => 300) do
            print_timer(string_buffer, timers_to_print; linechars=:ascii)
        end
        result = String(take!(string_buffer))

        # Remove μ from result, so that it converts nicely to ASCII, which will be needed
        # when we save the string to an HDF5 or NetCDF file. μ often appears because times
        # may be printed in microseconds.
        result = replace(result, "μ" => "u")
        result = ascii(replace(result, !isascii=>' '))
    end
end

"""
    reset_mk_timers!()

Reset all global state of timers.
"""
function reset_mk_timers!()
    reset_timer!(global_timer)
    empty!(timer_names_per_rank_moments)
    empty!(timer_names_per_rank_dfns)
end

end #timer_utils
