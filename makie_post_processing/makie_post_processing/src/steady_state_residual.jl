using moment_kinetics.analysis: steady_state_residuals

"""
     _get_steady_state_residual_fig_axes(n_runs)

Utility method to avoid code duplication when creating the fig_axes OrderedDict for
calculate_steady_state_residual.

`n_runs` sets the number of axes to create in each entry.
"""
function _get_steady_state_residual_fig_axes(n_runs)
    return OrderedDict(
                "RMS absolute residual"=>get_1d_ax(n_runs, xlabel="time",
                                                   ylabel="RMS absolute residual",
                                                   yscale=log10, get_legend_place=:right),
                "max absolute residual"=>get_1d_ax(n_runs, xlabel="time",
                                                   ylabel="max absolute residual",
                                                   yscale=log10, get_legend_place=:right),
                "RMS relative residual"=>get_1d_ax(n_runs, xlabel="time",
                                                   ylabel="RMS relative residual",
                                                   yscale=log10, get_legend_place=:right),
                "max relative residual"=>get_1d_ax(n_runs, xlabel="time",
                                                   ylabel="max relative residual",
                                                   yscale=log10, get_legend_place=:right))
end

# Utility method to avoid code duplication when saving the calculate_steady_state_residual
# plots
function _save_residual_plots(fig_axes, plot_prefix)
    try
        for (key, fa) ∈ fig_axes
            for (ax, lp) ∈ zip(fa[2], fa[3])
                Legend(lp, ax)
            end
            save(plot_prefix * replace(key, " "=>"_") * ".pdf", fa[1])
        end
    catch e
        return makie_post_processing_error_handler(
                   e,
                   "Error in _save_residual_plots().")
    end
end

"""
calculate_steady_state_residual(run_info, variable_name; is=1, data=nothing,
                                plot_prefix=nothing, fig_axes=nothing, i_run=1)

Calculate and plot the 'residuals' for `variable_name`.

The information for the runs to plot is passed in `run_info` (as returned by
[`get_run_info`](@ref)). If `run_info` is a Vector, comparison plots are made where plots
from the different runs are displayed in a horizontal row.

If the variable has a species dimension, `is` selects which species to analyse.

By default the variable will be loaded from file. If the data has already been loaded, it
can be passed to `data` instead. `data` should be a Vector of the same length as `run_info`
if `run_info` is a Vector.

If `plot_prefix` is passed, it gives the path and prefix for plots to be saved to. They
will be saved with the format `plot_prefix<some_identifying_string>.pdf`.

`fig_axes` can be passed an OrderedDict of Tuples as returned by
[`_get_steady_state_residual_fig_axes`](@ref) - each tuple contains the Figure `fig` and
Axis or Vector{Axis} `ax` to which to add the plot corresponding to its key. If `run_info`
is a Vector, `ax` for each entry must be a Vector of the same length.
"""
function calculate_steady_state_residual end

function calculate_steady_state_residual(run_info::Vector{Any}, variable_name; is=1,
                                         data=nothing, plot_prefix=nothing,
                                         fig_axes=nothing)
    try
        n_runs = length(run_info)
        if data === nothing
            data = [nothing for _ ∈ 1:n_runs]
        end
        if fig_axes === nothing
            fig_axes = _get_steady_state_residual_fig_axes(length(run_info))
        end

        for (i, (ri, d)) ∈ enumerate(zip(run_info, data))
            calculate_steady_state_residual(ri, variable_name; is=is, data=d,
                                            fig_axes=fig_axes, i_run=i)
        end

        if plot_prefix !== nothing
            _save_residual_plots(fig_axes, plot_prefix)
        end

        return fig_axes
    catch e
        return makie_post_processing_error_handler(
                   e,
                   "Error in calculate_steady_state_residual().")
    end
end

function calculate_steady_state_residual(run_info, variable_name; is=1, data=nothing,
                                         plot_prefix=nothing, fig_axes=nothing,
                                         i_run=1)

    if data === nothing
        data = get_variable(run_info, variable_name; is=is)
    end

    t_dim = ndims(data)
    nt = size(data, t_dim)
    variable = selectdim(data, t_dim, 2:nt)
    variable_at_previous_time = selectdim(data, t_dim, 1:nt-1)
    dt = @views @. run_info.time[2:nt] - run_info.time[1:nt-1]
    residual_norms = steady_state_residuals(variable, variable_at_previous_time, dt)

    textoutput_file = run_info.run_prefix * "_residuals.txt"
    open(textoutput_file, "a") do io
        for (key, residual) ∈ residual_norms
            # Use lpad to get fixed-width strings to print, so we get nice columns of
            # output. 24 characters should be enough to represent any float with at
            # least a couple of spaces in front to separate columns (e.g.  "
            # -3.141592653589793e100"
            line = string((lpad(string(x), 24) for x ∈ residual)...)

            # Print to stdout as well for convenience
            println(key, ": ", line)

            line *= "  # " * variable_name
            if is !== nothing
                line *= string(is)
            end
            line *= " " * key
            println(io, line)
        end
    end

    if fig_axes === nothing
        fig_axes = _get_steady_state_residual_fig_axes(1)
    end

    t = @view run_info.time[2:end]
    with_theme(Theme(Lines=(cycle=[:color, :linestyle],))) do
        for (key, norm) ∈ residual_norms
            @views plot_1d(t, norm; label="$variable_name", ax=fig_axes[key][2][i_run])
        end
    end

    if plot_prefix !== nothing
        _save_residual_plots(fig_axes, plot_prefix)
    end

    return fig_axes
end
