include("shared_utils.jl")

using moment_kinetics.analysis: analyze_fields_data
using .shared_utils: calculate_and_write_frequencies

"""
    sound_wave_plots(run_info::Vector{Any}; plot_prefix)
    sound_wave_plots(run_info; outfile=nothing, ax=nothing, phi=nothing)

Calculate decay rate and frequency for the damped 'sound wave' in a 1D1V simulation in a
periodic box. Plot the mode amplitude vs. time along with the fitted decay rate.

The information for the runs to analyse and plot is passed in `run_info` (as returned by
[`get_run_info`](@ref)). If `run_info` is a Vector, comparison plots are made where line
plots from the different runs are overlayed on the same axis.

Settings are read from the `[sound_wave]` section of the input.

When `run_info` is a Vector, `plot_prefix` is required and gives the path and prefix for
plots to be saved to. They will be saved with the format
`plot_prefix<some_identifying_string>.pdf`.
When `run_info` is not a Vector, `outfile` can be passed, to save the plot to `outfile`.

When `run_info` is not a Vector, ax can be passed to add the plot to an existing `Axis`.

When `run_info` is not a Vector, the array containing data for phi can be passed to `phi` -
by default this data is loaded from the output file.
"""
function sound_wave_plots end

function sound_wave_plots(run_info::Vector{Any}; plot_prefix)
    input = Dict_to_NamedTuple(input_dict["sound_wave_fit"])

    if !input.calculate_frequency && !input.plot
        return nothing
    end

    println("Doing analysis and making plots for sound wave test")
    flush(stdout)

    try
        outfile = plot_prefix * "delta_phi0_vs_t.pdf"

        if length(run_info) == 1
            return sound_wave_plots(run_info[1]; outfile=outfile)
        end

        if input.plot
            fig, ax = get_1d_ax(xlabel="time", ylabel="δϕ", yscale=log10)
        else
            ax = nothing
        end

        for ri ∈ run_info
            sound_wave_plots(ri; ax=ax)
        end

        if input.plot
            put_legend_below(fig, ax)
            # Ensure the first row width is 3/4 of the column width so that
            # the plot does not get squashed by the legend
            rowsize!(fig.layout, 1, Aspect(1, 3/4))
            resize_to_layout!(fig)

            save(outfile, fig)

            return fig
        end
    catch e
        return makie_post_processing_error_handler(
                   e,
                   "Error in sound_wave_plots().")
    end

    return nothing
end

function sound_wave_plots(run_info; outfile=nothing, ax=nothing, phi=nothing)
    input = Dict_to_NamedTuple(input_dict["sound_wave_fit"])

    if !input.calculate_frequency && !input.plot
        return nothing
    end

    if ax === nothing && input.plot
        fig, ax = get_1d_ax(xlabel="time", ylabel="δϕ", yscale=log10)
    else
        fig = nothing
    end

    time = run_info.time

    # This analysis is only designed for 1D cases, so only use phi[:,ir0,:]
    if phi === nothing
        phi = get_variable(run_info, "phi"; ir=input.ir0)
    else
        select_slice(phi, :t, :z; input=input)
    end

    phi_fldline_avg, delta_phi = analyze_fields_data(phi, run_info.nt, run_info.z)

    if input.calculate_frequency
        frequency, growth_rate, shifted_time, fitted_delta_phi =
            calculate_and_write_frequencies(run_info.run_prefix, run_info.nt, time,
                                            run_info.z.grid, 1, run_info.nt, input.iz0,
                                            delta_phi, (calculate_frequencies=true,))
    end

    if input.plot
        if outfile === nothing
            # May be plotting multipe runs
            delta_phi_label = run_info.run_name * " δϕ"
            fit_label = run_info.run_name * " fit"
        else
            # Only plotting this run
            delta_phi_label = "δϕ"
            fit_label = "fit"
        end

        @views lines!(ax, time, positive_or_nan.(abs.(delta_phi[input.iz0,:]), epsilon=1.e-20), label=delta_phi_label)

        if input.calculate_frequency
            @views lines!(ax, time, positive_or_nan.(abs.(fitted_delta_phi), epsilon=1.e-20), label=fit_label)
        end

        if outfile !== nothing
            if fig === nothing
                error("Cannot save figure from this function when `ax` was passed. Please "
                      * "save the figure that contains `ax`")
            end
            put_legend_below(fig, ax)
            # Ensure the first row width is 3/4 of the column width so that
            # the plot does not get squashed by the legend
            rowsize!(fig.layout, 1, Aspect(1, 3/4))
            resize_to_layout!(fig)
            save(outfile, fig)
        end
    end

    return fig
end
