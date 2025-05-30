using moment_kinetics.analysis: check_Chodura_condition

"""
    Chodura_condition_plots(run_info::Vector{Any}; plot_prefix)
    Chodura_condition_plots(run_info; plot_prefix=nothing, axes=nothing)

Plot the criterion from the Chodura condition at the sheath boundaries.

The information for the runs to plot is passed in `run_info` (as returned by
[`get_run_info`](@ref)). If `run_info` is a Vector, comparison plots are made where line
plots from the different runs are overlayed on the same axis, and heatmap plots are
displayed in a horizontal row.

Settings are read from the `[Chodura_condition]` section of the input.

When `run_info` is a Vector, `plot_prefix` is required and gives the path and prefix for
plots to be saved to. They will be saved with the format
`plot_prefix<some_identifying_string>.pdf`. When `run_info` is not a Vector, `plot_prefix`
is optional - plots will be saved only if it is passed.

When `run_info` is not a Vector, a Vector of Axis objects can be passed to `axes`, and each
plot will be added to one of `axes`.
"""
function Chodura_condition_plots end

function Chodura_condition_plots(run_info::Vector{Any}; plot_prefix)
    input = Dict_to_NamedTuple(input_dict_dfns["Chodura_condition"])

    if !any(v for (k,v) ∈ pairs(input) if startswith(String(k), "plot"))
        # No plots to make here
        return nothing
    end
    if !any(ri !== nothing for ri ∈ run_info)
        println("Warning: no distribution function output, skipping Chodura "
                * "condition plots")
        return nothing
    end

    try
        println("Making Chodura condition plots")
        flush(stdout)

        n_runs = length(run_info)

        if n_runs == 1
            Chodura_condition_plots(run_info[1], plot_prefix=plot_prefix)
            return nothing
        end

        figs = []
        axes = [[] for _ ∈ run_info]
        if input.plot_vs_t
            fig, ax = get_1d_ax(title="Chodura ratio at z=-L/2", xlabel="time",
                                ylabel="ratio")
            push!(figs, fig)
            for a ∈ axes
                push!(a, ax)
            end

            fig, ax = get_1d_ax(title="Chodura ratio at z=+L/2", xlabel="time",
                                ylabel="ratio")
            push!(figs, fig)
            for a ∈ axes
                push!(a, ax)
            end
        else
            push!(figs, nothing)
            for a ∈ axes
                push!(a, nothing)
            end
            push!(figs, nothing)
            for a ∈ axes
                push!(a, nothing)
            end
        end
        if input.plot_vs_r
            fig, ax = get_1d_ax(title="Chodura ratio at z=-L/2", xlabel="r",
                                ylabel="ratio")
            push!(figs, fig)
            for a ∈ axes
                push!(a, ax)
            end

            fig, ax = get_1d_ax(title="Chodura ratio at z=+L/2", xlabel="r",
                                ylabel="ratio")
            push!(figs, fig)
            for a ∈ axes
                push!(a, ax)
            end
        else
            push!(figs, nothing)
            for a ∈ axes
                push!(a, nothing)
            end
            push!(figs, nothing)
            for a ∈ axes
                push!(a, nothing)
            end
        end
        if input.plot_vs_r_t
            fig, ax, colorbar_place = get_2d_ax(n_runs; title="Chodura ratio at z=-L/2",
                                                xlabel="r", ylabel="time")
            push!(figs, fig)
            for (a, b, cbp) ∈ zip(axes, ax, colorbar_place)
                push!(a, (b, cbp))
            end

            fig, ax, colorbar_place = get_2d_ax(n_runs; title="Chodura ratio at z=+L/2",
                                                xlabel="r", ylabel="time")
            push!(figs, fig)
            for (a, b, cbp) ∈ zip(axes, ax, colorbar_place)
                push!(a, (b, cbp))
            end
        else
            push!(figs, nothing)
            for a ∈ axes
                push!(a, nothing)
            end
            push!(figs, nothing)
            for a ∈ axes
                push!(a, nothing)
            end
        end
        if input.plot_f_over_vpa2
            println("going to plot f_over_vpa2")
            fig, ax = get_1d_ax(title="f/vpa^2 lower wall", xlabel="vpa", ylabel="f / vpa^2")
            push!(figs, fig)
            for a ∈ axes
                push!(a, ax)
            end

            fig, ax = get_1d_ax(title="f/vpa^2 upper wall", xlabel="vpa", ylabel="f / vpa^2")
            push!(figs, fig)
            for a ∈ axes
                push!(a, ax)
            end
        else
            push!(figs, nothing)
            for a ∈ axes
                push!(a, nothing)
            end
            push!(figs, nothing)
            for a ∈ axes
                push!(a, nothing)
            end
        end
        if input.animate_f_over_vpa2
            fig, ax = get_1d_ax(title="f/vpa^2 lower wall", xlabel="vpa", ylabel="f / vpa^2")
            frame_index = Observable(1)
            push!(figs, fig)
            for a ∈ axes
                push!(a, (ax, frame_index))
            end

            fig, ax = get_1d_ax(title="f/vpa^2 upper wall", xlabel="vpa", ylabel="f / vpa^2")
            frame_index = Observable(1)
            push!(figs, fig)
            for a ∈ axes
                push!(a, (ax, frame_index))
            end
        else
            push!(figs, nothing)
            for a ∈ axes
                push!(a, nothing)
            end
            push!(figs, nothing)
            for a ∈ axes
                push!(a, nothing)
            end
        end

        for (ri, ax) ∈ zip(run_info, axes)
            Chodura_condition_plots(ri; axes=ax)
        end

        if input.plot_vs_t
            fig = figs[1]
            ax = axes[1][1]
            put_legend_below(fig, ax)
            # Ensure the first row width is 3/4 of the column width so that
            # the plot does not get squashed by the legend
            rowsize!(fig.layout, 1, Aspect(1, 3/4))
            resize_to_layout!(fig)
            outfile = string(plot_prefix, "Chodura_ratio_lower_vs_t.pdf")
            save(outfile, fig)

            fig = figs[2]
            ax = axes[1][2]
            put_legend_below(fig, ax)
            # Ensure the first row width is 3/4 of the column width so that
            # the plot does not get squashed by the legend
            rowsize!(fig.layout, 1, Aspect(1, 3/4))
            resize_to_layout!(fig)
            outfile = string(plot_prefix, "Chodura_ratio_upper_vs_t.pdf")
            save(outfile, fig)
        end
        if input.plot_vs_r
            fig = figs[3]
            ax = axes[1][3]
            put_legend_below(fig, ax)
            # Ensure the first row width is 3/4 of the column width so that
            # the plot does not get squashed by the legend
            rowsize!(fig.layout, 1, Aspect(1, 3/4))
            resize_to_layout!(fig)
            outfile = string(plot_prefix, "Chodura_ratio_lower_vs_r.pdf")
            save(outfile, fig)

            fig = figs[4]
            ax = axes[1][4]
            put_legend_below(fig, ax)
            # Ensure the first row width is 3/4 of the column width so that
            # the plot does not get squashed by the legend
            rowsize!(fig.layout, 1, Aspect(1, 3/4))
            resize_to_layout!(fig)
            outfile = string(plot_prefix, "Chodura_ratio_upper_vs_r.pdf")
            save(outfile, fig)
        end
        if input.plot_vs_r_t
            fig = figs[5]
            outfile = string(plot_prefix, "Chodura_ratio_lower_vs_r_t.pdf")
            save(outfile, fig)

            fig = figs[6]
            outfile = string(plot_prefix, "Chodura_ratio_upper_vs_r_t.pdf")
            save(outfile, fig)
        end
        if input.plot_f_over_vpa2
            fig = figs[7]
            println("check axes ", axes)
            ax = axes[1][7]
            put_legend_below(fig, ax)
            # Ensure the first row width is 3/4 of the column width so that
            # the plot does not get squashed by the legend
            rowsize!(fig.layout, 1, Aspect(1, 3/4))
            resize_to_layout!(fig)
            outfile = string(plot_prefix, "pdf_unnorm_over_vpa2_wall-_vs_vpa.pdf")
            save(outfile, fig)

            fig = figs[8]
            ax = axes[1][8]
            put_legend_below(fig, ax)
            # Ensure the first row width is 3/4 of the column width so that
            # the plot does not get squashed by the legend
            rowsize!(fig.layout, 1, Aspect(1, 3/4))
            resize_to_layout!(fig)
            outfile = string(plot_prefix, "pdf_unnorm_over_vpa2_wall+_vs_vpa.pdf")
            save(outfile, fig)
        end
        if input.animate_f_over_vpa2
            nt = minimum(ri.nt for ri ∈ run_info)

            fig = figs[9]
            ax = axes[1][9][1]
            frame_index = axes[1][9][2]
            put_legend_below(fig, ax)
            # Ensure the first row width is 3/4 of the column width so that
            # the plot does not get squashed by the legend
            rowsize!(fig.layout, 1, Aspect(1, 3/4))
            resize_to_layout!(fig)
            outfile = string(plot_prefix, "pdf_unnorm_over_vpa2_wall-_vs_vpa." * input.animation_ext)
            save_animation(fig, frame_index, nt, outfile)

            fig = figs[10]
            ax = axes[1][10][1]
            frame_index = axes[1][10][2]
            put_legend_below(fig, ax)
            # Ensure the first row width is 3/4 of the column width so that
            # the plot does not get squashed by the legend
            rowsize!(fig.layout, 1, Aspect(1, 3/4))
            resize_to_layout!(fig)
            outfile = string(plot_prefix, "pdf_unnorm_over_vpa2_wall+_vs_vpa." * input.animation_ext)
            save_animation(fig, frame_index, nt, outfile)
        end
    catch e
        return makie_post_processing_error_handler(
                   e,
                   "Error in Chodura_condition_plots().")
    end

    return nothing
end

function Chodura_condition_plots(run_info; plot_prefix=nothing, axes=nothing)

    if run_info === nothing
        println("In Chodura_condition_plots(), run_info===nothing so skipping")
        return nothing
    end
    if run_info.z.bc != "wall"
        println("In Chodura_condition_plots(), z.bc!=\"wall\" - there is no wall - so "
                * "skipping")
        return nothing
    end

    input = Dict_to_NamedTuple(input_dict_dfns["Chodura_condition"])

    time = run_info.time
    density = get_variable(run_info, "density")
    upar = get_variable(run_info, "parallel_flow")
    vth = get_variable(run_info, "thermal_speed")
    temp_e = get_variable(run_info, "electron_temperature")
    Er = get_variable(run_info, "Er")
    f_lower = get_variable(run_info, "f", iz=1)
    f_upper = get_variable(run_info, "f", iz=run_info.z.n_global)

    Chodura_ratio_lower, Chodura_ratio_upper, cutoff_lower, cutoff_upper =
        check_Chodura_condition(run_info.r_local, run_info.z_local, run_info.vperp,
                                run_info.vpa, density, upar, vth, temp_e,
                                run_info.composition, Er, run_info.geometry,
                                run_info.z.bc, nothing;
                                evolve_density=run_info.evolve_density,
                                evolve_upar=run_info.evolve_upar,
                                evolve_p=run_info.evolve_p,
                                f_lower=f_lower, f_upper=f_upper, find_extra_offset=true)

    if input.plot_vs_t
        if axes === nothing
            fig, ax = get_1d_ax(title="Chodura ratio at z=-L/2", xlabel="time",
                                ylabel="ratio")
        else
            fig = nothing
            ax = axes[1]
        end
        plot_1d(time, Chodura_ratio_lower[input.ir0,:], ax=ax, label=run_info.run_name)
        if plot_prefix !== nothing
            outfile = string(plot_prefix, "Chodura_ratio_lower_vs_t.pdf")
            save(outfile, fig)
        end

        if axes === nothing
            fig, ax = get_1d_ax(title="Chodura ratio at z=+L/2", xlabel="time",
                                ylabel="ratio")
        else
            fig = nothing
            ax = axes[2]
        end
        plot_1d(time, Chodura_ratio_upper[input.ir0,:], ax=ax, label=run_info.run_name)
        if plot_prefix !== nothing
            outfile = string(plot_prefix, "Chodura_ratio_upper_vs_t.pdf")
            save(outfile, fig)
        end
    end

    if input.plot_vs_r
        if axes === nothing
            fig, ax = get_1d_ax(title="Chodura ratio at z=-L/2", xlabel="r",
                                ylabel="ratio")
        else
            fig = nothing
            ax = axes[3]
        end
        plot_1d(run_info.r.grid, Chodura_ratio_lower[:,input.it0], ax=ax, label=run_info.run_name)
        if plot_prefix !== nothing
            outfile = string(plot_prefix, "Chodura_ratio_lower_vs_r.pdf")
            save(outfile, fig)
        end

        if axes === nothing
            fig, ax = get_1d_ax(title="Chodura ratio at z=+L/2", xlabel="r",
                                ylabel="ratio")
        else
            fig = nothing
            ax = axes[4]
        end
        plot_1d(run_info.r.grid, Chodura_ratio_upper[:,input.it0], ax=ax, label=run_info.run_name)
        if plot_prefix !== nothing
            outfile = string(plot_prefix, "Chodura_ratio_upper_vs_r.pdf")
            save(outfile, fig)
        end
    end

    if input.plot_vs_r_t
        if axes === nothing
            fig, ax, colorbar_place = get_2d_ax(title="Chodura ratio at z=-L/2",
                                                xlabel="r", ylabel="time")
            title = nothing
        else
            fig = nothing
            ax, colorbar_place = axes[5]
            title = run_info.run_name
        end
        plot_2d(run_info.r.grid, time, Chodura_ratio_lower, ax=ax,
                colorbar_place=colorbar_place, title=title)
        if plot_prefix !== nothing
            outfile = string(plot_prefix, "Chodura_ratio_lower_vs_r_t.pdf")
            save(outfile, fig)
        end

        if axes === nothing
            fig, ax, colorbar_place = get_2d_ax(title="Chodura ratio at z=+L/2",
                                                xlabel="r", ylabel="time")
            title = nothing
        else
            fig = nothing
            ax, colorbar_place = axes[6]
            title = run_info.run_name
        end
        plot_2d(run_info.r.grid, time, Chodura_ratio_upper, ax=ax,
                colorbar_place=colorbar_place, title=title)
        if plot_prefix !== nothing
            outfile = string(plot_prefix, "Chodura_ratio_upper_vs_r_t.pdf")
            save(outfile, fig)
        end
    end

    if input.plot_f_over_vpa2
        if axes === nothing
            fig, ax, = get_1d_ax(title="f/vpa^2 lower wall",
                                 xlabel="vpa", ylabel="f / vpa^2")
            title = nothing
            label = ""
        else
            fig = nothing
            ax = axes[7]
            label = run_info.run_name
        end
        f_input = copy(input_dict_dfns["f"])
        f_input["it0"] = input.it0
        f_input["ir0"] = input.ir0
        f_input["iz0"] = 1
        plot_f_unnorm_vs_vpa(run_info; f_over_vpa2=true, input=f_input, is=1, fig=fig,
                             ax=ax, label=label)
        vlines!(ax, cutoff_lower[input.ir0,input.it0]; linestyle=:dash, color=:red)
        if plot_prefix !== nothing && fig !== nothing
            outfile=plot_prefix * "pdf_unnorm_over_vpa2_wall-_vs_vpa.pdf"
            save(outfile, fig)
        end

        if axes === nothing
            fig, ax, = get_1d_ax(title="f/vpa^2 upper wall",
                                 xlabel="vpa", ylabel="f / vpa^2")
            title = nothing
            label = ""
        else
            fig = nothing
            ax = axes[8]
            label = run_info.run_name
        end
        f_input = copy(input_dict_dfns["f"])
        f_input["it0"] = input.it0
        f_input["ir0"] = input.ir0
        f_input["iz0"] = run_info.z.n
        plot_f_unnorm_vs_vpa(run_info; f_over_vpa2=true, input=f_input, is=1, fig=fig,
                             ax=ax, label=label)
        vlines!(ax, cutoff_upper[input.ir0,input.it0]; linestyle=:dash, color=:red)
        if plot_prefix !== nothing && fig !== nothing
            outfile=plot_prefix * "pdf_unnorm_over_vpa2_wall+_vs_vpa.pdf"
            save(outfile, fig)
        end
    end

    if input.animate_f_over_vpa2
        if axes === nothing
            fig, ax, = get_1d_ax(title="f/vpa^2 lower wall",
                                 xlabel="vpa", ylabel="f / vpa^2")
            frame_index = Observable(1)
            title = nothing
            label = ""
        else
            fig = nothing
            ax, frame_index = axes[9]
            label = run_info.run_name
        end
        f_input = copy(input_dict_dfns["f"])
        f_input["ir0"] = input.ir0
        f_input["iz0"] = 1
        animate_f_unnorm_vs_vpa(run_info; f_over_vpa2=true, input=f_input, is=1, iz=1,
                                fig=fig, ax=ax, frame_index=frame_index, label=label)
        vlines!(ax, @lift cutoff_lower[input.ir0,$frame_index]; linestyle=:dash, color=:red)
        if plot_prefix !== nothing && fig !== nothing
            outfile=plot_prefix * "pdf_unnorm_over_vpa2_wall-_vs_vpa." * input.animation_ext
            save_animation(fig, frame_index, run_info.nt, outfile)
        end

        if axes === nothing
            fig, ax, = get_1d_ax(title="f/vpa^2 upper wall",
                                 xlabel="vpa", ylabel="f / vpa^2")
            frame_index = Observable(1)
            title = nothing
            label = ""
        else
            fig = nothing
            ax, frame_index = axes[10]
            label = run_info.run_name
        end
        f_input = copy(input_dict_dfns["f"])
        f_input["ir0"] = input.ir0
        f_input["iz0"] = run_info.z.n
        animate_f_unnorm_vs_vpa(run_info; f_over_vpa2=true, input=f_input, is=1,
                                iz=run_info.z.n, fig=fig, ax=ax, frame_index=frame_index,
                                label=label)
        vlines!(ax, @lift cutoff_upper[input.ir0,$frame_index]; linestyle=:dash, color=:red)
        if plot_prefix !== nothing && fig !== nothing
            outfile=plot_prefix * "pdf_unnorm_over_vpa2_wall+_vs_vpa." * input.animation_ext
            save_animation(fig, frame_index, run_info.nt, outfile)
        end
    end

    return nothing
end
