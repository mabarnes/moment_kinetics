"""
    plot_charged_pdf_2D_at_wall(run_info; plot_prefix, electron=false)

Make plots/animations of the ion distribution function at wall boundaries.

The information for the runs to plot is passed in `run_info` (as returned by
[`get_run_info`](@ref)). If `run_info` is a Vector, comparison plots are made where line
plots/animations from the different runs are overlayed on the same axis, and heatmap
plots/animations are displayed in a horizontal row.

Settings are read from the `[wall_pdf]` section of the input.

`plot_prefix` is required and gives the path and prefix for plots to be saved to. They
will be saved with the format `plot_prefix<some_identifying_string>.pdf`. When `run_info`
is not a Vector, `plot_prefix` is optional - plots/animations will be saved only if it is
passed.

If `electron=true` is passed, plot electron distribution function instead of ion
distribution function.
"""
function plot_charged_pdf_2D_at_wall(run_info; plot_prefix, electron=false)
    try
        if electron
            electron_prefix = "electron_"
            electron_suffix = "_electron"
        else
            electron_prefix = ""
            electron_suffix = ""
        end
        input = Dict_to_NamedTuple(input_dict_dfns["wall_pdf$electron_suffix"])
        if !(input.plot || input.animate || input.advection_velocity)
            # nothing to do
            return nothing
        end
        if !any(ri !== nothing for ri ∈ run_info)
            println("Warning: no distribution function output, skipping wall_pdf plots")
            return nothing
        end

        z_lower = 1
        z_upper = run_info[1].z.n
        if !all(ri.z.n == z_upper for ri ∈ run_info)
            println("Cannot run plot_charged_pdf_2D_at_wall() for runs with different "
                    * "z-grid sizes. Got $(Tuple(ri.z.n for ri ∈ run_info))")
            return nothing
        end

        if electron
            println("Making plots of electron distribution function at walls")
        else
            println("Making plots of ion distribution function at walls")
        end
        flush(stdout)

        has_rdim = any(ri !== nothing && ri.r.n > 1 for ri ∈ run_info)
        has_zdim = any(ri !== nothing && ri.z.n > 1 for ri ∈ run_info)
        is_1V = all(ri !== nothing && ri.vperp.n == 1 for ri ∈ run_info)
        moment_kinetic = !electron &&
                         any(ri !== nothing
                             && (ri.evolve_density || ri.evolve_upar || ri.evolve_p)
                             for ri ∈ run_info)

        nt = minimum(ri.nt for ri ∈ run_info)

        for (z, z_range, label) ∈ ((z_lower, z_lower:z_lower+input.n_points_near_wall, "wall-"),
                                   (z_upper, z_upper-input.n_points_near_wall:z_upper, "wall+"))
            f_input = copy(input_dict_dfns["f"])
            f_input["iz0"] = z

            if input.plot
                fig, ax = get_1d_ax(; xlabel="vpa", ylabel="f$electron_suffix")
                for iz ∈ z_range
                    for ri ∈ run_info
                        if length(run_info) > 1
                            run_label = ri.run_name * " "
                        else
                            run_label = ""
                        end
                        plot_vs_vpa(ri, "f$electron_suffix"; is=1, iz=iz, input=f_input,
                                    label="$(run_label)iz=$iz", ax=ax)
                    end
                end
                put_legend_below(fig, ax)
                # Ensure the first row width is 3/4 of the column width so that
                # the plot does not get squashed by the legend
                rowsize!(fig.layout, 1, Aspect(1, 3/4))
                resize_to_layout!(fig)
                outfile=plot_prefix * "pdf$(electron_suffix)_$(label)_vs_vpa.pdf"
                save(outfile, fig)

                fig, ax = get_1d_ax(; xlabel="vpa", ylabel="f")
                for iz ∈ z_range
                    for ri ∈ run_info
                        if length(run_info) > 1
                            run_label = ri.run_name * " "
                        else
                            run_label = ""
                        end
                        plot_vs_vpa(ri, "f$electron_suffix"; is=1, iz=iz, input=f_input,
                                    label="$(run_label)iz=$iz", ax=ax, yscale=log10,
                                    transform=(x)->positive_or_nan(x; epsilon=1.e-20))
                    end
                end
                put_legend_below(fig, ax)
                # Ensure the first row width is 3/4 of the column width so that
                # the plot does not get squashed by the legend
                rowsize!(fig.layout, 1, Aspect(1, 3/4))
                resize_to_layout!(fig)
                outfile=plot_prefix * "logpdf$(electron_suffix)_$(label)_vs_vpa.pdf"
                save(outfile, fig)

                if moment_kinetic
                    fig, ax = get_1d_ax(; xlabel="vpa_unnorm", ylabel="f$(electron_suffix)_unnorm")
                    for iz ∈ z_range
                        for ri ∈ run_info
                            if length(run_info) > 1
                                run_label = ri.run_name * " "
                            else
                                run_label = ""
                            end
                            plot_f_unnorm_vs_vpa(ri; input=f_input, is=1, iz=iz,
                                                 label="$(run_label)iz=$iz", ax=ax)
                        end
                    end
                    put_legend_below(fig, ax)
                    # Ensure the first row width is 3/4 of the column width so that
                    # the plot does not get squashed by the legend
                    rowsize!(fig.layout, 1, Aspect(1, 3/4))
                    resize_to_layout!(fig)
                    outfile=plot_prefix * "pdf_unnorm_$(label)_vs_vpa.pdf"
                    save(outfile, fig)

                    fig, ax = get_1d_ax(; xlabel="vpa_unnorm", ylabel="f_unnorm")
                    for iz ∈ z_range
                        for ri ∈ run_info
                            if length(run_info) > 1
                                run_label = ri.run_name * " "
                            else
                                run_label = ""
                            end
                            plot_f_unnorm_vs_vpa(ri; input=f_input, is=1, iz=iz,
                                                 label="$(run_label)iz=$iz", ax=ax, yscale=log10,
                                                 transform=(x)->positive_or_nan(x; epsilon=1.e-20))
                        end
                    end
                    put_legend_below(fig, ax)
                    # Ensure the first row width is 3/4 of the column width so that
                    # the plot does not get squashed by the legend
                    rowsize!(fig.layout, 1, Aspect(1, 3/4))
                    resize_to_layout!(fig)
                    outfile=plot_prefix * "logpdf_unnorm_$(label)_vs_vpa.pdf"
                    save(outfile, fig)
                end

                if !is_1V
                    plot_vs_vpa_vperp(run_info, "f$electron_suffix"; is=1, input=f_input,
                                      outfile=plot_prefix * "pdf$(electron_suffix)_$(label)_vs_vpa_vperp.pdf")
                end

                if has_zdim
                    plot_vs_vpa_z(run_info, "f$electron_suffix"; is=1, input=f_input, iz=z_range,
                                  outfile=plot_prefix * "pdf$(electron_suffix)_$(label)_vs_vpa_z.pdf")
                end

                if has_rdim && has_zdim
                    plot_vs_z_r(run_info, "f$electron_suffix"; is=1, input=f_input, iz=z_range,
                                outfile=plot_prefix * "pdf$(electron_suffix)_$(label)_vs_z_r.pdf")
                end

                if has_rdim
                    plot_vs_vpa_r(run_info, "f$electron_suffix"; is=1, input=f_input,
                                  outfile=plot_prefix * "pdf$(electron_suffix)_$(label)_vs_vpa_r.pdf")
                end
            end

            if input.animate
                fig, ax = get_1d_ax(; xlabel="vpa", ylabel="f$electron_suffix")
                frame_index = Observable(1)
                for iz ∈ z_range
                    for ri ∈ run_info
                        if length(run_info) > 1
                            run_label = ri.run_name * " "
                        else
                            run_label = ""
                        end
                        animate_vs_vpa(ri, "f$electron_suffix"; is=1, iz=iz, input=f_input,
                                       label="$(run_label)iz=$iz", ax=ax,
                                       frame_index=frame_index)
                    end
                end
                put_legend_below(fig, ax)
                # Ensure the first row width is 3/4 of the column width so that
                # the plot does not get squashed by the legend
                rowsize!(fig.layout, 1, Aspect(1, 3/4))
                resize_to_layout!(fig)
                outfile=plot_prefix * "pdf$(electron_suffix)_$(label)_vs_vpa." * input.animation_ext
                save_animation(fig, frame_index, nt, outfile)

                fig, ax = get_1d_ax(; xlabel="vpa", ylabel="f$electron_suffix", yscale=log10)
                frame_index = Observable(1)
                for iz ∈ z_range
                    for ri ∈ run_info
                        if length(run_info) > 1
                            run_label = ri.run_name * " "
                        else
                            run_label = ""
                        end
                        animate_vs_vpa(ri, "f$electron_suffix"; is=1, iz=iz, input=f_input,
                                       label="$(run_label)iz=$iz", ax=ax,
                                       frame_index=frame_index,
                                       transform=(x)->positive_or_nan(x; epsilon=1.e-20))
                    end
                end
                put_legend_below(fig, ax)
                # Ensure the first row width is 3/4 of the column width so that
                # the plot does not get squashed by the legend
                rowsize!(fig.layout, 1, Aspect(1, 3/4))
                resize_to_layout!(fig)
                outfile=plot_prefix * "logpdf$(electron_suffix)_$(label)_vs_vpa." * input.animation_ext
                save_animation(fig, frame_index, nt, outfile)

                if moment_kinetic
                    fig, ax = get_1d_ax(; xlabel="vpa", ylabel="f")
                    frame_index = Observable(1)
                    for iz ∈ z_range
                        for ri ∈ run_info
                            if length(run_info) > 1
                                run_label = ri.run_name * " "
                            else
                                run_label = ""
                            end
                            animate_f_unnorm_vs_vpa(ri; is=1, iz=iz, input=f_input,
                                                    label="$(run_label)iz=$iz", ax=ax,
                                                    frame_index=frame_index)
                        end
                    end
                    put_legend_below(fig, ax)
                    # Ensure the first row width is 3/4 of the column width so that
                    # the plot does not get squashed by the legend
                    rowsize!(fig.layout, 1, Aspect(1, 3/4))
                    resize_to_layout!(fig)
                    outfile=plot_prefix * "pdf_unnorm_$(label)_vs_vpa." * input.animation_ext
                    save_animation(fig, frame_index, nt, outfile)

                    fig, ax = get_1d_ax(; xlabel="vpa", ylabel="f")
                    frame_index = Observable(1)
                    for iz ∈ z_range
                        for ri ∈ run_info
                            if length(run_info) > 1
                                run_label = ri.run_name * " "
                            else
                                run_label = ""
                            end
                            animate_f_unnorm_vs_vpa(ri; is=1, iz=iz, input=f_input,
                                                    label="$(run_label)iz=$iz", ax=ax,
                                                    frame_index=frame_index, yscale=log10,
                                                    transform=(x)->positive_or_nan(x; epsilon=1.e-20))
                        end
                    end
                    put_legend_below(fig, ax)
                    # Ensure the first row width is 3/4 of the column width so that
                    # the plot does not get squashed by the legend
                    rowsize!(fig.layout, 1, Aspect(1, 3/4))
                    resize_to_layout!(fig)
                    outfile=plot_prefix * "logpdf_unnorm_$(label)_vs_vpa." * input.animation_ext
                    save_animation(fig, frame_index, nt, outfile)
                end

                if !is_1V
                    animate_vs_vpa_vperp(run_info, "f$electron_suffix"; is=1, input=f_input,
                                         outfile=plot_prefix * "pdf$(electron_suffix)_$(label)_vs_vpa_vperp." * input.animation_ext)
                end

                if has_zdim
                    animate_vs_vpa_z(run_info, "f$electron_suffix"; is=1, input=f_input, iz=z_range,
                                     outfile=plot_prefix * "pdf$(electron_suffix)_$(label)_vs_vpa_z." * input.animation_ext)
                end

                if has_rdim && has_zdim
                    animate_vs_z_r(run_info, "f$electron_suffix"; is=1, input=f_input, iz=z_range,
                                   outfile=plot_prefix * "pdf$(electron_suffix)_$(label)_vs_z_r." * input.animation_ext)
                end

                if has_rdim
                    animate_vs_vpa_r(run_info, "f$electron_suffix"; is=1, input=f_input,
                                     outfile=plot_prefix * "pdf$(electron_suffix)_$(label)_vs_vpa_r." * input.animation_ext)
                end
            end

            if input.advection_velocity
                # Need to get variable without selecting wall point so that z-derivatives
                # can be calculated.
                vpa_advect_speed = get_variable(run_info, "$(electron_prefix)vpa_advect_speed")
                animate_vs_vpa(run_info, "$(electron_prefix)vpa_advect_speed";
                               data=vpa_advect_speed, is=1, input=f_input,
                               outfile=plot_prefix * "$(electron_prefix)vpa_advect_speed_$(label)_vs_vpa." * input.animation_ext)
            end
        end
    catch e
        return makie_post_processing_error_handler(
                   e,
                   "Error in plot_charged_pdf_2D_at_wall().")
    end

    return nothing
end

"""
    plot_neutral_pdf_2D_at_wall(run_info; plot_prefix)

Make plots/animations of the neutral particle distribution function at wall boundaries.

The information for the runs to plot is passed in `run_info` (as returned by
[`get_run_info`](@ref)). If `run_info` is a Vector, comparison plots are made where line
plots/animations from the different runs are overlayed on the same axis, and heatmap
plots/animations are displayed in a horizontal row.

Settings are read from the `[wall_pdf_neutral]` section of the input.

`plot_prefix` is required and gives the path and prefix for plots to be saved to. They
will be saved with the format `plot_prefix<some_identifying_string>.pdf`. When `run_info`
is not a Vector, `plot_prefix` is optional - plots/animations will be saved only if it is
passed.
"""
function plot_neutral_pdf_2D_at_wall(run_info; plot_prefix)
    try
        input = Dict_to_NamedTuple(input_dict_dfns["wall_pdf_neutral"])
        if !(input.plot || input.animate || input.advection_velocity)
            # nothing to do
            return nothing
        end
        if !any(ri !== nothing for ri ∈ run_info)
            println("Warning: no distribution function output, skipping wall_pdf plots")
            return nothing
        end

        z_lower = 1
        z_upper = run_info[1].z.n
        if !all(ri.z.n == z_upper for ri ∈ run_info)
            println("Cannot run plot_neutral_pdf_2D_at_wall() for runs with different "
                    * "z-grid sizes. Got $(Tuple(ri.z.n for ri ∈ run_info))")
            return nothing
        end

        println("Making plots of neutral distribution function at walls")
        flush(stdout)

        has_rdim = any(ri !== nothing && ri.r.n > 1 for ri ∈ run_info)
        has_zdim = any(ri !== nothing && ri.z.n > 1 for ri ∈ run_info)
        is_1V = all(ri !== nothing && ri.vzeta.n == 1 && ri.vr.n == 1 for ri ∈ run_info)
        moment_kinetic = any(ri !== nothing
                             && (ri.evolve_density || ri.evolve_upar || ri.evolve_p)
                             for ri ∈ run_info)
        nt = minimum(ri.nt for ri ∈ run_info)

        for (z, z_range, label) ∈ ((z_lower, z_lower:z_lower+input.n_points_near_wall, "wall-"),
                                   (z_upper, z_upper-input.n_points_near_wall:z_upper, "wall+"))
            f_neutral_input = copy(input_dict_dfns["f_neutral"])
            f_neutral_input["iz0"] = z

            if input.plot
                fig, ax = get_1d_ax(; xlabel="vz", ylabel="f_neutral")
                for iz ∈ z_range
                    for ri ∈ run_info
                        if length(run_info) > 1
                            run_label = ri.run_name * " "
                        else
                            run_label = ""
                        end
                        plot_vs_vz(ri, "f_neutral"; is=1, iz=iz, input=f_neutral_input,
                                   label="$(run_label)iz=$iz", ax=ax)
                    end
                end
                put_legend_below(fig, ax)
                # Ensure the first row width is 3/4 of the column width so that
                # the plot does not get squashed by the legend
                rowsize!(fig.layout, 1, Aspect(1, 3/4))
                resize_to_layout!(fig)
                outfile=plot_prefix * "pdf_neutral_$(label)_vs_vz.pdf"
                save(outfile, fig)

                fig, ax = get_1d_ax(; xlabel="vz", ylabel="f_neutral")
                for iz ∈ z_range
                    for ri ∈ run_info
                        if length(run_info) > 1
                            run_label = ri.run_name * " "
                        else
                            run_label = ""
                        end
                        plot_vs_vz(ri, "f_neutral"; is=1, iz=iz, input=f_neutral_input,
                                   label="$(run_label)iz=$iz", ax=ax, yscale=log10,
                                   transform=(x)->positive_or_nan(x; epsilon=1.e-20))
                    end
                end
                put_legend_below(fig, ax)
                # Ensure the first row width is 3/4 of the column width so that
                # the plot does not get squashed by the legend
                rowsize!(fig.layout, 1, Aspect(1, 3/4))
                resize_to_layout!(fig)
                outfile=plot_prefix * "logpdf_neutral_$(label)_vs_vpa.pdf"
                save(outfile, fig)

                if moment_kinetic
                    fig, ax = get_1d_ax(; xlabel="vz_unnorm", ylabel="f_neutral_unnorm")
                    for iz ∈ z_range
                        for ri ∈ run_info
                            if length(run_info) > 1
                                run_label = ri.run_name * " "
                            else
                                run_label = ""
                            end
                            plot_f_unnorm_vs_vpa(ri; neutral=true, input=f_neutral_input,
                                                 is=1, iz=iz, label="$(run_label)iz=$iz",
                                                 ax=ax)
                        end
                    end
                    put_legend_below(fig, ax)
                    # Ensure the first row width is 3/4 of the column width so that
                    # the plot does not get squashed by the legend
                    rowsize!(fig.layout, 1, Aspect(1, 3/4))
                    resize_to_layout!(fig)
                    outfile=plot_prefix * "pdf_neutral_unnorm_$(label)_vs_vpa.pdf"
                    save(outfile, fig)

                    fig, ax = get_1d_ax(; xlabel="vz_unnorm", ylabel="f_neutral_unnorm")
                    for iz ∈ z_range
                        for ri ∈ run_info
                            if length(run_info) > 1
                                run_label = ri.run_name * " "
                            else
                                run_label = ""
                            end
                            plot_f_unnorm_vs_vpa(ri; neutral=true, input=f_neutral_input,
                                                 is=1, iz=iz, label="$(run_label)iz=$iz",
                                                 ax=ax, yscale=log10,
                                                 transform=(x)->positive_or_nan(x; epsilon=1.e-20))
                        end
                    end
                    put_legend_below(fig, ax)
                    # Ensure the first row width is 3/4 of the column width so that
                    # the plot does not get squashed by the legend
                    rowsize!(fig.layout, 1, Aspect(1, 3/4))
                    resize_to_layout!(fig)
                    outfile=plot_prefix * "logpdf_neutral_unnorm_$(label)_vs_vpa.pdf"
                    save(outfile, fig)
                end

                if !is_1V
                    plot_vs_vzeta_vr(run_info, "f_neutral"; is=1, input=f_neutral_input,
                                     outfile=plot_prefix * "pdf_neutral_$(label)_vs_vr_vzeta.pdf")
                    plot_vs_vzeta_vz(run_info, "f_neutral"; is=1, input=f_neutral_input,
                                     outfile=plot_prefix * "pdf_neutral_$(label)_vs_vz_vzeta.pdf")
                    plot_vs_vr_vz(run_info, "f_neutral"; is=1, input=f_neutral_input,
                                  outfile=plot_prefix * "pdf_neutral_$(label)_vs_vz_vr.pdf")
                end

                if has_zdim
                    plot_vs_vz_z(run_info, "f_neutral"; is=1, input=f_neutral_input, iz=z_range,
                                 outfile=plot_prefix * "pdf_neutral_$(label)_vs_vz_z.pdf")
                end

                if has_zdim && !is_1V
                    plot_vs_vzeta_z(run_info, "f_neutral"; is=1, input=f_neutral_input, iz=z_range,
                                    outfile=plot_prefix * "pdf_neutral_$(label)_vs_vzeta_z.pdf")
                    plot_vs_vr_z(run_info, "f_neutral"; is=1, input=f_neutral_input, iz=z_range,
                                 outfile=plot_prefix * "pdf_neutral_$(label)_vs_vr_z.pdf")
                end

                if has_rdim && has_zdim
                    plot_vs_z_r(run_info, "f_neutral"; is=1, input=f_neutral_input, iz=z_range,
                                outfile=plot_prefix * "pdf_neutral_$(label)_vs_z_r.pdf")
                end

                if has_rdim
                    plot_vs_vz_r(run_info, "f_neutral"; is=1, input=f_neutral_input,
                                 outfile=plot_prefix * "pdf_neutral_$(label)_vs_vz_r.pdf")
                    if !is_1V
                        plot_vs_vzeta_r(run_info, "f_neutral"; is=1, input=f_neutral_input,
                                        outfile=plot_prefix * "pdf_neutral_$(label)_vs_vzeta_r.pdf")

                        plot_vs_vr_r(run_info, "f_neutral"; is=1, input=f_neutral_input,
                                     outfile=plot_prefix * "pdf_neutral_$(label)_vs_vr_r.pdf")
                    end
                end
            end

            if input.animate
                fig, ax = get_1d_ax(; xlabel="vz", ylabel="f_neutral")
                frame_index = Observable(1)
                for iz ∈ z_range
                    for ri ∈ run_info
                        if length(run_info) > 1
                            run_label = ri.run_name * " "
                        else
                            run_label = ""
                        end
                        animate_vs_vz(ri, "f_neutral"; is=1, iz=iz, input=f_neutral_input,
                                      label="$(run_label)iz=$iz", ax=ax,
                                      frame_index=frame_index)
                    end
                end
                put_legend_below(fig, ax)
                # Ensure the first row width is 3/4 of the column width so that
                # the plot does not get squashed by the legend
                rowsize!(fig.layout, 1, Aspect(1, 3/4))
                resize_to_layout!(fig)
                outfile=plot_prefix * "pdf_neutral_$(label)_vs_vz." * input.animation_ext
                save_animation(fig, frame_index, nt, outfile)

                fig, ax = get_1d_ax(; xlabel="vz", ylabel="f_neutral", yscale=log10)
                frame_index = Observable(1)
                for iz ∈ z_range
                    for ri ∈ run_info
                        if length(run_info) > 1
                            run_label = ri.run_name * " "
                        else
                            run_label = ""
                        end
                        animate_vs_vz(ri, "f_neutral"; is=1, iz=iz, input=f_neutral_input,
                                      label="$(run_label)iz=$iz", ax=ax,
                                      frame_index=frame_index,
                                      transform=(x)->positive_or_nan(x; epsilon=1.e-20))
                    end
                end
                put_legend_below(fig, ax)
                # Ensure the first row width is 3/4 of the column width so that
                # the plot does not get squashed by the legend
                rowsize!(fig.layout, 1, Aspect(1, 3/4))
                resize_to_layout!(fig)
                outfile=plot_prefix * "logpdf_neutral_$(label)_vs_vz." * input.animation_ext
                save_animation(fig, frame_index, nt, outfile)

                if moment_kinetic
                    fig, ax = get_1d_ax(; xlabel="vz", ylabel="f_neutral")
                    frame_index = Observable(1)
                    for iz ∈ z_range
                        for ri ∈ run_info
                            if length(run_info) > 1
                                run_label = ri.run_name * " "
                            else
                                run_label = ""
                            end
                            animate_f_unnorm_vs_vpa(ri; neutral=true, is=1, iz=iz,
                                                    input=f_neutral_input,
                                                    label="$(run_label)iz=$iz", ax=ax,
                                                    frame_index=frame_index)
                        end
                    end
                    put_legend_below(fig, ax)
                    # Ensure the first row width is 3/4 of the column width so that
                    # the plot does not get squashed by the legend
                    rowsize!(fig.layout, 1, Aspect(1, 3/4))
                    resize_to_layout!(fig)
                    outfile=plot_prefix * "pdf_neutral_unnorm_$(label)_vs_vz." * input.animation_ext
                    save_animation(fig, frame_index, nt, outfile)

                    fig, ax = get_1d_ax(; xlabel="vz", ylabel="f_neutral")
                    frame_index = Observable(1)
                    for iz ∈ z_range
                        for ri ∈ run_info
                            if length(run_info) > 1
                                run_label = ri.run_name * " "
                            else
                                run_label = ""
                            end
                            animate_f_unnorm_vs_vpa(ri; neutral=true, is=1, iz=iz,
                                                    input=f_neutral_input, label="$(run_label)iz=$iz",
                                                    ax=ax, frame_index=frame_index, yscale=log10,
                                                    transform=(x)->positive_or_nan(x; epsilon=1.e-20))
                        end
                    end
                    put_legend_below(fig, ax)
                    # Ensure the first row width is 3/4 of the column width so that
                    # the plot does not get squashed by the legend
                    rowsize!(fig.layout, 1, Aspect(1, 3/4))
                    resize_to_layout!(fig)
                    outfile=plot_prefix * "logpdf_neutral_unnorm_$(label)_vs_vz." * input.animation_ext
                    save_animation(fig, frame_index, nt, outfile)
                end

                if !is_1V
                    animate_vs_vzeta_vr(run_info, "f_neutral"; is=1, input=f_neutral_input,
                                        outfile=plot_prefix * "pdf_neutral_$(label)_vs_vr_vzeta." * input.animation_ext)
                    animate_vs_vzeta_vz(run_info, "f_neutral"; is=1, input=f_neutral_input,
                                        outfile=plot_prefix * "pdf_neutral_$(label)_vs_vz_vzeta." * input.animation_ext)
                    animate_vs_vr_vz(run_info, "f_neutral"; is=1, input=f_neutral_input,
                                     outfile=plot_prefix * "pdf_neutral_$(label)_vs_vz_vr." * input.animation_ext)
                end

                if has_zdim
                    animate_vs_vz_z(run_info, "f_neutral"; is=1, input=f_neutral_input, iz=z_range,
                                    outfile=plot_prefix * "pdf_neutral_$(label)_vs_vz_z." * input.animation_ext)
                end

                if has_zdim && !is_1V
                    animate_vs_vzeta_z(run_info, "f_neutral"; is=1, input=f_neutral_input, iz=z_range,
                                       outfile=plot_prefix * "pdf_neutral_$(label)_vs_vzeta_z." * input.animation_ext)
                    animate_vs_vr_z(run_info, "f_neutral"; is=1, input=f_neutral_input, iz=z_range,
                                    outfile=plot_prefix * "pdf_neutral_$(label)_vs_vr_z." * input.animation_ext)
                end

                if has_rdim && has_zdim
                    animate_vs_z_r(run_info, "f_neutral"; is=1, input=f_neutral_input, iz=z_range,
                                   outfile=plot_prefix * "pdf_neutral_$(label)_vs_z_r." * input.animation_ext)
                end

                if has_rdim
                    animate_vs_vz_r(run_info, "f_neutral"; is=1, input=f_neutral_input,
                                    outfile=plot_prefix * "pdf_neutral_$(label)_vs_vz_r." * input.animation_ext)
                    if !is_1V
                        animate_vs_vzeta_r(run_info, "f_neutral"; is=1, input=f_neutral_input,
                                           outfile=plot_prefix * "pdf_neutral_$(label)_vs_vzeta_r." * input.animation_ext)
                        animate_vs_vr_r(run_info, "f_neutral"; is=1, input=f_neutral_input,
                                        outfile=plot_prefix * "pdf_neutral_$(label)_vs_vr_r." * input.animation_ext)
                    end
                end
            end

            if input.advection_velocity
                # Need to get variable without selecting wall point so that z-derivatives
                # can be calculated.
                vz_advect_speed = get_variable(run_info, "neutral_vz_advect_speed")
                animate_vs_vz(run_info, "neutral_vz_advect_speed"; data=vz_advect_speed,
                              is=1, input=f_neutral_input,
                              outfile=plot_prefix * "neutral_vz_advect_speed_$(label)_vs_vz." * input.animation_ext)
            end
        end
    catch e
        return makie_post_processing_error_handler(
                   e,
                   "Error in plot_neutral_pdf_2D_at_wall().")
    end

    return nothing
end
