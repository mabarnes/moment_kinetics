"""
A function to plot collisionalities. The mean free path is plotted (or animated) 
along with the lengthscales of the gradients of density, parallel flow and temperature.

There are also functions to check the calculations of the mean free path and the 
comparison of temperature, L_T and dT_dz. They would only be for making sure
lengthscales and mean free path calculations are sensible.
"""
function collisionality_plots(run_info, run_info_dfns, plot_prefix=nothing)
    if !isa(run_info, AbstractVector)
        run_info = Any[run_info]
    end
    if !isa(run_info_dfns, AbstractVector)
        run_info_dfns = Any[run_info_dfns]
    end
    input = Dict_to_NamedTuple(input_dict["collisionality_plots"])

    if input.plot
        println("Making plots for collisionality")

        temperature_input = Dict_to_NamedTuple(input_dict["temperature"])
        mfp = get_variable(run_info, "mfp")
        L_T = get_variable(run_info, "L_T")
        L_n = get_variable(run_info, "L_n")
        L_upar = get_variable(run_info, "L_upar")
        nt = minimum(length(mfp[ri][1,1,1,:]) for ri in eachindex(run_info))
        # print warning if the lengths of all the mfp[ri][1,1,1,:] are not the same
        if any(length(mfp[ri][1,1,1,:]) != nt for ri in eachindex(run_info))
            println("Warning: The number of timesteps of some simulations are different, " *
                    "so only the first common timesteps will be animated.")
        end

        # write function to check that mfp[ri][1, 1, 1, :] is the same length (i.e. nt) for all ri 
        if input.plot_mfp_vs_z
            variable_name = "mean_free_path"
            variable = mfp
            variable_prefix = plot_prefix * variable_name
            plot_vs_z(run_info, variable_name, is=1, data=variable, input=temperature_input,
                    outfile=variable_prefix * "_vs_z.pdf")
        end

        if input.animate_mfp_vs_z
            variable_name = "mean_free_path"
            variable = mfp
            variable_prefix = plot_prefix * variable_name
            animate_vs_z(run_info, variable_name, is=1, data=variable, input=temperature_input,
                            outfile=variable_prefix * "_vs_z." * input.animation_ext)
        end

        if input.plot_dT_dz_vs_z
            variable_name = "dT_dz"
            variable = nothing
            try
                variable = get_variable(run_info, variable_name)
            catch e
                return makie_post_processing_error_handler(
                        e,
                        "collisionality_plots () failed for $variable_name - could not load data.")
            end

            variable_prefix = plot_prefix * variable_name
            plot_vs_z(run_info, variable_name, is=1, data=variable, input=temperature_input,
                    outfile=variable_prefix * "_vs_z.pdf")
        end

        if input.animate_dT_dz_vs_z
            variable_name = "dT_dz"
            variable = nothing
            try
                variable = get_variable(run_info, variable_name)
            catch e
                return makie_post_processing_error_handler(
                        e,
                        "collisionality_plots () failed for $variable_name - could not load data.")
            end

            variable_prefix = plot_prefix * variable_name
            animate_vs_z(run_info, variable_name, is=1, data=variable, input=temperature_input,
                            outfile=variable_prefix * "_vs_z." * input.animation_ext)
        end

        if input.plot_nu_ii_vth_mfp_vs_z
            vth = get_variable(run_info, "thermal_speed")
            nu_ii = get_variable(run_info, "collision_frequency_ii")
            variable_prefix = plot_prefix * "checking_mfp_vth_and_nu_ii"

            fig, ax, legend_place = get_1d_ax(1; get_legend_place=:below)
            for ri ∈ eachindex(run_info)
                if length(run_info) > 1
                    run_label = run_info[ri].run_name * " "
                else
                    run_label = " "
                end
                plot_1d(run_info[ri].z.grid, vth[ri][:,1,1,end], xlabel="z",
                        ylabel="values", label=run_label*"vth", ax=ax[1], title = "checking_mfp_vth")
                plot_1d(run_info[ri].z.grid, nu_ii[ri][:,1,1,end], label=run_label*"nu_ii", ax=ax[1])
                plot_1d(run_info[ri].z.grid, mfp[ri][:,1,1,end], label=run_label*"mfp", ax=ax[1])
            end
            Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false,
                orientation=:vertical)
            outfile = variable_prefix * "_vs_z.pdf"
            save(outfile, fig)
        end

        if input.plot_LT_dT_dz_temp_vs_z
            dT_dz = get_variable(run_info, "dT_dz")
            temp = get_variable(run_info, "temperature")
            variable_prefix = plot_prefix * "LT_dT_dz_temp"
            fig, ax, legend_place = get_1d_ax(1; get_legend_place=:below)
            for ri ∈ eachindex(run_info)
                if length(run_info) > 1
                    run_label = run_info[ri].run_name * " "
                else
                    run_label = " "
                end
                plot_1d(run_info[ri].z.grid, L_T[ri][:,1,1,end], xlabel="z",
                        ylabel="values", label=run_label*"L_T", ax=ax[1], title = "LT_dT_dz_temp")
                plot_1d(run_info[ri].z.grid, dT_dz[ri][:,1,1,end], label=run_label*"dT_dz", ax=ax[1])
                plot_1d(run_info[ri].z.grid, temp[ri][:,1,1,end], label=run_label*"temp", ax=ax[1])
            end
            Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false,
                orientation=:vertical)
            outfile = variable_prefix * "_vs_z.pdf"
            save(outfile, fig)
        end

        if input.plot_LT_mfp_vs_z
            variable_prefix = plot_prefix * "LT_mfp"
            fig, ax, legend_place = get_1d_ax(1; get_legend_place=:below)
            for ri ∈ eachindex(run_info)
                if length(run_info) > 1
                    run_label = run_info[ri].run_name * " "
                else
                    run_label = " "
                end
                plot_1d(run_info[ri].z.grid, L_T[ri][:,1,1,end], xlabel="z",
                        ylabel="values", label=run_label*"L_T", ax=ax[1], title = "L_T and mean free path comparison")
                plot_1d(run_info[ri].z.grid, mfp[ri][:,1,1,end], label=run_label*"mfp", ax=ax[1])
            end
            Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false,
                orientation=:vertical)
            outfile = variable_prefix * "_vs_z.pdf"
            save(outfile, fig)
        end

        if input.animate_LT_mfp_vs_z
            variable_prefix = plot_prefix * "LT_mfp"
            fig, ax, legend_place = get_1d_ax(1; get_legend_place=:below)
            frame_index = Observable(1)
            for ri ∈ eachindex(run_info)
                if length(run_info) > 1
                    run_label = run_info[ri].run_name * " "
                else
                    run_label = " "
                end
                animate_1d(run_info[ri].z.grid, L_T[ri][:,1,1,:],
                        frame_index=frame_index, xlabel="z", ylabel="values",
                        label=run_label*"L_T", ax=ax[1], title = "L_T and mean free path comparison")
                animate_1d(run_info[ri].z.grid, mfp[ri][:,1,1,:],
                        frame_index=frame_index, xlabel="z", ylabel="values",
                        label=run_label*"mfp", ax=ax[1])
            end
            Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false,
                orientation=:vertical)
            outfile = variable_prefix * "_vs_z." * input.animation_ext
            save_animation(fig, frame_index, nt, outfile)
        end

        if input.plot_Ln_mfp_vs_z
            variable_prefix = plot_prefix * "Ln_mfp"
            fig, ax, legend_place = get_1d_ax(1; get_legend_place=:below)
            for ri ∈ eachindex(run_info)
                if length(run_info) > 1
                    run_label = run_info[ri].run_name * " "
                else
                    run_label = " "
                end
                plot_1d(run_info[ri].z.grid, L_n[ri][:,1,1,end], xlabel="z",
                        ylabel="values", label=run_label*"L_n", ax=ax[1], title = "Ln and mean free path comparison")
                plot_1d(run_info[ri].z.grid, mfp[ri][:,1,1,end], label=run_label*"mfp", ax=ax[1])
            end
            Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false,
                orientation=:vertical)
            outfile = variable_prefix * "_vs_z.pdf"
            save(outfile, fig)
        end

        if input.animate_Ln_mfp_vs_z
            variable_prefix = plot_prefix * "Ln_mfp"
            fig, ax, legend_place = get_1d_ax(1; get_legend_place=:below)
            frame_index = Observable(1)
            for ri ∈ eachindex(run_info)
                if length(run_info) > 1
                    run_label = run_info[ri].run_name * " "
                else
                    run_label = " "
                end
                animate_1d(run_info[ri].z.grid, L_n[ri][:,1,1,:],
                        frame_index=frame_index, xlabel="z", ylabel="values",
                        label=run_label*"L_n", ax=ax[1], title = "L_n and mean free path comparison")
                animate_1d(run_info[ri].z.grid, mfp[ri][:,1,1,:],
                        frame_index=frame_index, xlabel="z", ylabel="values",
                        label=run_label*"mfp", ax=ax[1])
            end
            Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false,
                orientation=:vertical)
            outfile = variable_prefix * "_vs_z." * input.animation_ext
            save_animation(fig, frame_index, nt, outfile)
        end

        if input.plot_Lupar_mfp_vs_z
            variable_prefix = plot_prefix * "Lupar_mfp"
            fig, ax, legend_place = get_1d_ax(1; get_legend_place=:below)
            for ri ∈ eachindex(run_info)
                if length(run_info) > 1
                    run_label = run_info[ri].run_name * " "
                else
                    run_label = " "
                end
                plot_1d(run_info[ri].z.grid, L_upar[ri][:,1,1,end], xlabel="z",
                        ylabel="values", label=run_label*"L_upar", ax=ax[1], title = "Lupar and mean free path comparison")
                plot_1d(run_info[ri].z.grid, mfp[ri][:,1,1,end], label=run_label*"mfp", ax=ax[1])
            end
            Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false,
                orientation=:vertical)
            outfile = variable_prefix * "_vs_z.pdf"
            save(outfile, fig)
        end

        if input.animate_Lupar_mfp_vs_z
            variable_prefix = plot_prefix * "Lupar_mfp"
            fig, ax, legend_place = get_1d_ax(1; get_legend_place=:below)
            frame_index = Observable(1)
            for ri ∈ eachindex(run_info)
                if length(run_info) > 1
                    run_label = run_info[ri].run_name * " "
                else
                    run_label = " "
                end
                animate_1d(run_info[ri].z.grid, L_upar[ri][:,1,1,:],
                        frame_index=frame_index, xlabel="z", ylabel="values",
                        label=run_label*"L_upar", ax=ax[1], title = "L_upar and mean free path comparison")
                animate_1d(run_info[ri].z.grid, mfp[ri][:,1,1,:],
                        frame_index=frame_index, xlabel="z", ylabel="values",
                        label=run_label*"mfp", ax=ax[1])
            end
            Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false,
                orientation=:vertical)
            outfile = variable_prefix * "_vs_z." * input.animation_ext
            save_animation(fig, frame_index, nt, outfile)
        end

        if input.plot_Lupar_Ln_LT_mfp_vs_z
            variable_prefix = plot_prefix * "Lupar_Ln_LT_mfp"
            fig, ax, legend_place = get_1d_ax(1; get_legend_place=:below)
            for ri ∈ eachindex(run_info)
                if length(run_info) > 1
                    run_label = run_info[ri].run_name * " "
                else
                    run_label = " "
                end
                plot_1d(run_info[ri].z.grid, L_upar[ri][:,1,1,end], xlabel="z",
                        ylabel="values", label=run_label*"L_upar", ax=ax[1], title = "Lupar Ln LT and mean free path comparison")
                plot_1d(run_info[ri].z.grid, L_n[ri][:,1,1,end], label=run_label*"L_n", ax=ax[1])
                plot_1d(run_info[ri].z.grid, L_T[ri][:,1,1,end], label=run_label*"L_T", ax=ax[1])
                plot_1d(run_info[ri].z.grid, mfp[ri][:,1,1,end], label=run_label*"mfp", ax=ax[1])
            end
            Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false,
                orientation=:horizontal)
            outfile = variable_prefix * "_vs_z.pdf"
            save(outfile, fig)
        end

        if input.animate_Lupar_Ln_LT_mfp_vs_z
            variable_prefix = plot_prefix * "Lupar_Ln_LT_mfp"
            fig, ax, legend_place = get_1d_ax(1; get_legend_place=:below)
            frame_index = Observable(1)
            for ri ∈ eachindex(run_info)
                if length(run_info) > 1
                    run_label = run_info[ri].run_name * " "
                else
                    run_label = " "
                end
                animate_1d(run_info[ri].z.grid, L_upar[ri][:,1,1,:],
                        frame_index=frame_index, xlabel="z", ylabel="values",
                        label=run_label*"L_upar", ax=ax[1], title = "Lupar Ln LT and mean free path comparison")
                animate_1d(run_info[ri].z.grid, L_n[ri][:,1,1,:],
                        frame_index=frame_index, xlabel="z", ylabel="values",
                        label=run_label*"L_n", ax=ax[1])
                animate_1d(run_info[ri].z.grid, L_T[ri][:,1,1,:],
                        frame_index=frame_index, xlabel="z", ylabel="values",
                        label=run_label*"L_T", ax=ax[1])
                animate_1d(run_info[ri].z.grid, mfp[ri][:,1,1,:],
                        frame_index=frame_index, xlabel="z", ylabel="values",
                        label=run_label*"mfp", ax=ax[1])
            end
            Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false,
                orientation=:vertical)
            outfile = variable_prefix * "_vs_z." * input.animation_ext
            save_animation(fig, frame_index, nt, outfile)
        end

        if input.plot_overlay_coll_krook_heat_flux
            variable_prefix = plot_prefix * "coll_krook_vs_original_heat_flux"
            coll_krook_q = get_variable(run_info, "coll_krook_heat_flux")
            original_q = get_variable(run_info, "parallel_heat_flux")
            fig, ax, legend_place = get_1d_ax(1; get_legend_place=:below)
            for ri ∈ eachindex(run_info)
                if length(run_info) > 1
                    run_label = run_info[ri].run_name * " "
                else
                    run_label = " "
                end
                if get(run_info[ri].input["composition"], "ion_physics", "") !== coll_krook_ions
                    plot_1d(run_info[ri].z.grid, coll_krook_q[ri][:,1,1,end], xlabel="z",
                            ylabel="values", label=run_label*"coll_krook_q_overlay", ax=ax[1], title = "coll_krook heat flux overlay")
                end
                plot_1d(run_info[ri].z.grid, original_q[ri][:,1,1,end], label=run_label*"original_q", ax=ax[1])
            end
            Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false,
                orientation=:vertical)
            outfile = variable_prefix * "_vs_z.pdf"
            save(outfile, fig)
        end

        if input.animate_overlay_coll_krook_heat_flux
            variable_prefix = plot_prefix * "coll_krook_vs_original_heat_flux"
            coll_krook_q = get_variable(run_info, "coll_krook_heat_flux")
            original_q = get_variable(run_info, "parallel_heat_flux")
            fig, ax, legend_place = get_1d_ax(1; get_legend_place=:below)
            frame_index = Observable(1)
            for ri ∈ eachindex(run_info)
                if length(run_info) > 1
                    run_label = run_info[ri].run_name * " "
                else
                    run_label = " "
                end
                if get(run_info[ri].input["composition"], "ion_physics", "") !== coll_krook_ions
                    animate_1d(run_info[ri].z.grid, coll_krook_q[ri][:,1,1,:],
                            frame_index=frame_index, xlabel="z", ylabel="values",
                            label=run_label*"coll_krook_q_overlay", ax=ax[1], title = "coll_krook heat flux overlay")
                end
                animate_1d(run_info[ri].z.grid, original_q[ri][:,1,1,:],
                        frame_index=frame_index, xlabel="z", ylabel="values",
                        label=run_label*"original_q", ax=ax[1])
            end
            Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false,
                orientation=:vertical)
            outfile = variable_prefix * "_vs_z." * input.animation_ext
            save_animation(fig, frame_index, nt, outfile)
        end

        if run_info_dfns[1].dfns && input.plot_compare_Maxwellian
            f_input = Dict_to_NamedTuple(input_dict["f"])

            f = get_variable(run_info_dfns, "f_unnorm"; it=f_input.it0, is=1, ir=f_input.ir0, ivperp=f_input.ivperp0)
            f_M = get_variable(run_info_dfns, "local_Maxwellian"; it=f_input.it0, is=1, ir=f_input.ir0, ivperp=f_input.ivperp0)

            fig, ax, colorbar_place = get_2d_ax(3*length(run_info_dfns); xlabel="wpa", ylabel="z")

            for i ∈ length(run_info_dfns)
                plot_2d(run_info_dfns[i].vpa.grid, run_info_dfns[i].z.grid, f[i];
                        title="f", xlabel="wpa", ylabel="z", ax=ax[(i-1)*3+1],
                        colorbar_place=colorbar_place[(i-1)*3+1])
                plot_2d(run_info_dfns[i].vpa.grid, run_info_dfns[i].z.grid, f_M[i];
                        title="f_M", xlabel="wpa", ylabel="z", ax=ax[(i-1)*3+2],
                        colorbar_place=colorbar_place[(i-1)*3+2])
                plot_2d(run_info_dfns[i].vpa.grid, run_info_dfns[i].z.grid, f[i].-f_M[i];
                        title="f - f_M", xlabel="wpa", ylabel="z", ax=ax[(i-1)*3+3],
                        colorbar_place=colorbar_place[(i-1)*3+3])
            end

            save(plot_prefix * "Maxwellian_difference_vs_vpa_z.pdf", fig)

            iz0 = f_input.iz0
            fig, ax, legend_place = get_1d_ax(2; xlabel="wpa", get_legend_place=:below)

            for i ∈ length(run_info_dfns)
                plot_vs_vpa(run_info_dfns[i], "f"; data=f[i][:,iz0], label="f", ax=ax[1])
                plot_vs_vpa(run_info_dfns[i], "f_M"; data=f_M[i][:,iz0], label="f_M", ax=ax[1])
                plot_vs_vpa(run_info_dfns[i], "f - f_M"; data=f[i][:,iz0].-f_M[i][:,iz0],
                            ax=ax[2])
            end
            ax[1].ylabel = "f"
            Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false)
            ax[2].ylabel = "f - f_M"
            Legend(legend_place[2], ax[2]; tellheight=true, tellwidth=false)

            save(plot_prefix * "Maxwellian_difference_vs_vpa.pdf", fig)

            fig, ax, colorbar_place = get_2d_ax(3*length(run_info_dfns); xlabel="wpa", ylabel="z")

            for i ∈ length(run_info_dfns)
                plot_2d(run_info_dfns[i].vpa.grid, run_info_dfns[i].z.grid, f[i];
                        title="f", xlabel="wpa", ylabel="z", ax=ax[(i-1)*3+1],
                        colorbar_place=colorbar_place[(i-1)*3+1], colorscale=log10,
                        transform=x->positive_or_nan(x; epsilon=1.e-16))
                plot_2d(run_info_dfns[i].vpa.grid, run_info_dfns[i].z.grid, f_M[i];
                        title="f_M", xlabel="wpa", ylabel="z", ax=ax[(i-1)*3+2],
                        colorbar_place=colorbar_place[(i-1)*3+2], colorscale=log10,
                        transform=x->positive_or_nan(x; epsilon=1.e-16))
                plot_2d(run_info_dfns[i].vpa.grid, run_info_dfns[i].z.grid, f[i].-f_M[i];
                        title="f - f_M", xlabel="wpa", ylabel="z", ax=ax[(i-1)*3+3],
                        colorbar_place=colorbar_place[(i-1)*3+3], colorscale=log10,
                        transform=x->positive_or_nan(abs(x); epsilon=1.e-16))
            end

            save(plot_prefix * "Maxwellian_difference_log_vs_vpa_z.pdf", fig)

            iz0 = f_input.iz0
            fig, ax, legend_place = get_1d_ax(; yscale=log10, xlabel="wpa",
                                              get_legend_place=:below)

            for i ∈ length(run_info_dfns)
                plot_vs_vpa(run_info_dfns[i], "f"; data=f[i][:,iz0], label="f", ax=ax,
                            transform=x->positive_or_nan(x; epsilon=1.e-16))
                plot_vs_vpa(run_info_dfns[i], "f_M"; data=f_M[i][:,iz0], label="f_M", ax=ax,
                            transform=x->positive_or_nan(x; epsilon=1.e-16))
                plot_vs_vpa(run_info_dfns[i], "f - f_M"; data=f[i][:,iz0].-f_M[i][:,iz0],
                            label="f - f_M", ax=ax,
                            transform=x->positive_or_nan(abs(x); epsilon=1.e-16))
            end
            ax.ylabel = "f"
            Legend(legend_place, ax; tellheight=true, tellwidth=false)

            save(plot_prefix * "Maxwellian_difference_log_vs_vpa.pdf", fig)
        end

        if run_info_dfns[1].dfns && input.animate_compare_Maxwellian
            f_input = Dict_to_NamedTuple(input_dict["f"])

            f = get_variable(run_info_dfns, "f_unnorm"; is=1, ir=f_input.ir0, ivperp=f_input.ivperp0)
            f_M = get_variable(run_info_dfns, "local_Maxwellian"; is=1, ir=f_input.ir0, ivperp=f_input.ivperp0)

            fig, ax, colorbar_place = get_2d_ax(3*length(run_info_dfns); xlabel="wpa", ylabel="z")
            frame_index = Observable(1)

            for i ∈ length(run_info_dfns)
                animate_2d(run_info_dfns[i].vpa.grid, run_info_dfns[i].z.grid, f[i];
                           title="f", xlabel="wpa", ylabel="z", ax=ax[(i-1)*3+1],
                           colorbar_place=colorbar_place[(i-1)*3+1],
                           frame_index=frame_index)

                animate_2d(run_info_dfns[i].vpa.grid, run_info_dfns[i].z.grid, f_M[i];
                           title="f_M", xlabel="wpa", ylabel="z", ax=ax[(i-1)*3+2],
                           colorbar_place=colorbar_place[(i-1)*3+2],
                           frame_index=frame_index)

                animate_2d(run_info_dfns[i].vpa.grid, run_info_dfns[i].z.grid,
                           f[i].-f_M[i]; title="f - f_M", xlabel="wpa", ylabel="z",
                           ax=ax[(i-1)*3+3], colorbar_place=colorbar_place[(i-1)*3+3],
                           frame_index=frame_index)
            end

            save_animation(fig, frame_index, nt, plot_prefix * "Maxwellian_difference_vs_vpa_z.gif")

            iz0 = f_input.iz0
            fig, ax, legend_place = get_1d_ax(2; xlabel="wpa", get_legend_place=:below)
            frame_index = Observable(1)

            for i ∈ 1:length(run_info)
                animate_vs_vpa(run_info_dfns[i], "f"; data=f[i][:,iz0,:], label="f",
                               ax=ax[1], frame_index=frame_index)
                animate_vs_vpa(run_info_dfns[i], "f_M"; data=f_M[i][:,iz0,:], label="f_M",
                               ax=ax[1], frame_index=frame_index)
                animate_vs_vpa(run_info_dfns[i], "f - f_M";
                               data=f[i][:,iz0,:].-f_M[i][:,iz0,:], ax=ax[2],
                               frame_index=frame_index)
            end
            ax[1].ylabel = "f"
            Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false)
            ax[2].ylabel = "f - f_M"
            Legend(legend_place[2], ax[2]; tellheight=true, tellwidth=false)

            save_animation(fig, frame_index, nt, plot_prefix * "Maxwellian_difference_vs_vpa.gif")

            fig, ax, colorbar_place = get_2d_ax(3*length(run_info_dfns); xlabel="wpa", ylabel="z")
            frame_index = Observable(1)

            for i ∈ length(run_info_dfns)
                animate_2d(run_info_dfns[i].vpa.grid, run_info_dfns[i].z.grid, f[i];
                           title="f", xlabel="wpa", ylabel="z", ax=ax[(i-1)*3+1],
                           colorbar_place=colorbar_place[(i-1)*3+1],
                           frame_index=frame_index, colorscale=log10,
                           transform=x->positive_or_nan(x; epsilon=1.e-16))

                animate_2d(run_info_dfns[i].vpa.grid, run_info_dfns[i].z.grid, f_M[i];
                           title="f_M", xlabel="wpa", ylabel="z", ax=ax[(i-1)*3+2],
                           colorbar_place=colorbar_place[(i-1)*3+2],
                           frame_index=frame_index, colorscale=log10,
                           transform=x->positive_or_nan(x; epsilon=1.e-16))

                animate_2d(run_info_dfns[i].vpa.grid, run_info_dfns[i].z.grid,
                           f[i].-f_M[i]; title="f - f_M", xlabel="wpa", ylabel="z",
                           ax=ax[(i-1)*3+3], colorbar_place=colorbar_place[(i-1)*3+3],
                           frame_index=frame_index, colorscale=log10,
                           transform=x->positive_or_nan(abs(x); epsilon=1.e-16))
            end

            save_animation(fig, frame_index, nt, plot_prefix * "Maxwellian_difference_log_vs_vpa_z.gif")

            iz0 = f_input.iz0
            fig, ax, legend_place = get_1d_ax(; yscale=log10, xlabel="wpa",
                                              get_legend_place=:below)
            frame_index = Observable(1)

            for i ∈ 1:length(run_info)
                animate_vs_vpa(run_info_dfns[i], "f"; data=f[i][:,iz0,:], label="f",
                               ax=ax, frame_index=frame_index,
                               transform=x->positive_or_nan(x; epsilon=1.e-16))
                animate_vs_vpa(run_info_dfns[i], "f_M"; data=f_M[i][:,iz0,:], label="f_M",
                               ax=ax, frame_index=frame_index,
                               transform=x->positive_or_nan(x; epsilon=1.e-16))
                animate_vs_vpa(run_info_dfns[i], "f - f_M"; label="f - f_M",
                               data=f[i][:,iz0,:].-f_M[i][:,iz0,:], ax=ax,
                               frame_index=frame_index,
                               transform=x->positive_or_nan(abs(x); epsilon=1.e-16))
            end
            ax.ylabel = "f"
            Legend(legend_place, ax; tellheight=true, tellwidth=false)

            save_animation(fig, frame_index, nt, plot_prefix * "Maxwellian_difference_log_vs_vpa.gif")
        end
    end
end
