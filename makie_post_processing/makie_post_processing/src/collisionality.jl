"""
A function to plot collisionalities. The mean free path is plotted (or animated) 
along with the lengthscales of the gradients of density, parallel flow and temperature.

There are also functions to check the calculations of the mean free path and the 
comparison of temperature, L_T and dT_dz. They would only be for making sure
lengthscales and mean free path calculations are sensible.
"""
function collisionality_plots(run_info, plot_prefix=nothing)
    if !isa(run_info, Tuple)
        run_info = (run_info,)
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
    end
end
