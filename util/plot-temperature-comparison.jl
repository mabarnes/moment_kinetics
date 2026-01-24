using makie_post_processing
using makie_post_processing: save_animation, plot_1d
using Makie
using moment_kinetics
using moment_kinetics.type_definitions
using moment_kinetics.StatsBase

function compare_temperature_profiles(paths...)
    ri = get_run_info(paths...)

    fig, ax, leg = get_1d_ax(; xlabel="z", ylabel="T", get_legend_place=:below)
    ind = Observable(1)

    # Choose colours explicitly using Cycled so that electron temperatures lines match ion
    # temperature lines.

    # Plot ion temperatures
    for (i, r) ∈ enumerate(ri)
        animate_vs_z(r, "temperature"; ax=ax, frame_index=ind, color=Cycled(i))
    end

    # Plot electron temperatures
    for (i, r) ∈ enumerate(ri)
        animate_vs_z(r, "electron_temperature"; ax=ax, frame_index=ind, color=Cycled(i), linestyle=:dash)
    end

    Legend(leg, ax; tellheight=true, tellwidth=false)

    if isa(ri, AbstractVector)
        nt = minimum(r.nt for r ∈ ri)
    else
        nt = ri.nt
    end
    save_animation(fig, ind, nt, "comparison_plots/compare_temperatures.gif")

    return nothing
end

function compare_sheath_speed(paths...)
    run_info = get_run_info(paths...)

    for (iz, label) ∈ ((1, "lower"), (-1, "upper"))
        nu_list = mk_float[]
        ion_upar_list = mk_float[]
        cs1_list = mk_float[]
        cs2_list = mk_float[]
        cs3_list = mk_float[]

        for ri ∈ run_info
            if ri.vperp.n != 1
                error("compare_sheath_speed() only supports 1V runs at the moment.")
            end
            nu_ei = get_variable(ri, "collision_frequency_ei"; it=ri.nt, is=1, ir=1)
            ion_upar = get_variable(ri, "parallel_flow";
                                    it=ri.nt, is=1, ir=1, iz=(iz == -1 ? ri.z.n : iz))[]
            Ti = get_variable(ri, "parallel_temperature";
                              it=ri.nt, is=1, ir=1, iz=(iz == -1 ? ri.z.n : iz))[]
            Te = get_variable(ri, "electron_parallel_temperature";
                              it=ri.nt, is=1, ir=1, iz=(iz == -1 ? ri.z.n : iz))[]

            mean_nu_ei = mean(nu_ei)
            println(mean_nu_ei, " ", ri.run_name)

            push!(nu_list, mean_nu_ei)
            push!(ion_upar_list, abs(ion_upar))
            push!(cs1_list, sqrt(Ti + Te))
            push!(cs2_list, sqrt(5/3*Ti + Te))
            push!(cs3_list, sqrt(3*Ti + Te))
        end

        fig, ax, leg = get_1d_ax(; xlabel="<ν_ei>", ylabel="u_∥", get_legend_place=:right)

        plot_1d(nu_list, ion_upar_list; ax=ax, label="|u_i∥|")
        plot_1d(nu_list, cs1_list; ax=ax, label="√(Ti+Te)")
        plot_1d(nu_list, cs2_list; ax=ax, label="√(5/3*Ti+Te)")
        plot_1d(nu_list, cs3_list; ax=ax, label="√(3*Ti+Te)")

        Legend(leg, ax)

        save("comparison_plots/compare_sheath_speed_$label.pdf", fig)

        ratio_fig, ratio_ax, ratio_leg = get_1d_ax(; xlabel="<ν_ei>", ylabel="u_∥ / c_s", get_legend_place=:right)

        plot_1d(nu_list, ion_upar_list ./ cs1_list; ax=ratio_ax, label="|u_i∥|/√(Ti+Te)")
        plot_1d(nu_list, ion_upar_list ./ cs2_list; ax=ratio_ax, label="|u_i∥|/√(5/3*Ti+Te)")
        plot_1d(nu_list, ion_upar_list ./ cs3_list; ax=ratio_ax, label="|u_i∥|/√(3*Ti+Te)")

        Legend(ratio_leg, ratio_ax)

        save("comparison_plots/compare_sheath_ratio_$label.pdf", ratio_fig)
    end

    return nothing
end
