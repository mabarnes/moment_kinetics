using CairoMakie
using LaTeXStrings
using MathTeXEngine

using moment_kinetics.makie_post_processing

function main()
    output_dir = "wall-bc"
    ext = ".png"

    CairoMakie.activate!(; px_per_unit=4)
    update_theme!(fontsize=24, fonts=(; regular=texfont(:text), bold=texfont(:bold),
                                      italic=texfont(:italic),
                                      bold_italic=texfont(:bolditalic)))

    #run_dir = "runs/wall-bc_recyclefraction0.5"
    #run_dir = "runs/wall-bc_recyclefraction0.5_split3"

    #run_dirs = ("runs/wall-bc_volumerecycle", "runs/wall-bc_volumerecycle_split1",
    #            "runs/wall-bc_volumerecycle_split2", "runs/wall-bc_volumerecycle_split3")
    run_dirs = ("../../runs/wall-bc_recyclefraction0.5",
                "../../runs/wall-bc_recyclefraction0.5_split1",
                "../../runs/wall-bc_recyclefraction0.5_split2",
                "../../runs/wall-bc_recyclefraction0.5_split3",
               )

    prefixes = ("full-f", "split1", "split2", "split3")
    short_labels = (L"full-f$$", L"evolving-$n$", L"evolving-$n,u_\parallel$",
                    L"evolving-$n,u_\parallel,p_\parallel$")
    #subtitles = (L"full-f$$", L"evolving-$n$", L"evolving-$n,u_\parallel$",
    #             L"evolving-$n,u_\parallel,p_\parallel$")

    run_info = Tuple(get_run_info(d; itime_min=-1, do_setup=false) for d ∈ run_dirs)
    run_info_dfns = Tuple(get_run_info(d; itime_min=-1, dfns=true, do_setup=false) for d ∈ run_dirs)
    setup_makie_post_processing_input!(
        joinpath(output_dir, "post_processing_input_eftc2023.toml"),
        run_info_moments=run_info, run_info_dfns=run_info_dfns)

    fig, axes = get_1d_ax(3; xlabel=L"z/L",
                          subtitles=(L"density$$", L"parallel flow$$", L"temperature$$"),
                          resolution=(1200, 400))
    for ((var_names, ylabel), ax) ∈ zip(((("density", "density_neutral"), L"n/n_\mathrm{ref}"),
                                         (("parallel_flow", "uz_neutral"), L"u_\parallel/v_\mathrm{ref}"),
                                         (("temperature", "temperature_neutral"), L"T/T_\mathrm{ref}")),
                                        axes)
        for (ri, label1, linestyle) ∈ zip(run_info, short_labels,
                                          (nothing, :dash, :dashdot, :dot))
            for (var_name, label2) ∈ zip(var_names, ("ion", "neutral"))
                if var_name == "temperature"
                    data = postproc_load_variable(ri, "thermal_speed", it=ri.nt, is=1, ir=1).^2
                elseif var_name == "temperature_neutral"
                    data = postproc_load_variable(ri, "thermal_speed_neutral", it=ri.nt, is=1, ir=1).^2
                else
                    data = nothing
                end
                plot_vs_z(ri, var_name; ax=ax, data=data, ylabel=ylabel,
                          label=LaTeXString(label1*" "*label2), linestyle=linestyle)
            end
        end
    end
    Legend(fig[2, 1], axes[1]; tellheight=true, tellwidth=false, orientation=:horizontal, nbanks=2)
    save(joinpath(output_dir, "moments" * ext), fig)

    #plot_vs_vpa_z(run_info_dfns, "f"; title=L"f_i", outfile=joinpath(output_dir, "f_ion$ext"), xlabel=L"v_\parallel", ylabel=L"z")
    #plot_vs_vz_z(run_info_dfns, "f_neutral"; title=L"f_n", outfile=joinpath(output_dir, "f_neutral$ext"), xlabel=L"v_\parallel", ylabel=L"z")

    ion_cbar_max = 3
    neutral_cbar_max = 8
    lims = (-12.0, 12.0, -0.5, 0.5)
    axis_args = Dict(:limits=>lims, :xgridvisible=>false, :ygridvisible=>false, :xticks=>-10:5:10)
    #plot_f_unnorm_vs_vpa_z(run_info_dfns; title=L"f_i", outfile=joinpath(output_dir, "f_ion$ext"), xlabel=L"v_\parallel", ylabel=L"z", rasterize=4.0, colorrange=(0, ion_cbar_max), subtitles=subtitles, axis_args=axis_args)
    #plot_f_unnorm_vs_vpa_z(run_info_dfns; neutral=true, title=L"f_n", outfile=joinpath(output_dir, "f_neutral$ext"), xlabel=L"v_\parallel", ylabel=L"z", rasterize=4.0, colorrange=(0, neutral_cbar_max), subtitles=subtitles, axis_args=axis_args)

    #plot_f_unnorm_vs_vpa_z(run_info_dfns; title=L"f_i", outfile=joinpath(output_dir, "logf_ion$ext"), transform=positive_or_nan, colorscale=log10, xlabel=L"v_\parallel", ylabel=L"z", rasterize=4.0, colorrange=(1e-16, ion_cbar_max), subtitles=subtitles, axis_args=axis_args)
    #plot_f_unnorm_vs_vpa_z(run_info_dfns; neutral=true, title=L"f_n", outfile=joinpath(output_dir, "logf_neutral$ext"), transform=positive_or_nan, colorscale=log10, xlabel=L"v_\parallel", ylabel=L"z", rasterize=4.0, colorrange=(1e-16, neutral_cbar_max), subtitles=subtitles, axis_args=axis_args)

    fig = Figure(resolution=(1200, 500))

    # Make column headings
    for (i, st) ∈ enumerate(short_labels)
        Label(fig[1, i], st, tellwidth=false)
    end

    axes_ion = [Axis(fig[2,i]; axis_args...) for i ∈ 1:4]
    axes_neutral = [Axis(fig[3,i]; axis_args...) for i ∈ 1:4]
    hm_ion = nothing
    hm_neutral = nothing
    for (ri, p, ax_ion, ax_neutral) ∈ zip(run_info_dfns, prefixes, axes_ion, axes_neutral)
        hm_ion = plot_f_unnorm_vs_vpa_z(ri; xlabel=L"v_\parallel/v_\mathrm{ref}", ylabel=L"z/L",
                                        rasterize=8.0, colorrange=(0, ion_cbar_max),
                                        ax=ax_ion, title="")
        hm_neutral = plot_f_unnorm_vs_vpa_z(ri; neutral=true, xlabel=L"v_\parallel/v_\mathrm{ref}",
                                            ylabel=L"z/L", rasterize=8.0,
                                            colorrange=(0, neutral_cbar_max),
                                            ax=ax_neutral, title="")

        #plot_f_unnorm_vs_vpa_z(ri;
        #                       outfile=joinpath(output_dir, "$(p)_logf_ion$ext"),
        #                       transform=positive_or_nan, colorscale=log10,
        #                       xlabel=L"v_\parallel", ylabel=L"z", rasterize=4.0,
        #                       colorrange=(1e-16, ion_cbar_max), title=st,
        #                       axis_args=axis_args)
        #plot_f_unnorm_vs_vpa_z(ri; neutral=true,
        #                       outfile=joinpath(output_dir, "$(p)_logf_neutral$ext"),
        #                       transform=positive_or_nan, colorscale=log10,
        #                       xlabel=L"v_\parallel", ylabel=L"z", rasterize=4.0,
        #                       colorrange=(1e-16, neutral_cbar_max), title=st,
        #                       axis_args=axis_args)
    end

    for ax ∈ axes_ion
        hidexdecorations!(ax)
    end
    for ax ∈ axes_ion[2:end]
        hideydecorations!(ax)
    end
    for ax ∈ axes_neutral[2:end]
        hideydecorations!(ax)
    end

    Label(fig[2,0], L"ion$$", rotation=π/2, tellheight=false)
    Label(fig[3,0], L"neutral$$", rotation=π/2, tellheight=false)

    Colorbar(fig[2, end+1], hm_ion)
    Colorbar(fig[3, end], hm_neutral)

    save(joinpath(output_dir, "pdfs$ext"), fig)
end

main()
