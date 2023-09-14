using CairoMakie
using LaTeXStrings

using moment_kinetics.makie_post_processing

function main()
    output_dir = "publication_inputs/2023_EFTC_jto-poster/wall-bc/"
    ext = ".png"

    CairoMakie.activate!(; px_per_unit=4)

    #run_dir = "runs/wall-bc_recyclefraction0.5"
    #run_dir = "runs/wall-bc_recyclefraction0.5_split3"

    run_dirs = ("runs/wall-bc_volumerecycle", "runs/wall-bc_volumerecycle_split1",
                "runs/wall-bc_volumerecycle_split2", "runs/wall-bc_volumerecycle_split3")
    prefixes = ("full-f", "split1", "split2", "split3")
    short_labels = (L"full-f$$", L"n", L"n,u_\parallel", L"n,u_\parallel,p_\parallel")
    subtitles = (L"full-f$$", L"evolving-$n$", L"evolving-$n,u_\parallel$", L"evolving-$n,u_\parallel,p_\parallel$")

    run_info = Tuple(get_run_info(d; itime_min=-1) for d ∈ run_dirs)
    run_info_dfns = Tuple(get_run_info(d; itime_min=-1, dfns=true) for d ∈ run_dirs)
    setup_makie_post_processing_input!(
        joinpath(output_dir, "post_processing_input_eftc2023.toml"),
        run_info_moments=run_info, run_info_dfns=run_info_dfns)

    for (var_names, ylabel) ∈ ((("density", "density_neutral"), L"n"),
                               (("parallel_flow", "uz_neutral"), L"u_\parallel"),
                               (("temperature", "temperature_neutral"), L"T"))
        local fig, ax = get_1d_ax(xlabel=L"z", ylabel=ylabel)
        for (ri, label1, linestyle) ∈ zip(run_info,
                                          short_labels,
                                          (nothing, :dash, :dashdot, :dot))
            for (var_name, label2) ∈ zip(var_names, ("ion", "neutral"))
                if var_name == "temperature"
                    data = postproc_load_variable(ri, "thermal_speed", it=ri.nt, is=1, ir=1).^2
                elseif var_name == "temperature_neutral"
                    data = postproc_load_variable(ri, "thermal_speed_neutral", it=ri.nt, is=1, ir=1).^2
                else
                    data = nothing
                end
                plot_vs_z(ri, var_name; ax=ax, data=data,
                          label=LaTeXString(label1*" "*label2), linestyle=linestyle)
            end
        end
        put_legend_below(fig, ax; orientation=:horizontal, nbanks=2)
        save(joinpath(output_dir, var_names[1] * ext), fig)
    end

    #plot_vs_vpa_z(run_info_dfns, "f"; title=L"f_i", outfile=joinpath(output_dir, "f_ion$ext"), xlabel=L"v_\parallel", ylabel=L"z")
    #plot_vs_vz_z(run_info_dfns, "f_neutral"; title=L"f_n", outfile=joinpath(output_dir, "f_neutral$ext"), xlabel=L"v_\parallel", ylabel=L"z")

    ion_cbar_max = 3
    neutral_cbar_max = 5
    lims = (-10.0, 10.0, -0.5, 0.5)
    axis_args = Dict(:limits=>lims, :xgridvisible=>false, :ygridvisible=>false)
    #plot_f_unnorm_vs_vpa_z(run_info_dfns; title=L"f_i", outfile=joinpath(output_dir, "f_ion$ext"), xlabel=L"v_\parallel", ylabel=L"z", rasterize=4.0, colorrange=(0, ion_cbar_max), subtitles=subtitles, axis_args=axis_args)
    #plot_f_unnorm_vs_vpa_z(run_info_dfns; neutral=true, title=L"f_n", outfile=joinpath(output_dir, "f_neutral$ext"), xlabel=L"v_\parallel", ylabel=L"z", rasterize=4.0, colorrange=(0, neutral_cbar_max), subtitles=subtitles, axis_args=axis_args)

    #plot_f_unnorm_vs_vpa_z(run_info_dfns; title=L"f_i", outfile=joinpath(output_dir, "logf_ion$ext"), transform=positive_or_nan, colorscale=log10, xlabel=L"v_\parallel", ylabel=L"z", rasterize=4.0, colorrange=(1e-16, ion_cbar_max), subtitles=subtitles, axis_args=axis_args)
    #plot_f_unnorm_vs_vpa_z(run_info_dfns; neutral=true, title=L"f_n", outfile=joinpath(output_dir, "logf_neutral$ext"), transform=positive_or_nan, colorscale=log10, xlabel=L"v_\parallel", ylabel=L"z", rasterize=4.0, colorrange=(1e-16, neutral_cbar_max), subtitles=subtitles, axis_args=axis_args)
    for (ri, p, st) ∈ zip(run_info_dfns, prefixes, subtitles)
        plot_f_unnorm_vs_vpa_z(ri;
                               outfile=joinpath(output_dir, "$(p)_f_ion$ext"),
                               xlabel=L"v_\parallel", ylabel=L"z", rasterize=4.0,
                               colorrange=(0, ion_cbar_max), title=st,
                               axis_args=axis_args)
        plot_f_unnorm_vs_vpa_z(ri; neutral=true,
                               outfile=joinpath(output_dir, "$(p)_f_neutral$ext"),
                               xlabel=L"v_\parallel", ylabel=L"z", rasterize=4.0,
                               colorrange=(0, neutral_cbar_max), title=st,
                               axis_args=axis_args)

        plot_f_unnorm_vs_vpa_z(ri;
                               outfile=joinpath(output_dir, "$(p)_logf_ion$ext"),
                               transform=positive_or_nan, colorscale=log10,
                               xlabel=L"v_\parallel", ylabel=L"z", rasterize=4.0,
                               colorrange=(1e-16, ion_cbar_max), title=st,
                               axis_args=axis_args)
        plot_f_unnorm_vs_vpa_z(ri; neutral=true,
                               outfile=joinpath(output_dir, "$(p)_logf_neutral$ext"),
                               transform=positive_or_nan, colorscale=log10,
                               xlabel=L"v_\parallel", ylabel=L"z", rasterize=4.0,
                               colorrange=(1e-16, neutral_cbar_max), title=st,
                               axis_args=axis_args)
    end
end

main()
