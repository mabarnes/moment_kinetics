# Manufactured solutions analysis
#################################

using moment_kinetics.manufactured_solns: manufactured_solutions,
                                          manufactured_electric_fields

"""
     manufactured_solutions_get_field_and_field_sym(run_info, variable_name;
         it=nothing, ir=nothing, iz=nothing, ivperp=nothing, ivpa=nothing, ivzeta=nothing,
         ivr=nothing, ivz=nothing)

Get the data `variable` for `variable_name` from the output, and calculate the
manufactured solution `variable_sym`.

The information for the runs to analyse and plot is passed in `run_info` (as returned by
[`get_run_info`](@ref)).

`it`, `ir`, `iz`, `ivperp`, `ivpa`, `ivzeta`, `ivr`, `ivz` can be used to select a subset
of the grid by passing an integer or range for any dimension.

Returns `variable`, `variable_sym`.
"""
function manufactured_solutions_get_field_and_field_sym(run_info, variable_name;
        it=nothing, ir=nothing, iz=nothing, ivperp=nothing, ivpa=nothing, ivzeta=nothing,
        ivr=nothing, ivz=nothing, nvperp)

    variable_name = Symbol(variable_name)

    func_name_lookup = (phi=:phi_func, Er=:Er_func, Ez=:Ez_func, density=:densi_func,
                        parallel_flow=:upari_func, pressure=:pi_func,
                        density_neutral=:densn_func, f=:dfni_func, f_neutral=:dfnn_func)

    nt = run_info.nt
    nr = run_info.r.n
    nz = run_info.z.n
    if it === nothing
        it = 1:nt
    end
    if ir === nothing
        ir = 1:nr
    end
    if iz === nothing
        iz = 1:nz
    end
    tinds = run_info.itime_min:run_info.itime_skip:run_info.itime_max
    tinds = tinds[it]

    if nr > 1
        Lr_in = run_info.r.L
    else
        Lr_in = 1.0
    end

    if variable_name ∈ (:phi, :Er, :Ez)
        manufactured_funcs =
            manufactured_electric_fields(Lr_in, run_info.z.L, run_info.r.bc,
                                         run_info.z.bc, run_info.composition,
                                         run_info.r.n, run_info.manufactured_solns_input,
                                         run_info.species)
    elseif variable_name ∈ (:density, :parallel_flow, :pressure,
                            :density_neutral, :f, :f_neutral)
        manufactured_funcs =
            manufactured_solutions(run_info.manufactured_solns_input, Lr_in, run_info.z.L,
                                   run_info.r.bc, run_info.z.bc, run_info.geometry.input,
                                   run_info.composition, run_info.species, run_info.r.n,
                                   nvperp)
    end

    variable_func = manufactured_funcs[func_name_lookup[variable_name]]

    variable = get_variable(run_info, String(variable_name); it=tinds, is=1, ir=ir, iz=iz,
                            ivperp=ivperp, ivpa=ivpa, ivzeta=ivzeta, ivr=ivr, ivz=ivz)
    variable_sym = similar(variable)

    time = run_info.time
    r_grid = run_info.r.grid
    z_grid = run_info.z.grid

    if variable_name == :f
        vperp_grid = run_info.vperp.grid
        vpa_grid = run_info.vpa.grid
        nvperp = run_info.vperp.n
        nvpa = run_info.vpa.n
        if ivperp === nothing
            ivperp = 1:nvperp
        end
        if ivpa === nothing
            ivpa = 1:nvpa
        end
        counter = 1
        for iit ∈ it, iir ∈ ir, iiz ∈ iz, iivperp ∈ ivperp, iivpa ∈ ivpa
            variable_sym[counter] =
                variable_func(vpa_grid[iivpa], vperp_grid[iivperp], z_grid[iiz],
                              r_grid[iir], time[iit])
            counter += 1
        end
    elseif variable_name == :f_neutral
        vzeta_grid = run_info.vzeta.grid
        vr_grid = run_info.vr.grid
        vz_grid = run_info.vz.grid
        nvzeta = run_info.vzeta.n
        nvr = run_info.vr.n
        nvz = run_info.vz.n
        if ivzeta === nothing
            ivzeta = 1:nvzeta
        end
        if ivr === nothing
            ivr = 1:nvr
        end
        if ivz === nothing
            ivz = 1:nvz
        end
        counter = 1
        for iit ∈ it, iir ∈ ir, iiz ∈ iz, iivzeta ∈ ivzeta, iivr ∈ ivr, iivz ∈ ivz
            variable_sym[counter] =
            variable_func(vz_grid[iivz], vr_grid[iivr], vzeta_grid[iivzeta], z_grid[iiz],
                          r_grid[iir], time[iit])
            counter += 1
        end
    else
        counter = 1
        for iit ∈ it, iir ∈ ir, iiz ∈ iz
            variable_sym[counter] = variable_func(z_grid[iiz], r_grid[iir], time[iit])
            counter += 1
        end
    end

    return variable, variable_sym
end

"""
    compare_moment_symbolic_test(run_info, plot_prefix, field_label, field_sym_label,
                                 norm_label, variable_name; io=nothing)

Compare the computed and manufactured solutions for a field or moment variable
`variable_name`.

The information for the run to analyse is passed in `run_info` (as returned by
[`get_run_info`](@ref)).

`plot_prefix` is required and gives the path and prefix for plots to be saved to. They
will be saved with the format `plot_prefix<some_identifying_string>.pdf` for plots and
`plot_prefix<some_identifying_string>.gif`, etc. for animations.

`field_label` is the label that will be used for the name of the computed variable in
plots, `field_sym_label` is the label for the manufactured solution, and `norm_label` is
the label for the error (the difference between the computed and manufactured solutions).

If `io` is passed then error norms will be written to that file.
"""
function compare_moment_symbolic_test(run_info, plot_prefix, field_label, field_sym_label,
                                      norm_label, variable_name; io=nothing,
                                      input=nothing, nvperp)

    println("Doing MMS analysis and making plots for $variable_name")
    flush(stdout)

    if input === nothing
        input = Dict_to_NamedTuple(input_dict["manufactured_solns"])
    end

    field, field_sym =
        manufactured_solutions_get_field_and_field_sym(run_info, variable_name; nvperp=nvperp)
    error = field .- field_sym

    nt = run_info.nt
    time = run_info.time
    r = run_info.r
    z = run_info.z

    if !input.calculate_error_norms
        field_norm = nothing
    else
        field_norm = zeros(mk_float,nt)
        for it in 1:nt
            dummy = 0.0
            #dummy_N = 0.0
            for ir in 1:r.n
                for iz in 1:z.n
                    dummy += (field[iz,ir,it] - field_sym[iz,ir,it])^2
                    #dummy_N +=  (field_sym[iz,ir,it])^2
                end
            end
            #field_norm[it] = dummy/dummy_N
            field_norm[it] = sqrt(dummy/(r.n*z.n))
        end
        println_to_stdout_and_file(io, join(field_norm, " "), " # ", variable_name)
        plot_vs_t(run_info, norm_label, input=input, data=field_norm,
                  outfile=plot_prefix*variable_name*"_norm_vs_t.pdf")
    end

    has_rdim = (r.n > 1)
    has_zdim = (z.n > 1)

    if has_rdim && input.wall_plots
        # plot last (by default) timestep field vs r at z_wall

        fig, ax, legend_place = get_1d_ax(2; get_legend_place=:below)
        plot_1d(r.grid, select_slice(field, :r; input=input, iz=1), xlabel=L"r",
                ylabel=field_label, label=field_label, ax=ax[1])
        plot_1d(r.grid, select_slice(field_sym, :r; input=input, iz=1),
                label=field_sym_label, ax=ax[1])
        Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false,
               orientation=:horizontal)
        plot_1d(r.grid, select_slice(error, :r; input=input, iz=1), xlabel=L"r",
                ylabel=norm_label, ax=ax[2])
        outfile = plot_prefix * "MMS_" * variable_name * "(z_wall-)_vs_r.pdf"
        save(outfile, fig)

        fig, ax, legend_place = get_1d_ax(2; get_legend_place=:below)
        plot_1d(r.grid, select_slice(field, :r; input=input, iz=z.n), xlabel=L"r",
                ylabel=field_label, label=field_label, ax=ax[1])
        plot_1d(r.grid, select_slice(field_sym, :r; input=input, iz=z.n),
                label=field_sym_label, ax=ax[1])
        Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false,
               orientation=:horizontal)
        plot_1d(r.grid, select_slice(error, :r; input=input, iz=z.n), xlabel=L"r",
                ylabel=norm_label, ax=ax[2])
        outfile = plot_prefix * "MMS_" * variable_name * "(z_wall+)_vs_r.pdf"
        save(outfile, fig)
    end

    if input.plot_vs_t
        fig, ax, legend_place = get_1d_ax(2; get_legend_place=:below)
        plot_1d(time, select_slice(field, :t; input=input), xlabel=L"t",
                ylabel=field_label, label=field_label, ax=ax[1])
        plot_1d(time, select_slice(field_sym, :t; input=input), label=field_sym_label,
                ax=ax[1])
        Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false,
               orientation=:horizontal)
        plot_1d(time, select_slice(error, :t; input=input), xlabel=L"t",
                ylabel=norm_label, ax=ax[2])
        outfile = plot_prefix * "MMS_" * variable_name * "_vs_t.pdf"
        save(outfile, fig)
    end
    if has_rdim && input.plot_vs_r
        fig, ax, legend_place = get_1d_ax(2; get_legend_place=:below)
        plot_1d(r.grid, select_slice(field, :r; input=input), xlabel=L"r",
                ylabel=field_label, label=field_label, ax=ax[1])
        plot_1d(r.grid, select_slice(field_sym, :r; input=input), label=field_sym_label,
                ax=ax[1])
        Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false,
               orientation=:horizontal)
        plot_1d(r.grid, select_slice(error, :r; input=input), xlabel=L"r",
                ylabel=norm_label, ax=ax[2])
        outfile = plot_prefix * "MMS_" * variable_name * "_vs_r.pdf"
        save(outfile, fig)
    end
    if has_zdim && input.plot_vs_z
        fig, ax, legend_place = get_1d_ax(2; get_legend_place=:below)
        plot_1d(z.grid, select_slice(field, :z; input=input), xlabel=L"z",
                ylabel=field_label, label=field_label, ax=ax[1])
        plot_1d(z.grid, select_slice(field_sym, :z; input=input), label=field_sym_label,
                ax=ax[1])
        Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false,
               orientation=:horizontal)
        plot_1d(z.grid, select_slice(error, :z; input=input), xlabel=L"z",
                ylabel=norm_label, ax=ax[2])
        outfile = plot_prefix * "MMS_" * variable_name * "_vs_z.pdf"
        save(outfile, fig)
    end
    if has_rdim && input.plot_vs_r_t
        fig, ax, colorbar_place = get_2d_ax(3)
        plot_2d(r.grid, time, select_slice(field, :t, :r; input=input), title=field_label,
                xlabel=L"r", ylabel=L"t", ax=ax[1], colorbar_place=colorbar_place[1])
        plot_2d(r.grid, time, select_slice(field_sym, :t, :r; input=input),
                title=field_sym_label, xlabel=L"r", ylabel=L"t", ax=ax[2],
                colorbar_place=colorbar_place[2])
        plot_2d(r.grid, time, select_slice(error, :t, :r; input=input), title=norm_label,
                xlabel=L"r", ylabel=L"t", ax=ax[3], colorbar_place=colorbar_place[3])
        outfile = plot_prefix * "MMS_" * variable_name * "_vs_r_t.pdf"
        save(outfile, fig)
    end
    if has_zdim && input.plot_vs_z_t
        fig, ax, colorbar_place = get_2d_ax(3)
        plot_2d(z.grid, time, select_slice(field, :t, :z; input=input), title=field_label,
                xlabel=L"z", ylabel=L"t", ax=ax[1], colorbar_place=colorbar_place[1])
        plot_2d(z.grid, time, select_slice(field_sym, :t, :z; input=input),
                title=field_sym_label, xlabel=L"z", ylabel=L"t", ax=ax[2],
                colorbar_place=colorbar_place[2])
        plot_2d(z.grid, time, select_slice(error, :t, :z; input=input), title=norm_label,
                xlabel=L"z", ylabel=L"t", ax=ax[3], colorbar_place=colorbar_place[3])
        outfile = plot_prefix * "MMS_" * variable_name * "_vs_z_t.pdf"
        save(outfile, fig)
    end
    if has_rdim && has_zdim && input.plot_vs_z_r
        fig, ax, colorbar_place = get_2d_ax(3)
        plot_2d(z.grid, r.grid, select_slice(field, :r, :z; input=input),
                title=field_label, xlabel=L"z", ylabel=L"r", ax=ax[1],
                colorbar_place=colorbar_place[1])
        plot_2d(z.grid, r.grid, select_slice(field_sym, :r, :z; input=input),
                title=field_sym_label, xlabel=L"z", ylabel=L"r", ax=ax[2],
                colorbar_place=colorbar_place[2])
        plot_2d(z.grid, r.grid, select_slice(error, :r, :z; input=input),
                title=norm_label, xlabel=L"z", ylabel=L"r", ax=ax[3],
                colorbar_place=colorbar_place[3])
        outfile = plot_prefix * "MMS_" * variable_name * "_vs_z_r.pdf"
        save(outfile, fig)
    end
    if has_rdim && input.animate_vs_r
        fig, ax, legend_place = get_1d_ax(2; get_legend_place=:below)
        frame_index = Observable(1)
        animate_1d(r.grid, select_slice(field, :t, :r; input=input),
                   frame_index=frame_index, xlabel="r", ylabel=field_label,
                   label=field_label, ax=ax[1])
        animate_1d(r.grid, select_slice(field_sym, :t, :r; input=input),
                   frame_index=frame_index, label=field_sym_label, ax=ax[1])
        Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false,
               orientation=:horizontal)
        animate_1d(r.grid, select_slice(error, :t, :r; input=input),
                   frame_index=frame_index, xlabel="r", ylabel=norm_label, ax=ax[2])
        outfile = plot_prefix * "MMS_" * variable_name * "_vs_r." * input.animation_ext
        save_animation(fig, frame_index, nt, outfile)
    end
    if has_zdim && input.animate_vs_z
        fig, ax, legend_place = get_1d_ax(2; get_legend_place=:below)
        frame_index = Observable(1)
        animate_1d(z.grid, select_slice(field, :t, :z; input=input),
                   frame_index=frame_index, xlabel="z", ylabel=field_label,
                   label=field_label, ax=ax[1])
        animate_1d(z.grid, select_slice(field_sym, :t, :z; input=input),
                   frame_index=frame_index, label=field_sym_label, ax=ax[1])
        Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false,
               orientation=:horizontal)
        animate_1d(z.grid, select_slice(error, :t, :z; input=input),
                   frame_index=frame_index, xlabel="z", ylabel=norm_label, ax=ax[2])
        outfile = plot_prefix * "MMS_" * variable_name * "_vs_z." * input.animation_ext
        save_animation(fig, frame_index, nt, outfile)
    end
    if has_rdim && has_zdim && input.animate_vs_z_r
        fig, ax, colorbar_place = get_2d_ax(3)
        frame_index = Observable(1)
        animate_2d(z.grid, r.grid, select_slice(field, :t, :r, :z; input=input),
                   frame_index=frame_index, title=field_label, xlabel=L"z", ylabel=L"y",
                   ax=ax[1], colorbar_place=colorbar_place[1])
        animate_2d(z.grid, r.grid, select_slice(field_sym, :t, :r, :z; input=input),
                   frame_index=frame_index, title=field_sym_label, xlabel=L"z",
                   ylabel=L"y", ax=ax[2], colorbar_place=colorbar_place[2])
        animate_2d(z.grid, r.grid, select_slice(error, :t, :r, :z; input=input),
                   frame_index=frame_index, title=norm_label, xlabel=L"z", ylabel=L"y",
                   ax=ax[3], colorbar_place=colorbar_place[3])
        outfile = plot_prefix * "MMS_" * variable_name * "_vs_z_r." * input.animation_ext
        save_animation(fig, frame_index, nt, outfile)
    end

    return field_norm
end

"""
    _MMS_pdf_plots(run_info, input, variable_name, plot_prefix, field_label,
                   field_sym_label, norm_label, plot_dims, animate_dims)

Utility function for making plots to avoid duplicated code in
[`compare_ion_pdf_symbolic_test`](@ref) and
[`compare_neutral_pdf_symbolic_test`](@ref).

The information for the run to analyse is passed in `run_info` (as returned by
[`get_run_info`](@ref)).

`input` is a NamedTuple of settings to use.

`variable_name` is the name of the variable being plotted.

`plot_prefix` gives the path and prefix for plots to be saved to. They will be saved with
the format `plot_prefix<some_identifying_string>.pdf` for plots and
`plot_prefix<some_identifying_string>.gif`, etc. for animations.

`field_label` is the label for the computed variable that will be used in
plots/animations, `field_sym_label` is the label for the manufactured solution, and
`norm_label` is the label for the error.

`plot_dims` are the dimensions of the variable, and `animate_dims` are the same but
omitting `:t`.
"""
function _MMS_pdf_plots(run_info, input, variable_name, plot_prefix, field_label,
                        field_sym_label, norm_label, plot_dims, animate_dims, neutrals)

    nt = run_info.nt
    time = run_info.time

    if neutrals
        all_dims_no_t = (:r, :z, :vzeta, :vr, :vz)
    else
        all_dims_no_t = (:r, :z, :vperp, :vpa)
    end
    all_dims = tuple(:t, all_dims_no_t...)
    all_plot_slices = Tuple(Symbol(:i, d)=>input[Symbol(:i, d, :0)] for d ∈ all_dims)
    all_animate_slices = Tuple(Symbol(:i, d)=>input[Symbol(:i, d, :0)] for d ∈ all_dims_no_t)

    # Options to produce either regular or log-scale plots
    epsilon = 1.0e-30 # minimum data value to include in log plots
    for (log, yscale, transform, error_transform) ∈
            (("", nothing, identity, identity),
             (:_log, log10, x->positive_or_nan(x; epsilon=1.e-30), x->positive_or_nan.(abs.(x); epsilon=1.e-30)))
        for dim ∈ plot_dims
            if input[Symbol(:plot, log, :_vs_, dim)]
                coord = dim === :t ? time : run_info[dim].grid

                slices = (k=>v for (k, v) ∈ all_plot_slices if k != Symbol(:i, dim))
                f, f_sym =
                    manufactured_solutions_get_field_and_field_sym(
                        run_info, variable_name; nvperp=run_info.vperp.n, slices...)
                error = f .- f_sym

                fig, ax, legend_place = get_1d_ax(2; yscale=yscale, get_legend_place=:below)
                plot_1d(coord, f, xlabel=L"%$dim", ylabel=field_label, label=field_label,
                        ax=ax[1], transform=transform)
                plot_1d(coord, f_sym, label=field_sym_label, ax=ax[1],
                        transform=transform)
                Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false,
                       orientation=:horizontal)
                plot_1d(coord, error, xlabel=L"%$dim", ylabel=norm_label, ax=ax[2],
                        transform=error_transform)
                outfile = plot_prefix * "MMS" * String(log) * "_" * variable_name * "_vs_$dim.pdf"
                save(outfile, fig)
            end
        end
        for (dim1, dim2) ∈ combinations(plot_dims, 2)
            if input[Symbol(:plot, log, :_vs_, dim2, :_, dim1)]
                coord1 = dim1 === :t ? time : run_info[dim1].grid
                coord2 = dim2 === :t ? time : run_info[dim2].grid

                slices = (k=>v for (k, v) ∈ all_plot_slices
                          if k ∉ (Symbol(:i, dim1), Symbol(:i, dim2)))
                f, f_sym =
                manufactured_solutions_get_field_and_field_sym(
                    run_info, variable_name; nvperp=run_info.vperp.n, slices...)
                error = f .- f_sym

                fig, ax, colorbar_place = get_2d_ax(3)
                plot_2d(coord2, coord1, f, title=field_label, xlabel=L"%$dim2",
                        ylabel=L"%$dim1", ax=ax[1], colorbar_place=colorbar_place[1],
                        colorscale=yscale, transform=transform)
                plot_2d(coord2, coord1, f_sym, title=field_sym_label, xlabel=L"%$dim2",
                        ylabel=L"%$dim1", ax=ax[2], colorbar_place=colorbar_place[2],
                        colorscale=yscale, transform=transform)
                plot_2d(coord2, coord1, error, title=norm_label, xlabel=L"%$dim2",
                        ylabel=L"%$dim1", ax=ax[3], colorbar_place=colorbar_place[3],
                        colorscale=yscale, transform=error_transform)
                outfile = plot_prefix * "MMS" * String(log) * "_" * variable_name * "_vs_$(dim2)_$(dim1).pdf"
                save(outfile, fig)
            end
        end
        for dim ∈ animate_dims
            if input[Symbol(:animate, log, :_vs_, dim)]
                coord = dim === :t ? time : run_info[dim].grid

                slices = (k=>v for (k, v) ∈ all_animate_slices if k != Symbol(:i, dim))
                f, f_sym =
                    manufactured_solutions_get_field_and_field_sym(
                        run_info, variable_name; nvperp=run_info.vperp.n, slices...)
                error = f .- f_sym

                fig, ax, legend_place = get_1d_ax(2; yscale=yscale, get_legend_place=:below)
                frame_index = Observable(1)
                animate_1d(coord, f, frame_index=frame_index, xlabel=L"%$dim",
                           ylabel=field_label, label=field_label, ax=ax[1],
                           transform=transform)
                animate_1d(coord, f_sym, frame_index=frame_index, label=field_sym_label,
                           ax=ax[1], transform=transform)
                Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false,
                       orientation=:horizontal)
                animate_1d(coord, error, frame_index=frame_index, xlabel=L"%$dim",
                           ylabel=norm_label, label=field_label, ax=ax[2],
                           transform=error_transform)
                outfile = plot_prefix * "MMS" * String(log) * "_" * variable_name * "_vs_$dim." * input.animation_ext
                save_animation(fig, frame_index, nt, outfile)
            end
        end
        for (dim1, dim2) ∈ combinations(animate_dims, 2)
            if input[Symbol(:animate, log, :_vs_, dim2, :_, dim1)]
                coord1 = dim1 === :t ? time : run_info[dim1].grid
                coord2 = dim2 === :t ? time : run_info[dim2].grid

                slices = (k=>v for (k, v) ∈ all_animate_slices
                          if k ∉ (Symbol(:i, dim1), Symbol(:i, dim2)))
                f, f_sym =
                manufactured_solutions_get_field_and_field_sym(
                    run_info, variable_name; nvperp=run_info.vperp.n, slices...)
                error = f .- f_sym

                fig, ax, colorbar_place = get_2d_ax(3)
                frame_index = Observable(1)
                animate_2d(coord2, coord1, f, frame_index=frame_index, xlabel=L"%$dim2",
                           ylabel=L"%$dim1", title=field_label, ax=ax[1],
                           colorbar_place=colorbar_place[1], colorscale=yscale,
                           transform=transform)
                animate_2d(coord2, coord1, f_sym, frame_index=frame_index,
                           xlabel=L"%$dim2", ylabel=L"%$dim1", title=field_sym_label,
                           ax=ax[2], colorbar_place=colorbar_place[2], colorscale=yscale,
                           transform=transform)
                animate_2d(coord2, coord1, error, frame_index=frame_index,
                           xlabel=L"%$dim2", ylabel=L"%$dim1", title=norm_label,
                           ax=ax[3], colorbar_place=colorbar_place[3], colorscale=yscale,
                           transform=error_transform)
                outfile = plot_prefix * "MMS" * String(log) * "_" * variable_name * "_vs_$(dim2)_$(dim1)." * input.animation_ext
                save_animation(fig, frame_index, nt, outfile)
            end
        end
    end
end

"""
    compare_ion_pdf_symbolic_test(run_info, plot_prefix; io=nothing,
                                      input=nothing)

Compare the computed and manufactured solutions for the ion distribution function.

The information for the run to analyse is passed in `run_info` (as returned by
[`get_run_info`](@ref)).

`plot_prefix` is required and gives the path and prefix for plots to be saved to. They
will be saved with the format `plot_prefix<some_identifying_string>.pdf` for plots and
`plot_prefix<some_identifying_string>.gif`, etc. for animations.

If `io` is passed then error norms will be written to that file.

`input` is a NamedTuple of settings to use. If not given it will be read from the
`[manufactured_solns]` section of [`input_dict_dfns`][@ref].

Note: when calculating error norms, data is loaded only for 1 time point and for an r-z
chunk that is the same size as computed by 1 block of the simulation at run time. This
should prevent excessive memory requirements for this function.
"""
function compare_ion_pdf_symbolic_test(run_info, plot_prefix; io=nothing,
                                           input=nothing)

    field_label = L"\tilde{f}_i"
    field_sym_label = L"\tilde{f}_i^{sym}"
    norm_label = L"\varepsilon(\tilde{f}_i)"
    variable_name = "f"

    println("Doing MMS analysis and making plots for $variable_name")
    flush(stdout)

    if input === nothing
        input = Dict_to_NamedTuple(input_dict_dfns["manufactured_solns"])
    end

    nt = run_info.nt
    r = run_info.r
    z = run_info.z
    vperp = run_info.vperp
    vpa = run_info.vpa

    if !input.calculate_error_norms
        field_norm = nothing
    else
        # Load data in chunks, with the same size as the chunks that were saved during the
        # run, to avoid running out of memory
        r_chunks = UnitRange{mk_int}[]
        chunk = run_info.r_chunk_size
        nchunks = (r.n ÷ chunk)
        if nchunks == 1
            r_chunks = [1:r.n]
        else
            for i ∈ 1:nchunks
                if i == nchunks
                    push!(r_chunks, (i-1)*chunk+1:i*chunk+1)
                else
                    push!(r_chunks, (i-1)*chunk+1:i*chunk)
                end
            end
        end
        z_chunks = UnitRange{mk_int}[]
        chunk = run_info.z_chunk_size
        nchunks = (z.n ÷ chunk)
        if nchunks == 1
            z_chunks = [1:z.n]
        else
            for i ∈ 1:nchunks
                if i == nchunks
                    push!(z_chunks, (i-1)*chunk+1:i*chunk+1)
                else
                    push!(z_chunks, (i-1)*chunk+1:i*chunk)
                end
            end
        end
        field_norm = zeros(mk_float,nt)
        for it in 1:nt
            dummy = 0.0
            #dummy_N = 0.0
            for r_chunk ∈ r_chunks, z_chunk ∈ z_chunks
                f, f_sym =
                    manufactured_solutions_get_field_and_field_sym(
                        run_info, variable_name; nvperp=run_info.vperp.n, it=it,
                        ir=r_chunk, iz=z_chunk)
                dummy += sum(@. (f - f_sym)^2)
                #dummy_N += sum(f_sym.^2)
            end

            #field_norm[it] = dummy/dummy_N
            field_norm[it] = sqrt(dummy/(r.n*z.n*vperp.n*vpa.n))
        end
        println_to_stdout_and_file(io, join(field_norm, " "), " # ", variable_name)
        plot_vs_t(run_info, norm_label, input=input, data=field_norm,
                  outfile=plot_prefix*"f_norm_vs_t.pdf")
    end

    has_rdim = (r.n > 1)
    has_zdim = (z.n > 1)
    is_1V = (vperp.n == 1)

    if input.wall_plots
        for (iz, z_label) ∈ ((1, "wall-"), (z.n, "wall+"))
            f, f_sym =
                manufactured_solutions_get_field_and_field_sym(
                    run_info, variable_name; nvperp=run_info.vperp.n, it=input.it0,
                    ir=input.ir0, iz=iz, ivperp=input.ivperp0)
            error = f .- f_sym

            fig, ax, legend_place = get_1d_ax(2; get_legend_place=:below)
            plot_1d(vpa.grid, f, ax=ax[1], label="num",
                    xlabel=L"v_{\parallel}/L_{v_{\parallel}}", ylabel=field_label)
            plot_1d(vpa.grid, f_sym, ax=ax[1], label="sym")
            Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false,
                   orientation=:horizontal)

            plot_1d(vpa.grid, error, ax=ax[2], xlabel=L"v_{\parallel}/L_{v_{\parallel}}",
                    ylabel=norm_label)

            outfile = plot_prefix * variable_name * "(" * z_label * ")_vs_vpa.pdf"
            save(outfile, fig)

            if has_rdim
                f, f_sym =
                manufactured_solutions_get_field_and_field_sym(
                    run_info, variable_name; nvperp=run_info.vperp.n, it=input.it0, iz=iz,
                    ivperp=input.ivperp0)
                error = f .- f_sym

                fig, ax, colorbar_place = get_2d_ax(3)
                plot_2d(vpa.grid, r.grid, f, ax=ax[1], colorbar_place=colorbar_place[1],
                        title=field_label, xlabel=L"v_{\parallel}/L_{v_{\parallel}}",
                        ylabel=L"r")
                plot_2d(vpa.grid, r.grid, f_sym, ax=ax[2],
                        colorbar_place=colorbar_place[2], title=field_sym_label,
                        xlabel=L"v_{\parallel}/L_{v_{\parallel}}", ylabel=L"r")
                plot_2d(vpa.grid, r.grid, error, ax=ax[3],
                        colorbar_place=colorbar_place[3], title=norm_label,
                        xlabel=L"v_{\parallel}/L_{v_{\parallel}}", ylabel=L"r")

                outfile = plot_prefix * variable_name * "(" * z_label * ")_vs_vpa_r.pdf"
                save(outfile, fig)
            end

            if !is_1V
                f, f_sym =
                manufactured_solutions_get_field_and_field_sym(
                    run_info, variable_name; nvperp=run_info.vperp.n, it=input.it0, iz=iz,
                    ir=input.ir0)
                error = f .- f_sym

                fig, ax, colorbar_place = get_2d_ax(3)
                plot_2d(vpa.grid, vperp.grid, f, ax=ax[1],
                        colorbar_place=colorbar_place[1], title=field_label,
                        xlabel=L"v_{\parallel}/L_{v_{\parallel}}",
                        ylabel=L"v_{\perp}/L_{v_{\perp}}")
                plot_2d(vpa.grid, vperp.grid, f_sym, ax=ax[2],
                        colorbar_place=colorbar_place[2], title=field_sym_label,
                        xlabel=L"v_{\parallel}/L_{v_{\parallel}}",
                        ylabel=L"v_{\perp}/L_{v_{\perp}}")
                plot_2d(vpa.grid, vperp.grid, error, ax=ax[3],
                        colorbar_place=colorbar_place[3], title=norm_label,
                        xlabel=L"v_{\parallel}/L_{v_{\parallel}}",
                        ylabel=L"v_{\perp}/L_{v_{\perp}}")

                outfile = plot_prefix * variable_name * "(" * z_label * ")_vs_vpa_vperp.pdf"
                save(outfile, fig)
            end
        end
    end

    animate_dims = setdiff(ion_dimensions, (:s,))
    if !has_rdim
        animate_dims = setdiff(animate_dims, (:r,))
    end
    if !has_zdim
        animate_dims = setdiff(animate_dims, (:z,))
    end
    if is_1V
        animate_dims = setdiff(animate_dims, (:vperp,))
    end
    plot_dims = tuple(:t, animate_dims...)
    _MMS_pdf_plots(run_info, input, variable_name, plot_prefix, field_label,
                   field_sym_label, norm_label, plot_dims, animate_dims, false)

    return field_norm
end

"""
    compare_neutral_pdf_symbolic_test(run_info, plot_prefix; io=nothing,
                                      input=nothing)

Compare the computed and manufactured solutions for the neutral distribution function.

The information for the run to analyse is passed in `run_info` (as returned by
[`get_run_info`](@ref)).

`plot_prefix` is required and gives the path and prefix for plots to be saved to. They
will be saved with the format `plot_prefix<some_identifying_string>.pdf` for plots and
`plot_prefix<some_identifying_string>.gif`, etc. for animations.

If `io` is passed then error norms will be written to that file.

`input` is a NamedTuple of settings to use. If not given it will be read from the
`[manufactured_solns]` section of [`input_dict_dfns`][@ref].

Note: when calculating error norms, data is loaded only for 1 time point and for an r-z
chunk that is the same size as computed by 1 block of the simulation at run time. This
should prevent excessive memory requirements for this function.
"""
function compare_neutral_pdf_symbolic_test(run_info, plot_prefix; io=nothing,
                                           input=nothing)

    field_label = L"\tilde{f}_n"
    field_sym_label = L"\tilde{f}_n^{sym}"
    norm_label = L"\varepsilon(\tilde{f}_n)"
    variable_name = "f_neutral"

    println("Doing MMS analysis and making plots for $variable_name")
    flush(stdout)

    if input === nothing
        input = Dict_to_NamedTuple(input_dict_dfns["manufactured_solns"])
    end

    nt = run_info.nt
    r = run_info.r
    z = run_info.z
    vzeta = run_info.vzeta
    vr = run_info.vr
    vz = run_info.vz

    # Load data in chunks, with the same size as the chunks that were saved during the
    # run, to avoid running out of memory
    if !input.calculate_error_norms
        field_norm = nothing
    else
        r_chunks = UnitRange{mk_int}[]
        chunk = run_info.r_chunk_size
        nchunks = (r.n ÷ chunk)
        if nchunks == 1
            r_chunks = [1:r.n]
        else
            for i ∈ 1:nchunks
                if i == nchunks
                    push!(r_chunks, (i-1)*chunk+1:i*chunk+1)
                else
                    push!(r_chunks, (i-1)*chunk+1:i*chunk)
                end
            end
        end
        z_chunks = UnitRange{mk_int}[]
        chunk = run_info.z_chunk_size
        nchunks = (z.n ÷ chunk)
        if nchunks == 1
            z_chunks = [1:z.n]
        else
            for i ∈ 1:nchunks
                if i == nchunks
                    push!(z_chunks, (i-1)*chunk+1:i*chunk+1)
                else
                    push!(z_chunks, (i-1)*chunk+1:i*chunk)
                end
            end
        end
        field_norm = zeros(mk_float,nt)
        for it in 1:nt
            dummy = 0.0
            #dummy_N = 0.0
            for r_chunk ∈ r_chunks, z_chunk ∈ z_chunks
                f, f_sym =
                    manufactured_solutions_get_field_and_field_sym(
                        run_info, variable_name; nvperp=run_info.vperp.n, it=it,
                        ir=r_chunk, iz=z_chunk)
                dummy += sum(@. (f - f_sym)^2)
                #dummy_N += sum(f_sym.^2)
            end

            #field_norm[it] = dummy/dummy_N
            field_norm[it] = sqrt(dummy/(r.n*z.n*vzeta.n*vr.n*vz.n))
        end
        println_to_stdout_and_file(io, join(field_norm, " "), " # ", variable_name)
        plot_vs_t(run_info, norm_label, input=input, data=field_norm,
                  outfile=plot_prefix*variable_name*"_norm_vs_t.pdf")
    end

    has_rdim = (r.n > 1)
    has_zdim = (z.n > 1)
    is_1V = (vzeta.n == 1 && vr.n == 1)

    if input.wall_plots
        for (iz, z_label) ∈ ((1, "wall-"), (z.n, "wall+"))
            f, f_sym =
                manufactured_solutions_get_field_and_field_sym(
                    run_info, variable_name; nvperp=run_info.vperp.n, it=input.it0,
                    ir=input.ir0, iz=iz, ivzeta=input.ivzeta0, ivr=input.ivr0)
            error = f .- f_sym

            fig, ax, legend_place = get_1d_ax(2; get_legend_place=:below)
            plot_1d(vz.grid, f, ax=ax[1], label="num",
                    xlabel=L"v_{z}/L_{v_{z}}", ylabel=field_label)
            plot_1d(vz.grid, f_sym, ax=ax[1], label="sym")
            Legend(legend_place[1], ax[1]; tellheight=true, tellwidth=false,
                   orientation=:horizontal)

            plot_1d(vz.grid, error, ax=ax[2], xlabel=L"v_{z}/L_{v_{z}}",
                    ylabel=norm_label)

            outfile = plot_prefix * variable_name * "(" * z_label * ")_vs_vz.pdf"
            save(outfile, fig)

            if has_rdim
                f, f_sym =
                manufactured_solutions_get_field_and_field_sym(
                    run_info, variable_name; nvperp=run_info.vperp.n, it=input.it0, iz=iz,
                    ivzeta=input.ivzeta0, ivr=input.ivr0)
                error = f .- f_sym

                fig, ax, colorbar_place = get_2d_ax(3)
                plot_2d(vz.grid, r.grid, f, ax=ax[1], colorbar_place=colorbar_place[1],
                        title=field_label, xlabel=L"v_{z}/L_{v_{z}}",
                        ylabel=L"r")
                plot_2d(vz.grid, r.grid, f_sym, ax=ax[2],
                        colorbar_place=colorbar_place[2], title=field_sym_label,
                        xlabel=L"v_{z}/L_{v_{z}}", ylabel=L"r")
                plot_2d(vz.grid, r.grid, error, ax=ax[3],
                        colorbar_place=colorbar_place[3], title=norm_label,
                        xlabel=L"v_{z}/L_{v_{z}}", ylabel=L"r")

                outfile = plot_prefix * variable_name * "(" * z_label * ")_vs_vz_r.pdf"
                save(outfile, fig)
            end

            if !is_1V
                f, f_sym =
                manufactured_solutions_get_field_and_field_sym(
                    run_info, variable_name; nvperp=run_info.vperp.n, it=input.it0, iz=iz,
                    ir=input.ir0, ivzeta=input.ivzeta0)
                error = f .- f_sym

                fig, ax, colorbar_place = get_2d_ax(3)
                plot_2d(vz.grid, vr.grid, f, ax=ax[1],
                        colorbar_place=colorbar_place[1], title=field_label,
                        xlabel=L"v_{z}/L_{v_{z}}",
                        ylabel=L"v_{r}/L_{v_{r}}")
                plot_2d(vz.grid, vr.grid, f_sym, ax=ax[2],
                        colorbar_place=colorbar_place[2], title=field_sym_label,
                        xlabel=L"v_{z}/L_{v_{z}}",
                        ylabel=L"v_{r}/L_{v_{r}}")
                plot_2d(vz.grid, vr.grid, error, ax=ax[3],
                        colorbar_place=colorbar_place[3], title=norm_label,
                        xlabel=L"v_{z}/L_{v_{z}}",
                        ylabel=L"v_{r}/L_{v_{r}}")

                outfile = plot_prefix * variable_name * "(" * z_label * ")_vs_vz_vr.pdf"
                save(outfile, fig)

                f, f_sym =
                manufactured_solutions_get_field_and_field_sym(
                    run_info, variable_name; nvperp=run_info.vperp.n, it=input.it0, iz=iz,
                    ir=input.ir0, ivr=input.ivr0)
                error = f .- f_sym

                fig, ax, colorbar_place = get_2d_ax(3)
                plot_2d(vz.grid, vzeta.grid, f, ax=ax[1],
                        colorbar_place=colorbar_place[1], title=field_label,
                        xlabel=L"v_{z}/L_{v_{z}}",
                        ylabel=L"v_{\zeta}/L_{v_{\zeta}}")
                plot_2d(vz.grid, vzeta.grid, f_sym, ax=ax[2],
                        colorbar_place=colorbar_place[2], title=field_sym_label,
                        xlabel=L"v_{z}/L_{v_{z}}",
                        ylabel=L"v_{\zeta}/L_{v_{\zeta}}")
                plot_2d(vz.grid, vzeta.grid, error, ax=ax[3],
                        colorbar_place=colorbar_place[3], title=norm_label,
                        xlabel=L"v_{z}/L_{v_{z}}",
                        ylabel=L"v_{\zeta}/L_{v_{\zeta}}")

                outfile = plot_prefix * variable_name * "(" * z_label * ")_vs_vz_vzeta.pdf"
                save(outfile, fig)
            end
        end
    end

    animate_dims = setdiff(neutral_dimensions, (:sn,))
    if !has_rdim
        animate_dims = setdiff(animate_dims, (:r,))
    end
    if !has_zdim
        animate_dims = setdiff(animate_dims, (:z,))
    end
    if !has_zdim
        animate_dims = setdiff(animate_dims, (:z,))
    end
    if is_1V
        animate_dims = setdiff(animate_dims, (:vzeta, :vr))
    end
    plot_dims = tuple(:t, animate_dims...)
    _MMS_pdf_plots(run_info, input, variable_name, plot_prefix, field_label,
                   field_sym_label, norm_label, plot_dims, animate_dims, true)

    return field_norm
end

"""
    manufactured_solutions_analysis(run_info; plot_prefix)
    manufactured_solutions_analysis(run_info::Tuple; plot_prefix)

Compare computed and manufactured solutions for field and moment variables for a 'method
of manufactured solutions' (MMS) test.

The information for the run to analyse is passed in `run_info` (as returned by
[`get_run_info`](@ref)).

`plot_prefix` is required and gives the path and prefix for plots to be saved to. They
will be saved with the format `plot_prefix<some_identifying_string>.pdf` for plots and
`plot_prefix<some_identifying_string>.gif`, etc. for animations.

Settings are read from the `[manufactured_solns]` section of the input.

While a Tuple of `run_info` can be passed for compatibility with `makie_post_process()`,
at present comparison of multiple runs is not supported - passing a Tuple of length
greater than one will result in an error.
"""
function manufactured_solutions_analysis end

function manufactured_solutions_analysis(run_info::Tuple; plot_prefix, nvperp)
    if !any(ri !== nothing && ri.manufactured_solns_input.use_for_advance &&
            ri.manufactured_solns_input.use_for_init for ri ∈ run_info)
        # No manufactured solutions tests
        return nothing
    end

    input = Dict_to_NamedTuple(input_dict["manufactured_solns"])
    if !any(v for v ∈ values(input) if isa(v, Bool))
        # Skip as there is nothing to do
        return nothing
    end

    if length(run_info) > 1
        println("Analysing more than one run at once not supported for"
                * "manufactured_solutions_analysis()")
        return nothing
    end
    try
        return manufactured_solutions_analysis(run_info[1]; plot_prefix=plot_prefix,
                                               nvperp=nvperp)
    catch e
        return makie_post_processing_error_handler(
                   e,
                   "Error in manufactured_solutions_analysis().")
    end
end

function manufactured_solutions_analysis(run_info; plot_prefix, nvperp)
    manufactured_solns_input = run_info.manufactured_solns_input
    if !(manufactured_solns_input.use_for_advance && manufactured_solns_input.use_for_init)
        return nothing
    end

    if nvperp === nothing
        error("No `nvperp` found - must have distributions function outputs to plot MMS "
              * "tests")
    end

    input = Dict_to_NamedTuple(input_dict["manufactured_solns"])

    open(run_info.run_prefix * "MMS_errors.txt", "w") do io
        println_to_stdout_and_file(io, "# ", run_info.run_name)
        println_to_stdout_and_file(io, join(run_info.time, " "), " # time / (Lref/cref): ")

        for (variable_name, field_label, field_sym_label, norm_label) ∈
                (("phi", L"\tilde{\phi}", L"\tilde{\phi}^{sym}", L"\varepsilon(\tilde{\phi})"),
                 ("Er", L"\tilde{E}_r", L"\tilde{E}_r^{sym}", L"\varepsilon(\tilde{E}_r)"),
                 ("Ez", L"\tilde{E}_z", L"\tilde{E}_z^{sym}", L"\varepsilon(\tilde{E}_z)"),
                 ("density", L"\tilde{n}_i", L"\tilde{n}_i^{sym}", L"\varepsilon(\tilde{n}_i)"),
                 ("parallel_flow", L"\tilde{u}_{i,\parallel}", L"\tilde{u}_{i,\parallel}^{sym}", L"\varepsilon(\tilde{u}_{i,\parallel})"),
                 ("pressure", L"\tilde{p}_{i}", L"\tilde{p}_{i}^{sym}", L"\varepsilon(\tilde{p}_{i})"),
                 ("density_neutral", L"\tilde{n}_n", L"\tilde{n}_n^{sym}", L"\varepsilon(\tilde{n}_n)"))

            if contains(variable_name, "neutral") && run_info.n_neutral_species == 0
                continue
            end
            if contains(variable_name, "Er") && run_info.r.n_global == 1
                continue
            end

            compare_moment_symbolic_test(run_info, plot_prefix, field_label, field_sym_label,
                                         norm_label, variable_name; io=io, input=input,
                                         nvperp=nvperp)
        end
    end

    return nothing
end

"""
    manufactured_solutions_analysis_dfns(run_info; plot_prefix)
    manufactured_solutions_analysis_dfns(run_info::Tuple; plot_prefix)

Compare computed and manufactured solutions for distribution function variables for a
'method of manufactured solutions' (MMS) test.

The information for the run to analyse is passed in `run_info` (as returned by
[`get_run_info`](@ref)).

`plot_prefix` is required and gives the path and prefix for plots to be saved to. They
will be saved with the format `plot_prefix<some_identifying_string>.pdf` for plots and
`plot_prefix<some_identifying_string>.gif`, etc. for animations.

Settings are read from the `[manufactured_solns]` section of the input.

While a Tuple of `run_info` can be passed for compatibility with `makie_post_process()`,
at present comparison of multiple runs is not supported - passing a Tuple of length
greater than one will result in an error.
"""
function manufactured_solutions_analysis_dfns end

function manufactured_solutions_analysis_dfns(run_info::Tuple; plot_prefix)
    if !any(ri !== nothing && ri.manufactured_solns_input.use_for_advance &&
            ri.manufactured_solns_input.use_for_init for ri ∈ run_info)
        # No manufactured solutions tests
        return nothing
    end

    input = Dict_to_NamedTuple(input_dict_dfns["manufactured_solns"])
    if !any(v for v ∈ values(input) if isa(v, Bool))
        # Skip as there is nothing to do
        return nothing
    end

    if length(run_info) > 1
        println("Analysing more than one run at once not supported for"
                * "manufactured_solutions_analysis_dfns()")
        return nothing
    end
    try
        return manufactured_solutions_analysis_dfns(run_info[1]; plot_prefix=plot_prefix)
    catch e
        return makie_post_processing_error_handler(
                   e,
                   "Error in manufactured_solutions_analysis_dfns().")
    end
end

function manufactured_solutions_analysis_dfns(run_info; plot_prefix)
    manufactured_solns_input = run_info.manufactured_solns_input
    if !(manufactured_solns_input.use_for_advance && manufactured_solns_input.use_for_init)
        return nothing
    end

    input = Dict_to_NamedTuple(input_dict_dfns["manufactured_solns"])

    open(run_info.run_prefix * "MMS_dfns_errors.txt", "w") do io
        println_to_stdout_and_file(io, "# ", run_info.run_name)
        println_to_stdout_and_file(io, join(run_info.time, " "), " # time / (Lref/cref): ")

        compare_ion_pdf_symbolic_test(run_info, plot_prefix; io=io, input=input)

        if run_info.n_neutral_species > 0
            compare_neutral_pdf_symbolic_test(run_info, plot_prefix; io=io, input=input)
        end
    end

    return nothing
end
