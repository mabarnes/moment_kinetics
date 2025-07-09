using moment_kinetics.analysis: get_unnormalised_f_dzdt_1d, get_unnormalised_f_coords_2d,
                                get_unnormalised_f_1d, vpagrid_to_v_parallel_2d,
                                get_unnormalised_f_2d
using moment_kinetics.array_allocation: allocate_float
using moment_kinetics.initial_conditions: vpagrid_to_dzdt

using Combinatorics

const one_dimension_combinations_no_t = setdiff(all_dimensions, (:s, :sn))
const one_dimension_combinations = (:t, one_dimension_combinations_no_t...)
const two_dimension_combinations_no_t = Tuple(
          Tuple(c) for c in unique((combinations(setdiff(ion_dimensions, (:s,)), 2)...,
                                    combinations(setdiff(neutral_dimensions, (:sn,)), 2)...)))
const two_dimension_combinations = Tuple(
         Tuple(c) for c in
         unique((combinations((:t, setdiff(ion_dimensions, (:s,))...), 2)...,
                 combinations((:t, setdiff(neutral_dimensions, (:sn,))...), 2)...)))

# Generate 1d plot functions for each dimension
for dim ∈ one_dimension_combinations
    function_name_str = "plot_vs_$dim"
    function_name = Symbol(function_name_str)
    spaces = " " ^ (length(function_name_str) + 1)
    dim_str = String(dim)
    if dim == :t
        dim_grid = :( run_info.time )
    else
        dim_grid = :( run_info.$dim.grid )
    end
    idim = Symbol(:i, dim)
    eval(quote
             export $function_name

             """
                 $($function_name_str)(run_info::Vector{Any}, var_name; is=1, data=nothing,
                 $($spaces)input=nothing, outfile=nothing, yscale=nothing,
                 transform=identity, axis_args=Dict{Symbol,Any}(), it=nothing,
                 $($spaces)ir=nothing, iz=nothing, ivperp=nothing, ivpa=nothing,
                 $($spaces)ivzeta=nothing, ivr=nothing, ivz=nothing, kwargs...)
                 $($function_name_str)(run_info, var_name; is=1, data=nothing,
                 $($spaces)input=nothing, ax=nothing, label=nothing,
                 $($spaces)outfile=nothing, yscale=nothing, transform=identity,
                 $($spaces)axis_args=Dict{Symbol,Any}(), it=nothing, ir=nothing,
                 $($spaces)iz=nothing, ivperp=nothing, ivpa=nothing, ivzeta=nothing,
                 $($spaces)ivr=nothing, ivz=nothing, kwargs...)

             Plot `var_name` from the run(s) represented by `run_info` (as returned by
             [`get_run_info`](@ref)) vs $($dim_str).

             If a Vector of `run_info` is passed, the plots from each run are overlayed on
             the same axis, and a legend is added.

             `it`, `is`, `ir`, `iz`, `ivperp`, `ivpa`, `ivzeta`, `ivr`, and `ivz` can be
             used to select different indices (for non-plotted dimensions) or range (for
             the plotted dimension) to use.

             If `outfile` is given, the plot will be saved to a file with that name. The
             suffix determines the file type.

             `yscale` can be used to set the scaling function for the y-axis. Options are
             `identity`, `log`, `log2`, `log10`, `sqrt`, `Makie.logit`,
             `Makie.pseudolog10` and `Makie.Symlog10`. `transform` is a function that is
             applied element-by-element to the data before it is plotted. For example when
             using a log scale on data that may contain some negative values it might be
             useful to pass `transform=abs` (to plot the absolute value) or
             `transform=positive_or_nan` (to ignore any negative or zero values).

             `axis_args` are passed as keyword arguments to `get_1d_ax()`, and from there
             to the `Axis` constructor.

             Extra `kwargs` are passed to Makie's `lines!() function`.

             When a single `run_info` is passed, `label` can be used to set the label for
             the line created by this plot, which would be used if it is added to a
             `Legend`.

             When a single `run_info` is passed, an `Axis` can be passed to `ax`. If it
             is, the plot will be added to `ax`.

             By default the data for the variable is loaded from the output represented by
             `run_info`. The data can optionally be passed to `data` if you have already
             loaded it.

             Returns the `Figure`, unless `ax` was passed in which case the object
             returned by Makie's `lines!()` function is returned.

             By default relevant settings are read from the `var_name` section of
             [`input_dict_dfns`](@ref) (if output that has distribution functions is being
             read) or [`input_dict`](@ref) (otherwise). The settings can also be passed as
             an `AbstractDict` or `NamedTuple` via the `input` argument.  Sometimes
             needed, for example if `var_name` is not present in `input_dict` (in which
             case you would have had to create the array to be plotted and pass it to
             `data`).
             """
             function $function_name end

             function $function_name(run_info::Vector{Any}, var_name; is=1, data=nothing,
                                     input=nothing, outfile=nothing, yscale=nothing,
                                     transform=identity, axis_args=Dict{Symbol,Any}(),
                                     $idim=nothing, kwargs...)

                 try
                     if data === nothing
                         data = [nothing for _ in run_info]
                     end

                     if input === nothing
                         if run_info[1].dfns
                             if var_name ∈ keys(input_dict_dfns)
                                 input = input_dict_dfns[var_name]
                             else
                                 input = input_dict_dfns
                             end
                         else
                             if var_name ∈ keys(input_dict)
                                 input = input_dict[var_name]
                             else
                                 input = input_dict
                             end
                         end
                     end
                     if input isa AbstractDict
                         input = Dict_to_NamedTuple(input)
                     end

                     n_runs = length(run_info)

                     fig, ax = get_1d_ax(; xlabel="$($dim_str)",
                                         ylabel=get_variable_symbol(var_name),
                                         yscale=yscale, axis_args...)
                     for (d, ri) ∈ zip(data, run_info)
                         $function_name(ri, var_name, is=is, data=d, input=input, ax=ax,
                                        transform=transform, label=ri.run_name,
                                        $idim=$idim, kwargs...)
                     end

                     if input.show_element_boundaries && Symbol($dim_str) != :t
                         # Just plot element boundaries from first run, assuming that all
                         # runs being compared use the same grid.
                         ri = run_info[1]
                         element_boundary_inds =
                             [i for i ∈ 1:ri.$dim.ngrid-1:ri.$dim.n_global
                                if $idim === nothing || i ∈ $idim]
                         element_boundary_positions = ri.$dim.grid[element_boundary_inds]
                         vlines!(ax, element_boundary_positions, color=:black, alpha=0.3)
                     end

                     if n_runs > 1
                         put_legend_below(fig, ax)
                         # Ensure the first row width is 3/4 of the column width so that
                         # the plot does not get squashed by the legend
                         rowsize!(fig.layout, 1, Aspect(1, 3/4))
                         resize_to_layout!(fig)
                     end

                     if outfile !== nothing
                         save(outfile, fig)
                     end
                     return fig
                 catch e
                     return makie_post_processing_error_handler(
                                e,
                                "$($function_name_str) failed for $var_name, is=$is.")
                 end
             end

             function $function_name(run_info, var_name; is=1, data=nothing,
                                     input=nothing, fig=nothing, ax=nothing,
                                     label=nothing, outfile=nothing,
                                     axis_args=Dict{Symbol,Any}(), it=nothing,
                                     ir=nothing, iz=nothing, ivperp=nothing,
                                     ivpa=nothing, ivzeta=nothing, ivr=nothing,
                                     ivz=nothing, kwargs...)
                 if input === nothing
                     if run_info.dfns
                         if var_name ∈ keys(input_dict_dfns)
                             input = input_dict_dfns[var_name]
                         else
                             input = input_dict_dfns
                         end
                     else
                         if var_name ∈ keys(input_dict)
                             input = input_dict[var_name]
                         else
                             input = input_dict
                         end
                     end
                 end
                 if isa(input, AbstractDict)
                     input = Dict_to_NamedTuple(input)
                 end
                 if data === nothing
                     dim_slices = get_dimension_slice_indices($(QuoteNode(dim));
                                                              run_info=run_info,
                                                              input=input, it=it, is=is,
                                                              ir=ir, iz=iz, ivperp=ivperp,
                                                              ivpa=ivpa, ivzeta=ivzeta,
                                                              ivr=ivr, ivz=ivz)
                     data = get_variable(run_info, var_name; dim_slices...)
                 else
                     data = select_slice(data, $(QuoteNode(dim)); input=input, it=it,
                                         is=is, ir=ir, iz=iz, ivperp=ivperp, ivpa=ivpa,
                                         ivzeta=ivzeta, ivr=ivr, ivz=ivz)
                 end

                 if ax === nothing
                     fig, ax = get_1d_ax(; xlabel="$($dim_str)",
                                         ylabel=get_variable_symbol(var_name),
                                         axis_args...)
                     ax_was_nothing = true
                 else
                     ax_was_nothing = false
                 end

                 x = $dim_grid
                 if $idim !== nothing
                     x = x[$idim]
                 end
                 plot_1d(x, data; label=label, ax=ax, kwargs...)

                 if input.show_element_boundaries && Symbol($dim_str) != :t && ax_was_nothing
                     element_boundary_inds =
                         [i for i ∈ 1:run_info.$dim.ngrid-1:run_info.$dim.n_global
                            if $idim === nothing || i ∈ $idim]
                     element_boundary_positions = run_info.$dim.grid[element_boundary_inds]
                     vlines!(ax, element_boundary_positions, color=:black, alpha=0.3)
                 end

                 if outfile !== nothing
                     if fig === nothing
                         error("When `outfile` is passed to save the plot, must either pass both "
                               * "`fig` and `ax` or neither. Only `ax` was passed.")
                     end
                     save(outfile, fig)
                 end

                 return fig
             end
         end)
end

# Generate 2d plot functions for all combinations of dimensions
for (dim1, dim2) ∈ two_dimension_combinations
    function_name_str = "plot_vs_$(dim2)_$(dim1)"
    function_name = Symbol(function_name_str)
    spaces = " " ^ (length(function_name_str) + 1)
    dim1_str = String(dim1)
    dim2_str = String(dim2)
    if dim1 == :t
        dim1_grid = :( run_info.time )
    else
        dim1_grid = :( run_info.$dim1.grid )
    end
    dim2_grid = :( run_info.$dim2.grid )
    idim1 = Symbol(:i, dim1)
    idim2 = Symbol(:i, dim2)
    eval(quote
             export $function_name

             """
                 $($function_name_str)(run_info::Vector{Any}, var_name; is=1, data=nothing,
                 $($spaces)input=nothing, outfile=nothing, colorscale=identity,
                 $($spaces)transform=identity, axis_args=Dict{Symbol,Any}(),
                 $($spaces)it=nothing, ir=nothing, iz=nothing, ivperp=nothing,
                 $($spaces)ivpa=nothing, ivzeta=nothing, ivr=nothing, ivz=nothing,
                 $($spaces)kwargs...)
                 $($function_name_str)(run_info, var_name; is=1, data=nothing,
                 $($spaces)input=nothing, ax=nothing,
                 $($spaces)colorbar_place=nothing, title=nothing,
                 $($spaces)outfile=nothing, colorscale=identity, transform=identity,
                 $($spaces)axis_args=Dict{Symbol,Any}(), it=nothing, ir=nothing,
                 $($spaces)iz=nothing, ivperp=nothing, ivpa=nothing, ivzeta=nothing,
                 $($spaces)ivr=nothing, ivz=nothing, kwargs...)

             Plot `var_name` from the run(s) represented by `run_info` (as returned by
             [`get_run_info`](@ref))vs $($dim1_str) and $($dim2_str).

             If a Vector of `run_info` is passed, the plots from each run are displayed in
             a horizontal row, and the subtitle for each subplot is the 'run name'.

             `it`, `is`, `ir`, `iz`, `ivperp`, `ivpa`, `ivzeta`, `ivr`, and `ivz` can be
             used to select different indices (for non-plotted dimensions) or range (for
             the plotted dimension) to use.

             If `outfile` is given, the plot will be saved to a file with that name. The
             suffix determines the file type.

             `colorscale` can be used to set the scaling function for the colors. Options
             are `identity`, `log`, `log2`, `log10`, `sqrt`, `Makie.logit`,
             `Makie.pseudolog10` and `Makie.Symlog10`. `transform` is a function that is
             applied element-by-element to the data before it is plotted. For example when
             using a log scale on data that may contain some negative values it might be
             useful to pass `transform=abs` (to plot the absolute value) or
             `transform=positive_or_nan` (to ignore any negative or zero values).

             `axis_args` are passed as keyword arguments to `get_2d_ax()`, and from there
             to the `Axis` constructor.

             Extra `kwargs` are passed to Makie's `heatmap!() function`.

             When a single `run_info` is passed, `title` can be used to set the title for
             the (sub-)plot.

             When a single `run_info` is passed, an `Axis` can be passed to `ax`. If it
             is, the plot will be added to `ax`. A colorbar will be created in
             `colorbar_place` if it is given a `GridPosition`.

             By default the data for the variable is loaded from the output represented by
             `run_info`. The data can optionally be passed to `data` if you have already
             loaded it.

             Returns the `Figure`, unless `ax` was passed in which case the object
             returned by Makie's `heatmap!()` function is returned.

             By default relevant settings are read from the `var_name` section of
             [`input_dict_dfns`](@ref) (if output that has distribution functions is being
             read) or [`input_dict`](@ref) (otherwise). The settings can also be passed as
             an `AbstractDict` or `NamedTuple` via the `input` argument.  Sometimes
             needed, for example if `var_name` is not present in `input_dict` (in which
             case you would have had to create the array to be plotted and pass it to
             `data`).
             """
             function $function_name end

             function $function_name(run_info::Vector{Any}, var_name; is=1, data=nothing,
                                     input=nothing, outfile=nothing, transform=identity,
                                     axis_args=Dict{Symbol,Any}(), kwargs...)

                 try
                     if data === nothing
                         data = [nothing for _ in run_info]
                     end
                     fig, ax, colorbar_places = get_2d_ax(length(run_info);
                                                          title=get_variable_symbol(var_name),
                                                          axis_args...)
                     for (d, ri, a, cp) ∈ zip(data, run_info, ax, colorbar_places)
                         $function_name(ri, var_name; is=is, data=d, input=input, ax=a,
                                        transform=transform, colorbar_place=cp,
                                        title=ri.run_name, kwargs...)
                     end

                     if outfile !== nothing
                         save(outfile, fig)
                     end
                     return fig
                 catch e
                     return makie_post_processing_error_handler(
                                e,
                                "$($function_name_str) failed for $var_name, is=$is.")
                 end
             end

             function $function_name(run_info, var_name; is=1, data=nothing,
                                     input=nothing, ax=nothing,
                                     colorbar_place=nothing, title=nothing,
                                     outfile=nothing, axis_args=Dict{Symbol,Any}(),
                                     it=nothing, ir=nothing, iz=nothing, ivperp=nothing,
                                     ivpa=nothing, ivzeta=nothing, ivr=nothing,
                                     ivz=nothing, kwargs...)
                 if input === nothing
                     if run_info.dfns
                         if var_name ∈ keys(input_dict_dfns)
                             input = input_dict_dfns[var_name]
                         else
                             input = input_dict_dfns
                         end
                     else
                         if var_name ∈ keys(input_dict)
                             input = input_dict[var_name]
                         else
                             input = input_dict
                         end
                     end
                 end
                 if isa(input, AbstractDict)
                     input = Dict_to_NamedTuple(input)
                 end
                 if data === nothing
                     dim_slices = get_dimension_slice_indices($(QuoteNode(dim1)),
                                                              $(QuoteNode(dim2));
                                                              run_info=run_info,
                                                              input=input, it=it, is=is,
                                                              ir=ir, iz=iz, ivperp=ivperp,
                                                              ivpa=ivpa, ivzeta=ivzeta,
                                                              ivr=ivr, ivz=ivz)
                     data = get_variable(run_info, var_name; dim_slices...)
                 else
                     data = select_slice(data, $(QuoteNode(dim2)), $(QuoteNode(dim1));
                                         input=input, it=it, is=is, ir=ir, iz=iz,
                                         ivperp=ivperp, ivpa=ivpa, ivzeta=ivzeta, ivr=ivr,
                                         ivz=ivz)
                 end
                 if input === nothing
                     colormap = "reverse_deep"
                 else
                     colormap = input.colormap
                 end
                 if title === nothing
                     title = get_variable_symbol(var_name)
                 end

                 if ax === nothing
                     fig, ax, colorbar_place = get_2d_ax(; title=title, axis_args...)
                     ax_was_nothing = true
                 else
                     fig = nothing
                     ax_was_nothing = false
                 end

                 x = $dim2_grid
                 if $idim2 !== nothing
                     x = x[$idim2]
                 end
                 y = $dim1_grid
                 if $idim1 !== nothing
                     y = y[$idim1]
                 end
                 plot_2d(x, y, data; ax=ax, xlabel="$($dim2_str)",
                         ylabel="$($dim1_str)", colorbar_place=colorbar_place,
                         colormap=colormap, kwargs...)

                 if input.show_element_boundaries && Symbol($dim2_str) != :t
                     element_boundary_inds =
                         [i for i ∈ 1:run_info.$dim2.ngrid-1:run_info.$dim2.n_global
                            if $idim2 === nothing || i ∈ $idim2]
                     element_boundary_positions = run_info.$dim2.grid[element_boundary_inds]
                     vlines!(ax, element_boundary_positions, color=:white, alpha=0.5)
                 end
                 if input.show_element_boundaries && Symbol($dim1_str) != :t
                     element_boundary_inds =
                         [i for i ∈ 1:run_info.$dim1.ngrid-1:run_info.$dim1.n_global
                            if $idim1 === nothing || i ∈ $idim1]
                     element_boundary_positions = run_info.$dim1.grid[element_boundary_inds]
                     hlines!(ax, element_boundary_positions, color=:white, alpha=0.5)
                 end

                 if outfile !== nothing
                     if fig === nothing
                         error("When `outfile` is passed to save the plot, must either pass both "
                               * "`fig` and `ax` or neither. Only `ax` was passed.")
                     end
                     save(outfile, fig)
                 end

                 return fig
             end
         end)
end

# Generate 1d animation functions for each dimension
for dim ∈ one_dimension_combinations_no_t
    function_name_str = "animate_vs_$dim"
    function_name = Symbol(function_name_str)
    spaces = " " ^ (length(function_name_str) + 1)
    dim_str = String(dim)
    dim_grid = :( run_info.$dim.grid )
    idim = Symbol(:i, dim)
    eval(quote
             export $function_name

             """
                 $($function_name_str)(run_info::Vector{Any}, var_name; is=1, data=nothing,
                 $($spaces)input=nothing, outfile=nothing, yscale=nothing,
                 $($spaces)transform=identity, ylims=nothing,
                 $($spaces)axis_args=Dict{Symbol,Any}(), it=nothing, ir=nothing, iz=nothing,
                 $($spaces)ivperp=nothing, ivpa=nothing, ivzeta=nothing, ivr=nothing,
                 $($spaces)ivz=nothing, kwargs...)
                 $($function_name_str)(run_info, var_name; is=1, data=nothing,
                 $($spaces)input=nothing, frame_index=nothing, ax=nothing,
                 $($spaces)fig=nothing, outfile=nothing, yscale=nothing,
                 $($spaces)transform=identity, ylims=nothing, label=nothing,
                 $($spaces)axis_args=Dict{Symbol,Any}(), it=nothing, ir=nothing, iz=nothing,
                 $($spaces)ivperp=nothing, ivpa=nothing, ivzeta=nothing, ivr=nothing,
                 $($spaces)ivz=nothing, kwargs...)

             Animate `var_name` from the run(s) represented by `run_info` (as returned by
             [`get_run_info`](@ref))vs $($dim_str).

             If a Vector of `run_info` is passed, the animations from each run are
             overlayed on the same axis, and a legend is added.

             `it`, `is`, `ir`, `iz`, `ivperp`, `ivpa`, `ivzeta`, `ivr`, and `ivz` can be
             used to select different indices (for non-plotted dimensions) or range (for
             the plotted dimension) to use.

             `ylims` can be passed a Tuple (ymin, ymax) to set the y-axis limits. By
             default the minimum and maximum of the data (over all time points) will be
             used.

             `yscale` can be used to set the scaling function for the y-axis. Options are
             `identity`, `log`, `log2`, `log10`, `sqrt`, `Makie.logit`,
             `Makie.pseudolog10` and `Makie.Symlog10`. `transform` is a function that is
             applied element-by-element to the data before it is plotted. For example when
             using a log scale on data that may contain some negative values it might be
             useful to pass `transform=abs` (to plot the absolute value) or
             `transform=positive_or_nan` (to ignore any negative or zero values).

             `axis_args` are passed as keyword arguments to `get_1d_ax()`, and from there
             to the `Axis` constructor.

             Extra `kwargs` are passed to Makie's `lines!() function`.

             When a single `run_info` is passed, an `Axis` can be passed to `ax`. If it
             is, the plot will be added to `ax`.

             When a single `run_info` is passed, `label` can be passed to set a custom
             label for the line. By default the `run_info.run_name` is used.

             `outfile` is required for animations unless `ax` is passed. The animation
             will be saved to a file named `outfile`.  The suffix determines the file
             type. If both `outfile` and `ax` are passed, then the `Figure` containing
             `ax` must be passed to `fig` to allow the animation to be saved.

             By default the data for the variable is loaded from the output represented by
             `run_info`. The data can optionally be passed to `data` if you have already
             loaded it.

             Returns the `Figure`, unless `ax` was passed in which case returns `nothing`.

             By default relevant settings are read from the `var_name` section of
             [`input_dict_dfns`](@ref) (if output that has distribution functions is being
             read) or [`input_dict`](@ref) (otherwise). The settings can also be passed as
             an `AbstractDict` or `NamedTuple` via the `input` argument.  Sometimes
             needed, for example if `var_name` is not present in `input_dict` (in which
             case you would have had to create the array to be plotted and pass it to
             `data`).
             """
             function $function_name end

             function $function_name(run_info::Vector{Any}, var_name; is=1, data=nothing,
                                     input=nothing, outfile=nothing, yscale=nothing,
                                     ylims=nothing, axis_args=Dict{Symbol,Any}(),
                                     it=nothing, $idim=nothing, kwargs...)

                 try
                     if data === nothing
                         data = [nothing for _ in run_info]
                     end
                     if outfile === nothing
                         error("`outfile` is required for $($function_name_str)")
                     end

                     if input === nothing
                         if run_info[1].dfns
                             if var_name ∈ keys(input_dict_dfns)
                                 input = input_dict_dfns[var_name]
                             else
                                 input = input_dict_dfns
                             end
                         else
                             if var_name ∈ keys(input_dict)
                                 input = input_dict[var_name]
                             else
                                 input = input_dict
                             end
                         end
                     end
                     if input isa AbstractDict
                         input = Dict_to_NamedTuple(input)
                     end

                     n_runs = length(run_info)

                     frame_index = Observable(1)
                     if length(run_info) == 1 ||
                         all(ri.nt == run_info[1].nt &&
                             all(isapprox.(ri.time, run_info[1].time))
                             for ri ∈ run_info[2:end])
                         # All times are the same
                         time = select_time_slice(run_info[1].time, it)
                         title = lift(i->string("t = ", time[i]), frame_index)
                     else
                         title = lift(i->join((string("t", irun, " = ",
                                                      select_time_slice(ri.time, it)[i])
                                               for (irun,ri) ∈ enumerate(run_info)), "; "),
                                      frame_index)
                     end
                     fig, ax = get_1d_ax(; xlabel="$($dim_str)",
                                         ylabel=get_variable_symbol(var_name),
                                         title=title, yscale=yscale, axis_args...)

                     for (d, ri) ∈ zip(data, run_info)
                         $function_name(ri, var_name; is=is, data=d, input=input,
                                        ylims=ylims, frame_index=frame_index, ax=ax,
                                        it=it, $idim=$idim, kwargs...)
                     end

                     if input.show_element_boundaries
                         # Just plot element boundaries from first run, assuming that all
                         # runs being compared use the same grid.
                         ri = run_info[1]
                         element_boundary_inds =
                             [i for i ∈ 1:ri.$dim.ngrid-1:ri.$dim.n_global
                                if $idim === nothing || i ∈ $idim]
                         element_boundary_positions = ri.$dim.grid[element_boundary_inds]
                         vlines!(ax, element_boundary_positions, color=:black, alpha=0.3)
                     end

                     if n_runs > 1
                         put_legend_below(fig, ax)
                         # Ensure the first row width is 3/4 of the column width so that
                         # the plot does not get squashed by the legend
                         rowsize!(fig.layout, 1, Aspect(1, 3/4))
                         resize_to_layout!(fig)
                     end

                     if it === nothing
                         nt = minimum(ri.nt for ri ∈ run_info)
                     else
                         nt = length(it)
                     end
                     save_animation(fig, frame_index, nt, outfile)

                     return fig
                 catch e
                     return makie_post_processing_error_handler(
                                e,
                                "$($function_name_str)() failed for $var_name, is=$is.")
                 end
             end

             function $function_name(run_info, var_name; is=1, data=nothing,
                                     input=nothing, frame_index=nothing, ax=nothing,
                                     fig=nothing, outfile=nothing, yscale=nothing,
                                     ylims=nothing, label=nothing,
                                     axis_args=Dict{Symbol,Any}(), it=nothing, ir=nothing,
                                     iz=nothing, ivperp=nothing, ivpa=nothing,
                                     ivzeta=nothing, ivr=nothing, ivz=nothing, kwargs...)
                 if input === nothing
                     if run_info.dfns
                         if var_name ∈ keys(input_dict_dfns)
                             input = input_dict_dfns[var_name]
                         else
                             input = input_dict_dfns
                         end
                     else
                         if var_name ∈ keys(input_dict)
                             input = input_dict[var_name]
                         else
                             input = input_dict
                         end
                     end
                 end
                 if isa(input, AbstractDict)
                     input = Dict_to_NamedTuple(input)
                 end
                 if data === nothing
                     dim_slices = get_dimension_slice_indices(:t, $(QuoteNode(dim));
                                                              run_info=run_info,
                                                              input=input, it=it, is=is,
                                                              ir=ir, iz=iz, ivperp=ivperp,
                                                              ivpa=ivpa, ivzeta=ivzeta,
                                                              ivr=ivr, ivz=ivz)
                     data = VariableCache(run_info, var_name, chunk_size_1d;
                                          dim_slices...)
                 else
                     data = select_slice(data, $(QuoteNode(dim)), :t; input=input, it=it,
                                         is=is, ir=ir, iz=iz, ivperp=ivperp, ivpa=ivpa,
                                         ivzeta=ivzeta, ivr=ivr, ivz=ivz)
                 end
                 if frame_index === nothing
                     ind = Observable(1)
                 else
                     ind = frame_index
                 end
                 if ax === nothing
                     time = select_time_slice(run_info.time, it)
                     title = lift(i->string("t = ", time[i]), ind)
                     fig, ax = get_1d_ax(; xlabel="$($dim_str)",
                                         ylabel=get_variable_symbol(var_name),
                                         yscale=yscale, title=title, axis_args...)
                 else
                     fig = nothing
                 end
                 if label === nothing
                     label = run_info.run_name
                 end

                 x = $dim_grid
                 if $idim !== nothing
                     x = x[$idim]
                 end
                 animate_1d(x, data; ax=ax, ylims=ylims, frame_index=ind,
                            label=label, kwargs...)

                 if input.show_element_boundaries && fig !== nothing
                     element_boundary_inds =
                         [i for i ∈ 1:run_info.$dim.ngrid-1:run_info.$dim.n_global
                            if $idim === nothing || i ∈ $idim]
                     element_boundary_positions = run_info.$dim.grid[element_boundary_inds]
                     vlines!(ax, element_boundary_positions, color=:black, alpha=0.3)
                 end

                 if frame_index === nothing
                     if outfile === nothing
                         error("`outfile` is required for $($function_name_str)")
                     end
                     if fig === nothing
                         error("When `outfile` is passed to save the plot, must either pass both "
                               * "`fig` and `ax` or neither. Only `ax` was passed.")
                     end

                     if isa(data, VariableCache)
                         nt = data.n_tinds
                     else
                         nt = size(data, 2)
                     end

                     save_animation(fig, ind, nt, outfile)
                 end

                 return fig
             end
         end)
end

# Generate 2d animation functions for all combinations of dimensions
for (dim1, dim2) ∈ two_dimension_combinations_no_t
    function_name_str = "animate_vs_$(dim2)_$(dim1)"
    function_name = Symbol(function_name_str)
    spaces = " " ^ (length(function_name_str) + 1)
    dim1_str = String(dim1)
    dim2_str = String(dim2)
    dim1_grid = :( run_info.$dim1.grid )
    dim2_grid = :( run_info.$dim2.grid )
    idim1 = Symbol(:i, dim1)
    idim2 = Symbol(:i, dim2)
    eval(quote
             export $function_name

             """
                 $($function_name_str)(run_info::Vector{Any}, var_name; is=1, data=nothing,
                 $($spaces)input=nothing, outfile=nothing, colorscale=identity,
                 $($spaces)transform=identity, axis_args=Dict{Symbol,Any}(),
                 $($spaces)it=nothing, ir=nothing, iz=nothing, ivperp=nothing,
                 $($spaces)ivpa=nothing, ivzeta=nothing, ivr=nothing, ivz=nothing,
                 $($spaces)kwargs...)
                 $($function_name_str)(run_info, var_name; is=1, data=nothing,
                 $($spaces)input=nothing, frame_index=nothing, ax=nothing,
                 $($spaces)fig=nothing, colorbar_place=colorbar_place,
                 $($spaces)title=nothing, outfile=nothing, colorscale=identity,
                 $($spaces)transform=identity, axis_args=Dict{Symbol,Any}(),
                 $($spaces)it=nothing, ir=nothing, iz=nothing, ivperp=nothing,
                 $($spaces)ivpa=nothing, ivzeta=nothing, ivr=nothing, ivz=nothing,
                 $($spaces)kwargs...)

             Animate `var_name` from the run(s) represented by `run_info` (as returned by
             [`get_run_info`](@ref))vs $($dim1_str) and $($dim2_str).

             If a Vector of `run_info` is passed, the animations from each run are
             created in a horizontal row, with each sub-animation having the 'run name' as
             its subtitle.

             `it`, `is`, `ir`, `iz`, `ivperp`, `ivpa`, `ivzeta`, `ivr`, and `ivz` can be
             used to select different indices (for non-plotted dimensions) or range (for
             the plotted dimension) to use.

             `colorscale` can be used to set the scaling function for the colors. Options
             are `identity`, `log`, `log2`, `log10`, `sqrt`, `Makie.logit`,
             `Makie.pseudolog10` and `Makie.Symlog10`. `transform` is a function that is
             applied element-by-element to the data before it is plotted. For example when
             using a log scale on data that may contain some negative values it might be
             useful to pass `transform=abs` (to plot the absolute value) or
             `transform=positive_or_nan` (to ignore any negative or zero values).

             `axis_args` are passed as keyword arguments to `get_2d_ax()`, and from there
             to the `Axis` constructor.

             Extra `kwargs` are passed to Makie's `heatmap!() function`.

             When a single `run_info` is passed, an `Axis` can be passed to `ax`. If it
             is, the plot will be created in `ax`. When `ax` is passed, a colorbar will be
             created at `colorbar_place` if a `GridPosition` is passed to
             `colorbar_place`.

             `outfile` is required for animations unless `ax` is passed. The animation
             will be saved to a file named `outfile`.  The suffix determines the file
             type. If both `outfile` and `ax` are passed, then the `Figure` containing
             `ax` must be passed to `fig` to allow the animation to be saved.

             When a single `run_info` is passed, the (sub-)title can be set with the
             `title` argument.

             By default the data for the variable is loaded from the output represented by
             `run_info`. The data can optionally be passed to `data` if you have already
             loaded it.

             Returns the `Figure`, unless `ax` was passed in which case returns `nothing`.

             By default relevant settings are read from the `var_name` section of
             [`input_dict_dfns`](@ref) (if output that has distribution functions is being
             read) or [`input_dict`](@ref) (otherwise). The settings can also be passed as
             an `AbstractDict` or `NamedTuple` via the `input` argument.  Sometimes
             needed, for example if `var_name` is not present in `input_dict` (in which
             case you would have had to create the array to be plotted and pass it to
             `data`).
             """
             function $function_name end

             function $function_name(run_info::Vector{Any}, var_name; is=1, data=nothing,
                                     input=nothing, outfile=nothing, transform=identity,
                                     axis_args=Dict{Symbol,Any}(), it=nothing, kwargs...)

                 try
                     if data === nothing
                         data = [nothing for _ in run_info]
                     end
                     if outfile === nothing
                         error("`outfile` is required for $($function_name_str)")
                     end

                     frame_index = Observable(1)

                     if length(run_info) > 1
                         title = get_variable_symbol(var_name)
                         subtitles = (lift(i->string(ri.run_name, "\nt = ",
                                                     select_time_slice(ri.time, it)[i]),
                                           frame_index)
                                      for ri ∈ run_info)
                     else
                         time = select_time_slice(run_info[1].time, it)
                         title = lift(i->string(get_variable_symbol(var_name), "\nt = ",
                                                time[i]),
                                      frame_index)
                         subtitles = nothing
                     end
                     fig, ax, colorbar_places = get_2d_ax(length(run_info);
                                                          title=title,
                                                          subtitles=subtitles,
                                                          axis_args...)

                     for (d, ri, a, cp) ∈ zip(data, run_info, ax, colorbar_places)
                         $function_name(ri, var_name; is=is, data=d, input=input,
                                        transform=transform, frame_index=frame_index,
                                        ax=a, colorbar_place=cp, it=it, kwargs...)
                     end

                     if it === nothing
                         nt = minimum(ri.nt for ri ∈ run_info)
                     else
                         nt = length(it)
                     end
                     save_animation(fig, frame_index, nt, outfile)

                     return fig
                 catch e
                     return makie_post_processing_error_handler(
                                e,
                                "$($function_name_str) failed for $var_name, is=$is.")
                 end
             end

             function $function_name(run_info, var_name; is=1, data=nothing,
                                     input=nothing, frame_index=nothing, ax=nothing,
                                     fig=nothing, colorbar_place=nothing,
                                     title=nothing, outfile=nothing,
                                     axis_args=Dict{Symbol,Any}(), it=nothing, ir=nothing,
                                     iz=nothing, ivperp=nothing, ivpa=nothing,
                                     ivzeta=nothing, ivr=nothing, ivz=nothing, kwargs...)
                 if input === nothing
                     if run_info.dfns
                         if var_name ∈ keys(input_dict_dfns)
                             input = input_dict_dfns[var_name]
                         else
                             input = input_dict_dfns
                         end
                     else
                         if var_name ∈ keys(input_dict)
                             input = input_dict[var_name]
                         else
                             input = input_dict
                         end
                     end
                 end
                 if isa(input, AbstractDict)
                     input = Dict_to_NamedTuple(input)
                 end
                 if frame_index === nothing
                     ind = Observable(1)
                 else
                     ind = frame_index
                 end
                 if data === nothing
                     dim_slices = get_dimension_slice_indices(:t, $(QuoteNode(dim1)),
                                                              $(QuoteNode(dim2));
                                                              run_info=run_info,
                                                              input=input, it=it, is=is,
                                                              ir=ir, iz=iz, ivperp=ivperp,
                                                              ivpa=ivpa, ivzeta=ivzeta,
                                                              ivr=ivr, ivz=ivz)
                     data = VariableCache(run_info, var_name, chunk_size_2d;
                                          dim_slices...)
                 else
                     data = select_slice(data, $(QuoteNode(dim2)), $(QuoteNode(dim1)), :t;
                                         input=input, it=it, is=is, ir=ir, iz=iz,
                                         ivperp=ivperp, ivpa=ivpa, ivzeta=ivzeta, ivr=ivr,
                                         ivz=ivz)
                 end
                 if input === nothing
                     colormap = "reverse_deep"
                 else
                     colormap = input.colormap
                 end
                 if title === nothing && ax == nothing
                     time = select_time_slice(run_info.time, it)
                     title = lift(i->string(get_variable_symbol(var_name), "\nt = ",
                                            time[i]),
                                  ind)
                 end

                 if ax === nothing
                     fig, ax, colorbar_place = get_2d_ax(; title=title, axis_args...)
                     ax_was_nothing = true
                 else
                     ax_was_nothing = false
                 end

                 x = $dim2_grid
                 if $idim2 !== nothing
                     x = x[$idim2]
                 end
                 y = $dim1_grid
                 if $idim1 !== nothing
                     y = y[$idim1]
                 end
                 anim = animate_2d(x, y, data; xlabel="$($dim2_str)",
                                   ylabel="$($dim1_str)", frame_index=ind, ax=ax,
                                   colorbar_place=colorbar_place, colormap=colormap,
                                   kwargs...)

                 if input.show_element_boundaries
                     element_boundary_inds =
                         [i for i ∈ 1:run_info.$dim2.ngrid-1:run_info.$dim2.n_global
                            if $idim2 === nothing || i ∈ $idim2]
                     element_boundary_positions = run_info.$dim2.grid[element_boundary_inds]
                     vlines!(ax, element_boundary_positions, color=:white, alpha=0.5)
                 end
                 if input.show_element_boundaries
                     element_boundary_inds =
                         [i for i ∈ 1:run_info.$dim1.ngrid-1:run_info.$dim1.n_global
                            if $idim1 === nothing || i ∈ $idim1]
                     element_boundary_positions = run_info.$dim1.grid[element_boundary_inds]
                     hlines!(ax, element_boundary_positions, color=:white, alpha=0.5)
                 end

                 if frame_index === nothing
                     if outfile === nothing
                         error("`outfile` is required for $($function_name_str)")
                     end
                     if ax_was_nothing && fig === nothing
                         error("When `outfile` is passed to save the plot, must either pass both "
                               * "`fig` and `ax` or neither. Only `ax` was passed.")
                     end
                     if isa(data, VariableCache)
                         nt = data.n_tinds
                     else
                         nt = size(data, 3)
                     end
                     save_animation(fig, ind, nt, outfile)
                 end

                 return fig
             end
         end)
end

"""
    get_1d_ax(n=nothing; title=nothing, subtitles=nothing, yscale=nothing,
              get_legend_place=nothing, size=nothing, kwargs...)

Create a new `Figure` `fig` and `Axis` `ax` intended for 1d plots.

`title` gives an overall title to the `Figure`.

`yscale` can be used to set the scaling function for the y-axis. Options are `identity`,
`log`, `log2`, `log10`, `sqrt`, `Makie.logit`, `Makie.pseudolog10` and `Makie.Symlog10`.

By default creates a single `Axis`, and returns `(fig, ax)`.
If a number of axes `n` is passed, then `ax` is a `Vector{Axis}` of length `n` (even if
`n` is 1). The axes are created in a horizontal row, and the width of the figure is
increased in proportion to `n`.

`get_legend_place` can be set to one of (:left, :right, :above, :below) to create a
`GridPosition` for a legend in the corresponding place relative to each `Axis`. If
`get_legend_place` is set, `(fig, ax, legend_place)` is returned where `legend_place` is a
`GridPosition` (if `n=nothing`) or a Tuple of `n` `GridPosition`s.

When `n` is passed, `subtitles` can be passed a Tuple of length `n` which will be used to
set a subtitle for each `Axis` in `ax`.

`size` is passed through to the `Figure` constructor. Its default value is `(600, 400)` if
`n` is not passed, or `(600*n, 400)` if `n` is passed.

Extra `kwargs` are passed to the `Axis()` constructor.
"""
function get_1d_ax(n=nothing; title=nothing, subtitles=nothing, yscale=nothing,
                   get_legend_place=nothing, size=nothing, kwargs...)
    valid_legend_places = (nothing, :left, :right, :above, :below)
    if get_legend_place ∉ valid_legend_places
        error("get_legend_place=$get_legend_place is not one of $valid_legend_places")
    end
    if yscale !== nothing
        kwargs = tuple(kwargs..., :yscale=>yscale)
    end
    if n == nothing
        if size == nothing
            size = (600, 400)
        end
        fig = Figure(size=size)
        ax = Axis(fig[1,1]; kwargs...)
        if get_legend_place === :left
            legend_place = fig[1,0]
        elseif get_legend_place === :right
            legend_place = fig[1,2]
        elseif get_legend_place === :above
            legend_place = fig[0,1]
        elseif get_legend_place === :below
            legend_place = fig[2,1]
        end
        if title !== nothing
            title_layout = fig[0,1] = GridLayout()
            Label(title_layout[1,1:2], title)
        end
    else
        if size == nothing
            size = (600*n, 400)
        end
        fig = Figure(size=size)
        plot_layout = fig[1,1] = GridLayout()

        if title !== nothing
            title_layout = fig[0,1] = GridLayout()
            Label(title_layout[1,1:2], title)
        end

        if get_legend_place === :left
            if subtitles === nothing
                ax = [Axis(plot_layout[1,2*i]; kwargs...) for i in 1:n]
            else
                ax = [Axis(plot_layout[1,2*i]; title=st, kwargs...)
                      for (i,st) in zip(1:n, subtitles)]
            end
            legend_place = [plot_layout[1,2*i-1] for i in 1:n]
        elseif get_legend_place === :right
            if subtitles === nothing
                ax = [Axis(plot_layout[1,2*i-1]; kwargs...) for i in 1:n]
            else
                ax = [Axis(plot_layout[1,2*i-1]; title=st, kwargs...)
                      for (i,st) in zip(1:n, subtitles)]
            end
            legend_place = [plot_layout[1,2*i] for i in 1:n]
        elseif get_legend_place === :above
            if subtitles === nothing
                ax = [Axis(plot_layout[2,i]; kwargs...) for i in 1:n]
            else
                ax = [Axis(plot_layout[2,i]; title=st, kwargs...)
                      for (i,st) in zip(1:n, subtitles)]
            end
            legend_place = [plot_layout[1,i] for i in 1:n]
        elseif get_legend_place === :below
            if subtitles === nothing
                ax = [Axis(plot_layout[1,i]; kwargs...) for i in 1:n]
            else
                ax = [Axis(plot_layout[1,i]; title=st, kwargs...)
                      for (i,st) in zip(1:n, subtitles)]
            end
            legend_place = [plot_layout[2,i] for i in 1:n]
        else
            if subtitles === nothing
                ax = [Axis(plot_layout[1,i]; kwargs...) for i in 1:n]
            else
                ax = [Axis(plot_layout[1,i]; title=st, kwargs...)
                      for (i,st) in zip(1:n, subtitles)]
            end
        end
    end

    if get_legend_place === nothing
        return fig, ax
    else
        return fig, ax, legend_place
    end
end

"""
    get_2d_ax(n=nothing; title=nothing, subtitles=nothing, size=nothing, kwargs...)

Create a new `Figure` `fig` and `Axis` `ax` intended for 2d plots.

`title` gives an overall title to the `Figure`.

By default creates a single `Axis`, and returns `(fig, ax, colorbar_place)`, where
`colorbar_place` is a location in the grid layout that can be passed to `Colorbar()`
located immediately to the right of `ax`.
If a number of axes `n` is passed, then `ax` is a `Vector{Axis}` and `colorbar_place` is a
`Vector{GridPosition}` of length `n` (even if `n` is 1). The axes are created in a
horizontal row, and the width of the figure is increased in proportion to `n`.

When `n` is passed, `subtitles` can be passed a Tuple of length `n` which will be used to
set a subtitle for each `Axis` in `ax`.

`size` is passed through to the `Figure` constructor. Its default value is `(600, 400)` if
`n` is not passed, or `(600*n, 400)` if `n` is passed.

Extra `kwargs` are passed to the `Axis()` constructor.
"""
function get_2d_ax(n=nothing; title=nothing, subtitles=nothing, size=nothing, kwargs...)
    if n == nothing
        if size == nothing
            size = (600, 400)
        end
        fig = Figure(size=size)
        if title !== nothing
            title_layout = fig[1,1] = GridLayout()
            Label(title_layout[1,1:2], title)
            irow = 2
        else
            irow = 1
        end
        ax = Axis(fig[irow,1]; kwargs...)
        colorbar_place = fig[irow,2]
    else
        if size == nothing
            size = (600*n, 400)
        end
        fig = Figure(size=size)

        if title !== nothing
            title_layout = fig[1,1] = GridLayout()
            Label(title_layout[1,1:2], title)

            plot_layout = fig[2,1] = GridLayout()
        else
            plot_layout = fig[1,1] = GridLayout()
        end
        if subtitles === nothing
            ax = [Axis(plot_layout[1,2*i-1]; kwargs...) for i in 1:n]
        else
            ax = [Axis(plot_layout[1,2*i-1]; title=st, kwargs...)
                  for (i,st) in zip(1:n, subtitles)]
        end
        colorbar_place = [plot_layout[1,2*i] for i in 1:n]
    end

    return fig, ax, colorbar_place
end

"""
    plot_1d(xcoord, data; ax=nothing, xlabel=nothing, ylabel=nothing, title=nothing,
            yscale=nothing, transform=identity, axis_args=Dict{Symbol,Any}(),
            kwargs...)

Make a 1d plot of `data` vs `xcoord`.

`xlabel`, `ylabel` and `title` can be passed to set axis labels and title for the
(sub-)plot.

`yscale` can be used to set the scaling function for the y-axis. Options are `identity`,
`log`, `log2`, `log10`, `sqrt`, `Makie.logit`, `Makie.pseudolog10` and `Makie.Symlog10`.
`transform` is a function that is applied element-by-element to the data before it is
plotted. For example when using a log scale on data that may contain some negative values
it might be useful to pass `transform=abs` (to plot the absolute value) or
`transform=positive_or_nan` (to ignore any negative or zero values).

If `ax` is passed, the plot will be added to that existing `Axis`, otherwise a new
`Figure` and `Axis` will be created.

`axis_args` are passed as keyword arguments to `get_1d_ax()`, and from there to the `Axis`
constructor.

Other `kwargs` are passed to Makie's `lines!()` function.

If `ax` is not passed, returns the `Figure`, otherwise returns the object returned by
`lines!()`.
"""
function plot_1d(xcoord, data; ax=nothing, xlabel=nothing, ylabel=nothing, title=nothing,
                 yscale=nothing, transform=identity, axis_args=Dict{Symbol,Any}(),
                 kwargs...)
    if ax === nothing
        fig, ax = get_1d_ax(; axis_args...)
    else
        fig = nothing
    end

    if xlabel !== nothing
        ax.xlabel = xlabel
    end
    if ylabel !== nothing
        ax.ylabel = ylabel
    end
    if title !== nothing
        ax.title = title
    end

    if transform !== identity
        # Use transform to allow user to do something like data = abs.(data)
        # Don't actually apply identity transform in case this function is called with
        # `data` being a Makie Observable (in which case transform.(data) would be an
        # error).
        data = transform.(data)
    end

    l = lines!(ax, xcoord, data; kwargs...)

    if yscale !== nothing
        ax.yscale = yscale
    end

    if fig === nothing
        return l
    else
        return fig
    end
end

"""
    plot_2d(xcoord, ycoord, data; ax=nothing, colorbar_place=nothing, xlabel=nothing,
            ylabel=nothing, title=nothing, colormap="reverse_deep",
            colorscale=nothing, transform=identity, axis_args=Dict{Symbol,Any}(),
            kwargs...)

Make a 2d plot of `data` vs `xcoord` and `ycoord`.

`xlabel`, `ylabel` and `title` can be passed to set axis labels and title for the
(sub-)plot.

`colorscale` can be used to set the scaling function for the colors. Options are
`identity`, `log`, `log2`, `log10`, `sqrt`, `Makie.logit`, `Makie.pseudolog10` and
`Makie.Symlog10`. `transform` is a function that is applied element-by-element to the data
before it is plotted. For example when using a log scale on data that may contain some
negative values it might be useful to pass `transform=abs` (to plot the absolute value) or
`transform=positive_or_nan` (to ignore any negative or zero values).

If `ax` is passed, the plot will be added to that existing `Axis`, otherwise a new
`Figure` and `Axis` will be created.

`colormap` is included explicitly because we do some special handling so that extra Makie
functionality can be specified by a prefix to the `colormap` string, rather than the
standard Makie mechanism of creating a struct that modifies the colormap. For example
`Reverse("deep")` can be passed as `"reverse_deep"`. This is useful so that these extra
colormaps can be specified in an input file, but is not needed for interactive use.

When `xcoord` and `ycoord` are both one-dimensional, uses Makie's `heatmap!()` function
for the plot. If either or both of `xcoord` and `ycoord` are two-dimensional, instead uses
[`irregular_heatmap!`](@ref).

`axis_args` are passed as keyword arguments to `get_2d_ax()`, and from there to the `Axis`
constructor.

Other `kwargs` are passed to Makie's `heatmap!()` function.

If `ax` is not passed, returns the `Figure`, otherwise returns the object returned by
`heatmap!()`.
"""
function plot_2d(xcoord, ycoord, data; ax=nothing, colorbar_place=nothing, xlabel=nothing,
                 ylabel=nothing, title=nothing, colormap="reverse_deep",
                 colorscale=nothing, transform=identity, axis_args=Dict{Symbol,Any}(),
                 kwargs...)
    if ax === nothing
        fig, ax, colorbar_place = get_2d_ax(; axis_args...)
    else
        fig = nothing
    end

    if xlabel !== nothing
        ax.xlabel = xlabel
    end
    if ylabel !== nothing
        ax.ylabel = ylabel
    end
    if title !== nothing
        ax.title = title
    end
    colormap = parse_colormap(colormap)
    if colorscale !== nothing
        kwargs = tuple(kwargs..., :colorscale=>colorscale)
    end

    if transform !== identity
        # Use transform to allow user to do something like data = abs.(data)
        # Don't actually apply identity transform in case this function is called with
        # `data` being a Makie Observable (in which case transform.(data) would be an
        # error).
        data = transform.(data)
    end

    if isa(data, AbstractArray)
        datamin, datamax = NaNMath.extrema(data)
        if isnan(datamin) && isnan(datamax)
            datamin = 1.0
            datamax = 1.0
        end
        if datamin == datamax
            # Would error because the color scale has zero size, so pick some arbitrary
            # non-identical limits
            kwargs = tuple(kwargs..., :colorrange=>(datamin - 1.0e-3, datamin + 1.0e-3))
        end
    end

    # Convert grid point values to 'cell face' values for heatmap
    if xcoord isa Observable
        xcoord = lift(grid_points_to_faces, xcoord)
    else
        xcoord = grid_points_to_faces(xcoord)
    end
    if ycoord isa Observable
        ycoord = lift(grid_points_to_faces, ycoord)
    else
        ycoord = grid_points_to_faces(ycoord)
    end

    if xcoord isa Observable
        ndims_x = ndims(xcoord.val)
    else
        ndims_x = ndims(xcoord)
    end
    if ycoord isa Observable
        ndims_y = ndims(ycoord.val)
    else
        ndims_y = ndims(ycoord)
    end
    if ndims_x == 1 && ndims_y == 1
        hm = heatmap!(ax, xcoord, ycoord, data; colormap=colormap, kwargs...)
    else
        hm = irregular_heatmap!(ax, xcoord, ycoord, data; colormap=colormap, kwargs...)
    end

    if colorbar_place === nothing
        println("Warning: colorbar_place argument is required to make a color bar")
    else
        Colorbar(colorbar_place, hm)
    end

    if fig === nothing
        return hm
    else
        return fig
    end
end

"""
    animate_1d(xcoord, data; frame_index=nothing, ax=nothing, fig=nothing,
               xlabel=nothing, ylabel=nothing, title=nothing, yscale=nothing,
               transform=identity, outfile=nothing, ylims=nothing,
               axis_args=Dict{Symbol,Any}(), kwargs...)

Make a 1d animation of `data` vs `xcoord`.

`xlabel`, `ylabel` and `title` can be passed to set axis labels and title for the
(sub-)plot.

`ylims` can be passed a Tuple (ymin, ymax) to set the y-axis limits. By default the
minimum and maximum of the data (over all time points) will be used.

`yscale` can be used to set the scaling function for the y-axis. Options are `identity`,
`log`, `log2`, `log10`, `sqrt`, `Makie.logit`, `Makie.pseudolog10` and `Makie.Symlog10`.
`transform` is a function that is applied element-by-element to the data before it is
plotted. For example when using a log scale on data that may contain some negative values
it might be useful to pass `transform=abs` (to plot the absolute value) or
`transform=positive_or_nan` (to ignore any negative or zero values).

If `ax` is passed, the animation will be added to that existing `Axis`, otherwise a new
`Figure` and `Axis` will be created. If `ax` is passed, you should also pass an
`Observable{mk_int}` to `frame_index` so that the data for this animation can be updated
when `frame_index` is changed.

If `outfile` is passed the animation will be saved to a file with that name. The suffix
determines the file type. If `ax` is passed at the same time as `outfile` then the
`Figure` containing `ax` must also be passed (to the `fig` argument) so that the animation
can be saved.

`axis_args` are passed as keyword arguments to `get_1d_ax()`, and from there to the `Axis`
constructor.

Other `kwargs` are passed to Makie's `lines!()` function.

If `ax` is not passed, returns the `Figure`, otherwise returns the object returned by
`lines!()`.
"""
function animate_1d(xcoord, data; frame_index=nothing, ax=nothing, fig=nothing,
                    xlabel=nothing, ylabel=nothing, title=nothing, yscale=nothing,
                    transform=identity, ylims=nothing, outfile=nothing,
                    axis_args=Dict{Symbol,Any}(), kwargs...)

    if frame_index === nothing
        ind = Observable(1)
    else
        ind = frame_index
    end

    if ax === nothing
        fig, ax = get_1d_ax(; title=title, xlabel=xlabel, ylabel=ylabel, yscale=yscale,
                            axis_args...)
    end

    if !isa(data, VariableCache)
        # Apply transform before calculating extrema
        data = transform.(data)
    end

    if ylims === nothing
        if isa(data, VariableCache)
            datamin, datamax = variable_cache_extrema(data; transform=transform)
        else
            datamin, datamax = NaNMath.extrema(data)
        end
        if isnan(datamin) && isnan(datamax)
            datamin = 1.0
            datamax = 1.0
        end
        if ax.limits.val[2] === nothing
            # No limits set yet, need to use minimum and maximum of data over all time,
            # otherwise the automatic axis scaling would use the minimum and maximum of
            # the data at the initial time point.
            # If datamin==datamax, plot is probably not that interesting, but also the
            # limits do not change with time, so might as well leave limits as whatever
            # the default is.
            if datamin != datamax
                ylims!(ax, datamin, datamax)
            end
        else
            # Expand currently set limits to ensure they include the minimum and maxiumum
            # of the data.
            current_ymin, current_ymax = ax.limits.val[2]
            ylims!(ax, min(datamin, current_ymin), max(datamax, current_ymax))
        end
    else
        # User passed ylims explicitly, so set those.
        ylims!(ax, ylims)
    end

    # Use transform to allow user to do something like data = abs.(data)
    if isa(data, VariableCache)
        line_data = @lift(transform.(get_cache_slice(data, $ind)))
    else
        line_data = @lift(@view data[:,$ind])
    end
    lines!(ax, xcoord, line_data; kwargs...)

    if outfile !== nothing
        if fig === nothing
            error("When `outfile` is passed to save the animation, must either pass both "
                  * "`fig` and `ax` or neither. Only `ax` was passed.")
        end
        nt = size(data, 2)
        save_animation(fig, ind, nt, outfile)
    end
end

"""
    animate_2d(xcoord, ycoord, data; frame_index=nothing, ax=nothing, fig=nothing,
               colorbar_place=nothing, xlabel=nothing, ylabel=nothing, title=nothing,
               outfile=nothing, colormap="reverse_deep", colorscale=nothing,
               transform=identity, axis_args=Dict{Symbol,Any}(), kwargs...)

Make a 2d animation of `data` vs `xcoord` and `ycoord`.

`xlabel`, `ylabel` and `title` can be passed to set axis labels and title for the
(sub-)plot.

`colorscale` can be used to set the scaling function for the colors. Options are
`identity`, `log`, `log2`, `log10`, `sqrt`, `Makie.logit`, `Makie.pseudolog10` and
`Makie.Symlog10`. `transform` is a function that is applied element-by-element to the data
before it is plotted. For example when using a log scale on data that may contain some
negative values it might be useful to pass `transform=abs` (to plot the absolute value) or
`transform=positive_or_nan` (to ignore any negative or zero values).

If `ax` is passed, the animation will be added to that existing `Axis`, otherwise a new
`Figure` and `Axis` will be created. If `ax` is passed, you should also pass an
`Observable{mk_int}` to `frame_index` so that the data for this animation can be updated
when `frame_index` is changed.

If `outfile` is passed the animation will be saved to a file with that name. The suffix
determines the file type. If `ax` is passed at the same time as `outfile` then the
`Figure` containing `ax` must also be passed (to the `fig` argument) so that the animation
can be saved.

`colormap` is included explicitly because we do some special handling so that extra Makie
functionality can be specified by a prefix to the `colormap` string, rather than the
standard Makie mechanism of creating a struct that modifies the colormap. For example
`Reverse("deep")` can be passed as `"reverse_deep"`. This is useful so that these extra
colormaps can be specified in an input file, but is not needed for interactive use.

When `xcoord` and `ycoord` are both one-dimensional, uses Makie's `heatmap!()` function
for the plot. If either or both of `xcoord` and `ycoord` are two-dimensional, instead uses
[`irregular_heatmap!`](@ref).

`axis_args` are passed as keyword arguments to `get_2d_ax()`, and from there to the `Axis`
constructor.

Other `kwargs` are passed to Makie's `heatmap!()` function.

If `ax` is not passed, returns the `Figure`, otherwise returns the object returned by
`heatmap!()`.
"""
function animate_2d(xcoord, ycoord, data; frame_index=nothing, ax=nothing, fig=nothing,
                    colorbar_place=nothing, xlabel=nothing, ylabel=nothing, title=nothing,
                    outfile=nothing, colormap="reverse_deep", colorscale=nothing,
                    transform=identity, axis_args=Dict{Symbol,Any}(), kwargs...)
    colormap = parse_colormap(colormap)

    if ax === nothing
        fig, ax, colorbar_place = get_2d_ax(; title=title, axis_args...)
    end
    if frame_index === nothing
        ind = Observable(1)
    else
        ind = frame_index
    end
    if xlabel !== nothing
        ax.xlabel = xlabel
    end
    if ylabel !== nothing
        ax.ylabel = ylabel
    end
    if colorscale !== nothing
        kwargs = tuple(kwargs..., :colorscale=>colorscale)
    end

    xcoord = grid_points_to_faces(xcoord)
    ycoord = grid_points_to_faces(ycoord)

    # Use transform to allow user to do something like data = abs.(data)
    if colorscale !== nothing
        extrema_check_colorscale = colorscale
    else
        extrema_check_colorscale = identity
    end
    if isa(data, VariableCache)
        datamin, datamax = variable_cache_extrema(data; transform=transform)
        heatmap_data = @lift(transform.(get_cache_slice(data, $ind)))
    else
        datamin, datamax = NaNMath.extrema(data)
        data = transform.(data)
        heatmap_data = @lift(@view data[:,:,$ind])
    end
    if !isfinite(extrema_check_colorscale(datamin)) && !isfinite(extrema_check_colorscale(datamax))
        # Would error because the color scale has zero size, so pick some arbitrary
        # non-identical limits
        kwargs = tuple(kwargs..., :colorrange=>(1.0 - 1.0e-3, 1.0 + 1.0e-3))
    end
    if ndims(xcoord) == 1 && ndims(ycoord) == 1
        hm = heatmap!(ax, xcoord, ycoord, heatmap_data; colormap=colormap, kwargs...)
    else
        hm = irregular_heatmap!(ax, xcoord, ycoord, heatmap_data; colormap=colormap, kwargs...)
    end
    Colorbar(colorbar_place, hm)

    if outfile !== nothing
        if fig === nothing
            error("When `outfile` is passed to save the animation, must either pass both "
                  * "`fig` and `ax` or neither. Only `ax` was passed.")
        end
        nt = size(data, 3)
        save_animation(fig, ind, nt, outfile)
    end

    return fig
end

"""
    plot_f_unnorm_vs_vpa(run_info; input=nothing, electron=false, neutral=false,
                         it=nothing, is=1, iz=nothing, fig=nothing, ax=nothing,
                         outfile=nothing, yscale=identity, transform=identity,
                         axis_args=Dict{Symbol,Any}(), kwargs...)

Plot an unnormalized distribution function against \$v_\\parallel\$ at a fixed z.

This function is only needed for moment-kinetic runs. These are currently only supported
for the 1D1V case.

The information for the runs to plot is passed in `run_info` (as returned by
[`get_run_info`](@ref)). If `run_info` is a Vector, comparison plots are made where plots
from the different runs are overlayed on the same axis.

By default plots the ion distribution function. If `electron=true` is passed, plots the
electron distribution function instead. If `neutral=true` is passed, plots the neutral
distribution function instead.

`is` selects which species to analyse.

`it` and `iz` specify the indices of the time- and z-points to choose. By default they are
taken from `input`.

If `input` is not passed, it is taken from `input_dict_dfns["f"]`.

The data needed will be loaded from file.

If `outfile` is given, the plot will be saved to a file with that name. The suffix
determines the file type.

When `run_info` is not a Vector, an Axis can be passed to `ax` to have the plot added to
`ax`. When `ax` is passed, if `outfile` is passed to save the plot, then the Figure
containing `ax` must be passed to `fig`.

`yscale` can be used to set the scaling function for the y-axis. Options are `identity`,
`log`, `log2`, `log10`, `sqrt`, `Makie.logit`, `Makie.pseudolog10` and `Makie.Symlog10`.
`transform` is a function that is applied element-by-element to the data before it is
plotted. For example when using a log scale on data that may contain some negative values
it might be useful to pass `transform=abs` (to plot the absolute value) or
`transform=positive_or_nan` (to ignore any negative or zero values).

`axis_args` are passed as keyword arguments to `get_1d_ax()`, and from there to the `Axis`
constructor.

Any extra `kwargs` are passed to [`plot_1d`](@ref).
"""
function plot_f_unnorm_vs_vpa end

function plot_f_unnorm_vs_vpa(run_info::Vector{Any}; f_over_vpa2=false, electron=false,
                              neutral=false, outfile=nothing,
                              axis_args=Dict{Symbol,Any}(), kwargs...)
    try
        n_runs = length(run_info)

        species_label = neutral ? "n" : "i"
        divide_by = f_over_vpa2 ? L"/v_\parallel^2" : ""
        ylabel = L"f_{%$species_label,\mathrm{unnormalized}}%$divide_by"
        fig, ax = get_1d_ax(; xlabel=L"v_\parallel", ylabel=ylabel, axis_args...)

        for ri ∈ run_info
            plot_f_unnorm_vs_vpa(ri; f_over_vpa2=f_over_vpa2, electron=electron,
                                 neutral=neutral, ax=ax, kwargs...)
        end

        if n_runs > 1
            put_legend_below(fig, ax)
            # Ensure the first row width is 3/4 of the column width so that
            # the plot does not get squashed by the legend
            rowsize!(fig.layout, 1, Aspect(1, 3/4))
            resize_to_layout!(fig)
        end

        if outfile !== nothing
            save(outfile, fig)
        end

        return fig
    catch e
        return makie_post_processing_error_handler(
                   e,
                   "Error in plot_f_unnorm_vs_vpa().")
    end
end

function plot_f_unnorm_vs_vpa(run_info; f_over_vpa2=false, input=nothing, electron=false,
                              neutral=false, it=nothing, is=1, iz=nothing, fig=nothing,
                              ax=nothing, outfile=nothing, transform=identity,
                              axis_args=Dict{Symbol,Any}(), kwargs...)

    if electron && neutral
        error("does not make sense to pass electron=true and neutral=true at the same "
              * "time")
    end

    if input === nothing
        if neutral
            input = Dict_to_NamedTuple(input_dict_dfns["f_neutral"])
        else
            input = Dict_to_NamedTuple(input_dict_dfns["f"])
        end
    elseif input isa AbstractDict
        input = Dict_to_NamedTuple(input)
    end

    if it == nothing
        it = input.it0
    end
    if iz == nothing
        iz = input.iz0
    end

    if ax === nothing
        species_label = neutral ? "n" : electron ? "e" : "i"
        divide_by = f_over_vpa2 ? L"/v_\parallel^2" : ""
        ylabel = L"f_{%$species_label,\mathrm{unnormalized}}%$divide_by"
        fig, ax = get_1d_ax(; xlabel=L"v_\parallel", ylabel=ylabel, axis_args...)
    end

    if neutral
        f = get_variable(run_info, "f_neutral"; it=it, is=is, ir=input.ir0, iz=iz,
                         ivzeta=input.ivzeta0, ivr=input.ivr0)
        density = get_variable(run_info, "density_neutral"; it=it, is=is, ir=input.ir0,
                               iz=iz)
        upar = get_variable(run_info, "uz_neutral"; it=it, is=is, ir=input.ir0, iz=iz)
        vth = get_variable(run_info, "thermal_speed_neutral"; it=it, is=is, ir=input.ir0,
                           iz=iz)
        vcoord = run_info.vz
    else
        suffix = electron ? "_electron" : ""
        prefix = electron ? "electron_" : ""
        f = get_variable(run_info, "f$suffix"; it=it, is=is, ir=input.ir0, iz=iz,
                         ivperp=input.ivperp0)
        density = get_variable(run_info, "$(prefix)density"; it=it, is=is, ir=input.ir0, iz=iz)
        upar = get_variable(run_info, "$(prefix)parallel_flow"; it=it, is=is, ir=input.ir0, iz=iz)
        vth = get_variable(run_info, "$(prefix)thermal_speed"; it=it, is=is, ir=input.ir0, iz=iz)
        vcoord = run_info.vpa
    end

    f_unnorm, dzdt = get_unnormalised_f_dzdt_1d(f, vcoord.grid, density, upar, vth,
                                                run_info.evolve_density,
                                                run_info.evolve_upar,
                                                run_info.evolve_p)

    if f_over_vpa2
        dzdt2 = dzdt.^2
        for i ∈ eachindex(dzdt2)
            if dzdt2[i] == 0.0
                dzdt2[i] = 1.0
            end
        end
        f_unnorm ./= dzdt2
    end

    f_unnorm = transform.(f_unnorm)

    l = plot_1d(dzdt, f_unnorm; ax=ax, label=run_info.run_name, kwargs...)

    if input.show_element_boundaries && fig !== nothing
        element_boundary_inds =
        [i for i ∈ 1:run_info.vpa.ngrid-1:run_info.vpa.n_global]
        element_boundary_positions = dzdt[element_boundary_inds]
        vlines!(ax, element_boundary_positions, color=:black, alpha=0.3)
    end

    if outfile !== nothing
        if fig === nothing
            error("When ax is passed, fig must also be passed to save the plot using "
                  * "outfile")
        end
        save(outfile, fig)
    end

    if fig !== nothing
        return fig
    else
        return l
    end
end

"""
    plot_f_unnorm_vs_vpa_z(run_info; input=nothing, electron=false, neutral=false,
                           it=nothing, is=1, fig=nothing, ax=nothing, outfile=nothing,
                           yscale=identity, transform=identity, rasterize=true,
                           subtitles=nothing, axis_args=Dict{Symbol,Any}(), kwargs...)

Plot unnormalized distribution function against \$v_\\parallel\$ and z.

This function is only needed for moment-kinetic runs. These are currently only supported
for the 1D1V case.

The information for the runs to plot is passed in `run_info` (as returned by
[`get_run_info`](@ref)). If `run_info` is a Vector, comparison plots are made where plots
from the different runs are displayed in a horizontal row.

By default plots the ion distribution function. If `electron=true` is passed, plots the
electron distribution function instead. If `neutral=true` is passed, plots the neutral
distribution function instead.

`is` selects which species to analyse.

`it` specifies the time-index to choose. By default it is taken from `input`.

If `input` is not passed, it is taken from `input_dict_dfns["f"]`.

The data needed will be loaded from file.

If `outfile` is given, the plot will be saved to a file with that name. The suffix
determines the file type.

When `run_info` is not a Vector, an Axis can be passed to `ax` to have the plot created in
`ax`. When `ax` is passed, if `outfile` is passed to save the plot, then the Figure
containing `ax` must be passed to `fig`.

`yscale` can be used to set the scaling function for the y-axis. Options are `identity`,
`log`, `log2`, `log10`, `sqrt`, `Makie.logit`, `Makie.pseudolog10` and `Makie.Symlog10`.
`transform` is a function that is applied element-by-element to the data before it is
plotted. For example when using a log scale on data that may contain some negative values
it might be useful to pass `transform=abs` (to plot the absolute value) or
`transform=positive_or_nan` (to ignore any negative or zero values).

`rasterize` is passed through to Makie's `mesh!()` function. The default is to rasterize
plots as vectorized plots from `mesh!()` have a very large file size. Pass `false` to keep
plots vectorized. Pass a number to increase the resolution of the rasterized plot by that
factor.

When `run_info` is a Vector, `subtitles` can be passed a Vector (with the same length as
`run_info`) to set the subtitle for each subplot.

`axis_args` are passed as keyword arguments to `get_2d_ax()`, and from there to the `Axis`
constructor.

Any extra `kwargs` are passed to [`plot_2d`](@ref).
"""
function plot_f_unnorm_vs_vpa_z end

function plot_f_unnorm_vs_vpa_z(run_info::Vector{Any}; electron=false, neutral=false,
                                outfile=nothing, axis_args=Dict{Symbol,Any}(),
                                title=nothing, subtitles=nothing, kwargs...)
    try
        n_runs = length(run_info)
        if subtitles === nothing
            subtitles = [nothing for _ ∈ 1:n_runs]
        end
        if title !== nothing
            title = neutral ? L"f_{n,\mathrm{unnormalized}}" : electron ? L"f_{e,\mathrm{unnormalized}}" : L"f_{i,\mathrm{unnormalized}}"
        end
        fig, axes, colorbar_places =
            get_2d_ax(n_runs; title=title, xlabel=L"v_\parallel", ylabel=L"z",
                      axis_args...)

        for (ri, ax, colorbar_place, st) ∈ zip(run_info, axes, colorbar_places, subtitles)
            plot_f_unnorm_vs_vpa_z(ri; electron=electron, neutral=neutral, ax=ax,
                                   colorbar_place=colorbar_place, title=st, kwargs...)
        end

        if outfile !== nothing
            save(outfile, fig)
        end

        return fig
    catch e
        return makie_post_processing_error_handler(
                   e,
                   "Error in plot_f_unnorm_vs_vpa_z().")
    end
end

function plot_f_unnorm_vs_vpa_z(run_info; input=nothing, electron=false, neutral=false,
                                it=nothing, is=1, fig=nothing, ax=nothing,
                                colorbar_place=nothing, title=nothing, outfile=nothing,
                                transform=identity, rasterize=true,
                                axis_args=Dict{Symbol,Any}(), kwargs...)

    if electron && neutral
        error("does not make sense to pass electron=true and neutral=true at the same "
              * "time")
    end

    if input === nothing
        if neutral
            input = Dict_to_NamedTuple(input_dict_dfns["f_neutral"])
        else
            input = Dict_to_NamedTuple(input_dict_dfns["f"])
        end
    elseif input isa AbstractDict
        input = Dict_to_NamedTuple(input)
    end

    if it == nothing
        it = input.it0
    end

    if ax === nothing
        if title === nothing
            title = neutral ? L"f_{n,\mathrm{unnormalized}}" : electron ? L"f_{e,\mathrm{unnormalized}}" : L"f_{i,\mathrm{unnormalized}}"
        end
        fig, ax, colorbar_place = get_2d_ax(; title=title, xlabel=L"v_\parallel",
                                            ylabel=L"z", axis_args...)
    else
        if title === nothing
            ax.title = run_info.run_name
        else
            ax.title = title
        end
    end

    if neutral
        f = get_variable(run_info, "f_neutral"; it=it, is=is, ir=input.ir0,
                         ivzeta=input.ivzeta0, ivr=input.ivr0)
        density = get_variable(run_info, "density_neutral"; it=it, is=is, ir=input.ir0)
        upar = get_variable(run_info, "uz_neutral"; it=it, is=is, ir=input.ir0)
        vth = get_variable(run_info, "thermal_speed_neutral"; it=it, is=is, ir=input.ir0)
        vpa_grid = run_info.vz.grid
    else
        suffix = electron ? "_electron" : ""
        prefix = electron ? "electron_" : ""
        f = get_variable(run_info, "f$suffix"; it=it, is=is, ir=input.ir0, ivperp=input.ivperp0)
        density = get_variable(run_info, "$(prefix)density"; it=it, is=is, ir=input.ir0)
        upar = get_variable(run_info, "$(prefix)parallel_flow"; it=it, is=is, ir=input.ir0)
        vth = get_variable(run_info, "$(prefix)thermal_speed"; it=it, is=is, ir=input.ir0)
        vpa_grid = run_info.vpa.grid
    end

    f_unnorm, z, dzdt = get_unnormalised_f_coords_2d(f, run_info.z.grid,
                                                     vpa_grid, density, upar,
                                                     vth, run_info.evolve_density,
                                                     run_info.evolve_upar,
                                                     run_info.evolve_p)

    f_unnorm = transform.(f_unnorm)

    # Rasterize the plot, otherwise the output files are very large
    hm = plot_2d(dzdt, z, f_unnorm; ax=ax, colorbar_place=colorbar_place,
                 rasterize=rasterize, kwargs...)

    if outfile !== nothing
        if fig === nothing
            error("When ax is passed, fig must also be passed to save the plot using "
                  * "outfile")
        end
        save(outfile, fig)
    end

    if fig !== nothing
        return fig
    else
        return hm
    end
end

"""
    animate_f_unnorm_vs_vpa(run_info; input=nothing, electron=false, neutral=false, is=1,
                            iz=nothing, fig=nothing, ax=nothing, frame_index=nothing,
                            outfile=nothing, yscale=identity, transform=identity,
                            axis_args=Dict{Symbol,Any}(), kwargs...)

Plot an unnormalized distribution function against \$v_\\parallel\$ at a fixed z.

This function is only needed for moment-kinetic runs. These are currently only supported
for the 1D1V case.

The information for the runs to animate is passed in `run_info` (as returned by
[`get_run_info`](@ref)). If `run_info` is a Vector, comparison plots are made where plots
from the different runs are overlayed on the same axis.

By default animates the ion distribution function. If `electron=true` is passed, animates
the electron distribution function instead. If `neutral=true` is passed, animates the
neutral distribution function instead.

`is` selects which species to analyse.

`it` and `iz` specify the indices of the time- and z-points to choose. By default they are
taken from `input`.

If `input` is not passed, it is taken from `input_dict_dfns["f"]`.

The data needed will be loaded from file.

`outfile` is required for animations unless `ax` is passed. The animation will be saved to
a file named `outfile`.  The suffix determines the file type. If both `outfile` and `ax`
are passed, then the `Figure` containing `ax` must be passed to `fig` to allow the
animation to be saved.

When `run_info` is not a Vector, an Axis can be passed to `ax` to have the plot added to
`ax`. When `ax` is passed, if `outfile` is passed to save the plot, then the Figure
containing `ax` must be passed to `fig`.

`yscale` can be used to set the scaling function for the y-axis. Options are `identity`,
`log`, `log2`, `log10`, `sqrt`, `Makie.logit`, `Makie.pseudolog10` and `Makie.Symlog10`.
`transform` is a function that is applied element-by-element to the data before it is
plotted. For example when using a log scale on data that may contain some negative values
it might be useful to pass `transform=abs` (to plot the absolute value) or
`transform=positive_or_nan` (to ignore any negative or zero values).

`axis_args` are passed as keyword arguments to `get_1d_ax()`, and from there to the `Axis`
constructor.

Any extra `kwargs` are passed to `lines!()` (which is used to create the plot, as we have
to handle time-varying coordinates so cannot use [`animate_1d`](@ref)).
"""
function animate_f_unnorm_vs_vpa end

function animate_f_unnorm_vs_vpa(run_info::Vector{Any}; f_over_vpa2=false, electron=false,
                                 neutral=false, outfile=nothing,
                                 axis_args=Dict{Symbol,Any}(), kwargs...)
    try
        n_runs = length(run_info)

        frame_index = Observable(1)

        species_label = neutral ? "n" : electron ? "e" : "i"
        divide_by = f_over_vpa2 ? L"/v_\parallel^2" : ""
        ylabel = L"f_{%$species_label,\mathrm{unnormalized}}%$divide_by"
        if length(run_info) == 1 || all(all(isapprox.(ri.time, run_info[1].time)) for ri ∈ run_info[2:end])
            # All times are the same
            title = lift(i->LaTeXString(string("t = ", run_info[1].time[i])), frame_index)
        else
            title = lift(i->LaTeXString(join((string("t", irun, " = ", ri.time[i])
                                              for (irun,ri) ∈ enumerate(run_info)), "; ")),
                         frame_index)
        end
        fig, ax = get_1d_ax(; xlabel=L"v_\parallel", ylabel=ylabel, title=title,
                            axis_args...)

        for ri ∈ run_info
            animate_f_unnorm_vs_vpa(ri; f_over_vpa2=f_over_vpa2, electron=electron,
                                    neutral=neutral, ax=ax, frame_index=frame_index,
                                    kwargs...)
        end

        if n_runs > 1
            put_legend_below(fig, ax)
            # Ensure the first row width is 3/4 of the column width so that
            # the plot does not get squashed by the legend
            rowsize!(fig.layout, 1, Aspect(1, 3/4))
            resize_to_layout!(fig)
        end

        if outfile !== nothing
            nt = minimum(ri.nt for ri ∈ run_info)
            save_animation(fig, frame_index, nt, outfile)
        end

        return fig
    catch e
        return makie_post_processing_error_handler(
                   e,
                   "Error in animate_f_unnorm_vs_vpa().")
    end
end

function animate_f_unnorm_vs_vpa(run_info; f_over_vpa2=false, input=nothing,
                                 electron=false, neutral=false, is=1, iz=nothing,
                                 fig=nothing, ax=nothing, frame_index=nothing,
                                 outfile=nothing, yscale=nothing, transform=identity,
                                 axis_args=Dict{Symbol,Any}(), kwargs...)

    if electron && neutral
        error("does not make sense to pass electron=true and neutral=true at the same "
              * "time")
    end

    if input === nothing
        if neutral
            input = Dict_to_NamedTuple(input_dict_dfns["f_neutral"])
        else
            input = Dict_to_NamedTuple(input_dict_dfns["f"])
        end
    elseif input isa AbstractDict
        input = Dict_to_NamedTuple(input)
    end

    if iz == nothing
        iz = input.iz0
    end

    if ax === nothing
        frame_index = Observable(1)
        title = lift(i->LaTeXString(string("t = ", run_info.time[i])), frame_index)
        species_label = neutral ? "n" : "i"
        divide_by = f_over_vpa2 ? L"/v_\parallel^2" : ""
        ylabel = L"f_{%$species_label,\mathrm{unnormalized}}%$divide_by"
        fig, ax = get_1d_ax(; xlabel=L"v_\parallel", ylabel=ylabel, title=title,
                            axis_args...)
    end
    if frame_index === nothing
        error("Must pass an Observable to `frame_index` when passing `ax`.")
    end

    if neutral
        f = VariableCache(run_info, "f_neutral", chunk_size_1d; it=nothing, is=is,
                          ir=input.ir0, iz=iz, ivperp=nothing, ivpa=nothing,
                          ivzeta=input.ivzeta0, ivr=input.ivr0, ivz=nothing)
        density = get_variable(run_info, "density_neutral"; is=is, ir=input.ir0, iz=iz)
        upar = get_variable(run_info, "uz_neutral"; is=is, ir=input.ir0, iz=iz)
        vth = get_variable(run_info, "thermal_speed_neutral"; is=is, ir=input.ir0, iz=iz)
        vcoord = run_info.vz
    else
        suffix = electron ? "_electron" : ""
        prefix = electron ? "electron_" : ""
        f = VariableCache(run_info, "f$suffix", chunk_size_2d; it=nothing, is=is,
                          ir=input.ir0, iz=iz, ivperp=input.ivperp0, ivpa=nothing,
                          ivzeta=nothing, ivr=nothing, ivz=nothing)
        density = get_variable(run_info, "$(prefix)density"; is=is, ir=input.ir0, iz=iz)
        upar = get_variable(run_info, "$(prefix)parallel_flow"; is=is, ir=input.ir0, iz=iz)
        vth = get_variable(run_info, "$(prefix)thermal_speed"; is=is, ir=input.ir0, iz=iz)
        vcoord = run_info.vpa
    end

    function get_this_f_unnorm(it)
        f_unnorm = get_unnormalised_f_1d(get_cache_slice(f, it), density[it], vth[it],
                                         run_info.evolve_density, run_info.evolve_p)

        if f_over_vpa2
            # We actually want v_∥ here, not v_z, so pass bz=1, vEz=0
            this_dzdt = vpagrid_to_dzdt(vcoord.grid, vth[it], upar[it], 1.0, 0.0,
                                        run_info.evolve_p, run_info.evolve_upar)
            this_dzdt2 = this_dzdt.^2
            for i ∈ eachindex(this_dzdt2)
                if this_dzdt2[i] == 0.0
                    this_dzdt2[i] = 1.0
                end
            end

            f_unnorm = @. copy(f_unnorm) / this_dzdt2
        end

        return f_unnorm
    end

    # Get extrema of dzdt
    dzdtmin = Inf
    dzdtmax = -Inf
    fmin = Inf
    fmax = -Inf
    for it ∈ 1:run_info.nt
        # We actually want v_∥ here, not v_z, so pass bz=1, vEz=0
        this_dzdt = vpagrid_to_dzdt(vcoord.grid, vth[it], upar[it], 1.0, 0.0,
                                    run_info.evolve_p, run_info.evolve_upar)
        this_dzdtmin, this_dzdtmax = extrema(this_dzdt)
        dzdtmin = min(dzdtmin, this_dzdtmin)
        dzdtmax = max(dzdtmax, this_dzdtmax)

        this_f_unnorm = get_this_f_unnorm(it)

        this_fmin, this_fmax = NaNMath.extrema(transform.(this_f_unnorm))
        fmin = min(fmin, this_fmin)
        fmax = max(fmax, this_fmax)
    end
    if isnan(fmin) && isnan(fmax)
        fmin = 1.0
        fmax = 1.0
    end
    yheight = fmax - fmin
    xwidth = dzdtmax - dzdtmin
    if yscale ∈ (log, log10)
        # Need to calclutate y offsets differently to non-logarithmic y-axis case, to
        # ensure ymin is not negative.
        limits!(ax, dzdtmin - 0.01*xwidth, dzdtmax + 0.01*xwidth,
                fmin * (fmin/fmax)^0.01, fmax * (fmax/fmin)^0.01)
    else
        limits!(ax, dzdtmin - 0.01*xwidth, dzdtmax + 0.01*xwidth,
                fmin - 0.01*yheight, fmax + 0.01*yheight)
    end

    # We actually want v_∥ here, not v_z, so pass bz=1, vEz=0
    dzdt = @lift vpagrid_to_dzdt(vcoord.grid, vth[$frame_index], upar[$frame_index], 1.0,
                                 0.0, run_info.evolve_p, run_info.evolve_upar)
    f_unnorm = @lift transform.(get_this_f_unnorm($frame_index))

    l = plot_1d(dzdt, f_unnorm; ax=ax, label=run_info.run_name, yscale=yscale, kwargs...)

    if input.show_element_boundaries && fig !== nothing
        element_boundary_inds =
        [i for i ∈ 1:run_info.vpa.ngrid-1:run_info.vpa.n_global]
        element_boundary_positions = @lift $dzdt[element_boundary_inds]
        vlines!(ax, element_boundary_positions, color=:black, alpha=0.3)
    end


    if outfile !== nothing
        if fig === nothing
            error("When ax is passed, fig must also be passed to save the plot using "
                  * "outfile")
        end
        save_animation(fig, frame_index, run_info.nt, outfile)
    end

    if fig !== nothing
        return fig
    else
        return l
    end
end

"""
    animate_f_unnorm_vs_vpa_z(run_info; input=nothing, electron=false, neutral=false,
                              is=1, fig=nothing, ax=nothing, frame_index=nothing,
                              outfile=nothing, yscale=identity, transform=identity,
                              axis_args=Dict{Symbol,Any}(), kwargs...)

Animate an unnormalized distribution function against \$v_\\parallel\$ and z.

This function is only needed for moment-kinetic runs. These are currently only supported
for the 1D1V case.

The information for the runs to plot is passed in `run_info` (as returned by
[`get_run_info`](@ref)). If `run_info` is a Vector, comparison plots are made where plots
from the different runs are displayed in a horizontal row.

By default animates the ion distribution function. If `electron=true` is passed, animates
the electron distribution function instead. If `neutral=true` is passed, animates the
neutral distribution function instead.

`is` selects which species to analyse.

If `input` is not passed, it is taken from `input_dict_dfns["f"]`.

The data needed will be loaded from file.

`outfile` is required for animations unless `ax` is passed. The animation will be saved to
a file named `outfile`.  The suffix determines the file type. If both `outfile` and `ax`
are passed, then the `Figure` containing `ax` must be passed to `fig` to allow the
animation to be saved.

When `run_info` is not a Vector, an Axis can be passed to `ax` to have the animation
created in `ax`. When `ax` is passed, if `outfile` is passed to save the animation, then
the Figure containing `ax` must be passed to `fig`.

`yscale` can be used to set the scaling function for the y-axis. Options are `identity`,
`log`, `log2`, `log10`, `sqrt`, `Makie.logit`, `Makie.pseudolog10` and `Makie.Symlog10`.
`transform` is a function that is applied element-by-element to the data before it is
plotted. For example when using a log scale on data that may contain some negative values
it might be useful to pass `transform=abs` (to plot the absolute value) or
`transform=positive_or_nan` (to ignore any negative or zero values).

`axis_args` are passed as keyword arguments to `get_2d_ax()`, and from there to the `Axis`
constructor.

Any extra `kwargs` are passed to [`plot_2d`](@ref) (which is used to create the plot, as
we have to handle time-varying coordinates so cannot use [`animate_2d`](@ref)).
"""
function animate_f_unnorm_vs_vpa_z end

function animate_f_unnorm_vs_vpa_z(run_info::Vector{Any}; electron=false, neutral=false,
                                   outfile=nothing, axis_args=Dict{Symbol,Any}(),
                                   kwargs...)
    try
        n_runs = length(run_info)

        frame_index = Observable(1)

        var_name = neutral ? L"f_{n,\mathrm{unnormalized}}" : electron ? L"f_{e,\mathrm{unnormalized}}" : L"f_{i,\mathrm{unnormalized}}"
        if length(run_info) > 1
            title = var_name
            subtitles = (lift(i->LaTeXString(string(ri.run_name, "\nt = ", ri.time[i])),
                              frame_index)
                         for ri ∈ run_info)
        else
            title = lift(i->LaTeXString(string(var_name, L",\;t = ",
                                               run_info[1].time[i])),
                         frame_index)
            subtitles = nothing
        end
        fig, axes, colorbar_places = get_2d_ax(n_runs; title=title, subtitles=subtitles,
                                               xlabel=L"v_\parallel", ylabel=L"z",
                                               axis_args...)

        for (ri, ax, colorbar_place) ∈ zip(run_info, axes, colorbar_places)
            animate_f_unnorm_vs_vpa_z(ri; electron=electron, neutral=neutral, ax=ax,
                                      colorbar_place=colorbar_place, frame_index=frame_index,
                                      kwargs...)
        end

        if outfile !== nothing
            nt = minimum(ri.nt for ri ∈ run_info)
            save_animation(fig, frame_index, nt, outfile)
        end

        return fig
    catch e
        return makie_post_processing_error_handler(
                   e,
                   "Error in animate_f_unnorm_vs_vpa_z().")
    end
end

function animate_f_unnorm_vs_vpa_z(run_info; input=nothing, electron=false, neutral=false,
                                   is=1, fig=nothing, ax=nothing, colorbar_place=nothing,
                                   frame_index=nothing, outfile=nothing,
                                   transform=identity, axis_args=Dict{Symbol,Any}(),
                                   kwargs...)

    if electron && neutral
        error("does not make sense to pass electron=true and neutral=true at the same "
              * "time")
    end

    if input === nothing
        if neutral
            input = Dict_to_NamedTuple(input_dict_dfns["f_neutral"])
        else
            input = Dict_to_NamedTuple(input_dict_dfns["f"])
        end
    elseif input isa AbstractDict
        input = Dict_to_NamedTuple(input)
    end

    if ax === nothing
        frame_index = Observable(1)
        var_name = neutral ? L"f_{n,\mathrm{unnormalized}}" : L"f_{i,\mathrm{unnormalized}}"
        title = lift(i->LaTeXString(string(var_name, "\nt = ", run_info.time[i])),
                     frame_index)
        fig, ax, colorbar_place = get_2d_ax(; title=title, xlabel=L"v_\parallel",
                                            ylabel=L"z", axis_args...)
    end
    if frame_index === nothing
        error("Must pass an Observable to `frame_index` when passing `ax`.")
    end

    if neutral
        f = VariableCache(run_info, "f_neutral", chunk_size_2d; it=nothing, is=is,
                          ir=input.ir0, iz=nothing, ivperp=nothing, ivpa=nothing,
                          ivzeta=input.ivzeta0, ivr=input.ivr0, ivz=nothing)
        density = VariableCache(run_info, "density_neutral", chunk_size_1d; it=nothing,
                                is=is, ir=input.ir0, iz=nothing, ivperp=nothing,
                                ivpa=nothing, ivzeta=nothing, ivr=nothing, ivz=nothing)
        upar = VariableCache(run_info, "uz_neutral", chunk_size_1d; it=nothing, is=is,
                             ir=input.ir0, iz=nothing, ivperp=nothing, ivpa=nothing,
                             ivzeta=nothing, ivr=nothing, ivz=nothing)
        vth = VariableCache(run_info, "thermal_speed_neutral", chunk_size_1d; it=nothing,
                            is=is, ir=input.ir0, iz=nothing, ivperp=nothing, ivpa=nothing,
                            ivzeta=nothing, ivr=nothing, ivz=nothing)
        vpa_grid = run_info.vz.grid
    else
        suffix = electron ? "_electron" : ""
        prefix = electron ? "electron_" : ""
        f = VariableCache(run_info, "f$suffix", chunk_size_2d; it=nothing, is=is,
                          ir=input.ir0, iz=nothing, ivperp=input.ivperp0, ivpa=nothing,
                          ivzeta=nothing, ivr=nothing, ivz=nothing)
        density = VariableCache(run_info, "$(prefix)density", chunk_size_1d; it=nothing,
                                is=is, ir=input.ir0, iz=nothing, ivperp=nothing,
                                ivpa=nothing, ivzeta=nothing, ivr=nothing, ivz=nothing)
        upar = VariableCache(run_info, "$(prefix)parallel_flow", chunk_size_1d;
                             it=nothing, is=is, ir=input.ir0, iz=nothing, ivperp=nothing,
                             ivpa=nothing, ivzeta=nothing, ivr=nothing, ivz=nothing)
        vth = VariableCache(run_info, "$(prefix)thermal_speed", chunk_size_1d; it=nothing,
                            is=is, ir=input.ir0, iz=nothing, ivperp=nothing, ivpa=nothing,
                            ivzeta=nothing, ivr=nothing, ivz=nothing)
        vpa_grid = run_info.vpa.grid
    end

    # Get extrema of dzdt
    dzdtmin = Inf
    dzdtmax = -Inf
    for it ∈ 1:run_info.nt
        this_dzdt = vpagrid_to_v_parallel_2d(vpa_grid, get_cache_slice(vth, it),
                                             get_cache_slice(upar, it), run_info.evolve_p,
                                             run_info.evolve_upar)
        this_dzdtmin, this_dzdtmax = extrema(this_dzdt)
        dzdtmin = min(dzdtmin, this_dzdtmin)
        dzdtmax = max(dzdtmax, this_dzdtmax)
    end
    # Set x-limits of ax so that plot always fits within axis
    xlims!(ax, dzdtmin, dzdtmax)

    dzdt = @lift vpagrid_to_v_parallel_2d(vpa_grid, get_cache_slice(vth, $frame_index),
                                          get_cache_slice(upar, $frame_index),
                                          run_info.evolve_p, run_info.evolve_upar)
    f_unnorm = @lift transform.(get_unnormalised_f_2d(
                                    get_cache_slice(f, $frame_index),
                                    get_cache_slice(density, $frame_index),
                                    get_cache_slice(vth, $frame_index),
                                    run_info.evolve_density, run_info.evolve_p))

    hm = plot_2d(dzdt, run_info.z.grid, f_unnorm; ax=ax, colorbar_place=colorbar_place,
                 kwargs...)

    if outfile !== nothing
        if fig === nothing
            error("When ax is passed, fig must also be passed to save the plot using "
                  * "outfile")
        end
        save_animation(fig, frame_index, run_info.nt, outfile)
    end

    if fig !== nothing
        return fig
    else
        return hm
    end
end

"""
    save_animation(fig, frame_index, nt, outfile)

Animate `fig` and save the result in `outfile`.

`frame_index` is the `Observable{mk_int}` that updates the data used to make `fig` to a
new time point. `nt` is the total number of time points to create.

The suffix of `outfile` determines the file type.
"""
function save_animation(fig, frame_index, nt, outfile)
    record(fig, outfile, 1:nt, framerate=5) do it
        frame_index[] = it
    end
    return nothing
end

"""
   put_legend_above(fig, ax; kwargs...)

Add a legend corresponding to the plot in `ax` to `fig` on the left of a new row at the
top of the figure layout.

Additional `kwargs` are passed to the `Legend()` constructor.
"""
function put_legend_above(fig, ax; kwargs...)
    return Legend(fig[0,1], ax; tellheight=true, tellwidth=false, kwargs...)
end

"""
   put_legend_below(fig, ax; kwargs...)

Add a legend corresponding to the plot in `ax` to `fig` on the left of a new row at the
bottom of the figure layout.

Additional `kwargs` are passed to the `Legend()` constructor.
"""
function put_legend_below(fig, ax; kwargs...)
    return Legend(fig[end+1,1], ax; tellheight=true, tellwidth=false, kwargs...)
end

"""
   put_legend_left(fig, ax; kwargs...)

Add a legend corresponding to the plot in `ax` to `fig` on the bottom of a new column at
the left of the figure layout.

Additional `kwargs` are passed to the `Legend()` constructor.
"""
function put_legend_left(fig, ax; kwargs...)
    return Legend(fig[end,0], ax; kwargs...)
end

"""
   put_legend_right(fig, ax; kwargs...)

Add a legend corresponding to the plot in `ax` to `fig` on the bottom of a new column at
the right of the figure layout.

Additional `kwargs` are passed to the `Legend()` constructor.
"""
function put_legend_right(fig, ax; kwargs...)
    return Legend(fig[end,end+1], ax; kwargs...)
end

"""
    curvilinear_grid_mesh(xs, ys, zs, colors)

Tesselates the grid defined by `xs` and `ys` in order to form a mesh with per-face coloring
given by `colors`.

The grid defined by `xs` and `ys` must have dimensions `(nx, ny) == size(colors) .+ 1`, as
is the case for heatmap/image.

Code from: https://github.com/MakieOrg/Makie.jl/issues/742#issuecomment-1415809653
"""
function curvilinear_grid_mesh(xs, ys, zs, colors)
    if zs isa Observable
        nx, ny = size(zs.val)
    else
        nx, ny = size(zs)
    end
    if colors isa Observable
        ni, nj = size(colors.val)
        eltype_colors = eltype(colors.val)
    else
        ni, nj = size(colors)
        eltype_colors = eltype(colors)
    end
    @assert (nx == ni+1) & (ny == nj+1) "Expected nx, ny = ni+1, nj+1; got nx=$nx, ny=$ny, ni=$ni, nj=$nj.  nx/y are size(zs), ni/j are size(colors)."
    if xs isa Observable && ys isa Observable && zs isa Observable
        input_points_vec = lift((x, y, z)->Makie.matrix_grid(identity, x, y, z), xs, ys, zs)
    elseif xs isa Observable && ys isa Observable
        input_points_vec = lift((x, y)->Makie.matrix_grid(identity, x, y, zs), xs, ys)
    elseif ys isa Observable && zs isa Observable
        input_points_vec = lift((y, z)->Makie.matrix_grid(identity, xs, y, z), ys, zs)
    elseif xs isa Observable && zs isa Observable
        input_points_vec = lift((x, z)->Makie.matrix_grid(identity, x, ys, z), xs, zs)
    elseif xs isa Observable
        input_points_vec = lift(x->Makie.matrix_grid(identity, x, ys, zs), xs)
    elseif ys isa Observable
        input_points_vec = lift(y->Makie.matrix_grid(identity, xs, y, zs), ys)
    elseif zs isa Observable
        input_points_vec = lift(z->Makie.matrix_grid(identity, xs, ys, z), zs)
    else
        input_points_vec = Makie.matrix_grid(identity, xs, ys, zs)
    end
    if input_points_vec isa Observable
        input_points = lift(x->reshape(x, (ni, nj) .+ 1), input_points_vec)
    else
        input_points = reshape(input_points_vec, (ni, nj) .+ 1)
    end

    n_input_points = (ni + 1) * (nj + 1)

    function get_triangle_points(input_points)
        triangle_points = Vector{Point3f}()
        sizehint!(triangle_points, n_input_points * 2 * 3)
        @inbounds for j in 1:nj
            for i in 1:ni
                # push two triangles to make a square
                # first triangle
                push!(triangle_points, input_points[i, j])
                push!(triangle_points, input_points[i+1, j])
                push!(triangle_points, input_points[i+1, j+1])
                # second triangle
                push!(triangle_points, input_points[i+1, j+1])
                push!(triangle_points, input_points[i, j+1])
                push!(triangle_points, input_points[i, j])
            end
        end
        return triangle_points
    end
    if input_points isa Observable
        triangle_points = lift(get_triangle_points, input_points)
    else
        triangle_points = get_triangle_points(input_points)
    end

    function get_triangle_colors(colors)
        triangle_colors = Vector{eltype_colors}()
        sizehint!(triangle_colors, n_input_points * 2 * 3)
        @inbounds for j in 1:nj
            for i in 1:ni
                # push two triangles to make a square
                # first triangle
                push!(triangle_colors, colors[i, j]); push!(triangle_colors, colors[i, j]); push!(triangle_colors, colors[i, j])
                # second triangle
                push!(triangle_colors, colors[i, j]); push!(triangle_colors, colors[i, j]); push!(triangle_colors, colors[i, j])
            end
        end
        return triangle_colors
    end
    if colors isa Observable
        triangle_colors = lift(get_triangle_colors, colors)
    else
        triangle_colors = get_triangle_colors(colors)
    end

    # Triangle faces is a constant vector of indices. Note this depends on the loop
    # structure here being the same as that in get_triangle_points() and
    # get_triangle_colors()
    triangle_faces = Vector{CairoMakie.Makie.GeometryBasics.TriangleFace{UInt32}}()
    sizehint!(triangle_faces, n_input_points * 2)
    point_ind = 1
    @inbounds for j in 1:nj
        for i in 1:ni
            # push two triangles to make a square
            # first triangle
            push!(triangle_faces, CairoMakie.Makie.GeometryBasics.TriangleFace{UInt32}((point_ind, point_ind+1, point_ind+2)))
            point_ind += 3
            # second triangle
            push!(triangle_faces, CairoMakie.Makie.GeometryBasics.TriangleFace{UInt32}((point_ind, point_ind+1, point_ind+2)))
            point_ind += 3
        end
    end

    return triangle_points, triangle_faces, triangle_colors
end

"""
    irregular_heatmap(xs, ys, zs; kwargs...)

Plot a heatmap where `xs` and `ys` are allowed to define irregularly spaced, 2d grids.
`zs` gives the value in each cell of the grid.

The grid defined by `xs` and `ys` must have dimensions `(nx, ny) == size(zs) .+ 1`, as
is the case for heatmap/image.

`xs` be an array of size (nx,ny) or a vector of size (nx).

`ys` be an array of size (nx,ny) or a vector of size (ny).

`kwargs` are passed to Makie's `mesh()` function.

Code adapted from: https://github.com/MakieOrg/Makie.jl/issues/742#issuecomment-1415809653
"""
function irregular_heatmap(xs, ys, zs; kwargs...)
    fig = Figure()
    ax = Axis(fig[1,1])
    hm = irregular_heatmap!(ax, xs, ys, zs; kwargs...)

    return fig, ax, hm
end

"""
    irregular_heatmap!(ax, xs, ys, zs; kwargs...)

Plot a heatmap onto the Axis `ax` where `xs` and `ys` are allowed to define irregularly
spaced, 2d grids.  `zs` gives the value in each cell of the grid.

The grid defined by `xs` and `ys` must have dimensions `(nx, ny) == size(zs) .+ 1`, as
is the case for heatmap/image.

`xs` be an array of size (nx,ny) or a vector of size (nx).

`ys` be an array of size (nx,ny) or a vector of size (ny).

`kwargs` are passed to Makie's `mesh()` function.

Code adapted from: https://github.com/MakieOrg/Makie.jl/issues/742#issuecomment-1415809653
"""
function irregular_heatmap!(ax, xs, ys, zs; kwargs...)
    if xs isa Observable
        ndims_x = ndims(xs.val)
        if ndims_x == 1
            nx = length(xs.val)
        else
            nx = size(xs.val, 1)
        end
    else
        ndims_x = ndims(xs)
        if ndims(xs) == 1
            nx = length(xs)
        else
            nx = size(xs, 1)
        end
    end
    if ys isa Observable
        ndims_y = ndims(ys.val)
        if ndims_y == 1
            ny = length(ys.val)
        else
            ny = size(ys.val, 2)
        end
    else
        ndims_y = ndims(ys)
        if ndims_y == 1
            ny = length(ys)
        else
            ny = size(ys, 2)
        end
    end

    if zs isa Observable
        ni, nj = size(zs.val)
    else
        ni, nj = size(zs)
    end
    @assert (nx == ni+1) & (ny == nj+1) "Expected nx, ny = ni+1, nj+1; got nx=$nx, ny=$ny, ni=$ni, nj=$nj.  nx/y are size(xs)/size(ys), ni/j are size(zs)."

    if ndims_x == 1
        # Copy to an array of size (nx,ny)
        if xs isa Observable
            xs = lift(x->repeat(x, 1, ny), x)
        else
            xs = repeat(xs, 1, ny)
        end
    end
    if ndims_y == 1
        # Copy to an array of size (nx,ny)
        if ys isa Observable
            ys = lift(x->repeat(x', nx, 1), ys)
        else
            ys = repeat(ys', nx, 1)
        end
    end

    vertices, faces, colors = curvilinear_grid_mesh(xs, ys, zeros(nx, ny), zs)

    return mesh!(ax, vertices, faces; color = colors, shading = NoShading, kwargs...)
end

"""
    grid_points_to_faces(coord::AbstractVector)
    grid_points_to_faces(coord::Observable{T} where T <: AbstractVector)
    grid_points_to_faces(coord::AbstractMatrix)
    grid_points_to_faces(coord::Observable{T} where T <: AbstractMatrix)

Turn grid points in `coord` into 'cell faces'.

Returns `faces`, which has a length one greater than `coord`. The first and last values of
`faces` are the first and last values of `coord`. The intermediate values are the mid
points between grid points.
"""
function grid_points_to_faces end

function grid_points_to_faces(coord::AbstractVector)
    n = length(coord)
    faces = allocate_float(n+1)
    faces[1] = coord[1]
    for i ∈ 2:n
        faces[i] = 0.5*(coord[i-1] + coord[i])
    end
    faces[n+1] = coord[n]

    return faces
end

function grid_points_to_faces(coord::Observable{T} where T <: AbstractVector)
    n = length(coord.val)
    faces = allocate_float(n+1)
    faces[1] = coord.val[1]
    for i ∈ 2:n
        faces[i] = 0.5*(coord.val[i-1] + coord.val[i])
    end
    faces[n+1] = coord.val[n]

    return faces
end

function grid_points_to_faces(coord::AbstractMatrix)
    ni, nj = size(coord)
    faces = allocate_float(ni+1, nj+1)
    faces[1,1] = coord[1,1]
    for j ∈ 2:nj
        faces[1,j] = 0.5*(coord[1,j-1] + coord[1,j])
    end
    faces[1,nj+1] = coord[1,nj]
    for i ∈ 2:ni
        faces[i,1] = 0.5*(coord[i-1,1] + coord[i,1])
        for j ∈ 2:nj
            faces[i,j] = 0.25*(coord[i-1,j-1] + coord[i-1,j] + coord[i,j-1] + coord[i,j])
        end
        faces[i,nj+1] = 0.5*(coord[i-1,nj] + coord[i,nj])
    end
    faces[ni+1,1] = coord[ni,1]
    for j ∈ 2:nj
        faces[ni+1,j] = 0.5*(coord[ni,j-1] + coord[ni,j])
    end
    faces[ni+1,nj+1] = coord[ni,nj]

    return faces
end

function grid_points_to_faces(coord::Observable{T} where T <: AbstractMatrix)
    ni, nj = size(coord.val)
    faces = allocate_float(ni+1, nj+1)
    faces[1,1] = coord.val[1,1]
    for j ∈ 2:nj
        faces[1,j] = 0.5*(coord.val[1,j-1] + coord.val[1,j])
    end
    faces[1,nj+1] = coord.val[1,nj]
    for i ∈ 2:ni
        faces[i,1] = 0.5*(coord.val[i-1,1] + coord.val[i,1])
        for j ∈ 2:nj
            faces[i,j] = 0.25*(coord.val[i-1,j-1] + coord.val[i-1,j] + coord.val[i,j-1] + coord.val[i,j])
        end
        faces[i,nj+1] = 0.5*(coord.val[i-1,nj] + coord.val[i,nj])
    end
    faces[ni+1,1] = coord.val[ni,1]
    for j ∈ 2:nj
        faces[ni+1,j] = 0.5*(coord.val[ni,j-1] + coord.val[ni,j])
    end
    faces[ni+1,nj+1] = coord.val[ni,nj]

    return faces
end

"""
    get_variable_symbol(variable_name)

Get a symbol corresponding to a `variable_name`

For example `get_variable_symbol("phi")` returns `"ϕ"`.

If the symbol has not been defined, just return `variable_name`.
"""
function get_variable_symbol(variable_name)
    symbols_for_variables = Dict("phi"=>"ϕ", "Er"=>"Er", "Ez"=>"Ez", "density"=>"n",
                                 "parallel_flow"=>"u∥", "parallel_pressure"=>"p∥",
                                 "pressure"=>"p",
                                 "parallel_heat_flux"=>"q∥", "thermal_speed"=>"vth",
                                 "temperature"=>"T", "density_neutral"=>"nn",
                                 "uzeta_neutral"=>"unζ", "ur_neutral"=>"unr",
                                 "uz_neutral"=>"unz", "pzeta_neutral"=>"pnζ",
                                 "pr_neutral"=>"pnr", "pz_neutral"=>"pnz",
                                 "qzeta_neutral"=>"qnζ", "qr_neutral"=>"qnr",
                                 "qz_neutral"=>"qnz", "thermal_speed_neutral"=>"vnth",
                                 "temperature_neutral"=>"Tn")

    return get(symbols_for_variables, variable_name, variable_name)
end

"""
    parse_colormap(colormap)

Parse a `colormap` option

Allows us to have a string option which can be set in the input file and still use
Reverse, etc. conveniently.
"""
function parse_colormap(colormap)
    if colormap === nothing
        return colormap
    elseif startswith(colormap, "reverse_")
        # Use split to remove the "reverse_" prefix
        return Reverse(String(split(colormap, "reverse_", keepempty=false)[1]))
    else
        return colormap
    end
end
