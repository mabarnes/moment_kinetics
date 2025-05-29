"""
    plots_for_variable(run_info, variable_name; plot_prefix, has_rdim=true,
                       has_zdim=true, is_1V=false,
                       steady_state_residual_fig_axes=nothing)

Make plots for the EM field or moment variable `variable_name`.

Which plots to make are determined by the settings in the section of the input whose
heading is the variable name.

`run_info` is the information returned by [`get_run_info`](@ref).

`plot_prefix` is required and gives the path and prefix for plots to be saved to. They
will be saved with the format `plot_prefix<some_identifying_string>.pdf` for plots and
`plot_prefix<some_identifying_string>.gif`, etc. for animations.

`has_rdim`, `has_zdim` and/or `is_1V` can be passed to allow the function to skip some
plots that do not make sense for 0D/1D or 1V simulations (regardless of the settings).

`steady_state_residual_fig_axes` contains the figure, axes and legend places for steady
state residual plots.
"""
function plots_for_variable(run_info, variable_name; plot_prefix, has_rdim=true,
                            has_zdim=true, is_1V=false,
                            steady_state_residual_fig_axes=nothing)
    input = Dict_to_NamedTuple(input_dict[variable_name])

    # test if any plot is needed
    if !(any(v for (k,v) in pairs(input) if
           startswith(String(k), "plot") || startswith(String(k), "animate") ||
           k == :steady_state_residual))
        return nothing
    end

    if !has_rdim && variable_name == "Er"
        return nothing
    elseif !has_zdim && variable_name == "Ez"
        return nothing
    elseif variable_name == "collision_frequency" &&
            all(ri.collisions.krook_collisions_option == "none" for ri ∈ run_info)
        # No Krook collisions active, so do not make plots.
        return nothing
    elseif variable_name ∈ union(electron_moment_variables, electron_source_variables, electron_dfn_variables) &&
            all(ri.composition.electron_physics ∈ (boltzmann_electron_response, boltzmann_electron_response_with_simple_sheath)
                for ri ∈ run_info)
        return nothing
    end

    println("Making plots for $variable_name")
    flush(stdout)

    variable = nothing
    try
        variable = get_variable(run_info, variable_name)
    catch e
        if isa(e, KeyError)
            println("Key $(e.key) not found when loading $variable_name - probably not "
                    * "present in output")
            return nothing
        else
            return makie_post_processing_error_handler(
                       e,
                       "plots_for_variable() failed for $variable_name - could not load data.")
        end
    end

    if variable_name ∈ em_variables
        species_indices = (nothing,)
    elseif variable_name ∈ neutral_moment_variables ||
           variable_name ∈ neutral_dfn_variables
        species_indices = 1:maximum(ri.n_neutral_species for ri ∈ run_info)
    elseif variable_name ∈ ion_moment_variables ||
           variable_name ∈ ion_dfn_variables
        species_indices = 1:maximum(ri.n_ion_species for ri ∈ run_info)
    elseif variable_name in ion_source_variables
        species_indices = 1:maximum(length(ri.external_source_settings.ion) for ri ∈ run_info)
    elseif variable_name in electron_source_variables
        species_indices = 1:maximum(length(ri.external_source_settings.electron) for ri ∈ run_info)
    elseif variable_name in neutral_source_variables
        species_indices = 1:maximum(length(ri.external_source_settings.neutral) for ri ∈ run_info)
    else
        species_indices = 1:1
        #error("variable_name=$variable_name not found in any defined group")
    end
    for is ∈ species_indices
        if is !== nothing
            variable_prefix = plot_prefix * variable_name * "_spec$(is)_"
            log_variable_prefix = plot_prefix * "log" * variable_name * "_spec$(is)_"
        else
            variable_prefix = plot_prefix * variable_name * "_"
            log_variable_prefix = plot_prefix * "log" * variable_name * "_"
        end
        if has_rdim && input.plot_vs_r_t
            plot_vs_r_t(run_info, variable_name, is=is, data=variable, input=input,
                        outfile=variable_prefix * "vs_r_t.pdf")
        end
        if has_zdim && input.plot_vs_z_t
            plot_vs_z_t(run_info, variable_name, is=is, data=variable, input=input,
                        outfile=variable_prefix * "vs_z_t.pdf")
        end
        if has_rdim && input.plot_vs_r
            plot_vs_r(run_info, variable_name, is=is, data=variable, input=input,
                      outfile=variable_prefix * "vs_r.pdf")
        end
        if has_zdim && input.plot_vs_z
            plot_vs_z(run_info, variable_name, is=is, data=variable, input=input,
                      outfile=variable_prefix * "vs_z.pdf")
        end
        if has_rdim && has_zdim && input.plot_vs_z_r
            plot_vs_z_r(run_info, variable_name, is=is, data=variable, input=input,
                        outfile=variable_prefix * "vs_z_r.pdf")
        end
        if has_zdim && input.animate_vs_z
            animate_vs_z(run_info, variable_name, is=is, data=variable, input=input,
                         outfile=variable_prefix * "vs_z." * input.animation_ext)
        end
        if has_rdim && input.animate_vs_r
            animate_vs_r(run_info, variable_name, is=is, data=variable, input=input,
                         outfile=variable_prefix * "vs_r." * input.animation_ext)
        end
        if has_rdim && has_zdim && input.animate_vs_z_r
            animate_vs_z_r(run_info, variable_name, is=is, data=variable, input=input,
                           outfile=variable_prefix * "vs_r." * input.animation_ext)
        end
        if input.steady_state_residual
            calculate_steady_state_residual(run_info, variable_name; is=is, data=variable,
                                            fig_axes=steady_state_residual_fig_axes)
        end
    end

    return nothing
end

"""
    plots_for_dfn_variable(run_info, variable_name; plot_prefix, has_rdim=true,
                           has_zdim=true, is_1V=false)

Make plots for the distribution function variable `variable_name`.

Which plots to make are determined by the settings in the section of the input whose
heading is the variable name.

`run_info` is the information returned by [`get_run_info()`](@ref). The `dfns=true` keyword
argument must have been passed to [`get_run_info()`](@ref) so that output files containing
the distribution functions are being read.

`plot_prefix` is required and gives the path and prefix for plots to be saved to. They
will be saved with the format `plot_prefix<some_identifying_string>.pdf` for plots and
`plot_prefix<some_identifying_string>.gif`, etc. for animations.

`has_rdim`, `has_zdim` and/or `is_1V` can be passed to allow the function to skip some
plots that do not make sense for 0D/1D or 1V simulations (regardless of the settings).
"""
function plots_for_dfn_variable(run_info, variable_name; plot_prefix, has_rdim=true,
                                has_zdim=true, is_1V=false)
    input = Dict_to_NamedTuple(input_dict_dfns[variable_name])

    is_neutral = variable_name ∈ neutral_dfn_variables
    is_electron = variable_name ∈ electron_dfn_variables

    if is_neutral
        animate_dims = setdiff(neutral_dimensions, (:sn,))
        if is_1V
            animate_dims = setdiff(animate_dims, (:vzeta, :vr))
        end
    else
        animate_dims = setdiff(ion_dimensions, (:s,))
        if is_1V
            animate_dims = setdiff(animate_dims, (:vperp,))
        end
    end
    if !has_rdim
        animate_dims = setdiff(animate_dims, (:r,))
    end
    if !has_zdim
        animate_dims = setdiff(animate_dims, (:z,))
    end
    plot_dims = tuple(:t, animate_dims...)

    moment_kinetic = any(ri !== nothing
                         && (ri.evolve_density || ri.evolve_upar || ri.evolve_p)
                         for ri ∈ run_info)

    # test if any plot is needed
    if !any(v for (k,v) in pairs(input) if
            startswith(String(k), "plot") || startswith(String(k), "animate"))
        return nothing
    end

    println("Making plots for $variable_name")
    flush(stdout)

    if is_neutral
        species_indices = 1:maximum(ri.n_neutral_species for ri ∈ run_info)
    else
        species_indices = 1:maximum(ri.n_ion_species for ri ∈ run_info)
    end
    for is ∈ species_indices
        variable_prefix = plot_prefix * variable_name * "_"
        log_variable_prefix = plot_prefix * "log" * variable_name * "_"

        # Note that we use `yscale=log10` and `transform=positive_or_nan` rather than
        # defining a custom scaling function (which would return NaN for negative
        # values) because it messes up the automatic minimum value for the colorscale:
        # The transform removes any zero or negative values from the data, so the
        # minimum value for the colorscale is set by the smallest positive value; with
        # only the custom colorscale, the minimum would be negative and the
        # corresponding color would be the color for NaN, which does not go on the
        # Colorbar and so causes an error.
        for (log, yscale, transform, var_prefix) ∈
                ((:"", nothing, identity, variable_prefix),
                 (:_log, log10, x->positive_or_nan(x; epsilon=1.e-20), log_variable_prefix))
            for dim ∈ plot_dims
                if input[Symbol(:plot, log, :_vs_, dim)]
                    func = getfield(makie_post_processing, Symbol(:plot_vs_, dim))
                    outfile = var_prefix * "vs_$dim.pdf"
                    func(run_info, variable_name, is=is, input=input, outfile=outfile,
                         yscale=yscale, transform=transform)
                end
            end
            for (dim1, dim2) ∈ combinations(plot_dims, 2)
                if input[Symbol(:plot, log, :_vs_, dim2, :_, dim1)]
                    func = getfield(makie_post_processing,
                                    Symbol(:plot_vs_, dim2, :_, dim1))
                    outfile = var_prefix * "vs_$(dim2)_$(dim1).pdf"
                    func(run_info, variable_name, is=is, input=input, outfile=outfile,
                         colorscale=yscale, transform=transform)
                end
            end
            for dim ∈ animate_dims
                if input[Symbol(:animate, log, :_vs_, dim)]
                    func = getfield(makie_post_processing, Symbol(:animate_vs_, dim))
                    outfile = var_prefix * "vs_$dim." * input.animation_ext
                    func(run_info, variable_name, is=is, input=input, outfile=outfile,
                         yscale=yscale, transform=transform)
                end
            end
            for (dim1, dim2) ∈ combinations(animate_dims, 2)
                if input[Symbol(:animate, log, :_vs_, dim2, :_, dim1)]
                    func = getfield(makie_post_processing,
                                    Symbol(:animate_vs_, dim2, :_, dim1))
                    outfile = var_prefix * "vs_$(dim2)_$(dim1)." * input.animation_ext
                    func(run_info, variable_name, is=is, input=input, outfile=outfile,
                         colorscale=yscale, transform=transform)
                end
            end

            if moment_kinetic
                if is_neutral
                    if input[Symbol(:plot, log, :_unnorm_vs_vz)]
                        outfile = var_prefix * "unnorm_vs_vz.pdf"
                        plot_f_unnorm_vs_vpa(run_info; input=input, neutral=true, is=is,
                                             outfile=outfile, yscale=yscale, transform=transform)
                    end
                    if has_zdim && input[Symbol(:plot, log, :_unnorm_vs_vz_z)]
                        outfile = var_prefix * "unnorm_vs_vz_z.pdf"
                        plot_f_unnorm_vs_vpa_z(run_info; input=input, neutral=true, is=is,
                                               outfile=outfile, colorscale=yscale,
                                               transform=transform)
                    end
                    if input[Symbol(:animate, log, :_unnorm_vs_vz)]
                        outfile = var_prefix * "unnorm_vs_vz." * input.animation_ext
                        animate_f_unnorm_vs_vpa(run_info; input=input, neutral=true, is=is,
                                                outfile=outfile, yscale=yscale,
                                                transform=transform)
                    end
                    if has_zdim && input[Symbol(:animate, log, :_unnorm_vs_vz_z)]
                        outfile = var_prefix * "unnorm_vs_vz_z." * input.animation_ext
                        animate_f_unnorm_vs_vpa_z(run_info; input=input, neutral=true, is=is,
                                                  outfile=outfile, colorscale=yscale,
                                                  transform=transform)
                    end
                else
                    if input[Symbol(:plot, log, :_unnorm_vs_vpa)]
                        outfile = var_prefix * "unnorm_vs_vpa.pdf"
                        plot_f_unnorm_vs_vpa(run_info; input=input, electron=is_electron,
                                             is=is, outfile=outfile, yscale=yscale,
                                             transform=transform)
                    end
                    if has_zdim && input[Symbol(:plot, log, :_unnorm_vs_vpa_z)]
                        outfile = var_prefix * "unnorm_vs_vpa_z.pdf"
                        plot_f_unnorm_vs_vpa_z(run_info; input=input,
                                               electron=is_electron, is=is,
                                               outfile=outfile, colorscale=yscale,
                                               transform=transform)
                    end
                    if input[Symbol(:animate, log, :_unnorm_vs_vpa)]
                        outfile = var_prefix * "unnorm_vs_vpa." * input.animation_ext
                        animate_f_unnorm_vs_vpa(run_info; input=input,
                                                electron=is_electron, is=is,
                                                outfile=outfile, yscale=yscale,
                                                transform=transform)
                    end
                    if has_zdim && input[Symbol(:animate, log, :_unnorm_vs_vpa_z)]
                        outfile = var_prefix * "unnorm_vs_vpa_z." * input.animation_ext
                        animate_f_unnorm_vs_vpa_z(run_info; input=input,
                                                  electron=is_electron, is=is,
                                                  outfile=outfile, colorscale=yscale,
                                                  transform=transform)
                    end
                end
                check_moment_constraints(run_info, is_neutral; input=input, plot_prefix)
            end
        end
    end

    return nothing
end
