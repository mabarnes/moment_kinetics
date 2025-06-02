using moment_kinetics.array_allocation: allocate_float
using moment_kinetics.load_data: regrid_variable

"""
    plots_for_variable(run_info, variable_name; plot_prefix, has_rdim=true,
                       has_zdim=true, is_1V=false,
                       steady_state_residual_fig_axes=nothing, kwargs...)

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

`kwargs...` are passed through to `plot_vs_*()` and `animate_vs_*()`.
"""
function plots_for_variable(run_info, variable_name; plot_prefix, has_rdim=true,
                            has_zdim=true, is_1V=false,
                            steady_state_residual_fig_axes=nothing,
                            subtract_from_info=nothing, interpolate_to_other_grid=false,
                            kwargs...)
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
    if subtract_from_info !== nothing
        subtract_from = nothing
        try
            subtract_from = get_variable(subtract_from_info, variable_name)

        catch e
            if isa(e, KeyError)
                println("Key $(e.key) not found when loading $variable_name to subtract "
                        * "- probably not present in output")
                return nothing
            else
                return makie_post_processing_error_handler(
                           e,
                           "plots_for_variable() failed when loading $variable_name to subtract.")
            end
        end
        # Here we can pass 'run_info' objects as the 'moments' argument of
        # regrid_time_evolving because they contain the necessary
        # `evolve_density`/`evolve_upar`/`evolve_p` flags.
        if interpolate_to_other_grid
            variable = [v .- regrid_time_evolving(subtract_from, _run_info_to_coords(ri),
                                                  _run_info_to_coords(subtract_from_info),
                                                  ri, subtract_from_info.evolve_density,
                                                  subtract_from_info.evolve_upar,
                                                  subtract_from_info.evolve_p)
                        for (ri, v) ∈ zip(run_info, variable)]
        else
            variable = [regrid_time_evolving(v, _run_info_to_coords(subtract_from_info),
                                             _run_info_to_coords(ri), subtract_from_info,
                                             ri.evolve_density, ri.evolve_upar,
                                             ri.evolve_p) .- subtract_from
                        for (ri, v) ∈ zip(run_info, variable)]

            # Override run_info so that we plot against the coordinates from
            # `subtract_from_info` for all plots.
            run_info = Any[(_run_info_to_coords(subtract_from_info)...,
                            run_name=ri.run_name) for ri ∈ run_info]
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
            plot_vs_r_t(run_info, variable_name; is=is, data=variable, input=input,
                        outfile=variable_prefix * "vs_r_t.pdf", kwargs...)
        end
        if has_zdim && input.plot_vs_z_t
            plot_vs_z_t(run_info, variable_name; is=is, data=variable, input=input,
                        outfile=variable_prefix * "vs_z_t.pdf", kwargs...)
        end
        if has_rdim && input.plot_vs_r
            plot_vs_r(run_info, variable_name; is=is, data=variable, input=input,
                      outfile=variable_prefix * "vs_r.pdf", kwargs...)
        end
        if has_zdim && input.plot_vs_z
            plot_vs_z(run_info, variable_name; is=is, data=variable, input=input,
                      outfile=variable_prefix * "vs_z.pdf", kwargs...)
        end
        if has_rdim && has_zdim && input.plot_vs_z_r
            plot_vs_z_r(run_info, variable_name; is=is, data=variable, input=input,
                        outfile=variable_prefix * "vs_z_r.pdf", kwargs...)
        end
        if has_zdim && input.animate_vs_z
            animate_vs_z(run_info, variable_name; is=is, data=variable, input=input,
                         outfile=variable_prefix * "vs_z." * input.animation_ext,
                         kwargs...)
        end
        if has_rdim && input.animate_vs_r
            animate_vs_r(run_info, variable_name; is=is, data=variable, input=input,
                         outfile=variable_prefix * "vs_r." * input.animation_ext,
                         kwargs...)
        end
        if has_rdim && has_zdim && input.animate_vs_z_r
            animate_vs_z_r(run_info, variable_name; is=is, data=variable, input=input,
                           outfile=variable_prefix * "vs_z_r." * input.animation_ext,
                           kwargs...)
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
                           has_zdim=true, is_1V=false, kwargs...)

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

`kwargs...` are passed through to `plot_vs_*()`, `plot_*_unnorm_vs_*()`, `animate_vs_*()`,
and `animate_*_unnorm_vs_*()`.
"""
function plots_for_dfn_variable(run_info, variable_name; plot_prefix, has_rdim=true,
                                has_zdim=true, is_1V=false, subtract_from_info=nothing,
                                interpolate_to_other_grid=false, kwargs...)
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

    variable = nothing
    if subtract_from_info !== nothing
        # When using subtract_from_info, it is generally necessary to interpolate,
        # which requires the full array to be loaded. For animations, it would be
        # possible to load the arrays one time point at a time, but it is not
        # convenient to do this with the current setup of `generic_plot_functions.jl`,
        # so we leave that as a future optimisation.
        subtract_from = nothing
        try
            variable = get_variable(run_info, variable_name)
            subtract_from = get_variable(subtract_from_info, variable_name)
        catch e
            if isa(e, KeyError)
                println("Key $(e.key) not found when loading $variable_name - probably not "
                        * "present in output")
                return nothing
            else
                return makie_post_processing_error_handler(
                           e,
                           "plots_for_dfn_variable() failed for $variable_name - could not load data.")
            end
        end
        # Here we can pass 'run_info' objects as the 'moments' argument of
        # regrid_time_evolving because they contain the necessary
        # `evolve_density`/`evolve_upar`/`evolve_p` flags.
        if interpolate_to_other_grid
            if variable_name == "f"
                moments = ((evolve_density=ri.evolve_density,
                            evolve_upar=ri.evolve_upar,
                            evolve_p=ri.evolve_p,
                            ion=(dens=get_variable(ri, "density"),
                                 upar=get_variable(ri, "parallel_flow"),
                                 vth=get_variable(ri, "thermal_speed"),
                                )
                           ) for ri ∈ run_info)
            elseif variable_name == "f_electron"
                moments = ((evolve_density=ri.evolve_density,
                            evolve_upar=ri.evolve_upar,
                            evolve_p=ri.evolve_p,
                            electron=(dens=get_variable(ri, "electron_density"),
                                      upar=get_variable(ri, "electron_parallel_flow"),
                                      vth=get_variable(ri, "electron_thermal_speed"),
                                     )
                           ) for ri ∈ run_info)
            elseif variable_name == "f_neutral"
                moments = ((evolve_density=ri.evolve_density,
                            evolve_upar=ri.evolve_upar,
                            evolve_p=ri.evolve_p,
                            neutral=(dens=get_variable(ri, "density_neutral"),
                                     uz=get_variable(ri, "uz_neutral"),
                                     vth=get_variable(ri, "thermal_speed_neutral"),
                                    )
                           ) for ri ∈ run_info)
            else
                error("Unsupported variable '$variable_name'")
            end
            variable = [v .- regrid_time_evolving(subtract_from, _run_info_to_coords(ri),
                                                  _run_info_to_coords(subtract_from_info),
                                                  m, subtract_from_info.evolve_density,
                                                  subtract_from_info.evolve_upar,
                                                  subtract_from_info.evolve_p)
                        for (ri, v, m) ∈ zip(run_info, variable, moments)]
        else
            if variable_name == "f"
                moments = (evolve_density=subtract_from_info.evolve_density,
                           evolve_upar=subtract_from_info.evolve_upar,
                           evolve_p=subtract_from_info.evolve_p,
                           ion=(dens=get_variable(subtract_from_info, "density"),
                                upar=get_variable(subtract_from_info, "parallel_flow"),
                                vth=get_variable(subtract_from_info, "thermal_speed"),
                               )
                          )
            elseif variable_name == "f_electron"
                moments = (evolve_density=subtract_from_info.evolve_density,
                           evolve_upar=subtract_from_info.evolve_upar,
                           evolve_p=subtract_from_info.evolve_p,
                           electron=(dens=get_variable(subtract_from_info, "electron_density"),
                                     upar=get_variable(subtract_from_info, "electron_parallel_flow"),
                                     vth=get_variable(subtract_from_info, "electron_thermal_speed"),
                                    )
                          )
            elseif variable_name == "f_neutral"
                moments = (evolve_density=subtract_from_info.evolve_density,
                           evolve_upar=subtract_from_info.evolve_upar,
                           evolve_p=subtract_from_info.evolve_p,
                           neutral=(dens=get_variable(subtract_from_info, "density_neutral"),
                                    uz=get_variable(subtract_from_info, "uz_neutral"),
                                    vth=get_variable(subtract_from_info, "thermal_speed_neutral"),
                                   )
                          )
            else
                error("Unsupported variable '$variable_name'")
            end
            variable = [regrid_time_evolving(v, _run_info_to_coords(subtract_from_info),
                                             _run_info_to_coords(ri), moments,
                                             ri.evolve_density, ri.evolve_upar,
                                             ri.evolve_p) .- subtract_from
                        for (ri, v) ∈ zip(run_info, variable)]

            # Override run_info so that we plot against the coordinates from
            # `subtract_from_info` for all plots.
            run_info = Any[(_run_info_to_coords(subtract_from_info)...,
                            run_name=ri.run_name) for ri ∈ run_info]
        end
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
                    func(run_info, variable_name; data=variable, is=is, input=input,
                         outfile=outfile, yscale=yscale, transform=transform, kwargs...)
                end
            end
            for (dim1, dim2) ∈ combinations(plot_dims, 2)
                if input[Symbol(:plot, log, :_vs_, dim2, :_, dim1)]
                    func = getfield(makie_post_processing,
                                    Symbol(:plot_vs_, dim2, :_, dim1))
                    outfile = var_prefix * "vs_$(dim2)_$(dim1).pdf"
                    func(run_info, variable_name; data=variable, is=is, input=input,
                         outfile=outfile, colorscale=yscale, transform=transform,
                         kwargs...)
                end
            end
            for dim ∈ animate_dims
                if input[Symbol(:animate, log, :_vs_, dim)]
                    func = getfield(makie_post_processing, Symbol(:animate_vs_, dim))
                    outfile = var_prefix * "vs_$dim." * input.animation_ext
                    func(run_info, variable_name; data=variable, is=is, input=input,
                         outfile=outfile, yscale=yscale, transform=transform, kwargs...)
                end
            end
            for (dim1, dim2) ∈ combinations(animate_dims, 2)
                if input[Symbol(:animate, log, :_vs_, dim2, :_, dim1)]
                    func = getfield(makie_post_processing,
                                    Symbol(:animate_vs_, dim2, :_, dim1))
                    outfile = var_prefix * "vs_$(dim2)_$(dim1)." * input.animation_ext
                    func(run_info, variable_name; data=variable, is=is, input=input,
                         outfile=outfile, colorscale=yscale, transform=transform,
                         kwargs...)
                end
            end

            # At present, plot_f_unnorm_vs_*() and animate_f_unnorm_vs_*() do not support
            # passing in a `data` array, so we cannot make these plots when using
            # `subtract_from_info`.
            if moment_kinetic && subtract_from_info === nothing
                if is_neutral
                    if input[Symbol(:plot, log, :_unnorm_vs_vz)]
                        outfile = var_prefix * "unnorm_vs_vz.pdf"
                        plot_f_unnorm_vs_vpa(run_info; input=input, neutral=true, is=is,
                                             outfile=outfile, yscale=yscale,
                                             transform=transform, kwargs...)
                    end
                    if has_zdim && input[Symbol(:plot, log, :_unnorm_vs_vz_z)]
                        outfile = var_prefix * "unnorm_vs_vz_z.pdf"
                        plot_f_unnorm_vs_vpa_z(run_info; input=input, neutral=true, is=is,
                                               outfile=outfile, colorscale=yscale,
                                               transform=transform, kwargs...)
                    end
                    if input[Symbol(:animate, log, :_unnorm_vs_vz)]
                        outfile = var_prefix * "unnorm_vs_vz." * input.animation_ext
                        animate_f_unnorm_vs_vpa(run_info; input=input, neutral=true,
                                                is=is, outfile=outfile, yscale=yscale,
                                                transform=transform, kwargs...)
                    end
                    if has_zdim && input[Symbol(:animate, log, :_unnorm_vs_vz_z)]
                        outfile = var_prefix * "unnorm_vs_vz_z." * input.animation_ext
                        animate_f_unnorm_vs_vpa_z(run_info; input=input, neutral=true,
                                                  is=is, outfile=outfile,
                                                  colorscale=yscale, transform=transform,
                                                  kwargs...)
                    end
                else
                    if input[Symbol(:plot, log, :_unnorm_vs_vpa)]
                        outfile = var_prefix * "unnorm_vs_vpa.pdf"
                        plot_f_unnorm_vs_vpa(run_info; input=input, electron=is_electron,
                                             is=is, outfile=outfile, yscale=yscale,
                                             transform=transform, kwargs...)
                    end
                    if has_zdim && input[Symbol(:plot, log, :_unnorm_vs_vpa_z)]
                        outfile = var_prefix * "unnorm_vs_vpa_z.pdf"
                        plot_f_unnorm_vs_vpa_z(run_info; input=input,
                                               electron=is_electron, is=is,
                                               outfile=outfile, colorscale=yscale,
                                               transform=transform, kwargs...)
                    end
                    if input[Symbol(:animate, log, :_unnorm_vs_vpa)]
                        outfile = var_prefix * "unnorm_vs_vpa." * input.animation_ext
                        animate_f_unnorm_vs_vpa(run_info; input=input,
                                                electron=is_electron, is=is,
                                                outfile=outfile, yscale=yscale,
                                                transform=transform, kwargs...)
                    end
                    if has_zdim && input[Symbol(:animate, log, :_unnorm_vs_vpa_z)]
                        outfile = var_prefix * "unnorm_vs_vpa_z." * input.animation_ext
                        animate_f_unnorm_vs_vpa_z(run_info; input=input,
                                                  electron=is_electron, is=is,
                                                  outfile=outfile, colorscale=yscale,
                                                  transform=transform, kwargs...)
                    end
                end
                if subtract_from_info === nothing
                    check_moment_constraints(run_info, is_neutral; input=input, plot_prefix)
                end
            end
        end
    end

    return nothing
end

function _run_info_to_coords(ri)
    return (n_ion_species=ri.n_ion_species, n_neutral_species=ri.n_neutral_species,
            time=ri.time, r=ri.r, r_spectral=ri.r_spectral, z=ri.z,
            z_spectral=ri.z_spectral, vperp=ri.vperp, vperp_spectral=ri.vperp_spectral,
            vpa=ri.vpa, vpa_spectral=ri.vpa_spectral, vzeta=ri.vzeta,
            vzeta_spectral=ri.vzeta_spectral, vr=ri.vr, vr_spectral=ri.vr_spectral,
            vz=ri.vz, vz_spectral=ri.vz_spectral)
end

function regrid_time_evolving(variable, new_coords, old_coords, moments,
                              old_evolve_density, old_evolve_upar, old_evolve_p)
    tdim = ndims(variable)
    nt = size(variable, tdim)

    first_regridded = regrid_variable(selectdim(variable, tdim, 1), new_coords,
                                      old_coords, moments, old_evolve_density,
                                      old_evolve_upar, old_evolve_p)

    result = allocate_float(size(first_regridded)..., nt)
    selectdim(result, tdim, 1) .= first_regridded

    for it ∈ 2:nt
        selectdim(result, tdim, it) .= regrid_variable(selectdim(variable, tdim, it),
                                                       new_coords, old_coords, moments,
                                                       old_evolve_density,
                                                       old_evolve_upar, old_evolve_p)
    end

    return result
end
