using moment_kinetics.input_structs

"""
    timestep_diagnostics(run_info, run_info_dfns; plot_prefix=nothing, it=nothing)

Plot a time-trace of some adaptive-timestep diagnostics: steps per output, timestep
failures per output, how many times per output each variable caused a timestep failure,
and which factor limited the length of successful timesteps (CFL, accuracy, max_timestep).

If `plot_prefix` is passed, it gives the path and prefix for plots to be saved to. They
will be saved with the format `plot_prefix_timestep_diagnostics.pdf`.

`it` can be used to select a subset of the time points by passing a range.
"""
function timestep_diagnostics(run_info, run_info_dfns; plot_prefix=nothing, it=nothing,
                              electron=false)
    if !isa(run_info, Tuple)
        run_info = (run_info,)
    end

    input = Dict_to_NamedTuple(input_dict["timestep_diagnostics"])

    if input.plot || input.animate_CFL || input.plot_timestep_residual ||
            input.animate_timestep_residual || input.plot_timestep_error ||
            input.animate_timestep_error || input.plot_steady_state_residual ||
            input.animate_steady_state_residual
        if electron
            println("Making electron timestep diagnostics plots")
        else
            println("Making timestep diagnostics plots")
        end
    end

    steps_fig = nothing
    dt_fig = nothing
    CFL_fig = nothing

    if electron
        electron_prefix = "electron_"
    else
        electron_prefix = ""
    end

    if input.plot
        try
            # Plot numbers of steps and numbers of failures
            ###############################################

            steps_fig, ax = get_1d_ax(; xlabel="time", ylabel="number of steps per output")
            # Put failures a separate y-axis
            ax_failures = Axis(steps_fig[1, 1]; ylabel="number of failures per output",
                               yaxisposition = :right)
            hidespines!(ax_failures)
            hidexdecorations!(ax_failures)
            hideydecorations!(ax_failures; ticks=false, label=false, ticklabels=false)

            for ri ∈ run_info
                if length(run_info) == 1
                    prefix = ""
                else
                    prefix = ri.run_name * " "
                end

                if it !== nothing
                    time = ri.time[it]
                else
                    time = ri.time
                end
                plot_1d(time, get_variable(ri, "$(electron_prefix)steps_per_output";
                                              it=it); label=prefix * "steps", ax=ax)
                # Fudge to create an invisible line on ax_failures that cycles the line colors
                # and adds a label for "steps_per_output" to the plot because we create the
                # legend from ax_failures.
                plot_1d([ri.time[1]], [0]; label=prefix * "steps", ax=ax_failures)
                plot_1d(time,
                        get_variable(ri, "$(electron_prefix)failures_per_output"; it=it);
                        label=prefix * "failures", ax=ax_failures)

                if "$(electron_prefix)failure_caused_by" ∈ ri.variable_names
                    # Old version, where "failure_caused_by" was written out as an array,
                    # requiring a counter to get the correct variable to plot.
                    failure_caused_by_per_output =
                        get_variable(ri, "$(electron_prefix)failure_caused_by_per_output";
                                     it=it)
                    counter = 0
                    # pdf failure counter
                    counter += 1
                    if electron
                        label = prefix * "failures caused by f_electron"
                    else
                        label = prefix * "failures caused by f_ion"
                    end
                    plot_1d(time, @view failure_caused_by_per_output[counter,:];
                            label=label, ax=ax_failures)
                    if !electron && ri.evolve_density
                        # Ion density failure counter
                        counter += 1
                        plot_1d(time, @view failure_caused_by_per_output[counter,:];
                                linestyle=:dash, label=prefix * "failures caused by n_ion",
                                ax=ax_failures)
                    end
                    if !electron && ri.evolve_upar
                        # Ion flow failure counter
                        counter += 1
                        plot_1d(time, @view failure_caused_by_per_output[counter,:];
                                linestyle=:dash, label=prefix * "failures caused by u_ion",
                                ax=ax_failures)
                    end
                    if !electron && ri.evolve_p
                        # Ion parallel pressure failure counter
                        counter += 1
                        plot_1d(time, @view failure_caused_by_per_output[counter,:];
                                linestyle=:dash, label=prefix * "failures caused by p_ion",
                                ax=ax_failures)
                    end
                    if electron || ri.composition.electron_physics ∈ (braginskii_fluid,
                                                                      kinetic_electrons,
                                                                      kinetic_electrons_with_temperature_equation)
                        # Electron parallel pressure failure counter
                        counter += 1
                        plot_1d(time, @view failure_caused_by_per_output[counter,:];
                                linestyle=:dash, label=prefix * "failures caused by p_electron",
                                ax=ax_failures)
                        if !electron && ri.composition.electron_physics ∈ (kinetic_electrons,
                                                                           kinetic_electrons_with_temperature_equation)
                            # Kinetic electron nonlinear solver failure
                            counter += 1
                            plot_1d(time, @view failure_caused_by_per_output[counter,:];
                                    linestyle=:dash, label=prefix * "failures caused by kinetic electron solve",
                                    ax=ax_failures)
                        end
                    end
                    if !electron && ri.n_neutral_species > 0
                        # Neutral pdf failure counter
                        counter += 1
                        plot_1d(time, @view failure_caused_by_per_output[counter,:];
                                label=prefix * "failures caused by f_neutral", ax=ax_failures)
                        if ri.evolve_density
                            # Neutral density failure counter
                            counter += 1
                            plot_1d(time, @view failure_caused_by_per_output[counter,:];
                                    linestyle=:dash,
                                    label=prefix * "failures caused by n_neutral", ax=ax_failures)
                        end
                        if ri.evolve_upar
                            # Neutral flow failure counter
                            counter += 1
                            plot_1d(time, @view failure_caused_by_per_output[counter,:];
                                    linestyle=:dash,
                                    label=prefix * "failures caused by u_neutral", ax=ax_failures)
                        end
                        if ri.evolve_p
                            # Neutral flow failure counter
                            counter += 1
                            plot_1d(time, @view failure_caused_by_per_output[counter,:];
                                    linestyle=:dash,
                                    label=prefix * "failures caused by p_neutral", ax=ax_failures)
                        end
                        if occursin("ARK", ri.t_input["type"])
                            # Nonlinear iteration failed to converge in implicit part of
                            # timestep
                            counter += 1
                            plot_1d(time, @view failure_caused_by_per_output[counter,:];
                                    linestyle=:dot,
                                    label=prefix * "nonlinear iteration convergence failure", ax=ax_failures)
                        end
                        if ri.composition.electron_physics ∈ (kinetic_electrons,
                                                              kinetic_electrons_with_temperature_equation)
                            # Kinetic electron iteration failed to converge
                            counter += 1
                            plot_1d(time, @view failure_caused_by_per_output[counter,:];
                                    linestyle=:dot,
                                    label=prefix * "nonlinear iteration convergence failure", ax=ax_failures)
                        end
                    end

                    if counter > size(failure_caused_by_per_output, 1)
                        error("Tried to plot non-existent variables in "
                              * "failure_caused_by_per_output. Settings not understood "
                              * "correctly.")
                    end
                    if counter < size(failure_caused_by_per_output, 1)
                        error("Some variables in failure_caused_by_per_output not plotted. "
                              * "Settings not understood correctly.")
                    end
                else
                    # New version, where "failure_caused_by_*" are written as separate
                    # variables, which we can loop over.
                    failure_vars = [v for v ∈ ri.variable_names
                                    if startswith(v, "$(electron_prefix)failure_caused_by")]
                    for v ∈ failure_vars
                        label = prefix * v
                        if occursin("neutral", v)
                            linestyle = :dash
                        elseif occursin("convergence", v)
                            linestyle = :dot
                        else
                            linestyle = :solid
                        end
                        plot_vs_t(ri, "$(v)_per_step"; linestyle=linestyle, label=label,
                                  ax=ax_failures)
                    end
                end
            end

            put_legend_below(steps_fig, ax_failures)
            # Ensure the first row width is 3/4 of the column width so that
            # the plot does not get squashed by the legend
            rowsize!(steps_fig.layout, 1, Aspect(1, 3/4))
            resize_to_layout!(steps_fig)

            if plot_prefix !== nothing
                outfile = plot_prefix * electron_prefix * "timestep_diagnostics.pdf"
                save(outfile, steps_fig)
            else
                display(steps_fig)
            end
        catch e
            makie_post_processing_error_handler(
                e,
                "Error in timestep_diagnostics() steps_fig.")
        end

        try
            # Plot average timesteps
            ########################

            if plot_prefix !== nothing
                outfile = plot_prefix * "$(electron_prefix)successful_dt.pdf"
            else
                outfile = nothing
            end
            dt_fig = plot_vs_t(run_info, "$(electron_prefix)average_successful_dt"; outfile=outfile)

            if plot_prefix === nothing
                display(dt_fig)
            end
        catch e
            makie_post_processing_error_handler(
                e,
                "Error in timestep_diagnostics() dt_fig.")
        end

        try
            # PLot minimum CFL factors
            ##########################

            CFL_fig, ax = get_1d_ax(; xlabel="time", ylabel="(grid spacing) / speed")
            #maxval = Inf
            for ri ∈ run_info
                if length(run_info) == 1
                    prefix = ""
                else
                    prefix = ri.run_name * " "
                end
                if it !== nothing
                    time = ri.time[it]
                else
                    time = ri.time
                end

                if electron
                    CFL_vars = ["minimum_CFL_electron_z", "minimum_CFL_electron_vpa"]
                    implicit_CFL_vars = String[]
                else
                    CFL_vars = String[]
                    implicit_CFL_vars = String[]

                    push!(CFL_vars, "minimum_CFL_ion_z")
                    if occursin("ARK", ri.t_input["type"]) && ri.t_input["kinetic_ion_solver"] == full_implicit_ion_advance
                        push!(implicit_CFL_vars, "minimum_CFL_ion_z")
                    end
                    push!(CFL_vars, "minimum_CFL_ion_vpa")
                    if occursin("ARK", ri.t_input["type"]) && ( (ri.t_input["kinetic_ion_solver"] == full_implicit_ion_advance) ||
                                                                (ri.t_input["kinetic_ion_solver"] == implicit_ion_vpa_advection))
                        push!(implicit_CFL_vars, "minimum_CFL_ion_vpa")
                    end
                    if ri.n_neutral_species > 0
                        push!(CFL_vars, "minimum_CFL_neutral_z", "minimum_CFL_neutral_vz")
                    end
                end
                if it !== nothing
                    time = ri.time[it]
                else
                    time = ri.time
                end
                for varname ∈ CFL_vars
                    var = get_variable(ri, varname)
                    #maxval = NaNMath.min(maxval, NaNMath.maximum(var))
                    if occursin("neutral", varname)
                        if varname ∈ implicit_CFL_vars
                            linestyle = :dashdot
                        else
                            linestyle = :dash
                        end
                    else
                        if varname ∈ implicit_CFL_vars
                            linestyle = :dot
                        else
                            linestyle = nothing
                        end
                    end
                    plot_1d(time, var; ax=ax, label=prefix*electron_prefix*varname,
                            linestyle=linestyle, yscale=log10,
                            transform=x->positive_or_nan(x; epsilon=1.e-20))
                end
            end
            #ylims!(ax, 0.0, 10.0 * maxval)
            put_legend_below(CFL_fig, ax)
            # Ensure the first row width is 3/4 of the column width so that
            # the plot does not get squashed by the legend
            rowsize!(CFL_fig.layout, 1, Aspect(1, 3/4))
            resize_to_layout!(CFL_fig)

            if plot_prefix !== nothing
                outfile = plot_prefix * electron_prefix * "CFL_factors.pdf"
                save(outfile, CFL_fig)
            else
                display(CFL_fig)
            end
        catch e
            makie_post_processing_error_handler(
                e,
                "Error in timestep_diagnostics() CFL_fig.")
        end

        try
            limits_fig, ax = get_1d_ax(; xlabel="time", ylabel="number of limits per factor per output",
                                       size=(600, 500))

            for ri ∈ run_info
                if length(run_info) == 1
                    prefix = ""
                else
                    prefix = ri.run_name * " "
                end

                if "$(electron_prefix)limit_caused_by_per_output" ∈ ri.variable_names
                    # Old version, where "limit_caused_by" was written out as an array,
                    # requiring a counter to get the correct variable to plot.
                    if it !== nothing
                        time = ri.time[it]
                    else
                        time = ri.time
                    end

                    limit_caused_by_per_output =
                        get_variable(ri, "$(electron_prefix)limit_caused_by_per_output";
                                     it=it)
                    counter = 0

                    # Maximum timestep increase limit counter
                    counter += 1
                    plot_1d(time, @view limit_caused_by_per_output[counter,:];
                            label=prefix * "max timestep increase", ax=ax)

                    # Slower maximum timestep increase near last failure limit counter
                    counter += 1
                    plot_1d(time, @view limit_caused_by_per_output[counter,:];
                            label=prefix * "max timestep increase near last fail", ax=ax)

                    # Minimum timestep limit counter
                    counter += 1
                    plot_1d(time, @view limit_caused_by_per_output[counter,:];
                            label=prefix * "min timestep", ax=ax)

                    # Maximum timestep limit counter
                    counter += 1
                    plot_1d(time, @view limit_caused_by_per_output[counter,:];
                            label=prefix * "max timestep", ax=ax)

                    # High nonlinear iterations count
                    counter += 1
                    plot_1d(time, @view limit_caused_by_per_output[counter,:];
                            label=prefix * "high nl iterations", ax=ax)

                    # Accuracy limit counters
                    counter += 1
                    if electron
                        label = prefix * "electron pdf RK accuracy"
                    else
                        label = prefix * "ion pdf RK accuracy"
                    end
                    plot_1d(time, @view limit_caused_by_per_output[counter,:];
                            label=label, ax=ax, linestyle=:dash)
                    if !electron && ri.evolve_density
                        counter += 1
                        plot_1d(time, @view limit_caused_by_per_output[counter,:];
                                label=prefix * "ion density RK accuracy", ax=ax,
                                linestyle=:dash)
                    end
                    if !electron && ri.evolve_upar
                        counter += 1
                        plot_1d(time, @view limit_caused_by_per_output[counter,:];
                                label=prefix * "ion upar RK accuracy", ax=ax,
                                linestyle=:dash)
                    end
                    if !electron && ri.evolve_p
                        counter += 1
                        plot_1d(time, @view limit_caused_by_per_output[counter,:];
                                label=prefix * "ion ppar RK accuracy", ax=ax,
                                linestyle=:dash)
                    end
                    if electron || ri.composition.electron_physics ∈ (braginskii_fluid,
                                                                      kinetic_electrons,
                                                                      kinetic_electrons_with_temperature_equation)
                        counter += 1
                        plot_1d(time, @view limit_caused_by_per_output[counter,:];
                                label=prefix * "electron ppar RK accuracy", ax=ax,
                                linestyle=:dash)
                    end
                    if !electron && ri.n_neutral_species > 0
                        counter += 1
                        plot_1d(time, @view limit_caused_by_per_output[counter,:];
                                label=prefix * "neutral pdf RK accuracy", ax=ax,
                                linestyle=:dash)
                        if ri.evolve_density
                            counter += 1
                            plot_1d(time, @view limit_caused_by_per_output[counter,:];
                                    label=prefix * "neutral density RK accuracy", ax=ax,
                                    linestyle=:dash)
                        end
                        if ri.evolve_upar
                            counter += 1
                            plot_1d(time, @view limit_caused_by_per_output[counter,:];
                                    label=prefix * "neutral uz RK accuracy", ax=ax,
                                    linestyle=:dash)
                        end
                        if ri.evolve_p
                            counter += 1
                            plot_1d(time, @view limit_caused_by_per_output[counter,:];
                                    label=prefix * "neutral pz RK accuracy", ax=ax,
                                    linestyle=:dash)
                        end
                    end

                    if electron || !(occursin("ARK", ri.t_input["type"]) && ri.t_input["kinetic_ion_solver"] == full_implicit_ion_advance)
                        # Ion z advection
                        counter += 1
                        if electron
                            label = prefix * "electron z advect"
                        else
                            label = prefix * "ion z advect"
                        end
                        plot_1d(time, @view limit_caused_by_per_output[counter,:];
                                label=label, ax=ax, linestyle=:dot)
                    end

                    if electron || !(occursin("ARK", ri.t_input["type"]) && (ri.t_input["kinetic_ion_solver"] == full_implicit_ion_advance) ||
                                                                            (ri.t_input["kinetic_ion_solver"] == implicit_ion_vpa_advection))
                        # Ion vpa advection
                        counter += 1
                        if electron
                            label = prefix * "electron vpa advect"
                        else
                            label = prefix * "ion vpa advect"
                        end
                        plot_1d(time, @view limit_caused_by_per_output[counter,:];
                                label=label, ax=ax, linestyle=:dot)
                    end

                    if !electron && ri.n_neutral_species > 0
                        # Neutral z advection
                        counter += 1
                        plot_1d(time, @view limit_caused_by_per_output[counter,:];
                                label=prefix * "neutral z advect", ax=ax, linestyle=:dot)

                        # Neutral vz advection
                        counter += 1
                        plot_1d(time, @view limit_caused_by_per_output[counter,:];
                                label=prefix * "neutral vz advect", ax=ax, linestyle=:dot)
                    end

                    if counter > size(limit_caused_by_per_output, 1)
                        error("Tried to plot non-existent variables in "
                              * "limit_caused_by_per_output. Settings not understood "
                              * "correctly.")
                    end
                    if counter < size(limit_caused_by_per_output, 1)
                        error("Some variables in limit_caused_by_per_output not plotted. "
                              * "Settings not understood correctly.")
                    end
                else
                    # New version, where "limit_caused_by_*" are written as separate
                    # variables, which we can loop over.
                    limit_vars = [v for v ∈ ri.variable_names
                                  if startswith(v, "$(electron_prefix)limit_caused_by")]
                    for v ∈ limit_vars
                        label = prefix * v
                        if occursin("accuracy", v)
                            linestyle = :dash
                        elseif occursin("CFL", v)
                            linestyle = :dot
                        else
                            linestyle = :solid
                        end
                        plot_vs_t(ri, "$(v)_per_step"; linestyle=linestyle, label=label,
                                  ax=ax)
                    end
                end
            end

            put_legend_below(limits_fig, ax)
            # Ensure the first row width is 3/4 of the column width so that
            # the plot does not get squashed by the legend
            rowsize!(limits_fig.layout, 1, Aspect(1, 3/4))
            resize_to_layout!(limits_fig)

            if plot_prefix !== nothing
                outfile = plot_prefix * electron_prefix * "timestep_limits.pdf"
                save(outfile, limits_fig)
            else
                display(limits_fig)
            end
        catch e
            makie_post_processing_error_handler(
                e,
                "Error in timestep_diagnostics() limits_fig.")
        end

        try
            # Plot nonlinear solver diagnostics (if any)
            nl_solvers_fig, ax = get_1d_ax(; xlabel="time", ylabel="iterations per solve/nonlinear-iteration")
            has_nl_solver = false

            for ri ∈ run_info
                if length(run_info) == 1
                    prefix = ""
                else
                    prefix = ri.run_name * " "
                end
                if it !== nothing
                    time = ri.time[it]
                else
                    time = ri.time
                end

                nl_nonlinear_iterations_names = Tuple(v for v ∈ ri.variable_names
                                                      if occursin("_nonlinear_iterations", v))
                if nl_nonlinear_iterations_names != ()
                    has_nl_solver = true
                    nl_prefixes = (split(v, "_nonlinear_iterations")[1]
                                   for v ∈ nl_nonlinear_iterations_names)
                    for p ∈ nl_prefixes
                        nonlinear_iterations = get_variable(ri, "$(p)_nonlinear_iterations_per_solve")
                        linear_iterations = get_variable(ri, "$(p)_linear_iterations_per_nonlinear_iteration")
                        precon_iterations = get_variable(ri, "$(p)_precon_iterations_per_linear_iteration")
                        plot_1d(time, nonlinear_iterations, label=prefix * " " * p * " NL per solve", ax=ax)
                        plot_1d(time, linear_iterations, label=prefix * " " * p * " L per NL", ax=ax)
                        plot_1d(time, precon_iterations, label=prefix * " " * p * " P per L", ax=ax)
                    end
                end
            end

            if has_nl_solver
                put_legend_below(nl_solvers_fig, ax)
                # Ensure the first row width is 3/4 of the column width so that
                # the plot does not get squashed by the legend
                rowsize!(nl_solvers_fig.layout, 1, Aspect(1, 3/4))
                resize_to_layout!(nl_solvers_fig)

                if plot_prefix !== nothing
                    outfile = plot_prefix * "nonlinear_solver_iterations.pdf"
                    save(outfile, nl_solvers_fig)
                else
                    display(nl_solvers_fig)
                end
            end
        catch e
            makie_post_processing_error_handler(
                e,
                "Error in timestep_diagnostics() nl_solvers_fig.")
        end

        try
            # Plot electron solver diagnostics
            electron_solver_fig, ax = get_1d_ax(; xlabel="time", ylabel="electron steps per ion step")

            has_electron_solve = false
            for ri ∈ run_info
                if length(run_info) == 1
                    prefix = ""
                else
                    prefix = ri.run_name * " "
                end
                if it !== nothing
                    time = ri.time[it]
                else
                    time = ri.time
                end

                if ri.composition.electron_physics ∈ (kinetic_electrons,
                                                      kinetic_electrons_with_temperature_equation)
                    has_electron_solve = true
                    electron_steps_per_ion_step = get_variable(ri, "electron_steps_per_ion_step")
                    plot_1d(time, electron_steps_per_ion_step, label=prefix * " electron steps per solve", ax=ax)
                end
            end

            if has_electron_solve
                put_legend_below(electron_solver_fig, ax)
                # Ensure the first row width is 3/4 of the column width so that
                # the plot does not get squashed by the legend
                rowsize!(electron_solver_fig.layout, 1, Aspect(1, 3/4))
                resize_to_layout!(electron_solver_fig)

                if has_electron_solve
                    outfile = plot_prefix * "electron_steps.pdf"
                    save(outfile, electron_solver_fig)
                else
                    display(electron_solver_fig)
                end
            end
        catch e
            makie_post_processing_error_handler(
                e,
                "Error in timestep_diagnostics() nl_solvers_fig.")
        end
    end

    if input.animate_CFL
        try
            if plot_prefix === nothing
                error("plot_prefix is required when animate_CFL=true")
            end
            if !electron
                data = get_variable(run_info, "CFL_ion_z")
                datamin = minimum(minimum(d) for d ∈ data)
                animate_vs_vpa_z(run_info, "CFL_ion_z"; data=data, it=it,
                                 outfile=plot_prefix * "CFL_ion_z_vs_vpa_z.gif",
                                 colorscale=log10,
                                 transform=x->positive_or_nan(x; epsilon=1.e-30),
                                 colorrange=(datamin, datamin * 1000.0),
                                 axis_args=Dict(:bottomspinevisible=>false,
                                                :topspinevisible=>false,
                                                :leftspinevisible=>false,
                                                :rightspinevisible=>false))
                data = get_variable(run_info, "CFL_ion_vpa")
                datamin = minimum(minimum(d) for d ∈ data)
                animate_vs_vpa_z(run_info, "CFL_ion_vpa"; data=data, it=it,
                                 outfile=plot_prefix * "CFL_ion_vpa_vs_vpa_z.gif",
                                 colorscale=log10,
                                 transform=x->positive_or_nan(x; epsilon=1.e-30),
                                 colorrange=(datamin, datamin * 1000.0),
                                 axis_args=Dict(:bottomspinevisible=>false,
                                                :topspinevisible=>false,
                                                :leftspinevisible=>false,
                                                :rightspinevisible=>false))
            end
            if electron || any(ri.composition.electron_physics ∈ (kinetic_electrons,
                                                                  kinetic_electrons_with_temperature_equation)
                               for ri ∈ run_info)
                data = get_variable(run_info, "CFL_electron_z")
                datamin = minimum(minimum(d) for d ∈ data)
                animate_vs_vpa_z(run_info, "CFL_electron_z"; data=data, it=it,
                                 outfile=plot_prefix * "CFL_electron_z_vs_vpa_z.gif",
                                 colorscale=log10,
                                 transform=x->positive_or_nan(x; epsilon=1.e-30),
                                 colorrange=(datamin, datamin * 1000.0),
                                 axis_args=Dict(:bottomspinevisible=>false,
                                                :topspinevisible=>false,
                                                :leftspinevisible=>false,
                                                :rightspinevisible=>false))
                data = get_variable(run_info, "CFL_electron_vpa")
                datamin = minimum(minimum(d) for d ∈ data)
                animate_vs_vpa_z(run_info, "CFL_electron_vpa"; data=data, it=it,
                                 outfile=plot_prefix * "CFL_electron_vpa_vs_vpa_z.gif",
                                 colorscale=log10,
                                 transform=x->positive_or_nan(x; epsilon=1.e-30),
                                 colorrange=(datamin, datamin * 1000.0),
                                 axis_args=Dict(:bottomspinevisible=>false,
                                                :topspinevisible=>false,
                                                :leftspinevisible=>false,
                                                :rightspinevisible=>false))
            end
            if !electron && any(ri.n_neutral_species > 0 for ri ∈ run_info)
                data = get_variable(run_info, "CFL_neutral_z")
                datamin = minimum(minimum(d) for d ∈ data)
                animate_vs_vz_z(run_info, "CFL_neutral_z"; data=data, it=it,
                                outfile=plot_prefix * "CFL_neutral_z_vs_vz_z.gif",
                                colorscale=log10,
                                transform=x->positive_or_nan(x; epsilon=1.e-30),
                                colorrange=(datamin, datamin * 1000.0),
                                axis_args=Dict(:bottomspinevisible=>false,
                                               :topspinevisible=>false,
                                               :leftspinevisible=>false,
                                               :rightspinevisible=>false))
                data = get_variable(run_info, "CFL_neutral_vz")
                datamin = minimum(minimum(d) for d ∈ data)
                animate_vs_vz_z(run_info, "CFL_neutral_vz"; data=data, it=it,
                                outfile=plot_prefix * "CFL_neutral_vz_vs_vz_z.gif",
                                colorscale=log10,
                                transform=x->positive_or_nan(x; epsilon=1.e-30),
                                colorrange=(datamin, datamin * 1000.0),
                                axis_args=Dict(:bottomspinevisible=>false,
                                               :topspinevisible=>false,
                                               :leftspinevisible=>false,
                                               :rightspinevisible=>false))
            end
        catch e
            makie_post_processing_error_handler(
                e,
                "Error in timestep_diagnostics() CFL animations.")
        end
    end

    if run_info_dfns[1].dfns
        this_input_dict = input_dict_dfns
    else
        this_input_dict = input_dict
    end
    if electron
        variable_list = (v for v ∈ union((ri.evolving_variables for ri in run_info_dfns)...)
                         if occursin("electron", v))
    else
        variable_list = (v for v ∈ union((ri.evolving_variables for ri in run_info_dfns)...)
                         if !occursin("electron", v))
    end
    all_variable_names = union((ri.variable_names for ri ∈ run_info_dfns)...)

    if input.plot_timestep_residual
        try
            for variable_name ∈ variable_list
                loworder_name = variable_name * "_loworder"
                if loworder_name ∉ all_variable_names
                    # No data to calculate residual for this variable
                    continue
                end
                residual_name = variable_name * "_timestep_residual"
                if variable_name == "f_neutral"
                    plot_vs_vz_z(run_info_dfns, residual_name;
                                 input=this_input_dict[variable_name],
                                 outfile=plot_prefix * residual_name * "_vs_vz_z.pdf")
                elseif variable_name ∈ ("f", "f_electron")
                    plot_vs_vpa_z(run_info_dfns, residual_name;
                                  input=this_input_dict[variable_name],
                                  outfile=plot_prefix * residual_name * "_vs_vpa_z.pdf")
                else
                    plot_vs_z(run_info_dfns, residual_name;
                              input=this_input_dict[variable_name],
                              outfile=plot_prefix * residual_name * "_vs_z.pdf")
                end
            end
        catch e
            makie_post_processing_error_handler(
                e,
                "Error in timestep_diagnostics() timestep residual plots.")
        end
    end

    if input.animate_timestep_residual
        try
            for variable_name ∈ variable_list
                loworder_name = variable_name * "_loworder"
                if loworder_name ∉ all_variable_names
                    # No data to calculate residual for this variable
                    continue
                end
                residual_name = variable_name * "_timestep_residual"
                if variable_name == "f_neutral"
                    animate_vs_vz_z(run_info_dfns, residual_name;
                                    input=this_input_dict[variable_name],
                                    outfile=plot_prefix * residual_name * "_vs_vz_z." * this_input_dict[variable_name]["animation_ext"])
                elseif variable_name ∈ ("f", "f_electron")
                    animate_vs_vpa_z(run_info_dfns, residual_name;
                                     input=this_input_dict[variable_name],
                                     outfile=plot_prefix * residual_name * "_vs_vpa_z." * this_input_dict[variable_name]["animation_ext"])
                else
                    animate_vs_z(run_info_dfns, residual_name;
                                 input=this_input_dict[variable_name],
                                 outfile=plot_prefix * residual_name * "_vs_z." * this_input_dict[variable_name]["animation_ext"])
                end
            end
        catch e
            makie_post_processing_error_handler(
                e,
                "Error in timestep_diagnostics() timestep residual animations.")
        end
    end

    if input.plot_timestep_error
        try
            for variable_name ∈ variable_list
                loworder_name = variable_name * "_loworder"
                if loworder_name ∉ all_variable_names
                    # No data to calculate error for this variable
                    continue
                end
                error_name = variable_name * "_timestep_error"
                if variable_name == "f_neutral"
                    plot_vs_vz_z(run_info_dfns, error_name;
                                 input=this_input_dict[variable_name],
                                 outfile=plot_prefix * error_name * "_vs_vz_z.pdf")
                elseif variable_name ∈ ("f", "f_electron")
                    plot_vs_vpa_z(run_info_dfns, error_name;
                                  input=this_input_dict[variable_name],
                                  outfile=plot_prefix * error_name * "_vs_vpa_z.pdf")
                else
                    plot_vs_z(run_info_dfns, error_name;
                              input=this_input_dict[variable_name],
                              outfile=plot_prefix * error_name * "_vs_z.pdf")
                end
            end
        catch e
            makie_post_processing_error_handler(
                e,
                "Error in timestep_diagnostics() timestep error plots.")
        end
    end

    if input.animate_timestep_error
        try
            for variable_name ∈ variable_list
                loworder_name = variable_name * "_loworder"
                if loworder_name ∉ all_variable_names
                    # No data to calculate error for this variable
                    continue
                end
                error_name = variable_name * "_timestep_error"
                if variable_name == "f_neutral"
                    animate_vs_vz_z(run_info_dfns, error_name;
                                    input=this_input_dict[variable_name],
                                    outfile=plot_prefix * error_name * "_vs_vz_z." * this_input_dict[variable_name]["animation_ext"])
                elseif variable_name ∈ ("f", "f_electron")
                    animate_vs_vpa_z(run_info_dfns, error_name;
                                     input=this_input_dict[variable_name],
                                     outfile=plot_prefix * error_name * "_vs_vpa_z." * this_input_dict[variable_name]["animation_ext"])
                else
                    animate_vs_z(run_info_dfns, error_name;
                                 input=this_input_dict[variable_name],
                                 outfile=plot_prefix * error_name * "_vs_z." * this_input_dict[variable_name]["animation_ext"])
                end
            end
        catch e
            makie_post_processing_error_handler(
                e,
                "Error in timestep_diagnostics() timestep error animations.")
        end
    end

    if input.plot_steady_state_residual
        try
            for variable_name ∈ variable_list
                loworder_name = variable_name * "_loworder"
                if loworder_name ∉ all_variable_names
                    # No data to calculate residual for this variable
                    continue
                end
                residual_name = variable_name * "_steady_state_residual"
                if variable_name == "f_neutral"
                    plot_vs_vz_z(run_info_dfns, residual_name;
                                 input=this_input_dict[variable_name],
                                 outfile=plot_prefix * residual_name * "_vs_vz_z.pdf")
                elseif variable_name ∈ ("f", "f_electron")
                    plot_vs_vpa_z(run_info_dfns, residual_name;
                                  input=this_input_dict[variable_name],
                                  outfile=plot_prefix * residual_name * "_vs_vpa_z.pdf")
                else
                    plot_vs_z(run_info_dfns, residual_name;
                              input=this_input_dict[variable_name],
                              outfile=plot_prefix * residual_name * "_vs_z.pdf")
                end
            end
        catch e
            makie_post_processing_error_handler(
                e,
                "Error in timestep_diagnostics() steady state residual plots.")
        end
    end

    if input.animate_steady_state_residual
        try
            for variable_name ∈ variable_list
                loworder_name = variable_name * "_loworder"
                if loworder_name ∉ all_variable_names
                    # No data to calculate residual for this variable
                    continue
                end
                residual_name = variable_name * "_steady_state_residual"
                if variable_name == "f_neutral"
                    animate_vs_vz_z(run_info_dfns, residual_name;
                                    input=this_input_dict[variable_name],
                                    outfile=plot_prefix * residual_name * "_vs_vz_z." * this_input_dict[variable_name]["animation_ext"])
                elseif variable_name ∈ ("f", "f_electron")
                    animate_vs_vpa_z(run_info_dfns, residual_name;
                                     input=this_input_dict[variable_name],
                                     outfile=plot_prefix * residual_name * "_vs_vpa_z." * this_input_dict[variable_name]["animation_ext"])
                else
                    animate_vs_z(run_info_dfns, residual_name;
                                 input=this_input_dict[variable_name],
                                 outfile=plot_prefix * residual_name * "_vs_z." * this_input_dict[variable_name]["animation_ext"])
                end
            end
        catch e
            makie_post_processing_error_handler(
                e,
                "Error in timestep_diagnostics() steady state residual animations.")
        end
    end

    return steps_fig, dt_fig, CFL_fig
end
