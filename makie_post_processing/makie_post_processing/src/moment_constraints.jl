using moment_kinetics.calculus: integral

"""
    check_moment_constraints(run_info, is_neutral; input, plot_prefix)

Plots to check moment constraints. Comparison plots not currently supported.
"""
function check_moment_constraints end

function check_moment_constraints(run_info::Tuple, is_neutral; input, plot_prefix)
    if !input.check_moments
        return nothing
    end

    # For now, don't support comparison plots
    if length(run_info) > 1
        error("Comparison plots not supported by check_moment_constraints()")
    end
    return check_moment_constraints(run_info[1], is_neutral; input=input,
                                    plot_prefix=plot_prefix)
end

function check_moment_constraints(run_info, is_neutral; input, plot_prefix)
    if !input.check_moments
        return nothing
    end

    # For now assume there is only one ion or neutral species
    is = 1

    if is_neutral
        fn = get_variable(run_info, "f_neutral")
        if run_info.evolve_density
            moment = zeros(run_info.z.n, run_info.r.n, run_info.nt)
            for it ∈ 1:run_info.nt, ir ∈ 1:run_info.r.n, iz ∈ 1:run_info.z.n
                moment[iz,ir,it] = integral(
                    @view(fn[:,:,:,iz,ir,is,it]), run_info.vz.grid, 0, run_info.vz.wgts,
                    run_info.vr.grid, 0, run_info.vr.wgts, run_info.vzeta.grid, 0,
                    run_info.vzeta.wgts)
            end
            error = moment .- 1.0
            animate_vs_z(run_info, "density moment neutral"; data=error, input=input,
                         outfile=plot_prefix * "density_moment_neutral_check.gif")
        end

        if run_info.evolve_upar
            moment = zeros(run_info.z.n, run_info.r.n, run_info.nt)
            for it ∈ 1:run_info.nt, ir ∈ 1:run_info.r.n, iz ∈ 1:run_info.z.n
                moment[iz,ir,it] = integral(
                    @view(fn[:,:,:,iz,ir,is,it]), run_info.vz.grid, 1, run_info.vz.wgts,
                    run_info.vr.grid, 0, run_info.vr.wgts, run_info.vzeta.grid, 0,
                    run_info.vzeta.wgts)
            end
            error = moment
            animate_vs_z(run_info, "parallel flow neutral"; data=error, input=input,
                         outfile=plot_prefix * "parallel_flow_moment_neutral_check.gif")
        end

        if run_info.evolve_p
            moment = zeros(run_info.z.n, run_info.r.n, run_info.nt)
            for it ∈ 1:run_info.nt, ir ∈ 1:run_info.r.n, iz ∈ 1:run_info.z.n
                moment[iz,ir,it] = integral(
                    (vzeta,vr,vz) -> 0.5*(vzeta^2 + vr^2 + vz^2),
                    @view(fn[:,:,:,iz,ir,is,it]), run_info.vzeta, run_info.vr,
                    run_info.vz)
            end
            error = moment .- 1.5
            animate_vs_z(run_info, "pressure neutral"; data=error, input=input,
                         outfile=plot_prefix * "pressure_moment_neutral_check.gif")
        end
    else
        f = get_variable(run_info, "f")
        if run_info.evolve_density
            moment = zeros(run_info.z.n, run_info.r.n, run_info.nt)
            for it ∈ 1:run_info.nt, ir ∈ 1:run_info.r.n, iz ∈ 1:run_info.z.n
                moment[iz,ir,it] = integral(
                    @view(f[:,:,iz,ir,is,it]), run_info.vpa.grid, 0, run_info.vpa.wgts,
                    run_info.vperp.grid, 0, run_info.vperp.wgts)
            end
            error = moment .- 1.0
            animate_vs_z(run_info, "density moment"; data=error, input=input,
                         outfile=plot_prefix * "density_moment_check.gif")
        end

        if run_info.evolve_upar
            moment = zeros(run_info.z.n, run_info.r.n, run_info.nt)
            for it ∈ 1:run_info.nt, ir ∈ 1:run_info.r.n, iz ∈ 1:run_info.z.n
                moment[iz,ir,it] = integral(
                    @view(f[:,:,iz,ir,is,it]), run_info.vpa.grid, 1, run_info.vpa.wgts,
                    run_info.vperp.grid, 0, run_info.vperp.wgts)
            end
            error = moment
            animate_vs_z(run_info, "parallel flow moment"; data=error, input=input,
                         outfile=plot_prefix * "parallel_flow_moment_check.gif")
        end

        if run_info.evolve_p
            moment = zeros(run_info.z.n, run_info.r.n, run_info.nt)
            for it ∈ 1:run_info.nt, ir ∈ 1:run_info.r.n, iz ∈ 1:run_info.z.n
                moment[iz,ir,it] = integral((vperp,vpa) -> 0.5*(vperp^2+vpa^2),
                                            @view(f[:,:,iz,ir,is,it]), run_info.vperp,
                                            run_info.vpa)
            end
            error = moment .- 1.5
            animate_vs_z(run_info, "pressure moment"; data=error, input=input,
                         outfile=plot_prefix * "pressure_moment_check.gif")
        end
    end

    return nothing
end

"""
    constraints_plots(run_info; plot_prefix=plot_prefix)

Plot and/or animate the coefficients used to correct the normalised distribution
function(s) (aka shape functions) to obey the moment constraints.

If there were no discretisation errors, we would have \$A=1\$, \$B=0\$, \$C=0\$. The
plots/animations show \$(A-1)\$ so that all three coefficients can be shown nicely on the
same axes.
"""
function constraints_plots(run_info; plot_prefix=plot_prefix)
    input = Dict_to_NamedTuple(input_dict["constraints"])

    if !(input.plot || input.animate)
        return nothing
    end

    try
        println("Making plots of moment constraints coefficients")

        if !isa(run_info, Tuple)
            run_info = (run_info,)
        end

        it0 = input.it0
        ir0 = input.ir0

        if input.plot
            if any(ri.evolve_density || ri.evolve_upar || ri.evolve_p
                   for ri ∈ run_info)

                # Ions
                frame_index = Observable(1)
                fig, ax = get_1d_ax(; xlabel="z", ylabel="constraint coefficient")
                for ri ∈ run_info
                    if !(ri.evolve_density || ri.evolve_upar || ri.evolve_p)
                        continue
                    end
                    nspecies = ri.n_ion_species
                    for is ∈ 1:nspecies
                        if length(run_info) > 1
                            prefix = ri.run_name * ", "
                        else
                            prefix = ""
                        end
                        if nspecies > 1
                            suffix = ", species $is"
                        else
                            suffix = ""
                        end

                        varname = "ion_constraints_A_coefficient"
                        label = prefix * "(A-1)" * suffix
                        data = get_variable(ri, varname; it=it0, is=is, ir=ir0)
                        data .-= 1.0
                        plot_vs_z(ri, varname; label=label, data=data, ax=ax, input=input)

                        varname = "ion_constraints_B_coefficient"
                        label = prefix * "B" * suffix
                        plot_vs_z(ri, varname; label=label, ax=ax, it=it0, is=is, ir=ir0,
                                  input=input)

                        varname = "ion_constraints_C_coefficient"
                        label = prefix * "C" * suffix
                        plot_vs_z(ri, varname; label=label, ax=ax, it=it0, is=is, ir=ir0,
                                  input=input)
                    end
                end
                put_legend_below(fig, ax)
                # Ensure the first row width is 3/4 of the column width so that
                # the plot does not get squashed by the legend
                rowsize!(fig.layout, 1, Aspect(1, 3/4))
                resize_to_layout!(fig)
                save(plot_prefix * "ion_constraints.pdf", fig)
            end

            # Neutrals
            if any(ri.n_neutral_species > 0
                   && (ri.evolve_density || ri.evolve_upar || ri.evolve_p)
                   for ri ∈ run_info)

                fig, ax = get_1d_ax(; xlabel="z", ylabel="constraint coefficient")
                for ri ∈ run_info
                    if !(ri.evolve_density || ri.evolve_upar || ri.evolve_p)
                        continue
                    end
                    nspecies = ri.n_neutral_species
                    for is ∈ 1:nspecies
                        if length(run_info) > 1
                            prefix = ri.run_name * ", "
                        else
                            prefix = ""
                        end
                        if nspecies > 1
                            suffix = ", species $is"
                        else
                            suffix = ""
                        end

                        varname = "neutral_constraints_A_coefficient"
                        label = prefix * "(A-1)" * suffix
                        data = get_variable(ri, varname; it=it0, is=is, ir=ir0)
                        data .-= 1.0
                        plot_vs_z(ri, varname; label=label, data=data, ax=ax, input=input)

                        varname = "neutral_constraints_B_coefficient"
                        label = prefix * "B" * suffix
                        plot_vs_z(ri, varname; label=label, ax=ax, it=it0, is=is, ir=ir0,
                                  input=input)

                        varname = "neutral_constraints_C_coefficient"
                        label = prefix * "C" * suffix
                        plot_vs_z(ri, varname; label=label, ax=ax, it=it0, is=is, ir=ir0,
                                  input=input)
                    end
                end
                put_legend_below(fig, ax)
                # Ensure the first row width is 3/4 of the column width so that
                # the plot does not get squashed by the legend
                rowsize!(fig.layout, 1, Aspect(1, 3/4))
                resize_to_layout!(fig)
                save(plot_prefix * "neutral_constraints.pdf", fig)
            end

            # Electrons
            if any(ri.composition.electron_physics ∈ (kinetic_electrons,
                                                      kinetic_electrons_with_temperature_equation)
                   for ri ∈ run_info)

                fig, ax = get_1d_ax(; xlabel="z", ylabel="constraint coefficient")
                for ri ∈ run_info
                    if length(run_info) > 1
                        prefix = ri.run_name * ", "
                    else
                        prefix = ""
                    end

                    varname = "electron_constraints_A_coefficient"
                    label = prefix * "(A-1)"
                    data = get_variable(ri, varname; it=it0, ir=ir0)
                    data .-= 1.0
                    plot_vs_z(ri, varname; label=label, data=data, ax=ax, input=input)

                    varname = "electron_constraints_B_coefficient"
                    label = prefix * "B"
                    plot_vs_z(ri, varname; label=label, ax=ax, it=it0, ir=ir0,
                              input=input)

                    varname = "electron_constraints_C_coefficient"
                    label = prefix * "C"
                    plot_vs_z(ri, varname; label=label, ax=ax, it=it0, ir=ir0,
                              input=input)
                end
                put_legend_below(fig, ax)
                # Ensure the first row width is 3/4 of the column width so that
                # the plot does not get squashed by the legend
                rowsize!(fig.layout, 1, Aspect(1, 3/4))
                resize_to_layout!(fig)
                save(plot_prefix * "electron_constraints.pdf", fig)
            end
        end

        if input.animate
            nt = minimum(ri.nt for ri ∈ run_info)

            if any(ri.evolve_density || ri.evolve_upar || ri.evolve_p
                   for ri ∈ run_info)

                # Ions
                frame_index = Observable(1)
                fig, ax = get_1d_ax(; xlabel="z", ylabel="constraint coefficient")

                # Calculate plot limits manually so we can exclude the first time point, which
                # often has a large value for (A-1) due to the way initialisation is done,
                # which can make the subsequent values hard to see.
                ymin = Inf
                ymax = -Inf
                for ri ∈ run_info
                    if !(ri.evolve_density || ri.evolve_upar || ri.evolve_p)
                        continue
                    end
                    nspecies = ri.n_ion_species
                    for is ∈ 1:nspecies
                        if length(run_info) > 1
                            prefix = ri.run_name * ", "
                        else
                            prefix = ""
                        end
                        if nspecies > 1
                            suffix = ", species $is"
                        else
                            suffix = ""
                        end

                        varname = "ion_constraints_A_coefficient"
                        label = prefix * "(A-1)" * suffix
                        data = get_variable(ri, varname; is=is, ir=ir0)
                        data .-= 1.0
                        ymin = min(ymin, minimum(data[:,2:end]))
                        ymax = max(ymax, maximum(data[:,2:end]))
                        animate_vs_z(ri, varname; label=label, data=data,
                                     frame_index=frame_index, ax=ax, input=input)

                        varname = "ion_constraints_B_coefficient"
                        label = prefix * "B" * suffix
                        data = get_variable(ri, varname; is=is, ir=ir0)
                        ymin = min(ymin, minimum(data[:,2:end]))
                        ymax = max(ymax, maximum(data[:,2:end]))
                        animate_vs_z(ri, varname; label=label, data=data,
                                     frame_index=frame_index, ax=ax, is=is, ir=ir0,
                                     input=input)

                        varname = "ion_constraints_C_coefficient"
                        label = prefix * "C" * suffix
                        data = get_variable(ri, varname; is=is, ir=ir0)
                        ymin = min(ymin, minimum(data[:,2:end]))
                        ymax = max(ymax, maximum(data[:,2:end]))
                        animate_vs_z(ri, varname; label=label, data=data,
                                     frame_index=frame_index, ax=ax, is=is, ir=ir0,
                                     input=input)
                    end
                end
                put_legend_below(fig, ax)
                # Ensure the first row width is 3/4 of the column width so that
                # the plot does not get squashed by the legend
                rowsize!(fig.layout, 1, Aspect(1, 3/4))
                resize_to_layout!(fig)
                ylims!(ax, ymin, ymax)
                save_animation(fig, frame_index, nt,
                               plot_prefix * "ion_constraints." * input.animation_ext)
            end

            # Neutrals
            if any(ri.n_neutral_species > 0
                   && (ri.evolve_density || ri.evolve_upar || ri.evolve_p)
                   for ri ∈ run_info)

                frame_index = Observable(1)
                fig, ax = get_1d_ax(; xlabel="z", ylabel="constraint coefficient")

                # Calculate plot limits manually so we can exclude the first time point, which
                # often has a large value for (A-1) due to the way initialisation is done,
                # which can make the subsequent values hard to see.
                ymin = Inf
                ymax = -Inf
                for ri ∈ run_info
                    if !(ri.evolve_density || ri.evolve_upar || ri.evolve_p)
                        continue
                    end
                    nspecies = ri.n_neutral_species
                    for is ∈ 1:nspecies
                        if length(run_info) > 1
                            prefix = ri.run_name * ", "
                        else
                            prefix = ""
                        end
                        if nspecies > 1
                            suffix = ", species $is"
                        else
                            suffix = ""
                        end

                        varname = "neutral_constraints_A_coefficient"
                        label = prefix * "(A-1)" * suffix
                        data = get_variable(ri, varname; is=is, ir=ir0)
                        data .-= 1.0
                        ymin = min(ymin, minimum(data[:,2:end]))
                        ymax = max(ymax, maximum(data[:,2:end]))
                        animate_vs_z(ri, varname; label=label, data=data,
                                     frame_index=frame_index, ax=ax, input=input)

                        varname = "neutral_constraints_B_coefficient"
                        label = prefix * "B" * suffix
                        data = get_variable(ri, varname; is=is, ir=ir0)
                        ymin = min(ymin, minimum(data[:,2:end]))
                        ymax = max(ymax, maximum(data[:,2:end]))
                        animate_vs_z(ri, varname; label=label, data=data,
                                     frame_index=frame_index, ax=ax, is=is, ir=ir0,
                                     input=input)

                        varname = "neutral_constraints_C_coefficient"
                        label = prefix * "C" * suffix
                        data = get_variable(ri, varname; is=is, ir=ir0)
                        ymin = min(ymin, minimum(data[:,2:end]))
                        ymax = max(ymax, maximum(data[:,2:end]))
                        animate_vs_z(ri, varname; label=label, data=data,
                                     frame_index=frame_index, ax=ax, is=is, ir=ir0,
                                     input=input)
                    end
                end
                put_legend_below(fig, ax)
                # Ensure the first row width is 3/4 of the column width so that
                # the plot does not get squashed by the legend
                rowsize!(fig.layout, 1, Aspect(1, 3/4))
                resize_to_layout!(fig)
                ylims!(ax, ymin, ymax)
                save_animation(fig, frame_index, nt,
                               plot_prefix * "neutral_constraints." * input.animation_ext)
            end

            # Electrons
            if any(ri.composition.electron_physics ∈ (kinetic_electrons,
                                                      kinetic_electrons_with_temperature_equation)
                   for ri ∈ run_info)

                frame_index = Observable(1)
                fig, ax = get_1d_ax(; xlabel="z", ylabel="constraint coefficient")

                # Calculate plot limits manually so we can exclude the first time point, which
                # often has a large value for (A-1) due to the way initialisation is done,
                # which can make the subsequent values hard to see.
                ymin = Inf
                ymax = -Inf
                for ri ∈ run_info
                    if length(run_info) > 1
                        prefix = ri.run_name * ", "
                    else
                        prefix = ""
                    end

                    varname = "electron_constraints_A_coefficient"
                    label = prefix * "(A-1)"
                    data = get_variable(ri, varname; ir=ir0)
                    data .-= 1.0
                    ymin = min(ymin, minimum(data[:,2:end]))
                    ymax = max(ymax, maximum(data[:,2:end]))
                    animate_vs_z(ri, varname; label=label, data=data,
                                 frame_index=frame_index, ax=ax, input=input)

                    varname = "electron_constraints_B_coefficient"
                    label = prefix * "B"
                    data = get_variable(ri, varname; ir=ir0)
                    ymin = min(ymin, minimum(data[:,2:end]))
                    ymax = max(ymax, maximum(data[:,2:end]))
                    animate_vs_z(ri, varname; label=label, data=data,
                                 frame_index=frame_index, ax=ax, ir=ir0, input=input)

                    varname = "electron_constraints_C_coefficient"
                    label = prefix * "C"
                    data = get_variable(ri, varname; ir=ir0)
                    ymin = min(ymin, minimum(data[:,2:end]))
                    ymax = max(ymax, maximum(data[:,2:end]))
                    animate_vs_z(ri, varname; label=label, data=data,
                                 frame_index=frame_index, ax=ax, ir=ir0, input=input)
                end
                put_legend_below(fig, ax)
                # Ensure the first row width is 3/4 of the column width so that
                # the plot does not get squashed by the legend
                rowsize!(fig.layout, 1, Aspect(1, 3/4))
                resize_to_layout!(fig)
                ylims!(ax, ymin, ymax)
                save_animation(fig, frame_index, nt,
                               plot_prefix * "electron_constraints." * input.animation_ext)
            end
        end
    catch e
        return makie_post_processing_error_handler(
                   e,
                   "Error in constraints_plots().")
    end
end
