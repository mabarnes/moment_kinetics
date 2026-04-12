"""
Plots related to trapping. Includes adding the trapped-passing boundary line to plots of f against
vpa and vperp. This only works for one run, as multiple runs have multiple different ax objects for
the same fig when they are 2d plots (because you can't overlay 2d plots together!).
"""
function trapping_plots(run_info, run_info_dfns, plot_prefix=nothing)
    if !isa(run_info, AbstractVector)
        run_info = Any[run_info]
    end
    if !isa(run_info_dfns, AbstractVector)
        run_info_dfns = Any[run_info_dfns]
    end
    input = Dict_to_NamedTuple(input_dict["trapping_plots"])

    if length(run_info_dfns) > 1
        error("trapping_plots() not currently implemented for multiple runs, as multiple ax arguments can't be passed to plot_2d().")
    end

    if input.plot
        println("Making plots for trapping")
        z_midpoint = run_info_dfns[1].z.n % 2 == 1 ? Int((run_info_dfns[1].z.n + 1) / 2) : Int((run_info_dfns[1].z.n) / 2)

        # define safe square root operator to just return NaN for complex results. 
        safe_sqrt(x) = x < 0 ? NaN : sqrt(x)

        if input.plot_trapped_passing_boundary_at_midplane
            variable_name = "f"
            variable = get_variable(run_info_dfns, variable_name)[1][:,:,z_midpoint,1,1,end]
            thermal_speed = get_variable(run_info_dfns, "thermal_speed")[1][:,1,1,end]
            midpoint_thermal_speed = thermal_speed[z_midpoint]
            midpoint_upar = get_variable(run_info_dfns, "parallel_flow")[1][z_midpoint,1,1,end]
            println("size(variable) = ", size(get_variable(run_info_dfns, variable_name)[1]))
            variable_prefix = plot_prefix * variable_name * "_trapped_passing_boundary"

            fig, ax, colorbar_place = get_2d_ax()
            B_top = maximum(run_info_dfns[1].geometry.Bmag[:,1])
            B_top_index = argmax(run_info_dfns[1].geometry.Bmag[:,1])
            B_bottom_index = z_midpoint
            B_bottom = run_info_dfns[1].geometry.Bmag[B_bottom_index]
            phi_top = get_variable(run_info_dfns, "phi")[1][B_top_index,1,end]
            phi_bottom = get_variable(run_info_dfns, "phi")[1][B_bottom_index,1,end]
            plot_2d(run_info_dfns[1].vpa.grid, run_info_dfns[1].vperp.grid, variable; ax=ax, 
                    colorbar_place = colorbar_place, xlabel="vpa", ylabel="vperp", title="f at midplane")

            # calculate x values and y values for trapped-passing boundary
            vpa_limits = run_info_dfns[1].vpa.grid[1], run_info_dfns[1].vpa.grid[end]
            vpa_values = range(vpa_limits[1], vpa_limits[2], length=3000)
            vperp_values = similar(vpa_values)

            # line is different depending on velocity coordinates
            if run_info_dfns[1].evolve_density && run_info_dfns[1].evolve_upar && run_info_dfns[1].evolve_p
                @. vperp_values = safe_sqrt( (((vpa_values*midpoint_thermal_speed + midpoint_upar))^2 - 2 * (phi_top - phi_bottom))*(B_bottom/(B_top - B_bottom)) ) / midpoint_thermal_speed
            elseif !run_info_dfns[1].evolve_density && !run_info_dfns[1].evolve_upar && !run_info_dfns[1].evolve_p
                @. vperp_values = safe_sqrt( (((vpa_values))^2 - 2 * (phi_top - phi_bottom))*(B_bottom/(B_top - B_bottom)) )
            else
                error("Trapped-passing boundary plotting not implemented for case where some but not all of n, upar and p are evolved.")
            end

            # Use the limits of the existing plot so that lines!() doesn't change the limits
            xl = ax.finallimits[].origin[1], ax.finallimits[].origin[1] + ax.finallimits[].widths[1]
            yl = ax.finallimits[].origin[2], ax.finallimits[].origin[2] + ax.finallimits[].widths[2]
            lines!(ax, vpa_values, vperp_values, color=:red, linewidth = 1.0)
            xlims!(ax, xl...)
            ylims!(ax, yl...)
            outfile = variable_prefix * "_vs_vpa_vperp.pdf"
            save(outfile, fig)

        end

        if input.plot_Ez_and_dBdz
            Ez = get_variable(run_info_dfns, "Ez")[1][:,1,end]
            dBdz = run_info_dfns[1].geometry.dBdz[:,1]
            fig, ax, = get_1d_ax()
            plot_1d(run_info_dfns[1].z.grid, -dBdz; ax=ax, xlabel="z", ylabel="dBdz", title="Ez and dB/dz vs z")
            autolimits!(ax)
            yl = ax.finallimits[].origin[2], ax.finallimits[].origin[2] + ax.finallimits[].widths[2]
            lines!(ax, run_info_dfns[1].z.grid, Ez, color=:red, label="Ez")
            ylims!(ax, yl)
            # plot_1d(run_info_dfns[1].z.grid, Ez; ax=ax, xlabel="z", ylabel="Ez", title="Ez vs z")
            # lines!(ax, run_info_dfns[1].z.grid, -dBdz, color=:red, label="dB/dz")
            outfile = plot_prefix * "Ez_and_dBdz_vs_z.pdf"
            save(outfile, fig)

        end
    end
end
