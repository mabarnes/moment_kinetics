using CairoMakie
using LsqFit
using makie_post_processing
using makie_post_processing.Printf
using StatsBase

# Increase this well above 1.0 to get poster-quality plots, but because we are outputting
# as png here, also increases file size quite a bit.
resolution_increase_factor = 1.0

plots_dir = "publication_inputs/APS2025"

# 1D backgrounds
for (sim_dir, label) ∈ (("runs/1D1V-instability-test/", "background_1D"),
                        ("runs/1D1V-instability-test_no-Krook/", "background_1D_no-Krook")
                       )
    dir_1d = mkpath(joinpath(plots_dir, label))

    # load data
    ri_1d = get_run_info(sim_dir; dfns=true)

    parallel_coordinate = ri_1d.z.grid ./ ri_1d.geometry.input.pitch

    # Plot sources
    source_amplitude = get_variable(ri_1d, "external_source_amplitude"; it=ri_1d.nt, is=1, ir=1)[:,1]
    source_T = get_variable(ri_1d, "external_source_T_array"; it=ri_1d.nt, is=1, ir=1)[:,1]

    fig_1d, ax_1d, l = lines(parallel_coordinate, source_amplitude; label="source amplitude")
    lines!(ax_1d, parallel_coordinate, source_T; label="source temperature")
    ax_1d.xlabel = "parallel distance"
    Legend(fig_1d[2,1], ax_1d; tellheight=true, tellwidth=false)

    save(joinpath(dir_1d, label * "_source.png"), fig_1d; px_per_unit=resolution_increase_factor)

    # Plot moment profiles
    n = get_variable(ri_1d, "density"; it=ri_1d.nt, is=1, ir=1)
    T = get_variable(ri_1d, "temperature"; it=ri_1d.nt, is=1, ir=1)

    fig_1d, ax_1d, l = lines(parallel_coordinate, n; label="n")
    lines!(ax_1d, parallel_coordinate, T; label="T")
    ax_1d.xlabel = "parallel distance"
    Legend(fig_1d[2,1], ax_1d; tellheight=true, tellwidth=false)

    save(joinpath(dir_1d, label * "_nT_profiles.png"), fig_1d; px_per_unit=resolution_increase_factor)

    u = get_variable(ri_1d, "parallel_flow"; it=ri_1d.nt, is=1, ir=1)

    fig_1d, ax_1d, l = lines(parallel_coordinate, u; label="u")
    ax_1d.xlabel = "parallel distance"
    ax_1d.ylabel = "u_∥"
    Legend(fig_1d[2,1], ax_1d; tellheight=true, tellwidth=false)

    save(joinpath(dir_1d, label * "_upar_profiles.png"), fig_1d; px_per_unit=resolution_increase_factor)

    if label == "background_1D"
        nu_ii_Krook = get_variable(ri_1d, "Krook_collision_frequency_ii"; it=ri_1d.nt,
                                   is=1, ir=1)

        fig_1d, ax_1d, l = lines(parallel_coordinate, nu_ii_Krook; label="nu_ii_Krook")
        ax_1d.xlabel = "parallel distance"
        ax_1d.ylabel = "nu_ii"
        Legend(fig_1d[2,1], ax_1d; tellheight=true, tellwidth=false)

        save(joinpath(dir_1d, label * "_nu_ii_Krook_profiles.png"), fig_1d; px_per_unit=resolution_increase_factor)
    end

    # Plot distribution function
    f = get_variable(ri_1d, "f"; it=ri_1d.nt, is=1, ir=1, ivperp=1)

    fig_1d, ax_1d, hm = heatmap(ri_1d.vpa.grid, parallel_coordinate, f)
    Colorbar(fig_1d[1,2], hm)
    ax_1d.xlabel = "parallel distance"
    ax_1d.ylabel = "v_∥"

    save(joinpath(dir_1d, label * "_f.png"), fig_1d; px_per_unit=resolution_increase_factor)

    # Classical particle diffusivity and classical heat diffusivity.
    nu_ii = get_variable(ri_1d, "Krook_collision_frequency_ii"; it=ri_1d.nt, is=1, ir=1)
    nu_ei = get_variable(ri_1d, "collision_frequency_ei"; it=ri_1d.nt, is=1, ir=1)
    # Dimensionless Ω_i = Ω_{i,ref} (as B=B_ref is constant) is 1/rhostar due to reference
    # parameter definitions (see
    # https://mabarnes.github.io/moment_kinetics/dev/moment_kinetic_equations/#Dimensionless-equations-for-code).
    Omega_i = 1.0 / ri_1d.geometry.rhostar
    Omega_e = Omega_i / ri_1d.composition.me_over_mi
    vth_i = get_variable(ri_1d, "thermal_speed"; it=ri_1d.nt, is=1, ir=1)
    vth_e = get_variable(ri_1d, "electron_thermal_speed"; it=ri_1d.nt, is=1, ir=1)
    rho_i = vth_i ./ Omega_i
    rho_e = vth_e ./ Omega_e
    D_classical = @. rho_e^2 * nu_ei
    chi_i_classical = @. rho_i^2 * nu_ii

    fig_1d, ax_1d, l = lines(parallel_coordinate, D_classical)
    ax_1d.xlabel = "parallel distance"
    ax_1d.ylabel = "D_classical"

    save(joinpath(dir_1d, label * "_D_classical.png"), fig_1d; px_per_unit=resolution_increase_factor)

    fig_1d, ax_1d, l = lines(parallel_coordinate, chi_i_classical)
    ax_1d.xlabel = "parallel distance"
    ax_1d.ylabel = "chi_i_classical"

    save(joinpath(dir_1d, label * "_chi_i_classical.png"), fig_1d; px_per_unit=resolution_increase_factor)

    fig_1d, ax_1d, l = lines(parallel_coordinate, rho_i)
    ax_1d.xlabel = "parallel distance"
    ax_1d.ylabel = "rho_i"

    save(joinpath(dir_1d, label * "_rho_i.png"), fig_1d; px_per_unit=resolution_increase_factor)
end

# Instability analysis - scan in radial resolution
dir_r_nelement_scan = mkpath(joinpath(plots_dir, "instability_2D_r-resolution-scan"))
r_r_nelement_scan_run_dirs = ("runs/2D1V-instability-test_Lr1cm/",
                            "runs/2D1V-instability-test_Lr1cm-rnelement8/",
                            "runs/2D1V-instability-test_Lr1cm-rnelement16/",
                            "runs/2D1V-instability-test_Lr1cm-rnelement32/",
                           )
ri_r_nelement_scan = get_run_info(r_r_nelement_scan_run_dirs...; dfns=true)

function plot_mode_amplitude(this_ri, phi, ax, irun;
                             label="r_nelement = $(this_ri.r.nelement_global)")
    # Take a crude but hopefully robust estimate of the mode amplitude.
    # Take the maximum over z of the the RMS over r of the variable.
    # Pass `corrected=false` to `StatsBase.std` so that it just calculates the RMS,
    # not the unbiased standard deviation.
    amplitude = maximum(std(phi; corrected=false, dims=2); dims=1)[1,1,:]

    # Fit A*exp(γ*time) to exponentially-growing interval.
    γ, A, tmin, tmax = makie_post_processing.partial_fit_exponential_growth(this_ri.time, amplitude)
    println(this_ri.run_name, " γ = $γ")

    lines!(ax, this_ri.time, amplitude; color=Cycled(irun), linestyle=:dot)

    # Plot the fitted exponential growth
    fit_t = this_ri.time[@. this_ri.time > tmin && this_ri.time < tmax]
    fit_amplitude = @. A * exp(γ * fit_t)
    lines!(ax, fit_t, fit_amplitude; color=Cycled(irun), label=label)
    label_t = 0.5 * (fit_t[1] + fit_t[end])
    gamma_string = @sprintf("%.5g", γ)
    with_theme(Text=(; cycle=:color)) do
        text!(ax, Point2f(label_t, A * exp(γ * label_t)); text="γ = $gamma_string",
              align=(:left, :top), color=Cycled(irun))
    end

    return γ
end

fig, ax = get_1d_ax(xlabel="time", ylabel="amplitude", yscale=log10)
# Ensure the first row width is 3/4 of the column width so that the plot does not get
# squashed by the legend
rowsize!(fig.layout, 1, Aspect(1, 3/4))

for (irun, this_ri) ∈ enumerate(ri_r_nelement_scan)
    phi = get_variable(this_ri, "phi")

    plot_mode_amplitude(this_ri, phi, ax, irun)

    # make animation of perturbation
    _, perturbation = makie_post_processing.get_r_perturbation(phi)
    outfile = joinpath(dir_r_nelement_scan, "phi_perturbation_r-nelement$(this_ri.r.nelement_global).gif")
    title = "r_nelement = $(this_ri.r.nelement_global)"
    makie_post_processing.animate_2d(this_ri.z.grid, this_ri.r.grid, perturbation,
                                     xlabel="z", ylabel="r", title=title,
                                     colormap="reverse_deep", outfile=outfile)

    # Plot final time point
    final_perturbation = @view perturbation[:,:,end]
    final_fig, final_ax, hm = heatmap(this_ri.z.grid, this_ri.r.grid, final_perturbation)
    Colorbar(final_fig[1,2], hm)
    final_ax.xlabel = "z"
    final_ax.ylabel = "r"
    final_ax.title = "r_nelement = $(this_ri.r.nelement_global)"
    save(joinpath(dir_r_nelement_scan,
                  "final_phi_perturbation_r-nelement$(this_ri.r.nelement_global).png"),
                  final_fig; px_per_unit=resolution_increase_factor)
end

Legend(fig[2,1], ax; tellheight=true, tellwidth=false)
resize_to_layout!(fig)
save(joinpath(dir_r_nelement_scan, "growth_rate_r-resolution-scan.png"), fig; px_per_unit=resolution_increase_factor)

# Check convergence with r_nelement of stabilisation at Dr=1e-8.
stabilised_run_dirs = ("runs/2D1V-instability-test_Lr1cm_rdiss1e-8/",
                       "runs/2D1V-instability-test_Lr1cm-rnelement16-rdiss1e-8/",
                       "runs/2D1V-instability-test_Lr1cm-rnelement32-rdiss1e-8/",
                      )
ri_stabilised = get_run_info(stabilised_run_dirs...; dfns=true)

fig, ax = get_1d_ax(xlabel="time", ylabel="amplitude", yscale=log10)
# Ensure the first row width is 3/4 of the column width so that the plot does not get
# squashed by the legend
rowsize!(fig.layout, 1, Aspect(1, 3/4))

for (irun, this_ri) ∈ enumerate(ri_stabilised)
    phi = get_variable(this_ri, "phi")
    plot_mode_amplitude(this_ri, phi, ax, irun)
end

Legend(fig[2,1], ax; tellheight=true, tellwidth=false)
resize_to_layout!(fig)
save(joinpath(dir_r_nelement_scan, "stabilised-resolution-scan.png"), fig; px_per_unit=resolution_increase_factor)

# Show apparently converged mode with Dr=7e-9
converged_rdiss7em9_run_dirs = ("runs/2D1V-instability-test_Lr1cm_rdiss7e-9/",
                                "runs/2D1V-instability-test_Lr1cm-rnelement8-rdiss7e-9/",
                                "runs/2D1V-instability-test_Lr1cm-rnelement16-rdiss7e-9/",
                                #"runs/2D1V-instability-test_Lr1cm-rnelement32-rdiss7e-9/",
                               )
ri_converged = get_run_info(converged_rdiss7em9_run_dirs...; dfns=true)

fig, ax = get_1d_ax(xlabel="time", ylabel="amplitude", yscale=log10)
# Ensure the first row width is 3/4 of the column width so that the plot does not get
# squashed by the legend
rowsize!(fig.layout, 1, Aspect(1, 3/4))

for (irun, this_ri) ∈ enumerate(ri_converged)
    phi = get_variable(this_ri, "phi")

    plot_mode_amplitude(this_ri, phi, ax, irun)

    # make animation of perturbation
    _, perturbation = makie_post_processing.get_r_perturbation(phi)
    outfile = joinpath(dir_r_nelement_scan, "phi_perturbation_rdiss7e-9_r-nelement$(this_ri.r.nelement_global).gif")
    title = "r_nelement = $(this_ri.r.nelement_global), Dr = 7e-9"
    makie_post_processing.animate_2d(this_ri.z.grid, this_ri.r.grid, perturbation,
                                     xlabel="z", ylabel="r", title=title,
                                     colormap="reverse_deep", outfile=outfile)

    # Plot final time point
    final_perturbation = @view perturbation[:,:,end]
    final_fig, final_ax, hm = heatmap(this_ri.z.grid, this_ri.r.grid, final_perturbation)
    Colorbar(final_fig[1,2], hm)
    final_ax.xlabel = "z"
    final_ax.ylabel = "r"
    final_ax.title = "r_nelement = $(this_ri.r.nelement_global), Dr = 7e-9"
    save(joinpath(dir_r_nelement_scan,
                  "final_phi_perturbation_rdiss7e-9_r-nelement$(this_ri.r.nelement_global).png"),
                  final_fig; px_per_unit=resolution_increase_factor)
end

Legend(fig[2,1], ax; tellheight=true, tellwidth=false)
resize_to_layout!(fig)
save(joinpath(dir_r_nelement_scan, "converged-rdiss7e-9-resolution-scan.png"), fig; px_per_unit=resolution_increase_factor)

# Scan dissipation between Dr=1e-9 (unsure if the instability is radial-grid-scale) and Dr=7e-9 (instability seems well resolved).
dir_rdiss_scan = mkpath(joinpath(plots_dir, "instability_2D_rdiss-scan"))
rdiss_scan_run_dirs = ("runs/2D1V-instability-test_Lr1cm-rnelement16-rdiss1e-9/",
                       "runs/2D1V-instability-test_Lr1cm-rnelement16-rdiss2e-9/",
                       "runs/2D1V-instability-test_Lr1cm-rnelement16-rdiss3e-9/",
                       "runs/2D1V-instability-test_Lr1cm-rnelement16-rdiss4e-9/",
                       "runs/2D1V-instability-test_Lr1cm-rnelement16-rdiss5e-9/",
                       "runs/2D1V-instability-test_Lr1cm-rnelement16-rdiss6e-9/",
                       "runs/2D1V-instability-test_Lr1cm-rnelement16-rdiss7e-9/",
                      )
ri_rdiss_scan = get_run_info(rdiss_scan_run_dirs...; dfns=true)

fig, ax = get_1d_ax(xlabel="time", ylabel="amplitude", yscale=log10)
# Ensure the first row width is 3/4 of the column width so that the plot does not get
# squashed by the legend
rowsize!(fig.layout, 1, Aspect(1, 3/4))

for (irun, this_ri) ∈ enumerate(ri_rdiss_scan)
    phi = get_variable(this_ri, "phi")

    plot_mode_amplitude(this_ri, phi, ax, irun; label="rdiss = $(irun)e-9")

    # make animation of perturbation
    _, perturbation = makie_post_processing.get_r_perturbation(phi)
    outfile = joinpath(dir_rdiss_scan, "phi_perturbation_rdiss$(irun)e-9.gif")
    title = "r_nelement = 16, Dr = $(irun)e-9"
    makie_post_processing.animate_2d(this_ri.z.grid, this_ri.r.grid, perturbation,
                                     xlabel="z", ylabel="r", title=title,
                                     colormap="reverse_deep", outfile=outfile)

    # Plot final time point
    final_perturbation = @view perturbation[:,:,end]
    final_fig, final_ax, hm = heatmap(this_ri.z.grid, this_ri.r.grid, final_perturbation)
    Colorbar(final_fig[1,2], hm)
    final_ax.xlabel = "z"
    final_ax.ylabel = "r"
    final_ax.title = "r_nelement = 16, Dr = $(irun)e-9"
    save(joinpath(dir_rdiss_scan,
                  "final_phi_perturbation_rdiss$(irun)e-9.png"),
                  final_fig; px_per_unit=resolution_increase_factor)
end

Legend(fig[2,1], ax; tellheight=true, tellwidth=false)
resize_to_layout!(fig)
save(joinpath(dir_rdiss_scan, "rdiss-scan.png"), fig; px_per_unit=resolution_increase_factor)

# Compare case with Krook collisions switched off
dir_no_Krook = mkpath(joinpath(plots_dir, "instability_2D_no-Krook"))
no_Krook_run_dirs = ("runs/2D1V-instability-test_Lr1cm-no-Krook/",
                     "runs/2D1V-instability-test_Lr1cm-rnelement8-no-Krook/",
                     "runs/2D1V-instability-test_Lr1cm-rnelement16-no-Krook/",
                     "runs/2D1V-instability-test_Lr1cm-rnelement32-no-Krook/",
                    )
ri_no_Krook = get_run_info(no_Krook_run_dirs...; dfns=true)
fig, ax = get_1d_ax(xlabel="time", ylabel="amplitude", yscale=log10)
# Ensure the first row width is 3/4 of the column width so that the plot does not get
# squashed by the legend
rowsize!(fig.layout, 1, Aspect(1, 3/4))

for (irun, this_ri) ∈ enumerate(ri_no_Krook)
    phi = get_variable(this_ri, "phi")

    plot_mode_amplitude(this_ri, phi, ax, irun)

    # make animation of perturbation
    _, perturbation = makie_post_processing.get_r_perturbation(phi)
    outfile = joinpath(dir_no_Krook, "phi_perturbation_no-Krook_r-nelement$(this_ri.r.nelement_global).gif")
    title = "no Krook, r_nelement = $(this_ri.r.nelement_global)"
    makie_post_processing.animate_2d(this_ri.z.grid, this_ri.r.grid, perturbation,
                                     xlabel="z", ylabel="r", title=title,
                                     colormap="reverse_deep", outfile=outfile)

    # Plot final time point
    final_perturbation = @view perturbation[:,:,end]
    final_fig, final_ax, hm = heatmap(this_ri.z.grid, this_ri.r.grid, final_perturbation)
    Colorbar(final_fig[1,2], hm)
    final_ax.xlabel = "z"
    final_ax.ylabel = "r"
    final_ax.title = "no Krook, r_nelement = $(this_ri.r.nelement_global)"
    save(joinpath(dir_no_Krook,
                  "final_phi_perturbation_no-Krook_r-nelement$(this_ri.r.nelement_global).png"),
                  final_fig; px_per_unit=resolution_increase_factor)
end

Legend(fig[2,1], ax; tellheight=true, tellwidth=false)
resize_to_layout!(fig)
save(joinpath(dir_no_Krook, "growth_rate_no-Krook_r-resolution-scan.png"), fig; px_per_unit=resolution_increase_factor)

# Compare case in periodic box, with source-sustained 'background' perturbation having a
# z-variation with a wavenumber of 5.
dir_periodic5 = mkpath(joinpath(plots_dir, "instability_2D_periodic5"))
AT = 0.1
periodic5_run_dirs = (("runs/2D1V-instability-periodic5-test-AT0.1-An0.0/", 0.0),
                      ("runs/2D1V-instability-periodic5-test-AT0.1-An0.005/", 0.005),
                      ("runs/2D1V-instability-periodic5-test-AT0.1-An0.01/", 0.01),
                      ("runs/2D1V-instability-periodic5-test-AT0.1-An0.015/", 0.015),
                      ("runs/2D1V-instability-periodic5-test-AT0.1-An0.02/", 0.02),
                      ("runs/2D1V-instability-periodic5-test-AT0.1-An0.025/", 0.025),
                      ("runs/2D1V-instability-periodic5-test-AT0.1-An0.03/", 0.03),
                      ("runs/2D1V-instability-periodic5-test-AT0.1-An0.04/", 0.04),
                      ("runs/2D1V-instability-periodic5-test-AT0.1-An0.05/", 0.05),
                      ("runs/2D1V-instability-periodic5-test-AT0.1-An-0.005/", -0.005),
                      ("runs/2D1V-instability-periodic5-test-AT0.1-An-0.01/", -0.01),
                      ("runs/2D1V-instability-periodic5-test-AT0.1-An-0.02/", -0.02),
                      ("runs/2D1V-instability-periodic5-test-AT0.1-An-0.03/", -0.03),
                      ("runs/2D1V-instability-periodic5-test-AT0.1-An-0.04/", -0.04),
                      ("runs/2D1V-instability-periodic5-test-AT0.1-An-0.05/", -0.05),
                      ("runs/2D1V-instability-periodic5-test-AT0.1-An-0.1/", -0.1),
                     )
ri_periodic5 = Any[(get_run_info(d; dfns=true), An) for (d, An) ∈ periodic5_run_dirs]

fig_bg, ax_bg = get_1d_ax(xlabel="z", ylabel="T_∥")
Tpar = get_variable(ri_periodic5[1][1], "parallel_pressure"; it=1, is=1, ir=1) ./ get_variable(ri_periodic5[1][1], "density"; it=1, is=1, ir=1)
lines!(ax_bg, ri_periodic5[1][1].z.grid, Tpar)
save(joinpath(dir_periodic5, "periodic5_background_Tpar.png"), fig_bg; px_per_unit=resolution_increase_factor)

fig_bg, ax_bg = get_1d_ax(xlabel="z", ylabel="n")
# Ensure the first row width is 3/4 of the column width so that the plot does not get
# squashed by the legend
rowsize!(fig_bg.layout, 1, Aspect(1, 3/4))

fig, ax = get_1d_ax(xlabel="time", ylabel="amplitude", yscale=log10)
# Ensure the first row width is 3/4 of the column width so that the plot does not get
# squashed by the legend
rowsize!(fig.layout, 1, Aspect(1, 3/4))

An_list = Float64[]
gammas = Float64[]
for (irun, (this_ri, An)) ∈ enumerate(ri_periodic5)
    lines!(ax_bg, this_ri.z.grid, get_variable(this_ri, "density"; it=1, is=1, ir=1);
           label=this_ri.run_name)

    phi = get_variable(this_ri, "phi")

    γ = plot_mode_amplitude(this_ri, phi, ax, irun; label="An=$An")
    push!(An_list, An)
    push!(gammas, γ)

    # make animation of perturbation
    _, perturbation = makie_post_processing.get_r_perturbation(phi)
    outfile = joinpath(dir_periodic5, "phi_perturbation_periodic5_AT$(AT)_An$(An).gif")
    title = "periodic5, AT=$(AT) An=$(An)"
    makie_post_processing.animate_2d(this_ri.z.grid, this_ri.r.grid, perturbation,
                                     xlabel="z", ylabel="r", title=title,
                                     colormap="reverse_deep", outfile=outfile)

    # Plot final time point
    final_perturbation = @view perturbation[:,:,end]
    final_fig, final_ax, hm = heatmap(this_ri.z.grid, this_ri.r.grid, final_perturbation)
    Colorbar(final_fig[1,2], hm)
    final_ax.xlabel = "z"
    final_ax.ylabel = "r"
    final_ax.title = "periodic5, AT=$(AT) An=$(An)"
    save(joinpath(dir_periodic5,
                  "final_phi_perturbation_periodic5_AT$(AT)_An$(An).png"),
                  final_fig; px_per_unit=resolution_increase_factor)
end

Legend(fig_bg[2,1], ax_bg; tellheight=true, tellwidth=false)
resize_to_layout!(fig_bg)
save(joinpath(dir_periodic5, "periodic5_background_n.png"), fig_bg; px_per_unit=resolution_increase_factor)

xlims!(ax, 0.0, 20.0)
Legend(fig[2,1], ax; tellheight=true, tellwidth=false)
resize_to_layout!(fig)
save(joinpath(dir_periodic5, "growth_rate_periodic5-AT$(AT)-An-scan.png"), fig; px_per_unit=resolution_increase_factor)

max_positive_gamma_ind = 6
fig, ax, s = scatter(An_list[1:max_positive_gamma_ind] ./ AT, gammas[1:max_positive_gamma_ind])
first_negative_An_ind = 10
scatter!(ax, An_list[first_negative_An_ind:end] ./ AT, gammas[first_negative_An_ind:end])
ax.xlabel = "An/AT = LT/Ln"
ax.ylabel = "γ"

# Also plot a linear fit to the scan.
linear_fit(x, p) = @. p[1] * x + p[2]
x = An_list[1:max_positive_gamma_ind] ./ AT
y = gammas[1:max_positive_gamma_ind]
fit = curve_fit(linear_fit, x, y, [0.0, 0.0])
p = coef(fit)
x_fit = collect(extrema(x))
x_mid = 0.5 * (x_fit[1] + x_fit[2])
lines!(x_fit, linear_fit(x_fit, p); color=:grey, linestyle=:dash)
text!(ax, Point2f(x_mid, linear_fit(x_mid, p)); text="γ = $(round(p[1]; sigdigits=5)) (An/AT - $(round(-p[2]/p[1]; sigdigits=5)))",
      align=(:right, :top), color=:grey)

save(joinpath(dir_periodic5, "growth_rate_vs_LToverLn.png"), fig; px_per_unit=resolution_increase_factor)

# Compare case in periodic box, with source-sustained 'background' perturbation having a
# z-variation with a wavenumber of 5, using Fourier discretization for r.
dir_periodic5_Fourier_r = mkpath(joinpath(plots_dir, "instability_2D_periodic5_Fourier-r"))
AT = 0.1
periodic5_run_dirs = (("runs/2D1V-instability-periodic5-test-Fourier-r-AT0.1-An0.0/", 0.0),
                      ("runs/2D1V-instability-periodic5-test-Fourier-r-AT0.1-An0.005/", 0.005),
                      ("runs/2D1V-instability-periodic5-test-Fourier-r-AT0.1-An0.01/", 0.01),
                      ("runs/2D1V-instability-periodic5-test-Fourier-r-AT0.1-An0.015/", 0.015),
                      ("runs/2D1V-instability-periodic5-test-Fourier-r-AT0.1-An0.02/", 0.02),
                      ("runs/2D1V-instability-periodic5-test-Fourier-r-AT0.1-An0.025/", 0.025),
                      ("runs/2D1V-instability-periodic5-test-Fourier-r-AT0.1-An0.03/", 0.03),
                      ("runs/2D1V-instability-periodic5-test-Fourier-r-AT0.1-An0.04/", 0.04),
                      ("runs/2D1V-instability-periodic5-test-Fourier-r-AT0.1-An0.05/", 0.05),
                      ("runs/2D1V-instability-periodic5-test-Fourier-r-AT0.1-An-0.005/", -0.005),
                      ("runs/2D1V-instability-periodic5-test-Fourier-r-AT0.1-An-0.01/", -0.01),
                      ("runs/2D1V-instability-periodic5-test-Fourier-r-AT0.1-An-0.02/", -0.02),
                      ("runs/2D1V-instability-periodic5-test-Fourier-r-AT0.1-An-0.03/", -0.03),
                      ("runs/2D1V-instability-periodic5-test-Fourier-r-AT0.1-An-0.04/", -0.04),
                      ("runs/2D1V-instability-periodic5-test-Fourier-r-AT0.1-An-0.05/", -0.05),
                      ("runs/2D1V-instability-periodic5-test-Fourier-r-AT0.1-An-0.1/", -0.1),
                     )
ri_periodic5 = Any[(get_run_info(d; dfns=true), An) for (d, An) ∈ periodic5_run_dirs]

fig_bg, ax_bg = get_1d_ax(xlabel="z", ylabel="T_∥")
Tpar = get_variable(ri_periodic5[1][1], "parallel_pressure"; it=1, is=1, ir=1) ./ get_variable(ri_periodic5[1][1], "density"; it=1, is=1, ir=1)
lines!(ax_bg, ri_periodic5[1][1].z.grid, Tpar)
save(joinpath(dir_periodic5_Fourier_r, "periodic5_Fourier-r_background_Tpar.png"), fig_bg; px_per_unit=resolution_increase_factor)

fig_bg, ax_bg = get_1d_ax(xlabel="z", ylabel="n")
# Ensure the first row width is 3/4 of the column width so that the plot does not get
# squashed by the legend
rowsize!(fig_bg.layout, 1, Aspect(1, 3/4))

fig, ax = get_1d_ax(xlabel="time", ylabel="amplitude", yscale=log10)
# Ensure the first row width is 3/4 of the column width so that the plot does not get
# squashed by the legend
rowsize!(fig.layout, 1, Aspect(1, 3/4))

An_list = Float64[]
gammas = Float64[]
for (irun, (this_ri, An)) ∈ enumerate(ri_periodic5)
    lines!(ax_bg, this_ri.z.grid, get_variable(this_ri, "density"; it=1, is=1, ir=1);
           label=this_ri.run_name)

    phi = get_variable(this_ri, "phi")

    γ = plot_mode_amplitude(this_ri, phi, ax, irun; label="An=$An")
    push!(An_list, An)
    push!(gammas, γ)

    # make animation of perturbation
    _, perturbation = makie_post_processing.get_r_perturbation(phi)
    outfile = joinpath(dir_periodic5_Fourier_r, "phi_perturbation_Fourier-r_periodic5_AT$(AT)_An$(An).gif")
    title = "periodic5, AT=$(AT) An=$(An)"
    makie_post_processing.animate_2d(this_ri.z.grid, this_ri.r.grid, perturbation,
                                     xlabel="z", ylabel="r", title=title,
                                     colormap="reverse_deep", outfile=outfile)

    # Plot final time point
    final_perturbation = @view perturbation[:,:,end]
    final_fig, final_ax, hm = heatmap(this_ri.z.grid, this_ri.r.grid, final_perturbation)
    Colorbar(final_fig[1,2], hm)
    final_ax.xlabel = "z"
    final_ax.ylabel = "r"
    final_ax.title = "periodic5, AT=$(AT) An=$(An)"
    save(joinpath(dir_periodic5_Fourier_r,
                  "final_phi_perturbation_periodic5_Fourier-r_AT$(AT)_An$(An).png"),
                  final_fig; px_per_unit=resolution_increase_factor)
end

Legend(fig_bg[2,1], ax_bg; tellheight=true, tellwidth=false)
resize_to_layout!(fig_bg)
save(joinpath(dir_periodic5_Fourier_r, "periodic5_Fourier-r_background_n.png"), fig_bg; px_per_unit=resolution_increase_factor)

xlims!(ax, 0.0, 20.0)
Legend(fig[2,1], ax; tellheight=true, tellwidth=false)
resize_to_layout!(fig)
save(joinpath(dir_periodic5_Fourier_r, "growth_rate_periodic5_Fourier-r-AT$(AT)-An-scan.png"), fig; px_per_unit=resolution_increase_factor)

max_positive_gamma_ind = 6
fig, ax, s = scatter(An_list[1:max_positive_gamma_ind] ./ AT, gammas[1:max_positive_gamma_ind])
first_negative_An_ind = 10
scatter!(ax, An_list[first_negative_An_ind:end] ./ AT, gammas[first_negative_An_ind:end])
ax.xlabel = "An/AT = LT/Ln"
ax.ylabel = "γ"

# Also plot a linear fit to the scan.
linear_fit(x, p) = @. p[1] * x + p[2]
x = An_list[1:max_positive_gamma_ind] ./ AT
y = gammas[1:max_positive_gamma_ind]
fit = curve_fit(linear_fit, x, y, [0.0, 0.0])
p = coef(fit)
x_fit = collect(extrema(x))
x_mid = 0.5 * (x_fit[1] + x_fit[2])
lines!(x_fit, linear_fit(x_fit, p); color=:grey, linestyle=:dash)
text!(ax, Point2f(x_mid, linear_fit(x_mid, p)); text="γ = $(round(p[1]; sigdigits=5)) (An/AT - $(round(-p[2]/p[1]; sigdigits=5)))",
      align=(:right, :top), color=:grey)

save(joinpath(dir_periodic5_Fourier_r, "Fourier-r_growth_rate_vs_LToverLn.png"), fig; px_per_unit=resolution_increase_factor)
