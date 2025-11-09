using CairoMakie
using makie_post_processing.LaTeXStrings
using MathTeXEngine
using makie_post_processing.NaNMath
using LsqFit
using makie_post_processing
using makie_post_processing.Printf
using StatsBase

# Increase this well above 1.0 to get poster-quality plots, but because we are outputting
# as png here, also increases file size quite a bit.
#CairoMakie.activate!(; px_per_unit=1.0)
CairoMakie.activate!(; px_per_unit=16.0)

function get_mantissa_exponent(x)
    # Hacky and may sometimes be inaccurate, so check the output!
    if x == 0.0
        return 0.0, 0.0
    else
        exponent = floor(Int64, log10(abs(x)))
        mantissa = x / 10.0^exponent
        return mantissa, exponent
    end
end

function latex_single_tick_format(x)
    println("tick formatter: ", x, " ", typeof(x))
    if abs(x) != 0.0 && (abs(x) < 1.0e-4 || abs(x) > 1.0e4)
        rounded = round(x; sigdigits=3)
        exponent = floor(Int64, log10(abs(rounded)))
        mantissa = @sprintf("%.1f", rounded / 10.0^exponent)
        println("converting ", x, " ", "\$$mantissa \\times 10^{$exponent}\$")
        #return LaTeXString("\$$mantissa \\times 10^$exponent\$")
        # Don't understand why using \times doesn't work at the moment (14/11/2025), but
        # using the unicode symbol for now as a workaround
        return LaTeXString("\$$mantissa × 10^{$exponent}\$")
    else
        return L"%$(x)"
    end
end
function latex_tick_formatter(x)
    return latex_single_tick_format.(x)
end
# Not sure how to pass latex_tick_formatter() through update_theme!() - it does seem to
# work when passed as an argument, e.g. to `Colorbar()` below.
update_theme!(fontsize=20, fonts=(; font="CMU", regular=texfont(:text),
                                  bold=texfont(:bold), italic=texfont(:italic),
                                  bold_italic=texfont(:bolditalic)))


plots_dir = "publication_inputs/APS2025"

# 1D backgrounds
for (sim_dir, label) ∈ (("runs/1D1V-instability-test_Lvpa24/", "background_1D"),
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

    save(joinpath(dir_1d, label * "_source.png"), fig_1d)

    reference_parameters = makie_post_processing.moment_kinetics.reference_parameters.setup_reference_parameters(ri_1d.input, true)
    Nref = reference_parameters.Nref
    Tref = reference_parameters.Tref
    Lref = reference_parameters.Lref
    cref = reference_parameters.cref
    omegaref = cref / Lref
    Dref = cref * Lref

    z = ri_1d.z.grid .* Lref

    # Plot moment profiles
    n = get_variable(ri_1d, "density"; it=ri_1d.nt, is=1, ir=1)
    T = get_variable(ri_1d, "temperature"; it=ri_1d.nt, is=1, ir=1)

    println("n = ", n)
    println("T = ", T)

    fig_1d = Figure()
    #with_theme(Axis=(; cycle=:yticklabelcolor=>:color)) do
    #    ax_1d_n = Axis(fig_1d[1,1]; yticklabelcolor=Cycled(1))
    #end
    ax_1d_n = Axis(fig_1d[1,1]; ylabelcolor=:blue, yticklabelcolor=:blue,
                   leftspinecolor=:blue, rightspinecolor=:red)
    lines!(ax_1d_n, z, n .* Nref ./ 1e19; label="n", color=:blue)
    if label == "background_1D"
        ylims!(ax_1d_n, 0.0, 2.2)
    else
        ylims!(ax_1d_n, 0.0, 4.4)
    end
    ax_1d_n.xlabel = L"$z$ (m)"
    ax_1d_n.ylabel = L"$n$ ($10^{19}$ m$^{-3}$)"

    #with_theme(Axis=(; cycle=:yticklabelcolor=>:color)) do
    #    ax_1d_T = Axis(fig_1d[1,1]; yticklabelcolor=Cycled(2), yaxisposition=:right)
    #end
    ax_1d_T = Axis(fig_1d[1,1]; ylabelcolor=:red, yticklabelcolor=:red,
                   yaxisposition=:right)
    ylims!(ax_1d_T, 0.0, 22.0)
    hidespines!(ax_1d_T)
    hidexdecorations!(ax_1d_T)
    lines!(ax_1d_T, z, T .* Tref; label="T", color=:red)
    ax_1d_T.ylabel = L"$T$ (eV)"

    #Legend(fig_1d[2,1], ax_1d; tellheight=true, tellwidth=false)

    save(joinpath(dir_1d, label * "_nT_profiles.png"), fig_1d)

    u = get_variable(ri_1d, "parallel_flow"; it=ri_1d.nt, is=1, ir=1)

    fig_1d, ax_1d, l = lines(z, u .* cref ./ 1e4; label="u", color=:black)
    ax_1d.xlabel = L"$z$ (m)"
    ax_1d.ylabel = L"$u_∥$ ($10^4$ m.s$^-1$)"
    #Legend(fig_1d[2,1], ax_1d; tellheight=true, tellwidth=false)

    save(joinpath(dir_1d, label * "_upar_profiles.png"), fig_1d)

    if label == "background_1D"
        nu_ii_Krook = get_variable(ri_1d, "Krook_collision_frequency_ii"; it=ri_1d.nt,
                                   is=1, ir=1)

        fig_1d, ax_1d, l = lines(z, nu_ii_Krook .* omegaref ./ 1e4; label="nu_ii_Krook")
        ax_1d.xlabel = L"$z$ (m)"
        ax_1d.ylabel = L"$ν_{ii}$ ($10^4$ s$^{-1}$)"
        Legend(fig_1d[2,1], ax_1d; tellheight=true, tellwidth=false)

        save(joinpath(dir_1d, label * "_nu_ii_Krook_profiles.png"), fig_1d)
    end

    # Plot distribution function
    f = get_variable(ri_1d, "f"; it=ri_1d.nt, is=1, ir=1, ivperp=1)

    fig_1d, ax_1d, hm = heatmap(ri_1d.vpa.grid, z, f)
    Colorbar(fig_1d[1,2], hm)
    ax_1d.xlabel = L"$v_∥ / c_{ref}$"
    ax_1d.ylabel = L"$z$ (m)"

    save(joinpath(dir_1d, label * "_f.png"), fig_1d)

    fig_1d = Figure()
    ax_1d = Axis(fig_1d[1,1])
    ax_1d.xlabel = L"$v_∥ / c_{ref}$"
    ax_1d.ylabel = L"$f$ (arb.)"
    rowsize!(fig_1d.layout, 1, Aspect(1, 3/4))
    zinds = (1, 11, 23, 33)
    for iz ∈ zinds
        lines!(ax_1d, ri_1d.vpa.grid, f[:,iz], label=L"$z = %$(z[iz])$ m")
    end
    Legend(fig_1d[2,1], ax_1d; tellheight=true, tellwidth=false)
    resize_to_layout!(fig_1d)
    save(joinpath(dir_1d, label * "_f_vs_vpa.png"), fig_1d)

    fig_1d = Figure()
    ax_1d = Axis(fig_1d[1,1]; yscale=log10)
    ax_1d.xlabel = L"$v_∥ / c_{ref}$"
    ax_1d.ylabel = L"$f$ (arb.)"
    rowsize!(fig_1d.layout, 1, Aspect(1, 3/4))
    zinds = (1, 11, 23, 33)
    for iz ∈ zinds
        filtered_f = @view f[:,iz]
        filtered_f[filtered_f .< 1.0e-16] .= NaN
        lines!(ax_1d, ri_1d.vpa.grid, f[:,iz], label=L"$z = %$(z[iz])$ m")
    end
    Legend(fig_1d[2,1], ax_1d; tellheight=true, tellwidth=false)
    resize_to_layout!(fig_1d)
    save(joinpath(dir_1d, label * "_logf_vs_vpa.png"), fig_1d)

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

    fig_1d, ax_1d, l = lines(z, D_classical .* Dref ./ 1e4)
    ax_1d.xlabel = L"$z$ (m)"
    ax_1d.ylabel = L"$D_{classical}$ ($10^4$ m$^2$.$s^{-1}$)"

    save(joinpath(dir_1d, label * "_D_classical.png"), fig_1d)

    fig_1d, ax_1d, l = lines(z, chi_i_classical .* Dref ./ 1e4)
    ax_1d.xlabel = L"$z$ (m)"
    ax_1d.ylabel = L"$χ_{i,classical}$ ($10^4$ m$^2$.$s^{-1}$)"

    save(joinpath(dir_1d, label * "_chi_i_classical.png"), fig_1d)

    fig_1d, ax_1d, l = lines(z, rho_i .* Lref ./ 1e4)
    ax_1d.xlabel = L"$z$ (m)"
    ax_1d.ylabel = L"$ρ_i$ ($10^-4$ m)"

    save(joinpath(dir_1d, label * "_rho_i.png"), fig_1d)
end

# Instability analysis - scan in radial resolution
dir_r_nelement_scan = mkpath(joinpath(plots_dir, "instability_2D_r-resolution-scan"))
r_r_nelement_scan_run_dirs = ("runs/2D1V-instability-test_Lvpa24_Lr1cm/",
                            "runs/2D1V-instability-test_Lvpa24_Lr1cm-rnelement8/",
                            "runs/2D1V-instability-test_Lvpa24_Lr1cm-rnelement16/",
                            "runs/2D1V-instability-test_Lvpa24_Lr1cm-rnelement32/",
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

    # Get time index from the middle of the fitted part of the mode growth.
    tmid = 0.5 * (tmin + tmax)
    mid_ind = searchsortedfirst(this_ri.time, tmid)

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

    return γ, mid_ind
end

function get_perturbation(x; relative=false)
    if relative
        return x ./ mean(x) .- 1
    else
        return x .- mean(x)
    end
end

function get_perturbation_f(f; relative=false)
    if relative
        return f ./ mean(f; dims=2) .- 1
    else
        return f .- mean(f; dims=2)
    end
end

function plot_radial_profiles(this_ri, phi, irun, tind;
                              label="r_nelement = $(this_ri.r.nelement_global)",
                              prefix)
    reference_parameters = makie_post_processing.moment_kinetics.reference_parameters.setup_reference_parameters(this_ri.input, true)
    Lref = reference_parameters.Lref
    Nref = reference_parameters.Nref
    cref = reference_parameters.cref
    Tref = reference_parameters.Tref

    f = get_variable(this_ri, "f"; it=tind, is=1, ivperp=1)
    n = get_variable(this_ri, "density"; it=tind, is=1) .* Nref ./ 1e19
    upar = get_variable(this_ri, "parallel_flow"; it=tind, is=1) .* cref
    T = get_variable(this_ri, "temperature"; it=tind, is=1) .* Tref
    phi_Volts = @views phi[:,:,tind] .* Tref

    r = this_ri.r.grid .* Lref
    z = this_ri.z.grid .* Lref

    phiscale_pow = -5
    phiscale = 10.0^phiscale_pow
    nscale_pow = -5
    nscale = 10.0^nscale_pow
    uparscale_pow = -1
    uparscale = 10.0^uparscale_pow
    Tscale_pow = -5
    Tscale = 10.0^Tscale_pow

    #radial_phi_fig, radial_phi_ax = get_1d_ax(; xlabel=L"r (m)", ylabel=L"$δϕ$ ($10^{%$(phiscale_pow)}$ V)", subtitles=(label,))
    #radial_n_fig, radial_n_ax = get_1d_ax(; xlabel=L"r (m)", ylabel=L"$δn$ ($10^{%$(19-nscale_pow)}$ m$^-3$)", subtitles=(label,))
    #radial_T_fig, radial_T_ax = get_1d_ax(; xlabel=L"r (m)", ylabel=L"$δT$ ($10^{%$(Tscale_pow)}$ eV)", subtitles=(label,))
    radial_phi_fig = Figure()
    radial_phi_ax = Axis(radial_phi_fig[1,1])
    radial_phi_ax.xlabel = L"$r$ (m)"
    radial_phi_ax.ylabel = L"$δϕ$ ($10^{%$(phiscale_pow)}$ V)"
    radial_phi_ax.title = label
    rowsize!(radial_phi_fig.layout, 1, Aspect(1, 1))

    radial_n_fig = Figure()
    radial_n_ax = Axis(radial_n_fig[1,1])
    radial_n_ax.xlabel = L"$r$ (m)"
    radial_n_ax.ylabel = L"$δn$ ($10^{%$(19+nscale_pow)}$ m$^{-3}$)"
    #radial_n_ax.ylabel = L"$δn/n_0$ ($×10^{%$(nscale_pow)}$)"
    radial_n_ax.title = label
    rowsize!(radial_n_fig.layout, 1, Aspect(1, 1))

    radial_upar_fig = Figure()
    radial_upar_ax = Axis(radial_upar_fig[1,1])
    radial_upar_ax.xlabel = L"$r$ (m)"
    radial_upar_ax.ylabel = L"$δu_∥$ ($10^{%$(uparscale_pow)}$ m.s$^{-1}$)"
    #radial_upar_ax.ylabel = L"$δu_∥/u_{∥,0}$ ($×10^{%$(uparscale_pow)}$)"
    radial_upar_ax.title = label
    rowsize!(radial_upar_fig.layout, 1, Aspect(1, 1))

    radial_T_fig = Figure()
    radial_T_ax = Axis(radial_T_fig[1,1])
    radial_T_ax.xlabel = L"$r$ (m)"
    radial_T_ax.ylabel = L"$δT$ ($10^{%$(Tscale_pow)}$ eV)"
    #radial_T_ax.ylabel = L"$δT/T_0$ ($×10^{%$(Tscale_pow)}$)"
    radial_T_ax.title = label
    rowsize!(radial_T_fig.layout, 1, Aspect(1, 1))

    zinds = (1, 11, 23, 33)
    for iz ∈ zinds
        delta_phi = get_perturbation(phi_Volts[iz, :]) ./ phiscale
        lines!(radial_phi_ax, r, delta_phi; label=L"$z$ = $%$(z[iz])$ m")

        delta_n = get_perturbation(n[iz, :]) ./ nscale
        #delta_n = get_perturbation(n[iz, :], relative=true) ./ nscale
        lines!(radial_n_ax, r, delta_n; label=L"$z$ = $%$(z[iz])$ m")

        delta_upar = get_perturbation(upar[iz, :]) ./ uparscale
        #delta_upar = get_perturbation(upar[iz, :], relative=true) ./ uparscale
        lines!(radial_upar_ax, r, delta_upar; label=L"$z$ = $%$(z[iz])$ m")

        delta_T = get_perturbation(T[iz, :]) ./ Tscale
        #delta_T = get_perturbation(T[iz, :], relative=true) ./ Tscale
        lines!(radial_T_ax, r, delta_T; label=L"$z$ = $%$(z[iz])$ m")
    end
    Legend(radial_phi_fig[2,1], radial_phi_ax; tellheight=true, tellwidth=false)
    Legend(radial_n_fig[2,1], radial_n_ax; tellheight=true, tellwidth=false)
    Legend(radial_upar_fig[2,1], radial_upar_ax; tellheight=true, tellwidth=false)
    Legend(radial_T_fig[2,1], radial_T_ax; tellheight=true, tellwidth=false)
    resize_to_layout!(radial_phi_fig)
    resize_to_layout!(radial_n_fig)
    resize_to_layout!(radial_upar_fig)
    resize_to_layout!(radial_T_fig)
    save(prefix * "_phi_vs_r.png", radial_phi_fig)
    save(prefix * "_n_vs_r.png", radial_n_fig)
    save(prefix * "_upar_vs_r.png", radial_upar_fig)
    save(prefix * "_T_vs_r.png", radial_T_fig)

    for iz ∈ zinds
        radial_f_fig = Figure()
        radial_f_ax = Axis(radial_f_fig[1,1])
        radial_f_ax.xlabel = L"$v_∥ / c_{ref}$"
        radial_f_ax.ylabel = L"$r$ (m)"
        zstr = @sprintf("%.3f", z[iz])
        radial_f_ax.title = latexstring(label, L" $z=%$(zstr)$ m")

        delta_f = get_perturbation_f(f[:, iz, :])
        maxabs = maximum(abs.(delta_f))
        hm = heatmap!(this_ri.vpa.grid, r, delta_f; label=L"$z$ = $%$(z[iz])$ m",
                      colormap=:balance, colorrange=(-maxabs, maxabs))
        Colorbar(radial_f_fig[1,2], hm; label=L"$δf$ (arb.)")

        save(prefix * "_f_vs_vpa_r_iz$iz.png", radial_f_fig)
    end

    return nothing
end

fig, ax = get_1d_ax(xlabel="time", ylabel="amplitude", yscale=log10)
# Ensure the first row width is 3/4 of the column width so that the plot does not get
# squashed by the legend
rowsize!(fig.layout, 1, Aspect(1, 3/4))

for (irun, this_ri) ∈ enumerate(ri_r_nelement_scan)
    phi = get_variable(this_ri, "phi")

    _, mid_ind = plot_mode_amplitude(this_ri, phi, ax, irun)

    # make animation of perturbation
    _, perturbation = makie_post_processing.get_r_perturbation(phi)
    outfile = joinpath(dir_r_nelement_scan, "phi_perturbation_r-nelement$(this_ri.r.nelement_global).gif")
    title = "r_nelement = $(this_ri.r.nelement_global)"
    makie_post_processing.animate_2d(this_ri.z.grid, this_ri.r.grid, perturbation,
                                     xlabel="z", ylabel="r", title=title,
                                     colormap="reverse_deep", outfile=outfile)

    # Plot final time point
    final_perturbation = @view perturbation[:,:,mid_ind]
    final_fig, final_ax, hm = heatmap(this_ri.z.grid, this_ri.r.grid, final_perturbation)
    Colorbar(final_fig[1,2], hm)
    final_ax.xlabel = "z"
    final_ax.ylabel = "r"
    final_ax.title = "r_nelement = $(this_ri.r.nelement_global)"
    save(joinpath(dir_r_nelement_scan,
                  "final_phi_perturbation_r-nelement$(this_ri.r.nelement_global).png"),
                  final_fig)
end

Legend(fig[2,1], ax; tellheight=true, tellwidth=false)
resize_to_layout!(fig)
save(joinpath(dir_r_nelement_scan, "growth_rate_r-resolution-scan.png"), fig)

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
save(joinpath(dir_r_nelement_scan, "stabilised-resolution-scan.png"), fig)

# Show apparently converged mode with Dr=7e-9
converged_rdiss7em9_run_dirs = ("runs/2D1V-instability-test_Lvpa24_Lr1cm_rdiss7e-9/",
                                "runs/2D1V-instability-test_Lvpa24_Lr1cm-rnelement8-rdiss7e-9/",
                                "runs/2D1V-instability-test_Lvpa24_Lr1cm-rnelement16-rdiss7e-9/",
                                #"runs/2D1V-instability-test_Lvpa24_Lr1cm-rnelement32-rdiss7e-9/",
                               )
ri_converged = get_run_info(converged_rdiss7em9_run_dirs...; dfns=true)

fig, ax = get_1d_ax(xlabel="time", ylabel="amplitude", yscale=log10)
# Ensure the first row width is 3/4 of the column width so that the plot does not get
# squashed by the legend
rowsize!(fig.layout, 1, Aspect(1, 3/4))

for (irun, this_ri) ∈ enumerate(ri_converged)
    phi = get_variable(this_ri, "phi")

    _, mid_ind = plot_mode_amplitude(this_ri, phi, ax, irun)

    # make animation of perturbation
    _, perturbation = makie_post_processing.get_r_perturbation(phi)
    outfile = joinpath(dir_r_nelement_scan, "phi_perturbation_rdiss7e-9_r-nelement$(this_ri.r.nelement_global).gif")
    title = "r_nelement = $(this_ri.r.nelement_global), Dr = 7e-9"
    makie_post_processing.animate_2d(this_ri.z.grid, this_ri.r.grid, perturbation,
                                     xlabel="z", ylabel="r", title=title,
                                     colormap="reverse_deep", outfile=outfile)

    # Plot final time point
    final_perturbation = @view perturbation[:,:,mid_ind]
    final_fig, final_ax, hm = heatmap(this_ri.z.grid, this_ri.r.grid, final_perturbation)
    Colorbar(final_fig[1,2], hm)
    final_ax.xlabel = "z"
    final_ax.ylabel = "r"
    final_ax.title = "r_nelement = $(this_ri.r.nelement_global), Dr = 7e-9"
    save(joinpath(dir_r_nelement_scan,
                  "final_phi_perturbation_rdiss7e-9_r-nelement$(this_ri.r.nelement_global).png"),
                  final_fig)
end

Legend(fig[2,1], ax; tellheight=true, tellwidth=false)
resize_to_layout!(fig)
save(joinpath(dir_r_nelement_scan, "converged-rdiss7e-9-resolution-scan.png"), fig)

# Scan dissipation between Dr=1e-9 (unsure if the instability is radial-grid-scale) and Dr=7e-9 (instability seems well resolved).
dir_rdiss_scan = mkpath(joinpath(plots_dir, "instability_2D_rdiss-scan"))
rdiss_scan_run_dirs = (("runs/2D1V-instability-test_Lvpa24_Lr1cm-rnelement16/", 0),
                       ("runs/2D1V-instability-test_Lvpa24_Lr1cm-rnelement16-rdiss1e-9/", 1),
                       ("runs/2D1V-instability-test_Lvpa24_Lr1cm-rnelement16-rdiss2e-9/", 2),
                       ("runs/2D1V-instability-test_Lvpa24_Lr1cm-rnelement16-rdiss3e-9/", 3),
                       ("runs/2D1V-instability-test_Lvpa24_Lr1cm-rnelement16-rdiss4e-9/", 4),
                       ("runs/2D1V-instability-test_Lvpa24_Lr1cm-rnelement16-rdiss5e-9/", 5),
                       ("runs/2D1V-instability-test_Lvpa24_Lr1cm-rnelement16-rdiss6e-9/", 6),
                       ("runs/2D1V-instability-test_Lvpa24_Lr1cm-rnelement16-rdiss7e-9/", 7),
                      )
ri_rdiss_scan = Any[(get_run_info(d; dfns=true), rdiss_val)
                    for (d, rdiss_val) ∈ rdiss_scan_run_dirs]

fig, ax = get_1d_ax(xlabel="time", ylabel="amplitude", yscale=log10)
# Ensure the first row width is 3/4 of the column width so that the plot does not get
# squashed by the legend
rowsize!(fig.layout, 1, Aspect(1, 3/4))

for (this_ri, rdiss_val) ∈ ri_rdiss_scan
    phi = get_variable(this_ri, "phi")

    γ, mid_ind = plot_mode_amplitude(this_ri, phi, ax, rdiss_val + 1; label="rdiss = $(rdiss_val)e-9")

    # make animation of perturbation
    _, perturbation = makie_post_processing.get_r_perturbation(phi)
    outfile = joinpath(dir_rdiss_scan, "phi_perturbation_rdiss$(rdiss_val)e-9.gif")
    title = "r_nelement = 16, Dr = $(rdiss_val)e-9"
    #makie_post_processing.animate_2d(this_ri.z.grid, this_ri.r.grid, perturbation,
    #                                 xlabel="z", ylabel="r", title=title,
    #                                 colormap="reverse_deep", outfile=outfile)

    reference_parameters = makie_post_processing.moment_kinetics.reference_parameters.setup_reference_parameters(this_ri.input, true)
    Lref = reference_parameters.Lref
    Tref = reference_parameters.Tref
    cref = reference_parameters.cref
    Dref = cref * Lref
    omegaref = cref / Lref
    println("omegaref = ", omegaref)
    z = @. this_ri.z.grid * Lref
    r = this_ri.r.grid .* Lref
    perturbation_Volts = perturbation .* Tref
    Dr = this_ri.num_diss_params.ion.r_dissipation_coefficient * Dref
    gamma_per_second = γ * omegaref

    if Dr == 0.0
        Dr_string = "0"
    else
        mantissa, exponent = get_mantissa_exponent(Dr)
        Dr_m = @sprintf("%.2f", mantissa)
        Dr_string = "$Dr_m × 10^{$(exponent)} \$ m\$^2 \$ s\$^{-1}"
    end
    if gamma_per_second == 0.0
        gamma_string = "0"
    else
        mantissa, exponent = get_mantissa_exponent(gamma_per_second)
        gamma_m = @sprintf("%.2f", mantissa)
        gamma_string = "$gamma_m × 10^{$(exponent)} \$ s\$^{-1}"
    end

    # Plot time point in middle of linear growth phase
    mid_perturbation = @view perturbation[:,:,mid_ind]
    max_pert = maximum(abs.(mid_perturbation))
    _, exponent = get_mantissa_exponent(max_pert)

    #mid_fig, mid_ax, hm = heatmap(z, r, mid_perturbation ./ (10.0^exponent))
    #mid_ax.xlabel = L"$z$ (m)"
    #mid_ax.ylabel = L"$r$ (m)"
    mid_fig, mid_ax, hm = heatmap(r, z, transpose(mid_perturbation) ./ (10.0^exponent))
    mid_ax.ylabel = L"$z$ (m)"
    mid_ax.xlabel = L"$r$ (m)"

    Colorbar(mid_fig[1,2], hm; label=L"$δϕ$ ($10^{%$exponent}$ V)")
    #mid_ax.title = "r_nelement = 16, Dr = $(rdiss_val)e-9"
    mid_ax.title = L"$D_r = %$(Dr_string)$; $γ = %$(gamma_string)$"
    save(joinpath(dir_rdiss_scan,
                  "mid_phi_perturbation_rdiss$(rdiss_val)e-9.png"),
                  mid_fig)

    # Plot some radial profiles
    plot_radial_profiles(this_ri, phi, rdiss_val + 1, mid_ind;
                         label=L"$D_r = %$(Dr_string)$; $γ = %$(gamma_string)$",
                         prefix=joinpath(dir_rdiss_scan, "mid_r-profiles_rdiss$(rdiss_val)e-9"))

    # Plot final time point
    final_perturbation = @view perturbation[:,:,end]
    max_pert = NaNMath.maximum(abs.(final_perturbation))
    _, exponent = get_mantissa_exponent(max_pert)

    #final_fig, final_ax, hm = heatmap(z, r, final_perturbation ./ (10.0^exponent))
    #final_ax.xlabel = L"$z$ (m)"
    #final_ax.ylabel = L"$r$ (m)"
    final_fig, final_ax, hm = heatmap(r, z, transpose(final_perturbation) ./ (10.0^exponent))
    final_ax.ylabel = L"$z$ (m)"
    final_ax.xlabel = L"$r$ (m)"

    Colorbar(final_fig[1,2], hm; label=L"$δϕ$ ($10^{%$exponent}$ V)")
    #final_ax.title = "r_nelement = 16, Dr = $(rdiss_val)e-9"
    final_ax.title = L"$D_r = %$(Dr_string)$; $γ = %$(gamma_string)$"
    save(joinpath(dir_rdiss_scan,
                  "final_phi_perturbation_rdiss$(rdiss_val)e-9.png"),
                  final_fig)
end

Legend(fig[2,1], ax; tellheight=true, tellwidth=false)
resize_to_layout!(fig)
save(joinpath(dir_rdiss_scan, "rdiss-scan.png"), fig)

# High resolution version of scan of dissipation between Dr=1e-9 (unsure if the instability is radial-grid-scale) and Dr=7e-9 (instability seems well resolved).
dir_rdiss_scan = mkpath(joinpath(plots_dir, "instability_2D_rdiss-scan"))
rdiss_scan_run_dirs = (#("runs/2D1V-instability-test_Lr1cm-rnelement32-znelement32/", 0),
                       ("runs/2D1V-instability-test_Lr1cm-rnelement32-znelement32-rdiss1e-9/", 1),
                       #("runs/2D1V-instability-test_Lr1cm-rnelement32-znelement32-rdiss2e-9/", 2),
                       #("runs/2D1V-instability-test_Lr1cm-rnelement32-znelement32-rdiss3e-9/", 3),
                       #("runs/2D1V-instability-test_Lr1cm-rnelement32-znelement32-rdiss4e-9/", 4),
                       #("runs/2D1V-instability-test_Lr1cm-rnelement32-znelement32-rdiss5e-9/", 5),
                       #("runs/2D1V-instability-test_Lr1cm-rnelement32-znelement32-rdiss6e-9/", 6),
                       #("runs/2D1V-instability-test_Lr1cm-rnelement32-znelement32-rdiss7e-9/", 7),
                      )
ri_rdiss_scan = Any[(get_run_info(d; dfns=true), rdiss_val)
                    for (d, rdiss_val) ∈ rdiss_scan_run_dirs]

fig, ax = get_1d_ax(xlabel="time", ylabel="amplitude", yscale=log10)
# Ensure the first row width is 3/4 of the column width so that the plot does not get
# squashed by the legend
rowsize!(fig.layout, 1, Aspect(1, 3/4))

for (this_ri, rdiss_val) ∈ ri_rdiss_scan
    phi = get_variable(this_ri, "phi")

    γ, mid_ind = plot_mode_amplitude(this_ri, phi, ax, rdiss_val + 1; label="rdiss = $(rdiss_val)e-9")

    # make animation of perturbation
    _, perturbation = makie_post_processing.get_r_perturbation(phi)
    outfile = joinpath(dir_rdiss_scan, "phi_perturbation_rdiss$(rdiss_val)e-9_highres.gif")
    title = "r_nelement = 16, Dr = $(rdiss_val)e-9"
    #makie_post_processing.animate_2d(this_ri.z.grid, this_ri.r.grid, perturbation,
    #                                 xlabel="z", ylabel="r", title=title,
    #                                 colormap="reverse_deep", outfile=outfile)

    reference_parameters = makie_post_processing.moment_kinetics.reference_parameters.setup_reference_parameters(this_ri.input, true)
    Lref = reference_parameters.Lref
    Tref = reference_parameters.Tref
    cref = reference_parameters.cref
    Dref = cref * Lref
    omegaref = cref / Lref
    println("omegaref = ", omegaref)
    z = @. this_ri.z.grid * Lref
    r = this_ri.r.grid .* Lref
    perturbation_Volts = perturbation .* Tref
    Dr = this_ri.num_diss_params.ion.r_dissipation_coefficient * Dref
    gamma_per_second = γ * omegaref

    if Dr == 0.0
        Dr_string = "0"
    else
        mantissa, exponent = get_mantissa_exponent(Dr)
        Dr_m = @sprintf("%.2f", mantissa)
        Dr_string = "$Dr_m × 10^{$(exponent)} \$ m\$^2 \$ s\$^{-1}"
    end
    if gamma_per_second == 0.0
        gamma_string = "0"
    else
        mantissa, exponent = get_mantissa_exponent(gamma_per_second)
        gamma_m = @sprintf("%.2f", mantissa)
        gamma_string = "$gamma_m × 10^{$(exponent)} \$ s\$^{-1}"
    end

    # Plot time point in middle of linear growth phase
    mid_perturbation = @view perturbation[:,:,mid_ind]
    max_pert = maximum(abs.(mid_perturbation))
    _, exponent = get_mantissa_exponent(max_pert)

    #mid_fig, mid_ax, hm = heatmap(z, r, mid_perturbation ./ (10.0^exponent))
    #mid_ax.xlabel = L"$z$ (m)"
    #mid_ax.ylabel = L"$r$ (m)"
    mid_fig, mid_ax, hm = heatmap(r, z, transpose(mid_perturbation) ./ (10.0^exponent))
    mid_ax.ylabel = L"$z$ (m)"
    mid_ax.xlabel = L"$r$ (m)"

    Colorbar(mid_fig[1,2], hm; label=L"$δϕ$ ($10^{%$exponent}$ V)")
    #mid_ax.title = "r_nelement = 16, Dr = $(rdiss_val)e-9"
    mid_ax.title = L"$D_r = %$(Dr_string)$; $γ = %$(gamma_string)$"
    save(joinpath(dir_rdiss_scan,
                  "mid_phi_perturbation_rdiss$(rdiss_val)e-9_highres.png"),
                  mid_fig)

    # Plot some radial profiles
    plot_radial_profiles(this_ri, phi, rdiss_val + 1, mid_ind;
                         label=L"$D_r = %$(Dr_string)$; $γ = %$(gamma_string)$",
                         prefix=joinpath(dir_rdiss_scan, "mid_r-profiles_rdiss$(rdiss_val)e-9_highres"))

    # Plot final time point
    final_perturbation = @view perturbation[:,:,end]
    max_pert = NaNMath.maximum(abs.(final_perturbation))
    _, exponent = get_mantissa_exponent(max_pert)

    #final_fig, final_ax, hm = heatmap(z, r, final_perturbation ./ (10.0^exponent))
    #final_ax.xlabel = L"$z$ (m)"
    #final_ax.ylabel = L"$r$ (m)"
    final_fig, final_ax, hm = heatmap(r, z, transpose(final_perturbation) ./ (10.0^exponent))
    final_ax.ylabel = L"$z$ (m)"
    final_ax.xlabel = L"$r$ (m)"

    Colorbar(final_fig[1,2], hm; label=L"$δϕ$ ($10^{%$exponent}$ V)")
    #final_ax.title = "r_nelement = 16, Dr = $(rdiss_val)e-9"
    final_ax.title = L"$D_r = %$(Dr_string)$; $γ = %$(gamma_string)$"
    save(joinpath(dir_rdiss_scan,
                  "final_phi_perturbation_rdiss$(rdiss_val)e-9_highres.png"),
                  final_fig)
end

Legend(fig[2,1], ax; tellheight=true, tellwidth=false)
resize_to_layout!(fig)
save(joinpath(dir_rdiss_scan, "rdiss-scan_highres.png"), fig)

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

    _, mid_ind = plot_mode_amplitude(this_ri, phi, ax, irun)

    # make animation of perturbation
    _, perturbation = makie_post_processing.get_r_perturbation(phi)
    outfile = joinpath(dir_no_Krook, "phi_perturbation_no-Krook_r-nelement$(this_ri.r.nelement_global).gif")
    title = "no Krook, r_nelement = $(this_ri.r.nelement_global)"
    makie_post_processing.animate_2d(this_ri.z.grid, this_ri.r.grid, perturbation,
                                     xlabel="z", ylabel="r", title=title,
                                     colormap="reverse_deep", outfile=outfile)

    # Plot final time point
    final_perturbation = @view perturbation[:,:,mid_ind]
    final_fig, final_ax, hm = heatmap(this_ri.z.grid, this_ri.r.grid, final_perturbation)
    Colorbar(final_fig[1,2], hm)
    final_ax.xlabel = "z"
    final_ax.ylabel = "r"
    final_ax.title = "no Krook, r_nelement = $(this_ri.r.nelement_global)"
    save(joinpath(dir_no_Krook,
                  "final_phi_perturbation_no-Krook_r-nelement$(this_ri.r.nelement_global).png"),
                  final_fig)
end

Legend(fig[2,1], ax; tellheight=true, tellwidth=false)
resize_to_layout!(fig)
save(joinpath(dir_no_Krook, "growth_rate_no-Krook_r-resolution-scan.png"), fig)

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
save(joinpath(dir_periodic5, "periodic5_background_Tpar.png"), fig_bg)

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

    γ, mid_ind = plot_mode_amplitude(this_ri, phi, ax, irun; label="An=$An")
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
    final_perturbation = @view perturbation[:,:,mid_ind]
    final_fig, final_ax, hm = heatmap(this_ri.z.grid, this_ri.r.grid, final_perturbation)
    Colorbar(final_fig[1,2], hm)
    final_ax.xlabel = "z"
    final_ax.ylabel = "r"
    final_ax.title = "periodic5, AT=$(AT) An=$(An)"
    save(joinpath(dir_periodic5,
                  "final_phi_perturbation_periodic5_AT$(AT)_An$(An).png"),
                  final_fig)
end

Legend(fig_bg[2,1], ax_bg; tellheight=true, tellwidth=false)
resize_to_layout!(fig_bg)
save(joinpath(dir_periodic5, "periodic5_background_n.png"), fig_bg)

xlims!(ax, 0.0, 20.0)
Legend(fig[2,1], ax; tellheight=true, tellwidth=false)
resize_to_layout!(fig)
save(joinpath(dir_periodic5, "growth_rate_periodic5-AT$(AT)-An-scan.png"), fig)

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

save(joinpath(dir_periodic5, "growth_rate_vs_LToverLn.png"), fig)

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
save(joinpath(dir_periodic5_Fourier_r, "periodic5_Fourier-r_background_Tpar.png"), fig_bg)

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

    γ, mid_ind = plot_mode_amplitude(this_ri, phi, ax, irun; label="An=$An")
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
    final_perturbation = @view perturbation[:,:,mid_ind]
    final_fig, final_ax, hm = heatmap(this_ri.z.grid, this_ri.r.grid, final_perturbation)
    Colorbar(final_fig[1,2], hm)
    final_ax.xlabel = "z"
    final_ax.ylabel = "r"
    final_ax.title = "periodic5, AT=$(AT) An=$(An)"
    save(joinpath(dir_periodic5_Fourier_r,
                  "final_phi_perturbation_periodic5_Fourier-r_AT$(AT)_An$(An).png"),
                  final_fig)
end

Legend(fig_bg[2,1], ax_bg; tellheight=true, tellwidth=false)
resize_to_layout!(fig_bg)
save(joinpath(dir_periodic5_Fourier_r, "periodic5_Fourier-r_background_n.png"), fig_bg)

xlims!(ax, 0.0, 20.0)
Legend(fig[2,1], ax; tellheight=true, tellwidth=false)
resize_to_layout!(fig)
save(joinpath(dir_periodic5_Fourier_r, "growth_rate_periodic5_Fourier-r-AT$(AT)-An-scan.png"), fig)

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

save(joinpath(dir_periodic5_Fourier_r, "Fourier-r_growth_rate_vs_LToverLn.png"), fig)

# Compare case in periodic box with a parallel length of 2m (instead of the previous 10m),
# and the source-sustained 'background' perturbation having a z-variation with a
# wavenumber of 1, using Fourier discretization for r.
dir_periodic5_Fourier_r = mkpath(joinpath(plots_dir, "instability_2D_periodic_Lz0.02_Fourier-r"))
AT = 0.1
Lz002_run_dirs = (("runs/2D1V-instability-periodic-Lz0.02-test-Fourier-r-AT0.1-An0.0/", 0.0),
                  ("runs/2D1V-instability-periodic-Lz0.02-test-Fourier-r-AT0.1-An0.005/", 0.005),
                  ("runs/2D1V-instability-periodic-Lz0.02-test-Fourier-r-AT0.1-An0.01/", 0.01),
                  ("runs/2D1V-instability-periodic-Lz0.02-test-Fourier-r-AT0.1-An0.015/", 0.015),
                  ("runs/2D1V-instability-periodic-Lz0.02-test-Fourier-r-AT0.1-An0.02/", 0.02),
                  ("runs/2D1V-instability-periodic-Lz0.02-test-Fourier-r-AT0.1-An0.025/", 0.025),
                  ("runs/2D1V-instability-periodic-Lz0.02-test-Fourier-r-AT0.1-An0.03/", 0.03),
                  ("runs/2D1V-instability-periodic-Lz0.02-test-Fourier-r-AT0.1-An0.04/", 0.04),
                  ("runs/2D1V-instability-periodic-Lz0.02-test-Fourier-r-AT0.1-An-0.005/", -0.005),
                  ("runs/2D1V-instability-periodic-Lz0.02-test-Fourier-r-AT0.1-An-0.01/", -0.01),
                  ("runs/2D1V-instability-periodic-Lz0.02-test-Fourier-r-AT0.1-An-0.02/", -0.02),
                  ("runs/2D1V-instability-periodic-Lz0.02-test-Fourier-r-AT0.1-An-0.03/", -0.03),
                  ("runs/2D1V-instability-periodic-Lz0.02-test-Fourier-r-AT0.1-An-0.04/", -0.04),
                  ("runs/2D1V-instability-periodic-Lz0.02-test-Fourier-r-AT0.1-An-0.05/", -0.05),
                  ("runs/2D1V-instability-periodic-Lz0.02-test-Fourier-r-AT0.1-An-0.1/", -0.1),
                 )
ri_Lz002 = Any[(get_run_info(d; dfns=true), An) for (d, An) ∈ Lz002_run_dirs]

fig_bg, ax_bg = get_1d_ax(xlabel="z", ylabel="T_∥")
Tpar = get_variable(ri_Lz002[1][1], "parallel_pressure"; it=1, is=1, ir=1) ./ get_variable(ri_Lz002[1][1], "density"; it=1, is=1, ir=1)
lines!(ax_bg, ri_Lz002[1][1].z.grid, Tpar)
save(joinpath(dir_periodic5_Fourier_r, "periodic_Lz0.02_Fourier-r_background_Tpar.png"), fig_bg)

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
for (irun, (this_ri, An)) ∈ enumerate(ri_Lz002)
    lines!(ax_bg, this_ri.z.grid, get_variable(this_ri, "density"; it=1, is=1, ir=1);
           label=this_ri.run_name)

    phi = get_variable(this_ri, "phi")

    γ, mid_ind = plot_mode_amplitude(this_ri, phi, ax, irun; label="An=$An")
    push!(An_list, An)
    push!(gammas, γ)

    # make animation of perturbation
    _, perturbation = makie_post_processing.get_r_perturbation(phi)
    outfile = joinpath(dir_periodic5_Fourier_r, "phi_perturbation_Fourier-r_periodic_Lz0.02_AT$(AT)_An$(An).gif")
    title = "periodic, Lz0.02, AT=$(AT) An=$(An)"
    makie_post_processing.animate_2d(this_ri.z.grid, this_ri.r.grid, perturbation,
                                     xlabel="z", ylabel="r", title=title,
                                     colormap="reverse_deep", outfile=outfile)

    # Plot final time point
    final_perturbation = @view perturbation[:,:,mid_ind]
    final_fig, final_ax, hm = heatmap(this_ri.z.grid, this_ri.r.grid, final_perturbation)
    Colorbar(final_fig[1,2], hm)
    final_ax.xlabel = "z"
    final_ax.ylabel = "r"
    final_ax.title = "periodic, Lz0.02, AT=$(AT) An=$(An)"
    save(joinpath(dir_periodic5_Fourier_r,
                  "final_phi_perturbation_periodic_Lz0.02_Fourier-r_AT$(AT)_An$(An).png"),
                  final_fig)
end

Legend(fig_bg[2,1], ax_bg; tellheight=true, tellwidth=false)
resize_to_layout!(fig_bg)
save(joinpath(dir_periodic5_Fourier_r, "periodic_Lz0.02_Fourier-r_background_n.png"), fig_bg)

xlims!(ax, 0.0, 4.0)
Legend(fig[2,1], ax; tellheight=true, tellwidth=false)
resize_to_layout!(fig)
save(joinpath(dir_periodic5_Fourier_r, "growth_rate_periodic_Lz0.02_Fourier-r-AT$(AT)-An-scan.png"), fig)

max_positive_gamma_ind = 8
fig, ax, s = scatter(An_list[1:max_positive_gamma_ind] ./ AT, gammas[1:max_positive_gamma_ind])
first_negative_An_ind = 9
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
ylims!(ax, 0.0, nothing)

save(joinpath(dir_periodic5_Fourier_r, "Fourier-r_Lz0.02_growth_rate_vs_LToverLn.png"), fig)

# z-resolution scan.
dir_zres = mkpath(joinpath(plots_dir, "instability_2D_periodic5_Fourier-r_z-resolution"))
AT = 0.1
zres_run_dirs = (("runs/2D1V-instability-periodic5-test-Fourier-r-AT0.1-An0.0/", 32),
                  ("runs/2D1V-instability-periodic5-test-Fourier-r-AT0.1-An0.0-z-nelement64/", 64),
                  ("runs/2D1V-instability-periodic5-test-Fourier-r-AT0.1-An0.0-z-nelement128/", 128),
                  ("runs/2D1V-instability-periodic5-test-Fourier-r-AT0.1-An0.0-z-nelement256/", 256),
                  ("runs/2D1V-instability-periodic5-test-Fourier-r-AT0.1-An0.0-z-nelement512/", 512),
                 )
ri_zres = Any[(get_run_info(d; dfns=true), An) for (d, An) ∈ zres_run_dirs]

fig, ax = get_1d_ax(xlabel="time", ylabel="amplitude", yscale=log10)
# Ensure the first row width is 3/4 of the column width so that the plot does not get
# squashed by the legend
rowsize!(fig.layout, 1, Aspect(1, 3/4))

zres_list = Int64[]
gammas = Float64[]
for (irun, (this_ri, nel)) ∈ enumerate(ri_zres)
    phi = get_variable(this_ri, "phi")

    γ, mid_ind = plot_mode_amplitude(this_ri, phi, ax, irun; label="nel=$nel")
    push!(zres_list, nel)
    push!(gammas, γ)

    # make animation of perturbation
    _, perturbation = makie_post_processing.get_r_perturbation(phi)
    outfile = joinpath(dir_zres, "phi_perturbation_Fourier-r_periodic5_z-nelement$(nel).gif")
    title = "periodic5, z_nelement=$(nel)"
    makie_post_processing.animate_2d(this_ri.z.grid, this_ri.r.grid, perturbation,
                                     xlabel="z", ylabel="r", title=title,
                                     colormap="reverse_deep", outfile=outfile)

    # Plot final time point
    final_perturbation = @view perturbation[:,:,mid_ind]
    final_fig, final_ax, hm = heatmap(this_ri.z.grid, this_ri.r.grid, final_perturbation)
    Colorbar(final_fig[1,2], hm)
    final_ax.xlabel = "z"
    final_ax.ylabel = "r"
    final_ax.title = "periodic5, z_nelement=$(nel)"
    save(joinpath(dir_zres,
                  "final_phi_perturbation_periodic5_z-nelement$(nel).png"),
                  final_fig)
end

xlims!(ax, 0.0, 4.0)
Legend(fig[2,1], ax; tellheight=true, tellwidth=false)
resize_to_layout!(fig)
save(joinpath(dir_zres, "growth_rate_periodic5_zresolution.png"), fig)

fig, ax, s = scatter(zres_list)
ax.xlabel = "z_nelement"
ax.ylabel = "γ"

ylims!(ax, 0.0, nothing)

save(joinpath(dir_zres, "Fourier-r_growth_rate_vs_z-nelement.png"), fig)
