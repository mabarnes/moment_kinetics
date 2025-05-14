#using moment_kinetics.makie_post_processing

using makie_post_processing: CairoMakie
using DelimitedFiles
using LaTeXStrings
using MathTeXEngine
using Optim
using SpecialFunctions

using PlasmaDispersionFunctions

using moment_kinetics.parameter_scans

const ext = ".png"
const plot_dir = "sound-wave"

CairoMakie.activate!(; px_per_unit=4)
update_theme!(fontsize=24, fonts=(; regular=texfont(:text), bold=texfont(:bold),
                                  italic=texfont(:italic),
                                  bold_italic=texfont(:bolditalic)))

"""
Equation (4.13) from Parra et al. "1D drift kinetic models with periodic
boundary conditions" (TN-01)
"""
function plasmaZ(zeta)
    #return exp(-zeta^2)*sqrt(π)*(im - erfi(zeta))
    #return (sqrt(π)*exp(-zeta^2)*im - 2*dawson(zeta))
    return plasma_dispersion_function(zeta)
end

# Only use one Lpar
const Lpar = 1.0
const kpar = 2*π/Lpar

"""
The solution of (4.7) for a complex frequency

Extra `kwargs...` are passed to `optimize()`.
"""
function solve_dispersion_relation(ni, nn, Th, Te, Rin; initial_guesses=nothing,
                                   n_solutions=1, kwargs...)
    vth = sqrt(2*Th) # in moment_kinetics normalised units

    function zeta(omega)
        return omega / (kpar * vth)
    end
    function zetain(omega)
        return (omega + im*(ni+nn)*Rin) / (kpar * vth)
    end
    function Aii(omega)
        return (
            1 + Te/Th + ni/(ni+nn) * Te/Th * zeta(omega) * plasmaZ(zeta(omega))
            + nn/(ni+nn) * ((1 + Te/Th)*zetain(omega) - zeta(omega))*plasmaZ(zetain(omega))
        )
    end
    function Ain(omega)
        return -ni/(ni+nn) * (zetain(omega) - zeta(omega)) * plasmaZ(zetain(omega))
    end
    function Ani(omega)
        return - nn/(ni+nn) * (((1 + Te/Th)*zetain(omega) - zeta(omega)) * plasmaZ(zetain(omega))
                               - Te/Th * zeta(omega) * plasmaZ(zeta(omega)))
    end
    function Ann(omega)
        return 1 + ni/(ni+nn)*(zetain(omega) - zeta(omega)) * plasmaZ(zetain(omega))
    end

    function detA(omega)
        return Aii(omega)*Ann(omega) - Ain(omega)*Ani(omega)
    end

    omega_list = ComplexF64[]
    omega = 0.0 + 0.0im
    for i ∈ 1:n_solutions # find at most n_solutions solutions
        if initial_guesses !== nothing && i <= length(initial_guesses)
            initial_guess = [real(initial_guesses[i]), imag(initial_guesses[i])]
        else
            initial_guess = [0.0, 0.0]
        end
        function target(x)
            omega = x[1] + im*x[2]
            d = detA(omega)
            # Push away from positive gamma (where there are no solutions)
            d *= exp(0.1*imag(omega))
            for omega_solution ∈ omega_list
                # Use (1 + 1/abs(omega - omega_solution))^2 because the zero-frequency
                # solution gets picked up twice with 1/abs(omega - omega_solution)^1.
                # Maybe that one is a double pole??
                d *= (1.0 + 1.0/abs(omega - omega_solution))^2
            end
            return real(d * conj(d))
        end
        result = optimize(target, initial_guess; g_tol=1.0e-40, kwargs...)
        minimizer = result.minimizer
        omega = minimizer[1] + im*minimizer[2]
        if abs(detA(omega)) < 1.e-10
            push!(omega_list, omega)
        end
        if abs(detA(omega)) > 1.0e-1
            break
        end
    end

    #return detA, omega_list
    return omega_list
end

# For these parameters, solve_dispersion_relation() finds 3 solutions - a pair with the
# same decay rate and positive/negative frequency, and a zero-frequency mode (also
# decaying)
const default_parameters = (ni=1.0, nn=1.0, Th=1.0, Te=1.0, Ri=1.0)
const default_solution = solve_dispersion_relation(values(default_parameters)...; n_solutions=3)
const default_zero_frequency = first(omega for omega ∈ default_solution if abs(real(omega)) < 1e-5)
const default_positive_frequency = first(omega for omega ∈ default_solution if real(omega) > 1e-5)

"""
    get_sequence_vs_Ri(ni, nn, Th, Te; starting_omega)

Get growth rate for the zero frequency mode for a range of Ri from 0 to 2 for the
parameters `ni`, `nn`, `Th`, `Te`.

`starting_omega` gives the solution to start from (e.g. zero-frequency or
positive-frequency).

Extra `kwargs...` are passed to `solve_dispersion_relation`
"""
function get_sequence_vs_Ri(ni, nn, Th, Te; starting_omega, kwargs...)
    # Get cached results if they exist
    cachedir = "omega_caches"
    mkpath(cachedir)
    is_zero = (abs(real(starting_omega)) < 1.e-10)
    if is_zero
        cachefile = joinpath(cachedir, "omega_zero_ni$(ni)_nn$(nn)_Th$(Th)_Te$(Te).cache")
    else
        cachefile = joinpath(cachedir, "omega_positive_ni$(ni)_nn$(nn)_Th$(Th)_Te$(Te).cache")
    end
    if isfile(cachefile)
        results = readdlm(cachefile)
        Ri_result = results[:,1]
        omega_real = results[:,2]
        gamma = results[:,3]
        return Ri_result, omega_real, gamma
    end

    # Get a starting point with Ri=1
    function get_param_sequence(value, default)
        return collect(range(default, stop=value, length=100))
    end

    if abs(real(starting_omega)) < 1.0e-10 && ni == 0.0
        # The ni=0 case has funny behaviour with a transition at about Rin=4.
        # Want to start following the solution from R>4.
        Ri = 5.0 * sqrt(2)
    else
        Ri = 1.0 * sqrt(2)
    end

    omega_initial = starting_omega
    for (this_ni, this_nn, this_Th, this_Te, this_Ri) ∈
        zip(get_param_sequence(ni, default_parameters.ni),
            get_param_sequence(nn, default_parameters.nn),
            get_param_sequence(Th, default_parameters.Th),
            get_param_sequence(Te, default_parameters.Te),
            get_param_sequence(Ri, default_parameters.Ri))
        omega_initial = solve_dispersion_relation(this_ni, this_nn, this_Th, this_Te,
                                                  this_Ri;
                                                  initial_guesses=[omega_initial],
                                                  kwargs...)[1]
    end
    vth = sqrt(2 * Th)
    Ri_result = collect(0.0:0.001*sqrt(2):2.0*π*vth)
    omega_result = similar(Ri_result, ComplexF64)
    omega_result .= NaN
    startind = findfirst(x->abs(x-Ri)<1.e-8*sqrt(2), Ri_result)

    # Go down from omega_initial
    omega = omega_initial
    for i ∈ startind:-1:1
        omega = solve_dispersion_relation(ni, nn, Th, Te, Ri_result[i]; initial_guesses=[omega],
                                          kwargs...)
        if length(omega) == 0
            # Failed to find solution
            break
        elseif (i < startind - 1 && abs(real(starting_omega)) < 1.e-10 * sqrt(2) && (imag(omega[1])-imag(omega_result[i+1])) * (imag(omega_result[startind-1])-imag(omega_result[startind])) < 0.0)
            # Expect growth rate to be monotonic for zero frequency mode. Stop if it is not
            break
        end
        omega = omega[1]
        omega_result[i] = omega
    end

    # Go up from omega_initial
    omega = omega_initial
    for i ∈ startind+1:length(Ri_result)
        omega = solve_dispersion_relation(ni, nn, Th, Te, Ri_result[i]; initial_guesses=[omega],
                                          kwargs...)
        if length(omega) == 0
            # Failed to find solution
            break
        else
            omega = omega[1]
        end
        #println("continuing omega up ", i, " ", omega)
        omega_result[i] = omega
    end

    # Cache the results
    omega_real = real.(omega_result)
    gamma = imag.(omega_result)
    open(cachefile, "w") do io
        writedlm(io, [Ri_result omega_real gamma])
    end
    return Ri_result, omega_real, gamma
end

function plot_zero_frequency!(ax, ni, nn, Th, Te; kwargs...)
    println("plotting zero frequency for ni=$ni, nn=$nn, Th=$Th, Te=$Te")
    Ri, omega, gamma = get_sequence_vs_Ri(ni, nn, Th, Te;
                                          starting_omega=default_zero_frequency)
    # Hopefully there were only a few entries that jumped to the wrong root. Delete them.
    wrong_root_indices = findall(x->abs(x)>1.0e-10 * sqrt(2), omega)
    deleteat!(Ri, wrong_root_indices)
    deleteat!(omega, wrong_root_indices)
    deleteat!(gamma, wrong_root_indices)
    #if any(@. abs(omega) > 1.e-10)
    #    error("expected zero-frequency mode, got $omega")
    #end

    # There is the odd point with gamma=0 that doesn't look right, so skip those too
    wrong_root_indices = findall(x->abs(x)<1.0e-10 * sqrt(2), gamma)
    deleteat!(Ri, wrong_root_indices)
    deleteat!(omega, wrong_root_indices)
    deleteat!(gamma, wrong_root_indices)

    vth = sqrt(Th)
    return lines!(ax, (ni+nn).*Ri./(kpar*vth), gamma./(kpar*vth), linestyle=:dash; kwargs...),
           Ri, gamma
end

function plot_positive_frequency!(ax_omega, ax_gamma, ni, nn, Th, Te; kwargs...)
    println("plotting positive frequency for ni=$ni, nn=$nn, Th=$Th, Te=$Te")
    Ri, omega, gamma = get_sequence_vs_Ri(ni, nn, Th, Te; starting_omega=default_positive_frequency)
    #if any(@. omega < 1.e-5)
    #    error("expected positive-frequency mode, got $omega")
    #end

    Ri_gamma = copy(Ri)
    # Points with omega=0 overlap with zero-frequency mode on gamma-plot and look weird,
    # so chop them out here
    wrong_root_indices = findall(x->abs(x)<1.0e-10 * sqrt(2), omega)
    deleteat!(Ri_gamma, wrong_root_indices)
    deleteat!(gamma, wrong_root_indices)

    vth = sqrt(Th)
    return lines!(ax_omega, (ni+nn).*Ri./(kpar*vth), omega./(kpar*vth); kwargs...),
           lines!(ax_gamma, (ni+nn).*Ri_gamma./(kpar*vth), gamma./(kpar*vth); kwargs...),
           Ri_gamma, gamma
end

function find_crossing_xvalue(x1, y1, x2, y2)
    @boundscheck length(x1) == length(y1) || throw(DimensionMismatch("x1 and y1 have different lengths"))
    @boundscheck length(x2) == length(y2) || throw(DimensionMismatch("x2 and y2 have different lengths"))
    for i1 ∈ 1:length(x1)-1
        i2 = findfirst(x->(x >= x1[i1]), x2)
        if i2 === nothing || i2 == length(x2)
            # No entries left in x2/y2
            break
        end
        if (y2[i2] - y1[i1]) * (y2[i2+1] - y1[i1+1]) <= 0.0
            # Found a crossing. Return its x-value
            xl1 = x1[i1]
            yl1 = y1[i1]
            xu1 = x1[i1+1]
            yu1 = y1[i1+1]
            xl2 = x2[i2]
            yl2 = y2[i2]
            xu2 = x2[i2+1]
            yu2 = y2[i2+1]
            # Line 1 goes between (xl1,yl1) and (xu1, yu1) => y = yl1 + (yu1-yl1)*(x-xl1)/(xu1-xl1)
            # Line 2 goes between (xl2,yl2) and (xu2, yu2) => y = yl2 + (yu2-yl2)*(x-xl2)/(xu2-xl2)
            # Intersection when:
            # yl1 + (yu1-yl1)*(x-xl1)/(xu1-xl1) = yl2 + (yu2-yl2)*(x-xl2)/(xu2-xl2)
            # (yl1-yl2)*(xu1-xl1)*(xu2-xl2) + (yu1-yl1)*(x-xl1)*(xu2-xl2) = (yu2-yl2)*(x-xl2)*(xu1-xl1)
            # (yl1-yl2)*(xu1-xl1)*(xu2-xl2) + (yu2-yl2)*xl2*(xu1-xl1) - (yu1-yl1)*xl1*(xu2-xl2) = (yu2-yl2)*x*(xu1-xl1) - (yu1-yl1)*x*(xu2-xl2)
            # (yl1-yl2)*(xu1-xl1)*(xu2-xl2) + (yu2-yl2)*xl2*(xu1-xl1) - (yu1-yl1)*xl1*(xu2-xl2) = (yu2-yl2)*x*(xu1-xl1) - (yu1-yl1)*x*(xu2-xl2)
            # x = [(yl1-yl2)*(xu1-xl1)*(xu2-xl2) + (yu2-yl2)*xl2*(xu1-xl1) - (yu1-yl1)*xl1*(xu2-xl2)] / [(yu2-yl2)*(xu1-xl1) - (yu1-yl1)*(xu2-xl2)]
            return ((yl1-yl2)*(xu1-xl1)*(xu2-xl2) + (yu2-yl2)*xl2*(xu1-xl1) - (yu1-yl1)*xl1*(xu2-xl2)) / ((yu2-yl2)*(xu1-xl1) - (yu1-yl1)*(xu2-xl2))
        end
    end
    return nothing
end

function get_sim_omega_gamma(sim)
    try
        s = nothing
        open(joinpath("..", "..", sim["output"]["base_directory"], sim["output"]["run_name"], basename(sim["output"]["run_name"]) * ".frequency_fit.txt"),
             "r") do io
            s = split(readline(io))
        end
        gamma = parse(Float64, s[2])
        # No real difference between positive and negative omega, so take abs() in case
        # simulation fit picked up negative value
        omega = abs(parse(Float64, s[4]))
        fit_errors = parse.(Float64, s[6:8])

        if fit_errors[1] > 0.1
            # Fit was bad, so don't plot
            return NaN, NaN
        end

        return omega, gamma
    catch e
        println("Error for ", sim["output"]["run_name"], " ", e)
        return NaN, NaN
    end
end

function plot_sim_output!(ax_omega, ax_gamma, sims, ni, nn, Th, Te; kwargs...)
    vth = sqrt(Th)

    Ri = zeros(length(sims))
    omega = zeros(length(sims))
    gamma = zeros(length(sims))
    for (i, s) ∈ enumerate(sims)
        Ri[i] = s["reactions"]["charge_exchange_frequency"]
        omega[i], gamma[i] = get_sim_omega_gamma(s)
    end

    #println("for sims ")
    #println(Ri)
    #println(omega)
    #println(gamma)

    return scatter!(ax_omega, (ni+nn).*Ri./(kpar*vth), omega./(kpar*vth); kwargs...),
           scatter!(ax_gamma, (ni+nn).*Ri./(kpar*vth), gamma./(kpar*vth); kwargs...)
    # *2 is a fudge!
    #return scatter!(ax_omega, (ni+nn).*Ri./(kpar*vth*2), omega./(kpar*vth); kwargs...),
    #       scatter!(ax_gamma, (ni+nn).*Ri./(kpar*vth*2), gamma./(kpar*vth); kwargs...)
end

const marker1 = BezierPath([MoveTo(-0.5, 0.05), LineTo(0.5, 0.05), LineTo(0.5, -0.05),
                            LineTo(-0.5, -0.05), ClosePath()])
const marker2 = BezierPath([MoveTo((-0.5+0.05)/sqrt(2), (-0.5-0.05)/sqrt(2)),
                            LineTo((0.5+0.05)/sqrt(2), (0.5-0.05)/sqrt(2)),
                            LineTo((0.5-0.05)/sqrt(2), (0.5+0.05)/sqrt(2)),
                            LineTo((-0.5-0.05)/sqrt(2), (-0.5+0.05)/sqrt(2)), ClosePath()])
const marker3 = BezierPath([MoveTo(0.05, -0.5), LineTo(0.05, 0.5), LineTo(-0.05, 0.5),
                            LineTo(-0.05, -0.5), ClosePath()])
const marker4 = BezierPath([MoveTo((0.5+0.05)/sqrt(2), (-0.5+0.05)/sqrt(2)),
                            LineTo((-0.5+0.05)/sqrt(2), (0.5+0.05)/sqrt(2)),
                            LineTo((-0.5-0.05)/sqrt(2), (0.5-0.05)/sqrt(2)),
                            LineTo((0.5-0.05)/sqrt(2), (-0.5-0.05)/sqrt(2)), ClosePath()])

function plot_n_scan()

    Th = 1.0
    Te = 1.0

    fig = Figure(; resolution=(1200, 600))

    ax_gamma = Axis(fig[1,1],
                    xlabel=L"(n_i + n_n)R_{in}/|k_\parallel|v_\mathrm{th}",
                    ylabel=L"\gamma/|k_\parallel|v_\mathrm{th}")

    ax_omega = Axis(fig[1,2],
                    xlabel=L"(n_i + n_n)R_{in}/|k_\parallel|v_\mathrm{th}",
                    ylabel=L"\omega/|k_\parallel|v_\mathrm{th}")

    orig_stdout = stdout
    redirect_stdout(open("/dev/null", "w"))
    sim_inputs = get_scan_inputs(joinpath(plot_dir, "scan_sound-wave_nratio.toml"))
    sim_inputs_split1 = get_scan_inputs(joinpath(plot_dir,
                                                 "scan_sound-wave_nratio_split1.toml"))
    sim_inputs_split2 = get_scan_inputs(joinpath(plot_dir,
                                                 "scan_sound-wave_nratio_split2.toml"))
    sim_inputs_split3 = get_scan_inputs(joinpath(plot_dir,
                                                 "scan_sound-wave_nratio_split3.toml"))
    redirect_stdout(orig_stdout)

    legend_data_list1 = []
    legend_label_list1 = []
    legend_data_list2 = []
    legend_label_list2 = []
    for (ni, nn, label) ∈ ((2.0, 0.0, L"n_i / n_\mathrm{tot} = 1"),
                           (1.5, 0.5, L"n_i / n_\mathrm{tot} = 3/4"),
                           (1.0, 1.0, L"n_i / n_\mathrm{tot} = 1/2"),
                           (0.5, 1.5, L"n_i / n_\mathrm{tot} = 1/4"),
                           (0.0, 2.0, L"n_i / n_\mathrm{tot} = 0"),
                          )
        sims = Tuple(i for i ∈ sim_inputs if isapprox(i["initial_density1"], ni, atol=2.0e-5))
        sims_split1 = Tuple(i for i ∈ sim_inputs_split1 if isapprox(i["initial_density1"], ni, atol=2.0e-5))
        sims_split2 = Tuple(i for i ∈ sim_inputs_split2 if isapprox(i["initial_density1"], ni, atol=2.0e-5))
        sims_split3 = Tuple(i for i ∈ sim_inputs_split3 if isapprox(i["initial_density1"], ni, atol=2.0e-5))

        p_omega, p_gamma, Ri_positive, gamma_positive =
            plot_positive_frequency!(ax_omega, ax_gamma, ni, nn, Th, Te; label=label)
        p_gamma_z, Ri_zero, gamma_zero = plot_zero_frequency!(ax_gamma, ni, nn, Th, Te; color=p_gamma.color)
        s_omega, s_gamma = plot_sim_output!(ax_omega, ax_gamma, sims, ni, nn, Th, Te; color=p_gamma.color, marker=marker1)
        s_omega1, s_gamma1 = plot_sim_output!(ax_omega, ax_gamma, sims_split1, ni, nn, Th, Te; color=p_gamma.color, marker=marker2)
        s_omega2, s_gamma2 = plot_sim_output!(ax_omega, ax_gamma, sims_split2, ni, nn, Th, Te; color=p_gamma.color, marker=marker3)
        s_omega3, s_gamma3 = plot_sim_output!(ax_omega, ax_gamma, sims_split3, ni, nn, Th, Te; color=p_gamma.color, marker=marker4)

        vth = sqrt(Th)
        crossing_x = find_crossing_xvalue(Ri_positive, gamma_positive, Ri_zero, gamma_zero)
        if crossing_x !== nothing
            vlines!(ax_omega, (ni + nn) * crossing_x / (kpar*vth), linestyle=:dot, color=p_gamma.color)
            #vlines!(ax_gamma, (ni + nn) * crossing_x / (kpar*vth), linestyle=:dot, color=p_gamma.color)
        end

        push!(legend_data_list1, [p_omega, s_omega, s_omega1, s_omega2, s_omega3])
        push!(legend_label_list1, label)
    end

    # Add marker types to the legend
    push!(legend_data_list2, MarkerElement(marker=marker1, color=:black))
    push!(legend_label_list2, L"$$full-f")
    push!(legend_data_list2, MarkerElement(marker=marker2, color=:black))
    push!(legend_label_list2, L"evolving $n$")
    push!(legend_data_list2, MarkerElement(marker=marker3, color=:black))
    push!(legend_label_list2, L"evolving $n,\,u_\parallel$")
    push!(legend_data_list2, MarkerElement(marker=marker4, color=:black))
    push!(legend_label_list2, L"evolving $n,\,u_\parallel,\,p_\parallel$")

    Legend(fig[2,1:2], [legend_data_list1, legend_data_list2], [legend_label_list1, legend_label_list2], ["", ""]; tellheight=true, tellwidth=false, nbanks=5)

    save(joinpath(plot_dir, "n_scan$ext"), fig)

    return nothing
end

function plot_T_scan()

    ni = 1.0
    nn = 1.0
    Te = 1.0

    fig = Figure(; resolution=(900, 450))

    ax_gamma = Axis(fig[1,1],
                    xlabel=L"(n_i + n_n)R_{in}/|k_\parallel|v_\mathrm{th}",
                    ylabel=L"\gamma/|k_\parallel|v_\mathrm{th}",
                    limits=(-.1, 2.1, -2.5, 0.0))

    ax_omega = Axis(fig[1,2],
                    xlabel=L"(n_i + n_n)R_{in}/|k_\parallel|v_\mathrm{th}",
                    ylabel=L"\omega/|k_\parallel|v_\mathrm{th}",
                    limits=(-.1, 2.1, -0.1, 2.1))

    orig_stdout = stdout
    redirect_stdout(open("/dev/null", "w"))
    sim_inputs025 = (get_scan_inputs(joinpath(plot_dir, "scan_sound-wave_T0.25.toml")),
                     get_scan_inputs(joinpath(plot_dir, "scan_sound-wave_T0.25_split1.toml")),
                     get_scan_inputs(joinpath(plot_dir, "scan_sound-wave_T0.25_split2.toml")),
                     get_scan_inputs(joinpath(plot_dir, "scan_sound-wave_T0.25_split3.toml")))
    sim_inputs05 = (get_scan_inputs(joinpath(plot_dir, "scan_sound-wave_T0.5.toml")),
                    get_scan_inputs(joinpath(plot_dir, "scan_sound-wave_T0.5_split1.toml")),
                    get_scan_inputs(joinpath(plot_dir, "scan_sound-wave_T0.5_split2.toml")),
                    get_scan_inputs(joinpath(plot_dir, "scan_sound-wave_T0.5_split3.toml")))
    sim_inputs1 = (get_scan_inputs(joinpath(plot_dir, "scan_sound-wave_T1.toml")),
                   get_scan_inputs(joinpath(plot_dir, "scan_sound-wave_T1_split1.toml")),
                   get_scan_inputs(joinpath(plot_dir, "scan_sound-wave_T1_split2.toml")),
                   get_scan_inputs(joinpath(plot_dir, "scan_sound-wave_T1_split3.toml")))
    sim_inputs2 = (get_scan_inputs(joinpath(plot_dir, "scan_sound-wave_T2.toml")),
                   get_scan_inputs(joinpath(plot_dir, "scan_sound-wave_T2_split1.toml")),
                   get_scan_inputs(joinpath(plot_dir, "scan_sound-wave_T2_split2.toml")),
                   get_scan_inputs(joinpath(plot_dir, "scan_sound-wave_T2_split3.toml")))
    sim_inputs4 = (get_scan_inputs(joinpath(plot_dir, "scan_sound-wave_T4.toml")),
                   get_scan_inputs(joinpath(plot_dir, "scan_sound-wave_T4_split1.toml")),
                   get_scan_inputs(joinpath(plot_dir, "scan_sound-wave_T4_split2.toml")),
                   get_scan_inputs(joinpath(plot_dir, "scan_sound-wave_T4_split3.toml")))
    redirect_stdout(orig_stdout)

    legend_data_list1 = []
    legend_label_list1 = []
    legend_data_list2 = []
    legend_label_list2 = []
    for (Th, label, sims) ∈ ((0.25, L"T/T_e = 1/4", sim_inputs025),
                             (0.5, L"T/T_e = 1/2", sim_inputs05),
                             (1.0, L"T/T_e = 1", sim_inputs1),
                             (2.0, L"T/T_e = 2", sim_inputs2),
                             (4.0, L"T/T_e = 4", sim_inputs4),
                            )

        p_omega, p_gamma, Ri_positive, gamma_positive =
            plot_positive_frequency!(ax_omega, ax_gamma, ni, nn, Th, Te; label=label)
        p_gamma_z, Ri_zero, gamma_zero = plot_zero_frequency!(ax_gamma, ni, nn, Th, Te; color=p_gamma.color)
        s_omega, s_gamma = plot_sim_output!(ax_omega, ax_gamma, sims[1], ni, nn, Th, Te; color=p_gamma.color, marker=marker1)
        s_omega1, s_gamma1 = plot_sim_output!(ax_omega, ax_gamma, sims[2], ni, nn, Th, Te; color=p_gamma.color, marker=marker2)
        s_omega2, s_gamma2 = plot_sim_output!(ax_omega, ax_gamma, sims[3], ni, nn, Th, Te; color=p_gamma.color, marker=marker3)
        s_omega3, s_gamma3 = plot_sim_output!(ax_omega, ax_gamma, sims[4], ni, nn, Th, Te; color=p_gamma.color, marker=marker4)

        vth = sqrt(Th)
        crossing_x = find_crossing_xvalue(Ri_positive, gamma_positive, Ri_zero, gamma_zero)
        if crossing_x !== nothing
            vlines!(ax_omega, (ni + nn) * crossing_x / (kpar * vth), linestyle=:dot, color=p_gamma.color)
            #vlines!(ax_gamma, (ni + nn) * crossing_x / (kpar * vth), linestyle=:dot, color=p_gamma.color)
        end

        push!(legend_data_list1, [p_omega, s_omega, s_omega1, s_omega2, s_omega3])
        push!(legend_label_list1, label)
    end

    # Add marker types to the legend
    push!(legend_data_list2, MarkerElement(marker=marker1, color=:black))
    push!(legend_label_list2, L"$$full-f")
    push!(legend_data_list2, MarkerElement(marker=marker2, color=:black))
    push!(legend_label_list2, L"evolving $n$")
    push!(legend_data_list2, MarkerElement(marker=marker3, color=:black))
    push!(legend_label_list2, L"evolving $n,\,u_\parallel$")
    push!(legend_data_list2, MarkerElement(marker=marker4, color=:black))
    push!(legend_label_list2, L"evolving $n,\,u_\parallel,\,p_\parallel$")

    Legend(fig[2,1:2], [legend_data_list1, legend_data_list2], [legend_label_list1, legend_label_list2], ["", ""]; tellheight=true, tellwidth=false, nbanks=5)

    save(joinpath(plot_dir, "T_scan$ext"), fig)

    return nothing
end

function make_plots()
    plot_n_scan()
    plot_T_scan()
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_plots()
end
