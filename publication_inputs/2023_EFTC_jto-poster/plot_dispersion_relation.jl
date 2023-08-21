#using moment_kinetics.makie_post_processing

using Optim
#using Roots
using SpecialFunctions

"""
Equation (4.13) from Parra et al. "1D drift kinetic models with periodic
boundary conditions" (TN-01)
"""
function plasmaZ(zeta)
    #return exp(-zeta^2)*sqrt(π)*(im - erfi(zeta))
    return (sqrt(π)*exp(-zeta^2)*im - 2*dawson(zeta))
end

# Only use one Lpar
const Lpar = 1.0

"""
The solution of (4.7) for a complex frequency
"""
function solve_dispersion_relation(ni, nn, Th, Te, Rin; initial_guesses=nothing, n_solutions=1)
    kpar = 2*π/Lpar
    vth = sqrt(Th) # in moment_kinetics normalised units

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
            initial_guess = [real(initial_guesses[i]), imag(real(initial_guesses[i]))]
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
        result = optimize(target, initial_guess; g_tol=1.0e-40)
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
"""
function get_sequence_vs_Ri(ni, nn, Th, Te; starting_omega)
    # Get a starting point with Ri=1
    function get_param_sequence(value, default)
        sign = value > default ? -1 : 1
        sequence = reverse(collect(value:sign*0.01:default))
        return sequence
    end
    omega_initial = starting_omega
    for (this_ni, this_nn, this_Th, this_Te) ∈
        zip(get_param_sequence(ni, default_parameters.ni),
            get_param_sequence(nn, default_parameters.nn),
            get_param_sequence(Th, default_parameters.Th),
            get_param_sequence(Te, default_parameters.Te))
        omega_initial = solve_dispersion_relation(this_ni, this_nn, this_Th, this_Te, 1.0;
                                                  initial_guesses=[omega_initial])
    end
    Ri = collect(0.0:0.001:2.0)
    omega_result = similar(Ri)
    startind = findfirst(x->abs(x-1.0)<1.e-8, Ri)

    # Go down from omega_initial
    omega = omega_initial
    for i ∈ startind:-1:1
        omega = solve_dispersion_relation(ni, nn, Th, Te, Ri[i]; initial_guesses=[omega])
        omega_result[i] = omega
    end

    # Go up from omega_initial
    omega = omega_initial
    for i ∈ startind+1:length(Ri)
        omega = solve_dispersion_relation(ni, nn, Th, Te, Ri[i]; initial_guesses=[omega])
        omega_result[i] = omega
    end

    return Ri, real.(omega), imag.(omega)
end

function plot_zero_frequency!(ax, ni, nn, Th, Te)
    Ri, omega, gamma = get_sequence_vs_Ri(ni, nn, Th, Te; starting_omega=default_zero_frequency)
end
