using Plots

using moment_kinetics.input_structs: grid_input, advection_input
using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.calculus: derivative!

function setup(discretization, ngrid)
    # define inputs needed for the test
    L = 2.0
    bc = "periodic"
    # fd_option and adv_input not actually used so given values unimportant
    fd_option = ""
    adv_input = advection_input("default", 1.0, 0.0, 0.0)
    # create the 'input' struct containing input info needed to create a
    # coordinate
    input = grid_input("coord", ngrid, 1, L,
        discretization, fd_option, bc, adv_input)
    # create the coordinate struct 'x'
    x, spectral = define_coordinate(input)

    return x, spectral
end

discretization_list = ["finite_difference", "chebyshev_pseudospectral",
                       "chebyshev_pseudospectral_matrix_multiply", "lagrange_uniform"]

p = plot(yaxis=:log)
ngrids = [n for n ∈ 8:40]

for discretization ∈ discretization_list
    errors = Vector{Float64}(undef, 0)
    for ngrid ∈ ngrids
        x, spectral = setup(discretization, ngrid)

        f = sinpi.(x.grid)

        df_exact = π * cospi.(x.grid)

        df_numerical = similar(f)
        derivative!(df_numerical, f, x, spectral)

        # RMS error
        push!(errors, sqrt(sum(df_numerical - df_exact)^2))

    end
    println(discretization)
    println(ngrids)
    println(errors)
    println()
    plot!(p, ngrids, errors, label=discretization)
end

savefig(p, "derivative_convergence.pdf")
