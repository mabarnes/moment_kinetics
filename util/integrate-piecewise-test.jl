using moment_kinetics.input_structs: grid_input, advection_input
using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.calculus: integral

using Plots
pyplot()

function test_one(type, n, ngrid, nelement)
    # define inputs needed for the test
    L = 2.0
    bc = "constant"
    # fd_option and adv_input not actually used so given values unimportant
    fd_option = ""
    adv_input = advection_input("default", 1.0, 0.0, 0.0)
    # create the 'input' struct containing input info needed to create a
    # coordinate
    input = grid_input("coord", ngrid, nelement, L, type, fd_option, bc,
        adv_input)
    # create the coordinate struct 'x'
    x, spectral = define_coordinate(input)

    # Break point at some non-special location
    x0 = -0.376489
    #x0 = -1.0

    #if nelement==1 && ngrid==3
    #    println("\nn=$n")
    #end
    #println("ngrid=$ngrid, nelement=$nelement")
    f = zeros(x.n)
    for i ∈ 1:x.n
        if x.grid[i] > x0
            f[i] = (x.grid[i] - x0)^n
        end
    end
    expected = (x.grid[end] - x0)^(n+1)/(n+1)
    actual = integral(f, x.wgts)
    local err = actual - expected

    # Estimate dx
    local dx = x.L / x.n

    #println("expected=$expected, actual=$actual, error=$err")
    return dx, err
end

type = "finite_difference"
for n ∈ 1:8
    plot(reuse=false, title="$(type) polynomial order $n")
    nelement = 1
    dx = []
    err = []
    for ngrid ∈ 10:300
        this_dx, this_err = test_one(type, n, ngrid, nelement)
        push!(dx, this_dx)
        push!(err, abs(this_err))
    end

    println(dx)
    println(err)
    inds = err .> 0
    plot!(dx[inds], err[inds], axis=:log)

    # check convergence order
    trendpower = min(n + 1, 4)
    prefactor = err[1] / dx[1]^(trendpower)
    plot!(dx, prefactor*dx.^(trendpower), linestyle=:dot, color=:black, linewidth=2,
          label="dx^$trendpower", ylims=ylims())

    gui()
    savefig("$(type)_n$(n).png")
end

type = "chebyshev_pseudospectral"
println("\n$type")
#@testset "$n $nelement $ngrid" for n ∈ 1:8, nelement ∈ 1:5, ngrid ∈ 3:33
#for n ∈ 1:8, ngrid ∈ 3:33
for n ∈ 1:8
    plot(reuse=false, title="$(type) polynomial order $n")
    dx = []
    err = []
    for ngrid ∈ 3:33
        dx = []
        err = []
        for nelement ∈ 1:40
            this_dx, this_err = test_one(type, n, ngrid, nelement)
            push!(dx, this_dx)
            push!(err, abs(this_err))
        end

        println(dx)
        println(err)
        inds = err .> 0
        plot!(dx[inds], err[inds], axis=:log, label="ngrid=$ngrid")
    end

    # check convergence order
    trendpower = n + 1
    prefactor = err[1] / dx[1]^(trendpower)
    plot!(dx, prefactor*dx.^(trendpower), linestyle=:dot, color=:black, linewidth=2,
          label="dx^$trendpower", ylims=ylims())

    gui()
    savefig("$(type)_n$(n).png")
end
