"""
We implement the Runge-Kutta timestepper in `moment_kinetics` in a different form from the
most conventional one, so in some cases we need to convert the 'conventional' coefficients
into ones that we can use.
"""

using Symbolics

# Following two functions copied and modified from Symbolics.jl's linear_algebra.jl so
# that we can hack them to force them to return a Rational{BigInt} result.
# Modifications:
#  * Add prefix `my_` to the function names, to avoid confusion/conflicts
#  * Change `Num.()` to `Rational{BigInt}.()` in `_my_solve` so that `A` and `b` are
#    arrays of `Rational{BigInt}` (so that we avoid any rounding errors). For the case
#    that we want, the entries of `A` and `b` are all numerical values (not actual
#    symbolic expressions), so this hack can be done.
#  * Change `/` to `//` in `my_sym_lu2()`
using Symbolics: linear_expansion, SymbolicUtils, value, sym_lu, Num, RCNum, _iszero, nterms
using LinearAlgebra
function my_solve_for(eq, var; simplify=false, check=true) # scalar case
    # simplify defaults for `false` as canonicalization should handle most of
    # the cases.
    a, b, islinear = linear_expansion(eq, var)
    check && @assert islinear
    islinear || return nothing
    # a * x + b = 0
    if eq isa AbstractArray && var isa AbstractArray
        x = _my_solve(a, -b, simplify)
    else
        x = a \ -b
    end
    simplify || return x
    if x isa AbstractArray
        SymbolicUtils.simplify.(simplify_fractions.(x))
    else
        SymbolicUtils.simplify(simplify_fractions(x))
    end
end

function _my_solve(A::AbstractMatrix, b::AbstractArray, do_simplify)
    #A = Num.(value.(SymbolicUtils.quick_cancel.(A)))
    #b = Num.(value.(SymbolicUtils.quick_cancel.(b)))
    A = Rational{BigInt}.(value.(SymbolicUtils.quick_cancel.(A)))
    b = Rational{BigInt}.(value.(SymbolicUtils.quick_cancel.(b)))
    sol = value.(sym_lu(A) \ b)
    do_simplify ? SymbolicUtils.simplify_fractions.(sol) : sol
end

function my_solve_for2(eq, var; simplify=false, check=true) # scalar case
    # simplify defaults for `false` as canonicalization should handle most of
    # the cases.
    a, b, islinear = linear_expansion(eq, var)
    check && @assert islinear
    islinear || return nothing
    # a * x + b = 0
    if eq isa AbstractArray && var isa AbstractArray
        x = _my_solve2(a, -b, simplify)
    else
        x = a \ -b
    end
    simplify || return x
    if x isa AbstractArray
        SymbolicUtils.simplify.(simplify_fractions.(x))
    else
        SymbolicUtils.simplify(simplify_fractions(x))
    end
end

function _my_solve2(A::AbstractMatrix, b::AbstractArray, do_simplify)
    A = Num.(value.(SymbolicUtils.quick_cancel.(A)))
    b = Num.(value.(SymbolicUtils.quick_cancel.(b)))
    sol = value.(my_sym_lu2(A) \ b)
    do_simplify ? SymbolicUtils.simplify_fractions.(sol) : sol
end

function my_sym_lu2(A; check=true)
    SINGULAR = typemax(Int)
    m, n = size(A)
    F = map(x->x isa RCNum ? x : Num(x), A)
    minmn = min(m, n)
    p = Vector{LinearAlgebra.BlasInt}(undef, minmn)
    info = 0
    for k = 1:minmn
        kp = k
        amin = SINGULAR
        for i in k:m
            absi = _iszero(F[i, k]) ? SINGULAR : nterms(F[i,k])
            if absi < amin
                kp = i
                amin = absi
            end
        end

        p[k] = kp

        if amin == SINGULAR && !(amin isa Symbolic) && (amin isa Number) && iszero(info)
            info = k
        end

        # swap
        for j in 1:n
            F[k, j], F[kp, j] = F[kp, j], F[k, j]
        end

        for i in k+1:m
            F[i, k] = F[i, k] // F[k, k]
        end
        for j = k+1:n
            for i in k+1:m
                F[i, j] = F[i, j] - F[i, k] * F[k, j]
            end
        end
    end
    check && LinearAlgebra.checknonsingular(info)
    LU(F, p, convert(LinearAlgebra.BlasInt, info))
end

"""
    convert_butcher_tableau_for_moment_kinetics(a, b)

Convert a Butcher tableau describing a Runge-Kutta method (see e.g.
https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods) to coefficients for
moment_kinetics, which implements Runge-Kutta timestepping in terms of 'forward Euler
steps'.

Ignores the \$c\$ coefficient in the Butcher tableau as we do not have explicit time
dependence in the RHS in moment_kinetics.

`a` is an array giving the \$a_{i,j}\$ Butcher coefficients.

For a fixed step RK method `b` would be a vector. For an embedded RK method that uses
adaptive timestepping, `b` will be an \$2\\times n\$ matrix. The first row gives the
higher-order updated solution, and the second row gives the lower-order updated solution
that can be used to calculate an error estimate.

Currently assumes the method is explicit, so `a` has no non-zero diagonal or
upper-triangular elements.

Returns an array `rk_coefs` of size `n_rk_stages`x`n_rk_stages` where `size(a) =
(n_rk_stages, n_rk_stages)`.
"""
function convert_butcher_tableau_for_moment_kinetics(a, b; low_storage=true)
    using_rationals = eltype(a) <: Rational
    n_rk_stages = size(a, 1)
    if size(b, 1) > 1
        adaptive = true
        output_size = n_rk_stages + 1
    else
        adaptive = false
        b = reshape(b, 1, length(b))
        output_size = n_rk_stages
    end

    # f is the RHS function: dy/dt = f(y)

    # y are the updated solution vectors - the things given as the arguments to k[i] in
    #     the Wikipedia page k[i] = f(t+c[i]*h, y[i]), except y[n_rk_stages+1] which is
    #     the higher-order updated solution.
    # y_out are the same as y, but given as expressions in terms of y and f
    # k are the RHS evaluations as defined on the Wikipedia page
    # k_subs are the k evaluated in terms of y by back-substituting the definitions of y.
    @variables y[1:n_rk_stages+1] k[1:n_rk_stages] yn rk_coefs[1:n_rk_stages+1, 1:output_size]
    y = Symbolics.scalarize(y)
    k = Symbolics.scalarize(k)
    rk_coefs = Symbolics.scalarize(rk_coefs)

    # Expressions defined using the 'standard' Butcher formulae
    y_k_expressions = [
                       yn + (i == 1 ? 0 : sum(a[i,j] * k[j] for j ∈ 1:i-1))
                       for i ∈ 1:n_rk_stages
                      ]
    # Final entry of y_k_expressions is y^(n+1)
    push!(y_k_expressions, yn + sum(b[1,i] * k[i] for i ∈ 1:n_rk_stages))

    if adaptive
        y_err = sum((b[2,i] - b[1,i]) * k[i] for i ∈ 1:n_rk_stages)
    end

    # Define expressions for y[i] using the rk_coefs as used in moment_kinetics
    y_rk_coefs_expressions = [
                              sum(rk_coefs[j,i-1] * y[j] for j ∈ 1:i-1) + rk_coefs[i,i-1] * (y[i-1] + k[i-1])
                              for i ∈ 2:(n_rk_stages + 1)
                             ]
    # Substitute to eliminate y[i] from the expressions
    y_rk_coefs_expressions = [
                              substitute(e, Dict(y[i] => y_k_expressions[i] for i ∈ 1:n_rk_stages))
                              for e ∈ y_rk_coefs_expressions
                             ]
    if adaptive
        y_rk_coefs_err = sum(rk_coefs[j,n_rk_stages+1] * y[j] for j ∈ 1:n_rk_stages+1)
        y_rk_coefs_err = substitute(y_rk_coefs_err, Dict(y[i] => y_k_expressions[i] for i ∈ 1:n_rk_stages+1))
    end

    # Construct equations that can be solved for rk_coefs entries by equating the
    # coefficients of each k[i] in the two sets of expressions
    rk_coefs_equations = []
    for (i, (rk_coefs_expr, Butcher_expr)) ∈ enumerate(zip(y_rk_coefs_expressions, y_k_expressions[2:end]))
        lhs = Symbolics.coeff(rk_coefs_expr, yn)
        rhs = Symbolics.coeff(Butcher_expr, yn)
        if isa(lhs, Number) && lhs == 0 && isa(rhs, Number) && rhs == 0
            push!(rk_coefs_equations, rk_coefs[1,i] ~ 0)
        else
            push!(rk_coefs_equations, lhs ~ rhs)
        end
        for j ∈ 1:n_rk_stages
            lhs = Symbolics.coeff(rk_coefs_expr, k[j])
            rhs = Symbolics.coeff(Butcher_expr, k[j])
            if isa(lhs, Number) && lhs == 0 && isa(rhs, Number) && rhs == 0
                push!(rk_coefs_equations, rk_coefs[j+1,i] ~ 0)
            else
                push!(rk_coefs_equations, lhs ~ rhs + 0)
            end
        end
    end
    if adaptive
        i = n_rk_stages + 1
        lhs = Symbolics.coeff(y_rk_coefs_err, yn)
        rhs = Symbolics.coeff(y_err, yn)
        if isa(lhs, Number) && lhs == 0 && isa(rhs, Number) && rhs == 0
            push!(rk_coefs_equations, rk_coefs[1,i] ~ 0)
        else
            push!(rk_coefs_equations, lhs ~ rhs)
        end
        for j ∈ 1:n_rk_stages
            lhs = Symbolics.coeff(y_rk_coefs_err, k[j])
            rhs = Symbolics.coeff(y_err, k[j])
            if isa(lhs, Number) && lhs == 0 && isa(rhs, Number) && rhs == 0
                push!(rk_coefs_equations, rk_coefs[j+1,i] ~ 0)
            else
                push!(rk_coefs_equations, lhs ~ rhs + 0)
            end
        end
    end

    # Solve rk_coefs_equations for the rk_coefs entries
    if using_rationals
        rk_coefs_values = my_solve_for(rk_coefs_equations, [rk_coefs...])
    else
        rk_coefs_values = Symbolics.solve_for(rk_coefs_equations, [rk_coefs...])
    end
    rk_coefs_values = reshape(rk_coefs_values, n_rk_stages+1, output_size)

    if low_storage
        if using_rationals
            rk_coefs_out = zeros(Rational{Int64}, 3, output_size)
        else
            rk_coefs_out = zeros(3, output_size)
        end
        for i in 1:n_rk_stages
            if i == 1
                j = i
                rk_coefs_out[1,i] = rk_coefs_values[1,i]
                rk_coefs_out[3,i] = rk_coefs_values[2,i]
                for j ∈ 3:n_rk_stages+1
                    if rk_coefs_values[j,i] != 0
                        error("Found non-zero coefficient where zero was expected for low-storage coefficients")
                    end
                end
            else
                j = 1
                rk_coefs_out[1,i] = rk_coefs_values[1,i]
                for j ∈ 2:i-1
                    if rk_coefs_values[j,i] != 0
                        error("Found non-zero coefficient where zero was expected for low-storage coefficients")
                    end
                end
                rk_coefs_out[2,i] = rk_coefs_values[i,i]
                rk_coefs_out[3,i] = rk_coefs_values[i+1,i]
                for j ∈ i+2:n_rk_stages+1
                    if rk_coefs_values[j,i] != 0
                        error("Found non-zero coefficient where zero was expected for low-storage coefficients")
                    end
                end
            end
        end
        if adaptive
            i = n_rk_stages+1
            j = 1
            rk_coefs_out[1,i] = rk_coefs_values[1,i]
            for j ∈ 2:i-2
                if rk_coefs_values[j,i] != 0
                    error("Found non-zero coefficient where zero was expected for low-storage coefficients")
                end
            end
            rk_coefs_out[2,i] = rk_coefs_values[i-1,i]
            rk_coefs_out[3,i] = rk_coefs_values[i,i]
        end
    else
        rk_coefs_out = rk_coefs_values
    end

    return rk_coefs_out
end
function convert_butcher_tableau_for_moment_kinetics(a::Matrix{Rational{Int64}},
                                                     b::Matrix{Rational{Int64}};
                                                     low_storage=true)
    a = Matrix{Rational{BigInt}}(a)
    b = Matrix{Rational{BigInt}}(b)
    return convert_butcher_tableau_for_moment_kinetics(a, b; low_storage=low_storage)
end

function convert_rk_coefs_to_butcher_tableau(rk_coefs::AbstractArray{T,N}) where {T,N}
    using_rationals = eltype(rk_coefs) <: Rational
    adaptive = (abs(sum(rk_coefs[:,end])) < 1.0e-13)
    low_storage = size(rk_coefs, 1) == 3
    if adaptive
        n_rk_stages = size(rk_coefs, 2) - 1
    else
        n_rk_stages = size(rk_coefs, 2)
    end

    @variables y[1:n_rk_stages+1] yn k[1:n_rk_stages]
    y = Symbolics.scalarize(y)
    k = Symbolics.scalarize(k)

    if low_storage
        y_expressions = [
                         yn,
                         (rk_coefs[1,i]*y[1] + rk_coefs[2,i]*y[i] + rk_coefs[3,i]*(y[i] + k[i])
                          for i ∈ 1:n_rk_stages)...
                        ]
    else
        y_expressions = [
                         yn,
                         (sum(rk_coefs[j,i]*y[j] for j ∈ 1:i) + rk_coefs[i+1,i]*(y[i] + k[i])
                          for i ∈ 1:n_rk_stages)...
                        ]
    end
    y_expressions = [simplify(expand(e)) for e ∈ y_expressions]
    if adaptive
        i = n_rk_stages + 1
        if low_storage
            y_err = simplify(expand(rk_coefs[1,i]*y[1] + rk_coefs[2,i]*y[n_rk_stages] + rk_coefs[3,i]*y[n_rk_stages+1]))
        else
            y_err = simplify(expand(sum(rk_coefs[j,i]*y[j] for j ∈ 1:i)))
        end
    end

    # Set up equations to solve for each y[i] in terms of k[i]
    y_equations = [y[i] ~ y_expressions[i] for i ∈ 1:n_rk_stages+1]
    if using_rationals
        y_k_expressions = my_solve_for2(y_equations, y)
    else
        y_k_expressions = Symbolics.solve_for(y_equations, y)
    end

    if adaptive
        b = zeros(T, 2, n_rk_stages)
    else
        b = zeros(T, 1, n_rk_stages)
    end

    for j ∈ 1:n_rk_stages
        b[1, j] = Symbolics.coeff(y_k_expressions[n_rk_stages+1], k[j])
    end
    if adaptive
        error_coeffs = zeros(T, n_rk_stages)
        y_k_err = substitute(y_err, Dict(y[i] => y_k_expressions[i] for i ∈ 1:n_rk_stages+1))
        y_k_err = simplify(expand(y_k_err))
        for j ∈ 1:n_rk_stages
            error_coeffs[j] = Symbolics.coeff(y_k_err, k[j])
        end
        @. b[2,:] = error_coeffs + b[1,:]
    end

    a = zeros(T, n_rk_stages, n_rk_stages)
    for i ∈ 1:n_rk_stages
        for j ∈ 1:n_rk_stages
            a[i,j] = Symbolics.coeff(y_k_expressions[i], k[j])
        end
    end

    return a, b
end

function convert_and_check_butcher_tableau(name, a, b; low_storage=true)
    println(name)
    rk_coefs = convert_butcher_tableau_for_moment_kinetics(a, b; low_storage=low_storage)
    print("a="); display(a)
    print("b="); display(b)
    print("rk_coefs="); display(rk_coefs)
    println("a=$a")
    println("b=$b")
    println("rk_coefs=$rk_coefs")
    println()

    check_end = size(rk_coefs, 2)
    if size(b, 1) > 1
        # Adaptive timestep
        if abs(sum(rk_coefs[:,end])) > 1.0e-13
            error("Sum of error coefficients should be 0")
        end
        check_end -= 1
    end
    for i ∈ 1:check_end
        if abs(sum(rk_coefs[:,i]) - 1) > 1.0e-13
            error("Sum of RK coefficients should be 1 for each stage")
        end
    end

    # Consistency check: converting back should give the original a, b.
    a_check, b_check = convert_rk_coefs_to_butcher_tableau(rk_coefs)

    if isa(a[1], Real)
        if maximum(abs.(a_check .- a)) > 1.0e-13
            error("Converting rk_coefs back to Butcher tableau gives different 'a':\n"
                  * "Original: $a\n"
                  * "New:      $a_check")
        end
        if maximum(abs.(b_check .- b)) > 1.0e-13
            error("Converting rk_coefs back to Butcher tableau gives different 'b':\n"
                  * "Original: $b\n"
                  * "New:      $b_check")
        end
    else
        if a_check != a
            error("Converting rk_coefs back to Butcher tableau gives different 'a':\n"
                  * "Original: $a\n"
                  * "New:      $a_check")
        end
        if b_check != b
            error("Converting rk_coefs back to Butcher tableau gives different 'b':\n"
                  * "Original: $b\n"
                  * "New:      $b_check")
        end
    end
end

function convert_and_check_rk_coefs(name, rk_coefs)
    println(name)

    print("rk_coefs="); display(rk_coefs)
    a, b = convert_rk_coefs_to_butcher_tableau(rk_coefs)
    print("a="); display(a)
    print("b="); display(b)
    println("a=$a")
    println("b=$b")
    println()
end

# 'Standard form' of coefficients from
# https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method,
# 'COEFFICIENTS FOR RK4(5), FORMULA 2 Table III in Fehlberg'

# Neglect the 'A' coefficients as we do not have an explicit time dependence in the RHS,
# so do not use them.

# Use `//` to get rational numbers to avoid round-off errors
B = [ 0 0 0 0 0 0;
      1//4 0 0 0 0 0;
      3//32 9//32 0 0 0 0;
      1932//2197 -7200//2197 7296//2197 0 0 0;
      439//216 -8 3680//513 -845//4104 0 0;
      -8//27 2 -3544//2565 1859//4104 -11//40 0]

C = [ 25//216 0 1408//2565 2197//4104 -1//5 0 ]
CH = [ 16//135 0 6656//12825 28561//56430 -9//50 2//55 ]
# The following is the version from Wikipedia, it appears to have a typo in the 4th
# element, as we should have CT=C-CH -- CT = [ -1//360 0 128//4275 2187//75240 -1//50 -2//55 ]
CT = [ -1//360 0 128//4275 2197//75240 -1//50 -2//55 ]

# 'COEFFICIENTS FOR Sarafyan's RK4(5), Table IV in Fehlberg'

# Neglect the 'A' coefficients as we do not have an explicit time dependence in the RHS,
# so do not use them.

# Use `//` to get rational numbers to avoid round-off errors
#B = [ 0 0 0 0 0 0;
#      1//2 0 0 0 0 0;
#      1//4 1//4 0 0 0 0;
#      0 -1 2 0 0 0;
#      7//27 10//27 0 1//27 0 0;
#      28//625 -1//5 546//625 54//625 -378//625 0]
#
#C = [ 1//6 0 2//3 1//6 0 0 ]
#CH = [ 1//24 0 0 5//48 27//56 125//336 ]
#CT = [ 1//8 0 2//3 1//16 -27//56 -125//336 ]
a = B
b = vcat(CH,C)
convert_and_check_butcher_tableau("RKF5(4)", a, b; low_storage=false)

convert_and_check_butcher_tableau(
    "SSPRK3",
    # From https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
    [0 0 0;
     1 0 0;
     1//4 1//4 0],
    [1//6 1//6 2//3],
   )

convert_and_check_butcher_tableau(
    "Heun's method SSPRK2",
    # From https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
    [0    0;
     1//1 0],
    [1//2 1//2],
   )

convert_and_check_butcher_tableau(
    "Gottlieb et al 4-stage 3rd order",
    # From https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
    [0 0 0;
     1 0 0;
     1//2 1//2 0],
    [1//6 1//6 2//3],
   )

convert_and_check_butcher_tableau(
    "RK4",
    # From https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
    [0 0 0 0;
     1//2 0 0 0;
     0 1//2 0 0;
     0 0 1 0],
    [1//6 1//3 1//3 1//6];
    low_storage=false,
   )

#Optimal 4th order strong-stability preserving embedded Runge-Kutta method with 10 stages,
#from [Fekete, Conde and Shadid, "Embedded pairs for optimal explicit strong stability
#preserving Runge-Kutta methods", Journal of Computational and Applied Mathematics 421
#(2022) 114325, https://doi.org/10.1016/j.cam.2022.114325]. This methods is from section
#2.3, with the '\$\\tilde{b}^T_4\$' embedded pair, which is recommended in the conclusions.
convert_and_check_butcher_tableau(
    "Fekete 10(4)",
    [0     0     0     0     0     0    0    0    0    0;
     1//6  0     0     0     0     0    0    0    0    0;
     1//6  1//6  0     0     0     0    0    0    0    0;
     1//6  1//6  1//6  0     0     0    0    0    0    0;
     1//6  1//6  1//6  1//6  0     0    0    0    0    0;
     1//15 1//15 1//15 1//15 1//15 0    0    0    0    0;
     1//15 1//15 1//15 1//15 1//15 1//6 0    0    0    0;
     1//15 1//15 1//15 1//15 1//15 1//6 1//6 0    0    0;
     1//15 1//15 1//15 1//15 1//15 1//6 1//6 1//6 0    0;
     1//15 1//15 1//15 1//15 1//15 1//6 1//6 1//6 1//6 0;
    ],
    [1//10 1//10 1//10 1//10 1//10 1//10 1//10 1//10 1//10 1//10;
     #0     3//8  0     1//8  0     0     0     3//8  0     1//8 ]
     #3//14 0     0     2//7  0     0     0     3//7  0     1//14]
     #0     2//9  0     0     5//18 1//3  0     0     0     1//6 ]
     1//5  0     0     3//10 0     0     1//5  0     3//10 0    ]
     #1//10 0     0     2//5  0     3//10 0     0     0     1//5 ]
     #1//6  0     0     0     1//3  5//18 0     0     2//9  0    ]
     #0     2//5  0     1//10 0     0     0     1//5  3//10 0    ]
     #1//7  0     5//14 0     0     0     0     3//14 2//7 0    ]
    ; low_storage=false)

#6-stage, 4th order strong-stability preserving embedded Runge-Kutta method from [Fekete,
#Conde and Shadid, "Embedded pairs for optimal explicit strong stability preserving
#Runge-Kutta methods", Journal of Computational and Applied Mathematics 421 (2022) 114325,
#https://doi.org/10.1016/j.cam.2022.114325]. This method is from section 2.3. Provided
#because it has fewer stages than the 10-stage 4th-order method, but not recommended by
#Fekete et al.
convert_and_check_butcher_tableau(
    "Fekete 6(4)",
    [0               0               0               0               0               0;
     0.3552975516919 0               0               0               0               0;
     0.2704882223931 0.3317866983600 0               0               0               0;
     0.1223997401356 0.1501381660925 0.1972127376054 0               0               0;
     0.0763425067155 0.0936433683640 0.1230044665810 0.2718245927242 0               0;
     0.0763425067155 0.0936433683640 0.1230044665810 0.2718245927242 0.4358156542577 0;
    ],
    [0.1522491819555 0.1867521364225 0.1555370561501 0.1348455085546 0.2161974490441 0.1544186678729;
     0.1210663237182 0.2308844004550 0.0853424972752 0.3450614904457 0.0305351538213 0.1871101342844];
    low_storage=false)

"""
    construct_fekete_3rd_order(nstage)

Construct optimal 3rd order strong-stability preserving embedded Runge-Kutta method with
`nstage` stages, from [Fekete, Conde and Shadid, "Embedded pairs for optimal explicit
strong stability preserving Runge-Kutta methods", Journal of Computational and Applied
Mathematics 421 (2022) 114325, https://doi.org/10.1016/j.cam.2022.114325]. These methods
are from section 2.2, with the 'Optimization (10)' embedded pair, which is recommended in
the conclusions.
"""
function construct_fekete_3rd_order(nstage)
    n = floor(Int64, sqrt(nstage))
    if n^2 != nstage
        error("nstage must be a square, got ", nstage)
    end
    a = zeros(Rational{Int64}, nstage, nstage)
    sub_rectangle_height = (n*(n-1))÷2
    for i ∈ 2:(nstage - sub_rectangle_height)
        for j ∈ 1:i-1
            a[i,j] = 1//(n*(n-1))
        end
    end
    for i ∈ (nstage - sub_rectangle_height)+1:nstage
        for j ∈ 1:((n-2)*(n-1))÷2
            a[i,j] = 1//(n*(n-1))
        end
        for j ∈ ((n-2)*(n-1))÷2+1:((n-2)*(n-1))÷2+(2*n-1)
            a[i,j] = 1//(n*(2*n-1))
        end
        for j ∈ ((n-2)*(n-1))÷2+(2*n-1)+1:i-1
            a[i,j] = 1//(n*(n-1))
        end
    end

    b = zeros(Rational{Int64}, 2, nstage)

    b[1,:] .= 1//(n*(n-1))
    b[1, ((n-1)*(n-2))÷2+1:((n-1)*(n-2))÷2+(2*n-1)] .= 1//(n*(2*n-1))

    # 'Pair' from 'optimization 10'
    b[2, :] .= 1//n^2

    return a, b
end

convert_and_check_butcher_tableau(
    "Fekete 4(3)",
    construct_fekete_3rd_order(4)...
   )

"""
    construct_fekete_2nd_order(nstage)

Construct optimal 2nd order strong-stability preserving embedded Runge-Kutta method with
`nstage` stages, from [Fekete, Conde and Shadid, "Embedded pairs for optimal explicit
strong stability preserving Runge-Kutta methods", Journal of Computational and Applied
Mathematics 421 (2022) 114325, https://doi.org/10.1016/j.cam.2022.114325]. These methods
are from section 2.1, with the 'Optimization (10)' embedded pair, which is recommended in
the conclusions.
"""
function construct_fekete_2nd_order(nstage)
    a = zeros(Rational{Int64}, nstage, nstage)
    for i ∈ 2:nstage
        for j ∈ 1:i-1
            a[i,j] = 1//(nstage - 1)
        end
    end

    b = zeros(Rational{Int64}, 2, nstage)

    b[1,:] .= 1//nstage

    # 'Pair' from 'optimization 10'
    b[2, 1] = (nstage + 1) // nstage^2
    b[2, 2:end-1] .= 1 // nstage
    b[2, end] = (nstage - 1) // nstage^2

    return a, b
end

convert_and_check_butcher_tableau(
    "Fekete 4(2)",
    construct_fekete_2nd_order(4)...;
    low_storage=false,
   )

convert_and_check_butcher_tableau(
    "Fekete 3(2)",
    construct_fekete_2nd_order(3)...;
    low_storage=false,
   )

convert_and_check_butcher_tableau(
    "Fekete 2(2)",
    construct_fekete_2nd_order(2)...
   )

convert_and_check_rk_coefs(
    "mk's ssprk4",
    [1//2 0    2//3 0   ;
     1//2 1//2 0    0   ;
     0    1//2 1//6 0   ;
     0    0    1//6 1//2;
     0    0    0    1//2],
   )

convert_and_check_rk_coefs(
    "mk's ssprk3",
    [0  3//4 1//3;
     1  0    0   ;
     0  1//4 0   ;
     0  0    2//3],
   )

convert_and_check_rk_coefs(
    "mk's ssprk2",
    [0 1//2;
     0 0   ;
     1 1//2],
   )
