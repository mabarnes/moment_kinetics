"""
We implement the Runge-Kutta timestepper in `moment_kinetics` in a different form from the
most conventional one, so in some cases we need to convert the 'conventional' coefficients
into ones that we can use.
"""
module CalculateRKCoeffs

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
function convert_butcher_tableau_for_moment_kinetics(a, b,
                                                     a_implicit=zeros(size(a)),
                                                     b_implicit=zeros(size(b));
                                                     low_storage=true)
    using_rationals = eltype(a) <: Rational || eltype(b) <: Rational || eltype(a_implicit) <: Rational || eltype(b_implicit) <: Rational
    imex = any(a_implicit .!= 0)
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
    @variables y_tilde[1:n_rk_stages+1] k[1:n_rk_stages] yn rk_coefs[1:n_rk_stages+1, 1:output_size]
    @variables y[1:n_rk_stages] k_implicit[1:n_rk_stages] rk_coefs_implicit[1:n_rk_stages, 1:output_size+1]
    y_tilde = Symbolics.scalarize(y_tilde)
    k = Symbolics.scalarize(k)
    rk_coefs = Symbolics.scalarize(rk_coefs)
    y = Symbolics.scalarize(y)
    k_implicit = Symbolics.scalarize(k_implicit)
    rk_coefs_implicit = Symbolics.scalarize(rk_coefs_implicit)

    # Expressions defined using the 'standard' Butcher formulae
    y_tilde_k_expressions = [
                             yn + (i == 1 ? 0 : sum(a[i,j] * k[j] for j ∈ 1:i-1) + sum(a_implicit[i,j] * k_implicit[j] for j ∈ 1:i-1))
                             for i ∈ 1:n_rk_stages
                            ]
    # Note that when using an IMEX scheme, if a_implicit[i,i]==0, then k_implicit[i] is
    # actually an explicit RHS evaluation (evaluated using y_tilde[i]), and the explicit
    # RHS k[i] will be evaluated using y_tilde[i] instead of y[i] so that we can store
    # (y_tilde[i] + k_implicit[i]) in y[i], as a way to have k_implicit[i] available.
    implicit_coefficient_is_zero = [imex && a_implicit[i,i] == 0 for i ∈ 1:n_rk_stages]
    y_k_expressions = [
                       y_tilde_k_expressions[i] + (implicit_coefficient_is_zero[i] ? 1 : a_implicit[i,i]) * k_implicit[i]
                       for i ∈ 1:n_rk_stages
                      ]
    # Final entry of y_k_expressions is y^(n+1)
    push!(y_tilde_k_expressions, yn +
                                 sum(b[1,i] * k[i] for i ∈ 1:n_rk_stages) +
                                 sum(b_implicit[1,i] * k_implicit[i] for i ∈ 1:n_rk_stages))

    if adaptive
        y_loworder = yn +
                     sum(b[2,i] * k[i] for i ∈ 1:n_rk_stages) +
                     sum(b_implicit[2,i] * k_implicit[i] for i ∈ 1:n_rk_stages)
    end

    # Define expressions for y_tilde[i] using the rk_coefs as used in moment_kinetics
    # Note that we need a special case for an imex scheme with some a[i,i]=0, as for those
    # entries we hacked y[i] to allow k_implicit[i] to be saved, and we need to use
    # y_tilde[i] as the starting point for the forward-Euler derivative instead of y[i].
    y_tilde_rk_coefs_expressions = [
                                    yn, # i=1
                                    (sum(rk_coefs[j,i-1] * y_tilde[j] for j ∈ 1:i-1)
                                     + rk_coefs[i,i-1] * ((implicit_coefficient_is_zero[i-1] ? y_tilde[i-1] : y[i-1]) + k[i-1]) +
                                     sum(rk_coefs_implicit[j,i] * y[j] for j ∈ 1:i-1)
                                     for i ∈ 2:n_rk_stages+1)...
                                   ]
    # Note the 'implicit step' is treated specially, as the coefficient will be used to
    # scale the timestep in the code, rather than as the coefficient of some version of
    # y/y_tilde. rk_coefs_implicit[i,i] should end up being equal to a_implicit[i,i].
    y_rk_coefs_expressions = [
                              e + rk_coefs_implicit[i,i] * k_implicit[i]
                              for (i,e) ∈ enumerate(y_tilde_rk_coefs_expressions[1:n_rk_stages])
                             ]

    # Substitute to eliminate y_tilde[i] from the expressions
    y_tilde_rk_coefs_expressions = [
                                    substitute(e, Dict(y_tilde[i] => y_tilde_k_expressions[i] for i ∈ 1:n_rk_stages+1))
                                    for e ∈ y_tilde_rk_coefs_expressions
                                   ]
    y_rk_coefs_expressions = [
                              substitute(e, Dict(y_tilde[i] => y_tilde_k_expressions[i] for i ∈ 1:n_rk_stages+1))
                              for e ∈ y_rk_coefs_expressions
                             ]


    # Substitute to eliminate y[i] from the expressions
    y_tilde_rk_coefs_expressions = [
                                    substitute(e, Dict(y[i] => y_k_expressions[i] for i ∈ 1:n_rk_stages))
                                    for e ∈ y_tilde_rk_coefs_expressions
                                   ]
    y_rk_coefs_expressions = [
                              substitute(e, Dict(y[i] => y_k_expressions[i] for i ∈ 1:n_rk_stages))
                              for e ∈ y_rk_coefs_expressions
                             ]

    if adaptive
        y_rk_coefs_err = sum(rk_coefs[j,n_rk_stages+1] * y_tilde[j] for j ∈ 1:n_rk_stages+1) +
                         sum(rk_coefs_implicit[j,n_rk_stages+2] * y[j] for j ∈ 1:n_rk_stages)
        y_rk_coefs_err = substitute(y_rk_coefs_err, Dict(y_tilde[i] => y_tilde_k_expressions[i] for i ∈ 1:n_rk_stages+1))
        y_rk_coefs_err = substitute(y_rk_coefs_err, Dict(y[i] => y_k_expressions[i] for i ∈ 1:n_rk_stages))
    end

    # Construct equations that can be solved for rk_coefs entries by equating the
    # coefficients of each k[i], k_implicit[i] in the two sets of expressions
    rk_coefs_equations = []
    for (i, (rk_coefs_expr, Butcher_expr)) ∈ enumerate(zip(y_rk_coefs_expressions, y_k_expressions))
        for j ∈ 1:n_rk_stages
            lhs = Symbolics.coeff(rk_coefs_expr, k_implicit[j])
            rhs = Symbolics.coeff(Butcher_expr, k_implicit[j])
            if isa(lhs, Number) && lhs == 0 && isa(rhs, Number) && rhs == 0
                push!(rk_coefs_equations, rk_coefs_implicit[j,i] ~ 0)
            else
                push!(rk_coefs_equations, lhs ~ rhs + 0)
            end
        end
        if i == 1
            # EXplicit RK coefficients have no entries for i=1, because y_tilde[1]=yn
            # always.
            continue
        end
        lhs = Symbolics.coeff(rk_coefs_expr, yn)
        rhs = Symbolics.coeff(Butcher_expr, yn)
        if isa(lhs, Number) && lhs == 0 && isa(rhs, Number) && rhs == 0
            push!(rk_coefs_equations, rk_coefs[1,i-1] ~ 0)
        else
            push!(rk_coefs_equations, lhs ~ rhs)
        end
        for j ∈ 1:n_rk_stages
            lhs = Symbolics.coeff(rk_coefs_expr, k[j])
            rhs = Symbolics.coeff(Butcher_expr, k[j])
            if isa(lhs, Number) && lhs == 0 && isa(rhs, Number) && rhs == 0
                push!(rk_coefs_equations, rk_coefs[j+1,i-1] ~ 0)
            else
                push!(rk_coefs_equations, lhs ~ rhs + 0)
            end
        end
    end

    # Include contribution from y_tilde[n_rk_stages+1]
    i = n_rk_stages + 1
    rk_coefs_expr = y_tilde_rk_coefs_expressions[n_rk_stages+1]
    Butcher_expr = y_tilde_k_expressions[n_rk_stages+1]
    for j ∈ 1:n_rk_stages
        lhs = Symbolics.coeff(rk_coefs_expr, k_implicit[j])
        rhs = Symbolics.coeff(Butcher_expr, k_implicit[j])
        if isa(lhs, Number) && lhs == 0 && isa(rhs, Number) && rhs == 0
            push!(rk_coefs_equations, rk_coefs_implicit[j,i] ~ 0)
        else
            push!(rk_coefs_equations, lhs ~ rhs + 0)
        end
    end
    lhs = Symbolics.coeff(rk_coefs_expr, yn)
    rhs = Symbolics.coeff(Butcher_expr, yn)
    if isa(lhs, Number) && lhs == 0 && isa(rhs, Number) && rhs == 0
        push!(rk_coefs_equations, rk_coefs[1,i-1] ~ 0)
    else
        push!(rk_coefs_equations, lhs ~ rhs)
    end
    for j ∈ 1:n_rk_stages
        lhs = Symbolics.coeff(rk_coefs_expr, k[j])
        rhs = Symbolics.coeff(Butcher_expr, k[j])
        if isa(lhs, Number) && lhs == 0 && isa(rhs, Number) && rhs == 0
            push!(rk_coefs_equations, rk_coefs[j+1,i-1] ~ 0)
        else
            push!(rk_coefs_equations, lhs ~ rhs + 0)
        end
    end

    if adaptive
        i = n_rk_stages + 1
        lhs = Symbolics.coeff(y_rk_coefs_err, yn)
        rhs = Symbolics.coeff(y_loworder, yn)
        if isa(lhs, Number) && lhs == 0 && isa(rhs, Number) && rhs == 0
            push!(rk_coefs_equations, rk_coefs[1,i] ~ 0)
        else
            push!(rk_coefs_equations, lhs ~ rhs)
        end
        for j ∈ 1:n_rk_stages
            lhs = Symbolics.coeff(y_rk_coefs_err, k[j])
            rhs = Symbolics.coeff(y_loworder, k[j])
            if isa(lhs, Number) && lhs == 0 && isa(rhs, Number) && rhs == 0
                push!(rk_coefs_equations, rk_coefs[j+1,i] ~ 0)
            else
                push!(rk_coefs_equations, lhs ~ rhs + 0)
            end
        end
        i = n_rk_stages + 2
        for j ∈ 1:n_rk_stages
            lhs = Symbolics.coeff(y_rk_coefs_err, k_implicit[j])
            rhs = Symbolics.coeff(y_loworder, k_implicit[j])
            if isa(lhs, Number) && lhs == 0 && isa(rhs, Number) && rhs == 0
                push!(rk_coefs_equations, rk_coefs_implicit[j,i] ~ 0)
            else
                push!(rk_coefs_equations, lhs ~ rhs + 0)
            end
        end
    end

    # Solve rk_coefs_equations for the rk_coefs entries
    if using_rationals
        rk_coefs_values = my_solve_for(rk_coefs_equations, [rk_coefs..., rk_coefs_implicit...])
    else
        rk_coefs_values = Symbolics.symbolic_linear_solve(rk_coefs_equations, [rk_coefs..., rk_coefs_implicit...])
    end
    rk_coefs_implicit_values = reshape(rk_coefs_values[(n_rk_stages+1)*output_size+1:end], n_rk_stages, output_size+1)
    rk_coefs_values = reshape(rk_coefs_values[1:(n_rk_stages+1)*output_size], n_rk_stages+1, output_size)

    if low_storage
        if using_rationals
            rk_coefs_out = zeros(Rational{Int64}, 3, output_size)
            rk_coefs_implicit_out = zeros(Rational{Int64}, 3, output_size+1)
        else
            rk_coefs_out = zeros(3, output_size)
            rk_coefs_implicit_out = zeros(3, output_size+1)
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
        for i in 1:n_rk_stages
            if i == 1
                j = i
                rk_coefs_implicit_out[1,i] = rk_coefs_implicit_values[1,i]
                rk_coefs_implicit_out[3,i] = rk_coefs_implicit_values[2,i]
                for j ∈ 3:n_rk_stages
                    if rk_coefs_implicit_values[j,i] != 0
                        error("Found non-zero coefficient where zero was expected for low-storage coefficients")
                    end
                end
            else
                j = 1
                rk_coefs_implicit_out[1,i] = rk_coefs_implicit_values[1,i]
                for j ∈ 2:i-1
                    if rk_coefs_implicit_values[j,i] != 0
                        error("Found non-zero coefficient where zero was expected for low-storage coefficients")
                    end
                end
                rk_coefs_implicit_out[2,i] = rk_coefs_implicit_values[i,i]
                if i == n_rk_stages
                    rk_coefs_implicit_out[3,i] = 0
                else
                    rk_coefs_implicit_out[3,i] = rk_coefs_implicit_values[i+1,i]
                end
                for j ∈ i+2:n_rk_stages
                    if rk_coefs_implicit_values[j,i] != 0
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

            j = 1
            rk_coefs_implicit_out[1,i] = rk_coefs_implicit_values[1,i]
            for j ∈ 2:i-2
                if rk_coefs_implicit_values[j,i] != 0
                    error("Found non-zero coefficient where zero was expected for low-storage coefficients")
                end
            end
            j = n_rk_stages
            rk_coefs_implicit_out[2,i] = rk_coefs_implicit_values[j,i]
            rk_coefs_implicit_out[3,i] = 0 #rk_coefs_implicit_values[j+1,i]
        end
    else
        rk_coefs_out = rk_coefs_values
        rk_coefs_implicit_out = rk_coefs_implicit_values
    end

    return rk_coefs_out, rk_coefs_implicit_out, implicit_coefficient_is_zero
end
function convert_butcher_tableau_for_moment_kinetics(a::Matrix{Rational{Int64}},
                                                     b::Matrix{Rational{Int64}},
                                                     a_implicit::Matrix{Rational{Int64}}=zeros(Rational{Int64}, size(a)),
                                                     b_implicit::Matrix{Rational{Int64}}=zeros(Rational{Int64}, size(b));
                                                     low_storage=true)
    a = Matrix{Rational{BigInt}}(a)
    b = Matrix{Rational{BigInt}}(b)
    a_implicit = Matrix{Rational{BigInt}}(a_implicit)
    b_implicit = Matrix{Rational{BigInt}}(b_implicit)
    return convert_butcher_tableau_for_moment_kinetics(a, b, a_implicit, b_implicit;
                                                       low_storage=low_storage)
end

function convert_rk_coefs_to_butcher_tableau(rk_coefs::AbstractArray{T,N},
                                             adaptive, low_storage,
                                             rk_coefs_implicit=zeros(T, size(rk_coefs, 1) - 1, size(rk_coefs, 2) + 1),
                                             implicit_coefficient_is_zero=nothing
                                            ) where {T,N}
    using_rationals = eltype(rk_coefs) <: Rational || eltype(rk_coefs_implicit) <: Rational
    if adaptive
        n_rk_stages = size(rk_coefs, 2) - 1
    else
        n_rk_stages = size(rk_coefs, 2)
    end
    if implicit_coefficient_is_zero === nothing
        implicit_coefficient_is_zero = zeros(Bool, n_rk_stages)
    end

    @variables y_tilde[1:n_rk_stages+1] yn k[1:n_rk_stages]
    y_tilde = Symbolics.scalarize(y_tilde)
    k = Symbolics.scalarize(k)
    @variables y[1:n_rk_stages] k_implicit[1:n_rk_stages]
    y = Symbolics.scalarize(y)
    k_implicit = Symbolics.scalarize(k_implicit)

    if low_storage
        y_tilde_expressions = [
                               yn,
                               (rk_coefs[1,i-1]*y_tilde[1] + rk_coefs[2,i-1]*y_tilde[i-1]
                                + rk_coefs[3,i-1]*((implicit_coefficient_is_zero[i-1] ? y_tilde[i-1] : y[i-1]) + k[i-1])
                                + rk_coefs_implicit[1,i]*y[1] + rk_coefs_implicit[2,i]*y[i-1]
                                for i ∈ 2:n_rk_stages+1)...
                              ]
        y_expressions = [
                         y_tilde_expressions[i] + rk_coefs_implicit[3,i] * k_implicit[i]
                         for i ∈ 1:n_rk_stages
                        ]
    else
        y_tilde_expressions = [
                               yn,
                               (sum(rk_coefs[j,i-1]*y_tilde[j] for j ∈ 1:i-1)
                                + rk_coefs[i,i-1]*((implicit_coefficient_is_zero[i-1] ? y_tilde[i-1] : y[i-1]) + k[i-1])
                                + sum(rk_coefs_implicit[j,i]*y[j] for j ∈ 1:i-1)
                                for i ∈ 2:n_rk_stages+1)...
                              ]
        y_expressions = [
                         y_tilde_expressions[i] + rk_coefs_implicit[i,i] * k_implicit[i]
                         for i ∈ 1:n_rk_stages
                        ]
    end
    y_tilde_expressions = [simplify(expand(e)) for e ∈ y_tilde_expressions]
    y_expressions = [simplify(expand(e)) for e ∈ y_expressions]
    if adaptive
        if low_storage
            i = n_rk_stages + 1
            y_loworder = rk_coefs[1,i]*y_tilde[1] + rk_coefs[2,i]*y_tilde[n_rk_stages] + rk_coefs[3,i]*y_tilde[n_rk_stages+1] +
                         rk_coefs_implicit[1,i+1]*y[1] + rk_coefs_implicit[2,i+1]*y[n_rk_stages-1] + rk_coefs_implicit[3,i+1]*y[n_rk_stages]
        else
            y_loworder = sum(rk_coefs[j,n_rk_stages+1]*y_tilde[j] for j ∈ 1:n_rk_stages+1) +
                         sum(rk_coefs_implicit[j,n_rk_stages+2]*y[j] for j ∈ 1:n_rk_stages)
        end
        y_loworder = simplify(expand(y_loworder))
    end

    # Set up equations to solve for each y_tilde[i] and y[i] in terms of k[i] and
    # k_impliti[i]
    y_tilde_equations = [y_tilde[i] ~ y_tilde_expressions[i] for i ∈ 1:n_rk_stages+1]
    y_equations = [y[i] ~ y_expressions[i] for i ∈ 1:n_rk_stages]
    equations = vcat(y_tilde_equations, y_equations)
    if using_rationals
        expressions = my_solve_for2(equations, vcat(y_tilde, y))
    else
        expressions = Symbolics.symbolic_linear_solve(equations, vcat(y_tilde, y))
    end
    y_tilde_k_expressions = expressions[1:n_rk_stages+1]
    y_k_expressions = expressions[n_rk_stages+2:end]

    if adaptive
        b = zeros(T, 2, n_rk_stages)
        b_implicit = zeros(T, 2, n_rk_stages)
    else
        b = zeros(T, 1, n_rk_stages)
        b_implicit = zeros(T, 1, n_rk_stages)
    end

    for j ∈ 1:n_rk_stages
        b[1, j] = Symbolics.coeff(y_tilde_k_expressions[n_rk_stages+1], k[j])
        b_implicit[1, j] = Symbolics.coeff(y_tilde_k_expressions[n_rk_stages+1], k_implicit[j])
    end
    if adaptive
        y_k_loworder = substitute(y_loworder, Dict(y_tilde[i] => y_tilde_k_expressions[i] for i ∈ 1:n_rk_stages+1))
        y_k_loworder = substitute(y_k_loworder, Dict(y[i] => y_k_expressions[i] for i ∈ 1:n_rk_stages))
        y_k_loworder = simplify(expand(y_k_loworder))
        for j ∈ 1:n_rk_stages
            b[2,j] = Symbolics.coeff(y_k_loworder, k[j])
            b_implicit[2,j] = Symbolics.coeff(y_k_loworder, k_implicit[j])
        end
    end

    a = zeros(T, n_rk_stages, n_rk_stages)
    a_implicit = zeros(T, n_rk_stages, n_rk_stages)
    for i ∈ 1:n_rk_stages
        for j ∈ 1:n_rk_stages
            a[i,j] = Symbolics.coeff(y_k_expressions[i], k[j])
            if j == i && implicit_coefficient_is_zero[i]
                a_implicit[i,j] = 0
            else
                a_implicit[i,j] = Symbolics.coeff(y_k_expressions[i], k_implicit[j])
            end
        end
    end

    return a, b, a_implicit, b_implicit
end

function convert_and_check_butcher_tableau(name, a, b,
                                           a_implicit=zeros(eltype(a), size(a)),
                                           b_implicit=zeros(eltype(b), size(b));
                                           low_storage=true)
    imex = any(a_implicit .!= 0) || any(b_implicit .!= 0)

    println(name)
    rk_coefs, rk_coefs_implicit, implicit_coefficient_is_zero =
        convert_butcher_tableau_for_moment_kinetics(a, b, a_implicit, b_implicit;
                                                    low_storage=low_storage)
    print("a="); display(a)
    print("b="); display(b)
    if imex
        print("a_implicit="); display(a_implicit)
        print("b_implicit="); display(b_implicit)
    end
    print("rk_coefs="); display(rk_coefs)
    if imex
        print("rk_coefs_implicit="); display(rk_coefs_implicit)
    end
    print("rk_coefs(Float64)="); display(Float64.(rk_coefs))
    if imex
        print("rk_coefs_implicit(Float64)="); display(Float64.(rk_coefs_implicit))
    end
    println("a=$a")
    println("b=$b")
    if imex
        println("a_implicit=$a_implicit")
        println("b_implicit=$b_implicit")
    end
    println("rk_coefs=$rk_coefs")
    if imex
        println("rk_coefs_implicit=$rk_coefs_implicit")
        println("implicit_coefficient_is_zero=$implicit_coefficient_is_zero")
    end
    println()

    check_end = size(rk_coefs, 2)
    if size(b, 1) > 1
        # Adaptive timestep
        error_sum = sum(rk_coefs[:,end]) + sum(rk_coefs_implicit[:,end])
        if abs(error_sum - 1) > 1.0e-13
            error("Sum of loworder coefficients should be 1. Got ", error_sum, " ≈ ", Float64(error_sum))
        end
        check_end -= 1
        adaptive = true
    else
        adaptive = false
    end
    for i ∈ 1:check_end
        if low_storage
            error_sum = sum(rk_coefs[:,i]) + sum(rk_coefs_implicit[:,i+1])
        else
            error_sum = sum(rk_coefs[:,i]) + sum(rk_coefs_implicit[1:i,i+1])
        end
        if abs(error_sum - 1) > 1.0e-13
            error("Sum of RK coefficients should be 1 for each stage. Got ", error_sum, " ≈ ", Float64(error_sum))
        end
    end
    if imex
        check_end_implicit = size(rk_coefs_implicit, 2)
        if size(b_implicit, 1) > 1
            # Adaptive timestep
            check_end_implicit -= 1
        end
        for i ∈ 1:check_end_implicit - 1
            if !all(abs.(rk_coefs_implicit[i+1:end,i]) .< 1.0e-13)
                error("Implicit RK coefficients should be 0 for j>i. Got ", rk_coefs_implicit[i+1:end,i], " ≈ ", Float64.(rk_coefs_implicit[i+1:end,i]))
            end
        end
        for i ∈ 1:check_end_implicit - 1
            if a_implicit[i,i] == 0
                if rk_coefs_implicit[i,i] != 1
                    error("Diagonal RK coefficient should be 1 when a_implicit[$i,$i]=0, got rk_coefs_implicit[$i,$i]=", rk_coefs_implicit[i,i])
                end
            elseif abs(rk_coefs_implicit[i,i] - a_implicit[i,i]) > 1.0e-13
                error("Diagonal RK coefficient should be equal to a_implicit[i,i] for each stage. Got rk_coefs_implicit[$i,$i]=", rk_coefs_implicit[i,i] - a_implicit[i,i], " a_implicit[$i,$i]=", a_implicit[i,i])
            end
        end
    end

    # Consistency check: converting back should give the original a, b.
    a_check, b_check, a_check_implicit, b_check_implicit =
        convert_rk_coefs_to_butcher_tableau(rk_coefs, adaptive, low_storage, rk_coefs_implicit, implicit_coefficient_is_zero)

    if eltype(a) == Rational
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
    else
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
    end
    if eltype(a_implicit) == Rational
        if a_check_implicit != a_implicit
            error("Converting rk_coefs back to Butcher tableau gives different 'a_implicit':\n"
                  * "Original: $a_implicit\n"
                  * "New:      $a_check_implicit")
        end
        if b_check_implicit != b_implicit
            error("Converting rk_coefs back to Butcher tableau gives different 'b_implicit':\n"
                  * "Original: $b_implicit\n"
                  * "New:      $b_check_implicit")
        end
    else
        if maximum(abs.(a_check_implicit .- a_implicit)) > 1.0e-13
            error("Converting rk_coefs back to Butcher tableau gives different 'a_implicit':\n"
                  * "Original: $a_implicit\n"
                  * "New:      $a_check_implicit")
        end
        if maximum(abs.(b_check_implicit .- b_implicit)) > 1.0e-13
            error("Converting rk_coefs back to Butcher tableau gives different 'b_implicit':\n"
                  * "Original: $b_implicit\n"
                  * "New:      $b_check_implicit")
        end
    end
end

function convert_and_check_rk_coefs(name, rk_coefs, adaptive=false, low_storage=true,
                                    rk_coefs_implicit=zeros(eltype(rk_coefs),
                                                            size(rk_coefs, 1),
                                                            size(rk_coefs, 2) + 1),
                                    implicit_coefficient_is_zero=nothing)
    imex = any(rk_coefs_implicit .!= 0)

    println(name)

    print("rk_coefs="); display(rk_coefs)
    if imex
        print("rk_coefs_implicit="); display(rk_coefs_implicit)
    end
    a, b, a_implicit, b_implicit = convert_rk_coefs_to_butcher_tableau(rk_coefs, adaptive, low_storage, rk_coefs_implicit, implicit_coefficient_is_zero)
    print("a="); display(a)
    print("b="); display(b)
    if imex
        print("a_implicit="); display(a_implicit)
        print("b_implicit="); display(b_implicit)
    end
    println("a=$a")
    println("b=$b")
    if imex
        println("a_implicit=$a_implicit")
        println("b_implicit=$b_implicit")
    end
    println()
end

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

function calculate_all_coeffs()
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

    convert_and_check_butcher_tableau(
        "Fekete 4(3)",
        construct_fekete_3rd_order(4)...
       )

    convert_and_check_butcher_tableau(
        "Fekete 4(3) not low-storage",
        construct_fekete_3rd_order(4)...;
        low_storage=false
       )

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

    println("\n\nIMEX methods\n============\n")

    # 4th-order, 7-stage IMEX method 'ARK4(3)7L[2]SA₁' from Kennedy & Carpenter 2019
    # (https://doi.org/10.1016/j.apnum.2018.10.007)
    convert_and_check_butcher_tableau(
        "KennedyCarpenterARK437",
        Rational{BigInt}[0                              0                              0                              0                              0                             0                             0;
                         247//1000                      0                              0                              0                              0                             0                             0;
                         247//4000                      2694949928731//7487940209513   0                              0                              0                             0                             0;
                         464650059369//8764239774964    878889893998//2444806327765   -952945855348//12294611323341   0                              0                             0                             0;
                         476636172619//8159180917465   -1271469283451//7793814740893  -859560642026//4356155882851    1723805262919//4571918432560   0                             0                             0;
                         6338158500785//11769362343261 -4970555480458//10924838743837  3326578051521//2647936831840  -880713585975//1841400956686   -1428733748635//8843423958496  0                             0;
                         760814592956//3276306540349    760814592956//3276306540349   -47223648122716//6934462133451  71187472546993//9669769126921 -13330509492149//9695768672337 11565764226357//8513123442827 0;
        ],
        Rational{BigInt}[0 0 9164257142617//17756377923965 -10812980402763//74029279521829 1335994250573//5691609445217 2273837961795//8368240463276 247//2000  ;
                         0 0 4469248916618//8635866897933  -621260224600//4094290005349    696572312987//2942599194819  1532940081127//5565293938103 2441//20000],
        Rational{BigInt}[0                               0                              0                              0                              0                             0                            0          ;
                         1235//10000                     1235//10000                    0                              0                              0                             0                            0          ;
                         624185399699//4186980696204     624185399699//4186980696204    1235//10000                    0                              0                             0                            0          ;
                         1258591069120//10082082980243   1258591069120//10082082980243 -322722984531//8455138723562    1235//10000                    0                             0                            0          ;
                         -436103496990//5971407786587   -436103496990//5971407786587   -2689175662187//11046760208243  4431412449334//12995360898505  1235//10000                   0                            0          ;
                         -2207373168298//14430576638973 -2207373168298//14430576638973  242511121179//3358618340039    3145666661981//7780404714551   5882073923981//14490790706663 1235//10000                  0          ;
                         0                               0                              9164257142617//17756377923965 -10812980402763//74029279521829 1335994250573//5691609445217  2273837961795//8368240463276 1235//10000;
                        ],
        Rational{BigInt}[0 0 9164257142617//17756377923965 -10812980402763//74029279521829 1335994250573//5691609445217 2273837961795//8368240463276 247//2000  ;
                         0 0 4469248916618//8635866897933  -621260224600//4094290005349    696572312987//2942599194819  1532940081127//5565293938103 2441//20000],
        ; low_storage=false)

    # The 5th order KennedyCarpenter548 method seems to be missing the 8'th row of a_implicit
    # coefficients in the Kennedy&Carpenter2019 paper, so this is not correct.
    ## 5th-order, 8-stage IMEX method 'ARK5(4)8L[2]SA₂' from Kennedy & Carpenter 2019
    ## (https://doi.org/10.1016/j.apnum.2018.10.007)
    #convert_and_check_butcher_tableau(
    #    "KennedyCarpenterARK548",
    #    Rational{BigInt}[ 0                               0                             0                              0                              0                              0                              0                            0;
    #                      4//9                            0                             0                              0                              0                              0                              0                            0;
    #                      1//9                            1183333538310//1827251437969  0                              0                              0                              0                              0                            0;
    #                      895379019517//9750411845327     477606656805//13473228687314 -112564739183//9373365219272    0                              0                              0                              0                            0;
    #                      -4458043123994//13015289567637 -2500665203865//9342069639922  983347055801//8893519644487    2185051477207//2551468980502   0                              0                              0                            0;
    #                      -167316361917//17121522574472   1605541814917//7619724128744  991021770328//13052792161721   2342280609577//11279663441611  3012424348531//12792462456678  0                              0                            0;
    #                      6680998715867//14310383562358   5029118570809//3897454228471  2415062538259//6382199904604  -3924368632305//6964820224454  -4331110370267//15021686902756 -3944303808049//11994238218192  0                            0;
    #                      2193717860234//3570523412979    2193717860234//3570523412979  5952760925747//18750164281544 -4412967128996//6196664114337   4151782504231//36106512998704  572599549169//6265429158920   -457874356192//11306498036315 0;
    #                    ],
    #    Rational{BigInt}[ 0 0 3517720773327//20256071687669 4569610470461//17934693873752 2819471173109//11655438449929 3296210113763//10722700128969 -1142099968913//5710983926999  2//9                        ;
    #                      0 0 520639020421//8300446712847   4550235134915//17827758688493 1482366381361//6201654941325  5551607622171//13911031047899 -5266607656330//36788968843917 1074053359553//5740751784926;
    #                    ],
    #    Rational{BigInt}[ 0                             0                             0                             0                              0                           0                            0    0   ;
    #                      2//9                          2//9                          0                             0                              0                           0                            0    0   ;
    #                      2366667076620//8822750406821  2366667076620//8822750406821  2//9                          0                              0                           0                            0    0   ;
    #                     -257962897183//4451812247028  -257962897183//4451812247028   128530224461//14379561246022  2//9                           0                           0                            0    0   ;
    #                     -486229321650//11227943450093 -486229321650//11227943450093 -225633144460//6633558740617   1741320951451//6824444397158   2//9                        0                            0    0   ;
    #                      621307788657//4714163060173   621307788657//4714163060173  -125196015625//3866852212004   940440206406//7593089888465    961109811699//6734810228204 2//9                         0    0   ;
    #                      2036305566805//6583108094622  2036305566805//6583108094622 -3039402635899//4450598839912 -1829510709469//31102090912115 -286320471013//6931253422520 8651533662697//9642993110008 2//9 0   ;
    #                      0                             0                             0                             0                              0                           0                            0    2//9;
    #                    ],
    #    Rational{BigInt}[ 0 0 3517720773327//20256071687669 4569610470461//17934693873752 2819471173109//11655438449929 3296210113763//10722700128969 -1142099968913//5710983926999  2//9                        ;
    #                      0 0 520639020421//8300446712847   4550235134915//17827758688493 1482366381361//6201654941325  5551607622171//13911031047899 -5266607656330//36788968843917 1074053359553//5740751784926;
    #                    ],
    #   ; low_storage=false)

    # 3rd-order, 4-stage IMEX method from Kennedy & Carpenter 2003
    # (https://doi.org/10.1016/S0168-9274(02)00138-1,
    # https://ntrs.nasa.gov/api/citations/20010075154/downloads/20010075154.pdf)
    convert_and_check_butcher_tableau(
        "KennedyCarpenterARK324",
        Rational{BigInt}[0                              0                            0                               0;
                         1767732205903//2027836641118   0                            0                               0;
                         5535828885825//10492691773637  788022342437//10882634858940 0                               0;
                         6485989280629//16251701735622 -4246266847089//9704473918619 10755448449292//10357097424841  0;
        ],
        Rational{BigInt}[1471266399579//7840856788654  -4482444167858//7529755066697   11266239266428//11593286722821 1767732205903//4055673282236;
                         2756255671327//12835298489170 -10771552573575//22201958757719 9247589265047//10645013368117  2193209047091//5459859503100],
        Rational{BigInt}[0                              0                            0                               0                           ;
                         1767732205903//4055673282236   1767732205903//4055673282236 0                               0                           ;
                         2746238789719//10658868560708 -640167445237//6845629431997  1767732205903//4055673282236    0                           ;
                         1471266399579//7840856788654  -4482444167858//7529755066697 11266239266428//11593286722821  1767732205903//4055673282236;
                        ],
        Rational{BigInt}[1471266399579//7840856788654  -4482444167858//7529755066697   11266239266428//11593286722821 1767732205903//4055673282236;
                         2756255671327//12835298489170 -10771552573575//22201958757719 9247589265047//10645013368117  2193209047091//5459859503100],
        ; low_storage=false)

    # 2nd-order, 2-stage IMEX method 'IMEX-SSP2(2,2,2)' from Pareschi & Russo 2005, Table II
    # (https://doi.org/10.1007/s10915-004-4636-4)
    gamma = 1 - 1 / sqrt(BigFloat(2))
    convert_and_check_butcher_tableau(
        "PareschiRusso2(2,2,2)",
        BigFloat[0 0;
                 1 0;
                ],
        BigFloat[1//2 1//2],
        BigFloat[gamma     0    ;
                 1-2*gamma gamma;
                ],
        BigFloat[1//2 1//2],
        ; low_storage=false)

    # 2nd-order, 3-stage IMEX method 'IMEX-SSP2(3,2,2)' from Pareschi & Russo 2005, Table III
    # (https://doi.org/10.1007/s10915-004-4636-4)
    convert_and_check_butcher_tableau(
        "PareschiRusso2(3,2,2)",
        Rational{Int64}[0 0 0;
                        0 0 0;
                        0 1 0;
                ],
        Rational{Int64}[0 1//2 1//2],
        Rational{Int64}[ 1//2 0    0   ;
                        -1//2 1//2 0   ;
                         0    1//2 1//2;
                ],
        Rational{Int64}[0 1//2 1//2],
        ; low_storage=false)

    # 2nd-order, 3-stage IMEX method 'IMEX-SSP2(3,3,2)' from Pareschi & Russo 2005, Table IV
    # (https://doi.org/10.1007/s10915-004-4636-4)
    convert_and_check_butcher_tableau(
        "PareschiRusso2(3,3,2)",
        Rational{Int64}[0    0    0;
                        1//2 0    0;
                        1//2 1//2 0;
                ],
        Rational{Int64}[1//3 1//3 1//3],
        Rational{Int64}[1//4 0    0   ;
                        0    1//4 0   ;
                        1//3 1//3 1//3;
                ],
        Rational{Int64}[1//3 1//3 1//3],
        ; low_storage=false)

    # 3rd-order, 4-stage IMEX method 'IMEX-SSP3(4,3,3)' from Pareschi & Russo 2005, Table VI
    # (https://doi.org/10.1007/s10915-004-4636-4)
    alpha = 0.24169426078821
    beta = 0.06042356519705
    eta = 0.12915286960590
    convert_and_check_butcher_tableau(
        "PareschiRusso3(4,3,3)",
        typeof(alpha)[0 0    0    0;
                      0 0    0    0;
                      0 1    0    0;
                      0 1//4 1//4 0;
                     ],
        typeof(alpha)[0 1//6 1//6 2//3],
        typeof(alpha)[alpha  0       0                   0    ;
                      -alpha alpha   0                   0    ;
                      0      1-alpha alpha               0    ;
                      beta   eta     1//2-beta-eta-alpha alpha;
                     ],
        typeof(alpha)[0 1//6 1//6 2//3],
        ; low_storage=false)

    # 1st-order, 1-stage IMEX method, combination of forward and backward Euler steps.
    convert_and_check_butcher_tableau(
        "EulerIMEX",
        Rational{Int64}[0;
                       ],
        Rational{Int64}[1],
        Rational{Int64}[1;
                       ],
        Rational{Int64}[1],
        ; low_storage=false)
end

end # CalculateRKCoeffs

if abspath(PROGRAM_FILE) == @__FILE__
    using .CalculateRKCoeffs
    CalculateRKCoeffs.calculate_all_coeffs()
end
