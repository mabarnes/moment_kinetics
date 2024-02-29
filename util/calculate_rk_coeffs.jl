"""
We implement the Runge-Kutta timestepper in `moment_kinetics` in a different form from the
most conventional one, so in some cases we need to convert the 'conventional' coefficients
into ones that we can use.
"""

using Symbolics

#println("Coefficients for RK45 - Runge-Kutta-Fehlberg adaptive timestepping scheme")
#
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
#
## f is the RHS function: dy/dt = f(y)
#
## y1...y6, yout1...yout6, ylow, yhigh, yhigh_out, yerr are defined below
## k1...k6 are as defined on the Wikipedia page
## f1...f6 are h*f(y1)...h*f(y6)
## h is the timestep
#@variables y1 y2 y3 y4 y5 y6 yout1 yout2 yout3 yout4 yout5 yout6 ylow yhigh yhigh_out yerr k1 k2 k3 k4 k5 k6 f1 f2 f3 f4 f5 f6 h
#y = [y1, y2, y3, y4, y5, y6, yhigh]
#yout = [yout1, yout2, yout3, yout4, yout5, yout6, yhigh_out]
#k = [k1, k2, k3, k4, k5, k6]
#f = [f1, f2, f3, f4, f5, f6]
#
## From the Wikipedia page
## k[i] = h*f(y1 + ∑_j<i B[i,j]*k[j])
##
## We define the argument of f() in that expression as y[i+1] so
## k[i] = h*f(y[i+1])
##
## In moment_kinetics, we need to calculate each y[i] as a sum of the previous y[i] and the
## result of the 'forward Euler step' (y1 + h*f(y[i-1]))
#
## y1 is the 'intial' state y(t)
#k[1] = (y2 - y1) // B[2,1]
#for i ∈ 2:5
#    k[i] = (y[i+1] - y1 - sum(B[i+1,j]*k[j] for j ∈ 1:i-1)) // B[i+1,i]
#    k[i] = simplify(expand(k[i]))
#end
#
#yout[1] = y1
#yout[2] = y1 + B[2,1]*f[1]
#for i ∈ 3:6
#    yout[i] = y1 + sum((B[i,j]*k[j] for j ∈ 1:i-2)) + B[i,i-1]*f[i-1]
#    yout[i] = simplify(expand(yout[i]))
#end
#
## y7 is the 5th order approximation to y(t+h)
#yout[7] = y1 + sum((CH[j]*k[j] for j ∈ 1:5)) + CH[6]*f[6]
#yout[7] = simplify(expand(yout[7]))
#k[6] = (yhigh - y1 - sum((CH[j]*k[j] for j ∈ 1:5))) // CH[6]
#k[6] = simplify(expand(k[6]))
#
## ylow is the 4th order approximation to y(t+h), not actually used?
#ylow = y1 + sum((C[j]*k[j] for j ∈ 1:5))
#ylow = simplify(expand(ylow))
#
#println("CT ", CT)
#yerr = sum((CT[j]*k[j] for j ∈ 1:6))
#yerr = simplify(expand(yerr))
#
#rk_coeffs = zeros(Rational{Int64}, 7, 7)
#for i ∈ 1:6
#    f_coeff = Symbolics.coeff(yout[i+1], f[i])
#    rk_coeffs[1,i] = Symbolics.coeff(yout[i+1], y1) - f_coeff # Subtract 1 because 1 lot of y1 is included in the 'forward Euler step'
#    for j ∈ 2:i
#        rk_coeffs[j,i] = Symbolics.coeff(yout[i+1], y[j])
#    end
#
#    # Coefficient of the result of the 'forward Euler step' (y1 + h*f(y[i])
#    rk_coeffs[i+1,i] = f_coeff
#end
#
## Use final column of rk_coeffs to store the coefficients used to calculate the truncation
## error estimate
#for j ∈ 1:7
#    rk_coeffs[j,7] = Symbolics.coeff(yerr, y[j])
#end
#
#for i ∈ 1:6
#    println("k$i = ", k[i])
#end
#for i ∈ 1:6
#    println("yout$i = ", yout[i])
#end
#println("ylow = $ylow")
#println("yhigh_out = ", yout[end])
#println("yerr = $yerr")
#println("diff = ", simplify(expand(ylow - y[7])))
#for i ∈ 1:7
#    println("rk_coeffs[:,$i] = ", rk_coeffs[:,i])
#end
#
#println("\nrk_coeffs matrix:")
#display(rk_coeffs)
#println(rk_coeffs)
#for i ∈ 1:7
#    println("i=$i, sum=", sum(rk_coeffs[:,i]))
#end

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

Returns an array `rk_coeffs` of size `n_rk_stages`x`n_rk_stages` where `size(a) =
(n_rk_stages, n_rk_stages)`.
"""
function convert_butcher_tableau_for_moment_kinetics(a, b; low_storage=true)
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
    @variables y[1:n_rk_stages+1] y_out[1:n_rk_stages+1] k[1:n_rk_stages] k_subs[1:n_rk_stages]
    y = Symbolics.scalarize(y)
    y_out = Symbolics.scalarize(y_out)
    k = Symbolics.scalarize(k)
    k_subs = Symbolics.scalarize(k_subs)

    k_subs[1] = (y[2] - y[1]) // a[2,1]
    k_subs[1] = simplify(expand(k_subs[1]))
    for i ∈ 2:n_rk_stages-1
        k_subs[i] = (y[i+1] - y[1] - sum(a[i+1,j]*k_subs[j] for j ∈ 1:i-1)) // a[i+1,i]
        k_subs[i] = simplify(expand(k_subs[i]))
    end

    y_out[1] = y[1]
    y_out[2] = y[1] + a[2,1] * k[1]
    y_out[2] = simplify(expand(y_out[2]))
    for i ∈ 3:n_rk_stages
        y_out[i] = y[1] + sum(a[i,j]*k_subs[j] for j ∈ 1:i-2) + a[i,i-1]*k[i-1]
        y_out[i] = simplify(expand(y_out[i]))
    end

    y_out[n_rk_stages+1] = y[1] + sum(b[1,j]*k_subs[j] for j ∈ 1:n_rk_stages-1) +
                           b[1,n_rk_stages]*k[n_rk_stages]
    y_out[n_rk_stages+1] = simplify(expand(y_out[n_rk_stages+1]))
    k_subs[n_rk_stages] = (y[n_rk_stages+1] - y[1]
                           - sum(b[1,j]*k_subs[j] for j ∈ 1:n_rk_stages-1)) //
                          b[1,n_rk_stages]
    k_subs[n_rk_stages] = simplify(expand(k_subs[n_rk_stages]))
    #println("y_out")
    #for i ∈ 1:n_rk_stages+1
    #    println(y_out[i])
    #end
    #println("k")
    #for i ∈ 1:n_rk_stages
    #    println(k_subs[i])
    #end

    if low_storage
        rk_coeffs = zeros(Rational{Int64}, 3, output_size)
        for i in 1:n_rk_stages
            k_coeff = Symbolics.coeff(y_out[i+1], k[i])

            if i == 1
                j = i
                rk_coeffs[1,i] = Symbolics.coeff(y_out[i+1], y[j])
                #println("k_coeff=$k_coeff, yout[$i]=", y_out[i+1])
                #println("before rk_coeffs[:,$i]=", rk_coeffs[:,i])
                # Subtract k_coeff because k_coeff*y[i] is included in the 'forward Euler step'
                rk_coeffs[1,i] -= k_coeff

                # Coefficient of the result of the 'forward Euler step' (y1 + h*f(y[i])
                rk_coeffs[3,i] = k_coeff
                #println("after rk_coeffs[:,$i]=", rk_coeffs[:,i])
            else
                j = 1
                rk_coeffs[1,i] = Symbolics.coeff(y_out[i+1], y[j])
                for j ∈ 2:i-2
                    if Symbolics.coeff(y_out[i+1], y[j]) != 0
                        error("Found non-zero coefficient where zero was expected for low-storage coefficients")
                    end
                end
                j = i
                rk_coeffs[2,i] = Symbolics.coeff(y_out[i+1], y[j])
                #println("k_coeff=$k_coeff, yout[$i]=", y_out[i+1])
                #println("before rk_coeffs[:,$i]=", rk_coeffs[:,i])
                # Subtract k_coeff because k_coeff*y[i] is included in the 'forward Euler step'
                rk_coeffs[2,i] -= k_coeff

                # Coefficient of the result of the 'forward Euler step' (y1 + h*f(y[i])
                rk_coeffs[3,i] = k_coeff
                #println("after rk_coeffs[:,$i]=", rk_coeffs[:,i])
            end
        end

        #for i ∈ 1:n_rk_stages
        #    println("k$i = ", k_subs[i])
        #end
        if adaptive
            error_coefficients = b[2,:] .- b[1,:]
            #println("error_coefficients=", error_coefficients)
            #println("error coefficients ", error_coefficients)
            y_err = sum(error_coefficients[j]*k_subs[j] for j ∈ 1:n_rk_stages)
            y_err = simplify(expand(y_err))

            # Use final column of rk_coeffs to store the coefficients used to calculate the truncation
            # error estimate
            j = 1
            rk_coeffs[1,n_rk_stages+1] = Symbolics.coeff(y_err, y[j])
            for j ∈ 2:n_rk_stages-1
                if Symbolics.coeff(y_err, y[j]) != 0
                    error("Found non-zero error coefficient where zero was expected for low-storage coefficients")
                end
            end
            j = n_rk_stages
            rk_coeffs[2,n_rk_stages+1] = Symbolics.coeff(y_err, y[j])
            j = n_rk_stages + 1
            rk_coeffs[3,n_rk_stages+1] = Symbolics.coeff(y_err, y[j])
        end
    else
        rk_coeffs = zeros(Rational{Int64}, n_rk_stages+1, output_size)
        for i in 1:n_rk_stages
            k_coeff = Symbolics.coeff(y_out[i+1], k[i])

            for j ∈ 1:i
                rk_coeffs[j,i] = Symbolics.coeff(y_out[i+1], y[j])
            end
            #println("k_coeff=$k_coeff, yout[$i]=", y_out[i+1])
            #println("before rk_coeffs[:,$i]=", rk_coeffs[:,i])
            # Subtract k_coeff because k_coeff*y[i] is included in the 'forward Euler step'
            rk_coeffs[i,i] -= k_coeff

            # Coefficient of the result of the 'forward Euler step' (y1 + h*f(y[i])
            rk_coeffs[i+1,i] = k_coeff
            #println("after rk_coeffs[:,$i]=", rk_coeffs[:,i])
        end

        #for i ∈ 1:n_rk_stages
        #    println("k$i = ", k_subs[i])
        #end
        if adaptive
            error_coefficients = b[2,:] .- b[1,:]
            #println("error_coefficients=", error_coefficients)
            #println("error coefficients ", error_coefficients)
            y_err = sum(error_coefficients[j]*k_subs[j] for j ∈ 1:n_rk_stages)
            y_err = simplify(expand(y_err))

            # Use final column of rk_coeffs to store the coefficients used to calculate the truncation
            # error estimate
            for j ∈ 1:n_rk_stages+1
                rk_coeffs[j,n_rk_stages+1] = Symbolics.coeff(y_err, y[j])
            end
        end
    end

    return rk_coeffs
end

function convert_rk_coeffs_to_butcher_tableau(rkcoeffs::AbstractArray{T,N}) where {T,N}
    n_rk_stages = size(rkcoeffs, 1) - 1
    adaptive = (size(rkcoeffs, 2) == n_rk_stages + 1)

    @variables y[1:n_rk_stages+1] y_out[1:n_rk_stages+1] k[1:n_rk_stages] k_subs[1:n_rk_stages]
    y = Symbolics.scalarize(y)
    k = Symbolics.scalarize(k)

    for i ∈ 1:n_rk_stages
        y[i+1] = sum(rkcoeffs[j,i]*y[j] for j ∈ 1:i) + rkcoeffs[i+1,i]*(y[i] + k[i])
        y[i+1] = simplify(expand(y[i+1]))
    end
    #for i ∈ 1:n_rk_stages+1
    #    println("i=$i, y[$i]=", y[i])
    #end

    if adaptive
        b = zeros(T, 2, n_rk_stages)
    else
        b = zeros(T, 1, n_rk_stages)
    end

    for j ∈ 1:n_rk_stages
        b[1, j] = Symbolics.coeff(y[n_rk_stages+1], k[j])
    end
    if adaptive
        yerr = sum(rkcoeffs[j,n_rk_stages+1]*y[j] for j ∈ 1:n_rk_stages+1)
        error_coeffs = zeros(T, n_rk_stages)
        for j ∈ 1:n_rk_stages
            error_coeffs[j] = Symbolics.coeff(yerr, k[j])
        end
        #println("error_coeffs=", error_coeffs)
        # b[2,:] is the lower-order solution
        @. b[2,:] = error_coeffs + b[1,:]
    end

    a = zeros(T, n_rk_stages, n_rk_stages)
    for i ∈ 1:n_rk_stages
        for j ∈ 1:n_rk_stages
            a[i,j] = Symbolics.coeff(y[i], k[j])
        end
    end

    return a, b
end

println("\nRKF5(4)")
a = B
b = vcat(CH,C)
rk_coeffs = convert_butcher_tableau_for_moment_kinetics(a, b; low_storage=false)
print("a="); display(a)
print("b="); display(b)
print("rk_coeffs="); display(rk_coeffs)
println("a=$a")
println("b=$b")
println("rk_coeffs=$rk_coeffs")
for i ∈ 1:size(rk_coeffs, 2)
    println("i=$i, sum=", sum(rk_coeffs[:,i]))
end
println("convert back:")
a, b = convert_rk_coeffs_to_butcher_tableau(rk_coeffs)
print("a="); display(a)
print("b="); display(b)

println("\nSSPRK3")
# From https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
a = [0 0 0;
     1 0 0;
     1//4 1//4 0]
b = [1//6 1//6 2//3]
rk_coeffs = convert_butcher_tableau_for_moment_kinetics(a, b)
print("a="); display(a)
print("b="); display(b)
print("a="); println(a)
print("b="); println(b)
print("rk_coeffs="); display(rk_coeffs)
println("a=$a")
println("b=$b")
println("rk_coeffs=$rk_coeffs")
for i ∈ 1:size(rk_coeffs, 2)
    println("i=$i, sum=", sum(rk_coeffs[:,i]))
end
println("convert back:")
a, b = convert_rk_coeffs_to_butcher_tableau(rk_coeffs)
print("a="); display(a)
print("b="); display(b)

println("\nHeun's method SSPRK2")
# From https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
a = [0 0;
     1 0]
b = [1//2 1//2]
rk_coeffs = convert_butcher_tableau_for_moment_kinetics(a, b)
print("a="); display(a)
print("b="); display(b)
display(rk_coeffs)
println("a=$a")
println("b=$b")
println("rk_coeffs=$rk_coeffs")
for i ∈ 1:size(rk_coeffs, 2)
    println("i=$i, sum=", sum(rk_coeffs[:,i]))
end
println("convert back:")
a, b = convert_rk_coeffs_to_butcher_tableau(rk_coeffs)
print("a="); display(a)
print("b="); display(b)

println("\nGottlieb et al 4-stage 3rd order")
# From https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
a = [0 0 0;
     1 0 0;
     1//2 1//2 0]
b = [1//6 1//6 2//3]
rk_coeffs = convert_butcher_tableau_for_moment_kinetics(a, b)
print("a="); display(a)
print("b="); display(b)
print("rk_coeffs="); display(rk_coeffs)
println("a=$a")
println("b=$b")
println("rk_coeffs=$rk_coeffs")
for i ∈ 1:size(rk_coeffs, 2)
    println("i=$i, sum=", sum(rk_coeffs[:,i]))
end
println("convert back:")
a, b = convert_rk_coeffs_to_butcher_tableau(rk_coeffs)
print("a="); display(a)
print("b="); display(b)

println("\nRK4")
# From https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
a = [0 0 0 0;
     1//2 0 0 0;
     0 1//2 0 0;
     0 0 1 0]
b = [1//6 1//3 1//3 1//6]
rk_coeffs = convert_butcher_tableau_for_moment_kinetics(a, b; low_storage=false)
print("a="); display(a)
print("b="); display(b)
print("rk_coeffs="); display(rk_coeffs)
println("a=$a")
println("b=$b")
println("rk_coeffs=$rk_coeffs")
for i ∈ 1:size(rk_coeffs, 2)
    println("i=$i, sum=", sum(rk_coeffs[:,i]))
end
println("convert back:")
a, b = convert_rk_coeffs_to_butcher_tableau(rk_coeffs)
print("a="); display(a)
print("b="); display(b)

println("\nmk's ssprk3")
mk_ssprk3 = [1//2 0    2//3 0   ;
             1//2 1//2 0    0   ;
             0    1//2 1//6 0   ;
             0    0    1//6 1//2;
             0    0    0    1//2]
print("rk_coeffs="); display(mk_ssprk3)
println("convert back:")
a, b = convert_rk_coeffs_to_butcher_tableau(mk_ssprk3 )
print("a="); display(a)
print("b="); display(b)
println("a=$a")
println("b=$b")

"""
    construct_fekete_3rd_order(nstage)

Construct optimal 3rd order strong-stability preserving embedded Runge-Kutta method with
`nstage` stages, from [Fekete, Conde and Shadid, "Embedded pairs for optimal explicit
strong stability preserving Runge-Kutta methods", Journal of Computational and Applied
Mathematics 421 (2022) 114325, https://doi.org/10.1016/j.cam.2022.114325]
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

println("\nFekete 4(3)")
a, b = construct_fekete_3rd_order(4)
rk_coeffs = convert_butcher_tableau_for_moment_kinetics(a, b)
print("a="); display(a)
print("b="); display(b)
print("rk_coeffs="); display(rk_coeffs)
println("a=$a")
println("b=$b")
println("rk_coeffs=$rk_coeffs")
for i ∈ 1:size(rk_coeffs, 2)
    println("i=$i, sum=", sum(rk_coeffs[:,i]))
end
println("convert back:")
a, b = convert_rk_coeffs_to_butcher_tableau(rk_coeffs)
print("a="); display(a)
print("b="); display(b)

println("\nmk's ssprk3")
mk_ssprk3 = [0  3//4 1//3;
             1  0    0   ;
             0  1//4 0   ;
             0  0    2//3]
print("rk_coeffs="); display(mk_ssprk3)
println("convert back:")
a, b = convert_rk_coeffs_to_butcher_tableau(mk_ssprk3)
print("a="); display(a)
print("b="); display(b)
println("a=$a")
println("b=$b")

println("\nmk's ssprk4")
mk_ssprk4 = [1//2 0    2//3 0   ;
             1//2 1//2 0    0   ;
             0    1//2 1//6 0   ;
             0    0    1//6 1//2;
             0    0    0    1//2]
print("rk_coeffs="); display(mk_ssprk4)
println("convert back:")
a, b = convert_rk_coeffs_to_butcher_tableau(mk_ssprk4)
print("a="); display(a)
print("b="); display(b)
println("a=$a")
println("b=$b")

println("\nmk's ssprk2")
mk_ssprk2 = [0.0 0.5 0.0;
             1.0 0.0 0.0;
             0.0 0.5 0.0]
print("rk_coeffs="); display(mk_ssprk2)
println("convert back:")
a, b = convert_rk_coeffs_to_butcher_tableau(mk_ssprk2)
print("a="); display(a)
print("b="); display(b)
println("a=$a")
println("b=$b")
