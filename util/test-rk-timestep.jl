include("calculate_rk_coeffs.jl")

multiplier = 1
dt = 1.0e-2 / multiplier
function f(y)
    return y
    #return 1.0
end
y0 = 1.0
nsteps = 100 * multiplier

t = [i*dt for i ∈ 0:nsteps]
analytic = @. y0*exp(t)
#analytic = @. 1.0 + t

function rk_advance(rk_coeffs, y0, dt, nsteps)
    n_rk_stages = size(rk_coeffs, 1) - 1
    #println("n_rk_stages=$n_rk_stages, ", size(rk_coeffs))
    yscratch = zeros(n_rk_stages + 1)
    yscratch[1] = y0
    adaptive = size(rk_coeffs, 2) > n_rk_stages

    result = zeros(nsteps+1)
    result[1] = y0

    error = zeros(nsteps+1)

    for it ∈ 1:nsteps
        for istage ∈ 1:n_rk_stages
            yscratch[istage+1] = yscratch[istage] + dt*f(yscratch[istage])
            this_coeffs = rk_coeffs[:,istage]
            yscratch[istage+1] = sum(this_coeffs[i]*yscratch[i] for i ∈ 1:istage+1)
        end
        #k1 = 2*(yscratch[2] - yscratch[1])
        #k2 = 2*(yscratch[3] - yscratch[1])
        #k3 = yscratch[4] - yscratch[1]
        #k4 = 6*(yscratch[5] - yscratch[1]) - k1 - 2*k2 - 2*k3
        #println("kcheck = ", k1, " ", k2, " ", k3, " ", k4)
        if adaptive
            error[it+1] = sum(rk_coeffs[i, n_rk_stages+1]*yscratch[i] for i ∈ 1:n_rk_stages+1)
        end
        yscratch[1] = yscratch[end]
        result[it+1] = yscratch[end]
    end

    return result, error
end

function rk_advance_non_adaptive(rk_coeffs, y0, dt, nsteps)
    n_rk_stages = size(rk_coeffs, 2)
    println("check n_rk_stages=$n_rk_stages")

    yscratch = zeros(n_rk_stages + 1)
    yscratch[1] = y0

    result = zeros(nsteps+1)
    result[1] = y0

    for it ∈ 1:nsteps
        for istage ∈ 1:n_rk_stages
            yscratch[istage+1] = yscratch[istage] + dt*f(yscratch[istage])
            this_coeffs = rk_coeffs[:,istage]
            #println("istage=$istage, this_coeffs=$this_coeffs")
            yscratch[istage+1] = this_coeffs[1]*yscratch[1] + this_coeffs[2]*yscratch[istage] + this_coeffs[3]*yscratch[istage+1]
            #println("istage=$istage, ", yscratch[istage+1])
        end
        #println("before yscratch=$yscratch")
        yscratch[1] = yscratch[n_rk_stages+1]
        #println("after yscratch=$yscratch")
        result[it+1] = yscratch[n_rk_stages+1]
    end

    return result
end

function rk_advance_butcher(a, b, y0, dt, nsteps)
    n_rk_stages = size(a, 2)
    kscratch = zeros(n_rk_stages)
    y = y0
    if ndims(b) == 1
        b = b'
    end
    adaptive = size(b, 1) > 1

    result = zeros(nsteps+1)
    result[1] = y0

    error = zeros(nsteps+1)

    for it ∈ 1:nsteps
        kscratch[1] = dt*f(y)
        for i ∈ 2:n_rk_stages
            kscratch[i] = dt*f(y + sum(a[i,j] * kscratch[j] for j ∈ 1:i-1))
        end
        if adaptive
            y_loworder = y + sum(b[2,j]*kscratch[j] for j ∈ 1:n_rk_stages)
        end
        y = y + sum(b[1,j]*kscratch[j] for j ∈ 1:n_rk_stages)
        if adaptive
            error[it+1] = y_loworder - y
        end
        result[it+1] = y
    end

    return result, error
end

function rk4_by_hand(y0, dt, nsteps)
    result = zeros(nsteps+1)
    y = y0
    result[1] = y
    for it ∈ 1:nsteps
        k1 = dt*f(y)
        k2 = dt*f(y + k1/2)
        k3 = dt*f(y + k2/2)
        k4 = dt*f(y + k3)
        println("k=", k1, " ", k2, " ", k3, " ", k4)
        y = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        result[it+1] = y
    end
    return result
end

methods = Dict(
    "SSPRK3" => (rk_coeffs=Float64[0 3//4 1//3; 1 0 0; 0 1//4 0; 0 0 2//3],
                 a=Float64[0 0 0; 1 0 0; 1//4 1//4 0],
                 b=Float64[1//6 1//6 2//3]),

    "RK4" => (rk_coeffs = Float64[1//2 1 1 -1//3; 1//2 -1//2 0 1//3; 0 1//2 -1 2//3; 0 0 1 1//6; 0 0 0 1//6],
              a = Float64[0 0 0 0; 1//2 0 0 0; 0 1//2 0 0; 0 0 1 0],
              b = Float64[1//6 1//3 1//3 1//6]),

    "RKF45" => (rk_coeffs = Float64[3//4 5//8 10469//2197 115//324 121//240 641//1980 11//36; 1//4 3//32 17328//2197 95//54 33//10 232//165 4//3; 0 9//32 -32896//2197 -95744//29241 -1408//285 -512//171 -512//171; 0 0 7296//2197 553475//233928 6591//1520 2197//836 2197//836; 0 0 0 -845//4104 -77//40 -56//55 -1; 0 0 0 0 -11//40 34//55 8//11; 0 0 0 0 0 2//55 -1],
                a = Float64[0 0 0 0 0 0; 1//4 0 0 0 0 0; 3//32 9//32 0 0 0 0; 1932//2197 -7200//2197 7296//2197 0 0 0; 439//216 -8 3680//513 -845//4104 0 0; -8//27 2 -3544//2565 1859//4104 -11//40 0],
                b = Float64[16//135 0 6656//12825 28561//56430 -9//50 2//55; 25//216 0 1408//2565 2197//4104 -1//5 0]),

    "RKF45 truncated" => (rk_coeffs = Float64[3//4 5//8 10469//2197 115//324 121//240 641//1980; 1//4 3//32 17328//2197 95//54 33//10 232//165; 0 9//32 -32896//2197 -95744//29241 -1408//285 -512//171; 0 0 7296//2197 553475//233928 6591//1520 2197//836; 0 0 0 -845//4104 -77//40 -56//55; 0 0 0 0 -11//40 34//55; 0 0 0 0 0 2//55],
                a = Float64[0 0 0 0 0 0; 1//4 0 0 0 0 0; 3//32 9//32 0 0 0 0; 1932//2197 -7200//2197 7296//2197 0 0 0; 439//216 -8 3680//513 -845//4104 0 0; -8//27 2 -3544//2565 1859//4104 -11//40 0],
                b = Float64[16//135 0 6656//12825 28561//56430 -9//50 2//55]),

    "Heun SSPRK2" => (rk_coeffs = Float64[0 1//2; 1 0; 0 1//2],
                      a = Float64[0 0; 1 0],
                      b = Float64[1//2 1//2]),

    "Gottlieb 43" => (rk_coeffs = Float64[0 1//2 2//3; 1 0 0; 0 1//2 -1//3; 0 0 2//3],
                      a = Float64[0 0 0; 1 0 0; 1//2 1//2 0],
                      b = Float64[1//6 1//6 2//3]),

    "mk ssprk3" => (rk_coeffs = Float64[1//2 0    2//3 0   ;
                                        1//2 1//2 0    0   ;
                                        0    1//2 1//6 0   ;
                                        0    0    1//6 1//2;
                                        0    0    0    1//2],
                    a = Float64[0 0 0 0; 1//2 0 0 0; 1//2 1//2 0 0; 1//6 1//6 1//6 0],
                    b = Float64[1//6 1//6 1//6 1//2]),

    "mk ssprk2" => (rk_coeffs = Float64[0.0 0.5 0.0;
                                        1.0 0.0 0.0;
                                        0.0 0.5 0.0],
                    a = Float64[0.0 0.0; 1.0 0.0],
                    b = Float64[0.5 0.5; 0.5 0.5]),

    "Fekete 43" => (rk_coeffs = Float64[1//2 0 2//3 0 -1//2; 1//2 1//2 0 0 0; 0 1//2 1//6 0 0; 0 0 1//6 1//2 1; 0 0 0 1//2 -1//2],
                    a = Float64[0 0 0 0; 1//2 0 0 0; 1//2 1//2 0 0; 1//6 1//6 1//6 0],
                    b = Float64[1//6 1//6 1//6 1//2; 1//4 1//4 1//4 1//4]),

    "Fekete 43 truncated" => (rk_coeffs = Float64[1//2 0 2//3 0; 1//2 1//2 0 0; 0 1//2 1//6 0; 0 0 1//6 1//2; 0 0 0 1//2],
                    a = Float64[0 0 0 0; 1//2 0 0 0; 1//2 1//2 0 0; 1//6 1//6 1//6 0],
                    b = Float64[1//6 1//6 1//6 1//2]),

    "Fekete 42" => (rk_coeffs = Float64[2//3 0 0 1//4 -1//8; 1//3 2//3 0 0 3//16; 0 1//3 2//3 0 0; 0 0 1//3 1//2 3//16; 0 0 0 1//4 -1//4],
                    a = Float64[0 0 0 0; 1//3 0 0 0; 1//3 1//3 0 0; 1//3 1//3 1//3 0],
                    b = Float64[1//4 1//4 1//4 1//4; 5//16 1//4 1//4 3//16]),

    "Fekete 10,4" => (rk_coeffs = Float64[5//6 0 0 0 3//5 0 0 0 0 -1//2 -1//5; 1//6 5//6 0 0 0 0 0 0 0 0 6//5; 0 1//6 5//6 0 0 0 0 0 0 0 0; 0 0 1//6 5//6 0 0 0 0 0 0 -9//5; 0 0 0 1//6 1//3 0 0 0 0 0 9//5; 0 0 0 0 1//15 5//6 0 0 0 9//10 0; 0 0 0 0 0 1//6 5//6 0 0 0 -6//5; 0 0 0 0 0 0 1//6 5//6 0 0 6//5; 0 0 0 0 0 0 0 1//6 5//6 0 -9//5; 0 0 0 0 0 0 0 0 1//6 1//2 9//5; 0 0 0 0 0 0 0 0 0 1//10 -1],
                      a = Float64[0 0 0 0 0 0 0 0 0 0; 1//6 0 0 0 0 0 0 0 0 0; 1//6 1//6 0 0 0 0 0 0 0 0; 1//6 1//6 1//6 0 0 0 0 0 0 0; 1//6 1//6 1//6 1//6 0 0 0 0 0 0; 1//15 1//15 1//15 1//15 1//15 0 0 0 0 0; 1//15 1//15 1//15 1//15 1//15 1//6 0 0 0 0; 1//15 1//15 1//15 1//15 1//15 1//6 1//6 0 0 0; 1//15 1//15 1//15 1//15 1//15 1//6 1//6 1//6 0 0; 1//15 1//15 1//15 1//15 1//15 1//6 1//6 1//6 1//6 0],
                      b = Float64[1//10 1//10 1//10 1//10 1//10 1//10 1//10 1//10 1//10 1//10; 1//5 0 0 3//10 0 0 1//5 0 3//10 0]),

    "Fekete 6,4" => (rk_coeffs = [0.6447024483081 0.2386994475333264 0.5474858792272213 0.3762853856474131 0.0 -0.18132326703443313 -0.0017300417984673078; 0.3552975516919 0.4295138541066736 -6.461498003318411e-14 -1.1871059690804486e-13 0.0 2.9254376698872875e-14 -0.18902907903375094; 0.0 0.33178669836 0.25530138316744333 -3.352873534367973e-14 0.0 0.2059808002676668 0.2504712436879622; 0.0 0.0 0.1972127376054 0.3518900216285391 0.0 0.4792670116241715 -0.9397479180374522; 0.0 0.0 0.0 0.2718245927242 0.5641843457422999 9.986456106503283e-14 1.1993626679930305; 0.0 0.0 0.0 0.0 0.4358156542577 0.3416567872695656 -0.5310335716309745; 0.0 0.0 0.0 0.0 0.0 0.1544186678729 0.2117066988196524],
                     a = [0.0 0.0 0.0 0.0 0.0 0.0; 0.3552975516919 0.0 0.0 0.0 0.0 0.0; 0.2704882223931 0.33178669836 0.0 0.0 0.0 0.0; 0.1223997401356 0.1501381660925 0.1972127376054 0.0 0.0 0.0; 0.0763425067155 0.093643368364 0.123004466581 0.2718245927242 0.0 0.0; 0.0763425067155 0.093643368364 0.123004466581 0.2718245927242 0.4358156542577 0.0],
                     b = [0.1522491819555 0.1867521364225 0.1555370561501 0.1348455085546 0.2161974490441 0.1544186678729; 0.1210663237182 0.230884400455 0.0853424972752 0.3450614904457 0.0305351538213 0.1871101342844]),
   )

a, b = convert_rk_coeffs_to_butcher_tableau(methods["RKF45"].rk_coeffs)
methods["RKF45 attempt 2"] = (rk_coeffs = methods["RKF45"].rk_coeffs,
                             a = a, b = b)

for (k,v) ∈ methods
    println("\n", k)
    result, error = rk_advance(v.rk_coeffs, y0, dt, nsteps)
    result_butcher, error_butcher = rk_advance_butcher(v.a, v.b, y0, dt, nsteps)

    #for i ∈ 1:multiplier:nsteps+1
    #    println("$i t=", t[i], " analytic=", analytic[i], " result=", result[i], " result_butcher=", result_butcher[i])
    #end
    println("t=", t[end])
    println("analytic       = ", analytic[end])
    println("result         = ", result[end])
    println("result_butcher = ", result_butcher[end])
    println("error         = ", error[end])
    println("error_butcher = ", error_butcher[end])
end

n_rk_stages = 3
println("\nnon-adaptive mk ssprk$n_rk_stages")
rk_coefs = zeros(3,n_rk_stages)
rk_coefs .= 0.0
if n_rk_stages == 4
    rk_coefs[1,1] = 0.5
    rk_coefs[3,1] = 0.5
    rk_coefs[2,2] = 0.5
    rk_coefs[3,2] = 0.5
    rk_coefs[1,3] = 2.0/3.0
    rk_coefs[2,3] = 1.0/6.0
    rk_coefs[3,3] = 1.0/6.0
    rk_coefs[2,4] = 0.5
    rk_coefs[3,4] = 0.5
elseif n_rk_stages == 3
    rk_coefs[3,1] = 1.0
    rk_coefs[1,2] = 0.75
    rk_coefs[3,2] = 0.25
    rk_coefs[1,3] = 1.0/3.0
    rk_coefs[3,3] = 2.0/3.0
elseif n_rk_stages == 2
    rk_coefs[3,1] = 1.0
    rk_coefs[1,2] = 0.5
    rk_coefs[3,2] = 0.5
elseif n_rk_stages == 1
    rk_coefs[3,1] = 1.0
else
    error("Unsupported number of RK stages, n_rk_stages=$n_rk_stages")
end
result = rk_advance_non_adaptive(rk_coefs, y0, dt, nsteps)
println("t=", t[end])
println("analytic       = ", analytic[end])
println("result         = ", result[end])
