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
