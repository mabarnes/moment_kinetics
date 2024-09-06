using Pkg
Pkg.activate(".")

import moment_kinetics
    using moment_kinetics.input_structs: grid_input, advection_input
    using moment_kinetics.coordinates: define_coordinate
    using moment_kinetics.chebyshev: setup_chebyshev_pseudospectral
    using moment_kinetics.calculus: derivative!, integral #derivative_handle_wall_bc!
    using Plots

discretization = "chebyshev_pseudospectral"
#discretization = "finite_difference"
    etol = 1.0e-15
outprefix = "derivative_test"
    ###################
    ## df/dx Nonperiodic (No) BC test
    ###################
    
    # define inputs needed for the test
    ngrid = 17 #number of points per element 
    nelement_local = 4 # number of elements per rank
    nelement_global = nelement_local # total number of elements 
    L = 1.0 #physical box size in reference units 
    # create the 'input' struct containing input info needed to create a
    # coordinate
    x_input = OptionsDict("coord" => OptionsDict("ngrid"=>ngrid,
                                                 "nelement"=>nelement_global,
                                                 "nelement_local"=>nelement_local,
                                                 "L"=>L,
                                                 "discretization"=>discretization,
                                                 "bc"=>""))
    z_input = OptionsDict("coord" => OptionsDict("ngrid"=>ngrid,
                                                 "nelement"=>nelement_global,
                                                 "nelement_local"=>nelement_local,
                                                 "L"=>L,
                                                 "discretization"=>discretization,
                                                 "bc"=>"wall"))
    println("made inputs")
    # create the coordinate struct 'x'
    # This test runs effectively in serial, so use `ignore_MPI=true` to avoid
    # errors due to communicators not being fully set up.
    x, spectral = define_coordinate(x_input, "coord"; ignore_MPI=true)
    z, spectral = define_coordinate(z_input, "coord"; ignore_MPI=true)
    println("made x")
    # create arrays needed for Chebyshev pseudospectral treatment in x
    # and create the plans for the forward and backward fast Chebyshev
    # transforms
    if discretization == "chebyshev_pseudospectral"
    spectral = setup_chebyshev_pseudospectral(x)
    else
    spectral = false
end
println("made spectral")
    # create array for the function f(x) to be differentiated/integrated
    f = Array{Float64,1}(undef, x.n)
    # create array for the derivative df/dx
    df = Array{Float64,1}(undef, x.n)
    df_opt = Array{Float64,1}(undef, x.n)
df_exact = Array{Float64,1}(undef, x.n)
vz = Array{Float64,1}(undef,x.n)


xcut = 0.125
teststring = ".x2."*string(xcut)
## x^2 test 
for ix in 1:x.n
    vz[ix] = x.grid[ix] - xcut 
    if x.grid[ix] > xcut
        f[ix] = (x.grid[ix] - xcut)^2
        df_exact[ix] = 2.0*(x.grid[ix] - xcut)
    else
        f[ix] = 0.0
        df_exact[ix] = 0.0
    end
end
    
# differentiate f using standard Chebyshev method 
derivative!(df, f, x, spectral)
# differentiate f using Chebyshev with spline for subgrid differentiation
iz = 1 # a wall boundary point
derivative!(df_opt, f, x, spectral, iz, z, vz)

# plot df, df_opt & df_exact 
plot([x.grid,x.grid,x.grid], [vz, f,df_opt,df_exact], xlabel="x", ylabel="", label=["vz" "f" "df_opt" "df_exact"],
     shape =:circle, markersize = 5, linewidth=2)
outfile = outprefix*teststring*".onlyspline.pdf"
savefig(outfile)
plot([x.grid,x.grid,x.grid], [f,df,df_opt,df_exact], xlabel="x", ylabel="", label=["f" "df_cheb" "df_opt" "df_exact"],
     shape =:circle, markersize = 5, linewidth=2)
outfile = outprefix*teststring*".withspline.pdf"
savefig(outfile)
plot([x.grid,x.grid,x.grid], [f,df,df_exact], xlabel="x", ylabel="", label=["f" "df_cheb" "df_exact"],
     shape =:circle, markersize = 5, linewidth=2)
outfile = outprefix*teststring*".pdf"
savefig(outfile)
