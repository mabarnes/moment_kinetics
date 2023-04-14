using Printf
using Plots
using LaTeXStrings
using MPI
using SpecialFunctions: erf

function G_Maxwellian(Bmag,vpa,mu,ivpa,imu)
    # speed variable
    eta = sqrt(vpa.grid[ivpa]^2 + 2.0*Bmag*mu.grid[imu])
    zero = 1.0e-8
    if eta < zero
        G = 2.0/sqrt(pi)
    else 
        # G_M = (1/2 eta)*( eta erf'(eta) + (1 + 2 eta^2) erf(eta))
        G = (1.0/sqrt(pi))*exp(-eta^2) + ((0.5/eta) + eta)*erf(eta)
    end
    return G
end
function H_Maxwellian(Bmag,vpa,mu,ivpa,imu)
    # speed variable
    eta = sqrt(vpa.grid[ivpa]^2 + 2.0*Bmag*mu.grid[imu])
    zero = 1.0e-8
    if eta < zero
        # erf(eta)/eta ~ 2/sqrt(pi) + O(eta^2) for eta << 1 
        H = 2.0/sqrt(pi)
    else 
        # H_M =  erf(eta)/eta
        H = erf(eta)/eta
    end
    return H
end

if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")

    import moment_kinetics
	using moment_kinetics.input_structs: grid_input, advection_input
	using moment_kinetics.coordinates: define_coordinate
	using moment_kinetics.chebyshev: setup_chebyshev_pseudospectral
	using moment_kinetics.fokker_planck: evaluate_RMJ_collision_operator!
	using moment_kinetics.fokker_planck: calculate_Rosenbluth_potentials!
	using moment_kinetics.fokker_planck: calculate_Rosenbluth_H_from_G!
	using moment_kinetics.fokker_planck: init_fokker_planck_collisions
    using moment_kinetics.type_definitions: mk_float, mk_int
    
    discretization = "chebyshev_pseudospectral"
    #discretization = "finite_difference"
	
    # define inputs needed for the test
	vpa_ngrid = 17 #number of points per element 
	vpa_nelement_local = 5 # number of elements per rank
	vpa_nelement_global = vpa_nelement_local # total number of elements 
	vpa_L = 8.0 #physical box size in reference units 
	bc = "zero" 
	mu_ngrid = 17 #number of points per element 
	mu_nelement_local = 5 # number of elements per rank
	mu_nelement_global = mu_nelement_local # total number of elements 
    mu_L = 32.0 #physical box size in reference units 
	bc = "zero" 
    
    # fd_option and adv_input not actually used so given values unimportant
	fd_option = "fourth_order_centered"
	adv_input = advection_input("default", 1.0, 0.0, 0.0)
	nrank = 1
    irank = 0
    comm = MPI.COMM_NULL
	# create the 'input' struct containing input info needed to create a
	# coordinate
    vpa_input = grid_input("vpa", vpa_ngrid, vpa_nelement_global, vpa_nelement_local, 
		nrank, irank, vpa_L, discretization, fd_option, bc, adv_input,comm)
	mu_input = grid_input("mu", mu_ngrid, mu_nelement_global, mu_nelement_local, 
		nrank, irank, mu_L, discretization, fd_option, bc, adv_input,comm)
	
    # create the coordinate structs
	println("made inputs")
	vpa = define_coordinate(vpa_input)
	mu = define_coordinate(mu_input)
    vpa_spectral = setup_chebyshev_pseudospectral(vpa)
    mu_spectral = setup_chebyshev_pseudospectral(mu)
    
    # set up necessary inputs for collision operator functions 
    nmu = mu.n
    nvpa = vpa.n
    
    Bmag = 1.0
    cfreqssp = 1.0
    ms = 1.0
    msp = 1.0
    
    fkarrays = init_fokker_planck_collisions(mu, vpa)
    
    Cssp = Array{mk_float,2}(undef,nvpa,nmu)
    fs_in = Array{mk_float,2}(undef,nvpa,nmu)
    G_Maxwell = Array{mk_float,2}(undef,nvpa,nmu)
    H_Maxwell = Array{mk_float,2}(undef,nvpa,nmu)
    H_check = Array{mk_float,2}(undef,nvpa,nmu)
    G_err = Array{mk_float,2}(undef,nvpa,nmu)
    H_err = Array{mk_float,2}(undef,nvpa,nmu)
    H_check_err = Array{mk_float,2}(undef,nvpa,nmu)
    for imu in 1:nmu
        for ivpa in 1:nvpa
            fs_in[ivpa,imu] = exp( - vpa.grid[ivpa]^2 - 2.0*Bmag*mu.grid[imu] ) 
            G_Maxwell[ivpa,imu] = G_Maxwellian(Bmag,vpa,mu,ivpa,imu)
            H_Maxwell[ivpa,imu] = H_Maxwellian(Bmag,vpa,mu,ivpa,imu)
        end
    end
    
    fsp_in = fs_in 
    
    # evaluate the Rosenbluth potentials 
    @views calculate_Rosenbluth_potentials!(fkarrays.Rosenbluth_G, fkarrays.Rosenbluth_H, 
     fsp_in, mu, mu_spectral, vpa, vpa_spectral, Bmag,
     fkarrays.elliptic_integral_E_factor,fkarrays.elliptic_integral_K_factor,
     fkarrays.buffer_vpamu_1,fkarrays.buffer_vpamu_2,fkarrays.buffer_vpamu_3)
    @views calculate_Rosenbluth_H_from_G!(H_check,G_Maxwell,
     vpa,vpa_spectral,mu,mu_spectral,Bmag,
     fkarrays.buffer_vpamu_1,fkarrays.buffer_vpamu_2)
    @. G_err = abs(fkarrays.Rosenbluth_G - G_Maxwell)
    @. H_err = abs(fkarrays.Rosenbluth_H - H_Maxwell)
    @. H_check_err = abs(H_check - H_Maxwell)
    println("max(G_err)",maximum(G_err))
    println("max(H_err)",maximum(H_err))
    println("max(H_check_err)",maximum(H_check_err))
    zero = 1.0e-3
    #println(G_Maxwell[41,:])
    #println(G_Maxwell[:,1])
    for imu in 1:nmu
        for ivpa in 1:nvpa
            if (maximum(G_err[ivpa,imu]) > zero)
                #println("ivpa: ",ivpa," imu: ",imu," G_err: ",G_err[ivpa,imu])
                println("ivpa: ",ivpa," vpa: ",vpa.grid[ivpa]," imu: ",imu," mu: ",mu.grid[imu]," G_err: ",G_err[ivpa,imu]," G_Maxwell: ",G_Maxwell[ivpa,imu]," G_num: ",fkarrays.Rosenbluth_G[ivpa,imu])
                #println("ivpa: ",ivpa," vpa: ",vpa.grid[ivpa]," imu: ",imu," mu: ",mu.grid[imu]," G_err: ",G_err[ivpa,imu])
            end
            #H_err[ivpa,imu]
        end
    end
    #println(H_Maxwell[:,1])
    #println(fkarrays.Rosenbluth_H[:,1])
    #zero = 0.1
    for imu in 1:nmu
        for ivpa in 1:nvpa
            if (maximum(H_err[ivpa,imu]) > zero)
                #println("ivpa: ",ivpa," imu: ",imu," G_err: ",G_err[ivpa,imu])
                println("ivpa: ",ivpa," vpa: ",vpa.grid[ivpa]," imu: ",imu," mu: ",mu.grid[imu]," H_err: ",H_err[ivpa,imu]," H_Maxwell: ",H_Maxwell[ivpa,imu]," H_num: ",fkarrays.Rosenbluth_H[ivpa,imu])
                #println("ivpa: ",ivpa," vpa: ",vpa.grid[ivpa]," imu: ",imu," mu: ",mu.grid[imu]," G_err: ",G_err[ivpa,imu])
            end
            #H_err[ivpa,imu]
        end
    end
    
    # evaluate the collision operator
    @views evaluate_RMJ_collision_operator!(Cssp, fs_in, fsp_in, ms, msp, cfreqssp, 
     mu, vpa, mu_spectral, vpa_spectral, Bmag, fkarrays)
    
    zero = 1.0e1
    Cssp_err = maximum(abs.(Cssp))
    if Cssp_err > zero
        println("ERROR: C_ss'[F_Ms,F_Ms] /= 0")
        for imu in 1:mu.n
            for ivpa in 1:vpa.n
                if maximum(abs.(Cssp[ivpa,imu])) > zero
                    print("ivpa: ",ivpa," imu: ",imu," C: ")
                    #println("ivpa: ",ivpa," imu: ",imu," C: ", Cssp[ivpa,imu])
                    #println(" imu: ",imu," C[:,imu]:")
                    @printf("%.1e", Cssp[ivpa,imu])
                    println("")
                end
            end
        end
    end
    println("max(abs(C_ss'[F_Ms,F_Ms])): ", Cssp_err)
    
end 