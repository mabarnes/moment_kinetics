using Printf
using Plots
using LaTeXStrings
using MPI

if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")

    import moment_kinetics
	using moment_kinetics.input_structs: grid_input, advection_input
	using moment_kinetics.coordinates: define_coordinate
	using moment_kinetics.chebyshev: setup_chebyshev_pseudospectral
	using moment_kinetics.fokker_planck: evaluate_RMJ_collision_operator!
	using moment_kinetics.fokker_planck: init_fokker_planck_collisions
    using moment_kinetics.type_definitions: mk_float, mk_int
    
    discretization = "chebyshev_pseudospectral"
    #discretization = "finite_difference"
	
    # define inputs needed for the test
	vpa_ngrid = 17 #number of points per element 
	vpa_nelement_local = 2 # number of elements per rank
	vpa_nelement_global = vpa_nelement_local # total number of elements 
	vpa_L = 8.0 #physical box size in reference units 
	bc = "zero" 
	mu_ngrid = 17 #number of points per element 
	mu_nelement_local = 2 # number of elements per rank
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
    for imu in 1:nmu
        for ivpa in 1:nvpa
            fs_in[ivpa,imu] = exp( - vpa.grid[ivpa]^2 - 2.0*Bmag*mu.grid[imu] ) 
        end
    end
    
    fsp_in = fs_in 
    
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