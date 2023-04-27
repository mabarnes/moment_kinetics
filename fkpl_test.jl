using Printf
using Plots
using LaTeXStrings
using Measures
using MPI
using SpecialFunctions: erf

function eta_speed(vpa,vperp,ivpa,ivperp)
    eta = sqrt(vpa.grid[ivpa]^2 + vperp.grid[ivperp]^2)
    return eta
end

function G_Maxwellian(vpa,vperp,ivpa,ivperp)
    # speed variable
    eta = eta_speed(vpa,vperp,ivpa,ivperp)
    zero = 1.0e-10
    if eta < zero
        G = 2.0/sqrt(pi)
    else 
        # G_M = (1/2 eta)*( eta erf'(eta) + (1 + 2 eta^2) erf(eta))
        G = (1.0/sqrt(pi))*exp(-eta^2) + ((0.5/eta) + eta)*erf(eta)
    end
    return G
end
function H_Maxwellian(vpa,vperp,ivpa,ivperp)
    # speed variable
    eta = eta_speed(vpa,vperp,ivpa,ivperp)
    zero = 1.0e-10
    if eta < zero
        # erf(eta)/eta ~ 2/sqrt(pi) + O(eta^2) for eta << 1 
        H = 2.0/sqrt(pi)
    else 
        # H_M =  erf(eta)/eta
        H = erf(eta)/eta
    end
    return H
end

    #function Gamma_vpa_Maxwellian(Bmag,vpa,mu,ivpa,imu)
    #    #Gamma = 0.0
    #    #return Gamma
    #end
    #function Gamma_vpa_GMaxwellian(Bmag,vpa,mu,ivpa,imu)
    #    ## speed variable
    #    #eta = sqrt(vpa.grid[ivpa]^2 + 2.0*Bmag*mu.grid[imu])
    #    #
    #    #d2Gdeta2 = (erf(eta)/(eta^3)) - (2.0/sqrt(pi))*(exp(-eta^2)/(eta^2))
    #    #zero = 1.0e-10
    #    #if eta > zero
    #    #    #Gamma = -2.0*vpa.grid[ivpa]*exp(-eta^2)*d2Gdeta2
    #    #else
    #    #    #Gamma = 0.0
    #    #end
    #    #return Gamma
    #end
    #function Gamma_vpa_HMaxwellian(Bmag,vpa,mu,ivpa,imu)
    #    ## speed variable
    #    #eta = sqrt(vpa.grid[ivpa]^2 + 2.0*Bmag*mu.grid[imu])
    #    #
    #    #dHdeta = (2.0/sqrt(pi))*(exp(-eta^2)/eta) - (erf(eta)/(eta^2))
    #    #zero = 1.0e-10
    #    #if eta > zero
    #    #    #Gamma = -2.0*vpa.grid[ivpa]*exp(-eta^2)*(1.0/eta)*dHdeta
    #    #else
    #    #    #Gamma = 0.0
    #    #end
    #    #return Gamma
    #end

    #function Gamma_mu_Maxwellian(Bmag,vpa,mu,ivpa,imu)
    #    #Gamma = 0.0
    #    #return Gamma
    #end
    #function Gamma_mu_GMaxwellian(Bmag,vpa,mu,ivpa,imu)
    #    ## speed variable
    #    #eta = sqrt(vpa.grid[ivpa]^2 + 2.0*Bmag*mu.grid[imu])
    #    #
    #    #d2Gdeta2 = (erf(eta)/(eta^3)) - (2.0/sqrt(pi))*(exp(-eta^2)/(eta^2))
    #    #zero = 1.0e-10
    #    #if eta > zero
    #    #    #Gamma = -4.0*mu.grid[imu]*exp(-eta^2)*d2Gdeta2
    #    #else
    #    #    #Gamma = 0.0
    #    #end
    #    #return Gamma
    #end
    #function Gamma_mu_HMaxwellian(Bmag,vpa,mu,ivpa,imu)
    #    ## speed variable
    #    #eta = sqrt(vpa.grid[ivpa]^2 + 2.0*Bmag*mu.grid[imu])
    #    #
    #    #dHdeta = (2.0/sqrt(pi))*(exp(-eta^2)/eta) - (erf(eta)/(eta^2))
    #    #zero = 1.0e-10
    #    #if eta > zero
    #    #    #Gamma = -4.0*mu.grid[imu]*exp(-eta^2)*(1.0/eta)*dHdeta
    #    #else
    #    #    #Gamma = 0.0
    #    #end
    #    #return Gamma
    #end
   
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")

    import moment_kinetics
	using moment_kinetics.input_structs: grid_input, advection_input
	using moment_kinetics.coordinates: define_coordinate
	using moment_kinetics.chebyshev: setup_chebyshev_pseudospectral
	using moment_kinetics.fokker_planck: evaluate_RMJ_collision_operator!
	using moment_kinetics.fokker_planck: calculate_Rosenbluth_potentials!
	#using moment_kinetics.fokker_planck: calculate_Rosenbluth_H_from_G!
	using moment_kinetics.fokker_planck: init_fokker_planck_collisions
    using moment_kinetics.type_definitions: mk_float, mk_int
    
    discretization = "chebyshev_pseudospectral"
    #discretization = "finite_difference"
	
    # define inputs needed for the test
	vpa_ngrid = 3 #number of points per element 
	vpa_nelement_local = 3 # number of elements per rank
	vpa_nelement_global = vpa_nelement_local # total number of elements 
	vpa_L = 6.0 #physical box size in reference units 
	bc = "zero" 
	vperp_ngrid = 3 #number of points per element 
	vperp_nelement_local = 3 # number of elements per rank
	vperp_nelement_global = vperp_nelement_local # total number of elements 
    vperp_L = 6.0 #physical box size in reference units 
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
	vperp_input = grid_input("vperp", vperp_ngrid, vperp_nelement_global, vperp_nelement_local, 
		nrank, irank, vperp_L, discretization, fd_option, bc, adv_input,comm)
	
    # create the coordinate structs
	println("made inputs")
	vpa = define_coordinate(vpa_input)
	vperp = define_coordinate(vperp_input)
    println(vperp.grid)
    println(vperp.wgts)
    vpa_spectral = setup_chebyshev_pseudospectral(vpa)
    vperp_spectral = setup_chebyshev_pseudospectral(vperp)
    
    # set up necessary inputs for collision operator functions 
    nvperp = vperp.n
    nvpa = vpa.n
    
    cfreqssp = 1.0
    ms = 1.0
    msp = 1.0
    
    fkarrays = init_fokker_planck_collisions(vperp, vpa)
    
    Cssp = Array{mk_float,2}(undef,nvpa,nvperp)
    fs_in = Array{mk_float,2}(undef,nvpa,nvperp)
    G_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
    H_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
    #H_check = Array{mk_float,2}(undef,nvpa,nvperp)
    G_err = Array{mk_float,2}(undef,nvpa,nvperp)
    H_err = Array{mk_float,2}(undef,nvpa,nvperp)
    #H_check_err = Array{mk_float,2}(undef,nvpa,nvperp)
    #Gam_vperp_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
    #Gam_vperp_err = Array{mk_float,2}(undef,nvpa,nvperp)
    #Gam_vperp_GMaxwell = Array{mk_float,2}(undef,nvpa,nvperp)
    #Gam_vperp_HMaxwell = Array{mk_float,2}(undef,nvpa,nvperp)
    #Gam_vperp_Gerr = Array{mk_float,2}(undef,nvpa,nvperp)
    #Gam_vperp_Herr = Array{mk_float,2}(undef,nvpa,nvperp)
    #Gam_vpa_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
    #Gam_vpa_err = Array{mk_float,2}(undef,nvpa,nvperp)
    #Gam_vpa_GMaxwell = Array{mk_float,2}(undef,nvpa,nvperp)
    #Gam_vpa_HMaxwell = Array{mk_float,2}(undef,nvpa,nvperp)
    #Gam_vpa_Gerr = Array{mk_float,2}(undef,nvpa,nvperp)
    #Gam_vpa_Herr = Array{mk_float,2}(undef,nvpa,nvperp)

    for ivperp in 1:nvperp
        for ivpa in 1:nvpa
            fs_in[ivpa,ivperp] = exp( - vpa.grid[ivpa]^2 - vperp.grid[ivperp]^2 ) 
            G_Maxwell[ivpa,ivperp] = G_Maxwellian(vpa,vperp,ivpa,ivperp)
            H_Maxwell[ivpa,ivperp] = H_Maxwellian(vpa,vperp,ivpa,ivperp)
        end
    end
    
    fsp_in = fs_in 
    
    # evaluate the Rosenbluth potentials 
    @views calculate_Rosenbluth_potentials!(fkarrays.Rosenbluth_G, fkarrays.Rosenbluth_H, 
     fsp_in, fkarrays.elliptic_integral_E_factor,fkarrays.elliptic_integral_K_factor,
     fkarrays.buffer_vpavperp_1,vperp,vpa)
     
    #@views calculate_Rosenbluth_H_from_G!(H_check,G_Maxwell,
    # vpa,vpa_spectral,vperp,vperp_spectral,Bmag,
    # fkarrays.buffer_vpavperp_1,fkarrays.buffer_vpavperp_2)
    @. G_err = abs(fkarrays.Rosenbluth_G - G_Maxwell)
    @. H_err = abs(fkarrays.Rosenbluth_H - H_Maxwell)
    #@. H_check_err = abs(H_check - H_Maxwell)
    println("max(G_err)",maximum(G_err))
    println("max(H_err)",maximum(H_err))
    #println("max(H_check_err)",maximum(H_check_err))
    zero = 1.0e-3
    #println(G_Maxwell[41,:])
    #println(G_Maxwell[:,1])
    for ivperp in 1:nvperp
        for ivpa in 1:nvpa
            if (maximum(G_err[ivpa,ivperp]) > zero)
                #println("ivpa: ",ivpa," ivperp: ",ivperp," G_err: ",G_err[ivpa,ivperp])
                #println("ivpa: ",ivpa," vpa: ",vpa.grid[ivpa]," ivperp: ",ivperp," vperp: ",vperp.grid[ivperp]," G_err: ",G_err[ivpa,ivperp]," G_Maxwell: ",G_Maxwell[ivpa,ivperp]," G_num: ",fkarrays.Rosenbluth_G[ivpa,ivperp])
                #println("ivpa: ",ivpa," vpa: ",vpa.grid[ivpa]," ivperp: ",ivperp," vperp: ",vperp.grid[ivperp]," G_err: ",G_err[ivpa,ivperp])
            end
            #H_err[ivpa,ivperp]
        end
    end
    #println(H_Maxwell[:,1])
    #println(fkarrays.Rosenbluth_H[:,1])
    #zero = 0.1
    for ivperp in 1:nvperp
        for ivpa in 1:nvpa
            if (maximum(H_err[ivpa,ivperp]) > zero)
                #println("ivpa: ",ivpa," ivperp: ",ivperp," H_err: ",H_err[ivpa,ivperp])
                println("ivpa: ",ivpa," vpa: ",vpa.grid[ivpa]," ivperp: ",ivperp," vperp: ",vperp.grid[ivperp]," H_err: ",H_err[ivpa,ivperp]," H_Maxwell: ",H_Maxwell[ivpa,ivperp]," H_num: ",fkarrays.Rosenbluth_H[ivpa,ivperp])
            end
            #H_err[ivpa,ivperp]
        end
    end
    
    ## evaluate the collision operator with numerically computed G & H 
    #println("TEST: Css'[F_M,F_M] with numerical G[F_M] & H[F_M]")
    #@views evaluate_RMJ_collision_operator!(Cssp, fs_in, fsp_in, ms, msp, cfreqssp, 
    # mu, vpa, mu_spectral, vpa_spectral, Bmag, fkarrays)
    #
    #zero = 1.0e1
    #Cssp_err = maximum(abs.(Cssp))
    #if Cssp_err > zero
    #    #println("ERROR: C_ss'[F_Ms,F_Ms] /= 0")
    #    #for imu in 1:mu.n
    #    #    #for ivpa in 1:vpa.n
    #    #    #    #if maximum(abs.(Cssp[ivpa,imu])) > zero
    #    #    #    #    ###print("ivpa: ",ivpa," imu: ",imu," C: ")
    #    #    #    #    ##println("ivpa: ",ivpa," imu: ",imu," C: ", Cssp[ivpa,imu])
    #    #    #    #    ##println(" imu: ",imu," C[:,imu]:")
    #    #    #    #    ###@printf("%.1e", Cssp[ivpa,imu])
    #    #    #    #    ###println("")
    #    #    #    #end
    #    #    #end
    #    #end
    #end
    #println("max(abs(C_ss'[F_Ms,F_Ms])): ", Cssp_err)
    #
    #
    ## evaluate the collision operator with analytically computed G & H 
    #zero = 1.0e-6
    #println("TEST: Css'[F_M,F_M] with analytical G[F_M] & H[F_M]")
    #@views @. fkarrays.Rosenbluth_G = G_Maxwell
    #@views @. fkarrays.Rosenbluth_H = H_Maxwell
    #@views evaluate_RMJ_collision_operator!(Cssp, fs_in, fsp_in, ms, msp, cfreqssp, 
    # mu, vpa, mu_spectral, vpa_spectral, Bmag, fkarrays)
    #
    #for imu in 1:mu.n
    #    #for ivpa in 1:vpa.n
    #    #    #Gam_vpa_Maxwell[ivpa,imu] = Gamma_vpa_Maxwellian(Bmag,vpa,mu,ivpa,imu)
    #    #    #Gam_mu_Maxwell[ivpa,imu] = Gamma_mu_Maxwellian(Bmag,vpa,mu,ivpa,imu)
    #    #    #
    #    #    #Gam_vpa_err[ivpa,imu] = abs(fkarrays.Gamma_vpa[ivpa,imu] - Gam_vpa_Maxwell[ivpa,imu])
    #    #    #Gam_mu_err[ivpa,imu] = abs(fkarrays.Gamma_mu[ivpa,imu] - Gam_mu_Maxwell[ivpa,imu])
    #    #end
    #end
    #max_Gam_vpa_err = maximum(Gam_vpa_err)
    #println("max(abs(Gamma_vpa[F_Ms,F_Ms])): ", max_Gam_vpa_err)
    #if max_Gam_vpa_err > zero
    #    #for imu in 1:mu.n
    #    #    #for ivpa in 1:vpa.n
    #    #    #    #if Gam_vpa_err[ivpa,imu] > zero 
    #    #    #    #    ##println("ivpa: ",ivpa," imu: ",imu," Gam_vpa_err: ",Gam_vpa_err[ivpa,imu]," Gam_vpa_num: ",fkarrays.Gamma_vpa[ivpa,imu]," Gam_vpa_Maxwell: ",Gam_vpa_Maxwell[ivpa,imu])
    #    #    #    #end
    #    #    #end
    #    #end
    #    #@views heatmap(mu.grid, vpa.grid, Gam_vpa_Maxwell[:,:], xlabel=L"\mu", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
    #    #    #    #windowsize = (360,240), margin = 15pt)
    #    #    #    #outfile = string("fkpl_Gam_vpa_Maxwell.pdf")
    #    #    #    #savefig(outfile)
    #    #@views heatmap(mu.grid, vpa.grid, fkarrays.Gamma_vpa[:,:], xlabel=L"\mu", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
    #    #    #    #windowsize = (360,240), margin = 15pt)
    #    #    #    #outfile = string("fkpl_Gam_vpa_num.pdf")
    #    #    #    #savefig(outfile)
    #end
    #max_Gam_mu_err = maximum(Gam_mu_err)
    #println("max(abs(Gamma_mu[F_Ms,F_Ms])): ", max_Gam_mu_err)
    #if max_Gam_mu_err > zero
    #    #for imu in 1:mu.n
    #    #    #for ivpa in 1:vpa.n
    #    #    #    #if Gam_mu_err[ivpa,imu] > zero 
    #    #    #    #    ##println("ivpa: ",ivpa," imu: ",imu," Gam_mu_err: ",Gam_mu_err[ivpa,imu]," Gam_mu_num: ",fkarrays.Gamma_mu[ivpa,imu]," Gam_mu_Maxwell: ",Gam_mu_Maxwell[ivpa,imu])
    #    #    #    #end
    #    #    #end
    #    #end
    #    #@views heatmap(mu.grid, vpa.grid, Gam_mu_Maxwell[:,:], xlabel=L"\mu", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
    #    #    #    #windowsize = (360,240), margin = 15pt)
    #    #    #    #outfile = string("fkpl_Gam_mu_Maxwell.pdf")
    #    #    #    #savefig(outfile)
    #    #@views heatmap(mu.grid, vpa.grid, fkarrays.Gamma_mu[:,:], xlabel=L"\mu", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
    #    #    #    #windowsize = (360,240), margin = 15pt)
    #    #    #    #outfile = string("fkpl_Gam_mu_num.pdf")
    #    #    #    #savefig(outfile)
    #end
    #
    #for imu in 1:mu.n
    #    #for ivpa in 1:vpa.n
    #    #    #Gam_vpa_GMaxwell[ivpa,imu] = Gamma_vpa_GMaxwellian(Bmag,vpa,mu,ivpa,imu)
    #    #    #Gam_mu_GMaxwell[ivpa,imu] = Gamma_mu_GMaxwellian(Bmag,vpa,mu,ivpa,imu)
    #    #    #
    #    #    #Gam_vpa_Gerr[ivpa,imu] = abs(fkarrays.Gamma_vpa_G[ivpa,imu] - Gam_vpa_GMaxwell[ivpa,imu])
    #    #    #Gam_mu_Gerr[ivpa,imu] = abs(fkarrays.Gamma_mu_G[ivpa,imu] - Gam_mu_GMaxwell[ivpa,imu])
    #    #end
    #end
    #max_Gam_vpa_Gerr = maximum(Gam_vpa_Gerr)
    #println("max(abs(Gamma_vpa_G[F_Ms,F_Ms])): ", max_Gam_vpa_Gerr)
    #if max_Gam_vpa_Gerr > zero
    #    #for imu in 1:mu.n
    #    #    #for ivpa in 1:vpa.n
    #    #    #    #if Gam_vpa_Gerr[ivpa,imu] > zero 
    #    #    #    #    ##println("ivpa: ",ivpa," imu: ",imu," Gam_vpa_Gerr: ",Gam_vpa_Gerr[ivpa,imu]," Gam_vpa_G_num: ",fkarrays.Gamma_vpa_G[ivpa,imu]," Gam_vpa_GMaxwell: ",Gam_vpa_GMaxwell[ivpa,imu])
    #    #    #    #end
    #    #    #end
    #    #end
    #    #@views heatmap(mu.grid, vpa.grid, Gam_vpa_GMaxwell[:,:], xlabel=L"\mu", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
    #    #    #    #windowsize = (360,240), margin = 15pt)
    #    #    #    #outfile = string("fkpl_Gam_vpa_GMaxwell.pdf")
    #    #    #    #savefig(outfile)
    #    #@views heatmap(mu.grid, vpa.grid, fkarrays.Gamma_vpa_G[:,:], xlabel=L"\mu", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
    #    #    #    #windowsize = (360,240), margin = 15pt)
    #    #    #    #outfile = string("fkpl_Gam_vpa_G_num.pdf")
    #    #    #    #savefig(outfile)
    #end
    #max_Gam_mu_Gerr = maximum(Gam_mu_Gerr)
    #println("max(abs(Gamma_mu_G[F_Ms,F_Ms])): ", max_Gam_mu_Gerr)
    #if max_Gam_mu_Gerr > zero
    #    #for imu in 1:mu.n
    #    #    #for ivpa in 1:vpa.n
    #    #    #    #if Gam_mu_Gerr[ivpa,imu] > zero 
    #    #    #    #    ##println("ivpa: ",ivpa," imu: ",imu," Gam_mu_Gerr: ",Gam_mu_Gerr[ivpa,imu]," Gam_mu_G_num: ",fkarrays.Gamma_mu_G[ivpa,imu]," Gam_mu_GMaxwell: ",Gam_mu_GMaxwell[ivpa,imu])
    #    #    #    #end
    #    #    #end
    #    #end
    #    #@views heatmap(mu.grid, vpa.grid, Gam_mu_GMaxwell[:,:], xlabel=L"\mu", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
    #    #    #    #windowsize = (360,240), margin = 15pt)
    #    #    #    #outfile = string("fkpl_Gam_mu_GMaxwell.pdf")
    #    #    #    #savefig(outfile)
    #    #@views heatmap(mu.grid, vpa.grid, fkarrays.Gamma_mu_G[:,:], xlabel=L"\mu", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
    #    #    #    #windowsize = (360,240), margin = 15pt)
    #    #    #    #outfile = string("fkpl_Gam_mu_G_num.pdf")
    #    #    #    #savefig(outfile)
    #end
    #
    #for imu in 1:mu.n
    #    #for ivpa in 1:vpa.n
    #    #    #Gam_vpa_HMaxwell[ivpa,imu] = Gamma_vpa_HMaxwellian(Bmag,vpa,mu,ivpa,imu)
    #    #    #Gam_mu_HMaxwell[ivpa,imu] = Gamma_mu_HMaxwellian(Bmag,vpa,mu,ivpa,imu)
    #    #    #
    #    #    #Gam_vpa_Herr[ivpa,imu] = abs(fkarrays.Gamma_vpa_H[ivpa,imu] - Gam_vpa_HMaxwell[ivpa,imu])
    #    #    #Gam_mu_Herr[ivpa,imu] = abs(fkarrays.Gamma_mu_H[ivpa,imu] - Gam_mu_HMaxwell[ivpa,imu])
    #    #end
    #end
    #max_Gam_vpa_Herr = maximum(Gam_vpa_Herr)
    #println("max(abs(Gamma_vpa_H[F_Ms,F_Ms])): ", max_Gam_vpa_Herr)
    #if max_Gam_vpa_Herr > zero
    #    #for imu in 1:mu.n
    #    #    #for ivpa in 1:vpa.n
    #    #    #    #if Gam_vpa_Herr[ivpa,imu] > zero 
    #    #    #    #    ##println("ivpa: ",ivpa," imu: ",imu," Gam_vpa_Herr: ",Gam_vpa_Herr[ivpa,imu]," Gam_vpa_H_num: ",fkarrays.Gamma_vpa_H[ivpa,imu]," Gam_vpa_HMaxwell: ",Gam_vpa_HMaxwell[ivpa,imu])
    #    #    #    #end
    #    #    #end
    #    #end
    #    #@views heatmap(mu.grid, vpa.grid, Gam_vpa_HMaxwell[:,:], xlabel=L"\mu", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
    #    #    #    #windowsize = (360,240), margin = 15pt)
    #    #    #    #outfile = string("fkpl_Gam_vpa_HMaxwell.pdf")
    #    #    #    #savefig(outfile)
    #    #@views heatmap(mu.grid, vpa.grid, fkarrays.Gamma_vpa_H[:,:], xlabel=L"\mu", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
    #    #    #    #windowsize = (360,240), margin = 15pt)
    #    #    #    #outfile = string("fkpl_Gam_vpa_H_num.pdf")
    #    #    #    #savefig(outfile)
    #end
    #max_Gam_mu_Herr = maximum(Gam_mu_Herr)
    #println("max(abs(Gamma_mu_H[F_Ms,F_Ms])): ", max_Gam_mu_Herr)
    #if max_Gam_mu_Herr > zero
    #    #for imu in 1:mu.n
    #    #    #for ivpa in 1:vpa.n
    #    #    #    #if Gam_mu_Herr[ivpa,imu] > zero 
    #    #    #    #    ##println("ivpa: ",ivpa," imu: ",imu," Gam_mu_Herr: ",Gam_mu_Herr[ivpa,imu]," Gam_mu_H_num: ",fkarrays.Gamma_mu_H[ivpa,imu]," Gam_mu_HMaxwell: ",Gam_mu_HMaxwell[ivpa,imu])
    #    #    #    #end
    #    #    #end
    #    #end
    #    #@views heatmap(mu.grid, vpa.grid, Gam_mu_HMaxwell[:,:], xlabel=L"\mu", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
    #    #    #    #windowsize = (360,240), margin = 15pt)
    #    #    #    #outfile = string("fkpl_Gam_mu_HMaxwell.pdf")
    #    #    #    #savefig(outfile)
    #    #@views heatmap(mu.grid, vpa.grid, fkarrays.Gamma_mu_H[:,:], xlabel=L"\mu", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
    #    #    #    #windowsize = (360,240), margin = 15pt)
    #    #    #    #outfile = string("fkpl_Gam_mu_H_num.pdf")
    #    #    #    #savefig(outfile)
    #end
    #
    #Cssp_err = maximum(abs.(Cssp))
    #if Cssp_err > zero
    #    #println("ERROR: C_ss'[F_Ms,F_Ms] /= 0")
    #    #for imu in 1:mu.n
    #    #    #for ivpa in 1:vpa.n
    #    #    #    #if maximum(abs.(Cssp[ivpa,imu])) > zero
    #    #    #    #    #print("ivpa: ",ivpa," imu: ",imu," C: ")
    #    #    #    #    ##println("ivpa: ",ivpa," imu: ",imu," C: ", Cssp[ivpa,imu])
    #    #    #    #    ##println(" imu: ",imu," C[:,imu]:")
    #    #    #    #    #@printf("%.1e", Cssp[ivpa,imu])
    #    #    #    #    #println("")
    #    #    #    #end
    #    #    #end
    #    #end
    #    #@views heatmap(mu.grid, vpa.grid, Cssp[:,:], xlabel=L"\mu", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
    #    #    #    #windowsize = (360,240), margin = 15pt)
    #    #    #    #outfile = string("fkpl_Cssp_num.pdf")
    #    #    #    #savefig(outfile)
    #end
    #println("max(abs(C_ss'[F_Ms,F_Ms])): ", Cssp_err)
    
end 
