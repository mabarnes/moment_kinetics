using Printf
using Plots
using LaTeXStrings
using Measures
using MPI
using SpecialFunctions: erf, ellipe, ellipk
using FastGaussQuadrature
using Dates

import moment_kinetics
using moment_kinetics.input_structs: grid_input, advection_input
using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.chebyshev: setup_chebyshev_pseudospectral
using moment_kinetics.gauss_legendre: setup_gausslegendre_pseudospectral
using moment_kinetics.fokker_planck: evaluate_RMJ_collision_operator!
using moment_kinetics.fokker_planck: calculate_Rosenbluth_potentials!
#using moment_kinetics.fokker_planck: calculate_Rosenbluth_H_from_G!
using moment_kinetics.fokker_planck: init_fokker_planck_collisions
using moment_kinetics.fokker_planck: calculate_collisional_fluxes, calculate_Maxwellian_Rosenbluth_coefficients
using moment_kinetics.fokker_planck: Cflux_vpa_Maxwellian_inputs, Cflux_vperp_Maxwellian_inputs
using moment_kinetics.fokker_planck: calculate_Rosenbluth_H_from_G!
using moment_kinetics.fokker_planck: d2Gdvpa2, dGdvperp, d2Gdvperpdvpa, d2Gdvperp2
using moment_kinetics.fokker_planck: dHdvpa, dHdvperp, Cssp_Maxwellian_inputs, F_Maxwellian
using moment_kinetics.type_definitions: mk_float, mk_int
using moment_kinetics.calculus: derivative!
using moment_kinetics.velocity_moments: get_density, get_upar, get_ppar, get_pperp, get_pressure
using moment_kinetics.communication
using moment_kinetics.looping
using moment_kinetics.array_allocation: allocate_shared_float

function eta_speed(upar,vth,vpa,vperp,ivpa,ivperp)
    eta = sqrt((vpa.grid[ivpa]-upar)^2 + vperp.grid[ivperp]^2)/vth
    return eta
end

function G_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
    # speed variable
    eta = eta_speed(upar,vth,vpa,vperp,ivpa,ivperp)
    zero = 1.0e-10
    if eta < zero
        G = 2.0/sqrt(pi)
    else 
        # G_M = (1/2 eta)*( eta erf'(eta) + (1 + 2 eta^2) erf(eta))
        G = (1.0/sqrt(pi))*exp(-eta^2) + ((0.5/eta) + eta)*erf(eta)
    end
    return G*dens*vth
end
function H_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
    # speed variable
    eta = eta_speed(upar,vth,vpa,vperp,ivpa,ivperp)
    zero = 1.0e-10
    if eta < zero
        # erf(eta)/eta ~ 2/sqrt(pi) + O(eta^2) for eta << 1 
        H = 2.0/sqrt(pi)
    else 
        # H_M =  erf(eta)/eta
        H = erf(eta)/eta
    end
    return H*dens/vth
end

function pressure(ppar,pperp)
    pres = (1.0/3.0)*(ppar + 2.0*pperp) 
    return pres
end

function get_vth(pres,dens,mass)
        return sqrt(pres/(dens*mass))
end

function expected_nelement_scaling!(expected,nelement_list,ngrid,nscan)
    for iscan in 1:nscan
        expected[iscan] = (1.0/nelement_list[iscan])^(ngrid - 1)
    end
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

    function calculate_d2Gdvpa2!(d2Gdvpa2,G,vpa,vpa_spectral,vperp,vperp_spectral)
        for ivperp in 1:vperp.n
            @views derivative!(vpa.scratch, G[:,ivperp], vpa, vpa_spectral)
            @views derivative!(vpa.scratch2, vpa.scratch, vpa, vpa_spectral)
            @. d2Gdvpa2[:,ivperp] = vpa.scratch2
        end
    end
    
    function calculate_d2Gdvperpdvpa!(d2Gdvperpdvpa,G,vpa,vpa_spectral,vperp,vperp_spectral, buffer_vpavperp)
        for ivpa in 1:vpa.n
            @views derivative!(vperp.scratch, G[ivpa,:], vperp, vperp_spectral)
            @. buffer_vpavperp[ivpa,:] = vperp.scratch
        end
        for ivperp in 1:vperp.n
            @views derivative!(vpa.scratch, buffer_vpavperp[:,ivperp], vpa, vpa_spectral)
            @. d2Gdvperpdvpa[:,ivperp] = vpa.scratch
        end
    end
    
    function calculate_d2Gdvperp2!(d2Gdvperp2,G,vpa,vpa_spectral,vperp,vperp_spectral)
        for ivpa in 1:vpa.n
            @views derivative!(vperp.scratch, G[ivpa,:], vperp, vperp_spectral)
            @views derivative!(vperp.scratch2, vperp.scratch, vperp, vperp_spectral)
            @. d2Gdvperp2[ivpa,:] = vperp.scratch2
        end
    end
    
    function calculate_dHdvpa!(dHdvpa,H,vpa,vpa_spectral,vperp,vperp_spectral)
        for ivperp in 1:vperp.n
            @views derivative!(vpa.scratch, H[:,ivperp], vpa, vpa_spectral)
            @. dHdvpa[:,ivperp] = vpa.scratch
        end
    end

    function calculate_dHdvperp!(dHdvperp,H,vpa,vpa_spectral,vperp,vperp_spectral)
        for ivpa in 1:vpa.n
            @views derivative!(vperp.scratch, H[ivpa,:], vperp, vperp_spectral)
            @. dHdvperp[ivpa,:] = vperp.scratch
        end
    end
    
    function init_grids(nelement,ngrid)
        discretization = "gausslegendre_pseudospectral"
        #discretization = "chebyshev_pseudospectral"
        #discretization = "finite_difference"
        
        # define inputs needed for the test
        vpa_ngrid = ngrid #number of points per element 
        vpa_nelement_local = nelement # number of elements per rank
        vpa_nelement_global = vpa_nelement_local # total number of elements 
        vpa_L = 12.0 #physical box size in reference units 
        bc = "zero" 
        vperp_ngrid = ngrid #number of points per element 
        vperp_nelement_local = nelement # number of elements per rank
        vperp_nelement_global = vperp_nelement_local # total number of elements 
        vperp_L = 6.0 #physical box size in reference units 
        bc = "zero" 
        
        # fd_option and adv_input not actually used so given values unimportant
        fd_option = "fourth_order_centered"
        cheb_option = "matrix"
        adv_input = advection_input("default", 1.0, 0.0, 0.0)
        nrank = 1
        irank = 0
        comm = MPI.COMM_NULL
        # create the 'input' struct containing input info needed to create a
        # coordinate
        vpa_input = grid_input("vpa", vpa_ngrid, vpa_nelement_global, vpa_nelement_local, 
            nrank, irank, vpa_L, discretization, fd_option, cheb_option, bc, adv_input,comm)
        vperp_input = grid_input("vperp", vperp_ngrid, vperp_nelement_global, vperp_nelement_local, 
            nrank, irank, vperp_L, discretization, fd_option, cheb_option, bc, adv_input,comm)
        
        # create the coordinate structs
        #println("made inputs")
        vpa = define_coordinate(vpa_input)
        vperp = define_coordinate(vperp_input)
        #println(vperp.grid)
        #println(vperp.wgts)
        if discretization == "chebyshev_pseudospectral" 
            vpa_spectral = setup_chebyshev_pseudospectral(vpa)
            vperp_spectral = setup_chebyshev_pseudospectral(vperp)
            #println("using chebyshev_pseudospectral")
        elseif discretization == "gausslegendre_pseudospectral"
            vpa_spectral = setup_gausslegendre_pseudospectral(vpa)
            vperp_spectral = setup_gausslegendre_pseudospectral(vperp)
            #println("using gausslegendre_pseudospectral")
        end
        return vpa, vperp, vpa_spectral, vperp_spectral
    end
    
    test_Rosenbluth_integrals = false#true
    test_collision_operator_fluxes = false#true 
    test_Lagrange_integral = false #true
    test_Lagrange_integral_scan = true
    #ngrid = 9
    #nelement = 8 
    
    function test_Rosenbluth_potentials(nelement,ngrid;numerical_G = false)
        vpa, vperp, vpa_spectral, vperp_spectral =  init_grids(nelement,ngrid)
        # set up necessary inputs for collision operator functions 
        nvperp = vperp.n
        nvpa = vpa.n
        
        fs_in = Array{mk_float,2}(undef,nvpa,nvperp)
        G_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
        H_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
        H_check = Array{mk_float,2}(undef,nvpa,nvperp)
        G_err = Array{mk_float,2}(undef,nvpa,nvperp)
        H_err = Array{mk_float,2}(undef,nvpa,nvperp)
        H_check_err = Array{mk_float,2}(undef,nvpa,nvperp)
        
        dHdvpa_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
        dHdvpa_check = Array{mk_float,2}(undef,nvpa,nvperp)
        dHdvpa_err = Array{mk_float,2}(undef,nvpa,nvperp)
        dHdvperp_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
        dHdvperp_check = Array{mk_float,2}(undef,nvpa,nvperp)
        dHdvperp_err = Array{mk_float,2}(undef,nvpa,nvperp)
        d2Gdvpa2_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
        d2Gdvpa2_check = Array{mk_float,2}(undef,nvpa,nvperp)
        d2Gdvpa2_err = Array{mk_float,2}(undef,nvpa,nvperp)
        d2Gdvperp2_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
        d2Gdvperp2_check = Array{mk_float,2}(undef,nvpa,nvperp)
        d2Gdvperp2_err = Array{mk_float,2}(undef,nvpa,nvperp)
        d2Gdvperpdvpa_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
        d2Gdvperpdvpa_check = Array{mk_float,2}(undef,nvpa,nvperp)
        d2Gdvperpdvpa_err = Array{mk_float,2}(undef,nvpa,nvperp)
        
        
        # set up test Maxwellian
        dens = 3.0/4.0
        upar = 2.0/3.0
        ppar = 2.0/3.0
        pperp = 2.0/3.0
        pres = get_pressure(ppar,pperp) 
        mi = 1.0
        vths = get_vth(pres,dens,mi)
        
        n_ion_species = 1
        dens_in = Array{mk_float,1}(undef,n_ion_species)
        upar_in = Array{mk_float,1}(undef,n_ion_species)
        vths_in = Array{mk_float,1}(undef,n_ion_species)
        dens_in[1] = dens
        upar_in[1] = upar
        vths_in[1] = vths
        fkarrays = init_fokker_planck_collisions(vperp, vpa, init_integral_factors = true)
        

        for ivperp in 1:nvperp
            for ivpa in 1:nvpa
                fs_in[ivpa,ivperp] = (dens/vths^3)*exp( - ((vpa.grid[ivpa]-upar)^2 + vperp.grid[ivperp]^2)/vths^2 ) 
                G_Maxwell[ivpa,ivperp] = G_Maxwellian(dens,upar,vths,vpa,vperp,ivpa,ivperp)
                H_Maxwell[ivpa,ivperp] = H_Maxwellian(dens,upar,vths,vpa,vperp,ivpa,ivperp)
            end
        end
        
        fsp_in = fs_in 
        
        # evaluate the Rosenbluth potentials 
        @views calculate_Rosenbluth_potentials!(fkarrays.Rosenbluth_G, fkarrays.Rosenbluth_H, 
         fsp_in, fkarrays.elliptic_integral_E_factor,fkarrays.elliptic_integral_K_factor,
         fkarrays.buffer_vpavperp_1,vperp,vpa)
        
        #@views heatmap(vperp.grid, vpa.grid, fkarrays.Rosenbluth_G[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
        #     windowsize = (360,240), margin = 15pt)
        #     outfile = string("fkpl_RosenG.pdf")
        #     savefig(outfile)
        #@views heatmap(vperp.grid, vpa.grid, G_Maxwell[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
        #     windowsize = (360,240), margin = 15pt)
        #     outfile = string("fkpl_G_Maxwell.pdf")
        #     savefig(outfile)
       
        G_in = G_Maxwell # use analytical G to calculate a H for the check
        @views calculate_Rosenbluth_H_from_G!(H_check,G_in,
         vpa,vpa_spectral,vperp,vperp_spectral,
         fkarrays.buffer_vpavperp_1,fkarrays.buffer_vpavperp_2)
        
        if numerical_G 
            G_in = fkarrays.Rosenbluth_G
            H_in = fkarrays.Rosenbluth_H
            @views calculate_Rosenbluth_H_from_G!(H_in,G_in,
                vpa,vpa_spectral,vperp,vperp_spectral,
                fkarrays.buffer_vpavperp_1,fkarrays.buffer_vpavperp_2)
            # using numerical G, errors are much worse than using smooth analytical G
        else 
            G_in = G_Maxwell # use analytical G for coeffs 
            H_in = H_Maxwell # use analytical H for coeffs 
        end
        @views calculate_d2Gdvpa2!(d2Gdvpa2_check,G_in,vpa,vpa_spectral,vperp,vperp_spectral) 
        @views calculate_d2Gdvperpdvpa!(d2Gdvperpdvpa_check,G_in,vpa,vpa_spectral,vperp,vperp_spectral, fkarrays.buffer_vpavperp_1) 
        @views calculate_d2Gdvperp2!(d2Gdvperp2_check,G_in,vpa,vpa_spectral,vperp,vperp_spectral) 
        @views calculate_dHdvperp!(dHdvperp_check,H_in,vpa,vpa_spectral,vperp,vperp_spectral) 
        @views calculate_dHdvpa!(dHdvpa_check,H_in,vpa,vpa_spectral,vperp,vperp_spectral) 
        
        #@views heatmap(vperp.grid, vpa.grid, H_check[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
        #     windowsize = (360,240), margin = 15pt)
        #     outfile = string("fkpl_RosenH.pdf")
        #     savefig(outfile)
        #@views heatmap(vperp.grid, vpa.grid, H_Maxwell[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
        #     windowsize = (360,240), margin = 15pt)
        #     outfile = string("fkpl_H_Maxwell.pdf")
        #     savefig(outfile)
       
        
        @. G_err = abs(fkarrays.Rosenbluth_G - G_Maxwell)
        @. H_err = abs(fkarrays.Rosenbluth_H - H_Maxwell)
        @. H_check_err = abs(H_check - H_Maxwell)
        max_G_err = maximum(G_err)
        max_H_err = maximum(H_err)
        max_H_check_err = maximum(H_check_err)
        println("max(G_err)",max_G_err)
        println("max(H_err)",max_H_err)
        println("max(H_check_err)",max_H_check_err)
        
        for ivperp in 1:nvperp
            for ivpa in 1:nvpa
                ## evaluate the collision operator with analytically computed G & H from a shifted Maxwellian
                ((d2Gdvpa2_Maxwell[ivpa,ivperp], d2Gdvperpdvpa_Maxwell[ivpa,ivperp], 
                        d2Gdvperp2_Maxwell[ivpa,ivperp],dHdvpa_Maxwell[ivpa,ivperp],
                        dHdvperp_Maxwell[ivpa,ivperp]) = calculate_Maxwellian_Rosenbluth_coefficients(dens_in[:],
                             upar_in[:],vths_in[:],vpa,vperp,ivpa,ivperp,n_ion_species) )
            end
        end
        
        @. d2Gdvperpdvpa_err = abs(d2Gdvperpdvpa_check - d2Gdvperpdvpa_Maxwell)
        @. d2Gdvpa2_err = abs(d2Gdvpa2_check - d2Gdvpa2_Maxwell)
        @. d2Gdvperp2_err = abs(d2Gdvperp2_check - d2Gdvperp2_Maxwell)
        @. dHdvperp_err = abs(dHdvperp_check - dHdvperp_Maxwell)
        @. dHdvpa_err = abs(dHdvpa_check - dHdvpa_Maxwell)
        max_dHdvpa_err = maximum(dHdvpa_err)
        max_dHdvperp_err = maximum(dHdvperp_err)
        max_d2Gdvperp2_err = maximum(d2Gdvperp2_err)
        max_d2Gdvpa2_err = maximum(d2Gdvpa2_err)
        max_d2Gdvperpdvpa_err = maximum(d2Gdvperpdvpa_err)
        println("max(dHdvpa_err)",max_dHdvpa_err)
        println("max(dHdvperp_err)",max_dHdvperp_err)
        println("max(d2Gdvperp2_err)",max_d2Gdvperp2_err)
        println("max(d2Gdvpa2_err)",max_d2Gdvpa2_err)
        println("max(d2Gdvperpdvpa_err)",max_d2Gdvperpdvpa_err)
    #    
    #    zero = 1.0e-3
    #    #println(G_Maxwell[41,:])
    #    #println(G_Maxwell[:,1])
    #    for ivperp in 1:nvperp
    #    #for ivpa in 1:nvpa
    #    #    if (maximum(G_err[ivpa,ivperp]) > zero)
    #    #    ##println("ivpa: ",ivpa," ivperp: ",ivperp," G_err: ",G_err[ivpa,ivperp])
    #    #    ##println("ivpa: ",ivpa," vpa: ",vpa.grid[ivpa]," ivperp: ",ivperp," vperp: ",vperp.grid[ivperp]," G_err: ",G_err[ivpa,ivperp]," G_Maxwell: ",G_Maxwell[ivpa,ivperp]," G_num: ",fkarrays.Rosenbluth_G[ivpa,ivperp])
    #    #    ##println("ivpa: ",ivpa," vpa: ",vpa.grid[ivpa]," ivperp: ",ivperp," vperp: ",vperp.grid[ivperp]," G_err: ",G_err[ivpa,ivperp])
    #    #    end
    #    #    #H_err[ivpa,ivperp]
    #    #end
    #    end
    #    #println(H_Maxwell[:,1])
    #    #println(fkarrays.Rosenbluth_H[:,1])
    #    #zero = 0.1
    #    for ivperp in 1:nvperp
    #    #for ivpa in 1:nvpa
    #    #    if (maximum(H_err[ivpa,ivperp]) > zero)
    #    #    ###println("ivpa: ",ivpa," ivperp: ",ivperp," H_err: ",H_err[ivpa,ivperp])
    #    #    ##println("ivpa: ",ivpa," vpa: ",vpa.grid[ivpa]," ivperp: ",ivperp," vperp: ",vperp.grid[ivperp]," H_err: ",H_err[ivpa,ivperp]," H_Maxwell: ",H_Maxwell[ivpa,ivperp]," H_num: ",fkarrays.Rosenbluth_H[ivpa,ivperp])
    #    #    end
    #    #    #H_err[ivpa,ivperp]
    #    #end
    #    end
        return max_G_err, max_H_err, max_H_check_err, max_dHdvpa_err, max_dHdvperp_err, max_d2Gdvperp2_err, max_d2Gdvpa2_err, max_d2Gdvperpdvpa_err
    end
    
    function test_Lagrange_Rosenbluth_potentials(ngrid,nelement; standalone=true)
        # set up grids for input Maxwellian
        vpa, vperp, vpa_spectral, vperp_spectral =  init_grids(nelement,ngrid)
        # set up necessary inputs for collision operator functions 
        nvperp = vperp.n
        nvpa = vpa.n
        
        # Set up MPI
        if standalone
            initialize_comms!()
        end
        setup_distributed_memory_MPI(1,1,1,1)
        looping.setup_loop_ranges!(block_rank[], block_size[];
                                       s=1, sn=1,
                                       r=1, z=1, vperp=vperp.n, vpa=vpa.n,
                                       vzeta=1, vr=1, vz=1)
        
        @serial_region begin
            println("beginning allocation   ", Dates.format(now(), dateformat"H:MM:SS"))
        end
        
        fs_in = Array{mk_float,2}(undef,nvpa,nvperp)
        dfsdvpa = Array{mk_float,2}(undef,nvpa,nvperp)
        d2fsdvpa2 = Array{mk_float,2}(undef,nvpa,nvperp)
        dfsdvperp = Array{mk_float,2}(undef,nvpa,nvperp)
        d2fsdvperpdvpa = Array{mk_float,2}(undef,nvpa,nvperp)
        d2fsdvperp2 = Array{mk_float,2}(undef,nvpa,nvperp)
        
        fsp_in = Array{mk_float,2}(undef,nvpa,nvperp)
        d2fspdvpa2 = Array{mk_float,2}(undef,nvpa,nvperp)
        dfspdvperp = Array{mk_float,2}(undef,nvpa,nvperp)
        d2fspdvperpdvpa = Array{mk_float,2}(undef,nvpa,nvperp)
        d2fspdvperp2 = Array{mk_float,2}(undef,nvpa,nvperp)
        
        #G_weights = Array{mk_float,4}(undef,nvpa,nvperp,nvpa,nvperp)
        G_weights = allocate_shared_float(nvpa,nvperp,nvpa,nvperp)
        G1_weights = allocate_shared_float(nvpa,nvperp,nvpa,nvperp)
        G2_weights = allocate_shared_float(nvpa,nvperp,nvpa,nvperp)
        G3_weights = allocate_shared_float(nvpa,nvperp,nvpa,nvperp)
        Gsp = allocate_shared_float(nvpa,nvperp)
        d2Gspdvpa2 = allocate_shared_float(nvpa,nvperp)
        dGspdvperp = allocate_shared_float(nvpa,nvperp)
        d2Gspdvperpdvpa = allocate_shared_float(nvpa,nvperp)
        d2Gspdvperp2 = allocate_shared_float(nvpa,nvperp)
        #Gsp = Array{mk_float,2}(undef,nvpa,nvperp)
        G_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
        G_err = allocate_shared_float(nvpa,nvperp)
        d2Gdvpa2_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
        d2Gdvpa2_err = allocate_shared_float(nvpa,nvperp)
        dGdvperp_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
        dGdvperp_err = allocate_shared_float(nvpa,nvperp)
        d2Gdvperpdvpa_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
        d2Gdvperpdvpa_err = allocate_shared_float(nvpa,nvperp)
        d2Gdvperp2_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
        d2Gdvperp2_err = allocate_shared_float(nvpa,nvperp)
        
        H_weights = allocate_shared_float(nvpa,nvperp,nvpa,nvperp)
        Hsp = allocate_shared_float(nvpa,nvperp)
        Hsp_from_Gsp = allocate_shared_float(nvpa,nvperp)
        dHspdvpa = allocate_shared_float(nvpa,nvperp)
        dHspdvperp = allocate_shared_float(nvpa,nvperp)
        #Gsp = Array{mk_float,2}(undef,nvpa,nvperp)
        H_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
        H_err = allocate_shared_float(nvpa,nvperp)
        dHdvpa_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
        dHdvpa_err = allocate_shared_float(nvpa,nvperp)
        dHdvperp_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
        dHdvperp_err = allocate_shared_float(nvpa,nvperp)
        
        Cssp_numerical = allocate_shared_float(nvpa,nvperp)
        Cssp_err = allocate_shared_float(nvpa,nvperp)
        Cssp_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
        Cflux_vpa = allocate_shared_float(nvpa,nvperp)
        Cflux_vpa_err = allocate_shared_float(nvpa,nvperp)
        Cflux_vperp = allocate_shared_float(nvpa,nvperp)
        Cflux_vperp_err = allocate_shared_float(nvpa,nvperp)
        Cflux_vpa_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
        Cflux_vperp_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
        
        @serial_region begin
            println("setting up input arrays   ", Dates.format(now(), dateformat"H:MM:SS"))
        end
        
        # set up test Maxwellian
        # species s 
        denss = 1.0 #3.0/4.0
        upars = 0.5 #2.0/3.0
        ppars = 1.0 #2.0/3.0
        pperps = 1.0 #2.0/3.0
        press = get_pressure(ppars,pperps) 
        ms = 1.0
        vths = get_vth(press,denss,ms)
        # species sp 
        denssp = 1.0 #3.0/4.0
        uparsp = 0.5 #2.0/3.0
        pparsp = 1.0 #2.0/3.0
        pperpsp = 1.0 #2.0/3.0
        pressp = get_pressure(pparsp,pperpsp) 
        msp = 1.0
        vthsp = get_vth(pressp,denssp,msp)
        
        nussp = 1.0
        for ivperp in 1:nvperp
            for ivpa in 1:nvpa
                fs_in[ivpa,ivperp] = F_Maxwellian(denss,upars,vths,vpa,vperp,ivpa,ivperp) #(denss/vths^3)*exp( - ((vpa.grid[ivpa]-upar)^2 + vperp.grid[ivperp]^2)/vths^2 ) 
                fsp_in[ivpa,ivperp] = F_Maxwellian(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp) #(denss/vths^3)*exp( - ((vpa.grid[ivpa]-upar)^2 + vperp.grid[ivperp]^2)/vths^2 ) 
                G_Maxwell[ivpa,ivperp] = G_Maxwellian(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
                H_Maxwell[ivpa,ivperp] = H_Maxwellian(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
                d2Gdvpa2_Maxwell[ivpa,ivperp] = d2Gdvpa2(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
                dGdvperp_Maxwell[ivpa,ivperp] = dGdvperp(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
                d2Gdvperpdvpa_Maxwell[ivpa,ivperp] = d2Gdvperpdvpa(denssp,upars,vthsp,vpa,vperp,ivpa,ivperp)
                d2Gdvperp2_Maxwell[ivpa,ivperp] = d2Gdvperp2(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
                dHdvperp_Maxwell[ivpa,ivperp] = dHdvperp(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
                dHdvpa_Maxwell[ivpa,ivperp] = dHdvpa(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
                Cssp_Maxwell[ivpa,ivperp] = Cssp_Maxwellian_inputs(denss,upars,vths,ms,
                                                                  denssp,uparsp,vthsp,msp,
                                                                  nussp,vpa,vperp,ivpa,ivperp)
                Cflux_vpa_Maxwell[ivpa,ivperp] = Cflux_vpa_Maxwellian_inputs(ms,denss,upars,vths,
                                                             msp,denssp,uparsp,vthsp,
                                                             vpa,vperp,ivpa,ivperp)
                Cflux_vperp_Maxwell[ivpa,ivperp] = Cflux_vperp_Maxwellian_inputs(ms,denss,upars,vths,
                                                             msp,denssp,uparsp,vthsp,
                                                             vpa,vperp,ivpa,ivperp)
            end
        end
        for ivperp in 1:nvperp
            # s
            @views derivative!(vpa.scratch, fs_in[:,ivperp], vpa, vpa_spectral)
            @. dfsdvpa[:,ivperp] = vpa.scratch
            @views derivative!(vpa.scratch2, vpa.scratch, vpa, vpa_spectral)
            @. d2fsdvpa2[:,ivperp] = vpa.scratch2
            # sp
            @views derivative!(vpa.scratch, fsp_in[:,ivperp], vpa, vpa_spectral)
            @views derivative!(vpa.scratch2, vpa.scratch, vpa, vpa_spectral)
            @. d2fspdvpa2[:,ivperp] = vpa.scratch2
        end
        for ivpa in 1:vpa.n
            # s
            @views derivative!(vperp.scratch, fs_in[ivpa,:], vperp, vperp_spectral)
            @. dfsdvperp[ivpa,:] = vperp.scratch
            @views derivative!(vperp.scratch2, vperp.scratch, vperp, vperp_spectral)
            @. d2fsdvperp2[ivpa,:] = vperp.scratch2
            # sp
            @views derivative!(vperp.scratch, fsp_in[ivpa,:], vperp, vperp_spectral)
            @. dfspdvperp[ivpa,:] = vperp.scratch
            @views derivative!(vperp.scratch2, vperp.scratch, vperp, vperp_spectral)
            @. d2fspdvperp2[ivpa,:] = vperp.scratch2            
        end
        for ivperp in 1:nvperp
            # s
            @views derivative!(vpa.scratch, dfsdvperp[:,ivperp], vpa, vpa_spectral)
            @. d2fsdvperpdvpa[:,ivperp] = vpa.scratch
            # sp
            @views derivative!(vpa.scratch, dfspdvperp[:,ivperp], vpa, vpa_spectral)
            @. d2fspdvperpdvpa[:,ivperp] = vpa.scratch
        end
        
        function get_imin_imax(coord,iel)
            j = iel
            if j > 1
                k = 1
            else
                k = 0
            end
            imin = coord.imin[j] - k
            imax = coord.imax[j]
            return imin, imax
        end
        
        function get_nodes(coord,iel)
            # get imin and imax of this element on full grid
            (imin, imax) = get_imin_imax(coord,iel)
            nodes = coord.grid[imin:imax]
            return nodes
        end
        """
        Lagrange polynomial
        args: 
        j - index of l_j from list of nodes
        x_nodes - array of x node values
        x - point where interpolated value is returned
        """
        function lagrange_poly(j,x_nodes,x)
            # get number of nodes
            n = size(x_nodes,1)
            # location where l(x0) = 1
            x0 = x_nodes[j]
            # evaluate polynomial
            poly = 1.0
            for i in 1:j-1
                    poly *= (x - x_nodes[i])/(x0 - x_nodes[i])
            end
            for i in j+1:n
                    poly *= (x - x_nodes[i])/(x0 - x_nodes[i])
            end
            return poly
        end
        
        function get_scaled_x_w!(x_scaled, w_scaled, x, w, node_min, node_max)
            shift = 0.5*(node_min + node_max)
            scale = 0.5*(node_max - node_min)
            @. x_scaled = scale*x + shift
            @. w_scaled = scale*w
            return nothing
        end
        
        
        @serial_region begin
            println("setting up GL quadrature   ", Dates.format(now(), dateformat"H:MM:SS"))
        end
        
        # get Gauss-Legendre points and weights on (-1,1)
        nquad = 2*ngrid
        x, w = gausslegendre(nquad)
        x_vpa, w_vpa = Array{mk_float,1}(undef,nquad), Array{mk_float,1}(undef,nquad)
        x_vperp, w_vperp = Array{mk_float,1}(undef,nquad), Array{mk_float,1}(undef,nquad)
        
        
        @serial_region begin
            println("beginning weights calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
        end
        
        # precalculated weights, integrating over Lagrange polynomials
        begin_vperp_vpa_region()
        @loop_vperp_vpa ivperp ivpa begin
        #for ivperp in 1:nvperp
        #    for ivpa in 1:nvpa 
                vperp_val = vperp.grid[ivperp]
                vpa_val = vpa.grid[ivpa]
                @. G_weights[ivpa,ivperp,:,:] = 0.0  
                @. G1_weights[ivpa,ivperp,:,:] = 0.0  
                @. G2_weights[ivpa,ivperp,:,:] = 0.0  
                @. G3_weights[ivpa,ivperp,:,:] = 0.0  
                @. H_weights[ivpa,ivperp,:,:] = 0.0  
                # loop over elements and grid points within elements on primed coordinate
                for ielement_vperp in 1:vperp.nelement_local
                    
                    vperp_nodes = get_nodes(vperp,ielement_vperp)
                    vperp_max = vperp_nodes[end]
                    if ielement_vperp > 1 # Gauss-Lobatto
                        vperp_min = vperp_nodes[1] 
                    else # adjust for the Gauss-Radau element
                        vperp_min = 0.0
                    end
                    get_scaled_x_w!(x_vperp, w_vperp, x, w, vperp_min, vperp_max)
                    
                    for ielement_vpa in 1:vpa.nelement_local
                        
                        vpa_nodes = get_nodes(vpa,ielement_vpa)
                        # assumme Gauss-Lobatto elements
                        vpa_min, vpa_max = vpa_nodes[1], vpa_nodes[end]
                        get_scaled_x_w!(x_vpa, w_vpa, x, w, vpa_min, vpa_max)
                        
                        for igrid_vperp in 1:vperp.ngrid
                            for igrid_vpa in 1:vpa.ngrid
                                # get grid index for point on full grid  
                                ivpap = vpa.igrid_full[igrid_vpa,ielement_vpa]   
                                ivperpp = vperp.igrid_full[igrid_vperp,ielement_vperp]   
                                # carry out integration over Lagrange polynomial at this node, on this element
                                for kvperp in 1:nquad 
                                    for kvpa in 1:nquad 
                                        x_kvpa = x_vpa[kvpa]
                                        x_kvperp = x_vperp[kvperp]
                                        w_kvperp = w_vperp[kvperp]
                                        w_kvpa = w_vpa[kvpa]
                                        denom = (vpa_val - x_kvpa)^2 + (vperp_val + x_kvperp)^2 
                                        mm = 4.0*vperp_val*x_kvperp/denom
                                        prefac = sqrt(denom)
                                        ellipe_mm = ellipe(mm) 
                                        ellipk_mm = ellipk(mm) 
                                        G_elliptic_integral_factor = 2.0*ellipe_mm*prefac/pi
                                        G1_elliptic_integral_factor = -(2.0*prefac/pi)*( (2.0 - mm)*ellipe_mm - 2.0*(1.0 - mm)*ellipk_mm )/(3.0*mm)
                                        G2_elliptic_integral_factor = (2.0*prefac/pi)*( (7.0*mm^2 + 8.0*mm - 8.0)*ellipe_mm + 4.0*(2.0 - mm)*(1.0 - mm)*ellipk_mm )/(15.0*mm^2)
                                        G3_elliptic_integral_factor = (2.0*prefac/pi)*( 8.0*(mm^2 - mm + 1.0)*ellipe_mm - 4.0*(2.0 - mm)*(1.0 - mm)*ellipk_mm )/(15.0*mm^2)
                                        H_elliptic_integral_factor = 2.0*ellipk_mm/(pi*prefac)
                                        lagrange_poly_vpa = lagrange_poly(igrid_vpa,vpa_nodes,x_kvpa)
                                        lagrange_poly_vperp = lagrange_poly(igrid_vperp,vperp_nodes,x_kvperp)
                                        
                                        (G_weights[ivpa,ivperp,ivpap,ivperpp] += 
                                            lagrange_poly_vpa*lagrange_poly_vperp*
                                            G_elliptic_integral_factor*x_kvperp*w_kvperp*w_kvpa*2.0/sqrt(pi))
                                        
                                        (G1_weights[ivpa,ivperp,ivpap,ivperpp] += 
                                            lagrange_poly_vpa*lagrange_poly_vperp*
                                            G1_elliptic_integral_factor*x_kvperp*w_kvperp*w_kvpa*2.0/sqrt(pi))
                                        
                                        (G2_weights[ivpa,ivperp,ivpap,ivperpp] += 
                                            lagrange_poly_vpa*lagrange_poly_vperp*
                                            G2_elliptic_integral_factor*x_kvperp*w_kvperp*w_kvpa*2.0/sqrt(pi))
                                        
                                        (G3_weights[ivpa,ivperp,ivpap,ivperpp] += 
                                            lagrange_poly_vpa*lagrange_poly_vperp*
                                            G3_elliptic_integral_factor*w_kvperp*w_kvpa*2.0/sqrt(pi))
                                        
                                        (H_weights[ivpa,ivperp,ivpap,ivperpp] += 
                                            lagrange_poly_vpa*lagrange_poly_vperp*
                                            H_elliptic_integral_factor*x_kvperp*w_kvperp*w_kvpa*2.0/sqrt(pi))
                                    end
                                end
                            end
                        end
                    end
                end
            #end
        end
        
        #_block_synchronize()
        begin_serial_region()
        @serial_region begin
            println("beginning integration   ", Dates.format(now(), dateformat"H:MM:SS"))
        end
        
        begin_vperp_vpa_region()
        
        # use precalculated weights to calculate Gsp using nodal values of fs
        @loop_vperp_vpa ivperp ivpa begin
        #for ivperp in 1:nvperp
            #for ivpa in 1:nvpa 
                d2Gspdvpa2[ivpa,ivperp] = 0.0
                dGspdvperp[ivpa,ivperp] = 0.0
                d2Gspdvperpdvpa[ivpa,ivperp] = 0.0
                d2Gspdvperp2[ivpa,ivperp] = 0.0
                Gsp[ivpa,ivperp] = 0.0
                Hsp[ivpa,ivperp] = 0.0
                for ivperpp in 1:nvperp
                    for ivpap in 1:nvpa
                        d2Gspdvpa2[ivpa,ivperp] += G_weights[ivpa,ivperp,ivpap,ivperpp]*d2fspdvpa2[ivpap,ivperpp]
                        dGspdvperp[ivpa,ivperp] += G1_weights[ivpa,ivperp,ivpap,ivperpp]*dfspdvperp[ivpap,ivperpp]
                        d2Gspdvperpdvpa[ivpa,ivperp] += G1_weights[ivpa,ivperp,ivpap,ivperpp]*d2fspdvperpdvpa[ivpap,ivperpp]
                        d2Gspdvperp2[ivpa,ivperp] += G2_weights[ivpa,ivperp,ivpap,ivperpp]*d2fspdvperp2[ivpap,ivperpp] + G3_weights[ivpa,ivperp,ivpap,ivperpp]*dfspdvperp[ivpap,ivperpp]
                        Gsp[ivpa,ivperp] += G_weights[ivpa,ivperp,ivpap,ivperpp]*fsp_in[ivpap,ivperpp]
                        Hsp[ivpa,ivperp] += H_weights[ivpa,ivperp,ivpap,ivperpp]*fsp_in[ivpap,ivperpp]
                    end
                end
            #end
            
            (Hsp_from_Gsp[ivpa,ivperp] = 0.5*( d2Gspdvpa2[ivpa,ivperp] +
                                              d2Gspdvperp2[ivpa,ivperp] +
                            (1.0/vperp.grid[ivperp])*dGspdvperp[ivpa,ivperp]))
        end
        
        begin_vperp_region()
        @loop_vperp ivperp begin
            @views derivative!(vpa.scratch, Hsp_from_Gsp[:,ivperp], vpa, vpa_spectral)
            @. dHspdvpa[:,ivperp] = vpa.scratch
        end
        begin_vpa_region()
        @loop_vpa ivpa begin
            @views derivative!(vperp.scratch, Hsp_from_Gsp[ivpa,:], vperp, vperp_spectral)
            @. dHspdvperp[ivpa,:] = vperp.scratch
        end
        
        # evaluate collsion operator
        begin_vperp_vpa_region()
        @loop_vperp_vpa ivperp ivpa begin
            # fully expanded form
            (Cssp_numerical[ivpa,ivperp] =  nussp*( d2fsdvpa2[ivpa,ivperp]*d2Gspdvpa2[ivpa,ivperp] +
                              d2fsdvperp2[ivpa,ivperp]*d2Gspdvperp2[ivpa,ivperp] +
                              2.0*d2fsdvperpdvpa[ivpa,ivperp]*d2Gspdvperpdvpa[ivpa,ivperp] +                
                              (1.0/(vperp.grid[ivperp]^2))*dfsdvperp[ivpa,ivperp]*dGspdvperp[ivpa,ivperp] +                
                              2.0*(1.0 - (ms/msp))*(dfsdvpa[ivpa,ivperp]*dHspdvpa[ivpa,ivperp] + dfsdvperp[ivpa,ivperp]*dHspdvperp[ivpa,ivperp]) +                
                              (8.0/sqrt(pi))*(ms/msp)*fs_in[ivpa,ivperp]*fsp_in[ivpa,ivperp]) )
            # collisional fluxes 
            ((Cflux_vpa[ivpa,ivperp],Cflux_vperp[ivpa,ivperp]) =
              calculate_collisional_fluxes(fs_in[ivpa,ivperp],
                    dfsdvpa[ivpa,ivperp],dfsdvperp[ivpa,ivperp],
                    d2Gspdvpa2[ivpa,ivperp],d2Gspdvperpdvpa[ivpa,ivperp],
                    d2Gspdvperp2[ivpa,ivperp],dHspdvpa[ivpa,ivperp],dHspdvperp[ivpa,ivperp],
                    ms,msp) )
        end
    
        
        plot_H = false #true
        plot_dHdvpa = false #true
        plot_dHdvperp = false #true
        plot_d2Gdvperp2 = false #true
        plot_d2Gdvperpdvpa = false #true
        plot_dGdvperp = false #true
        plot_d2Gdvpa2 = false #true
        plot_G = false #true
        plot_C = false #true
        
        begin_serial_region()
        @serial_region begin
            println("finished integration   ", Dates.format(now(), dateformat"H:MM:SS"))
            @. Cssp_err = abs(Cssp_numerical - Cssp_Maxwell)
            max_C_err = maximum(Cssp_err)
            max_C_Maxwell_val = maximum(Cssp_Maxwell)
            max_C_numerical_val = maximum(Cssp_numerical)
            println("max_C_err: ",max_C_err)
            println("max_C_Maxwell_val: ",max_C_Maxwell_val)
            println("max_C_numerical_val: ",max_C_numerical_val)
            @. Cflux_vpa_err = abs(Cflux_vpa - Cflux_vpa_Maxwell)
            max_Cflux_vpa_err = maximum(Cflux_vpa_err)
            println("max_Cflux_vpa_err: ",max_Cflux_vpa_err)
            @. Cflux_vperp_err = abs(Cflux_vperp - Cflux_vperp_Maxwell)
            max_Cflux_vperp_err = maximum(Cflux_vperp_err)
            println("max_Cflux_vperp_err: ",max_Cflux_vperp_err)
            @. H_err = abs(Hsp - H_Maxwell)
            max_H_err = maximum(H_err)
            println("max_H_err: ",max_H_err)
            @. H_err = abs(Hsp_from_Gsp - H_Maxwell)
            max_H_err = maximum(H_err)
            println("max_H_from_G_err: ",max_H_err)
            @. dHdvperp_err = abs(dHspdvperp - dHdvperp_Maxwell)
            max_dHdvperp_err = maximum(dHdvperp_err)
            println("max_dHdvperp_err: ",max_dHdvperp_err)
            @. dHdvpa_err = abs(dHspdvpa - dHdvpa_Maxwell)
            max_dHdvpa_err = maximum(dHdvpa_err)
            println("max_dHdvpa_err: ",max_dHdvpa_err)
            if plot_C
                @views heatmap(vperp.grid, vpa.grid, Cssp_numerical[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                     windowsize = (360,240), margin = 15pt)
                     outfile = string("fkpl_C_lagrange.pdf")
                     savefig(outfile)
                @views heatmap(vperp.grid, vpa.grid, Cssp_Maxwell[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                     windowsize = (360,240), margin = 15pt)
                     outfile = string("fkpl_C_Maxwell.pdf")
                     savefig(outfile)
                 @views heatmap(vperp.grid, vpa.grid, Cssp_err[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                     windowsize = (360,240), margin = 15pt)
                     outfile = string("fkpl_C_err.pdf")
                     savefig(outfile)
            end
            if plot_H
                @views heatmap(vperp.grid, vpa.grid, Hsp[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                     windowsize = (360,240), margin = 15pt)
                     outfile = string("fkpl_H_lagrange.pdf")
                     savefig(outfile)
                @views heatmap(vperp.grid, vpa.grid, Hsp_from_Gsp[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                     windowsize = (360,240), margin = 15pt)
                     outfile = string("fkpl_H_from_G_lagrange.pdf")
                     savefig(outfile)
                @views heatmap(vperp.grid, vpa.grid, H_Maxwell[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                     windowsize = (360,240), margin = 15pt)
                     outfile = string("fkpl_H_Maxwell.pdf")
                     savefig(outfile)
                 @views heatmap(vperp.grid, vpa.grid, H_err[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                     windowsize = (360,240), margin = 15pt)
                     outfile = string("fkpl_H_err.pdf")
                     savefig(outfile)
            end
            if plot_dHdvpa
                @views heatmap(vperp.grid, vpa.grid, dHspdvpa[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                     windowsize = (360,240), margin = 15pt)
                     outfile = string("fkpl_dHdvpa_lagrange.pdf")
                     savefig(outfile)
                @views heatmap(vperp.grid, vpa.grid, dHdvpa_Maxwell[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                     windowsize = (360,240), margin = 15pt)
                     outfile = string("fkpl_dHdvpa_Maxwell.pdf")
                     savefig(outfile)
                 @views heatmap(vperp.grid, vpa.grid, dHdvpa_err[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                     windowsize = (360,240), margin = 15pt)
                     outfile = string("fkpl_dHdvpa_err.pdf")
                     savefig(outfile)
            end
            if plot_dHdvperp
                @views heatmap(vperp.grid, vpa.grid, dHspdvperp[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                     windowsize = (360,240), margin = 15pt)
                     outfile = string("fkpl_dHdvperp_lagrange.pdf")
                     savefig(outfile)
                @views heatmap(vperp.grid, vpa.grid, dHdvperp_Maxwell[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                     windowsize = (360,240), margin = 15pt)
                     outfile = string("fkpl_dHdvperp_Maxwell.pdf")
                     savefig(outfile)
                 @views heatmap(vperp.grid, vpa.grid, dHdvperp_err[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                     windowsize = (360,240), margin = 15pt)
                     outfile = string("fkpl_dHdvperp_err.pdf")
                     savefig(outfile)
            end
            @. d2Gdvperp2_err = abs(d2Gspdvperp2 - d2Gdvperp2_Maxwell)
            max_d2Gdvperp2_err = maximum(d2Gdvperp2_err)
            println("max_d2Gdvperp2_err: ",max_d2Gdvperp2_err)
            if plot_d2Gdvperp2
                @views heatmap(vperp.grid, vpa.grid, d2Gspdvperp2[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                     windowsize = (360,240), margin = 15pt)
                     outfile = string("fkpl_d2Gdvperp2_lagrange.pdf")
                     savefig(outfile)
                @views heatmap(vperp.grid, vpa.grid, d2Gdvperp2_Maxwell[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                     windowsize = (360,240), margin = 15pt)
                     outfile = string("fkpl_d2Gdvperp2_Maxwell.pdf")
                     savefig(outfile)
                 @views heatmap(vperp.grid, vpa.grid, d2Gdvperp2_err[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                     windowsize = (360,240), margin = 15pt)
                     outfile = string("fkpl_d2Gdvperp2_err.pdf")
                     savefig(outfile)
            end
            @. d2Gdvperpdvpa_err = abs(d2Gspdvperpdvpa - d2Gdvperpdvpa_Maxwell)
            max_d2Gdvperpdvpa_err = maximum(d2Gdvperpdvpa_err)
            println("max_d2Gdvperpdvpa_err: ",max_d2Gdvperpdvpa_err)
            if plot_d2Gdvperpdvpa
                @views heatmap(vperp.grid, vpa.grid, d2Gspdvperpdvpa[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                     windowsize = (360,240), margin = 15pt)
                     outfile = string("fkpl_d2Gdvperpdvpa_lagrange.pdf")
                     savefig(outfile)
                @views heatmap(vperp.grid, vpa.grid, d2Gdvperpdvpa_Maxwell[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                     windowsize = (360,240), margin = 15pt)
                     outfile = string("fkpl_d2Gdvperpdvpa_Maxwell.pdf")
                     savefig(outfile)
                 @views heatmap(vperp.grid, vpa.grid, d2Gdvperpdvpa_err[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                     windowsize = (360,240), margin = 15pt)
                     outfile = string("fkpl_d2Gdvperpdvpa_err.pdf")
                     savefig(outfile)
            end
            @. dGdvperp_err = abs(dGspdvperp - dGdvperp_Maxwell)
            max_dGdvperp_err = maximum(dGdvperp_err)
            println("max_dGdvperp_err: ",max_dGdvperp_err)
            if plot_dGdvperp
                @views heatmap(vperp.grid, vpa.grid, dGspdvperp[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                     windowsize = (360,240), margin = 15pt)
                     outfile = string("fkpl_dGdvperp_lagrange.pdf")
                     savefig(outfile)
                @views heatmap(vperp.grid, vpa.grid, dGdvperp_Maxwell[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                     windowsize = (360,240), margin = 15pt)
                     outfile = string("fkpl_dGdvperp_Maxwell.pdf")
                     savefig(outfile)
                 @views heatmap(vperp.grid, vpa.grid, dGdvperp_err[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                     windowsize = (360,240), margin = 15pt)
                     outfile = string("fkpl_dGdvperp_err.pdf")
                     savefig(outfile)
            end
            @. d2Gdvpa2_err = abs(d2Gspdvpa2 - d2Gdvpa2_Maxwell)
            max_d2Gdvpa2_err = maximum(d2Gdvpa2_err)
            println("max_d2Gdvpa2_err: ",max_d2Gdvpa2_err)
            if plot_d2Gdvpa2
                @views heatmap(vperp.grid, vpa.grid, d2Gspdvpa2[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                     windowsize = (360,240), margin = 15pt)
                     outfile = string("fkpl_d2Gdvpa2_lagrange.pdf")
                     savefig(outfile)
                @views heatmap(vperp.grid, vpa.grid, d2Gdvpa2_Maxwell[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                     windowsize = (360,240), margin = 15pt)
                     outfile = string("fkpl_d2Gdvpa2_Maxwell.pdf")
                     savefig(outfile)
                 @views heatmap(vperp.grid, vpa.grid, d2Gdvpa2_err[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                     windowsize = (360,240), margin = 15pt)
                     outfile = string("fkpl_d2Gdvpa2_err.pdf")
                     savefig(outfile)
            end
            @. G_err = abs(Gsp - G_Maxwell)
            max_G_err = maximum(G_err)
            println("max_G_err: ",max_G_err)
            if plot_G
                @views heatmap(vperp.grid, vpa.grid, Gsp[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                     windowsize = (360,240), margin = 15pt)
                     outfile = string("fkpl_G_lagrange.pdf")
                     savefig(outfile)
                @views heatmap(vperp.grid, vpa.grid, G_Maxwell[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                     windowsize = (360,240), margin = 15pt)
                     outfile = string("fkpl_G_Maxwell.pdf")
                     savefig(outfile)
                 @views heatmap(vperp.grid, vpa.grid, G_err[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                     windowsize = (360,240), margin = 15pt)
                     outfile = string("fkpl_G_err.pdf")
                     savefig(outfile)
            end
        end
        _block_synchronize()
        if standalone 
            finalize_comms!()
        end
        #println(maximum(G_err), maximum(H_err), maximum(dHdvpa_err), maximum(dHdvperp_err), maximum(d2Gdvperp2_err), maximum(d2Gdvpa2_err), maximum(d2Gdvperpdvpa_err), maximum(dGdvperp_err))
        results = maximum(Cssp_err), maximum(Cflux_vpa_err), maximum(Cflux_vperp_err), maximum(G_err), maximum(H_err), maximum(dHdvpa_err), maximum(dHdvperp_err), maximum(d2Gdvperp2_err), maximum(d2Gdvpa2_err), maximum(d2Gdvperpdvpa_err), maximum(dGdvperp_err)
        return results 
    end
    
    function test_collision_operator(nelement,ngrid)
    
        vpa, vperp, vpa_spectral, vperp_spectral =  init_grids(nelement,ngrid)
        # set up necessary inputs for collision operator functions 
        nvperp = vperp.n
        nvpa = vpa.n
        
        cfreqssp = 1.0
        ms = 1.0
        msp = 1.0
        
        
        Cssp = Array{mk_float,2}(undef,nvpa,nvperp)
        Cssp_err = Array{mk_float,2}(undef,nvpa,nvperp)
        
        Gam_vperp_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
        Gam_vperp_err = Array{mk_float,2}(undef,nvpa,nvperp)
        #Gam_vperp_GMaxwell = Array{mk_float,2}(undef,nvpa,nvperp)
        #Gam_vperp_HMaxwell = Array{mk_float,2}(undef,nvpa,nvperp)
        #Gam_vperp_Gerr = Array{mk_float,2}(undef,nvpa,nvperp)
        #Gam_vperp_Herr = Array{mk_float,2}(undef,nvpa,nvperp)
        Gam_vpa_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
        Gam_vpa_err = Array{mk_float,2}(undef,nvpa,nvperp)
        #Gam_vpa_GMaxwell = Array{mk_float,2}(undef,nvpa,nvperp)
        #Gam_vpa_HMaxwell = Array{mk_float,2}(undef,nvpa,nvperp)
        #Gam_vpa_Gerr = Array{mk_float,2}(undef,nvpa,nvperp)
        #Gam_vpa_Herr = Array{mk_float,2}(undef,nvpa,nvperp)

        d2Gdvpa2 = Array{mk_float,2}(undef,nvpa,nvperp)
        d2Gdvperpdvpa = Array{mk_float,2}(undef,nvpa,nvperp)
        d2Gdvperp2 = Array{mk_float,2}(undef,nvpa,nvperp)
        dHdvpa = Array{mk_float,2}(undef,nvpa,nvperp)
        dHdvperp = Array{mk_float,2}(undef,nvpa,nvperp)
        Gam_vpa = Array{mk_float,2}(undef,nvpa,nvperp)
        Gam_vpa_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
        Gam_vperp = Array{mk_float,2}(undef,nvpa,nvperp)
        dfns =  Array{mk_float,2}(undef,nvpa,nvperp)
        dfnsp =  Array{mk_float,2}(undef,nvpa,nvperp)
        pdf_buffer_1 =  Array{mk_float,2}(undef,nvpa,nvperp)
        pdf_buffer_2 =  Array{mk_float,2}(undef,nvpa,nvperp)
        n_ion_species = 1
        dens_in = Array{mk_float,1}(undef,n_ion_species)
        upar_in = Array{mk_float,1}(undef,n_ion_species)
        vths_in = Array{mk_float,1}(undef,n_ion_species)
        densp_in = Array{mk_float,1}(undef,n_ion_species)
        uparp_in = Array{mk_float,1}(undef,n_ion_species)
        vthsp_in = Array{mk_float,1}(undef,n_ion_species)
        
        # 2D isotropic Maxwellian test
        # assign a known isotropic Maxwellian distribution in normalised units
        # first argument (derivatives)
        dens = 1.0#3.0/4.0
        upar = 2.0/3.0
        ppar = 2.0/3.0
        pperp = 2.0/3.0
        pres = get_pressure(ppar,pperp) 
        mi = 1.0
        vths = get_vth(pres,dens,mi)
        # second argument (potentials)
        densp = 1.0 #2.0#3.0/4.0
        uparp = 2.0/3.0#1.0/3.0#3.0/4.0
        pparp = 2.0/3.0#4.0/3.0
        pperpp = 2.0/3.0#4.0/3.0
        presp = get_pressure(pparp,pperpp) 
        mip = 1.0#1.5
        vthsp = get_vth(presp,densp,mip)
        for ivperp in 1:vperp.n
            for ivpa in 1:vpa.n
                vpa_val = vpa.grid[ivpa]
                vperp_val = vperp.grid[ivperp]
                dfns[ivpa,ivperp] = (dens/vths^3)*exp( - ((vpa_val-upar)^2 + vperp_val^2)/vths^2 )
                dfnsp[ivpa,ivperp] = (densp/vthsp^3)*exp( - ((vpa_val-uparp)^2 + vperp_val^2)/vthsp^2 )
            end
        end
        pdf_in = dfns
        dens_in[1] = get_density(dfns,vpa,vperp)
        upar_in[1] = get_upar(dfns,vpa,vperp,dens_in[1])
        ppar_in = get_ppar(dfns,vpa,vperp,upar_in[1],mi)
        pperp_in = get_pperp(dfns,vpa,vperp,mi)
        pres_in = pressure(ppar_in,pperp_in)
        vths_in[1] = get_vth(pres_in,dens_in[1],mi)
        
        densp_in[1] = get_density(dfnsp,vpa,vperp)
        uparp_in[1] = get_upar(dfnsp,vpa,vperp,densp_in[1])
        pparp_in = get_ppar(dfnsp,vpa,vperp,uparp_in[1],mip)
        pperpp_in = get_pperp(dfnsp,vpa,vperp,mip)
        presp_in = pressure(pparp_in,pperpp_in)
        vthsp_in[1] = get_vth(presp_in,densp_in[1],mip)
        
        println("Isotropic 2D Maxwellian: first argument (derivatives)")
        println("dens_test: ", dens_in[1], " dens: ", dens, " error: ", abs(dens_in[1]-dens))
        println("upar_test: ", upar_in[1], " upar: ", upar, " error: ", abs(upar_in[1]-upar))
        println("ppar_test: ", ppar_in, " ppar: ", ppar, " error: ", abs(ppar_in-ppar))
        println("pperp_test: ", pperp_in, " pperp: ", pperp, " error: ", abs(pperp_in-pperp))
        println("vth_test: ", vths_in[1], " vth: ", vths, " error: ", abs(vths_in[1]-vths))
        println("Isotropic 2D Maxwellian: second argument (potentials)")
        println("dens_test: ", densp_in[1], " dens: ", densp, " error: ", abs(densp_in[1]-densp))
        println("upar_test: ", uparp_in[1], " upar: ", uparp, " error: ", abs(uparp_in[1]-uparp))
        println("ppar_test: ", pparp_in, " ppar: ", pparp, " error: ", abs(pparp_in-pparp))
        println("pperp_test: ", pperpp_in, " pperp: ", pperpp, " error: ", abs(pperpp_in-pperpp))
        println("vth_test: ", vthsp_in[1], " vth: ", vthsp, " error: ", abs(vthsp_in[1]-vthsp))
        

        for ivperp in 1:nvperp
            for ivpa in 1:nvpa
                Gam_vpa_Maxwell[ivpa,ivperp] = Cflux_vpa_Maxwellian_inputs(mi,dens_in[1],upar_in[1],vths_in[1],
                                                                         mip,densp_in[1],uparp_in[1],vthsp_in[1],
                                                                         vpa,vperp,ivpa,ivperp)
                Gam_vperp_Maxwell[ivpa,ivperp] = Cflux_vperp_Maxwellian_inputs(mi,dens_in[1],upar_in[1],vths_in[1],
                                                                         mip,densp_in[1],uparp_in[1],vthsp_in[1],
                                                                         vpa,vperp,ivpa,ivperp)
            end
        end
        
        # d F / d vpa
        for ivperp in 1:nvperp
            @views derivative!(vpa.scratch, pdf_in[:,ivperp], vpa, vpa_spectral)
            @. pdf_buffer_1[:,ivperp] = vpa.scratch
        end
        # d F / d vperp
        for ivpa in 1:nvpa
            @views derivative!(vperp.scratch, pdf_in[ivpa,:], vperp, vperp_spectral)
            @. pdf_buffer_2[ivpa,:] = vperp.scratch
        end
        
        for ivperp in 1:nvperp
            for ivpa in 1:nvpa
            ## evaluate the collision operator with analytically computed G & H from a shifted Maxwellian
            ((Rosenbluth_d2Gdvpa2, Rosenbluth_d2Gdvperpdvpa, 
                    Rosenbluth_d2Gdvperp2,Rosenbluth_dHdvpa,
                    Rosenbluth_dHdvperp) = calculate_Maxwellian_Rosenbluth_coefficients(densp_in[:],
                         uparp_in[:],vthsp_in[:],vpa,vperp,ivpa,ivperp,n_ion_species) )
                       
            # now form the collisional fluxes at this s,z,r
            ( (Cflux_vpa,Cflux_vperp) = calculate_collisional_fluxes(pdf_in[ivpa,ivperp],
                    pdf_buffer_1[ivpa,ivperp],pdf_buffer_2[ivpa,ivperp],
                    Rosenbluth_d2Gdvpa2,Rosenbluth_d2Gdvperpdvpa,
                    Rosenbluth_d2Gdvperp2,Rosenbluth_dHdvpa,Rosenbluth_dHdvperp,
                    mi,mip) )
                    
                    
            d2Gdvpa2[ivpa,ivperp] = Rosenbluth_d2Gdvpa2
            d2Gdvperpdvpa[ivpa,ivperp] = Rosenbluth_d2Gdvperpdvpa
            d2Gdvperp2[ivpa,ivperp] = Rosenbluth_d2Gdvperp2
            dHdvpa[ivpa,ivperp] = Rosenbluth_dHdvpa
            dHdvperp[ivpa,ivperp] = Rosenbluth_dHdvperp
            Gam_vpa[ivpa,ivperp] = Cflux_vpa
            Gam_vperp[ivpa,ivperp] = Cflux_vperp
            end
        end
        
        @. Gam_vpa_err = abs(Gam_vpa - Gam_vpa_Maxwell)
        @. Gam_vperp_err = abs(Gam_vperp - Gam_vperp_Maxwell)
        max_Gam_vpa_err = maximum(Gam_vpa_err)
        max_Gam_vperp_err = maximum(Gam_vperp_err)
        println("max(Gam_vpa_err): ",max_Gam_vpa_err)
        println("max(Gam_vperp_err): ",max_Gam_vperp_err)
        
        # d Gam_|| / d vpa
        for ivperp in 1:nvperp
            @views derivative!(vpa.scratch, Gam_vpa[:,ivperp], vpa, vpa_spectral)
            @. pdf_buffer_1[:,ivperp] = vpa.scratch
        end
        # (1/vperp) d vperp Gam_perp / d vperp
        for ivpa in 1:nvpa
            @views @. vperp.scratch2 = vperp.grid*Gam_vperp[ivpa,:]
            @views derivative!(vperp.scratch, vperp.scratch2, vperp, vperp_spectral)
            @. pdf_buffer_2[ivpa,:] = vperp.scratch/vperp.grid
        end
        
        @. Cssp = pdf_buffer_1 + pdf_buffer_2
        @. Cssp_err = abs(Cssp)
        max_Cssp_err = maximum(Cssp_err)
        zero = 1.0e-6 
        if ( abs(dens - densp) > zero || abs(upar - uparp) > zero || abs(vths - vthsp) > zero)
            println("Cssp test not supported for F_Ms /= F_Ms', ignore result")
        end
        println("max(Cssp_err): ",max_Cssp_err)
        

        if max_Gam_vpa_err > zero && false
           @views heatmap(vperp.grid, vpa.grid, Cssp[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                 windowsize = (360,240), margin = 15pt)
                 outfile = string("fkpl_Cssp.pdf")
                 savefig(outfile)
           @views heatmap(vperp.grid, vpa.grid, Gam_vpa[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                 windowsize = (360,240), margin = 15pt)
                 outfile = string("fkpl_Gam_vpa.pdf")
                 savefig(outfile)
           @views heatmap(vperp.grid, vpa.grid, Gam_vpa_Maxwell[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                 windowsize = (360,240), margin = 15pt)
                 outfile = string("fkpl_Gam_vpa_Maxwell.pdf")
                 savefig(outfile)
           @views heatmap(vperp.grid, vpa.grid, Gam_vpa_err[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                 windowsize = (360,240), margin = 15pt)
                 outfile = string("fkpl_Gam_vpa_err.pdf")
                 savefig(outfile)
        end
        if max_Gam_vperp_err > zero && false
           @views heatmap(vperp.grid, vpa.grid, Gam_vperp[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                 windowsize = (360,240), margin = 15pt)
                 outfile = string("fkpl_Gam_vperp.pdf")
                 savefig(outfile)
           @views heatmap(vperp.grid, vpa.grid, Gam_vperp_Maxwell[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                 windowsize = (360,240), margin = 15pt)
                 outfile = string("fkpl_Gam_vperp_Maxwell.pdf")
                 savefig(outfile)
           @views heatmap(vperp.grid, vpa.grid, Gam_vperp_err[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                 windowsize = (360,240), margin = 15pt)
                 outfile = string("fkpl_Gam_vperp_err.pdf")
                 savefig(outfile)
        end
        return max_Gam_vpa_err, max_Gam_vperp_err, max_Cssp_err
    end
    
    if test_Rosenbluth_integrals
        ngrid = 8
        nscan = 5
        nelement_list = Int[2, 4, 8, 16, 32]
        max_G_err = Array{mk_float,1}(undef,nscan)
        max_H_err = Array{mk_float,1}(undef,nscan)
        max_H_check_err = Array{mk_float,1}(undef,nscan)
        max_dHdvpa_err = Array{mk_float,1}(undef,nscan)
        max_dHdvperp_err = Array{mk_float,1}(undef,nscan)
        max_d2Gdvperp2_err = Array{mk_float,1}(undef,nscan)
        max_d2Gdvpa2_err = Array{mk_float,1}(undef,nscan)
        max_d2Gdvperpdvpa_err = Array{mk_float,1}(undef,nscan)
        expected = Array{mk_float,1}(undef,nscan)
        expected_nelement_scaling!(expected,nelement_list,ngrid,nscan)
        expected_label = L"(1/N_{el})^{n_g - 1}"
        
        
        for iscan in 1:nscan
            local nelement = nelement_list[iscan]
            ((max_G_err[iscan], max_H_err[iscan], 
            max_H_check_err[iscan], max_dHdvpa_err[iscan],
            max_dHdvperp_err[iscan], max_d2Gdvperp2_err[iscan],
            max_d2Gdvpa2_err[iscan], max_d2Gdvperpdvpa_err[iscan])
            = test_Rosenbluth_potentials(nelement,ngrid))
        end
        fontsize = 8
        ytick_sequence = Array([1.0e-13,1.0e-12,1.0e-11,1.0e-10,1.0e-9,1.0e-8,1.0e-7,1.0e-6,1.0e-5,1.0e-4,1.0e-3,1.0e-2,1.0e-1,1.0e-0,1.0e1])
        xlabel = L"N_{element}"
        Glabel = L"\epsilon(G)"
        Hlabel = L"\epsilon(H)"
        dHdvpalabel = L"\epsilon(dH/d v_{\|\|})"
        dHdvperplabel = L"\epsilon(dH/d v_{\perp})"
        d2Gdvperp2label = L"\epsilon(d^2G/d v_{\perp}^2)"
        d2Gdvpa2label = L"\epsilon(d^2G/d v_{\|\|}^2)"
        d2Gdvperpdvpalabel = L"\epsilon(d^2G/d v_{\perp} d v_{\|\|})"
        plot(nelement_list, [max_G_err,max_H_check_err,max_dHdvpa_err,max_dHdvperp_err,max_d2Gdvperp2_err,max_d2Gdvpa2_err,max_d2Gdvperpdvpa_err, expected],
        xlabel=xlabel, label=[Glabel Hlabel dHdvpalabel dHdvperplabel d2Gdvperp2label d2Gdvpa2label d2Gdvperpdvpalabel expected_label], ylabel="",
         shape =:circle, xscale=:log10, yscale=:log10, xticks = (nelement_list, nelement_list), yticks = (ytick_sequence, ytick_sequence), markersize = 5, linewidth=2, 
          xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, legendfontsize = fontsize,
          foreground_color_legend = nothing, background_color_legend = nothing, legend=:bottomleft)
        outfile = "fkpl_coeffs_analytical_test.pdf"
        savefig(outfile)
        println(outfile)
        
        for iscan in 1:nscan
            local nelement = nelement_list[iscan]
            ((max_G_err[iscan], max_H_err[iscan], 
            max_H_check_err[iscan], max_dHdvpa_err[iscan],
            max_dHdvperp_err[iscan], max_d2Gdvperp2_err[iscan],
            max_d2Gdvpa2_err[iscan], max_d2Gdvperpdvpa_err[iscan])
            = test_Rosenbluth_potentials(nelement,ngrid,numerical_G = true))
        end
        fontsize = 8
        ytick_sequence = Array([1.0e-13,1.0e-12,1.0e-11,1.0e-10,1.0e-9,1.0e-8,1.0e-7,1.0e-6,1.0e-5,1.0e-4,1.0e-3,1.0e-2,1.0e-1,1.0e-0,1.0e1])
        xlabel = L"N_{element}"
        Glabel = L"\epsilon(G)"
        Hlabel = L"\epsilon(H)"
        dHdvpalabel = L"\epsilon(dH/d v_{\|\|})"
        dHdvperplabel = L"\epsilon(dH/d v_{\perp})"
        d2Gdvperp2label = L"\epsilon(d^2G/d v_{\perp}^2)"
        d2Gdvpa2label = L"\epsilon(d^2G/d v_{\|\|}^2)"
        d2Gdvperpdvpalabel = L"\epsilon(d^2G/d v_{\perp} d v_{\|\|})"
        plot(nelement_list, [max_G_err,max_H_check_err,max_dHdvpa_err,max_dHdvperp_err,max_d2Gdvperp2_err,max_d2Gdvpa2_err,max_d2Gdvperpdvpa_err, expected],
        xlabel=xlabel, label=[Glabel Hlabel dHdvpalabel dHdvperplabel d2Gdvperp2label d2Gdvpa2label d2Gdvperpdvpalabel expected_label], ylabel="",
         shape =:circle, xscale=:log10, yscale=:log10, xticks = (nelement_list, nelement_list), yticks = (ytick_sequence, ytick_sequence), markersize = 5, linewidth=2, 
          xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, legendfontsize = fontsize,
          foreground_color_legend = nothing, background_color_legend = nothing, legend=:bottomleft)
        outfile = "fkpl_coeffs_numerical_test.pdf"
        savefig(outfile)
        println(outfile)
    end
    
    if test_collision_operator_fluxes
        ngrid = 8
        nscan = 5
        nelement_list = Int[2, 4, 8, 16, 32]
        max_Gam_vpa_err = Array{mk_float,1}(undef,nscan)
        max_Gam_vperp_err = Array{mk_float,1}(undef,nscan)
        max_Cssp_err = Array{mk_float,1}(undef,nscan)
        expected = Array{mk_float,1}(undef,nscan)
        expected_nelement_scaling!(expected,nelement_list,ngrid,nscan)
        
        for iscan in 1:nscan
         ((max_Gam_vpa_err[iscan], max_Gam_vperp_err[iscan], max_Cssp_err[iscan])
           = test_collision_operator(nelement_list[iscan],ngrid))
        end
        fontsize = 10
        ytick_sequence = Array([1.0e-13,1.0e-12,1.0e-11,1.0e-10,1.0e-9,1.0e-8,1.0e-7,1.0e-6,1.0e-5,1.0e-4,1.0e-3,1.0e-2,1.0e-1,1.0e-0,1.0e1])
        xlabel = L"N_{element}"
        Gam_vpa_label = L"\epsilon(\Gamma_{\|\|})"
        Gam_vperp_label = L"\epsilon(\Gamma_{\perp})"
        Cssp_err_label = L"\epsilon(C)"
        expected_label = L"(1/N_{el})^{n_g - 1}"
        plot(nelement_list, [max_Gam_vpa_err,max_Gam_vperp_err, expected],
        xlabel=xlabel, label=[Gam_vpa_label Gam_vperp_label expected_label], ylabel="",
         shape =:circle, xscale=:log10, yscale=:log10, xticks = (nelement_list, nelement_list), yticks = (ytick_sequence, ytick_sequence), markersize = 5, linewidth=2, 
          xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, legendfontsize = fontsize,
          foreground_color_legend = nothing, background_color_legend = nothing, legend=:bottomleft)
        outfile = "fkpl_fluxes_test.pdf"
        savefig(outfile)
        println(outfile)
    end
    
    if test_Lagrange_integral
        ngrid = 9
        nelement = 4
        test_Lagrange_Rosenbluth_potentials(ngrid,nelement,standalone=true)
    end
    if test_Lagrange_integral_scan
        initialize_comms!()
        ngrid = 5
        nscan = 5
        nelement_list = Int[2, 4, 8, 16, 32]
        #nelement_list = Int[2, 4, 8, 16]
        #nelement_list = Int[2, 4]
        #nelement_list = Int[2]
        max_C_err = Array{mk_float,1}(undef,nscan)
        max_Gvpa_err = Array{mk_float,1}(undef,nscan)
        max_Gvperp_err = Array{mk_float,1}(undef,nscan)
        max_G_err = Array{mk_float,1}(undef,nscan)
        max_H_err = Array{mk_float,1}(undef,nscan)
        max_dHdvpa_err = Array{mk_float,1}(undef,nscan)
        max_dHdvperp_err = Array{mk_float,1}(undef,nscan)
        max_d2Gdvperp2_err = Array{mk_float,1}(undef,nscan)
        max_d2Gdvpa2_err = Array{mk_float,1}(undef,nscan)
        max_d2Gdvperpdvpa_err = Array{mk_float,1}(undef,nscan)
        max_dGdvperp_err = Array{mk_float,1}(undef,nscan)
        expected = Array{mk_float,1}(undef,nscan)
        expected_nelement_scaling!(expected,nelement_list,ngrid,nscan)
        expected_label = L"(1/N_{el})^{n_g - 1}"
        
        for iscan in 1:nscan
            local nelement = nelement_list[iscan]
            ((max_C_err[iscan], max_Gvpa_err[iscan], max_Gvperp_err[iscan], 
            max_G_err[iscan], max_H_err[iscan], 
            max_dHdvpa_err[iscan],
            max_dHdvperp_err[iscan], max_d2Gdvperp2_err[iscan],
            max_d2Gdvpa2_err[iscan], max_d2Gdvperpdvpa_err[iscan],
            max_dGdvperp_err[iscan])
            = test_Lagrange_Rosenbluth_potentials(ngrid,nelement,standalone=false))
        end
        if global_rank[]==0
            fontsize = 8
            ytick_sequence = Array([1.0e-13,1.0e-12,1.0e-11,1.0e-10,1.0e-9,1.0e-8,1.0e-7,1.0e-6,1.0e-5,1.0e-4,1.0e-3,1.0e-2,1.0e-1,1.0e-0,1.0e1])
            xlabel = L"N_{element}"
            Clabel = L"\epsilon(C)"
            Gvpalabel = L"\epsilon(\Gamma_{\|\|})"
            Gvperplabel = L"\epsilon(\Gamma_{\perp})"
            Glabel = L"\epsilon(G)"
            Hlabel = L"\epsilon(H)"
            dHdvpalabel = L"\epsilon(dH/d v_{\|\|})"
            dHdvperplabel = L"\epsilon(dH/d v_{\perp})"
            d2Gdvperp2label = L"\epsilon(d^2G/d v_{\perp}^2)"
            d2Gdvpa2label = L"\epsilon(d^2G/d v_{\|\|}^2)"
            d2Gdvperpdvpalabel = L"\epsilon(d^2G/d v_{\perp} d v_{\|\|})"
            dGdvperplabel = L"\epsilon(dG/d v_{\perp})"
            #println(max_G_err,max_H_err,max_dHdvpa_err,max_dHdvperp_err,max_d2Gdvperp2_err,max_d2Gdvpa2_err,max_d2Gdvperpdvpa_err,max_dGdvperp_err, expected)
            plot(nelement_list, [max_C_err,max_Gvpa_err,max_Gvperp_err,max_G_err,max_H_err,max_dHdvpa_err,max_dHdvperp_err,max_d2Gdvperp2_err,max_d2Gdvpa2_err,max_d2Gdvperpdvpa_err,max_dGdvperp_err, expected],
            xlabel=xlabel, label=[Clabel Gvpalabel Gvperplabel Glabel Hlabel dHdvpalabel dHdvperplabel d2Gdvperp2label d2Gdvpa2label d2Gdvperpdvpalabel dGdvperplabel expected_label], ylabel="",
             shape =:circle, xscale=:log10, yscale=:log10, xticks = (nelement_list, nelement_list), yticks = (ytick_sequence, ytick_sequence), markersize = 5, linewidth=2, 
              xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, legendfontsize = fontsize,
              foreground_color_legend = nothing, background_color_legend = nothing, legend=:bottomleft)
            #outfile = "fkpl_coeffs_numerical_lagrange_integration_test_ngrid_"*string(ngrid)*".pdf"
            outfile = "fkpl_coeffs_numerical_lagrange_integration_test_ngrid_"*string(ngrid)*"_GLL.pdf"
            savefig(outfile)
            println(outfile)
        end
        finalize_comms!()
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
