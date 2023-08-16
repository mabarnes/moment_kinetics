using Printf
using Plots
using LaTeXStrings
using Measures
using MPI
using SpecialFunctions: erf, ellipe, ellipk
using FastGaussQuadrature
using Dates
using LinearAlgebra: mul!

import moment_kinetics
using moment_kinetics.input_structs: grid_input, advection_input
using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.chebyshev: setup_chebyshev_pseudospectral
using moment_kinetics.gauss_legendre: setup_gausslegendre_pseudospectral, gausslegendre_mass_matrix_solve!
using moment_kinetics.fokker_planck: evaluate_RMJ_collision_operator!
using moment_kinetics.fokker_planck: calculate_Rosenbluth_potentials!
#using moment_kinetics.fokker_planck: calculate_Rosenbluth_H_from_G!
using moment_kinetics.fokker_planck: init_fokker_planck_collisions
using moment_kinetics.fokker_planck: calculate_collisional_fluxes, calculate_Maxwellian_Rosenbluth_coefficients
using moment_kinetics.fokker_planck: Cflux_vpa_Maxwellian_inputs, Cflux_vperp_Maxwellian_inputs
using moment_kinetics.fokker_planck: calculate_Rosenbluth_H_from_G!
using moment_kinetics.fokker_planck: d2Gdvpa2, dGdvperp, d2Gdvperpdvpa, d2Gdvperp2
using moment_kinetics.fokker_planck: dHdvpa, dHdvperp, Cssp_Maxwellian_inputs
using moment_kinetics.fokker_planck: F_Maxwellian, dFdvpa_Maxwellian, dFdvperp_Maxwellian
using moment_kinetics.fokker_planck: d2Fdvpa2_Maxwellian, d2Fdvperpdvpa_Maxwellian, d2Fdvperp2_Maxwellian
using moment_kinetics.type_definitions: mk_float, mk_int
using moment_kinetics.calculus: derivative!, second_derivative!
using moment_kinetics.velocity_moments: get_density, get_upar, get_ppar, get_pperp, get_pressure
using moment_kinetics.velocity_moments: integrate_over_vspace
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

function expected_nelement_integral_scaling!(expected,nelement_list,ngrid,nscan)
    for iscan in 1:nscan
        expected[iscan] = (1.0/nelement_list[iscan])^(ngrid+1)
    end
end
"""
L2norm assuming the input is the 
absolution error ff_err = ff - ff_exact
We compute sqrt( int (ff_err)^2 d^3 v / int d^3 v)
where the volume of velocity space is finite
"""
function L2norm_vspace(ff_err,vpa,vperp)
    ff_ones = copy(ff_err)
    @. ff_ones = 1.0
    gg = copy(ff_err)
    @. gg = (ff_err)^2
    num = integrate_over_vspace(@view(gg[:,:]), vpa.grid, 0, vpa.wgts, vperp.grid, 0, vperp.wgts)
    denom = integrate_over_vspace(@view(ff_ones[:,:]), vpa.grid, 0, vpa.wgts, vperp.grid, 0, vperp.wgts)
    L2norm = sqrt(num/denom)
    return L2norm
end

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
    
    test_Lagrange_integral = false #true
    test_Lagrange_integral_scan = true
    
    function test_Lagrange_Rosenbluth_potentials(ngrid,nelement; standalone=true)
        # set up grids for input Maxwellian
        vpa, vperp, vpa_spectral, vperp_spectral =  init_grids(nelement,ngrid)
        # set up necessary inputs for collision operator functions 
        nvperp = vperp.n
        nvpa = vpa.n
        #ivpa_field = floor(mk_int,nvpa/2 + nvpa/8 - 1)
        #ivperp_field = floor(mk_int,nvperp/10)
        ivpa_field = floor(mk_int,nvpa/2 + nvpa/16)
        ivperp_field = floor(mk_int,nvperp/8 -1)
        #ivpa_field = floor(mk_int,nvpa/2)
        #ivperp_field = floor(mk_int,1)
        #ivpa_field = 37
        #ivperp_field = 15
        println("Investigating vpa = ",vpa.grid[ivpa_field], " vperp = ",vperp.grid[ivperp_field])
        
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
        dfsdvpa_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
        d2fsdvpa2_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
        dfsdvperp_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
        d2fsdvperpdvpa_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
        d2fsdvperp2_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
        dfsdvpa_err = Array{mk_float,2}(undef,nvpa,nvperp)
        d2fsdvpa2_err = Array{mk_float,2}(undef,nvpa,nvperp)
        dfsdvperp_err = Array{mk_float,2}(undef,nvpa,nvperp)
        d2fsdvperpdvpa_err = Array{mk_float,2}(undef,nvpa,nvperp)
        d2fsdvperp2_err = Array{mk_float,2}(undef,nvpa,nvperp)
        
        fsp_in = Array{mk_float,2}(undef,nvpa,nvperp)
        d2fspdvpa2 = Array{mk_float,2}(undef,nvpa,nvperp)
        dfspdvperp = Array{mk_float,2}(undef,nvpa,nvperp)
        dfspdvpa = Array{mk_float,2}(undef,nvpa,nvperp)
        d2fspdvperpdvpa = Array{mk_float,2}(undef,nvpa,nvperp)
        d2fspdvperp2 = Array{mk_float,2}(undef,nvpa,nvperp)
        d2fspdvpa2_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
        dfspdvperp_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
        dfspdvpa_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
        d2fspdvperpdvpa_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
        d2fspdvperp2_Maxwell = Array{mk_float,2}(undef,nvpa,nvperp)
        d2fspdvpa2_err = Array{mk_float,2}(undef,nvpa,nvperp)
        dfspdvpa_err = Array{mk_float,2}(undef,nvpa,nvperp)
        dfspdvperp_err = Array{mk_float,2}(undef,nvpa,nvperp)
        d2fspdvperpdvpa_err = Array{mk_float,2}(undef,nvpa,nvperp)
        d2fspdvperp2_err = Array{mk_float,2}(undef,nvpa,nvperp)
        
        #G_weights = Array{mk_float,4}(undef,nvpa,nvperp,nvpa,nvperp)
        G_weights = allocate_shared_float(1,1,nvpa,nvperp)
        G1_weights = allocate_shared_float(1,1,nvpa,nvperp)
        G2_weights = allocate_shared_float(1,1,nvpa,nvperp)
        G3_weights = allocate_shared_float(1,1,nvpa,nvperp)
        Gsp = allocate_shared_float(1,1)
        d2Gspdvpa2 = allocate_shared_float(1,1)
        dGspdvperp = allocate_shared_float(1,1)
        d2Gspdvperpdvpa = allocate_shared_float(1,1)
        d2Gspdvperp2 = allocate_shared_float(1,1)
        #Gsp = Array{mk_float,2}(undef,1,1)
        G_Maxwell = Array{mk_float,2}(undef,1,1)
        G_err = allocate_shared_float(1,1)
        d2Gdvpa2_Maxwell = Array{mk_float,2}(undef,1,1)
        d2Gdvpa2_err = allocate_shared_float(1,1)
        dGdvperp_Maxwell = Array{mk_float,2}(undef,1,1)
        dGdvperp_err = allocate_shared_float(1,1)
        d2Gdvperpdvpa_Maxwell = Array{mk_float,2}(undef,1,1)
        d2Gdvperpdvpa_err = allocate_shared_float(1,1)
        d2Gdvperp2_Maxwell = Array{mk_float,2}(undef,1,1)
        d2Gdvperp2_err = allocate_shared_float(1,1)
        
        n_weights = allocate_shared_float(1,1,nvpa,nvperp)
        nsp = allocate_shared_float(1,1)
        n_err = allocate_shared_float(1,1)
        
        H_weights = allocate_shared_float(1,1,nvpa,nvperp)
        H1_weights = allocate_shared_float(1,1,nvpa,nvperp)
        H2_weights = allocate_shared_float(1,1,nvpa,nvperp)
        H3_weights = allocate_shared_float(1,1,nvpa,nvperp)
        Hsp_from_Gsp = allocate_shared_float(1,1)
        Hsp = allocate_shared_float(1,1)
        dHspdvpa = allocate_shared_float(1,1)
        dHspdvperp = allocate_shared_float(1,1)
        #Gsp = Array{mk_float,2}(undef,nvpa,nvperp)
        H_Maxwell = Array{mk_float,2}(undef,1,1)
        H_err = allocate_shared_float(1,1)
        dHdvpa_Maxwell = Array{mk_float,2}(undef,1,1)
        dHdvpa_err = allocate_shared_float(1,1)
        dHdvperp_Maxwell = Array{mk_float,2}(undef,1,1)
        dHdvperp_err = allocate_shared_float(1,1)
        
        
        @serial_region begin
            println("setting up input arrays   ", Dates.format(now(), dateformat"H:MM:SS"))
        end
        
        # set up test Maxwellian
        # species s 
        denss = 1.0 #3.0/4.0
        upars = 0.0 #2.0/3.0
        ppars = 1.0 #2.0/3.0
        pperps = 1.0 #2.0/3.0
        press = get_pressure(ppars,pperps) 
        ms = 1.0
        vths = get_vth(press,denss,ms)
        # species sp 
        denssp = 1.0 #3.0/4.0
        uparsp = 0.0 #2.0/3.0
        pparsp = 1.0 #2.0/3.0
        pperpsp = 1.0 #2.0/3.0
        pressp = get_pressure(pparsp,pperpsp) 
        msp = 1.0
        vthsp = get_vth(pressp,denssp,msp)
        
        nussp = 1.0
        for ivperp in 1:nvperp
            for ivpa in 1:nvpa
                fs_in[ivpa,ivperp] = F_Maxwellian(denss,upars,vths,vpa,vperp,ivpa,ivperp) #(denss/vths^3)*exp( - ((vpa.grid[ivpa]-upar)^2 + vperp.grid[ivperp]^2)/vths^2 ) 
                dfsdvpa_Maxwell[ivpa,ivperp] = dFdvpa_Maxwellian(denss,upars,vths,vpa,vperp,ivpa,ivperp)
                d2fsdvpa2_Maxwell[ivpa,ivperp] = d2Fdvpa2_Maxwellian(denss,upars,vths,vpa,vperp,ivpa,ivperp)
                dfsdvperp_Maxwell[ivpa,ivperp] = dFdvperp_Maxwellian(denss,upars,vths,vpa,vperp,ivpa,ivperp)
                d2fsdvperpdvpa_Maxwell[ivpa,ivperp] = d2Fdvperpdvpa_Maxwellian(denss,upars,vths,vpa,vperp,ivpa,ivperp)
                d2fsdvperp2_Maxwell[ivpa,ivperp] = d2Fdvperp2_Maxwellian(denss,upars,vths,vpa,vperp,ivpa,ivperp)
                
                fsp_in[ivpa,ivperp] = F_Maxwellian(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp) #(denss/vths^3)*exp( - ((vpa.grid[ivpa]-upar)^2 + vperp.grid[ivperp]^2)/vths^2 ) 
                dfspdvpa_Maxwell[ivpa,ivperp] = dFdvpa_Maxwellian(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
                d2fspdvpa2_Maxwell[ivpa,ivperp] = d2Fdvpa2_Maxwellian(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
                dfspdvperp_Maxwell[ivpa,ivperp] = dFdvperp_Maxwellian(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
                d2fspdvperpdvpa_Maxwell[ivpa,ivperp] = d2Fdvperpdvpa_Maxwellian(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
                d2fspdvperp2_Maxwell[ivpa,ivperp] = d2Fdvperp2_Maxwellian(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
            end
        end

        ivpa = ivpa_field
        ivperp = ivperp_field
        G_Maxwell[1,1] = G_Maxwellian(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
        H_Maxwell[1,1] = H_Maxwellian(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
        d2Gdvpa2_Maxwell[1,1] = d2Gdvpa2(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
        dGdvperp_Maxwell[1,1] = dGdvperp(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
        d2Gdvperpdvpa_Maxwell[1,1] = d2Gdvperpdvpa(denssp,upars,vthsp,vpa,vperp,ivpa,ivperp)
        d2Gdvperp2_Maxwell[1,1] = d2Gdvperp2(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
        dHdvperp_Maxwell[1,1] = dHdvperp(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
        dHdvpa_Maxwell[1,1] = dHdvpa(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
        
        for ivperp in 1:nvperp
            # s
            @views derivative!(vpa.scratch, fs_in[:,ivperp], vpa, vpa_spectral)
            @. dfsdvpa[:,ivperp] = vpa.scratch
            @views derivative!(vpa.scratch2, vpa.scratch, vpa, vpa_spectral)
            @. d2fsdvpa2[:,ivperp] = vpa.scratch2
            # sp
            @views derivative!(vpa.scratch, fsp_in[:,ivperp], vpa, vpa_spectral)
            @. dfspdvpa[:,ivperp] = vpa.scratch
            @views derivative!(vpa.scratch2, vpa.scratch, vpa, vpa_spectral)
            @. d2fspdvpa2[:,ivperp] = vpa.scratch2
        end
        if vpa.discretization == "gausslegendre_pseudospectral"
            @serial_region begin
                println("use weak-form second derivative for vpa")
            end
            for ivperp in 1:nvperp
               @views second_derivative!(vpa.scratch2, fs_in[:,ivperp], vpa, vpa_spectral)
               @. d2fsdvpa2[:,ivperp] = vpa.scratch2 
               @views second_derivative!(vpa.scratch2, fsp_in[:,ivperp], vpa, vpa_spectral)
               @. d2fspdvpa2[:,ivperp] = vpa.scratch2 
            end
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
        
        # error analysis of distribution function
        @serial_region begin
            @. dfsdvpa_err = abs(dfsdvpa - dfsdvpa_Maxwell)
            max_dfsdvpa_err = maximum(dfsdvpa_err)
            println("max_dfsdvpa_err: ",max_dfsdvpa_err)
            @. d2fsdvpa2_err = abs(d2fsdvpa2 - d2fsdvpa2_Maxwell)
            max_d2fsdvpa2_err = maximum(d2fsdvpa2_err)
            println("max_d2fsdvpa2_err: ",max_d2fsdvpa2_err)
            @. dfsdvperp_err = abs(dfsdvperp - dfsdvperp_Maxwell)
            max_dfsdvperp_err = maximum(dfsdvperp_err)
            println("max_dfsdvperp_err: ",max_dfsdvperp_err)
            @. d2fsdvperpdvpa_err = abs(d2fsdvperpdvpa - d2fsdvperpdvpa_Maxwell)
            max_d2fsdvperpdvpa_err = maximum(d2fsdvperpdvpa_err)
            println("max_d2fsdvperpdvpa_err: ",max_d2fsdvperpdvpa_err)
            @. d2fsdvperp2_err = abs(d2fsdvperp2 - d2fsdvperp2_Maxwell)
            max_d2fsdvperp2_err = maximum(d2fsdvperp2_err)
            println("max_d2fsdvperp2_err: ",max_d2fsdvperp2_err)
            
            @. dfspdvpa_err = abs(dfspdvpa - dfspdvpa_Maxwell)
            max_dfspdvpa_err = maximum(dfspdvpa_err)
            @. d2fspdvpa2_err = abs(d2fspdvpa2 - d2fspdvpa2_Maxwell)
            max_d2fspdvpa2_err = maximum(d2fspdvpa2_err)
            println("max_d2fspdvpa2_err: ",max_d2fspdvpa2_err)
            @. dfspdvperp_err = abs(dfspdvperp - dfspdvperp_Maxwell)
            max_dfspdvperp_err = maximum(dfspdvperp_err)
            println("max_dfspdvperp_err: ",max_dfspdvperp_err)
            @. d2fspdvperpdvpa_err = abs(d2fspdvperpdvpa - d2fspdvperpdvpa_Maxwell)
            max_d2fspdvperpdvpa_err = maximum(d2fspdvperpdvpa_err)
            println("max_d2fspdvperpdvpa_err: ",max_d2fspdvperpdvpa_err)
            @. d2fspdvperp2_err = abs(d2fspdvperp2 - d2fspdvperp2_Maxwell)
            max_d2fspdvperp2_err = maximum(d2fspdvperp2_err)
            println("max_d2fspdvperp2_err: ",max_d2fspdvperp2_err)
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
        
        function get_scaled_x_w!(x_scaled, w_scaled, x_legendre, w_legendre, x_laguerre, w_laguerre, node_min, node_max, nodes, igrid_coord, coord_val)
            #println("nodes ",nodes)
            zero = 1.0e-10 
            @. x_scaled = 0.0
            @. w_scaled = 0.0
            nnodes = size(nodes,1)
            nquad_legendre = size(x_legendre,1)
            nquad_laguerre = size(x_laguerre,1)
            # assume x_scaled, w_scaled are arrays of length 2*nquad
            # use only nquad points for most elements, but use 2*nquad for
            # elements with interior divergences
            #println("coord: ",coord_val," node_max: ",node_max," node_min: ",node_min) 
            if abs(coord_val - node_max) < zero # divergence at upper endpoint 
                node_cut = nodes[nnodes-1]
                
                n = nquad_laguerre + nquad_legendre
                shift = 0.5*(node_min + node_cut)
                scale = 0.5*(node_cut - node_min)
                @. x_scaled[1:nquad_legendre] = scale*x_legendre + shift
                @. w_scaled[1:nquad_legendre] = scale*w_legendre

                @. x_scaled[1+nquad_legendre:n] = node_max + (node_cut - node_max)*exp(-x_laguerre)
                @. w_scaled[1+nquad_legendre:n] = (node_max - node_cut)*w_laguerre
                
                nquad_coord = n
                #println("upper divergence")
            elseif abs(coord_val - node_min) < zero # divergence at lower endpoint
                n = nquad_laguerre + nquad_legendre
                nquad = size(x_laguerre,1)
                node_cut = nodes[2]
                for j in 1:nquad_laguerre
                    x_scaled[nquad_laguerre+1-j] = node_min + (node_cut - node_min)*exp(-x_laguerre[j])
                    w_scaled[nquad_laguerre+1-j] = (node_cut - node_min)*w_laguerre[j]
                end
                shift = 0.5*(node_max + node_cut)
                scale = 0.5*(node_max - node_cut)
                @. x_scaled[1+nquad_laguerre:n] = scale*x_legendre + shift
                @. w_scaled[1+nquad_laguerre:n] = scale*w_legendre

                nquad_coord = n
                #println("lower divergence")
            else #if (coord_val - node_min)*(coord_val - node_max) < - zero # interior divergence
                #println("igrid_coord ", igrid_coord, " ", nodes[igrid_coord]," ", coord_val)
                n = 2*nquad_laguerre
                node_cut_high = (nodes[igrid_coord+1] + nodes[igrid_coord])/2.0
                if igrid_coord == 1
                    # exception for vperp coordinate near orgin
                    k = 0
                    node_cut_low = node_min
                    nquad_coord = nquad_legendre + 2*nquad_laguerre
                else
                    # fill in lower Gauss-Legendre points
                    node_cut_low = (nodes[igrid_coord-1] + nodes[igrid_coord])/2.0
                    shift = 0.5*(node_cut_low + node_min)
                    scale = 0.5*(node_cut_low - node_min)
                    @. x_scaled[1:nquad_legendre] = scale*x_legendre + shift
                    @. w_scaled[1:nquad_legendre] = scale*w_legendre
                    k = nquad_legendre
                    nquad_coord = 2*(nquad_laguerre + nquad_legendre)
                end
                # lower half of domain  
                for j in 1:nquad_laguerre  
                    x_scaled[k+j] = coord_val + (node_cut_low - coord_val)*exp(-x_laguerre[j])
                    w_scaled[k+j] = (coord_val - node_cut_low)*w_laguerre[j]
                end  
                # upper half of domain
                for j in 1:nquad_laguerre
                    x_scaled[k+n+1-j] = coord_val + (node_cut_high - coord_val)*exp(-x_laguerre[j])
                    w_scaled[k+n+1-j] = (node_cut_high - coord_val)*w_laguerre[j]
                end
                # fill in upper Gauss-Legendre points
                shift = 0.5*(node_cut_high + node_max)
                scale = 0.5*(node_max - node_cut_high)
                @. x_scaled[k+n+1:nquad_coord] = scale*x_legendre + shift
                @. w_scaled[k+n+1:nquad_coord] = scale*w_legendre
                
                #println("intermediate divergence")
            #else # no divergences
            #    nquad = size(x_legendre,1) 
            #    shift = 0.5*(node_min + node_max)
            #    scale = 0.5*(node_max - node_min)
            #    @. x_scaled[1:nquad] = scale*x_legendre + shift
            #    @. w_scaled[1:nquad] = scale*w_legendre
            #    #println("no divergence")
            #    nquad_coord = nquad
            end
            #println("x_scaled",x_scaled)
            #println("w_scaled",w_scaled)
            return nquad_coord
        end
        
        function get_scaled_x_w_no_divergences!(x_scaled, w_scaled, x_legendre, w_legendre, node_min, node_max)
            zero = 1.0e-6 
            @. x_scaled = 0.0
            @. w_scaled = 0.0
            #println("coord: ",coord_val," node_max: ",node_max," node_min: ",node_min) 
            nquad = size(x_legendre,1) 
            shift = 0.5*(node_min + node_max)
            scale = 0.5*(node_max - node_min)
            @. x_scaled[1:nquad] = scale*x_legendre + shift
            @. w_scaled[1:nquad] = scale*w_legendre
            #println("x_scaled",x_scaled)
            #println("w_scaled",w_scaled)
            return nquad
        end
        
        # function returns 1 if igrid = 1 or 0 if 1 < igrid <= ngrid
        function ng_low(igrid,ngrid)
            return floor(mk_int, (ngrid - igrid)/(ngrid - 1))
        end
        # function returns 1 if igrid = ngrid or 0 if 1 =< igrid < ngrid
        function ng_hi(igrid,ngrid)
            return floor(mk_int, igrid/ngrid)
        end
        # function returns 1 for nelement >= ielement > 1, 0 for ielement =1 
        function nel_low(ielement,nelement)
            return floor(mk_int, (ielement - 2 + nelement)/nelement)
        end
        # function returns 1 for nelement > ielement >= 1, 0 for ielement =nelement 
        function nel_hi(ielement,nelement)
            return 1- floor(mk_int, ielement/nelement)
        end
        
        function local_element_integration!(G_weights,G1_weights,G2_weights,G3_weights,
                                    H_weights,H1_weights,H2_weights,H3_weights,n_weights,
                                    nquad_vpa,ielement_vpa,vpa_nodes,vpa, # info about primed vperp grids
                                    nquad_vperp,ielement_vperp,vperp_nodes,vperp, # info about primed vperp grids
                                    x_vpa, w_vpa, x_vperp, w_vperp, # points and weights for primed (source) grids
                                    vpa_val, vperp_val, ivpa, ivperp) # values and indices for unprimed (field) grids
            for igrid_vperp in 1:vperp.ngrid
                for igrid_vpa in 1:vpa.ngrid
                    # get grid index for point on full grid  
                    ivpap = vpa.igrid_full[igrid_vpa,ielement_vpa]   
                    ivperpp = vperp.igrid_full[igrid_vperp,ielement_vperp]   
                    # carry out integration over Lagrange polynomial at this node, on this element
                    for kvperp in 1:nquad_vperp
                        for kvpa in 1:nquad_vpa 
                            x_kvpa = x_vpa[kvpa]
                            x_kvperp = x_vperp[kvperp]
                            w_kvperp = w_vperp[kvperp]
                            w_kvpa = w_vpa[kvpa]
                            denom = (vpa_val - x_kvpa)^2 + (vperp_val + x_kvperp)^2 
                            mm = min(4.0*vperp_val*x_kvperp/denom,1.0 - 1.0e-15)
                            #mm = 4.0*vperp_val*x_kvperp/denom/(1.0 + 10^-15)
                            #mm = 4.0*vperp_val*x_kvperp/denom
                            prefac = sqrt(denom)
                            ellipe_mm = ellipe(mm) 
                            ellipk_mm = ellipk(mm) 
                            #if mm_test > 1.0
                            #    println("mm: ",mm_test," ellipe: ",ellipe_mm," ellipk: ",ellipk_mm)
                            #end
                            G_elliptic_integral_factor = 2.0*ellipe_mm*prefac/pi
                            G1_elliptic_integral_factor = -(2.0*prefac/pi)*( (2.0 - mm)*ellipe_mm - 2.0*(1.0 - mm)*ellipk_mm )/(3.0*mm)
                            G2_elliptic_integral_factor = (2.0*prefac/pi)*( (7.0*mm^2 + 8.0*mm - 8.0)*ellipe_mm + 4.0*(2.0 - mm)*(1.0 - mm)*ellipk_mm )/(15.0*mm^2)
                            G3_elliptic_integral_factor = (2.0*prefac/pi)*( 8.0*(mm^2 - mm + 1.0)*ellipe_mm - 4.0*(2.0 - mm)*(1.0 - mm)*ellipk_mm )/(15.0*mm^2)
                            H_elliptic_integral_factor = 2.0*ellipk_mm/(pi*prefac)
                            H1_elliptic_integral_factor = -(2.0/(pi*prefac))*( (mm-2.0)*(ellipk_mm/mm) + (2.0*ellipe_mm/mm) )
                            H2_elliptic_integral_factor = (2.0/(pi*prefac))*( (3.0*mm^2 - 8.0*mm + 8.0)*(ellipk_mm/(3.0*mm^2)) + (4.0*mm - 8.0)*ellipe_mm/(3.0*mm^2) )
                            lagrange_poly_vpa = lagrange_poly(igrid_vpa,vpa_nodes,x_kvpa)
                            lagrange_poly_vperp = lagrange_poly(igrid_vperp,vperp_nodes,x_kvperp)
                            
                            (G_weights[1,1,ivpap,ivperpp] += 
                                lagrange_poly_vpa*lagrange_poly_vperp*
                                G_elliptic_integral_factor*x_kvperp*w_kvperp*w_kvpa*2.0/sqrt(pi))
                            
                            (G1_weights[1,1,ivpap,ivperpp] += 
                                lagrange_poly_vpa*lagrange_poly_vperp*
                                G1_elliptic_integral_factor*x_kvperp*w_kvperp*w_kvpa*2.0/sqrt(pi))
                            
                            (G2_weights[1,1,ivpap,ivperpp] += 
                                lagrange_poly_vpa*lagrange_poly_vperp*
                                G2_elliptic_integral_factor*x_kvperp*w_kvperp*w_kvpa*2.0/sqrt(pi))
                            
                            (G3_weights[1,1,ivpap,ivperpp] += 
                                lagrange_poly_vpa*lagrange_poly_vperp*
                                G3_elliptic_integral_factor*w_kvperp*w_kvpa*2.0/sqrt(pi))
                            
                            (H_weights[1,1,ivpap,ivperpp] += 
                                lagrange_poly_vpa*lagrange_poly_vperp*
                                H_elliptic_integral_factor*x_kvperp*w_kvperp*w_kvpa*2.0/sqrt(pi))
                                
                            (H1_weights[1,1,ivpap,ivperpp] += 
                                lagrange_poly_vpa*lagrange_poly_vperp*
                                H1_elliptic_integral_factor*x_kvperp*w_kvperp*w_kvpa*2.0/sqrt(pi))
                                
                            (H2_weights[1,1,ivpap,ivperpp] += 
                                lagrange_poly_vpa*lagrange_poly_vperp*
                                (H1_elliptic_integral_factor*vperp_val - H2_elliptic_integral_factor*x_kvperp)*
                                x_kvperp*w_kvperp*w_kvpa*2.0/sqrt(pi))
                            (H3_weights[1,1,ivpap,ivperpp] += 
                                lagrange_poly_vpa*lagrange_poly_vperp*
                                H_elliptic_integral_factor*(vpa_val - x_kvpa)*
                                x_kvperp*w_kvperp*w_kvpa*2.0/sqrt(pi))
                            
                            (n_weights[1,1,ivpap,ivperpp] += 
                                lagrange_poly_vpa*lagrange_poly_vperp*
                                x_kvperp*w_kvperp*w_kvpa*2.0/sqrt(pi))
                        end
                    end
                end
            end
            return nothing
        end
        
        function loop_over_vpa_elements!(G_weights,G1_weights,G2_weights,G3_weights,
                                    H_weights,H1_weights,H2_weights,H3_weights,n_weights,
                                    vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vperp grids
                                    vperp,ielement_vperpp,
                                    #nquad_vperp,ielement_vperpp,vperp_nodes,vperp, # info about primed vperp grids
                                    x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                                    igrid_vpa, igrid_vperp, vpa_val, vperp_val, ivpa, ivperp)
            
            vperp_nodes = get_nodes(vperp,ielement_vperpp)
            vperp_max = vperp_nodes[end]
            vperp_min = vperp_nodes[1]*nel_low(ielement_vperpp,vperp.nelement_local) 
            nquad_vperp = get_scaled_x_w_no_divergences!(x_vperp, w_vperp, x_legendre, w_legendre, vperp_min, vperp_max)
            for ielement_vpap in 1:ielement_vpa_low-1 
                # do integration over part of the domain with no divergences
                vpa_nodes = get_nodes(vpa,ielement_vpap)
                vpa_min, vpa_max = vpa_nodes[1], vpa_nodes[end]
                nquad_vpa = get_scaled_x_w_no_divergences!(x_vpa, w_vpa, x_legendre, w_legendre, vpa_min, vpa_max)
                local_element_integration!(G_weights,G1_weights,G2_weights,G3_weights,
                            H_weights,H1_weights,H2_weights,H3_weights,n_weights,
                            nquad_vpa,ielement_vpap,vpa_nodes,vpa,
                            nquad_vperp,ielement_vperpp,vperp_nodes,vperp,
                            x_vpa, w_vpa, x_vperp, w_vperp, 
                            vpa_val, vperp_val, ivpa, ivperp)
            end
            nquad_vperp = get_scaled_x_w!(x_vperp, w_vperp, x_legendre, w_legendre, x_laguerre, w_laguerre, vperp_min, vperp_max, vperp_nodes, igrid_vperp, vperp_val)
            for ielement_vpap in ielement_vpa_low:ielement_vpa_hi
                # use general grid function that checks divergences
                vpa_nodes = get_nodes(vpa,ielement_vpap)
                vpa_min, vpa_max = vpa_nodes[1], vpa_nodes[end]
                #nquad_vpa = get_scaled_x_w_no_divergences!(x_vpa, w_vpa, x_legendre, w_legendre, vpa_min, vpa_max)
                nquad_vpa = get_scaled_x_w!(x_vpa, w_vpa, x_legendre, w_legendre, x_laguerre, w_laguerre, vpa_min, vpa_max, vpa_nodes, igrid_vpa, vpa_val)
                local_element_integration!(G_weights,G1_weights,G2_weights,G3_weights,
                            H_weights,H1_weights,H2_weights,H3_weights,n_weights,
                            nquad_vpa,ielement_vpap,vpa_nodes,vpa,
                            nquad_vperp,ielement_vperpp,vperp_nodes,vperp,
                            x_vpa, w_vpa, x_vperp, w_vperp, 
                            vpa_val, vperp_val, ivpa, ivperp)
            end
            nquad_vperp = get_scaled_x_w_no_divergences!(x_vperp, w_vperp, x_legendre, w_legendre, vperp_min, vperp_max)
            for ielement_vpap in ielement_vpa_hi+1:vpa.nelement_local
                # do integration over part of the domain with no divergences
                vpa_nodes = get_nodes(vpa,ielement_vpap)
                vpa_min, vpa_max = vpa_nodes[1], vpa_nodes[end]
                nquad_vpa = get_scaled_x_w_no_divergences!(x_vpa, w_vpa, x_legendre, w_legendre, vpa_min, vpa_max)
                local_element_integration!(G_weights,G1_weights,G2_weights,G3_weights,
                            H_weights,H1_weights,H2_weights,H3_weights,n_weights,
                            nquad_vpa,ielement_vpap,vpa_nodes,vpa,
                            nquad_vperp,ielement_vperpp,vperp_nodes,vperp,
                            x_vpa, w_vpa, x_vperp, w_vperp, 
                            vpa_val, vperp_val, ivpa, ivperp)
                            
            end
            return nothing
        end
        
        function loop_over_vpa_elements_no_divergences!(G_weights,G1_weights,G2_weights,G3_weights,
                                    H_weights,H1_weights,H2_weights,H3_weights,n_weights,
                                    vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vperp grids
                                    nquad_vperp,ielement_vperpp,vperp_nodes,vperp, # info about primed vperp grids
                                    x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                                    vpa_val, vperp_val, ivpa, ivperp)
            for ielement_vpap in 1:vpa.nelement_local
                # do integration over part of the domain with no divergences
                vpa_nodes = get_nodes(vpa,ielement_vpap)
                vpa_min, vpa_max = vpa_nodes[1], vpa_nodes[end]
                nquad_vpa = get_scaled_x_w_no_divergences!(x_vpa, w_vpa, x_legendre, w_legendre, vpa_min, vpa_max)
                local_element_integration!(G_weights,G1_weights,G2_weights,G3_weights,
                            H_weights,H1_weights,H2_weights,H3_weights,n_weights,
                            nquad_vpa,ielement_vpap,vpa_nodes,vpa,
                            nquad_vperp,ielement_vperpp,vperp_nodes,vperp,
                            x_vpa, w_vpa, x_vperp, w_vperp, 
                            vpa_val, vperp_val, ivpa, ivperp)
                            
            end
            return nothing
        end
        
        function loop_over_vperp_vpa_elements!(G_weights,G1_weights,G2_weights,G3_weights,
                        H_weights,H1_weights,H2_weights,H3_weights,n_weights,
                        vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vpa grids
                        vperp,ielement_vperp_low,ielement_vperp_hi, # info about primed vperp grids
                        x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                        igrid_vpa, igrid_vperp, vpa_val, vperp_val, ivpa, ivperp)
            for ielement_vperpp in 1:ielement_vperp_low-1
                
                vperp_nodes = get_nodes(vperp,ielement_vperpp)
                vperp_max = vperp_nodes[end]
                vperp_min = vperp_nodes[1]*nel_low(ielement_vperpp,vperp.nelement_local) 
                nquad_vperp = get_scaled_x_w_no_divergences!(x_vperp, w_vperp, x_legendre, w_legendre, vperp_min, vperp_max)
                loop_over_vpa_elements_no_divergences!(G_weights,G1_weights,G2_weights,G3_weights,
                        H_weights,H1_weights,H2_weights,H3_weights,n_weights,
                        vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vpa grids
                        nquad_vperp,ielement_vperpp,vperp_nodes,vperp, # info about primed vperp grids
                        x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                        vpa_val, vperp_val, ivpa, ivperp)
            end
            for ielement_vperpp in ielement_vperp_low:ielement_vperp_hi
                
                #vperp_nodes = get_nodes(vperp,ielement_vperpp)
                #vperp_max = vperp_nodes[end]
                #vperp_min = vperp_nodes[1]*nel_low(ielement_vperpp,vperp.nelement_local) 
                #nquad_vperp = get_scaled_x_w_no_divergences!(x_vperp, w_vperp, x_legendre, w_legendre, vperp_min, vperp_max)
                #nquad_vperp = get_scaled_x_w!(x_vperp, w_vperp, x_legendre, w_legendre, x_laguerre, w_laguerre, vperp_min, vperp_max, vperp_nodes, igrid_vperp, vperp_val)
                loop_over_vpa_elements!(G_weights,G1_weights,G2_weights,G3_weights,
                        H_weights,H1_weights,H2_weights,H3_weights,n_weights,
                        vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vpa grids
                        vperp,ielement_vperpp,
                        #nquad_vperp,ielement_vperpp,vperp_nodes,vperp, # info about primed vperp grids
                        x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                        igrid_vpa, igrid_vperp, vpa_val, vperp_val, ivpa, ivperp)
            end
            for ielement_vperpp in ielement_vperp_hi+1:vperp.nelement_local
                
                vperp_nodes = get_nodes(vperp,ielement_vperpp)
                vperp_max = vperp_nodes[end]
                vperp_min = vperp_nodes[1]*nel_low(ielement_vperpp,vperp.nelement_local) 
                nquad_vperp = get_scaled_x_w_no_divergences!(x_vperp, w_vperp, x_legendre, w_legendre, vperp_min, vperp_max)
                loop_over_vpa_elements_no_divergences!(G_weights,G1_weights,G2_weights,G3_weights,
                        H_weights,H1_weights,H2_weights,H3_weights,n_weights,
                        vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vpa grids
                        nquad_vperp,ielement_vperpp,vperp_nodes,vperp, # info about primed vperp grids
                        x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                        vpa_val, vperp_val, ivpa, ivperp)
            end
            return nothing
        end
        
        function loop_over_vperp_vpa_elements_no_divergences!(G_weights,G1_weights,G2_weights,G3_weights,
                        H_weights,H1_weights,H2_weights,H3_weights,n_weights,
                        vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vpa grids
                        vperp,ielement_vperp_low,ielement_vperp_hi, # info about primed vperp grids
                        x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                        igrid_vpa, igrid_vperp, vpa_val, vperp_val, ivpa, ivperp)
            for ielement_vperpp in 1:vperp.nelement_local
                vperp_nodes = get_nodes(vperp,ielement_vperpp)
                vperp_max = vperp_nodes[end]
                vperp_min = vperp_nodes[1]*nel_low(ielement_vperpp,nelement_vperp) 
                nquad_vperp = get_scaled_x_w_no_divergences!(x_vperp, w_vperp, x_legendre, w_legendre, vperp_min, vperp_max)
                loop_over_vpa_elements_no_divergences!(G_weights,G1_weights,G2_weights,G3_weights,
                        H_weights,H1_weights,H2_weights,H3_weights,n_weights,
                        vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vpa grids
                        nquad_vperp,ielement_vperpp,vperp_nodes,vperp, # info about primed vperp grids
                        x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                        vpa_val, vperp_val, ivpa, ivperp)
            end
            return nothing
        end
        
        @serial_region begin
            println("setting up GL quadrature   ", Dates.format(now(), dateformat"H:MM:SS"))
        end
        
        # get Gauss-Legendre points and weights on (-1,1)
        nquad = 2*ngrid
        x_legendre, w_legendre = gausslegendre(nquad)
        #nlaguerre = min(9,nquad) # to prevent points to close to the boundaries
        nlaguerre = nquad
        x_laguerre, w_laguerre = gausslaguerre(nlaguerre)
        
        #x_hlaguerre, w_hlaguerre = gausslaguerre(halfnquad)
        x_vpa, w_vpa = Array{mk_float,1}(undef,4*nquad), Array{mk_float,1}(undef,4*nquad)
        x_vperp, w_vperp = Array{mk_float,1}(undef,4*nquad), Array{mk_float,1}(undef,4*nquad)
        
        
        @serial_region begin
            println("beginning weights calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
        end
        
        nelement_vpa, ngrid_vpa = vpa.nelement_local, vpa.ngrid
        nelement_vperp, ngrid_vperp = vperp.nelement_local, vperp.ngrid
        # precalculated weights, integrating over Lagrange polynomials
        #begin_vperp_vpa_region()
        #@loop_vperp_vpa ivperp ivpa begin
        ivpa = ivpa_field
        ivperp = ivperp_field
            #limits where checks required to determine which divergence-safe grid is needed
            igrid_vpa, ielement_vpa = vpa.igrid[ivpa], vpa.ielement[ivpa]
            ielement_vpa_low = ielement_vpa - ng_low(igrid_vpa,ngrid_vpa)*nel_low(ielement_vpa,nelement_vpa)
            ielement_vpa_hi = ielement_vpa + ng_hi(igrid_vpa,ngrid_vpa)*nel_hi(ielement_vpa,nelement_vpa)
            #println("igrid_vpa: ielement_vpa: ielement_vpa_low: ielement_vpa_hi:", igrid_vpa," ",ielement_vpa," ",ielement_vpa_low," ",ielement_vpa_hi)
            igrid_vperp, ielement_vperp = vperp.igrid[ivperp], vperp.ielement[ivperp]
            ielement_vperp_low = ielement_vperp - ng_low(igrid_vperp,ngrid_vperp)*nel_low(ielement_vperp,nelement_vperp)
            ielement_vperp_hi = ielement_vperp + ng_hi(igrid_vperp,ngrid_vperp)*nel_hi(ielement_vperp,nelement_vperp)
            #println("igrid_vperp: ielement_vperp: ielement_vperp_low: ielement_vperp_hi:", igrid_vperp," ",ielement_vperp," ",ielement_vperp_low," ",ielement_vperp_hi)
            
            vperp_val = vperp.grid[ivperp]
            vpa_val = vpa.grid[ivpa]
            @. G_weights[1,1,:,:] = 0.0  
            @. G1_weights[1,1,:,:] = 0.0  
            @. G2_weights[1,1,:,:] = 0.0  
            @. G3_weights[1,1,:,:] = 0.0  
            @. H_weights[1,1,:,:] = 0.0  
            @. H1_weights[1,1,:,:] = 0.0  
            @. H2_weights[1,1,:,:] = 0.0  
            @. H3_weights[1,1,:,:] = 0.0  
            @. n_weights[1,1,:,:] = 0.0  
            # loop over elements and grid points within elements on primed coordinate
            loop_over_vperp_vpa_elements!(G_weights,G1_weights,G2_weights,G3_weights,
            #loop_over_vperp_vpa_elements_no_divergences!(G_weights,G1_weights,G2_weights,G3_weights,
                    H_weights,H1_weights,H2_weights,H3_weights,n_weights,
                    vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vpa grids
                    vperp,ielement_vperp_low,ielement_vperp_hi, # info about primed vperp grids
                    x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                    igrid_vpa, igrid_vperp, vpa_val, vperp_val, ivpa, ivperp)
        #end
        
        #_block_synchronize()
        begin_serial_region()
        @serial_region begin
            println("beginning integration   ", Dates.format(now(), dateformat"H:MM:SS"))
        end
        
        begin_vperp_vpa_region()
        
        # use precalculated weights to calculate Gsp using nodal values of fs
        #@loop_vperp_vpa ivperp ivpa begin
        #for ivperp in 1:nvperp
            #for ivpa in 1:nvpa 
                d2Gspdvpa2[1,1] = 0.0
                dGspdvperp[1,1] = 0.0
                d2Gspdvperpdvpa[1,1] = 0.0
                d2Gspdvperp2[1,1] = 0.0
                Gsp[1,1] = 0.0
                Hsp[1,1] = 0.0
                dHspdvpa[1,1] = 0.0
                dHspdvperp[1,1] = 0.0
                nsp[1,1] = 0.0
                for ivperpp in 1:nvperp
                    for ivpap in 1:nvpa
                        #d2Gspdvpa2[1,1] += G_weights[1,1,ivpap,ivperpp]*d2fspdvpa2[ivpap,ivperpp]
                        d2Gspdvpa2[1,1] += H3_weights[1,1,ivpap,ivperpp]*dfspdvpa[ivpap,ivperpp]
                        dGspdvperp[1,1] += G1_weights[1,1,ivpap,ivperpp]*dfspdvperp[ivpap,ivperpp]
                        d2Gspdvperpdvpa[1,1] += G1_weights[1,1,ivpap,ivperpp]*d2fspdvperpdvpa[ivpap,ivperpp]
                        #d2Gspdvperp2[1,1] += G2_weights[1,1,ivpap,ivperpp]*d2fspdvperp2[ivpap,ivperpp] + G3_weights[1,1,ivpap,ivperpp]*dfspdvperp[ivpap,ivperpp]
                        d2Gspdvperp2[1,1] += H2_weights[1,1,ivpap,ivperpp]*dfspdvperp[ivpap,ivperpp]
                        Gsp[1,1] += G_weights[1,1,ivpap,ivperpp]*fsp_in[ivpap,ivperpp]
                        Hsp[1,1] += H_weights[1,1,ivpap,ivperpp]*fsp_in[ivpap,ivperpp]
                        dHspdvpa[1,1] += H_weights[1,1,ivpap,ivperpp]*dfspdvpa[ivpap,ivperpp]
                        dHspdvperp[1,1] += H1_weights[1,1,ivpap,ivperpp]*dfspdvperp[ivpap,ivperpp]
                        nsp[1,1] += n_weights[1,1,ivpap,ivperpp]*fsp_in[ivpap,ivperpp]
                    end
                end
            #end
            
            (Hsp_from_Gsp[1,1] = 0.5*( d2Gspdvpa2[1,1] +
                                              d2Gspdvperp2[1,1] +
                            (1.0/vperp.grid[ivperp])*dGspdvperp[1,1]))
        #end
        
        plot_H = false #true
        plot_dHdvpa = false #true
        plot_dHdvperp = false #true
        plot_d2Gdvperp2 = false #true
        plot_d2Gdvperpdvpa = false #true
        plot_dGdvperp = false #true
        plot_d2Gdvpa2 = false #true
        plot_G = false #true
        plot_n = false #true
        
        begin_serial_region()
        @serial_region begin
            println("finished integration   ", Dates.format(now(), dateformat"H:MM:SS"))
            @. n_err = abs(nsp - denssp)
            max_n_err = maximum(n_err)
            #println("max_n_err: ",max_n_err)
            println("spot check n_err: ",n_err[end,end], " nsp: ",nsp[end,end])
            @. H_err = abs(Hsp_from_Gsp - H_Maxwell)
            max_H_err = maximum(H_err)
            #println("max_H_from_G_err: ",max_H_err)
            @. H_err = abs(Hsp - H_Maxwell)
            max_H_err = maximum(H_err)
            #println("max_H_err: ",max_H_err)
            println("spot check H_err: ",H_err[end,end], " H: ",Hsp[end,end])
            @. dHdvperp_err = abs(dHspdvperp - dHdvperp_Maxwell)
            max_dHdvperp_err = maximum(dHdvperp_err)
            #println("max_dHdvperp_err: ",max_dHdvperp_err)
            println("spot check dHdvperp_err: ",dHdvperp_err[end,end], " dHdvperp: ",dHspdvperp[end,end])
            @. dHdvpa_err = abs(dHspdvpa - dHdvpa_Maxwell)
            max_dHdvpa_err = maximum(dHdvpa_err)
            #println("max_dHdvpa_err: ",max_dHdvpa_err)
            println("spot check dHdvpa_err: ",dHdvpa_err[end,end], " dHdvpa: ",dHspdvpa[end,end])
            
            if plot_n
                @views heatmap(vperp.grid, vpa.grid, nsp[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                     windowsize = (360,240), margin = 15pt)
                     outfile = string("fkpl_n_lagrange.pdf")
                     savefig(outfile)
                @views heatmap(vperp.grid, vpa.grid, n_err[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                     windowsize = (360,240), margin = 15pt)
                     outfile = string("fkpl_n_err.pdf")
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
            #println("max_d2Gdvperp2_err: ",max_d2Gdvperp2_err)
            println("spot check d2Gdvperp2_err: ",d2Gdvperp2_err[end,end], " d2Gdvperp2: ",d2Gspdvperp2[end,end])
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
            #println("max_d2Gdvperpdvpa_err: ",max_d2Gdvperpdvpa_err)
            println("spot check d2Gdvperpdpva_err: ",d2Gdvperpdvpa_err[end,end], " d2Gdvperpdvpa: ",d2Gspdvperpdvpa[end,end])
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
            #println("max_dGdvperp_err: ",max_dGdvperp_err)
            println("spot check dGdvperp_err: ",dGdvperp_err[end,end], " dGdvperp: ",dGspdvperp[end,end])
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
            #println("max_d2Gdvpa2_err: ",max_d2Gdvpa2_err)
            println("spot check d2Gdvpa2_err: ",d2Gdvpa2_err[end,end], " d2Gdvpa2: ",d2Gspdvpa2[end,end])
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
            #println("max_G_err: ",max_G_err)
            println("spot check G_err: ",G_err[end,end], " G: ",Gsp[end,end])
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
        (results = (maximum(G_err), maximum(H_err),
        maximum(dHdvpa_err), maximum(dHdvperp_err), maximum(d2Gdvperp2_err), maximum(d2Gdvpa2_err), maximum(d2Gdvperpdvpa_err), maximum(dGdvperp_err),
        maximum(dfsdvpa_err), maximum(dfsdvperp_err), maximum(d2fsdvpa2_err), maximum(d2fsdvperpdvpa_err), maximum(d2fsdvperp2_err), 
        maximum(dfspdvperp_err), maximum(d2fspdvpa2_err), maximum(d2fspdvperpdvpa_err), maximum(d2fspdvperp2_err),
        maximum(n_err)))
        return results 
    end
    
    if test_Lagrange_integral
        ngrid = 9
        nelement = 4
        test_Lagrange_Rosenbluth_potentials(ngrid,nelement,standalone=true)
    end
    if test_Lagrange_integral_scan
        initialize_comms!()
        ngrid = 9
        nscan = 1
        #nelement_list = Int[2, 4, 8, 16, 32, 64, 128]
        #nelement_list = Int[2, 4, 8, 16, 32]
        #nelement_list = Int[2, 4, 8, 16]
        #nelement_list = Int[2, 4, 8]
        nelement_list = Int[8]
        max_G_err = Array{mk_float,1}(undef,nscan)
        max_H_err = Array{mk_float,1}(undef,nscan)
        max_dHdvpa_err = Array{mk_float,1}(undef,nscan)
        max_dHdvperp_err = Array{mk_float,1}(undef,nscan)
        max_d2Gdvperp2_err = Array{mk_float,1}(undef,nscan)
        max_d2Gdvpa2_err = Array{mk_float,1}(undef,nscan)
        max_d2Gdvperpdvpa_err = Array{mk_float,1}(undef,nscan)
        max_dGdvperp_err = Array{mk_float,1}(undef,nscan)
        max_dfsdvpa_err = Array{mk_float,1}(undef,nscan)
        max_dfsdvperp_err = Array{mk_float,1}(undef,nscan)
        max_d2fsdvpa2_err = Array{mk_float,1}(undef,nscan)
        max_d2fsdvperpdvpa_err = Array{mk_float,1}(undef,nscan)
        max_d2fsdvperp2_err = Array{mk_float,1}(undef,nscan)
        max_dfspdvperp_err = Array{mk_float,1}(undef,nscan)
        max_d2fspdvpa2_err = Array{mk_float,1}(undef,nscan)
        max_d2fspdvperpdvpa_err = Array{mk_float,1}(undef,nscan)
        max_d2fspdvperp2_err = Array{mk_float,1}(undef,nscan)
        max_n_err = Array{mk_float,1}(undef,nscan)
        
        expected = Array{mk_float,1}(undef,nscan)
        expected_nelement_scaling!(expected,nelement_list,ngrid,nscan)
        expected_integral = Array{mk_float,1}(undef,nscan)
        expected_nelement_integral_scaling!(expected_integral,nelement_list,ngrid,nscan)
        
        expected_label = L"(1/N_{el})^{n_g - 1}"
        expected_integral_label = L"(1/N_{el})^{n_g +1}"
        
        for iscan in 1:nscan
            local nelement = nelement_list[iscan]
            ((max_G_err[iscan], max_H_err[iscan], 
            max_dHdvpa_err[iscan],
            max_dHdvperp_err[iscan], max_d2Gdvperp2_err[iscan],
            max_d2Gdvpa2_err[iscan], max_d2Gdvperpdvpa_err[iscan],
            max_dGdvperp_err[iscan], max_dfsdvpa_err[iscan],
            max_dfsdvperp_err[iscan], max_d2fsdvpa2_err[iscan],
            max_d2fsdvperpdvpa_err[iscan], max_d2fsdvperp2_err[iscan],
            max_dfspdvperp_err[iscan], max_d2fspdvpa2_err[iscan],
            max_d2fspdvperpdvpa_err[iscan], max_d2fspdvperp2_err[iscan],
            max_n_err[iscan])
            = test_Lagrange_Rosenbluth_potentials(ngrid,nelement,standalone=false))
        end
        if global_rank[]==0
            fontsize = 8
            ytick_sequence = Array([1.0e-13,1.0e-12,1.0e-11,1.0e-10,1.0e-9,1.0e-8,1.0e-7,1.0e-6,1.0e-5,1.0e-4,1.0e-3,1.0e-2,1.0e-1,1.0e-0,1.0e1])
            xlabel = L"N_{element}"
            nlabel = L"\epsilon(n)"
            Glabel = L"\epsilon(G)"
            Hlabel = L"\epsilon(H)"
            dHdvpalabel = L"\epsilon(dH/d v_{\|\|})"
            dHdvperplabel = L"\epsilon(dH/d v_{\perp})"
            d2Gdvperp2label = L"\epsilon(d^2G/d v_{\perp}^2)"
            d2Gdvpa2label = L"\epsilon(d^2G/d v_{\|\|}^2)"
            d2Gdvperpdvpalabel = L"\epsilon(d^2G/d v_{\perp} d v_{\|\|})"
            dGdvperplabel = L"\epsilon(dG/d v_{\perp})"
            #println(max_G_err,max_H_err,max_dHdvpa_err,max_dHdvperp_err,max_d2Gdvperp2_err,max_d2Gdvpa2_err,max_d2Gdvperpdvpa_err,max_dGdvperp_err, expected, expected_integral)
            plot(nelement_list, [max_G_err,max_H_err,max_dHdvpa_err,max_dHdvperp_err,max_d2Gdvperp2_err,max_d2Gdvpa2_err,max_d2Gdvperpdvpa_err,max_dGdvperp_err, expected, expected_integral],
            xlabel=xlabel, label=[Glabel Hlabel dHdvpalabel dHdvperplabel d2Gdvperp2label d2Gdvpa2label d2Gdvperpdvpalabel dGdvperplabel expected_label expected_integral_label], ylabel="",
             shape =:circle, xscale=:log10, yscale=:log10, xticks = (nelement_list, nelement_list), yticks = (ytick_sequence, ytick_sequence), markersize = 5, linewidth=2, 
              xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, legendfontsize = fontsize,
              foreground_color_legend = nothing, background_color_legend = nothing, legend=:bottomleft)
            #outfile = "fkpl_coeffs_numerical_lagrange_integration_test_ngrid_"*string(ngrid)*".pdf"
            outfile = "fkpl_coeffs_numerical_lagrange_integration_test_ngrid_"*string(ngrid)*"_GLL.pdf"
            savefig(outfile)
            println(outfile)
            plot(nelement_list, [max_G_err,max_H_err,max_n_err,expected,expected_integral],
            xlabel=xlabel, label=[Glabel Hlabel nlabel expected_label expected_integral_label], ylabel="",
             shape =:circle, xscale=:log10, yscale=:log10, xticks = (nelement_list, nelement_list), yticks = (ytick_sequence, ytick_sequence), markersize = 5, linewidth=2, 
              xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, legendfontsize = fontsize,
              foreground_color_legend = nothing, background_color_legend = nothing, legend=:bottomleft)
            #outfile = "fkpl_coeffs_numerical_lagrange_integration_test_ngrid_"*string(ngrid)*".pdf"
            outfile = "fkpl_potentials_numerical_lagrange_integration_test_ngrid_"*string(ngrid)*"_GLL.pdf"
            savefig(outfile)
            println(outfile)
            #println(max_G_err,max_H_err,max_dHdvpa_err,max_dHdvperp_err,max_d2Gdvperp2_err,max_d2Gdvpa2_err,max_d2Gdvperpdvpa_err,max_dGdvperp_err, expected)
            plot(nelement_list, [max_dHdvpa_err,max_dHdvperp_err,max_d2Gdvperp2_err,max_d2Gdvpa2_err,max_d2Gdvperpdvpa_err,max_dGdvperp_err, expected, expected_integral],
            xlabel=xlabel, label=[dHdvpalabel dHdvperplabel d2Gdvperp2label d2Gdvpa2label d2Gdvperpdvpalabel dGdvperplabel expected_label expected_integral_label], ylabel="",
             shape =:circle, xscale=:log10, yscale=:log10, xticks = (nelement_list, nelement_list), yticks = (ytick_sequence, ytick_sequence), markersize = 5, linewidth=2, 
              xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, legendfontsize = fontsize,
              foreground_color_legend = nothing, background_color_legend = nothing, legend=:bottomleft)
            #outfile = "fkpl_coeffs_numerical_lagrange_integration_test_ngrid_"*string(ngrid)*".pdf"
            outfile = "fkpl_essential_coeffs_numerical_lagrange_integration_test_ngrid_"*string(ngrid)*"_GLL.pdf"
            savefig(outfile)
            println(outfile)
            
            dfsdvpa_label = L"\epsilon(d F_s / d v_{\|\|})"
            dfsdvperp_label = L"\epsilon(d F_s /d v_{\perp})"
            d2fsdvpa2_label = L"\epsilon(d^2 F_s /d v_{\|\|}^2)"
            d2fsdvperpdvpa_label = L"\epsilon(d^2 F_s /d v_{\perp}d v_{\|\|})"
            d2fsdvperp2_label = L"\epsilon(d^2 F_s/ d v_{\perp}^2)"
            plot(nelement_list, [max_dfsdvpa_err,max_dfsdvperp_err,max_d2fsdvpa2_err,max_d2fsdvperpdvpa_err,max_d2fsdvperp2_err,expected],
            xlabel=xlabel, label=[dfsdvpa_label dfsdvperp_label d2fsdvpa2_label d2fsdvperpdvpa_label d2fsdvperp2_label expected_label], ylabel="",
             shape =:circle, xscale=:log10, yscale=:log10, xticks = (nelement_list, nelement_list), yticks = (ytick_sequence, ytick_sequence), markersize = 5, linewidth=2, 
              xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, legendfontsize = fontsize,
              foreground_color_legend = nothing, background_color_legend = nothing, legend=:bottomleft)
            #outfile = "fkpl_coeffs_numerical_lagrange_integration_test_ngrid_"*string(ngrid)*".pdf"
            outfile = "fkpl_fs_numerical_test_ngrid_"*string(ngrid)*"_GLL.pdf"
            savefig(outfile)
            println(outfile)
            
            dfspdvperp_label = L"\epsilon(d F_{s^\prime} /d v_{\perp})"
            d2fspdvpa2_label = L"\epsilon(d^2 F_{s^\prime} /d v_{\|\|}^2)"
            d2fspdvperpdvpa_label = L"\epsilon(d^2 F_{s^\prime} /d v_{\perp}d v_{\|\|})"
            d2fspdvperp2_label = L"\epsilon(d^2 F_{s^\prime}/ d v_{\perp}^2)"
            plot(nelement_list, [max_dfspdvperp_err,max_d2fspdvpa2_err,max_d2fspdvperpdvpa_err,max_d2fspdvperp2_err,expected],
            xlabel=xlabel, label=[dfspdvperp_label d2fspdvpa2_label d2fspdvperpdvpa_label d2fspdvperp2_label expected_label], ylabel="",
             shape =:circle, xscale=:log10, yscale=:log10, xticks = (nelement_list, nelement_list), yticks = (ytick_sequence, ytick_sequence), markersize = 5, linewidth=2, 
              xtickfontsize = fontsize, xguidefontsize = fontsize, ytickfontsize = fontsize, yguidefontsize = fontsize, legendfontsize = fontsize,
              foreground_color_legend = nothing, background_color_legend = nothing, legend=:bottomleft)
            #outfile = "fkpl_coeffs_numerical_lagrange_integration_test_ngrid_"*string(ngrid)*".pdf"
            outfile = "fkpl_fsp_numerical_test_ngrid_"*string(ngrid)*"_GLL.pdf"
            savefig(outfile)
            println(outfile)
        end
        finalize_comms!()
    end
    
end 
