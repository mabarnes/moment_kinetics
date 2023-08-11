using Printf
using Plots
using LaTeXStrings
using Measures
using MPI
using SpecialFunctions: erf, ellipe, ellipk
using FastGaussQuadrature
using Dates
using LinearAlgebra: mul!
using Cubature: hcubature

import moment_kinetics
using moment_kinetics.input_structs: grid_input, advection_input
using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.chebyshev: setup_chebyshev_pseudospectral
using moment_kinetics.gauss_legendre: setup_gausslegendre_pseudospectral, gausslegendre_mass_matrix_solve!
using moment_kinetics.gauss_legendre: ielement_global_func
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
using moment_kinetics.calculus: derivative!, second_derivative!
using moment_kinetics.velocity_moments: get_density, get_upar, get_ppar, get_pperp, get_pressure
using moment_kinetics.communication
using moment_kinetics.looping
using moment_kinetics.array_allocation: allocate_shared_float

function init_grids(nelement,nelement_local,ngrid,max_cores_per_shm_block)
    discretization = "gausslegendre_pseudospectral"
    #discretization = "chebyshev_pseudospectral"
    #discretization = "finite_difference"
    
    # define inputs needed for the test
    vpa_ngrid = ngrid #number of points per element 
    vpa_nelement_local = nelement # number of elements per rank
    vpa_nelement_global = vpa_nelement_local # total number of elements 
    vpa_L = 12.0 #physical box size in reference units 
    vperp_ngrid = ngrid #number of points per element 
    vperp_nelement_local = nelement # number of elements per rank
    vperp_nelement_global = vperp_nelement_local # total number of elements 
    vperp_L = 6.0 #physical box size in reference units 
    
    # fd_option and adv_input not actually used so given values unimportant
    fd_option = "fourth_order_centered"
    cheb_option = "matrix"
    adv_input = advection_input("default", 1.0, 0.0, 0.0)
    bc = "zero" 
    nrank = 1
    irank = 0
    comm = MPI.COMM_NULL
    # create the 'input' struct containing input info needed to create a
    # coordinate
    vpa_input = grid_input("vpa", vpa_ngrid, vpa_nelement_global, vpa_nelement_local, 
        nrank, irank, vpa_L, discretization, fd_option, cheb_option, bc, adv_input, comm)
    vperp_input = grid_input("vperp", vperp_ngrid, vperp_nelement_global, vperp_nelement_local, 
        nrank, irank, vperp_L, discretization, fd_option, cheb_option, bc, adv_input, comm)
    
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
    
    # define distribution memory MPI versions of these coordinates
    vpa_MPI_ngrid = ngrid #number of points per element 
    vpa_MPI_nelement_local = nelement_local # nelement # number of elements per rank
    vpa_MPI_nelement_global = nelement # total number of elements 
    #vpa_MPI_L = 12.0 #physical box size in reference units 
    vperp_MPI_ngrid = ngrid #number of points per element 
    vperp_MPI_nelement_local = nelement_local# nelement # number of elements per rank
    vperp_MPI_nelement_global = nelement # total number of elements 
    #vperp_MPI_L = 6.0 #physical box size in reference units 
    
    # using standard MPI routine with assumption of exactly the right number of cores
    #irank_vpa, nrank_vpa, comm_sub_vpa, irank_vperp, nrank_vperp, comm_sub_vperp = setup_distributed_memory_MPI(vpa_MPI_nelement_global,vpa_MPI_nelement_local,
    #                                                                                                               vperp_MPI_nelement_global,vperp_MPI_nelement_local)
    # novel MPI routine that tries to use only a subset of the cores available
    # need to specify max number of cores in a shared-memory region (depends on hardware)
    
    ( (irank_vpa, nrank_vpa, comm_sub_vpa, irank_vperp, nrank_vperp, comm_sub_vperp) =
     setup_distributed_memory_MPI_for_weights_precomputation(vpa_MPI_nelement_global,vpa_MPI_nelement_local,
                      vperp_MPI_nelement_global,vperp_MPI_nelement_local,max_cores_per_shm_block,printout=false))
    vpa_MPI_input = grid_input("vpa", vpa_ngrid, vpa_MPI_nelement_global, vpa_MPI_nelement_local, 
        nrank_vpa, irank_vpa, vpa_L, discretization, fd_option, cheb_option, bc, adv_input, comm_sub_vpa)
    vperp_MPI_input = grid_input("vperp", vperp_MPI_ngrid, vperp_MPI_nelement_global, vperp_MPI_nelement_local, 
        nrank_vperp, irank_vperp, vperp_L, discretization, fd_option, cheb_option, bc, adv_input, comm_sub_vperp)
    vpa_MPI = define_coordinate(vpa_MPI_input)
    #println("vpa ",vpa_MPI.grid)
    vperp_MPI = define_coordinate(vperp_MPI_input)
    #println("vperp ",vperp_MPI.grid)
    
    return vpa, vperp, vpa_spectral, vperp_spectral, vpa_MPI, vperp_MPI
end

function get_vth(pres,dens,mass)
        return sqrt(pres/(dens*mass))
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

function eta_speed(upar,vth,vpa,vperp,ivpa,ivperp)
    eta = sqrt((vpa.grid[ivpa]-upar)^2 + vperp.grid[ivperp]^2)/vth
    return eta
end

"""
functions for doing the Lagrange polynomial integration
"""

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

if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")

    # Set up MPI
    initialize_comms!()
    
    nelement_global = 2 # global number of elements in vpa (vperp)
    nelement_local = 2 # local number of elements in each distributed memory `chunk' of the vpa (vperp) grid
    ngrid = 7 # number of points per element
    max_cores_per_shm_block = 20 # maximum number of cores in a shared-memory block 
    
    vpa, vperp, vpa_spectral, vperp_spectral, vpa_MPI, vperp_MPI = init_grids(nelement_global,nelement_local,ngrid,max_cores_per_shm_block)
    
    nvpa_local = vpa_MPI.n
    nvpa_global = vpa_MPI.n_global     
    nvperp_local = vperp_MPI.n
    nvperp_global = vperp_MPI.n_global     
    
    # coord.n is the local number of points in each shared memory block
    looping.setup_loop_ranges!(block_rank[], block_size[];
                                       s=1, sn=1,
                                       r=1, z=1, vperp=nvperp_local, vpa=nvpa_local,
                                       vzeta=1, vr=1, vz=1)
    if global_rank[] == 0
        @serial_region begin
            println("beginning allocation   ", Dates.format(now(), dateformat"H:MM:SS"))
        end
    end
    fsp_in = Array{mk_float,2}(undef,nvpa_global,nvperp_global)
    G_Maxwell = Array{mk_float,2}(undef,nvpa_local,nvperp_local)
    G_weights = allocate_shared_float(nvpa_local,nvperp_local,nvpa_global,nvperp_global)
    Gsp = allocate_shared_float(nvpa_local,nvperp_local)
    G_err = allocate_shared_float(nvpa_local,nvperp_local)
    
    Gsp_global = Array{mk_float,2}(undef,nvpa_global,nvperp_global)
    G_Maxwell_global = Array{mk_float,2}(undef,nvpa_global,nvperp_global)
    G_err_global = Array{mk_float,2}(undef,nvpa_global,nvperp_global)
    G_weights_global = Array{mk_float,4}(undef,nvpa_global,nvperp_global,nvpa_global,nvperp_global)
    
    if global_rank[] == 0
        @serial_region begin
            println("setting up input arrays   ", Dates.format(now(), dateformat"H:MM:SS"))
        end
    end
    # set up test Maxwellian
    # species sp 
    denssp = 1.0 #3.0/4.0
    uparsp = 0.5 #2.0/3.0
    pparsp = 1.0 #2.0/3.0
    pperpsp = 1.0 #2.0/3.0
    pressp = get_pressure(pparsp,pperpsp) 
    msp = 1.0
    vthsp = get_vth(pressp,denssp,msp)
        
    for ivperp in 1:nvperp_global
        for ivpa in 1:nvpa_global
            fsp_in[ivpa,ivperp] = F_Maxwellian(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
            G_Maxwell_global[ivpa,ivperp] = G_Maxwellian(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
        end
    end
    for ivperp in 1:nvperp_local
        for ivpa in 1:nvpa_local
            G_Maxwell[ivpa,ivperp] = G_Maxwellian(denssp,uparsp,vthsp,vpa_MPI,vperp_MPI,ivpa,ivperp)
        end
    end
    
    if global_rank[] == 0
        @serial_region begin
            println("setting up GL quadrature   ", Dates.format(now(), dateformat"H:MM:SS"))
        end
    end
    # get Gauss-Legendre points and weights on (-1,1)
    nquad = 2*ngrid
    x, w = gausslegendre(nquad)
    x_vpa, w_vpa = Array{mk_float,1}(undef,nquad), Array{mk_float,1}(undef,nquad)
    x_vperp, w_vperp = Array{mk_float,1}(undef,nquad), Array{mk_float,1}(undef,nquad)
    
    if global_rank[] == 0
        @serial_region begin
            println("beginning weights calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
        end
    end

    # function returns 1 for nelement >= ielement > 1, 0 for ielement =1 
    function nel_low(ielement,nelement)
        return floor(mk_int, (ielement - 2 + nelement)/nelement)
    end

    function G_weights_integrand(v::Vector{mk_float},vpa_val,vperp_val,
                                   igrid_vpap,ielement_vpap,
                                   igrid_vperpp,ielement_vperpp,
                                   vpap_struct,vperpp_struct)
    #function G_weights_integrand(v::Vector{mk_float},vpa_val=0.0,vperp_val=0.0,
    #                               igrid_vpap=1,ielement_vpap=1,
    #                               igrid_vperpp=1,ielement_vperpp=1,
    #                               vpap_struct=vpa,vperpp_struct=vperp)
        
        vpa_nodes = get_nodes(vpap_struct,ielement_vpap)
        ivpap = vpap_struct.igrid_full[igrid_vpap,ielement_vpap]
        
        vperp_nodes = get_nodes(vperpp_struct,ielement_vperpp)
        ivperpp = vperpp_struct.igrid_full[igrid_vperpp,ielement_vperpp]

        vpap_val = v[1]
        vperpp_val = v[2]
        denom = (vpa_val - vpap_val)^2 + (vperp_val + vperpp_val)^2 
        mm = 4.0*vperp_val*vperpp_val/denom
        prefac = sqrt(denom)
        ellipe_mm = ellipe(mm) 
        ellipk_mm = ellipk(mm) 
        G_elliptic_integral_factor = 2.0*ellipe_mm*prefac/pi
        lagrange_poly_vpa = lagrange_poly(igrid_vpap,vpa_nodes,vpap_val)
        lagrange_poly_vperp = lagrange_poly(igrid_vperpp,vperp_nodes,vperpp_val)
        
        (G_weight = lagrange_poly_vpa*lagrange_poly_vperp*
        G_elliptic_integral_factor*vperpp_val*2.0/sqrt(pi))
        return G_weight
    end
    
    # precalculated weights, integrating over Lagrange polynomials
    begin_vperp_vpa_region()
    @loop_vperp_vpa ivperp ivpa begin
            vperp_val = vperp_MPI.grid[ivperp]
            vpa_val = vpa_MPI.grid[ivpa]
            @. G_weights[ivpa,ivperp,:,:] = 0.0  
            
            # loop over elements and grid points within elements on primed coordinate
            for ielement_vperp in 1:vperp.nelement_local
                
                vperp_nodes = get_nodes(vperp,ielement_vperp)
                vperp_max = vperp_nodes[end]
                vperp_min = vperp_nodes[1]*nel_low(ielement_vperp,vperp.nelement_local)
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
                            #for kvperp in 1:nquad 
                            #    for kvpa in 1:nquad 
                            #        x_kvpa = x_vpa[kvpa]
                            #        x_kvperp = x_vperp[kvperp]
                            #        w_kvperp = w_vperp[kvperp]
                            #        w_kvpa = w_vpa[kvpa]
                            #        denom = (vpa_val - x_kvpa)^2 + (vperp_val + x_kvperp)^2 
                            #        mm = 4.0*vperp_val*x_kvperp/denom
                            #        prefac = sqrt(denom)
                            #        ellipe_mm = ellipe(mm) 
                            #        ellipk_mm = ellipk(mm) 
                            #        G_elliptic_integral_factor = 2.0*ellipe_mm*prefac/pi
                            #        lagrange_poly_vpa = lagrange_poly(igrid_vpa,vpa_nodes,x_kvpa)
                            #        lagrange_poly_vperp = lagrange_poly(igrid_vperp,vperp_nodes,x_kvperp)
                                    
                            #        (G_weights[ivpa,ivperp,ivpap,ivperpp] += 
                            #            lagrange_poly_vpa*lagrange_poly_vperp*
                            #            G_elliptic_integral_factor*x_kvperp*w_kvperp*w_kvpa*2.0/sqrt(pi))
                            G_weights_int(v) = G_weights_integrand(v,vpa_val,vperp_val,
                                                                       igrid_vpa,ielement_vpa,
                                                                       igrid_vperp,ielement_vperp,
                                                                       vpa,vperp)
                            #G_weights_int(v) = G_weights_integrand(v,vpa_val=vpa_val,vperp_val=vperp_val,
                            #                                           igrid_vpap=igrid_vpa,ielement_vpap=ielement_vpa,
                            #                                           igrid_vperpp=igrid_vperp,ielement_vperpp=ielement_vperp,
                            #                                           vpap_struct=vpa,vperpp_struct=vperp)
                            #(val,err) = hcubature(G_weights_int, [vpa_min,vperp_min], [vpa_max, vperp_max]; reltol=1e-8, abstol=0, maxevals=0)    
                            (val,err) = hcubature(G_weights_int, [vpa_min,vperp_min], [vpa_max, vperp_max]; abstol=1e-8)    
                            #println("integrated successfully")
                            G_weights[ivpa,ivperp,ivpap,ivperpp] += val
                            #    end
                            #end
                        end
                    end
                end
            end
    end
    
    #_block_synchronize()
    begin_serial_region()
    if global_rank[] == 0
        @serial_region begin
            println("beginning integration   ", Dates.format(now(), dateformat"H:MM:SS"))
        end
    end
    begin_vperp_vpa_region()
    
    # use precalculated weights to calculate Gsp using nodal values of fs
    @loop_vperp_vpa ivperp ivpa begin
            Gsp[ivpa,ivperp] = 0.0
            for ivperpp in 1:nvperp_global
                for ivpap in 1:nvpa_global
                    Gsp[ivpa,ivperp] += G_weights[ivpa,ivperp,ivpap,ivperpp]*fsp_in[ivpap,ivperpp]
                end
            end
    end
    
    begin_serial_region()
    if global_rank[] == 0
        @serial_region begin
            println("finished integration   ", Dates.format(now(), dateformat"H:MM:SS"))
        end
    end
    
    # local plotting
    plot_local_G = false #true
    print_local_G = true
    
    begin_serial_region()
    @serial_region begin
        if print_local_G
            @. G_err = abs(Gsp - G_Maxwell)
            max_G_err = maximum(G_err)
            println("max_G_err: ",max_G_err)
        end
        if plot_local_G
            @views heatmap(vperp_MPI.grid, vpa_MPI.grid, Gsp[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                 windowsize = (360,240), margin = 15pt)
                 outfile = string("fkpl_G_lagrange"*string(vpa_MPI.irank)*"."*string(vperp_MPI.irank)*".pdf")
                 savefig(outfile)
            @views heatmap(vperp_MPI.grid, vpa_MPI.grid, G_Maxwell[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                 windowsize = (360,240), margin = 15pt)
                 outfile = string("fkpl_G_Maxwell"*string(vpa_MPI.irank)*"."*string(vperp_MPI.irank)*".pdf")
                 savefig(outfile)
             @views heatmap(vperp_MPI.grid, vpa_MPI.grid, G_err[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                 windowsize = (360,240), margin = 15pt)
                 outfile = string("fkpl_G_err"*string(vpa_MPI.irank)*"."*string(vperp_MPI.irank)*".pdf")
                 savefig(outfile)
        end
    end

    mpi_test = false #true
    if mpi_test
        # allreduce operations
        # get local data into global array before allreduce
        begin_serial_region()
        @serial_region begin
             @. Gsp_global = 0.0
             @. G_weights_global = 0.0
             # set up indexes for doing the assignment to the global array
             iel_vpa_min = ielement_global_func(vpa_MPI.nelement_local,vpa_MPI.irank,1)
             iel_vpa_max = ielement_global_func(vpa_MPI.nelement_local,vpa_MPI.irank,vpa_MPI.nelement_local)
             ivpa_global_min = vpa.imin[iel_vpa_min]
             ivpa_global_max = vpa.imax[iel_vpa_max]
             if vpa_MPI.irank > 0
                j = 1 #exclude lowest point
             else 
                j = 0 #include lowest point
             end
             ivpa_local_min = 1 + j
             ivpa_local_max = vpa_MPI.n
             
             iel_vperp_min = ielement_global_func(vperp_MPI.nelement_local,vperp_MPI.irank,1)
             iel_vperp_max = ielement_global_func(vperp_MPI.nelement_local,vperp_MPI.irank,vperp_MPI.nelement_local)
             ivperp_global_min = vperp.imin[iel_vperp_min]
             ivperp_global_max = vperp.imax[iel_vperp_max]
             if vperp_MPI.irank > 0
                k = 1 #exclude lowest point
             else 
                k = 0 #include lowest point
             end
             ivperp_local_min = 1 + k
             ivperp_local_max = vperp_MPI.n
             
             @. Gsp_global[ivpa_global_min:ivpa_global_max,ivperp_global_min:ivperp_global_max] += Gsp[ivpa_local_min:ivpa_local_max,ivperp_local_min:ivperp_local_max]
             @. G_weights_global[ivpa_global_min:ivpa_global_max,ivperp_global_min:ivperp_global_max,:,:] += G_weights[ivpa_local_min:ivpa_local_max,ivperp_local_min:ivperp_local_max,:,:]
             
             # first reduce along vperp dimension, then along vpa
             MPI.Allreduce!(Gsp_global,+,vperp_MPI.comm)
             MPI.Allreduce!(Gsp_global,+,vpa_MPI.comm)
             MPI.Allreduce!(G_weights_global,+,vperp_MPI.comm)
             MPI.Allreduce!(G_weights_global,+,vpa_MPI.comm)
             # then broadcast to all cores, noting that some may
             # not be included in the communicators used in the calculation
             MPI.Bcast!(Gsp_global,comm_world,root=0)
             MPI.Bcast!(G_weights_global,comm_world,root=0)
        end 
        
        # global plotting
        plot_G = false #true
        
        begin_serial_region()
        if global_rank[] == 0
            @serial_region begin
                @. G_err_global = abs(Gsp_global - G_Maxwell_global)
                max_G_err = maximum(G_err_global)
                println("max_G_err (global): ",max_G_err)
                if plot_G
                    @views heatmap(vperp.grid, vpa.grid, Gsp_global[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                         windowsize = (360,240), margin = 15pt)
                         outfile = string("fkpl_G_lagrange.pdf")
                         savefig(outfile)
                    @views heatmap(vperp.grid, vpa.grid, G_Maxwell_global[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                         windowsize = (360,240), margin = 15pt)
                         outfile = string("fkpl_G_Maxwell.pdf")
                         savefig(outfile)
                    @views heatmap(vperp.grid, vpa.grid, G_err_global[:,:], xlabel=L"v_{\perp}", ylabel=L"v_{||}", c = :deep, interpolation = :cubic,
                         windowsize = (360,240), margin = 15pt)
                         outfile = string("fkpl_G_err.pdf")
                         savefig(outfile)
                end
            end 
        end
        
        # check that the weights have been calculated properly by recalculating G
        # using the looping and MPI arrangement of the main code 
        z_irank, z_nrank_per_group, z_comm, r_irank, r_nrank_per_group, r_comm = setup_distributed_memory_MPI(1,1,1,1)
        # coord.n is the local number of points in each shared memory block
        looping.setup_loop_ranges!(block_rank[], block_size[];
                                           s=1, sn=1,
                                           r=1, z=1, vperp=nvperp_global, vpa=nvpa_global,
                                           vzeta=1, vr=1, vz=1)
        Gsp_global_shared = allocate_shared_float(nvpa_global,nvperp_global)
        # use precalculated weights to calculate Gsp using nodal values of fs
        begin_vperp_vpa_region()
        @loop_vperp_vpa ivperp ivpa begin
            Gsp_global_shared[ivpa,ivperp] = 0.0
            for ivperpp in 1:nvperp_global
                for ivpap in 1:nvpa_global
                    Gsp_global_shared[ivpa,ivperp] += G_weights_global[ivpa,ivperp,ivpap,ivperpp]*fsp_in[ivpap,ivperpp]
                end
            end
        end
        begin_serial_region()
        @serial_region begin
            @. G_err_global = abs(Gsp_global_shared - G_Maxwell_global)
            max_G_err = maximum(G_err_global)
            println("max_G_err (global from redistributed weights): ",max_G_err)
        end
    end
    # clean up MPI objects
    finalize_comms!()

        
end