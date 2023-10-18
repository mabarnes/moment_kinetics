"""
module for including the Full-F Fokker-Planck Collision Operator
"""
module fokker_planck


export init_fokker_planck_collisions, fokkerplanck_arrays_struct
export explicit_fokker_planck_collisions!
export calculate_Rosenbluth_potentials!
export calculate_collisional_fluxes, calculate_Maxwellian_Rosenbluth_coefficients
export Cflux_vpa_Maxwellian_inputs, Cflux_vperp_Maxwellian_inputs
export calculate_Rosenbluth_H_from_G!

export d2Gdvpa2, dGdvperp, d2Gdvperpdvpa, d2Gdvperp2
export dHdvpa, dHdvperp, Cssp_Maxwellian_inputs
export F_Maxwellian, dFdvpa_Maxwellian, dFdvperp_Maxwellian
export d2Fdvpa2_Maxwellian, d2Fdvperpdvpa_Maxwellian, d2Fdvperp2_Maxwellian
export H_Maxwellian, G_Maxwellian

export Cssp_fully_expanded_form, get_local_Cssp_coefficients!, init_fokker_planck_collisions
# testing
export symmetric_matrix_inverse

using SpecialFunctions: ellipk, ellipe, erf
using FastGaussQuadrature
using Dates
using ..initial_conditions: enforce_boundary_conditions!
using ..type_definitions: mk_float, mk_int
using ..array_allocation: allocate_float, allocate_shared_float
using ..communication: MPISharedArray
using ..velocity_moments: integrate_over_vspace
using ..velocity_moments: get_density, get_upar, get_ppar, get_pperp, get_qpar, get_pressure, get_rmom
using ..calculus: derivative!, second_derivative!
using ..looping
"""
a struct of dummy arrays and precalculated coefficients
for the Fokker-Planck collision operator 
"""

struct fokkerplanck_arrays_struct
    G1_weights::MPISharedArray{mk_float,4}
    H0_weights::MPISharedArray{mk_float,4}
    H1_weights::MPISharedArray{mk_float,4}
    H2_weights::MPISharedArray{mk_float,4}
    H3_weights::MPISharedArray{mk_float,4}
    #Rosenbluth_G::Array{mk_float,2}
    d2Gdvpa2::MPISharedArray{mk_float,2}
    d2Gdvperpdvpa::MPISharedArray{mk_float,2}
    d2Gdvperp2::MPISharedArray{mk_float,2}
    dGdvperp::MPISharedArray{mk_float,2}
    #Rosenbluth_H::Array{mk_float,2}
    dHdvpa::MPISharedArray{mk_float,2}
    dHdvperp::MPISharedArray{mk_float,2}
    #Cflux_vpa::MPISharedArray{mk_float,2}
    #Cflux_vperp::MPISharedArray{mk_float,2}
    buffer_vpavperp_1::Array{mk_float,2}
    buffer_vpavperp_2::Array{mk_float,2}
    Cssp_result_vpavperp::MPISharedArray{mk_float,2}
    dfdvpa::MPISharedArray{mk_float,2}
    d2fdvpa2::MPISharedArray{mk_float,2}
    d2fdvperpdvpa::MPISharedArray{mk_float,2}
    dfdvperp::MPISharedArray{mk_float,2}
    d2fdvperp2::MPISharedArray{mk_float,2}
end

"""
allocate the required ancilliary arrays 
"""

function allocate_fokkerplanck_arrays(vperp,vpa)
    nvpa = vpa.n
    nvperp = vperp.n
    
    G1_weights = allocate_shared_float(nvpa,nvperp,nvpa,nvperp)
    H0_weights = allocate_shared_float(nvpa,nvperp,nvpa,nvperp)
    H1_weights = allocate_shared_float(nvpa,nvperp,nvpa,nvperp)
    H2_weights = allocate_shared_float(nvpa,nvperp,nvpa,nvperp)
    H3_weights = allocate_shared_float(nvpa,nvperp,nvpa,nvperp)
    #Rosenbluth_G = allocate_float(nvpa,nvperp)
    d2Gdvpa2 = allocate_shared_float(nvpa,nvperp)
    d2Gdvperpdvpa = allocate_shared_float(nvpa,nvperp)
    d2Gdvperp2 = allocate_shared_float(nvpa,nvperp)
    dGdvperp = allocate_shared_float(nvpa,nvperp)
    #Rosenbluth_H = allocate_float(nvpa,nvperp)
    dHdvpa = allocate_shared_float(nvpa,nvperp)
    dHdvperp = allocate_shared_float(nvpa,nvperp)
    #Cflux_vpa = allocate_shared_float(nvpa,nvperp)
    #Cflux_vperp = allocate_shared_float(nvpa,nvperp)
    buffer_vpavperp_1 = allocate_float(nvpa,nvperp)
    buffer_vpavperp_2 = allocate_float(nvpa,nvperp)
    Cssp_result_vpavperp = allocate_shared_float(nvpa,nvperp)
    dfdvpa = allocate_shared_float(nvpa,nvperp)
    d2fdvpa2 = allocate_shared_float(nvpa,nvperp)
    d2fdvperpdvpa = allocate_shared_float(nvpa,nvperp)
    dfdvperp = allocate_shared_float(nvpa,nvperp)
    d2fdvperp2 = allocate_shared_float(nvpa,nvperp)
    
    return fokkerplanck_arrays_struct(G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                               d2Gdvpa2,d2Gdvperpdvpa,d2Gdvperp2,dGdvperp,
                               dHdvpa,dHdvperp,buffer_vpavperp_1,buffer_vpavperp_2,
                               Cssp_result_vpavperp, dfdvpa, d2fdvpa2,
                               d2fdvperpdvpa, dfdvperp, d2fdvperp2)
end


"""
a struct to contain the integration weights for the boundary points
in the (vpa,vperp) domain
"""
struct boundary_integration_weights_struct
    lower_vpa_boundary::MPISharedArray{mk_float,3}
    upper_vpa_boundary::MPISharedArray{mk_float,3}
    upper_vperp_boundary::MPISharedArray{mk_float,3}
end

"""
a struct used for calculating the integration weights for the 
boundary of the velocity space domain in (vpa,vperp) coordinates
"""
struct fokkerplanck_boundary_data_arrays_struct
    G0_weights::boundary_integration_weights_struct
    G1_weights::boundary_integration_weights_struct
    H0_weights::boundary_integration_weights_struct
    H1_weights::boundary_integration_weights_struct
    H2_weights::boundary_integration_weights_struct
    H3_weights::boundary_integration_weights_struct
end


function allocate_boundary_integration_weight(vpa,vperp)
    nvpa = vpa.n
    nvperp = vperp.n
    lower_vpa_boundary = allocate_shared_float(nvpa,nvperp,nvperp)
    upper_vpa_boundary = allocate_shared_float(nvpa,nvperp,nvperp)
    upper_vperp_boundary = allocate_shared_float(nvpa,nvperp,nvpa)
    return boundary_integration_weights_struct()
end

function allocate_boundary_integration_weights(vpa,vperp)
    G0_weights = allocate_boundary_integration_weight(vpa,vperp)
    G1_weights = allocate_boundary_integration_weight(vpa,vperp)
    H0_weights = allocate_boundary_integration_weight(vpa,vperp)
    H1_weights = allocate_boundary_integration_weight(vpa,vperp)
    H2_weights = allocate_boundary_integration_weight(vpa,vperp)
    H3_weights = allocate_boundary_integration_weight(vpa,vperp)
    return fokkerplanck_boundary_data_arrays_struct(G0_weights,
            G1_weights,H0_weights,H1_weights,H2_weights,H3_weights)
end

# initialise the elliptic integral factor arrays 
# note the definitions of ellipe & ellipk
# `https://specialfunctions.juliamath.org/stable/functions_list/#SpecialFunctions.ellipe`
# `https://specialfunctions.juliamath.org/stable/functions_list/#SpecialFunctions.ellipk`
# `ellipe(m) = \int^{\pi/2}\_0 \sqrt{ 1 - m \sin^2(\theta)} d \theta`
# `ellipe(k) = \int^{\pi/2}\_0 \frac{1}{\sqrt{ 1 - m \sin^2(\theta)}} d \theta`

function init_elliptic_integral_factors!(elliptic_integral_E_factor, elliptic_integral_K_factor, vperp, vpa)
    
    # must loop over vpa, vperp, vpa', vperp'
    # hence mix of looping macros for unprimed variables 
    # & standard local `for' loop for primed variables
    nvperp = vperp.n
    nvpa = vpa.n
    zero = 1.0e-10
    for ivperpp in 1:nvperp
        for ivpap in 1:nvpa
            for ivperp in 1:nvperp
                for ivpa in 1:nvpa                        
                    # the argument of the elliptic integrals 
                    # mm = 4 vperp vperp' / ( (vpa- vpa')^2 + (vperp + vperp'))
                    denom = (vpa.grid[ivpa] - vpa.grid[ivpap])^2 + (vperp.grid[ivperp] + vperp.grid[ivperpp])^2 
                    if denom < zero 
                        println("denom = zero ",ivperpp," ",ivpap," ",ivperp," ",ivpa)
                    end
                    #    #then vpa = vpa' = vperp' = vperp = 0 
                    #    mm = 0.0
                    #    prefac = 0.0 # because vperp' wgt = 0 here 
                    #else    
                        mm = 4.0*vperp.grid[ivperp]*vperp.grid[ivperpp]/denom
                        prefac = sqrt(denom)
                    #end
                    #println(mm," ",prefac," ",denom," ",ivperpp," ",ivpap," ",ivperp," ",ivpa)
                    elliptic_integral_E_factor[ivpap,ivperpp,ivpa,ivperp] = 2.0*ellipe(mm)*prefac/pi
                    elliptic_integral_K_factor[ivpap,ivperpp,ivpa,ivperp] = 2.0*ellipk(mm)/(pi*prefac)
                    #println(elliptic_integral_K_factor[ivpap,ivperpp,ivpa,ivperp]," ",mm," ",prefac," ",denom," ",ivperpp," ",ivpap," ",ivperp," ",ivpa)
                    
                end
            end
        end
    end

end

"""
function that initialises the arrays needed for Fokker Planck collisions
using numerical integration to compute the Rosenbluth potentials only
at the boundary and using an elliptic solve to obtain the potentials 
in the rest of the velocity space domain.
"""

function init_fokker_planck_collisions_new(vpa,vperp; precompute_weights=false)
    bwgt = allocate_boundary_integration_weights(vpa,vperp)
    if vperp.n > 1 && precompute_weights
        @views init_Rosenbluth_potential_boundary_integration_weights!(bwgt.G0_weights, bwgt.G1_weights, bwgt.H0_weights, bwgt.H1_weights,
                                        bwgt.H2_weights, bwgt.H3_weights, vpa, vperp)
    end
    return fka
end

"""
function that initialises the arrays needed for Fokker Planck collisions
"""

function init_fokker_planck_collisions(vperp,vpa; precompute_weights=false)
    fka = allocate_fokkerplanck_arrays(vperp,vpa)
    if vperp.n > 1 && precompute_weights
        @views init_Rosenbluth_potential_integration_weights!(fka.G1_weights, fka.H0_weights, fka.H1_weights,
                                        fka.H2_weights, fka.H3_weights, vperp, vpa)
    end
    return fka
end

"""
function that precomputes the required integration weights
"""
function init_Rosenbluth_potential_integration_weights!(G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,vperp,vpa)
    @serial_region begin
        println("setting up GL quadrature   ", Dates.format(now(), dateformat"H:MM:SS"))
    end
    
    # get Gauss-Legendre points and weights on (-1,1)
    ngrid = max(vpa.ngrid,vperp.ngrid)
    nquad = 2*ngrid
    x_legendre, w_legendre = gausslegendre(nquad)
    #nlaguerre = min(9,nquad) # to prevent points to close to the boundaries
    nlaguerre = nquad
    x_laguerre, w_laguerre = gausslaguerre(nlaguerre)
    
    x_vpa, w_vpa = Array{mk_float,1}(undef,4*nquad), Array{mk_float,1}(undef,4*nquad)
    x_vperp, w_vperp = Array{mk_float,1}(undef,4*nquad), Array{mk_float,1}(undef,4*nquad)
    
    @serial_region begin
        println("beginning weights calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
    end

    # precalculated weights, integrating over Lagrange polynomials
    begin_vperp_vpa_region()
    @loop_vperp_vpa ivperp ivpa begin
        #limits where checks required to determine which divergence-safe grid is needed
        igrid_vpa, ielement_vpa, ielement_vpa_low, ielement_vpa_hi, igrid_vperp, ielement_vperp, ielement_vperp_low, ielement_vperp_hi = get_element_limit_indices(ivpa,ivperp,vpa,vperp)
        
        vperp_val = vperp.grid[ivperp]
        vpa_val = vpa.grid[ivpa]
        for ivperpp in 1:vperp.n
            for ivpap in 1:vpa.n
                # G_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0  
                G1_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0  
                # G2_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0  
                # G3_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0  
                H0_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0  
                H1_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0  
                H2_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0  
                H3_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0  
                #@. n_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0  
            end
        end
        # loop over elements and grid points within elements on primed coordinate
        @views loop_over_vperp_vpa_elements!(G1_weights[:,:,ivpa,ivperp],
                H0_weights[:,:,ivpa,ivperp],H1_weights[:,:,ivpa,ivperp],
                H2_weights[:,:,ivpa,ivperp],H3_weights[:,:,ivpa,ivperp],
                vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vpa grids
                vperp,ielement_vperp_low,ielement_vperp_hi, # info about primed vperp grids
                x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                x_legendre,w_legendre,x_laguerre,w_laguerre,
                igrid_vpa, igrid_vperp, vpa_val, vperp_val)
    end
    
    
    @serial_region begin
        println("finished weights calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
    end
    return nothing
end

"""
function for getting the indices used to choose the integration
quadrature 
"""
function get_element_limit_indices(ivpa,ivperp,vpa,vperp)
    nelement_vpa, ngrid_vpa = vpa.nelement_local, vpa.ngrid
    nelement_vperp, ngrid_vperp = vperp.nelement_local, vperp.ngrid
    #limits where checks required to determine which divergence-safe grid is needed
    igrid_vpa, ielement_vpa = vpa.igrid[ivpa], vpa.ielement[ivpa]
    ielement_vpa_low = ielement_vpa - ng_low(igrid_vpa,ngrid_vpa)*nel_low(ielement_vpa,nelement_vpa)
    ielement_vpa_hi = ielement_vpa + ng_hi(igrid_vpa,ngrid_vpa)*nel_hi(ielement_vpa,nelement_vpa)
    #println("igrid_vpa: ielement_vpa: ielement_vpa_low: ielement_vpa_hi:", igrid_vpa," ",ielement_vpa," ",ielement_vpa_low," ",ielement_vpa_hi)
    igrid_vperp, ielement_vperp = vperp.igrid[ivperp], vperp.ielement[ivperp]
    ielement_vperp_low = ielement_vperp - ng_low(igrid_vperp,ngrid_vperp)*nel_low(ielement_vperp,nelement_vperp)
    ielement_vperp_hi = ielement_vperp + ng_hi(igrid_vperp,ngrid_vperp)*nel_hi(ielement_vperp,nelement_vperp)
    #println("igrid_vperp: ielement_vperp: ielement_vperp_low: ielement_vperp_hi:", igrid_vperp," ",ielement_vperp," ",ielement_vperp_low," ",ielement_vperp_hi)
    return igrid_vpa, ielement_vpa, ielement_vpa_low, ielement_vpa_hi, 
            igrid_vperp, ielement_vperp, ielement_vperp_low, ielement_vperp_hi
end
"""
function that precomputes the required integration weights
only along the velocity space boundaries
"""
function init_Rosenbluth_potential_boundary_integration_weights!(G0_weights,
      G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,vpa,vperp)
    @serial_region begin
        println("setting up GL quadrature   ", Dates.format(now(), dateformat"H:MM:SS"))
    end
    
    nelement_vpa, ngrid_vpa = vpa.nelement_local, vpa.ngrid
    nelement_vperp, ngrid_vperp = vperp.nelement_local, vperp.ngrid
    ngrid = max(ngrid_vpa,ngrid_vperp)
    
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

    # precalculate weights, integrating over Lagrange polynomials
    # first compute weights along vpa boundaries
    begin_vperp_region()
    ivpa = 1 #
    @loop_vperp ivperp begin
        #limits where checks required to determine which divergence-safe grid is needed
        igrid_vpa, ielement_vpa, ielement_vpa_low, ielement_vpa_hi, igrid_vperp, ielement_vperp, ielement_vperp_low, ielement_vperp_hi = get_element_limit_indices(ivpa,ivperp,vpa,vperp)
        
        vperp_val = vperp.grid[ivperp]
        vpa_val = vpa.grid[ivpa]
        for ivperpp in 1:vperp.n
            for ivpap in 1:vpa.n
                G0_weights.lower_vpa_boundary[ivpap,ivperpp,ivperp] = 0.0  
                G1_weights.lower_vpa_boundary[ivpap,ivperpp,ivperp] = 0.0  
                # G2_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0  
                # G3_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0  
                H0_weights.lower_vpa_boundary[ivpap,ivperpp,ivperp] = 0.0  
                H1_weights.lower_vpa_boundary[ivpap,ivperpp,ivperp] = 0.0  
                H2_weights.lower_vpa_boundary[ivpap,ivperpp,ivperp] = 0.0  
                H3_weights.lower_vpa_boundary[ivpap,ivperpp,ivperp] = 0.0  
                #@. n_weights[ivpap,ivperpp,ivpa,ivperp] = 0.0  
            end
        end
        # loop over elements and grid points within elements on primed coordinate
        @views loop_over_vperp_vpa_elements!(G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vpa grids
                vperp,ielement_vperp_low,ielement_vperp_hi, # info about primed vperp grids
                x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                x_legendre,w_legendre,x_laguerre,w_laguerre,
                igrid_vpa, igrid_vperp, vpa_val, vperp_val, ivpa, ivperp)
    end
    
    # now compute the weights along the vperp boundary
    @serial_region begin
        println("finished weights calculation   ", Dates.format(now(), dateformat"H:MM:SS"))
    end
    return nothing
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
        node_cut = (nodes[nnodes-1] + nodes[nnodes])/2.0
        
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
        node_cut = (nodes[1] + nodes[2])/2.0
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
        #println(nodes[igrid_coord]," ", coord_val)
        n = 2*nquad_laguerre
        node_cut_high = (nodes[igrid_coord+1] + nodes[igrid_coord])/2.0
        if igrid_coord == 1
            # exception for vperp coordinate near orgin
            k = 0
            node_cut_low = node_min
            nquad_coord = nquad_legendre + 2*nquad_laguerre
        else
            # fill in lower Gauss-Legendre points
            node_cut_low = (nodes[igrid_coord-1]+nodes[igrid_coord])/2.0
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

function local_element_integration!(G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                            nquad_vpa,ielement_vpa,vpa_nodes,vpa, # info about primed vperp grids
                            nquad_vperp,ielement_vperp,vperp_nodes,vperp, # info about primed vperp grids
                            x_vpa, w_vpa, x_vperp, w_vperp, # points and weights for primed (source) grids
                            vpa_val, vperp_val) # values and indices for unprimed (field) grids
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
                    #G_elliptic_integral_factor = 2.0*ellipe_mm*prefac/pi
                    G1_elliptic_integral_factor = -(2.0*prefac/pi)*( (2.0 - mm)*ellipe_mm - 2.0*(1.0 - mm)*ellipk_mm )/(3.0*mm)
                    #G2_elliptic_integral_factor = (2.0*prefac/pi)*( (7.0*mm^2 + 8.0*mm - 8.0)*ellipe_mm + 4.0*(2.0 - mm)*(1.0 - mm)*ellipk_mm )/(15.0*mm^2)
                    #G3_elliptic_integral_factor = (2.0*prefac/pi)*( 8.0*(mm^2 - mm + 1.0)*ellipe_mm - 4.0*(2.0 - mm)*(1.0 - mm)*ellipk_mm )/(15.0*mm^2)
                    H_elliptic_integral_factor = 2.0*ellipk_mm/(pi*prefac)
                    H1_elliptic_integral_factor = -(2.0/(pi*prefac))*( (mm-2.0)*(ellipk_mm/mm) + (2.0*ellipe_mm/mm) )
                    H2_elliptic_integral_factor = (2.0/(pi*prefac))*( (3.0*mm^2 - 8.0*mm + 8.0)*(ellipk_mm/(3.0*mm^2)) + (4.0*mm - 8.0)*ellipe_mm/(3.0*mm^2) )
                    lagrange_poly_vpa = lagrange_poly(igrid_vpa,vpa_nodes,x_kvpa)
                    lagrange_poly_vperp = lagrange_poly(igrid_vperp,vperp_nodes,x_kvperp)
                    
                    #(G_weights[ivpap,ivperpp] += 
                    #    lagrange_poly_vpa*lagrange_poly_vperp*
                    #    G_elliptic_integral_factor*x_kvperp*w_kvperp*w_kvpa*2.0/sqrt(pi))
                    
                    (G1_weights[ivpap,ivperpp] += 
                        lagrange_poly_vpa*lagrange_poly_vperp*
                        G1_elliptic_integral_factor*x_kvperp*w_kvperp*w_kvpa*2.0/sqrt(pi))
                    
                    #(G2_weights[ivpap,ivperpp] += 
                    #    lagrange_poly_vpa*lagrange_poly_vperp*
                    #    G2_elliptic_integral_factor*x_kvperp*w_kvperp*w_kvpa*2.0/sqrt(pi))
                    
                    #(G3_weights[ivpap,ivperpp] += 
                    #    lagrange_poly_vpa*lagrange_poly_vperp*
                    #    G3_elliptic_integral_factor*w_kvperp*w_kvpa*2.0/sqrt(pi))
                    
                    (H0_weights[ivpap,ivperpp] += 
                        lagrange_poly_vpa*lagrange_poly_vperp*
                        H_elliptic_integral_factor*x_kvperp*w_kvperp*w_kvpa*2.0/sqrt(pi))
                        
                    (H1_weights[ivpap,ivperpp] += 
                        lagrange_poly_vpa*lagrange_poly_vperp*
                        H1_elliptic_integral_factor*x_kvperp*w_kvperp*w_kvpa*2.0/sqrt(pi))
                        
                    (H2_weights[ivpap,ivperpp] += 
                        lagrange_poly_vpa*lagrange_poly_vperp*
                        (H1_elliptic_integral_factor*vperp_val - H2_elliptic_integral_factor*x_kvperp)*
                        x_kvperp*w_kvperp*w_kvpa*2.0/sqrt(pi))
                    (H3_weights[ivpap,ivperpp] += 
                        lagrange_poly_vpa*lagrange_poly_vperp*
                        H_elliptic_integral_factor*(vpa_val - x_kvpa)*
                        x_kvperp*w_kvperp*w_kvpa*2.0/sqrt(pi))
                    
                    #(n_weights[ivpap,ivperpp] += 
                    #    lagrange_poly_vpa*lagrange_poly_vperp*
                    #    x_kvperp*w_kvperp*w_kvpa*2.0/sqrt(pi))
                end
            end
        end
    end
    return nothing
end

function loop_over_vpa_elements!(G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                            vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vperp grids
                            vperp,ielement_vperpp, # info about primed vperp grids
                            x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                            x_legendre,w_legendre,x_laguerre,w_laguerre,
                            igrid_vpa, igrid_vperp, vpa_val, vperp_val)
    vperp_nodes = get_nodes(vperp,ielement_vperpp)
    vperp_max = vperp_nodes[end]
    vperp_min = vperp_nodes[1]*nel_low(ielement_vperpp,vperp.nelement_local) 
    nquad_vperp = get_scaled_x_w_no_divergences!(x_vperp, w_vperp, x_legendre, w_legendre, vperp_min, vperp_max)
    for ielement_vpap in 1:ielement_vpa_low-1 
        # do integration over part of the domain with no divergences
        vpa_nodes = get_nodes(vpa,ielement_vpap)
        vpa_min, vpa_max = vpa_nodes[1], vpa_nodes[end]
        nquad_vpa = get_scaled_x_w_no_divergences!(x_vpa, w_vpa, x_legendre, w_legendre, vpa_min, vpa_max)
        @views local_element_integration!(G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                    nquad_vpa,ielement_vpap,vpa_nodes,vpa,
                    nquad_vperp,ielement_vperpp,vperp_nodes,vperp,
                    x_vpa, w_vpa, x_vperp, w_vperp, 
                    vpa_val, vperp_val)
    end
    nquad_vperp = get_scaled_x_w!(x_vperp, w_vperp, x_legendre, w_legendre, x_laguerre, w_laguerre, vperp_min, vperp_max, vperp_nodes, igrid_vperp, vperp_val)
    for ielement_vpap in ielement_vpa_low:ielement_vpa_hi
    #for ielement_vpap in 1:vpa.nelement_local
        # use general grid function that checks divergences
        vpa_nodes = get_nodes(vpa,ielement_vpap)
        vpa_min, vpa_max = vpa_nodes[1], vpa_nodes[end]
        #nquad_vpa = get_scaled_x_w_no_divergences!(x_vpa, w_vpa, x_legendre, w_legendre, vpa_min, vpa_max)
        nquad_vpa = get_scaled_x_w!(x_vpa, w_vpa, x_legendre, w_legendre, x_laguerre, w_laguerre, vpa_min, vpa_max, vpa_nodes, igrid_vpa, vpa_val)
        @views local_element_integration!(G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                    nquad_vpa,ielement_vpap,vpa_nodes,vpa,
                    nquad_vperp,ielement_vperpp,vperp_nodes,vperp,
                    x_vpa, w_vpa, x_vperp, w_vperp, 
                    vpa_val, vperp_val)
    end
    nquad_vperp = get_scaled_x_w_no_divergences!(x_vperp, w_vperp, x_legendre, w_legendre, vperp_min, vperp_max)
    for ielement_vpap in ielement_vpa_hi+1:vpa.nelement_local
        # do integration over part of the domain with no divergences
        vpa_nodes = get_nodes(vpa,ielement_vpap)
        vpa_min, vpa_max = vpa_nodes[1], vpa_nodes[end]
        nquad_vpa = get_scaled_x_w_no_divergences!(x_vpa, w_vpa, x_legendre, w_legendre, vpa_min, vpa_max)
        @views local_element_integration!(G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                    nquad_vpa,ielement_vpap,vpa_nodes,vpa,
                    nquad_vperp,ielement_vperpp,vperp_nodes,vperp,
                    x_vpa, w_vpa, x_vperp, w_vperp, 
                    vpa_val, vperp_val)
                    
    end
    return nothing
end

function loop_over_vpa_elements_no_divergences!(G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                            vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vperp grids
                            nquad_vperp,ielement_vperpp,vperp_nodes,vperp, # info about primed vperp grids
                            x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                            x_legendre,w_legendre,
                            vpa_val, vperp_val)
    for ielement_vpap in 1:vpa.nelement_local
        # do integration over part of the domain with no divergences
        vpa_nodes = get_nodes(vpa,ielement_vpap)
        vpa_min, vpa_max = vpa_nodes[1], vpa_nodes[end]
        nquad_vpa = get_scaled_x_w_no_divergences!(x_vpa, w_vpa, x_legendre, w_legendre, vpa_min, vpa_max)
        @views local_element_integration!(G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                    nquad_vpa,ielement_vpap,vpa_nodes,vpa,
                    nquad_vperp,ielement_vperpp,vperp_nodes,vperp,
                    x_vpa, w_vpa, x_vperp, w_vperp, 
                    vpa_val, vperp_val)
                    
    end
    return nothing
end

function loop_over_vperp_vpa_elements!(G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vpa grids
                vperp,ielement_vperp_low,ielement_vperp_hi, # info about primed vperp grids
                x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                x_legendre,w_legendre,x_laguerre,w_laguerre,
                igrid_vpa, igrid_vperp, vpa_val, vperp_val)
    for ielement_vperpp in 1:ielement_vperp_low-1
        
        vperp_nodes = get_nodes(vperp,ielement_vperpp)
        vperp_max = vperp_nodes[end]
        vperp_min = vperp_nodes[1]*nel_low(ielement_vperpp,vperp.nelement_local) 
        nquad_vperp = get_scaled_x_w_no_divergences!(x_vperp, w_vperp, x_legendre, w_legendre, vperp_min, vperp_max)
        @views loop_over_vpa_elements_no_divergences!(G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vpa grids
                nquad_vperp,ielement_vperpp,vperp_nodes,vperp, # info about primed vperp grids
                x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                x_legendre,w_legendre,
                vpa_val, vperp_val)
    end
    for ielement_vperpp in ielement_vperp_low:ielement_vperp_hi
        
        #vperp_nodes = get_nodes(vperp,ielement_vperpp)
        #vperp_max = vperp_nodes[end]
        #vperp_min = vperp_nodes[1]*nel_low(ielement_vperpp,vperp.nelement_local) 
        #nquad_vperp = get_scaled_x_w_no_divergences!(x_vperp, w_vperp, x_legendre, w_legendre, vperp_min, vperp_max)
        #nquad_vperp = get_scaled_x_w!(x_vperp, w_vperp, x_legendre, w_legendre, x_laguerre, w_laguerre, vperp_min, vperp_max, vperp_nodes, igrid_vperp, vperp_val)
        @views loop_over_vpa_elements!(G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vpa grids
                vperp,ielement_vperpp, # info about primed vperp grids
                x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                x_legendre,w_legendre,x_laguerre,w_laguerre,
                igrid_vpa, igrid_vperp, vpa_val, vperp_val)
    end
    for ielement_vperpp in ielement_vperp_hi+1:vperp.nelement_local
        
        vperp_nodes = get_nodes(vperp,ielement_vperpp)
        vperp_max = vperp_nodes[end]
        vperp_min = vperp_nodes[1]*nel_low(ielement_vperpp,vperp.nelement_local) 
        nquad_vperp = get_scaled_x_w_no_divergences!(x_vperp, w_vperp, x_legendre, w_legendre, vperp_min, vperp_max)
        @views loop_over_vpa_elements_no_divergences!(G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vpa grids
                nquad_vperp,ielement_vperpp,vperp_nodes,vperp, # info about primed vperp grids
                x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                x_legendre,w_legendre,
                vpa_val, vperp_val)
    end
    return nothing
end

function loop_over_vperp_vpa_elements_no_divergences!(G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vpa grids
                vperp,ielement_vperp_low,ielement_vperp_hi, # info about primed vperp grids
                x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                x_legendre,w_legendre,
                igrid_vpa, igrid_vperp, vpa_val, vperp_val)
    for ielement_vperpp in 1:vperp.nelement_local
        vperp_nodes = get_nodes(vperp,ielement_vperpp)
        vperp_max = vperp_nodes[end]
        vperp_min = vperp_nodes[1]*nel_low(ielement_vperpp,nelement_vperp) 
        nquad_vperp = get_scaled_x_w_no_divergences!(x_vperp, w_vperp, x_legendre, w_legendre, vperp_min, vperp_max)
        @views loop_over_vpa_elements_no_divergences!(G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                vpa,ielement_vpa_low,ielement_vpa_hi, # info about primed vpa grids
                nquad_vperp,ielement_vperpp,vperp_nodes,vperp, # info about primed vperp grids
                x_vpa, w_vpa, x_vperp, w_vperp, # arrays to store points and weights for primed (source) grids
                x_legendre,w_legendre,
                vpa_val, vperp_val)
    end
    return nothing
end

"""
calculates the (normalised) Rosenbluth potential G 
"""
# G(vpa,vperp) = \int^\infty_0 \int^{\infty}_{-\infty} ((vpa- vpa')^2 + (vperp + vperp'))^{1/2}
#                 * (2 ellipe(mm)/ \pi) F(vpa',vperp') (2 vperp'/\sqrt{\pi}) d vperp' d vpa'


function calculate_Rosenbluth_potentials!(Rosenbluth_G,Rosenbluth_H,fsp_in,
     elliptic_integral_E_factor,elliptic_integral_K_factor,buffer_vpavperp,vperp,vpa)
    
    for ivperp in 1:vperp.n 
        for ivpa in 1:vpa.n
            # G
            @views @. buffer_vpavperp[:,:] = fsp_in*elliptic_integral_E_factor[ivpa,ivperp,:,:]
            @views Rosenbluth_G[ivpa,ivperp] = integrate_over_vspace(buffer_vpavperp, vpa.grid, 0, vpa.wgts, vperp.grid, 0, vperp.wgts)
            # H 
            @views @. buffer_vpavperp[:,:] = fsp_in*elliptic_integral_K_factor[ivpa,ivperp,:,:]
            @views Rosenbluth_H[ivpa,ivperp] = integrate_over_vspace(buffer_vpavperp, vpa.grid, 0, vpa.wgts, vperp.grid, 0, vperp.wgts)
        end
    end

end

"""
Computes the Laplacian of G in vpa vperp coordinates to obtain H
""" 
function calculate_Rosenbluth_H_from_G!(Rosenbluth_H,Rosenbluth_G,vpa,vpa_spectral,vperp,vperp_spectral,buffer_vpavperp_1,buffer_vpavperp_2)
    Rosenbluth_H .= 0.0
    for ivperp in 1:vperp.n
        @views derivative!(vpa.scratch, Rosenbluth_G[:,ivperp], vpa, vpa_spectral)
        @views derivative!(vpa.scratch2, vpa.scratch, vpa, vpa_spectral)
        @views @. buffer_vpavperp_1[:,ivperp] = vpa.scratch2
    end 
    for ivpa in 1:vpa.n
        @views derivative!(vperp.scratch, Rosenbluth_G[ivpa,:], vperp, vperp_spectral)
        @. vperp.scratch = vperp.grid*vperp.scratch
        @views derivative!(vperp.scratch2, vperp.scratch, vperp, vperp_spectral)
        @views @. buffer_vpavperp_2[ivpa,:] = vperp.scratch2/vperp.grid
    end
    @views @. Rosenbluth_H = 0.5*(buffer_vpavperp_1 + buffer_vpavperp_2)
end
 

"""
calculates the (normalised) Rosenbluth potential coefficients d2Gdvpa2, d2Gdvperpdvpa, ..., dHdvperp for a Maxwellian inputs.
"""
function calculate_Maxwellian_Rosenbluth_coefficients(dens,upar,vth,vpa,vperp,ivpa,ivperp,n_ion_species) # Rosenbluth_d2Gdvpa2,Rosenbluth_d2Gdvperpdvpa,Rosenbluth_d2Gdvperp2,Rosenbluth_dHdvpa,Rosenbluth_dHdvperp,
    # zero coefficients prior to looping over s'
    Rosenbluth_d2Gdvpa2 = 0.0
    Rosenbluth_d2Gdvperpdvpa = 0.0
    Rosenbluth_d2Gdvperp2 = 0.0
    Rosenbluth_dHdvpa = 0.0
    Rosenbluth_dHdvperp = 0.0
    
    # fill in value at (ivpa,ivperp)
    for isp in 1:n_ion_species
        Rosenbluth_d2Gdvpa2 += d2Gdvpa2(dens[isp],upar[isp],vth[isp],vpa,vperp,ivpa,ivperp)
        Rosenbluth_d2Gdvperpdvpa += d2Gdvperpdvpa(dens[isp],upar[isp],vth[isp],vpa,vperp,ivpa,ivperp)
        Rosenbluth_d2Gdvperp2 += d2Gdvperp2(dens[isp],upar[isp],vth[isp],vpa,vperp,ivpa,ivperp)
        Rosenbluth_dHdvpa += dHdvpa(dens[isp],upar[isp],vth[isp],vpa,vperp,ivpa,ivperp)
        Rosenbluth_dHdvperp += dHdvperp(dens[isp],upar[isp],vth[isp],vpa,vperp,ivpa,ivperp)
    end
    return Rosenbluth_d2Gdvpa2,Rosenbluth_d2Gdvperpdvpa,Rosenbluth_d2Gdvperp2,Rosenbluth_dHdvpa,Rosenbluth_dHdvperp
end

"""
calculates the collisional fluxes given input F_s and G_sp, H_sp
"""
function calculate_collisional_fluxes(F,dFdvpa,dFdvperp,
                            d2Gdvpa2,d2Gdvperpdvpa,d2Gdvperp2,dHdvpa,dHdvperp,
                            ms,msp)
    # fill in value at (ivpa,ivperp)
    Cflux_vpa = dFdvpa*d2Gdvpa2 + dFdvperp*d2Gdvperpdvpa - 2.0*(ms/msp)*F*dHdvpa
    #Cflux_vpa = dFdvpa*d2Gdvpa2 + dFdvperp*d2Gdvperpdvpa # - 2.0*(ms/msp)*F*dHdvpa
    #Cflux_vpa =  - 2.0*(ms/msp)*F*dHdvpa
    Cflux_vperp = dFdvpa*d2Gdvperpdvpa + dFdvperp*d2Gdvperp2 - 2.0*(ms/msp)*F*dHdvperp
    return Cflux_vpa, Cflux_vperp
end

"""
returns (normalised) C[Fs,Fs']

"""
#returns (normalised) C[F_s,F_s'] = C[F_s,F_s'](vpa,vperp) given inputs
#distribution F_s = F_s(vpa,vperp) 
#distribution F_s' = F_s'(vpa,vperp) 
#mass m_s 
#mass m_s'
#collision frequency nu_{ss'} = gamma_{ss'} n_{ref} / 2 (m_s)^2 (c_{ref})^3
#with gamma_ss' = 2 pi (Z_s Z_s')^2 e^4 ln \Lambda_{ss'} / (4 pi \epsilon_0)^2 
function evaluate_RMJ_collision_operator!(Cssp_out,fs_in,fsp_in,ms,msp,cfreqssp, fokkerplanck_arrays::fokkerplanck_arrays_struct, vperp, vpa, vperp_spectral, vpa_spectral)
    # calculate the Rosenbluth potentials
    # and store in fokkerplanck_arrays_struct
    @views calculate_Rosenbluth_potentials!(fokkerplanck_arrays.Rosenbluth_G,fokkerplanck_arrays.Rosenbluth_H,fsp_in,
     fokkerplanck_arrays.elliptic_integral_E_factor,
     fokkerplanck_arrays.elliptic_integral_K_factor,
     fokkerplanck_arrays.buffer_vpavperp_1,vperp,vpa)
    
    # short names for buffer arrays 
    buffer_1 = fokkerplanck_arrays.buffer_vpavperp_1
    buffer_2 = fokkerplanck_arrays.buffer_vpavperp_2
    Rosenbluth_G = fokkerplanck_arrays.Rosenbluth_G
    Rosenbluth_H = fokkerplanck_arrays.Rosenbluth_H
    nvperp = vperp.n 
    nvpa = vpa.n 
    # zero Cssp to prepare for addition of collision terms 
    Cssp_out .= 0.0
    
    #  + d^2 F_s / d vpa^2 * d^2 G_sp / d vpa^2 
    for ivperp in 1:nvperp
        vpa.scratch2 .= 1.0 # remove Q argument from second_derivative! as never different from 1?
        @views second_derivative!(vpa.scratch, fs_in[:,ivperp], vpa.scratch2, vpa, vpa_spectral)
        @views @. buffer_1[:,ivperp] = vpa.scratch
        @views second_derivative!(vpa.scratch, Rosenbluth_G[:,ivperp], vpa.scratch2, vpa, vpa_spectral)
        @views @. buffer_2[:,ivperp] = vpa.scratch
    end 
    @views @. Cssp_out += buffer_1*buffer_2
    
    #  + 2 d^2 F_s / d vpa d vperp * d^2 G_sp / d vpa d vperp 
    for ivperp in 1:nvperp
        @views derivative!(vpa.scratch, fs_in[:,ivperp], vpa, vpa_spectral)
        @views @. buffer_1[:,ivperp] = vpa.scratch
        @views derivative!(vpa.scratch, Rosenbluth_G[:,ivperp], vpa, vpa_spectral)
        @views @. buffer_2[:,ivperp] = vpa.scratch
    end 
    for ivpa in 1:nvpa
        @views derivative!(vperp.scratch, buffer_1[ivpa,:], vperp, vperp_spectral)
        @views @. buffer_1[ivpa,:] = vperp.scratch
        @views derivative!(vperp.scratch, buffer_2[ivpa,:], vperp, vperp_spectral)
        @views @. buffer_2[ivpa,:] = vperp.scratch
    end 
    @views @. Cssp_out += 2.0*buffer_1*buffer_2
    
    #  + d^2 F_s / d vperp^2 * d^2 G_sp / d vperp^2 
    for ivpa in 1:nvpa
        vperp.scratch2 .= 1.0 # remove Q argument from second_derivative! as never different from 1?
        @views second_derivative!(vperp.scratch, fs_in[ivpa,:], vperp.scratch2, vperp, vperp_spectral)
        @views @. buffer_1[ivpa,:] = vperp.scratch
        @views second_derivative!(vperp.scratch, Rosenbluth_G[ivpa,:], vperp.scratch2, vperp, vperp_spectral)
        @views @. buffer_2[ivpa,:] = vperp.scratch
    end 
    @views @. Cssp_out += buffer_1*buffer_2
    
    #  + ( 1/vperp^2) d F_s / d vperp * d G_sp / d vperp 
    for ivpa in 1:nvpa
        @views derivative!(vperp.scratch, fs_in[ivpa,:], vperp, vperp_spectral)
        @views @. buffer_1[ivpa,:] = vperp.scratch/(vperp.grid^2) # MRH this line causes divide by zero!
        @views derivative!(vperp.scratch, Rosenbluth_G[ivpa,:], vperp, vperp_spectral)
        @views @. buffer_2[ivpa,:] = vperp.scratch
    end 
    @views @. Cssp_out += buffer_1*buffer_2
    
    #  + 2( 1 - ms/msp) * d F_s / d vpa * d H_sp / d vpa 
    for ivperp in 1:nvperp
        @views derivative!(vpa.scratch, fs_in[:,ivperp], vpa, vpa_spectral)
        @views @. buffer_1[:,ivperp] = vpa.scratch
        @views derivative!(vpa.scratch, Rosenbluth_H[:,ivperp], vpa, vpa_spectral)
        @views @. buffer_2[:,ivperp] = vpa.scratch
    end 
    @views @. Cssp_out += 2.0*(1.0 - ms/msp)*buffer_1*buffer_2
    
    #  + 2( 1 - ms/msp) * d F_s / d vperp * d H_sp / d vperp 
    for ivpa in 1:nvpa
        @views derivative!(vperp.scratch, fs_in[ivpa,:], vperp, vperp_spectral)
        @views @. buffer_1[ivpa,:] = vperp.scratch
        @views derivative!(vperp.scratch, Rosenbluth_H[ivpa,:], vperp, vperp_spectral)
        @views @. buffer_2[ivpa,:] = vperp.scratch
    end 
    @views @. Cssp_out += 2.0*(1.0 - ms/msp)*buffer_1*buffer_2
    
    # + (8 ms / \sqrt{\pi} msp ) F_s F_sp 
    @views @. Cssp_out += ((8.0*ms)/(sqrt(pi)*msp))*fs_in*fsp_in
    
    # multiply by overall collision frequency
    @views @. Cssp_out = cfreqssp*Cssp_out
end 

function explicit_fokker_planck_collisions_old!(pdf_out,pdf_in,composition,collisions,dt,fokkerplanck_arrays::fokkerplanck_arrays_struct,
                                             scratch_dummy, r, z, vperp, vpa, vperp_spectral, vpa_spectral)
    n_ion_species = composition.n_ion_species
    @boundscheck vpa.n == size(pdf_out,1) || throw(BoundsError(pdf_out))
    @boundscheck vperp.n == size(pdf_out,2) || throw(BoundsError(pdf_out))
    @boundscheck z.n == size(pdf_out,3) || throw(BoundsError(pdf_out))
    @boundscheck r.n == size(pdf_out,4) || throw(BoundsError(pdf_out))
    @boundscheck n_ion_species == size(pdf_out,5) || throw(BoundsError(pdf_out))
    @boundscheck vpa.n == size(pdf_in,1) || throw(BoundsError(pdf_in))
    @boundscheck vperp.n == size(pdf_in,2) || throw(BoundsError(pdf_in))
    @boundscheck z.n == size(pdf_in,3) || throw(BoundsError(pdf_in))
    @boundscheck r.n == size(pdf_in,4) || throw(BoundsError(pdf_in))
    @boundscheck n_ion_species == size(pdf_in,5) || throw(BoundsError(pdf_in))
    Cssp_result_vpavperp = scratch_dummy.dummy_vpavperp
    mi = 1.0 # generalise this to an Array with size n_ion_species
    cfreqii = collisions.nuii # generalise this to an Array with size (n_ion_species,n_ion_species)
    
    begin_r_z_region()
    # serial in s vperp vpa for now 
    for is in 1:n_ion_species
        for isp in 1:n_ion_species       
            @loop_r_z ir iz begin
                @views evaluate_RMJ_collision_operator!(Cssp_result_vpavperp,pdf_in[:,:,iz,ir,is],pdf_in[:,:,iz,ir,isp],
                                                        mi, mi, cfreqii, fokkerplanck_arrays, vperp, vpa,
                                                        vperp_spectral, vpa_spectral)
                @views @. pdf_out[:,:,iz,ir,is] += dt*Cssp_result_vpavperp[:,:]
            end
        end
    end
end

"""
Function to carry out the integration of the revelant
distribution functions to form the required coefficients
for the full-F operator. We assume that the weights are
precalculated. The function takes as arguments the arrays
of coefficients (which we fill), the required distributions,
the precomputed weights, the indicies of the `field' velocities,
and the sizes of the primed vpa and vperp coordinates arrays.
"""
function get_local_Cssp_coefficients!(d2Gspdvpa2,dGspdvperp,d2Gspdvperpdvpa,
                                        d2Gspdvperp2,dHspdvpa,dHspdvperp,
                                        dfspdvpa,dfspdvperp,d2fspdvperpdvpa,
                                        G1_weights,H0_weights,H1_weights,H2_weights,H3_weights,
                                        ivpa,ivperp,nvpa,nvperp)
    d2Gspdvpa2[ivpa,ivperp] = 0.0
    dGspdvperp[ivpa,ivperp] = 0.0
    d2Gspdvperpdvpa[ivpa,ivperp] = 0.0
    d2Gspdvperp2[ivpa,ivperp] = 0.0
    dHspdvpa[ivpa,ivperp] = 0.0
    dHspdvperp[ivpa,ivperp] = 0.0
    for ivperpp in 1:nvperp
        for ivpap in 1:nvpa
            #d2Gspdvpa2[ivpa,ivperp] += G_weights[ivpap,ivperpp,ivpa,ivperp]*d2fspdvpa2[ivpap,ivperpp]
            d2Gspdvpa2[ivpa,ivperp] += H3_weights[ivpap,ivperpp,ivpa,ivperp]*dfspdvpa[ivpap,ivperpp]
            dGspdvperp[ivpa,ivperp] += G1_weights[ivpap,ivperpp,ivpa,ivperp]*dfspdvperp[ivpap,ivperpp]
            d2Gspdvperpdvpa[ivpa,ivperp] += G1_weights[ivpap,ivperpp,ivpa,ivperp]*d2fspdvperpdvpa[ivpap,ivperpp]
            #d2Gspdvperp2[ivpa,ivperp] += G2_weights[ivpap,ivperpp,ivpa,ivperp]*d2fspdvperp2[ivpap,ivperpp] + G3_weights[ivpap,ivperpp,ivpa,ivperp]*dfspdvperp[ivpap,ivperpp]
            d2Gspdvperp2[ivpa,ivperp] += H2_weights[ivpap,ivperpp,ivpa,ivperp]*dfspdvperp[ivpap,ivperpp]
            dHspdvpa[ivpa,ivperp] += H0_weights[ivpap,ivperpp,ivpa,ivperp]*dfspdvpa[ivpap,ivperpp]
            dHspdvperp[ivpa,ivperp] += H1_weights[ivpap,ivperpp,ivpa,ivperp]*dfspdvperp[ivpap,ivperpp]
        end
    end
    return nothing
end
"""
Function calculating the fully expanded form of the collision operator
taking floats as arguments. This function is designed to be used at the 
lowest level of a coordinate loop, with derivatives and integrals
all previously calculated.
"""

function Cssp_fully_expanded_form(nussp,ms,msp,
            d2fsdvpa2,d2fsdvperp2,d2fsdvperpdvpa,dfsdvpa,dfsdvperp,fs,
            d2Gspdvpa2,d2Gspdvperp2,d2Gspdvperpdvpa,dGspdvperp,
            dHspdvpa,dHspdvperp,fsp,vperp_val)
    ( Cssp = nussp*( d2fsdvpa2*d2Gspdvpa2 +
              d2fsdvperp2*d2Gspdvperp2 +
              2.0*d2fsdvperpdvpa*d2Gspdvperpdvpa +                
              (1.0/(vperp_val^2))*dfsdvperp*dGspdvperp +                
              2.0*(1.0 - (ms/msp))*(dfsdvpa*dHspdvpa + dfsdvperp*dHspdvperp) +                
              (8.0/sqrt(pi))*(ms/msp)*fs*fsp) )
    return Cssp
end

"""
Evaluate the Fokker Planck collision Operator
using dummy arrays to store the 5 required derivatives.
For a single species, ir, and iz, this routine leaves 
in place the fokkerplanck_arrays struct with testable 
distributions function derivatives, Rosenbluth potentials,
and collision operator in place.
"""

function explicit_fokker_planck_collisions!(pdf_out,pdf_in,dSdt,composition,collisions,dt,fokkerplanck_arrays::fokkerplanck_arrays_struct,
                                             scratch_dummy, r, z, vperp, vpa, vperp_spectral, vpa_spectral, boundary_distributions, advance,
                                             vpa_advect, z_advect, r_advect; diagnose_entropy_production = true)
    n_ion_species = composition.n_ion_species
    @boundscheck vpa.n == size(pdf_out,1) || throw(BoundsError(pdf_out))
    @boundscheck vperp.n == size(pdf_out,2) || throw(BoundsError(pdf_out))
    @boundscheck z.n == size(pdf_out,3) || throw(BoundsError(pdf_out))
    @boundscheck r.n == size(pdf_out,4) || throw(BoundsError(pdf_out))
    @boundscheck n_ion_species == size(pdf_out,5) || throw(BoundsError(pdf_out))
    @boundscheck vpa.n == size(pdf_in,1) || throw(BoundsError(pdf_in))
    @boundscheck vperp.n == size(pdf_in,2) || throw(BoundsError(pdf_in))
    @boundscheck z.n == size(pdf_in,3) || throw(BoundsError(pdf_in))
    @boundscheck r.n == size(pdf_in,4) || throw(BoundsError(pdf_in))
    @boundscheck n_ion_species == size(pdf_in,5) || throw(BoundsError(pdf_in))
    @boundscheck z.n == size(dSdt,1) || throw(BoundsError(dSdt))
    @boundscheck r.n == size(dSdt,2) || throw(BoundsError(dSdt))
    @boundscheck n_ion_species == size(dSdt,3) || throw(BoundsError(dSdt))
    
    # setup species information
    mass = Array{mk_float,1}(undef,n_ion_species)
    mass[1] = 1.0 # generalise!
    nussp = Array{mk_float,2}(undef,n_ion_species,n_ion_species)
    nussp[1,1] = collisions.nuii # generalise!
    
    # assign Cssp to a dummy array
    Cssp = scratch_dummy.dummy_s
    
    # first, compute the require derivatives and store in the buffer arrays
    dfdvpa = scratch_dummy.buffer_vpavperpzrs_1
    d2fdvpa2 = scratch_dummy.buffer_vpavperpzrs_2
    d2fdvperpdvpa = scratch_dummy.buffer_vpavperpzrs_3
    dfdvperp = scratch_dummy.buffer_vpavperpzrs_4
    d2fdvperp2 = scratch_dummy.buffer_vpavperpzrs_5
    logfC = scratch_dummy.buffer_vpavperpzrs_6

    begin_s_r_z_vperp_region()
    @loop_s_r_z_vperp is ir iz ivperp begin
        @views derivative!(vpa.scratch, pdf_in[:,ivperp,iz,ir,is], vpa, vpa_spectral)
        @. dfdvpa[:,ivperp,iz,ir,is] = vpa.scratch
        @views derivative!(vpa.scratch2, vpa.scratch, vpa, vpa_spectral)
        @. d2fdvpa2[:,ivperp,iz,ir,is] = vpa.scratch2
    end
    if vpa.discretization == "gausslegendre_pseudospectral"
        @loop_s_r_z_vperp is ir iz ivperp begin
           @views second_derivative!(vpa.scratch2, pdf_in[:,ivperp,iz,ir,is], vpa, vpa_spectral)
           @. d2fdvpa2[:,ivperp,iz,ir,is] = vpa.scratch2 
        end
    end

    begin_s_r_z_vpa_region()

    @loop_s_r_z_vpa is ir iz ivpa begin
        @views derivative!(vperp.scratch, pdf_in[ivpa,:,iz,ir,is], vperp, vperp_spectral)
        @. dfdvperp[ivpa,:,iz,ir,is] = vperp.scratch
        @views derivative!(vperp.scratch2, vperp.scratch, vperp, vperp_spectral)
        @. d2fdvperp2[ivpa,:,iz,ir,is] = vperp.scratch2
        @views derivative!(vperp.scratch, dfdvpa[ivpa,:,iz,ir,is], vperp, vperp_spectral)
        @. d2fdvperpdvpa[ivpa,:,iz,ir,is] = vperp.scratch
    end

    # to permit moment conservation, store the current moments of pdf_out
    # this involves imposing the boundary conditions to the present pre-collisions pdf_out
    if collisions.numerical_conserving_terms == "density+u||+T" || collisions.numerical_conserving_terms == "density"  
        store_moments_in_buffer!(pdf_out,boundary_distributions,
      vpa, vperp, z, r, vpa_advect, z_advect, r_advect, composition,
      scratch_dummy, advance, vperp_spectral, vpa_spectral)
    end
    # now parallelise over all dimensions and calculate the 
    # collision operator coefficients and the collision operator
    # in one loop, noting that we only require data local to 
    # each ivpa,ivperp,iz,ir,is now that the derivatives are precomputed
    fka = fokkerplanck_arrays
    begin_s_r_z_vperp_vpa_region()
    @loop_s_r_z is ir iz begin
        @loop_vperp_vpa ivperp ivpa begin
            for isp in 1:n_ion_species # make sure to sum over all ion species
                # get the local (in ivpa, ivperp) values of the coeffs
                @views get_local_Cssp_coefficients!(fka.d2Gdvpa2,fka.dGdvperp,fka.d2Gdvperpdvpa,
                                            fka.d2Gdvperp2,fka.dHdvpa,fka.dHdvperp,
                                            dfdvpa[:,:,iz,ir,isp],dfdvperp[:,:,iz,ir,isp],d2fdvperpdvpa[:,:,iz,ir,isp],
                                            fka.G1_weights,fka.H0_weights,fka.H1_weights,fka.H2_weights,fka.H3_weights,
                                            ivpa,ivperp,vpa.n,vperp.n)
                
                (Cssp[isp] = Cssp_fully_expanded_form(nussp[is,isp],mass[is],mass[isp],
                                                d2fdvpa2[ivpa,ivperp,iz,ir,is],d2fdvperp2[ivpa,ivperp,iz,ir,is],d2fdvperpdvpa[ivpa,ivperp,iz,ir,is],dfdvpa[ivpa,ivperp,iz,ir,is],dfdvperp[ivpa,ivperp,iz,ir,is],pdf_in[ivpa,ivperp,iz,ir,is],
                                                fka.d2Gdvpa2[ivpa,ivperp],fka.d2Gdvperp2[ivpa,ivperp],fka.d2Gdvperpdvpa[ivpa,ivperp],fka.dGdvperp[ivpa,ivperp],
                                                fka.dHdvpa[ivpa,ivperp],fka.dHdvperp[ivpa,ivperp],pdf_in[ivpa,ivperp,iz,ir,isp],vperp.grid[ivperp]) )
                pdf_out[ivpa,ivperp,iz,ir,is] += dt*Cssp[isp]
                # for testing
                fka.Cssp_result_vpavperp[ivpa,ivperp] = Cssp[isp]
                fka.dfdvpa[ivpa,ivperp] = dfdvpa[ivpa,ivperp,iz,ir,is]
                fka.d2fdvpa2[ivpa,ivperp] = d2fdvpa2[ivpa,ivperp,iz,ir,is]
                fka.d2fdvperpdvpa[ivpa,ivperp] = d2fdvperpdvpa[ivpa,ivperp,iz,ir,is]
                fka.dfdvperp[ivpa,ivperp] = dfdvperp[ivpa,ivperp,iz,ir,is]
                fka.d2fdvperp2[ivpa,ivperp] = d2fdvperp2[ivpa,ivperp,iz,ir,is]
            end
            # store the entropy production
            # we use ln|f| to avoid problems with f < 0. This is ok if C_s[f,f] is small where f ~< 0
            # + 1.0e-15 in case f = 0 exactly
            #println(Cssp[:])
            #println(pdf_in[ivpa,ivperp,iz,ir,is])
            logfC[ivpa,ivperp,iz,ir,is] = log(abs(pdf_in[ivpa,ivperp,iz,ir,is]) + 1.0e-15)*sum(Cssp[:])
            #println(dfdvpa[ivpa,ivperp,iz,ir,is])
       end
    end
    if diagnose_entropy_production
        # compute entropy production diagnostic
        begin_s_r_z_region()
        @loop_s_r_z is ir iz begin
           @views dSdt[iz,ir,is] = -integrate_over_vspace(logfC[:,:,iz,ir,is], vpa.grid, 0, vpa.wgts, vperp.grid, 0, vperp.wgts)
        end
    end
    if collisions.numerical_conserving_terms == "density+u||+T"
        # use an ad-hoc numerical model to conserve density, upar, vth
        # a different model is required for inter-species collisions
        # simply conserving particle density may be more appropriate in the multi-species case
        apply_numerical_conserving_terms!(pdf_out,pdf_in,boundary_distributions,
      vpa, vperp, z, r, vpa_advect, z_advect, r_advect, composition,
      scratch_dummy, advance, vperp_spectral, vpa_spectral)
    elseif collisions.numerical_conserving_terms == "density"
        apply_density_conserving_terms!(pdf_out,pdf_in,boundary_distributions,
      vpa, vperp, z, r, vpa_advect, z_advect, r_advect, composition,
      scratch_dummy, advance, vperp_spectral, vpa_spectral)
    elseif !(collisions.numerical_conserving_terms == "none")
        println("ERROR: collisions.numerical_conserving_terms = ",collisions.numerical_conserving_terms," NOT SUPPORTED")
    end
    return nothing 
end

function explicit_fokker_planck_collisions_Maxwellian_coefficients!(pdf_out,pdf_in,dens_in,upar_in,vth_in,
                                             composition,collisions,dt,fokkerplanck_arrays::fokkerplanck_arrays_struct,
                                             scratch_dummy, r, z, vperp, vpa, vperp_spectral, vpa_spectral)
    n_ion_species = composition.n_ion_species
    @boundscheck vpa.n == size(pdf_out,1) || throw(BoundsError(pdf_out))
    @boundscheck vperp.n == size(pdf_out,2) || throw(BoundsError(pdf_out))
    @boundscheck z.n == size(pdf_out,3) || throw(BoundsError(pdf_out))
    @boundscheck r.n == size(pdf_out,4) || throw(BoundsError(pdf_out))
    @boundscheck n_ion_species == size(pdf_out,5) || throw(BoundsError(pdf_out))
    @boundscheck vpa.n == size(pdf_in,1) || throw(BoundsError(pdf_in))
    @boundscheck vperp.n == size(pdf_in,2) || throw(BoundsError(pdf_in))
    @boundscheck z.n == size(pdf_in,3) || throw(BoundsError(pdf_in))
    @boundscheck r.n == size(pdf_in,4) || throw(BoundsError(pdf_in))
    @boundscheck n_ion_species == size(pdf_in,5) || throw(BoundsError(pdf_in))
    
    mi = 1.0 # generalise this to an Array with size n_ion_species
    mip = 1.0 # generalise this to an Array with size n_ion_species
    cfreqii = collisions.nuii # generalise this to an Array with size (n_ion_species,n_ion_species)
    fk = fokkerplanck_arrays
    Cssp_result_vpavperp = scratch_dummy.dummy_vpavperp
    pdf_buffer_1 = scratch_dummy.buffer_vpavperpzrs_1
    pdf_buffer_2 = scratch_dummy.buffer_vpavperpzrs_2
    
    # precompute derivatives of the pdfs to benefit from parallelisation
    # d F / d vpa
    begin_s_r_z_vperp_region()
    @loop_s_r_z_vperp is ir iz ivperp begin
        @views derivative!(vpa.scratch, pdf_in[:,ivperp,iz,ir,is], vpa, vpa_spectral)
        @. pdf_buffer_1[:,ivperp,iz,ir,is] = vpa.scratch
    end
    # d F / d vperp
    begin_s_r_z_vpa_region()
    @loop_s_r_z_vpa is ir iz ivpa begin
        @views derivative!(vperp.scratch, pdf_in[ivpa,:,iz,ir,is], vperp, vperp_spectral)
        @. pdf_buffer_2[ivpa,:,iz,ir,is] = vperp.scratch
    end
    
    begin_s_r_z_vperp_vpa_region()
    @loop_s_r_z is ir iz begin
        @loop_vperp_vpa ivperp ivpa begin
            # first compute local (in z,r) Rosenbluth potential coefficients, summing over all s'
            ((Rosenbluth_d2Gdvpa2, Rosenbluth_d2Gdvperpdvpa, 
            Rosenbluth_d2Gdvperp2,Rosenbluth_dHdvpa,
            Rosenbluth_dHdvperp) = calculate_Maxwellian_Rosenbluth_coefficients(dens_in[iz,ir,:],
                 upar_in[iz,ir,:],vth_in[iz,ir,:],vpa,vperp,ivpa,ivperp,n_ion_species) )
                 
            # now form the collisional fluxes at this s,z,r
            ( (Cflux_vpa,Cflux_vperp) = calculate_collisional_fluxes(pdf_in[ivpa,ivperp,iz,ir,is],
                    pdf_buffer_1[ivpa,ivperp,iz,ir,is],pdf_buffer_2[ivpa,ivperp,iz,ir,is],
                    Rosenbluth_d2Gdvpa2,Rosenbluth_d2Gdvperpdvpa,
                    Rosenbluth_d2Gdvperp2,Rosenbluth_dHdvpa,Rosenbluth_dHdvperp,
                    mi,mip) )
            
            # now overwrite the buffer arrays with the local values as we no longer need dFdvpa or dFdvperp at s,r,z
            pdf_buffer_1[ivpa,ivperp,iz,ir,is] = Cflux_vpa
            pdf_buffer_2[ivpa,ivperp,iz,ir,is] = Cflux_vperp
        end
        
    end
    
    # now differentiate the fluxes to obtain the explicit operator 
    
    # d Cflux_vpa / d vpa
    begin_s_r_z_vperp_region()
    @loop_s_r_z_vperp is ir iz ivperp begin
        @views derivative!(vpa.scratch, pdf_buffer_1[:,ivperp,iz,ir,is], vpa, vpa_spectral)
        @. pdf_buffer_1[:,ivperp,iz,ir,is] = vpa.scratch
    end
    # (1/vperp) d Cflux_vperp / d vperp
    begin_s_r_z_vpa_region()
    @loop_s_r_z_vpa is ir iz ivpa begin
        @views @. vperp.scratch2 = vperp.grid*pdf_buffer_2[ivpa,:,iz,ir,is]
        @views derivative!(vperp.scratch, vperp.scratch2, vperp, vperp_spectral)
        @. pdf_buffer_2[ivpa,:,iz,ir,is] = vperp.scratch[:]/vperp.grid[:]
    end
    
    # now add the result to the outgoing pdf
    # d F / d t = nu_ii * ( d Cflux_vpa / d vpa + (1/vperp) d Cflux_vperp / d vperp)
    begin_s_r_z_vperp_vpa_region()
    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        pdf_out[ivpa,ivperp,iz,ir,is] += dt*cfreqii*(pdf_buffer_1[ivpa,ivperp,iz,ir,is] + pdf_buffer_1[ivpa,ivperp,iz,ir,is])
    end
end

# below are a series of functions that can be used to test the calculation 
# of the Rosenbluth potentials for a shifted Maxwellian
# or provide an estimate for collisional coefficients 

# G (defined by Del^4 G = -(8/sqrt(pi))*F 
# with F = cref^3 pi^(3/2) F_Maxwellian / nref 
# the normalised Maxwellian
function G_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
    # speed variable
    eta = eta_func(upar,vth,vpa,vperp,ivpa,ivperp)
    zero = 1.0e-10
    if eta < zero
        G = 2.0/sqrt(pi)
    else 
        # G_M = (1/2 eta)*( eta erf'(eta) + (1 + 2 eta^2) erf(eta))
        G = (1.0/sqrt(pi))*exp(-eta^2) + ((0.5/eta) + eta)*erf(eta)
    end
    return G*dens*vth
end

# H (defined by Del^2 H = -(4/sqrt(pi))*F 
# with F = cref^3 pi^(3/2) F_Maxwellian / nref 
# the normalised Maxwellian
function H_Maxwellian(dens,upar,vth,vpa,vperp,ivpa,ivperp)
    # speed variable
    eta = eta_func(upar,vth,vpa,vperp,ivpa,ivperp)
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

# 1D derivative functions

function dGdeta(eta::mk_float)
    # d \tilde{G} / d eta
    dGdeta_fac = (1.0/sqrt(pi))*exp(-eta^2)/eta + (1.0 - 0.5/(eta^2))*erf(eta)
    return dGdeta_fac
end

function d2Gdeta2(eta::mk_float)
    # d \tilde{G} / d eta
    d2Gdeta2_fac = erf(eta)/(eta^3) - (2.0/sqrt(pi))*exp(-eta^2)/(eta^2)
    return d2Gdeta2_fac
end

function ddGddeta(eta::mk_float)
    # d / d eta ( (1/ eta) d \tilde{G} d eta 
    ddGddeta_fac = (1.5/(eta^2) - 1.0)*erf(eta)/(eta^2) - (3.0/sqrt(pi))*exp(-eta^2)/(eta^3)
    return ddGddeta_fac
end

function dHdeta(eta::mk_float)
    dHdeta_fac = (2.0/sqrt(pi))*(exp(-eta^2))/eta - erf(eta)/(eta^2)
    return dHdeta_fac
end

# functions of vpa & vperp 
function eta_func(upar::mk_float,vth::mk_float,
             vpa,vperp,ivpa,ivperp)
    speed = sqrt( (vpa.grid[ivpa] - upar)^2 + vperp.grid[ivperp]^2)/vth
    return speed
end

function d2Gdvpa2(dens::mk_float,upar::mk_float,vth::mk_float,
                            vpa,vperp,ivpa,ivperp)
    eta = eta_func(upar,vth,vpa,vperp,ivpa,ivperp)
    fac = dGdeta(eta) + ddGddeta(eta)*((vpa.grid[ivpa] - upar)^2)/(vth^2)
    d2Gdvpa2_fac = fac*dens/(eta*vth)
    return d2Gdvpa2_fac
end

function d2Gdvperpdvpa(dens::mk_float,upar::mk_float,vth::mk_float,
                            vpa,vperp,ivpa,ivperp)
    eta = eta_func(upar,vth,vpa,vperp,ivpa,ivperp)
    fac = ddGddeta(eta)*vperp.grid[ivperp]*(vpa.grid[ivpa] - upar)/(vth^2)
    d2Gdvperpdvpa_fac = fac*dens/(eta*vth)
    return d2Gdvperpdvpa_fac
end

function d2Gdvperp2(dens::mk_float,upar::mk_float,vth::mk_float,
                            vpa,vperp,ivpa,ivperp)
    eta = eta_func(upar,vth,vpa,vperp,ivpa,ivperp)
    fac = dGdeta(eta) + ddGddeta(eta)*(vperp.grid[ivperp]^2)/(vth^2)
    d2Gdvperp2_fac = fac*dens/(eta*vth)
    return d2Gdvperp2_fac
end

function dGdvperp(dens::mk_float,upar::mk_float,vth::mk_float,
                            vpa,vperp,ivpa,ivperp)
    eta = eta_func(upar,vth,vpa,vperp,ivpa,ivperp)
    fac = dGdeta(eta)*vperp.grid[ivperp]*dens/(vth*eta)
    return fac 
end

function dHdvperp(dens::mk_float,upar::mk_float,vth::mk_float,
                            vpa,vperp,ivpa,ivperp)
    eta = eta_func(upar,vth,vpa,vperp,ivpa,ivperp)
    fac = dHdeta(eta)*vperp.grid[ivperp]*dens/(eta*vth^3)
    return fac 
end

function dHdvpa(dens::mk_float,upar::mk_float,vth::mk_float,
                            vpa,vperp,ivpa,ivperp)
    eta = eta_func(upar,vth,vpa,vperp,ivpa,ivperp)
    fac = dHdeta(eta)*(vpa.grid[ivpa]-upar)*dens/(eta*vth^3)
    return fac 
end

function F_Maxwellian(dens::mk_float,upar::mk_float,vth::mk_float,
                        vpa,vperp,ivpa,ivperp)
    eta = eta_func(upar,vth,vpa,vperp,ivpa,ivperp)
    fac = (dens/(vth^3))*exp(-eta^2)
    return fac
end

function dFdvpa_Maxwellian(dens::mk_float,upar::mk_float,vth::mk_float,
                        vpa,vperp,ivpa,ivperp)
    eta = eta_func(upar,vth,vpa,vperp,ivpa,ivperp)
    fac = -2.0*(dens/(vth^4))*((vpa.grid[ivpa] - upar)/vth)*exp(-eta^2)
    return fac
end

function dFdvperp_Maxwellian(dens::mk_float,upar::mk_float,vth::mk_float,
                        vpa,vperp,ivpa,ivperp)
    eta = eta_func(upar,vth,vpa,vperp,ivpa,ivperp)
    fac = -2.0*(dens/(vth^4))*(vperp.grid[ivperp]/vth)*exp(-eta^2)
    return fac
end

function d2Fdvperpdvpa_Maxwellian(dens::mk_float,upar::mk_float,vth::mk_float,
                        vpa,vperp,ivpa,ivperp)
    eta = eta_func(upar,vth,vpa,vperp,ivpa,ivperp)
    fac = 4.0*(dens/(vth^5))*(vperp.grid[ivperp]/vth)*((vpa.grid[ivpa] - upar)/vth)*exp(-eta^2)
    return fac
end

function d2Fdvpa2_Maxwellian(dens::mk_float,upar::mk_float,vth::mk_float,
                        vpa,vperp,ivpa,ivperp)
    eta = eta_func(upar,vth,vpa,vperp,ivpa,ivperp)
    fac = 4.0*(dens/(vth^5))*( ((vpa.grid[ivpa] - upar)/vth)^2 - 0.5 )*exp(-eta^2)
    return fac
end

function d2Fdvperp2_Maxwellian(dens::mk_float,upar::mk_float,vth::mk_float,
                        vpa,vperp,ivpa,ivperp)
    eta = eta_func(upar,vth,vpa,vperp,ivpa,ivperp)
    fac = 4.0*(dens/(vth^5))*((vperp.grid[ivperp]/vth)^2 - 0.5)*exp(-eta^2)
    return fac
end

function Cssp_Maxwellian_inputs(denss::mk_float,upars::mk_float,vths::mk_float,ms::mk_float,
                                denssp::mk_float,uparsp::mk_float,vthsp::mk_float,msp::mk_float,
                                nussp::mk_float,vpa,vperp,ivpa,ivperp)
    
    d2Fsdvpa2 = d2Fdvpa2_Maxwellian(denss,upars,vths,vpa,vperp,ivpa,ivperp)
    d2Fsdvperp2 = d2Fdvperp2_Maxwellian(denss,upars,vths,vpa,vperp,ivpa,ivperp)
    d2Fsdvperpdvpa = d2Fdvperpdvpa_Maxwellian(denss,upars,vths,vpa,vperp,ivpa,ivperp)
    dFsdvperp = dFdvperp_Maxwellian(denss,upars,vths,vpa,vperp,ivpa,ivperp)
    dFsdvpa = dFdvpa_Maxwellian(denss,upars,vths,vpa,vperp,ivpa,ivperp)
    Fs = F_Maxwellian(denss,upars,vths,vpa,vperp,ivpa,ivperp)
    
    d2Gspdvpa2 = d2Gdvpa2(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
    d2Gspdvperp2 = d2Gdvperp2(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
    d2Gspdvperpdvpa = d2Gdvperpdvpa(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
    dGspdvperp = dGdvperp(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
    dHspdvperp = dHdvperp(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
    dHspdvpa = dHdvpa(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
    Fsp = F_Maxwellian(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
    
    ( Cssp_Maxwellian = 
        d2Fsdvpa2*d2Gspdvpa2 + 
        d2Fsdvperp2*d2Gspdvperp2 + 
        2.0*d2Fsdvperpdvpa*d2Gspdvperpdvpa + 
        (1.0/(vperp.grid[ivperp]^2))*dFsdvperp*dGspdvperp +
        2.0*(1.0 - (ms/msp))*(dFsdvpa*dHspdvpa + dFsdvperp*dHspdvperp) +
        (8.0/sqrt(pi))*(ms/msp)*Fs*Fsp ) 
        
    Cssp_Maxwellian *= nussp
    return Cssp_Maxwellian
end

function Cflux_vpa_Maxwellian_inputs(ms::mk_float,denss::mk_float,upars::mk_float,vths::mk_float,
                                     msp::mk_float,denssp::mk_float,uparsp::mk_float,vthsp::mk_float,
                                     vpa,vperp,ivpa,ivperp)
    etap = eta_func(uparsp,vthsp,vpa,vperp,ivpa,ivperp)
    eta = eta_func(upars,vths,vpa,vperp,ivpa,ivperp)
    prefac = -2.0*denss*denssp*exp( -eta^2)/(vthsp*vths^5)
    (fac = (vpa.grid[ivpa]-uparsp)*(d2Gdeta2(etap) + (ms/msp)*((vths/vthsp)^2)*dHdeta(etap)/etap)
             + (uparsp - upars)*( dGdeta(etap) + ((vpa.grid[ivpa]-uparsp)^2/vthsp^2)*ddGddeta(etap) )/etap )
    Cflux = prefac*fac
    #fac *= (ms/msp)*(vths/vthsp)*dHdeta(etap)/etap
    #fac *= d2Gdeta2(etap) 
    return Cflux
end

function Cflux_vperp_Maxwellian_inputs(ms::mk_float,denss::mk_float,upars::mk_float,vths::mk_float,
                                     msp::mk_float,denssp::mk_float,uparsp::mk_float,vthsp::mk_float,
                                     vpa,vperp,ivpa,ivperp)
    etap = eta_func(uparsp,vthsp,vpa,vperp,ivpa,ivperp)
    eta = eta_func(upars,vths,vpa,vperp,ivpa,ivperp)
    prefac = -2.0*(vperp.grid[ivperp])*denss*denssp*exp( -eta^2)/(vthsp*vths^5)
    (fac = (d2Gdeta2(etap) + (ms/msp)*((vths/vthsp)^2)*dHdeta(etap)/etap)
             + ((uparsp - upars)*(vpa.grid[ivpa]-uparsp)/vthsp^2)*ddGddeta(etap)/etap )
    Cflux = prefac*fac
    #fac *= (ms/msp)*(vths/vthsp)*dHdeta(etap)/etap
    #fac *= d2Gdeta2(etap) 
    return Cflux
end

# solves A x = b for a matrix of the form
# A00  0    A02
# 0    A11  A12
# A02  A12  A22
# appropriate for the moment numerical conserving terms
function symmetric_matrix_inverse(A00,A02,A11,A12,A22,b0,b1,b2)
    # matrix determinant
    detA = A00*(A11*A22 - A12^2) - A11*A02^2
    # cofactors C (also a symmetric matrix)
    C00 = A11*A22 - A12^2
    C01 = A12*A02
    C02 = -A11*A02
    C11 = A00*A22 - A02^2
    C12 = -A00*A12
    C22 = A00*A11
    x0 = ( C00*b0 + C01*b1 + C02*b2 )/detA
    x1 = ( C01*b0 + C11*b1 + C12*b2 )/detA
    x2 = ( C02*b0 + C12*b1 + C22*b2 )/detA
    #println("b0: ",b0," b1: ",b1," b2: ",b2)
    #println("A00: ",A00," A02: ",A02," A11: ",A11," A12: ",A12," A22: ",A22, " detA: ",detA)
    #println("C00: ",C00," C02: ",C02," C11: ",C11," C12: ",C12," C22: ",C22)
    #println("x0: ",x0," x1: ",x1," x2: ",x2)
    return x0, x1, x2
end

# applies the numerical conservation to pdf_out, the advanced distribution function
# uses the low-level moment integration routines from velocity moments
# conserves n, upar, total pressure of each species
# only correct for the self collision operator
# multi-species cases requires conservation of  particle number and total momentum and total energy ( sum_s m_s upar_s, ... )
function apply_numerical_conserving_terms!(pdf_out,pdf_in,boundary_distributions,
      vpa, vperp, z, r, vpa_advect, z_advect, r_advect, composition,
      scratch_dummy, advance, vperp_spectral, vpa_spectral)
    # enforce bc prior to imposing conservation
    enforce_boundary_conditions!(pdf_out, boundary_distributions.pdf_rboundary_charged,
      vpa.bc, z.bc, r.bc, vpa, vperp, z, r, vpa_advect, z_advect, r_advect, composition,
      scratch_dummy, advance, vperp_spectral, vpa_spectral)
    # buffer arrays
    buffer_pdf = scratch_dummy.buffer_vpavperpzrs_1
    dummy_vpavperp = scratch_dummy.dummy_vpavperp
    # data precalculated by store_moments_in_buffer!
    buffer_n = scratch_dummy.buffer_zrs_1
    buffer_upar = scratch_dummy.buffer_zrs_2
    buffer_pressure = scratch_dummy.buffer_zrs_3
    mass = 1.0
    begin_s_r_z_region()
    @loop_s_r_z is ir iz begin
        # get moments of incoming and outgoing distribution functions
        n_in = buffer_n[iz,ir,is]
        upar_in = buffer_upar[iz,ir,is]
        pressure_in = buffer_pressure[iz,ir,is]
        
        n_out = get_density(@view(pdf_out[:,:,iz,ir,is]), vpa, vperp)
        upar_out = get_upar(@view(pdf_out[:,:,iz,ir,is]), vpa, vperp, n_out)
        ppar_out = get_ppar(@view(pdf_out[:,:,iz,ir,is]), vpa, vperp, upar_out, mass)
        pperp_out = get_pperp(@view(pdf_out[:,:,iz,ir,is]), vpa, vperp, mass)
        pressure_out = get_pressure(ppar_out,pperp_out)
        qpar_out = get_qpar(@view(pdf_out[:,:,iz,ir,is]), vpa, vperp, upar_out, mass, dummy_vpavperp)
        rmom_out = get_rmom(@view(pdf_out[:,:,iz,ir,is]), vpa, vperp, upar_out, mass, dummy_vpavperp)
        
        # form the appropriate matrix coefficients
        b0, b1, b2 = n_in, n_in*(upar_in - upar_out), (3.0/2.0)*(pressure_in/mass) + n_in*(upar_in - upar_out)^2
        A00, A02, A11, A12, A22 = n_out, (3.0/2.0)*(pressure_out/mass), 0.5*ppar_out/mass, qpar_out/mass, rmom_out/mass
        
        # obtain the coefficients for the corrections 
        (x0, x1, x2) = symmetric_matrix_inverse(A00,A02,A11,A12,A22,b0,b1,b2)
        
        # fill the buffer with the corrected pdf 
        @loop_vperp_vpa ivperp ivpa begin
            wpar = vpa.grid[ivpa] - upar_out
            buffer_pdf[ivpa,ivperp,iz,ir,is] = (x0 + x1*wpar + x2*(vperp.grid[ivperp]^2 + wpar^2) )*pdf_out[ivpa,ivperp,iz,ir,is]
        end
        
    end
    begin_s_r_z_vperp_vpa_region()
    # update pdf_out
    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        pdf_out[ivpa,ivperp,iz,ir,is] = buffer_pdf[ivpa,ivperp,iz,ir,is]
    end
    return nothing
end

# function which corrects only for the loss of particles due to numerical error
# suitable for use with multiple species collisions
function apply_density_conserving_terms!(pdf_out,pdf_in,boundary_distributions,
      vpa, vperp, z, r, vpa_advect, z_advect, r_advect, composition,
      scratch_dummy, advance, vperp_spectral, vpa_spectral)
    # enforce bc prior to imposing conservation
    enforce_boundary_conditions!(pdf_out, boundary_distributions.pdf_rboundary_charged,
      vpa.bc, z.bc, r.bc, vpa, vperp, z, r, vpa_advect, z_advect, r_advect, composition,
      scratch_dummy, advance, vperp_spectral, vpa_spectral)
    # buffer array
    buffer_pdf = scratch_dummy.buffer_vpavperpzrs_1
    # data precalculated by store_moments_in_buffer!
    buffer_n = scratch_dummy.buffer_zrs_1
    begin_s_r_z_region()
    @loop_s_r_z is ir iz begin
        # get density moment of incoming and outgoing distribution functions
        n_in = buffer_n[iz,ir,is]
        
        n_out = get_density(@view(pdf_out[:,:,iz,ir,is]), vpa, vperp)
        
        # obtain the coefficient for the corrections 
        x0 = n_in/n_out
        
        # update pdf_out with the corrections 
        @loop_vperp_vpa ivperp ivpa begin
            buffer_pdf[ivpa,ivperp,iz,ir,is] = x0*pdf_out[ivpa,ivperp,iz,ir,is]
        end
        
    end
    begin_s_r_z_vperp_vpa_region()
    # update pdf_out
    @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
        pdf_out[ivpa,ivperp,iz,ir,is] = buffer_pdf[ivpa,ivperp,iz,ir,is]
    end
    return nothing
end

function store_moments_in_buffer!(pdf_out,boundary_distributions,
      vpa, vperp, z, r, vpa_advect, z_advect, r_advect, composition,
      scratch_dummy, advance, vperp_spectral, vpa_spectral)
    # enforce bc prior to calculating the moments
    enforce_boundary_conditions!(pdf_out, boundary_distributions.pdf_rboundary_charged,
      vpa.bc, z.bc, r.bc, vpa, vperp, z, r, vpa_advect, z_advect, r_advect, composition,
      scratch_dummy, advance, vperp_spectral, vpa_spectral)
    # buffer arrays
    density = scratch_dummy.buffer_zrs_1
    upar = scratch_dummy.buffer_zrs_2
    pressure = scratch_dummy.buffer_zrs_3
    # set the mass
    mass = 1.0
    
    begin_s_r_z_region()
    @loop_s_r_z is ir iz begin
        density[iz,ir,is] = get_density(@view(pdf_out[:,:,iz,ir,is]), vpa, vperp)
        upar[iz,ir,is] = get_upar(@view(pdf_out[:,:,iz,ir,is]), vpa, vperp, density[iz,ir,is])
        ppar = get_ppar(@view(pdf_out[:,:,iz,ir,is]), vpa, vperp, upar[iz,ir,is], mass)
        pperp = get_pperp(@view(pdf_out[:,:,iz,ir,is]), vpa, vperp, mass)
        pressure[iz,ir,is] = get_pressure(ppar,pperp)
    end
    return nothing
end
end
