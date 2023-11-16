"""
module for including functions used 
in testing the implementation of the 
the Full-F Fokker-Planck Collision Operator
"""
module fokker_planck_test

export Cflux_vpa_Maxwellian_inputs, Cflux_vperp_Maxwellian_inputs
export d2Gdvpa2_Maxwellian, dGdvperp_Maxwellian, d2Gdvperpdvpa_Maxwellian, d2Gdvperp2_Maxwellian
export dHdvpa_Maxwellian, dHdvperp_Maxwellian, Cssp_Maxwellian_inputs
export F_Maxwellian, dFdvpa_Maxwellian, dFdvperp_Maxwellian
export d2Fdvpa2_Maxwellian, d2Fdvperpdvpa_Maxwellian, d2Fdvperp2_Maxwellian
export H_Maxwellian, G_Maxwellian

export Cssp_fully_expanded_form, calculate_collisional_fluxes

export print_test_data, plot_test_data, fkpl_error_data, allocate_error_data

using Plots
using LaTeXStrings
using Measures
using ..type_definitions: mk_float, mk_int
using SpecialFunctions: erf
using ..velocity_moments: get_density
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

function d2Gdvpa2_Maxwellian(dens::mk_float,upar::mk_float,vth::mk_float,
                            vpa,vperp,ivpa,ivperp)
    eta = eta_func(upar,vth,vpa,vperp,ivpa,ivperp)
    fac = dGdeta(eta) + ddGddeta(eta)*((vpa.grid[ivpa] - upar)^2)/(vth^2)
    d2Gdvpa2_fac = fac*dens/(eta*vth)
    return d2Gdvpa2_fac
end

function d2Gdvperpdvpa_Maxwellian(dens::mk_float,upar::mk_float,vth::mk_float,
                            vpa,vperp,ivpa,ivperp)
    eta = eta_func(upar,vth,vpa,vperp,ivpa,ivperp)
    fac = ddGddeta(eta)*vperp.grid[ivperp]*(vpa.grid[ivpa] - upar)/(vth^2)
    d2Gdvperpdvpa_fac = fac*dens/(eta*vth)
    return d2Gdvperpdvpa_fac
end

function d2Gdvperp2_Maxwellian(dens::mk_float,upar::mk_float,vth::mk_float,
                            vpa,vperp,ivpa,ivperp)
    eta = eta_func(upar,vth,vpa,vperp,ivpa,ivperp)
    fac = dGdeta(eta) + ddGddeta(eta)*(vperp.grid[ivperp]^2)/(vth^2)
    d2Gdvperp2_fac = fac*dens/(eta*vth)
    return d2Gdvperp2_fac
end

function dGdvperp_Maxwellian(dens::mk_float,upar::mk_float,vth::mk_float,
                            vpa,vperp,ivpa,ivperp)
    eta = eta_func(upar,vth,vpa,vperp,ivpa,ivperp)
    fac = dGdeta(eta)*vperp.grid[ivperp]*dens/(vth*eta)
    return fac 
end

function dHdvperp_Maxwellian(dens::mk_float,upar::mk_float,vth::mk_float,
                            vpa,vperp,ivpa,ivperp)
    eta = eta_func(upar,vth,vpa,vperp,ivpa,ivperp)
    fac = dHdeta(eta)*vperp.grid[ivperp]*dens/(eta*vth^3)
    return fac 
end

function dHdvpa_Maxwellian(dens::mk_float,upar::mk_float,vth::mk_float,
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
    
    d2Gspdvpa2 = d2Gdvpa2_Maxwellian(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
    d2Gspdvperp2 = d2Gdvperp2_Maxwellian(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
    d2Gspdvperpdvpa = d2Gdvperpdvpa_Maxwellian(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
    dGspdvperp = dGdvperp_Maxwellian(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
    dHspdvperp = dHdvperp_Maxwellian(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
    dHspdvpa = dHdvpa_Maxwellian(denssp,uparsp,vthsp,vpa,vperp,ivpa,ivperp)
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
Below are functions which are used for storing and printing data from the tests 
"""

function plot_test_data(func_exact,func_num,func_err,func_name,vpa,vperp)
    @views heatmap(vperp.grid, vpa.grid, func_num[:,:], ylabel=L"v_{\|\|}", xlabel=L"v_{\perp}", c = :deep, interpolation = :cubic,
                windowsize = (360,240), margin = 15pt)
                outfile = string(func_name*"_num.pdf")
                savefig(outfile)
    @views heatmap(vperp.grid, vpa.grid, func_exact[:,:], ylabel=L"v_{\|\|}", xlabel=L"v_{\perp}", c = :deep, interpolation = :cubic,
                windowsize = (360,240), margin = 15pt)
                outfile = string(func_name*"_exact.pdf")
                savefig(outfile)
    @views heatmap(vperp.grid, vpa.grid, func_err[:,:], ylabel=L"v_{\|\|}", xlabel=L"v_{\perp}", c = :deep, interpolation = :cubic,
                windowsize = (360,240), margin = 15pt)
                outfile = string(func_name*"_err.pdf")
                savefig(outfile)
    return nothing
end

function print_test_data(func_exact,func_num,func_err,func_name)
    @. func_err = abs(func_num - func_exact)
    max_err = maximum(func_err)
    println("maximum("*func_name*"_err): ",max_err)
    return max_err
end

function print_test_data(func_exact,func_num,func_err,func_name,vpa,vperp,dummy;print_to_screen=true)
    @. func_err = abs(func_num - func_exact)
    max_err = maximum(func_err)
    @. dummy = func_err^2
    # compute the numerator
    num = get_density(dummy,vpa,vperp)
    # compute the denominator
    @. dummy = 1.0
    denom = get_density(dummy,vpa,vperp)
    L2norm = sqrt(num/denom)
    if print_to_screen 
        println("maximum("*func_name*"_err): ",max_err," L2("*func_name*"_err): ",L2norm)
    end
    return max_err, L2norm
end

mutable struct error_data
    max::mk_float
    L2::mk_float
end

mutable struct moments_error_data
    delta_density::mk_float
    delta_upar::mk_float
    delta_pressure::mk_float
end

struct fkpl_error_data
    C_M::error_data
    H_M::error_data
    dHdvpa_M::error_data
    dHdvperp_M::error_data
    G_M::error_data
    dGdvperp_M::error_data
    d2Gdvpa2_M::error_data
    d2Gdvperpdvpa_M::error_data
    d2Gdvperp2_M::error_data
    moments::moments_error_data
end

function allocate_error_data()
    C_M = error_data(0.0,0.0)
    H_M = error_data(0.0,0.0)
    dHdvpa_M = error_data(0.0,0.0)
    dHdvperp_M = error_data(0.0,0.0)
    G_M = error_data(0.0,0.0)
    dGdvperp_M = error_data(0.0,0.0)
    d2Gdvpa2_M = error_data(0.0,0.0)
    d2Gdvperpdvpa_M = error_data(0.0,0.0)
    d2Gdvperp2_M = error_data(0.0,0.0)
    moments = moments_error_data(0.0,0.0,0.0)
    return fkpl_error_data(C_M,H_M,dHdvpa_M,dHdvperp_M,
        G_M,dGdvperp_M,d2Gdvpa2_M,d2Gdvperpdvpa_M,d2Gdvperp2_M,
        moments)
end

end
