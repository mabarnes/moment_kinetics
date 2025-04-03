"""
"""
module bgk

export init_bgk_pdf!

using SpecialFunctions: erfi
using Roots: find_zero
using ..array_allocation: allocate_float, allocate_int
using ..looping
using ..quadrature: composite_simpson_weights
using ..calculus: integral
using ..coordinates: equally_spaced_grid

"""
"""
function init_bgk_pdf!(pdf, phi_max, tau, z, Lz, vpa)
    # For simplicity, just run in serial for now
    @serial_region begin
        nz = length(z)
        dum, wgts, integrand = setup_dummy_integrals()
        # find the allowed wave amplitude (phi_max - phi_min),
        # determined by considering the limiting case where vpa=0 so x = e*phi/T.
        # the location where ftrap(x=e*phi/T) = 0 corresponds to the minimum
        # value of phi (and thus x) before which f goes negative.
        dphi = allowed_wave_amplitude!(phi_max, tau, dum, wgts, integrand)
        # add a buffer in to dphi to avoid problems with slightly negative numbers
        dphi *= 0.9
        # define phi(z) to have max value phi_max and min value phi_max - dphi
        phi = allocate_float(nz)
    #    @. phi = dphi*exp(-200.0*(2.0*z/Lz)^2) + (phi_max-dphi)
    #    @. phi = -dphi*exp(-200.0*(2.0*z/Lz)^2) + phi_max
    #    @. phi = -(1.0+sinpi(2.0*z/Lz))*0.5*dphi + phi_max
        @. phi = -(1.0+cospi(2.0*z/Lz))*0.5*dphi + phi_max
    #    phi[1] = phi_max
    #    phi[end] = phi_max
#=
        for iz ∈ 1:nz
            println("z: ", z[iz], "  phi: ", phi[iz])
        end
=#
        # construct a grid in total parallel energy that contains all
        # (z,vpa) points from the code grid
        x = total_energy_grid(vpa, phi)
        # get the max vpa index still in the trapped region for each z grid location
        ivpa_min = trapped_passing_boundary(x, phi_max)
        # fill in the pdf for the trapped part of phase space
        trapped_pdf!(pdf, phi_max, tau, x, dum, wgts, integrand, ivpa_min)
        # fill in the pdf for the passing part of phase space
        passing_pdf!(pdf, phi_max, tau, x, ivpa_min)
#=
        for ivpa ∈ 1:length(vpa)
            for iz ∈ 1:nz
                println("x: ", x[iz,ivpa], "  z: ", z[iz], "  vpa: ", vpa[ivpa], "  f: ", pdf[iz,ivpa], " ivpa_min: ", ivpa_min[iz])
            end
        end
=#
    end
end

"""
# inputs
- pdf is the particle distribution function, with the passing part of phase space not filled in
- phi_max is the maximum value that e * phi / Te takes
- tau = Ti/Te is the ion-electron temperature ratio
- x = m*vpa^2/2Te + e*phi/Te is 1D array containing the total parallel energy (conserved)

# output
- pdf = particle distribution function; this function fills in the part of phase space
  where x > e*phi_max/T
"""
function passing_pdf!(pdf, phi_max, tau, x, ivpa_min)
    nvpa = size(x,1)
    for iz ∈ 1:size(x,2)
        if ivpa_min[iz] > 1
            for ivpa ∈ 1:ivpa_min[iz]-1
                pdf[ivpa,iz] = exp(((1.0+tau)*phi_max - x[ivpa,iz])/tau)/sqrt(pi*tau)
                if pdf[ivpa,iz] < 0.0
                    println("warning: pdf obtained in bgk init is negative: pdf[", iz, ",", ivpa, "] = ", pdf[iz,ivpa], " is negative. setting pdf there to zero.")
                    pdf[ivpa,iz] = 0.0
                end
            end
            for ivpa ∈ nvpa:-1:nvpa-ivpa_min[iz]+2
                pdf[iz,ivpa] = exp(((1.0+tau)*phi_max - x[ivpa,iz])/tau)/sqrt(pi*tau)
                if pdf[iz,ivpa] < 0.0
                    println("warning: pdf obtained in bgk init is negative: pdf[", ivpa, ",", iz, "] = ", pdf[ivpa,iz], " is negative. setting pdf there to zero.")
                    pdf[ivpa,iz] = 0.0
                end
            end
        end
    end
    return nothing
end

"""
"""
function allowed_wave_amplitude!(phi_max, tau, y, wgts, integrand)
    n = length(y)
    # trapped_pdf_single evaluates the trapped pdf at the given total parallel energy value, x0
    function trapped_pdf_single(x0)
        # construct the integrand
        for i ∈ 1:n
            integrand[i] = asin((phi_max-x0-y[i])/(phi_max+y[i]-x0))*exp((tau*phi_max-y[i])/tau)
        end
        # carry out the semi-infinite integral in y
        total = integral(integrand, wgts)
        # calculate the trapped pdf
        return exp(phi_max)/(2.0*sqrt(pi*tau)) - total/(pi*tau)^1.5 - exp(x0)*erfi(sqrt(phi_max-x0))/sqrt(pi)
    end
    # look for a zero of ftrap using function trapped_pdf_single, with initial guess x = 0
    phi_min = find_zero(trapped_pdf_single, -0.1)
    return phi_max - phi_min
end

"""
# inputs
- phi_max is the maximum value that e * phi / Te takes
- tau = Ti/Te is the ion-electron temperature ratio
- x = vpa^2 + e*phi is a 2D array containing the total parallel energy on the (z,vpa) grid
- y = dummy coordinate for the necessary integrals in the function
- integrand = dummy array used to hold integrands defined and integrated in this function
- wgts = integration weights associated with y integrals

# output
- pdf is the particle distribution function for all of phase space, with this function
  filling in only the part with x < e*phi_max/T
"""
function trapped_pdf!(pdf, phi_max, tau, x, y, wgts, integrand, ivpa_min)
    nvpa = size(x,1)
    # trapped_pdf_single evaluates the trapped pdf at the given total parallel energy value, x0
    function trapped_pdf_single(x0)
        # construct the integrand
        for i ∈ 1:length(integrand)
            if abs(phi_max - x0) < 1.0e-12
                arg = -1.0
            else
                arg = (phi_max-x0-y[i])/(phi_max+y[i]-x0)
            end
            if arg < -1.0
                println("warning: argument of asin in bgk init is ", arg, " at x = ", x0, ". setting arg = -1.0.")
                arg = -1.0
            elseif arg > 1.0
                println("warning: argument of asin in bgk init is ", arg, " at x = ", x0, ". setting arg = 1.0.")
                arg = 1.0
            end
            #integrand[i] = asin((phi_max-x0-y[i])/(phi_max+y[i]-x0))*exp((tau*phi_max-y[i])/tau)
            integrand[i] = asin(arg)*exp((tau*phi_max-y[i])/tau)
        end
        # carry out the semi-infinite integral in y
        total = integral(integrand, wgts)
        # calculate the trapped pdf
        return exp(phi_max)/(2.0*sqrt(pi*tau)) - total/(pi*tau)^1.5 - exp(x0)*erfi(sqrt(phi_max-x0))/sqrt(pi)
    end
    for iz ∈ 1:size(x,2)
        if ivpa_min[iz] <= nvpa
            ivpa_max = nvpa-ivpa_min[iz]+1
            for ivpa ∈ ivpa_min[iz]:ivpa_max
                pdf[ivpa,iz] = trapped_pdf_single(x[ivpa,iz])
                if pdf[ivpa,iz] < 0.0
                    println("warning: pdf obtained in bgk init is negative: pdf[", iz, ",", ivpa, "] = ", pdf[iz,ivpa], " is negative. setting pdf there to zero.")
                    pdf[ivpa,iz] = 0.0
                end
            end
        end
    end
end

"""
# inputs:
- vpa = parallel velocity normalized by vts = sqrt(2*Te/ms)
- phi = electrostatic potential normalized by Te/e
# output: x = vpa^2 + phi is the total parallel energy
"""
function total_energy_grid(vpa, phi)
    nvpa = length(vpa)
    nz = length(phi)
    x = allocate_float(nvpa, nz)
    for iz ∈ 1:nz
        for ivpa ∈ 1:nvpa
            x[ivpa,iz] = vpa[ivpa]^2 + phi[iz]
        end
    end
    return x
end

"""
"""
function trapped_passing_boundary(x, phi_max)
    nvpa = size(x,1)
    # initialize the lower boundary for the trapped domain to be beyond the boundary of vpa
    ivpa_min = allocate_int(nvpa)
    @. ivpa_min = nvpa+1
    for iz ∈ 1:size(x,2)
        # start from most negative vpa and look for first vpa value where x < phi_max
        for ivpa ∈ 1:nvpa
            if x[ivpa,iz] <= phi_max
                ivpa_min[iz] = ivpa
                break
            end
        end
    end
    return ivpa_min
end

"""
"""
function setup_dummy_integrals()
    # construct a simple, equally-spaced grid of n points for the integration
    # on a semi-infinity domain (0,∞) over the dummy variable y
    n = 10000
    ymax = 100.0
    y = equally_spaced_grid(n, ymax, 1, 1, 0)
    # shift y so that it starts at zero instead of -ymax/2
    @. y += 0.5*ymax
    # assign integration weights to the y grid points using composite Simpson's rule
    wgts = composite_simpson_weights(y)
    integrand = allocate_float(n)
    return y, wgts, integrand
end

end
