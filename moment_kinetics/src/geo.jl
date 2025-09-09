"""
module for including axisymmetric geometry
in coordinates (z,r), with z the vertical 
coordinate and r the radial coordinate
"""
module geo

export init_magnetic_geometry
export setup_geometry_input

using ..input_structs: geometry_input, set_defaults_and_check_section!
using ..file_io: input_option_error
using ..array_allocation: allocate_float
using ..type_definitions: mk_float, mk_int
using ..reference_parameters: setup_reference_parameters

using OrderedCollections: OrderedDict

"""
struct containing the geometric data necessary for 
non-trivial axisymmetric geometries, to be passed 
around the inside of the code, replacing the 
`geometry_input` struct from input_structs.jl

The arrays of 2 dimensions are functions of (z,r)
"""
struct geometric_coefficients
# for now include the reference parameters in `geometry_input`
input::geometry_input
# also include a copy of rhostar for ease of use
rhostar::mk_float
# the spatially varying coefficients
# Bz/Bref
Bzed::Array{mk_float,2}
# Bzeta/Bref
Bzeta::Array{mk_float,2}
# Btot/Bref
Bmag::Array{mk_float,2}
# bz -- unit vector component in z direction
bzed::Array{mk_float,2}
# bz -- unit vector component in zeta direction
bzeta::Array{mk_float,2}


# now the new coefficients

# d Bmag d z
dBdz::Array{mk_float,2}
# d Bmag d r
dBdr::Array{mk_float,2}
# jacobian =  r grad r x grad z . grad zeta
jacobian::Array{mk_float,2}

# magnetic drift physics coefficients
# cvdriftr = (b/B) x (b.grad b) . grad r
cvdriftr::Array{mk_float,2}
# cvdriftz = (b/B) x (b.grad b) . grad z
cvdriftz::Array{mk_float,2}
# gbdriftr = (b/B^2) x grad B . grad r
gbdriftr::Array{mk_float,2}
# gbdriftz = (b/B^2) x grad B . grad z
gbdriftz::Array{mk_float,2}
end

"""
    function get_default_rhostar(reference_params)

Calculate
\$c_\\mathrm{ref} / \\Omega_\\mathrm{ref} L_\\mathrm{ref} = m_i c_\\mathrm{ref} / e B_\\mathrm{ref} L_\\mathrm{ref}\$.

This is similar to the de-dimensionalised ion gyroradius at reference parameters, which
would be sqrt(2)*rhostar, as \$v_{Ti}(T_\\mathrm{ref}) = \\sqrt{2 T_\\mathrm{ref} / m_i} = \\sqrt{2} c_\\mathrm{ref}\$.
"""
function get_default_rhostar(reference_params)
    return reference_params.cref / reference_params.Omegaref / reference_params.Lref
end

"""
function to read the geometry input data from the TOML file

the TOML namelist should be structured like

[geometry]
pitch = 1.0
rhostar = 1.0
DeltaB = 0.0
option = ""

"""
function setup_geometry_input(toml_input::AbstractDict, warn_unexpected::Bool)

    reference_params = setup_reference_parameters(toml_input, warn_unexpected)
    reference_rhostar = get_default_rhostar(reference_params)
    # read the input toml and specify a sensible default
    input_section = set_defaults_and_check_section!(
        toml_input, "geometry", warn_unexpected;
        # begin default inputs (as kwargs)
        # rhostar ion (ref)
        rhostar = reference_rhostar, #used to premultiply ExB drift terms
        # magnetic geometry option
        option = "constant-helical",# "1D-mirror"
        # pitch ( = Bzed/Bmag if geometry_option == "constant-helical")
        pitch = 1.0,
        # DeltaB ( = (Bzed(z=L/2) - Bzed(0))/Bref if geometry_option == "1D-mirror")
        DeltaB = 0.0,
        # constant for testing nonzero Er when nr = 1
        Er_constant = 0.0,
        # constant for testing nonzero Ez when nz = 1
        Ez_constant = 0.0,
        # constant for testing nonzero dBdz when nz = 1
        dBdz_constant = 0.0,
        # constant for testing nonzero dBdr when nr = 1
        dBdr_constant = 0.0)
    
    input = OrderedDict(Symbol(k)=>v for (k,v) in input_section)
    #println(input)
    return geometry_input(; input...)
end

"""
function to initialise the geometry coefficients
input_data -- geometry_input type
z -- coordinate type
r -- coordinate type
"""
function init_magnetic_geometry(geometry_input_data::geometry_input,z,r)
    nz = z.n
    nr = r.n
    Bzed = allocate_float(z, r)
    Bzeta = allocate_float(z, r)
    Bmag = allocate_float(z, r)
    bzed = allocate_float(z, r)
    bzeta = allocate_float(z, r)
    dBdr = allocate_float(z, r)
    dBdz = allocate_float(z, r)
    jacobian = allocate_float(z, r)
    cvdriftr = allocate_float(z, r)
    cvdriftz = allocate_float(z, r)
    gbdriftr = allocate_float(z, r)
    gbdriftz = allocate_float(z, r)
    
    option = geometry_input_data.option
    rhostar = geometry_input_data.rhostar
    if option == "constant-helical" || option == "default"
        # \vec{B} = B ( bz \hat{z} + bzeta \hat{zeta} ) 
        # with B a constant and \hat{z} x \hat{r} . \hat{zeta} = 1
        pitch = geometry_input_data.pitch
        for ir in 1:nr
            for iz in 1:nz
                bzed[iz,ir] = pitch
                bzeta[iz,ir] = sqrt(1 - bzed[iz,ir]^2)
                Bmag[iz,ir] = 1.0
                Bzed[iz,ir] = Bmag[iz,ir]*bzed[iz,ir]
                Bzeta[iz,ir] = Bmag[iz,ir]*bzeta[iz,ir]
                dBdr[iz,ir] = 0.0
                dBdz[iz,ir] = 0.0
                jacobian[iz,ir] = 1.0

                cvdriftr[iz,ir] = 0.0
                cvdriftz[iz,ir] = 0.0
                gbdriftr[iz,ir] = 0.0
                gbdriftz[iz,ir] = 0.0
            end
        end
    elseif option == "1D-mirror"
        # a 1D configuration for testing mirror and vperp physics 
        # with \vec{B} = B(z) bz \hat{z} and
        # with B = B(z) a specified function
        #if nr > 1
        #    input_option_error("$option: You have specified nr > 1 -> set nr = 1", option)
        #end
        DeltaB = geometry_input_data.DeltaB
        if DeltaB < -0.99999999
            input_option_error("$option: You have specified DeltaB < -1 -> set DeltaB > -1", option)
        end
        pitch = geometry_input_data.pitch
        for ir in 1:nr
            for iz in 1:nz
                bzed[iz,ir] = pitch
                bzeta[iz,ir] = sqrt(1 - bzed[iz,ir]^2)
                # B(z)/Bref = 1 + DeltaB*( 2(2z/L)^2 - (2z/L)^4)
                # chosen so that
                # B(z)/Bref = 1 + DeltaB at 2z/L = +- 1 
                # d B(z)d z = 0 at 2z/L = +- 1 
                zfac = 2.0*z.grid[iz]/z.L
                Bmag[iz,ir] = 1.0 + DeltaB*( 2.0*zfac^2 - zfac^4)
                Bzed[iz,ir] = Bmag[iz,ir]*bzed[iz,ir]
                Bzeta[iz,ir] = Bmag[iz,ir]*bzeta[iz,ir]
                dBdr[iz,ir] = 0.0
                dBdz[iz,ir] = (2.0/z.L)*4.0*DeltaB*zfac*(1.0 - zfac^2)
                jacobian[iz,ir] = 1.0

                cvdriftr[iz,ir] = 0.0
                cvdriftz[iz,ir] = 0.0
                gbdriftr[iz,ir] = 0.0
                gbdriftz[iz,ir] = 0.0               
            end
        end
    elseif option == "low-beta-helix"
        # a 2D configuration for testing magnetic drift physics
        # with \vec{B} = (B0/r) \hat{zeta} + Bz \hat{z}
        # with B0 and Bz constants
        pitch = geometry_input_data.pitch
        B0 = 1.0 # chose reference field strength to be Bzeta at r = 1
        Bz = pitch*B0 # pitch determines ratio of Bz/B0 at r = 1
        for ir in 1:nr
            rr = r.grid[ir]
            for iz in 1:nz
                Bmag[iz,ir] = sqrt( (B0/rr)^2 + Bz^2 )
                bzed[iz,ir] = Bz/Bmag[iz,ir]
                bzeta[iz,ir] = B0/(rr*Bmag[iz,ir])
                Bzed[iz,ir] = bzed[iz,ir]*Bmag[iz,ir]
                Bzeta[iz,ir] = bzeta[iz,ir]*Bmag[iz,ir]
                dBdz[iz,ir] = 0.0
                dBdr[iz,ir] = -(Bmag[iz,ir]/rr)*bzeta[iz,ir]^2
                jacobian[iz,ir] = 1.0
                
                cvdriftr[iz,ir] = 0.0
                cvdriftz[iz,ir] = -(bzeta[iz,ir]/Bmag[iz,ir])*(bzeta[iz,ir]^2)/rr
                gbdriftr[iz,ir] = 0.0
                gbdriftz[iz,ir] = cvdriftz[iz,ir]
            end
        end
    elseif option == "0D-Spitzer-test"
        # a 0D configuration with certain geometrical factors
        # set to be constants to enable testing of velocity
        # space operators such as mirror or vperp advection terms
        pitch = geometry_input_data.pitch
        dBdz_constant = geometry_input_data.dBdz_constant
        dBdr_constant = geometry_input_data.dBdr_constant
        B0 = 1.0 # chose reference field strength to be Bzeta at r = 1
        for ir in 1:nr
            for iz in 1:nz
                Bmag[iz,ir] = B0
                bzed[iz,ir] = pitch
                bzeta[iz,ir] = sqrt(1 - pitch^2)
                Bzed[iz,ir] = bzed[iz,ir]*Bmag[iz,ir]
                Bzeta[iz,ir] = bzeta[iz,ir]*Bmag[iz,ir]
                dBdz[iz,ir] = dBdz_constant
                dBdr[iz,ir] = dBdr_constant
                jacobian[iz,ir] = 1.0
                
                cvdriftr[iz,ir] = 0.0
                cvdriftz[iz,ir] = 0.0
                gbdriftr[iz,ir] = 0.0
                gbdriftz[iz,ir] = 0.0
            end
        end
    else 
        input_option_error("$option", option)
    end

    geometry = geometric_coefficients(geometry_input_data, rhostar,
               Bzed,Bzeta,Bmag,bzed,bzeta,dBdz,dBdr,jacobian,
               cvdriftr,cvdriftz,gbdriftr,gbdriftz)
    return geometry
end

end
