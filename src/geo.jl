"""
module for including axisymmetric geometry
in coordinates (z,r), with z the vertical 
coordinate and r the radial coordinate
"""
module geo

export init_magnetic_geometry

using ..input_structs: geometry_input
using ..file_io: input_option_error
using ..array_allocation: allocate_float
using ..type_definitions: mk_float, mk_int

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
# jacobian =  grad r x grad z . grad zeta
jacobian::Array{mk_float,2}
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
    Bzed = allocate_float(nz,nr)
    Bzeta = allocate_float(nz,nr)
    Bmag = allocate_float(nz,nr)
    bzed = allocate_float(nz,nr)
    bzeta = allocate_float(nz,nr)
    dBdr = allocate_float(nz,nr)
    dBdz = allocate_float(nz,nr)
    jacobian = allocate_float(nz,nr)
    
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
            end
        end
    elseif option == "1D-mirror"
        # a 1D configuration for testing mirror and vperp physics 
        # with \vec{B} = B(z) bz \hat{z} and
        # with B = B(z) a specified function
        if nr > 1
            input_option_error("$option: You have specified nr > 1 -> set nr = 1", option)
        end
        DeltaB = geometry_input_data.DeltaB
        for ir in 1:nr
            for iz in 1:nz
                bzed[iz,ir] = 1.0
                bzeta[iz,ir] = 0.0
                # B(z)/Bref = 1 + DeltaB*( 2(2z/L)^2 - (2z/L)^4)
                # chosen so that
                # B(z)/Bref = 1 + DeltaB at 2z/L = +- 1 
                # d B(z)d z = 0 at 2z/L = +- 1 
                zfac = 2.0*z.grid[iz]/z.L
                Bmag[iz,ir] = 1.0 + DeltaB*( 2.0*zfac^2 - zfac^4)
                Bzed[iz,ir] = Bmag[iz,ir]*bzed[iz,ir]
                Bzeta[iz,ir] = Bmag[iz,ir]*bzeta[iz,ir]
                dBdr[iz,ir] = 0.0
                dBdz[iz,ir] = 4.0*DeltaB*zfac*(1.0 - zfac^2)
                jacobian[iz,ir] = 1.0
            end
        end
    else 
        input_option_error("$option", option)
    end

    geometry = geometric_coefficients(geometry_input_data, rhostar,
               Bzed,Bzeta,Bmag,bzed,bzeta,dBdz,dBdr,jacobian)
    return geometry
end

end
