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
    
    option = geometry_input_data.option
    rhostar = geometry_input_data.rhostar
    if option == "constant-helical"
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
            end
        end
    else 
        input_option_error("$option", option)
    end

    geometry = geometric_coefficients(geometry_input_data, rhostar,
               Bzed,Bzeta,Bmag,bzed,bzeta,dBdz,dBdr)
    return geometry
end

end
