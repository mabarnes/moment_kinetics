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
using ..array_allocation: allocate_shared_float
using ..type_definitions: mk_float, mk_int, MPISharedArray
using ..reference_parameters: setup_reference_parameters
using ..derivatives: derivative_z!
using ..looping

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
    Bzed::MPISharedArray{mk_float,2}
    # Bzeta/Bref
    Bzeta::MPISharedArray{mk_float,2}
    # Btot/Bref
    Bmag::MPISharedArray{mk_float,2}
    # bz -- unit vector component in z direction
    bzed::MPISharedArray{mk_float,2}
    # bz -- unit vector component in zeta direction
    bzeta::MPISharedArray{mk_float,2}


    # now the new coefficients

    # d Bmag d z
    dBdz::MPISharedArray{mk_float,2}
    # d Bmag d r
    dBdr::MPISharedArray{mk_float,2}
    # jacobian =  r grad r x grad z . grad zeta
    jacobian::MPISharedArray{mk_float,2}

    # magnetic drift physics coefficients
    # curvature_drift_r = (b/B) x (b.grad b) . grad r
    curvature_drift_r::MPISharedArray{mk_float,2}
    # curvature_drift_z = (b/B) x (b.grad b) . grad z
    curvature_drift_z::MPISharedArray{mk_float,2}
    # grad_B_drift_r = (b/B^2) x grad B . grad r
    grad_B_drift_r::MPISharedArray{mk_float,2}
    # grad_B_drift_z = (b/B^2) x grad B . grad z
    grad_B_drift_z::MPISharedArray{mk_float,2}
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
function init_magnetic_geometry(geometry_input_data::geometry_input,z,r,z_spectral)
    nz = z.n
    nr = r.n
    Bzed = allocate_shared_float(z, r)
    Bzeta = allocate_shared_float(z, r)
    Bmag = allocate_shared_float(z, r)
    bzed = allocate_shared_float(z, r)
    bzeta = allocate_shared_float(z, r)
    dBdr = allocate_shared_float(z, r)
    dBdz = allocate_shared_float(z, r)
    jacobian = allocate_shared_float(z, r)
    curvature_drift_r = allocate_shared_float(z, r)
    curvature_drift_z = allocate_shared_float(z, r)
    grad_B_drift_r = allocate_shared_float(z, r)
    grad_B_drift_z = allocate_shared_float(z, r)

    @begin_r_z_region()
    
    option = geometry_input_data.option
    rhostar = geometry_input_data.rhostar
    if option == "constant-helical" || option == "default"
        # \vec{B} = B ( bz \hat{z} + bzeta \hat{zeta} ) 
        # with B a constant and \hat{z} x \hat{r} . \hat{zeta} = 1
        pitch = geometry_input_data.pitch
        @loop_r_z ir iz begin
            bzed[iz,ir] = pitch
            bzeta[iz,ir] = sqrt(1 - bzed[iz,ir]^2)
            Bmag[iz,ir] = 1.0
            Bzed[iz,ir] = Bmag[iz,ir]*bzed[iz,ir]
            Bzeta[iz,ir] = Bmag[iz,ir]*bzeta[iz,ir]
            dBdr[iz,ir] = 0.0
            dBdz[iz,ir] = 0.0
            jacobian[iz,ir] = 1.0

            curvature_drift_r[iz,ir] = 0.0
            curvature_drift_z[iz,ir] = 0.0
            grad_B_drift_r[iz,ir] = 0.0
            grad_B_drift_z[iz,ir] = 0.0
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
        @loop_r_z ir iz begin
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

            curvature_drift_r[iz,ir] = 0.0
            curvature_drift_z[iz,ir] = 0.0
            grad_B_drift_r[iz,ir] = 0.0
            grad_B_drift_z[iz,ir] = 0.0               
        end
    elseif option == "low-beta-helix"
        # a 2D configuration for testing magnetic drift physics
        # with \vec{B} = (B0/r) \hat{zeta} + Bz \hat{z}
        # with B0 and Bz constants
        pitch = geometry_input_data.pitch
        B0 = 1.0 # chose reference field strength to be Bzeta at r = 1
        Bz = pitch*B0 # pitch determines ratio of Bz/B0 at r = 1
        @loop_r_z ir iz begin
            Bmag[iz,ir] = sqrt( (B0/r.grid[ir])^2 + Bz^2 )
            bzed[iz,ir] = Bz/Bmag[iz,ir]
            bzeta[iz,ir] = B0/(r.grid[ir]*Bmag[iz,ir])
            Bzed[iz,ir] = bzed[iz,ir]*Bmag[iz,ir]
            Bzeta[iz,ir] = bzeta[iz,ir]*Bmag[iz,ir]
            dBdz[iz,ir] = 0.0
            dBdr[iz,ir] = -(Bmag[iz,ir]/r.grid[ir])*bzeta[iz,ir]^2
            jacobian[iz,ir] = 1.0
            
            curvature_drift_r[iz,ir] = 0.0
            curvature_drift_z[iz,ir] = -(bzeta[iz,ir]/Bmag[iz,ir])*(bzeta[iz,ir]^2)/r.grid[ir]
            grad_B_drift_r[iz,ir] = 0.0
            grad_B_drift_z[iz,ir] = curvature_drift_z[iz,ir]
        end
    elseif option == "0D-Spitzer-test"
        # a 0D configuration with certain geometrical factors
        # set to be constants to enable testing of velocity
        # space operators such as mirror or vperp advection terms
        pitch = geometry_input_data.pitch
        dBdz_constant = geometry_input_data.dBdz_constant
        dBdr_constant = geometry_input_data.dBdr_constant
        B0 = 1.0 # chose reference field strength to be Bzeta at r = 1
        @loop_r_z ir iz begin
            Bmag[iz,ir] = B0
            bzed[iz,ir] = pitch
            bzeta[iz,ir] = sqrt(1 - pitch^2)
            Bzed[iz,ir] = bzed[iz,ir]*Bmag[iz,ir]
            Bzeta[iz,ir] = bzeta[iz,ir]*Bmag[iz,ir]
            dBdz[iz,ir] = dBdz_constant
            dBdr[iz,ir] = dBdr_constant
            jacobian[iz,ir] = 1.0
            
            curvature_drift_r[iz,ir] = 0.0
            curvature_drift_z[iz,ir] = 0.0
            grad_B_drift_r[iz,ir] = 0.0
            grad_B_drift_z[iz,ir] = 0.0
        end
    elseif option == "1D-Helical-ITG"
        # a 1D configuration for finding an ITG mode via implementing
        # grad B drift and adding a radial temperature gradient term to the 
        # drift kinetic equation we use. Although there is no R domain 
        # in this setup, we can still claim that there is an ExB drift 
        # advecting new temperature plasma into the domain because of the 
        # "presence" of this radial temperature gradient.
        # The ordering required for the mode to happen requires that 
        # B_z/B_zeta << rho_i/2*L_B, where L_B is length scale of radial B 
        # variation. Also need L_T << L_B. Will probably update this soon, 
        # depending on what I see.
        pitch = geometry_input_data.pitch
        dB_dr = geometry_input_data.dBdr_constant
        B_0 = 1.0
        @loop_r_z ir iz begin
            bzed[iz,ir] = pitch
            bzeta[iz,ir] = sqrt(1 - bzed[iz,ir]^2)
            Bmag[iz,ir] = B_0
            Bzed[iz,ir] = Bmag[iz,ir]*bzed[iz,ir]
            Bzeta[iz,ir] = Bmag[iz,ir]*bzeta[iz,ir]
            dBdr[iz,ir] = dB_dr
            dBdz[iz,ir] = 0.0
            jacobian[iz,ir] = 1.0

            curvature_drift_r[iz,ir] = 0.0
            curvature_drift_z[iz,ir] = 0.0

            grad_B_drift_r[iz,ir] = 0.0
            grad_B_drift_z[iz,ir] = bzeta[iz,ir] * dBdr[iz,ir]/Bmag[iz,ir]
        end
        L_B = 1/(dB_dr*1/Bmag[1,1])
        println("L_B = $L_B")
    elseif option == "1D-mirror-MAST-edge"
        # a 1D configuration along z, with no pitch, but the magnetic field 
        # strength varies along the line. Its variation matches very closely to 
        # a single field line in the edge of MAST (from a hypnotoad analysis of 
        # an eqdsk file), assuming uniform z (i.e. arc length along field line.)
        # Aim is to see the mirror force in action, and trapping physics in the edge.
        # field line fit (really quite accurate!) is: 
        # a0=0.32884641, a2=-0.4199673, a4=3.1528366, a6=-6.3052343, a8=4.0532678
        # z grid needs to be mapped from -1 to 1 for these polynomials to work.
        a0, a2, a4, a6, a8 = 0.32884641, -0.4199673, 3.1528366, -6.3052343, 4.0532678

        pitch = geometry_input_data.pitch
        if pitch != 1.0
            input_option_error("option: You have specified pitch != 1, but z is arc length coordinate in 1D-mirror-MAST-edge geometry", option)
        end

        B_0 = 1.0

        @loop_r_z ir iz begin
            bzed[iz,ir] = pitch
            bzeta[iz,ir] = sqrt(1 - bzed[iz,ir]^2)
            Bmag[iz,ir] = B_0 * (a0 + (z.grid[iz]*(2.0/z.L))^2 * (a2 + (z.grid[iz]*(2.0/z.L))^2 * (a4 + 
                                        (z.grid[iz]*(2.0/z.L))^2 * (a6 + a8 * (z.grid[iz]*(2.0/z.L))^2))))
            Bzed[iz,ir] = Bmag[iz,ir]*bzed[iz,ir]
            Bzeta[iz,ir] = Bmag[iz,ir]*bzeta[iz,ir]
            dBdr[iz,ir] = 0.0

            jacobian[iz,ir] = 1.0


            curvature_drift_r[iz,ir] = 0.0
            curvature_drift_z[iz,ir] = 0.0

            grad_B_drift_r[iz,ir] = 0.0
            grad_B_drift_z[iz,ir] = 0.0
        end

        # now calculate dBdz using spectral derivative
        @views derivative_z!(dBdz, Bmag, r.scratch_shared, r.scratch_shared2,
                             r.scratch_shared3, r.scratch_shared4, z_spectral, z)
    else
        input_option_error("$option", option)
    end

    geometry = geometric_coefficients(geometry_input_data, rhostar,
               Bzed,Bzeta,Bmag,bzed,bzeta,dBdz,dBdr,jacobian,
               curvature_drift_r,curvature_drift_z,grad_B_drift_r,grad_B_drift_z)
    return geometry
end

end
