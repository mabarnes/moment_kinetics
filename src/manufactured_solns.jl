"""
"""
module manufactured_solns

export manufactured_solutions
export manufactured_sources
export manufactured_solutions_as_arrays
export manufactured_rhs_as_array

using SpecialFunctions
using Symbolics

using ..input_structs: advance_info
using ..array_allocation: allocate_float
using ..coordinates: coordinate
using ..input_structs: geometry_input, advance_info
using ..type_definitions

    @variables r z vpa vperp t

    # Define some constants in mk_float. Avoids loss of precision due to implicit
    # conversions from FLoat64->Float128 if we use Float128.
    const half = 1/mk_float(2)
    const one = mk_float(1)
    const two = mk_float(2)
    const four = mk_float(4)
    const five = mk_float(5)
    const ten = mk_float(10)

    function densi_sym(Lr,Lz,r_bc,z_bc)
        # Note: explicitly convert numerical factors to mk_float so the output gets full
        # precision if we use quad-precision (Float128)
        if r_bc == "periodic" && z_bc == "periodic"
            densi = 1 +  (sin(two*pi*r/Lr) + sin(two*pi*z/Lz))/ten*cos(two*pi*t)
        elseif r_bc == "Dirichlet" && z_bc == "periodic"
            #densi = 1 +  1//2*sin(2*pi*z/Lz)*(r/Lr + 1//2) + 0.2*sin(2*pi*r/Lr)*sin(2*pi*t)
            #densi = 1 +  1//2*sin(2*pi*z/Lz)*(r/Lr + 1//2) + sin(2*pi*r/Lr)*sin(2*pi*t)
            densi = 1 +  half*(r/Lr + half)/two
        elseif r_bc == "periodic" && z_bc == "wall"
            densi = (half - z/Lz)/four + (z/Lz + half)/four + (z/Lz + half)*(half - z/Lz)/five  #+  1//2*(r/Lr + 1//2) + 1//2*(z/Lz + 1//2)
        else
            error("Unsupported options r_bc=$r_bc, z_bc=$z_bc")
        end
        return densi
    end

    function dfni_sym(Lr,Lz,Lvpa,Lvperp,r_bc,z_bc)
        # Note: explicitly convert numerical factors to mk_float so the output gets full
        # precision if we use quad-precision (Float128)
        densi = densi_sym(Lr,Lz,r_bc,z_bc)
        if (r_bc == "periodic" && z_bc == "periodic") || (r_bc == "Dirichlet" && z_bc == "periodic")
            ## This version with upar is very expensive to evaluate, probably due to
            ## spatially-varying erf()?
            #upar = (sin(two*pi*r/Lr) + sin(2*pi*z/Lz))/ten
            #dfni = densi * exp(- (vpa-upar)^2 - vperp^2) #/ pi^1.5
            ## Force the symbolic function dfni to vanish when vpa=±Lvpa/2, so that it is
            ## consistent with "zero" bc in vpa.
            ## Normalisation factors on 2nd and 3rd lines ensure that when this f is
            ## integrated on -Lvpa/2≤vpa≤Lvpa/2 and 0≤vperp≤Lvperp the result (taking
            ## into account the normalisations used by moment_kinetics) is exactly n.
            ## Note that:
            ##  ∫_-Lvpa/2^Lvpa/2  [1-(2*(vpa-upar)/Lvpa)^2]exp(-(vpa-upar)^2) dvpa
            ##  = [sqrt(π)(Lvpa^2-4*upar^2-2)*(erf(Lvpa/2-upar)+erf(Lvpa/2+upar))
            ##     +2*exp(-(Lvpa+2upar)^2/4)*(exp(2*Lvpa*upar)*(Lvpa+2upar)+Lvpa-2*upar)] / 2 / Lvpa^2
            ##
            ##  ∫_0^Lvperp vperp*exp(-vperp^2) = (1 - exp(-Lvperp^2))/2
            #dfni = (one - (two*vpa/Lvpa)^2) * dfni *
            #          two*Lvpa / ((Lvpa^2-four*upar^2-2)*(erf(Lvpa/two-upar)+erf(Lvpa/two+upar)) + two/sqrt(π)*exp(-(Lvpa+two*upar)^2/four)*(exp(two*Lvpa*upar)+Lvpa-two*upar)) / # vpa normalization
            #         (one - exp(-Lvperp^2)) # vperp normalisation

            # Ad-hoc odd-in-vpa component of dfni which gives non-zero upar and dfni
            # positive everywhere, while the odd component integrates to 0 so does not
            # need to be accounted for in normalisation.
            upar = (sin(two*pi*r/Lr) + sin(2*pi*z/Lz)/ten)
            dfni = densi * (exp(- vpa^2 - vperp^2)
                            + (sin(two*pi*r/Lr) + sin(2*pi*z/Lz))/ten
                              * vpa * exp(-two*(vpa^2+vperp^2))) #/ pi^1.5
            # Force the symbolic function dfni to vanish when vpa=±Lvpa/2, so that it is
            # consistent with "zero" bc in vpa.
            # Normalisation factors on 2nd and 3rd lines ensure that when this f is
            # integrated on -Lvpa/2≤vpa≤Lvpa/2 and 0≤vperp≤Lvperp the result (taking
            # into account the normalisations used by moment_kinetics) is exactly n.
            # Note that:
            #  ∫_-Lvpa/2^Lvpa/2  [1-(2*vpa/Lvpa)^2]exp(-vpa^2) dvpa
            #  = [sqrt(π)(Lvpa^2-2)*erf(Lvpa/2) + 2*exp(-Lvpa^2/4)Lvpa] / Lvpa^2
            #
            #  ∫_0^Lvperp vperp*exp(-vperp^2) = (1 - exp(-Lvperp^2))/2
            dfni = (one - (two*vpa/Lvpa)^2) * dfni *
                      Lvpa / ((Lvpa^2-2)*erf(Lvpa/two) + two/sqrt(π)*exp(-Lvpa^2/four)*Lvpa) / # vpa normalization
                     (one - exp(-Lvperp^2)) # vperp normalisation
        elseif r_bc == "periodic" && z_bc == "wall"
            Hplus = (sign(vpa) + one) * half
            Hminus = (sign(-vpa) + one) * half
            ffa =  exp(- vperp^2)
            dfni = ffa * ( (half - z/Lz) * Hminus * vpa^2 + (z/Lz + half) * Hplus * vpa^2 + (z/Lz + half)*(half - z/Lz)/five ) * exp( - vpa^2 )
        else
            error("Unsupported options r_bc=$r_bc, z_bc=$z_bc")
        end
        return dfni
    end

    function manufactured_solutions(Lr,Lz,Lvpa,Lvperp,r_bc,z_bc)

        densi = densi_sym(Lr,Lz,r_bc,z_bc)
        dfni = dfni_sym(Lr,Lz,Lvpa,Lvperp,r_bc,z_bc)
        
        #build julia functions from these symbolic expressions
        # cf. https://docs.juliahub.com/Symbolics/eABRO/3.4.0/tutorials/symbolic_functions/
        densi_func = build_function(densi, z, r, t, expression=Val{false})
        dfni_func = build_function(dfni, vpa, vperp, z, r, t, expression=Val{false})
        # return function
        # call like: 
        # densi_func(zval, rval, tval) 
        # dfni_func(vpaval, vperpval, zval, rval, tval) 
        return dfni_func, densi_func
    end 

    function manufactured_rhs_sym(Lr::mk_float,Lz::mk_float,Lvpa::mk_float,Lvperp::mk_float,r_bc::String,z_bc::String,geometry::geometry_input,advance::Union{advance_info,Nothing}=nothing)
        # Note: explicitly convert numerical factors to mk_float so the output gets full
        # precision if we use quad-precision (Float128)

        densi = densi_sym(Lr,Lz,r_bc,z_bc)
        dfni = dfni_sym(Lr,Lz,Lvpa,Lvperp,r_bc,z_bc)
        
        Dr = Differential(r) 
        Dz = Differential(z) 
        Dvpa = Differential(vpa) 
        Dvperp = Differential(vperp) 
    
        Bzed = geometry.Bzed
        Bmag = geometry.Bmag
        rhostar = geometry.rstar
        #Bzed = 1.0 #geometry.Bzed
        #Bmag = 1.0 #geometry.Bmag
        #rhostar = 1.0 #geometry.rstar
        
        phi = log(densi)
        Er = -Dr(phi)
        Ez = -Dz(phi)
    
        rhs = 0 * z
        if advance === nothing || advance.vpa_advection
            rhs += - ( half*Ez*Bzed/Bmag ) * Dvpa(dfni)
        end
        if advance === nothing || advance.z_advection
            rhs += -( vpa * (Bzed/Bmag) - half*rhostar*Er ) * Dz(dfni)
        end
        if advance === nothing || advance.r_advection
            rhs += - ( half*rhostar*Ez ) * Dr(dfni)
        end
        #if advance === nothing || advance.cx_collsions
        #    # placeholder
        #end
        #if advance === nothing || advance.ionization_collsions
        #    # placeholder
        #end
        #if advance === nothing || advance.source_terms
        #    # placeholder
        #end
        #if advance === nothing || advance.continuity
        #    # placeholder
        #end
        #if advance === nothing || advance.force_balance
        #    # placeholder
        #end
        #if advance === nothing || advance.energy
        #    # placeholder
        #end

        return expand_derivatives(rhs)
    end

    function manufactured_rhs(Lr::mk_float,Lz::mk_float,Lvpa::mk_float,Lvperp::mk_float,r_bc::String,z_bc::String,geometry::geometry_input,advance::Union{advance_info,Nothing}=nothing)
        rhs_sym = manufactured_rhs_sym(Lr,Lz,Lvpa,Lvperp,r_bc,z_bc,geometry,advance)
        return build_function(rhs_sym, vpa, vperp, z, r, t, expression=Val{false})
    end

    #function manufactured_sources(dfni,densi,geometry)
    function manufactured_sources(Lr::mk_float,Lz::mk_float,Lvpa::mk_float,Lvperp::mk_float,r_bc::String,z_bc::String,geometry::geometry_input)

        dfni = dfni_sym(Lr,Lz,Lvpa,Lvperp,r_bc,z_bc)

        Dt = Differential(t)

        S = Dt(dfni) - manufactured_rhs_sym(Lr,Lz,Lvpa,Lvperp,r_bc,z_bc,geometry)
        Source_i = expand_derivatives(S)
        
        Source_i_func = build_function(Source_i, vpa, vperp, z, r, t, expression=Val{false})
        return Source_i_func
    end 

    """
        manufactured_solutions_as_arrays(
            t::mk_float, r::AbstractVector, z::AbstractVector, vperp::AbstractVector,
            vpa::AbstractVector)

    Create array filled with manufactured solutions.

    Returns
    -------
    (densi, phi, dfni)
    """
    function manufactured_solutions_as_arrays(
        t::mk_float, r::coordinate, z::coordinate, vperp::coordinate,
        vpa::coordinate)

        dfni_func, densi_func = manufactured_solutions(r.L, z.L, vpa.L, r.bc, z.bc)

        densi = allocate_float(z.n, r.n)
        dfni = allocate_float(vpa.n, vperp.n, z.n, r.n)

        for ir ∈ 1:r.n, iz ∈ 1:z.n
            densi[iz,ir] = densi_func(z.grid[iz], r.grid[ir], t)
            for ivperp ∈ 1:vperp.n, ivpa ∈ 1:vpa.n
                dfni[ivpa,ivperp,iz,ir] = dfni_func(vpa.grid[ivpa], vperp.grid[ivperp], z.grid[iz],
                                        r.grid[ir], t)
            end
        end

        phi = log.(densi)

        return densi, phi, dfni
    end

    """
        manufactured_rhs_as_array(
            t::mk_float, r::AbstractVector, z::AbstractVector, vperp::AbstractVector,
            vpa::AbstractVector, geometry::geometry_input, advance::Union{advance_info,Nothing})

    Create array filled with manufactured rhs.

    Returns
    -------
    rhs
    """
    function manufactured_rhs_as_array(
        t::mk_float, r::coordinate, z::coordinate, vperp::coordinate,
        vpa::coordinate, geometry::geometry_input, advance::Union{advance_info,Nothing})

        rhs_func = manufactured_rhs(r.L, z.L, vpa.L, vperp.L, r.bc, z.bc, geometry, advance)

        rhs = allocate_float(vpa.n, vperp.n, z.n, r.n)

        for ir ∈ 1:r.n, iz ∈ 1:z.n
            for ivperp ∈ 1:vperp.n, ivpa ∈ 1:vpa.n
                rhs[ivpa,ivperp,iz,ir] = rhs_func(vpa.grid[ivpa], vperp.grid[ivperp],
                                                  z.grid[iz], r.grid[ir], t)
            end
        end

        return rhs
    end

end
