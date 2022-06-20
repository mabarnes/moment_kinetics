"""
"""
module manufactured_solns

export manufactured_solutions
export manufactured_sources
export manufactured_solutions_as_arrays
export manufactured_rhs_as_array

using SpecialFunctions
using Symbolics
using ..array_allocation: allocate_float
using ..coordinates: coordinate
using ..input_structs: geometry_input, advance_info, species_composition, collisions_input
using ..type_definitions

    @variables r z vpa vperp t vz vr vzeta

    #standard functions for building densities
    function nplus_sym(Lr,r_bc)
        #if r_bc == "periodic"
        nplus = 1.0 + 0.3*sin(2.0*pi*r/Lr)
        #end
        return nplus
    end
    
    function nminus_sym(Lr,r_bc)
        #if r_bc == "periodic"
        nminus = 1.0 + 0.3*sin(2.0*pi*r/Lr)
        #end
        return nminus
    end
    
    function nzero_sym(Lr,r_bc)
        #if r_bc == "periodic"
        nzero = 1.0 + 0.3*sin(2.0*pi*r/Lr)# 1.0 #+ (r/Lr + 0.5)*(0.5 - r/Lr)
        #end
        return nzero
    end

    function knudsen_cosine(composition)
        T_wall = composition.T_wall
        # prefac here may cause problems with NaNs if vz = vr = vzeta = 0 is on grid
        prefac = abs(vz)/sqrt(vz^2 + vr^2 + vzeta^2)
        exponetial = exp( - (vz^2 + vr^2 + vzeta^2)/T_wall )
        knudsen_pdf = (3.0*sqrt(pi)/T_wall^2)*prefac*exponetial
    
        return knudsen_pdf
    end

    # neutral density symbolic function
    function densn_sym(Lr,Lz,r_bc,z_bc,geometry,composition)
        if r_bc == "periodic" && z_bc == "periodic" 
            densn = 1.5 +  0.1*(cos(2.0*pi*r/Lr) + cos(2.0*pi*z/Lz))*sin(2.0*pi*t)  
        elseif (r_bc == "periodic" && z_bc == "wall")
            T_wall = composition.T_wall
            Bzed = geometry.Bzed
            Bmag = geometry.Bmag
            Gamma_minus = 0.5*(Bzed/Bmag)*nminus_sym(Lr,r_bc)/sqrt(pi)
            Gamma_plus = 0.5*(Bzed/Bmag)*nplus_sym(Lr,r_bc)/sqrt(pi)
            # exact integral of corresponding dfnn below
            densn = 3.0*sqrt(pi)/(4.0*sqrt(T_wall))*( (0.5 - z/Lz)*Gamma_minus + (0.5 + z/Lz)*Gamma_plus + 2.0 )
        else
            densn = 1.0
        end
        return densn
    end

    # neutral distribution symbolic function
    function dfnn_sym(Lr,Lz,r_bc,z_bc,geometry,composition)
        densn = densn_sym(Lr,Lz,r_bc,z_bc,geometry,composition)
        if (r_bc == "periodic" && z_bc == "periodic")
            dfnn = densn * exp( - vz^2 - vr^2 - vzeta^2)
        elseif (r_bc == "periodic" && z_bc == "wall")
            Hplus = 0.5*(sign(vz) + 1.0)
            Hminus = 0.5*(sign(-vz) + 1.0)
            FKw = knudsen_cosine(composition)
            Bzed = geometry.Bzed
            Bmag = geometry.Bmag
            Gamma_minus = 0.5*(Bzed/Bmag)*nminus_sym(Lr,r_bc)/sqrt(pi)
            Gamma_plus = 0.5*(Bzed/Bmag)*nplus_sym(Lr,r_bc)/sqrt(pi)
            dfnn = Hplus *( ( 0.5 - z/Lz)*Gamma_minus + 1.0 )*FKw + Hminus*( ( 0.5 + z/Lz)*Gamma_plus + 1.0 )*FKw 
        end
        return dfnn
    end
    function gyroaveraged_dfnn_sym(Lr,Lz,r_bc,z_bc,geometry,composition)
        densn = densn_sym(Lr,Lz,r_bc,z_bc,geometry,composition)
        #if (r_bc == "periodic" && z_bc == "periodic")
            dfnn = densn * exp( - vpa^2 - vperp^2 )
        #end
        return dfnn
    end
    
    # Define some constants in mk_float. Avoids loss of precision due to implicit
    # conversions from FLoat64->Float128 if we use Float128.
    const half = 1/mk_float(2)
    const one = mk_float(1)
    const three_over_two = mk_float(3)/mk_float(2)
    const two = mk_float(2)
    const four = mk_float(4)
    const five = mk_float(5)
    const ten = mk_float(10)

    function densi_sym(Lr,Lz,r_bc,z_bc)
        # Note: explicitly convert numerical factors to mk_float so the output gets full
        # precision if we use quad-precision (Float128)
        if r_bc == "periodic" && z_bc == "periodic"
            densi = three_over_two +  (sin(two*pi*r/Lr) + sin(two*pi*z/Lz))/ten*cos(two*pi*t)
        elseif r_bc == "Dirichlet" && z_bc == "periodic"
            #densi = 1 +  1//2*sin(2*pi*z/Lz)*(r/Lr + 1//2) + 0.2*sin(2*pi*r/Lr)*sin(2*pi*t)
            #densi = 1 +  1//2*sin(2*pi*z/Lz)*(r/Lr + 1//2) + sin(2*pi*r/Lr)*sin(2*pi*t)
            densi = one +  half*(r/Lr + half)/two
        elseif r_bc == "periodic" && z_bc == "wall"
            densi = (half - z/Lz)/four + (z/Lz + half)/four + (z/Lz + half)*(half - z/Lz)/five  #+  1//2*(r/Lr + 1//2) + 1//2*(z/Lz + 1//2)
        else
            error("Unsupported options r_bc=$r_bc, z_bc=$z_bc")
        end
        return densi
    end

    

    # ion distribution symbolic function
    function dfni_sym(Lr,Lz,Lvpa,Lvperp,r_bc,z_bc,geometry,nr)
        # Note: explicitly convert numerical factors to mk_float so the output gets full
        # precision if we use quad-precision (Float128)
        densi = densi_sym(Lr,Lz,r_bc,z_bc)
        # define derivative operators
        Dr = Differential(r) 
        Dz = Differential(z) 
        # calculate the necessary electric fields
        dense = densi # get the electron density via quasineutrality with Zi = 1
        phi = log(dense) # use the adiabatic response of electrons for me/mi -> 0
        Er = -Dr(phi)
        #Ez = -Dz(phi)
        # get geometric/composition data
        Bzed = geometry.Bzed
        Bmag = geometry.Bmag
        rhostar = geometry.rstar
        if nr > 1 #keep radial derivatives
            rfac = 1.0
        else # drop radial derivative terms
            rfac = 0.0
        end
        
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
            vpabar = vpa - (rhostar/two)*(Bmag/Bzed)*expand_derivatives(Er)*rfac # effective velocity in z direction * (Bmag/Bzed)
            Hplus = half*(sign(vpabar) + one)
            Hminus = half*(sign(-vpabar) + one)
            ffa =  exp(- vperp^2)
            dfni = ffa * ( nminus_sym(Lr,r_bc)* (half - z/Lz) * Hminus * vpabar^2 + nminus_sym(Lr,r_bc)*(z/Lz + half) * Hplus * vpabar^2 + nzero_sym(Lr,r_bc)*(z/Lz + half)*(half - z/Lz) ) * exp( - vpabar^2 )
        else
            error("Unsupported options r_bc=$r_bc, z_bc=$z_bc")
        end
        return dfni
    end
    function cartesian_dfni_sym(Lr,Lz,r_bc,z_bc)
        densi = densi_sym(Lr,Lz,r_bc,z_bc)
        #if (r_bc == "periodic" && z_bc == "periodic") || (r_bc == "Dirichlet" && z_bc == "periodic")
            dfni = densi * exp( - vz^2 - vr^2 - vzeta^2) 
        #end
        return dfni
    end

    function manufactured_solutions(Lr,Lz,Lvpa,Lvperp,r_bc,z_bc,geometry,composition,nr)

        densi = densi_sym(Lr,Lz,r_bc,z_bc)
        dfni = dfni_sym(Lr,Lz,Lvpa,Lvperp,r_bc,z_bc,geometry,nr)
        
        densn = densn_sym(Lr,Lz,r_bc,z_bc,geometry,composition)
        dfnn = dfnn_sym(Lr,Lz,r_bc,z_bc,geometry,composition)
        
        #build julia functions from these symbolic expressions
        # cf. https://docs.juliahub.com/Symbolics/eABRO/3.4.0/tutorials/symbolic_functions/
        densi_func = build_function(densi, z, r, t, expression=Val{false})
        densn_func = build_function(densn, z, r, t, expression=Val{false})
        dfni_func = build_function(dfni, vpa, vperp, z, r, t, expression=Val{false})
        dfnn_func = build_function(dfnn, vz, vr, vzeta, z, r, t, expression=Val{false})
        # return function
        # call like: 
        # densi_func(zval, rval, tval) 
        # dfni_func(vpaval, vperpval, zval, rval, tval) 
        # densn_func(zval, rval, tval) 
        # dfnn_func(vzval, vrval, vzetapval, zval, rval, tval) 
        
        manufactured_solns_list = (densi_func = densi_func, densn_func = densn_func, dfni_func = dfni_func, dfnn_func = dfnn_func)
        
        return manufactured_solns_list
    end 

    function manufactured_rhs_sym(Lr::mk_float,Lz::mk_float,Lvpa::mk_float,Lvperp::mk_float,r_bc::String,z_bc::String,
                                  composition::species_composition,geometry::geometry_input,collisions::collisions_input,
                                  nr::mk_int,advance::Union{advance_info,Nothing}=nothing)
        # Note: explicitly convert numerical factors to mk_float so the output gets full
        # precision if we use quad-precision (Float128)

        # ion manufactured solutions
        densi = densi_sym(Lr,Lz,r_bc,z_bc)
        dfni = dfni_sym(Lr,Lz,Lvpa,Lvperp,r_bc,z_bc,geometry,nr)
        vrvzvzeta_dfni = cartesian_dfni_sym(Lr,Lz,r_bc,z_bc) #dfni in vr vz vzeta coordinates
        
        # neutral manufactured solutions
        densn = densn_sym(Lr,Lz,r_bc,z_bc,geometry,composition)
        dfnn = dfnn_sym(Lr,Lz,r_bc,z_bc,geometry,composition)
        gav_dfnn = gyroaveraged_dfnn_sym(Lr,Lz,r_bc,z_bc,geometry,composition) # gyroaverage < dfnn > in vpa vperp coordinates
        
        # define derivative operators
        Dr = Differential(r) 
        Dz = Differential(z) 
        Dvpa = Differential(vpa) 
        Dvperp = Differential(vperp) 
    
        # get geometric/composition data
        Bzed = geometry.Bzed
        Bmag = geometry.Bmag
        rhostar = geometry.rstar
        #exceptions for cases with missing terms 
        if composition.n_neutral_species > 0
            cx_frequency = collisions.charge_exchange
            ionization_frequency = collisions.ionization
        else 
            cx_frequency = 0.0
            ionization_frequency = 0.0
        end
        if nr > 1
            rfac = 1.0
        else
            rfac = 0.0
        end
        
        # calculate the electric fields
        dense = densi # get the electron density via quasineutrality with Zi = 1
        phi = log(dense) # use the adiabatic response of electrons for me/mi -> 0
        Er = -Dr(phi)
        Ez = -Dz(phi)
    
        rhs_ion = 0 * z
        if advance === nothing || advance.vpa_advection
            rhs_ion += - ( half*Ez*Bzed/Bmag ) * Dvpa(dfni)
        end
        if advance === nothing || advance.z_advection
            rhs_ion += -( vpa * (Bzed/Bmag) - half*rhostar*Er ) * Dz(dfni)
        end
        if advance === nothing || advance.r_advection
            rhs_ion += - ( half*rhostar*Ez ) * Dr(dfni)
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
        rhs_neutral = 0 * z
        if advance == nothing || advance.neutral_z_advection
            rhs_neutral += -vz * Dz(dfnn)
        end
        if advance == nothing || advance.neutral_r_advection
            rhs_neutral += - rfac*vr * Dr(dfnn)
        end
        if advance == nothing || advance.cx_collisions
            rhs_neutral += - cx_frequency* (densi*dfnn - densn*vrvzvzeta_dfni)
        end
        if advance == nothing || advance.ionization_collisions
            rhs_neutral += - ionization_frequency*dense*dfnn
        end

        return expand_derivatives(rhs_ion), expand_derivatives(rhs_neutral)
    end

    function manufactured_rhs(Lr::mk_float, Lz::mk_float, Lvpa::mk_float,
                              Lvperp::mk_float, r_bc::String, z_bc::String,
                              composition::species_composition,
                              geometry::geometry_input, collisions::collisions_input,
                              nr::mk_int, advance::Union{advance_info,Nothing}=nothing)
        rhs_ion_sym, rhs_neutral_sym = manufactured_rhs_sym(Lr,Lz,Lvpa,Lvperp,r_bc,z_bc,composition,geometry,collisions,nr,advance)
        return build_function(rhs_ion_sym, vpa, vperp, z, r, t, expression=Val{false}),
               build_function(rhs_neutral_sym, vz, vr, vzeta, z, r, t, expression=Val{false})
    end

    function manufactured_sources(Lr::mk_float,Lz::mk_float,Lvpa::mk_float,Lvperp::mk_float,r_bc::String,z_bc::String,
                                  composition::species_composition,geometry::geometry_input,collisions::collisions_input,
                                  nr::mk_int)

        dfni = dfni_sym(Lr,Lz,Lvpa,Lvperp,r_bc,z_bc)
        dfnn = dfnn_sym(Lr,Lz,r_bc,z_bc)
        rhs_ion, rhs_neutral = manufactured_rhs_sym(Lr,Lz,Lvpa,Lvperp,r_bc,z_bc,composition,geometry,collisions,nr)

        Dt = Differential(t)

        Si = Dt(dfni) - rhs_ion
        Source_i = expand_derivatives(Si)
        Sn = Dt(dfnn) - rhs_neutral
        Source_n = expand_derivatives(Sn)
        
        Source_i_func = build_function(Source_i, vpa, vperp, z, r, t, expression=Val{false})
        Source_n_func = build_function(Source_n, vz, vr, vzeta, z, r, t, expression=Val{false})
        
        manufactured_sources_list = (Source_i_func = Source_i_func, Source_n_func = Source_n_func)
        
        return manufactured_sources_list
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
            t::mk_float, r::coordinate, z::coordinate, vperp::coordinate, vpa::coordinate,
            composition::species_composition, geometry::geometry_input,
            collisions::collisions_input, advance::Union{advance_info,Nothing})

    Create arrays filled with manufactured rhs.

    Returns
    -------
    rhs_ion, rhs_neutral
    """
    function manufactured_rhs_as_array(
        t::mk_float, r::coordinate, z::coordinate, vperp::coordinate, vpa::coordinate,
        vzeta::coordinate, vr::coordinate, vz::coordinate,
        composition::species_composition, geometry::geometry_input,
        collisions::collisions_input, advance::Union{advance_info,Nothing})

        rhs_ion_func, rhs_neutral_func = manufactured_rhs(r.L, z.L, vpa.L, vperp.L, r.bc, z.bc, composition, geometry, collisions, r.n, advance)

        rhs_ion = allocate_float(vpa.n, vperp.n, z.n, r.n)

        for ir ∈ 1:r.n, iz ∈ 1:z.n
            for ivperp ∈ 1:vperp.n, ivpa ∈ 1:vpa.n
                rhs_ion[ivpa,ivperp,iz,ir] =
                    rhs_ion_func(vpa.grid[ivpa], vperp.grid[ivperp], z.grid[iz],
                                 r.grid[ir], t)
            end
        end

        rhs_neutral = allocate_float(vz.n, vr.n, vzeta.n, z.n, r.n)

        for ir ∈ 1:r.n, iz ∈ 1:z.n
            for ivz ∈ 1:vz.n, ivr ∈ 1:vr.n, ivzeta ∈ 1:vzeta.n
                rhs_neutral[ivz,ivr,ivzeta,iz,ir] =
                    rhs_neutral_func(vz.grid[ivz], vr.grid[ivr], vzeta.grid[ivzeta],
                                     z.grid[iz], r.grid[ir], t)
            end
        end

        return rhs_ion, rhs_neutral
    end

end
