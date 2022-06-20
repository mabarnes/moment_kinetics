"""
"""
module manufactured_solns

export manufactured_solutions
export manufactured_sources
export manufactured_electric_fields
export manufactured_solutions_as_arrays
export manufactured_rhs_as_array

using SpecialFunctions
using Symbolics
using IfElse
using ..input_structs
using ..array_allocation: allocate_float
using ..coordinates: coordinate
using ..input_structs: geometry_input, advance_info, species_composition, collisions_input
using ..type_definitions

    @variables r z vpa vperp t vz vr vzeta
    typed_zero(vz) = zero(vz)
    @register_symbolic typed_zero(vz)
    zero_val = 1.0e-8
    
    #standard functions for building densities
    function nplus_sym(Lr,Lz,r_bc,z_bc)
        if r_bc == "periodic"
            nplus = 1.0 + 0.05*sin(2.0*pi*r/Lr)*cos(pi*z/Lz)
        elseif r_bc == "Dirichlet"
            nplus = 1.0 - 0.2*r/Lr 
        end
        return nplus
    end
    
    function nminus_sym(Lr,Lz,r_bc,z_bc)
        if r_bc == "periodic"
            nminus = 1.0 + 0.05*sin(2.0*pi*r/Lr)*cos(pi*z/Lz)
        elseif r_bc == "Dirichlet"
            nminus = 1.0 - 0.2*r/Lr
        end
        return nminus
    end
    
    function nzero_sym(Lr,Lz,r_bc,z_bc)
        if r_bc == "periodic"
            nzero = 1.0 + 0.05*sin(2.0*pi*r/Lr)*cos(pi*z/Lz) # 1.0 #+ (r/Lr + 0.5)*(0.5 - r/Lr)
        elseif r_bc == "Dirichlet" 
            nzero = 1.0 - 0.2*r/Lr
        end
        return nzero
    end

    function knudsen_cosine(composition)
        T_wall = composition.T_wall
        exponetial = exp( - (vz^2 + vr^2 + vzeta^2)/T_wall )
        if composition.use_test_neutral_wall_pdf
            #test dfn
            knudsen_pdf = (4.0/T_wall^(5.0/2.0))*abs(vz)*exponetial
        else
            #proper Knudsen dfn
            # prefac here may cause problems with NaNs if vz = vr = vzeta = 0 is on grid
            fac = abs(vz)/sqrt(vz^2 + vr^2 + vzeta^2)
            prefac = IfElse.ifelse( abs(vz) < 1000.0*zero_val,typed_zero(vz),fac)
            knudsen_pdf = (3.0*sqrt(pi)/T_wall^2)*prefac*exponetial
        end
        return knudsen_pdf
    end

    # neutral density symbolic function
    function densn_sym(Lr,Lz,r_bc,z_bc,geometry,composition)
        if z_bc == "periodic" 
            if r_bc == "periodic" 
                densn = 1.5 +  0.1*(cos(2.0*pi*r/Lr) + cos(2.0*pi*z/Lz))*cos(2.0*pi*t)  
            elseif r_bc == "Dirichlet"
                densn = 1.5 + 0.3*r/Lr
            end
        elseif z_bc == "wall"
            T_wall = composition.T_wall
            Bzed = geometry.Bzed
            Bmag = geometry.Bmag
            Gamma_minus = 0.5*(Bzed/Bmag)*nminus_sym(Lr,Lz,r_bc,z_bc)/sqrt(pi)
            Gamma_plus = 0.5*(Bzed/Bmag)*nplus_sym(Lr,Lz,r_bc,z_bc)/sqrt(pi)
            # exact integral of corresponding dfnn below
            if composition.use_test_neutral_wall_pdf
                #test 
                prefactor = 2.0/sqrt(pi*T_wall)
            else
                #proper prefactor
                prefactor = 3.0*sqrt(pi)/(4.0*sqrt(T_wall))
            end
            densn = prefactor*(Gamma_minus*(0.5 - z/Lz)^2 + Gamma_plus*(0.5 + z/Lz)^2 + 2.0 )
        else
            densn = 1.0
        end
        return densn
    end

    # neutral distribution symbolic function
    function dfnn_sym(Lr,Lz,r_bc,z_bc,geometry,composition)
        densn = densn_sym(Lr,Lz,r_bc,z_bc,geometry,composition)
        if z_bc == "periodic"
            dfnn = densn * exp( - vz^2 - vr^2 - vzeta^2)
        elseif z_bc == "wall"
            Hplus = 0.5*(sign(vz) + 1.0)
            Hminus = 0.5*(sign(-vz) + 1.0)
            FKw = knudsen_cosine(composition)
            Bzed = geometry.Bzed
            Bmag = geometry.Bmag
            Gamma_minus = 0.5*(Bzed/Bmag)*nminus_sym(Lr,Lz,r_bc,z_bc)/sqrt(pi)
            Gamma_plus = 0.5*(Bzed/Bmag)*nplus_sym(Lr,Lz,r_bc,z_bc)/sqrt(pi)
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
    
    function densi_sym(Lr,Lz,r_bc,z_bc)
        # Note: explicitly convert numerical factors to mk_float so the output gets full
        # precision if we use quad-precision (Float128)
        if z_bc == "periodic"
            if r_bc == "periodic"
               densi = 1.5 + (sin(2.0*pi*r/Lr) + sin(2.0*pi*z/Lz))/10.0*cos(2.0*pi*t)
            elseif r_bc == "Dirichlet" 
                #densi = 1 +  1//2*sin(2*pi*z/Lz)*(r/Lr + 1//2) + 0.2*sin(2*pi*r/Lr)*sin(2*pi*t)
                #densi = 1 +  1//2*sin(2*pi*z/Lz)*(r/Lr + 1//2) + sin(2*pi*r/Lr)*sin(2*pi*t)
                densi = 1.0 +  0.5*(r/Lr + 0.5)/2.0
            end
        elseif z_bc == "wall"
            densi = 0.25*(0.5 - z/Lz)*nminus_sym(Lr,Lz,r_bc,z_bc) + 0.25*(z/Lz + 0.5)*nplus_sym(Lr,Lz,r_bc,z_bc) + (z/Lz + 0.5)*(0.5 - z/Lz)*nzero_sym(Lr,Lz,r_bc,z_bc)  #+  0.5*(r/Lr + 0.5) + 0.5*(z/Lz + 0.5)
        else
            error("Unsupported options r_bc=$r_bc, z_bc=$z_bc")
        end
        return densi
    end

    function jpari_into_LHS_wall_sym(Lr,Lz,r_bc,z_bc)
        if z_bc == "periodic"
            jpari_into_LHS_wall_sym = 0.0
        elseif z_bc == "wall"
            #appropriate for wall bc test when Er = 0 (nr == 1)
            jpari_into_LHS_wall_sym = -0.5*nminus_sym(Lr,Lz,r_bc,z_bc)/sqrt(pi)
        end
        return jpari_into_LHS_wall_sym
    end
    
    # ion distribution symbolic function
    function dfni_sym(Lr,Lz,r_bc,z_bc,composition,geometry,nr)
        # Note: explicitly convert numerical factors to mk_float so the output gets full
        # precision if we use quad-precision (Float128)
        densi = densi_sym(Lr,Lz,r_bc,z_bc)
        
        # calculate the electric fields and the potential
        Er, Ez, phi = electric_fields(Lr,Lz,r_bc,z_bc,composition,nr)
        
        # get geometric/composition data
        Bzed = geometry.Bzed
        Bmag = geometry.Bmag
        rhostar = geometry.rhostar
        
        if z_bc == "periodic"
            #MHversion dfni = densi * exp( - vpa^2 - vperp^2) 
            ## This version with upar is very expensive to evaluate, probably due to
            ## spatially-varying erf()?
            #upar = (sin(2.0*pi*r/Lr) + sin(2*pi*z/Lz))/10.0
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
            #dfni = (1.0 - (2.0*vpa/Lvpa)^2) * dfni *
            #          2.0*Lvpa / ((Lvpa^2-4.0*upar^2-2)*(erf(Lvpa/2.0-upar)+erf(Lvpa/2.0+upar)) + 2.0/sqrt(π)*exp(-(Lvpa+2.0*upar)^2/4.0)*(exp(2.0*Lvpa*upar)+Lvpa-2.0*upar)) / # vpa normalization
            #         (1.0 - exp(-Lvperp^2)) # vperp normalisation

            # Ad-hoc odd-in-vpa component of dfni which gives non-zero upar and dfni
            # positive everywhere, while the odd component integrates to 0 so does not
            # need to be accounted for in normalisation.
            dfni = densi * (exp(- vpa^2 - vperp^2)
                            + (sin(2.0*pi*r/Lr) + sin(2*pi*z/Lz))/10.0
                              * vpa * exp(-2.0*(vpa^2+vperp^2))) #/ pi^1.5
        elseif z_bc == "wall"
            vpabar = vpa - (rhostar/2.0)*(Bmag/Bzed)*Er # effective velocity in z direction * (Bmag/Bzed)
            Hplus = 0.5*(sign(vpabar) + 1.0)
            Hminus = 0.5*(sign(-vpabar) + 1.0)
            ffa =  exp(- vperp^2)
            dfni = ffa * ( nminus_sym(Lr,Lz,r_bc,z_bc)* (0.5 - z/Lz) * Hminus * vpabar^2 + nplus_sym(Lr,Lz,r_bc,z_bc)*(z/Lz + 0.5) * Hplus * vpabar^2 + nzero_sym(Lr,Lz,r_bc,z_bc)*(z/Lz + 0.5)*(0.5 - z/Lz) ) * exp( - vpabar^2 )
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

    function electric_fields(Lr,Lz,r_bc,z_bc,composition,nr)
       
        # define derivative operators
        Dr = Differential(r) 
        Dz = Differential(z) 
 
        # get N_e factor for boltzmann response
        if composition.electron_physics == boltzmann_electron_response_with_simple_sheath && nr == 1 
            # so 1D MMS test with 3V neutrals where ion current can be calculated prior to knowing Er
            jpari_into_LHS_wall = jpari_into_LHS_wall_sym(Lr,Lz,r_bc,z_bc)
            N_e = -2.0*sqrt(pi*composition.me_over_mi)*exp(-composition.phi_wall/composition.T_e)*jpari_into_LHS_wall
        elseif composition.electron_physics == boltzmann_electron_response_with_simple_sheath && nr > 1 
            println("ERROR: simple sheath MMS test not supported for nr > 1")
            println("INFO: In general, not possible to analytically calculate jpari for sheath prior to knowing Er, but Er depends on jpari")
            println("Setting N_e = 1.0. Expect MMS test to fail!")
            N_e = 1.0
        elseif composition.electron_physics == boltzmann_electron_response 
            # all other cases
            # N_e equal to reference density 
            N_e = 1.0
        end 
        
        if nr > 1 # keep radial electric field
            rfac = 1.0
        else      # drop radial electric field
            rfac = 0
        end
        
        densi = densi_sym(Lr,Lz,r_bc,z_bc)
        # calculate the electric fields
        dense = densi # get the electron density via quasineutrality with Zi = 1
        phi = composition.T_e*log(dense/N_e) # use the adiabatic response of electrons for me/mi -> 0
        Er = -Dr(phi)*rfac
        Ez = -Dz(phi)
        
        Er_expanded = expand_derivatives(Er)
        Ez_expanded = expand_derivatives(Ez)
       
        return Er_expanded, Ez_expanded, phi
    end

    function manufactured_solutions(Lr,Lz,r_bc,z_bc,geometry,composition,nr)
        densi = densi_sym(Lr,Lz,r_bc,z_bc)
        dfni = dfni_sym(Lr,Lz,r_bc,z_bc,composition,geometry,nr)
        
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

    function manufactured_electric_fields(Lr,Lz,r_bc,z_bc,composition,nr)
        
        # calculate the electric fields and the potential
        Er, Ez, phi = electric_fields(Lr,Lz,r_bc,z_bc,composition,nr)
        
        Er_func = build_function(Er, z, r, t, expression=Val{false})
        Ez_func = build_function(Ez, z, r, t, expression=Val{false})
        phi_func = build_function(phi, z, r, t, expression=Val{false})
        
        manufactured_E_fields = (Er_func = Er_func, Ez_func = Ez_func, phi_func = phi_func)
        
        return manufactured_E_fields
    end 

    function manufactured_rhs_sym(Lr::mk_float,Lz::mk_float,r_bc::String,z_bc::String,
                                  composition::species_composition,geometry::geometry_input,collisions::collisions_input,
                                  nr::mk_int,advance::Union{advance_info,Nothing}=nothing)
        # Note: explicitly convert numerical factors to mk_float so the output gets full
        # precision if we use quad-precision (Float128)

        # ion manufactured solutions
        densi = densi_sym(Lr,Lz,r_bc,z_bc)
        dfni = dfni_sym(Lr,Lz,r_bc,z_bc,composition,geometry,nr)
        vrvzvzeta_dfni = cartesian_dfni_sym(Lr,Lz,r_bc,z_bc) #dfni in vr vz vzeta coordinates
        
        # neutral manufactured solutions
        densn = densn_sym(Lr,Lz,r_bc,z_bc,geometry,composition)
        dfnn = dfnn_sym(Lr,Lz,r_bc,z_bc,geometry,composition)
        gav_dfnn = gyroaveraged_dfnn_sym(Lr,Lz,r_bc,z_bc,geometry,composition) # gyroaverage < dfnn > in vpa vperp coordinates
        
        dense = densi # get the electron density via quasineutrality with Zi = 1
        
        # define derivative operators
        Dr = Differential(r) 
        Dz = Differential(z) 
        Dvpa = Differential(vpa) 
        Dvperp = Differential(vperp) 
    
        # get geometric/composition data
        Bzed = geometry.Bzed
        Bmag = geometry.Bmag
        rhostar = geometry.rhostar
        #exceptions for cases with missing terms 
        if composition.n_neutral_species > 0
            cx_frequency = collisions.charge_exchange
            ionization_frequency = collisions.ionization
        else 
            cx_frequency = 0.0
            ionization_frequency = 0.0
        end
        if nr > 1 # keep radial derivatives
            rfac = 1.0
        else      # drop radial derivative terms
            rfac = 0
        end
        
        # calculate the electric fields and the potential
        Er, Ez, phi = electric_fields(Lr,Lz,r_bc,z_bc,composition,nr)
        
        rhs_ion = 0 * z
        if advance === nothing || advance.vpa_advection
            rhs_ion += - ( 0.5*Ez*Bzed/Bmag ) * Dvpa(dfni)
        end
        if advance === nothing || advance.z_advection
            rhs_ion += -( vpa * (Bzed/Bmag) - 0.5*rhostar*Er ) * Dz(dfni)
        end
        if advance === nothing || advance.r_advection
            rhs_ion += - ( 0.5*rhostar*Ez ) * Dr(dfni)
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

    function manufactured_rhs(Lr::mk_float, Lz::mk_float, r_bc::String, z_bc::String,
                              composition::species_composition, geometry::geometry_input,
                              collisions::collisions_input, nr::mk_int,
                              advance::Union{advance_info,Nothing}=nothing)
        rhs_ion_sym, rhs_neutral_sym = manufactured_rhs_sym(Lr, Lz, r_bc, z_bc,
            composition, geometry, collisions, nr, advance)
        return build_function(rhs_ion_sym, vpa, vperp, z, r, t, expression=Val{false}),
               build_function(rhs_neutral_sym, vz, vr, vzeta, z, r, t, expression=Val{false})
    end

    function manufactured_sources(Lr::mk_float, Lz::mk_float, r_bc::String, z_bc::String,
                                  composition::species_composition,
                                  geometry::geometry_input, collisions::collisions_input,
                                  nr::mk_int)

        dfni = dfni_sym(Lr,Lz,r_bc,z_bc,composition,geometry,nr)
        dfnn = dfnn_sym(Lr,Lz,r_bc,z_bc,geometry,composition)
        rhs_ion, rhs_neutral = manufactured_rhs_sym(Lr,Lz,r_bc,z_bc,composition,geometry,collisions,nr)

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
            t::mk_float, r::coordinate, z::coordinate, vperp::coordinate,
            vpa::coordinate)

    Create array filled with manufactured solutions.

    Returns
    -------
    (densi, phi, dfni)
    """
    function manufactured_solutions_as_arrays(
        t::mk_float, r::coordinate, z::coordinate, vperp::coordinate,
        vpa::coordinate, vzeta::coordinate, vr::coordinate, vz::coordinate)

        dfni_func, densi_func = manufactured_solutions(r.L, z.L, vperp.L, vpa.L, vzeta.L,
                                                       vr.L, vz.L, r.bc, z.bc)

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

        rhs_ion_func, rhs_neutral_func = manufactured_rhs(r.L, z.L, r.bc, z.bc,
            composition, geometry, collisions, r.n, advance)

        rhs_ion = allocate_float(vpa.n, vperp.n, z.n, r.n)

        for ir ∈ 1:r.n, iz ∈ 1:z.n
            for ivperp ∈ 1:vperp.n, ivpa ∈ 1:vpa.n
                rhs_ion[ivpa,ivperp,iz,ir] =
                    rhs_ion_func(vpa.grid[ivpa], vperp.grid[ivperp], z.grid[iz],
                                 r.grid[ir], t)
            end
        end

        if composition.n_neutral_species > 0
            rhs_neutral = allocate_float(vz.n, vr.n, vzeta.n, z.n, r.n)

            for ir ∈ 1:r.n, iz ∈ 1:z.n
                for ivz ∈ 1:vz.n, ivr ∈ 1:vr.n, ivzeta ∈ 1:vzeta.n
                    rhs_neutral[ivz,ivr,ivzeta,iz,ir] =
                    rhs_neutral_func(vz.grid[ivz], vr.grid[ivr], vzeta.grid[ivzeta],
                                     z.grid[iz], r.grid[ir], t)
                end
            end
        else
            rhs_neutral = nothing
        end

        return rhs_ion, rhs_neutral
    end

end
