"""
"""
module manufactured_solns

export manufactured_solutions
export manufactured_sources
export manufactured_electric_fields

using ..array_allocation: allocate_shared_float
using ..input_structs
using ..looping
using ..type_definitions: mk_float, mk_int

using Symbolics
using IfElse

    @variables r z vpa vperp t vz vr vzeta
    typed_zero(vz) = zero(vz)
    @register_symbolic typed_zero(vz)
    zero_val = 1.0e-8
    #epsilon_offset = 0.001 

    dfni_vpa_power_opt = "4" #"2"
    if dfni_vpa_power_opt == "2"
       pvpa = 2
       nconst = 0.25
       pconst = 3.0/4.0
       fluxconst = 0.5
    elseif dfni_vpa_power_opt == "4"
       pvpa = 4
       nconst = 3.0/8.0
       pconst = 15.0/8.0
       fluxconst = 1.0
    end

    #standard functions for building densities
    function nplus_sym(Lr,Lz,r_bc,z_bc,epsilon,alpha)
        if r_bc == "periodic"
            nplus = exp(sqrt(epsilon + 0.5 - z/Lz)) * exp(1.0 + 0.05*sin(2.0*pi*r/Lr)*((1.0 - alpha)*cos(pi*z/Lz) + alpha))
        elseif r_bc == "Dirichlet"
            nplus = exp(1.0 - 0.2*r/Lr) 
        end
        return nplus
    end
    
    function nminus_sym(Lr,Lz,r_bc,z_bc,epsilon,alpha)
        if r_bc == "periodic"
            nminus = exp(sqrt(epsilon + 0.5 + z/Lz)) * exp(1.0 + 0.05*sin(2.0*pi*r/Lr)*((1.0 - alpha)*cos(pi*z/Lz) + alpha))
        elseif r_bc == "Dirichlet"
            nminus = exp(1.0 - 0.2*r/Lr)
        end
        return nminus
    end
    
    function nzero_sym(Lr,Lz,r_bc,z_bc,alpha)
        if r_bc == "periodic"
            nzero = exp(1.0 + 0.05*sin(2.0*pi*r/Lr)*((1.0 - alpha)*cos(pi*z/Lz) + alpha))
        elseif r_bc == "Dirichlet" 
            nzero = exp(1.0 - 0.2*r/Lr)
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
    function densn_sym(Lr, Lz, r_bc, z_bc, geometry, composition,
                       manufactured_solns_input, species)
        if manufactured_solns_input.type == "default"
            if z_bc == "periodic"
                if r_bc == "periodic"
                    densn = 1.5 +  0.1*(cos(2.0*pi*r/Lr) + cos(2.0*pi*z/Lz)) #*sin(2.0*pi*t)
                elseif r_bc == "Dirichlet"
                    densn = 1.5 + 0.3*r/Lr
                end
            elseif z_bc == "wall"
                T_wall = composition.T_wall
                Bzed = geometry.Bzed
                Bmag = geometry.Bmag
                epsilon = manufactured_solns_input.epsilon_offset
                alpha = manufactured_solns_input.alpha_switch
                Gamma_minus = fluxconst*(Bzed/Bmag)*nminus_sym(Lr,Lz,r_bc,z_bc,epsilon,alpha)/sqrt(pi)
                Gamma_plus = fluxconst*(Bzed/Bmag)*nplus_sym(Lr,Lz,r_bc,z_bc,epsilon,alpha)/sqrt(pi)
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
        elseif manufactured_solns_input.type == "2D-instability"
            densn = 1.5 +  0.1*(cos(2.0*pi*r/Lr) + cos(2.0*pi*z/Lz))
        else
            error("Unrecognized option "
                  * "manufactured_solns:type=$(manufactured_solns_input.type)")
        end
        return densn
    end

    # neutral distribution symbolic function
    function dfnn_sym(Lr, Lz, r_bc, z_bc, geometry, composition, manufactured_solns_input,
                      species)
        densn = densn_sym(Lr, Lz, r_bc, z_bc, geometry, composition,
                          manufactured_solns_input, species)
        if z_bc == "periodic"
            dfnn = densn * exp( - vz^2 - vr^2 - vzeta^2)
        elseif z_bc == "wall"
            Hplus = 0.5*(sign(vz) + 1.0)
            Hminus = 0.5*(sign(-vz) + 1.0)
            FKw = knudsen_cosine(composition)
            Bzed = geometry.Bzed
            Bmag = geometry.Bmag
            epsilon = manufactured_solns_input.epsilon_offset
            alpha = manufactured_solns_input.alpha_switch
            Gamma_minus = fluxconst*(Bzed/Bmag)*nminus_sym(Lr,Lz,r_bc,z_bc,epsilon,alpha)/sqrt(pi)
            Gamma_plus = fluxconst*(Bzed/Bmag)*nplus_sym(Lr,Lz,r_bc,z_bc,epsilon,alpha)/sqrt(pi)
            dfnn = Hplus *( Gamma_minus*( 0.5 - z/Lz)^2 + 1.0 )*FKw + Hminus*( Gamma_plus*( 0.5 + z/Lz)^2 + 1.0 )*FKw 
        end
        return dfnn
    end
    function gyroaveraged_dfnn_sym(Lr, Lz, r_bc, z_bc, geometry, composition,
                                   manufactured_solns_input, species)
        densn = densn_sym(Lr, Lz, r_bc, z_bc, geometry, composition,
                          manufactured_solns_input, species)
        #if (r_bc == "periodic" && z_bc == "periodic")
            dfnn = densn * exp( - vpa^2 - vperp^2 )
        #end
        return dfnn
    end
    
    # ion density symbolic function
    function densi_sym(Lr, Lz, r_bc, z_bc, composition, manufactured_solns_input, species)
        if manufactured_solns_input.type == "default"
            if z_bc == "periodic"
                if r_bc == "periodic"
                    densi = 1.5 +  0.1*(sin(2.0*pi*r/Lr) + sin(2.0*pi*z/Lz))#*sin(2.0*pi*t)
                elseif r_bc == "Dirichlet"
                    #densi = 1.0 +  0.5*sin(2.0*pi*z/Lz)*(r/Lr + 0.5) + 0.2*sin(2.0*pi*r/Lr)*sin(2.0*pi*t)
                    #densi = 1.0 +  0.5*sin(2.0*pi*z/Lz)*(r/Lr + 0.5) + sin(2.0*pi*r/Lr)*sin(2.0*pi*t)
                    densi = 1.0 +  0.5*(r/Lr)*sin(2.0*pi*z/Lz)
                end
            elseif z_bc == "wall"
                epsilon = manufactured_solns_input.epsilon_offset
                alpha = manufactured_solns_input.alpha_switch
                densi = nconst*(0.5 - z/Lz)*nminus_sym(Lr,Lz,r_bc,z_bc,epsilon,alpha) + nconst*(z/Lz + 0.5)*nplus_sym(Lr,Lz,r_bc,z_bc,epsilon,alpha) + (z/Lz + 0.5)*(0.5 - z/Lz)*nzero_sym(Lr,Lz,r_bc,z_bc,alpha)  #+  0.5*(r/Lr + 0.5) + 0.5*(z/Lz + 0.5)
            end
        elseif manufactured_solns_input.type == "2D-instability"
            # Input for instability test
            background_wavenumber = 1 + round(mk_int,
                                              species.z_IC.temperature_phase)
            initial_density = species.initial_density
            density_amplitude = species.z_IC.density_amplitude
            density_phase = species.z_IC.density_phase
            T0 = Ti_sym(Lr, Lz, r_bc, z_bc, composition, manufactured_solns_input,
                        species)
            eta0 = (initial_density
                    * (1.0 + density_amplitude
                       * sin(2.0*π*background_wavenumber*z/Lz
                             + density_phase)))
            densi = eta0^((T0/(1+T0)))
        else
            error("Unrecognized option "
                  * "manufactured_solns:type=$(manufactured_solns_input.type)")
        end
        return densi
    end

    function Ti_sym(Lr, Lz, r_bc, z_bc, composition, manufactured_solns_input, species)
        if manufactured_solns_input.type == "default"
            error("Ti_sym() is not used for the default case")
        elseif manufactured_solns_input.type == "2D-instability"
            background_wavenumber = 1 + round(mk_int,
                                              species.z_IC.temperature_phase)
            initial_temperature = species.initial_temperature
            temperature_amplitude = species.z_IC.temperature_amplitude
            T0 = (initial_temperature
                  * (1.0 + temperature_amplitude
                     * sin(2.0*π*background_wavenumber*z/Lz)
                    ))
        else
            error("Unrecognized option "
                  * "manufactured_solns:type=$(manufactured_solns_input.type)")
        end
    end
 
    # ion mean parallel flow symbolic function 
    function upari_sym(Lr,Lz,r_bc,z_bc,composition,geometry,nr,manufactured_solns_input,species)
        if z_bc == "periodic"
            upari = 0.0
        elseif z_bc == "wall"
            densi = densi_sym(Lr,Lz,r_bc,z_bc,composition,manufactured_solns_input,species)
            Er, Ez, phi = electric_fields(Lr,Lz,r_bc,z_bc,composition,nr,manufactured_solns_input,species)
            rhostar = geometry.rhostar
            bzed = geometry.bzed
            epsilon = manufactured_solns_input.epsilon_offset
            alpha = manufactured_solns_input.alpha_switch
            upari =  ( (fluxconst/(sqrt(pi)*densi))*((z/Lz + 0.5)*nplus_sym(Lr,Lz,r_bc,z_bc,epsilon,alpha) 
                     - (0.5 - z/Lz)*nminus_sym(Lr,Lz,r_bc,z_bc,epsilon,alpha)) 
                     + alpha*(rhostar/(2.0*bzed))*Er )
        end
        return upari
    end
    
    # ion parallel pressure symbolic function 
    function ppari_sym(Lr,Lz,r_bc,z_bc,composition,manufactured_solns_input,species)
        # normalisation factor due to pressure normalisation convention in master pref = nref mref cref^2
        norm_fac = 0.5
        if z_bc == "periodic"
            densi = densi_sym(Lr,Lz,r_bc,z_bc,composition,manufactured_solns_input,species)
            ppari = densi
        elseif z_bc == "wall"
            densi = densi_sym(Lr,Lz,r_bc,z_bc,composition,manufactured_solns_input,species)
            epsilon = manufactured_solns_input.epsilon_offset
            alpha = manufactured_solns_input.alpha_switch
            ppari = ( pconst*((0.5 - z/Lz)*nminus_sym(Lr,Lz,r_bc,z_bc,epsilon,alpha) 
                      + (z/Lz + 0.5)*nplus_sym(Lr,Lz,r_bc,z_bc,epsilon,alpha)) 
                      + (z/Lz + 0.5)*(0.5 - z/Lz)*nzero_sym(Lr,Lz,r_bc,z_bc,alpha)  
                      - (2.0/(pi*densi))*((z/Lz + 0.5)*nplus_sym(Lr,Lz,r_bc,z_bc,epsilon,alpha) 
                      - (0.5 - z/Lz)*nminus_sym(Lr,Lz,r_bc,z_bc,epsilon,alpha))^2 )
        end
        return ppari*norm_fac
    end
    
    # ion perpendicular pressure symbolic function 
    function pperpi_sym(Lr,Lz,r_bc,z_bc,composition,manufactured_solns_input,species,nvperp)
        densi = densi_sym(Lr,Lz,r_bc,z_bc,composition,manufactured_solns_input,species)
        normfac = 0.5 # if pressure normalised to 0.5* nref * Tref = mref cref^2
        #normfac = 1.0 # if pressure normalised to nref*Tref
        if nvperp > 1
            pperpi = densi # simple vperp^2 dependence of dfni
        else
            pperpi = 0.0 # marginalised model has nvperp = 1, vperp[1] = 0
        end
        return pperpi*normfac
    end
    
    # ion thermal speed symbolic function 
    function vthi_sym(Lr,Lz,r_bc,z_bc,composition,manufactured_solns_input,species,nvperp)
        densi = densi_sym(Lr,Lz,r_bc,z_bc,composition,manufactured_solns_input,species)
        ppari = ppari_sym(Lr,Lz,r_bc,z_bc,composition,manufactured_solns_input,species)
        pperpi = pperpi_sym(Lr,Lz,r_bc,z_bc,composition,manufactured_solns_input,species,nvperp)
        isotropic_pressure = (1.0/3.0)*(ppari + 2.0*pperpi)
        normfac = 2.0 # if pressure normalised to 0.5* nref * Tref = mref cref^2
        #normfac = 1.0 # if pressure normalised to nref*Tref
        if nvperp > 1
            vthi = sqrt(normfac*isotropic_pressure/densi) # thermal speed definition of 2V model
        else
            vthi = sqrt(normfac*ppari/densi) # thermal speed definition of 1V model
        end
        return vthi
    end
    
    function jpari_into_LHS_wall_sym(Lr,Lz,r_bc,z_bc,composition,manufactured_solns_input)
        if z_bc == "periodic"
            jpari_into_LHS_wall_sym = 0.0
        elseif z_bc == "wall"
            #appropriate for wall bc test when Er = 0 (nr == 1)
            epsilon = manufactured_solns_input.epsilon_offset
            alpha = manufactured_solns_input.alpha_switch
            jpari_into_LHS_wall_sym = -fluxconst*nminus_sym(Lr,Lz,r_bc,z_bc,epsilon,alpha)/sqrt(pi)
        end
        return jpari_into_LHS_wall_sym
    end
    
    # ion distribution symbolic function
    function dfni_sym(Lr, Lz, r_bc, z_bc, composition, geometry, nr,
                      manufactured_solns_input, species)
        densi = densi_sym(Lr, Lz, r_bc, z_bc, composition, manufactured_solns_input,
                          species)

        if manufactured_solns_input.type == "default"
            # calculate the electric fields and the potential
            Er, Ez, phi = electric_fields(Lr, Lz, r_bc, z_bc, composition, nr,
                                          manufactured_solns_input, species)

            # get geometric/composition data
            Bzed = geometry.Bzed
            Bmag = geometry.Bmag
            rhostar = geometry.rhostar
            epsilon = manufactured_solns_input.epsilon_offset
            alpha = manufactured_solns_input.alpha_switch
            if z_bc == "periodic"
                dfni = densi * exp( - vpa^2 - vperp^2)
            elseif z_bc == "wall"
                vpabar = vpa - alpha*(rhostar/2.0)*(Bmag/Bzed)*Er # for alpha = 1.0, effective velocity in z direction * (Bmag/Bzed)
                Hplus = 0.5*(sign(vpabar) + 1.0)
                Hminus = 0.5*(sign(-vpabar) + 1.0)
                ffa =  exp(- vperp^2)
                dfni = ffa * ( nminus_sym(Lr,Lz,r_bc,z_bc,epsilon,alpha)* (0.5 - z/Lz) * Hminus * vpabar^pvpa + nplus_sym(Lr,Lz,r_bc,z_bc,epsilon,alpha)*(z/Lz + 0.5) * Hplus * vpabar^pvpa + nzero_sym(Lr,Lz,r_bc,z_bc,alpha)*(z/Lz + 0.5)*(0.5 - z/Lz) ) * exp( - vpabar^2 )
            end
        elseif manufactured_solns_input.type == "2D-instability"
            # Input for instability test
            T0 = Ti_sym(Lr, Lz, r_bc, z_bc, composition, manufactured_solns_input,
                        species)
            vth = sqrt(T0)

            # Note this is for a '1V' test
            dfni = densi/vth * exp(-(vpa/vth)^2)
        else
            error("Unrecognized option "
                  * "manufactured_solns:type=$(manufactured_solns_input.type)")
        end
        return dfni
    end
    function cartesian_dfni_sym(Lr, Lz, r_bc, z_bc, composition, manufactured_solns_input,
                                species)
        densi = densi_sym(Lr, Lz, r_bc, z_bc, composition, manufactured_solns_input,
                          species)
        #if (r_bc == "periodic" && z_bc == "periodic") || (r_bc == "Dirichlet" && z_bc == "periodic")
            dfni = densi * exp( - vz^2 - vr^2 - vzeta^2) 
        #end
        return dfni
    end

    function electric_fields(Lr, Lz, r_bc, z_bc, composition, nr,
                             manufactured_solns_input, species)
       
        # define derivative operators
        Dr = Differential(r) 
        Dz = Differential(z) 
 
        # get N_e factor for boltzmann response
        if composition.electron_physics == boltzmann_electron_response_with_simple_sheath && nr == 1 
            # so 1D MMS test with 3V neutrals where ion current can be calculated prior to knowing Er
            jpari_into_LHS_wall = jpari_into_LHS_wall_sym(Lr, Lz, r_bc, z_bc,
                                                          manufactured_solns_input)
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
            rfac = 0.0
        end
        
        densi = densi_sym(Lr, Lz, r_bc, z_bc, composition, manufactured_solns_input,
                          species)
        # calculate the electric fields
        dense = densi # get the electron density via quasineutrality with Zi = 1
        phi = composition.T_e*log(dense/N_e) # use the adiabatic response of electrons for me/mi -> 0
        Er = -Dr(phi)*rfac + composition.Er_constant
        Ez = -Dz(phi)
        
        Er_expanded = expand_derivatives(Er)
        Ez_expanded = expand_derivatives(Ez)
       
        return Er_expanded, Ez_expanded, phi
    end

    function manufactured_solutions(manufactured_solns_input, Lr, Lz, r_bc, z_bc,
                                    geometry, composition, species, nr, nvperp)
        charged_species = species.charged[1]
        if composition.n_neutral_species > 0
            neutral_species = species.neutral[1]
        else
            neutral_species = nothing
        end

        densi = densi_sym(Lr, Lz, r_bc, z_bc, composition, manufactured_solns_input,
                          charged_species)
        upari = upari_sym(Lr, Lz, r_bc, z_bc, composition, geometry, nr, manufactured_solns_input,
                          charged_species)
        ppari = ppari_sym(Lr, Lz, r_bc, z_bc, composition, manufactured_solns_input,
                          charged_species)
        pperpi = pperpi_sym(Lr, Lz, r_bc, z_bc, composition, manufactured_solns_input,
                          charged_species, nvperp)
        vthi = vthi_sym(Lr, Lz, r_bc, z_bc, composition, manufactured_solns_input,
                          charged_species, nvperp)
        dfni = dfni_sym(Lr, Lz, r_bc, z_bc, composition, geometry, nr,
                        manufactured_solns_input, charged_species)

        densn = densn_sym(Lr, Lz, r_bc, z_bc, geometry,composition,
                          manufactured_solns_input, neutral_species)
        dfnn = dfnn_sym(Lr, Lz, r_bc, z_bc, geometry, composition,
                        manufactured_solns_input, neutral_species)

        #build julia functions from these symbolic expressions
        # cf. https://docs.juliahub.com/Symbolics/eABRO/3.4.0/tutorials/symbolic_functions/
        densi_func = build_function(densi, z, r, t, expression=Val{false})
        upari_func = build_function(upari, z, r, t, expression=Val{false})
        ppari_func = build_function(ppari, z, r, t, expression=Val{false})
        pperpi_func = build_function(pperpi, z, r, t, expression=Val{false})
        vthi_func = build_function(vthi, z, r, t, expression=Val{false})
        densn_func = build_function(densn, z, r, t, expression=Val{false})
        dfni_func = build_function(dfni, vpa, vperp, z, r, t, expression=Val{false})
        dfnn_func = build_function(dfnn, vz, vr, vzeta, z, r, t, expression=Val{false})
        # return function
        # call like: 
        # densi_func(zval, rval, tval) 
        # dfni_func(vpaval, vperpval, zval, rval, tval) 
        # densn_func(zval, rval, tval) 
        # dfnn_func(vzval, vrval, vzetapval, zval, rval, tval) 
        
        manufactured_solns_list = (densi_func = densi_func, densn_func = densn_func, 
                                   dfni_func = dfni_func, dfnn_func = dfnn_func, 
                                   upari_func = upari_func, ppari_func = ppari_func,
                                   pperpi_func = pperpi_func, vthi_func = vthi_func)
        
        return manufactured_solns_list
    end 
    
    function manufactured_electric_fields(Lr, Lz, r_bc, z_bc, composition, nr,
                                          manufactured_solns_input, species)
        
        # calculate the electric fields and the potential
        Er, Ez, phi = electric_fields(Lr, Lz, r_bc, z_bc, composition, nr,
                                      manufactured_solns_input, species)
        
        Er_func = build_function(Er, z, r, t, expression=Val{false})
        Ez_func = build_function(Ez, z, r, t, expression=Val{false})
        phi_func = build_function(phi, z, r, t, expression=Val{false})
        
        manufactured_E_fields = (Er_func = Er_func, Ez_func = Ez_func, phi_func = phi_func)
        
        return manufactured_E_fields
    end

    function manufactured_sources(manufactured_solns_input, r_coord, z_coord, vperp_coord,
            vpa_coord, vzeta_coord, vr_coord, vz_coord, composition, geometry, collisions,
            num_diss_params, species)

        charged_species = species.charged[1]
        if composition.n_neutral_species > 0
            neutral_species = species.neutral[1]
        else
            neutral_species = nothing
        end

        # ion manufactured solutions
        densi = densi_sym(r_coord.L, z_coord.L, r_coord.bc, z_coord.bc, composition,
                          manufactured_solns_input, charged_species)
        upari = upari_sym(r_coord.L, z_coord.L, r_coord.bc, z_coord.bc, composition, geometry, r_coord.n, manufactured_solns_input, charged_species)
        vthi = vthi_sym(r_coord.L, z_coord.L, r_coord.bc, z_coord.bc, composition, manufactured_solns_input,
                          charged_species, vperp_coord.n)
        dfni = dfni_sym(r_coord.L, z_coord.L, r_coord.bc, z_coord.bc, composition,
                        geometry, r_coord.n, manufactured_solns_input, charged_species)
        #dfni in vr vz vzeta coordinates
        vrvzvzeta_dfni = cartesian_dfni_sym(r_coord.L, z_coord.L, r_coord.bc, z_coord.bc,
                                            composition, manufactured_solns_input,
                                            charged_species)

        # neutral manufactured solutions
        densn = densn_sym(r_coord.L,z_coord.L, r_coord.bc, z_coord.bc, geometry,
                          composition, manufactured_solns_input, neutral_species)
        dfnn = dfnn_sym(r_coord.L, z_coord.L, r_coord.bc, z_coord.bc, geometry,
                        composition, manufactured_solns_input, neutral_species)
        # gyroaverage < dfnn > in vpa vperp coordinates
        gav_dfnn = gyroaveraged_dfnn_sym(r_coord.L, z_coord.L, r_coord.bc, z_coord.bc,
                                         geometry, composition, manufactured_solns_input,
                                         neutral_species)

        dense = densi # get the electron density via quasineutrality with Zi = 1

        # define derivative operators
        Dr = Differential(r) 
        Dz = Differential(z) 
        Dvpa = Differential(vpa) 
        Dvperp = Differential(vperp) 
        Dvz = Differential(vz) 
        Dt = Differential(t) 

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
        if r_coord.n > 1 # keep radial derivatives
            rfac = 1.0
        else      # drop radial derivative terms
            rfac = 0.0
        end
        
        # calculate the electric fields and the potential
        Er, Ez, phi = electric_fields(r_coord.L, z_coord.L, r_coord.bc, z_coord.bc,
                                      composition, r_coord.n, manufactured_solns_input,
                                      charged_species)

        # the ion source to maintain the manufactured solution
        Si = ( Dt(dfni) + ( vpa * (Bzed/Bmag) - 0.5*rhostar*Er ) * Dz(dfni) + ( 0.5*rhostar*Ez*rfac ) * Dr(dfni) + ( 0.5*Ez*Bzed/Bmag ) * Dvpa(dfni)
               + cx_frequency*( densn*dfni - densi*gav_dfnn )  - ionization_frequency*dense*gav_dfnn)
        nu_krook = collisions.krook_collision_frequency_prefactor
        if nu_krook > 0.0
            Ti_over_Tref = vthi^2
            if collisions.krook_collisions_option == "manual"
                nuii_krook = nu_krook
            else # default option
                nuii_krook = nu_krook * densi * Ti_over_Tref^(-1.5)
            end
            if vperp_coord.n > 1
                pvth  = 3
            else 
                pvth = 1
            end
            FMaxwellian = (densi/vthi^pvth)*exp( -( ( vpa-upari)^2 + vperp^2 )/vthi^2)
            Si += - nuii_krook*(FMaxwellian - dfni)
        end
        include_num_diss_in_MMS = true
        if num_diss_params.vpa_dissipation_coefficient > 0.0 && include_num_diss_in_MMS
            Si += - num_diss_params.vpa_dissipation_coefficient*Dvpa(Dvpa(dfni))
        end
        if num_diss_params.vperp_dissipation_coefficient > 0.0 && include_num_diss_in_MMS
            Si += - num_diss_params.vperp_dissipation_coefficient*Dvperp(Dvperp(dfni))
        end
        if num_diss_params.r_dissipation_coefficient > 0.0 && include_num_diss_in_MMS
            Si += - rfac*num_diss_params.r_dissipation_coefficient*Dr(Dr(dfni))
        end
        if num_diss_params.z_dissipation_coefficient > 0.0 && include_num_diss_in_MMS
            Si += - num_diss_params.z_dissipation_coefficient*Dz(Dz(dfni))
        end

        Source_i = expand_derivatives(Si)
        
        # the neutral source to maintain the manufactured solution
        Sn = Dt(dfnn) + vz * Dz(dfnn) + rfac*vr * Dr(dfnn) + cx_frequency* (densi*dfnn - densn*vrvzvzeta_dfni) + ionization_frequency*dense*dfnn
        if num_diss_params.vz_dissipation_coefficient > 0.0 && include_num_diss_in_MMS
            Sn += - num_diss_params.vz_dissipation_coefficient*Dvz(Dvz(dfnn))
        end
        if num_diss_params.r_dissipation_coefficient > 0.0 && include_num_diss_in_MMS
            Sn += - rfac*num_diss_params.r_dissipation_coefficient*Dr(Dr(dfnn))
        end
        if num_diss_params.z_dissipation_coefficient > 0.0 && include_num_diss_in_MMS
            Sn += - num_diss_params.z_dissipation_coefficient*Dz(Dz(dfnn))
        end
        
        Source_n = expand_derivatives(Sn)
        
        Source_i_func = build_function(Source_i, vpa, vperp, z, r, t, expression=Val{false})
        Source_n_func = build_function(Source_n, vz, vr, vzeta, z, r, t, expression=Val{false})
        
        if expand_derivatives(Dt(Source_i)) == 0 && expand_derivatives(Dt(Source_n)) == 0
            # Time independent, so store arrays instead of functions

            Source_i_array = allocate_shared_float(vpa_coord.n,vperp_coord.n,z_coord.n,r_coord.n)
            begin_s_r_z_region()
            @loop_s is begin
                if is == 1
                    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
                        Source_i_array[ivpa,ivperp,iz,ir,is] = Source_i_func(vpa_coord.grid[ivpa],vperp_coord.grid[ivperp],z_coord.grid[iz],r_coord.grid[ir],0.0)
                    end
                end
            end

            if composition.n_neutral_species > 0
                Source_n_array = allocate_shared_float(vz_coord.n,vr_coord.n,vzeta_coord.n,z_coord.n,r_coord.n)
                begin_sn_r_z_region()
                @loop_sn isn begin
                    if isn == 1
                        @loop_r_z_vzeta_vr_vz ir iz ivzeta ivr ivz begin
                            Source_n_array[ivz,ivr,ivzeta,iz,ir,isn] = Source_n_func(vz_coord.grid[ivz],vr_coord.grid[ivr],vzeta_coord.grid[ivzeta],z_coord.grid[iz],r_coord.grid[ir],0.0)
                        end
                    end
                end
            else
                Source_n_array = zeros(mk_float,0)
            end

            manufactured_sources_list = (time_independent_sources = true, Source_i_array = Source_i_array, Source_n_array = Source_n_array)
        else
            manufactured_sources_list = (time_independent_sources = false, Source_i_func = Source_i_func, Source_n_func = Source_n_func)
        end
        
        return manufactured_sources_list
    end 
    
end
