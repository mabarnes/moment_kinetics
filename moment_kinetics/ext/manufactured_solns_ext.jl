# This extension provides the actual implementation for the manufactured_solns module,
# which is available only if the Symbolics package is installed
#
# Note that if there are errors when precompiling an extension, they may not be shown by
# default. To see the error, precompile by running
# `using Pkg; Pkg.precompile(strict=true)`.
module manufactured_solns_ext

using moment_kinetics.input_structs
using moment_kinetics.looping
using moment_kinetics.moment_kinetics_structs
using moment_kinetics.type_definitions: mk_int, mk_float

import moment_kinetics.manufactured_solns: manufactured_solutions_extension_loadable,
                                           manufactured_solutions,
                                           manufactured_sources_setup,
                                           manufactured_electric_fields,
                                           manufactured_geometry

using Symbolics
using IfElse

    function manufactured_solutions_extension_loadable()
        # Used to check whether this extension is loaded.
        return true
    end

    @variables r z vpa vperp t vz vr vzeta
    const typed_zero(vz) = zero(vz)
    @register_symbolic typed_zero(vz)
    const zero_val = 1.0e-8
    #epsilon_offset = 0.001 

    const dfni_vpa_power_opt = "4" #"2"
    if dfni_vpa_power_opt == "2"
       const pvpa = 2
       const nconst = 0.25
       const pconst = 3.0/4.0
       const fluxconst = 0.5
    elseif dfni_vpa_power_opt == "4"
       const pvpa = 4
       const nconst = 3.0/8.0
       const pconst = 15.0/8.0
       const fluxconst = 1.0
    end
    
    # struct of symbolic functions for geometric coefficients
    # Note that we restrict the types of the variables in the struct
    # to be either a float or a Symbolics Num type. The Union appears
    # to be required to permit geometry options where a symbolic variable
    # does not appear in a particular geometric coefficient, because 
    # that coefficient is a constant. 
    struct geometric_coefficients_sym{}
        Er_constant::mk_float
        Ez_constant::mk_float
        rhostar::mk_float
        Bzed::Union{mk_float,Num}
        Bzeta::Union{mk_float,Num}
        Bmag::Union{mk_float,Num}
        bzed::Union{mk_float,Num}
        bzeta::Union{mk_float,Num}
        dBdz::Union{mk_float,Num}
        dBdr::Union{mk_float,Num}
        jacobian::Union{mk_float,Num}
    end
    
    function geometry_sym(geometry_input_data::geometry_input,Lz,Lr,nr)
        # define derivative operators
        Dr = Differential(r)
        Dz = Differential(z)
        # compute symbolic geometry functions
        option = geometry_input_data.option
        rhostar = geometry_input_data.rhostar
        pitch = geometry_input_data.pitch
        Er_constant = geometry_input_data.Er_constant
        Ez_constant = geometry_input_data.Ez_constant
        if option == "constant-helical" || option == "default"
            bzed = pitch
            bzeta = sqrt(1 - bzed^2)
            Bmag = 1.0
            Bzed = Bmag*bzed
            Bzeta = Bmag*bzeta
            dBdr = 0.0
            dBdz = 0.0
            jacobian = 1.0
        elseif option == "1D-mirror"
            DeltaB = geometry_input_data.DeltaB
            bzed = pitch
            bzeta = sqrt(1 - bzed^2)
            # B(z)/Bref = 1 + DeltaB*( 2(2z/L)^2 - (2z/L)^4)
            # chosen so that
            # B(z)/Bref = 1 + DeltaB at 2z/L = +- 1 
            # d B(z)d z = 0 at 2z/L = +- 1 
            zfac = 2.0*z/Lz
            Bmag = 1.0 + DeltaB*( 2.0*zfac^2 - zfac^4)
            Bzed = Bmag*bzed
            Bzeta = Bmag*bzeta
            if nr > 1
                dBdr = expand_derivatives(Dr(Bmag))
            else
                dBdr = 0.0
            end
            dBdz = expand_derivatives(Dz(Bmag))
            jacobian = 1.0
        elseif option == "0D-Spitzer-test"
            Bmag = 1.0
            dBdz = geometry_input_data.dBdz_constant
            dBdr = geometry_input_data.dBdr_constant
            bzed = pitch
            bzeta = sqrt(1 - bzed^2)
            Bzed = Bmag*bzed
            Bzeta = Bmag*bzeta
            jacobian = 1.0
        else
            input_option_error("$option", option)
        end
        geo_sym = geometric_coefficients_sym(Er_constant,Ez_constant,
          rhostar,Bzed,Bzeta,Bmag,bzed,bzeta,dBdz,dBdr,jacobian)
        return geo_sym
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
        exponential = exp( - 0.5 * (vz^2 + vr^2 + vzeta^2)/T_wall )
        if composition.use_test_neutral_wall_pdf
            #test dfn
            knudsen_pdf = (1.0/π/T_wall^(5.0/2.0))*abs(vz)*exponential
        else
            #proper Knudsen dfn
            # prefac here may cause problems with NaNs if vz = vr = vzeta = 0 is on grid
            fac = abs(vz)/sqrt(vz^2 + vr^2 + vzeta^2)
            prefac = IfElse.ifelse( abs(vz) < 1000.0*zero_val,typed_zero(vz),fac)
            knudsen_pdf = (0.75*pi/T_wall^2)*prefac*exponential
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
    function dfnn_sym(Lr, Lz, r_bc, z_bc, geometry, composition, nvzeta, nvr,
                      manufactured_solns_input, species)
        densn = densn_sym(Lr, Lz, r_bc, z_bc, geometry, composition,
                          manufactured_solns_input, species)
        if nvzeta == 1 && nvr == 1
            Maxwellian_prefactor = 1 / sqrt(π)
        else
            Maxwellian_prefactor = 1 / π^1.5
        end
        if z_bc == "periodic"
            dfnn = densn * Maxwellian_prefactor * exp( - (vz^2 + vr^2 + vzeta^2) )
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
                                   nvzeta, nvr, manufactured_solns_input, species)
        densn = densn_sym(Lr, Lz, r_bc, z_bc, geometry, composition,
                          manufactured_solns_input, species)
        if nvzeta == 1 && nvr == 1
            Maxwellian_prefactor = 1 / sqrt(π)
        else
            Maxwellian_prefactor = 1 / π^1.5
        end
        #if (r_bc == "periodic" && z_bc == "periodic")
            dfnn = densn * Maxwellian_prefactor * exp( - vpa^2 - vperp^2 )
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
            return T0
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
            Er, Ez, phi = electric_fields(Lr,Lz,r_bc,z_bc,composition,geometry,nr,manufactured_solns_input,species)
            Bzeta = geometry.Bzeta
            Bmag = geometry.Bmag
            rhostar = geometry.rhostar
            jacobian = geometry.jacobian
            ExBgeofac = rhostar*Bzeta*jacobian/Bmag^2
            bzed = geometry.bzed
            epsilon = manufactured_solns_input.epsilon_offset
            alpha = manufactured_solns_input.alpha_switch
            upari =  ( (fluxconst/(sqrt(pi)*densi))*((z/Lz + 0.5)*nplus_sym(Lr,Lz,r_bc,z_bc,epsilon,alpha) 
                     - (0.5 - z/Lz)*nminus_sym(Lr,Lz,r_bc,z_bc,epsilon,alpha)) 
                     + alpha*(ExBgeofac/bzed)*Er )
        end
        return upari
    end
    
    # ion parallel pressure symbolic function 
    function ppari_sym(Lr,Lz,r_bc,z_bc,composition,manufactured_solns_input,species)
        if manufactured_solns_input.type == "2D-instability"
            densi = densi_sym(Lr,Lz,r_bc,z_bc,composition,manufactured_solns_input,species)
            Tpari = Ti_sym(Lr, Lz, r_bc, z_bc, composition, manufactured_solns_input,
                           species)
            ppari = densi * Tpari
        elseif z_bc == "periodic"
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
        return ppari
    end
    
    # ion perpendicular pressure symbolic function 
    function pperpi_sym(Lr,Lz,r_bc,z_bc,composition,manufactured_solns_input,species,nvperp)
        densi = densi_sym(Lr,Lz,r_bc,z_bc,composition,manufactured_solns_input,species)
        if nvperp > 1
            if manufactured_solns_input.type == "2D-instability"
                Tperpi = Ti_sym(Lr, Lz, r_bc, z_bc, composition, manufactured_solns_input,
                               species)
                pperpi = densi * Tperpi
            else
                pperpi = densi # simple vperp^2 dependence of dfni
            end
        else
            pperpi = 0.0 # marginalised model has nvperp = 1, vperp[1] = 0
        end
        return pperpi
    end
    
    # ion thermal speed symbolic function 
    function vthi_sym(Lr,Lz,r_bc,z_bc,composition,manufactured_solns_input,species,nvperp)
        densi = densi_sym(Lr,Lz,r_bc,z_bc,composition,manufactured_solns_input,species)
        ppari = ppari_sym(Lr,Lz,r_bc,z_bc,composition,manufactured_solns_input,species)
        pperpi = pperpi_sym(Lr,Lz,r_bc,z_bc,composition,manufactured_solns_input,species,nvperp)
        isotropic_pressure = (1.0/3.0)*(ppari + 2.0*pperpi)
        vthi = sqrt(2.0*isotropic_pressure/densi) # thermal speed definition of 2V model
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
    function dfni_sym(Lr, Lz, r_bc, z_bc, composition, geometry, nr, nvperp,
                      manufactured_solns_input, species)
        densi = densi_sym(Lr, Lz, r_bc, z_bc, composition, manufactured_solns_input,
                          species)

        if nvperp == 1
            Maxwellian_prefactor = 1 / sqrt(π)
        else
            Maxwellian_prefactor = 1 / π^1.5
        end
        if manufactured_solns_input.type == "default"
            # calculate the electric fields and the potential
            Er, Ez, phi = electric_fields(Lr, Lz, r_bc, z_bc, composition, geometry, nr,
                                          manufactured_solns_input, species)

            # get geometric/composition data
            Bzed = geometry.Bzed
            Bzeta = geometry.Bzeta
            Bmag = geometry.Bmag
            rhostar = geometry.rhostar
            jacobian = geometry.jacobian
            ExBgeofac = rhostar*Bzeta*jacobian/Bmag^2
            epsilon = manufactured_solns_input.epsilon_offset
            alpha = manufactured_solns_input.alpha_switch
            if z_bc == "periodic"
                dfni = densi * Maxwellian_prefactor * exp( - vpa^2 - vperp^2)
            elseif z_bc == "wall"
                vpabar = vpa - alpha*ExBgeofac*(Bmag/Bzed)*Er # for alpha = 1.0, effective velocity in z direction * (Bmag/Bzed)
                Hplus = 0.5*(sign(vpabar) + 1.0)
                Hminus = 0.5*(sign(-vpabar) + 1.0)
                ffa =  1.0 * Maxwellian_prefactor * exp(- vperp^2)
                dfni = ffa * ( nminus_sym(Lr,Lz,r_bc,z_bc,epsilon,alpha)* (0.5 - z/Lz) * Hminus * vpabar^pvpa + nplus_sym(Lr,Lz,r_bc,z_bc,epsilon,alpha)*(z/Lz + 0.5) * Hplus * vpabar^pvpa + nzero_sym(Lr,Lz,r_bc,z_bc,alpha)*(z/Lz + 0.5)*(0.5 - z/Lz) ) * exp( - vpabar^2 )
            end
        elseif manufactured_solns_input.type == "2D-instability"
            # Input for instability test
            T0 = Ti_sym(Lr, Lz, r_bc, z_bc, composition, manufactured_solns_input,
                        species)
            vth = sqrt(2.0 * T0)

            # Note this is for a '1V' test
            dfni = densi/vth * Maxwellian_prefactor * exp(-(vpa/vth)^2)
        else
            error("Unrecognized option "
                  * "manufactured_solns:type=$(manufactured_solns_input.type)")
        end
        return dfni
    end
    function cartesian_dfni_sym(Lr, Lz, r_bc, z_bc, composition, nvperp,
                                manufactured_solns_input, species)
        densi = densi_sym(Lr, Lz, r_bc, z_bc, composition, manufactured_solns_input,
                          species)
        if nvperp == 1
            Maxwellian_prefactor = 1 / sqrt(π)
        else
            Maxwellian_prefactor = 1 / π^1.5
        end
        #if (r_bc == "periodic" && z_bc == "periodic") || (r_bc == "Dirichlet" && z_bc == "periodic")
            dfni = densi * Maxwellian_prefactor * exp( - vz^2 - vr^2 - vzeta^2)
        #end
        return dfni
    end

    function electric_fields(Lr, Lz, r_bc, z_bc, composition, geometry, nr,
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
        phi = expand(composition.T_e*log(dense/N_e)) # use the adiabatic response of electrons for me/mi -> 0
        Er = -Dr(phi)*rfac + geometry.Er_constant
        Ez = -Dz(phi)      + geometry.Ez_constant
        
        Er_expanded = expand_derivatives(Er)
        Ez_expanded = expand_derivatives(Ez)
       
        return Er_expanded, Ez_expanded, phi
    end

    function manufactured_solutions(manufactured_solns_input, Lr, Lz, r_bc, z_bc,
                                    geometry_input_data::geometry_input, composition,
                                    species, nr, nvperp, nvzeta, nvr)

        # calculate the geometry symbolically
        geometry = geometry_sym(geometry_input_data,Lz,Lr,nr)
        ion_species = species.ion[1]
        if composition.n_neutral_species > 0
            neutral_species = species.neutral[1]
        else
            neutral_species = nothing
        end

        densi = densi_sym(Lr, Lz, r_bc, z_bc, composition, manufactured_solns_input,
                          ion_species)
        upari = upari_sym(Lr, Lz, r_bc, z_bc, composition, geometry, nr, manufactured_solns_input,
                          ion_species)
        ppari = ppari_sym(Lr, Lz, r_bc, z_bc, composition, manufactured_solns_input,
                          ion_species)
        pperpi = pperpi_sym(Lr, Lz, r_bc, z_bc, composition, manufactured_solns_input,
                          ion_species, nvperp)
        vthi = vthi_sym(Lr, Lz, r_bc, z_bc, composition, manufactured_solns_input,
                          ion_species, nvperp)
        dfni = dfni_sym(Lr, Lz, r_bc, z_bc, composition, geometry, nr, nvperp,
                        manufactured_solns_input, ion_species)

        densn = densn_sym(Lr, Lz, r_bc, z_bc, geometry,composition,
                          manufactured_solns_input, neutral_species)
        dfnn = dfnn_sym(Lr, Lz, r_bc, z_bc, geometry, composition, nvzeta, nvr,
                        manufactured_solns_input, neutral_species)

        #build julia functions from these symbolic expressions
        # cf. https://docs.juliahub.com/Symbolics/eABRO/3.4.0/tutorials/symbolic_functions/
        densi_func_inner = build_function(densi, z, r, t, expression=Val{false})
        densi_func = (args...) -> Symbolics.value(densi_func_inner(args...))
        upari_func_inner = build_function(upari, z, r, t, expression=Val{false})
        upari_func = (args...) -> Symbolics.value(upari_func_inner(args...))
        ppari_func_inner = build_function(ppari, z, r, t, expression=Val{false})
        ppari_func = (args...) -> Symbolics.value(ppari_func_inner(args...))
        pperpi_func_inner = build_function(pperpi, z, r, t, expression=Val{false})
        pperpi_func = (args...) -> Symbolics.value(pperpi_func_inner(args...))
        vthi_func_inner = build_function(vthi, z, r, t, expression=Val{false})
        vthi_func = (args...) -> Symbolics.value(vthi_func_inner(args...))
        densn_func_inner = build_function(densn, z, r, t, expression=Val{false})
        densn_func = (args...) -> Symbolics.value(densn_func_inner(args...))
        dfni_func_inner = build_function(dfni, vpa, vperp, z, r, t, expression=Val{false})
        dfni_func = (args...) -> Symbolics.value(dfni_func_inner(args...))
        dfnn_func_inner = build_function(dfnn, vz, vr, vzeta, z, r, t, expression=Val{false})
        dfnn_func = (args...) -> Symbolics.value(dfnn_func_inner(args...))
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
    
    function manufactured_electric_fields(Lr, Lz, r_bc, z_bc, composition, geometry_input_data::geometry_input, nr,
                                          manufactured_solns_input, species)
        
        # calculate the geometry symbolically
        geometry = geometry_sym(geometry_input_data,Lz,Lr,nr)
        # calculate the electric fields and the potential
        Er, Ez, phi = electric_fields(Lr, Lz, r_bc, z_bc, composition, geometry, nr,
                                      manufactured_solns_input, species)
        
        Er_func_inner = build_function(Er, z, r, t, expression=Val{false})
        Er_func = (args...) -> Symbolics.value(Er_func_inner(args...))
        Ez_func_inner = build_function(Ez, z, r, t, expression=Val{false})
        Ez_func = (args...) -> Symbolics.value(Ez_func_inner(args...))
        phi_func_inner = build_function(phi, z, r, t, expression=Val{false})
        phi_func = (args...) -> Symbolics.value(phi_func_inner(args...))
        
        manufactured_E_fields = (Er_func = Er_func, Ez_func = Ez_func, phi_func = phi_func)
        
        return manufactured_E_fields
    end

    function manufactured_geometry(geometry_input_data::geometry_input,Lz,Lr,nr)
        
        # calculate the geometry symbolically
        geosym = geometry_sym(geometry_input_data,Lz,Lr,nr)
        Bmag = geosym.Bmag
        bzed = geosym.bzed
        dBdz = geosym.dBdz
        Bmag_func = build_function(Bmag, z, r, expression=Val{false})
        phi_func = (args...) -> Symbolics.value(phi_func_inner(args...))
        bzed_func = build_function(bzed, z, r, expression=Val{false})
        phi_func = (args...) -> Symbolics.value(phi_func_inner(args...))
        dBdz_func = build_function(dBdz, z, r, expression=Val{false})
        phi_func = (args...) -> Symbolics.value(phi_func_inner(args...))
        
        manufactured_geometry = (Bmag_func = Bmag_func,
                                 bzed_func = bzed_func,
                                 dBdz_func = dBdz_func)
        return manufactured_geometry
    end

    # Original implementation of manufactured solutions used a string to specify the
    # radial boundary conditions, as that was what the rest of the code did at the time.
    # This function provides a workaround to convert the `boundaries::boundary_info`
    # object into a string until we want/need to update the manufactured solutions test.
    function get_mms_r_bc(boundaries::boundary_info)
        if length(boundaries.r.inner_sections) > 1 || length(boundaries.r.outer_sections) > 1
            error("Manufactured solutions do not support multiple radial boundary sections")
        end
        if length(boundaries.r.inner_sections) == 0 || length(boundaries.r.outer_sections) == 0
            # No radial domain, so r_bc does not matter
            return "periodic"
        end

        r_bc_type = typeof(boundaries.r.inner_sections[1].ion)
        if typeof(boundaries.r.outer_sections[1].ion) !== r_bc_type
            error("Inner and outer radial boundary conditions are different "
                  * "- unsupported by manufactured solutions")
        end
        if r_bc_type === ion_r_boundary_section_periodic
            r_bc = "periodic"
            if (typeof(boundaries.r.inner_sections[1].electron) !== electron_r_boundary_section_periodic
                || typeof(boundaries.r.outer_sections[1].electron) !== electron_r_boundary_section_periodic)
                error("Electron radial boundary settings different from ion settings "
                      * "- unsupported by manufactured solutions")
            end
            if (typeof(boundaries.r.inner_sections[1].neutral) !== neutral_r_boundary_section_periodic
                || typeof(boundaries.r.outer_sections[1].neutral) !== neutral_r_boundary_section_periodic)
                error("Neutral radial boundary settings different from ion settings "
                      * "- unsupported by manufactured solutions")
            end
        elseif r_bc_type === ion_r_boundary_section_Dirichlet
            r_bc = "Dirichlet"
            if (typeof(boundaries.r.inner_sections[1].electron) !== electron_r_boundary_section_Dirichlet
                || typeof(boundaries.r.outer_sections[1].electron) !== electron_r_boundary_section_Dirichlet)
                error("Electron radial boundary settings different from ion settings "
                      * "- unsupported by manufactured solutions")
            end
            if (typeof(boundaries.r.inner_sections[1].neutral) !== neutral_r_boundary_section_Dirichlet
                || typeof(boundaries.r.outer_sections[1].neutral) !== neutral_r_boundary_section_Dirichlet)
                error("Neutral radial boundary settings different from ion settings "
                      * "- unsupported by manufactured solutions")
            end
        else
            error("Radial boundary type $r_bc_type not supported in manufactured solutions")
        end

        return r_bc
    end

    function manufactured_sources_setup(manufactured_solns_input, r_coord, z_coord, vperp_coord,
            vpa_coord, vzeta_coord, vr_coord, vz_coord, boundaries::boundary_info,
            composition, geometry_input_data::geometry_input, collisions, num_diss_params,
            species)

        geometry = geometry_sym(geometry_input_data,z_coord.L,r_coord.L,r_coord.n)
        ion_species = species.ion[1]
        if composition.n_neutral_species > 0
            neutral_species = species.neutral[1]
        else
            neutral_species = nothing
        end

        r_bc = get_mms_r_bc(boundaries)

        # ion manufactured solutions
        densi = densi_sym(r_coord.L, z_coord.L, r_bc, z_coord.bc, composition,
                          manufactured_solns_input, ion_species)
        upari = upari_sym(r_coord.L, z_coord.L, r_bc, z_coord.bc, composition, geometry, r_coord.n, manufactured_solns_input, ion_species)
        vthi = vthi_sym(r_coord.L, z_coord.L, r_bc, z_coord.bc, composition, manufactured_solns_input,
                          ion_species, vperp_coord.n)
        dfni = dfni_sym(r_coord.L, z_coord.L, r_bc, z_coord.bc, composition,
                        geometry, r_coord.n, vperp_coord.n, manufactured_solns_input,
                        ion_species)
        #dfni in vr vz vzeta coordinates
        vrvzvzeta_dfni = cartesian_dfni_sym(r_coord.L, z_coord.L, r_bc, z_coord.bc,
                                            composition, vperp_coord.n,
                                            manufactured_solns_input, ion_species)

        # neutral manufactured solutions
        densn = densn_sym(r_coord.L,z_coord.L, r_bc, z_coord.bc, geometry,
                          composition, manufactured_solns_input, neutral_species)
        dfnn = dfnn_sym(r_coord.L, z_coord.L, r_bc, z_coord.bc, geometry,
                        composition, vzeta_coord.n, vr_coord.n, manufactured_solns_input,
                        neutral_species)
        # gyroaverage < dfnn > in vpa vperp coordinates
        gav_dfnn = gyroaveraged_dfnn_sym(r_coord.L, z_coord.L, r_bc, z_coord.bc,
                                         geometry, composition, vzeta_coord.n, vr_coord.n,
                                         manufactured_solns_input, neutral_species)

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
        Bzeta = geometry.Bzeta
        Bmag = geometry.Bmag
        dBdz = geometry.dBdz
        dBdr = geometry.dBdr
        rhostar = geometry.rhostar
        jacobian = geometry.jacobian
        ExBgeofac = rhostar*Bzeta*jacobian/Bmag^2
        #exceptions for cases with missing terms 
        if composition.n_neutral_species > 0
            cx_frequency = collisions.reactions.charge_exchange_frequency
            ionization_frequency = collisions.reactions.ionization_frequency
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
        Er, Ez, phi = electric_fields(r_coord.L, z_coord.L, r_bc, z_coord.bc,
                                      composition, geometry, r_coord.n, manufactured_solns_input,
                                      ion_species)

        # the adiabatic invariant (for compactness)
        mu = 0.5*(vperp^2)/Bmag
        # the ion characteristic velocities
        dzdt = vpa * (Bzed/Bmag) - ExBgeofac*Er
        drdt = ExBgeofac*Ez*rfac
        dvpadt = (Bzed/Bmag)*Ez - mu*(Bzed/Bmag)*dBdz
        dvperpdt = (0.5*vperp/Bmag)*(dzdt*dBdz + drdt*dBdr)
        # the ion source to maintain the manufactured solution
        Si = ( Dt(dfni) 
               + dzdt * Dz(expand(dfni))
               + drdt * Dr(dfni)
               + dvpadt * Dvpa(dfni)
               + dvperpdt * Dvperp(dfni)
               + cx_frequency*( densn*dfni - densi*gav_dfnn )
               - ionization_frequency*dense*gav_dfnn )
        nu_krook = collisions.krook.nuii0
        if nu_krook > 0.0
            if collisions.krook.frequency_option == "manual"
                nuii_krook = nu_krook
            else # default option
                nuii_krook = nu_krook * densi / vthi^3
            end
            if vperp_coord.n > 1
                pvth  = 3
                Krook_vthi = vthi
            else 
                pvth = 1
                Krook_vthi = sqrt(3.0) * vthi
                nuii_krook /= 3.0^1.5
            end
            FMaxwellian = (densi/Krook_vthi^pvth)/π^(pvth/2)*exp( -( ( vpa-upari)^2 + vperp^2 )/Krook_vthi^2)
            Si += - nuii_krook*(FMaxwellian - dfni)
        end
        include_num_diss_in_MMS = true
        if num_diss_params.ion.vpa_dissipation_coefficient > 0.0 && include_num_diss_in_MMS
            Si += - num_diss_params.ion.vpa_dissipation_coefficient*Dvpa(Dvpa(dfni))
        end
        if num_diss_params.ion.vperp_dissipation_coefficient > 0.0 && include_num_diss_in_MMS
            Si += - num_diss_params.ion.vperp_dissipation_coefficient/vperp*Dvperp(vperp*Dvperp(dfni))
        end
        if num_diss_params.ion.r_dissipation_coefficient > 0.0 && include_num_diss_in_MMS
            Si += - rfac*num_diss_params.ion.r_dissipation_coefficient*Dr(Dr(dfni))
        end
        if num_diss_params.ion.z_dissipation_coefficient > 0.0 && include_num_diss_in_MMS
            Si += - num_diss_params.ion.z_dissipation_coefficient*Dz(Dz(dfni))
        end

        Source_i = expand_derivatives(Si)
        
        # the neutral source to maintain the manufactured solution
        Sn = Dt(dfnn) + vz * Dz(dfnn) + rfac*vr * Dr(dfnn) + cx_frequency* (densi*dfnn - densn*vrvzvzeta_dfni) + ionization_frequency*dense*dfnn
        if num_diss_params.neutral.vz_dissipation_coefficient > 0.0 && include_num_diss_in_MMS
            Sn += - num_diss_params.neutral.vz_dissipation_coefficient*Dvz(Dvz(dfnn))
        end
        if num_diss_params.neutral.r_dissipation_coefficient > 0.0 && include_num_diss_in_MMS
            Sn += - rfac*num_diss_params.neutral.r_dissipation_coefficient*Dr(Dr(dfnn))
        end
        if num_diss_params.neutral.z_dissipation_coefficient > 0.0 && include_num_diss_in_MMS
            Sn += - num_diss_params.neutral.z_dissipation_coefficient*Dz(Dz(dfnn))
        end
        
        Source_n = expand_derivatives(Sn)
        
        Source_i_func_inner = build_function(Source_i, vpa, vperp, z, r, t, expression=Val{false})
        Source_i_func = (args...) -> Symbolics.value(Source_i_func_inner(args...))
        Source_n_func_inner = build_function(Source_n, vz, vr, vzeta, z, r, t, expression=Val{false})
        Source_n_func = (args...) -> Symbolics.value(Source_n_func_inner(args...))
        
        if expand_derivatives(Dt(Source_i)) == 0 && expand_derivatives(Dt(Source_n)) == 0
            time_independent_sources = true
        else
            time_independent_sources = false
        end
        
        return time_independent_sources, Source_i_func, string(Source_i), Source_n_func,
               string(Source_n)
    end 
    
end
