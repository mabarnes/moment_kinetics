"""
"""
module manufactured_solns

export manufactured_solutions
export manufactured_sources

using Symbolics
using ..input_structs

    @variables r z vpa vperp t vz vr vzeta

    #standard functions for building densities
    function nplus_sym(Lr,r_bc)
        #if r_bc == "periodic"
        nplus = 1.0 + 0.05*sin(2.0*pi*r/Lr)
        #end
        return nplus
    end
    
    function nminus_sym(Lr,r_bc)
        #if r_bc == "periodic"
        nminus = 1.0 + 0.05*sin(2.0*pi*r/Lr)
        #end
        return nminus
    end
    
    function nzero_sym(Lr,r_bc)
        #if r_bc == "periodic"
        nzero = 1.0 + 0.05*sin(2.0*pi*r/Lr)# 1.0 #+ (r/Lr + 0.5)*(0.5 - r/Lr)
        #end
        return nzero
    end

    function knudsen_cosine(composition)
        T_wall = composition.T_wall
        # prefac here may cause problems with NaNs if vz = vr = vzeta = 0 is on grid
        prefac = abs(vz)/sqrt(vz^2 + vr^2 + vzeta^2)
        exponetial = exp( - (vz^2 + vr^2 + vzeta^2)/T_wall )
        if composition.use_test_neutral_wall_pdf
            #test dfn
            knudsen_pdf = (4.0/T_wall^(5.0/2.0))*abs(vz)*exponetial
        else
            #proper Knudsen dfn
            knudsen_pdf = (3.0*sqrt(pi)/T_wall^2)*prefac*exponetial
        end
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
            if composition.use_test_neutral_wall_pdf
                #test 
                prefactor = 2.0/sqrt(pi*T_wall)
            else
                #proper prefactor
                prefactor = 3.0*sqrt(pi)/(4.0*sqrt(T_wall))
            end
            densn = prefactor*( (0.5 - z/Lz)*Gamma_minus + (0.5 + z/Lz)*Gamma_plus + 2.0 )
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
    
    # ion density symbolic function
    function densi_sym(Lr,Lz,r_bc,z_bc)
        if r_bc == "periodic" && z_bc == "periodic"
            densi = 1.5 +  0.1*(sin(2.0*pi*r/Lr) + sin(2.0*pi*z/Lz))*sin(2.0*pi*t)  
        elseif r_bc == "Dirichlet" && z_bc == "periodic"
            #densi = 1.0 +  0.5*sin(2.0*pi*z/Lz)*(r/Lr + 0.5) + 0.2*sin(2.0*pi*r/Lr)*sin(2.0*pi*t)
            #densi = 1.0 +  0.5*sin(2.0*pi*z/Lz)*(r/Lr + 0.5) + sin(2.0*pi*r/Lr)*sin(2.0*pi*t)
            densi = 1.0 +  0.5*(r/Lr + 0.5) 
        elseif r_bc == "periodic" && z_bc == "wall"
            densi = 0.25*(0.5 - z/Lz)*nminus_sym(Lr,r_bc) + 0.25*(z/Lz + 0.5)*nplus_sym(Lr,r_bc) + (z/Lz + 0.5)*(0.5 - z/Lz)*nzero_sym(Lr,r_bc)  #+  0.5*(r/Lr + 0.5) + 0.5*(z/Lz + 0.5)
        end
        return densi
    end

    function jpari_into_LHS_wall_sym(Lr,Lz,r_bc,z_bc)
        if r_bc == "periodic" && z_bc == "periodic"
            jpari_into_LHS_wall_sym = 0.0
        elseif r_bc == "periodic" && z_bc == "wall"
            #appropriate for wall bc test when Er = 0 (nr == 1)
            jpari_into_LHS_wall_sym = -0.5*nminus_sym(Lr,r_bc)/sqrt(pi)
        end
        return jpari_into_LHS_wall_sym
    end
    
    # ion distribution symbolic function
    function dfni_sym(Lr,Lz,r_bc,z_bc,composition,geometry,nr)
        densi = densi_sym(Lr,Lz,r_bc,z_bc)
        # define derivative operators
        Dr = Differential(r) 
        Dz = Differential(z) 
        # calculate the necessary electric fields
        dense = densi # get the electron density via quasineutrality with Zi = 1
        phi = composition.T_e*log(dense) # use the adiabatic response of electrons for me/mi -> 0
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
            dfni = densi * exp( - vpa^2 - vperp^2) 
        elseif r_bc == "periodic" && z_bc == "wall"
            vpabar = vpa - (rhostar/2.0)*(Bmag/Bzed)*expand_derivatives(Er)*rfac # effective velocity in z direction * (Bmag/Bzed)
            Hplus = 0.5*(sign(vpabar) + 1.0)
            Hminus = 0.5*(sign(-vpabar) + 1.0)
            ffa =  exp(- vperp^2)
            dfni = ffa * ( nminus_sym(Lr,r_bc)* (0.5 - z/Lz) * Hminus * vpabar^2 + nplus_sym(Lr,r_bc)*(z/Lz + 0.5) * Hplus * vpabar^2 + nzero_sym(Lr,r_bc)*(z/Lz + 0.5)*(0.5 - z/Lz) ) * exp( - vpabar^2 )
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

    function manufactured_sources(Lr,Lz,r_bc,z_bc,composition,geometry,collisions,nr)
        
        # ion manufactured solutions
        densi = densi_sym(Lr,Lz,r_bc,z_bc)
        dfni = dfni_sym(Lr,Lz,r_bc,z_bc,composition,geometry,nr)
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
        Dt = Differential(t) 
    
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
        
        # get N_e factor for boltzmann response
        if composition.electron_physics == boltzmann_electron_response_with_simple_sheath && nr == 1 
            # so 1D MMS test with 3V neutrals where ion current can be calculated prior to knowing Er
            jpari_into_LHS_wall = jpari_into_LHS_wall_sym(Lr,Lz,r_bc,z_bc)
            N_e = -2.0*sqrt(pi*composition.me_over_mi)*exp(-composition.phi_wall/composition.T_e)*jpari_into_LHS_wall
        elseif composition.electron_physics == boltzmann_electron_response_with_simple_sheath && nr > 1 
            println("simple sheath MMS test not supported for nr > 1")
            N_e = 1.0
        elseif composition.electron_physics == boltzmann_electron_response 
            # all other cases
            # N_e equal to reference density 
            N_e = 1.0 
        end 
        # calculate the electric fields
        dense = densi # get the electron density via quasineutrality with Zi = 1
        phi = composition.T_e*log(dense/N_e) # use the adiabatic response of electrons for me/mi -> 0
        Er = -Dr(phi)
        Ez = -Dz(phi)
    
        # the ion source to maintain the manufactured solution
        Si = ( Dt(dfni) + ( vpa * (Bzed/Bmag) - 0.5*rhostar*Er*rfac ) * Dz(dfni) + ( 0.5*rhostar*Ez*rfac ) * Dr(dfni) + ( 0.5*Ez*Bzed/Bmag ) * Dvpa(dfni)
               + cx_frequency*( densn*dfni - densi*gav_dfnn ) ) - ionization_frequency*dense*gav_dfnn 
        Source_i = expand_derivatives(Si)
        
        # the neutral source to maintain the manufactured solution
        Sn = Dt(dfnn) + vz * Dz(dfnn) + rfac*vr * Dr(dfnn) + cx_frequency* (densi*dfnn - densn*vrvzvzeta_dfni) + ionization_frequency*dense*dfnn
        Source_n = expand_derivatives(Sn)
        
        Source_i_func = build_function(Source_i, vpa, vperp, z, r, t, expression=Val{false})
        Source_n_func = build_function(Source_n, vz, vr, vzeta, z, r, t, expression=Val{false})
        
        manufactured_sources_list = (Source_i_func = Source_i_func, Source_n_func = Source_n_func)
        
        return manufactured_sources_list
    end 
    
end
