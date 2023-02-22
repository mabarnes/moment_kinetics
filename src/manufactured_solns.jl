"""
"""
module manufactured_solns

export manufactured_solutions
export manufactured_sources
export manufactured_electric_fields

using Symbolics
using IfElse
using ..input_structs
import LambertW: lambertw

    @variables r z vpa vperp t vz vr vzeta
    typed_zero(vz) = zero(vz)
    @register_symbolic typed_zero(vz)
    zero_val = 1.0e-8
    #epsilon_offset = 0.001 
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
                densn = 1.5 +  0.1*(cos(2.0*pi*r/Lr) + cos(2.0*pi*z/Lz)) #*sin(2.0*pi*t)  
            elseif r_bc == "Dirichlet"
                densn = 1.5 + 0.3*r/Lr
            end
        elseif z_bc == "wall"
            T_wall = composition.T_wall
            Bzed = geometry.Bzed
            Bmag = geometry.Bmag
            epsilon = composition.epsilon_offset
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
            epsilon = composition.epsilon_offset
            Gamma_minus = 0.5*(Bzed/Bmag)*nminus_sym(Lr,Lz,r_bc,z_bc)/sqrt(pi)
            Gamma_plus = 0.5*(Bzed/Bmag)*nplus_sym(Lr,Lz,r_bc,z_bc)/sqrt(pi)
            dfnn = Hplus *( Gamma_minus*( 0.5 - z/Lz)^2 + 1.0 )*FKw + Hminus*( Gamma_plus*( 0.5 + z/Lz)^2 + 1.0 )*FKw 
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
    function densi_sym(Lr,Lz,r_bc,z_bc,composition,nr,nz)
        if z_bc == "periodic"
            if r_bc == "periodic"
                densi = 1.5 +  0.1*(sin(2.0*pi*r/Lr) + sin(2.0*pi*z/Lz))#*sin(2.0*pi*t)  
            elseif r_bc == "Dirichlet" 
                #densi = 1.0 +  0.5*sin(2.0*pi*z/Lz)*(r/Lr + 0.5) + 0.2*sin(2.0*pi*r/Lr)*sin(2.0*pi*t)
                #densi = 1.0 +  0.5*sin(2.0*pi*z/Lz)*(r/Lr + 0.5) + sin(2.0*pi*r/Lr)*sin(2.0*pi*t)
                densi = 1.0 +  0.5*(r/Lr)*sin(2.0*pi*z/Lz)
            end
        elseif z_bc == "wall"
            _, _, phi = electric_fields(Lr,Lz,r_bc,z_bc,composition,nr,nz)
            phi_w = substitute(phi, z=>0.5*Lz)
            densi = (1+2*(phi-phi_w))*(nplus_sym(Lr,Lz,r_bc,z_bc)/4*(0.5+z/Lz) + nminus_sym(Lr,Lz,r_bc,z_bc)/4*(0.5-z/Lz) + (0.5+z/Lz)*(0.5-z/Lz)*nzero_sym(Lr,Lz,r_bc,z_bc)) * exp(-(phi-phi_w))
        end
        return densi
    end

    function jpari_into_LHS_wall_sym(Lr,Lz,r_bc,z_bc,composition)
        if z_bc == "periodic"
            jpari_into_LHS_wall_sym = 0.0
        elseif z_bc == "wall"
            #appropriate for wall bc test when Er = 0 (nr == 1)
            epsilon = composition.epsilon_offset
            jpari_into_LHS_wall_sym = -0.5*nminus_sym(Lr,Lz,r_bc,z_bc,epsilon)/sqrt(pi)
        end
        return jpari_into_LHS_wall_sym
    end
    
    function Heaviside(x)
        if x > 0.0
            return 1.0
        elseif x < 0.0
            return 0.0
        else
            return 0.5
        end
    end
    @register_symbolic Heaviside(x)
    @register_symbolic HeavisidePrime(x)
    Symbolics.derivative(::typeof(Heaviside), args::NTuple{1,Any}, ::Val{1}) =
    HeavisidePrime(args[1])
    @register_symbolic lambertw(x,i::Integer)
    Symbolics.derivative(::typeof(lambertw), args::NTuple{2,Any}, ::Val{1}) =
        lambertw(args...) / args[1] / (1 + lambertw(args...))

    # ion distribution symbolic function
    function dfni_sym(Lr,Lz,r_bc,z_bc,composition,geometry,nr,nz)
        densi = densi_sym(Lr,Lz,r_bc,z_bc,composition,nr,nz)
        
        # calculate the electric fields and the potential
        Er, Ez, phi = electric_fields(Lr,Lz,r_bc,z_bc,composition,nr,nz)
        
        # get geometric/composition data
        Bzed = geometry.Bzed
        Bmag = geometry.Bmag
        rhostar = geometry.rhostar
        epsilon = composition.epsilon_offset
        if z_bc == "periodic"
            dfni = densi * exp( - vpa^2 - vperp^2) 
        elseif z_bc == "wall"
            phi_w = substitute(phi, z=>0.5*Lz)
            vpabar = vpa - (rhostar/2.0)*(Bmag/Bzed)*Er # effective velocity in z direction * (Bmag/Bzed)
            ffa =  exp(- vperp^2)
            dfni = ffa * (Heaviside(vpabar)*(vpabar^2 + (phi-phi_w))*(0.5+z/Lz)*nplus_sym(Lr,Lz,r_bc,z_bc) + Heaviside(-vpabar)*(vpabar^2 + (phi-phi_w))*(0.5-z/Lz)*nminus_sym(Lr,Lz,r_bc,z_bc) + (0.5+z/Lz)*(0.5-z/Lz)*nzero_sym(Lr,Lz,r_bc,z_bc))*exp(-vpabar^2-(phi-phi_w))
        end
        return dfni
    end
    function cartesian_dfni_sym(Lr,Lz,r_bc,z_bc,composition,nr,nz)
        densi = densi_sym(Lr,Lz,r_bc,z_bc,composition,nr,nz)
        #if (r_bc == "periodic" && z_bc == "periodic") || (r_bc == "Dirichlet" && z_bc == "periodic")
            dfni = densi * exp( - vz^2 - vr^2 - vzeta^2) 
        #end
        return dfni
    end

    function electric_fields(Lr,Lz,r_bc,z_bc,composition,nr,nz)
       
        # define derivative operators
        Dr = Differential(r) 
        Dz = Differential(z) 
 
        # get N_e factor for boltzmann response
        if composition.electron_physics == boltzmann_electron_response_with_simple_sheath && nr == 1 
            # so 1D MMS test with 3V neutrals where ion current can be calculated prior to knowing Er
            jpari_into_LHS_wall = jpari_into_LHS_wall_sym(Lr,Lz,r_bc,z_bc,composition)
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

        if nz > 1 # keep radial electric field
            zfac = 1.0
        else      # drop radial electric field
            zfac = 0.0
        end

        if z_bc == "wall"
            # At the wall, density is nminus/4=nplus/4
            phi_wall = 0 #composition.T_e*log(nminus_sym(Lr,Lz,r_bc,z_bc)/4/N_e)
            # Note need the '-1' branch of the Lambert W function, not the '0' branch, to
            # give phi(z)>=phi_wall.
            calZ = -1*exp(4*z^2/Lz^2-2)
            phi = phi_wall - lambertw(calZ,-1)/2 - 1 + 2*z^2/Lz
        else
            densi = densi_sym(Lr,Lz,r_bc,z_bc,composition,nr,nz)
            # calculate the electric fields
            dense = densi # get the electron density via quasineutrality with Zi = 1
            phi = composition.T_e*log(dense/N_e) # use the adiabatic response of electrons for me/mi -> 0
        end
        Er = -Dr(phi)*rfac + composition.Er_constant
        Ez = -Dz(phi)*zfac
        
        Er_expanded = expand_derivatives(Er)
        Ez_expanded = expand_derivatives(Ez)
       
        return Er_expanded, Ez_expanded, phi
    end

    function manufactured_solutions(Lr,Lz,r_bc,z_bc,geometry,composition,nr,nz)
        densi = densi_sym(Lr,Lz,r_bc,z_bc,composition,nr,nz)
        dfni = dfni_sym(Lr,Lz,r_bc,z_bc,composition,geometry,nr,nz)
        
        densn = densn_sym(Lr,Lz,r_bc,z_bc,geometry,composition)
        dfnn = dfnn_sym(Lr,Lz,r_bc,z_bc,geometry,composition)

        Er, Ez, phi = electric_fields(Lr,Lz,r_bc,z_bc,composition,nr,nz)

        #build julia functions from these symbolic expressions
        # cf. https://docs.juliahub.com/Symbolics/eABRO/3.4.0/tutorials/symbolic_functions/
        densi_func = build_function(densi, z, r, t, expression=Val{false})
        densn_func = build_function(densn, z, r, t, expression=Val{false})
        dfni_func = build_function(dfni, vpa, vperp, z, r, t, expression=Val{false})
        dfnn_func = build_function(dfnn, vz, vr, vzeta, z, r, t, expression=Val{false})
        Er_func = build_function(Er, z, r, t, expression=Val(false))
        # return function
        # call like: 
        # densi_func(zval, rval, tval) 
        # dfni_func(vpaval, vperpval, zval, rval, tval) 
        # densn_func(zval, rval, tval) 
        # dfnn_func(vzval, vrval, vzetapval, zval, rval, tval) 
        
        manufactured_solns_list = (densi_func = densi_func, densn_func = densn_func, dfni_func = dfni_func, dfnn_func = dfnn_func, Er_func = Er_func)
        
        return manufactured_solns_list
    end 
    
    function manufactured_electric_fields(Lr,Lz,r_bc,z_bc,composition,nr,nz)
        
        # calculate the electric fields and the potential
        Er, Ez, phi = electric_fields(Lr,Lz,r_bc,z_bc,composition,nr,nz)
        
        Er_func = build_function(Er, z, r, t, expression=Val{false})
        Ez_func = build_function(Ez, z, r, t, expression=Val{false})
        phi_func = build_function(phi, z, r, t, expression=Val{false})
        
        manufactured_E_fields = (Er_func = Er_func, Ez_func = Ez_func, phi_func = phi_func)
        
        return manufactured_E_fields
    end 

    function manufactured_sources(Lr,Lz,r_bc,z_bc,composition,geometry,collisions,nr,nz,nvpa,num_diss_params)
        
        # ion manufactured solutions
        densi = densi_sym(Lr,Lz,r_bc,z_bc,composition,nr,nz)
        dfni = dfni_sym(Lr,Lz,r_bc,z_bc,composition,geometry,nr,nz)
        vrvzvzeta_dfni = cartesian_dfni_sym(Lr,Lz,r_bc,z_bc,composition,nr,nz) #dfni in vr vz vzeta coordinates
        
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
        if nr > 1 # keep radial derivatives
            rfac = 1.0
        else      # drop radial derivative terms
            rfac = 0.0
        end
        
        # calculate the electric fields and the potential
        Er, Ez, phi = electric_fields(Lr,Lz,r_bc,z_bc,composition,nr,nz)
        
        # the ion source to maintain the manufactured solution
        Si = ( Dt(dfni) + ( vpa * (Bzed/Bmag) - 0.5*rhostar*Er ) * Dz(dfni) + ( 0.5*rhostar*Ez*rfac ) * Dr(dfni) + ( 0.5*Ez*Bzed/Bmag ) * Dvpa(dfni)
               + cx_frequency*( densn*dfni - densi*gav_dfnn ) ) - ionization_frequency*dense*gav_dfnn 
               - num_diss_params.vpa_dissipation_coefficient*Dvpa(Dvpa(dfni))
        Source_i = expand_derivatives(Si)
        if z_bc == "wall"
            Source_i = substitute(Source_i, HeavisidePrime(-vpa)=>HeavisidePrime(vpa))
            # Use ~1/sqrt(nvpa) for the width so that as resolution increases the source
            # is narrower but also contains more grid points, so can be better resolved
            delta_width = Lz/sqrt(nvpa)
            approx_delta_func = 1/sqrt(π)/delta_width * exp(-vpa^2/delta_width^2)
            Source_i = substitute(Source_i, HeavisidePrime(vpa)=>approx_delta_func)
        end
        
        # the neutral source to maintain the manufactured solution
        Sn = Dt(dfnn) + vz * Dz(dfnn) + rfac*vr * Dr(dfnn) + cx_frequency* (densi*dfnn - densn*vrvzvzeta_dfni) + ionization_frequency*dense*dfnn
        Source_n = expand_derivatives(Sn)
        
        Source_i_func = build_function(Source_i, vpa, vperp, z, r, t, expression=Val{false})
        Source_n_func = build_function(Source_n, vz, vr, vzeta, z, r, t, expression=Val{false})
        
        manufactured_sources_list = (Source_i_func = Source_i_func, Source_n_func = Source_n_func)
        
        return manufactured_sources_list
    end 
    
end
