"""
"""
module manufactured_solns

export manufactured_solutions
export manufactured_sources

using Symbolics

    @variables r z vpa vperp t vz vr vzeta

    # neutral density symbolic function
    function densn_sym(Lr,Lz,r_bc,z_bc)
        if r_bc == "periodic" && z_bc == "periodic"
            densn = 1.0 +  0.1*(cos(2.0*pi*r/Lr) + cos(2.0*pi*z/Lz))*sin(2.0*pi*t)  
        end
        return densn
    end

    # neutral distribution symbolic function
    function dfnn_sym(Lr,Lz,r_bc,z_bc)
        densn = densn_sym(Lr,Lz,r_bc,z_bc)
        if (r_bc == "periodic" && z_bc == "periodic")
            dfnn = densn * exp( - vz^2 - vr^2 - vzeta^2)
        end
        return dfnn
    end
    function gyroaveraged_dfnn_sym(Lr,Lz,r_bc,z_bc)
        densn = densn_sym(Lr,Lz,r_bc,z_bc)
        if (r_bc == "periodic" && z_bc == "periodic")
            dfnn = densn * exp( - vpa^2 - vperp^2 )
        end
        return dfnn
    end
    
    # ion density symbolic function
    function densi_sym(Lr,Lz,r_bc,z_bc)
        if r_bc == "periodic" && z_bc == "periodic"
            densi = 1.0 +  0.1*(sin(2.0*pi*r/Lr) + sin(2.0*pi*z/Lz))*sin(2.0*pi*t)  
        elseif r_bc == "Dirichlet" && z_bc == "periodic"
            #densi = 1.0 +  0.5*sin(2.0*pi*z/Lz)*(r/Lr + 0.5) + 0.2*sin(2.0*pi*r/Lr)*sin(2.0*pi*t)
            #densi = 1.0 +  0.5*sin(2.0*pi*z/Lz)*(r/Lr + 0.5) + sin(2.0*pi*r/Lr)*sin(2.0*pi*t)
            densi = 1.0 +  0.5*(r/Lr + 0.5) 
        elseif r_bc == "periodic" && z_bc == "wall"
            densi = 0.25*(0.5 - z/Lz) + 0.25*(z/Lz + 0.5)+ 0.2*(z/Lz + 0.5)*(0.5 - z/Lz)  #+  0.5*(r/Lr + 0.5) + 0.5*(z/Lz + 0.5)
        end
        return densi
    end

    # ion distribution symbolic function
    function dfni_sym(Lr,Lz,r_bc,z_bc)
        densi = densi_sym(Lr,Lz,r_bc,z_bc)
        if (r_bc == "periodic" && z_bc == "periodic") || (r_bc == "Dirichlet" && z_bc == "periodic")
            dfni = densi * exp( - vpa^2 - vperp^2) 
        elseif r_bc == "periodic" && z_bc == "wall"
            Hplus = 0.5*(sign(vpa) + 1.0)
            Hminus = 0.5*(sign(-vpa) + 1.0)
            ffa =  exp(- vperp^2)
            dfni = ffa * ( (0.5 - z/Lz) * Hminus * vpa^2 + (z/Lz + 0.5) * Hplus * vpa^2 + 0.2*(z/Lz + 0.5)*(0.5 - z/Lz) ) * exp( - vpa^2 )
        end
        return dfni
    end
    function cartesian_dfni_sym(Lr,Lz,r_bc,z_bc)
        densi = densi_sym(Lr,Lz,r_bc,z_bc)
        if (r_bc == "periodic" && z_bc == "periodic") || (r_bc == "Dirichlet" && z_bc == "periodic")
            dfni = densi * exp( - vz^2 - vr^2 - vzeta^2) 
        end
        return dfni
    end

    function manufactured_solutions(Lr,Lz,r_bc,z_bc)

        densi = densi_sym(Lr,Lz,r_bc,z_bc)
        dfni = dfni_sym(Lr,Lz,r_bc,z_bc)
        
        densn = densn_sym(Lr,Lz,r_bc,z_bc)
        dfnn = dfnn_sym(Lr,Lz,r_bc,z_bc)
        
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

    function manufactured_sources(Lr,Lz,r_bc,z_bc,geometry,collisions)
        
        # ion manufactured solutions
        densi = densi_sym(Lr,Lz,r_bc,z_bc)
        dfni = dfni_sym(Lr,Lz,r_bc,z_bc)
        vrvzvzeta_dfni = cartesian_dfni_sym(Lr,Lz,r_bc,z_bc) #dfni in vr vz vzeta coordinates
        
        # neutral manufactured solutions
        densn = densn_sym(Lr,Lz,r_bc,z_bc)
        dfnn = dfnn_sym(Lr,Lz,r_bc,z_bc)
        gav_dfnn = gyroaveraged_dfnn_sym(Lr,Lz,r_bc,z_bc) # gyroaverage < dfnn > in vpa vperp coordinates
        
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
        cx_frequency = collisions.charge_exchange
        
        # calculate the electric fields
        phi = log(densi)
        Er = -Dr(phi)
        Ez = -Dz(phi)
    
        # the ion source to maintain the manufactured solution
        Si = ( Dt(dfni) + ( vpa * (Bzed/Bmag) - 0.5*rhostar*Er ) * Dz(dfni) + ( 0.5*rhostar*Ez ) * Dr(dfni) + ( 0.5*Ez*Bzed/Bmag ) * Dvpa(dfni)
               + cx_frequency*( densn*dfni - densi*gav_dfnn ) ) 
        Source_i = expand_derivatives(Si)
        
        # the neutral source to maintain the manufactured solution
        Sn = Dt(dfnn) + vz * Dz(dfnn) + vr * Dr(dfnn) + cx_frequency* (densi*dfnn - densn*vrvzvzeta_dfni)
        Source_n = expand_derivatives(Sn)
        
        Source_i_func = build_function(Source_i, vpa, vperp, z, r, t, expression=Val{false})
        Source_n_func = build_function(Source_n, vz, vr, vzeta, z, r, t, expression=Val{false})
        
        manufactured_sources_list = (Source_i_func = Source_i_func, Source_n_func = Source_n_func)
        
        return manufactured_sources_list
    end 
    
end
