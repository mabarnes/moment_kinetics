"""
"""
module manufactured_solns

export manufactured_solutions
export manufactured_sources

using Symbolics

    @variables r z vpa vperp t

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

    function dfni_sym(Lr,Lz,r_bc,z_bc)
        densi = densi_sym(Lr,Lz,r_bc,z_bc)
        if (r_bc == "periodic" && z_bc == "periodic") || (r_bc == "Dirichlet" && z_bc == "periodic")
            dfni = densi * exp( - vpa^2 - vperp^2) #/ sqrt(pi^3)
        elseif r_bc == "periodic" && z_bc == "wall"
            Hplus = 0.5*(sign(vpa) + 1.0)
            Hminus = 0.5*(sign(-vpa) + 1.0)
            ffa =  exp(- vperp^2)
            dfni = ffa * ( (0.5 - z/Lz) * Hminus * vpa^2 + (z/Lz + 0.5) * Hplus * vpa^2 + 0.2*(z/Lz + 0.5)*(0.5 - z/Lz) ) * exp( - vpa^2 )
        end
        return dfni
    end

    function manufactured_solutions(Lr,Lz,r_bc,z_bc)

        densi = densi_sym(Lr,Lz,r_bc,z_bc)
        dfni = dfni_sym(Lr,Lz,r_bc,z_bc)
        
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

    #function manufactured_sources(dfni,densi,geometry)
    function manufactured_sources(Lr,Lz,r_bc,z_bc,geometry)
        
        densi = densi_sym(Lr,Lz,r_bc,z_bc)
        dfni = dfni_sym(Lr,Lz,r_bc,z_bc)
        
        Dr = Differential(r) 
        Dz = Differential(z) 
        Dvpa = Differential(vpa) 
        Dvperp = Differential(vperp) 
        Dt = Differential(t) 
    
        Bzed = geometry.Bzed
        Bmag = geometry.Bmag
        rhostar = geometry.rstar
        #Bzed = 1.0 #geometry.Bzed
        #Bmag = 1.0 #geometry.Bmag
        #rhostar = 1.0 #geometry.rstar
        
        phi = log(densi)
        Er = -Dr(phi)
        Ez = -Dz(phi)
    
        S = Dt(dfni) + ( vpa * (Bzed/Bmag) - 0.5*rhostar*Er ) * Dz(dfni) + ( 0.5*rhostar*Ez ) * Dr(dfni) + ( 0.5*Ez*Bzed/Bmag ) * Dvpa(dfni)
        Source_i = expand_derivatives(S)
        
        Source_i_func = build_function(Source_i, vpa, vperp, z, r, t, expression=Val{false})
        return Source_i_func
    end 
    
end
