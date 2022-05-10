"""
"""
module manufactured_solns

export manufactured_solutions
export manufactured_sources

using Symbolics

    @variables r z vpa vperp t

    function densi_sym()
        densi = 1.0 +  0.5*(sin(r) + sin(z))*sin(t)  
        return densi
    end

    function dfni_sym()
        densi = densi_sym()
        dfni = densi * exp( - vpa^2 - vperp^2) / sqrt(pi^3)
        return dfni
    end

    function manufactured_solutions()

        densi = densi_sym()
        dfni = dfni_sym()
        
        #build julia functions from these symbolic expressions
        # cf. https://docs.juliahub.com/Symbolics/eABRO/3.4.0/tutorials/symbolic_functions/
        densi_func = build_function(densi, z, r, t, expression=Val{false})
        dfni_func = build_function(dfni, vpa, vperp, z, r, t, expression=Val{false})
        # return function
        # call like: 
        # densi_func(zval, rval, tval) 
        return dfni_func, densi_func
    end 

    #function manufactured_sources(dfni,densi,geometry)
    function manufactured_sources(geometry)
        
        densi = densi_sym()
        dfni = dfni_sym()
        
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
        
        Source_i_func = build_function(Source_i, [vpa, vperp, z, r, t])
        return eval(Source_i_func[2])
    end 
    
end
