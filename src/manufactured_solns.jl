"""
"""
module manufactured_solns

using Symbolics

    @variables r z vpa vperp t

    function manufactured_solutions()

        densi = 1.0 + sin(t) + sin(r) + sin(z) 
        dfni = densi * exp( - vpa^2 - vperp^2) / sqrt(pi^3)

        return densi, dfni
    end 

    #function manufactured_sources(dfni,densi,geometry)
    function manufactured_sources(dfni,densi)
        
        Dr = Differential(r) 
        Dz = Differential(z) 
        Dvpa = Differential(vpa) 
        Dvperp = Differential(vperp) 
        Dt = Differential(t) 
    
        #Bzed = geometry.Bzed
        #Bmag = geometry.Bmag
        #rhostar = geometry.rstar
        Bzed = 1.0 #geometry.Bzed
        Bmag = 1.0 #geometry.Bmag
        rhostar = 1.0 #geometry.rstar
        
        phi = log(densi)
        Er = -Dr(phi)
        Ez = -Dz(phi)
    
        Source_i = Dt(dfni) + ( vpa * (Bzed/Bmag) - 0.5*rhostar*Er ) * Dz(dfni) + ( 0.5*rhostar*Ez ) * Dr(dfni) + ( 0.5*Ez*Bzed/Bmag ) * Dvpa(dfni)
        return expand_derivatives(Source_i)
    end 
    
    densi, dfni = manufactured_solutions
    
    Source_i = manufactured_sources(dfni,densi)
    write(Source_i)
end
