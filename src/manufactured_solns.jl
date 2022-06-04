"""
"""
module manufactured_solns

export manufactured_solutions
export manufactured_sources
export manufactured_solutions_as_arrays
export manufactured_rhs_as_array

using Symbolics

using ..array_allocation: allocate_float
using ..coordinates: coordinate
using ..input_structs: geometry_input
using ..type_definitions

    @variables r z vpa vperp t

    function densi_sym(Lr,Lz,r_bc,z_bc)
        if r_bc == "periodic" && z_bc == "periodic"
            densi = 1.0 +  0.1*(sin(2.0*pi*r/Lr) + sin(2.0*pi*z/Lz))*cos(2.0*pi*t)
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
            dfni = densi * (1.0 + 0.1*vpa*(sin(2.0*pi*r/Lr) + sin(2.0*pi*z/Lz))) *
                       exp( - vpa^2 - vperp^2) #/ sqrt(pi^3)
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

    function manufactured_rhs_sym(Lr,Lz,r_bc,z_bc,geometry)

        densi = densi_sym(Lr,Lz,r_bc,z_bc)
        dfni = dfni_sym(Lr,Lz,r_bc,z_bc)
        
        Dr = Differential(r) 
        Dz = Differential(z) 
        Dvpa = Differential(vpa) 
        Dvperp = Differential(vperp) 
    
        Bzed = geometry.Bzed
        Bmag = geometry.Bmag
        rhostar = geometry.rstar
        #Bzed = 1.0 #geometry.Bzed
        #Bmag = 1.0 #geometry.Bmag
        #rhostar = 1.0 #geometry.rstar
        
        phi = log(densi)
        Er = -Dr(phi)
        Ez = -Dz(phi)
    
        rhs = -( vpa * (Bzed/Bmag) - 0.5*rhostar*Er ) * Dz(dfni) - ( 0.5*rhostar*Ez ) * Dr(dfni) - ( 0.5*Ez*Bzed/Bmag ) * Dvpa(dfni)

        return expand_derivatives(rhs)
    end

    function manufactured_rhs(Lr,Lz,r_bc,z_bc,geometry)
        rhs_sym = manufactured_rhs_sym(Lr,Lz,r_bc,z_bc,geometry)
        return build_function(rhs_sym, vpa, vperp, z, r, t, expression=Val{false})
    end

    #function manufactured_sources(dfni,densi,geometry)
    function manufactured_sources(Lr,Lz,r_bc,z_bc,geometry)

        dfni = dfni_sym(Lr,Lz,r_bc,z_bc)

        Dt = Differential(t)

        S = Dt(dfni) - manufactured_rhs_sym(Lr,Lz,r_bc,z_bc,geometry)
        Source_i = expand_derivatives(S)
        
        Source_i_func = build_function(Source_i, vpa, vperp, z, r, t, expression=Val{false})
        return Source_i_func
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

        dfni_func, densi_func = manufactured_solutions(r.L, z.L, r.bc, z.bc)

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
            t::mk_float, r::AbstractVector, z::AbstractVector, vperp::AbstractVector,
            vpa::AbstractVector, geometry::geometry_input)

    Create array filled with manufactured rhs.

    Returns
    -------
    rhs
    """
    function manufactured_rhs_as_array(
        t::mk_float, r::coordinate, z::coordinate, vperp::coordinate,
        vpa::coordinate, geometry::geometry_input)

        rhs_func = manufactured_rhs(r.L, z.L, r.bc, z.bc, geometry)

        rhs = allocate_float(vpa.n, vperp.n, z.n, r.n)

        for ir ∈ 1:r.n, iz ∈ 1:z.n
            for ivperp ∈ 1:vperp.n, ivpa ∈ 1:vpa.n
                rhs[ivpa,ivperp,iz,ir] = rhs_func(vpa.grid[ivpa], vperp.grid[ivperp],
                                                  z.grid[iz], r.grid[ir], t)
            end
        end

        return rhs
    end

end
