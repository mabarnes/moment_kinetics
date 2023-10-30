module VelocityIntegralTests

include("setup.jl")

using moment_kinetics.input_structs: grid_input, advection_input
using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.velocity_moments: get_density, get_upar, get_ppar, get_pperp, get_pressure
using moment_kinetics.array_allocation: allocate_float

using MPI

function runtests()
    @testset "velocity integral tests" verbose=use_verbose begin
        println("velocity integral tests")

        # Tolerance for tests
        atol = 1.0e-13

        # define inputs needed for the test
        ngrid = 17 #number of points per element 
        nelement_local = 20 # number of elements per rank
        nelement_global = nelement_local # total number of elements 
        Lvpa = 18.0 #physical box size in reference units 
        Lvperp = 9.0 #physical box size in reference units 
        bc = "" #not required to take a particular value, not used 
        # fd_option and adv_input not actually used so given values unimportant
        discretization = "chebyshev_pseudospectral"
        fd_option = "fourth_order_centered"
        adv_input = advection_input("default", 1.0, 0.0, 0.0)
        nrank = 1
        irank = 0
        comm = MPI.COMM_NULL
        # create the 'input' struct containing input info needed to create a
        # coordinate
        vr_input = grid_input("vperp", 1, 1, 1, nrank, irank, 1.0, discretization,
                              fd_option, bc, adv_input, comm, "uniform")
        vz_input = grid_input("vpa", ngrid, nelement_global, nelement_local, nrank, irank,
                              Lvpa, discretization, fd_option, bc, adv_input, comm,
                              "uniform")
        vpa_input = grid_input("vpa", ngrid, nelement_global, nelement_local, nrank,
                               irank, Lvpa, discretization, fd_option, bc, adv_input,
                               comm, "uniform")
        vperp_input = grid_input("vperp", ngrid, nelement_global, nelement_local, nrank,
                                 irank, Lvperp, discretization, fd_option, bc, adv_input,
                                 comm, "uniform")
        # create the coordinate struct 'x'
        vpa, vpa_spectral = define_coordinate(vpa_input)
        vperp, vperp_spectral = define_coordinate(vperp_input)
        vz, vz_spectral = define_coordinate(vz_input)
        vr, vr_spectral = define_coordinate(vr_input)

        dfn = allocate_float(vpa.n,vperp.n)
        dfn1D = allocate_float(vz.n, vr.n)

        function pressure(ppar,pperp)
            pres = (1.0/3.0)*(ppar + 2.0*pperp) 
            return pres
        end

        @testset "2D isotropic Maxwellian" begin
            # assign a known isotropic Maxwellian distribution in normalised units
            dens = 3.0/4.0
            upar = 2.0/3.0
            ppar = 2.0/3.0
            pperp = 2.0/3.0
            pres = get_pressure(ppar,pperp) 
            mass = 1.0
            vth = sqrt(2.0*pres/(dens*mass))
            for ivperp in 1:vperp.n
                for ivpa in 1:vpa.n
                    vpa_val = vpa.grid[ivpa]
                    vperp_val = vperp.grid[ivperp]
                    dfn[ivpa,ivperp] = (dens/vth^3)*exp( - ((vpa_val-upar)^2 + vperp_val^2)/vth^2 )
                end
            end

            # now check that we can extract the correct moments from the distribution

            dens_test = get_density(dfn,vpa,vperp)
            upar_test = get_upar(dfn,vpa,vperp,dens_test)
            ppar_test = get_ppar(dfn,vpa,vperp,upar_test)
            pperp_test = get_pperp(dfn,vpa,vperp)
            pres_test = pressure(ppar_test,pperp_test)
            # output test results 
            @test isapprox(dens_test, dens; atol=atol)
            @test isapprox(upar_test, upar; atol=atol)
            @test isapprox(pres_test, pres; atol=atol)
        end

        @testset "1D Maxwellian" begin
            dens = 3.0/4.0
            upar = 2.0/3.0
            ppar = 2.0/3.0 
            mass = 1.0
            vth = sqrt(2.0*ppar/(dens*mass))
            for ivz in 1:vz.n
                for ivr in 1:vr.n
                    vz_val = vz.grid[ivz]
                    dfn1D[ivz,ivr] = (dens/vth)*exp( - ((vz_val-upar)^2)/vth^2 )
                end
            end
            dens_test = get_density(dfn1D,vz,vr)
            upar_test = get_upar(dfn1D,vz,vr,dens_test)
            ppar_test = get_ppar(dfn1D,vz,vr,upar_test)
            # output test results 
            @test isapprox(dens_test, dens; atol=atol)
            @test isapprox(upar_test, upar; atol=atol)
            @test isapprox(ppar_test, ppar; atol=atol)
        end

        @testset "biMaxwellian" begin
            # assign a known biMaxwellian distribution in normalised units
            dens = 3.0/4.0
            upar = 2.0/3.0
            ppar = 4.0/5.0
            pperp = 1.0/4.0 
            mass = 1.0
            vthpar = sqrt(2.0*ppar/(dens*mass))
            vthperp = sqrt(2.0*pperp/(dens*mass))
            for ivperp in 1:vperp.n
                for ivpa in 1:vpa.n
                    vpa_val = vpa.grid[ivpa]
                    vperp_val = vperp.grid[ivperp]
                    dfn[ivpa,ivperp] = (dens/(vthpar*vthperp^2))*exp( - ((vpa_val-upar)^2)/vthpar^2 - (vperp_val^2)/vthperp^2 )
                end
            end

            # now check that we can extract the correct moments from the distribution

            dens_test = get_density(dfn,vpa,vperp)
            upar_test = get_upar(dfn,vpa,vperp,dens_test)
            ppar_test = get_ppar(dfn,vpa,vperp,upar_test)
            pperp_test = get_pperp(dfn,vpa,vperp)
            # output test results 

            @test isapprox(dens_test, dens; atol=atol)
            @test isapprox(upar_test, upar; atol=atol)
            @test isapprox(ppar_test, ppar; atol=atol)
        end
    end
end 

end # VelocityIntegralTests

using .VelocityIntegralTests

VelocityIntegralTests.runtests()
