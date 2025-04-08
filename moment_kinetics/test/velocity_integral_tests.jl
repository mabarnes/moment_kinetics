module VelocityIntegralTests

include("setup.jl")

using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.velocity_moments: get_density, get_upar, get_ppar, get_p
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
        discretization = "chebyshev_pseudospectral"
        cheb_option = "FFT"
        # create the 'input' struct containing input info needed to create
        # coordinates
        coords_input = OptionsDict("vperp1d"=>OptionsDict("ngrid"=>1, "nelement"=>1, "nelement_local"=>1, "L"=>1.0,
                                                          "discretization"=>discretization,
                                                          "cheb_option"=>cheb_option, "bc"=>bc,
                                                          "element_spacing_option"=>"uniform"),
                                   "vpa1d"=>OptionsDict("ngrid"=>ngrid, "nelement"=>nelement_global,
                                                        "nelement_local"=>nelement_local, "L"=>Lvpa,
                                                        "discretization"=>discretization,
                                                        "cheb_option"=>cheb_option, "bc"=>bc,
                                                        "element_spacing_option"=>"uniform"),
                                   "vperp"=>OptionsDict("ngrid"=>ngrid, "nelement"=>nelement_global,
                                                        "nelement_local"=>nelement_local, "L"=>Lvperp,
                                                        "discretization"=>discretization,
                                                        "cheb_option"=>cheb_option, "bc"=>bc,
                                                        "element_spacing_option"=>"uniform"),
                                   "vpa"=>OptionsDict("ngrid"=>ngrid, "nelement"=>nelement_global,
                                                      "nelement_local"=>nelement_local, "L"=>Lvpa,
                                                      "discretization"=>discretization,
                                                      "cheb_option"=>cheb_option, "bc"=>bc,
                                                      "element_spacing_option"=>"uniform"),
                                  )

        # create the coordinate structs
        vpa, vpa_spectral = define_coordinate(coords_input, "vpa"; ignore_MPI=true)
        vperp, vperp_spectral = define_coordinate(coords_input, "vperp"; ignore_MPI=true)
        vz, vz_spectral = define_coordinate(coords_input, "vpa1d"; ignore_MPI=true)
        vr, vr_spectral = define_coordinate(coords_input, "vperp1d"; ignore_MPI=true)

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
            pres = pressure(ppar,pperp)
            mass = 1.0
            vth = sqrt(2.0*pres/(dens*mass))
            for ivperp in 1:vperp.n
                for ivpa in 1:vpa.n
                    vpa_val = vpa.grid[ivpa]
                    vperp_val = vperp.grid[ivperp]
                    dfn[ivpa,ivperp] = (dens/vth^3/π^1.5)*exp( - ((vpa_val-upar)^2 + vperp_val^2)/vth^2 )
                end
            end

            # now check that we can extract the correct moments from the distribution

            dens_test = get_density(dfn,vpa,vperp)
            upar_test = get_upar(dfn,dens_test,vpa,vperp,false)
            pres_test = get_p(dfn,dens_test,upar_test,vpa,vperp,false,false)
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
                    dfn1D[ivz,ivr] = (dens/vth/sqrt(π))*exp( - ((vz_val-upar)^2)/vth^2 )
                end
            end
            dens_test = get_density(dfn1D,vz,vr)
            upar_test = get_upar(dfn1D,dens_test,vz,vr,false)
            ppar_test = get_ppar(nothing,upar_test,nothing,nothing,dfn1D,vz,vr,false,false,false)
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
                    dfn[ivpa,ivperp] = (dens/(vthpar*vthperp^2)/π^1.5)*exp( - ((vpa_val-upar)^2)/vthpar^2 - (vperp_val^2)/vthperp^2 )
                end
            end

            # now check that we can extract the correct moments from the distribution

            dens_test = get_density(dfn,vpa,vperp)
            upar_test = get_upar(dfn,dens_test,vpa,vperp,false)
            ppar_test = get_ppar(nothing,upar_test,nothing,nothing,dfn,vpa,vperp,false,false,false)

            @test isapprox(dens_test, dens; atol=atol)
            @test isapprox(upar_test, upar; atol=atol)
            @test isapprox(ppar_test, ppar; atol=atol)
        end
    end
end 

end # VelocityIntegralTests

using .VelocityIntegralTests

VelocityIntegralTests.runtests()
