module GyroAverageTests

include("setup.jl")

export gyroaverage_test

using MPI
using SpecialFunctions: besselj0

import moment_kinetics
using moment_kinetics.input_structs
using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.geo: init_magnetic_geometry
using moment_kinetics.communication
using moment_kinetics.looping
using moment_kinetics.array_allocation: allocate_float, allocate_shared_float
using moment_kinetics.gyroaverages: gyroaverage_pdf!
using moment_kinetics.gyroaverages: gyroaverage_field!, init_gyro_operators
using moment_kinetics.type_definitions: mk_float, mk_int

print_test_results = false

function runtests()
    # basic tests using periodic boundary conditions for the test function
    # functionality to change rhostar and vperp_max = Lvperp and the pitch = bzed is provided
    # the test could be sped up by moving lists of kz and kr inside the gyroaverage_test() function, 
    # so that the gyromatrix does not have to be reinitialised
    @testset " - test real-space path-integral gyroaverage (periodic functions)" begin
        ngrid_vperp = 3
        ngrid = 5; nelement = 4; ngrid_gyrophase = 100
        z_bc = "periodic"; r_bc = "periodic"
        kr = 1; kz = 1; Lvperp = 3.0; pitch = 0.5; rhostar = 0.1; phaser = 0.0; phasez = 0.0
        @testset "kr $kr kz $kz vperpmax $Lvperp rhostar $rhostar phaser $phaser phasez $phasez" begin
            absolute_error = 2.0e-4
            gyroaverage_test(absolute_error; rhostar=rhostar, pitch=pitch, ngrid=ngrid, kr=kr, kz=kz, phaser=phaser, phasez=phasez, nelement=nelement, ngrid_vperp=ngrid_vperp, nelement_vperp=1, Lvperp=Lvperp, ngrid_gyrophase=ngrid_gyrophase, discretization="chebyshev_pseudospectral", r_bc=r_bc, z_bc=z_bc)
        end
        kr = 1; kz = 3; Lvperp = 3.0; pitch = 0.5; rhostar = 0.1; phaser = 0.0; phasez = 0.0
        @testset "kr $kr kz $kz vperpmax $Lvperp rhostar $rhostar phaser $phaser phasez $phasez" begin
            absolute_error = 2.0e-2
            gyroaverage_test(absolute_error; rhostar=rhostar, pitch=pitch, ngrid=ngrid, kr=kr, kz=kz, phaser=phaser, phasez=phasez, nelement=nelement, ngrid_vperp=ngrid_vperp, nelement_vperp=1, Lvperp=Lvperp, ngrid_gyrophase=ngrid_gyrophase, discretization="chebyshev_pseudospectral", r_bc=r_bc, z_bc=z_bc)
        end
        kr = 3; kz = 1; Lvperp = 3.0; pitch = 0.5; rhostar = 0.1; phaser = 0.0; phasez = 0.0
        @testset "kr $kr kz $kz vperpmax $Lvperp rhostar $rhostar phaser $phaser phasez $phasez" begin
            absolute_error = 2.0e-2
            gyroaverage_test(absolute_error; rhostar=rhostar, pitch=pitch, ngrid=ngrid, kr=kr, kz=kz, phaser=phaser, phasez=phasez, nelement=nelement, ngrid_vperp=ngrid_vperp, nelement_vperp=1, Lvperp=Lvperp, ngrid_gyrophase=ngrid_gyrophase, discretization="chebyshev_pseudospectral", r_bc=r_bc, z_bc=z_bc)
        end
        ngrid = 5; nelement = 32; ngrid_gyrophase = 100
        z_bc = "periodic"; r_bc = "periodic"
        kr = 1; kz = 1; Lvperp = 3.0; pitch = 0.5; rhostar = 0.1; phaser = 0.0; phasez = 0.0
        @testset "kr $kr kz $kz vperpmax $Lvperp rhostar $rhostar phaser $phaser phasez $phasez" begin
            absolute_error = 2.0e-7
            gyroaverage_test(absolute_error; rhostar=rhostar, pitch=pitch, ngrid=ngrid, kr=kr, kz=kz, phaser=phaser, phasez=phasez, nelement=nelement, ngrid_vperp=ngrid_vperp, nelement_vperp=1, Lvperp=Lvperp, ngrid_gyrophase=ngrid_gyrophase, discretization="chebyshev_pseudospectral", r_bc=r_bc, z_bc=z_bc)
        end
        ngrid = 5; nelement = 32; ngrid_gyrophase = 100
        z_bc = "periodic"; r_bc = "periodic"
        kr = 1; kz = 3; Lvperp = 3.0; pitch = 0.5; rhostar = 0.1; phaser = 0.0; phasez = 0.0
        @testset "kr $kr kz $kz vperpmax $Lvperp rhostar $rhostar phaser $phaser phasez $phasez" begin
            absolute_error = 5.0e-6
            gyroaverage_test(absolute_error; rhostar=rhostar, pitch=pitch, ngrid=ngrid, kr=kr, kz=kz, phaser=phaser, phasez=phasez, nelement=nelement, ngrid_vperp=ngrid_vperp, nelement_vperp=1, Lvperp=Lvperp, ngrid_gyrophase=ngrid_gyrophase, discretization="chebyshev_pseudospectral", r_bc=r_bc, z_bc=z_bc)
        end
        ngrid = 5; nelement = 32; ngrid_gyrophase = 100
        z_bc = "periodic"; r_bc = "periodic"
        kr = 3; kz = 1; Lvperp = 3.0; pitch = 0.5; rhostar = 0.1; phaser = 0.0; phasez = 0.0
        @testset "kr $kr kz $kz vperpmax $Lvperp rhostar $rhostar phaser $phaser phasez $phasez" begin
            absolute_error = 5.0e-6
            gyroaverage_test(absolute_error; rhostar=rhostar, pitch=pitch, ngrid=ngrid, kr=kr, kz=kz, phaser=phaser, phasez=phasez, nelement=nelement, ngrid_vperp=ngrid_vperp, nelement_vperp=1, Lvperp=Lvperp, ngrid_gyrophase=ngrid_gyrophase, discretization="chebyshev_pseudospectral", r_bc=r_bc, z_bc=z_bc)
        end
    end
end

function gyroaverage_test(absolute_error; rhostar=0.1, pitch=0.5, ngrid=5, kr=2, kz=2, phaser=0.0, phasez=0.0, nelement=4, ngrid_vperp=3, nelement_vperp=1, Lvperp=3.0, ngrid_gyrophase=100, discretization="chebyshev_pseudospectral", r_bc="periodic", z_bc = "wall", print_test_results=print_test_results)

        #ngrid = 17
        #nelement = 4
        r_ngrid = ngrid #number of points per element 
        r_nelement_local = nelement # number of elements per rank
        r_nelement_global = r_nelement_local # total number of elements 
        r_L = 1.0

        z_ngrid = ngrid #number of points per element 
        z_nelement_local = nelement # number of elements per rank
        z_nelement_global = z_nelement_local # total number of elements 
        z_L = 1.0

        vperp_ngrid = ngrid_vperp #number of points per element 
        vperp_nelement_local = nelement_vperp # number of elements per rank
        vperp_nelement_global = vperp_nelement_local # total number of elements 
        vperp_L = Lvperp
        vperp_bc = "zero"
        
        vpa_ngrid = 1 #number of points per element 
        vpa_nelement_local = 1 # number of elements per rank
        vpa_nelement_global = vpa_nelement_local # total number of elements 
        vpa_L = 1.0
        vpa_bc = "" # should not be used
        
        gyrophase_ngrid = ngrid_gyrophase #number of points per element 
        gyrophase_nelement_local = 1 # number of elements per rank
        gyrophase_nelement_global = gyrophase_nelement_local # total number of elements 
        gyrophase_discretization = "finite_difference"
        gyrophase_L = 2.0*pi
        
        # fd_option and adv_input not actually used so given values unimportant
        fd_option = "fourth_order_centered"
        cheb_option = "matrix"
        adv_input = advection_input("default", 1.0, 0.0, 0.0)
        nrank = 1
        irank = 0#1
        comm = MPI.COMM_NULL
        element_spacing_option = "uniform"
        # create the 'input' struct containing input info needed to create a
        # coordinate
        
        r_input = grid_input("r", r_ngrid, r_nelement_global, r_nelement_local, 
                nrank, irank, r_L, discretization, fd_option, cheb_option, r_bc, adv_input,comm,element_spacing_option)
        z_input = grid_input("z", z_ngrid, z_nelement_global, z_nelement_local, 
                nrank, irank, z_L, discretization, fd_option, cheb_option, z_bc, adv_input,comm,element_spacing_option)
        vperp_input = grid_input("vperp", vperp_ngrid, vperp_nelement_global, vperp_nelement_local, 
                nrank, irank, vperp_L, discretization, fd_option, cheb_option, vperp_bc, adv_input,comm,element_spacing_option)
        vpa_input = grid_input("vpa", vpa_ngrid, vpa_nelement_global, vpa_nelement_local, 
                nrank, irank, vpa_L, discretization, fd_option, cheb_option, vpa_bc, adv_input,comm,element_spacing_option)
        gyrophase_input = grid_input("gyrophase", gyrophase_ngrid, gyrophase_nelement_global, gyrophase_nelement_local, 
                nrank, irank, gyrophase_L, gyrophase_discretization, fd_option, cheb_option, "periodic", adv_input,comm,element_spacing_option)
        
        # create the coordinate structs
        r, r_spectral = define_coordinate(r_input,init_YY=false)
        z, z_spectral = define_coordinate(z_input,init_YY=false)
        vperp, vperp_spectral = define_coordinate(vperp_input,init_YY=false)
        vpa, vpa_spectral = define_coordinate(vpa_input,init_YY=false)
        gyrophase, gyrophase_spectral = define_coordinate(gyrophase_input,init_YY=false)
        
        # create test geometry
        #rhostar = 0.1 #rhostar of ions for ExB drift
        option = "constant-helical"
        #pitch = 1.0
        DeltaB = 1.0
        geometry_in = geometry_input(rhostar,option,pitch,DeltaB)
        geometry = init_magnetic_geometry(geometry_in,z,r)
        
        # create test composition
        composition = create_test_composition()
            
        # Set up MPI
        initialize_comms!()
        setup_distributed_memory_MPI(1,1,1,1)
        looping.setup_loop_ranges!(block_rank[], block_size[];
                                       s=composition.n_ion_species, sn=1,
                                       r=r.n, z=z.n, vperp=vperp.n, vpa=vpa.n,
                                       vzeta=1, vr=1, vz=1)
                                       
        # initialise the matrix for the gyroaverages
        gyro = init_gyro_operators(vperp,z,r,gyrophase,geometry,composition,print_info=print_test_results)
        # initialise a test field
        phi = allocate_shared_float(z.n,r.n)
        gphi = allocate_shared_float(vperp.n,z.n,r.n)
        gphi_exact = allocate_float(vperp.n,z.n,r.n)
        gphi_err = allocate_float(vperp.n,z.n,r.n)
        begin_serial_region()
        @serial_region begin 
            fill_test_arrays!(phi,gphi_exact,vperp,z,r,geometry,kz,kr,phasez,phaser)
        end
        
        # gyroaverage phi
        gyroaverage_field!(gphi,phi,gyro,vperp,z,r)
        
        # compute errors
        begin_serial_region()
        @serial_region begin
            @. gphi_err = abs(gphi - gphi_exact)
            if print_test_results
                println("Test gyroaverage_field!()")
            end
            for ivperp in 1:vperp.n
                if print_test_results
                    println("ivperp: ",ivperp," max(abs(gphi_err)): ",maximum(gphi_err[ivperp,:,:])," max(abs(gphi)): ",maximum(gphi[ivperp,:,:]))
                end
                @test maximum(gphi_err[ivperp,:,:]) < absolute_error
            end
            if print_test_results
                println("")
            end
        end
        
        # repeat the test for a pdf
        # initialise a test field
        n_ion_species = composition.n_ion_species
        pdf = allocate_shared_float(vpa.n,vperp.n,z.n,r.n,n_ion_species)
        gpdf = allocate_shared_float(vpa.n,vperp.n,z.n,r.n,n_ion_species)
        gpdf_exact = allocate_float(vpa.n,vperp.n,z.n,r.n,n_ion_species)
        gpdf_err = allocate_float(vpa.n,vperp.n,z.n,r.n,n_ion_species)
        begin_serial_region()
        @serial_region begin
            fill_pdf_test_arrays!(pdf,gpdf_exact,vpa,vperp,z,r,composition,geometry,kz,kr,phasez,phaser)
        end
        
        gyroaverage_pdf!(gpdf,pdf,gyro,vpa,vperp,z,r,composition)
        # compute errors
        begin_serial_region()
        @serial_region begin
            @. gpdf_err = abs(gpdf - gpdf_exact)
            if print_test_results
                println("Test gyroaverage_pdf!()")
            end
            for is in 1:n_ion_species
                for ivperp in 1:vperp.n
                    for ivpa in 1:vpa.n
                        if print_test_results
                            println("ivpa: ",ivpa," ivperp: ",ivperp," is: ",is," max(abs(gpdf_err)): ",maximum(gpdf_err[ivpa,ivperp,:,:,is]),
                                " max(abs(gpdf)): ",maximum(gpdf[ivpa,ivperp,:,:,is]), " max(abs(gpdf_exact)): ",maximum(gpdf_exact[ivpa,ivperp,:,:,is]))
                        end
                        @test maximum(gpdf_err[ivpa,ivperp,:,:,is]) < absolute_error
                    end
                end
            end
            if print_test_results
                println("")
            end
        end
        
        finalize_comms!()
end

function create_test_composition()
    electron_physics = boltzmann_electron_response
    n_ion_species = 1
    n_neutral_species = 0
    n_species = n_ion_species + n_neutral_species
    use_test_neutral_wall_pdf = false
    # electron temperature over reference temperature
    T_e = 1.0
    # temperature at the entrance to the wall in terms of the electron temperature
    T_wall = 1.0
    # wall potential at z = 0
    phi_wall = 0.0
    # constant to test nonzero Er
    Er_constant = 0.0
    # ratio of the neutral particle mass to the ion particle mass
    mn_over_mi = 1.0
    # ratio of the electron particle mass to the ion particle mass
    me_over_mi = 1.0/1836.0
    # The ion flux reaching the wall that is recycled as neutrals is reduced by
    # `recycling_fraction` to account for ions absorbed by the wall.
    recycling_fraction = 1.0
    gyrokinetic_ions = true
    return composition = species_composition(n_species, n_ion_species, n_neutral_species,
            electron_physics, use_test_neutral_wall_pdf, T_e, T_wall, phi_wall, Er_constant,
            mn_over_mi, me_over_mi, recycling_fraction, gyrokinetic_ions, allocate_float(n_species))
end

function fill_test_arrays!(phi,gphi,vperp,z,r,geometry,kz,kr,phasez,phaser)
   for ir in 1:r.n
      for iz in 1:z.n
         Bmag = geometry.Bmag[iz,ir] 
         bzeta = geometry.bzeta[iz,ir] 
         rhostar = geometry.rhostar 
         # convert integer "wavenumbers" to actual wavenumbers 
         kkr = 2.0*pi*kr/r.L
         kkz = 2.0*pi*kz/z.L
         kperp = sqrt(kkr^2 + (bzeta*kkz)^2)
         
         phi[iz,ir] = sin(r.grid[ir]*kkr + phaser)*sin(z.grid[iz]*kkz + phasez)
         for ivperp in 1:vperp.n
            krho = kperp*vperp.grid[ivperp]*rhostar/Bmag
            # use that phi is a sum of Kronecker deltas in k space to write gphi
            gphi[ivperp,iz,ir] = besselj0(krho)*phi[iz,ir]
         end
      end
   end  
   return nothing
end

function fill_pdf_test_arrays!(pdf,gpdf,vpa,vperp,z,r,composition,geometry,kz,kr,phasez,phaser)
   for is in 1:composition.n_ion_species
       for ir in 1:r.n
          for iz in 1:z.n
             Bmag = geometry.Bmag[iz,ir] 
             bzeta = geometry.bzeta[iz,ir] 
             rhostar = geometry.rhostar 
             # convert integer "wavenumbers" to actual wavenumbers 
             kkr = 2.0*pi*kr/r.L
             kkz = 2.0*pi*kz/z.L
             kperp = sqrt(kkr^2 + (bzeta*kkz)^2)
             
             for ivperp in 1:vperp.n
                for ivpa in 1:vpa.n
                    pdf[ivpa,ivperp,iz,ir,is] = sin(r.grid[ir]*kkr + phaser)*sin(z.grid[iz]*kkz + phasez)
                    krho = kperp*vperp.grid[ivperp]*rhostar/Bmag
                    # use that pdf is a sum of Kronecker deltas in k space to write gpdf
                    gpdf[ivpa,ivperp,iz,ir,is] = besselj0(krho)*pdf[ivpa,ivperp,iz,ir,is]
                end
             end
          end
       end
   end  
   return nothing
end

end #GyroAverageTests

using .GyroAverageTests

GyroAverageTests.runtests()
