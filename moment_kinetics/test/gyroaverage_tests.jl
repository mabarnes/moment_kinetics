module GyroAverageTests

include("setup.jl")

export gyroaverage_test

using MPI
using SpecialFunctions: besselj0

import moment_kinetics
using moment_kinetics.input_structs
using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.geo: init_magnetic_geometry, setup_geometry_input
using moment_kinetics.communication
using moment_kinetics.looping
using moment_kinetics.array_allocation: allocate_float, allocate_shared_float
using moment_kinetics.gyroaverages: gyroaverage_pdf!
using moment_kinetics.gyroaverages: gyroaverage_field!, init_gyro_operators
using moment_kinetics.type_definitions: mk_float, mk_int
using moment_kinetics.species_input: get_species_input

print_test_results = false

function runtests()
    # basic tests using periodic boundary conditions for the test function
    # functionality to change rhostar and vperp_max = Lvperp and the pitch = bzed is provided
    # the test could be sped up by moving lists of kz and kr inside the gyroaverage_test() function, 
    # so that the gyromatrix does not have to be reinitialised

    # Only needed to save FFTW 'wisdom'
    test_output_directory = get_MPI_tempdir()

    @testset "Gyroaverage tests" verbose=use_verbose begin
        println("Gyroaverages test")
        @testset " - test real-space path-integral gyroaverage (periodic functions)" begin
            ngrid_vperp = 3
            ngrid = 5; nelement = 4; ngrid_gyrophase = 100
            z_bc = "periodic"; r_bc = "periodic"
            kr = 1; kz = 1; Lvperp = 3.0; pitch = 0.5; rhostar = 0.1; phaser = 0.0; phasez = 0.0
            @testset "kr $kr kz $kz vperpmax $Lvperp rhostar $rhostar phaser $phaser phasez $phasez" begin
                absolute_error = 2.0e-4
                gyroaverage_test(absolute_error; rhostar=rhostar, pitch=pitch, ngrid=ngrid, kr=kr, kz=kz, phaser=phaser, phasez=phasez, nelement=nelement, ngrid_vperp=ngrid_vperp, nelement_vperp=1, Lvperp=Lvperp, ngrid_gyrophase=ngrid_gyrophase, discretization="chebyshev_pseudospectral", r_bc=r_bc, z_bc=z_bc, test_output_directory=test_output_directory)
            end
            kr = 1; kz = 3; Lvperp = 3.0; pitch = 0.5; rhostar = 0.1; phaser = 0.0; phasez = 0.0
            @testset "kr $kr kz $kz vperpmax $Lvperp rhostar $rhostar phaser $phaser phasez $phasez" begin
                absolute_error = 2.0e-2
                gyroaverage_test(absolute_error; rhostar=rhostar, pitch=pitch, ngrid=ngrid, kr=kr, kz=kz, phaser=phaser, phasez=phasez, nelement=nelement, ngrid_vperp=ngrid_vperp, nelement_vperp=1, Lvperp=Lvperp, ngrid_gyrophase=ngrid_gyrophase, discretization="chebyshev_pseudospectral", r_bc=r_bc, z_bc=z_bc, test_output_directory=test_output_directory)
            end
            kr = 3; kz = 1; Lvperp = 3.0; pitch = 0.5; rhostar = 0.1; phaser = 0.0; phasez = 0.0
            @testset "kr $kr kz $kz vperpmax $Lvperp rhostar $rhostar phaser $phaser phasez $phasez" begin
                absolute_error = 2.0e-2
                gyroaverage_test(absolute_error; rhostar=rhostar, pitch=pitch, ngrid=ngrid, kr=kr, kz=kz, phaser=phaser, phasez=phasez, nelement=nelement, ngrid_vperp=ngrid_vperp, nelement_vperp=1, Lvperp=Lvperp, ngrid_gyrophase=ngrid_gyrophase, discretization="chebyshev_pseudospectral", r_bc=r_bc, z_bc=z_bc, test_output_directory=test_output_directory)
            end
            ngrid = 5; nelement = 8; ngrid_gyrophase = 100
            z_bc = "periodic"; r_bc = "periodic"
            kr = 1; kz = 1; Lvperp = 3.0; pitch = 0.5; rhostar = 0.1; phaser = 0.0; phasez = 0.0
            @testset "kr $kr kz $kz vperpmax $Lvperp rhostar $rhostar phaser $phaser phasez $phasez" begin
                absolute_error = 4.0e-6
                gyroaverage_test(absolute_error; rhostar=rhostar, pitch=pitch, ngrid=ngrid, kr=kr, kz=kz, phaser=phaser, phasez=phasez, nelement=nelement, ngrid_vperp=ngrid_vperp, nelement_vperp=1, Lvperp=Lvperp, ngrid_gyrophase=ngrid_gyrophase, discretization="chebyshev_pseudospectral", r_bc=r_bc, z_bc=z_bc, test_output_directory=test_output_directory)
            end
            ngrid = 5; nelement = 8; ngrid_gyrophase = 100
            z_bc = "periodic"; r_bc = "periodic"
            kr = 1; kz = 3; Lvperp = 3.0; pitch = 0.5; rhostar = 0.1; phaser = 0.0; phasez = 0.0
            @testset "kr $kr kz $kz vperpmax $Lvperp rhostar $rhostar phaser $phaser phasez $phasez" begin
                absolute_error = 3.0e-3
                gyroaverage_test(absolute_error; rhostar=rhostar, pitch=pitch, ngrid=ngrid, kr=kr, kz=kz, phaser=phaser, phasez=phasez, nelement=nelement, ngrid_vperp=ngrid_vperp, nelement_vperp=1, Lvperp=Lvperp, ngrid_gyrophase=ngrid_gyrophase, discretization="chebyshev_pseudospectral", r_bc=r_bc, z_bc=z_bc, test_output_directory=test_output_directory)
            end
            ngrid = 5; nelement = 8; ngrid_gyrophase = 100
            z_bc = "periodic"; r_bc = "periodic"
            kr = 3; kz = 1; Lvperp = 3.0; pitch = 0.5; rhostar = 0.1; phaser = 0.0; phasez = 0.0
            @testset "kr $kr kz $kz vperpmax $Lvperp rhostar $rhostar phaser $phaser phasez $phasez" begin
                absolute_error = 3.0e-3
                gyroaverage_test(absolute_error; rhostar=rhostar, pitch=pitch, ngrid=ngrid, kr=kr, kz=kz, phaser=phaser, phasez=phasez, nelement=nelement, ngrid_vperp=ngrid_vperp, nelement_vperp=1, Lvperp=Lvperp, ngrid_gyrophase=ngrid_gyrophase, discretization="chebyshev_pseudospectral", r_bc=r_bc, z_bc=z_bc, test_output_directory=test_output_directory)
            end
        end
    end
end

function gyroaverage_test(absolute_error; rhostar=0.1, pitch=0.5, ngrid=5, kr=2, kz=2, phaser=0.0, phasez=0.0, nelement=4, ngrid_vperp=3, nelement_vperp=1, Lvperp=3.0, ngrid_gyrophase=100, discretization="chebyshev_pseudospectral", r_bc="periodic", z_bc = "wall", print_test_results=print_test_results, test_output_directory)

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
        gyrophase_bc = "periodic"
        
        cheb_option = "matrix"
        element_spacing_option = "uniform"
        # create the 'input' struct containing input info needed to create a
        # coordinate
        
        # Set up MPI
        initialize_comms!()
        setup_distributed_memory_MPI(1,1,1,1)
        irank_z, nrank_z, comm_sub_z, irank_r, nrank_r, comm_sub_r = setup_distributed_memory_MPI(z_nelement_global,z_nelement_local,r_nelement_global,r_nelement_local)
        
        coords_input = OptionsDict(
            "r"=>OptionsDict("ngrid"=>r_ngrid, "nelement"=>r_nelement_global,
                             "nelement_local"=>r_nelement_local, "L"=>r_L,
                             "discretization"=>discretization, "cheb_option"=>cheb_option,
                             "bc"=>r_bc,
                             "element_spacing_option"=>element_spacing_option),
            "z"=>OptionsDict("ngrid"=>z_ngrid, "nelement"=>z_nelement_global,
                             "nelement_local"=>z_nelement_local, "L"=>z_L,
                             "discretization"=>discretization, "cheb_option"=>cheb_option,
                             "bc"=>z_bc, "element_spacing_option"=>element_spacing_option),
            "vperp"=>OptionsDict("ngrid"=>vperp_ngrid, "nelement"=>vperp_nelement_global,
                                 "nelement_local"=>vperp_nelement_local, "L"=>vperp_L,
                                 "discretization"=>discretization,
                                 "cheb_option"=>cheb_option, "bc"=>vperp_bc,
                                 "element_spacing_option"=>element_spacing_option),
            "vpa"=>OptionsDict("ngrid"=>vpa_ngrid, "nelement"=>vpa_nelement_global,
                               "nelement_local"=>vpa_nelement_local, "L"=>vpa_L,
                               "discretization"=>discretization,
                               "cheb_option"=>cheb_option, "bc"=>vpa_bc,
                               "element_spacing_option"=>element_spacing_option),
            "gyrophase"=>OptionsDict("ngrid"=>gyrophase_ngrid,
                                     "nelement"=>gyrophase_nelement_global,
                                     "nelement_local"=>gyrophase_nelement_local,
                                     "L"=>gyrophase_L, "discretization"=>discretization,
                                     "cheb_option"=>cheb_option, "bc"=>gyrophase_bc,
                                     "element_spacing_option"=>element_spacing_option),
        )
        
        # create the coordinate structs
        r, r_spectral = define_coordinate(coords_input, "r"; collision_operator_dim=false,
                                          run_directory=test_output_directory,
                                          irank=irank_r, nrank=nrank_r, comm=comm_sub_r)
        z, z_spectral = define_coordinate(coords_input, "z"; collision_operator_dim=false,
                                          run_directory=test_output_directory,
                                          irank=irank_z, nrank=nrank_z, comm=comm_sub_z)
        vperp, vperp_spectral = define_coordinate(coords_input, "vperp";
                                                  collision_operator_dim=false,
                                                  run_directory=test_output_directory)
        vpa, vpa_spectral = define_coordinate(coords_input, "vpa";
                                              collision_operator_dim=false,
                                              run_directory=test_output_directory)
        gyrophase, gyrophase_spectral = define_coordinate(coords_input, "gyrophase";
                                                          collision_operator_dim=false,
                                                          run_directory=test_output_directory)
        
        # create test geometry
        option = "constant-helical"
        inputdict = OptionsDict("geometry" => OptionsDict("option" => option, "rhostar" => rhostar, "pitch" => pitch))
        geometry_in = setup_geometry_input(inputdict, true)
        geometry = init_magnetic_geometry(geometry_in,z,r)
        
        # create test composition
        composition = create_test_composition()
            
        # setup shared-memory MPI ranges
        looping.setup_loop_ranges!(block_rank[], block_size[];
                                       s=composition.n_ion_species, sn=1,
                                       r=r.n, z=z.n, vperp=vperp.n, vpa=vpa.n,
                                       vzeta=1, vr=1, vz=1)
                                       
        # initialise the matrix for the gyroaverages
        gyro = init_gyro_operators(vperp,z,r,gyrophase,geometry,composition,print_info=print_test_results)
        # initialise a test field
        phi = allocate_shared_float(z.n,r.n)
        gphi = allocate_shared_float(vperp.n,z.n,r.n,composition.n_ion_species)
        gphi_exact = allocate_float(vperp.n,z.n,r.n,composition.n_ion_species)
        gphi_err = allocate_float(vperp.n,z.n,r.n)
        begin_serial_region()
        @serial_region begin 
            fill_test_arrays!(phi,gphi_exact,vperp,z,r,geometry,composition,kz,kr,phasez,phaser)
        end
        
        # gyroaverage phi
        gyroaverage_field!(gphi,phi,gyro,vperp,z,r,composition)
        
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
    input_dict = OptionsDict("composition" => OptionsDict("n_ion_species" => 1, "n_neutral_species" => 0, "ion_physics" => "gyrokinetic_ions") )
    #println(input_dict)
    return get_species_input(input_dict, true)
end

function fill_test_arrays!(phi,gphi,vperp,z,r,geometry,composition,kz,kr,phasez,phaser)
   n_ion_species = composition.n_ion_species
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
         for is in 1:n_ion_species
             for ivperp in 1:vperp.n
                krho = kperp*vperp.grid[ivperp]*rhostar/Bmag
                # use that phi is a sum of Kronecker deltas in k space to write gphi
                gphi[ivperp,iz,ir,is] = besselj0(krho)*phi[iz,ir]
             end
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
