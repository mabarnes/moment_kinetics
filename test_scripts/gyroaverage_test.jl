export gyroaverage_test

using Printf
using Plots
using LaTeXStrings
using MPI
using Measures
using SpecialFunctions: besselj0

import moment_kinetics
using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.input_structs
using moment_kinetics.geo: init_magnetic_geometry
using moment_kinetics.communication
using moment_kinetics.looping
using moment_kinetics.array_allocation: allocate_float, allocate_shared_float
using moment_kinetics.gyroaverages: gyroaverage_pdf!
using moment_kinetics.gyroaverages: gyroaverage_field!, init_gyro_operators
using moment_kinetics.species_input: get_species_input
using moment_kinetics.type_definitions: mk_float, mk_int

function print_matrix(matrix,name::String,n::mk_int,m::mk_int)
        println("\n ",name," \n")
        for i in 1:n
            for j in 1:m
                @printf("%.2f ", matrix[i,j])
            end
            println("")
        end
        println("\n")
    end
    
    function print_vector(vector,name::String,m::mk_int)
        println("\n ",name," \n")
        for j in 1:m
            @printf("%.3f ", vector[j])
        end
        println("")
        println("\n")
    end


function gyroaverage_test(;rhostar=0.1, pitch=0.5, ngrid=5, kr=2, kz=2, phaser=0.0, phasez=0.0, nelement=4, ngrid_vperp=3, nelement_vperp=1, Lvperp=3.0, ngrid_gyrophase=100, discretization="chebyshev_pseudospectral", r_bc="periodic", z_bc = "wall", plot_test_results=false)

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
        
        fd_option = "fourth_order_centered"
        cheb_option = "matrix"
        element_spacing_option = "uniform"
        # create the 'input' struct containing input info needed to create a
        # coordinate
        
        coords_input = OptionsDict(
            "r"=>OptionsDict("ngrid"=>r_ngrid, "nelement"=>r_nelement_global,
                             "nelement_local"=>r_nelement_local, "L"=>r_L,
                             "discretization"=>discretization,
                             "finite_difference_option"=>fd_option,
                             "cheb_option"=>cheb_option, "bc"=>r_bc,
                             "element_spacing_option"=>element_spacing_option),
            "z"=>OptionsDict("ngrid"=>z_ngrid, "nelement"=>z_nelement_global,
                             "nelement_local"=>z_nelement_local, "L"=>z_L,
                             "discretization"=>discretization,
                             "finite_difference_option"=>fd_option,
                             "cheb_option"=>cheb_option, "bc"=>z_bc,
                             "element_spacing_option"=>element_spacing_option),
            "vperp"=>OptionsDict("ngrid"=>vperp_ngrid, "nelement"=>vperp_nelement_global,
                                 "nelement_local"=>vperp_nelement_local, "L"=>vperp_L,
                                 "discretization"=>discretization,
                                 "finite_difference_option"=>fd_option,
                                 "cheb_option"=>cheb_option, "bc"=>vperp_bc,
                                 "element_spacing_option"=>element_spacing_option),
            "vpa"=>OptionsDict("ngrid"=>vpa_ngrid, "nelement"=>vpa_nelement_global,
                               "nelement_local"=>vpa_nelement_local, "L"=>vpa_L,
                               "discretization"=>discretization,
                               "finite_difference_option"=>fd_option,
                               "cheb_option"=>cheb_option, "bc"=>vpa_bc,
                               "element_spacing_option"=>element_spacing_option),
            "gyrophase"=>OptionsDict("ngrid"=>gyrophase_ngrid,
                                     "nelement"=>gyrophase_nelement_global,
                                     "nelement_local"=>gyrophase_nelement_local,
                                     "L"=>gyrophase_L, "discretization"=>discretization,
                                     "finite_difference_option"=>fd_option,
                                     "cheb_option"=>cheb_option, "bc"=>"periodic",
                                     "element_spacing_option"=>element_spacing_option),
        )
        
        # create the coordinate structs
        r, r_spectral = define_coordinate(coords_input, "r"; collision_operator_dim=false)
        z, z_spectral = define_coordinate(coords_input, "z"; collision_operator_dim=false)
        vperp, vperp_spectral = define_coordinate(coords_input, "vperp";
                                                  collision_operator_dim=false)
        vpa, vpa_spectral = define_coordinate(coords_input, "vpa";
                                              collision_operator_dim=false)
        gyrophase, gyrophase_spectral = define_coordinate(coords_input, "gyrophase";
                                                          collision_operator_dim=false)
        
        # create test geometry
        #rhostar = 0.1 #rhostar of ions for ExB drift
        option = "constant-helical"
        #pitch = 1.0
        DeltaB = 1.0
        geometry_in = geometry_input(rhostar,option,pitch,DeltaB,0.0,0.0,0.0,0.0)
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
        gyro = init_gyro_operators(vperp,z,r,gyrophase,geometry,composition)
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
        gyroaverage_field!(gphi,phi,gyro,vperp,z,r,composition)
        
        # compute errors
        begin_serial_region()
        @serial_region begin
            @. gphi_err = abs(gphi - gphi_exact)
            println("Test gyroaverage_field!()")
            for ivperp in 1:vperp.n
                println("ivperp: ",ivperp," max(abs(gphi_err)): ",maximum(gphi_err[ivperp,:,:])," max(abs(gphi)): ",maximum(gphi[ivperp,:,:]))
            end
            println("")
            if plot_test_results
                @views heatmap(r.grid, z.grid, phi[:,:], xlabel=L"r", ylabel=L"z", c = :deep, interpolation = :cubic,
                    windowsize = (360,240), margin = 15pt)
                outfile = "phi_vs_r_z.pdf"
                savefig(outfile)
                println("Saved outfile: "*outfile)
                for ivperp in 1:vperp.n
                    @views heatmap(r.grid, z.grid, gphi[ivperp,:,:], xlabel=L"r", ylabel=L"z", c = :deep, interpolation = :cubic,
                        windowsize = (360,240), margin = 15pt)
                    outfile = "gphi_ivperp_"*string(ivperp)*"_vs_r_z.pdf"
                    savefig(outfile)
                    println("Saved outfile: "*outfile)
                end
            end
        end
        
        # repeat the test for a pdf
        # initialise a test field
        nvpa = 1
        n_ion_species = composition.n_ion_species
        pdf = allocate_shared_float(nvpa,vperp.n,z.n,r.n,n_ion_species)
        gpdf = allocate_shared_float(nvpa,vperp.n,z.n,r.n,n_ion_species)
        gpdf_exact = allocate_float(nvpa,vperp.n,z.n,r.n,n_ion_species)
        gpdf_err = allocate_float(nvpa,vperp.n,z.n,r.n,n_ion_species)
        begin_serial_region()
        @serial_region begin
            fill_pdf_test_arrays!(pdf,gpdf_exact,vpa,vperp,z,r,composition,geometry,kz,kr,phasez,phaser)
        end
        
        gyroaverage_pdf!(gpdf,pdf,gyro,vpa,vperp,z,r,composition)
        # compute errors
        begin_serial_region()
        @serial_region begin
            @. gpdf_err = abs(gpdf - gpdf_exact)
            println("Test gyroaverage_pdf!()")
            for is in 1:n_ion_species
                for ivperp in 1:vperp.n
                    for ivpa in 1:nvpa
                        println("ivpa: ",ivpa," ivperp: ",ivperp," is: ",is," max(abs(gphi_err)): ",maximum(gpdf_err[ivpa,ivperp,:,:,is]),
                         " max(abs(gpdf)): ",maximum(gpdf[ivpa,ivperp,:,:,is]))
                    end
                end
            end
            println("")
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
    ion_physics = gyrokinetic_ions
    species_opts = OptionsDict("n_ion_species" => n_ion_species,
                               "n_neutral_species" => n_neutral_species, "T_e" => T_e,
                               "T_wall" => T_wall, "phi_wall" => phi_wall,
                               "mn_over_mi" => mn_over_mi, "me_over_mi" => me_over_mi,
                               "recycling_fraction" => recycling_fraction,
                               "ion_physics" => ion_physics)
    return get_species_input(OptionsDict("composition" => species_opts), false)
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

if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")
    # example function call with arguments for a successful test
    #gyroaverage_test()
    gyroaverage_test(rhostar=0.01,pitch=0.5,kr=2,kz=3,phaser=0.25*pi,phasez=0.5*pi,ngrid=9,nelement=8,ngrid_vperp=5,nelement_vperp=3,Lvperp=18.0,ngrid_gyrophase=100,r_bc="periodic",z_bc="periodic")
end
