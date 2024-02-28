export gyroaverage_test

using Printf
using Plots
using LaTeXStrings
using MPI
using Measures

import moment_kinetics
using moment_kinetics.input_structs
using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.geo: init_magnetic_geometry
using moment_kinetics.communication
using moment_kinetics.looping
using moment_kinetics.array_allocation: allocate_float
using moment_kinetics.gyroaverages: gyroaverage_field!, init_gyro_operators
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


function gyroaverage_test(; ngrid=5, nelement=4, ngrid_vperp=3, nelement_vperp=1, Lvperp=3.0, ngrid_gyrophase=5, discretization="chebyshev_pseudospectral")

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
        
        gyrophase_ngrid = ngrid_gyrophase #number of points per element 
        gyrophase_nelement_local = 1 # number of elements per rank
        gyrophase_nelement_global = gyrophase_nelement_local # total number of elements 
        gyrophase_discretization = "finite_difference"
        gyrophase_L = 2.0*pi
        
        bc = "zero" 
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
                nrank, irank, r_L, discretization, fd_option, cheb_option, bc, adv_input,comm,element_spacing_option)
        z_input = grid_input("z", z_ngrid, z_nelement_global, z_nelement_local, 
                nrank, irank, z_L, discretization, fd_option, cheb_option, bc, adv_input,comm,element_spacing_option)
        vperp_input = grid_input("vperp", vperp_ngrid, vperp_nelement_global, vperp_nelement_local, 
                nrank, irank, vperp_L, discretization, fd_option, cheb_option, bc, adv_input,comm,element_spacing_option)
        gyrophase_input = grid_input("gyrophase", gyrophase_ngrid, gyrophase_nelement_global, gyrophase_nelement_local, 
                nrank, irank, gyrophase_L, gyrophase_discretization, fd_option, cheb_option, bc, adv_input,comm,element_spacing_option)
        
        # create the coordinate structs
        r, r_spectral = define_coordinate(r_input,init_YY=false)
        z, z_spectral = define_coordinate(z_input,init_YY=false)
        vperp, vperp_spectral = define_coordinate(vperp_input,init_YY=false)
        gyrophase, gyrophase_spectral = define_coordinate(gyrophase_input,init_YY=false)
        
        # create test geometry
        rhostar = 0.1 #rhostar of ions for ExB drift
        option = "constant-helical"
        pitch = 1.0
        DeltaB = 1.0
        geometry_in = geometry_input(rhostar,option,pitch,DeltaB)
        geometry = init_magnetic_geometry(geometry_in,z,r)
        
        # create test composition
        composition = create_test_composition()
            
        # Set up MPI
        initialize_comms!()
        setup_distributed_memory_MPI(1,1,1,1)
        looping.setup_loop_ranges!(block_rank[], block_size[];
                                       s=1, sn=1,
                                       r=r.n, z=z.n, vperp=vperp.n, vpa=1,
                                       vzeta=1, vr=1, vz=1)
                                       
        # initialise the matrix for the gyroaverages
        gyro = init_gyro_operators(vperp,z,r,gyrophase,geometry,composition)
        # initialise a test field
        kr = 2
        kz = 2
        phi = allocate_float(z.n,r.n)
        gphi = allocate_float(vperp.n,z.n,r.n)
        for ir in 1:r.n
            for iz in 1:z.n
                phi[iz,ir] = sin(2.0*pi*r.grid[ir]*kr/r.L)*sin(2.0*pi*z.grid[iz]*kz/z.L)
            end
        end
        
        for ir in 1:r.n
            for iz in 1:z.n
                for ivperp in 1:vperp.n
        #print_matrix(gyro.gyromatrix[:,:,vperp.n,Int(floor(z.n/2)),Int(floor(r.n/2))],"gmatrix",z.n,r.n)
                    #print_matrix(gyro.gyromatrix[:,:,ivperp,iz,ir],"gmatrix_ivperp_"*string(ivperp)*"_iz_"*string(iz)*"_ir_"*string(ir),z.n,r.n)
                end
            end
        end
        println(maximum(abs.(gyro.gyromatrix)))
        #end
        
        # gyroaverage phi
        gyroaverage_field!(gphi,phi,gyro,vperp,z,r)
        
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
    return composition = species_composition(n_species, n_ion_species, n_neutral_species,
            electron_physics, use_test_neutral_wall_pdf, T_e, T_wall, phi_wall, Er_constant,
            mn_over_mi, me_over_mi, recycling_fraction, allocate_float(n_species))
end

if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")

    gyroaverage_test()
end