export run_poisson_test
#using Printf
#using Plots
#using LaTeXStrings
using MPI
#using Measures
#using Dates
import moment_kinetics
using moment_kinetics.array_allocation: allocate_float, allocate_shared_float
using moment_kinetics.input_structs: grid_input, advection_input
using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.type_definitions: mk_float, mk_int
using moment_kinetics.spatial_poisson: init_spatial_poisson, spatial_poisson_solve!
using moment_kinetics.communication
using moment_kinetics.looping

function run_poisson_test(; nelement_radial=5,ngrid_radial=5,Lradial=1.0,nelement_polar=1,ngrid_polar=1,Lpolar=2.0*pi)

   nelement_local_polar = nelement_polar # number of elements per rank
   nelement_global_polar = nelement_local_polar # total number of elements 
   nelement_local_radial = nelement_radial # number of elements per rank
   nelement_global_radial = nelement_local_radial # total number of elements 
   #Lvpa = 12.0 #physical box size in reference units 
   #Lvperp = 6.0 #physical box size in reference units 
   bc = "" #not required to take a particular value, not used 
   # fd_option and adv_input not actually used so given values unimportant
   #discretization = "chebyshev_pseudospectral"
   discretization = "gausslegendre_pseudospectral"
   fd_option = "fourth_order_centered"
   cheb_option = "matrix"
   adv_input = advection_input("default", 1.0, 0.0, 0.0)
   nrank = 1
   irank = 0
   comm = MPI.COMM_NULL
   # create the 'input' struct containing input info needed to create a
   # coordinate
   element_spacing_option = "uniform"
   polar_input = grid_input("polar", ngrid_polar, nelement_global_polar, nelement_local_polar, 
      nrank, irank, Lpolar, discretization, fd_option, cheb_option, bc, adv_input,comm,element_spacing_option)
   radial_input = grid_input("r", ngrid_radial, nelement_global_radial, nelement_local_radial, 
      nrank, irank, Lradial, discretization, fd_option, cheb_option, bc, adv_input,comm,element_spacing_option)
   # create the coordinate struct 'x'
   println("made inputs")
  # println("vpa: ngrid: ",ngrid," nelement: ",nelement_local_vpa, " Lvpa: ",Lvpa)
  # println("vperp: ngrid: ",ngrid," nelement: ",nelement_local_vperp, " Lvperp: ",Lvperp)
  
   # Set up MPI
   initialize_comms!()
   setup_distributed_memory_MPI(1,1,1,1)
   polar, polar_spectral = define_coordinate(polar_input)
   radial, radial_spectral = define_coordinate(radial_input)
   looping.setup_loop_ranges!(block_rank[], block_size[];
                                 s=1, sn=1,
                                 r=radial.n, z=polar.n, vperp=1, vpa=1,
                                 vzeta=1, vr=1, vz=1)
   
   begin_serial_region()
  
   poisson = init_spatial_poisson(radial,polar,radial_spectral)
   phi = allocate_float(radial.n,polar.n)
   rho = allocate_float(radial.n,polar.n)
   @. rho = 1.0
   spatial_poisson_solve!(phi,rho,poisson,radial,polar)
   println(phi)
   finalize_comms!()
   return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")
    
    run_poisson_test()
end