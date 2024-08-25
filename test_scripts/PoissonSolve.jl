export run_poisson_test
#using Printf
using Plots
#gr()
using LaTeXStrings
using MPI
using Measures
#using Dates
import moment_kinetics
using moment_kinetics.array_allocation: allocate_float, allocate_shared_float
using moment_kinetics.input_structs: grid_input, advection_input
using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.type_definitions: mk_float, mk_int
using moment_kinetics.spatial_poisson: init_spatial_poisson, spatial_poisson_solve!
using moment_kinetics.communication
using moment_kinetics.looping
using moment_kinetics.calculus: derivative!

function plot_test_data(func_exact,func_num,func_err,func_name,radial,polar)
    @views heatmap(polar.grid, radial.grid, func_num[:,:], ylabel=L"r", xlabel=L"\theta", c = :deep, interpolation = :cubic,
                windowsize = (360,240), margin = 15pt, projection =:polar)
                outfile = string(func_name*"_num.pdf")
                savefig(outfile)
    @views heatmap(polar.grid, radial.grid, func_exact[:,:], ylabel=L"r", xlabel=L"\theta", c = :deep, interpolation = :cubic,
                windowsize = (360,240), margin = 15pt, projection =:polar)
                outfile = string(func_name*"_exact.pdf")
                savefig(outfile)
    @views heatmap(polar.grid, radial.grid, func_err[:,:], ylabel=L"r", xlabel=L"\theta", c = :deep, interpolation = :cubic,
                windowsize = (360,240), margin = 15pt, projection =:polar)
                outfile = string(func_name*"_err.pdf")
                savefig(outfile)
    return nothing
end


function run_poisson_test(; nelement_radial=5,ngrid_radial=5,Lradial=1.0,nelement_polar=1,ngrid_polar=1,Lpolar=2.0*pi,kk=1.0)

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
      nrank, irank, Lpolar, "fourier_pseudospectral", fd_option, cheb_option, "none", adv_input,comm,element_spacing_option)
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
  
   println("Test fourier_pseudospectral")
   ff = allocate_float(polar.n)
   df = allocate_float(polar.n)
   df_exact = allocate_float(polar.n)
   df_err = allocate_float(polar.n)
   for i in 1:polar.n
      ff[i] = sin(2.0*pi*polar.grid[i]/polar.L)
      df_exact[i] = cos(2.0*pi*polar.grid[i]/polar.L)*(2.0*pi/polar.L)
   end
   #println("ff: ",ff)
   derivative!(df,ff,polar,polar_spectral)
   #println("polar.grid: ",polar.grid)
   @. df_err = abs(df - df_exact)
   println("maximum(df_err): ",maximum(df_err))
   #println("df_exact: ",df_exact)
   #println(polar.wgts)
   println("wgts err: ",abs( 1.0 - (sum(polar.wgts)/polar.L) ))
   
   println("Test Poisson")
   poisson = init_spatial_poisson(radial,polar,radial_spectral)
   phi = allocate_float(radial.n,polar.n)
   exact_phi = allocate_float(radial.n,polar.n)
   err_phi = allocate_float(radial.n,polar.n)
   rho = allocate_float(radial.n,polar.n)
   
   for ipol in 1:polar.n
      for irad in 1:radial.n
         exact_phi[irad,ipol] = 0.25*(radial.grid[irad]^2 -1)
         rho[irad,ipol] = 1.0
      end
   end
   #@. rho = sinpi(2*radial.grid/radial.L)
   spatial_poisson_solve!(phi,rho,poisson,radial,polar,polar_spectral)
   #println(phi)
   #println(exact_phi)
   println("Maximum error value Test rho=1 : ",maximum(abs.(phi- exact_phi)))
   
   for ipol in 1:polar.n
      for irad in 1:radial.n
         exact_phi[irad,ipol] = 0.25*(radial.grid[irad]^2 -1)*cos(2.0*pi*kk*polar.grid[ipol]/polar.L)
         rho[irad,ipol] = cos(2.0*kk*pi*polar.grid[ipol]/polar.L)
      end
   end
   
   #@. rho = sinpi(2*radial.grid/radial.L)
   spatial_poisson_solve!(phi,rho,poisson,radial,polar,polar_spectral)
   @. err_phi = abs(phi - exact_phi)
   #println(phi)
   #println(exact_phi)
   println("Maximum error value Test rho = cos(2 pi kk P/L): ",maximum(err_phi))
   plot_test_data(exact_phi,phi,err_phi,"phi",radial,polar)
   finalize_comms!()
   return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")
    
    run_poisson_test()
end