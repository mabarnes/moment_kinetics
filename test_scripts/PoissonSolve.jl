export run_poisson_test
using Plots
using LaTeXStrings
using MPI
using Measures
#using Dates
import moment_kinetics
using moment_kinetics.array_allocation: allocate_float, allocate_shared_float
using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.type_definitions: mk_float, mk_int, OptionsDict
using moment_kinetics.spatial_poisson: init_spatial_poisson, spatial_poisson_solve!
using moment_kinetics.spatial_poisson: init_spatial_poisson2D, spatial_poisson2D_solve!
using moment_kinetics.communication
using moment_kinetics.looping
using moment_kinetics.calculus: derivative!

function plot_test_data(func_exact,func_num,func_err,func_name,radial,polar)
    @views heatmap(radial.grid, polar.grid, func_num[:,:], xlabel=L"r", ylabel=L"\theta", c = :deep, interpolation = :cubic, #, projection =:polar
                windowsize = (360,240), margin = 15pt)
                outfile = string(func_name*"_num.pdf")
                savefig(outfile)
    @views heatmap(radial.grid, polar.grid, func_exact[:,:], xlabel=L"r", ylabel=L"\theta", c = :deep, interpolation = :cubic, #, projection =:polar
                windowsize = (360,240), margin = 15pt)
                outfile = string(func_name*"_exact.pdf")
                savefig(outfile)
    @views heatmap(radial.grid, polar.grid, func_err[:,:], xlabel=L"r", ylabel=L"\theta", c = :deep, interpolation = :cubic, #, projection =:polar
                windowsize = (360,240), margin = 15pt)
                outfile = string(func_name*"_err.pdf")
                savefig(outfile)
    return nothing
end


function run_poisson_test(; nelement_radial=5,ngrid_radial=5,Lradial=1.0,nelement_polar=1,ngrid_polar=5,Lpolar=2.0*pi,kk::mk_int=1,plot_results=false)

   nelement_local_polar = nelement_polar # number of elements per rank
   nelement_global_polar = nelement_local_polar # total number of elements 
   nelement_local_radial = nelement_radial # number of elements per rank
   nelement_global_radial = nelement_local_radial # total number of elements 
   bc = "none" #not required to take a particular value, not used, set to "none" to avoid extra BC impositions 
   GL_discretization = "gausslegendre_pseudospectral"
   F_discretization = "fourier_pseudospectral"
   element_spacing_option = "uniform"
   coords_input = OptionsDict(
                    "radial"=>OptionsDict("ngrid"=>ngrid_radial, "nelement"=>nelement_global_radial,
                                         "nelement_local"=>nelement_local_radial, "L"=>Lradial,
                                         "discretization"=>GL_discretization, "bc"=>bc,
                                         "element_spacing_option"=>element_spacing_option),
                    "polar"=>OptionsDict("ngrid"=>ngrid_polar, "nelement"=>nelement_global_polar,
                                       "nelement_local"=>nelement_local_polar, "L"=>Lpolar,
                                       "discretization"=>F_discretization, "bc"=>bc,
                                       "element_spacing_option"=>element_spacing_option),
                     )
   # Set up MPI
   initialize_comms!()
   setup_distributed_memory_MPI(1,1,1,1)
   # ignore MPI here to avoid FFTW wisdom problems, test runs on a single core below
   polar, polar_spectral = define_coordinate(coords_input, "polar", ignore_MPI=true)
   radial, radial_spectral = define_coordinate(coords_input, "radial", ignore_MPI=true)
   looping.setup_loop_ranges!(block_rank[], block_size[];
                                 s=1, sn=1,
                                 r=radial.n, z=polar.n, vperp=1, vpa=1,
                                 vzeta=1, vr=1, vz=1)
   
   begin_serial_region()
   @serial_region begin
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
       phi = allocate_float(polar.n,radial.n)
       exact_phi = allocate_float(polar.n,radial.n)
       err_phi = allocate_float(polar.n,radial.n)
       rho = allocate_float(polar.n,radial.n)
       
       for irad in 1:radial.n
          for ipol in 1:polar.n
             exact_phi[ipol,irad] = 0.25*(radial.grid[irad]^2 -1)
             rho[ipol,irad] = 1.0
          end
       end
       spatial_poisson_solve!(phi,rho,poisson,radial,polar,polar_spectral)
       println("Maximum error value Test rho=1 : ",maximum(abs.(phi- exact_phi)))
       
       if kk < 1
          error("ERROR: kk >=1 required for test")
       end
       if !(mod(kk,1) == 0)
          error("ERROR: kk integer required for test")
       end
       if abs(polar.L - 2.0*pi) >1.0e-14
          error("ERROR: polar coordinate assumed angle for definition of the following test - set Lpolar = 2.0*pi")
       end
       
       for irad in 1:radial.n
          for ipol in 1:polar.n
             exact_phi[ipol,irad] = (1.0 - radial.grid[irad])*(radial.grid[irad]^kk)*cos(2.0*pi*kk*polar.grid[ipol]/polar.L)
             rho[ipol,irad] = (kk^2 - (kk+1)^2)*(radial.grid[irad]^(kk-1))*cos(2.0*kk*pi*polar.grid[ipol]/polar.L)
          end
       end
       
       spatial_poisson_solve!(phi,rho,poisson,radial,polar,polar_spectral)
       @. err_phi = abs(phi - exact_phi)
       println("Maximum error value Test rho = (kk^2 - (kk+1)^2) * cos(2 pi kk P/L) * r^(kk-1): ",maximum(err_phi))
       if plot_results
          plot_test_data(exact_phi,phi,err_phi,"phi",radial,polar)
       end
   end
   finalize_comms!()
   return nothing
end

function run_poisson2D_test(; nelement_radial=5,ngrid_radial=5,Lradial=1.0,nelement_polar=5,ngrid_polar=5,Lpolar=2.0*pi,kk::mk_int=1,phase::mk_float=pi/3.0,plot_results=false)

   nelement_local_polar = nelement_polar # number of elements per rank
   nelement_global_polar = nelement_local_polar # total number of elements 
   nelement_local_radial = nelement_radial # number of elements per rank
   nelement_global_radial = nelement_local_radial # total number of elements 
   bc = "none" #not required to take a particular value, not used, set to "none" to avoid extra BC impositions 
   discretization = "gausslegendre_pseudospectral"
   element_spacing_option = "uniform"
   coords_input = OptionsDict(
                    "radial"=>OptionsDict("ngrid"=>ngrid_radial, "nelement"=>nelement_global_radial,
                                         "nelement_local"=>nelement_local_radial, "L"=>Lradial,
                                         "discretization"=>discretization, "bc"=>bc,
                                         "element_spacing_option"=>element_spacing_option),
                    "polar"=>OptionsDict("ngrid"=>ngrid_polar, "nelement"=>nelement_global_polar,
                                       "nelement_local"=>nelement_local_polar, "L"=>Lpolar,
                                       "discretization"=>discretization, "bc"=>bc,
                                       "element_spacing_option"=>element_spacing_option),
                     )
   # Set up MPI
   initialize_comms!()
   setup_distributed_memory_MPI(1,1,1,1)
   # ignore MPI here to avoid FFTW wisdom problems, test runs on a single core below
   polar, polar_spectral = define_coordinate(coords_input, "polar", ignore_MPI=true)
   radial, radial_spectral = define_coordinate(coords_input, "radial", ignore_MPI=true)
   looping.setup_loop_ranges!(block_rank[], block_size[];
                                 s=1, sn=1,
                                 r=radial.n, z=polar.n, vperp=1, vpa=1,
                                 vzeta=1, vr=1, vz=1)
   
   begin_serial_region()
   @serial_region begin
       println("Test Poisson")
       poisson = init_spatial_poisson2D(radial,polar,radial_spectral,polar_spectral)
       phi = allocate_float(polar.n,radial.n)
       exact_phi = allocate_float(polar.n,radial.n)
       err_phi = allocate_float(polar.n,radial.n)
       rho = allocate_float(polar.n,radial.n)
       
       for irad in 1:radial.n
          for ipol in 1:polar.n
             exact_phi[ipol,irad] = 0.25*(radial.grid[irad]^2 -1)
             rho[ipol,irad] = 1.0
          end
       end
       spatial_poisson2D_solve!(phi,rho,poisson)
       println("Maximum error value Test rho=1 : ",maximum(abs.(phi- exact_phi)))
       if plot_results
          plot_test_data(exact_phi,phi,err_phi,"phi_1",radial,polar)
          println("phi(theta=-pi,r=0): ",phi[1,1])
          println("phi(theta=pi,r=0): ",phi[end,1])
       end
       
       if kk < 1
          error("ERROR: kk >=1 required for test")
       end
       if !(mod(kk,1) == 0)
          error("ERROR: kk integer required for test")
       end
       if abs(polar.L - 2.0*pi) >1.0e-14
          error("ERROR: polar coordinate assumed angle for definition of the following test - set Lpolar = 2.0*pi")
       end
       
       for irad in 1:radial.n
          for ipol in 1:polar.n
             exact_phi[ipol,irad] = (1.0 - radial.grid[irad])*(radial.grid[irad]^kk)*cos(2.0*pi*kk*polar.grid[ipol]/polar.L + phase)
             rho[ipol,irad] = (kk^2 - (kk+1)^2)*(radial.grid[irad]^(kk-1))*cos(2.0*kk*pi*polar.grid[ipol]/polar.L  + phase)
          end
       end
       
       spatial_poisson2D_solve!(phi,rho,poisson)
       @. err_phi = abs(phi - exact_phi)
       println("Maximum error value Test rho = (kk^2 - (kk+1)^2) * cos(2 pi kk P/L + phase) * r^(kk-1): ",maximum(err_phi))
       if plot_results
          plot_test_data(exact_phi,phi,err_phi,"phi_2",radial,polar)
          println("phi(theta=-pi,r=0): ",phi[1,1])
          println("phi(theta=pi,r=0): ",phi[end,1])
       end
   end
   finalize_comms!()
   return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")
    
    run_poisson_test()
    run_poisson2D_test()
end