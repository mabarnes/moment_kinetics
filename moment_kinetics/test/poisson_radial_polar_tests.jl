module poisson_radial_polar_tests
# currently only runs on a single core

include("setup.jl")

export run_poisson_radial_polar_test
export run_test
using MPI
import moment_kinetics
using moment_kinetics.array_allocation: allocate_float, allocate_shared_float
using moment_kinetics.input_structs: grid_input, advection_input
using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.type_definitions: mk_float, mk_int
using moment_kinetics.spatial_poisson: init_spatial_poisson, spatial_poisson_solve!
using moment_kinetics.communication
using moment_kinetics.looping
using moment_kinetics.calculus: derivative!

function run_test()
    @testset "poisson_radial_polar_tests" begin
        println("poisson_radial_polar_tests")
        nelement_radial=5
        ngrid_radial=5
        Lradial=1.0
        ngrid_polar=1
        kk=1
        @testset "nelement_radial $nelement_radial ngrid_radial $ngrid_radial Lradial $Lradial ngrid_polar $ngrid_polar kk $kk" begin
             run_poisson_radial_polar_test(nelement_radial=nelement_radial,ngrid_radial=ngrid_radial,Lradial=Lradial,ngrid_polar=ngrid_polar,kk=kk,
                              atol_phi1=4.0e-14)
        end
        nelement_radial=5
        ngrid_radial=5
        Lradial=1.0
        ngrid_polar=8
        kk=3
        @testset "nelement_radial $nelement_radial ngrid_radial $ngrid_radial Lradial $Lradial ngrid_polar $ngrid_polar kk $kk" begin
             run_poisson_radial_polar_test(nelement_radial=nelement_radial,ngrid_radial=ngrid_radial,Lradial=Lradial,ngrid_polar=ngrid_polar,kk=kk,
                              atol_fourier=1.0e-14,atol_phi1=4.0e-14,atol_phi2=1.0e-14)
        end
        nelement_radial=5
        ngrid_radial=9
        Lradial=1.0
        ngrid_polar=8
        kk=3
        @testset "nelement_radial $nelement_radial ngrid_radial $ngrid_radial Lradial $Lradial ngrid_polar $ngrid_polar kk $kk" begin
             run_poisson_radial_polar_test(nelement_radial=nelement_radial,ngrid_radial=ngrid_radial,Lradial=Lradial,ngrid_polar=ngrid_polar,kk=kk,
                              atol_fourier=1.0e-14,atol_phi1=3.0e-12,atol_phi2=1.0e-14)
        end
        nelement_radial=10
        ngrid_radial=5
        Lradial=1.0
        ngrid_polar=8
        kk=3
        @testset "nelement_radial $nelement_radial ngrid_radial $ngrid_radial Lradial $Lradial ngrid_polar $ngrid_polar kk $kk" begin
             run_poisson_radial_polar_test(nelement_radial=nelement_radial,ngrid_radial=ngrid_radial,Lradial=Lradial,ngrid_polar=ngrid_polar,kk=kk,
                              atol_fourier=1.0e-14,atol_phi1=9.0e-13,atol_phi2=1.0e-14)
        end
        nelement_radial=40
        ngrid_radial=5
        Lradial=1.0
        ngrid_polar=16
        kk=5
        @testset "nelement_radial $nelement_radial ngrid_radial $ngrid_radial Lradial $Lradial ngrid_polar $ngrid_polar kk $kk" begin
             run_poisson_radial_polar_test(nelement_radial=nelement_radial,ngrid_radial=ngrid_radial,Lradial=Lradial,ngrid_polar=ngrid_polar,kk=kk,
                              atol_fourier=1.0e-14,atol_phi1=9.0e-13,atol_phi2=2.0e-11)
        end
    end
end

function run_poisson_radial_polar_test(; nelement_radial=5,ngrid_radial=5,Lradial=1.0,nelement_polar=1,ngrid_polar=1,Lpolar=2.0*pi,kk::mk_int=1,
                           atol_fourier=1.0e-14,atol_phi1=1.0e-14,atol_phi2=1.0e-14,print_to_screen=false)

   nelement_local_polar = nelement_polar # number of elements per rank
   nelement_global_polar = nelement_local_polar # total number of elements 
   nelement_local_radial = nelement_radial # number of elements per rank
   nelement_global_radial = nelement_local_radial # total number of elements 
   bc = "none" #not required to take a particular value, not used, set to "none" to avoid extra BC impositions 
   # fd_option and adv_input not actually used so given values unimportant
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
      nrank, irank, Lpolar, "fourier_pseudospectral", fd_option, cheb_option, bc, adv_input,comm,element_spacing_option)
   radial_input = grid_input("r", ngrid_radial, nelement_global_radial, nelement_local_radial, 
      nrank, irank, Lradial, "gausslegendre_pseudospectral", fd_option, cheb_option, bc, adv_input,comm,element_spacing_option)
   
   # Set up MPI
   initialize_comms!()
   setup_distributed_memory_MPI(1,1,1,1)
   # ignore MPI here to avoid FFTW wisdom problems, test runs on a single core below
   polar, polar_spectral = define_coordinate(polar_input, ignore_MPI=true)
   radial, radial_spectral = define_coordinate(radial_input)
   looping.setup_loop_ranges!(block_rank[], block_size[];
                                 s=1, sn=1,
                                 r=radial.n, z=polar.n, vperp=1, vpa=1,
                                 vzeta=1, vr=1, vz=1)
   
   begin_serial_region()
   @serial_region begin # run tests purely in serial for now
       if ngrid_polar > 1
           if print_to_screen println("Test fourier_pseudospectral") end
           ff = allocate_float(polar.n)
           df = allocate_float(polar.n)
           df_exact = allocate_float(polar.n)
           df_err = allocate_float(polar.n)
           for i in 1:polar.n
              ff[i] = sin(2.0*pi*polar.grid[i]/polar.L)
              df_exact[i] = cos(2.0*pi*polar.grid[i]/polar.L)*(2.0*pi/polar.L)
           end
           derivative!(df,ff,polar,polar_spectral)
           @. df_err = abs(df - df_exact)
           max_df_err = maximum(df_err)
           if print_to_screen println("maximum(df_err): ",max_df_err) end
           @test max_df_err < atol_fourier
           wgts_err = abs( 1.0 - (sum(polar.wgts)/polar.L) )
           if print_to_screen println("wgts err: ",wgts_err) end
           @test wgts_err < atol_fourier
       end
       
       if print_to_screen println("Test Poisson") end
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
       spatial_poisson_solve!(phi,rho,poisson,radial,polar,polar_spectral)
       max_phi_err = maximum(abs.(phi- exact_phi))
       if print_to_screen println("Maximum error value Test rho=1 : ",max_phi_err) end
       @test max_phi_err < atol_phi1
       
       if ngrid_polar > 1
           if kk < 1
              error("ERROR: kk >=1 required for test")
           end
           if !(mod(kk,1) == 0)
              error("ERROR: kk integer required for test")
           end
           if abs(polar.L - 2.0*pi) >1.0e-14
              error("ERROR: polar coordinate assumed angle for definition of the following test - set Lpolar = 2.0*pi")
           end
           
           for ipol in 1:polar.n
              for irad in 1:radial.n
                 exact_phi[irad,ipol] = (1.0 - radial.grid[irad])*(radial.grid[irad]^kk)*cos(2.0*pi*kk*polar.grid[ipol]/polar.L)
                 rho[irad,ipol] = (kk^2 - (kk+1)^2)*(radial.grid[irad]^(kk-1))*cos(2.0*kk*pi*polar.grid[ipol]/polar.L)
              end
           end
           
           spatial_poisson_solve!(phi,rho,poisson,radial,polar,polar_spectral)
           @. err_phi = abs(phi - exact_phi)
           max_phi_err = maximum(err_phi)
           if print_to_screen println("Maximum error value Test rho = (kk^2 - (kk+1)^2) * cos(2 pi kk P/L) * r^(kk-1): ",max_phi_err) end
           @test max_phi_err < atol_phi2
       end
   end
   finalize_comms!()
   return nothing
end

end #poisson_radial_polar_tests

using .poisson_radial_polar_tests

poisson_radial_polar_tests.run_test()
