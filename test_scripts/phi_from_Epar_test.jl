export test_phi_from_Epar

using Printf
using Plots
using LaTeXStrings
using MPI
using Measures
using Dates

import moment_kinetics
using moment_kinetics.array_allocation: allocate_float, allocate_shared_float
using moment_kinetics.coordinates: define_coordinate, get_coordinate_input
using moment_kinetics.communication: setup_distributed_memory_MPI, block_rank, block_size, finalize_comms!
using moment_kinetics.type_definitions: OptionsDict
using moment_kinetics.looping
using moment_kinetics.input_structs: set_defaults_and_check_section!, em_fields_input
using moment_kinetics.em_fields: setup_em_fields, calculate_phi_from_Epar!

function test_phi_from_Epar(;ngrid_z=5, nelement_local_z=1, nelement_global_z=1, element_spacing_option_z="uniform", Lz = 1.0, 
                             ngrid_r=1, nelement_local_r=1, nelement_global_r=1, element_spacing_option_r="uniform", Lr = 1.0,
                             discretization="chebyshev_pseudospectral", ignore_MPI=false)
    input_dict = OptionsDict(
            "r"=>OptionsDict("ngrid"=>ngrid_r, "nelement"=>nelement_global_r,
                                 "nelement_local"=>nelement_local_r, "L"=>Lr,
                                 "discretization"=>discretization,
                                 "element_spacing_option"=>element_spacing_option_r),
            "z"=>OptionsDict("ngrid"=>ngrid_z, "nelement"=>nelement_global_z,
                               "nelement_local"=>nelement_local_z, "L"=>Lz,
                               "discretization"=>discretization,
                               "element_spacing_option"=>element_spacing_option_z),
        )
    
    # set up distributed-memory MPI information for z and r coords
    # need grid and MPI information to determine these values 
    # MRH just put dummy values now 
    r_coord_input = get_coordinate_input(input_dict, "r"; ignore_MPI=ignore_MPI)
    z_coord_input = get_coordinate_input(input_dict, "z"; ignore_MPI=ignore_MPI)
    if ignore_MPI
        irank_z = irank_r = 0
        nrank_z = nrank_r = 1
        comm_sub_z = comm_sub_r = MPI.COMM_NULL
    else
        irank_z, nrank_z, comm_sub_z, irank_r, nrank_r, comm_sub_r =
            setup_distributed_memory_MPI(z_coord_input.nelement,
                                         z_coord_input.nelement_local,
                                         r_coord_input.nelement,
                                         r_coord_input.nelement_local)
    end
    z, z_spectral = define_coordinate(z_coord_input; 
                                      run_directory="", ignore_MPI=ignore_MPI,
                                      irank=irank_z, nrank=nrank_z, comm=comm_sub_z)
    r, r_spectral = define_coordinate(r_coord_input;
                                      run_directory="",ignore_MPI=ignore_MPI,
                                      irank=irank_r, nrank=nrank_r, comm=comm_sub_r)
    
    em_input = set_defaults_and_check_section!(
                                        input_dict, em_fields_input, "em_fields"
                                       )
    looping.setup_loop_ranges!(block_rank[], block_size[];
                                      s=1,
                                      sn=1,
                                      r=r.n, z=z.n, vperp=1, vpa=1,
                                      vzeta=1, vr=1, vz=1)

    # create the "fields" structure that contains arrays
    # for the electrostatic potential phi and the electromagnetic fields
    # set vperp.n = n_ion_species = 1
    fields = setup_em_fields(1, z.n, r.n, 1,
                             em_input)

    # setup test
    phi = fields.phi
    phi_exact = deepcopy(phi)
    phi_error = deepcopy(phi)
    Epar = fields.Ez
    epsilon = 0.1
    offset = -3.0
    begin_serial_region()

    @serial_region begin
        @loop_r_z ir iz begin
            zplus = 0.5 + z.grid[iz]/z.L + epsilon 
            zminus = 0.5 - z.grid[iz]/z.L + epsilon 
            phi_exact[iz,ir] = sqrt(zplus*zminus) + offset
            Epar[iz,ir] = -(0.5/z.L)*( sqrt(zminus/zplus) - sqrt(zplus/zminus) ) 
        end
    end

    # now calculate phi
    # calculate_phi_from_Epar!() assumes that phi[1,:] and phi[end,:] have
    # been set with the correct values by a kinetic boundary condition
    if z.irank == 0
        @serial_region begin
            @loop_r ir begin
                phi[1,ir] = phi_exact[1,ir]
        
            end
        end
    end
    if z.irank == z.nrank - 1
        @serial_region begin
            @loop_r ir begin
                phi[end,ir] = phi_exact[end,ir]
            end
        end
    end
    calculate_phi_from_Epar!(phi, Epar, r, z, z_spectral)

    # evaluate the error
    @serial_region begin
        @loop_r_z ir iz begin
            phi_error[iz,ir] = phi[iz,ir] - phi_exact[iz,ir]
        end
        #println("phi:", phi)
        #println("phi_exact:", phi_exact)
        #println("phi_error:", phi_error)
        println("z.irank: ",z.irank," max(abs(phi_error)): ",maximum(abs.(phi_error) ))

    end

    finalize_comms!()
end
# run test by, e.g.,
# mpirun -n 8 --output-filename testpath julia --project -O3 test_scripts/phi_from_Epar_test.jl
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")
    # for precompilation
    # test_phi_from_Epar()
    test_phi_from_Epar(ngrid_z=5, nelement_global_z=16, nelement_local_z=2)
    test_phi_from_Epar(ngrid_z=5, nelement_global_z=16, nelement_local_z=4,
                       ngrid_r=5, nelement_global_r=2, nelement_local_r=1)
    
end