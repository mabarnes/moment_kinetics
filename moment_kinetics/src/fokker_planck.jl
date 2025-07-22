"""
Module for including the Full-F Fokker-Planck Collision Operator.

The functions in this module are split into two groups. 

The first set of functions implement the weak-form
Collision operator using the Rosenbluth-MacDonald-Judd
formulation in a divergence form. The Green's functions
for the Rosenbluth potentials are used to obtain the Rosenbluth
potentials at the boundaries. To find the potentials
everywhere else elliptic solves of the PDEs for the
Rosenbluth potentials are performed with Dirichlet
boundary conditions. These routines provide the default collision operator
used in the code.

The second set of functions are used to set up the necessary arrays to 
compute the Rosenbluth potentials everywhere in vpa, vperp
by direct integration of the Green's functions. These functions are 
supported for the purposes of testing and debugging.

Lower-level routines are provided by functions from
[`moment_kinetics.fokker_planck_calculus`](@ref).

Parallelisation of the collision operator uses a special 'anysv' region type, see
[Collision operator and `anysv` region](@ref).
"""
module fokker_planck

# Import moment_kinetics so that we can refer to it in docstrings
import moment_kinetics

export init_fp_collisions
export explicit_fokker_planck_collisions_weak_form!
export explicit_fp_collisions_weak_form_Maxwellian_cross_species!
# implicit advance
export implicit_ion_fokker_planck_self_collisions!
       
using OrderedCollections: OrderedDict
using ..type_definitions: mk_float, mk_int, OptionsDict
using ..array_allocation: allocate_float
using ..communication
using ..velocity_moments: get_density, get_upar, get_p, get_ppar, get_pperp, get_qpar, get_rmom
using ..looping
using ..timer_utils
using ..input_structs: fkpl_collisions_input, set_defaults_and_check_section!, Dict_to_NamedTuple
using ..looping: get_best_ranges
using ..reference_parameters: get_reference_collision_frequency_ii,
                            setup_reference_parameters
using FokkerPlanck: fokker_planck_self_collisions_backward_euler_step!,
                    fokker_planck_self_collision_operator_weak_form!,
                    fokker_planck_cross_species_collision_operator_Maxwellian_Fsp!,
                    calculate_entropy_production,
                    init_fokker_planck_collisions,
                    natural_boundary_condition, zero_boundary_condition
using FokkerPlanck
using FiniteElementMatrices: element_coordinates

"""
Function for reading Fokker Planck collision operator input parameters. 
Structure the namelist as follows.

    [fokker_planck_collisions]
    use_fokker_planck = true
    nuii = 1.0
    frequency_option = "manual"
"""
function setup_fkpl_collisions_input(toml_input::AbstractDict, warn_unexpected::Bool)
    reference_params = setup_reference_parameters(toml_input, warn_unexpected)
    # get reference collision frequency (note factor of 1/2 due to definition choices)
    nuii_fkpl_default = 0.5*get_reference_collision_frequency_ii(reference_params)
    # read the input toml and specify a sensible default
    input_section = set_defaults_and_check_section!(
        toml_input, "fokker_planck_collisions", warn_unexpected;
        # begin default inputs (as kwargs)
        use_fokker_planck = false,
        nuii = -1.0,
        frequency_option = "reference_parameters",
        self_collisions = true,
        use_conserving_corrections = true,
        boundary_data_option = FokkerPlanck.direct_integration,
        use_test_particle_preconditioner = true,
        slowing_down_test = false,
        sd_density = 1.0,
        sd_temp = 0.01,
        sd_q = 1.0,
        sd_mi = 0.25,
        sd_me = 0.25/1836.0,
        Zi = 1.0)
    # ensure that the collision frequency is consistent with the input option
    frequency_option = input_section["frequency_option"]
    if frequency_option == "reference_parameters"
        input_section["nuii"] = nuii_fkpl_default
    elseif frequency_option == "manual" 
        # use the frequency from the input file
        # do nothing
    else
        error("Invalid option [fokker_planck_collisions] "
              * "frequency_option=$(frequency_option) passed")
    end
    # finally, ensure nuii < 0 if use_fokker_planck is false
    # so that nuii > 0 is the only check required in the rest of the code
    if !input_section["use_fokker_planck"]
        input_section["nuii"] = -1.0
    end
    input = OrderedDict(Symbol(k)=>v for (k,v) in input_section)
    #println(input)
    if input_section["slowing_down_test"]
        # calculate nu_alphae and vc3 (critical speed of slowing down)
        # as a diagnostic to aid timestep choice
        # nu_alphae/nuref
        Zalpha = input_section["Zi"]
        Zi = input_section["sd_q"]
        ni = input_section["sd_density"]
        ne = Zi*ni+Zalpha # assume unit density for alphas in initial state
        Te = input_section["sd_temp"]
        me = input_section["sd_me"]
        mi = input_section["sd_mi"]
        nu_alphae = (8.0/(3.0*sqrt(pi)))*ne*sqrt(me)*(Te^(-1.5))*(Zalpha^(2.0))
        vc3 = (3.0/4.0)*sqrt(pi)*(Zi^2)*(ni/ne)*(1.0/mi)*(1.0/sqrt(me))*(Te^(1.5))
        nu_alphaalpha = Zalpha^4
        nu_alphai = nu_alphae*vc3
        if global_rank[] == 0
            println("slowing_down_test = true")
            println("nu_alphaalpha/nuref = $nu_alphaalpha")
            println("nu_alphai/nuref = $nu_alphai")
            println("nu_alphae/nuref = $nu_alphae")
            println("vc3/cref^3 = $vc3")
            println("critical speed vc/cref = ",vc3^(1.0/3.0))
        end
    end
    return fkpl_collisions_input(; input...)
end

########################################################
# begin functions associated with the weak-form operator
# where the potentials are computed by elliptic solve
########################################################

"""
Function that initialises the arrays needed for Fokker Planck collisions.
This function uses the interface for constructing the FokkerPlanck internal
grids from an external tensor-product finite element grid.

N.B. Inputs are required here to set nonlinear solver options for 
Fokker-Planck collisions. An input namelist of form

```
    [fokker_planck_collisions_nonlinear_solver]
    atol = 1.0e-10
    rtol = 0.0
    nonlinear_max_iterations=20
```
is used.
"""
function init_fp_collisions(toml_input::AbstractDict,
                input_flags::fkpl_collisions_input,
                vpa,vperp; print_to_screen=false)
    function set_bc(coord)
        bc = coord.bc
        if bc == "none"
            fp_bc = natural_boundary_condition
        elseif bc == "zero"
            fp_bc = zero_boundary_condition
        else
            error_str=""" Invalid boundary condition "$(bc)" supplied for $(coord.name)
            Use bc="zero" or bc="none"
            """
            error(error_str)
        end
        return fp_bc
    end
    vpa_input = Array{element_coordinates,1}(undef,vpa.nelement_local)
    vperp_input = Array{element_coordinates,1}(undef,vperp.nelement_local)
    # set vpa input
    for ielement in 1:vpa.nelement_local
        imin = vpa.igrid_full[1,ielement]
        imax = vpa.igrid_full[end,ielement]
        scale = vpa.element_scale[ielement]
        shift = vpa.element_shift[ielement]
        refnodes = allocate_float(vpa.ngrid)
        @. refnodes = (vpa.grid[imin:imax] - shift)/scale
        vpa_input[ielement] = element_coordinates(refnodes,scale,shift)
    end
    # set vperp input
    for ielement in 1:vperp.nelement_local
        imin = vperp.igrid_full[1,ielement]
        imax = vperp.igrid_full[end,ielement]
        scale = vperp.element_scale[ielement]
        shift = vperp.element_shift[ielement]
        refnodes = allocate_float(vperp.ngrid)
        @. refnodes = (vperp.grid[imin:imax] - shift)/scale
        vperp_input[ielement] = element_coordinates(refnodes,scale,shift)
    end
    # get nonlinear solver inputs
    section_name = "fokker_planck_collisions_nonlinear_solver"
    warn_unexpected= false
    fp_nl_inputs = Dict_to_NamedTuple(set_defaults_and_check_section!(
        toml_input, section_name, warn_unexpected;
        rtol=0.0,
        atol=1.0e-10,
        nonlinear_max_iterations=20,
        ))
    #println(fp_nl_inputs)
    # initialise all arrays needed to evaluate the nonlinear Fokker-Planck operator
    #println(input_flags.boundary_data_option)
    fkpl_arrays = init_fokker_planck_collisions(
                        vpa_input,
                        vperp_input;
                        bc_vpa=set_bc(vpa),
                        bc_vperp=set_bc(vperp),
                        boundary_data_option=input_flags.boundary_data_option,
                        nl_solver_atol=fp_nl_inputs.atol,
                        nl_solver_rtol=fp_nl_inputs.rtol,
                        nl_solver_nonlinear_max_iterations=fp_nl_inputs.nonlinear_max_iterations,
                        print_to_screen=print_to_screen)
    return fkpl_arrays
end

"""
Function for advancing with the explicit, weak-form, self-collision operator
using the existing method for computing the Rosenbluth potentials, with
the addition of cross-species collisions against fixed Maxwellian distribution functions
where the Rosenbluth potentials are specified using analytical results.
"""
@timeit global_timer explicit_fp_collisions_weak_form_Maxwellian_cross_species!(
                         pdf_out, pdf_in, dSdt, composition, collisions, dt,
                         fkpl_arrays, r, z, vperp,
                         vpa, vperp_spectral, vpa_spectral;
                         diagnose_entropy_production=false) = begin
    # N.B. only self-collisions are currently supported
    # This can be modified by adding a loop over s' below
    n_ion_species = composition.n_ion_species
    @boundscheck vpa.n == size(pdf_out,1) || throw(BoundsError(pdf_out))
    @boundscheck vperp.n == size(pdf_out,2) || throw(BoundsError(pdf_out))
    @boundscheck z.n == size(pdf_out,3) || throw(BoundsError(pdf_out))
    @boundscheck r.n == size(pdf_out,4) || throw(BoundsError(pdf_out))
    @boundscheck n_ion_species == size(pdf_out,5) || throw(BoundsError(pdf_out))
    @boundscheck vpa.n == size(pdf_in,1) || throw(BoundsError(pdf_in))
    @boundscheck vperp.n == size(pdf_in,2) || throw(BoundsError(pdf_in))
    @boundscheck z.n == size(pdf_in,3) || throw(BoundsError(pdf_in))
    @boundscheck r.n == size(pdf_in,4) || throw(BoundsError(pdf_in))
    @boundscheck n_ion_species == size(pdf_in,5) || throw(BoundsError(pdf_in))
    @boundscheck z.n == size(dSdt,1) || throw(BoundsError(dSdt))
    @boundscheck r.n == size(dSdt,2) || throw(BoundsError(dSdt))
    @boundscheck n_ion_species == size(dSdt,3) || throw(BoundsError(dSdt))
    
    use_conserving_corrections = collisions.fkpl.use_conserving_corrections
    fkin = collisions.fkpl
    # masses charge numbers and collision frequencies
    mref = 1.0 # generalise if multiple ions evolved
    Zs = fkin.Zi # generalise if multiple ions evolved
    nuref = fkin.nuii # generalise if multiple ions evolved
    msp = [fkin.sd_mi, fkin.sd_me]
    Zsp = [fkin.sd_q, -1.0]
    # assume here that ne = sum_i n_i and that initial condition
    # for beam ions has unit density
    ns = 1.0 # initial density of evolved ions (hardcode to be the same as initial conditions)
    # get electron density from quasineutrality ne = sum_s Zs ns
    densp = [fkin.sd_density, fkin.sd_q*fkin.sd_density+ns*Zs]
 
    uparsp = [0.0, 0.0]
    vthsp = [sqrt(2.0*fkin.sd_temp/msp[1]), sqrt(2.0*fkin.sd_temp/msp[2])]
    
    # N.B. parallelisation using only over space here
    @begin_r_z_region()
    @loop_r_z ir iz begin
        # single evolved species
        is = 1
        # computes sum over s' of  C[Fs,Fs'] with Fs' an assumed Maxwellian 
        @views fokker_planck_cross_species_collision_operator_Maxwellian_Fsp!(pdf_in[:,:,iz,ir,is],
                                     nuref,mref,Zs,msp,Zsp,densp,uparsp,vthsp,
                                     fkpl_arrays,use_conserving_corrections=use_conserving_corrections)
        # advance this part of s,r,z with the resulting sum_s' C[Fs,Fs']
        CC = fkpl_arrays.CC
        @loop_vperp_vpa ivperp ivpa begin
            pdf_out[ivpa,ivperp,iz,ir,is] += dt*CC[ivpa,ivperp]
        end
        
    end
    return nothing
end


"""
Function for advancing with the explicit, weak-form, self-collision operator.
"""
@timeit global_timer explicit_fokker_planck_collisions_weak_form!(
                         pdf_out, pdf_in,
                         dSdt, density, vth, evolve_p, evolve_density,
                         composition, collisions, dt,
                         fkpl_arrays, r, z, vperp,
                         vpa;
                         diagnose_entropy_production=false) = begin
    # N.B. only self-collisions are currently supported
    # This can be modified by adding a loop over s' below
    n_ion_species = composition.n_ion_species
    @boundscheck vpa.n == size(pdf_out,1) || throw(BoundsError(pdf_out))
    @boundscheck vperp.n == size(pdf_out,2) || throw(BoundsError(pdf_out))
    @boundscheck z.n == size(pdf_out,3) || throw(BoundsError(pdf_out))
    @boundscheck r.n == size(pdf_out,4) || throw(BoundsError(pdf_out))
    @boundscheck n_ion_species == size(pdf_out,5) || throw(BoundsError(pdf_out))
    @boundscheck vpa.n == size(pdf_in,1) || throw(BoundsError(pdf_in))
    @boundscheck vperp.n == size(pdf_in,2) || throw(BoundsError(pdf_in))
    @boundscheck z.n == size(pdf_in,3) || throw(BoundsError(pdf_in))
    @boundscheck r.n == size(pdf_in,4) || throw(BoundsError(pdf_in))
    @boundscheck n_ion_species == size(pdf_in,5) || throw(BoundsError(pdf_in))
    @boundscheck z.n == size(dSdt,1) || throw(BoundsError(dSdt))
    @boundscheck r.n == size(dSdt,2) || throw(BoundsError(dSdt))
    @boundscheck n_ion_species == size(dSdt,3) || throw(BoundsError(dSdt))
    
    # masses and collision frequencies
    ms, msp = 1.0, 1.0 # generalise!
    nuref = collisions.fkpl.nuii # generalise!
    Zi = collisions.fkpl.Zi # generalise!
    nuss = nuref*(Zi^4) # include charge number factor for self collisions
    use_conserving_corrections = collisions.fkpl.use_conserving_corrections
    # N.B. parallelisation only over spatial quantities
    @begin_r_z_region()
    @loop_r_z ir iz begin
        # single species support only
        is = 1
        prefactor = moment_kinetic_collision_frequency_prefactor(density[iz,ir,is],
                                            vth[iz,ir,is], evolve_p, evolve_density)
        # first argument is Fs, and second argument is Fs' in C[Fs,Fs'] 
        @views fokker_planck_self_collision_operator_weak_form!(
            pdf_in[:,:,iz,ir,is], ms, nuss*prefactor, fkpl_arrays,
            use_conserving_corrections = use_conserving_corrections)
        # advance this part of s,r,z with the resulting C[Fs,Fs]
        CC = fkpl_arrays.CC
        @loop_vperp_vpa ivperp ivpa begin
            pdf_out[ivpa,ivperp,iz,ir,is] += dt*CC[ivpa,ivperp]
        end
        if diagnose_entropy_production
            @views dSdt[iz,ir,is] = calculate_entropy_production(pdf_in[:,:,iz,ir,is], fkpl_arrays)
        end
    end
    return nothing
end

"""
Function to account for the normalisation of the
moment kinetic normalised distribution function
by multiplying the input collision frequency by
the normalisation factors.

Only applicable for self collisions.
"""
function moment_kinetic_collision_frequency_prefactor(density::mk_float, vth::mk_float,
                                                    evolve_p::Bool, evolve_density::Bool)
    if evolve_p && evolve_density
        collision_prefactor = density/(vth^3)
    elseif evolve_density
        collision_prefactor = density
    else
        collision_prefactor = 1.0
    end
    return collision_prefactor
end

######################################################
# end functions associated with the weak-form operator
# where the potentials are computed by elliptic solve
######################################################

#################################################
# Functions associated with implicit timestepping
#################################################

function implicit_ion_fokker_planck_self_collisions!(pdf_out, pdf_in,
                    dSdt, density, vth, evolve_p, evolve_density,
                    composition, collisions, fkpl_arrays, 
                    vpa, vperp, z, r, delta_t; diagnose_entropy_production=false)
    # bounds checking
    n_ion_species = composition.n_ion_species
    @boundscheck vpa.n == size(pdf_out,1) || throw(BoundsError(pdf_out))
    @boundscheck vperp.n == size(pdf_out,2) || throw(BoundsError(pdf_out))
    @boundscheck z.n == size(pdf_out,3) || throw(BoundsError(pdf_out))
    @boundscheck r.n == size(pdf_out,4) || throw(BoundsError(pdf_out))
    @boundscheck n_ion_species == size(pdf_out,5) || throw(BoundsError(pdf_out))
    @boundscheck vpa.n == size(pdf_in,1) || throw(BoundsError(pdf_in))
    @boundscheck vperp.n == size(pdf_in,2) || throw(BoundsError(pdf_in))
    @boundscheck z.n == size(pdf_in,3) || throw(BoundsError(pdf_in))
    @boundscheck r.n == size(pdf_in,4) || throw(BoundsError(pdf_in))
    @boundscheck n_ion_species == size(pdf_in,5) || throw(BoundsError(pdf_in))
    @boundscheck z.n == size(dSdt,1) || throw(BoundsError(dSdt))
    @boundscheck r.n == size(dSdt,2) || throw(BoundsError(dSdt))
    @boundscheck n_ion_species == size(dSdt,3) || throw(BoundsError(dSdt))

    # diagnostic parameter for user
    success = true

    # masses and collision frequencies
    ms, msp = 1.0, 1.0 # generalise!
    nuref = collisions.fkpl.nuii # generalise!
    Zi = collisions.fkpl.Zi # generalise!
    nuss = nuref*(Zi^4) # include charge number factor for self collisions
    use_conserving_corrections = collisions.fkpl.use_conserving_corrections
    test_particle_preconditioner = collisions.fkpl.use_test_particle_preconditioner
    # N.B. parallelisation only over space here
    @begin_r_z_region()
    @loop_r_z ir iz begin
        # single species only
        is = 1
        prefactor = moment_kinetic_collision_frequency_prefactor(density[iz,ir,is],
                                            vth[iz,ir,is], evolve_p, evolve_density)
        @views Fold = pdf_in[:,:,iz,ir,is]
        local_success = fokker_planck_self_collisions_backward_euler_step!(Fold, delta_t, ms, nuss*prefactor, fkpl_arrays;
                            use_conserving_corrections=use_conserving_corrections,
                            test_particle_preconditioner=test_particle_preconditioner)
        # global success true only if local_success true
        success = success && local_success
        Fnew = fkpl_arrays.Fnew
        # store Fnew = F^n+1 in the appropriate distribution function array
        @loop_vperp_vpa ivperp ivpa begin
            pdf_out[ivpa,ivperp,iz,ir,is] = Fnew[ivpa,ivperp]
        end
        # compute dSdt
        if diagnose_entropy_production
            @views dSdt[iz,ir,is] = calculate_entropy_production(pdf_out[:,:,iz,ir,is], fkpl_arrays)
        end
    end
    return success
end

end
