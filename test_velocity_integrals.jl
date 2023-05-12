using Printf
using Plots
using LaTeXStrings
using MPI
using Measures

if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")

    import moment_kinetics
	using moment_kinetics.input_structs: grid_input, advection_input
	using moment_kinetics.coordinates: define_coordinate
    using moment_kinetics.velocity_moments: get_density, get_upar, get_ppar, get_pperp
    using moment_kinetics.array_allocation: allocate_float
    
    # define inputs needed for the test
	ngrid = 17 #number of points per element 
	nelement_local = 20 # number of elements per rank
	nelement_global = nelement_local # total number of elements 
	Lvpa = 12.0 #physical box size in reference units 
	Lvperp = 6.0 #physical box size in reference units 
	bc = "" #not required to take a particular value, not used 
	# fd_option and adv_input not actually used so given values unimportant
	discretization = "chebyshev_pseudospectral"
    fd_option = "fourth_order_centered"
	adv_input = advection_input("default", 1.0, 0.0, 0.0)
	nrank = 1
    irank = 0
    comm = MPI.COMM_NULL
	# create the 'input' struct containing input info needed to create a
	# coordinate
    vpa_input = grid_input("vpa", ngrid, nelement_global, nelement_local, 
		nrank, irank, Lvpa, discretization, fd_option, bc, adv_input,comm)
	vperp_input = grid_input("vperp", ngrid, nelement_global, nelement_local, 
		nrank, irank, Lvperp, discretization, fd_option, bc, adv_input,comm)
	# create the coordinate struct 'x'
	println("made inputs")
	println("vpa: ngrid: ",ngrid," nelement: ",nelement_local)
	println("vperp: ngrid: ",ngrid," nelement: ",nelement_local)
	vpa, vpa_spectral = define_coordinate(vpa_input)
	vperp, vperp_spectral = define_coordinate(vperp_input)
    
    dfn = allocate_float(vpa.n,vperp.n)
    
    function pressure(ppar,pperp)
        pres = (1.0/3.0)*(ppar + 2.0*pperp) 
        return pres
    end
    # assign a known isotropic Maxwellian distribution in normalised units
    dens = 3.0/4.0
    upar = 2.0/3.0
    ppar = 2.0/3.0
    pperp = 2.0/3.0
    pres = pressure(ppar,pperp) 
    mass = 1.0
    vth = sqrt(2.0*pres/(dens*mass))
    for ivperp in 1:vperp.n
        for ivpa in 1:vpa.n
            vpa_val = vpa.grid[ivpa]
            vperp_val = vperp.grid[ivperp]
            dfn[ivpa,ivperp] = (dens/vth^3)*exp( - ((vpa_val-upar)^2 + vperp_val^2)/vth^2 )
        end
    end
    
    # now check that we can extract the correct moments from the distribution
    
    dens_test = get_density(dfn,vpa,vperp)
    upar_test = get_upar(dfn,vpa,vperp,dens_test)
    ppar_test = get_ppar(dfn,vpa,vperp,upar_test)
    pperp_test = get_pperp(dfn,vpa,vperp)
    pres_test = pressure(ppar_test,pperp_test)
    # output test results 
    println("")
    println("Isotropic Maxwellian")
    println("dens_test: ", dens_test, " dens: ", dens, " error: ", abs(dens_test-dens))
    println("upar_test: ", upar_test, " upar: ", upar, " error: ", abs(upar_test-upar))
    println("pres_test: ", pres_test, " pres: ", pres, " error: ", abs(pres_test-pres))
    println("")
    
    # assign a known biMaxwellian distribution in normalised units
    dens = 3.0/4.0
    upar = 2.0/3.0
    ppar = 4.0/5.0
    pperp = 1.0/4.0 
    mass = 1.0
    vthpar = sqrt(2.0*ppar/(dens*mass))
    vthperp = sqrt(2.0*pperp/(dens*mass))
    for ivperp in 1:vperp.n
        for ivpa in 1:vpa.n
            vpa_val = vpa.grid[ivpa]
            vperp_val = vperp.grid[ivperp]
            dfn[ivpa,ivperp] = (dens/(vthpar*vthperp^2))*exp( - ((vpa_val-upar)^2)/vthpar^2 - (vperp_val^2)/vthperp^2 )
        end
    end
    
    # now check that we can extract the correct moments from the distribution
    
    dens_test = get_density(dfn,vpa,vperp)
    upar_test = get_upar(dfn,vpa,vperp,dens_test)
    ppar_test = get_ppar(dfn,vpa,vperp,upar_test)
    pperp_test = get_pperp(dfn,vpa,vperp)
    # output test results 
    
    println("")
    println("biMaxwellian")
    println("dens_test: ", dens_test, " dens: ", dens, " error: ", abs(dens_test-dens))
    println("upar_test: ", upar_test, " upar: ", upar, " error: ", abs(upar_test-upar))
    println("ppar_test: ", ppar_test, " ppar: ", ppar, " error: ", abs(ppar_test-ppar))
    println("pperp_test: ", pperp_test, " pperp: ", pperp, " error: ", abs(pperp_test-pperp))
    println("")
end 
