module vpa_advection

export vpa_advection!
export update_speed_vpa!

using moment_kinetics_input: advection_speed, advection_speed_option_vpa
using semi_lagrange: find_approximate_characteristic!
using time_advance: advance_f_local!, rk_update_f!
using source_terms: update_boundary_indices!
using source_terms: set_igrid_ielem
using em_fields: update_phi!
using chebyshev: chebyshev_derivative!
using chebyshev: chebyshev_info
using finite_differences: derivative_finite_difference!
using initial_conditions: enforce_vpa_boundary_condition!

# argument chebyshev indicates that a chebyshev pseudopectral method is being used
function vpa_advection!(ff, ff_scratch, phi, moments, SL, source, vpa, z,
	n_rk_stages, use_semi_lagrange, dt, vpa_spectral, z_spectral, composition)
    # check to ensure that all array indices accessed in this function
    # are in-bounds
    @boundscheck size(ff,1) == z.n || throw(BoundsError(ff))
    @boundscheck size(ff,2) == vpa.n || throw(BoundsError(ff))
	@boundscheck size(ff,3) == composition.n_ion_species || throw(BoundsError(ff))
	@boundscheck size(ff_scratch,1) == z.n || throw(BoundsError(ff_scratch))
    @boundscheck size(ff_scratch,2) == vpa.n || throw(BoundsError(ff_scratch))
	@boundscheck size(ff_scratch,3) == composition.n_ion_species || throw(BoundsError(ff_scratch))
	@boundscheck size(ff_scratch,4) == 3 || throw(BoundsError(ff_scratch))
    # SSP RK for explicit time advance
	ff_scratch[:,:,:,1] .= ff
    for istage ∈ 1:n_rk_stages
		# calculate the advection speed corresponding to current f
	    update_speed_vpa!(source, phi, moments, view(ff_scratch,:,:,:,istage), vpa, z,
			composition, z_spectral)
		for is ∈ 1:composition.n_ion_species
			# update the upwind/downwind boundary indices and upwind_increment
			# NB: not sure if this will work properly with SL method at the moment
			# NB: if the speed is actually time-dependent
			update_boundary_indices!(view(source,:,is))
		    # if using interpolation-free Semi-Lagrange,
		    # follow characteristics backwards in time from level m+1 to level m
		    # to get departure points.  then find index of grid point nearest
		    # the departure point at time level m and use this to define
		    # an approximate characteristic
		    if use_semi_lagrange
		        for iz ∈ 1:z.n
		            find_approximate_characteristic!(SL[iz], source[iz,is], vpa, dt)
		        end
		    end
	        for iz ∈ 1:z.n
				@views advance_f_local!(ff_scratch[iz,:,is,istage+1], ff_scratch[iz,:,is,istage],
					ff[iz,:,is], SL[iz], source[iz,is], vpa, dt, istage, vpa_spectral)
			end
			enforce_vpa_boundary_condition!(view(ff_scratch,:,:,is,istage+1), vpa.bc, source)
			moments.dens_updated[is] = false ; moments.ppar_updated[is] = false
		end
	end
	for is ∈ 1:composition.n_ion_species
		@views rk_update_f!(ff[:,:,is], ff_scratch[:,:,is,:], z.n, vpa.n, n_rk_stages)
    end
end
# calculate the advection speed in the z-direction at each grid point
function update_speed_vpa!(source, phi, moments, ff, vpa, z, composition, z_spectral)
    @boundscheck z.n == size(source,1) || throw(BoundsError(source))
	@boundscheck composition.n_ion_species == size(source,2) || throw(BoundsError(source))
	@boundscheck vpa.n == size(source[1,1].speed,1) || throw(BoundsError(speed))
    if advection_speed_option_vpa == "default"
		# dvpa/dt = Ze/m ⋅ E_parallel
        update_speed_default!(source, phi, moments, ff, vpa, z, composition, z_spectral)
		#update_speed_constant!(source, vpa.n, z.n)
    elseif advection_speed_option_vpa == "constant"
		# dvpa/dt = constant
		for is ∈ 1:composition.n_ion_species
			update_speed_constant!(view(source,:,is), vpa.n, z.n)
		end
    elseif advection_speed_option_vpa == "linear"
		# dvpa/dt = constant ⋅ (vpa + L_vpa/2)
		for is ∈ 1:composition.n_ion_species
			update_speed_linear!(view(source,:,is), vpa, z.n)
		end
	end
	@inbounds begin
		for is ∈ 1:composition.n_ion_species
			for iz ∈ 1:z.n
				@. source[iz,is].modified_speed = source[iz,is].speed
			end
		end
	end
    return nothing
end
# update the advection speed dvpa/dt = Ze/m E_parallel
# if z_spectral argument has type chebyshev_info, then use Chebyshev spectral treatment
function update_speed_default!(source, phi, moments, ff, vpa, z, composition, z_spectral::chebyshev_info)
	update_phi!(phi, moments, ff, vpa, z.n, composition)
	# dphi/dz is calculated and stored in z.scratch
	chebyshev_derivative!(z.scratch2d, phi, z_spectral, z)
#	# get the Chebyshev coefficients for phi and store in z_spectral.f
#	update_fcheby!(z_spectral, phi, z)
#	# dphi/dz is calculated and stored in z.scratch
#	update_df_chebyshev!(z.scratch2d, z_spectral, z)
	@inbounds @fastmath begin
		for is ∈ 1:composition.n_ion_species
			for iz ∈ 1:z.n
#				igrid = z.igrid[iz]
#				ielem = z.ielement[iz]
				for ivpa ∈ 1:vpa.n
					igrid, ielem = set_igrid_ielem(z.igrid[iz], z.ielement[iz],
						-vpa.grid[ivpa], z.ngrid, z.nelement)
					source[iz,is].speed[ivpa] = -0.5*z.scratch2d[igrid,ielem]
				end
			end
		end
	end
end
# 'z_not_spectral' is a dummy input whose type indicates whether a spectral
# or finite diifference discretization is used for z
function update_speed_default!(source, phi, moments, ff, vpa, z, composition, z_not_spectral::Bool)
	# update the electrostatic potential phi
	update_phi!(phi, moments, ff, vpa, z.n, composition)
	# calculate the derivative of phi with respect to z
	# and store in z.scratch
	derivative_finite_difference!(z.scratch, phi, z.cell_width, z.bc,
		"second_order_centered", z.igrid, z.ielement)
	# advection velocity in vpa is -dphi/dz = -z.scratch
	@inbounds @fastmath begin
		for is ∈ 1:composition.n_ion_species
			for iz ∈ 1:z.n
				for ivpa ∈ 1:vpa.n
					source[iz,is].speed[ivpa] = -0.5*z.scratch[iz]
				end
			end
		end
	end
end
# update the advection speed dvpa/dt = constant
function update_speed_constant!(source, nvpa, nz)
	@inbounds @fastmath begin
		for iz ∈ 1:nz
			for ivpa ∈ 1:nvpa
				source[iz].speed[ivpa] = advection_speed
			end
		end
	end
end
# update the advection speed dvpa/dt = const*(vpa + L/2)
function update_speed_linear(source, vpa, nz)
	@inbounds @fastmath begin
		for iz ∈ 1:nz
			for ivpa ∈ 1:vpa.n
				source[iz].speed[ivpa] = advection_speed*(vpa.grid[ivpa]+0.5*vpa.L)
			end
		end
	end
end

end
