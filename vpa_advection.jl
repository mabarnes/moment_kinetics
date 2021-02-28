module vpa_advection

export vpa_advection!
export update_speed_vpa!

using semi_lagrange: find_approximate_characteristic!
using advection: update_boundary_indices!
using advection: advance_f_local!, set_igrid_ielem
using em_fields: update_phi!
using chebyshev: chebyshev_derivative!
using chebyshev: chebyshev_info
using finite_differences: derivative_finite_difference!
using initial_conditions: enforce_vpa_boundary_condition!
using velocity_moments: reset_moments_status!

# argument chebyshev indicates that a chebyshev pseudopectral method is being used
function vpa_advection!(ff, ff_scratch, fields, moments, SL, source, vpa, z,
	n_rk_stages, use_semi_lagrange, dt, t, vpa_spectral, z_spectral, composition)
    # check to ensure that all array indices accessed in this function
    # are in-bounds
    @boundscheck size(ff,1) == z.n || throw(BoundsError(ff))
    @boundscheck size(ff,2) == vpa.n || throw(BoundsError(ff))
	@boundscheck size(ff,3) == composition.n_ion_species || throw(BoundsError(ff))
	@boundscheck size(ff_scratch,1) == z.n || throw(BoundsError(ff_scratch))
    @boundscheck size(ff_scratch,2) == vpa.n || throw(BoundsError(ff_scratch))
	@boundscheck size(ff_scratch,3) == composition.n_ion_species || throw(BoundsError(ff_scratch))
	@boundscheck size(ff_scratch,4) == n_rk_stages+1 || throw(BoundsError(ff_scratch))
    # SSP RK for explicit time advance
	ff_scratch[:,:,:,1] .= ff
    for istage ∈ 1:n_rk_stages
		# for SSP RK3, need to redefine ff_scratch[3]
        if istage == 3
            @. ff_scratch[:,:,:,istage] = 0.25*(ff_scratch[:,:,:,istage] +
                ff_scratch[:,:,:,istage-1] + 2.0*ff)
        end
		@views vpa_advection_single_stage!(ff_scratch[:,:,:,istage+1], ff_scratch[:,:,:,istage],
			ff, fields, moments, SL, source, vpa, z, use_semi_lagrange, dt, t, vpa_spectral,
			z_spectral, composition, istage)
		reset_moments_status!(moments)
	end
end
function vpa_advection_single_stage!(f_out, f_in, ff, fields, moments, SL, source,
	vpa, z, use_semi_lagrange, dt, t, vpa_spectral, z_spectral, composition, istage)

	# calculate the advection speed corresponding to current f
	update_speed_vpa!(source, fields, moments, f_in, vpa, z, composition, t, z_spectral)
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
			@views advance_f_local!(f_out[iz,:,is], f_in[iz,:,is],
				ff[iz,:,is], SL[iz], source[iz,is], vpa, dt, istage, vpa_spectral,
				use_semi_lagrange)
		end
		enforce_vpa_boundary_condition!(view(f_out,:,:,is), vpa.bc, source)
	end
end
# calculate the advection speed in the z-direction at each grid point
function update_speed_vpa!(source, fields, moments, ff, vpa, z, composition, t, z_spectral)
    @boundscheck z.n == size(source,1) || throw(BoundsError(source))
	@boundscheck composition.n_ion_species == size(source,2) || throw(BoundsError(source))
	@boundscheck vpa.n == size(source[1,1].speed,1) || throw(BoundsError(speed))
    if vpa.advection.option == "default"
		# dvpa/dt = Ze/m ⋅ E_parallel
        update_speed_default!(source, fields, moments, ff, vpa, z, composition, t, z_spectral)
		#update_speed_constant!(source, vpa.n, z.n)
    elseif vpa.advection.option == "constant"
		# dvpa/dt = constant
		for is ∈ 1:composition.n_ion_species
			update_speed_constant!(view(source,:,is), vpa, z.n)
		end
    elseif vpa.advection.option == "linear"
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
function update_speed_default!(source, fields, moments, ff, vpa, z, composition,
	t, z_spectral::chebyshev_info)
	# calculate the updated electrostatic potnetial phi
	update_phi!(fields, moments, ff, vpa, z.n, composition, t)
	# dphi/dz is calculated and stored in z.scratch
	chebyshev_derivative!(z.scratch2d, fields.phi, z_spectral, z)
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
function update_speed_default!(source, fields, moments, ff, vpa, z, composition,
	t, z_not_spectral::Bool)
	# update the electrostatic potential phi
	update_phi!(fields, moments, ff, vpa, z.n, composition, t)
	# calculate the derivative of phi with respect to z
	# and store in z.scratch
	derivative_finite_difference!(z.scratch, fields.phi, z.cell_width, z.bc,
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
function update_speed_constant!(source, vpa, nz)
	@inbounds @fastmath begin
		for iz ∈ 1:nz
			for ivpa ∈ 1:vpa.n
				source[iz].speed[ivpa] = vpa.advection.constant_speed
			end
		end
	end
end
# update the advection speed dvpa/dt = const*(vpa + L/2)
function update_speed_linear(source, vpa, nz)
	@inbounds @fastmath begin
		for iz ∈ 1:nz
			for ivpa ∈ 1:vpa.n
				source[iz].speed[ivpa] = vpa.advection.constant_speed*(vpa.grid[ivpa]+0.5*vpa.L)
			end
		end
	end
end

end
