"""
"""
module finite_differences

using ..type_definitions: mk_float
import ..calculus: elementwise_derivative!, second_derivative!,
                   derivative_elements_to_full_grid!
import ..calculus: elementwise_indefinite_integration!
using ..moment_kinetics_structs: discretization_info

"""
Finite difference discretization
"""
struct finite_difference_info <: discretization_info end

"""
"""
function fd_check_option(option, ngrid)
    if option == "second_order_upwind"
        if ngrid < 3
            error("ngrid < 3 incompatible with 2rd order upwind differences.")
        end
    elseif option == "third_order_upwind"
        if ngrid < 4
            error("ngrid < 4 incompatible with 3rd order upwind differences.")
        end
    elseif option == "fourth_order_centered"
        if ngrid < 3
            error("ngrid < 3 incompatible with 4th order centered differences.")
        end
    elseif !(option in ("first_order_upwind", "second_order_centered"))
        error("finite difference option '$option' is not recognised")
    end
end

"""
    elementwise_derivative!(coord, f, adv_fac, not_spectral::finite_difference_info)

Calculate the derivative of f using finite differences, with particular scheme
specified by coord.finite_difference_option; result stored in coord.scratch_2d.
"""
function elementwise_derivative!(coord, f, adv_fac, not_spectral::finite_difference_info)
    return derivative_finite_difference!(coord.scratch_2d, f, coord.cell_width, adv_fac,
        coord.bc, coord.finite_difference_option, coord.igrid, coord.ielement)
end

"""
    elementwise_derivative!(coord, f, not_spectral::finite_difference_info)

Calculate the derivative of f using 4th order centered finite differences; result stored
in coord.scratch_2d.
"""
function elementwise_derivative!(coord, f, not_spectral::finite_difference_info)
    return derivative_finite_difference!(coord.scratch_2d, f, coord.cell_width,
        coord.bc, "fourth_order_centered", coord.igrid, coord.ielement)
end

"""
    elementwise_second_derivative!(coord, f, not_spectral::finite_difference_info)

Calculate the second derivative of f using 2nd order centered finite differences; result
stored in coord.scratch_2d.
"""
function elementwise_second_derivative!(coord, f, not_spectral::finite_difference_info)
    return second_derivative_finite_difference!(coord.scratch_2d, f, coord.cell_width,
        coord.bc, coord.igrid, coord.ielement)
end

function second_derivative!(df, f, coord, spectral::finite_difference_info)
    # Finite difference version must use an appropriate second derivative stencil, not
    # apply the 1st derivative twice as for the spectral element method

    # get the derivative at each grid point within each element and store in
    # coord.scratch_2d
    elementwise_second_derivative!(coord, f, spectral)
    # map the derivative from the elem;ntal grid to the full grid;
    # at element boundaries, use the average of the derivatives from neighboring elements.
    derivative_elements_to_full_grid!(df, coord.scratch_2d, coord)
end

"""
"""
function derivative_finite_difference!(df, f, del, adv_fac, bc, finite_difference_option,
                                       igrid, ielement)
	if finite_difference_option == "second_order_upwind"
		upwind_second_order!(df, f, del, adv_fac, bc, igrid, ielement)
	elseif finite_difference_option == "third_order_upwind"
		upwind_third_order!(df, f, del, adv_fac, bc, igrid, ielement)
	elseif finite_difference_option == "second_order_centered"
		centered_second_order!(df, f, del, bc, igrid, ielement)
	elseif finite_difference_option == "fourth_order_centered"
		centered_fourth_order!(df, f, del, bc, igrid, ielement)
	elseif finite_difference_option == "first_order_upwind"
		upwind_first_order!(df, f, del, adv_fac, bc, igrid, ielement)
	end
	# have not filled df array values for the first grid point in each element
	# after the first, as these are repeated points that overlap with the neighbouring element
	# fill them in now in case they are accessed elsewhere
	nelement = size(df,2)
	ngrid = size(df,1)
	#@inbounds begin
		if nelement > 1
			for ielem ∈ 2:nelement
				df[1,ielem] = df[ngrid,ielem-1]
			end
		end
	#end
	return nothing
end

"""
"""
function derivative_finite_difference!(df, f, del, bc, finite_difference_option, igrid,
                                       ielement)
	if finite_difference_option == "fourth_order_centered"
		centered_fourth_order!(df, f, del, bc, igrid, ielement)
	elseif finite_difference_option == "second_order_centered"
		centered_second_order!(df, f, del, bc, igrid, ielement)
	end
	return nothing
end

"""
"""
function upwind_first_order!(df, f, del, adv_fac, bc, igrid, ielement)
    n = length(del)
	@boundscheck n == length(f) || throw(BoundsError(f))
    @boundscheck n == length(del) || throw(BoundsError(del))
	@boundscheck n == length(adv_fac) || throw(BoundsError(adv_fac))
    @inbounds @fastmath begin
        for i ∈ 2:n-1
            if adv_fac[i] < 0
                #df[i] =  (f[i]-f[i-1])/del[i]
				df[igrid[i],ielement[i]] =  (f[i]-f[i-1])/del[i]
            else
                #df[i] = (f[i+1]-f[i])/del[i+1]
				df[igrid[i],ielement[i]] = (f[i+1]-f[i])/del[i+1]
            end
        end
        # fill in points at start of elements, in case we are using more than one
        for j ∈ 2:ielement[end]
            df[1, j] = df[end, j-1]
        end
		i = 1
                if adv_fac[i] >= 0.0 || bc ∈ ("zero", "both_zero", "wall")
                        # For boundary conditions that set the values of fields at the
                        # boundary, just use a one-sided difference away from the
                        # boundary.
			df[igrid[i],ielement[i]] = (f[i+1]-f[i])/del[i+1]
                else
			if bc == "periodic"
				tmp = f[n-1]
			elseif bc == "constant"
				tmp = f[1]
			elseif bc == "zero" || bc == "both_zero" || bc == "wall"
				tmp = 0.0
			end
			#df[i] = (f[i]-tmp)/del[i]
			df[igrid[i],ielement[i]] = (f[i]-tmp)/del[i]
		end
		i = n
		if adv_fac[i] <= 0 || bc ∈ ("zero", "both_zero", "wall")
                        # For boundary conditions that set the values of fields at the
                        # boundary, just use a one-sided difference away from the
                        # boundary.
			df[igrid[i],ielement[i]] =  (f[i]-f[i-1])/del[i]
                else
			if bc == "periodic"
				tmp = f[2]
			elseif bc == "constant"
				tmp = f[n]
			end
			#df[i] = (tmp-f[i])/del[1]
			df[igrid[i],ielement[i]] = (tmp-f[i])/del[1]
		end
    end
	return nothing
end

"""
"""
function upwind_second_order!(df, f, del, adv_fac, bc, igrid, ielement)
    n = length(del)
	@boundscheck n == length(f) || throw(BoundsError(f))
    @boundscheck n == length(del) || throw(BoundsError(del))
	@boundscheck n == length(adv_fac) || throw(BoundsError(adv_fac))
    @inbounds @fastmath begin
        for i ∈ 3:n-2
            if adv_fac[i] < 0
                #df[i] =  (3*f[i]-4*f[i-1]+f[i-2])/(2*del[i])
				df[igrid[i],ielement[i]] =  (3*f[i]-4*f[i-1]+f[i-2])/(2*del[i])
            else
                #df[i] = (-f[i+2]+4*f[i+1]-3*f[i])/(2*del[i+1])
				df[igrid[i],ielement[i]] = (-f[i+2]+4*f[i+1]-3*f[i])/(2*del[i+1])
            end
        end
		i = 2
		if adv_fac[i] >= 0
			df[igrid[i],ielement[i]] = (-f[i+2]+4*f[i+1]-3*f[i])/(2*del[i+1])
                elseif bc ∈ ("zero", "both_zero", "wall")
                        # For boundary conditions that set the values of fields at the
                        # boundary, use a centred difference for this second point.
			df[igrid[i],ielement[i]] = (f[i+1]-f[i-1])/(2*del[i])
                else
			if bc == "periodic"
				tmp1 = f[n-1]
			elseif bc == "constant"
				tmp1 = f[1]
			end
			#df[i] = (3.0*f[i]-4.0*f[i-1]+tmp1)/(2.0*del[i])
			df[igrid[i],ielement[i]] = (3.0*f[i]-4.0*f[i-1]+tmp1)/(2.0*del[i])
		end
		i = 1
		if adv_fac[i] >= 0 || bc ∈ ("zero", "both_zero", "wall")
                        # For boundary conditions that set the values of fields at the
                        # boundary, just use a one-sided difference away from the
                        # boundary.
			df[igrid[i],ielement[i]] = (-f[i+2]+4*f[i+1]-3*f[i])/(2*del[i+1])
                else
			if bc == "periodic"
				tmp1 = f[n-1]
				tmp2 = f[n-2]
			elseif bc == "constant"
				tmp2 = f[1]
				tmp1 = tmp2
			end
			#df[i] = (3.0*f[i]-4.0*tmp1+tmp2)/(2.0*del[i])
			df[igrid[i],ielement[i]] = (3.0*f[i]-4.0*tmp1+tmp2)/(2.0*del[i])
		end
                i = n-1
		if adv_fac[i] <= 0
			df[igrid[i],ielement[i]] =  (3*f[i]-4*f[i-1]+f[i-2])/(2*del[i])
                elseif bc ∈ ("zero", "both_zero", "wall")
                        # For boundary conditions that set the values of fields at the
                        # boundary, use a centred difference for this second point.
			df[igrid[i],ielement[i]] = (f[i+1]-f[i-1])/(2*del[i])
		else
			if bc == "periodic"
				tmp1 = f[2]
			elseif bc == "constant"
				tmp1 = f[n]
                        end
			#df[i] = (-tmp1+4*f[i+1]-3*f[i])/(2*del[i+1])
			df[igrid[i],ielement[i]] = (-tmp1+4*f[i+1]-3*f[i])/(2*del[1])
			#println("i: ", i, "  adv_fac: ", adv_fac[i], "  tmp1: ", tmp1)
		end
                i = n
		if adv_fac[i] <= 0 || bc ∈ ("zero", "both_zero", "wall")
                        # For boundary conditions that set the values of fields at the
                        # boundary, just use a one-sided difference away from the
                        # boundary.
			df[igrid[i],ielement[i]] =  (3*f[i]-4*f[i-1]+f[i-2])/(2*del[i-1])
                else
			if bc == "periodic"
				tmp2 = f[3]
				tmp1 = f[2]
			elseif bc == "constant"
				tmp2 = f[i]
				tmp1 = tmp2
			elseif bc == "zero" || bc == "both_zero" || bc == "wall"
				tmp2 = 0.0
				tmp1 = tmp2
			end
			#df[i] = (-tmp2+4*tmp1-3*f[i])/(2*del[1])
			df[igrid[i],ielement[i]] = (-tmp2+4*tmp1-3*f[i])/(2*del[1])
		end
	end
        # fill in points at start of elements, in case we are using more than one
        for j ∈ 2:ielement[end]
            df[1, j] = df[end, j-1]
        end
	return nothing
end

"""
"""
function upwind_third_order!(df, f, del, adv_fac, bc, igrid, ielement)
    n = length(del)
	@boundscheck n == length(f) && n > 3 || throw(BoundsError(f))
    @boundscheck n == length(del) || throw(BoundsError(del))
	@boundscheck n == length(adv_fac) || throw(BoundsError(adv_fac))
    #@inbounds @fastmath begin
        for i ∈ 3:n-2
			if adv_fac[i] < 0.0
				df[igrid[i],ielement[i]] =  (2.0*f[i+1]+3.0*f[i]-6.0*f[i-1]+f[i-2])/(6.0*del[i])
            else
				df[igrid[i],ielement[i]] = (-f[i+2]+6.0*f[i+1]-3.0*f[i]-2.0*f[i-1])/(6.0*del[i+1])
            end
        end
		i = 2
		if adv_fac[i] >= 0 || bc ∈ ("zero", "both_zero", "wall", "none")
                        # For boundary conditions that set the values of fields at the
                        # boundary, use the unbalanced difference away from the
                        # boundary.
			df[igrid[i],ielement[i]] = (-f[i+2]+6.0*f[i+1]-3.0*f[i]-2.0*f[i-1])/(6.0*del[i+1])
                else
			if bc == "periodic"
				tmp1 = f[n-1]
			elseif bc == "constant"
				tmp1 = f[1]
                        else
                            error("unexpected bc '$bc'")
			end
			df[igrid[i],ielement[i]] = (2.0*f[i+1]+3.0*f[i]-6.0*f[i-1]+tmp1)/(6.0*del[i])
		end
		i = 1
		if bc == "periodic"
			tmp1 = f[n-1]
			tmp2 = f[n-2]
		elseif bc == "constant"
			tmp2 = f[1]
			tmp1 = tmp2
		end
		if adv_fac[i] >= 0 || bc ∈ ("zero", "both_zero", "wall", "none")
                    if bc == "periodic"
			df[igrid[i],ielement[i]] = (-f[i+2]+6.0*f[i+1]-3.0*f[i]-2.0*tmp1)/(6.0*del[i+1])
                    else
                        # Downwind boundary, don't want to impose tmp1=0.0, so instead
                        # use one-sided difference formula
                        df[igrid[i],ielement[i]] = (2.0*f[i+3]-9.0*f[i+2]+18.0*f[i+1]-11.0*f[i])/(6.0*del[i+1])
                    end
		else
			df[igrid[i],ielement[i]] = (2.0*f[i+1]+3.0*f[i]-6.0*tmp1+tmp2)/(6.0*del[i])
		end
                i = n-1
		if adv_fac[i] <= 0 || bc ∈ ("zero", "both_zero", "wall", "none")
                        # For boundary conditions that set the values of fields at the
                        # boundary, use the unbalanced difference away from the
                        # boundary.
			df[igrid[i],ielement[i]] =  (2.0*f[i+1]+3.0*f[i]-6.0*f[i-1]+f[i-2])/(6.0*del[i])
		else
			if bc == "periodic"
				tmp1 = f[2]
			elseif bc == "constant"
				tmp1 = f[n]
                        else
                            error("unexpected bc '$bc'")
			end
			df[igrid[i],ielement[i]] = (-tmp1+6.0*f[i+1]-3.0*f[i]-2.0*f[i-1])/(6.0*del[1])
		end
		if bc == "periodic"
			tmp2 = f[3]
			tmp1 = f[2]
		elseif bc == "constant"
			tmp2 = f[n]
			tmp1 = tmp2
		end
                i = n
		if adv_fac[i] <= 0 || bc ∈ ("zero", "both_zero", "wall", "none")
                    if bc == "periodic"
			df[igrid[i],ielement[i]] =  (2.0*tmp1+3.0*f[i]-6.0*f[i-1]+f[i-2])/(6.0*del[i-1])
                    else
                        # Downwind boundary, don't want to impose tmp1=0.0, so instead
                        # use one-sided difference formula
                        df[igrid[i],ielement[i]] =  (11.0*f[i]-18.0*f[i-1]+9.0*f[i-2]-2.0*f[i-3])/(6.0*del[i-1])
                    end
		else
			df[igrid[i],ielement[i]] = (-tmp2+6.0*tmp1-3.0*f[i]-2.0*f[i-1])/(6.0*del[1])
		end
	#end
        # fill in points at start of elements, in case we are using more than one
        for j ∈ 2:ielement[end]
            df[1, j] = df[end, j-1]
        end
	return nothing
end

"""
take the derivative of input function f and return as df
using second-order, centered differences.
input/output array df is 2D array of size ngrid x nelement
"""
function centered_second_order!(df::Array{mk_float,2}, f, del, bc, igrid, ielement)
	n = length(f)
	# get derivative at internal points
	for i ∈ 2:n-1
		df[igrid[i],ielement[i]] = 0.5*(f[i+1]-f[i-1])/del[i]
	end
        # fill in points at start of elements, in case we are using more than one
        for j ∈ 2:ielement[end]
            df[1, j] = df[end, j-1]
        end
	# use BCs to treat boundary points
	if bc == "periodic"
		i = 1
		ghost = f[n-1]
		df[igrid[i],ielement[i]] = 0.5*(f[i+1]-ghost)/del[i]
		i = n
		ghost = f[2]
		df[igrid[i],ielement[i]] = 0.5*(ghost-f[i-1])/del[1]
	elseif bc == "constant"
		i = 1
		ghost = f[1]
		df[igrid[i],ielement[i]] = 0.5*(f[i+1]-ghost)/del[i]
		i = n
		ghost = f[n]
		df[igrid[i],ielement[i]] = 0.5*(ghost-f[i-1])/del[n-1]
	elseif bc == "zero" || bc == "both_zero" || bc == "wall"
                # For boundary conditions that set the values of fields at the boundary,
                # just use a first-order, one-sided difference away from the boundary.
		i = 1
                df[igrid[i],ielement[i]] = (f[i+1]-f[i])/del[i+1]
		i = n
                df[igrid[i],ielement[i]] =  (f[i]-f[i-1])/del[i-1]
	end
end

"""
take the derivative of input function f and return as df
using second-order, centered differences.
input/output df is 1D array of size n (full grid)
"""
function centered_second_order!(df::Array{mk_float,1}, f, del, bc, igrid, ielement)
	n = length(f)
	# get derivative at internal points
	for i ∈ 2:n-1
		df[i] = 0.5*(f[i+1]-f[i-1])/del[i]
	end
        # fill in points at start of elements, in case we are using more than one
        for j ∈ 2:ielement[end]
            df[1, j] = df[end, j-1]
        end
	# use BCs to treat boundary points
	if bc == "periodic"
		i = 1
		ghost = f[n-1]
		df[i] = 0.5*(f[i+1]-ghost)/del[i]
		i = n
		ghost = f[2]
		df[i] = 0.5*(ghost-f[i-1])/del[1]
	elseif bc == "constant"
		i = 1
		ghost = f[1]
		df[i] = 0.5*(f[i+1]-ghost)/del[i]
		i = n
		ghost = f[n]
		df[i] = 0.5*(ghost-f[i-1])/del[n-1]
	elseif bc == "zero" || bc == "both_zero" || bc == "wall"
                # For boundary conditions that set the values of fields at the boundary,
                # just use a one-sided difference away from the boundary.
		i = 1
                df[igrid[i],ielement[i]] = (f[i+1]-f[i])/del[i+1]
		i = n
                df[igrid[i],ielement[i]] =  (f[i]-f[i-1])/del[i-1]
	end
end

"""
take the derivative of input function f and return as df
using fourth-order, centered differences.
input/output array df is 2D array of size ngrid x nelement
"""
function centered_fourth_order!(df::Array{mk_float,2}, f, del, bc, igrid, ielement)
	n = length(f)
	# get derivative at internal points
	for i ∈ 3:n-2
		df[igrid[i],ielement[i]] = (8.0*(f[i+1]-f[i-1])+f[i-2]-f[i+2])/(12.0*del[i])
	end
	# use BCs to treat boundary points
	if bc == "periodic"
		i = 1
		ghost1 = f[n-1]
		ghost2 = f[n-2]
		df[igrid[i],ielement[i]] = (8.0*(f[i+1]-ghost1)+ghost2-f[i+2])/(12.0*del[i])
		i = 2
		df[igrid[i],ielement[i]] = (8.0*(f[i+1]-f[i-1])+ghost1-f[i+2])/(12.0*del[i])
		i = n
		ghost1 = f[2]
		ghost2 = f[3]
		df[igrid[i],ielement[i]] = (8.0*(ghost1-f[i-1])+f[i-2]-ghost2)/(12.0*del[i])
		i = n-1
		df[igrid[i],ielement[i]] = (8.0*(f[i+1]-f[i-1])+f[i-2]-ghost1)/(12.0*del[i])
	elseif bc == "constant"
		i = 1
		ghost = f[1]
		df[igrid[i],ielement[i]] = (8.0*(f[i+1]-ghost)+ghost-f[i+2])/(12.0*del[i])
		i = 2
		df[igrid[i],ielement[i]] = (8.0*(f[i+1]-f[i-1])+ghost-f[i+2])/(12.0*del[i])
		i = n
		ghost = f[n]
		df[igrid[i],ielement[i]] = (8.0*(ghost-f[i-1])+f[i-2]-ghost)/(12.0*del[i])
		i = n-1
		df[igrid[i],ielement[i]] = (8.0*(f[i+1]-f[i-1])+f[i-2]-ghost)/(12.0*del[i])
	elseif bc == "zero" || bc == "both_zero" || bc == "wall"
                # For boundary conditions that set the values of fields at the boundary,
                # use 3rd order one-sided or unbalanced difference away from the boundary.
		i = 1
                df[igrid[i],ielement[i]] = (2.0*f[i+3]-9.0*f[i+2]+18.0*f[i+1]-11.0*f[i])/(6.0*del[i+1])
		i = 2
                df[igrid[i],ielement[i]] = (-f[i+2]+6.0*f[i+1]-3.0*f[i]-2.0*f[i-1])/(6.0*del[i+1])
		i = n
                df[igrid[i],ielement[i]] =  (11.0*f[i]-18.0*f[i-1]+9.0*f[i-2]-2.0*f[i-3])/(6.0*del[i-1])
		i = n-1
                df[igrid[i],ielement[i]] =  (2.0*f[i+1]+3.0*f[i]-6.0*f[i-1]+f[i-2])/(6.0*del[i])
	end
        # fill in points at start of elements, in case we are using more than one
        for j ∈ 2:ielement[end]
            df[1, j] = df[end, j-1]
        end
end

"""
Take the second derivative of input function f and return as df using second-order,
centered differences.
output array df is 2D array of size ngrid x nelement
"""
function second_derivative_finite_difference!(df::Array{mk_float,2}, f, del, bc, igrid, ielement)
    n = length(f)
    # get derivative at internal points
    for i ∈ 2:n-1
        df[igrid[i],ielement[i]] = (f[i+1] - 2.0*f[i] + f[i-1]) / del[i]^2
    end
    # fill in points at start of elements, in case we are using more than one
    for j ∈ 2:ielement[end]
        df[1, j] = df[end, j-1]
    end
    # use BCs to treat boundary points
    if bc == "periodic"
        i = 1
        ghost = f[n-1]
        df[igrid[i],ielement[i]] = (f[i+1] - 2.0*f[i] + ghost) / del[i]^2
        i = n
        ghost = f[2]
        df[igrid[i],ielement[i]] = (ghost - 2.0*f[i] + f[i-1]) / del[i]^2
    elseif bc == "constant"
        i = 1
        ghost = f[1]
        df[igrid[i],ielement[i]] = (f[i+1] - 2.0*f[i] + ghost) / del[i]^2
        i = n
        ghost = f[n]
        df[igrid[i],ielement[i]] = (ghost - 2.0*f[i] + f[i-1]) / del[i]^2
    elseif bc == "zero" || bc == "both_zero" || bc == "wall"
        # For boundary conditions that set the values of fields at the boundary,
        # just use a one-sided difference away from the boundary.
        i = 1
        df[igrid[i],ielement[i]] = (-f[i+3] + 4.0*f[i+2] - 5.0*f[i+1] + 2.0*f[i]) / del[i]^2
        i = n
        df[igrid[i],ielement[i]] = (2.0*f[i] - 5.0*f[i-1] + 4.0*f[i-2] - f[i-3]) / del[i]^2
    end
end

"""
    elementwise_indefinite_integration!(coord, f, not_spectral::finite_difference_info)

Calculate the primitive of f using second-order accurate trapezium rule; result stored
in coord.scratch_2d.
"""
function elementwise_indefinite_integration!(coord, f, not_spectral::finite_difference_info)
    return primitive_finite_difference_integration!(coord, f, coord.finite_difference_option)
end

"""
"""
function primitive_finite_difference_integration!(coord, f, finite_difference_option)
	# space here to add different integration methods, if required
	primitive_second_order_integration!(coord, f)
	return nothing
end

"""
Integrate the input function f and return as pf
using second-order trapezium rule.
Do the integral on each element separately.
"""
function primitive_second_order_integration!(coord, f)
	n = length(f)
	ngrid = coord.ngrid
	nelement = coord.nelement_local
	# the primitive
	pf = coord.scratch_2d
	# the grid
	z = coord.grid
	igrid_full = coord.igrid_full
	for j in 1:nelement
		# lower value of primitive is zero
		# indefinite_integral_elements_to_full_grid!() in calculus.jl
		# will calculate the correct integration constants
		pf[1,j] = 0.0
		# do the integral on the element
		for i in 2:ngrid
			# the index on the full grid
			k = igrid_full[i,j]
			# the summation
			pf[i,j] = pf[i-1,j] + 0.5*(z[k] - z[k-1])*(f[k]+f[k-1])
		end
    end
end

end
