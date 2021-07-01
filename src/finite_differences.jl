function derivative_finite_difference!(df, f, del, adv_fac, bc, fd_option, igrid, ielement)
	if fd_option == "second_order_upwind"
		upwind_second_order!(df, f, del, adv_fac, bc, igrid, ielement)
	elseif fd_option == "third_order_upwind"
		upwind_third_order!(df, f, del, adv_fac, bc, igrid, ielement)
	elseif fd_option == "fourth_order_upwind"
		upwind_fourth_order!(df, f, del, bc, igrid, ielement)
	elseif fd_option == "second_order_centered"
		centered_second_order!(df, f, del, bc, igrid, ielement)
	elseif fd_option == "first_order_upwind"
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
function derivative_finite_difference!(df, f, del, bc, fd_option, igrid, ielement)
	if fd_option == "fourth_order_centered"
		centered_fourth_order!(df, f, del, bc, igrid, ielement)
	elseif fd_option == "second_order_centered"
		centered_second_order!(df, f, del, bc, igrid, ielement)
	end
	return nothing
end

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
		i = 1
		if adv_fac[i] < 0
			if bc == "periodic"
				tmp = f[n-1]
			elseif bc == "constant"
				tmp = f[1]
			elseif bc == "zero"
				tmp = 0.0
			end
			#df[i] = (f[i]-tmp)/del[i]
			df[igrid[i],ielement[i]] = (f[i]-tmp)/del[i]
		else
			#df[i] = (f[i+1]-f[i])/del[i+1]
			df[igrid[i],ielement[i]] = (f[i+1]-f[i])/del[i+1]
		end
		i = n
		if adv_fac[i] > 0
			if bc == "periodic"
				tmp = f[2]
			elseif bc == "constant"
				tmp = f[n]
			elseif bc == "zero"
				tmp = 0.0
			end
			#df[i] = (tmp-f[i])/del[1]
			df[igrid[i],ielement[i]] = (tmp-f[i])/del[1]
		else
			#df[i] =  (f[i]-f[i-1])/del[i]
			df[igrid[i],ielement[i]] =  (f[i]-f[i-1])/del[i]
		end
    end
	return nothing
end

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
		if adv_fac[i] < 0
			if bc == "periodic"
				tmp1 = f[n-1]
			elseif bc == "constant"
				tmp1 = f[1]
			elseif bc == "zero"
				tmp1 = 0.0
			end
			#df[i] = (3.0*f[i]-4.0*f[i-1]+tmp1)/(2.0*del[i])
			df[igrid[i],ielement[i]] = (3.0*f[i]-4.0*f[i-1]+tmp1)/(2.0*del[i])
		else
			#df[i] = (-f[i+2]+4*f[i+1]-3*f[i])/(2*del[i+1])
			df[igrid[i],ielement[i]] = (-f[i+2]+4*f[i+1]-3*f[i])/(2*del[i+1])
		end
		i = 1
		if adv_fac[i] < 0
			if bc == "periodic"
				tmp1 = f[n-1]
				tmp2 = f[n-2]
			elseif bc == "constant"
				tmp2 = f[1]
				tmp1 = tmp2
			elseif bc == "zero"
				tmp2 = 0.0
				tmp1 = tmp2
			end
			#df[i] = (3.0*f[i]-4.0*tmp1+tmp2)/(2.0*del[i])
			df[igrid[i],ielement[i]] = (3.0*f[i]-4.0*tmp1+tmp2)/(2.0*del[i])
		else
			#df[i] = (-f[i+2]+4*f[i+1]-3*f[i])/(2*del[i+1])
			df[igrid[i],ielement[i]] = (-f[i+2]+4*f[i+1]-3*f[i])/(2*del[i+1])
		end
		if adv_fac[n-1] > 0
			if bc == "periodic"
				tmp1 = f[2]
			elseif bc == "constant"
				tmp1 = f[n]
			elseif bc == "zero"
				tmp1 = 0.0
			end
			i = n-1
			#df[i] = (-tmp1+4*f[i+1]-3*f[i])/(2*del[i+1])
			df[igrid[i],ielement[i]] = (-tmp1+4*f[i+1]-3*f[i])/(2*del[1])
			#println("i: ", i, "  adv_fac: ", adv_fac[i], "  tmp1: ", tmp1)
		else
			i = n-1
			#df[i] =  (3*f[i]-4*f[i-1]+f[i-2])/(2*del[i])
			df[igrid[i],ielement[i]] =  (3*f[i]-4*f[i-1]+f[i-2])/(2*del[i])
		end
		if adv_fac[n] > 0
			if bc == "periodic"
				tmp2 = f[3]
				tmp1 = f[2]
			elseif bc == "constant"
				tmp2 = f[n]
				tmp1 = tmp2
			elseif bc == "zero"
				tmp2 = 0.0
				tmp1 = tmp2
			end
			#df[i] = (-tmp2+4*tmp1-3*f[i])/(2*del[1])
			df[igrid[n],ielement[n]] = (-tmp2+4*tmp1-3*f[n])/(2*del[1])
		else
			#df[i] =  (3*f[i]-4*f[i-1]+f[i-2])/(2*del[i])
			df[igrid[n],ielement[n]] =  (3*f[n]-4*f[n-1]+f[n-2])/(2*del[n-1])
		end
	end
	return nothing
end
function upwind_third_order!(df, f, del, adv_fac, bc, igrid, ielement)
    n = length(del)
	@boundscheck n == length(f) && n > 3 || throw(BoundsError(f))
    @boundscheck n == length(del) || throw(BoundsError(del))
	@boundscheck n == length(adv_fac) || throw(BoundsError(adv_fac))
    #@inbounds @fastmath begin
        for i ∈ 3:n-2
            if adv_fac[i] < 0
				df[igrid[i],ielement[i]] =  (2*f[i+1]+3*f[i]-6*f[i-1]+f[i-2])/(6*del[i])
            else
				df[igrid[i],ielement[i]] = (-f[i+2]+6*f[i+1]-3*f[i]-2*f[i-1])/(6*del[i+1])
            end
        end
		i = 2
		if adv_fac[i] < 0
			if bc == "periodic"
				tmp1 = f[n-1]
			elseif bc == "constant"
				tmp1 = f[1]
			elseif bc == "zero"
				tmp1 = 0.0
			end
			df[igrid[i],ielement[i]] = (2.0*f[i+1]+3.0*f[i]-6.0*f[i-1]+tmp1)/(6.0*del[i])
		else
			df[igrid[i],ielement[i]] = (-f[i+2]+6*f[i+1]-3*f[i]-2*f[i-1])/(6*del[i+1])
		end
		i = 1
		if bc == "periodic"
			tmp1 = f[n-1]
			tmp2 = f[n-2]
		elseif bc == "constant"
			tmp2 = f[1]
			tmp1 = tmp2
		elseif bc == "zero"
			tmp2 = 0.0
			tmp1 = tmp2
		end
		if adv_fac[i] < 0
			df[igrid[i],ielement[i]] = (2.0*f[i+1]+3.0*f[i]-6.0*tmp1+tmp2)/(6.0*del[i])
		else
			df[igrid[i],ielement[i]] = (-f[i+2]+6*f[i+1]-3*f[i]-2*tmp1)/(6*del[i+1])
		end
		if adv_fac[n-1] > 0
			if bc == "periodic"
				tmp1 = f[2]
			elseif bc == "constant"
				tmp1 = f[n]
			elseif bc == "zero"
				tmp1 = 0.0
			end
			i = n-1
			df[igrid[i],ielement[i]] = (-tmp1+6*f[i+1]-3*f[i]-2*f[i-1])/(6*del[1])
		else
			i = n-1
			df[igrid[i],ielement[i]] =  (2*f[i+1]+3*f[i]-6*f[i-1]+f[i-2])/(6*del[i])
		end
		if bc == "periodic"
			tmp2 = f[3]
			tmp1 = f[2]
		elseif bc == "constant"
			tmp2 = f[n]
			tmp1 = tmp2
		elseif bc == "zero"
			tmp2 = 0.0
			tmp1 = tmp2
		end
		if adv_fac[n] > 0
			df[igrid[n],ielement[n]] = (-tmp2+6*tmp1-3*f[n]-2*f[n-1])/(6*del[1])
		else
			df[igrid[n],ielement[n]] =  (2*tmp1+3*f[n]-6*f[n-1]+f[n-2])/(6*del[n-1])
		end
	#end
	return nothing
end
# take the derivative of input function f and return as df
# using second-order, centered differences.
# input/output array df is 2D array of size ngrid x nelement
function centered_second_order!(df::Array{mk_float,2}, f, del, bc, igrid, ielement)
	n = length(f)
	# get derivative at internal points
	for i ∈ 2:n-1
		df[igrid[i],ielement[i]] = 0.5*(f[i+1]-f[i-1])/del[i]
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
	elseif bc == "zero"
		i = 1
		df[igrid[i],ielement[i]] = 0.5*f[i+1]/del[i]
		i = n
		df[igrid[i],ielement[i]] = -0.5*f[i-1]/del[n-1]
	end
end
# take the derivative of input function f and return as df
# using second-order, centered differences.
# input/output df is 1D array of size n (full grid)
function centered_second_order!(df::Array{mk_float,1}, f, del, bc, igrid, ielement)
	n = length(f)
	# get derivative at internal points
	for i ∈ 2:n-1
		df[i] = 0.5*(f[i+1]-f[i-1])/del[i]
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
	elseif bc == "zero"
		i = 1
		df[i] = 0.5*f[i+1]/del[i]
		i = n
		df[i] = -0.5*f[i-1]/del[n-1]
	end
end
# take the derivative of input function f and return as df
# using fourth-order, centered differences.
# input/output array df is 2D array of size ngrid x nelement
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
	elseif bc == "zero"
		i = 1
		df[igrid[i],ielement[i]] = (8.0*f[i+1]-f[i+2])/(12.0*del[i])
		i = 2
		df[igrid[i],ielement[i]] = (8.0*(f[i+1]-f[i-1])-f[i+2])/(12.0*del[i])
		i = n
		df[igrid[i],ielement[i]] = (-8.0*f[i-1]+f[i-2])/(12.0*del[i])
		i = n-1
		df[igrid[i],ielement[i]] = (8.0*(f[i+1]-f[i-1])+f[i-2])/(12.0*del[i])
	end
end
