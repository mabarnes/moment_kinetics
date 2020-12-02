module finite_differences

export update_df_finite_difference!

function update_df_finite_difference!(df, f, del, adv_fac, bc, fd_option)
	if fd_option == "second_order_upwind"
		upwind_second_order!(df, f, del, adv_fac, bc)
	elseif fd_option == "first_order_upwind"
		upwind_first_order!(df, f, del, adv_fac, bc)
	end
end

function upwind_first_order!(df, f, del, adv_fac, bc)
    n = length(del)
    @boundscheck n == length(df) || throw(BoundsError(df))
	@boundscheck n == length(f) || throw(BoundsError(f))
    @boundscheck n == length(del) || throw(BoundsError(del))
	@boundscheck n == length(adv_fac) || throw(BoundsError(adv_fac))
    @inbounds @fastmath begin
        for i ∈ 2:n-1
            if adv_fac[i] < 0
                df[i] =  (f[i]-f[i-1])/del[i]
            else
                df[i] = (f[i+1]-f[i])/del[i+1]
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
			df[i] = (f[i]-tmp)/del[i]
		else
			df[i] = (f[i+1]-f[i])/del[i+1]
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
			df[i] = (tmp-f[i])/del[1]
		else
			df[i] =  (f[i]-f[i-1])/del[i]
		end
    end
end

function upwind_second_order!(df, f, del, adv_fac, bc)
    n = length(del)
    @boundscheck n == length(df) || throw(BoundsError(df))
	@boundscheck n == length(f) || throw(BoundsError(f))
    @boundscheck n == length(del) || throw(BoundsError(del))
	@boundscheck n == length(adv_fac) || throw(BoundsError(adv_fac))
    @inbounds @fastmath begin
        for i ∈ 3:n-2
            if adv_fac[i] < 0
                df[i] =  (3*f[i]-4*f[i-1]+f[i-2])/(2*del[i])
            else
                df[i] = (-f[i+2]+4*f[i+1]-3*f[i])/(2*del[i+1])
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
			df[i] = (3.0*f[i]-4.0*f[i-1]+tmp1)/(2.0*del[i])
		else
			df[i] = (-f[i+2]+4*f[i+1]-3*f[i])/(2*del[i+1])
		end
		i = 1
		if adv_fac[i] < 0
			if bc == "periodic"
				tmp2 = f[n-2]
			elseif bc == "constant"
				tmp2 = f[1]
			elseif bc == "zero"
				tmp2 = 0.0
			end
			df[i] = (3.0*f[i]-4.0*tmp1+tmp2)/(2.0*del[i])
		else
			df[i] = (-f[i+2]+4*f[i+1]-3*f[i])/(2*del[i+1])
		end
		i = n-1
		if adv_fac[i] > 0
			if bc == "periodic"
				tmp1 = f[2]
			elseif bc == "constant"
				tmp1 = f[n]
			elseif bc == "zero"
				tmp1 = 0.0
			end
			df[i] = (-tmp1+4*f[i+1]-3*f[i])/(2*del[i+1])
		else
			df[i] =  (3*f[i]-4*f[i-1]+f[i-2])/(2*del[i])
		end
		i = n
		if adv_fac[i] > 0
			if bc == "periodic"
				tmp2 = f[3]
			elseif bc == "constant"
				tmp2 = f[n]
			elseif bc == "zero"
				tmp2 = 0.0
			end
			df[i] = (-tmp2+4*tmp1-3*f[i])/(2*del[1])
		else
			df[i] =  (3*f[i]-4*f[i-1]+f[i-2])/(2*del[i])
		end
    end
end

end
