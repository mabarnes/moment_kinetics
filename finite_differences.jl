module finite_differences

export update_df_finite_difference!

function update_df_finite_difference!(df, f, del, adv_fac, bc)
    n = length(del)
    @boundscheck n == length(df) || throw(BoundsError(df))
	@boundscheck n == length(f) || throw(BoundsError(f))
    @boundscheck n == length(del) || throw(BoundsError(del))
	@boundscheck n == length(adv_fac) || throw(BoundsError(adv_fac))
    @inbounds @fastmath begin
        if bc == "zero"
            df[1] = 0.
        elseif bc == "periodic"
            df[1] = (f[1]-f[n-1])/del[1]
        end
        for i ∈ 3:n-2
            if adv_fac[i] < 0
                df[i] =  (3*f[i]-4*f[i-1]+f[i-2])/(2*del[i])
            else
                df[i] = (-f[i+2]+4*f[i+1]-3*f[i])/(2*del[i+1])
            end
        end
        if adv_fac[1] > 0
            df[1] = (-f[3]+4*f[2]-3*f[1])/(2*del[2])
        end
        if adv_fac[2] > 0
            df[2] = (-f[4]+4*f[3]-3*f[2])/(2*del[3])
        else
            # have to modify for periodic
            df[2] = (3*f[2]-4*f[1])/(2*del[2])
        end
        if adv_fac[n] < 0
            df[n] = (3*f[n]-4*f[n-1]+f[n-2])/(2*del[n])
        end
        if adv_fac[n-1] < 0
            df[n-1] = (3*f[n-1]-4*f[n-2]+f[n-3])/(2*del[n-1])
        else
            # have to modify for periodic
            df[n-1] = (4*f[n]-3*f[n-1])/(2*del[n])
        end
    end
end

end
