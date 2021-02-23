module time_advance

export rk_update_f!

function rk_update_f!(ff, ff_rk, nz, nvpa, n_rk_stages)
    @boundscheck nz == size(ff_rk,1) || throw(BoundsError(ff_rk))
    @boundscheck nvpa == size(ff_rk,2) || throw(BoundsError(ff_rk))
    @boundscheck n_rk_stages+1 == size(ff_rk,3) || throw(BoundsError(ff_rk))
    @boundscheck nz == size(ff,1) || throw(BoundsError(ff_rk))
    @boundscheck nvpa == size(ff,2) || throw(BoundsError(ff_rk))
    if n_rk_stages == 1
        @inbounds begin
            for ivpa ∈ 1:nvpa
                for iz ∈ 1:nz
                    ff[iz,ivpa] = ff_rk[iz,ivpa,2]
                end
            end
        end
    elseif n_rk_stages == 2
        @inbounds begin
            for ivpa ∈ 1:nvpa
                for iz ∈ 1:nz
                    ff[iz,ivpa] = 0.5*(ff_rk[iz,ivpa,2] + ff_rk[iz,ivpa,3])
                end
            end
        end
    elseif n_rk_stages == 3
        @inbounds begin
            for ivpa ∈ 1:nvpa
                for iz ∈ 1:nz
                    ff[iz,ivpa] = (2.0*(ff_rk[iz,ivpa,3] + ff_rk[iz,ivpa,4])-ff_rk[iz,ivpa,1])/3.0
                end
            end
        end
    end
end

end
