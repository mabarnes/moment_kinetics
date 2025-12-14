"""
This module contains all the necessary derivatives needed to carry out
distributed memory differential operations on the arrays in moment kinetics.
We provide separate derivative functions for each (i) distributed dimension
and (ii) array shape. We do not need to provide derivatives for non-distributed
dimensions as these can by handled by the derivative! function from calculus.jl

"""
module derivatives

export derivative_r!, derivative_r_chrg!, derivative_r_ntrl!
export derivative_z!, derivative_z_chrg!, derivative_z_ntrl!

using ..calculus: derivative!, second_derivative!, reconcile_element_boundaries_MPI!,
                  reconcile_element_boundaries_MPI_anyzv!,
                  reconcile_element_boundaries_MPI_z_pdf_vpavperpz!
using ..communication
using ..type_definitions: mk_float
using ..looping

using MPI

"""
Centered derivatives
df/dr group of rountines for
fields & moments -> [z,r]
dfns (ion) -> [vpa,vperp,z,r,s]
dfns (neutrals) -> [vz,vr,vzeta,z,r,sn]
"""

#df/dr
#2D version for f[z,r] -> Er, Ez, phi
function derivative_r!(dfdr::AbstractArray{mk_float,2}, f::AbstractArray{mk_float,2},
        dfdr_lower_endpoints::AbstractArray{mk_float,1},
        dfdr_upper_endpoints::AbstractArray{mk_float,1},
        r_receive_buffer1::AbstractArray{mk_float,1},
        r_receive_buffer2::AbstractArray{mk_float,1}, r_spectral, r)

        @begin_z_region()

	# differentiate f w.r.t r
	@loop_z iz begin
		@views derivative!(dfdr[iz,:], f[iz,:], r, r_spectral)
		# get external endpoints to reconcile via MPI
		dfdr_lower_endpoints[iz] = r.scratch_2d[1,1]
		dfdr_upper_endpoints[iz] = r.scratch_2d[end,end]
	end
	# now reconcile element boundaries across
	# processes with large message involving all other dimensions
	if r.nelement_local < r.nelement_global
		reconcile_element_boundaries_MPI!(dfdr,
		 dfdr_lower_endpoints,dfdr_upper_endpoints,
		 r_receive_buffer1, r_receive_buffer2, r)
	end
	
end

#df/dr
#3D version for f[s,z,r] -> moments n, u, T etc
function derivative_r!(dfdr::AbstractArray{mk_float,3}, f::AbstractArray{mk_float,3},
        dfdr_lower_endpoints::AbstractArray{mk_float,2},
        dfdr_upper_endpoints::AbstractArray{mk_float,2},
        r_receive_buffer1::AbstractArray{mk_float,2},
        r_receive_buffer2::AbstractArray{mk_float,2}, r_spectral, r; neutrals=false)

    # differentiate f w.r.t r
    if neutrals
	@loop_sn_z isn iz begin
		@views derivative!(dfdr[iz,:,isn], f[iz,:,isn], r, r_spectral)
		# get external endpoints to reconcile via MPI
		dfdr_lower_endpoints[iz,isn] = r.scratch_2d[1,1]
		dfdr_upper_endpoints[iz,isn] = r.scratch_2d[end,end]
	end
    else
	@loop_s_z is iz begin
		@views derivative!(dfdr[iz,:,is], f[iz,:,is], r, r_spectral)
		# get external endpoints to reconcile via MPI
		dfdr_lower_endpoints[iz,is] = r.scratch_2d[1,1]
		dfdr_upper_endpoints[iz,is] = r.scratch_2d[end,end]
	end
    end

    # Sometimes an array might contain no data (e.g. if n_neutral_species=0). Then don't
    # need to reconcile boundaries
    if length(dfdr) > 0
	# now reconcile element boundaries across
	# processes with large message involving all other dimensions
	if r.nelement_local < r.nelement_global
		reconcile_element_boundaries_MPI!(dfdr,
		 dfdr_lower_endpoints,dfdr_upper_endpoints,
		 r_receive_buffer1, r_receive_buffer2, r; neutrals)
	end
    end
end

#df/dr
#5D version for f[vpa,vperp,z,r,s] -> ion particle dfn
function derivative_r!(dfdr::AbstractArray{mk_float,5}, f::AbstractArray{mk_float,5},
        dfdr_lower_endpoints::AbstractArray{mk_float,4},
        dfdr_upper_endpoints::AbstractArray{mk_float,4},
        r_receive_buffer1::AbstractArray{mk_float,4},
        r_receive_buffer2::AbstractArray{mk_float,4}, r_spectral, r)

        @begin_s_z_vperp_vpa_region()

	# differentiate f w.r.t r
	@loop_s_z_vperp_vpa is iz ivperp ivpa begin
		@views derivative!(dfdr[ivpa,ivperp,iz,:,is], f[ivpa,ivperp,iz,:,is], r, r_spectral)
		# get external endpoints to reconcile via MPI
		dfdr_lower_endpoints[ivpa,ivperp,iz,is] = r.scratch_2d[1,1]
		dfdr_upper_endpoints[ivpa,ivperp,iz,is] = r.scratch_2d[end,end]
	end
	# now reconcile element boundaries across
	# processes with large message involving all other dimensions
	if r.nelement_local < r.nelement_global
		reconcile_element_boundaries_MPI!(dfdr,
		 dfdr_lower_endpoints,dfdr_upper_endpoints,
		 r_receive_buffer1, r_receive_buffer2, r)
	end
	
end

#6D version for f[vz,vz,vzeta,z,r,sn] -> neutral particle dfn (species indexing taken outside this loop)
function derivative_r!(dfdr::AbstractArray{mk_float,6}, f::AbstractArray{mk_float,6},
        dfdr_lower_endpoints::AbstractArray{mk_float,5},
        dfdr_upper_endpoints::AbstractArray{mk_float,5},
        r_receive_buffer1::AbstractArray{mk_float,5},
        r_receive_buffer2::AbstractArray{mk_float,5}, r_spectral, r)

        @begin_sn_z_vzeta_vr_vz_region()

	# differentiate f w.r.t r
	@loop_sn_z_vzeta_vr_vz isn iz ivzeta ivr ivz begin
		@views derivative!(dfdr[ivz,ivr,ivzeta,iz,:,isn], f[ivz,ivr,ivzeta,iz,:,isn], r, r_spectral)
		# get external endpoints to reconcile via MPI
		dfdr_lower_endpoints[ivz,ivr,ivzeta,iz,isn] = r.scratch_2d[1,1]
		dfdr_upper_endpoints[ivz,ivr,ivzeta,iz,isn] = r.scratch_2d[end,end]
	end
	# now reconcile element boundaries across
	# processes with large message involving all other dimensions
	if r.nelement_local < r.nelement_global
		reconcile_element_boundaries_MPI!(dfdr,
		 dfdr_lower_endpoints,dfdr_upper_endpoints,
		 r_receive_buffer1, r_receive_buffer2, r)
	end
	
end

"""
Centered derivatives
df/dz group of rountines for
fields & moments -> [z,r]
dfns (ion) -> [vpa,vperp,z,r,s]
dfns (neutrals) -> [vz,vr,vzeta,z,r,sn]
"""

#df/dz
#1D version for f[z], used by implicit solvers
function derivative_z!(dfdz::AbstractArray{mk_float,1}, f::AbstractArray{mk_float,1},
        dfdz_lower_endpoints::AbstractArray{mk_float,0},
        dfdz_upper_endpoints::AbstractArray{mk_float,0},
        z_send_buffer::AbstractArray{mk_float,0},
        z_receive_buffer::AbstractArray{mk_float,0}, z_spectral, z)

    @begin_serial_region()

    @serial_region begin
        # differentiate f w.r.t z
        derivative!(dfdz, f, z, z_spectral)
        # get external endpoints to reconcile via MPI
        dfdz_lower_endpoints[] = z.scratch_2d[1,1]
        dfdz_upper_endpoints[] = z.scratch_2d[end,end]
    end

    # now reconcile element boundaries across
    # processes with large message involving all y
    if z.nelement_local < z.nelement_global
        reconcile_element_boundaries_MPI!(
            dfdz, dfdz_lower_endpoints, dfdz_upper_endpoints, z_send_buffer,
            z_receive_buffer, z)
    end
end

# Alternative version of 1D deriv for use within an anyzv parallel region.
function derivative_z_anyzv!(dfdz::AbstractArray{mk_float,1}, f::AbstractArray{mk_float,1},
        dfdz_lower_endpoints::AbstractArray{mk_float,0},
        dfdz_upper_endpoints::AbstractArray{mk_float,0},
        z_send_buffer::AbstractArray{mk_float,0},
        z_receive_buffer::AbstractArray{mk_float,0}, z_spectral, z)

    @begin_anyzv_region()

    @anyzv_serial_region begin
        # differentiate f w.r.t z
        derivative!(dfdz, f, z, z_spectral)
        # get external endpoints to reconcile via MPI
        dfdz_lower_endpoints[] = z.scratch_2d[1,1]
        dfdz_upper_endpoints[] = z.scratch_2d[end,end]
    end

    # now reconcile element boundaries across
    # processes with large message involving all y
    if z.nelement_local < z.nelement_global
        reconcile_element_boundaries_MPI_anyzv!(
            dfdz, dfdz_lower_endpoints, dfdz_upper_endpoints, z_send_buffer,
            z_receive_buffer, z)
    end
end

#df/dz
#2D version for f[z,r] -> Er, Ez, phi
function derivative_z!(dfdz::AbstractArray{mk_float,2}, f::AbstractArray{mk_float,2},
        dfdz_lower_endpoints::AbstractArray{mk_float,1},
        dfdz_upper_endpoints::AbstractArray{mk_float,1},
        z_send_buffer::AbstractArray{mk_float,1},
        z_receive_buffer::AbstractArray{mk_float,1}, z_spectral, z)

        @begin_r_region()

	# differentiate f w.r.t z
	@loop_r ir begin
		@views derivative!(dfdz[:,ir], f[:,ir], z, z_spectral)
		# get external endpoints to reconcile via MPI
		dfdz_lower_endpoints[ir] = z.scratch_2d[1,1]
		dfdz_upper_endpoints[ir] = z.scratch_2d[end,end]
	end
	# now reconcile element boundaries across
	# processes with large message involving all y
	if z.nelement_local < z.nelement_global
		reconcile_element_boundaries_MPI!(dfdz,
		 dfdz_lower_endpoints,dfdz_upper_endpoints,
		 z_send_buffer, z_receive_buffer, z)
	end
	
end

#df/dz
#3D version for f[z,r] -> moments n, u, T etc
function derivative_z!(dfdz::AbstractArray{mk_float,3}, f::AbstractArray{mk_float,3},
        dfdz_lower_endpoints::AbstractArray{mk_float,2},
        dfdz_upper_endpoints::AbstractArray{mk_float,2},
        z_send_buffer::AbstractArray{mk_float,2},
        z_receive_buffer::AbstractArray{mk_float,2}, z_spectral, z; neutrals=false)

    # differentiate f w.r.t z
    if neutrals
	@loop_sn_r isn ir begin
		@views derivative!(dfdz[:,ir,isn], f[:,ir,isn], z, z_spectral)
		# get external endpoints to reconcile via MPI
		dfdz_lower_endpoints[ir,isn] = z.scratch_2d[1,1]
		dfdz_upper_endpoints[ir,isn] = z.scratch_2d[end,end]
	end
    else
	@loop_s_r is ir begin
		@views derivative!(dfdz[:,ir,is], f[:,ir,is], z, z_spectral)
		# get external endpoints to reconcile via MPI
		dfdz_lower_endpoints[ir,is] = z.scratch_2d[1,1]
		dfdz_upper_endpoints[ir,is] = z.scratch_2d[end,end]
	end
    end

    # Sometimes an array might contain no data (e.g. if n_neutral_species=0). Then don't
    # need to reconcile boundaries
    if length(dfdz) > 0
	# now reconcile element boundaries across
	# processes with large message involving all y
	if z.nelement_local < z.nelement_global
		reconcile_element_boundaries_MPI!(dfdz,
		 dfdz_lower_endpoints,dfdz_upper_endpoints,
		 z_send_buffer, z_receive_buffer, z; neutrals)
	end
    end
end

# df/dz
# 3D version for f[vpa,vperp,z]. Uses modified function name to avoid clash with 'standard'
# 3D version for ion/neutral moments.
function derivative_z_pdf_vpavperpz!(dfdz::AbstractArray{mk_float,3}, f::AbstractArray{mk_float,3},
        dfdz_lower_endpoints::AbstractArray{mk_float,2},
        dfdz_upper_endpoints::AbstractArray{mk_float,2},
        z_receive_buffer1::AbstractArray{mk_float,2},
        z_receive_buffer2::AbstractArray{mk_float,2}, z_spectral, z)

    # differentiate f w.r.t z
    @loop_vperp_vpa ivperp ivpa begin
        @views derivative!(dfdz[ivpa,ivperp,:], f[ivpa,ivperp,:], z, z_spectral)
        # get external endpoints to reconcile via MPI
        dfdz_lower_endpoints[ivpa,ivperp] = z.scratch_2d[1,1]
        dfdz_upper_endpoints[ivpa,ivperp] = z.scratch_2d[end,end]
    end

    # now reconcile element boundaries across
    # processes with large message
    if z.nelement_local < z.nelement_global
        reconcile_element_boundaries_MPI_z_pdf_vpavperpz!(
            dfdz, dfdz_lower_endpoints, dfdz_upper_endpoints, z_receive_buffer1,
            z_receive_buffer2, z)
    end
end

#5D version for f[vpa,vperp,z,r,s] -> dfn ions
function derivative_z!(dfdz::AbstractArray{mk_float,5}, f::AbstractArray{mk_float,5},
        dfdz_lower_endpoints::AbstractArray{mk_float,4},
        dfdz_upper_endpoints::AbstractArray{mk_float,4},
        z_send_buffer::AbstractArray{mk_float,4},
        z_receive_buffer::AbstractArray{mk_float,4}, z_spectral, z)

        @begin_s_r_vperp_vpa_region()

	# differentiate f w.r.t z
	@loop_s_r_vperp_vpa is ir ivperp ivpa begin
		@views derivative!(dfdz[ivpa,ivperp,:,ir,is], f[ivpa,ivperp,:,ir,is], z, z_spectral)
		# get external endpoints to reconcile via MPI
		dfdz_lower_endpoints[ivpa,ivperp,ir,is] = z.scratch_2d[1,1]
		dfdz_upper_endpoints[ivpa,ivperp,ir,is] = z.scratch_2d[end,end]
	end
	# now reconcile element boundaries across
	# processes with large message involving all y
	if z.nelement_local < z.nelement_global
		reconcile_element_boundaries_MPI!(dfdz,
		 dfdz_lower_endpoints,dfdz_upper_endpoints,
		 z_send_buffer, z_receive_buffer, z)
	end
	
end

#4D version for f[vpa,vperp,z,r] -> dfn electron particles
function derivative_z!(dfdz::AbstractArray{mk_float,4}, f::AbstractArray{mk_float,4},
	dfdz_lower_endpoints::AbstractArray{mk_float,3},
	dfdz_upper_endpoints::AbstractArray{mk_float,3},
	z_send_buffer::AbstractArray{mk_float,3},
	z_receive_buffer::AbstractArray{mk_float,3}, z_spectral, z)

        @begin_r_vperp_vpa_region()

	# differentiate f w.r.t z
	@loop_r_vperp_vpa ir ivperp ivpa begin
		@views derivative!(dfdz[ivpa,ivperp,:,ir], f[ivpa,ivperp,:,ir], z, z_spectral)
		# get external endpoints to reconcile via MPI
		dfdz_lower_endpoints[ivpa,ivperp,ir] = z.scratch_2d[1,1]
		dfdz_upper_endpoints[ivpa,ivperp,ir] = z.scratch_2d[end,end]
	end
	# now reconcile element boundaries across
	# processes with large message involving all y
	if z.nelement_local < z.nelement_global
		reconcile_element_boundaries_MPI!(dfdz,
		dfdz_lower_endpoints, dfdz_upper_endpoints,
		z_send_buffer, z_receive_buffer, z)
	end

end

#6D version for f[vz,vr,vzeta,z,r] -> dfn neutral particles
function derivative_z!(dfdz::AbstractArray{mk_float,6}, f::AbstractArray{mk_float,6},
        dfdz_lower_endpoints::AbstractArray{mk_float,5},
        dfdz_upper_endpoints::AbstractArray{mk_float,5},
        z_send_buffer::AbstractArray{mk_float,5},
        z_receive_buffer::AbstractArray{mk_float,5}, z_spectral, z)

        @begin_sn_r_vzeta_vr_vz_region()

	# differentiate f w.r.t z
	@loop_sn_r_vzeta_vr_vz isn ir ivzeta ivr ivz begin
		@views derivative!(dfdz[ivz,ivr,ivzeta,:,ir,isn], f[ivz,ivr,ivzeta,:,ir,isn], z, z_spectral)
		# get external endpoints to reconcile via MPI
		dfdz_lower_endpoints[ivz,ivr,ivzeta,ir,isn] = z.scratch_2d[1,1]
		dfdz_upper_endpoints[ivz,ivr,ivzeta,ir,isn] = z.scratch_2d[end,end]
	end
	# now reconcile element boundaries across
	# processes with large message involving all y
	if z.nelement_local < z.nelement_global
		reconcile_element_boundaries_MPI!(dfdz,
		 dfdz_lower_endpoints,dfdz_upper_endpoints,
		 z_send_buffer, z_receive_buffer, z)
	end
	
end

"""
Centered derivatives
df2/dr2 group of rountines for
fields & moments -> [z,r]
dfns (ion) -> [vpa,vperp,z,r,s]
dfns (neutrals) -> [vz,vr,vzeta,z,r,sn]
"""

#d2f/dr2
#2D version for f[z,r] -> Er, Ez, phi
function second_derivative_r!(d2fdr2::AbstractArray{mk_float,2}, f::AbstractArray{mk_float,2},
        d2fdr2_lower_endpoints::AbstractArray{mk_float,1},
        d2fdr2_upper_endpoints::AbstractArray{mk_float,1},
        r_receive_buffer1::AbstractArray{mk_float,1},
        r_receive_buffer2::AbstractArray{mk_float,1}, r_spectral, r)

    @begin_z_region()

    # differentiate f w.r.t r
    @loop_z iz begin
        @views second_derivative!(d2fdr2[iz,:], f[iz,:], r, r_spectral)
        # get external endpoints to reconcile via MPI
        d2fdr2_lower_endpoints[iz] = r.scratch_2d[1,1]
        d2fdr2_upper_endpoints[iz] = r.scratch_2d[end,end]
    end
    # now reconcile element boundaries across
    # processes with large message involving all other dimensions
    if r.nelement_local < r.nelement_global
        reconcile_element_boundaries_MPI!(d2fdr2, d2fdr2_lower_endpoints,
                                          d2fdr2_upper_endpoints, r_receive_buffer1,
                                          r_receive_buffer2, r)
    end
end

#d2f/dr2
#3D version for f[s,z,r] -> moments n, u, T etc
function second_derivative_r!(d2fdr2::AbstractArray{mk_float,3}, f::AbstractArray{mk_float,3},
        d2fdr2_lower_endpoints::AbstractArray{mk_float,2},
        d2fdr2_upper_endpoints::AbstractArray{mk_float,2},
        r_receive_buffer1::AbstractArray{mk_float,2},
        r_receive_buffer2::AbstractArray{mk_float,2}, r_spectral, r; neutrals=false)

    # differentiate f w.r.t r
    if neutrals
        @loop_sn_z isn iz begin
            @views second_derivative!(d2fdr2[iz,:,isn], f[iz,:,isn], r, r_spectral)
            # get external endpoints to reconcile via MPI
            d2fdr2_lower_endpoints[iz,isn] = r.scratch_2d[1,1]
            d2fdr2_upper_endpoints[iz,isn] = r.scratch_2d[end,end]
        end
    else
        @loop_s_z is iz begin
            @views second_derivative!(d2fdr2[iz,:,is], f[iz,:,is], r, r_spectral)
            # get external endpoints to reconcile via MPI
            d2fdr2_lower_endpoints[iz,is] = r.scratch_2d[1,1]
            d2fdr2_upper_endpoints[iz,is] = r.scratch_2d[end,end]
        end
    end

    # Sometimes an array might contain no data (e.g. if n_neutral_species=0). Then don't
    # need to reconcile boundaries
    if length(d2fdr2) > 0
        # now reconcile element boundaries across
        # processes with large message involving all other dimensions
        if r.nelement_local < r.nelement_global
            reconcile_element_boundaries_MPI!(d2fdr2, d2fdr2_lower_endpoints,
                                              d2fdr2_upper_endpoints, r_receive_buffer1,
                                              r_receive_buffer2, r; neutrals)
        end
    end
end

#d2f/dr2
#5D version for f[vpa,vperp,z,r,s] -> ion particle dfn
function second_derivative_r!(d2fdr2::AbstractArray{mk_float,5}, f::AbstractArray{mk_float,5},
        d2fdr2_lower_endpoints::AbstractArray{mk_float,4},
        d2fdr2_upper_endpoints::AbstractArray{mk_float,4},
        r_receive_buffer1::AbstractArray{mk_float,4},
        r_receive_buffer2::AbstractArray{mk_float,4}, r_spectral, r)

    @begin_s_z_vperp_vpa_region()

    # differentiate f w.r.t r
    @loop_s_z_vperp_vpa is iz ivperp ivpa begin
        @views second_derivative!(d2fdr2[ivpa,ivperp,iz,:,is], f[ivpa,ivperp,iz,:,is], r, r_spectral)
        # get external endpoints to reconcile via MPI
        d2fdr2_lower_endpoints[ivpa,ivperp,iz,is] = r.scratch_2d[1,1]
        d2fdr2_upper_endpoints[ivpa,ivperp,iz,is] = r.scratch_2d[end,end]
    end
    # now reconcile element boundaries across
    # processes with large message involving all other dimensions
    if r.nelement_local < r.nelement_global
        reconcile_element_boundaries_MPI!(d2fdr2, d2fdr2_lower_endpoints,
                                          d2fdr2_upper_endpoints, r_receive_buffer1,
                                          r_receive_buffer2, r)
    end
end

#6D version for f[vz,vz,vzeta,z,r,sn] -> neutral particle dfn (species indexing taken outside this loop)
function second_derivative_r!(d2fdr2::AbstractArray{mk_float,6}, f::AbstractArray{mk_float,6},
        d2fdr2_lower_endpoints::AbstractArray{mk_float,5},
        d2fdr2_upper_endpoints::AbstractArray{mk_float,5},
        r_receive_buffer1::AbstractArray{mk_float,5},
        r_receive_buffer2::AbstractArray{mk_float,5}, r_spectral, r)

    @begin_sn_z_vzeta_vr_vz_region()

    # differentiate f w.r.t r
    @loop_sn_z_vzeta_vr_vz isn iz ivzeta ivr ivz begin
        @views second_derivative!(d2fdr2[ivz,ivr,ivzeta,iz,:,isn], f[ivz,ivr,ivzeta,iz,:,isn], r, r_spectral)
        # get external endpoints to reconcile via MPI
        d2fdr2_lower_endpoints[ivz,ivr,ivzeta,iz,isn] = r.scratch_2d[1,1]
        d2fdr2_upper_endpoints[ivz,ivr,ivzeta,iz,isn] = r.scratch_2d[end,end]
    end
    # now reconcile element boundaries across
    # processes with large message involving all other dimensions
    if r.nelement_local < r.nelement_global
        reconcile_element_boundaries_MPI!(d2fdr2, d2fdr2_lower_endpoints,
                                          d2fdr2_upper_endpoints, r_receive_buffer1,
                                          r_receive_buffer2, r)
    end
end

"""
Centered derivatives
df/dz group of rountines for
fields & moments -> [z,r]
dfns (ion) -> [vpa,vperp,z,r,s]
dfns (neutrals) -> [vz,vr,vzeta,z,r,sn]
"""

#d2f/dz2
#2D version for f[z,r] -> Er, Ez, phi
function second_derivative_z!(d2fdz2::AbstractArray{mk_float,2}, f::AbstractArray{mk_float,2},
        d2fdz2_lower_endpoints::AbstractArray{mk_float,1},
        d2fdz2_upper_endpoints::AbstractArray{mk_float,1},
        z_send_buffer::AbstractArray{mk_float,1},
        z_receive_buffer::AbstractArray{mk_float,1}, z_spectral, z)

    @begin_r_region()

    # differentiate f w.r.t z
    @loop_r ir begin
        @views second_derivative!(d2fdz2[:,ir], f[:,ir], z, z_spectral)
        # get external endpoints to reconcile via MPI
        d2fdz2_lower_endpoints[ir] = z.scratch_2d[1,1]
        d2fdz2_upper_endpoints[ir] = z.scratch_2d[end,end]
    end
    # now reconcile element boundaries across
    # processes with large message involving all y
    if z.nelement_local < z.nelement_global
        reconcile_element_boundaries_MPI!(d2fdz2, d2fdz2_lower_endpoints,
                                          d2fdz2_upper_endpoints, z_send_buffer,
                                          z_receive_buffer, z)
    end
end

#d2f/dz2
#3D version for f[z,r] -> moments n, u, T etc
function second_derivative_z!(d2fdz2::AbstractArray{mk_float,3}, f::AbstractArray{mk_float,3},
        d2fdz2_lower_endpoints::AbstractArray{mk_float,2},
        d2fdz2_upper_endpoints::AbstractArray{mk_float,2},
        z_send_buffer::AbstractArray{mk_float,2},
        z_receive_buffer::AbstractArray{mk_float,2}, z_spectral, z; neutrals=false)

    # differentiate f w.r.t z
    if neutrals
        @loop_sn_r isn ir begin
            @views second_derivative!(d2fdz2[:,ir,isn], f[:,ir,isn], z, z_spectral)
            # get external endpoints to reconcile via MPI
            d2fdz2_lower_endpoints[ir,isn] = z.scratch_2d[1,1]
            d2fdz2_upper_endpoints[ir,isn] = z.scratch_2d[end,end]
        end
    else
        @loop_s_r is ir begin
            @views second_derivative!(d2fdz2[:,ir,is], f[:,ir,is], z, z_spectral)
            # get external endpoints to reconcile via MPI
            d2fdz2_lower_endpoints[ir,is] = z.scratch_2d[1,1]
            d2fdz2_upper_endpoints[ir,is] = z.scratch_2d[end,end]
        end
    end

    # Sometimes an array might contain no data (e.g. if n_neutral_species=0). Then don't
    # need to reconcile boundaries
    if length(d2fdz2) > 0
        # now reconcile element boundaries across
        # processes with large message involving all y
        if z.nelement_local < z.nelement_global
            reconcile_element_boundaries_MPI!(d2fdz2, d2fdz2_lower_endpoints,
                                              d2fdz2_upper_endpoints, z_send_buffer,
                                              z_receive_buffer, z; neutrals)
        end
    end
end

#5D version for f[vpa,vperp,z,r,s] -> dfn ions
function second_derivative_z!(d2fdz2::AbstractArray{mk_float,5}, f::AbstractArray{mk_float,5},
        d2fdz2_lower_endpoints::AbstractArray{mk_float,4},
        d2fdz2_upper_endpoints::AbstractArray{mk_float,4},
        z_send_buffer::AbstractArray{mk_float,4},
        z_receive_buffer::AbstractArray{mk_float,4}, z_spectral, z)

    @begin_s_r_vperp_vpa_region()

    # differentiate f w.r.t z
    @loop_s_r_vperp_vpa is ir ivperp ivpa begin
        @views second_derivative!(d2fdz2[ivpa,ivperp,:,ir,is], f[ivpa,ivperp,:,ir,is], z, z_spectral)
        # get external endpoints to reconcile via MPI
        d2fdz2_lower_endpoints[ivpa,ivperp,ir,is] = z.scratch_2d[1,1]
        d2fdz2_upper_endpoints[ivpa,ivperp,ir,is] = z.scratch_2d[end,end]
    end
    # now reconcile element boundaries across
    # processes with large message involving all y
    if z.nelement_local < z.nelement_global
        reconcile_element_boundaries_MPI!(d2fdz2, d2fdz2_lower_endpoints,
                                          d2fdz2_upper_endpoints, z_send_buffer,
                                          z_receive_buffer, z)
    end
end

#6D version for f[vz,vr,vzeta,z,r] -> dfn neutral particles
function second_derivative_z!(d2fdz2::AbstractArray{mk_float,6}, f::AbstractArray{mk_float,6},
        d2fdz2_lower_endpoints::AbstractArray{mk_float,5},
        d2fdz2_upper_endpoints::AbstractArray{mk_float,5},
        z_send_buffer::AbstractArray{mk_float,5},
        z_receive_buffer::AbstractArray{mk_float,5}, z_spectral, z)

    @begin_sn_r_vzeta_vr_vz_region()

    # differentiate f w.r.t z
    @loop_sn_r_vzeta_vr_vz isn ir ivzeta ivr ivz begin
        @views second_derivative!(d2fdz2[ivz,ivr,ivzeta,:,ir,isn], f[ivz,ivr,ivzeta,:,ir,isn], z, z_spectral)
        # get external endpoints to reconcile via MPI
        d2fdz2_lower_endpoints[ivz,ivr,ivzeta,ir,isn] = z.scratch_2d[1,1]
        d2fdz2_upper_endpoints[ivz,ivr,ivzeta,ir,isn] = z.scratch_2d[end,end]
    end
    # now reconcile element boundaries across
    # processes with large message involving all y
    if z.nelement_local < z.nelement_global
        reconcile_element_boundaries_MPI!(d2fdz2, d2fdz2_lower_endpoints,
                                          d2fdz2_upper_endpoints, z_send_buffer,
                                          z_receive_buffer, z)
    end
end

"""
Upwind derivatives
df/dr group of rountines for
fields & moments -> [z,r]
dfns (ion) -> [vpa,vperp,z,r,s]
dfns (neutrals) -> [vz,vr,vzeta,z,r,sn]
"""

#df/dr
#2D version for f[z,r] -> Er, Ez, phi
function derivative_r!(dfdr::AbstractArray{mk_float,2}, f::AbstractArray{mk_float,2},
        speed, speed_lower_buffer::AbstractArray{mk_float,1},
        speed_upper_buffer::AbstractArray{mk_float,1},
        dfdr_lower_endpoints::AbstractArray{mk_float,1},
        dfdr_upper_endpoints::AbstractArray{mk_float,1},
        r_receive_buffer1::AbstractArray{mk_float,1},
        r_receive_buffer2::AbstractArray{mk_float,1}, r_spectral, r)

    @begin_z_region()

    # differentiate f w.r.t r
    @loop_z iz begin
        # Note that for moments, `speed` has its dimensions in the same order as the
        # moment arrays, not with the derivative dimension moved to be left-most index.
        @views derivative!(dfdr[iz,:], f[iz,:], r, speed[iz,:], r_spectral)
        # get external endpoints to reconcile via MPI
        dfdr_lower_endpoints[iz] = r.scratch_2d[1,1]
        dfdr_upper_endpoints[iz] = r.scratch_2d[end,end]
        speed_lower_buffer[iz] = speed[1,iz]
        speed_upper_buffer[iz] = speed[end,iz]
    end
    # now reconcile element boundaries across
    # processes with large message involving all other dimensions
    if r.nelement_local < r.nelement_global
        reconcile_element_boundaries_MPI!(
            dfdr, speed_lower_buffer, speed_upper_buffer, dfdr_lower_endpoints,
            dfdr_upper_endpoints, r_receive_buffer1, r_receive_buffer2, r)
    end
end

#df/dr
#3D version for f[z,r,s] -> moments n, u, T etc
function derivative_r!(dfdr::AbstractArray{mk_float,3}, f::AbstractArray{mk_float,3},
        speed, speed_lower_buffer::AbstractArray{mk_float,2},
        speed_upper_buffer::AbstractArray{mk_float,2},
        dfdr_lower_endpoints::AbstractArray{mk_float,2},
        dfdr_upper_endpoints::AbstractArray{mk_float,2},
        r_receive_buffer1::AbstractArray{mk_float,2},
        r_receive_buffer2::AbstractArray{mk_float,2}, r_spectral, r; neutrals=false)

    # differentiate f w.r.t r
    if neutrals
        @loop_sn_z isn iz begin
            # Note that for moments, `speed` has its dimensions in the same order as the
            # moment arrays, not with the derivative dimension moved to be left-most
            # index.
            @views derivative!(dfdr[iz,:,isn], f[iz,:,isn], r, speed[iz,:,isn], r_spectral)
            # get external endpoints to reconcile via MPI
            dfdr_lower_endpoints[iz,isn] = r.scratch_2d[1,1]
            dfdr_upper_endpoints[iz,isn] = r.scratch_2d[end,end]
            speed_lower_buffer[iz,isn] = speed[iz,1,isn]
            speed_upper_buffer[iz,isn] = speed[iz,end,isn]
        end
    else
        @loop_s_z is iz begin
            # Note that for moments, `speed` has its dimensions in the same order as the
            # moment arrays, not with the derivative dimension moved to be left-most
            # index.
            @views derivative!(dfdr[iz,:,is], f[iz,:,is], r, speed[iz,:,is], r_spectral)
            # get external endpoints to reconcile via MPI
            dfdr_lower_endpoints[iz,is] = r.scratch_2d[1,1]
            dfdr_upper_endpoints[iz,is] = r.scratch_2d[end,end]
            speed_lower_buffer[iz,is] = speed[iz,1,is]
            speed_upper_buffer[iz,is] = speed[iz,end,is]
        end
    end

    # Sometimes an array might contain no data (e.g. if n_neutral_species=0). Then don't
    # need to reconcile boundaries
    if length(dfdr) > 0
        # now reconcile element boundaries across
        # processes with large message involving all other dimensions
        if r.nelement_local < r.nelement_global
            reconcile_element_boundaries_MPI!(
                dfdr, speed_lower_buffer, speed_upper_buffer, dfdr_lower_endpoints,
                dfdr_upper_endpoints, r_receive_buffer1, r_receive_buffer2, r; neutrals)
        end
    end
end

#df/dr
#5D version for f[vpa,vperp,z,r,s] -> ion particle dfn
function derivative_r!(dfdr::AbstractArray{mk_float,5}, f::AbstractArray{mk_float,5},
        advect, speed_lower_buffer::AbstractArray{mk_float,4},
        speed_upper_buffer::AbstractArray{mk_float,4},
        dfdr_lower_endpoints::AbstractArray{mk_float,4},
        dfdr_upper_endpoints::AbstractArray{mk_float,4},
        r_receive_buffer1::AbstractArray{mk_float,4},
        r_receive_buffer2::AbstractArray{mk_float,4}, r_spectral, r)

        @begin_s_z_vperp_vpa_region()

	# differentiate f w.r.t r
	@loop_s_z_vperp_vpa is iz ivperp ivpa begin
		@views derivative!(dfdr[ivpa,ivperp,iz,:,is], f[ivpa,ivperp,iz,:,is], r, advect[:,ivpa,ivperp,iz,is], r_spectral)
		# get external endpoints to reconcile via MPI
		dfdr_lower_endpoints[ivpa,ivperp,iz,is] = r.scratch_2d[1,1]
		dfdr_upper_endpoints[ivpa,ivperp,iz,is] = r.scratch_2d[end,end]
		speed_lower_buffer[ivpa,ivperp,iz,is] = advect[1,ivpa,ivperp,iz,is]
		speed_upper_buffer[ivpa,ivperp,iz,is] = advect[end,ivpa,ivperp,iz,is]
	end
	# now reconcile element boundaries across
	# processes with large message involving all other dimensions
	if r.nelement_local < r.nelement_global
		reconcile_element_boundaries_MPI!(dfdr,
		 speed_lower_buffer, speed_upper_buffer,
		 dfdr_lower_endpoints,dfdr_upper_endpoints,
		 r_receive_buffer1, r_receive_buffer2, r)
	end
	
end

#6D version for f[vz,vz,vzeta,z,r,sn] -> neutral particle dfn (species indexing taken outside this loop)
function derivative_r!(dfdr::AbstractArray{mk_float,6}, f::AbstractArray{mk_float,6},
        advect, speed_lower_buffer::AbstractArray{mk_float,5},
        speed_upper_buffer::AbstractArray{mk_float,5},
        dfdr_lower_endpoints::AbstractArray{mk_float,5},
        dfdr_upper_endpoints::AbstractArray{mk_float,5},
        r_receive_buffer1::AbstractArray{mk_float,5},
        r_receive_buffer2::AbstractArray{mk_float,5}, r_spectral, r)

        @begin_sn_z_vzeta_vr_vz_region()

	# differentiate f w.r.t r
	@loop_sn_z_vzeta_vr_vz isn iz ivzeta ivr ivz begin
		@views derivative!(dfdr[ivz,ivr,ivzeta,iz,:,isn], f[ivz,ivr,ivzeta,iz,:,isn],
                            r, advect[:,ivz,ivr,ivzeta,iz,isn], r_spectral)
		# get external endpoints to reconcile via MPI
		dfdr_lower_endpoints[ivz,ivr,ivzeta,iz,isn] = r.scratch_2d[1,1]
		dfdr_upper_endpoints[ivz,ivr,ivzeta,iz,isn] = r.scratch_2d[end,end]
		speed_lower_buffer[ivz,ivr,ivzeta,iz,isn] = advect[1,ivz,ivr,ivzeta,iz,isn]
		speed_upper_buffer[ivz,ivr,ivzeta,iz,isn] = advect[end,ivz,ivr,ivzeta,iz,isn]
	end
	# now reconcile element boundaries across
	# processes with large message involving all other dimensions
	if r.nelement_local < r.nelement_global
		reconcile_element_boundaries_MPI!(dfdr,
		 speed_lower_buffer, speed_upper_buffer,
		 dfdr_lower_endpoints,dfdr_upper_endpoints,
		 r_receive_buffer1, r_receive_buffer2, r)
	end
	
end

"""
Upwind derivatives
df/dz group of rountines for
fields & moments -> [z,r]
dfns (ion) -> [vpa,vperp,z,r,s]
dfns (neutrals) -> [vz,vr,vzeta,z,r,sn]
"""

#2D version for f[z,r] -> Er, Ez, phi
function derivative_z!(dfdz::AbstractArray{mk_float,2}, f::AbstractArray{mk_float,2},
        speed, speed_lower_buffer::AbstractArray{mk_float,1},
        speed_upper_buffer::AbstractArray{mk_float,1},
        dfdz_lower_endpoints::AbstractArray{mk_float,1},
        dfdz_upper_endpoints::AbstractArray{mk_float,1},
        z_send_buffer::AbstractArray{mk_float,1},
        z_receive_buffer::AbstractArray{mk_float,1}, z_spectral, z)

    @begin_r_region()

    # differentiate f w.r.t z
    @loop_r ir begin
        @views derivative!(dfdz[:,ir], f[:,ir], z, speed[:,ir], z_spectral)
        # get external endpoints to reconcile via MPI
        dfdz_lower_endpoints[ir] = z.scratch_2d[1,1]
        dfdz_upper_endpoints[ir] = z.scratch_2d[end,end]
        speed_lower_buffer[ir] = speed[1,ir]
        speed_upper_buffer[ir] = speed[end,ir]
    end
    # now reconcile element boundaries across
    # processes with large message
    if z.nelement_local < z.nelement_global
        reconcile_element_boundaries_MPI!(
            dfdz, speed_lower_buffer, speed_upper_buffer,
            dfdz_lower_endpoints,dfdz_upper_endpoints, z_send_buffer, z_receive_buffer, z)
    end
end

#3D version for f[z,r] -> moments n, u, T etc
function derivative_z!(dfdz::AbstractArray{mk_float,3}, f::AbstractArray{mk_float,3},
        speed, speed_lower_buffer::AbstractArray{mk_float,2},
        speed_upper_buffer::AbstractArray{mk_float,2},
        dfdz_lower_endpoints::AbstractArray{mk_float,2},
        dfdz_upper_endpoints::AbstractArray{mk_float,2},
        z_send_buffer::AbstractArray{mk_float,2},
        z_receive_buffer::AbstractArray{mk_float,2}, z_spectral, z; neutrals=false)

    # differentiate f w.r.t z
    if neutrals
        @loop_sn_r isn ir begin
            @views derivative!(dfdz[:,ir,isn], f[:,ir,isn], z, speed[:,ir,isn], z_spectral)
            # get external endpoints to reconcile via MPI
            dfdz_lower_endpoints[ir,isn] = z.scratch_2d[1,1]
            dfdz_upper_endpoints[ir,isn] = z.scratch_2d[end,end]
            speed_lower_buffer[ir,isn] = speed[1,ir,isn]
            speed_upper_buffer[ir,isn] = speed[end,ir,isn]
        end
    else
        @loop_s_r is ir begin
            @views derivative!(dfdz[:,ir,is], f[:,ir,is], z, speed[:,ir,is], z_spectral)
            # get external endpoints to reconcile via MPI
            dfdz_lower_endpoints[ir,is] = z.scratch_2d[1,1]
            dfdz_upper_endpoints[ir,is] = z.scratch_2d[end,end]
            speed_lower_buffer[ir,is] = speed[1,ir,is]
            speed_upper_buffer[ir,is] = speed[end,ir,is]
        end
    end

    # Sometimes an array might contain no data (e.g. if n_neutral_species=0). Then don't
    # need to reconcile boundaries
    if length(dfdz) > 0
        # now reconcile element boundaries across
        # processes with large message
        if z.nelement_local < z.nelement_global
            reconcile_element_boundaries_MPI!(
                dfdz, speed_lower_buffer, speed_upper_buffer,
                dfdz_lower_endpoints,dfdz_upper_endpoints, z_send_buffer,
                z_receive_buffer, z; neutrals)
        end
    end
end

# df/dz
# 3D version for f[vpa,vperp,z]. Uses modified function name to avoid clash with 'standard'
# 3D version for ion/neutral moments.
function derivative_z_pdf_vpavperpz!(dfdz::AbstractArray{mk_float,3}, f::AbstractArray{mk_float,3},
        speed, speed_lower_buffer::AbstractArray{mk_float,2},
        speed_upper_buffer::AbstractArray{mk_float,2},
        dfdz_lower_endpoints::AbstractArray{mk_float,2},
        dfdz_upper_endpoints::AbstractArray{mk_float,2},
        z_receive_buffer1::AbstractArray{mk_float,2},
        z_receive_buffer2::AbstractArray{mk_float,2}, z_spectral, z)

    # differentiate f w.r.t z
    @loop_vperp_vpa ivperp ivpa begin
        @views derivative!(dfdz[ivpa,ivperp,:], f[ivpa,ivperp,:], z, speed[:,ivpa,ivperp], z_spectral)
        # get external endpoints to reconcile via MPI
        dfdz_lower_endpoints[ivpa,ivperp] = z.scratch_2d[1,1]
        dfdz_upper_endpoints[ivpa,ivperp] = z.scratch_2d[end,end]
        speed_lower_buffer[ivpa,ivperp] = speed[1,ivpa,ivperp]
        speed_upper_buffer[ivpa,ivperp] = speed[end,ivpa,ivperp]
    end

    # now reconcile element boundaries across
    # processes with large message
    if z.nelement_local < z.nelement_global
        reconcile_element_boundaries_MPI_z_pdf_vpavperpz!(
            dfdz, speed_lower_buffer, speed_upper_buffer, dfdz_lower_endpoints,
            dfdz_upper_endpoints, z_receive_buffer1, z_receive_buffer2, z)
    end
end

#5D version for f[vpa,vperp,z,r,s] -> dfn ion particles
function derivative_z!(dfdz::AbstractArray{mk_float,5}, f::AbstractArray{mk_float,5},
        advect, speed_lower_buffer::AbstractArray{mk_float,4},
        speed_upper_buffer::AbstractArray{mk_float,4},
        dfdz_lower_endpoints::AbstractArray{mk_float,4},
        dfdz_upper_endpoints::AbstractArray{mk_float,4},
        z_send_buffer::AbstractArray{mk_float,4},
        z_receive_buffer::AbstractArray{mk_float,4}, z_spectral, z)

        @begin_s_r_vperp_vpa_region()

	# differentiate f w.r.t z
	@loop_s_r_vperp_vpa is ir ivperp ivpa begin
		@views derivative!(dfdz[ivpa,ivperp,:,ir,is], f[ivpa,ivperp,:,ir,is], z, advect[:,ivpa,ivperp,ir,is], z_spectral)
		# get external endpoints to reconcile via MPI
		dfdz_lower_endpoints[ivpa,ivperp,ir,is] = z.scratch_2d[1,1]
		dfdz_upper_endpoints[ivpa,ivperp,ir,is] = z.scratch_2d[end,end]
		speed_lower_buffer[ivpa,ivperp,ir,is] = advect[1,ivpa,ivperp,ir,is]
		speed_upper_buffer[ivpa,ivperp,ir,is] = advect[end,ivpa,ivperp,ir,is]
	end
	# now reconcile element boundaries across
	# processes with large message
	if z.nelement_local < z.nelement_global
		reconcile_element_boundaries_MPI!(dfdz,
		 speed_lower_buffer, speed_upper_buffer,
		 dfdz_lower_endpoints,dfdz_upper_endpoints,
		 z_send_buffer, z_receive_buffer, z)
	end
	
end

#4D version for f[vpa,vperp,z,r] -> dfn electron particles
function derivative_z!(dfdz::AbstractArray{mk_float,4}, f::AbstractArray{mk_float,4},
	advect, speed_lower_buffer::AbstractArray{mk_float,3},
	speed_upper_buffer::AbstractArray{mk_float,3},
	dfdz_lower_endpoints::AbstractArray{mk_float,3},
	dfdz_upper_endpoints::AbstractArray{mk_float,3},
	z_send_buffer::AbstractArray{mk_float,3},
	z_receive_buffer::AbstractArray{mk_float,3}, z_spectral, z)

    @begin_r_vperp_vpa_region()

    # differentiate the pdf f w.r.t z
    @loop_r_vperp_vpa ir ivperp ivpa begin
            @views derivative!(dfdz[ivpa,ivperp,:,ir], f[ivpa,ivperp,:,ir], z, advect[:,ivpa,ivperp,ir], z_spectral)
            # get external endpoints to reconcile via MPI
            dfdz_lower_endpoints[ivpa,ivperp,ir] = z.scratch_2d[1,1]
            dfdz_upper_endpoints[ivpa,ivperp,ir] = z.scratch_2d[end,end]
            speed_lower_buffer[ivpa,ivperp,ir] = advect[1,ivpa,ivperp,ir]
            speed_upper_buffer[ivpa,ivperp,ir] = advect[end,ivpa,ivperp,ir]
    end
    # now reconcile element boundaries across
    # processes with large message
    if z.nelement_local < z.nelement_global
            reconcile_element_boundaries_MPI!(dfdz,
             speed_lower_buffer, speed_upper_buffer,
             dfdz_lower_endpoints,dfdz_upper_endpoints,
             z_send_buffer, z_receive_buffer, z)
    end

end

#6D version for f[vz,vr,vzeta,z,r,sn] -> dfn neutral particles
function derivative_z!(dfdz::AbstractArray{mk_float,6}, f::AbstractArray{mk_float,6},
        advect, speed_lower_buffer::AbstractArray{mk_float,5},
        speed_upper_buffer::AbstractArray{mk_float,5},
        dfdz_lower_endpoints::AbstractArray{mk_float,5},
        dfdz_upper_endpoints::AbstractArray{mk_float,5},
        z_send_buffer::AbstractArray{mk_float,5},
        z_receive_buffer::AbstractArray{mk_float,5}, z_spectral, z)

        @begin_sn_r_vzeta_vr_vz_region()

	# differentiate f w.r.t z
	@loop_sn_r_vzeta_vr_vz isn ir ivzeta ivr ivz begin
		@views derivative!(dfdz[ivz,ivr,ivzeta,:,ir,isn], f[ivz,ivr,ivzeta,:,ir,isn],
                            z, advect[:,ivz,ivr,ivzeta,ir,isn], z_spectral)
		# get external endpoints to reconcile via MPI
		dfdz_lower_endpoints[ivz,ivr,ivzeta,ir,isn] = z.scratch_2d[1,1]
		dfdz_upper_endpoints[ivz,ivr,ivzeta,ir,isn] = z.scratch_2d[end,end]
		speed_lower_buffer[ivz,ivr,ivzeta,ir,isn] = advect[1,ivz,ivr,ivzeta,ir,isn]
		speed_upper_buffer[ivz,ivr,ivzeta,ir,isn] = advect[end,ivz,ivr,ivzeta,ir,isn]
	end
    # now reconcile element boundaries across
	# processes with large message
	if z.nelement_local < z.nelement_global
		reconcile_element_boundaries_MPI!(dfdz,
		 speed_lower_buffer, speed_upper_buffer,
		 dfdz_lower_endpoints,dfdz_upper_endpoints,
		 z_send_buffer, z_receive_buffer, z)
	end
	
end

end
