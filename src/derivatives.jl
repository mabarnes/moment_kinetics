"""
This module contains all the necessary derivatives needed to carry out 
distributed memory differential operations on the arrays in moment kinetics.
We provide separate derivative functions for each (i) distributed dimension
and(ii) array shape. We do not need to provide derivatives for non-distributed
dimensions as these can by handled by the derivative! function from calculus.jl 

Currently only centered differences are supported
"""
module derivatives

export derivative_r!, derivative_r_chrg!, derivative_r_ntrl!
export derivative_z!, derivative_z_chrg!, derivative_z_ntrl!

using ..calculus: derivative!, reconcile_element_boundaries_MPI!
using ..type_definitions: mk_float
using ..looping

"""
Centered derivatives
df/dr group of rountines for 
fields & moments -> [z,r]
dfns (charged) -> [vpa,vperp,z,r]
dfns (neutrals) -> [vz,vr,vzeta,z,r]
"""

#df/dr 
#2D version for f[z,r] -> Er, Ez, phi, & moments n, u, T etc (species indexing taken outside this loop)
function derivative_r!(dfdr::Array{mk_float,2},f::Array{mk_float,2},
	dfdr_lower_endpoints::Array{mk_float,1}, dfdr_upper_endpoints::Array{mk_float,1},
	r_send_buffer::Array{mk_float,1},r_receive_buffer::Array{mk_float,1},
	r_spectral,r)

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
		 r_send_buffer, r_receive_buffer, r)
	end
	
end

#df/dr
#5D version for f[vpa,vperp,z,r,s] -> charged particle dfn (species indexing taken outside this loop)
function derivative_r!(dfdr::Array{mk_float,5},f::Array{mk_float,5},
	dfdr_lower_endpoints::Array{mk_float,4}, dfdr_upper_endpoints::Array{mk_float,4},
	r_send_buffer::Array{mk_float,4},r_receive_buffer::Array{mk_float,4},
	r_spectral,r)

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
		 r_send_buffer, r_receive_buffer, r)
	end
	
end

#6D version for f[vz,vz,vzeta,z,r,sn] -> neutral particle dfn (species indexing taken outside this loop)
function derivative_r!(dfdr::Array{mk_float,6},f::Array{mk_float,6},
	dfdr_lower_endpoints::Array{mk_float,5}, dfdr_upper_endpoints::Array{mk_float,5},
	r_send_buffer::Array{mk_float,5},r_receive_buffer::Array{mk_float,5},
	r_spectral,r)

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
		 r_send_buffer, r_receive_buffer, r)
	end
	
end

"""
Centered derivatives
df/dz group of rountines for 
fields & moments -> [z,r]
dfns (charged) -> [vpa,vperp,z,r]
dfns (neutrals) -> [vz,vr,vzeta,z,r]
"""

#df/dz
#2D version for f[z,r] -> Er, Ez, phi
function derivative_z!(dfdz::Array{mk_float,2},f::Array{mk_float,2},
	dfdz_lower_endpoints::Array{mk_float,1}, dfdz_upper_endpoints::Array{mk_float,1},
	z_send_buffer::Array{mk_float,1},z_receive_buffer::Array{mk_float,1},
	z_spectral,z)

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

#5D version for f[vpa,vperp,z,r,s] -> dfn charged particles
function derivative_z!(dfdz::Array{mk_float,5},f::Array{mk_float,5},
	dfdz_lower_endpoints::Array{mk_float,4}, dfdz_upper_endpoints::Array{mk_float,4},
	z_send_buffer::Array{mk_float,4},z_receive_buffer::Array{mk_float,4},
	z_spectral,z)

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

#6D version for f[vz,vr,vzeta,z,r] -> dfn neutral particles
function derivative_z!(dfdz::Array{mk_float,6},f::Array{mk_float,6},
	dfdz_lower_endpoints::Array{mk_float,5}, dfdz_upper_endpoints::Array{mk_float,5},
	z_send_buffer::Array{mk_float,5},z_receive_buffer::Array{mk_float,5},
	z_spectral,z)

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
Upwind derivatives 
df/dr group of rountines for 
dfns (charged) -> [vpa,vperp,z,r]
dfns (neutrals) -> [vz,vr,vzeta,z,r]
"""

#df/dr
#5D version for f[vpa,vperp,z,r,s] -> charged particle dfn (species indexing taken outside this loop)
function derivative_r!(dfdr::Array{mk_float,5},f::Array{mk_float,5}, advect::T,
	adv_fac_lower_buffer::Array{Float64,4},adv_fac_upper_buffer::Array{Float64,4},
	dfdr_lower_endpoints::Array{mk_float,4}, dfdr_upper_endpoints::Array{mk_float,4},
	r_send_buffer::Array{mk_float,4},r_receive_buffer::Array{mk_float,4},
	r_spectral,r) where T

	# differentiate f w.r.t r
	@loop_s_z_vperp_vpa is iz ivperp ivpa begin
		@views derivative!(dfdr[ivpa,ivperp,iz,:,is], f[ivpa,ivperp,iz,:,is], r, advect[is].adv_fac[:,ivpa,ivperp,iz], r_spectral)
		# get external endpoints to reconcile via MPI
		dfdr_lower_endpoints[ivpa,ivperp,iz,is] = r.scratch_2d[1,1]
		dfdr_upper_endpoints[ivpa,ivperp,iz,is] = r.scratch_2d[end,end]
		adv_fac_lower_buffer[ivpa,ivperp,iz,is] = advect[is].adv_fac[1,ivpa,ivperp,iz]
		adv_fac_upper_buffer[ivpa,ivperp,iz,is] = advect[is].adv_fac[end,ivpa,ivperp,iz]
	end
	# now reconcile element boundaries across
	# processes with large message involving all other dimensions 
	if r.nelement_local < r.nelement_global
		reconcile_element_boundaries_MPI!(dfdr,
		 adv_fac_lower_buffer, adv_fac_upper_buffer,
		 dfdr_lower_endpoints,dfdr_upper_endpoints,
		 r_send_buffer, r_receive_buffer, r)
	end
	
end

#6D version for f[vz,vz,vzeta,z,r,sn] -> neutral particle dfn (species indexing taken outside this loop)
function derivative_r!(dfdr::Array{mk_float,6},f::Array{mk_float,6}, advect::T,
	adv_fac_lower_buffer::Array{Float64,5},adv_fac_upper_buffer::Array{Float64,5},
	dfdr_lower_endpoints::Array{mk_float,5}, dfdr_upper_endpoints::Array{mk_float,5},
	r_send_buffer::Array{mk_float,5},r_receive_buffer::Array{mk_float,5},
	r_spectral,r) where T

	# differentiate f w.r.t r
	@loop_sn_z_vzeta_vr_vz isn iz ivzeta ivr ivz begin
		@views derivative!(dfdr[ivz,ivr,ivzeta,iz,:,isn], f[ivz,ivr,ivzeta,iz,:,isn],
                            r, advect[isn].adv_fac[:,ivz,ivr,ivzeta,iz], r_spectral)
		# get external endpoints to reconcile via MPI
		dfdr_lower_endpoints[ivz,ivr,ivzeta,iz,isn] = r.scratch_2d[1,1]
		dfdr_upper_endpoints[ivz,ivr,ivzeta,iz,isn] = r.scratch_2d[end,end]
		adv_fac_lower_buffer[ivz,ivr,ivzeta,iz,isn] = advect[isn].adv_fac[1,ivz,ivr,ivzeta,iz]
		adv_fac_upper_buffer[ivz,ivr,ivzeta,iz,isn] = advect[isn].adv_fac[end,ivz,ivr,ivzeta,iz]
	end
	# now reconcile element boundaries across
	# processes with large message involving all other dimensions 
	if r.nelement_local < r.nelement_global
		reconcile_element_boundaries_MPI!(dfdr,
		 adv_fac_lower_buffer, adv_fac_upper_buffer,
		 dfdr_lower_endpoints,dfdr_upper_endpoints,
		 r_send_buffer, r_receive_buffer, r)
	end
	
end

"""
Upwind derivatives
df/dz group of rountines for 
dfns (charged) -> [vpa,vperp,z,r]
dfns (neutrals) -> [vz,vr,vzeta,z,r]
"""

#5D version for f[vpa,vperp,z,r,s] -> dfn charged particles
function derivative_z!(dfdz::Array{mk_float,5},f::Array{mk_float,5}, advect::T,
	adv_fac_lower_buffer::Array{Float64,4},adv_fac_upper_buffer::Array{Float64,4},
	dfdz_lower_endpoints::Array{mk_float,4}, dfdz_upper_endpoints::Array{mk_float,4},
	z_send_buffer::Array{mk_float,4},z_receive_buffer::Array{mk_float,4},
	z_spectral,z) where T

	# differentiate f w.r.t z
	@loop_s_r_vperp_vpa is ir ivperp ivpa begin
		@views derivative!(dfdz[ivpa,ivperp,:,ir,is], f[ivpa,ivperp,:,ir,is], z, advect[is].adv_fac[:,ivpa,ivperp,ir], z_spectral)
		# get external endpoints to reconcile via MPI
		dfdz_lower_endpoints[ivpa,ivperp,ir,is] = z.scratch_2d[1,1]
		dfdz_upper_endpoints[ivpa,ivperp,ir,is] = z.scratch_2d[end,end]
		adv_fac_lower_buffer[ivpa,ivperp,ir,is] = advect[is].adv_fac[1,ivpa,ivperp,ir]
		adv_fac_upper_buffer[ivpa,ivperp,ir,is] = advect[is].adv_fac[end,ivpa,ivperp,ir]
	end
	# now reconcile element boundaries across
	# processes with large message
	if z.nelement_local < z.nelement_global
		reconcile_element_boundaries_MPI!(dfdz,
		 adv_fac_lower_buffer, adv_fac_upper_buffer,
		 dfdz_lower_endpoints,dfdz_upper_endpoints,
		 z_send_buffer, z_receive_buffer, z)
	end
	
end

#6D version for f[vz,vr,vzeta,z,r,sn] -> dfn neutral particles
function derivative_z!(dfdz::Array{mk_float,6},f::Array{mk_float,6}, advect::T,
	adv_fac_lower_buffer::Array{Float64,5},adv_fac_upper_buffer::Array{Float64,5},
	dfdz_lower_endpoints::Array{mk_float,5}, dfdz_upper_endpoints::Array{mk_float,5},
	z_send_buffer::Array{mk_float,5},z_receive_buffer::Array{mk_float,5},
	z_spectral,z) where T

	# differentiate f w.r.t z
	@loop_sn_r_vzeta_vr_vz isn ir ivzeta ivr ivz begin
		@views derivative!(dfdz[ivz,ivr,ivzeta,:,ir,isn], f[ivz,ivr,ivzeta,:,ir,isn],
                            z, advect[isn].adv_fac[:,ivz,ivr,ivzeta,ir], z_spectral)
		# get external endpoints to reconcile via MPI
		dfdz_lower_endpoints[ivz,ivr,ivzeta,ir,isn] = z.scratch_2d[1,1]
		dfdz_upper_endpoints[ivz,ivr,ivzeta,ir,isn] = z.scratch_2d[end,end]
		adv_fac_lower_buffer[ivz,ivr,ivzeta,ir,isn] = advect[isn].adv_fac[1,ivz,ivr,ivzeta,ir]
		adv_fac_upper_buffer[ivz,ivr,ivzeta,ir,isn] = advect[isn].adv_fac[end,ivz,ivr,ivzeta,ir]
	end
    # now reconcile element boundaries across
	# processes with large message
	if z.nelement_local < z.nelement_global
		reconcile_element_boundaries_MPI!(dfdz,
		 adv_fac_lower_buffer, adv_fac_upper_buffer,
		 dfdz_lower_endpoints,dfdz_upper_endpoints,
		 z_send_buffer, z_receive_buffer, z)
	end
	
end

end 