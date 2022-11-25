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

using ..calculus: derivative!, reconcile_element_boundaries_centered_MPI!
using ..type_definitions: mk_float
using ..looping

"""
df/dr group of rountines for 
fields -> [z,r]
moments (charged) -> [z,r,s]
moments (neutral) -> [z,r,sn]
dfns (charged) -> [vpa,vperp,z,r,s]
dfns (neutrals) -> [vz,vr,vzeta,z,r,sn]
"""

#df/dr 
#2D version for f[z,r] -> Er, Ez, phi 
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
		reconcile_element_boundaries_centered_MPI!(dfdr,
		 dfdr_lower_endpoints,dfdr_upper_endpoints,
		 r_send_buffer, r_receive_buffer, r)
	end
	
end

#df/dr
#3D version for f[z,r,s] -> n_s, u_s, T_s etc of charged particles
function derivative_r_chrg!(dfdr::Array{mk_float,3},f::Array{mk_float,3},
	dfdr_lower_endpoints::Array{mk_float,2}, dfdr_upper_endpoints::Array{mk_float,2},
	r_send_buffer::Array{mk_float,2},r_receive_buffer::Array{mk_float,2},
	r_spectral,r)

	# differentiate f w.r.t r
	@loop_z_s iz is begin
		@views derivative!(dfdr[iz,:,is], f[iz,:,is], r, r_spectral)
		# get external endpoints to reconcile via MPI
		dfdr_lower_endpoints[iz,is] = r.scratch_2d[1,1]
		dfdr_upper_endpoints[iz,is] = r.scratch_2d[end,end] 
	end
	# now reconcile element boundaries across
	# processes with large message involving all other dimensions
	if r.nelement_local < r.nelement_global
		reconcile_element_boundaries_centered_MPI!(dfdr,
		 dfdr_lower_endpoints,dfdr_upper_endpoints,
		 r_send_buffer, r_receive_buffer, r)
	end
	
end
#3D version for f[z,r,s] -> n_s, u_s, T_s etc of neutral particles
function derivative_r_ntrl!(dfdr::Array{mk_float,3},f::Array{mk_float,3},
	dfdr_lower_endpoints::Array{mk_float,2}, dfdr_upper_endpoints::Array{mk_float,2},
	r_send_buffer::Array{mk_float,2},r_receive_buffer::Array{mk_float,2},
	r_spectral,r)

	# differentiate f w.r.t r
	@loop_z_sn iz isn begin
		@views derivative!(dfdr[iz,:,isn], f[iz,:,isn], r, r_spectral)
		# get external endpoints to reconcile via MPI
		dfdr_lower_endpoints[iz,isn] = r.scratch_2d[1,1]
		dfdr_upper_endpoints[iz,isn] = r.scratch_2d[end,end] 
	end
	# now reconcile element boundaries across
	# processes with large message involving all other dimensions
	if r.nelement_local < r.nelement_global
		reconcile_element_boundaries_centered_MPI!(dfdr,
		 dfdr_lower_endpoints,dfdr_upper_endpoints,
		 r_send_buffer, r_receive_buffer, r)
	end
	
end

#df/dr
#5D version for f[vpa,vperp,z,r,s] -> charged particle dfn
function derivative_r_chrg!(dfdr::Array{mk_float,5},f::Array{mk_float,5},
	dfdr_lower_endpoints::Array{mk_float,4}, dfdr_upper_endpoints::Array{mk_float,4},
	r_send_buffer::Array{mk_float,4},r_receive_buffer::Array{mk_float,4},
	r_spectral,r)

	# differentiate f w.r.t r
	@loop_vpa_vperp_z_s ivpa ivperp iz is begin
		@views derivative!(dfdr[ivpa,ivperp,iz,:,is], f[ivpa,ivperp,iz,:,is], r, r_spectral)
		# get external endpoints to reconcile via MPI
		dfdr_lower_endpoints[ivpa,ivperp,iz,is] = r.scratch_2d[1,1]
		dfdr_upper_endpoints[ivpa,ivperp,iz,is] = r.scratch_2d[end,end] 
	end
	# now reconcile element boundaries across
	# processes with large message involving all other dimensions 
	if r.nelement_local < r.nelement_global
		reconcile_element_boundaries_centered_MPI!(dfdr,
		 dfdr_lower_endpoints,dfdr_upper_endpoints,
		 r_send_buffer, r_receive_buffer, r)
	end
	
end

#6D version for f[vz,vz,vzeta,z,r,sn] -> neutral particle dfn
function derivative_r_ntrl!(dfdr::Array{mk_float,6},f::Array{mk_float,6},
	dfdr_lower_endpoints::Array{mk_float,5}, dfdr_upper_endpoints::Array{mk_float,5},
	r_send_buffer::Array{mk_float,5},r_receive_buffer::Array{mk_float,5},
	r_spectral,r)

	# differentiate f w.r.t r
	@loop_vz_vz_vzeta_z_s ivz ivr ivzeta iz is begin
		@views derivative!(dfdr[ivz,ivr,ivzeta,iz,:,is], f[ivz,ivr,ivzeta,iz,:,is], r, r_spectral)
		# get external endpoints to reconcile via MPI
		dfdr_lower_endpoints[ivz,ivr,ivzeta,iz,is] = r.scratch_2d[1,1]
		dfdr_upper_endpoints[ivz,ivr,ivzeta,iz,is] = r.scratch_2d[end,end] 
	end
	# now reconcile element boundaries across
	# processes with large message involving all other dimensions 
	if r.nelement_local < r.nelement_global
		reconcile_element_boundaries_centered_MPI!(dfdr,
		 dfdr_lower_endpoints,dfdr_upper_endpoints,
		 r_send_buffer, r_receive_buffer, r)
	end
	
end

"""
df/dz group of rountines for 
fields -> [z,r]
moments (charged) -> [z,r,s]
moments (neutral) -> [z,r,sn]
dfns (charged) -> [vpa,vperp,z,r,s]
dfns (neutrals) -> [vz,vr,vzeta,z,r,sn]
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
		# get ezternal endpoints to reconcile via MPI
		dfdz_lower_endpoints[ir] = z.scratch_2d[1,1]
		dfdz_upper_endpoints[ir] = z.scratch_2d[end,end] 
	end
	# now reconcile element boundaries across
	# processes with large message involving all y 
	if z.nelement_local < z.nelement_global
		reconcile_element_boundaries_centered_MPI!(dfdz,
		 dfdz_lower_endpoints,dfdz_upper_endpoints,
		 z_send_buffer, z_receive_buffer, z)
	end
	
end

#3D version for f[z,r,s] -> n_s, u_s, T_s etc of charged particles
function derivative_z_chrg!(dfdz::Array{mk_float,3},f::Array{mk_float,3},
	dfdz_lower_endpoints::Array{mk_float,2}, dfdz_upper_endpoints::Array{mk_float,2},
	z_send_buffer::Array{mk_float,2},z_receive_buffer::Array{mk_float,2},
	z_spectral,z)

	# differentiate f w.r.t z
	@loop_r_s ir is begin
		@views derivative!(dfdz[:,ir,is], f[:,ir,is], z, z_spectral)
		# get ezternal endpoints to reconcile via MPI
		dfdz_lower_endpoints[ir,is] = z.scratch_2d[1,1]
		dfdz_upper_endpoints[ir,is] = z.scratch_2d[end,end] 
	end
	# now reconcile element boundaries across
	# processes with large message involving all y 
	if z.nelement_local < z.nelement_global
		reconcile_element_boundaries_centered_MPI!(dfdz,
		 dfdz_lower_endpoints,dfdz_upper_endpoints,
		 z_send_buffer, z_receive_buffer, z)
	end
	
end

#3D version for f[z,r,s] -> n_s, u_s, T_s etc of neutral particles
function derivative_z_ntrl!(dfdz::Array{mk_float,3},f::Array{mk_float,3},
	dfdz_lower_endpoints::Array{mk_float,2}, dfdz_upper_endpoints::Array{mk_float,2},
	z_send_buffer::Array{mk_float,2},z_receive_buffer::Array{mk_float,2},
	z_spectral,z)

	# differentiate f w.r.t z
	@loop_r_sn ir isn begin
		@views derivative!(dfdz[:,ir,isn], f[:,ir,isn], z, z_spectral)
		# get ezternal endpoints to reconcile via MPI
		dfdz_lower_endpoints[ir,isn] = z.scratch_2d[1,1]
		dfdz_upper_endpoints[ir,isn] = z.scratch_2d[end,end] 
	end
	# now reconcile element boundaries across
	# processes with large message involving all y 
	if z.nelement_local < z.nelement_global
		reconcile_element_boundaries_centered_MPI!(dfdz,
		 dfdz_lower_endpoints,dfdz_upper_endpoints,
		 z_send_buffer, z_receive_buffer, z)
	end
	
end

#5D version for f[vz,vr,vzeta,z,r,s] -> dfn charged particles
function derivative_z_chrg!(dfdz::Array{mk_float,5},f::Array{mk_float,5},
	dfdz_lower_endpoints::Array{mk_float,4}, dfdz_upper_endpoints::Array{mk_float,4},
	z_send_buffer::Array{mk_float,4},z_receive_buffer::Array{mk_float,4},
	z_spectral,z)

	# differentiate f w.r.t z
	@loop_vpa_vperp_r_s ivz ivr ivzeta ir is begin
		@views derivative!(dfdz[ivpa,ivperp,:,ir,is], f[ivpa,ivperp,:,ir,is], z, z_spectral)
		# get ezternal endpoints to reconcile via MPI
		dfdz_lower_endpoints[ivpa,ivperp,ir,is] = z.scratch_2d[1,1]
		dfdz_upper_endpoints[ivpa,ivperp,ir,is] = z.scratch_2d[end,end] 
	end
	# now reconcile element boundaries across
	# processes with large message involving all y 
	if z.nelement_local < z.nelement_global
		reconcile_element_boundaries_centered_MPI!(dfdz,
		 dfdz_lower_endpoints,dfdz_upper_endpoints,
		 z_send_buffer, z_receive_buffer, z)
	end
	
end

#6D version for f[vz,vr,vzeta,z,r,s] -> dfn neutral particles
function derivative_z_ntrl!(dfdz::Array{mk_float,6},f::Array{mk_float,6},
	dfdz_lower_endpoints::Array{mk_float,5}, dfdz_upper_endpoints::Array{mk_float,5},
	z_send_buffer::Array{mk_float,5},z_receive_buffer::Array{mk_float,5},
	z_spectral,z)

	# differentiate f w.r.t z
	@loop_vz_vr_vzeta_r_sn ivz ivr ivzeta ir isn begin
		@views derivative!(dfdz[ivz,ivr,ivzeta,:,ir,isn], f[ivz,ivr,ivzeta,:,ir,isn], z, z_spectral)
		# get ezternal endpoints to reconcile via MPI
		dfdz_lower_endpoints[ivz,ivr,ivzeta,ir,isn] = z.scratch_2d[1,1]
		dfdz_upper_endpoints[ivz,ivr,ivzeta,ir,isn] = z.scratch_2d[end,end] 
	end
	# now reconcile element boundaries across
	# processes with large message involving all y 
	if z.nelement_local < z.nelement_global
		reconcile_element_boundaries_centered_MPI!(dfdz,
		 dfdz_lower_endpoints,dfdz_upper_endpoints,
		 z_send_buffer, z_receive_buffer, z)
	end
	
end

end 