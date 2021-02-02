module post_processing_input

export load_post_processing_options
export pp

using type_definitions: mk_int

struct pp_input
    # if calculate_frequencies = true, calculate and print the frequency and growth/decay
    # rate of phi, using values at iz = iz0
    calculate_frequencies::Bool
    # if plot_phi0_vs_t = true, create plot of phi(z0) vs time
    plot_phi0_vs_t::Bool
    # if plot_phi_vs_z_t = true, create plot of phi vs z and time
    plot_phi_vs_z_t::Bool
    # if animate_phi_vs_z = true, create animation of phi vs z at different time slices
    animate_phi_vs_z::Bool
    # if plot_dens0_vs_t = true, create plot of ion_density(z0) vs time
    plot_dens0_vs_t::Bool
    # if plot_dens_vs_z_t = true, create plot of ion density vs z and time
    plot_dens_vs_z_t::Bool
    # if animate_dens_vs_z = true, create animation of ion density vs z at different time slices
    animate_dens_vs_z::Bool
    # if animate_f_vs_z_vpa = true, create animation of f(z,vpa) at different time slices
    animate_f_vs_z_vpa::Bool
    # if animate_f_vs_z_vpa0 = true, create animation of f(z,vpa0) at different time slices
    animate_f_vs_z_vpa0::Bool
    # if animate_f_vs_z0_vpa = true, create animation of f(z0,vpa) at different time slices
    animate_f_vs_z0_vpa::Bool
    # animations will use one in every nwrite_movie data slices
    nwrite_movie::mk_int
    # itime_min is the minimum time index at which to start animations
    itime_min::mk_int
    # itime_max is the final time index at which to end animations
    # if itime_max < 0, the value used will be the total number of time slices
    itime_max::mk_int
    # iz0 is the iz index used when plotting data at a single z location
    iz0::mk_int
    # ivpa0 is the ivpa index used when plotting data at a single vpa location
    ivpa0::mk_int
end

# if calculate_frequencies = true, calculate and print the frequency and growth/decay
# rate of phi, using values at iz = iz0
const calculate_frequencies = true
# if plot_phi0_vs_t = true, create plot of phi(z0) vs time
const plot_phi0_vs_t = true
# if plot_phi_vs_z_t = true, create heatmap of phi vs z and time
const plot_phi_vs_z_t = true
# if animate_phi_vs_z = true, create animation of phi(z) at different time slices
const animate_phi_vs_z = true
# if plot_dens0_vs_t = true, create plot of ion_density(z0) vs time
const plot_dens0_vs_t = true
# if plot_dens_vs_z_t = true, create heatmap of ion density vs z and time
const plot_dens_vs_z_t = true
# if animate_dens_vs_z = true, create animation of ion_density(z) at different time slices
const animate_dens_vs_z = true
# if animate_f_vs_z_vpa = true, create animation of f(z,vpa) at different time slices
const animate_f_vs_z_vpa = true
# if animate_f_vs_z_vpa0 = true, create animation of f(z,vpa0) at different time slices
const animate_f_vs_z_vpa0 = true
# if animate_f_vs_z0_vpa = true, create animation of f(z0,vpa) at different time slices
const animate_f_vs_z0_vpa = true
# animations will use one in every nwrite_movie data slices
const nwrite_movie = 5
# itime_min is the minimum time index at which to start animations
const itime_min = -1
# itime_max is the final time index at which to end animations
# if itime_max < 0, the value used will be the total number of time slices
const itime_max = -1
# iz0 is the iz index used when plotting data at a single z location
# by default, it will be set to cld(nz,2) unless a non-negative value provided here
const iz0 = -1
# ivpa0 is the ivpa index used when plotting data at a single vpa location
# by default, it will be set to cld(nz,2) unless a non-negative value provided here
const ivpa0 = -1

pp = pp_input(calculate_frequencies, plot_phi0_vs_t, plot_phi_vs_z_t,
    animate_phi_vs_z, plot_dens0_vs_t, plot_dens_vs_z_t, animate_dens_vs_z,
    animate_f_vs_z_vpa, animate_f_vs_z_vpa0, animate_f_vs_z0_vpa,
    nwrite_movie, itime_min, itime_max, iz0, ivpa0)

end
