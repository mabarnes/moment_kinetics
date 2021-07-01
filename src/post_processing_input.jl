# if calculate_frequencies = true, calculate and print the frequency and growth/decay
# rate of phi, using values at iz = iz0
const calculate_frequencies = true
# if plot_phi0_vs_t = true, create plot of phi(z0) vs time
const plot_phi0_vs_t = true
# if plot_phi_vs_z_t = true, create heatmap of phi vs z and time
const plot_phi_vs_z_t = true
# if animate_phi_vs_z = true, create animation of phi(z) at different time slices
const animate_phi_vs_z = false
# if plot_dens0_vs_t = true, create plots of species density(z0) vs time
const plot_dens0_vs_t = true
# if plot_upar0_vs_t = true, create plots of species upar(z0) vs time
const plot_upar0_vs_t = true
# if plot_ppar0_vs_t = true, create plots of species ppar(z0) vs time
const plot_ppar0_vs_t = true
# if plot_qpar0_vs_t = true, create plots of species qpar(z0) vs time
const plot_qpar0_vs_t = true
# if plot_dens_vs_z_t = true, create heatmap of species density vs z and time
const plot_dens_vs_z_t = false
# if plot_upar_vs_z_t = true, create heatmap of species parallel flow vs z and time
const plot_upar_vs_z_t = false
# if plot_ppar_vs_z_t = true, create heatmap of species parallel pressure vs z and time
const plot_ppar_vs_z_t = false
# if plot_qpar_vs_z_t = true, create heatmap of species parallel heat flux vs z and time
const plot_qpar_vs_z_t = false
# if animate_dens_vs_z = true, create animation of species density(z) at different time slices
const animate_dens_vs_z = false
# if animate_upar_vs_z = true, create animation of species parallel flow(z) at different time slices
const animate_upar_vs_z = false
# if animate_f_vs_z_vpa = true, create animation of f(z,vpa) at different time slices
const animate_f_vs_z_vpa = false
# if animate_deltaf_vs_z_vpa = true, create animation of δf(z,vpa) at different time slices
const animate_deltaf_vs_z_vpa = false
# if animate_f_vs_z_vpa0 = true, create animation of f(z,vpa0) at different time slices
const animate_f_vs_z_vpa0 = false
# if animate_deltaf_vs_z_vpa0 = true, create animation of δf(z,vpa0) at different time slices
const animate_deltaf_vs_z_vpa0 = false
# if animate_f_vs_z0_vpa = true, create animation of f(z0,vpa) at different time slices
const animate_f_vs_z0_vpa = false
# if animate_deltaf_vs_z0_vpa = true, create animation of δf(z0,vpa) at different time slices
const animate_deltaf_vs_z0_vpa = false
# animations will use one in every nwrite_movie data slices
const nwrite_movie = 10
# itime_min is the minimum time index at which to start animations
const itime_min = 100
# itime_max is the final time index at which to end animations
# if itime_max < 0, the value used will be the total number of time slices
const itime_max = 200
# iz0 is the iz index used when plotting data at a single z location
# by default, it will be set to cld(nz,2) unless a non-negative value provided here
const iz0 = -1
# ivpa0 is the ivpa index used when plotting data at a single vpa location
# by default, it will be set to cld(nz,2) unless a non-negative value provided here
const ivpa0 = -1

pp = pp_input(calculate_frequencies, plot_phi0_vs_t, plot_phi_vs_z_t,
    animate_phi_vs_z, plot_dens0_vs_t, plot_upar0_vs_t, plot_ppar0_vs_t, plot_qpar0_vs_t,
    plot_dens_vs_z_t, plot_upar_vs_z_t, plot_ppar_vs_z_t, plot_qpar_vs_z_t,
    animate_dens_vs_z, animate_upar_vs_z,
    animate_f_vs_z_vpa, animate_f_vs_z_vpa0, animate_f_vs_z0_vpa,
    animate_deltaf_vs_z_vpa, animate_deltaf_vs_z_vpa0, animate_deltaf_vs_z0_vpa,
    nwrite_movie, itime_min, itime_max, iz0, ivpa0)
