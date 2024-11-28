# -*- coding: utf-8 -*-
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg') # this line allows plots to be made without using a display environment variable
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import h5py
from plot_mk_utils import plot_1d_list_pdf, plot_1d_loglog_list_pdf
from plot_mk_utils import plot_1d_semilog_list_pdf


def get_fkpl_error_data(filename):
    f = h5py.File(filename,'r')
    print(f.keys())
    ncore = np.copy(f['ncore'][...])
    ngrid = np.copy(f['ngrid'][...])
    print("ngrid: ",ngrid)
    nelement_list = np.copy(f['nelement_list'][:])
    print("nelement_list: ",nelement_list[:])
    max_C_err = np.copy(f['max_C_err'][:])
    max_G_err = np.copy(f['max_G_err'][:])
    max_H_err = np.copy(f['max_H_err'][:])
    max_dHdvpa_err = np.copy(f['max_dHdvpa_err'][:])
    max_dHdvperp_err = np.copy(f['max_dHdvperp_err'][:])
    max_d2Gdvperpdvpa_err = np.copy(f['max_d2Gdvperpdvpa_err'][:])
    max_d2Gdvpa2_err = np.copy(f['max_d2Gdvpa2_err'][:])
    max_d2Gdvperp2_err = np.copy(f['max_d2Gdvperp2_err'][:])
    L2_C_err = np.copy(f['L2_C_err'][:])
    L2_G_err = np.copy(f['L2_G_err'][:])
    L2_H_err = np.copy(f['L2_H_err'][:])
    L2_dHdvpa_err = np.copy(f['L2_dHdvpa_err'][:])
    L2_dHdvperp_err = np.copy(f['L2_dHdvperp_err'][:])
    L2_d2Gdvperpdvpa_err = np.copy(f['L2_d2Gdvperpdvpa_err'][:])
    L2_d2Gdvpa2_err = np.copy(f['L2_d2Gdvpa2_err'][:])
    L2_d2Gdvperp2_err = np.copy(f['L2_d2Gdvperp2_err'][:])
    expected_diff = np.copy(f['expected_diff'][:])
    expected_integral = np.copy(f['expected_integral'][:])
    calculate_times = np.copy(f['calculate_times'][:])
    init_times = np.copy(f['init_times'][:])
    expected_t_2 = np.copy(f['expected_t_2'][:])
    expected_t_3 = np.copy(f['expected_t_3'][:])
    n_err = np.copy(f['n_err'][:])
    u_err = np.copy(f['u_err'][:])
    p_err = np.copy(f['p_err'][:])
    print(p_err)
    
    nelement_string = "N_{\\rm EL}" #\\scriptscriptstyle 
    ngrid_string = "N_{\\rm GR}" 
    
    
    file = filename+".plots.pdf"
    pdf = PdfPages(file)
    
    marker_list = ['r--o','b--s','g--.','m--x','c--v','k']
    C_list = [max_C_err,L2_C_err,n_err,u_err,p_err,expected_diff]
    nelements = [nelement_list for item in C_list] 
    ylab_list = ["$\\epsilon_{\\infty}(C[F,F])$", "$\\epsilon_{2}(C[F,F])$",
                 "$|\\Delta n|$",
                 "$|\\Delta u_{||}|$",
                 "$|\\Delta p |$",
                 "$(1/"+nelement_string+")^{"+ngrid_string+"-1}$"]
    plot_1d_loglog_list_pdf (nelements,C_list,marker_list,"$"+ nelement_string+"$", pdf,
      title='',ylab='',xlims=None,ylims=None,aspx=9,aspy=6, xticks = nelement_list, yticks = None,
      markersize=10, legend_title="", use_legend=True,loc_opt='lower left', ylab_list = ylab_list,
      bbox_to_anchor_opt=(0.05, 0.05), legend_fontsize=15, ncol_opt=1)
      
    marker_list = ['r--o','b--s','m-o','k-s']
    time_list = [init_times, calculate_times, expected_t_3, expected_t_2]
    nelements = [nelement_list for item in time_list] 
    ylab_list = ["time/init (ms)", "time/step (ms)",
                 "$"+nelement_string+"^3$",
                 "$"+nelement_string+"^2$"]
    plot_1d_loglog_list_pdf (nelements,time_list,marker_list,"$"+ nelement_string+"$", pdf,
      title='',ylab='',xlims=None,ylims=None,aspx=9,aspy=6, xticks = nelement_list, yticks = None,
      markersize=10, legend_title="", use_legend=True,loc_opt='upper left', ylab_list = ylab_list,
      bbox_to_anchor_opt=(0.05, 0.95), legend_fontsize=15, ncol_opt=1)
    
    
    marker_list = ['r--o','b--^','g--.','m--x','c--v','k']
    Infnorm_list = [max_dHdvpa_err,
    max_dHdvperp_err,max_d2Gdvperpdvpa_err,max_d2Gdvpa2_err,max_d2Gdvperp2_err, 
    #expected_diff, 
    expected_integral]
    nelements = [nelement_list for item in Infnorm_list] 
    ylab_list = ["$\\epsilon_{\\infty}(d H / d v_{||})$","$\\epsilon_{\\infty}(d H / d v_{\\perp})$",
                 "$\\epsilon_{\\infty}(d^2 G / d v_{\\perp} d v_{||})$", 
                 "$\\epsilon_{\\infty}(d^2 G / d v^2_{||})$", 
                 "$\\epsilon_{\\infty}(d^2 G / d v^2_{\\perp})$",
                 #"$(1/"+nelement_string+")^{"+ngrid_string+"-1}$",
                 "$(1/"+nelement_string+")^{"+ngrid_string+"+1}$"]
    plot_1d_loglog_list_pdf (nelements,Infnorm_list,marker_list,"$"+ nelement_string+"$", pdf,
      title='',ylab='',xlims=None,ylims=None,aspx=9,aspy=6, xticks = nelement_list, yticks = None,
      markersize=10, legend_title="", use_legend=True,loc_opt='lower left', ylab_list = ylab_list,
      bbox_to_anchor_opt=(0.05, 0.05), legend_fontsize=15, ncol_opt=1)
      
    L2norm_list = [L2_dHdvpa_err,
    L2_dHdvperp_err,L2_d2Gdvperpdvpa_err,L2_d2Gdvpa2_err,L2_d2Gdvperp2_err, 
    #expected_diff, 
    expected_integral]
    nelements = [nelement_list for item in L2norm_list] 
    ylab_list = ["$\\epsilon_{2}(d H / d v_{||})$","$\\epsilon_{2}(d H / d v_{\\perp})$",
                 "$\\epsilon_{2}(d^2 G / d v_{\\perp} d v_{||})$", 
                 "$\\epsilon_{2}(d^2 G / d v^2_{||})$", 
                 "$\\epsilon_{2}(d^2 G / d v^2_{\\perp})$",
                 #"$(1/"+nelement_string+")^{"+ngrid_string+"-1}$",
                 "$(1/"+nelement_string+")^{"+ngrid_string+"+1}$"]
    plot_1d_loglog_list_pdf (nelements,L2norm_list,marker_list,"$"+ nelement_string+"$", pdf,
      title='',ylab='',xlims=None,ylims=None,aspx=9,aspy=6, xticks = nelement_list, yticks = None,
      markersize=10, legend_title="", use_legend=True,loc_opt='upper right', ylab_list = ylab_list,
      bbox_to_anchor_opt=(0.95, 0.95), legend_fontsize=15, ncol_opt=1)
    pdf.close()
    print(file)
    
    f.close()
    return None

workdir = ""    
filename = workdir + "moment_kinetics_collisions/fkpl_error_data_ngrid_9_ncore_2.h5"
get_fkpl_error_data(filename)
