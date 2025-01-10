import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.rc('font', size=20)
rcParams.update({'text.latex.preamble' : r'\usepackage{bm}'})
rcParams.update({'figure.autolayout': True})

def plot_1d_list_pdf (xlist,ylist,marker_list,xlab, pdf,
  title='',ylab='',xlims=None,ylims=None,aspx=12,aspy=8, xticks = None, xticks_labels=None, yticks = None,
  markersize=5, legend_title="", use_legend=False,loc_opt='upper right', ylab_list = None,
  bbox_to_anchor_opt=(0.95, 0.95), legend_fontsize=10, ncol_opt=1,
  legend_shadow=False,legend_frame=False, vlines = None,hlines = None,marker_fill_style = None,
  cartoon=False, linewidth=None, texts = None, slines=None):

    fig=plt.figure(figsize=(aspx,aspy))
    nlist = len(ylist)
    if(ylab_list is None):
        ylab_list = [None for i in range(0,nlist)]
    for iy in range(0,nlist):
        plt.plot(xlist[iy],ylist[iy],marker_list[iy],markersize=markersize,label=ylab_list[iy],
        fillstyle = marker_fill_style, linewidth = linewidth)
    plt.xlabel(xlab)
    if len(ylab) > 0:
        plt.ylabel(ylab)
    if len(title) > 0:
        plt.title(title)
    if(not xlims is None):
        plt.xlim(xlims[0],xlims[1])
    if(not ylims is None):
        plt.ylim(ylims[0],ylims[1])
    if(not vlines is None):
        for xin,xlabel,xcolor,xlinestyle in vlines:
            plt.axvline(x=xin, label=xlabel, color=xcolor,linestyle=xlinestyle,linewidth=linewidth)   
    if(not hlines is None):
        for yin,ylabel,ycolor,ylinestyle in hlines:
            plt.axhline(y=yin, label=ylabel, color=ycolor,linestyle=ylinestyle,linewidth=linewidth)   
    if(not texts is None):
        for xin, yin, textin in texts:    
            print(xin,yin,textin)
            plt.text(xin,yin,textin)
    if (not slines is None):
        for m,c,marker,label in slines:
            plt.plot(xlist[0],m*xlist[0]+c,marker,label=label)
    if(use_legend):
        plt.legend(title=legend_title,loc=loc_opt, bbox_to_anchor=bbox_to_anchor_opt,
        fontsize=legend_fontsize, frameon=legend_frame, handlelength=1, labelspacing=0.5,
        ncol=ncol_opt, columnspacing = 0.5 , handletextpad = 0.5, shadow=legend_shadow)
    if(not xticks is None):
        plt.xticks(xticks)
    if(not yticks is None):
        plt.yticks(yticks)    
    if (cartoon):
        plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
        plt.box(False)
    pdf.savefig(fig)# pdf is the object of the current open PDF file to which the figures are appended
    plt.close (fig)
    return

def plot_1d_semilog_list_pdf (xlist,ylist,marker_list,xlab, pdf,
  title='',ylab='',xlims=None,ylims=None,aspx=12,aspy=8, xticks = None, yticks = None,
  markersize=5, legend_title="", use_legend=False,loc_opt='upper right', ylab_list = None,
  bbox_to_anchor_opt=(0.95, 0.95), legend_fontsize=10, ncol_opt=1,
  legend_shadow=False,legend_frame=False):

    fig=plt.figure(figsize=(aspx,aspy))
    nlist = len(ylist)
    if(ylab_list is None):
        ylab_list = [None for i in range(0,nlist)]
    for iy in range(0,nlist):
        plt.semilogy(xlist[iy],ylist[iy],marker_list[iy],markersize=markersize,label=ylab_list[iy])
    plt.xlabel(xlab)
    if len(ylab) > 0:
        plt.ylabel(ylab)
    if len(title) > 0:
        plt.title(title)
    if(not xlims is None):
        plt.xlim(xlims[0],xlims[1])
    if(not ylims is None):
        plt.ylim(ylims[0],ylims[1])
    if(use_legend):
        plt.legend(title=legend_title,loc=loc_opt, bbox_to_anchor=bbox_to_anchor_opt,
        fontsize=legend_fontsize, frameon=legend_frame, handlelength=1, labelspacing=0.5,
        ncol=ncol_opt, columnspacing = 0.5 , handletextpad = 0.5, shadow=legend_shadow)
    if(not xticks is None):
        plt.xticks(xticks)
    if(not yticks is None):
        plt.yticks(yticks)    
    pdf.savefig(fig)# pdf is the object of the current open PDF file to which the figures are appended
    plt.close (fig)
    return

def plot_1d_loglog_list_pdf (xlist,ylist,marker_list,xlab, pdf,
  title='',ylab='',xlims=None,ylims=None,aspx=12,aspy=8, xticks = None, yticks = None,
  markersize=5, legend_title="", use_legend=False,loc_opt='upper right', ylab_list = None,
  bbox_to_anchor_opt=(0.95, 0.95), legend_fontsize=10, ncol_opt=1,
  legend_shadow=False,legend_frame=False):

    fig=plt.figure(figsize=(aspx,aspy))
    nlist = len(ylist)
    if(ylab_list is None):
        ylab_list = [None for i in range(0,nlist)]
    for iy in range(0,nlist):
        plt.loglog(xlist[iy],ylist[iy],marker_list[iy],markersize=markersize,label=ylab_list[iy])
    plt.xlabel(xlab)
    if len(ylab) > 0:
        plt.ylabel(ylab)
    if len(title) > 0:
        plt.title(title)
    if(not xlims is None):
        plt.xlim(xlims[0],xlims[1])
    if(not ylims is None):
        plt.ylim(ylims[0],ylims[1])
    if(use_legend):
        plt.legend(title=legend_title,loc=loc_opt, bbox_to_anchor=bbox_to_anchor_opt,
        fontsize=legend_fontsize, frameon=legend_frame, handlelength=1, labelspacing=0.5,
        ncol=ncol_opt, columnspacing = 0.5 , handletextpad = 0.5, shadow=legend_shadow)
    if(not xticks is None):
        plt.xticks([])
        plt.minorticks_off()
        #print(plt.xticks())
        plt.xticks(xticks,[str(tick) for tick in xticks])
        #print(plt.xticks())
        
    if(not yticks is None):
        plt.yticks(yticks)    
    pdf.savefig(fig)# pdf is the object of the current open PDF file to which the figures are appended
    plt.close (fig)
    return

def plot_2d_pdf(x,y,z,pdf,title="",ylab="",xlab=""):

    # make data
    X, Y = np.meshgrid(x, y)
    levels = np.linspace(z.min(), z.max(), 7)

    # plot
    fig = plt.figure()

    plt.contourf(X, Y, z, levels=levels)
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    pdf.savefig(fig)
    plt.close(fig)
    return None
