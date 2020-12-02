# add the current directory to the path where the code looks for external modules
push!(LOAD_PATH, ".")

using NCDatasets
using Plots

# get the run_name from the command-line
run_name = ARGS[1]

# create the netcdf filename from the given run_name
filename = string(run_name, ".cdf")

print("Opening ", filename, " to read NetCDF data...")
# open the netcdf file with given filename for reading
fid = NCDataset(filename,"r")
println("done.")

print("Loading coordinate data...")
# define a handle for the z coordinate
cdfvar = fid["z"]
# get the number of z grid points
nz = length(cdfvar)
# load the data for z
z = cdfvar.var[:]

# define a handle for the vpa coordinate
cdfvar = fid["vpa"]
# get the number of vpa grid points
nvpa = length(cdfvar)
# load the data for vpa
vpa = cdfvar.var[:]

# define a handle for the time coordinate
cdfvar = fid["time"]
# get the number of time grid points
ntime = length(cdfvar)
# load the data for time
time = cdfvar.var[:]
println("done.")

print("Loading fields data...")
# define a handle for the electrostatic potential
cdfvar = fid["phi"]
# load the electrostatic potential data
phi = cdfvar.var[:,:]
println("done.")

println("Plotting fields data...")
# plot the time trace of phi(z=z0)
i=cld(nz,2)
#plot(time, log.(phi[i,:]), yscale = :log10)
plot(time, phi[i,:])
outfile = string(run_name, "_phi0_vs_t.pdf")
savefig(outfile)
# make a heatmap plot of ϕ(z,t)
heatmap(time, z, phi, xlabel="time", ylabel="z", title="ϕ", c = :deep)
outfile = string(run_name, "_phi_vs_z_t.pdf")
savefig(outfile)
# make a gif animation of ϕ(z,t)
anim = @animate for i ∈ 1:ntime
    plot(z, phi[:,i], xlabel="z", ylabel="ϕ")
end
outfile = string(run_name, "_phi_vs_z.gif")
gif(anim, outfile, fps=5)

print("Loading distribution function data...")
# define a handle for the distribution function
cdfvar = fid["f"]
# load the distribution function data
ff = cdfvar.var[:,:,:]
println("done.")

println("Plotting distribution function data...")
cmlog(cmlin::ColorGradient) = RGB[cmlin[x] for x=LinRange(0,1,30)]
logdeep = cgrad(:deep, scale=:log) |> cmlog
# make a gif animation of f(vpa,z,t)
fmin = minimum(ff)
fmax = maximum(ff)
anim = @animate for i ∈ 1:ntime
    #heatmap(vpa, z, log.(abs.(ff[:,:,i])), xlabel="vpa", ylabel="z", clims = (fmin,fmax), c = :deep)
    heatmap(vpa, z, log.(abs.(ff[:,:,i])), xlabel="vpa", ylabel="z", fillcolor = logdeep)
end
outfile = string(run_name, "_f_vs_z_vpa.gif")
gif(anim, outfile, fps=5)
# make a gif animation of f(vpa0,z,t)
ivpa=cld(nvpa,2)
anim = @animate for i ∈ 1:ntime
    plot(z, ff[:,ivpa,i], ylims = (fmin,fmax))
end
outfile = string(run_name, "_f_vs_z.gif")
gif(anim, outfile, fps=5)

close(fid)
