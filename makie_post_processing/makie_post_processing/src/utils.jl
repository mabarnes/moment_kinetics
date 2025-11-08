# Utility functions
###################
#
# These are more-or-less generic, but only used in this module for now, so keep them here.

"""
    clear_Dict!(d::AbstractDict)

Remove all entries from an AbstractDict, leaving it empty
"""
function clear_Dict!(d::AbstractDict)
    # This is one way to clear all entries from a dict, by using a filter which is false
    # for every entry
    if !isempty(d)
        filter!(x->false, d)
    end

    return d
end

"""
    convert_to_OrderedDicts!(d::AbstractDict)

Recursively convert an AbstractDict to OrderedDict.

Any nested AbstractDicts are also converted to OrderedDict.
"""
function convert_to_OrderedDicts!(d::AbstractDict)
    for (k, v) ∈ d
        if isa(v, AbstractDict)
            d[k] = convert_to_OrderedDicts!(v)
        end
    end
    return OrderedDict(d)
end

"""
    println_to_stdout_and_file(io, stuff...)

Print `stuff` both to stdout and to a file `io`.
"""
function println_to_stdout_and_file(io, stuff...)
    println(stuff...)
    if io !== nothing
        println(io, stuff...)
    end
end

"""
    positive_or_nan(x; epsilon=0)

If the argument `x` is zero or negative, replace it with NaN, otherwise return `x`.

`epsilon` can be passed if the number should be forced to be above some value (typically
we would assume epsilon is small and positive, but nothing about this function forces it
to be).
"""
function positive_or_nan(x; epsilon=0)
    return x > epsilon ? x : NaN
end

# Define partial_fit_exponential_growth() here rather than in moment_kinetics.analysis so
# that the moment_kinetics package does not have to have `Optim` as a dependency. Only
# `makie_post_processing` needs `Optim` as a dependency, so in projects set up to do only
# simulations and not post-processing, `Optim` is not included.
using Optim
"""
    partial_fit_exponential_growth(time, amplitude)

When in a simulation for a linear instability, typically there will be some initial
transient phase, followed by an exponential growth phase, and finally (possibly) some
nonlinear phase (saturation, crash, etc.). When identifying the growth rate, we only want
to look at the exponential growth phase, which requires identifying some time interval
within which we would fit a linear function to the log of `amplitude`. This function
attempts to automatically identify an appropriate interval by simultaneously minimising
the error on the fit and maximising the size of the interval.

`time` gives the time points corresponding to the values in `amplitude`.

Fits a function
```
amplitude_fit = A * exp(γ * time)
```
to `amplitude` in an interval `tmin` to `tmax`, optimizing `tmin` and `tmax` to get the
best fit while maximising `(tmax - tmin)`. Uses some arbitrary but hopefully sensible
weighting to balance the two.

Returns `(γ, A, tmin, tmax)`
"""
function partial_fit_exponential_growth(time, amplitude)
    if !issorted(time)
        error("`time` must be monotonically increasing")
    end
    if size(time) != size(amplitude)
        error("size(time)=$(size(time)) is not the same as "
              * "size(amplitude)=$(size(amplitude)).")
    end
    if length(time) < 2
        error("Cannot fit to a single point")
    end

    log_amplitude = log.(amplitude)

    # Find sizes of the 'cells' in time. Not really well defined if the grid spacing
    # changes, but hopefully good enough for this kind of fitting.
    face_positions = [time[1] - 0.5*(time[2] - time[1])]
    face_positions = vcat(face_positions, @views @. 0.5 * (time[1:end-1] + time[2:end]))
    push!(face_positions, time[end] + 0.5 * (time[end] - time[end-1]))
    cell_sizes = face_positions[2:end] .- face_positions[1:end-1]

    function linear_fit(x, m, c)
        # Fit y = m*x + c
        return @. m*x + c
    end

    function linear_fit_error(m, c, tmin, tmax)
        if tmin > tmax
            # If this happens, the interval is too small and should be penalised by the
            # inveral-size cost function.
            return 0.0
        end
        imin = min(searchsortedfirst(time, tmin), length(time) - 1) # Limit max value to
                                                                    # avoid bounds error
                                                                    # below.
        imax = searchsortedfirst(time, tmax) # Note that imax may be length(time)+1 if tmax > time[end].

        # Start by calculating the fit error for points that are not at the ends of the
        # interval. Weight each point by the size of the associated grid cell.
        fit_values = linear_fit(time, m, c)

        if imin < imax - 2
            r = imin+1:imax-2
            fit_error = sum(@. (log_amplitude[r] - fit_values[r])^2 * cell_sizes[r])
        else
            fit_error = 0.0
        end

        # Add contributions from first and last points in the interval.
        fit_error += (log_amplitude[imin] - fit_values[imin])^2 * (face_positions[imin+1] - tmin)
        fit_error += (log_amplitude[imax-1] - fit_values[imax-1])^2 * (tmax - face_positions[imax-1])

        # Make continuously-varying versions of imax and imin, to avoid jumps in this cost
        # function.
        i = min(imax, length(time)) # For this index, limit max value to avoid bounds
                                    # errors in the next line.
        imax_continuous = i + (tmax - time[i-1]) / (time[i] - time[i-1])
        i = max(imin, 2) # For this index, limit min value to avoid bounds errors in the
                         # next line.
        imin_continuous = i + (tmin - time[i-1]) / (time[i] - time[i-1])

        return fit_error / (imax_continuous - imin_continuous + 1)
    end

    # First find a linear fit to the full time series. This will give us an error value
    # that we can use to normalise the cost functions so that the fitting cost and the
    # interval-size cost are similar in magnitude.
    full_interval_linear_fit_cost(p) = linear_fit_error(p[1], p[2], time[1], time[end])
    initial_fit = optimize(full_interval_linear_fit_cost, [0.0, 0.0])
    initial_error = Optim.minimum(initial_fit)
    initial_m, initial_c = Optim.minimizer(initial_fit)

    function cost_function(p)
        # Evaluate a 'cost' that balances minimising the error on the fit with maximising
        # the size of the interval.
        m, c, tmin, tmax = p

        if tmin > tmax
            return Inf
        end

        linear_fit_cost = linear_fit_error(m, c, tmin, tmax) / initial_error

        # Interval size penalty - needs to be large for small intervals.
        # Multiply by 0.1 as this seems to give a reasonable balance with the normalised
        # cost of the linear fit error (possibly this prefactor should be a settable
        # parameter).
        interval_cost = 0.1 * (time[end] - time[1]) / (tmax - tmin)

        # Don't want `tmin` or `tmax` to be outside the bounds of `time`, so add penalty
        # to ensure this does not happen. Without this there is some chance that if the
        # fit error was exactly zero on and end-point, the `interval_cost` would push tmin
        # or tmax off towards infinity.
        # Normalise using the total interval length so that this cost does not depend on
        # the time-resolution of the input. Multiply by 100 so this is a 'large' cost.
        bounds_cost = 0.0
        total_t = time[end] - time[1]
        if tmin < time[1]
            bounds_cost += 100.0 * (time[1] - tmin) / total_t
        end
        if tmax > time[end]
            bounds_cost += 100.0 * (tmax - time[end]) / total_t
        end

        cost = linear_fit_cost + interval_cost + bounds_cost

        return cost
    end

    # Find the best combined linear fit and interval.
    fit = optimize(cost_function, [initial_m, initial_c, time[1], time[end]], LBFGS())

    m, c, tmin, tmax = Optim.minimizer(fit)

    # `(m,c)` for the linear fit to `log(amplitude)` correspond to `(γ,log(A))` for the
    # exponential fit to `amplitude`.
    γ = m
    A = exp(c)

    return γ, A, tmin, tmax
end
