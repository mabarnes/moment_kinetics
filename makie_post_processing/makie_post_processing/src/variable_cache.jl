const chunk_size_1d = 10000
const chunk_size_2d = 100
struct VariableCache{T1,T2,T3}
    run_info::T1
    variable_name::String
    t_chunk_size::mk_int
    n_tinds::mk_int
    tinds_range_global::Union{UnitRange{mk_int},StepRange{mk_int}}
    tinds_chunk::Union{Base.RefValue{UnitRange{mk_int}},Base.RefValue{StepRange{mk_int}}}
    data_chunk::T2
    dim_slices::T3
end

function VariableCache(run_info, variable_name::String, t_chunk_size::mk_int;
                       it::Union{Nothing,AbstractRange}, is, iz, ir, ivperp, ivpa, ivzeta,
                       ivr, ivz)
    if it === nothing
        tinds_range_global = run_info.itime_min:run_info.itime_skip:run_info.itime_max
    else
        tinds_range_global = it
    end
    n_tinds = length(tinds_range_global)

    t_chunk_size = min(t_chunk_size, n_tinds)
    tinds_chunk = 1:t_chunk_size
    dim_slices = (is=is, iz=iz, ir=ir, ivperp=ivperp, ivpa=ivpa, ivzeta=ivzeta, ivr=ivr,
                  ivz=ivz)
    data_chunk = get_variable(run_info, variable_name; it=tinds_range_global[tinds_chunk],
                              dim_slices...)

    return VariableCache(run_info, variable_name, t_chunk_size,
                         n_tinds, tinds_range_global, Ref(tinds_chunk),
                         data_chunk, dim_slices)
end

function get_cache_slice(variable_cache::VariableCache, tind)
    tinds_chunk = variable_cache.tinds_chunk[]
    local_tind = findfirst(i->i==tind, tinds_chunk)

    if local_tind === nothing
        if tind > variable_cache.n_tinds
            error("tind=$tind is bigger than the number of time indices "
                  * "($(variable_cache.n_tinds))")
        end
        # tind is not in the cache, so get a new chunk
        chunk_size = variable_cache.t_chunk_size
        new_chunk_start = ((tind-1) ÷ chunk_size) * chunk_size + 1
        new_chunk = new_chunk_start:min(new_chunk_start + chunk_size - 1, variable_cache.n_tinds)
        variable_cache.tinds_chunk[] = new_chunk
        selectdim(variable_cache.data_chunk,
                  ndims(variable_cache.data_chunk), 1:length(new_chunk)) .=
            get_variable(variable_cache.run_info, variable_cache.variable_name;
                         it=variable_cache.tinds_range_global[new_chunk],
                         variable_cache.dim_slices...)
        local_tind = findfirst(i->i==tind, new_chunk)
    end

    return selectdim(variable_cache.data_chunk, ndims(variable_cache.data_chunk),
                     local_tind)
end

function variable_cache_extrema(variable_cache::VariableCache; transform=identity)
    # Bit of a hack to iterate through all chunks that can be in the cache
    chunk_size = variable_cache.t_chunk_size
    data_min = data_max = NaN
    for it ∈ ((i - 1) * chunk_size + 1 for i ∈ 1:(variable_cache.n_tinds ÷ chunk_size))
        get_cache_slice(variable_cache, it)
        this_min, this_max = NaNMath.extrema(transform.(variable_cache.data_chunk))
        data_min = NaNMath.min(data_min, this_min)
        data_max = NaNMath.max(data_max, this_max)
    end

    return data_min, data_max
end
