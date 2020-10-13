module array_allocation

export allocate_float, allocate_int

# allocate 1d array with n entries of type Int64
function allocate_int(n)
    return array = Array{Int64,1}(undef, n)
end

# allocate 1d array with n entries of type Float64
function allocate_float(n)
    return array = Array{Float64,1}(undef, n)
end

# allocate a 2d array of size n×m with entries of type Float64
function allocate_float(n,m)
    return array = Array{Float64,2}(undef, n, m)
end

# allocate a 3d array of size n×m with entries of type Float64
function allocate_float(n,m,p)
    return array = Array{Float64,3}(undef, n, m, p)
end

# allocate 1d array with n entries of type Complex{Float64}
function allocate_complex(n)
    return array = Array{Complex{Float64},1}(undef, n)
end

# allocate a 2d array of size n×m with entries of type Complex{Float64}
function allocate_complex(n,m)
    return array = Array{Complex{Float64},2}(undef, n, m)
end

end
