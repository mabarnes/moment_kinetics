module array_allocation

export allocate_float, allocate_int, allocate_complex, allocate_bool

using type_definitions: mk_float, mk_int

# allocate 1d array with n entries of type Bool
function allocate_bool(n)
    return array = Array{Bool,1}(undef, n)
end

# allocate 1d array with n entries of type mk_int
function allocate_int(n)
    return array = Array{mk_int,1}(undef, n)
end
# allocate 1d array with n entries of type mk_int
function allocate_int(n,m)
    return array = Array{mk_int,1}(undef, n, m)
end

# allocate 1d array with n entries of type mk_float
function allocate_float(n)
    return array = Array{mk_float,1}(undef, n)
end
# allocate a 2d array of size n×m with entries of type mk_float
function allocate_float(n,m)
    return array = Array{mk_float,2}(undef, n, m)
end
# allocate a 3d array of size n×m with entries of type mk_float
function allocate_float(n,m,p)
    return array = Array{mk_float,3}(undef, n, m, p)
end
# allocate a 4d array of size n×mxpxq with entries of type mk_float
function allocate_float(n,m,p,q)
    return array = Array{mk_float,4}(undef, n, m, p, q)
end
# allocate 1d array with n entries of type Complex{mk_float}
function allocate_complex(n)
    return array = Array{Complex{mk_float},1}(undef, n)
end

# allocate a 2d array of size n×m with entries of type Complex{mk_float}
function allocate_complex(n,m)
    return array = Array{Complex{mk_float},2}(undef, n, m)
end

end
