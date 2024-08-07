export dict_merge_test

dict_base = Dict("a" => 1,
                  "bdict" => Dict("a" => 2, "f" =>3),
                  "cdict" => Dict("a" => 2, "f" =>3))

function dict_merge_test(dict_base=dict_base; args...)
    println("before merge: ",dict_base)
    for (k,v) in args
        println(k, " ", v)
        if String(k) in keys(dict_base)
            if isa(v,AbstractDict)
                v = merge(dict_base[String(k)],v)
            end            
        end
        dict_mod = Dict(String(k) => v)
        dict_base = merge(dict_base, dict_mod)
    end
    println("after merge: ",dict_base)
end


if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")

    dict_merge_test(a=4,bdict=Dict("g"=>6.0,"h" => 8),zed=74,xx=Dict("sz" => 5))
end