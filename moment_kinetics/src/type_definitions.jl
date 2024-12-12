"""
"""
module type_definitions

export mk_float
export mk_int
export OptionsDict

using OrderedCollections: OrderedDict

"""
"""
const mk_float = Float64

"""
"""
const mk_int = Int64

"""
"""
const OptionsDict = OrderedDict{String,Any}

end
