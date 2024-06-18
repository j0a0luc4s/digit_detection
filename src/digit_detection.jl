module digit_detection

using LinearAlgebra
using Statistics

include("givens.jl")
include("house.jl")
include("bidiagonalize.jl")
include("svd.jl")

include("idx.jl")

include("train.jl")
include("test.jl")

end # module digit_detection
