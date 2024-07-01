module digit_detection

using LinearAlgebra
using Statistics
using Distributions
using Plots

include("givens.jl")
include("house.jl")
include("bidiagonalize.jl")
include("svd.jl")

include("idx.jl")

include("train.jl")
include("test.jl")
include("showcase.jl")

end # module digit_detection
