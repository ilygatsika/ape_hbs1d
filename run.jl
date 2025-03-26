# Run all the simulations
# -----------------------

# Run simulations from Section 4.1.1.
include("examples/model.jl")

# Run simulations from Section 4.1.3.
include("examples/spectral_basis.jl")
include("examples/main.jl")

# Run simulations from Section 4.1.4.
include("examples/sensitivity.jl")

# Run simulations from Section 4.1.5.
newARGS = ["1.0", "1.0"] # = Z1 and Z2
include("examples/adapt.jl")
newARGS = ["1.0", "3.0"] # = Z1 and Z2
include("examples/adapt.jl")

