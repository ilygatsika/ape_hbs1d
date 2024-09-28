using JSON3
using ape_hbs1d # this is our code
using PyPlot

include("src/utils.jl")

"""
.. set some directories
"""
output_dir = "out"
figure_dir = "img"
(!isdir(output_dir)) && (mkdir(output_dir))
(!isdir(figure_dir)) && (mkdir(figure_dir))

"""
.. then configure plot environment
   user can customize plots to apply globally
"""

# science is included in plt.style.available
# in Python 3.10.12
# but other versions might need:
# science = pyimport("scienceplots")

PyPlot.matplotlib."pyplot".style.use("science")
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["legend.labelspacing"] = 0.001
#rcParams["lines.markersize"] = 3.5


