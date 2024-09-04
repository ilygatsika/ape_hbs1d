using PyPlot
using LaTeXStrings

# Format PyPlot
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 15
rcParams["legend.fontsize"] = "medium"
rcParams["lines.linewidth"] = 0.85
#rcParams["pdf.use14corefonts"] = true
rcParams["lines.markersize"] = 6
rcParams["lines.markerfacecolor"] = "none"
rcParams["legend.edgecolor"] = "k"
rcParams["legend.labelspacing"] = 0.01
rcParams["legend.fancybox"] = false
rcParams["mathtext.fontset"] = "cm"
rcParams["axes.labelpad"] = 1.4
rcParams["axes.linewidth"] = 0.5

