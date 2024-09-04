import json
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['lines.linewidth'] = 0.85
mpl.rcParams["pdf.use14corefonts"] = True
mpl.rcParams['lines.markersize'] = 4.5
mpl.rcParams['lines.markerfacecolor'] = 'none'
mpl.rcParams["legend.edgecolor"] = 'k'
mpl.rcParams["legend.labelspacing"] = 0.01
mpl.rcParams["legend.fancybox"] = False
mpl.rcParams["axes.labelpad"] = 1.4
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['axes.titlesize'] = 'medium'
# For latex font
mpl.rcParams['font.size'] = 11.5
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['text.usetex'] = True
mpl.rcParams["text.latex.preamble"].join([
        r"\usepackage{amsmath}",
        ])
# Number font
mpl.rcParams['mathtext.fontset'] = 'stixsans'


with open("out/res_main.json", "r") as file:
    data = json.load(file)

nb_list = np.array(data["Nb_list"], dtype=int)
vec_ell = np.array(data["\u2113_list"], dtype=float)
n = vec_ell.shape[0]
m = nb_list.shape[0]
Herr_src = np.array(data["Herr_src"]).reshape(n,m)
Hest_src = np.array(data["Hest_src"]).reshape(n,m)
Herr_eig = np.array(data["Herr_eig"]).reshape(n,m)
Hest_eig = np.array(data["Hest_eig"]).reshape(n,m)

marker = ["s", "o", "x", "^", "*"]
color  = ["k", "g", "c", "m", "orange"]
plt.figure(figsize=(3,2.7))
plt.plot(nb_list, Herr_src[0,:], "-sk", label="exact")
for i in range(n):
    ell = 2*vec_ell[i]
    plt.plot(nb_list, np.sort(Hest_src[:,i]), 
            marker=marker[i], c=color[i],
            label="$2\ell=%s$" %ell)
plt.ylabel("approx. error")
plt.yscale("log")
plt.xlabel("$2N$ discr. basis functions")
plt.tight_layout()
plt.legend()
plt.savefig("img/src_pb.pdf")
plt.close()

plt.figure(figsize=(3,2.7))
plt.plot(nb_list, Herr_eig[0,:], "-sk", label="exact")
for i in range(n):
    ell = 2*vec_ell[i]
    plt.plot(nb_list, Hest_eig[i,:], 
            marker=marker[i], c=color[i],
            label="$2\ell=%s$" %ell)
plt.ylabel("approx. error")
plt.yscale("log")
plt.xlabel("$2N$ discr. basis functions")
plt.tight_layout()
plt.legend()
plt.savefig("img/eig_pb.pdf")
plt.close()
