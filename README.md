# Exploring of the scLUCA dataset

For building coabundance networks.

We are basing ourselves on the wonderful work by [Salcher, Sturm, Horvath et al. 2022](https://pubmed.ncbi.nlm.nih.gov/36368318/)

## Troubleshooting

Due to the long training and annealing times, jlab sometimes cannot connect.
Use this to get a console to the kernel:
`ipython console --existing /root/.local/share/jupyter/runtime/kernel-50c440a3-554d-4d98-bb90-bdda9a8923d5.json`

This could be easier in a newer version of jlab. To locate the corresponding json
you can use htop with option to not display user branches and seeing the memory it
is using.


We have installed this:
```
apt install nodejs
pip install scarches==0.5.9
pip install jupyterlab==3.2.4
alias jl='jupyter-lab --no-browser --ip=0.0.0.0 --allow-root /root/host_home'
apt install nano
apt install libgraphviz-dev
apt install graphviz
pip install graphviz
pip install networkx
```