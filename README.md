# This repository

Provides simple NEAT (Neuro-Evolution of Augmenting Topologies) algorithm
implementation.

# NEAT

Is a non-gradient algorithm of Neural Network training introduced by
[Kenneth O. Stanley](http://www.cs.ucf.edu/~kstanley/neat.html) in
[Stanley, Kenneth O., and Risto Miikkulainen. "Evolving neural networks through augmenting topologies." *Evolutionary computation* 10.2 (2002): 99-127.](http://www.mitpressjournals.org/doi/abs/10.1162/106365602320169811#.WDU_debhCHs).

# How to set up?

For now setup has not been prepared well. Following steps have been tested:

## Create conda environment the following way:

`conda create -n neat graphviz numpy networkx matplotlib ipython jupyter param scipy`

## Install package out of conda:

`pip install tqdm`

## For linux-64 (tested on Ubuntu 14.10):

`conda install --channel https://conda.anaconda.org/achennu pygraphviz`

## For win-64 (tested on Win10):

```
conda install pydot-ng
pip install pygraphviz-1.3.1-cp27-none-win_amd64.whl
```

## For win-64 (tested on Win10):

Copy into environment an `etc` archive with `bat`-files setting up `graphviz`
location to path. This may need adjustment of paths in `.bat` files.
