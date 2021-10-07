# Deep learning and spectral embedding for graph partitioning

Deep learning model that integrates spectral embedding to partition a graph such that the normalized cut is the smallest possible. Based on ...

## Minimize the normalized cut

Given a graph G=(V,E), the goal is to find a partition of the set of vertices V such that its normalized cut is minimized. This is known to be an NP-complete problem, hence we look for approximate solutions to it. The Fiedler vector is known to be a relaxed solution of the minimum cut problem, however it is expensive to compute. Our idea is to approximate the Fiedler vector by training a neural network and use this vector as node features for a graph (embedding module). Then this graph is fed to another neural network that is trained by minimizing the expected value of the normalized cut in an unsupervised fashion (partitioning module). The codes for training and testing are explained below.

## Training

Simply run ``training.py``. This will train the embedding and the partitioning modules separately and will save the tuned weights.

## Testing

Run ``test.py`` with the following arguments
- ``nmin``: minimum graph size (default: 50)
- ``nmax``: maximum graph size (default: 100)
- ``ntest``: number of testing graphs (default: 1000)
- ``dataset``: dataset type to choose among ``'delaunay'``, ``'suitesparse'``, and the Finite Elements triangulations ``graded_l``, ``hole3``, ``hole6`` (default: ``'delaunay'``). With the first choice, random Delaunay graphs in the unit square and in the rectangle $[0,2]x[0,1]$ are generated before the evaluation. With the second choice, the user needs to download the matrices from the [SuiteSparse matrix collection](https://sparse.tamu.edu/) in the Matrix Market format and put the ``.mtx`` files in the folder ``dl-graph-partitioning/suitesparse``. In the paper we focus on matrices coming from 2D/3D discretizations. For the Finite Elements triangulations, the user can download the matrices from [here](https://portal.nersc.gov/project/sparse/strumpack/fe_triangulations.tar.xz) and put the 3 folders in ``drl-graph-partitioning/``.

