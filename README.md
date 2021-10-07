# Deep learning and spectral embedding for graph partitioning

Deep learning model that integrates spectral embedding to partition a graph such that the normalized cut is the smallest possible. Based on ...

## Minimize the normalized cut

Given a graph G=(V,E), the goal is to find a partition of the set of vertices V such that its normalized cut is minimized. This is known to be an NP-complete problem, hence we look for approximate solutions to it. The Fiedler vector is known to be a relaxed solution of the minimum cut problem, however it is expensive to compute. Our idea is to approximate the Fiedler vector by training a neural network and use this vector as node features for a graph (embedding module). Then this graph is fed to another neural network that is trained by minimizing the expected value of the normalized cut in an unsupervised fashion (partitioning module). The codes for training and testing are explained below.

## Training

Simply run ``training.py``. This will train the embedding and the partitioning module separately and will save the tuned weights.

## Testing




