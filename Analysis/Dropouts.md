# Dropouts:
The term “dropout” refers to dropping out units (both hidden and visible) in a neural network.
At each training stage, individual nodes are either dropped out of the net with probability 1-p or kept with probability p, so that a reduced network is left; incoming and outgoing edges to a dropped-out node are also removed.

This allows, every node to be equally important. Making them sparse, since they all have a probability p with which they may not be trained. This allows the model to regularized and therefore preventing Over-Fitting.
Dropout is an approach to regularization in neural networks which helps reducing interdependent learning amongst the neurons.

## Training Phase:
Training Phase: For each hidden layer, for each training sample, for each iteration, ignore (zero out) a random fraction, p, of nodes (and corresponding activations).

## Testing Phase:
Use all activations, but reduce them by a factor p (to account for the missing activations during training).

## Results:
1. Dropout forces a neural network to learn more robust features that are useful in conjunction with many different random subsets of the other neurons.
2. Dropout roughly doubles the number of iterations required to converge. However, training time for each epoch is less.
3. With H hidden units, each of which can be dropped, we have
2^H possible models. In testing phase, the entire network is considered and each activation is reduced by a factor p.